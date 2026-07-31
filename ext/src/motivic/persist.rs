//! Disk cache for the motivic lift, routed through the resolution's ZarrV3 save
//! store (PR #260). The motivic weights and the lifted $A_C$ differentials are
//! stored per bidegree under the `motivic` subgroup of the resolution's own
//! store, reusing the [`SaveKind::Differential`] shard kind (a subgroup owns its
//! own shard arrays, so reusing an existing kind inside it is collision-free).
//!
//! This is a **pure memoization cache**: every stored quantity is a deterministic
//! function of `(module, box)`, so a miss, a mismatch, or corruption is always
//! just "recompute", never a failure. And a *converged* per-bidegree cell depends
//! only on the resolution up to its own internal degree — not on the box — so a
//! cell computed for one box is byte-identical for any larger box, which is what
//! makes lazy box growth valid (Phase 2).

use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use sseq::coordinates::Bidegree;

use super::{Gen, MotivicResolution};
use crate::{
    chain_complex::ChainComplex,
    save::{SaveKind, ZarrSaveStore},
};

/// One generator's motivic data at a bidegree: its weight, and — if it was lifted
/// (every generator with `s ≥ 1`) — the $\mathbb{F}_2$ support of its lifted
/// $A_C$ differential. `lifted` is `None` only for the `s = 0` unit, which has a
/// weight but no differential to lift.
#[derive(Serialize, Deserialize)]
struct GenRecord {
    idx: u64,
    weight: i32,
    lifted: Option<Vec<u64>>,
}

/// All motivic data at one bidegree `(s, t)` — the payload of a single shard
/// element keyed by `Bidegree::n_s(t - s, s)`.
#[derive(Serialize, Deserialize)]
struct BidegreeRecord {
    gens: Vec<GenRecord>,
}

/// One source generator's lifted support in a *chain-map* cache (a product `φ_a`
/// or a Massey null-homotopy `H`), stored per source bidegree. Unlike the
/// differential cache there is no weight here — weights live in the differential
/// cache and are loaded once for the whole resolution.
#[derive(Serialize, Deserialize)]
struct LiftedGenRecord {
    idx: u64,
    support: Vec<u64>,
}

/// All source generators of a chain-map lift at one bidegree — one shard element.
#[derive(Serialize, Deserialize)]
struct LiftedBidegreeRecord {
    gens: Vec<LiftedGenRecord>,
}

impl MotivicResolution {
    /// The `motivic` subgroup of the resolution's save store, if one is open. The
    /// subgroup shares the resolution's store, so it inherits #260's algebra and
    /// module-spec binding for free (the guard "is this cache for *my* problem?").
    fn motivic_store(&self) -> Option<ZarrSaveStore> {
        self.resolution
            .save_dir()
            .store()
            .and_then(|s| s.subgroup("motivic").ok())
    }

    /// Whether generator `g`'s lifted differential is safe to persist and reuse at
    /// any larger box — i.e. box-independent because it fully **converged**. Two
    /// cases: low filtration (`s < 2`), whose lifted value is always the mod-τ seed,
    /// and internal degree `t ≤ max.n()` (see [`Self::cache_t_bound`]).
    fn lift_is_box_independent(&self, g: Gen) -> bool {
        g.s < 2 || g.t <= self.cache_t_bound()
    }

    /// Internal-degree bound below which a lifted cell has fully converged and is
    /// therefore box-independent (safe to cache and reuse at any larger box).
    ///
    /// The τ-adic correction of a cell at `(s, t)` runs entirely at internal degree
    /// `t`: every round solves a quasi-inverse at degree `t`, and the lower cells it
    /// reads all lie at degree `≤ t`. So convergence needs the whole
    /// internal-degree-`t` column of the resolution to be fully resolved. That
    /// column's lowest-filtration cell sits at stem `≈ t`, which is interior only
    /// when it is strictly inside the computed box — i.e. `t ≤ compute.n() - 1`,
    /// since the compute edge (stem `compute.n()`) is kernel-only. A cell with a
    /// larger `t` is lifted but only *partially* (it reads edge data); the report
    /// never reads it, but a larger box converges it to a different value, so caching
    /// that partial and reusing it corrupts the larger box (missing products, or —
    /// one stem lower — a `d² ≠ 0` panic). The bound is on `t`, NOT stem: a
    /// high-filtration cell at a modest stem still has large `t` and reaches the edge.
    ///
    /// Note this is `compute.n()`, not the report box `max.n()`: with the default
    /// `+1` margin they coincide (`compute.n() = max.n() + 1`), but a larger
    /// `MOT_MARGIN` resolves deeper, so more products become final AND cacheable —
    /// the one real lever for extending reuse (there is no resumable partial lift;
    /// an unconverged cell is a complete computation on incomplete inputs, not a
    /// half-finished one). Verified against cold builds (grow A→B == cold B) to n=70.
    pub(super) fn cache_t_bound(&self) -> i32 {
        self.compute.n() - 1
    }

    // --- Chain-map lift caches (Phase 3): products φ_a and null-homotopies H ----
    //
    // Each `a` (or pair `a, b`) gets its own subgroup, whose `ChainMap` /
    // `ChainHomotopy` shard arrays are private to it (a subgroup owns its arrays),
    // so reusing those kinds across many products is collision-free. The cell value
    // is box-independent under the same reasoning as the differential lift — a
    // converged in-cone cell, or a low-filtration mod-τ seed, is a function of the
    // resolution alone — with the seed boundary shifted by `a` (resp. `a, b`).

    /// The `motivic/products/{a}` subgroup for the product-lift cache of `φ_a`.
    pub(super) fn product_store(&self, a: Gen) -> Option<ZarrSaveStore> {
        self.motivic_store()?
            .subgroup(&format!("products/{}_{}_{}", a.s, a.t, a.idx))
            .ok()
    }

    /// The `motivic/homotopies/{a}__{b}` subgroup for the null-homotopy cache of the
    /// nullhomotopy of `φ_b ∘ φ_a`.
    pub(super) fn homotopy_store(&self, a: Gen, b: Gen) -> Option<ZarrSaveStore> {
        self.motivic_store()?
            .subgroup(&format!(
                "homotopies/{}_{}_{}__{}_{}_{}",
                a.s, a.t, a.idx, b.s, b.t, b.idx
            ))
            .ok()
    }

    /// Load a cached chain-map lift (`φ_a` or `H`) from `store`: every source
    /// generator whose bidegree has a record, keyed by generator. Source degrees
    /// run `s ∈ [s_lo, s_hi]`, `t ∈ [t_lo, compute.n() + s]` — the range the lift
    /// itself walks. Absent or corrupt records are skipped (that bidegree becomes
    /// frontier). Bumps [`super::PRODUCT_CELLS_REUSED`] by the number of cells read.
    pub(super) fn load_lifted_map(
        &self,
        store: &ZarrSaveStore,
        kind: SaveKind,
        s_lo: i32,
        s_hi: i32,
        t_lo: i32,
    ) -> HashMap<Gen, BTreeSet<usize>> {
        let mut map: HashMap<Gen, BTreeSet<usize>> = HashMap::new();
        let mut reused = 0u64;
        for s in s_lo..=s_hi {
            for t in t_lo..=(self.compute.n() + s) {
                if self.num_gens(s, t) == 0 {
                    continue;
                }
                let b = Bidegree::n_s(t - s, s);
                let Ok(Some(bytes)) = store.read(kind, b) else {
                    continue;
                };
                let Ok(rec) = bitcode::deserialize::<LiftedBidegreeRecord>(&bytes) else {
                    continue;
                };
                for gr in rec.gens {
                    let g = Gen {
                        s,
                        t,
                        idx: gr.idx as usize,
                    };
                    map.insert(g, gr.support.into_iter().map(|x| x as usize).collect());
                    reused += 1;
                }
            }
        }
        super::PRODUCT_CELLS_REUSED.fetch_add(reused, std::sync::atomic::Ordering::Relaxed);
        map
    }

    /// Save a chain-map lift (`φ_a` or `H`) to `store`, one shard element per source
    /// bidegree. Only box-independent cells — those for which `keep(g)` holds — are
    /// written; the out-of-cone seed placeholders (which a larger box would replace)
    /// are dropped, mirroring the differential cache. Writes are idempotent.
    pub(super) fn save_lifted_map(
        &self,
        store: &ZarrSaveStore,
        kind: SaveKind,
        map: &HashMap<Gen, BTreeSet<usize>>,
        keep: impl Fn(Gen) -> bool,
    ) {
        let mut by_bidegree: HashMap<(i32, i32), Vec<LiftedGenRecord>> = HashMap::new();
        for (g, support) in map {
            if !keep(*g) {
                continue;
            }
            by_bidegree.entry((g.s, g.t)).or_default().push(LiftedGenRecord {
                idx: g.idx as u64,
                support: support.iter().map(|&x| x as u64).collect(),
            });
        }
        for ((s, t), mut gens) in by_bidegree {
            gens.sort_by_key(|r| r.idx);
            let b = Bidegree::n_s(t - s, s);
            let bytes = bitcode::serialize(&LiftedBidegreeRecord { gens }).unwrap();
            let _ = store.write(kind, b, &bytes);
        }
    }

    /// Save the weights and lifted differentials to the `motivic` subgroup, one
    /// shard element per bidegree. No-op if no store is open.
    ///
    /// Weights are persisted for every generator (always box-independent); the
    /// lifted support only for box-independent cells (see
    /// [`Self::lift_is_box_independent`]). No box tag is written — a later, larger
    /// box reads these cells back and reuses them verbatim, recomputing only its
    /// frontier. Writes are idempotent (same bytes for an unchanged cell), so a
    /// warm re-save is harmless.
    pub(super) fn save_lift(&self) {
        let Some(store) = self.motivic_store() else {
            return;
        };
        // Group every generator's (weight, optional lifted support) by bidegree.
        let mut by_bidegree: HashMap<(i32, i32), Vec<GenRecord>> = HashMap::new();
        for (g, &weight) in self.weights.iter() {
            let lifted = if self.lift_is_box_independent(*g) {
                self.lifted
                    .get(g)
                    .map(|sup| sup.iter().map(|&b| b as u64).collect())
            } else {
                None
            };
            by_bidegree.entry((g.s, g.t)).or_default().push(GenRecord {
                idx: g.idx as u64,
                weight,
                lifted,
            });
        }
        for ((s, t), mut gens) in by_bidegree {
            // Sort by idx so the serialized shard bytes are deterministic (golden-stable).
            gens.sort_by_key(|r| r.idx);
            let b = Bidegree::n_s(t - s, s);
            let bytes = bitcode::serialize(&BidegreeRecord { gens }).unwrap();
            let _ = store.write(SaveKind::Differential, b, &bytes);
        }
    }

    /// Populate the weights and lifted differentials with whatever the `motivic`
    /// subgroup holds — **incremental**, not all-or-nothing. Whatever cells are on
    /// disk are restored; the caller's `compute_weights`/`lift` then fill only the
    /// gaps (the frontier). A larger box therefore reuses a smaller box's cached
    /// cells directly (lazy growth); an empty or absent store leaves both maps
    /// empty and everything recomputes. A corrupt record for a bidegree is skipped,
    /// so that bidegree recomputes — a miss is never a failure.
    ///
    /// Increments [`super::LIFT_CACHE_LOADS`] once if any lifted cell was restored,
    /// so a run can tell a genuine (partial) disk reuse from a cold recompute.
    pub(super) fn load_lift(&mut self) {
        let Some(store) = self.motivic_store() else {
            return;
        };
        let mut weights: HashMap<Gen, i32> = HashMap::new();
        let mut lifted: HashMap<Gen, BTreeSet<usize>> = HashMap::new();
        for s in 0..=self.max_s() {
            let t_max = self.compute.n() + s;
            for t in 0..=t_max {
                if self.num_gens(s, t) == 0 {
                    continue;
                }
                let b = Bidegree::n_s(t - s, s);
                let Ok(Some(bytes)) = store.read(SaveKind::Differential, b) else {
                    continue; // absent ⇒ this bidegree is frontier, recompute it
                };
                let Ok(rec) = bitcode::deserialize::<BidegreeRecord>(&bytes) else {
                    continue; // corrupt ⇒ recompute this bidegree
                };
                for gr in rec.gens {
                    let g = Gen {
                        s,
                        t,
                        idx: gr.idx as usize,
                    };
                    weights.insert(g, gr.weight);
                    if let Some(sup) = gr.lifted {
                        lifted.insert(g, sup.into_iter().map(|b| b as usize).collect());
                    }
                }
            }
        }
        if !lifted.is_empty() {
            super::LIFT_CACHE_LOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.weights = Arc::new(weights);
        self.lifted = lifted;
    }
}
