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
    /// any larger box — i.e. box-independent. Two cases are: low filtration
    /// (`s < 2`), whose lifted value is always the mod-τ seed, and inside the report
    /// cone (`stem ≤ max.n() + max.s()`), whose lift converges. An out-of-cone
    /// `s ≥ 2` cell holds only a mod-τ seed *placeholder* that a larger box would
    /// replace with a real lift, so it must never be cached (the one persistence
    /// hazard the plan calls out). All generators at a bidegree share this predicate.
    fn lift_is_box_independent(&self, g: Gen) -> bool {
        g.s < 2 || (g.t - g.s) <= self.max.n() + self.max.s()
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
