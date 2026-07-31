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

    /// Save the weights and lifted differentials to the `motivic` subgroup, one
    /// shard element per bidegree. No-op if no store is open. Cells are
    /// box-independent, so no box tag is written; a later, larger box reuses these
    /// cells verbatim.
    pub(super) fn save_lift(&self) {
        let Some(store) = self.motivic_store() else {
            return;
        };
        // Group every generator's (weight, optional lifted support) by bidegree.
        let mut by_bidegree: HashMap<(i32, i32), Vec<GenRecord>> = HashMap::new();
        for (g, &weight) in self.weights.iter() {
            by_bidegree
                .entry((g.s, g.t))
                .or_default()
                .push(GenRecord {
                    idx: g.idx as u64,
                    weight,
                    lifted: self
                        .lifted
                        .get(g)
                        .map(|sup| sup.iter().map(|&b| b as u64).collect()),
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

    /// Load the weights and lifted differentials from the `motivic` subgroup,
    /// returning whether a **complete** cache for this box was found. Complete
    /// means every non-empty bidegree in the box has a record whose generator
    /// count matches the resolution — a partial or stale cache reads as a miss and
    /// the caller recomputes from scratch. (Phase 2 turns this into a per-cell
    /// incremental skip; Phase 1 keeps the all-or-nothing gate.)
    pub(super) fn load_lift(&mut self) -> bool {
        let Some(store) = self.motivic_store() else {
            return false;
        };
        let mut weights: HashMap<Gen, i32> = HashMap::new();
        let mut lifted: HashMap<Gen, BTreeSet<usize>> = HashMap::new();
        for s in 0..=self.max_s() {
            let t_max = self.compute.n() + s;
            for t in 0..=t_max {
                let n_gens = self.num_gens(s, t);
                if n_gens == 0 {
                    continue;
                }
                let b = Bidegree::n_s(t - s, s);
                let Ok(Some(bytes)) = store.read(SaveKind::Differential, b) else {
                    return false; // a required cell is absent ⇒ not a complete cache
                };
                let Ok(rec) = bitcode::deserialize::<BidegreeRecord>(&bytes) else {
                    return false; // corrupt ⇒ recompute
                };
                if rec.gens.len() != n_gens {
                    return false; // stale/partial ⇒ recompute
                }
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
        self.weights = Arc::new(weights);
        self.lifted = lifted;
        super::LIFT_CACHE_LOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }
}
