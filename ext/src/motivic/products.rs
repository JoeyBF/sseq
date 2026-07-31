//! Motivic Ext products over $\mathbb{F}_2[\tau]$: lift the left-multiplication
//! chain map $\varphi_a$ to $A_C$ and read the product $a \cdot b$ off it. The
//! $\tau$-powers of the result are the hidden extensions (e.g. $h_0^2 h_2 = \tau
//! h_1^3$). The lift is the [`TauLift`] driver applied to the product chain map
//! ([`ProductCells`]).

use std::{
    collections::{BTreeSet, HashMap},
    sync::{Arc, atomic::Ordering},
};

use algebra::{
    CTauAlgebra,
    module::{FreeModule, Module, homomorphism::ModuleHomomorphism},
};
use fp::{prime::TWO, vector::FpVector};
use maybe_rayon::prelude::*;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

use super::{Gen, MotivicResolution, TauLift};
use crate::{chain_complex::ChainComplex, save::SaveKind};

impl MotivicResolution {
    /// Lift the product chain map $\varphi_a$ (left-multiplication by the generator
    /// `a`) to $A_C$ over $\mathbb{F}_2[\tau]$: `g ↦ φₐ(g)` supports keyed by source
    /// generator, computed up to source homological degree `max_s`.
    ///
    /// The mod-τ seeds are the ExtAlgebra's [`ResolutionHomomorphism`] (the Cτ
    /// product); the τ-corrections make it an honest chain map over $A_C$
    /// (`dφₐ = φₐd`), via the [`TauLift`] driver ([`ProductCells`]). Processed with
    /// `s` ascending, since a cell's constant defect `φₐ(dg)` reads `φₐ` at `s-1`.
    /// This is the motivic product: reducing mod τ recovers the Cτ product, and the
    /// τ-powers are the hidden extensions (e.g. `h₀²h₂ = τ·h₁³`).
    #[tracing::instrument(skip(self, a), fields(a = %a, num_corrections = tracing::field::Empty, cells_reused = tracing::field::Empty))]
    pub(super) fn lift_product(&self, a: Gen, max_s: i32) -> HashMap<Gen, BTreeSet<usize>> {
        let iters_before = super::TAULIFT_ITERS.load(Ordering::Relaxed);
        let reused_before = super::PRODUCT_CELLS_REUSED.load(Ordering::Relaxed);
        let wa = self.weights[&a];

        // Restore cached box-independent φₐ cells (Phase 3). Cells run over source
        // degrees s ∈ [a.s, max_s], t ∈ [a.t, compute.n() + s].
        let store = self.product_store(a);
        let mut phi: HashMap<Gen, BTreeSet<usize>> = store
            .as_ref()
            .map(|st| self.load_lifted_map(st, SaveKind::ChainMap, a.s, max_s, a.t))
            .unwrap_or_default();
        let reused = super::PRODUCT_CELLS_REUSED.load(Ordering::Relaxed) - reused_before;
        tracing::Span::current().record("cells_reused", reused);

        // The frontier: source generators not already loaded. If empty, the cached
        // φₐ is complete for this box — skip the ExtAlgebra product map and the whole
        // τ-correction (the expensive part).
        let missing: Vec<Gen> = (a.s..=max_s)
            .flat_map(|s| {
                (a.t..=(self.compute.n() + s))
                    .flat_map(move |t| (0..self.num_gens(s, t)).map(move |idx| Gen { s, t, idx }))
            })
            .filter(|g| !phi.contains_key(g))
            .collect();

        if !missing.is_empty() {
            // Mod-τ seeds: φₐ(g) mod τ = the ExtAlgebra chain map on each generator.
            // φₐ shifts degree by aₜ, so it is zero on source generators below aₜ.
            let a_deg = Bidegree::n_s(a.t - a.s, a.s);
            let hom = self
                .ext()
                .generator_product_map(BidegreeGenerator::new(a_deg, a.idx));
            hom.extend_all();
            let mut seeds: HashMap<Gen, BTreeSet<usize>> = HashMap::new();
            for s in a.s..=max_s {
                let map = hom.get_map(s);
                for t in a.t..=(self.compute.n() + s) {
                    for idx in 0..self.num_gens(s, t) {
                        let support: BTreeSet<usize> = map
                            .output(t, idx)
                            .iter_nonzero()
                            .filter(|(_, v)| *v != 0)
                            .map(|(i, _)| i)
                            .collect();
                        if !support.is_empty() {
                            seeds.insert(Gen { s, t, idx }, support);
                        }
                    }
                }
            }

            // Lift the frontier, s ascending (a cell's constant defect reads φₐ at
            // s-1, already present — loaded or computed at the lower s).
            for s in a.s..=max_s {
                let gens: Vec<Gen> = (a.t..=(self.compute.n() + s))
                    .flat_map(|t| (0..self.num_gens(s, t)).map(move |idx| Gen { s, t, idx }))
                    .filter(|g| !phi.contains_key(g))
                    .collect();
                let lifted: Vec<(Gen, BTreeSet<usize>)> = gens
                    .into_maybe_par_iter()
                    .map(|g| {
                        let cells = ProductCells {
                            res: self,
                            a,
                            wa,
                            seeds: &seeds,
                            phi: &phi,
                        };
                        (g, cells.lift_or_seed(g))
                    })
                    .collect();
                phi.extend(lifted);
            }

            // Persist the box-independent cells for reuse at this or a larger box.
            if let Some(ref st) = store {
                self.save_lifted_map(st, SaveKind::ChainMap, &phi, |g| {
                    self.product_lift_is_box_independent(a, g)
                });
            }
        }
        tracing::Span::current().record(
            "num_corrections",
            super::TAULIFT_ITERS.load(Ordering::Relaxed) - iters_before,
        );
        phi
    }

    /// Whether `φₐ(g)` is box-independent — safe to cache and reuse at any larger
    /// box because it fully converged. A cell is a plain mod-τ seed when `g.s ≤ a.s`
    /// (out_s < 1), otherwise a lift that converges only inside the **report box**
    /// (`stem ≤ max.n()`) — the same convergence bound as the differential lift (see
    /// `lift_is_box_independent`). Margin cells `(max.n(), cone]` are partial lifts a
    /// larger box would replace, so they are never persisted.
    pub(super) fn product_lift_is_box_independent(&self, a: Gen, g: Gen) -> bool {
        g.s - a.s < 1 || g.t <= self.cache_t_bound()
    }

    /// The motivic product `a · b` of two resolution generators, over
    /// $\mathbb{F}_2[\tau]$: a list of `(target generator, τ-power)` at bidegree
    /// `(aₛ+bₛ, aₜ+bₜ)`. The τ-powers are the hidden extensions — 0 for a product
    /// visible mod τ (the Cτ product), positive where the Cτ product vanishes but
    /// the motivic product does not (e.g. `h₀·(h₀h₂) = τ·h₁³`).
    ///
    /// Read off the lifted chain map [`Self::lift_product`]: `a·b` is the cocycle
    /// `gₖ ↦ ε_b(φₐ(gₖ))` — the coefficient of the augmentation term `1 ⊗ b` in
    /// `φₐ(gₖ)`. Since `φₐ(gₖ)` has weight `w(gₖ) − w(a)` and `1 ⊗ b` has weight
    /// `w(b)`, that coefficient's forced τ-power is `w(a) + w(b) − w(gₖ)`.
    /// All left products `a · b` for a fixed `a`, from a **single** lift of `φₐ`.
    ///
    /// This is the batched form of [`Self::motivic_product`]: it lifts the product chain map `φₐ`
    /// once (the expensive step) and reads off `a · b` for every source generator `b` at once,
    /// returning a map `b ↦ [(target, τ-power)]`. Prefer this when multiplying many `b` by the same
    /// `a` (e.g. every generator by a fixed `hᵢ`), where calling [`Self::motivic_product`] in a loop
    /// would re-lift `φₐ` on every call.
    ///
    /// A `b` absent from the map (or with an empty list) has `a · b = 0`.
    #[tracing::instrument(skip(self, a), fields(a = %a, num_products = tracing::field::Empty))]
    pub fn motivic_products_by(&self, a: Gen) -> HashMap<Gen, Vec<(Gen, u32)>> {
        let phi = self.lift_product(a, self.max.s());
        let wa = self.weights[&a];
        let mut out: HashMap<Gen, Vec<(Gen, u32)>> = HashMap::new();
        for (&gk, support) in &phi {
            let b_s = gk.s - a.s;
            let b_t = gk.t - a.t;
            if b_s < 0 || b_t < 0 {
                continue;
            }
            // `φₐ(gk)` lives in `F_{bₛ}` at degree `bₜ`; a support entry that is the augmentation
            // term `1 ⊗ b` (generator `b` at homological degree 0) contributes `gk` to `a · b`.
            let out_mod = self.module(b_s);
            for b_idx in 0..self.num_gens(b_s, b_t) {
                let idx_1b = out_mod.operation_generator_to_index(0, 0, b_t, b_idx);
                if support.contains(&idx_1b) {
                    let b = Gen {
                        s: b_s,
                        t: b_t,
                        idx: b_idx,
                    };
                    let wb = self.weights[&b];
                    out.entry(b)
                        .or_default()
                        .push((gk, (wa + wb - self.weights[&gk]) as u32));
                }
            }
        }
        tracing::Span::current().record("num_products", out.len());
        out
    }

    pub fn motivic_product(&self, a: Gen, b: Gen) -> Vec<(Gen, u32)> {
        let target_s = a.s + b.s;
        let target_t = a.t + b.t;
        // Out of the stem-computed box ⇒ uncomputed, reported as the zero product.
        if target_s > self.max.s() || target_t - target_s > self.compute.n() {
            return Vec::new();
        }
        let phi = self.lift_product(a, target_s);
        let out_mod = self.module(b.s); // φₐ(gₖ) ∈ F_{bₛ} at degree bₜ
        let idx_1b = out_mod.operation_generator_to_index(0, 0, b.t, b.idx);
        let wa = self.weights[&a];
        let wb = self.weights[&b];
        (0..self.num_gens(target_s, target_t))
            .filter_map(|k| {
                let gk = Gen {
                    s: target_s,
                    t: target_t,
                    idx: k,
                };
                phi.get(&gk)
                    .filter(|support| support.contains(&idx_1b))
                    .map(|_| (gk, (wa + wb - self.weights[&gk]) as u32))
            })
            .collect()
    }
}

/// The product-lift instance of [`TauLift`]: lift the left-multiplication chain
/// map $\varphi_a$ so that `dφₐ = φₐd` over `A_C`. For a source generator `g` at
/// `(s, t)`, `φₐ(g)` lives in `F_{s−aₛ}` at degree `t−aₜ`; the defect module is
/// `F_{s−aₛ−1}`, the variable part of the defect is `d(φₐ g)`, and the constant
/// part is `φₐ(dg)`.
struct ProductCells<'a> {
    res: &'a MotivicResolution,
    a: Gen,
    wa: i32,
    /// Mod-τ seeds `φₐ(g)` from the ExtAlgebra chain map.
    seeds: &'a HashMap<Gen, BTreeSet<usize>>,
    /// `φₐ` lifted at strictly lower homological degree (read by the constant defect).
    phi: &'a HashMap<Gen, BTreeSet<usize>>,
}

impl ProductCells<'_> {
    /// Correct `φₐ(g)` if the cell admits a defect and stays in the box; else return
    /// the mod-τ seed unchanged.
    fn lift_or_seed(&self, g: Gen) -> BTreeSet<usize> {
        let out_s = g.s - self.a.s;
        let stem = g.t - g.s;
        let in_cone = stem <= self.res.max.n() + self.res.max.s();
        if out_s >= 1
            && in_cone
            && self.res.weights.contains_key(&g)
            && self.res.lifted.contains_key(&g)
        {
            self.lift_cell(g)
        } else {
            self.seed(g)
        }
    }
}

impl TauLift for ProductCells<'_> {
    fn source_weight(&self, g: Gen) -> i32 {
        // φₐ(g) has weight w(g) − w(a): the chain map for left-multiplication by `a`
        // lowers weight by w(a) (its τ⁰ entries recover the Cτ product).
        self.res.weights[&g] - self.wa
    }

    fn seed(&self, g: Gen) -> BTreeSet<usize> {
        self.seeds.get(&g).cloned().unwrap_or_default()
    }

    fn defect_module(&self, g: Gen) -> (Arc<FreeModule<CTauAlgebra>>, i32) {
        (self.res.module(g.s - self.a.s - 1), g.t - self.a.t)
    }

    fn defect_weight(&self, g: Gen, bidx: usize) -> i32 {
        self.res
            .entry_weight(g.s - self.a.s - 1, g.t - self.a.t, bidx)
    }

    fn seed_constant(&self, g: Gen, parity: &mut FpVector) {
        // The support-independent part of the defect: φₐ(dg) = Σ φₐ over the lifted
        // differential of g, landing in F_{s−aₛ−1}.
        let f_sm1 = self.res.module(g.s - 1);
        let inner_out = self.res.module(g.s - self.a.s - 1);
        for &bidx in &self.res.lifted[&g] {
            // inner = φₐ shifts degree by aₜ.
            self.res.compose_into(
                &f_sm1,
                g.t,
                g.s - 1,
                self.phi,
                self.a.t,
                &inner_out,
                bidx,
                parity,
            );
        }
    }

    fn accumulate(&self, g: Gen, bidx: usize, parity: &mut FpVector) {
        // The variable part: d(φₐ g), applying the lifted differential to the current
        // φₐ(g) support (which lives in F_{s−aₛ}). inner = d is degree-preserving.
        let out_mod = self.res.module(g.s - self.a.s);
        let inner_out = self.res.module(g.s - self.a.s - 1);
        self.res.compose_into(
            &out_mod,
            g.t - self.a.t,
            g.s - self.a.s,
            &self.res.lifted,
            0,
            &inner_out,
            bidx,
            parity,
        );
    }

    fn solve(&self, g: Gen, e: &FpVector) -> Option<FpVector> {
        let out_s = g.s - self.a.s;
        let out_t = g.t - self.a.t;
        let d = self.res.resolution.differential(out_s);
        let qi = d.quasi_inverse(out_t)?;
        let mut c = FpVector::new(TWO, self.res.module(out_s).dimension(out_t));
        qi.apply(c.as_slice_mut(), 1, e.as_slice());
        Some(c)
    }
}
