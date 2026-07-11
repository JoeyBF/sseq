//! The C-motivic Adams $E_2$ by deformation: lift the $A_C/\tau$ resolution to
//! $A_C$ over $\mathbb{F}_2[\tau]$.
//!
//! Phase 1 ([`crate`]'s `resolve_motivic_ctau` example) resolves the trivial
//! module over $A_C/\tau$ with the ordinary engine. That resolution is *minimal
//! mod $\tau$*: its differentials $\bar d_s$ have entries in the augmentation
//! ideal (positive-degree operations). This module performs Phase 2 of
//! `MOTIVIC_PLAN.md`: it lifts $\bar d_s$ to honest differentials $d_s$ over
//! $A_C$ (coefficients in $\mathbb{F}_2[\tau]$) with $d_{s-1} d_s = 0$, reducing
//! to $\bar d_s$ mod $\tau$.
//!
//! # Weights and the valuation representation
//!
//! $A_C/\tau$ is bigraded by (stem $t$, motivic weight $w$), but we present it to
//! the engine graded by $t$ alone (it is connected and finite-type there). The
//! minimal resolution nonetheless comes out **weight-homogeneous**: every
//! generator has a well-defined weight, computed here by descending the
//! differential ([`MotivicResolution::generator_weight`]). Homogeneity forces the
//! $\tau$-power of every differential entry — a term $m \otimes g_j$ (operation
//! $m$, target generator $g_j$) in $d_s(g_k)$ carries exactly
//! $\tau^{\,w(g_k) - w(m) - w(g_j)}$ — so a lifted differential is an
//! $\mathbb{F}_2$ support (which basis elements occur) plus the weights, with the
//! $\tau$-powers reconstructed on demand. This is the one-integer-per-entry
//! valuation representation of the plan.
//!
//! # The lift
//!
//! Initialize $d_s = \bar d_s$ (each mod-$\tau$ operation at $\tau^0$). Over
//! $A_C$ the composite $d_{s-1} d_s$ is $\equiv 0 \bmod \tau$ (its $\tau^0$ part
//! is $\bar d_{s-1}\bar d_s = 0$) but not exactly $0$: the $A_C$ products
//! $m \cdot m'$ generate $\tau$-divisible terms. That remainder is a
//! $\bar d_{s-2}$-cycle, so the quasi-inverse of $\bar d_{s-1}$ (already computed
//! by the engine) yields a $\tau$-power correction to $d_s$ that cancels the
//! lowest-order remainder; iterating to bounded $\tau$-order gives $d_{s-1}d_s=0$.
//! (Guozhen Wang's Adams–Novikov lift, module side; see `MOTIVIC_PLAN.md` §5.)
//!
//! # The cohomology $H(\delta)$ — the motivic Adams $E_2$
//!
//! The lift creates $\delta$, the identity-operation (augmentation) part of the
//! differential ([`MotivicResolution::delta`]): a differential on the free
//! $\mathbb{F}_2[\tau]$-modules of generators at fixed internal degree $t$. The
//! motivic Adams $E_2$ is `Ext_{A_C} = H(δ)`, a graded $\mathbb{F}_2[\tau]$-module
//! — free $\oplus\ \mathbb{F}_2[\tau]/\tau^k$, since $\tau$ is the only homogeneous
//! prime. Because $\delta$ raises the weight, `{weight ≤ cap}` is a subcomplex and
//! the whole computation is pure $\mathbb{F}_2$ linear algebra. The three anchors
//! fall out:
//!
//! - **invert $\tau$** — the free rank (all generators): the classical Adams $E_2$
//!   ([`MotivicResolution::classical_ext_rank`], regressed against `Ext_A`).
//! - **$\tau = 0$** — the generator counts: the algebraic Novikov $E_2$ (Phase 1).
//! - **keep $\tau$** — free plus $\tau$-torsion: the motivic $E_2$, including the
//!   $h_1$-tower classes ($h_1^n$ for all $n$) that the classical page kills
//!   ([`MotivicResolution::has_tau_torsion`]).

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

use algebra::{
    CTauAlgebra,
    module::{FDModule, FreeModule, Module, homomorphism::ModuleHomomorphism},
};
use bivec::BiVec;
use fp::{matrix::Matrix, prime::TWO, vector::FpVector};
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

use crate::{
    chain_complex::{ChainComplex, FiniteChainComplex},
    resolution::Resolution,
};

/// The $A_C/\tau$ resolution type: the trivial module resolved by the ordinary
/// engine over the mod-$\tau$ Steenrod algebra.
pub type CTauResolution = Resolution<FiniteChainComplex<FDModule<CTauAlgebra>>>;

/// A generator of the resolution, identified by homological degree `s`, internal
/// degree `t`, and index within that `(s, t)` bidegree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Gen {
    pub s: i32,
    pub t: i32,
    pub idx: usize,
}

/// The C-motivic resolution: the mod-$\tau$ model plus the weight assignment and
/// (Phase 2) the lifted $A_C$ differentials.
pub struct MotivicResolution {
    algebra: Arc<CTauAlgebra>,
    resolution: CTauResolution,
    /// Motivic weight of each generator.
    weights: HashMap<Gen, i32>,
    /// The lifted $A_C$ differential of each generator: the set of $F_{s-1}$ basis
    /// elements in its image. The coefficient of each is $1 \in \mathbb{F}_2$ and
    /// its $\tau$-power is forced by the weights, so the support is the whole
    /// datum (see the module docs).
    lifted: HashMap<Gen, BTreeSet<usize>>,
    /// The box the results are trusted/reported in.
    max: Bidegree,
    /// The (padded) box actually resolved. Lifting a generator at stem `n` reaches
    /// data at stems up to `n + 2` and needs the quasi-inverse of the previous
    /// differential one stem out, so we resolve a margin beyond `max` and only
    /// verify/report within `max`.
    compute: Bidegree,
}

impl MotivicResolution {
    /// Resolve $A_C/\tau$ through the stem/filtration box `max` and assign weights
    /// to every generator.
    pub fn new(max: Bidegree) -> Self {
        let algebra = Arc::new(CTauAlgebra::new());
        let trivial = Arc::new(FDModule::new(
            Arc::clone(&algebra),
            "F2".to_string(),
            BiVec::from_vec(0, vec![1]),
        ));
        let cc: Arc<FiniteChainComplex<FDModule<CTauAlgebra>>> =
            Arc::new(FiniteChainComplex::ccdz(trivial));
        let resolution = Resolution::new(cc);
        // Pad the resolved box: lifting reaches a few stems out (and needs
        // quasi-inverses one stem beyond the report box).
        let compute = Bidegree::n_s(max.n() + max.s() + 3, max.s());
        let profile = std::env::var("MOT_PROFILE").is_ok();
        let t0 = std::time::Instant::now();
        resolution.compute_through_stem(compute);
        if profile {
            use std::sync::atomic::Ordering;
            use algebra::motivic::milnor::{PRODUCT_HITS, PRODUCT_MISSES, PRODUCT_NANOS};
            eprintln!("[profile] resolution: {:?}", t0.elapsed());
            eprintln!(
                "[profile]   products: {} miss + {} hit, {:?} in closed-form",
                PRODUCT_MISSES.load(Ordering::Relaxed),
                PRODUCT_HITS.load(Ordering::Relaxed),
                std::time::Duration::from_nanos(PRODUCT_NANOS.load(Ordering::Relaxed)),
            );
        }

        let mut this = Self {
            algebra,
            resolution,
            weights: HashMap::new(),
            lifted: HashMap::new(),
            max,
            compute,
        };
        let t1 = std::time::Instant::now();
        this.compute_weights();
        if profile {
            eprintln!("[profile] weights:    {:?}", t1.elapsed());
        }
        let t2 = std::time::Instant::now();
        this.lift();
        if profile {
            eprintln!("[profile] lift:       {:?}", t2.elapsed());
        }
        this
    }

    /// The mod-$\tau$ resolution (the Phase 1 model).
    pub fn resolution(&self) -> &CTauResolution {
        &self.resolution
    }

    /// The box results are reported/trusted in.
    pub fn max(&self) -> Bidegree {
        self.max
    }

    /// The algebraic Novikov $E_2$ rank at `(s, t)` — set $\tau = 0$: the number
    /// of generators (with $\delta \equiv 0 \bmod \tau$, `Ext = generators`).
    pub fn algebraic_novikov_rank(&self, s: i32, t: i32) -> usize {
        self.num_gens(s, t)
    }

    /// The maximum homological degree computed.
    fn max_s(&self) -> i32 {
        self.max.s()
    }

    /// The free module $F_s$.
    fn module(&self, s: i32) -> Arc<FreeModule<CTauAlgebra>> {
        self.resolution.module(s)
    }

    /// The motivic weight of a generator (panics if out of the computed range or
    /// if the generator's weight could not be determined).
    pub fn generator_weight(&self, g: Gen) -> i32 {
        self.weights[&g]
    }

    /// The number of generators in bidegree `(s, t)`.
    fn num_gens(&self, s: i32, t: i32) -> usize {
        self.module(s).number_of_gens_in_degree(t)
    }

    /// The weight of an $F_s$ basis element `bidx` in degree `t`: decode it into
    /// `operation ⊗ generator` and add the operation's weight to the generator's.
    fn entry_weight(&self, s: i32, t: i32, bidx: usize) -> i32 {
        let module = self.module(s);
        let og = module.index_to_op_gen(t, bidx);
        let op_w = self.algebra.weight(og.operation_degree, og.operation_index);
        let gen_w = self.weights[&Gen {
            s,
            t: og.generator_degree,
            idx: og.generator_index,
        }];
        op_w + gen_w
    }

    /// Lift every differential to $A_C$. Processed with `s` ascending so that
    /// `d_{s-1}` is finalized before `d_s` is corrected against it.
    ///
    /// `d_1` needs no correction: `d_0 d_1 = 0` already, since `d_0` is the
    /// augmentation into the trivial module and `d_1`'s entries lie in the
    /// augmentation ideal. For `s ≥ 2` we start from the mod-$\tau$ support and
    /// cancel the $\tau$-divisible remainder of `d_{s-1} d_s`.
    fn lift(&mut self) {
        // Wavefront over `s`: correcting a generator reads only its `s-1`
        // neighbors (the mod-τ targets at stem ≤ n and the δ-targets at stem
        // n+1), and runs its own weight-loop locally. So `s` is a sequential
        // barrier, but at each `s` every generator is independent — fan them out.
        for s in 1..=self.max_s() {
            let t_max = self.compute.n() + s;
            let gens: Vec<Gen> = (0..=t_max)
                .flat_map(|t| (0..self.num_gens(s, t)).map(move |idx| Gen { s, t, idx }))
                .collect();
            let lifted: Vec<(Gen, BTreeSet<usize>)> = gens
                .into_maybe_par_iter()
                .map(|g| (g, self.lift_generator(g)))
                .collect();
            self.lifted.extend(lifted);
        }
    }

    /// Lift a single generator's differential to `A_C`. Parallel-safe: it reads
    /// only already-finalized `s-1` data (and the shared, thread-safe product
    /// cache), writing nothing.
    fn lift_generator(&self, g: Gen) -> BTreeSet<usize> {
        // Start from the mod-τ differential support (τ^0 terms).
        let mut support: BTreeSet<usize> = self
            .resolution
            .differential(g.s)
            .output(g.t, g.idx)
            .iter_nonzero()
            .filter(|(_, v)| *v != 0)
            .map(|(i, _)| i)
            .collect();
        // Correct only generators whose δ-correction cone stays inside the padded
        // box. The cone of a generator at stem `n` reaches up to stem `n + s` (each
        // augmentation correction pushes one stem out), so a generator with stem
        // `> report.n + report.s` cannot converge — and it is never referenced by
        // the report cohomology (differentials go to stem ≤ n, δ-terms to n+1, and
        // the report cone is bounded by report.n + report.s). Leaving those as
        // their mod-τ support is correct.
        let stem = g.t - g.s;
        let in_cone = stem <= self.max.n() + self.max.s();
        if g.s >= 2 && in_cone && self.weights.contains_key(&g) {
            self.correct(g, &mut support);
        }
        support
    }

    /// The composite `d_{s-1} d_s(g_k)` over `A_C`, as a map from $F_{s-2}$ basis
    /// element (in degree `g_k.t`) to `(parity mod 2, forced τ-power)`. Only
    /// odd-parity (surviving) terms are returned. The `A_C` products
    /// `m · m'` are where the $\tau$-divisible terms are generated.
    fn composite(&self, g_k: Gen, support: &BTreeSet<usize>) -> BTreeMap<usize, i32> {
        let s = g_k.s;
        let t = g_k.t;
        let w_k = self.weights[&g_k];
        let f_sm1 = self.module(s - 1);
        let f_sm2 = self.module(s - 2);
        let engine = self.algebra.engine();

        let mut parity: HashMap<usize, u32> = HashMap::new();
        for &bidx in support {
            let og = f_sm1.index_to_op_gen(t, bidx);
            let (m_deg, m_idx) = (og.operation_degree, og.operation_index);
            let gj = Gen {
                s: s - 1,
                t: og.generator_degree,
                idx: og.generator_index,
            };
            for &bidx2 in &self.lifted[&gj] {
                let og2 = f_sm2.index_to_op_gen(gj.t, bidx2);
                let (mp_deg, mp_idx) = (og2.operation_degree, og2.operation_index);
                let (gl_deg, gl_idx) = (og2.generator_degree, og2.generator_index);
                let z_deg = m_deg + mp_deg;
                for (_tau, z_idx) in engine.product_indexed(m_deg, m_idx, mp_deg, mp_idx) {
                    let fidx = f_sm2.operation_generator_to_index(z_deg, z_idx, gl_deg, gl_idx);
                    *parity.entry(fidx).or_insert(0) ^= 1;
                }
            }
        }

        parity
            .into_iter()
            .filter(|(_, p)| p & 1 == 1)
            .map(|(fidx, _)| {
                // Every path to `fidx` has the same τ-power = W(fidx) − W(g_k)
                // (weight-homogeneity), so the parity above is well-defined.
                let power = self.entry_weight(s - 2, t, fidx) - w_k;
                (fidx, power)
            })
            .collect()
    }

    /// Correct `d_s(g_k)` (its `support`) so that `d_{s-1} d_s(g_k) = 0` over
    /// `A_C`. The remainder is `≡ 0 mod τ`; we cancel it one $\tau$-order at a
    /// time using the quasi-inverse of the mod-$\tau$ differential `d̄_{s-1}` (which
    /// the engine already computed). Each correction is weight-homogeneous of the
    /// weight matching the current $\tau$-order, so its $F_{s-1}$ support enters at
    /// a single forced $\tau$-power.
    fn correct(&self, g_k: Gen, support: &mut BTreeSet<usize>) {
        let s = g_k.s;
        let t = g_k.t;
        let p = self.resolution.prime();
        let f_sm1 = self.module(s - 1);
        let f_sm2 = self.module(s - 2);
        let f_sm1_dim = f_sm1.dimension(t);
        let f_sm2_dim = f_sm2.dimension(t);
        let d_prev = self.resolution.differential(s - 1);

        // The quasi-inverse of the mod-τ differential at this degree. If it was
        // not computed (a generator just past the padded box), skip correction.
        let Some(qi) = d_prev.quasi_inverse(t) else {
            return;
        };

        for _ in 0..256 {
            let err = self.composite(g_k, support);
            if err.is_empty() {
                return;
            }
            let min_power = *err.values().min().unwrap();
            assert!(
                min_power >= 1,
                "mod-τ composite d̄² ≠ 0 at (s={}, t={}, idx={}) — model is not a complex",
                g_k.s,
                g_k.t,
                g_k.idx
            );

            // The error at the lowest τ-order, as an F_{s-2} vector.
            let mut e_bar = FpVector::new(p, f_sm2_dim);
            for (&fidx, &power) in &err {
                if power == min_power {
                    e_bar.set_entry(fidx, 1);
                }
            }

            // Solve d̄_{s-1}(c') = e_bar via the stored quasi-inverse.
            let mut c = FpVector::new(p, f_sm1_dim);
            qi.apply(c.as_slice_mut(), 1, e_bar.as_slice());

            // Toggle c's support into d_s(g_k). Each such basis element is forced
            // (by weight) to τ-power min_power, cancelling this order.
            for (idx, v) in c.iter_nonzero() {
                if v == 0 {
                    continue;
                }
                debug_assert_eq!(
                    self.entry_weight(s - 1, t, idx) - self.weights[&g_k],
                    min_power,
                    "correction term at inconsistent τ-power"
                );
                if !support.insert(idx) {
                    support.remove(&idx);
                }
            }
        }
        // Non-convergence within the iteration cap means the generator's cone
        // reached past the padded box — which only happens outside the report
        // cone (report-cone generators converge, their cones staying inside the
        // box). Such generators are never read by the report cohomology, so we
        // leave the partial lift rather than panic. `verify_d_squared_zero` guards
        // the report box in tests.
        tracing::debug!(
            "motivic lift did not converge at (s={}, t={}, idx={}); leaving partial (outside report cone)",
            g_k.s, g_k.t, g_k.idx
        );
    }

    /// Assign a motivic weight to every generator by descending its differential:
    /// weight-homogeneity forces `w(g) = w(m) + w(g')` for every term `m ⊗ g'` of
    /// `d(g)`. The unit generator has weight 0; higher weights propagate. Panics
    /// if any generator turns out weight-inhomogeneous (which would violate the
    /// bigraded structure and break the valuation representation).
    fn compute_weights(&mut self) {
        // s = 0: the single generator is the unit, weight 0.
        self.weights.insert(Gen { s: 0, t: 0, idx: 0 }, 0);

        for s in 1..=self.max_s() {
            let d = self.resolution.differential(s);
            let target = self.module(s - 1);
            let t_max = self.compute.n() + s;
            for t in 0..=t_max {
                for idx in 0..self.num_gens(s, t) {
                    let out = d.output(t, idx);
                    let mut weight: Option<i32> = None;
                    for (bidx, v) in out.iter_nonzero() {
                        if v == 0 {
                            continue;
                        }
                        let og = target.index_to_op_gen(t, bidx);
                        let op_w = self.algebra.weight(og.operation_degree, og.operation_index);
                        let tgt = Gen {
                            s: s - 1,
                            t: og.generator_degree,
                            idx: og.generator_index,
                        };
                        let Some(&tgt_w) = self.weights.get(&tgt) else {
                            continue;
                        };
                        let w = op_w + tgt_w;
                        match weight {
                            None => weight = Some(w),
                            Some(w0) => assert_eq!(
                                w0, w,
                                "weight-inhomogeneous generator at (s={s}, t={t}, idx={idx})"
                            ),
                        }
                    }
                    if let Some(w) = weight {
                        self.weights.insert(Gen { s, t, idx }, w);
                    }
                }
            }
        }
    }

    /// The $\delta$-entries out of generator `g`: the identity-operation
    /// (augmentation) part of the lifted differential `d_s(g)`. Each entry is a
    /// target generator `g'` (at the same internal degree `t`, homological degree
    /// `s-1`) together with the $\tau$-power on the unit operation.
    ///
    /// This is the datum Phase 3 takes cohomology of: `δ` is a differential on the
    /// free $\mathbb{F}_2[\tau]$-modules of generators, and `Ext_{A_C} = H(δ)`
    /// (the motivic Adams $E_2$). Over a field the augmentation part of a minimal
    /// differential vanishes; here the $\tau$-power corrections create it.
    pub fn delta(&self, g: Gen) -> Vec<(Gen, u32)> {
        let module = self.module(g.s - 1);
        let w_k = self.weights[&g];
        let mut out = Vec::new();
        for &bidx in &self.lifted[&g] {
            let og = module.index_to_op_gen(g.t, bidx);
            // The identity operation is the unit: degree 0, index 0.
            if og.operation_degree == 0 {
                let gj = Gen {
                    s: g.s - 1,
                    t: og.generator_degree,
                    idx: og.generator_index,
                };
                let power = (self.weights[&gj] - w_k) as u32;
                out.push((gj, power));
            }
        }
        out
    }

    // ---- Phase 3: the cohomology H(δ) = the motivic Adams E₂ ----
    //
    // δ is a differential on the free F₂[τ]-modules of generators (fixed internal
    // degree `t`), with δ↓: gens(s,t) → gens(s−1,t) given by [`delta`]. Because
    // everything is weight-graded and τ is the only homogeneous prime, `Ext = H(δ)`
    // is a graded F₂[τ]-module: free ⊕ F₂[τ]/τᵏ. Two facts make this pure-F₂ work:
    //
    //  * δ↓ raises the (algebra) weight (a δ-entry g_k → g_j carries τ^{w_j−w_k},
    //    w_j ≥ w_k), so the dual δ* lowers it and `{weight ≤ cap}` is a subcomplex.
    //  * The free rank (what survives inverting τ = the classical Adams E₂) is the
    //    stable value, obtained with *all* generators (cap = ∞); each lower cap
    //    removes high-weight generators and exposes τ-torsion.

    /// The generators at `(s, t)` with algebra weight `≤ cap`, as their indices.
    fn gen_list(&self, s: i32, t: i32, cap: i32) -> Vec<usize> {
        (0..self.num_gens(s, t))
            .filter(|&idx| self.weights.get(&Gen { s, t, idx }).is_some_and(|&w| w <= cap))
            .collect()
    }

    /// The rank over $\mathbb{F}_2$ of the dual differential
    /// $\delta^*_{s} : C^{s} \to C^{s+1}$ (the transpose of $\delta$ from
    /// $(s+1,t)$ to $(s,t)$), restricted to generators of algebra weight `≤ cap`.
    fn delta_star_rank(&self, s: i32, t: i32, cap: i32) -> usize {
        let cols = self.gen_list(s, t, cap); // C^s   (source, g_k at s)
        let rows = self.gen_list(s + 1, t, cap); // C^{s+1} (g_l at s+1)
        if cols.is_empty() || rows.is_empty() {
            return 0;
        }
        let col_pos: HashMap<usize, usize> =
            cols.iter().enumerate().map(|(i, &g)| (g, i)).collect();
        let mut m = Matrix::new(TWO, rows.len(), cols.len());
        for (ri, &l) in rows.iter().enumerate() {
            for (gk, _power) in self.delta(Gen { s: s + 1, t, idx: l }) {
                if let Some(&cj) = col_pos.get(&gk.idx) {
                    m.row_mut(ri).set_entry(cj, 1);
                }
            }
        }
        m.row_reduce()
    }

    /// The $\mathbb{F}_2$-dimension of `Ext^{s,t}` in the weight slice `≤ cap`:
    /// $\dim H^s = \dim C^s - \mathrm{rank}\,\delta^*_s - \mathrm{rank}\,\delta^*_{s-1}$.
    ///
    /// With `cap = i32::MAX` this is the classical Adams $E_2$ rank (invert $\tau$);
    /// with finite caps it exposes the motivic $\tau$-torsion. Requires the lift to
    /// be computed at `s+1` (so `s + 1 ≤ max_s`).
    fn ext_dim(&self, s: i32, t: i32, cap: i32) -> usize {
        let n = self.gen_list(s, t, cap).len();
        let r_s = self.delta_star_rank(s, t, cap);
        let r_prev = if s > 0 {
            self.delta_star_rank(s - 1, t, cap)
        } else {
            0
        };
        n - r_s - r_prev
    }

    /// The classical Adams $E_2$ rank at `(s, t)` — invert $\tau$: the free rank of
    /// `H(δ)`, computed with all generators.
    pub fn classical_ext_rank(&self, s: i32, t: i32) -> usize {
        self.ext_dim(s, t, i32::MAX)
    }

    /// Whether `(s, t)` carries a $\tau$-torsion class in the motivic $E_2$: some
    /// weight slice has larger `H(δ)` than the free (classical) rank. The extra
    /// dimension is a class that dies when $\tau$ is inverted — genuine motivic
    /// $\tau$-torsion. Scans weight caps within the generators' weight range.
    pub fn has_tau_torsion(&self, s: i32, t: i32) -> bool {
        let free = self.classical_ext_rank(s, t);
        // The generator weights at these degrees bound the useful cap range.
        let weights: Vec<i32> = ((s - 1).max(0)..=s + 1)
            .flat_map(|ss| (0..self.num_gens(ss, t)).map(move |idx| (ss, idx)))
            .filter_map(|(ss, idx)| self.weights.get(&Gen { s: ss, t, idx }).copied())
            .collect();
        let (Some(&lo), Some(&hi)) = (weights.iter().min(), weights.iter().max()) else {
            return false;
        };
        (lo..=hi).any(|cap| self.ext_dim(s, t, cap) > free)
    }

    /// The mod-$\tau$ support of `d_s(g)`: the lifted terms whose forced
    /// $\tau$-power is $0$. These should reproduce the engine's $\bar d_s$ exactly.
    fn mod_tau_support(&self, g: Gen) -> BTreeSet<usize> {
        let w = self.weights[&g];
        self.lifted[&g]
            .iter()
            .copied()
            .filter(|&bidx| w - self.entry_weight(g.s - 1, g.t, bidx) == 0)
            .collect()
    }

    /// Verify `d_{s-1} d_s = 0` over `A_C` for every generator in range: the
    /// defining property of the lifted resolution.
    pub fn verify_d_squared_zero(&self) {
        for s in 2..=self.max_s() {
            for t in 0..=(self.max.n() + s) {
                for idx in 0..self.num_gens(s, t) {
                    let g = Gen { s, t, idx };
                    let err = self.composite(g, &self.lifted[&g]);
                    assert!(
                        err.is_empty(),
                        "d² ≠ 0 at (s={s}, t={t}, idx={idx}): {} surviving terms",
                        err.len()
                    );
                }
            }
        }
    }

    /// Verify that reducing every lifted differential mod $\tau$ recovers the
    /// original mod-$\tau$ resolution the engine computed.
    pub fn verify_mod_tau_reduction(&self) {
        for s in 1..=self.max_s() {
            for t in 0..=(self.max.n() + s) {
                for idx in 0..self.num_gens(s, t) {
                    let g = Gen { s, t, idx };
                    let engine_support: BTreeSet<usize> = self
                        .resolution
                        .differential(s)
                        .output(t, idx)
                        .iter_nonzero()
                        .filter(|(_, v)| *v != 0)
                        .map(|(i, _)| i)
                        .collect();
                    assert_eq!(
                        self.mod_tau_support(g),
                        engine_support,
                        "mod-τ reduction differs from the model at (s={s}, t={t}, idx={idx})"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lift_is_a_complex_and_reduces_correctly() {
        // Build once (expensive: the padded resolution plus the lift), then check
        // everything: every generator in the report box has a weight; reducing
        // each lifted differential mod τ recovers the Phase 1 model; and d² = 0
        // over A_C (the corrections worked).
        let max = Bidegree::n_s(8, 5);
        let res = MotivicResolution::new(max);

        // The unit has weight 0, and every generator in range has a weight.
        assert_eq!(res.generator_weight(Gen { s: 0, t: 0, idx: 0 }), 0);
        let mut count = 0;
        for s in 0..=res.max_s() {
            for t in 0..=(max.n() + s) {
                for idx in 0..res.num_gens(s, t) {
                    let _ = res.generator_weight(Gen { s, t, idx });
                    count += 1;
                }
            }
        }
        assert!(count > 15, "expected many generators, got {count}");

        res.verify_mod_tau_reduction();
        res.verify_d_squared_zero();

        // The lift must create a nontrivial augmentation part δ (the τ-power
        // corrections on the unit operation) — this is what Phase 3 takes
        // cohomology of. Every δ-entry carries a positive τ-power.
        let mut delta_entries = 0;
        for s in 1..=res.max_s() {
            for t in 0..=(max.n() + s) {
                for idx in 0..res.num_gens(s, t) {
                    for (_target, power) in res.delta(Gen { s, t, idx }) {
                        assert!(power >= 1, "δ entry with non-positive τ-power");
                        delta_entries += 1;
                    }
                }
            }
        }
        assert!(delta_entries > 0, "the lift produced no δ (augmentation) terms");
    }

    #[test]
    fn anchor_invert_tau_is_classical_adams_e2() {
        // Anchor 1: inverting τ (the free rank of H(δ), all generators) reproduces
        // the classical Adams E₂ — Ext over the mod-2 Steenrod algebra. We resolve
        // the classical sphere in-process and compare rank-for-rank. This is where
        // the τ-torsion (e.g. the h₁-tower classes beyond h₁⁴) is *killed*: the
        // motivic algebraic-Novikov extras collapse back onto classical Ext.
        use crate::{chain_complex::FreeChainComplex, utils::construct_standard};

        let max = Bidegree::n_s(8, 5);
        let res = MotivicResolution::new(max);

        let classical = construct_standard::<false, _, _>("S_2", None).unwrap();
        classical.compute_through_stem(max);

        // classical_ext_rank(s, t) needs the lift at s+1, so s ≤ max_s − 1.
        for s in 0..res.max_s() {
            for t in s..=(max.n() + s) {
                let n = t - s;
                if n > max.n() {
                    continue;
                }
                let got = res.classical_ext_rank(s, t);
                let want = classical.number_of_gens_in_bidegree(Bidegree::n_s(n, s));
                assert_eq!(
                    got, want,
                    "invert-τ mismatch at (n={n}, s={s}): H(δ) free rank {got} ≠ classical {want}"
                );
            }
        }
    }

    #[test]
    fn anchor_keep_tau_has_h1_tower_torsion() {
        // Anchor 3: keeping τ, the motivic E₂ carries τ-torsion the classical page
        // does not. The cleanest witness is the h₁-tower: h₁ⁿ ∈ Ext^{n,2n} is
        // nonzero for all n motivically, but classically h₁⁴ = 0. So at (s,t)=(4,8)
        // the free (classical) rank is 0, yet a τ-torsion class is present.
        let max = Bidegree::n_s(8, 5);
        let res = MotivicResolution::new(max);

        assert_eq!(
            res.classical_ext_rank(4, 8),
            0,
            "classical h₁⁴ should vanish"
        );
        assert!(
            res.has_tau_torsion(4, 8),
            "motivic E₂ should carry τ-torsion (h₁⁴) at (s=4, t=8)"
        );

        // Sanity: h₁³ (s=3, t=6) is already nonzero classically, so it is free —
        // not flagged as (extra) torsion beyond its classical class.
        assert_eq!(res.classical_ext_rank(3, 6), 1, "classical h₁³ should survive");
    }
}
