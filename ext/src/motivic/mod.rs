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
    sync::{Arc, OnceLock},
};

use algebra::{
    CTauAlgebra,
    module::{FDModule, FreeModule, Module, homomorphism::ModuleHomomorphism},
    motivic::MotivicMilnorAlgebra,
};
use bivec::BiVec;
use fp::{matrix::Matrix, prime::TWO, vector::FpVector};
use maybe_rayon::prelude::*;
use sseq::{
    Sseq, SseqProfile,
    coordinates::{Bidegree, degree::MultiDegree, element::MultiDegreeElement},
};

use crate::{
    chain_complex::{ChainComplex, FiniteChainComplex},
    ext_algebra::{ExtAlgebra, ExtDifferential},
    resolution::Resolution,
};

/// The $A_C/\tau$ resolution type: the trivial module resolved by the ordinary
/// engine over the mod-$\tau$ Steenrod algebra.
pub type CTauResolution = Resolution<FiniteChainComplex<FDModule<CTauAlgebra>>>;

/// The direction of the **deformation spectral sequence** (the algebraic Novikov
/// / $\tau$-Bockstein SS), trigraded by (stem $n$, Adams filtration $s$, weight
/// $w$). Each $d_r$ carries δ's fixed $(n, s) \mapsto (n+1, s-1)$ shift — δ lowers
/// Novikov filtration at fixed internal degree, so the stem rises — and jumps the
/// weight by $r$ (the $\tau$-power). $E_1 = \Ext_{A_C/\tau}$; inverting $\tau$
/// ($w \to \infty$) gives $E_\infty = $ classical $\Ext_A$, and finite-page deaths
/// are the motivic $\tau$-torsion.
pub struct Deformation;

impl SseqProfile<3> for Deformation {
    const MIN_R: i32 = 1;

    fn profile(r: i32, b: MultiDegree<3>) -> MultiDegree<3> {
        b + MultiDegree::from([1, -1, r])
    }

    fn profile_inverse(r: i32, b: MultiDegree<3>) -> MultiDegree<3> {
        b + MultiDegree::from([-1, 1, -r])
    }

    fn differential_length(offset: MultiDegree<3>) -> i32 {
        offset.coords()[2] // the weight component
    }
}

/// A generator of the resolution, identified by homological degree `s`, internal
/// degree `t`, and index within that `(s, t)` bidegree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Gen {
    pub s: i32,
    pub t: i32,
    pub idx: usize,
}

/// The dualized differential $\delta = \Hom(d, k)$ of the motivic resolution,
/// stored on the [`ExtAlgebra`] side (see the resolution/functor split in
/// [`MotivicResolution`]). It maps each generator to the target generators of the
/// augmentation (unit-operation) part of its lifted $A_C$ differential, together
/// with the forced $\tau$-power. As an [`ExtDifferential`] it shifts by
/// $(n, s) \mapsto (n+1, s-1)$ — δ lowers Novikov filtration at fixed internal
/// degree, which raises the stem — and its cohomology is the motivic Adams $E_2$.
struct MotivicCoboundary {
    resolution: Arc<CTauResolution>,
    /// `g ↦ [(target generator, τ-power)]` for the augmentation part of `d(g)`.
    /// Generators with no δ-terms are absent.
    deltas: HashMap<Gen, Vec<(Gen, u32)>>,
    /// The motivic weight of every generator — the $\mathbb{F}_2[\tau]$ grading
    /// the capped cohomology slices by. A generator absent here is weightless and
    /// excluded from every slice (matching the old `gen_list`). `Arc`-shared with
    /// [`MotivicResolution`], not copied.
    weights: Arc<HashMap<Gen, i32>>,
}

impl MotivicCoboundary {
    /// The generators at filtration `s`, internal degree `t` with weight `≤ cap`
    /// (weightless generators excluded), as their indices.
    fn kept(&self, s: i32, t: i32, cap: i32) -> Vec<usize> {
        (0..self.resolution.module(s).number_of_gens_in_degree(t))
            .filter(|&idx| {
                self.weights
                    .get(&Gen { s, t, idx })
                    .is_some_and(|&w| w <= cap)
            })
            .collect()
    }
}

impl ExtDifferential for MotivicCoboundary {
    fn shift(&self) -> Bidegree {
        Bidegree::n_s(1, -1)
    }

    fn matrix(&self, b: Bidegree) -> Option<Matrix> {
        self.matrix_capped(b, i32::MAX)
    }

    fn graded_dimension(&self, b: Bidegree, cap: i32) -> Option<usize> {
        if b.s() < 0 {
            return Some(0);
        }
        Some(self.kept(b.s(), b.t(), cap).len())
    }

    fn matrix_capped(&self, b: Bidegree, cap: i32) -> Option<Matrix> {
        let (t, s) = (b.t(), b.s());
        if s < 0 {
            return None;
        }
        let rows = self.kept(s, t, cap);
        if s == 0 {
            // δ out of filtration 0 lands in filtration −1: no targets.
            return Some(Matrix::new(TWO, rows.len(), 0));
        }
        let cols = self.kept(s - 1, t, cap);
        let col_pos: HashMap<usize, usize> =
            cols.iter().enumerate().map(|(p, &i)| (i, p)).collect();
        let mut m = Matrix::new(TWO, rows.len(), cols.len());
        for (rp, &idx) in rows.iter().enumerate() {
            if let Some(targets) = self.deltas.get(&Gen { s, t, idx }) {
                for &(tgt, _power) in targets {
                    if let Some(&cp) = col_pos.get(&tgt.idx) {
                        m.row_mut(rp).set_entry(cp, 1);
                    }
                }
            }
        }
        Some(m)
    }
}

/// The C-motivic resolution: the mod-$\tau$ model plus the weight assignment and
/// (Phase 2) the lifted $A_C$ differentials.
pub struct MotivicResolution {
    algebra: Arc<CTauAlgebra>,
    resolution: Arc<CTauResolution>,
    /// The Ext DGA: the mod-τ resolution wrapped so it stores the *dualized*
    /// differential δ = Hom(d, k) and takes its cohomology (the motivic Adams
    /// E₂). Built lazily on first cohomology query. This is the "Ext side" of the
    /// resolution/functor split — the raw differential stays in the resolution
    /// (`lifted`); δ lives here.
    ext: OnceLock<ExtAlgebra<CTauResolution>>,
    /// Motivic weight of each generator. `Arc`-shared with the Ext DGA's
    /// [`MotivicCoboundary`] (which slices by it) rather than copied.
    weights: Arc<HashMap<Gen, i32>>,
    /// The lifted $A_C$ differential of each generator: the set of $F_{s-1}$ basis
    /// elements in its image. The coefficient of each is $1 \in \mathbb{F}_2$ and
    /// its $\tau$-power is forced by the weights, so the support is the whole
    /// datum (see the module docs).
    lifted: HashMap<Gen, BTreeSet<usize>>,
    /// The box the results are trusted/reported in.
    max: Bidegree,
    /// The (padded) square actually resolved: `{stem ≤ compute.n(), filt ≤
    /// compute.s()}`. It is the report box `max` with a small stem margin for the
    /// lift's δ-reach (see [`Self::new`]).
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
        let resolution = Arc::new(Resolution::new(cc));
        // Resolve the report box **plus exactly one stem** with
        // `compute_through_stem`. Ext at `(s, t)` is `H(δ)`, and δ maps `(s, t) →
        // (s-1, t)` — same internal degree, one lower Novikov filtration, hence
        // stem `n → n+1`. So computing Ext at stem `n` needs the δ *out of* stem
        // `n`, whose targets are the generators at stem `n+1`; those must exist
        // (`delta_star_rank` reads `num_gens` there). That is the whole margin: a
        // hard, structural `+1`, not a fudge.
        //
        // Nothing at stem `n+2` is needed, and `compute_through_stem` gives the
        // `n+1` strip cheaply: at its edge it records only *kernels* one stem out
        // (`resolution.rs`), never resolving generators there. The lift of the
        // stem-`(n+1)` boundary generators therefore can't emit δ-terms into
        // stem `n+2` (those generators don't exist), and such a term would land in
        // internal degree `> n+s` anyway — invisible to every stem-`n` composite.
        // So `+1` is provably sufficient; `MOT_MARGIN` is only an escape hatch.
        let margin: i32 = std::env::var("MOT_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let compute = Bidegree::n_s(max.n() + margin, max.s());
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
            ext: OnceLock::new(),
            weights: Arc::new(HashMap::new()),
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

    /// XOR the contribution of a single `d_s(g_k)` support element `bidx` (a basis
    /// element of $F_{s-1}$ in degree `t`) into a running `d_{s-1} d_s` parity
    /// vector over the $F_{s-2}$ basis in degree `t`.
    ///
    /// `composite` is mod-2 linear in the support and each output basis element's
    /// $\tau$-power is a function of that element alone (weight-homogeneity), so
    /// this per-term atom is all both [`Self::composite`] and the incremental
    /// [`Self::correct`] need: toggling a term in or out of the support is exactly
    /// one call here (XOR is its own inverse).
    fn accumulate_term(
        &self,
        s: i32,
        t: i32,
        f_sm1: &FreeModule<CTauAlgebra>,
        f_sm2: &FreeModule<CTauAlgebra>,
        engine: &MotivicMilnorAlgebra,
        bidx: usize,
        parity: &mut FpVector,
    ) {
        let og = f_sm1.index_to_op_gen(t, bidx);
        let (m_deg, m_idx) = (og.operation_degree, og.operation_index);
        let gj = Gen {
            s: s - 1,
            t: og.generator_degree,
            idx: og.generator_index,
        };
        let gj_lifted = self.lifted.get(&gj).unwrap_or_else(|| {
            panic!(
                "composite references generator {gj:?} outside the resolved box \
                 (stem {} > compute stem {}); increase the lift stem margin (MOT_MARGIN)",
                gj.t - gj.s,
                self.compute.n(),
            )
        });
        for &bidx2 in gj_lifted {
            let og2 = f_sm2.index_to_op_gen(gj.t, bidx2);
            let (mp_deg, mp_idx) = (og2.operation_degree, og2.operation_index);
            let (gl_deg, gl_idx) = (og2.generator_degree, og2.generator_index);
            let z_deg = m_deg + mp_deg;
            engine.product_indexed_with(m_deg, m_idx, mp_deg, mp_idx, |terms| {
                for &(_tau, z_idx) in terms {
                    let fidx = f_sm2.operation_generator_to_index(z_deg, z_idx, gl_deg, gl_idx);
                    parity.add_basis_element(fidx, 1); // XOR at p = 2
                }
            });
        }
    }

    /// The composite `d_{s-1} d_s(g_k)` over `A_C`, as a map from $F_{s-2}$ basis
    /// element (in degree `g_k.t`) to its forced $\tau$-power. Only odd-parity
    /// (surviving) terms are returned. The `A_C` products `m · m'` are where the
    /// $\tau$-divisible terms are generated.
    fn composite(&self, g_k: Gen, support: &BTreeSet<usize>) -> BTreeMap<usize, i32> {
        let s = g_k.s;
        let t = g_k.t;
        let w_k = self.weights[&g_k];
        let f_sm1 = self.module(s - 1);
        let f_sm2 = self.module(s - 2);
        let engine = self.algebra.engine();

        let mut parity = FpVector::new(TWO, f_sm2.dimension(t));
        for &bidx in support {
            self.accumulate_term(s, t, &f_sm1, &f_sm2, engine, bidx, &mut parity);
        }

        parity
            .iter_nonzero()
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
        let w_k = self.weights[&g_k];
        let f_sm1 = self.module(s - 1);
        let f_sm2 = self.module(s - 2);
        let f_sm1_dim = f_sm1.dimension(t);
        let f_sm2_dim = f_sm2.dimension(t);
        let engine = self.algebra.engine();
        let d_prev = self.resolution.differential(s - 1);

        // The quasi-inverse of the mod-τ differential at this degree. If it was
        // not computed (a generator just past the padded box), skip correction.
        let Some(qi) = d_prev.quasi_inverse(t) else {
            return;
        };

        // Maintain the composite `d_{s-1} d_s(g_k)` incrementally. `composite` is
        // mod-2 linear in the support, so instead of re-sweeping the whole support
        // every τ-order (the old inner loop), we keep a running parity vector and
        // XOR in only each term we toggle. Seed it with the current support.
        let mut parity = FpVector::new(p, f_sm2_dim);
        for &bidx in support.iter() {
            self.accumulate_term(s, t, &f_sm1, &f_sm2, engine, bidx, &mut parity);
        }

        for _ in 0..256 {
            // The lowest τ-order among the surviving error terms.
            let min_power = parity
                .iter_nonzero()
                .map(|(fidx, _)| self.entry_weight(s - 2, t, fidx) - w_k)
                .min();
            let Some(min_power) = min_power else {
                return; // d_{s-1} d_s(g_k) = 0
            };
            assert!(
                min_power >= 1,
                "mod-τ composite d̄² ≠ 0 at (s={}, t={}, idx={}) — model is not a complex",
                g_k.s,
                g_k.t,
                g_k.idx
            );

            // The error at the lowest τ-order, as an F_{s-2} vector.
            let mut e_bar = FpVector::new(p, f_sm2_dim);
            for (fidx, _) in parity.iter_nonzero() {
                if self.entry_weight(s - 2, t, fidx) - w_k == min_power {
                    e_bar.set_entry(fidx, 1);
                }
            }

            // Solve d̄_{s-1}(c') = e_bar via the stored quasi-inverse.
            let mut c = FpVector::new(p, f_sm1_dim);
            qi.apply(c.as_slice_mut(), 1, e_bar.as_slice());

            // Toggle c's support into d_s(g_k), updating the running parity in
            // lockstep. Each such basis element is forced (by weight) to τ-power
            // min_power, cancelling this order.
            for (idx, v) in c.iter_nonzero() {
                if v == 0 {
                    continue;
                }
                debug_assert_eq!(
                    self.entry_weight(s - 1, t, idx) - w_k,
                    min_power,
                    "correction term at inconsistent τ-power"
                );
                if !support.insert(idx) {
                    support.remove(&idx);
                }
                self.accumulate_term(s, t, &f_sm1, &f_sm2, engine, idx, &mut parity);
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
        // Built locally, then `Arc`-shared into `self.weights` (and the Ext DGA) — no copy.
        let mut weights: HashMap<Gen, i32> = HashMap::new();
        // s = 0: the single generator is the unit, weight 0.
        weights.insert(Gen { s: 0, t: 0, idx: 0 }, 0);

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
                        let Some(&tgt_w) = weights.get(&tgt) else {
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
                        weights.insert(Gen { s, t, idx }, w);
                    }
                }
            }
        }
        self.weights = Arc::new(weights);
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
    // is a graded F₂[τ]-module: free ⊕ F₂[τ]/τᵏ. δ↓ raises the weight, so
    // `{weight ≤ cap}` is a subcomplex — and taking that cohomology (free rank at
    // cap = ∞, torsion exposed at lower caps) is now the job of the [`ExtAlgebra`]
    // built by [`Self::build_ext`], via [`MotivicCoboundary`] as its differential.

    /// Build the Ext DGA: apply $\Hom(-, k)$ to the raw lifted differential
    /// (`lifted`), producing the dualized coboundary δ, and hand it to an
    /// [`ExtAlgebra`] over the mod-τ resolution. This is where the $\Hom$ functor
    /// is applied and its result stored on the Ext side; the resolution keeps only
    /// the raw differential.
    fn build_ext(&self) -> ExtAlgebra<CTauResolution> {
        let mut deltas: HashMap<Gen, Vec<(Gen, u32)>> = HashMap::new();
        for s in 1..=self.max_s() {
            let t_max = self.compute.n() + s;
            for t in 0..=t_max {
                for idx in 0..self.num_gens(s, t) {
                    let g = Gen { s, t, idx };
                    if self.lifted.contains_key(&g) && self.weights.contains_key(&g) {
                        let d = self.delta(g);
                        if !d.is_empty() {
                            deltas.insert(g, d);
                        }
                    }
                }
            }
        }
        let coboundary = Arc::new(MotivicCoboundary {
            resolution: Arc::clone(&self.resolution),
            deltas,
            weights: Arc::clone(&self.weights),
        });
        ExtAlgebra::without_unit(Arc::clone(&self.resolution)).with_differential(coboundary)
    }

    /// The Ext DGA (built lazily): the mod-τ resolution wrapped so it stores the
    /// dualized differential δ.
    ///
    /// Its cochain level is `Ext_{A_C/τ}` — the algebraic Novikov $E_2$, equal to
    /// the Adams $E_2$ of $C\tau$ — so its [`multiply`](ExtAlgebra::multiply) and
    /// Massey products are the **$C\tau$ (E₁-page) products**. Its
    /// [`cohomology_dimension`](ExtAlgebra::cohomology_dimension) is the motivic
    /// Adams $E_2$ = `H(δ)`. (Products *on* `H(δ)` — the motivic ring — are the
    /// cochain products descended through the cohomology, which is separate.)
    pub fn ext(&self) -> &ExtAlgebra<CTauResolution> {
        self.ext.get_or_init(|| self.build_ext())
    }

    /// The deformation spectral sequence as an [`Sseq`], trigraded by $(n, s, w)$
    /// ([`Deformation`]). $E_1 = \Ext_{A_C/\tau}$ — the mod-τ generators, grouped
    /// by weight — with $d_1$ the **weight-1 part of δ** (the induced differential
    /// on the associated graded, read directly off [`delta`](Self::delta)).
    /// [`Sseq::update`] then computes $E_2 = H(d_1)$, whose [`page_data`] are the
    /// `Subquotient`s.
    ///
    /// The higher $d_r$ are then computed by the **τ-Bockstein zig-zag** on the full
    /// $\mathbb{F}_2[\tau]$-linear δ (`slice`/`inject`/`apply_delta` below), pushing
    /// $\delta(\tilde x)$ up in weight one order at a time — each correction solved by
    /// $d_1$'s stored quasi-inverse — and reading the weight-$(w+r)$ part, until a
    /// page adds nothing ($E_\infty$). The products (via [`Sseq::multiply`]) build on
    /// this.
    ///
    /// [`page_data`]: Sseq::page_data
    pub fn deformation_sseq(&self) -> Sseq<3, Deformation> {
        let mut sseq = Sseq::<3, Deformation>::new(TWO);

        // Group generators by (n, s, w); a generator's position within its group is
        // its coordinate in that multidegree's space.
        let mut groups: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
        let mut pos: HashMap<Gen, (MultiDegree<3>, usize)> = HashMap::new();
        for s in 0..=self.max_s() {
            let t_max = self.compute.n() + s;
            for t in 0..=t_max {
                for idx in 0..self.num_gens(s, t) {
                    let g = Gen { s, t, idx };
                    if let Some(&w) = self.weights.get(&g) {
                        let key = [t - s, s, w];
                        let list = groups.entry(key).or_default();
                        pos.insert(g, (MultiDegree::from(key), list.len()));
                        list.push(idx);
                    }
                }
            }
        }
        for (&key, list) in &groups {
            sseq.set_dimension(MultiDegree::from(key), list.len());
        }

        // d₁ = the weight-1 part of δ; a generator with δ = ∅ is a permanent cycle.
        for (&g, &(deg, p)) in &pos {
            let mut source = FpVector::new(TWO, sseq.dimension(deg));
            source.set_entry(p, 1);
            // s = 0 is the unit: no differential (δ is only defined for s ≥ 1).
            if g.s == 0 {
                sseq.add_permanent_class(&MultiDegreeElement::new(deg, source));
                continue;
            }
            let d = self.delta(g);
            if d.is_empty() {
                sseq.add_permanent_class(&MultiDegreeElement::new(deg, source));
                continue;
            }
            let target_deg = Deformation::profile(1, deg);
            let mut target = FpVector::new(TWO, sseq.dimension(target_deg));
            let mut hit = false;
            for (gj, power) in d {
                if power == 1 {
                    target.set_entry(pos[&gj].1, 1);
                    hit = true;
                }
            }
            // No weight-1 term ⟹ g is a d₁-cycle (its leading δ term is higher order).
            if hit {
                sseq.add_differential(1, &MultiDegreeElement::new(deg, source), target.as_slice());
            }
        }
        sseq.update();

        // Higher d_r by the τ-Bockstein zig-zag on the full δ. `apply_delta`
        // evaluates δ on a raw cochain over the generators at (s, t); `slice`
        // extracts a fixed-weight part into (n,s,w)-group coordinates; `inject` adds
        // a group-coordinate vector back into a raw cochain.
        let apply_delta = |xtilde: &FpVector, s: i32, t: i32| -> FpVector {
            let mut dvec = FpVector::new(TWO, self.num_gens(s - 1, t));
            for (gi, _) in xtilde.iter_nonzero() {
                for (gj, _power) in self.delta(Gen { s, t, idx: gi }) {
                    dvec.add_basis_element(gj.idx, 1);
                }
            }
            dvec
        };
        let slice = |dvec: &FpVector, key: [i32; 3]| -> FpVector {
            let g = groups.get(&key).map(Vec::as_slice).unwrap_or(&[]);
            let mut y = FpVector::new(TWO, g.len());
            for (p, &gi) in g.iter().enumerate() {
                if dvec.entry(gi) != 0 {
                    y.set_entry(p, 1);
                }
            }
            y
        };
        let inject = |xtilde: &mut FpVector, key: [i32; 3], gv: &FpVector| {
            if let Some(g) = groups.get(&key) {
                for (p, _) in gv.iter_nonzero() {
                    xtilde.add_basis_element(g[p], 1);
                }
            }
        };

        let degrees: Vec<MultiDegree<3>> = sseq.iter_degrees().collect();
        let mut r = 2;
        loop {
            let mut to_add: Vec<(MultiDegree<3>, FpVector, FpVector)> = Vec::new();
            for &b in &degrees {
                let [n, s, w] = b.coords();
                if s < 1 {
                    continue;
                }
                let page = sseq.page_data(b);
                if page.len() <= r {
                    continue;
                }
                let t = n + s;
                for rep in page[r].gens() {
                    // x̃ = the E_r representative, lifted to a raw cochain over (s, t).
                    let mut xtilde = FpVector::new(TWO, self.num_gens(s, t));
                    if let Some(g) = groups.get(&[n, s, w]) {
                        for (p, _) in rep.iter_nonzero() {
                            xtilde.add_basis_element(g[p], 1);
                        }
                    }
                    // Push δ(x̃) up in weight, correcting each intermediate order.
                    let mut ok = true;
                    for k in 1..r {
                        let y = slice(&apply_delta(&xtilde, s, t), [n + 1, s - 1, w + k]);
                        if y.is_zero() {
                            continue;
                        }
                        let src = MultiDegree::from([n, s, w + k - 1]);
                        if !sseq.defined(src) || sseq.differentials(src).len() <= 1 {
                            ok = false; // no d₁ to invert (should not happen for a genuine cycle)
                            break;
                        }
                        let mut c = FpVector::new(TWO, sseq.dimension(src));
                        sseq.differentials(src)[1].quasi_inverse(c.as_slice_mut(), y.as_slice());
                        inject(&mut xtilde, [n, s, w + k - 1], &c);
                    }
                    if !ok {
                        continue;
                    }
                    let target = slice(&apply_delta(&xtilde, s, t), [n + 1, s - 1, w + r]);
                    if !target.is_zero() {
                        to_add.push((b, rep.to_owned(), target));
                    }
                }
            }
            let mut added = false;
            for (b, source, target) in to_add {
                if sseq.add_differential(r, &MultiDegreeElement::new(b, source), target.as_slice()) {
                    added = true;
                }
            }
            sseq.update();
            if !added {
                break;
            }
            r += 1;
        }

        sseq
    }

    /// The classical Adams $E_2$ rank at `(s, t)` — invert $\tau$: the free rank of
    /// `H(δ)`, computed with all generators.
    ///
    /// This is [`ExtAlgebra::cohomology_dimension`] of the Ext DGA — the motivic
    /// Adams $E_2$ as the cohomology of the dualized differential δ.
    pub fn classical_ext_rank(&self, s: i32, t: i32) -> usize {
        self.ext()
            .cohomology_dimension(Bidegree::n_s(t - s, s))
            .unwrap_or(0)
    }

    /// Whether `(s, t)` carries a $\tau$-torsion class in the motivic $E_2$: some
    /// weight slice has larger `H(δ)` than the free (classical) rank. The extra
    /// dimension is a class that dies when $\tau$ is inverted — genuine motivic
    /// $\tau$-torsion. Scans weight caps within the generators' weight range.
    pub fn has_tau_torsion(&self, s: i32, t: i32) -> bool {
        let b = Bidegree::n_s(t - s, s);
        let ext = self.ext();
        let free = ext.cohomology_dimension(b).unwrap_or(0);
        // The generator weights at these degrees bound the useful cap range.
        let weights: Vec<i32> = ((s - 1).max(0)..=s + 1)
            .flat_map(|ss| (0..self.num_gens(ss, t)).map(move |idx| (ss, idx)))
            .filter_map(|(ss, idx)| self.weights.get(&Gen { s: ss, t, idx }).copied())
            .collect();
        let (Some(&lo), Some(&hi)) = (weights.iter().min(), weights.iter().max()) else {
            return false;
        };
        (lo..=hi).any(|cap| ext.cohomology_dimension_capped(b, cap).unwrap_or(0) > free)
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
    use sseq::coordinates::BidegreeGenerator;

    use super::*;

    #[test]
    fn ctau_products_run_via_ext_algebra() {
        // ExtAlgebra's product machinery runs on the motivic (mod-τ) resolution:
        // the cochain ring is Ext_{A_C/τ} = the algebraic Novikov E₂ = the Adams
        // E₂ of Cτ. Exercise the h₀-tower, which is present there:
        //   h₀ ∈ (n=0, s=1); h₀ⁿ is the (single) nonzero class of (n=0, s=n).
        let res = MotivicResolution::new(Bidegree::n_s(8, 5));
        let ext = res.ext();

        assert_eq!(ext.dimension(Bidegree::n_s(0, 1)), 1, "h₀ is 1-dimensional");
        let h0 = ext.generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));

        // h₀ⁿ = h₀·h₀ⁿ⁻¹ stays nonzero up the tower (Cτ has an infinite h₀-tower).
        let mut power = h0.clone();
        for n in 2..=4 {
            power = ext.multiply(&h0, &power);
            let deg = Bidegree::n_s(0, n);
            assert_eq!(power.degree(), deg, "h₀^{n} lands in (0,{n})");
            assert_eq!(ext.dimension(deg), 1, "(0,{n}) is 1-dimensional");
            assert!(!power.vec().is_zero(), "h₀^{n} should be nonzero");
        }
    }

    #[test]
    fn deformation_sseq_foundation() {
        // The deformation SS stored as an Sseq: E₁ = Ext_{A_C/τ} trigraded by
        // (n,s,w), d₁ = the weight-1 part of δ.
        let res = MotivicResolution::new(Bidegree::n_s(8, 5));
        let sseq = res.deformation_sseq();

        // E₁ grouping: summing the weight slices at (n,s) recovers the algebraic
        // Novikov rank (the mod-τ generator count).
        let mut totals: HashMap<(i32, i32), usize> = HashMap::new();
        for deg in sseq.iter_degrees() {
            let [n, s, _] = deg.coords();
            *totals.entry((n, s)).or_default() += sseq.dimension(deg);
        }
        for (&(n, s), &total) in &totals {
            if (0..=8).contains(&n) && (0..=5).contains(&s) {
                assert_eq!(
                    total,
                    res.algebraic_novikov_rank(s, n + s),
                    "E₁ weight slices at (n={n}, s={s}) should sum to the generator count"
                );
            }
        }

        // h₀ ∈ (n=0, s=1, w=0): δ(h₀) = ∅, so it is a permanent cycle.
        let h0 = MultiDegree::from([0, 1, 0]);
        assert_eq!(sseq.get_dimension(h0), Some(1), "h₀ is 1-dimensional");
        assert_eq!(
            sseq.permanent_classes(h0).dimension(),
            1,
            "h₀ (δ = ∅) should be a permanent cycle"
        );

        // Every d₁ we added is consistent.
        for deg in sseq.iter_degrees() {
            assert!(!sseq.inconsistent(deg), "inconsistent differential at {deg}");
        }
    }

    #[test]
    fn deformation_sseq_converges_to_classical() {
        // The strong oracle for the full d_r tower: E_∞ survivors summed over weight
        // = the classical Adams E₂ (invert τ). Every differential is validated —
        // a wrong d_r would leave the free rank off at some bidegree.
        let max = Bidegree::n_s(8, 5);
        let res = MotivicResolution::new(max);
        let sseq = res.deformation_sseq();

        let mut einf: HashMap<(i32, i32), usize> = HashMap::new();
        for deg in sseq.iter_degrees() {
            let [n, s, _] = deg.coords();
            let page = sseq.page_data(deg);
            let last = page.len() - 1; // the final (E_∞) page for this degree
            *einf.entry((n, s)).or_default() += page[last].dimension();
        }

        for s in 0..res.max_s() {
            for n in 0..=8 {
                let got = einf.get(&(n, s)).copied().unwrap_or(0);
                let want = res.classical_ext_rank(s, n + s);
                assert_eq!(got, want, "E_∞ free rank at (n={n}, s={s})");
            }
        }
    }

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
