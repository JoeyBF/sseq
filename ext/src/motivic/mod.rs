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
//! - **keep $\tau$** — free plus $\tau$-torsion: the motivic $E_2$ as a full
//!   $\mathbb{F}_2[\tau]$-module ([`MotivicResolution::tau_module`], structure
//!   theorem), including the $h_1$-tower classes ($h_1^n$, which are
//!   $\mathbb{F}_2[\tau]/\tau$-torsion for $n \ge 4$) that the classical page kills.

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
use once::MultiIndexed;
use sseq::{
    Product, Sseq, SseqProfile,
    coordinates::{Bidegree, BidegreeGenerator, degree::MultiDegree, element::MultiDegreeElement},
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

/// The motivic Adams $E_2$ at one bidegree as an $\mathbb{F}_2[\tau]$-module
/// (structure theorem over the PID $\mathbb{F}_2[\tau]$):
/// $\mathbb{F}_2[\tau]^{\,\mathrm{free}} \oplus \bigoplus_i
/// \mathbb{F}_2[\tau]/\tau^{k_i}$. Produced by [`MotivicResolution::tau_module`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TauModule {
    /// The free rank — the classical Adams $E_2$ (invert $\tau$).
    pub free: usize,
    /// The orders $k_i$ of the $\tau$-torsion summands $\mathbb{F}_2[\tau]/\tau^{k_i}$,
    /// ascending. Empty iff the module is $\tau$-torsion-free.
    pub torsion: Vec<u32>,
}

impl std::fmt::Display for TauModule {
    /// A compact chart label, e.g. `2` (free rank 2, no torsion), `τ` (one
    /// $\mathbb{F}_2[\tau]/\tau$), `1+τ²`, or `·` for the zero module.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        if self.free > 0 {
            parts.push(self.free.to_string());
        }
        for &k in &self.torsion {
            parts.push(if k == 1 {
                "τ".to_string()
            } else {
                format!("τ^{k}")
            });
        }
        if parts.is_empty() {
            write!(f, "·")
        } else {
            write!(f, "{}", parts.join("+"))
        }
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
    /// The deformation (algebraic Novikov / τ-Bockstein) spectral sequence, built
    /// lazily — the source of truth for the free rank and τ-torsion. See
    /// [`Self::deformation_sseq`].
    deformation: OnceLock<Sseq<3, Deformation>>,
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
            deformation: OnceLock::new(),
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
    ///
    /// The correction is the [`TauLift`] driver applied to the differential cell
    /// (via [`DifferentialCells`]); it runs only for generators whose δ-correction
    /// cone stays inside the padded box. The cone of a generator at stem `n` reaches
    /// up to stem `n + s` (each augmentation correction pushes one stem out), so a
    /// generator with stem `> report.n + report.s` cannot converge — and it is never
    /// referenced by the report cohomology (differentials go to stem ≤ n, δ-terms to
    /// n+1, and the report cone is bounded by report.n + report.s). Leaving those as
    /// their mod-τ support is correct.
    fn lift_generator(&self, g: Gen) -> BTreeSet<usize> {
        let cells = DifferentialCells(self);
        let stem = g.t - g.s;
        let in_cone = stem <= self.max.n() + self.max.s();
        if g.s >= 2 && in_cone && self.weights.contains_key(&g) {
            cells.lift_cell(g)
        } else {
            cells.seed(g)
        }
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

    /// Group generators by their multidegree `(n, s, w)`. Returns the per-multidegree
    /// generator-index lists (a generator's position in its list is its Sseq
    /// coordinate) and the reverse map `Gen ↦ (multidegree, position)`.
    fn sseq_grouping(
        &self,
    ) -> (
        HashMap<[i32; 3], Vec<usize>>,
        HashMap<Gen, (MultiDegree<3>, usize)>,
    ) {
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
        (groups, pos)
    }

    /// A [`Product`] on the deformation SS for each requested Cτ generator `a`
    /// (given as `(bidegree, index)`): multiplication by `a`, taken from
    /// [`ExtAlgebra::multiply_into`] on the mod-τ resolution and split into weight
    /// blocks for the trigrading. Feeding one to [`Sseq::multiply`] applies it on
    /// any page — the Cτ ring on $E_1$, the motivic Adams $E_2$ ring on $E_\infty$.
    pub fn deformation_products(&self, gens: &[(Bidegree, usize)]) -> Vec<Product<3>> {
        let ext = self.ext();
        let (groups, _) = self.sseq_grouping();
        let empty = Vec::new();
        gens.iter()
            .map(|&(a_deg, a_idx)| {
                let a_gen = Gen { s: a_deg.s(), t: a_deg.t(), idx: a_idx };
                let a_w = self.weights[&a_gen];
                let a_elem = ext.generator(BidegreeGenerator::new(a_deg, a_idx));
                let matrices = MultiIndexed::new();
                for (&[n, s, w], src_group) in &groups {
                    let Some(full) = ext.multiply_into(&a_elem, Bidegree::n_s(n, s)) else {
                        continue;
                    };
                    let tgt = groups
                        .get(&[n + a_deg.n(), s + a_deg.s(), w + a_w])
                        .unwrap_or(&empty);
                    // Row `i` = a · (source generator i), restricted to the
                    // weight-`(w + a_w)` target generators, in group coordinates.
                    let rows: Vec<Vec<u32>> = src_group
                        .iter()
                        .map(|&raw_i| tgt.iter().map(|&raw_j| full.row(raw_i).entry(raw_j)).collect())
                        .collect();
                    matrices.insert(MultiDegree::from([n, s, w]), Matrix::from_vec(TWO, &rows));
                }
                Product {
                    b: MultiDegree::from([a_deg.n(), a_deg.s(), a_w]),
                    left: true,
                    matrices,
                }
            })
            .collect()
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
    pub fn deformation_sseq(&self) -> &Sseq<3, Deformation> {
        self.deformation.get_or_init(|| self.build_deformation_sseq())
    }

    fn build_deformation_sseq(&self) -> Sseq<3, Deformation> {
        let mut sseq = Sseq::<3, Deformation>::new(TWO);

        // Group generators by (n, s, w) — a generator's position within its group
        // is its coordinate in that multidegree.
        let (groups, pos) = self.sseq_grouping();
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
                // Only report-box sources: their δ-targets (stem n+1 ≤ compute.n())
                // are resolved, and every differential touching a report degree has
                // a report-box source, so the report E_∞ is unaffected. Margin
                // sources would reach stem max.n+2 (unresolved).
                if s < 1 || n > self.max.n() {
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
    /// the motivic $E_2$.
    ///
    /// Read off the deformation SS: the $E_\infty$ survivors at `(n, s)` summed over
    /// weight. Inverting $\tau$ ($w \to \infty$) is exactly the classical Adams
    /// $E_2$; the τ-torsion classes die on finite pages and drop out.
    pub fn classical_ext_rank(&self, s: i32, t: i32) -> usize {
        let n = t - s;
        let sseq = self.deformation_sseq();
        sseq.iter_degrees()
            .filter(|d| {
                let c = d.coords();
                c[0] == n && c[1] == s
            })
            .map(|d| {
                let page = sseq.page_data(d);
                page[page.len() - 1].dimension() // E_∞ at (n, s, w)
            })
            .sum()
    }

    /// The motivic Adams $E_2$ at `(s, t)` as an $\mathbb{F}_2[\tau]$-module, via the
    /// structure theorem for modules over the PID $\mathbb{F}_2[\tau]$:
    /// $$H(\delta)^{n,s} \;\cong\; \mathbb{F}_2[\tau]^{\,\mathrm{free}} \;\oplus\;
    /// \bigoplus_i \mathbb{F}_2[\tau]/\tau^{k_i}.$$
    /// The free rank is the classical Adams $E_2$ (invert $\tau$); the torsion orders
    /// $k_i$ are the sizes of the $\tau$-torsion summands the classical page cannot
    /// see (e.g. the $h_1$-tower).
    ///
    /// $(C^\bullet, \delta)$ is a complex of *free* $\mathbb{F}_2[\tau]$-modules of
    /// generators. `Ext = H(δ)` is its *cohomology*, so by universal coefficients its
    /// torsion at $s$ is the torsion of the homology one degree down, i.e. the
    /// non-unit invariant factors of the *outgoing* boundary
    /// $\delta\colon \mathrm{gens}(s, t) \to \mathrm{gens}(s{-}1, t)$: a class is
    /// $\tau^{k}$-torsion exactly when its own δ is $\tau^{k}$ times a cycle
    /// (e.g. $\delta(h_1^4) = \tau\, y$ puts $\mathbb{F}_2[\tau]/\tau$ at $h_1^4$).
    /// Those factors are read off the Smith normal form of that δ matrix over
    /// $\mathbb{F}_2[\tau]$ ([`Self::tau_torsion_orders`]); the only homogeneous prime
    /// is $\tau$, so every invariant factor is a power $\tau^{k_i}$. This is the same
    /// data the deformation SS carries as its $d_r$ differentials *supported* at
    /// `(n, s)` (order $= r$), computed locally here so it is exact up to the resolved
    /// range rather than the SS's report box.
    pub fn tau_module(&self, s: i32, t: i32) -> TauModule {
        TauModule {
            free: self.classical_ext_rank(s, t),
            torsion: self.tau_torsion_orders(s, t),
        }
    }

    /// Whether `(s, t)` carries a $\tau$-torsion class in the motivic $E_2$ — a class
    /// that dies when $\tau$ is inverted, invisible to the classical Adams $E_2$.
    /// The boolean shadow of [`Self::tau_module`].
    pub fn has_tau_torsion(&self, s: i32, t: i32) -> bool {
        !self.tau_torsion_orders(s, t).is_empty()
    }

    /// The orders $k_i$ of the $\tau$-torsion summands $\mathbb{F}_2[\tau]/\tau^{k_i}$
    /// of the motivic $E_2$ at `(s, t)`, ascending: the non-unit invariant factors
    /// (as powers of $\tau$) of the outgoing coboundary
    /// $\delta\colon \mathrm{gens}(s, t) \to \mathrm{gens}(s{-}1, t)$, over
    /// $\mathbb{F}_2[\tau]$. (Outgoing, not incoming: `Ext = H(δ)` is cohomology, so
    /// the torsion sits on the class whose own δ is $\tau$-divisible.)
    fn tau_torsion_orders(&self, s: i32, t: i32) -> Vec<u32> {
        if s < 1 {
            return Vec::new(); // the s = 0 unit is free; no outgoing δ.
        }
        let rows = self.num_gens(s, t);
        let cols = self.num_gens(s - 1, t);
        if rows == 0 || cols == 0 {
            return Vec::new();
        }
        // The δ matrix over F₂[τ]: entry (i, j) is the sum of τ^power over the terms
        // of δ(gᵢ) hitting the j-th generator at (s-1, t). Packed as a bitmask of
        // τ-exponents (F₂ coefficients).
        let mut m = vec![vec![0u128; cols]; rows];
        for (ri, row) in m.iter_mut().enumerate() {
            for (gj, power) in self.delta(Gen { s, t, idx: ri }) {
                if gj.s == s - 1 && gj.idx < cols {
                    row[gj.idx] ^= 1u128 << power;
                }
            }
        }
        let mut orders: Vec<u32> = f2tau::invariant_factors(m)
            .into_iter()
            .map(|f| f2tau::deg(f) as u32)
            .collect();
        orders.sort_unstable();
        orders
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

/// The shared τ-adic lifting problem, in the style of [`crate::secondary::SecondaryLift`].
///
/// Every "make it motivic" step in this module has the same shape: a map given over
/// the mod-τ algebra $A_C/\tau$ must be lifted to an honest map over $A_C$
/// (coefficients in $\mathbb{F}_2[\tau]$). The lift starts from the mod-τ datum (the
/// $\tau^0$ part) and cancels the τ-divisible *defect* — the amount by which the
/// defining equation fails over $A_C$ — one weight-order at a time, solving each
/// order with the quasi-inverse of the target complex's mod-τ differential.
/// Weight-homogeneity forces every correction to a single τ-power, so the
/// order-by-order cancellation converges.
///
/// The differential lift ($d^2 = 0$, [`DifferentialCells`]) is the first instance;
/// the product lift ($d\varphi = \varphi d$) and — eventually — the chain-homotopy
/// lift (Massey products) are the same driver with a different *defect*. An
/// implementor supplies the object-specific hooks and inherits [`Self::lift_cell`].
trait TauLift {
    /// The weight the defect is graded against — the source generator's weight.
    fn source_weight(&self, g: Gen) -> i32;

    /// The mod-τ ($\tau^0$) support to start from, as basis-element indices of the
    /// output module (where the lifted support and the corrections live).
    fn seed(&self, g: Gen) -> BTreeSet<usize>;

    /// The defect module `(module, t)` — where the error `e` lives — used to size
    /// the running parity vector.
    fn defect_module(&self, g: Gen) -> (Arc<FreeModule<CTauAlgebra>>, i32);

    /// The weight of defect-module basis element `bidx`.
    fn defect_weight(&self, g: Gen, bidx: usize) -> i32;

    /// Seed the running defect with the part that does not depend on the output
    /// support (e.g. the `φ(dg)` term of a chain map). Default: none — the
    /// differential's `d²` is entirely a function of its support.
    fn seed_constant(&self, _g: Gen, _parity: &mut FpVector) {}

    /// XOR into `parity` the defect contribution of output-support element `bidx`.
    fn accumulate(&self, g: Gen, bidx: usize, parity: &mut FpVector);

    /// Solve `d̄(c) = e` for a correction `c` in the output module, via the target's
    /// mod-τ quasi-inverse. `None` when the quasi-inverse is unavailable (a cell just
    /// past the padded box) — the driver then leaves the mod-τ seed uncorrected.
    fn solve(&self, g: Gen, e: &FpVector) -> Option<FpVector>;

    /// The shared driver: cancel the τ-divisible defect one weight-order at a time.
    /// Returns the lifted support — the seed if there is nothing to correct, or a
    /// partial lift if a cell outside the report cone does not converge (those are
    /// never read by the report cohomology).
    fn lift_cell(&self, g: Gen) -> BTreeSet<usize> {
        let (def_mod, def_t) = self.defect_module(g);
        let def_dim = def_mod.dimension(def_t);
        let w_k = self.source_weight(g);

        // Maintain the defect incrementally: it is mod-2 linear in the support, so we
        // keep a running parity vector and XOR in only each term we toggle.
        let mut support = self.seed(g);
        let mut parity = FpVector::new(TWO, def_dim);
        self.seed_constant(g, &mut parity);
        for &bidx in support.iter() {
            self.accumulate(g, bidx, &mut parity);
        }

        for _ in 0..256 {
            // The lowest τ-order among the surviving error terms.
            let Some(min_power) = parity
                .iter_nonzero()
                .map(|(fidx, _)| self.defect_weight(g, fidx) - w_k)
                .min()
            else {
                return support; // defect fully cancelled
            };
            assert!(
                min_power >= 1,
                "mod-τ defect ≠ 0 at (s={}, t={}, idx={}) — the model is not a complex",
                g.s, g.t, g.idx
            );

            // The error at that lowest τ-order, as a defect-module vector.
            let mut e = FpVector::new(TWO, def_dim);
            for (fidx, _) in parity.iter_nonzero() {
                if self.defect_weight(g, fidx) - w_k == min_power {
                    e.set_entry(fidx, 1);
                }
            }

            // Solve d̄(c) = e and toggle c into the support, updating the running
            // parity in lockstep. Each correction term is forced (by weight) to
            // τ-power min_power, cancelling this order.
            let Some(c) = self.solve(g, &e) else {
                return support; // out of range: leave the partial lift
            };
            for (idx, v) in c.iter_nonzero() {
                if v == 0 {
                    continue;
                }
                if !support.insert(idx) {
                    support.remove(&idx);
                }
                self.accumulate(g, idx, &mut parity);
            }
        }
        // Non-convergence within the cap: only outside the report cone (report-cone
        // cells converge). Never read by the report cohomology, so leave the partial.
        tracing::debug!(
            "motivic lift did not converge at (s={}, t={}, idx={}); leaving partial (outside report cone)",
            g.s, g.t, g.idx
        );
        support
    }
}

/// The differential-lift instance of [`TauLift`]: lift `d_s(g)` so that
/// `d_{s-1} d_s(g) = 0` over `A_C`. Output module `F_{s-1}`, defect module
/// `F_{s-2}`, defect the composite `d_{s-1} d_s(g)` (accumulated by
/// [`MotivicResolution::accumulate_term`]); `d̄_{s-1}`'s quasi-inverse solves the
/// corrections.
struct DifferentialCells<'a>(&'a MotivicResolution);

impl TauLift for DifferentialCells<'_> {
    fn source_weight(&self, g: Gen) -> i32 {
        self.0.weights[&g]
    }

    fn seed(&self, g: Gen) -> BTreeSet<usize> {
        self.0
            .resolution
            .differential(g.s)
            .output(g.t, g.idx)
            .iter_nonzero()
            .filter(|(_, v)| *v != 0)
            .map(|(i, _)| i)
            .collect()
    }

    fn defect_module(&self, g: Gen) -> (Arc<FreeModule<CTauAlgebra>>, i32) {
        (self.0.module(g.s - 2), g.t)
    }

    fn defect_weight(&self, g: Gen, bidx: usize) -> i32 {
        self.0.entry_weight(g.s - 2, g.t, bidx)
    }

    fn accumulate(&self, g: Gen, bidx: usize, parity: &mut FpVector) {
        let f_sm1 = self.0.module(g.s - 1);
        let f_sm2 = self.0.module(g.s - 2);
        self.0
            .accumulate_term(g.s, g.t, &f_sm1, &f_sm2, self.0.algebra.engine(), bidx, parity);
    }

    fn solve(&self, g: Gen, e: &FpVector) -> Option<FpVector> {
        let d_prev = self.0.resolution.differential(g.s - 1);
        let qi = d_prev.quasi_inverse(g.t)?;
        let mut c = FpVector::new(TWO, self.0.module(g.s - 1).dimension(g.t));
        qi.apply(c.as_slice_mut(), 1, e.as_slice());
        Some(c)
    }
}

/// Minimal $\mathbb{F}_2[\tau]$ polynomial arithmetic and Smith normal form —
/// enough to extract the invariant factors of a small δ matrix, so the
/// $\tau$-torsion of the motivic $E_2$ can be read as an $\mathbb{F}_2[\tau]$-module
/// (see [`MotivicResolution::tau_module`]).
///
/// Polynomials are packed into a `u128` bitmask: bit `i` is the $\mathbb{F}_2$
/// coefficient of $\tau^i$. δ matrices here are tiny and their entries low-degree,
/// so 128 exponent bits are ample.
mod f2tau {
    /// Degree of `a` (`-1` for the zero polynomial).
    pub fn deg(a: u128) -> i32 {
        if a == 0 {
            -1
        } else {
            127 - a.leading_zeros() as i32
        }
    }

    /// Polynomial product over $\mathbb{F}_2$ (carryless multiply).
    fn mul(mut a: u128, b: u128) -> u128 {
        let mut r = 0;
        let mut shift = 0;
        while a != 0 {
            if a & 1 == 1 {
                r ^= b << shift;
            }
            a >>= 1;
            shift += 1;
        }
        r
    }

    /// Quotient $\lfloor a / b \rfloor$ over $\mathbb{F}_2[\tau]$ (the remainder is
    /// dropped). `b` must be nonzero.
    fn div(a: u128, b: u128) -> u128 {
        let db = deg(b);
        let (mut q, mut rem) = (0u128, a);
        while deg(rem) >= db {
            let sh = deg(rem) - db;
            q ^= 1u128 << sh;
            rem ^= b << sh;
        }
        q
    }

    /// The non-unit invariant factors (degree $\ge 1$) of `m` over
    /// $\mathbb{F}_2[\tau]$, by Smith normal form. `m` is consumed.
    ///
    /// Standard Euclidean SNF: pivot on the minimum-degree nonzero entry, clear its
    /// row and column by division, and pull any lower-degree residual back into the
    /// pivot until the pivot divides the remaining block; then recurse on the
    /// complementary submatrix.
    // The row/column clears index two rows (or columns) at once — `m[i][j]` against
    // `m[r0][j]` — so index loops are clearer than split-borrow iterator gymnastics.
    #[allow(clippy::needless_range_loop)]
    pub fn invariant_factors(mut m: Vec<Vec<u128>>) -> Vec<u128> {
        let rows = m.len();
        let cols = m.first().map_or(0, Vec::len);
        let mut factors = Vec::new();
        let (mut r0, mut c0) = (0, 0);
        while r0 < rows && c0 < cols {
            // Pivot = minimum-degree nonzero entry of the active submatrix.
            let mut piv: Option<(usize, usize)> = None;
            for i in r0..rows {
                for j in c0..cols {
                    if m[i][j] != 0
                        && piv.is_none_or(|(pi, pj)| deg(m[i][j]) < deg(m[pi][pj]))
                    {
                        piv = Some((i, j));
                    }
                }
            }
            let Some((pi, pj)) = piv else { break };
            m.swap(r0, pi);
            for row in &mut m {
                row.swap(c0, pj);
            }

            loop {
                let mut changed = false;
                let p = m[r0][c0];
                for i in 0..rows {
                    if i != r0 && m[i][c0] != 0 {
                        let q = div(m[i][c0], p);
                        if q != 0 {
                            for j in 0..cols {
                                m[i][j] ^= mul(q, m[r0][j]);
                            }
                            changed = true;
                        }
                    }
                }
                let p = m[r0][c0];
                for j in 0..cols {
                    if j != c0 && m[r0][j] != 0 {
                        let q = div(m[r0][j], p);
                        if q != 0 {
                            for i in 0..rows {
                                m[i][j] ^= mul(q, m[i][c0]);
                            }
                            changed = true;
                        }
                    }
                }
                // A nonzero residual left in the pivot row/column (degree below the
                // pivot): swap it in and keep reducing.
                let mut resid = None;
                for i in r0 + 1..rows {
                    if m[i][c0] != 0 {
                        resid = Some((i, c0));
                    }
                }
                for j in c0 + 1..cols {
                    if m[r0][j] != 0 {
                        resid = Some((r0, j));
                    }
                }
                if let Some((i, j)) = resid {
                    m.swap(r0, i);
                    if j != c0 {
                        for row in &mut m {
                            row.swap(c0, j);
                        }
                    }
                    changed = true;
                }
                if !changed {
                    break;
                }
            }

            if deg(m[r0][c0]) >= 1 {
                factors.push(m[r0][c0]);
            }
            r0 += 1;
            c0 += 1;
        }
        factors
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
    fn deformation_products_h0_tower() {
        // Products wired into the Sseq: multiplication by h₀ (from ExtAlgebra on the
        // mod-τ resolution) climbs the h₀-tower via Sseq::multiply. h₀ ∈ (0,1,0),
        // and h₀ⁿ ∈ (0,n,0) stays nonzero (Cτ has an infinite h₀-tower), on E₁ and
        // hence on every page.
        let res = MotivicResolution::new(Bidegree::n_s(8, 5));
        let sseq = res.deformation_sseq();
        let prods = res.deformation_products(&[(Bidegree::n_s(0, 1), 0)]);
        let h0 = &prods[0];

        let mut v = FpVector::new(TWO, 1);
        v.set_entry(0, 1);
        let mut class = MultiDegreeElement::new(MultiDegree::from([0, 1, 0]), v);
        for n in 2..=4 {
            class = sseq.multiply(&class, h0).expect("h₀ product in range");
            assert_eq!(class.degree(), MultiDegree::from([0, n, 0]), "h₀^{n} degree");
            assert!(!class.vec().is_zero(), "h₀^{n} should be nonzero");
        }
    }

    #[test]
    fn motivic_ctau_ring_relations() {
        // The Cτ ring (= algebraic Novikov E₂ = E₁ of the deformation SS) acting
        // through Sseq::multiply, checked against the standard Adams-E₂ relations.
        // hᵢ lives at (2ⁱ−1, 1) with weight −(2ⁱ−1) (this presentation's sign).
        let res = MotivicResolution::new(Bidegree::n_s(16, 10));
        let sseq = res.deformation_sseq();

        let h = [(0, "h0"), (1, "h1"), (3, "h2"), (7, "h3")];
        let prods =
            res.deformation_products(&h.iter().map(|&(n, _)| (Bidegree::n_s(n, 1), 0)).collect::<Vec<_>>());
        let wt = |n: i32| res.generator_weight(Gen { s: 1, t: n + 1, idx: 0 });
        let class_at = |n: i32, s: i32, w: i32| -> MultiDegreeElement<3> {
            let deg = MultiDegree::from([n, s, w]);
            let mut v = FpVector::new(TWO, sseq.get_dimension(deg).unwrap_or(0).max(1));
            v.set_entry(0, 1);
            MultiDegreeElement::new(deg, v)
        };
        // A product landing in an empty (undefined) bidegree is zero: Sseq::multiply
        // returns None exactly then, so treat None as the zero class.
        let is_zero = |c: &Option<MultiDegreeElement<3>>| {
            c.as_ref().is_none_or(|x| x.vec().is_zero())
        };

        // h₁-tower: h₁ⁿ ≠ 0 for all n (into the τ-torsion range n ≥ 4), at (n, n, −n).
        let mut c = Some(class_at(1, 1, wt(1)));
        for n in 2..=6 {
            c = c.and_then(|x| sseq.multiply(&x, &prods[1]));
            let x = c.as_ref().unwrap_or_else(|| panic!("h₁^{n} fell out of range"));
            assert_eq!(x.degree(), MultiDegree::from([n, n, -n]), "h₁^{n} degree");
            assert!(!x.vec().is_zero(), "h₁^{n} should be nonzero (Cτ h₁-tower)");
        }

        // Adjacent Hopf elements multiply to zero: h₀h₁ = h₁h₂ = h₂h₃ = 0.
        for i in 0..3 {
            let (n, _) = h[i];
            let prod = sseq.multiply(&class_at(n, 1, wt(n)), &prods[i + 1]);
            assert!(is_zero(&prod), "{}·{} should vanish", h[i].1, h[i + 1].1);
        }

        // The motivic relation h₀²h₂ = τ·h₁³ shows up here as a *hidden τ-extension*:
        // h₁³ ≠ 0 at weight −3, while h₀²h₂ vanishes in the Cτ ring (τ = 0) — it would
        // land at (3, 3, −2), one weight up, empty on E₁. Recovering the τ requires
        // the F₂[τ] product lift; the Cτ ring only sees the τ = 0 shadow.
        let h1_cubed = {
            let mut c = Some(class_at(1, 1, wt(1)));
            for _ in 0..2 {
                c = c.and_then(|x| sseq.multiply(&x, &prods[1]));
            }
            c.expect("h₁³ in range")
        };
        assert_eq!(h1_cubed.degree(), MultiDegree::from([3, 3, -3]));
        assert!(!h1_cubed.vec().is_zero(), "h₁³ ≠ 0");
        let h0sq_h2 = {
            let mut c = Some(class_at(3, 1, wt(3)));
            for _ in 0..2 {
                c = c.and_then(|x| sseq.multiply(&x, &prods[0]));
            }
            c
        };
        assert!(is_zero(&h0sq_h2), "h₀²h₂ vanishes in the Cτ ring (the τ is hidden)");
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
                // Cross-check the Sseq E_∞ against the independent ExtAlgebra path.
                let want = res
                    .ext()
                    .cohomology_dimension(Bidegree::n_s(n, s))
                    .unwrap_or(0);
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
        // does not. The cleanest witness is the h₁-tower: h₁ⁿ = (h₁)ⁿ ∈ Ext^{n,2n}
        // is nonzero for all n motivically, but classically h₁⁴ = 0. So at
        // (s,t)=(4,8) the free (classical) rank is 0, yet h₁⁴ itself is a τ-torsion
        // class there — δ(h₁⁴) = τ·y, so τ·[h₁⁴] = 0: an F₂[τ]/τ summand at (4,4).
        let max = Bidegree::n_s(8, 5);
        let res = MotivicResolution::new(max);

        let m = res.tau_module(4, 8);
        assert_eq!(m.free, 0, "classical h₁⁴ should vanish (free rank 0)");
        assert_eq!(
            m.torsion,
            vec![1],
            "h₁⁴ at (s=4, t=8) should be a single F₂[τ]/τ torsion summand"
        );
        assert!(res.has_tau_torsion(4, 8));

        // Sanity: h₁³ (s=3, t=6) is already nonzero classically, so it is free —
        // not flagged as (extra) torsion beyond its classical class.
        let m3 = res.tau_module(3, 6);
        assert_eq!(m3.free, 1, "classical h₁³ should survive");
        assert!(m3.torsion.is_empty(), "h₁³ is free, not τ-torsion");
    }

    #[test]
    fn tau_module_h1_tower_is_free_then_tau_torsion() {
        // The whole h₁-tower h₁ⁿ ∈ Ext^{n,2n} (stem n, filtration n): free (a genuine
        // classical class) for n ≤ 3, then a single F₂[τ]/τ summand for n ≥ 4 — the
        // motivic τ-torsion the classical page cannot see.
        let res = MotivicResolution::new(Bidegree::n_s(10, 9));
        for n in 1..=3 {
            let m = res.tau_module(n, 2 * n);
            assert_eq!((m.free, m.torsion.as_slice()), (1, [].as_slice()), "h₁^{n} free");
        }
        for n in 4..=8 {
            let m = res.tau_module(n, 2 * n);
            assert_eq!((m.free, m.torsion.as_slice()), (0, [1].as_slice()), "h₁^{n} = F₂[τ]/τ");
        }
    }

    #[test]
    fn tau_module_torsion_matches_deformation_sseq_sources() {
        // The F₂[τ]-module torsion (SNF of the outgoing δ) is exactly the data the
        // deformation SS carries: a class at (n, s) is F₂[τ]/τ^r-torsion iff it
        // supports a d_r there. Cross-check the local SNF reading against the SS's
        // differential sources over the interior (away from the report edge, where
        // the finite compute box makes the SS over-count).
        let res = MotivicResolution::new(Bidegree::n_s(16, 11));
        let sseq = res.deformation_sseq();

        let mut sources: HashMap<(i32, i32), Vec<i32>> = HashMap::new();
        for b in sseq.iter_degrees() {
            let [n, s, _] = b.coords();
            let diffs = sseq.differentials(b);
            for r in diffs.min_degree()..diffs.len() {
                for _ in 0..diffs[r].get_source_target_pairs().len() {
                    sources.entry((n, s)).or_default().push(r);
                }
            }
        }

        for n in 0..=12 {
            for s in 1..=10 {
                let mut snf: Vec<i32> =
                    res.tau_module(s, n + s).torsion.iter().map(|&k| k as i32).collect();
                snf.sort_unstable();
                let mut ss = sources.get(&(n, s)).cloned().unwrap_or_default();
                ss.sort_unstable();
                assert_eq!(snf, ss, "torsion vs d_r sources at (n={n}, s={s})");
            }
        }
    }
}
