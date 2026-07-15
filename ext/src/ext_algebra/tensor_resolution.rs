//! Nassau's field-resolution trick: $\Ext_A(M, k)$ from a resolution of $k$.
//!
//! A minimal resolution engine (Nassau's algorithm) resolves a *finite* module directly, and its
//! generators already *are* $\Ext$ — nothing to do. It cannot resolve an *infinite* `M`. Nassau's
//! workaround: resolve the base field `k` (finite, fast) as $P_\bullet \to k$, then tensor with
//! `M`. Since the Steenrod algebra `A` is a Hopf algebra, $P_s \otimes_k M$ is free, so
//! $P_\bullet \otimes M \to M$ is a free — but **non-minimal** — resolution of `M`. Its coboundary
//! is non-zero, so $\Ext_A(M, k)$ is the genuine cohomology of $\Hom_A(P_\bullet \otimes M, k)$,
//! which [`ExtAlgebra`](super::ExtAlgebra) takes via [`ExtDifferential`](super::ExtDifferential).
//!
//! # The coboundary, in closed form
//! We never build the tensor module. Writing $Q(D_s) = (P_s \otimes M)/(\bar A \cdot (P_s\otimes M))$
//! for the indecomposables (the cochain generators of $\Hom_A(P_s\otimes M, k)$), the Hopf
//! "tensor-with-free is free" isomorphism identifies $Q(D_s) \cong \bigoplus_i M[d_i]$, where the
//! $x_i$ are the generators of the free module $P_s$ in internal degrees $d_i$. So a cochain
//! generator at internal degree `t` is a pair $(i, \alpha)$ — a $P_s$-generator $x_i$ and an
//! `M`-basis element $m_\alpha$ with $d_i + |m_\alpha| = t$ — and
//! $$ \dim \Hom_A(P_s\otimes M, k)_t = \sum_i \dim M_{t - d_i}. $$
//! The coboundary $\delta\colon C^s \to C^{s+1}$, dual to $Q(\partial)$, has the closed form
//! $$ \delta_{(i,\alpha),(l,\gamma)} = [m_\alpha]\bigl(\chi(a_{li}) \cdot m_\gamma\bigr), $$
//! where $a_{li} \in A$ is the component of the $k$-resolution differential $d_P(z_l)$ on the
//! generator $x_i$ (an algebra element of degree $|z_l| - d_i$), $\chi$ is the [antipode](Antipode),
//! and the action is on `M`. For a **minimal** $P_\bullet$ every $a_{li} \in \bar A$, so $\chi(a_{li})$
//! is a genuine positive-degree operator and $\delta \neq 0$ — exactly what makes this a non-minimal
//! resolution whose cohomology must be taken.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use algebra::{
    Algebra, Bialgebra,
    module::{
        FreeModule, Module, ZeroModule,
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
    },
    pair_algebra::PairAlgebra,
};
use dashmap::DashMap;
use fp::{
    matrix::{AffineSubspace, AugmentedMatrix, Matrix},
    prime::{Prime, ValidPrime},
    vector::FpVector,
};
use once::{OnceBiVec, OnceVec};
use sseq::coordinates::{Bidegree, BidegreeElement};

use super::{Cochain, ExtAlgebra, ExtDifferential, massey::MasseyResult};
use crate::{
    chain_complex::{AugmentedChainComplex, ChainComplex, FreeChainComplex},
    resolution::secondary::SecondaryResolution,
    resolution_homomorphism::ResolutionHomomorphism,
    secondary::SecondaryLift,
};

/// The Hopf antipode $\chi\colon A \to A$ of a connected graded bialgebra, computed generically and
/// memoised.
///
/// For a connected graded Hopf algebra the antipode is determined by
/// $\sum_{(x)} \chi(x_{(1)}) x_{(2)} = \varepsilon(x)$, which at $p = 2$ unrolls to the recursion
/// $$ \chi(x) = x + \sum_{\text{mid}} \chi(x_{(1)})\, x_{(2)}, $$
/// the sum over coproduct terms with $0 < |x_{(1)}| < |x|$. We only ever need $\chi$ on the
/// [`Bialgebra::decompose`] atoms (via the anti-homomorphism property $\chi(ab) = \chi(b)\chi(a)$),
/// whose coproducts are directly available.
///
/// Currently specialised to $p = 2$ (all Hopf signs are $+$); the differential-entry degrees stay
/// within the resolved range.
pub struct Antipode<A: Bialgebra> {
    algebra: Arc<A>,
    /// `(degree, basis index)` → $\chi$ of that basis element, as a vector over the algebra basis.
    cache: DashMap<(i32, usize), FpVector>,
}

impl<A: Bialgebra> Antipode<A> {
    pub fn new(algebra: Arc<A>) -> Self {
        assert_eq!(
            algebra.prime(),
            ValidPrime::new(2),
            "Antipode is currently implemented only at p = 2"
        );
        Self {
            algebra,
            cache: DashMap::new(),
        }
    }

    fn prime(&self) -> ValidPrime {
        self.algebra.prime()
    }

    /// $\chi$ of the basis element `(degree, idx)`, as a vector over the degree-`degree` algebra
    /// basis. Requires the algebra basis computed through `degree`.
    pub fn apply(&self, degree: i32, idx: usize) -> FpVector {
        if degree == 0 {
            let mut v = FpVector::new(self.prime(), self.algebra.dimension(0));
            v.add_basis_element(idx, 1);
            return v;
        }
        if let Some(v) = self.cache.get(&(degree, idx)) {
            return v.clone();
        }

        // χ(x) = χ(atom_1) · χ(atom_2) · … in acting order, since χ is an anti-homomorphism and
        // `decompose` lists the factors in the order they act (rightmost algebra factor first).
        let atoms = self.algebra.decompose(degree, idx);
        let result = if atoms.len() == 1 {
            // `x` is itself an atom (a single generator whose product is `x`); use the recursion.
            self.apply_atom(degree, idx)
        } else {
            let mut acc = FpVector::new(self.prime(), self.algebra.dimension(0));
            acc.add_basis_element(0, 1);
            let mut acc_deg = 0;
            for &(a_deg, a_idx) in &atoms {
                let chi_atom = self.apply(a_deg, a_idx);
                let out_deg = acc_deg + a_deg;
                let mut next = FpVector::new(self.prime(), self.algebra.dimension(out_deg));
                for (l_idx, l_coeff) in acc.iter_nonzero() {
                    for (r_idx, r_coeff) in chi_atom.iter_nonzero() {
                        self.algebra.multiply_basis_elements(
                            next.as_slice_mut(),
                            l_coeff * r_coeff,
                            acc_deg,
                            l_idx,
                            a_deg,
                            r_idx,
                        );
                    }
                }
                acc = next;
                acc_deg = out_deg;
            }
            acc
        };

        self.cache
            .entry((degree, idx))
            .or_insert(result)
            .value()
            .clone()
    }

    /// $\chi$ of a single [`decompose`](Bialgebra::decompose) atom, via the recursion
    /// $\chi(g) = g + \sum_{\text{mid}} \chi(g_{(1)}) g_{(2)}$ over its coproduct.
    fn apply_atom(&self, degree: i32, idx: usize) -> FpVector {
        let mut result = FpVector::new(self.prime(), self.algebra.dimension(degree));
        result.add_basis_element(idx, 1);
        for (l_deg, l_idx, r_deg, r_idx) in self.algebra.coproduct(degree, idx) {
            // Skip the two boundary terms g ⊗ 1 (l_deg == degree) and 1 ⊗ g (l_deg == 0).
            if l_deg == 0 || l_deg == degree {
                continue;
            }
            let chi_left = self.apply(l_deg, l_idx);
            for (j, coeff) in chi_left.iter_nonzero() {
                self.algebra.multiply_basis_elements(
                    result.as_slice_mut(),
                    coeff,
                    l_deg,
                    j,
                    r_deg,
                    r_idx,
                );
            }
        }
        result
    }

    /// $\chi$ of a general algebra element `elt` of degree `degree` (linear extension).
    fn apply_element(&self, degree: i32, elt: FpVector) -> FpVector {
        let mut out = FpVector::new(self.prime(), self.algebra.dimension(degree));
        for (idx, coeff) in elt.iter_nonzero() {
            out.as_slice_mut()
                .add(self.apply(degree, idx).as_slice(), coeff);
        }
        out
    }
}

/// The full coproduct $\Delta(x) = \sum_{(x)} x_{(1)} \otimes x_{(2)}$ of a basis element, memoised.
///
/// At $p = 2$ every term has coefficient $1$, so a coproduct is a *set* of basis-element pairs
/// $(|x_{(1)}|, x_{(1)}, |x_{(2)}|, x_{(2)})$. It is obtained by folding the coproducts of the
/// [`decompose`](Bialgebra::decompose) atoms in $A \otimes A$ — the same route the [`Antipode`]
/// uses, but keeping *all* terms (including the two boundary terms $x \otimes 1$ and $1 \otimes x$).
pub(crate) struct FullCoproduct<A: Bialgebra> {
    algebra: Arc<A>,
    cache: DashMap<(i32, usize), Arc<Vec<(i32, usize, i32, usize)>>>,
}

impl<A: Bialgebra> FullCoproduct<A> {
    pub(crate) fn new(algebra: Arc<A>) -> Self {
        Self {
            algebra,
            cache: DashMap::new(),
        }
    }

    /// $\Delta$ of the basis element `(degree, idx)`, as the list of surviving pairs
    /// `(|a'|, a', |a''|, a'')`. Requires the algebra basis computed through `degree`.
    pub(crate) fn terms(&self, degree: i32, idx: usize) -> Arc<Vec<(i32, usize, i32, usize)>> {
        if degree == 0 {
            // Δ(1) = 1 ⊗ 1 (the unit is grouplike).
            return Arc::new(vec![(0, 0, 0, 0)]);
        }
        if let Some(v) = self.cache.get(&(degree, idx)) {
            return Arc::clone(&v);
        }

        let p = self.algebra.prime();
        // Δ as a map (l_deg, l_idx, r_deg, r_idx) → coeff, seeded with the unit 1 ⊗ 1.
        let mut terms: HashMap<(i32, usize, i32, usize), u32> = HashMap::new();
        terms.insert((0, 0, 0, 0), 1);
        for (a_deg, a_idx) in self.algebra.decompose(degree, idx) {
            let mut next: HashMap<(i32, usize, i32, usize), u32> = HashMap::new();
            for (&(ll, li, rl, ri), &c) in &terms {
                for (cl, cli, cr, cri) in self.algebra.coproduct(a_deg, a_idx) {
                    // (ll ⊗ rl) · (cl ⊗ cr) = (ll · cl) ⊗ (rl · cr).
                    let mut left = FpVector::new(p, self.algebra.dimension(ll + cl));
                    self.algebra
                        .multiply_basis_elements(left.as_slice_mut(), 1, ll, li, cl, cli);
                    let mut right = FpVector::new(p, self.algebra.dimension(rl + cr));
                    self.algebra
                        .multiply_basis_elements(right.as_slice_mut(), 1, rl, ri, cr, cri);
                    for (lj, lc) in left.iter_nonzero() {
                        for (rj, rc) in right.iter_nonzero() {
                            *next.entry((ll + cl, lj, rl + cr, rj)).or_insert(0) += c * lc * rc;
                        }
                    }
                }
            }
            terms = next;
        }
        let result: Vec<(i32, usize, i32, usize)> = terms
            .into_iter()
            .filter(|&(_, c)| c % p.as_u32() != 0)
            .map(|(k, _)| k)
            .collect();
        Arc::clone(
            self.cache
                .entry((degree, idx))
                .or_insert_with(|| Arc::new(result))
                .value(),
        )
    }
}

/// The dualised differential $\Hom(d, k)$ of the field-resolution complex $P_\bullet \otimes M$,
/// as an [`ExtDifferential`] over the backing $k$-resolution `CC`.
///
/// Attach it to an [`ExtAlgebra`] over the $k$-resolution to compute $\Ext_A(M, k)$ — see
/// [`field_resolution_ext`] and the [module docs](self).
pub struct TensorResolutionDifferential<CC, N>
where
    CC: FreeChainComplex,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra>,
{
    /// $P_\bullet$: a (minimal) free resolution of `k`.
    resolution: Arc<CC>,
    /// The module `M` being resolved.
    module: Arc<N>,
    antipode: Antipode<CC::Algebra>,
    /// `(s, t)` → the ordered cochain basis of $\Hom_A(P_s \otimes M, k)_t$: pairs
    /// `(generator degree d_i, generator index i, M-basis index α)`. Cached so that `matrix(b)`'s
    /// rows and `matrix(source)`'s columns share one coordinate system.
    cochain_bases: DashMap<(i32, i32), Arc<Vec<(i32, usize, usize)>>>,
}

impl<CC, N> TensorResolutionDifferential<CC, N>
where
    CC: FreeChainComplex,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra>,
{
    pub fn new(resolution: Arc<CC>, module: Arc<N>) -> Self {
        let antipode = Antipode::new(resolution.algebra());
        Self {
            resolution,
            module,
            antipode,
            cochain_bases: DashMap::new(),
        }
    }

    fn prime(&self) -> ValidPrime {
        self.resolution.prime()
    }

    /// True once $P_s$ is resolved through internal degree `t` (so its generators up to `t` and the
    /// differential out of it are available).
    fn computed(&self, s: i32, t: i32) -> bool {
        s >= 0 && self.resolution.has_computed_bidegree(Bidegree::s_t(s, t))
    }

    /// The ordered cochain basis of $\Hom_A(P_s \otimes M, k)_t$ (see [`Self::cochain_bases`]).
    fn cochain_basis(&self, s: i32, t: i32) -> Arc<Vec<(i32, usize, usize)>> {
        if let Some(b) = self.cochain_bases.get(&(s, t)) {
            return Arc::clone(&b);
        }
        let module_s = self.resolution.module(s);
        let mut basis = Vec::new();
        // Generators of P_s in increasing degree; then M-basis in the complementary degree.
        for (d_i, i) in module_s.iter_gens(t) {
            let m_dim = self.module.dimension(t - d_i);
            for alpha in 0..m_dim {
                basis.push((d_i, i, alpha));
            }
        }
        Arc::clone(
            self.cochain_bases
                .entry((s, t))
                .or_insert_with(|| Arc::new(basis))
                .value(),
        )
    }

    /// The closed-form matrix of an $A$-linear "untwisted" cochain operator $\Hom_A(Q_{\mathrm{tgt}},
    /// k) \to \Hom_A(Q_{\mathrm{src}}, k)$ read off a map on the backing free resolution `P`.
    ///
    /// Rows index the source cochain basis $(d_i, i, \alpha)$ at $(\text{src\_s}, \text{src\_t})$,
    /// columns the target cochain basis $(e_l, l, \gamma)$ at $(\text{tgt\_s}, \text{tgt\_t})$.
    /// `get_image(e_l, l)` gives the image of the target generator $z_l$ as an element of
    /// $P_{\text{src\_s}}$ at internal degree $e_l - (\text{tgt\_t} - \text{src\_t})$ (for the
    /// differential this is $d_P(z_l)$; for the cup product by a class it is $f_x(z_l)$). The entry
    /// is $[m_\alpha]\bigl(\chi(a_{li}) \cdot m_\gamma\bigr)$, where $a_{li}$ is the component of that
    /// image on the generator $(d_i, i)$ — the same untwisting the free differential uses, so only
    /// the augmentation ($a' = 1$) part survives in $\Hom_A(-, k)$.
    fn closed_form_matrix(
        &self,
        src_s: i32,
        src_t: i32,
        tgt_s: i32,
        tgt_t: i32,
        get_image: impl Fn(i32, usize) -> FpVector,
    ) -> Matrix {
        let p = self.prime();
        let algebra = self.resolution.algebra();
        let rows = self.cochain_basis(src_s, src_t); // (d_i, i, α)  — Q(D_src)
        let cols = self.cochain_basis(tgt_s, tgt_t); // (e_l, l, γ) — Q(D_tgt)
        let mut matrix = Matrix::new(p, rows.len(), cols.len());

        let module_src = self.resolution.module(src_s);
        let internal_shift = tgt_t - src_t;

        // Row lookup: (generator degree, generator index) → its first row (α = 0). The α rows for a
        // generator are contiguous by construction of `cochain_basis`.
        let mut row_base: HashMap<(i32, usize), usize> = HashMap::new();
        for (r, &(d_i, i, alpha)) in rows.iter().enumerate() {
            if alpha == 0 {
                row_base.insert((d_i, i), r);
            }
        }

        for (c, &(e_l, l, gamma)) in cols.iter().enumerate() {
            let img_deg = e_l - internal_shift; // degree of the image in P_{src_s}
            if img_deg < 0 {
                continue;
            }
            let img = get_image(e_l, l); // image of z_l ∈ (P_{src_s})_{img_deg}
            if img.is_zero() {
                continue;
            }
            let gamma_deg = tgt_t - e_l; // |m_γ|

            // For each P_{src_s}-generator (d_i, i): extract a_{li}, act χ(a_{li}) on m_γ, scatter.
            for (&(d_i, i), &base) in &row_base {
                let op_deg = img_deg - d_i;
                if op_deg < 0 {
                    continue;
                }
                let width = algebra.dimension(op_deg);
                if width == 0 {
                    continue;
                }
                // a_{li} = block of the image at generator (d_i, i), as an algebra element.
                let offset = module_src.generator_offset(img_deg, d_i, i);
                let mut a_li = FpVector::new(p, width);
                for q in 0..width {
                    let v = img.entry(offset + q);
                    if v != 0 {
                        a_li.add_basis_element(q, v);
                    }
                }
                if a_li.is_zero() {
                    continue;
                }
                // χ(a_{li}) · m_γ  ∈ M_{src_t - d_i}.
                let chi = self.antipode.apply_element(op_deg, a_li);
                let mut acted = FpVector::new(p, self.module.dimension(src_t - d_i));
                for (op_idx, coeff) in chi.iter_nonzero() {
                    self.module.act_on_basis(
                        acted.as_slice_mut(),
                        coeff,
                        op_deg,
                        op_idx,
                        gamma_deg,
                        gamma,
                    );
                }
                if acted.is_zero() {
                    continue;
                }
                // Entry for each (i, α): coefficient of m_α.
                for alpha in 0..acted.len() {
                    let v = acted.entry(alpha);
                    if v != 0 {
                        matrix.row_mut(base + alpha).add_basis_element(c, v);
                    }
                }
            }
        }
        matrix
    }

    /// The matrix of the cup product by `x ∈ Ext(k, k)` on cochains: $\Hom_A(Q_{s+s_x}, k)_{t+t_x}
    /// \xleftarrow{\ x \cup -\ } \Hom_A(Q_s, k)_t$ (rows = source at `(src_s, src_t)`, columns =
    /// target at `(src_s + x_shift.s, src_t + x_shift.t)`).
    ///
    /// `f_x` is the chain self-map of `P` realising `x` (built by
    /// [`from_class`](ResolutionHomomorphism::from_class) in the unit and extended through the target
    /// filtration). The cup product reads it through the same untwisting as
    /// [`closed_form_matrix`](Self::closed_form_matrix): the module action `a ∪ v = v ∘ (f_x ⊗ id)`
    /// keeps only the augmentation part, giving entry `[m_β](χ(b_li) · m_γ)` with `b_li` the
    /// component of `f_x(z_l)` on the generator `x_i`.
    pub(crate) fn cup_matrix(
        &self,
        f_x: &ResolutionHomomorphism<CC, CC>,
        src_s: i32,
        src_t: i32,
        x_shift: Bidegree,
    ) -> Matrix {
        let tgt_s = src_s + x_shift.s();
        let tgt_t = src_t + x_shift.t();
        let map = f_x.get_map(tgt_s);
        self.closed_form_matrix(src_s, src_t, tgt_s, tgt_t, move |e_l, l| {
            map.output(e_l, l).to_owned()
        })
    }
}

impl<CC, N> ExtDifferential for TensorResolutionDifferential<CC, N>
where
    CC: FreeChainComplex,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra>,
{
    fn shift(&self) -> Bidegree {
        // δ: Ext^{s,t} → Ext^{s+1,t}, i.e. (n, s) → (n-1, s+1) with t fixed.
        Bidegree::n_s(-1, 1)
    }

    fn dimension(&self, b: Bidegree) -> Option<usize> {
        let (s, t) = (b.s(), b.t());
        if !self.computed(s, t) {
            return None;
        }
        Some(self.cochain_basis(s, t).len())
    }

    fn matrix(&self, b: Bidegree) -> Option<Matrix> {
        let (s, t) = (b.s(), b.t());
        // Need P_s (rows / a_{li}) and P_{s+1} (columns / d_P out of s+1) through degree t.
        if !self.computed(s, t) || !self.computed(s + 1, t) {
            return None;
        }
        // δ: C^s → C^{s+1} at the same internal degree, read off the free differential d_P.
        let p = self.prime();
        let module_s = self.resolution.module(s);
        let d_p = self.resolution.differential(s + 1); // P_{s+1} → P_s
        Some(self.closed_form_matrix(s, t, s + 1, t, |e_l, l| {
            let mut dp = FpVector::new(p, module_s.dimension(e_l));
            d_p.apply_to_generator(&mut dp, 1, e_l, l);
            dp
        }))
    }
}

/// Build an [`ExtAlgebra`] over the $k$-resolution whose cohomology is $\Ext_A(M, k)$, computed by
/// Nassau's field-resolution trick (see the [module docs](self)).
///
/// `resolution` must be a free resolution of the base field `k` over the same algebra as `module`.
/// Take cohomology with [`ExtAlgebra::cohomology_dimension`] /
/// [`ExtAlgebra::cohomology_subquotient`].
pub fn field_resolution_ext<CC, N>(resolution: Arc<CC>, module: Arc<N>) -> ExtAlgebra<CC>
where
    CC: FreeChainComplex + 'static,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra> + 'static,
{
    let diff = Arc::new(TensorResolutionDifferential::new(
        Arc::clone(&resolution),
        module,
    ));
    ExtAlgebra::without_unit(resolution).with_differential(diff)
}

/// Build an [`ExtAlgebra`] whose $\Ext(M, k)$ supports **products and Massey products** by the
/// field-resolution trick (see the [module docs](self)).
///
/// Unlike [`field_resolution_ext`] — which only exposes the additive $\Ext$ via the closed-form
/// [`TensorResolutionDifferential`] — this builds the *genuine* free resolution
/// $Q_\bullet = P_\bullet \otimes M$ ([`TensorResolution`]) as the product-carrying `resolution`,
/// and keeps the **minimal** $P_\bullet$ as the separate `unit`. The resulting
/// `ExtAlgebra<TensorResolution<CC, N>, CC>` computes $\Ext(k, k)$-module products of $\Ext(M, k)$
/// classes: product maps have $Q_\bullet$ as (non-minimal) source and the minimal $P_\bullet$ as
/// target, so only `P_\bullet` needs to be an [`AugmentedChainComplex`]. The non-minimality is
/// handled by the sequential extension order ([`with_sequential_source`](ExtAlgebra::with_sequential_source))
/// and the cohomology transport in [`multiply_into`](ExtAlgebra::multiply_into).
///
/// `resolution` must be a (minimal) free resolution of the base field `k` over the same algebra as
/// `module`.
///
/// For **Massey products** on the field trick use [`FieldMassey`], *not* the generic
/// [`ExtAlgebra::massey`] on the returned algebra: the chain-map/null-homotopy bracket the latter
/// uses degenerates on the non-minimal $Q_\bullet$, whereas [`FieldMassey`] reads the bracket from
/// the cochain DGA and stays correct.
pub fn field_resolution_products<CC, N>(
    resolution: Arc<CC>,
    module: Arc<N>,
) -> ExtAlgebra<TensorResolution<CC, N>, CC>
where
    CC: FreeChainComplex + crate::chain_complex::AugmentedChainComplex + 'static,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra> + ZeroModule + 'static,
{
    let q = Arc::new(TensorResolution::new(Arc::clone(&resolution), module));
    let diff = Arc::new(DualizedDifferential::new(Arc::clone(&q)));
    ExtAlgebra::new_with_unit(q, resolution)
        .with_differential(diff)
        .with_sequential_source(true)
}

/// The dualised differential $\Hom_A(\partial, k)$ of *any* free chain complex `Q`, read straight
/// off `Q`'s own differential, as an [`ExtDifferential`].
///
/// Its cohomology is $\Ext$ computed from `Q`: zero coboundary (so $\Ext = $ generators) for a
/// minimal `Q`, and the genuine non-minimal coboundary for a tensored `Q` such as
/// [`TensorResolution`]. Because the cochain basis at each bidegree is exactly `Q`'s generators, an
/// [`ExtAlgebra`] carrying this differential shares one coordinate system with `Q`'s secondary
/// machinery — which is what lets [`FieldResolutionSecondary`] transport the Adams $d_2$ between
/// them by [`lift`](ExtAlgebra::lift)/[`project`](ExtAlgebra::project).
pub struct DualizedDifferential<CC: FreeChainComplex> {
    complex: Arc<CC>,
}

impl<CC: FreeChainComplex> DualizedDifferential<CC> {
    pub fn new(complex: Arc<CC>) -> Self {
        Self { complex }
    }
}

impl<CC: FreeChainComplex> ExtDifferential for DualizedDifferential<CC> {
    fn shift(&self) -> Bidegree {
        // δ: Ext^{s,t} → Ext^{s+1,t}, i.e. (n, s) → (n-1, s+1) with t fixed.
        Bidegree::n_s(-1, 1)
    }

    fn dimension(&self, b: Bidegree) -> Option<usize> {
        if b.s() < 0 || !self.complex.has_computed_bidegree(b) {
            return None;
        }
        Some(self.complex.number_of_gens_in_bidegree(b))
    }

    fn matrix(&self, b: Bidegree) -> Option<Matrix> {
        let (s, t) = (b.s(), b.t());
        let target = b + self.shift(); // (s + 1, t)
        if s < 0
            || !self.complex.has_computed_bidegree(b)
            || !self.complex.has_computed_bidegree(target)
        {
            return None;
        }
        let p = self.complex.prime();
        let rows = self.complex.number_of_gens_in_bidegree(b); // C^s_t
        let cols = self.complex.number_of_gens_in_bidegree(target); // C^{s+1}_t
        let mut matrix = Matrix::new(p, rows, cols);
        // ∂_{s+1}: Q_{s+1} → Q_s; `hom_k(t)` is indexed [Q_s gen][Q_{s+1} gen] = [row][col], with
        // entry = coefficient of Q_s-generator `row` in ∂ of Q_{s+1}-generator `col` — exactly the
        // coboundary δ_{row,col} (and the augmentation part of `TensorResolution`'s free ∂).
        let hk = self.complex.differential(s + 1).hom_k(t);
        for (row, cols_of_row) in hk.iter().enumerate() {
            for (col, &v) in cols_of_row.iter().enumerate() {
                if v != 0 {
                    matrix.row_mut(row).set_entry(col, v);
                }
            }
        }
        Some(matrix)
    }
}

/// A materialised, **genuine** free resolution $Q_\bullet = P_\bullet \otimes M \to M$ of `M`,
/// built by Nassau's field-resolution trick (see the [module docs](self)).
///
/// Unlike [`TensorResolutionDifferential`] — which only produces the *dualised* $\Hom_A(Q, k)$ and
/// so knows just the additive $\Ext$ — this is the whole chain complex, with free modules and free
/// differentials, and therefore is a bona fide [`FreeChainComplex`]. That unlocks everything the
/// secondary/product machinery needs from a resolution: in particular
/// [`SecondaryResolution`](crate::resolution::secondary::SecondaryResolution) accepts it directly,
/// giving the Adams $d_2$ on $\Ext_A(M, k)$ for infinite/tensored `M`.
///
/// # The free differential, in closed form
/// $Q_s = P_s \otimes M$ carries the diagonal $A$-action; the untwisting isomorphism
/// $\Phi\colon P_s \otimes M^{\mathrm{triv}} \xrightarrow{\ \sim\ } P_s \otimes M$,
/// $b\,x_i \otimes m \mapsto \sum b_{(1)} x_i \otimes b_{(2)} m$, presents it as *free* on the pairs
/// $x_i \otimes m_\alpha$. Conjugating $\partial_P \otimes \mathrm{id}$ by $\Phi$ (its inverse is
/// $\Psi(b\,y \otimes m) = \sum b_{(1)} y \otimes \chi(b_{(2)}) m$) gives the free differential
/// $$ \partial_Q(x_i \otimes m_\alpha)
///      = \sum_j \sum_{(a_{ij})} a'_{ij} \cdot \bigl(y_j \otimes \chi(a''_{ij})\, m_\alpha\bigr), $$
/// where $\partial_P(x_i) = \sum_j a_{ij} y_j$ and $\Delta(a_{ij}) = \sum a'_{ij} \otimes a''_{ij}$
/// is the [full coproduct](FullCoproduct). Its $\Hom_A(-, k)$ keeps only the augmentation part
/// $a' = 1$, recovering the closed-form [`TensorResolutionDifferential`]; and it squares to zero by
/// construction, being conjugate to $\partial_P \otimes \mathrm{id}$.
///
/// Generators of $Q_s$ at internal degree `t` are added in the same $(x_i, m_\alpha)$ order as
/// [`TensorResolutionDifferential::cochain_basis`], so the two share one coordinate system.
pub struct TensorResolution<CC, N>
where
    CC: FreeChainComplex,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra> + ZeroModule,
{
    /// $P_\bullet$: a (minimal) free resolution of `k`.
    resolution: Arc<CC>,
    /// The module `M` being resolved.
    module: Arc<N>,
    antipode: Antipode<CC::Algebra>,
    coproduct: FullCoproduct<CC::Algebra>,
    zero_module: Arc<FreeModule<CC::Algebra>>,
    /// `s` → $Q_s = P_s \otimes M$.
    modules: OnceBiVec<Arc<FreeModule<CC::Algebra>>>,
    /// `s` → $\partial_s\colon Q_s \to Q_{s-1}$ (with $\partial_0\colon Q_0 \to 0$).
    differentials: OnceVec<Arc<FreeModuleHomomorphism<FreeModule<CC::Algebra>>>>,
    lock: Mutex<()>,
}

impl<CC, N> TensorResolution<CC, N>
where
    CC: FreeChainComplex,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra> + ZeroModule,
{
    pub fn new(resolution: Arc<CC>, module: Arc<N>) -> Self {
        let algebra = resolution.algebra();
        let antipode = Antipode::new(Arc::clone(&algebra));
        let coproduct = FullCoproduct::new(Arc::clone(&algebra));
        let zero_module = Arc::new(FreeModule::new(Arc::clone(&algebra), "0".to_string(), 0));
        Self {
            resolution,
            module,
            antipode,
            coproduct,
            zero_module,
            modules: OnceBiVec::new(0),
            differentials: OnceVec::new(),
            lock: Mutex::new(()),
        }
    }

    /// Build $Q_s$ (free modules and differentials) for all `s ≤ s_max` through internal degree
    /// `t_max`. Idempotent and monotone: re-calling with a larger box extends in place.
    fn extend(&self, s_max: i32, t_max: i32) {
        if s_max < 0 || t_max < 0 {
            return;
        }
        let _lock = self.lock.lock().unwrap();
        let algebra = self.resolution.algebra();

        // Grow the substrate first: P_• through (s_max, t_max); M and the algebra through t_max.
        self.resolution
            .compute_through_bidegree(Bidegree::s_t(s_max, t_max));
        self.module.compute_basis(t_max);
        algebra.compute_basis(t_max);
        self.zero_module.compute_basis(t_max);

        // Ensure a free module exists for each 0..=s_max, then grow its generators + basis to t_max.
        for s in self.modules.len()..=s_max {
            self.modules.push(Arc::new(FreeModule::new(
                Arc::clone(&algebra),
                format!("(P⊗M)_{s}"),
                0,
            )));
        }
        for s in 0..=s_max {
            let fm = &self.modules[s];
            fm.compute_basis(t_max);
            let p_s = self.resolution.module(s);
            for d in (fm.max_computed_degree() + 1)..=t_max {
                // Generators of Q_s at degree d: the (i, α) with |x_i| + |m_α| = d.
                let count: usize = p_s
                    .iter_gens(d)
                    .map(|(d_i, _)| self.module.dimension(d - d_i))
                    .sum();
                fm.add_generators(d, count, None);
            }
        }

        // Ensure the differentials exist, then grow their outputs to t_max.
        for s in self.differentials.len() as i32..=s_max {
            let d = if s == 0 {
                FreeModuleHomomorphism::new(
                    Arc::clone(&self.modules[0]),
                    Arc::clone(&self.zero_module),
                    0,
                )
            } else {
                FreeModuleHomomorphism::new(
                    Arc::clone(&self.modules[s]),
                    Arc::clone(&self.modules[s - 1]),
                    0,
                )
            };
            self.differentials.push(Arc::new(d));
        }
        for s in 0..=s_max {
            let d = &self.differentials[s as usize];
            if s == 0 {
                d.extend_by_zero(t_max);
            } else {
                for degree in d.next_degree()..=t_max {
                    let rows = self.differential_rows(s, degree);
                    d.add_generators_from_rows(degree, rows);
                }
            }
        }

        // The secondary machinery lifts intermediates through the differentials, so each needs its
        // quasi-inverse (image/kernel/QI). Minimal resolvers build these while resolving; here we
        // compute them explicitly for the materialised free differentials.
        for s in 0..=s_max {
            self.differentials[s as usize].compute_auxiliary_data_through_degree(t_max);
        }
    }

    /// The outputs of $\partial_s\colon Q_s \to Q_{s-1}$ on the generators of $Q_s$ at degree
    /// `degree`, one [`FpVector`] (in $(Q_{s-1})_{\text{degree}}$) per generator, in generator order.
    fn differential_rows(&self, s: i32, degree: i32) -> Vec<FpVector> {
        let p = self.resolution.prime();
        let algebra = self.resolution.algebra();
        let fm = &self.modules[s];
        let target = &self.modules[s - 1];
        let p_s = self.resolution.module(s);
        let p_prev = self.resolution.module(s - 1);
        let d_p = self.resolution.differential(s);
        let tgt_dim = target.dimension(degree);

        let mut rows: Vec<FpVector> = Vec::with_capacity(fm.number_of_gens_in_degree(degree));
        for (d_i, i) in p_s.iter_gens(degree) {
            let e_alpha = degree - d_i;
            let m_dim = self.module.dimension(e_alpha);
            if m_dim == 0 {
                continue;
            }
            // ∂_P(x_i) ∈ (P_{s-1})_{d_i}.
            let mut dp = FpVector::new(p, p_prev.dimension(d_i));
            d_p.apply_to_generator(&mut dp, 1, d_i, i);

            for alpha in 0..m_dim {
                let mut out = FpVector::new(p, tgt_dim);
                if !dp.is_zero() {
                    // For each P_{s-1}-generator (d'_j, j): extract a_{ij}, apply the Hopf formula.
                    for (dpj, j) in p_prev.iter_gens(d_i) {
                        let op_deg = d_i - dpj; // |a_{ij}|
                        let width = algebra.dimension(op_deg);
                        if width == 0 {
                            continue;
                        }
                        let off = p_prev.generator_offset(d_i, dpj, j);
                        for a_idx in 0..width {
                            let coeff = dp.entry(off + a_idx);
                            if coeff == 0 {
                                continue;
                            }
                            // ∂_Q term: Σ_(a) a' · (y_j ⊗ χ(a'') m_α).
                            for &(l_deg, l_idx, r_deg, r_idx) in
                                self.coproduct.terms(op_deg, a_idx).iter()
                            {
                                // χ(a'') m_α ∈ M_{r_deg + e_alpha}.
                                let mbeta_deg = r_deg + e_alpha;
                                let mut acted = FpVector::new(p, self.module.dimension(mbeta_deg));
                                if r_deg == 0 {
                                    // a'' = 1, so χ(a'') m_α = m_α. Do this directly: some module
                                    // actions (e.g. `RealProjectiveSpace`) short-circuit the
                                    // identity operation `op_degree == 0` and would drop this term.
                                    acted.add_basis_element(alpha, coeff);
                                } else {
                                    let chi = self.antipode.apply(r_deg, r_idx);
                                    for (op_idx, op_c) in chi.iter_nonzero() {
                                        self.module.act_on_basis(
                                            acted.as_slice_mut(),
                                            (op_c * coeff) % p.as_u32(),
                                            r_deg,
                                            op_idx,
                                            e_alpha,
                                            alpha,
                                        );
                                    }
                                }
                                if acted.is_zero() {
                                    continue;
                                }
                                // Place a' · gen(j, β) for each β: block = generator (j, β) of
                                // degree d'_j + |m_β|; within-block offset = l_idx (a' at deg l_deg).
                                let gen_deg = dpj + mbeta_deg;
                                for (beta, b_c) in acted.iter_nonzero() {
                                    let gen_idx =
                                        self.local_gen_index(s - 1, gen_deg, dpj, j, beta);
                                    let block = target.generator_offset(degree, gen_deg, gen_idx);
                                    debug_assert_eq!(degree - gen_deg, l_deg);
                                    out.add_basis_element(block + l_idx, b_c);
                                }
                            }
                        }
                    }
                }
                rows.push(out);
            }
        }
        rows
    }

    /// Index of the generator $(y_j, m_\beta)$ among the generators of $Q_s$ at degree `gen_deg`,
    /// where `y_j` is the P_s-generator `(target_pdeg, target_pidx)`. Matches the add-order in
    /// [`Self::extend`] (iterate P_s generators, then M-basis β).
    fn local_gen_index(
        &self,
        s: i32,
        gen_deg: i32,
        target_pdeg: i32,
        target_pidx: usize,
        beta: usize,
    ) -> usize {
        let p_s = self.resolution.module(s);
        let mut idx = 0;
        for (dk, k) in p_s.iter_gens(gen_deg) {
            if dk == target_pdeg && k == target_pidx {
                return idx + beta;
            }
            idx += self.module.dimension(gen_deg - dk);
        }
        panic!("generator ({target_pdeg}, {target_pidx}) not found in Q_{s} at degree {gen_deg}");
    }
}

impl<CC, N> ChainComplex for TensorResolution<CC, N>
where
    CC: FreeChainComplex,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra> + ZeroModule,
{
    type Algebra = CC::Algebra;
    type Homomorphism = FreeModuleHomomorphism<FreeModule<CC::Algebra>>;
    type Module = FreeModule<CC::Algebra>;

    fn algebra(&self) -> Arc<Self::Algebra> {
        self.resolution.algebra()
    }

    fn min_degree(&self) -> i32 {
        0
    }

    fn zero_module(&self) -> Arc<Self::Module> {
        Arc::clone(&self.zero_module)
    }

    fn module(&self, s: i32) -> Arc<Self::Module> {
        Arc::clone(&self.modules[s])
    }

    fn differential(&self, s: i32) -> Arc<Self::Homomorphism> {
        Arc::clone(&self.differentials[s as usize])
    }

    fn has_computed_bidegree(&self, b: Bidegree) -> bool {
        b.s() >= 0
            && self.differentials.len() > b.s() as usize
            && self.differential(b.s()).next_degree() > b.t()
    }

    fn compute_through_bidegree(&self, b: Bidegree) {
        self.extend(b.s(), b.t());
    }

    fn next_homological_degree(&self) -> i32 {
        self.modules.len()
    }
}

/// The Adams $d_2$ on $\Ext_A(M, k)$ for a module resolved by the field-resolution trick.
///
/// It wraps the genuine [`TensorResolution`] $Q_\bullet = P_\bullet \otimes M$ in the standard
/// [`SecondaryResolution`] machinery (which computes $d_2$ from any free resolution — $d_2$ is a
/// chain-homotopy invariant), and an [`ExtAlgebra`] whose $E_2$ page is the cohomology of
/// $\Hom_A(Q, k)$ (via [`DualizedDifferential`]). Because $Q$ is *non-minimal*, $d_2$ is only
/// defined on cohomology classes, not on cochain generators; the transport
/// $$ d_2 = \text{project} \circ (\text{cochain } d_2) \circ \text{lift} $$
/// restricts to cocycles and quotients the spurious coboundary part — exactly the
/// [`lift`](ExtAlgebra::lift)/[`project`](ExtAlgebra::project) of the cohomology-first foundation.
///
/// Works for finite *and* infinite `M`. Cross-checked against the direct
/// [`SecondaryExtAlgebra`](super::SecondaryExtAlgebra) on the minimal resolution of `M`.
pub struct FieldResolutionSecondary<CC, N>
where
    CC: FreeChainComplex + 'static,
    CC::Algebra: Bialgebra + PairAlgebra,
    N: Module<Algebra = CC::Algebra> + ZeroModule + 'static,
{
    resolution: Arc<TensorResolution<CC, N>>,
    secondary: Arc<SecondaryResolution<TensorResolution<CC, N>>>,
    /// The $E_2$ page of $\Ext_A(M, k)$, in $Q_\bullet$-generator (cochain) coordinates.
    e2: ExtAlgebra<TensorResolution<CC, N>>,
}

impl<CC, N> FieldResolutionSecondary<CC, N>
where
    CC: FreeChainComplex + 'static,
    CC::Algebra: Bialgebra + PairAlgebra,
    N: Module<Algebra = CC::Algebra> + ZeroModule + 'static,
{
    /// Build the secondary layer over the field-resolution trick. `resolution` resolves the base
    /// field `k`; `module` is `M`. Construction is cheap — call
    /// [`compute_through_stem`](Self::compute_through_stem) to do the work.
    pub fn new(resolution: Arc<CC>, module: Arc<N>) -> Self {
        let q = Arc::new(TensorResolution::new(resolution, module));
        // Q• is non-minimal, so its composite ∂∂ genuinely hits same-degree generators.
        let secondary = Arc::new(SecondaryResolution::new_with_hit_generator(
            Arc::clone(&q),
            true,
        ));
        let e2 = ExtAlgebra::without_unit(Arc::clone(&q))
            .with_differential(Arc::new(DualizedDifferential::new(Arc::clone(&q))));
        Self {
            resolution: q,
            secondary,
            e2,
        }
    }

    fn prime(&self) -> ValidPrime {
        self.resolution.prime()
    }

    /// The materialised free resolution $Q_\bullet = P_\bullet \otimes M$.
    pub fn resolution(&self) -> &Arc<TensorResolution<CC, N>> {
        &self.resolution
    }

    /// The $E_2$ page $\Ext_A(M, k)$, with cohomology exposed via
    /// [`lift`](ExtAlgebra::lift)/[`project`](ExtAlgebra::project).
    pub fn ext(&self) -> &ExtAlgebra<TensorResolution<CC, N>> {
        &self.e2
    }

    /// Compute $Q_\bullet$ and its secondary homotopies far enough to read $d_2$ on the box up to
    /// `max`. Resolves $Q_\bullet$ with the margin the Adams $d_2$ needs (one extra stem and two
    /// extra filtrations), then extends the secondary resolution.
    pub fn compute_through_stem(&self, max: Bidegree) {
        // d2 out of (n, s) reads the target (n-1, s+2), and the secondary homotopies at s need the
        // resolution two filtrations higher; also grow one extra stem for the incoming coboundary.
        let margin = Bidegree::n_s(max.n() + 1, max.s() + 3);
        self.resolution.compute_through_bidegree(margin);
        self.compute_secondary();
    }

    /// Compute the secondary homotopies on the non-minimal $Q_\bullet$.
    ///
    /// This mirrors [`SecondaryLift::extend_all`] but drives the final homotopy pass in an order
    /// suited to a *non-minimal* resolution. The standard [`iter_s_t`](sseq::coordinates::iter_s_t)
    /// driver only guarantees $h_{s-1}(t')$ for $t' < t$ before computing $h_s(t)$ — enough for a
    /// minimal resolution, whose differential has no identity component. Our $Q_\bullet$ *does*
    /// (that non-minimality is the whole point), so $h_s(t)$ depends on $h_{s-1}(t)$ as well; we
    /// therefore compute `t` outer, `s` inner and strictly increasing (sequential in `s`).
    fn compute_secondary(&self) {
        let sec = &self.secondary;
        sec.initialize_homotopies();
        sec.compute_composites();
        sec.compute_intermediates();

        // Base case: the homotopies at s = shift.s() (= 2) are zero.
        let shift = sec.shift();
        {
            let h = &sec.homotopies()[shift.s()];
            h.homotopies.extend_by_zero(h.composites.max_degree());
        }

        let min_t = sec.homotopies()[shift.s()].homotopies.min_degree();
        let s_range = sec.homotopies().range();
        let max = sec.max().restrict(s_range.end);
        // `s` starts one above the base case (whose homotopies are the zero map, set above), and
        // runs `t`-outer / `s`-inner strictly increasing so `h_{s-1}(t)` is ready before `h_s(t)`.
        sseq::coordinates::iter_s_t_sequential(
            &|b| sec.compute_homotopy_step(b),
            Bidegree::s_t(s_range.start + 1, min_t),
            max,
        );
    }

    /// The dimension of $\Ext_A(M, k)$ (the $E_2$ page) at bidegree `b`.
    pub fn cohomology_dimension(&self, b: Bidegree) -> Option<usize> {
        self.e2.cohomology_dimension(b)
    }

    /// The Adams differential $d_2(x)$, a class in bidegree `(n - 1, s + 2)`.
    ///
    /// Returns `None` if the target bidegree is out of the computed range. A computed-but-zero
    /// differential is `Some` of a zero class.
    pub fn d2(&self, x: &BidegreeElement) -> Option<BidegreeElement> {
        let b = x.degree();
        let target = b + Bidegree::n_s(-1, 2);
        if !(b.t() > 0 && self.resolution.has_computed_bidegree(target)) {
            return None;
        }

        // Lift the class to a cocycle representative in Q_•-generator coordinates.
        let cocycle = self.e2.lift(x);

        // Cochain-level d2: `m[i]` is the d2 of the i-th Q_•-generator at `b`, as a vector over the
        // Q_•-generators at `target`. This is exactly what `SecondaryResolution::e3_page` reads.
        let m = self.secondary.homotopy(b.s() + 2).homotopies.hom_k(b.t());

        let target_dim = self.resolution.number_of_gens_in_bidegree(target);
        let mut out = FpVector::new(self.prime(), target_dim);
        if !m.is_empty() && !m[0].is_empty() {
            let p = self.prime().as_u32();
            for (i, ci) in cocycle.vec().iter_nonzero() {
                for (k, &v) in m[i].iter().enumerate() {
                    out.add_basis_element(k, (ci * v) % p);
                }
            }
        }

        // Project the resulting cochain back to an Ext class (quotient by coboundaries).
        Some(self.e2.project(&Cochain::new(target, out)))
    }

    /// Whether `x` is a $d_2$-cycle (survives to $E_3$). `None` if $d_2$ is out of range.
    pub fn survives(&self, x: &BidegreeElement) -> Option<bool> {
        self.d2(x).map(|d| d.vec().is_zero())
    }
}

/// Primary Massey products $\langle a, b, c\rangle$ on $\Ext_A(M, k)$ for the field trick, computed
/// by the **cochain DGA** (cup product + $\delta$-preimage) rather than chain-map null-homotopies.
///
/// On a non-minimal resolution the chain-map + null-homotopy construction degenerates — the lifted
/// product map $f_c$ can be forced to zero at higher filtration, so the composite $f_b \circ f_c$
/// vanishes and the bracket is lost. The dual construction works *precisely* when the cochain
/// differential $\delta \neq 0$, which is exactly the non-minimal $Q_\bullet$ the field trick
/// produces. Because the unit $P_\bullet$ is minimal, $a \cdot b = 0$ holds at the cochain level (so
/// the $a\cup b$ null-homotopy $u = 0$), and the bracket reduces to
/// $$ \langle a, b, c\rangle = [\, a \cup v \,], \qquad \delta_Q\, v = b \cup c. $$
/// Cup products are the closed-form [`cup_matrix`](TensorResolutionDifferential::cup_matrix); the
/// $\delta$-preimage is a quasi-inverse of the cochain differential. Every operand is robust: the
/// cup reads the chain map $f_x$'s `hom_k` (the verified product data), and $\delta_Q \neq 0$ makes
/// `v` a genuine preimage — so the bracket is nonzero exactly where the null-homotopy read zero.
pub struct FieldMassey<CC, N>
where
    CC: FreeChainComplex + AugmentedChainComplex + Sync + 'static,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra> + ZeroModule + 'static,
{
    /// $\Ext_A(M, k)$ with products (resolution $= Q_\bullet$, unit $= P_\bullet$).
    ext: ExtAlgebra<TensorResolution<CC, N>, CC>,
    /// $\Ext_A(k, k)$, for validating $a \cdot b = 0$ in the unit.
    unit_ext: ExtAlgebra<CC, CC>,
    /// The closed-form cochain engine: $\delta_Q$ and the cup products $x \cup -$.
    cup: TensorResolutionDifferential<CC, N>,
    /// $P_\bullet$, for realising a class of $\Ext(k, k)$ as a chain self-map $f_x$.
    unit: Arc<CC>,
    /// Cache of $f_x\colon P_\bullet \to P_\bullet$ per multiplier class of $\Ext(k, k)$.
    f_maps: DashMap<BidegreeElement, Arc<ResolutionHomomorphism<CC, CC>>>,
}

impl<CC, N> FieldMassey<CC, N>
where
    CC: FreeChainComplex + AugmentedChainComplex + Sync + 'static,
    CC::Algebra: Bialgebra,
    N: Module<Algebra = CC::Algebra> + ZeroModule + 'static,
{
    /// Build the field-trick Massey engine. `resolution` resolves the base field `k`; `module` is
    /// `M`. Construction is cheap — call [`compute_through_bidegree`](Self::compute_through_bidegree).
    pub fn new(resolution: Arc<CC>, module: Arc<N>) -> Self {
        let ext = field_resolution_products(Arc::clone(&resolution), Arc::clone(&module));
        let unit_ext = ExtAlgebra::new(Arc::clone(&resolution), Arc::clone(&resolution));
        let cup = TensorResolutionDifferential::new(Arc::clone(&resolution), module);
        Self {
            ext,
            unit_ext,
            cup,
            unit: resolution,
            f_maps: DashMap::new(),
        }
    }

    /// The underlying $\Ext_A(M, k)$ (products, cohomology, `lift`/`project`).
    pub fn ext(&self) -> &ExtAlgebra<TensorResolution<CC, N>, CC> {
        &self.ext
    }

    fn prime(&self) -> ValidPrime {
        self.ext.prime()
    }

    /// Compute $Q_\bullet$, $P_\bullet$ and the $\Ext(k,k)$ side far enough to read brackets landing
    /// in the box up to `max`. Grows one extra stem and two extra filtrations for the cup products
    /// and the $\delta$-preimage.
    pub fn compute_through_bidegree(&self, max: Bidegree) {
        let margin = Bidegree::n_s(max.n() + 2, max.s() + 3);
        self.ext.compute_through_bidegree(margin);
        self.unit_ext.compute_through_bidegree(margin);
    }

    /// The chain self-map $f_x\colon P_\bullet \to P_\bullet$ realising `x ∈ Ext(k, k)`, cached.
    fn f_map(&self, x: &BidegreeElement) -> Arc<ResolutionHomomorphism<CC, CC>> {
        if let Some(m) = self.f_maps.get(x) {
            return Arc::clone(&m);
        }
        let hom = Arc::new(ResolutionHomomorphism::from_class(
            format!("cup_{x}"),
            Arc::clone(&self.unit),
            Arc::clone(&self.unit),
            x.degree(),
            &x.vec().iter().collect::<Vec<_>>(),
        ));
        Arc::clone(self.f_maps.entry(x.clone()).or_insert(hom).value())
    }

    /// The cup product `x ∪ v` of `x ∈ Ext(k, k)` with a cochain `v ∈ Hom(Q•, k)`, landing at
    /// `v.degree() + x.degree()`. The unit `P•` is minimal, so `f_x` needs no sequential extension.
    fn cup(&self, x: &BidegreeElement, v: &Cochain) -> Cochain {
        let src = v.degree();
        let tgt = src + x.degree();
        let f_x = self.f_map(x);
        f_x.extend_through_stem(tgt);
        let m = self.cup.cup_matrix(&f_x, src.s(), src.t(), x.degree());
        let mut out = FpVector::new(self.prime(), m.columns());
        m.apply(out.as_slice_mut(), 1, v.vec());
        Cochain::new(tgt, out)
    }

    /// A cochain `v` with `δ_Q v = z`, or `None` if `z` is not a coboundary (so the bracket is
    /// undefined). `δ_Q` lowers filtration for the preimage: `v` lives at `z.degree() + (n+1, s-1)`.
    fn delta_preimage(&self, z: &Cochain) -> Option<Cochain> {
        let p = self.prime();
        let v_deg = z.degree() + Bidegree::n_s(1, -1);
        // D = δ_Q out of v_deg : C^{v_deg} → C^{z.degree()}. Rows index v, columns index z.
        let d = self.cup.matrix(v_deg)?;
        let (r, c) = (d.rows(), d.columns());
        if z.vec().len() != c {
            return None;
        }
        // Solve v·D = z via the quasi-inverse of [D | I].
        let mut aug = AugmentedMatrix::<2>::new(p, r, [c, r]);
        for i in 0..r {
            aug.row_mut(i).slice_mut(0, c).add(d.row(i), 1);
        }
        aug.segment(1, 1).add_identity();
        aug.row_reduce();
        let qi = aug.compute_quasi_inverse();

        let mut v = FpVector::new(p, r);
        qi.apply(v.as_slice_mut(), 1, z.vec());

        // Verify: `qi.apply` silently drops any component of `z` outside `im δ_Q`, so re-check that
        // δ_Q v = z (i.e. `v·D - z = 0`).
        let mut back = FpVector::new(p, c);
        d.apply(back.as_slice_mut(), 1, v.as_slice());
        back.as_slice_mut().add(z.vec(), p.as_u32() - 1);
        if !back.is_zero() {
            return None;
        }
        Some(Cochain::new(v_deg, v))
    }

    /// The triple Massey product $\langle a, b, c\rangle$ with `a, b ∈ Ext(k, k)` and
    /// `c ∈ Ext(M, k)`, via the cochain DGA. Returns `None` when `a·b ≠ 0`, `b·c ≠ 0`, or the
    /// bracket bidegree is out of the computed range.
    pub fn massey(
        &self,
        a: &BidegreeElement,
        b: &BidegreeElement,
        c: &BidegreeElement,
    ) -> Option<MasseyResult> {
        let tot = a.degree() + b.degree() + c.degree() - Bidegree::s_t(1, 0);
        // The cup/preimage pipeline reads cochains at `b∪c` (= `b + c`), its δ-preimage `v`, and the
        // bracket `tot`. Every one must be resolved, or the cochain reads run off the computed box.
        // `has_computed_bidegree` is panic-safe (unlike `cohomology`/`cochain_dimension`).
        let res = self.ext.resolution();
        let bc_deg = b.degree() + c.degree();
        let v_deg = bc_deg + Bidegree::n_s(1, -1);
        for d in [bc_deg, v_deg, tot] {
            if d.s() < 0 || d.t() < 0 || !res.has_computed_bidegree(d) {
                return None;
            }
        }
        self.ext.cohomology(tot)?;

        // a·b = 0 is required for u = 0 (the a∪b null-homotopy) to be valid.
        match self.unit_ext.try_multiply(a, b) {
            Some(ab) if ab.vec().is_zero() => {}
            _ => return None,
        }

        // v with δ_Q v = b∪c; `None` iff b·c ≠ 0 (b∪c not a coboundary).
        let bc = self.cup(b, &self.ext.lift(c));
        let v = self.delta_preimage(&bc)?;
        let av = self.cup(a, &v);
        let representative = self.ext.project(&av).into_vec();
        let indeterminacy = self.ext.massey_indeterminacy(a, c, tot);
        Some(MasseyResult {
            degree: tot,
            coset: AffineSubspace::new(representative, indeterminacy),
        })
    }

    /// The family of Massey products $\langle a, b, -\rangle$ for fixed `a, b ∈ Ext(k, k)` and every
    /// valid third factor `c ∈ Ext(M, k)` (the kernel of multiplication by `b`) across the computed
    /// range. Brackets that contain `0` are omitted. Assumes `a·b = 0`.
    pub fn massey_iter_c(
        &self,
        a: &BidegreeElement,
        b: &BidegreeElement,
    ) -> Vec<(BidegreeElement, MasseyResult)> {
        let mut results = Vec::new();
        for c_deg in self.ext.resolution().iter_nonzero_stem() {
            let Some(kernel) = self.ext.massey_kernel(b, c_deg) else {
                continue;
            };
            for row in kernel.iter() {
                let c = BidegreeElement::new(c_deg, row.to_owned());
                let Some(result) = self.massey(a, b, &c) else {
                    continue;
                };
                if result.contains_zero() {
                    continue;
                }
                results.push((c, result));
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use algebra::{
        SteenrodAlgebra,
        module::{FDModule, Module},
    };
    use fp::prime::TWO;
    use sseq::coordinates::Bidegree;

    use super::*;
    use crate::{
        chain_complex::{ChainComplex, FreeChainComplex},
        utils::construct_standard,
    };

    /// A helper: the antipode of `S_2`'s algebra, with the algebra basis computed through `max`.
    fn sphere_antipode(max: i32) -> (Arc<SteenrodAlgebra>, Antipode<SteenrodAlgebra>) {
        let res = construct_standard::<false, _, _>("S_2", None).unwrap();
        let algebra = res.algebra();
        algebra.compute_basis(max + 1);
        let antipode = Antipode::new(Arc::clone(&algebra));
        (algebra, antipode)
    }

    #[test]
    fn antipode_low_degree_and_involution() {
        // Basis-agnostic checks (the default basis is Milnor). The unique degree-1 generator is
        // primitive, so χ fixes it; and the Steenrod algebra is cocommutative, so χ² = id.
        let (algebra, chi) = sphere_antipode(10);

        assert_eq!(algebra.dimension(1), 1);
        let mut deg1 = FpVector::new(TWO, 1);
        deg1.add_basis_element(0, 1);
        assert_eq!(
            chi.apply(1, 0),
            deg1,
            "χ fixes the primitive degree-1 generator"
        );

        for degree in 0..=10 {
            for idx in 0..algebra.dimension(degree) {
                let chi_x = chi.apply(degree, idx);
                let chi_chi_x = chi.apply_element(degree, chi_x);
                let mut x = FpVector::new(TWO, algebra.dimension(degree));
                x.add_basis_element(idx, 1);
                assert_eq!(chi_chi_x, x, "χ² = id at degree {degree}, idx {idx}");
            }
        }
    }

    #[test]
    fn antipode_hopf_identity() {
        // Σ χ(x_(1)) x_(2) = ε(x)·1 for every basis element x through some degree.
        let (algebra, chi) = sphere_antipode(10);
        let coproduct = FullCoproduct::new(Arc::clone(&algebra));
        for degree in 1..=10 {
            for idx in 0..algebra.dimension(degree) {
                let mut acc = FpVector::new(TWO, algebra.dimension(degree));
                // Use the same decompose/coproduct route as the antipode for the full coproduct.
                for &(l_deg, l_idx, r_deg, r_idx) in coproduct.terms(degree, idx).iter() {
                    let chi_left = chi.apply(l_deg, l_idx);
                    for (j, coeff) in chi_left.iter_nonzero() {
                        algebra.multiply_basis_elements(
                            acc.as_slice_mut(),
                            coeff,
                            l_deg,
                            j,
                            r_deg,
                            r_idx,
                        );
                    }
                }
                assert!(
                    acc.is_zero(),
                    "Hopf identity failed at degree {degree}, idx {idx}: {acc:?}"
                );
            }
        }
    }

    /// The tensor trick reproduces the direct minimal resolution of a finite module.
    fn assert_matches_direct(module_name: &str, sphere: &str, nn: i32, ss: i32) {
        let t_max = nn + ss;

        // k-resolution and M over the same algebra.
        let k_res = Arc::new(construct_standard::<false, _, _>(sphere, None).unwrap());
        k_res.compute_through_bidegree(Bidegree::s_t(ss + 1, t_max));
        k_res.algebra().compute_basis(t_max + 1);

        let m_json = crate::utils::parse_module_name(module_name).unwrap();
        let module = Arc::new(
            FDModule::from_json(k_res.algebra(), &m_json).expect("finite module for cross-check"),
        );
        module.compute_basis(t_max);

        let alg = field_resolution_ext(Arc::clone(&k_res), module);

        // Direct minimal resolution of M (its generator counts are Ext(M,k)).
        let direct = Arc::new(construct_standard::<false, _, _>(module_name, None).unwrap());
        direct.compute_through_stem(Bidegree::n_s(nn, ss));

        for n in 0..=nn {
            for s in 0..=ss {
                let b = Bidegree::n_s(n, s);
                assert_eq!(
                    alg.cohomology_dimension(b),
                    Some(direct.number_of_gens_in_bidegree(b)),
                    "Ext({module_name}, k) mismatch at {b:?}",
                );
            }
        }
    }

    #[test]
    fn tensor_trick_matches_direct_c2() {
        assert_matches_direct("C2", "S_2", 10, 6);
    }

    #[test]
    fn cohomology_basis_and_lift_project() {
        use sseq::coordinates::BidegreeGenerator;

        // On the field trick (non-minimal) `Ext*` exposes the cohomology: `dimension` is the
        // cohomology dim (≤ the cochain-generator count, strictly less where the coboundary bites),
        // and `project ∘ lift` is the identity on cohomology classes.
        let (nn, ss) = (12, 7);
        let t_max = nn + ss;
        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        k_res.compute_through_bidegree(Bidegree::s_t(ss + 1, t_max));
        k_res.algebra().compute_basis(t_max + 1);
        let m = Arc::new(
            FDModule::from_json(
                k_res.algebra(),
                &crate::utils::parse_module_name("C2").unwrap(),
            )
            .unwrap(),
        );
        m.compute_basis(t_max);
        let alg = field_resolution_ext(Arc::clone(&k_res), m);

        let mut saw_nonminimal = false;
        for n in 0..=nn {
            for s in 0..=ss {
                let b = Bidegree::n_s(n, s);
                let dim = alg.dimension(b);
                assert_eq!(Some(dim), alg.cohomology_dimension(b));
                let cochain = alg.cochain_dimension(b);
                assert!(
                    cochain >= dim,
                    "cochain {cochain} < cohomology {dim} at {b:?}"
                );
                saw_nonminimal |= cochain > dim;

                for i in 0..dim {
                    let class = alg.generator(BidegreeGenerator::new(b, i));
                    assert_eq!(
                        class,
                        alg.project(&alg.lift(&class)),
                        "project∘lift {b:?}#{i}"
                    );
                }
            }
        }
        assert!(
            saw_nonminimal,
            "expected a non-minimal bidegree where cochain dim > cohomology dim"
        );
    }

    #[test]
    fn tensor_trick_matches_direct_rp_inf() {
        // The real payoff: `RP^∞` is *infinite*, so Nassau's engine cannot resolve it — but the
        // non-nassau standard resolver works degreewise and *can*. Cross-check the whole Ext chart
        // of the tensor trick against that direct resolution of RP^∞.
        use algebra::module::RealProjectiveSpace;

        let (nn, ss) = (18, 8);
        let t_max = nn + ss;

        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        k_res.compute_through_bidegree(Bidegree::s_t(ss + 1, t_max));
        k_res.algebra().compute_basis(t_max + 1);
        let rp = Arc::new(RealProjectiveSpace::new(k_res.algebra(), 1, None, false));
        rp.compute_basis(t_max);
        let alg = field_resolution_ext(Arc::clone(&k_res), rp);

        // Direct resolution of the infinite module RP^∞ (non-nassau; resolves degreewise).
        let direct = Arc::new(construct_standard::<false, _, _>("RP_inf", None).unwrap());
        direct.compute_through_stem(Bidegree::n_s(nn, ss));

        for n in 0..=nn {
            for s in 0..=ss {
                let b = Bidegree::n_s(n, s);
                assert_eq!(
                    alg.cohomology_dimension(b),
                    Some(direct.number_of_gens_in_bidegree(b)),
                    "Ext(RP^∞, k) mismatch at {b:?}",
                );
            }
        }

        // Sanity: Ext^0 = A-module indecomposables of RP^∞ = F_2[x], which sit in degrees 2^k − 1
        // (since Sq^1 x = x^2 makes even powers decomposable).
        for n in 1..=nn {
            let expected = usize::from(((n + 1) as u32).is_power_of_two());
            assert_eq!(
                alg.cohomology_dimension(Bidegree::n_s(n, 0)),
                Some(expected),
                "Ext^0(RP^∞, k) at n = {n}",
            );
        }
    }

    #[test]
    fn closed_form_dimension() {
        // dim C^s_t = Σ_i dim M_{t - d_i}, straight from generator degrees and M's dimensions.
        let (nn, ss) = (10, 6);
        let t_max = nn + ss;
        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        k_res.compute_through_bidegree(Bidegree::s_t(ss + 1, t_max));
        k_res.algebra().compute_basis(t_max + 1);
        let m_json = crate::utils::parse_module_name("C2").unwrap();
        let module = Arc::new(FDModule::from_json(k_res.algebra(), &m_json).unwrap());
        module.compute_basis(t_max);

        let diff = TensorResolutionDifferential::new(Arc::clone(&k_res), Arc::clone(&module));

        for n in 0..=nn {
            for s in 0..=ss {
                let b = Bidegree::n_s(n, s);
                let t = b.t();
                let expected: usize = k_res
                    .module(s)
                    .iter_gens(t)
                    .map(|(d_i, _)| module.dimension(t - d_i))
                    .sum();
                assert_eq!(diff.dimension(b), Some(expected), "dim C^{s}_{t} at {b:?}");
            }
        }
    }

    pub(crate) fn finite_module_pub(
        algebra: &Arc<SteenrodAlgebra>,
        name: &str,
        t_max: i32,
    ) -> Arc<FDModule<SteenrodAlgebra>> {
        finite_module(algebra, name, t_max)
    }

    /// Build the trivial module for `name` over the sphere algebra, basis computed through `t_max`.
    fn finite_module(
        algebra: &Arc<SteenrodAlgebra>,
        name: &str,
        t_max: i32,
    ) -> Arc<FDModule<SteenrodAlgebra>> {
        let m = Arc::new(
            FDModule::from_json(
                Arc::clone(algebra),
                &crate::utils::parse_module_name(name).unwrap(),
            )
            .unwrap(),
        );
        m.compute_basis(t_max);
        m
    }

    #[test]
    fn tensor_resolution_is_a_complex() {
        // ∂ ∘ ∂ = 0 on every generator of Q_• = P_• ⊗ C2 over a box — the free differential is
        // conjugate to ∂_P ⊗ id, so it must square to zero.
        use algebra::module::homomorphism::ModuleHomomorphism;

        let (ss, t_max) = (6, 16);
        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let m = finite_module(&k_res.algebra(), "C2", t_max);
        let q = TensorResolution::new(Arc::clone(&k_res), m);
        q.compute_through_bidegree(Bidegree::s_t(ss, t_max));

        for s in 2..=ss {
            let d_s = q.differential(s);
            let d_prev = q.differential(s - 1);
            let q_s = q.module(s);
            let q_prev2 = q.module(s - 2);
            for t in 0..=t_max {
                for i in 0..q_s.number_of_gens_in_degree(t) {
                    let dx = d_s.output(t, i); // ∂(gen) ∈ Q_{s-1} at t
                    let mut ddx = FpVector::new(TWO, q_prev2.dimension(t));
                    d_prev.apply(ddx.as_slice_mut(), 1, t, dx.as_slice());
                    assert!(ddx.is_zero(), "∂² ≠ 0 at s = {s}, t = {t}, gen {i}");
                }
            }
        }
    }

    #[test]
    fn q_complex_computes_ext_c2() {
        // Gate A: the genuine Q_• complex (via its dualised differential) computes the same additive
        // Ext(C2, k) as the closed form and the direct minimal resolution — and is genuinely
        // non-minimal (some bidegree has more cochains than cohomology).
        let (nn, ss) = (10, 6);
        let t_max = nn + ss;
        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        k_res.compute_through_bidegree(Bidegree::s_t(ss + 1, t_max));
        k_res.algebra().compute_basis(t_max + 1);
        let m = finite_module(&k_res.algebra(), "C2", t_max);

        let q = Arc::new(TensorResolution::new(Arc::clone(&k_res), Arc::clone(&m)));
        q.compute_through_bidegree(Bidegree::s_t(ss + 1, t_max));
        let e2 = ExtAlgebra::without_unit(Arc::clone(&q))
            .with_differential(Arc::new(DualizedDifferential::new(Arc::clone(&q))));

        let closed = field_resolution_ext(Arc::clone(&k_res), Arc::clone(&m));
        let direct = Arc::new(construct_standard::<false, _, _>("C2", None).unwrap());
        direct.compute_through_stem(Bidegree::n_s(nn, ss));

        let mut saw_nonminimal = false;
        for n in 0..=nn {
            for s in 0..=ss {
                let b = Bidegree::n_s(n, s);
                let d = direct.number_of_gens_in_bidegree(b);
                assert_eq!(
                    e2.cohomology_dimension(b),
                    Some(d),
                    "Q• Ext vs direct at {b:?}"
                );
                assert_eq!(
                    e2.cohomology_dimension(b),
                    closed.cohomology_dimension(b),
                    "Q• Ext vs closed form at {b:?}"
                );
                saw_nonminimal |= e2.cochain_dimension(b) > d;
            }
        }
        assert!(
            saw_nonminimal,
            "expected a non-minimal bidegree (cochains > cohomology)"
        );
    }

    #[test]
    fn field_d2_reproduces_sphere() {
        // Gate B sanity: with M = k the tensor complex Q_• collapses to the minimal P_• (the trivial
        // module makes χ(a) act as ε(a)), so the field-trick d2 must reproduce the standard Adams
        // d2: d2(h4) = h0 h3² at (14, 3), with h0, h1, h2 permanent.
        use sseq::coordinates::BidegreeGenerator;

        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let m = finite_module(&k_res.algebra(), "S_2", 24);
        let sec = FieldResolutionSecondary::new(k_res, m);
        sec.compute_through_stem(Bidegree::n_s(16, 4));

        for (n, s) in [(0, 1), (1, 1), (3, 1)] {
            let b = Bidegree::n_s(n, s);
            let h = sec.ext().generator(BidegreeGenerator::new(b, 0));
            assert_eq!(
                sec.survives(&h),
                Some(true),
                "h at (n = {n}, s = {s}) should survive d2"
            );
        }

        let h4 = sec
            .ext()
            .generator(BidegreeGenerator::new(Bidegree::n_s(15, 1), 0));
        let d = sec.d2(&h4).expect("d2(h4) target should be computed");
        assert_eq!(d.degree(), Bidegree::n_s(14, 3));
        assert_eq!(sec.cohomology_dimension(Bidegree::n_s(14, 3)), Some(1));
        assert!(!d.vec().is_zero(), "d2(h4) = h0 h3² should be nonzero");
        assert_eq!(sec.survives(&h4), Some(false), "h4 should not survive d2");
    }

    #[test]
    fn field_d2_matches_direct_c2() {
        // Gate B: on the genuinely non-minimal Q_• = P_• ⊗ C2, the field-trick d2 agrees with the
        // direct minimal resolution's secondary d2. Ranks of the outgoing d2 (a basis-independent
        // invariant of the Adams differential) match at every bidegree, and the E2 dimensions agree.
        use fp::matrix::Matrix;
        use sseq::coordinates::BidegreeGenerator;

        use crate::ext_algebra::SecondaryExtAlgebra;

        let (nn, ss) = (12, 6);

        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let m = finite_module(&k_res.algebra(), "C2", nn + ss + 8);
        let field = FieldResolutionSecondary::new(k_res, m);
        field.compute_through_stem(Bidegree::n_s(nn, ss));

        let direct_res = Arc::new(construct_standard::<false, _, _>("C2", None).unwrap());
        direct_res.compute_through_stem(Bidegree::n_s(nn + 1, ss + 3));
        let direct_e2 = Arc::new(ExtAlgebra::new(
            Arc::clone(&direct_res),
            Arc::clone(&direct_res),
        ));
        let direct_sec = SecondaryExtAlgebra::new(Arc::clone(&direct_e2));
        direct_sec.extend_all();

        let rank_of =
            |dim: usize, target_dim: usize, d2_of: &mut dyn FnMut(usize) -> FpVector| -> usize {
                if dim == 0 || target_dim == 0 {
                    return 0;
                }
                let rows: Vec<FpVector> = (0..dim).map(&mut *d2_of).collect();
                Matrix::from_rows(TWO, rows, target_dim).row_reduce()
            };

        let mut compared = 0;
        for n in 1..=nn {
            for s in 1..=ss {
                let b = Bidegree::n_s(n, s);
                let target = b + Bidegree::n_s(-1, 2);
                let (Some(fd), Some(ftd)) = (
                    field.cohomology_dimension(b),
                    field.cohomology_dimension(target),
                ) else {
                    continue;
                };
                assert_eq!(fd, direct_e2.dimension(b), "E2 dim mismatch at {b:?}");
                assert_eq!(
                    ftd,
                    direct_e2.dimension(target),
                    "E2 dim mismatch at target {target:?}"
                );

                let field_rank = rank_of(fd, ftd, &mut |i| {
                    field
                        .d2(&field.ext().generator(BidegreeGenerator::new(b, i)))
                        .map(BidegreeElement::into_vec)
                        .unwrap_or_else(|| FpVector::new(TWO, ftd))
                });
                let direct_rank = rank_of(fd, ftd, &mut |i| {
                    direct_sec
                        .d2(&direct_e2.generator(BidegreeGenerator::new(b, i)))
                        .map(BidegreeElement::into_vec)
                        .unwrap_or_else(|| FpVector::new(TWO, ftd))
                });
                assert_eq!(field_rank, direct_rank, "d2 rank mismatch at {b:?}");
                compared += 1;
            }
        }
        assert!(compared > 0, "no bidegrees compared");
    }

    #[test]
    fn field_products_match_direct_c2() {
        // The field-trick $\Ext(k, k)$-module products on $\Ext(C2, k)$ agree with the direct
        // minimal resolution's. The two Ext bases differ by a change of basis, so we compare the
        // basis-independent **rank** of multiplication by `h0` at every bidegree (plus the Ext
        // dimensions). Sharing the *same* minimal `S_2` unit removes any ambiguity on the operand
        // side. That `multiply` runs to completion on the non-minimal `Q•` (sequential extension +
        // lift/project transport) is itself the regression proof.
        use sseq::coordinates::BidegreeGenerator;

        use super::field_resolution_products;

        let (nn, ss) = (8, 5);
        let t_max = nn + ss + 6;

        let s2 = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        s2.compute_through_bidegree(Bidegree::s_t(ss + 3, t_max));
        s2.algebra().compute_basis(t_max + 1);
        let m = finite_module(&s2.algebra(), "C2", t_max);

        // Field side: products via the tensor trick; `unit` is the shared minimal `S_2`.
        let field = field_resolution_products(Arc::clone(&s2), Arc::clone(&m));
        field.compute_through_bidegree(Bidegree::s_t(ss + 3, t_max));

        // Direct side: minimal resolution of C2, sharing the same minimal `S_2` unit.
        let c2 = Arc::new(construct_standard::<false, _, _>("C2", None).unwrap());
        c2.compute_through_stem(Bidegree::n_s(nn + 1, ss + 3));
        let direct = ExtAlgebra::new(Arc::clone(&c2), Arc::clone(&s2));

        let h0 = BidegreeGenerator::new(Bidegree::n_s(0, 1), 0);
        let field_h0 = field.unit_generator(h0);
        let direct_h0 = direct.unit_generator(h0);

        let rank_of =
            |dim: usize, tdim: usize, f: &mut dyn FnMut(usize) -> Option<FpVector>| -> usize {
                if dim == 0 || tdim == 0 {
                    return 0;
                }
                let rows: Vec<FpVector> = (0..dim)
                    .map(|i| f(i).unwrap_or_else(|| FpVector::new(TWO, tdim)))
                    .collect();
                Matrix::from_rows(TWO, rows, tdim).row_reduce()
            };

        let mut compared = 0;
        let mut saw_nonzero = false;
        for n in 0..=nn {
            for s in 0..=ss {
                let b = Bidegree::n_s(n, s);
                let target = b + h0.degree();
                let (Some(fd), Some(ftd)) = (
                    field.cohomology_dimension(b),
                    field.cohomology_dimension(target),
                ) else {
                    continue;
                };
                assert_eq!(fd, direct.dimension(b), "Ext dim mismatch at {b:?}");
                assert_eq!(
                    ftd,
                    direct.dimension(target),
                    "Ext dim mismatch at target {target:?}"
                );

                let field_rank = rank_of(fd, ftd, &mut |i| {
                    field
                        .try_multiply(&field.generator(BidegreeGenerator::new(b, i)), &field_h0)
                        .map(BidegreeElement::into_vec)
                });
                let direct_rank = rank_of(fd, ftd, &mut |i| {
                    direct
                        .try_multiply(&direct.generator(BidegreeGenerator::new(b, i)), &direct_h0)
                        .map(BidegreeElement::into_vec)
                });
                assert_eq!(field_rank, direct_rank, "(·h0) rank mismatch at {b:?}");
                saw_nonzero |= field_rank > 0;
                compared += 1;
            }
        }
        assert!(compared > 0, "no bidegrees compared");
        assert!(saw_nonzero, "expected nonzero h0-multiplication somewhere");
    }

    #[test]
    fn field_massey_dga_family_matches_direct_c2() {
        // The cochain-DGA field-trick Massey family ⟨h0, h1, -⟩ on Ext(C2, k) agrees with the direct
        // minimal resolution across the whole computed range: same set of nonzero brackets, keyed by
        // bracket bidegree with matching indeterminacy dimensions. This is the family the chain-map /
        // null-homotopy path could not compute (see the module docs on `FieldMassey`).
        use sseq::coordinates::BidegreeGenerator;

        use crate::ext_algebra::massey::MasseyResult;

        let (nn, ss) = (8, 5);
        let s2 = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let m = finite_module(&s2.algebra(), "C2", nn + ss + 8);
        let fm = FieldMassey::new(Arc::clone(&s2), m);
        fm.compute_through_bidegree(Bidegree::n_s(nn, ss));

        let c2 = Arc::new(construct_standard::<false, _, _>("C2", None).unwrap());
        c2.compute_through_stem(Bidegree::n_s(nn + 1, ss + 3));
        let direct = ExtAlgebra::new(Arc::clone(&c2), Arc::clone(&s2));

        let ext = fm.ext();
        let h0 = ext.unit_generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));
        let h1 = ext.unit_generator(BidegreeGenerator::new(Bidegree::n_s(1, 1), 0));
        let d_h0 = direct.unit_generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));
        let d_h1 = direct.unit_generator(BidegreeGenerator::new(Bidegree::n_s(1, 1), 0));

        // Key a family by bracket bidegree → sorted (bracket degree, indeterminacy dim). Restrict to
        // the range both resolutions cover.
        let summarize = |family: Vec<(BidegreeElement, MasseyResult)>| -> Vec<(String, usize)> {
            let mut keyed: Vec<(String, usize)> = family
                .into_iter()
                .filter(|(_, r)| r.degree.n() <= nn && r.degree.s() <= ss)
                .map(|(_, r)| (format!("{:?}", r.degree), r.coset.linear_part().dimension()))
                .collect();
            keyed.sort();
            keyed
        };

        let field_fam = summarize(fm.massey_iter_c(&h0, &h1));
        let direct_fam = summarize(direct.massey_iter_c(&d_h0, &d_h1));
        assert!(!field_fam.is_empty(), "expected some nonzero brackets");
        assert_eq!(
            field_fam, direct_fam,
            "⟨h0, h1, -⟩ over C2 disagrees field (DGA) vs direct"
        );
    }

    #[test]
    fn field_massey_dga_cup_matches_products() {
        // Stage 1 of the cochain-DGA Massey: the closed-form cup product `h0 ∪ v` reproduces the
        // chain-map product on Ext(C2, k). For every class y, project(h0 ∪ lift(y)) == y · h0.
        use sseq::coordinates::BidegreeGenerator;

        let (nn, ss) = (8, 5);
        let s2 = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let m = finite_module(&s2.algebra(), "C2", nn + ss + 8);
        let fm = FieldMassey::new(Arc::clone(&s2), m);
        fm.compute_through_bidegree(Bidegree::n_s(nn, ss));
        let ext = fm.ext();
        let h0 = ext.unit_generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));

        let mut checked = 0;
        for n in 0..=nn {
            for s in 0..=ss {
                let b = Bidegree::n_s(n, s);
                let target = b + h0.degree();
                if ext.cohomology(b).is_none() || ext.cohomology(target).is_none() {
                    continue;
                }
                for i in 0..ext.dimension(b) {
                    let y = ext.generator(BidegreeGenerator::new(b, i));
                    let via_cup = ext.project(&fm.cup(&h0, &ext.lift(&y)));
                    let via_mult = ext.multiply(&y, &h0);
                    assert_eq!(via_cup, via_mult, "h0 ∪ y != y · h0 at {b:?}#{i}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no cup products checked");
    }

    #[test]
    fn field_massey_dga_matches_direct_c2() {
        // The cochain-DGA field-trick Massey ⟨h0, h1, c⟩ on Ext(C2, k) agrees with the direct
        // minimal resolution — the bracket the chain-map/null-homotopy path loses to non-minimality.
        use sseq::coordinates::BidegreeGenerator;

        let (nn, ss) = (8, 5);
        let s2 = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let m = finite_module(&s2.algebra(), "C2", nn + ss + 8);
        let fm = FieldMassey::new(Arc::clone(&s2), m);
        fm.compute_through_bidegree(Bidegree::n_s(nn, ss));

        let c2 = Arc::new(construct_standard::<false, _, _>("C2", None).unwrap());
        c2.compute_through_stem(Bidegree::n_s(nn + 1, ss + 3));
        let direct = ExtAlgebra::new(Arc::clone(&c2), Arc::clone(&s2));

        let ext = fm.ext();
        let h0 = ext.unit_generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));
        let h1 = ext.unit_generator(BidegreeGenerator::new(Bidegree::n_s(1, 1), 0));
        let d_h0 = direct.unit_generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));
        let d_h1 = direct.unit_generator(BidegreeGenerator::new(Bidegree::n_s(1, 1), 0));

        // The classic bracket ⟨h0, h1, c⟩ at c = (2, 2): the null-homotopy path read 0 here.
        let cd = Bidegree::n_s(2, 2);
        let c_field = ext.generator(BidegreeGenerator::new(cd, 0));
        let c_direct = direct.generator(BidegreeGenerator::new(cd, 0));
        let fb = fm
            .massey(&h0, &h1, &c_field)
            .expect("field ⟨h0,h1,c⟩ defined");
        let db = direct
            .massey(&d_h0, &d_h1, &c_direct)
            .expect("direct ⟨h0,h1,c⟩ defined");
        assert_eq!(fb.degree, db.degree, "bracket bidegree");
        assert_eq!(
            fb.coset.linear_part().dimension(),
            db.coset.linear_part().dimension(),
            "indeterminacy dimension"
        );
        // (4,3) is 1-dimensional with indet 0, so the representative's vanishing is basis-free.
        assert_eq!(
            fb.coset.offset().is_zero(),
            db.coset.offset().is_zero(),
            "⟨h0,h1,c⟩ vanishing disagrees: field={:?} direct={:?}",
            fb.coset.offset(),
            db.coset.offset()
        );
        assert!(
            !fb.coset.offset().is_zero(),
            "field ⟨h0,h1,c⟩ should be nonzero (the bracket the null-homotopy path lost)"
        );
    }

    #[test]
    fn tensor_resolution_is_a_complex_rp_inf() {
        // ∂ ∘ ∂ = 0 for the *infinite* module RP^∞, whose nontrivial higher action exercises the
        // full antipode-Hopf differential (including the identity-operation term dropped by
        // `RealProjectiveSpace`'s `act_on_basis` unless special-cased).
        use algebra::module::{RealProjectiveSpace, homomorphism::ModuleHomomorphism};

        let (ss, t_max) = (7, 20);
        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let rp = Arc::new(RealProjectiveSpace::new(k_res.algebra(), 1, None, false));
        rp.compute_basis(t_max);
        let q = TensorResolution::new(Arc::clone(&k_res), rp);
        q.compute_through_bidegree(Bidegree::s_t(ss, t_max));

        for s in 2..=ss {
            let d_s = q.differential(s);
            let d_prev = q.differential(s - 1);
            let q_s = q.module(s);
            let q_prev2 = q.module(s - 2);
            for t in 0..=t_max {
                for i in 0..q_s.number_of_gens_in_degree(t) {
                    let dx = d_s.output(t, i);
                    let mut ddx = FpVector::new(TWO, q_prev2.dimension(t));
                    d_prev.apply(ddx.as_slice_mut(), 1, t, dx.as_slice());
                    assert!(ddx.is_zero(), "∂² ≠ 0 at s = {s}, t = {t}, gen {i}");
                }
            }
        }
    }

    #[test]
    #[ignore = "blocked by a MATHEMATICAL obstruction, not minimality: H*(RP^∞) admits several \
                secondary (B-module) structures, and the secondary machinery assumes the `zero` \
                one — which does not lift for RP^∞ (see secondary_zero_structure_fails_for_rp_inf; \
                the DIRECT minimal resolution fails identically at (10,3)). The minimality \
                hypothesis itself IS removed (hit_generator flag + the act empty-block fix let the \
                secondary machinery run on the non-minimal Q•, verified for C2). Determining the \
                geometrically correct secondary structure of RP^∞ is separate theoretical work."]
    fn field_d2_matches_direct_rp_inf() {
        // The payoff (blocked, see #[ignore]): the Adams d2 on `Ext(RP^∞, k)` for the *infinite*
        // module RP^∞, via the field trick, should agree with the direct minimal resolution's
        // secondary d2. Ranks of the outgoing d2 match at every bidegree, E2 dims agree.
        use algebra::module::RealProjectiveSpace;
        use fp::matrix::Matrix;
        use sseq::coordinates::BidegreeGenerator;

        use crate::ext_algebra::SecondaryExtAlgebra;

        let (nn, ss) = (10, 5);

        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let rp = Arc::new(RealProjectiveSpace::new(k_res.algebra(), 1, None, false));
        rp.compute_basis(nn + ss + 8);
        let field = FieldResolutionSecondary::new(k_res, rp);
        field.compute_through_stem(Bidegree::n_s(nn, ss));

        let direct_res = Arc::new(construct_standard::<false, _, _>("RP_inf", None).unwrap());
        direct_res.compute_through_stem(Bidegree::n_s(nn + 1, ss + 3));
        let direct_e2 = Arc::new(ExtAlgebra::new(
            Arc::clone(&direct_res),
            Arc::clone(&direct_res),
        ));
        let direct_sec = SecondaryExtAlgebra::new(Arc::clone(&direct_e2));
        direct_sec.extend_all();

        let rank_of =
            |dim: usize, target_dim: usize, d2_of: &mut dyn FnMut(usize) -> FpVector| -> usize {
                if dim == 0 || target_dim == 0 {
                    return 0;
                }
                let rows: Vec<FpVector> = (0..dim).map(&mut *d2_of).collect();
                Matrix::from_rows(TWO, rows, target_dim).row_reduce()
            };

        let mut compared = 0;
        for n in 1..=nn {
            for s in 1..=ss {
                let b = Bidegree::n_s(n, s);
                let target = b + Bidegree::n_s(-1, 2);
                let (Some(fd), Some(ftd)) = (
                    field.cohomology_dimension(b),
                    field.cohomology_dimension(target),
                ) else {
                    continue;
                };
                assert_eq!(fd, direct_e2.dimension(b), "E2 dim mismatch at {b:?}");
                assert_eq!(
                    ftd,
                    direct_e2.dimension(target),
                    "E2 dim mismatch at target {target:?}"
                );

                let field_rank = rank_of(fd, ftd, &mut |i| {
                    field
                        .d2(&field.ext().generator(BidegreeGenerator::new(b, i)))
                        .map(BidegreeElement::into_vec)
                        .unwrap_or_else(|| FpVector::new(TWO, ftd))
                });
                let direct_rank = rank_of(fd, ftd, &mut |i| {
                    direct_sec
                        .d2(&direct_e2.generator(BidegreeGenerator::new(b, i)))
                        .map(BidegreeElement::into_vec)
                        .unwrap_or_else(|| FpVector::new(TWO, ftd))
                });
                assert_eq!(field_rank, direct_rank, "d2 rank mismatch at {b:?}");
                compared += 1;
            }
        }
        assert!(compared > 0, "no bidegrees compared");
    }

    #[test]
    fn secondary_zero_structure_fails_for_rp_inf() {
        // The obstruction blocking infinite-module d2 is MATHEMATICAL, not the minimality
        // hypothesis: H*(RP^∞) admits several secondary (B-module) structures, and the machinery
        // assumes the `zero` one — which does not lift for RP^∞. This is a property of RP^∞ itself,
        // independent of the field trick: even the DIRECT *minimal* resolution's secondary lift
        // fails, at (n, s) = (10, 3). (Contrast the sphere and C2, whose zero structure lifts.)
        //
        // We drive the secondary homotopies exactly as `compute_homotopies` does — but stop before
        // the failing step — then compute it via the fallible path and check it reports the lift
        // failure rather than silently producing a wrong d2. (Mirrors `secondary::cofib_h4`.)
        use crate::secondary::SecondaryLift;

        let res = Arc::new(construct_standard::<false, _, _>("RP_inf", None).unwrap());
        res.compute_through_stem(Bidegree::n_s(12, 6));
        let lift = crate::secondary::SecondaryResolution::new(Arc::clone(&res));

        let failing = Bidegree::n_s(10, 3);

        lift.initialize_homotopies();
        lift.compute_composites();
        lift.compute_intermediates();
        let shift = lift.shift();
        {
            let h = &lift.homotopies()[shift.s()];
            h.homotopies.extend_by_zero(h.composites.max_degree());
        }
        let min_t = lift.homotopies()[shift.s()].homotopies.min_degree();
        let s_range = lift.homotopies().range();
        let min = Bidegree::s_t(s_range.start + 1, min_t);
        let max = lift.max().restrict(s_range.end);
        sseq::coordinates::iter_s_t(
            &|b| {
                if b.s() > failing.s() || (b.s() == failing.s() && b.t() >= failing.t()) {
                    return b.t()..b.t() + 1;
                }
                lift.compute_homotopy_step(b)
            },
            min,
            max,
        );

        let result = lift.try_compute_homotopy_step(failing);
        assert!(
            result.is_err(),
            "expected RP^∞'s zero secondary structure to fail to lift"
        );
        assert!(
            result.unwrap_err().to_string().contains("Failed to lift"),
            "expected a lift failure at {failing}"
        );
    }
}

#[cfg(test)]
mod heavy_tests {
    use std::{sync::Arc, time::Instant};

    use fp::{matrix::Matrix, prime::TWO, vector::FpVector};
    use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

    use super::{FieldResolutionSecondary, tests::finite_module_pub};
    use crate::{
        chain_complex::ChainComplex,
        ext_algebra::{ExtAlgebra, SecondaryExtAlgebra},
        utils::construct_standard,
    };

    /// The field-trick Adams d2 for C2 matches the direct minimal resolution's d2 at *every*
    /// bidegree up to **stem 60** (verified: 1080 bidegrees, 21 with nonzero d2, 0 mismatches).
    /// `#[ignore]`d because the non-minimal Q• secondary is expensive (~4.5 min); run with
    /// `cargo test --release -- --ignored field_d2_matches_direct_c2_stem60 --nocapture`.
    #[test]
    #[ignore = "heavy (~4.5 min): field-trick C2 d2 vs direct up to stem 60"]
    fn field_d2_matches_direct_c2_stem60() {
        let (nn, ss): (i32, i32) = (60, 18);
        let t0 = Instant::now();

        let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let m = finite_module_pub(&k_res.algebra(), "C2", nn + ss + 8);
        let field = FieldResolutionSecondary::new(k_res, m);
        field.compute_through_stem(Bidegree::n_s(nn, ss));
        eprintln!("field secondary computed in {:.1?}", t0.elapsed());

        let t1 = Instant::now();
        let direct_res = Arc::new(construct_standard::<false, _, _>("C2", None).unwrap());
        direct_res.compute_through_stem(Bidegree::n_s(nn + 1, ss + 3));
        let direct_e2 = Arc::new(ExtAlgebra::new(
            Arc::clone(&direct_res),
            Arc::clone(&direct_res),
        ));
        let direct_sec = SecondaryExtAlgebra::new(Arc::clone(&direct_e2));
        direct_sec.extend_all();
        eprintln!("direct secondary computed in {:.1?}", t1.elapsed());

        let rank_of = |dim: usize, tdim: usize, f: &mut dyn FnMut(usize) -> FpVector| -> usize {
            if dim == 0 || tdim == 0 {
                return 0;
            }
            Matrix::from_rows(TWO, (0..dim).map(&mut *f).collect(), tdim).row_reduce()
        };

        let mut compared = 0;
        let mut nonzero_d2 = 0;
        let mut mismatches = 0;
        for n in 1..=nn {
            for s in 1..=ss {
                let b = Bidegree::n_s(n, s);
                let target = b + Bidegree::n_s(-1, 2);
                let (Some(fd), Some(ftd)) = (
                    field.cohomology_dimension(b),
                    field.cohomology_dimension(target),
                ) else {
                    continue;
                };
                if fd != direct_e2.dimension(b) {
                    eprintln!(
                        "E2 dim mismatch at {b:?}: field {fd} direct {}",
                        direct_e2.dimension(b)
                    );
                    mismatches += 1;
                    continue;
                }
                let fr = rank_of(fd, ftd, &mut |i| {
                    field
                        .d2(&field.ext().generator(BidegreeGenerator::new(b, i)))
                        .map(BidegreeElement::into_vec)
                        .unwrap_or_else(|| FpVector::new(TWO, ftd))
                });
                let dr = rank_of(
                    direct_e2.dimension(b),
                    direct_e2.dimension(target),
                    &mut |i| {
                        direct_sec
                            .d2(&direct_e2.generator(BidegreeGenerator::new(b, i)))
                            .map(BidegreeElement::into_vec)
                            .unwrap_or_else(|| FpVector::new(TWO, direct_e2.dimension(target)))
                    },
                );
                if fr != dr {
                    eprintln!("d2 RANK mismatch at {b:?}: field {fr} direct {dr}");
                    mismatches += 1;
                }
                if fr > 0 {
                    nonzero_d2 += 1;
                }
                compared += 1;
            }
        }
        eprintln!(
            "compared {compared} bidegrees, {nonzero_d2} with nonzero d2, {mismatches} mismatches"
        );
        assert_eq!(
            mismatches, 0,
            "field-trick d2 disagrees with direct somewhere"
        );
    }
}
