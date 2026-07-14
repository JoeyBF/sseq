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

use std::{collections::HashMap, sync::Arc};

use algebra::{Algebra, Bialgebra, module::Module};
use dashmap::DashMap;
use fp::{matrix::Matrix, prime::ValidPrime, vector::FpVector};
use sseq::coordinates::Bidegree;

use super::{ExtAlgebra, ExtDifferential};
use crate::chain_complex::FreeChainComplex;

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

        let p = self.prime();
        let algebra = self.resolution.algebra();
        let rows = self.cochain_basis(s, t); // (d_i, i, α)  — Q(D_s)
        let cols = self.cochain_basis(s + 1, t); // (e_l, l, γ) — Q(D_{s+1})
        let mut matrix = Matrix::new(p, rows.len(), cols.len());

        let module_s = self.resolution.module(s);
        let d_p = self.resolution.differential(s + 1); // P_{s+1} → P_s

        // Row lookup: (generator degree, generator index) → its first row (α = 0). The α rows for a
        // generator are contiguous by construction of `cochain_basis`.
        let mut row_base: HashMap<(i32, usize), usize> = HashMap::new();
        for (r, &(d_i, i, alpha)) in rows.iter().enumerate() {
            if alpha == 0 {
                row_base.insert((d_i, i), r);
            }
        }

        for (c, &(e_l, l, gamma)) in cols.iter().enumerate() {
            // d_P(z_l) ∈ (P_s)_{e_l}.
            let mut dp = FpVector::new(p, module_s.dimension(e_l));
            d_p.apply_to_generator(&mut dp, 1, e_l, l);
            if dp.is_zero() {
                continue;
            }
            let gamma_deg = t - e_l; // |m_γ|

            // For each P_s-generator (d_i, i): extract a_{li}, act χ(a_{li}) on m_γ, scatter to rows.
            for (&(d_i, i), &base) in &row_base {
                let op_deg = e_l - d_i;
                if op_deg < 0 {
                    continue;
                }
                let width = algebra.dimension(op_deg);
                if width == 0 {
                    continue;
                }
                // a_{li} = block of d_P(z_l) at generator (d_i, i), as an algebra element.
                let offset = module_s.generator_offset(e_l, d_i, i);
                let mut a_li = FpVector::new(p, width);
                for q in 0..width {
                    let v = dp.entry(offset + q);
                    if v != 0 {
                        a_li.add_basis_element(q, v);
                    }
                }
                if a_li.is_zero() {
                    continue;
                }
                // χ(a_{li}) · m_γ  ∈ M_{t - d_i}.
                let chi = self.antipode.apply_element(op_deg, a_li);
                let mut acted = FpVector::new(p, self.module.dimension(t - d_i));
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
        Some(matrix)
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
        for degree in 1..=10 {
            for idx in 0..algebra.dimension(degree) {
                let mut acc = FpVector::new(TWO, algebra.dimension(degree));
                // Use the same decompose/coproduct route as the antipode for the full coproduct.
                for (l_deg, l_idx, r_deg, r_idx) in full_coproduct(&algebra, degree, idx) {
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

    /// The full coproduct Δ(x) of a basis element, folding `decompose` atoms' coproducts in A⊗A.
    /// Used only by the Hopf-identity test as an independent check.
    fn full_coproduct(
        algebra: &SteenrodAlgebra,
        degree: i32,
        idx: usize,
    ) -> Vec<(i32, usize, i32, usize)> {
        // Represent Δ as a map (l_deg, l_idx, r_deg, r_idx) → coeff.
        let mut terms: HashMap<(i32, usize, i32, usize), u32> = HashMap::new();
        terms.insert((0, 0, 0, 0), 1);
        let mut left_deg = 0;
        for (a_deg, a_idx) in algebra.decompose(degree, idx) {
            let mut next: HashMap<(i32, usize, i32, usize), u32> = HashMap::new();
            for (&(ll, li, rl, ri), &c) in &terms {
                for (cl, cli, cr, cri) in algebra.coproduct(a_deg, a_idx) {
                    // (ll⊗rl)·(cl⊗cr) = (ll·cl)⊗(rl·cr).
                    let mut left = FpVector::new(TWO, algebra.dimension(ll + cl));
                    algebra.multiply_basis_elements(left.as_slice_mut(), 1, ll, li, cl, cli);
                    let mut right = FpVector::new(TWO, algebra.dimension(rl + cr));
                    algebra.multiply_basis_elements(right.as_slice_mut(), 1, rl, ri, cr, cri);
                    for (lj, lc) in left.iter_nonzero() {
                        for (rj, rc) in right.iter_nonzero() {
                            *next.entry((ll + cl, lj, rl + cr, rj)).or_insert(0) += c * lc * rc;
                        }
                    }
                }
            }
            terms = next;
            left_deg += a_deg;
        }
        let _ = left_deg;
        terms
            .into_iter()
            .filter(|&(_, c)| c % 2 == 1)
            .map(|(k, _)| k)
            .collect()
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
}
