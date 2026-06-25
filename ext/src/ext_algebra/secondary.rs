//! The secondary ($d_2$) layer of [`ExtAlgebra`].
//!
//! [`SecondaryExtAlgebra`] composes an [`ExtAlgebra`] with the secondary resolutions of `M` and
//! the unit `k`, and exposes:
//! - the secondary differential [`d2`](SecondaryExtAlgebra::d2) (and the survival check
//!   [`survives`](SecondaryExtAlgebra::survives)),
//! - the $E_3$-page data [`page_data`](SecondaryExtAlgebra::page_data), and
//! - the $\Mod_{C\lambda^2}$ secondary product
//!   [`secondary_multiply_into`](SecondaryExtAlgebra::secondary_multiply_into).
//!
//! These wrap [`SecondaryResolution`] and [`SecondaryResolutionHomomorphism`]; no new linear
//! algebra is implemented here. The layer is split out from [`ExtAlgebra`] because the secondary
//! machinery requires `CC::Algebra: PairAlgebra`, a bound the primary layer does not impose.

use std::sync::{Arc, OnceLock};

use algebra::{module::Module, pair_algebra::PairAlgebra};
use dashmap::DashMap;
use fp::{
    matrix::{Matrix, Subquotient, Subspace},
    prime::Prime,
    vector::FpVector,
};
use itertools::Itertools;
use sseq::coordinates::{Bidegree, BidegreeElement};

use super::ExtAlgebra;
use crate::{
    chain_complex::{ChainHomotopy, FreeChainComplex},
    resolution_homomorphism::ResolutionHomomorphism,
    secondary::{
        LAMBDA_BIDEGREE, SecondaryChainHomotopy, SecondaryLift, SecondaryResolution,
        SecondaryResolutionHomomorphism,
    },
};

/// A single secondary product `x · y` in $\Mod_{C\lambda^2}$, where `y` is an $E_3$-surviving
/// class. See [`SecondaryExtAlgebra::secondary_multiply_into`].
pub struct SecondaryProduct {
    /// The multiplicand: an $E_3$-surviving generator of the unit at the queried bidegree `b`.
    pub source: BidegreeElement,
    /// The $\Ext$ part of the product, in bidegree `b + x.degree()`.
    pub ext_part: FpVector,
    /// The $\lambda$ part of the product, in bidegree `b + x.degree() + LAMBDA_BIDEGREE`, already
    /// reduced by the image of $d_2$.
    pub lambda_part: FpVector,
}

/// The secondary layer over an [`ExtAlgebra`]: the $d_2$ differential and the $\Mod_{C\lambda^2}$
/// product. See the [module documentation](self).
pub struct SecondaryExtAlgebra<CC: FreeChainComplex>
where
    CC::Algebra: PairAlgebra,
{
    alg: Arc<ExtAlgebra<CC>>,
    res_lift: Arc<SecondaryResolution<CC>>,
    /// `Arc`-shared with `res_lift` when `M == k`.
    unit_lift: Arc<SecondaryResolution<CC>>,
    /// $E_3$ page of the resolution, filled by [`extend_all`](Self::extend_all).
    res_sseq: OnceLock<Arc<sseq::Sseq<2, sseq::Adams>>>,
    /// $E_3$ page of the unit, filled by [`extend_all`](Self::extend_all).
    unit_sseq: OnceLock<Arc<sseq::Sseq<2, sseq::Adams>>>,
    /// Secondary lift of the multiplication map, cached per multiplier class `(degree, coords)`.
    secondary_products: DashMap<(Bidegree, Vec<u32>), Arc<SecondaryResolutionHomomorphism<CC, CC>>>,
}

impl<CC: FreeChainComplex> SecondaryExtAlgebra<CC>
where
    CC::Algebra: PairAlgebra,
{
    /// Build the secondary layer over `alg`. Construction is cheap; call [`extend_all`](Self::extend_all)
    /// to actually compute the secondary resolutions and $E_3$ pages.
    pub fn new(alg: Arc<ExtAlgebra<CC>>) -> Self {
        let res_lift = Arc::new(SecondaryResolution::new(Arc::clone(alg.resolution())));
        let unit_lift = if alg.is_unit() {
            Arc::clone(&res_lift)
        } else {
            Arc::new(SecondaryResolution::new(Arc::clone(alg.unit())))
        };
        Self {
            alg,
            res_lift,
            unit_lift,
            res_sseq: OnceLock::new(),
            unit_sseq: OnceLock::new(),
            secondary_products: DashMap::new(),
        }
    }

    /// Extend the secondary resolutions as far as the underlying resolutions allow, then compute
    /// the $E_3$ pages. Must be called before [`d2`](Self::d2), [`page_data`](Self::page_data) or
    /// [`secondary_multiply_into`](Self::secondary_multiply_into).
    pub fn extend_all(&self) {
        self.res_lift.extend_all();
        if !self.alg.is_unit() {
            self.unit_lift.extend_all();
        }

        let res_sseq = Arc::new(self.res_lift.e3_page());
        let unit_sseq = if self.alg.is_unit() {
            Arc::clone(&res_sseq)
        } else {
            Arc::new(self.unit_lift.e3_page())
        };
        let _ = self.res_sseq.set(res_sseq);
        let _ = self.unit_sseq.set(unit_sseq);
    }

    /// Sharding entry point: compute only the secondary resolution data for filtration `s`,
    /// distributed across machines sharing a save directory (see the `secondary` example docs).
    /// Mirrors [`SecondaryLift::compute_partial`]. Returns before any $E_3$ page is built.
    pub fn compute_partial(&self, s: i32) {
        self.res_lift.compute_partial(s);
        if !self.alg.is_unit() {
            self.unit_lift.compute_partial(s);
        }
    }

    /// The primary [`ExtAlgebra`] this is built on.
    pub fn ext_algebra(&self) -> &Arc<ExtAlgebra<CC>> {
        &self.alg
    }

    fn prime(&self) -> fp::prime::ValidPrime {
        self.alg.prime()
    }

    /// The secondary differential $d_2(x)$, a class in bidegree `(n - 1, s + 2)`.
    ///
    /// Returns `None` if the target bidegree has not been computed (so $d_2$ is unknown). A
    /// computed-but-zero differential is `Some` of a zero class.
    pub fn d2(&self, x: &BidegreeElement) -> Option<BidegreeElement> {
        let b = x.degree();
        let target = b + Bidegree::n_s(-1, 2);
        let res = self.res_lift.underlying();
        if !(b.t() > 0 && res.has_computed_bidegree(target)) {
            return None;
        }

        let target_dim = res.number_of_gens_in_bidegree(target);
        let mut out = FpVector::new(self.prime(), target_dim);

        // `m[i]` is the d2 of the i-th generator of `b`, as a vector at `target`. This is exactly
        // the matrix `SecondaryResolution::e3_page` reads to install d2 differentials.
        let m = self.res_lift.homotopy(b.s() + 2).homotopies.hom_k(b.t());
        if !m.is_empty() && !m[0].is_empty() {
            let p = self.prime().as_u32();
            for (i, c) in x.vec().iter_nonzero() {
                for (k, &v) in m[i].iter().enumerate() {
                    out.add_basis_element(k, (c * v) % p);
                }
            }
        }
        Some(BidegreeElement::new(target, out))
    }

    /// Whether `x` is a $d_2$-cycle (a permanent class through $E_3$). Treats an uncomputed $d_2$
    /// target as "survives" (there is nothing for it to hit).
    pub fn survives(&self, x: &BidegreeElement) -> bool {
        self.d2(x).is_none_or(|d| d.vec().is_zero())
    }

    /// The $E_3$-page subquotient of $\Ext(M, k)$ at bidegree `b`.
    pub fn page_data(&self, b: Bidegree) -> &Subquotient {
        e3_page_data(self.res_sseq.get().expect("call extend_all() first"), b)
    }

    /// The $E_3$-page subquotient of the unit $\Ext(k, k)$ at bidegree `b`.
    pub fn unit_page_data(&self, b: Bidegree) -> &Subquotient {
        e3_page_data(self.unit_sseq.get().expect("call extend_all() first"), b)
    }
}

/// The $E_3$-page subquotient recorded by `sseq` at bidegree `b` (clamped to the last computed
/// page). Promotes the `get_page_data` helper that the secondary examples duplicated.
fn e3_page_data(sseq: &sseq::Sseq<2, sseq::Adams>, b: Bidegree) -> &Subquotient {
    let d = sseq.page_data(b);
    &d[std::cmp::min(3, d.len() - 1)]
}

impl<CC: FreeChainComplex + crate::chain_complex::AugmentedChainComplex> SecondaryExtAlgebra<CC>
where
    CC::Algebra: PairAlgebra,
{
    /// The secondary lift of multiplication by `x`, built and cached per multiplier class. The
    /// returned lift is *not* extended; [`secondary_multiply_into`](Self::secondary_multiply_into)
    /// extends it as needed. Exposed so callers can drive sharded computation
    /// (`lift.underlying().extend_all()` then `lift.compute_partial(s)`).
    pub fn secondary_product_lift(
        &self,
        x: &BidegreeElement,
    ) -> Arc<SecondaryResolutionHomomorphism<CC, CC>> {
        let shift = x.degree();
        let coords: Vec<u32> = x.vec().iter().collect();
        let key = (shift, coords.clone());

        if let Some(map) = self.secondary_products.get(&key) {
            return Arc::clone(&map);
        }

        let name = format!(
            "prod_{}_{}_{}",
            shift.n(),
            shift.s(),
            coords
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("_")
        );
        let underlying = Arc::new(ResolutionHomomorphism::from_class(
            name,
            Arc::clone(self.alg.resolution()),
            Arc::clone(self.alg.unit()),
            shift,
            &coords,
        ));
        let lift = Arc::new(SecondaryResolutionHomomorphism::new(
            Arc::clone(&self.res_lift),
            Arc::clone(&self.unit_lift),
            underlying,
        ));

        Arc::clone(self.secondary_products.entry(key).or_insert(lift).value())
    }

    /// The secondary product of `x` with every $E_3$-surviving class of the unit at bidegree `b`,
    /// computed in $\Mod_{C\lambda^2}$.
    ///
    /// Returns one [`SecondaryProduct`] per surviving generator at `b`; the $\lambda$ part is
    /// already reduced by the image of $d_2$. The caller must have run [`extend_all`](Self::extend_all)
    /// and computed both resolutions far enough.
    pub fn secondary_multiply_into(
        &self,
        x: &BidegreeElement,
        b: Bidegree,
    ) -> Vec<SecondaryProduct> {
        let p = self.prime();
        let shift = x.degree();
        let res_sseq = self.res_sseq.get().expect("call extend_all() first");

        let ext_dim = self.alg.resolution().number_of_gens_in_bidegree(b + shift);
        let lambda_dim = self
            .alg
            .resolution()
            .number_of_gens_in_bidegree(b + shift + LAMBDA_BIDEGREE);

        let page = self.unit_page_data(b);
        let n = page.subspace_dimension();
        if n == 0 {
            return Vec::new();
        }

        let lift = self.secondary_product_lift(x);
        lift.underlying().extend_all();
        lift.extend_all();

        let mut outputs = vec![FpVector::new(p, ext_dim + lambda_dim); n];
        lift.hom_k(
            Some(&**res_sseq),
            b,
            page.subspace_gens(),
            outputs.iter_mut().map(FpVector::as_slice_mut),
        );

        page.subspace_gens()
            .zip(outputs)
            .map(|(g, out)| SecondaryProduct {
                source: BidegreeElement::new(b, g.to_owned()),
                ext_part: out.slice(0, ext_dim).to_owned(),
                lambda_part: out.slice(ext_dim, ext_dim + lambda_dim).to_owned(),
            })
            .collect()
    }

    /// The underlying multiplication map for `class` and its secondary lift, named for
    /// bookkeeping/saving. `source`/`target` are the secondary resolutions the lift bridges.
    fn massey_secondary_map(
        &self,
        source: &Arc<SecondaryResolution<CC>>,
        target: &Arc<SecondaryResolution<CC>>,
        name: &str,
        class: &BidegreeElement,
    ) -> Arc<SecondaryResolutionHomomorphism<CC, CC>> {
        let coords: Vec<u32> = class.vec().iter().collect();
        let map_name = format!(
            "massey_{name}_{}_{}",
            class.degree().n(),
            class.degree().s()
        );
        let underlying = Arc::new(ResolutionHomomorphism::from_class(
            map_name,
            source.underlying(),
            target.underlying(),
            class.degree(),
            &coords,
        ));
        Arc::new(SecondaryResolutionHomomorphism::new(
            Arc::clone(source),
            Arc::clone(target),
            underlying,
        ))
    }

    /// The λ-part map of `class` (one filtration up), bridging the same endpoints as `lift`.
    fn massey_lambda_map(
        &self,
        lift: &Arc<SecondaryResolutionHomomorphism<CC, CC>>,
        name: &str,
        class: &BidegreeElement,
    ) -> Arc<ResolutionHomomorphism<CC, CC>> {
        let coords: Vec<u32> = class.vec().iter().collect();
        let map_name = format!(
            "massey_{name}_lambda_{}_{}",
            class.degree().n(),
            class.degree().s()
        );
        Arc::new(ResolutionHomomorphism::from_class(
            map_name,
            lift.source(),
            lift.target(),
            class.degree(),
            &coords,
        ))
    }

    /// Build the data for the secondary Massey product family $\langle a, b, -\rangle$, computed in
    /// $\Mod_{C\lambda^2}$.
    ///
    /// `a` is taken in $\Ext(M, k)$ and `b` in $\Ext(k, k)$; the third factor will range over
    /// $\Ext(M, k)$. Each of `a` and `b` may carry a $\lambda$ part — its lift into
    /// $\Mod_{C\lambda^2}$ — supplied as a class one filtration up (at `degree + LAMBDA_BIDEGREE`);
    /// pass `None` for a zero $\lambda$ part. This assumes `a · b = 0` in $\Mod_{C\lambda^2}$; it is
    /// **not** verified.
    ///
    /// Construction builds the underlying multiplication maps, their secondary lifts, the
    /// null-homotopy of `a · b`, and the secondary chain homotopy, extending the resolutions as far
    /// as they allow. Drive the result with [`SecondaryMassey::extend`] (or
    /// [`compute_partial`](SecondaryMassey::compute_partial) for sharded computation) before reading
    /// off [`family`](SecondaryMassey::family).
    pub fn secondary_massey(
        &self,
        a: &BidegreeElement,
        a_lambda: Option<&BidegreeElement>,
        b: &BidegreeElement,
        b_lambda: Option<&BidegreeElement>,
    ) -> SecondaryMassey<CC> {
        let p = self.prime();
        let resolution = Arc::clone(self.alg.resolution());
        let unit = Arc::clone(self.alg.unit());
        let is_unit = self.alg.is_unit();

        // Reach the λ bidegree of each input, as the example's `get_hom` does per map. (We use a
        // rectangle here rather than a stem, since `compute_through_stem` is concrete-only; for the
        // small input bidegrees this is a no-op when the resolutions are already computed.)
        resolution.compute_through_bidegree(a.degree() + LAMBDA_BIDEGREE);
        unit.compute_through_bidegree(b.degree() + LAMBDA_BIDEGREE);

        // The combined (Ext + λ) class of `b`, as `product_nullhomotopy` expects. Built from the
        // input elements before they are turned into maps below.
        let b_class = {
            let ext_dim = unit.number_of_gens_in_bidegree(b.degree());
            let lambda_dim = unit.number_of_gens_in_bidegree(b.degree() + LAMBDA_BIDEGREE);
            let mut class = FpVector::new(p, ext_dim + lambda_dim);
            for (i, v) in b.vec().iter_nonzero() {
                class.set_entry(i, v);
            }
            if let Some(bl) = b_lambda {
                for (i, v) in bl.vec().iter_nonzero() {
                    class.set_entry(ext_dim + i, v);
                }
            }
            class
        };

        // `a: res → unit`, `b: unit → unit`.
        let a_lift = self.massey_secondary_map(&self.res_lift, &self.unit_lift, "a", a);
        let b_lift = self.massey_secondary_map(&self.unit_lift, &self.unit_lift, "b", b);
        let a_lambda = a_lambda.map(|c| self.massey_lambda_map(&a_lift, "a", c));
        let b_lambda = b_lambda.map(|c| self.massey_lambda_map(&b_lift, "b", c));

        let shift = Bidegree::s_t(
            (a_lift.underlying().shift + b_lift.underlying().shift).s(),
            (a_lift.shift() + b_lift.shift()).t(),
        );
        let b_shift = b_lift.underlying().shift;

        // Extend the resolutions, then the secondary resolutions, then the $E_3$ pages.
        if !is_unit {
            let res_max = Bidegree::n_s(
                resolution.module(0).max_computed_degree(),
                resolution.next_homological_degree() - 1,
            );
            unit.compute_through_bidegree(res_max - a_lift.underlying().shift);
        }
        if is_unit {
            self.res_lift.extend_all();
        } else {
            maybe_rayon::join(
                || self.res_lift.extend_all(),
                || self.unit_lift.extend_all(),
            );
        }

        let res_sseq = Arc::new(self.res_lift.e3_page());
        let unit_sseq = if is_unit {
            Arc::clone(&res_sseq)
        } else {
            Arc::new(self.unit_lift.e3_page())
        };

        // Extend the homomorphisms.
        maybe_rayon::scope(|s| {
            s.spawn(|_| {
                a_lift.underlying().extend_all();
                a_lift.extend_all();
            });
            s.spawn(|_| {
                b_lift.underlying().extend_all();
                b_lift.extend_all();
            });
            if let Some(a_lambda) = &a_lambda {
                s.spawn(|_| a_lambda.extend_all());
            }
            if let Some(b_lambda) = &b_lambda {
                s.spawn(|_| b_lambda.extend_all());
            }
        });

        // The null-homotopy of `a · b`, installed as the first homotopy of the chain homotopy.
        let chain_homotopy = Arc::new(ChainHomotopy::new(a_lift.underlying(), b_lift.underlying()));
        chain_homotopy.initialize_homotopies((b_shift + a_lift.underlying().shift).s());
        {
            let v = a_lift.product_nullhomotopy(
                a_lambda.as_deref(),
                &res_sseq,
                b_shift,
                b_class.as_slice(),
            );
            let homotopy = chain_homotopy.homotopy(b_shift.s() + a_lift.underlying().shift.s() - 1);
            let htpy_source = a_lift.shift() + b_shift;
            homotopy.extend_by_zero(htpy_source.t() - 1);
            homotopy.add_generators_from_rows(
                htpy_source.t(),
                v.into_iter()
                    .map(|x| FpVector::from_slice(p, &[x]))
                    .collect(),
            );
        }
        chain_homotopy.extend_all();

        let ch_lift = SecondaryChainHomotopy::new(
            Arc::clone(&a_lift),
            Arc::clone(&b_lift),
            a_lambda.clone(),
            b_lambda.clone(),
            Arc::clone(&chain_homotopy),
        );

        // `a_lambda` and `b_class` are only needed to build the chain homotopy above; `ch_lift`
        // keeps the λ-part maps alive past this point.
        SecondaryMassey {
            alg: Arc::clone(&self.alg),
            a_lift,
            b_lift,
            b_lambda,
            chain_homotopy,
            ch_lift,
            unit_sseq,
            shift,
            b_shift,
        }
    }
}

/// One entry of the secondary Massey product family $\langle a, b, -\rangle$ in
/// $\Mod_{C\lambda^2}$, as returned by [`SecondaryMassey::family`].
///
/// The bracket is computed up to a sign. The third factor `[third_factor] + λ·third_factor_lambda`
/// is an $E_3$-surviving class with `b · (third factor) = 0`; the bracket representative is
/// `[representative] + λ·representative_lambda`.
pub struct SecondaryMasseyResult {
    /// The $\Ext$ part of the third factor, an $E_3$-surviving class at bidegree `c`.
    pub third_factor: BidegreeElement,
    /// The $\lambda$ part of the third factor, at `third_factor.degree() + LAMBDA_BIDEGREE`.
    pub third_factor_lambda: FpVector,
    /// The $\Ext$ part of the bracket representative, at `c + shift - (1, 0)`.
    pub representative: BidegreeElement,
    /// The $\lambda$ part of the bracket representative, at
    /// `representative.degree() + LAMBDA_BIDEGREE`.
    pub representative_lambda: FpVector,
}

/// The data backing a secondary Massey product family $\langle a, b, -\rangle$, built by
/// [`SecondaryExtAlgebra::secondary_massey`]. Drive it with [`extend`](Self::extend) (or
/// [`compute_partial`](Self::compute_partial)) and read the brackets off with
/// [`family`](Self::family).
pub struct SecondaryMassey<CC: FreeChainComplex>
where
    CC::Algebra: PairAlgebra,
{
    alg: Arc<ExtAlgebra<CC>>,
    a_lift: Arc<SecondaryResolutionHomomorphism<CC, CC>>,
    b_lift: Arc<SecondaryResolutionHomomorphism<CC, CC>>,
    b_lambda: Option<Arc<ResolutionHomomorphism<CC, CC>>>,
    chain_homotopy: Arc<ChainHomotopy<CC, CC, CC>>,
    ch_lift: SecondaryChainHomotopy<CC, CC, CC>,
    /// $E_3$ page of the unit; the only page the readout needs.
    unit_sseq: Arc<sseq::Sseq<2, sseq::Adams>>,
    /// `a.degree() + b.degree()`, the total bidegree shift of the bracket.
    shift: Bidegree,
    /// `b.degree()`.
    b_shift: Bidegree,
}

impl<CC: FreeChainComplex> SecondaryMassey<CC>
where
    CC::Algebra: PairAlgebra,
{
    /// Sharding entry point: compute only the secondary chain homotopy data for filtration `s`,
    /// distributed across machines sharing a save directory (see the `secondary_massey` example).
    pub fn compute_partial(&self, s: i32) {
        self.ch_lift.compute_partial(s);
    }

    /// Extend the secondary chain homotopy as far as the underlying data allows. Must be called
    /// before [`family`](Self::family).
    pub fn extend(&self) {
        self.ch_lift.extend_all();
    }

    /// Read off the family of secondary Massey products $\langle a, b, -\rangle$ over every
    /// $E_3$-surviving third factor, in $\Mod_{C\lambda^2}$, up to a sign.
    ///
    /// One [`SecondaryMasseyResult`] per surviving generator at each computed bidegree. Requires
    /// [`extend`](Self::extend) to have completed.
    pub fn family(&self) -> Vec<SecondaryMasseyResult> {
        let p = self.alg.prime();
        let resolution = self.alg.resolution();
        let unit = self.alg.unit();
        let shift = self.shift;
        let b_shift = self.b_shift;
        let chain_homotopy = &self.chain_homotopy;
        let ch_lift = &self.ch_lift;
        let a = &self.a_lift;
        let b = &self.b_lift;
        let b_lambda = self.b_lambda.as_deref();
        let unit_sseq = &self.unit_sseq;

        let h_0 = ch_lift.algebra().p_tilde();

        let mut results = Vec::new();

        // Iterate through the multiplicand.
        for c in unit.iter_stem() {
            if !resolution.has_computed_bidegree(c + shift - Bidegree::s_t(2, 0))
                || !resolution.has_computed_bidegree(c + shift + Bidegree::s_t(0, 1))
            {
                continue;
            }

            let source = c + shift - Bidegree::s_t(1, 0);

            let source_num_gens = resolution.number_of_gens_in_bidegree(source);
            let source_lambda_num_gens =
                resolution.number_of_gens_in_bidegree(source + LAMBDA_BIDEGREE);

            if source_num_gens + source_lambda_num_gens == 0 {
                continue;
            }

            // The kernel of multiplication by `b` on the $E_3$ page.
            let target_num_gens = unit.number_of_gens_in_bidegree(c);
            let target_lambda_num_gens = unit.number_of_gens_in_bidegree(c + LAMBDA_BIDEGREE);
            let target_all_gens = target_num_gens + target_lambda_num_gens;

            let prod_num_gens = unit.number_of_gens_in_bidegree(c + b_shift);
            let prod_lambda_num_gens =
                unit.number_of_gens_in_bidegree(c + b_shift + LAMBDA_BIDEGREE);
            let prod_all_gens = prod_num_gens + prod_lambda_num_gens;

            let e3_kernel = {
                let target_page_data = e3_page_data(unit_sseq, c);
                let target_lambda_page_data = e3_page_data(unit_sseq, c + LAMBDA_BIDEGREE);
                let product_lambda_page_data =
                    e3_page_data(unit_sseq, c + b_shift + LAMBDA_BIDEGREE);

                // We first compute elements whose product vanishes mod lambda, and later see what
                // the possible lifts are. We do it this way to avoid Z/p^2 problems.
                let e2_kernel: Subspace = {
                    let mut product_matrix = Matrix::new(
                        p,
                        target_page_data.subspace_dimension(),
                        target_num_gens + prod_num_gens,
                    );

                    let m0 = Matrix::from_vec(
                        p,
                        &b.underlying()
                            .get_map(c.s() + b.underlying().shift.s())
                            .hom_k(c.t()),
                    );
                    for (g, mut out) in target_page_data
                        .subspace_gens()
                        .zip_eq(product_matrix.iter_mut())
                    {
                        out.slice_mut(prod_num_gens, prod_num_gens + target_num_gens)
                            .add(g, 1);
                        for (i, v) in g.iter_nonzero() {
                            out.slice_mut(0, prod_num_gens).add(m0.row(i), v);
                        }
                    }
                    product_matrix.row_reduce();
                    product_matrix.compute_kernel(prod_num_gens)
                };

                // Now compute the e3 kernel.
                {
                    // First add the lifts from Ext.
                    let e2_ker_dim = e2_kernel.dimension();
                    let mut product_matrix = Matrix::new(
                        p,
                        e2_ker_dim + target_lambda_page_data.quotient_dimension(),
                        target_all_gens + prod_all_gens,
                    );

                    b.hom_k_with(
                        b_lambda,
                        Some(&**unit_sseq),
                        c,
                        e2_kernel.basis(),
                        product_matrix
                            .slice_mut(0, e2_ker_dim, 0, prod_all_gens)
                            .iter_mut(),
                    );
                    for (v, mut t) in e2_kernel.basis().zip(product_matrix.iter_mut()) {
                        t.slice_mut(prod_all_gens, prod_all_gens + target_num_gens)
                            .assign(v);
                    }

                    // Now add the lambda multiples.
                    let m = Matrix::from_vec(
                        p,
                        &b.underlying()
                            .get_map(b_shift.s() + c.s() + 1)
                            .hom_k(c.t() + 1),
                    );

                    let mut count = 0;
                    for (i, &v) in target_lambda_page_data.quotient_pivots().iter().enumerate() {
                        if v >= 0 {
                            continue;
                        }
                        let mut row = product_matrix.row_mut(e2_ker_dim + count as usize);
                        row.add_basis_element(prod_all_gens + target_num_gens + i, 1);
                        row.slice_mut(prod_num_gens, prod_all_gens).add(m.row(i), 1);
                        product_lambda_page_data
                            .reduce_by_quotient(row.slice_mut(prod_num_gens, prod_all_gens));
                        count += 1;
                    }

                    product_matrix.row_reduce();
                    product_matrix.compute_kernel(prod_all_gens)
                }
            };

            if e3_kernel.dimension() == 0 {
                continue;
            }

            let m0 = chain_homotopy.homotopy(source.s()).hom_k(c.t());
            let mt = Matrix::from_vec(p, &chain_homotopy.homotopy(source.s() + 1).hom_k(c.t() + 1));
            let m1 = Matrix::from_vec(
                p,
                &ch_lift.homotopies()[source.s() + 1].homotopies.hom_k(c.t()),
            );
            let mp = Matrix::from_vec(
                p,
                &resolution
                    .filtration_one_product(1, h_0, Bidegree::s_t(source.s(), c.t() + shift.t()))
                    .unwrap(),
            );
            let ma = a
                .underlying()
                .get_map(source.s())
                .hom_k(c.t() + b_shift.t());
            let mb = b.underlying().get_map(c.s() + b_shift.s()).hom_k(c.t());

            for g in e3_kernel.iter() {
                let third_factor =
                    BidegreeElement::new(c, g.restrict(0, target_num_gens).to_owned());
                let third_factor_lambda = g.restrict(target_num_gens, target_all_gens).to_owned();

                let mut scratch0: Vec<u32> = vec![0; source_num_gens];
                let mut scratch1 = FpVector::new(p, source_lambda_num_gens);

                // First deal with the null-homotopy of ab.
                for (i, v) in g.restrict(0, target_num_gens).iter_nonzero() {
                    scratch0
                        .iter_mut()
                        .zip_eq(&m0[i])
                        .for_each(|(a, b)| *a += v * b);
                    scratch1.as_slice_mut().add(m1.row(i), v);
                }
                for (i, v) in g.restrict(target_num_gens, target_all_gens).iter_nonzero() {
                    scratch1.as_slice_mut().add(mt.row(i), v);
                }
                // Now do the -1 part of the null-homotopy of bc.
                {
                    let sign = p * p - 1;
                    let out = b.product_nullhomotopy(b_lambda, unit_sseq, c, g);
                    for (i, v) in out.iter_nonzero() {
                        scratch0
                            .iter_mut()
                            .zip_eq(&ma[i])
                            .for_each(|(a, b)| *a += v * b * sign);
                    }
                }

                for (i, v) in scratch0.iter().enumerate() {
                    let extra = *v / p;
                    scratch1.as_slice_mut().add(mp.row(i), extra % p);
                }

                let representative = BidegreeElement::new(
                    source,
                    FpVector::from_slice(p, &scratch0.iter().map(|x| *x % p).collect::<Vec<_>>()),
                );

                // Then deal with the rest of the null-homotopy of bc. This is just the
                // null-homotopy of 2.
                let mut scratch0 = vec![0u32; prod_num_gens];
                for (i, v) in g.restrict(0, target_num_gens).iter_nonzero() {
                    scratch0
                        .iter_mut()
                        .zip_eq(&mb[i])
                        .for_each(|(a, b)| *a += v * b);
                }
                for (i, v) in scratch0.iter().enumerate() {
                    let extra = (*v / p) % p;
                    if extra == 0 {
                        continue;
                    }
                    for gen_idx in 0..source_lambda_num_gens {
                        let m = a.underlying().get_map((source + LAMBDA_BIDEGREE).s());
                        let dx = m.output((source + LAMBDA_BIDEGREE).t(), gen_idx);
                        let idx = unit.module((c + shift).s()).operation_generator_to_index(
                            1,
                            h_0,
                            (c + shift).t(),
                            i,
                        );
                        scratch1.add_basis_element(gen_idx, dx.entry(idx));
                    }
                }

                results.push(SecondaryMasseyResult {
                    third_factor,
                    third_factor_lambda,
                    representative,
                    representative_lambda: scratch1,
                });
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use sseq::coordinates::BidegreeGenerator;

    use super::*;
    use crate::utils::construct_standard;

    #[test]
    fn test_sphere_d2() {
        let res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        // Far enough to reach the first Adams differential d2(h4) = h0 h3^2 at (14, 3).
        res.compute_through_stem(Bidegree::n_s(16, 6));
        let alg = Arc::new(ExtAlgebra::new(Arc::clone(&res), res, true));
        let sec = SecondaryExtAlgebra::new(alg);
        sec.extend_all();

        let alg = sec.ext_algebra();

        // h_0, h_1, h_2 are permanent cycles.
        for (n, s) in [(0, 1), (1, 1), (3, 1)] {
            let h = alg.generator(BidegreeGenerator::new(Bidegree::n_s(n, s), 0));
            assert!(sec.survives(&h), "h at (n={n}, s={s}) should survive d2");
            assert!(
                sec.d2(&h).is_none_or(|d| d.vec().is_zero()),
                "d2 of a permanent class should vanish"
            );
        }

        // The first Adams differential: d2(h4) = h0 h3^2, the generator of Ext^{3,17} at (14, 3).
        let h4 = alg.generator(BidegreeGenerator::new(Bidegree::n_s(15, 1), 0));
        let d = sec.d2(&h4).expect("d2(h4) target should be computed");
        assert_eq!(d.degree(), Bidegree::n_s(14, 3));
        assert_eq!(alg.dimension(Bidegree::n_s(14, 3)), 1);
        assert!(!d.vec().is_zero(), "d2(h4) = h0 h3^2 should be nonzero");
        assert!(!sec.survives(&h4), "h4 should not survive d2");
    }
}
