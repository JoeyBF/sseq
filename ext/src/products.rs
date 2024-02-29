use std::sync::Arc;

use algebra::{module::Module, MuAlgebra};
use fp::{
    matrix::{AffineSubspace, Matrix, Subspace},
    prime::ValidPrime,
    vector::FpVector,
};
use maybe_rayon::prelude::*;
use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

use crate::{
    chain_complex::{
        AugmentedChainComplex, BoundedChainComplex, ChainComplex, ChainHomotopy, FreeChainComplex,
    },
    resolution_homomorphism::MuResolutionHomomorphism,
    utils::QueryModuleResolution,
};

type DashMap<K, V> = dashmap::DashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

pub struct ProductStructure {
    p: ValidPrime,
    resolution: Arc<QueryModuleResolution>,
    cache: MuResolutionHomomorphismCache<false, QueryModuleResolution, QueryModuleResolution>,
    multiplication_table: DashMap<(BidegreeGenerator, Bidegree), Matrix>,
    massey_products: DashMap<(BidegreeElement, BidegreeElement, BidegreeElement), AffineSubspace>,
}

impl ProductStructure {
    pub fn new(resolution: Arc<QueryModuleResolution>) -> Self {
        assert!(
            resolution.target().max_s() == 1 && resolution.target().module(0).is_unit(),
            "Product structure not supported for non-unit resolution"
        );
        let cache =
            MuResolutionHomomorphismCache::new(Arc::clone(&resolution), Arc::clone(&resolution));
        Self {
            p: ValidPrime::new(2),
            resolution,
            cache,
            multiplication_table: DashMap::default(),
            massey_products: DashMap::default(),
        }
    }

    pub fn resolution(&self) -> Arc<QueryModuleResolution> {
        Arc::clone(&self.resolution)
    }

    fn iterate_all_vectors<F: FnMut(BidegreeElement) + Send>(&self, mut f: F) {
        self.resolution
            .iter_stem()
            .maybe_par_bridge()
            .for_each(|deg| {
                if deg == Bidegree::zero() {
                    return;
                }
                if !self.resolution.has_computed_bidegree(deg) {
                    return;
                }
                let space =
                    Subspace::entire_space(self.p, self.resolution.number_of_gens_in_bidegree(deg));
                space
                    .iter_all_vectors()
                    // .skip(1)
                    .maybe_par_bridge()
                    .for_each(|vec| {
                        let tracing_span = tracing::info_span!("iterate_all_vectors", deg = %deg);
                        let _tracing_guard = tracing_span.enter();
                        let a = BidegreeElement::new(deg, vec);
                        f(a);
                    });
            });
    }

    pub fn product(
        &self,
        x: &BidegreeElement,
        y: &BidegreeElement,
    ) -> Result<BidegreeElement, String> {
        let tot = x.degree() + y.degree();
        if !self.resolution().has_computed_bidegree(tot) {
            return Err(format!("Bidegree {tot} not computed"));
        }
        let target_dim = self.resolution().number_of_gens_in_bidegree(tot);
        let mut result = FpVector::new(self.p, target_dim);
        for (x_gen, x_coeff) in x.decompose() {
            for (y_gen, y_coeff) in y.decompose() {
                result
                    .as_slice_mut()
                    .add(self.product_gen(x_gen, y_gen)?.vec(), x_coeff * y_coeff);
            }
        }
        Ok(BidegreeElement::new(tot, result))
    }

    pub fn product_matrix(&self, x: Bidegree, y: BidegreeElement) -> Result<Matrix, String> {
        let tot = x + y.degree();
        if !self.resolution.has_computed_bidegree(tot) {
            return Err(format!("Bidegree {tot} not computed"));
        }
        let x_num_gens = self.resolution.number_of_gens_in_bidegree(x);
        let target_dim = self.resolution.number_of_gens_in_bidegree(tot);
        let mut result = Matrix::new(self.p, x_num_gens, target_dim);
        for (idx, row) in result.iter_mut().enumerate() {
            let basis_vec = {
                let mut basis = FpVector::from_slice(self.p, &vec![0; x_num_gens]);
                basis.add_basis_element(idx, 1);
                basis
            };
            let g = BidegreeElement::new(x, basis_vec);
            let prod = self.product(&g, &y)?;
            row.as_slice_mut().add(prod.vec(), 1);
        }
        Ok(result)
    }

    pub fn product_gen(
        &self,
        x: BidegreeGenerator,
        y: BidegreeGenerator,
    ) -> Result<BidegreeElement, String> {
        let tot = x.degree() + y.degree();
        if !self.resolution.has_computed_bidegree(tot) {
            return Err(format!("Bidegree {tot} not computed"));
        }
        if let Some(matrix) = self.multiplication_table.get(&(x, y.degree())) {
            let result_vec = matrix.row(y.idx()).to_owned();
            Ok(BidegreeElement::new(tot, result_vec))
        } else {
            let x_num_gens = self.resolution.number_of_gens_in_bidegree(x.degree());
            let x_class = {
                let mut class = vec![0; x_num_gens];
                class[x.idx()] = 1;
                class
            };
            let x_hom = self.cache.from_class(
                format!("{x}"),
                BidegreeElement::new(x.degree(), FpVector::from_slice(self.p, &x_class)),
            );
            let matrix_vec = x_hom.get_map(tot.s()).hom_k(y.t());
            let matrix = Matrix::from_vec(self.resolution.prime(), &matrix_vec);
            let result_vec = matrix.row(y.idx()).to_owned();
            if !result_vec.is_zero() {
                println!("{x} * {y} = {result_vec}");
            }
            self.multiplication_table.insert((x, y.degree()), matrix);
            Ok(BidegreeElement::new(tot, result_vec))
        }
    }

    pub fn compute_all_products(&self) {
        self.resolution
            .iter_stem()
            .maybe_par_bridge()
            .for_each(|x| {
                if x == Bidegree::zero() {
                    // We don't compute products with the identity.
                    return;
                }
                if !self.resolution.has_computed_bidegree(x) {
                    return;
                }
                (0..self.resolution.number_of_gens_in_bidegree(x))
                    .into_maybe_par_iter()
                    .for_each(|x_idx| {
                        self.resolution
                            .iter_stem()
                            .maybe_par_bridge()
                            .for_each(|y| {
                                if !self.resolution.has_computed_bidegree(y)
                                    || !self.resolution.has_computed_bidegree(x + y)
                                {
                                    return;
                                }

                                (0..self.resolution.number_of_gens_in_bidegree(y))
                                    .into_maybe_par_iter()
                                    .for_each(|y_idx| {
                                        if let Err(e) = self.product_gen(
                                            BidegreeGenerator::new(x, x_idx),
                                            BidegreeGenerator::new(y, y_idx),
                                        ) {
                                            panic!("Failed to compute products: {e}");
                                        }
                                    });
                            });
                    });
            });
    }

    #[tracing::instrument(skip(self, bc), fields(a = %a, b = %b, c = %c))]
    fn compute_massey_products_a_b_c(
        &self,
        a: &BidegreeElement,
        b: &BidegreeElement,
        c: &BidegreeElement,
        bc: Arc<ChainHomotopy<QueryModuleResolution, QueryModuleResolution, QueryModuleResolution>>,
    ) -> AffineSubspace {
        // The Massey product shifts the bidegree by this amount
        let shift = b.degree() + c.degree() - Bidegree::s_t(1, 0);

        let tot = a.degree() + shift;

        let target_num_gens = self
            .resolution
            .number_of_gens_in_bidegree(a.degree() + shift);
        if target_num_gens == 0 {
            return AffineSubspace::new(FpVector::new(self.p, 0), Subspace::new(self.p, 0));
        }

        if !self.resolution.has_computed_bidegree(a.degree() + shift) {
            panic!("git gud");
        }

        bc.extend(tot);
        let htpy_map = bc.homotopy(tot.s());
        let offset_a = self
            .resolution
            .module(a.s())
            .generator_offset(a.t(), a.t(), 0);

        let mut answer = vec![0; target_num_gens];
        for (i, ans) in answer.iter_mut().enumerate() {
            let output = htpy_map.output(tot.t(), i);
            for (k, entry) in a.vec().iter().enumerate() {
                if entry != 0 {
                    //answer[i] += entry * output.entry(self.resolution.module(s1).generator_offset(t1,t1,k));
                    *ans += entry * output.entry(offset_a + k);
                }
            }
        }
        let massey = AffineSubspace::new(
            FpVector::from_slice(self.p, &answer),
            self.compute_indeterminacy(&a, b.degree(), &c),
        );

        // println!(
        //     "<{a}, {b}, {c}> = {offset} + [{indeterminacy:#}]",
        //     offset = massey.offset(),
        //     indeterminacy = massey.linear_part()
        // );
        massey
    }

    pub fn compute_massey_product_b_c(
        &self,
        b: &BidegreeElement,
        c: &BidegreeElement,
    ) -> anyhow::Result<()> {
        match self.product_is_zero(&b, &c) {
            Ok(false) | Err(_) => return Ok(()),
            _ => {}
        };

        // The Massey product shifts the bidegree by this amount
        let shift = b.degree() + c.degree() - Bidegree::s_t(1, 0);

        if !self
            .resolution
            .has_computed_bidegree(shift + Bidegree::s_t(0, self.resolution.min_degree()))
        {
            return Ok(());
        }

        let b_hom = self.cache.from_class(String::new(), b.clone());
        let c_hom = self.cache.from_class(String::new(), c.clone());

        let homotopy = Arc::new(ChainHomotopy::new(Arc::clone(&c_hom), Arc::clone(&b_hom)));

        self.iterate_all_vectors(|a| {
            match self.product_is_zero(&a, &b) {
                Ok(false) | Err(_) => return,
                _ => {}
            };
            if !self.resolution.has_computed_bidegree(a.degree() + shift) {
                return;
            }

            let massey = self.compute_massey_products_a_b_c(&a, &b, &c, Arc::clone(&homotopy));
            self.massey_products
                .insert((a.clone(), b.clone(), c.clone()), massey);
        });
        Ok(())
    }

    pub fn compute_all_massey_products(&self) {
        self.iterate_all_vectors(|c| {
            self.iterate_all_vectors(|b| {
                if !self
                    .resolution
                    .has_computed_bidegree(b.degree() + c.degree())
                {
                    return;
                }
                if let Err(e) = self.compute_massey_product_b_c(&b, &c) {
                    panic!("Failed to compute products: {e}");
                }
                // println!("Finished c = {c}, b = {b}");
            });
            println!("Finished c = {c}");
        });
    }

    #[tracing::instrument(skip(self, a, b_deg, c), fields(a = %a, b_deg = %b_deg, c = %c))]
    fn compute_indeterminacy(
        &self,
        a: &BidegreeElement,
        b_deg: Bidegree,
        c: &BidegreeElement,
    ) -> Subspace {
        let left = a.degree() + b_deg - Bidegree::s_t(1, 0);
        let right = b_deg + c.degree() - Bidegree::s_t(1, 0);
        let total = Bidegree::massey_bidegree(a.degree(), b_deg, c.degree());

        let left_dim = self.resolution().number_of_gens_in_bidegree(left);
        let right_dim = self.resolution().number_of_gens_in_bidegree(right);
        let total_dim = self.resolution().number_of_gens_in_bidegree(total);

        tracing::info!(
            "left = {left}, right = {right}, total = {total}, left_dim = {left_dim}, right_dim = \
             {right_dim}, total_dim = {total_dim}",
        );

        let l_indet = if right_dim == 0 {
            Subspace::new(self.p, total_dim)
        } else {
            let mut a_mul = Matrix::new(self.p, right_dim, total_dim);

            for (idx, _) in a.vec().iter_nonzero() {
                let gen = BidegreeGenerator::new(a.degree(), idx);
                a_mul += &self.multiplication_table.get(&(gen, right)).unwrap();
            }
            tracing::debug!(a_mul = %a_mul, columns = a_mul.columns());

            let (padded_cols, mut matrix) = Matrix::augmented_from_vec(self.p, &a_mul.to_vec());
            matrix.row_reduce();
            matrix.compute_image(a_mul.columns(), padded_cols)
        };

        tracing::info!(l_indet = ?l_indet, dim = l_indet.ambient_dimension());

        let r_indet = if left_dim == 0 {
            Subspace::new(self.p, total_dim)
        } else {
            let mut c_mul = Matrix::new(self.p, left_dim, total_dim);

            for (idx, _) in c.vec().iter_nonzero() {
                let gen = BidegreeGenerator::new(c.degree(), idx);
                c_mul += &self.multiplication_table.get(&(gen, left)).unwrap();
            }
            tracing::debug!(c_mul = %c_mul, columns = c_mul.columns());

            let (padded_cols, mut matrix) = Matrix::augmented_from_vec(self.p, &c_mul.to_vec());
            matrix.row_reduce();
            matrix.compute_image(c_mul.columns(), padded_cols)
        };
        tracing::info!(r_indet = ?r_indet, dim = r_indet.ambient_dimension());

        l_indet.sum(&r_indet)
    }

    fn product_is_zero(&self, a: &BidegreeElement, b: &BidegreeElement) -> Result<bool, String> {
        self.product(a, b).map(|prod| prod.vec().is_zero())
    }
}

pub struct MuResolutionHomomorphismCache<const U: bool, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: AugmentedChainComplex<Algebra = CC1::Algebra>,
{
    source: Arc<CC1>,
    target: Arc<CC2>,
    homs: DashMap<BidegreeElement, Arc<MuResolutionHomomorphism<U, CC1, CC2>>>,
}

impl<const U: bool, CC1, CC2> MuResolutionHomomorphismCache<U, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: AugmentedChainComplex<Algebra = CC1::Algebra>,
{
    pub fn new(source: Arc<CC1>, target: Arc<CC2>) -> Self {
        Self {
            source: Arc::clone(&source),
            target: Arc::clone(&target),
            homs: DashMap::default(),
        }
    }

    pub fn from_class(
        &self,
        name: String,
        element: BidegreeElement,
    ) -> Arc<MuResolutionHomomorphism<U, CC1, CC2>> {
        let x = self
            .homs
            .entry(element.clone().to_owned())
            .or_insert_with(|| {
                let result = MuResolutionHomomorphism::from_class(
                    name,
                    Arc::clone(&self.source),
                    Arc::clone(&self.target),
                    element.degree(),
                    &element.vec().iter().collect::<Vec<_>>(),
                );
                result.extend_all();
                Arc::new(result)
            })
            .downgrade();
        let y = &*x;
        Arc::clone(y)
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use once_cell::sync::Lazy;
    use proptest::prelude::*;

    use super::*;

    const MAX_STEM: i32 = 70;

    static MASSEYS: Lazy<ProductStructure> = Lazy::new(|| {
        let resolution =
            Arc::new(crate::utils::construct("S_2", Some(PathBuf::from("/mnt/tmpfs/"))).unwrap());
        resolution.compute_through_stem(Bidegree::n_s(MAX_STEM, MAX_STEM as u32 / 2 + 2));
        let result = ProductStructure::new(resolution);
        result.compute_all_products();
        result.compute_all_massey_products();
        result
    });

    fn arb_bidegree() -> impl Strategy<Value = Bidegree> {
        (0..=MAX_STEM)
            .prop_flat_map(|n| (Just(n), 1..=(n as u32 / 2 + 2)))
            .prop_map(|(n, s)| Bidegree::n_s(n, s))
    }

    fn arb_element_in_bidegree(b: Bidegree) -> impl Strategy<Value = BidegreeElement> {
        let num_gens = MASSEYS.resolution().number_of_gens_in_bidegree(b);
        vec![0; num_gens]
            .into_iter()
            .map(|_| 0..=1u32)
            .collect::<Vec<_>>()
            .prop_map(move |coeffs| {
                BidegreeElement::new(b, FpVector::from_slice(ValidPrime::new(2), &coeffs))
            })
    }

    fn arb_bidegree_element() -> impl Strategy<Value = BidegreeElement> {
        arb_bidegree().prop_flat_map(|bidegree| arb_element_in_bidegree(bidegree))
    }

    fn arb_bidegree_element_pair() -> impl Strategy<Value = (BidegreeElement, BidegreeElement)> {
        arb_bidegree().prop_flat_map(|bidegree| {
            (
                arb_element_in_bidegree(bidegree),
                arb_element_in_bidegree(bidegree),
            )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 10_000_000,
            max_shrink_time: 300_000,
            max_shrink_iters: 1_000_000_000,
            .. ProptestConfig::default()
        })]

        #[test]
        /// `<a, b, c> = <c, b, a>`
        fn test_commutativity(
            a in arb_bidegree_element(),
            b in arb_bidegree_element(),
            c in arb_bidegree_element(),
        ) {
            match
                MASSEYS.massey_products.get(&(a.clone(), b.clone(), c.clone()))
            {
                Some(abc) => {
                    let cba = MASSEYS.massey_products.get(&(c.clone(), b.clone(), a.clone())).unwrap();
                    prop_assert_eq!(abc.value(), cba.value());
                },
                _ => {},
            }
        }

        #[test]
        /// `<0, b, c>` contains `0`
        fn test_zero1(
            a in arb_bidegree(),
            b in arb_bidegree_element(),
            c in arb_bidegree_element(),
        ) {
            match MASSEYS.product_is_zero(&b, &c) {
                Ok(true) => {
                    let zero = BidegreeElement::new(
                        a,
                        FpVector::new(
                            ValidPrime::new(2),
                            MASSEYS.resolution().number_of_gens_in_bidegree(a)
                        )
                    );
                    if let Some(zbc) = MASSEYS.massey_products.get(&(zero.clone(), b.clone(), c.clone())) {
                        prop_assert!(zbc.value().offset().is_zero());
                    }
                },
                _ => {},
            }
        }

        #[test]
        /// `<a, 0, c>` contains `0`
        fn test_zero2(
            a in arb_bidegree_element(),
            b in arb_bidegree(),
            c in arb_bidegree_element(),
        ) {
            let zero = BidegreeElement::new(
                b,
                FpVector::new(
                    ValidPrime::new(2),
                    MASSEYS.resolution().number_of_gens_in_bidegree(b)
                )
            );
            if let Some(azc) = MASSEYS.massey_products.get(&(a.clone(), zero.clone(), c.clone())) {
                prop_assert!(azc.value().offset().is_zero());
            }
        }

        #[test]
        /// `<a1, b, c> + <a2, b, c>` contains `<a1 + a2, b, c>`
        fn test_left_linearity(
            (a1, a2) in arb_bidegree_element_pair(),
            b in arb_bidegree_element(),
            c in arb_bidegree_element(),
        ) {
            match (
                MASSEYS.product_is_zero(&a2, &b),
                MASSEYS.massey_products.get(&(a1.clone(), b.clone(), c.clone())),
            ) {
                (Ok(true), Some(a1bc)) => {
                    let a2bc = MASSEYS.massey_products.get(&(a2.clone(), b.clone(), c.clone())).unwrap();
                    let a1a2bc = MASSEYS.massey_products.get(&(a1 + a2, b.clone(), c.clone())).unwrap();
                    prop_assert!(a1bc.value().sum(a2bc.value()).contains_space(a1a2bc.value()));
                },
                _ => {},
            }
        }

        #[test]
        /// `<a, b1, c> + <a, b2, c> = <a, b1 + b2, c>`
        fn test_middle_linearity(
            a in arb_bidegree_element(),
            (b1, b2) in arb_bidegree_element_pair(),
            c in arb_bidegree_element(),
        ) {
            match (
                MASSEYS.product_is_zero(&a, &b2),
                MASSEYS.product_is_zero(&b2, &c),
                MASSEYS.massey_products.get(&(a.clone(), b1.clone(), c.clone())),
            ) {
                (Ok(true), Ok(true), Some(ab1c)) => {
                    let ab2c = MASSEYS.massey_products.get(&(a.clone(), b2.clone(), c.clone())).unwrap();
                    let ab1b2c = MASSEYS.massey_products.get(&(a.clone(), b1 + b2, c.clone())).unwrap();
                    prop_assert_eq!(ab1b2c.value().clone(), ab1c.value().sum(ab2c.value()));
                },
                _ => {},
            }
        }

        #[test]
        /// `a <b, c, d> = <a, b, c> d`
        fn test_outer_associativity(
            a in arb_bidegree_element(),
            b in arb_bidegree_element(),
            c in arb_bidegree_element(),
            d in arb_bidegree_element(),
        ) {
            match (
                MASSEYS.product_is_zero(&c, &d),
                MASSEYS.massey_products.get(&(a.clone(), b.clone(), c.clone())),
                MASSEYS.product_matrix(
                    Bidegree::massey_bidegree(a.degree(), b.degree(), c.degree()),
                    d.clone(),
                ),
                MASSEYS.resolution().number_of_gens_in_bidegree(d.degree()) > 0,
            ) {
                (Ok(true), Some(abc), Ok(d_mul_mat), true) => {
                    let abc_d = abc.apply_matrix(&d_mul_mat);

                    let bcd = MASSEYS.massey_products.get(&(b.clone(), c.clone(), d.clone())).unwrap();
                    let a_mul_mat = MASSEYS.product_matrix(
                        Bidegree::massey_bidegree(b.degree(), c.degree(), d.degree()),
                        a.clone(),
                    ).unwrap();
                    let a_bcd = bcd.apply_matrix(&a_mul_mat);

                    prop_assert_eq!(a_bcd, abc_d);
                },
                _ => {},
            }
        }

        #[test]
        /// `<ab, c, d>` contains `a <b, c, d>`
        fn test_left_associativity(
            a in arb_bidegree_element(),
            b in arb_bidegree_element(),
            c in arb_bidegree_element(),
            d in arb_bidegree_element(),
        ) {
            match (
                MASSEYS.massey_products.get(&(b.clone(), c.clone(), d.clone())),
                MASSEYS.product(&a, &b),
            ) {
                (Some(bcd), Ok(ab)) => {
                    if ab.vec().len() == 0 {
                        return Ok(());
                    }

                    let tot = a.degree() + Bidegree::massey_bidegree(b.degree(), c.degree(), d.degree());
                    if !MASSEYS.resolution().has_computed_bidegree(tot)
                        || MASSEYS.resolution().number_of_gens_in_bidegree(tot) == 0 {
                        return Ok(());
                    }
                    let ab_c_d = MASSEYS.massey_products.get(&(ab.clone(), c.clone(), d.clone())).unwrap();

                    let a_mul_mat = MASSEYS.product_matrix(
                        Bidegree::massey_bidegree(b.degree(), c.degree(), d.degree()),
                        a.clone(),
                    ).unwrap();
                    let a_bcd = bcd.apply_matrix(&a_mul_mat);

                    prop_assert!(ab_c_d.contains_space(&a_bcd));
                },
                _ => {},
            }
        }

        #[test]
        /// `<a, bc, d>` contains `<ab, c, d>`
        fn test_inner_associativity(
            a in arb_bidegree_element(),
            b in arb_bidegree_element(),
            c in arb_bidegree_element(),
            d in arb_bidegree_element(),
        ) {
            if let (Ok(ab), Ok(bc)) = (
                MASSEYS.product(&a, &b),
                MASSEYS.product(&b, &c),
            ) {
                if let (Some(a_bc_d), Some(ab_c_d)) = (
                    MASSEYS.massey_products.get(&(a.clone(), bc.clone(), d.clone())),
                    MASSEYS.massey_products.get(&(ab.clone(), c.clone(), d.clone())),
                ) {
                    prop_assert!(a_bc_d.contains_space(&ab_c_d));
                }
            }
        }
    }
}
