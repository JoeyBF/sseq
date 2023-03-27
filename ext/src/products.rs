use std::sync::Arc;

use algebra::{module::Module, MuAlgebra};
use fp::{
    matrix::{Matrix, Subspace},
    prime::ValidPrime,
    vector::{prelude::*, FpVector},
};
use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

use crate::{
    chain_complex::{
        AugmentedChainComplex, BoundedChainComplex, ChainComplex, ChainHomotopy, FreeChainComplex,
    },
    resolution_homomorphism::MuResolutionHomomorphism,
    save::SaveOption,
    utils::QueryModuleResolution,
};

#[cfg(feature = "concurrent")]
use rayon::prelude::*;

type DashMap<K, V> = dashmap::DashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

pub struct ProductStructure {
    p: ValidPrime,
    resolution: Arc<QueryModuleResolution>,
    cache: MuResolutionHomomorphismCache<false, QueryModuleResolution, QueryModuleResolution>,
    multiplication_table: DashMap<(BidegreeGenerator, Bidegree), Matrix>,
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
        }
    }

    pub fn resolution(&self) -> Arc<QueryModuleResolution> {
        Arc::clone(&self.resolution)
    }

    pub fn product(
        &self,
        x: &BidegreeElement<FpVector>,
        y: &BidegreeElement<FpVector>,
    ) -> Result<BidegreeElement<FpVector>, String> {
        let tot = x.degree() + y.degree();
        if !self.resolution().has_computed_bidegree(tot) {
            return Err(format!("Bidegree {tot} not computed"));
        }
        let target_dim = self.resolution().number_of_gens_in_bidegree(tot);
        let mut result = FpVector::new(self.p, target_dim);
        for (x_gen, x_coeff) in x.decompose() {
            for (y_gen, y_coeff) in y.decompose() {
                result.add(self.product_gen(x_gen, y_gen)?.vec(), x_coeff * y_coeff);
            }
        }
        Ok(BidegreeElement::new(x.degree(), result))
    }

    pub fn product_gen(
        &self,
        x: BidegreeGenerator,
        y: BidegreeGenerator,
    ) -> Result<BidegreeElement<FpVector>, String> {
        let tot = x.degree() + y.degree();
        if !self.resolution.has_computed_bidegree(tot) {
            return Err(format!("Bidegree {tot} not computed"));
        }
        if let Some(matrix) = self.multiplication_table.get(&(x, y.degree())) {
            let result_vec = matrix.row(y.idx()).into_owned();
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
            let result_vec = matrix.row(y.idx()).into_owned();
            let h0_degree = Bidegree::n_s(0, 1);
            if x.degree() == h0_degree && y.degree() == h0_degree {
                eprintln!("h0 * h0 = [{result_vec:?}]");
            }
            self.multiplication_table.insert((x, y.degree()), matrix);
            Ok(BidegreeElement::new(tot, result_vec))
        }
    }

    pub fn compute_all_products(&self) {
        #[cfg(feature = "concurrent")]
        self.compute_all_products_concurrent();
        #[cfg(not(feature = "concurrent"))]
        self.compute_all_products_serial();
    }

    pub fn compute_all_massey_products(&self) {
        #[cfg(feature = "concurrent")]
        self.compute_all_massey_products_concurrent();
        #[cfg(not(feature = "concurrent"))]
        self.compute_all_massey_products_serial();
    }

    #[cfg(feature = "concurrent")]
    fn compute_all_products_concurrent(&self) {
        use rayon::prelude::*;

        self.resolution.iter_stem().par_bridge().for_each(|x| {
            if x == Bidegree::zero() {
                // We don't compute products with the identity.
                return;
            }
            if !self.resolution.has_computed_bidegree(x) {
                return;
            }
            (0..self.resolution.number_of_gens_in_bidegree(x))
                .into_par_iter()
                .for_each(|x_idx| {
                    self.resolution.iter_stem().par_bridge().for_each(|y| {
                        if !self.resolution.has_computed_bidegree(y)
                            || !self.resolution.has_computed_bidegree(x + y)
                        {
                            return;
                        }

                        (0..self.resolution.number_of_gens_in_bidegree(y))
                            .into_par_iter()
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

    #[cfg_attr(feature = "concurrent", allow(dead_code))]
    fn compute_all_products_serial(&self) {
        for x_deg in self.resolution.iter_stem() {
            if x_deg == Bidegree::zero() {
                // We don't compute products with the identity.
                return;
            }
            if !self.resolution.has_computed_bidegree(x_deg) {
                return;
            }
            for x_idx in 0..self.resolution.number_of_gens_in_bidegree(x_deg) {
                for y_deg in self.resolution.iter_stem() {
                    if !self.resolution.has_computed_bidegree(y_deg)
                        || !self.resolution.has_computed_bidegree(x_deg + y_deg)
                    {
                        return;
                    }

                    for y_idx in 0..self.resolution.number_of_gens_in_bidegree(y_deg) {
                        if let Err(e) = self.product_gen(
                            BidegreeGenerator::new(x_deg, x_idx),
                            BidegreeGenerator::new(y_deg, y_idx),
                        ) {
                            panic!("Failed to compute products: {e}");
                        }
                    }
                }
            }
        }
    }

    fn compute_massey_products_a_b_c(
        &self,
        a: BidegreeElement<FpVector>,
        b: BidegreeElement<FpVector>,
        c: BidegreeElement<FpVector>,
        bc: Arc<ChainHomotopy<QueryModuleResolution, QueryModuleResolution, QueryModuleResolution>>,
    ) {
        // The Massey product shifts the bidegree by this amount
        let shift = b.degree() + c.degree() - Bidegree::s_t(1, 0);

        if !self.resolution.has_computed_bidegree(a.degree() + shift) {
            return;
        }

        let tot = a.degree() + shift;
        if tot != Bidegree::n_s(48, 9) || a.degree() != Bidegree::n_s(37, 7) {
            return;
        }

        let target_num_gens = self.resolution.number_of_gens_in_bidegree(tot);
        if target_num_gens == 0 {
            return;
        }

        bc.extend(tot);
        let htpy_map = bc.homotopy(tot.s());
        let offset_a = self
            .resolution
            .module(a.s())
            .generator_offset(a.t(), a.t(), 0);

        let mut answer = vec![0; target_num_gens];
        for (i, ans) in answer.iter_mut().enumerate().take(target_num_gens) {
            let output = htpy_map.output(tot.t(), i);
            for (k, entry) in a.vec().iter().enumerate() {
                if entry != 0 {
                    //answer[i] += entry * output.entry(self.resolution.module(s1).generator_offset(t1,t1,k));
                    *ans += entry * output.entry(offset_a + k);
                }
            }
        }
        let answer = FpVector::from_slice(self.p, &answer);
        let indeterminacy = self.compute_indeterminacy(&a, b.degree(), &c);
        if !indeterminacy.contains(answer.as_slice()) {
            println!(
                "<{a}, {b}, {c}> = {answer} + {indeterminacy_string}",
                indeterminacy_string = indeterminacy.to_string_oneline()
            );
        }
    }

    pub fn compute_massey_product_b_c(
        &self,
        b: &BidegreeElement<FpVector>,
        c: &BidegreeElement<FpVector>,
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

        let homotopy = Arc::new(ChainHomotopy::new_with_save_option(
            Arc::clone(&c_hom),
            Arc::clone(&b_hom),
            SaveOption::No,
        ));

        #[cfg(not(feature = "concurrent"))]
        let iter_stem = self.resolution.iter_stem();
        #[cfg(feature = "concurrent")]
        let iter_stem = self.resolution.iter_stem().par_bridge();

        iter_stem.for_each(|a_deg| {
            if (a_deg.n(), a_deg.s()) > (c.degree().n(), c.degree().s())
                || a_deg == Bidegree::zero()
            {
                return;
            }
            let a_space =
                Subspace::entire_space(self.p, self.resolution.number_of_gens_in_bidegree(a_deg));

            #[cfg(not(feature = "concurrent"))]
            let all_vectors = a_space.iter_all_vectors().skip(1);
            #[cfg(feature = "concurrent")]
            let all_vectors = a_space.iter_all_vectors().skip(1).par_bridge();

            all_vectors.for_each(|a_vec| {
                let a = BidegreeElement::new(a_deg, a_vec);
                match self.product_is_zero(&a, &b) {
                    Ok(false) | Err(_) => return,
                    _ => {}
                };

                self.compute_massey_products_a_b_c(a, b.clone(), c.clone(), Arc::clone(&homotopy))
            });
        });

        Ok(())
    }

    #[cfg(feature = "concurrent")]
    fn compute_all_massey_products_concurrent(&self) {
        use rayon::prelude::*;

        self.resolution.iter_stem().par_bridge().for_each(|c_deg| {
            if c_deg == Bidegree::zero() {
                // We don't compute products with the identity.
                return;
            }
            if !self.resolution.has_computed_bidegree(c_deg) {
                return;
            }
            let c_space =
                Subspace::entire_space(self.p, self.resolution.number_of_gens_in_bidegree(c_deg));
            c_space
                .iter_all_vectors()
                .skip(1)
                .par_bridge()
                .for_each(|c_vec| {
                    let c = BidegreeElement::new(c_deg, c_vec);
                    self.resolution.iter_stem().par_bridge().for_each(|b_deg| {
                        if !self.resolution.has_computed_bidegree(b_deg)
                            || !self.resolution.has_computed_bidegree(b_deg + c_deg)
                        {
                            return;
                        }
                        let b_num_gens = self.resolution.number_of_gens_in_bidegree(b_deg);
                        let b_space = Subspace::entire_space(self.p, b_num_gens);
                        b_space
                            .iter_all_vectors()
                            .skip(1)
                            .par_bridge()
                            .for_each(|b_vec| {
                                let b = BidegreeElement::new(b_deg, b_vec);
                                if let Err(e) = self.compute_massey_product_b_c(&b, &c) {
                                    panic!("Failed to compute products: {e}");
                                }
                            });
                    });
                });
        });
    }

    #[cfg_attr(feature = "concurrent", allow(dead_code))]
    fn compute_all_massey_products_serial(&self) {
        self.resolution.iter_stem().for_each(|c_deg| {
            if c_deg == Bidegree::zero() {
                // We don't compute products with the identity.
                return;
            }
            if !self.resolution.has_computed_bidegree(c_deg) {
                return;
            }
            let c_space =
                Subspace::entire_space(self.p, self.resolution.number_of_gens_in_bidegree(c_deg));
            c_space.iter_all_vectors().skip(1).for_each(|c_vec| {
                let c = BidegreeElement::new(c_deg, c_vec);
                self.resolution.iter_stem().for_each(|b_deg| {
                    if !self.resolution.has_computed_bidegree(b_deg)
                        || !self.resolution.has_computed_bidegree(b_deg + c_deg)
                    {
                        return;
                    }
                    let b_num_gens = self.resolution.number_of_gens_in_bidegree(b_deg);
                    let b_space = Subspace::entire_space(self.p, b_num_gens);
                    b_space.iter_all_vectors().skip(1).for_each(|b_vec| {
                        let b = BidegreeElement::new(b_deg, b_vec);
                        if let Err(e) = self.compute_massey_product_b_c(&b, &c) {
                            panic!("Failed to compute products: {e}");
                        }
                    });
                });
            });
        });
    }

    fn compute_indeterminacy(
        &self,
        a: &BidegreeElement<FpVector>,
        b_deg: Bidegree,
        c: &BidegreeElement<FpVector>,
    ) -> Subspace {
        let left = a.degree() + b_deg - Bidegree::s_t(1, 0);
        let right = b_deg + c.degree() - Bidegree::s_t(1, 0);
        let total = Bidegree::massey_bidegree(a.degree(), b_deg, c.degree());

        let left_dim = self.resolution().number_of_gens_in_bidegree(left);
        let right_dim = self.resolution().number_of_gens_in_bidegree(right);
        let total_dim = self.resolution().number_of_gens_in_bidegree(total);

        let l_indet = if right_dim == 0 {
            Subspace::new(self.p, right_dim, total_dim)
        } else {
            let mut a_mul = Matrix::new(self.p, right_dim, total_dim);

            for (idx, _) in a.vec().iter_nonzero() {
                let gen = BidegreeGenerator::new(a.degree(), idx);
                a_mul += &self.multiplication_table.get(&(gen, right)).unwrap();
            }

            let (padded_cols, mut matrix) = Matrix::augmented_from_vec(self.p, &a_mul.to_vec());
            matrix.row_reduce();
            matrix.compute_image(a_mul.columns(), padded_cols)
        };

        let r_indet = if left_dim == 0 {
            Subspace::new(self.p, left_dim, total_dim)
        } else {
            let mut c_mul = Matrix::new(self.p, left_dim, total_dim);

            for (idx, _) in c.vec().iter_nonzero() {
                let gen = BidegreeGenerator::new(c.degree(), idx);
                c_mul += &self.multiplication_table.get(&(gen, left)).unwrap();
            }

            let (padded_cols, mut matrix) = Matrix::augmented_from_vec(self.p, &c_mul.to_vec());
            matrix.row_reduce();
            matrix.compute_image(c_mul.columns(), padded_cols)
        };
        l_indet.sum(&r_indet)
    }

    fn product_is_zero(
        &self,
        a: &BidegreeElement<FpVector>,
        b: &BidegreeElement<FpVector>,
    ) -> Result<bool, String> {
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
    homs: DashMap<BidegreeElement<FpVector>, Arc<MuResolutionHomomorphism<U, CC1, CC2>>>,
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
        element: BidegreeElement<FpVector>,
    ) -> Arc<MuResolutionHomomorphism<U, CC1, CC2>> {
        let x = self
            .homs
            .entry(element.clone().into_owned())
            .or_insert_with(|| {
                let result = MuResolutionHomomorphism::from_class_with_save_option(
                    name,
                    Arc::clone(&self.source),
                    Arc::clone(&self.target),
                    element.degree(),
                    &element.vec().iter().collect::<Vec<_>>(),
                    SaveOption::No,
                );
                result.extend_all();
                Arc::new(result)
            })
            .downgrade();
        let y = &*x;
        Arc::clone(y)
    }
}
