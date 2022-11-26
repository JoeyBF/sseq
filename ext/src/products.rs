use std::sync::Arc;

use algebra::{module::Module, MuAlgebra};
use dashmap::DashMap;
use fp::{matrix::Matrix, vector::FpVector};

use crate::{
    chain_complex::{AugmentedChainComplex, BoundedChainComplex, ChainComplex, FreeChainComplex},
    resolution_homomorphism::MuResolutionHomomorphism,
    utils::QueryModuleResolution,
};

/// (s, t)
pub type Bidegree = (u32, i32);
/// (s, t, vec)
pub type BidegreeElement = (u32, i32, FpVector);
/// (s, t, idx)
pub type BidegreeGenerator = (u32, i32, usize);

pub struct ProductStructure {
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
        let cache = MuResolutionHomomorphismCache {
            source: Arc::clone(&resolution),
            target: Arc::clone(&resolution),
            homs: DashMap::new(),
        };
        Self {
            resolution,
            cache,
            multiplication_table: DashMap::new(),
        }
    }

    pub fn resolution(&self) -> Arc<QueryModuleResolution> {
        Arc::clone(&self.resolution)
    }

    pub fn product_gen(
        &self,
        x: BidegreeGenerator,
        y: BidegreeGenerator,
    ) -> Result<BidegreeElement, String> {
        let (x_s, x_t, x_idx) = x;
        let (y_s, y_t, y_idx) = y;
        let (tot_s, tot_t) = (x_s + y_s, x_t + y_t);
        if !self.resolution.has_computed_bidegree(tot_s, tot_t) {
            return Err(format!("Bidegree ({tot_s}, {tot_t}) not computed"));
        }
        if let Some(matrix) = self.multiplication_table.get(&(x, (y_s, y_t))) {
            let result_vec = matrix.row(y_idx).to_owned();
            Ok((tot_s, tot_t, result_vec))
        } else {
            let x_num_gens = self.resolution.module(x_s).number_of_gens_in_degree(x_t);
            // let y_num_gens = self.resolution.module(y_s).number_of_gens_in_degree(y_t);
            // let tot_num_gens = self
            //     .resolution
            //     .module(tot_s)
            //     .number_of_gens_in_degree(tot_t);
            let x_class = {
                let mut class = vec![0; x_num_gens];
                class[x_idx] = 1;
                class
            };
            let x_hom = self
                .cache
                .from_class(format!("({x_s},{x_t},{x_idx})"), x_s, x_t, &x_class);
            let matrix_vec = x_hom.get_map(tot_s).hom_k(y_t);
            let matrix = Matrix::from_vec(self.resolution.prime(), &matrix_vec);
            let result_vec = matrix.row(y_idx).to_owned();
            self.multiplication_table.insert((x, (y_s, y_t)), matrix);
            Ok((tot_s, tot_t, result_vec))
        }
    }

    pub fn compute_all_products(&self) {
        #[cfg(feature = "concurrent")]
        self.compute_all_products_concurrent();
        #[cfg(not(feature = "concurrent"))]
        self.compute_all_products_serial();
    }

    #[cfg(feature = "concurrent")]
    fn compute_all_products_concurrent(&self) {
        use rayon::prelude::*;

        self.resolution
            .iter_stem()
            .par_bridge()
            .for_each(|(x_s, _, x_t)| {
                if (x_s, x_t) == (0, 0) {
                    // We don't compute products with the identity.
                    return;
                }
                if !self.resolution.has_computed_bidegree(x_s, x_t) {
                    return;
                }
                (0..self.resolution.module(x_s).number_of_gens_in_degree(x_t))
                    .into_par_iter()
                    .for_each(|x_idx| {
                        let timer = crate::utils::Timer::start();
                        self.resolution
                            .iter_stem()
                            .par_bridge()
                            .for_each(|(y_s, _, y_t)| {
                                if !self.resolution.has_computed_bidegree(y_s, y_t)
                                    || !self.resolution.has_computed_bidegree(x_s + y_s, x_t + y_t)
                                {
                                    return;
                                }

                                (0..self.resolution.module(y_s).number_of_gens_in_degree(y_t))
                                    .into_par_iter()
                                    .for_each(|y_idx| {
                                        if let Err(e) =
                                            self.product_gen((x_s, x_t, x_idx), (y_s, y_t, y_idx))
                                        {
                                            panic!("Failed to compute products: {e}");
                                        }
                                    });
                            });
                        timer.end(format_args!(
                            "Computed products ({x_s}, {x_t}, {x_idx}) * y"
                        ));
                    });
            });
    }

    #[cfg_attr(feature = "concurrent", allow(dead_code))]
    fn compute_all_products_serial(&self) {
        for (x_s, _, x_t) in self.resolution.iter_stem() {
            for x_idx in 0..self.resolution.module(x_s).number_of_gens_in_degree(x_t) {
                for (y_s, _, y_t) in self.resolution.iter_stem() {
                    for y_idx in 0..self.resolution.module(y_s).number_of_gens_in_degree(y_t) {
                        if let Err(e) = self.product_gen((x_s, x_t, x_idx), (y_s, y_t, y_idx)) {
                            panic!("Failed to compute products: {e}");
                        }
                    }
                }
            }
        }
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
    homs: DashMap<(String, u32, i32, Vec<u32>), Arc<MuResolutionHomomorphism<U, CC1, CC2>>>,
}

impl<const U: bool, CC1, CC2> MuResolutionHomomorphismCache<U, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: AugmentedChainComplex<Algebra = CC1::Algebra>,
{
    pub fn from_class(
        &self,
        name: String,
        shift_s: u32,
        shift_t: i32,
        class: &[u32],
    ) -> Arc<MuResolutionHomomorphism<U, CC1, CC2>> {
        let x = self
            .homs
            .entry((name.clone(), shift_s, shift_t, class.into()))
            .or_insert_with(|| {
                let result = MuResolutionHomomorphism::new_with_save(
                    name,
                    Arc::clone(&self.source),
                    Arc::clone(&self.target),
                    shift_s,
                    shift_t,
                    false,
                );

                let num_gens = result
                    .source
                    .module(shift_s)
                    .number_of_gens_in_degree(shift_t);
                assert_eq!(num_gens, class.len());

                let mut matrix = Matrix::new(result.source.prime(), num_gens, 1);
                for (k, &v) in class.iter().enumerate() {
                    matrix[k].set_entry(0, v);
                }

                result.extend_step(shift_s, shift_t, Some(&matrix));
                result.extend_all();
                Arc::new(result)
            })
            .downgrade();
        let y = &*x;
        Arc::clone(y)
    }
}
