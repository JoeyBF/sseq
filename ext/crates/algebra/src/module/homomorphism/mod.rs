use std::sync::Arc;

use fp::{
    matrix::{AugmentedMatrix, Matrix, MatrixSliceMut, QuasiInverse, Subspace},
    prime::ValidPrime,
    vector::{FpSlice, FpSliceMut},
};

use crate::module::Module;

mod free_module_homomorphism;
mod full_module_homomorphism;
mod generic_zero_homomorphism;
mod hom_pullback;
mod quotient_homomorphism;

pub use free_module_homomorphism::{
    FreeModuleHomomorphism, MuFreeModuleHomomorphism, UnstableFreeModuleHomomorphism,
};
pub use full_module_homomorphism::FullModuleHomomorphism;
pub use generic_zero_homomorphism::GenericZeroHomomorphism;
pub use hom_pullback::HomPullback;
// If `concurrent` is disabled, `MaybeIndexedParallelIterator` implements `Iterator`, and so has an
// `enumerate` method. However, if it is disabled, it does *not* implement `Iterator`, and the
// `enumerate` method is provided by `rayon::prelude::IndexedParallelIterator`, which is loaded in
// the `maybe_rayon` prelude.
#[allow(unused_imports)]
use maybe_rayon::prelude::*;
pub use quotient_homomorphism::{QuotientHomomorphism, QuotientHomomorphismSource};

/// A trait that represents a homomorphism between two modules.
///
/// Each `ModuleHomomorphism` may come with auxiliary data, namely the kernel, image and
/// quasi_inverse at each degree (the quasi-inverse is a map that is a right inverse when restricted
/// to the image). These are computed via
/// [`ModuleHomomorphism::compute_auxiliary_data_through_degree`] and retrieved through
/// [`ModuleHomomorphism::kernel`], [`ModuleHomomorphism::quasi_inverse`] and
/// [`ModuleHomomorphism::image`].
///
/// Note that an instance of a `ModuleHomomorphism` need not have the data available, even after
/// `compute_auxiliary_data_through_degree` is invoked.
pub trait ModuleHomomorphism: Send + Sync {
    type Source: Module;
    type Target: Module<Algebra = <Self::Source as Module>::Algebra>;

    fn source(&self) -> Arc<Self::Source>;
    fn target(&self) -> Arc<Self::Target>;
    fn degree_shift(&self) -> i32;

    /// Calling this function when `input_idx < source().dimension(input_degree)` results in
    /// undefined behaviour. Implementations are encouraged to panic when this happens (this is
    /// usually the case because of out-of-bounds errors.
    fn apply_to_basis_element(
        &self,
        result: FpSliceMut,
        coeff: u32,
        input_degree: i32,
        input_idx: usize,
    );

    #[allow(unused_variables)]
    fn kernel(&self, degree: i32) -> Option<&Subspace> {
        None
    }

    #[allow(unused_variables)]
    fn quasi_inverse(&self, degree: i32) -> Option<&QuasiInverse> {
        None
    }

    #[allow(unused_variables)]
    fn image(&self, degree: i32) -> Option<&Subspace> {
        None
    }

    #[allow(unused_variables)]
    fn compute_auxiliary_data_through_degree(&self, degree: i32) {}

    fn apply(&self, mut result: FpSliceMut, coeff: u32, input_degree: i32, input: FpSlice) {
        let p = self.prime();
        for (i, v) in input.iter_nonzero() {
            self.apply_to_basis_element(result.copy(), (coeff * v) % p, input_degree, i);
        }
    }

    fn prime(&self) -> ValidPrime {
        self.source().prime()
    }

    fn min_degree(&self) -> i32 {
        self.source().min_degree()
    }

    /// Compute the auxiliary data associated to the homomorphism at input degree `degree`. Returns
    /// it in the order image, kernel, quasi_inverse
    fn auxiliary_data(&self, degree: i32) -> (Subspace, Subspace, QuasiInverse) {
        let p = self.prime();
        let output_degree = degree - self.degree_shift();
        self.source().compute_basis(degree);
        self.target().compute_basis(output_degree);
        let source_dimension = self.source().dimension(degree);
        let target_dimension = self.target().dimension(output_degree);
        let mut matrix =
            AugmentedMatrix::<2>::new(p, source_dimension, [target_dimension, source_dimension]);

        self.get_matrix(matrix.segment(0, 0), degree);
        matrix.segment(1, 1).add_identity();

        // A ZERO TARGET NEEDS NO REDUCTION. Past the target module's top cell -- which for a finite
        // module is almost the whole resolution, and for the sphere is every degree above 0 --
        // `target_dimension` is 0, so the augmented matrix is `[ | I]`: no target columns at all and
        // the source identity beside them. That is ALREADY in reduced row echelon form, with pivot
        // i in column i, so `row_reduce` can only rediscover the identity.
        //
        // It is not cheap to rediscover. `step0` calls this for every `t` through
        // `compute_auxiliary_data_through_degree`, and the resulting matrix is square in the SOURCE
        // dimension: the stem-400 logs show `gpu_row_reduce{rows=269175 cols=269175}` and up,
        // 8.4-9.3 GiB apiece, under `step0{t=382}` spans of 57-145 seconds each -- one per degree.
        // Skipping the reduction also skips the whole GPU round trip beneath it (upload, reduce,
        // download and the host marshalling around them).
        //
        // `step1` already relies on exactly this fact for the same map, taking
        // `Subspace::entire_space` when `cc_module.dimension(t) == 0` rather than computing a
        // kernel; this makes `step0` stop paying for what `step1` knows for free.
        if target_dimension == 0 {
            matrix.initialize_pivots();
            let piv = matrix.pivots_mut();
            for (i, e) in piv.iter_mut().enumerate().take(source_dimension) {
                *e = i as isize;
            }
        } else {
            matrix.row_reduce();
        }

        (
            matrix.compute_image(),
            matrix.compute_kernel(),
            matrix.compute_quasi_inverse(),
        )
    }

    /// Write the matrix of the homomorphism at input degree `degree` to `matrix`.
    ///
    /// The (sliced) dimensions of `matrix` must be equal to source_dimension x
    /// target_dimension
    fn get_matrix(&self, mut matrix: MatrixSliceMut, degree: i32) {
        assert_eq!(self.source().dimension(degree), matrix.rows());
        assert_eq!(
            self.target().dimension(degree - self.degree_shift()),
            matrix.columns()
        );

        if matrix.columns() == 0 {
            return;
        }

        matrix
            .maybe_par_iter_mut()
            .enumerate()
            .for_each(|(i, row)| self.apply_to_basis_element(row, 1, degree, i));
    }

    /// Get the values of the homomorphism on the specified inputs to `matrix`.
    fn get_partial_matrix(&self, degree: i32, inputs: &[usize]) -> Matrix {
        let mut matrix = Matrix::new(self.prime(), inputs.len(), self.target().dimension(degree));

        if matrix.columns() == 0 {
            return matrix;
        }

        matrix
            .maybe_par_iter_mut()
            .enumerate()
            .for_each(|(i, row)| self.apply_to_basis_element(row, 1, degree, inputs[i]));

        matrix
    }

    /// Attempt to apply quasi inverse to the input. Returns whether the operation was
    /// successful. This is required to either always succeed or always fail for each degree.
    #[must_use]
    fn apply_quasi_inverse(&self, result: FpSliceMut, degree: i32, input: FpSlice) -> bool {
        if let Some(qi) = self.quasi_inverse(degree) {
            qi.apply(result, 1, input);
            true
        } else {
            false
        }
    }
}

pub trait ZeroHomomorphism<S: Module, T: Module<Algebra = S::Algebra>>:
    ModuleHomomorphism<Source = S, Target = T>
{
    fn zero_homomorphism(s: Arc<S>, t: Arc<T>, degree_shift: i32) -> Self;
}

pub trait IdentityHomomorphism<S: Module>: ModuleHomomorphism<Source = S, Target = S> {
    fn identity_homomorphism(s: Arc<S>) -> Self;
}
