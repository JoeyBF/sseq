use crate::algebra::Algebra;

/// An [`Algebra`] equipped with a coproduct operation that makes it into a
/// bialgebra.
#[enum_dispatch::enum_dispatch]
pub trait Bialgebra: Algebra {
    /// Computes a coproduct $\Delta(x) = \sum_j c_j A_j \otimes B_j$ as the list of
    /// `(deg A_j, idx A_j, deg B_j, idx B_j, c_j)`.
    ///
    /// `x` must have been returned by [`Bialgebra::decompose()`]. An algebra whose `decompose`
    /// returns each basis element unchanged has a coproduct on every basis element, which makes
    /// it the dual of a coalgebra in the same basis.
    fn coproduct(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize, i32, usize, u32)>;

    /// Decomposes an element of the algebra into a product of elements, each of
    /// which we can compute a coproduct on efficiently.
    ///
    /// The product is laid out such that the first element of the vector is
    /// applied to a module element first when acting on it.
    ///
    /// This function is to be used with [`Bialgebra::coproduct()`].
    ///
    /// This API is motivated by the fact that, in the admissible basis for the Adem algebra,
    /// an element naturally decomposes into a product of Steenrod squares, each of which has an
    /// easy coproduct formula.
    fn decompose(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize)>;
}
