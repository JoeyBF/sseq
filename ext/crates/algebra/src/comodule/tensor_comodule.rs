use std::sync::Arc;

use fp::{
    prime::minus_one_to_the_n,
    vector::{FpSliceMut, FpVector},
};
use once::OnceVec;

use super::Comodule;
use crate::algebra::Bialgebra;

/// The tensor product of two comodules, with the diagonal coaction.
///
/// An operation acts through its coproduct, $a(m \otimes n) = \sum \pm a'm \otimes a''n$, with the
/// Koszul sign $(-1)^{|a''||m|}$. The basis of each degree is the products of basis elements,
/// grouped by the degree of the left factor, left factor major.
pub struct TensorComodule<M, N> {
    left: Arc<M>,
    right: Arc<N>,
    /// `blocks[d - min_degree]` lists, for each degree `i` of the left factor, `(i, offset)`.
    blocks: OnceVec<Vec<(i32, usize)>>,
    /// The dimension in each degree from the minimum.
    dims: OnceVec<usize>,
}

impl<M: Comodule, N: Comodule<Algebra = M::Algebra>> TensorComodule<M, N> {
    /// The tensor product of `left` and `right`.
    pub fn new(left: Arc<M>, right: Arc<N>) -> Self {
        Self {
            left,
            right,
            blocks: OnceVec::new(),
            dims: OnceVec::new(),
        }
    }

    /// The lowest degree, the sum of those of the factors.
    fn min(&self) -> i32 {
        self.left.min_degree() + self.right.min_degree()
    }

    /// The left degree, left index and right index of basis element `idx` of `degree`.
    fn split(&self, degree: i32, idx: usize) -> (i32, usize, usize) {
        let blocks = &self.blocks[(degree - self.min()) as usize];
        let k = blocks.partition_point(|&(_, off)| off <= idx) - 1;
        let (i, off) = blocks[k];
        let right_dim = self.right.dimension(degree - i);
        ((i), (idx - off) / right_dim, (idx - off) % right_dim)
    }

    /// The offset of the block whose left factor has degree `i`, in `degree`.
    fn offset(&self, degree: i32, i: i32) -> Option<usize> {
        let blocks = &self.blocks[(degree - self.min()) as usize];
        blocks.iter().find(|b| b.0 == i).map(|b| b.1)
    }

    /// Adds `coeff` times the image of `input` in `degree` under one operation that
    /// [`Bialgebra::decompose`] returns to `result`.
    fn act_factor(
        &self,
        result: &mut FpVector,
        coeff: u32,
        op_degree: i32,
        op_index: usize,
        degree: i32,
        input: &FpVector,
    ) where
        M::Algebra: Bialgebra,
    {
        let p = self.prime();
        let algebra = self.algebra();
        let target = degree - op_degree;
        for (idx, v) in input.iter_nonzero() {
            let (i, l, r) = self.split(degree, idx);
            for (ld, li, rd, ri, c) in algebra.coproduct(op_degree, op_index) {
                let (lt, rt) = (i - ld, degree - i - rd);
                let (ldim, rdim) = (self.left.dimension(lt), self.right.dimension(rt));
                if ldim == 0 || rdim == 0 {
                    continue;
                }
                let Some(off) = self.offset(target, lt) else {
                    continue;
                };
                let mut lv = FpVector::new(p, ldim);
                self.left.act_on_basis(lv.as_slice_mut(), 1, ld, li, i, l);
                if lv.is_zero() {
                    continue;
                }
                let mut rv = FpVector::new(p, rdim);
                self.right
                    .act_on_basis(rv.as_slice_mut(), 1, rd, ri, degree - i, r);
                let scale = coeff * v % p * c % p * minus_one_to_the_n(p, rd * i) % p;
                for (a, x) in lv.iter_nonzero() {
                    for (b, y) in rv.iter_nonzero() {
                        result.add_basis_element(off + a * rdim + b, scale * x % p * y % p);
                    }
                }
            }
        }
    }
}

impl<M: Comodule, N: Comodule> std::fmt::Display for TensorComodule<M, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} ⊗ {}", self.left, self.right)
    }
}

impl<M, N> Comodule for TensorComodule<M, N>
where
    M: Comodule,
    M::Algebra: Bialgebra,
    N: Comodule<Algebra = M::Algebra>,
{
    type Algebra = M::Algebra;

    fn algebra(&self) -> Arc<M::Algebra> {
        self.left.algebra()
    }

    fn min_degree(&self) -> i32 {
        self.left.min_degree() + self.right.min_degree()
    }

    fn compute_basis(&self, degree: i32) {
        let (lmin, rmin) = (self.left.min_degree(), self.right.min_degree());
        self.left.compute_basis(degree - rmin);
        self.right.compute_basis(degree - lmin);
        let min = self.min_degree();
        self.blocks.extend((degree - min).max(-1) as usize, |k| {
            let d = min + k as i32;
            let mut off = 0;
            (lmin..=d - rmin)
                .map(|i| {
                    let b = (i, off);
                    off += self.left.dimension(i) * self.right.dimension(d - i);
                    b
                })
                .collect()
        });
        self.dims.extend((degree - min).max(-1) as usize, |k| {
            let d = min + k as i32;
            (lmin..=d - rmin)
                .map(|i| self.left.dimension(i) * self.right.dimension(d - i))
                .sum()
        });
    }

    fn max_computed_degree(&self) -> i32 {
        self.min_degree() + self.dims.len() as i32 - 1
    }

    fn dimension(&self, degree: i32) -> usize {
        if degree < self.min_degree() {
            return 0;
        }
        self.dims[(degree - self.min_degree()) as usize]
    }

    fn max_degree(&self) -> Option<i32> {
        Some(self.left.max_degree()? + self.right.max_degree()?)
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        let (i, l, r) = self.split(degree, idx);
        format!(
            "{} ⊗ {}",
            self.left.basis_element_to_string(i, l),
            self.right.basis_element_to_string(degree - i, r)
        )
    }

    fn act_on_basis(
        &self,
        mut result: FpSliceMut,
        coeff: u32,
        op_degree: i32,
        op_index: usize,
        degree: i32,
        idx: usize,
    ) {
        let p = self.prime();
        let mut current = FpVector::new(p, self.dimension(degree));
        current.set_entry(idx, 1);
        let mut current_degree = degree;
        // `decompose` lists the factors in the order they act.
        for (fd, fi) in self.algebra().decompose(op_degree, op_index) {
            let mut next = FpVector::new(p, self.dimension(current_degree - fd));
            self.act_factor(&mut next, 1, fd, fi, current_degree, &current);
            current = next;
            current_degree -= fd;
        }
        result.add(current.as_slice(), coeff);
    }
}

#[cfg(test)]
mod tests {
    use fp::prime::ValidPrime;
    use rstest::rstest;

    use super::*;
    use crate::{
        MilnorAlgebra,
        comodule::{FiniteComodule, check_associativity},
    };

    #[rstest]
    #[case(2, 12)]
    #[case(3, 24)]
    fn tensor_is_a_module(#[case] p: u32, #[case] max: i32) {
        let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(p), false));
        let a = Arc::new(FiniteComodule::coregular(Arc::clone(&alg), max));
        let b = Arc::new(FiniteComodule::coregular(Arc::clone(&alg), max / 2));
        let t = TensorComodule::new(a, b);
        check_associativity(&t, max).unwrap();
    }
}
