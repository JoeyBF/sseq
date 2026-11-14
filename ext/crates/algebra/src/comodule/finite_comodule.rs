use std::sync::Arc;

use bivec::BiVec;
use fp::vector::{FpSliceMut, FpVector};

use super::Comodule;
use crate::algebra::Algebra;

/// A finite comodule, with the action of every operation stored as a vector.
pub struct FiniteComodule<A: Algebra> {
    algebra: Arc<A>,
    name: String,
    graded_dimension: BiVec<usize>,
    /// `actions[d][n - 1][op][i]` is the image of basis element `i` of degree `d` under operation
    /// `op` of degree `n`, in degree `d - n`; operations of degree zero act as the identity.
    actions: BiVec<Vec<Vec<Vec<FpVector>>>>,
}

impl<A: Algebra> std::fmt::Display for FiniteComodule<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl<A: Algebra> FiniteComodule<A> {
    /// The comodule with the given dimensions and zero action, to be filled with
    /// [`Self::set_action`].
    pub fn new(algebra: Arc<A>, name: String, graded_dimension: BiVec<usize>) -> Self {
        let min = graded_dimension.min_degree();
        let max = graded_dimension.max_degree();
        algebra.compute_basis(max - min);
        let p = algebra.prime();
        let mut actions = BiVec::with_capacity(min, max + 1);
        for d in min..=max {
            let dim = graded_dimension[d];
            actions.push(
                (1..=d - min)
                    .map(|n| {
                        let target_dim = graded_dimension[d - n];
                        (0..algebra.dimension(n))
                            .map(|_| (0..dim).map(|_| FpVector::new(p, target_dim)).collect())
                            .collect()
                    })
                    .collect(),
            );
        }
        Self {
            algebra,
            name,
            graded_dimension,
            actions,
        }
    }

    /// Sets the image of basis element `idx` of `degree` under `(op_degree, op_index)`.
    pub fn set_action(
        &mut self,
        op_degree: i32,
        op_index: usize,
        degree: i32,
        idx: usize,
        output: &[u32],
    ) {
        assert!(op_degree > 0);
        self.actions[degree][op_degree as usize - 1][op_index][idx].copy_from_slice(output);
    }

    /// The image of basis element `idx` of `degree` under `(op_degree, op_index)`, for
    /// `op_degree > 0` and a nonzero target degree.
    pub fn action(&self, op_degree: i32, op_index: usize, degree: i32, idx: usize) -> &FpVector {
        &self.actions[degree][op_degree as usize - 1][op_index][idx]
    }

    /// The dual of the algebra in degrees up to `max_degree`, with $\langle b, a \cdot f \rangle =
    /// f(ba)$ on the basis dual to the algebra's.
    ///
    /// This is the coalgebra coacting on itself, truncated; the truncation is a subcomodule because
    /// the action lowers degree. Its Cotor is the ground field in degree zero, in every degree
    /// where the truncation is invisible.
    pub fn coregular(algebra: Arc<A>, max_degree: i32) -> Self {
        algebra.compute_basis(max_degree);
        let dims: BiVec<usize> =
            BiVec::from_vec(0, (0..=max_degree).map(|d| algebra.dimension(d)).collect());
        let mut m = Self::new(Arc::clone(&algebra), "coregular".to_string(), dims);
        let p = algebra.prime();
        for d in 1..=max_degree {
            for n in 1..=d {
                for a in 0..algebra.dimension(n) {
                    // a . f_c = sum over b of [coefficient of c in b a] f_b.
                    let products: Vec<FpVector> = (0..algebra.dimension(d - n))
                        .map(|b| {
                            let mut ba = FpVector::new(p, algebra.dimension(d));
                            algebra.multiply_basis_elements(ba.as_slice_mut(), 1, d - n, b, n, a);
                            ba
                        })
                        .collect();
                    let images: Vec<Vec<u32>> = (0..algebra.dimension(d))
                        .map(|c| products.iter().map(|ba| ba.entry(c)).collect())
                        .collect();
                    for (c, image) in images.iter().enumerate() {
                        m.set_action(n, a, d, c, image);
                    }
                }
            }
        }
        m
    }

    /// Checks that the action is associative, $a \cdot (b \cdot m) = (ab) \cdot m$, on every
    /// pair of basis operations and basis element.
    pub fn check_validity(&self) -> Result<(), String> {
        let p = self.prime();
        let alg = &self.algebra;
        let (min, max) = (self.min_degree(), self.graded_dimension.max_degree());
        for d in min..=max {
            for idx in 0..self.dimension(d) {
                for nb in 1..=d - min {
                    for na in 1..=d - min - nb {
                        let out_dim = self.dimension(d - na - nb);
                        if out_dim == 0 {
                            continue;
                        }
                        for b in 0..alg.dimension(nb) {
                            let mut bm = FpVector::new(p, self.dimension(d - nb));
                            self.act_on_basis(bm.as_slice_mut(), 1, nb, b, d, idx);
                            for a in 0..alg.dimension(na) {
                                let mut lhs = FpVector::new(p, out_dim);
                                self.act(lhs.as_slice_mut(), 1, na, a, d - nb, bm.as_slice());
                                let mut ab = FpVector::new(p, alg.dimension(na + nb));
                                alg.multiply_basis_elements(ab.as_slice_mut(), 1, na, a, nb, b);
                                let mut rhs = FpVector::new(p, out_dim);
                                for (k, c) in ab.iter_nonzero() {
                                    self.act_on_basis(rhs.as_slice_mut(), c, na + nb, k, d, idx);
                                }
                                if lhs != rhs {
                                    return Err(format!(
                                        "a = ({na}, {a}), b = ({nb}, {b}) on ({d}, {idx}): \
                                         {lhs} != {rhs}"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl<A: Algebra> Comodule for FiniteComodule<A> {
    type Algebra = A;

    fn algebra(&self) -> Arc<A> {
        Arc::clone(&self.algebra)
    }

    fn min_degree(&self) -> i32 {
        self.graded_dimension.min_degree()
    }

    fn max_computed_degree(&self) -> i32 {
        i32::MAX
    }

    fn max_degree(&self) -> Option<i32> {
        Some(self.graded_dimension.max_degree())
    }

    fn dimension(&self, degree: i32) -> usize {
        if degree < self.graded_dimension.min_degree()
            || degree > self.graded_dimension.max_degree()
        {
            return 0;
        }
        self.graded_dimension[degree]
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        format!("x{degree}_{idx}")
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
        if op_degree == 0 {
            result.add_basis_element(idx, coeff);
        } else if self.dimension(degree - op_degree) > 0 {
            result.add(
                self.action(op_degree, op_index, degree, idx).as_slice(),
                coeff,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use fp::prime::ValidPrime;
    use rstest::rstest;

    use super::*;
    use crate::{MilnorAlgebra, comodule::OpId, comodule::OpSet};

    #[rstest]
    #[case(2, 20)]
    #[case(3, 40)]
    fn coregular_is_a_module(#[case] p: u32, #[case] max_degree: i32) {
        let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(p), false));
        let m = FiniteComodule::coregular(alg, max_degree);
        m.check_validity().unwrap();
    }

    #[test]
    fn act_batch_agrees_with_act_on_basis() {
        let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(2), false));
        let m = FiniteComodule::coregular(Arc::clone(&alg), 16);
        let ops: OpSet = (1..=16)
            .flat_map(|n| {
                (0..alg.dimension(n)).map(move |i| OpId {
                    degree: n,
                    index: i as u32,
                })
            })
            .collect();
        let d = 16;
        let block = m.act_batch(&ops, d);
        for idx in 0..m.dimension(d) {
            for (local, op) in ops.iter().enumerate() {
                let mut v = FpVector::new(m.prime(), m.dimension(d - op.degree));
                m.act_on_basis(v.as_slice_mut(), 1, op.degree, op.index as usize, d, idx);
                let from_block: Vec<(usize, u32)> = block.act(idx, local as u32).iter().collect();
                let direct: Vec<(usize, u32)> = v.iter_nonzero().collect();
                assert_eq!(from_block, direct);
            }
        }
    }
}
