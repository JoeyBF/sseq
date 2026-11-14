use std::sync::Arc;

use fp::{
    prime::ValidPrime,
    vector::{FpSlice, FpSliceMut, FpVector},
};

use super::{ActionBlock, OpSet};
use crate::algebra::Algebra;

/// A bounded-below comodule of finite type over the dual of [`Self::Algebra`], presented as a
/// left module over that algebra whose operations lower degree.
///
/// A right comodule $\psi\colon M \to M \otimes \Gamma$ over $\Gamma = A^\vee$ is the same thing as
/// a locally finite left $A$-module through $a \cdot m = \sum m_i \langle a, \gamma_i \rangle$
/// where $\psi(m) = \sum m_i \otimes \gamma_i$; an operation of degree $n$ takes degree $d$ to
/// degree $d - n$. A left comodule becomes a right one through the antipode. With $C$ a free
/// resolution of the ground field over $A$, $\mathrm{Hom}_A(C, M)$ then computes
/// $\mathrm{Cotor}_\Gamma(M, \mathbb{F}_p) = \mathrm{Ext}_A(\mathbb{F}_p, M)$.
///
/// This is not a [`Module`](crate::module::Module): a module is bounded below with operations
/// raising degree, and the dual of an infinite comodule such as $\mathbb{F}_2[v_0, v_1, \dots]$ is
/// unbounded below.
pub trait Comodule: std::fmt::Display + Send + Sync + 'static {
    type Algebra: Algebra;

    /// The algebra acting, dual to the coalgebra coacting.
    fn algebra(&self) -> Arc<Self::Algebra>;

    /// The lowest degree in which the comodule is nonzero.
    fn min_degree(&self) -> i32;

    /// Computes whatever is needed to answer queries in degrees up to `degree`. Idempotent.
    #[allow(unused_variables)]
    fn compute_basis(&self, degree: i32) {}

    /// The highest degree in which queries can be answered.
    fn max_computed_degree(&self) -> i32;

    /// The dimension in `degree`.
    fn dimension(&self, degree: i32) -> usize;

    /// The name of a basis element.
    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String;

    /// Adds `coeff` times the image of basis element `idx` of `degree` under the operation
    /// `(op_degree, op_index)` to `result`, which lives in degree `degree - op_degree`.
    fn act_on_basis(
        &self,
        result: FpSliceMut,
        coeff: u32,
        op_degree: i32,
        op_index: usize,
        degree: i32,
        idx: usize,
    );

    /// The prime.
    fn prime(&self) -> ValidPrime {
        self.algebra().prime()
    }

    /// A degree above which the comodule is zero, if there is one.
    fn max_degree(&self) -> Option<i32> {
        None
    }

    /// [`Self::act_on_basis`] on a general element of `degree`.
    fn act(
        &self,
        mut result: FpSliceMut,
        coeff: u32,
        op_degree: i32,
        op_index: usize,
        degree: i32,
        input: FpSlice,
    ) {
        let p = self.prime();
        for (i, v) in input.iter_nonzero() {
            self.act_on_basis(result.copy(), coeff * v % p, op_degree, op_index, degree, i);
        }
    }

    /// The action of every operation in `ops` on every basis element of `degree`.
    ///
    /// This is the unit of work for a consumer such as a Cotor complex, which needs the same
    /// operations on a whole degree at once: an implementor can share work across the
    /// operations and across the basis, and the caller decides how long the block lives. The
    /// default acts one pair at a time.
    fn act_batch(&self, ops: &OpSet, degree: i32) -> ActionBlock {
        let p = self.prime();
        let mut block = ActionBlock::new(p);
        let mut out: Vec<FpVector> = Vec::new();
        for idx in 0..self.dimension(degree) {
            let mut row = Vec::new();
            for (local, op) in ops.up_to_degree(degree - self.min_degree()) {
                let target_dim = self.dimension(degree - op.degree);
                if target_dim == 0 {
                    continue;
                }
                out.clear();
                out.push(FpVector::new(p, target_dim));
                self.act_on_basis(
                    out[0].as_slice_mut(),
                    1,
                    op.degree,
                    op.index as usize,
                    degree,
                    idx,
                );
                row.extend(
                    out[0]
                        .iter_nonzero()
                        .map(|(t, c)| (local as u32, t as u32, c)),
                );
            }
            block.push_row(row);
        }
        block
    }
}
