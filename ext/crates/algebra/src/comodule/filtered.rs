use std::sync::Arc;

use fp::vector::{FpSliceMut, FpVector};

use super::{ActionBlock, Comodule, OpSet};

/// A comodule with an increasing filtration that the action does not raise, whose basis in each
/// degree is sorted by filtration, so that each filtration piece is a prefix.
pub trait FilteredComodule: Comodule {
    /// The filtration of a basis element.
    fn filtration(&self, degree: i32, idx: usize) -> i32;

    /// The number of basis elements of `degree` of filtration at most `f`.
    fn filtration_dimension(&self, degree: i32, f: i32) -> usize {
        let dim = self.dimension(degree);
        // The basis is sorted by filtration, so this is a partition point.
        let (mut lo, mut hi) = (0, dim);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.filtration(degree, mid) <= f {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// The part of filtration at most `max` of a [`FilteredComodule`], a subcomodule.
pub struct SubComodule<M> {
    parent: Arc<M>,
    max: i32,
}

/// The part of filtration above `min` of a [`FilteredComodule`], a quotient comodule.
pub struct QuotientComodule<M> {
    parent: Arc<M>,
    min: i32,
}

impl<M: FilteredComodule> SubComodule<M> {
    /// The part of `parent` of filtration at most `max`.
    pub fn new(parent: Arc<M>, max: i32) -> Self {
        Self { parent, max }
    }
}

impl<M: FilteredComodule> QuotientComodule<M> {
    /// The part of `parent` of filtration above `min`.
    pub fn new(parent: Arc<M>, min: i32) -> Self {
        Self { parent, min }
    }

    /// The index in the parent of the first basis element of `degree` kept.
    fn start(&self, degree: i32) -> usize {
        self.parent.filtration_dimension(degree, self.min)
    }
}

impl<M: FilteredComodule> std::fmt::Display for SubComodule<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "F_{} {}", self.max, self.parent)
    }
}

impl<M: FilteredComodule> std::fmt::Display for QuotientComodule<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} / F_{}", self.parent, self.min)
    }
}

impl<M: FilteredComodule> Comodule for SubComodule<M> {
    type Algebra = M::Algebra;

    fn algebra(&self) -> Arc<M::Algebra> {
        self.parent.algebra()
    }

    fn min_degree(&self) -> i32 {
        self.parent.min_degree()
    }

    fn compute_basis(&self, degree: i32) {
        self.parent.compute_basis(degree);
    }

    fn max_computed_degree(&self) -> i32 {
        self.parent.max_computed_degree()
    }

    fn dimension(&self, degree: i32) -> usize {
        self.parent.filtration_dimension(degree, self.max)
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        self.parent.basis_element_to_string(degree, idx)
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
        let target = degree - op_degree;
        let mut full = FpVector::new(self.prime(), self.parent.dimension(target));
        self.parent
            .act_on_basis(full.as_slice_mut(), coeff, op_degree, op_index, degree, idx);
        result.add(full.slice(0, self.dimension(target)), 1);
    }

    fn act_batch(&self, ops: &OpSet, degree: i32) -> ActionBlock {
        // The action does not raise filtration, so the rows kept never reach past the prefix.
        let dim = self.dimension(degree);
        self.parent
            .act_batch(ops, degree)
            .restrict(0..dim, 0..usize::MAX)
    }
}

impl<M: FilteredComodule> FilteredComodule for SubComodule<M> {
    fn filtration(&self, degree: i32, idx: usize) -> i32 {
        self.parent.filtration(degree, idx)
    }
}

impl<M: FilteredComodule> Comodule for QuotientComodule<M> {
    type Algebra = M::Algebra;

    fn algebra(&self) -> Arc<M::Algebra> {
        self.parent.algebra()
    }

    fn min_degree(&self) -> i32 {
        self.parent.min_degree()
    }

    fn compute_basis(&self, degree: i32) {
        self.parent.compute_basis(degree);
    }

    fn max_computed_degree(&self) -> i32 {
        self.parent.max_computed_degree()
    }

    fn dimension(&self, degree: i32) -> usize {
        self.parent.dimension(degree) - self.start(degree)
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        self.parent
            .basis_element_to_string(degree, idx + self.start(degree))
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
        let target = degree - op_degree;
        let mut full = FpVector::new(self.prime(), self.parent.dimension(target));
        self.parent.act_on_basis(
            full.as_slice_mut(),
            coeff,
            op_degree,
            op_index,
            degree,
            idx + self.start(degree),
        );
        let start = self.start(target);
        result.add(full.slice(start, full.len()), 1);
    }

    fn act_batch(&self, ops: &OpSet, degree: i32) -> ActionBlock {
        // Targets are numbered in their own degree, so each operation's degree sets the cut; the
        // parent's block is filtered entry by entry.
        let block = self.parent.act_batch(ops, degree);
        let start = self.start(degree);
        let mut out = ActionBlock::new(self.prime());
        for idx in start..self.parent.dimension(degree) {
            let row: Vec<(u32, u32, u32)> = block
                .row(idx)
                .flat_map(|(local, entry)| {
                    let cut = self.start(degree - ops.get(local as usize).degree);
                    entry
                        .iter()
                        .filter(move |&(t, _)| t >= cut)
                        .map(move |(t, c)| (local, (t - cut) as u32, c))
                        .collect::<Vec<_>>()
                })
                .collect();
            out.push_row(row);
        }
        out
    }
}

impl<M: FilteredComodule> FilteredComodule for QuotientComodule<M> {
    fn filtration(&self, degree: i32, idx: usize) -> i32 {
        self.parent.filtration(degree, idx + self.start(degree))
    }
}
