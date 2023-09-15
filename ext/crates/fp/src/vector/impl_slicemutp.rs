use crate::limb::Limb;

use super::{
    inner::{FpVectorBaseP, SliceMutP, SliceP},
    repr::{ReprMut, ViewMutRepr},
};

impl<'a, const P: u32> SliceMutP<'a, P> {
    pub(super) fn new(limbs: &'a mut [Limb], start: usize, end: usize) -> Self {
        let data = ViewMutRepr::new(limbs, start, end);
        Self { data }
    }

    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> SliceP<'_, P> {
        assert!(start <= end && end <= self.len());
        SliceP::new(self.limbs(), self.start(), self.end())
    }

    /// `coeff` need not be reduced mod p.
    /// Adds v otimes w to self.
    pub fn add_tensor(&mut self, offset: usize, coeff: u32, left: SliceP<P>, right: SliceP<P>) {
        let right_dim = right.len();

        for (i, v) in left.iter_nonzero() {
            let entry = (v * coeff) % *self.prime();
            self.slice_mut(offset + i * right_dim, offset + (i + 1) * right_dim)
                .add(right, entry);
        }
    }

    /// Generates a version of itself with a shorter lifetime
    #[inline]
    #[must_use]
    pub fn copy(&mut self) -> SliceMutP<'_, P> {
        self.as_slice_mut()
    }
}

impl<'a, R: ReprMut, const P: u32> From<&'a mut FpVectorBaseP<R, P>> for SliceMutP<'a, P> {
    fn from(v: &'a mut FpVectorBaseP<R, P>) -> Self {
        v.as_slice_mut()
    }
}
