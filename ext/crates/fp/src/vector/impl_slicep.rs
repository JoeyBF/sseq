use crate::limb::Limb;

use super::{
    inner::{FpVectorBaseP, SliceP},
    repr::{Repr, ViewRepr},
};

// Public methods

impl<'a, const P: u32> SliceP<'a, P> {
    pub(super) fn new(limbs: &'a [Limb], start: usize, end: usize) -> Self {
        let data = ViewRepr::new(limbs, start, end);
        Self { data }
    }

    /// A variant of `limbs` that takes ownership of self.
    pub(super) fn into_limbs(self) -> &'a [Limb] {
        self.data.into_limbs()
    }

    // We don't implement `slice` directly on `FpVectorBaseP` because `SliceP` takes in `self` while
    // other implementations take in `&self`. We need to take ownership because otherwise the
    // compiler complains that the reference contained in the return value might not live long
    // enough.
    pub fn slice(self, start: usize, end: usize) -> Self {
        let new_start = self.start() + start;
        let new_end = self.start() + end;
        Self::new(self.into_limbs(), new_start, new_end)
    }
}

// Limb methods

impl<'a, const P: u32> SliceP<'a, P> {}

impl<'a, R: Repr, const P: u32> From<&'a FpVectorBaseP<R, P>> for SliceP<'a, P> {
    fn from(v: &'a FpVectorBaseP<R, P>) -> Self {
        SliceP::new(v.limbs(), 0, v.len())
    }
}
