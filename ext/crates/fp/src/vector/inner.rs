// This generates better llvm optimization
#![allow(clippy::int_plus_one)]

use crate::{
    field::{Field, element::FieldElement},
    limb::Limb,
    prime::{Prime, ValidPrime},
    vector::repr::{CowRepr, OwnedRepr, Repr, ReprMut, ViewMutRepr, ViewRepr},
};

// /// A vector over a finite field.
// ///
// /// Interally, it packs entries of the vectors into limbs. However, this is an abstraction that must
// /// not leave the `fp` library.
// #[derive(Debug, Hash, Eq, PartialEq, Clone)]
// pub struct FqVector<F: Field> {
//     fq: F,
//     len: usize,
//     limbs: Vec<Limb>,
// }

// /// A slice of an `FqVector`.
// ///
// /// This immutably borrows the vector and implements `Copy`.
// #[derive(Debug, Copy, Clone)]
// pub struct FqSlice<'a, F: Field> {
//     fq: F,
//     limbs: &'a [Limb],
//     start: usize,
//     end: usize,
// }

// /// A mutable slice of an `FqVector`.
// ///
// /// This mutably borrows the vector. Since it is a mutable borrow, it cannot implement `Copy`.
// /// However, it has a [`FqSliceMut::copy`] function that imitates the reborrowing, that mutably
// /// borrows `FqSliceMut` and returns a `FqSliceMut` with a shorter lifetime.
// #[derive(Debug)]
// pub struct FqSliceMut<'a, F: Field> {
//     fq: F,
//     limbs: &'a mut [Limb],
//     start: usize,
//     end: usize,
// }

// // See impl_* for implementations

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FqVectorBase<R, F> {
    fq: F,
    repr: R,
}

pub type FqVector<F> = FqVectorBase<OwnedRepr, F>;
pub type FqSlice<'a, F> = FqVectorBase<ViewRepr<'a>, F>;
pub type FqSliceMut<'a, F> = FqVectorBase<ViewMutRepr<'a>, F>;
pub type FqCow<'a, F> = FqVectorBase<CowRepr<'a>, F>;

impl<R: Repr, F: Field> FqVectorBase<R, F> {
    pub fn fq(&self) -> F {
        self.fq
    }

    pub fn prime(&self) -> ValidPrime {
        self.fq().characteristic().to_dyn()
    }

    pub fn len(&self) -> usize {
        self.repr.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> FqSlice<'_, F> {
        assert!(start <= end && end <= self.len());

        FqSlice::new(
            self.fq(),
            self.limbs(),
            self.start() + start,
            self.start() + end,
        )
    }

    pub fn entry(&self, index: usize) -> FieldElement<F> {
        debug_assert!(
            index < self.len(),
            "Index {} too large, length of vector is only {}.",
            index,
            self.len()
        );
        let bit_mask = self.fq().bitmask();
        let limb_index = self.fq().limb_bit_index_pair(index + self.start());
        let mut result = self.limbs()[limb_index.limb];
        result >>= limb_index.bit_index;
        result &= bit_mask;
        self.fq().decode(result)
    }

    pub(super) fn start(&self) -> usize {
        self.repr.start()
    }

    pub(super) fn end(&self) -> usize {
        self.repr.end()
    }

    pub(super) fn limbs(&self) -> &[Limb] {
        self.repr.limbs()
    }
}

impl<R: ReprMut, F: Field> FqVectorBase<R, F> {
    pub fn set_entry(&mut self, index: usize, value: FieldElement<F>) {
        assert_eq!(self.fq(), value.field());
        assert!(index < self.len());

        let bit_mask = self.fq().bitmask();
        let limb_index = self.fq().limb_bit_index_pair(index + self.start());

        let mut result = self.limbs()[limb_index.limb];
        result &= !(bit_mask << limb_index.bit_index);
        result |= self.fq().encode(value) << limb_index.bit_index;
        self.limbs_mut()[limb_index.limb] = result;
    }

    #[must_use]
    pub fn slice_mut(&mut self, start: usize, end: usize) -> FqSliceMut<'_, F> {
        assert!(start <= end && end <= self.len());
        let orig_start = self.start();

        FqSliceMut::new(
            self.fq(),
            self.limbs_mut(),
            orig_start + start,
            orig_start + end,
        )
    }

    pub(super) fn limbs_mut(&mut self) -> &mut [Limb] {
        self.repr.limbs_mut()
    }
}

// Accessors

impl<F: Field> FqVector<F> {
    pub fn from_raw_parts(fq: F, len: usize, limbs: Vec<Limb>) -> Self {
        debug_assert_eq!(limbs.len(), fq.number(len));
        let repr = OwnedRepr::new(limbs, len);
        Self { fq, repr }
    }

    pub(super) fn vec_mut(&mut self) -> &mut Vec<Limb> {
        self.repr.vec()
    }

    pub(super) fn len_mut(&mut self) -> &mut usize {
        self.repr.len_mut()
    }
}

impl<'a, F: Field> FqSlice<'a, F> {
    pub(super) fn new(fq: F, limbs: &'a [Limb], start: usize, end: usize) -> Self {
        Self {
            fq,
            repr: ViewRepr::new(limbs, start, end),
        }
    }

    pub(super) fn into_limbs(self) -> &'a [Limb] {
        self.repr.into_limbs()
    }
}

impl<'a, F: Field> FqSliceMut<'a, F> {
    pub(super) fn new(fq: F, limbs: &'a mut [Limb], start: usize, end: usize) -> Self {
        Self {
            fq,
            repr: ViewMutRepr::new(limbs, start, end),
        }
    }
}
