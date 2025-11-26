// This generates better llvm optimization
#![allow(clippy::int_plus_one)]

use std::ops::Range;

use crate::{
    field::{Field, element::FieldElement},
    limb::Limb,
    prime::{Prime, ValidPrime},
    vector::{
        iter::FqVectorIterator,
        repr::{CowRepr, OwnedRepr, Repr, ReprKind, ReprMut, ViewMutRepr, ViewRepr},
    },
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

    pub fn as_slice(&self) -> FqSlice<'_, F> {
        self.slice(0, self.len())
    }

    /// TODO: implement prime 2 version
    pub fn iter(&self) -> FqVectorIterator<'_, F> {
        FqVectorIterator::new(self.as_slice())
    }

    pub fn is_zero(&self) -> bool {
        if R::repr_kind() == ReprKind::Owned {
            return self.limbs().iter().all(|&x| x == 0);
        }

        let limb_range = self.limb_range();
        if limb_range.is_empty() {
            return true;
        }
        let (min_mask, max_mask) = self.limb_masks();
        if self.limbs()[limb_range.start] & min_mask != 0 {
            return false;
        }

        let inner_range = self.limb_range_inner();
        if !inner_range.is_empty() && self.limbs()[inner_range].iter().any(|&x| x != 0) {
            return false;
        }
        if self.limbs()[limb_range.end - 1] & max_mask != 0 {
            return false;
        }
        true
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

    // Repr accessors

    pub(super) fn start(&self) -> usize {
        self.repr.start()
    }

    pub(super) fn end(&self) -> usize {
        self.repr.end()
    }

    pub(super) fn limbs(&self) -> &[Limb] {
        self.repr.limbs()
    }

    // Limb methods

    #[inline]
    pub(super) fn offset(&self) -> usize {
        let bit_length = self.fq().bit_length();
        let entries_per_limb = self.fq().entries_per_limb();
        (self.start() % entries_per_limb) * bit_length
    }

    #[inline]
    pub(super) fn limb_range(&self) -> Range<usize> {
        self.fq().range(self.start(), self.end())
    }

    /// This function underflows if `self.end() == 0`, which happens if and only if we are taking a
    /// slice of width 0 at the start of an `FpVector`. This should be a very rare edge case.
    /// Dealing with the underflow properly would probably require using `saturating_sub` or
    /// something of that nature, and that has a nontrivial (10%) performance hit.
    #[inline]
    pub(super) fn limb_range_inner(&self) -> Range<usize> {
        let range = self.limb_range();
        (range.start + 1)..(usize::max(range.start + 1, range.end - 1))
    }

    #[inline(always)]
    pub(super) fn min_limb_mask(&self) -> Limb {
        !0 << self.offset()
    }

    #[inline(always)]
    pub(super) fn max_limb_mask(&self) -> Limb {
        let num_entries = 1 + (self.end() - 1) % self.fq().entries_per_limb();
        let bit_max = num_entries * self.fq().bit_length();

        (!0) >> (crate::constants::BITS_PER_LIMB - bit_max)
    }

    #[inline(always)]
    pub(super) fn limb_masks(&self) -> (Limb, Limb) {
        if self.limb_range().len() == 1 {
            (
                self.min_limb_mask() & self.max_limb_mask(),
                self.min_limb_mask() & self.max_limb_mask(),
            )
        } else {
            (self.min_limb_mask(), self.max_limb_mask())
        }
    }
}

impl<R: ReprMut, F: Field> FqVectorBase<R, F> {
    #[inline]
    #[must_use]
    pub fn as_slice_mut(&mut self) -> FqSliceMut<'_, F> {
        self.slice_mut(0, self.len())
    }

    pub fn set_to_zero(&mut self) {
        if R::repr_kind() == ReprKind::Owned {
            // This is sound because `fq.encode(fq.zero())` is always zero.
            for limb in self.limbs_mut() {
                *limb = 0;
            }
            return;
        }

        let limb_range = self.limb_range();
        if limb_range.is_empty() {
            return;
        }
        let (min_mask, max_mask) = self.limb_masks();
        self.limbs_mut()[limb_range.start] &= !min_mask;

        let inner_range = self.limb_range_inner();
        for limb in self.limbs_mut()[inner_range].iter_mut() {
            *limb = 0;
        }
        self.limbs_mut()[limb_range.end - 1] &= !max_mask;
    }

    pub fn scale(&mut self, c: FieldElement<F>) {
        assert_eq!(self.fq(), c.field());
        let fq = self.fq();

        if c == fq.zero() {
            self.set_to_zero();
        }

        if fq.q() == 2 {
            return;
        }

        if R::repr_kind() == ReprKind::Owned {
            for limb in self.limbs_mut() {
                *limb = fq.fma_limb(0, *limb, c.clone());
            }
        } else {
            let limb_range = self.limb_range();
            if limb_range.is_empty() {
                return;
            }
            let (min_mask, max_mask) = self.limb_masks();

            let limb = self.limbs()[limb_range.start];
            let masked_limb = limb & min_mask;
            let rest_limb = limb & !min_mask;
            self.limbs_mut()[limb_range.start] = fq.fma_limb(0, masked_limb, c.clone()) | rest_limb;

            let inner_range = self.limb_range_inner();
            for limb in self.limbs_mut()[inner_range].iter_mut() {
                *limb = fq.fma_limb(0, *limb, c.clone());
            }
            if limb_range.len() > 1 {
                let full_limb = self.limbs()[limb_range.end - 1];
                let masked_limb = full_limb & max_mask;
                let rest_limb = full_limb & !max_mask;
                self.limbs_mut()[limb_range.end - 1] = fq.fma_limb(0, masked_limb, c) | rest_limb;
            }
        }

        self.reduce_limbs();
    }

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

    pub fn add_basis_element(&mut self, index: usize, value: FieldElement<F>) {
        assert_eq!(self.fq(), value.field());
        if self.fq().q() == 2 {
            let pair = self.fq().limb_bit_index_pair(index + self.start());
            self.limbs_mut()[pair.limb] ^= self.fq().encode(value) << pair.bit_index;
        } else {
            let mut x = self.entry(index);
            x += value;
            self.set_entry(index, x);
        }
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

    pub(super) fn reduce_limbs(&mut self) {
        let fq = self.fq();
        if fq.q() != 2 {
            let limb_range = self.limb_range();

            for limb in self.limbs_mut()[limb_range].iter_mut() {
                *limb = fq.reduce(*limb);
            }
        }
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
