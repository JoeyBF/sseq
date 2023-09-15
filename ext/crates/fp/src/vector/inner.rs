// This generates better llvm optimization
#![allow(clippy::int_plus_one)]

use std::cmp::Ordering;

use itertools::Itertools;

use crate::{
    constants,
    limb::{self, Limb},
    prime::ValidPrime,
    simd,
};

use super::{
    iter::{FpVectorIterator, FpVectorNonZeroIteratorP},
    repr::{OwnedRepr, Repr, ReprKind, ReprMut, ViewMutRepr, ViewRepr},
};

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct FpVectorBaseP<R, const P: u32> {
    pub(super) data: R,
}

/// An `FpVectorP` is a vector over $\mathbb{F}_p$ for a fixed prime, implemented using const
/// generics. Due to limitations with const generics, we cannot constrain P to actually be a prime,
/// so we allow it to be any u32. However, most functions will panic if P is not a prime.
///
/// Interally, it packs entries of the vectors into limbs. However, this is an abstraction that
/// must not leave the `fp` library.
pub type FpVectorP<const P: u32> = FpVectorBaseP<OwnedRepr, P>;

/// A SliceP is a slice of an FpVectorP. This immutably borrows the vector and implements Copy
pub type SliceP<'a, const P: u32> = FpVectorBaseP<ViewRepr<'a>, P>;

/// A `SliceMutP` is a mutable slice of an `FpVectorP`. This mutably borrows the vector. Since it
/// is a mutable borrow, it cannot implement `Copy`. However, it has a [`SliceMutP::copy`] function
/// that imitates the reborrowing, that mutably borrows `SliceMutP` and returns a `SliceMutP` with
/// a shorter lifetime.
pub type SliceMutP<'a, const P: u32> = FpVectorBaseP<ViewMutRepr<'a>, P>;

// See impl_* for more specific implementations on these type aliases.

// Public methods

impl<R: Repr, const P: u32> FpVectorBaseP<R, P> {
    pub const fn prime(&self) -> ValidPrime {
        ValidPrime::new(P)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
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

    pub fn entry(&self, index: usize) -> u32 {
        debug_assert!(
            index < self.len(),
            "Index {} too large, length of vector is only {}.",
            index,
            self.len()
        );
        let bit_mask = limb::bitmask::<P>();
        let limb_index = limb::limb_bit_index_pair::<P>(index + self.start());
        let mut result = self.limbs()[limb_index.limb];
        result >>= limb_index.bit_index;
        result &= bit_mask;
        result as u32
    }

    pub fn as_slice(&self) -> SliceP<'_, P> {
        SliceP::new(self.limbs(), self.start(), self.end())
    }

    pub fn iter(&self) -> FpVectorIterator {
        FpVectorIterator::new(self.as_slice())
        // self.as_slice().iter()
    }

    pub fn iter_nonzero(&self) -> FpVectorNonZeroIteratorP<'_, P> {
        FpVectorNonZeroIteratorP::new(self)
        // self.as_slice().iter_nonzero()
    }

    /// Converts a slice to an owned FpVectorP. This is vastly more efficient if the start of the vector is aligned.
    #[must_use]
    pub fn to_owned(self) -> FpVectorP<P> {
        let bit_shift = self.start() * limb::bit_length_const::<P>();
        let mut owned = self.data.to_owned();
        owned.shift_bits(bit_shift);
        FpVectorP::from_data(owned)
    }
}

impl<R: ReprMut, const P: u32> FpVectorBaseP<R, P> {
    pub fn add_basis_element(&mut self, index: usize, value: u32) {
        if P == 2 {
            // Checking for value % 2 == 0 appears to be less performant
            let pair = limb::limb_bit_index_pair::<2>(index + self.start());
            self.limbs_mut()[pair.limb] ^= (value as Limb % 2) << pair.bit_index;
        } else {
            let mut x = self.entry(index);
            x += value;
            x %= P;
            self.set_entry(index, x);
        }
    }

    pub fn add<R2: Repr>(&mut self, other: impl Into<FpVectorBaseP<R2, P>>, c: u32) {
        self.add_offset(other, c, 0);
    }

    /// Add `other` to `self` on the assumption that the first `offset` entries of `other` are
    /// empty.
    pub fn add_offset<R2: Repr>(
        &mut self,
        other: impl Into<FpVectorBaseP<R2, P>>,
        c: u32,
        offset: usize,
    ) {
        let other = other.into();
        assert_eq!(self.len(), other.len());
        if R::repr_kind() == ReprKind::Owned && R2::repr_kind() == ReprKind::Owned {
            let min_limb = offset / limb::entries_per_limb_const::<P>();
            if P == 2 {
                if c != 0 {
                    simd::add_simd(self.limbs_mut(), other.limbs(), min_limb);
                }
            } else {
                for (left, right) in self
                    .limbs_mut()
                    .iter_mut()
                    .zip_eq(other.limbs())
                    .skip(min_limb)
                {
                    *left = limb::add::<P>(*left, *right, c);
                }
                for limb in &mut self.limbs_mut()[min_limb..] {
                    *limb = limb::reduce::<P>(*limb);
                }
            }
        } else {
            debug_assert!(c < P);
            if self.as_slice().is_empty() {
                return;
            }

            if P == 2 {
                if c != 0 {
                    match self.as_slice().offset().cmp(&other.offset()) {
                        Ordering::Equal => self.add_shift_none(other, 1),
                        Ordering::Less => self.add_shift_left(other, 1),
                        Ordering::Greater => self.add_shift_right(other, 1),
                    };
                }
            } else {
                match self.as_slice().offset().cmp(&other.offset()) {
                    Ordering::Equal => self.add_shift_none(other, c),
                    Ordering::Less => self.add_shift_left(other, c),
                    Ordering::Greater => self.add_shift_right(other, c),
                };
            }
        }
    }

    /// Adds `c` * `other` to `self`. `other` must have the same length, offset, and prime as self, and `c` must be between `0` and `p - 1`.
    pub fn add_shift_none<R2: Repr>(&mut self, other: impl Into<FpVectorBaseP<R2, P>>, c: u32) {
        let other = other.into();
        let target_range = self.limb_range();
        let source_range = other.limb_range();

        let (min_mask, max_mask) = other.limb_masks();

        self.limbs_mut()[target_range.start] = limb::add::<P>(
            self.limbs()[target_range.start],
            other.limbs()[source_range.start] & min_mask,
            c,
        );
        self.limbs_mut()[target_range.start] = limb::reduce::<P>(self.limbs()[target_range.start]);

        let target_inner_range = self.as_slice().limb_range_inner();
        let source_inner_range = other.limb_range_inner();
        if !source_inner_range.is_empty() {
            // We need to clone here because `Range` doesn't implement Copy, even though it's just
            // two numbers. This is because Rust doesn't make any iterator Copy as it can lead to
            // unexpected behavior.
            for (left, right) in self.limbs_mut()[target_inner_range.clone()]
                .iter_mut()
                .zip_eq(&other.limbs()[source_inner_range])
            {
                *left = limb::add::<P>(*left, *right, c);
            }
            for left in &mut self.limbs_mut()[target_inner_range] {
                *left = limb::reduce::<P>(*left);
            }
        }
        if source_range.len() > 1 {
            // The first and last limbs are distinct, so we process the last.
            self.limbs_mut()[target_range.end - 1] = limb::add::<P>(
                self.limbs()[target_range.end - 1],
                other.limbs()[source_range.end - 1] & max_mask,
                c,
            );
            self.limbs_mut()[target_range.end - 1] =
                limb::reduce::<P>(self.limbs()[target_range.end - 1]);
        }
    }

    fn add_shift_left<R2: Repr>(&mut self, other: FpVectorBaseP<R2, P>, c: u32) {
        struct AddShiftLeftData {
            offset_shift: usize,
            tail_shift: usize,
            zero_bits: usize,
            min_source_limb: usize,
            min_target_limb: usize,
            number_of_source_limbs: usize,
            number_of_target_limbs: usize,
            min_mask: Limb,
            max_mask: Limb,
        }

        impl AddShiftLeftData {
            fn new<R3: Repr, R4: Repr, const P: u32>(
                target: &FpVectorBaseP<R3, P>,
                source: &FpVectorBaseP<R4, P>,
            ) -> Self {
                debug_assert!(target.prime() == source.prime());
                debug_assert!(target.offset() <= source.offset());
                debug_assert!(
                    target.len() == source.len(),
                    "self.dim {} not equal to other.dim {}",
                    target.len(),
                    source.len()
                );
                let offset_shift = source.offset() - target.offset();
                let bit_length = limb::bit_length_const::<P>();
                let entries_per_limb = limb::entries_per_limb_const::<P>();
                let usable_bits_per_limb = bit_length * entries_per_limb;
                let tail_shift = usable_bits_per_limb - offset_shift;
                let zero_bits = constants::BITS_PER_LIMB - usable_bits_per_limb;
                let source_range = source.limb_range();
                let target_range = target.limb_range();
                let min_source_limb = source_range.start;
                let min_target_limb = target_range.start;
                let number_of_source_limbs = source_range.len();
                let number_of_target_limbs = target_range.len();
                let (min_mask, max_mask) = source.limb_masks();

                Self {
                    offset_shift,
                    tail_shift,
                    zero_bits,
                    min_source_limb,
                    min_target_limb,
                    number_of_source_limbs,
                    number_of_target_limbs,
                    min_mask,
                    max_mask,
                }
            }

            fn mask_first_limb<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                (other.limbs()[i] & self.min_mask) >> self.offset_shift
            }

            fn mask_middle_limb_a<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                other.limbs()[i] >> self.offset_shift
            }

            fn mask_middle_limb_b<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                (other.limbs()[i] << (self.tail_shift + self.zero_bits)) >> self.zero_bits
            }

            fn mask_last_limb_a<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                let source_limb_masked = other.limbs()[i] & self.max_mask;
                source_limb_masked << self.tail_shift
            }

            fn mask_last_limb_b<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                let source_limb_masked = other.limbs()[i] & self.max_mask;
                source_limb_masked >> self.offset_shift
            }
        }

        let dat = AddShiftLeftData::new(self, &other);
        let mut i = 0;
        {
            self.limbs_mut()[i + dat.min_target_limb] = limb::add::<P>(
                self.limbs()[i + dat.min_target_limb],
                dat.mask_first_limb(&other, i + dat.min_source_limb),
                c,
            );
        }
        for i in 1..dat.number_of_source_limbs - 1 {
            self.limbs_mut()[i + dat.min_target_limb] = limb::add::<P>(
                self.limbs()[i + dat.min_target_limb],
                dat.mask_middle_limb_a(&other, i + dat.min_source_limb),
                c,
            );
            self.limbs_mut()[i + dat.min_target_limb - 1] = limb::add::<P>(
                self.limbs()[i + dat.min_target_limb - 1],
                dat.mask_middle_limb_b(&other, i + dat.min_source_limb),
                c,
            );
            self.limbs_mut()[i + dat.min_target_limb - 1] =
                limb::reduce::<P>(self.limbs()[i + dat.min_target_limb - 1]);
        }
        i = dat.number_of_source_limbs - 1;
        if i > 0 {
            self.limbs_mut()[i + dat.min_target_limb - 1] = limb::add::<P>(
                self.limbs()[i + dat.min_target_limb - 1],
                dat.mask_last_limb_a(&other, i + dat.min_source_limb),
                c,
            );
            self.limbs_mut()[i + dat.min_target_limb - 1] =
                limb::reduce::<P>(self.limbs()[i + dat.min_target_limb - 1]);
            if dat.number_of_source_limbs == dat.number_of_target_limbs {
                self.limbs_mut()[i + dat.min_target_limb] = limb::add::<P>(
                    self.limbs()[i + dat.min_target_limb],
                    dat.mask_last_limb_b(&other, i + dat.min_source_limb),
                    c,
                );
                self.limbs_mut()[i + dat.min_target_limb] =
                    limb::reduce::<P>(self.limbs()[i + dat.min_target_limb]);
            }
        } else {
            self.limbs_mut()[i + dat.min_target_limb] =
                limb::reduce::<P>(self.limbs()[i + dat.min_target_limb]);
        }
    }

    fn add_shift_right<R2: Repr>(&mut self, other: FpVectorBaseP<R2, P>, c: u32) {
        struct AddShiftRightData {
            offset_shift: usize,
            tail_shift: usize,
            zero_bits: usize,
            min_source_limb: usize,
            min_target_limb: usize,
            number_of_source_limbs: usize,
            number_of_target_limbs: usize,
            min_mask: Limb,
            max_mask: Limb,
        }

        impl AddShiftRightData {
            fn new<R3: Repr, R4: Repr, const P: u32>(
                target: &FpVectorBaseP<R3, P>,
                source: &FpVectorBaseP<R4, P>,
            ) -> Self {
                debug_assert!(target.prime() == source.prime());
                debug_assert!(target.offset() >= source.offset());
                debug_assert!(
                    target.len() == source.len(),
                    "self.dim {} not equal to other.dim {}",
                    target.len(),
                    source.len()
                );
                let offset_shift = target.offset() - source.offset();
                let bit_length = limb::bit_length_const::<P>();
                let entries_per_limb = limb::entries_per_limb_const::<P>();
                let usable_bits_per_limb = bit_length * entries_per_limb;
                let tail_shift = usable_bits_per_limb - offset_shift;
                let zero_bits = constants::BITS_PER_LIMB - usable_bits_per_limb;
                let source_range = source.limb_range();
                let target_range = target.limb_range();
                let min_source_limb = source_range.start;
                let min_target_limb = target_range.start;
                let number_of_source_limbs = source_range.len();
                let number_of_target_limbs = target_range.len();
                let (min_mask, max_mask) = source.limb_masks();
                Self {
                    offset_shift,
                    tail_shift,
                    zero_bits,
                    min_source_limb,
                    min_target_limb,
                    number_of_source_limbs,
                    number_of_target_limbs,
                    min_mask,
                    max_mask,
                }
            }

            fn mask_first_limb_a<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                let source_limb_masked = other.limbs()[i] & self.min_mask;
                (source_limb_masked << (self.offset_shift + self.zero_bits)) >> self.zero_bits
            }

            fn mask_first_limb_b<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                let source_limb_masked = other.limbs()[i] & self.min_mask;
                source_limb_masked >> self.tail_shift
            }

            fn mask_middle_limb_a<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                (other.limbs()[i] << (self.offset_shift + self.zero_bits)) >> self.zero_bits
            }

            fn mask_middle_limb_b<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                other.limbs()[i] >> self.tail_shift
            }

            fn mask_last_limb_a<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                let source_limb_masked = other.limbs()[i] & self.max_mask;
                source_limb_masked << self.offset_shift
            }

            fn mask_last_limb_b<R3: Repr, const P: u32>(
                &self,
                other: &FpVectorBaseP<R3, P>,
                i: usize,
            ) -> Limb {
                let source_limb_masked = other.limbs()[i] & self.max_mask;
                source_limb_masked >> self.tail_shift
            }
        }

        let dat = AddShiftRightData::new(self, &other);
        let mut i = 0;
        {
            self.limbs_mut()[i + dat.min_target_limb] = limb::add::<P>(
                self.limbs()[i + dat.min_target_limb],
                dat.mask_first_limb_a(&other, i + dat.min_source_limb),
                c,
            );
            self.limbs_mut()[i + dat.min_target_limb] =
                limb::reduce::<P>(self.limbs()[i + dat.min_target_limb]);
            if dat.number_of_target_limbs > 1 {
                self.limbs_mut()[i + dat.min_target_limb + 1] = limb::add::<P>(
                    self.limbs()[i + dat.min_target_limb + 1],
                    dat.mask_first_limb_b(&other, i + dat.min_source_limb),
                    c,
                );
            }
        }
        for i in 1..dat.number_of_source_limbs - 1 {
            self.limbs_mut()[i + dat.min_target_limb] = limb::add::<P>(
                self.limbs()[i + dat.min_target_limb],
                dat.mask_middle_limb_a(&other, i + dat.min_source_limb),
                c,
            );
            self.limbs_mut()[i + dat.min_target_limb] =
                limb::reduce::<P>(self.limbs()[i + dat.min_target_limb]);
            self.limbs_mut()[i + dat.min_target_limb + 1] = limb::add::<P>(
                self.limbs()[i + dat.min_target_limb + 1],
                dat.mask_middle_limb_b(&other, i + dat.min_source_limb),
                c,
            );
        }
        i = dat.number_of_source_limbs - 1;
        if i > 0 {
            self.limbs_mut()[i + dat.min_target_limb] = limb::add::<P>(
                self.limbs()[i + dat.min_target_limb],
                dat.mask_last_limb_a(&other, i + dat.min_source_limb),
                c,
            );
            self.limbs_mut()[i + dat.min_target_limb] =
                limb::reduce::<P>(self.limbs()[i + dat.min_target_limb]);
            if dat.number_of_target_limbs > dat.number_of_source_limbs {
                self.limbs_mut()[i + dat.min_target_limb + 1] = limb::add::<P>(
                    self.limbs()[i + dat.min_target_limb + 1],
                    dat.mask_last_limb_b(&other, i + dat.min_source_limb),
                    c,
                );
            }
        }
        if dat.number_of_target_limbs > dat.number_of_source_limbs {
            self.limbs_mut()[i + dat.min_target_limb + 1] =
                limb::reduce::<P>(self.limbs()[i + dat.min_target_limb + 1]);
        }
    }

    /// Given a mask v, add the `v[i]`th entry of `other` to the `i`th entry of `self`.
    pub fn add_masked<R2: Repr>(
        &mut self,
        other: impl Into<FpVectorBaseP<R2, P>>,
        c: u32,
        mask: &[usize],
    ) {
        let other = other.into();
        // TODO: If this ends up being a bottleneck, try to use PDEP/PEXT
        assert_eq!(self.len(), mask.len());
        for (i, &x) in mask.iter().enumerate() {
            let entry = other.entry(x);
            if entry != 0 {
                self.add_basis_element(i, entry * c);
            }
        }
    }

    /// Given a mask v, add the `i`th entry of `other` to the `v[i]`th entry of `self`.
    pub fn add_unmasked<R2: Repr>(
        &mut self,
        other: impl Into<FpVectorBaseP<R2, P>>,
        c: u32,
        mask: &[usize],
    ) {
        let other = other.into();
        assert!(other.len() <= mask.len());
        for (i, v) in other.iter_nonzero() {
            self.add_basis_element(mask[i], v * c);
        }
    }

    /// TODO: improve efficiency
    pub fn assign<R2: Repr>(&mut self, other: impl Into<FpVectorBaseP<R2, P>>) {
        let other = other.into();
        debug_assert_eq!(self.len(), other.len());

        if R::repr_kind() == ReprKind::Owned && R2::repr_kind() == ReprKind::Owned {
            self.limbs_mut().copy_from_slice(other.limbs());
            return;
        }

        if self.offset() != other.offset() {
            self.set_to_zero();
            self.add(other, 1);
            return;
        }
        let target_range = self.limb_range();
        let source_range = other.limb_range();

        if target_range.is_empty() {
            return;
        }

        let (min_mask, max_mask) = other.limb_masks();

        let result = other.limbs()[source_range.start] & min_mask;
        self.limbs_mut()[target_range.start] &= !min_mask;
        self.limbs_mut()[target_range.start] |= result;

        let target_inner_range = self.as_slice().limb_range_inner();
        let source_inner_range = other.limb_range_inner();
        if !target_inner_range.is_empty() && !source_inner_range.is_empty() {
            self.limbs_mut()[target_inner_range]
                .copy_from_slice(&other.limbs()[source_inner_range]);
        }

        let result = other.limbs()[source_range.end - 1] & max_mask;
        self.limbs_mut()[target_range.end - 1] &= !max_mask;
        self.limbs_mut()[target_range.end - 1] |= result;
    }

    pub fn set_entry(&mut self, index: usize, value: u32) {
        debug_assert!(index < self.len());
        let bit_mask = limb::bitmask::<P>();
        let limb_index = limb::limb_bit_index_pair::<P>(index + self.start());
        let mut result = self.limbs()[limb_index.limb];
        result &= !(bit_mask << limb_index.bit_index);
        result |= (value as Limb) << limb_index.bit_index;
        self.limbs_mut()[limb_index.limb] = result;
    }

    pub fn set_to_zero(&mut self) {
        if R::repr_kind() == ReprKind::Owned {
            for limb in self.limbs_mut().iter_mut() {
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
        for limb in &mut self.limbs_mut()[inner_range] {
            *limb = 0;
        }
        self.limbs_mut()[limb_range.end - 1] &= !max_mask;
    }

    pub fn scale(&mut self, c: u32) {
        if P == 2 {
            if c == 0 {
                self.set_to_zero();
            }
        } else if R::repr_kind() == ReprKind::Owned {
            match P {
                3 | 5 => {
                    for limb in self.limbs_mut() {
                        *limb = limb::reduce::<P>(*limb * c as Limb);
                    }
                }
                _ => {
                    for limb in self.limbs_mut() {
                        *limb = limb::pack::<_, P>(limb::unpack::<P>(*limb).map(|x| (x * c) % P));
                    }
                }
            }
        } else {
            let c = c as Limb;
            let limb_range = self.limb_range();
            if limb_range.is_empty() {
                return;
            }
            let (min_mask, max_mask) = self.limb_masks();

            let limb = self.limbs()[limb_range.start];
            let masked_limb = limb & min_mask;
            let rest_limb = limb & !min_mask;
            self.limbs_mut()[limb_range.start] = (masked_limb * c) | rest_limb;

            let inner_range = self.limb_range_inner();
            for limb in &mut self.limbs_mut()[inner_range] {
                *limb *= c;
            }
            if limb_range.len() > 1 {
                let full_limb = self.limbs()[limb_range.end - 1];
                let masked_limb = full_limb & max_mask;
                let rest_limb = full_limb & !max_mask;
                self.limbs_mut()[limb_range.end - 1] = (masked_limb * c) | rest_limb;
            }
            self.reduce_limbs();
        }
    }

    #[must_use]
    pub fn slice_mut(&mut self, start: usize, end: usize) -> SliceMutP<'_, P> {
        assert!(start <= end && end <= self.len());
        let new_start = self.start() + start;
        let new_end = self.start() + end;
        SliceMutP::new(self.limbs_mut(), new_start, new_end)
    }

    pub fn as_slice_mut(&mut self) -> SliceMutP<'_, P> {
        self.slice_mut(0, self.len())
    }
}

// Limb methods

impl<R: Repr, const P: u32> FpVectorBaseP<R, P> {
    pub(crate) fn limbs(&self) -> &[Limb] {
        self.data.limbs()
    }

    pub(super) fn start(&self) -> usize {
        self.data.start()
    }

    pub(super) fn end(&self) -> usize {
        self.data.end()
    }

    #[inline]
    pub(super) fn offset(&self) -> usize {
        let bit_length = limb::bit_length_const::<P>();
        let entries_per_limb = limb::entries_per_limb_const::<P>();
        (self.start() % entries_per_limb) * bit_length
    }

    #[inline]
    pub(super) fn limb_range(&self) -> std::ops::Range<usize> {
        limb::range::<P>(self.start(), self.end())
    }

    /// This function underflows if `self.end == 0`, which happens if and only if we are taking a
    /// slice of width 0 at the start of an `FpVector`. This should be a very rare edge case.
    /// Dealing with the underflow properly would probably require using `saturating_sub` or
    /// something of that nature, and that has a nontrivial (10%) performance hit.
    #[inline]
    pub(super) fn limb_range_inner(&self) -> std::ops::Range<usize> {
        let range = self.limb_range();
        (range.start + 1)..(usize::max(range.start + 1, range.end - 1))
    }

    #[inline(always)]
    pub(super) fn min_limb_mask(&self) -> Limb {
        !0 << self.offset()
    }

    #[inline(always)]
    pub(super) fn max_limb_mask(&self) -> Limb {
        let num_entries = 1 + (self.end() - 1) % limb::entries_per_limb_const::<P>();
        let bit_max = num_entries * limb::bit_length_const::<P>();

        (!0) >> (constants::BITS_PER_LIMB - bit_max)
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

impl<R: ReprMut, const P: u32> FpVectorBaseP<R, P> {
    pub(crate) fn limbs_mut(&mut self) -> &mut [Limb] {
        self.data.limbs_mut()
    }

    pub(super) fn reduce_limbs(&mut self) {
        if P != 2 {
            let limb_range = self.limb_range();

            for limb in &mut self.limbs_mut()[limb_range] {
                *limb = limb::reduce::<P>(*limb);
            }
        }
    }
}
