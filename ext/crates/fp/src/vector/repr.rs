use std::borrow::Cow;

// use crate::{constants, limb::Limb};
use crate::limb::Limb;

/// Poor man's specialization. This is a discriminant that functions that take in a generic
/// `FpVectorBaseP<R, P>` can use to optimize for various backing representations.
///
/// For example, the `add` function between fully generic `FpVectorBaseP<R1: ReprMut, P>` and
/// `FpVectorBase<R2: Repr, P>` will have to worry about offsets, masking, etc., while checking that
/// both are just `FpVectorP<P>` allows for a quick simd sum and reduction.
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ReprKind {
    Owned,
    View,
    ViewMut,
    Cow,
}

pub trait Repr: Sized {
    fn start(&self) -> usize;

    fn end(&self) -> usize;

    fn limbs(&self) -> &[Limb];

    fn repr_kind() -> ReprKind;

    fn len(&self) -> usize {
        self.end() - self.start()
    }

    fn slice(&self, start: usize, end: usize) -> ViewRepr<'_> {
        assert!(start <= end && end <= self.len());
        ViewRepr::new(self.limbs(), self.start() + start, self.start() + end)
    }

    fn to_owned(self) -> OwnedRepr {
        OwnedRepr {
            limbs: Vec::from(self.limbs()),
            len: self.len(),
        }
    }
}

pub trait ReprMut: Repr {
    fn limbs_mut(&mut self) -> &mut [Limb];

    fn slice_mut(&mut self, start: usize, end: usize) -> ViewMutRepr<'_> {
        assert!(start <= end && end <= self.len());
        let self_start = self.start(); // Need to capture this before calling `limbs_mut`
        ViewMutRepr::new(self.limbs_mut(), self_start + start, self_start + end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OwnedRepr {
    limbs: Vec<Limb>,
    len: usize,
}

impl OwnedRepr {
    pub(super) fn new(limbs: Vec<Limb>, len: usize) -> Self {
        Self { limbs, len }
    }

    // pub(super) fn slice(&self, start: usize, end: usize) -> ViewRepr<'_> {
    //     ViewRepr::new(&self.limbs, start, end)
    // }

    // /// Shifts the bits of the limbs by `bit_shift` bits. This is used after calling `to_owned` to
    // /// make sure that the limbs are properly aligned. The `to_owned` method can't to this operation
    // /// itself because it doesn't know the bit length of the underlying field.
    // pub(super) fn shift_bits(&mut self, bit_shift: usize) {
    //     let num_to_trim = bit_shift / constants::BITS_PER_LIMB;
    //     self.limbs.drain(0..num_to_trim);

    //     let sub_limb_shift = bit_shift % constants::BITS_PER_LIMB;
    //     if sub_limb_shift == 0 {
    //         return;
    //     }

    //     let carryover_mask = (1 << sub_limb_shift) - 1;
    //     let mut carryover_cur;
    //     let mut carryover_prev = 0;
    //     for limb in self.limbs.iter_mut().rev() {
    //         carryover_cur = *limb & carryover_mask;
    //         *limb >>= sub_limb_shift;
    //         *limb |= carryover_prev << (constants::BITS_PER_LIMB - sub_limb_shift);
    //         carryover_prev = carryover_cur;
    //     }
    // }

    pub(super) fn vec(&mut self) -> &mut Vec<Limb> {
        &mut self.limbs
    }

    pub(super) fn len_mut(&mut self) -> &mut usize {
        &mut self.len
    }

    // pub(super) fn extend_len(&mut self, len: usize, num_limbs: usize) {
    //     if self.len() >= len {
    //         return;
    //     }
    //     self.len = len;
    //     self.limbs.resize(num_limbs, 0);
    // }

    // pub(super) fn set_scratch_vector_size(&mut self, len: usize, num_limbs: usize) {
    //     self.limbs.clear();
    //     self.limbs.resize(num_limbs, 0);
    //     self.len = len;
    // }
}

impl Repr for OwnedRepr {
    fn start(&self) -> usize {
        0
    }

    fn end(&self) -> usize {
        self.len
    }

    fn limbs(&self) -> &[Limb] {
        &self.limbs
    }

    fn repr_kind() -> ReprKind {
        ReprKind::Owned
    }

    fn to_owned(self) -> OwnedRepr {
        self
    }
}

impl ReprMut for OwnedRepr {
    fn limbs_mut(&mut self) -> &mut [Limb] {
        &mut self.limbs
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViewRepr<'a> {
    limbs: &'a [Limb],
    start: usize,
    end: usize,
}

impl<'a> ViewRepr<'a> {
    pub(super) fn new(limbs: &'a [Limb], start: usize, end: usize) -> Self {
        Self { limbs, start, end }
    }

    pub(super) fn into_limbs(self) -> &'a [Limb] {
        self.limbs
    }
}

impl Repr for ViewRepr<'_> {
    fn start(&self) -> usize {
        self.start
    }

    fn end(&self) -> usize {
        self.end
    }

    fn limbs(&self) -> &[Limb] {
        self.limbs
    }

    fn repr_kind() -> ReprKind {
        ReprKind::View
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ViewMutRepr<'a> {
    limbs: &'a mut [Limb],
    start: usize,
    end: usize,
}

impl<'a> ViewMutRepr<'a> {
    pub(super) fn new(limbs: &'a mut [Limb], start: usize, end: usize) -> Self {
        Self { limbs, start, end }
    }
}

impl Repr for ViewMutRepr<'_> {
    fn start(&self) -> usize {
        self.start
    }

    fn end(&self) -> usize {
        self.end
    }

    fn limbs(&self) -> &[Limb] {
        &*self.limbs
    }

    fn repr_kind() -> ReprKind {
        ReprKind::ViewMut
    }
}

impl ReprMut for ViewMutRepr<'_> {
    fn limbs_mut(&mut self) -> &mut [Limb] {
        self.limbs
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CowRepr<'a> {
    limbs: Cow<'a, [Limb]>,
    start: usize,
    end: usize,
}

impl CowRepr<'_> {
    fn ensure_owned(&mut self) {
        if matches!(self.limbs, Cow::Borrowed(_)) {
            self.limbs.to_mut();
            self.end = self.len();
            self.start = 0;
        }
    }
}

impl Repr for CowRepr<'_> {
    fn start(&self) -> usize {
        self.start
    }

    fn end(&self) -> usize {
        self.end
    }

    fn limbs(&self) -> &[Limb] {
        self.limbs.as_ref()
    }

    fn repr_kind() -> ReprKind {
        ReprKind::Cow
    }
}

impl ReprMut for CowRepr<'_> {
    fn limbs_mut(&mut self) -> &mut [Limb] {
        self.ensure_owned();
        self.limbs.to_mut()
    }
}
