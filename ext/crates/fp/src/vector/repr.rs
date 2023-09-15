use crate::{
    constants,
    limb::{self, Limb},
};

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
}

pub trait Repr: Sized {
    fn start(&self) -> usize;

    fn end(&self) -> usize;

    fn limbs(&self) -> &[Limb];

    fn repr_kind() -> ReprKind;

    fn to_owned(self) -> OwnedRepr {
        OwnedRepr {
            limbs: Vec::from(self.limbs()),
            len: self.len(),
        }
    }

    fn is_aligned<const P: u32>(&self) -> bool {
        self.start() % limb::entries_per_limb_const::<P>() == 0
    }

    fn len(&self) -> usize {
        self.end() - self.start()
    }
}

pub trait ReprMut: Repr {
    fn limbs_mut(&mut self) -> &mut [Limb];
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

    pub(super) fn extend_len<const P: u32>(&mut self, len: usize) {
        if self.len >= len {
            return;
        }
        self.len = len;
        self.limbs.resize(limb::number::<P>(len), 0);
    }

    pub(super) fn set_size<const P: u32>(&mut self, len: usize) {
        self.limbs.clear();
        self.limbs.resize(limb::number::<P>(len), 0);
        self.len = len;
    }

    pub(super) fn copy_from_slice<const P: u32>(&mut self, slice: &[u32]) {
        assert_eq!(self.len(), slice.len());

        self.limbs.clear();
        self.limbs.extend(
            slice
                .chunks(limb::entries_per_limb_const::<P>())
                .map(|x| limb::pack::<_, P>(x.iter().copied())),
        );
    }

    pub(super) fn trim_start<const P: u32>(&mut self, n: usize) {
        assert!(n <= self.len);
        let entries_per = limb::entries_per_limb_const::<P>();
        assert_eq!(n % entries_per, 0);
        let num_limbs = n / entries_per;
        self.limbs.drain(0..num_limbs);
        self.len -= n;
    }

    pub(super) fn shift_bits(&mut self, bit_shift: usize) {
        let num_to_trim = bit_shift / constants::BITS_PER_LIMB;
        self.limbs.drain(0..num_to_trim);

        let sub_limb_shift = bit_shift % constants::BITS_PER_LIMB;
        if sub_limb_shift == 0 {
            return;
        }

        let carryover_mask = (1 << sub_limb_shift) - 1;
        let mut carryover_cur;
        let mut carryover_prev = 0;
        for limb in self.limbs.iter_mut().rev() {
            carryover_cur = *limb & carryover_mask;
            *limb >>= sub_limb_shift;
            *limb |= carryover_prev << (constants::BITS_PER_LIMB - sub_limb_shift);
            carryover_prev = carryover_cur;
        }
    }
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

    fn is_aligned<const P: u32>(&self) -> bool {
        true
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
