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

    pub(super) fn vec(&mut self) -> &mut Vec<Limb> {
        &mut self.limbs
    }

    pub(super) fn len_mut(&mut self) -> &mut usize {
        &mut self.len
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

    fn repr_kind() -> ReprKind {
        ReprKind::Owned
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
        self.limbs.to_mut()
    }
}
