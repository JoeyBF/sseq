// This generates better llvm optimization
#![allow(clippy::int_plus_one)]

use crate::{field::Field, limb::Limb};

/// A vector over a finite field.
///
/// Interally, it packs entries of the vectors into limbs. However, this is an abstraction that must
/// not leave the `fp` library.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct FqVector<F: Field> {
    pub(super) fq: F,
    pub(super) len: usize,
    pub(super) limbs: Vec<Limb>,
}

/// A slice of an `FqVector`.
///
/// This immutably borrows the vector and implements `Copy`.
#[derive(Debug, Copy, Clone)]
pub struct FqSlice<'a, F: Field> {
    pub(super) fq: F,
    pub(super) limbs: &'a [Limb],
    pub(super) start: usize,
    pub(super) end: usize,
}

/// A mutable slice of an `FqVector`.
///
/// This mutably borrows the vector. Since it is a mutable borrow, it cannot implement `Copy`.
/// However, it has a [`FqSliceMut::copy`] function that imitates the reborrowing, that mutably
/// borrows `FqSliceMut` and returns a `FqSliceMut` with a shorter lifetime.
#[derive(Debug)]
pub struct FqSliceMut<'a, F: Field> {
    pub(super) fq: F,
    pub(super) limbs: &'a mut [Limb],
    pub(super) start: usize,
    pub(super) end: usize,
}

// See impl_* for implementations
