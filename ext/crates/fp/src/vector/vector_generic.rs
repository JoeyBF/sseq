//! This module is provides wrappers around the contents of [`crate::vector::inner`]. The main
//! purpose is to put [`FpVectorP`] for different `p` into a single enum. It does the same for the
//! various slice structs.
//!
//! The main magic occurs in the macro `dispatch_vector_inner`, which we use to provide wrapper
//! functions around the `FpVectorP` functions.
//!
//! This module is only used when the `odd-primes` feature is enabled.

use std::convert::TryInto;
use std::io::{Read, Write};
use std::mem::size_of;

use itertools::Itertools;
#[cfg(feature = "json")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::limb::{entries_per_limb, Limb};
use crate::prime::ValidPrime;
use crate::vector::inner::FpVectorP;

use super::{
    inner::FpVectorBaseP,
    iter::{FpVectorIterator, FpVectorNonZeroIteratorP},
    repr::{OwnedRepr, Repr, ReprMut, ViewMutRepr, ViewRepr},
};

macro_rules! dispatch_vector_inner {
    // other is a type, but marking it as a :ty instead of :tt means we cannot use it to access its
    // enum variants.
    ($vis:vis fn $method:ident(&self, other: &$other:tt $(, $arg:ident: $ty:ty )* ) $(-> $ret:ty)?) => {
        $vis fn $method(&self, other: &$other, $($arg: $ty),* ) $(-> $ret)* {
            match (self, other) {
                (Self::_2(x), $other::_2(y)) => x.$method(y, $($arg),*),
                (Self::_3(x), $other::_3(y)) => x.$method(y, $($arg),*),
                (Self::_5(x), $other::_5(y)) => x.$method(y, $($arg),*),
                (Self::_7(x), $other::_7(y)) => x.$method(y, $($arg),*),
                (l, r) => {
                    panic!("Applying {} to vectors over different primes ({} and {})", stringify!($method), l.prime(), r.prime());
                }
            }
        }
    };
    ($vis:vis fn $method:ident(&mut self, other: &$other:tt $(, $arg:ident: $ty:ty )* ) $(-> $ret:ty)?) => {
        #[allow(unused_parens)]
        $vis fn $method(&mut self, other: &$other, $($arg: $ty),* ) $(-> $ret)* {
            match (self, other) {
                (Self::_2(x), $other::_2(y)) => x.$method(y, $($arg),*),
                (Self::_3(x), $other::_3(y)) => x.$method(y, $($arg),*),
                (Self::_5(x), $other::_5(y)) => x.$method(y, $($arg),*),
                (Self::_7(x), $other::_7(y)) => x.$method(y, $($arg),*),
                (l, r) => {
                    panic!("Applying {} to vectors over different primes ({} and {})", stringify!($method), l.prime(), r.prime());
                }
            }
        }
    };
    ($vis:vis fn $method:ident(&mut self, other: $other:tt $(, $arg:ident: $ty:ty )* ) $(-> $ret:ty)?) => {
        $vis fn $method(&mut self, other: $other, $($arg: $ty),* ) $(-> $ret)* {
            match (self, other) {
                (Self::_2(x), $other::_2(y)) => x.$method(y, $($arg),*),
                (Self::_3(x), $other::_3(y)) => x.$method(y, $($arg),*),
                (Self::_5(x), $other::_5(y)) => x.$method(y, $($arg),*),
                (Self::_7(x), $other::_7(y)) => x.$method(y, $($arg),*),
                (l, r) => {
                    panic!("Applying {} to vectors over different primes ({} and {})", stringify!($method), l.prime(), r.prime());
                }
            }
        }
    };
    ($vis:vis fn $method:ident(&mut self $(, $arg:ident: $ty:ty )* ) -> (dispatch $ret:tt)) => {
        #[must_use]
        $vis fn $method(&mut self, $($arg: $ty),* ) -> $ret {
            match self {
                Self::_2(x) => $ret::_2(x.$method($($arg),*)),
                Self::_3(x) => $ret::_3(x.$method($($arg),*)),
                Self::_5(x) => $ret::_5(x.$method($($arg),*)),
                Self::_7(x) => $ret::_7(x.$method($($arg),*)),
            }
        }
    };
    ($vis:vis fn $method:ident(&self $(, $arg:ident: $ty:ty )* ) -> (dispatch $ret:tt)) => {
        #[must_use]
        $vis fn $method(&self, $($arg: $ty),* ) -> $ret {
            match self {
                Self::_2(x) => $ret::_2(x.$method($($arg),*)),
                Self::_3(x) => $ret::_3(x.$method($($arg),*)),
                Self::_5(x) => $ret::_5(x.$method($($arg),*)),
                Self::_7(x) => $ret::_7(x.$method($($arg),*)),
            }
        }
    };
    ($vis:vis fn $method:ident(self $(, $arg:ident: $ty:ty )* ) -> (dispatch $ret:tt)) => {
        #[must_use]
        $vis fn $method(self, $($arg: $ty),* ) -> $ret {
            match self {
                Self::_2(x) => $ret::_2(x.$method($($arg),*)),
                Self::_3(x) => $ret::_3(x.$method($($arg),*)),
                Self::_5(x) => $ret::_5(x.$method($($arg),*)),
                Self::_7(x) => $ret::_7(x.$method($($arg),*)),
            }
        }
    };

    ($vis:vis fn $method:ident(self $(, $arg:ident: $ty:ty )* ) -> (dispatch $ret:tt $lifetime:tt)) => {
        #[must_use]
        $vis fn $method(self, $($arg: $ty),* ) -> $ret<$lifetime> {
            match self {
                Self::_2(x) => $ret::_2(x.$method($($arg),*)),
                Self::_3(x) => $ret::_3(x.$method($($arg),*)),
                Self::_5(x) => $ret::_5(x.$method($($arg),*)),
                Self::_7(x) => $ret::_7(x.$method($($arg),*)),
            }
        }
    };

    ($vis:vis fn $method:ident(&mut self $(, $arg:ident: $ty:ty )* ) $(-> $ret:ty)?) => {
        #[allow(unused_parens)]
        $vis fn $method(&mut self, $($arg: $ty),* ) $(-> $ret)* {
            match self {
                Self::_2(x) => x.$method($($arg),*),
                Self::_3(x) => x.$method($($arg),*),
                Self::_5(x) => x.$method($($arg),*),
                Self::_7(x) => x.$method($($arg),*),
            }
        }
    };
    ($vis:vis fn $method:ident(&self $(, $arg:ident: $ty:ty )* ) $(-> $ret:ty)?) => {
        #[allow(unused_parens)]
        $vis fn $method(&self, $($arg: $ty),* ) $(-> $ret)* {
            match self {
                Self::_2(x) => x.$method($($arg),*),
                Self::_3(x) => x.$method($($arg),*),
                Self::_5(x) => x.$method($($arg),*),
                Self::_7(x) => x.$method($($arg),*),
            }
        }
    };
    ($vis:vis fn $method:ident(self $(, $arg:ident: $ty:ty )* ) $(-> $ret:ty)?) => {
        #[allow(unused_parens)]
        $vis fn $method(self, $($arg: $ty),* ) $(-> $ret)* {
            match self {
                Self::_2(x) => x.$method($($arg),*),
                Self::_3(x) => x.$method($($arg),*),
                Self::_5(x) => x.$method($($arg),*),
                Self::_7(x) => x.$method($($arg),*),
            }
        }
    };
    ($vis:vis fn $method:ident<$gen:ident : $bound:ident>(&mut self, other: $other:ty $(, $arg:ident: $ty:ty )* ) $(-> $ret:ty)?) => {
        $vis fn $method<$gen: $bound>(&mut self, other: $other, $($arg: $ty),* ) $(-> $ret)* {
            let other = other.into();
            match (self, other) {
                (Self::_2(x), FpVectorBase::<$gen>::_2(y)) => x.$method(y, $($arg),*),
                (Self::_3(x), FpVectorBase::<$gen>::_3(y)) => x.$method(y, $($arg),*),
                (Self::_5(x), FpVectorBase::<$gen>::_5(y)) => x.$method(y, $($arg),*),
                (Self::_7(x), FpVectorBase::<$gen>::_7(y)) => x.$method(y, $($arg),*),
                (l, r) => {
                    panic!("Applying {} to vectors over different primes ({} and {})", stringify!($method), l.prime(), r.prime());
                }
            }
        }
    };
}

macro_rules! dispatch_vector {
    () => {};
    ($vis:vis fn $method:ident $(<$gen:ident : $bound:ident>)? ($($tt:tt)*) $(-> $ret:tt)?; $($tail:tt)*) => {
        dispatch_vector_inner! {
            $vis fn $method $(<$gen : $bound>)? ($($tt)*) $(-> $ret)*
        }
        dispatch_vector!{$($tail)*}
    }
}

macro_rules! match_p {
    ($p:ident, $($val:tt)*) => {
        match *$p {
            2 => Self::_2($($val)*),
            3 => Self::_3($($val)*),
            5 => Self::_5($($val)*),
            7 => Self::_7($($val)*),
            _ => panic!("Prime not supported: {}", *$p)
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub enum FpVectorBase<R: Repr> {
    _2(FpVectorBaseP<R, 2>),
    _3(FpVectorBaseP<R, 3>),
    _5(FpVectorBaseP<R, 5>),
    _7(FpVectorBaseP<R, 7>),
}

pub type FpVector = FpVectorBase<OwnedRepr>;
pub type Slice<'a> = FpVectorBase<ViewRepr<'a>>;
pub type SliceMut<'a> = FpVectorBase<ViewMutRepr<'a>>;

pub enum FpVectorNonZeroIterator<'a> {
    _2(FpVectorNonZeroIteratorP<'a, 2>),
    _3(FpVectorNonZeroIteratorP<'a, 3>),
    _5(FpVectorNonZeroIteratorP<'a, 5>),
    _7(FpVectorNonZeroIteratorP<'a, 7>),
}

impl<R: Repr> FpVectorBase<R> {
    dispatch_vector! {
        pub fn prime(&self) -> ValidPrime;
        pub fn len(&self) -> usize;
        pub fn is_empty(&self) -> bool;
        pub fn entry(&self, index: usize) -> u32;
        pub fn as_slice(&self) -> (dispatch Slice);
        pub fn is_zero(&self) -> bool;
        pub fn iter(&self) -> FpVectorIterator;
        pub fn iter_nonzero(&self) -> (dispatch FpVectorNonZeroIterator);
        pub fn to_owned(self) -> (dispatch FpVector);

        pub(crate) fn limbs(&self) -> (&[Limb]);
    }
}

impl<R: ReprMut> FpVectorBase<R> {
    dispatch_vector! {
        pub fn scale(&mut self, c: u32);
        pub fn set_to_zero(&mut self);
        pub fn add<R2: Repr>(&mut self, other: impl Into<FpVectorBase<R2>>, c: u32);
        pub fn add_offset<R2: Repr>(&mut self, other: impl Into<FpVectorBase<R2>>, c: u32, offset: usize);
        pub fn assign<R2: Repr>(&mut self, other: impl Into<FpVectorBase<R2>>);
        pub fn set_entry(&mut self, index: usize, value: u32);
        pub fn slice_mut(&mut self, start: usize, end: usize) -> (dispatch SliceMut);
        pub fn as_slice_mut(&mut self) -> (dispatch SliceMut);
        pub fn add_basis_element(&mut self, index: usize, value: u32);
        pub fn add_masked<R2: Repr>(&mut self, other: impl Into<FpVectorBase<R2>>, c: u32, mask: &[usize]);
        pub fn add_unmasked<R2: Repr>(&mut self, other: impl Into<FpVectorBase<R2>>, c: u32, mask: &[usize]);

        pub(crate) fn limbs_mut(&mut self) -> (&mut [Limb]);
    }
}

impl FpVector {
    pub fn new(p: ValidPrime, len: usize) -> FpVector {
        match_p!(p, FpVectorP::new_(len))
    }

    pub fn new_with_capacity(p: ValidPrime, len: usize, capacity: usize) -> FpVector {
        match_p!(p, FpVectorP::new_with_capacity_(len, capacity))
    }

    pub fn from_slice(p: ValidPrime, slice: &[u32]) -> Self {
        match_p!(p, FpVectorP::from(&slice))
    }

    pub fn num_limbs(p: ValidPrime, len: usize) -> usize {
        let entries_per_limb = entries_per_limb(p);
        (len + entries_per_limb - 1) / entries_per_limb
    }
    pub(crate) fn padded_len(p: ValidPrime, len: usize) -> usize {
        Self::num_limbs(p, len) * entries_per_limb(p)
    }

    pub fn update_from_bytes(&mut self, data: &mut impl Read) -> std::io::Result<()> {
        let limbs = self.limbs_mut();

        if cfg!(target_endian = "little") {
            let num_bytes = std::mem::size_of_val(limbs);
            unsafe {
                let buf: &mut [u8] =
                    std::slice::from_raw_parts_mut(limbs.as_mut_ptr() as *mut u8, num_bytes);
                data.read_exact(buf).unwrap();
            }
        } else {
            for entry in limbs {
                let mut bytes: [u8; size_of::<Limb>()] = [0; size_of::<Limb>()];
                data.read_exact(&mut bytes)?;
                *entry = Limb::from_le_bytes(bytes);
            }
        };
        Ok(())
    }

    pub fn from_bytes(p: ValidPrime, len: usize, data: &mut impl Read) -> std::io::Result<Self> {
        let mut v = Self::new(p, len);
        v.update_from_bytes(data)?;
        Ok(v)
    }

    pub fn to_bytes(&self, buffer: &mut impl Write) -> std::io::Result<()> {
        // self.limbs is allowed to have more limbs than necessary, but we only save the
        // necessary ones.
        let num_limbs = Self::num_limbs(self.prime(), self.len());

        if cfg!(target_endian = "little") {
            let num_bytes = num_limbs * size_of::<Limb>();
            unsafe {
                let buf: &[u8] =
                    std::slice::from_raw_parts_mut(self.limbs().as_ptr() as *mut u8, num_bytes);
                buffer.write_all(buf)?;
            }
        } else {
            for limb in &self.limbs()[0..num_limbs] {
                let bytes = limb.to_le_bytes();
                buffer.write_all(&bytes)?;
            }
        }
        Ok(())
    }

    dispatch_vector! {
        pub fn assign_partial(&mut self, other: &Self);
        pub fn slice(&self, start: usize, end: usize) -> (dispatch Slice);
        pub fn extend_len(&mut self, dim: usize);
        pub fn set_scratch_vector_size(&mut self, dim: usize);
        pub fn copy_from_slice(&mut self, slice: &[u32]);
        pub(crate) fn trim_start(&mut self, n: usize);
        pub fn add_truncate(&mut self, other: &Self, c: u32) -> (Option<()>);
        pub fn sign_rule(&self, other: &Self) -> bool;
        pub fn add_carry(&mut self, other: &Self, c: u32, rest: &mut [FpVector]) -> bool;
        pub fn first_nonzero(&self) -> (Option<(usize, u32)>);
        pub fn density(&self) -> f32;
    }
}

impl<'a> Slice<'a> {
    dispatch_vector! {
        pub fn slice(self, start: usize, end: usize) -> (dispatch Slice 'a);
    }
}

impl<'a> SliceMut<'a> {
    dispatch_vector! {
        pub fn copy(&mut self) -> (dispatch SliceMut);
    }

    pub fn add_tensor(&mut self, offset: usize, coeff: u32, left: Slice, right: Slice) {
        match (self, left, right) {
            (SliceMut::_2(x), Slice::_2(y), Slice::_2(z)) => x.add_tensor(offset, coeff, y, z),
            (SliceMut::_3(x), Slice::_3(y), Slice::_3(z)) => x.add_tensor(offset, coeff, y, z),
            (SliceMut::_5(x), Slice::_5(y), Slice::_5(z)) => x.add_tensor(offset, coeff, y, z),
            (SliceMut::_7(x), Slice::_7(y), Slice::_7(z)) => x.add_tensor(offset, coeff, y, z),
            _ => {
                panic!("Applying add_tensor to vectors over different primes");
            }
        }
    }
}

impl<'a> FpVectorNonZeroIterator<'a> {
    dispatch_vector! {
        fn next(&mut self) -> (Option<(usize, u32)>);
    }
}

impl<R: Repr> std::fmt::Display for FpVectorBase<R> {
    /// # Example
    /// ```
    /// # use fp::vector::FpVector;
    /// # use fp::prime::ValidPrime;
    /// let v = FpVector::from_slice(ValidPrime::new(2), &[0, 1, 0]);
    /// assert_eq!(&format!("{v}"), "[0, 1, 0]");
    /// assert_eq!(&format!("{v:#}"), "010");
    /// ```
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if f.alternate() {
            for v in self.iter() {
                write!(f, "{v}")?;
            }
            Ok(())
        } else {
            write!(f, "[{}]", self.iter().format(", "))
        }
    }
}

impl<R1: Repr, R2: ReprMut> std::ops::AddAssign<&FpVectorBase<R1>> for FpVectorBase<R2> {
    fn add_assign(&mut self, other: &FpVectorBase<R1>) {
        self.add(other, 1);
    }
}

impl<R: Repr> From<&FpVectorBase<R>> for Vec<u32> {
    fn from(v: &FpVectorBase<R>) -> Vec<u32> {
        v.iter().collect()
    }
}

impl<'a> Iterator for FpVectorNonZeroIterator<'a> {
    type Item = (usize, u32);

    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

impl<'a> IntoIterator for &'a FpVector {
    type IntoIter = FpVectorIterator<'a>;
    type Item = u32;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

macro_rules! impl_try_into {
    ($var:tt, $p:literal) => {
        impl<'a> TryInto<&'a mut FpVectorP<$p>> for &'a mut FpVector {
            type Error = ();

            fn try_into(self) -> Result<&'a mut FpVectorP<$p>, ()> {
                match self {
                    FpVector::$var(x) => Ok(x),
                    _ => Err(()),
                }
            }
        }
    };
}

impl_try_into!(_2, 2);
impl_try_into!(_3, 3);
impl_try_into!(_5, 5);
impl_try_into!(_7, 7);

#[cfg(feature = "json")]
impl Serialize for FpVector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Vec::<u32>::from(self).serialize(serializer)
    }
}

#[cfg(feature = "json")]
impl<'de> Deserialize<'de> for FpVector {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        panic!("Deserializing FpVector not supported");
        // This is needed for ext-websocket/actions to be happy
    }
}

impl<'a, R: Repr> From<&'a FpVectorBase<R>> for Slice<'a> {
    fn from(value: &'a FpVectorBase<R>) -> Self {
        value.as_slice()
    }
}

impl<'a, R: ReprMut> From<&'a mut FpVectorBase<R>> for SliceMut<'a> {
    fn from(value: &'a mut FpVectorBase<R>) -> Self {
        value.slice_mut(0, value.len())
    }
}
