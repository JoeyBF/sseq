//! This module replaces `vector` when `odd-primes` is disabled. Instead of producing enum
//! wrappers, it simply rexports `FooP<2>` as `Foo`.

use std::{
    io::{Read, Write},
    mem::size_of,
};

use itertools::Itertools;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::iter::{FpVectorIterator, FpVectorNonZeroIteratorP};
use crate::{
    limb::{entries_per_limb, Limb},
    prime::{Prime, ValidPrime, P2},
    vector::inner::{FpVectorP, SliceMutP, SliceP},
};

pub type FpVector = FpVectorP<P2>;
pub type Slice<'a> = SliceP<'a, P2>;
pub type SliceMut<'a> = SliceMutP<'a, P2>;
pub type FpVectorNonZeroIterator<'a> = FpVectorNonZeroIteratorP<'a, P2>;

// `FpVector` implementations

impl FpVector {
    pub fn from_slice<P: Prime>(p: P, slice: &[u32]) -> Self {
        if p == 2 {
            Self::from((P2, &slice))
        } else {
            panic!("Only p = 2 is supported")
        }
    }

    pub fn num_limbs(_p: ValidPrime, len: usize) -> usize {
        let entries_per_limb = entries_per_limb(P2);
        (len + entries_per_limb - 1) / entries_per_limb
    }

    #[allow(dead_code)]
    pub(crate) fn padded_len(p: ValidPrime, len: usize) -> usize {
        Self::num_limbs(p, len) * entries_per_limb(P2)
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
}

impl std::fmt::Display for FpVector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl std::ops::AddAssign<&Self> for FpVector {
    fn add_assign(&mut self, other: &Self) {
        self.add(other, 1);
    }
}

impl<'a> IntoIterator for &'a FpVector {
    type IntoIter = FpVectorIterator<'a>;
    type Item = u32;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Serialize for FpVector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Vec::<u32>::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FpVector {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        panic!("Deserializing FpVector not supported");
        // This is needed for ext-websocket/actions to be happy
    }
}

// `Slice` implementations

impl<'a> std::fmt::Display for Slice<'a> {
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

// Type conversions

impl<'a, 'b> From<&'a mut SliceMut<'b>> for SliceMut<'a> {
    fn from(slice: &'a mut SliceMut<'b>) -> Self {
        slice.copy()
    }
}

impl<'a, 'b> From<&'a Slice<'b>> for Slice<'a> {
    fn from(slice: &'a Slice<'b>) -> Self {
        *slice
    }
}

impl<'a, 'b> From<&'a SliceMut<'b>> for Slice<'a> {
    fn from(slice: &'a SliceMut<'b>) -> Self {
        slice.as_slice()
    }
}
