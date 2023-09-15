use itertools::Itertools;

use crate::limb::{self, Limb};

use super::{
    inner::{FpVectorP, SliceP},
    repr::OwnedRepr,
};

impl<const P: u32> FpVectorP<P> {
    pub fn new_(len: usize) -> Self {
        let number_of_limbs = limb::number::<P>(len);
        let data = OwnedRepr::new(vec![0; number_of_limbs], len);
        Self { data }
    }

    pub(super) fn from_data(data: OwnedRepr) -> Self {
        Self { data }
    }

    pub fn from_raw_parts(len: usize, limbs: Vec<Limb>) -> Self {
        debug_assert_eq!(limbs.len(), limb::number::<P>(len));
        let data = OwnedRepr::new(limbs, len);
        Self { data }
    }

    pub fn new_with_capacity_(len: usize, capacity: usize) -> Self {
        let mut limbs = Vec::with_capacity(limb::number::<P>(capacity));
        limbs.resize(limb::number::<P>(len), 0);
        let data = OwnedRepr::new(limbs, len);
        Self { data }
    }

    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> SliceP<'_, P> {
        assert!(start <= end && end <= self.len());
        SliceP::new(self.limbs(), start, end)
    }

    /// A version of [`FpVectorP::assign`] that allows `other` to be shorter than `self`.
    pub fn assign_partial(&mut self, other: &Self) {
        debug_assert!(other.len() <= self.len());
        self.limbs_mut()[0..other.limbs().len()].copy_from_slice(other.limbs());
        for limb in self.limbs_mut()[other.limbs().len()..].iter_mut() {
            *limb = 0;
        }
    }

    /// This function ensures the length of the vector is at least `len`. See also
    /// `set_scratch_vector_size`.
    pub fn extend_len(&mut self, len: usize) {
        self.data.extend_len::<P>(len)
    }

    /// This clears the vector and sets the length to `len`. This is useful for reusing
    /// allocations of temporary vectors.
    pub fn set_scratch_vector_size(&mut self, len: usize) {
        self.data.set_size::<P>(len)
    }

    /// This replaces the contents of the vector with the contents of the slice. The two must have
    /// the same length.
    pub fn copy_from_slice(&mut self, slice: &[u32]) {
        self.data.copy_from_slice::<P>(slice)
    }

    /// Permanently remove the first `n` elements in the vector. `n` must be a multiple of
    /// the number of entries per limb
    pub(crate) fn trim_start(&mut self, n: usize) {
        self.data.trim_start::<P>(n)
    }

    pub fn sign_rule(&self, other: &Self) -> bool {
        assert_eq!(P, 2);
        let mut result = 0;
        for target_limb_idx in 0..self.limbs().len() {
            let target_limb = other.limbs()[target_limb_idx];
            let source_limb = self.limbs()[target_limb_idx];
            result ^= limb::sign_rule(target_limb, source_limb);
            if target_limb.count_ones() % 2 == 0 {
                continue;
            }
            for _ in 0..target_limb_idx {
                result ^= source_limb.count_ones() % 2;
            }
        }
        result == 1
    }

    pub fn add_truncate(&mut self, other: &Self, c: u32) -> Option<()> {
        for (left, right) in self.limbs_mut().iter_mut().zip_eq(other.limbs()) {
            *left = limb::add::<P>(*left, *right, c);
            *left = limb::truncate::<P>(*left)?;
        }
        Some(())
    }

    fn add_carry_limb<T>(&mut self, idx: usize, source: Limb, c: u32, rest: &mut [T]) -> bool
    where
        for<'a> &'a mut T: TryInto<&'a mut Self>,
    {
        if P == 2 {
            if c == 0 {
                return false;
            }
            let mut cur_vec = self;
            let mut carry = source;
            for carry_vec in rest.iter_mut() {
                let carry_vec = carry_vec
                    .try_into()
                    .ok()
                    .expect("rest vectors in add_carry must be of the same prime");
                let rem = cur_vec.limbs()[idx] ^ carry;
                let quot = cur_vec.limbs()[idx] & carry;
                cur_vec.limbs_mut()[idx] = rem;
                carry = quot;
                cur_vec = carry_vec;
                if quot == 0 {
                    return false;
                }
            }
            cur_vec.limbs_mut()[idx] ^= carry;
            true
        } else {
            unimplemented!()
        }
    }

    pub fn add_carry<T>(&mut self, other: &Self, c: u32, rest: &mut [T]) -> bool
    where
        for<'a> &'a mut T: TryInto<&'a mut Self>,
    {
        let mut result = false;
        for i in 0..self.limbs().len() {
            result |= self.add_carry_limb(i, other.limbs()[i], c, rest);
        }
        result
    }

    /// Find the index and value of the first non-zero entry of the vector. `None` if the vector is zero.
    pub fn first_nonzero(&self) -> Option<(usize, u32)> {
        let entries_per_limb = limb::entries_per_limb_const::<P>();
        let bit_length = limb::bit_length_const::<P>();
        let bitmask = limb::bitmask::<P>();
        for (i, &limb) in self.limbs().iter().enumerate() {
            if limb == 0 {
                continue;
            }
            let index = limb.trailing_zeros() as usize / bit_length;
            return Some((
                i * entries_per_limb + index,
                ((limb >> (index * bit_length)) & bitmask) as u32,
            ));
        }
        None
    }

    pub fn density(&self) -> f32 {
        let num_nonzero = if P == 2 {
            self.limbs()
                .iter()
                .copied()
                .map(Limb::count_ones)
                .sum::<u32>() as usize
        } else {
            self.iter_nonzero().count()
        };
        num_nonzero as f32 / self.len() as f32
    }
}

impl<T: AsRef<[u32]>, const P: u32> From<&T> for FpVectorP<P> {
    fn from(slice: &T) -> Self {
        let mut v = Self::new_(slice.as_ref().len());
        v.copy_from_slice(slice.as_ref());
        v
    }
}

impl<const P: u32> From<&FpVectorP<P>> for Vec<u32> {
    fn from(vec: &FpVectorP<P>) -> Vec<u32> {
        vec.iter().collect()
    }
}
