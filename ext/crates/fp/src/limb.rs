pub(crate) use crate::constants::{BITS_PER_LIMB, Limb};

/// A struct containing the information required to access a specific entry in an array of `Limb`s.
#[derive(Debug, Copy, Clone)]
pub(crate) struct LimbBitIndexPair {
    pub(crate) limb: usize,
    pub(crate) bit_index: usize,
}

/// Read an array of `Limb`s.
pub(crate) fn from_bytes(limbs: &mut [Limb], data: &mut impl std::io::Read) -> std::io::Result<()> {
    if cfg!(target_endian = "little") {
        let num_bytes = std::mem::size_of_val(limbs);
        let buf: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(limbs.as_mut_ptr() as *mut u8, num_bytes) };
        data.read_exact(buf)
    } else {
        for entry in limbs {
            let mut bytes: [u8; size_of::<Limb>()] = [0; size_of::<Limb>()];
            data.read_exact(&mut bytes)?;
            *entry = Limb::from_le_bytes(bytes);
        }
        Ok(())
    }
}

/// Store an array of `Limb`s.
pub(crate) fn to_bytes(limbs: &[Limb], data: &mut impl std::io::Write) -> std::io::Result<()> {
    if cfg!(target_endian = "little") {
        let num_bytes = std::mem::size_of_val(limbs);
        let buf: &[u8] =
            unsafe { std::slice::from_raw_parts(limbs.as_ptr() as *const u8, num_bytes) };
        data.write_all(buf)
    } else {
        for limb in limbs {
            let bytes = limb.to_le_bytes();
            data.write_all(&bytes)?;
        }
        Ok(())
    }
}

pub(crate) fn sign_rule(mut target: Limb, mut source: Limb) -> u32 {
    let mut result = 0;
    let mut n = 1;
    // Empirically, the compiler unrolls this loop because BITS_PER_LIMB is a constant.
    while 2 * n < BITS_PER_LIMB {
        // This is 1 every 2n bits.
        let mask: Limb = !0 / ((1 << (2 * n)) - 1);
        result ^= (mask & (source >> n) & target).count_ones() % 2;
        source = source ^ (source >> n);
        target = target ^ (target >> n);
        n *= 2;
    }
    result ^= (1 & (source >> (BITS_PER_LIMB / 2)) & target) as u32;
    result
}

/// Transpose a [`BITS_PER_LIMB`]-square bit matrix in place.
///
/// On return, bit `j` of `block[i]` holds what was bit `i` of `block[j]`.
///
/// The implementation is the recursive delta swap of Hacker's Delight 7-3: each round exchanges
/// two off-diagonal quadrants of every sub-block at the current scale, so the whole transpose costs
/// `log2(BITS_PER_LIMB)` masked passes over the block rather than the `BITS_PER_LIMB^2` single-bit
/// extractions the entry-at-a-time form performs.
pub(crate) fn transpose_square_block(block: &mut [Limb; BITS_PER_LIMB]) {
    let mut s = BITS_PER_LIMB / 2;
    // Selects the columns whose index has bit `s` clear, i.e. the left half of each 2s-wide group.
    let mut m: Limb = !0 >> (BITS_PER_LIMB / 2);
    while s != 0 {
        let mut k = 0;
        while k < BITS_PER_LIMB {
            // `k` never has bit `s` set, so `k` and `k | s` are the upper and lower row halves of
            // one 2s-square block; this exchanges its upper-right and lower-left quadrants.
            let t = ((block[k] >> s) ^ block[k | s]) & m;
            block[k | s] ^= t;
            block[k] ^= t << s;
            k = (k + s + 1) & !s;
        }
        s >>= 1;
        m ^= m << s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The entry-at-a-time transpose, as an oracle for [`transpose_square_block`].
    fn naive_transpose(block: &[Limb; BITS_PER_LIMB]) -> [Limb; BITS_PER_LIMB] {
        let mut out = [0; BITS_PER_LIMB];
        for (i, &row) in block.iter().enumerate() {
            for (j, slot) in out.iter_mut().enumerate() {
                *slot |= ((row >> j) & 1) << i;
            }
        }
        out
    }

    #[test]
    fn transpose_block_matches_naive() {
        // A xorshift keeps the test deterministic without pulling `rand` into a unit test.
        let mut state: u64 = 0x243f_6a88_85a3_08d3;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..64 {
            let mut block = [0; BITS_PER_LIMB];
            for entry in &mut block {
                *entry = next();
            }
            let expected = naive_transpose(&block);
            transpose_square_block(&mut block);
            assert_eq!(block, expected);
        }
    }

    #[test]
    fn transpose_block_is_an_involution() {
        let mut state: u64 = 0x1319_8a2e_0370_7344;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut block = [0; BITS_PER_LIMB];
        for entry in &mut block {
            *entry = next();
        }
        let original = block;
        transpose_square_block(&mut block);
        transpose_square_block(&mut block);
        assert_eq!(block, original);
    }

    #[test]
    fn transpose_block_sends_single_bit_to_its_mirror() {
        for i in [0, 1, 17, 62, 63] {
            for j in [0, 5, 31, 63] {
                let mut block = [0; BITS_PER_LIMB];
                block[i] = 1 << j;
                transpose_square_block(&mut block);
                let mut expected = [0; BITS_PER_LIMB];
                expected[j] = 1 << i;
                assert_eq!(block, expected, "bit ({i}, {j})");
            }
        }
    }
}
