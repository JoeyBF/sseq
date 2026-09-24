//! Tuning knobs shared by the Rust host code and the CUDA kernel.
//!
//! Follows the convention PR 298 established for fp-cuda: the constants live here, reach NVRTC as
//! `-D` options, and the kernel defines none of them itself — it `#error`s on a missing one, so a
//! forgotten define is a compile error rather than a silently wrong default.
//!
//! Every value below is carried over from the cubecl kernel in `milnor_gpu`, where each was
//! measured individually. They are repeated rather than imported because `milnor_gpu` is behind the
//! `gpu` feature and will be deleted; the numbers, and the reasons for them, must outlive it.

/// Matrices per thread tile.
///
/// A thread covers `MATRIX_GROUP` x `TERM_GROUP` pairs. `col_sums`/`masks` depend only on the
/// matrix and a term's p-part only on the term, so an MxT tile reads `2M + T` values per column to
/// evaluate `M*T` pairs — 1.17 loads per pair at 2x3, against 1.67 at 1x3 and 3 at 1x1. The kernel
/// is issue-limited on integer work, so fewer loads and fewer addresses is the lever.
pub const MATRIX_GROUP: usize = 2;

/// Terms per thread tile.
///
/// NOT symmetric with [`MATRIX_GROUP`]: terms are few (`nt ~ 5`), so the ragged tail dominates and
/// 4 loses to 3 purely on wasted lanes. Matrices are many (`~20 000`), so a partial matrix tile
/// costs a few idle lanes out of thousands.
pub const TERM_GROUP: usize = 3;

/// Pairs per coarse-index bucket, as a power of two.
///
/// `prod_coarse[ci]` is the product owning pair `ci << COARSE_LOG`, bracketing the binary search
/// before it starts. Each search step is a DEPENDENT global load — a full latency stall — and
/// ablation put the unbracketed search at ~12% of kernel time. Two cheap loads replace ~15
/// dependent ones.
pub const COARSE_LOG: usize = 20;

/// Longest p-part the packed accumulator holds.
pub const PPART_MAX_LEN: usize = 10;

/// Column index below which every packed digit lies wholly under bit 32.
///
/// Splits the column loop so "is this digit inside the packed accumulator?" and "where does it
/// land?" are both answered at COMPILE time: below this the accumulate is 32-bit (half the shifts,
/// half the ORs), above it there is no accumulator test. In the cubecl kernel the merged form spent
/// 36 of 121 SASS instructions per column on exactly those two questions.
pub const COL_SPLIT_32: usize = 3;

/// Upper bound on a thread's column count, i.e. `max(cs_len, mk_len)`.
///
/// MUST NOT be hardcoded smaller: `mk_len = rows + cols - 1` grows with internal degree — 16
/// suffices to t~510, but a 9th xi appears at t>=511 making it 17, then 18 past 1023. A fixed 16
/// would silently truncate at stem 300: wrong answers, no error. The host asserts the per-launch
/// value fits this cap.
pub const WORKING_CAP: usize = 32;

/// Threads per block.
pub const THREADS: usize = 256;

/// Bit position of each packed p-part digit, from `PPart`'s own layout.
///
/// The accumulator the kernel assembles a p-part into is ONE `u64`, not an array, and these are
/// where each digit lives in it. Read from `PPart::shift` rather than restated, so a change to the
/// packing cannot leave the kernel silently reading the wrong bits.
pub fn pp_shifts() -> Vec<u32> {
    (0..PPART_MAX_LEN)
        .map(|i| crate::algebra::milnor_algebra::PPart::shift(i))
        .collect()
}

/// Field mask of each packed digit.
pub fn pp_masks() -> Vec<u32> {
    (0..PPART_MAX_LEN)
        .map(|i| {
            let w = crate::algebra::milnor_algebra::PPart::width(i);
            ((1u64 << w) - 1) as u32
        })
        .collect()
}

/// Render a list as a C brace initializer, so a table can travel as a `-D` option.
fn brace(values: &[u32]) -> String {
    let body: Vec<String> = values.iter().map(u32::to_string).collect();
    format!("{{{}}}", body.join(","))
}

/// The knobs as NVRTC `-D` options.
///
/// Per-launch values (`num_segs`, `sq_len`, `cols`, `cs_transposed` in the cubecl kernel) are NOT
/// here: they vary per launch, and specialising on a value that changes forces a recompile per
/// distinct value. The cubecl kernel keeps `search_iters` a runtime scalar for exactly that reason
/// — making it comptime widened a benchmark spread from 0.7% to 5.6%.
pub fn defines() -> Vec<(&'static str, String)> {
    [
        ("MATRIX_GROUP", MATRIX_GROUP),
        ("TERM_GROUP", TERM_GROUP),
        ("COARSE_LOG", COARSE_LOG),
        ("PPART_MAX_LEN", PPART_MAX_LEN),
        ("COL_SPLIT_32", COL_SPLIT_32),
        ("WORKING_CAP", WORKING_CAP),
        ("THREADS", THREADS),
    ]
    .iter()
    .map(|(name, value)| (*name, value.to_string()))
    .chain([
        ("PP_SHIFTS", brace(&pp_shifts())),
        ("PP_MASKS", brace(&pp_masks())),
    ])
    .collect()
}

/// Rows of the enumeration's per-thread matrix: `rows = |p_part| <= MAX_XI_TAU`.
pub const ENUM_ROW_CAP: usize = fp::MAX_MULTINOMIAL_LEN;

/// Columns of the enumeration's per-thread matrix.
///
/// `cols` is the widest BIT-LENGTH of any p-part entry, so its true bound is `PPart::width(0)` --
/// the field holding `r_1`, the widest -- and NOT [`WORKING_CAP`], which sizes an unrelated array
/// (the multiply kernel's assembled p-part) and is nearly 3x larger. That conflation once made
/// `matrix`, the hottest per-thread array, 320 `u32` instead of 110. It is CUDA LOCAL memory --
/// dynamically indexed, so it cannot be register-allocated and every access is a real off-chip
/// load. Deriving the cap from the width table keeps it correct if `PPart`'s layout ever changes.
pub const ENUM_COL_CAP: usize = crate::algebra::milnor_algebra::PPart::width(0) as usize;

/// `matrix` is `rows * cols`, `col_sums` is `cols - 1`, `masks` is `rows + cols - 1`.
pub const ENUM_MATRIX_CAP: usize = ENUM_ROW_CAP * ENUM_COL_CAP;
pub const ENUM_MASK_CAP: usize = ENUM_ROW_CAP + ENUM_COL_CAP;

/// Threads per block for the enumeration kernel.
///
/// Small on purpose, and it costs nothing. The kernel is one thread per `R`, and a production
/// launch carries ~1293 `R`s on average -- about 6 blocks against an H200's 3168 block slots, i.e.
/// `Waves Per SM = 0.002`. The SMs are empty either way, which a block-size sweep confirmed from
/// the other side: 256/64/32 threads measured 561/569/542 s, flat. Grid width does not set this
/// kernel's time; its LONGEST SINGLE `R` does, because one thread walks that `R`'s odometer
/// sequentially.
pub const ENUM_THREADS: usize = 32;

/// The enumeration knobs as NVRTC `-D` options.
pub fn enum_defines() -> Vec<(&'static str, String)> {
    [
        ("ENUM_ROW_CAP", ENUM_ROW_CAP),
        ("ENUM_COL_CAP", ENUM_COL_CAP),
        ("ENUM_MATRIX_CAP", ENUM_MATRIX_CAP),
        ("ENUM_MASK_CAP", ENUM_MASK_CAP),
        ("ENUM_THREADS", ENUM_THREADS),
    ]
    .iter()
    .map(|(name, value)| (*name, value.to_string()))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed accumulator must actually hold what the column split assumes.
    #[test]
    fn column_split_is_inside_the_accumulator() {
        assert!(
            COL_SPLIT_32 < PPART_MAX_LEN,
            "COL_SPLIT_32 must fall inside the packed accumulator"
        );
        assert!(
            PPART_MAX_LEN <= WORKING_CAP,
            "the accumulator cannot be longer than a thread's column bound"
        );
    }

    /// [`COL_SPLIT_32`] must be exactly where `PPart`'s packing crosses bit 32.
    ///
    /// The kernel accumulates the columns below it in `u32`, which is sound only if every digit
    /// there ends below bit 32, and worth doing only if the next digit does not. Both halves are
    /// read off `PPart::shift` rather than trusted to a comment, so a change to the packing fails
    /// here instead of silently dropping the high half of a digit on the device.
    #[test]
    fn column_split_is_exactly_at_bit_32() {
        use crate::algebra::milnor_algebra::PPart;
        assert!(
            PPart::shift(COL_SPLIT_32) <= 32,
            "digit {COL_SPLIT_32} starts at bit {}, so a 32-bit accumulate would drop real bits",
            PPart::shift(COL_SPLIT_32),
        );
        assert!(
            PPart::shift(COL_SPLIT_32 + 1) > 32,
            "the split is short: digit {COL_SPLIT_32} also fits below bit 32",
        );
    }

    /// Every knob the kernel needs is actually emitted, or NVRTC would `#error`.
    #[test]
    fn defines_cover_every_knob() {
        let names: Vec<&str> = defines().into_iter().map(|(n, _)| n).collect();
        for want in [
            "MATRIX_GROUP",
            "TERM_GROUP",
            "COARSE_LOG",
            "PPART_MAX_LEN",
            "COL_SPLIT_32",
            "WORKING_CAP",
            "THREADS",
        ] {
            assert!(
                names.contains(&want),
                "{want} is not emitted as a -D option"
            );
        }
    }
}
