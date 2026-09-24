// In-kernel admissible-matrix enumeration: one thread per distinct R, generating that R's
// col_sums/masks for EVERY admissible matrix directly into device scratch.
//
// This is the on-GPU replacement for uploading an enumerated master. It matters because the master
// does not fit: it reaches ~89 GB at the frontier against a 143 GB card that also has to hold
// everything else, and re-uploading it after an eviction costs more than recomputing it. A thread
// here walks the same odometer `AdmissibleMatrix::next` walks on the host, so what it writes is
// byte-identical to what `admissible_matrices` would have produced.
//
// AS WITH THE MULTIPLY, THIS IS THE SIMPLEST CORRECT FORM. The cubecl kernel this replaces also
// carries: a seed/split scheme that lets several threads share one R's walk, a transposed output
// layout, four emit modes for ablation, and per-thread state in shared memory at a
// bank-conflict-free stride. Each is re-addable with the output digest held fixed, and each is a
// chance to change the answer quietly. What is here is one thread per R, local state, one layout.
//
// A note on what is NOT worth optimising, measured on the cubecl version so the numbers carry over:
// this kernel is LATENCY-bound and STARVED, not bandwidth- or compute-bound (5.16% SM throughput,
// 0.65% DRAM, 98.83% L2 hit rate, "No Eligible" 76.57%). Registers are not the limiter either --
// 40/thread already allows 24 blocks/SM. The binding fact is that one thread per R gives too FEW
// threads and wildly UNEQUAL ones: a warp runs at its longest member and ~82% of lanes idle
// (5.88/32 active). A launch's duration is set by its longest single R, so widening the grid does
// nothing -- 6.5x more blocks bought 3.4%. The structural fixes are batching far more R's per
// launch, or replacing thread-per-R with lane cooperation over dynamically pulled work.

#ifndef ENUM_ROW_CAP
#error "ENUM_ROW_CAP must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef ENUM_COL_CAP
#error "ENUM_COL_CAP must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef ENUM_MATRIX_CAP
#error "ENUM_MATRIX_CAP must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef ENUM_MASK_CAP
#error "ENUM_MASK_CAP must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef ENUM_THREADS
#error "ENUM_THREADS must be provided as an NVRTC -D option (see params.rs)"
#endif

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

// Enumerate every admissible matrix of each R.
//
// Inputs are per R: `p_parts` (n_r x width, zero-padded), `r_rows`/`r_cols` (its dimensions), and
// `r_cs_out`/`r_mk_out` (its base offset, in u16 units, into the shared output scratch -- a host
// prefix sum of num_mats*cs_len / num_mats*mk_len). `out_counts[ri]` receives how many matrices the
// thread emitted, so a COUNT-ONLY pre-pass (`emit = 0`) can drive that prefix sum with no host
// enumeration at all: the counts that decide how much scratch to allocate come from the device.
//
// The p-part arrives as u32 and is written out as u16. That is not a narrowing risk here -- entries
// are bounded by the p-part field widths, far under 2^16 -- but it is the same u16 master the
// multiply kernel reads, so the two must agree on the type exactly.
extern "C" __global__ __launch_bounds__(ENUM_THREADS) void enumerate_admissible(
    const u32 *__restrict__ p_parts,
    const u32 *__restrict__ r_rows,
    const u32 *__restrict__ r_cols,
    // u64: the scratch offsets index buffers reaching billions of elements in a big batch, past
    // u32::MAX. These ARE the multiply kernel's r_cs_offset / r_mk_offset.
    const u64 *__restrict__ r_cs_out,
    const u64 *__restrict__ r_mk_out,
    u16 *out_cs,
    u16 *out_mk,
    u32 *out_counts,
    u32 width,
    u32 n_r,
    // 0 counts only, 1 also writes. A runtime flag rather than a compile-time one: specialising it
    // would double the module cache for a branch that is uniform across the launch.
    u32 emit) {
    u32 ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= n_r) return;

    u32 rows = r_rows[ti];
    u32 cols = r_cols[ti];
    u32 cs_len = cols - 1;
    u32 mk_len = rows + cols - 1;
    u64 pbase = (u64)ti * width;
    u64 cs_base = r_cs_out[ti];
    u64 mk_base = r_mk_out[ti];

    // Per-thread state, mirroring `AdmissibleMatrix`. CUDA local arrays are uninitialised, so every
    // slot a read can reach is explicitly zeroed. Only the region THIS R reaches is cleared, not
    // the whole cap: a typical R is far smaller (rows ~4-8 against 10, cols ~6-9), and these are
    // local-memory stores, not register writes.
    u32 matrix[ENUM_MATRIX_CAP];
    u32 totals[ENUM_ROW_CAP];
    u32 col_sums[ENUM_COL_CAP];
    u32 masks[ENUM_MASK_CAP];
    for (u32 i = 0; i < rows * cols; ++i) matrix[i] = 0;
    for (u32 i = 0; i < rows; ++i) totals[i] = 0;
    for (u32 i = 0; i < cs_len; ++i) col_sums[i] = 0;
    for (u32 i = 0; i < mk_len; ++i) masks[i] = 0;

    // Column 0 of the matrix -- and the initial masks -- is the padded p-part. That first state is
    // itself the first admissible matrix, which is why the loop below emits before it steps.
    for (u32 i = 0; i < rows; ++i) {
        u32 x = p_parts[pbase + i];
        matrix[i * cols] = x;
        masks[i] = x;
    }

    u32 mat = 0;
    bool more = true;
    while (more) {
        if (emit) {
            u64 co = cs_base + (u64)mat * cs_len;
            for (u32 j = 0; j < cs_len; ++j) out_cs[co + j] = (u16)col_sums[j];
            u64 mo = mk_base + (u64)mat * mk_len;
            for (u32 j = 0; j < mk_len; ++j) out_mk[mo + j] = (u16)masks[j];
        }
        ++mat;

        // One `next()` step. `found` is the reference's `return true`; `handled` is its `continue`.
        // The loops guard on `!found` rather than breaking, which keeps the control flow uniform.
        bool found = false;
        for (u32 row = 0; row < rows && !found; ++row) {
            u32 p_to_the_j = 1;
            // `totals[row]` is thread-private and touched several times per column. The step is a
            // dependent chain, so those round trips to local memory ARE the latency: keep it in a
            // register for the row scan and write back once at the end.
            u32 tot = matrix[row * cols];
            for (u32 col = 1; col < cols && !found; ++col) {
                p_to_the_j *= 2u;
                bool handled = false;
                if (p_to_the_j <= tot) {
                    // Bitsum along the anti-diagonal to the bottom-left, with a saturating start.
                    u32 d = 0;
                    u32 c = (row + col + 1 > rows) ? (row + col + 1 - rows) : 0;
                    for (; c < col; ++c) d |= matrix[(row + col - c) * cols + c];

                    u32 cur = matrix[row * cols + col];
                    u32 new_entry = ((cur | d) + 1u) & ~d;
                    u32 inc = new_entry - cur;
                    u32 sub = inc * p_to_the_j;
                    if (tot < sub) {
                        tot += p_to_the_j * cur;
                        handled = true;
                    } else {
                        matrix[row * cols] = tot - sub;
                        masks[row] = matrix[row * cols];
                        col_sums[col - 1] += inc;
                        for (u32 j = 1; j < col; ++j) {
                            masks[row + j] &= ~matrix[row * cols + j];
                            col_sums[j - 1] -= matrix[row * cols + j];
                            matrix[row * cols + j] = 0;
                        }
                        matrix[row * cols + col] = new_entry;
                        // Carry: reset every row below this one back to its running total.
                        for (u32 i = 0; i < row; ++i) {
                            matrix[i * cols] = totals[i];
                            masks[i] = totals[i];
                            for (u32 j2 = 1; j2 < cols; ++j2) {
                                if (i + j2 > row) masks[i + j2] &= ~matrix[i * cols + j2];
                                col_sums[j2 - 1] -= matrix[i * cols + j2];
                                matrix[i * cols + j2] = 0;
                            }
                        }
                        masks[row + col] = d | new_entry;
                        found = true;
                        handled = true;
                    }
                }
                if (!handled) tot += p_to_the_j * matrix[row * cols + col];
            }
            // Write back once per row scan. Rows below this one are read by the carry block above,
            // and each was written back by its own iteration.
            totals[row] = tot;
        }
        more = found;
    }

    out_counts[ti] = mat;
}
