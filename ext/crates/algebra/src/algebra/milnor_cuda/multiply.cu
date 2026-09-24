// The batched Milnor multiply, in CUDA C.
//
// THIS IS DELIBERATELY THE SIMPLEST CORRECT FORM. The cubecl kernel it replaces carries several
// measured optimisations layered on top of this arithmetic -- MxT thread tiling, a coarse index
// bracketing the product search, a 32-bit column loop, a packed u64 accumulator, a transposed
// master, and a per-launch column bound. Every one of them is worth re-introducing, and every one
// is a chance to change the answer silently. So the port starts from a version whose only job is
// to agree with `cpu_multiply_batch`, and each optimisation goes back in one at a time with the
// output digest as the invariant.
//
// Tuning constants arrive as NVRTC `-D` options from `params.rs` -- there is no default here, so a
// forgotten define is a compile error rather than a silently wrong value.

#ifndef PPART_MAX_LEN
#error "PPART_MAX_LEN must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef THREADS
#error "THREADS must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef COARSE_LOG
#error "COARSE_LOG must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef PP_SHIFTS
#error "PP_SHIFTS must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef PP_MASKS
#error "PP_MASKS must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef TERM_GROUP
#error "TERM_GROUP must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef COL_SPLIT_32
#error "COL_SPLIT_32 must be provided as an NVRTC -D option (see params.rs)"
#endif
#ifndef MATRIX_GROUP
#error "MATRIX_GROUP must be provided as an NVRTC -D option (see params.rs)"
#endif

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

// Marks a full output column that the signature mask discards. Mirrors `COL_MAP_DROP`.
#define COL_MAP_DROP 0xffffffffu

// Set by `pair_col` to report that a column kills the whole product, above the 16 bits the value
// itself occupies. Returning it inside the value rather than out of band keeps the caller
// branchless -- it ORs the flag into an accumulator and stores the low half unconditionally.
#define PAIR_COL_REJECT (1u << 16)

// The per-column rule of the Milnor product test: with `b`/`cs`/`mk` the term / col_sums / masks
// entries (zero outside their lengths), reject if `cs > b` or `(b - cs) & mk`, else `(b - cs) | mk`.
//
// This is the whole per-term test and output assembly of
// `MilnorAlgebra::multiply_basis_element_by_element_2`, with its three tail branches collapsed into
// one uniform per-position rule.
//
// NO `low` SPLIT, because it is redundant rather than a trade. The two-armed form is
//   j <  low: reject if cs > b or (b-cs) & mk; else (b-cs) | mk
//   j >= low: reject if cs > 0 or b & mk;      else b | mk
// with `low = min(term_len, cs_len)`. Past `low` at most one of `b`, `cs` is in range, so at least
// one is zero, and the two cases are:
//   * cs = 0: the first arm gives diff = b, rejects on `b & mk`, else `b | mk`; the second has
//     `cs > 0` false, rejects on `b & mk`, else `b | mk`. Identical.
//   * b = 0: the first rejects exactly when `cs > 0`, and otherwise (cs = 0 too) gives diff = 0,
//     no `0 & mk`, value `mk`; the second rejects exactly when `cs > 0` and otherwise gives
//     `0 | mk = mk`. Identical.
// So the first arm computes the second's answer wherever the second applies, and one unconditional
// rule covers every column. The split cost a compare, a branch and its reconvergence pair per LANE
// per column -- MATRIX_GROUP * TERM_GROUP of them per iteration -- on a loop that does not actually
// diverge. What is left is branchless: a compare, a subtract, two LOP3s and a select.
__device__ __forceinline__ u32 pair_col(u32 b, u32 cs, u32 mk) {
    if (cs > b) return PAIR_COL_REJECT;
    u32 diff = b - cs;
    if (diff & mk) return PAIR_COL_REJECT;
    return diff | mk;
}

// Where each p-part digit sits inside the packed accumulator, straight from `PPart`'s own layout.
//
// `__device__ const` rather than a kernel argument: the values are compile-time constants, every
// lane of a warp reads the same entry, and a broadcast out of the constant cache beats a global
// load. They arrive as -D brace initializers so `PPart::shift` stays the single source of truth.
__device__ static const u32 PP_SHIFT[PPART_MAX_LEN] = PP_SHIFTS;
__device__ static const u32 PP_MASK[PPART_MAX_LEN] = PP_MASKS;

// The index of P(working) in the Milnor basis of its degree, from the flat `g` table with no
// hashing -- the device port of `MilnorAlgebra::seqno`.
//
// `g` has row width `width`, entry (e, h) at g[e*width + h]; `xi` are the xi-degrees. Entries at
// index >= PPART_MAX_LEN cannot exist: at p = 2 the entry r_n multiplies deg(xi_n) = 2^n - 1, so a
// p-part of length 11 needs degree >= 2^11 - 1 = 2047, while `PPart::MAX_DEGREE` is 2045. Stopping
// there is forced by that degree bound, not a cap something could exceed.
__device__ __forceinline__ u32 seqno_core(const u32 *__restrict__ g,
                                          const u32 *__restrict__ xi,
                                          u64 working, u32 wlen, u32 width) {
    // cur_d = sum_h working[h] * xi[h], reading digits out of the packed word.
    u32 cur_d = 0;
    for (u32 h = 0; h < wlen; ++h)
        cur_d += (u32)((working >> PP_SHIFT[h]) & PP_MASK[h]) * xi[h];

    // Rank by consuming positions from high to low; position 0 contributes nothing.
    u32 rank = 0;
    for (u32 hh = 1; hh < wlen; ++hh) {
        u32 h = wlen - hh;
        u32 r = (u32)((working >> PP_SHIFT[h]) & PP_MASK[h]);
        if (r != 0) {
            u32 below = cur_d - r * xi[h];
            rank += g[(u64)cur_d * width + h] - g[(u64)below * width + h];
            cur_d = below;
        }
    }
    return rank;
}

// Write one accepted pair's F2 bit. Factored out because a tile has MATRIX_GROUP*TERM_GROUP of
// these, and duplicating the bounds reasoning that many times is how one copy ends up missing a
// guard.
__device__ __forceinline__ void emit_bit(const u32 *__restrict__ g, const u32 *__restrict__ xi,
                                         u32 *out, const u32 *__restrict__ col_map, u32 col_map_len,
                                         u32 use_col_map, u64 working, u32 out_offset, u32 row_base,
                                         u32 width, u32 num_limbs, u64 out_len) {
    // No explicit trailing-zero trim is needed: seqno_core skips zero entries and `working` beyond
    // the assembled length is zero, so running the full PPART_MAX_LEN is equivalent to the CPU's
    // trimmed p_part.
    u32 idx = seqno_core(g, xi, working, PPART_MAX_LEN, width);

    // `out_offset` shifts the basis index to this product's target-generator block within the row
    // (0 for a single-block output). Both are bit offsets, added before splitting into (limb, bit).
    u64 bit_pos = (u64)out_offset + idx;
    if (use_col_map) {
        // bit_pos can exceed the full width -- a kept block's out_offset + seqno may span past it,
        // which is exactly what the unmasked path drops via the num_limbs test below.
        if (bit_pos >= (u64)col_map_len) return;
        u32 mapped = col_map[bit_pos];
        if (mapped == COL_MAP_DROP) return;
        bit_pos = mapped;
    }
    // Two independent bounds, both required: `limb < num_limbs` keeps the write inside this row
    // (out_offset + seqno can span past it, which would silently corrupt the NEXT row), and
    // `word < out_len` guards the buffer itself. compute-sanitizer caught both as distinct
    // out-of-bounds atomics when either was missing.
    u64 limb = bit_pos / 32;
    if (limb >= (u64)num_limbs) return;
    u64 word = (u64)row_base + limb;
    if (word >= out_len) return;
    atomicXor(&out[word], 1u << (u32)(bit_pos % 32));
}

// One launch covering all (R, s) products of a batch; one thread per TILE of
// MATRIX_GROUP x TERM_GROUP pairs.
//
// The pair a thread handles is DECODED rather than tabulated. A per-pair table would be seven
// arrays of total-pair length -- gigabytes at scale, almost all of it redundant. Instead
// `prod_pair_start` is a prefix sum of each product's pair count, so a binary search finds the
// owning product and the remainder splits into (matrix, term). Admissible-matrix data is
// deduplicated by distinct R, so an R shared across many rows is stored and uploaded once and
// `prod_r_index` points at it.
//
// Output is `num_rows` F2 vectors of `num_limbs` u32 limbs, row r at out[r*num_limbs ..]. Bits are
// XOR-accumulated, so a pair evaluated twice CANCELS and a pair dropped silently changes the
// answer -- which is why the decode order is free to be any permutation of the same (m, t) set,
// and why the atomics need no ordering between them.
extern "C" __global__ __launch_bounds__(THREADS) void multiply_batch(
    // Admissible-matrix master, per distinct R. ONE pointer, not sixteen segments: the cubecl
    // kernel took 48 buffer arguments purely because cubecl could not grow an allocation in place,
    // so the master was split across MASTER_MAX_SEG separate buffers bound individually. CUDA's
    // virtual memory API grows a reservation without moving it, so the segmentation -- and the
    // `seg_read_*` helpers, and the comptime segment-count specialisation, and the 16-segment
    // ceiling that made small-memory cards fail -- has nothing left to solve.
    const u16 *__restrict__ cs,
    const u16 *__restrict__ mk,
    // The Milnor basis, ONE PACKED u64 PER ELEMENT: pp[gei] is the whole p-part, with entry i at
    // bit PP_SHIFT[i]. `PPart` is already exactly this u64, so nothing is converted on either side
    // -- and a term's contribution to a column becomes a register shift instead of a global load,
    // which removes the T of the tile's 2M + T loads per column outright.
    const u64 *__restrict__ pp,
    // Trimmed p-part length. Still needed even though the packed word gives the entries: the
    // column loop bounds itself by it and `low` is min(term_len, cs_len).
    const u32 *__restrict__ ln,
    // Term global basis index, indexed by prod_term_start[p] + t.
    const u32 *__restrict__ term_gei,
    const u32 *__restrict__ g,
    const u32 *__restrict__ xi,
    u32 *out,
    // Full output column -> masked position, or COL_MAP_DROP for the columns the signature mask
    // discards. Read only when use_col_map != 0; a one-element dummy is bound otherwise, since the
    // argument is not optional. Applying it here rather than on readback is what lets `out` be
    // allocated at the masked width instead of the full one.
    const u32 *__restrict__ col_map,
    u32 col_map_len,
    u32 use_col_map,
    // Per-R tables, indexed by prod_r_index[p]. The offsets are u64: the master exceeds u32::MAX
    // elements at high stems.
    const u64 *__restrict__ r_cs_offset,
    const u64 *__restrict__ r_mk_offset,
    const u32 *__restrict__ r_cs_len,
    const u32 *__restrict__ r_mk_len,
    const u32 *__restrict__ r_num_mats,
    // Per-product records.
    const u32 *__restrict__ prod_r_index,
    const u32 *__restrict__ prod_term_start,
    const u32 *__restrict__ prod_num_terms,
    const u32 *__restrict__ prod_row_base,
    const u32 *__restrict__ prod_out_offset,
    // Prefix sum of pair counts, length num_products + 1. u64: a single unsplittable row's pair
    // space can exceed 2^32.
    const u64 *__restrict__ prod_pair_start,
    // Coarse index over the pair space: entry `ci` is the product owning pair `ci << COARSE_LOG`,
    // so the product owning any pair in that bucket lies in `[prod_coarse[ci], prod_coarse[ci+1]]`.
    // One entry past the last bucket, so `ci + 1` is always readable.
    const u32 *__restrict__ prod_coarse,
    u32 num_products,
    // First pair index this launch covers, so a pair space past the 32-bit thread index can be
    // walked in pieces. 0 for every launch of a normally-sized block.
    u64 pair_offset,
    u32 width,
    u32 num_limbs,
    u64 out_len) {
    u64 k = pair_offset + (u64)blockIdx.x * blockDim.x + threadIdx.x;
    // Also retires the trailing threads of the final chunk, whose grid is rounded up to whole
    // blocks.
    if (k >= prod_pair_start[num_products]) return;

    // Largest product p with prod_pair_start[p] <= k. Every product owns at least one pair, so
    // prod_pair_start is strictly increasing and p is unique.
    //
    // `prod_coarse` BRACKETS the search before it starts. Each step of an unbracketed search is a
    // DEPENDENT global load of prod_pair_start[mid] -- a full latency stall before the thread can
    // touch its own data. MEASURED on a real stem-130 batch, interleaved, kernel time only:
    // 33.43e9 pairs/s unbracketed against 37.46e9 bracketed, i.e. 1.120x, matching the ~12% the
    // cubecl ablation found. Two cheap loads replace ~15 dependent ones.
    //
    // The bracket is valid because products are ordered and each owns at least one pair; the
    // sentinel entry keeps ci+1 readable for the final bucket.
    u32 ci = (u32)(k >> COARSE_LOG);
    u32 lo = prod_coarse[ci];
    u32 hi = prod_coarse[ci + 1] + 1;
    if (hi > num_products) hi = num_products;
    while (hi - lo > 1) {
        u32 mid = (lo + hi) / 2;
        if (prod_pair_start[mid] <= k) lo = mid; else hi = mid;
    }
    u32 p = lo;

    u32 ri = prod_r_index[p];
    u64 local = k - prod_pair_start[p];
    u32 num_mats = r_num_mats[ri];
    u32 nt = prod_num_terms[p];

    // A THREAD COVERS A TILE of MATRIX_GROUP matrices x TERM_GROUP terms.
    //
    // col_sums/masks depend only on the matrix and a term's p-part only on the term, so an MxT tile
    // reads 2M + T values per column to evaluate M*T pairs -- 1.17 loads per pair at 2x3, against
    // 1.67 at 1x3 and 3 at 1x1. The kernel is issue-limited on integer work, so fewer loads and
    // fewer addresses is the lever.
    //
    // The two axes are NOT symmetric. Terms are few (nt ~ 5), so the ragged tail dominates the
    // choice of TERM_GROUP and 4 loses to 3 purely on wasted lanes. Matrices are many (~20000), so
    // a partial matrix tile costs a few idle lanes out of thousands.
    //
    // Matrix varies fastest. The opposite order gives each lane of a warp a different term, whose
    // p-part sits at an arbitrary basis index -- a 32-way scatter across a multi-GB resident basis.
    u32 mg_count = (num_mats + MATRIX_GROUP - 1) / MATRIX_GROUP;
    u32 mg = (u32)(local % mg_count);
    u32 tg = (u32)(local / mg_count);
    u32 m_base = mg * MATRIX_GROUP;
    u32 t_base = tg * TERM_GROUP;

    u32 cs_len = r_cs_len[ri];
    u32 mk_len = r_mk_len[ri];

    // Per-term p-part offsets and lengths. Lanes past `nt` carry term_len = 0 and are excluded at
    // the emit below: an all-zero term against an all-zero column does NOT reject, so they would
    // otherwise emit a spurious seqno(0) bit.
    // The term's whole p-part, held in a REGISTER for the length of the column loop. One load per
    // term for the entire tile, against one per term per column before.
    //
    // The trimmed LENGTH is no longer kept per lane: with the digits coming out of the packed word
    // it is read only to form `low[tt]` and to widen `cols`, both done here.
    u64 b_bits[TERM_GROUP];
    u32 cols = (cs_len > mk_len) ? cs_len : mk_len;
#pragma unroll
    for (u32 tt = 0; tt < TERM_GROUP; ++tt) {
        u32 tl = 0;
        u64 bb = 0;
        if (t_base + tt < nt) {
            u32 gei = term_gei[prod_term_start[p] + t_base + tt];
            tl = ln[gei];
            bb = pp[gei];
        }
        b_bits[tt] = bb;
        if (tl > cols) cols = tl;
    }

    // COLUMNS AT OR PAST PPART_MAX_LEN DO NOTHING, so do not visit them.
    //
    // Both of the inputs that can reject are bounded by PPART_MAX_LEN: a term's p-part has at most
    // `PPart::MAX_LEN` = 10 entries, and cs_len = cols - 1 where cols is the widest bit-length of a
    // p-part entry, bounded by PPart::width(0) = 11. So for j >= 10 both b and cs are zero, and
    // `pair_col`'s second arm reduces to "cs > 0" (false) and "b & mk" (zero) -- it CANNOT reject.
    // The accumulate is already guarded by j < PPART_MAX_LEN. The column therefore contributes
    // nothing at all.
    //
    // mk_len is rows + cols - 1 and runs well past 10 (to 18 in principle, ~9.1 mean at the
    // frontier), so this is real work: the cubecl kernel keeps a whole third comptime loop segment
    // for `PPART_MAX_LEN..cols`, describing it as there "for correctness at any shape". The bound
    // above says there is no such shape. `enum_caps_bound_real_rs` checks the cs_len half of that
    // argument against every real R, and the digest checks the conclusion.
    if (cols > PPART_MAX_LEN) cols = PPART_MAX_LEN;

    // Matrix lane bases, CLAMPED rather than branched. A trailing lane of a partial tile then reads
    // a real (duplicate) matrix instead of nothing, and the emit tail drops it on the same
    // condition -- so nothing it computes is ever observed, while the branch and its reconvergence
    // leave the column loop entirely. num_mats >= 1 for every live R (mg_count divides by it), so
    // the clamp always names a real matrix.
    u64 cs_b[MATRIX_GROUP];
    u64 mk_b[MATRIX_GROUP];
#pragma unroll
    for (u32 mm = 0; mm < MATRIX_GROUP; ++mm) {
        u32 mi = m_base + mm;
        if (mi >= num_mats) mi = num_mats - 1;
        cs_b[mm] = r_cs_offset[ri] + (u64)mi * cs_len;
        mk_b[mm] = r_mk_offset[ri] + (u64)mi * mk_len;
    }

    u64 working[MATRIX_GROUP * TERM_GROUP];
    // 32-BIT ACCUMULATOR for the columns below COL_SPLIT_32, folded into `working` before the
    // 64-bit segment starts, so the two are never live at the same time.
    //
    // `PPart`'s first three fields are 11, 10 and 9 bits at shifts 0, 11 and 21, so digit 2 ends at
    // bit 30 and those three digits fit wholly in a u32, while digit 3 starts at bit 30 and
    // straddles the boundary. Below the split the shift-and-or is 32-bit: half the shifts and half
    // the ORs, per LANE per column, and there are MATRIX_GROUP * TERM_GROUP lanes.
    //
    // `column_split_is_exactly_at_bit_32` checks both halves of that against `PPart::shift`, so a
    // change to the packing fails a test rather than silently dropping the high half of a digit.
    u32 acc32[MATRIX_GROUP * TERM_GROUP];
    u32 rejected[MATRIX_GROUP * TERM_GROUP];
#pragma unroll
    for (u32 i = 0; i < MATRIX_GROUP * TERM_GROUP; ++i) {
        working[i] = 0;
        acc32[i] = 0;
        rejected[i] = 0;
    }

// The read half of a column, shared by both segments. A macro rather than two hand copies: the two
// differ ONLY in the width of the accumulate, and letting them drift would be a correctness bug
// that appears only for p-parts long enough to reach the second segment.
#define READ_COLUMN(jj)                                           \
    u32 cv[MATRIX_GROUP];                                         \
    u32 mv[MATRIX_GROUP];                                         \
    _Pragma("unroll") for (u32 mm = 0; mm < MATRIX_GROUP; ++mm) { \
        cv[mm] = ((jj) < cs_len) ? (u32)cs[cs_b[mm] + (jj)] : 0u; \
        mv[mm] = ((jj) < mk_len) ? (u32)mk[mk_b[mm] + (jj)] : 0u; \
    }                                                             \
    u32 sh = PP_SHIFT[jj];                                        \
    u32 fm = PP_MASK[jj];                                         \
    u32 bv[TERM_GROUP];                                           \
    _Pragma("unroll") for (u32 tt = 0; tt < TERM_GROUP; ++tt)     \
        bv[tt] = (u32)((b_bits[tt] >> sh) & fm);

    // Past the longest of the three inputs, b, cs and mk are all zero, so pair_col returns 0 -- no
    // rejection and nothing added. Stopping there is exact, not a truncation.
    u32 end_lo = (cols < COL_SPLIT_32) ? cols : COL_SPLIT_32;
    for (u32 j = 0; j < end_lo; ++j) {
        READ_COLUMN(j)
#pragma unroll
        for (u32 mm = 0; mm < MATRIX_GROUP; ++mm) {
#pragma unroll
            for (u32 tt = 0; tt < TERM_GROUP; ++tt) {
                u32 i = mm * TERM_GROUP + tt;
                u32 val = pair_col(bv[tt], cv[mm], mv[mm]);
                rejected[i] |= val & PAIR_COL_REJECT;
                acc32[i] |= (val & fm) << sh;
            }
        }
    }
    // EARLY EXIT when the whole tile is already dead.
    //
    // Roughly 99.5% of pairs reject (9.2M bits set from 1.74e9 pairs on the profile batch), and a
    // rejection is permanent -- `rejected` only ever accumulates. So if every lane of the tile has
    // rejected by the end of the first segment, the rest of the column loop and the emit cannot
    // produce anything.
    //
    // ONE branch per thread, not one per column. Branching per column was measured ~11% SLOWER on
    // the cubecl kernel: twelve divergent branches per thread cost more than the work they skip,
    // on a loop that does not otherwise diverge. This asks once, at a point where the answer is
    // usually yes.
    //
    // A SECOND check after the very first column was tried and rejected: 84.08 -> 84.78 e9 pairs/s,
    // +0.8%, for a peeled iteration carrying a third copy of the tile accumulate. Real but not
    // worth the drift risk of another copy. Do not re-try it expecting more.
    u32 all_rejected = rejected[0];
#pragma unroll
    for (u32 i = 1; i < MATRIX_GROUP * TERM_GROUP; ++i) all_rejected &= rejected[i];
    if (all_rejected) return;

#pragma unroll
    for (u32 i = 0; i < MATRIX_GROUP * TERM_GROUP; ++i) working[i] = (u64)acc32[i];

    for (u32 j = end_lo; j < cols; ++j) {
        READ_COLUMN(j)
#pragma unroll
        for (u32 mm = 0; mm < MATRIX_GROUP; ++mm) {
#pragma unroll
            for (u32 tt = 0; tt < TERM_GROUP; ++tt) {
                u32 i = mm * TERM_GROUP + tt;
                u32 val = pair_col(bv[tt], cv[mm], mv[mm]);
                rejected[i] |= val & PAIR_COL_REJECT;
                working[i] |= (u64)(val & fm) << sh;
            }
        }
    }
#undef READ_COLUMN


    // Emit, dropping the lanes a partial tile invented.
#pragma unroll
    for (u32 mm = 0; mm < MATRIX_GROUP; ++mm) {
        if (m_base + mm >= num_mats) continue;
#pragma unroll
        for (u32 tt = 0; tt < TERM_GROUP; ++tt) {
            if (t_base + tt >= nt) continue;
            u32 i = mm * TERM_GROUP + tt;
            if (rejected[i]) continue;
            emit_bit(g, xi, out, col_map, col_map_len, use_col_map, working[i], prod_out_offset[p],
                     prod_row_base[p], width, num_limbs, out_len);
        }
    }
}
