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

// The per-column rule of the Milnor product test, for column `j` with `b`/`cs`/`mk` the term /
// col_sums / masks entries (zero outside their lengths) and `low = min(term_len, cs_len)`:
//   j <  low: reject if cs > b or (b-cs) & mk; else (b-cs) | mk
//   j >= low: reject if cs > 0 or b & mk;      else b | mk
//
// This is the whole per-term test and output assembly of
// `MilnorAlgebra::multiply_basis_element_by_element_2`, with its three tail branches collapsed into
// one uniform per-position rule.
//
// Past `low` at most one of `b`, `cs` is in range, so the two arms in fact coincide there and a
// single unconditional rule would do -- but proving that away is an optimisation, and this file is
// the unoptimised reference.
__device__ __forceinline__ u32 pair_col(u32 j, u32 low, u32 b, u32 cs, u32 mk) {
    if (j < low) {
        if (cs > b) return PAIR_COL_REJECT;
        u32 diff = b - cs;
        if (diff & mk) return PAIR_COL_REJECT;
        return diff | mk;
    }
    if (cs > 0u) return PAIR_COL_REJECT;
    if (b & mk) return PAIR_COL_REJECT;
    return b | mk;
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
    // Width-padded Milnor basis: element `gei`'s p-part at pp[gei*width ..], true length ln[gei].
    const u16 *__restrict__ pp,
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
    u32 term_len[TERM_GROUP];
    u64 b_base[TERM_GROUP];
    u32 low[TERM_GROUP];
    u32 cols = (cs_len > mk_len) ? cs_len : mk_len;
#pragma unroll
    for (u32 tt = 0; tt < TERM_GROUP; ++tt) {
        u32 tl = 0;
        u64 bb = 0;
        if (t_base + tt < nt) {
            u32 gei = term_gei[prod_term_start[p] + t_base + tt];
            tl = ln[gei];
            bb = (u64)gei * width;
        }
        term_len[tt] = tl;
        b_base[tt] = bb;
        low[tt] = (tl < cs_len) ? tl : cs_len;
        if (tl > cols) cols = tl;
    }

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
    u32 rejected[MATRIX_GROUP * TERM_GROUP];
#pragma unroll
    for (u32 i = 0; i < MATRIX_GROUP * TERM_GROUP; ++i) {
        working[i] = 0;
        rejected[i] = 0;
    }

    // Past the longest of the three inputs, b, cs and mk are all zero, so pair_col returns 0 -- no
    // rejection and nothing added. Stopping there is exact, not a truncation.
    for (u32 j = 0; j < cols; ++j) {
        // 2M + T loads feeding M*T evaluations: this is the whole point of the tile.
        u32 cv[MATRIX_GROUP];
        u32 mv[MATRIX_GROUP];
#pragma unroll
        for (u32 mm = 0; mm < MATRIX_GROUP; ++mm) {
            cv[mm] = (j < cs_len) ? (u32)cs[cs_b[mm] + j] : 0u;
            mv[mm] = (j < mk_len) ? (u32)mk[mk_b[mm] + j] : 0u;
        }
        u32 bv[TERM_GROUP];
#pragma unroll
        for (u32 tt = 0; tt < TERM_GROUP; ++tt)
            bv[tt] = (j < term_len[tt]) ? (u32)pp[b_base[tt] + j] : 0u;

        // One shift/mask pair per column, shared by every lane of the tile.
        u32 jj = (j < PPART_MAX_LEN) ? j : 0;
        u32 sh = PP_SHIFT[jj];
        u32 fm = PP_MASK[jj];
#pragma unroll
        for (u32 mm = 0; mm < MATRIX_GROUP; ++mm) {
#pragma unroll
            for (u32 tt = 0; tt < TERM_GROUP; ++tt) {
                u32 i = mm * TERM_GROUP + tt;
                u32 val = pair_col(j, low[tt], bv[tt], cv[mm], mv[mm]);
                rejected[i] |= val & PAIR_COL_REJECT;
                if (j < PPART_MAX_LEN) working[i] |= (u64)(val & fm) << sh;
            }
        }
    }

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
