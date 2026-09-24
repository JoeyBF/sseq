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

// The index of P(working) in the Milnor basis of its degree, from the flat `g` table with no
// hashing -- the device port of `MilnorAlgebra::seqno`.
//
// `g` has row width `width`, entry (e, h) at g[e*width + h]; `xi` are the xi-degrees. Entries at
// index >= PPART_MAX_LEN cannot exist: at p = 2 the entry r_n multiplies deg(xi_n) = 2^n - 1, so a
// p-part of length 11 needs degree >= 2^11 - 1 = 2047, while `PPart::MAX_DEGREE` is 2045. Stopping
// there is forced by that degree bound, not a cap something could exceed.
__device__ __forceinline__ u32 seqno_core(const u32 *__restrict__ g,
                                          const u32 *__restrict__ xi,
                                          const u32 *working, u32 wlen, u32 width) {
    // cur_d = sum_h working[h] * xi[h]
    u32 cur_d = 0;
    for (u32 h = 0; h < wlen; ++h) cur_d += working[h] * xi[h];

    // Rank by consuming positions from high to low; position 0 contributes nothing.
    u32 rank = 0;
    for (u32 hh = 1; hh < wlen; ++hh) {
        u32 h = wlen - hh;
        u32 r = working[h];
        if (r != 0) {
            u32 below = cur_d - r * xi[h];
            rank += g[(u64)cur_d * width + h] - g[(u64)below * width + h];
            cur_d = below;
        }
    }
    return rank;
}

// One launch covering all (R, s) products of a batch; one thread per (product, matrix, term) pair.
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
    // MATRIX varies fastest, term slowest. The obvious decode has the opposite order, and it is the
    // kernel's dominant cost: consecutive threads then share a matrix but each takes a DIFFERENT
    // term, whose p-part sits at an arbitrary basis index -- a 32-way scatter across a multi-GB
    // resident basis, ~2 useful bytes per 32-byte sector fetched. Kept from the start because it is
    // a permutation of the same pair set, not a change to what is computed.
    u32 m = (u32)(local % num_mats);
    u32 t = (u32)(local / num_mats);

    u32 cs_len = r_cs_len[ri];
    u32 mk_len = r_mk_len[ri];
    u64 cs_base = r_cs_offset[ri] + (u64)m * cs_len;
    u64 mk_base = r_mk_offset[ri] + (u64)m * mk_len;

    u32 gei = term_gei[prod_term_start[p] + t];
    u32 term_len = ln[gei];
    u64 b_base = (u64)gei * width;

    // Past the longest of the three inputs, b, cs and mk are all zero, so pair_col returns 0 -- no
    // rejection and nothing added to `working`. Stopping there is exact, not a truncation.
    u32 cols = cs_len;
    if (mk_len > cols) cols = mk_len;
    if (term_len > cols) cols = term_len;

    u32 low = (term_len < cs_len) ? term_len : cs_len;
    u32 working[PPART_MAX_LEN];
#pragma unroll
    for (u32 i = 0; i < PPART_MAX_LEN; ++i) working[i] = 0;

    u32 rejected = 0;
    for (u32 j = 0; j < cols; ++j) {
        u32 b = (j < term_len) ? (u32)pp[b_base + j] : 0u;
        u32 c = (j < cs_len) ? (u32)cs[cs_base + j] : 0u;
        u32 msk = (j < mk_len) ? (u32)mk[mk_base + j] : 0u;
        u32 val = pair_col(j, low, b, c, msk);
        rejected |= val & PAIR_COL_REJECT;
        if (j < PPART_MAX_LEN) working[j] = val & 0xffffu;
    }
    if (rejected) return;

    // No explicit trailing-zero trim is needed: seqno_core skips zero entries and `working` beyond
    // the assembled length is zero, so running the full PPART_MAX_LEN is equivalent to the CPU's
    // trimmed p_part. `xi` is host-padded to at least PPART_MAX_LEN so the cur_d sum stays in
    // bounds; the extra terms are 0 * xi.
    u32 idx = seqno_core(g, xi, working, PPART_MAX_LEN, width);

    // `out_offset` shifts the basis index to this product's target-generator block within the row
    // (0 for a single-block output). Both are bit offsets, added before splitting into (limb, bit).
    u64 bit_pos = (u64)prod_out_offset[p] + idx;
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
    u64 word = (u64)prod_row_base[p] + limb;
    if (word >= out_len) return;
    atomicXor(&out[word], 1u << (u32)(bit_pos % 32));
}
