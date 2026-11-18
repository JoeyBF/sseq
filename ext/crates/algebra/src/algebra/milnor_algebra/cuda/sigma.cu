// The product terms of the pair-algebra product over Z/4: `for_each_term_mod4` of
// `antidiagonal.rs` on the device, one thread per (R, S) pair. Compiled after `multiply.cu`, whose
// `seqno`, `PP_SHIFT` and `PP_MASK` it uses.

#define M4_CELLS (PPART_MAX_LEN * (PPART_MAX_LEN - 1) / 2)

// An anti-diagonal's multinomial coefficient mod 4 as its entries are added: `Mod4`.
struct M4 {
    u32 sum;
    u32 carried;
    u32 odd;
};

// Add `v`; false if the coefficient becomes 0 mod 4, leaving `st` as it was.
__device__ __forceinline__ bool m4_add(M4 &st, u32 v) {
    u32 n = st.sum + v;
    u32 carries = __popc(st.sum) + __popc(v) - __popc(n);
    if (carries == 0) {
        st.odd ^= __popc(n & (n >> 1) & (v ^ (v >> 1))) & 1u;
        st.sum = n;
        return true;
    }
    if (carries == 1 && !st.carried) {
        st.carried = 1;
        st.odd = 0;
        st.sum = n;
        return true;
    }
    return false;
}

__device__ __forceinline__ u32 m4_coefficient(const M4 &st) {
    return st.carried ? 2u : (st.odd ? 3u : 1u);
}

// The least x > cur with no bit of `forbid` and every bit of `force`: `next_with`.
__device__ __forceinline__ u64 m4_next_with(u64 cur, u64 forbid, u64 force) {
    u64 x = cur + 1;
    for (;;) {
        u64 bad = x & forbid;
        if (bad) {
            u32 h = 63 - __clzll(bad);
            x = (x | ((1ull << h) - 1)) + 1;
            continue;
        }
        u64 missing = force & ~x;
        if (missing) {
            u32 h = 63 - __clzll(missing);
            x = ((x >> h) | 1ull) << h;
            continue;
        }
        return x > 0xffffffffull ? ~0ull : x;
    }
}

// The least v > cur addable to `st`: `Mod4::next`.
__device__ __forceinline__ u64 m4_next(const M4 &st, u32 cur) {
    u64 s = st.sum;
    u64 best = m4_next_with(cur, s, 0);
    if (!st.carried) {
        u64 bits = s & ~(s >> 1);
        while (bits) {
            u32 i = __ffsll(bits) - 1;
            bits &= bits - 1;
            u64 forbid = (s & ~(1ull << i)) | (1ull << (i + 1));
            u64 c = m4_next_with(cur, forbid, 1ull << i);
            best = c < best ? c : best;
        }
    }
    return best;
}

__device__ __forceinline__ u32 pp_get(u64 p, u32 i) { return (u32)((p >> PP_SHIFT[i]) & PP_MASK[i]); }

__device__ __forceinline__ u32 pp_len(u64 p) {
    u32 len = 0;
    for (u32 i = 0; i < PPART_MAX_LEN; ++i)
        if (pp_get(p, i)) len = i + 1;
    return len;
}

// Add c (1, 2 or 3) mod 4 to the digit pair (ones, twos) of the term's bit. atomicXor returns the
// old bit of `ones`, so the carry into `twos` is exact whatever order threads add in.
__device__ __forceinline__ void m4_emit(const u32 *__restrict__ g, const u32 *__restrict__ xi,
                                        u32 *ones, u32 *twos, u64 t, u32 c, u64 row_word,
                                        u32 col_offset, u32 width, u32 num_limbs) {
    u64 bit = (u64)col_offset + seqno(g, xi, t, width);
    u64 limb = bit / 32;
    if (limb >= (u64)num_limbs) return;
    u64 w = row_word + limb;
    u32 mask = 1u << (u32)(bit % 32);
    u32 carry = 0;
    if (c & 1u) carry = (atomicXor(&ones[w], mask) & mask) ? 1u : 0u;
    if (((c >> 1) ^ carry) & 1u) atomicXor(&twos[w], mask);
}

extern "C" __global__ void sigma_mod4(const u64 *__restrict__ item_r, const u64 *__restrict__ item_s,
                                      const int *__restrict__ item_degree,
                                      const u32 *__restrict__ item_row,
                                      const u32 *__restrict__ item_col, u32 num_items,
                                      const u32 *__restrict__ g, const u32 *__restrict__ xi,
                                      u32 width, u32 *ones, u32 *twos, u32 num_limbs) {
    u32 item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= num_items) return;
    const u64 r = item_r[item], s = item_s[item];
    const int degree = item_degree[item];
    const u64 row_word = (u64)item_row[item] * num_limbs;
    const u32 col = item_col[item];
    const u32 rows = pp_len(r), cols = pp_len(s);
    if (rows == 0 || cols == 0) {
        m4_emit(g, xi, ones, twos, rows == 0 ? s : r, 1, row_word, col, width, num_limbs);
        return;
    }
    u32 max_diag = rows + cols < PPART_MAX_LEN ? rows + cols : PPART_MAX_LEN;
    while (max_diag > 1 && (long long)((1ll << max_diag) - 1) > (long long)degree) --max_diag;

    u32 rl[PPART_MAX_LEN + 1], sl[PPART_MAX_LEN + 1];
    for (u32 i = 0; i <= PPART_MAX_LEN; ++i) rl[i] = sl[i] = 0;
    for (u32 i = 0; i < rows; ++i) rl[i + 1] = pp_get(r, i);
    for (u32 j = 0; j < cols; ++j) sl[j + 1] = pp_get(s, j);

    unsigned char crow[M4_CELLS], ccol[M4_CELLS], clast[M4_CELLS];
    u32 ncells = 0;
    for (u32 d = max_diag; d >= 2; --d) {
        u32 lo = d > cols ? d - cols : 1;
        if (lo < 1) lo = 1;
        u32 hi = rows < d - 1 ? rows : d - 1;
        for (u32 i = lo; i <= hi; ++i) {
            crow[ncells] = (unsigned char)i;
            ccol[ncells] = (unsigned char)(d - i);
            clast[ncells] = 0;
            ++ncells;
        }
        clast[ncells - 1] = 1;
    }

    M4 first = {rl[max_diag], 0, 0};
    if (!m4_add(first, sl[max_diag])) return;
    if (max_diag == 1) {
        m4_emit(g, xi, ones, twos, (u64)first.sum << PP_SHIFT[0], m4_coefficient(first), row_word,
                col, width, num_limbs);
        return;
    }

    u32 value[M4_CELLS], bound[M4_CELLS];
    M4 before[M4_CELLS];
    u64 t_bits = 0;
    u32 k = 0;
    M4 state = first;
    for (;;) {
        // Enter cell k with value 0.
        u32 i = crow[k], j = ccol[k];
        before[k] = state;
        u32 b = rl[i] >> j;
        bound[k] = b < sl[j] ? b : sl[j];
        value[k] = 0;
        M4 after = state;
        bool entered = false;
        for (;;) {
            if (!clast[k]) {
                ++k;
                state = after;
                entered = true;
                break;
            }
            u32 d = crow[k] + ccol[k];
            u32 shift = PP_SHIFT[d - 1];
            t_bits = (t_bits & ~((u64)PP_MASK[d - 1] << shift)) | ((u64)after.sum << shift);
            M4 nxt = {rl[d - 1], after.carried, after.odd};
            if (m4_add(nxt, sl[d - 1])) {
                if (d == 2) {
                    m4_emit(g, xi, ones, twos, t_bits | ((u64)nxt.sum << PP_SHIFT[0]),
                            m4_coefficient(nxt), row_word, col, width, num_limbs);
                } else {
                    ++k;
                    state = nxt;
                    entered = true;
                    break;
                }
            }
            // Backtrack to the latest cell that can still be increased.
            bool advanced = false;
            for (;;) {
                u32 ci = crow[k], cj = ccol[k];
                u32 v = value[k];
                u64 n = m4_next(before[k], v);
                if (n <= (u64)bound[k]) {
                    u32 nv = (u32)n;
                    rl[ci] -= (nv - v) << cj;
                    sl[cj] -= nv - v;
                    value[k] = nv;
                    after = before[k];
                    m4_add(after, nv);
                    advanced = true;
                    break;
                }
                rl[ci] += v << cj;
                sl[cj] += v;
                if (k == 0) return;
                --k;
            }
            if (!advanced) return;
        }
        if (!entered) return;
    }
}
