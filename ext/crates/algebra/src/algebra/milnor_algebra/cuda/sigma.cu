// The pair-algebra product sigma(R, S) of basis elements (`sigma_multiply_basis_direct` of
// `pair_algebra.rs`) on the device, one thread per (R, S) pair. Compiled after `multiply.cu`, whose
// `seqno`, `PP_SHIFT` and `PP_MASK` it uses.

#define M4_CELLS (PPART_MAX_LEN * (PPART_MAX_LEN - 1) / 2)

// Row stride of the table of Y-vector offsets: entry (D, i, j) at yoff[(D * Y_STRIDE + i) *
// Y_STRIDE + j]. Mirrored by `Y_STRIDE` in `mod.rs`.
#define Y_STRIDE 16
#define Y_ABSENT 0xffffffffu

// Which parts of sigma a launch computes. Mirrored in `mod.rs`.
#define PART_PRODUCT 1u
#define PART_Y 2u

// An anti-diagonal's multinomial coefficient mod 4 as its entries are added: `Mod4`.
struct M4 {
    u32 sum;
    u32 carried;
    u32 odd;
};

// The same mod 2: the entries must be bitwise disjoint.
struct M2 {
    u32 sum;
};

// Add `v`; false if the coefficient becomes 0 mod 4, leaving `st` as it was.
__device__ __forceinline__ bool st_add(M4 &st, u32 v) {
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

__device__ __forceinline__ bool st_add(M2 &st, u32 v) {
    if (st.sum & v) return false;
    st.sum |= v;
    return true;
}

__device__ __forceinline__ u32 st_coefficient(const M4 &st) {
    return st.carried ? 2u : (st.odd ? 3u : 1u);
}

__device__ __forceinline__ u32 st_coefficient(const M2 &) { return 1u; }

// A fresh diagonal starting from `base`, carrying the coefficient of the diagonals before it.
__device__ __forceinline__ M4 st_diagonal(const M4 &prev, u32 base) {
    return {base, prev.carried, prev.odd};
}

__device__ __forceinline__ M2 st_diagonal(const M2 &, u32 base) { return {base}; }

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
__device__ __forceinline__ u64 st_next(const M4 &st, u32 cur) {
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

__device__ __forceinline__ u64 st_next(const M2 &st, u32 cur) { return m4_next_with(cur, st.sum, 0); }

__device__ __forceinline__ u32 pp_get(u64 p, u32 i) { return (u32)((p >> PP_SHIFT[i]) & PP_MASK[i]); }

__device__ __forceinline__ u64 pp_set(u64 p, u32 i, u32 v) {
    return (p & ~((u64)PP_MASK[i] << PP_SHIFT[i])) | ((u64)v << PP_SHIFT[i]);
}

__device__ __forceinline__ u32 pp_len(u64 p) {
    u32 len = 0;
    for (u32 i = 0; i < PPART_MAX_LEN; ++i)
        if (pp_get(p, i)) len = i + 1;
    return len;
}

// The terms of P(r) P(s), of degree `degree`, with their coefficients mod 4 (St = M4) or 2
// (St = M2): `for_each_term_mod4` and `for_each_term` of `antidiagonal.rs`, the same walk as
// `multiply.cu`'s with the state in registers and local memory.
template <class St, class Emit>
__device__ __forceinline__ void walk(u64 r, u64 s, int degree, Emit &emit) {
    const u32 rows = pp_len(r), cols = pp_len(s);
    if (rows == 0 || cols == 0) {
        emit(rows == 0 ? s : r, 1u);
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
        u32 hi = rows < d - 1 ? rows : d - 1;
        for (u32 i = lo; i <= hi; ++i) {
            crow[ncells] = (unsigned char)i;
            ccol[ncells] = (unsigned char)(d - i);
            clast[ncells] = 0;
            ++ncells;
        }
        clast[ncells - 1] = 1;
    }

    St first = st_diagonal(St{}, rl[max_diag]);
    if (!st_add(first, sl[max_diag])) return;
    if (max_diag == 1) {
        emit((u64)first.sum << PP_SHIFT[0], st_coefficient(first));
        return;
    }

    u32 value[M4_CELLS], bound[M4_CELLS];
    St before[M4_CELLS];
    u64 t_bits = 0;
    u32 k = 0;
    St state = first;
    for (;;) {
        // Enter cell k with value 0.
        u32 i = crow[k], j = ccol[k];
        before[k] = state;
        u32 b = rl[i] >> j;
        bound[k] = b < sl[j] ? b : sl[j];
        value[k] = 0;
        St after = state;
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
            St nxt = st_diagonal(after, rl[d - 1]);
            if (st_add(nxt, sl[d - 1])) {
                if (d == 2) {
                    emit(t_bits | ((u64)nxt.sum << PP_SHIFT[0]), st_coefficient(nxt));
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
                u64 n = st_next(before[k], v);
                if (n <= (u64)bound[k]) {
                    u32 nv = (u32)n;
                    rl[ci] -= (nv - v) << cj;
                    sl[cj] -= nv - v;
                    value[k] = nv;
                    after = before[k];
                    st_add(after, nv);
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

// Adds a product term with coefficient c (1, 2 or 3) mod 4 into the digit pair (ones, twos) of its
// bit. atomicXor returns the old bit of `ones`, so the carry into `twos` is exact whatever order
// threads add in.
struct Mod4Emit {
    const u32 *g, *xi;
    u32 *ones, *twos;
    u64 row_word;
    u32 col, width, num_limbs;
    __device__ __forceinline__ void operator()(u64 t, u32 c) {
        u64 bit = (u64)col + seqno(g, xi, t, width);
        u64 limb = bit / 32;
        if (limb >= (u64)num_limbs) return;
        u64 w = row_word + limb;
        u32 mask = 1u << (u32)(bit % 32);
        u32 carry = 0;
        if (c & 1u) carry = (atomicXor(&ones[w], mask) & mask) ? 1u : 0u;
        if (((c >> 1) ^ carry) & 1u) atomicXor(&twos[w], mask);
    }
};

// XORs a Y term into its vector, which starts at bit `col` of the row in `ones`.
struct Mod2Emit {
    const u32 *g, *xi;
    u32 *ones;
    u64 row_word;
    u32 col, width, num_limbs;
    __device__ __forceinline__ void operator()(u64 t, u32) {
        u64 bit = (u64)col + seqno(g, xi, t, width);
        u64 limb = bit / 32;
        if (limb >= (u64)num_limbs) return;
        atomicXor(&ones[row_word + limb], 1u << (u32)(bit % 32));
    }
};

// sigma(R, S) of each item, R and S of total degree `item_degree`, into row `item_row`: the product
// terms over Z/4 into (ones, twos) from bit `item_col`, and (PART_Y) the Y term
// P(R - 2^k (e_m + e_n)) P(S - e_{k+1}) mod 2, for each k and 0 <= m < n (e_0 = 0), into `ones`
// from bit item_col + yoff(D, m + k, n + k).
extern "C" __global__ void sigma_pair(const u64 *__restrict__ item_r, const u64 *__restrict__ item_s,
                                      const int *__restrict__ item_degree,
                                      const u32 *__restrict__ item_row,
                                      const u32 *__restrict__ item_col, u32 num_items,
                                      const u32 *__restrict__ g, const u32 *__restrict__ xi,
                                      u32 width, const u32 *__restrict__ yoff, u32 parts,
                                      u32 *ones, u32 *twos, u32 num_limbs) {
    u32 item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= num_items) return;
    const u64 r = item_r[item], s = item_s[item];
    const int degree = item_degree[item];
    const u64 row_word = (u64)item_row[item] * num_limbs;
    const u32 col = item_col[item];

    if (parts & PART_PRODUCT) {
        Mod4Emit emit{g, xi, ones, twos, row_word, col, width, num_limbs};
        walk<M4>(r, s, degree, emit);
    }
    if (!(parts & PART_Y)) return;
    const u32 rows = pp_len(r), cols = pp_len(s);
    for (u32 k = 0; k < cols; ++k) {
        u32 sk = pp_get(s, k);
        if (sk == 0) continue;
        u64 s1 = pp_set(s, k, sk - 1);
        u32 step = 1u << k;
        for (u32 n = 1; n <= rows; ++n) {
            u32 rn = pp_get(r, n - 1);
            if (rn < step) continue;
            u64 r1 = pp_set(r, n - 1, rn - step);
            for (u32 m = 0; m < n; ++m) {
                u64 r2 = r1;
                if (m > 0) {
                    u32 rm = pp_get(r1, m - 1);
                    if (rm < step) continue;
                    r2 = pp_set(r1, m - 1, rm - step);
                }
                if (n + k >= Y_STRIDE) continue;
                u32 off = yoff[((u64)degree * Y_STRIDE + m + k) * Y_STRIDE + n + k];
                if (off == Y_ABSENT) continue;
                int y_degree = degree + 1 - (1 << (n + k)) - (1 << (m + k));
                Mod2Emit emit{g, xi, ones, row_word, col + off, width, num_limbs};
                walk<M2>(r2, s1, y_degree, emit);
            }
        }
    }
}

// THE PERSISTENT SIGMA KERNEL. Each lane works through the SUB-PRODUCTS of its pair: the product
// over Z/4 (if PART_PRODUCT), then each Y product mod 2 (if PART_Y), in the order of `sigma_pair`.
// A lane that finishes a pair takes the next from a global counter, and every trip of the loop is
// one step of `multiply.cu`'s walk for every live lane, so the lanes of a warp stay in step however
// unevenly their pairs' work falls.
//
// The walk state is u16 in shared memory with stride NT, as in `antidiag_body`, plus FLG per cell:
// the carry and parity bits of the mod-4 coefficient before the cell (zero mod 2). Terms are queued
// per warp with their target word's offset and a code: SIG_Y for a Y term, else the coefficient
// mod 4, and emitted 32 at a time.

// Bytes of shared memory the sigma walk state of a block takes (16-byte aligned); the queue
// follows. Mirrored by `sigma_shared_bytes` in `mod.rs`.
#define SIG_STATE_U16(N) (3 * ((N) * ((N) - 1) / 2) + 2 * ((N) + 1))
#define SIG_STATE_BYTES(N) (((SIG_STATE_U16(N) * 2u * NT) + 15u) & ~15u)

// The code of a Y term in the queue: plain XOR into `ones`.
#define SIG_Y 4u

// Carry and parity bits of a packed mod-4 coefficient.
#define F_CARRIED 1u
#define F_ODD 2u

// `st_add` on a state unpacked as (sum, flags).
__device__ __forceinline__ bool sig_add(bool mod4, u32 &sum, u32 &flags, u32 v) {
    if (!mod4) {
        if (sum & v) return false;
        sum |= v;
        return true;
    }
    M4 st = {sum, flags & F_CARRIED, (flags & F_ODD) ? 1u : 0u};
    if (!st_add(st, v)) return false;
    sum = st.sum;
    flags = (st.carried ? F_CARRIED : 0u) | (st.odd ? F_ODD : 0u);
    return true;
}

// The coefficient of a completed walk with these flags, or SIG_Y mod 2.
__device__ __forceinline__ u32 sig_code(bool mod4, u32 flags) {
    if (!mod4) return SIG_Y;
    return (flags & F_CARRIED) ? 2u : ((flags & F_ODD) ? 3u : 1u);
}

// Emits one queued term: XOR for a Y term, else add its coefficient mod 4 as `Mod4Emit` does.
__device__ __forceinline__ void sig_emit(const u32 *__restrict__ g, const u32 *__restrict__ xi,
                                         u32 *ones, u32 *twos, u64 t, u32 row_word, u32 col,
                                         u32 code, u32 width, u32 num_limbs) {
    u64 bit = (u64)col + seqno(g, xi, t, width);
    u64 limb = bit / 32;
    if (limb >= (u64)num_limbs) return;
    u64 w = (u64)row_word + limb;
    u32 mask = 1u << (u32)(bit % 32);
    if (code == SIG_Y) {
        atomicXor(&ones[w], mask);
        return;
    }
    u32 carry = 0;
    if (code & 1u) carry = (atomicXor(&ones[w], mask) & mask) ? 1u : 0u;
    if (((code >> 1) ^ carry) & 1u) atomicXor(&twos[w], mask);
}

template <int N>
__device__ __forceinline__ void sigma_body(const u64 *__restrict__ item_r,
                                           const u64 *__restrict__ item_s,
                                           const int *__restrict__ item_degree,
                                           const u32 *__restrict__ item_row,
                                           const u32 *__restrict__ item_col, u32 num_items,
                                           u32 *counter, const u32 *__restrict__ g,
                                           const u32 *__restrict__ xi, u32 width,
                                           const u32 *__restrict__ yoff, u32 parts, u32 *ones,
                                           u32 *twos, u32 num_limbs) {
    constexpr u32 CELLS = N * (N - 1) / 2;
    u16 *base = smem + threadIdx.x;
#define CELL(d, i) (((d) - 2) * ((d) - 1) / 2 + (i) - 1)
#define VAL(c) base[(c) * NT]
#define BEF(c) base[(CELLS + (c)) * NT]
#define FLG(c) base[(2 * CELLS + (c)) * NT]
#define RL(k) base[(3 * CELLS + (k)) * NT]
#define SL(k) base[(3 * CELLS + N + 1 + (k)) * NT]
#define LO(d) max(1, (int)(d) - (int)cols)
#define HI(d) min((int)rows, (int)(d) - 1)

    const u32 lane = threadIdx.x & 31u;
    const u32 nwarps = NT >> 5;
    unsigned char *qbase = (unsigned char *)smem + SIG_STATE_BYTES(N);
    u64 *wq_t = (u64 *)qbase + (threadIdx.x >> 5) * QCAP;
    u32 *wq_rw = (u32 *)(qbase + nwarps * QCAP * 8u) + (threadIdx.x >> 5) * QCAP;
    u32 *wq_cc = (u32 *)(qbase + nwarps * QCAP * 12u) + (threadIdx.x >> 5) * QCAP;
    volatile u32 *wq_cnt = (u32 *)(qbase + nwarps * QCAP * 16u) + (threadIdx.x >> 5);
    if (lane == 0) *wq_cnt = 0;
    __syncwarp();
    // The code rides in the top 3 bits of the column, which the host keeps below 2^29.
#define QUEUE(t_, code_)                                                                          \
    {                                                                                             \
        u32 s_ = atomicAdd((u32 *)wq_cnt, 1u);                                                    \
        wq_t[s_] = (t_);                                                                          \
        wq_rw[s_] = row_word;                                                                     \
        wq_cc[s_] = walk_col | ((code_) << 29);                                                   \
    }

    u32 item = atomicAdd(counter, 1u);
    // The pair: its factors, degree, row and column, and the next sub-product: -1 the product,
    // else the Y product (yk, yn, ym), or done.
    bool have_pair = false;
    u64 pr = 0, ps = 0;
    int pdeg = 0;
    u32 row_word = 0, pcol = 0, prow_len = 0, pcol_len = 0;
    int yk = 0, yn = 1, ym = 0;
    bool product_next = false;
    // The walk.
    bool live = false, back = false, mod4 = false;
    int d = 0, i = 0, max_diag = 0;
    u32 rows = 0, cols = 0, tally = 0, flags = 0, walk_col = 0;
    u64 t_bits = 0;

    for (;;) {
        if (!live && !have_pair && item < num_items) {
            pr = item_r[item];
            ps = item_s[item];
            pdeg = item_degree[item];
            row_word = item_row[item] * num_limbs;
            pcol = item_col[item];
            prow_len = pp_len(pr);
            pcol_len = pp_len(ps);
            product_next = (parts & PART_PRODUCT) != 0;
            yk = 0;
            yn = 1;
            ym = 0;
            if (!(parts & PART_Y)) yk = (int)pcol_len;
            have_pair = true;
            item = atomicAdd(counter, 1u);
        }
        if (!__any_sync(0xffffffffu, live || have_pair)) break;
        if (!live && have_pair) {
            // ---- START the next sub-product of the pair, if any.
            u64 r = 0, s = 0;
            int deg = 0;
            bool found = false;
            if (product_next) {
                product_next = false;
                r = pr;
                s = ps;
                deg = pdeg;
                walk_col = pcol;
                mod4 = true;
                found = true;
            }
            while (!found && yk < (int)pcol_len) {
                int k = yk, n = yn, m = ym;
                // Advance the cursor past (k, n, m) first.
                if (++ym >= yn) {
                    ym = 0;
                    if (++yn > (int)prow_len) {
                        yn = 1;
                        ++yk;
                    }
                }
                u32 sk = pp_get(ps, k);
                if (sk == 0) {
                    yk = k + 1;
                    yn = 1;
                    ym = 0;
                    continue;
                }
                u32 step = 1u << k;
                u32 rn = pp_get(pr, n - 1);
                if (rn < step) {
                    yn = n + 1;
                    ym = 0;
                    if (yn > (int)prow_len) {
                        yn = 1;
                        yk = k + 1;
                    }
                    continue;
                }
                u64 r1 = pp_set(pr, n - 1, rn - step);
                if (m > 0) {
                    u32 rm = pp_get(r1, m - 1);
                    if (rm < step) continue;
                    r1 = pp_set(r1, m - 1, rm - step);
                }
                if (n + k >= Y_STRIDE) continue;
                u32 off = yoff[((u64)pdeg * Y_STRIDE + m + k) * Y_STRIDE + n + k];
                if (off == Y_ABSENT) continue;
                r = r1;
                s = pp_set(ps, k, sk - 1);
                deg = pdeg + 1 - (1 << (n + k)) - (1 << (m + k));
                walk_col = pcol + off;
                mod4 = false;
                found = true;
            }
            if (!found) {
                have_pair = false;
            } else {
                // ---- INIT the walk of P(r) P(s).
                rows = 0;
                cols = 0;
#pragma unroll
                for (u32 k = 0; k < N; ++k) {
                    u32 rv = pp_get(r, k), sv = pp_get(s, k);
                    RL(k + 1) = (u16)rv;
                    SL(k + 1) = (u16)sv;
                    if (rv) rows = k + 1;
                    if (sv) cols = k + 1;
                }
                if (rows == 0 || cols == 0) {
                    QUEUE(rows == 0 ? s : r, mod4 ? 1u : SIG_Y);
                } else {
                    int md = min((int)(rows + cols), N);
                    while (md > 1 && (1 << md) - 1 > deg) --md;
                    max_diag = md;
                    u32 sum = RL(md), fl = 0;
                    if (sig_add(mod4, sum, fl, SL(md))) {
                        if (md == 1) {
                            QUEUE((u64)sum << PP_SHIFT[0], sig_code(mod4, fl));
                        } else {
                            d = md;
                            i = LO(d);
                            tally = sum;
                            flags = fl;
                            t_bits = 0;
                            back = false;
                            live = true;
                        }
                    }
                }
            }
        } else if (live) {
            // ---- STEP
            int j = d - i;
            u32 c = CELL(d, i);
            bool adv = false;
            u32 used = 0, ufl = 0;
            if (!back) {
                // ENTER: value 0 changes nothing.
                BEF(c) = (u16)tally;
                FLG(c) = (u16)flags;
                VAL(c) = 0;
                used = tally;
                ufl = flags;
                adv = true;
            } else {
                // BACKTRACK: the next value of this cell the coefficient allows, if it still fits
                // what is left of row i and column j; else give the cell up.
                u32 v = VAL(c);
                u32 mask = BEF(c);
                u32 bfl = FLG(c);
                u64 next;
                if (mod4) {
                    M4 st = {mask, bfl & F_CARRIED, (bfl & F_ODD) ? 1u : 0u};
                    next = st_next(st, v);
                } else {
                    next = ((v | mask) + 1u) & ~mask;
                }
                u32 rli = RL(i), slj = SL(j);
                u64 inc = next - v;
                if (next <= 0xffffu && (inc << j) <= rli && inc <= slj) {
                    RL(i) = (u16)(rli - ((u32)inc << j));
                    SL(j) = (u16)(slj - (u32)inc);
                    VAL(c) = (u16)next;
                    used = mask;
                    ufl = bfl;
                    sig_add(mod4, used, ufl, (u32)next);
                    adv = true;
                    back = false;
                } else {
                    RL(i) = (u16)(rli + (v << j));
                    SL(j) = (u16)(slj + v);
                    if (i > LO(d)) {
                        --i;
                    } else if (d < max_diag) {
                        ++d;
                        i = HI(d);
                    } else {
                        live = false;
                    }
                }
            }
            if (adv) {
                if (i < HI(d)) {
                    ++i;
                    tally = used;
                    flags = ufl;
                } else {
                    // Diagonal d is complete: T_d = used, and the slacks of d - 1 are known.
                    u32 sh = PP_SHIFT[d - 1];
                    t_bits = (t_bits & ~((u64)PP_MASK[d - 1] << sh)) | ((u64)used << sh);
                    u32 sum = RL(d - 1), fl = ufl;
                    if (sig_add(mod4, sum, fl, SL(d - 1))) {
                        if (d == 2) {
                            QUEUE(t_bits | ((u64)sum << PP_SHIFT[0]), sig_code(mod4, fl));
                            back = true;
                        } else {
                            --d;
                            i = LO(d);
                            tally = sum;
                            flags = fl;
                        }
                    } else {
                        back = true;
                    }
                }
            }
        }
        // Each trip queues at most one term per lane, so the queue holds at most 63 + 32 < QCAP
        // after a trip and is drained back below 32 here.
        __syncwarp();
        u32 cnt = *wq_cnt;
        if (cnt >= 32u) {
            u32 cc = wq_cc[lane];
            sig_emit(g, xi, ones, twos, wq_t[lane], wq_rw[lane], cc & 0x1fffffffu, cc >> 29, width,
                     num_limbs);
            u32 rest = cnt - 32u;
            bool mv = lane < rest;
            u64 tt = 0;
            u32 rw = 0, c2 = 0;
            if (mv) {
                tt = wq_t[32u + lane];
                rw = wq_rw[32u + lane];
                c2 = wq_cc[32u + lane];
            }
            __syncwarp();
            if (mv) {
                wq_t[lane] = tt;
                wq_rw[lane] = rw;
                wq_cc[lane] = c2;
            }
            if (lane == 0) *wq_cnt = rest;
            __syncwarp();
        }
    }
    __syncwarp();
    {
        u32 cnt = *wq_cnt;
        if (lane < cnt) {
            u32 cc = wq_cc[lane];
            sig_emit(g, xi, ones, twos, wq_t[lane], wq_rw[lane], cc & 0x1fffffffu, cc >> 29, width,
                     num_limbs);
        }
    }
#undef QUEUE
#undef CELL
#undef VAL
#undef BEF
#undef FLG
#undef RL
#undef SL
#undef LO
#undef HI
}

// One entry point per walk width N, as for `multiply_antidiag_N`.
#define SIGMA_ENTRY(NN)                                                                           \
    extern "C" __global__ void __launch_bounds__(NT) sigma_step_##NN(                             \
        const u64 *__restrict__ item_r, const u64 *__restrict__ item_s,                            \
        const int *__restrict__ item_degree, const u32 *__restrict__ item_row,                     \
        const u32 *__restrict__ item_col, u32 num_items, u32 *counter, const u32 *__restrict__ g,  \
        const u32 *__restrict__ xi, u32 width, const u32 *__restrict__ yoff, u32 parts, u32 *ones, \
        u32 *twos, u32 num_limbs) {                                                                \
        sigma_body<NN>(item_r, item_s, item_degree, item_row, item_col, num_items, counter, g, xi, \
                       width, yoff, parts, ones, twos, num_limbs);                                 \
    }

SIGMA_ENTRY(6)
SIGMA_ENTRY(7)
SIGMA_ENTRY(8)
SIGMA_ENTRY(9)
SIGMA_ENTRY(10)
