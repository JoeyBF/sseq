//! Chevalley–Eilenberg cohomology of a finite-dimensional Lie algebra over `F_p`.
//!
//! Given the Morava Lie algebra `L(n, n)` (see [`crate::lie::morava_lie`]), this computes
//! `H^*(L(n, n); F_p)` with trivial coefficients by building the Koszul/Chevalley–Eilenberg complex
//! `C^k = /\^k(L^*)` and ranking the differentials over `F_p`. The total dimension
//!
//! ```text
//! dim_{F_p} H^*(L(n, n)) = 2^{n^2} - 2 * sum_k rank(d_k)
//! ```
//!
//! is grading-independent, so it can be checked against the Salch validation table
//! (`2, 12, 152, 3440, ...`; see `ext/docs/chromatic-computations.md` §3) *without* first solving
//! the graded-chart problem.
//!
//! # The differential
//!
//! `d : /\^k(L^*) -> /\^{k+1}(L^*)` is the derivation extending, on generators,
//! `d(x_a^*) = - sum_{b<c} c^a_{bc} x_b^* /\ x_c^*`, where `[x_b, x_c] = sum_a c^a_{bc} x_a`. On a
//! basis monomial `e_{a_0} /\ ... /\ e_{a_{k-1}}` (indices ascending),
//!
//! ```text
//! d(e_{a_0} /\ ... /\ e_{a_{k-1}}) = sum_r (-1)^r e_{a_0} /\ ... /\ d(e_{a_r}) /\ ... /\ e_{a_{k-1}}.
//! ```
//!
//! # Block decomposition and scaling
//!
//! The `i`-weight of [`MoravaLie`] is additive under the bracket, hence preserved by `d`. So the
//! complex splits as a direct sum over total `i`-weight `w`, and within each weight over
//! cohomological degree `k`; we rank each `(w, k)` block independently. To scale past `n = 4` (whose
//! full complex has dimension `2^16`) we never materialize the whole complex:
//!
//! * **Streaming enumeration.** The monomials of a fixed weight `w` are generated on demand by a
//!   pruned depth-first subset walk ([`enumerate_weight`]) — we hold one weight's block in memory at
//!   a time, not all `2^{n^2}` monomials.
//! * **Sparse rank.** Each differential column has only `O(k * bracket size)` nonzeros, so we rank
//!   with incremental sparse row reduction over `F_p` ([`sparse_rank`]) rather than a dense matrix.
//!   The codomain is *not* enumerated separately: a target monomial's bitmask is used directly as a
//!   column identifier.
//!
//! This lets `n = 4` run in well under a second, and makes the corners and small/large weights of
//! `n = 5` (whose middle blocks reach `~565k` — see [`cohomology_by_weight`]) reachable with a size
//! cap. Full `n = 5` still needs either the `Z/n`-equivariant splitting or structured/black-box
//! sparse rank; see the handoff §3.

use std::collections::{BTreeMap, HashMap};

use fp::prime::Prime;
use maybe_rayon::prelude::*;

use crate::lie::morava_lie::MoravaLie;

/// The result of a (complete) Chevalley–Eilenberg cohomology computation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CohomologyStats {
    /// The prime `p`.
    pub prime: u32,
    /// The Lie-algebra dimension `N` (`= n^2` for `L(n, n)`).
    pub dim: usize,
    /// `total_dim = dim_{F_p} H^*`, the sum of all Betti numbers.
    pub total_dim: usize,
    /// `ranks[k] = rank(d_k)`, `d_k : C^k -> C^{k+1}`, for `k in 0..=N`.
    pub ranks: Vec<usize>,
    /// `betti[k] = dim_{F_p} H^k`, for `k in 0..=N`.
    pub betti: Vec<usize>,
}

/// Per-weight rank data, the streaming unit of the computation. One [`WeightReport`] summarizes the
/// contribution of the total-`i`-weight-`w` summand of the complex.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightReport {
    /// The total `i`-weight `w` of this summand.
    pub weight: u32,
    /// `ranks[k] = rank(d_k)` restricted to weight `w`, for `k in 0..=N`. All-zero entries are kept
    /// so the vector can be summed across weights positionally.
    pub ranks: Vec<usize>,
    /// `dim C^k` restricted to weight `w` (the block sizes), for `k in 0..=N`.
    pub block_dims: Vec<usize>,
    /// Whether every block of this weight was ranked (`false` if some exceeded the size cap).
    pub complete: bool,
    /// The largest block (domain size) encountered in this weight.
    pub max_block: usize,
}

/// Options controlling the streaming computation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// If set, skip any `(w, k)` block whose domain size exceeds this cap (recording the weight as
    /// incomplete). `None` ranks everything, however large.
    pub max_block: Option<usize>,
}

/// Computes the Chevalley–Eilenberg cohomology of `lie` over `F_p`, ranking every block.
///
/// This is the complete computation used by the validation ladder; it is fast through `n = 4`. For
/// `n = 5` prefer [`cohomology_by_weight`] with a `max_block` cap, since some middle blocks are too
/// large to rank densely-in-memory here.
pub fn chevalley_eilenberg_cohomology(lie: &MoravaLie) -> CohomologyStats {
    let reports = cohomology_by_weight(lie, Options::default());
    debug_assert!(reports.iter().all(|r| r.complete));
    let n_dim = lie.dim();
    let mut ranks = vec![0usize; n_dim + 1];
    for r in &reports {
        for (k, &rk) in r.ranks.iter().enumerate() {
            ranks[k] += rk;
        }
    }
    let total: usize = 1usize << n_dim;
    let total_rank: usize = ranks.iter().sum();
    let total_dim = total - 2 * total_rank;

    // dim C^k = binomial(N, k).
    let dim_ck = binomials(n_dim);
    let mut betti = vec![0usize; n_dim + 1];
    for k in 0..=n_dim {
        let prev = if k == 0 { 0 } else { ranks[k - 1] };
        betti[k] = dim_ck[k] - ranks[k] - prev;
    }

    CohomologyStats {
        prime: lie.prime().as_u32(),
        dim: n_dim,
        total_dim,
        ranks,
        betti,
    }
}

/// Streams the computation weight by weight, returning a [`WeightReport`] per total `i`-weight.
///
/// Only one weight's blocks are held in memory at a time. With `opts.max_block` set, blocks larger
/// than the cap are skipped and their weight marked incomplete, which is how partial `n = 5` data is
/// gathered (compute the reachable weights/corners, leave the `~565k` middle blocks out).
pub fn cohomology_by_weight(lie: &MoravaLie, opts: Options) -> Vec<WeightReport> {
    let n_dim = lie.dim();
    let p = lie.prime().as_u32();
    let cobracket = lie.cobracket();
    let weights: Vec<u32> = (0..n_dim).map(|a| lie.weight(a)).collect();
    let max_weight: u32 = weights.iter().sum();

    // Pass 1 (sequential): enumerate each weight, record its metadata, and collect the individual
    // `(weight, k)` blocks that are within the size cap as independent ranking tasks. Blocks over the
    // cap are never stored. Only one weight's monomials are materialized at a time.
    let mut reports: Vec<WeightReport> = Vec::new();
    let mut tasks: Vec<(usize, usize, Vec<u32>)> = Vec::new(); // (report index, k, domain masks)
    for w in 0..=max_weight {
        let mut by_k: Vec<Vec<u32>> = vec![Vec::new(); n_dim + 1];
        enumerate_weight(&weights, w, &mut |mask| {
            by_k[mask.count_ones() as usize].push(mask);
        });
        if by_k.iter().all(|b| b.is_empty()) {
            continue;
        }
        let block_dims: Vec<usize> = by_k.iter().map(|b| b.len()).collect();
        let max_block = block_dims.iter().copied().max().unwrap_or(0);
        let report_idx = reports.len();
        let mut complete = true;

        for k in 0..n_dim {
            if by_k[k].is_empty() || by_k[k + 1].is_empty() {
                continue; // d_k has rank 0
            }
            if opts
                .max_block
                .is_some_and(|cap| by_k[k].len() > cap || by_k[k + 1].len() > cap)
            {
                complete = false;
                continue;
            }
            tasks.push((report_idx, k, std::mem::take(&mut by_k[k])));
        }

        reports.push(WeightReport {
            weight: w,
            ranks: vec![0usize; n_dim + 1],
            block_dims,
            complete,
            max_block,
        });
    }

    // Pass 2 (parallel): rank the independent blocks. Blocks are independent summands, so this is
    // embarrassingly parallel; `cobracket` is shared read-only.
    let cobracket_ref = &cobracket;
    let ranked: Vec<(usize, usize, usize)> = tasks
        .into_maybe_par_iter()
        .map(|(report_idx, k, masks)| (report_idx, k, block_rank(&masks, cobracket_ref, p)))
        .collect();
    for (report_idx, k, rank) in ranked {
        reports[report_idx].ranks[k] = rank;
    }
    reports
}

/// A location in the trigraded chart of `H^*(L(n,n))`, in the order the spectral-sequence display
/// uses: stem `n = t − s`, filtration `s`, and `i`-weight `w` (all preserved by the differential
/// except `s`, which it raises by 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChartDegree {
    /// Stem `n = t − s` (the `x`-axis of the standard `(n, s)` chart).
    pub n: i64,
    /// Cohomological / May-filtration degree `s` (the `y`-axis).
    pub s: usize,
    /// `i`-weight `w = Σ i` (the natural third grading).
    pub weight: u32,
}

/// The **trigraded** cohomology `dim H^{n,s,w}(L(n,n))`, keyed as `(stem n = t − s, s, i-weight w)`
/// — the order in which bigraded spectral sequences are displayed (`n` on the `x`-axis, `s` on the
/// `y`-axis), with the `i`-weight as the third grading.
///
/// The internal degree `t ∈ Z/(p^n − 1)` is [`MoravaLie::internal_degree`]; `n = t − s`. The
/// differential preserves both `t` and the `i`-weight while raising `s`, so the complex splits into
/// `(weight, t, s)` blocks and `H^{n,s,w} = dim C − rank(d_s) − rank(d_{s−1})` within each `(w, t)`
/// column. We parallelize over `i`-weight (each weight ranks all of its small `(t, s)` blocks), which
/// keeps memory low and avoids per-block task overhead.
pub fn trigraded_cohomology(lie: &MoravaLie) -> BTreeMap<ChartDegree, usize> {
    let n_dim = lie.dim();
    let p = lie.prime().as_u32();
    let modulus = lie.internal_modulus();
    let cobracket = lie.cobracket();
    let weights: Vec<u32> = (0..n_dim).map(|a| lie.weight(a)).collect();
    let int_deg: Vec<u64> = (0..n_dim).map(|a| lie.internal_degree(a)).collect();
    let max_weight: u32 = weights.iter().sum();
    let cobracket_ref = &cobracket;
    let weights_ref = &weights;
    let int_deg_ref = &int_deg;

    // Phase 1 (parallel over i-weight): enumerate each weight and split it into `(weight, t, s)`
    // blocks. `dim C^{s,t,w}` is read off directly. Weights are independent (d is block-diagonal in
    // weight), and this holds only one weight's monomials per task.
    let per_weight: Vec<(Vec<((u32, u64, usize), Vec<u32>)>, Vec<((u32, u64, usize), usize)>)> =
        (0..=max_weight)
            .collect::<Vec<_>>()
            .into_maybe_par_iter()
            .map(|w| {
                let mut by_ts: HashMap<(u64, usize), Vec<u32>> = HashMap::new();
                enumerate_weight(weights_ref, w, &mut |mask| {
                    let s = mask.count_ones() as usize;
                    let mut t = 0u64;
                    let mut bits = mask;
                    while bits != 0 {
                        let a = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        t = (t + int_deg_ref[a]) % modulus;
                    }
                    by_ts.entry((t, s)).or_default().push(mask);
                });
                let mut blocks = Vec::with_capacity(by_ts.len());
                let mut cdims = Vec::with_capacity(by_ts.len());
                for ((t, s), masks) in by_ts {
                    cdims.push(((w, t, s), masks.len()));
                    blocks.push(((w, t, s), masks));
                }
                (blocks, cdims)
            })
            .collect();

    // Flatten. `cdim` keeps dim C^{s,t,w}; `blocks` are the ranking tasks.
    let mut cdim: HashMap<(u32, u64, usize), usize> = HashMap::new();
    let mut blocks: Vec<((u32, u64, usize), Vec<u32>)> = Vec::new();
    for (bs, cs) in per_weight {
        for (key, d) in cs {
            cdim.insert(key, d);
        }
        blocks.extend(bs);
    }

    // Phase 2 (flat parallel over all blocks): rank each — balanced regardless of weight size.
    let ranked: Vec<((u32, u64, usize), usize)> = blocks
        .into_maybe_par_iter()
        .map(|(key, masks)| (key, block_rank(&masks, cobracket_ref, p)))
        .collect();
    let rank: HashMap<(u32, u64, usize), usize> = ranked.into_iter().collect();

    // H^{s,t,w} = dim C − rank(d_s) − rank(d_{s-1}); key by (stem n = t − s, s, weight).
    let mut result = BTreeMap::new();
    for (&(w, t, s), &cd) in &cdim {
        let r_s = *rank.get(&(w, t, s)).unwrap_or(&0);
        let r_prev = if s == 0 {
            0
        } else {
            *rank.get(&(w, t, s - 1)).unwrap_or(&0)
        };
        let h = cd - r_s - r_prev;
        if h > 0 {
            result.insert(
                ChartDegree {
                    n: t as i64 - s as i64,
                    s,
                    weight: w,
                },
                h,
            );
        }
    }
    result
}

/// The **bigraded** cohomology `dim H^{s,t}(L(n,n))` (summed over the `i`-weight), keyed `(s, t)`.
/// `Σ_t H^{s,t}` equals the `s`-graded Betti number, and `Σ_{s,t}` the total dimension.
pub fn bigraded_cohomology(lie: &MoravaLie) -> BTreeMap<(usize, u64), usize> {
    let modulus = lie.internal_modulus();
    let mut result: BTreeMap<(usize, u64), usize> = BTreeMap::new();
    for (deg, d) in trigraded_cohomology(lie) {
        // t = n + s (mod p^n − 1); n was defined as t − s from a representative t ∈ [0, modulus).
        let t = (deg.n + deg.s as i64).rem_euclid(modulus as i64) as u64;
        *result.entry((deg.s, t)).or_insert(0) += d;
    }
    result
}

/// The Chevalley–Eilenberg differential of the basis monomial `mask`, as a list of
/// `(target_mask, coeff)` terms with `coeff` reduced to `1..p`. Duplicate targets are merged.
pub fn differential_terms(
    mask: u32,
    cobracket: &[Vec<(usize, usize, i32)>],
    p: u32,
) -> Vec<(u32, u32)> {
    let mut acc: HashMap<u32, i32> = HashMap::new();
    let mut bits = mask;
    while bits != 0 {
        let a = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let r = (mask & ((1u32 << a) - 1)).count_ones();
        let koszul = if r.is_multiple_of(2) { 1i32 } else { -1i32 };
        for &(b, c, coeff) in &cobracket[a] {
            let bit_b = 1u32 << b;
            let bit_c = 1u32 << c;
            if mask & bit_b != 0 || mask & bit_c != 0 {
                continue; // repeated generator wedges to zero
            }
            let target = (mask & !(1u32 << a)) | bit_b | bit_c;
            let sort_sign = wedge_sort_sign(mask, a, b, c);
            *acc.entry(target).or_insert(0) += -koszul * coeff * sort_sign;
        }
    }
    acc.into_iter()
        .filter_map(|(t, v)| {
            let v = v.rem_euclid(p as i32) as u32;
            (v != 0).then_some((t, v))
        })
        .collect()
}

/// The sign (`+1`/`-1`) sorting the wedge `[a_0, ..., a_{r-1}, b, c, a_{r+1}, ...]` — the set bits of
/// `mask` with `a = a_r` replaced by `(b, c)` (`b < c`) — into ascending order. Callers guarantee
/// `b, c` are not already in `mask`.
fn wedge_sort_sign(mask: u32, a: usize, b: usize, c: usize) -> i32 {
    let mut seq: Vec<usize> = Vec::with_capacity(mask.count_ones() as usize + 1);
    let mut bits = mask;
    while bits != 0 {
        let x = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        if x == a {
            seq.push(b);
            seq.push(c);
        } else {
            seq.push(x);
        }
    }
    let mut inversions = 0usize;
    for i in 0..seq.len() {
        for j in (i + 1)..seq.len() {
            if seq[i] > seq[j] {
                inversions += 1;
            }
        }
    }
    if inversions.is_multiple_of(2) { 1 } else { -1 }
}

/// Enumerates every subset (bitmask) of `{0..weights.len()}` whose total `weight` equals `target`,
/// calling `emit` on each. A pruned DFS: we descend only while the running weight can still reach
/// `target` using the remaining generators.
pub(crate) fn enumerate_weight(weights: &[u32], target: u32, emit: &mut impl FnMut(u32)) {
    let n = weights.len();
    // suffix_max[i] = sum of weights[i..]; the most weight the remaining generators can add.
    let mut suffix_max = vec![0u32; n + 1];
    for i in (0..n).rev() {
        suffix_max[i] = suffix_max[i + 1] + weights[i];
    }
    fn dfs(
        idx: usize,
        cur_mask: u32,
        cur_weight: u32,
        target: u32,
        weights: &[u32],
        suffix_max: &[u32],
        emit: &mut impl FnMut(u32),
    ) {
        if cur_weight == target {
            // No remaining generator has weight 0, so the only completion is the empty tail.
            emit(cur_mask);
            return;
        }
        if idx == weights.len() || cur_weight > target {
            return;
        }
        // Prune: even taking everything left cannot reach the target.
        if cur_weight + suffix_max[idx] < target {
            return;
        }
        // Include generator `idx`.
        dfs(
            idx + 1,
            cur_mask | (1u32 << idx),
            cur_weight + weights[idx],
            target,
            weights,
            suffix_max,
            emit,
        );
        // Exclude generator `idx`.
        dfs(
            idx + 1,
            cur_mask,
            cur_weight,
            target,
            weights,
            suffix_max,
            emit,
        );
    }
    dfs(0, 0, 0, target, weights, &suffix_max, emit);
}

/// The rank over `F_p` of one `(weight, degree)` differential block, whose domain is the monomial
/// list `masks` and whose rows are `differential_terms(mask, ..)`. This is the hot path.
fn block_rank(masks: &[u32], cobracket: &[Vec<(usize, usize, i32)>], p: u32) -> usize {
    sparse_rank(
        masks.iter().map(|&mask| differential_terms(mask, cobracket, p)),
        p,
    )
}

/// The rank over `F_p` of the matrix whose rows are the given sparse vectors. Each row is a list of
/// `(column_id, coeff)` pairs; column identifiers are arbitrary `u32`s (we use target bitmasks).
///
/// Incremental sparse row-echelon reduction. Rows and pivots are stored as `(column, coeff)` pairs
/// sorted by column ascending; a pivot's leading column is its smallest and carries coefficient `1`,
/// so eliminating it only touches columns `>=` that lead — a working row's leading column strictly
/// increases and the reduction terminates. Each step is a linear merge ([`axpy`]) — cache-friendly
/// and free of tree/hash-map per-node overhead, which is what keeps the raw sparse differential
/// blocks tractable.
pub fn sparse_rank(rows: impl IntoIterator<Item = Vec<(u32, u32)>>, p: u32) -> usize {
    // pivots[lead] = a normalized echelon row (leading coeff 1 at column `lead`).
    let mut pivots: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut rank = 0usize;
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for row in rows {
        let mut work = canonicalize_row(row, p);
        loop {
            let Some(&(lead, lead_coeff)) = work.first() else {
                break; // reduced to zero: linearly dependent
            };
            match pivots.get(&lead) {
                Some(pivot) => {
                    // work <- work - lead_coeff * pivot (both sorted; leading column cancels).
                    axpy(&work, pivot, lead_coeff, p, &mut merged);
                    std::mem::swap(&mut work, &mut merged);
                }
                None => {
                    // New pivot: normalize the leading coefficient to 1.
                    let inv = mod_inverse(lead_coeff, p);
                    if inv != 1 {
                        for e in work.iter_mut() {
                            e.1 = (e.1 * inv) % p;
                        }
                    }
                    pivots.insert(lead, work);
                    rank += 1;
                    break;
                }
            }
        }
    }
    rank
}

/// Reduces a row mod `p`, sorts by column, merges duplicate columns, and drops zeros.
fn canonicalize_row(mut row: Vec<(u32, u32)>, p: u32) -> Vec<(u32, u32)> {
    for e in row.iter_mut() {
        e.1 %= p;
    }
    row.sort_unstable_by_key(|&(c, _)| c);
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(row.len());
    for (c, v) in row {
        if v == 0 {
            continue;
        }
        if let Some(last) = out.last_mut()
            && last.0 == c
        {
            last.1 = (last.1 + v) % p;
            if last.1 == 0 {
                out.pop();
            }
            continue;
        }
        out.push((c, v));
    }
    out
}

/// Computes `work - factor * pivot` over `F_p` into `out` (cleared first). Both inputs are sorted by
/// column ascending; the result is sorted, with zeros dropped. `pivot`'s leading column equals
/// `work`'s and carries coefficient `1`, so that entry cancels.
fn axpy(work: &[(u32, u32)], pivot: &[(u32, u32)], factor: u32, p: u32, out: &mut Vec<(u32, u32)>) {
    out.clear();
    let (mut i, mut j) = (0usize, 0usize);
    while i < work.len() && j < pivot.len() {
        let (wc, wv) = work[i];
        let (pc, pv) = pivot[j];
        if wc < pc {
            out.push((wc, wv));
            i += 1;
        } else if wc > pc {
            let nv = (p - factor * pv % p) % p;
            if nv != 0 {
                out.push((pc, nv));
            }
            j += 1;
        } else {
            let nv = (wv + p - factor * pv % p) % p;
            if nv != 0 {
                out.push((wc, nv));
            }
            i += 1;
            j += 1;
        }
    }
    out.extend_from_slice(&work[i..]);
    for &(pc, pv) in &pivot[j..] {
        let nv = (p - factor * pv % p) % p;
        if nv != 0 {
            out.push((pc, nv));
        }
    }
}

/// `a^{-1} mod p` for prime `p`, via Fermat's little theorem.
fn mod_inverse(a: u32, p: u32) -> u32 {
    debug_assert!(!a.is_multiple_of(p));
    mod_pow(a % p, p - 2, p)
}

fn mod_pow(base: u32, mut exp: u32, p: u32) -> u32 {
    let mut result = 1u64;
    let mut b = base as u64 % p as u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * b % p as u64;
        }
        b = b * b % p as u64;
        exp >>= 1;
    }
    result as u32
}

/// `binomial(n, k)` for `k in 0..=n` (the dimensions `dim C^k = binom(N, k)`).
fn binomials(n: usize) -> Vec<usize> {
    let mut row = vec![0usize; n + 1];
    row[0] = 1;
    for i in 1..=n {
        for k in (1..=i).rev() {
            row[k] += row[k - 1];
        }
    }
    row
}

#[cfg(test)]
mod tests {
    use fp::prime::ValidPrime;

    use super::*;
    use crate::lie::morava_lie::BracketConvention;

    fn p(v: u32) -> ValidPrime {
        ValidPrime::new(v)
    }

    /// `d^2 = 0`, block by block, over every monomial: the correctness gate on the sign convention.
    fn check_d_squared(lie: &MoravaLie) {
        let n_dim = lie.dim();
        let prime = lie.prime().as_u32();
        let cobracket = lie.cobracket();
        for mask in 0u32..(1u32 << n_dim) {
            let first = differential_terms(mask, &cobracket, prime);
            let mut second: HashMap<u32, i64> = HashMap::new();
            for &(t, c) in &first {
                for (t2, c2) in differential_terms(t, &cobracket, prime) {
                    *second.entry(t2).or_insert(0) += (c as i64) * (c2 as i64);
                }
            }
            for (&t2, &v) in &second {
                assert_eq!(
                    v.rem_euclid(prime as i64),
                    0,
                    "d^2 != 0 at monomial {mask:#b} target {t2:#b}"
                );
            }
        }
    }

    #[test]
    fn sparse_rank_matches_known_small() {
        // A 3x3 rank-2 matrix over F_7.
        let rows = vec![
            vec![(0u32, 1u32), (1, 2), (2, 3)],
            vec![(0, 2), (1, 4), (2, 6)], // = 2 * row0, dependent
            vec![(1, 1), (2, 5)],
        ];
        assert_eq!(sparse_rank(rows, 7), 2);
        // Full-rank identity-ish.
        let rows = vec![vec![(0u32, 3u32)], vec![(1, 5)], vec![(2, 1)]];
        assert_eq!(sparse_rank(rows, 7), 3);
        // Empty.
        assert_eq!(sparse_rank(Vec::<Vec<(u32, u32)>>::new(), 5), 0);
    }

    #[test]
    fn graded_charts_refine_betti() {
        // The tri/bigraded charts must sum correctly: Σ over the finer gradings = betti[s], and the
        // grand total = dim H^*. Checks both trigraded (n,s,w) and its bigraded (s,t) collapse.
        for (n, prime, total) in [(2u32, 5u32, 12usize), (3, 7, 152), (4, 7, 3440)] {
            let lie = MoravaLie::new(p(prime), n);
            let stats = chevalley_eilenberg_cohomology(&lie);

            let tri = trigraded_cohomology(&lie);
            assert_eq!(tri.values().sum::<usize>(), total, "trigraded total wrong at n={n}");
            let mut per_s = vec![0usize; lie.dim() + 1];
            for (deg, &d) in &tri {
                per_s[deg.s] += d;
            }
            assert_eq!(per_s, stats.betti, "Σ_{{n,w}} H must equal betti[s] at n={n}");

            let bi = bigraded_cohomology(&lie);
            assert_eq!(bi.values().sum::<usize>(), total, "bigraded total wrong at n={n}");
            let mut per_s_bi = vec![0usize; lie.dim() + 1];
            for (&(s, _t), &d) in &bi {
                per_s_bi[s] += d;
            }
            assert_eq!(per_s_bi, stats.betti, "Σ_t H^{{s,t}} must equal betti[s] at n={n}");
        }
    }

    #[test]
    fn h1_dimension_pattern() {
        // dim H^1(L(n,n)) = N - rank(d_1) = dim of the coabelianization. Empirically this is n + 1.
        // The n = 5 case exercises the i + k = n = 5 brackets, which n <= 4 never reach, so it is a
        // fast independent check underpinning the full n = 5 computation (dim H^* = 128992). Ranking
        // d_1 : C^1 -> C^2 is cheap (domain dimension N = n^2).
        for (n, prime) in [(2u32, 5u32), (3, 7), (4, 7), (5, 11)] {
            let lie = MoravaLie::new(p(prime), n);
            let cobracket = lie.cobracket();
            let n_dim = lie.dim();
            let rows = (0..n_dim).map(|a| differential_terms(1u32 << a, &cobracket, prime));
            let rank_d1 = sparse_rank(rows, prime);
            assert_eq!(
                n_dim - rank_d1,
                (n + 1) as usize,
                "dim H^1(L({n},{n})) should be n + 1"
            );
        }
    }

    #[test]
    fn enumerate_weight_counts() {
        // weights [1,2,2,3]: subsets of weight 3 are {3}, {1,2a}, {1,2b} -> 3 of them.
        let weights = [1u32, 2, 2, 3];
        let mut count = 0;
        enumerate_weight(&weights, 3, &mut |_| count += 1);
        assert_eq!(count, 3);
        // Total over all weights must be 2^4 = 16.
        let total_w: u32 = weights.iter().sum();
        let mut total = 0;
        for w in 0..=total_w {
            enumerate_weight(&weights, w, &mut |_| total += 1);
        }
        assert_eq!(total, 16);
    }

    #[test]
    fn d_squared_is_zero_n2() {
        for conv in [BracketConvention::GreenBook, BracketConvention::Salch] {
            check_d_squared(&MoravaLie::with_convention(p(5), 2, conv));
        }
    }

    #[test]
    fn d_squared_is_zero_n3() {
        for conv in [BracketConvention::GreenBook, BracketConvention::Salch] {
            check_d_squared(&MoravaLie::with_convention(p(7), 3, conv));
        }
    }

    #[test]
    fn validation_ladder_n1() {
        let stats = chevalley_eilenberg_cohomology(&MoravaLie::new(p(3), 1));
        assert_eq!(stats.total_dim, 2);
        assert_eq!(stats.betti, vec![1, 1]);
    }

    #[test]
    fn validation_ladder_n2() {
        for conv in [BracketConvention::GreenBook, BracketConvention::Salch] {
            let stats =
                chevalley_eilenberg_cohomology(&MoravaLie::with_convention(p(5), 2, conv));
            assert_eq!(stats.total_dim, 12, "n=2 total dim wrong for {conv:?}");
            let mut rev = stats.betti.clone();
            rev.reverse();
            assert_eq!(stats.betti, rev, "n=2 Betti not symmetric for {conv:?}");
        }
    }

    #[test]
    fn validation_ladder_n3() {
        // The rung that disambiguates the two OCR readings of green-book Thm 6.3.3 (handoff §2):
        // GreenBook reproduces the known 152, the swapped reading gives 128.
        let green = chevalley_eilenberg_cohomology(&MoravaLie::with_convention(
            p(7),
            3,
            BracketConvention::GreenBook,
        ));
        assert_eq!(green.total_dim, 152, "GreenBook must reproduce the known 152");
        let salch = chevalley_eilenberg_cohomology(&MoravaLie::with_convention(
            p(7),
            3,
            BracketConvention::Salch,
        ));
        assert_ne!(
            salch.total_dim, 152,
            "the swapped-subscript reading should NOT match; it is the wrong transcription"
        );
    }

    #[test]
    fn validation_ladder_n4() {
        // The near-term prize: independent confirmation of Salch's in-progress H^*(L(4,4)) = 3440.
        let stats = chevalley_eilenberg_cohomology(&MoravaLie::new(p(7), 4));
        assert_eq!(stats.total_dim, 3440);
        // Poincaré self-duality of Lie-algebra cohomology.
        let mut rev = stats.betti.clone();
        rev.reverse();
        assert_eq!(stats.betti, rev);
    }
}
