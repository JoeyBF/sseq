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
//! # Block decomposition
//!
//! The `i`-weight of [`MoravaLie`] is additive under the bracket, hence preserved by `d`. So the
//! complex splits as a direct sum over total `i`-weight `w`, and within each weight over
//! cohomological degree `k`. We rank each `(w, k)` block independently — the blocks are small even
//! though the full complex has dimension `2^{n^2}` (e.g. `65536` at `n = 4`).

use std::collections::HashMap;

use fp::{matrix::Matrix, prime::Prime};

use crate::lie::morava_lie::MoravaLie;

/// The maximum Lie-algebra dimension this (dense, bitmask-enumerating) implementation will accept.
/// `n = 4` gives dimension `16`; `n = 5` gives `25`, whose `2^25`-dimensional complex is beyond a
/// dense brute-force pass (see `ext/docs/chromatic-computations.md` §3, "the moonshot").
const MAX_DIM: usize = 20;

/// The result of a Chevalley–Eilenberg cohomology computation.
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

/// Computes the Chevalley–Eilenberg cohomology of `lie` over `F_p`.
///
/// Panics if `lie.dim() > MAX_DIM` (the dense enumerator would exhaust memory).
pub fn chevalley_eilenberg_cohomology(lie: &MoravaLie) -> CohomologyStats {
    let n_dim = lie.dim();
    assert!(
        n_dim <= MAX_DIM,
        "Chevalley–Eilenberg dimension {n_dim} exceeds the dense brute-force limit {MAX_DIM}; \
         this needs the streaming/height-shifting approach (handoff §3)."
    );
    let prime = lie.prime();
    let p = prime.as_u32();
    let cobracket = lie.cobracket();
    let weights: Vec<u32> = (0..n_dim).map(|a| lie.weight(a)).collect();

    // Bucket every basis monomial (a subset of the n_dim generators, encoded as a bitmask) by its
    // total i-weight. Within a weight we later split by cohomological degree (popcount).
    let total = 1usize << n_dim;
    let mut by_weight: HashMap<u32, Vec<u32>> = HashMap::new();
    // dim C^k = number of monomials of popcount k, accumulated for the Betti numbers.
    let mut dim_ck = vec![0usize; n_dim + 1];
    for mask in 0..total {
        let mask = mask as u32;
        let k = mask.count_ones() as usize;
        dim_ck[k] += 1;
        let mut w = 0u32;
        let mut bits = mask;
        while bits != 0 {
            let a = bits.trailing_zeros() as usize;
            w += weights[a];
            bits &= bits - 1;
        }
        by_weight.entry(w).or_default().push(mask);
    }

    // rank(d_k) summed over all weights.
    let mut ranks = vec![0usize; n_dim + 1];

    for masks in by_weight.values() {
        // Split this weight's monomials by cohomological degree.
        let mut by_k: HashMap<usize, Vec<u32>> = HashMap::new();
        for &m in masks {
            by_k.entry(m.count_ones() as usize).or_default().push(m);
        }
        for (&k, domain) in &by_k {
            let Some(codomain) = by_k.get(&(k + 1)) else {
                continue; // d_k is zero into an empty target
            };
            if domain.is_empty() || codomain.is_empty() {
                continue;
            }
            // Column index of each codomain monomial.
            let col_of: HashMap<u32, usize> = codomain
                .iter()
                .enumerate()
                .map(|(idx, &m)| (m, idx))
                .collect();
            let ncols = codomain.len();
            let mut rows: Vec<Vec<u32>> = Vec::with_capacity(domain.len());
            for &mask in domain {
                let mut row = vec![0u32; ncols];
                differential(mask, &cobracket, p, |target, coeff| {
                    let col = col_of[&target];
                    row[col] = (row[col] + coeff) % p;
                });
                rows.push(row);
            }
            let mut matrix = Matrix::from_vec(prime, &rows);
            ranks[k] += matrix.row_reduce();
        }
    }

    let total_rank: usize = ranks.iter().sum();
    let total_dim = total - 2 * total_rank;

    // betti[k] = dim C^k - rank(d_k) - rank(d_{k-1}).
    let mut betti = vec![0usize; n_dim + 1];
    for k in 0..=n_dim {
        let prev = if k == 0 { 0 } else { ranks[k - 1] };
        betti[k] = dim_ck[k] - ranks[k] - prev;
    }

    CohomologyStats {
        prime: p,
        dim: n_dim,
        total_dim,
        ranks,
        betti,
    }
}

/// Applies the Chevalley–Eilenberg differential to the basis monomial `mask`, calling
/// `emit(target_mask, coeff)` for each nonzero term (`coeff` already reduced mod `p`, in `1..p`).
fn differential(
    mask: u32,
    cobracket: &[Vec<(usize, usize, i32)>],
    p: u32,
    mut emit: impl FnMut(u32, u32),
) {
    let mut bits = mask;
    while bits != 0 {
        let a = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        // r = number of set bits of `mask` strictly below `a` (the Koszul position of e_a).
        let r = (mask & ((1u32 << a) - 1)).count_ones();
        let koszul = if r.is_multiple_of(2) { 1i32 } else { -1i32 };
        for &(b, c, coeff) in &cobracket[a] {
            let bit_b = 1u32 << b;
            let bit_c = 1u32 << c;
            // A repeated generator wedges to zero. (b, c != a since the bracket raises weight.)
            if mask & bit_b != 0 || mask & bit_c != 0 {
                continue;
            }
            let target = (mask & !(1u32 << a)) | bit_b | bit_c;
            // Sign of reordering [prefix, b, c, suffix] (the wedge order) into ascending order.
            let sort_sign = wedge_sort_sign(mask, a, b, c);
            // d(e_a) contributes with an overall -1: (-1)^r * (-1) * coeff * sort_sign.
            let signed = -koszul * coeff * sort_sign;
            let reduced = signed.rem_euclid(p as i32) as u32;
            if reduced != 0 {
                emit(target, reduced);
            }
        }
    }
}

/// The sign (`+1`/`-1`) of the permutation that sorts the wedge
/// `e_{a_0} /\ ... /\ e_{a_{r-1}} /\ e_b /\ e_c /\ e_{a_{r+1}} /\ ... /\ e_{a_{k-1}}` into ascending
/// index order, where `{a_0 < ... < a_{k-1}}` are the set bits of `mask` and `a = a_r` is being
/// replaced by the pair `(b, c)` with `b < c`. Callers guarantee `b, c` are not already in `mask`.
fn wedge_sort_sign(mask: u32, a: usize, b: usize, c: usize) -> i32 {
    // Build the wedge order: ascending set bits of `mask`, with `a` replaced by `b, c`.
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
    // Count inversions (small sequence); parity gives the sign.
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

#[cfg(test)]
mod tests {
    use fp::prime::ValidPrime;

    use super::*;
    use crate::lie::morava_lie::BracketConvention;

    fn p(v: u32) -> ValidPrime {
        ValidPrime::new(v)
    }

    /// Verify `d^2 = 0` block by block: for every weight and degree `k`, the composite
    /// `C^k -> C^{k+1} -> C^{k+2}` must vanish. This is the correctness gate on the sign convention.
    fn check_d_squared(lie: &MoravaLie) {
        let n_dim = lie.dim();
        let p = lie.prime().as_u32();
        let cobracket = lie.cobracket();
        for mask in 0u32..(1u32 << n_dim) {
            // (target -> coeff) after one differential.
            let mut first: HashMap<u32, u32> = HashMap::new();
            differential(mask, &cobracket, p, |t, c| {
                let e = first.entry(t).or_insert(0);
                *e = (*e + c) % p;
            });
            // Apply d again and accumulate; everything must cancel.
            let mut second: HashMap<u32, i64> = HashMap::new();
            for (&t, &c) in &first {
                if c == 0 {
                    continue;
                }
                differential(t, &cobracket, p, |t2, c2| {
                    *second.entry(t2).or_insert(0) += (c as i64) * (c2 as i64);
                });
            }
            for (&t2, &v) in &second {
                assert_eq!(
                    v.rem_euclid(p as i64),
                    0,
                    "d^2 != 0 at monomial {mask:#b} target {t2:#b}"
                );
            }
        }
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
        // H^*(L(1,1)) = E(h_{1,0}), dimension 2.
        let stats = chevalley_eilenberg_cohomology(&MoravaLie::new(p(3), 1));
        assert_eq!(stats.total_dim, 2);
        assert_eq!(stats.betti, vec![1, 1]);
    }

    #[test]
    fn validation_ladder_n2() {
        // dim H^*(L(2,2)) = 12 (Salch table; known for all p). The two bracket transcriptions
        // happen to coincide at n = 2.
        for conv in [BracketConvention::GreenBook, BracketConvention::Salch] {
            let stats =
                chevalley_eilenberg_cohomology(&MoravaLie::with_convention(p(5), 2, conv));
            assert_eq!(stats.total_dim, 12, "n=2 total dim wrong for {conv:?}");
            // Poincaré duality of Lie-algebra cohomology: betti is symmetric.
            let mut rev = stats.betti.clone();
            rev.reverse();
            assert_eq!(stats.betti, rev, "n=2 Betti not symmetric for {conv:?}");
        }
    }

    #[test]
    fn validation_ladder_n3() {
        // dim H^*(L(3,3)) = 152 (Salch table; known for p >= 5). This is the rung that
        // disambiguates the two OCR readings of green-book Thm 6.3.3 (handoff §2): the GreenBook
        // transcription reproduces 152, while the swapped-subscript ("Salch eq. (8)") reading
        // gives 128 -- so GreenBook is the correct one, and it is our default.
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
}
