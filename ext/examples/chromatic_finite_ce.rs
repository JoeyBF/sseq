//! Large-prime chromatic computation: `H^*(L(n, n); F_p)` by finite Chevalley–Eilenberg cohomology.
//!
//! This is the tool described in `ext/docs/chromatic-computations.md` §3. For `p > n + 1` the
//! height-`n` Morava stabilizer algebra's May spectral sequence collapses immediately (Salch,
//! arXiv:2312.17185), so `H^*(S(n)) = H^*(L(n, n))`, the *ordinary* cohomology of the finite
//! `n^2`-dimensional Morava Lie algebra `L(n, n)` (green book Thm 6.3.3 / Thm 1.4). We build that
//! Lie algebra, form the Chevalley–Eilenberg (Koszul) complex, and rank its differentials over
//! `F_p` — pure finite linear algebra, no resolution engine or field trick.
//!
//! The total dimension `dim H^* = 2^{n^2} - 2 sum_k rank(d_k)` is grading-independent, so it is
//! checked directly against Salch's validation table:
//!
//! ```text
//!   n  | dim H^*(L(n,n)) | status
//!   ---+-----------------+--------------------------------------------------
//!   1  | 2               | trivial (= E(h_{1,0}))
//!   2  | 12              | known, all p
//!   3  | 152             | known, p >= 5
//!   4  | 3440            | being written up (Salch); this is an independent check
//!   5  | 128512 (conj.)  | open -- middle blocks reach ~565k; only partial here
//! ```
//!
//! Usage: `cargo run --release --example chromatic_finite_ce -- [max_n] [max_block]`
//!   * `max_n`     : highest height to compute (default 4).
//!   * `max_block` : per-block domain-size cap; blocks larger than this are skipped and their weight
//!                   reported as incomplete. Default: none for `n <= 4`, `200000` when `max_n >= 5`.
//!
//! Examples:
//!   `cargo run --release --example chromatic_finite_ce -- 4`            (the full ladder)
//!   `cargo run --release --example chromatic_finite_ce -- 5 60000`      (partial n = 5, cap 60k)

use algebra::lie::{MoravaLie, Options, cohomology_by_weight};
use fp::prime::ValidPrime;

/// Reference total dimensions `(n, dim, known)` (Salch, arXiv:2312.17185). `known = true` are
/// established; `known = false` are *conjectural*. NOTE: this tool computes `dim H^*(L(5,5)) = 128992`
/// (reproduced at `p = 7` and `p = 11` with identical Betti numbers), which **disagrees** with the
/// conjectural `128512` — see `ext/docs/chromatic-computations.md` §3.
const EXPECTED: &[(u32, usize, bool)] = &[
    (1, 2, true),
    (2, 12, true),
    (3, 152, true),
    (4, 3440, true),
    (5, 128512, false),
];

/// The smallest prime `> n + 1` (the large-prime regime where the collapse holds).
fn large_prime_for(n: u32) -> u32 {
    let mut p = n + 2;
    while !is_prime(p) {
        p += 1;
    }
    p
}

fn is_prime(p: u32) -> bool {
    p >= 2 && (2..p).take_while(|k| k * k <= p).all(|k| p % k != 0)
}

fn expected_for(n: u32) -> Option<(usize, bool)> {
    EXPECTED.iter().find(|(m, _, _)| *m == n).map(|&(_, d, k)| (d, k))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let max_n: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let max_block: Option<usize> = match args.next().and_then(|s| s.parse().ok()) {
        Some(0) => None,       // 0 means "uncapped"
        Some(c) => Some(c),
        None if max_n >= 5 => Some(200_000),
        None => None,
    };
    // Optional 3rd arg: override the prime for *every* n (to probe characteristic dependence).
    let prime_override: Option<u32> = args.next().and_then(|s| s.parse().ok());

    println!(
        "{:>3}  {:>2}  {:>6}  {:>12}  {:>12}   {}",
        "n", "p", "dim L", "dim H^*", "expected", "status"
    );
    println!("{}", "-".repeat(78));

    for n in 1..=max_n {
        let p = prime_override.unwrap_or_else(|| large_prime_for(n));
        let lie = MoravaLie::new(ValidPrime::new(p), n);
        let dim = lie.dim();
        let opts = Options { max_block };

        let reports = cohomology_by_weight(&lie, opts);
        let complete = reports.iter().all(|r| r.complete);
        let total: usize = 1usize << dim;
        let total_rank: usize = reports
            .iter()
            .flat_map(|r| r.ranks.iter())
            .copied()
            .sum();

        if complete {
            let total_dim = total - 2 * total_rank;
            let expected = expected_for(n);
            let status = match expected {
                Some((e, _)) if e == total_dim => "OK".to_string(),
                // A mismatch against a *known* value is a bug; against a *conjecture* it is a finding.
                Some((e, true)) => format!("MISMATCH (known {e})"),
                Some((e, false)) => format!("computed (conj. was {e})"),
                None => "(no reference)".to_string(),
            };
            println!(
                "{n:>3}  {p:>2}  {dim:>6}  {total_dim:>12}  {:>12}   {status}",
                expected
                    .map(|(e, _)| e.to_string())
                    .unwrap_or_else(|| "?".into()),
            );
            // Betti numbers (Poincaré self-dual) as a graded sanity check.
            let mut betti = vec![0usize; dim + 1];
            let mut ranks = vec![0usize; dim + 1];
            for r in &reports {
                for (k, &rk) in r.ranks.iter().enumerate() {
                    ranks[k] += rk;
                }
            }
            let dim_ck = binomials(dim);
            for k in 0..=dim {
                let prev = if k == 0 { 0 } else { ranks[k - 1] };
                betti[k] = dim_ck[k] - ranks[k] - prev;
            }
            let betti: Vec<String> = betti.iter().map(|b| b.to_string()).collect();
            println!("        H^k dims: [{}]", betti.join(", "));
        } else {
            // Partial: report what was reachable and where the wall is.
            let done: Vec<u32> = reports
                .iter()
                .filter(|r| r.complete)
                .map(|r| r.weight)
                .collect();
            let incomplete: Vec<&_> = reports.iter().filter(|r| !r.complete).collect();
            let biggest = reports.iter().map(|r| r.max_block).max().unwrap_or(0);
            // Partial lower bound on the *co*rank: 2^N - 2 * (rank so far) upper-bounds dim H^*,
            // but with blocks skipped we cannot pin the total; report the reachable structure.
            let expected = expected_for(n)
                .map(|(e, _)| e.to_string())
                .unwrap_or_else(|| "?".into());
            println!(
                "{n:>3}  {p:>2}  {dim:>6}  {:>12}  {expected:>12}   PARTIAL (cap {})",
                "-",
                max_block.unwrap_or(0),
            );
            println!(
                "        weights complete: {}/{}   ranked rank-sum (partial): {}   biggest block seen: {}",
                done.len(),
                reports.len(),
                total_rank,
                biggest,
            );
            let sample: Vec<String> = incomplete
                .iter()
                .take(8)
                .map(|r| format!("w={}({})", r.weight, r.max_block))
                .collect();
            println!(
                "        {} weights blocked by cap; e.g. {}",
                incomplete.len(),
                sample.join(", "),
            );
        }
    }
}

/// `binomial(n, k)` for `k in 0..=n`.
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
