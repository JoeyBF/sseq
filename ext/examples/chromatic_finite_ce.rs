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
//!   5  | 128512 (conj.)  | open -- beyond this dense brute-force pass
//! ```
//!
//! Run with e.g. `cargo run --release --example chromatic_finite_ce -- 4` (compute up to `n = 4`).

use algebra::lie::{MoravaLie, chevalley_eilenberg_cohomology};
use fp::prime::ValidPrime;

/// The known/conjectural total dimensions, indexed by `n` (Salch, arXiv:2312.17185).
const EXPECTED: &[(u32, usize)] = &[(1, 2), (2, 12), (3, 152), (4, 3440), (5, 128512)];

/// The smallest prime `> n + 1` (the large-prime regime where the collapse holds).
fn large_prime_for(n: u32) -> u32 {
    let mut p = n + 2;
    while !is_prime(p) {
        p += 1;
    }
    p
}

fn is_prime(p: u32) -> bool {
    if p < 2 {
        return false;
    }
    (2..p).take_while(|k| k * k <= p).all(|k| p % k != 0)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let max_n: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);

    println!(
        "{:>3}  {:>2}  {:>6}  {:>12}  {:>12}   {}",
        "n", "p", "dim L", "dim H^*", "expected", "status"
    );
    println!("{}", "-".repeat(72));

    for n in 1..=max_n {
        let p = large_prime_for(n);
        let lie = MoravaLie::new(ValidPrime::new(p), n);
        let dim = lie.dim();

        if dim > 20 {
            println!(
                "{n:>3}  {p:>2}  {dim:>6}  {:>12}  {:>12}   SKIPPED (2^{dim} complex is beyond the \
                 dense pass; see handoff §3)",
                "-",
                EXPECTED
                    .iter()
                    .find(|(m, _)| *m == n)
                    .map(|(_, d)| d.to_string())
                    .unwrap_or_else(|| "?".to_string()),
            );
            continue;
        }

        let stats = chevalley_eilenberg_cohomology(&lie);
        let expected = EXPECTED.iter().find(|(m, _)| *m == n).map(|(_, d)| *d);
        let status = match expected {
            Some(e) if e == stats.total_dim => "OK".to_string(),
            Some(e) => format!("MISMATCH (expected {e})"),
            None => "(no reference)".to_string(),
        };
        println!(
            "{n:>3}  {p:>2}  {dim:>6}  {:>12}  {:>12}   {status}",
            stats.total_dim,
            expected
                .map(|e| e.to_string())
                .unwrap_or_else(|| "?".to_string()),
        );
        // Print the Betti numbers (Lie-algebra cohomology is Poincaré self-dual, so this row is
        // palindromic) as a graded sanity check beyond the total dimension.
        let betti: Vec<String> = stats.betti.iter().map(|b| b.to_string()).collect();
        println!("        H^k dims: [{}]", betti.join(", "));
    }
}
