//! Full `H^*(L(n, n); F_p)` total dimension via the `Z/n`-character splitting — the route to `n = 5`.
//!
//! Uses [`equivariant_total_dim`], which splits each Chevalley–Eilenberg block into its `n` character
//! pieces (each `~1/n` the size) over `F_p` with `p ≡ 1 (mod n)`. Validated against the known totals
//! `12, 152, 3440` at `n = 2, 3, 4`; the goal is the open `n = 5` total (conjecturally `128512`).
//!
//! Run: `cargo run --release --features concurrent --example chromatic_n5 -- [max_n]` (default 5).

use std::time::Instant;

use algebra::lie::{MoravaLie, equivariant_total_dim};
use fp::prime::ValidPrime;

/// `(n, prime, expected)` — a prime `≡ 1 (mod n)` and `> n + 1`; expected total (conj. at n = 5).
const CASES: &[(u32, u32, Option<usize>)] = &[
    (2, 5, Some(12)),
    (3, 7, Some(152)),
    (4, 13, Some(3440)),
    (5, 11, Some(128512)),
    (6, 13, Some(7621888)),
];

fn main() {
    let max_n: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    println!("{:>3}  {:>3}  {:>12}  {:>12}   {:>9}   status", "n", "p", "dim H^*", "expected", "secs");
    println!("{}", "-".repeat(66));

    for &(n, p, expected) in CASES {
        if n > max_n {
            break;
        }
        let lie = MoravaLie::new(ValidPrime::new(p), n);
        let start = Instant::now();
        let total = equivariant_total_dim(&lie);
        let secs = start.elapsed().as_secs_f64();
        let status = match expected {
            Some(e) if e == total => "OK",
            Some(_) => "MISMATCH",
            None => "(new)",
        };
        println!(
            "{n:>3}  {p:>3}  {total:>12}  {:>12}   {secs:>9.2}   {status}",
            expected.map(|e| e.to_string()).unwrap_or_else(|| "?".into()),
        );
    }
}
