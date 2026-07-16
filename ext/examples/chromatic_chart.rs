//! Bigraded `(s, t)` chart of `H^*(L(n, n); F_p)` — the large-prime `H^*(S(n))` at the `E_1 = E_∞`
//! of Ravenel's May spectral sequence (collapses for `p > n + 1`; Salch, arXiv:2312.17185).
//!
//! `s` is the cohomological (May-filtration) degree; `t ∈ Z/(p^n − 1)` is the internal degree
//! `t_{i,j} = p^j(p^i − 1) mod (p^n − 1)` (the periodic "chromatic" grading — see
//! `MoravaLie::internal_degree` and `ext/docs/chromatic-computations.md` §6). The Chevalley–Eilenberg
//! differential is homogeneous for this grading, so `H^*` is genuinely bigraded and `Σ_t H^{s,t}`
//! recovers the cohomological Betti numbers.
//!
//! Prints the `(s, t)` table and renders it to an SVG via the `sseq` charting backend (Adams
//! convention: the dot at stem `n = t − s`, filtration `s`).
//!
//! Usage: `cargo run --release --features concurrent --example chromatic_chart -- [n] [p] [out.svg]`

use std::io::Write as _;

use algebra::lie::{MoravaLie, bigraded_cohomology};
use fp::prime::ValidPrime;
use sseq::{Adams, Product, Sseq, charting::SvgBackend, coordinates::Bidegree};

fn is_prime(p: u32) -> bool {
    p >= 2 && (2..p).take_while(|k| k * k <= p).all(|k| p % k != 0)
}

fn large_prime_for(n: u32) -> u32 {
    let mut p = n + 2;
    while !is_prime(p) {
        p += 1;
    }
    p
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let n: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let p: u32 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| large_prime_for(n));
    let out = args
        .next()
        .unwrap_or_else(|| format!("chromatic_L{n}{n}_p{p}.svg"));

    let lie = MoravaLie::new(ValidPrime::new(p), n);
    let modulus = lie.internal_modulus();
    let chart = bigraded_cohomology(&lie);
    let total: usize = chart.values().sum();

    println!("H^{{s,t}}(L({n},{n})) over F_{p},  t ∈ Z/{modulus},  total dim = {total}");
    // Cohomological Betti row (Σ_t H^{s,t}) — the coarse invariant, always shown.
    let mut per_s = vec![0usize; lie.dim() + 1];
    for (&(s, _t), &d) in &chart {
        per_s[s] += d;
    }
    while per_s.last() == Some(&0) {
        per_s.pop();
    }
    println!("  H^s dims (Σ_t): {per_s:?}");
    // Full (s,t) table only when compact.
    if chart.len() <= 80 {
        println!("  {:>3}  {:>6}  {:>5}", "s", "t", "dim");
        for (&(s, t), &d) in &chart {
            println!("  {s:>3}  {t:>6}  {d:>5}");
        }
    } else {
        println!("  ({} occupied (s,t) bidegrees — see the SVG)", chart.len());
    }

    // Build the E_2( = E_∞) page and render. Dimensions only — no differentials at large primes.
    let mut sseq = Sseq::<2, Adams>::new(ValidPrime::new(p));
    for (&(s, t), &d) in &chart {
        sseq.set_dimension(Bidegree::s_t(s as i32, t as i32), d);
    }
    let file = std::fs::File::create(&out)?;
    sseq.write_to_graph(
        SvgBackend::new(file),
        2,
        false,
        std::iter::empty::<&(String, Product<2>)>(),
        |_| Ok(()),
    )?;
    // Flush note.
    std::io::stdout().flush().ok();
    println!("wrote SVG chart to {out}");
    Ok(())
}
