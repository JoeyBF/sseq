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

use std::collections::BTreeMap;

use algebra::lie::{MoravaLie, trigraded_cohomology};
use fp::prime::ValidPrime;
use sseq::{
    Adams, Product, Sseq,
    charting::{SeqSeeBackend, SvgBackend},
    coordinates::Bidegree,
};

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
    // Trigraded H^{n,s,w}, keyed (stem n = t−s, filtration s, i-weight w) — the sseq display order.
    let chart = trigraded_cohomology(&lie);
    let total: usize = chart.values().sum();

    println!(
        "H^*(L({n},{n})) over F_{p},  stem n = t−s,  t ∈ Z/{modulus},  total dim = {total}",
    );
    // Cohomological Betti row (Σ over n, w) — the coarse invariant, always shown.
    let mut per_s = vec![0usize; lie.dim() + 1];
    for (deg, &d) in &chart {
        per_s[deg.s] += d;
    }
    while per_s.last() == Some(&0) {
        per_s.pop();
    }
    println!("  H^s dims (Σ_{{n,w}}): {per_s:?}");
    // Full (n, s, weight) table only when compact.
    if chart.len() <= 100 {
        println!("  {:>4}  {:>3}  {:>4}  {:>4}", "n", "s", "wt", "dim");
        for (deg, &d) in &chart {
            println!("  {:>4}  {:>3}  {:>4}  {:>4}", deg.n, deg.s, deg.weight, d);
        }
    } else {
        println!("  ({} occupied (n,s,weight) cells — see the SVG)", chart.len());
    }

    // Sum the weight grading into each (s, t) cell.
    let mut cells: BTreeMap<(usize, u64), usize> = BTreeMap::new();
    for (deg, &d) in &chart {
        let t = (deg.n + deg.s as i64) as u64; // t = n + s, a representative in [0, modulus)
        *cells.entry((deg.s, t)).or_insert(0) += d;
    }

    // Two displays:
    //   * Adams (pass `adams` as a 4th arg): honest (s, t), plotter draws stem n = t − s on x.
    //   * default: plot t directly on x (no diagonal shift) and TRIM empty columns — reindex the
    //     occupied t-values to consecutive integers. Since every t is a multiple of p − 1 (as
    //     t_{i,j} = pʲ(pⁱ − 1) ≡ 0 mod p − 1), and most multiples are still empty, this collapses a
    //     ~16806-wide height-5 chart down to just its populated columns.
    let adams = std::env::args().any(|a| a == "adams");
    let mut sseq = Sseq::<2, Adams>::new(ValidPrime::new(p));
    if adams {
        for (&(s, t), &d) in &cells {
            sseq.set_dimension(Bidegree::s_t(s as i32, t as i32), d);
        }
    } else {
        // Occupied t-values, in order, mapped to consecutive columns.
        let mut occupied_t: Vec<u64> = cells.keys().map(|&(_, t)| t).collect();
        occupied_t.sort_unstable();
        occupied_t.dedup();
        let col: std::collections::HashMap<u64, i32> = occupied_t
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, i as i32))
            .collect();
        println!(
            "  trimmed (t,s) display: {} occupied t-columns (of {} multiples of p−1 = {})",
            occupied_t.len(),
            modulus / (p as u64 - 1),
            p - 1,
        );
        for (&(s, t), &d) in &cells {
            // x = trimmed column, y = s.  (Bidegree::n_s puts arg0 on the x-axis directly.)
            sseq.set_dimension(Bidegree::n_s(col[&t], s as i32), d);
        }
    }
    // Render. Dispatch on the output extension: `.json` -> SeqSee descriptor, else SVG. No
    // differentials or products at large primes (the page is already E_∞).
    let file = std::fs::File::create(&out)?;
    let empty = std::iter::empty::<&(String, Product<2>)>();
    if out.ends_with(".json") {
        sseq.write_to_graph(SeqSeeBackend::new(file), 2, false, empty, |_| Ok(()))?;
        println!("wrote SeqSee JSON descriptor to {out}");
    } else {
        sseq.write_to_graph(SvgBackend::new(file), 2, false, empty, |_| Ok(()))?;
        println!("wrote SVG chart to {out}");
    }
    Ok(())
}
