//! The C-motivic Adams $E_2$ page (over $\mathbb{C}$, $p = 2$) by deformation.
//!
//! Resolves $A_C/\tau$ over $\mathbb{F}_2$, lifts to an honest $A_C$ resolution
//! over $\mathbb{F}_2[\tau]$, and reads the three pages of the one object off the
//! cohomology $H(\delta)$ (see [`ext::motivic`]):
//!
//! - **alg-Nov** — the algebraic Novikov $E_2$ (set $\tau = 0$: generator counts).
//! - **classical** — the classical Adams $E_2$ (invert $\tau$: free rank of $H(δ)$).
//! - **τ-tors** — a `*` marks bidegrees carrying genuine motivic $\tau$-torsion
//!   (a class that dies when $\tau$ is inverted, e.g. the $h_1$-tower beyond
//!   $h_1^3$).
//!
//! Output is `n,s,alg_nov,classical,tau_torsion`, one line per nonzero bidegree.
//! Run with e.g. `cargo run --release --example resolve_motivic` (reads `Max n` /
//! `Max s`, defaulting to 12 / 8).

use ext::motivic::MotivicResolution;
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let max = Bidegree::n_s(
        query::with_default("Max n", "12", str::parse),
        query::with_default("Max s", "8", str::parse),
    );

    let res = MotivicResolution::new(max);

    let profile = std::env::var("MOT_PROFILE").is_ok();
    let t_coh = std::time::Instant::now();

    // The classical rank needs the lift one homological degree up, so the top
    // reported filtration is `max.s() - 1`.
    let top_s = max.s() - 1;

    // The cohomology at each (s, n) is independent — compute the chart in
    // parallel, then print in deterministic order.
    let cells: Vec<(i32, i32)> = (0..=top_s)
        .flat_map(|s| (0..=max.n()).map(move |n| (s, n)))
        .collect();
    let mut lines: Vec<(i32, i32, String)> = cells
        .into_maybe_par_iter()
        .filter_map(|(s, n)| {
            let t = n + s;
            let alg_nov = res.algebraic_novikov_rank(s, t);
            let classical = res.classical_ext_rank(s, t);
            let torsion = res.has_tau_torsion(s, t);
            (alg_nov > 0 || classical > 0 || torsion).then(|| {
                (
                    s,
                    n,
                    format!("{n},{s},{alg_nov},{classical},{}", if torsion { "*" } else { "" }),
                )
            })
        })
        .collect();
    lines.sort_by_key(|&(s, n, _)| (s, n));

    if profile {
        eprintln!("[profile] cohomology: {:?}", t_coh.elapsed());
    }

    println!("n,s,alg_nov,classical,tau_torsion");
    for (_, _, line) in lines {
        println!("{line}");
    }

    Ok(())
}
