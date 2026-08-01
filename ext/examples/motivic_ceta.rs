//! Dan's suggestion: most of the C-motivic Adams $E_2$ is infinite $h_1$-towers
//! (motivically $h_1$ is non-nilpotent), which march up the $n = s$ diagonal and
//! bloat the resolution to high filtration at every stem. The cofiber of $\eta$
//! collapses each tower to a single class, so resolving $C\eta$ should carry far
//! fewer generators per stem — and the sphere is recovered from the cofiber long
//! exact sequence whose connecting map is $\cdot h_1$:
//!
//! ```text
//!  → Ext^{s,t}(S⁰) --·h₁--> Ext^{s+1,t+2}(S⁰) → Ext^{s+1,t+2}(Cη) → Ext^{s+1,t}(S⁰) →
//! ```
//!
//! This example measures the payoff: resolve the sphere and $C\eta$ through the same
//! box and compare, per stem, the total generator count (summed over filtration) and
//! total $\mathbb{F}_2[\tau]$-module free rank, plus wall-clock. A large gap that
//! grows with the stem is the $h_1$-tower collapse.
//!
//! $C\eta$ is the 2-cell module $\{x_0, x_2\}$ joined by $\mathrm{Sq}^2 = P(0,1)$ (the
//! operation detecting $h_1 = \eta$), the exact analogue of the Moore space $S/2$ in
//! `resolve_motivic_moore.rs` (there $\mathrm{Sq}^1 = Q_0$ cones off $h_0 = 2$).
//!
//! Run: `MOT_N=30 cargo run --release --features concurrent --example motivic_ceta`

use ext::motivic::MotivicResolution;
use sseq::coordinates::Bidegree;

/// Total generators (algebraic-Novikov rank) at filtration `s`, summed over the
/// report stems `0..=n`.
fn filt_total(res: &MotivicResolution, s: i32, n: i32) -> usize {
    (0..=n).map(|stem| res.algebraic_novikov_rank(s, stem + s)).sum()
}

/// Generators split into the "generic band" (s ≤ n/2, below the h₁ line) and the
/// "η-periodic wedge" (s > n/2, where the h₁-towers pile up), summed over stems.
fn banded_totals(res: &MotivicResolution, n: i32, s_max: i32) -> (usize, usize) {
    let (mut generic, mut wedge) = (0usize, 0usize);
    for stem in 0..=n {
        for s in 0..=s_max {
            let g = res.algebraic_novikov_rank(s, stem + s);
            if 2 * s <= stem {
                generic += g;
            } else {
                wedge += g;
            }
        }
    }
    (generic, wedge)
}

fn main() -> anyhow::Result<()> {
    let n: i32 = std::env::var("MOT_N").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    // h₁-towers march up the n = s diagonal, so to see the towers we want s up to ~n.
    // Give both charts the same box.
    let s: i32 = std::env::var("MOT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(n);
    let max = Bidegree::n_s(n, s);

    let t_sphere = std::time::Instant::now();
    let sphere = MotivicResolution::new(max);
    let t_sphere = t_sphere.elapsed();

    let ceta_descriptor = serde_json::json!({
        "type": "finite dimensional module",
        "name": "Ceta",
        "gens": { "x0": 0, "x2": 2 },
        "actions": ["P(0, 1) x0 = x2"],
    });
    let ceta_module = MotivicResolution::module_from_json(&ceta_descriptor)?;
    let t_ceta = std::time::Instant::now();
    let ceta = MotivicResolution::with_module(ceta_module, max, None);
    let t_ceta = t_ceta.elapsed();

    println!("# box n≤{n}, s≤{s}");
    println!("# sphere resolved in {t_sphere:?};  Cη resolved in {t_ceta:?}");
    println!("#");
    println!("# Generators per FILTRATION row (summed over stems 0..={n}).");
    println!("# High-s rows are where the h₁-towers live — that is the collapse to watch.");
    println!("#   s | sphere | Cη  | sphere−Cη");
    let (mut sph_tot, mut cet_tot) = (0usize, 0usize);
    for row in 0..=s {
        let sg = filt_total(&sphere, row, n);
        let cg = filt_total(&ceta, row, n);
        sph_tot += sg;
        cet_tot += cg;
        // skip empty tails to keep the table short
        if sg == 0 && cg == 0 {
            continue;
        }
        println!("  {row:3} | {sg:6} | {cg:3} | {:+}", sg as i64 - cg as i64);
    }
    println!("#");

    let (sph_gen, sph_wedge) = banded_totals(&sphere, n, s);
    let (cet_gen, cet_wedge) = banded_totals(&ceta, n, s);
    println!("# generic band (2s ≤ n):   sphere {sph_gen:5}, Cη {cet_gen:5}");
    println!(
        "# η-periodic wedge (2s > n): sphere {sph_wedge:5}, Cη {cet_wedge:5}  \
         → Cη removes {:.0}% of the wedge",
        100.0 * (sph_wedge as f64 - cet_wedge as f64) / sph_wedge.max(1) as f64
    );
    println!("#");
    println!(
        "# TOTAL generators in box: sphere {sph_tot}, Cη {cet_tot}  \
         (Cη carries {:.1}%)",
        100.0 * cet_tot as f64 / sph_tot as f64
    );

    Ok(())
}
