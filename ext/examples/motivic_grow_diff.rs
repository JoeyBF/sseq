//! Localize the grow discrepancy: build box B by growing from a smaller box A on a
//! shared store, and cold (no store), then compare τ-module (differential-derived)
//! and h_i products cell by cell over the report box.
//!
//! Run: `MOT_A=40 MOT_B=50 cargo run --release --features concurrent --example motivic_grow_diff`

use std::{collections::BTreeSet, path::PathBuf};

use ext::motivic::{Gen, MotivicResolution};
use sseq::coordinates::Bidegree;

fn build(a: i32, dir: Option<PathBuf>) -> MotivicResolution {
    MotivicResolution::with_module(
        MotivicResolution::trivial_module(),
        Bidegree::n_s(a, a / 2 + 2),
        dir,
    )
}

fn products(res: &MotivicResolution, n: i32) -> BTreeSet<(i32, i32, usize, i32, i32, usize, u32)> {
    let mut out = BTreeSet::new();
    for i in 0..3 {
        let t = 1i32 << i;
        if t - 1 > n || res.algebraic_novikov_rank(1, t) == 0 {
            continue;
        }
        for (b, terms) in res.motivic_products_by(Gen { s: 1, t, idx: 0 }) {
            if b.t - b.s > n {
                continue;
            }
            for (g, p) in terms {
                out.insert((b.s, b.t, b.idx, g.s, g.t, g.idx, p));
            }
        }
    }
    out
}

fn main() {
    let a: i32 = std::env::var("MOT_A").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let b: i32 = std::env::var("MOT_B").ok().and_then(|v| v.parse().ok()).unwrap_or(50);

    let dir = std::env::temp_dir().join(format!("grow-diff-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let _ = build(a, Some(dir.clone()));
    let grown = build(b, Some(dir.clone()));
    let cold = build(b, None);

    // Is the small box's REPORT region (stem ≤ a, all filtrations) box-stable? I.e.
    // does a cold box A agree with a cold box B on their overlap? If yes, the cells
    // with a < t ≤ ... are already box-independent and could be cached at box A. If
    // no, the region above t = a genuinely changes with the box (edge effect) and
    // there is nothing extra to reuse.
    let cold_a = build(a, None);
    // τ-module (ranks) cold-A vs cold-B over stem ≤ a: does the +1 margin already
    // stabilize ranks even where products are not yet stable?
    let mut cc_tau_diffs = 0;
    let mut first_diff_t = i32::MAX;
    for s in 0..cold_a.max().s() {
        for n in 0..=a {
            let (ta, tb) = (cold_a.tau_module(s, n + s), cold.tau_module(s, n + s));
            if ta.free != tb.free || ta.torsion != tb.torsion {
                cc_tau_diffs += 1;
                first_diff_t = first_diff_t.min(n + s);
            }
        }
    }
    println!(
        "cold-{a} vs cold-{b} τ-module on stem≤{a}: diffs={cc_tau_diffs}{}",
        if cc_tau_diffs > 0 { format!(" (lowest at t={first_diff_t}, max.n={a})") } else { String::new() }
    );
    let (pa, pb_on_a) = (products(&cold_a, a), products(&cold, a));
    println!(
        "cold-{a} vs cold-{b} on stem≤{a}: cold_a={} cold_b={} (differ: only_in_A={}, only_in_B={})",
        pa.len(),
        pb_on_a.len(),
        pa.difference(&pb_on_a).count(),
        pb_on_a.difference(&pa).count()
    );

    // τ-module over the report box.
    let mut tau_diffs = 0;
    for s in 0..grown.max().s() {
        for n in 0..=b {
            let t = n + s;
            let (gtm, ctm) = (grown.tau_module(s, t), cold.tau_module(s, t));
            if gtm.free != ctm.free || gtm.torsion != ctm.torsion {
                tau_diffs += 1;
                if tau_diffs <= 5 {
                    println!(
                        "TAU DIFF (n={n},s={s}): grown free={} tor={:?} | cold free={} tor={:?}",
                        gtm.free, gtm.torsion, ctm.free, ctm.torsion
                    );
                }
            }
        }
    }

    let (gp, cp) = (products(&grown, b), products(&cold, b));
    let only_cold: Vec<_> = cp.difference(&gp).take(8).collect();
    let only_grown: Vec<_> = gp.difference(&cp).take(8).collect();
    println!("\ngrow {a}->{b}: tau_module diffs={tau_diffs}");
    println!("products: grown={} cold={} (in cold not grown={}, in grown not cold={})",
        gp.len(), cp.len(), cp.difference(&gp).count(), gp.difference(&cp).count());
    for x in &only_cold {
        println!("  MISSING from grown: b=({},{})#{} -> ({},{})#{} tau^{}", x.0, x.1, x.2, x.3, x.4, x.5, x.6);
    }
    for x in &only_grown {
        println!("  EXTRA in grown:     b=({},{})#{} -> ({},{})#{} tau^{}", x.0, x.1, x.2, x.3, x.4, x.5, x.6);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
