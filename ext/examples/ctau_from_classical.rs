//! Compares $\mathrm{Ext}_{A/\tau}(\mathbb{F}_2)$ computed from classical data with a direct resolution.
//!
//! Usage: `ctau_from_classical <max_n> <max_s> [split|untwisted|plan] [direct]`. `split` resolves
//! each $Q_k^\vee$ on its own; `untwisted` uses one resolution of $\mathbb{F}_2$. Prints each
//! (stem, s) where the result disagrees with the direct resolution, then `n,s,w,dim` for every
//! class.
//!
//! `plan` only sizes the coboundaries `untwisted` would reduce, from the backbone's generator
//! counts, and estimates their cost from `RREF_SECONDS` (default 720), the time to reduce a dense
//! $2^{20} \times 2^{20}$ matrix, scaled cubically.
//!
//! With `untwisted` or `plan`, `CTAU_BACKBONE=<dir>` loads (or saves) the resolution of
//! $\mathbb{F}_2$ there, and the time to resolve it is reported apart from the rest.
//! `CTAU_SERIAL_ABOVE=<entries>` sets [`UntwistedOptions::serial_above`].
use std::{sync::Arc, time::Instant};

use ext::algebraic_novikov::{
    Convention, UntwistedOptions, ctau_e2, ctau_e2_direct, ctau_e2_untwisted_with, unit_module,
    untwisted_backbone_range, untwisted_plan,
};

/// Prints a summary of the coboundaries [`untwisted_plan`] lists.
fn print_plan(plan: &[ext::algebraic_novikov::UntwistedMatrix]) {
    let rref_seconds: f64 = std::env::var("RREF_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(720.0);
    let cost = |m: &ext::algebraic_novikov::UntwistedMatrix| {
        let (r, c) = (m.rows as f64, m.cols as f64);
        rref_seconds * r * c * r.min(c) / 2f64.powi(60)
    };
    // By the larger side, in powers of two.
    let mut buckets = std::collections::BTreeMap::<u32, (usize, f64)>::new();
    for m in plan {
        let side = m.rows.max(m.cols);
        let b = usize::BITS - side.leading_zeros();
        let e = buckets.entry(b).or_default();
        e.0 += 1;
        e.1 += cost(m);
    }
    println!("{} coboundaries", plan.len());
    println!("side <= 2^b: count, estimated seconds");
    for (b, (count, secs)) in &buckets {
        println!("  2^{b:<2}: {count:>8} {secs:>14.1}");
    }
    let largest = plan.iter().max_by_key(|m| m.rows * m.cols);
    if let Some(m) = largest {
        println!(
            "largest: {} x {} at k = {}, u = {}, a = {} ({:.1} GiB dense)",
            m.rows,
            m.cols,
            m.k,
            m.u,
            m.a,
            (m.rows * m.cols) as f64 / 8.0 / 2f64.powi(30)
        );
    }
    let total: f64 = plan.iter().map(cost).sum();
    println!("estimated total: {:.1} hours", total / 3600.0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: i32 = args[1].parse().unwrap();
    let max_s: i32 = args[2].parse().unwrap();
    let start = Instant::now();
    let tri = match args.get(3).map(String::as_str) {
        Some(mode @ ("untwisted" | "plan")) => {
            let save = std::env::var_os("CTAU_BACKBONE").map(std::path::PathBuf::from);
            let res =
                ext::nassau::Resolution::new_with_save(Arc::new(unit_module()), save).unwrap();
            res.compute_through_stem(untwisted_backbone_range(max_n, max_s));
            eprintln!("backbone: {:?}", start.elapsed());
            if mode == "plan" {
                print_plan(&untwisted_plan(&res, max_n, max_s));
                return;
            }
            let mut options = UntwistedOptions::default();
            if let Some(v) = std::env::var("CTAU_SERIAL_ABOVE")
                .ok()
                .and_then(|v| v.parse().ok())
            {
                options.serial_above = v;
            }
            let start = Instant::now();
            let tri = ctau_e2_untwisted_with(&res, max_n, max_s, options);
            eprintln!("untwisted complexes: {:?}", start.elapsed());
            tri
        }
        _ => ctau_e2(max_n, max_s, Convention::Xi),
    };
    eprintln!("Ext_P(Q): {:?}", start.elapsed());
    if args.get(4).map(String::as_str) == Some("direct") {
        let start = Instant::now();
        let direct = ctau_e2_direct(max_n, max_s);
        eprintln!("direct A/tau: {:?}", start.elapsed());
        let ours = tri.bigraded();
        let mut keys: Vec<_> = ours.keys().chain(direct.keys()).copied().collect();
        keys.sort();
        keys.dedup();
        let mut bad = 0;
        for k in keys {
            let (a, b) = (
                ours.get(&k).copied().unwrap_or(0),
                direct.get(&k).copied().unwrap_or(0),
            );
            if a != b {
                bad += 1;
                eprintln!("MISMATCH (n, s) = {k:?}: Ext_P(Q) {a}, direct {b}");
            }
        }
        eprintln!(
            "{bad} mismatches over {} classes",
            direct.values().sum::<usize>()
        );
    }
    for ((n, s, w), d) in &tri.0 {
        println!("{n},{s},{w},{d}");
    }
}
