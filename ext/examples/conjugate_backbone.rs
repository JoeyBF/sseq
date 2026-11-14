//! Conjugates a saved backbone: replaces each differential coefficient by its antipode, as
//! `ext::conjugate::ConjugatedBackbone`, and reports the cost by stem.
//!
//! Usage: `conjugate_backbone <max_stem> <max_s> [out_dir]`, with `CTAU_BACKBONE=<dir>` the saved
//! resolution of $\mathbb{F}_2$, read through classical stem `max_stem` and filtration
//! `max_s + 1`. Prints, for each band of stems, the Milnor terms read, the antipode terms summed
//! before cancelling, the terms kept and the summed time of the bidegrees, then the totals, the
//! wall time and the peak memory. With `out_dir`, writes the result there.
use std::{sync::Arc, time::Instant};

use ext::{algebraic_novikov::unit_module, conjugate::ConjugatedBackbone};
use sseq::coordinates::Bidegree;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: i32 = args[1].parse().unwrap();
    let max_s: i32 = args[2].parse().unwrap();
    let save = std::env::var_os("CTAU_BACKBONE").map(std::path::PathBuf::from);
    let res = ext::nassau::Resolution::new_with_save(Arc::new(unit_module()), save).unwrap();
    let start = Instant::now();
    res.compute_through_stem(Bidegree::n_s(max_n, max_s + 1));
    println!("backbone: {:?}", start.elapsed());

    let start = Instant::now();
    let (conj, stats) = ConjugatedBackbone::compute(&res, max_s, max_n + max_s + 1);
    let wall = start.elapsed();
    let band = 20;
    let mut bands: std::collections::BTreeMap<i32, (usize, usize, usize, f64)> = Default::default();
    for (&n, &(a, b, c, t)) in &stats.by_stem {
        let e = bands.entry(n / band * band).or_default();
        e.0 += a;
        e.1 += b;
        e.2 += c;
        e.3 += t;
    }
    println!("stems      Milnor terms   summed         kept           kept/read  seconds");
    for (n, (a, b, c, t)) in bands {
        println!(
            "{n:>4}-{:<4} {a:>14} {b:>14} {c:>14} {:>9.2} {t:>8.1}",
            n + band - 1,
            c as f64 / a.max(1) as f64
        );
    }
    let (a, b, c, t) = stats.total();
    println!(
        "total      {a:>14} {b:>14} {c:>14} {:>9.2} {t:>8.1}",
        c as f64 / a.max(1) as f64
    );
    let peak = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM"))
                .map(|l| l.split_whitespace().nth(1).unwrap().to_string())
        })
        .unwrap_or_default();
    println!("wall {wall:?}, peak {peak} kB, {} terms", conj.terms());
    if let Some(dir) = args.get(3) {
        let start = Instant::now();
        conj.save(std::path::Path::new(dir)).unwrap();
        println!("saved to {dir} ({:?})", start.elapsed());
    }
}
