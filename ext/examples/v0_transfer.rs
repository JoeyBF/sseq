//! Computes $\mathrm{Ext}_P(Q)$ by transferring the $v_0$-Bockstein perturbation, and optionally
//! compares with the direct reduction.
//!
//! Usage: `v0_transfer <max_n> <max_s> [direct]`, with `CTAU_BACKBONE=<dir>` to load or save the
//! resolution of $\mathbb{F}_2$, `CHECK_CSV=<file>` to compare with rows `stem,s,u,dim`, and
//! `CTAU_PSI_BANDS=<n>` to take the weights in `n` bands, trading time for memory.
//! Prints the number of classes and the time of each method.
use std::{sync::Arc, time::Instant};

use ext::{
    algebraic_novikov::{ctau_e2_untwisted_on, unit_module, untwisted_backbone_range},
    v0_transfer::ext_q_by_v0_transfer_in_bands,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: i32 = args[1].parse().unwrap();
    let max_s: i32 = args[2].parse().unwrap();
    let save = std::env::var_os("CTAU_BACKBONE").map(std::path::PathBuf::from);
    let res = ext::nassau::Resolution::new_with_save(Arc::new(unit_module()), save).unwrap();
    res.compute_through_stem(untwisted_backbone_range(max_n, max_s));

    let start = Instant::now();
    let bands = std::env::var("CTAU_PSI_BANDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let ours = ext_q_by_v0_transfer_in_bands(&res, max_n, max_s, bands);
    println!(
        "transfer: {} classes ({:?})",
        ours.values().sum::<usize>(),
        start.elapsed()
    );
    // CHECK_CSV=<file> compares with rows `stem,s,u,dim` computed before.
    if let Ok(path) = std::env::var("CHECK_CSV") {
        let text = std::fs::read_to_string(path).unwrap();
        let mut theirs = std::collections::BTreeMap::new();
        for line in text.lines() {
            let f: Vec<i64> = line.split(',').map(|x| x.trim().parse().unwrap()).collect();
            let (n, s, u, d) = (f[0] as i32, f[1] as i32, f[2] as i32, f[3] as usize);
            if n <= max_n && s <= max_s && d > 0 {
                theirs.insert((n, s, u), d);
            }
        }
        println!(
            "against the file: {} classes; equal: {}",
            theirs.values().sum::<usize>(),
            theirs == ours
        );
    }
    if args.get(3).map(String::as_str) == Some("direct") {
        let start = Instant::now();
        let direct: std::collections::BTreeMap<_, _> = ctau_e2_untwisted_on(&res, max_n, max_s)
            .0
            .into_iter()
            .collect();
        println!(
            "direct: {} classes ({:?}); equal: {}",
            direct.values().sum::<usize>(),
            start.elapsed(),
            direct == ours
        );
    }
}
