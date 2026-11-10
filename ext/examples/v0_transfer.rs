//! Computes $\mathrm{Ext}_P(Q)$ by transferring the $v_0$-Bockstein perturbation, and optionally
//! compares with the direct reduction.
//!
//! Usage: `v0_transfer <max_n> <max_s> [direct]`, with `CTAU_BACKBONE=<dir>` to load or save the
//! resolution of $\mathbb{F}_2$. Prints the number of classes and the time of each method.
use std::{sync::Arc, time::Instant};

use ext::{
    algebraic_novikov::{ctau_e2_untwisted_on, unit_module, untwisted_backbone_range},
    v0_transfer::ext_q_by_v0_transfer,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: i32 = args[1].parse().unwrap();
    let max_s: i32 = args[2].parse().unwrap();
    let save = std::env::var_os("CTAU_BACKBONE").map(std::path::PathBuf::from);
    let res = ext::nassau::Resolution::new_with_save(Arc::new(unit_module()), save).unwrap();
    res.compute_through_stem(untwisted_backbone_range(max_n, max_s));

    let start = Instant::now();
    let ours = ext_q_by_v0_transfer(&res, max_n, max_s);
    println!(
        "transfer: {} classes ({:?})",
        ours.values().sum::<usize>(),
        start.elapsed()
    );
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
