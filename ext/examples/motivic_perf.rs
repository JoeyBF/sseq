//! Scaling measurement for the motivic product lift — the pipeline's dominant cost
//! at large stems. For a range of boxes it times a single h₂ product lift and splits
//! the cost into its drivers:
//!
//!   * `corrections`  — total τ-adic correction rounds ([`TAULIFT_ITERS`]): the
//!                      *breadth × depth* of the order-by-order lift.
//!   * `product_ops`  — A_C Milnor products performed, with cache hit-rate: the
//!                      per-round work.
//!   * `µs/corr`      — wall time per correction round: isolates whether the blow-up
//!                      is more rounds (depth/breadth) or heavier rounds (degree).
//!
//! Run: `cargo run --release --example motivic_perf` (add `--features concurrent` to
//! match production parallelism). Boxes mirror the n*.log runs: `max_s ≈ n/2 + 2`.

use std::{
    sync::atomic::Ordering::Relaxed,
    time::Instant,
};

use algebra::motivic::milnor::{PRODUCT_HITS, PRODUCT_MISSES};
use ext::motivic::{Gen, MotivicResolution, TAULIFT_ITERS};
use sseq::coordinates::Bidegree;

fn main() {
    let h2 = Gen { s: 1, t: 4, idx: 0 };
    // (max_n, max_s) — max_s tracks the n*.log runs (~n/2 + 2).
    let boxes: &[(i32, i32)] = &[(40, 22), (50, 27), (60, 32), (70, 37)];

    println!(
        "{:>4} {:>4} | {:>9} | {:>12} | {:>11} | {:>13} {:>7} | {:>9}",
        "n", "s", "resolve", "lift_product", "corrections", "product_ops", "hit%", "µs/corr"
    );
    for &(n, s) in boxes {
        let t0 = Instant::now();
        let res = MotivicResolution::new(Bidegree::n_s(n, s));
        let resolve = t0.elapsed();

        let (h, m, c) = (
            PRODUCT_HITS.load(Relaxed),
            PRODUCT_MISSES.load(Relaxed),
            TAULIFT_ITERS.load(Relaxed),
        );
        let t1 = Instant::now();
        let prods = res.motivic_products_by(h2);
        let lift = t1.elapsed();

        let hits = PRODUCT_HITS.load(Relaxed) - h;
        let miss = PRODUCT_MISSES.load(Relaxed) - m;
        let corr = TAULIFT_ITERS.load(Relaxed) - c;
        let ops = hits + miss;
        println!(
            "{n:>4} {s:>4} | {:>9.2?} | {:>12.2?} | {corr:>11} | {ops:>12} | {:>6.1}% | {:>7.1} | ({} products)",
            resolve,
            lift,
            100.0 * hits as f64 / ops.max(1) as f64,
            lift.as_micros() as f64 / corr.max(1) as f64,
            prods.len(),
        );
    }
}
