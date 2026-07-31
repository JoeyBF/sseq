//! Grow-the-box benchmark for the motivic save cache (SAVE-INTEGRATION Phases 1–3).
//!
//! Builds the motivic $E_2$ at a sweep of stems on a **shared** save store and, for
//! each, reports how much of the differential lift and the product lifts were reused
//! from disk vs recomputed. A correctly-caching pipeline should reuse essentially
//! the whole previous box and pay only for the new strip.
//!
//! Run (from `ext/`):
//! ```text
//! MOT_SAVE=/tmp/mot-grow cargo run --release --features concurrent --example motivic_grow_bench
//! ```
//! Env: `MOT_SAVE` (required, cache dir — reused across the sweep), `MOT_NS`
//! (comma-separated stems, default `40,50,60,70,80,90,100`), `MOT_MAXS` (fixed
//! filtration bound, default `52`).

use std::{path::PathBuf, sync::atomic::Ordering, time::Instant};

use ext::motivic::{
    Gen, LIFT_CACHE_LOADS, LIFT_CELLS_REUSED, MotivicResolution, PRODUCT_CELLS_REUSED,
};
use sseq::coordinates::Bidegree;

fn main() {
    // `MOT_SAVE` = cache dir (shared across the sweep). Empty/unset ⇒ no store (a
    // pure-compute baseline, for isolating the store I/O from the math).
    let dir: Option<PathBuf> = std::env::var("MOT_SAVE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    // A natural chart uses filtration up to ≈ the vanishing line s ≈ n/2 (Isaksen's
    // n=100 chart runs to s=52). `MOT_MAXS` pins a fixed bound instead; scaling it
    // with n keeps every box a realistic chart rather than an over-tall one.
    let fixed_s: Option<i32> = std::env::var("MOT_MAXS").ok().and_then(|v| v.parse().ok());
    let max_s_of = |n: i32| fixed_s.unwrap_or(n / 2 + 2);
    let ns: Vec<i32> = std::env::var("MOT_NS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![40, 50, 60, 70, 80, 90, 100]);

    println!(
        "# motivic grow-the-box bench: max_s={}, store={}",
        fixed_s.map_or_else(|| "n/2+2".to_string(), |s| s.to_string()),
        dir.as_deref().map_or("(none)".into(), |d| d.display().to_string())
    );
    println!(
        "# {:>4} {:>4}  {:>9}  {:>9}  {:>9}  {:>12}  {:>12}  {:>7}",
        "n", "s", "build_s", "prods_s", "total_s", "lift_reused", "prod_reused", "loads"
    );

    for n in ns {
        let max_s = max_s_of(n);
        let lift_r0 = LIFT_CELLS_REUSED.load(Ordering::Relaxed);
        let prod_r0 = PRODUCT_CELLS_REUSED.load(Ordering::Relaxed);
        let loads0 = LIFT_CACHE_LOADS.load(Ordering::Relaxed);

        // Build (resolution + weights + differential lift), reusing the store.
        let t_build = Instant::now();
        let res = MotivicResolution::with_module(
            MotivicResolution::trivial_module(),
            Bidegree::n_s(n, max_s),
            dir.clone(),
        );
        let build_s = t_build.elapsed().as_secs_f64();

        // Mirror the chart's product workload: the filtration-one Hopf maps h0,h1,h2.
        let t_prod = Instant::now();
        let mut n_products = 0usize;
        // Order-independent fingerprint of every product a·b = Σ (target, τ-power),
        // so a grown box can be checked byte-for-byte against a cold one.
        let mut fp: u64 = 0;
        for i in 0..3 {
            let t = 1i32 << i;
            if t - 1 > n || res.algebraic_novikov_rank(1, t) == 0 {
                continue;
            }
            let hg = Gen { s: 1, t, idx: 0 };
            for (b, terms) in res.motivic_products_by(hg) {
                n_products += 1;
                for (tgt, power) in terms {
                    // Only fingerprint products inside the report box (stem ≤ n): the
                    // margin is intentionally not comparable across boxes.
                    if b.t - b.s <= n {
                        let h = (i as u64)
                            .wrapping_mul(0x9E3779B97F4A7C15)
                            .rotate_left(5)
                            ^ ((b.s as u64) << 40 ^ (b.t as u64) << 24 ^ (b.idx as u64) << 8)
                            ^ ((tgt.s as u64) << 48 ^ (tgt.t as u64) << 16 ^ (tgt.idx as u64) << 4
                                ^ power as u64)
                                .wrapping_mul(0xD1B54A32D192ED03);
                        fp = fp.wrapping_add(h);
                    }
                }
            }
        }
        let prods_s = t_prod.elapsed().as_secs_f64();

        let lift_reused = LIFT_CELLS_REUSED.load(Ordering::Relaxed) - lift_r0;
        let prod_reused = PRODUCT_CELLS_REUSED.load(Ordering::Relaxed) - prod_r0;
        let loads = LIFT_CACHE_LOADS.load(Ordering::Relaxed) - loads0;

        println!(
            "  {n:>4} {max_s:>4}  {build_s:>9.1}  {prods_s:>9.1}  {:>9.1}  {lift_reused:>12}  \
             {prod_reused:>12}  {loads:>7}   (n_products={n_products}, fp={fp:016x})",
            build_s + prods_s
        );
    }
}
