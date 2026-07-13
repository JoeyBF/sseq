//! M5 — validation & benchmark for the motivic-Nassau signature engine.
//!
//! Resolves `k` over `A_C/τ` on a `(max_s, max_t)` rectangle with **both** the
//! signature engine and the generic engine, then:
//!   1. asserts the signature engine's per-`(s,t)` generator ranks equal the
//!      generic engine's, bidegree for bidegree (basis-independent correctness);
//!   2. reports the `dim C / dim E₀C` shrink factors (the speedup lever);
//!   3. reports the resolution-phase wall-clock of each engine on the same box;
//!   4. reports how many steps used a signature shortcut vs. fell back.
//!
//! Run with the mandatory flags:
//! ```text
//! RUSTFLAGS="-C target-cpu=x86-64-v3" cargo run --release --features sig-nassau \
//!     --example motivic_nassau_validate -- 40 22
//! ```
//! Args are `max_n` (stem) and `max_s`; defaults 40 and 22 (the golden box).

use std::{sync::Arc, time::Instant};

use algebra::{module::FDModule, motivic::CTauAlgebra};
use bivec::BiVec;
use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
    motivic_nassau::SignatureResolution,
    resolution::Resolution,
};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let max_n: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);
    let max_s: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(22);
    let max_t = max_n + max_s;

    println!("# motivic-Nassau validation  (box: max_n={max_n}, max_s={max_s}, max_t={max_t})");

    // --- Signature engine ---
    let mut sres = SignatureResolution::new();
    let start = Instant::now();
    sres.compute_through_stem(max_s, max_t);
    let sig_time = start.elapsed();

    // --- Generic engine (same rectangle) ---
    let galg = Arc::new(CTauAlgebra::new());
    let gmod = Arc::new(FDModule::new(
        Arc::clone(&galg),
        "k".to_string(),
        BiVec::from_vec(0, vec![1]),
    ));
    let gcc = Arc::new(FiniteChainComplex::<FDModule<CTauAlgebra>>::ccdz(gmod));
    let gres = Resolution::new(gcc);
    let start = Instant::now();
    gres.compute_through_bidegree(Bidegree::s_t(max_s, max_t));
    let gen_time = start.elapsed();

    // --- 1. Correctness: rank-for-rank equality ---
    let mut mismatches = 0usize;
    let mut total = 0usize;
    for s in 0..=max_s {
        for t in 0..=max_t {
            let want = gres.number_of_gens_in_bidegree(Bidegree::s_t(s, t));
            let got = sres.number_of_gens_in_bidegree(s, t);
            if got != want {
                if mismatches < 20 {
                    println!("  MISMATCH (s={s}, t={t}): signature={got} generic={want}");
                }
                mismatches += 1;
            }
            total += got;
        }
    }
    if mismatches == 0 {
        println!(
            "[1] CORRECT: signature ranks == generic ranks at every bidegree ({total} generators)"
        );
    } else {
        println!("[1] FAILED: {mismatches} bidegree rank mismatches");
    }

    // --- 2. Shrink factors ---
    let records = sres.shrink_records();
    if records.is_empty() {
        println!("[2] no signature steps (box too small / below every vanishing line)");
    } else {
        let sum_c: usize = records.iter().map(|r| r.dim_c).sum();
        let sum_e0: usize = records.iter().map(|r| r.dim_e0.max(1)).sum();
        let mut ratios: Vec<f64> = records
            .iter()
            .filter(|r| r.dim_e0 > 0)
            .map(|r| r.dim_c as f64 / r.dim_e0 as f64)
            .collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let max_r = ratios.last().copied().unwrap_or(1.0);
        let med_r = ratios.get(ratios.len() / 2).copied().unwrap_or(1.0);
        // The single largest-C signature step, where the shrink matters most.
        let biggest = records.iter().max_by_key(|r| r.dim_c).unwrap();
        println!(
            "[2] SHRINK  dim C / dim E₀C:  aggregate {:.1}×  (Σ dimC={sum_c}, Σ dimE₀={sum_e0}),  \
             median {med_r:.1}×,  max {max_r:.1}×",
            sum_c as f64 / sum_e0 as f64
        );
        println!(
            "    largest step: (s={}, t={}) B={}  dim C={} → dim E₀C={}  ({:.1}×)",
            biggest.s,
            biggest.t,
            biggest.b,
            biggest.dim_c,
            biggest.dim_e0,
            biggest.dim_c as f64 / biggest.dim_e0.max(1) as f64
        );
    }

    // --- 3. Timing ---
    println!(
        "[3] TIMING  signature engine {:.3}s   generic engine {:.3}s   (same {}×{} box)",
        sig_time.as_secs_f64(),
        gen_time.as_secs_f64(),
        max_s,
        max_t
    );

    // --- 4. Shortcut vs fallback ---
    let (sig_steps, fallbacks) = sres.stats();
    println!("[4] STEPS  signature-shortcut {sig_steps}   fallback (plain) {fallbacks}");

    if mismatches > 0 {
        std::process::exit(1);
    }
    Ok(())
}
