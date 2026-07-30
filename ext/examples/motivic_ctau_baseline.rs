//! M0 baseline harness for the "motivic Nassau" spike.
//!
//! Resolves the trivial module `k = F₂` over `A_C/τ` (the [`CTauAlgebra`]) with the
//! **generic** minimal-resolution engine, and prints the per-`(s, t)` generator
//! ranks together with the per-`(s, t, w)` weight-graded ranks. This is the golden
//! fixture the signature-filtration engine (`motivic_nassau`) is validated against:
//! rank-for-rank equality on the same box is the primary correctness gate (M5).
//!
//! Run with the mandatory build flags:
//! ```text
//! RUSTFLAGS="-C target-cpu=x86-64-v3" cargo run --release --example motivic_ctau_baseline -- 30 16
//! ```
//! The two positional args are `max_n` (stem) and `max_s`; they default to 30 and 16.

use std::{sync::Arc, time::Instant};

use algebra::{module::FDModule, motivic::CTauAlgebra};
use bivec::BiVec;
use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
    resolution::Resolution,
};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let max_n: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);
    let max_s: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(22);

    let algebra = Arc::new(CTauAlgebra::new());

    // The trivial module k = F₂ concentrated in degree 0.
    let module = Arc::new(FDModule::new(
        Arc::clone(&algebra),
        "k".to_string(),
        BiVec::from_vec(0, vec![1]),
    ));
    let cc = Arc::new(FiniteChainComplex::<FDModule<CTauAlgebra>>::ccdz(module));
    let res = Resolution::new(cc);

    // max t for a stem-n, filtration-s box.
    let max = Bidegree::s_t(max_s, max_n + max_s);

    let start = Instant::now();
    res.compute_through_stem(max);
    let elapsed = start.elapsed();

    // Per-(s, t) generator ranks, in the golden-fixture format the validator reads:
    //   s t rank
    println!("# GENERATOR RANKS  (s t rank)  over A_C/τ, generic engine");
    println!("# box: max_n={max_n} max_s={max_s}");
    let mut total = 0usize;
    for b in res.iter_stem() {
        let rank = res.number_of_gens_in_bidegree(b);
        if rank > 0 {
            println!("{} {} {}", b.s(), b.t(), rank);
            total += rank;
        }
    }
    println!("# total generators: {total}");
    println!("# resolution wall-clock: {:.3}s", elapsed.as_secs_f64());
    Ok(())
}
