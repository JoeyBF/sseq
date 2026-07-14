//! Classical benchmark: resolve the sphere (`k` over the full mod-`p` Steenrod
//! algebra) with the generic signature engine vs the generic minimal-resolution
//! engine, on the same box. Shows the signature shortcut helps the *classical*
//! Adams `E₂` too — independent of the motivic story.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=x86-64-v3" cargo run --release --features sig-nassau \
//!     --example nassau_bench -- 2 60 30
//! ```
//! Args: prime, max_n (stem), max_s. Defaults 2, 60, 30.

use std::{sync::Arc, time::Instant};

use algebra::{milnor_algebra::MilnorAlgebra, module::FDModule};
use bivec::BiVec;
use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
    motivic_nassau::SignatureResolution,
    resolution::Resolution,
};
use fp::prime::ValidPrime;
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let p: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let max_n: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(60);
    let max_s: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let prime = ValidPrime::new(p);
    let max_t = max_n + max_s;

    println!("# classical Adams E₂ (sphere over the full mod-{p} Steenrod algebra)");
    println!("# box: stem ≤ {max_n}, s ≤ {max_s}");

    // Signature engine.
    // Shared get_partial_matrix (GPM) instrumentation, sampled per engine.
    use std::sync::atomic::Ordering::Relaxed;

    use algebra::module::homomorphism::{GPM_CALLS, GPM_INPUTS, GPM_NANOS};
    let gpm = || {
        (
            GPM_CALLS.load(Relaxed),
            GPM_INPUTS.load(Relaxed),
            GPM_NANOS.load(Relaxed),
        )
    };
    let gpm_since = |b: (u64, u64, u64)| {
        let n = gpm();
        (n.0 - b.0, n.1 - b.1, n.2 - b.2)
    };

    let g0 = gpm();
    let mut sres = SignatureResolution::new(Arc::new(MilnorAlgebra::new(prime, false)));
    let start = Instant::now();
    sres.compute_through_stem(max_s, max_t);
    let sig_time = start.elapsed();
    let sig_gpm = gpm_since(g0);

    // Generic engine, same rectangle.
    let galg = Arc::new(MilnorAlgebra::new(prime, false));
    let gmod = Arc::new(FDModule::new(
        Arc::clone(&galg),
        "k".to_string(),
        BiVec::from_vec(0, vec![1]),
    ));
    let gcc = Arc::new(FiniteChainComplex::<FDModule<MilnorAlgebra>>::ccdz(gmod));
    let gres = Resolution::new(gcc);
    let g1 = gpm();
    let start = Instant::now();
    // Same stem region as the signature engine (fair comparison).
    gres.compute_through_stem(Bidegree::s_t(max_s, max_t));
    let gen_time = start.elapsed();
    let gen_gpm = gpm_since(g1);

    // Correctness.
    let mut mism = 0;
    for s in 0..=max_s {
        for t in 0..=max_t {
            // Both engines now compute the stem region {n ≤ max_n, s ≤ max_s};
            // only compare there.
            if t - s > max_n {
                continue;
            }
            if sres.number_of_gens_in_bidegree(s, t)
                != gres.number_of_gens_in_bidegree(Bidegree::s_t(s, t))
            {
                mism += 1;
            }
        }
    }
    println!(
        "[1] {}",
        if mism == 0 {
            "CORRECT: signature == generic at every bidegree".to_string()
        } else {
            format!("FAILED: {mism} mismatches")
        }
    );

    // Shrink.
    let records = sres.shrink_records();
    if !records.is_empty() {
        let sum_c: usize = records.iter().map(|r| r.dim_c).sum();
        let sum_e0: usize = records.iter().map(|r| r.dim_e0.max(1)).sum();
        let max_r = records
            .iter()
            .filter(|r| r.dim_e0 > 0)
            .map(|r| r.dim_c as f64 / r.dim_e0 as f64)
            .fold(1.0, f64::max);
        println!(
            "[2] SHRINK dim C / dim E₀C: aggregate {:.1}×, max {max_r:.1}×",
            sum_c as f64 / sum_e0 as f64
        );
    }
    // Master's hand-tuned nassau.rs (p = 2 only).
    let (master, master_gpm) = if p == 2 {
        let mres = ext::utils::construct_nassau("S_2", None)?;
        let g2 = gpm();
        let start = Instant::now();
        // Same stem region as the other two engines (fair region).
        mres.compute_through_stem(Bidegree::s_t(max_s, max_t));
        (Some(start.elapsed()), Some(gpm_since(g2)))
    } else {
        (None, None)
    };

    match master {
        Some(m) => println!(
            "[3] TIMING signature {:.3}s   generic {:.3}s   master-nassau.rs {:.3}s",
            sig_time.as_secs_f64(),
            gen_time.as_secs_f64(),
            m.as_secs_f64()
        ),
        None => println!(
            "[3] TIMING signature {:.3}s   generic {:.3}s",
            sig_time.as_secs_f64(),
            gen_time.as_secs_f64()
        ),
    }
    let (sig_steps, plain) = sres.stats();
    println!("[4] STEPS signature-shortcut {sig_steps}   plain {plain}");

    let [
        mask_calls,
        basis_sig_calls,
        mask_ns,
        partial_ns,
        linalg_ns,
        itersig_ns,
    ] = ext::motivic_nassau::prof::snapshot();
    let ms = |ns: u64| ns as f64 / 1e6;
    println!(
        "[5] PROFILE (signature engine, {:.1}s total):\n    signature_mask: {:.0} ms   \
         ({mask_calls} calls, {basis_sig_calls} basis_element_signature calls)\n    \
         get_partial_matrix: {:.0} ms\n    linalg (row_reduce/kernel/qi): {:.0} ms\n    \
         iter_signatures: {:.0} ms",
        sig_time.as_secs_f64(),
        ms(mask_ns),
        ms(partial_ns),
        ms(linalg_ns),
        ms(itersig_ns),
    );

    // get_partial_matrix (GPM), per engine — the apples-to-apples comparison.
    let gpm_line = |name: &str, (calls, inputs, ns): (u64, u64, u64), total: f64| {
        let t = ns as f64 / 1e9;
        println!(
            "    {name:>16}: {t:6.3}s in GPM ({:4.0}% of run)   {calls:>7} calls   {inputs:>9} \
             inputs   {:.2} µs/call",
            100.0 * t / total,
            if calls > 0 {
                ns as f64 / 1e3 / calls as f64
            } else {
                0.0
            },
        );
    };
    println!("[6] get_partial_matrix, per engine:");
    gpm_line("signature", sig_gpm, sig_time.as_secs_f64());
    gpm_line("generic", gen_gpm, gen_time.as_secs_f64());
    if let (Some(m), Some(mg)) = (master, master_gpm) {
        gpm_line("master-nassau", mg, m.as_secs_f64());
    }

    if mism > 0 {
        std::process::exit(1);
    }
    Ok(())
}
