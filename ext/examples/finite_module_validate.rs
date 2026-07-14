//! Validate the signature engine on **finite modules** (not just `k`).
//!
//! The signature machinery (`step_general`, `s ≥ 2`) is intrinsic to the free
//! `A`-modules and their `A`-linear differentials — it never inspects what is
//! being resolved. So the same engine should resolve any bounded module `M`; the
//! only module-specific part is the seed (`step0`/`step1`). This resolves `M`
//! with both the signature engine and the generic minimal-resolution engine over
//! the full mod-2 Steenrod algebra and checks the `Ext` ranks agree bidegree for
//! bidegree.
//!
//! ```text
//! cargo run --release --features sig-nassau --example finite_module_validate -- C2v14 30 20
//! ```
//! Args: module name (default `C2v14`), max_n (stem), max_s.

use std::sync::Arc;

use algebra::{milnor_algebra::MilnorAlgebra, module::FDModule};
use ext::{
    chain_complex::{FiniteChainComplex, FreeChainComplex},
    motivic_nassau::SignatureResolution,
    resolution::Resolution,
    utils::load_module_json,
};
use fp::prime::TWO;
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "C2v14".to_string());
    let max_n: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let max_s: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let max_t = max_n + max_s;

    let json = load_module_json(&name)?;
    let algebra = Arc::new(MilnorAlgebra::new(TWO, false));
    // FDModule::from_json reads gens/actions and ignores the `cofiber` key, so this
    // is the base finite module M (the cofiber Yoneda complex is a separate case).
    let module = Arc::new(FDModule::from_json(Arc::clone(&algebra), &json)?);

    println!("# finite-module validation: Ext_A(M, k) for M = {name}");
    println!("# stem region: n ≤ {max_n}, s ≤ {max_s}");

    // Generic engine.
    let gcc = Arc::new(FiniteChainComplex::<FDModule<MilnorAlgebra>>::ccdz(Arc::clone(&module)));
    let gres = Resolution::new(gcc);
    gres.compute_through_stem(Bidegree::s_t(max_s, max_t));

    // Signature engine, same module.
    let mut sres = SignatureResolution::from_module(Arc::clone(&module));
    sres.compute_through_stem(max_s, max_t);

    // Rank-for-rank over the shared stem region.
    let mut mism = 0usize;
    let mut total = 0usize;
    for s in 0..=max_s {
        for t in 0..=max_t {
            if t - s > max_n {
                continue;
            }
            let want = gres.number_of_gens_in_bidegree(Bidegree::s_t(s, t));
            let got = sres.number_of_gens_in_bidegree(s, t);
            if got != want {
                if mism < 20 {
                    println!("  MISMATCH (s={s}, t={t}): signature={got} generic={want}");
                }
                mism += 1;
            }
            total += got;
        }
    }
    println!(
        "[1] {}",
        if mism == 0 {
            format!("CORRECT: signature ranks == generic ranks at every bidegree ({total} generators)")
        } else {
            format!("FAILED: {mism} bidegree rank mismatches")
        }
    );

    let records = sres.shrink_records();
    if records.is_empty() {
        println!("[2] no signature steps in this range (below every vanishing line)");
    } else {
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
    let (sig_steps, plain) = sres.stats();
    println!("[3] STEPS signature-shortcut {sig_steps}   plain {plain}");

    if mism > 0 {
        std::process::exit(1);
    }
    Ok(())
}
