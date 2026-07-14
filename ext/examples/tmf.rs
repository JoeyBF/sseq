//! tmf via the signature engine: resolve `k` over `A(2)` and print the Adams
//! `E₂ = Ext_{A(2)}(F₂, F₂)` chart (the tmf Adams `E₂`).
//!
//! `A(2)` is a *finite* ambient, so the engine takes plain steps (the full-algebra
//! vanishing-line shortcut does not apply); the point here is that the same engine
//! resolves over a finite sub-Hopf-algebra and reproduces the famous tmf chart.
//! Cross-checked against the generic engine over `A(2)`.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=x86-64-v3" cargo run --release --features sig-nassau \
//!     --example tmf -- 30 16
//! ```
//! Args are `max_n` (stem) and `max_s`; defaults 30 and 16.

use std::sync::Arc;

use algebra::{
    milnor_algebra::{MilnorAlgebra, MilnorProfile},
    module::FDModule,
};
use bivec::BiVec;
use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
    motivic_nassau::SignatureResolution,
    resolution::Resolution,
};
use sseq::coordinates::Bidegree;

fn a2() -> MilnorAlgebra {
    // A(2): ξ_1 < 2³, ξ_2 < 2², ξ_3 < 2¹ (dim 64).
    MilnorAlgebra::new_with_profile(
        fp::prime::TWO,
        MilnorProfile {
            q_part: !0,
            p_part: vec![3, 2, 1],
            truncated: true,
        },
        false,
    )
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let max_n: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let max_s: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(16);
    let max_t = max_n + max_s;

    // Signature engine over A(2).
    let mut sres = SignatureResolution::new(Arc::new(a2()));
    sres.compute_through_stem(max_s, max_t);

    // Generic engine over A(2), for the cross-check.
    let galg = Arc::new(a2());
    let gmod = Arc::new(FDModule::new(
        Arc::clone(&galg),
        "k".to_string(),
        BiVec::from_vec(0, vec![1]),
    ));
    let gcc = Arc::new(FiniteChainComplex::<FDModule<MilnorAlgebra>>::ccdz(gmod));
    let gres = Resolution::new(gcc);
    gres.compute_through_stem(Bidegree::s_t(max_s, max_t));

    let mut mismatches = 0;
    for b in gres.iter_stem() {
        if b.t() > max_t {
            continue;
        }
        if sres.number_of_gens_in_bidegree(b.s(), b.t()) != gres.number_of_gens_in_bidegree(b) {
            mismatches += 1;
        }
    }
    println!("# tmf = Ext_A(2)(F₂,F₂)   box: stem ≤ {max_n}, s ≤ {max_s}");
    println!(
        "# vs generic engine over A(2): {}",
        if mismatches == 0 {
            "MATCH at every bidegree".to_string()
        } else {
            format!("{mismatches} MISMATCHES")
        }
    );
    let (sig, plain) = sres.stats();
    println!("# steps: signature-shortcut {sig}, plain {plain}  (A(2) finite ⇒ plain)\n");

    // ASCII chart: rows = filtration s (high at top), cols = stem n. A rank r>0 is
    // printed as its digit (or `·` for 1), blank for 0.
    for s in (0..=max_s).rev() {
        print!("{s:2} ");
        for n in 0..=max_n {
            let r = sres.number_of_gens_in_bidegree(s, n + s);
            if r == 0 {
                print!(" ");
            } else if r == 1 {
                print!("·");
            } else {
                print!("{r}");
            }
        }
        println!();
    }
    print!("   ");
    for n in 0..=max_n {
        print!("{}", if n % 10 == 0 { '|' } else { ' ' });
    }
    println!("\n   0         10        20        30   (stem)");

    if mismatches > 0 {
        std::process::exit(1);
    }
    Ok(())
}
