//! Height-1 chromatic smoke test (see `ext/docs/chromatic-computations.md`).
//!
//! Resolves the trivial module `F_2` over `gr S(1) = u(L(1)) = Lambda(x_1) (x) F_2[x_2]`, the
//! associated graded of the height-1 Morava stabilizer algebra, and checks the additive Ext chart
//! against Ravenel's known answer (green book Thm 6.3.21(a); at height 1 the May SS collapses, so
//! this is already `H^*(S(1))`):
//!
//! ```text
//! Ext_{gr S(1)}(F_2, F_2) = P(h_1) (x) E(rho_1) = F_2[h_1] (x) Lambda(rho_1),
//!     h_1 = h_{1,0} in (s, t) = (1, 2),    rho_1 = h_{2,0} in (s, t) = (1, 6).
//! ```
//!
//! This is a self-contained validation that the generic resolution machinery accepts a brand-new
//! `Algebra` and returns the right chart. It uses only the core resolution engine (no field-trick
//! machinery). Run with e.g. `cargo run --example chromatic_grs1 -- 32 18`.

use std::sync::Arc;

use algebra::{
    MoravaStabilizerAlgebra,
    module::{FDModule, homomorphism::FullModuleHomomorphism},
};
use bivec::BiVec;
use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
    resolution::Resolution,
};
use sseq::coordinates::Bidegree;

/// `dim Ext^{s, t}` predicted by `F_2[h_1] (x) Lambda(rho_1)`, `|h_1| = 2`, `|rho_1| = 6`: the number
/// of monomials `h_1^a rho_1^b` (`a >= 0`, `b in {0, 1}`) with `a + b = s` and `2a + 6b = t`.
fn predicted_chart(max_s: usize, max_t: usize) -> Vec<Vec<usize>> {
    let mut f = vec![vec![0usize; max_t + 1]; max_s + 1];
    for (s, row) in f.iter_mut().enumerate() {
        // b = 0: the class h_1^s at (s, 2s).
        let t = 2 * s;
        if t <= max_t {
            row[t] += 1;
        }
        // b = 1: the class h_1^{s-1} rho_1 at (s, 2(s-1) + 6) = (s, 2s + 4), for s >= 1.
        if s >= 1 {
            let t = 2 * s + 4;
            if t <= max_t {
                row[t] += 1;
            }
        }
    }
    f
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let mut args = std::env::args().skip(1);
    let max_n: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_s: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(18);

    // gr S(1) at p = 2.
    let algebra = Arc::new(MoravaStabilizerAlgebra::new(fp::prime::TWO, 1));

    // The trivial module F_2: one basis element in degree 0, all positive-degree actions zero.
    let k = Arc::new(FDModule::new(
        Arc::clone(&algebra),
        "F_2".to_string(),
        BiVec::from_vec(0, vec![1]),
    ));

    // Minimal free resolution of F_2 over gr S(1).
    let cc: FiniteChainComplex<
        FDModule<MoravaStabilizerAlgebra>,
        FullModuleHomomorphism<FDModule<MoravaStabilizerAlgebra>>,
    > = FiniteChainComplex::ccdz(k);
    let resolution = Arc::new(Resolution::new(Arc::new(cc)));

    let max = Bidegree::n_s(max_n, max_s);
    resolution.compute_through_stem(max);

    println!("Ext_{{gr S(1)}}(F_2, F_2), computed chart (rows = s, cols = stem n):");
    println!("{}", resolution.graded_dimension_string());

    // Validate against F_2[h_1] (x) Lambda(rho_1).
    let max_t = (max_n + max_s) as usize;
    let predicted = predicted_chart(max_s as usize, max_t);

    let mut mismatches = Vec::new();
    for b in resolution.iter_stem() {
        let computed = resolution.number_of_gens_in_bidegree(b);
        let s = b.s() as usize;
        let t = b.t() as usize;
        let expected = if s <= max_s as usize && t <= max_t {
            predicted[s][t]
        } else {
            0
        };
        if computed != expected {
            mismatches.push((b.s(), b.t(), b.n(), computed, expected));
        }
    }

    if mismatches.is_empty() {
        println!("\nPASS: every computed bidegree matches F_2[h_1] (x) Lambda(rho_1).");
    } else {
        println!(
            "\nFAIL: {} bidegree(s) disagree with the prediction:",
            mismatches.len()
        );
        println!("  s   t   n   computed  expected");
        for (s, t, n, computed, expected) in &mismatches {
            println!("  {s:<3} {t:<3} {n:<3} {computed:<9} {expected}");
        }
        anyhow::bail!("chart does not match F_2[h_1] (x) Lambda(rho_1)");
    }

    // The two cohomology generators (both filtration one).
    println!("\nGenerators of Ext_{{gr S(1)}}(F_2, F_2) = F_2[h_1] (x) Lambda(rho_1):");
    println!("  h_1   = h_{{1,0}} = [t_1]: (s, t) = (1, 2),  stem n = 1   (polynomial)");
    println!("  rho_1 = h_{{2,0}} = [t_2]: (s, t) = (1, 6),  stem n = 5   (exterior)");

    Ok(())
}
