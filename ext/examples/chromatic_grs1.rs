//! Height-1 chromatic smoke test (see `ext/docs/chromatic-field-trick.md`, phase 1).
//!
//! Resolves the trivial module `F_2` over `gr S(1) = u(L(1)) = Lambda(x_1, x_2, ...)`, the
//! associated graded of the height-1 Morava stabilizer algebra, and checks the additive Ext chart
//! against the known answer
//!
//! ```text
//! Ext_{gr S(1)}(F_2, F_2) = F_2[h_1, h_2, h_3, ...],   h_i in (s, t) = (1, 2 (2^i - 1)).
//! ```
//!
//! This exercises the generic resolution machinery on a brand-new `Algebra`/`Bialgebra`, proving the
//! stack accepts a non-Steenrod bialgebra end to end. Run with e.g.
//! `cargo run --example chromatic_grs1 -- 32 18`.

use std::sync::Arc;

use algebra::{
    Algebra, MoravaStabilizerAlgebra,
    module::{FDModule, Module, homomorphism::FullModuleHomomorphism},
};
use bivec::BiVec;
use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
    ext_algebra::field_resolution_ext,
    resolution::Resolution,
};
use sseq::coordinates::Bidegree;

/// `dim Ext^{s, t}` predicted by `F_2[h_1, h_2, ...]`, `|h_i| = 2 (2^i - 1)`: the number of
/// multisets of generators of total cardinality `s` and total internal degree `t`.
fn predicted_chart(max_s: usize, max_t: usize) -> Vec<Vec<usize>> {
    // Generator internal degrees d_i = 2 (2^i - 1) that fit in the range.
    let mut degrees = Vec::new();
    let mut i = 1u32;
    loop {
        let d = 2 * ((1i64 << i) - 1);
        if d as usize > max_t {
            break;
        }
        degrees.push(d as usize);
        i += 1;
    }

    // f[s][t] = number of multisets of size s and weight t. Unbounded knapsack over generators,
    // tracking both cardinality and weight.
    let mut f = vec![vec![0usize; max_t + 1]; max_s + 1];
    f[0][0] = 1;
    for &d in &degrees {
        for s in 1..=max_s {
            for t in d..=max_t {
                f[s][t] += f[s - 1][t - d];
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

    // Validate against F_2[h_1, h_2, ...].
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
        println!(
            "\nPASS: every computed bidegree matches F_2[h_1, h_2, ...] (|h_i| = 2(2^i - 1))."
        );
    } else {
        println!(
            "\nFAIL: {} bidegree(s) disagree with the prediction:",
            mismatches.len()
        );
        println!("  s   t   n   computed  expected");
        for (s, t, n, computed, expected) in &mismatches {
            println!("  {s:<3} {t:<3} {n:<3} {computed:<9} {expected}");
        }
        anyhow::bail!("chart does not match F_2[h_1, h_2, ...]");
    }

    // ------------------------------------------------------------------------------------------
    // Second validation: run the *field trick* (the whole point of the handoff) over gr S(1).
    //
    // `field_resolution_ext` computes Ext_A(M, F_2) by tensoring the base resolution P_. with M and
    // taking cohomology of Hom_A(P_. (x) M, F_2). This exercises the Bialgebra-generic stack — the
    // antipode chi and the closed-form tensor differential delta_Q. Feeding M = F_2 must reproduce
    // the very same chart, confirming the field-trick machinery accepts this new Bialgebra.
    // ------------------------------------------------------------------------------------------
    let t_max = max_n + max_s;
    resolution.compute_through_bidegree(Bidegree::s_t(max_s + 1, t_max));
    resolution.algebra().compute_basis(t_max + 1);

    let m = Arc::new(FDModule::new(
        resolution.algebra(),
        "F_2".to_string(),
        BiVec::from_vec(0, vec![1]),
    ));
    m.compute_basis(t_max);

    let ext = field_resolution_ext(Arc::clone(&resolution), m);

    let mut trick_mismatches = 0usize;
    for s in 0..=max_s {
        for n in 0..=max_n {
            let t = (n + s) as usize;
            let expected = if (s as usize) <= max_s as usize && t <= max_t {
                predicted[s as usize][t]
            } else {
                0
            };
            if let Some(dim) = ext.cohomology_dimension(Bidegree::n_s(n, s))
                && dim != expected
            {
                trick_mismatches += 1;
            }
        }
    }
    if trick_mismatches == 0 {
        println!(
            "PASS: the field trick (antipode + tensor differential) reproduces the same chart."
        );
    } else {
        println!("FAIL: field trick disagrees in {trick_mismatches} bidegree(s).");
        anyhow::bail!("field trick chart does not match F_2[h_1, h_2, ...]");
    }

    // Show where the polynomial generators h_i land.
    println!("\nPolynomial generators h_i (filtration-one classes):");
    let mut i = 1u32;
    loop {
        let d = 2 * ((1i64 << i) - 1);
        if d > (max_n + max_s) as i64 {
            break;
        }
        println!("  h_{i}: (s, t) = (1, {d}),  stem n = {}", d - 1);
        i += 1;
    }

    Ok(())
}
