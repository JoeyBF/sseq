//! Validate the signature engine on **finite chain complexes** — specifically the
//! Yoneda cofibers named by the `cofiber` attribute of a module descriptor.
//!
//! The `cofiber` spec `{s, t, idx}` names an Ext class of the base module `M`;
//! its Yoneda representative is a bounded chain complex of `FDModule`s (a finite
//! `(s+1)`-term complex), and its Ext is the cofiber's Adams `E₂`. We build that
//! complex exactly as `utils::construct_standard` does (but over `MilnorAlgebra`,
//! keeping `FDModule`s), then resolve it with **both** the generic minimal-
//! resolution engine and the signature engine and check the ranks agree.
//!
//! ```text
//! cargo run --release --features sig-nassau --example cofiber_validate -- C4 30 18
//! ```
//! Args: cofiber module name (default `C4`), max_n (stem), max_s.

use std::sync::Arc;

use algebra::{milnor_algebra::MilnorAlgebra, module::FDModule};
use ext::{
    chain_complex::{ChainComplex, ChainMap, FiniteChainComplex, FreeChainComplex},
    motivic_nassau::SignatureResolution,
    resolution::Resolution,
    utils::load_module_json,
    yoneda::yoneda_representative,
};
use fp::{matrix::Matrix, prime::TWO};
use sseq::coordinates::{Bidegree, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "C4".to_string());
    let max_n: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let max_s: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(18);
    let max_t = max_n + max_s;

    let json = load_module_json(&name)?;
    anyhow::ensure!(
        !json["cofiber"].is_null(),
        "{name} has no cofiber attribute"
    );
    let algebra = Arc::new(MilnorAlgebra::new(TWO, false));
    let module = Arc::new(FDModule::from_json(Arc::clone(&algebra), &json)?);

    // --- Build the cofiber Yoneda complex (mirrors utils::construct_standard). ---
    use algebra::module::{Module, homomorphism::FreeModuleHomomorphism};
    let cofiber = &json["cofiber"];
    let shift = json["shift"].as_i64().unwrap_or(0) as i32;
    let cofiber = BidegreeGenerator::s_t(
        cofiber["s"].as_i64().unwrap() as i32,
        cofiber["t"].as_i64().unwrap() as i32 + shift,
        cofiber["idx"].as_u64().unwrap() as usize,
    );
    let base_max = Bidegree::n_s(module.max_degree().unwrap(), 0);

    let base_cc = Arc::new(FiniteChainComplex::<FDModule<MilnorAlgebra>>::ccdz(
        Arc::clone(&module),
    ));
    let resolution = Resolution::new(base_cc);
    resolution.compute_through_stem(cofiber.degree() + base_max);

    let cmap = FreeModuleHomomorphism::new(
        resolution.module(cofiber.s()),
        Arc::clone(&module),
        cofiber.t(),
    );
    let num_gens = resolution
        .module(cofiber.s())
        .number_of_gens_in_degree(cofiber.t());
    let mut new_output = Matrix::new(TWO, num_gens, 1);
    new_output.row_mut(cofiber.idx()).set_entry(0, 1);
    cmap.add_generators_from_matrix_rows(cofiber.t(), new_output.as_slice_mut());
    cmap.extend_by_zero((base_max + cofiber.degree()).t());

    let cm = ChainMap {
        s_shift: cofiber.s(),
        chain_maps: vec![cmap],
    };
    let yoneda = yoneda_representative(Arc::new(resolution), cm);
    let mut cofiber_cc = FiniteChainComplex::from(yoneda);
    cofiber_cc.pop(); // drop the (contractible) top term, as construct_standard does
    let cofiber_cc = Arc::new(cofiber_cc);

    use ext::chain_complex::BoundedChainComplex;
    println!("# cofiber validation: Ext of the Yoneda cofiber {name}");
    println!(
        "# cofiber class (s={}, t={}, idx={}); complex has {} terms",
        cofiber.s(),
        cofiber.t(),
        cofiber.idx(),
        cofiber_cc.max_s() + 1,
    );
    println!("# stem region: n ≤ {max_n}, s ≤ {max_s}");

    // --- Resolve the chain complex with both engines. ---
    let gres = Resolution::new(Arc::clone(&cofiber_cc));
    gres.compute_through_stem(Bidegree::s_t(max_s, max_t));

    let mut sres = SignatureResolution::from_chain_complex(Arc::clone(&cofiber_cc));
    sres.compute_through_stem(max_s, max_t);

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
            format!(
                "CORRECT: signature ranks == generic ranks at every bidegree ({total} generators)"
            )
        } else {
            format!("FAILED: {mism} bidegree rank mismatches")
        }
    );
    let records = sres.shrink_records();
    if !records.is_empty() {
        let sum_c: usize = records.iter().map(|r| r.dim_c).sum();
        let sum_e0: usize = records.iter().map(|r| r.dim_e0.max(1)).sum();
        println!(
            "[2] SHRINK dim C / dim E₀C: aggregate {:.1}×",
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
