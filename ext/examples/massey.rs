//! Computes the triple Massey product up to a sign
//!
//! This is optimized to compute <a, b, -> for fixed a, b and all -, where a and b have small
//! degree.

use std::sync::Arc;

use ext::{chain_complex::ChainComplex, ext_algebra::ExtAlgebra};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let resolution = Arc::new(ext::utils::query_module(None, true)?);
    let alg = ExtAlgebra::from_resolution(Arc::clone(&resolution))?;

    eprintln!("\nComputing Massey products <a, b, ->");
    eprintln!("\nEnter a:");

    let a_deg = Bidegree::n_s(
        query::raw("n of Ext class a", str::parse),
        query::raw("s of Ext class a", str::parse::<std::num::NonZeroI32>).get(),
    );
    alg.unit().compute_through_stem(a_deg);
    let a_class = query::vector("Input Ext class a", alg.unit_dimension(a_deg));
    let a = alg.unit_element(a_deg, &a_class);

    eprintln!("\nEnter b:");

    let b_deg = Bidegree::n_s(
        query::raw("n of Ext class b", str::parse),
        query::raw("s of Ext class b", str::parse::<std::num::NonZeroI32>).get(),
    );
    alg.unit().compute_through_stem(b_deg);
    let b_class = query::vector("Input Ext class b", alg.unit_dimension(b_deg));
    let b = alg.unit_element(b_deg, &b_class);

    // The Massey product shifts the bidegree by this amount.
    let shift = a_deg + b_deg - Bidegree::s_t(1, 0);

    if !alg.is_unit() {
        alg.unit().compute_through_stem(shift);
    }

    if !resolution.has_computed_bidegree(shift + Bidegree::s_t(0, resolution.min_degree())) {
        eprintln!("No computable bidegrees");
        return Ok(());
    }

    for (c, result) in alg.massey_family(&a, &b) {
        let entries: Vec<u32> = result.representative.vec().iter().collect();
        println!(
            "<a, b, {c_string}> = {entries:?}",
            c_string = c.to_basis_string()
        );
    }

    Ok(())
}
