use std::sync::Arc;

use algebra::{
    Algebra,
    module::{HomModule, Module},
};
use ext::chain_complex::{AugmentedChainComplex, BoundedChainComplex, ChainComplex};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let resolution = Arc::new(ext::utils::query_module_only(
        "Module",
        Some(algebra::AlgebraType::Milnor),
        true,
    )?);
    let target_cc = resolution.target();

    if target_cc.max_s() > 1 {
        anyhow::bail!("Cannot compute secondary space for non-module");
    }

    let module = target_cc.module(0);

    let max_nonzero = module
        .max_degree()
        .ok_or_else(|| anyhow::anyhow!("Expected bounded module"))?;

    resolution.compute_through_bidegree(Bidegree::n_s(max_nonzero, 2));

    let hom = HomModule::new(resolution.module(2), Arc::clone(&module));
    resolution.algebra().compute_basis(2 * max_nonzero);
    hom.compute_basis(max_nonzero);

    for degree in hom.min_degree()..hom.max_computed_degree() {
        for idx in 0..hom.dimension(degree) {
            println!(
                "({degree},{idx}): {}",
                hom.basis_element_to_string(degree, idx)
            )
        }
    }

    Ok(())
}
