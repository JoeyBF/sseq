use std::sync::Arc;

use algebra::module::Module;
use ext::double::DoubleChainComplex;
use ext::{
    chain_complex::{AugmentedChainComplex, BoundedChainComplex, ChainComplex, FreeChainComplex},
    resolution_homomorphism::ResolutionHomomorphism,
    utils,
};
use fp::vector::FpVector;
use itertools::Itertools;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let res = utils::query_module(None, true)?;
    assert!(
        res.prime() == 2 && res.target().max_s() == 1 && res.target().module(0).is_unit(),
        "Sq^0 can only be computed for the sphere at the prime 2"
    );
    assert_eq!(
        res.algebra().kind(),
        algebra::SteenrodKind::Classical,
        "Sq^0 can only be computed over the classical Steenrod algebra"
    );

    let res = Arc::new(res);
    let doubled = Arc::new(DoubleChainComplex::new(Arc::clone(&res)));
    doubled.compute_through_bidegree(Bidegree::s_t(res.next_homological_degree() - 1, 0));

    let hom = ResolutionHomomorphism::new(
        String::from("Sq^0"),
        Arc::clone(&res),
        doubled,
        Bidegree::zero(),
    );
    hom.extend_step_raw(
        Bidegree::zero(),
        Some(vec![FpVector::from_slice(res.prime(), &[1])]),
    );
    hom.extend_all();

    for b in res.iter_nonzero_stem() {
        let doubled_b = Bidegree::s_t(b.s(), 2 * b.t());
        if !res.has_computed_bidegree(doubled_b) {
            continue;
        }

        let source_num_gens = res.number_of_gens_in_bidegree(doubled_b);
        let module = res.module(b.s());
        let offset = module.generator_offset(b.t(), b.t(), 0);
        let map = hom.get_map(b.s());

        for i in 0..res.number_of_gens_in_bidegree(b) {
            let g = BidegreeGenerator::new(b, i);
            println!(
                "Sq^0 x_{g} = [{}]",
                (0..source_num_gens)
                    .map(|j| map.output(doubled_b.t(), j).entry(offset + i))
                    .format(", ")
            )
        }
    }
    Ok(())
}
