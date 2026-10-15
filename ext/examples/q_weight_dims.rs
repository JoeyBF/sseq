//! Prints the number of generators of $\Ext^{s, n + s}$ of each $Q$-weight $e$ in the format
//! `n,s,e,num_gens`. See [`ext::q_weight`] for the definition of the $Q$-weight. This only makes
//! sense at odd primes.
//!
//! The first prompt asks whether to resolve the module as an unstable module.
//!
//! # Example
//! ```text
//! $ printf 'y\nS_3[3]\n\n40\n15\n' | cargo run --example q_weight_dims
//! ```
use ext::{
    chain_complex::ChainComplex,
    q_weight::QWeights,
    utils::{query_module, query_unstable_module},
};
use sseq::coordinates::Bidegree;

fn print_weights(q: &QWeights, bidegrees: impl Iterator<Item = Bidegree>) {
    for b in bidegrees {
        for (e, num_gens) in q.weighted_dimensions(b) {
            println!("{},{},{e},{num_gens}", b.n(), b.s());
        }
    }
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    if query::yes_no("Unstable") {
        let res = query_unstable_module(false)?;
        let max = Bidegree::n_s(
            query::raw("Max n", str::parse),
            query::raw("Max s", str::parse),
        );
        res.compute_through_stem(max);
        print_weights(&QWeights::new(&res)?, res.iter_stem());
    } else {
        let res = query_module(None, false)?;
        print_weights(&QWeights::new(&res)?, res.iter_stem());
    }
    Ok(())
}
