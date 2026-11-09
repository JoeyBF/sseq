//! Prints $\mathrm{Ext}_P(Q)$ and the rank of the first algebraic Novikov differential out of it.

use std::sync::Arc;

use ext::{
    algebraic_novikov::unit_module,
    chain_complex::{ChainComplex, FiniteChainComplex},
    novikov_lift::novikov_d2,
    resolution::Resolution,
};
use sseq::coordinates::Bidegree;

/// Usage: `novikov_d2 <max_u> <max_s>`; prints `stem,s,weight,dim,rank` by source.
fn main() {
    let args: Vec<i32> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (max_u, max_s) = (args[0], args[1]);
    let start = std::time::Instant::now();
    let cc: Arc<FiniteChainComplex<algebra::module::FDModule<algebra::MilnorAlgebra>>> =
        Arc::new(FiniteChainComplex::ccdz(Arc::new(unit_module())));
    let res = Resolution::new(cc);
    res.compute_through_bidegree(Bidegree::s_t(max_s + 2, max_u));
    let out = novikov_d2(&res, max_u, max_s);
    for ((n, s, u), (dim, rank)) in out {
        println!("{n},{s},{u},{dim},{rank}");
    }
    eprintln!("total {:?}", start.elapsed());
}
