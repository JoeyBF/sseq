//! Prints the first $\tau$-Bockstein differential's ranks on $\mathrm{Ext}_P(Q)$ from the untwisted resolution.

use std::sync::Arc;

use ext::{algebraic_novikov::unit_module, chain_complex::ChainComplex, tau_lift::TauLift};
use sseq::coordinates::Bidegree;

/// Usage: `tau_lift <max_h> <max_t>`; prints `stem,s,weight,dim E_1,rank d_1` by source.
fn main() {
    let args: Vec<i32> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (max_h, max_t) = (args[0], args[1]);
    let start = std::time::Instant::now();
    let res = ext::nassau::Resolution::new(Arc::new(unit_module()));
    res.compute_through_bidegree(Bidegree::s_t(max_h, max_t / 2));
    let lift = TauLift::new(&res, max_h, max_t);
    eprintln!("lifted: {:?}", start.elapsed());
    lift.check_square_zero();
    eprintln!("D^2 = 0 checked");
    eprintln!("δ_1 shapes (Δa, Δk): {:?}", lift.delta1_shapes());
    eprintln!(
        "δ_1 backbone pairs, local: {:?}",
        lift.delta1_locality(&res)
    );
    let (pages, shift) = lift.pages();
    eprintln!("d_1 weight shift: {shift:?}; total {:?}", start.elapsed());
    for ((n, s, w), (dim, rank)) in pages {
        println!("{n},{s},{w},{dim},{rank}");
    }
}
