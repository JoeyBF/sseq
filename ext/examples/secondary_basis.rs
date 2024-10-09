use std::sync::Arc;

use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    secondary::*,
    utils::query_module,
};
use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging();

    let resolution = Arc::new(query_module(Some(algebra::AlgebraType::Milnor), true)?);
    let p = resolution.prime();

    let lift = SecondaryResolution::new(Arc::clone(&resolution));
    lift.extend_all();

    let sseq = lift.e3_page();
    let get_page_data = |b: Bidegree| {
        let d = sseq.page_data(b.n(), b.s() as i32);
        &d[std::cmp::min(3, d.len() - 1)]
    };

    for b in resolution.iter_nonzero_stem() {
        let subquotient = get_page_data(b);

        let ambient = subquotient.ambient_dimension();

        let boundary = subquotient
            .zeros()
            .iter()
            .map(|v| ("B", BidegreeElement::new(b, v.to_owned())));
        let cycle = subquotient
            .gens()
            .map(|v| ("Z", BidegreeElement::new(b, v.to_owned())));
        let rest = subquotient
            .complement_pivots()
            .map(|i| ("E", BidegreeGenerator::new(b, i).into_element(p, ambient)));

        for (prefix, v) in boundary.chain(cycle).chain(rest) {
            println!("{} {:#}", prefix, v);
        }
    }

    Ok(())
}
