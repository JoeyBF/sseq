//! Computes the ring structure of $\Ext_A(\F_2, \F_2)$ by W. Lin's 2026 method (see
//! [`ext::lin_ring`]): `h_i` products read off the differentials, one chain map per basis monomial
//! lifted only over a corner near the origin, and everything else by associativity.
//!
//! The module must be `S_2` over the Milnor basis. This prints the indecomposables, then every
//! non-zero product `x_(n, s, i) · x_(n', s', i') = [...]` of two basis elements whose product lies
//! in the resolved range, sorted; this is the format of `all_products` with `EXT_DUMP_PRODUCTS=1`,
//! which lifts a chain map for every basis element instead, so the two outputs can be diffed.
//!
//! # Sample output
//! ```text
//! indecomposable x_(0, 1, 0)
//! indecomposable x_(1, 1, 0)
//! ...
//! x_(0, 1, 0) · x_(0, 1, 0) = [1]
//! ...
//! ```

use std::sync::Arc;

use algebra::AlgebraType;
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    hi_products::hi_products,
    lin_ring::compute,
    utils::query_module,
};
use sseq::coordinates::{Bidegree, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let res = Arc::new(query_module(Some(AlgebraType::Milnor), true)?);
    let max_n = res.iter_stem().map(|b| b.n()).max().unwrap_or(0);
    let max_s = res.iter_stem().map(|b| b.s()).max().unwrap_or(0);
    let max_t = max_n + max_s;

    let ks: Vec<u32> = (0..=max_t.max(1).ilog2()).collect();
    let hi = hi_products(&*res, max_n, max_s, &ks)?;
    let mut ring = compute(&res, &hi, max_n, max_s, max_t)?;

    for g in &ring.gens {
        println!("indecomposable x_{}", BidegreeGenerator::new(g.b, g.idx));
    }

    let bidegrees: Vec<Bidegree> = res
        .iter_stem()
        .filter(|&b| res.number_of_gens_in_bidegree(b) > 0)
        .collect();
    let mut lines = Vec::new();
    for &b in &bidegrees {
        for &c in &bidegrees {
            let bc = b + c;
            if bc.n() > max_n || bc.s() > max_s || res.number_of_gens_in_bidegree(bc) == 0 {
                continue;
            }
            for (x, rows) in ring.basis_block(b, c).iter().enumerate() {
                for (y, row) in rows.iter().enumerate() {
                    if row.is_zero() {
                        continue;
                    }
                    let coords: Vec<u32> = row.iter().collect();
                    lines.push(format!(
                        "x_{} · x_{} = {coords:?}",
                        BidegreeGenerator::new(b, x),
                        BidegreeGenerator::new(c, y)
                    ));
                }
            }
        }
    }
    lines.sort();
    for l in lines {
        println!("{l}");
    }
    eprintln!(
        "lin_ring: {} indecomposables, {} basis monomials, {} chain maps",
        ring.gens.len(),
        ring.basis.len(),
        ring.maps()
    );
    Ok(())
}
