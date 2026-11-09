//! Computes the ring structure of $\Ext_A(\F_2, \F_2)$ through an [`ExtAlgebra`] by W. Lin's 2026
//! method (see [`ext::ext_algebra::lin_ring`]): `h_i` products read off the differentials, one
//! chain map per basis monomial lifted only over a corner near the origin, and everything else by
//! associativity.
//!
//! The module must be `S_2` over the Milnor basis. After the resolution, this asks for the internal
//! degree `t` to compute the ring through (default: everything resolved). It prints the
//! indecomposables, then every non-zero product `x_(n, s, i) · x_(n', s', i') = [...]` of two basis
//! elements whose product lies in the range, sorted; this is the format of `all_products` with
//! `EXT_DUMP_PRODUCTS=1`, which lifts a chain map for every basis element instead, so the two
//! outputs can be diffed.
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
    ext_algebra::{ExtAlgebra, RingRange},
    utils::query_module,
};
use sseq::coordinates::{Bidegree, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let res = Arc::new(query_module(Some(AlgebraType::Milnor), true)?);
    let max_n = res.iter_stem().map(|b| b.n()).max().unwrap_or(0);
    let max_s = res.iter_stem().map(|b| b.s()).max().unwrap_or(0);
    let max_t = query::with_default(
        "Ring through internal degree t",
        &(max_n + max_s).to_string(),
        str::parse,
    );
    let range = RingRange {
        max_n,
        max_s,
        max_t,
    };

    let ext = ExtAlgebra::new(Arc::clone(&res));
    let start = std::time::Instant::now();
    ext.compute_ring(range)?;
    let elapsed = start.elapsed();

    for g in ext.indecomposables().unwrap() {
        println!("indecomposable x_{g}");
    }

    let bidegrees: Vec<Bidegree> = res
        .iter_stem()
        .filter(|&b| range.contains(b) && res.number_of_gens_in_bidegree(b) > 0)
        .collect();
    let mut lines = Vec::new();
    for &b in &bidegrees {
        for &c in &bidegrees {
            let bc = b + c;
            if !range.contains(bc) || res.number_of_gens_in_bidegree(bc) == 0 {
                continue;
            }
            for g in ext.basis(b) {
                let matrix = ext
                    .known_multiply_into(&ext.generator(g), c)
                    .expect("product in the ring's range");
                for (y, row) in matrix.iter().enumerate() {
                    if row.is_zero() {
                        continue;
                    }
                    let coords: Vec<u32> = row.iter().collect();
                    lines.push(format!(
                        "x_{g} · x_{} = {coords:?}",
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
    ext.with_ring(|ring| {
        let (checks, failures) = ring.check_h_associativity();
        eprintln!("lin_ring: h_i associativity {checks} checks, {failures} failures");
        eprintln!(
            "lin_ring: {} indecomposables, {} basis monomials, {} chain maps over {} cells, ring \
             in {:.1} s",
            ring.gens.len(),
            ring.basis.len(),
            ring.maps(),
            ring.corner_cells(),
            elapsed.as_secs_f64()
        )
    });
    Ok(())
}
