//! Writes a chart of $\Ext_A(\F_2, \F_2)$ with more products than `chart`'s `h_i`, through an
//! [`ExtAlgebra`]: `h_0, .., h_3` over the whole resolved range (read off the differentials), and
//! products by the indecomposables of filtration at least 2 up to a chosen stem (`c_0`, `Ph_1`,
//! `d_0`, ...) over the range of a ring computed by W. Lin's method (see
//! [`ext::ext_algebra::lin_ring`]).
//!
//! The module must be `S_2` over the Milnor basis. After the resolution, this asks for the output
//! format, the internal degree `t` to compute the ring through, and the largest stem of a
//! multiplier.

use std::sync::Arc;

use algebra::AlgebraType;
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::{ExtAlgebra, RingRange},
    utils::query_module,
};
use sseq::{
    charting::{SeqSeeBackend, SvgBackend, TikzBackend},
    coordinates::{Bidegree, BidegreeGenerator},
};

/// Customary names of the indecomposables in the lowest stems, where each bidegree has one.
fn name(g: BidegreeGenerator) -> String {
    let known = match (g.n(), g.s()) {
        (8, 3) => Some("c_0"),
        (9, 5) => Some("Ph_1"),
        (11, 5) => Some("Ph_2"),
        (14, 4) => Some("d_0"),
        (17, 4) => Some("e_0"),
        (18, 4) => Some("f_0"),
        (19, 3) => Some("c_1"),
        (20, 4) => Some("g"),
        _ => None,
    };
    match known {
        Some(n) if g.idx() == 0 => n.to_string(),
        _ => format!("x_{g}"),
    }
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let res = Arc::new(query_module(Some(AlgebraType::Milnor), true)?);
    let format = query::with_default("Output format (svg/tikz/seqsee)", "seqsee", |x| match x {
        "svg" | "tikz" | "seqsee" => Ok(x.to_string()),
        _ => Err(format!(
            "unknown format '{x}'; expected one of svg, tikz, seqsee"
        )),
    });
    let max_s = res.iter_stem().map(|b| b.s()).max().unwrap_or(0);
    let ring_n: i32 = query::with_default("Ring through stem", "150", str::parse);
    let max_t: i32 = query::with_default(
        "Ring through internal degree t",
        &(ring_n + max_s).to_string(),
        str::parse,
    );
    let max_stem: i32 = query::with_default("Largest stem of a multiplier", "20", str::parse);

    let ext = ExtAlgebra::new(Arc::clone(&res));
    let start = std::time::Instant::now();
    ext.compute_ring(RingRange {
        max_n: ring_n,
        max_s,
        max_t,
    })?;
    eprintln!(
        "ring through stem {ring_n}, t {max_t} in {:.1} s",
        start.elapsed().as_secs_f64()
    );
    for g in ext.indecomposables().unwrap() {
        eprintln!("indecomposable x_{g}");
    }

    let mut products = Vec::new();
    for i in 0..4 {
        let b = Bidegree::s_t(1, 1 << i);
        if ext.dimension(b) == 1 {
            let h = ext.generator(BidegreeGenerator::new(b, 0));
            products.push((format!("h_{i}"), ext.known_product(&h)));
        }
    }
    for g in ext.indecomposables().unwrap() {
        if g.s() >= 2 && g.n() <= max_stem {
            products.push((name(g), ext.known_product(&ext.generator(g))));
        }
    }
    for (name, p) in &products {
        let edges: usize = p
            .matrices
            .iter()
            .map(|(_, m)| m.iter().map(|r| r.iter_nonzero().count()).sum::<usize>())
            .sum();
        eprintln!("{name}: {edges} products");
    }

    let sseq = res.to_sseq();
    let out = std::io::stdout();
    match format.as_str() {
        "svg" => {
            sseq.write_to_graph(SvgBackend::new(out), 2, false, products.iter(), |_| Ok(()))?
        }
        "tikz" => {
            sseq.write_to_graph(TikzBackend::new(out), 2, false, products.iter(), |_| Ok(()))?
        }
        "seqsee" => {
            sseq.write_to_graph(SeqSeeBackend::new(out), 2, false, products.iter(), |_| {
                Ok(())
            })?
        }
        _ => unreachable!(),
    }
    Ok(())
}
