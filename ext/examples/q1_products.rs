//! How much of $\mathrm{Ext}_P(Q/I_1)$ is decomposable under $v_1$ and the $h_i$, and how many
//! module generators remain.
//!
//! Usage: `q1_products <max_n> <max_s>`. Prints totals, the generators by filtration, and checks
//! the class count against the quotient-layer spectral sequence.
use std::sync::Arc;

use ext::{
    algebraic_novikov::unit_module,
    chain_complex::{ChainComplex, FiniteChainComplex},
    q1_products::{Backbone, decomposables},
    resolution::Resolution,
    vn_bockstein::{Layer, page_dims, vn_bockstein},
};
use sseq::coordinates::Bidegree;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: i32 = args[1].parse().unwrap();
    let max_s: i32 = args[2].parse().unwrap();
    let max_u = (max_n + max_s) / 2;
    let h: Vec<u32> = (0..).take_while(|&i| 1 << i <= max_u).collect();
    let top = 1 << h.last().unwrap();
    let res: Arc<Backbone> = Arc::new(Resolution::new(Arc::new(FiniteChainComplex::ccdz(
        Arc::new(unit_module()),
    ))));
    res.compute_through_bidegree(Bidegree::s_t(max_s + 2, max_u + top + 2));

    let d = decomposables(Arc::clone(&res), max_n, max_s, &h);
    let sum = |m: &std::collections::BTreeMap<(i32, i32, i32), usize>| m.values().sum::<usize>();
    let classes = sum(&d.classes);
    println!(
        "Ext(Q/I_1): {classes} classes; spans of products: v1 {}, h_i {}, all {}; generators {}; \
         products not read (truncation) {}",
        sum(&d.by_v1),
        sum(&d.by_h),
        sum(&d.by_any),
        classes - sum(&d.by_any),
        d.dropped
    );
    println!(
        "d_1(v_1 x) = h_0 x + v_1 d_1(x): holds {}, fails {}; d_1(h_i x) = h_i d_1(x): holds {}, fails {}",
        d.leibniz_v1.0, d.leibniz_v1.1, d.linear_h.0, d.linear_h.1
    );
    let reference = page_dims(
        &vn_bockstein(&*res, Layer::Quotient(1), max_n, max_s),
        i32::MAX,
        max_n,
        max_s,
    );
    let reference: std::collections::BTreeMap<_, _> = reference.into_iter().collect();
    println!(
        "against the quotient layer: {} classes, equal by degree: {}",
        reference.values().sum::<usize>(),
        reference == d.classes
    );
    let mut gens_by_s: std::collections::BTreeMap<i32, (usize, usize)> = Default::default();
    for (&key, &c) in &d.classes {
        let e = gens_by_s.entry(key.1).or_default();
        e.0 += c;
        e.1 += c - d.by_any[&key];
    }
    println!("by s: (classes, generators)");
    for (s, (c, g)) in gens_by_s {
        print!(" {s}:({c},{g})");
    }
    println!();
}
