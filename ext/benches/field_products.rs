//! Wall-clock comparison of products and Massey products computed two ways over the same box: the
//! **field trick** (`S_2` resolution ⊗ C2, non-minimal `Q• = P• ⊗ C2`) vs a **native** minimal C2
//! resolution. The two agree numerically (see the `ext_algebra` tests); this measures the cost.
//!
//! Run with `cargo bench --bench field_products`. Override the box with `FIELD_BENCH_BOX="n,s"`.

use std::{sync::Arc, time::Instant};

use algebra::module::{FDModule, Module};
use ext::{
    chain_complex::ChainComplex,
    ext_algebra::{ExtAlgebra, field_resolution_ext, field_resolution_products},
    utils::{construct_standard, parse_module_name},
};
use sseq::coordinates::{Bidegree, BidegreeGenerator};

/// Multiply every Ext class in the box by `h0` (exercising the product maps) and return a checksum
/// (which also defeats dead-code elimination).
macro_rules! products_workload {
    ($alg:expr, $h0:expr, $nn:expr, $ss:expr) => {{
        let mut checksum = 0u64;
        for n in 0..=$nn {
            for s in 0..=$ss {
                let b = Bidegree::n_s(n, s);
                for i in 0..$alg.dimension(b) {
                    let y = $alg.generator(BidegreeGenerator::new(b, i));
                    if let Some(p) = $alg.try_multiply(&y, $h0) {
                        checksum += p.vec().iter().map(u64::from).sum::<u64>();
                    }
                }
            }
        }
        checksum
    }};
}

fn main() {
    let (nn, ss): (i32, i32) = std::env::var("FIELD_BENCH_BOX")
        .ok()
        .and_then(|s| {
            let (n, s) = s.split_once(',')?;
            Some((n.trim().parse().ok()?, s.trim().parse().ok()?))
        })
        .unwrap_or((30, 12));
    let margin = Bidegree::n_s(nn + 2, ss + 3);

    // Shared unit infrastructure (`S_2`), computed once and excluded from both setup timings.
    let k_res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
    k_res.compute_through_bidegree(Bidegree::s_t(margin.s() + 1, margin.n() + margin.s() + 2));

    let m =
        Arc::new(FDModule::from_json(k_res.algebra(), &parse_module_name("C2").unwrap()).unwrap());
    m.compute_basis(nn + ss + 8);

    // ---- Additive Ext only, closed form (no materialised Q•): the "fixed cost for every M". ----
    let t = Instant::now();
    let closed = field_resolution_ext(Arc::clone(&k_res), Arc::clone(&m));
    let mut cdim = 0usize;
    for n in 0..=nn {
        for s in 0..=ss {
            cdim += closed
                .cohomology_dimension(Bidegree::n_s(n, s))
                .unwrap_or(0);
        }
    }
    let field_additive = t.elapsed();

    // ---- Field trick structure: products + Massey, entirely closed form (Q• is never built). ----
    let t = Instant::now();
    let field = field_resolution_products(Arc::clone(&k_res), m);
    field.compute_through_bidegree(margin);
    let field_setup = t.elapsed();

    let h0f = field.unit_generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));
    let h1f = field.unit_generator(BidegreeGenerator::new(Bidegree::n_s(1, 1), 0));

    let t = Instant::now();
    let fsum = products_workload!(field, &h0f, nn, ss);
    let field_products = t.elapsed();

    let t = Instant::now();
    let fmassey = field.massey_iter_c(&h0f, &h1f).len();
    let field_massey = t.elapsed();

    // ---- Native minimal C2 resolution. ----
    let t = Instant::now();
    let c2 = Arc::new(construct_standard::<false, _, _>("C2", None).unwrap());
    c2.compute_through_stem(margin);
    let direct = ExtAlgebra::new(Arc::clone(&c2), Arc::clone(&k_res));
    let direct_setup = t.elapsed();

    let h0d = direct.unit_generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));
    let h1d = direct.unit_generator(BidegreeGenerator::new(Bidegree::n_s(1, 1), 0));

    let t = Instant::now();
    let dsum = products_workload!(direct, &h0d, nn, ss);
    let direct_products = t.elapsed();

    let t = Instant::now();
    let dmassey = direct.massey_iter_c(&h0d, &h1d).len();
    let direct_massey = t.elapsed();

    println!("box  n ≤ {nn}, s ≤ {ss}");
    println!("FIELD   additive-Ext (closed form) {field_additive:>9.2?}   (dim sum {cdim})");
    println!(
        "FIELD   setup {field_setup:>9.2?}   products {field_products:>9.2?}   massey \
         {field_massey:>9.2?}   ({fmassey} brackets, sum {fsum})"
    );
    println!(
        "DIRECT  setup {direct_setup:>9.2?}   products {direct_products:>9.2?}   massey \
         {direct_massey:>9.2?}   ({dmassey} brackets, sum {dsum})"
    );
}
