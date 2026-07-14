//! Compute `Ext_A(M, k)` for an **infinite** module `M` via Nassau's field-resolution trick:
//! resolve the base field `k`, tensor with `M` to get a non-minimal free resolution, and take the
//! cohomology of `Hom_A(P_• ⊗ M, k)`.
//!
//! The default `M` is `RP^∞` (real projective space), which is infinite-dimensional and therefore
//! *cannot* be resolved by the usual minimal-resolution engine — the whole point of the trick.
//!
//! Run with e.g. `cargo run --example field_resolution` (p = 2).

use std::sync::Arc;

use algebra::{
    Algebra,
    module::{Module, RealProjectiveSpace},
};
use ext::{
    chain_complex::ChainComplex,
    ext_algebra::field_resolution_ext,
    utils::{construct_standard, unicode_num},
};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    eprintln!("Computes Ext_A(RP^∞, F_2) by resolving k and tensoring with RP^∞ (p = 2).");

    let max_n: i32 = query::with_default("Max n", "25", str::parse);
    let max_s: i32 = query::with_default("Max s", "15", str::parse);
    let max = Bidegree::n_s(max_n, max_s);
    let t_max = max.t();

    // Resolve the base field k = S_2 (this is the "field resolution" P_•).
    let resolution = Arc::new(construct_standard::<false, _, _>("S_2", None)?);
    resolution.compute_through_bidegree(Bidegree::s_t(max_s + 1, t_max));
    resolution.algebra().compute_basis(t_max + 1);

    // M = RP^∞, an infinite module. `None` max degree ⇒ genuinely unbounded.
    let rp_inf = Arc::new(RealProjectiveSpace::new(
        resolution.algebra(),
        1,
        None,
        false,
    ));
    rp_inf.compute_basis(t_max);

    let ext = field_resolution_ext(Arc::clone(&resolution), rp_inf);

    // Print the Ext chart: rows are s (high to low), columns are n.
    println!("Ext_A(RP^∞, F_2), rows s = {max_s}..0 (top to bottom), columns n = 0..{max_n}:");
    let mut out = String::new();
    for s in (0..=max_s).rev() {
        for n in 0..=max_n {
            let dim = ext
                .cohomology_dimension(Bidegree::n_s(n, s))
                .expect("computed range");
            out.push(unicode_num(dim));
            out.push(' ');
        }
        out.push('\n');
        if out.trim_start().is_empty() {
            out.clear();
        }
    }
    print!("{out}");

    Ok(())
}
