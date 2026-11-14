//! Computes $\mathrm{Ext}_P(Q)$ through `CotorComplex` on $Q/(v_0 - 1)$, filtered by length.
//!
//! Usage: `cotor_q <max_n> <max_s> [compare|old]`, with `CTAU_BACKBONE=<dir>` to load or save the
//! resolution of $\mathbb{F}_2$. Prints the number of classes and the time, and with `compare`
//! checks against `ctau_e2_untwisted_on`; with `old` runs only that. `CTAU_MATRIX=1` computes
//! the action inline from Milnor matrices (`MatrixQComodule`) instead of by the walk.
use std::{collections::BTreeMap, sync::Arc, time::Instant};

use ext::{
    algebraic_novikov::{
        ctau_e2_untwisted_on, q_mod_v0_comodule, unit_module, untwisted_backbone_range,
    },
    chain_complex::ChainComplex,
    cotor::CotorComplex,
};
use maybe_rayon::prelude::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: i32 = args[1].parse().unwrap();
    let max_s: i32 = args[2].parse().unwrap();
    let save = std::env::var_os("CTAU_BACKBONE").map(std::path::PathBuf::from);
    let res =
        Arc::new(ext::nassau::Resolution::new_with_save(Arc::new(unit_module()), save).unwrap());
    let max_u = (max_n + max_s) / 2;
    res.compute_through_stem(untwisted_backbone_range(max_n, max_s));

    let mode = args.get(3).map(String::as_str);
    if mode == Some("old") {
        let start = Instant::now();
        let theirs = ctau_e2_untwisted_on(&*res, max_n, max_s).0;
        println!(
            "untwisted: {} classes ({:?})",
            theirs.values().sum::<usize>(),
            start.elapsed()
        );
        return;
    }
    let start = Instant::now();
    let ours = if std::env::var_os("CTAU_MATRIX").is_some() {
        let m = Arc::new(algebra::comodule::MatrixQComodule::new(
            res.algebra(),
            max_u,
        ));
        run(&res, m, max_n, max_s)
    } else {
        run(
            &res,
            Arc::new(q_mod_v0_comodule(res.algebra(), max_u)),
            max_n,
            max_s,
        )
    };
    println!(
        "cotor: {} classes ({:?})",
        ours.values().sum::<usize>(),
        start.elapsed()
    );
    if mode == Some("compare") {
        let start = Instant::now();
        let theirs = ctau_e2_untwisted_on(&*res, max_n, max_s).0;
        println!(
            "untwisted: {} classes ({:?}); equal: {}",
            theirs.values().sum::<usize>(),
            start.elapsed(),
            theirs == ours
        );
    }
}

/// $\mathrm{Ext}_P(Q_k)$ for every $k$, keyed by (stem, s, weight), from the length-filtered
/// comodule `m`.
fn run<M>(
    res: &Arc<ext::nassau::Resolution<algebra::module::FDModule<algebra::MilnorAlgebra>>>,
    m: Arc<M>,
    max_n: i32,
    max_s: i32,
) -> BTreeMap<(i32, i32, i32), usize>
where
    M: algebra::comodule::FilteredComodule<Algebra = algebra::MilnorAlgebra> + Sync,
{
    let max_u = (max_n + max_s) / 2;
    let cx = CotorComplex::new(Arc::clone(res), m, max_s, max_u);
    let t_act = Instant::now();
    // A coboundary at (a, u) reads degrees u - t with t at least a, and 2u - a <= max_n + 1.
    let nnz = cx.precompute_actions((max_n + 1) / 2);
    println!("action: {nnz} entries ({:?})", t_act.elapsed());
    // The coboundary from filtration a at weight u is read for Q_k with k up to the most any
    // class at (a, u) or (a + 1, u) needs.
    let wanted: Vec<(i32, i32)> = (0..=max_u)
        .flat_map(|u| (0..=max_s).map(move |a| (a, u)))
        .filter(|&(a, u)| 2 * u - a <= max_n + 1)
        .collect();
    let ranks: BTreeMap<(i32, i32), Vec<usize>> = wanted
        .into_maybe_par_iter()
        .map(|(a, u)| ((a, u), cx.coboundary_ranks_by_filtration(a, u, max_s - a)))
        .collect();
    let mut ours = BTreeMap::new();
    for (&(a, u), out) in &ranks {
        let n = 2 * u - a;
        if !(0..=max_n).contains(&n) {
            continue;
        }
        let dims = cx.cochain_dimensions_by_filtration(a, u, max_s - a);
        for k in 0..=max_s - a {
            let incoming = if a > 0 {
                ranks
                    .get(&(a - 1, u))
                    .and_then(|r| r.get(k as usize))
                    .copied()
                    .unwrap_or(0)
            } else {
                0
            };
            let d = dims[k as usize] - out[k as usize] - incoming;
            if d > 0 {
                ours.insert((n, a + k, u), d);
            }
        }
    }
    ours
}
