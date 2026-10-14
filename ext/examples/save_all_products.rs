//! Timing harness for PR #275: products by EVERY basis element of Ext on an existing save.
//!
//! ```text
//! save_all_products SAVE_DIR MAX_N MAX_S OUT.jsonl
//! ```
//!
//! Loads the save through `(MAX_N, MAX_S)` (refusing to resolve anything that is not saved), then
//! extends the product map of every generator of `Ext` in that range through the PR's
//! [`ExtAlgebra::extend_all_products`] (one [`MultiLift`](ext::lift::MultiLift), one batched
//! quasi-inverse per output bidegree). With `EXT_NASSAU_NO_SAVE_QI=1` every quasi-inverse is
//! recomputed on demand from the saved differentials.
//!
//! By default the product maps are built exactly as `extend_all_products` builds them
//! (`ResolutionHomomorphism::from_class` per generator, one `MultiLift`) but unnamed, so no chain
//! map is persisted. `PRODUCTS_PERSIST=1` calls `extend_all_products` itself, whose maps are named
//! `prod_n_s_i` and write every chain-map step into `products/...` inside SAVE_DIR (an fsync'd
//! zarr shard write per map per bidegree, which dominates the run). Either way, run on a scratch
//! copy: opening the store may update its root attributes.
//!
//! Output, JSON lines (the format `scripts/validate_products_wu.py` reads): `{"dims": ...}`, one
//! block `{"class", "shift", "n", "s", "prod"}` per `hK` class (K = 0..3) and source bidegree, then
//! a `{"summary": ...}` line with timings, counts over ALL maps, and the lift / qi statistics.
//! Adapted from `save_products` (branch lift-275, b7c38ac611).

use std::{
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use anyhow::{Context, bail};
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::ExtAlgebra,
    lift::{Liftable, MultiLift},
    resolution_homomorphism::ResolutionHomomorphism,
    save::SaveKind,
    utils::construct_from_save,
};
use fp::vector::FpVector;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        bail!("usage: save_all_products SAVE_DIR MAX_N MAX_S OUT.jsonl");
    }
    let save_dir = &args[1];
    let max_n: i32 = args[2].parse().context("MAX_N")?;
    let max_s: i32 = args[3].parse().context("MAX_S")?;
    let out_path = &args[4];

    let res = Arc::new(construct_from_save(save_dir)?);
    let store = res
        .save_dir()
        .store()
        .context("save directory did not open a store")?;
    let mut missing = Vec::new();
    for s in 1..=max_s {
        for n in 0..=max_n {
            let b = Bidegree::n_s(n, s);
            if !store.exists(SaveKind::NassauDifferential, b) {
                missing.push(b);
            }
        }
    }
    if !missing.is_empty() {
        bail!(
            "{} bidegrees of the range are not in the save (first: {}); refusing to resolve them",
            missing.len(),
            missing[0]
        );
    }

    let t0 = Instant::now();
    res.compute_through_stem(Bidegree::n_s(max_n, max_s));
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!("[save_all_products] loaded through (n, s) = ({max_n}, {max_s}) in {load_s:.1}s");

    let gens: Vec<BidegreeGenerator> = res
        .iter_stem()
        .flat_map(|b| (0..res.number_of_gens_in_bidegree(b)).map(move |i| BidegreeGenerator::new(b, i)))
        .collect();
    let num_maps = gens.len();
    // `PRODUCTS_PERSIST=1`: the PR's `ExtAlgebra::extend_all_products` verbatim, whose product maps
    // are named and so write their chain maps into the save store. Default: the same MultiLift over
    // the same `ResolutionHomomorphism::from_class` maps, but unnamed, so nothing is persisted.
    let persist = std::env::var_os("PRODUCTS_PERSIST").is_some();
    let e2 = ExtAlgebra::without_unit(Arc::clone(&res));
    eprintln!("[save_all_products] extending {num_maps} product maps (persist = {persist})");

    let t1 = Instant::now();
    let homs: Vec<Arc<ResolutionHomomorphism<_, _>>> = if persist {
        e2.extend_all_products();
        gens.iter().map(|&g| e2.generator_product_map(g)).collect()
    } else {
        let homs: Vec<_> = gens
            .iter()
            .map(|&g| {
                let dim = res.number_of_gens_in_bidegree(g.degree());
                let mut class = vec![0u32; dim];
                class[g.idx()] = 1;
                Arc::new(ResolutionHomomorphism::from_class(
                    String::new(),
                    Arc::clone(&res),
                    Arc::clone(&res),
                    g.degree(),
                    &class,
                ))
            })
            .collect();
        let liftables: Vec<Arc<dyn Liftable>> = homs
            .iter()
            .map(|h| Arc::clone(h) as Arc<dyn Liftable>)
            .collect();
        MultiLift::new(Arc::clone(&res), liftables).extend_all();
        homs
    };
    let lift_s = t1.elapsed().as_secs_f64();
    ext::lift::flush_cell_log();
    let (cells, inputs, prepare_s, solve_s, finish_s) = ext::lift::stats();
    let qi = ext::nassau::qi_stats::snapshot();
    eprintln!(
        "[save_all_products] extend_all_products: {num_maps} maps in {lift_s:.3}s; cells {cells} \
         inputs {inputs} prepare {prepare_s:.1}s solve {solve_s:.1}s finish {finish_s:.1}s; qi {qi}"
    );

    // Extract products (untimed by the lift timer, timed separately).
    let t2 = Instant::now();
    let p = res.prime();
    let nonzero = AtomicUsize::new(0);
    let zero = AtomicUsize::new(0);
    let blocks = AtomicUsize::new(0);
    let entries = AtomicUsize::new(0);
    let in_range = |b: Bidegree| b.s() >= 0 && b.s() <= max_s && b.n() >= 0 && b.n() <= max_n;
    let unit_bidegrees: Vec<Bidegree> = res
        .iter_stem()
        .filter(|&b| res.number_of_gens_in_bidegree(b) > 0)
        .collect();
    let hk_names: Vec<(Bidegree, String)> = (0..4)
        .map(|k| (Bidegree::n_s((1 << k) - 1, 1), format!("h{k}")))
        .collect();
    let hk_lines = std::sync::Mutex::new(Vec::<String>::new());
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(8, |x| x.get());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let gi = next.fetch_add(1, Ordering::Relaxed);
                    if gi >= gens.len() {
                        break;
                    }
                    let g = gens[gi];
                    let hom = &homs[gi];
                    let shift = g.degree();
                    let hk = hk_names
                        .iter()
                        .find(|(b, _)| *b == shift && g.idx() == 0 && res.number_of_gens_in_bidegree(shift) == 1)
                        .map(|(_, name)| name.clone());
                    for &b in &unit_bidegrees {
                        let src = b + shift;
                        if !in_range(src) || !res.has_computed_bidegree(src) {
                            continue;
                        }
                        if hom.next_homological_degree() <= src.s()
                            || hom.get_map(src.s()).next_degree() <= src.t()
                        {
                            continue;
                        }
                        let dim = res.number_of_gens_in_bidegree(b);
                        let src_dim = res.number_of_gens_in_bidegree(src);
                        if src_dim == 0 {
                            continue;
                        }
                        blocks.fetch_add(1, Ordering::Relaxed);
                        let mut prod: Vec<Vec<usize>> = Vec::new();
                        for idx in 0..dim {
                            let mut v = FpVector::new(p, src_dim);
                            hom.act(v.as_slice_mut(), 1, BidegreeGenerator::new(b, idx));
                            let row: Vec<usize> = v.iter_nonzero().map(|(j, _)| j).collect();
                            if row.is_empty() {
                                zero.fetch_add(1, Ordering::Relaxed);
                            } else {
                                nonzero.fetch_add(1, Ordering::Relaxed);
                                entries.fetch_add(row.len(), Ordering::Relaxed);
                            }
                            if hk.is_some() {
                                prod.push(row);
                            }
                        }
                        if let Some(name) = &hk {
                            hk_lines.lock().unwrap().push(
                                serde_json::json!({"class": name, "shift": [shift.n(), shift.s()],
                                                   "n": b.n(), "s": b.s(), "prod": prod})
                                .to_string(),
                            );
                        }
                    }
                }
            });
        }
    });
    let extract_s = t2.elapsed().as_secs_f64();

    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    let mut dims = Vec::new();
    for s in 0..=max_s {
        for n in 0..=max_n {
            let b = Bidegree::n_s(n, s);
            if res.has_computed_bidegree(b) {
                let d = res.number_of_gens_in_bidegree(b);
                if d > 0 {
                    dims.push([n as usize, s as usize, d]);
                }
            }
        }
    }
    writeln!(out, "{}", serde_json::json!({ "dims": dims }))?;
    let mut hk_lines = hk_lines.into_inner().unwrap();
    hk_lines.sort();
    for l in &hk_lines {
        writeln!(out, "{l}")?;
    }
    let summary = serde_json::json!({"summary": {
        "save": save_dir, "max_n": max_n, "max_s": max_s, "maps": num_maps, "persist": persist,
        "blocks": blocks.load(Ordering::Relaxed),
        "nonzero_products": nonzero.load(Ordering::Relaxed),
        "zero_products": zero.load(Ordering::Relaxed),
        "nonzero_entries": entries.load(Ordering::Relaxed),
        "load_s": load_s, "lift_s": lift_s, "extract_s": extract_s,
        "multilift": {"cells": cells, "inputs": inputs, "prepare_s": prepare_s,
                      "solve_s": solve_s, "finish_s": finish_s},
        "qi": qi,
        "env": {
            "EXT_NASSAU_NO_SAVE_QI": std::env::var("EXT_NASSAU_NO_SAVE_QI").ok(),
            "EXT_NASSAU_RECOMPUTE_QI": std::env::var("EXT_NASSAU_RECOMPUTE_QI").ok(),
        },
    }});
    writeln!(out, "{summary}")?;
    out.flush()?;
    eprintln!("[save_all_products] extracted in {extract_s:.1}s; {summary}");
    Ok(())
}
