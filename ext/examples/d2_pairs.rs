//! Counts the products the d2 composite of every generator of a saved resolution takes.

use std::{io::Write, sync::Arc, time::Instant};

use ext::chain_complex::ChainComplex;
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

/// Usage: `d2_pairs <store> <max_n> <max_s> <out>`.
///
/// The composite of a generator g in row s multiplies, in the pair algebra, each nonzero entry of
/// the operation from g to a generator g1 in d(g) by each nonzero entry of d(g1). So it takes
/// `pairs(g) = sum over g1 of nnz(d(g) at g1) * nnz(d(g1))` products of basis elements. Writes
/// `n s idx pairs dg_nnz` for every generator with s >= 2, reading each differential once.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 5, "usage: d2_pairs <store> <max_n> <max_s> <out>");
    let store = std::path::PathBuf::from(&args[1]);
    let max = Bidegree::n_s(args[2].parse()?, args[3].parse()?);

    let start = Instant::now();
    let res = Arc::new(ext::utils::construct("S_2@milnor", Some(store))?);
    res.compute_through_stem(max);
    eprintln!("PAIRS open_s={:.1}", start.elapsed().as_secs_f64());

    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[4])?);

    // nnz(d(g)) for the generators of the previous row, by degree then index.
    let row_nnz = |s: i32| -> Vec<Vec<u64>> {
        let d = res.differential(s);
        let module = res.module(s);
        let max_t = max.n() + s;
        (0..=max_t)
            .into_maybe_par_iter()
            .map(|t| {
                if !res.has_computed_bidegree(Bidegree::s_t(s, t)) {
                    return Vec::new();
                }
                (0..module.number_of_gens_in_degree(t))
                    .map(|i| d.output(t, i).iter_nonzero().count() as u64)
                    .collect()
            })
            .collect()
    };

    let mut prev = row_nnz(1);
    for s in 2..max.s() {
        let d = res.differential(s);
        let module = res.module(s);
        let below = res.module(s - 1);
        let max_t = max.n() + s;
        let rows: Vec<Vec<(u64, u64)>> = (0..=max_t)
            .into_maybe_par_iter()
            .map(|t| {
                if !res.has_computed_bidegree(Bidegree::s_t(s, t)) {
                    return Vec::new();
                }
                (0..module.number_of_gens_in_degree(t))
                    .map(|i| {
                        let dg = d.output(t, i);
                        let mut pairs = 0;
                        for (gen_deg, gen_idx, _, slice) in below.iter_slices(t, dg.as_slice()) {
                            let nnz = slice.iter_nonzero().count() as u64;
                            if nnz > 0 {
                                pairs += nnz * prev[gen_deg as usize][gen_idx];
                            }
                        }
                        (pairs, dg.iter_nonzero().count() as u64)
                    })
                    .collect()
            })
            .collect();
        for (t, gens) in rows.iter().enumerate() {
            for (i, (pairs, nnz)) in gens.iter().enumerate() {
                writeln!(out, "{} {s} {i} {pairs} {nnz}", t as i32 - s)?;
            }
        }
        out.flush()?;
        eprintln!("PAIRS row s={s} done at {:.1}s", start.elapsed().as_secs_f64());
        prev = rows
            .into_iter()
            .map(|gens| gens.into_iter().map(|(_, nnz)| nnz).collect())
            .collect();
    }
    Ok(())
}
