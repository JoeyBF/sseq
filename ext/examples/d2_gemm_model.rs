//! Sizes the d2 composites of a saved resolution as 1-bit GEMMs against right-multiplication slabs.

use std::{collections::HashMap, io::Write, sync::Arc, time::Instant};

use algebra::Algebra;
use ext::chain_complex::ChainComplex;
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

/// Usage: `d2_gemm_model <store> <max_n> <max_s> <out>`.
///
/// The composite of a generator g multiplies each op `x = d(g)_{g1}` (degree k) by each component
/// `y = d(g1)_{g2}` (degree e). As GEMMs: for each basis element S of `A_e` in some `y` of g1, one
/// row `x * slab_S`, `slab_S` being right multiplication by `P(S)` from `A_k` to pair elements of
/// degree `k + e` (`W(k + e)` bits): `dim A_k * W` bit multiply-adds, its result added into every
/// g2 whose `y` contains S. Writes, per generator g with s >= 2, `n s idx pairs rows gemm_macs
/// write_bits min_write_bits`: the walk's products, the GEMM rows, their multiply-adds, the bits
/// the rows add into composites, and the bits of the composite itself (one pair element per g2
/// hit). Also prints a histogram of M = the number of ops of one degree into one g1, the rows a
/// slab GEMM gets when batched by g1, weighted by multiply-adds.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 5, "usage: d2_gemm_model <store> <max_n> <max_s> <out>");
    let store = std::path::PathBuf::from(&args[1]);
    let max = Bidegree::n_s(args[2].parse()?, args[3].parse()?);

    let start = Instant::now();
    let res = Arc::new(ext::utils::construct("S_2@milnor", Some(store))?);
    res.compute_through_stem(max);
    eprintln!("MODEL open_s={:.1}", start.elapsed().as_secs_f64());
    let algebra = res.algebra();
    let top = max.n() + max.s() + 2;
    algebra.compute_basis(top);
    let dim: Vec<u64> = (0..=top).map(|d| algebra.dimension(d) as u64).collect();
    // Bits of a pair element of degree d: the product terms and every ys[k][l].
    let width: Vec<u64> = (0..=top as usize)
        .map(|d| {
            let mut b = dim[d];
            if d > 0 {
                for k in 0..=fp::prime::log2(d) {
                    let rem = d + 1 - (1 << k);
                    for l in 0..=fp::prime::log2(rem) {
                        b += dim[rem - (1 << l)];
                    }
                }
            }
            b
        })
        .collect();

    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[4])?);
    // M histogram: log2 bucket -> multiply-adds.
    let mut m_hist = [0f64; 24];

    for s in 2..max.s() {
        let (d, d_below) = (res.differential(s), res.differential(s - 1));
        let (module, mid, bot) = (res.module(s), res.module(s - 1), res.module(s - 2));
        let max_t = max.n() + s;
        // Per g1 of row s - 1: per component degree e, (distinct S, total nnz of the y's), and
        // the g2 it hits.
        type G1 = (Vec<(i32, u64, u64)>, Vec<(i32, usize)>);
        let g1_data: HashMap<(i32, usize), G1> = (0..=max_t)
            .into_maybe_par_iter()
            .flat_map(|t1| {
                if !res.has_computed_bidegree(Bidegree::s_t(s - 1, t1)) {
                    return Vec::new();
                }
                (0..mid.number_of_gens_in_degree(t1))
                    .map(|i| {
                        let dg1 = d_below.output(t1, i);
                        let mut by_e: HashMap<i32, (std::collections::HashSet<usize>, u64)> =
                            HashMap::new();
                        let mut g2s = Vec::new();
                        for (g2d, g2i, _, y) in bot.iter_slices(t1, dg1.as_slice()) {
                            let mut any = false;
                            let e = t1 - g2d;
                            let entry = by_e.entry(e).or_default();
                            for (sidx, _) in y.iter_nonzero() {
                                entry.0.insert(sidx);
                                entry.1 += 1;
                                any = true;
                            }
                            if any {
                                g2s.push((g2d, g2i));
                            }
                        }
                        let per_e = by_e
                            .into_iter()
                            .map(|(e, (set, n))| (e, set.len() as u64, n))
                            .collect();
                        ((t1, i), (per_e, g2s))
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let rows: Vec<(String, Vec<(u64, f64)>)> = (0..=max_t)
            .into_maybe_par_iter()
            .flat_map(|t| {
                if !res.has_computed_bidegree(Bidegree::s_t(s, t)) {
                    return Vec::new();
                }
                (0..module.number_of_gens_in_degree(t))
                    .map(|idx| {
                        let dg = d.output(t, idx);
                        let (mut pairs, mut nrows, mut macs, mut writes) = (0u64, 0u64, 0f64, 0f64);
                        let mut hit: std::collections::HashSet<(i32, usize)> = Default::default();
                        let mut ms = Vec::new();
                        for (g1d, g1i, _, x) in mid.iter_slices(t, dg.as_slice()) {
                            let nnz_x = x.iter_nonzero().count() as u64;
                            if nnz_x == 0 {
                                continue;
                            }
                            let Some((per_e, g2s)) = g1_data.get(&(g1d, g1i)) else {
                                continue;
                            };
                            let k = (t - g1d) as usize;
                            let mut m = 0f64;
                            for &(e, distinct, nnz) in per_e {
                                let w = width[k + e as usize] as f64;
                                pairs += nnz_x * nnz;
                                nrows += distinct;
                                m += dim[k] as f64 * distinct as f64 * w;
                                writes += nnz as f64 * w;
                            }
                            macs += m;
                            ms.push(((g1d, g1i, k), m));
                            hit.extend(g2s.iter().copied());
                        }
                        let min_write: f64 =
                            hit.iter().map(|&(g2d, _)| width[(t - g2d) as usize] as f64).sum();
                        let line = format!(
                            "{} {s} {idx} {pairs} {nrows} {macs:.4e} {writes:.4e} {min_write:.4e}",
                            t - s
                        );
                        // Report each (g1, k) once per g; M is aggregated below.
                        let ms = ms
                            .into_iter()
                            .map(|((g1d, g1i, k), m)| {
                                ((g1d as u64) << 44 | (g1i as u64) << 12 | k as u64, m)
                            })
                            .collect();
                        (line, ms)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // M per (g1, k): how many generators of row s have an op of degree k into g1.
        let mut m_count: HashMap<u64, (u32, f64)> = HashMap::new();
        for (line, ms) in &rows {
            writeln!(out, "{line}")?;
            for &(key, m) in ms {
                let e = m_count.entry(key).or_default();
                e.0 += 1;
                e.1 += m;
            }
        }
        for (count, m) in m_count.values() {
            m_hist[(32 - count.leading_zeros()) as usize] += m;
        }
        out.flush()?;
        eprintln!("MODEL row s={s} done at {:.1}s", start.elapsed().as_secs_f64());
    }
    let total: f64 = m_hist.iter().sum();
    eprintln!("MODEL M histogram (rows per (g1, k) batch, share of multiply-adds):");
    for (b, m) in m_hist.iter().enumerate() {
        if *m > 0.0 {
            eprintln!("  M in [{}, {}): {:.3}", 1u64 << b.saturating_sub(1), 1u64 << b, m / total);
        }
    }
    Ok(())
}
