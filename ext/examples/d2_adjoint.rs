//! Measures how much of the secondary resolution the d2 of a single class depends on.

use std::{collections::HashSet, sync::Arc, time::Instant};

use algebra::module::{Module, homomorphism::ModuleHomomorphism};
use ext::chain_complex::ChainComplex;
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

/// A generator of a free module: degree and index.
type Gen = (i32, usize);

/// Usage: `d2_adjoint <max_n> <max_s> <n,s> [<n,s> ...]`.
///
/// Resolves the sphere through `(max_n, max_s)` in memory, then for every class x at each listed
/// bidegree `(n, s)`, `t = n + s`, follows x backwards through the homotopy recursion of
/// `SecondaryResolution`:
///
/// - `h_r(g) = QI_{r-2}(A_{r-1}(dg) + h_{r-1}(dg))`, `QI_{r-2}` inverting `d_{r-2}`, and
/// - `d2(x)(g') = x(h_{s+2}(g'))` for `g'` in row `s + 2`, degree `t + 1`.
///
/// `B_1` is the set of generators of `F_{s-1}` whose blocks the functional `x QI_s` (at degree `t`)
/// reads; `B_k` (k >= 2) those of `F_{s-k}` that the quasi-inverses of `d_{s-k+1}` (at every degree
/// up to `t`) need to produce the coordinates of `B_{k-1}`. At level k only the components on `B_k`
/// of the composites of row `s + 2 - k` matter. Prints, per class and level, `|B_k|` against the
/// generators of `F_{s-k}` of degree at most `t`, and the share of the composite pairs (of
/// generators of degree at most `t + 1`) that land in `B_k`.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 4, "usage: d2_adjoint <max_n> <max_s> <n,s> ...");
    let max = Bidegree::n_s(args[1].parse()?, args[2].parse()?);
    let targets: Vec<Bidegree> = args[3..]
        .iter()
        .map(|a| {
            let (n, s) = a.split_once(',').expect("n,s");
            Ok(Bidegree::n_s(n.parse()?, s.parse()?))
        })
        .collect::<anyhow::Result<_>>()?;

    let start = Instant::now();
    let res = Arc::new(ext::utils::construct("S_2@milnor", None)?);
    res.compute_through_stem(max);
    eprintln!("ADJ resolve_s={:.1}", start.elapsed().as_secs_f64());

    for b in targets {
        let (s, t) = (b.s(), b.t());
        let module = res.module(s);
        for idx in 0..module.number_of_gens_in_degree(t) {
            let col = module.generator_offset(t, t, idx);
            // Level 1: the coordinates of F_{s-1}(t) whose preimage row has x's column set.
            let ds = res.differential(s);
            let qi = ds.quasi_inverse(t).expect("quasi-inverse in memory");
            let below = res.module(s - 1);
            let mut blocks: HashSet<Gen> = HashSet::new();
            for_each_pivot(qi, below.dimension(t), |i, row| {
                if row.entry(col) != 0 {
                    let op = below.index_to_op_gen(t, i);
                    blocks.insert((op.generator_degree, op.generator_index));
                }
            });
            report(&res, b, idx, 1, &blocks);

            // Level k: the quasi-inverse of d_{s-k+1}, F_{s-k}(t') -> F_{s-k+1}(t'), restricted
            // to output columns in the blocks of B_{k-1}.
            for k in 2..s {
                let upper = res.module(s - k + 1);
                let lower = res.module(s - k);
                let d = res.differential(s - k + 1);
                let next: HashSet<Gen> = (lower.min_degree()..=t)
                    .into_maybe_par_iter()
                    .map(|tp| {
                        let mut found = HashSet::new();
                        let Some(qi) = d.quasi_inverse(tp) else {
                            return found;
                        };
                        let ranges = column_ranges(&upper, tp, &blocks);
                        if ranges.is_empty() {
                            return found;
                        }
                        for_each_pivot(qi, lower.dimension(tp), |i, row| {
                            if ranges.iter().any(|&(a, e)| !row.restrict(a, e).is_zero()) {
                                let op = lower.index_to_op_gen(tp, i);
                                found.insert((op.generator_degree, op.generator_index));
                            }
                        });
                        found
                    })
                    .reduce(HashSet::new, |mut a, b| {
                        a.extend(b);
                        a
                    });
                blocks = next;
                report(&res, b, idx, k, &blocks);
                if blocks.is_empty() {
                    break;
                }
            }
        }
        eprintln!("ADJ done ({},{}) at {:.1}s", b.n(), s, start.elapsed().as_secs_f64());
    }
    Ok(())
}

/// Calls `f(i, preimage row)` for each coordinate `i` of the quasi-inverse's input that has one.
fn for_each_pivot(
    qi: &fp::matrix::QuasiInverse,
    input_dim: usize,
    mut f: impl FnMut(usize, fp::vector::FpSlice),
) {
    let preimage = qi.preimage();
    let mut row = 0;
    for i in 0..input_dim {
        if let Some(pivots) = qi.pivots()
            && (i >= pivots.len() || pivots[i] < 0)
        {
            continue;
        }
        if row >= preimage.rows() {
            break;
        }
        f(i, preimage.row(row));
        row += 1;
    }
}

/// The column ranges of the blocks of `blocks` in degree `t` of `module`, adjacent ones merged.
fn column_ranges<A: algebra::Algebra>(
    module: &algebra::module::FreeModule<A>,
    t: i32,
    blocks: &HashSet<Gen>,
) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = blocks
        .iter()
        .filter(|&&(deg, _)| deg <= t)
        .map(|&(deg, idx)| {
            let a = module.generator_offset(t, deg, idx);
            (a, a + module.algebra().dimension(t - deg))
        })
        .filter(|&(a, e)| e > a)
        .collect();
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, e) in ranges {
        match merged.last_mut() {
            Some(last) if last.1 == a => last.1 = e,
            _ => merged.push((a, e)),
        }
    }
    merged
}

/// Prints level `k` of class `idx` at `b`: `|B_k|` against the generators of `F_{s-k}` of degree
/// at most `t`, and the share of the pairs of the composites of row `s + 2 - k` (generators of
/// degree at most `t + 1`) whose target generator is in `B_k`.
fn report(
    res: &Arc<ext::utils::QueryModuleResolution>,
    b: Bidegree,
    idx: usize,
    k: i32,
    blocks: &HashSet<Gen>,
) {
    let (s, t) = (b.s(), b.t());
    let lower = res.module(s - k);
    let total_gens: usize = (lower.min_degree()..=t)
        .map(|d| lower.number_of_gens_in_degree(d))
        .sum();
    // Pairs of the composites of row r = s + 2 - k: for g in row r, g1 in d(g), g2 in d(g1),
    // nnz(op from g to g1) * nnz(op from g1 to g2).
    let r = s + 2 - k;
    let (mut pairs_all, mut pairs_needed) = (0u64, 0u64);
    if r >= 2 && r <= res.next_homological_degree() - 1 {
        let (dr, dr1) = (res.differential(r), res.differential(r - 1));
        let (mid, bot) = (res.module(r - 1), res.module(r - 2));
        let mut w_in: std::collections::HashMap<Gen, u64> = Default::default();
        let top = res.module(r);
        for deg in top.min_degree()..=(t + 1).min(top.max_computed_degree()) {
            for i in 0..top.number_of_gens_in_degree(deg) {
                let dg = dr.output(deg, i);
                for (gd, gi, _, op) in mid.iter_slices(deg, dg.as_slice()) {
                    let nnz = op.iter_nonzero().count() as u64;
                    if nnz > 0 {
                        *w_in.entry((gd, gi)).or_default() += nnz;
                    }
                }
            }
        }
        for (&(gd, gi), &w) in &w_in {
            let dg1 = dr1.output(gd, gi);
            for (hd, hi, _, op) in bot.iter_slices(gd, dg1.as_slice()) {
                let nnz = op.iter_nonzero().count() as u64;
                pairs_all += w * nnz;
                if blocks.contains(&(hd, hi)) {
                    pairs_needed += w * nnz;
                }
            }
        }
    }
    println!(
        "ADJ x=({},{},{idx}) level={k} row={} gens_needed={} gens_total={} gens_frac={:.3} \
         composite_row={r} pairs_needed={pairs_needed} pairs_total={pairs_all} pairs_frac={:.3}",
        b.n(),
        s,
        s - k,
        blocks.len(),
        total_gens,
        blocks.len() as f64 / total_gens.max(1) as f64,
        pairs_needed as f64 / pairs_all.max(1) as f64
    );
}
