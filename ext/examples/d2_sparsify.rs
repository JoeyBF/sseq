//! Measures how much sparser a minimal resolution's differentials can be made, weighted by d2 cost.

use std::{collections::HashMap, sync::Arc, time::Instant};

use algebra::module::{Module, homomorphism::ModuleHomomorphism};
use ext::chain_complex::ChainComplex;
use fp::{matrix::Matrix, vector::FpVector};
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

/// Usage: `d2_sparsify <max_n> <max_s> <n,s> [<n,s> ...]`.
///
/// Resolves the sphere through `(max_n, max_s)` in memory. At each listed bidegree, `t = n + s`, a
/// generator g of `F_s` may be replaced by `g + sum a_j g_j` (the `g_j` of lower degree), which
/// changes `d(g)` by any element of `D = d((A_+ F_s)_t)` and leaves the resolution minimal. Each
/// `d(g)` is reduced against the row-reduced `D` with the columns ordered by weight, heaviest
/// first, so that the heaviest columns are the pivots it is cleared on. The weight of a column of
/// `F_{s-1}(t)` is `nnz(d(g1))` for its generator g1, so the weighted support of `d(g)` is the
/// number of products its d2 composite takes. Prints, per bidegree, the summed `nnz(d(g))` and
/// weighted support before and after.
///
/// This is first order: changing `d` on `F_s` also changes how `d_{s+1}` is written in the
/// generators of `F_s`, which this does not follow.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 4, "usage: d2_sparsify <max_n> <max_s> <n,s> ...");
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
    eprintln!("SPARSE resolve_s={:.1}", start.elapsed().as_secs_f64());
    let p = res.prime();

    for b in targets {
        let (s, t) = (b.s(), b.t());
        let (module, below) = (res.module(s), res.module(s - 1));
        let (d, d_below) = (res.differential(s), res.differential(s - 1));
        let num_gens = module.number_of_gens_in_degree(t);
        if num_gens == 0 {
            continue;
        }
        let cols = below.dimension(t);

        // The weight of each column: nnz(d(g1)) for its generator g1.
        let mut gen_nnz: HashMap<(i32, usize), u64> = HashMap::new();
        let weight: Vec<u64> = (0..cols)
            .map(|c| {
                let op = below.index_to_op_gen(t, c);
                let key = (op.generator_degree, op.generator_index);
                *gen_nnz.entry(key).or_insert_with(|| {
                    d_below.output(key.0, key.1).iter_nonzero().count() as u64
                })
            })
            .collect();
        // Heaviest first; `perm[new] = old`, `inv[old] = new`.
        let mut perm: Vec<usize> = (0..cols).collect();
        perm.sort_by_key(|&c| (std::cmp::Reverse(weight[c]), c));
        let mut inv = vec![0; cols];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let permute = |v: &FpVector| {
            let mut out = FpVector::new(p, cols);
            for (i, _) in v.iter_nonzero() {
                out.set_entry(inv[i], 1);
            }
            out
        };

        // D: d of every basis element of F_s(t) that is not a generator of degree t.
        let decomposables: Vec<usize> = (0..module.dimension(t))
            .filter(|&i| module.index_to_op_gen(t, i).generator_degree < t)
            .collect();
        let rows: Vec<FpVector> = decomposables
            .into_maybe_par_iter()
            .map(|i| {
                let mut v = FpVector::new(p, cols);
                d.apply_to_basis_element(v.as_slice_mut(), 1, t, i);
                permute(&v)
            })
            .collect();
        let mut m = Matrix::new(p, rows.len(), cols);
        for (k, v) in rows.iter().enumerate() {
            m.row_mut(k).add(v.as_slice(), 1);
        }
        let rank = m.row_reduce();

        let (mut nnz_before, mut nnz_after, mut cost_before, mut cost_after) = (0u64, 0, 0, 0);
        for g in 0..num_gens {
            let mut v = permute(&d.output(t, g));
            nnz_before += v.iter_nonzero().count() as u64;
            cost_before += v.iter_nonzero().map(|(c, _)| weight[perm[c]]).sum::<u64>();
            let pivot_cols: Vec<usize> = v
                .iter_nonzero()
                .map(|(c, _)| c)
                .filter(|&c| m.pivots()[c] >= 0)
                .collect();
            for c in pivot_cols {
                let row = m.pivots()[c] as usize;
                v.as_slice_mut().add(m.row(row), 1);
            }
            nnz_after += v.iter_nonzero().count() as u64;
            cost_after += v.iter_nonzero().map(|(c, _)| weight[perm[c]]).sum::<u64>();
        }
        println!(
            "SPARSE ({},{s}) gens={num_gens} cols={cols} rank_D={rank} nnz {nnz_before} -> \
             {nnz_after} ({:.2}x) cost {cost_before} -> {cost_after} ({:.2}x)",
            b.n(),
            nnz_before as f64 / nnz_after.max(1) as f64,
            cost_before as f64 / cost_after.max(1) as f64,
        );
    }
    Ok(())
}
