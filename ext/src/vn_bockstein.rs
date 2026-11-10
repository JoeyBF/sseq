//! The $v_n$-Bockstein spectral sequence from $\mathrm{Ext}_P(Q^{(n-1)})[v_n]$ to $\mathrm{Ext}_P(Q^{(n)})$.

use std::collections::BTreeMap;

use algebra::MilnorAlgebra;
use maybe_rayon::prelude::*;

use crate::{
    algebraic_novikov::{
        ActMemo, Convention, InvertedSlice, QDown, cochain_blocks, differential_slice,
        invert_slice, qmono_len, untwisted_backbone_range, untwisted_filtrations,
        untwisted_max_length, untwisted_max_weight,
    },
    chain_complex::FreeChainComplex,
    filtered_sseq::{FilteredMap, Pages, Pair},
};

/// The spectral sequence of $\mathrm{Hom}(C, Q^{(n)})$, $Q^{(n)} = \mathbb{F}_2[v_0, \dots, v_n]$,
/// filtered by the exponent of $v_n$, keyed by the degree `(k, u, a)` of the untwisted complex.
///
/// $\psi$ never raises the exponent of $v_n$, so the monomials with exponent at most $j$ span a
/// subcomodule whose quotients are copies of $Q^{(n-1)}$ shifted by $v_n^j$: $E_1$ is
/// $\mathrm{Ext}(Q^{(n-1)})[v_n]$ and $E_\infty$ is $\mathrm{Ext}(Q^{(n)})$.
pub fn vn_bockstein<CC>(res: &CC, n: usize, max_n: i32, max_s: i32) -> Pages<(i32, i32, i32)>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    let algebra = res.algebra();
    let max_u = (max_n + max_s) / 2;
    let range = untwisted_backbone_range(max_n, max_s);
    assert!(
        res.has_computed_bidegree(range),
        "the backbone must be resolved through {range}"
    );
    let q = QDown::new(
        max_s as u32,
        |j| untwisted_max_weight(max_n, max_s, j as i32),
        Convention::Conjugate,
    )
    .restrict_to(n);

    // The filtered coboundary δ^a at weight u on Q_{k_max}, with each cochain's length.
    let build = |a: i32, u: i32, k_max: u32, inverted: &InvertedSlice, memo: &ActMemo| {
        let q_dim = |d: i32| q.dimension(d, k_max);
        let (src, _) = cochain_blocks(res, q_dim, a, u);
        let (tgt, _) = cochain_blocks(res, q_dim, a + 1, u);
        let mut map = FilteredMap::default();
        let mut src_len = Vec::new();
        let mut tgt_len = Vec::new();
        for (t, &(_, gens)) in tgt.iter().enumerate() {
            let d = u - t as i32;
            for _ in 0..gens {
                for idx in 0..q_dim(d) {
                    let m = q.monomial(d, idx);
                    map.target.push(m[n] as i32);
                    tgt_len.push(qmono_len(m));
                }
            }
        }
        for (t_i, &(_, gens)) in src.iter().enumerate() {
            let t_i = t_i as i32;
            let block = q_dim(u - t_i);
            for i in 0..gens {
                let terms = inverted.get(&(t_i, i as u32));
                for idx in 0..block {
                    let mono = q.monomial(u - t_i, idx);
                    map.source.push(mono[n] as i32);
                    src_len.push(qmono_len(mono));
                    let mut col = Vec::new();
                    for (r, targets) in terms.into_iter().flatten() {
                        let gammas = q.act_memo(*r, mono, memo, 1 << 24);
                        for &(t_l, l) in targets {
                            if t_l > u {
                                continue;
                            }
                            let (off, _) = tgt[t_l as usize];
                            let tb = q_dim(u - t_l);
                            for g in gammas.iter() {
                                col.push((off + l as usize * tb + q.index(u - t_l, g)) as u32);
                            }
                        }
                    }
                    col.sort_unstable();
                    // Terms reached twice cancel.
                    let mut dedup: Vec<u32> = Vec::with_capacity(col.len());
                    for c in col {
                        if dedup.last() == Some(&c) {
                            dedup.pop();
                        } else {
                            dedup.push(c);
                        }
                    }
                    map.columns.push(dedup);
                }
            }
        }
        (map, src_len, tgt_len)
    };

    let mut pages = Pages::default();
    for a in 0..=max_s {
        let jobs: Vec<(i32, i32)> = (0..=max_u)
            .filter_map(|u| untwisted_max_length(max_n, max_s, a, u).map(|k| (u, k)))
            .collect();
        if jobs.is_empty() {
            continue;
        }
        let inverted = invert_slice(
            &algebra,
            &differential_slice(res, a, max_u.min((max_n + a + 1) / 2)),
        );
        let memo = ActMemo::default();
        let results: Vec<_> = jobs
            .into_maybe_par_iter()
            .map(|(u, k_max)| {
                let (map, src_len, tgt_len) = build(a, u, k_max as u32, &inverted, &memo);
                // The complex for Q_k changes only at the lengths some cochain has, and the
                // reductions for different k are independent.
                let lengths: std::collections::BTreeSet<u32> =
                    src_len.iter().chain(&tgt_len).copied().collect();
                let fresh: Vec<i32> = (0..=k_max)
                    .filter(|&k| k == 0 || lengths.contains(&(k as u32)))
                    .collect();
                let pairs: BTreeMap<i32, Vec<Pair>> = fresh
                    .into_maybe_par_iter()
                    .map(|k| (k, map.pairs(|s| src_len[s] <= k as u32)))
                    .collect();
                (0..=k_max)
                    .map(|k| {
                        let (_, p) = pairs.range(..=k).next_back().unwrap();
                        let dim = src_len.iter().filter(|&&l| l <= k as u32).count();
                        (k, u, dim, p.clone())
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        for (k, u, dim, pairs) in results.into_iter().flatten() {
            pages.add((k, u, a), dim, (k, u, a + 1), &pairs);
        }
    }
    // Degrees reached only as targets still need their dimension.
    for k in 0..=max_s {
        for u in 0..=untwisted_max_weight(max_n, max_s, k) {
            for a in untwisted_filtrations(max_n, max_s, k, u) {
                pages
                    .chains
                    .entry((k, u, a))
                    .or_insert_with(|| cochain_blocks(res, |d| q.dimension(d, k as u32), a, u).1);
            }
        }
    }
    pages
}

/// The dimensions of $E_r$ in the trigrading `(stem, s, u)` of $\mathrm{Ext}(Q)$, within stem
/// `max_n` and `s <= max_s`.
pub fn page_dims(
    pages: &Pages<(i32, i32, i32)>,
    r: i32,
    max_n: i32,
    max_s: i32,
) -> BTreeMap<(i32, i32, i32), usize> {
    let mut out = BTreeMap::new();
    for k in 0..=max_s {
        for u in 0..=untwisted_max_weight(max_n, max_s, k) {
            for a in untwisted_filtrations(max_n, max_s, k, u) {
                let stem = 2 * u - a;
                if !(0..=max_n).contains(&stem) {
                    continue;
                }
                let d = pages.dim((k, u, a), r);
                if d > 0 {
                    out.insert((stem, a + k, u), d);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebraic_novikov::ctau_e2_untwisted_on;

    /// $E_\infty$ of every stage is the Ext of that stage, and at the last stage it is
    /// $\mathrm{Ext}_P(Q)$.
    #[test]
    fn converges_to_stage() {
        let (max_n, max_s) = (30, 15);
        let res = crate::nassau::Resolution::new(std::sync::Arc::new(
            crate::algebraic_novikov::unit_module(),
        ));
        res.compute_through_stem(untwisted_backbone_range(max_n, max_s));
        let full = ctau_e2_untwisted_on(&res, max_n, max_s);
        let mut prev_einf = None;
        for n in 0..=5 {
            let pages = vn_bockstein(&res, n, max_n, max_s);
            let e1 = page_dims(&pages, 1, max_n, max_s);
            let einf = page_dims(&pages, i32::MAX, max_n, max_s);
            if let Some(prev) = prev_einf {
                // E_1 = E_∞ of the previous stage, tensored with F_2[v_n].
                let vn = (1 << n) - 1;
                let mut expect: BTreeMap<(i32, i32, i32), usize> = BTreeMap::new();
                for (&(stem, s, u), &d) in &prev {
                    for j in 0.. {
                        let key = (stem + 2 * j * vn, s + j, u + j * vn);
                        if key.0 > max_n || key.1 > max_s {
                            break;
                        }
                        *expect.entry(key).or_insert(0) += d;
                    }
                }
                assert_eq!(e1, expect, "E_1 of stage {n}");
            }
            prev_einf = Some(einf);
        }
        let full: BTreeMap<_, _> = full.0.into_iter().collect();
        assert_eq!(prev_einf.unwrap(), full);
    }
}
