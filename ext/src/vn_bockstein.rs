//! The $v_n$-Bockstein spectral sequence from $\mathrm{Ext}_P(Q^{(n-1)})[v_n]$ to $\mathrm{Ext}_P(Q^{(n)})$.

use std::collections::BTreeMap;

use algebra::MilnorAlgebra;
use maybe_rayon::prelude::*;

use crate::{
    algebraic_novikov::{
        ActMemo, Convention, InvertedSlice, QDown, QMono, cochain_blocks, differential_slice,
        invert_slice, qmono_len, untwisted_backbone_range, untwisted_filtrations,
        untwisted_max_length, untwisted_max_weight,
    },
    chain_complex::FreeChainComplex,
    filtered_sseq::{FilteredMap, Pages, Pair},
};

/// A cochain of $\mathrm{Hom}(C, Q^{(n)})$ at a fixed weight: the degree and index of the
/// generator of $C$ it is dual to, and the monomial.
type Cochain = (i32, u32, QMono);

/// The complexes $\mathrm{Hom}(C, Q_k^{(n)})$ of one stage, filtered by the exponent of $v_n$.
struct Stage<'a, CC> {
    /// The backbone.
    res: &'a CC,
    /// The bases of $Q_k^{(n)}$.
    q: QDown,
    /// The stage.
    n: usize,
}

impl<CC> Stage<'_, CC>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    /// The cochains of filtration `a` at weight `u` on $Q_{k_\max}$, in the order of
    /// [`cochain_blocks`].
    fn cochains(&self, a: i32, u: i32, k_max: u32) -> Vec<Cochain> {
        let q_dim = |d: i32| self.q.dimension(d, k_max);
        let (blocks, _) = cochain_blocks(self.res, q_dim, a, u);
        let mut out = Vec::new();
        for (t, &(_, gens)) in blocks.iter().enumerate() {
            let d = u - t as i32;
            for g in 0..gens {
                for idx in 0..q_dim(d) {
                    out.push((t as i32, g as u32, *self.q.monomial(d, idx)));
                }
            }
        }
        out
    }

    /// The filtered coboundary $\delta^a$ at weight `u` on $Q_{k_\max}$, with its source and
    /// target cochains.
    fn build(
        &self,
        a: i32,
        u: i32,
        k_max: u32,
        inverted: &InvertedSlice,
        memo: &ActMemo,
    ) -> (FilteredMap, Vec<Cochain>, Vec<Cochain>) {
        let (q, n) = (&self.q, self.n);
        let q_dim = |d: i32| q.dimension(d, k_max);
        let (tgt, _) = cochain_blocks(self.res, q_dim, a + 1, u);
        let sources = self.cochains(a, u, k_max);
        let targets = self.cochains(a + 1, u, k_max);
        let mut map = FilteredMap {
            source: sources.iter().map(|c| c.2[n] as i32).collect(),
            target: targets.iter().map(|c| c.2[n] as i32).collect(),
            columns: Vec::with_capacity(sources.len()),
        };
        for (t_i, i, mono) in &sources {
            let mut col = Vec::new();
            for (r, targets) in inverted.get(&(*t_i, *i)).into_iter().flatten() {
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
        (map, sources, targets)
    }
}

/// The lengths $k \leq$ `k_max` at which the complex on $Q_k$ changes: those some cochain has.
fn fresh_lengths(k_max: i32, cochains: &[&[Cochain]]) -> Vec<i32> {
    let lengths: std::collections::BTreeSet<u32> = cochains
        .iter()
        .flat_map(|c| c.iter().map(|c| qmono_len(&c.2)))
        .collect();
    (0..=k_max)
        .filter(|&k| k == 0 || lengths.contains(&(k as u32)))
        .collect()
}

/// The stage `n` and its bases, after checking the backbone reaches far enough.
fn stage<CC>(res: &CC, n: usize, max_n: i32, max_s: i32) -> Stage<'_, CC>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
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
    Stage { res, q, n }
}

/// The differential slice of $P_{a+1}$ that the coboundaries out of filtration `a` read.
fn slice<CC>(res: &CC, a: i32, max_n: i32, max_s: i32) -> InvertedSlice
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra>,
{
    let max_u = (max_n + max_s) / 2;
    invert_slice(
        &res.algebra(),
        &differential_slice(res, a, max_u.min((max_n + a + 1) / 2)),
    )
}

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
    let max_u = (max_n + max_s) / 2;
    let stage = stage(res, n, max_n, max_s);
    let mut pages = Pages::default();
    for a in 0..=max_s {
        let jobs: Vec<(i32, i32)> = (0..=max_u)
            .filter_map(|u| untwisted_max_length(max_n, max_s, a, u).map(|k| (u, k)))
            .collect();
        if jobs.is_empty() {
            continue;
        }
        let inverted = slice(res, a, max_n, max_s);
        let memo = ActMemo::default();
        let results: Vec<_> = jobs
            .into_maybe_par_iter()
            .map(|(u, k_max)| {
                let (map, sources, targets) = stage.build(a, u, k_max as u32, &inverted, &memo);
                let src_len: Vec<u32> = sources.iter().map(|c| qmono_len(&c.2)).collect();
                // The reductions for different k are independent.
                let pairs: BTreeMap<i32, Vec<Pair>> = fresh_lengths(k_max, &[&sources, &targets])
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
                pages.chains.entry((k, u, a)).or_insert_with(|| {
                    cochain_blocks(res, |d| stage.q.dimension(d, k as u32), a, u).1
                });
            }
        }
    }
    pages
}

/// The shape of a cocycle representing a class of $\mathrm{Ext}(Q^{(n)})$.
#[derive(Debug, Clone, Copy)]
pub struct CycleShape {
    /// The stem of the class.
    pub stem: i32,
    /// Its Adams filtration $a + k$.
    pub s: i32,
    /// Its weight.
    pub u: i32,
    /// The exponent of $v_n$ in its leading term: its filtration in the $v_n$-Bockstein
    /// spectral sequence.
    pub filtration: i32,
    /// How many exponents of $v_n$ below the leading one the cocycle reaches down to.
    pub depth: i32,
    /// The number of cochains in the cocycle.
    pub support: usize,
    /// The number of distinct generators of $C$ among them.
    pub generators: usize,
    /// The number of cochains in its degree.
    pub dim: usize,
    /// The number of generators of $C$ that cochains in its degree are dual to.
    pub dim_generators: usize,
}

/// A cocycle for every class of $\mathrm{Ext}(Q^{(n)})$ within stem `max_n` and `s <= max_s`, as
/// the persistence reduction of [`vn_bockstein`] leaves it, described by [`CycleShape`].
///
/// The reduction adds only sources of filtration at most that of the leading term, so the
/// cocycle is a leading term plus the zig-zag that the perturbation lemma would build from the
/// lower filtrations. It is one choice among many and its support is an upper bound.
pub fn cycle_shapes<CC>(res: &CC, n: usize, max_n: i32, max_s: i32) -> Vec<CycleShape>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    let max_u = (max_n + max_s) / 2;
    let stage = stage(res, n, max_n, max_s);
    let mut out = Vec::new();
    let mut below: Option<InvertedSlice> = None;
    for a in 0..=max_s {
        let inverted = slice(res, a, max_n, max_s);
        let memo = ActMemo::default();
        let jobs: Vec<(i32, i32)> = (0..=max_u)
            .filter_map(|u| untwisted_max_length(max_n, max_s, a, u).map(|k| (u, k)))
            .filter(|&(u, _)| 2 * u - a <= max_n)
            .collect();
        let shapes: Vec<CycleShape> = jobs
            .into_maybe_par_iter()
            .flat_map(|(u, k_max)| {
                let (map, sources, _) = stage.build(a, u, k_max as u32, &inverted, &memo);
                // δ^{a-1}, to know which cocycles are coboundaries.
                let lower = below.as_ref().and_then(|b| {
                    let k = untwisted_max_length(max_n, max_s, a - 1, u)?;
                    let (m, src, tgt) = stage.build(a - 1, u, k as u32, b, &memo);
                    Some((m, src, tgt))
                });
                let position: rustc_hash::FxHashMap<Cochain, usize> =
                    sources.iter().enumerate().map(|(i, c)| (*c, i)).collect();
                let len: Vec<u32> = sources.iter().map(|c| qmono_len(&c.2)).collect();
                let mut shapes = Vec::new();
                for k in 0..=k_max {
                    if a + k > max_s {
                        break;
                    }
                    let keep = |s: usize| len[s] <= k as u32;
                    let (_, cycles) = map.pairs_and_cycles(keep);
                    let killed: rustc_hash::FxHashSet<usize> = lower
                        .as_ref()
                        .map(|(m, src, tgt)| {
                            m.pairs(|s| qmono_len(&src[s].2) <= k as u32)
                                .iter()
                                .map(|p| position[&tgt[p.target]])
                                .collect()
                        })
                        .unwrap_or_default();
                    let dim = (0..sources.len()).filter(|&s| keep(s)).count();
                    let dim_generators = sources
                        .iter()
                        .filter(|c| qmono_len(&c.2) <= k as u32)
                        .map(|c| (c.0, c.1))
                        .collect::<rustc_hash::FxHashSet<_>>()
                        .len();
                    for (s, sum) in cycles {
                        if killed.contains(&s) {
                            continue;
                        }
                        let filtration = map.source[s];
                        let lowest = sum.iter().map(|&c| map.source[c as usize]).min().unwrap();
                        let generators = sum
                            .iter()
                            .map(|&c| (sources[c as usize].0, sources[c as usize].1))
                            .collect::<rustc_hash::FxHashSet<_>>()
                            .len();
                        shapes.push(CycleShape {
                            stem: 2 * u - a,
                            s: a + k,
                            u,
                            filtration,
                            depth: filtration - lowest,
                            support: sum.len(),
                            generators,
                            dim,
                            dim_generators,
                        });
                    }
                }
                shapes
            })
            .collect();
        out.extend(shapes);
        below = Some(inverted);
    }
    out
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
