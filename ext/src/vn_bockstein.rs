//! The $v_n$-Bockstein spectral sequence from $\mathrm{Ext}_P(Q^{(n-1)})[v_n]$ to $\mathrm{Ext}_P(Q^{(n)})$.

use std::collections::BTreeMap;

use algebra::MilnorAlgebra;
use maybe_rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    algebraic_novikov::{
        Convention, InvertedSlice, PackedXi, QDown, QMono, cochain_blocks, differential_slice,
        invert_slice, qmono_len, untwisted_backbone_range, untwisted_filtrations,
        untwisted_max_length, untwisted_max_weight, xi_degree,
    },
    chain_complex::FreeChainComplex,
    filtered_sseq::{FilteredMap, Pages, Pair},
};

/// A cochain of $\mathrm{Hom}(C, Q^{(n)})$ at a fixed weight: the degree and index of the
/// generator of $C$ it is dual to, and the monomial.
pub(crate) type Cochain = (i32, u32, QMono);

/// A layer of $Q$ with the filtration whose spectral sequence builds it from a smaller one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// $Q^{(n)} = \mathbb{F}_2[v_0, \dots, v_n]$, filtered by the exponent of $v_n$: from
    /// $\mathrm{Ext}(Q^{(n-1)})[v_n]$.
    Adjoin(usize),
    /// $Q/I_m = \mathbb{F}_2[v_m, v_{m+1}, \dots]$, filtered by the powers of $v_m$, which is
    /// primitive there: from $\mathrm{Ext}(Q/I_{m+1})[v_m]$.
    Quotient(usize),
}

impl Layer {
    /// Whether `m` is in the basis of the layer.
    fn contains(self, m: &QMono) -> bool {
        match self {
            Self::Adjoin(n) => m[n + 1..].iter().all(|&e| e == 0),
            Self::Quotient(lo) => m[1..lo.max(1)].iter().all(|&e| e == 0),
        }
    }

    /// Whether the cochains on $Q_k$ are those of length exactly $k$.
    fn exact_length(self) -> bool {
        matches!(self, Self::Quotient(lo) if lo > 0)
    }

    /// Whether a monomial of length `len` is a cochain of the layer on $Q_k$: for a quotient
    /// by $v_0$, only those without $v_0$.
    fn on_length(self, len: u32, k: u32) -> bool {
        if self.exact_length() {
            len == k
        } else {
            len <= k
        }
    }

    /// The filtration of `m`: increasing, with $\delta$ never raising it. For $Q/I_0$ the power of
    /// $v_0$ is the length deficit, so the length serves, shifted by a constant.
    fn filtration(self, m: &QMono) -> i32 {
        match self {
            Self::Adjoin(n) => m[n] as i32,
            Self::Quotient(0) => qmono_len(m) as i32,
            Self::Quotient(lo) => -(m[lo] as i32),
        }
    }
}

/// The complexes $\mathrm{Hom}(C, Q_k)$ of one [`Layer`].
pub(crate) struct Stage<'a, CC> {
    /// The backbone.
    res: &'a CC,
    /// The coaction on each basis monomial, by degree and index, cut down to the operations
    /// [`Self::build`] is given; filled by [`Self::bound_by`].
    psi: Vec<Vec<crate::algebraic_novikov::Psi<u32>>>,
    /// The number of each operation in the backbone, as `psi` keys them.
    op_ids: FxHashMap<PackedXi, u32>,
    /// The bases of the layer.
    q: QDown,
    /// The layer.
    layer: Layer,
}

impl<CC> Stage<'_, CC>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    /// Tabulates the coaction for [`Self::build`], given every operation it will meet in
    /// `slices` and the longest monomial it reads at each `(a, u)`, or `None` where it reads
    /// nothing.
    pub(crate) fn bound_by(
        mut self,
        slices: &[InvertedSlice],
        max_len: impl Fn(i32, i32) -> Option<u32>,
    ) -> Self {
        let max_u = self.q.max_degree();
        self.tabulate(slices, max_len, 0..=max_u);
        self
    }

    /// [`Self::bound_by`] for the builds at the weights in `weights` only, replacing the table.
    ///
    /// A weight u reads degree d only through the generators of degree u - d, so a narrow band of
    /// weights needs, in each degree, only the operations of a narrow band of generators: the
    /// table shrinks with the band, at the cost of one walk per band.
    pub(crate) fn tabulate(
        &mut self,
        slices: &[InvertedSlice],
        max_len: impl Fn(i32, i32) -> Option<u32>,
        weights: std::ops::RangeInclusive<i32>,
    ) {
        self.psi = Vec::new();
        // Every operation, numbered, and the operations on the generators of each degree.
        let mut op_ids: FxHashMap<PackedXi, u32> = FxHashMap::default();
        let mut ops_by_t: FxHashMap<i32, Vec<u32>> = FxHashMap::default();
        for slice in slices {
            for (&(t, _), terms) in slice {
                for (r, _) in terms {
                    let next = op_ids.len() as u32;
                    let id = *op_ids.entry(*r).or_insert(next);
                    ops_by_t.entry(t).or_default().push(id);
                }
            }
        }
        // The longest monomial read in each degree, and the operations acting on it: a block at
        // (a, t) of weight u reads degree u - t with the operations on the generators of degree t.
        let max_u = self.q.max_degree();
        let mut wanted = vec![-1i32; max_u as usize + 1];
        let mut allowed = vec![vec![false; op_ids.len()]; max_u as usize + 1];
        for a in 0..slices.len() as i32 {
            for u in weights.clone() {
                let Some(k) = max_len(a, u) else { continue };
                for t in 0..=u {
                    if self
                        .res
                        .number_of_gens_in_bidegree(sseq::coordinates::Bidegree::s_t(a, t))
                        > 0
                    {
                        let d = (u - t) as usize;
                        wanted[d] = wanted[d].max(k as i32);
                        for &id in ops_by_t.get(&t).into_iter().flatten() {
                            allowed[d][id as usize] = true;
                        }
                    }
                }
            }
        }
        self.psi = self.q.psi_table(&op_ids, &allowed, &wanted);
        self.op_ids = op_ids;
    }

    /// The cochains of filtration `a` at weight `u` on $Q_{k_\max}$, in the order of
    /// [`cochain_blocks`].
    pub(crate) fn cochains(&self, a: i32, u: i32, k_max: u32) -> Vec<Cochain> {
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
    pub(crate) fn build(
        &self,
        a: i32,
        u: i32,
        k_max: u32,
        inverted: &InvertedSlice,
    ) -> (FilteredMap, Vec<Cochain>, Vec<Cochain>) {
        self.build_map(a, u, k_max, inverted, (a + 1, u))
    }

    /// The pullback along a map of the backbone from filtration `target.0` to `a`, given by
    /// `inverted` as [`invert_slice`] gives it, from the cochains of filtration `a` at weight `u`
    /// to those of filtration `target.0` at weight `target.1`. The coboundary is the pullback
    /// along the differential; a chain map lifting a class of internal degree t gives the
    /// product with that class, shifting the weight by t.
    pub(crate) fn build_map(
        &self,
        a: i32,
        u: i32,
        k_max: u32,
        inverted: &InvertedSlice,
        target: (i32, i32),
    ) -> (FilteredMap, Vec<Cochain>, Vec<Cochain>) {
        let (q, layer) = (&self.q, self.layer);
        let (a_out, u_out) = target;
        let q_dim = |d: i32| q.dimension(d, k_max);
        let (tgt, _) = cochain_blocks(self.res, q_dim, a_out, u_out);
        let sources = self.cochains(a, u, k_max);
        let targets = self.cochains(a_out, u_out, k_max);
        let mut map = FilteredMap {
            source: sources.iter().map(|c| layer.filtration(&c.2)).collect(),
            target: targets.iter().map(|c| layer.filtration(&c.2)).collect(),
            columns: Vec::with_capacity(sources.len()),
        };
        map.columns = vec![Vec::new(); sources.len()];
        let (src, _) = cochain_blocks(self.res, q_dim, a, u);
        for (t_i, &(src_off, gens)) in src.iter().enumerate() {
            let t_i = t_i as i32;
            let block = q_dim(u - t_i);
            if block == 0 || gens == 0 {
                continue;
            }
            // The operations any generator of this degree needs, each acted once per monomial.
            let mut ops: Vec<PackedXi> = Vec::new();
            let mut op_index: FxHashMap<PackedXi, usize> = FxHashMap::default();
            let terms: Vec<Vec<(usize, &[(i32, u32)])>> = (0..gens)
                .map(|i| {
                    inverted
                        .get(&(t_i, i as u32))
                        .into_iter()
                        .flatten()
                        .filter(|(_, targets)| targets.iter().any(|&(t_l, _)| t_l <= u_out))
                        .map(|(r, targets)| {
                            let k = *op_index.entry(*r).or_insert_with(|| {
                                ops.push(*r);
                                ops.len() - 1
                            });
                            (k, targets.as_slice())
                        })
                        .collect()
                })
                .collect();
            let mut images: Vec<Vec<usize>> = vec![Vec::new(); ops.len()];
            // Each operation's number in the table, and back.
            let ids: Vec<u32> = ops.iter().map(|r| self.op_ids[r]).collect();
            let by_id: FxHashMap<u32, usize> =
                ids.iter().enumerate().map(|(k, &id)| (id, k)).collect();
            for idx in 0..block {
                let mono = q.monomial(u - t_i, idx);
                // P(R) x_i lies in degree t_i + |R|, the degree of every z_l it appears in.
                let d_out = |r: PackedXi| u - t_i - xi_degree(r);
                let psi = &self.psi[(u - t_i) as usize][idx];
                let len = qmono_len(mono);
                // In a quotient, terms outside its basis vanish.
                let fill = |image: &mut Vec<usize>, r: PackedXi, gammas: &[u32]| {
                    let d = d_out(r);
                    image.clear();
                    image.extend(
                        gammas
                            .iter()
                            .map(|&g| g as usize)
                            .filter(|&g| layer.on_length(qmono_len(q.monomial(d, g)), len)),
                    );
                };
                // Walk whichever is shorter: the operations, or the terms of ψ(m).
                if psi.len() < ops.len() {
                    for image in &mut images {
                        image.clear();
                    }
                    for (id, gammas) in psi.iter() {
                        if let Some(&k) = by_id.get(&id) {
                            fill(&mut images[k], ops[k], gammas);
                        }
                    }
                } else {
                    for (k, &r) in ops.iter().enumerate() {
                        fill(&mut images[k], r, psi.get(ids[k]));
                    }
                }
                for (i, gen_terms) in terms.iter().enumerate() {
                    let mut col = Vec::new();
                    for &(k, targets) in gen_terms {
                        if images[k].is_empty() {
                            continue;
                        }
                        for &(t_l, l) in targets {
                            if t_l > u_out {
                                continue;
                            }
                            let (off, _) = tgt[t_l as usize];
                            let tb = q_dim(u_out - t_l);
                            col.extend(
                                images[k].iter().map(|g| (off + l as usize * tb + g) as u32),
                            );
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
                    map.columns[src_off + i * block + idx] = dedup;
                }
            }
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

/// The bases of `layer`, after checking the backbone reaches far enough.
pub(crate) fn stage<CC>(res: &CC, layer: Layer, max_n: i32, max_s: i32) -> Stage<'_, CC>
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
    .restrict(|m| layer.contains(m));
    Stage {
        res,
        psi: Vec::new(),
        op_ids: FxHashMap::default(),
        q,
        layer,
    }
}

/// The differential slice of $P_{a+1}$ that the coboundaries out of filtration `a` read.
pub(crate) fn slice<CC>(res: &CC, a: i32, max_n: i32, max_s: i32) -> InvertedSlice
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra>,
{
    let max_u = (max_n + max_s) / 2;
    invert_slice(
        &res.algebra(),
        &differential_slice(res, a, max_u.min((max_n + a + 1) / 2)),
    )
}

/// The spectral sequence of $\mathrm{Hom}(C, L)$ for the [`Layer`] $L$, keyed by the degree
/// `(k, u, a)` of the untwisted complex.
///
/// For [`Layer::Adjoin`], $\psi$ never raises the exponent of $v_n$, so the monomials with
/// exponent at most $j$ span a subcomodule whose quotients are copies of $Q^{(n-1)}$ shifted by
/// $v_n^j$: $E_1$ is $\mathrm{Ext}(Q^{(n-1)})[v_n]$ and $E_\infty$ is $\mathrm{Ext}(Q^{(n)})$.
/// For [`Layer::Quotient`], the multiples of $v_m^j$ form a subcomodule with quotients copies of
/// $Q/I_{m+1}$: $E_1$ is $\mathrm{Ext}(Q/I_{m+1})[v_m]$ and $E_\infty$ is $\mathrm{Ext}(Q/I_m)$.
pub fn vn_bockstein<CC>(res: &CC, layer: Layer, max_n: i32, max_s: i32) -> Pages<(i32, i32, i32)>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    let max_u = (max_n + max_s) / 2;
    let all: Vec<InvertedSlice> = (0..=max_s).map(|a| slice(res, a, max_n, max_s)).collect();
    let stage = stage(res, layer, max_n, max_s).bound_by(&all, |a, u| {
        untwisted_max_length(max_n, max_s, a, u).map(|k| k as u32)
    });
    let mut pages = Pages::default();
    for a in 0..=max_s {
        let jobs: Vec<(i32, i32)> = (0..=max_u)
            .filter_map(|u| untwisted_max_length(max_n, max_s, a, u).map(|k| (u, k)))
            .collect();
        if jobs.is_empty() {
            continue;
        }
        let inverted = &all[a as usize];
        let results: Vec<_> = jobs
            .into_maybe_par_iter()
            .map(|(u, k_max)| {
                let (map, sources, targets) = stage.build(a, u, k_max as u32, inverted);
                let src_len: Vec<u32> = sources.iter().map(|c| qmono_len(&c.2)).collect();
                // The reductions for different k are independent.
                let pairs: BTreeMap<i32, Vec<Pair>> = fresh_lengths(k_max, &[&sources, &targets])
                    .into_maybe_par_iter()
                    .map(|k| (k, map.pairs(|s| layer.on_length(src_len[s], k as u32))))
                    .collect();
                (0..=k_max)
                    .map(|k| {
                        let p = match pairs.get(&k) {
                            Some(p) => p,
                            // Without cochains of length k, a layer on exact lengths is empty
                            // there and any other repeats the last length that has some.
                            None if layer.exact_length() => &Vec::new(),
                            None => pairs.range(..=k).next_back().unwrap().1,
                        };
                        let dim = src_len
                            .iter()
                            .filter(|&&l| layer.on_length(l, k as u32))
                            .count();
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
                    stage
                        .cochains(a, u, k as u32)
                        .iter()
                        .filter(|c| layer.on_length(qmono_len(&c.2), k as u32))
                        .count()
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

/// One differential of the spectral sequence of a [`Layer`], placed by the degree of its source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Differential {
    /// The length $k$ of the complex $\mathrm{Hom}(C, L_k)$ it lives on.
    pub k: i32,
    /// The weight.
    pub u: i32,
    /// The filtration of the backbone; the target is one higher.
    pub a: i32,
    /// The filtration of the source in the spectral sequence.
    pub from: i32,
    /// The filtration of the target.
    pub to: i32,
}

/// Every differential of [`vn_bockstein`] within stem `max_n` and `s <= max_s`, with the
/// filtrations of its source and target, for locating each end on $E_1$.
pub fn differentials<CC>(res: &CC, layer: Layer, max_n: i32, max_s: i32) -> Vec<Differential>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    let max_u = (max_n + max_s) / 2;
    let all: Vec<InvertedSlice> = (0..=max_s).map(|a| slice(res, a, max_n, max_s)).collect();
    let stage = stage(res, layer, max_n, max_s).bound_by(&all, |a, u| {
        untwisted_max_length(max_n, max_s, a, u).map(|k| k as u32)
    });
    let jobs: Vec<(i32, i32, i32)> = (0..=max_s)
        .flat_map(|a| {
            (0..=max_u)
                .filter_map(move |u| untwisted_max_length(max_n, max_s, a, u).map(|k| (a, u, k)))
        })
        .collect();
    jobs.into_maybe_par_iter()
        .flat_map(|(a, u, k_max)| {
            let (map, sources, _) = stage.build(a, u, k_max as u32, &all[a as usize]);
            let len: Vec<u32> = sources.iter().map(|c| qmono_len(&c.2)).collect();
            let mut out = Vec::new();
            for k in 0..=k_max.min(max_s - a) {
                for p in map.pairs(|s| layer.on_length(len[s], k as u32)) {
                    out.push(Differential {
                        k,
                        u,
                        a,
                        from: map.source[p.source],
                        to: map.target[p.target],
                    });
                }
            }
            out
        })
        .collect()
}

/// A cocycle for every class of $\mathrm{Ext}(Q^{(n)})$ within stem `max_n` and `s <= max_s`, as
/// the persistence reduction of [`vn_bockstein`] leaves it, described by [`CycleShape`].
///
/// The reduction adds only sources of filtration at most that of the leading term, so the
/// cocycle is a leading term plus the zig-zag that the perturbation lemma would build from the
/// lower filtrations. It is one choice among many and its support is an upper bound.
pub fn cycle_shapes<CC>(res: &CC, layer: Layer, max_n: i32, max_s: i32) -> Vec<CycleShape>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    let max_u = (max_n + max_s) / 2;
    let all: Vec<InvertedSlice> = (0..=max_s).map(|a| slice(res, a, max_n, max_s)).collect();
    let stage = stage(res, layer, max_n, max_s).bound_by(&all, |a, u| {
        untwisted_max_length(max_n, max_s, a, u).map(|k| k as u32)
    });
    let mut out = Vec::new();
    let mut below: Option<&InvertedSlice> = None;
    for a in 0..=max_s {
        let inverted = &all[a as usize];
        let jobs: Vec<(i32, i32)> = (0..=max_u)
            .filter_map(|u| untwisted_max_length(max_n, max_s, a, u).map(|k| (u, k)))
            .filter(|&(u, _)| 2 * u - a <= max_n)
            .collect();
        let shapes: Vec<CycleShape> = jobs
            .into_maybe_par_iter()
            .flat_map(|(u, k_max)| {
                let (map, sources, _) = stage.build(a, u, k_max as u32, inverted);
                // δ^{a-1}, to know which cocycles are coboundaries.
                let lower = below.and_then(|b| {
                    let k = untwisted_max_length(max_n, max_s, a - 1, u)?;
                    let (m, src, tgt) = stage.build(a - 1, u, k as u32, b);
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
                    let keep = |s: usize| layer.on_length(len[s], k as u32);
                    let (_, cycles) = map.pairs_and_cycles(keep);
                    let killed: rustc_hash::FxHashSet<usize> = lower
                        .as_ref()
                        .map(|(m, src, tgt)| {
                            m.pairs(|s| layer.on_length(qmono_len(&src[s].2), k as u32))
                                .iter()
                                .map(|p| position[&tgt[p.target]])
                                .collect()
                        })
                        .unwrap_or_default();
                    let dim = (0..sources.len()).filter(|&s| keep(s)).count();
                    let dim_generators = sources
                        .iter()
                        .filter(|c| layer.on_length(qmono_len(&c.2), k as u32))
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
            let pages = vn_bockstein(&res, Layer::Adjoin(n), max_n, max_s);
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

    /// The quotient layers, from $Q/I_m$ with $v_m$ beyond the range down to $Q/I_0 = Q$: each
    /// $E_1$ is the previous $E_\infty$ tensored with $\mathbb{F}_2[v_m]$, and the last $E_\infty$
    /// is $\mathrm{Ext}_P(Q)$.
    #[test]
    fn quotients_converge() {
        let (max_n, max_s) = (30, 15);
        let res = crate::nassau::Resolution::new(std::sync::Arc::new(
            crate::algebraic_novikov::unit_module(),
        ));
        res.compute_through_stem(untwisted_backbone_range(max_n, max_s));
        let full = ctau_e2_untwisted_on(&res, max_n, max_s);
        let top = 5;
        let mut prev = page_dims(
            &vn_bockstein(&res, Layer::Quotient(top + 1), max_n, max_s),
            i32::MAX,
            max_n,
            max_s,
        );
        for m in (0..=top).rev() {
            let pages = vn_bockstein(&res, Layer::Quotient(m), max_n, max_s);
            let vm = (1 << m) - 1;
            let mut expect: BTreeMap<(i32, i32, i32), usize> = BTreeMap::new();
            for (&(stem, s, u), &d) in &prev {
                for j in 0.. {
                    let key = (stem + 2 * j * vm, s + j, u + j * vm);
                    if key.0 > max_n || key.1 > max_s {
                        break;
                    }
                    *expect.entry(key).or_insert(0) += d;
                }
            }
            assert_eq!(page_dims(&pages, 1, max_n, max_s), expect, "E_1 of Q/I_{m}");
            prev = page_dims(&pages, i32::MAX, max_n, max_s);
        }
        let full: BTreeMap<_, _> = full.0.into_iter().collect();
        assert_eq!(prev, full);
    }
}
