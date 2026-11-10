//! $\mathrm{Ext}_P(Q)$ from $\mathrm{Ext}_P(Q/I_1)$ by transferring the $v_0$-Bockstein perturbation.

use std::collections::BTreeMap;

use algebra::MilnorAlgebra;
use maybe_rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    algebraic_novikov::{
        ActMemo, InvertedSlice, qmono_len, untwisted_filtrations, untwisted_max_weight,
    },
    chain_complex::FreeChainComplex,
    vn_bockstein::{Cochain, Layer, Stage, slice, stage},
};

/// A vector over $\mathbb{F}_2$ as packed bits.
type Bits = Vec<u64>;

/// The zero vector of dimension `n`.
fn zeros(n: usize) -> Bits {
    vec![0; n.div_ceil(64)]
}

/// Flips bit `i`.
fn flip(v: &mut Bits, i: usize) {
    v[i / 64] ^= 1 << (i % 64);
}

/// Adds `b` to `a`.
fn add(a: &mut Bits, b: &Bits) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= y;
    }
}

/// The lowest set bit.
fn low(v: &Bits) -> Option<usize> {
    v.iter()
        .enumerate()
        .find(|(_, w)| **w != 0)
        .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
}

/// The set bits.
fn ones(v: &Bits) -> impl Iterator<Item = usize> + '_ {
    v.iter().enumerate().flat_map(|(i, &w)| {
        let mut w = w;
        std::iter::from_fn(move || {
            (w != 0).then(|| {
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                i * 64 + b
            })
        })
    })
}

/// A reduced basis of the image of a map, each vector with its preimage, by lowest set bit.
#[derive(Default)]
struct Image {
    /// For each lowest bit, the reduced image vector and a preimage of it.
    by_low: FxHashMap<usize, (Bits, Bits)>,
}

impl Image {
    /// Reduces `y` against the basis, adding the preimages of what it removes to `pre`.
    fn reduce(&self, y: &mut Bits, pre: &mut Bits) {
        while let Some(l) = low(y) {
            let Some((v, p)) = self.by_low.get(&l) else {
                return;
            };
            add(y, v);
            add(pre, p);
        }
    }
}

/// The cohomology of one slice $\mathrm{Hom}(C, (Q/I_1)_j)^a$ at a fixed weight, with the data
/// of a deformation retraction onto it.
struct Slice {
    /// The dimension of the cochains.
    dim: usize,
    /// The image of $\delta_0$ out of this degree, with preimages here.
    out: Image,
    /// A basis of the cocycles: the image of $\delta_0$ into this degree, with preimages in the
    /// degree below, and representatives of the cohomology, with their index.
    cycles: FxHashMap<usize, (Bits, Cycle)>,
    /// The representatives of the cohomology.
    classes: Vec<Bits>,
}

/// What a vector in [`Slice::cycles`] is.
enum Cycle {
    /// A coboundary, with a preimage of dimension `dim` below.
    Boundary(Bits),
    /// The representative of a class.
    Class(usize),
}

impl Slice {
    /// The projection $p$ to the cohomology and the homotopy $h$ to the degree below, of a
    /// cochain `c`, given the dimension of the degree below and $\delta_0$ of `c`.
    fn p_h(&self, c: &Bits, mut dc: Bits, below: usize) -> (Bits, Bits) {
        // c - σ(δc) is a cocycle; split it into a coboundary and a class.
        let mut z = c.clone();
        let mut x = zeros(self.dim);
        self.out.reduce(&mut dc, &mut x);
        add(&mut z, &x);
        let mut p = zeros(self.classes.len());
        let mut h = zeros(below);
        while let Some(l) = low(&z) {
            let (v, kind) = &self.cycles[&l];
            add(&mut z, v);
            match kind {
                Cycle::Boundary(pre) => add(&mut h, pre),
                Cycle::Class(i) => flip(&mut p, *i),
            }
        }
        (p, h)
    }
}

/// The dimensions of $\mathrm{Ext}_P(Q)$ in the trigrading `(stem, s, u)` within stem `max_n`
/// and `s <= max_s`, keyed as [`crate::algebraic_novikov::TrigradedDims`].
///
/// $\mathrm{Hom}(C, Q_k)$ is $\bigoplus_{j \leq k} \mathrm{Hom}(C, (Q/I_1)_j)\, v_0^{k-j}$, and
/// $\delta = \sum_r v_0^r \delta_r$ with $\delta_r$ lowering the length by $r$. Each slice of
/// $\delta_0$ is retracted onto its cohomology $\mathrm{Ext}(Q/I_1)$, and the perturbation lemma
/// carries $\delta' = \sum_{r \geq 1} v_0^r \delta_r$ to $D = p \delta' \sum_n (h \delta')^n i$ on
/// $\mathrm{Ext}(Q/I_1)[v_0]$, whose cohomology in each $v_0$-degree is $\mathrm{Ext}(Q_k)$. Only
/// slices are ever reduced, never a degree of $\mathrm{Hom}(C, Q_k)$.
pub fn ext_q_by_v0_transfer<CC>(
    res: &CC,
    max_n: i32,
    max_s: i32,
) -> BTreeMap<(i32, i32, i32), usize>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    let max_u = (max_n + max_s) / 2;
    let stage = stage(
        res,
        Layer::Adjoin(crate::algebraic_novikov::QDOWN_VARS - 1),
        max_n,
        max_s,
    );
    let slices: Vec<InvertedSlice> = (0..=max_s)
        .into_maybe_par_iter()
        .map(|a| slice(res, a, max_n, max_s))
        .collect();
    let per_u: Vec<BTreeMap<(i32, i32, i32), usize>> = (0..=max_u)
        .into_maybe_par_iter()
        .map(|u| weight(&stage, &slices, u, max_n, max_s))
        .collect();
    per_u.into_iter().flatten().collect()
}

/// [`ext_q_by_v0_transfer`] at weight `u`.
fn weight<CC>(
    stage: &Stage<'_, CC>,
    inverted: &[InvertedSlice],
    u: i32,
    max_n: i32,
    max_s: i32,
) -> BTreeMap<(i32, i32, i32), usize>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    // Classes are wanted for stem 2u - a <= max_n; the degree below feeds the coboundaries.
    let a_lo = (2 * u - max_n - 1).max(0);
    if a_lo > max_s {
        return BTreeMap::new();
    }
    let memo = ActMemo::default();
    // Lengths j <= max_s - a are all a class of s <= max_s can use.
    let k_of = |a: i32| (max_s - a).max(0) as u32;

    // Cochains of each degree by length: (length, index in its slice) of each, and the slices.
    let mut slot: BTreeMap<i32, FxHashMap<Cochain, (u32, usize)>> = BTreeMap::new();
    let mut dims: BTreeMap<(i32, u32), usize> = BTreeMap::new();
    for a in a_lo..=max_s + 1 {
        let mut m = FxHashMap::default();
        // A degree also holds the length one past its own, as the target of the degree below.
        let k = if a == 0 { k_of(0) } else { k_of(a - 1) };
        for c in stage.cochains(a, u, k) {
            let j = qmono_len(&c.2);
            let d = dims.entry((a, j)).or_insert(0);
            m.insert(c, (j, *d));
            *d += 1;
        }
        slot.insert(a, m);
    }
    let dim = |a: i32, j: u32| dims.get(&(a, j)).copied().unwrap_or(0);

    // δ^a as columns by source slot: (target length, index) of each term, for the targets kept.
    let mut delta: BTreeMap<i32, FxHashMap<(u32, usize), Vec<(u32, usize)>>> = BTreeMap::new();
    for a in a_lo..=max_s {
        let (map, sources, targets) = stage.build(a, u, k_of(a), &inverted[a as usize], &memo);
        let mut cols = FxHashMap::default();
        for (s, c) in sources.iter().enumerate() {
            let Some(&src) = slot[&a].get(c) else {
                continue;
            };
            let col: Vec<(u32, usize)> = map.columns[s]
                .iter()
                .filter_map(|&t| slot[&(a + 1)].get(&targets[t as usize]).copied())
                .collect();
            cols.insert(src, col);
        }
        delta.insert(a, cols);
    }

    // Reduce each slice of δ_0 and retract it onto its cohomology.
    let mut slices: BTreeMap<(i32, u32), Slice> = BTreeMap::new();
    for a in a_lo..=max_s {
        for j in 0..=k_of(a) {
            let n = dim(a, j);
            if n == 0 {
                continue;
            }
            // δ_0 out of (a, j): reduce its columns, keeping preimages; zero columns are cocycles.
            let mut out = Image::default();
            let mut kernel = Vec::new();
            for i in 0..n {
                let mut y = zeros(dim(a + 1, j));
                for &(jt, t) in &delta[&a][&(j, i)] {
                    if jt == j {
                        flip(&mut y, t);
                    }
                }
                let mut pre = zeros(n);
                flip(&mut pre, i);
                while let Some(l) = low(&y) {
                    match out.by_low.get(&l) {
                        Some((v, p)) => {
                            add(&mut y, v);
                            add(&mut pre, p);
                        }
                        None => break,
                    }
                }
                match low(&y) {
                    Some(l) => {
                        out.by_low.insert(l, (y, pre));
                    }
                    None => kernel.push(pre),
                }
            }
            // The coboundaries into (a, j), from the slice below, then classes completing them.
            let mut cycles: FxHashMap<usize, (Bits, Cycle)> = FxHashMap::default();
            if let Some(below) = slices.get(&(a - 1, j)) {
                for (l, (v, p)) in &below.out.by_low {
                    cycles.insert(*l, (v.clone(), Cycle::Boundary(p.clone())));
                }
            }
            let mut classes = Vec::new();
            for mut z in kernel {
                while let Some(l) = low(&z) {
                    match cycles.get(&l) {
                        Some((v, _)) => add(&mut z, &v.clone()),
                        None => break,
                    }
                }
                if let Some(l) = low(&z) {
                    cycles.insert(l, (z.clone(), Cycle::Class(classes.len())));
                    classes.push(z);
                }
            }
            slices.insert(
                (a, j),
                Slice {
                    dim: n,
                    out,
                    cycles,
                    classes,
                },
            );
        }
    }

    // δ applied to a vector of slice (a, j): the δ_0 part, and the δ' part by target length.
    let apply = |a: i32, j: u32, v: &Bits| -> (Bits, BTreeMap<u32, Bits>) {
        let mut d0 = zeros(dim(a + 1, j));
        let mut rest: BTreeMap<u32, Bits> = BTreeMap::new();
        for i in ones(v) {
            for &(jt, t) in &delta[&a][&(j, i)] {
                if jt == j {
                    flip(&mut d0, t);
                } else {
                    flip(rest.entry(jt).or_insert_with(|| zeros(dim(a + 1, jt))), t);
                }
            }
        }
        (d0, rest)
    };

    // D out of each class of (a, j): its components in the classes of (a + 1, j') for j' < j.
    // D^a as columns over the classes of degree a + 1, numbered by length then index.
    let class_offset = |a: i32| -> (BTreeMap<u32, usize>, usize) {
        let mut off = BTreeMap::new();
        let mut total = 0;
        for j in 0..=k_of(a) {
            off.insert(j, total);
            total += slices.get(&(a, j)).map_or(0, |s| s.classes.len());
        }
        (off, total)
    };
    let mut out = BTreeMap::new();
    let mut ranks: BTreeMap<(i32, u32), usize> = BTreeMap::new();
    for a in a_lo..=max_s {
        let (target_off, target_total) = class_offset(a + 1);
        let mut table: FxHashMap<usize, Bits> = FxHashMap::default();
        let mut rank = 0;
        for j in 0..=k_of(a) {
            if let Some(slice) = slices.get(&(a, j)) {
                for x in &slice.classes {
                    // y = δ'(i x), then repeatedly collect p(y) and continue with δ'(h y).
                    let mut col: Bits = zeros(target_total);
                    let (_, mut y) = apply(a, j, x);
                    while !y.is_empty() {
                        let mut next: BTreeMap<u32, Bits> = BTreeMap::new();
                        for (jt, v) in y {
                            let Some(t) = slices.get(&(a + 1, jt)) else {
                                continue;
                            };
                            let (dv, _) = apply(a + 1, jt, &v);
                            let (p, h) = t.p_h(&v, dv, dim(a, jt));
                            for c in ones(&p) {
                                flip(&mut col, target_off[&jt] + c);
                            }
                            if low(&h).is_some() {
                                let (_, rest) = apply(a, jt, &h);
                                for (jn, w) in rest {
                                    add(
                                        next.entry(jn).or_insert_with(|| zeros(dim(a + 1, jn))),
                                        &w,
                                    );
                                }
                            }
                        }
                        y = next;
                    }
                    // Rank of D^a restricted to sources of length at most j, incrementally.
                    while let Some(l) = low(&col) {
                        match table.get(&l) {
                            Some(v) => add(&mut col, v),
                            None => break,
                        }
                    }
                    if let Some(l) = low(&col) {
                        table.insert(l, col);
                        rank += 1;
                    }
                }
            }
            ranks.insert((a, j), rank);
        }
    }
    let rank = |a: i32, k: u32| -> usize {
        ranks
            .range((a, 0)..=(a, k))
            .next_back()
            .map_or(0, |(_, &r)| r)
    };
    for k in 0..=max_s {
        if u > untwisted_max_weight(max_n, max_s, k) {
            continue;
        }
        for a in untwisted_filtrations(max_n, max_s, k, u) {
            let stem = 2 * u - a;
            if stem > max_n || stem < 0 || a < a_lo {
                continue;
            }
            let h: usize = (0..=k as u32)
                .map(|j| slices.get(&(a, j)).map_or(0, |s| s.classes.len()))
                .sum();
            let d = h - rank(a, k as u32) - if a > a_lo { rank(a - 1, k as u32) } else { 0 };
            if d > 0 {
                out.insert((stem, a + k, u), d);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebraic_novikov::{ctau_e2_untwisted_on, unit_module, untwisted_backbone_range};

    /// The transfer reproduces $\mathrm{Ext}_P(Q)$.
    #[test]
    fn matches_direct() {
        let (max_n, max_s) = (40, 20);
        let res = crate::nassau::Resolution::new(std::sync::Arc::new(unit_module()));
        res.compute_through_stem(untwisted_backbone_range(max_n, max_s));
        let direct: BTreeMap<_, _> = ctau_e2_untwisted_on(&res, max_n, max_s)
            .0
            .into_iter()
            .collect();
        assert_eq!(ext_q_by_v0_transfer(&res, max_n, max_s), direct);
    }
}
