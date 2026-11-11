//! Products on $\mathrm{Ext}_P(Q/I_1)$ by $v_1$ and by the $h_i$, on cocycle representatives.

use std::{collections::BTreeMap, sync::Arc};

use algebra::{MilnorAlgebra, module::FDModule};
use rustc_hash::FxHashMap;
use sseq::coordinates::Bidegree;

use crate::{
    algebraic_novikov::{InvertedSlice, invert_slice, map_slice, qmono_len, untwisted_max_length},
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
    resolution::Resolution,
    resolution_homomorphism::ResolutionHomomorphism,
    v0_transfer::{Bits, add, flip, low, ones, zeros},
    vn_bockstein::{Cochain, Layer, slice, stage},
};

/// A minimal resolution of $\mathbb{F}_2$ over the Milnor basis that keeps its quasi-inverses, so
/// classes lift to chain maps.
pub type Backbone = Resolution<FiniteChainComplex<FDModule<MilnorAlgebra>>>;

/// A degree of $\mathrm{Hom}(C, Q/I_1)$: weight, filtration, length.
type Deg = (i32, i32, u32);

/// The cohomology of one degree of $\mathrm{Hom}(C, Q/I_1)$ and a way to read a cocycle's class.
struct Classes {
    /// For each lowest bit, a coboundary (`None`) or the representative of class `i`.
    table: FxHashMap<usize, (Bits, Option<usize>)>,
    /// The number of classes.
    count: usize,
}

impl Classes {
    /// The coordinates of the class of the cocycle `v`, or `None` if `v` is not a cocycle here.
    fn of(&self, mut v: Bits) -> Option<Bits> {
        let mut out = zeros(self.count);
        while let Some(l) = low(&v) {
            let (w, kind) = self.table.get(&l)?;
            add(&mut v, w);
            if let Some(i) = kind {
                flip(&mut out, *i);
            }
        }
        Some(out)
    }
}

/// How much of $\mathrm{Ext}_P(Q/I_1)$ is decomposable, by degree `(stem, s, u)`.
#[derive(Debug, Default)]
pub struct Decomposables {
    /// The number of classes in each degree.
    pub classes: BTreeMap<(i32, i32, i32), usize>,
    /// The dimension of the span of products by $v_1$.
    pub by_v1: BTreeMap<(i32, i32, i32), usize>,
    /// The dimension of the span of products by the $h_i$.
    pub by_h: BTreeMap<(i32, i32, i32), usize>,
    /// The dimension of the span of all the products.
    pub by_any: BTreeMap<(i32, i32, i32), usize>,
    /// Products whose image could not be read as a class.
    pub dropped: usize,
    /// Classes on which $d_1(v_1 x) = h_0 x + v_1 d_1(x)$ holds, and fails.
    pub leibniz_v1: (usize, usize),
    /// Pairs of a class and an $h_i$ on which $d_1(h_i x) = h_i d_1(x)$ holds, and fails.
    pub linear_h: (usize, usize),
}

/// The rank of a list of vectors.
fn rank(vectors: &[Bits]) -> usize {
    let mut table: FxHashMap<usize, Bits> = FxHashMap::default();
    for v in vectors {
        let mut v = v.clone();
        while let Some(l) = low(&v) {
            match table.get(&l) {
                Some(w) => add(&mut v, w),
                None => break,
            }
        }
        if let Some(l) = low(&v) {
            table.insert(l, v);
        }
    }
    table.len()
}

/// Products on $\mathrm{Ext}_P(Q/I_1)$ by $v_1$ and by $h_i$ for `i` in `h`, within stem `max_n`
/// and `s <= max_s`.
///
/// Classes are cocycles of $\mathrm{Hom}(C, Q/I_1)$, length by length. $v_1$ is primitive in
/// $Q/I_1$, so it multiplies cochains coefficientwise. $h_i$ lifts to a chain map
/// $f \colon C_{a+1} \to C_a$ of internal degree $2^i$, and its product is the pullback along $f$,
/// which has the shape of a coboundary. `res` must be resolved through `s = max_s + 2` and internal
/// degree `(max_n + max_s) / 2 + 2^max(h) + 2`.
pub fn decomposables(res: Arc<Backbone>, max_n: i32, max_s: i32, h: &[u32]) -> Decomposables {
    let max_u = (max_n + max_s) / 2;
    let mut stage = stage(&*res, Layer::Quotient(1), max_n, max_s);
    let diff: Vec<InvertedSlice> = (0..=max_s).map(|a| slice(&*res, a, max_n, max_s)).collect();

    // The pullback along a lift of each h_i, filtration by filtration.
    let h_maps: Vec<(i32, Vec<InvertedSlice>)> = h
        .iter()
        .map(|&i| {
            let t = 1 << i;
            let hom = ResolutionHomomorphism::from_class(
                format!("h{i}"),
                Arc::clone(&res),
                Arc::clone(&res),
                Bidegree::s_t(1, t),
                &[1],
            );
            hom.extend(Bidegree::s_t(max_s + 1, max_u + t));
            let slices = (0..=max_s)
                .map(|a| {
                    let map = hom.get_map(a + 1);
                    invert_slice(
                        &res.algebra(),
                        &map_slice(
                            &*res,
                            a,
                            max_u,
                            t,
                            |tz| res.number_of_gens_in_bidegree(Bidegree::s_t(a + 1, tz)),
                            |tz, l| {
                                // Below its shift the map is zero and not stored.
                                if tz < t {
                                    return Vec::new();
                                }
                                map.output(tz, l).iter_nonzero().map(|(j, _)| j).collect()
                            },
                        ),
                    )
                })
                .collect();
            (t, slices)
        })
        .collect();

    // The coaction table covers the operations of the differentials and of the maps alike.
    let merged: Vec<InvertedSlice> = (0..=max_s as usize)
        .map(|a| {
            let mut m = diff[a].clone();
            for (_, slices) in &h_maps {
                for (k, v) in &slices[a] {
                    m.entry(*k).or_default().extend(v.iter().cloned());
                }
            }
            m
        })
        .collect();
    let max_len = |a: i32, u: i32| untwisted_max_length(max_n, max_s, a, u).map(|k| k as u32);
    stage.tabulate(&merged, max_len, 0..=max_u);

    // Every cochain of each (u, a), by length and index within its length.
    let a_lo = |u: i32| (2 * u - max_n - 1).max(0);
    let mut basis: FxHashMap<(i32, i32), FxHashMap<Cochain, (u32, usize)>> = FxHashMap::default();
    let mut listed: FxHashMap<Deg, Vec<Cochain>> = FxHashMap::default();
    for u in 0..=max_u {
        for a in a_lo(u)..=max_s + 1 {
            let mut m = FxHashMap::default();
            for c in stage.cochains(a, u, max_s as u32) {
                let k = qmono_len(&c.2);
                let list = listed.entry((u, a, k)).or_default();
                m.insert(c, (k, list.len()));
                list.push(c);
            }
            basis.insert((u, a), m);
        }
    }
    let dim = |d: Deg| listed.get(&d).map_or(0, Vec::len);

    // The pullback along `inverted` from (u, a) to `target`, as columns over the target's cochains
    // of the same length, for the sources of each length.
    let pullback = |u: i32, a: i32, inverted: &InvertedSlice, target: (i32, i32)| {
        let mut cols: FxHashMap<Cochain, Vec<(u32, usize)>> = FxHashMap::default();
        let Some(k_max) = max_len(a, u) else {
            return cols;
        };
        let Some(into) = basis.get(&(target.1, target.0)) else {
            return cols;
        };
        let (map, sources, targets) = stage.build_map(a, u, k_max, inverted, target);
        for (s, c) in sources.iter().enumerate() {
            let col = map.columns[s]
                .iter()
                .filter_map(|&t| into.get(&targets[t as usize]).copied())
                .collect();
            cols.insert(*c, col);
        }
        cols
    };

    // The coboundaries, then the classes of each degree.
    let mut delta: FxHashMap<(i32, i32), FxHashMap<Cochain, Vec<(u32, usize)>>> =
        FxHashMap::default();
    for u in 0..=max_u {
        for a in a_lo(u)..=max_s {
            delta.insert((u, a), pullback(u, a, &diff[a as usize], (a + 1, u)));
        }
    }
    let column = |d: Deg, i: usize, map: &FxHashMap<Cochain, Vec<(u32, usize)>>| -> Bits {
        let mut v = zeros(dim((d.0, d.1 + 1, d.2)));
        for &(k, t) in map.get(&listed[&d][i]).into_iter().flatten() {
            if k == d.2 {
                flip(&mut v, t);
            }
        }
        v
    };
    let mut classes: FxHashMap<Deg, Classes> = FxHashMap::default();
    let mut reps: FxHashMap<Deg, Vec<Bits>> = FxHashMap::default();
    for (&(u, a, k), list) in &listed {
        let Some(out) = delta.get(&(u, a)) else {
            continue;
        };
        if max_len(a, u).is_none_or(|km| k > km) {
            continue;
        }
        let n = list.len();
        // The kernel of δ out of (u, a, k), from a reduction that tracks preimages.
        let mut image: FxHashMap<usize, (Bits, Bits)> = FxHashMap::default();
        let mut kernel = Vec::new();
        for i in 0..n {
            let mut y = column((u, a, k), i, out);
            let mut pre = zeros(n);
            flip(&mut pre, i);
            while let Some(l) = low(&y) {
                match image.get(&l) {
                    Some((w, p)) => {
                        add(&mut y, w);
                        add(&mut pre, p);
                    }
                    None => break,
                }
            }
            match low(&y) {
                Some(l) => {
                    image.insert(l, (y, pre));
                }
                None => kernel.push(pre),
            }
        }
        // The coboundaries into (u, a, k), then classes completing them.
        let mut table: FxHashMap<usize, (Bits, Option<usize>)> = FxHashMap::default();
        if let Some(below) = delta.get(&(u, a - 1)) {
            for i in 0..dim((u, a - 1, k)) {
                let mut y = column((u, a - 1, k), i, below);
                while let Some(l) = low(&y) {
                    match table.get(&l) {
                        Some((w, _)) => add(&mut y, &w.clone()),
                        None => break,
                    }
                }
                if let Some(l) = low(&y) {
                    table.insert(l, (y, None));
                }
            }
        }
        let mut found = Vec::new();
        for mut z in kernel {
            while let Some(l) = low(&z) {
                match table.get(&l) {
                    Some((w, _)) => add(&mut z, &w.clone()),
                    None => break,
                }
            }
            if let Some(l) = low(&z) {
                table.insert(l, (z.clone(), Some(found.len())));
                found.push(z);
            }
        }
        classes.insert(
            (u, a, k),
            Classes {
                table,
                count: found.len(),
            },
        );
        reps.insert((u, a, k), found);
    }

    // Degrees whose classes are counted: stem within range and s <= max_s.
    let counted = |(u, a, k): Deg| {
        let stem = 2 * u - a;
        (0..=max_n).contains(&stem)
            && a + k as i32 <= max_s
            && max_len(a, u).is_some_and(|km| k <= km)
    };
    // A map on classes: for each class of each source degree, the coordinates of its image, or
    // `None` where the image could not be read.
    type ClassMap = FxHashMap<Deg, Vec<Option<Bits>>>;
    let mut dropped = 0;
    let mut read = |target: Deg, v: Bits| -> Option<Bits> {
        let x = classes.get(&target)?.of(v);
        if x.is_none() {
            dropped += 1;
        }
        x
    };
    // Products by v_1: (u, a, k) to (u + 1, a, k + 1).
    let mut v1_map: ClassMap = FxHashMap::default();
    for (&(u, a, k), rs) in &reps {
        let target = (u + 1, a, k + 1);
        if !counted(target) || !counted((u, a, k)) {
            continue;
        }
        let Some(into) = basis.get(&(u + 1, a)) else {
            continue;
        };
        let images = rs
            .iter()
            .map(|r| {
                let mut v = zeros(dim(target));
                for i in ones(r) {
                    let mut c = listed[&(u, a, k)][i];
                    c.2[1] += 1;
                    // Outside the truncation the product is not read.
                    let &(_, t) = into.get(&c)?;
                    flip(&mut v, t);
                }
                read(target, v)
            })
            .collect();
        v1_map.insert((u, a, k), images);
    }
    // Products by h_i: (u, a, k) to (u + 2^i, a + 1, k).
    let mut h_map: FxHashMap<i32, ClassMap> = FxHashMap::default();
    for (t, slices) in &h_maps {
        let map = h_map.entry(*t).or_default();
        for u in 0..=max_u - t {
            for a in a_lo(u)..=max_s {
                let cols = pullback(u, a, &slices[a as usize], (a + 1, u + t));
                for k in 0..=max_s as u32 {
                    let target = (u + t, a + 1, k);
                    let Some(rs) = reps.get(&(u, a, k)) else {
                        continue;
                    };
                    if !counted(target) || !counted((u, a, k)) {
                        continue;
                    }
                    let images = rs
                        .iter()
                        .map(|r| {
                            let mut v = zeros(dim(target));
                            for i in ones(r) {
                                for &(kk, j) in
                                    cols.get(&listed[&(u, a, k)][i]).into_iter().flatten()
                                {
                                    if kk == k {
                                        flip(&mut v, j);
                                    }
                                }
                            }
                            read(target, v)
                        })
                        .collect();
                    map.insert((u, a, k), images);
                }
            }
        }
    }
    // d_1 of the v_0-Bockstein: the part of the coboundary of Q that lowers the length by one,
    // from (u, a, k) to (u, a + 1, k - 1).
    let q_stage = crate::vn_bockstein::stage(
        &*res,
        Layer::Adjoin(crate::algebraic_novikov::QDOWN_VARS - 1),
        max_n,
        max_s,
    )
    .bound_by(&diff, max_len);
    let mut d1_map: ClassMap = FxHashMap::default();
    for u in 0..=max_u {
        for a in a_lo(u)..=max_s {
            let Some(k_max) = max_len(a, u) else {
                continue;
            };
            let (map, sources, targets) = q_stage.build(a, u, k_max, &diff[a as usize]);
            let at: FxHashMap<Cochain, usize> =
                sources.iter().enumerate().map(|(i, c)| (*c, i)).collect();
            let Some(into) = basis.get(&(u, a + 1)) else {
                continue;
            };
            for k in 1..=k_max {
                let target = (u, a + 1, k - 1);
                let Some(rs) = reps.get(&(u, a, k)) else {
                    continue;
                };
                if !counted((u, a, k)) || !counted(target) {
                    continue;
                }
                let images = rs
                    .iter()
                    .map(|r| {
                        let mut v = zeros(dim(target));
                        for i in ones(r) {
                            let s = at[&listed[&(u, a, k)][i]];
                            for &t in &map.columns[s] {
                                if let Some(&(kk, j)) = into.get(&targets[t as usize])
                                    && kk == k - 1
                                {
                                    flip(&mut v, j);
                                }
                            }
                        }
                        read(target, v)
                    })
                    .collect();
                d1_map.insert((u, a, k), images);
            }
        }
    }

    // The image under `map` of the class with coordinates `x` in degree `d`.
    let apply = |map: &ClassMap, d: Deg, x: &Bits| -> Option<Bits> {
        let images = map.get(&d)?;
        let mut out: Option<Bits> = None;
        for i in ones(x) {
            let y = images[i].as_ref()?;
            match &mut out {
                Some(o) => add(o, y),
                None => out = Some(y.clone()),
            }
        }
        out
    };
    let mut out = Decomposables::default();
    // d_1(v_1 x) = h_0 x + v_1 d_1(x), and d_1(h_i x) = h_i d_1(x), on every class where every
    // term is read; an empty sum is zero.
    let same = |x: Option<Bits>, y: Option<Bits>| match (x, y) {
        (Some(x), Some(y)) => x == y,
        (Some(x), None) | (None, Some(x)) => low(&x).is_none(),
        (None, None) => true,
    };
    for (&d, cls) in &classes {
        let (u, a, k) = d;
        if !counted(d) || k == 0 {
            continue;
        }
        for i in 0..cls.count {
            let mut e = zeros(cls.count);
            flip(&mut e, i);
            // Leibniz for v_1.
            let v1x = apply(&v1_map, d, &e);
            let dx = apply(&d1_map, d, &e);
            if let (Some(v1x), true) = (&v1x, h_map.contains_key(&1)) {
                let lhs = apply(&d1_map, (u + 1, a, k + 1), v1x);
                let h0x = apply(&h_map[&1], d, &e);
                let v1dx = dx
                    .as_ref()
                    .and_then(|y| apply(&v1_map, (u, a + 1, k - 1), y));
                let rhs = match (h0x, v1dx) {
                    (Some(mut p), Some(q)) => {
                        add(&mut p, &q);
                        Some(p)
                    }
                    (p, q) => p.or(q),
                };
                if counted((u + 1, a + 1, k)) && counted((u + 1, a, k + 1)) {
                    *if same(lhs, rhs) {
                        &mut out.leibniz_v1.0
                    } else {
                        &mut out.leibniz_v1.1
                    } += 1;
                }
            }
            // Linearity over h_i.
            for (&t, map) in &h_map {
                let Some(hx) = apply(map, d, &e) else {
                    continue;
                };
                if !counted((u + t, a + 2, k - 1)) {
                    continue;
                }
                let lhs = apply(&d1_map, (u + t, a + 1, k), &hx);
                let rhs = dx.as_ref().and_then(|y| apply(map, (u, a + 1, k - 1), y));
                *if same(lhs, rhs) {
                    &mut out.linear_h.0
                } else {
                    &mut out.linear_h.1
                } += 1;
            }
        }
    }

    // Decomposables: the span of the products landing in each degree.
    let mut landing: FxHashMap<Deg, (Vec<Bits>, Vec<Bits>)> = FxHashMap::default();
    for (&(u, a, k), images) in &v1_map {
        landing
            .entry((u + 1, a, k + 1))
            .or_default()
            .0
            .extend(images.iter().flatten().cloned());
    }
    for (&t, map) in &h_map {
        for (&(u, a, k), images) in map {
            landing
                .entry((u + t, a + 1, k))
                .or_default()
                .1
                .extend(images.iter().flatten().cloned());
        }
    }
    out.dropped = dropped;
    for (&d, cls) in &classes {
        if !counted(d) || cls.count == 0 {
            continue;
        }
        let (u, a, k) = d;
        let key = (2 * u - a, a + k as i32, u);
        out.classes.insert(key, cls.count);
        let (v1, hs) = landing.remove(&d).unwrap_or_default();
        let all: Vec<Bits> = v1.iter().chain(&hs).cloned().collect();
        out.by_v1.insert(key, rank(&v1));
        out.by_h.insert(key, rank(&hs));
        out.by_any.insert(key, rank(&all));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebraic_novikov::unit_module,
        vn_bockstein::{page_dims, vn_bockstein},
    };

    /// The classes are those of the quotient layer $Q/I_1$, and $d_1$ of the $v_0$-Bockstein is a
    /// derivation with $d_1(v_1) = h_0$, linear over the $h_i$.
    #[test]
    fn classes_and_leibniz() {
        let (max_n, max_s) = (24, 12);
        let max_u = (max_n + max_s) / 2;
        let res: Arc<Backbone> = Arc::new(Resolution::new(Arc::new(FiniteChainComplex::ccdz(
            Arc::new(unit_module()),
        ))));
        res.compute_through_bidegree(Bidegree::s_t(max_s + 2, max_u + 16 + 2));
        let d = decomposables(Arc::clone(&res), max_n, max_s, &[0, 1, 2, 3, 4]);
        let layer = page_dims(
            &vn_bockstein(&*res, Layer::Quotient(1), max_n, max_s),
            i32::MAX,
            max_n,
            max_s,
        );
        assert_eq!(d.classes, layer);
        assert_eq!(d.dropped, 0);
        assert!(d.leibniz_v1.0 > 0 && d.leibniz_v1.1 == 0);
        assert!(d.linear_h.0 > 0 && d.linear_h.1 == 0);
    }
}
