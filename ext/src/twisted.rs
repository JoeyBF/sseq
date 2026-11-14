//! A free resolution over the Milnor algebra built from its exterior part and the doubled backbone.

use std::sync::Arc;

use algebra::{
    Algebra, MilnorAlgebra,
    milnor_algebra::{MilnorBasisElement, PPart},
    module::{FreeModule, Module},
};
use fp::{
    matrix::{AugmentedMatrix, QuasiInverse},
    vector::{FpSlice, FpVector},
};
use rustc_hash::FxHashMap;
use sseq::coordinates::Bidegree;

use crate::chain_complex::FreeChainComplex;

/// A generator $x^S \otimes w$ of the total complex: the row $p$ (filtration of $w$ in the doubled
/// backbone), the Koszul length $q$, the Koszul generator and the backbone generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TwistedGen {
    pub p: i32,
    pub q: i32,
    pub koszul: usize,
    pub w: usize,
}

/// Wall's twisted tensor product for the extension $\Lambda \to A \to B$ at $p = 2$: $\Lambda$ the
/// exterior algebra on the Milnor primitives $P^0_t$, $B = A /\!/ \Lambda$ the even part, which
/// is $A$ with degrees doubled.
///
/// Each row $A \otimes_\Lambda K \otimes W_p$, with $K$ the Koszul resolution of $\mathbb{F}_2$
/// over $\Lambda$, resolves $B \otimes W_p$ freely over $A$, and the doubled backbone
/// $B \otimes W_\bullet$ resolves $\mathbb{F}_2$ over $B$. The total complex
/// $X_n = \bigoplus_{p + q = n} A \otimes K_q \otimes W_p$ with $d = d_0 + d_1 + d_2 + \cdots$, $d_k$
/// lowering the row by $k$, is then a free resolution of $\mathbb{F}_2$ over $A$. Here $d_0$ is
/// the Koszul differential, $d_1$ lifts the doubled backbone's, and each $d_k$ is solved for
/// through $d_0$ so that $d^2 = 0$; since $d_0$ acts on each copy of $K$ separately, every solve
/// is a preimage under the one Koszul complex.
///
/// This materialises every $d_k$, for studying them; nothing here is meant to scale.
pub struct TwistedResolution {
    algebra: Arc<MilnorAlgebra>,
    max_n: i32,
    max_t: i32,
    /// Koszul generators: exponents $S$ ($S[t - 1]$ the exponent of $x_t$, dual to $P^0_t$),
    /// degree and length.
    koszul: Vec<(Vec<u32>, i32, i32)>,
    koszul_index: FxHashMap<Vec<u32>, usize>,
    /// The Koszul complex $A \otimes_\Lambda K$, one free module per length, and where each Koszul
    /// generator sits in its module.
    kmods: Vec<Arc<FreeModule<MilnorAlgebra>>>,
    kpos: Vec<(i32, usize)>,
    /// Preimages under $d_0$ from length `q + 1` to `q`, by `(q, degree)`.
    kqi: FxHashMap<(i32, i32), QuasiInverse>,
    /// Backbone generators doubled: `(p, doubled degree, index in the backbone)`.
    w: Vec<(i32, i32, usize)>,
    /// The doubled differential of each backbone generator: `(op degree, op index, target w)`.
    w_diff: Vec<Vec<(i32, usize, usize)>>,
    /// The total complex: one free module per homological degree, its generators by degree and
    /// index, where each generator sits, and the differential of each generator.
    xmods: Vec<Arc<FreeModule<MilnorAlgebra>>>,
    xgens: Vec<Vec<Vec<TwistedGen>>>,
    xpos: Vec<FxHashMap<(usize, usize), (i32, usize)>>,
    d: Vec<Vec<Vec<FpVector>>>,
    /// The number of terms of each generator's differential by $k$, the rows it drops.
    pub terms_by_k: Vec<usize>,
}

impl TwistedResolution {
    /// The total complex through homological degree `max_n` and internal degree `max_t`, from
    /// a minimal resolution `backbone` of $\mathbb{F}_2$ computed through stem `max_t / 2` and
    /// filtration `max_n`.
    pub fn new<CC>(backbone: &CC, max_n: i32, max_t: i32) -> Self
    where
        CC: FreeChainComplex<Algebra = MilnorAlgebra>,
    {
        let algebra = backbone.algebra();
        algebra.compute_basis(max_t);
        // Koszul generators.
        let gens_t: Vec<i32> = (1..)
            .map(|t| (1 << t) - 1)
            .take_while(|&d| d <= max_t)
            .collect();
        let mut koszul = Vec::new();
        fn fill(
            i: usize,
            degs: &[i32],
            left: i32,
            q_left: i32,
            s: &mut Vec<u32>,
            out: &mut Vec<(Vec<u32>, i32, i32)>,
            max_t: i32,
            max_n: i32,
        ) {
            if i == degs.len() {
                let deg = max_t - left;
                out.push((s.clone(), deg, max_n - q_left));
                return;
            }
            let mut e = 0;
            while e as i32 * degs[i] <= left && e as i32 <= q_left {
                s.push(e);
                fill(
                    i + 1,
                    degs,
                    left - e as i32 * degs[i],
                    q_left - e as i32,
                    s,
                    out,
                    max_t,
                    max_n,
                );
                s.pop();
                e += 1;
            }
        }
        fill(
            0,
            &gens_t,
            max_t,
            max_n,
            &mut Vec::new(),
            &mut koszul,
            max_t,
            max_n,
        );
        koszul.sort_by_key(|(s, d, q)| (*q, *d, s.clone()));
        let koszul_index = koszul
            .iter()
            .enumerate()
            .map(|(i, (s, _, _))| (s.clone(), i))
            .collect();

        // The Koszul complex as free modules.
        let kmods: Vec<Arc<FreeModule<MilnorAlgebra>>> = (0..=max_n)
            .map(|q| Arc::new(FreeModule::new(Arc::clone(&algebra), format!("K{q}"), 0)))
            .collect();
        let mut kpos = vec![(0, 0); koszul.len()];
        for q in 0..=max_n {
            for t in 0..=max_t {
                let here: Vec<usize> = (0..koszul.len())
                    .filter(|&i| koszul[i].2 == q && koszul[i].1 == t)
                    .collect();
                for (j, &i) in here.iter().enumerate() {
                    kpos[i] = (t, j);
                }
                kmods[q as usize].add_generators(t, here.len(), None);
            }
            kmods[q as usize].compute_basis(max_t);
        }

        // Backbone generators doubled.
        let mut w = Vec::new();
        let mut w_index: FxHashMap<(i32, i32, usize), usize> = FxHashMap::default();
        for p in 0..=max_n {
            for t in 0..=max_t / 2 {
                if !backbone.has_computed_bidegree(Bidegree::s_t(p, t)) {
                    continue;
                }
                for l in 0..backbone.number_of_gens_in_bidegree(Bidegree::s_t(p, t)) {
                    w_index.insert((p, 2 * t, l), w.len());
                    w.push((p, 2 * t, l));
                }
            }
        }
        let w_diff = w
            .iter()
            .map(|&(p, t2, l)| {
                if p == 0 {
                    return Vec::new();
                }
                let t = t2 / 2;
                let d = backbone.differential(p);
                let target = backbone.module(p - 1);
                d.output(t, l)
                    .iter_nonzero()
                    .filter_map(|(idx, _)| {
                        let og = target.index_to_op_gen(t, idx);
                        let r = algebra
                            .basis_element_from_index(og.operation_degree, og.operation_index)
                            .p_part;
                        let doubled: Vec<u32> = r.iter().map(|e| 2 * e).collect();
                        let elt = MilnorBasisElement {
                            q_part: 0,
                            p_part: PPart::from_slice(&doubled),
                            degree: 2 * og.operation_degree,
                        };
                        let target_w =
                            *w_index.get(&(p - 1, 2 * og.generator_degree, og.generator_index))?;
                        Some((elt.degree, algebra.basis_element_to_index(&elt), target_w))
                    })
                    .collect()
            })
            .collect();

        let mut me = Self {
            algebra,
            max_n,
            max_t,
            koszul,
            koszul_index,
            kmods,
            kpos,
            kqi: FxHashMap::default(),
            w,
            w_diff,
            xmods: Vec::new(),
            xgens: Vec::new(),
            xpos: Vec::new(),
            d: Vec::new(),
            terms_by_k: vec![0; max_n as usize + 2],
        };
        me.koszul_quasi_inverses();
        me.build_total();
        me
    }

    /// $d_0$ of Koszul generator `i`, as an element of the Koszul module one length down.
    fn koszul_d(&self, i: usize) -> FpVector {
        let (s, deg, q) = &self.koszul[i];
        let target = &self.kmods[*q as usize - 1];
        let mut out = FpVector::new(self.algebra.prime(), target.dimension(*deg));
        for (t, &e) in s.iter().enumerate() {
            if e == 0 {
                continue;
            }
            let mut smaller = s.clone();
            smaller[t] -= 1;
            let j = self.koszul_index[&smaller];
            let (gd, gi) = self.kpos[j];
            let op = self.primitive(t + 1);
            let idx = target.operation_generator_to_index(op.0, op.1, gd, gi);
            out.add_basis_element(idx, 1);
        }
        out
    }

    /// $P^0_t$ as `(degree, index)`.
    fn primitive(&self, t: usize) -> (i32, usize) {
        let mut r = vec![0u32; t];
        r[t - 1] = 1;
        let elt = MilnorBasisElement {
            q_part: 0,
            p_part: PPart::from_slice(&r),
            degree: (1 << t) - 1,
        };
        (elt.degree, self.algebra.basis_element_to_index(&elt))
    }

    /// Applies an $A$-linear map given on generators to an element of a free module.
    fn apply(
        source: &FreeModule<MilnorAlgebra>,
        target: &FreeModule<MilnorAlgebra>,
        degree: i32,
        input: FpSlice,
        on_gen: impl Fn(i32, usize) -> Option<FpVector>,
    ) -> FpVector {
        let mut out = FpVector::new(source.prime(), target.dimension(degree));
        for (i, c) in input.iter_nonzero() {
            let og = source.index_to_op_gen(degree, i);
            if let Some(v) = on_gen(og.generator_degree, og.generator_index) {
                target.act(
                    out.as_slice_mut(),
                    c,
                    og.operation_degree,
                    og.operation_index,
                    og.generator_degree,
                    v.as_slice(),
                );
            }
        }
        out
    }

    /// Preimages under the Koszul differential from length `q + 1` to `q`, in every degree.
    fn koszul_quasi_inverses(&mut self) {
        let p = self.algebra.prime();
        // d_0 on generators of each length, by (degree, index).
        let mut on_gen: Vec<FxHashMap<(i32, usize), FpVector>> =
            vec![FxHashMap::default(); self.max_n as usize + 1];
        for i in 0..self.koszul.len() {
            let q = self.koszul[i].2;
            if q > 0 {
                on_gen[q as usize].insert(self.kpos[i], self.koszul_d(i));
            }
        }
        for q in 0..self.max_n {
            let (src, tgt) = (&self.kmods[q as usize + 1], &self.kmods[q as usize]);
            for t in 0..=self.max_t {
                let (rows, cols) = (src.dimension(t), tgt.dimension(t));
                if rows == 0 || cols == 0 {
                    continue;
                }
                let mut m = AugmentedMatrix::<2>::new(p, rows, [cols, rows]);
                for i in 0..rows {
                    let mut e = FpVector::new(p, rows);
                    e.set_entry(i, 1);
                    let image = Self::apply(src, tgt, t, e.as_slice(), |gd, gi| {
                        on_gen[q as usize + 1].get(&(gd, gi)).cloned()
                    });
                    m.segment(0, 0).row_mut(i).add(image.as_slice(), 1);
                    m.segment(1, 1).row_mut(i).set_entry(i, 1);
                }
                m.row_reduce();
                self.kqi.insert((q, t), m.compute_quasi_inverse());
            }
        }
    }

    /// The total complex and its differential.
    fn build_total(&mut self) {
        for n in 0..=self.max_n {
            let module = Arc::new(FreeModule::new(
                Arc::clone(&self.algebra),
                format!("X{n}"),
                0,
            ));
            let mut by_deg: Vec<Vec<TwistedGen>> = vec![Vec::new(); self.max_t as usize + 1];
            for (wi, &(p, wt, _)) in self.w.iter().enumerate() {
                for (ki, (_, kd, q)) in self.koszul.iter().enumerate() {
                    if p + q == n && wt + kd <= self.max_t {
                        by_deg[(wt + kd) as usize].push(TwistedGen {
                            p,
                            q: *q,
                            koszul: ki,
                            w: wi,
                        });
                    }
                }
            }
            let mut pos = FxHashMap::default();
            for (t, gens) in by_deg.iter().enumerate() {
                module.add_generators(t as i32, gens.len(), None);
                for (j, g) in gens.iter().enumerate() {
                    pos.insert((g.koszul, g.w), (t as i32, j));
                }
            }
            module.compute_basis(self.max_t);
            self.xmods.push(module);
            self.xgens.push(by_deg);
            self.xpos.push(pos);
        }
        self.d.push(Vec::new());
        for n in 1..=self.max_n {
            let mut d_n: Vec<Vec<FpVector>> = Vec::new();
            for t in 0..=self.max_t {
                let gens = self.xgens[n as usize][t as usize].clone();
                let row: Vec<FpVector> = gens
                    .iter()
                    .map(|&g| self.differential_of(n, t, g))
                    .collect();
                d_n.push(row);
            }
            self.d.push(d_n);
        }
    }

    /// The element `coeff * op * (x^S ⊗ w)` of $X_n$ in degree `t`, where `(S, w)` is the
    /// generator `key`.
    fn x_basis(&self, n: i32, t: i32, op: (i32, usize), key: (usize, usize)) -> Option<usize> {
        let &(gd, gi) = self.xpos[n as usize].get(&key)?;
        (t - op.0 == gd)
            .then(|| self.xmods[n as usize].operation_generator_to_index(op.0, op.1, gd, gi))
    }

    /// $d$ applied to an element of $X_n$ in degree `t`, with $d$ known on $X_n$ in degrees below
    /// or at `t` (as far as `self.d` holds it).
    fn d_of(&self, n: i32, t: i32, input: FpSlice) -> FpVector {
        Self::apply(
            &self.xmods[n as usize],
            &self.xmods[n as usize - 1],
            t,
            input,
            |gd, gi| {
                self.d[n as usize]
                    .get(gd as usize)
                    .and_then(|r| r.get(gi))
                    .cloned()
            },
        )
    }

    /// Builds $d$ of generator `g` of $X_n$ in degree `t`, solving for each $d_k$ in turn.
    fn differential_of(&mut self, n: i32, t: i32, g: TwistedGen) -> FpVector {
        let pr = self.algebra.prime();
        let target = Arc::clone(&self.xmods[n as usize - 1]);
        let mut z = FpVector::new(pr, target.dimension(t));
        // d_0: the Koszul differential.
        if g.q > 0 {
            let (s, _, _) = &self.koszul[g.koszul];
            for (ti, &e) in s.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                let mut smaller = s.clone();
                smaller[ti] -= 1;
                let k2 = self.koszul_index[&smaller];
                let op = self.primitive(ti + 1);
                if let Some(i) = self.x_basis(n - 1, t, op, (k2, g.w)) {
                    z.add_basis_element(i, 1);
                }
            }
        }
        let zero_koszul = self.koszul_index[&vec![0; self.koszul[0].0.len()]];
        for k in 1..=g.p {
            let row = g.p - k;
            let mut y = FpVector::new(pr, target.dimension(t));
            if g.q == 0 && k == 1 {
                // d_1 on the bottom of the row: the doubled differential, lifted.
                for &(od, oi, wj) in &self.w_diff[g.w] {
                    if let Some(i) = self.x_basis(n - 1, t, (od, oi), (zero_koszul, wj)) {
                        y.add_basis_element(i, 1);
                    }
                }
            } else if n >= 2 {
                // d_0(y_k) must cancel the row `row` part of d(z).
                let dz = self.d_of(n - 1, t, z.as_slice());
                let lower = &self.xmods[n as usize - 2];
                // Group the row-`row` part of d(z) by backbone generator.
                let mut by_w: FxHashMap<usize, Vec<(usize, u32)>> = FxHashMap::default();
                for (i, c) in dz.iter_nonzero() {
                    let og = lower.index_to_op_gen(t, i);
                    let xg = self.xgens[n as usize - 2][og.generator_degree as usize]
                        [og.generator_index];
                    assert!(xg.p <= row, "d(z) has a term above row {row}: {xg:?}");
                    if xg.p == row {
                        by_w.entry(xg.w).or_default().push((i, c));
                    }
                }
                let q_low = n - 2 - row;
                for (wj, entries) in by_w {
                    let wt = self.w[wj].1;
                    let kt = t - wt;
                    let kmod = &self.kmods[q_low as usize];
                    let mut r = FpVector::new(pr, kmod.dimension(kt));
                    for (i, c) in entries {
                        let og = lower.index_to_op_gen(t, i);
                        let xg = self.xgens[n as usize - 2][og.generator_degree as usize]
                            [og.generator_index];
                        let (kd, ki) = self.kpos[xg.koszul];
                        let idx = kmod.operation_generator_to_index(
                            og.operation_degree,
                            og.operation_index,
                            kd,
                            ki,
                        );
                        r.add_basis_element(idx, c);
                    }
                    let up = &self.kmods[q_low as usize + 1];
                    let mut pre = FpVector::new(pr, up.dimension(kt));
                    if let Some(qi) = self.kqi.get(&(q_low, kt)) {
                        qi.apply(pre.as_slice_mut(), 1, r.as_slice());
                    }
                    // Embed the preimage into X_{n-1}, row `row`.
                    for (i, c) in pre.iter_nonzero() {
                        let og = up.index_to_op_gen(kt, i);
                        let koszul_gen =
                            self.koszul_at(q_low + 1, og.generator_degree, og.generator_index);
                        if let Some(j) = self.x_basis(
                            n - 1,
                            t,
                            (og.operation_degree, og.operation_index),
                            (koszul_gen, wj),
                        ) {
                            y.add_basis_element(j, c);
                        }
                    }
                }
            }
            self.terms_by_k[k as usize] += y.iter_nonzero().count();
            z.add(&y, 1);
        }
        z
    }

    /// The Koszul generator of length `q` at `(degree, index)` of its module.
    fn koszul_at(&self, q: i32, degree: i32, index: usize) -> usize {
        (0..self.koszul.len())
            .find(|&i| self.koszul[i].2 == q && self.kpos[i] == (degree, index))
            .unwrap()
    }

    /// Checks $d^2 = 0$ on every generator.
    pub fn check_d_squared(&self) -> Result<(), String> {
        for n in 2..=self.max_n {
            for t in 0..=self.max_t {
                for (i, v) in self.d[n as usize][t as usize].iter().enumerate() {
                    let dd = self.d_of(n - 1, t, v.as_slice());
                    if !dd.is_zero() {
                        return Err(format!(
                            "d^2 != 0 on {:?} in degree {t}",
                            self.xgens[n as usize][t as usize][i]
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// $\mathrm{Ext}^{n, t}_A(\mathbb{F}_2, \mathbb{F}_2)$ from $\mathrm{Hom}_A(X, \mathbb{F}_2)$: the
    /// generators minus the ranks of the indecomposable parts of $d$ in and out.
    pub fn ext(&self, n: i32, t: i32) -> usize {
        let rank = |n: i32| -> usize {
            if n < 1 || n > self.max_n {
                return 0;
            }
            let src = &self.xgens[n as usize][t as usize];
            let tgt = &self.xgens[n as usize - 1][t as usize];
            if src.is_empty() || tgt.is_empty() {
                return 0;
            }
            let module = &self.xmods[n as usize - 1];
            let mut m = fp::matrix::Matrix::new(self.algebra.prime(), src.len(), tgt.len());
            for (i, v) in self.d[n as usize][t as usize].iter().enumerate() {
                for (j, _) in tgt.iter().enumerate() {
                    let idx = module.operation_generator_to_index(0, 0, t, j);
                    if v.entry(idx) != 0 {
                        m.row_mut(i).set_entry(j, 1);
                    }
                }
            }
            m.row_reduce()
        };
        self.xgens[n as usize][t as usize].len() - rank(n) - rank(n + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebraic_novikov::unit_module;

    /// The twisted tensor product is a resolution: $d^2 = 0$, and its Ext is that of the
    /// minimal resolution.
    #[test]
    fn twisted_is_a_resolution() {
        let (max_n, max_t) = std::env::var("ZZ_TWIST").ok().map_or((5, 20), |v| {
            let mut it = v.split(',').map(|x| x.parse().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        });
        let res = crate::nassau::Resolution::new(Arc::new(unit_module()));
        res.compute_through_stem(Bidegree::n_s(max_t, max_n + 1));
        let x = TwistedResolution::new(&res, max_n, max_t);
        x.check_d_squared().unwrap();
        for n in 0..max_n {
            for t in n..=max_t {
                assert_eq!(
                    x.ext(n, t),
                    res.number_of_gens_in_bidegree(Bidegree::s_t(n, t)),
                    "Ext at (n, t) = ({n}, {t})"
                );
            }
        }
        let gens: usize = x.xgens.iter().flatten().map(Vec::len).sum();
        let minimal: usize = (0..=max_n)
            .flat_map(|n| (0..=max_t).map(move |t| (n, t)))
            .map(|(n, t)| res.number_of_gens_in_bidegree(Bidegree::s_t(n, t)))
            .sum();
        eprintln!(
            "TWIST max_n {max_n} max_t {max_t}: generators {gens} (minimal {minimal}), d_k terms {:?}",
            x.terms_by_k
        );
    }
}
