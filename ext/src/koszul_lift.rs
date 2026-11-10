//! The first $\tau$-Bockstein differential on $\mathrm{Ext}_P(Q)$ in the Koszul gauge.

use std::collections::BTreeMap;

use algebra::{
    Algebra, MilnorAlgebra, MotivicMilnorAlgebra,
    milnor_algebra::{MilnorBasisElement, PPart},
};
use fp::{prime::TWO, vector::FpVector};
use rustc_hash::FxHashMap;

use crate::{
    algebraic_novikov::{
        ActMemo, Convention, PackedXi, QDOWN_VARS, QDown, QMono, chi_table, monomials, pack,
        qmono_len, unpack, v_degree,
    },
    tau_lift::{Backbone, Gen},
};

/// An element of the free $A/\tau$-module on the generators: each generator's coefficient, an
/// element of $A/\tau$ in the Milnor basis.
type Elem = FxHashMap<Gen, FpVector>;

/// A coefficient in $A_\mathbb{C}$: by power of $\tau$, an element of one degree of the basis.
type TauCoef = BTreeMap<u32, FpVector>;

/// An element of the free $A_\mathbb{C}$-module on the generators.
type TElem = FxHashMap<Gen, TauCoef>;

/// Adds $\tau^e w$ to the coefficient of `g`.
fn add_tau(e: &mut TElem, g: Gen, pow: u32, w: &FpVector) {
    let c = e.entry(g).or_default();
    match c.get_mut(&pow) {
        Some(v) => {
            v.add(w, 1);
            if v.is_zero() {
                c.remove(&pow);
            }
        }
        None => {
            if !w.is_zero() {
                c.insert(pow, w.clone());
            }
        }
    }
    if c.is_empty() {
        e.remove(&g);
    }
}

/// The halved degree of a monomial in the $v_i$.
fn qdeg(k: &QMono) -> i32 {
    k.iter()
        .enumerate()
        .map(|(i, &e)| e as i32 * v_degree(i))
        .sum()
}

/// The halved degree of a packed $P(R)$.
fn rdeg(r: PackedXi) -> i32 {
    unpack(r)
        .iter()
        .enumerate()
        .map(|(j, &e)| e as i32 * v_degree(j + 1))
        .sum()
}

/// The sub-exponent vectors $R' \leq R$, packed.
fn splits(r: PackedXi) -> Vec<PackedXi> {
    let mut out = vec![0];
    for (j, e) in unpack(r).into_iter().enumerate() {
        let unit = pack(&[vec![0; j], vec![1]].concat());
        out = out
            .into_iter()
            .flat_map(|s| (0..=e as u128).map(move |k| s + k * unit))
            .collect();
    }
    out
}

/// The motivic degree of a generator.
fn gen_degree(g: &Gen) -> i32 {
    2 * g.1.0 + 2 * qdeg(&g.2) + qmono_len(&g.2) as i32
}

/// The motivic weight of a generator.
fn gen_weight(g: &Gen) -> i32 {
    g.1.0 + qdeg(&g.2)
}

/// The untwisted resolution with the $\tau$-correction built by homological perturbation along
/// the Koszul contraction, so that each correction is a local expression in the backbone.
///
/// Writing $A/\tau$ in the basis $P(R) Q^E$, the Koszul part $D_0$ is right multiplication by the
/// $Q_i$, contracted by $h(P(R) Q^E \otimes \gamma_K) = P(R) Q^{E - e_i} \otimes \gamma_{K + e_i}$
/// for $i$ the first index in $E$ or $K$, when it is in $E$. With $W = D_1^2 / \tau$, the
/// $\tau$-part of the $A_\mathbb{C}$ products of the backbone part $D_1$, the correction is
/// $D_2 + D_3 + \dots$ with
/// $$D_2 = h(W + D_2 D_0), \qquad D_3 = h(D_1 D_2 + D_2 D_1 + D_3 D_0)$$
/// on generators, by induction on the length of $K$. The constant terms of $D_3$ are
/// $\delta_1$.
pub struct KoszulLift {
    /// $A/\tau$.
    algebra: MilnorAlgebra,
    /// The generators by homological degree $a + |K|$ and motivic degree.
    pub gens: Vec<BTreeMap<i32, Vec<Gen>>>,
    /// The terms $P(R') \cdot g'$ of $D_1$ on each generator.
    d1: FxHashMap<Gen, Vec<(PackedXi, Gen)>>,
    /// $D_2$ and $D_3$ on each generator.
    d2: FxHashMap<Gen, Elem>,
    d3: FxHashMap<Gen, Elem>,
    /// Per degree, $P(R) Q^E$ in the Milnor basis, by Milnor basis label.
    pq_to_milnor: Vec<Vec<FpVector>>,
    /// Per degree, the Milnor basis in the basis $P(R) Q^E$.
    milnor_to_pq: Vec<Vec<FpVector>>,
}

impl KoszulLift {
    /// The lift through homological degree `max_h` and motivic degree `max_t`.
    pub fn new<B: Backbone>(res: &B, max_h: i32, max_t: i32) -> Self {
        let algebra = MilnorAlgebra::motivic_mod_tau(Default::default(), false);
        algebra.compute_basis(max_t);
        let engine = MotivicMilnorAlgebra::new();
        engine.compute_basis(max_t);

        // Q monomials by (length, halved degree), with v_0 explicit.
        let max_d = max_t / 2;
        let mut q_by: FxHashMap<(u32, i32), Vec<QMono>> = FxHashMap::default();
        for (d, bucket) in monomials(max_h as u32 + 2, |_| max_d).iter().enumerate() {
            for m in bucket {
                let len: u32 = m.iter().sum();
                for j in len..=max_h as u32 + 2 {
                    if 2 * d as i32 + j as i32 > max_t {
                        break;
                    }
                    let mut q = [0u16; QDOWN_VARS];
                    for (i, &e) in m.iter().enumerate() {
                        q[i] = e as u16;
                    }
                    q[0] += (j - len) as u16;
                    q_by.entry((j, d as i32)).or_default().push(q);
                }
            }
        }
        let q_down = QDown::new(max_h as u32 + 2, |_| max_d, Convention::Xi);
        let memo = ActMemo::default();
        let mut up_cache: FxHashMap<(PackedXi, QMono), Vec<QMono>> = FxHashMap::default();
        let mut up = |s: PackedXi, k: QMono| -> Vec<QMono> {
            if s == 0 {
                return vec![k];
            }
            up_cache
                .entry((s, k))
                .or_insert_with(|| {
                    let d2 = qdeg(&k) + rdeg(s);
                    let len = qmono_len(&k);
                    q_by.get(&(len, d2))
                        .into_iter()
                        .flatten()
                        .filter(|k2| {
                            q_down.act_memo(s, k2, &memo, 1 << 22).iter().any(|out| {
                                let mut out = *out;
                                out[0] += (len - qmono_len(&out)) as u16;
                                out == k
                            })
                        })
                        .copied()
                        .collect()
                })
                .clone()
        };
        let chi = chi_table(max_d);

        // Generators, by homological degree and motivic degree.
        let mut gens: Vec<BTreeMap<i32, Vec<Gen>>> = Vec::new();
        for h in 0..=max_h + 1 {
            let mut by_t: BTreeMap<i32, Vec<Gen>> = BTreeMap::new();
            for a in 0..=h {
                let j = (h - a) as u32;
                for t_z in 0..=max_t / 2 {
                    for z in 0..res.num_gens(a, t_z) as u32 {
                        for dd in 0..=max_d {
                            let t = 2 * t_z + 2 * dd + j as i32;
                            if t > max_t {
                                break;
                            }
                            for &k in q_by.get(&(j, dd)).into_iter().flatten() {
                                by_t.entry(t).or_default().push((a, (t_z, z), k));
                            }
                        }
                    }
                }
            }
            for t in 0..=max_t {
                by_t.entry(t).or_default().sort();
            }
            gens.push(by_t);
        }

        // D_1 on each generator.
        let mut d1: FxHashMap<Gen, Vec<(PackedXi, Gen)>> = FxHashMap::default();
        for by_t in &gens {
            for list in by_t.values() {
                for &g in list {
                    let (a, (t_z, z), k) = g;
                    let mut terms = Vec::new();
                    if a > 0 {
                        for (r, x) in res.terms(a, t_z, z) {
                            for r1 in splits(r) {
                                for &s in chi.get(&(r - r1)).into_iter().flatten() {
                                    for k2 in up(s, k) {
                                        terms.push((r1, (a - 1, x, k2)));
                                    }
                                }
                            }
                        }
                    }
                    d1.insert(g, terms);
                }
            }
        }

        // The change of basis between Q^E P(R) (Milnor) and P(R) Q^E.
        let label = |d: i32, i: usize| {
            let e = algebra.basis_element_from_index(d, i);
            (e.q_part, e.p_part)
        };
        let mut pq_to_milnor = Vec::new();
        let mut milnor_to_pq = Vec::new();
        for d in 0..=max_t {
            let dim = algebra.dimension(d);
            let rows: Vec<FpVector> = (0..dim)
                .map(|i| {
                    let (q, p) = label(d, i);
                    let mut pe = MilnorBasisElement {
                        q_part: 0,
                        p_part: p,
                        degree: 0,
                    };
                    algebra.compute_degree(&mut pe);
                    let mut qe = MilnorBasisElement {
                        q_part: q,
                        p_part: PPart::from_slice(&[]),
                        degree: 0,
                    };
                    algebra.compute_degree(&mut qe);
                    let mut v = FpVector::new(TWO, dim);
                    algebra.multiply_basis_elements(
                        v.as_slice_mut(),
                        1,
                        pe.degree,
                        algebra.basis_element_to_index(&pe),
                        qe.degree,
                        algebra.basis_element_to_index(&qe),
                    );
                    v
                })
                .collect();
            milnor_to_pq.push(invert(&rows));
            pq_to_milnor.push(rows);
        }

        let mut lift = Self {
            algebra,
            gens,
            d1,
            d2: FxHashMap::default(),
            d3: FxHashMap::default(),
            pq_to_milnor,
            milnor_to_pq,
        };

        // W(g): the τ-part of the A_C products along D_1 D_1.
        let p_index = |algebra: &MilnorAlgebra, r: PackedXi| -> (i32, usize) {
            let p: Vec<_> = unpack(r).into_iter().map(|x| x as _).collect();
            let mut e = MilnorBasisElement {
                q_part: 0,
                p_part: PPart::from_slice(&p),
                degree: 0,
            };
            algebra.compute_degree(&mut e);
            (e.degree, algebra.basis_element_to_index(&e))
        };
        let mut order: Vec<Gen> = lift.d1.keys().copied().collect();
        order.sort_by_key(|g| (g.0, qmono_len(&g.2), *g));
        for &g in &order {
            let mut w = Elem::default();
            for &(r1, g1) in &lift.d1[&g] {
                let (o1, i1) = p_index(&lift.algebra, r1);
                for &(r2, g2) in lift.d1.get(&g1).into_iter().flatten() {
                    let (o2, i2) = p_index(&lift.algebra, r2);
                    // The engine orders its factors oppositely to the Milnor algebra.
                    for idx in engine.product_indexed(o2, i2, o1, i1) {
                        if engine.tau_exponent(o2, i2, o1, i1, idx) == 1 {
                            let dim = lift.algebra.dimension(o1 + o2);
                            w.entry(g2)
                                .or_insert_with(|| FpVector::new(TWO, dim))
                                .add_basis_element(idx, 1);
                        }
                    }
                }
            }
            // D_2 = h(W + Σ Q_i D_2(g_i)).
            let mut arg = w;
            for (i, gi) in lower(&g) {
                if let Some(e) = lift.d2.get(&gi) {
                    let qe = lift.q_mul(i, e, gen_degree(&gi));
                    add_elem(&mut arg, &qe);
                }
            }
            let d2 = lift.contract(&arg, gen_degree(&g));
            lift.d2.insert(g, d2);
        }
        for &g in &order {
            // D_3 = h(D_1 D_2 + D_2 D_1 + Σ Q_i D_3(g_i)).
            let t = gen_degree(&g);
            let mut arg = lift.apply_d1(&lift.d2[&g], t);
            for &(r1, g1) in &lift.d1[&g] {
                if let Some(e) = lift.d2.get(&g1) {
                    let pe = lift.p_mul(r1, e, gen_degree(&g1));
                    add_elem(&mut arg, &pe);
                }
            }
            for (i, gi) in lower(&g) {
                if let Some(e) = lift.d3.get(&gi) {
                    let qe = lift.q_mul(i, e, gen_degree(&gi));
                    add_elem(&mut arg, &qe);
                }
            }
            let d3 = lift.contract(&arg, t);
            lift.d3.insert(g, d3);
        }
        lift
    }

    /// $c \cdot e$ for $c$ the basis element `(deg, idx)`, with `e` of total degree `t`.
    fn left_mul(&self, deg: i32, idx: usize, e: &Elem, t: i32) -> Elem {
        let mut out = Elem::default();
        for (g, v) in e {
            let vd = t - gen_degree(g);
            let dim = self.algebra.dimension(deg + vd);
            let mut w = FpVector::new(TWO, dim);
            for (j, _) in v.iter_nonzero() {
                self.algebra
                    .multiply_basis_elements(w.as_slice_mut(), 1, deg, idx, vd, j);
            }
            if !w.is_zero() {
                add_term(&mut out, *g, &w);
            }
        }
        out
    }

    /// $Q_i \cdot e$, with `e` of total degree `t`.
    fn q_mul(&self, i: usize, e: &Elem, t: i32) -> Elem {
        let mut q = MilnorBasisElement {
            q_part: 1 << i,
            p_part: PPart::from_slice(&[]),
            degree: 0,
        };
        self.algebra.compute_degree(&mut q);
        self.left_mul(q.degree, self.algebra.basis_element_to_index(&q), e, t)
    }

    /// $P(R) \cdot e$, with `e` of total degree `t`.
    fn p_mul(&self, r: PackedXi, e: &Elem, t: i32) -> Elem {
        let p: Vec<_> = unpack(r).into_iter().map(|x| x as _).collect();
        let mut pe = MilnorBasisElement {
            q_part: 0,
            p_part: PPart::from_slice(&p),
            degree: 0,
        };
        self.algebra.compute_degree(&mut pe);
        self.left_mul(pe.degree, self.algebra.basis_element_to_index(&pe), e, t)
    }

    /// $D_1$ on an element of total degree `t`: each coefficient times the terms of $D_1$ on
    /// its generator.
    fn apply_d1(&self, e: &Elem, t: i32) -> Elem {
        let mut out = Elem::default();
        for (g, v) in e {
            let vd = t - gen_degree(g);
            for &(r1, g1) in &self.d1[g] {
                let p: Vec<_> = unpack(r1).into_iter().map(|x| x as _).collect();
                let mut pe = MilnorBasisElement {
                    q_part: 0,
                    p_part: PPart::from_slice(&p),
                    degree: 0,
                };
                self.algebra.compute_degree(&mut pe);
                let pidx = self.algebra.basis_element_to_index(&pe);
                let mut w = FpVector::new(TWO, self.algebra.dimension(vd + pe.degree));
                for (j, _) in v.iter_nonzero() {
                    self.algebra.multiply_basis_elements(
                        w.as_slice_mut(),
                        1,
                        vd,
                        j,
                        pe.degree,
                        pidx,
                    );
                }
                if !w.is_zero() {
                    add_term(&mut out, g1, &w);
                }
            }
        }
        out
    }

    /// The Koszul contraction $h$ on an element of total degree `t`.
    fn contract(&self, e: &Elem, t: i32) -> Elem {
        let mut out = Elem::default();
        for (g, v) in e {
            let d = (t - gen_degree(g)) as usize;
            // Into the basis P(R) Q^E.
            let mut pq = FpVector::new(TWO, v.len());
            for (j, _) in v.iter_nonzero() {
                pq.add(&self.milnor_to_pq[d][j], 1);
            }
            for (l, _) in pq.iter_nonzero() {
                let elt = self.algebra.basis_element_from_index(d as i32, l);
                let (q, k) = (elt.q_part, g.2);
                let first = (0..QDOWN_VARS).find(|&i| q >> i & 1 == 1 || k[i] > 0);
                let Some(i) = first else { continue };
                if q >> i & 1 == 0 {
                    continue;
                }
                let mut k2 = k;
                k2[i] += 1;
                let g2 = (g.0, g.1, k2);
                let mut rest = MilnorBasisElement {
                    q_part: q & !(1 << i),
                    p_part: elt.p_part,
                    degree: 0,
                };
                self.algebra.compute_degree(&mut rest);
                let ri = self.algebra.basis_element_to_index(&rest);
                let w = &self.pq_to_milnor[rest.degree as usize][ri];
                add_term(&mut out, g2, w);
            }
        }
        out
    }

    /// The generators $g'$ of $M_{h-1}$ in degree `t` with $\delta_0 g'^* \ni g^*$ ($D_1$ has a
    /// constant $g'$ on $g$), and likewise for $\delta_1$ from $D_3$.
    fn constants(&self, h: usize, t: i32, which: u8) -> Vec<FxHashMap<usize, Vec<usize>>> {
        let index: FxHashMap<Gen, usize> = self.gens[h - 1][&t]
            .iter()
            .enumerate()
            .map(|(i, g)| (*g, i))
            .collect();
        let mut m: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for (i, g) in self.gens[h][&t].iter().enumerate() {
            let targets: Vec<Gen> = if which == 0 {
                let mut c: FxHashMap<Gen, u8> = FxHashMap::default();
                for &(r, g2) in &self.d1[g] {
                    if r == 0 {
                        *c.entry(g2).or_default() ^= 1;
                    }
                }
                c.into_iter()
                    .filter(|&(_, x)| x == 1)
                    .map(|(g2, _)| g2)
                    .collect()
            } else {
                self.d3
                    .get(g)
                    .into_iter()
                    .flatten()
                    .filter(|(g2, v)| gen_degree(g2) == gen_degree(g) && v.entry(0) == 1)
                    .map(|(g2, _)| *g2)
                    .collect()
            };
            for g2 in targets {
                if let Some(&j) = index.get(&g2) {
                    m.entry(j).or_default().push(i);
                }
            }
        }
        vec![m]
    }

    /// $\dim E_1$ and the rank of $d_1$ out of it by (stem, $s$, weight), for $s$ below the top.
    pub fn pages(&self) -> BTreeMap<(i32, i32, i32), (usize, usize)> {
        let top = self.gens.len() - 2;
        let mut out = BTreeMap::new();
        let max_t = *self.gens[0].keys().max().unwrap();
        for t in 0..=max_t {
            let n = |h: usize| self.gens[h][&t].len();
            let d0: Vec<FxHashMap<usize, Vec<usize>>> = (0..=top + 1)
                .map(|h| {
                    if h == 0 {
                        FxHashMap::default()
                    } else {
                        self.constants(h, t, 0).remove(0)
                    }
                })
                .collect();
            let d1: Vec<FxHashMap<usize, Vec<usize>>> = (0..=top + 1)
                .map(|h| {
                    if h == 0 {
                        FxHashMap::default()
                    } else {
                        self.constants(h, t, 1).remove(0)
                    }
                })
                .collect();
            for h in 0..top {
                let mut bound = Echelon::default();
                for g in 0..n(h) {
                    let mut v = FpVector::new(TWO, n(h + 1));
                    for &g2 in d0[h + 1].get(&g).into_iter().flatten() {
                        v.add_basis_element(g2, 1);
                    }
                    bound.insert(v);
                }
                let b_rank = bound.rows.len();
                let weights: std::collections::BTreeSet<i32> =
                    self.gens[h][&t].iter().map(gen_weight).collect();
                for w in weights {
                    let here: Vec<usize> = (0..n(h))
                        .filter(|&g| gen_weight(&self.gens[h][&t][g]) == w)
                        .collect();
                    let incoming: Vec<Vec<usize>> = if h == 0 {
                        Vec::new()
                    } else {
                        (0..n(h - 1))
                            .map(|g| d0[h].get(&g).cloned().unwrap_or_default())
                            .collect()
                    };
                    let reps = cohomology(
                        &here,
                        |g| d0[h + 1].get(&g).cloned().unwrap_or_default(),
                        n(h + 1),
                        incoming,
                        n(h),
                    );
                    if reps.is_empty() {
                        continue;
                    }
                    let mut ech = bound.clone();
                    for rep in &reps {
                        let mut img = FpVector::new(TWO, n(h + 1));
                        for (g, _) in rep.iter_nonzero() {
                            for &g2 in d1[h + 1].get(&g).into_iter().flatten() {
                                img.add_basis_element(g2, 1);
                            }
                        }
                        ech.insert(img);
                    }
                    out.insert(
                        (t - h as i32, h as i32, w),
                        (reps.len(), ech.rows.len() - b_rank),
                    );
                }
            }
        }
        out
    }

    /// Checks $D_0 D_2 + D_2 D_0 = W$ is consistent, by $\delta_0 \delta_1 + \delta_1 \delta_0 = 0$
    /// on cochains: returns the number of generators where it fails.
    pub fn check_cocycle(&self) -> usize {
        let mut bad = 0;
        let top = self.gens.len() - 2;
        let max_t = *self.gens[0].keys().max().unwrap();
        for t in 0..=max_t {
            for h in 1..top {
                // (δ_0 δ_1 + δ_1 δ_0) on each cochain g^* of M_{h-1}, read on M_{h+1}.
                let d0a = self.constants(h, t, 0).remove(0);
                let d1a = self.constants(h, t, 1).remove(0);
                let d0b = self.constants(h + 1, t, 0).remove(0);
                let d1b = self.constants(h + 1, t, 1).remove(0);
                for g in 0..self.gens[h - 1][&t].len() {
                    let mut acc: FxHashMap<usize, u8> = FxHashMap::default();
                    for &m in d0a.get(&g).into_iter().flatten() {
                        for &x in d1b.get(&m).into_iter().flatten() {
                            *acc.entry(x).or_default() ^= 1;
                        }
                    }
                    for &m in d1a.get(&g).into_iter().flatten() {
                        for &x in d0b.get(&m).into_iter().flatten() {
                            *acc.entry(x).or_default() ^= 1;
                        }
                    }
                    if acc.values().any(|&x| x == 1) {
                        bad += 1;
                    }
                }
            }
        }
        bad
    }
}

impl KoszulLift {
    /// The differential of the untwisted resolution lifted to all of $A_\mathbb{C}$, by the same
    /// perturbation to every order: $D_m = h(\sum_{i + j = m} D_i D_j + D_m D_0)$ for $m \geq 2$,
    /// with every product exact in $A_\mathbb{C}$. Returns, for each generator $g$, the
    /// generators $g'$ with a constant $\tau^e g'$ in $D g$: the cochain differential
    /// $\delta = \sum_e \tau^e \delta_e$.
    pub fn all_orders(&self) -> FxHashMap<Gen, Vec<(Gen, u32)>> {
        let max_t = *self.gens[0].keys().max().unwrap();
        let engine = MotivicMilnorAlgebra::new();
        engine.compute_basis(max_t);
        // a · b in A_C, the Milnor algebra's order: the engine's is opposite.
        let mul = |d1: i32, i1: usize, d2: i32, i2: usize| -> Vec<(usize, u32)> {
            engine
                .product_indexed(d2, i2, d1, i1)
                .into_iter()
                .map(|idx| (idx, engine.tau_exponent(d2, i2, d1, i1, idx)))
                .collect()
        };
        let dim = |d: i32| self.algebra.dimension(d);
        // c · e, for c a coefficient of degree dc and e of total degree t.
        let left = |c: &TauCoef, dc: i32, e: &TElem, t: i32| -> TElem {
            let mut out = TElem::default();
            for (g, v) in e {
                let vd = t - gen_degree(g);
                for (&e1, cv) in c {
                    for (i1, _) in cv.iter_nonzero() {
                        for (&e2, w) in v {
                            for (i2, _) in w.iter_nonzero() {
                                for (idx, x) in mul(dc, i1, vd, i2) {
                                    let mut u = FpVector::new(TWO, dim(dc + vd));
                                    u.set_entry(idx, 1);
                                    add_tau(&mut out, *g, e1 + e2 + x, &u);
                                }
                            }
                        }
                    }
                }
            }
            out
        };
        let unit = |d: i32, idx: usize| -> TauCoef {
            let mut v = FpVector::new(TWO, dim(d));
            v.set_entry(idx, 1);
            [(0, v)].into_iter().collect()
        };
        let p_of = |r: PackedXi| -> (i32, usize) {
            let p: Vec<_> = unpack(r).into_iter().map(|x| x as _).collect();
            let mut e = MilnorBasisElement {
                q_part: 0,
                p_part: PPart::from_slice(&p),
                degree: 0,
            };
            self.algebra.compute_degree(&mut e);
            (e.degree, self.algebra.basis_element_to_index(&e))
        };
        let q_of = |i: usize| -> (i32, usize) {
            let mut e = MilnorBasisElement {
                q_part: 1 << i,
                p_part: PPart::from_slice(&[]),
                degree: 0,
            };
            self.algebra.compute_degree(&mut e);
            (e.degree, self.algebra.basis_element_to_index(&e))
        };
        // D_1 on an element of total degree t: each coefficient times P(R') at the target.
        let apply_d1 = |e: &TElem, t: i32| -> TElem {
            let mut out = TElem::default();
            for (g, v) in e {
                let vd = t - gen_degree(g);
                for &(r1, g1) in &self.d1[g] {
                    let (pd, pi) = p_of(r1);
                    for (&e1, w) in v {
                        for (i, _) in w.iter_nonzero() {
                            for (idx, x) in mul(vd, i, pd, pi) {
                                let mut u = FpVector::new(TWO, dim(vd + pd));
                                u.set_entry(idx, 1);
                                add_tau(&mut out, g1, e1 + x, &u);
                            }
                        }
                    }
                }
            }
            out
        };
        // The Koszul contraction, power by power of τ.
        let contract = |e: &TElem, t: i32| -> TElem {
            let mut out = TElem::default();
            for (g, c) in e {
                for (&pow, w) in c {
                    let single: Elem = [(*g, w.clone())].into_iter().collect();
                    for (g2, v) in self.contract(&single, t) {
                        add_tau(&mut out, g2, pow, &v);
                    }
                }
            }
            out
        };

        let mut order: Vec<Gen> = self.d1.keys().copied().collect();
        order.sort_by_key(|g| (g.0, qmono_len(&g.2), *g));
        let max_a = order.iter().map(|g| g.0).max().unwrap_or(0) as usize;
        // d[m][g] = D_m(g), for m >= 1.
        let mut d: Vec<FxHashMap<Gen, TElem>> = vec![FxHashMap::default(); max_a + 1];
        for &g in &order {
            let _ = gen_degree(&g);
            let mut e = TElem::default();
            for &(r1, g1) in &self.d1[&g] {
                let (pd, pi) = p_of(r1);
                for (pow, v) in unit(pd, pi) {
                    add_tau(&mut e, g1, pow, &v);
                }
            }
            d[1].insert(g, e);
        }
        for &g in &order {
            let t = gen_degree(&g);
            for m in 2..=g.0 as usize {
                let mut arg = TElem::default();
                for i in 1..m {
                    let inner = &d[m - i][&g];
                    let term = if i == 1 {
                        apply_d1(inner, t)
                    } else {
                        let mut acc = TElem::default();
                        for (g2, c) in inner {
                            let dc = t - gen_degree(g2);
                            if let Some(dg2) = d[i].get(g2) {
                                for (g3, v) in left(c, dc, dg2, gen_degree(g2)) {
                                    for (pow, w) in v {
                                        add_tau(&mut acc, g3, pow, &w);
                                    }
                                }
                            }
                        }
                        acc
                    };
                    for (g3, v) in term {
                        for (pow, w) in v {
                            add_tau(&mut arg, g3, pow, &w);
                        }
                    }
                }
                for (i, gi) in lower(&g) {
                    if let Some(dm) = d[m].get(&gi) {
                        let (qd, qi) = q_of(i);
                        for (g3, v) in left(&unit(qd, qi), qd, dm, gen_degree(&gi)) {
                            for (pow, w) in v {
                                add_tau(&mut arg, g3, pow, &w);
                            }
                        }
                    }
                }
                let dm = contract(&arg, t);
                d[m].insert(g, dm);
            }
        }

        // The constant terms of every D_m.
        let mut delta: FxHashMap<Gen, Vec<(Gen, u32)>> = FxHashMap::default();
        for dm in &d {
            for (g, e) in dm {
                for (g2, c) in e {
                    if gen_degree(g2) != gen_degree(g) {
                        continue;
                    }
                    for (&pow, v) in c {
                        if v.entry(0) == 1 {
                            delta.entry(*g).or_default().push((*g2, pow));
                        }
                    }
                }
            }
        }
        delta
    }

    /// $\mathrm{Ext}$ of the cochain complex with differential $\delta$ at $\tau = 1$, by
    /// (stem, $s$), for $s$ below the top: the classical $\mathrm{Ext}_A$.
    pub fn classical(
        &self,
        delta: &FxHashMap<Gen, Vec<(Gen, u32)>>,
    ) -> BTreeMap<(i32, i32), usize> {
        let top = self.gens.len() - 2;
        let max_t = *self.gens[0].keys().max().unwrap();
        let mut out = BTreeMap::new();
        for t in 0..=max_t {
            let n = |h: usize| self.gens[h][&t].len();
            let rank = |h: usize| -> usize {
                // δ: V^h → V^{h+1}, row g2 (in V^h) has entries at the g in V^{h+1}.
                let index: FxHashMap<Gen, usize> = self.gens[h][&t]
                    .iter()
                    .enumerate()
                    .map(|(i, g)| (*g, i))
                    .collect();
                let mut rows = vec![FpVector::new(TWO, n(h + 1)); n(h)];
                for (j, g) in self.gens[h + 1][&t].iter().enumerate() {
                    for (g2, _) in delta.get(g).into_iter().flatten() {
                        if let Some(&i) = index.get(g2) {
                            rows[i].add_basis_element(j, 1);
                        }
                    }
                }
                let mut ech = Echelon::default();
                rows.into_iter().filter(|r| ech.insert(r.clone())).count()
            };
            let ranks: Vec<usize> = (0..=top).map(rank).collect();
            for h in 0..top {
                let dim = n(h) - ranks[h] - if h > 0 { ranks[h - 1] } else { 0 };
                if dim > 0 {
                    out.insert((t - h as i32, h as i32), dim);
                }
            }
        }
        out
    }
}

/// The generators $g$ with $K$ lowered by one $v_i$, and the $i$.
fn lower(g: &Gen) -> Vec<(usize, Gen)> {
    (0..QDOWN_VARS)
        .filter(|&i| g.2[i] > 0)
        .map(|i| {
            let mut k = g.2;
            k[i] -= 1;
            (i, (g.0, g.1, k))
        })
        .collect()
}

/// Adds `w` to the coefficient of `g`.
fn add_term(e: &mut Elem, g: Gen, w: &FpVector) {
    match e.get_mut(&g) {
        Some(v) => {
            v.add(w, 1);
            if v.is_zero() {
                e.remove(&g);
            }
        }
        None => {
            if !w.is_zero() {
                e.insert(g, w.clone());
            }
        }
    }
}

/// `a += b`.
fn add_elem(a: &mut Elem, b: &Elem) {
    for (g, v) in b {
        add_term(a, *g, v);
    }
}

/// The inverse of an invertible matrix over $\mathbb{F}_2$ given by its rows.
fn invert(rows: &[FpVector]) -> Vec<FpVector> {
    let n = rows.len();
    let mut m: Vec<FpVector> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut v = FpVector::new(TWO, 2 * n);
            for (j, _) in r.iter_nonzero() {
                v.set_entry(j, 1);
            }
            v.set_entry(n + i, 1);
            v
        })
        .collect();
    for col in 0..n {
        let p = (col..n)
            .find(|&i| m[i].entry(col) == 1)
            .expect("invertible");
        m.swap(col, p);
        let pivot = m[col].clone();
        for (i, row) in m.iter_mut().enumerate() {
            if i != col && row.entry(col) == 1 {
                row.add(&pivot, 1);
            }
        }
    }
    // Row operations took [M | I] to [I | M^{-1}]: row i expresses basis element i.
    m.iter()
        .map(|row| {
            let mut v = FpVector::new(TWO, n);
            for j in 0..n {
                if row.entry(n + j) == 1 {
                    v.set_entry(j, 1);
                }
            }
            v
        })
        .collect()
}

/// Row-echelon form over $\mathbb{F}_2$.
#[derive(Default, Clone)]
struct Echelon {
    /// Rows with distinct pivots.
    rows: Vec<(usize, FpVector)>,
}

impl Echelon {
    /// Adds `v`, returning whether it was independent of the rows.
    fn insert(&mut self, mut v: FpVector) -> bool {
        for (p, row) in &self.rows {
            if v.entry(*p) == 1 {
                v.add(row, 1);
            }
        }
        match v.first_nonzero() {
            Some((p, _)) => {
                self.rows.push((p, v));
                true
            }
            None => false,
        }
    }
}

/// Representatives of $\ker \delta_0 / \operatorname{im} \delta_0$ on the cochains `here`.
fn cohomology(
    here: &[usize],
    out: impl Fn(usize) -> Vec<usize>,
    next_dim: usize,
    incoming: Vec<Vec<usize>>,
    dim: usize,
) -> Vec<FpVector> {
    let mut ech: Vec<(usize, FpVector)> = Vec::new();
    let mut kernel = Vec::new();
    for &g in here {
        let mut v = FpVector::new(TWO, next_dim + dim);
        for g2 in out(g) {
            v.add_basis_element(g2, 1);
        }
        v.add_basis_element(next_dim + g, 1);
        for (p, row) in &ech {
            if v.entry(*p) == 1 {
                v.add(row, 1);
            }
        }
        match v.first_nonzero() {
            Some((p, _)) if p < next_dim => ech.push((p, v)),
            _ => {
                let mut k = FpVector::new(TWO, dim);
                for (j, _) in v.iter_nonzero() {
                    k.set_entry(j - next_dim, 1);
                }
                kernel.push(k);
            }
        }
    }
    let mut basis = Echelon::default();
    for gs in incoming {
        let mut v = FpVector::new(TWO, dim);
        for g in gs {
            v.add_basis_element(g, 1);
        }
        basis.insert(v);
    }
    kernel
        .into_iter()
        .filter(|k| basis.insert(k.clone()))
        .collect()
}
