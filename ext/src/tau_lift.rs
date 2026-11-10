//! The first $\tau$-Bockstein differential on $\mathrm{Ext}_P(Q)$, from the untwisted resolution.

use std::{collections::BTreeMap, sync::Arc};

use algebra::{
    Algebra, MilnorAlgebra, MotivicMilnorAlgebra,
    milnor_algebra::{MilnorBasisElement, PPart},
    module::{
        FreeModule, Module,
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
    },
};
use fp::{prime::TWO, vector::FpVector};
use rustc_hash::FxHashMap;
use sseq::coordinates::Bidegree;

use crate::{
    algebraic_novikov::{
        ActMemo, Convention, PackedXi, QDOWN_VARS, QDown, QMono, chi_table, monomials, pack,
        qmono_len, unpack, v_degree,
    },
    chain_complex::FreeChainComplex,
};

/// A generator $z \otimes \gamma_K$ of the untwisted resolution: the filtration $a$, the backbone
/// generator $z$ of $C_a$, and the monomial $K$ whose dual is $\gamma_K$.
pub type Gen = (i32, (i32, u32), QMono);

/// The halved degree of a monomial in the $v_i$.
fn qdeg(k: &QMono) -> i32 {
    k.iter()
        .enumerate()
        .map(|(i, &e)| e as i32 * v_degree(i))
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

/// The untwisted resolution of $\mathbb{F}_2$ over $A/\tau$, as an explicit free complex, and its
/// lift to $A_\mathbb{C}$ modulo $\tau^2$.
///
/// $M_h$ is free on the $z \otimes \gamma_K$ with $z \in C_a$, $|K| = h - a$, in motivic degree
/// $2|z| + 2d(K) + |K|$. The differential is
/// $$D(z \otimes \gamma) = \sum_i Q_i (z \otimes \gamma / y_i) + \sum_{P(R) x \in dz}
/// \sum_{R' + R'' = R} P(R')\, (x \otimes \chi(P(R''))\, \gamma),$$
/// with $P$ acting on $\Gamma = Q^\vee$ as the transpose of the coefficient of $\xi^S$ in the
/// coaction. Read in $A_\mathbb{C}$, $D^2 = \tau e$, and the correction $c$ with $Dc = e$ makes
/// $D + \tau c$ a complex modulo $\tau^2$. The constant terms of $D$ and $c$ are $\delta_0$ and
/// $\delta_1$ on $C^\vee \otimes Q$.
pub struct TauLift {
    /// $M_h$.
    modules: Vec<Arc<FreeModule<MilnorAlgebra>>>,
    /// `d[h]`: $M_h \to M_{h-1}$, for $h \geq 1$.
    d: Vec<Arc<FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>>>,
    /// The generators of $M_h$ by degree.
    pub gens: Vec<BTreeMap<i32, Vec<Gen>>>,
    /// The correction on each generator of $M_h$, an element of $M_{h-1}$.
    correction: Vec<FxHashMap<(i32, usize), FpVector>>,
    /// The largest motivic degree.
    max_t: i32,
}

impl TauLift {
    /// Builds $M_0, \dots, M_{\text{max\_h}}$ through motivic degree `max_t` from the classical
    /// resolution `res`, and lifts the differential.
    pub fn new<CC>(res: &CC, max_h: i32, max_t: i32) -> Self
    where
        CC: FreeChainComplex<Algebra = MilnorAlgebra>,
    {
        assert!(res.has_computed_bidegree(Bidegree::s_t(max_h, max_t / 2)));
        let algebra = Arc::new(MilnorAlgebra::motivic_mod_tau(Default::default(), false));
        algebra.compute_basis(max_t);
        let backbone = res.algebra();

        // Q monomials by (length, halved degree), with v_0 explicit.
        let max_d = max_t / 2;
        let mut q_by: FxHashMap<(u32, i32), Vec<QMono>> = FxHashMap::default();
        for (d, bucket) in monomials(max_h as u32, |_| max_d).iter().enumerate() {
            for m in bucket {
                let len: u32 = m.iter().sum();
                for j in len..=max_h as u32 {
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

        // P(S) acting on Γ: γ_K ↦ Σ γ_{K'} over the K' whose coaction has ξ^S ⊗ v^K.
        let q_down = QDown::new(max_h as u32, |_| max_d, Convention::Xi);
        let memo = ActMemo::default();
        let mut up_cache: FxHashMap<(PackedXi, QMono), Vec<QMono>> = FxHashMap::default();
        let mut up = |s: PackedXi, k: QMono| -> Vec<QMono> {
            if s == 0 {
                return vec![k];
            }
            up_cache
                .entry((s, k))
                .or_insert_with(|| {
                    let d2 = qdeg(&k) + packed_halved_degree(s);
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

        let mut modules: Vec<Arc<FreeModule<MilnorAlgebra>>> = Vec::new();
        let mut d: Vec<Arc<FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>>> = Vec::new();
        let mut gens = Vec::new();
        let mut index = Vec::new();
        for h in 0..=max_h {
            let module = Arc::new(FreeModule::new(Arc::clone(&algebra), format!("M{h}"), 0));
            let mut by_t: BTreeMap<i32, Vec<Gen>> = BTreeMap::new();
            for a in 0..=h {
                let j = (h - a) as u32;
                for t_z in 0..=max_t / 2 {
                    for z in 0..res.number_of_gens_in_bidegree(Bidegree::s_t(a, t_z)) as u32 {
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
            let mut idx = FxHashMap::default();
            for t in 0..=max_t {
                module.compute_basis(t);
                let list = by_t.entry(t).or_default();
                list.sort();
                for (i, g) in list.iter().enumerate() {
                    idx.insert(*g, (t, i));
                }
                module.add_generators(t, list.len(), None);
            }
            gens.push(by_t);
            index.push(idx);
            modules.push(module);
        }
        for h in 1..=max_h as usize {
            let hom = Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&modules[h]),
                Arc::clone(&modules[h - 1]),
                0,
            ));
            let target = &modules[h - 1];
            for t in 0..=max_t {
                target.compute_basis(t);
                let rows = gens[h][&t]
                    .iter()
                    .map(|&(a, (t_z, z), k)| {
                        let mut v = FpVector::new(TWO, target.dimension(t));
                        // The Koszul part.
                        for i in 0..QDOWN_VARS {
                            if k[i] == 0 {
                                continue;
                            }
                            let mut k2 = k;
                            k2[i] -= 1;
                            let (t2, i2) = index[h - 1][&(a, (t_z, z), k2)];
                            let op = q_elt(&algebra, i);
                            let op_idx = algebra.basis_element_to_index(&op);
                            v.add_basis_element(
                                target.operation_generator_to_index(op.degree, op_idx, t2, i2),
                                1,
                            );
                        }
                        // The backbone part.
                        if a > 0 {
                            let dz = res.differential(a);
                            let src = res.module(a - 1);
                            for (e, _) in dz.output(t_z, z as usize).iter_nonzero() {
                                let og = src.index_to_op_gen(t_z, e);
                                let r: Vec<u32> = backbone
                                    .basis_element_from_index(
                                        og.operation_degree,
                                        og.operation_index,
                                    )
                                    .p_part
                                    .iter()
                                    .collect();
                                let x = (og.generator_degree, og.generator_index as u32);
                                for r1 in splits(pack(&r)) {
                                    let r2 = pack(&r) - r1;
                                    let op = p_elt(&algebra, r1);
                                    let op_idx = algebra.basis_element_to_index(&op);
                                    for &s in chi.get(&r2).into_iter().flatten() {
                                        for k2 in up(s, k) {
                                            let (t2, i2) = index[h - 1][&(a - 1, x, k2)];
                                            v.add_basis_element(
                                                target.operation_generator_to_index(
                                                    op.degree, op_idx, t2, i2,
                                                ),
                                                1,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        v
                    })
                    .collect();
                hom.add_generators_from_rows(t, rows);
            }
            d.push(hom);
        }
        // d[h] is d.get(h - 1); shift so that d[h] indexes directly.
        let mut dd = vec![d[0].clone()];
        dd.extend(d);

        let mut lift = Self {
            modules,
            d: dd,
            gens,
            correction: vec![FxHashMap::default(); max_h as usize + 1],
            max_t,
        };
        lift.correct();
        lift
    }

    /// Asserts $D^2 = 0$ over $A/\tau$.
    pub fn check_square_zero(&self) {
        for h in 2..self.modules.len() {
            for t in 0..=self.max_t {
                for (i, _) in self.gens[h][&t].iter().enumerate() {
                    let v = self.d[h].output(t, i);
                    let mut w = FpVector::new(TWO, self.modules[h - 2].dimension(t));
                    self.d[h - 1].apply(w.as_slice_mut(), 1, t, v.as_slice());
                    assert!(w.is_zero(), "D^2 != 0 on M_{h} at t = {t}, generator {i}");
                }
            }
        }
    }

    /// Solves $D c = e$ for every generator, $h$ ascending.
    fn correct(&mut self) {
        let engine = MotivicMilnorAlgebra::new();
        engine.compute_basis(self.max_t);
        for h in 2..self.modules.len() {
            let (mid, low) = (&self.modules[h - 1], &self.modules[h - 2]);
            for t in 0..=self.max_t {
                self.d[h - 1].compute_auxiliary_data_through_degree(t);
                for i in 0..self.gens[h][&t].len() {
                    let mut e = FpVector::new(TWO, low.dimension(t));
                    for (idx1, _) in self.d[h].output(t, i).iter_nonzero() {
                        let og1 = mid.index_to_op_gen(t, idx1);
                        let (t1, g1) = (og1.generator_degree, og1.generator_index);
                        // The τ-part of the A_C products along D(D(g)).
                        for (idx2, _) in self.d[h - 1].output(t1, g1).iter_nonzero() {
                            let og2 = low.index_to_op_gen(t1, idx2);
                            let (o1, i1, o2, i2) = (
                                og1.operation_degree,
                                og1.operation_index,
                                og2.operation_degree,
                                og2.operation_index,
                            );
                            // The engine orders its factors oppositely to the Milnor algebra.
                            for idx in engine.product_indexed(o2, i2, o1, i1) {
                                if engine.tau_exponent(o2, i2, o1, i1, idx) == 1 {
                                    e.add_basis_element(
                                        low.operation_generator_to_index(
                                            o1 + o2,
                                            idx,
                                            og2.generator_degree,
                                            og2.generator_index,
                                        ),
                                        1,
                                    );
                                }
                            }
                        }
                        // The coefficient times the correction already on the target.
                        if let Some(c) = self.correction[h - 1].get(&(t1, g1)) {
                            low.act(
                                e.as_slice_mut(),
                                1,
                                og1.operation_degree,
                                og1.operation_index,
                                t1,
                                c.as_slice(),
                            );
                        }
                    }
                    let mut c = FpVector::new(TWO, mid.dimension(t));
                    if !e.is_zero() {
                        let qi = self.d[h - 1].quasi_inverse(t).unwrap();
                        qi.apply(c.as_slice_mut(), 1, e.as_slice());
                        let mut check = FpVector::new(TWO, low.dimension(t));
                        self.d[h - 1].apply(check.as_slice_mut(), 1, t, c.as_slice());
                        assert_eq!(
                            check, e,
                            "the τ-error on M_{h} at t = {t} is not a boundary"
                        );
                    }
                    self.correction[h].insert((t, i), c);
                }
            }
        }
    }

    /// The generators of $M_{h-1}$ with a constant coefficient in `v`, an element of $M_{h-1}$
    /// in degree `t`.
    fn constants(&self, h: usize, t: i32, v: &FpVector) -> Vec<usize> {
        v.iter_nonzero()
            .filter_map(|(idx, _)| {
                let og = self.modules[h - 1].index_to_op_gen(t, idx);
                (og.operation_degree == 0).then_some(og.generator_index)
            })
            .collect()
    }

    /// How often $\delta_1$ links a generator of filtration $a$ and length $k$ to one of
    /// filtration $a - \Delta a$ and length $k + \Delta k$: counts by $(\Delta a, \Delta k)$.
    pub fn delta1_shapes(&self) -> BTreeMap<(i32, i32), usize> {
        let mut out = BTreeMap::new();
        for h in 2..self.modules.len() {
            for (&(t, i), c) in &self.correction[h] {
                let g = self.gens[h][&t][i];
                for j in self.constants(h, t, c) {
                    let g2 = self.gens[h - 1][&t][j];
                    *out.entry((g.0 - g2.0, qmono_len(&g2.2) as i32 - qmono_len(&g.2) as i32))
                        .or_default() += 1;
                }
            }
        }
        out
    }

    /// The backbone pairs $(z, x)$ linked by $\delta_1$, and how many of them are joined by a
    /// path of three backbone differentials $z \to y \to w \to x$ with nonzero coefficients.
    pub fn delta1_locality<CC>(&self, res: &CC) -> (usize, usize)
    where
        CC: FreeChainComplex<Algebra = MilnorAlgebra>,
    {
        // Backbone generators each generator's differential reaches.
        let reach = |a: i32, z: (i32, u32)| -> Vec<(i32, u32)> {
            let src = res.module(a - 1);
            let mut out: Vec<(i32, u32)> = res
                .differential(a)
                .output(z.0, z.1 as usize)
                .iter_nonzero()
                .map(|(e, _)| {
                    let og = src.index_to_op_gen(z.0, e);
                    (og.generator_degree, og.generator_index as u32)
                })
                .collect();
            out.sort();
            out.dedup();
            out
        };
        let mut pairs = std::collections::BTreeSet::new();
        for h in 2..self.modules.len() {
            for (&(t, i), c) in &self.correction[h] {
                let g = self.gens[h][&t][i];
                for j in self.constants(h, t, c) {
                    let g2 = self.gens[h - 1][&t][j];
                    pairs.insert((g.0, g.1, g2.1));
                }
            }
        }
        let local = pairs
            .iter()
            .filter(|&&(a, z, x)| {
                reach(a, z).into_iter().any(|y| {
                    reach(a - 1, y)
                        .into_iter()
                        .any(|w| reach(a - 2, w).contains(&x))
                })
            })
            .count();
        (pairs.len(), local)
    }

    /// The motivic weight of a generator.
    fn weight(g: &Gen) -> i32 {
        g.1.0 + qdeg(&g.2)
    }

    /// $\dim E_1$ and the rank of $d_1$ out of it, by (stem, $s$, weight) of the source, for
    /// $s$ below the top homological degree; and the weight shift of $d_1$.
    pub fn pages(&self) -> (BTreeMap<(i32, i32, i32), (usize, usize)>, Option<i32>) {
        let top = self.modules.len() - 1;
        let mut shift = None;
        let mut out = BTreeMap::new();
        for t in 0..=self.max_t {
            // g'^* ↦ the g with a constant g' in D(g) (δ_0) or in c(g) (δ_1), per M_{h+1}.
            let cob = |h: usize, which: u8| -> FxHashMap<usize, Vec<usize>> {
                let mut m: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
                for i in 0..self.gens[h][&t].len() {
                    let v = if which == 0 {
                        self.d[h].output(t, i)
                    } else {
                        &self.correction[h][&(t, i)]
                    };
                    for g2 in self.constants(h, t, v) {
                        m.entry(g2).or_default().push(i);
                    }
                }
                m
            };
            let d0: Vec<_> = (0..=top)
                .map(|h| {
                    if h == 0 {
                        FxHashMap::default()
                    } else {
                        cob(h, 0)
                    }
                })
                .collect();
            let d1: Vec<_> = (0..=top)
                .map(|h| {
                    if h < 2 {
                        FxHashMap::default()
                    } else {
                        cob(h, 1)
                    }
                })
                .collect();
            let n = |h: usize| self.gens[h][&t].len();
            let vec_of = |h: usize, gs: &mut dyn Iterator<Item = usize>| {
                let mut v = FpVector::new(TWO, n(h));
                for g in gs {
                    v.add_basis_element(g, 1);
                }
                v
            };
            for h in 0..top {
                let weights: std::collections::BTreeSet<i32> =
                    self.gens[h][&t].iter().map(Self::weight).collect();
                // Boundaries in M_{h+1}, all weights.
                let mut bound = Echelon::default();
                for g in 0..n(h) {
                    bound.insert(vec_of(
                        h + 1,
                        &mut d0[h + 1].get(&g).into_iter().flatten().copied(),
                    ));
                }
                let b_rank = bound.rows.len();
                for w in weights {
                    let here: Vec<usize> = (0..n(h))
                        .filter(|&g| Self::weight(&self.gens[h][&t][g]) == w)
                        .collect();
                    let reps = cohomology(
                        &here,
                        |g| d0[h + 1].get(&g).cloned().unwrap_or_default(),
                        n(h + 1),
                        if h == 0 {
                            Vec::new()
                        } else {
                            (0..n(h - 1))
                                .map(|g| d0[h].get(&g).cloned().unwrap_or_default())
                                .collect()
                        },
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
                                let ws = Self::weight(&self.gens[h + 1][&t][g2]) - w;
                                assert!(
                                    shift.is_none_or(|s| s == ws),
                                    "d_1 shifts weight by both {shift:?} and {ws}"
                                );
                                shift = Some(ws);
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
        (out, shift)
    }
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

/// Representatives of $\ker \delta_0 / \operatorname{im} \delta_0$ on the cochains `here` of a
/// space of dimension `dim`, with `out(g)` the cochains $\delta_0 g^*$ reaches in a space of
/// dimension `next_dim`, and `incoming` the images of the cochains one degree down.
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

/// The halved degree of a packed $P(R)$.
fn packed_halved_degree(r: PackedXi) -> i32 {
    unpack(r)
        .iter()
        .enumerate()
        .map(|(j, &e)| e as i32 * v_degree(j + 1))
        .sum()
}

/// $P(R)$ in $A/\tau$.
fn p_elt(algebra: &MilnorAlgebra, r: PackedXi) -> MilnorBasisElement {
    let p: Vec<_> = unpack(r).into_iter().map(|x| x as _).collect();
    let mut elt = MilnorBasisElement {
        q_part: 0,
        p_part: PPart::from_slice(&p),
        degree: 0,
    };
    algebra.compute_degree(&mut elt);
    elt
}

/// $Q_i$ in $A/\tau$.
fn q_elt(algebra: &MilnorAlgebra, i: usize) -> MilnorBasisElement {
    let mut elt = MilnorBasisElement {
        q_part: 1 << i,
        p_part: PPart::from_slice(&[]),
        degree: 0,
    };
    algebra.compute_degree(&mut elt);
    elt
}
