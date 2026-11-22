//! The $C\tau$ Adams $E_2$ page, with products and cofibers, from a resolution of the classical sphere.

use std::sync::Arc;

use algebra::{
    Algebra, MilnorAlgebra,
    milnor_algebra::{MilnorBasisElement, PPart},
    module::{Module, homomorphism::ModuleHomomorphism},
};
use dashmap::DashMap;
use fp::{
    matrix::{AugmentedMatrix, Matrix, QuasiInverse, Subquotient, Subspace},
    prime::TWO,
    vector::{FpSlice, FpVector},
};
use rustc_hash::{FxHashMap, FxHashSet};
use sseq::coordinates::Bidegree;

use crate::chain_complex::FreeChainComplex;

/// The number of polynomial generators $q_1, q_2, \dots$ a [`QMono`] holds.
///
/// It bounds the weights [`QAlgebra::new`] accepts: the first generator left out must not fit.
pub const Q_VARS: usize = 9;

/// A monomial in $q_1, q_2, \dots$ as [`Q_VARS`] exponents; the power of $q_0$ is implicit.
///
/// $Q = \mathbb{F}_2[q_0, q_1, \dots]$ is graded by polynomial degree $k$, and $m \mapsto
/// q_0^{k - |m|} m$ identifies the $q_0$-free monomials of length at most $k$ with a basis of
/// $Q_k$.
pub type QMono = [u16; Q_VARS];

/// A polynomial in the $q_i$ over $\mathbb{F}_2$: the monomials with coefficient 1.
pub type QPoly = Vec<QMono>;

/// A monomial $\xi^R$, equivalently the Milnor basis element $P(R)$, as [`XI_BITS`]-bit exponents
/// of $\xi_1, \xi_2, \dots$, so that multiplying monomials is adding.
pub type PackedXi = u128;

/// The bits per exponent of a [`PackedXi`]; [`QAlgebra::new`] checks that degrees stay in range.
pub const XI_BITS: u32 = 12;

/// The degree of $q_i$ in halved degrees, which is also that of $\xi_i$.
pub fn q_degree(i: usize) -> i32 {
    (1 << i) - 1
}

/// The degree of a monomial, in halved degrees.
pub fn mono_degree(m: &QMono) -> i32 {
    m.iter()
        .enumerate()
        .map(|(i, &e)| e as i32 * q_degree(i + 1))
        .sum()
}

/// The length of a monomial, the least $k$ with $m \in Q_k$.
pub fn mono_len(m: &QMono) -> u32 {
    m.iter().map(|&e| e as u32).sum()
}

/// The product of two monomials.
pub fn mono_mul(a: &QMono, b: &QMono) -> QMono {
    let mut r = *a;
    for (x, y) in r.iter_mut().zip(b) {
        *x += y;
    }
    r
}

/// Packs an exponent sequence.
pub fn pack(r: impl IntoIterator<Item = u32>) -> PackedXi {
    r.into_iter()
        .enumerate()
        .fold(0, |acc, (j, e)| acc | (e as u128) << (j as u32 * XI_BITS))
}

/// Unpacks into a trimmed exponent sequence.
pub fn unpack(mut r: PackedXi) -> Vec<u32> {
    let mut v = Vec::new();
    while r != 0 {
        v.push((r & ((1 << XI_BITS) - 1)) as u32);
        r >>= XI_BITS;
    }
    v
}

/// The degree of $\xi^R$.
pub fn xi_degree(r: PackedXi) -> i32 {
    unpack(r)
        .iter()
        .enumerate()
        .map(|(j, &e)| e as i32 * q_degree(j + 1))
        .sum()
}

/// Whether every exponent of `s` is at most that of `r`, so that $P(R - S)$ exists.
pub fn fits(s: PackedXi, r: PackedXi) -> bool {
    let mask = (1u128 << XI_BITS) - 1;
    (0..128 / XI_BITS).all(|j| (s >> (j * XI_BITS)) & mask <= (r >> (j * XI_BITS)) & mask)
}

/// Adds `x` to a set of terms with coefficient 1, over $\mathbb{F}_2$.
fn toggle<T: Eq + std::hash::Hash>(set: &mut FxHashSet<T>, x: T) {
    if !set.remove(&x) {
        set.insert(x);
    }
}

/// Adds `x` to a polynomial.
fn poly_add(p: &mut FxHashSet<QMono>, x: impl IntoIterator<Item = QMono>) {
    for m in x {
        toggle(p, m);
    }
}

/// $\chi \xi_j$ for `j` in `0..=max` as sets of packed monomials, from $\sum_{i=0}^{j}
/// \xi_{j-i}^{2^i} \chi\xi_i = 0$.
fn conjugates(max: usize) -> Vec<FxHashSet<PackedXi>> {
    let mut chi: Vec<FxHashSet<PackedXi>> = vec![std::iter::once(0).collect()];
    for j in 1..=max {
        let mut c = FxHashSet::default();
        for (i, chi_i) in chi.iter().enumerate() {
            // ξ_{j-i}^{2^i}
            let x = 1u128 << ((j - i - 1) as u32 * XI_BITS + i as u32);
            for &m in chi_i {
                toggle(&mut c, m + x);
            }
        }
        chi.push(c);
    }
    chi
}

/// $Q = \mathbb{F}_2[q_0, q_1, \dots]$ as an algebra over the Steenrod algebra, in halved degrees.
///
/// $\mathrm{Ext}_{E(\tau_0, \tau_1, \dots)}(\mathbb{F}_2, \mathbb{F}_2) = Q$ with $q_i$ dual to
/// $\tau_i$, and the coaction of $P_* = \mathbb{F}_2[\xi_1, \xi_2, \dots]$ (which is $A_*$ with
/// degrees doubled) on it is $q_n \mapsto \sum_i \xi_{n-i}^{2^i} \otimes q_i$. The left action of
/// $A$ this module holds is the contragredient one,
/// $P(R) \cdot m = \sum \langle P(R), \chi c \rangle m'$ over the terms $c \otimes m'$ of the
/// coaction on $m$. It lowers degree, never lengthens a monomial, and satisfies the Cartan formula.
pub struct QAlgebra {
    /// The $q_0$-free monomials of each degree, sorted by length and then exponents.
    by_degree: Vec<Vec<QMono>>,
    /// `factors[n]` lists the terms $(\xi^S, i)$ of $q_n \mapsto \sum \chi(\xi_{n-i}^{2^i})
    /// \otimes q_i$.
    factors: Vec<Vec<(PackedXi, usize)>>,
    /// The position of each monomial in its degree.
    index: FxHashMap<QMono, u32>,
    /// `shorter[d][l]` is the number of monomials of degree `d` shorter than `l`.
    shorter: Vec<Vec<u32>>,
    /// The coaction on each power of a generator asked for; see [`Self::power_coaction`].
    powers: DashMap<(usize, u16), Arc<Vec<(PackedXi, QMono)>>>,
    /// The memo of [`Self::coaction`].
    memo: DashMap<QMono, Arc<Coaction>>,
    /// The number of terms at which the memo is emptied; see [`Self::set_memo_limit`].
    memo_limit: usize,
    /// The terms put into the memo since it was last emptied.
    memo_count: std::sync::atomic::AtomicUsize,
}

/// The coaction on one monomial $m$: the terms $\xi^R \otimes m'$ with coefficient 1, with the
/// coefficients already conjugated, grouped by $R$.
pub struct Coaction {
    /// Each $R$ that occurs with the start of its range of `gammas`, sorted, and then a sentinel
    /// holding the length of `gammas`.
    index: Vec<(PackedXi, u32)>,
    /// Each $m'$ as its position among the monomials of its degree, and its length.
    gammas: Vec<(u32, u16)>,
}

impl Coaction {
    /// $P(R) \cdot m$, as the position of each monomial in its degree, and its length.
    pub fn get(&self, r: PackedXi) -> &[(u32, u16)] {
        let ops = &self.index[..self.index.len() - 1];
        match ops.binary_search_by_key(&r, |x| x.0) {
            Ok(i) => &self.gammas[self.index[i].1 as usize..self.index[i + 1].1 as usize],
            Err(_) => &[],
        }
    }

    /// The number of terms.
    pub fn len(&self) -> usize {
        self.gammas.len()
    }

    /// The number of distinct $R$.
    pub fn num_ops(&self) -> usize {
        self.index.len() - 1
    }

    /// Each $R$ that occurs with $P(R) \cdot m$, as in [`Self::get`].
    pub fn iter(&self) -> impl Iterator<Item = (PackedXi, &[(u32, u16)])> {
        self.index
            .windows(2)
            .map(|w| (w[0].0, &self.gammas[w[0].1 as usize..w[1].1 as usize]))
    }

    /// Whether there are no terms; there is always $1 \otimes m$.
    pub fn is_empty(&self) -> bool {
        self.gammas.is_empty()
    }
}

impl QAlgebra {
    /// The monomials of degree at most `max_degree` and length at most `max_len`.
    pub fn new(max_degree: i32, max_len: u32) -> Self {
        Self::quotient(max_degree, max_len, 0)
    }

    /// [`Self::new`] for the quotient $Q / (q_0, \dots, q_{n-1})$, for `n > 0`: the monomials
    /// in $q_n, q_{n+1}, \dots$, and the coaction with every term in the ideal dropped.
    ///
    /// The ideal is invariant, so this is again an algebra over the Steenrod algebra; it is
    /// the associated graded of $BP_*/I_n$. Since $q_0$ is killed nothing shortens a monomial:
    /// each length is its own module.
    pub fn quotient(max_degree: i32, max_len: u32, n: usize) -> Self {
        assert!(
            max_degree < q_degree(Q_VARS + 1) && max_degree < 1 << XI_BITS,
            "degree {max_degree} needs more than q_{Q_VARS}"
        );
        let mut by_degree = vec![Vec::new(); max_degree.max(0) as usize + 1];
        /// Chooses the exponent of `q_{i+1}`, then recurses downwards.
        #[allow(clippy::too_many_arguments)]
        fn rec(
            i: usize,
            remaining: u32,
            degree: i32,
            max_degree: i32,
            first: usize,
            current: &mut QMono,
            out: &mut Vec<Vec<QMono>>,
        ) {
            let d = q_degree(i + 1);
            let mut e = 0;
            // q_{i+1} is killed when it comes before q_first
            let most = if i + 1 < first { 0 } else { remaining };
            while e <= most && degree + e as i32 * d <= max_degree {
                current[i] = e as u16;
                if i == 0 {
                    out[(degree + e as i32 * d) as usize].push(*current);
                } else {
                    rec(
                        i - 1,
                        remaining - e,
                        degree + e as i32 * d,
                        max_degree,
                        first,
                        current,
                        out,
                    );
                }
                e += 1;
            }
            current[i] = 0;
        }
        rec(
            Q_VARS - 1,
            max_len,
            0,
            max_degree,
            n,
            &mut [0; Q_VARS],
            &mut by_degree,
        );
        for bucket in &mut by_degree {
            bucket.sort_by_key(|m| (mono_len(m), *m));
        }
        let chi = conjugates(Q_VARS);
        let factors = (0..=Q_VARS)
            .map(|m| {
                (0..=m)
                    .filter(|&i| n == 0 || i >= n)
                    .flat_map(|i| chi[m - i].iter().map(move |&x| (x << i, i)))
                    .collect()
            })
            .collect();
        let index = by_degree
            .iter()
            .flat_map(|b| b.iter().enumerate().map(|(i, &m)| (m, i as u32)))
            .collect();
        let shorter = by_degree
            .iter()
            .map(|b| {
                let top = b.last().map_or(0, mono_len);
                (0..=top + 1)
                    .map(|l| b.partition_point(|m| mono_len(m) < l) as u32)
                    .collect()
            })
            .collect();
        Self {
            by_degree,
            factors,
            index,
            shorter,
            powers: DashMap::default(),
            memo: DashMap::default(),
            memo_limit: usize::MAX,
            memo_count: Default::default(),
        }
    }

    /// Bounds the memo of [`Self::coaction`]: it is emptied when it passes `limit` terms, trading
    /// recomputation for memory.
    pub fn set_memo_limit(&mut self, limit: usize) {
        self.memo_limit = limit;
    }

    /// The position of `m` among the monomials of its degree.
    pub fn index(&self, m: &QMono) -> usize {
        self.index[m] as usize
    }

    /// The number of monomials of degree `d` of length less than `l`.
    pub fn shorter(&self, d: i32, l: u32) -> usize {
        let row = &self.shorter[d as usize];
        row[(l as usize).min(row.len() - 1)] as usize
    }

    /// The largest length of a monomial of degree `d`, if there is one.
    pub fn max_len(&self, d: i32) -> Option<u32> {
        self.monomials(d).last().map(mono_len)
    }

    /// The monomials of degree `d`, sorted by length; empty outside the range.
    pub fn monomials(&self, d: i32) -> &[QMono] {
        usize::try_from(d)
            .ok()
            .and_then(|d| self.by_degree.get(d))
            .map_or(&[], |b| b)
    }

    /// The length of the monomial at `idx` in degree `d`.
    pub fn len_at(&self, d: i32, idx: u32) -> u16 {
        mono_len(&self.by_degree[d as usize][idx as usize]) as u16
    }

    /// The coaction on $q_n^e$ as terms $(\xi^R, m')$: the product over the binary digits of
    /// `e` of Frobenius twists of the coaction on $q_n$. Kept, since few `(n, e)` occur.
    fn power_coaction(&self, n: usize, e: u16) -> Arc<Vec<(PackedXi, QMono)>> {
        if let Some(hit) = self.powers.get(&(n, e)) {
            return Arc::clone(&hit);
        }
        let mut acc: FxHashSet<(PackedXi, QMono)> = std::iter::once((0, [0; Q_VARS])).collect();
        for j in 0..16 {
            if e >> j & 1 == 0 {
                continue;
            }
            let mut next = FxHashSet::default();
            for &(r, m) in &acc {
                for &(s, i) in &self.factors[n + 1] {
                    let mut m = m;
                    if i > 0 {
                        m[i - 1] += 1 << j;
                    }
                    toggle(&mut next, (r + (s << j), m));
                }
            }
            acc = next;
        }
        let out = Arc::new(acc.into_iter().collect::<Vec<_>>());
        self.powers.insert((n, e), Arc::clone(&out));
        out
    }

    /// The terms $(\xi^R, m')$ of the coaction on `m` with `keep(R)`, computed from scratch and
    /// not kept: the product of the coactions on the powers of each generator, filtered as the
    /// last factor is multiplied in.
    ///
    /// Unlike [`Self::coaction`] this holds nothing between calls, which is what tabulating
    /// every monomial of a degree needs.
    pub fn coaction_filtered(
        &self,
        m: &QMono,
        keep: impl Fn(PackedXi) -> bool,
    ) -> Vec<(PackedXi, QMono)> {
        let mut factors: Vec<Arc<Vec<(PackedXi, QMono)>>> = m
            .iter()
            .enumerate()
            .filter(|x| *x.1 > 0)
            .map(|(n, &e)| self.power_coaction(n, e))
            .collect();
        factors.sort_by_key(|f| f.len());
        let Some(last) = factors.pop() else {
            return if keep(0) { vec![(0, *m)] } else { Vec::new() };
        };
        let mut acc: Vec<(PackedXi, QMono)> = vec![(0, [0; Q_VARS])];
        for f in &factors {
            let mut next: FxHashSet<(PackedXi, QMono)> = FxHashSet::default();
            for &(r, x) in &acc {
                for &(s, y) in f.iter() {
                    toggle(&mut next, (r + s, mono_mul(&x, &y)));
                }
            }
            acc = next.into_iter().collect();
        }
        let mut out: FxHashSet<(PackedXi, QMono)> = FxHashSet::default();
        for &(r, x) in &acc {
            for &(s, y) in last.iter() {
                if keep(r + s) {
                    toggle(&mut out, (r + s, mono_mul(&x, &y)));
                }
            }
        }
        out.into_iter().collect()
    }

    /// The whole coaction on `m`, from which [`Self::act`] reads $P(R) \cdot m$ for any $R$.
    ///
    /// Peels the largest binary chunk $q_n^{2^j}$ off $m$ and uses that the coaction is
    /// multiplicative, so the coaction on what is left is shared between monomials.
    pub fn coaction(&self, m: &QMono) -> Arc<Coaction> {
        if let Some(hit) = self.memo.get(m) {
            return Arc::clone(&hit);
        }
        let mut terms: Vec<(PackedXi, QMono)> = match m.iter().rposition(|&e| e > 0) {
            None => vec![(0, *m)],
            Some(n) => {
                let j = 15 - m[n].leading_zeros();
                let mut rest = *m;
                rest[n] -= 1 << j;
                let rest_degree = mono_degree(&rest);
                let rest = self.coaction(&rest);
                let mut set: FxHashSet<(PackedXi, QMono)> = FxHashSet::default();
                for (r, gammas) in rest.iter() {
                    let bucket = &self.by_degree[(rest_degree - xi_degree(r)) as usize];
                    for &(idx, _) in gammas {
                        for &(s, i) in &self.factors[n + 1] {
                            let mut gamma = bucket[idx as usize];
                            if i > 0 {
                                gamma[i - 1] += 1 << j;
                            }
                            toggle(&mut set, (r + (s << j), gamma));
                        }
                    }
                }
                set.into_iter().collect()
            }
        };
        terms.sort_unstable();
        let mut index = Vec::new();
        let mut lo = 0u32;
        for run in terms.chunk_by(|x, y| x.0 == y.0) {
            index.push((run[0].0, lo));
            lo += run.len() as u32;
        }
        index.push((PackedXi::MAX, lo));
        let out = Arc::new(Coaction {
            index,
            gammas: terms
                .into_iter()
                .map(|x| (self.index[&x.1], mono_len(&x.1) as u16))
                .collect(),
        });
        if self
            .memo_count
            .fetch_add(out.gammas.len(), std::sync::atomic::Ordering::Relaxed)
            > self.memo_limit
        {
            self.memo_count
                .store(0, std::sync::atomic::Ordering::Relaxed);
            self.memo.clear();
        }
        self.memo.insert(*m, Arc::clone(&out));
        out
    }

    /// $P(R) \cdot m$.
    pub fn act(&self, r: PackedXi, m: &QMono) -> QPoly {
        let psi = self.coaction(m);
        let terms = psi.get(r);
        if terms.is_empty() {
            return Vec::new();
        }
        let bucket = &self.by_degree[(mono_degree(m) - xi_degree(r)) as usize];
        terms.iter().map(|&(idx, _)| bucket[idx as usize]).collect()
    }

    /// $P(R) \cdot f$.
    pub fn act_poly(&self, r: PackedXi, f: &[QMono]) -> QPoly {
        if r == 0 {
            return f.to_vec();
        }
        let mut out = FxHashSet::default();
        for m in f {
            poly_add(&mut out, self.act(r, m).iter().copied());
        }
        out.into_iter().collect()
    }

    /// Forgets the memo of [`Self::coaction`].
    pub fn clear_memo(&self) {
        self.memo.clear();
    }
}

/// A term $P(R) x_i$ of an element of a free module, for the generator $x_i$ of index `i` in
/// degree `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Term {
    pub op: PackedXi,
    pub t: i32,
    pub i: u32,
}

/// Translates between Milnor basis elements and [`PackedXi`].
pub struct OpTable {
    algebra: Arc<MilnorAlgebra>,
    packed: DashMap<(i32, usize), PackedXi>,
}

impl OpTable {
    /// An empty table for `algebra`, which must be the Milnor basis at the prime 2.
    pub fn new(algebra: Arc<MilnorAlgebra>) -> Self {
        Self {
            algebra,
            packed: DashMap::default(),
        }
    }

    /// The exponents of the basis element `idx` of degree `degree`.
    pub fn pack(&self, degree: i32, idx: usize) -> PackedXi {
        // nearly every call finds its answer, and many threads ask at once: read first
        if let Some(packed) = self.packed.get(&(degree, idx)) {
            return *packed;
        }
        *self.packed.entry((degree, idx)).or_insert_with(|| {
            pack(
                self.algebra
                    .basis_element_from_index(degree, idx)
                    .p_part
                    .iter(),
            )
        })
    }

    /// The index of $P(R)$ in its degree.
    pub fn index(&self, r: PackedXi) -> (i32, usize) {
        let degree = xi_degree(r);
        self.algebra.compute_basis(degree);
        let idx = self.algebra.basis_element_to_index(&MilnorBasisElement {
            q_part: 0,
            p_part: PPart::from_slice(&unpack(r)),
            degree,
        });
        (degree, idx)
    }

    /// The terms of the element `v` of degree `degree` of the free module `module`.
    pub fn terms(
        &self,
        module: &algebra::module::FreeModule<MilnorAlgebra>,
        degree: i32,
        v: FpSlice,
    ) -> Vec<Term> {
        v.iter_nonzero()
            .map(|(idx, _)| {
                let og = module.index_to_op_gen(degree, idx);
                Term {
                    op: self.pack(og.operation_degree, og.operation_index),
                    t: og.generator_degree,
                    i: og.generator_index as u32,
                }
            })
            .collect()
    }
}

/// The part of a minimal resolution $P_\bullet \to \mathbb{F}_2$ over $A$ that the $C\tau$
/// computation reads: the generators and the differentials on them, through a range.
pub struct Backbone {
    /// `gens[a][t]` is the number of generators of $P_a$ in degree `t`.
    gens: Vec<Vec<usize>>,
    /// `d[a][t][l]` lists the terms of $d z_l$ for the generator $z_l$ of $P_a$ in degree `t`.
    d: Vec<Vec<Vec<Vec<Term>>>>,
}

impl Backbone {
    /// Reads $P_a$ for `a <= max_a` in degrees up to `max_t(a)` off `res`, which must hold them.
    pub fn new<CC>(res: &CC, ops: &OpTable, max_a: i32, max_t: impl Fn(i32) -> i32 + Sync) -> Self
    where
        CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
    {
        use maybe_rayon::prelude::*;
        let per_a: Vec<(Vec<usize>, Vec<Vec<Vec<Term>>>)> = (0..=max_a)
            .into_maybe_par_iter()
            .map(|a| {
                let mut gens = Vec::new();
                let mut d = Vec::new();
                for t in 0..=max_t(a) {
                    let b = Bidegree::s_t(a, t);
                    let n = if t >= a {
                        assert!(res.has_computed_bidegree(b), "the backbone lacks {b}");
                        res.number_of_gens_in_bidegree(b)
                    } else {
                        0
                    };
                    gens.push(n);
                    d.push(if a == 0 {
                        vec![Vec::new(); n]
                    } else {
                        let diff = res.differential(a);
                        let target = res.module(a - 1);
                        (0..n)
                            .map(|l| ops.terms(&target, t, diff.output(t, l).as_slice()))
                            .collect()
                    });
                }
                (gens, d)
            })
            .collect();
        let (gens, d) = per_a.into_iter().unzip();
        Self { gens, d }
    }

    /// The number of generators of $P_a$ in degree `t`, zero outside the range read.
    pub fn gens(&self, a: i32, t: i32) -> usize {
        if a < 0 || t < 0 {
            return 0;
        }
        self.gens
            .get(a as usize)
            .and_then(|g| g.get(t as usize))
            .map_or(0, |&n| n)
    }

    /// The terms of $d z_l$.
    pub fn d(&self, a: i32, t: i32, l: usize) -> &[Term] {
        &self.d[a as usize][t as usize][l]
    }

    /// The largest filtration read.
    pub fn max_a(&self) -> i32 {
        self.gens.len() as i32 - 1
    }
}

/// A finite complex smashed with $C\tau$, given by the cells of the classical complex it doubles.
///
/// The cochains are the $A$-linear maps $P_a \to Q \otimes DM$, where the dual $DM$ of the
/// classical cohomology $M$ has a basis of cells $y_c$ and $A$ acts diagonally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cells {
    /// The halved degree of each cell.
    pub degrees: Vec<i32>,
    /// $(R, c, c')$ for each nonzero $P(R) \cdot y_c \ni y_{c'}$ with $R \neq 0$.
    pub actions: Vec<(PackedXi, usize, usize)>,
    /// Whether to kill $q_0$, which smashes with the mod 2 Moore spectrum.
    pub mod_q0: bool,
}

impl Cells {
    /// $C\tau$ itself.
    pub fn sphere() -> Self {
        Self {
            degrees: vec![0],
            actions: Vec::new(),
            mod_q0: false,
        }
    }

    /// $C\tau \wedge S/2$; the attaching map $2$ is detected by $h_0 = q_0$.
    pub fn moore() -> Self {
        Self {
            mod_q0: true,
            ..Self::sphere()
        }
    }

    /// $C\tau \wedge C(h_{j+1})$: the cofiber of the map detected by the motivic $h_{j+1}$, which
    /// is the classical $h_j$ in doubled degrees, so that the top cell is hit by $Sq^{2^j}$.
    pub fn hopf(j: u32) -> Self {
        Self {
            degrees: vec![0, 1 << j],
            actions: vec![(1 << j, 1, 0)],
            mod_q0: false,
        }
    }
}

/// The cochains of filtration `a` and weight `u` of a complex: a basis $x_i \mapsto m \otimes y_c$
/// sorted by the length of $m$, so that those of $Q_k$ are a prefix.
pub struct Space {
    pub a: i32,
    pub u: i32,
    /// The basis as (cell, degree of $x_i$, index of $x_i$, monomial).
    pub basis: Vec<(u8, i32, u32, QMono)>,
    index: FxHashMap<(u8, i32, u32, QMono), u32>,
    /// `prefix[k]` is the number of basis elements of length at most `k`.
    prefix: Vec<usize>,
}

impl Space {
    /// The cochains of `cells` on `bb` with monomials of length at most `max_len`.
    fn new(bb: &Backbone, q: &QAlgebra, cells: &Cells, max_len: u32, a: i32, u: i32) -> Self {
        let mut basis = Vec::new();
        for (c, &e) in cells.degrees.iter().enumerate() {
            for t in 0..=(u - e) {
                for i in 0..bb.gens(a, t) {
                    for m in q.monomials(u - e - t) {
                        if mono_len(m) <= max_len {
                            basis.push((c as u8, t, i as u32, *m));
                        }
                    }
                }
            }
        }
        basis.sort_by_key(|x| (mono_len(&x.3), *x));
        let index = basis
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, i as u32))
            .collect();
        let prefix = (0..=max_len + 1)
            .map(|k| basis.partition_point(|x| mono_len(&x.3) <= k))
            .collect();
        Self {
            a,
            u,
            basis,
            index,
            prefix,
        }
    }

    /// The dimension, over every length held.
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    /// The number of basis elements of length at most `k`.
    pub fn prefix(&self, k: i32) -> usize {
        if k < 0 {
            0
        } else {
            self.prefix[(k as usize).min(self.prefix.len() - 1)]
        }
    }

    /// The position of a basis element.
    pub fn index(&self, c: usize, t: i32, i: u32, m: &QMono) -> Option<usize> {
        self.index.get(&(c as u8, t, i, *m)).map(|&x| x as usize)
    }
}

/// A tridegree of $\mathrm{Ext}$ over $A/\tau$ in the coordinates of the computation: Novikov
/// filtration `a`, weight `u` and polynomial degree `k`.
///
/// The motivic stem is $2u - a$, the Adams filtration $a + k$ and the weight $u$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tri {
    pub a: i32,
    pub u: i32,
    pub k: i32,
}

impl Tri {
    /// From the coordinates of the computation.
    pub fn new(a: i32, u: i32, k: i32) -> Self {
        Self { a, u, k }
    }

    /// From motivic (stem, Adams filtration, weight).
    pub fn from_motivic(n: i32, s: i32, w: i32) -> Self {
        let a = 2 * w - n;
        Self { a, u: w, k: s - a }
    }

    /// The motivic stem.
    pub fn stem(&self) -> i32 {
        2 * self.u - self.a
    }

    /// The Adams filtration.
    pub fn filtration(&self) -> i32 {
        self.a + self.k
    }

    /// The motivic weight.
    pub fn weight(&self) -> i32 {
        self.u
    }
}

impl std::ops::Add for Tri {
    type Output = Self;

    /// The tridegree of a product.
    fn add(self, o: Self) -> Self {
        Self::new(self.a + o.a, self.u + o.u, self.k + o.k)
    }
}

/// $\mathrm{Ext}_{A/\tau}$ of a complex through a range, as the cohomology of the untwisted
/// cochains on a [`Backbone`].
///
/// $\mathrm{Ext}_{A/\tau}(\mathbb{F}_2, \mathbb{F}_2) = \mathrm{Ext}_A(\mathbb{F}_2, Q)$ in halved
/// degrees, and a cochain is an $A$-linear map $c\colon P_a \to Q \otimes DM$, determined by its
/// values on generators, with $(\delta c)(z) = \sum \theta \cdot c(x)$ over the terms $\theta x$
/// of $dz$.
pub struct CtauExt {
    pub cells: Cells,
    pub max_n: i32,
    pub max_s: i32,
    spaces: FxHashMap<(i32, i32), Space>,
    delta: FxHashMap<(i32, i32), Matrix>,
    coh: FxHashMap<Tri, Subquotient>,
}

/// The matrix of $c \mapsto (z \mapsto \sum \theta' \cdot c(x))$ over the terms $\theta x$ of
/// `terms(z)` and the splittings $\theta \mapsto \theta' \otimes \theta''$ with $\theta'' = P(S)$
/// for `S = shift`, from `src` to `tgt`; rows are indexed by `src`.
///
/// With `terms` the differential and no shift this is the coboundary; with the differential and
/// $S = (2^j)$ it is the cup product with $h_j$; with a chain map it is the action of the class
/// the chain map lifts.
pub fn pullback<'a>(
    q: &QAlgebra,
    bb: &Backbone,
    cells: &Cells,
    src: &Space,
    tgt: &Space,
    terms: impl Fn(i32, usize) -> &'a [Term],
    shift: PackedXi,
) -> Matrix {
    let mut matrix = Matrix::new(TWO, src.dim(), tgt.dim());
    // each cell's diagonal terms: the identity and the cell actions
    let cell_moves: Vec<Vec<(PackedXi, usize)>> = (0..cells.degrees.len())
        .map(|c| {
            std::iter::once((shift, c))
                .chain(
                    cells
                        .actions
                        .iter()
                        .filter(|x| x.1 == c)
                        .map(|x| (shift + x.0, x.2)),
                )
                .collect()
        })
        .collect();
    for t_l in 0..=tgt.u {
        for l in 0..bb.gens(tgt.a, t_l) {
            for term in terms(t_l, l) {
                for (c, moves) in cell_moves.iter().enumerate() {
                    let e = src.u - cells.degrees[c] - term.t;
                    for &(s, c2) in moves {
                        if !fits(s, term.op) {
                            continue;
                        }
                        let r = term.op - s;
                        for m in q.monomials(e) {
                            let Some(row) = src.index(c, term.t, term.i, m) else {
                                continue;
                            };
                            for gamma in q.act(r, m).iter() {
                                let col = tgt
                                    .index(c2, t_l, l as u32, gamma)
                                    .expect("the action stays in the basis");
                                matrix.row_mut(row).add_basis_element(col, 1);
                            }
                        }
                    }
                }
            }
        }
    }
    matrix
}

impl CtauExt {
    /// The largest weight at filtration `a` in the range.
    fn max_u(&self, a: i32) -> i32 {
        (self.max_n + a) / 2
    }

    /// The largest filtration a [`Backbone`] must be read through for a range, and the largest
    /// degree at each filtration.
    ///
    /// The cochains of filtration `a` are read at weights up to that of stem `max_n` in
    /// filtration `a + 1`, so the generators lie in stems up to about `max_n / 2`.
    pub fn backbone_range(max_n: i32, max_s: i32) -> (i32, impl Fn(i32) -> i32 + Sync) {
        (max_s + 1, move |a| {
            ((max_n + a + 1) / 2).min((max_n + max_s) / 2)
        })
    }

    /// Computes $\mathrm{Ext}$ of `cells` in stems `0..=max_n` and Adams filtrations
    /// `0..=max_s`.
    pub fn new(q: &QAlgebra, bb: &Backbone, cells: Cells, max_n: i32, max_s: i32) -> Self {
        use maybe_rayon::prelude::*;
        let mut this = Self {
            cells,
            max_n,
            max_s,
            spaces: FxHashMap::default(),
            delta: FxHashMap::default(),
            coh: FxHashMap::default(),
        };
        let max_len = max_s as u32;
        let mut keys = Vec::new();
        for a in 0..=max_s + 1 {
            // the weights filtration a + 1 reads, for the coboundaries into its cochains
            for u in 0..=this.max_u(a + 1).min(this.max_u(max_s)) {
                keys.push((a, u));
            }
        }
        let spaces: Vec<Space> = keys
            .clone()
            .into_maybe_par_iter()
            .map(|(a, u)| Space::new(bb, q, &this.cells, max_len, a, u))
            .collect();
        this.spaces = keys.iter().copied().zip(spaces).collect();
        let dkeys: Vec<(i32, i32)> = keys
            .iter()
            .copied()
            .filter(|&(a, u)| this.spaces.contains_key(&(a + 1, u)))
            .collect();
        let deltas: Vec<Matrix> = dkeys
            .clone()
            .into_maybe_par_iter()
            .map(|(a, u)| {
                pullback(
                    q,
                    bb,
                    &this.cells,
                    &this.spaces[&(a, u)],
                    &this.spaces[&(a + 1, u)],
                    |t, l| bb.d(a + 1, t, l),
                    0,
                )
            })
            .collect();
        this.delta = dkeys.iter().copied().zip(deltas).collect();
        let mut tris = Vec::new();
        for a in 0..=max_s {
            for u in 0..=this.max_u(a) {
                for k in 0..=max_s - a {
                    tris.push(Tri::new(a, u, k));
                }
            }
        }
        let coh: Vec<Subquotient> = tris
            .clone()
            .into_maybe_par_iter()
            .map(|x| this.compute_cohomology(x))
            .collect();
        this.coh = tris.into_iter().zip(coh).collect();
        this
    }

    /// The number of cochains over every filtration and weight held, and the number of nonzero
    /// entries of the coboundaries between them.
    pub fn coboundary_stats(&self) -> (usize, usize) {
        let cochains = self.spaces.values().map(Space::dim).sum();
        let entries = self
            .delta
            .values()
            .map(|d| {
                (0..d.rows())
                    .map(|r| d.row(r).iter_nonzero().count())
                    .sum::<usize>()
            })
            .sum();
        (cochains, entries)
    }

    /// The cochains of filtration `a` and weight `u`.
    pub fn space(&self, a: i32, u: i32) -> Option<&Space> {
        self.spaces.get(&(a, u))
    }

    /// The columns of the cochains at `x`: those of length at most `k`, or exactly `k` when
    /// $q_0$ is killed.
    fn window(&self, a: i32, u: i32, k: i32) -> (usize, usize) {
        let sp = &self.spaces[&(a, u)];
        let lo = if self.cells.mod_q0 {
            sp.prefix(k - 1)
        } else {
            0
        };
        (lo, sp.prefix(k))
    }

    /// The cohomology at `x` as a subquotient of the window of its cochains.
    fn compute_cohomology(&self, x: Tri) -> Subquotient {
        let (lo, hi) = self.window(x.a, x.u, x.k);
        let n = hi - lo;
        let kernel = if n > 0 {
            let d = &self.delta[&(x.a, x.u)];
            let (tlo, thi) = self.window(x.a + 1, x.u, x.k);
            let mut m = AugmentedMatrix::<2>::new(TWO, n, [thi - tlo, n]);
            let id = m.start[1];
            for r in 0..n {
                let mut row = m.row_mut(r);
                row.slice_mut(0, thi - tlo)
                    .assign(d.row(lo + r).restrict(tlo, thi));
                row.add_basis_element(id + r, 1);
            }
            m.row_reduce();
            m.compute_kernel()
        } else {
            Subspace::new(TWO, 0)
        };
        let image = match (x.a > 0).then(|| self.delta.get(&(x.a - 1, x.u))).flatten() {
            Some(d) if n > 0 => {
                let (slo, shi) = self.window(x.a - 1, x.u, x.k);
                let mut m = Matrix::new(TWO, shi - slo, n);
                for r in 0..shi - slo {
                    m.row_mut(r).assign(d.row(slo + r).restrict(lo, hi));
                }
                Subspace::from_matrix(m)
            }
            _ => Subspace::new(TWO, n),
        };
        Subquotient::from_parts(kernel, image)
    }

    /// The cohomology at `x`, if `x` is in the range.
    pub fn cohomology(&self, x: Tri) -> Option<&Subquotient> {
        self.coh.get(&x)
    }

    /// The dimension at `x`, zero outside the range.
    pub fn dimension(&self, x: Tri) -> usize {
        self.coh.get(&x).map_or(0, |h| h.dimension())
    }

    /// The tridegrees in the range with nonzero cohomology, sorted.
    pub fn nonzero(&self) -> Vec<Tri> {
        let mut v: Vec<Tri> = self
            .coh
            .iter()
            .filter(|(_, h)| h.dimension() > 0)
            .map(|(&x, _)| x)
            .collect();
        v.sort_by_key(|x| (x.stem(), x.filtration(), x.weight()));
        v
    }

    /// The cocycle representing basis class `idx` at `x`, over the whole [`Space`].
    pub fn representative(&self, x: Tri, idx: usize) -> FpVector {
        let (lo, hi) = self.window(x.a, x.u, x.k);
        let mut v = FpVector::new(TWO, self.spaces[&(x.a, x.u)].dim());
        let g = self.coh[&x].gens().nth(idx).expect("a basis class");
        v.slice_mut(lo, hi).assign(g);
        v
    }

    /// The class at `x` of a cochain over the whole [`Space`], which must be a cocycle of length
    /// at most `x.k`; `None` if `x` is outside the range.
    pub fn class(&self, x: Tri, v: &FpVector) -> Option<Vec<u32>> {
        let h = self.coh.get(&x)?;
        let (lo, hi) = self.window(x.a, x.u, x.k);
        assert!(
            v.slice(hi, v.len()).is_zero(),
            "a cochain at {x:?} is longer than k"
        );
        let mut w = FpVector::new(TWO, hi - lo);
        w.as_slice_mut().assign(v.slice(lo, hi));
        let coords = h.reduce(w.as_slice_mut());
        assert!(w.is_zero(), "a cochain at {x:?} is not a cocycle");
        Some(coords)
    }

    /// The matrix of a cochain map on cohomology, from `x` to `y`: row `i` is the image of basis
    /// class `i`. `None` if `y` is outside the range.
    pub fn induced(&self, x: Tri, y: Tri, map: &Matrix) -> Option<Vec<Vec<u32>>> {
        self.coh.get(&y)?;
        let tgt = &self.spaces[&(y.a, y.u)];
        Some(
            (0..self.dimension(x))
                .map(|i| {
                    let mut w = FpVector::new(TWO, tgt.dim());
                    map.apply(w.as_slice_mut(), 1, self.representative(x, i).as_slice());
                    self.class(y, &w).unwrap()
                })
                .collect(),
        )
    }

    /// Multiplication by $h_0 = q_0$ from `x`, into polynomial degree `x.k + 1`.
    pub fn h0(&self, x: Tri) -> Option<Vec<Vec<u32>>> {
        assert!(!self.cells.mod_q0);
        let y = x + Tri::new(0, 0, 1);
        self.coh.get(&y)?;
        Some(
            (0..self.dimension(x))
                .map(|i| self.class(y, &self.representative(x, i)).unwrap())
                .collect(),
        )
    }

    /// The cochain map of the cup product with a class of [`Tri`] `shift`, whose pullback is
    /// described as in [`pullback`], out of filtration `a` and weight `u`.
    pub fn operator<'a>(
        &self,
        q: &QAlgebra,
        bb: &Backbone,
        a: i32,
        u: i32,
        by: Tri,
        terms: impl Fn(i32, usize) -> &'a [Term],
        shift: PackedXi,
    ) -> Option<Matrix> {
        let src = self.spaces.get(&(a, u))?;
        let tgt = self.spaces.get(&(a + by.a, u + by.u))?;
        Some(pullback(q, bb, &self.cells, src, tgt, terms, shift))
    }

    /// Multiplication by the motivic $h_{j+1}$, the classical $h_j$, from `x`.
    ///
    /// $(h_j c)(z) = \sum P(R - (2^j)) \cdot c(x)$ over the terms $P(R) x$ of $dz$: the connecting
    /// map of $Q \otimes D(C h_j)$, by the Cartan formula.
    pub fn h(&self, q: &QAlgebra, bb: &Backbone, j: u32, x: Tri) -> Option<Vec<Vec<u32>>> {
        let by = Tri::new(1, 1 << j, 0);
        let m = self.operator(q, bb, x.a, x.u, by, |t, l| bb.d(x.a + 1, t, l), 1 << j)?;
        self.induced(x, x + by, &m)
    }
}

/// The map on cochains induced by including the bottom cell, $C\tau \to C\tau \wedge C$, from
/// `x`, and its target.
pub fn bottom_cell(s: &CtauExt, c: &CtauExt, x: Tri) -> Option<Vec<Vec<u32>>> {
    c.cohomology(x)?;
    let (src, tgt) = (s.space(x.a, x.u)?, c.space(x.a, x.u)?);
    Some(
        (0..s.dimension(x))
            .map(|i| {
                let v = s.representative(x, i);
                let mut w = FpVector::new(TWO, tgt.dim());
                for (pos, _) in v.iter_nonzero() {
                    let (_, t, g, m) = src.basis[pos];
                    // modulo q_0 the shorter monomials vanish
                    if c.cells.mod_q0 && (mono_len(&m) as i32) < x.k {
                        continue;
                    }
                    w.add_basis_element(tgt.index(0, t, g, &m).unwrap(), 1);
                }
                c.class(x, &w).unwrap()
            })
            .collect(),
    )
}

/// The map induced by collapsing onto the top cell, $C\tau \wedge C \to \Sigma C\tau$, from `x`,
/// with its target.
///
/// On a Hopf cofiber it reads the coordinates on the top cell. When $q_0$ is killed it is the
/// Bockstein: the coboundary in $C\tau$ of a cocycle modulo $q_0$, which is shorter by one.
pub fn top_cell(s: &CtauExt, c: &CtauExt, x: Tri) -> Option<(Tri, Vec<Vec<u32>>)> {
    let src = c.space(x.a, x.u)?;
    if c.cells.mod_q0 {
        let y = Tri::new(x.a + 1, x.u, x.k - 1);
        s.cohomology(y)?;
        let d = s.delta.get(&(x.a, x.u))?;
        let tgt = s.space(y.a, y.u)?;
        let rows = (0..c.dimension(x))
            .map(|i| {
                let mut w = FpVector::new(TWO, tgt.dim());
                d.apply(w.as_slice_mut(), 1, c.representative(x, i).as_slice());
                s.class(y, &w).unwrap()
            })
            .collect();
        return Some((y, rows));
    }
    let e = *c.cells.degrees.last().unwrap();
    let top = c.cells.degrees.len() - 1;
    let y = Tri::new(x.a, x.u - e, x.k);
    s.cohomology(y)?;
    let tgt = s.space(y.a, y.u)?;
    let rows = (0..c.dimension(x))
        .map(|i| {
            let v = c.representative(x, i);
            let mut w = FpVector::new(TWO, tgt.dim());
            for (pos, _) in v.iter_nonzero() {
                let (cell, t, g, m) = src.basis[pos];
                if cell as usize == top {
                    w.add_basis_element(tgt.index(0, t, g, &m).unwrap(), 1);
                }
            }
            s.class(y, &w).unwrap()
        })
        .collect();
    Some((y, rows))
}

/// A chain map $P_{\bullet + s} \to P_\bullet$ lifting a class of the classical
/// $\mathrm{Ext}^{s, t}$, as its values on generators.
pub struct ChainMapTerms {
    /// The classical filtration and internal degree of the class.
    pub s: i32,
    pub t: i32,
    /// `(a, t_l)` maps to the terms of the image of each generator of $P_a$ of degree `t_l`.
    pub values: FxHashMap<(i32, i32), Vec<Vec<Term>>>,
}

impl ChainMapTerms {
    /// Reads the saved lift of generator `idx` of the classical bidegree `b` out of `store`,
    /// for the generators of $P_a$ with `a <= max_a` in degrees up to `max_t(a)`.
    ///
    /// Nothing is lifted: a step the store lacks is left out, and [`Self::get`] reports it. The
    /// first step is the class itself, which has no saved step.
    pub fn read<CC>(
        res: &CC,
        ops: &OpTable,
        store: &crate::save::ZarrSaveStore,
        b: Bidegree,
        idx: usize,
        max_a: i32,
        max_t: impl Fn(i32) -> i32,
    ) -> anyhow::Result<Self>
    where
        CC: FreeChainComplex<Algebra = MilnorAlgebra>,
    {
        let mut values = FxHashMap::default();
        for a in b.s()..=max_a {
            let target = res.module(a - b.s());
            for t in a..=max_t(a) {
                let input = Bidegree::s_t(a, t);
                if !res.has_computed_bidegree(input) {
                    continue;
                }
                let gens = res.number_of_gens_in_bidegree(input);
                let out_t = t - b.t();
                let dim = if out_t < 0 {
                    0
                } else {
                    target.dimension(out_t)
                };
                let rows = if a == b.s() {
                    (0..gens)
                        .map(|l| {
                            if t == b.t() && l == idx {
                                vec![Term { op: 0, t: 0, i: 0 }]
                            } else {
                                Vec::new()
                            }
                        })
                        .collect()
                } else if gens == 0 || dim == 0 {
                    vec![Vec::new(); gens]
                } else if let Some(data) = store.read(crate::save::SaveKind::ChainMap, input)? {
                    let mut cursor = &data[..];
                    let mut rows = Vec::with_capacity(gens);
                    for _ in 0..gens {
                        let Ok(v) = FpVector::from_bytes(TWO, dim, &mut cursor) else {
                            break;
                        };
                        rows.push(ops.terms(&target, out_t, v.as_slice()));
                    }
                    if rows.len() < gens {
                        continue;
                    }
                    rows
                } else {
                    continue;
                };
                values.insert((a, t), rows);
            }
        }
        Ok(Self {
            s: b.s(),
            t: b.t(),
            values,
        })
    }

    /// The terms of the image of generator `l` of $P_a$ in degree `t`; `None` if that step is
    /// not held.
    pub fn get(&self, a: i32, t: i32, l: usize) -> Option<&[Term]> {
        self.values.get(&(a, t)).map(|v| &v[l][..])
    }
}

/// A cocycle on the bar resolution with values in $Q$, representing a class $\beta$ of
/// $\mathrm{Ext}_{A/\tau}$, for multiplying by $\beta$.
///
/// With $\Gamma\colon B_{a'}(A) \to Q$ such a cocycle, the product with the class of a cocycle
/// $c\colon P_a \to Q$ is represented by
/// $$ z \mapsto \sum \Gamma[\theta_{a'}' | \dots | \theta_1'] \cdot \theta_{a'}'' \cdots
/// \theta_1'' c(z_0) $$
/// over the chains of terms $\theta_j z_{j-1}$ of $d z_j$ from $z = z_{a'}$ down to a generator
/// $z_0$ of $P_a$, and the splittings of each $\theta_j$ with $\theta_j' \neq 1$: the lift of
/// $c$ to the bar resolution of $Q$ along the contracting homotopy $[\dots] \mapsto 1[\dots]$,
/// followed by $\Gamma \otimes 1$ and the multiplication of $Q$.
pub struct BarCocycle {
    /// The tridegree of $\beta$.
    pub degree: Tri,
    /// The value on each bar word $[\theta_{a'} | \dots | \theta_1]$ on which it is nonzero.
    pub values: FxHashMap<Vec<PackedXi>, QPoly>,
    /// For each proper suffix of a word of `values`, the letters that extend it to the left.
    pub(crate) extensions: FxHashMap<Vec<PackedXi>, Vec<PackedXi>>,
}

/// Sections of the differentials of a resolution in low degrees, for pulling a cocycle back to
/// the bar resolution.
pub struct BarComparison<'a, CC: FreeChainComplex<Algebra = MilnorAlgebra>> {
    res: &'a CC,
    ops: &'a OpTable,
    sections: DashMap<(i32, i32), Arc<QuasiInverse>>,
    /// $\psi_j$ on each bar word, as an element of $P_j$.
    psi: DashMap<Vec<PackedXi>, Arc<FpVector>>,
}

impl<'a, CC: FreeChainComplex<Algebra = MilnorAlgebra>> BarComparison<'a, CC> {
    /// An empty comparison on `res`; sections are computed as words ask for them.
    pub fn new(res: &'a CC, ops: &'a OpTable) -> Self {
        Self {
            res,
            ops,
            sections: DashMap::default(),
            psi: DashMap::default(),
        }
    }

    /// A section of $d\colon P_j \to P_{j-1}$ in degree `t`, by row reduction.
    fn section(&self, j: i32, t: i32) -> Arc<QuasiInverse> {
        if let Some(hit) = self.sections.get(&(j, t)) {
            return Arc::clone(&hit);
        }
        let d = self.res.differential(j);
        let (src, tgt) = (self.res.module(j), self.res.module(j - 1));
        src.compute_basis(t);
        tgt.compute_basis(t);
        let (n, m) = (src.dimension(t), tgt.dimension(t));
        let mut matrix = AugmentedMatrix::<2>::new(TWO, n, [m, n]);
        let id = matrix.start[1];
        for r in 0..n {
            let mut row = matrix.row_mut(r);
            d.apply_to_basis_element(row.slice_mut(0, m), 1, t, r);
            row.add_basis_element(id + r, 1);
        }
        matrix.row_reduce();
        let qi = Arc::new(matrix.compute_quasi_inverse());
        self.sections.insert((j, t), Arc::clone(&qi));
        qi
    }

    /// The comparison map $\psi\colon B_j(A) \to P_j$ on the word $1[\theta_j | \dots |
    /// \theta_1]$, by $\psi_j = \sigma_j \psi_{j-1} d$ with $\sigma_j$ the [`Self::section`].
    pub fn psi(&self, word: &[PackedXi]) -> Arc<FpVector> {
        if let Some(hit) = self.psi.get(word) {
            return Arc::clone(&hit);
        }
        let j = word.len() as i32;
        let t: i32 = word.iter().map(|&r| xi_degree(r)).sum();
        let module = self.res.module(j);
        module.compute_basis(t);
        let out = if j == 0 {
            let mut v = FpVector::new(TWO, module.dimension(0));
            v.set_entry(0, 1);
            v
        } else {
            let algebra = self.res.algebra();
            let below = self.res.module(j - 1);
            below.compute_basis(t);
            let mut cycle = FpVector::new(TWO, below.dimension(t));
            // θ_j ψ[θ_{j-1} | ... | θ_1]
            let (deg, idx) = self.ops.index(word[0]);
            let rest = self.psi(&word[1..]);
            below.act(cycle.as_slice_mut(), 1, deg, idx, t - deg, rest.as_slice());
            // ψ[... | θ_{i+1} θ_i | ...]
            for i in 0..word.len() - 1 {
                let (d1, i1) = self.ops.index(word[i]);
                let (d2, i2) = self.ops.index(word[i + 1]);
                let mut prod = FpVector::new(TWO, algebra.dimension(d1 + d2));
                algebra.multiply_basis_elements(prod.as_slice_mut(), 1, d1, i1, d2, i2);
                for (p, _) in prod.iter_nonzero() {
                    let mut w = word.to_vec();
                    w[i] = self.ops.pack(d1 + d2, p);
                    w.remove(i + 1);
                    cycle.add(&self.psi(&w), 1u32);
                }
            }
            let mut v = FpVector::new(TWO, module.dimension(t));
            self.section(j, t)
                .apply(v.as_slice_mut(), 1, cycle.as_slice());
            v
        };
        let out = Arc::new(out);
        self.psi.insert(word.to_vec(), Arc::clone(&out));
        out
    }

    /// The pullback to the bar resolution of the cocycle representing basis class `idx` of
    /// $C\tau$ at `x`.
    pub fn cocycle(&self, q: &QAlgebra, ext: &CtauExt, x: Tri, idx: usize) -> BarCocycle {
        let rep = ext.representative(x, idx);
        let space = ext.space(x.a, x.u).unwrap();
        // g(x_i) for each generator of P_a
        let mut g: FxHashMap<(i32, u32), QPoly> = FxHashMap::default();
        for (pos, _) in rep.iter_nonzero() {
            let (_, t, i, m) = space.basis[pos];
            g.entry((t, i)).or_default().push(m);
        }
        self.cocycle_of(q, x, &g)
    }

    /// The pullback to the bar resolution of the cocycle $P_a \to Q_k$ with the values `g` on
    /// generators, by degree and index, of weight `x.u`.
    pub fn cocycle_of(&self, q: &QAlgebra, x: Tri, g: &FxHashMap<(i32, u32), QPoly>) -> BarCocycle {
        let algebra = self.res.algebra();
        algebra.compute_basis(x.u);
        let module = self.res.module(x.a);
        // the letters by degree
        let letters: Vec<Vec<PackedXi>> = (0..=x.u)
            .map(|d| {
                (0..algebra.dimension(d))
                    .map(|i| self.ops.pack(d, i))
                    .collect()
            })
            .collect();
        let mut values = FxHashMap::default();
        let mut word = Vec::new();
        self.enumerate(q, &module, g, &letters, x, &mut word, 0, &mut values);
        let mut extensions: FxHashMap<Vec<PackedXi>, Vec<PackedXi>> = FxHashMap::default();
        for w in values.keys() {
            for i in 0..w.len() {
                let e = extensions.entry(w[i + 1..].to_vec()).or_default();
                if !e.contains(&w[i]) {
                    e.push(w[i]);
                }
            }
        }
        BarCocycle {
            degree: x,
            values,
            extensions,
        }
    }

    /// Extends `word` to the left by every letter and evaluates the cocycle `g` on the complete
    /// words.
    #[allow(clippy::too_many_arguments)]
    fn enumerate(
        &self,
        q: &QAlgebra,
        module: &algebra::module::FreeModule<MilnorAlgebra>,
        g: &FxHashMap<(i32, u32), QPoly>,
        letters: &[Vec<PackedXi>],
        x: Tri,
        word: &mut Vec<PackedXi>,
        degree: i32,
        values: &mut FxHashMap<Vec<PackedXi>, QPoly>,
    ) {
        if word.len() as i32 == x.a {
            // the value lies in Q_k in degree u - |word|
            if !q
                .monomials(x.u - degree)
                .iter()
                .any(|m| mono_len(m) as i32 <= x.k)
            {
                return;
            }
            let v = self.psi(word);
            let mut out = FxHashSet::default();
            for (pos, _) in v.iter_nonzero() {
                let og = module.index_to_op_gen(degree, pos);
                if let Some(f) = g.get(&(og.generator_degree, og.generator_index as u32)) {
                    let r = self.ops.pack(og.operation_degree, og.operation_index);
                    poly_add(&mut out, q.act_poly(r, f));
                }
            }
            if !out.is_empty() {
                values.insert(word.clone(), out.into_iter().collect());
            }
            return;
        }
        for d in 1..=x.u - degree {
            for &r in &letters[d as usize] {
                word.insert(0, r);
                self.enumerate(q, module, g, letters, x, word, degree + d, values);
                word.remove(0);
            }
        }
    }
}

impl BarCocycle {
    /// The cocycle representing $\beta \cdot [c]$ for the cocycle `c` of $C\tau$ at filtration
    /// `a` and weight `u`, as a cochain at filtration `a + β.a` and weight `u + β.u`.
    pub fn cup(
        &self,
        q: &QAlgebra,
        bb: &Backbone,
        ext: &CtauExt,
        a: i32,
        u: i32,
        c: &FpVector,
    ) -> Option<FpVector> {
        let src = ext.space(a, u)?;
        let tgt = ext.space(a + self.degree.a, u + self.degree.u)?;
        let mut values: FxHashMap<(i32, u32), QPoly> = FxHashMap::default();
        for (pos, _) in c.iter_nonzero() {
            let (_, t, i, m) = src.basis[pos];
            values.entry((t, i)).or_default().push(m);
        }
        // lifts[j][(t, l)]: the lift of c on the generator z_l of P_{a+j}, by bar word
        type Lift = FxHashMap<Vec<PackedXi>, FxHashSet<QMono>>;
        let mut lifts: FxHashMap<(i32, u32), Lift> = values
            .into_iter()
            .map(|(key, f)| {
                let mut lift = Lift::default();
                lift.insert(Vec::new(), f.into_iter().collect());
                (key, lift)
            })
            .collect();
        for j in 1..=self.degree.a {
            let mut next: FxHashMap<(i32, u32), Lift> = FxHashMap::default();
            for t_l in 0..=tgt.u {
                for l in 0..bb.gens(a + j, t_l) {
                    let mut lift = Lift::default();
                    for term in bb.d(a + j, t_l, l) {
                        let Some(below) = lifts.get(&(term.t, term.i)) else {
                            continue;
                        };
                        for (w, f) in below {
                            let Some(letters) = self.extensions.get(w) else {
                                continue;
                            };
                            let f: QPoly = f.iter().copied().collect();
                            for &r in letters {
                                if !fits(r, term.op) {
                                    continue;
                                }
                                let mut w2 = Vec::with_capacity(w.len() + 1);
                                w2.push(r);
                                w2.extend_from_slice(w);
                                poly_add(lift.entry(w2).or_default(), q.act_poly(term.op - r, &f));
                            }
                        }
                    }
                    lift.retain(|_, f| !f.is_empty());
                    if !lift.is_empty() {
                        next.insert((t_l, l as u32), lift);
                    }
                }
            }
            lifts = next;
        }
        let mut out = FpVector::new(TWO, tgt.dim());
        for ((t_l, l), lift) in lifts {
            let mut value: FxHashSet<QMono> = FxHashSet::default();
            for (w, f) in lift {
                let Some(gamma) = self.values.get(&w) else {
                    continue;
                };
                for x in gamma {
                    for y in &f {
                        toggle(&mut value, mono_mul(x, y));
                    }
                }
            }
            for m in value {
                // a monomial too long for the range is in no class of the range
                if let Some(pos) = tgt.index(0, t_l, l, &m) {
                    out.add_basis_element(pos, 1);
                }
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every exponent sequence of degree `d`.
    fn sequences(algebra: &MilnorAlgebra, d: i32) -> Vec<PackedXi> {
        (0..algebra.dimension(d))
            .map(|i| pack(algebra.basis_element_from_index(d, i).p_part.iter()))
            .collect()
    }

    /// $P(R)(P(S) m) = (P(R) P(S)) m$, with the product taken in the Milnor basis.
    #[test]
    fn q_is_a_module() {
        let max = 14;
        let algebra = Arc::new(MilnorAlgebra::new(TWO, false));
        algebra.compute_basis(max);
        let ops = OpTable::new(Arc::clone(&algebra));
        let q = QAlgebra::new(max, 4);
        for d in 0..=max {
            for m in q.monomials(d) {
                for d1 in 1..=d {
                    for d2 in 1..=d - d1 {
                        for (i1, &r) in sequences(&algebra, d1).iter().enumerate() {
                            for (i2, &s) in sequences(&algebra, d2).iter().enumerate() {
                                let mut lhs: FxHashSet<QMono> = FxHashSet::default();
                                poly_add(&mut lhs, q.act_poly(r, &q.act(s, m)));
                                let mut prod = FpVector::new(TWO, algebra.dimension(d1 + d2));
                                algebra.multiply_basis_elements(
                                    prod.as_slice_mut(),
                                    1,
                                    d1,
                                    i1,
                                    d2,
                                    i2,
                                );
                                let mut rhs: FxHashSet<QMono> = FxHashSet::default();
                                for (i, _) in prod.iter_nonzero() {
                                    poly_add(
                                        &mut rhs,
                                        q.act(ops.pack(d1 + d2, i), m).iter().copied(),
                                    );
                                }
                                assert_eq!(lhs, rhs, "P({r:x}) P({s:x}) on {m:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// $P(R)(m m') = \sum_{R' + R'' = R} P(R') m \cdot P(R'') m'$.
    #[test]
    fn q_satisfies_cartan() {
        let max = 12;
        let algebra = MilnorAlgebra::new(TWO, false);
        algebra.compute_basis(max);
        let q = QAlgebra::new(max, 4);
        let all: Vec<PackedXi> = (0..=max).flat_map(|d| sequences(&algebra, d)).collect();
        for d1 in 0..=max {
            for d2 in 0..=max - d1 {
                for m1 in q.monomials(d1).iter().filter(|m| mono_len(m) <= 2) {
                    for m2 in q.monomials(d2).iter().filter(|m| mono_len(m) <= 2) {
                        for &r in all.iter().filter(|&&r| xi_degree(r) <= d1 + d2) {
                            let lhs: FxHashSet<QMono> =
                                q.act(r, &mono_mul(m1, m2)).iter().copied().collect();
                            let mut rhs: FxHashSet<QMono> = FxHashSet::default();
                            for &r1 in all.iter().filter(|&&r1| fits(r1, r)) {
                                for x in q.act(r1, m1).iter() {
                                    for y in q.act(r - r1, m2).iter() {
                                        toggle(&mut rhs, mono_mul(x, y));
                                    }
                                }
                            }
                            assert_eq!(lhs, rhs, "P({r:x}) on {m1:?} {m2:?}");
                        }
                    }
                }
            }
        }
    }

    /// The action lowers the degree by that of the operation and never lengthens a monomial.
    #[test]
    fn q_action_degrees() {
        let max = 20;
        let algebra = MilnorAlgebra::new(TWO, false);
        algebra.compute_basis(max);
        let q = QAlgebra::new(max, 6);
        for d in 0..=max {
            for m in q.monomials(d) {
                for e in 1..=d {
                    for r in sequences(&algebra, e) {
                        for g in q.act(r, m).iter() {
                            assert_eq!(mono_degree(g), d - e);
                            assert!(mono_len(g) <= mono_len(m));
                        }
                    }
                }
            }
        }
    }
}
