//! The backbone lifted to $BP_*BP$ modulo $I^2$, and the first algebraic Novikov differential.

use std::{collections::BTreeMap, sync::Arc};

use algebra::{
    MilnorAlgebra,
    milnor_algebra::{MilnorBasisElement, PPart},
    module::{Module, homomorphism::ModuleHomomorphism},
};
use dashmap::DashMap;
use fp::{prime::TWO, vector::FpVector};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use rustc_hash::{FxHashMap, FxHashSet};
use sseq::coordinates::Bidegree;

use crate::{
    algebraic_novikov::{
        PackedXi, QDOWN_VARS, QMono, XI_BITS, chi_table, monomials, pack, packed_fits, qmono_len,
        toggle, unpack, v_degree,
    },
    bp::BpHopfAlgebroid,
    chain_complex::FreeChainComplex,
};

/// An element of $BP_*$ modulo $2^{64}$: monomials in $v_1, v_2, \dots$ (slot 0 unused) to
/// coefficients.
pub type BpElt = FxHashMap<QMono, i64>;

/// An element of $BP_*BP$ modulo $2^{64}$: monomials in the $t_j$ to their left coefficients.
type BpPoly = FxHashMap<PackedXi, BpElt>;

/// The $I$-adic filtration of $c\, v^K$: the 2-adic valuation of $c$ plus the length of $K$.
fn filtration(m: &QMono, c: i64) -> u32 {
    c.trailing_zeros() + qmono_len(m)
}

/// Adds $c\, v^K$ to `a`.
fn add_term(a: &mut BpElt, m: QMono, c: i64) {
    let e = a.entry(m).or_insert(0);
    *e = e.wrapping_add(c);
    if *e == 0 {
        a.remove(&m);
    }
}

/// The product of two monomials.
fn qmono_mul(a: &QMono, b: &QMono) -> QMono {
    std::array::from_fn(|i| a[i] + b[i])
}

/// The degree of a packed monomial in the $t_j$, halved: $|t_j| = 2^j - 1$.
fn packed_degree(r: PackedXi) -> i32 {
    unpack(r)
        .iter()
        .enumerate()
        .map(|(j, &e)| e as i32 * v_degree(j + 1))
        .sum()
}

/// $t_n$, packed.
fn packed_t(n: usize) -> PackedXi {
    1 << ((n as u32 - 1) * XI_BITS)
}

/// The exponent of $t_n$ in a packed monomial.
fn packed_exponent(r: PackedXi, n: usize) -> u32 {
    ((r >> ((n as u32 - 1) * XI_BITS)) & ((1 << XI_BITS) - 1)) as u32
}

/// The residue of an integral rational modulo $2^{64}$.
fn wrap(x: &BigRational) -> i64 {
    assert!(x.is_integer(), "{x} is not integral");
    let m = BigInt::from(1u128 << 64);
    let r = ((x.numer() % &m) + &m) % &m;
    r.to_u64().unwrap() as i64
}

/// The odd part of $n!$ modulo 4.
fn odd_factorial_mod4(mut n: u64) -> u8 {
    let mut r = 1;
    while n > 0 {
        // The odd numbers up to n that are 3 mod 4 number (n + 1) / 4.
        if ((n + 1) / 4) % 2 == 1 {
            r = (r * 3) % 4;
        }
        n >>= 1;
    }
    r
}

/// $\binom{a + b}{a}$ modulo 4.
fn binomial_mod4(a: u64, b: u64) -> u8 {
    let carries = a.count_ones() + b.count_ones() - (a + b).count_ones();
    if carries >= 2 {
        return 0;
    }
    let odd = odd_factorial_mod4(a + b) * odd_factorial_mod4(a) * odd_factorial_mod4(b) % 4;
    (odd << carries) % 4
}

/// A multinomial coefficient modulo 4.
fn multinomial_mod4(parts: impl Iterator<Item = u32>) -> u8 {
    let mut total = 0u64;
    let mut r = 1;
    for x in parts {
        r = r * binomial_mod4(total, x as u64) % 4;
        if r == 0 {
            return 0;
        }
        total += x as u64;
    }
    r
}

/// An operator on $BP_*$ modulo $I^2$: $\sum_U \theta_U + \sum_{(h, U)} v_h \theta_U$, every
/// coefficient 1 (a coefficient $2$ is $v_0$).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Op {
    /// The $U$ with $\theta_U$ in the leading part.
    pub lead: FxHashSet<PackedXi>,
    /// The $(h, U)$ with $v_h \theta_U$ in the $I$-linear part.
    pub first: FxHashSet<(u8, PackedXi)>,
}

impl Op {
    /// Whether this is zero modulo $I^2$.
    pub fn is_zero(&self) -> bool {
        self.lead.is_empty() && self.first.is_empty()
    }
}

/// $(BP_*, BP_*BP)$ modulo $I^2$ through a halved degree, as operators on $BP_*$.
///
/// A comodule map between extended comodules $BP_*BP \otimes C \to BP_*BP \otimes C'$ is
/// determined by left $BP_*$-linear functionals on $BP_*BP$, and on primitives it acts on $BP_*$
/// through the operators $\theta_U$, where $\theta_U(x)$ is the left coefficient of $t^U$ in
/// $\eta_R(x)$. The functional of a composite is $\theta_S \circ \theta_T = \sum_U c^U_{S,T}
/// \theta_U$, with $c^U_{S,T}$ the left coefficient of $t^S \otimes t^T$ in $\Delta(t^U)$, and
/// $\theta_S \circ (b\, \theta_T) = \sum_{W + V = S} \theta_W(b)\, \theta_V \circ \theta_T$.
pub struct BpModI2 {
    /// The largest halved degree of a $t$-monomial kept.
    max_degree: i32,
    /// $\eta_R(v_n)$ for $n \geq 1$ (index 0 is $2$), terms of filtration at most 2.
    eta: Vec<BpPoly>,
    /// `eta_first[j]` lists the $(W, h)$ with $\theta_W(v_j) \equiv v_h$ modulo $I^2$.
    eta_first: Vec<Vec<(PackedXi, u8)>>,
    /// `delta_first[n]` lists the $(h, A, B)$ with $v_h t^A \otimes t^B$ in $\Delta(t_n)$
    /// modulo $I^2$.
    delta_first: Vec<Vec<(u8, PackedXi, PackedXi)>>,
    /// `chi[R']` lists the $R$ with $\xi^{R'}$ in $\chi(\xi^R)$.
    chi: FxHashMap<PackedXi, Vec<PackedXi>>,
    /// The leading part of $\theta_S \circ \theta_T$ modulo 4, by $(S, T)$.
    lead_memo: DashMap<(PackedXi, PackedXi), Arc<Vec<(PackedXi, u8)>>>,
    /// $\eta_R(v^K)$ modulo $I^{|K| + 2}$, by $K$.
    eta_memo: DashMap<QMono, Arc<BpPoly>>,
}

impl BpModI2 {
    /// The structure through halved degree `max_degree`.
    pub fn new(max_degree: i32) -> Self {
        let h = BpHopfAlgebroid::new(2 * max_degree);
        let n = h.n;
        assert!(n < QDOWN_VARS, "v_{n} does not fit a QMono");

        // A term of η_R(v_k): v-exponents then t-exponents.
        let split = |key: &[u32]| -> (QMono, PackedXi) {
            let mut v = [0u16; QDOWN_VARS];
            for i in 0..n {
                v[i + 1] = key[i] as u16;
            }
            (v, pack(&key[n..2 * n]))
        };

        let mut eta = vec![BpPoly::default()];
        eta[0].insert(0, [(([0u16; QDOWN_VARS]), 2)].into_iter().collect());
        let mut eta_first = vec![vec![(0, 0)]];
        for k in 1..=n {
            let mut p = BpPoly::default();
            let mut first = Vec::new();
            for (key, c) in &h.eta_r[k] {
                let (v, t) = split(key);
                let c = wrap(c);
                if c == 0 || packed_degree(t) > max_degree {
                    continue;
                }
                let f = filtration(&v, c);
                assert!(f >= 1, "η_R(v_{k}) has a term outside I");
                if f == 1 {
                    let h = if qmono_len(&v) == 1 {
                        v.iter().position(|&e| e == 1).unwrap() as u8
                    } else {
                        0
                    };
                    first.push((t, h));
                }
                if f <= 2 {
                    add_term(p.entry(t).or_default(), v, c);
                }
            }
            eta.push(p);
            eta_first.push(first);
        }

        let mut delta_first = vec![Vec::new()];
        for k in 1..=n {
            let mut lead = FxHashSet::default();
            let mut first = Vec::new();
            for (key, c) in &h.delta[k] {
                let (v, _) = split(key);
                let (a, b) = (pack(&key[n..2 * n]), pack(&key[2 * n..3 * n]));
                let c = wrap(c);
                if c == 0 {
                    continue;
                }
                match filtration(&v, c) {
                    0 => {
                        assert_eq!(c, 1, "Δ(t_{k}) has a leading coefficient other than 1");
                        lead.insert((a, b));
                    }
                    1 => {
                        let h = if qmono_len(&v) == 1 {
                            v.iter().position(|&e| e == 1).unwrap() as u8
                        } else {
                            0
                        };
                        first.push((h, a, b));
                    }
                    _ => {}
                }
            }
            let expected: FxHashSet<_> = (0..=k)
                .map(|i| {
                    let a = if i == 0 { 0 } else { packed_t(i) };
                    let b = if i == k { 0 } else { packed_t(k - i) << i };
                    (a, b)
                })
                .collect();
            assert_eq!(
                lead, expected,
                "Δ(t_{k}) is not Σ t_i ⊗ t_(k-i)^(2^i) mod I"
            );
            delta_first.push(first);
        }

        let chi = chi_table(max_degree);

        Self {
            max_degree,
            eta,
            eta_first,
            delta_first,
            chi,
            lead_memo: DashMap::default(),
            eta_memo: DashMap::default(),
        }
    }

    /// The operator $\rho(a)$ of the untwisted complex as a sum of $\theta_U$, for
    /// $a = \sum P(R')$: $\rho(P(R')) = \sum_R [\xi^{R'} \in \chi(\xi^R)]\, \theta_R$.
    ///
    /// The same table converts back, as $\chi$ is an involution.
    pub fn to_theta(&self, a: impl IntoIterator<Item = PackedXi>) -> FxHashSet<PackedXi> {
        let mut out = FxHashSet::default();
        for r in a {
            if let Some(col) = self.chi.get(&r) {
                for &u in col {
                    toggle(&mut out, u);
                }
            }
        }
        out
    }

    /// The leading part of $\theta_S \circ \theta_T$ modulo 4: the coefficients of
    /// $t^S \otimes t^T$ in $\prod_n (\sum_{i+j=n} t_i \otimes t_j^{2^i})^{u_n}$.
    fn lead(&self, s: PackedXi, t: PackedXi) -> Arc<Vec<(PackedXi, u8)>> {
        if let Some(hit) = self.lead_memo.get(&(s, t)) {
            return Arc::clone(&hit);
        }
        let (rows, cols) = (unpack(s), unpack(t));
        let (p, q) = (rows.len(), cols.len());
        let mut x = vec![vec![0u32; q + 1]; p + 1];
        let mut col_rem = cols.clone();
        let mut out: FxHashMap<PackedXi, u8> = FxHashMap::default();

        /// Chooses the cell `(i, j)` (from 1), with `row_rem` left in row `i`.
        #[allow(clippy::too_many_arguments)]
        fn rec(
            i: usize,
            j: usize,
            row_rem: u32,
            rows: &[u32],
            x: &mut Vec<Vec<u32>>,
            col_rem: &mut Vec<u32>,
            out: &mut FxHashMap<PackedXi, u8>,
        ) {
            let (p, q) = (rows.len(), col_rem.len());
            if i > p {
                for (jj, &c) in col_rem.iter().enumerate() {
                    x[0][jj + 1] = c;
                }
                let mut u = Vec::with_capacity(p + q);
                let mut coeff = 1u8;
                for n in 1..=p + q {
                    let cells = (n.saturating_sub(q)..=n.min(p)).map(|ii| x[ii][n - ii]);
                    u.push(cells.clone().sum::<u32>());
                    coeff = coeff * multinomial_mod4(cells) % 4;
                    if coeff == 0 {
                        return;
                    }
                }
                let e = out.entry(pack(&u)).or_insert(0);
                *e = (*e + coeff) % 4;
                return;
            }
            if j > q {
                x[i][0] = row_rem;
                let next = rows.get(i).copied().unwrap_or(0);
                rec(i + 1, 1, next, rows, x, col_rem, out);
                return;
            }
            let w = 1u32 << i;
            let mut e = 0;
            while e <= row_rem && e * w <= col_rem[j - 1] {
                x[i][j] = e;
                col_rem[j - 1] -= e * w;
                rec(i, j + 1, row_rem - e, rows, x, col_rem, out);
                col_rem[j - 1] += e * w;
                e += 1;
            }
            x[i][j] = 0;
        }

        if p == 0 {
            out.insert(t, 1);
        } else {
            rec(1, 1, rows[0], &rows, &mut x, &mut col_rem, &mut out);
        }
        let out: Arc<Vec<_>> = Arc::new(out.into_iter().filter(|&(_, c)| c != 0).collect());
        self.lead_memo.insert((s, t), Arc::clone(&out));
        out
    }

    /// Toggles into `out` the $I$-linear part of $\theta_S \circ \theta_T$ coming from the
    /// $I$-linear terms of $\Delta$, keeping $v_h \theta_U$ with $|U| \leq$ `cap`.
    ///
    /// $(L_n + V_n)^{u_n} \equiv L_n^{u_n} + u_n L_n^{u_n - 1} V_n$, so the coefficient of
    /// $t^S \otimes t^T$ at $U = W + e_n$ picks a term $v_h t^A \otimes t^B$ of $V_n$ and the
    /// leading coefficient of $t^{S - A} \otimes t^{T - B}$ in $\Delta(t^W)$, when $u_n$ is odd.
    fn first_into(&self, s: PackedXi, t: PackedXi, cap: i32, out: &mut FxHashSet<(u8, PackedXi)>) {
        let base = packed_degree(s) + packed_degree(t);
        for (n, terms) in self.delta_first.iter().enumerate().skip(1) {
            for &(h, a, b) in terms {
                if base + v_degree(h as usize) > cap || !packed_fits(a, s) || !packed_fits(b, t) {
                    continue;
                }
                for &(w, c) in self.lead(s - a, t - b).iter() {
                    if c % 2 == 1 && packed_exponent(w, n).is_multiple_of(2) {
                        toggle(out, (h, w + packed_t(n)));
                    }
                }
            }
        }
    }

    /// $\sum_i x_i \circ y_i$ modulo $I^2$, keeping $v_h \theta_U$ with $|U| \leq$ `cap`.
    ///
    /// The leading parts are summed modulo 4 before reducing, so that $2 \theta_U$ becomes
    /// $v_0 \theta_U$.
    pub fn compose_sum<'a>(
        &self,
        pairs: impl IntoIterator<Item = (&'a Op, &'a Op)>,
        cap: i32,
    ) -> Op {
        let mut counts: FxHashMap<PackedXi, u8> = FxHashMap::default();
        let mut first = FxHashSet::default();
        for (x, y) in pairs {
            for &s in &x.lead {
                for &t in &y.lead {
                    for &(u, c) in self.lead(s, t).iter() {
                        let e = counts.entry(u).or_insert(0);
                        *e = (*e + c) % 4;
                    }
                    self.first_into(s, t, cap, &mut first);
                }
                // θ_S ∘ (v_j θ_T) = Σ θ_W(v_j) θ_{S - W} ∘ θ_T.
                for &(j, t) in &y.first {
                    for &(w, h) in &self.eta_first[j as usize] {
                        if !packed_fits(w, s)
                            || packed_degree(s) - packed_degree(w) + packed_degree(t) > cap
                        {
                            continue;
                        }
                        for &(u, c) in self.lead(s - w, t).iter() {
                            if c % 2 == 1 {
                                toggle(&mut first, (h, u));
                            }
                        }
                    }
                }
            }
            for &(j, s) in &x.first {
                for &t in &y.lead {
                    if packed_degree(s) + packed_degree(t) > cap {
                        continue;
                    }
                    for &(u, c) in self.lead(s, t).iter() {
                        if c % 2 == 1 {
                            toggle(&mut first, (j, u));
                        }
                    }
                }
            }
        }
        let mut lead = FxHashSet::default();
        for (u, c) in counts {
            if c % 2 == 1 {
                lead.insert(u);
            }
            if c >= 2 && packed_degree(u) <= cap {
                toggle(&mut first, (0, u));
            }
        }
        Op { lead, first }
    }

    /// $\eta_R(v^K)$ modulo $I^{|K| + 2}$, for $K$ free of $v_0$.
    fn eta_of(&self, k: &QMono) -> Arc<BpPoly> {
        if let Some(hit) = self.eta_memo.get(k) {
            return Arc::clone(&hit);
        }
        let out = match k.iter().rposition(|&e| e > 0) {
            None => {
                let mut p = BpPoly::default();
                p.insert(0, [([0u16; QDOWN_VARS], 1)].into_iter().collect());
                p
            }
            Some(n) => {
                let mut rest = *k;
                rest[n] -= 1;
                let bound = qmono_len(k) + 1;
                let prev = self.eta_of(&rest);
                let mut p = BpPoly::default();
                for (&w1, e1) in prev.iter() {
                    for (&w2, e2) in &self.eta[n] {
                        let w = w1 + w2;
                        if packed_degree(w) > self.max_degree {
                            continue;
                        }
                        let entry = p.entry(w).or_default();
                        for (m1, &c1) in e1 {
                            for (m2, &c2) in e2 {
                                let c = c1.wrapping_mul(c2);
                                let m = qmono_mul(m1, m2);
                                if c != 0 && filtration(&m, c) <= bound {
                                    add_term(entry, m, c);
                                }
                            }
                        }
                    }
                }
                p.retain(|_, e| !e.is_empty());
                p
            }
        };
        let out = Arc::new(out);
        self.eta_memo.insert(*k, Arc::clone(&out));
        out
    }

    /// `op` applied to `x`, keeping terms of filtration at most `bound`, which must be at most
    /// one more than the filtration of `x`.
    pub fn apply(&self, op: &Op, x: &BpElt, bound: u32, out: &mut BpElt) {
        for (k, &c) in x {
            debug_assert!(filtration(k, c) + 1 >= bound);
            let eta = self.eta_of(k);
            let mut push = |m: QMono, c: i64| {
                if c != 0 && filtration(&m, c) <= bound {
                    add_term(out, m, c);
                }
            };
            for u in &op.lead {
                if let Some(e) = eta.get(u) {
                    for (m, &d) in e {
                        push(*m, c.wrapping_mul(d));
                    }
                }
            }
            for &(h, u) in &op.first {
                if let Some(e) = eta.get(&u) {
                    for (m, &d) in e {
                        if h == 0 {
                            push(*m, c.wrapping_mul(d).wrapping_mul(2));
                        } else {
                            let mut m = *m;
                            m[h as usize] += 1;
                            push(m, c.wrapping_mul(d));
                        }
                    }
                }
            }
        }
    }
}

/// The monomials of $Q_j$ in the filtration-$j$ layer of `x`: bit $j - |K|$ of each $c\, v^K$.
fn layer(x: &BpElt, j: u32) -> Vec<QMono> {
    let mut out = Vec::new();
    for (k, &c) in x {
        let len = qmono_len(k);
        if len > j {
            continue;
        }
        let b = j - len;
        if b < 64 && (c >> b) & 1 == 1 {
            let mut m = *k;
            m[0] = b as u16;
            out.push(m);
        }
    }
    out
}

/// A generator of the backbone: its degree and index.
type Gen = (i32, usize);

/// The backbone lifted to $BP_*BP$ modulo $I^2$, through weight `max_degree`.
///
/// Modulo $I$ the operators of [`BpModI2`] are the action of $P$ on $Q$ used by the untwisted
/// complex, so the minimal resolution over $P$ lifts by choosing each $\theta_U$ with
/// coefficient 1 and then correcting by terms $v_h \theta_U$ until $\tilde d^2 \equiv 0$
/// modulo $I^2$, solving each correction with the backbone's quasi-inverse.
///
/// A term $v_h \theta_U$ of the differential on a generator in degree $t$ is kept only when
/// $t + |v_h| \leq D$, the weight. Every composite keeps or raises $t + |v_h|$, so the kept part
/// is computed exactly, and it is all that acts on cochains of weight at most $D$.
pub struct LiftedBackbone {
    /// `lambda[s][(t, l)]` is the lifted differential of generator `l` of $C_s$ in degree `t`,
    /// by target generator.
    lambda: Vec<FxHashMap<Gen, FxHashMap<Gen, Op>>>,
    /// The weight it is truncated at.
    max_degree: i32,
}

/// The $P(R)$ of a basis element of the Milnor algebra, packed.
fn packed_p_part(algebra: &MilnorAlgebra, degree: i32, idx: usize) -> PackedXi {
    let elt = algebra.basis_element_from_index(degree, idx);
    pack(&elt.p_part.iter().collect::<Vec<_>>())
}

impl LiftedBackbone {
    /// Lifts the differentials out of $C_1, \dots, C_{\text{max\_s}}$, generators of degree at
    /// most the structure's degree, which `res` must be resolved through.
    pub fn new<CC>(res: &CC, bp: &BpModI2, max_s: i32) -> Self
    where
        CC: FreeChainComplex<Algebra = MilnorAlgebra>,
    {
        let max_degree = bp.max_degree;
        assert!(res.has_computed_bidegree(Bidegree::s_t(max_s, max_degree)));
        let algebra = res.algebra();
        let mut lambda = vec![FxHashMap::default()];
        for s in 1..=max_s {
            let d = res.differential(s);
            let target = res.module(s - 1);
            let mut level: FxHashMap<Gen, FxHashMap<Gen, Op>> = FxHashMap::default();
            for t in 0..=max_degree {
                for l in 0..res.number_of_gens_in_bidegree(Bidegree::s_t(s, t)) {
                    let mut grouped: FxHashMap<Gen, Vec<PackedXi>> = FxHashMap::default();
                    for (idx, _) in d.output(t, l).iter_nonzero() {
                        let og = target.index_to_op_gen(t, idx);
                        grouped
                            .entry((og.generator_degree, og.generator_index))
                            .or_default()
                            .push(packed_p_part(
                                &algebra,
                                og.operation_degree,
                                og.operation_index,
                            ));
                    }
                    let ops = grouped
                        .into_iter()
                        .map(|(g, a)| {
                            (
                                g,
                                Op {
                                    lead: bp.to_theta(a),
                                    first: FxHashSet::default(),
                                },
                            )
                        })
                        .collect();
                    level.insert((t, l), ops);
                }
            }
            lambda.push(level);
        }
        let mut lifted = Self { lambda, max_degree };
        for s in 2..=max_s {
            for t in 0..=max_degree {
                for l in 0..res.number_of_gens_in_bidegree(Bidegree::s_t(s, t)) {
                    lifted.correct(res, bp, s, (t, l));
                }
            }
        }
        lifted
    }

    /// $\tilde d^2$ on generator `z` of $C_s$, by generator of $C_{s-2}$, modulo $I^2$ and the
    /// truncation.
    pub fn error(&self, bp: &BpModI2, s: i32, z: Gen) -> FxHashMap<Gen, Op> {
        let mut by_target: FxHashMap<Gen, Vec<(&Op, &Op)>> = FxHashMap::default();
        for (l, x) in &self.lambda[s as usize][&z] {
            for (i, y) in &self.lambda[s as usize - 1][l] {
                by_target.entry(*i).or_default().push((x, y));
            }
        }
        by_target
            .into_iter()
            .map(|(i, pairs)| (i, bp.compose_sum(pairs, self.max_degree - i.0)))
            .filter(|(_, op)| !op.is_zero())
            .collect()
    }

    /// Adds to the differential on `z` the terms $v_h \theta_U$ that cancel its error.
    fn correct<CC>(&mut self, res: &CC, bp: &BpModI2, s: i32, z: Gen)
    where
        CC: FreeChainComplex<Algebra = MilnorAlgebra>,
    {
        let error = self.error(bp, s, z);
        let algebra = res.algebra();
        let mut by_h: BTreeMap<u8, FxHashMap<Gen, Vec<PackedXi>>> = BTreeMap::new();
        for (i, op) in &error {
            assert!(
                op.lead.is_empty(),
                "the lift of d^2 on {z:?} in C_{s} is not zero mod I"
            );
            for &(h, u) in &op.first {
                by_h.entry(h).or_default().entry(*i).or_default().push(u);
            }
        }
        let source = res.module(s - 2);
        let mid = res.module(s - 1);
        for (h, parts) in by_h {
            let degree = z.0 + v_degree(h as usize);
            assert!(degree <= self.max_degree);
            let mut e = FpVector::new(TWO, source.dimension(degree));
            for ((t_i, i), us) in parts {
                for r in bp.to_theta(us) {
                    let elt = MilnorBasisElement {
                        q_part: 0,
                        p_part: PPart::from_slice(
                            &unpack(r).iter().map(|&x| x as _).collect::<Vec<_>>(),
                        ),
                        degree: degree - t_i,
                    };
                    let op_idx = algebra.basis_element_to_index(&elt);
                    e.add_basis_element(
                        source.operation_generator_to_index(degree - t_i, op_idx, t_i, i),
                        1,
                    );
                }
            }
            let mut y = vec![FpVector::new(TWO, mid.dimension(degree))];
            assert!(
                res.apply_quasi_inverse(&mut y, Bidegree::s_t(s - 1, degree), &[e.clone()]),
                "no quasi-inverse at ({}, {degree})",
                s - 1
            );
            let mut check = FpVector::new(TWO, source.dimension(degree));
            res.differential(s - 1)
                .apply(check.as_slice_mut(), 1, degree, y[0].as_slice());
            assert_eq!(check, e, "the error on {z:?} in C_{s} is not a cycle");
            let mut grouped: FxHashMap<Gen, Vec<PackedXi>> = FxHashMap::default();
            for (idx, _) in y[0].iter_nonzero() {
                let og = mid.index_to_op_gen(degree, idx);
                grouped
                    .entry((og.generator_degree, og.generator_index))
                    .or_default()
                    .push(packed_p_part(
                        &algebra,
                        og.operation_degree,
                        og.operation_index,
                    ));
            }
            let row = self.lambda[s as usize].get_mut(&z).unwrap();
            for (l, a) in grouped {
                let op = row.entry(l).or_default();
                for u in bp.to_theta(a) {
                    toggle(&mut op.first, (h, u));
                }
            }
        }
    }

    /// The generators of $C_{s+1}$ whose lifted differential reaches each generator of $C_s$.
    fn inverse(&self, s: i32) -> FxHashMap<Gen, Vec<(Gen, &Op)>> {
        let mut out: FxHashMap<Gen, Vec<(Gen, &Op)>> = FxHashMap::default();
        if let Some(level) = self.lambda.get(s as usize + 1) {
            for (z, row) in level {
                for (x, op) in row {
                    out.entry(*x).or_default().push((*z, op));
                }
            }
        }
        out
    }
}

/// $\mathrm{Ext}_P(Q)$ and the rank of the first algebraic Novikov differential out of it, by
/// (stem, $s$, weight) of the source, through weight `max_u` and $s \leq$ `max_s`.
///
/// The complex $BP_* \otimes C^\vee$ of the [`LiftedBackbone`], filtered $I$-adically, has
/// $E_1 = \mathrm{Ext}_P(Q)$; its $d_1$ is a $C\tau$ Adams $d_2$. A cocycle of $Q_k$ is lifted
/// with $v_0 \mapsto 2$, its coboundary is taken over $\mathbb{Z}/2^{64}$, and the
/// filtration-$(k+1)$ layer is read modulo the image of $\delta_0$.
pub fn novikov_d2<CC>(res: &CC, max_u: i32, max_s: i32) -> BTreeMap<(i32, i32, i32), (usize, usize)>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra>,
{
    let bp = BpModI2::new(max_u);
    let lifted = LiftedBackbone::new(res, &bp, max_s + 2);
    let mut out = BTreeMap::new();

    // The cochains x^* ⊗ v_0^{k - |K|} v^K of filtration a, weight u and length k.
    let q = monomials(max_s as u32 + 1, |_| max_u);
    let basis = |a: i32, u: i32, k: u32| -> Vec<(Gen, QMono)> {
        let mut b = Vec::new();
        for t in 0..=u {
            for x in 0..res.number_of_gens_in_bidegree(Bidegree::s_t(a, t)) {
                for m in &q[(u - t) as usize] {
                    let len: u32 = m.iter().sum();
                    if len > k {
                        continue;
                    }
                    let mut qm = [0u16; QDOWN_VARS];
                    for (i, &e) in m.iter().enumerate() {
                        qm[i] = e as u16;
                    }
                    qm[0] += (k - len) as u16;
                    b.push(((t, x), qm));
                }
            }
        }
        b
    };

    // The lifted coboundary of each basis cochain, by target generator, over Z/2^64.
    let mut cache: FxHashMap<(i32, i32, u32), Arc<Vec<FxHashMap<Gen, BpElt>>>> =
        FxHashMap::default();
    // Every weight is read on its own, so the cache holds one weight at a time.
    let mut coboundaries = |a: i32, u: i32, k: u32| -> Arc<Vec<FxHashMap<Gen, BpElt>>> {
        if cache.keys().next().is_some_and(|key| key.1 != u) {
            cache.clear();
        }
        if let Some(hit) = cache.get(&(a, u, k)) {
            return Arc::clone(hit);
        }
        let inverse = lifted.inverse(a);
        let values = basis(a, u, k)
            .into_iter()
            .map(|(x, m)| {
                let mut free = m;
                free[0] = 0;
                let lift: BpElt = [(free, 1i64 << m[0])].into_iter().collect();
                let mut out: FxHashMap<Gen, BpElt> = FxHashMap::default();
                for (z, op) in inverse.get(&x).into_iter().flatten() {
                    bp.apply(op, &lift, k + 1, out.entry(*z).or_default());
                }
                out.retain(|_, e| !e.is_empty());
                out
            })
            .collect();
        let values = Arc::new(values);
        cache.insert((a, u, k), Arc::clone(&values));
        values
    };

    for u in 0..=max_u {
        for a in 0..=max_s {
            for k in 0..=(max_s - a) as u32 {
                let src = basis(a, u, k);
                if src.is_empty() {
                    continue;
                }
                let tgt = basis(a + 1, u, k);
                let tgt_index: FxHashMap<(Gen, QMono), usize> =
                    tgt.iter().enumerate().map(|(i, &b)| (b, i)).collect();
                let src_index: FxHashMap<(Gen, QMono), usize> =
                    src.iter().enumerate().map(|(i, &b)| (b, i)).collect();
                let vector = |values: &FxHashMap<Gen, BpElt>,
                              j: u32,
                              index: &FxHashMap<(Gen, QMono), usize>,
                              dim: usize| {
                    let mut v = FpVector::new(TWO, dim);
                    for (z, e) in values {
                        for m in layer(e, j) {
                            v.add_basis_element(index[&(*z, m)], 1);
                        }
                    }
                    v
                };

                // The image of δ_0 into the source, and the kernel of δ_0 out of it.
                let mut image = Echelon::default();
                if a > 0 {
                    for values in coboundaries(a - 1, u, k).iter() {
                        image.insert(vector(values, k, &src_index, src.len()));
                    }
                }
                let values = coboundaries(a, u, k);
                let kernel = Echelon::kernel(
                    values
                        .iter()
                        .map(|v| vector(v, k, &tgt_index, tgt.len()))
                        .collect(),
                );
                let mut reps = Vec::new();
                for v in kernel {
                    if image.insert(v.clone()) {
                        reps.push(v);
                    }
                }
                if reps.is_empty() {
                    continue;
                }

                // d_1 of each representative, modulo the image of δ_0 at length k + 1.
                let tgt1 = basis(a + 1, u, k + 1);
                let tgt1_index: FxHashMap<(Gen, QMono), usize> =
                    tgt1.iter().enumerate().map(|(i, &b)| (b, i)).collect();
                let mut target = Echelon::default();
                for values in coboundaries(a, u, k + 1).iter() {
                    target.insert(vector(values, k + 1, &tgt1_index, tgt1.len()));
                }
                let mut rank = 0;
                for rep in &reps {
                    let mut sum: FxHashMap<Gen, BpElt> = FxHashMap::default();
                    for (i, _) in rep.iter_nonzero() {
                        for (z, e) in &values[i] {
                            let acc = sum.entry(*z).or_default();
                            for (m, &c) in e {
                                add_term(acc, *m, c);
                            }
                        }
                    }
                    for e in sum.values() {
                        assert!(layer(e, k).is_empty(), "a representative is not a cocycle");
                    }
                    if target.insert(vector(&sum, k + 1, &tgt1_index, tgt1.len())) {
                        rank += 1;
                    }
                }
                out.insert((2 * u - a, a + k as i32, u), (reps.len(), rank));
            }
        }
    }
    out
}

/// Row-echelon form over $\mathbb{F}_2$.
#[derive(Default)]
struct Echelon {
    /// Rows with distinct pivots.
    rows: Vec<(usize, FpVector)>,
}

impl Echelon {
    /// Reduces `v` against the rows.
    fn reduce(&self, v: &mut FpVector) {
        for (pivot, row) in &self.rows {
            if v.entry(*pivot) == 1 {
                v.add(row, 1);
            }
        }
    }

    /// Adds `v`, returning whether it was independent of the rows.
    fn insert(&mut self, mut v: FpVector) -> bool {
        self.reduce(&mut v);
        match v.first_nonzero() {
            Some((pivot, _)) => {
                for (_, row) in &mut self.rows {
                    if row.entry(pivot) == 1 {
                        row.add(&v, 1);
                    }
                }
                self.rows.push((pivot, v));
                true
            }
            None => false,
        }
    }

    /// A basis of the combinations of `rows` that sum to zero.
    fn kernel(rows: Vec<FpVector>) -> Vec<FpVector> {
        let n = rows.len();
        let width = rows.first().map_or(0, |r| r.len());
        let mut echelon = Self::default();
        let mut out = Vec::new();
        for (i, row) in rows.into_iter().enumerate() {
            let mut v = FpVector::new(TWO, width + n);
            for (j, _) in row.iter_nonzero() {
                v.set_entry(j, 1);
            }
            v.set_entry(width + i, 1);
            echelon.reduce(&mut v);
            match v.first_nonzero() {
                Some((pivot, _)) if pivot < width => {
                    for (_, r) in &mut echelon.rows {
                        if r.entry(pivot) == 1 {
                            r.add(&v, 1);
                        }
                    }
                    echelon.rows.push((pivot, v));
                }
                _ => {
                    let mut k = FpVector::new(TWO, n);
                    for (j, _) in v.iter_nonzero() {
                        k.set_entry(j - width, 1);
                    }
                    out.push(k);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra::Algebra;

    use crate::chain_complex::ChainComplex;

    /// The backbone through weight `max_u` and homological degree `max_s`.
    fn backbone(
        max_s: i32,
        max_u: i32,
    ) -> crate::resolution::Resolution<
        crate::chain_complex::FiniteChainComplex<algebra::module::FDModule<MilnorAlgebra>>,
    > {
        let module = Arc::new(crate::algebraic_novikov::unit_module());
        let cc = Arc::new(crate::chain_complex::FiniteChainComplex::ccdz(module));
        let res = crate::resolution::Resolution::new(cc);
        res.compute_through_bidegree(Bidegree::s_t(max_s, max_u));
        res
    }

    #[test]
    fn multinomials_mod_4() {
        let exact = |a: u64, b: u64| -> u64 {
            let mut r = 1u128;
            for i in 0..b {
                r = r * (a + b - i) as u128 / (i + 1) as u128;
            }
            (r % 4) as u64
        };
        for a in 0..40 {
            for b in 0..40 {
                assert_eq!(binomial_mod4(a, b) as u64, exact(a, b), "({a}, {b})");
            }
        }
    }

    /// Composing then acting agrees with acting twice, modulo $I^{k+2}$.
    #[test]
    fn composition_is_the_action() {
        let max = 14;
        let bp = BpModI2::new(max);
        let ops: Vec<PackedXi> = (1..=max)
            .flat_map(|d| {
                let algebra = MilnorAlgebra::new(TWO, false);
                algebra.compute_basis(d);
                (0..algebra.dimension(d))
                    .map(|i| packed_p_part(&algebra, d, i))
                    .collect::<Vec<_>>()
            })
            .collect();
        let q = monomials(4, |_| max);
        let mut checked = 0;
        for &s in &ops {
            for &t in &ops {
                let deg = packed_degree(s) + packed_degree(t);
                if deg > max {
                    continue;
                }
                let x = Op {
                    lead: [s].into_iter().collect(),
                    first: FxHashSet::default(),
                };
                let y = Op {
                    lead: [t].into_iter().collect(),
                    first: [(1, t)].into_iter().collect(),
                };
                let xy = bp.compose_sum([(&x, &y)], max);
                for d in deg..=max {
                    for m in &q[d as usize] {
                        let mut km = [0u16; QDOWN_VARS];
                        for (i, &e) in m.iter().enumerate() {
                            km[i] = e as u16;
                        }
                        for e0 in 0..2u16 {
                            let k = qmono_len(&km) + e0 as u32;
                            let lift: BpElt = [(km, 1i64 << e0)].into_iter().collect();
                            let mut once = BpElt::default();
                            bp.apply(&y, &lift, k + 1, &mut once);
                            let mut twice = BpElt::default();
                            bp.apply(&x, &once, k + 1, &mut twice);
                            let mut direct = BpElt::default();
                            bp.apply(&xy, &lift, k + 1, &mut direct);
                            assert_eq!(
                                layer(&twice, k).into_iter().collect::<FxHashSet<_>>(),
                                layer(&direct, k).into_iter().collect::<FxHashSet<_>>(),
                                "lead of θ{:?} ∘ θ{:?} on {m:?}",
                                unpack(s),
                                unpack(t)
                            );
                            assert_eq!(
                                layer(&twice, k + 1).into_iter().collect::<FxHashSet<_>>(),
                                layer(&direct, k + 1).into_iter().collect::<FxHashSet<_>>(),
                                "θ{:?} ∘ θ{:?} on {m:?} v0^{e0}",
                                unpack(s),
                                unpack(t)
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 1000);
    }

    /// The corrected lift squares to zero modulo $I^2$.
    #[test]
    fn lift_squares_to_zero() {
        let (max_s, max_u) = (6, 20);
        let res = backbone(max_s, max_u);
        let bp = BpModI2::new(max_u);
        let lifted = LiftedBackbone::new(&res, &bp, max_s);
        for s in 2..=max_s {
            for t in 0..=max_u {
                for l in 0..res.number_of_gens_in_bidegree(Bidegree::s_t(s, t)) {
                    assert!(lifted.error(&bp, s, (t, l)).is_empty(), "C_{s} ({t}, {l})");
                }
            }
        }
    }

    /// Agrees with Wang's algebraic Novikov tables through $t = 50$: $d_2(h_4) = h_0 h_3^2$
    /// among them, and as many $d_2$ in all.
    #[test]
    fn d2_matches_wang() {
        let (max_u, max_s) = (25, 12);
        let res = backbone(max_s + 2, max_u);
        let out = novikov_d2(&res, max_u, max_s);
        assert_eq!(out[&(15, 1, 8)], (1, 1));
        let (dims, ranks) = out
            .values()
            .fold((0, 0), |(d, r), &(dim, rank)| (d + dim, r + rank));
        assert_eq!((dims, ranks), (445, 118));
    }
}
