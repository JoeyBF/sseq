use std::sync::Arc;

use fp::{
    prime::{Prime, ValidPrime},
    vector::FpSliceMut,
};
use once::OnceVec;
use rustc_hash::FxHashMap;

use super::{ActionBlock, Comodule, FilteredComodule, OpSet};
use crate::algebra::{
    Algebra, MilnorAlgebra,
    combinatorics::xi_degrees,
    milnor_algebra::{MilnorBasisElement, PPart},
};

/// The most generators a [`PolynomialComodule`] can have.
pub const MAX_GENERATORS: usize = 16;

/// A monomial in the generators of a [`PolynomialComodule`], as exponents.
pub type Monomial = [u16; MAX_GENERATORS];

/// A sum of monomials with nonzero coefficients.
type Poly = Vec<(Monomial, u32)>;

/// A right comodule algebra over the dual Steenrod algebra that is polynomial on generators,
/// given by the coaction on the generators, optionally cut down to one weight.
///
/// Since $\psi$ is a ring map, the action of $A$ satisfies the Cartan formula
/// $P(R)(xy) = \sum_{S + T = R} P(S)x \cdot P(T)y$, and because $\psi(x^{p^j})$ is the
/// $p^j$-th power of $\psi(x)$, $P(T)$ kills $x^{p^j}$ unless $p^j$ divides $T$, and sends it to
/// $(P(T/p^j)x)^{p^j}$ if it does. [`Self::act`] peels a power $x_n^{p^j}$ off a monomial and recurses
/// on the rest, so only the $P(S)$ with $S \leq R$ for a requested $R$ are ever computed.
///
/// The coaction may involve only the polynomial part of the dual Steenrod algebra; operations
/// with an exterior part act as zero.
pub struct PolynomialComodule {
    algebra: Arc<MilnorAlgebra>,
    name: String,
    /// The degree of each generator.
    degrees: Vec<i32>,
    /// The weight of each generator, preserved by the coaction.
    weights: Vec<u32>,
    /// If set, the comodule is the span of the monomials of this weight.
    weight: Option<u32>,
    /// `actions[n][R]` is $P(R) x_n$, for the $R$ where it is nonzero.
    actions: Vec<FxHashMap<PPart, Poly>>,
    /// The basis monomials of each degree, sorted.
    basis: OnceVec<Vec<Monomial>>,
    /// The index of each basis monomial in its degree.
    index: OnceVec<FxHashMap<Monomial, u32>>,
}

impl std::fmt::Display for PolynomialComodule {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A generator of a [`PolynomialComodule`]: its name, degree and weight.
#[derive(Clone, Debug)]
pub struct Generator {
    pub name: String,
    pub degree: i32,
    pub weight: u32,
}

impl PolynomialComodule {
    /// The comodule on `generators`, with the coaction on generator `n` given in `coaction[n]`
    /// as terms `(m, R, c)` of $\psi(x_n) = \sum c\, m \otimes \xi^R$, and cut down to the
    /// monomials of total weight `weight` if it is set.
    ///
    /// The coaction must be counital, with the term $x_n \otimes 1$, and must preserve degree.
    /// With `weight` it must preserve weight; without, it must not raise weight, which makes the
    /// weight a filtration (see [`FilteredComodule`]), and every generator must have positive
    /// degree so that each degree is finite.
    pub fn new(
        algebra: Arc<MilnorAlgebra>,
        name: String,
        generators: &[Generator],
        coaction: &[Vec<(Monomial, Vec<u32>, u32)>],
        weight: Option<u32>,
    ) -> Self {
        assert!(generators.len() <= MAX_GENERATORS);
        assert_eq!(generators.len(), coaction.len());
        let p = algebra.prime();
        let degrees: Vec<i32> = generators.iter().map(|g| g.degree).collect();
        let weights: Vec<u32> = generators.iter().map(|g| g.weight).collect();
        assert!(weight.is_some() || degrees.iter().all(|&d| d > 0));
        let xi = xi_degrees(p);
        let q = algebra.q();
        let mut actions = vec![FxHashMap::default(); generators.len()];
        for (n, terms) in coaction.iter().enumerate() {
            for (m, r, c) in terms {
                let c = c % p.as_u32();
                if c == 0 {
                    continue;
                }
                let r_degree: i32 = r.iter().zip(xi).map(|(&e, &d)| e as i32 * d * q).sum();
                assert_eq!(
                    monomial_degree(&degrees, m) + r_degree,
                    degrees[n],
                    "a term of the coaction on generator {n} has the wrong degree"
                );
                if weight.is_some() {
                    assert_eq!(monomial_weight(&weights, m), weights[n]);
                } else {
                    assert!(monomial_weight(&weights, m) <= weights[n]);
                }
                let entry: &mut Poly = actions[n].entry(PPart::from_slice(r)).or_default();
                entry.push((*m, c));
            }
        }
        for (n, a) in actions.iter_mut().enumerate() {
            for poly in a.values_mut() {
                *poly = normalize(p, std::mem::take(poly));
            }
            let mut x = [0u16; MAX_GENERATORS];
            x[n] = 1;
            assert_eq!(
                a.get(&PPart::zero()),
                Some(&vec![(x, 1)]),
                "the coaction on generator {n} is not counital"
            );
        }
        Self {
            algebra,
            name,
            degrees,
            weights,
            weight,
            actions,
            basis: OnceVec::new(),
            index: OnceVec::new(),
        }
    }

    /// The right subcomodule $\mathbb{F}_p[\xi_k^{p^{e_k}}]$ of the polynomial part of the dual
    /// Steenrod algebra, with $e_k$ = `exponents(k)`, on the generators of degree at most
    /// `max_degree`, with $\psi = \Delta$, so $\psi(\xi_k^{p^e}) = \sum_i \xi_{k-i}^{p^{i+e}}
    /// \otimes \xi_i^{p^e}$.
    ///
    /// With every $e_k = 0$ this is the coalgebra itself; with $e = (2, 1, 0, \dots)$ it is
    /// $\mathbb{F}_2 \square_{A(1)_*} A_*$, whose Cotor is $\mathrm{Ext}_{A(1)}$. The exponents must
    /// make it a subcomodule: $e_{k-i} \leq e_k + i$.
    pub fn dual_steenrod_subalgebra(
        algebra: Arc<MilnorAlgebra>,
        exponents: impl Fn(usize) -> u32,
        max_degree: i32,
    ) -> Self {
        let p = algebra.prime().as_u32();
        let xi = xi_degrees(algebra.prime());
        let q = algebra.q();
        let gens: Vec<(usize, u32)> = (1..)
            .map(|k| (k, exponents(k)))
            .take_while(|&(k, e)| {
                xi[k - 1] * q <= max_degree || p.pow(e) as i32 * xi[k - 1] * q <= max_degree
            })
            .filter(|&(k, e)| p.pow(e) as i32 * xi[k - 1] * q <= max_degree)
            .collect();
        let generators: Vec<Generator> = gens
            .iter()
            .map(|&(k, e)| Generator {
                name: format!("xi{k}^{}", p.pow(e)),
                degree: p.pow(e) as i32 * xi[k - 1] * q,
                weight: 0,
            })
            .collect();
        let coaction: Vec<Vec<(Monomial, Vec<u32>, u32)>> = gens
            .iter()
            .map(|&(k, e)| {
                (0..=k)
                    .map(|i| {
                        // xi_{k-i}^{p^{i+e}} as a power of a generator, and xi_i^{p^e}.
                        let mut m = [0u16; MAX_GENERATORS];
                        if i < k {
                            let j = k - i;
                            let g = gens.iter().position(|&(kk, _)| kk == j).unwrap();
                            let ej = gens[g].1;
                            assert!(ej <= i as u32 + e, "not a subcomodule at xi_{j}");
                            m[g] = p.pow(i as u32 + e - ej) as u16;
                        }
                        let mut r = vec![0u32; i];
                        if i > 0 {
                            r[i - 1] = p.pow(e);
                        }
                        (m, r, 1)
                    })
                    .collect()
            })
            .collect();
        Self::new(
            algebra,
            "dual Steenrod subalgebra".to_string(),
            &generators,
            &coaction,
            None,
        )
    }

    /// The basis monomials of `degree`.
    pub fn basis(&self, degree: i32) -> &[Monomial] {
        &self.basis[degree as usize]
    }

    /// The index of a basis monomial of `degree`.
    pub fn index(&self, degree: i32, m: &Monomial) -> Option<usize> {
        self.index[degree as usize].get(m).map(|&i| i as usize)
    }

    /// The degree of a monomial.
    pub fn degree_of(&self, m: &Monomial) -> i32 {
        monomial_degree(&self.degrees, m)
    }

    /// The monomials of `degree` and, if set, of weight `self.weight`, sorted by weight and then
    /// exponents.
    fn enumerate(&self, degree: i32) -> Vec<Monomial> {
        let mut out = Vec::new();
        let mut m = [0u16; MAX_GENERATORS];
        self.enumerate_from(0, degree, self.weight, &mut m, &mut out);
        out.sort_unstable_by_key(|m| (monomial_weight(&self.weights, m), *m));
        out
    }

    /// Fills generators `n..` of `m` in every way that uses up `degree` and, if set, `weight`.
    fn enumerate_from(
        &self,
        n: usize,
        degree: i32,
        weight: Option<u32>,
        m: &mut Monomial,
        out: &mut Vec<Monomial>,
    ) {
        if n == self.degrees.len() {
            if degree == 0 && weight.is_none_or(|w| w == 0) {
                out.push(*m);
            }
            return;
        }
        let (d, w) = (self.degrees[n], self.weights[n]);
        let mut e = 0u32;
        loop {
            let used_d = d * e as i32;
            let used_w = w * e;
            if used_d > degree || weight.is_some_and(|wt| used_w > wt) {
                break;
            }
            m[n] = e as u16;
            self.enumerate_from(n + 1, degree - used_d, weight.map(|wt| wt - used_w), m, out);
            if d == 0 && w == 0 {
                break;
            }
            e += 1;
        }
        m[n] = 0;
    }

    /// $P(R) m$, with `memo` caching every $(S, m')$ met on the way.
    pub fn act(
        &self,
        r: PPart,
        r_degree: i32,
        m: &Monomial,
        memo: &mut FxHashMap<(PPart, Monomial), Poly>,
    ) -> Poly {
        if r_degree == 0 {
            return vec![(*m, 1)];
        }
        if r_degree > self.degree_of(m) {
            return Vec::new();
        }
        if let Some(v) = memo.get(&(r, *m)) {
            return v.clone();
        }
        let p = self.prime();
        let pu = p.as_u32();
        // The largest chunk x_n^{p^j}: the last generator present, its top p-adic digit.
        let n = (0..self.degrees.len()).rev().find(|&n| m[n] > 0).unwrap();
        let mut pj = 1u32;
        while pj * pu <= m[n] as u32 {
            pj *= pu;
        }
        let mut parent = *m;
        parent[n] -= pj as u16;

        let mut acc: FxHashMap<Monomial, u32> = FxHashMap::default();
        let xi = xi_degrees(p);
        let q = self.algebra.q();
        // Every T <= R with p^j dividing it, as p^j T'.
        let len = r.len();
        let mut t = PPart::zero();
        loop {
            let divisible = (0..len).all(|i| t.get(i).is_multiple_of(pj));
            if divisible {
                let mut t_small = PPart::zero();
                let mut s = PPart::zero();
                let mut t_degree = 0;
                for (i, &xi_i) in xi.iter().enumerate().take(len) {
                    t_small.set(i, t.get(i) / pj);
                    s.set(i, r.get(i) - t.get(i));
                    t_degree += t.get(i) as i32 * xi_i * q;
                }
                let chunk_image: Option<Poly> = if t_degree == 0 {
                    let mut c = [0u16; MAX_GENERATORS];
                    c[n] = 1;
                    Some(vec![(c, 1)])
                } else {
                    self.actions[n].get(&t_small).cloned()
                };
                if let Some(chunk_image) = chunk_image {
                    let left = self.act(s, r_degree - t_degree, &parent, memo);
                    for (a, ca) in &left {
                        for (b, cb) in &chunk_image {
                            let mut prod = *a;
                            for i in 0..MAX_GENERATORS {
                                prod[i] += b[i] * pj as u16;
                            }
                            let e = acc.entry(prod).or_insert(0);
                            *e = (*e + ca * cb) % pu;
                        }
                    }
                }
            }
            if increment(&mut t, r) {
                break;
            }
        }
        let out: Poly = normalize(p, acc.into_iter().collect());
        memo.insert((r, *m), out.clone());
        out
    }

    /// $\psi(x_n^{p^j})$, the Frobenius twist of $\psi(x_n)$, as `(R, m, c)` terms of
    /// $\sum c\, m \otimes \xi^R$.
    fn chunk_coaction(&self, n: usize, pj: u32) -> Vec<(PPart, Monomial, u32)> {
        let mut out = Vec::new();
        for (r, poly) in &self.actions[n] {
            let mut r_big = PPart::zero();
            let mut fits = true;
            for i in 0..r.len() {
                let e = r.get(i) * pj;
                if e > PPart::max_entry(i) {
                    fits = false;
                    break;
                }
                r_big.set(i, e);
            }
            if !fits {
                continue;
            }
            for (m, c) in poly {
                let mut big = *m;
                for e in &mut big {
                    *e *= pj as u16;
                }
                out.push((r_big, big, *c));
            }
        }
        out
    }

    /// The product of two coactions, keeping only the terms whose $R$ is at most `cap` in every
    /// entry: $R$ only grows under products, so the others never reach an operation below `cap`.
    fn multiply_coactions(
        &self,
        a: &[(PPart, Monomial, u32)],
        b: &[(PPart, Monomial, u32)],
        cap: PPart,
    ) -> Vec<(PPart, Monomial, u32)> {
        let pu = self.prime().as_u32();
        let mut acc: FxHashMap<(PPart, Monomial), u32> = FxHashMap::default();
        for (ra, ma, ca) in a {
            'next: for (rb, mb, cb) in b {
                let mut r = PPart::zero();
                for i in 0..cap.len() {
                    let e = ra.get(i) + rb.get(i);
                    if e > cap.get(i) {
                        continue 'next;
                    }
                    r.set(i, e);
                }
                if ra.len().max(rb.len()) > cap.len() {
                    continue;
                }
                let mut m = *ma;
                for (x, y) in m.iter_mut().zip(mb) {
                    *x += y;
                }
                let e = acc.entry((r, m)).or_insert(0);
                *e = (*e + ca * cb) % pu;
            }
        }
        acc.into_iter()
            .filter(|&(_, c)| c != 0)
            .map(|((r, m), c)| (r, m, c))
            .collect()
    }

    /// Visits `m`, of degree `degree` and weight `weight` with coaction `psi`, and the monomials
    /// obtained from it by multiplying by powers $x_n^{p^j}$ larger than every power already in
    /// it, so that each monomial is visited once, from the monomial [`Self::act`] peels it to.
    /// Returns the row of the action of `ops` on each basis monomial visited.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        m: Monomial,
        degree: i32,
        weight: u32,
        psi: &[(PPart, Monomial, u32)],
        local: &FxHashMap<PPart, u32>,
        cap: PPart,
        max_degree: i32,
    ) -> Vec<(i32, usize, Vec<(u32, u32, u32)>)> {
        use maybe_rayon::prelude::*;
        let p = self.prime().as_u32();
        let mut out = Vec::new();
        if self.weight.is_none_or(|w| w == weight)
            && let Some(idx) = self.index(degree, &m)
        {
            let mut row: Vec<(u32, u32, u32)> = psi
                .iter()
                .filter_map(|(r, target, c)| {
                    let &l = local.get(r)?;
                    let t = self.index(self.degree_of(target), target)?;
                    Some((l, t as u32, *c))
                })
                .collect();
            row.sort_unstable();
            out.push((degree, idx, row));
        }
        // The largest power already present: the last generator, its top p-adic digit.
        let last = (0..self.degrees.len()).rev().find(|&n| m[n] > 0);
        let mut children = Vec::new();
        for (n, (&d, &w)) in self
            .degrees
            .iter()
            .zip(&self.weights)
            .enumerate()
            .skip(last.unwrap_or(0))
        {
            let mut pj = 1u32;
            let mut j = 0;
            loop {
                let (cd, cw) = (degree + d * pj as i32, weight + w * pj);
                if cd > max_degree || self.weight.is_some_and(|wt| cw > wt) {
                    break;
                }
                let allowed = if Some(n) == last {
                    let e = m[n] as u32;
                    let (mut top, mut x) = (0, e);
                    while x >= p {
                        x /= p;
                        top += 1;
                    }
                    j > top || (j == top && x < p - 1)
                } else {
                    true
                };
                if allowed {
                    children.push((n, pj, cd, cw));
                }
                if d == 0 && w == 0 {
                    break;
                }
                pj *= p;
                j += 1;
            }
        }
        let rest: Vec<Vec<_>> = children
            .into_maybe_par_iter()
            .map(|(n, pj, cd, cw)| {
                let mut child = m;
                child[n] += pj as u16;
                let child_psi = self.multiply_coactions(psi, &self.chunk_coaction(n, pj), cap);
                self.walk(child, cd, cw, &child_psi, local, cap, max_degree)
            })
            .collect();
        out.extend(rest.into_iter().flatten());
        out
    }

    /// The polynomial part of an operation, or `None` if it has an exterior part.
    fn p_part(&self, op_degree: i32, op_index: usize) -> Option<PPart> {
        let MilnorBasisElement { q_part, p_part, .. } =
            self.algebra.basis_element_from_index(op_degree, op_index);
        (q_part == 0).then_some(p_part)
    }
}

/// Steps `t` through the p-parts bounded by `max`, in odometer order; returns `true` on wrapping.
fn increment(t: &mut PPart, max: PPart) -> bool {
    for i in 0..max.len() {
        if t.get(i) < max.get(i) {
            t.set(i, t.get(i) + 1);
            return false;
        }
        t.set(i, 0);
    }
    true
}

/// The degree of `m` with generators of degree `degrees`.
fn monomial_degree(degrees: &[i32], m: &Monomial) -> i32 {
    degrees.iter().zip(m).map(|(&d, &e)| d * e as i32).sum()
}

/// The weight of `m` with generators of weight `weights`.
fn monomial_weight(weights: &[u32], m: &Monomial) -> u32 {
    weights.iter().zip(m).map(|(&w, &e)| w * e as u32).sum()
}

/// Combines repeated monomials, drops zeros and sorts.
fn normalize(p: ValidPrime, mut poly: Poly) -> Poly {
    poly.sort_unstable();
    let mut out: Poly = Vec::with_capacity(poly.len());
    for (m, c) in poly {
        match out.last_mut() {
            Some((last, lc)) if *last == m => *lc = (*lc + c) % p.as_u32(),
            _ => out.push((m, c % p.as_u32())),
        }
        if out.last().is_some_and(|x| x.1 == 0) {
            out.pop();
        }
    }
    out
}

impl Comodule for PolynomialComodule {
    type Algebra = MilnorAlgebra;

    fn algebra(&self) -> Arc<MilnorAlgebra> {
        Arc::clone(&self.algebra)
    }

    fn min_degree(&self) -> i32 {
        0
    }

    fn compute_basis(&self, degree: i32) {
        self.algebra.compute_basis(degree);
        self.basis
            .extend(degree.max(-1) as usize, |d| self.enumerate(d as i32));
        self.index.extend(degree.max(-1) as usize, |d| {
            self.basis[d]
                .iter()
                .enumerate()
                .map(|(i, m)| (*m, i as u32))
                .collect()
        });
    }

    fn max_computed_degree(&self) -> i32 {
        self.basis.len() as i32 - 1
    }

    fn dimension(&self, degree: i32) -> usize {
        if degree < 0 {
            return 0;
        }
        self.basis[degree as usize].len()
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        let m = &self.basis[degree as usize][idx];
        let parts: Vec<String> = m
            .iter()
            .enumerate()
            .filter(|&(_, &e)| e > 0)
            .map(|(n, &e)| {
                if e == 1 {
                    format!("x{n}")
                } else {
                    format!("x{n}^{e}")
                }
            })
            .collect();
        if parts.is_empty() {
            "1".to_string()
        } else {
            parts.join(" ")
        }
    }

    fn act_on_basis(
        &self,
        mut result: FpSliceMut,
        coeff: u32,
        op_degree: i32,
        op_index: usize,
        degree: i32,
        idx: usize,
    ) {
        let Some(r) = self.p_part(op_degree, op_index) else {
            return;
        };
        let m = self.basis[degree as usize][idx];
        let mut memo = FxHashMap::default();
        let target = degree - op_degree;
        for (out, c) in self.act(r, op_degree, &m, &mut memo) {
            let i = self
                .index(target, &out)
                .expect("the action leaves the basis");
            result.add_basis_element(i, coeff * c % self.prime());
        }
    }

    fn act_batches(&self, ops: &OpSet, max_degree: i32) -> Vec<ActionBlock> {
        self.compute_basis(max_degree);
        let mut local: FxHashMap<PPart, u32> = FxHashMap::default();
        let mut cap = PPart::zero();
        for (l, op) in ops.up_to_degree(max_degree) {
            if let Some(r) = self.p_part(op.degree, op.index as usize) {
                local.insert(r, l as u32);
                for i in 0..r.len() {
                    cap.set(i, cap.get(i).max(r.get(i)));
                }
            }
        }
        let root = [0u16; MAX_GENERATORS];
        let rows = self.walk(
            root,
            0,
            0,
            &[(PPart::zero(), root, 1)],
            &local,
            cap,
            max_degree,
        );
        let mut by_degree: Vec<Vec<Option<Vec<(u32, u32, u32)>>>> = (0..=max_degree)
            .map(|d| vec![None; self.dimension(d)])
            .collect();
        for (d, idx, row) in rows {
            by_degree[d as usize][idx] = Some(row);
        }
        by_degree
            .into_iter()
            .map(|rows| {
                let mut block = ActionBlock::new(self.prime());
                for row in rows {
                    block.push_row(row.expect("the walk visits every basis monomial"));
                }
                block
            })
            .collect()
    }

    fn act_batch(&self, ops: &OpSet, degree: i32) -> ActionBlock {
        let mut block = ActionBlock::new(self.prime());
        let mut memo = FxHashMap::default();
        let parts: Vec<(usize, i32, Option<PPart>)> = ops
            .up_to_degree(degree)
            .map(|(local, op)| (local, op.degree, self.p_part(op.degree, op.index as usize)))
            .collect();
        for m in self.basis(degree) {
            let mut row = Vec::new();
            for &(local, op_degree, r) in &parts {
                let Some(r) = r else { continue };
                let target = degree - op_degree;
                let mut image: Vec<(u32, u32)> = self
                    .act(r, op_degree, m, &mut memo)
                    .into_iter()
                    .map(|(out, c)| {
                        let i = self
                            .index(target, &out)
                            .expect("the action leaves the basis");
                        (i as u32, c)
                    })
                    .collect();
                image.sort_unstable();
                row.extend(image.into_iter().map(|(i, c)| (local as u32, i, c)));
            }
            block.push_row(row);
        }
        block
    }
}

/// Filtered by weight, which the coaction does not raise.
impl FilteredComodule for PolynomialComodule {
    fn filtration(&self, degree: i32, idx: usize) -> i32 {
        monomial_weight(&self.weights, &self.basis[degree as usize][idx]) as i32
    }
}

#[cfg(test)]
mod tests {
    use fp::vector::FpVector;
    use rstest::rstest;

    use super::*;
    use crate::comodule::{FiniteComodule, OpId, check_associativity};

    /// The coalgebra as a polynomial comodule acts as the coregular comodule, matching $\xi^R$
    /// with the dual of $P(R)$.
    #[test]
    fn dual_steenrod_is_coregular() {
        let max = 20;
        let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(2), false));
        let poly = PolynomialComodule::dual_steenrod_subalgebra(Arc::clone(&alg), |_| 0, max);
        let reg = FiniteComodule::coregular(Arc::clone(&alg), max);
        poly.compute_basis(max);
        // The Milnor index of the monomial with exponents R.
        let to_milnor = |d: i32, m: &Monomial| -> usize {
            let r: Vec<u32> = m.iter().map(|&e| e as u32).collect();
            let len = r.iter().rposition(|&e| e > 0).map_or(0, |i| i + 1);
            let elt = MilnorBasisElement {
                q_part: 0,
                p_part: PPart::from_slice(&r[..len]),
                degree: d,
            };
            alg.basis_element_to_index(&elt)
        };
        for d in 0..=max {
            assert_eq!(poly.dimension(d), reg.dimension(d));
            for idx in 0..poly.dimension(d) {
                let m = poly.basis(d)[idx];
                for n in 1..=d {
                    for op in 0..alg.dimension(n) {
                        let mut a = FpVector::new(alg.prime(), poly.dimension(d - n));
                        poly.act_on_basis(a.as_slice_mut(), 1, n, op, d, idx);
                        let mut b = FpVector::new(alg.prime(), reg.dimension(d - n));
                        reg.act_on_basis(b.as_slice_mut(), 1, n, op, d, to_milnor(d, &m));
                        let mapped: Vec<usize> = a
                            .iter_nonzero()
                            .map(|(i, _)| to_milnor(d - n, &poly.basis(d - n)[i]))
                            .collect();
                        let mut mapped = mapped;
                        mapped.sort_unstable();
                        let direct: Vec<usize> = b.iter_nonzero().map(|(i, _)| i).collect();
                        assert_eq!(mapped, direct, "P{op} in degree {n} on {m:?}");
                    }
                }
            }
        }
    }

    #[rstest]
    #[case(2, &[0, 0, 0, 0], 24)]
    #[case(2, &[2, 1, 0, 0], 30)]
    #[case(2, &[1, 0, 0, 0], 24)]
    #[case(3, &[0, 0, 0], 60)]
    #[case(3, &[1, 0, 0], 60)]
    fn subalgebras_are_modules(#[case] p: u32, #[case] e: &[u32], #[case] max: i32) {
        let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(p), false));
        let e = e.to_vec();
        let m = PolynomialComodule::dual_steenrod_subalgebra(
            alg,
            move |k| e.get(k - 1).copied().unwrap_or(0),
            max,
        );
        check_associativity(&m, max).unwrap();
    }

    #[test]
    fn act_batch_agrees_with_act_on_basis() {
        let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(2), false));
        let m =
            PolynomialComodule::dual_steenrod_subalgebra(Arc::clone(&alg), |k| (k == 1) as u32, 24);
        m.compute_basis(24);
        let ops: OpSet = (1..=24)
            .flat_map(|n| {
                (0..alg.dimension(n)).map(move |i| OpId {
                    degree: n,
                    index: i as u32,
                })
            })
            .collect();
        for d in [12, 24] {
            let block = m.act_batch(&ops, d);
            for idx in 0..m.dimension(d) {
                for (local, op) in ops.iter().enumerate() {
                    if op.degree > d {
                        continue;
                    }
                    let mut v = FpVector::new(m.prime(), m.dimension(d - op.degree));
                    m.act_on_basis(v.as_slice_mut(), 1, op.degree, op.index as usize, d, idx);
                    let from_block: Vec<(usize, u32)> =
                        block.act(idx, local as u32).iter().collect();
                    assert_eq!(from_block, v.iter_nonzero().collect::<Vec<_>>());
                }
            }
        }
    }

    /// The coalgebra filtered by giving $\xi_k$ weight $2^{k-1}$, which the coaction does not
    /// raise.
    fn weighted_dual_steenrod(max: i32) -> PolynomialComodule {
        let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(2), false));
        let plain = PolynomialComodule::dual_steenrod_subalgebra(Arc::clone(&alg), |_| 0, max);
        let gens: Vec<Generator> = plain
            .degrees
            .iter()
            .enumerate()
            .map(|(k, &d)| Generator {
                name: format!("xi{}", k + 1),
                degree: d,
                weight: 1 << k,
            })
            .collect();
        // Rebuild the coaction on generators from the action of the plain comodule.
        let coaction: Vec<Vec<(Monomial, Vec<u32>, u32)>> = plain
            .actions
            .iter()
            .map(|a| {
                a.iter()
                    .flat_map(|(r, poly)| {
                        let r: Vec<u32> = r.iter().collect();
                        poly.iter().map(move |(m, c)| (*m, r.clone(), *c))
                    })
                    .collect()
            })
            .collect();
        PolynomialComodule::new(alg, "weighted".to_string(), &gens, &coaction, None)
    }

    #[test]
    fn sub_and_quotient_are_modules() {
        use crate::comodule::{FilteredComodule, QuotientComodule, SubComodule};
        let max = 20;
        let m = Arc::new(weighted_dual_steenrod(max));
        m.compute_basis(max);
        for d in 0..=max {
            for i in 1..m.dimension(d) {
                assert!(m.filtration(d, i - 1) <= m.filtration(d, i));
            }
        }
        let alg = m.algebra();
        let ops: OpSet = (1..=max)
            .flat_map(|n| {
                (0..alg.dimension(n)).map(move |i| OpId {
                    degree: n,
                    index: i as u32,
                })
            })
            .collect();
        for f in [1, 2, 3, 5] {
            let sub = SubComodule::new(Arc::clone(&m), f);
            let quo = QuotientComodule::new(Arc::clone(&m), f);
            check_associativity(&sub, max).unwrap();
            check_associativity(&quo, max).unwrap();
            for d in [max - 1, max] {
                assert_eq!(sub.dimension(d) + quo.dimension(d), m.dimension(d));
                let check = |c: &dyn Fn(&mut FpVector, i32, u32, usize),
                             block: &ActionBlock,
                             dim: usize,
                             tgt: &dyn Fn(i32) -> usize| {
                    for idx in 0..dim {
                        for (local, op) in ops.iter().enumerate() {
                            if op.degree > d {
                                continue;
                            }
                            let mut v = FpVector::new(alg.prime(), tgt(d - op.degree));
                            c(&mut v, op.degree, op.index, idx);
                            let from_block: Vec<(usize, u32)> =
                                block.act(idx, local as u32).iter().collect();
                            assert_eq!(from_block, v.iter_nonzero().collect::<Vec<_>>(), "f = {f}");
                        }
                    }
                };
                check(
                    &|v, n, i, idx| sub.act_on_basis(v.as_slice_mut(), 1, n, i as usize, d, idx),
                    &sub.act_batch(&ops, d),
                    sub.dimension(d),
                    &|e| sub.dimension(e),
                );
                check(
                    &|v, n, i, idx| quo.act_on_basis(v.as_slice_mut(), 1, n, i as usize, d, idx),
                    &quo.act_batch(&ops, d),
                    quo.dimension(d),
                    &|e| quo.dimension(e),
                );
            }
        }
    }

    #[test]
    fn act_batches_agrees_with_act_batch() {
        let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(2), false));
        let max = 24;
        alg.compute_basis(max);
        let ops: OpSet = (0..=max)
            .flat_map(|n| {
                (0..alg.dimension(n)).map(move |i| OpId {
                    degree: n,
                    index: i as u32,
                })
            })
            .collect();
        for m in [
            PolynomialComodule::dual_steenrod_subalgebra(Arc::clone(&alg), |_| 0, max),
            PolynomialComodule::dual_steenrod_subalgebra(
                Arc::clone(&alg),
                |k| (k == 1) as u32,
                max,
            ),
            weighted_dual_steenrod(max),
        ] {
            let all = m.act_batches(&ops, max);
            for d in 0..=max {
                let one = m.act_batch(&ops, d);
                for idx in 0..m.dimension(d) {
                    let a: Vec<_> = all[d as usize]
                        .row(idx)
                        .map(|(l, e)| (l, e.iter().collect::<Vec<_>>()))
                        .collect();
                    let b: Vec<_> = one
                        .row(idx)
                        .map(|(l, e)| (l, e.iter().collect::<Vec<_>>()))
                        .collect();
                    assert_eq!(a, b, "degree {d}, index {idx}");
                }
            }
        }
    }
}
