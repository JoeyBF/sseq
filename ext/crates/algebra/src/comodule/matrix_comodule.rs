use std::sync::Arc;

use fp::{prime::ValidPrime, vector::FpSliceMut};
use once::OnceVec;
use rustc_hash::FxHashMap;

use super::{ActionBlock, Comodule, FilteredComodule, MAX_GENERATORS, Monomial, OpSet};
use crate::algebra::{
    Algebra, MilnorAlgebra, MilnorAntipode,
    milnor_algebra::{MilnorBasisElement, PPart},
};

/// The most antidiagonals a matrix can have: one per generator.
const MAX_DIAG: usize = MAX_GENERATORS;

/// $Q/(v_0 - 1) = \mathbb{F}_2[v_1, v_2, \dots]$, with $|v_n| = 2^n - 1$, as a right comodule
/// over the dual Steenrod algebra at $p = 2$ through $\psi(v_n) = \sum_i v_i \otimes
/// \zeta_{n-i}^{2^i}$, filtered by length, with its action computed inline from Milnor matrices
/// and no table.
///
/// Expanding $\psi(v^S)$ and recording how many of the $s_n$ factors $v_n$ become $v_i$ gives a
/// matrix $x_{d,i}$, $d = n - i$, whose antidiagonal sums are $S$, whose column sums ($i \geq 1$)
/// are the exponents $S'$ of the new monomial, and whose weighted row sums
/// $M_d = \sum_i 2^i x_{d,i}$ give the term $\chi(\xi^M)$; mod 2 a matrix counts once exactly when
/// each antidiagonal splits the bits of $s_n$. This is the matrix of the Milnor product
/// $P(M) P(S')$ with output $P(S)$, so
/// $$P(R) \cdot v^S = \sum_{P(M) \in \chi(P(R))} \sum_{S'} [P(S) \in P(M) P(S')]\, v^{S'},$$
/// enumerated by [`for_each_column`] with the rows and antidiagonals fixed. Agrees with
/// `ext::algebraic_novikov::q_mod_v0_comodule`, the same comodule presented by generators.
pub struct MatrixQComodule {
    algebra: Arc<MilnorAlgebra>,
    antipode: MilnorAntipode,
    /// The number of generators, $v_1, \dots, v_n$.
    gens: usize,
    /// The basis monomials of each degree, sorted by length and then exponents.
    basis: OnceVec<Vec<Monomial>>,
    /// The index of each basis monomial in its degree.
    index: OnceVec<FxHashMap<Monomial, u32>>,
}

impl std::fmt::Display for MatrixQComodule {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Q/(v_0 - 1)")
    }
}

/// The length of $v^S$.
fn length(m: &Monomial) -> u32 {
    m.iter().map(|&e| e as u32).sum()
}

impl MatrixQComodule {
    /// The comodule over `algebra`, the Milnor algebra at $p = 2$, in degrees up to `max_degree`.
    pub fn new(algebra: Arc<MilnorAlgebra>, max_degree: i32) -> Self {
        assert_eq!(algebra.prime(), 2, "the matrix action is written for p = 2");
        let gens = (1..)
            .take_while(|&n| (1i32 << n) - 1 <= max_degree)
            .last()
            .unwrap_or(0);
        assert!(gens <= MAX_DIAG);
        Self {
            antipode: MilnorAntipode::new(Arc::clone(&algebra)),
            algebra,
            gens,
            basis: OnceVec::new(),
            index: OnceVec::new(),
        }
    }

    /// The basis monomials of `degree`.
    pub fn basis(&self, degree: i32) -> &[Monomial] {
        &self.basis[degree as usize]
    }

    /// The index of a basis monomial of `degree`.
    pub fn index(&self, degree: i32, m: &Monomial) -> Option<usize> {
        self.index[degree as usize].get(m).map(|&i| i as usize)
    }

    /// The monomials of `degree`, sorted by length and then exponents.
    fn enumerate(&self, degree: i32) -> Vec<Monomial> {
        fn fill(n: usize, gens: usize, left: i32, m: &mut Monomial, out: &mut Vec<Monomial>) {
            if n == gens {
                if left == 0 {
                    out.push(*m);
                }
                return;
            }
            let d = (1i32 << (n + 1)) - 1;
            let mut e = 0;
            while d * e <= left {
                m[n] = e as u16;
                fill(n + 1, gens, left - d * e, m, out);
                e += 1;
            }
            m[n] = 0;
        }
        let mut out = Vec::new();
        fill(0, self.gens, degree, &mut [0; MAX_GENERATORS], &mut out);
        out.sort_unstable_by_key(|m| (length(m), *m));
        out
    }

    /// $P(R) v^S$ as the monomials with odd coefficient.
    pub fn act(&self, r: PPart, m: &Monomial) -> Vec<Monomial> {
        let mut odd: FxHashMap<Monomial, bool> = FxHashMap::default();
        for &(rows, _) in self.antipode.of(r).iter() {
            for_each_column(rows, m, |cols| {
                let e = odd.entry(*cols).or_insert(false);
                *e = !*e;
            });
        }
        let mut out: Vec<Monomial> = odd.into_iter().filter(|x| x.1).map(|x| x.0).collect();
        out.sort_unstable();
        out
    }

    /// The polynomial part of an operation, or `None` if it has an exterior part.
    fn p_part(&self, op_degree: i32, op_index: usize) -> Option<PPart> {
        let MilnorBasisElement { q_part, p_part, .. } =
            self.algebra.basis_element_from_index(op_degree, op_index);
        (q_part == 0).then_some(p_part)
    }
}

/// Calls `emit` with the column sums $S'$ of every matrix with weighted row sums `rows` and
/// antidiagonal sums `antidiagonals` (the exponents of $v^S$, $v_n$ at index $n - 1$) whose
/// antidiagonals split the bits of their sums. Equal $S'$ may be emitted several times and count
/// mod 2.
///
/// The matrix is $x_{d,i}$ for $d, i \geq 0$, antidiagonal $n = d + i$. Its row $0$ holds the
/// new exponents directly and its column $0$ is $v_0$, set to 1. The antidiagonals are filled from
/// the last one down, as in the Milnor product: on reaching antidiagonal $n$, every other cell of
/// row $n$ is already filled, so its column-0 cell takes what is left of $M_n$, and row 0 takes the
/// bits of $s_n$ the other cells left.
pub fn for_each_column(rows: PPart, antidiagonals: &Monomial, mut emit: impl FnMut(&Monomial)) {
    let top = antidiagonals
        .iter()
        .rposition(|&e| e > 0)
        .map_or(0, |n| n + 1);
    // Rows past the last antidiagonal have no cells.
    if rows.len() > top {
        return;
    }
    let mut row_left = [0u32; MAX_DIAG + 1];
    for (d, left) in row_left.iter_mut().enumerate().skip(1).take(rows.len()) {
        *left = rows.get(d - 1);
    }
    let mut cols = [0u16; MAX_GENERATORS];
    fill(
        top,
        1,
        antidiagonals[top.saturating_sub(1)] as u32,
        &mut row_left,
        &mut cols,
        antidiagonals,
        &mut emit,
    );
}

/// Fills cell `(d, n - d)` of antidiagonal `n`, with `avail` the bits of $s_n$ not yet used, then
/// the rest.
fn fill(
    n: usize,
    d: usize,
    avail: u32,
    row_left: &mut [u32; MAX_DIAG + 1],
    cols: &mut Monomial,
    antidiagonals: &Monomial,
    emit: &mut impl FnMut(&Monomial),
) {
    if n == 0 {
        emit(cols);
        return;
    }
    if d == n {
        // Column 0 finishes row n; row 0 takes the remaining bits.
        let x = row_left[n];
        if x & !avail != 0 {
            return;
        }
        let rest = avail & !x;
        row_left[n] = 0;
        cols[n - 1] += rest as u16;
        let next = if n > 1 {
            antidiagonals[n - 2] as u32
        } else {
            0
        };
        fill(n - 1, 1, next, row_left, cols, antidiagonals, emit);
        cols[n - 1] -= rest as u16;
        row_left[n] = x;
        return;
    }
    // Interior cell (d, i), i = n - d >= 1: a sub-mask of `avail` that row d can still afford.
    let i = n - d;
    let budget = row_left[d] >> i;
    let mut x = avail;
    loop {
        if x <= budget {
            row_left[d] -= x << i;
            cols[i - 1] += x as u16;
            fill(n, d + 1, avail & !x, row_left, cols, antidiagonals, emit);
            cols[i - 1] -= x as u16;
            row_left[d] += x << i;
        }
        if x == 0 {
            break;
        }
        x = (x - 1) & avail;
    }
}

/// Calls `emit(M, S')` for every matrix with antidiagonal sums `antidiagonals` whose antidiagonals
/// split the bits of their sums and whose weighted row sums $M$ stay at most `cap` entrywise: the
/// whole coaction $\psi(v^S) = \sum v^{S'} \otimes \chi(\xi^M)$, up to the cap, in one pass.
pub fn for_each_matrix(
    antidiagonals: &Monomial,
    cap: PPart,
    mut emit: impl FnMut(PPart, &Monomial),
) {
    let top = antidiagonals
        .iter()
        .rposition(|&e| e > 0)
        .map_or(0, |n| n + 1);
    let mut rows = [0u32; MAX_DIAG + 1];
    let mut cols = [0u16; MAX_GENERATORS];
    let mut cap_rows = [0u32; MAX_DIAG + 1];
    for (d, c) in cap_rows
        .iter_mut()
        .enumerate()
        .skip(1)
        .take(MAX_DIAG.min(PPart::MAX_LEN))
    {
        *c = cap.get(d - 1);
    }
    fill_free(
        top,
        1,
        antidiagonals[top.saturating_sub(1)] as u32,
        &mut rows,
        &cap_rows,
        &mut cols,
        antidiagonals,
        &mut emit,
    );
}

/// [`fill`] with the rows free: cell `(d, n - d)` takes any sub-mask of `avail` keeping row `d`
/// within its cap, column 0 included.
#[allow(clippy::too_many_arguments)]
fn fill_free(
    n: usize,
    d: usize,
    avail: u32,
    rows: &mut [u32; MAX_DIAG + 1],
    cap: &[u32; MAX_DIAG + 1],
    cols: &mut Monomial,
    antidiagonals: &Monomial,
    emit: &mut impl FnMut(PPart, &Monomial),
) {
    if n == 0 {
        let mut m = PPart::zero();
        for (i, &r) in rows.iter().enumerate().skip(1).take(PPart::MAX_LEN) {
            if r > 0 {
                m.set(i - 1, r);
            }
        }
        emit(m, cols);
        return;
    }
    if d > n {
        // Row 0 takes the bits left.
        cols[n - 1] += avail as u16;
        let next = if n > 1 {
            antidiagonals[n - 2] as u32
        } else {
            0
        };
        fill_free(n - 1, 1, next, rows, cap, cols, antidiagonals, emit);
        cols[n - 1] -= avail as u16;
        return;
    }
    // Cell (d, i), i = n - d >= 0; i = 0 is column 0, v_0, set to 1.
    let i = n - d;
    if d > PPart::MAX_LEN {
        fill_free(n, d + 1, avail, rows, cap, cols, antidiagonals, emit);
        return;
    }
    let room = cap[d] - rows[d];
    let mut x = avail;
    loop {
        if (x as u64) << i <= room as u64 {
            rows[d] += x << i;
            if i > 0 {
                cols[i - 1] += x as u16;
            }
            fill_free(n, d + 1, avail & !x, rows, cap, cols, antidiagonals, emit);
            if i > 0 {
                cols[i - 1] -= x as u16;
            }
            rows[d] -= x << i;
        }
        if x == 0 {
            break;
        }
        x = (x - 1) & avail;
    }
}

impl Comodule for MatrixQComodule {
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
            .map(|(n, &e)| format!("v{}^{e}", n + 1))
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
        for out in self.act(r, &m) {
            let i = self
                .index(degree - op_degree, &out)
                .expect("a term outside the basis");
            result.add_basis_element(i, coeff);
        }
    }

    fn act_batch(&self, ops: &OpSet, degree: i32) -> ActionBlock {
        use maybe_rayon::prelude::*;
        // For each P(M), the operations P(R) with P(M) in chi(P(R)), and a cap on M.
        let mut inverse: FxHashMap<PPart, Vec<u32>> = FxHashMap::default();
        let mut cap = PPart::zero();
        for (l, op) in ops.up_to_degree(degree) {
            let Some(r) = self.p_part(op.degree, op.index as usize) else {
                continue;
            };
            for &(m, _) in self.antipode.of(r).iter() {
                inverse.entry(m).or_default().push(l as u32);
                for i in 0..m.len() {
                    cap.set(i, cap.get(i).max(m.get(i)));
                }
            }
        }
        let rows: Vec<Vec<(u32, u32, u32)>> = self
            .basis(degree)
            .into_maybe_par_iter()
            .map(|s| {
                let mut terms: Vec<(u32, u32)> = Vec::new();
                for_each_matrix(s, cap, |m, cols| {
                    if let Some(ls) = inverse.get(&m) {
                        let target = degree - self.antipode.degree(m);
                        let t = self.index(target, cols).expect("a term outside the basis") as u32;
                        terms.extend(ls.iter().map(|&l| (l, t)));
                    }
                });
                // Each pair counts mod 2.
                terms.sort_unstable();
                let mut row = Vec::with_capacity(terms.len());
                for chunk in terms.chunk_by(|a, b| a == b) {
                    if chunk.len() % 2 == 1 {
                        row.push((chunk[0].0, chunk[0].1, 1));
                    }
                }
                row
            })
            .collect();
        let mut block = ActionBlock::new(ValidPrime::new(2));
        for row in rows {
            block.push_row(row);
        }
        block
    }
}

/// A comodule over the dual of the Milnor algebra that can also act through its antipode-free
/// side: $m \cdot P(M) = \chi(P(M)) \cdot m$, the right action of the left coaction it comes from.
///
/// With a backbone whose differential coefficients are rewritten as $\chi(a)$ once, the
/// coboundary needs only this action and never the antipode.
pub trait ConjugateAction: Comodule<Algebra = MilnorAlgebra> {
    /// Calls `emit` with the index and coefficient of each term of $\chi(P(M))$ applied to basis
    /// element `idx` of `degree`, in degree `degree - |P(M)|`.
    fn act_by_conjugate(&self, m: PPart, degree: i32, idx: usize, emit: &mut dyn FnMut(usize, u32));
}

/// $v^S \cdot P(M) = \sum_{S'} [P(S) \in P(M) P(S')]\, v^{S'}$: one [`for_each_column`] with the
/// rows fixed to $M$.
impl ConjugateAction for MatrixQComodule {
    fn act_by_conjugate(
        &self,
        m: PPart,
        degree: i32,
        idx: usize,
        emit: &mut dyn FnMut(usize, u32),
    ) {
        let target = degree - self.antipode.degree(m);
        let mut terms: Vec<Monomial> = Vec::new();
        for_each_column(m, &self.basis[degree as usize][idx], |cols| {
            terms.push(*cols)
        });
        terms.sort_unstable();
        for chunk in terms.chunk_by(|a, b| a == b) {
            if chunk.len() % 2 == 1 {
                emit(
                    self.index(target, &chunk[0])
                        .expect("a term outside the basis"),
                    1,
                );
            }
        }
    }
}

/// Filtered by length, which the coaction does not raise.
impl FilteredComodule for MatrixQComodule {
    fn filtration(&self, degree: i32, idx: usize) -> i32 {
        length(&self.basis[degree as usize][idx]) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comodule::check_associativity;

    #[test]
    fn matrix_q_is_a_module() {
        let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(2), false));
        let m = MatrixQComodule::new(alg, 24);
        check_associativity(&m, 24).unwrap();
    }
}
