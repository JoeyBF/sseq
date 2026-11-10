//! The $C\tau$ Adams $E_2$ page of a finite complex, from the classical backbone of the sphere.

use std::collections::BTreeMap;

use algebra::{
    Algebra, MilnorAlgebra,
    milnor_algebra::{MilnorBasisElement, PPart},
    module::{FDModule, Module},
};
use fp::{matrix::Matrix, prime::TWO, vector::FpVector};
use rustc_hash::FxHashMap;
use sseq::coordinates::Bidegree;

use crate::{
    algebraic_novikov::{
        ActMemo, Convention, PackedXi, QDOWN_VARS, QDown, QMono, TrigradedDims, chi_table,
        differential_slice, invert_slice, monomials, pack, qmono_len, unpack, v_degree,
    },
    chain_complex::FreeChainComplex,
};

/// A basis element of a finite module over $A/\tau$: its degree, index and motivic weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Cell {
    /// The topological degree.
    t: i32,
    /// The index in its degree.
    idx: usize,
    /// The motivic weight.
    w: i32,
}

impl Cell {
    /// $t - 2w$, the excess of the degree over twice the weight.
    fn excess(&self) -> i32 {
        self.t - 2 * self.w
    }
}

/// The motivic weight of an operation of $A/\tau$: $|\xi_i| = 2^i - 1$ and $|\tau_i| = 2^i - 1$.
fn op_weight(elt: &MilnorBasisElement) -> i32 {
    let p: i32 = elt
        .p_part
        .iter()
        .enumerate()
        .map(|(i, r)| r as i32 * v_degree(i + 1))
        .sum();
    let q: i32 = (0..32)
        .filter(|i| elt.q_part >> i & 1 == 1)
        .map(|i| v_degree(i as usize))
        .sum();
    p + q
}

/// The motivic weight of every basis element of `module`, as a map from `(t, idx)`.
///
/// Each class lowest in its component of the action graph gets weight $\lfloor t/2 \rfloor$, the
/// weight of a cell $S^{t, \lfloor t/2 \rfloor}$, and the action determines the rest.
///
/// # Panics
///
/// If the action is not homogeneous for any such weights, or the basis is not homogeneous.
fn cell_weights(module: &FDModule<MilnorAlgebra>) -> BTreeMap<(i32, usize), i32> {
    let algebra = module.algebra();
    let (min, max) = (module.min_degree(), module.max_degree().unwrap());
    algebra.compute_basis(max - min);
    // Edges m -> m' with the weight of the operation.
    let mut edges: BTreeMap<(i32, usize), Vec<((i32, usize), i32)>> = BTreeMap::new();
    for t in min..=max {
        for idx in 0..module.dimension(t) {
            for op_deg in 1..=max - t {
                for op_idx in 0..algebra.dimension(op_deg) {
                    let w = op_weight(&algebra.basis_element_from_index(op_deg, op_idx));
                    let mut out = FpVector::new(TWO, module.dimension(t + op_deg));
                    module.act_on_basis(out.as_slice_mut(), 1, op_deg, op_idx, t, idx);
                    for (j, _) in out.iter_nonzero() {
                        edges
                            .entry((t, idx))
                            .or_default()
                            .push(((t + op_deg, j), w));
                        edges
                            .entry((t + op_deg, j))
                            .or_default()
                            .push(((t, idx), -w));
                    }
                }
            }
        }
    }
    let mut weights: BTreeMap<(i32, usize), i32> = BTreeMap::new();
    for t in min..=max {
        for idx in 0..module.dimension(t) {
            if weights.contains_key(&(t, idx)) {
                continue;
            }
            weights.insert((t, idx), t.div_euclid(2));
            let mut stack = vec![(t, idx)];
            while let Some(m) = stack.pop() {
                let w = weights[&m];
                for &(m2, dw) in edges.get(&m).into_iter().flatten() {
                    match weights.get(&m2) {
                        Some(&w2) => assert_eq!(w2, w + dw, "the action is not weight-homogeneous"),
                        None => {
                            weights.insert(m2, w + dw);
                            stack.push(m2);
                        }
                    }
                }
            }
        }
    }
    weights
}

/// The transpose of `op` acting on `module`: `out[(t, idx)]` lists the `(t', idx')` with
/// `(t, idx)` in `op` applied to `(t', idx')`.
fn transpose_action(
    module: &FDModule<MilnorAlgebra>,
    op_deg: i32,
    op_idx: usize,
) -> FxHashMap<(i32, usize), Vec<(i32, usize)>> {
    let mut out: FxHashMap<(i32, usize), Vec<(i32, usize)>> = FxHashMap::default();
    let max = module.max_degree().unwrap();
    for t in module.min_degree()..=max - op_deg {
        for idx in 0..module.dimension(t) {
            let mut v = FpVector::new(TWO, module.dimension(t + op_deg));
            module.act_on_basis(v.as_slice_mut(), 1, op_deg, op_idx, t, idx);
            for (j, _) in v.iter_nonzero() {
                out.entry((t + op_deg, j)).or_default().push((t, idx));
            }
        }
    }
    out
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

/// A cochain $x^* \otimes q \otimes m^*$: the filtration, the backbone generator `(t, idx)`, the
/// monomial of $Q$ and the cell of the module.
type Cochain = (i32, (i32, u32), QMono, Cell);

/// $\mathrm{Ext}_{A/\tau}(M, \mathbb{F}_2)$ for a finite $A/\tau$-module $M$, by (stem, $s$, weight),
/// through stem `max_n` and $s \leq$ `max_s`, from a resolution `res` of the classical
/// $\mathbb{F}_2$.
///
/// The untwisted resolution of $\mathbb{F}_2$ over $A/\tau$, tensored with $M$ and untwisted again,
/// is free on $C \otimes \Gamma \otimes M$. Its cochains are $C^\vee \otimes Q \otimes M^\vee$,
/// with two parts to the coboundary:
///
/// - the backbone part, in which each term $P(R)$ of the backbone differential acts on
///   $Q \otimes M^\vee$ through the coproduct, $\sum_{R' + R'' = R} \rho(P(R')) \otimes
///   \chi(P(R''))^T$;
/// - the Koszul part, $q \otimes m^* \mapsto \sum_i v_i q \otimes Q_i^T m^*$, in the same
///   filtration.
///
/// When $M = \mathbb{F}_2$ this is the untwisted complex of
/// [`ctau_e2_untwisted`](crate::algebraic_novikov::ctau_e2_untwisted).
pub fn ctau_e2_of_module<CC>(
    res: &CC,
    module: &FDModule<MilnorAlgebra>,
    max_n: i32,
    max_s: i32,
) -> TrigradedDims
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra>,
{
    let algebra = module.algebra();
    let weights = cell_weights(module);
    let cells: Vec<Cell> = weights
        .iter()
        .map(|(&(t, idx), &w)| Cell { t, idx, w })
        .collect();
    let min_excess = cells.iter().map(Cell::excess).min().unwrap_or(0);
    let min_cell_w = cells.iter().map(|c| c.w).min().unwrap_or(0);
    // n = 2u + e - a, with a <= max_s + 1 and n <= max_n + 1.
    let max_u = (max_n + 1 + max_s + 1 - min_excess).div_euclid(2);
    let max_q = max_u - min_cell_w;
    assert!(
        res.has_computed_bidegree(Bidegree::s_t(max_s + 2, max_u - min_cell_w)),
        "the backbone must be resolved through ({}, {})",
        max_s + 2,
        max_u - min_cell_w
    );

    let q_down = QDown::new(max_s as u32 + 1, |_| max_q.max(0), Convention::Conjugate);
    let memo = ActMemo::default();
    let chi = chi_table(max_q.max(0));
    let q_basis = monomials(max_s as u32 + 1, |_| max_q.max(0));

    // The transposed action of each P(S) and Q_i on M, built on first use.
    let mut transposes: FxHashMap<(i32, usize), FxHashMap<(i32, usize), Vec<(i32, usize)>>> =
        FxHashMap::default();
    let max_m = module.max_degree().unwrap() - module.min_degree();
    algebra.compute_basis(max_m.max(0));
    let mut transpose = |elt: MilnorBasisElement| -> FxHashMap<(i32, usize), Vec<(i32, usize)>> {
        if elt.degree > max_m {
            return FxHashMap::default();
        }
        let idx = algebra.basis_element_to_index(&elt);
        transposes
            .entry((elt.degree, idx))
            .or_insert_with(|| transpose_action(module, elt.degree, idx))
            .clone()
    };
    let p_elt = |r: PackedXi| {
        let p_part: Vec<_> = unpack(r).into_iter().map(|x| x as _).collect();
        let mut elt = MilnorBasisElement {
            q_part: 0,
            p_part: PPart::from_slice(&p_part),
            degree: 0,
        };
        algebra.compute_degree(&mut elt);
        elt
    };
    let q_elt = |i: usize| {
        let mut elt = MilnorBasisElement {
            q_part: 1 << i,
            p_part: PPart::from_slice(&[]),
            degree: 0,
        };
        algebra.compute_degree(&mut elt);
        elt
    };
    let cell_of: FxHashMap<(i32, usize), Cell> = cells.iter().map(|c| ((c.t, c.idx), *c)).collect();

    let slices: Vec<_> = (0..=max_s + 1)
        .map(|a| {
            invert_slice(
                &res.algebra(),
                &differential_slice(res, a, max_u - min_cell_w),
            )
        })
        .collect();

    let mut out = BTreeMap::new();
    for u in 0..=max_u {
        // The cochains of weight u, grouped by (s, n).
        let mut groups: BTreeMap<(i32, i32), Vec<Cochain>> = BTreeMap::new();
        for a in 0..=max_s + 1 {
            for &cell in &cells {
                let n = 2 * u + cell.excess() - a;
                if n > max_n + 1 || n < -1 {
                    continue;
                }
                for t_x in 0..=u - cell.w {
                    let d = u - cell.w - t_x;
                    let gens = res.number_of_gens_in_bidegree(Bidegree::s_t(a, t_x));
                    if gens == 0 || d as usize >= q_basis.len() {
                        continue;
                    }
                    for x in 0..gens as u32 {
                        for m in &q_basis[d as usize] {
                            let len: u32 = m.iter().sum();
                            for k in len..=(max_s + 1 - a) as u32 {
                                let mut q = [0u16; QDOWN_VARS];
                                for (i, &e) in m.iter().enumerate() {
                                    q[i] = e as u16;
                                }
                                q[0] += (k - len) as u16;
                                groups.entry((a + k as i32, n)).or_default().push((
                                    a,
                                    (t_x, x),
                                    q,
                                    cell,
                                ));
                            }
                        }
                    }
                }
            }
        }
        let index: FxHashMap<(i32, i32), FxHashMap<Cochain, usize>> = groups
            .iter()
            .map(|(&key, v)| (key, v.iter().enumerate().map(|(i, &c)| (c, i)).collect()))
            .collect();

        let mut rank_out: FxHashMap<(i32, i32), usize> = FxHashMap::default();
        for (&(s, n), sources) in &groups {
            if s > max_s {
                continue;
            }
            let Some(target) = index.get(&(s + 1, n - 1)) else {
                rank_out.insert((s, n), 0);
                continue;
            };
            let mut rows = Vec::with_capacity(sources.len());
            for &(a, (t_x, x), q, cell) in sources {
                let mut row = FpVector::new(TWO, target.len());
                let mut hit = |c: Cochain| {
                    if let Some(&j) = target.get(&c) {
                        row.add_basis_element(j, 1);
                    }
                };
                // The Koszul part.
                for i in 0..QDOWN_VARS {
                    let elt = q_elt(i);
                    if elt.degree > max_m {
                        break;
                    }
                    let mut q2 = q;
                    q2[i] += 1;
                    for &(t2, idx2) in transpose(elt)
                        .get(&(cell.t, cell.idx))
                        .into_iter()
                        .flatten()
                    {
                        hit((a, (t_x, x), q2, cell_of[&(t2, idx2)]));
                    }
                }
                // The backbone part.
                for (r, zs) in slices[a as usize].get(&(t_x, x)).into_iter().flatten() {
                    for r1 in splits(*r) {
                        let r2 = r - r1;
                        let qs = q_down.act_memo(r1, &q, &memo, 1 << 24);
                        if qs.is_empty() {
                            continue;
                        }
                        // The identity is P(0); the module stores no action in degree 0.
                        let mut ms = Vec::new();
                        if r2 == 0 {
                            ms.push((cell.t, cell.idx));
                        }
                        for &s2 in chi.get(&r2).into_iter().flatten().filter(|&&s2| s2 != 0) {
                            for &m2 in transpose(p_elt(s2))
                                .get(&(cell.t, cell.idx))
                                .into_iter()
                                .flatten()
                            {
                                ms.push(m2);
                            }
                        }
                        for &q2 in qs.iter() {
                            // The action keeps the length, leaving v_0 implicit.
                            let mut q2 = q2;
                            q2[0] += (qmono_len(&q) - qmono_len(&q2)) as u16;
                            for &m2 in &ms {
                                for &(t_z, z) in zs {
                                    hit((a + 1, (t_z, z), q2, cell_of[&m2]));
                                }
                            }
                        }
                    }
                }
                rows.push(row);
            }
            let mut matrix = Matrix::from_rows(TWO, rows, target.len());
            rank_out.insert((s, n), matrix.row_reduce());
        }
        for (&(s, n), sources) in &groups {
            if s > max_s || n > max_n || n < 0 {
                continue;
            }
            let incoming = if s > 0 {
                rank_out.get(&(s - 1, n + 1)).copied().unwrap_or(0)
            } else {
                0
            };
            let dim = sources.len() - rank_out[&(s, n)] - incoming;
            if dim > 0 {
                out.insert((n, s, u), dim);
            }
        }
    }
    TrigradedDims(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::chain_complex::ChainComplex;

    /// An $A/\tau$-module from its generators and actions, in the module-spec syntax.
    fn ctau_module(gens: &str, actions: &[&str]) -> FDModule<MilnorAlgebra> {
        let json = serde_json::json!({
            "p": 2,
            "type": "finite dimensional module",
            "gens": serde_json::from_str::<serde_json::Value>(gens).unwrap(),
            "actions": actions,
        });
        let algebra = Arc::new(MilnorAlgebra::motivic_mod_tau(Default::default(), false));
        FDModule::from_json(algebra, &json).unwrap()
    }

    /// Compares with a minimal resolution of the module over $A/\tau$ in every (stem, $s$).
    fn check(module: FDModule<MilnorAlgebra>, max_n: i32, max_s: i32) {
        let backbone =
            crate::nassau::Resolution::new(Arc::new(crate::algebraic_novikov::unit_module()));
        backbone.compute_through_bidegree(Bidegree::s_t(max_s + 2, (max_n + max_s) / 2 + 4));
        let ours = ctau_e2_of_module(&backbone, &module, max_n, max_s).bigraded();

        let direct = crate::nassau::Resolution::new(Arc::new(module));
        direct.compute_through_stem(Bidegree::n_s(max_n, max_s));
        let mut theirs = BTreeMap::new();
        for s in 0..=max_s {
            for n in 0..=max_n {
                let d = direct.number_of_gens_in_bidegree(Bidegree::n_s(n, s));
                if d > 0 {
                    theirs.insert((n, s), d);
                }
            }
        }
        assert_eq!(ours, theirs);
    }

    #[test]
    fn sphere() {
        check(ctau_module(r#"{"x0": 0}"#, &[]), 30, 12);
    }

    #[test]
    fn moore() {
        check(
            ctau_module(r#"{"x0": 0, "x1": 1}"#, &["Q_0 x0 = x1"]),
            30,
            12,
        );
    }

    #[test]
    fn c_eta() {
        check(
            ctau_module(r#"{"x0": 0, "x2": 2}"#, &["P(1) x0 = x2"]),
            30,
            12,
        );
    }

    #[test]
    fn c_nu() {
        check(
            ctau_module(r#"{"x0": 0, "x4": 4}"#, &["P(2) x0 = x4"]),
            30,
            12,
        );
    }

    #[test]
    fn moore_smash_c_eta() {
        check(
            ctau_module(
                r#"{"x0": 0, "x1": 1, "x2": 2, "x3": 3}"#,
                &["Q_0 x0 = x1", "P(1) x0 = x2", "Q_0 x2 = x3", "P(1) x1 = x3"],
            ),
            30,
            12,
        );
    }

    /// The same comparisons further out; slow in debug builds.
    #[test]
    #[ignore]
    fn further_out() {
        check(
            ctau_module(r#"{"x0": 0, "x4": 4}"#, &["P(2) x0 = x4"]),
            60,
            24,
        );
        check(
            ctau_module(
                r#"{"x0": 0, "x1": 1, "x2": 2, "x3": 3}"#,
                &["Q_0 x0 = x1", "P(1) x0 = x2", "Q_0 x2 = x3", "P(1) x1 = x3"],
            ),
            60,
            24,
        );
    }
}
