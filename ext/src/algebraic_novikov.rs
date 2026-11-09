//! The $C\tau$ Adams $E_2$ page from classical data, via Miller's square.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::Arc};

use algebra::{
    Algebra, MilnorAlgebra,
    milnor_algebra::{MilnorBasisElement, PPart},
    module::{FDModule, OperationGeneratorPair},
};
use bivec::BiVec;
use rustc_hash::{FxHashMap, FxHashSet};
use sseq::coordinates::Bidegree;

use crate::chain_complex::{ChainComplex, FreeChainComplex};

/// Which generators of $P_*$ appear in the coaction on $Q$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convention {
    /// $\psi(v_n) = \sum_i \xi_{n-i}^{2^i} \otimes v_i$.
    Xi,
    /// $\psi(v_n) = \sum_i \zeta_{n-i}^{2^i} \otimes v_i$ with $\zeta_j = \chi \xi_j$.
    Conjugate,
}

/// A monomial in the $\xi_j$, as exponents of $\xi_1, \xi_2, \dots$, trimmed.
type XiMonomial = Vec<u32>;

/// A monomial in the $v_i$, as exponents of $v_0, v_1, \dots$, trimmed.
pub type VMonomial = Vec<u32>;

/// A polynomial over $\mathbb{F}_2$ in the $\xi_j$: the set of monomials with coefficient 1.
type XiPoly = FxHashSet<XiMonomial>;

/// Drops trailing zeros, so exponent vectors compare and hash canonically.
fn trim(mut v: Vec<u32>) -> Vec<u32> {
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

/// The exponent vector of the product of two monomials.
fn add_exponents(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut r = vec![0; a.len().max(b.len())];
    for (i, &x) in a.iter().enumerate() {
        r[i] += x;
    }
    for (i, &x) in b.iter().enumerate() {
        r[i] += x;
    }
    r
}

/// Adds `x` to a set of terms with coefficient 1, over $\mathbb{F}_2$.
fn toggle<T: Eq + std::hash::Hash>(set: &mut FxHashSet<T>, x: T) {
    if !set.remove(&x) {
        set.insert(x);
    }
}

/// The product of two polynomials.
fn poly_mul(a: &XiPoly, b: &XiPoly) -> XiPoly {
    let mut r = XiPoly::default();
    for x in a {
        for y in b {
            toggle(&mut r, add_exponents(x, y));
        }
    }
    r
}

/// $\xi_j$ as a polynomial; $\xi_0 = 1$.
fn xi(j: usize) -> XiPoly {
    let mut m = vec![0; j];
    if j > 0 {
        m[j - 1] = 1;
    }
    std::iter::once(m).collect()
}

/// The Frobenius $f \mapsto f^{2^i}$.
fn frobenius(f: &XiPoly, i: u32) -> XiPoly {
    f.iter()
        .map(|m| m.iter().map(|&e| e << i).collect())
        .collect()
}

/// $\chi \xi_j$ for `j` in `0..=max`, from $\sum_{i=0}^{j} \xi_{j-i}^{2^i} \chi\xi_i = 0$.
fn conjugates(max: usize) -> Vec<XiPoly> {
    let mut chi: Vec<XiPoly> = vec![xi(0)];
    for j in 1..=max {
        let mut c = XiPoly::default();
        for (i, chi_i) in chi.iter().enumerate() {
            for m in poly_mul(&frobenius(&xi(j - i), i as u32), chi_i) {
                toggle(&mut c, m);
            }
        }
        chi.push(c);
    }
    chi
}

/// $|v_i| = 2^i - 1$ in halved degrees.
pub fn v_degree(i: usize) -> i32 {
    (1 << i) - 1
}

/// The degree of a monomial in the $v_i$.
pub fn monomial_degree(m: &[u32]) -> i32 {
    m.iter()
        .enumerate()
        .map(|(i, &e)| e as i32 * v_degree(i))
        .sum()
}

/// The monomials free of $v_0$ with length at most `max_len` and degree at most `max_degree`,
/// bucketed by degree and sorted by length within each bucket.
///
/// $m \mapsto v_0^{k - |m|} m$ identifies those of length at most `k` with a basis of $Q_k$, so
/// each $Q_k$ is a prefix of every bucket.
pub fn monomials(max_len: u32, max_degree: i32) -> Vec<Vec<VMonomial>> {
    // The largest i with |v_i| <= max_degree.
    let top = (0..)
        .take_while(|&i| v_degree(i) <= max_degree)
        .last()
        .unwrap_or(0);
    let mut out = vec![Vec::new(); max_degree as usize + 1];
    let mut current = vec![0u32; top + 1];
    /// Chooses the exponent of `v_i`, then recurses on `v_{i-1}`, stopping before `v_0`.
    fn rec(
        i: usize,
        remaining: u32,
        degree: i32,
        max_degree: i32,
        current: &mut Vec<u32>,
        out: &mut Vec<Vec<VMonomial>>,
    ) {
        if i == 0 {
            out[degree as usize].push(trim(current.clone()));
            return;
        }
        let d = v_degree(i);
        let mut e = 0;
        while e <= remaining && degree + e as i32 * d <= max_degree {
            current[i] = e;
            rec(
                i - 1,
                remaining - e,
                degree + e as i32 * d,
                max_degree,
                current,
                out,
            );
            e += 1;
        }
        current[i] = 0;
    }
    rec(top, max_len, 0, max_degree, &mut current, &mut out);
    for bucket in &mut out {
        bucket.sort_by_key(|m| (m.iter().sum::<u32>(), m.clone()));
    }
    out
}

/// The coaction on $Q$, memoised on monomials free of $v_0$.
///
/// $v_0$ is primitive, so $\psi(v_0^e m) = v_0^e \psi(m)$ and one table serves every $Q_k$.
pub struct Coaction {
    /// `factors[n]` is $\psi(v_n)$ as a list of $(\text{coefficient}, i)$ meaning
    /// $\text{coefficient} \otimes v_i$.
    factors: Vec<Vec<(XiPoly, usize)>>,
    /// $\psi(m)$ for the $v_0$-free monomials computed so far.
    memo: RefCell<FxHashMap<VMonomial, Rc<CoactionValue>>>,
}

/// $\psi(m)$ as the set of $(R, m')$ with $\xi^R \otimes m'$ having coefficient 1.
pub type CoactionValue = FxHashSet<(XiMonomial, VMonomial)>;

impl Coaction {
    /// The coaction on monomials of degree at most `max_degree`.
    pub fn new(max_degree: i32, convention: Convention) -> Self {
        let max_n = (0..)
            .take_while(|&i| v_degree(i) <= max_degree)
            .last()
            .unwrap_or(0);
        let gens: Vec<XiPoly> = match convention {
            Convention::Xi => (0..=max_n).map(xi).collect(),
            Convention::Conjugate => conjugates(max_n),
        };
        let factors = (0..=max_n)
            .map(|n| {
                (0..=n)
                    .map(|i| (frobenius(&gens[n - i], i as u32), i))
                    .collect()
            })
            .collect();
        Self {
            factors,
            memo: RefCell::default(),
        }
    }

    /// $\psi(m)$ for a monomial $m$ with no $v_0$, as $\psi(v_n) \psi(m / v_n)$ for its top $v_n$.
    pub fn v0_free(&self, m: &[u32]) -> Rc<CoactionValue> {
        if let Some(v) = self.memo.borrow().get(m) {
            return Rc::clone(v);
        }
        let value = match m.iter().rposition(|&e| e > 0) {
            None => std::iter::once((Vec::new(), Vec::new())).collect(),
            Some(n) => {
                let mut rest = m.to_vec();
                rest[n] -= 1;
                let rest = self.v0_free(&trim(rest));
                let mut next = CoactionValue::default();
                for (r, mm) in rest.iter() {
                    for (poly, i) in &self.factors[n] {
                        let mut mm2 = mm.clone();
                        if mm2.len() <= *i {
                            mm2.resize(i + 1, 0);
                        }
                        mm2[*i] += 1;
                        for s in poly {
                            toggle(&mut next, (trim(add_exponents(r, s)), mm2.clone()));
                        }
                    }
                }
                next
            }
        };
        let value = Rc::new(value);
        self.memo.borrow_mut().insert(m.to_vec(), Rc::clone(&value));
        value
    }
}

/// The action of $A$ on the bases of $Q_k^\vee$ for every `k <= max_len`, in degrees
/// `<= max_degree`.
///
/// The basis in each degree is dual to [`monomials`], whose length ordering makes the basis of
/// each $Q_k^\vee$ a prefix. $P(R)$ sends $m'^*$ to the sum of the $m^*$ for which
/// $\xi^R \otimes m'$ appears in $\psi(m)$. Since $v_0$ is primitive this does not depend on $k$,
/// so restricting to a prefix gives $Q_k^\vee$.
pub struct QDual {
    /// `prefix[d][j]` is the number of basis elements of degree `d` of length at most `j`.
    prefix: Vec<Vec<usize>>,
    /// `actions[d][i]` maps the operation `(op_deg, op_idx)` to the outputs it hits from basis
    /// element `i` of degree `d`, in increasing order, each with coefficient 1.
    actions: Vec<Vec<FxHashMap<(i32, usize), Vec<usize>>>>,
}

impl QDual {
    /// The basis action table under `coaction`.
    pub fn new(
        algebra: &MilnorAlgebra,
        max_len: u32,
        max_degree: i32,
        coaction: &Coaction,
    ) -> Self {
        let basis = monomials(max_len, max_degree);
        algebra.compute_basis(max_degree);
        let index: FxHashMap<&VMonomial, usize> = basis
            .iter()
            .flat_map(|b| b.iter().enumerate().map(|(i, m)| (m, i)))
            .collect();
        let prefix = basis
            .iter()
            .map(|b| {
                (0..=max_len)
                    .map(|j| b.partition_point(|m| m.iter().sum::<u32>() <= j))
                    .collect()
            })
            .collect();
        let mut actions: Vec<Vec<_>> = basis
            .iter()
            .map(|b| vec![FxHashMap::default(); b.len()])
            .collect();
        for (out_deg, bucket) in basis.iter().enumerate() {
            let out_deg = out_deg as i32;
            for (out_idx, m) in bucket.iter().enumerate() {
                for (r, m_in) in coaction.v0_free(m).iter() {
                    let in_deg = monomial_degree(m_in);
                    let op_deg = out_deg - in_deg;
                    if op_deg == 0 {
                        continue;
                    }
                    let mut free = m_in.clone();
                    if let Some(e) = free.first_mut() {
                        *e = 0;
                    }
                    let op_idx = algebra.basis_element_to_index(&MilnorBasisElement {
                        q_part: 0,
                        p_part: PPart::from_slice(r),
                        degree: op_deg,
                    });
                    actions[in_deg as usize][index[&trim(free)]]
                        .entry((op_deg, op_idx))
                        .or_insert_with(Vec::new)
                        .push(out_idx);
                }
            }
        }
        Self { prefix, actions }
    }

    /// The dimension of $Q_k^\vee$ in degree `d`, zero outside the truncation.
    pub fn dimension(&self, d: i32, k: u32) -> usize {
        usize::try_from(d)
            .ok()
            .and_then(|d| self.prefix.get(d))
            .map_or(0, |p| p[k as usize])
    }

    /// The basis elements of $Q_k^\vee$ hit by operation `(op_deg, op_idx)` from basis element
    /// `idx` of degree `d`.
    pub fn act(&self, op_deg: i32, op_idx: usize, d: i32, idx: usize, k: u32) -> &[usize] {
        let outs = self.actions[d as usize][idx]
            .get(&(op_deg, op_idx))
            .map_or(&[][..], Vec::as_slice);
        let dim = self.dimension(d + op_deg, k);
        &outs[..outs.partition_point(|&o| o < dim)]
    }

    /// $Q_k^\vee$ as a module.
    pub fn module(&self, algebra: Arc<MilnorAlgebra>, k: u32) -> FDModule<MilnorAlgebra> {
        let top = (0..self.prefix.len() as i32)
            .rev()
            .find(|&d| self.dimension(d, k) > 0)
            .unwrap_or(0);
        let mut graded_dimension = BiVec::new(0);
        for d in 0..=top {
            graded_dimension.push(self.dimension(d, k));
        }
        let mut module = FDModule::new(algebra, format!("Q_{k}^dual"), graded_dimension);
        for in_deg in 0..=top {
            for in_idx in 0..self.dimension(in_deg, k) {
                for &(op_deg, op_idx) in self.actions[in_deg as usize][in_idx].keys() {
                    if in_deg + op_deg > top {
                        continue;
                    }
                    let outs = self.act(op_deg, op_idx, in_deg, in_idx, k);
                    let v = module.action_mut(op_deg, op_idx, in_deg, in_idx);
                    for &out in outs {
                        v.add_basis_element(out, 1);
                    }
                }
            }
        }
        module
    }
}

/// The $A$-module $Q_k^\vee$ truncated to degrees `<= max_degree`, in halved degrees.
///
/// $\Sigma^{-k} Q_k$ is a $P_*$-comodule in even degrees, and $P_*$ is $A_*$ with every degree
/// doubled, so halving degrees makes it an $A_*$-comodule with $|v_i| = 2^i - 1$. Its dual is
/// infinite, but the part above `max_degree` is a submodule, so truncating leaves
/// $\mathrm{Ext}^{a, u}$ unchanged for $u \leq$ `max_degree`.
///
/// [`QDual`] describes the action.
pub fn q_dual_module(
    algebra: Arc<MilnorAlgebra>,
    k: u32,
    max_degree: i32,
    coaction: &Coaction,
) -> FDModule<MilnorAlgebra> {
    QDual::new(&algebra, k, max_degree, coaction).module(algebra, k)
}

/// The trigraded dimensions of $\mathrm{Ext}_{A/\tau}(\mathbb{F}_2)$, keyed by (stem, s, weight).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TrigradedDims(pub BTreeMap<(i32, i32, i32), usize>);

impl TrigradedDims {
    /// Sum over weights, keyed by (stem, s).
    pub fn bigraded(&self) -> BTreeMap<(i32, i32), usize> {
        let mut out = BTreeMap::new();
        for (&(n, s, _), &d) in &self.0 {
            *out.entry((n, s)).or_default() += d;
        }
        out
    }
}

/// $\mathrm{Ext}_{A/\tau}(\mathbb{F}_2)$ in stems `0..=max_n` and filtrations `0..=max_s`,
/// computed as $\bigoplus_k \mathrm{Ext}_P(Q_k)$ with one classical resolution per $k$.
///
/// The extension $E(Q_0, Q_1, \dots) \to A/\tau \to P$ gives a Cartan–Eilenberg spectral
/// sequence $\mathrm{Ext}_P(\mathbb{F}_2, Q) \Rightarrow \mathrm{Ext}_{A/\tau}(\mathbb{F}_2)$ with
/// $Q = \mathbb{F}_2[v_0, v_1, \dots]$, which collapses for weight reasons (Gheorghe–Wang–Xu).
/// A class of $\mathrm{Ext}_A^{a, u}(Q_k^\vee)$ (see [`q_dual_module`]) has stem $2u - a$,
/// filtration $a + k$ and motivic weight $u$.
///
/// `resolve` resolves a finite module through the given classical (stem, s) and returns the
/// number of generators at a classical $(s, t)$.
pub fn ctau_e2_with<F>(
    max_n: i32,
    max_s: i32,
    convention: Convention,
    mut resolve: F,
) -> TrigradedDims
where
    F: FnMut(Arc<FDModule<MilnorAlgebra>>, Bidegree) -> Box<dyn Fn(Bidegree) -> usize>,
{
    let algebra = Arc::new(MilnorAlgebra::new(fp::prime::TWO, false));
    // n = 2u - a <= max_n and a <= max_s give u <= (max_n + max_s) / 2.
    let max_u = (max_n + max_s) / 2;
    let table = QDual::new(
        &algebra,
        max_s as u32,
        max_u,
        &Coaction::new(max_u, convention),
    );
    let mut dims = TrigradedDims::default();
    for k in 0..=max_s {
        let max_a = max_s - k;
        let module = Arc::new(table.module(Arc::clone(&algebra), k as u32));
        let gens = resolve(module, Bidegree::n_s(max_u, max_a));
        for a in 0..=max_a {
            for u in a..=max_u {
                let n = 2 * u - a;
                if n > max_n {
                    continue;
                }
                let d = gens(Bidegree::s_t(a, u));
                if d > 0 {
                    dims.0.insert((n, a + k, u), d);
                }
            }
        }
    }
    dims
}

/// [`ctau_e2_with`] using Nassau's algorithm for each $Q_k^\vee$.
pub fn ctau_e2(max_n: i32, max_s: i32, convention: Convention) -> TrigradedDims {
    ctau_e2_with(max_n, max_s, convention, |module, through| {
        let res = crate::nassau::Resolution::new(module);
        res.compute_through_stem(through);
        Box::new(move |b| res.number_of_gens_in_bidegree(b))
    })
}

/// [`ctau_e2_with`] by the untwisted tensor resolution on one shared resolution of $\mathbb{F}_2$.
///
/// $P_\bullet \otimes Q_k^\vee$ is a free, non-minimal resolution of $Q_k^\vee$, and after
/// untwisting the diagonal action its cochains have basis $x_i \otimes m_\alpha$ and coboundary
/// $\delta_{(i, \alpha), (l, \gamma)} = [m_\alpha](\chi(a_{li}) \cdot m_\gamma)$, where
/// $d z_l = \sum_i a_{li} x_i$. Since $A_*$ is commutative, acting by $\chi(P(R))$ on
/// $Q_k^\vee$ is pairing $P(R)$ with the coaction whose coefficients are conjugated, which is the
/// [`Convention::Conjugate`] table; that table is not a module, but only its basis actions are
/// read here.
pub fn ctau_e2_untwisted(max_n: i32, max_s: i32) -> TrigradedDims {
    let res = crate::nassau::Resolution::new(Arc::new(unit_module()));
    res.compute_through_stem(untwisted_backbone_range(max_n, max_s));
    ctau_e2_untwisted_on(&res, max_n, max_s)
}

/// The classical $\mathbb{F}_2$, the module a backbone for [`ctau_e2_untwisted_on`] resolves.
pub fn unit_module() -> FDModule<MilnorAlgebra> {
    let mut unit = BiVec::new(0);
    unit.push(1);
    FDModule::new(
        Arc::new(MilnorAlgebra::new(fp::prime::TWO, false)),
        "F2".into(),
        unit,
    )
}

/// The classical (stem, s) a backbone must be resolved through for [`ctau_e2_untwisted_on`].
///
/// The weight $u$ runs up to `(max_n + max_s) / 2` and δ out of filtration $a$ reads $P_{a+1}$.
pub fn untwisted_backbone_range(max_n: i32, max_s: i32) -> Bidegree {
    Bidegree::n_s((max_n + max_s) / 2, max_s + 1)
}

/// [`ctau_e2_untwisted`] on an existing resolution `res` of the classical $\mathbb{F}_2$, computed
/// through [`untwisted_backbone_range`].
pub fn ctau_e2_untwisted_on<CC>(res: &CC, max_n: i32, max_s: i32) -> TrigradedDims
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    use fp::matrix::Matrix;
    use maybe_rayon::prelude::*;

    let p = fp::prime::TWO;
    let algebra = res.algebra();
    let max_u = (max_n + max_s) / 2;
    let range = untwisted_backbone_range(max_n, max_s);
    assert!(
        res.has_computed_bidegree(Bidegree::s_t(range.s(), max_u)),
        "the backbone must be resolved through {range}"
    );

    // The terms of each d z_l, read once and shared by every k and u.
    let terms: Vec<Vec<Vec<Vec<OperationGeneratorPair>>>> = (0..=max_s)
        .map(|a| {
            let d = res.differential(a + 1);
            let source = res.module(a);
            (0..=max_u)
                .map(|t| {
                    let gens = res.number_of_gens_in_bidegree(Bidegree::s_t(a + 1, t));
                    (0..gens)
                        .map(|l| {
                            d.output(t, l)
                                .iter_nonzero()
                                .map(|(idx, _)| source.index_to_op_gen(t, idx))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    let twisted = QDual::new(
        &algebra,
        max_s as u32,
        max_u,
        &Coaction::new(max_u, Convention::Conjugate),
    );
    // One cochain complex per (k, u), each independent of the others.
    let pairs: Vec<(i32, i32)> = (0..=max_s)
        .flat_map(|k| (0..=max_u).map(move |u| (k, u)))
        .collect();
    let complex = |&(k, u): &(i32, i32)| {
        let max_a = max_s - k;
        let k_u = k as u32;
        let mut found = Vec::new();
        // Offsets of the blocks x_i ⊗ Q_k^∨ in the cochains of filtration a at weight u.
        let offsets = |a: i32| {
            let mut table = Vec::new();
            let mut total = 0;
            for t in 0..=u {
                let gens = res.number_of_gens_in_bidegree(Bidegree::s_t(a, t));
                table.push((total, gens));
                total += gens * twisted.dimension(u - t, k_u);
            }
            (table, total)
        };
        let cochains: Vec<_> = (0..=max_a + 1).map(offsets).collect();
        let mut ranks = vec![0; max_a as usize + 1];
        for a in 0..=max_a {
            let (src, src_dim) = &cochains[a as usize];
            let (tgt, tgt_dim) = &cochains[a as usize + 1];
            if *src_dim == 0 || *tgt_dim == 0 {
                continue;
            }
            let mut matrix = Matrix::new(p, *tgt_dim, *src_dim);
            for t_l in 0..=u {
                let (tgt_off, _) = tgt[t_l as usize];
                let block = twisted.dimension(u - t_l, k_u);
                for (l, dz) in terms[a as usize][t_l as usize].iter().enumerate() {
                    for gamma in 0..block {
                        let mut row = matrix.row_mut(tgt_off + l * block + gamma);
                        for og in dz {
                            let out_dim = twisted.dimension(u - og.generator_degree, k_u);
                            let start =
                                src[og.generator_degree as usize].0 + og.generator_index * out_dim;
                            for &out in twisted.act(
                                og.operation_degree,
                                og.operation_index,
                                u - t_l,
                                gamma,
                                k_u,
                            ) {
                                row.add_basis_element(start + out, 1);
                            }
                        }
                    }
                }
            }
            ranks[a as usize] = matrix.row_reduce();
        }
        for a in 0..=max_a {
            let n = 2 * u - a;
            if n > max_n || n < 0 {
                continue;
            }
            let prev = if a > 0 { ranks[a as usize - 1] } else { 0 };
            let d = cochains[a as usize].1 - ranks[a as usize] - prev;
            if d > 0 {
                found.push(((n, a + k, u), d));
            }
        }
        found
    };
    let mut dims = TrigradedDims::default();
    for found in pairs
        .into_maybe_par_iter()
        .map(|x| complex(&x))
        .collect::<Vec<_>>()
    {
        dims.0.extend(found);
    }
    dims
}

/// $\mathrm{Ext}_{A/\tau}(\mathbb{F}_2)$ by resolving over $A/\tau$ with Nassau's algorithm, keyed by
/// (stem, s). This is the reference [`ctau_e2`] is checked against.
pub fn ctau_e2_direct(max_n: i32, max_s: i32) -> BTreeMap<(i32, i32), usize> {
    use algebra::milnor_algebra::MilnorProfile;

    let algebra = Arc::new(MilnorAlgebra::motivic_mod_tau(
        MilnorProfile::default(),
        false,
    ));
    let mut graded = BiVec::new(0);
    graded.push(1);
    let module = FDModule::new(algebra, "F2".into(), graded);
    let res = crate::nassau::Resolution::new(Arc::new(module));
    res.compute_through_stem(Bidegree::n_s(max_n, max_s));
    let mut out = BTreeMap::new();
    for b in res.iter_stem() {
        let d = res.number_of_gens_in_bidegree(b);
        if d > 0 && b.n() <= max_n && b.s() <= max_s {
            out.insert((b.n(), b.s()), d);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use algebra::module::Module;

    use super::*;

    /// $\chi(\xi_2) = \xi_2 + \xi_1^3$.
    #[test]
    fn conjugate_xi_2() {
        let c = conjugates(2);
        let expected: XiPoly = [vec![0, 1], vec![3]].into_iter().collect();
        assert_eq!(c[2], expected);
    }

    /// Length at most 2, degree at most 3: $1$, $v_1$, $v_1^2$, $v_2$.
    #[test]
    fn monomial_counts() {
        let m = monomials(2, 3);
        assert_eq!(m.iter().map(Vec::len).collect::<Vec<_>>(), vec![1, 1, 1, 1]);
    }

    /// P(R)(P(S) x) = (P(R)P(S)) x on every basis element.
    fn check_associative(module: &FDModule<MilnorAlgebra>) {
        let algebra = module.algebra();
        let p = algebra.prime();
        let max = module.max_degree().unwrap();
        for in_deg in 0..=max {
            for d1 in 1..=max - in_deg {
                for d2 in 1..=max - in_deg - d1 {
                    let out_deg = in_deg + d1 + d2;
                    for i1 in 0..algebra.dimension(d1) {
                        for i2 in 0..algebra.dimension(d2) {
                            let mut prod = fp::vector::FpVector::new(p, algebra.dimension(d1 + d2));
                            algebra.multiply_basis_elements(prod.as_slice_mut(), 1, d1, i1, d2, i2);
                            for x in 0..module.dimension(in_deg) {
                                let mut lhs =
                                    fp::vector::FpVector::new(p, module.dimension(out_deg));
                                let mut mid =
                                    fp::vector::FpVector::new(p, module.dimension(in_deg + d2));
                                module.act_on_basis(mid.as_slice_mut(), 1, d2, i2, in_deg, x);
                                module.act(
                                    lhs.as_slice_mut(),
                                    1,
                                    d1,
                                    i1,
                                    in_deg + d2,
                                    mid.as_slice(),
                                );
                                let mut rhs =
                                    fp::vector::FpVector::new(p, module.dimension(out_deg));
                                module.act_by_element_on_basis(
                                    rhs.as_slice_mut(),
                                    1,
                                    d1 + d2,
                                    prod.as_slice(),
                                    in_deg,
                                    x,
                                );
                                assert_eq!(
                                    lhs, rhs,
                                    "not a module: ops ({d1},{i1})({d2},{i2}) on ({in_deg},{x})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Agrees with a minimal resolution over $A/\tau$ in every (stem, s).
    #[test]
    fn untwisted_matches_direct_resolution() {
        assert_eq!(ctau_e2_untwisted(30, 15).bigraded(), ctau_e2_direct(30, 15));
    }

    /// Agrees with a minimal resolution over $A/\tau$ in every (stem, s).
    #[test]
    fn matches_direct_resolution() {
        let ours = ctau_e2(30, 15, Convention::Xi).bigraded();
        assert_eq!(ours, ctau_e2_direct(30, 15));
    }

    /// With conjugates in the coaction the dual is not even a module, so the convention matters.
    #[test]
    #[should_panic(expected = "not a module")]
    fn conjugate_convention_is_not_a_module() {
        let algebra = Arc::new(MilnorAlgebra::new(fp::prime::TWO, false));
        check_associative(&q_dual_module(
            algebra,
            1,
            12,
            &Coaction::new(12, Convention::Conjugate),
        ));
    }

    /// Classes whose motivic weight is known land in the right weight.
    #[test]
    fn weights_of_known_classes() {
        let dims = ctau_e2(20, 5, Convention::Xi);
        // h_0, h_1, h_2, h_3, c_0, Ph_1, d_0, e_0 and τg, each alone in its (stem, s).
        for (n, s, w) in [
            (0, 1, 0),
            (1, 1, 1),
            (3, 1, 2),
            (7, 1, 4),
            (8, 3, 5),
            (9, 5, 5),
            (14, 4, 8),
            (17, 4, 10),
            (20, 4, 11),
        ] {
            assert_eq!(dims.0.get(&(n, s, w)), Some(&1), "({n}, {s}, {w})");
        }
    }

    /// The $\xi$ convention gives a module for small `k`.
    #[test]
    fn q_dual_is_a_module() {
        let algebra = Arc::new(MilnorAlgebra::new(fp::prime::TWO, false));
        for k in 1..=3 {
            let m = q_dual_module(
                Arc::clone(&algebra),
                k,
                12,
                &Coaction::new(12, Convention::Xi),
            );
            check_associative(&m);
        }
    }
}
