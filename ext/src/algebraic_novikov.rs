//! The $C\tau$ Adams $E_2$ page from classical data, via Miller's square.

use std::{collections::BTreeMap, sync::Arc};

use algebra::{
    Algebra, MilnorAlgebra,
    milnor_algebra::{MilnorBasisElement, PPart},
    module::FDModule,
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
pub(crate) type XiPoly = FxHashSet<XiMonomial>;

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
pub(crate) fn toggle<T: Eq + std::hash::Hash>(set: &mut FxHashSet<T>, x: T) {
    if !set.remove(&x) {
        set.insert(x);
    }
}

/// The product of two polynomials.
pub(crate) fn poly_mul(a: &XiPoly, b: &XiPoly) -> XiPoly {
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
pub(crate) fn frobenius(f: &XiPoly, i: u32) -> XiPoly {
    f.iter()
        .map(|m| m.iter().map(|&e| e << i).collect())
        .collect()
}

/// $\chi \xi_j$ for `j` in `0..=max`, from $\sum_{i=0}^{j} \xi_{j-i}^{2^i} \chi\xi_i = 0$.
pub(crate) fn conjugates(max: usize) -> Vec<XiPoly> {
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

/// `chi[S]` lists the $R$ of degree at most `max_degree` with $\xi^S$ in $\chi(\xi^R)$, so that
/// $\chi(P(S)) = \sum_{R \in \mathtt{chi}[S]} P(R)$.
pub(crate) fn chi_table(max_degree: i32) -> FxHashMap<PackedXi, Vec<PackedXi>> {
    let top = (0..)
        .take_while(|&i| v_degree(i) <= max_degree)
        .last()
        .unwrap_or(0);
    let chi_xi = conjugates(top);
    let algebra = MilnorAlgebra::new(fp::prime::TWO, false);
    algebra.compute_basis(max_degree);
    let mut chi: FxHashMap<PackedXi, FxHashSet<PackedXi>> = FxHashMap::default();
    for d in 0..=max_degree {
        for idx in 0..algebra.dimension(d) {
            let r: Vec<u32> = algebra
                .basis_element_from_index(d, idx)
                .p_part
                .iter()
                .collect();
            let mut poly: XiPoly = std::iter::once(Vec::new()).collect();
            for (j, &e) in r.iter().enumerate() {
                for bit in 0..32 {
                    if e >> bit & 1 == 1 {
                        poly = poly_mul(&poly, &frobenius(&chi_xi[j + 1], bit));
                    }
                }
            }
            for m in poly {
                toggle(chi.entry(pack(&m)).or_default(), pack(&r));
            }
        }
    }
    chi.into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect()
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

/// The monomials free of $v_0$ with length at most `max_len` whose degree is at most
/// `cap(length)`, bucketed by degree and sorted by length within each bucket.
///
/// $m \mapsto v_0^{k - |m|} m$ identifies those of length at most `k` with a basis of $Q_k$, so
/// each $Q_k$ is a prefix of every bucket.
///
/// `cap` must be non-increasing, so `cap(0)` bounds every degree.
pub fn monomials(max_len: u32, cap: impl Fn(u32) -> i32) -> Vec<Vec<VMonomial>> {
    let max_degree = cap(0);
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
        cap: &dyn Fn(u32) -> i32,
        current: &mut Vec<u32>,
        out: &mut Vec<Vec<VMonomial>>,
    ) {
        if i == 0 {
            if degree <= cap(current.iter().sum()) {
                out[degree as usize].push(trim(current.clone()));
            }
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
                cap,
                current,
                out,
            );
            e += 1;
        }
        current[i] = 0;
    }
    rec(top, max_len, 0, max_degree, &cap, &mut current, &mut out);
    for bucket in &mut out {
        bucket.sort_by_key(|m| (m.iter().sum::<u32>(), m.clone()));
    }
    out
}

/// A monomial in the $\xi_j$, packed as [`XI_BITS`]-bit exponents of $\xi_1, \xi_2, \dots$ so that
/// multiplying monomials is adding.
pub(crate) type PackedXi = u128;

/// The bits per exponent in a [`PackedXi`]; [`Coaction::new`] checks the degree keeps every
/// exponent and every $\xi_j$ in range.
pub(crate) const XI_BITS: u32 = 12;

/// Packs an exponent vector.
pub(crate) fn pack(r: &[u32]) -> PackedXi {
    r.iter()
        .enumerate()
        .fold(0, |acc, (j, &e)| acc | (e as u128) << (j as u32 * XI_BITS))
}

/// Unpacks into a trimmed exponent vector.
pub(crate) fn unpack(mut r: PackedXi) -> Vec<u32> {
    let mut v = Vec::new();
    while r != 0 {
        v.push((r & ((1 << XI_BITS) - 1)) as u32);
        r >>= XI_BITS;
    }
    v
}

/// The coaction $\psi(v_n) = \sum_i c_{n,i} \otimes v_i$ on the generators of $Q$.
pub struct Coaction {
    /// `factors[n]` lists the monomials of each $c_{n,i}$, as $(\xi^R, i)$.
    factors: Vec<Vec<(PackedXi, usize)>>,
}

impl Coaction {
    /// The coaction on monomials of degree at most `max_degree`.
    pub fn new(max_degree: i32, convention: Convention) -> Self {
        assert!(
            max_degree < 1 << (128 / XI_BITS),
            "degree {max_degree} overflows PackedXi"
        );
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
                    .flat_map(|i| {
                        frobenius(&gens[n - i], i as u32)
                            .into_iter()
                            .map(move |m| (pack(&m), i))
                    })
                    .collect()
            })
            .collect();
        Self { factors }
    }
}

/// The action of $A$ on the bases of $Q_k^\vee$ for every `k <= max_len`, truncated.
///
/// The basis in each degree is dual to [`monomials`], whose length ordering makes the basis of
/// each $Q_k^\vee$ a prefix. $P(R)$ sends $m'^*$ to the sum of the $m^*$ for which
/// $\xi^R \otimes m'$ appears in $\psi(m)$. Since $v_0$ is primitive,
/// $\psi(v_0^e m) = v_0^e \psi(m)$, so this does not depend on $k$ and restricting to a prefix
/// gives $Q_k^\vee$.
pub struct QDual {
    /// `prefix[d][j]` is the number of basis elements of degree `d` of length at most `j`.
    prefix: Vec<Vec<usize>>,
    /// `offsets[d]` is the id of the first basis element of degree `d`.
    offsets: Vec<usize>,
    /// `actions[id]` lists `(op_deg, op_idx, out)` for each output `out` hit, with coefficient
    /// 1, by an operation from basis element `id`, sorted.
    actions: Vec<Vec<(i32, u32, u32)>>,
}

impl QDual {
    /// The basis action table under `coaction`, keeping a monomial of length `j` only in degrees
    /// up to `cap(j)`.
    ///
    /// `cap` must be non-increasing. The action raises both degree and length, so what is dropped
    /// is a submodule, and $\mathrm{Ext}^{a, u}(Q_k^\vee)$ is unchanged for $u \leq$ `cap(k)`.
    pub fn new(
        algebra: &MilnorAlgebra,
        max_len: u32,
        cap: impl Fn(u32) -> i32,
        coaction: &Coaction,
    ) -> Self {
        use maybe_rayon::prelude::*;

        let max_degree = cap(0);
        let basis = monomials(max_len, cap);
        algebra.compute_basis(max_degree);
        let prefix = basis
            .iter()
            .map(|b| {
                (0..=max_len)
                    .map(|j| b.partition_point(|m| m.iter().sum::<u32>() <= j))
                    .collect()
            })
            .collect::<Vec<Vec<usize>>>();
        let mut offsets = vec![0];
        for b in &basis {
            offsets.push(offsets.last().unwrap() + b.len());
        }
        let all: Vec<&VMonomial> = basis.iter().flatten().collect();
        let id_of: FxHashMap<&VMonomial, u32> = all
            .iter()
            .enumerate()
            .map(|(i, &m)| (m, i as u32))
            .collect();
        let degree_of: Vec<i32> = basis
            .iter()
            .enumerate()
            .flat_map(|(d, b)| std::iter::repeat_n(d as i32, b.len()))
            .collect();
        let max_n = coaction.factors.len() - 1;
        // times[id][i] is the id of the monomial times v_i, if that is in the basis.
        let times: Vec<Vec<Option<u32>>> = all
            .iter()
            .map(|m| {
                (0..=max_n)
                    .map(|i| {
                        let mut m = m.to_vec();
                        if m.len() <= i {
                            m.resize(i + 1, 0);
                        }
                        m[i] += 1;
                        id_of.get(&m).copied()
                    })
                    .collect()
            })
            .collect();

        // psi[id] is ψ of that monomial as sorted (ξ^R, v_0-free part of m') with coefficient 1,
        // built as ψ(m / v_n) ψ(v_n) for its top v_n. Within a degree these are independent.
        let mut psi: Vec<Vec<(PackedXi, u32)>> = vec![Vec::new(); all.len()];
        for (d, bucket) in basis.iter().enumerate() {
            let computed: Vec<Vec<(PackedXi, u32)>> = (0..bucket.len())
                .into_maybe_par_iter()
                .map(|idx| {
                    let m = &bucket[idx];
                    let Some(n) = m.iter().rposition(|&e| e > 0) else {
                        return vec![(0, id_of[m])];
                    };
                    let mut rest = m.clone();
                    rest[n] -= 1;
                    let rest = &psi[id_of[&trim(rest)] as usize];
                    let mut terms = Vec::with_capacity(rest.len() * coaction.factors[n].len());
                    for &(r, in_id) in rest {
                        for &(s, i) in &coaction.factors[n] {
                            let in_id = if i == 0 {
                                in_id
                            } else {
                                times[in_id as usize][i].expect("a term of ψ is in the basis")
                            };
                            terms.push((r + s, in_id));
                        }
                    }
                    terms.sort_unstable();
                    let mut kept = Vec::with_capacity(terms.len());
                    for run in terms.chunk_by(|a, b| a == b) {
                        if run.len() % 2 == 1 {
                            kept.push(run[0]);
                        }
                    }
                    kept
                })
                .collect();
            for (idx, terms) in computed.into_iter().enumerate() {
                psi[offsets[d] + idx] = terms;
            }
        }

        let mut op_index: FxHashMap<PackedXi, u32> = FxHashMap::default();
        let mut actions: Vec<Vec<(i32, u32, u32)>> = vec![Vec::new(); all.len()];
        for (out_id, terms) in psi.iter().enumerate() {
            let out_deg = degree_of[out_id];
            let out_idx = (out_id - offsets[out_deg as usize]) as u32;
            for &(r, in_id) in terms {
                let op_deg = out_deg - degree_of[in_id as usize];
                if op_deg == 0 {
                    continue;
                }
                let op_idx = *op_index.entry(r).or_insert_with(|| {
                    algebra.basis_element_to_index(&MilnorBasisElement {
                        q_part: 0,
                        p_part: PPart::from_slice(&unpack(r)),
                        degree: op_deg,
                    }) as u32
                });
                actions[in_id as usize].push((op_deg, op_idx, out_idx));
            }
        }
        drop(psi);
        actions.maybe_par_iter_mut().for_each(|a| a.sort_unstable());
        Self {
            prefix,
            offsets,
            actions,
        }
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
    pub fn act(
        &self,
        op_deg: i32,
        op_idx: usize,
        d: i32,
        idx: usize,
        k: u32,
    ) -> impl Iterator<Item = usize> + '_ {
        let list = &self.actions[self.offsets[d as usize] + idx];
        let op = (op_deg, op_idx as u32);
        let lo = list.partition_point(|&(a, b, _)| (a, b) < op);
        let hi = list.partition_point(|&(a, b, _)| (a, b) <= op);
        let dim = self.dimension(d + op_deg, k);
        list[lo..hi]
            .iter()
            .map(|&(_, _, out)| out as usize)
            .take_while(move |&out| out < dim)
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
                let list = &self.actions[self.offsets[in_deg as usize] + in_idx];
                for &(op_deg, op_idx, out) in list {
                    if in_deg + op_deg > top || out as usize >= self.dimension(in_deg + op_deg, k) {
                        continue;
                    }
                    module
                        .action_mut(op_deg, op_idx as usize, in_deg, in_idx)
                        .add_basis_element(out as usize, 1);
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
    QDual::new(&algebra, k, |_| max_degree, coaction).module(algebra, k)
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
        |_| max_u,
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
/// $\delta^a$ at weight $u$ reads generators of $P_a$ and $P_{a+1}$ in degrees up to $u$, and is
/// only needed for $u \leq$ `(max_n + a + 1) / 2`, so they lie in stems up to `(max_n + 1) / 2`.
pub fn untwisted_backbone_range(max_n: i32, max_s: i32) -> Bidegree {
    Bidegree::n_s((max_n + 2) / 2, max_s + 1)
}

/// `prefix[d][j]`: the number of $v_0$-free monomials of degree `d` and length at most `j`,
/// keeping a monomial of length `j` only in degrees up to `cap(j)`.
///
/// These are the dimensions of the $Q_k^\vee$ that [`QDual`] builds, counted without listing
/// the monomials.
pub fn q_prefix(max_len: u32, cap: impl Fn(u32) -> i32) -> Vec<Vec<usize>> {
    let max_degree = cap(0).max(0) as usize;
    let len = max_len as usize;
    // exact[d][j]: monomials in v_1, v_2, ... of degree d and length exactly j.
    let mut exact = vec![vec![0usize; len + 1]; max_degree + 1];
    exact[0][0] = 1;
    for i in (1..).take_while(|&i| v_degree(i) as usize <= max_degree) {
        let part = v_degree(i) as usize;
        for d in part..=max_degree {
            for j in 1..=len {
                exact[d][j] += exact[d - part][j - 1];
            }
        }
    }
    exact
        .iter()
        .enumerate()
        .map(|(d, row)| {
            let mut total = 0;
            (0..=len)
                .map(|j| {
                    if d as i32 <= cap(j as u32) {
                        total += row[j];
                    }
                    total
                })
                .collect()
        })
        .collect()
}

/// One coboundary $\delta^a$ of the untwisted complex at weight $u$, for $Q_k$ with the largest
/// $k$ it is read for.
///
/// $\psi$ never lengthens a monomial, so the coboundary for a smaller $k$ is this one restricted
/// to the columns of length at most $k$, and one reduction serves every $k$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UntwistedMatrix {
    /// The largest length of $Q_k$ it is read for.
    pub k: i32,
    /// The weight.
    pub u: i32,
    /// The source filtration; the target is `a + 1`.
    pub a: i32,
    /// The dimension of the target cochains.
    pub rows: usize,
    /// The dimension of the source cochains.
    pub cols: usize,
}

/// The cochains of filtration `a` at weight `u`: the offset and generator count of each block
/// $x_i \otimes Q_k^\vee$, by generator degree, and the total dimension.
pub(crate) fn cochain_blocks<CC: FreeChainComplex>(
    res: &CC,
    q_dim: impl Fn(i32) -> usize,
    a: i32,
    u: i32,
) -> (Vec<(usize, usize)>, usize) {
    let mut blocks = Vec::with_capacity(u as usize + 1);
    let mut total = 0;
    for t in 0..=u {
        let gens = res.number_of_gens_in_bidegree(Bidegree::s_t(a, t));
        blocks.push((total, gens));
        total += gens * q_dim(u - t);
    }
    (blocks, total)
}

/// The largest weight read for $Q_k$: $\mathrm{Ext}^{a, u}(Q_k)$ is wanted for
/// $a \leq$ `max_s - k` and stem $2u - a \leq$ `max_n`.
pub(crate) fn untwisted_max_weight(max_n: i32, max_s: i32, k: i32) -> i32 {
    (max_n + max_s - k) / 2
}

/// The filtrations whose coboundary is read at `(k, u)`: from one below the lowest wanted
/// filtration, for the coboundary into it, to the highest.
pub(crate) fn untwisted_filtrations(
    max_n: i32,
    max_s: i32,
    k: i32,
    u: i32,
) -> std::ops::RangeInclusive<i32> {
    (2 * u - max_n - 1).max(0)..=max_s - k
}

/// Every coboundary [`ctau_e2_untwisted_on`] reduces, from the generator counts of `res` alone.
///
/// Builds nothing, so it sizes a run, against the backbone it will use, before starting it.
pub fn untwisted_plan<CC: FreeChainComplex>(
    res: &CC,
    max_n: i32,
    max_s: i32,
) -> Vec<UntwistedMatrix> {
    let prefix = q_prefix(max_s as u32, |j| {
        untwisted_max_weight(max_n, max_s, j as i32)
    });
    let mut plan = Vec::new();
    for a in 0..=max_s {
        for u in 0..=(max_n + max_s) / 2 {
            let Some(k) = untwisted_max_length(max_n, max_s, a, u) else {
                continue;
            };
            let q_dim = |d: i32| {
                usize::try_from(d)
                    .ok()
                    .and_then(|d| prefix.get(d))
                    .map_or(0, |p| p[k as usize])
            };
            let cols = cochain_blocks(res, q_dim, a, u).1;
            let rows = cochain_blocks(res, q_dim, a + 1, u).1;
            if rows > 0 && cols > 0 {
                plan.push(UntwistedMatrix {
                    k,
                    u,
                    a,
                    rows,
                    cols,
                });
            }
        }
    }
    plan
}

/// The largest $k$ for which $\delta^a$ at weight $u$ is read, if any: the inverse of
/// [`untwisted_max_weight`] and [`untwisted_filtrations`].
pub(crate) fn untwisted_max_length(max_n: i32, max_s: i32, a: i32, u: i32) -> Option<i32> {
    let k = (max_s - a).min(max_n + max_s - 2 * u);
    (k >= 0 && a >= 2 * u - max_n - 1).then_some(k)
}

/// A term $P(R) x_i$ of some $d z_l$, packed.
#[derive(Clone, Copy)]
pub(crate) struct DifferentialTerm {
    /// The index of $P(R)$ in its degree.
    op_idx: u32,
    /// The index of $x_i$ in its degree.
    gen_idx: u32,
    /// The degree of $P(R)$.
    op_deg: u16,
    /// The degree of $x_i$.
    gen_deg: u16,
}

/// The terms of $d z_l$ for every generator $z_l$ of $P_{a+1}$ in degrees up to `max_t`, indexed
/// by degree then generator.
pub(crate) fn differential_slice<CC: FreeChainComplex>(
    res: &CC,
    a: i32,
    max_t: i32,
) -> Vec<Vec<Vec<DifferentialTerm>>> {
    let d = res.differential(a + 1);
    let source = res.module(a);
    (0..=max_t)
        .map(|t| {
            let gens = res.number_of_gens_in_bidegree(Bidegree::s_t(a + 1, t));
            (0..gens)
                .map(|l| {
                    d.output(t, l)
                        .iter_nonzero()
                        .map(|(idx, _)| {
                            let og = source.index_to_op_gen(t, idx);
                            DifferentialTerm {
                                op_idx: og.operation_index as u32,
                                gen_idx: og.generator_index as u32,
                                op_deg: og.operation_degree as u16,
                                gen_deg: og.generator_degree as u16,
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// The highest $v_i$ a [`QDown`] monomial can hold; its degree bounds the weights it supports.
pub(crate) const QDOWN_VARS: usize = 11;

/// A monomial in $v_0, \dots, v_{10}$, as exponents.
pub(crate) type QMono = [u16; QDOWN_VARS];

/// The bases of $Q_k^\vee$ for every `k <= max_len`, with the action of $A$ computed on demand.
///
/// [`QDual`] stores every action, which outgrows memory at large weights. This stores only
/// the bases, ordered as in [`monomials`], and computes which $m_\gamma$ the term
/// $\xi^R \otimes m_\gamma$ of $\psi(m)$ reaches when asked for a given $R$ and $m$.
pub struct QDown {
    /// The monomials of each degree, sorted by length and then exponents.
    by_degree: Vec<Vec<QMono>>,
    /// `prefix[d][j]` is the number of monomials of degree `d` of length at most `j`.
    prefix: Vec<Vec<usize>>,
    /// `factors[n]` lists the terms $(\xi^S, i)$ of $\psi(v_n) = \sum \xi^S \otimes v_i$.
    factors: Vec<Vec<(PackedXi, usize)>>,
    /// `reach[n][k]` is the largest exponent of $\xi_{k+1}$ in a term of $\psi(v_n)$.
    reach: Vec<Vec<u32>>,
    /// The index of each monomial of each degree, as [`Self::index`] finds it by search.
    positions: Vec<FxHashMap<QMono, u32>>,
}

/// The length of a monomial.
pub(crate) fn qmono_len(m: &QMono) -> u32 {
    m.iter().map(|&e| e as u32).sum()
}

/// The degree of $\xi^R$, halved: $\sum_k r_k (2^k - 1)$.
pub(crate) fn xi_degree(r: PackedXi) -> i32 {
    let mask = (1u128 << XI_BITS) - 1;
    (0..128 / XI_BITS)
        .map(|k| ((r >> (k * XI_BITS)) & mask) as i32 * ((1 << (k + 1)) - 1))
        .sum()
}

/// The exponentwise maximum of `a` and `b`.
pub(crate) fn packed_max(a: PackedXi, b: PackedXi) -> PackedXi {
    let mask = (1u128 << XI_BITS) - 1;
    (0..128 / XI_BITS).fold(0, |out, k| {
        let shift = k * XI_BITS;
        out | (((a >> shift) & mask).max((b >> shift) & mask) << shift)
    })
}

/// Whether every exponent of `s` is at most that of `r`.
pub(crate) fn packed_fits(s: PackedXi, r: PackedXi) -> bool {
    let mask = (1u128 << XI_BITS) - 1;
    (0..128 / XI_BITS).all(|j| (s >> (j * XI_BITS)) & mask <= (r >> (j * XI_BITS)) & mask)
}

impl QDown {
    /// The bases under the coaction of `convention`, keeping a monomial of length `j` only in
    /// degrees up to `cap(j)`, which must be non-increasing.
    pub fn new(max_len: u32, cap: impl Fn(u32) -> i32, convention: Convention) -> Self {
        let max_degree = cap(0);
        assert!(
            max_degree < (1 << (QDOWN_VARS - 1)),
            "degree {max_degree} needs more than v_{}",
            QDOWN_VARS - 1
        );
        let mut by_degree: Vec<Vec<QMono>> = monomials(max_len, cap)
            .into_iter()
            .map(|bucket| {
                bucket
                    .into_iter()
                    .map(|m| {
                        let mut q = [0u16; QDOWN_VARS];
                        for (i, &e) in m.iter().enumerate() {
                            q[i] = e as u16;
                        }
                        q
                    })
                    .collect()
            })
            .collect();
        for bucket in &mut by_degree {
            bucket.sort_by_key(|m| (qmono_len(m), *m));
        }
        let prefix = by_degree
            .iter()
            .map(|b| {
                (0..=max_len)
                    .map(|j| b.partition_point(|m| qmono_len(m) <= j))
                    .collect()
            })
            .collect();
        let factors = Coaction::new(max_degree, convention).factors;
        let reach = factors
            .iter()
            .map(|terms| {
                let mut most = vec![0u32; (128 / XI_BITS) as usize];
                for &(s, _) in terms {
                    for (k, e) in unpack(s).into_iter().enumerate() {
                        most[k] = most[k].max(e);
                    }
                }
                most
            })
            .collect();
        let mut out = Self {
            by_degree,
            prefix,
            factors,
            reach,
            positions: Vec::new(),
        };
        out.index_positions();
        out
    }

    /// Keeps only the monomials `keep` accepts, which must span a subcomodule or the basis of a
    /// quotient comodule.
    pub(crate) fn restrict(mut self, keep: impl Fn(&QMono) -> bool) -> Self {
        let max_len = self.prefix.first().map_or(0, |p| p.len() - 1) as u32;
        for bucket in &mut self.by_degree {
            bucket.retain(&keep);
        }
        self.index_positions();
        self.prefix = self
            .by_degree
            .iter()
            .map(|b| {
                (0..=max_len)
                    .map(|j| b.partition_point(|m| qmono_len(m) <= j))
                    .collect()
            })
            .collect();
        self
    }

    /// The largest degree of a basis monomial.
    pub(crate) fn max_degree(&self) -> i32 {
        self.by_degree.len() as i32 - 1
    }

    /// Fills [`Self::positions`] from the bases.
    fn index_positions(&mut self) {
        self.positions = self
            .by_degree
            .iter()
            .map(|b| b.iter().enumerate().map(|(i, m)| (*m, i as u32)).collect())
            .collect();
    }

    /// The dimension of $Q_k^\vee$ in degree `d`, zero outside the truncation.
    pub fn dimension(&self, d: i32, k: u32) -> usize {
        usize::try_from(d)
            .ok()
            .and_then(|d| self.prefix.get(d))
            .map_or(0, |p| p[k as usize])
    }

    /// The `idx`th basis monomial of degree `d`.
    pub(crate) fn monomial(&self, d: i32, idx: usize) -> &QMono {
        &self.by_degree[d as usize][idx]
    }

    /// The index of `m` among the monomials of degree `d`.
    pub(crate) fn index(&self, d: i32, m: &QMono) -> usize {
        self.by_degree[d as usize]
            .binary_search_by_key(&(qmono_len(m), *m), |x| (qmono_len(x), *x))
            .expect("a term of ψ is in the basis")
    }

    /// Whether $\xi^R$ can occur in $\psi(m)$ at all: each exponent of $R$ is at most what the
    /// factors $\psi(v_n)^{e_n}$ can supply together. Most operations fail this, and it costs
    /// no memo traffic.
    pub(crate) fn can_reach(&self, r: PackedXi, m: &QMono) -> bool {
        let mask = (1u128 << XI_BITS) - 1;
        let mut r = r;
        let mut k = 0;
        while r != 0 {
            let want = (r & mask) as u32;
            if want > 0 {
                let have: u32 = m
                    .iter()
                    .enumerate()
                    .skip(1)
                    .filter(|(_, e)| **e > 0)
                    .map(|(n, &e)| e as u32 * self.reach[n][k])
                    .sum();
                if have < want {
                    return false;
                }
            }
            r >>= XI_BITS;
            k += 1;
        }
        true
    }

    /// $\psi(m \cdot v_n^{2^j})$ from $\psi(m)$, as the basis $m_\gamma$ paired with each $\xi^R$,
    /// keeping only $R$ with every exponent at most that of `cap`. `d` is the degree of $m$.
    ///
    /// A term outside the truncated basis is dropped as soon as it appears: the factors still to
    /// come only lengthen it or leave it as it is, so it never comes back.
    fn psi_step(&self, below: &Psi, d: i32, n: usize, j: u32, cap: PackedXi) -> Psi {
        let d_m = d + (1 << j) * ((1 << n) - 1);
        // Every (R, γ) produced; a pair produced twice cancels.
        let mut pairs: Vec<(PackedXi, u32)> = Vec::new();
        for &(s, i) in &self.factors[n] {
            let s = s << j;
            if !packed_fits(s, cap) {
                continue;
            }
            for (r, gammas) in below.iter() {
                let r_new = r + s;
                if !packed_fits(r_new, cap) {
                    continue;
                }
                let (d_old, d_new) = (d - xi_degree(r), d_m - xi_degree(r_new));
                for &g in gammas {
                    let mut gamma = *self.monomial(d_old, g as usize);
                    if i > 0 {
                        gamma[i] += 1 << j;
                    }
                    if let Some(idx) = self.try_index(d_new, &gamma) {
                        pairs.push((r_new, idx as u32));
                    }
                }
            }
        }
        Psi::from_pairs(pairs)
    }

    /// $\psi(m)$ for every basis monomial $m$ of degree $d$ and length at most `wanted[d]`, by
    /// degree and index, keeping only the operations `ops` numbers that `allowed[d]` marks, keyed
    /// by their numbers; the other entries are empty.
    ///
    /// Peeling the largest binary chunk $v_n^{2^j}$ off each monomial makes the monomials a
    /// tree, and $\psi$ of a child is [`Self::psi_step`] of $\psi$ of its parent. A depth-first
    /// walk computes each once, holds the whole coaction, below the exponents of `ops`, only
    /// along the current path, and enters only subtrees holding a wanted monomial.
    pub(crate) fn psi_table(
        &self,
        ops: &FxHashMap<PackedXi, u32>,
        allowed: &[Vec<bool>],
        wanted: &[i32],
    ) -> Vec<Vec<Psi<u32>>> {
        let cap = ops.keys().copied().fold(0, packed_max);
        // reach[d]: the longest wanted monomial of degree d or more, which is as far as the
        // monomials of degree d can still lead.
        let mut reach = vec![-1; self.by_degree.len() + 1];
        for d in (0..self.by_degree.len()).rev() {
            reach[d] = reach[d + 1].max(wanted.get(d).copied().unwrap_or(-1));
        }
        let table: Vec<Vec<std::sync::OnceLock<Psi<u32>>>> = self
            .by_degree
            .iter()
            .map(|b| (0..b.len()).map(|_| std::sync::OnceLock::new()).collect())
            .collect();
        let walk = Walk {
            q: self,
            cap,
            ops,
            allowed,
            wanted,
            reach: &reach,
            table: &table,
        };
        let root = [0u16; QDOWN_VARS];
        walk.visit(
            root,
            0,
            Psi::from_pairs(vec![(0, self.index(0, &root) as u32)]),
        );
        table
            .into_iter()
            .map(|b| {
                b.into_iter()
                    .map(|c| c.into_inner().unwrap_or_default())
                    .collect()
            })
            .collect()
    }

    /// The basis monomials whose largest binary chunk, peeled off, leaves `m` of degree `d`:
    /// each with its degree and the chunk $v_n^{2^j}$.
    fn children(&self, m: &QMono, d: i32) -> Vec<(QMono, i32, usize, u32)> {
        let top = m.iter().rposition(|&e| e > 0).filter(|&n| n > 0);
        let mut out = Vec::new();
        for n in top.unwrap_or(1)..QDOWN_VARS {
            // For the top variable the chunk must exceed every power already there.
            let first = match top {
                Some(t) if t == n => 16 - m[n].leading_zeros(),
                _ => 0,
            };
            for j in first..16 {
                let d_child = d + (1 << j) * ((1 << n) - 1);
                if d_child as usize >= self.by_degree.len() {
                    break;
                }
                let mut child = *m;
                child[n] += 1 << j;
                // Longer and higher still is outside the truncation too.
                if self.try_index(d_child, &child).is_none() {
                    break;
                }
                out.push((child, d_child, n, j));
            }
        }
        out
    }

    /// The index of `m` among the monomials of degree `d`, if it is one.
    fn try_index(&self, d: i32, m: &QMono) -> Option<usize> {
        self.positions
            .get(usize::try_from(d).ok()?)?
            .get(m)
            .map(|&i| i as usize)
    }

    /// The $m_\gamma$ for which $\xi^R \otimes m_\gamma$ has coefficient 1 in $\psi(m)$, memoised in
    /// `memo` on every intermediate $(R', m')$.
    ///
    /// Peels the largest binary chunk $v_n^{2^j}$ off $m$:
    /// $\psi(m) = \psi(m / v_n^{2^j})\, \psi(v_n)^{2^j}$, so the products shared by different
    /// monomials and operations are computed once. `memo` is cleared when it passes
    /// `memo_limit` entries, trading recomputation for bounded memory.
    pub(crate) fn act_memo(
        &self,
        r: PackedXi,
        m: &QMono,
        memo: &ActMemo,
        memo_limit: usize,
    ) -> std::sync::Arc<Vec<QMono>> {
        use std::sync::Arc;
        let Some(n) = m.iter().rposition(|&e| e > 0).filter(|&n| n > 0) else {
            return Arc::new(if r == 0 { vec![*m] } else { Vec::new() });
        };
        if let Some(hit) = memo.get(&(r, *m)) {
            return Arc::clone(&hit);
        }
        let j = 15 - m[n].leading_zeros();
        let mut rest = *m;
        rest[n] -= 1 << j;
        let mut out: FxHashSet<QMono> = FxHashSet::default();
        for &(s, i) in &self.factors[n] {
            let s = s << j;
            if !packed_fits(s, r) {
                continue;
            }
            for gamma in self.act_memo(r - s, &rest, memo, memo_limit).iter() {
                let mut gamma = *gamma;
                if i > 0 {
                    gamma[i] += 1 << j;
                }
                toggle(&mut out, gamma);
            }
        }
        let out = Arc::new(out.into_iter().collect::<Vec<_>>());
        if memo.len() > memo_limit {
            memo.clear();
        }
        memo.insert((r, *m), Arc::clone(&out));
        out
    }
}

/// The state of [`QDown::psi_table`]'s walk down the tree of monomials.
struct Walk<'a> {
    /// The bases and coaction.
    q: &'a QDown,
    /// Every exponent of every operation in `ops`.
    cap: PackedXi,
    /// The operations kept, numbered.
    ops: &'a FxHashMap<PackedXi, u32>,
    /// For each degree, which numbered operations act on its monomials.
    allowed: &'a [Vec<bool>],
    /// The longest wanted monomial in each degree, or -1.
    wanted: &'a [i32],
    /// The longest wanted monomial in each degree or above.
    reach: &'a [i32],
    /// The coaction on each wanted monomial, by degree and index.
    table: &'a [Vec<std::sync::OnceLock<Psi<u32>>>],
}

impl Walk<'_> {
    /// Records `psi`, the coaction on `m` of degree `d`, if `m` is wanted, then visits the
    /// children of `m` that lead to wanted monomials, in parallel.
    fn visit(&self, m: QMono, d: i32, psi: Psi) {
        use maybe_rayon::prelude::*;
        let children: Vec<_> = self
            .q
            .children(&m, d)
            .into_iter()
            .filter(|(c, d_c, _, _)| qmono_len(c) as i32 <= self.reach[*d_c as usize])
            .collect();
        children
            .into_maybe_par_iter()
            .for_each(|(child, d_child, n, j)| {
                let child_psi = self.q.psi_step(&psi, d, n, j, self.cap);
                self.visit(child, d_child, child_psi);
            });
        if qmono_len(&m) as i32 <= self.wanted.get(d as usize).copied().unwrap_or(-1) {
            let kept = psi.restricted(self.ops, &self.allowed[d as usize]);
            let _ = self.table[d as usize][self.q.index(d, &m)].set(kept);
        }
    }
}

/// $\psi(m)$ for one monomial $m$: for each $\xi^R$, by increasing $R$, the indices of the
/// $m_\gamma$ paired with it among the monomials of their degree.
#[derive(Default)]
pub(crate) struct Psi<K = PackedXi> {
    /// The $R$ that occur, or their indices among the operations kept.
    keys: Vec<K>,
    /// Where the indices for each $R$ start in `gammas`, and their end.
    starts: Vec<u32>,
    /// The indices, for one $R$ after another.
    gammas: Vec<u32>,
}

impl<K: Ord + Copy> Psi<K> {
    /// The coaction from its (R, index) pairs, a pair occurring twice cancelling.
    fn from_pairs(mut pairs: Vec<(K, u32)>) -> Self {
        pairs.sort_unstable();
        let mut psi = Self {
            keys: Vec::new(),
            starts: vec![0],
            gammas: Vec::with_capacity(pairs.len()),
        };
        let mut k = 0;
        while k < pairs.len() {
            let r = pairs[k].0;
            let start = psi.gammas.len();
            while k < pairs.len() && pairs[k].0 == r {
                let g = pairs[k].1;
                let mut count = 0;
                while k < pairs.len() && pairs[k] == (r, g) {
                    count += 1;
                    k += 1;
                }
                if count % 2 == 1 {
                    psi.gammas.push(g);
                }
            }
            if psi.gammas.len() > start {
                psi.keys.push(r);
                psi.starts.push(psi.gammas.len() as u32);
            }
        }
        psi
    }

    /// The indices of the $m_\gamma$ paired with $\xi^R$.
    pub(crate) fn get(&self, r: K) -> &[u32] {
        match self.keys.binary_search(&r) {
            Ok(i) => &self.gammas[self.starts[i] as usize..self.starts[i + 1] as usize],
            Err(_) => &[],
        }
    }

    /// The number of $R$ that occur.
    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    /// Each $R$ with its indices.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (K, &[u32])> {
        self.keys.iter().enumerate().map(|(i, &r)| {
            (
                r,
                &self.gammas[self.starts[i] as usize..self.starts[i + 1] as usize],
            )
        })
    }
}

impl Psi {
    /// The part on the operations `ops` numbers and `allowed` marks, keyed by their numbers.
    fn restricted(&self, ops: &FxHashMap<PackedXi, u32>, allowed: &[bool]) -> Psi<u32> {
        Psi::from_pairs(
            self.iter()
                .filter_map(|(r, g)| {
                    let &id = ops.get(&r)?;
                    allowed[id as usize].then_some((id, g))
                })
                .flat_map(|(id, g)| g.iter().map(move |&g| (id, g)))
                .collect(),
        )
    }
}

/// The memo of [`QDown::act_memo`].
pub(crate) type ActMemo =
    dashmap::DashMap<(PackedXi, QMono), std::sync::Arc<Vec<QMono>>, rustc_hash::FxBuildHasher>;

/// The terms of each backbone differential landing on a generator $x_i$ of $P_a$, grouped by
/// operation: `(t_i, i)` maps to `(R, [(t_l, l)])` for each $z_l$ with $P(R) x_i$ in $d z_l$.
pub(crate) type InvertedSlice = FxHashMap<(i32, u32), Vec<(PackedXi, Vec<(i32, u32)>)>>;

/// Inverts a [`differential_slice`] of filtration `a`.
pub(crate) fn invert_slice(
    algebra: &MilnorAlgebra,
    slice: &[Vec<Vec<DifferentialTerm>>],
) -> InvertedSlice {
    let mut ops: FxHashMap<(u16, u32), PackedXi> = FxHashMap::default();
    let mut grouped: FxHashMap<(i32, u32), BTreeMap<PackedXi, Vec<(i32, u32)>>> =
        FxHashMap::default();
    for (t_l, gens) in slice.iter().enumerate() {
        for (l, dz) in gens.iter().enumerate() {
            for term in dz {
                let r = *ops.entry((term.op_deg, term.op_idx)).or_insert_with(|| {
                    let elt =
                        algebra.basis_element_from_index(term.op_deg as i32, term.op_idx as usize);
                    pack(&elt.p_part.iter().collect::<Vec<_>>())
                });
                grouped
                    .entry((term.gen_deg as i32, term.gen_idx))
                    .or_default()
                    .entry(r)
                    .or_default()
                    .push((t_l as i32, l as u32));
            }
        }
    }
    grouped
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().collect()))
        .collect()
}

/// Options for [`ctau_e2_untwisted_with`].
#[derive(Debug, Clone, Copy)]
pub struct UntwistedOptions {
    /// Coboundaries with more entries than this are built and reduced one at a time, so only one
    /// is ever held; smaller ones run in parallel. Raising it trades memory for parallelism.
    pub serial_above: usize,
    /// The most entries the action memo holds before it is cleared. Raising it trades memory
    /// for less recomputation.
    pub memo_limit: usize,
}

impl Default for UntwistedOptions {
    fn default() -> Self {
        Self {
            serial_above: 1 << 30,
            memo_limit: 1 << 24,
        }
    }
}

/// [`ctau_e2_untwisted`] on an existing resolution `res` of the classical $\mathbb{F}_2$, computed
/// through [`untwisted_backbone_range`].
pub fn ctau_e2_untwisted_on<CC>(res: &CC, max_n: i32, max_s: i32) -> TrigradedDims
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    ctau_e2_untwisted_with(res, max_n, max_s, UntwistedOptions::default())
}

/// [`ctau_e2_untwisted_on`] with `options`.
///
/// Filtration by filtration, it reads the differential out of $P_{a+1}$, reduces every
/// $\delta^a$ that needs it and keeps only the ranks, so neither the whole resolution nor more
/// than one large coboundary is held at once.
pub fn ctau_e2_untwisted_with<CC>(
    res: &CC,
    max_n: i32,
    max_s: i32,
    options: UntwistedOptions,
) -> TrigradedDims
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
        res.has_computed_bidegree(range),
        "the backbone must be resolved through {range}"
    );

    let twisted = QDown::new(
        max_s as u32,
        |j| untwisted_max_weight(max_n, max_s, j as i32),
        Convention::Conjugate,
    );
    let plan = untwisted_plan(res, max_n, max_s);

    // Column (x_i, m) of δ^a has a 1 in row (z_l, m_γ) when P(R) x_i is a term of d z_l and
    // ξ^R ⊗ m_γ is a term of the conjugate coaction on m.
    // The action depends on neither k nor u, so one memo serves every matrix of a pass.
    // The ranks of δ^a at weight u for Q_k, for every k up to `m.k`. Columns are ordered by
    // length, so the rank for k is the number of pivots among the columns of length at most k.
    let ranks_of = |m: &UntwistedMatrix, inverted: &InvertedSlice, memo: &ActMemo| -> Vec<usize> {
        let (u, k_u) = (m.u, m.k as u32);
        let q_dim = |d: i32| twisted.dimension(d, k_u);
        let (src, _) = cochain_blocks(res, q_dim, m.a, u);
        let (tgt, _) = cochain_blocks(res, q_dim, m.a + 1, u);

        // The position of each source cochain once sorted by length, and how many have each
        // length at most j.
        let mut lengths = Vec::with_capacity(m.cols);
        for (t_i, &(_, gens)) in src.iter().enumerate() {
            let d = u - t_i as i32;
            for _ in 0..gens {
                for idx in 0..q_dim(d) {
                    lengths.push(qmono_len(twisted.monomial(d, idx)));
                }
            }
        }
        let mut order: Vec<usize> = (0..lengths.len()).collect();
        order.sort_by_key(|&c| lengths[c]);
        let mut position = vec![0; order.len()];
        for (new, &old) in order.iter().enumerate() {
            position[old] = new;
        }
        let up_to: Vec<usize> = (0..=m.k as u32)
            .map(|j| lengths.iter().filter(|&&l| l <= j).count())
            .collect();

        let mut matrix = Matrix::new(p, m.rows, m.cols);
        for (t_i, &(src_off, gens)) in src.iter().enumerate() {
            let t_i = t_i as i32;
            let block = q_dim(u - t_i);
            if block == 0 {
                continue;
            }
            for i in 0..gens {
                let Some(terms) = inverted.get(&(t_i, i as u32)) else {
                    continue;
                };
                for idx in 0..block {
                    let col = position[src_off + i * block + idx];
                    let mono = twisted.monomial(u - t_i, idx);
                    for (r, targets) in terms {
                        if !twisted.can_reach(*r, mono) {
                            continue;
                        }
                        let gammas = twisted.act_memo(*r, mono, memo, options.memo_limit);
                        if gammas.is_empty() {
                            continue;
                        }
                        for &(t_l, l) in targets {
                            if t_l > u {
                                continue;
                            }
                            let (tgt_off, _) = tgt[t_l as usize];
                            let tgt_block = q_dim(u - t_l);
                            for gamma in gammas.iter() {
                                let g = twisted.index(u - t_l, gamma);
                                let row = tgt_off + l as usize * tgt_block + g;
                                matrix.row_mut(row).add_basis_element(col, 1);
                            }
                        }
                    }
                }
            }
        }
        matrix.row_reduce();
        let pivots = matrix.pivots();
        up_to
            .iter()
            .map(|&c| pivots[..c].iter().filter(|&&r| r >= 0).count())
            .collect()
    };

    let mut ranks: FxHashMap<(i32, i32, i32), usize> = FxHashMap::default();
    for a in 0..=max_s {
        let (large, small): (Vec<_>, Vec<_>) = plan
            .iter()
            .filter(|m| m.a == a)
            .partition(|m| m.rows * m.cols > options.serial_above);
        if large.is_empty() && small.is_empty() {
            continue;
        }
        // The largest weight δ^a is read at, from the inverse of `untwisted_filtrations`.
        let slice = invert_slice(
            &algebra,
            &differential_slice(res, a, max_u.min((max_n + a + 1) / 2)),
        );
        let memo = ActMemo::default();
        let small_ranks: Vec<_> = small
            .into_maybe_par_iter()
            .map(|m: &UntwistedMatrix| (m, ranks_of(m, &slice, &memo)))
            .collect();
        let large_ranks = large.into_iter().map(|m| (m, ranks_of(m, &slice, &memo)));
        for (m, by_k) in small_ranks.into_iter().chain(large_ranks) {
            for (k, r) in by_k.into_iter().enumerate() {
                ranks.insert((k as i32, m.u, m.a), r);
            }
        }
    }

    let mut dims = TrigradedDims::default();
    for k in 0..=max_s {
        let q_dim = |d: i32| twisted.dimension(d, k as u32);
        for u in 0..=untwisted_max_weight(max_n, max_s, k) {
            for a in untwisted_filtrations(max_n, max_s, k, u) {
                let n = 2 * u - a;
                if n > max_n || n < 0 {
                    continue;
                }
                let rank_of = |a| ranks.get(&(k, u, a)).copied().unwrap_or(0);
                let d = cochain_blocks(res, q_dim, a, u).1 - rank_of(a) - rank_of(a - 1);
                if d > 0 {
                    dims.0.insert((n, a + k, u), d);
                }
            }
        }
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
        let m = monomials(2, |_| 3);
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

    /// [`q_prefix`] counts the bases [`monomials`] lists.
    #[test]
    fn q_prefix_counts_monomials() {
        let cap = |j: u32| 40 - j as i32 / 2;
        let listed = monomials(12, cap);
        let counted = q_prefix(12, cap);
        for (d, bucket) in listed.iter().enumerate() {
            for j in 0..=12 {
                let n = bucket.iter().filter(|m| m.iter().sum::<u32>() <= j).count();
                assert_eq!(counted[d][j as usize], n, "degree {d}, length {j}");
            }
        }
    }

    /// Running every coboundary one at a time changes nothing.
    #[test]
    fn serial_matches_parallel() {
        let res = crate::nassau::Resolution::new(Arc::new(unit_module()));
        res.compute_through_stem(untwisted_backbone_range(30, 15));
        let serial = ctau_e2_untwisted_with(
            &res,
            30,
            15,
            UntwistedOptions {
                serial_above: 0,
                memo_limit: 16,
            },
        );
        assert_eq!(serial, ctau_e2_untwisted_on(&res, 30, 15));
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
