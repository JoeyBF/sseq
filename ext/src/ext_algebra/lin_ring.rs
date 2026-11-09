//! The multiplicative structure of `Ext_A(F_2, F_2)` by W. Lin's 2026 method ("Efficient
//! computations of Yoneda product of Ext groups", ICERM, June 2026): one chain map per basis
//! MONOMIAL, each lifted only over a corner near the origin.
//!
//! Order the generators (indecomposables) `a_1 < a_2 < ...` by `(t, s)` and take as additive basis
//! `B` the standard monomials of the relations for the order that compares sorted index lists
//! `i_1 <= i_2 <= ...` lexicographically (`a_2 a_9 < a_4 a_7`: the smallest factor as small as
//! possible). Write `lo(m) = i_1`. Then every product `a_j m` (`m ∈ B`) is
//!
//! * `P(j, m)`, read off the chain map `f_m` of `m` at the output `|a_j|`, if `j <= lo(m)`;
//! * `a_k (a_j m')` for `m = a_k m'`, `k = lo(m) < j`, otherwise -- by induction on the degree and
//!   then the left index, which drops from `j` to `k` at equal degree.
//!
//! So `f_m` is needed only at the outputs of generators `<= a_{lo(m)}` (a corner `t' <= t_{lo(m)}`),
//! and not at all when those are all `h_i` (read off the differentials, [`crate::hi_products`]):
//! every monomial with `lo(m) ∈ {h_0, .., h_3}` is free, since the first other generator, `c_0`,
//! has `t = 11`.
//!
//! [`compute`] sweeps internal degrees `t` in windows of [`WINDOW_T`]: a product read off a map lands
//! at least `t(c_0) = 11` above the monomial, so maps of monomials found in a window are lifted (over
//! their whole corner, one batched [`MultiLift`] sweep) before the next. At each bidegree the
//! candidates `a_k m'` (`m' ∈ B`, `k <= lo(m')`, all directly readable) span the decomposables;
//! taken in increasing order, the independent ones are the new basis monomials (the standard
//! monomials: a monomial is standard iff it is independent of the smaller ones, and every standard
//! monomial is a candidate since standard monomials are closed under division). The rest of
//! `Ext(b)` is spanned by new generators: the generators of the resolution (unit vectors) at the
//! non-pivot columns.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicI32, Ordering},
    },
    time::Instant,
};

use anyhow::ensure;
use fp::{prime::TWO, vector::FpVector};
use sseq::coordinates::{Bidegree, BidegreeGenerator};

use crate::{
    chain_complex::{AugmentedChainComplex, FreeChainComplex},
    hi_products::HiBlock,
    lift::{LiftRequest, Liftable, MultiLift},
    resolution_homomorphism::ResolutionHomomorphism,
};

/// Width of the `t` windows: the internal degree of `c_0`, the first generator off the `h_i` line.
/// [`compute`] checks that no generator off the `h_i` line has smaller `t`.
pub const WINDOW_T: i32 = 11;

/// The bidegrees `0 <= n <= max_n`, `0 <= s <= max_s`, `t <= max_t` over which a [`Ring`] is
/// computed. A product is known when both factors and the product lie in the range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingRange {
    pub max_n: i32,
    pub max_s: i32,
    pub max_t: i32,
}

impl RingRange {
    pub fn contains(&self, b: Bidegree) -> bool {
        (0..=self.max_s).contains(&b.s())
            && (0..=self.max_n).contains(&b.n())
            && b.t() <= self.max_t
    }
}

/// A generator `a_i`: the generator `idx` of the resolution at `b`; `h = Some(k)` for `h_k`.
pub struct Generator {
    pub b: Bidegree,
    pub idx: usize,
    pub h: Option<u32>,
}

impl Generator {
    pub fn name(&self) -> String {
        format!("x_{}_{}_{}", self.b.n(), self.b.s(), self.idx)
    }
}

/// How a basis monomial was built.
#[derive(Clone, Copy, Debug)]
pub enum Repr {
    Unit,
    Gen(usize),
    /// `a_k m'`, `k <= lo(m')`.
    Mul(usize, usize),
}

/// A basis monomial: its factors (generator indices, ascending), how it was built, and its value in
/// the resolution's dual basis of `Ext(b)`.
pub struct Monomial {
    pub b: Bidegree,
    pub factors: Vec<u32>,
    pub repr: Repr,
    pub vec: FpVector,
    /// Index into the maps, when `f_m` is lifted.
    map: Option<usize>,
}

impl Monomial {
    /// `lo(m)`; `usize::MAX` for the unit.
    fn lo(&self) -> usize {
        self.factors.first().map_or(usize::MAX, |&i| i as usize)
    }
}

/// A chain map lifted only over a downward-closed staircase of output bidegrees: row `s` through
/// stem `stair[s]` (`-1`: nothing), `stair[s - 1] >= stair[s]`.
struct Staircase<CC: FreeChainComplex> {
    hom: ResolutionHomomorphism<CC, CC>,
    stair: Arc<[AtomicI32]>,
}

impl<CC: FreeChainComplex + AugmentedChainComplex> Staircase<CC> {
    fn covers(&self, c: Bidegree) -> bool {
        c.s() >= 0
            && (c.s() as usize) < self.stair.len()
            && c.n() <= self.stair[c.s() as usize].load(Ordering::Relaxed)
    }
}

impl<CC> Liftable for Staircase<CC>
where
    CC: FreeChainComplex + AugmentedChainComplex + 'static,
{
    fn prepare(&self, b: Bidegree) -> Option<LiftRequest<'_>> {
        if !self.covers(b) {
            return None;
        }
        self.hom.prepare(b)
    }

    fn lifts_through(&self, target: &dyn std::any::Any) -> bool {
        self.hom.lifts_through(target)
    }
}

/// Per-window timings and sizes.
#[derive(Clone, Debug)]
pub struct WindowStats {
    pub t: (i32, i32),
    pub maps_lifted: usize,
    pub corner_cells: usize,
    pub lift_s: f64,
    pub decide_s: f64,
    pub new_generators: usize,
    pub new_monomials: usize,
}

/// The ring through the range, with the chain maps it lifted.
pub struct Ring<CC: FreeChainComplex> {
    pub gens: Vec<Generator>,
    pub basis: Vec<Monomial>,
    /// Per bidegree: its basis monomials, and the inverse of their matrix (row `y` = the
    /// `B`-coordinates of the resolution generator `y`).
    cells: HashMap<Bidegree, (Vec<usize>, Vec<FpVector>)>,
    maps: Vec<Arc<Staircase<CC>>>,
    /// `h_k` blocks by source: rows over the generators of the target.
    h_rows: HashMap<(u32, Bidegree), Vec<FpVector>>,
    memo: HashMap<(usize, usize), FpVector>,
    /// [`Ring::monomial_block`]'s memo.
    block_memo: HashMap<(usize, Bidegree), Arc<[FpVector]>>,
    range: RingRange,
    res: Arc<CC>,
    pub windows: Vec<WindowStats>,
    /// `(b, dim Ext(b), dim D(b))` for every non-zero bidegree in range.
    pub decomposables: Vec<(Bidegree, usize, usize)>,
    /// Candidates that were dependent: the leading terms of relations (not minimised).
    pub relations: usize,
    /// The dependent candidates all of whose divisors are standard: the leading terms of the
    /// reduced Gröbner basis of the relations, for this monomial order.
    pub gb_size: usize,
    standard: std::collections::HashSet<Vec<u32>>,
}

/// Invert a square F2 matrix `M` given by its rows: returns the rows of `M^{-1}`, so that row `y`
/// holds the coordinates of the unit vector `e_y` in the basis of `M`'s rows.
fn invert(rows: &[FpVector]) -> Vec<FpVector> {
    let d = rows.len();
    let mut a: Vec<FpVector> = rows.to_vec();
    let mut inv: Vec<FpVector> = (0..d)
        .map(|i| {
            let mut v = FpVector::new(TWO, d);
            v.set_entry(i, 1);
            v
        })
        .collect();
    for col in 0..d {
        let p = (col..d)
            .find(|&r| a[r].entry(col) != 0)
            .expect("basis monomials are independent");
        a.swap(col, p);
        inv.swap(col, p);
        for r in 0..d {
            if r != col && a[r].entry(col) != 0 {
                let (ar, ac) = (a[col].clone(), inv[col].clone());
                a[r].add(&ar, 1);
                inv[r].add(&ac, 1);
            }
        }
    }
    inv
}

impl<CC> Ring<CC>
where
    CC: FreeChainComplex + AugmentedChainComplex,
{
    fn dim(&self, b: Bidegree) -> usize {
        if self.range.contains(b) {
            self.res.number_of_gens_in_bidegree(b)
        } else {
            0
        }
    }

    /// `Σ_y v_y rows[y]`.
    fn apply(v: &FpVector, rows: &[FpVector], dim: usize) -> FpVector {
        let mut out = FpVector::new(TWO, dim);
        for (y, _) in v.iter_nonzero() {
            out.add(&rows[y], 1);
        }
        out
    }

    /// `B`-coordinates of `v ∈ Ext(b)`.
    fn coords(&self, b: Bidegree, v: &FpVector) -> Vec<usize> {
        let Some((ids, inv)) = self.cells.get(&b) else {
            assert!(v.is_zero(), "non-zero vector in {b}, which has no basis");
            return vec![];
        };
        let c = Self::apply(v, inv, ids.len());
        c.iter_nonzero().map(|(i, _)| ids[i]).collect()
    }

    /// `a_j m` read directly: an `h_k` from the differentials, otherwise `f_m` at `|a_j|`
    /// (requires `j <= lo(m)`).
    fn direct(&self, j: usize, m: usize) -> FpVector {
        let g = &self.gens[j];
        let mono = &self.basis[m];
        let target = g.b + mono.b;
        let d = self.dim(target);
        if let Repr::Unit = mono.repr {
            let mut v = FpVector::new(TWO, d);
            v.set_entry(g.idx, 1);
            return v;
        }
        if let Some(k) = g.h {
            return match self.h_rows.get(&(k, mono.b)) {
                Some(rows) => Self::apply(&mono.vec, rows, d),
                None => FpVector::new(TWO, d),
            };
        }
        assert!(j <= mono.lo(), "direct product needs j <= lo(m)");
        let map = &self.maps[mono.map.expect("monomial has no chain map")];
        let mut v = FpVector::new(TWO, d);
        if d > 0 {
            map.hom
                .act(v.as_slice_mut(), 1, BidegreeGenerator::new(g.b, g.idx));
        }
        v
    }

    /// `a_j m` for any generator and basis monomial with the product in range.
    pub fn mul(&mut self, j: usize, m: usize) -> FpVector {
        if let Some(v) = self.memo.get(&(j, m)) {
            return v.clone();
        }
        let v = match self.basis[m].repr {
            _ if self.gens[j].h.is_some() || j <= self.basis[m].lo() => self.direct(j, m),
            Repr::Unit => unreachable!(),
            // a_j a_l = a_l a_j with l < j <= lo(a_j).
            Repr::Gen(l) => {
                let mj = self.gen_monomial(j);
                self.direct(l, mj)
            }
            Repr::Mul(k, m1) => {
                let w = self.mul(j, m1);
                let wb = self.gens[j].b + self.basis[m1].b;
                self.mul_vec(k, wb, &w)
            }
        };
        self.memo.insert((j, m), v.clone());
        v
    }

    /// `a_k w` for `w ∈ Ext(b)`.
    pub fn mul_vec(&mut self, k: usize, b: Bidegree, w: &FpVector) -> FpVector {
        let d = self.dim(b + self.gens[k].b);
        let mut out = FpVector::new(TWO, d);
        for e in self.coords(b, w) {
            let p = self.mul(k, e);
            out.add(&p, 1);
        }
        out
    }

    /// The basis monomial of the generator `a_j`.
    fn gen_monomial(&self, j: usize) -> usize {
        let g = &self.gens[j];
        self.cells[&g.b]
            .0
            .iter()
            .copied()
            .find(|&m| matches!(self.basis[m].repr, Repr::Gen(l) if l == j))
            .expect("generator has a basis monomial")
    }

    /// `a_j e_y` for every resolution generator `y` of `Ext(c)`.
    pub fn block(&mut self, j: usize, c: Bidegree) -> Vec<FpVector> {
        let d = self.dim(c);
        (0..d)
            .map(|y| {
                let mut e = FpVector::new(TWO, d);
                e.set_entry(y, 1);
                self.mul_vec(j, c, &e)
            })
            .collect()
    }

    /// `m e_y` for the basis monomial `m` and every resolution generator `y` of `Ext(c)`, by
    /// multiplying in the factors of `m` one at a time. Memoised, so a product table over all
    /// monomials costs one [`Ring::mul_vec`] per block row.
    fn monomial_block(&mut self, m: usize, c: Bidegree) -> Arc<[FpVector]> {
        if let Some(v) = self.block_memo.get(&(m, c)) {
            return Arc::clone(v);
        }
        let v: Arc<[FpVector]> = match self.basis[m].repr {
            Repr::Unit => {
                let d = self.dim(c);
                (0..d)
                    .map(|y| {
                        let mut e = FpVector::new(TWO, d);
                        e.set_entry(y, 1);
                        e
                    })
                    .collect()
            }
            Repr::Gen(j) => self.block(j, c).into(),
            Repr::Mul(k, m1) => {
                let inner = self.monomial_block(m1, c);
                let wb = self.basis[m1].b + c;
                inner.iter().map(|w| self.mul_vec(k, wb, w)).collect()
            }
        };
        self.block_memo.insert((m, c), Arc::clone(&v));
        v
    }

    /// The products `e_x e_y` of the dual basis elements of `Ext(b)` and `Ext(c)` (the resolution's
    /// generators): `result[x][y] ∈ Ext(b + c)`. This is what a chain map per basis element (as in
    /// [`super::ExtModule::extend_all_products`]) computes, here by associativity
    /// from the generator maps; zero blocks when `b + c` is out of range.
    pub fn basis_block(&mut self, b: Bidegree, c: Bidegree) -> Vec<Vec<FpVector>> {
        let (db, dc, dbc) = (self.dim(b), self.dim(c), self.dim(b + c));
        let Some((ids, inv)) = self.cells.get(&b) else {
            return vec![];
        };
        let (ids, inv) = (ids.clone(), inv.clone());
        let mut out = vec![vec![FpVector::new(TWO, dbc); dc]; db];
        if dbc == 0 {
            return out;
        }
        for (i, &m) in ids.iter().enumerate() {
            let block = self.monomial_block(m, c);
            for (x, row) in inv.iter().enumerate() {
                if row.entry(i) != 0 {
                    for (o, v) in out[x].iter_mut().zip(block.iter()) {
                        o.add(v, 1);
                    }
                }
            }
        }
        out
    }

    /// Self-check: for every basis monomial `m = a_k m'` and `h_i` with index `> k`, `h_i m` from
    /// the differentials equals `a_k (h_i m')` (associativity through the chain maps). Returns
    /// `(checks, failures)`.
    pub fn check_h_associativity(&mut self) -> (usize, usize) {
        let (mut checks, mut fails) = (0, 0);
        for m in 0..self.basis.len() {
            let Repr::Mul(k, m1) = self.basis[m].repr else {
                continue;
            };
            for j in 0..self.gens.len() {
                if self.gens[j].h.is_none() || j <= k {
                    continue;
                }
                let target = self.gens[j].b + self.basis[m].b;
                if !self.range.contains(target) {
                    continue;
                }
                let free = self.direct(j, m);
                let w = self.mul(j, m1);
                let wb = self.gens[j].b + self.basis[m1].b;
                let via = self.mul_vec(k, wb, &w);
                checks += 1;
                if free != via {
                    fails += 1;
                }
            }
        }
        (checks, fails)
    }

    /// The range this ring was computed over.
    pub fn range(&self) -> RingRange {
        self.range
    }

    pub fn maps(&self) -> usize {
        self.maps.len()
    }

    /// Total output cells over all maps' staircases.
    pub fn corner_cells(&self) -> i64 {
        self.maps
            .iter()
            .map(|m| {
                m.stair
                    .iter()
                    .map(|r| (r.load(Ordering::Relaxed) + 1) as i64)
                    .sum::<i64>()
            })
            .sum()
    }
}

/// Compute the ring through `range`, given the `h_i` products `hi` (every `k` with
/// `2^k <= range.max_t`). `res` must be computed through `(range.max_n, range.max_s)`.
pub fn compute<CC>(res: &Arc<CC>, hi: &[HiBlock], range: RingRange) -> anyhow::Result<Ring<CC>>
where
    CC: FreeChainComplex + AugmentedChainComplex + Send + Sync + 'static,
{
    ensure!(res.prime() == 2, "lin_ring works at the prime 2");
    let RingRange {
        max_n,
        max_s,
        max_t,
    } = range;
    let in_range = move |b: Bidegree| range.contains(b);
    let mut ring = Ring {
        gens: vec![],
        basis: vec![],
        cells: HashMap::new(),
        maps: vec![],
        h_rows: HashMap::new(),
        memo: HashMap::new(),
        block_memo: HashMap::new(),
        range,
        res: Arc::clone(res),
        windows: vec![],
        decomposables: vec![],
        relations: 0,
        gb_size: 0,
        standard: [vec![]].into_iter().collect(),
    };
    for block in hi {
        let target = block.source + Bidegree::s_t(1, 1 << block.k);
        if !in_range(target) {
            continue;
        }
        let d = ring.dim(target);
        let rows = block
            .prod
            .iter()
            .map(|prod| {
                let mut v = FpVector::new(TWO, d);
                for &j in prod {
                    v.set_entry(j, 1);
                }
                v
            })
            .collect();
        ring.h_rows.insert((block.k, block.source), rows);
    }
    // The unit.
    let mut one = FpVector::new(TWO, 1);
    one.set_entry(0, 1);
    ring.basis.push(Monomial {
        b: Bidegree::s_t(0, 0),
        factors: vec![],
        repr: Repr::Unit,
        vec: one.clone(),
        map: None,
    });
    ring.cells.insert(Bidegree::s_t(0, 0), (vec![0], vec![one]));

    let mut pending: Vec<usize> = vec![]; // maps created since the last lift
    let t_end = max_t.min(max_n + max_s);
    let mut t0 = 1;
    while t0 <= t_end {
        let t1 = (t0 + WINDOW_T - 1).min(t_end);
        // Lift every map created in earlier windows over its whole corner.
        let t_lift = Instant::now();
        let mut bound = Bidegree::n_s(-1, -1);
        let mut cells = 0;
        let maps_lifted = pending.len();
        let liftables: Vec<Arc<dyn Liftable>> = std::mem::take(&mut pending)
            .into_iter()
            .map(|i| {
                let m = &ring.maps[i];
                for (s, r) in m.stair.iter().enumerate() {
                    let n = r.load(Ordering::Relaxed);
                    if n >= 0 {
                        bound = Bidegree::n_s(bound.n().max(n), bound.s().max(s as i32));
                        cells += (n + 1) as usize;
                    }
                }
                Arc::clone(m) as Arc<dyn Liftable>
            })
            .collect();
        if !liftables.is_empty() {
            MultiLift::new(Arc::clone(res), liftables).extend_through_stem(bound)?;
        }
        let lift_s = t_lift.elapsed().as_secs_f64();

        let t_decide = Instant::now();
        let (mut new_generators, mut new_monomials) = (0, 0);
        for t in t0..=t1 {
            for s in 0..=t.min(max_s) {
                let b = Bidegree::s_t(s, t);
                if !in_range(b) {
                    continue;
                }
                let d = ring.dim(b);
                if d == 0 {
                    continue;
                }
                // Candidates a_k m', k <= lo(m'), in increasing monomial order.
                let mut cands: Vec<(Vec<u32>, usize, usize)> = vec![];
                for (k, g) in ring.gens.iter().enumerate() {
                    let c = b - g.b;
                    let Some((ids, _)) = ring.cells.get(&c) else {
                        continue;
                    };
                    for &m in ids {
                        let mono = &ring.basis[m];
                        if mono.factors.is_empty() || k > mono.lo() {
                            continue;
                        }
                        let mut f = Vec::with_capacity(mono.factors.len() + 1);
                        f.push(k as u32);
                        f.extend_from_slice(&mono.factors);
                        cands.push((f, k, m));
                    }
                }
                cands.sort();
                let mut ids = vec![];
                let mut rows: Vec<FpVector> = vec![];
                let mut ech: Vec<(usize, FpVector)> = vec![];
                for (factors, k, m) in cands {
                    let v = ring.direct(k, m);
                    let mut r = v.clone();
                    for (p, row) in &ech {
                        if r.entry(*p) != 0 {
                            r.add(row, 1);
                        }
                    }
                    let Some((p, _)) = r.first_nonzero() else {
                        ring.relations += 1;
                        let minimal = (0..factors.len()).all(|i| {
                            let mut d = factors.clone();
                            d.remove(i);
                            ring.standard.contains(&d)
                        });
                        if minimal {
                            ring.gb_size += 1;
                        }
                        continue;
                    };
                    ring.standard.insert(factors.clone());
                    ech.push((p, r));
                    ids.push(ring.basis.len());
                    rows.push(v.clone());
                    ring.basis.push(Monomial {
                        b,
                        factors,
                        repr: Repr::Mul(k, m),
                        vec: v,
                        map: None,
                    });
                    new_monomials += 1;
                }
                ring.decomposables.push((b, d, ech.len()));
                let mut pivot = vec![false; d];
                for (p, _) in &ech {
                    pivot[*p] = true;
                }
                for idx in (0..d).filter(|&i| !pivot[i]) {
                    let j = ring.gens.len();
                    let h = (s == 1).then(|| t.trailing_zeros());
                    ensure!(
                        s != 1 || (t as u32).is_power_of_two(),
                        "a generator at {b} in filtration 1 is not an h_i"
                    );
                    ensure!(
                        h.is_some() || t >= WINDOW_T,
                        "a generator at {b} off the h_i line below t = {WINDOW_T}"
                    );
                    ring.gens.push(Generator { b, idx, h });
                    ring.standard.insert(vec![j as u32]);
                    let mut v = FpVector::new(TWO, d);
                    v.set_entry(idx, 1);
                    ids.push(ring.basis.len());
                    rows.push(v.clone());
                    ring.basis.push(Monomial {
                        b,
                        factors: vec![j as u32],
                        repr: Repr::Gen(j),
                        vec: v,
                        map: None,
                    });
                    new_generators += 1;
                }
                // New monomials whose corner holds a generator off the h_i line get a chain map.
                for &m in &ids {
                    let lo = ring.basis[m].lo();
                    let mut stair = vec![-1i32; (max_s - s + 1) as usize];
                    for g in ring.gens[..=lo].iter().filter(|g| g.h.is_none()) {
                        // Row s' <= s_g through stem t_g - s', within the target range.
                        for (sp, row) in stair.iter_mut().enumerate().take(g.b.s() as usize + 1) {
                            let n = (g.b.t() - sp as i32).min(max_n - b.n());
                            *row = (*row).max(n);
                        }
                    }
                    if stair.iter().all(|&n| n < 0) {
                        continue;
                    }
                    let mono = &ring.basis[m];
                    let class: Vec<u32> = mono.vec.iter().collect();
                    let hom = ResolutionHomomorphism::from_class(
                        String::new(),
                        Arc::clone(res),
                        Arc::clone(res),
                        b,
                        &class,
                    );
                    ring.basis[m].map = Some(ring.maps.len());
                    pending.push(ring.maps.len());
                    ring.maps.push(Arc::new(Staircase {
                        hom,
                        stair: stair.into_iter().map(AtomicI32::new).collect(),
                    }));
                }
                ensure!(
                    ids.len() == d,
                    "{b}: {} basis monomials, dim {d}",
                    ids.len()
                );
                let inv = invert(&rows);
                ring.cells.insert(b, (ids, inv));
            }
        }
        let stats = WindowStats {
            t: (t0, t1),
            maps_lifted,
            corner_cells: cells,
            lift_s,
            decide_s: t_decide.elapsed().as_secs_f64(),
            new_generators,
            new_monomials,
        };
        tracing::info!(
            ?stats,
            gens = ring.gens.len(),
            basis = ring.basis.len(),
            maps = ring.maps.len(),
            "lin_ring window"
        );
        ring.windows.push(stats);
        t0 = t1 + 1;
    }
    Ok(ring)
}
