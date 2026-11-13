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
//! So `f_m` is needed only at the outputs of generators `<= a_{lo(m)}`: reaching `a_j` takes the rows
//! `s' <= s_j` through stem `n_j`, since lifting goes by stem as resolving does,
//! and not at all when those are all `h_i` (read off the differentials, [`super::h_product`]):
//! every monomial with `lo(m) ∈ {h_0, .., h_3}` is free, since the first other generator, `c_0`,
//! has `t = 11`.
//!
//! [`compute`] sweeps internal degrees `t` in windows of [`WINDOW_T`]: a product read off a map lands
//! at least `t(c_0) = 11` above the monomial, so maps of monomials found in a window are lifted (over
//! their whole corner, one batched [`MultiLift`] sweep) before the next. With the generators in
//! [`GenOrder::Stem`] it sweeps stems instead, in windows of [`WINDOW_N`] = `n(c_0)` = 8, so that
//! after each window the whole ring is known through a stem. At each bidegree the
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
        atomic::{AtomicI32, AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::ensure;
use fp::{matrix::Matrix, prime::TWO, vector::FpVector};
use maybe_rayon::prelude::*;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

use crate::{
    chain_complex::{AugmentedChainComplex, FreeChainComplex},
    lift::{LiftRequest, Liftable, MultiLift},
    resolution_homomorphism::ResolutionHomomorphism,
};

/// Width of the `t` windows: the internal degree of `c_0`, the first generator off the `h_i` line.
/// [`compute`] checks that no generator off the `h_i` line has smaller `t`.
pub const WINDOW_T: i32 = 11;

/// Width of the stem windows of a [`GenOrder::Stem`] computation: the stem of `c_0`, the first
/// generator off the `h_i` line by stem. [`compute`] checks that no generator off the `h_i` line has
/// smaller stem.
pub const WINDOW_N: i32 = 8;

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

/// The name of the chain map of the monomial with these `factors`: `lin_ring/` and the factors as
/// `x_(n, s, i)`, joined by `.`. Basis monomials and their classes depend only on the resolution,
/// so the name identifies the same map in any range.
pub fn map_name(gens: &[BidegreeGenerator], factors: &[u32]) -> String {
    let parts: Vec<String> = factors
        .iter()
        .map(|&j| format!("x_{}", gens[j as usize]))
        .collect();
    format!("lin_ring/{}", parts.join("."))
}

/// `Some(k)` if the generator `g` is `h_k`. Every generator in filtration 1 is an `h_k`, `k = log_2 t`
/// ([`compute`] checks this).
fn h_index(g: BidegreeGenerator) -> Option<u32> {
    (g.s() == 1).then(|| g.t().trailing_zeros())
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
    /// [`map_name`] of the monomial, for the step log whether or not the map is saved.
    name: String,
    /// Steps prepared and their thread time since the last [`Staircase::take_window`].
    steps: AtomicU64,
    prepare_ns: AtomicU64,
}

/// `EXT_STEP_LOG_MIN_MS` (default 1): the shortest map step the step log records individually.
static STEP_LOG_MIN_NS: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
    let ms: f64 = std::env::var("EXT_STEP_LOG_MIN_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    (ms * 1e6) as u64
});

impl<CC: FreeChainComplex> Staircase<CC> {
    /// The steps prepared and their thread time in seconds since the last call, resetting both.
    fn take_window(&self) -> (u64, f64) {
        (
            self.steps.swap(0, Ordering::Relaxed),
            self.prepare_ns.swap(0, Ordering::Relaxed) as f64 / 1e9,
        )
    }
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
        let start = Instant::now();
        let req = self.hom.prepare(b)?;
        let ns = start.elapsed().as_nanos() as u64;
        self.steps.fetch_add(1, Ordering::Relaxed);
        self.prepare_ns.fetch_add(ns, Ordering::Relaxed);
        if ns >= *STEP_LOG_MIN_NS {
            tracing::info!(
                target: "ext::lift::steps",
                kind = "step",
                map = ?self.name,
                n = b.n(),
                s = b.s(),
                inputs = req.inputs.len(),
                prepare_ms = ns as f64 / 1e6,
            );
        }
        Some(req)
    }

    fn lifts_through(&self, target: &dyn std::any::Any) -> bool {
        self.hom.lifts_through(target)
    }

    /// As the map's own, inside the staircase; a deferred step is counted but not timed, since its
    /// products are multiplied with the other maps' at the bidegree.
    fn prepare_batched(&self, b: Bidegree) -> crate::lift::Prepared<'_> {
        use crate::lift::Prepared;
        if !self.covers(b) {
            return Prepared::Ready(None);
        }
        let start = Instant::now();
        let prepared = self.hom.prepare_batched(b);
        let ns = start.elapsed().as_nanos() as u64;
        self.steps.fetch_add(1, Ordering::Relaxed);
        self.prepare_ns.fetch_add(ns, Ordering::Relaxed);
        // A deferred step is logged with its product terms when they are built.
        prepared
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

/// One map's share of a window's lift, as [`compute`] hands it to `EXT_LIN_RING_DISPATCH` and
/// [`lift_jobs`] takes it back: the monomial's map, its staircase for the window, and what lifting
/// it cost in the window before.
#[derive(Clone, Debug, PartialEq)]
pub struct MapJob {
    /// The map's [`map_name`], which is where its steps are saved.
    pub name: String,
    pub b: Bidegree,
    /// The monomial's class in the resolution's dual basis of `Ext(b)`.
    pub class: Vec<u32>,
    /// Row `s` of the staircase through stem `stair[s]`, as in [`Staircase`].
    pub stair: Vec<i32>,
    /// Thread seconds the map's lift took in the window before; 0 if unknown.
    pub cost_s: f64,
}

impl MapJob {
    /// One tab-separated line: name, stem, filtration, class as a string of digits, staircase
    /// comma-separated, cost. Map names hold no tabs.
    pub fn to_line(&self) -> String {
        let class: String = self.class.iter().map(|c| c.to_string()).collect();
        let stair: Vec<String> = self.stair.iter().map(|n| n.to_string()).collect();
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.name,
            self.b.n(),
            self.b.s(),
            class,
            stair.join(","),
            self.cost_s
        )
    }

    /// The inverse of [`MapJob::to_line`].
    pub fn parse(line: &str) -> anyhow::Result<Self> {
        let f: Vec<&str> = line.trim_end_matches(['\n', '\r']).split('\t').collect();
        ensure!(
            f.len() == 6,
            "a map job has 6 fields, not {}: {line:?}",
            f.len()
        );
        let class = f[3]
            .chars()
            .map(|c| {
                c.to_digit(10)
                    .ok_or_else(|| anyhow::anyhow!("bad class digit in {line:?}"))
            })
            .collect::<anyhow::Result<_>>()?;
        let stair = f[4].split(',').map(str::parse).collect::<Result<_, _>>()?;
        Ok(Self {
            name: f[0].to_string(),
            b: Bidegree::n_s(f[1].parse()?, f[2].parse()?),
            class,
            stair,
            cost_s: f[5].parse()?,
        })
    }
}

/// Lift the maps of `jobs` over their staircases, saving each step in `res`'s store under the map's
/// name, and return each map's name and the thread seconds its new steps took. Steps the store
/// already has are read rather than lifted, so this resumes a partial run, and a map's earlier
/// windows must be in the store. This is a worker's half of `EXT_LIN_RING_DISPATCH` (see
/// [`compute`]); the result is the store.
pub fn lift_jobs<CC>(res: &Arc<CC>, jobs: &[MapJob]) -> anyhow::Result<Vec<(String, f64)>>
where
    CC: FreeChainComplex + AugmentedChainComplex + Send + Sync + 'static,
{
    ensure!(
        res.save_dir().store().is_some(),
        "lifting map jobs needs a save store to read and keep their steps"
    );
    let mut bound = Bidegree::n_s(-1, -1);
    let mut maps = vec![];
    for job in jobs {
        let hom = ResolutionHomomorphism::from_class(
            job.name.clone(),
            Arc::clone(res),
            Arc::clone(res),
            job.b,
            &job.class,
        );
        for (sp, &n) in job.stair.iter().enumerate() {
            if n >= 0 {
                bound = Bidegree::n_s(bound.n().max(n), bound.s().max(sp as i32));
            }
        }
        maps.push(Arc::new(Staircase {
            hom,
            name: job.name.clone(),
            steps: AtomicU64::new(0),
            prepare_ns: AtomicU64::new(0),
            stair: job.stair.iter().map(|&n| AtomicI32::new(n)).collect(),
        }));
    }
    if !maps.is_empty() {
        let liftables = maps
            .iter()
            .map(|m| Arc::clone(m) as Arc<dyn Liftable>)
            .collect();
        MultiLift::new(Arc::clone(res), liftables).extend_through_stem(bound)?;
    }
    Ok(maps
        .iter()
        .map(|m| (m.name.clone(), m.take_window().1))
        .collect())
}

/// The dispatch settings of [`compute`], from the environment.
struct Dispatch {
    /// `EXT_LIN_RING_DISPATCH`: the command, run with the jobs file as its one argument.
    command: std::ffi::OsString,
    /// `EXT_LIN_RING_DISPATCH_DIR`: where the jobs files go.
    dir: std::path::PathBuf,
    /// `EXT_LIN_RING_DISPATCH_MIN_S` (default 60): the cost from which a map is dispatched.
    min_s: f64,
    /// `EXT_LIN_RING_DISPATCH_COSTS`: `name<TAB>seconds` lines, the costs of maps lifted before
    /// this process started (for instance the last window's `map_window` events of the step log).
    hints: HashMap<String, f64>,
}

impl Dispatch {
    fn from_env() -> anyhow::Result<Option<Self>> {
        let Some(command) = std::env::var_os("EXT_LIN_RING_DISPATCH") else {
            return Ok(None);
        };
        let dir = std::env::var_os("EXT_LIN_RING_DISPATCH_DIR")
            .ok_or_else(|| {
                anyhow::anyhow!("EXT_LIN_RING_DISPATCH needs EXT_LIN_RING_DISPATCH_DIR")
            })?
            .into();
        let min_s = match std::env::var("EXT_LIN_RING_DISPATCH_MIN_S") {
            Ok(v) => v
                .parse()
                .map_err(|e| anyhow::anyhow!("EXT_LIN_RING_DISPATCH_MIN_S={v:?}: {e}"))?,
            Err(_) => 60.0,
        };
        let mut hints = HashMap::new();
        if let Some(path) = std::env::var_os("EXT_LIN_RING_DISPATCH_COSTS") {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("EXT_LIN_RING_DISPATCH_COSTS {path:?}: {e}"))?;
            for line in text.lines().filter(|l| !l.is_empty()) {
                let (name, cost) = line
                    .rsplit_once('\t')
                    .ok_or_else(|| anyhow::anyhow!("bad cost line {line:?}"))?;
                hints.insert(name.to_string(), cost.parse()?);
            }
        }
        Ok(Some(Self {
            command,
            dir,
            min_s,
            hints,
        }))
    }
}

/// The order of the generators, which fixes Lin's monomial basis and the corners of the maps.
///
/// `T`, by `(t, s)`, is the order the generators are found in. `Stem`, by `(n, s)`, reads every
/// product at its factor of smaller stem, so no map is lifted past stem `max_n / 2` and every
/// quasi-inverse used lies there. Both give the same generators, products and ranks; the basis
/// monomials and so the map names differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenOrder {
    T,
    Stem,
}

impl GenOrder {
    /// `EXT_LIN_RING_ORDER`: `t` (the default) or `stem`.
    fn from_env() -> anyhow::Result<Self> {
        match std::env::var("EXT_LIN_RING_ORDER").as_deref() {
            Err(_) | Ok("t") => Ok(Self::T),
            Ok("stem") => Ok(Self::Stem),
            Ok(v) => anyhow::bail!("EXT_LIN_RING_ORDER={v:?}: expected t or stem"),
        }
    }
}

/// The ring through the range, with the chain maps it lifted.
pub struct Ring<CC: FreeChainComplex> {
    order: GenOrder,
    pub gens: Vec<BidegreeGenerator>,
    pub basis: Vec<Monomial>,
    /// Per bidegree: its basis monomials, and the inverse of their matrix (row `y` = the
    /// `B`-coordinates of the resolution generator `y`).
    cells: HashMap<Bidegree, (Vec<usize>, Vec<FpVector>)>,
    maps: Vec<Arc<Staircase<CC>>>,
    /// `h_k` blocks by source: rows over the generators of the target.
    h_rows: HashMap<(u32, Bidegree), Matrix>,
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
    /// The degree [`compute`]'s windows sweep: `t` in [`GenOrder::T`], the stem in
    /// [`GenOrder::Stem`].
    pub fn window_degree(&self, b: Bidegree) -> i32 {
        match self.order {
            GenOrder::T => b.t(),
            GenOrder::Stem => b.n(),
        }
    }

    /// The sort key of generator `j` in [`GenOrder`].
    fn key(&self, j: usize) -> (i32, i32, usize) {
        let g = self.gens[j];
        match self.order {
            GenOrder::T => (g.t(), g.s(), g.idx()),
            GenOrder::Stem => (g.n(), g.s(), g.idx()),
        }
    }

    /// `a_j <= a_lo`, with `lo` from [`Monomial::lo`] (the unit's `usize::MAX` is above all).
    fn le(&self, j: usize, lo: usize) -> bool {
        lo == usize::MAX || self.key(j) <= self.key(lo)
    }

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

    /// `Σ_y v_y matrix[y]`.
    fn apply_matrix(v: &FpVector, matrix: &Matrix, dim: usize) -> FpVector {
        let mut out = FpVector::new(TWO, dim);
        for (y, _) in v.iter_nonzero() {
            out.as_slice_mut().add(matrix.row(y), 1);
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
        let g = self.gens[j];
        let mono = &self.basis[m];
        let target = g.degree() + mono.b;
        let d = self.dim(target);
        if let Repr::Unit = mono.repr {
            let mut v = FpVector::new(TWO, d);
            v.set_entry(g.idx(), 1);
            return v;
        }
        if let Some(k) = h_index(g) {
            return match self.h_rows.get(&(k, mono.b)) {
                Some(rows) => Self::apply_matrix(&mono.vec, rows, d),
                None => FpVector::new(TWO, d),
            };
        }
        assert!(self.le(j, mono.lo()), "direct product needs a_j <= a_lo(m)");
        let map = &self.maps[mono.map.expect("monomial has no chain map")];
        let mut v = FpVector::new(TWO, d);
        if d > 0 {
            map.hom.act(v.as_slice_mut(), 1, g);
        }
        v
    }

    /// `a_j m` for any generator and basis monomial with the product in range.
    pub fn mul(&mut self, j: usize, m: usize) -> FpVector {
        if let Some(v) = self.memo.get(&(j, m)) {
            return v.clone();
        }
        let v = match self.basis[m].repr {
            _ if h_index(self.gens[j]).is_some() || self.le(j, self.basis[m].lo()) => {
                self.direct(j, m)
            }
            Repr::Unit => unreachable!(),
            // a_j a_l = a_l a_j with a_l < a_j = a_lo(a_j).
            Repr::Gen(l) => {
                let mj = self.gen_monomial(j);
                self.direct(l, mj)
            }
            Repr::Mul(k, m1) => {
                let w = self.mul(j, m1);
                let wb = self.gens[j].degree() + self.basis[m1].b;
                self.mul_vec(k, wb, &w)
            }
        };
        self.memo.insert((j, m), v.clone());
        v
    }

    /// `a_k w` for `w ∈ Ext(b)`.
    pub fn mul_vec(&mut self, k: usize, b: Bidegree, w: &FpVector) -> FpVector {
        let d = self.dim(b + self.gens[k].degree());
        let mut out = FpVector::new(TWO, d);
        for e in self.coords(b, w) {
            let p = self.mul(k, e);
            out.add(&p, 1);
        }
        out
    }

    /// The basis monomial of the generator `a_j`.
    fn gen_monomial(&self, j: usize) -> usize {
        self.cells[&self.gens[j].degree()]
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
                if h_index(self.gens[j]).is_none() || self.le(j, k) {
                    continue;
                }
                let target = self.gens[j].degree() + self.basis[m].b;
                if !self.range.contains(target) {
                    continue;
                }
                let free = self.direct(j, m);
                let w = self.mul(j, m1);
                let wb = self.gens[j].degree() + self.basis[m1].b;
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

/// Compute the ring through `range`. `res` must be a minimal resolution of `F_2` over the Milnor
/// basis, computed through `(range.max_n, range.max_s)`.
///
/// With `save_maps`, each chain map is named after its monomial ([`map_name`]), so a resolution
/// with a save store keeps the map's steps under `products/<name>` and a later computation, over
/// this range or a larger one, reads them back instead of lifting them again.
///
/// Each map is lifted only as far as the next window reads it, so the decided degrees advance at
/// the same pace whatever the range. After each window `on_window(ring, t0, t1)` runs, with every
/// product landing in internal degrees `t0..=t1` available. If the file named by
/// `EXT_LIN_RING_STOP` exists after a window, the computation stops there and the ring's range is
/// cut to the degrees decided.
///
/// With `EXT_LIN_RING_DISPATCH` set (and `save_maps`), each window's maps that cost at least
/// `EXT_LIN_RING_DISPATCH_MIN_S` seconds in their last window are handed to that command, as
/// [`MapJob`] lines in a file under `EXT_LIN_RING_DISPATCH_DIR`, while this process lifts the rest.
/// The command is to lift them elsewhere ([`lift_jobs`]) and put their steps in this store before
/// it exits; it may write the maps' new costs, as `name<TAB>seconds` lines, to the jobs file's
/// name with `.costs` appended. Afterwards the dispatched maps are lifted here too, which reads the
/// steps the command saved and computes any it did not, so a failed dispatch costs time, never
/// correctness.
pub fn compute<CC>(
    res: &Arc<CC>,
    range: RingRange,
    save_maps: bool,
    on_window: &mut dyn FnMut(&mut Ring<CC>, i32, i32) -> anyhow::Result<()>,
) -> anyhow::Result<Ring<CC>>
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
        order: GenOrder::from_env()?,
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
    // The `h_k` blocks with source and target in range, read off the differentials.
    let jobs: Vec<(u32, Bidegree)> = (0..=max_t.max(1).ilog2())
        .flat_map(|k| {
            res.iter_stem()
                .filter(move |&b| {
                    let target = b + Bidegree::s_t(1, 1 << k);
                    in_range(b)
                        && in_range(target)
                        && res.number_of_gens_in_bidegree(b) > 0
                        && res.number_of_gens_in_bidegree(target) > 0
                })
                .map(move |b| (k, b))
        })
        .collect();
    ring.h_rows = jobs
        .into_maybe_par_iter()
        .map(|(k, b)| {
            let m = super::h_product(&**res, k, b)
                .ok_or_else(|| anyhow::anyhow!("h_{k} products at {b} are not readable"))?;
            Ok(((k, b), m))
        })
        .collect::<anyhow::Result<_>>()?;
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

    // Per map: its monomial, and how many of the generators it reads have been added to its
    // staircase. The generators are in `(t, s)` order, so those `f_m` must reach by degree `t` are
    // a prefix of the ones up to `lo(m)`.
    let mut map_mono: Vec<usize> = vec![];
    let mut cursor: Vec<usize> = vec![];
    let stop_file = std::env::var_os("EXT_LIN_RING_STOP").map(std::path::PathBuf::from);
    let dispatch = Dispatch::from_env()?;
    ensure!(
        dispatch.is_none() || save_maps,
        "EXT_LIN_RING_DISPATCH hands maps over through the store, so it needs save_maps"
    );
    // Per map: the thread seconds its last lift took, which decides whether it is dispatched.
    let mut last_cost: Vec<f64> = vec![];
    // Windows of `t` in the `t` order, of stems in the stem order, so that the order in which
    // generators are found is the order they are compared in. `t0..=t1` is the window, in that
    // degree.
    let by_stem = ring.order == GenOrder::Stem;
    let (width, t_end, mut t0) = if by_stem {
        (WINDOW_N, max_n, 0)
    } else {
        (WINDOW_T, max_t.min(max_n + max_s), 1)
    };
    let deg = move |b: Bidegree| if by_stem { b.n() } else { b.t() };
    while t0 <= t_end {
        let t1 = (t0 + width - 1).min(t_end);
        // Extend every map to the generators this window reads: `a_j <= a_lo(m)` with
        // `|m| + |a_j|` in range and in the window or before.
        let t_lift = Instant::now();
        let mut bound = Bidegree::n_s(-1, -1);
        let mut cells = 0;
        let mut grown: Vec<usize> = vec![];
        for (i, map) in ring.maps.iter().enumerate() {
            let mono = &ring.basis[map_mono[i]];
            let (b, lo) = (mono.b, mono.lo());
            let mut grew = false;
            while cursor[i] < ring.gens.len() && deg(ring.gens[cursor[i]].degree()) <= t1 - deg(b) {
                let (j, g) = (cursor[i], ring.gens[cursor[i]]);
                cursor[i] += 1;
                if !ring.le(j, lo) || h_index(g).is_some() || !in_range(b + g.degree()) {
                    continue;
                }
                // Rows s' <= s_g through stem n_g, within the target range. That is all reading at
                // `g` needs: a resolution is minimal, so `d` of a generator involves only generators
                // of lower degree, and a lift step at stem `n` reads the row below through stem `n`
                // (as resolving through a stem does).
                for row in map.stair.iter().take(g.s() as usize + 1) {
                    let n = g.n().min(max_n - b.n());
                    let old = row.load(Ordering::Relaxed);
                    if n > old {
                        row.store(n, Ordering::Relaxed);
                        cells += (n - old.max(-1)) as usize;
                        grew = true;
                    }
                }
            }
            if grew {
                for (sp, r) in map.stair.iter().enumerate() {
                    let n = r.load(Ordering::Relaxed);
                    if n >= 0 {
                        bound = Bidegree::n_s(bound.n().max(n), bound.s().max(sp as i32));
                    }
                }
                grown.push(i);
            }
        }
        let maps_lifted = grown.len();
        let (remote, local): (Vec<usize>, Vec<usize>) = match &dispatch {
            Some(d) => grown.into_iter().partition(|&i| last_cost[i] >= d.min_s),
            None => (vec![], grown),
        };
        let lift = |which: &[usize]| -> anyhow::Result<()> {
            if !which.is_empty() {
                let liftables = which
                    .iter()
                    .map(|&i| Arc::clone(&ring.maps[i]) as Arc<dyn Liftable>)
                    .collect();
                MultiLift::new(Arc::clone(res), liftables).extend_through_stem(bound)?;
            }
            Ok(())
        };
        if let Some(d) = dispatch.as_ref().filter(|_| !remote.is_empty()) {
            let jobs_path = d.dir.join(format!("window_{t0}.tsv"));
            let mut text = String::new();
            for &i in &remote {
                let map = &ring.maps[i];
                let mono = &ring.basis[map_mono[i]];
                let job = MapJob {
                    name: map.name.clone(),
                    b: mono.b,
                    class: mono.vec.iter().collect(),
                    stair: map
                        .stair
                        .iter()
                        .map(|r| r.load(Ordering::Relaxed))
                        .collect(),
                    cost_s: last_cost[i],
                };
                text.push_str(&job.to_line());
                text.push('\n');
            }
            std::fs::create_dir_all(&d.dir)?;
            std::fs::write(&jobs_path, text)?;
            tracing::info!(
                t0,
                remote = remote.len(),
                local = local.len(),
                remote_cost_s = remote.iter().map(|&i| last_cost[i]).sum::<f64>(),
                jobs = ?jobs_path,
                "lin_ring dispatch"
            );
            let child = std::process::Command::new(&d.command)
                .arg(&jobs_path)
                .spawn();
            lift(&local)?;
            match child.and_then(|mut c| c.wait()) {
                Ok(status) if status.success() => {}
                Ok(status) => tracing::warn!(%status, "lin_ring dispatch failed; lifting here"),
                Err(e) => tracing::warn!(%e, "lin_ring dispatch did not run; lifting here"),
            }
            let mut costs_path = jobs_path.into_os_string();
            costs_path.push(".costs");
            if let Ok(text) = std::fs::read_to_string(&costs_path) {
                let by_name: HashMap<&str, usize> =
                    remote.iter().map(|&i| (&*ring.maps[i].name, i)).collect();
                for line in text.lines() {
                    if let Some((name, cost)) = line.rsplit_once('\t')
                        && let (Some(&i), Ok(cost)) = (by_name.get(name), cost.parse::<f64>())
                    {
                        last_cost[i] = cost;
                    }
                }
            }
            let start = Instant::now();
            lift(&remote)?;
            tracing::info!(
                t0,
                merge_s = start.elapsed().as_secs_f64(),
                "lin_ring dispatched maps read back"
            );
        } else {
            lift(&local)?;
        }
        for (i, map) in ring.maps.iter().enumerate() {
            let (steps, prepare_s) = map.take_window();
            if steps > 0 {
                last_cost[i] = prepare_s;
                tracing::info!(
                    target: "ext::lift::steps",
                    kind = "map_window",
                    map = ?map.name,
                    t0,
                    t1,
                    steps,
                    prepare_s,
                );
            }
        }
        let lift_s = t_lift.elapsed().as_secs_f64();

        let t_decide = Instant::now();
        let (mut new_generators, mut new_monomials) = (0, 0);
        for w in t0..=t1 {
            for s in 0..=if by_stem { max_s } else { w.min(max_s) } {
                let b = if by_stem {
                    Bidegree::n_s(w, s)
                } else {
                    Bidegree::s_t(s, w)
                };
                let t = b.t();
                // The unit is in place from the start.
                if !in_range(b) || (b.n() == 0 && s == 0) {
                    continue;
                }
                let d = ring.dim(b);
                if d == 0 {
                    continue;
                }
                // Candidates a_k m', k <= lo(m'), in increasing monomial order.
                let mut cands: Vec<(Vec<u32>, usize, usize)> = vec![];
                for (k, g) in ring.gens.iter().enumerate() {
                    let c = b - g.degree();
                    let Some((ids, _)) = ring.cells.get(&c) else {
                        continue;
                    };
                    for &m in ids {
                        let mono = &ring.basis[m];
                        if mono.factors.is_empty() || !ring.le(k, mono.lo()) {
                            continue;
                        }
                        let mut f = Vec::with_capacity(mono.factors.len() + 1);
                        f.push(k as u32);
                        f.extend_from_slice(&mono.factors);
                        cands.push((f, k, m));
                    }
                }
                cands.sort_by_cached_key(|(f, _, _)| {
                    f.iter().map(|&j| ring.key(j as usize)).collect::<Vec<_>>()
                });
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
                    let g = BidegreeGenerator::new(b, idx);
                    ensure!(
                        s != 1 || (t as u32).is_power_of_two(),
                        "a generator at {b} in filtration 1 is not an h_i"
                    );
                    ensure!(
                        s == 1 || deg(b) >= width,
                        "a generator at {b} off the h_i line below the window width {width}"
                    );
                    ring.gens.push(g);
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
                    // A map exists if some generator it would be read at (off the h_i line, product
                    // in range) gives it a non-empty corner; it starts empty and grows per window.
                    //
                    // In stem order a generator found later (larger t) can still be below a_lo, so
                    // there the test is whether c_0 = x_(8, 3, 0), the first generator off the h_i
                    // line by stem, is below a_lo and could be read.
                    let reads_any = match ring.order {
                        GenOrder::T => ring.gens[..=lo].iter().any(|&g| {
                            h_index(g).is_none()
                                && in_range(b + g.degree())
                                && g.n().min(max_n - b.n()) >= 0
                        }),
                        GenOrder::Stem => {
                            ring.key(lo) >= (8, 3, 0) && in_range(b + Bidegree::n_s(8, 3))
                        }
                    };
                    if !reads_any {
                        continue;
                    }
                    let stair = vec![-1i32; (max_s - s + 1) as usize];
                    let mono = &ring.basis[m];
                    let class: Vec<u32> = mono.vec.iter().collect();
                    let label = map_name(&ring.gens, &mono.factors);
                    let name = if save_maps {
                        label.clone()
                    } else {
                        String::new()
                    };
                    let hom = ResolutionHomomorphism::from_class(
                        name,
                        Arc::clone(res),
                        Arc::clone(res),
                        b,
                        &class,
                    );
                    ring.basis[m].map = Some(ring.maps.len());
                    map_mono.push(m);
                    cursor.push(0);
                    last_cost.push(
                        dispatch
                            .as_ref()
                            .and_then(|d| d.hints.get(&label).copied())
                            .unwrap_or(0.0),
                    );
                    ring.maps.push(Arc::new(Staircase {
                        hom,
                        name: label,
                        steps: AtomicU64::new(0),
                        prepare_ns: AtomicU64::new(0),
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
        if crate::resolution_homomorphism::lift_batch::enabled() {
            tracing::info!(
                t1,
                "{}",
                crate::resolution_homomorphism::lift_batch::summary()
            );
        }
        tracing::info!(
            ?stats,
            gens = ring.gens.len(),
            basis = ring.basis.len(),
            maps = ring.maps.len(),
            "lin_ring window"
        );
        ring.windows.push(stats);
        on_window(&mut ring, t0, t1)?;
        if t1 < t_end && stop_file.as_ref().is_some_and(|f| f.exists()) {
            tracing::info!(window_end = t1, "lin_ring stopped by EXT_LIN_RING_STOP");
            if by_stem {
                ring.range.max_n = t1;
            } else {
                ring.range.max_t = t1;
            }
            break;
        }
        t0 = t1 + 1;
    }
    Ok(ring)
}
