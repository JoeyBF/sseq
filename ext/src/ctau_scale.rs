//! The $C\tau$ Adams $E_2$ page at scale: one sparse reduction per filtration and weight.

use std::sync::Arc;

use algebra::{Algebra, MilnorAlgebra, module::FreeModule, module::Module};
use byteorder::{LittleEndian, ReadBytesExt};
use fp::{
    matrix::{SparseEchelon, SparseWorkspace},
    prime::TWO,
    vector::FpVector,
};
use rustc_hash::{FxHashMap, FxHashSet};
use sseq::coordinates::Bidegree;

use crate::{
    ctau::{
        BarCocycle, OpTable, PackedXi, QAlgebra, QMono, fits, mono_degree, mono_len, xi_degree,
    },
    save::{SaveKind, ZarrSaveStore},
};

/// A term $P(R) x$ of $d z_l$, filed under the generator $x$ it lands on: `l` indexes $z_l$ among
/// the generators of degree `t`.
#[derive(Debug, Clone, Copy)]
pub struct InvTerm {
    pub op: PackedXi,
    pub t: u16,
    pub l: u32,
}

/// The terms of the differentials of $P_a$ that land on one generator $x$ of $P_{a-1}$, grouped
/// by operation.
#[derive(Default)]
pub struct GenTerms {
    /// Each operation $P(R)$ with its degree and its range of `targets`, sorted by degree.
    ops: Vec<(PackedXi, u16, u32, u32)>,
    /// The position in `ops` of each operation.
    by_op: FxHashMap<PackedXi, u32>,
    /// The generators $z_l$, as (degree, index), with $P(R) x$ in $d z_l$.
    targets: Vec<(u16, u32)>,
}

impl GenTerms {
    /// Groups `terms`, which land on a generator of degree `t`.
    fn new(mut terms: Vec<InvTerm>, t: u16) -> Self {
        terms.sort_unstable_by_key(|x| (x.t, x.op, x.l));
        let mut ops = Vec::new();
        let mut targets = Vec::with_capacity(terms.len());
        for run in terms.chunk_by(|x, y| x.t == y.t && x.op == y.op) {
            let lo = targets.len() as u32;
            targets.extend(run.iter().map(|x| (x.t, x.l)));
            ops.push((run[0].op, run[0].t - t, lo, targets.len() as u32));
        }
        let by_op = ops
            .iter()
            .enumerate()
            .map(|(i, x)| (x.0, i as u32))
            .collect();
        Self {
            ops,
            by_op,
            targets,
        }
    }

    /// The generators $z_l$ with $P(R) x$ in $d z_l$.
    pub fn targets(&self, op: PackedXi) -> &[(u16, u32)] {
        self.by_op.get(&op).map_or(&[], |&i| {
            let (_, _, lo, hi) = self.ops[i as usize];
            &self.targets[lo as usize..hi as usize]
        })
    }

    /// The operations of degree at most `degree`, each with its degree and its generators.
    pub fn ops(&self, degree: i32) -> impl Iterator<Item = (PackedXi, i32, &[(u16, u32)])> {
        self.ops
            .iter()
            .take_while(move |x| x.1 as i32 <= degree)
            .map(|&(op, d, lo, hi)| (op, d as i32, &self.targets[lo as usize..hi as usize]))
    }

    /// The number of distinct operations.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }
}

/// What the cochains read of a minimal resolution: its generators, and the terms of its
/// differentials filed by the generator they land on.
pub trait Backbone: Sync {
    /// The number of generators of $P_a$ in degree `t`, zero outside the range read.
    fn gens(&self, a: i32, t: i32) -> usize;

    /// The terms of the differentials of $P_a$ that land on generator `i` of degree `t` of
    /// $P_{a-1}$, if there are any.
    ///
    /// # Panics
    ///
    /// If the differentials of $P_a$ are not held.
    fn inv(&self, a: i32, t: i32, i: u32) -> Option<&GenTerms>;
}

/// The terms of the differentials of one $P_a$, filed under the generators `(t, i)` of
/// $P_{a-1}$.
pub type Slice = FxHashMap<(u16, u32), GenTerms>;

/// The generators of a saved minimal resolution, and a way to read the differentials of one
/// filtration at a time straight off the save, keeping only sparse terms.
///
/// The resolution itself is never loaded: its loader holds every differential as a dense vector.
/// Bidegrees strictly between stem 0 and the Adams vanishing line are not read, and neither is
/// stem 0, where $P_a$ has the one generator $x_a$ with $d x_a = Sq^1 x_{a-1}$: the motivic
/// $h_1$-tower needs it far above the filtrations a save holds.
pub struct BackboneSource<'a> {
    store: &'a ZarrSaveStore,
    algebra: Arc<MilnorAlgebra>,
    ops: &'a OpTable,
    /// `gens[a][t]`.
    gens: Vec<Vec<u32>>,
    max_t: Box<dyn Fn(i32) -> i32 + Sync + 'a>,
}

impl<'a> BackboneSource<'a> {
    /// Reads the generators of $P_a$ for `a <= max_a` in degrees up to `max_t(a + 1)`, where
    /// `max_t` must not decrease. Fails, rather than resolve, if the save lacks a bidegree.
    pub fn open(
        store: &'a ZarrSaveStore,
        algebra: &Arc<MilnorAlgebra>,
        ops: &'a OpTable,
        max_a: i32,
        max_t: impl Fn(i32) -> i32 + Sync + 'a,
    ) -> anyhow::Result<Self> {
        use maybe_rayon::prelude::*;
        algebra.compute_basis(max_t(max_a + 1));
        let gens: Vec<anyhow::Result<Vec<u32>>> = (0..=max_a)
            .into_maybe_par_iter()
            .map(|a| {
                (0..=max_t(a + 1))
                    .map(|t| {
                        Ok(if a == 0 {
                            u32::from(t == 0)
                        } else if t < a || vanishes(a, t) {
                            0
                        } else if t == a {
                            1
                        } else {
                            (&blob(store, a, t)?[..]).read_u64::<LittleEndian>()? as u32
                        })
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            store,
            algebra: Arc::clone(algebra),
            ops,
            gens: gens.into_iter().collect::<anyhow::Result<_>>()?,
            max_t: Box::new(max_t),
        })
    }

    /// The largest filtration held.
    pub fn max_a(&self) -> i32 {
        self.gens.len() as i32 - 1
    }

    /// The number of generators of $P_a$ in degree `t`, zero outside the range read.
    pub fn gens(&self, a: i32, t: i32) -> usize {
        if a < 0 || t < 0 {
            return 0;
        }
        self.gens
            .get(a as usize)
            .and_then(|g| g.get(t as usize))
            .map_or(0, |&n| n as usize)
    }

    /// The differentials of $P_a$, read off the save.
    pub fn slice(&self, a: i32) -> anyhow::Result<Slice> {
        let mut by_gen: FxHashMap<(u16, u32), Vec<InvTerm>> = FxHashMap::default();
        if a <= 0 || a > self.max_a() {
            return Ok(Slice::default());
        }
        let max_t = (self.max_t)(a);
        let target = FreeModule::new(Arc::clone(&self.algebra), format!("P{}", a - 1), 0);
        for (t, &n) in self.gens[a as usize - 1].iter().enumerate() {
            target.add_generators(t as i32, n as usize, None);
        }
        target.compute_basis(max_t);
        for t in a..=max_t {
            if self.gens[a as usize][t as usize] == 0 {
                continue;
            }
            if t == a {
                by_gen.entry((a as u16 - 1, 0)).or_default().push(InvTerm {
                    op: 1,
                    t: t as u16,
                    l: 0,
                });
                continue;
            }
            let data = blob(self.store, a, t)?;
            let mut cursor = &data[..];
            let n = cursor.read_u64::<LittleEndian>()? as usize;
            let dim = cursor.read_u64::<LittleEndian>()? as usize;
            anyhow::ensure!(
                dim <= target.dimension(t),
                "({a}, {t}): saved dimension {dim} exceeds {}",
                target.dimension(t)
            );
            for l in 0..n {
                let v = FpVector::from_bytes(TWO, dim, &mut cursor)?;
                for (idx, _) in v.iter_nonzero() {
                    let og = target.index_to_op_gen(t, idx);
                    by_gen
                        .entry((og.generator_degree as u16, og.generator_index as u32))
                        .or_default()
                        .push(InvTerm {
                            op: self.ops.pack(og.operation_degree, og.operation_index),
                            t: t as u16,
                            l: l as u32,
                        });
                }
            }
        }
        Ok(by_gen
            .into_iter()
            .map(|(key, terms)| (key, GenTerms::new(terms, key.0)))
            .collect())
    }

    /// The backbone with the differentials of the given filtrations.
    pub fn window(&self, slices: Vec<(i32, Arc<Slice>)>) -> BackboneWindow<'_> {
        BackboneWindow {
            gens: &self.gens,
            slices: slices.into_iter().collect(),
        }
    }
}

/// $\mathrm{Ext}^{a, t}$ vanishes for $0 < t - a < 2a - 3$.
fn vanishes(a: i32, t: i32) -> bool {
    t - a > 0 && t - a < 2 * a - 3
}

/// The saved differential of a bidegree, or an error rather than a resolution if it is missing.
fn blob(store: &ZarrSaveStore, a: i32, t: i32) -> anyhow::Result<Vec<u8>> {
    store
        .read(SaveKind::NassauDifferential, Bidegree::s_t(a, t))?
        .ok_or_else(|| anyhow::anyhow!("the save lacks ({a}, {t}); refusing to resolve"))
}

/// A [`BackboneSource`] with the differentials of a few filtrations in memory.
pub struct BackboneWindow<'a> {
    gens: &'a [Vec<u32>],
    slices: FxHashMap<i32, Arc<Slice>>,
}

impl Backbone for BackboneWindow<'_> {
    fn gens(&self, a: i32, t: i32) -> usize {
        if a < 0 || t < 0 {
            return 0;
        }
        self.gens
            .get(a as usize)
            .and_then(|g| g.get(t as usize))
            .map_or(0, |&n| n as usize)
    }

    fn inv(&self, a: i32, t: i32, i: u32) -> Option<&GenTerms> {
        self.slices
            .get(&a)
            .unwrap_or_else(|| panic!("the differentials of P_{a} are not held"))
            .get(&(t as u16, i))
    }
}

/// A backbone with every differential in memory, for small ranges.
pub struct StreamedBackbone {
    /// `gens[a][t]`.
    gens: Vec<Vec<u32>>,
    /// `inv[a]` files the terms of the differentials of the generators of $P_a$.
    inv: Vec<Slice>,
    terms: usize,
}

impl StreamedBackbone {
    /// Reads $P_a$ for `a <= max_a` in degrees up to `max_t(a)`; see [`BackboneSource`].
    pub fn read(
        store: &ZarrSaveStore,
        algebra: &Arc<MilnorAlgebra>,
        ops: &OpTable,
        max_a: i32,
        max_t: impl Fn(i32) -> i32 + Sync,
    ) -> anyhow::Result<Self> {
        use maybe_rayon::prelude::*;
        let source = BackboneSource::open(store, algebra, ops, max_a, max_t)?;
        let inv: Vec<anyhow::Result<Slice>> = (0..=max_a)
            .into_maybe_par_iter()
            .map(|a| source.slice(a))
            .collect();
        let inv: Vec<Slice> = inv.into_iter().collect::<anyhow::Result<_>>()?;
        let terms = inv
            .iter()
            .map(|m| m.values().map(|g| g.targets.len()).sum::<usize>())
            .sum();
        Ok(Self {
            gens: source.gens.clone(),
            inv,
            terms,
        })
    }

    /// The number of terms held.
    pub fn terms(&self) -> usize {
        self.terms
    }

    /// Every operation that occurs in a differential, with the largest of `reach(g, t)` over
    /// the generators of $P_g$ of degree `t` it lands on.
    pub fn operations(&self, reach: impl Fn(i32, i32) -> i32) -> FxHashMap<PackedXi, i32> {
        let mut out = FxHashMap::default();
        for (a, m) in self.inv.iter().enumerate() {
            add_operations(&mut out, m, a as i32, &reach);
        }
        out
    }
}

/// Adds the operations of the differentials of $P_a$ to `out`, each with the largest
/// `reach(a - 1, t)` over the generators of degree `t` it lands on.
pub fn add_operations(
    out: &mut FxHashMap<PackedXi, i32>,
    slice: &Slice,
    a: i32,
    reach: &impl Fn(i32, i32) -> i32,
) {
    for (&(t, _), g) in slice {
        let r = reach(a - 1, t as i32);
        for x in &g.ops {
            let slot = out.entry(x.0).or_insert(i32::MIN);
            *slot = (*slot).max(r);
        }
    }
}

impl Backbone for StreamedBackbone {
    fn gens(&self, a: i32, t: i32) -> usize {
        if a < 0 || t < 0 {
            return 0;
        }
        self.gens
            .get(a as usize)
            .and_then(|g| g.get(t as usize))
            .map_or(0, |&n| n as usize)
    }

    fn inv(&self, a: i32, t: i32, i: u32) -> Option<&GenTerms> {
        self.inv.get(a as usize)?.get(&(t as u16, i))
    }
}

/// The action of the operations that occur in a backbone on the monomials of each degree,
/// filed by operation.
///
/// The coboundary of a whole block of cochains (one generator, every monomial of one degree) is
/// read off these by walking the operations that land on the generator, so that building a
/// coboundary costs its number of entries rather than a search per cochain. The coaction on a
/// monomial has far more terms than the backbone has operations, and only the latter are kept,
/// and only in the degrees where a generator they land on carries monomials: the monomials of
/// high degree sit on the generators of low degree, which few operations reach.
pub struct OpTables {
    /// For each degree, the range of `pairs` of each operation that acts.
    index: Vec<FxHashMap<PackedXi, (u32, u32)>>,
    /// For each degree, the nonzero $P(R) \cdot m \ni m'$ as the positions of $m$ and $m'$ in
    /// their degrees, by operation and then by $m$.
    pairs: Vec<Vec<[u16; 2]>>,
    entries: usize,
}

impl OpTables {
    /// The tables through `max_degree` for the operations `needed`, each kept in the degrees up
    /// to its value there: the largest degree of the monomials carried by a generator it lands
    /// on (see [`ScaledExt::reach`]).
    ///
    /// The coaction on each monomial is computed, filtered and dropped; nothing of it is kept
    /// but the table.
    pub fn new(q: &QAlgebra, needed: &FxHashMap<PackedXi, i32>, max_degree: i32) -> Self {
        use maybe_rayon::prelude::*;
        let mut index = Vec::new();
        let mut pairs = Vec::new();
        let mut entries = 0;
        for e in 0..=max_degree {
            let monos = q.monomials(e);
            assert!(monos.len() <= u16::MAX as usize, "degree {e} overflows u16");
            let rows: Vec<Vec<(PackedXi, u16)>> = (0..monos.len())
                .into_maybe_par_iter()
                .map(|idx| {
                    q.coaction_filtered(&monos[idx], |r| {
                        needed.get(&r).is_some_and(|&reach| e <= reach)
                    })
                    .into_iter()
                    .map(|(r, g)| (r, q.index(&g) as u16))
                    .collect()
                })
                .collect();
            let mut all: Vec<(PackedXi, u16, u16)> = rows
                .into_iter()
                .enumerate()
                .flat_map(|(idx, row)| row.into_iter().map(move |(r, g)| (r, idx as u16, g)))
                .collect();
            all.sort_unstable();
            entries += all.len();
            let mut table = FxHashMap::default();
            let mut lo = 0u32;
            for run in all.chunk_by(|x, y| x.0 == y.0) {
                table.insert(run[0].0, (lo, lo + run.len() as u32));
                lo += run.len() as u32;
            }
            index.push(table);
            pairs.push(all.into_iter().map(|x| [x.1, x.2]).collect());
        }
        Self {
            index,
            pairs,
            entries,
        }
    }

    /// The nonzero $P(R) \cdot m \ni m'$ for the monomials $m$ of degree `e`, as positions of
    /// $m$ and $m'$ in their degrees, sorted by $m$.
    pub fn get(&self, e: i32, r: PackedXi) -> &[[u16; 2]] {
        self.index[e as usize].get(&r).map_or(&[], |&(lo, hi)| {
            &self.pairs[e as usize][lo as usize..hi as usize]
        })
    }

    /// The number of entries over every degree and operation.
    pub fn entries(&self) -> usize {
        self.entries
    }
}

/// The operations `ops` (with their reach) and each of them with each of `shifts` taken off,
/// as the cell actions and cup products do, and the identity: what [`OpTables::new`] is to hold.
pub fn needed_operations(
    mut ops: FxHashMap<PackedXi, i32>,
    shifts: &[PackedXi],
) -> FxHashMap<PackedXi, i32> {
    for (op, r) in ops.clone() {
        for &s in shifts {
            if fits(s, op) {
                let slot = ops.entry(op - s).or_insert(i32::MIN);
                *slot = (*slot).max(r);
            }
        }
    }
    ops.insert(0, i32::MAX);
    ops
}

/// A finite complex smashed with $C\tau$, as cells on which the cochains of $C\tau$ are copied.
///
/// A cell shifts the cochains it carries: by `da` in Novikov filtration, `du` in weight and `dk`
/// in polynomial degree, so that a class on it sits where its motivic tridegree says.
#[derive(Debug, Clone)]
pub struct Complex {
    pub name: String,
    /// `(da, du, dk)` of each cell.
    pub cells: Vec<[i32; 3]>,
    /// $(R, c, c')$ for each $P(R) \cdot y_c \ni y_{c'}$ on the dual cells, $R \neq 0$.
    pub actions: Vec<(PackedXi, usize, usize)>,
    /// $(c, c')$ when the coboundary of a cochain on cell `c` contains the same cochain on `c'`:
    /// the cone of multiplication by $q_0$.
    pub attach: Vec<(usize, usize)>,
    /// The largest polynomial degree (Chow degree) computed, if bounded. The [`QAlgebra`] used
    /// must then hold the monomials of length up to this and no longer.
    ///
    /// Each polynomial degree is its own complex, so the bound is exact on what it keeps. A
    /// tower that would run past it shows up as a cocycle that never becomes a coboundary.
    pub max_k: Option<u32>,
    /// `n` when the [`QAlgebra`] used is the quotient by $q_0, \dots, q_{n-1}$, and 0 for all
    /// of $Q$. In a quotient no cochain is shorter than its source, so every class is a tower
    /// of length one.
    pub quotient: usize,
}

impl Complex {
    /// $C\tau$.
    pub fn sphere() -> Self {
        Self {
            name: "Ctau".into(),
            cells: vec![[0, 0, 0]],
            actions: Vec::new(),
            attach: Vec::new(),
            max_k: None,
            quotient: 0,
        }
    }

    /// $C\tau \wedge S/2$, as the cone of $q_0$. Its top cell is in stem 1, which is one Novikov
    /// filtration down and one polynomial degree up.
    pub fn moore() -> Self {
        Self {
            name: "C2".into(),
            cells: vec![[0, 0, 0], [1, 0, 1]],
            actions: Vec::new(),
            attach: vec![(1, 0)],
            max_k: None,
            quotient: 0,
        }
    }

    /// The lowest filtration with cochains: a cell shifted down in filtration carries the unit
    /// of the backbone below filtration 0.
    pub fn min_a(&self) -> i32 {
        -self.cells.iter().map(|c| c[0]).max().unwrap()
    }

    /// $C\tau$ smashed with the cofiber of the map detected by the motivic $h_{j+1}$.
    pub fn hopf(name: &str, j: u32) -> Self {
        Self {
            name: name.into(),
            cells: vec![[0, 0, 0], [0, 1 << j, 0]],
            actions: vec![(1 << j, 1, 0)],
            attach: Vec::new(),
            max_k: None,
            quotient: 0,
        }
    }
}

/// A generator of the backbone on a cell, carrying the monomials of one degree.
#[derive(Debug, Clone, Copy)]
struct Block {
    cell: u8,
    t: u16,
    i: u32,
    /// The degree of the monomials.
    e: u16,
}

/// The cochains of a [`Complex`] at one filtration and weight, numbered by level (the polynomial
/// degree at which a cochain first appears), then by block, then by monomial.
///
/// The level must come first for the reduction to find the towers. Within a level the order is
/// free, and it decides how much the elimination has to add: the coboundary raises the degree
/// of the generator (it is strictly triangular for the algebraic Atiyah–Hirzebruch filtration),
/// and blocks are ordered by decreasing generator degree so that the pivot of a coboundary is
/// its shortest Atiyah–Hirzebruch differential. `MOTIVIC_CTAU.md` has the measurements.
///
/// Positions are computed, not stored: the monomials of one degree and length are consecutive in
/// [`QAlgebra::monomials`].
pub struct Space {
    pub a: i32,
    pub u: i32,
    blocks: Vec<Block>,
    block_id: FxHashMap<(u8, u16, u32), u32>,
    /// The first position of each level, and the dimension.
    level_start: Vec<u32>,
    /// `base[level * blocks.len() + block]` is the first position of a block within a level.
    base: Vec<u32>,
    /// `dk` of each cell.
    dk: Vec<i32>,
}

impl Space {
    /// The cochains of `cx` on `bb` at filtration `a` and weight `u`.
    pub fn new(bb: &dyn Backbone, q: &QAlgebra, cx: &Complex, a: i32, u: i32) -> Self {
        let mut blocks = Vec::new();
        let mut levels = 0;
        for (c, &[da, du, dk]) in cx.cells.iter().enumerate() {
            for t in 0..=(u - du) {
                let e = u - du - t;
                let Some(top) = q.max_len(e) else { continue };
                let n = bb.gens(a + da, t);
                if n > 0 {
                    levels = levels.max((top as i32 + dk + 1) as usize);
                }
                for i in 0..n {
                    blocks.push(Block {
                        cell: c as u8,
                        t: t as u16,
                        i: i as u32,
                        e: e as u16,
                    });
                }
            }
        }
        // Later rows are preferred as pivots, so within a level the generators of lowest degree
        // come last: a coboundary is then cleared at the term of its lowest operation.
        blocks.sort_by_key(|b| (std::cmp::Reverse(b.t), b.cell, b.i));
        let block_id = blocks
            .iter()
            .enumerate()
            .map(|(b, x)| ((x.cell, x.t, x.i), b as u32))
            .collect();
        let dk: Vec<i32> = cx.cells.iter().map(|c| c[2]).collect();
        let mut base = Vec::with_capacity(levels * blocks.len());
        let mut level_start = Vec::with_capacity(levels + 1);
        let mut total = 0u64;
        for level in 0..levels {
            level_start.push(total as u32);
            for b in &blocks {
                base.push(total as u32);
                let len = level as i32 - dk[b.cell as usize];
                if len >= 0 {
                    let e = b.e as i32;
                    total += (q.shorter(e, len as u32 + 1) - q.shorter(e, len as u32)) as u64;
                }
            }
        }
        assert!(
            total < u32::MAX as u64,
            "cochains at ({a}, {u}) overflow u32"
        );
        level_start.push(total as u32);
        Self {
            a,
            u,
            blocks,
            block_id,
            level_start,
            base,
            dk,
        }
    }

    /// The dimension.
    pub fn dim(&self) -> usize {
        *self.level_start.last().unwrap() as usize
    }

    /// The level of a position.
    pub fn level(&self, pos: u32) -> u16 {
        (self.level_start.partition_point(|&s| s <= pos) - 1) as u16
    }

    /// The block of the generator `i` of degree `t` on a cell.
    pub fn block(&self, cell: usize, t: u16, i: u32) -> usize {
        self.block_id[&(cell as u8, t, i)] as usize
    }

    /// The position in block `b` of the monomial of length `len` at `idx` in its degree.
    pub fn pos_in(&self, q: &QAlgebra, b: usize, idx: u32, len: u16) -> u32 {
        let block = self.blocks[b];
        let level = (len as i32 + self.dk[block.cell as usize]) as usize;
        self.base[level * self.blocks.len() + b] + idx
            - q.shorter(block.e as i32, len as u32) as u32
    }

    /// The position of $x \mapsto m$ on a cell, for the generator `i` of degree `t`.
    pub fn pos(&self, q: &QAlgebra, cell: usize, t: u16, i: u32, m: &QMono) -> u32 {
        self.pos_in(
            q,
            self.block(cell, t, i),
            q.index(m) as u32,
            mono_len(m) as u16,
        )
    }

    /// The block of a position and the position of its monomial in its degree.
    pub fn locate(&self, q: &QAlgebra, pos: u32) -> (usize, u32) {
        let level = self.level(pos) as usize;
        let row = &self.base[level * self.blocks.len()..(level + 1) * self.blocks.len()];
        let b = row.partition_point(|&s| s <= pos) - 1;
        let block = self.blocks[b];
        let len = (level as i32 - self.dk[block.cell as usize]) as u32;
        (b, pos - row[b] + q.shorter(block.e as i32, len) as u32)
    }

    /// The cell, generator and monomial at a position, and its level.
    pub fn decode(&self, q: &QAlgebra, pos: u32) -> (usize, u16, u32, QMono, u16) {
        let level = self.level(pos) as usize;
        let row = &self.base[level * self.blocks.len()..(level + 1) * self.blocks.len()];
        let b = row.partition_point(|&s| s <= pos) - 1;
        let block = self.blocks[b];
        let len = (level as i32 - self.dk[block.cell as usize]) as u32;
        let idx = (pos - row[b]) as usize + q.shorter(block.e as i32, len);
        let m = q.monomials(block.e as i32)[idx];
        (block.cell as usize, block.t, block.i, m, level as u16)
    }
}

/// Appends to `out` the entries of the cochain $z \mapsto \sum P(R - S) \cdot c(x)$ over the terms
/// $P(R) x$ of $dz$, for the basis cochain `c` at `pos` of `src`: the coboundary for `shift = 0`
/// (with the attaching terms of `cx`), the cup product with the classical $h_j$ for
/// $S = (2^j)$. A row may repeat, and then cancels.
#[allow(clippy::too_many_arguments)]
pub fn pullback_column(
    q: &QAlgebra,
    bb: &dyn Backbone,
    cx: &Complex,
    src: &Space,
    tgt: &Space,
    pos: u32,
    shift: PackedXi,
    out: &mut Vec<u32>,
) {
    let (cell, t, i, m, _) = src.decode(q, pos);
    let e = mono_degree(&m);
    let shift_degree = xi_degree(shift);
    let da = cx.cells[cell][0];
    // the cell stays, or an action moves it
    let mut moves = vec![(shift, shift_degree, cell)];
    for &(r, c, c2) in &cx.actions {
        if c == cell {
            moves.push((shift + r, shift_degree + xi_degree(r), c2));
        }
    }
    let most = moves.iter().map(|x| x.1).max().unwrap();
    if let Some(terms) = bb.inv(src.a + da + 1, t as i32, i) {
        let psi = q.coaction(&m);
        let mut emit = |c2: usize, gammas: &[(u32, u16)], targets: &[(u16, u32)]| {
            for &(t_l, l) in targets {
                let b = tgt.block(c2, t_l, l);
                for &(idx, len) in gammas {
                    out.push(tgt.pos_in(q, b, idx, len));
                }
            }
        };
        // Join the operations of the differentials with those of the coaction from the
        // smaller side.
        if psi.num_ops() * moves.len() < terms.num_ops() {
            for (r, gammas) in psi.iter() {
                for &(s, _, c2) in &moves {
                    emit(c2, gammas, terms.targets(r + s));
                }
            }
        } else {
            for (op, op_degree, targets) in terms.ops(e + most) {
                for &(s, s_degree, c2) in &moves {
                    if op_degree - s_degree <= e && fits(s, op) {
                        let gammas = psi.get(op - s);
                        if !gammas.is_empty() {
                            emit(c2, gammas, targets);
                        }
                    }
                }
            }
        }
    }
    if shift == 0 {
        for &(c, c2) in &cx.attach {
            if c == cell {
                out.push(tgt.pos(q, c2, t, i, &m));
            }
        }
    }
}

/// The cochain map of [`pullback_column`] on every basis cochain of `src` that `skip` does not
/// reject, as (position, entries of its image), sorted by position.
///
/// A block at a time: for each generator, the operations that land on it are walked once and
/// their action on every monomial of the block is read off `tables`, which must hold `shift`
/// and its sums with the cell actions of `cx`.
#[allow(clippy::too_many_arguments)]
pub fn pullback_columns(
    q: &QAlgebra,
    bb: &dyn Backbone,
    tables: &OpTables,
    cx: &Complex,
    src: &Space,
    tgt: &Space,
    shift: PackedXi,
    skip: impl Fn(u32) -> bool + Sync,
) -> Vec<(u32, Vec<u32>)> {
    use maybe_rayon::prelude::*;
    let shift_degree = xi_degree(shift);
    let built: Vec<Vec<(u32, Vec<u32>)>> = (0..src.blocks.len())
        .into_maybe_par_iter()
        .map(|b| {
            let block = src.blocks[b];
            let (e, cell) = (block.e as i32, block.cell as usize);
            let monos = q.monomials(e);
            // the position of each monomial of the block, unless it is cleared
            let positions: Vec<Option<u32>> = monos
                .iter()
                .enumerate()
                .map(|(idx, m)| {
                    let pos = src.pos_in(q, b, idx as u32, mono_len(m) as u16);
                    (!skip(pos)).then_some(pos)
                })
                .collect();
            let mut local: Vec<Vec<u32>> = vec![Vec::new(); monos.len()];
            let mut moves = vec![(shift, shift_degree, cell)];
            for &(r, c, c2) in &cx.actions {
                if c == cell {
                    moves.push((shift + r, shift_degree + xi_degree(r), c2));
                }
            }
            let most = moves.iter().map(|x| x.1).max().unwrap();
            let da = cx.cells[cell][0];
            if let Some(terms) = bb.inv(src.a + da + 1, block.t as i32, block.i) {
                for (op, op_degree, targets) in terms.ops(e + most) {
                    for &(s, s_degree, c2) in &moves {
                        if op_degree - s_degree > e || !fits(s, op) {
                            continue;
                        }
                        let list = tables.get(e, op - s);
                        if list.is_empty() {
                            continue;
                        }
                        // the degree of the monomials P(R - S) lands in
                        let below = e - (op_degree - s_degree);
                        for &(t_l, l) in targets {
                            let tb = tgt.block(c2, t_l, l);
                            for &[m, g] in list {
                                if positions[m as usize].is_some() {
                                    let len = q.len_at(below, g as u32);
                                    local[m as usize].push(tgt.pos_in(q, tb, g as u32, len));
                                }
                            }
                        }
                    }
                }
            }
            for &(c, c2) in &cx.attach {
                if c == cell && shift == 0 {
                    let tb = tgt.block(c2, block.t, block.i);
                    for (idx, m) in monos.iter().enumerate() {
                        if positions[idx].is_some() {
                            local[idx].push(tgt.pos_in(q, tb, idx as u32, mono_len(m) as u16));
                        }
                    }
                }
            }
            positions
                .into_iter()
                .zip(local)
                .filter_map(|(pos, column)| Some((pos?, column)))
                .collect()
        })
        .collect();
    let mut columns: Vec<(u32, Vec<u32>)> = built.into_iter().flatten().collect();
    columns.sort_unstable_by_key(|x| x.0);
    columns
}

/// Marks a column that generates no tower.
const NO_TOWER: u32 = u32::MAX;

/// The level at which an essential class dies: never.
pub const NEVER: u16 = u16::MAX;

/// The cocycles of a complex at one filtration and weight, as an echelon form whose columns are
/// each born (as a cocycle) at one level and die (become a coboundary) at a later one.
///
/// A column that dies strictly after it is born generates a tower: the classes $q_0^i g$ for
/// $i$ below the difference. In the lowest filtration nothing is a coboundary, the columns are
/// the cocycles themselves and every tower is infinite.
pub struct Reduced {
    pub echelon: SparseEchelon,
    birth: Vec<u16>,
    death: Vec<u16>,
    /// The tower each column generates, or [`NO_TOWER`].
    tower_of: Vec<u32>,
    /// The column generating each tower.
    pub towers: Vec<u32>,
}

impl Reduced {
    /// Nothing, on `rows` cochains.
    fn new(rows: usize) -> Self {
        Self {
            echelon: SparseEchelon::new(rows),
            birth: Vec::new(),
            death: Vec::new(),
            tower_of: Vec::new(),
            towers: Vec::new(),
        }
    }

    /// Adds a reduced cocycle born at `birth` that dies at `death`.
    pub(crate) fn push(&mut self, column: Vec<u32>, birth: u16, death: u16) {
        let id = self.echelon.insert(column);
        debug_assert_eq!(id, self.birth.len());
        self.birth.push(birth);
        self.death.push(death);
        if death > birth {
            self.tower_of.push(self.towers.len() as u32);
            self.towers.push(id as u32);
        } else {
            self.tower_of.push(NO_TOWER);
        }
    }

    /// The cocycles on `rows` cochains with the given reduced columns, each with the levels at
    /// which it is born and dies, in the order they were found.
    pub fn from_columns(
        rows: usize,
        columns: impl IntoIterator<Item = (Vec<u32>, u16, u16)>,
    ) -> Self {
        let mut this = Self::new(rows);
        for (column, birth, death) in columns {
            this.push(column, birth, death);
        }
        this
    }

    /// Each reduced column with the levels at which it is born and dies.
    pub fn columns(&self) -> impl Iterator<Item = (&[u32], u16, u16)> {
        (0..self.birth.len()).map(|c| (self.echelon.column(c), self.birth[c], self.death[c]))
    }

    /// The level at which tower `i` is born, and its length ([`NEVER`] if infinite).
    pub fn tower(&self, i: usize) -> (u16, u16) {
        let c = self.towers[i] as usize;
        let length = if self.death[c] == NEVER {
            NEVER
        } else {
            self.death[c] - self.birth[c]
        };
        (self.birth[c], length)
    }

    /// The cocycle generating tower `i`.
    pub fn generator(&self, i: usize) -> &[u32] {
        self.echelon.column(self.towers[i] as usize)
    }

    /// The class at `level` of the cocycle with the given entries, as the towers whose class at
    /// that level it is the sum of.
    ///
    /// # Panics
    ///
    /// If the entries are not a cocycle.
    pub fn class(
        &self,
        ws: &mut SparseWorkspace,
        entries: impl IntoIterator<Item = u32>,
        level: u16,
    ) -> Vec<u32> {
        let mut used = Vec::new();
        let rest = self.echelon.reduce(ws, entries, Some(&mut used));
        assert!(
            rest.is_empty(),
            "not a cocycle: {} entries left",
            rest.len()
        );
        let mut out: Vec<u32> = used
            .into_iter()
            .filter(|&c| self.death[c as usize] > level)
            .map(|c| self.tower_of[c as usize])
            .collect();
        debug_assert!(out.iter().all(|&t| t != NO_TOWER));
        out.sort_unstable();
        out
    }
}

/// What a run of [`reduce_weight`] did.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReduceStats {
    /// The cochains over the filtrations reduced.
    pub cochains: usize,
    /// The entries of the coboundary columns built.
    pub input_entries: usize,
    /// The entries of the reduced columns kept.
    pub reduced_entries: usize,
    /// The longest reduced column.
    pub longest: usize,
    /// The largest cochain space.
    pub largest: usize,
    /// The entries of the columns added during reduction.
    pub additions: usize,
}

impl ReduceStats {
    /// The totals of two runs.
    pub fn merge(self, o: Self) -> Self {
        Self {
            cochains: self.cochains + o.cochains,
            input_entries: self.input_entries + o.input_entries,
            reduced_entries: self.reduced_entries + o.reduced_entries,
            longest: self.longest.max(o.longest),
            largest: self.largest.max(o.largest),
            additions: self.additions + o.additions,
        }
    }
}

/// The sources reduced against a frozen echelon at a time; see [`reduce_step`].
const REDUCE_BATCH: usize = 4096;

/// The sources one thread takes of a batch.
const REDUCE_PIECE: usize = 64;

/// Reduces the coboundary from `src` to `tgt`, skipping the sources `cleared` accepts, and
/// returns the cocycles of `tgt`, and with `track` the cocycles of `src` that are no
/// coboundary of anything: the sources whose coboundary reduces to zero, each as the
/// combination of sources that has zero coboundary.
///
/// Sources are taken in order of level and rows are numbered by level, so a reduced column is a
/// cocycle born at the level of its pivot that becomes a coboundary at the level of its source.
///
/// A batch of sources is first reduced in parallel against the columns found before the batch,
/// which does nearly all the additions, and then in order against the columns of the batch
/// itself. Larger batches leave more to the second, sequential, step.
#[allow(clippy::too_many_arguments)]
pub fn reduce_step(
    q: &QAlgebra,
    bb: &dyn Backbone,
    tables: &OpTables,
    cx: &Complex,
    src: &Space,
    tgt: &Space,
    cleared: impl Fn(u32) -> bool + Sync,
    track: bool,
    stats: &mut ReduceStats,
) -> (Reduced, Option<Reduced>) {
    use maybe_rayon::prelude::*;
    let mut red = Reduced::new(tgt.dim());
    // with `track`, the combination of sources each column is the coboundary of
    let mut combos: Vec<Vec<u32>> = Vec::new();
    let mut essential = track.then(|| Reduced::new(src.dim()));
    stats.cochains += src.dim();
    stats.largest = stats.largest.max(src.dim());
    // The columns are built a block at a time, in parallel.
    let columns = pullback_columns(q, bb, tables, cx, src, tgt, 0, cleared);
    let mut ws = SparseWorkspace::new(tgt.dim());
    for batch in columns.chunks(REDUCE_BATCH) {
        let echelon = &red.echelon;
        let pieces: Vec<&[(u32, Vec<u32>)]> = batch.chunks(REDUCE_PIECE).collect();
        let first: Vec<Vec<(Vec<u32>, Vec<u32>, usize)>> = pieces
            .into_maybe_par_iter()
            .map(|piece| {
                let mut ws = SparseWorkspace::new(echelon.rows());
                piece
                    .iter()
                    .map(|(_, column)| {
                        let mut used = Vec::new();
                        let rest = echelon.reduce(
                            &mut ws,
                            column.iter().copied(),
                            track.then_some(&mut used),
                        );
                        (rest, used, ws.take_additions())
                    })
                    .collect()
            })
            .collect();
        for ((pos, column), (rest, mut used, added)) in
            batch.iter().zip(first.into_iter().flatten())
        {
            let level = src.level(*pos);
            stats.input_entries += column.len();
            let rest = if rest.is_empty() {
                rest
            } else {
                red.echelon
                    .reduce(&mut ws, rest, track.then_some(&mut used))
            };
            stats.additions += added + ws.take_additions();
            let combo = track.then(|| {
                let mut set = std::collections::BTreeSet::from([*pos]);
                for &c in &used {
                    for &r in &combos[c as usize] {
                        if !set.remove(&r) {
                            set.insert(r);
                        }
                    }
                }
                set.into_iter().collect::<Vec<u32>>()
            });
            if rest.is_empty() {
                if let (Some(ess), Some(combo)) = (essential.as_mut(), combo) {
                    ess.push(combo, level, NEVER);
                }
                continue;
            }
            stats.reduced_entries += rest.len();
            stats.longest = stats.longest.max(rest.len());
            let birth = tgt.level(*rest.last().unwrap());
            red.push(rest, birth, level);
            if let Some(combo) = combo {
                combos.push(combo);
            }
        }
    }
    (red, essential)
}

/// Reduces the coboundaries of `cx` at weight `u` out of filtrations `a_lo..a_hi`, and returns
/// the cocycles of filtrations `a_lo + 1..=a_hi`, and of `a_lo` if that is the lowest filtration
/// of `cx`.
///
/// Sources are taken in order of level and rows are numbered by level, so a reduced column is a
/// cocycle born at the level of its pivot that becomes a coboundary at the level of its source.
/// A cochain that is the pivot of a coboundary into its own filtration is skipped: its
/// coboundary reduces to zero.
pub fn reduce_weight(
    q: &QAlgebra,
    bb: &dyn Backbone,
    tables: &OpTables,
    cx: &Complex,
    u: i32,
    a_lo: i32,
    a_hi: i32,
) -> (Vec<((i32, i32), Reduced)>, ReduceStats) {
    let mut out: Vec<((i32, i32), Reduced)> = Vec::new();
    let mut stats = ReduceStats::default();
    let mut src = Space::new(bb, q, cx, a_lo, u);
    for a in a_lo..a_hi {
        let tgt = Space::new(bb, q, cx, a + 1, u);
        let cleared = out
            .last()
            .filter(|(key, _)| key.0 == a)
            .map(|(_, r)| &r.echelon);
        let (red, essential) = reduce_step(
            q,
            bb,
            tables,
            cx,
            &src,
            &tgt,
            |pos| cleared.is_some_and(|e| e.pivot_column(pos as usize).is_some()),
            a == cx.min_a(),
            &mut stats,
        );
        if let Some(ess) = essential {
            out.push(((a, u), ess));
        }
        out.push(((a + 1, u), red));
        src = tgt;
    }
    (out, stats)
}

/// $\mathrm{Ext}_{A/\tau}$ of a complex through a stem, as the towers of each filtration and
/// weight under multiplication by $h_0 = q_0$.
pub struct ScaledExt {
    pub cx: Complex,
    pub max_n: i32,
    reduced: FxHashMap<(i32, i32), Reduced>,
    /// The id of the first tower of each filtration and weight, in the order of
    /// [`Self::keys`].
    first: FxHashMap<(i32, i32), u32>,
    /// The cochain spaces asked for so far.
    spaces: dashmap::DashMap<(i32, i32), Arc<Space>>,
    keys: Vec<(i32, i32)>,
    pub stats: ReduceStats,
}

impl ScaledExt {
    /// The largest filtration and the largest degree at each filtration that a backbone must be
    /// read through for stems up to `max_n` of any [`Complex`] here.
    pub fn backbone_range(max_n: i32) -> (i32, impl Fn(i32) -> i32 + Sync) {
        (max_n + 2, move |a| ((max_n + a) / 2 + 2).min(max_n + 2))
    }

    /// Computes the towers of `cx` in stems `0..=max_n`.
    pub fn new(
        q: &QAlgebra,
        bb: &dyn Backbone,
        tables: &OpTables,
        cx: Complex,
        max_n: i32,
    ) -> Self {
        use maybe_rayon::prelude::*;
        // heaviest weights first, so that they do not all end together
        let mut weights: Vec<i32> = (0..=max_n).collect();
        weights.sort_by_key(|&u| (2 * u - max_n).abs());
        let results: Vec<_> = weights
            .into_maybe_par_iter()
            .map(|u| {
                // stem 2u - a <= max_n, and the coboundary into the lowest such filtration
                let a_lo = (2 * u - max_n - 1).max(cx.min_a());
                reduce_weight(q, bb, tables, &cx, u, a_lo, u.max(a_lo + 1))
            })
            .collect();
        let mut reduced = FxHashMap::default();
        let mut stats = ReduceStats::default();
        for (list, s) in results {
            stats = stats.merge(s);
            for (key, red) in list {
                if 2 * key.1 - key.0 <= max_n && 2 * key.1 - key.0 >= 0 {
                    reduced.insert(key, red);
                }
            }
        }
        let mut keys: Vec<(i32, i32)> = reduced.keys().copied().collect();
        keys.sort_by_key(|&(a, u)| (2 * u - a, a));
        let mut first = FxHashMap::default();
        let mut total = 0u32;
        for &key in &keys {
            first.insert(key, total);
            total += reduced[&key].towers.len() as u32;
        }
        Self {
            cx,
            max_n,
            reduced,
            first,
            spaces: Default::default(),
            keys,
            stats,
        }
    }

    /// The cochains at filtration `a` and weight `u`, built on first use and kept.
    pub fn space(&self, q: &QAlgebra, bb: &dyn Backbone, a: i32, u: i32) -> Arc<Space> {
        if let Some(hit) = self.spaces.get(&(a, u)) {
            return Arc::clone(&hit);
        }
        let space = Arc::new(Space::new(bb, q, &self.cx, a, u));
        self.spaces.insert((a, u), Arc::clone(&space));
        space
    }

    /// Drops the spaces [`Self::space`] keeps.
    pub fn clear_spaces(&self) {
        self.spaces.clear();
    }

    /// The tower generator of id `id`: its filtration, weight and index there.
    pub fn locate(&self, id: u32) -> (i32, i32, usize) {
        let k = self.keys.partition_point(|key| self.first[key] <= id) - 1;
        let (a, u) = self.keys[k];
        (a, u, (id - self.first[&(a, u)]) as usize)
    }

    /// The largest degree of the monomials that a generator of $P_g$ of degree `t` carries in
    /// any cochain space read for stems up to `max_n`, for [`OpTables::new`].
    pub fn reach(max_n: i32) -> impl Fn(i32, i32) -> i32 {
        // a space at filtration a and weight u is read when 2u - a <= max_n + 1, and the
        // generator sits in filtration a or, on a cell shifted down, a + 1
        move |g, t| (max_n + 1 + g) / 2 - t
    }

    /// The filtrations and weights held, by stem and then filtration.
    pub fn keys(&self) -> &[(i32, i32)] {
        &self.keys
    }

    /// The cocycles at filtration `a` and weight `u`, if in range.
    pub fn reduced(&self, a: i32, u: i32) -> Option<&Reduced> {
        self.reduced.get(&(a, u))
    }

    /// The id of tower `i` at filtration `a` and weight `u`.
    pub fn tower_id(&self, a: i32, u: i32, i: u32) -> u32 {
        self.first[&(a, u)] + i
    }

    /// The number of towers.
    pub fn num_towers(&self) -> usize {
        self.reduced.values().map(|r| r.towers.len()).sum()
    }
}

/// The image of each tower generator of `src` at filtration `a` and weight `u` under a cochain
/// map into `tgt` at `(a2, u2)` that shifts the level by `dk`, as tower ids of `tgt`. `map`
/// appends the entries of the image of a cochain.
///
/// The map commutes with $q_0$, so this determines it on every class. `None` if the target is
/// out of range.
#[allow(clippy::too_many_arguments)]
pub fn induced(
    src: &ScaledExt,
    tgt: &ScaledExt,
    (a, u): (i32, i32),
    (a2, u2): (i32, i32),
    dk: i32,
    ws: &mut SparseWorkspace,
    mut map: impl FnMut(&[u32], &mut Vec<u32>),
) -> Option<Vec<Vec<u32>>> {
    let from = src.reduced(a, u)?;
    let to = tgt.reduced(a2, u2)?;
    let mut entries = Vec::new();
    Some(
        (0..from.towers.len())
            .map(|i| {
                entries.clear();
                map(from.generator(i), &mut entries);
                let level = from.tower(i).0 as i32 + dk;
                if level < 0 {
                    assert!(
                        to.echelon
                            .reduce(ws, entries.iter().copied(), None)
                            .is_empty()
                    );
                    return Vec::new();
                }
                to.class(ws, entries.iter().copied(), level as u16)
                    .into_iter()
                    .map(|t| tgt.tower_id(a2, u2, t))
                    .collect()
            })
            .collect(),
    )
}

/// The cocycle representing $\beta \cdot [c]$ for the cocycle `c` of $C\tau$ in `src`, in `tgt`,
/// which must be `src` shifted by the degree of `gamma`; see [`BarCocycle`] for the formula. A
/// row may repeat, and then cancels.
pub fn bar_cup(
    q: &QAlgebra,
    bb: &dyn Backbone,
    gamma: &BarCocycle,
    src: &Space,
    tgt: &Space,
    c: &[u32],
) -> Vec<u32> {
    type Lift = FxHashMap<Vec<PackedXi>, FxHashSet<QMono>>;
    let mut lifts: FxHashMap<(u16, u32), Lift> = FxHashMap::default();
    for &pos in c {
        let (_, t, i, m, _) = src.decode(q, pos);
        lifts
            .entry((t, i))
            .or_default()
            .entry(Vec::new())
            .or_default()
            .insert(m);
    }
    for j in 1..=gamma.degree.a {
        let mut next: FxHashMap<(u16, u32), Lift> = FxHashMap::default();
        for (&(t, i), lift) in &lifts {
            let Some(terms) = bb.inv(src.a + j, t as i32, i) else {
                continue;
            };
            for (w, f) in lift {
                let Some(letters) = gamma.extensions.get(w) else {
                    continue;
                };
                let f: Vec<QMono> = f.iter().copied().collect();
                for (op, _, targets) in terms.ops(tgt.u - t as i32) {
                    for &r in letters {
                        if !fits(r, op) {
                            continue;
                        }
                        let image = q.act_poly(op - r, &f);
                        if image.is_empty() {
                            continue;
                        }
                        let mut w2 = Vec::with_capacity(w.len() + 1);
                        w2.push(r);
                        w2.extend_from_slice(w);
                        for &target in targets {
                            let slot = next
                                .entry(target)
                                .or_default()
                                .entry(w2.clone())
                                .or_default();
                            for m in &image {
                                if !slot.remove(m) {
                                    slot.insert(*m);
                                }
                            }
                        }
                    }
                }
            }
        }
        lifts = next;
    }
    let mut out = Vec::new();
    for ((t, i), lift) in lifts {
        for (w, f) in lift {
            let Some(value) = gamma.values.get(&w) else {
                continue;
            };
            for x in value {
                for y in &f {
                    out.push(tgt.pos(q, 0, t, i, &crate::ctau::mono_mul(x, y)));
                }
            }
        }
    }
    out
}
