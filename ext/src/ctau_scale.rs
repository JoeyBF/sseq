//! The $C\tau$ Adams $E_2$ page at scale: one sparse reduction per filtration and weight.

use std::sync::Arc;

use algebra::{Algebra, MilnorAlgebra, module::FreeModule, module::Module};
use byteorder::{LittleEndian, ReadBytesExt};
use fp::{
    matrix::{SparseEchelon, SparseWorkspace},
    prime::TWO,
    vector::FpVector,
};
use rustc_hash::FxHashMap;
use sseq::coordinates::Bidegree;

use crate::{
    ctau::{OpTable, PackedXi, QAlgebra, QMono, fits, mono_degree, mono_len, xi_degree},
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

/// The generators and differentials of a saved minimal resolution, read straight off the save
/// one bidegree at a time and kept only as sparse terms, grouped by the generator they land on.
///
/// The resolution itself is never loaded: its loader holds every differential as a dense vector.
pub struct StreamedBackbone {
    /// `gens[a][t]`.
    gens: Vec<Vec<u32>>,
    /// `inv[a]` files the terms of the differentials of the generators of $P_a$ under the
    /// generators `(t, i)` of $P_{a-1}$, each list sorted by the degree of $z_l$.
    inv: Vec<FxHashMap<(u16, u32), Vec<InvTerm>>>,
    terms: usize,
}

impl StreamedBackbone {
    /// Reads $P_a$ for `a <= max_a` in degrees up to `max_t(a)`, which must not decrease with
    /// `a`. Fails, rather than resolve, if the save lacks a bidegree.
    ///
    /// Bidegrees strictly between stem 0 and the Adams vanishing line are not read, and neither
    /// is stem 0, where $P_a$ has the one generator $x_a$ with $d x_a = Sq^1 x_{a-1}$: the
    /// motivic $h_1$-tower needs it far above the filtrations a save holds.
    pub fn read(
        store: &ZarrSaveStore,
        algebra: &Arc<MilnorAlgebra>,
        ops: &OpTable,
        max_a: i32,
        max_t: impl Fn(i32) -> i32 + Sync,
    ) -> anyhow::Result<Self> {
        use maybe_rayon::prelude::*;
        algebra.compute_basis(max_t(max_a + 1));
        // Ext^{a, t} vanishes for 0 < t - a < 2a - 3.
        let vanishes = |a: i32, t: i32| t - a > 0 && t - a < 2 * a - 3;
        let blob = |a: i32, t: i32| -> anyhow::Result<Vec<u8>> {
            store
                .read(SaveKind::NassauDifferential, Bidegree::s_t(a, t))?
                .ok_or_else(|| anyhow::anyhow!("the save lacks ({a}, {t}); refusing to resolve"))
        };
        // P_a is the target of the differentials of P_{a+1}, which index its basis through the
        // degrees they are read in.
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
                            (&blob(a, t)?[..]).read_u64::<LittleEndian>()? as u32
                        })
                    })
                    .collect()
            })
            .collect();
        let gens: Vec<Vec<u32>> = gens.into_iter().collect::<anyhow::Result<_>>()?;
        let inv: Vec<anyhow::Result<FxHashMap<(u16, u32), Vec<InvTerm>>>> = (0..=max_a)
            .into_maybe_par_iter()
            .map(|a| {
                let mut by_gen: FxHashMap<(u16, u32), Vec<InvTerm>> = FxHashMap::default();
                if a == 0 {
                    return Ok(by_gen);
                }
                let target = FreeModule::new(Arc::clone(algebra), format!("P{}", a - 1), 0);
                for (t, &n) in gens[a as usize - 1].iter().enumerate() {
                    target.add_generators(t as i32, n as usize, None);
                }
                target.compute_basis(max_t(a));
                for t in a..=max_t(a) {
                    if gens[a as usize][t as usize] == 0 {
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
                    let data = blob(a, t)?;
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
                                    op: ops.pack(og.operation_degree, og.operation_index),
                                    t: t as u16,
                                    l: l as u32,
                                });
                        }
                    }
                }
                Ok(by_gen)
            })
            .collect();
        let inv: Vec<_> = inv.into_iter().collect::<anyhow::Result<_>>()?;
        let terms = inv
            .iter()
            .map(|m| m.values().map(Vec::len).sum::<usize>())
            .sum();
        Ok(Self { gens, inv, terms })
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

    /// The terms of the differentials of $P_a$ that land on generator `i` of degree `t` of
    /// $P_{a-1}$, sorted by the degree of their source.
    pub fn inv(&self, a: i32, t: i32, i: u32) -> &[InvTerm] {
        self.inv
            .get(a as usize)
            .and_then(|m| m.get(&(t as u16, i)))
            .map_or(&[], |v| v)
    }

    /// The number of terms held.
    pub fn terms(&self) -> usize {
        self.terms
    }
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
}

impl Complex {
    /// $C\tau$.
    pub fn sphere() -> Self {
        Self {
            name: "Ctau".into(),
            cells: vec![[0, 0, 0]],
            actions: Vec::new(),
            attach: Vec::new(),
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
    pub fn new(bb: &StreamedBackbone, q: &QAlgebra, cx: &Complex, a: i32, u: i32) -> Self {
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

    /// The position of $x \mapsto m$ on a cell, for the generator `i` of degree `t`.
    pub fn pos(&self, q: &QAlgebra, cell: usize, t: u16, i: u32, m: &QMono) -> u32 {
        let b = self.block_id[&(cell as u8, t, i)] as usize;
        let len = mono_len(m);
        let level = (len as i32 + self.dk[cell]) as usize;
        let e = self.blocks[b].e as i32;
        self.base[level * self.blocks.len() + b] + (q.index(m) - q.shorter(e, len)) as u32
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
    bb: &StreamedBackbone,
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
    for term in bb.inv(src.a + da + 1, t as i32, i) {
        let op_degree = (term.t - t) as i32;
        if op_degree - most > e {
            break;
        }
        for &(s, s_degree, c2) in &moves {
            if op_degree - s_degree > e || !fits(s, term.op) {
                continue;
            }
            for gamma in q.act(term.op - s, &m).iter() {
                out.push(tgt.pos(q, c2, term.t, term.l, gamma));
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
    fn push(&mut self, column: Vec<u32>, birth: u16, death: u16) {
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
        }
    }
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
    bb: &StreamedBackbone,
    cx: &Complex,
    u: i32,
    a_lo: i32,
    a_hi: i32,
) -> (Vec<((i32, i32), Reduced)>, ReduceStats) {
    let mut out: Vec<((i32, i32), Reduced)> = Vec::new();
    let mut stats = ReduceStats::default();
    let mut ws = SparseWorkspace::new(0);
    let mut src = Space::new(bb, q, cx, a_lo, u);
    for a in a_lo..a_hi {
        let tgt = Space::new(bb, q, cx, a + 1, u);
        let mut red = Reduced::new(tgt.dim());
        // in the lowest filtration, the combination of sources each column is the coboundary of
        let mut combos: Vec<Vec<u32>> = Vec::new();
        let mut essential = (a == cx.min_a()).then(|| Reduced::new(src.dim()));
        let cleared = out
            .last()
            .filter(|(key, _)| key.0 == a)
            .map(|(_, r)| &r.echelon);
        let mut column = Vec::new();
        let mut used = Vec::new();
        stats.cochains += src.dim();
        stats.largest = stats.largest.max(src.dim());
        for level in 0..src.level_start.len().saturating_sub(1) {
            for pos in src.level_start[level]..src.level_start[level + 1] {
                if cleared.is_some_and(|e| e.pivot_column(pos as usize).is_some()) {
                    continue;
                }
                column.clear();
                pullback_column(q, bb, cx, &src, &tgt, pos, 0, &mut column);
                stats.input_entries += column.len();
                used.clear();
                let track = essential.is_some().then_some(&mut used);
                let rest = red.echelon.reduce(&mut ws, column.iter().copied(), track);
                let combo = essential.is_some().then(|| {
                    let mut set = std::collections::BTreeSet::from([pos]);
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
                        ess.push(combo, level as u16, NEVER);
                    }
                    continue;
                }
                stats.reduced_entries += rest.len();
                stats.longest = stats.longest.max(rest.len());
                let birth = tgt.level(*rest.last().unwrap());
                red.push(rest, birth, level as u16);
                if let Some(combo) = combo {
                    combos.push(combo);
                }
            }
        }
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
    pub fn new(q: &QAlgebra, bb: &StreamedBackbone, cx: Complex, max_n: i32) -> Self {
        use maybe_rayon::prelude::*;
        // heaviest weights first, so that they do not all end together
        let mut weights: Vec<i32> = (0..=max_n).collect();
        weights.sort_by_key(|&u| (2 * u - max_n).abs());
        let results: Vec<_> = weights
            .into_maybe_par_iter()
            .map(|u| {
                // stem 2u - a <= max_n, and the coboundary into the lowest such filtration
                let a_lo = (2 * u - max_n - 1).max(cx.min_a());
                reduce_weight(q, bb, &cx, u, a_lo, u.max(a_lo + 1))
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
            keys,
            stats,
        }
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
/// appends the entries of the image of a basis cochain.
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
    mut map: impl FnMut(u32, &mut Vec<u32>),
) -> Option<Vec<Vec<u32>>> {
    let from = src.reduced(a, u)?;
    let to = tgt.reduced(a2, u2)?;
    let mut entries = Vec::new();
    Some(
        (0..from.towers.len())
            .map(|i| {
                entries.clear();
                for &pos in from.generator(i) {
                    map(pos, &mut entries);
                }
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
