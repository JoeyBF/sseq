//! The tower export of the $C\tau$ Adams $E_2$ page from the transposed coboundaries, taken from the top filtration down.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

use fp::matrix::{SparseEchelon, SparseWorkspace};
use rustc_hash::FxHashMap;

use crate::{
    ctau::{PackedXi, QAlgebra},
    ctau_scale::{
        BackboneSource, Complex, Forward, NEVER, OpTables, Phase, Space, pullback_rows, timed,
    },
};

/// A functional on the cochains of one filtration and weight, found by the transposed
/// reduction of the coboundary into them: the coordinate of the cochain at `pos` plus those of
/// some later cochains.
///
/// Composed with the coboundary it vanishes on the cochains of level below `death`, so between
/// `birth` (the level of `pos`) and `death` it is a functional on the cohomology of that
/// polynomial degree. With `birth < death` it is a tower, and the towers alive at a level are
/// a basis of the dual of the cohomology there.
struct Functional {
    pos: u32,
    /// The positions of the coordinates summed, sorted; the first is `pos`.
    psi: Vec<u32>,
    birth: u16,
    death: u16,
}

/// The functionals on the cochains at one filtration and weight, one for each cochain that is
/// not the first source of a reduced row out of them, in the order found (from the last
/// position to the first).
#[derive(Default)]
struct Functionals {
    list: Vec<Functional>,
    /// The members of `list` that are towers.
    towers: Vec<u32>,
}

/// A tower of an exported complex, as a functional: see [`Functional`].
pub struct Tower {
    psi: Vec<u32>,
    death: u16,
}

/// A complex that has been exported, as what a later export reads of it: its towers by
/// filtration and weight.
pub struct Exported {
    pub cx: Complex,
    towers: FxHashMap<(i32, i32), Arc<Vec<Tower>>>,
}

/// The maps an export computes.
pub enum Maps<'a> {
    /// The towers and $h_1, \dots, h_4$.
    Products,
    /// Those, and the top cell of a two-cell complex into the sphere, already exported.
    WithTopCell(&'a Exported),
    /// Only the bottom cell, out of the sphere into a two-cell complex already exported: no
    /// tower of the sphere is written. A map within a filtration needs the rows out of its
    /// source and the towers of its target, which the two exports find a step apart, so the
    /// sphere is reduced again for this one.
    BottomCell(&'a Exported),
}

/// The index of the top cell map among the maps of [`Pulled::map`].
const TOP_CELL: u8 = 4;

/// The index of the bottom cell map.
const BOTTOM_CELL: u8 = 5;

/// The image of a tower functional under the transpose of a cochain map, reduced by the rows
/// out of its source space: what is left for the functionals of that space to account for.
struct Pulled {
    /// The map: $h_{j+1}$ for `j` below 4, [`TOP_CELL`] or [`BOTTOM_CELL`].
    map: u8,
    /// The filtration, weight and index of the tower pulled back.
    target: (i32, i32, u32),
    death: u16,
    /// Positions of the source space below `cut`, sorted.
    entries: Vec<u32>,
    /// One past the last position of level below `death`: where the functional stops being
    /// one on cohomology, and nothing later is computed.
    cut: u32,
}

/// What an export did.
#[derive(Debug, Default, Clone, Copy)]
pub struct DualStats {
    /// The cochains over the spaces reduced into.
    pub cochains: usize,
    /// The largest of those spaces.
    pub largest: usize,
    /// The rows reduced, and those of them that needed no addition.
    pub rows: usize,
    pub apparent: usize,
    /// The entries built for the coboundaries and for the maps, before those that repeat
    /// cancel: the work of reading them off the tables.
    pub built_entries: usize,
    pub map_built_entries: usize,
    /// The entries of the rows reduced, of the rows added during reduction, and of the reduced
    /// rows.
    pub input_entries: usize,
    pub additions: usize,
    pub reduced_entries: usize,
    pub towers: usize,
    /// The entries of the tower functionals pulled back by the $h_j$, and of the rows added
    /// to reduce them.
    pub map_entries: usize,
    pub map_additions: usize,
    /// The nonzero coefficients of the $h_j$.
    pub products: usize,
}

impl DualStats {
    /// The totals of two runs.
    pub fn merge(self, o: Self) -> Self {
        Self {
            cochains: self.cochains + o.cochains,
            largest: self.largest.max(o.largest),
            rows: self.rows + o.rows,
            apparent: self.apparent + o.apparent,
            built_entries: self.built_entries + o.built_entries,
            map_built_entries: self.map_built_entries + o.map_built_entries,
            input_entries: self.input_entries + o.input_entries,
            additions: self.additions + o.additions,
            reduced_entries: self.reduced_entries + o.reduced_entries,
            towers: self.towers + o.towers,
            map_entries: self.map_entries + o.map_entries,
            map_additions: self.map_additions + o.map_additions,
            products: self.products + o.products,
        }
    }
}

/// How an export bounds its memory.
#[derive(Debug, Clone)]
pub struct DualOptions {
    /// The weights of one filtration reduced at the same time: the peak is that many of the
    /// largest coboundaries, so lowering it trades time for memory.
    pub in_flight: usize,
}

/// The id of tower `i` at filtration `a` and weight `u`.
fn tower_id(a: i32, u: i32, i: usize) -> String {
    format!("{a}_{u}_{i}")
}

/// Everything a step reads that is fixed while it runs.
struct Step<'a> {
    q: &'a QAlgebra,
    tables: &'a OpTables,
    cx: &'a Complex,
    max_n: i32,
    maps: &'a Maps<'a>,
    /// The source filtration of the coboundary reduced.
    a: i32,
    /// The differentials of the backbone out of filtration `a + 1`.
    differentials: &'a Forward,
    bb: &'a dyn crate::ctau_scale::Backbone,
}

/// What reducing one coboundary leaves.
struct Reduction {
    u: i32,
    /// The reduced rows, as functionals on the source space with positions counted from its
    /// end.
    rows: SparseEchelon,
    /// The towers one filtration up.
    towers: Arc<Vec<Tower>>,
    /// The positions of the source space that are pivots, as a bit set.
    pivots: Vec<u64>,
    towers_csv: String,
    h_csv: String,
    cell_csv: String,
    stats: DualStats,
}

impl Step<'_> {
    /// Whether the classes at filtration `a` and weight `u` are in the export.
    fn in_range(&self, a: i32, u: i32) -> bool {
        (0..=self.max_n).contains(&(2 * u - a))
    }

    /// Reduces the transposed coboundary into filtration `a + 1` at weight `u`, skipping the
    /// rows `cleared` marks (the pivots of the coboundary out of that space), and finishes the
    /// products `pulled` into that space with the functionals found.
    fn reduce(
        &self,
        u: i32,
        cleared: Option<&Vec<u64>>,
        pulled: Option<&Vec<Pulled>>,
    ) -> anyhow::Result<Reduction> {
        use std::fmt::Write;
        let a = self.a;
        let src = Space::new(self.bb, self.q, self.cx, a, u);
        let tgt = Space::new(self.bb, self.q, self.cx, a + 1, u);
        let is_cleared =
            |pos: u32| cleared.is_some_and(|bits| bits[pos as usize / 64] >> (pos % 64) & 1 == 1);
        let (mut built, built_entries) = timed(Phase::Build, || {
            pullback_rows(
                self.q,
                self.differentials,
                self.tables,
                self.cx,
                &src,
                &tgt,
                0,
                |pos| if is_cleared(pos) { 0 } else { u16::MAX },
            )
        });
        let mut stats = DualStats {
            cochains: tgt.dim(),
            largest: tgt.dim(),
            built_entries,
            ..DualStats::default()
        };
        let flip = |pos: u32| src.dim() as u32 - 1 - pos;
        let mut rows = SparseEchelon::new(src.dim());
        let mut ws = SparseWorkspace::new(src.dim());
        let mut pivots = vec![0u64; src.dim().div_ceil(64)];
        let mut functionals = Functionals::default();
        // the functional of each reduced row
        let mut of_row: Vec<u32> = Vec::new();
        timed(Phase::Solve, || {
            for pos in (0..tgt.dim() as u32).rev() {
                if is_cleared(pos) {
                    continue;
                }
                let row = std::mem::take(&mut built[pos as usize]);
                stats.rows += 1;
                stats.input_entries += row.len();
                let mut used = Vec::new();
                let rest = match row.last() {
                    Some(&first) if rows.pivot_column(first as usize).is_some() => {
                        let rest = rows.reduce(&mut ws, row, Some(&mut used));
                        stats.additions += ws.take_additions();
                        rest
                    }
                    _ => {
                        stats.apparent += 1;
                        row
                    }
                };
                let psi = if used.is_empty() {
                    vec![pos]
                } else {
                    let mut psi = vec![pos];
                    for c in used {
                        psi.extend(&functionals.list[of_row[c as usize] as usize].psi);
                    }
                    crate::ctau_scale::cancel_pairs(&mut psi);
                    psi
                };
                let birth = tgt.level(pos);
                let death = match rest.last() {
                    Some(&first) => {
                        let first = flip(first);
                        pivots[first as usize / 64] |= 1 << (first % 64);
                        stats.reduced_entries += rest.len();
                        of_row.push(functionals.list.len() as u32);
                        rows.insert(rest);
                        src.level(first)
                    }
                    None => NEVER,
                };
                if birth < death {
                    functionals.towers.push(functionals.list.len() as u32);
                }
                functionals.list.push(Functional {
                    pos,
                    psi,
                    birth,
                    death,
                });
            }
        });
        drop(built);
        let mut towers_csv = String::new();
        let mut h_csv = String::new();
        let mut cell_csv = String::new();
        let mut towers = Vec::new();
        if self.in_range(a + 1, u) {
            stats.towers = functionals.towers.len();
            for (i, &f) in functionals.towers.iter().enumerate() {
                let f = &functionals.list[f as usize];
                let length = match (f.death, self.cx.max_k) {
                    (NEVER, _) if self.cx.quotient > 0 => "1".to_string(),
                    // a tower still standing at the bound: its length is at least this
                    (NEVER, Some(k)) if (a + 1, u) != (0, 0) => {
                        format!("{}+", k as i32 + 1 - f.birth as i32)
                    }
                    (NEVER, _) => "inf".to_string(),
                    (death, _) => (death - f.birth).to_string(),
                };
                writeln!(
                    towers_csv,
                    "{},{},{},{u},{length}",
                    tower_id(a + 1, u, i),
                    2 * u - a - 1,
                    a + 1 + f.birth as i32
                )?;
                towers.push(Tower {
                    psi: f.psi.clone(),
                    death: f.death,
                });
            }
            if let Some(pulled) = pulled {
                timed(Phase::Maps, || {
                    self.finish(
                        u,
                        &tgt,
                        &functionals,
                        pulled,
                        &mut h_csv,
                        &mut cell_csv,
                        &mut stats,
                    )
                })?;
            }
        }
        Ok(Reduction {
            u,
            rows,
            towers: Arc::new(towers),
            pivots,
            towers_csv,
            h_csv,
            cell_csv,
            stats,
        })
    }

    /// Writes each of `pulled`, functionals on the space `space` at filtration `a + 1` and
    /// weight `u`, in the functionals of that space, and the coefficients on towers as rows of
    /// the products.
    ///
    /// Below the level at which the tower pulled back dies its image is a functional on
    /// cohomology, so it involves no functional that has died by then: this is checked, and it
    /// is what makes the coefficients products.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        u: i32,
        space: &Space,
        functionals: &Functionals,
        pulled: &[Pulled],
        h_csv: &mut String,
        cell_csv: &mut String,
        stats: &mut DualStats,
    ) -> anyhow::Result<()> {
        use std::fmt::Write;
        const NONE: u32 = u32::MAX;
        let a = self.a + 1;
        let mut at = vec![NONE; space.dim()];
        for (i, f) in functionals.list.iter().enumerate() {
            at[f.pos as usize] = i as u32;
        }
        let mut tower_of = vec![NONE; functionals.list.len()];
        for (i, &f) in functionals.towers.iter().enumerate() {
            tower_of[f as usize] = i as u32;
        }
        // (map, source tower) -> target ids
        let mut products: BTreeMap<(u8, u32), Vec<String>> = BTreeMap::new();
        for p in pulled {
            let mut rest: std::collections::BTreeSet<u32> = p.entries.iter().copied().collect();
            while let Some(first) = rest.pop_first() {
                let i = at[first as usize];
                anyhow::ensure!(
                    i != NONE,
                    "({a}, {u}): position {first} is a pivot and no functional starts there"
                );
                let f = &functionals.list[i as usize];
                anyhow::ensure!(
                    f.death >= p.death && f.birth < f.death,
                    "({a}, {u}): a functional that dies at level {} in the image of a tower \
                     that dies at {}",
                    f.death,
                    p.death
                );
                products
                    .entry((p.map, tower_of[i as usize]))
                    .or_default()
                    .push(tower_id(p.target.0, p.target.1, p.target.2 as usize));
                for &x in &f.psi[1..] {
                    if x < p.cut && !rest.remove(&x) {
                        rest.insert(x);
                    }
                }
            }
        }
        for ((map, source), targets) in products {
            let source = tower_id(a, u, source as usize);
            if map < TOP_CELL {
                stats.products += targets.len();
                writeln!(h_csv, "{},{source},{}", map + 1, targets.join(" "))?;
            } else {
                writeln!(cell_csv, "{source},{}", targets.join(" "))?;
            }
        }
        Ok(())
    }

    /// Reduces a functional on the source space of `red`, held in `bits` at the positions
    /// from `keep_from` on (counted from the end), by the rows of `red`, and returns the
    /// positions left, which are those where a functional of the source space starts. `high`
    /// is one past the last word of `bits` that may be nonzero; `bits` is left zero.
    fn eliminate(
        red: &Reduction,
        bits: &mut [u64],
        mut high: usize,
        keep_from: u32,
        stats: &mut DualStats,
    ) -> Vec<u32> {
        let dim = red.rows.rows() as u32;
        let mut entries = Vec::new();
        let low = keep_from as usize / 64;
        while high > low {
            let word = bits[high - 1];
            if word == 0 {
                high -= 1;
                continue;
            }
            let r = (high - 1) * 64 + 63 - word.leading_zeros() as usize;
            if (r as u32) < keep_from {
                bits[high - 1] = 0;
                high -= 1;
                continue;
            }
            match red.rows.pivot_column(r) {
                Some(c) => {
                    let row = red.rows.column(c);
                    let from = row.partition_point(|&x| x < keep_from);
                    stats.map_additions += row.len() - from;
                    for &x in &row[from..] {
                        bits[x as usize / 64] ^= 1 << (x % 64);
                    }
                }
                None => {
                    bits[r / 64] ^= 1 << (r % 64);
                    entries.push(dim - 1 - r as u32);
                }
            }
        }
        entries
    }

    /// Pulls towers back to the source space of `red` along the maps of `self.maps`: those one
    /// filtration up (in `above`, by weight) along $h_1, \dots, h_4$, and those of another
    /// complex in the same filtration along a cell map. Each is reduced by the rows of `red`
    /// as far as the tower lives; what is left is on the positions where a functional of the
    /// source space starts, for [`Self::finish`] one step later.
    fn pull(
        &self,
        red: &Reduction,
        above: &FxHashMap<i32, Arc<Vec<Tower>>>,
        stats: &mut DualStats,
    ) -> Vec<Pulled> {
        let (a, u) = (self.a, red.u);
        let mut out = Vec::new();
        if a < self.cx.min_a() || !self.in_range(a, u) {
            return out;
        }
        let src = Space::new(self.bb, self.q, self.cx, a, u);
        let dim = src.dim() as u32;
        let mut bits = vec![0u64; src.dim().div_ceil(64)];
        // Reduces the functional with the given positions of `src`, counted from the end (a
        // repeated one cancels), kept below the level at which the tower `target` dies.
        let mut pull_one = |map: u8,
                            target: (i32, i32, u32),
                            death: u16,
                            positions: &mut dyn Iterator<Item = u32>,
                            stats: &mut DualStats| {
            let cut = if death == NEVER {
                dim
            } else {
                src.level_end(death as usize - 1) as u32
            };
            // positions from the end at which the image is kept
            let keep_from = dim - cut;
            let mut high = 0;
            for r in positions {
                if r >= keep_from {
                    bits[r as usize / 64] ^= 1 << (r % 64);
                    high = high.max(r as usize / 64 + 1);
                    stats.map_entries += 1;
                }
            }
            let entries = Self::eliminate(red, &mut bits, high, keep_from, stats);
            if !entries.is_empty() {
                out.push(Pulled {
                    map,
                    target,
                    death,
                    entries,
                    cut,
                });
            }
        };
        let products = !matches!(self.maps, Maps::BottomCell(_));
        for j in (0..4u8).filter(|_| products) {
            let weight = u + (1 << j);
            let Some(towers) = above.get(&weight).filter(|_| self.in_range(a + 1, weight)) else {
                continue;
            };
            if towers.is_empty() {
                continue;
            }
            let tgt = Space::new(self.bb, self.q, self.cx, a + 1, weight);
            // each row is read as far as the towers that use it live
            let mut depth = vec![0u16; tgt.dim()];
            for tower in towers.iter() {
                for &p in &tower.psi {
                    depth[p as usize] = depth[p as usize].max(tower.death);
                }
            }
            let shift: PackedXi = 1 << j;
            let (rows, built_entries) = timed(Phase::MapBuild, || {
                pullback_rows(
                    self.q,
                    self.differentials,
                    self.tables,
                    self.cx,
                    &src,
                    &tgt,
                    shift,
                    |pos| depth[pos as usize],
                )
            });
            stats.map_built_entries += built_entries;
            for (i, tower) in towers.iter().enumerate() {
                let mut positions = tower
                    .psi
                    .iter()
                    .flat_map(|&p| rows[p as usize].iter().copied());
                pull_one(
                    j,
                    (a + 1, weight, i as u32),
                    tower.death,
                    &mut positions,
                    stats,
                );
            }
        }
        match self.maps {
            Maps::Products => {}
            // the coordinates on the top cell are a cochain of the sphere
            Maps::WithTopCell(base) => {
                let top = self.cx.cells.len() - 1;
                let weight = u - self.cx.cells[top][1];
                if let Some(towers) = base
                    .towers
                    .get(&(a, weight))
                    .filter(|_| self.in_range(a, weight))
                {
                    let down = Space::new(self.bb, self.q, &base.cx, a, weight);
                    for (i, tower) in towers.iter().enumerate() {
                        let mut positions = tower.psi.iter().map(|&p| {
                            let (_, t, x, m, _) = down.decode(self.q, p);
                            dim - 1 - src.pos(self.q, top, t, x, &m)
                        });
                        pull_one(
                            TOP_CELL,
                            (a, weight, i as u32),
                            tower.death,
                            &mut positions,
                            stats,
                        );
                    }
                }
            }
            // a cochain of the sphere is one on the bottom cell
            Maps::BottomCell(cofiber) => {
                if let Some(towers) = cofiber.towers.get(&(a, u)) {
                    let up = Space::new(self.bb, self.q, &cofiber.cx, a, u);
                    for (i, tower) in towers.iter().enumerate() {
                        let mut positions = tower.psi.iter().filter_map(|&p| {
                            let (cell, t, x, m, _) = up.decode(self.q, p);
                            (cell == 0).then(|| dim - 1 - src.pos(self.q, 0, t, x, &m))
                        });
                        pull_one(
                            BOTTOM_CELL,
                            (a, u, i as u32),
                            tower.death,
                            &mut positions,
                            stats,
                        );
                    }
                }
            }
        }
        out
    }
}

/// Exports the towers of `cx` in stems `0..=max_n` into `dir`, with `h_1, ..., h_4`, in the
/// files and formats of [`crate::ctau_lazy::export`].
///
/// The coboundary $C^a \to C^{a+1}$ is reduced by rows, which are functionals on $C^a$: the
/// rows are taken from the last cochain of $C^{a+1}$ to the first and the pivot of a row is its
/// first source. The pairs of a pivot and the cochain its row started from are those of the
/// reduction by columns, so the towers are the same, and the work is far less: most rows are
/// reduced as they stand. Filtrations go from the top down, since the pivots of one
/// coboundary are the rows the coboundary below it need not reduce.
///
/// What this finds in place of cocycles is functionals ([`Functional`]): a tower is a
/// functional on the cohomology of the polynomial degrees it lives in, and the towers alive in
/// a degree are a basis of the dual. A class is known by its values on them, and the products
/// are computed on them, transposed: a tower functional composed with $h_j$ is written in the
/// tower functionals one filtration down. The basis of the cohomology dual to the towers is
/// therefore not that of the reduction by columns, though the towers correspond, and no
/// cocycle representing a class is produced.
///
/// # Panics
///
/// If a cell of `cx` is shifted in filtration or polynomial degree.
#[allow(clippy::too_many_arguments)]
pub fn export(
    q: &QAlgebra,
    source: &BackboneSource,
    tables: &OpTables,
    cx: &Complex,
    max_n: i32,
    maps: &Maps,
    keep: &(dyn Fn(PackedXi) -> bool + Sync),
    dir: &Path,
    options: &DualOptions,
) -> anyhow::Result<(Exported, DualStats)> {
    use maybe_rayon::prelude::*;
    assert!(
        cx.cells.iter().all(|c| c[0] == 0 && c[2] == 0),
        "cells shifted in filtration or polynomial degree"
    );
    let name = &cx.name;
    let open = |file: &str, header: &str| -> anyhow::Result<BufWriter<File>> {
        let mut w = BufWriter::new(File::create(dir.join(format!("{name}_{file}.csv")))?);
        writeln!(w, "{header}")?;
        Ok(w)
    };
    let mut w_own = match maps {
        Maps::BottomCell(_) => None,
        _ => Some((
            open("towers", "id,stem,filtration,weight,length")?,
            open("h", "j,source,targets")?,
        )),
    };
    let mut w_cell = match maps {
        Maps::Products => None,
        Maps::WithTopCell(_) => Some(open("pstar", "source,targets")?),
        Maps::BottomCell(cofiber) => {
            let path = dir.join(format!("{}_istar.csv", cofiber.cx.name));
            let mut w = BufWriter::new(File::create(path)?);
            writeln!(w, "source,targets")?;
            Some(w)
        }
    };
    let mut exported = Exported {
        cx: cx.clone(),
        towers: FxHashMap::default(),
    };
    let min_a = cx.min_a();
    // the lowest source filtration read at weight u
    let a_lo = |u: i32| (2 * u - max_n - 1).max(min_a);
    let mut stats = DualStats::default();
    // by weight, for the space one filtration above the source of the step
    let mut cleared: FxHashMap<i32, Vec<u64>> = FxHashMap::default();
    let mut pulled: FxHashMap<i32, Vec<Pulled>> = FxHashMap::default();
    // The step below the lowest filtration has no coboundary: every cochain of the lowest
    // filtration that is no pivot is a functional there.
    for a in (min_a - 1..max_n).rev() {
        let mut weights: Vec<i32> = (0..=max_n)
            .filter(|&u| {
                if a < min_a {
                    a_lo(u) == min_a
                } else {
                    // at u = a there is nothing above to reduce into, and products out of
                    a_lo(u) <= a && a <= u
                }
            })
            .collect();
        if weights.is_empty() {
            continue;
        }
        let differentials = timed(Phase::Slices, || source.forward(a + 1, keep))?;
        // the spaces read the generators of the backbone and nothing else of it
        let window = source.window(Vec::new());
        let step = Step {
            q,
            tables,
            cx,
            max_n,
            maps,
            a,
            differentials: &differentials,
            bb: &window,
        };
        // A product into weight u + 2^j is pulled back to weight u, so the higher weights go
        // first.
        weights.sort_unstable_by_key(|&u| std::cmp::Reverse(u));
        let mut above: FxHashMap<i32, Arc<Vec<Tower>>> = FxHashMap::default();
        let mut next_cleared: FxHashMap<i32, Vec<u64>> = FxHashMap::default();
        let mut next_pulled: FxHashMap<i32, Vec<Pulled>> = FxHashMap::default();
        for group in weights.chunks(options.in_flight.max(1)) {
            let reduced: Vec<anyhow::Result<Reduction>> = group
                .to_vec()
                .into_maybe_par_iter()
                .map(|u| step.reduce(u, cleared.get(&u), pulled.get(&u)))
                .collect();
            let reduced: Vec<Reduction> = reduced.into_iter().collect::<anyhow::Result<_>>()?;
            for red in &reduced {
                above.insert(red.u, Arc::clone(&red.towers));
                if !red.towers.is_empty() {
                    exported
                        .towers
                        .insert((a + 1, red.u), Arc::clone(&red.towers));
                }
                if let Some((w_towers, w_h)) = w_own.as_mut() {
                    w_towers.write_all(red.towers_csv.as_bytes())?;
                    w_h.write_all(red.h_csv.as_bytes())?;
                }
                if let Some(w) = w_cell.as_mut() {
                    w.write_all(red.cell_csv.as_bytes())?;
                }
                stats = stats.merge(red.stats);
            }
            let maps: Vec<(i32, Vec<u64>, Vec<Pulled>, DualStats)> = reduced
                .into_maybe_par_iter()
                .map(|red| {
                    let mut local = DualStats::default();
                    let out = timed(Phase::Maps, || step.pull(&red, &above, &mut local));
                    (red.u, red.pivots, out, local)
                })
                .collect();
            for (u, pivots, out, local) in maps {
                next_cleared.insert(u, pivots);
                next_pulled.insert(u, out);
                stats = stats.merge(local);
            }
        }
        cleared = next_cleared;
        pulled = next_pulled;
    }
    Ok((exported, stats))
}
