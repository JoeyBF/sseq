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
        BackboneSource, Complex, Forward, NEVER, OpTables, Phase, Space, forward, pullback_rows,
        timed,
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

/// The image of a tower functional under the transpose of a cochain map, reduced by the rows
/// out of its source space: what is left for the functionals of that space to account for.
struct Pulled {
    /// The map: $h_{j+1}$.
    j: u8,
    /// The weight and index of the tower pulled back, one filtration up.
    weight: i32,
    tower: u32,
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
    /// The entries of the rows built, of the rows added during reduction, and of the reduced
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
    functionals: Arc<Functionals>,
    /// The positions of the source space that are pivots, as a bit set.
    pivots: Vec<u64>,
    towers_csv: String,
    h_csv: String,
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
        let mut built = timed(Phase::Build, || {
            pullback_rows(
                self.q,
                self.differentials,
                self.tables,
                self.cx,
                &src,
                &tgt,
                0,
                |pos| !is_cleared(pos),
            )
        });
        let mut stats = DualStats {
            cochains: tgt.dim(),
            largest: tgt.dim(),
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
            }
            if let Some(pulled) = pulled {
                timed(Phase::Maps, || {
                    self.finish_products(u, &tgt, &functionals, pulled, &mut h_csv, &mut stats)
                })?;
            }
        }
        Ok(Reduction {
            u,
            rows,
            functionals: Arc::new(functionals),
            pivots,
            towers_csv,
            h_csv,
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
    fn finish_products(
        &self,
        u: i32,
        space: &Space,
        functionals: &Functionals,
        pulled: &[Pulled],
        h_csv: &mut String,
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
        // (j, source tower) -> target ids
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
                    .entry((p.j, tower_of[i as usize]))
                    .or_default()
                    .push(tower_id(a + 1, p.weight, p.tower as usize));
                for &x in &f.psi[1..] {
                    if x < p.cut && !rest.remove(&x) {
                        rest.insert(x);
                    }
                }
            }
        }
        for ((j, source), targets) in products {
            stats.products += targets.len();
            writeln!(
                h_csv,
                "{},{},{}",
                j + 1,
                tower_id(a, u, source as usize),
                targets.join(" ")
            )?;
        }
        Ok(())
    }

    /// Pulls the towers one filtration up back along $h_1, \dots, h_4$ to the source space of
    /// `red`, and reduces each by the rows of `red` as far as the tower lives: what is left is
    /// on the positions where a functional of the source space starts, for
    /// [`Self::finish_products`] one step later.
    fn pull_products(
        &self,
        red: &Reduction,
        above: &FxHashMap<i32, Arc<Functionals>>,
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
        for j in 0..4u8 {
            let weight = u + (1 << j);
            let Some(towers) = above.get(&weight).filter(|_| self.in_range(a + 1, weight)) else {
                continue;
            };
            if towers.towers.is_empty() {
                continue;
            }
            let tgt = Space::new(self.bb, self.q, self.cx, a + 1, weight);
            let mut wanted = vec![0u64; tgt.dim().div_ceil(64)];
            for &f in &towers.towers {
                for &p in &towers.list[f as usize].psi {
                    wanted[p as usize / 64] |= 1 << (p % 64);
                }
            }
            let shift: PackedXi = 1 << j;
            let rows = timed(Phase::MapBuild, || {
                pullback_rows(
                    self.q,
                    self.differentials,
                    self.tables,
                    self.cx,
                    &src,
                    &tgt,
                    shift,
                    |pos| wanted[pos as usize / 64] >> (pos % 64) & 1 == 1,
                )
            });
            for (i, &f) in towers.towers.iter().enumerate() {
                let f = &towers.list[f as usize];
                let cut = if f.death == NEVER {
                    dim
                } else {
                    src.level_end(f.death as usize - 1) as u32
                };
                // positions from the end at which the image is kept
                let keep_from = dim - cut;
                let mut high = 0;
                for &p in &f.psi {
                    for &r in &rows[p as usize] {
                        if r >= keep_from {
                            bits[r as usize / 64] ^= 1 << (r % 64);
                            high = high.max(r as usize / 64 + 1);
                            stats.map_entries += 1;
                        }
                    }
                }
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
                if !entries.is_empty() {
                    out.push(Pulled {
                        j,
                        weight,
                        tower: i as u32,
                        death: f.death,
                        entries,
                        cut,
                    });
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
    keep: &(dyn Fn(PackedXi) -> bool + Sync),
    dir: &Path,
    options: &DualOptions,
) -> anyhow::Result<DualStats> {
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
    let mut w_towers = open("towers", "id,stem,filtration,weight,length")?;
    let mut w_h = open("h", "j,source,targets")?;
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
        let slice = Arc::new(timed(Phase::Slices, || source.slice(a + 1, keep))?);
        let differentials = forward(&slice);
        let window = source.window(vec![(a + 1, slice)]);
        let step = Step {
            q,
            tables,
            cx,
            max_n,
            a,
            differentials: &differentials,
            bb: &window,
        };
        // A product into weight u + 2^j is pulled back to weight u, so the higher weights go
        // first.
        weights.sort_unstable_by_key(|&u| std::cmp::Reverse(u));
        let mut above: FxHashMap<i32, Arc<Functionals>> = FxHashMap::default();
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
                above.insert(red.u, Arc::clone(&red.functionals));
                w_towers.write_all(red.towers_csv.as_bytes())?;
                w_h.write_all(red.h_csv.as_bytes())?;
                stats = stats.merge(red.stats);
            }
            let maps: Vec<(i32, Vec<u64>, Vec<Pulled>, DualStats)> = reduced
                .into_maybe_par_iter()
                .map(|red| {
                    let mut local = DualStats::default();
                    let out = timed(Phase::Maps, || step.pull_products(&red, &above, &mut local));
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
    Ok(stats)
}
