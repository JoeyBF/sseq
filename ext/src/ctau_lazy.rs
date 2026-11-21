//! The tower export of the $C\tau$ Adams $E_2$ page in bounded memory: one filtration at a time, reduced columns spilled to disk.

use std::{
    fs::File,
    io::{BufWriter, Write},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use fp::matrix::SparseWorkspace;
use rustc_hash::FxHashMap;

use crate::{
    ctau::QAlgebra,
    ctau_scale::{
        Backbone, BackboneSource, Complex, NEVER, OpTables, ReduceStats, Reduced, Slice, Space,
        pullback_columns, reduce_step,
    },
};

/// Appends `x` in base 128, low digits first.
fn put_varint(out: &mut Vec<u8>, mut x: u64) {
    while x >= 0x80 {
        out.push(x as u8 | 0x80);
        x >>= 7;
    }
    out.push(x as u8);
}

/// Reads what [`put_varint`] wrote and advances `data`.
fn get_varint(data: &mut &[u8]) -> u64 {
    let mut x = 0;
    let mut shift = 0;
    loop {
        let b = data[0];
        *data = &data[1..];
        x |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 {
            return x;
        }
        shift += 7;
    }
}

/// Appends reduced columns with their births and deaths: each column as its length and the gaps
/// between its sorted entries, which are small.
fn encode<'a>(out: &mut Vec<u8>, columns: impl Iterator<Item = (&'a [u32], u16, u16)>) {
    for (column, birth, death) in columns {
        put_varint(out, birth as u64);
        put_varint(out, death as u64);
        put_varint(out, column.len() as u64);
        let mut last = 0;
        for &r in column {
            put_varint(out, (r - last) as u64);
            last = r;
        }
    }
}

/// Reads `count` columns written by [`encode`].
fn decode(data: &mut &[u8], count: usize) -> Vec<(Vec<u32>, u16, u16)> {
    (0..count)
        .map(|_| {
            let birth = get_varint(data) as u16;
            let death = get_varint(data) as u16;
            let len = get_varint(data) as usize;
            let mut column = Vec::with_capacity(len);
            let mut last = 0u32;
            for _ in 0..len {
                last += get_varint(data) as u32;
                column.push(last);
            }
            (column, birth, death)
        })
        .collect()
}

/// What a spilled record holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    /// Every reduced column of a filtration and weight: the cocycles, to take classes in.
    Echelon,
    /// The columns that generate towers only: the sources of the maps out.
    Generators,
}

/// One append-only file of records, indexed in memory, so that what is written sequentially is
/// read back with one seek per record.
pub struct Spill {
    path: PathBuf,
    file: File,
    /// The end of the file, and where each record is.
    state: Mutex<(u64, FxHashMap<(Kind, i32, i32), (u64, u64)>)>,
    written: AtomicU64,
    read: AtomicU64,
}

impl Spill {
    /// A new, empty spill file at `path`.
    pub fn create(path: PathBuf) -> anyhow::Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        Ok(Self {
            path,
            file,
            state: Mutex::new((0, FxHashMap::default())),
            written: AtomicU64::new(0),
            read: AtomicU64::new(0),
        })
    }

    /// Appends a record.
    fn put(&self, key: (Kind, i32, i32), data: &[u8]) -> anyhow::Result<()> {
        let mut state = self.state.lock().unwrap();
        let at = state.0;
        self.file.write_all_at(data, at)?;
        state.0 += data.len() as u64;
        state.1.insert(key, (at, data.len() as u64));
        self.written.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Reads a record back, if it was written.
    fn get(&self, key: (Kind, i32, i32)) -> anyhow::Result<Option<Vec<u8>>> {
        let Some(&(at, len)) = self.state.lock().unwrap().1.get(&key) else {
            return Ok(None);
        };
        let mut data = vec![0; len as usize];
        self.file.read_exact_at(&mut data, at)?;
        self.read.fetch_add(len, Ordering::Relaxed);
        Ok(Some(data))
    }

    /// The bytes written and read so far.
    pub fn traffic(&self) -> (u64, u64) {
        (
            self.written.load(Ordering::Relaxed),
            self.read.load(Ordering::Relaxed),
        )
    }

    /// The tower generators at filtration `a` and weight `u`, if any were written.
    fn generators(&self, a: i32, u: i32) -> anyhow::Result<Option<Vec<(Vec<u32>, u16, u16)>>> {
        Ok(self.get((Kind::Generators, a, u))?.map(|data| {
            let mut cursor = &data[..];
            let count = get_varint(&mut cursor) as usize;
            decode(&mut cursor, count)
        }))
    }

    /// The cocycles at filtration `a` and weight `u`, if they were written.
    fn echelon(&self, a: i32, u: i32) -> anyhow::Result<Option<Reduced>> {
        Ok(self.get((Kind::Echelon, a, u))?.map(|data| {
            let mut cursor = &data[..];
            let rows = get_varint(&mut cursor) as usize;
            let count = get_varint(&mut cursor) as usize;
            Reduced::from_columns(rows, decode(&mut cursor, count))
        }))
    }

    /// Writes the tower generators of `red`, and all of its columns if `whole`.
    fn put_reduced(&self, a: i32, u: i32, red: &Reduced, whole: bool) -> anyhow::Result<()> {
        let mut data = Vec::new();
        put_varint(&mut data, red.towers.len() as u64);
        encode(
            &mut data,
            (0..red.towers.len()).map(|i| {
                let (birth, length) = red.tower(i);
                let death = if length == NEVER {
                    NEVER
                } else {
                    birth + length
                };
                (red.generator(i), birth, death)
            }),
        );
        self.put((Kind::Generators, a, u), &data)?;
        if whole {
            data.clear();
            put_varint(&mut data, red.echelon.rows() as u64);
            put_varint(&mut data, red.echelon.len() as u64);
            encode(&mut data, red.columns());
            self.put((Kind::Echelon, a, u), &data)?;
        }
        Ok(())
    }
}

impl Drop for Spill {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// What an export did.
#[derive(Debug, Default, Clone, Copy)]
pub struct LazyStats {
    pub reduce: ReduceStats,
    pub towers: usize,
    /// Bytes spilled, and bytes read back, over this complex's file and its base's.
    pub written: u64,
    pub read: u64,
}

/// A complex that has been exported, as what a later complex reads of it: its spill file.
pub struct Exported {
    pub cx: Complex,
    spill: Spill,
}

/// How an export bounds its memory.
#[derive(Debug, Clone)]
pub struct LazyOptions {
    /// The weights of one filtration reduced at the same time: the peak is that many of the
    /// largest reductions, so lowering it trades time for memory.
    pub in_flight: usize,
    /// Where the spill files go. They are large and sequential.
    pub spill_dir: PathBuf,
}

/// The id of tower `i` at filtration `a` and weight `u`.
fn tower_id(a: i32, u: i32, i: usize) -> String {
    format!("{a}_{u}_{i}")
}

/// The CSV rows a task produced, by file.
#[derive(Default)]
struct Rows {
    towers: String,
    h: String,
    istar: String,
    pstar: String,
}

/// Everything a task reads that is fixed for a pass.
struct Pass<'a> {
    q: &'a QAlgebra,
    bb: &'a dyn Backbone,
    tables: &'a OpTables,
    cx: &'a Complex,
    max_n: i32,
    spill: &'a Spill,
    /// The sphere, already exported, when `cx` is a cofiber.
    base: Option<&'a Exported>,
    /// Whether later complexes take classes in this one.
    keep_echelons: bool,
}

impl Pass<'_> {
    /// Whether the classes at filtration `a` and weight `u` are in the export.
    fn in_range(&self, a: i32, u: i32) -> bool {
        (0..=self.max_n).contains(&(2 * u - a))
    }

    /// The rows of the towers of `red` at `(a, u)` and of every map into them: `h_j` from the
    /// generators one filtration down, the bottom cell from the sphere's generators, and the
    /// top cell into the sphere's cocycles.
    fn maps_into(&self, a: i32, u: i32, red: &Reduced, rows: &mut Rows) -> anyhow::Result<()> {
        use std::fmt::Write;
        if !self.in_range(a, u) {
            return Ok(());
        }
        for i in 0..red.towers.len() {
            let (birth, length) = red.tower(i);
            let length = if length == NEVER {
                "inf".to_string()
            } else {
                length.to_string()
            };
            writeln!(
                rows.towers,
                "{},{},{},{u},{length}",
                tower_id(a, u, i),
                2 * u - a,
                a + birth as i32
            )?;
        }
        let mut ws = SparseWorkspace::new(0);
        let targets = |towers: Vec<u32>| -> String {
            let ids: Vec<String> = towers
                .into_iter()
                .map(|t| tower_id(a, u, t as usize))
                .collect();
            ids.join(" ")
        };
        let tgt = Space::new(self.bb, self.q, self.cx, a, u);
        if a > self.cx.min_a() {
            for j in 0..4u32 {
                let u0 = u - (1 << j);
                if !self.in_range(a - 1, u0) {
                    continue;
                }
                let Some(gens) = self.spill.generators(a - 1, u0)? else {
                    continue;
                };
                if gens.is_empty() {
                    continue;
                }
                let src = Space::new(self.bb, self.q, self.cx, a - 1, u0);
                let mut wanted = vec![false; src.dim()];
                for (column, _, _) in &gens {
                    for &pos in column {
                        wanted[pos as usize] = true;
                    }
                }
                let mut columns: Vec<Vec<u32>> = vec![Vec::new(); src.dim()];
                for (pos, column) in pullback_columns(
                    self.q,
                    self.bb,
                    self.tables,
                    self.cx,
                    &src,
                    &tgt,
                    1 << j,
                    |pos| !wanted[pos as usize],
                ) {
                    columns[pos as usize] = column;
                }
                for (i, (column, birth, _)) in gens.iter().enumerate() {
                    let entries = column
                        .iter()
                        .flat_map(|&pos| columns[pos as usize].iter().copied());
                    let image = red.class(&mut ws, entries, *birth);
                    if !image.is_empty() {
                        writeln!(
                            rows.h,
                            "{},{},{}",
                            j + 1,
                            tower_id(a - 1, u0, i),
                            targets(image)
                        )?;
                    }
                }
            }
        }
        let Some(base) = self.base else {
            return Ok(());
        };
        // bottom cell: the same cochain on cell 0
        if let Some(gens) = base.spill.generators(a, u)? {
            let src = Space::new(self.bb, self.q, &base.cx, a, u);
            for (i, (column, birth, _)) in gens.iter().enumerate() {
                let entries = column.iter().map(|&pos| {
                    let (_, t, x, m, _) = src.decode(self.q, pos);
                    tgt.pos(self.q, 0, t, x, &m)
                });
                let image = red.class(&mut ws, entries, *birth);
                if !image.is_empty() {
                    writeln!(rows.istar, "{},{}", tower_id(a, u, i), targets(image))?;
                }
            }
        }
        // top cell: the coordinates on the top cell, which are a cocycle of the sphere
        let top = self.cx.cells.len() - 1;
        let [da, du, dk] = self.cx.cells[top];
        let (a2, u2) = (a + da, u - du);
        if red.towers.is_empty() || !self.in_range(a2, u2) {
            return Ok(());
        }
        let Some(below) = base.spill.echelon(a2, u2)? else {
            return Ok(());
        };
        let down = Space::new(self.bb, self.q, &base.cx, a2, u2);
        for i in 0..red.towers.len() {
            let entries: Vec<u32> = red
                .generator(i)
                .iter()
                .filter_map(|&pos| {
                    let (cell, t, x, m, _) = tgt.decode(self.q, pos);
                    (cell == top).then(|| down.pos(self.q, 0, t, x, &m))
                })
                .collect();
            let level = red.tower(i).0 as i32 - dk;
            if level < 0 {
                assert!(entries.is_empty());
                continue;
            }
            let image = below.class(&mut ws, entries, level as u16);
            if !image.is_empty() {
                let ids: Vec<String> = image
                    .into_iter()
                    .map(|t| tower_id(a2, u2, t as usize))
                    .collect();
                writeln!(rows.pstar, "{},{}", tower_id(a, u, i), ids.join(" "))?;
            }
        }
        Ok(())
    }
}

/// The rows of a bit set over `rows` positions that are pivots of `red`.
fn pivots(red: &Reduced) -> Vec<u64> {
    let mut bits = vec![0u64; red.echelon.rows().div_ceil(64)];
    for (column, _, _) in red.columns() {
        let p = *column.last().unwrap() as usize;
        bits[p / 64] |= 1 << (p % 64);
    }
    bits
}

/// Exports the towers of `cx` in stems `0..=max_n` into `dir`, with `h_1, ..., h_4` and, given
/// the sphere as `base`, the maps of the bottom and top cell.
///
/// Filtration by filtration: the differentials of one filtration of the backbone are read,
/// every weight's coboundary out of that filtration is reduced, the maps into the cocycles it
/// yields are computed while they are in memory, and they are written to the spill file and
/// dropped. What stays in memory between filtrations is one bit per cochain (the pivots, which
/// the next filtration skips). A map out of a filtration reads the generators back; the top
/// cell reads the sphere's cocycles back.
#[allow(clippy::too_many_arguments)]
pub fn export(
    q: &QAlgebra,
    source: &BackboneSource,
    tables: &OpTables,
    cx: Complex,
    max_n: i32,
    base: Option<&Exported>,
    keep_echelons: bool,
    dir: &Path,
    options: &LazyOptions,
) -> anyhow::Result<(Exported, LazyStats)> {
    use maybe_rayon::prelude::*;
    let name = cx.name.clone();
    let spill = Spill::create(options.spill_dir.join(format!("{name}.spill")))?;
    let open = |file: &str, header: &str| -> anyhow::Result<BufWriter<File>> {
        let mut w = BufWriter::new(File::create(dir.join(format!("{name}_{file}.csv")))?);
        writeln!(w, "{header}")?;
        Ok(w)
    };
    let mut w_towers = open("towers", "id,stem,filtration,weight,length")?;
    let mut w_h = open("h", "j,source,targets")?;
    let mut w_cells = match base {
        Some(_) => Some((
            open("istar", "source,targets")?,
            open("pstar", "source,targets")?,
        )),
        None => None,
    };
    let min_a = cx.min_a();
    // the lowest source filtration read at weight u, and one past the highest
    let a_lo = |u: i32| (2 * u - max_n - 1).max(min_a);
    let a_hi = |u: i32| u.max(a_lo(u) + 1);
    let shifts: Vec<i32> = {
        let mut v: Vec<i32> = cx.cells.iter().map(|c| c[0]).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut slices: FxHashMap<i32, Arc<Slice>> = FxHashMap::default();
    let mut cleared: FxHashMap<i32, Vec<u64>> = FxHashMap::default();
    let stats = Mutex::new(LazyStats::default());
    for a in min_a..=max_n {
        let mut weights: Vec<i32> = (0..=max_n)
            .filter(|&u| a_lo(u) <= a && a < a_hi(u))
            .collect();
        if weights.is_empty() {
            continue;
        }
        // the differentials out of the filtrations the cells put above a, and only those
        slices.retain(|&f, _| f > a);
        for &da in &shifts {
            let f = a + da + 1;
            if !slices.contains_key(&f) {
                slices.insert(f, Arc::new(source.slice(f)?));
            }
        }
        let window = source.window(slices.iter().map(|(&f, s)| (f, Arc::clone(s))).collect());
        let pass = Pass {
            q,
            bb: &window,
            tables,
            cx: &cx,
            max_n,
            spill: &spill,
            base,
            keep_echelons,
        };
        // heaviest weights first within a group in flight
        weights.sort_by_key(|&u| (2 * u - max_n).abs());
        let mut next_cleared: FxHashMap<i32, Vec<u64>> = FxHashMap::default();
        // In the lowest filtration the maps out of its own cocycles need every weight's, so
        // they are held (they are small) and the maps come after.
        let mut held: Vec<(i32, Reduced, Option<Reduced>)> = Vec::new();
        for group in weights.chunks(options.in_flight.max(1)) {
            type Done = (i32, Vec<u64>, Rows, Option<(Reduced, Option<Reduced>)>);
            let done: Vec<anyhow::Result<Done>> = group
                .to_vec()
                .into_maybe_par_iter()
                .map(|u| {
                    let src = Space::new(pass.bb, q, &cx, a, u);
                    let tgt = Space::new(pass.bb, q, &cx, a + 1, u);
                    let skip = cleared.get(&u);
                    let mut ws = SparseWorkspace::new(0);
                    let mut local = ReduceStats::default();
                    let (red, essential) = reduce_step(
                        q,
                        pass.bb,
                        tables,
                        &cx,
                        &src,
                        &tgt,
                        |pos| {
                            skip.is_some_and(|bits| bits[pos as usize / 64] >> (pos % 64) & 1 == 1)
                        },
                        &mut ws,
                        &mut local,
                    );
                    drop((src, tgt));
                    let bits = pivots(&red);
                    pass.spill.put_reduced(a + 1, u, &red, pass.keep_echelons)?;
                    if let Some(ess) = &essential {
                        pass.spill.put_reduced(a, u, ess, pass.keep_echelons)?;
                    }
                    {
                        let mut s = stats.lock().unwrap();
                        s.reduce = s.reduce.merge(local);
                        s.towers +=
                            red.towers.len() + essential.as_ref().map_or(0, |e| e.towers.len());
                    }
                    let mut rows = Rows::default();
                    if a == min_a {
                        return Ok((u, bits, rows, Some((red, essential))));
                    }
                    pass.maps_into(a + 1, u, &red, &mut rows)?;
                    Ok((u, bits, rows, None))
                })
                .collect();
            for d in done {
                let (u, bits, rows, kept) = d?;
                next_cleared.insert(u, bits);
                w_towers.write_all(rows.towers.as_bytes())?;
                w_h.write_all(rows.h.as_bytes())?;
                if let Some((wi, wp)) = w_cells.as_mut() {
                    wi.write_all(rows.istar.as_bytes())?;
                    wp.write_all(rows.pstar.as_bytes())?;
                }
                if let Some((red, essential)) = kept {
                    held.push((u, red, essential));
                }
            }
        }
        for (u, red, essential) in &held {
            let mut rows = Rows::default();
            if let Some(ess) = essential {
                pass.maps_into(a, *u, ess, &mut rows)?;
            }
            pass.maps_into(a + 1, *u, red, &mut rows)?;
            w_towers.write_all(rows.towers.as_bytes())?;
            w_h.write_all(rows.h.as_bytes())?;
            if let Some((wi, wp)) = w_cells.as_mut() {
                wi.write_all(rows.istar.as_bytes())?;
                wp.write_all(rows.pstar.as_bytes())?;
            }
        }
        cleared = next_cleared;
    }
    let mut stats = stats.into_inner().unwrap();
    let (written, read) = spill.traffic();
    let (bw, br) = base.map_or((0, 0), |b| b.spill.traffic());
    stats.written = written + bw;
    stats.read = read + br;
    Ok((Exported { cx, spill }, stats))
}
