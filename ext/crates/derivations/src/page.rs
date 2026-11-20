//! Adams d_r on E_r = H(E_{r-1}, d_{r-1}) for r >= 3, constrained by the Leibniz rule on E_r
//! (`--e3`), with each page known only up to certain/possible bounds where earlier differentials
//! are open.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{Ext, Mask, Products, eliminate, parse_vec};

/// An echelon form keyed by lowest set bit; each row remembers which basis vectors it combines.
#[derive(Default, Clone)]
struct Ech {
    rows: FxHashMap<u32, (Mask, Mask)>,
}

impl Ech {
    /// Reduces `v`; returns the remainder and the basis coefficients used.
    fn reduce(&self, mut v: Mask) -> (Mask, Mask) {
        let mut coef = 0;
        while v != 0 {
            let Some(&(row, c)) = self.rows.get(&v.trailing_zeros()) else {
                break;
            };
            v ^= row;
            coef ^= c;
        }
        (v, coef)
    }

    /// Adds `v` as a new row unless it is already spanned; returns whether it was new.
    fn insert(&mut self, v: Mask, coef: Mask) -> bool {
        let (mut r, mut c) = (v, coef);
        loop {
            if r == 0 {
                return false;
            }
            let p = r.trailing_zeros();
            match self.rows.get(&p) {
                Some(&(row, rc)) => {
                    r ^= row;
                    c ^= rc;
                }
                None => {
                    self.rows.insert(p, (r, c));
                    return true;
                }
            }
        }
    }

    /// Coordinates of `v` in the basis, if `v` lies in the span of the rows.
    fn project(&self, v: Mask) -> Option<Mask> {
        let (rest, coef) = self.reduce(v);
        (rest == 0).then_some(coef)
    }
}

/// E_r at one bidegree, as bounds. Sources are certain cycles modulo certain boundaries: every
/// relation among them holds in the true E_r. Targets are classes modulo every possible
/// boundary: the true E_r maps onto them, so true equations project to true equations.
#[derive(Default)]
struct Page {
    /// E2 masks of certain cycles, independent modulo the certain boundaries
    reps: Vec<Mask>,
    src: Ech,
    /// E2 masks spanning the possible E_r modulo the possible boundaries
    treps: Vec<Mask>,
    tgt: Ech,
    bcert: Vec<Mask>,
    bmax: Vec<Mask>,
    /// the possible boundaries are exactly the certain ones
    exact: bool,
    /// some class may be a cycle without being among the certain ones
    uncertain: bool,
}

impl Page {
    fn new(
        bcert: Vec<Mask>,
        bmax: Vec<Mask>,
        cycles: &[Mask],
        possible: &[Mask],
        exact: bool,
        uncertain: bool,
    ) -> Self {
        let mut pg = Page {
            exact,
            uncertain,
            ..Default::default()
        };
        for &v in &bcert {
            pg.src.insert(v, 0);
        }
        for &z in cycles {
            let k = pg.reps.len();
            if pg.src.insert(z, 1 << k) {
                pg.reps.push(z);
            }
        }
        for &v in &bmax {
            pg.tgt.insert(v, 0);
        }
        for &z in possible {
            let k = pg.treps.len();
            if pg.tgt.insert(z, 1 << k) {
                pg.treps.push(z);
            }
        }
        pg.bcert = bcert;
        pg.bmax = bmax;
        pg
    }
}

/// Kernel of a linear map given by the images of the basis vectors.
fn kernel(images: &[Mask]) -> Vec<Mask> {
    let mut rows: FxHashMap<u32, (Mask, Mask)> = FxHashMap::default();
    let mut ker = vec![];
    for (i, &img) in images.iter().enumerate() {
        let (mut r, mut src) = (img, 1 << i);
        loop {
            if r == 0 {
                ker.push(src);
                break;
            }
            let p = r.trailing_zeros();
            match rows.get(&p) {
                Some(&(row, s)) => {
                    r ^= row;
                    src ^= s;
                }
                None => {
                    rows.insert(p, (r, src));
                    break;
                }
            }
        }
    }
    ker
}

/// Kernel of a linear map whose values are sparse sets of keys (XOR of sorted key lists).
fn kernel_keys(images: &[Vec<u64>]) -> Vec<Mask> {
    let xor = |a: &[u64], b: &[u64]| crate::xor_sorted(a, b);
    let mut rows: Vec<(Vec<u64>, Mask)> = vec![];
    let mut ker = vec![];
    for (i, img) in images.iter().enumerate() {
        let (mut r, mut src) = (img.clone(), 1 << i);
        loop {
            let Some(&lead) = r.first() else {
                ker.push(src);
                break;
            };
            match rows.iter().find(|(row, _)| row[0] == lead) {
                Some((row, s)) => {
                    r = xor(&r, row);
                    src ^= s;
                }
                None => {
                    rows.push((r, src));
                    break;
                }
            }
        }
    }
    ker
}

/// Reads d2 files into global basis id -> target mask. Each file is either
/// `stem,Adams_filtration,index,target` or a `--solution` file
/// (`stem,Adams_filtration,index,status,target`), whose open classes are skipped. Earlier files
/// take precedence.
fn load_d2(ext: &Ext, paths: &str) -> FxHashMap<u32, Mask> {
    let mut d2 = FxHashMap::default();
    for path in paths.split(',') {
        let (mut read, mut open) = (0, 0);
        for line in BufReader::new(File::open(path).unwrap()).lines().skip(1) {
            let line = line.unwrap();
            let mut it = line.splitn(4, ',');
            let n: i32 = it.next().unwrap().parse().unwrap();
            let f: i32 = it.next().unwrap().parse().unwrap();
            let i: usize = it.next().unwrap().parse().unwrap();
            let mut rest = it.next().unwrap();
            if let Some(r) = rest.strip_prefix("determined,") {
                rest = r;
            } else if rest.starts_with("open,") {
                open += 1;
                continue;
            }
            if let Some(b) = ext.get(n, f) {
                d2.entry(ext.gid(b, i)).or_insert_with(|| parse_vec(rest));
                read += 1;
            }
        }
        eprintln!("d2 from {path}: {read} classes, {open} open skipped");
    }
    d2
}

/// Product of two E2 vectors, as a mask in the bidegree `a + c` (0 if that bidegree is empty).
fn mul_vec(ext: &Ext, p: &Products, a: u32, u: Mask, c: u32, w: Mask) -> Mask {
    let mut out = 0;
    for i in (0..ext.dim[a as usize]).filter(|&i| u >> i & 1 == 1) {
        for j in (0..ext.dim[c as usize]).filter(|&j| w >> j & 1 == 1) {
            out ^= p.mul(ext, a, i, c, j);
        }
    }
    out
}

/// The page E_r over the loaded range, with the unknowns d_r.
struct E3 {
    r: i32,
    /// bidegree id -> E_r there (`None`: unknown)
    page: Vec<Option<Page>>,
    /// bidegree id -> first variable of d_r on it, `u32::MAX` if none
    off: Vec<u32>,
    nvars: u32,
}

impl E3 {
    /// Number of source classes.
    fn dim(&self, b: u32) -> usize {
        self.page[b as usize].as_ref().map_or(0, |pg| pg.reps.len())
    }
    /// Number of target coordinates.
    fn tdim(&self, b: u32) -> usize {
        self.page[b as usize]
            .as_ref()
            .map_or(0, |pg| pg.treps.len())
    }
}

/// The bidegree `(n, f)`: `Some(Some(b))` if E_r is known there, `Some(None)` if Ext is zero
/// there, `None` if E_r there is unknown.
fn known(ext: &Ext, e3: &E3, n: i32, f: i32) -> Option<Option<u32>> {
    match ext.get(n, f) {
        Some(b) => e3.page[b as usize].as_ref().map(|_| Some(b)),
        None if ext.in_range(n, f) || f < 0 || n < 0 => Some(None),
        None => None,
    }
}

fn build_e3(ext: &Ext, d2: &FxHashMap<u32, Mask>) -> E3 {
    let nb = ext.deg.len();
    let mut page: Vec<Option<Page>> = (0..nb).map(|_| None).collect();
    let mut cycles_missing = 0;
    for b in 0..nb as u32 {
        let (n, f) = ext.deg[b as usize];
        // Z3 needs d2 out of b, B3 needs d2 into b: both bidegrees must be loaded (or have a
        // negative degree, where Ext is zero).
        let loaded = |n: i32, f: i32| n < 0 || f < 0 || ext.in_range(n, f);
        if !loaded(n - 1, f + 2) || !loaded(n + 1, f - 2) {
            continue;
        }
        let d2_of = |g: u32| d2.get(&g).copied();
        let out_known = ext.prime(b).is_none()
            || (0..ext.dim[b as usize]).all(|i| d2_of(ext.gid(b, i)).is_some());
        let into = ext.get(n + 1, f - 2);
        let in_known =
            into.is_none_or(|a| (0..ext.dim[a as usize]).all(|i| d2_of(ext.gid(a, i)).is_some()));
        if !out_known || !in_known {
            cycles_missing += 1;
            continue;
        }
        let images: Vec<Mask> = (0..ext.dim[b as usize])
            .map(|i| d2_of(ext.gid(b, i)).unwrap_or(0))
            .collect();
        let bnd: Vec<Mask> = into
            .map(|a| {
                (0..ext.dim[a as usize])
                    .map(|i| d2_of(ext.gid(a, i)).unwrap())
                    .collect()
            })
            .unwrap_or_default();
        let z = kernel(&images);
        page[b as usize] = Some(Page::new(bnd.clone(), bnd, &z, &z, true, false));
    }
    let mut e3 = E3 {
        r: 3,
        page,
        off: vec![u32::MAX; nb],
        nvars: 0,
    };
    eprintln!("E3: {cycles_missing} bidegrees without d2 data");
    assign_vars(ext, &mut e3);
    e3
}

/// Numbers the unknowns of d_r, by decreasing source stem.
fn assign_vars(ext: &Ext, e: &mut E3) {
    let nb = ext.deg.len();
    let mut order: Vec<u32> = (0..nb as u32).filter(|&b| e.dim(b) > 0).collect();
    order.sort_by_key(|&b| std::cmp::Reverse(ext.deg[b as usize]));
    for b in order {
        if let Some(Some(t)) = target(ext, e, b) {
            let m = e.tdim(t);
            if m > 0 {
                e.off[b as usize] = e.nvars;
                e.nvars += (e.dim(b) * m) as u32;
            }
        }
    }
    let pages = || e.page.iter().flatten();
    eprintln!(
        "E{}: {} bidegrees known ({} exact, {} with uncertain cycles), {} source classes, {} \
         target coordinates, {} d{} variables",
        e.r,
        pages().count(),
        pages().filter(|p| p.exact).count(),
        pages().filter(|p| p.uncertain).count(),
        pages().map(|p| p.reps.len()).sum::<usize>(),
        pages().map(|p| p.treps.len()).sum::<usize>(),
        e.nvars,
        e.r
    );
}

/// What is known of d_r on one source class: its value at the particular solution and, for
/// each free direction it depends on, how it moves (both in target coordinates).
#[derive(Clone, Default)]
struct Val {
    part: Mask,
    dirs: Vec<Mask>,
    /// sparse key form of (part, dirs) for kernels of combinations
    keys: Vec<u64>,
}

/// Bounds on E_{r+1} from those on E_r and what is known of d_r.
fn next_page(ext: &Ext, cur: &E3, vals: &FxHashMap<u32, Vec<Val>>) -> E3 {
    let nb = ext.deg.len();
    let r = cur.r;
    let as_e2 = |reps: &[Mask], coords: Mask| -> Mask {
        (0..reps.len())
            .filter(|&l| coords >> l & 1 == 1)
            .fold(0, |acc, l| acc ^ reps[l])
    };
    let page: Vec<Option<Page>> = (0..nb as u32)
        .map(|b| {
            let pg = cur.page[b as usize].as_ref()?;
            let (n, f) = ext.deg[b as usize];
            // certain cycles: combinations whose d_r is forced to vanish in the true E_r
            let (cycles, open_here): (Vec<Mask>, bool) = match target(ext, cur, b) {
                None => (vec![], !pg.reps.is_empty()),
                Some(None) => (pg.reps.clone(), false),
                Some(Some(t)) if cur.tdim(t) == 0 => (pg.reps.clone(), false),
                Some(Some(t)) => {
                    let vs = vals.get(&b).map_or(&[][..], |v| &v[..]);
                    let any_open = vs.iter().any(|v| !v.dirs.is_empty());
                    if cur.page[t as usize].as_ref().unwrap().exact {
                        let keys: Vec<Vec<u64>> = vs.iter().map(|v| v.keys.clone()).collect();
                        let z = kernel_keys(&keys)
                            .into_iter()
                            .map(|c| as_e2(&pg.reps, c))
                            .collect();
                        (z, any_open)
                    } else {
                        (vec![], !pg.reps.is_empty())
                    }
                }
            };
            // boundaries created by d_r into b
            let mut bcert = pg.bcert.clone();
            let mut bmax = pg.bmax.clone();
            let mut exact = pg.exact;
            match known(ext, cur, n + 1, f - r) {
                None => {
                    bmax.extend(&pg.treps);
                    exact = false;
                }
                Some(None) => {}
                Some(Some(a)) => {
                    let pa = cur.page[a as usize].as_ref().unwrap();
                    if pa.uncertain {
                        bmax.extend(&pg.treps);
                        exact = false;
                    }
                    if let Some(vs) = vals.get(&a) {
                        for v in vs {
                            let img = as_e2(&pg.treps, v.part);
                            bmax.push(img);
                            if v.dirs.is_empty() && pg.exact {
                                bcert.push(img);
                            } else {
                                exact = false;
                            }
                            for &d in &v.dirs {
                                bmax.push(as_e2(&pg.treps, d));
                            }
                        }
                    }
                }
            }
            Some(Page::new(
                bcert,
                bmax,
                &cycles,
                &pg.treps,
                exact,
                pg.uncertain || open_here,
            ))
        })
        .collect();
    let mut e = E3 {
        r: r + 1,
        page,
        off: vec![u32::MAX; nb],
        nvars: 0,
    };
    assign_vars(ext, &mut e);
    e
}

/// The d_r target bidegree of `b`, if E_r is known there (`Some(None)`: zero).
fn target(ext: &Ext, e3: &E3, b: u32) -> Option<Option<u32>> {
    let (n, f) = ext.deg[b as usize];
    known(ext, e3, n - 1, f + e3.r)
}

fn var(ext: &Ext, e3: &E3, b: u32, k: usize, l: usize) -> u32 {
    let t = target(ext, e3, b).unwrap().unwrap();
    e3.off[b as usize] + (k * e3.tdim(t) + l) as u32
}

/// Leibniz on E_r for every pair of source classes: d_r[xy] = d_r[x][y] + [x]d_r[y], projected
/// onto the target bounds. A term d_r[x][y] is only well defined when the target of x has no
/// boundary uncertainty, so pairs without that are skipped.
fn equations(ext: &Ext, p: &Products, e3: &E3) -> Vec<Vec<u32>> {
    let t = Instant::now();
    let ablate_zero = std::env::var("E3_ABLATE_ZERO").is_ok_and(|v| v == "1");
    let bids: Vec<u32> = (0..ext.deg.len() as u32)
        .filter(|&b| e3.dim(b) > 0)
        .collect();
    let exact_or_zero =
        |t: Option<u32>| t.is_none_or(|t| e3.page[t as usize].as_ref().unwrap().exact);
    let eqs: Vec<Vec<u32>> = bids
        .par_iter()
        .flat_map_iter(|&a| {
            let mut out = vec![];
            let (an, af) = ext.deg[a as usize];
            for &c in bids.iter().filter(|&&c| c >= a) {
                let (cn, cf) = ext.deg[c as usize];
                let Some(Some(sp)) = known(ext, e3, an + cn - 1, af + cf + e3.r) else {
                    continue;
                };
                let Some(s) = known(ext, e3, an + cn, af + cf) else {
                    continue;
                };
                let (Some(ap), Some(cp)) = (target(ext, e3, a), target(ext, e3, c)) else {
                    continue;
                };
                if !exact_or_zero(ap) || !exact_or_zero(cp) {
                    continue;
                }
                let m = e3.tdim(sp);
                if m == 0 {
                    continue;
                }
                let pg_a = e3.page[a as usize].as_ref().unwrap();
                let pg_c = e3.page[c as usize].as_ref().unwrap();
                let pg_sp = e3.page[sp as usize].as_ref().unwrap();
                'pair: for (k, &x) in pg_a.reps.iter().enumerate() {
                    let j0 = if a == c { k } else { 0 };
                    for (j, &y) in pg_c.reps.iter().enumerate().skip(j0) {
                        // per target coordinate, the variables of the equation
                        let mut terms: Vec<Vec<u32>> = vec![vec![]; m];
                        // the product as a combination of source classes, if it is certain
                        let pr = match s {
                            None => 0,
                            Some(s) => match e3.page[s as usize]
                                .as_ref()
                                .unwrap()
                                .src
                                .project(mul_vec(ext, p, a, x, c, y))
                            {
                                Some(pr) => pr,
                                None => continue,
                            },
                        };
                        // E3_ABLATE_ZERO=1 drops the pairs whose product is 0 in E_r: an ablation
                        if pr == 0 && ablate_zero {
                            continue;
                        }
                        if let Some(s) = s {
                            for q in (0..e3.dim(s)).filter(|&q| pr >> q & 1 == 1) {
                                for (r, tr) in terms.iter_mut().enumerate() {
                                    tr.push(var(ext, e3, s, q, r));
                                }
                            }
                        }
                        if let Some(ap) = ap {
                            let pg = e3.page[ap as usize].as_ref().unwrap();
                            for (l, &z) in pg.treps.iter().enumerate() {
                                let Some(q) = pg_sp.tgt.project(mul_vec(ext, p, ap, z, c, y))
                                else {
                                    continue 'pair;
                                };
                                for r in (0..m).filter(|&r| q >> r & 1 == 1) {
                                    terms[r].push(var(ext, e3, a, k, l));
                                }
                            }
                        }
                        if let Some(cp) = cp {
                            let pg = e3.page[cp as usize].as_ref().unwrap();
                            for (l, &z) in pg.treps.iter().enumerate() {
                                let Some(q) = pg_sp.tgt.project(mul_vec(ext, p, a, x, cp, z))
                                else {
                                    continue 'pair;
                                };
                                for r in (0..m).filter(|&r| q >> r & 1 == 1) {
                                    terms[r].push(var(ext, e3, c, j, l));
                                }
                            }
                        }
                        for mut tr in terms {
                            tr.sort_unstable();
                            let mut e: Vec<u32> = vec![];
                            for v in tr {
                                if e.last() == Some(&v) {
                                    e.pop();
                                } else {
                                    e.push(v);
                                }
                            }
                            if !e.is_empty() {
                                out.push(e);
                            }
                        }
                    }
                }
            }
            out
        })
        .collect();
    eprintln!(
        "E{} Leibniz: {} nontrivial equations, {:.1}s",
        e3.r,
        eqs.len(),
        t.elapsed().as_secs_f64()
    );
    eqs
}

/// A statement read from a file: E2 vectors of a source and its d_r target.
struct Statement {
    n: i32,
    f: i32,
    src: Mask,
    tgt: Mask,
    label: String,
}

/// Reads `stem,Adams_filtration,source,target[,label]` lines (vectors as `[0,1,...]` over the
/// E2 basis; the target lives in `(stem - 1, filtration + r)`).
fn read_statements(path: &str) -> Vec<Statement> {
    let mut out = vec![];
    for line in BufReader::new(File::open(path).unwrap()).lines().skip(1) {
        let line = line.unwrap();
        let mut it = line.splitn(3, ',');
        let n: i32 = it.next().unwrap().parse().unwrap();
        let f: i32 = it.next().unwrap().parse().unwrap();
        let rest = it.next().unwrap();
        let mut vecs = rest.split('"').filter(|s| s.contains('['));
        let src = parse_vec(vecs.next().unwrap());
        let tgt = parse_vec(vecs.next().unwrap_or("[]"));
        let label = rest.rsplit(',').next().unwrap_or("").to_string();
        out.push(Statement {
            n,
            f,
            src,
            tgt,
            label,
        });
    }
    out
}

/// The equations `d_r[src] = [tgt]` (affine: the constant column is `nvars`), projected onto the
/// target bounds.
fn statement_equations(ext: &Ext, e3: &E3, st: &Statement) -> Result<Vec<Vec<u32>>, String> {
    let b = ext.get(st.n, st.f).ok_or("source bidegree empty")?;
    let pg = e3.page[b as usize]
        .as_ref()
        .ok_or("page unknown at source")?;
    let s = pg
        .src
        .project(st.src)
        .ok_or("source is not a certain cycle on this page")?;
    let tb = match target(ext, e3, b).ok_or("page unknown at target")? {
        Some(t) => t,
        None => return Ok(vec![]),
    };
    let tpg = e3.page[tb as usize].as_ref().unwrap();
    let tc = tpg
        .tgt
        .project(st.tgt)
        .ok_or("target is not a cycle on this page")?;
    let m = e3.tdim(tb);
    if s == 0 {
        return Err(if tc == 0 {
            "source is a boundary on this page".into()
        } else {
            "source is a boundary but target is not".into()
        });
    }
    if e3.off[b as usize] == u32::MAX {
        return Ok(vec![]);
    }
    Ok((0..m)
        .map(|r| {
            let mut e: Vec<u32> = (0..e3.dim(b))
                .filter(|&k| s >> k & 1 == 1)
                .map(|k| var(ext, e3, b, k, r))
                .collect();
            if tc >> r & 1 == 1 {
                e.push(e3.nvars);
            }
            e
        })
        .collect())
}

/// Evaluates linear combinations of variables on the solution space: the value at the
/// particular solution, and the free columns (by index among the free columns) it depends on.
fn evaluate(
    nvars: u32,
    pivot: &[u32],
    rows: &[Vec<u32>],
    queries: &[Vec<u32>],
) -> Vec<(bool, Vec<u32>)> {
    let pivcols: Vec<u32> = (0..nvars)
        .rev()
        .filter(|&c| pivot[c as usize] != u32::MAX)
        .collect();
    let back = |val: &mut [u64]| {
        for &pc in &pivcols {
            let r = &rows[pivot[pc as usize] as usize];
            val[pc as usize] = r[1..].iter().fold(0, |acc, &c| acc ^ val[c as usize]);
        }
    };
    let n = nvars as usize;
    let mut part = vec![0u64; n + 1];
    part[n] = 1;
    back(&mut part);
    let free: Vec<u32> = (0..nvars)
        .filter(|&c| pivot[c as usize] == u32::MAX)
        .collect();
    let deps: Vec<(usize, u32)> = free
        .par_chunks(64)
        .enumerate()
        .flat_map_iter(|(ci, chunk)| {
            let mut val = vec![0u64; n + 1];
            for (l, &f) in chunk.iter().enumerate() {
                val[f as usize] = 1 << l;
            }
            back(&mut val);
            let mut out = vec![];
            for (qi, q) in queries.iter().enumerate() {
                let mut m = q.iter().fold(0, |acc, &v| acc ^ val[v as usize]);
                while m != 0 {
                    out.push((qi, (ci * 64) as u32 + m.trailing_zeros()));
                    m &= m - 1;
                }
            }
            out
        })
        .collect();
    let mut res: Vec<(bool, Vec<u32>)> = queries
        .iter()
        .map(|q| {
            (
                q.iter().fold(0, |acc, &v| acc ^ part[v as usize]) & 1 == 1,
                vec![],
            )
        })
        .collect();
    for (qi, f) in deps {
        res[qi].1.push(f);
    }
    res
}

/// `(page, path)` pairs from repeatable `<flag> r:path` arguments, plus the legacy page-3 flag.
fn page_files(args: &[String], flag: &str, legacy: &str) -> Vec<(i32, String)> {
    let mut out = vec![];
    for (i, a) in args.iter().enumerate() {
        if a == flag {
            let (r, path) = args[i + 1].split_once(':').expect("expected r:path");
            out.push((r.parse().unwrap(), path.to_string()));
        } else if a == legacy {
            out.push((3, args[i + 1].clone()));
        }
    }
    out
}

pub(crate) fn run(ext: &Ext, p: &Products, args: &[String]) {
    let arg = |flag: &str| args.iter().position(|a| a == flag).map(|i| &args[i + 1]);
    let last: i32 = arg("--pages").map_or(3, |s| s.parse().unwrap());
    let seeds = page_files(args, "--seed", "--seed-d3");
    let check_files = page_files(args, "--check", "--check-d3");
    let d2 = load_d2(ext, arg("--e3").unwrap());
    // Facts d_r(y) = 0 learned from page r + 1 (page -> (stem, filtration, E2 vector of y)).
    // They are kept as E2 vectors because the page bases change from one pass to the next.
    let mut back: FxHashMap<i32, std::collections::BTreeSet<(i32, i32, Mask)>> =
        FxHashMap::default();
    for pass in 1.. {
        let facts_before: usize = back.values().map(|s| s.len()).sum();
        println!("pass {pass}");
        let mut e = build_e3(ext, &d2);
        loop {
            let r = e.r;
            let mut eqs = equations(ext, p, &e);
            let mut n_back = 0;
            for &(n, f, src) in back.get(&r).into_iter().flatten() {
                let st = Statement {
                    n,
                    f,
                    src,
                    tgt: 0,
                    label: String::new(),
                };
                if let Ok(se) = statement_equations(ext, &e, &st) {
                    eqs.extend(se);
                    n_back += 1;
                }
            }
            if n_back > 0 {
                eprintln!("d{r}: {n_back} backward facts d{r}(y) = 0 used");
            }
            for (_, path) in seeds.iter().filter(|(sr, _)| *sr == r) {
                for st in read_statements(path) {
                    // a seed d_r(src) = tgt also says that src and tgt survive to E_r
                    if r > 3 {
                        let set = back.entry(r - 1).or_default();
                        set.insert((st.n, st.f, st.src));
                        if st.tgt != 0 {
                            set.insert((st.n - 1, st.f + r, st.tgt));
                        }
                    }
                    match statement_equations(ext, &e, &st) {
                        Ok(se) => {
                            eprintln!(
                                "seed d{r} at ({},{}) {}: {} equations",
                                st.n,
                                st.f,
                                st.label,
                                se.len()
                            );
                            eqs.extend(se);
                        }
                        Err(err) => {
                            eprintln!(
                                "seed d{r} at ({},{}) {} skipped: {err}",
                                st.n, st.f, st.label
                            )
                        }
                    }
                }
            }
            let (pivot, rows, _, contradictions) = eliminate(e.nvars, &eqs, false);
            let free = pivot.iter().filter(|&&x| x == u32::MAX).count();
            eprintln!(
                "d{r} solution space: dimension {free}, {} contradictions",
                contradictions.len()
            );

            // per source class with a known target: what is known of d_r
            let mut classes: Vec<(u32, usize)> = vec![];
            let mut queries: Vec<Vec<u32>> = vec![];
            let mut bids: Vec<u32> = (0..ext.deg.len() as u32)
                .filter(|&b| e.off[b as usize] != u32::MAX)
                .collect();
            bids.sort_by_key(|&b| ext.deg[b as usize]);
            for &b in &bids {
                let m = e.tdim(target(ext, &e, b).unwrap().unwrap());
                for k in 0..e.dim(b) {
                    classes.push((b, k));
                    for t in 0..m {
                        queries.push(vec![var(ext, &e, b, k, t)]);
                    }
                }
            }
            let n_class_queries = queries.len();
            let checks: Vec<(Statement, Result<Vec<Vec<u32>>, String>)> = check_files
                .iter()
                .filter(|(cr, _)| *cr == r)
                .flat_map(|(_, path)| read_statements(path))
                .map(|st| {
                    let res = statement_equations(ext, &e, &st);
                    (st, res)
                })
                .collect();
            // a check holds iff each of its affine equations evaluates to its constant
            for (_, res) in &checks {
                if let Ok(se) = res {
                    for eq in se {
                        queries.push(eq.iter().copied().filter(|&v| v != e.nvars).collect());
                    }
                }
            }
            let vals = evaluate(e.nvars, &pivot, &rows, &queries);

            let mut summary: BTreeMap<i32, (usize, usize, usize)> = BTreeMap::new();
            let out_path = if r == 3 {
                arg("--solution3").cloned()
            } else {
                None
            }
            .or_else(|| arg("--solution-pages").map(|pre| format!("{pre}_d{r}.csv")));
            let mut w = out_path.map(|p| BufWriter::new(File::create(p).unwrap()));
            if let Some(w) = w.as_mut() {
                writeln!(w, "stem,Adams_filtration,rep,status,target").unwrap();
            }
            let mut dr: FxHashMap<u32, Vec<Val>> = FxHashMap::default();
            let mut qi = 0;
            for &(b, k) in &classes {
                let m = e.tdim(target(ext, &e, b).unwrap().unwrap());
                let vs = &vals[qi..qi + m];
                qi += m;
                let mut v = Val::default();
                let mut dirs: BTreeMap<u32, Mask> = BTreeMap::new();
                for (t, (val, deps)) in vs.iter().enumerate() {
                    if *val {
                        v.part |= 1 << t;
                        v.keys.push(t as u64);
                    }
                    for &f in deps {
                        *dirs.entry(f).or_default() |= 1 << t;
                        v.keys.push(64 + (f as u64) * 64 + t as u64);
                    }
                }
                v.keys.sort_unstable();
                v.dirs = dirs.into_values().collect();
                let det = v.dirs.is_empty();
                let (n, f) = ext.deg[b as usize];
                let s = summary.entry(n).or_default();
                s.0 += 1;
                if det {
                    s.1 += 1;
                    if v.part != 0 {
                        s.2 += 1;
                    }
                }
                if let Some(w) = w.as_mut() {
                    let rep = e.page[b as usize].as_ref().unwrap().reps[k];
                    let tgt: Vec<String> = vs.iter().map(|(v, _)| (*v as u8).to_string()).collect();
                    writeln!(
                        w,
                        "{n},{f},\"{rep:#b}\",{},\"[{}]\"",
                        if det { "determined" } else { "open" },
                        tgt.join(",")
                    )
                    .unwrap();
                }
                dr.entry(b).or_default().push(v);
            }
            // a forced value d_r(x) = y in a target without boundary uncertainty is a true
            // d_{r-1}-cycle: d_{r-1}(y) = 0
            if r > 3 {
                for (&b, vs) in &dr {
                    let Some(Some(t)) = target(ext, &e, b) else {
                        continue;
                    };
                    let tp = e.page[t as usize].as_ref().unwrap();
                    if !tp.exact {
                        continue;
                    }
                    let (tn, tf) = ext.deg[t as usize];
                    for v in vs.iter().filter(|v| v.dirs.is_empty() && v.part != 0) {
                        let y = (0..tp.treps.len())
                            .filter(|&l| v.part >> l & 1 == 1)
                            .fold(0, |acc, l| acc ^ tp.treps[l]);
                        back.entry(r - 1).or_default().insert((tn, tf, y));
                    }
                }
            }
            assert_eq!(qi, n_class_queries);
            println!("page {r}");
            println!("stem,E{r}_classes_with_target,determined,determined_nonzero");
            let (mut tc, mut td, mut tz) = (0, 0, 0);
            for (s, (c, d, z)) in &summary {
                println!("{s},{c},{d},{z}");
                tc += c;
                td += d;
                tz += z;
            }
            println!(
                "total d{r}: {tc} classes, {td} determined ({tz} nonzero), {} open",
                tc - td
            );

            if !checks.is_empty() {
                let (mut ok, mut bad, mut open, mut skipped, mut exact_ok) = (0, 0, 0, 0, 0);
                for (st, res) in &checks {
                    match res {
                        Err(err) => {
                            skipped += 1;
                            println!(
                                "check d{r} ({},{}) {}: skipped ({err})",
                                st.n, st.f, st.label
                            );
                        }
                        Ok(se) => {
                            let vs = &vals[qi..qi + se.len()];
                            qi += se.len();
                            let holds = se.iter().zip(vs).map(|(eq, (v, deps))| {
                                let c = eq.contains(&e.nvars);
                                (deps.is_empty(), *v == c)
                            });
                            let all_det = holds.clone().all(|(d, _)| d);
                            let all_ok = holds.clone().all(|(d, h)| !d || h);
                            let status = if !all_ok {
                                bad += 1;
                                "WRONG"
                            } else if all_det {
                                ok += 1;
                                "correct"
                            } else {
                                open += 1;
                                "open"
                            };
                            // exact: the target page has no boundary uncertainty, so the verdict is
                            // about the true E_r rather than its quotient by the possible boundaries
                            let exact = ext
                                .get(st.n, st.f)
                                .and_then(|b| target(ext, &e, b))
                                .flatten()
                                .is_none_or(|t| e.page[t as usize].as_ref().unwrap().exact);
                            if status == "correct" && exact {
                                exact_ok += 1;
                            }
                            let how = if exact {
                                "exact"
                            } else {
                                "modulo possible boundaries"
                            };
                            println!(
                                "check d{r} ({},{}) {}: {status} ({how})",
                                st.n, st.f, st.label
                            );
                        }
                    }
                }
                println!(
                    "checks d{r}: {ok} correct and forced ({exact_ok} exact), {bad} wrong, {open} \
                 open, {skipped} skipped"
                );
            }
            if r >= last {
                break;
            }
            e = next_page(ext, &e, &dr);
        }
        let facts: usize = back.values().map(|s| s.len()).sum();
        eprintln!(
            "pass {pass}: {facts} backward facts ({} new)",
            facts - facts_before
        );
        if facts == facts_before {
            break;
        }
    }
}
