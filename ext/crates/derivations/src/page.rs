//! Adams d_r on E_r = H(E_{r-1}, d_{r-1}) for r >= 3, constrained by the Leibniz rule on E_r
//! (`--e3`).

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{Ext, Mask, Products, eliminate, parse_vec};

/// E_r at one bidegree: representatives of a basis of Z_r/B_r, and an echelon form of
/// `B_r + span(reps)` that projects any element of Z_r onto those representatives.
#[derive(Default)]
struct Page {
    /// E2 masks of the E_r basis representatives
    reps: Vec<Mask>,
    /// E2 masks spanning B_r
    bnd: Vec<Mask>,
    /// lowest set bit -> (row, its expression in the representatives modulo B3)
    rows: FxHashMap<u32, (Mask, Mask)>,
}

impl Page {
    /// Reduces `v`; returns the remainder and the representative coefficients used.
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

    /// The E3 coordinates of a d2-cycle, or `None` if `v` is not one.
    fn project(&self, v: Mask) -> Option<Mask> {
        let (rest, coef) = self.reduce(v);
        (rest == 0).then_some(coef)
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
    /// bidegree id -> E_r there (`None`: not determined by the earlier differentials)
    page: Vec<Option<Page>>,
    /// bidegree id -> first variable of d_r on it, `u32::MAX` if none
    off: Vec<u32>,
    nvars: u32,
}

impl E3 {
    fn dim(&self, b: u32) -> usize {
        self.page[b as usize].as_ref().map_or(0, |pg| pg.reps.len())
    }
}

/// The bidegree `(n, f)`: `Some(Some(b))` if E3 is known there, `Some(None)` if Ext is zero
/// there (so E3 is too), `None` if E3 there is unknown.
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
        let mut pg = Page::default();
        if let Some(a) = into {
            for i in 0..ext.dim[a as usize] {
                let v = d2_of(ext.gid(a, i)).unwrap();
                pg.insert(v, 0);
                pg.bnd.push(v);
            }
        }
        for z in kernel(&images) {
            let k = pg.reps.len();
            if pg.insert(z, 1 << k) {
                pg.reps.push(z);
            }
        }
        page[b as usize] = Some(pg);
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
            let m = e.dim(t);
            if m > 0 {
                e.off[b as usize] = e.nvars;
                e.nvars += (e.dim(b) * m) as u32;
            }
        }
    }
    let total: usize = (0..nb as u32).map(|b| e.dim(b)).sum();
    eprintln!(
        "E{}: {} bidegrees known, dim {total}, {} d{} variables",
        e.r,
        e.page.iter().filter(|p| p.is_some()).count(),
        e.nvars,
        e.r
    );
}

/// E_{r+1} wherever d_r is forced both out of and into the bidegree. `dr[b][k]` is the forced
/// d_r of the k-th E_r class at `b`, in E_r coordinates of the target (`None`: open).
fn next_page(ext: &Ext, cur: &E3, dr: &FxHashMap<u32, Vec<Option<Mask>>>) -> E3 {
    let nb = ext.deg.len();
    let r = cur.r;
    // d_r on all of E_r at b, if forced (zero when the target is zero)
    let images = |b: u32| -> Option<Vec<Mask>> {
        match target(ext, cur, b)? {
            Some(t) if cur.dim(t) > 0 => dr.get(&b)?.iter().copied().collect(),
            _ => Some(vec![0; cur.dim(b)]),
        }
    };
    let as_e2 = |b: u32, coords: Mask| -> Mask {
        let reps = &cur.page[b as usize].as_ref().unwrap().reps;
        (0..reps.len())
            .filter(|&l| coords >> l & 1 == 1)
            .fold(0, |acc, l| acc ^ reps[l])
    };
    let page: Vec<Option<Page>> = (0..nb as u32)
        .map(|b| {
            let pg = cur.page[b as usize].as_ref()?;
            let out = images(b)?;
            let (n, f) = ext.deg[b as usize];
            let src = known(ext, cur, n + 1, f - r)?;
            let into = match src {
                Some(a) => images(a)?,
                None => vec![],
            };
            let mut next = Page::default();
            for &v in &pg.bnd {
                next.insert(v, 0);
                next.bnd.push(v);
            }
            for &img in &into {
                let v = as_e2(b, img);
                next.insert(v, 0);
                next.bnd.push(v);
            }
            for z in kernel(&out) {
                let v = as_e2(b, z);
                let k = next.reps.len();
                if next.insert(v, 1 << k) {
                    next.reps.push(v);
                }
            }
            Some(next)
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
    e3.off[b as usize] + (k * e3.dim(t) + l) as u32
}

/// Leibniz on E3 for every pair of E3 basis classes: d3[xy] = d3[x][y] + [x]d3[y].
fn equations(ext: &Ext, p: &Products, e3: &E3) -> Vec<Vec<u32>> {
    let t = Instant::now();
    let ablate_zero = std::env::var("E3_ABLATE_ZERO").is_ok_and(|v| v == "1");
    let bids: Vec<u32> = (0..ext.deg.len() as u32)
        .filter(|&b| e3.dim(b) > 0)
        .collect();
    let eqs: Vec<Vec<u32>> = bids
        .par_iter()
        .flat_map_iter(|&a| {
            let mut out = vec![];
            let (an, af) = ext.deg[a as usize];
            for &c in bids.iter().filter(|&&c| c >= a) {
                let (cn, cf) = ext.deg[c as usize];
                // every bidegree involved must have known E3 (or be zero)
                let Some(Some(sp)) = known(ext, e3, an + cn - 1, af + cf + e3.r) else {
                    continue;
                };
                let Some(s) = known(ext, e3, an + cn, af + cf) else {
                    continue;
                };
                let (Some(ap), Some(cp)) = (target(ext, e3, a), target(ext, e3, c)) else {
                    continue;
                };
                let m = e3.dim(sp);
                if m == 0 {
                    continue;
                }
                let pg_a = e3.page[a as usize].as_ref().unwrap();
                let pg_c = e3.page[c as usize].as_ref().unwrap();
                let pg_sp = e3.page[sp as usize].as_ref().unwrap();
                let proj = |pg: &Page, v: Mask| {
                    pg.project(v).unwrap_or_else(|| {
                        panic!(
                            "a product of E_{} cycles is not a cycle: the earlier differentials \
                             are not Leibniz",
                            e3.r
                        )
                    })
                };
                for (k, &x) in pg_a.reps.iter().enumerate() {
                    let j0 = if a == c { k } else { 0 };
                    for (j, &y) in pg_c.reps.iter().enumerate().skip(j0) {
                        // per target coordinate r, the variables of the equation
                        let mut terms: Vec<Vec<u32>> = vec![vec![]; m];
                        // E3_ABLATE_ZERO=1 drops the pairs whose product is 0 in E3: an ablation
                        let pr = s.map_or(0, |s| {
                            proj(
                                e3.page[s as usize].as_ref().unwrap(),
                                mul_vec(ext, p, a, x, c, y),
                            )
                        });
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
                            for (l, &z) in e3.page[ap as usize]
                                .as_ref()
                                .unwrap()
                                .reps
                                .iter()
                                .enumerate()
                            {
                                let q = proj(pg_sp, mul_vec(ext, p, ap, z, c, y));
                                for r in (0..m).filter(|&r| q >> r & 1 == 1) {
                                    terms[r].push(var(ext, e3, a, k, l));
                                }
                            }
                        }
                        if let Some(cp) = cp {
                            for (l, &z) in e3.page[cp as usize]
                                .as_ref()
                                .unwrap()
                                .reps
                                .iter()
                                .enumerate()
                            {
                                let q = proj(pg_sp, mul_vec(ext, p, a, x, cp, z));
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

/// A d3 statement read from a file: E2 vectors of a source and its d3 target.
struct Statement {
    n: i32,
    f: i32,
    src: Mask,
    tgt: Mask,
    label: String,
}

/// Reads `stem,Adams_filtration,source,target[,label]` lines (vectors as `[0,1,...]` over the
/// E2 basis; the target lives in `(stem - 1, filtration + 3)`).
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

/// The equations `d_r[src] = [tgt]` (affine: the constant column is `nvars`).
fn statement_equations(ext: &Ext, e3: &E3, st: &Statement) -> Result<Vec<Vec<u32>>, String> {
    let b = ext.get(st.n, st.f).ok_or("source bidegree empty")?;
    let pg = e3.page[b as usize].as_ref().ok_or("E3 unknown at source")?;
    let s = pg
        .project(st.src)
        .ok_or("source is not a cycle on this page")?;
    let tb = match target(ext, e3, b).ok_or("E3 unknown at target")? {
        Some(t) => t,
        None => return Ok(vec![]),
    };
    let tpg = e3.page[tb as usize].as_ref().unwrap();
    let tc = tpg
        .project(st.tgt)
        .ok_or("target is not a cycle on this page")?;
    let m = e3.dim(tb);
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
/// particular solution, and whether it is the same on every solution.
fn evaluate(
    nvars: u32,
    pivot: &[u32],
    rows: &[Vec<u32>],
    queries: &[Vec<u32>],
) -> Vec<(bool, bool)> {
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
    let open: Vec<bool> = free
        .par_chunks(64)
        .map(|chunk| {
            let mut val = vec![0u64; n + 1];
            for (l, &f) in chunk.iter().enumerate() {
                val[f as usize] = 1 << l;
            }
            back(&mut val);
            queries
                .iter()
                .map(|q| q.iter().fold(0, |acc, &v| acc ^ val[v as usize]) != 0)
                .collect::<Vec<bool>>()
        })
        .reduce(
            || vec![false; queries.len()],
            |a, b| a.iter().zip(&b).map(|(x, y)| x | y).collect(),
        );
    queries
        .iter()
        .zip(open)
        .map(|(q, o)| {
            (
                q.iter().fold(0, |acc, &v| acc ^ part[v as usize]) & 1 == 1,
                !o,
            )
        })
        .collect()
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
    let mut e = build_e3(ext, &d2);
    loop {
        let r = e.r;
        let mut eqs = equations(ext, p, &e);
        for (_, path) in seeds.iter().filter(|(sr, _)| *sr == r) {
            for st in read_statements(path) {
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

        // per E_r basis class with a known target: is d_r forced, and to what
        let mut classes: Vec<(u32, usize)> = vec![];
        let mut queries: Vec<Vec<u32>> = vec![];
        let mut bids: Vec<u32> = (0..ext.deg.len() as u32)
            .filter(|&b| e.off[b as usize] != u32::MAX)
            .collect();
        bids.sort_by_key(|&b| ext.deg[b as usize]);
        for &b in &bids {
            let m = e.dim(target(ext, &e, b).unwrap().unwrap());
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
        let mut dr: FxHashMap<u32, Vec<Option<Mask>>> = FxHashMap::default();
        let mut qi = 0;
        for &(b, k) in &classes {
            let m = e.dim(target(ext, &e, b).unwrap().unwrap());
            let vs = &vals[qi..qi + m];
            qi += m;
            let det = vs.iter().all(|&(_, d)| d);
            let value: Mask = vs
                .iter()
                .enumerate()
                .filter(|(_, (v, _))| *v)
                .fold(0, |acc, (t, _)| acc | 1 << t);
            dr.entry(b).or_default().push(det.then_some(value));
            let (n, f) = ext.deg[b as usize];
            let s = summary.entry(n).or_default();
            s.0 += 1;
            if det {
                s.1 += 1;
                if value != 0 {
                    s.2 += 1;
                }
            }
            if let Some(w) = w.as_mut() {
                let rep = e.page[b as usize].as_ref().unwrap().reps[k];
                let tgt: Vec<String> = vs.iter().map(|&(v, _)| (v as u8).to_string()).collect();
                writeln!(
                    w,
                    "{n},{f},\"{rep:#b}\",{},\"[{}]\"",
                    if det { "determined" } else { "open" },
                    tgt.join(",")
                )
                .unwrap();
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
            let (mut ok, mut bad, mut open, mut skipped) = (0, 0, 0, 0);
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
                        let holds = se.iter().zip(vs).map(|(eq, &(v, d))| {
                            let c = eq.contains(&e.nvars);
                            (d, v == c)
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
                        println!("check d{r} ({},{}) {}: {status}", st.n, st.f, st.label);
                    }
                }
            }
            println!(
                "checks d{r}: {ok} correct and forced, {bad} wrong, {open} open, {skipped} skipped"
            );
        }
        if r >= last {
            break;
        }
        e = next_page(ext, &e, &dr);
    }
}
