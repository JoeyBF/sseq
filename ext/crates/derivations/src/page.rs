//! Adams d3 on E3 = H(E2, d2), constrained by the Leibniz rule on E3 (`--e3`).

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{Ext, Mask, Products, eliminate, parse_vec};

/// E3 at one bidegree: representatives of a basis of Z3/B3, and an echelon form of
/// `B3 + span(reps)` that projects any d2-cycle onto those representatives.
#[derive(Default)]
struct Page {
    /// E2 masks of the E3 basis representatives
    reps: Vec<Mask>,
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

struct E3 {
    /// bidegree id -> E3 there (`None`: E3 not determined by the loaded d2)
    page: Vec<Option<Page>>,
    /// bidegree id -> first variable of d3 on it, `u32::MAX` if none
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
                pg.insert(d2_of(ext.gid(a, i)).unwrap(), 0);
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
        page,
        off: vec![u32::MAX; nb],
        nvars: 0,
    };
    // variables by decreasing source stem
    let mut order: Vec<u32> = (0..nb as u32).filter(|&b| e3.dim(b) > 0).collect();
    order.sort_by_key(|&b| std::cmp::Reverse(ext.deg[b as usize]));
    for b in order {
        let (n, f) = ext.deg[b as usize];
        if let Some(Some(t)) = known(ext, &e3, n - 1, f + 3) {
            let m = e3.dim(t);
            if m > 0 {
                e3.off[b as usize] = e3.nvars;
                e3.nvars += (e3.dim(b) * m) as u32;
            }
        }
    }
    let total: usize = (0..nb as u32).map(|b| e3.dim(b)).sum();
    eprintln!(
        "E3: {} bidegrees known, dim E3 {total}, {cycles_missing} bidegrees without d2 data, {} \
         d3 variables",
        e3.page.iter().filter(|p| p.is_some()).count(),
        e3.nvars
    );
    e3
}

/// The d3 target bidegree of `b`, if E3 is known there (`Some(None)`: zero).
fn target(ext: &Ext, e3: &E3, b: u32) -> Option<Option<u32>> {
    let (n, f) = ext.deg[b as usize];
    known(ext, e3, n - 1, f + 3)
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
                let Some(Some(sp)) = known(ext, e3, an + cn - 1, af + cf + 3) else {
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
                            "a product of d2-cycles is not a d2-cycle: the d2 data is not Leibniz"
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
        "E3 Leibniz: {} nontrivial equations, {:.1}s",
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

/// The equations `d3[src] = [tgt]` (affine: the constant column is `nvars`).
fn statement_equations(ext: &Ext, e3: &E3, st: &Statement) -> Result<Vec<Vec<u32>>, String> {
    let b = ext.get(st.n, st.f).ok_or("source bidegree empty")?;
    let pg = e3.page[b as usize].as_ref().ok_or("E3 unknown at source")?;
    let s = pg.project(st.src).ok_or("source is not a d2-cycle")?;
    let tb = match target(ext, e3, b).ok_or("E3 unknown at target")? {
        Some(t) => t,
        None => return Ok(vec![]),
    };
    let tpg = e3.page[tb as usize].as_ref().unwrap();
    let tc = tpg.project(st.tgt).ok_or("target is not a d2-cycle")?;
    let m = e3.dim(tb);
    if s == 0 {
        return Err(if tc == 0 {
            "source is a d2-boundary".into()
        } else {
            "source is a d2-boundary but target is not".into()
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

pub(crate) fn run(ext: &Ext, p: &Products, args: &[String]) {
    let arg = |flag: &str| args.iter().position(|a| a == flag).map(|i| &args[i + 1]);
    let d2 = load_d2(ext, arg("--e3").unwrap());
    let e3 = build_e3(ext, &d2);
    let mut eqs = equations(ext, p, &e3);
    if let Some(path) = arg("--seed-d3") {
        for st in read_statements(path) {
            match statement_equations(ext, &e3, &st) {
                Ok(se) => {
                    eprintln!(
                        "seed d3 at ({},{}) {}: {} equations",
                        st.n,
                        st.f,
                        st.label,
                        se.len()
                    );
                    eqs.extend(se);
                }
                Err(e) => eprintln!("seed at ({},{}) {} skipped: {e}", st.n, st.f, st.label),
            }
        }
    }
    let (pivot, rows, _, contradictions) = eliminate(e3.nvars, &eqs, false);
    let free = pivot.iter().filter(|&&x| x == u32::MAX).count();
    eprintln!(
        "d3 solution space: dimension {free}, {} contradictions",
        contradictions.len()
    );

    // per E3 basis class with a known target: is d3 forced, and to what
    let mut classes: Vec<(u32, usize)> = vec![];
    let mut queries: Vec<Vec<u32>> = vec![];
    let mut bids: Vec<u32> = (0..ext.deg.len() as u32)
        .filter(|&b| e3.off[b as usize] != u32::MAX)
        .collect();
    bids.sort_by_key(|&b| ext.deg[b as usize]);
    for &b in &bids {
        let m = e3.dim(target(ext, &e3, b).unwrap().unwrap());
        for k in 0..e3.dim(b) {
            classes.push((b, k));
            for r in 0..m {
                queries.push(vec![var(ext, &e3, b, k, r)]);
            }
        }
    }
    let n_class_queries = queries.len();
    let checks: Vec<(Statement, Result<Vec<Vec<u32>>, String>)> = arg("--check-d3")
        .map(|path| {
            read_statements(path)
                .into_iter()
                .map(|st| {
                    let r = statement_equations(ext, &e3, &st);
                    (st, r)
                })
                .collect()
        })
        .unwrap_or_default();
    // a check holds iff each of its affine equations evaluates to 0 (constant column moved over)
    for (_, r) in &checks {
        if let Ok(se) = r {
            for e in se {
                queries.push(e.iter().copied().filter(|&v| v != e3.nvars).collect());
            }
        }
    }
    let vals = evaluate(e3.nvars, &pivot, &rows, &queries);

    let mut summary: BTreeMap<i32, (usize, usize, usize)> = BTreeMap::new();
    let mut w = arg("--solution3").map(|p| BufWriter::new(File::create(p).unwrap()));
    if let Some(w) = w.as_mut() {
        writeln!(w, "stem,Adams_filtration,rep,status,target").unwrap();
    }
    let mut qi = 0;
    for &(b, k) in &classes {
        let m = e3.dim(target(ext, &e3, b).unwrap().unwrap());
        let vs = &vals[qi..qi + m];
        qi += m;
        let det = vs.iter().all(|&(_, d)| d);
        let nonzero = vs.iter().any(|&(v, _)| v);
        let (n, f) = ext.deg[b as usize];
        let e = summary.entry(n).or_default();
        e.0 += 1;
        if det {
            e.1 += 1;
            if nonzero {
                e.2 += 1;
            }
        }
        if let Some(w) = w.as_mut() {
            let rep = e3.page[b as usize].as_ref().unwrap().reps[k];
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
    println!("stem,E3_classes_with_target,determined,determined_nonzero");
    let (mut tc, mut td, mut tz) = (0, 0, 0);
    for (s, (c, d, z)) in &summary {
        println!("{s},{c},{d},{z}");
        tc += c;
        td += d;
        tz += z;
    }
    println!(
        "total: {tc} classes, {td} determined ({tz} nonzero), {} open",
        tc - td
    );

    if !checks.is_empty() {
        let (mut ok, mut bad, mut open, mut skipped) = (0, 0, 0, 0);
        for (st, r) in &checks {
            match r {
                Err(e) => {
                    skipped += 1;
                    println!("check ({},{}) {}: skipped ({e})", st.n, st.f, st.label);
                }
                Ok(se) => {
                    let vs = &vals[qi..qi + se.len()];
                    qi += se.len();
                    // equation e holds iff (sum of variables) == constant
                    let holds = se.iter().zip(vs).map(|(e, &(v, d))| {
                        let c = e.contains(&e3.nvars);
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
                    println!("check ({},{}) {}: {status}", st.n, st.f, st.label);
                }
            }
        }
        println!("checks: {ok} correct and forced, {bad} wrong, {open} open, {skipped} skipped");
    }
}
