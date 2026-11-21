//! Adams d_r for r >= 3 on a diagram of spectra (`--e3`, `--diagram`), each page known as bounds.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{Ext, Module, Products, bits, eliminate, parse_vec64};

/// A vector over the E2 basis of one bidegree, or a set of page coordinates.
type V = u64;

/// An echelon form keyed by lowest set bit; each row remembers which basis vectors it combines.
#[derive(Default, Clone)]
struct Ech {
    rows: FxHashMap<u32, (V, V)>,
}

impl Ech {
    /// Reduces `v`; returns the remainder and the basis coefficients used.
    fn reduce(&self, mut v: V) -> (V, V) {
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
    fn insert(&mut self, v: V, coef: V) -> bool {
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
    fn project(&self, v: V) -> Option<V> {
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
    reps: Vec<V>,
    src: Ech,
    /// E2 masks spanning the possible E_r modulo the possible boundaries
    treps: Vec<V>,
    tgt: Ech,
    bcert: Vec<V>,
    bmax: Vec<V>,
    /// the possible boundaries are exactly the certain ones
    exact: bool,
    /// some class may be a cycle without being among the certain ones
    uncertain: bool,
}

impl Page {
    fn new(
        bcert: Vec<V>,
        bmax: Vec<V>,
        cycles: &[V],
        possible: &[V],
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
fn kernel(images: &[V]) -> Vec<V> {
    let mut rows: FxHashMap<u32, (V, V)> = FxHashMap::default();
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
fn kernel_keys(images: &[Vec<u64>]) -> Vec<V> {
    let xor = |a: &[u64], b: &[u64]| crate::xor_sorted(a, b);
    let mut rows: Vec<(Vec<u64>, V)> = vec![];
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

/// What a node's E2 page is a ring or module over.
enum Alg<'a> {
    /// A ring spectrum, with Leibniz on pairs of its own classes.
    Ring(&'a Ext, &'a Products),
    /// A module over the ring node with this index, through that ring's E2 action.
    Module(&'a Module, usize),
}

/// A spectrum of the diagram: its E2 page over the loaded range.
struct Node<'a> {
    name: String,
    alg: Alg<'a>,
    n_max: i32,
    f_max: i32,
}

impl Node<'_> {
    fn nb(&self) -> usize {
        match self.alg {
            Alg::Ring(e, _) => e.deg.len(),
            Alg::Module(m, _) => m.deg.len(),
        }
    }
    fn deg(&self, b: u32) -> (i32, i32) {
        match self.alg {
            Alg::Ring(e, _) => e.deg[b as usize],
            Alg::Module(m, _) => m.deg[b as usize],
        }
    }
    fn dim(&self, b: u32) -> usize {
        match self.alg {
            Alg::Ring(e, _) => e.dim[b as usize],
            Alg::Module(m, _) => m.dim[b as usize],
        }
    }
    fn in_range(&self, n: i32, f: i32) -> bool {
        n >= 0 && f >= 0 && n <= self.n_max && f <= self.f_max
    }
    fn get(&self, n: i32, f: i32) -> Option<u32> {
        if !self.in_range(n, f) {
            return None;
        }
        match self.alg {
            Alg::Ring(e, _) => e.bid.get(&(n, f)).copied(),
            Alg::Module(m, _) => m.bid.get(&(n, f)).copied(),
        }
    }
    fn gid(&self, b: u32, i: usize) -> u32 {
        match self.alg {
            Alg::Ring(e, _) => e.gid(b, i),
            Alg::Module(m, _) => m.gid(b, i),
        }
    }
}

/// A map of spectra of Adams filtration 0 between two nodes, on E2: it commutes with every d_r.
/// Maps of positive filtration (connecting maps) are not edges: their compatibility with d_r
/// holds only in a shifted form.
struct Edge {
    name: String,
    from: usize,
    to: usize,
    /// stem of the image minus stem of the source
    stem: i32,
    /// source global basis id -> image over the target basis
    img: FxHashMap<u32, V>,
}

impl Edge {
    /// Image of an E2 vector of source bidegree `b`.
    fn apply(&self, x: &Node, b: u32, v: V) -> V {
        bits(v).fold(0, |acc, i| {
            acc ^ self.img.get(&x.gid(b, i)).copied().unwrap_or(0)
        })
    }
}

/// Reads d2 files into global basis id -> target. Each file is either
/// `stem,Adams_filtration,index,target` or a `--solution` file
/// (`stem,Adams_filtration,index,status,target`), whose open classes are skipped. Earlier files
/// take precedence.
fn load_d2(node: &Node, paths: &str) -> FxHashMap<u32, V> {
    let mut d2 = FxHashMap::default();
    for path in paths.split(',').filter(|p| *p != "-") {
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
            if let Some(b) = node.get(n, f) {
                d2.entry(node.gid(b, i))
                    .or_insert_with(|| parse_vec64(rest));
                read += 1;
            }
        }
        eprintln!(
            "{}: d2 from {path}: {read} classes, {open} open skipped",
            node.name
        );
    }
    d2
}

/// Product of two E2 vectors of a ring, as a mask in the bidegree `a + c` (0 if that bidegree
/// is empty).
fn mul_vec(ext: &Ext, p: &Products, a: u32, u: V, c: u32, w: V) -> V {
    let mut out = 0;
    for i in bits(u) {
        for j in bits(w) {
            out ^= p.mul(ext, a, i, c, j) as V;
        }
    }
    out
}

/// Ring class `u` (bidegree `a`) times module class `w` (bidegree `b`), if every product of
/// basis elements it needs is known.
fn act(m: &Module, a: u32, u: V, b: u32, w: V) -> Option<V> {
    let mut out = 0;
    for i in bits(u) {
        for j in bits(w) {
            if !m.known(a, i, b, j) {
                return None;
            }
            out ^= m.mul(a, i, b, j);
        }
    }
    Some(out)
}

/// The page E_r of one node over the loaded range, with the unknowns d_r.
struct Pages {
    r: i32,
    /// bidegree id -> E_r there (`None`: unknown)
    page: Vec<Option<Page>>,
    /// bidegree id -> first (global) variable of d_r on it, `u32::MAX` if none
    off: Vec<u32>,
    /// bidegree id -> number of target coordinates of each of its source classes
    stride: Vec<u32>,
    nvars: u32,
}

impl Pages {
    fn new(r: i32, page: Vec<Option<Page>>) -> Self {
        let nb = page.len();
        Pages {
            r,
            page,
            off: vec![u32::MAX; nb],
            stride: vec![0; nb],
            nvars: 0,
        }
    }
    fn pg(&self, b: u32) -> &Page {
        self.page[b as usize].as_ref().unwrap()
    }
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
    /// Whether a target bidegree (`None`: Ext is zero there) has no boundary uncertainty.
    fn exact_or_zero(&self, t: Option<u32>) -> bool {
        t.is_none_or(|t| self.pg(t).exact)
    }
    /// The variable: coordinate `l` of d_r of source class `k` in bidegree `b`.
    fn var(&self, b: u32, k: usize, l: usize) -> u32 {
        debug_assert!(self.off[b as usize] != u32::MAX);
        self.off[b as usize] + k as u32 * self.stride[b as usize] + l as u32
    }
}

/// The bidegree `(n, f)`: `Some(Some(b))` if E_r is known there, `Some(None)` if Ext is zero
/// there, `None` if E_r there is unknown.
fn known(node: &Node, e: &Pages, n: i32, f: i32) -> Option<Option<u32>> {
    match node.get(n, f) {
        Some(b) => e.page[b as usize].as_ref().map(|_| Some(b)),
        None if node.in_range(n, f) || f < 0 || n < 0 => Some(None),
        None => None,
    }
}

/// The d_r target bidegree of `b`, if E_r is known there (`Some(None)`: zero).
fn target(node: &Node, e: &Pages, b: u32) -> Option<Option<u32>> {
    let (n, f) = node.deg(b);
    known(node, e, n - 1, f + e.r)
}

/// E3 where d2 is known on and into a bidegree, and unknown elsewhere.
fn build_e3(node: &Node, d2: &FxHashMap<u32, V>) -> Pages {
    let nb = node.nb();
    let mut page: Vec<Option<Page>> = (0..nb).map(|_| None).collect();
    let mut cycles_missing = 0;
    for b in 0..nb as u32 {
        let (n, f) = node.deg(b);
        // Z3 needs d2 out of b, B3 needs d2 into b: both bidegrees must be loaded (or have a
        // negative degree, where Ext is zero).
        let loaded = |n: i32, f: i32| n < 0 || f < 0 || node.in_range(n, f);
        if !loaded(n - 1, f + 2) || !loaded(n + 1, f - 2) {
            continue;
        }
        let d2_of = |g: u32| d2.get(&g).copied();
        let out_known = node.get(n - 1, f + 2).is_none()
            || (0..node.dim(b)).all(|i| d2_of(node.gid(b, i)).is_some());
        let into = node.get(n + 1, f - 2);
        let in_known =
            into.is_none_or(|a| (0..node.dim(a)).all(|i| d2_of(node.gid(a, i)).is_some()));
        if !out_known || !in_known {
            cycles_missing += 1;
            continue;
        }
        let images: Vec<V> = (0..node.dim(b))
            .map(|i| d2_of(node.gid(b, i)).unwrap_or(0))
            .collect();
        let bnd: Vec<V> = into
            .map(|a| {
                (0..node.dim(a))
                    .map(|i| d2_of(node.gid(a, i)).unwrap())
                    .collect()
            })
            .unwrap_or_default();
        let z = kernel(&images);
        page[b as usize] = Some(Page::new(bnd.clone(), bnd, &z, &z, true, false));
    }
    eprintln!(
        "{}: E3: {cycles_missing} bidegrees without d2 data",
        node.name
    );
    Pages::new(3, page)
}

/// E3 as bounds from E2 and what is known of d2: a class whose d2 is not in the files may have
/// any value.
fn bounds_e3(node: &Node, d2: &FxHashMap<u32, V>) -> Pages {
    let nb = node.nb();
    let page = (0..nb as u32)
        .map(|b| {
            let basis: Vec<V> = (0..node.dim(b)).map(|i| 1 << i).collect();
            Some(Page::new(vec![], vec![], &basis, &basis, true, false))
        })
        .collect();
    let e2 = Pages::new(2, page);
    let mut vals: FxHashMap<u32, Vec<Val>> = FxHashMap::default();
    let mut fresh = 0u64;
    let mut open = 0;
    for b in 0..nb as u32 {
        let Some(Some(t)) = target(node, &e2, b) else {
            continue;
        };
        let m = node.dim(t);
        let vs = (0..node.dim(b))
            .map(|i| match d2.get(&node.gid(b, i)) {
                Some(&part) => Val {
                    part,
                    dirs: vec![],
                    keys: bits(part).map(|t| t as u64).collect(),
                },
                None => {
                    open += 1;
                    let keys = (0..m)
                        .map(|t| {
                            fresh += 1;
                            64 + fresh * 64 + t as u64
                        })
                        .collect();
                    Val {
                        part: 0,
                        dirs: (0..m).map(|t| 1 << t).collect(),
                        keys,
                    }
                }
            })
            .collect();
        vals.insert(b, vs);
    }
    eprintln!("{}: E3 from bounds, {open} classes with open d2", node.name);
    next_page(node, &e2, &vals)
}

/// Numbers the unknowns of d_r from `start`, by decreasing source stem; returns the next free
/// variable.
fn assign_vars(node: &Node, e: &mut Pages, start: u32) -> u32 {
    let nb = node.nb();
    let mut order: Vec<u32> = (0..nb as u32).filter(|&b| e.dim(b) > 0).collect();
    order.sort_by_key(|&b| std::cmp::Reverse(node.deg(b)));
    e.off = vec![u32::MAX; nb];
    e.stride = vec![0; nb];
    let mut next = start;
    for b in order {
        if let Some(Some(t)) = target(node, e, b) {
            let m = e.tdim(t);
            if m > 0 {
                e.off[b as usize] = next;
                e.stride[b as usize] = m as u32;
                next += (e.dim(b) * m) as u32;
            }
        }
    }
    e.nvars = next - start;
    let pages = || e.page.iter().flatten();
    eprintln!(
        "{}: E{}: {} bidegrees known ({} exact, {} with uncertain cycles), {} source classes, {} \
         target coordinates, {} d{} variables",
        node.name,
        e.r,
        pages().count(),
        pages().filter(|p| p.exact).count(),
        pages().filter(|p| p.uncertain).count(),
        pages().map(|p| p.reps.len()).sum::<usize>(),
        pages().map(|p| p.treps.len()).sum::<usize>(),
        e.nvars,
        e.r
    );
    next
}

/// What is known of d_r on one source class: its value at the particular solution and, for
/// each free direction it depends on, how it moves (both in target coordinates).
#[derive(Clone, Default)]
struct Val {
    part: V,
    dirs: Vec<V>,
    /// sparse key form of (part, dirs) for kernels of combinations
    keys: Vec<u64>,
}

/// Bounds on E_{r+1} from those on E_r and what is known of d_r.
fn next_page(node: &Node, cur: &Pages, vals: &FxHashMap<u32, Vec<Val>>) -> Pages {
    let nb = node.nb();
    let r = cur.r;
    let as_e2 = |reps: &[V], coords: V| -> V { bits(coords).fold(0, |acc, l| acc ^ reps[l]) };
    let page: Vec<Option<Page>> = (0..nb as u32)
        .map(|b| {
            let pg = cur.page[b as usize].as_ref()?;
            let (n, f) = node.deg(b);
            // certain cycles: combinations whose d_r is forced to vanish in the true E_r
            let (cycles, open_here): (Vec<V>, bool) = match target(node, cur, b) {
                None => (vec![], !pg.reps.is_empty()),
                Some(None) => (pg.reps.clone(), false),
                Some(Some(t)) if cur.tdim(t) == 0 => (pg.reps.clone(), false),
                Some(Some(t)) => {
                    let vs = vals.get(&b).map_or(&[][..], |v| &v[..]);
                    let any_open = vs.iter().any(|v| !v.dirs.is_empty());
                    if cur.pg(t).exact {
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
            match known(node, cur, n + 1, f - r) {
                None => {
                    bmax.extend(&pg.treps);
                    exact = false;
                }
                Some(None) => {}
                Some(Some(a)) => {
                    if cur.pg(a).uncertain {
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
    Pages::new(r + 1, page)
}

/// Sorts a list of variables and cancels repeats.
fn reduce_eq(mut tr: Vec<u32>) -> Vec<u32> {
    tr.sort_unstable();
    let mut e: Vec<u32> = Vec::with_capacity(tr.len());
    for v in tr {
        if e.last() == Some(&v) {
            e.pop();
        } else {
            e.push(v);
        }
    }
    e
}

/// Leibniz on E_r of a ring for every pair of source classes: d_r[xy] = d_r[x][y] + [x]d_r[y],
/// projected onto the target bounds. A term d_r[x][y] is only well defined when the target of x
/// has no boundary uncertainty, so pairs without that are skipped.
fn ring_equations(node: &Node, e3: &Pages) -> Vec<Vec<u32>> {
    let Alg::Ring(ext, p) = node.alg else {
        unreachable!()
    };
    let t = Instant::now();
    let ablate_zero = std::env::var("E3_ABLATE_ZERO").is_ok_and(|v| v == "1");
    let bids: Vec<u32> = (0..node.nb() as u32).filter(|&b| e3.dim(b) > 0).collect();
    let eqs: Vec<Vec<u32>> = bids
        .par_iter()
        .flat_map_iter(|&a| {
            let mut out = vec![];
            let (an, af) = node.deg(a);
            for &c in bids.iter().filter(|&&c| c >= a) {
                let (cn, cf) = node.deg(c);
                let Some(Some(sp)) = known(node, e3, an + cn - 1, af + cf + e3.r) else {
                    continue;
                };
                let Some(s) = known(node, e3, an + cn, af + cf) else {
                    continue;
                };
                let (Some(ap), Some(cp)) = (target(node, e3, a), target(node, e3, c)) else {
                    continue;
                };
                if !e3.exact_or_zero(ap) || !e3.exact_or_zero(cp) {
                    continue;
                }
                let m = e3.tdim(sp);
                if m == 0 {
                    continue;
                }
                let pg_a = e3.pg(a);
                let pg_c = e3.pg(c);
                let pg_sp = e3.pg(sp);
                'pair: for (k, &x) in pg_a.reps.iter().enumerate() {
                    let j0 = if a == c { k } else { 0 };
                    for (j, &y) in pg_c.reps.iter().enumerate().skip(j0) {
                        // per target coordinate, the variables of the equation
                        let mut terms: Vec<Vec<u32>> = vec![vec![]; m];
                        // the product as a combination of source classes, if it is certain
                        let pr = match s {
                            None => 0,
                            Some(s) => match e3.pg(s).src.project(mul_vec(ext, p, a, x, c, y)) {
                                Some(pr) => pr,
                                None => continue,
                            },
                        };
                        // E3_ABLATE_ZERO=1 drops the pairs whose product is 0 in E_r: an ablation
                        if pr == 0 && ablate_zero {
                            continue;
                        }
                        if let Some(s) = s {
                            for q in bits(pr) {
                                for (r, tr) in terms.iter_mut().enumerate() {
                                    tr.push(e3.var(s, q, r));
                                }
                            }
                        }
                        if let Some(ap) = ap {
                            for (l, &z) in e3.pg(ap).treps.iter().enumerate() {
                                let Some(q) = pg_sp.tgt.project(mul_vec(ext, p, ap, z, c, y))
                                else {
                                    continue 'pair;
                                };
                                for r in bits(q) {
                                    terms[r].push(e3.var(a, k, l));
                                }
                            }
                        }
                        if let Some(cp) = cp {
                            for (l, &z) in e3.pg(cp).treps.iter().enumerate() {
                                let Some(q) = pg_sp.tgt.project(mul_vec(ext, p, a, x, cp, z))
                                else {
                                    continue 'pair;
                                };
                                for r in bits(q) {
                                    terms[r].push(e3.var(c, j, l));
                                }
                            }
                        }
                        for tr in terms {
                            let e = reduce_eq(tr);
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
        "{}: E{} Leibniz: {} nontrivial equations, {:.1}s",
        node.name,
        e3.r,
        eqs.len(),
        t.elapsed().as_secs_f64()
    );
    eqs
}

/// The module Leibniz rule on E_r, d_r[a m] = d_r[a] m + a d_r[m] for every source class `a` of
/// the ring and `m` of the module, projected onto the target bounds. As for a ring, each term
/// needs the target of its unknown to have no boundary uncertainty, and every product it uses
/// must be known (an unknown product drops the pair).
fn module_equations(mn: &Node, em: &Pages, rn: &Node, er: &Pages) -> Vec<Vec<u32>> {
    let Alg::Module(m, _) = mn.alg else {
        unreachable!()
    };
    let t = Instant::now();
    let abids: Vec<u32> = (0..rn.nb() as u32).filter(|&a| er.dim(a) > 0).collect();
    let bbids: Vec<u32> = (0..mn.nb() as u32).filter(|&b| em.dim(b) > 0).collect();
    let r = em.r;
    let (eqs, dropped): (Vec<Vec<Vec<u32>>>, Vec<usize>) = abids
        .par_iter()
        .map(|&a| {
            let mut out = vec![];
            let mut dropped = 0;
            let (an, af) = rn.deg(a);
            let Some(ap) = target(rn, er, a) else {
                return (out, 0);
            };
            if !er.exact_or_zero(ap) {
                return (out, 0);
            }
            for &b in &bbids {
                let (bn, bf) = mn.deg(b);
                let Some(Some(sp)) = known(mn, em, an + bn - 1, af + bf + r) else {
                    continue;
                };
                let Some(s) = known(mn, em, an + bn, af + bf) else {
                    continue;
                };
                let Some(bp) = target(mn, em, b) else {
                    continue;
                };
                if !em.exact_or_zero(bp) {
                    continue;
                }
                let dm = em.tdim(sp);
                if dm == 0 {
                    continue;
                }
                let pg_sp = em.pg(sp);
                for (k, &x) in er.pg(a).reps.iter().enumerate() {
                    'pair: for (j, &y) in em.pg(b).reps.iter().enumerate() {
                        let mut terms: Vec<Vec<u32>> = vec![vec![]; dm];
                        if let Some(s) = s {
                            let Some(pr) = act(m, a, x, b, y).and_then(|v| em.pg(s).src.project(v))
                            else {
                                dropped += 1;
                                continue;
                            };
                            for q in bits(pr) {
                                for (l, tr) in terms.iter_mut().enumerate() {
                                    tr.push(em.var(s, q, l));
                                }
                            }
                        }
                        if let Some(ap) = ap {
                            for (l, &z) in er.pg(ap).treps.iter().enumerate() {
                                let Some(q) =
                                    act(m, ap, z, b, y).and_then(|v| pg_sp.tgt.project(v))
                                else {
                                    dropped += 1;
                                    continue 'pair;
                                };
                                for c in bits(q) {
                                    terms[c].push(er.var(a, k, l));
                                }
                            }
                        }
                        if let Some(bp) = bp {
                            for (l, &z) in em.pg(bp).treps.iter().enumerate() {
                                let Some(q) =
                                    act(m, a, x, bp, z).and_then(|v| pg_sp.tgt.project(v))
                                else {
                                    dropped += 1;
                                    continue 'pair;
                                };
                                for c in bits(q) {
                                    terms[c].push(em.var(b, j, l));
                                }
                            }
                        }
                        for tr in terms {
                            let e = reduce_eq(tr);
                            if !e.is_empty() {
                                out.push(e);
                            }
                        }
                    }
                }
            }
            (out, dropped)
        })
        .unzip();
    let eqs: Vec<Vec<u32>> = eqs.into_iter().flatten().collect();
    eprintln!(
        "{}: E{r} module Leibniz over {}: {} nontrivial equations, {} pairs dropped, {:.1}s",
        mn.name,
        rn.name,
        eqs.len(),
        dropped.iter().sum::<usize>(),
        t.elapsed().as_secs_f64()
    );
    eqs
}

/// Naturality d_r(f x) = f(d_r x) along a filtration-0 map, for every source class `x`,
/// projected onto the target bounds of the image. f(d_r x) is only well defined when the target
/// of x has no boundary uncertainty (f of a possible boundary need not be a possible boundary of
/// the image); f x must be a certain cycle of the image to name its d_r.
fn map_equations(edge: &Edge, xn: &Node, ex: &Pages, yn: &Node, ey: &Pages) -> Vec<Vec<u32>> {
    let r = ex.r;
    let mut out = vec![];
    let mut dropped = 0;
    for c in 0..xn.nb() as u32 {
        if ex.dim(c) == 0 {
            continue;
        }
        let Some(t) = target(xn, ex, c) else {
            continue;
        };
        if !ex.exact_or_zero(t) {
            continue;
        }
        let (n, f) = xn.deg(c);
        let Some(Some(yt)) = known(yn, ey, n + edge.stem - 1, f + r) else {
            continue;
        };
        let m = ey.tdim(yt);
        if m == 0 {
            continue;
        }
        let Some(yb) = known(yn, ey, n + edge.stem, f) else {
            continue;
        };
        let t_images: Option<Vec<V>> = match t {
            None => Some(vec![]),
            Some(t) => ex
                .pg(t)
                .treps
                .iter()
                .map(|&z| ey.pg(yt).tgt.project(edge.apply(xn, t, z)))
                .collect(),
        };
        let Some(t_images) = t_images else {
            dropped += ex.dim(c);
            continue;
        };
        for (k, &x) in ex.pg(c).reps.iter().enumerate() {
            let src = match yb {
                None => 0,
                Some(yb) => match ey.pg(yb).src.project(edge.apply(xn, c, x)) {
                    Some(s) => s,
                    None => {
                        dropped += 1;
                        continue;
                    }
                },
            };
            for row in 0..m {
                let mut e: Vec<u32> = vec![];
                if let Some(yb) = yb {
                    e.extend(bits(src).map(|q| ey.var(yb, q, row)));
                }
                for (l, &img) in t_images.iter().enumerate() {
                    if img >> row & 1 == 1 {
                        e.push(ex.var(c, k, l));
                    }
                }
                let e = reduce_eq(e);
                if !e.is_empty() {
                    out.push(e);
                }
            }
        }
    }
    eprintln!(
        "{}: E{r} naturality: {} nontrivial equations, {dropped} classes dropped",
        edge.name,
        out.len()
    );
    out
}

/// A statement read from a file: E2 vectors of a source and its d_r target.
struct Statement {
    n: i32,
    f: i32,
    src: V,
    tgt: V,
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
        let src = parse_vec64(vecs.next().unwrap());
        let tgt = parse_vec64(vecs.next().unwrap_or("[]"));
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

/// The equations `d_r[src] = [tgt]` (affine: the constant column is `one`), projected onto the
/// target bounds.
fn statement_equations(
    node: &Node,
    e3: &Pages,
    st: &Statement,
    one: u32,
) -> Result<Vec<Vec<u32>>, String> {
    let b = node.get(st.n, st.f).ok_or("source bidegree empty")?;
    let pg = e3.page[b as usize]
        .as_ref()
        .ok_or("page unknown at source")?;
    let s = pg
        .src
        .project(st.src)
        .ok_or("source is not a certain cycle on this page")?;
    let tb = match target(node, e3, b).ok_or("page unknown at target")? {
        Some(t) => t,
        None => return Ok(vec![]),
    };
    let tpg = e3.pg(tb);
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
            let mut e: Vec<u32> = bits(s).map(|k| e3.var(b, k, r)).collect();
            if tc >> r & 1 == 1 {
                e.push(one);
            }
            e
        })
        .collect())
}

/// Evaluates linear combinations of variables on the solution space: the value at the
/// particular solution, and the free columns (by index among the free columns) it depends on.
pub(crate) fn evaluate(
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

/// `(page, node, path)` from repeatable `<flag> r:[node:]path` arguments, plus the legacy page-3
/// flag; `node` is `None` (the first node) unless the middle field names a node.
fn page_files(
    args: &[String],
    flag: &str,
    legacy: &str,
    names: &[String],
) -> Vec<(i32, usize, String)> {
    let mut out = vec![];
    for (i, a) in args.iter().enumerate() {
        if a == flag {
            let (r, rest) = args[i + 1].split_once(':').expect("expected r:path");
            let (node, path) = match rest.split_once(':') {
                Some((nm, path)) if names.iter().any(|x| x == nm) => {
                    (names.iter().position(|x| x == nm).unwrap(), path)
                }
                _ => (0, rest),
            };
            out.push((r.parse().unwrap(), node, path.to_string()));
        } else if a == legacy {
            out.push((3, 0, args[i + 1].clone()));
        }
    }
    out
}

/// A diagram as read from `--diagram <file>`, or from the legacy flags.
///
/// Lines (`#` starts a comment):
/// - `ring <name> <ring dir> <d2 files>`: a ring spectrum; the dir `-` is the main ring
///   directory, whose node must come first;
/// - `module <name> <ring node> <module dir> <shift> <d2 files>`: the two-cell complex `name`
///   from ext's `cofiber_export` over that ring, with its maps `i` from the ring and `p` to the
///   ring `shift` stems lower;
/// - `map <from> <to> <map csv>`: a filtration-0 ring map on E2, in the format of
///   `compare::load_iota`.
///
/// d2 files are comma-separated, `-` for none.
#[derive(Default)]
struct Spec {
    rings: Vec<(String, String, String)>,
    modules: Vec<(String, String, String, i32, String)>,
    maps: Vec<(String, String, String)>,
}

fn read_spec(args: &[String]) -> Spec {
    let arg = |flag: &str| args.iter().position(|a| a == flag).map(|i| &args[i + 1]);
    let mut spec = Spec::default();
    if let Some(path) = arg("--diagram") {
        for line in BufReader::new(File::open(path).unwrap()).lines() {
            let line = line.unwrap();
            let w: Vec<&str> = line.split('#').next().unwrap().split_whitespace().collect();
            let s = |i: usize| w[i].to_string();
            match w.first() {
                None => {}
                Some(&"ring") => spec.rings.push((s(1), s(2), s(3))),
                Some(&"module") => {
                    spec.modules
                        .push((s(1), s(2), s(3), w[4].parse().unwrap(), s(5)))
                }
                Some(&"map") => spec.maps.push((s(1), s(2), s(3))),
                Some(other) => panic!("diagram: unknown line kind {other}"),
            }
        }
        return spec;
    }
    spec.rings
        .push(("S".into(), "-".into(), arg("--e3").unwrap().clone()));
    if let Some(ci) = args.iter().position(|a| a == "--compare") {
        let (dir, iota, d2) = (&args[ci + 1], &args[ci + 2], &args[ci + 3]);
        spec.rings.push(("R".into(), dir.clone(), d2.clone()));
        spec.maps.push(("S".into(), "R".into(), iota.clone()));
    }
    spec
}

pub(crate) fn run(ext: &Ext, p: &Products, args: &[String]) {
    let arg = |flag: &str| args.iter().position(|a| a == flag).map(|i| &args[i + 1]);
    let last: i32 = arg("--pages").map_or(3, |s| s.parse().unwrap());
    let from_bounds = args.iter().any(|a| a == "--e3-bounds");
    let spec = read_spec(args);
    assert_eq!(spec.rings[0].1, "-", "the main ring must be the first node");

    // owned data first, then the nodes borrowing it
    let rings: Vec<Option<(Ext, Products)>> = spec
        .rings
        .iter()
        .map(|(_, dir, _)| {
            (dir != "-").then(|| {
                let e = crate::load(dir, ext.n_max, ext.f_max);
                let pr = crate::build_products(&e);
                crate::check_products(&e, &pr);
                (e, pr)
            })
        })
        .collect();
    let ring_of = |i: usize| -> (&Ext, &Products) {
        match &rings[i] {
            Some((e, p)) => (e, p),
            None => (ext, p),
        }
    };
    let ring_idx = |name: &str| {
        spec.rings
            .iter()
            .position(|r| r.0 == name)
            .unwrap_or_else(|| panic!("diagram: no ring {name}"))
    };
    let mods: Vec<Module> = spec
        .modules
        .iter()
        .map(|(name, ring, dir, shift, _)| {
            let (e, pr) = ring_of(ring_idx(ring));
            let mut m = crate::load_module(e, dir, name, *shift);
            crate::build_action(e, pr, &mut m);
            crate::check_action(e, &m, dir);
            m
        })
        .collect();
    // A node's range stops where its data does: past the last stem or filtration with a class,
    // Ext would read as zero where it was never computed.
    let extent = |deg: &[(i32, i32)], n_max: i32, f_max: i32| {
        let n = deg.iter().map(|d| d.0).max().unwrap_or(0);
        let f = deg.iter().map(|d| d.1).max().unwrap_or(0);
        (n_max.min(n), f_max.min(f))
    };
    let mut nodes: Vec<Node> = vec![];
    let mut d2_paths: Vec<&str> = vec![];
    for (i, (name, _, d2)) in spec.rings.iter().enumerate() {
        let (e, pr) = ring_of(i);
        let (n_max, f_max) = extent(&e.deg, e.n_max, e.f_max);
        nodes.push(Node {
            name: name.clone(),
            alg: Alg::Ring(e, pr),
            n_max,
            f_max,
        });
        d2_paths.push(d2);
    }
    for (m, (name, ring, _, _, d2)) in mods.iter().zip(&spec.modules) {
        let ring = ring_idx(ring);
        let (n_max, f_max) = extent(&m.deg, nodes[ring].n_max, nodes[ring].f_max);
        nodes.push(Node {
            name: name.clone(),
            alg: Alg::Module(m, ring),
            n_max,
            f_max,
        });
        d2_paths.push(d2);
    }
    for n in &nodes {
        eprintln!(
            "node {}: stems <= {}, filtrations <= {}",
            n.name, n.n_max, n.f_max
        );
    }
    let names: Vec<String> = nodes.iter().map(|n| n.name.clone()).collect();
    let node_idx = |name: &str| {
        names
            .iter()
            .position(|x| x == name)
            .unwrap_or_else(|| panic!("diagram: no node {name}"))
    };
    let mut edges: Vec<Edge> = vec![];
    for (from, to, path) in &spec.maps {
        let (x, y) = (node_idx(from), node_idx(to));
        let iota = crate::compare::load_iota(path);
        let mut img = FxHashMap::default();
        for ((n, f, i), v) in iota {
            if let (Some(b), Some(_)) = (nodes[x].get(n, f), nodes[y].get(n, f)) {
                img.insert(nodes[x].gid(b, i), v as V);
            }
        }
        edges.push(Edge {
            name: format!("{from}->{to}"),
            from: x,
            to: y,
            stem: 0,
            img,
        });
    }
    for (mi, m) in mods.iter().enumerate() {
        let (x, ring) = (spec.rings.len() + mi, ring_idx(&spec.modules[mi].1));
        edges.push(Edge {
            name: format!("i:{}->{}", names[ring], m.name),
            from: ring,
            to: x,
            stem: 0,
            img: m.istar.clone(),
        });
        edges.push(Edge {
            name: format!("p:{}->{}", m.name, names[ring]),
            from: x,
            to: ring,
            stem: -m.shift,
            img: m.pstar.iter().map(|(&k, &v)| (k, v as V)).collect(),
        });
    }
    let d2s: Vec<FxHashMap<u32, V>> = nodes
        .iter()
        .zip(&d2_paths)
        .map(|(n, paths)| load_d2(n, paths))
        .collect();
    let seeds = page_files(args, "--seed", "--seed-d3", &names);
    let check_files = page_files(args, "--check", "--check-d3", &names);

    // Facts d_r(y) = 0 learned from page r + 1, per node (page -> (stem, filtration, E2 vector
    // of y)). They are kept as E2 vectors because the page bases change from one pass to the
    // next.
    let mut back: Vec<FxHashMap<i32, BTreeSet<(i32, i32, V)>>> =
        vec![FxHashMap::default(); nodes.len()];
    for pass in 1.. {
        let t_pass = Instant::now();
        let facts_before: usize = back.iter().flat_map(|b| b.values()).map(|s| s.len()).sum();
        println!("pass {pass}");
        let mut es: Vec<Pages> = nodes
            .iter()
            .zip(&d2s)
            .map(|(n, d2)| {
                if from_bounds {
                    bounds_e3(n, d2)
                } else {
                    build_e3(n, d2)
                }
            })
            .collect();
        loop {
            let r = es[0].r;
            let mut total = 0;
            for (n, e) in nodes.iter().zip(es.iter_mut()) {
                total = assign_vars(n, e, total);
            }
            let mut eqs: Vec<Vec<u32>> = vec![];
            for (n, e) in nodes.iter().zip(&es) {
                match n.alg {
                    Alg::Ring(..) => eqs.extend(ring_equations(n, e)),
                    Alg::Module(_, ring) => {
                        eqs.extend(module_equations(n, e, &nodes[ring], &es[ring]))
                    }
                }
            }
            for edge in &edges {
                let (x, y) = (edge.from, edge.to);
                eqs.extend(map_equations(edge, &nodes[x], &es[x], &nodes[y], &es[y]));
            }
            for (ni, (n, e)) in nodes.iter().zip(&es).enumerate() {
                let mut n_back = 0;
                for &(sn, sf, src) in back[ni].get(&r).into_iter().flatten() {
                    let st = Statement {
                        n: sn,
                        f: sf,
                        src,
                        tgt: 0,
                        label: String::new(),
                    };
                    if let Ok(se) = statement_equations(n, e, &st, total) {
                        eqs.extend(se);
                        n_back += 1;
                    }
                }
                if n_back > 0 {
                    eprintln!("{}: d{r}: {n_back} backward facts d{r}(y) = 0 used", n.name);
                }
            }
            for (_, ni, path) in seeds.iter().filter(|(sr, ..)| *sr == r) {
                let (n, e) = (&nodes[*ni], &es[*ni]);
                for st in read_statements(path) {
                    // a seed d_r(src) = tgt also says that src and tgt survive to E_r
                    if r > 3 {
                        let set = back[*ni].entry(r - 1).or_default();
                        set.insert((st.n, st.f, st.src));
                        if st.tgt != 0 {
                            set.insert((st.n - 1, st.f + r, st.tgt));
                        }
                    }
                    match statement_equations(n, e, &st, total) {
                        Ok(se) => {
                            eprintln!(
                                "{}: seed d{r} at ({},{}) {}: {} equations",
                                n.name,
                                st.n,
                                st.f,
                                st.label,
                                se.len()
                            );
                            eqs.extend(se);
                        }
                        Err(err) => {
                            eprintln!(
                                "{}: seed d{r} at ({},{}) {} skipped: {err}",
                                n.name, st.n, st.f, st.label
                            )
                        }
                    }
                }
            }
            let (pivot, rows, _, contradictions) = eliminate(total, &eqs, false);
            let free = pivot.iter().filter(|&&x| x == u32::MAX).count();
            eprintln!(
                "d{r} solution space: dimension {free}, {} contradictions",
                contradictions.len()
            );
            println!("d{r}: {} contradictions", contradictions.len());

            // per node, per source class with a known target: what is known of d_r
            let mut classes: Vec<(usize, u32, usize)> = vec![];
            let mut queries: Vec<Vec<u32>> = vec![];
            for (ni, (n, e)) in nodes.iter().zip(&es).enumerate() {
                let mut bids: Vec<u32> = (0..n.nb() as u32)
                    .filter(|&b| e.off[b as usize] != u32::MAX)
                    .collect();
                bids.sort_by_key(|&b| n.deg(b));
                for &b in &bids {
                    for k in 0..e.dim(b) {
                        classes.push((ni, b, k));
                        for t in 0..e.stride[b as usize] as usize {
                            queries.push(vec![e.var(b, k, t)]);
                        }
                    }
                }
            }
            let n_class_queries = queries.len();
            let checks: Vec<(usize, Statement, Result<Vec<Vec<u32>>, String>)> = check_files
                .iter()
                .filter(|(cr, ..)| *cr == r)
                .flat_map(|(_, ni, path)| read_statements(path).into_iter().map(|st| (*ni, st)))
                .map(|(ni, st)| {
                    let res = statement_equations(&nodes[ni], &es[ni], &st, total);
                    (ni, st, res)
                })
                .collect();
            // a check holds iff each of its affine equations evaluates to its constant
            for (_, _, res) in &checks {
                if let Ok(se) = res {
                    for eq in se {
                        queries.push(eq.iter().copied().filter(|&v| v != total).collect());
                    }
                }
            }
            let vals = evaluate(total, &pivot, &rows, &queries);

            let mut summary: Vec<BTreeMap<i32, (usize, usize, usize)>> =
                vec![BTreeMap::new(); nodes.len()];
            let prefix = if r == 3 {
                arg("--solution3").cloned()
            } else {
                None
            }
            .map(|p| p.trim_end_matches(".csv").to_string())
            .or_else(|| arg("--solution-pages").map(|pre| format!("{pre}_d{r}")));
            let mut ws: Vec<Option<BufWriter<File>>> = names
                .iter()
                .enumerate()
                .map(|(ni, name)| {
                    let pre = prefix.as_ref()?;
                    let path = if ni == 0 {
                        format!("{pre}.csv")
                    } else {
                        format!("{pre}_{name}.csv")
                    };
                    let mut w = BufWriter::new(File::create(path).unwrap());
                    writeln!(w, "stem,Adams_filtration,rep,status,target").unwrap();
                    Some(w)
                })
                .collect();
            let mut dr: Vec<FxHashMap<u32, Vec<Val>>> = vec![FxHashMap::default(); nodes.len()];
            // per node: coordinates, forced, forced to 1
            let mut coords = vec![(0, 0, 0); nodes.len()];
            let mut qi = 0;
            for &(ni, b, k) in &classes {
                let e = &es[ni];
                let m = e.stride[b as usize] as usize;
                let vs = &vals[qi..qi + m];
                qi += m;
                for (v, d) in vs {
                    coords[ni].0 += 1;
                    coords[ni].1 += d.is_empty() as usize;
                    coords[ni].2 += (d.is_empty() && *v) as usize;
                }
                let mut v = Val::default();
                let mut dirs: BTreeMap<u32, V> = BTreeMap::new();
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
                let (n, f) = nodes[ni].deg(b);
                let s = summary[ni].entry(n).or_default();
                s.0 += 1;
                if det {
                    s.1 += 1;
                    if v.part != 0 {
                        s.2 += 1;
                    }
                }
                if let Some(w) = ws[ni].as_mut() {
                    let rep = e.pg(b).reps[k];
                    let tgt: Vec<String> = vs.iter().map(|(v, _)| (*v as u8).to_string()).collect();
                    writeln!(
                        w,
                        "{n},{f},\"{rep:#b}\",{},\"[{}]\"",
                        if det { "determined" } else { "open" },
                        tgt.join(",")
                    )
                    .unwrap();
                }
                dr[ni].entry(b).or_default().push(v);
            }
            assert_eq!(qi, n_class_queries);
            // a forced value d_r(x) = y in a target without boundary uncertainty is a true
            // d_{r-1}-cycle: d_{r-1}(y) = 0
            if r > 3 {
                for (ni, (n, e)) in nodes.iter().zip(&es).enumerate() {
                    for (&b, vs) in &dr[ni] {
                        let Some(Some(t)) = target(n, e, b) else {
                            continue;
                        };
                        let tp = e.pg(t);
                        if !tp.exact {
                            continue;
                        }
                        let (tn, tf) = n.deg(t);
                        for v in vs.iter().filter(|v| v.dirs.is_empty() && v.part != 0) {
                            let y = bits(v.part).fold(0, |acc, l| acc ^ tp.treps[l]);
                            back[ni].entry(r - 1).or_default().insert((tn, tf, y));
                        }
                    }
                }
            }
            println!("page {r}");
            for (ni, name) in names.iter().enumerate() {
                if args.iter().any(|a| a == "--per-stem") || ni == 0 {
                    println!("{name}: stem,E{r}_classes_with_target,determined,determined_nonzero");
                    for (s, (c, d, z)) in &summary[ni] {
                        println!("{s},{c},{d},{z}");
                    }
                }
            }
            for (ni, name) in names.iter().enumerate() {
                let (tc, td, tz) = summary[ni]
                    .values()
                    .fold((0, 0, 0), |a, s| (a.0 + s.0, a.1 + s.1, a.2 + s.2));
                println!(
                    "total d{r} {name}: {tc} classes, {td} determined ({tz} nonzero), {} open; \
                     {} coordinates, {} forced ({} equal to 1)",
                    tc - td,
                    coords[ni].0,
                    coords[ni].1,
                    coords[ni].2
                );
            }

            for (ni, name) in names.iter().enumerate() {
                if !checks.iter().any(|c| c.0 == ni) {
                    continue;
                }
                let (mut ok, mut bad, mut open, mut skipped, mut exact_ok) = (0, 0, 0, 0, 0);
                qi = n_class_queries;
                for (cni, st, res) in &checks {
                    let qn = res.as_ref().map_or(0, |se| se.len());
                    if *cni != ni {
                        qi += qn;
                        continue;
                    }
                    match res {
                        Err(err) => {
                            skipped += 1;
                            println!(
                                "check d{r} {name} ({},{}) {}: skipped ({err})",
                                st.n, st.f, st.label
                            );
                        }
                        Ok(se) => {
                            let vs = &vals[qi..qi + se.len()];
                            qi += se.len();
                            let holds = se.iter().zip(vs).map(|(eq, (v, deps))| {
                                let c = eq.contains(&total);
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
                            // exact: the target page has no boundary uncertainty, so the verdict
                            // is about the true E_r rather than its quotient by the possible
                            // boundaries
                            let exact = nodes[ni]
                                .get(st.n, st.f)
                                .and_then(|b| target(&nodes[ni], &es[ni], b))
                                .flatten()
                                .is_none_or(|t| es[ni].pg(t).exact);
                            if status == "correct" && exact {
                                exact_ok += 1;
                            }
                            let how = if exact {
                                "exact"
                            } else {
                                "modulo possible boundaries"
                            };
                            println!(
                                "check d{r} {name} ({},{}) {}: {status} ({how})",
                                st.n, st.f, st.label
                            );
                        }
                    }
                }
                println!(
                    "checks d{r} {name}: {ok} correct and forced ({exact_ok} exact), {bad} wrong, \
                     {open} open, {skipped} skipped"
                );
            }
            if r >= last {
                break;
            }
            es = nodes
                .iter()
                .zip(&es)
                .zip(&dr)
                .map(|((n, e), d)| next_page(n, e, d))
                .collect();
        }
        let facts: usize = back.iter().flat_map(|b| b.values()).map(|s| s.len()).sum();
        eprintln!(
            "pass {pass}: {facts} backward facts ({} new), {:.1}s",
            facts - facts_before,
            t_pass.elapsed().as_secs_f64()
        );
        if facts == facts_before {
            break;
        }
    }
}
