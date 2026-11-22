//! Adams d_r for r >= 2 on a diagram of spectra (`--diagram`, `--e3`), each page known as bounds.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{Ext, Module, Products, bits, eliminate, parse_vec64};

/// Determined differentials of one page, for the export: (node, source bidegree, source, target),
/// both as E2 vectors.
type Exported = Vec<(usize, u32, V, V)>;

/// Writes each node's determined differentials as an `Sseq<3, SyntheticAdams>` (stem, filtration,
/// λ-offset 0) to `<dir>/<node>.json`, and reports the degrees sseq finds inconsistent: sseq
/// recomputes the pages itself, so any would mean the engine's differentials disagree.
fn export_sseq(dir: &str, nodes: &[Node], pages: &BTreeMap<i32, Exported>) {
    use fp::{prime::ValidPrime, vector::FpVector};
    use sseq::{
        Sseq, SyntheticAdams,
        coordinates::{degree::MultiDegree, element::MultiDegreeElement},
    };
    std::fs::create_dir_all(dir).unwrap();
    let p = ValidPrime::new(2);
    let r_max = pages.keys().copied().max().unwrap_or(2);
    let vector = |v: V, dim: usize| -> FpVector {
        let mut out = FpVector::new(p, dim);
        for i in bits(v) {
            out.set_entry(i, 1);
        }
        out
    };
    for (ni, node) in nodes.iter().enumerate() {
        let t = Instant::now();
        let dim = |n: i32, f: i32| node.get(n, f).map_or(0, |b| node.dim(b));
        let mut sseq = Sseq::<3, SyntheticAdams>::new(p);
        for n in -1..=node.n_max + 1 {
            for f in 0..=node.f_max + r_max + 1 {
                sseq.set_dimension(MultiDegree::new([n, f, 0]), dim(n, f));
            }
        }
        let mut count = 0;
        for (&r, items) in pages {
            for &(_, b, src, tgt) in items.iter().filter(|x| x.0 == ni) {
                let (n, f) = node.deg(b);
                let source =
                    MultiDegreeElement::new(MultiDegree::new([n, f, 0]), vector(src, dim(n, f)));
                let target = vector(tgt, dim(n - 1, f + r));
                sseq.add_differential(r, &source, target.as_slice());
                count += 1;
            }
        }
        sseq.update();
        let bad = sseq
            .iter_degrees()
            .filter(|&b| sseq.inconsistent(b))
            .count();
        let path = format!("{dir}/{}.json", node.name);
        let json = serde_json::to_vec(&sseq).unwrap();
        std::fs::write(&path, &json).unwrap();
        eprintln!(
            "{}: exported {count} differentials to {path} ({:.1} MB), {bad} inconsistent degrees, \
             {:.1}s",
            node.name,
            json.len() as f64 / 1e6,
            t.elapsed().as_secs_f64()
        );
        println!(
            "export {}: {count} differentials, {bad} inconsistent degrees",
            node.name
        );
    }
}

/// A vector over the E2 basis of one bidegree, or a set of page coordinates.
type V = u64;

/// Where an equation comes from, kept compact (rendered by [`Labels::origin`]) because a
/// certificate run keeps one per equation.
#[derive(Clone, Copy)]
enum Origin {
    /// Leibniz for the classes `(a, k)` and `(c, j)` of a ring node; `row` is the target
    /// coordinate of the product's d_r.
    Ring {
        node: u16,
        a: u32,
        k: u8,
        c: u32,
        j: u8,
        row: u8,
    },
    /// Module Leibniz for the ring class `(a, k)` and the module class `(b, j)`.
    Module {
        node: u16,
        a: u32,
        k: u8,
        b: u32,
        j: u8,
        row: u8,
    },
    /// Naturality along an edge at the class `(c, k)` of its source.
    Map { edge: u16, c: u32, k: u8, row: u8 },
    /// A seed (0), a backward fact (1) or a d2 file entry (2) at `(n, f)`.
    Fact {
        kind: u8,
        node: u16,
        n: i32,
        f: i32,
        k: u8,
        row: u8,
    },
    /// The secondary-product coupling of pages r - 1 and r for `h_i` and the class at `(n, f)`.
    Coupling {
        node: u16,
        n: i32,
        f: i32,
        i: u8,
        row: u8,
    },
}

/// Where generated equations go: into the propagator, or kept with their origins.
enum Sink {
    Prop(crate::propagate::Propagator),
    Keep(Vec<Vec<u32>>, Vec<Origin>),
}

impl Sink {
    fn add(&mut self, eqs: Vec<(Vec<u32>, Origin)>) {
        match self {
            Sink::Prop(p) => {
                let eqs: Vec<Vec<u32>> = eqs.into_iter().map(|(e, _)| e).collect();
                p.absorb(&eqs);
            }
            Sink::Keep(e, o) => {
                for (eq, or) in eqs {
                    e.push(eq);
                    o.push(or);
                }
            }
        }
    }
}

/// Source bidegrees whose equations are generated together before they are absorbed: larger
/// chunks parallelise better, smaller ones hold fewer equations in memory at once.
const CHUNK: usize = 64;

/// An echelon form keyed by lowest set bit; each row remembers which basis vectors it combines.
#[derive(Default, Clone)]
struct Ech {
    rows: FxHashMap<u32, (V, V)>,
}

impl Ech {
    /// Reduces `v` to its normal form (no pivot bit set); returns the remainder and the basis
    /// coefficients used. Every row's bits lie at or above its pivot, so clearing pivots in
    /// increasing order never sets a pivot already passed.
    fn reduce(&self, mut v: V) -> (V, V) {
        let mut coef = 0;
        let mut passed: V = 0;
        loop {
            let rest = v & !passed;
            if rest == 0 {
                break;
            }
            let p = rest.trailing_zeros();
            match self.rows.get(&p) {
                Some(&(row, c)) => {
                    v ^= row;
                    coef ^= c;
                }
                None => passed |= 1 << p,
            }
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
    /// source global basis id -> image over the target basis (absent: zero)
    img: FxHashMap<u32, V>,
    /// source global basis ids whose image is not known
    unknown: rustc_hash::FxHashSet<u32>,
}

impl Edge {
    /// Image of an E2 vector of source bidegree `b`, unless it involves an unknown image.
    fn apply(&self, x: &Node, b: u32, v: V) -> Option<V> {
        bits(v).try_fold(0, |acc, i| {
            let g = x.gid(b, i);
            (!self.unknown.contains(&g)).then(|| acc ^ self.img.get(&g).copied().unwrap_or(0))
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

/// E2: every class is a certain cycle and nothing is a boundary.
fn e2_pages(node: &Node) -> Pages {
    let page = (0..node.nb() as u32)
        .map(|b| {
            let basis: Vec<V> = (0..node.dim(b)).map(|i| 1 << i).collect();
            Some(Page::new(vec![], vec![], &basis, &basis, true, false))
        })
        .collect();
    Pages::new(2, page)
}

/// The equations `d2(x) = t` for the classes whose d2 a file gives, on the E2 page.
fn d2_fact_equations(
    ni: usize,
    node: &Node,
    e: &Pages,
    d2: &FxHashMap<u32, V>,
    one: u32,
) -> Vec<(Vec<u32>, Origin)> {
    let mut out = vec![];
    for b in 0..node.nb() as u32 {
        if e.off[b as usize] == u32::MAX {
            continue;
        }
        for i in 0..node.dim(b) {
            let Some(&t) = d2.get(&node.gid(b, i)) else {
                continue;
            };
            for l in 0..e.stride[b as usize] as usize {
                let mut eq = vec![e.var(b, i, l)];
                if t >> l & 1 == 1 {
                    eq.push(one);
                }
                let (n, f) = node.deg(b);
                let (node, k, row) = (ni as u16, i as u8, l as u8);
                out.push((
                    eq,
                    Origin::Fact {
                        kind: 2,
                        node,
                        n,
                        f,
                        k,
                        row,
                    },
                ));
            }
        }
    }
    if !out.is_empty() {
        eprintln!("{}: {} d2 coordinates from files", node.name, out.len());
    }
    out
}

/// d2 of single module classes known modulo `i_*` of the sphere's d2 image out of their bidegree:
/// rows `stem,Adams_filtration,index,target` of `path` (target over the module basis at
/// `(stem - 1, filtration + 2)`). These are values that pin the cofiber's pages without being
/// exact on cochains (the cofiber d2 on lifts of non-d2-cycles, from `h_j`'s side).
fn modb_equations(
    ni: usize,
    mn: &Node,
    e: &Pages,
    rn: &Node,
    d2: &FxHashMap<u32, V>,
    path: &str,
    one: u32,
) -> Vec<(Vec<u32>, Origin)> {
    let Alg::Module(m, _) = mn.alg else {
        return vec![];
    };
    let istar = |n: i32, f: i32, v: V| -> V {
        let Some(sb) = rn.get(n, f) else {
            return 0;
        };
        bits(v).fold(0, |acc, l| {
            acc ^ m.istar.get(&rn.gid(sb, l)).copied().unwrap_or(0) as V
        })
    };
    let mut out = vec![];
    let (mut used, mut skipped) = (0, 0);
    for st in read_statements(path) {
        let (n, f) = (st.n, st.f);
        let Some(b) = mn.get(n, f) else {
            continue;
        };
        if e.off[b as usize] == u32::MAX || mn.get(n - 1, f + 2).is_none() {
            continue;
        }
        let mut ind = Ech::default();
        let mut known = true;
        // a d2 file has no rows where the sphere's d2 target is empty
        if let (Some(ab), Some(_)) = (rn.get(n, f), rn.get(n - 1, f + 2)) {
            for k in 0..rn.dim(ab) {
                match d2.get(&rn.gid(ab, k)) {
                    Some(&w) => {
                        ind.insert(istar(n - 1, f + 2, w), 0);
                    }
                    None => known = false,
                }
            }
        }
        if !known {
            skipped += 1;
            continue;
        }
        let rhs = ind.reduce(st.tgt).0;
        let cols: Vec<V> = (0..e.stride[b as usize] as usize)
            .map(|l| ind.reduce(1 << l).0)
            .collect();
        used += 1;
        for cc in 0..cols.len() {
            let mut eq = vec![];
            for k in bits(st.src) {
                for (l, &col) in cols.iter().enumerate() {
                    if col >> cc & 1 == 1 {
                        eq.push(e.var(b, k, l));
                    }
                }
            }
            if rhs >> cc & 1 == 1 {
                eq.push(one);
            }
            let eq = reduce_eq(eq);
            if !eq.is_empty() {
                let (node, k, row) = (ni as u16, st.src.trailing_zeros() as u8, cc as u8);
                out.push((
                    eq,
                    Origin::Fact {
                        kind: 3,
                        node,
                        n,
                        f,
                        k,
                        row,
                    },
                ));
            }
        }
    }
    eprintln!(
        "{}: {used} d2 values mod i_*(B2) from {path} ({skipped} skipped), {} equations",
        mn.name,
        out.len()
    );
    out
}

/// The d2 of a cofiber `C(h_i)` (a module over the first node with `shift = 2^i`) on lifts of
/// its top cell, from the sphere alone: for `y` a d2-cycle with `y h_i = lambda L` in Cλ²
/// (vanishing E2 product), every module class `ȳ` with `p_* ȳ = y` has `d2(ȳ) = i_*(L)` modulo
/// `i_*` of the sphere's d2 out of the bidegree of `ȳ`. Changing the lift of `y` moves `L` by
/// `h_i w`, which `i_*` kills; changing `ȳ` by `i_*(a)` moves `d2(ȳ)` by `i_*(d2 a)`.
fn cofiber_d2_equations(
    ni: usize,
    mn: &Node,
    e: &Pages,
    rn: &Node,
    d2: &FxHashMap<u32, V>,
    sec: &[Secondary],
    one: u32,
) -> Vec<(Vec<u32>, Origin)> {
    let Alg::Module(m, _) = mn.alg else {
        return vec![];
    };
    if m.shift <= 0 || m.shift & (m.shift - 1) != 0 {
        return vec![];
    }
    let hi = m.shift.trailing_zeros();
    // i_* of a sphere vector at (n, f), if every image it needs is known
    let istar = |n: i32, f: i32, v: V| -> V {
        let Some(sb) = rn.get(n, f) else {
            return 0;
        };
        bits(v).fold(0, |acc, l| {
            acc ^ m.istar.get(&rn.gid(sb, l)).copied().unwrap_or(0) as V
        })
    };
    let mut out = vec![];
    let mut used = 0;
    for s in sec.iter().filter(|s| s.i == hi) {
        let (n, f) = (s.n + m.shift, s.f);
        let Some(b) = mn.get(n, f) else {
            continue;
        };
        if e.off[b as usize] == u32::MAX || mn.get(n - 1, f + 2).is_none() {
            continue;
        }
        // a module class ȳ with p_* ȳ = y
        let mut pre = Ech::default();
        for j in 0..mn.dim(b) {
            let img = m.pstar.get(&mn.gid(b, j)).copied().unwrap_or(0) as V;
            pre.insert(img, 1 << j);
        }
        let Some(c) = pre.project(s.y) else {
            continue;
        };
        // the ambiguity: i_* d2 of the sphere at (n, f); every d2 there must be known
        let mut ind = Ech::default();
        let mut known = true;
        if let Some(ab) = rn.get(n, f) {
            for k in 0..rn.dim(ab) {
                match d2.get(&rn.gid(ab, k)) {
                    Some(&w) => {
                        ind.insert(istar(n - 1, f + 2, w), 0);
                    }
                    None => known = false,
                }
            }
        }
        if !known {
            continue;
        }
        let rhs = ind.reduce(istar(n - 1, f + 2, s.l)).0;
        let cols: Vec<V> = (0..e.stride[b as usize] as usize)
            .map(|l| ind.reduce(1 << l).0)
            .collect();
        used += 1;
        for cc in 0..cols.len() {
            let mut eq = vec![];
            for k in bits(c) {
                for (l, &col) in cols.iter().enumerate() {
                    if col >> cc & 1 == 1 {
                        eq.push(e.var(b, k, l));
                    }
                }
            }
            if rhs >> cc & 1 == 1 {
                eq.push(one);
            }
            let eq = reduce_eq(eq);
            if !eq.is_empty() {
                let (node, k, row) = (ni as u16, c.trailing_zeros() as u8, cc as u8);
                out.push((
                    eq,
                    Origin::Fact {
                        kind: 3,
                        node,
                        n,
                        f,
                        k,
                        row,
                    },
                ));
            }
        }
    }
    eprintln!(
        "{}: d2 on top-cell lifts from the sphere's secondary products: {used} lifts, {} \
         equations",
        mn.name,
        out.len()
    );
    out
}

/// E3 as bounds from E2 and what is known of d2: a class whose d2 is not in the files may have
/// any value.
fn bounds_e3(node: &Node, d2: &FxHashMap<u32, V>) -> Pages {
    let nb = node.nb();
    let e2 = e2_pages(node);
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
                // no target coordinates means every class there is a possible boundary; d_r is
                // then certainly zero only if they are all certain boundaries
                Some(Some(t)) if cur.tdim(t) == 0 && cur.pg(t).exact => (pg.reps.clone(), false),
                Some(Some(t)) if cur.tdim(t) == 0 => (vec![], !pg.reps.is_empty()),
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
fn ring_equations(
    ni: usize,
    node: &Node,
    e3: &Pages,
    gens: bool,
    sink: &mut dyn FnMut(Vec<(Vec<u32>, Origin)>),
) {
    let Alg::Ring(ext, p) = node.alg else {
        unreachable!()
    };
    let t = Instant::now();
    let ablate_zero = std::env::var("E3_ABLATE_ZERO").is_ok_and(|v| v == "1");
    let bids: Vec<u32> = (0..node.nb() as u32).filter(|&b| e3.dim(b) > 0).collect();
    // on E2 the source classes are the basis vectors, whose indecomposability is known
    let indec = |b: u32, x: V| ext.indec[node.gid(b, x.trailing_zeros() as usize) as usize];
    let mut count = 0;
    for chunk in bids.chunks(CHUNK) {
        let eqs: Vec<(Vec<u32>, Origin)> = chunk
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
                            if gens && !indec(a, x) && !indec(c, y) {
                                continue;
                            }
                            // per target coordinate, the variables of the equation
                            let mut terms: Vec<Vec<u32>> = vec![vec![]; m];
                            // the product as a combination of source classes, if it is certain
                            let pr = match s {
                                None => 0,
                                Some(s) => {
                                    match e3.pg(s).src.project(mul_vec(ext, p, a, x, c, y)) {
                                        Some(pr) => pr,
                                        None => continue,
                                    }
                                }
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
                            for (row, tr) in terms.into_iter().enumerate() {
                                let e = reduce_eq(tr);
                                if !e.is_empty() {
                                    let (k, j, row) = (k as u8, j as u8, row as u8);
                                    let node = ni as u16;
                                    out.push((
                                        e,
                                        Origin::Ring {
                                            node,
                                            a,
                                            k,
                                            c,
                                            j,
                                            row,
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }
                out
            })
            .collect();
        count += eqs.len();
        sink(eqs);
    }
    eprintln!(
        "{}: E{} Leibniz: {count} nontrivial equations, {:.1}s",
        node.name,
        e3.r,
        t.elapsed().as_secs_f64()
    );
}

/// The module Leibniz rule on E_r, d_r[a m] = d_r[a] m + a d_r[m] for every source class `a` of
/// the ring and `m` of the module, projected onto the target bounds. As for a ring, each term
/// needs the target of its unknown to have no boundary uncertainty, and every product it uses
/// must be known (an unknown product drops the pair).
fn module_equations(
    ni: usize,
    mn: &Node,
    em: &Pages,
    rn: &Node,
    er: &Pages,
    gens: bool,
    sink: &mut dyn FnMut(Vec<(Vec<u32>, Origin)>),
) {
    let (Alg::Module(m, _), Alg::Ring(ext, _)) = (&mn.alg, &rn.alg) else {
        unreachable!()
    };
    let t = Instant::now();
    let abids: Vec<u32> = (0..rn.nb() as u32).filter(|&a| er.dim(a) > 0).collect();
    let bbids: Vec<u32> = (0..mn.nb() as u32).filter(|&b| em.dim(b) > 0).collect();
    let r = em.r;
    let (mut count, mut n_dropped) = (0, 0);
    for chunk in abids.chunks(CHUNK) {
        let (eqs, dropped): (Vec<Vec<(Vec<u32>, Origin)>>, Vec<usize>) = chunk
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
                        // on E2 the ring classes are basis vectors
                        if gens && !ext.indec[rn.gid(a, x.trailing_zeros() as usize) as usize] {
                            continue;
                        }
                        'pair: for (j, &y) in em.pg(b).reps.iter().enumerate() {
                            let mut terms: Vec<Vec<u32>> = vec![vec![]; dm];
                            if let Some(s) = s {
                                let Some(pr) =
                                    act(m, a, x, b, y).and_then(|v| em.pg(s).src.project(v))
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
                            for (row, tr) in terms.into_iter().enumerate() {
                                let e = reduce_eq(tr);
                                if !e.is_empty() {
                                    let (k, j, row) = (k as u8, j as u8, row as u8);
                                    let node = ni as u16;
                                    out.push((
                                        e,
                                        Origin::Module {
                                            node,
                                            a,
                                            k,
                                            b,
                                            j,
                                            row,
                                        },
                                    ));
                                }
                            }
                        }
                    }
                }
                (out, dropped)
            })
            .unzip();
        let eqs: Vec<(Vec<u32>, Origin)> = eqs.into_iter().flatten().collect();
        count += eqs.len();
        n_dropped += dropped.iter().sum::<usize>();
        sink(eqs);
    }
    eprintln!(
        "{}: E{r} module Leibniz over {}: {count} nontrivial equations, {n_dropped} pairs \
         dropped, {:.1}s",
        mn.name,
        rn.name,
        t.elapsed().as_secs_f64()
    );
}

/// Naturality d_r(f x) = f(d_r x) along a filtration-0 map, for every source class `x`,
/// projected onto the target bounds of the image. f(d_r x) is only well defined when the target
/// of x has no boundary uncertainty (f of a possible boundary need not be a possible boundary of
/// the image); f x must be a certain cycle of the image to name its d_r.
fn map_equations(
    ei: usize,
    edge: &Edge,
    xn: &Node,
    ex: &Pages,
    yn: &Node,
    ey: &Pages,
) -> Vec<(Vec<u32>, Origin)> {
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
                .map(|&z| edge.apply(xn, t, z).and_then(|v| ey.pg(yt).tgt.project(v)))
                .collect(),
        };
        let Some(t_images) = t_images else {
            dropped += ex.dim(c);
            continue;
        };
        for (k, &x) in ex.pg(c).reps.iter().enumerate() {
            let src = match yb {
                None => 0,
                Some(yb) => match edge.apply(xn, c, x).and_then(|v| ey.pg(yb).src.project(v)) {
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
                    let (edge, k, row) = (ei as u16, k as u8, row as u8);
                    out.push((e, Origin::Map { edge, c, k, row }));
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

/// A secondary product `y h_i = E + lambda L` with `E = 0`: `y` (an E2 vector at `(n, f)`) is a
/// d2-cycle, and `L` is the E2 vector at `(n + 2^i - 1, f + 2)` (modulo the image of d2).
struct Secondary {
    n: i32,
    f: i32,
    y: V,
    i: u32,
    l: V,
}

/// Reads the lines `[y] [h_i] = [E] + λ [L]` of `secondary_product_easy` (y a sum of
/// `x_(n, s, k)`) whose `E` vanishes; the other lines carry no coupling.
fn read_secondary(path: &str) -> Vec<Secondary> {
    let vec = |s: &str| -> V {
        s.split(',')
            .map(str::trim)
            .enumerate()
            .filter(|(_, x)| *x == "1")
            .fold(0, |acc, (k, _)| acc | 1 << k)
    };
    let mut out = vec![];
    for line in BufReader::new(File::open(path).unwrap()).lines() {
        let line = line.unwrap();
        if !line.starts_with('[') {
            continue;
        }
        // [y] [hI] = [E] + λ [L]
        let (lhs, rhs) = line.split_once(" = ").unwrap();
        let (ys, hs) = lhs.rsplit_once("] [h").unwrap();
        let i: u32 = hs.trim_end_matches(']').parse().unwrap();
        let (e, l) = rhs.split_once(" + λ ").unwrap();
        if vec(e.trim_matches(|c| c == '[' || c == ']')) != 0 {
            continue;
        }
        let l = vec(l.trim_matches(|c| c == '[' || c == ']'));
        let (mut n, mut f, mut y) = (0, 0, 0);
        for term in ys.trim_start_matches('[').split(" + ") {
            let w: Vec<i32> = term
                .trim_start_matches("x_(")
                .trim_end_matches(')')
                .split(',')
                .map(|x| x.trim().parse().unwrap())
                .collect();
            (n, f) = (w[0], w[1]);
            y ^= 1 << w[2];
        }
        // SEC_I_MIN: skip products with h_i for i below it
        let i_min: u32 = std::env::var("SEC_I_MIN").map_or(0, |v| v.parse().unwrap());
        if i >= i_min {
            out.push(Secondary { n, f, y, i, l });
        }
    }
    eprintln!(
        "secondary products: {} with vanishing E2 product",
        out.len()
    );
    out
}

/// `L` for the lift of `h_i` that comes from homotopy: the table's `L` is for the standard lift in
/// Cλ², which can differ from it by `lambda c_i` with `c_i` at `(2^i - 1, 2)`; `SEC_LIFT` (bit `i`:
/// `c_i` is the class there) chooses `c_i`, changing `L` by `c_i y`.
fn true_l(node: &Node, s: &Secondary) -> V {
    let Alg::Ring(ext, p) = node.alg else {
        return s.l;
    };
    let mask: u32 = std::env::var("SEC_LIFT").map_or(0, |v| v.parse().unwrap());
    if mask >> s.i & 1 == 0 {
        return s.l;
    }
    match (node.get((1 << s.i) - 1, 2), node.get(s.n, s.f)) {
        (Some(cb), Some(b)) => s.l ^ mul_vec(ext, p, cb, 1, b, s.y),
        _ => s.l,
    }
}

/// The bidegree of `h_i`.
fn h_bidegree(node: &Node, i: u32) -> Option<u32> {
    node.get((1 << i) - 1, 1)
}

/// The value of d_r on a combination `c` of the page classes at `b`, if determined.
fn determined(dr: &FxHashMap<u32, Vec<Val>>, b: u32, c: V) -> Option<V> {
    let vs = dr.get(&b).map_or(&[][..], |v| &v[..]);
    bits(c).try_fold(0, |acc, k| {
        let v = vs.get(k)?;
        v.dirs.is_empty().then_some(acc ^ v.part)
    })
}

/// The ambiguity of `d_{r-1}(L)` (page r - 1 target coordinates at `tb`): `L` is only defined
/// modulo `h_i` times E2 at `(n, f + 1)` (the choice of lift of `y`), so it is d_{r-1} of every
/// certain cycle in that span. `None` when it cannot be bounded: page r - 1 unknown or uncertain
/// at `L`'s bidegree, or one of those d_{r-1} not determined.
fn lift_ambiguity(
    node: &Node,
    ep: &Pages,
    drp: &FxHashMap<u32, Vec<Val>>,
    s: &Secondary,
    hb: u32,
) -> Option<Ech> {
    let Alg::Ring(ext, p) = node.alg else {
        return None;
    };
    let mut ind = Ech::default();
    let hn = (1 << s.i) - 1;
    let (Some(wb), Some(lb)) = (node.get(s.n, s.f + 1), node.get(s.n + hn, s.f + 2)) else {
        return Some(ind);
    };
    let lp = ep.page[lb as usize].as_ref()?;
    if lp.uncertain {
        return None;
    }
    let images: Vec<V> = (0..node.dim(wb))
        .map(|j| mul_vec(ext, p, hb, 1, wb, 1 << j))
        .collect();
    let rest: Vec<V> = images.iter().map(|&v| lp.src.reduce(v).0).collect();
    for c in kernel(&rest) {
        let v = bits(c).fold(0, |acc, j| acc ^ images[j]);
        let coef = lp.src.project(v)?;
        if coef == 0 || ep.off[lb as usize] == u32::MAX {
            continue;
        }
        ind.insert(determined(drp, lb, coef)?, 0);
    }
    Some(ind)
}

/// For a secondary product `y h_i = lambda L` and a class `y` surviving to E_r with `d_r y = z`,
/// the lambda-Bockstein gives `h_i z = d_{r-1}(L)` in E_{r-1} at `(n + 2^i - 2, f + r + 1)`,
/// modulo `h_i` times the image of d_{r-1} from `(n, f + 1)` (the choice of lift of `y`). With
/// d_{r-1} known on page `ep`, these are affine equations on the d_r unknowns of `e`. Only used
/// where every page involved is exact and every d_{r-1} value is determined.
fn coupling_equations(
    ni: usize,
    node: &Node,
    e: &Pages,
    ep: &Pages,
    drp: &FxHashMap<u32, Vec<Val>>,
    sec: &[Secondary],
    one: u32,
) -> Vec<(Vec<u32>, Origin)> {
    let Alg::Ring(ext, p) = node.alg else {
        return vec![];
    };
    let r = e.r;
    if std::env::var("SEC_RMAX").is_ok_and(|v| r > v.parse::<i32>().unwrap()) {
        return vec![];
    }
    let mut out = vec![];
    let mut used = 0;
    for s in sec {
        let Some(b) = node.get(s.n, s.f) else {
            continue;
        };
        if e.off[b as usize] == u32::MAX {
            continue;
        }
        let Some(coef) = e.page[b as usize]
            .as_ref()
            .and_then(|pg| pg.src.project(s.y))
        else {
            continue;
        };
        if coef == 0 {
            continue;
        }
        let Some(Some(zb)) = target(node, e, b) else {
            continue;
        };
        if !e.pg(zb).exact || ep.page[zb as usize].is_none() {
            continue;
        }
        let Some(hb) = h_bidegree(node, s.i) else {
            continue;
        };
        let hn = (1 << s.i) - 1;
        let Some(Some(tb)) = known(node, ep, s.n + hn - 1, s.f + r + 1) else {
            continue;
        };
        let tp = ep.pg(tb);
        if !tp.exact {
            continue;
        }
        // d_{r-1}(L), in page r - 1 target coordinates at tb
        let l = true_l(node, s);
        let rhs = if l == 0 {
            0
        } else {
            let Some(lb) = node.get(s.n + hn, s.f + 2) else {
                continue;
            };
            let Some(cl) = ep.page[lb as usize]
                .as_ref()
                .and_then(|pg| pg.src.project(l))
            else {
                continue;
            };
            if ep.off[lb as usize] == u32::MAX {
                0
            } else {
                let Some(v) = determined(drp, lb, cl) else {
                    continue;
                };
                v
            }
        };
        let Some(ind) = lift_ambiguity(node, ep, drp, s, hb) else {
            continue;
        };
        // h_i times each target coordinate of d_r
        let lhs: Option<Vec<V>> = e
            .pg(zb)
            .treps
            .iter()
            .map(|&z| tp.tgt.project(mul_vec(ext, p, hb, 1, zb, z)))
            .map(|c| c.map(|c| ind.reduce(c).0))
            .collect();
        let Some(lhs) = lhs else { continue };
        let rhs = ind.reduce(rhs).0;
        used += 1;
        for c in 0..tp.treps.len() {
            let mut eq: Vec<u32> = vec![];
            for k in bits(coef) {
                for (l, &h) in lhs.iter().enumerate() {
                    if h >> c & 1 == 1 {
                        eq.push(e.var(b, k, l));
                    }
                }
            }
            if rhs >> c & 1 == 1 {
                eq.push(one);
            }
            let eq = reduce_eq(eq);
            if std::env::var("SEC_DEBUG")
                .is_ok_and(|d| eq == [one] || d == format!("{},{}", s.n, s.f))
            {
                eprintln!(
                    "SECDBG ({},{}) y={:b} h{} L={l:b} coef={coef:b} zb={:?} treps={:?} lhs={lhs:?} \
                     rhs={rhs:b} row={c} ind={:?} tdim={}",
                    s.n,
                    s.f,
                    s.y,
                    s.i,
                    node.deg(zb),
                    e.pg(zb).treps,
                    ind.rows,
                    tp.treps.len()
                );
            }
            if !eq.is_empty() {
                let (node, i, row) = (ni as u16, s.i as u8, c as u8);
                out.push((
                    eq,
                    Origin::Coupling {
                        node,
                        n: s.n,
                        f: s.f,
                        i,
                        row,
                    },
                ));
            }
        }
    }
    eprintln!(
        "{}: d{r} secondary coupling with d{}: {used} products used, {} equations",
        node.name,
        r - 1,
        out.len()
    );
    out
}

/// The backward half of [`coupling_equations`]: a forced `d_r y = z` gives `d_{r-1}(L) = h_i z`
/// on page r - 1, as a fact `(stem, filtration, L, h_i z)`. Only where nothing on page r - 1
/// leaves `(n, f + 1)` (no ambiguity from the lift of `y`).
fn coupling_facts(
    node: &Node,
    e: &Pages,
    ep: &Pages,
    drp: &FxHashMap<u32, Vec<Val>>,
    dr: &FxHashMap<u32, Vec<Val>>,
    sec: &[Secondary],
) -> Vec<(i32, i32, V, V)> {
    let Alg::Ring(ext, p) = node.alg else {
        return vec![];
    };
    let mut out = vec![];
    for s in sec {
        let l = true_l(node, s);
        if l == 0 {
            continue;
        }
        let Some(b) = node.get(s.n, s.f) else {
            continue;
        };
        if e.off[b as usize] == u32::MAX {
            continue;
        }
        let Some(coef) = e.page[b as usize]
            .as_ref()
            .and_then(|pg| pg.src.project(s.y))
        else {
            continue;
        };
        if coef == 0 {
            continue;
        }
        let Some(Some(zb)) = target(node, e, b) else {
            continue;
        };
        if !e.pg(zb).exact {
            continue;
        }
        let Some(hb) = h_bidegree(node, s.i) else {
            continue;
        };
        match lift_ambiguity(node, ep, drp, s, hb) {
            Some(ind) if ind.rows.is_empty() => {}
            _ => continue,
        }
        let vs = dr.get(&b).map_or(&[][..], |v| &v[..]);
        let Some(z) = bits(coef).try_fold(0, |acc, k| {
            let v = vs.get(k)?;
            v.dirs.is_empty().then_some(acc ^ v.part)
        }) else {
            continue;
        };
        let ze = bits(z).fold(0, |acc, l| acc ^ e.pg(zb).treps[l]);
        let tgt = mul_vec(ext, p, hb, 1, zb, ze);
        let hn = (1 << s.i) - 1;
        out.push((s.n + hn, s.f + 2, l, tgt));
    }
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
    evaluate_with(nvars, pivot, rows, queries, |_| true)
        .into_iter()
        .map(|(v, _, d)| (v, d))
        .collect()
}

/// Per query: its value at the particular solution, whether it is determined (the same on the
/// whole solution space), and, when `want_deps` asks for them and it is not determined, the free
/// columns (by index among the free columns) of its expression. Determination comes from one
/// fingerprint pass ([`crate::fingerprints`]); a query is reduced exactly only for its
/// dependencies or when the fingerprint and the OR pass disagree.
pub(crate) fn evaluate_with(
    nvars: u32,
    pivot: &[u32],
    rows: &[Vec<u32>],
    queries: &[Vec<u32>],
    want_deps: impl Fn(usize) -> bool + Sync,
) -> Vec<(bool, bool, Vec<u32>)> {
    let t = Instant::now();
    let fp = crate::fingerprints(nvars, pivot, rows);
    let free: Vec<u32> = (0..nvars)
        .filter(|&c| pivot[c as usize] == u32::MAX)
        .collect();
    let out: Vec<(bool, bool, Vec<u32>)> = queries
        .par_iter()
        .enumerate()
        .map(|(qi, q)| {
            let value = q.iter().fold(false, |acc, &v| acc ^ fp.part[v as usize]);
            let reach = q.iter().any(|&v| fp.reach[v as usize]);
            let tag = q.iter().fold(0, |acc, &v| acc ^ fp.tag[v as usize]);
            if !reach {
                return (value, true, vec![]);
            }
            if tag != 0 && !want_deps(qi) {
                return (value, false, vec![]);
            }
            let deps: Vec<u32> = crate::reduce_to_free(nvars, pivot, rows, q)
                .into_iter()
                .filter(|&c| c != nvars)
                .map(|c| free.binary_search(&c).unwrap() as u32)
                .collect();
            (value, deps.is_empty(), deps)
        })
        .collect();
    eprintln!(
        "evaluate: {} queries, {} determined, {:.1}s",
        out.len(),
        out.iter().filter(|x| x.1).count(),
        t.elapsed().as_secs_f64()
    );
    out
}

/// Which coordinate a certificate is asked for.
struct CertSpec {
    r: i32,
    node: usize,
    n: i32,
    f: i32,
    k: usize,
    l: usize,
}

/// Names variables and equation origins of one page of the diagram.
struct Labels<'a, 'b> {
    nodes: &'a [Node<'b>],
    es: &'a [Pages],
    edges: &'a [Edge],
    starts: &'a [u32],
    total: u32,
}

impl Labels<'_, '_> {
    /// `d<r>_<node>(x_(n,f,k))[l]`: coordinate `l` of d_r of class `k` at `(n, f)` (E2 basis index
    /// on page 2, page class otherwise).
    fn var(&self, y: u32) -> String {
        if y == self.total {
            return "1".into();
        }
        let ni = self.starts.partition_point(|&s| s <= y) - 1;
        let e = &self.es[ni];
        let b = (0..e.off.len())
            .filter(|&b| e.off[b] != u32::MAX && e.off[b] <= y)
            .max_by_key(|&b| e.off[b])
            .unwrap();
        let rel = (y - e.off[b]) as usize;
        let stride = e.stride[b] as usize;
        let (n, f) = self.nodes[ni].deg(b as u32);
        format!(
            "d{}_{}(x_({n},{f},{}))[{}]",
            e.r,
            self.nodes[ni].name,
            rel / stride,
            rel % stride
        )
    }

    fn class(&self, ni: u16, b: u32, k: u8) -> String {
        let (n, f) = self.nodes[ni as usize].deg(b);
        format!("x_({n},{f},{k})")
    }

    /// An equation's origin, in the form `scripts/min_certificate.py` weighs.
    fn origin(&self, o: Origin) -> String {
        match o {
            Origin::Ring {
                node,
                a,
                k,
                c,
                j,
                row,
            } => format!(
                "{}: {}*{}[{row}]",
                self.nodes[node as usize].name,
                self.class(node, a, k),
                self.class(node, c, j)
            ),
            Origin::Module {
                node,
                a,
                k,
                b,
                j,
                row,
            } => {
                let Alg::Module(_, ring) = self.nodes[node as usize].alg else {
                    unreachable!()
                };
                format!(
                    "{} module over {}: {}*{}[{row}]",
                    self.nodes[node as usize].name,
                    self.nodes[ring].name,
                    self.class(ring as u16, a, k),
                    self.class(node, b, j)
                )
            }
            Origin::Map { edge, c, k, row } => {
                let e = &self.edges[edge as usize];
                format!(
                    "naturality {}: {}[{row}]",
                    e.name,
                    self.class(e.from as u16, c, k)
                )
            }
            Origin::Fact {
                kind,
                node,
                n,
                f,
                k,
                row,
            } => format!(
                "{} {} x_({n},{f},{k})[{row}]",
                ["seed", "backward", "d2 file", "cofiber secondary"][kind as usize],
                self.nodes[node as usize].name
            ),
            Origin::Coupling { node, n, f, i, row } => format!(
                "secondary {}: h{i} * x_({n},{f},0)[{row}]",
                self.nodes[node as usize].name
            ),
        }
    }
}

/// Prints the certificate for `spec` (shrunk, ordered by stem) and, with `--dump-pool <file>`,
/// writes the pool around it.
#[allow(clippy::too_many_arguments)]
fn certify(
    spec: &CertSpec,
    labels: &Labels,
    eqs: &[Vec<u32>],
    origins: &[Origin],
    pivot: &[u32],
    rows: &[Vec<u32>],
    prov: &[(u32, Vec<u32>)],
    args: &[String],
) {
    let (node, e) = (&labels.nodes[spec.node], &labels.es[spec.node]);
    let b = node
        .get(spec.n, spec.f)
        .expect("certificate: empty bidegree");
    assert!(
        e.off[b as usize] != u32::MAX
            && spec.k < e.dim(b)
            && (spec.l as u32) < e.stride[b as usize],
        "certificate: no such d{} coordinate",
        spec.r
    );
    let x = e.var(b, spec.k, spec.l);
    let (rel, mut set) = crate::cert::certificate(eqs, pivot, rows, prov, x, labels.total);
    let terms: Vec<String> = rel.iter().map(|&y| labels.var(y)).collect();
    println!("relation: {} = 0", terms.join(" + "));
    let stem = |i: u32| match origins[i as usize] {
        Origin::Ring { node, a, c, .. } => {
            let nd = &labels.nodes[node as usize];
            nd.deg(a).0 + nd.deg(c).0
        }
        Origin::Module { node, b, .. } => labels.nodes[node as usize].deg(b).0,
        Origin::Map { edge, c, .. } => {
            let e = &labels.edges[edge as usize];
            labels.nodes[e.from].deg(c).0
        }
        Origin::Fact { n, .. } => n,
        Origin::Coupling { n, .. } => n,
    };
    set.sort_by_key(|&i| stem(i));
    println!("certificate: {} equations", set.len());
    for &i in &set {
        let terms: Vec<String> = eqs[i as usize].iter().map(|&y| labels.var(y)).collect();
        println!(
            "  {}: {} = 0",
            labels.origin(origins[i as usize]),
            terms.join(" + ")
        );
    }
    if let Some(i) = args.iter().position(|a| a == "--dump-pool") {
        let closure = !args.iter().any(|a| a == "--pool-touching");
        crate::cert::dump_pool(
            labels.total,
            eqs,
            &rel,
            &set,
            &args[i + 1],
            closure,
            &|y| labels.var(y),
            &|i| labels.origin(origins[i as usize]),
        );
    }
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
/// d2 files are comma-separated, `-` for none. With `--from-e2` d2 is solved from E2 and the
/// files only add facts to its equations; otherwise E3 is built from them.
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
    // --from-e2: solve d2 too, starting from E2; d2 files are then facts among the equations
    let from_e2 = args.iter().any(|a| a == "--from-e2");
    let first = if from_e2 { 2 } else { 3 };
    let gens_only = args.iter().any(|a| a == "--gens-only");
    // open d2 classes get exact dependence vectors (otherwise every coordinate of an open class
    // is a fresh unknown on E3, as for d2 files with open classes)
    let exact_e2 = args.iter().any(|a| a == "--e2-exact-deps");
    // --secondary <file>: secondary products of the first node (`secondary_product_easy`)
    let secondary = arg("--secondary").map(|p| read_secondary(p));
    // --export-sseq <dir>: write the final pass's determined differentials as Sseq<3>s
    let export = arg("--export-sseq");
    let mut exported: BTreeMap<i32, Exported> = BTreeMap::new();
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
        // a `--solve-map` map lists its open classes in `<map>_status.csv`
        let mut unknown = rustc_hash::FxHashSet::default();
        let status = format!("{}_status.csv", path.trim_end_matches(".csv"));
        if let Ok(file) = File::open(&status) {
            for line in BufReader::new(file).lines().skip(1) {
                let line = line.unwrap();
                let w: Vec<&str> = line.splitn(5, ',').collect();
                if w[3] != "open" {
                    continue;
                }
                let (n, f): (i32, i32) = (w[0].parse().unwrap(), w[1].parse().unwrap());
                if let Some(b) = nodes[x].get(n, f) {
                    unknown.insert(nodes[x].gid(b, w[2].parse().unwrap()));
                }
            }
            eprintln!("{from}->{to}: {} classes with unknown image", unknown.len());
        }
        edges.push(Edge {
            name: format!("{from}->{to}"),
            from: x,
            to: y,
            stem: 0,
            img,
            unknown,
        });
    }
    for (mi, m) in mods.iter().enumerate() {
        let (x, ring) = (spec.rings.len() + mi, ring_idx(&spec.modules[mi].1));
        // a module that is not a two-cell complex has no cell maps
        if m.istar.is_empty() && m.pstar.is_empty() {
            continue;
        }
        edges.push(Edge {
            name: format!("i:{}->{}", names[ring], m.name),
            from: ring,
            to: x,
            stem: 0,
            img: m.istar.clone(),
            unknown: Default::default(),
        });
        edges.push(Edge {
            name: format!("p:{}->{}", m.name, names[ring]),
            from: x,
            to: ring,
            stem: -m.shift,
            img: m.pstar.iter().map(|(&k, &v)| (k, v as V)).collect(),
            unknown: Default::default(),
        });
    }
    let d2s: Vec<FxHashMap<u32, V>> = nodes
        .iter()
        .zip(&d2_paths)
        .map(|(n, paths)| load_d2(n, paths))
        .collect();
    let seeds = page_files(args, "--seed", "--seed-d3", &names);
    // --certificate r:node:n:f:k:l: the equations proving coordinate l of d_r of class k at (n, f)
    let cert = arg("--certificate").map(|s| {
        let w: Vec<&str> = s.split(':').collect();
        let num = |i: usize| w[i].parse::<i32>().unwrap();
        CertSpec {
            r: num(0),
            node: node_idx(w[1]),
            n: num(2),
            f: num(3),
            k: num(4) as usize,
            l: num(5) as usize,
        }
    });
    let check_files = page_files(args, "--check", "--check-d3", &names);

    // Facts d_r(y) = 0 learned from page r + 1, per node (page -> (stem, filtration, E2 vector
    // of y)). They are kept as E2 vectors because the page bases change from one pass to the
    // next.
    let mut back: Vec<FxHashMap<i32, BTreeSet<(i32, i32, V, V)>>> =
        vec![FxHashMap::default(); nodes.len()];
    // d2 is solved again only when page 2 has new backward facts: (their number, d2 per node)
    let mut cache2: Option<(usize, Vec<FxHashMap<u32, Vec<Val>>>)> = None;
    for pass in 1.. {
        let t_pass = Instant::now();
        let facts_before: usize = back.iter().flat_map(|b| b.values()).map(|s| s.len()).sum();
        println!("pass {pass}");
        let mut es: Vec<Pages> = nodes
            .iter()
            .zip(&d2s)
            .map(|(n, d2)| {
                if from_e2 {
                    e2_pages(n)
                } else if from_bounds {
                    bounds_e3(n, d2)
                } else {
                    build_e3(n, d2)
                }
            })
            .collect();
        // the previous page and its d_r, for the secondary coupling
        let mut prev: Option<(Vec<Pages>, Vec<FxHashMap<u32, Vec<Val>>>)> = None;
        loop {
            let r = es[0].r;
            let facts2: usize = back.iter().filter_map(|b| b.get(&2)).map(|s| s.len()).sum();
            if r == 2 && last > 2 {
                if let Some((n2, dr)) = &cache2 {
                    if *n2 == facts2 {
                        eprintln!("d2: no new page-2 facts, reusing the d2 solution");
                        // the variable layout the coupling reads on the previous page
                        let mut total = 0;
                        for (n, e) in nodes.iter().zip(es.iter_mut()) {
                            total = assign_vars(n, e, total);
                        }
                        let next = nodes
                            .iter()
                            .zip(&es)
                            .zip(dr)
                            .map(|((n, e), d)| next_page(n, e, d))
                            .collect();
                        prev = Some((std::mem::replace(&mut es, next), dr.clone()));
                        continue;
                    }
                }
            }
            let mut total = 0;
            let mut starts = vec![];
            for (n, e) in nodes.iter().zip(es.iter_mut()) {
                starts.push(total);
                total = assign_vars(n, e, total);
            }
            // PAGE_DUMP=<prefix>: the first node's bounds per bidegree, to <prefix>_E<r>.csv
            if let Ok(prefix) = std::env::var("PAGE_DUMP") {
                use std::io::Write;
                let mut w =
                    std::io::BufWriter::new(File::create(format!("{prefix}_E{r}.csv")).unwrap());
                writeln!(w, "stem,filtration,dim,known,reps,treps,exact,uncertain").unwrap();
                for b in 0..nodes[0].nb() as u32 {
                    let (n, f) = nodes[0].deg(b);
                    let d = nodes[0].dim(b);
                    match es[0].page[b as usize].as_ref() {
                        None => writeln!(w, "{n},{f},{d},0,0,0,0,1").unwrap(),
                        Some(pg) => writeln!(
                            w,
                            "{n},{f},{d},1,{},{},{},{}",
                            pg.reps.len(),
                            pg.treps.len(),
                            pg.exact as u8,
                            pg.uncertain as u8
                        )
                        .unwrap(),
                    }
                }
            }
            // PAGE_DEBUG=node:n:f,...: the bounds at those bidegrees
            if let Ok(spec) = std::env::var("PAGE_DEBUG") {
                for w in spec.split(',') {
                    let w: Vec<&str> = w.split(':').collect();
                    let Some(ni) = nodes.iter().position(|x| x.name == w[0]) else {
                        continue;
                    };
                    let (n, f) = (w[1].parse().unwrap(), w[2].parse().unwrap());
                    let Some(b) = nodes[ni].get(n, f) else {
                        eprintln!("PAGEDBG E{r} {} ({n},{f}): not in range/zero", w[0]);
                        continue;
                    };
                    match es[ni].page[b as usize].as_ref() {
                        None => eprintln!("PAGEDBG E{r} {} ({n},{f}): unknown", w[0]),
                        Some(pg) => eprintln!(
                            "PAGEDBG E{r} {} ({n},{f}) dim {}: reps {:?} treps {:?} bcert {:?} \
                             bmax {:?} exact {} uncertain {}",
                            w[0],
                            nodes[ni].dim(b),
                            pg.reps,
                            pg.treps,
                            pg.bcert,
                            pg.bmax,
                            pg.exact,
                            pg.uncertain
                        ),
                    }
                }
            }
            // equations are absorbed as they are generated and never stored together, unless a
            // certificate on this page needs each one with its origin
            let cert_here = cert.as_ref().filter(|c| c.r == r);
            let mut sink = if cert_here.is_some() {
                Sink::Keep(vec![], vec![])
            } else {
                Sink::Prop(crate::propagate::Propagator::new(total))
            };
            let gens = r == 2 && gens_only;
            for (ni, (n, e)) in nodes.iter().zip(&es).enumerate() {
                match n.alg {
                    Alg::Ring(..) => ring_equations(ni, n, e, gens, &mut |c| sink.add(c)),
                    Alg::Module(_, ring) => {
                        module_equations(ni, n, e, &nodes[ring], &es[ring], gens, &mut |c| {
                            sink.add(c)
                        })
                    }
                }
            }
            for (ei, edge) in edges.iter().enumerate() {
                let (x, y) = (edge.from, edge.to);
                sink.add(map_equations(
                    ei, edge, &nodes[x], &es[x], &nodes[y], &es[y],
                ));
            }
            if r == 2 {
                for (ni, ((n, e), d2)) in nodes.iter().zip(&es).zip(&d2s).enumerate() {
                    sink.add(d2_fact_equations(ni, n, e, d2, total));
                }
                // COF_D2_MODB=name:path,...: module d2 values known mod i_*(B2)
                if let Ok(spec) = std::env::var("COF_D2_MODB") {
                    for item in spec.split(',') {
                        let (name, path) = item.split_once(':').expect("COF_D2_MODB=name:path");
                        let Some(ni) = nodes.iter().position(|x| x.name == name) else {
                            continue;
                        };
                        if let Alg::Module(_, 0) = nodes[ni].alg {
                            sink.add(modb_equations(
                                ni, &nodes[ni], &es[ni], &nodes[0], &d2s[0], path, total,
                            ));
                        }
                    }
                }
                // COF_D2=0 switches off the cofiber d2 from secondary products
                let cof = std::env::var("COF_D2").map_or(true, |v| v != "0");
                if let (Some(sec), true) = (&secondary, cof) {
                    for (ni, (n, e)) in nodes.iter().zip(&es).enumerate() {
                        if let Alg::Module(_, 0) = n.alg {
                            sink.add(cofiber_d2_equations(
                                ni, n, e, &nodes[0], &d2s[0], sec, total,
                            ));
                        }
                    }
                }
            }
            let fact = |kind: u8, ni: usize, st: &Statement, se: Vec<Vec<u32>>| {
                let (node, n, f) = (ni as u16, st.n, st.f);
                se.into_iter()
                    .enumerate()
                    .map(|(row, e)| {
                        let row = row as u8;
                        (
                            e,
                            Origin::Fact {
                                kind,
                                node,
                                n,
                                f,
                                k: 0,
                                row,
                            },
                        )
                    })
                    .collect::<Vec<_>>()
            };
            for (ni, (n, e)) in nodes.iter().zip(&es).enumerate() {
                let mut n_back = 0;
                for &(sn, sf, src, tgt) in back[ni].get(&r).into_iter().flatten() {
                    let st = Statement {
                        n: sn,
                        f: sf,
                        src,
                        tgt,
                        label: String::new(),
                    };
                    if let Ok(se) = statement_equations(n, e, &st, total) {
                        sink.add(fact(1, ni, &st, se));
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
                    if r > first {
                        let set = back[*ni].entry(r - 1).or_default();
                        set.insert((st.n, st.f, st.src, 0));
                        if st.tgt != 0 {
                            set.insert((st.n - 1, st.f + r, st.tgt, 0));
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
                            sink.add(fact(0, *ni, &st, se));
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
            // SEC_PARTS=fwd|back|both (default both): which half of the coupling to use
            let sec_parts = std::env::var("SEC_PARTS").unwrap_or_else(|_| "both".into());
            if let (Some(sec), Some((ep, drp)), true) = (&secondary, &prev, sec_parts != "back") {
                sink.add(coupling_equations(
                    0, &nodes[0], &es[0], &ep[0], &drp[0], sec, total,
                ));
            }
            let (pivot, rows, n_contradictions) = match sink {
                Sink::Prop(prop) => {
                    let (eqs, short) = prop.finish();
                    let (pivot, rows, _, c) = eliminate(total, &eqs, false);
                    (pivot, rows, c.len() + short)
                }
                Sink::Keep(eqs, origins) => {
                    let (pivot, rows, prov, c) = eliminate(total, &eqs, true);
                    let labels = Labels {
                        nodes: &nodes,
                        es: &es,
                        edges: &edges,
                        starts: &starts,
                        total,
                    };
                    let spec = cert_here.unwrap();
                    certify(spec, &labels, &eqs, &origins, &pivot, &rows, &prov, args);
                    // the equations behind the first few contradictions
                    for &e in c.iter().take(3) {
                        let set = crate::cert::contradiction(&eqs, &pivot, &rows, &prov, e, total);
                        println!("contradiction: {} equations", set.len());
                        for &i in &set {
                            let terms: Vec<String> =
                                eqs[i as usize].iter().map(|&y| labels.var(y)).collect();
                            println!(
                                "  {}: {} = 0",
                                labels.origin(origins[i as usize]),
                                terms.join(" + ")
                            );
                        }
                    }
                    (pivot, rows, c.len())
                }
            };
            let free = pivot.iter().filter(|&&x| x == u32::MAX).count();
            eprintln!("d{r} solution space: dimension {free}, {n_contradictions} contradictions");
            println!("d{r}: {n_contradictions} contradictions");

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
            let vals = evaluate_with(total, &pivot, &rows, &queries, |qi| {
                r > 2 || exact_e2 || qi >= n_class_queries
            });

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
                    // E2 classes are basis vectors: d2 files name them by index
                    let col = if r == 2 { "index" } else { "rep" };
                    writeln!(w, "stem,Adams_filtration,{col},status,target").unwrap();
                    Some(w)
                })
                .collect();
            let mut dr: Vec<FxHashMap<u32, Vec<Val>>> = vec![FxHashMap::default(); nodes.len()];
            // per node: coordinates, forced, forced to 1
            let mut coords = vec![(0, 0, 0); nodes.len()];
            let mut qi = 0;
            let mut fresh = 0u64;
            for &(ni, b, k) in &classes {
                let e = &es[ni];
                let m = e.stride[b as usize] as usize;
                let vs = &vals[qi..qi + m];
                qi += m;
                for (v, det, _) in vs {
                    coords[ni].0 += 1;
                    coords[ni].1 += *det as usize;
                    coords[ni].2 += (*det && *v) as usize;
                }
                let mut v = Val::default();
                if r == 2 && !exact_e2 && vs.iter().any(|x| !x.1) {
                    // an open d2 class: every coordinate is a fresh unknown
                    v.dirs = (0..m).map(|t| 1 << t).collect();
                    v.keys = (0..m)
                        .map(|t| {
                            fresh += 1;
                            64 + fresh * 64 + t as u64
                        })
                        .collect();
                } else {
                    let mut dirs: BTreeMap<u32, V> = BTreeMap::new();
                    for (t, (val, _, deps)) in vs.iter().enumerate() {
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
                }
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
                    let tgt: Vec<String> =
                        vs.iter().map(|(v, ..)| (*v as u8).to_string()).collect();
                    let status = if det { "determined" } else { "open" };
                    if r == 2 {
                        let i = rep.trailing_zeros();
                        writeln!(w, "{n},{f},{i},{status},\"[{}]\"", tgt.join(", ")).unwrap();
                    } else {
                        writeln!(w, "{n},{f},\"{rep:#b}\",{status},\"[{}]\"", tgt.join(","))
                            .unwrap();
                    }
                }
                dr[ni].entry(b).or_default().push(v);
            }
            assert_eq!(qi, n_class_queries);
            if r == 2 {
                cache2 = Some((facts2, dr.clone()));
            }
            // differentials known exactly: determined, into a target without boundary uncertainty
            if export.is_some() {
                let mut page = vec![];
                for (ni, (n, e)) in nodes.iter().zip(&es).enumerate() {
                    for (&b, vs) in &dr[ni] {
                        let Some(Some(t)) = target(n, e, b) else {
                            continue;
                        };
                        let tp = e.pg(t);
                        if !tp.exact {
                            continue;
                        }
                        for (k, v) in vs.iter().enumerate().filter(|(_, v)| v.dirs.is_empty()) {
                            let y = bits(v.part).fold(0, |acc, l| acc ^ tp.treps[l]);
                            page.push((ni, b, e.pg(b).reps[k], y));
                        }
                    }
                }
                exported.insert(r, page);
            }
            // a forced value d_r(x) = y in a target without boundary uncertainty is a true
            // d_{r-1}-cycle: d_{r-1}(y) = 0
            if r > first {
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
                            back[ni].entry(r - 1).or_default().insert((tn, tf, y, 0));
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
                            let holds = se.iter().zip(vs).map(|(eq, (v, det, _))| {
                                let c = eq.contains(&total);
                                (*det, *v == c)
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
            if let (Some(sec), Some((ep, drp)), true) = (&secondary, &prev, sec_parts != "fwd") {
                let facts = coupling_facts(&nodes[0], &es[0], &ep[0], &drp[0], &dr[0], sec);
                if !facts.is_empty() {
                    eprintln!("S: d{r}: {} secondary facts for d{}", facts.len(), r - 1);
                }
                back[0].entry(r - 1).or_default().extend(facts);
            }
            if r >= last {
                break;
            }
            let next = nodes
                .iter()
                .zip(&es)
                .zip(&dr)
                .map(|((n, e), d)| next_page(n, e, d))
                .collect();
            prev = Some((std::mem::replace(&mut es, next), dr));
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
    if let Some(dir) = export {
        export_sseq(dir, &nodes, &exported);
    }
}
