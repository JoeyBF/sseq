//! d2 on the sphere and on two-cell complexes at once: Leibniz on S, the module Leibniz rule on
//! each complex, and naturality along the bottom-cell inclusion and the top-cell collapse.
//!
//! Input: a directory written by ext's `cofiber_export` example. Every class's d2 is an unknown;
//! seeds fix the sphere's d2 from a d2 file. Reports which sphere classes are then forced.
//!
//! Usage: multi <dir> <N> <F> <modules: name:k,...|-> <seed d2.csv> <seed max stem> <report from stem>

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

use rustc_hash::FxHashMap;

type Mask = u64;
type Cls = (i32, i32, usize);

fn parse_vec(s: &str) -> Mask {
    let s = s
        .trim()
        .trim_matches('"')
        .trim_start_matches('[')
        .trim_end_matches(']');
    let mut m = 0;
    for (k, e) in s.split(',').enumerate() {
        if e.trim() == "1" {
            m |= 1 << k;
        }
    }
    m
}

fn lines(path: &str) -> impl Iterator<Item = String> {
    BufReader::new(File::open(path).unwrap_or_else(|e| panic!("{path}: {e}")))
        .lines()
        .skip(1)
        .map(|l| l.unwrap())
}

/// The E2 page of one spectrum, truncated to stem <= N and filtration <= F.
struct Page {
    dims: FxHashMap<(i32, i32), usize>,
}

impl Page {
    fn load(path: &str, n: i32, f: i32) -> Self {
        let mut dims = FxHashMap::default();
        for l in lines(path) {
            let v: Vec<i32> = l.split(',').map(|x| x.parse().unwrap()).collect();
            if v[0] <= n && v[1] <= f && v[2] > 0 {
                dims.insert((v[0], v[1]), v[2] as usize);
            }
        }
        Self { dims }
    }
    fn dim(&self, b: (i32, i32)) -> usize {
        self.dims.get(&b).copied().unwrap_or(0)
    }
}

/// Pairs `(first class, second class) -> product`, from `stem_1,...,index_2,product` rows.
fn load_products(path: &str, n: i32, f: i32, symmetric: bool) -> FxHashMap<(Cls, Cls), Mask> {
    let mut m = FxHashMap::default();
    for l in lines(path) {
        let mut it = l.splitn(7, ',');
        let mut v = [0i32; 6];
        for x in &mut v {
            *x = it.next().unwrap().parse().unwrap();
        }
        if v[0] + v[3] > n || v[1] + v[4] > f {
            continue;
        }
        let a = (v[0], v[1], v[2] as usize);
        let b = (v[3], v[4], v[5] as usize);
        let p = parse_vec(it.next().unwrap());
        m.insert((a, b), p);
        if symmetric {
            m.insert((b, a), p);
        }
    }
    m
}

fn load_map(path: &str) -> FxHashMap<Cls, Mask> {
    let mut m = FxHashMap::default();
    for l in lines(path) {
        let mut it = l.splitn(4, ',');
        let n: i32 = it.next().unwrap().parse().unwrap();
        let f: i32 = it.next().unwrap().parse().unwrap();
        let i: usize = it.next().unwrap().parse().unwrap();
        m.insert((n, f, i), parse_vec(it.next().unwrap()));
    }
    m
}

/// Unknowns: coordinate `k` of `d(x_(n, f, r))` in spectrum `sp`, columns ordered by decreasing
/// stem so that free columns in stems `<= s` count `D_s`. Column `n` is the constant.
struct Vars {
    off: FxHashMap<(usize, i32, i32), u32>,
    n: u32,
    info: Vec<(usize, i32, i32, usize, usize)>,
}

impl Vars {
    fn new(pages: &[&Page]) -> Self {
        let mut blocks = vec![];
        for (sp, p) in pages.iter().enumerate() {
            for (&(n, f), &d) in &p.dims {
                let dp = p.dim((n - 1, f + 2));
                if dp > 0 {
                    blocks.push((sp, n, f, d, dp));
                }
            }
        }
        blocks.sort_by_key(|&(sp, n, f, _, _)| (-n, f, sp));
        let mut off = FxHashMap::default();
        let mut info = vec![];
        let mut total = 0u32;
        for (sp, n, f, d, dp) in blocks {
            off.insert((sp, n, f), total);
            for r in 0..d {
                for k in 0..dp {
                    info.push((sp, n, f, r, k));
                }
            }
            total += (d * dp) as u32;
        }
        Self {
            off,
            n: total,
            info,
        }
    }
    fn var(&self, pages: &[&Page], sp: usize, b: (i32, i32), r: usize, k: usize) -> u32 {
        let dp = pages[sp].dim((b.0 - 1, b.1 + 2));
        self.off[&(sp, b.0, b.1)] + (r * dp + k) as u32
    }
    fn has(&self, sp: usize, b: (i32, i32)) -> bool {
        self.off.contains_key(&(sp, b.0, b.1))
    }
}

fn xor_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn push(eqs: &mut Vec<Vec<u32>>, mut eq: Vec<u32>) {
    eq.sort_unstable();
    let mut red: Vec<u32> = Vec::with_capacity(eq.len());
    for x in eq {
        if red.last() == Some(&x) {
            red.pop();
        } else {
            red.push(x);
        }
    }
    if !red.is_empty() {
        eqs.push(red);
    }
}

fn bits(m: Mask) -> impl Iterator<Item = usize> {
    let mut m = m;
    std::iter::from_fn(move || {
        if m == 0 {
            None
        } else {
            let k = m.trailing_zeros() as usize;
            m &= m - 1;
            Some(k)
        }
    })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, nmax, fmax) = (
        &a[1],
        a[2].parse::<i32>().unwrap(),
        a[3].parse::<i32>().unwrap(),
    );
    let mods: Vec<(String, i32)> = if a[4] == "-" {
        vec![]
    } else {
        a[4].split(',')
            .map(|s| {
                let (n, k) = s.split_once(':').unwrap();
                (n.to_string(), k.parse().unwrap())
            })
            .collect()
    };
    let (seed_path, seed_max, from) = (
        &a[5],
        a[6].parse::<i32>().unwrap(),
        a[7].parse::<i32>().unwrap(),
    );
    let t = Instant::now();

    let s_page = Page::load(&format!("{dir}/rank.csv"), nmax, fmax);
    let s_prod = load_products(&format!("{dir}/products.csv"), nmax, fmax, true);
    let m_pages: Vec<Page> = mods
        .iter()
        .map(|(m, _)| Page::load(&format!("{dir}/{m}_rank.csv"), nmax, fmax))
        .collect();
    let mut pages: Vec<&Page> = vec![&s_page];
    pages.extend(m_pages.iter());
    let v = Vars::new(&pages);
    let cst = v.n;
    let sp_mul = |x: Cls, y: Cls| s_prod.get(&(x, y)).copied().unwrap_or(0);
    let mut eqs: Vec<Vec<u32>> = vec![];

    // Leibniz on S over all pairs A <= B (by (n, f)), with (A + B)' non-zero
    let s_bids: Vec<(i32, i32)> = {
        let mut b: Vec<_> = s_page.dims.keys().copied().collect();
        b.sort();
        b
    };
    for (ia, &a_) in s_bids.iter().enumerate() {
        for &b_ in &s_bids[ia..] {
            let c = (a_.0 + b_.0, a_.1 + b_.1);
            let cp = (c.0 - 1, c.1 + 2);
            if c.0 > nmax || c.1 > fmax || s_page.dim(cp) == 0 {
                continue;
            }
            for i in 0..s_page.dim(a_) {
                for j in 0..s_page.dim(b_) {
                    if a_ == b_ && j < i {
                        continue;
                    }
                    let xy = sp_mul((a_.0, a_.1, i), (b_.0, b_.1, j));
                    for k in 0..s_page.dim(cp) {
                        let mut eq = vec![];
                        for r in bits(xy) {
                            eq.push(v.var(&pages, 0, c, r, k));
                        }
                        if v.has(0, a_) {
                            let ap = (a_.0 - 1, a_.1 + 2);
                            for cc in 0..s_page.dim(ap) {
                                if sp_mul((ap.0, ap.1, cc), (b_.0, b_.1, j)) >> k & 1 == 1 {
                                    eq.push(v.var(&pages, 0, a_, i, cc));
                                }
                            }
                        }
                        if v.has(0, b_) {
                            let bp = (b_.0 - 1, b_.1 + 2);
                            for cc in 0..s_page.dim(bp) {
                                if sp_mul((a_.0, a_.1, i), (bp.0, bp.1, cc)) >> k & 1 == 1 {
                                    eq.push(v.var(&pages, 0, b_, j, cc));
                                }
                            }
                        }
                        push(&mut eqs, eq);
                    }
                }
            }
        }
    }
    let n_s = eqs.len();

    for (mi, (name, shift)) in mods.iter().enumerate() {
        let sp = mi + 1;
        let mp = &m_pages[mi];
        let act = load_products(&format!("{dir}/{name}_action.csv"), nmax, fmax, false);
        let istar = load_map(&format!("{dir}/{name}_istar.csv"));
        let pstar = load_map(&format!("{dir}/{name}_pstar.csv"));
        let am = |x: Cls, m: Cls| act.get(&(x, m)).copied().unwrap_or(0);
        let before = eqs.len();
        let m_bids: Vec<(i32, i32)> = mp.dims.keys().copied().collect();
        // module Leibniz: d(a m) = d(a) m + a d(m)
        for &a_ in &s_bids {
            for &b_ in &m_bids {
                let c = (a_.0 + b_.0, a_.1 + b_.1);
                let cp = (c.0 - 1, c.1 + 2);
                if c.0 > nmax || c.1 > fmax || mp.dim(cp) == 0 {
                    continue;
                }
                for i in 0..s_page.dim(a_) {
                    for j in 0..mp.dim(b_) {
                        let am_ = am((a_.0, a_.1, i), (b_.0, b_.1, j));
                        for k in 0..mp.dim(cp) {
                            let mut eq = vec![];
                            for r in bits(am_) {
                                eq.push(v.var(&pages, sp, c, r, k));
                            }
                            if v.has(0, a_) {
                                let ap = (a_.0 - 1, a_.1 + 2);
                                for cc in 0..s_page.dim(ap) {
                                    if am((ap.0, ap.1, cc), (b_.0, b_.1, j)) >> k & 1 == 1 {
                                        eq.push(v.var(&pages, 0, a_, i, cc));
                                    }
                                }
                            }
                            if v.has(sp, b_) {
                                let bp = (b_.0 - 1, b_.1 + 2);
                                for cc in 0..mp.dim(bp) {
                                    if am((a_.0, a_.1, i), (bp.0, bp.1, cc)) >> k & 1 == 1 {
                                        eq.push(v.var(&pages, sp, b_, j, cc));
                                    }
                                }
                            }
                            push(&mut eqs, eq);
                        }
                    }
                }
            }
        }
        let n_mod = eqs.len() - before;
        // naturality of i: S -> M, d(i x) = i(d x), at B with B' non-zero in M
        for &b_ in &s_bids {
            let bp = (b_.0 - 1, b_.1 + 2);
            if mp.dim(bp) == 0 {
                continue;
            }
            for i in 0..s_page.dim(b_) {
                let ix = istar.get(&(b_.0, b_.1, i)).copied().unwrap_or(0);
                for k in 0..mp.dim(bp) {
                    let mut eq = vec![];
                    if v.has(sp, b_) {
                        for r in bits(ix) {
                            eq.push(v.var(&pages, sp, b_, r, k));
                        }
                    }
                    if v.has(0, b_) {
                        for cc in 0..s_page.dim(bp) {
                            let icc = istar.get(&(bp.0, bp.1, cc)).copied().unwrap_or(0);
                            if icc >> k & 1 == 1 {
                                eq.push(v.var(&pages, 0, b_, i, cc));
                            }
                        }
                    }
                    push(&mut eqs, eq);
                }
            }
        }
        // naturality of p: M -> S^{shift}, d(p m) = p(d m), at B in M, landing at B - shift in S
        for &b_ in &m_bids {
            let t_ = (b_.0 - shift, b_.1);
            let tp = (t_.0 - 1, t_.1 + 2);
            if s_page.dim(tp) == 0 {
                continue;
            }
            let bp = (b_.0 - 1, b_.1 + 2);
            for j in 0..mp.dim(b_) {
                let pm = pstar.get(&(b_.0, b_.1, j)).copied().unwrap_or(0);
                for k in 0..s_page.dim(tp) {
                    let mut eq = vec![];
                    if v.has(0, t_) {
                        for r in bits(pm) {
                            eq.push(v.var(&pages, 0, t_, r, k));
                        }
                    }
                    if v.has(sp, b_) {
                        for cc in 0..mp.dim(bp) {
                            let pcc = pstar.get(&(bp.0, bp.1, cc)).copied().unwrap_or(0);
                            if pcc >> k & 1 == 1 {
                                eq.push(v.var(&pages, sp, b_, j, cc));
                            }
                        }
                    }
                    push(&mut eqs, eq);
                }
            }
        }
        eprintln!(
            "{name}: {} module Leibniz, {} naturality equations",
            n_mod,
            eqs.len() - before - n_mod
        );
    }

    // seeds on S
    let mut seeded = 0;
    for l in lines(seed_path) {
        let mut it = l.splitn(4, ',');
        let n: i32 = it.next().unwrap().parse().unwrap();
        let f: i32 = it.next().unwrap().parse().unwrap();
        let i: usize = it.next().unwrap().parse().unwrap();
        let m = parse_vec(it.next().unwrap());
        if n > seed_max || f > fmax || !v.has(0, (n, f)) {
            continue;
        }
        seeded += 1;
        for k in 0..s_page.dim((n - 1, f + 2)) {
            let mut e = vec![v.var(&pages, 0, (n, f), i, k)];
            if m >> k & 1 == 1 {
                e.push(cst);
            }
            eqs.push(e);
        }
    }
    eprintln!(
        "{} unknowns ({} spectra), {} S Leibniz, {} total equations, {seeded} seeded classes, {:.1}s",
        v.n,
        pages.len(),
        n_s,
        eqs.len(),
        t.elapsed().as_secs_f64()
    );

    // echelon (pivot = smallest column; the constant column sorts last)
    eqs.sort_by_key(|e| e.len());
    let mut pivot = vec![u32::MAX; v.n as usize + 1];
    let mut rows: Vec<Vec<u32>> = vec![];
    let mut contradictions = 0;
    for mut r in eqs {
        while let Some(&c) = r.first() {
            if c == cst {
                contradictions += 1;
                break;
            }
            let p = pivot[c as usize];
            if p == u32::MAX {
                pivot[c as usize] = rows.len() as u32;
                rows.push(r);
                break;
            }
            r = xor_sorted(&r, &rows[p as usize]);
        }
    }
    eprintln!(
        "rank {} of {}, {contradictions} contradictions",
        rows.len(),
        v.n
    );

    // which coordinates depend on free columns; particular solution
    let n = v.n as usize;
    let pivcols: Vec<u32> = (0..v.n)
        .rev()
        .filter(|&c| pivot[c as usize] != u32::MAX)
        .collect();
    let back = |val: &mut [u64]| {
        for &p in &pivcols {
            let r = &rows[pivot[p as usize] as usize];
            val[p as usize] = r[1..].iter().fold(0, |acc, &c| acc ^ val[c as usize]);
        }
    };
    let mut part = vec![0u64; n + 1];
    part[n] = 1;
    back(&mut part);
    let free: Vec<u32> = (0..v.n)
        .filter(|&c| pivot[c as usize] == u32::MAX)
        .collect();
    let mut dep = vec![false; n];
    for chunk in free.chunks(64) {
        let mut val = vec![0u64; n + 1];
        for (l, &f) in chunk.iter().enumerate() {
            val[f as usize] = 1 << l;
        }
        back(&mut val);
        for (d, &x) in dep.iter_mut().zip(&val) {
            *d |= x != 0;
        }
    }
    // per sphere class: determined?
    let mut out =
        std::io::BufWriter::new(File::create(format!("{dir}/multi_solution.csv")).unwrap());
    use std::io::Write;
    writeln!(out, "stem,Adams_filtration,index,status,target").unwrap();
    let mut summary: BTreeMap<i32, (usize, usize)> = BTreeMap::new();
    let mut s_free = BTreeMap::new();
    for (c, &(sp, sn, _, _, _)) in v.info.iter().enumerate() {
        if sp == 0 && pivot[c] == u32::MAX {
            *s_free.entry(sn).or_insert(0usize) += 1;
        }
    }
    let mut keys: Vec<_> = s_page
        .dims
        .keys()
        .copied()
        .filter(|&b| v.has(0, b))
        .collect();
    keys.sort();
    for b in keys {
        let dp = s_page.dim((b.0 - 1, b.1 + 2));
        for r in 0..s_page.dim(b) {
            let vs: Vec<usize> = (0..dp)
                .map(|k| v.var(&pages, 0, b, r, k) as usize)
                .collect();
            let det = vs.iter().all(|&x| !dep[x]);
            let tgt: Vec<String> = vs.iter().map(|&x| (part[x] & 1).to_string()).collect();
            writeln!(
                out,
                "{},{},{r},{},\"[{}]\"",
                b.0,
                b.1,
                if det { "determined" } else { "open" },
                tgt.join(", ")
            )
            .unwrap();
            if b.0 >= from {
                let e = summary.entry(b.0).or_default();
                e.0 += 1;
                e.1 += det as usize;
            }
        }
    }
    let (tc, td) = summary
        .values()
        .fold((0, 0), |(a, b), &(c, d)| (a + c, b + d));
    println!(
        "modules [{}]: S classes in stems >= {from}: {tc}, determined {td}, open {}; S free columns {}",
        mods.iter()
            .map(|(m, _)| m.as_str())
            .collect::<Vec<_>>()
            .join(","),
        tc - td,
        s_free.values().sum::<usize>()
    );
}
