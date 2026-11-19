//! The space of derivations of degree (-1, 2) on the Adams E2 page of the sphere, truncated to
//! stem <= N and Adams filtration <= F: the "global algorithm" of Beauvais-Feisthauer's thesis
//! (ch. 2), solved by sparse F2 elimination instead of SAT.
//!
//! Input: a ring export directory (`rank.csv`, `indecomposables.csv`, `generator_products.csv`
//! in the format of the stem 256 data set). The full product table is rebuilt from the products
//! by indecomposables. Every Leibniz relation `d(xy) = d(x)y + x d(y)` on pairs of basis
//! elements is then reduced into an echelon form whose columns are ordered by DEcreasing stem of
//! the source bidegree. With that order the number of free columns in stems `<= s` is the
//! dimension of `D_s`, the restriction to stems `<= s` of the derivations on the whole range
//! (thesis Figure 8).
//!
//! Usage: derivations <ring dir> <N> <F> [--gens-only]
//!
//! `--gens-only` keeps only the relations where one factor is an indecomposable; these imply the
//! others, so the space must be the same.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

use rayon::prelude::*;
use rustc_hash::FxHashMap;

type Mask = u32;

struct Ext {
    n_max: i32,
    f_max: i32,
    /// (stem, filtration) -> bidegree id
    bid: FxHashMap<(i32, i32), u32>,
    deg: Vec<(i32, i32)>,
    dim: Vec<usize>,
    /// global basis offset of each bidegree
    off: Vec<u32>,
    /// global basis id -> bidegree id
    basis_bid: Vec<u32>,
    /// indecomposable flag per global basis id
    indec: Vec<bool>,
    /// (g, x) -> g * x, g indecomposable, as a mask in the target bidegree
    gen_prod: FxHashMap<(u32, u32), Mask>,
}

impl Ext {
    fn in_range(&self, n: i32, f: i32) -> bool {
        n >= 0 && f >= 0 && n <= self.n_max && f <= self.f_max
    }
    fn get(&self, n: i32, f: i32) -> Option<u32> {
        if !self.in_range(n, f) {
            return None;
        }
        self.bid.get(&(n, f)).copied()
    }
    fn sum(&self, a: u32, b: u32) -> Option<u32> {
        let (an, af) = self.deg[a as usize];
        let (bn, bf) = self.deg[b as usize];
        self.get(an + bn, af + bf)
    }
    /// The bidegree `(A + B)'` where `d(xy)` lives, which can be non-zero even when `A + B` is.
    fn sum_prime(&self, a: u32, b: u32) -> Option<u32> {
        let (an, af) = self.deg[a as usize];
        let (bn, bf) = self.deg[b as usize];
        let (n, f) = (an + bn, af + bf);
        if !self.in_range(n, f) {
            return None;
        }
        self.get(n - 1, f + 2)
    }
    /// target of d: (n - 1, f + 2)
    fn prime(&self, a: u32) -> Option<u32> {
        let (n, f) = self.deg[a as usize];
        self.get(n - 1, f + 2)
    }
    fn gid(&self, b: u32, i: usize) -> u32 {
        self.off[b as usize] + i as u32
    }
}

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

/// Split a CSV line whose last field may be a quoted list.
fn fields(line: &str) -> (Vec<i32>, &str) {
    let mut nums = Vec::with_capacity(6);
    let mut rest = line;
    for _ in 0..6 {
        let p = rest.find(',').unwrap();
        nums.push(rest[..p].parse().unwrap());
        rest = &rest[p + 1..];
    }
    (nums, rest)
}

fn load(dir: &str, n_max: i32, f_max: i32) -> Ext {
    let t = Instant::now();
    let mut ranks: BTreeMap<(i32, i32), usize> = BTreeMap::new();
    for line in BufReader::new(File::open(format!("{dir}/rank.csv")).unwrap())
        .lines()
        .skip(1)
    {
        let line = line.unwrap();
        let v: Vec<i32> = line.split(',').map(|x| x.parse().unwrap()).collect();
        if v[2] > 0 && v[0] <= n_max && v[1] <= f_max {
            ranks.insert((v[0], v[1]), v[2] as usize);
        }
    }
    let mut ext = Ext {
        n_max,
        f_max,
        bid: FxHashMap::default(),
        deg: vec![],
        dim: vec![],
        off: vec![],
        basis_bid: vec![],
        indec: vec![],
        gen_prod: FxHashMap::default(),
    };
    let mut total = 0u32;
    for (&(n, f), &d) in &ranks {
        assert!(d <= 32, "bidegree ({n},{f}) has rank {d} > 32");
        let id = ext.deg.len() as u32;
        ext.bid.insert((n, f), id);
        ext.deg.push((n, f));
        ext.dim.push(d);
        ext.off.push(total);
        for _ in 0..d {
            ext.basis_bid.push(id);
        }
        total += d as u32;
    }
    let full = !std::path::Path::new(&format!("{dir}/indecomposables.csv")).exists();
    ext.indec = vec![full; total as usize];
    let mut n_indec = 0;
    let indec_lines = if full {
        vec![]
    } else {
        BufReader::new(File::open(format!("{dir}/indecomposables.csv")).unwrap())
            .lines()
            .skip(1)
            .collect()
    };
    for line in indec_lines {
        let line = line.unwrap();
        let v: Vec<&str> = line.split(',').collect();
        let (n, f, i): (i32, i32, usize) = (
            v[0].parse().unwrap(),
            v[1].parse().unwrap(),
            v[2].parse().unwrap(),
        );
        if let Some(b) = ext.get(n, f) {
            let gi = ext.gid(b, i) as usize;
            ext.indec[gi] = true;
            n_indec += 1;
        }
    }
    let prod_file = if full {
        "products.csv"
    } else {
        "generator_products.csv"
    };
    for line in BufReader::new(File::open(format!("{dir}/{prod_file}")).unwrap())
        .lines()
        .skip(1)
    {
        let line = line.unwrap();
        let (v, rest) = fields(&line);
        if v[0] + v[3] > n_max || v[1] + v[4] > f_max {
            continue;
        }
        let (Some(g), Some(x)) = (ext.get(v[0], v[1]), ext.get(v[3], v[4])) else {
            continue;
        };
        let m = parse_vec(rest);
        if m != 0 {
            let key = (ext.gid(g, v[2] as usize), ext.gid(x, v[5] as usize));
            ext.gen_prod.insert(key, m);
            if full {
                ext.gen_prod.insert((key.1, key.0), m);
            }
        }
    }
    eprintln!(
        "loaded: {} bidegrees, dim E2 {}, {} indecomposables, {} generator products, {:.1}s",
        ext.deg.len(),
        total,
        n_indec,
        ext.gen_prod.len(),
        t.elapsed().as_secs_f64()
    );
    ext
}

/// One term of a recipe for a basis vector: an indecomposable, or g * y.
#[derive(Clone, Copy)]
enum Term {
    Ind(u32),
    Prod(u32, u32),
}

/// Express every basis vector of bidegree `a` through indecomposables and products g * y.
fn recipes(ext: &Ext, a: u32) -> Vec<Vec<Term>> {
    let d = ext.dim[a as usize];
    let (an, af) = ext.deg[a as usize];
    let mut cands: Vec<(Term, Mask)> = vec![];
    for i in 0..d {
        let g = ext.gid(a, i);
        if ext.indec[g as usize] {
            cands.push((Term::Ind(g), 1 << i));
        }
    }
    // decomposables: indecomposable g of bidegree G, y in A - G
    for (gid, &is) in ext.indec.iter().enumerate() {
        if !is {
            continue;
        }
        let gb = ext.basis_bid[gid];
        if gb == a {
            continue;
        }
        let (gn, gf) = ext.deg[gb as usize];
        let Some(y) = ext.get(an - gn, af - gf) else {
            continue;
        };
        for j in 0..ext.dim[y as usize] {
            let yg = ext.gid(y, j);
            if let Some(&m) = ext.gen_prod.get(&(gid as u32, yg)) {
                cands.push((Term::Prod(gid as u32, yg), m));
            }
        }
    }
    // Gaussian elimination with recipes (lists of candidate indices, XOR-combined)
    let mut piv: Vec<Option<(Mask, Vec<usize>)>> = vec![None; d];
    let mut rank = 0;
    for (ci, &(_, m)) in cands.iter().enumerate() {
        let mut m = m;
        let mut rec = vec![ci];
        while m != 0 {
            let c = m.trailing_zeros() as usize;
            match &piv[c] {
                Some((pm, pr)) => {
                    m ^= pm;
                    rec = xor_sorted(&rec, pr);
                }
                None => {
                    piv[c] = Some((m, rec));
                    rank += 1;
                    break;
                }
            }
        }
        if rank == d {
            break;
        }
    }
    assert_eq!(
        rank, d,
        "bidegree ({an},{af}) not spanned by products and indecomposables"
    );
    // back-substitute to unit vectors
    for c in (0..d).rev() {
        let (mut m, mut rec) = piv[c].clone().unwrap();
        let mut hi = m & !(1 << c);
        while hi != 0 {
            let k = hi.trailing_zeros() as usize;
            let (km, kr) = piv[k].as_ref().unwrap();
            m ^= km;
            rec = xor_sorted(&rec, kr);
            hi = m & !(1 << c);
        }
        debug_assert_eq!(m, 1 << c);
        piv[c] = Some((m, rec));
    }
    (0..d)
        .map(|c| {
            piv[c]
                .as_ref()
                .unwrap()
                .1
                .iter()
                .map(|&k| cands[k].0)
                .collect()
        })
        .collect()
}

fn xor_sorted<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
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

/// The full product table: blocks for pairs (A, B), A <= B in `order`.
struct Products {
    rank_of: Vec<u32>,
    /// (A, B) with rank A <= rank B -> block of dim A x dim B masks (row-major)
    blocks: FxHashMap<(u32, u32), Vec<Mask>>,
}

impl Products {
    fn mul(&self, ext: &Ext, a: u32, i: usize, b: u32, j: usize) -> Mask {
        let (a, i, b, j) = if self.rank_of[a as usize] <= self.rank_of[b as usize] {
            (a, i, b, j)
        } else {
            (b, j, a, i)
        };
        match self.blocks.get(&(a, b)) {
            Some(bl) => bl[i * ext.dim[b as usize] + j],
            None => 0,
        }
    }
}

fn apply_gen(ext: &Ext, g: u32, target: u32, v: Mask) -> Mask {
    let mut out = 0;
    let mut v = v;
    while v != 0 {
        let k = v.trailing_zeros() as usize;
        v &= v - 1;
        if let Some(&m) = ext.gen_prod.get(&(g, ext.gid(target, k))) {
            out ^= m;
        }
    }
    out
}

fn build_products(ext: &Ext) -> Products {
    let t = Instant::now();
    let nb = ext.deg.len();
    let mut order: Vec<u32> = (0..nb as u32).collect();
    order.sort_by_key(|&b| {
        let (n, f) = ext.deg[b as usize];
        (n + f, f)
    });
    let mut rank_of = vec![0; nb];
    for (r, &b) in order.iter().enumerate() {
        rank_of[b as usize] = r as u32;
    }
    let mut prods = Products {
        rank_of,
        blocks: FxHashMap::default(),
    };
    let mut entries = 0usize;
    for (r, &a) in order.iter().enumerate() {
        if ext.deg[a as usize] == (0, 0) {
            // the unit: 1 * e_j = e_j
            for &b in &order[r..] {
                if ext.sum(a, b).is_some() {
                    let bl: Vec<Mask> = (0..ext.dim[b as usize]).map(|j| 1 << j).collect();
                    entries += bl.len();
                    prods.blocks.insert((a, b), bl);
                }
            }
            continue;
        }
        let rec = recipes(ext, a);
        let partners: Vec<u32> = order[r..]
            .iter()
            .copied()
            .filter(|&b| ext.sum(a, b).is_some())
            .collect();
        let new: Vec<((u32, u32), Vec<Mask>)> = partners
            .par_iter()
            .map(|&b| {
                let c = ext.sum(a, b).unwrap();
                let (cn, cf) = ext.deg[c as usize];
                let db = ext.dim[b as usize];
                let mut bl = vec![0; ext.dim[a as usize] * db];
                for (i, ri) in rec.iter().enumerate() {
                    for j in 0..db {
                        let xj = ext.gid(b, j);
                        let mut m = 0;
                        for term in ri {
                            match *term {
                                Term::Ind(g) => {
                                    m ^= ext.gen_prod.get(&(g, xj)).copied().unwrap_or(0)
                                }
                                Term::Prod(g, y) => {
                                    let yb = ext.basis_bid[y as usize];
                                    let yi = (y - ext.off[yb as usize]) as usize;
                                    let v = prods.mul(ext, yb, yi, b, j);
                                    if v != 0 {
                                        let (gn, gf) = ext.deg[ext.basis_bid[g as usize] as usize];
                                        let tb = ext.get(cn - gn, cf - gf).unwrap();
                                        m ^= apply_gen(ext, g, tb, v);
                                    }
                                }
                            }
                        }
                        bl[i * db + j] = m;
                    }
                }
                ((a, b), bl)
            })
            .collect();
        for (k, bl) in new {
            entries += bl.len();
            prods.blocks.insert(k, bl);
        }
    }
    eprintln!(
        "product table: {} blocks, {} entries, {:.1}s",
        prods.blocks.len(),
        entries,
        t.elapsed().as_secs_f64()
    );
    prods
}

/// Self-check: the rebuilt table must agree with every generator product (both orders).
fn check_products(ext: &Ext, p: &Products) {
    let bad = ext
        .gen_prod
        .par_iter()
        .filter(|&(&(g, x), &m)| {
            let (gb, xb) = (ext.basis_bid[g as usize], ext.basis_bid[x as usize]);
            let gi = (g - ext.off[gb as usize]) as usize;
            let xi = (x - ext.off[xb as usize]) as usize;
            p.mul(ext, gb, gi, xb, xi) != m
        })
        .count();
    eprintln!("product check: {bad} mismatches against generator products");
    assert_eq!(bad, 0);
}

struct Vars {
    /// variable offset of each bidegree A (with A' in range), else u32::MAX
    off: Vec<u32>,
    n: u32,
    /// source stem of each variable
    stem: Vec<i32>,
}

fn vars(ext: &Ext) -> Vars {
    let nb = ext.deg.len();
    let mut ids: Vec<u32> = (0..nb as u32).filter(|&a| ext.prime(a).is_some()).collect();
    // decreasing stem: high stems are pivoted first
    ids.sort_by_key(|&a| {
        let (n, f) = ext.deg[a as usize];
        (-n, f)
    });
    let mut off = vec![u32::MAX; nb];
    let mut n = 0u32;
    let mut stem = vec![];
    for a in ids {
        off[a as usize] = n;
        let k = ext.dim[a as usize] * ext.dim[ext.prime(a).unwrap() as usize];
        n += k as u32;
        stem.extend(std::iter::repeat_n(ext.deg[a as usize].0, k));
    }
    Vars { off, n, stem }
}

/// variable: the k-th coordinate of d(e_r), e_r in bidegree c
fn var(ext: &Ext, v: &Vars, c: u32, r: usize, k: usize) -> u32 {
    let dp = ext.dim[ext.prime(c).unwrap() as usize];
    v.off[c as usize] + (r * dp + k) as u32
}

/// Where a Leibniz equation comes from: coordinate `k` of the relation for `x_(a, i) * x_(b, j)`.
#[derive(Clone, Copy)]
struct Origin {
    a: u32,
    b: u32,
    i: u8,
    j: u8,
    k: u8,
}

fn equations(ext: &Ext, p: &Products, v: &Vars, gens_only: bool) -> (Vec<Vec<u32>>, Vec<Origin>) {
    let t = Instant::now();
    let nb = ext.deg.len() as u32;
    let pairs: Vec<(u32, u32)> = (0..nb)
        .flat_map(|a| (0..nb).map(move |b| (a, b)))
        .filter(|&(a, b)| p.rank_of[a as usize] <= p.rank_of[b as usize])
        .filter(|&(a, b)| ext.sum_prime(a, b).is_some())
        .collect();
    let (eqs, origins): (Vec<Vec<u32>>, Vec<Origin>) = pairs
        .par_iter()
        .flat_map_iter(|&(a, b)| {
            // `A + B` may be zero-dimensional (`xy = 0` for degree reasons)
            let c = ext.sum(a, b);
            let cp = ext.sum_prime(a, b).unwrap();
            let (da, db, dcp) = (
                ext.dim[a as usize],
                ext.dim[b as usize],
                ext.dim[cp as usize],
            );
            let ap = ext.prime(a);
            let bp = ext.prime(b);
            let mut out = vec![];
            for i in 0..da {
                for j in 0..db {
                    if a == b && j < i {
                        continue;
                    }
                    if gens_only
                        && !ext.indec[ext.gid(a, i) as usize]
                        && !ext.indec[ext.gid(b, j) as usize]
                    {
                        continue;
                    }
                    let xy = if c.is_some() {
                        p.mul(ext, a, i, b, j)
                    } else {
                        0
                    };
                    for k in 0..dcp {
                        let mut eq: Vec<u32> = vec![];
                        // (e_i e_j) d^C
                        let mut m = xy;
                        while m != 0 {
                            let r = m.trailing_zeros() as usize;
                            m &= m - 1;
                            eq.push(var(ext, v, c.unwrap(), r, k));
                        }
                        // d(e_i) e_j: sum_c x^A_{i,c} [e_c e_j]_k
                        if let Some(ap) = ap {
                            for cc in 0..ext.dim[ap as usize] {
                                if p.mul(ext, ap, cc, b, j) >> k & 1 == 1 {
                                    eq.push(var(ext, v, a, i, cc));
                                }
                            }
                        }
                        // e_i d(e_j)
                        if let Some(bp) = bp {
                            for cc in 0..ext.dim[bp as usize] {
                                if p.mul(ext, a, i, bp, cc) >> k & 1 == 1 {
                                    eq.push(var(ext, v, b, j, cc));
                                }
                            }
                        }
                        eq.sort_unstable();
                        // cancel pairs (a = b, i = j gives d(x^2) = 2 x d(x) = 0 terms)
                        let mut red: Vec<u32> = Vec::with_capacity(eq.len());
                        for x in eq {
                            if red.last() == Some(&x) {
                                red.pop();
                            } else {
                                red.push(x);
                            }
                        }
                        if !red.is_empty() {
                            out.push((
                                red,
                                Origin {
                                    a,
                                    b,
                                    i: i as u8,
                                    j: j as u8,
                                    k: k as u8,
                                },
                            ));
                        }
                    }
                }
            }
            out
        })
        .unzip();
    let terms: usize = eqs.iter().map(|e| e.len()).sum();
    let mut hist = BTreeMap::new();
    for e in &eqs {
        *hist.entry(e.len()).or_insert(0usize) += 1;
    }
    eprintln!(
        "equations: {} nontrivial ({} pairs of bidegrees), {} terms, {:.1}s; sizes {:?}",
        eqs.len(),
        pairs.len(),
        terms,
        t.elapsed().as_secs_f64(),
        hist
    );
    (eqs, origins)
}

/// Echelon form (pivot = smallest column). Returns the pivot flag per column.
///
/// With `track`, also returns for every row the equation it started from and the rows it was
/// reduced by, so that any combination of rows can be expanded back into equations.
fn eliminate(
    nvars: u32,
    eqs: &[Vec<u32>],
    track: bool,
) -> (Vec<u32>, Vec<Vec<u32>>, Vec<(u32, Vec<u32>)>) {
    let t = Instant::now();
    let mut order: Vec<u32> = (0..eqs.len() as u32).collect();
    order.sort_by_key(|&e| eqs[e as usize].len());
    let mut pivot: Vec<u32> = vec![u32::MAX; nvars as usize];
    let mut rows: Vec<Vec<u32>> = vec![];
    let mut prov: Vec<(u32, Vec<u32>)> = vec![];
    let (mut xors, mut maxlen) = (0usize, 0usize);
    for (n, &e) in order.iter().enumerate() {
        let mut r = eqs[e as usize].clone();
        let mut used = vec![];
        while let Some(&c) = r.first() {
            let p = pivot[c as usize];
            if p == u32::MAX {
                maxlen = maxlen.max(r.len());
                pivot[c as usize] = rows.len() as u32;
                rows.push(r);
                if track {
                    prov.push((e, used));
                }
                break;
            }
            r = xor_sorted(&r, &rows[p as usize]);
            if track {
                used.push(p);
            }
            xors += 1;
        }
        if n % 1_000_000 == 0 && n > 0 {
            eprintln!(
                "  {n} rows, rank {}, {xors} xors, max row {maxlen}, {:.1}s",
                rows.len(),
                t.elapsed().as_secs_f64()
            );
        }
    }
    let fill: usize = rows.iter().map(|r| r.len()).sum();
    eprintln!(
        "echelon: rank {} of {} columns, {} xors, max row {}, {} entries, {:.1}s",
        rows.len(),
        nvars,
        xors,
        maxlen,
        fill,
        t.elapsed().as_secs_f64()
    );
    (pivot, rows, prov)
}

/// Number of nonzero coordinates of each vector of the reduced basis of the kernel: the vector of a
/// free column is 1 there, 0 on the other free columns, and solved on the pivots. Lanes of 64 free
/// columns are back-substituted at once.
fn kernel_density(pivot: &[u32], rows: &[Vec<u32>], stem: &[i32]) -> Vec<(u32, usize, i32, i32)> {
    let free: Vec<u32> = (0..pivot.len() as u32)
        .filter(|&c| pivot[c as usize] == u32::MAX)
        .collect();
    let pivcols: Vec<u32> = (0..pivot.len() as u32)
        .rev()
        .filter(|&c| pivot[c as usize] != u32::MAX)
        .collect();
    let chunks: Vec<&[u32]> = free.chunks(64).collect();
    chunks
        .par_iter()
        .flat_map_iter(|chunk| {
            let mut val = vec![0u64; pivot.len()];
            for (l, &f) in chunk.iter().enumerate() {
                val[f as usize] = 1 << l;
            }
            // a pivot row's other entries are at larger columns, already solved
            for &p in &pivcols {
                let r = &rows[pivot[p as usize] as usize];
                val[p as usize] = r[1..].iter().fold(0, |acc, &c| acc ^ val[c as usize]);
            }
            let mut nnz = vec![0usize; chunk.len()];
            let mut lo = vec![i32::MAX; chunk.len()];
            let mut hi = vec![i32::MIN; chunk.len()];
            for (c, &x) in val.iter().enumerate() {
                let mut x = x;
                while x != 0 {
                    let l = x.trailing_zeros() as usize;
                    nnz[l] += 1;
                    lo[l] = lo[l].min(stem[c]);
                    hi[l] = hi[l].max(stem[c]);
                    x &= x - 1;
                }
            }
            (0..chunk.len())
                .map(|l| (chunk[l], nnz[l], lo[l], hi[l]))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Evaluates every equation at a known d2 (`stem,Adams_filtration,index,target` in this basis).
fn check_d2(ext: &Ext, v: &Vars, eqs: &[Vec<u32>], path: &str) {
    let mut val = vec![false; v.n as usize];
    for line in BufReader::new(File::open(path).unwrap()).lines().skip(1) {
        let line = line.unwrap();
        let mut it = line.splitn(4, ',');
        let n: i32 = it.next().unwrap().parse().unwrap();
        let f: i32 = it.next().unwrap().parse().unwrap();
        let i: usize = it.next().unwrap().parse().unwrap();
        let m = parse_vec(it.next().unwrap());
        let Some(c) = ext.get(n, f) else { continue };
        if ext.prime(c).is_none() {
            continue;
        }
        let dp = ext.dim[ext.prime(c).unwrap() as usize];
        for k in 0..dp {
            val[var(ext, v, c, i, k) as usize] = m >> k & 1 == 1;
        }
    }
    // var -> (source bidegree, r, k) for reporting violations
    let describe = |x: u32| {
        let (c, _) = v
            .off
            .iter()
            .enumerate()
            .filter(|&(_, &o)| o != u32::MAX && o <= x)
            .max_by_key(|&(_, &o)| o)
            .unwrap();
        let dp = ext.dim[ext.prime(c as u32).unwrap() as usize];
        let rel = (x - v.off[c]) as usize;
        let (n, f) = ext.deg[c];
        format!(
            "d(x_({n},{f},{}))[{}]={}",
            rel / dp,
            rel % dp,
            val[x as usize] as u8
        )
    };
    for e in eqs
        .iter()
        .filter(|e| e.iter().fold(false, |acc, &x| acc ^ val[x as usize]))
    {
        let terms: Vec<String> = e.iter().map(|&x| describe(x)).collect();
        println!("violated: {}", terms.join(" + "));
    }
    let nonzero = val.iter().filter(|&&x| x).count();
    let bad = eqs
        .iter()
        .filter(|e| e.iter().fold(false, |acc, &x| acc ^ val[x as usize]))
        .count();
    println!(
        "d2 check: {nonzero} nonzero coordinates, {bad} of {} equations violated",
        eqs.len()
    );
}

/// Names a variable: coordinate `k` of `d(x_(n, f, r))`.
fn var_name(ext: &Ext, v: &Vars, y: u32) -> String {
    let (c, _) = v
        .off
        .iter()
        .enumerate()
        .filter(|&(_, &o)| o != u32::MAX && o <= y)
        .max_by_key(|&(_, &o)| o)
        .unwrap();
    let dp = ext.dim[ext.prime(c as u32).unwrap() as usize];
    let rel = (y - v.off[c]) as usize;
    let (n, f) = ext.deg[c];
    format!("d(x_({n},{f},{}))[{}]", rel / dp, rel % dp)
}

/// Whether `target` lies in the span of the equations `set`.
fn implied(eqs: &[Vec<u32>], set: &[u32], target: &[u32]) -> bool {
    let mut piv: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for &e in set {
        let mut r = eqs[e as usize].clone();
        while let Some(&c) = r.first() {
            match piv.get(&c) {
                Some(p) => r = xor_sorted(&r, p),
                None => {
                    piv.insert(c, r);
                    break;
                }
            }
        }
    }
    let mut r = target.to_vec();
    while let Some(&c) = r.first() {
        match piv.get(&c) {
            Some(p) => r = xor_sorted(&r, p),
            None => return false,
        }
    }
    true
}

/// Writes a set of Leibniz equations whose sum expresses the variable `x` through the free
/// variables, shrunk greedily, each with the product it comes from.
#[allow(clippy::too_many_arguments)]
fn certificate(
    ext: &Ext,
    p: &Products,
    v: &Vars,
    eqs: &[Vec<u32>],
    origins: &[Origin],
    pivot: &[u32],
    rows: &[Vec<u32>],
    prov: &[(u32, Vec<u32>)],
    x: u32,
) -> (Vec<u32>, Vec<u32>) {
    let t = Instant::now();
    assert_ne!(
        pivot[x as usize],
        u32::MAX,
        "{} is free",
        var_name(ext, v, x)
    );
    // back-substitute: x + (free variables), as a sum of rows
    let mut rp = vec![false; rows.len()];
    let mut e = rows[pivot[x as usize] as usize].clone();
    rp[pivot[x as usize] as usize] = true;
    loop {
        let Some(&c) = e.iter().find(|&&c| c != x && pivot[c as usize] != u32::MAX) else {
            break;
        };
        let r = pivot[c as usize] as usize;
        e = xor_sorted(&e, &rows[r]);
        rp[r] ^= true;
    }
    let rel: Vec<String> = e.iter().map(|&y| var_name(ext, v, y)).collect();
    println!("relation: {} = 0", rel.join(" + "));
    // expand rows into equations, latest row first
    let mut ep = vec![false; eqs.len()];
    for r in (0..rows.len()).rev() {
        if rp[r] {
            ep[prov[r].0 as usize] ^= true;
            for &u in &prov[r].1 {
                rp[u as usize] ^= true;
            }
        }
    }
    let mut set: Vec<u32> = (0..eqs.len() as u32).filter(|&i| ep[i as usize]).collect();
    let sum = set.iter().fold(vec![], |acc: Vec<u32>, &i| {
        xor_sorted(&acc, &eqs[i as usize])
    });
    assert_eq!(sum, e, "certificate does not sum to the relation");
    println!(
        "certificate: {} equations ({:.1}s)",
        set.len(),
        t.elapsed().as_secs_f64()
    );
    // greedy shrink, longest equations first
    if set.len() <= 20_000 {
        set.sort_by_key(|&i| std::cmp::Reverse(eqs[i as usize].len()));
        let mut k = 0;
        while k < set.len() {
            let trial: Vec<u32> = set
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != k)
                .map(|(_, &i)| i)
                .collect();
            if implied(eqs, &trial, &e) {
                set = trial;
            } else {
                k += 1;
            }
        }
        // the minimal set implies e; recover the combination that sums to it
        println!(
            "shrunk: {} equations ({:.1}s)",
            set.len(),
            t.elapsed().as_secs_f64()
        );
    }
    let cls = |b: u32, i: u8| {
        let (n, f) = ext.deg[b as usize];
        format!("x_({n},{f},{i})")
    };
    let deg_sum = |o: Origin| {
        let (an, af) = ext.deg[o.a as usize];
        let (bn, bf) = ext.deg[o.b as usize];
        (an + bn, af + bf)
    };
    let mut set = set;
    set.sort_by_key(|&i| deg_sum(origins[i as usize]));
    for &i in &set {
        let o = origins[i as usize];
        let (cn, cf) = deg_sum(o);
        let c = ext.sum(o.a, o.b);
        let prod = if c.is_some() {
            p.mul(ext, o.a, o.i as usize, o.b, o.j as usize)
        } else {
            0
        };
        let dc = c.map_or(0, |c| ext.dim[c as usize]);
        let terms: Vec<String> = eqs[i as usize]
            .iter()
            .map(|&y| var_name(ext, v, y))
            .collect();
        println!(
            "  {} * {} = {:?} in ({cn},{cf}); coordinate {} of d: {} = 0",
            cls(o.a, o.i),
            cls(o.b, o.j),
            (0..dc).map(|r| prod >> r & 1).collect::<Vec<_>>(),
            o.k,
            terms.join(" + ")
        );
    }
    (e, set)
}

/// Writes a pool of equations around a certificate for an exact minimum search: the equations
/// meeting the certificate's variables, and with `closure` also those lying entirely in the
/// variables these reach.
/// Format: `relation <vars>`, `name <var> <name>`, then `eq <id> <origin> : <vars>`.
fn dump_pool(
    ext: &Ext,
    v: &Vars,
    eqs: &[Vec<u32>],
    origins: &[Origin],
    rel: &[u32],
    set: &[u32],
    path: &str,
    closure: bool,
) {
    use std::io::Write;
    let mut v0 = vec![false; v.n as usize];
    for &i in set {
        for &y in &eqs[i as usize] {
            v0[y as usize] = true;
        }
    }
    let touching: Vec<u32> = (0..eqs.len() as u32)
        .into_par_iter()
        .filter(|&i| eqs[i as usize].iter().any(|&y| v0[y as usize]))
        .collect();
    let mut v1 = v0.clone();
    for &i in &touching {
        for &y in &eqs[i as usize] {
            v1[y as usize] = true;
        }
    }
    // the closure can be most of the system when h0 / h1 multiplications connect everything
    let pool: Vec<u32> = if closure {
        (0..eqs.len() as u32)
            .into_par_iter()
            .filter(|&i| eqs[i as usize].iter().all(|&y| v1[y as usize]))
            .collect()
    } else {
        touching.clone()
    };
    let mut w = std::io::BufWriter::new(File::create(path).unwrap());
    let rel_s: Vec<String> = rel.iter().map(|y| y.to_string()).collect();
    writeln!(w, "relation {}", rel_s.join(" ")).unwrap();
    let mut named = vec![false; v.n as usize];
    for &i in &pool {
        for &y in &eqs[i as usize] {
            if !named[y as usize] {
                named[y as usize] = true;
                writeln!(w, "name {y} {}", var_name(ext, v, y)).unwrap();
            }
        }
    }
    for &i in &pool {
        let o = origins[i as usize];
        let (an, af) = ext.deg[o.a as usize];
        let (bn, bf) = ext.deg[o.b as usize];
        let vs: Vec<String> = eqs[i as usize].iter().map(|y| y.to_string()).collect();
        writeln!(
            w,
            "eq {i} x_({an},{af},{})*x_({bn},{bf},{})[{}] : {}",
            o.i,
            o.j,
            o.k,
            vs.join(" ")
        )
        .unwrap();
    }
    eprintln!(
        "pool: {} touching, {} in the closure, written to {path}",
        touching.len(),
        pool.len()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let n_max: i32 = args[2].parse().unwrap();
    let f_max: i32 = args[3].parse().unwrap();
    let gens_only = args.iter().any(|a| a == "--gens-only");
    let ext = load(dir, n_max, f_max);
    let p = build_products(&ext);
    check_products(&ext, &p);
    let v = vars(&ext);
    eprintln!("dim H = {}", v.n);
    let (eqs, origins) = equations(&ext, &p, &v, gens_only);
    if let Some(i) = args.iter().position(|a| a == "--check-d2") {
        check_d2(&ext, &v, &eqs, &args[i + 1]);
    }
    let cert = args.iter().position(|a| a == "--certificate");
    let (pivot, rows, prov) = eliminate(v.n, &eqs, cert.is_some());
    if let Some(i) = cert {
        let q: Vec<i32> = args[i + 1..i + 5]
            .iter()
            .map(|a| a.parse().unwrap())
            .collect();
        let c = ext.get(q[0], q[1]).unwrap();
        let x = var(&ext, &v, c, q[2] as usize, q[3] as usize);
        let (rel, set) = certificate(&ext, &p, &v, &eqs, &origins, &pivot, &rows, &prov, x);
        if let Some(j) = args.iter().position(|a| a == "--dump-pool") {
            let closure = !args.iter().any(|a| a == "--pool-touching");
            dump_pool(&ext, &v, &eqs, &origins, &rel, &set, &args[j + 1], closure);
        }
    }
    let piv: Vec<bool> = pivot.iter().map(|&p| p != u32::MAX).collect();
    if args.iter().any(|a| a == "--density") {
        let mut dens = kernel_density(&pivot, &rows, &v.stem);
        dens.sort_by_key(|&(_, n, _, _)| std::cmp::Reverse(n));
        // free column -> source bidegree and coordinate
        let describe = |x: u32| {
            let (c, _) = v
                .off
                .iter()
                .enumerate()
                .filter(|&(_, &o)| o != u32::MAX && o <= x)
                .max_by_key(|&(_, &o)| o)
                .unwrap();
            let cp = ext.prime(c as u32).unwrap();
            let dp = ext.dim[cp as usize];
            let rel = (x - v.off[c]) as usize;
            let (n, f) = ext.deg[c];
            let (pn, pf) = ext.deg[cp as usize];
            format!("x_({n},{f},{}) -> x_({pn},{pf},{})", rel / dp, rel % dp)
        };
        if let Some(j) = args.iter().position(|a| a == "--density-dump") {
            use std::io::Write;
            let mut w = std::io::BufWriter::new(File::create(&args[j + 1]).unwrap());
            writeln!(w, "pivot,pivot_stem,nnz,min_stem,max_stem").unwrap();
            for &(x, n, lo, hi) in &dens {
                writeln!(
                    w,
                    "{},{},{n},{lo},{hi}",
                    var_name(&ext, &v, x),
                    v.stem[x as usize]
                )
                .unwrap();
            }
        }
        for &(x, n, _, _) in dens.iter().take(12) {
            eprintln!("density {n:>8}  {}", describe(x));
        }
        let mut hist = BTreeMap::new();
        for &(_, n, _, _) in &dens {
            *hist.entry(n.next_power_of_two()).or_insert(0usize) += 1;
        }
        eprintln!("density histogram (<= power of two): {hist:?}");
    }
    // --explain n f i k: express the coordinate k of d(x_(n,f,i)) in the free coordinates
    if let Some(i) = args.iter().position(|a| a == "--explain") {
        let q: Vec<i32> = args[i + 1..i + 5]
            .iter()
            .map(|a| a.parse().unwrap())
            .collect();
        let c = ext.get(q[0], q[1]).unwrap();
        let x = var(&ext, &v, c, q[2] as usize, q[3] as usize);
        let free: Vec<u32> = (0..v.n)
            .filter(|&c| pivot[c as usize] == u32::MAX)
            .collect();
        let deps: Vec<u32> = free
            .par_chunks(64)
            .flat_map_iter(|chunk| {
                let mut val = vec![0u64; v.n as usize];
                for (l, &f) in chunk.iter().enumerate() {
                    val[f as usize] = 1 << l;
                }
                for p in (0..v.n).rev().filter(|&c| pivot[c as usize] != u32::MAX) {
                    let r = &rows[pivot[p as usize] as usize];
                    val[p as usize] = r[1..].iter().fold(0, |acc, &c| acc ^ val[c as usize]);
                }
                let m = val[x as usize];
                chunk
                    .iter()
                    .enumerate()
                    .filter(move |&(l, _)| m >> l & 1 == 1)
                    .map(|(_, &f)| f)
                    .collect::<Vec<_>>()
            })
            .collect();
        let name = |y: u32| {
            let (c, _) = v
                .off
                .iter()
                .enumerate()
                .filter(|&(_, &o)| o != u32::MAX && o <= y)
                .max_by_key(|&(_, &o)| o)
                .unwrap();
            let dp = ext.dim[ext.prime(c as u32).unwrap() as usize];
            let rel = (y - v.off[c]) as usize;
            let (n, f) = ext.deg[c];
            format!("d(x_({n},{f},{}))[{}]", rel / dp, rel % dp)
        };
        let pivot_or_free = if pivot[x as usize] == u32::MAX {
            "free"
        } else {
            "pivot"
        };
        let terms: Vec<String> = deps.iter().map(|&y| name(y)).collect();
        println!(
            "explain: {} ({pivot_or_free}) = {}",
            name(x),
            if terms.is_empty() {
                "0".into()
            } else {
                terms.join(" + ")
            }
        );
    }
    let free_total = piv.iter().filter(|&&x| !x).count();
    println!(
        "N={n_max} F={f_max} gens_only={gens_only} dimE2={} dimH={} dimD={free_total}",
        ext.basis_bid.len(),
        v.n
    );
    // D_s = free columns in stems <= s
    let mut free_by_stem = vec![0usize; n_max as usize + 1];
    for (c, &is_piv) in piv.iter().enumerate() {
        if !is_piv {
            free_by_stem[v.stem[c] as usize] += 1;
        }
    }
    println!("stem,free,D_s");
    let mut cum = 0;
    for (s, &f) in free_by_stem.iter().enumerate() {
        cum += f;
        println!("{s},{f},{cum}");
    }
}
