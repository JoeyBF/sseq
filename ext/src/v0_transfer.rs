//! $\mathrm{Ext}_P(Q)$ from $\mathrm{Ext}_P(Q/I_1)$ by transferring the $v_0$-Bockstein perturbation.

use std::collections::BTreeMap;

use algebra::MilnorAlgebra;
use maybe_rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    algebraic_novikov::{InvertedSlice, untwisted_filtrations, untwisted_max_weight},
    chain_complex::FreeChainComplex,
    vn_bockstein::{Layer, Stage, slice, stage},
};

/// The bits of a packed term of a coboundary that hold the index within its slice; the length
/// takes the rest.
const INDEX_BITS: u32 = 24;

/// A vector over $\mathbb{F}_2$ as packed bits.
pub(crate) type Bits = Vec<u64>;

/// The zero vector of dimension `n`.
pub(crate) fn zeros(n: usize) -> Bits {
    vec![0; n.div_ceil(64)]
}

/// Flips bit `i`.
pub(crate) fn flip(v: &mut Bits, i: usize) {
    v[i / 64] ^= 1 << (i % 64);
}

/// Adds `b` to `a`.
pub(crate) fn add(a: &mut Bits, b: &Bits) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= y;
    }
}

/// The lowest set bit.
pub(crate) fn low(v: &Bits) -> Option<usize> {
    v.iter()
        .enumerate()
        .find(|(_, w)| **w != 0)
        .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
}

/// The set bits.
pub(crate) fn ones(v: &Bits) -> impl Iterator<Item = usize> + '_ {
    v.iter().enumerate().flat_map(|(i, &w)| {
        let mut w = w;
        std::iter::from_fn(move || {
            (w != 0).then(|| {
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                i * 64 + b
            })
        })
    })
}

/// A reduced basis of the image of a map, each vector with its preimage, by lowest set bit.
#[derive(Default)]
struct Image {
    /// For each lowest bit, the reduced image vector and a preimage of it.
    by_low: FxHashMap<usize, (Bits, Bits)>,
}

impl Image {
    /// Reduces `y` against the basis, adding the preimages of what it removes to `pre`.
    fn reduce(&self, y: &mut Bits, pre: &mut Bits) {
        while let Some(l) = low(y) {
            let Some((v, p)) = self.by_low.get(&l) else {
                return;
            };
            add(y, v);
            add(pre, p);
        }
    }
}

/// The cohomology of one slice $\mathrm{Hom}(C, (Q/I_1)_j)^a$ at a fixed weight, with the data
/// of a deformation retraction onto it.
struct Slice {
    /// The dimension of the cochains.
    dim: usize,
    /// The image of $\delta_0$ out of this degree, with preimages here.
    out: Image,
    /// A basis of the cocycles: the image of $\delta_0$ into this degree, with preimages in the
    /// degree below, and representatives of the cohomology, with their index.
    cycles: FxHashMap<usize, (Bits, Cycle)>,
    /// The representatives of the cohomology.
    classes: Vec<Bits>,
}

/// What a vector in [`Slice::cycles`] is.
enum Cycle {
    /// A coboundary, with a preimage of dimension `dim` below.
    Boundary(Bits),
    /// The representative of a class.
    Class(usize),
}

impl Slice {
    /// The projection $p$ to the cohomology and the homotopy $h$ to the degree below, of a
    /// cochain `c`, given the dimension of the degree below and $\delta_0$ of `c`.
    fn p_h(&self, c: &Bits, mut dc: Bits, below: usize) -> (Bits, Bits) {
        // c - σ(δc) is a cocycle; split it into a coboundary and a class.
        let mut z = c.clone();
        let mut x = zeros(self.dim);
        self.out.reduce(&mut dc, &mut x);
        add(&mut z, &x);
        let mut p = zeros(self.classes.len());
        let mut h = zeros(below);
        while let Some(l) = low(&z) {
            let (v, kind) = &self.cycles[&l];
            add(&mut z, v);
            match kind {
                Cycle::Boundary(pre) => add(&mut h, pre),
                Cycle::Class(i) => flip(&mut p, *i),
            }
        }
        (p, h)
    }
}

/// The dimensions of $\mathrm{Ext}_P(Q)$ in the trigrading `(stem, s, u)` within stem `max_n`
/// and `s <= max_s`, keyed as [`crate::algebraic_novikov::TrigradedDims`].
///
/// $\mathrm{Hom}(C, Q_k)$ is $\bigoplus_{j \leq k} \mathrm{Hom}(C, (Q/I_1)_j)\, v_0^{k-j}$, and
/// $\delta = \sum_r v_0^r \delta_r$ with $\delta_r$ lowering the length by $r$. Each slice of
/// $\delta_0$ is retracted onto its cohomology $\mathrm{Ext}(Q/I_1)$, and the perturbation lemma
/// carries $\delta' = \sum_{r \geq 1} v_0^r \delta_r$ to $D = p \delta' \sum_n (h \delta')^n i$ on
/// $\mathrm{Ext}(Q/I_1)[v_0]$, whose cohomology in each $v_0$-degree is $\mathrm{Ext}(Q_k)$. Only
/// slices are ever reduced, never a degree of $\mathrm{Hom}(C, Q_k)$.
pub fn ext_q_by_v0_transfer<CC>(
    res: &CC,
    max_n: i32,
    max_s: i32,
) -> BTreeMap<(i32, i32, i32), usize>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    ext_q_by_v0_transfer_in_bands(res, max_n, max_s, 1)
}

/// [`ext_q_by_v0_transfer`] with the weights taken in `bands` bands, one after another.
///
/// The coaction table each band needs shrinks with the band (see `Stage::tabulate`), so more bands
/// trade a walk down the tree of monomials per band for memory.
pub fn ext_q_by_v0_transfer_in_bands<CC>(
    res: &CC,
    max_n: i32,
    max_s: i32,
    bands: usize,
) -> BTreeMap<(i32, i32, i32), usize>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    transfer(res, max_n, max_s, bands, None, false)
}

/// [`ext_q_by_v0_transfer_in_bands`] from a conjugated backbone, with the action tabulated band by
/// band from Milnor matrices instead of from the coaction: `conjugated` must cover the
/// differentials out of filtrations up to `max_s + 1` from generators of degree up to
/// `(max_n + max_s) / 2`.
pub fn ext_q_by_v0_transfer_conjugated<CC>(
    res: &CC,
    conjugated: &crate::conjugate::ConjugatedBackbone,
    max_n: i32,
    max_s: i32,
    bands: usize,
) -> BTreeMap<(i32, i32, i32), usize>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    // `CTAU_PER_BLOCK` trades time for memory: no table of the action is kept, and each block
    // computes the action of its own operations.
    let per_block = std::env::var_os("CTAU_PER_BLOCK").is_some();
    transfer(res, max_n, max_s, bands, Some(conjugated), per_block)
}

/// The transfer, reading the backbone or, if given, the conjugated backbone, whose action is
/// computed block by block with `per_block` instead of tabulated.
fn transfer<CC>(
    res: &CC,
    max_n: i32,
    max_s: i32,
    bands: usize,
    conjugated: Option<&crate::conjugate::ConjugatedBackbone>,
    per_block: bool,
) -> BTreeMap<(i32, i32, i32), usize>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    let max_u = (max_n + max_s) / 2;
    let mut stage = stage(
        res,
        Layer::Adjoin(crate::algebraic_novikov::QDOWN_VARS - 1),
        max_n,
        max_s,
    );
    stage.inline = conjugated.is_some();
    stage.per_block = per_block;
    let slices: Vec<InvertedSlice> = (0..=max_s)
        .into_maybe_par_iter()
        .map(|a| match conjugated {
            Some(c) => crate::vn_bockstein::conjugated_slice(c, a, max_n, max_s),
            None => slice(res, a, max_n, max_s),
        })
        .collect();
    // The lengths `weight` builds with: see `k_of` there.
    let max_len = |a: i32, u: i32| {
        (a >= (2 * u - max_n - 1).max(0))
            .then(|| (max_s - a).min(max_n + max_s - 2 * u))
            .filter(|&k| k >= 0)
            .map(|k| k as u32)
    };
    // Bands of equal width: the table is largest for the middle weights, where the generators
    // that act carry the most operations, so no fixed skew suits every range.
    let ends: Vec<i32> = (1..=bands as i32)
        .map(|i| (max_u + 1) * i / bands as i32 - 1)
        .collect();
    let timing = std::env::var_os("CTAU_TIMING").is_some();
    // `CTAU_WEIGHTS_AT_ONCE` bounds the weights computed together, whose structures set the peak
    // beyond the table: fewer trade the parallelism across weights for memory.
    let at_once = std::env::var("CTAU_WEIGHTS_AT_ONCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(usize::MAX);
    TIMING.store(timing, std::sync::atomic::Ordering::Relaxed);
    let mut out = BTreeMap::new();
    let mut lo = 0;
    for hi in ends {
        if hi < lo {
            continue;
        }
        let mut clock = std::time::Instant::now();
        stage.tabulate(&slices, max_len, lo..=hi);
        lap(0, &mut clock);
        if timing {
            eprintln!(
                "weights {lo}..={hi}: table {} MB, resident {} MB",
                stage.table_bytes() >> 20,
                memory("VmRSS") >> 10
            );
        }
        // At most `at_once` weights are in memory together; each is parallel within itself.
        let weights: Vec<i32> = (lo..=hi).collect();
        for chunk in weights.chunks(at_once) {
            let per_u: Vec<BTreeMap<(i32, i32, i32), usize>> = chunk
                .to_vec()
                .into_maybe_par_iter()
                .map(|u| weight(&stage, &slices, u, max_n, max_s))
                .collect();
            out.extend(per_u.into_iter().flatten());
        }
        lo = hi + 1;
    }
    if timing {
        let report: Vec<String> = PHASE_NAMES
            .iter()
            .zip(&PHASES)
            .map(|(name, t)| {
                let t = t.swap(0, std::sync::atomic::Ordering::Relaxed);
                format!("{name} {:.1}", t as f64 * 1e-9)
            })
            .collect();
        eprintln!(
            "transfer seconds, summed over weights: {}",
            report.join(", ")
        );
        eprintln!("peak resident {} MB", memory("VmHWM") >> 10);
        let ([c, r, ph, d], u) = std::mem::take(&mut *LARGEST.lock().unwrap());
        eprintln!(
            "largest weight {u}: {} MB held at once: coboundaries {}, reductions {}, \
             p and h {}, D rows {}",
            (c + r + ph + d) >> 20,
            c >> 20,
            r >> 20,
            ph >> 20,
            d >> 20
        );
    }
    out
}

/// What the transfer spent, summed over the weights, in nanoseconds; printed by [`transfer`] when
/// `CTAU_TIMING` is set, with the size of the action's table for each band of weights and the
/// memory resident after tabulating it.
static PHASES: [std::sync::atomic::AtomicU64; PHASE_NAMES.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; PHASE_NAMES.len()];

/// Whether [`transfer`] reports its phases.
static TIMING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The most bytes the structures of one weight held at once, as [`weight`] measures them, and
/// that weight: the coboundaries, the reductions of the slices, p and h, and the rows of D.
static LARGEST: std::sync::Mutex<([usize; 4], i32)> = std::sync::Mutex::new(([0; 4], 0));

/// The phases of [`PHASES`]: tabulating the action, then the steps of [`weight`].
const PHASE_NAMES: [&str; 7] = [
    "tabulate",
    "cochains",
    "build",
    "slices",
    "p and h",
    "D",
    "of which D's rank",
];

/// The field `name` of `/proc/self/status`, in kB, or 0 where there is none.
fn memory(name: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(name))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

/// Adds the time since `start` to phase `i` of [`PHASES`], and restarts `start`.
fn lap(i: usize, start: &mut std::time::Instant) {
    let now = std::time::Instant::now();
    PHASES[i].fetch_add(
        (now - *start).as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    *start = now;
}

/// A coboundary out of one slice, column by column: the terms of column `i` are
/// `data[offsets[i]..offsets[i + 1]]`, each a target length above [`INDEX_BITS`] and an index in
/// its slice below.
#[derive(Default)]
struct Columns {
    offsets: Vec<u32>,
    data: Vec<u32>,
}

/// The terms of column `i` of `c`, as (target length, index).
fn terms(c: &Columns, i: usize) -> impl Iterator<Item = (u32, usize)> + '_ {
    c.data[c.offsets[i] as usize..c.offsets[i + 1] as usize]
        .iter()
        .map(|&p| (p >> INDEX_BITS, (p & ((1 << INDEX_BITS) - 1)) as usize))
}

/// The bytes the columns of a degree hold.
fn columns_bytes(delta: &[Columns]) -> usize {
    delta
        .iter()
        .map(|c| 4 * (c.data.len() + c.offsets.len()))
        .sum()
}

/// The bytes the reductions of the slices of a degree hold.
fn slices_bytes(slices: &FxHashMap<u32, Slice>) -> usize {
    let bits = |v: &Bits| v.len() * 8;
    slices
        .values()
        .map(|s| {
            s.out
                .by_low
                .values()
                .map(|(y, p)| bits(y) + bits(p))
                .sum::<usize>()
                + s.cycles
                    .values()
                    .map(|(v, c)| {
                        bits(v)
                            + match c {
                                Cycle::Boundary(p) => bits(p),
                                Cycle::Class(_) => 0,
                            }
                    })
                    .sum::<usize>()
                + s.classes.iter().map(bits).sum::<usize>()
        })
        .sum()
}

/// The bytes p and h of the slices of a degree hold.
fn ph_bytes(ph: &FxHashMap<u32, (Vec<Bits>, Vec<Bits>)>) -> usize {
    ph.values()
        .map(|(p, h)| p.iter().chain(h).map(|v| v.len() * 8).sum::<usize>())
        .sum()
}

/// Row `t` of length `jt` of `y`, made as a zero vector of dimension `n` if it is not yet.
fn row_of(y: &mut [Vec<Option<Bits>>], jt: u32, t: usize, n: usize) -> &mut Bits {
    y[jt as usize][t].get_or_insert_with(|| zeros(n))
}

/// The source classes of D taken at once: a row of its working vectors holds this many bits.
const D_CHUNK: usize = 4096;

/// Each cochain's slice and index in it, packed as the terms of [`Columns`], from the lengths of
/// the cochains in order: within a length, the order is that of the cochains.
fn positions(lengths: &[u32], dims: &mut Vec<usize>) -> Vec<u32> {
    lengths
        .iter()
        .map(|&j| {
            if dims.len() <= j as usize {
                dims.resize(j as usize + 1, 0);
            }
            let i = dims[j as usize];
            assert!(i < 1 << INDEX_BITS, "a slice outgrew INDEX_BITS");
            dims[j as usize] += 1;
            j << INDEX_BITS | i as u32
        })
        .collect()
}

/// [`ext_q_by_v0_transfer`] at weight `u`.
fn weight<CC>(
    stage: &Stage<'_, CC>,
    inverted: &[InvertedSlice],
    u: i32,
    max_n: i32,
    max_s: i32,
) -> BTreeMap<(i32, i32, i32), usize>
where
    CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
{
    // Classes are wanted for stem 2u - a <= max_n; the degree below feeds the coboundaries.
    let a_lo = (2 * u - max_n - 1).max(0);
    if a_lo > max_s {
        return BTreeMap::new();
    }
    let mut clock = std::time::Instant::now();
    // A class of s <= max_s and weight u has length at most max_s - a and, by
    // `untwisted_max_weight`, at most max_n + max_s - 2u.
    let k_of = |a: i32| (max_s - a).min(max_n + max_s - 2 * u).max(0) as u32;
    let index = |a: i32| (a - a_lo) as usize;

    // The cochains of each degree, by length. A degree also holds the length one past its own,
    // as the target of the degree below.
    let targets: Vec<(Vec<usize>, Vec<u32>)> = (a_lo..=max_s + 1)
        .into_maybe_par_iter()
        .map(|a| {
            let k = if a == 0 { k_of(0) } else { k_of(a - 1) };
            let mut dims = Vec::new();
            let pos = positions(&stage.lengths(a, u, k), &mut dims);
            (dims, pos)
        })
        .collect();
    let dim = |a: i32, j: u32| {
        targets
            .get(index(a))
            .and_then(|(d, _)| d.get(j as usize))
            .copied()
            .unwrap_or(0)
    };
    lap(1, &mut clock);

    // δ^a out of each slice (a, j).
    let build = |a: i32| -> Vec<Columns> {
        let k = k_of(a);
        let mut src_dims = Vec::new();
        let src = positions(&stage.lengths(a, u, k), &mut src_dims);
        let to = &targets[index(a + 1)].1;
        let mut out: Vec<Columns> = (0..=k)
            .map(|_| Columns {
                offsets: vec![0],
                data: Vec::new(),
            })
            .collect();
        // Appends the column of each cochain in `range`, its terms read off `entries`, the
        // cochain's place in its run above 32 bits and the target below.
        let append = |out: &mut Vec<Columns>, range: std::ops::Range<usize>, entries: &[u64]| {
            let mut e = 0;
            for (c, &s) in src[range].iter().enumerate() {
                let col = &mut out[(s >> INDEX_BITS) as usize];
                while e < entries.len() && (entries[e] >> 32) as usize == c {
                    col.data.push(to[entries[e] as u32 as usize]);
                    e += 1;
                }
                col.offsets
                    .push(u32::try_from(col.data.len()).expect("a slice's coboundary outgrew u32"));
            }
        };
        // The degrees of generators a few at a time, in parallel, each appended in order, so
        // that only those in flight hold their terms twice.
        let blocks = stage.column_blocks(a, u, k);
        let group = std::thread::available_parallelism().map_or(1, |n| n.get());
        let mut next = 0;
        for some in blocks.chunks(group) {
            let parts: Vec<Vec<u64>> = some
                .to_vec()
                .into_maybe_par_iter()
                .map(|(t_i, _, _)| {
                    stage.block_entries(a, u, k, &inverted[a as usize], (a + 1, u), t_i)
                })
                .collect();
            for (&(_, start, len), entries) in some.iter().zip(parts) {
                append(&mut out, next..start, &[]);
                append(&mut out, start..start + len, &entries);
                next = start + len;
            }
        }
        append(&mut out, next..src.len(), &[]);
        out
    };

    // The reduction of δ_0 out of each slice (a, j), retracted onto its cohomology, given that of
    // degree a - 1.
    let reduce = |a: i32,
                  delta: &[Columns],
                  below: Option<&FxHashMap<u32, Slice>>|
     -> FxHashMap<u32, Slice> {
        (0..=k_of(a))
            .filter(|&j| dim(a, j) > 0)
            .collect::<Vec<_>>()
            .into_maybe_par_iter()
            .map(|j| {
                let n = dim(a, j);
                // δ_0 out of (a, j): reduce its columns, keeping preimages; zero columns are
                // cocycles.
                let mut out = Image::default();
                let mut kernel = Vec::new();
                for i in 0..n {
                    let mut y = zeros(dim(a + 1, j));
                    for (jt, t) in terms(&delta[j as usize], i) {
                        if jt == j {
                            flip(&mut y, t);
                        }
                    }
                    let mut pre = zeros(n);
                    flip(&mut pre, i);
                    out.reduce(&mut y, &mut pre);
                    match low(&y) {
                        Some(l) => {
                            out.by_low.insert(l, (y, pre));
                        }
                        None => kernel.push(pre),
                    }
                }
                // The coboundaries into (a, j), from the slice below, then classes completing
                // them.
                let mut cycles: FxHashMap<usize, (Bits, Cycle)> = FxHashMap::default();
                if let Some(below) = below.and_then(|b| b.get(&j)) {
                    for (l, (v, p)) in &below.out.by_low {
                        cycles.insert(*l, (v.clone(), Cycle::Boundary(p.clone())));
                    }
                }
                let mut classes = Vec::new();
                for mut z in kernel {
                    while let Some(l) = low(&z) {
                        match cycles.get(&l) {
                            Some((v, _)) => add(&mut z, v),
                            None => break,
                        }
                    }
                    if let Some(l) = low(&z) {
                        cycles.insert(l, (z.clone(), Cycle::Class(classes.len())));
                        classes.push(z);
                    }
                }
                let slice = Slice {
                    dim: n,
                    out,
                    cycles,
                    classes,
                };
                (j, slice)
            })
            .collect()
    };

    // p and h of each basis cochain of each slice (a, j).
    let p_and_h = |a: i32,
                   delta: &[Columns],
                   slices: &FxHashMap<u32, Slice>|
     -> FxHashMap<u32, (Vec<Bits>, Vec<Bits>)> {
        slices
            .iter()
            .collect::<Vec<_>>()
            .into_maybe_par_iter()
            .map(|(&j, slice)| {
                let below = dim(a - 1, j);
                let cols = (0..slice.dim)
                    .map(|c| {
                        let mut e = zeros(slice.dim);
                        flip(&mut e, c);
                        let mut dc = zeros(dim(a + 1, j));
                        for (jt, t) in terms(&delta[j as usize], c) {
                            if jt == j {
                                flip(&mut dc, t);
                            }
                        }
                        slice.p_h(&e, dc, below)
                    })
                    .unzip();
                (j, cols)
            })
            .collect()
    };

    // D^a for all classes of degree a at once: a matrix whose columns are the classes of degree
    // a, by length then index, carried down the lengths of degree a + 1 from the top. Returns the
    // rank of D^a restricted to sources of length at most j, for each j.
    let d_ranks = |a: i32,
                   delta: &[Columns],
                   slices: &FxHashMap<u32, Slice>,
                   above: Option<&FxHashMap<u32, Slice>>,
                   ph: &FxHashMap<u32, (Vec<Bits>, Vec<Bits>)>|
     -> Vec<usize> {
        // Source classes by length then index, and where each length starts.
        let mut first: BTreeMap<u32, usize> = BTreeMap::new();
        let mut n_src = 0;
        for j in 0..=k_of(a) {
            first.insert(j, n_src);
            n_src += slices.get(&j).map_or(0, |s| s.classes.len());
        }
        let k_top = k_of(a + 1);
        // The source classes in order, as (length, index in its slice).
        let sources: Vec<(u32, usize)> = (0..=k_of(a))
            .flat_map(|j| {
                let n = slices.get(&j).map_or(0, |s| s.classes.len());
                (0..n).map(move |c| (j, c))
            })
            .collect();
        // Columns of D, by source class, over the target classes; the target classes are the
        // classes of degree a + 1 from the top length down.
        let n_tgt: usize = above.map_or(0, |above| above.values().map(|s| s.classes.len()).sum());
        let mut d_cols = vec![zeros(n_tgt); n_src];
        // The sources a chunk at a time, so that a row of y holds a chunk's bits, not all of them.
        for chunk_start in (0..n_src).step_by(D_CHUNK) {
            let chunk = &sources[chunk_start..n_src.min(chunk_start + D_CHUNK)];
            let n = chunk.len();
            // y[jt][t]: the sources of the chunk whose image so far has cochain t of (a + 1, jt);
            // rows are made when first reached.
            let mut y: Vec<Vec<Option<Bits>>> = (0..=k_top + 1)
                .map(|jt| vec![None; dim(a + 1, jt)])
                .collect();
            // y = δ'(i x) for every class x of the chunk.
            for (col, &(j, c)) in chunk.iter().enumerate() {
                for i in ones(&slices[&j].classes[c]) {
                    for (jt, t) in terms(&delta[j as usize], i) {
                        if jt < j {
                            flip(row_of(&mut y, jt, t, n), col);
                        }
                    }
                }
            }
            // From the top length down: collect p(y), and send δ'(h y) to the lower lengths.
            let mut target = 0;
            if let Some(above) = above {
                for jt in (0..=k_top).rev() {
                    let Some((p_cols, h_cols)) = ph.get(&jt) else {
                        continue;
                    };
                    let n_cls = above[&jt].classes.len();
                    let mut p_rows = vec![zeros(n); n_cls];
                    let mut h_rows: Vec<Option<Bits>> = vec![None; dim(a, jt)];
                    let rows = std::mem::take(&mut y[jt as usize]);
                    for (c, row) in rows.iter().enumerate() {
                        let Some(row) = row else { continue };
                        if low(row).is_none() {
                            continue;
                        }
                        for t in ones(&p_cols[c]) {
                            add(&mut p_rows[t], row);
                        }
                        for t in ones(&h_cols[c]) {
                            add(h_rows[t].get_or_insert_with(|| zeros(n)), row);
                        }
                    }
                    drop(rows);
                    for (i, row) in h_rows.iter().enumerate() {
                        let Some(row) = row else { continue };
                        if low(row).is_none() {
                            continue;
                        }
                        for (jn, t) in terms(&delta[jt as usize], i) {
                            if jn < jt {
                                add(row_of(&mut y, jn, t, n), row);
                            }
                        }
                    }
                    for row in &p_rows {
                        for c in ones(row) {
                            flip(&mut d_cols[chunk_start + c], target);
                        }
                        target += 1;
                    }
                }
            }
        }
        let mut rank_clock = std::time::Instant::now();
        // Rank of D^a restricted to sources of length at most j, incrementally.
        let mut table: FxHashMap<usize, Bits> = FxHashMap::default();
        let mut rank = 0;
        let mut out = Vec::new();
        for j in 0..=k_of(a) {
            let end = first.range(j + 1..).next().map_or(n_src, |(_, &e)| e);
            for mut col in d_cols[first[&j]..end].iter().cloned() {
                while let Some(l) = low(&col) {
                    match table.get(&l) {
                        Some(v) => add(&mut col, v),
                        None => break,
                    }
                }
                if let Some(l) = low(&col) {
                    table.insert(l, col);
                    rank += 1;
                }
            }
            out.push(rank);
        }
        lap(6, &mut rank_clock);
        out
    };

    // What the structures above hold, in bytes, for [`LARGEST`].
    let d_rows_bytes = |a: i32, slices: &FxHashMap<u32, Slice>| -> usize {
        let classes: usize = slices.values().map(|s| s.classes.len()).sum();
        let rows: usize = (0..=k_of(a + 1) + 1).map(|jt| dim(a + 1, jt)).sum();
        rows * classes.min(D_CHUNK).div_ceil(64) * 8
    };

    // The filtrations are taken in increasing order, holding only what the next steps read: the
    // reduction of degree a reads that of a - 1, p and h of a read δ out of a, and D out of a - 1
    // reads p and h of a, so each degree is dropped once D out of it is known.
    let mut ranks: Vec<Vec<usize>> = Vec::new();
    let mut class_counts: FxHashMap<(i32, u32), usize> = FxHashMap::default();
    let mut below: Option<(Vec<Columns>, FxHashMap<u32, Slice>)> = None;
    let mut largest = [0; 4];
    for a in a_lo..=max_s + 1 {
        let current = (a <= max_s).then(|| {
            let delta = build(a);
            lap(2, &mut clock);
            let slices = reduce(a, &delta, below.as_ref().map(|b| &b.1));
            lap(3, &mut clock);
            (delta, slices)
        });
        if let Some((delta, slices)) = &current {
            for (&j, slice) in slices {
                class_counts.insert((a, j), slice.classes.len());
            }
            let _ = delta;
        }
        if let Some((delta_below, slices_below)) = &below {
            let ph = current
                .as_ref()
                .map(|(delta, slices)| p_and_h(a, delta, slices))
                .unwrap_or_default();
            lap(4, &mut clock);
            if TIMING.load(std::sync::atomic::Ordering::Relaxed) {
                let sizes = [
                    columns_bytes(delta_below)
                        + current.as_ref().map_or(0, |(d, _)| columns_bytes(d)),
                    slices_bytes(slices_below)
                        + current.as_ref().map_or(0, |(_, s)| slices_bytes(s)),
                    ph_bytes(&ph),
                    d_rows_bytes(a - 1, slices_below),
                ];
                if sizes.iter().sum::<usize>() > largest.iter().sum::<usize>() {
                    largest = sizes;
                }
            }
            let slices_above = current.as_ref().map(|(_, s)| s);
            ranks.push(d_ranks(a - 1, delta_below, slices_below, slices_above, &ph));
            lap(5, &mut clock);
        }
        below = current;
    }
    if TIMING.load(std::sync::atomic::Ordering::Relaxed) {
        let mut l = LARGEST.lock().unwrap();
        if largest.iter().sum::<usize>() > l.0.iter().sum::<usize>() {
            *l = (largest, u);
        }
    }

    let rank = |a: i32, k: u32| -> usize {
        ranks
            .get(index(a))
            .map_or(0, |r| r[(k as usize).min(r.len() - 1)])
    };
    let mut out = BTreeMap::new();
    for k in 0..=max_s {
        if u > untwisted_max_weight(max_n, max_s, k) {
            continue;
        }
        for a in untwisted_filtrations(max_n, max_s, k, u) {
            let stem = 2 * u - a;
            if stem > max_n || stem < 0 || a < a_lo {
                continue;
            }
            let h: usize = (0..=k as u32)
                .map(|j| class_counts.get(&(a, j)).copied().unwrap_or(0))
                .sum();
            let d = h - rank(a, k as u32) - if a > a_lo { rank(a - 1, k as u32) } else { 0 };
            if d > 0 {
                out.insert((stem, a + k, u), d);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebraic_novikov::{ctau_e2_untwisted_on, unit_module, untwisted_backbone_range};

    /// The transfer reproduces $\mathrm{Ext}_P(Q)$, from the backbone and from the conjugated
    /// backbone with the action tabulated or computed block by block.
    #[test]
    fn matches_direct() {
        let (max_n, max_s) = (40, 20);
        let res = crate::nassau::Resolution::new(std::sync::Arc::new(unit_module()));
        res.compute_through_stem(untwisted_backbone_range(max_n, max_s));
        let direct: BTreeMap<_, _> = ctau_e2_untwisted_on(&res, max_n, max_s)
            .0
            .into_iter()
            .collect();
        assert_eq!(ext_q_by_v0_transfer(&res, max_n, max_s), direct);
        assert_eq!(ext_q_by_v0_transfer_in_bands(&res, max_n, max_s, 3), direct);
        let (conjugated, _) =
            crate::conjugate::ConjugatedBackbone::compute(&res, max_s, (max_n + max_s) / 2);
        assert_eq!(
            ext_q_by_v0_transfer_conjugated(&res, &conjugated, max_n, max_s, 1),
            direct
        );
        assert_eq!(
            transfer(&res, max_n, max_s, 1, Some(&conjugated), true),
            direct
        );
    }
}
