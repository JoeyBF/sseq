//! Prototype: one batch of a d2 composite as 1-bit GEMMs against right-multiplication slabs.

use std::{collections::HashMap, sync::Arc, time::Instant};

use algebra::{Algebra, pair_algebra::PairAlgebra};
use ext::chain_complex::ChainComplex;
use fp::{blas::cuda::GpuContext, matrix::Matrix, prime::TWO, vector::FpVector};
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

/// Usage: `d2_slab_proto build <store> <max_n> <max_s> <n> <s> <max_slabs> <dir>`, then
/// `d2_slab_proto gemm <dir> [iters]` on a Hopper GPU (the save and the GPU need not share a node).
///
/// `build`: at bidegree `(n, s)`, `t = n + s`, picks the generator g1 of row `s - 1` with the most
/// generators g of degree t mapping to it, and the component degree e of `d(g1)` with the most
/// distinct basis elements S. `X` (M x dim A_k, k = t - deg g1) holds the ops `x_g = d(g)_{g1}`;
/// for up to `max_slabs` of those S, `slab_S` (dim A_k rows) holds `sigma(P(R), P(S))` for every
/// basis element R of A_k as a pair element of degree k + e: the product terms' planes `B0`, `B1`
/// and the Y vectors `BY`, each S's columns side by side. Then on the GPU:
///
/// - `X B0` and `X BY` with the default kernel (bit 0 of each popcount),
/// - `X B0` with `ACC_BIT = 1` and `X B1`, so that the product terms of `x_g * P(S)` are
///   `ones = bit0(X B0)`, `twos = bit1(X B0) + bit0(X B1)`.
///
/// `build` writes `X`, `B0`, `B1`, `BY` and their shapes to `dir`; `gemm` runs the products and
/// checks every entry against the sum of the slab rows R in `x_g` on the CPU, and prints the
/// kernel-only times beside an estimate of the walk for the same products (the H200 steady-state
/// fit, 0.557 * D^0.81 ns per pair of total degree D).
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") if args.len() == 9 => build(&args[2..]),
        Some("gemm") if args.len() >= 3 => gemm(&args[2], args.get(3).map_or(Ok(5), |v| v.parse())?),
        _ => anyhow::bail!("see the usage in the source"),
    }
}

/// The `build` mode: pick the batch, build `X` and the slabs, write them to `dir`.
fn build(args: &[String]) -> anyhow::Result<()> {
    let store = std::path::PathBuf::from(&args[0]);
    let max = Bidegree::n_s(args[1].parse()?, args[2].parse()?);
    let b = Bidegree::n_s(args[3].parse()?, args[4].parse()?);
    let max_slabs: usize = args[5].parse()?;
    let dir = std::path::PathBuf::from(&args[6]);

    let start = Instant::now();
    let res = Arc::new(ext::utils::construct("S_2@milnor", Some(store))?);
    res.compute_through_stem(max);
    let algebra = res.algebra();
    eprintln!("PROTO open_s={:.1}", start.elapsed().as_secs_f64());

    let (s, t) = (b.s(), b.t());
    let (d, d_below) = (res.differential(s), res.differential(s - 1));
    let (module, mid, bot) = (res.module(s), res.module(s - 1), res.module(s - 2));

    // The g1 with the most ops of degree t into it.
    let mut ops: HashMap<(i32, usize), Vec<FpVector>> = HashMap::new();
    for g in 0..module.number_of_gens_in_degree(t) {
        let dg = d.output(t, g);
        for (g1d, g1i, _, x) in mid.iter_slices(t, dg.as_slice()) {
            if !x.is_zero() {
                let mut v = FpVector::new(TWO, x.len());
                v.as_slice_mut().add(x, 1);
                ops.entry((g1d, g1i)).or_default().push(v);
            }
        }
    }
    let (&(g1d, g1i), xs) = ops
        .iter()
        .max_by_key(|(key, xs)| (xs.len(), std::cmp::Reverse(**key)))
        .ok_or_else(|| anyhow::anyhow!("no generators at {b}"))?;
    let k = t - g1d;
    let dim_k = algebra.dimension(k);

    // The component degree of d(g1) with the most distinct basis elements.
    let dg1 = d_below.output(g1d, g1i);
    let mut by_e: HashMap<i32, std::collections::BTreeSet<usize>> = HashMap::new();
    for (g2d, _, _, y) in bot.iter_slices(g1d, dg1.as_slice()) {
        by_e.entry(g1d - g2d)
            .or_default()
            .extend(y.iter_nonzero().map(|(i, _)| i));
    }
    let (&e, set) = by_e
        .iter()
        .max_by_key(|(e, set)| (set.len(), std::cmp::Reverse(**e)))
        .ok_or_else(|| anyhow::anyhow!("d(g1) is zero"))?;
    let slabs: Vec<usize> = set.iter().copied().take(max_slabs).collect();
    // How many components y of degree e contain each kept S: the walk multiplies (R, S) once per
    // such g2, the GEMM once.
    let mut mult = 0usize;
    for (g2d, _, _, y) in bot.iter_slices(g1d, dg1.as_slice()) {
        if g1d - g2d == e {
            mult += slabs.iter().filter(|&&sidx| y.entry(sidx) != 0).count();
        }
    }
    eprintln!(
        "PROTO g2 multiplicity: {mult} (S, g2) incidences over {} slabs, {:.2} per slab",
        slabs.len(),
        mult as f64 / slabs.len().max(1) as f64
    );
    let deg = k + e;
    let m = xs.len();
    eprintln!(
        "PROTO batch g1=({g1d},{g1i}) k={k} dim_k={dim_k} M={m} e={e} distinct_S={} slabs={} \
         pair_degree={deg}",
        set.len(),
        slabs.len()
    );

    // The slabs: per R, per S, sigma(P(R), P(S)) split into planes.
    let start = Instant::now();
    let probe = algebra.new_pair_element(deg);
    let (o, _, ys) = probe.parts();
    let dim = o.len();
    let ylen: usize = ys.iter().flatten().map(|v| v.len()).sum();
    let rows: Vec<(FpVector, FpVector, FpVector)> = (0..dim_k)
        .into_maybe_par_iter()
        .map(|r| {
            let mut b0 = FpVector::new(TWO, dim * slabs.len());
            let mut b1 = FpVector::new(TWO, dim * slabs.len());
            let mut by = FpVector::new(TWO, ylen * slabs.len());
            for (j, &sidx) in slabs.iter().enumerate() {
                let mut elt = algebra.new_pair_element(deg);
                algebra.sigma_multiply_basis(&mut elt, 1, k, r, e, sidx);
                let (ones, twos, ys) = elt.parts();
                for (i, _) in ones.iter_nonzero() {
                    b0.set_entry(j * dim + i, 1);
                }
                for (i, _) in twos.iter_nonzero() {
                    b1.set_entry(j * dim + i, 1);
                }
                let mut off = j * ylen;
                for v in ys.iter().flatten() {
                    for (i, _) in v.iter_nonzero() {
                        by.set_entry(off + i, 1);
                    }
                    off += v.len();
                }
            }
            (b0, b1, by)
        })
        .collect();
    let slab_s = start.elapsed().as_secs_f64();
    let (mut v0, mut v1, mut vy) = (Vec::new(), Vec::new(), Vec::new());
    for (b0, b1, by) in rows {
        v0.push(b0);
        v1.push(b1);
        vy.push(by);
    }
    let b0 = Matrix::from_rows(TWO, v0, dim * slabs.len());
    let b1 = Matrix::from_rows(TWO, v1, dim * slabs.len());
    let by = Matrix::from_rows(TWO, vy, ylen * slabs.len());
    let x = Matrix::from_rows(TWO, xs.clone(), dim_k);
    let nnz_x: usize = xs.iter().map(|v| v.iter_nonzero().count()).sum();
    let density = |mat: &Matrix| {
        (0..mat.rows())
            .map(|r| mat.row(r).iter_nonzero().count())
            .sum::<usize>() as f64
            / (mat.rows() * mat.columns()).max(1) as f64
    };
    eprintln!(
        "PROTO slabs built in {slab_s:.1}s: B0 {}x{} density {:.2e}, BY {}x{} density {:.2e}, X \
         density {:.3}",
        b0.rows(),
        b0.columns(),
        density(&b0),
        by.rows(),
        by.columns(),
        density(&by),
        nnz_x as f64 / (m * dim_k) as f64
    );

    let pairs = nnz_x * slabs.len();
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("meta.txt"),
        format!(
            "{} {s} {m} {dim_k} {} {} {deg} {pairs}\n",
            b.n(),
            b0.columns(),
            by.columns()
        ),
    )?;
    for (name, mat) in [("x", &x), ("b0", &b0), ("b1", &b1), ("by", &by)] {
        let mut f = std::io::BufWriter::new(std::fs::File::create(dir.join(format!("{name}.bin")))?);
        mat.to_bytes(&mut f)?;
    }
    eprintln!("PROTO wrote {}", dir.display());
    Ok(())
}

/// The `gemm` mode: load the matrices from `dir`, run the four products, check, time.
fn gemm(dir: &str, iters: usize) -> anyhow::Result<()> {
    let dir = std::path::Path::new(dir);
    let meta: Vec<usize> = std::fs::read_to_string(dir.join("meta.txt"))?
        .split_whitespace()
        .map(|v| v.parse())
        .collect::<Result<_, _>>()?;
    let [n, s, m, dim_k, n0, ny, deg, pairs] = meta[..] else {
        anyhow::bail!("bad meta.txt");
    };
    let load = |name: &str, rows: usize, cols: usize| -> anyhow::Result<Matrix> {
        let mut f = std::io::BufReader::new(std::fs::File::open(dir.join(format!("{name}.bin")))?);
        Ok(Matrix::from_bytes(TWO, rows, cols, &mut f)?)
    };
    let (x, b0, b1, by) = (
        load("x", m, dim_k)?,
        load("b0", dim_k, n0)?,
        load("b1", dim_k, n0)?,
        load("by", dim_k, ny)?,
    );

    let gpu0 = GpuContext::new(0)?;
    let gpu1 = GpuContext::with_acc_bit(0, 1)?;
    let (mut t0, mut t1, mut t2, mut t3) = (0.0, 0.0, 0.0, 0.0);
    let mut bad = 0usize;
    let xrows: Vec<Vec<usize>> = (0..m)
        .map(|g| x.row(g).iter_nonzero().map(|(i, _)| i).collect())
        .collect();
    // The launch's grids are bounded (65535 blocks along y), so N goes in chunks of CHUNK columns.
    const CHUNK: usize = 1 << 22;
    let columns = |mat: &Matrix, a: usize, e: usize| {
        let rows = (0..mat.rows())
            .map(|r| {
                let mut v = FpVector::new(TWO, e - a);
                v.as_slice_mut().add(mat.row(r).restrict(a, e), 1);
                v
            })
            .collect();
        Matrix::from_rows(TWO, rows, e - a)
    };
    for a in (0..n0).step_by(CHUNK) {
        let e = (a + CHUNK).min(n0);
        let (c0, c1) = (columns(&b0, a, e), columns(&b1, a, e));
        let (p0, d0) = x.cuda_mul_timed(&gpu0, &c0, iters)?;
        let (p0b1, d1) = x.cuda_mul_timed(&gpu1, &c0, iters)?;
        let (p1, d2) = x.cuda_mul_timed(&gpu0, &c1, iters)?;
        (t0, t1, t2) = (t0 + d0, t1 + d1, t2 + d2);
        // Check: the sum over R in x_g of the slab rows, mod 4 across the two planes.
        for (g, rs) in xrows.iter().enumerate() {
            for c in 0..e - a {
                let sum: u32 = rs
                    .iter()
                    .map(|&r| c0.row(r).entry(c) + 2 * c1.row(r).entry(c))
                    .sum::<u32>()
                    % 4;
                let got =
                    p0.row(g).entry(c) + 2 * ((p0b1.row(g).entry(c) + p1.row(g).entry(c)) % 2);
                bad += (got != sum) as usize;
            }
        }
    }
    for a in (0..ny).step_by(CHUNK) {
        let e = (a + CHUNK).min(ny);
        let cy = columns(&by, a, e);
        let (py, d3) = x.cuda_mul_timed(&gpu0, &cy, iters)?;
        t3 += d3;
        for (g, rs) in xrows.iter().enumerate() {
            for c in 0..e - a {
                let sum: u32 = rs.iter().map(|&r| cy.row(r).entry(c)).sum::<u32>() % 2;
                bad += (py.row(g).entry(c) != sum) as usize;
            }
        }
    }
    let gemm_s = t0 + t1 + t2 + t3;

    let walk_s = pairs as f64 * 0.557 * (deg as f64).powf(0.81) * 1e-9;
    println!(
        "PROTO ({n},{s}) M={m} K={dim_k} N0={n0} NY={ny} kernel_s: b0={t0:.6} b0bit1={t1:.6} \
         b1={t2:.6} y={t3:.6} total={gemm_s:.6} walk_equiv_s={walk_s:.6} pairs={pairs} \
         walk_over_gemm={:.2} mismatches={bad}",
        walk_s / gemm_s
    );
    anyhow::ensure!(bad == 0, "{bad} entries differ from the CPU");
    Ok(())
}
