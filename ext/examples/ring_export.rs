//! Exports the multiplicative structure of $\Ext_A(\F_2, \F_2)$ as CSV files, computed once through
//! an [`ExtAlgebra`] by W. Lin's method (see [`ext::ext_algebra::lin_ring`]). Charts and other
//! formats are meant to be derived from these files rather than recomputed.
//!
//! The module must be `S_2` over the Milnor basis. After the resolution, this asks for the output
//! directory, the stem and internal degree `t` to compute the ring through, and whether to write the
//! full table of products of basis elements, and whether to keep the chain maps in the
//! resolution's save store (see [`ExtAlgebra::compute_ring_saving_maps`]). It writes, in the format of the
//! [stem 256 data set](https://doi.org/10.5281/zenodo.8021736):
//!
//! * `rank.csv`: `stem,Adams_filtration,rank` for every non-zero bidegree resolved. The basis of each
//!   bidegree, `x_(n, s, i)`, is the dual of the resolution's generators.
//! * `indecomposables.csv`: `stem,Adams_filtration,index,name`. The `h_i` are listed through the
//!   whole range; the others through the ring's range. A generator `x_(n, s, i)` is indecomposable
//!   modulo the products of the earlier ones (ordered by `t`, then `s`).
//! * `generator_products.csv`: `stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,
//!   product`, the non-zero products of an indecomposable (first) with a basis element (second).
//!   Products by `h_i` cover the whole resolved range; the rest cover the ring's range. These
//!   generate every product.
//! * `products.csv` (optional): the same columns for every pair of basis elements whose product lies
//!   in the ring's range, each unordered pair once (first `<=` second as `(n, s, i)`).
//! * `SPEC.md`: the ranges and conventions.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use algebra::AlgebraType;
use anyhow::Context;
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::{ExtAlgebra, Ring, RingRange},
    utils::query_module,
};
use fp::vector::FpSlice;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

/// A vector of `F_2` coefficients as in the data set: `[1]`, or `"[0, 1, 0]"` (quoted).
fn vector<I: Iterator<Item = u32>>(v: I) -> String {
    let entries: Vec<String> = v.map(|c| c.to_string()).collect();
    if entries.len() == 1 {
        format!("[{}]", entries[0])
    } else {
        format!("\"[{}]\"", entries.join(", "))
    }
}

fn csv(dir: &std::path::Path, name: &str, header: &str) -> anyhow::Result<BufWriter<File>> {
    let path = dir.join(name);
    let mut w = BufWriter::new(File::create(&path).with_context(|| format!("creating {path:?}"))?);
    writeln!(w, "{header}")?;
    Ok(w)
}

const PRODUCT_HEADER: &str =
    "stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,product";

/// Write the non-zero `rows`, the products of `x` with the basis of `c`.
fn write_rows<'a>(
    w: &mut impl Write,
    x: BidegreeGenerator,
    c: Bidegree,
    rows: impl Iterator<Item = FpSlice<'a>>,
    first_y: usize,
) -> anyhow::Result<usize> {
    let mut n = 0;
    for (y, row) in rows.enumerate().skip(first_y) {
        if row.is_zero() {
            continue;
        }
        writeln!(
            w,
            "{},{},{},{},{},{y},{}",
            x.n(),
            x.s(),
            x.idx(),
            c.n(),
            c.s(),
            vector(row.iter())
        )?;
        n += 1;
    }
    Ok(n)
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let res = Arc::new(query_module(Some(AlgebraType::Milnor), true)?);
    let dir: PathBuf = query::with_default("Output directory", "ring_export", |s| {
        Ok::<_, std::convert::Infallible>(PathBuf::from(s))
    });
    let max_s = res.iter_stem().map(|b| b.s()).max().unwrap_or(0);
    let ring_n: i32 = query::with_default("Ring through stem", "150", str::parse);
    let max_t: i32 = query::with_default(
        "Ring through internal degree t",
        &(ring_n + max_s).to_string(),
        str::parse,
    );
    let full: bool = query::yes_no("Write the full products table");
    let save_maps: bool = query::yes_no("Save the chain maps in the resolution's store");
    std::fs::create_dir_all(&dir)?;

    let range = RingRange {
        max_n: ring_n,
        max_s,
        max_t,
    };
    let ext = ExtAlgebra::new(Arc::clone(&res));
    let max_n = res.iter_stem().map(|b| b.n()).max().unwrap_or(0);

    let mut all: Vec<Bidegree> = res
        .iter_stem()
        .filter(|&b| res.number_of_gens_in_bidegree(b) > 0)
        .collect();
    all.sort_by_key(|b| (b.n(), b.s()));
    // The bidegrees in the ring's range by internal degree, each list in (n, s) order.
    // The ring is computed in windows of `t`, or of stems with `EXT_LIN_RING_ORDER=stem`; each
    // window's rows are written when it is done.
    let by_stem = std::env::var("EXT_LIN_RING_ORDER").is_ok_and(|v| v == "stem");
    let deg = move |b: Bidegree| if by_stem { b.n() } else { b.t() };
    let mut by_t: std::collections::HashMap<i32, Vec<Bidegree>> = Default::default();
    for &b in all.iter().filter(|&&b| range.contains(b)) {
        by_t.entry(deg(b)).or_default().push(b);
    }

    let mut w = csv(&dir, "rank.csv", "stem,Adams_filtration,rank")?;
    for &b in &all {
        writeln!(w, "{},{},{}", b.n(), b.s(), ext.dimension(b))?;
    }
    w.flush()?;

    // The h_i and their products need no ring: write them through the whole range first.
    let mut indec = csv(
        &dir,
        "indecomposables.csv",
        "stem,Adams_filtration,index,name",
    )?;
    let mut gen_w = csv(&dir, "generator_products.csv", PRODUCT_HEADER)?;
    let mut gen_products = 0;
    let mut n_indec = 0;
    for i in 0.. {
        let b = Bidegree::s_t(1, 1 << i);
        if !res.has_computed_bidegree(b) {
            break;
        }
        if ext.dimension(b) != 1 {
            continue;
        }
        let g = BidegreeGenerator::new(b, 0);
        writeln!(indec, "{},{},{},h_{i}", g.n(), g.s(), g.idx())?;
        n_indec += 1;
        let x = ext.generator(g);
        for &c in &all {
            if let Some(m) = ext.known_multiply_into(&x, c) {
                gen_products += write_rows(&mut gen_w, g, c, m.iter(), 0)?;
            }
        }
    }
    indec.flush()?;
    gen_w.flush()?;
    let mut full_w = if full {
        Some(csv(&dir, "products.csv", PRODUCT_HEADER)?)
    } else {
        None
    };
    let mut products = 0;

    let start = std::time::Instant::now();
    let write_spec = |decided_t: i32,
                      done: bool,
                      gens_n: usize,
                      basis_n: usize,
                      maps_n: usize,
                      gen_products: usize,
                      products: usize|
     -> anyhow::Result<()> {
        let mut w = BufWriter::new(File::create(dir.join("SPEC.md"))?);
        writeln!(
            w,
            "# Ext_A(F_2, F_2): ranks and products\n\n\
             Computed with ext-rs ({}) by W. Lin's method (`ext::ext_algebra::lin_ring`).\n\n\
             - Status: {} through {} <= {decided_t}, after {:.0} s.\n\
             - Resolution: stems 0..={max_n}, Adams filtration 0..={max_s}. `rank.csv` covers it \
             all.\n\
             - Ring range: stem <= {ring_n}, Adams filtration <= {max_s}, t <= {max_t}; so far \
             {gens_n} indecomposables, {basis_n} basis monomials, {maps_n} chain maps.\n\
             - `x_(n, s, i)` is the dual of the i-th generator of the resolution in bidegree (n, \
             s).\n\
             - `indecomposables.csv`: the h_i through the whole range, then the others as they are \
             found. Others are indecomposable modulo products of earlier generators ({}).\n\
             - `generator_products.csv`: {gen_products} non-zero products indecomposable x basis \
             element; h_i over the whole range, the rest over the decided part of the ring range.\n\
             - `products.csv`: {} non-zero products of pairs of basis elements in the decided part \
             of the ring range, each unordered pair once.\n\
             - Rows are grouped by the window ({}) in which the product lands. \
             Products of a pair whose product leaves the range are absent, not zero.",
            env!("CARGO_PKG_VERSION"),
            if done {
                "complete"
            } else {
                "in progress, decided"
            },
            if by_stem { "stem" } else { "internal degree t" },
            start.elapsed().as_secs_f64(),
            if by_stem {
                "by stem, then s"
            } else {
                "by t, then s"
            },
            if full {
                products.to_string()
            } else {
                "(not written)".to_string()
            },
            if by_stem {
                "of 8 stems"
            } else {
                "of 11 internal degrees"
            },
        )?;
        Ok(())
    };

    let mut last_t = 0;
    let mut on_window = |ring: &mut Ring<_>, t0: i32, t1: i32| -> anyhow::Result<()> {
        let t = std::time::Instant::now();
        // Indecomposables found in this window (the h_i are already written).
        for g in ring
            .gens
            .iter()
            .filter(|g| g.s() != 1 && (t0..=t1).contains(&deg(g.degree())))
        {
            writeln!(indec, "{},{},{},", g.n(), g.s(), g.idx())?;
            n_indec += 1;
        }
        // Products by the other indecomposables landing in this window, one ring lookup per pair
        // of bidegrees.
        let mut degrees: Vec<Bidegree> = ring
            .gens
            .iter()
            .filter(|g| g.s() != 1)
            .map(|g| g.degree())
            .collect();
        degrees.dedup();
        for &b in &degrees {
            let gens_here: Vec<usize> = ring
                .gens
                .iter()
                .filter(|g| g.degree() == b)
                .map(|g| g.idx())
                .collect();
            for tt in t0..=t1 {
                for &c in by_t.get(&(tt - deg(b))).into_iter().flatten() {
                    if !ring.range().contains(b + c) || res.number_of_gens_in_bidegree(b + c) == 0 {
                        continue;
                    }
                    let block = ring.basis_block(b, c);
                    for &x in &gens_here {
                        let rows = block[x].iter().map(|v| v.as_slice());
                        gen_products +=
                            write_rows(&mut gen_w, BidegreeGenerator::new(b, x), c, rows, 0)?;
                    }
                }
            }
        }
        if let Some(w) = &mut full_w {
            // Unordered pairs b <= c as (n, s) whose product lands in this window.
            for tt in t0..=t1 {
                for tb in 0..=tt / 2 {
                    for &b in by_t.get(&tb).into_iter().flatten() {
                        for &c in by_t.get(&(tt - tb)).into_iter().flatten() {
                            if (c.n(), c.s()) < (b.n(), b.s())
                                || !ring.range().contains(b + c)
                                || res.number_of_gens_in_bidegree(b + c) == 0
                            {
                                continue;
                            }
                            let block = ring.basis_block(b, c);
                            for (x, rows) in block.iter().enumerate() {
                                let first = if c == b { x } else { 0 };
                                let rows = rows.iter().map(|v| v.as_slice());
                                products +=
                                    write_rows(w, BidegreeGenerator::new(b, x), c, rows, first)?;
                            }
                        }
                    }
                }
            }
            w.flush()?;
        }
        indec.flush()?;
        gen_w.flush()?;
        last_t = t1;
        write_spec(
            t1,
            false,
            ring.gens.len(),
            ring.basis.len(),
            ring.maps(),
            gen_products,
            products,
        )?;
        tracing::info!(
            t0,
            t1,
            export_s = t.elapsed().as_secs_f64(),
            gen_products,
            "ring_export window"
        );
        Ok(())
    };
    ext.compute_ring_with(range, save_maps, &mut on_window)?;
    let done = ext
        .ring_range()
        .is_some_and(|r| r.max_t == max_t && r.max_n == ring_n);
    let (gens_n, basis_n, maps_n) = ext
        .with_ring(|r| (r.gens.len(), r.basis.len(), r.maps()))
        .unwrap();
    write_spec(
        last_t,
        done,
        gens_n,
        basis_n,
        maps_n,
        gen_products,
        products,
    )?;
    eprintln!(
        "ring through stem {ring_n}, t {last_t} in {:.1} s; {n_indec} indecomposables, \
         {gen_products} generator products",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
