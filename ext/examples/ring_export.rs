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
    ext_algebra::{ExtAlgebra, RingRange},
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
    let start = std::time::Instant::now();
    if save_maps {
        ext.compute_ring_saving_maps(range)?;
    } else {
        ext.compute_ring(range)?;
    }
    let ring_s = start.elapsed().as_secs_f64();
    eprintln!("ring through stem {ring_n}, t {max_t} in {ring_s:.1} s");

    let mut all: Vec<Bidegree> = res
        .iter_stem()
        .filter(|&b| res.number_of_gens_in_bidegree(b) > 0)
        .collect();
    all.sort_by_key(|b| (b.n(), b.s()));
    let in_ring: Vec<Bidegree> = all.iter().copied().filter(|&b| range.contains(b)).collect();

    let mut w = csv(&dir, "rank.csv", "stem,Adams_filtration,rank")?;
    for &b in &all {
        writeln!(w, "{},{},{}", b.n(), b.s(), ext.dimension(b))?;
    }
    w.flush()?;

    // The h_i through the whole range, then the ring's other indecomposables.
    let mut gens: Vec<(BidegreeGenerator, String)> = vec![];
    for i in 0.. {
        let b = Bidegree::s_t(1, 1 << i);
        if !res.has_computed_bidegree(b) {
            break;
        }
        if ext.dimension(b) == 1 {
            gens.push((BidegreeGenerator::new(b, 0), format!("h_{i}")));
        }
    }
    for g in ext.indecomposables().unwrap() {
        if g.s() != 1 {
            gens.push((g, String::new()));
        }
    }
    gens.sort_by_key(|(g, _)| (g.n(), g.s(), g.idx()));
    let mut w = csv(
        &dir,
        "indecomposables.csv",
        "stem,Adams_filtration,index,name",
    )?;
    for (g, name) in &gens {
        writeln!(w, "{},{},{},{name}", g.n(), g.s(), g.idx())?;
    }
    w.flush()?;

    let t = std::time::Instant::now();
    let mut w = csv(&dir, "generator_products.csv", PRODUCT_HEADER)?;
    let mut gen_products = 0;
    for (g, _) in &gens {
        let x = ext.generator(*g);
        for &c in &all {
            if let Some(m) = ext.known_multiply_into(&x, c) {
                gen_products += write_rows(&mut w, *g, c, m.iter(), 0)?;
            }
        }
    }
    w.flush()?;
    eprintln!(
        "generator_products.csv: {gen_products} rows in {:.1} s",
        t.elapsed().as_secs_f64()
    );

    let mut products = 0;
    if full {
        let t = std::time::Instant::now();
        let mut w = csv(&dir, "products.csv", PRODUCT_HEADER)?;
        for (i, &b) in in_ring.iter().enumerate() {
            // Unordered pairs: c >= b as (n, s); within b itself, y >= x.
            for &c in &in_ring[i..] {
                if !range.contains(b + c) || ext.dimension(b + c) == 0 {
                    continue;
                }
                // `block[x][y]` = x_(b, x) · x_(c, y), one ring lookup per pair of bidegrees.
                let block = ext.with_ring(|r| r.basis_block(b, c)).unwrap();
                for (x, rows) in block.iter().enumerate() {
                    let g = BidegreeGenerator::new(b, x);
                    let first = if c == b { x } else { 0 };
                    products += write_rows(&mut w, g, c, rows.iter().map(|v| v.as_slice()), first)?;
                }
            }
        }
        w.flush()?;
        eprintln!(
            "products.csv: {products} rows in {:.1} s",
            t.elapsed().as_secs_f64()
        );
    }

    let max_n = all.iter().map(|b| b.n()).max().unwrap_or(0);
    let (gens_n, basis_n, maps_n) = ext
        .with_ring(|r| (r.gens.len(), r.basis.len(), r.maps()))
        .unwrap();
    let mut w = BufWriter::new(File::create(dir.join("SPEC.md"))?);
    writeln!(
        w,
        "# Ext_A(F_2, F_2): ranks and products\n\n\
         Computed with ext-rs ({}) by W. Lin's method (`ext::ext_algebra::lin_ring`).\n\n\
         - Resolution: stems 0..={max_n}, Adams filtration 0..={max_s}. `rank.csv` covers it all.\n\
         - Ring range: stem <= {ring_n}, Adams filtration <= {max_s}, t <= {max_t}; {gens_n} \
         indecomposables, {basis_n} basis monomials, {maps_n} chain maps, {ring_s:.0} s.\n\
         - `x_(n, s, i)` is the dual of the i-th generator of the resolution in bidegree (n, s).\n\
         - `indecomposables.csv`: the h_i through the whole range, the others through the ring \
         range. Others are indecomposable modulo products of earlier generators (by t, then s).\n\
         - `generator_products.csv`: {gen_products} non-zero products indecomposable x basis \
         element; h_i over the whole range, the rest over the ring range.\n\
         - `products.csv`: {} non-zero products of pairs of basis elements in the ring range, each \
         unordered pair once.\n\
         - Products of a pair whose product leaves the range are absent, not zero.",
        env!("CARGO_PKG_VERSION"),
        if full {
            products.to_string()
        } else {
            "(not written)".to_string()
        }
    )?;
    w.flush()?;
    Ok(())
}
