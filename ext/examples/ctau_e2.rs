//! Exports the Adams E2 page of Cτ and of its two-cell complexes, from a saved resolution of the classical sphere.
//!
//! Nothing is resolved or lifted: the run stops if the save lacks a bidegree it would read, and a
//! product whose chain map step was never saved is reported as unknown.
//!
//! Tridegrees are motivic `stem,filtration,weight`, and a class is an index into the basis of its
//! tridegree; vectors list coordinates in the target's basis. Files, in the output directory:
//! * `classical_rank.csv`: the generators of the backbone (`filtration,degree,rank`);
//! * `Ctau_rank.csv`, and `<C>_rank.csv` for each complex `<C>` among `C2`, `Ceta`, `Cnu`,
//!   `Csigma` (`Ctau` smashed with the cofiber of `2`, `η`, `ν`, `σ`);
//! * `<C>_h.csv`: the action of `h_0, ..., h_4` (`j,class,product`);
//! * `<C>_istar.csv`, `<C>_pstar.csv`: the maps induced by the bottom cell `Ctau → C` and the top
//!   cell `C → Σ^{|cell|} Ctau`;
//! * `Ctau_indaction.csv`: the action of the classical indecomposables with a saved chain map,
//!   which act as classes of twice the stem plus the filtration (`classical class,class,product`),
//!   and `Ctau_unknown.csv`, the pairs whose step is not saved;
//! * `Ctau_products.csv`: the products `β · α` for every `β` of weight up to the product bound
//!   (`β,α,product`), and `Ctau_commutativity.txt`, the comparison of `β · α` with `α · β`.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use algebra::AlgebraType;
use ext::{
    chain_complex::ChainComplex,
    ctau::{
        Backbone, BarComparison, Cells, ChainMapTerms, CtauExt, OpTable, QAlgebra, Tri,
        bottom_cell, top_cell,
    },
    save::SaveKind,
    utils::query_module_only,
};
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

/// A vector of coordinates, quoted for a CSV cell.
fn vector(v: &[u32]) -> String {
    let e: Vec<String> = v.iter().map(|c| c.to_string()).collect();
    format!("\"[{}]\"", e.join(", "))
}

/// A tridegree as `stem,filtration,weight`.
fn tri(x: Tri) -> String {
    format!("{},{},{}", x.stem(), x.filtration(), x.weight())
}

fn csv(dir: &std::path::Path, name: &str, header: &str) -> anyhow::Result<BufWriter<File>> {
    let mut w = BufWriter::new(File::create(dir.join(name))?);
    writeln!(w, "{header}")?;
    Ok(w)
}

/// Writes the rows of a map on cohomology out of `x` that are nonzero.
fn write_rows(
    w: &mut impl Write,
    prefix: &str,
    x: Tri,
    rows: &[Vec<u32>],
) -> anyhow::Result<usize> {
    let mut n = 0;
    for (i, row) in rows.iter().enumerate() {
        if row.iter().any(|&c| c != 0) {
            writeln!(w, "{prefix}{},{i},{}", tri(x), vector(row))?;
            n += 1;
        }
    }
    Ok(n)
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let res = Arc::new(query_module_only(
        "Module",
        Some(AlgebraType::Milnor),
        false,
    )?);
    let dir: PathBuf = query::raw("Output directory", |s| {
        Ok::<_, anyhow::Error>(PathBuf::from(s))
    });
    let max_n: i32 = query::with_default("Max motivic stem", "60", str::parse);
    let max_s: i32 = query::with_default("Max filtration", "30", str::parse);
    let specs: String = query::with_default("Complexes", "C2,Ceta,Cnu,Csigma", |s| {
        Ok::<_, anyhow::Error>(s.to_string())
    });
    let product_weight: i32 = query::with_default("Product bound (weight of β)", "0", str::parse);
    let with_lin: bool = query::with_default("Saved chain maps (y/n)", "y", |s| {
        Ok::<_, anyhow::Error>(s.starts_with('y'))
    });
    std::fs::create_dir_all(&dir)?;
    let store = res.save_dir().store().expect("a saved resolution");

    let (max_a, max_t) = CtauExt::backbone_range(max_n, max_s);
    // The save must already hold everything read; computing a missing bidegree is not allowed.
    // Filtration 0 is the unit and is never saved.
    let top_stem = (0..=max_a).map(|a| max_t(a) - a).max().unwrap();
    let mut missing = Vec::new();
    for a in 1..=max_a {
        for n in 0..=top_stem {
            let b = Bidegree::n_s(n, a);
            if !store.exists(SaveKind::NassauDifferential, b) {
                missing.push(b);
            }
        }
    }
    anyhow::ensure!(
        missing.is_empty(),
        "the save lacks {} bidegrees, the first {}; refusing to resolve",
        missing.len(),
        missing[0]
    );
    let start = Instant::now();
    res.compute_through_stem(Bidegree::n_s(top_stem, max_a));
    eprintln!(
        "backbone through classical stem {top_stem}, filtration {max_a}: {:.1} s",
        start.elapsed().as_secs_f64()
    );

    let ops = OpTable::new(res.algebra());
    let start = Instant::now();
    let bb = Backbone::new(&*res, &ops, max_a, &max_t);
    let q = QAlgebra::new((max_n + max_s) / 2, max_s as u32);
    eprintln!("differentials read: {:.1} s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let ctau = CtauExt::new(&q, &bb, Cells::sphere(), max_n, max_s);
    let classes = ctau.nonzero();
    let largest = classes
        .iter()
        .map(|x| ctau.space(x.a, x.u).unwrap().dim())
        .max()
        .unwrap_or(0);
    eprintln!(
        "Ctau: {} classes in {} tridegrees, largest cochain space {largest}: {:.1} s",
        classes.iter().map(|&x| ctau.dimension(x)).sum::<usize>(),
        classes.len(),
        start.elapsed().as_secs_f64()
    );

    /// Writes the ranks and the action of `h_0, ..., h_4` on `ext`.
    fn ranks_and_h(
        dir: &std::path::Path,
        name: &str,
        ext: &CtauExt,
        q: &QAlgebra,
        bb: &Backbone,
    ) -> anyhow::Result<()> {
        let classes = ext.nonzero();
        let mut w = csv(
            dir,
            &format!("{name}_rank.csv"),
            "stem,filtration,weight,rank",
        )?;
        for &x in &classes {
            writeln!(w, "{},{}", tri(x), ext.dimension(x))?;
        }
        let rows: Vec<Vec<(usize, Tri, Vec<Vec<u32>>)>> = classes
            .clone()
            .into_maybe_par_iter()
            .map(|x| {
                let mut out = Vec::new();
                if !ext.cells.mod_q0
                    && let Some(m) = ext.h0(x)
                {
                    out.push((0, x, m));
                }
                for j in 0..4 {
                    if let Some(m) = ext.h(q, bb, j, x) {
                        out.push((j as usize + 1, x, m));
                    }
                }
                out
            })
            .collect();
        let mut w = csv(
            dir,
            &format!("{name}_h.csv"),
            "j,stem,filtration,weight,index,product",
        )?;
        for (j, x, m) in rows.into_iter().flatten() {
            write_rows(&mut w, &format!("{j},"), x, &m)?;
        }
        Ok(())
    }

    // the classical ranks the backbone holds, for comparing with the abutment of the τ-Bockstein
    // spectral sequence
    let mut w = csv(&dir, "classical_rank.csv", "filtration,degree,rank")?;
    for a in 0..=max_a {
        for t in a..=max_t(a) {
            if bb.gens(a, t) > 0 {
                writeln!(w, "{a},{t},{}", bb.gens(a, t))?;
            }
        }
    }
    drop(w);

    let start = Instant::now();
    ranks_and_h(&dir, "Ctau", &ctau, &q, &bb)?;
    eprintln!("Ctau h_j: {:.1} s", start.elapsed().as_secs_f64());

    for name in specs.split(',').filter(|s| !s.is_empty()) {
        let start = Instant::now();
        let cells = match name {
            "C2" => Cells::moore(),
            "Ceta" => Cells::hopf(0),
            "Cnu" => Cells::hopf(1),
            "Csigma" => Cells::hopf(2),
            _ => anyhow::bail!("unknown complex {name}"),
        };
        let cof = CtauExt::new(&q, &bb, cells, max_n, max_s);
        ranks_and_h(&dir, name, &cof, &q, &bb)?;
        let mut wi = csv(
            &dir,
            &format!("{name}_istar.csv"),
            "stem,filtration,weight,index,image",
        )?;
        for &x in &classes {
            if let Some(m) = bottom_cell(&ctau, &cof, x) {
                write_rows(&mut wi, "", x, &m)?;
            }
        }
        let mut wp = csv(
            &dir,
            &format!("{name}_pstar.csv"),
            "stem,filtration,weight,index,image",
        )?;
        for x in cof.nonzero() {
            if let Some((_, m)) = top_cell(&ctau, &cof, x) {
                write_rows(&mut wp, "", x, &m)?;
            }
        }
        eprintln!(
            "{name}: {} classes: {:.1} s",
            cof.nonzero()
                .iter()
                .map(|&x| cof.dimension(x))
                .sum::<usize>(),
            start.elapsed().as_secs_f64()
        );
    }

    if with_lin {
        let start = Instant::now();
        // the indecomposables are the saved maps named by a single generator
        let mut inds: Vec<(Bidegree, usize)> = Vec::new();
        for entry in std::fs::read_dir(store.path().join("products/lin_ring"))? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            let Some(inner) = name.strip_prefix("x_(").and_then(|s| s.strip_suffix(')')) else {
                continue;
            };
            let v: Vec<&str> = inner.split(", ").collect();
            if v.len() != 3 || name.contains('.') {
                continue;
            }
            let (n, s, i): (i32, i32, usize) = (v[0].parse()?, v[1].parse()?, v[2].parse()?);
            // the class acts through Tri(s, n + s, 0), of motivic stem 2n + s
            if 2 * n + s <= max_n && s <= max_s {
                inds.push((Bidegree::n_s(n, s), i));
            }
        }
        inds.sort_by_key(|&(b, i)| (b.n(), b.s(), i));
        let results: Vec<anyhow::Result<(Vec<String>, Vec<String>)>> = inds
            .clone()
            .into_maybe_par_iter()
            .map(|(b, i)| {
                let name = format!("products/lin_ring/x_({}, {}, {})", b.n(), b.s(), i);
                let map =
                    ChainMapTerms::read(&*res, &ops, &store.subgroup(&name)?, b, i, max_a, &max_t)?;
                let by = Tri::new(b.s(), b.t(), 0);
                let label = format!("{},{},{i},", b.n(), b.s());
                let (mut known, mut unknown) = (Vec::new(), Vec::new());
                for &x in &classes {
                    let y = x + by;
                    if ctau.dimension(y) == 0 {
                        continue;
                    }
                    let tgt = ctau.space(y.a, y.u).unwrap();
                    let held = (0..=tgt.u)
                        .all(|t| bb.gens(y.a, t) == 0 || map.values.contains_key(&(y.a, t)));
                    if !held {
                        unknown.push(format!("{label}{}", tri(x)));
                        continue;
                    }
                    let m = ctau
                        .operator(&q, &bb, x.a, x.u, by, |t, l| map.get(y.a, t, l).unwrap(), 0)
                        .unwrap();
                    let rows = ctau.induced(x, y, &m).unwrap();
                    let mut buf = Vec::new();
                    write_rows(&mut buf, &label, x, &rows)?;
                    known.push(String::from_utf8(buf)?);
                }
                Ok((known, unknown))
            })
            .collect();
        let mut w = csv(
            &dir,
            "Ctau_indaction.csv",
            "classical_stem,classical_filtration,classical_index,stem,filtration,weight,index,product",
        )?;
        let mut wu = csv(
            &dir,
            "Ctau_unknown.csv",
            "classical_stem,classical_filtration,classical_index,stem,filtration,weight",
        )?;
        let (mut nk, mut nu) = (0, 0);
        for r in results {
            let (known, unknown) = r?;
            for k in known {
                nk += k.lines().count();
                write!(w, "{k}")?;
            }
            for u in unknown {
                nu += 1;
                writeln!(wu, "{u}")?;
            }
        }
        eprintln!(
            "saved chain maps: {} indecomposables, {nk} nonzero products, {nu} unknown pairs: {:.1} s",
            inds.len(),
            start.elapsed().as_secs_f64()
        );
    }

    if product_weight > 0 {
        let start = Instant::now();
        let cmp = BarComparison::new(&*res, &ops);
        let betas: Vec<Tri> = classes
            .iter()
            .copied()
            .filter(|x| x.u <= product_weight && x.u > 0)
            .collect();
        // products[(β, i, α)]: the rows of β_i · α_j over j
        let results: Vec<((Tri, usize), Vec<(Tri, Vec<Vec<u32>>)>)> = betas
            .iter()
            .flat_map(|&b| (0..ctau.dimension(b)).map(move |i| (b, i)))
            .collect::<Vec<_>>()
            .into_maybe_par_iter()
            .map(|(b, i)| {
                let gamma = cmp.cocycle(&q, &ctau, b, i);
                let mut out = Vec::new();
                for &x in &classes {
                    let y = x + b;
                    if ctau.dimension(y) == 0 {
                        continue;
                    }
                    let rows = (0..ctau.dimension(x))
                        .map(|j| {
                            let c = ctau.representative(x, j);
                            let p = gamma.cup(&q, &bb, &ctau, x.a, x.u, &c).unwrap();
                            ctau.class(y, &p).unwrap()
                        })
                        .collect();
                    out.push((x, rows));
                }
                ((b, i), out)
            })
            .collect();
        let mut w = csv(
            &dir,
            "Ctau_products.csv",
            "stem_1,filtration_1,weight_1,index_1,stem_2,filtration_2,weight_2,index_2,product",
        )?;
        let mut table = std::collections::HashMap::new();
        let mut n = 0;
        for ((b, i), out) in &results {
            for (x, rows) in out {
                n += write_rows(&mut w, &format!("{},{i},", tri(*b)), *x, rows)?;
                for (j, row) in rows.iter().enumerate() {
                    table.insert((*b, *i, *x, j), row.clone());
                }
            }
        }
        let (mut agree, mut differ) = (0, 0);
        let mut wc = BufWriter::new(File::create(dir.join("Ctau_commutativity.txt"))?);
        for (&(b, i, x, j), row) in &table {
            if let Some(other) = table.get(&(x, j, b, i)) {
                if other == row {
                    agree += 1;
                } else {
                    differ += 1;
                    writeln!(
                        wc,
                        "{} #{i} times {} #{j}: {row:?} vs {other:?}",
                        tri(b),
                        tri(x)
                    )?;
                }
            }
        }
        writeln!(wc, "{agree} ordered pairs agree, {differ} differ")?;
        eprintln!(
            "products by {} classes of weight <= {product_weight}: {n} nonzero, commutativity {agree} agree / {differ} differ: {:.1} s",
            results.len(),
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
