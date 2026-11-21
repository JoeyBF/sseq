//! Exports the Adams E2 page of Cτ and its two-cell complexes as towers, by sparse reduction on a saved resolution.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use algebra::AlgebraType;
use ext::{
    chain_complex::ChainComplex,
    ctau::{BarComparison, OpTable, QAlgebra, QPoly, Tri},
    ctau_scale::{
        Complex, NEVER, OpTables, ScaledExt, StreamedBackbone, bar_cup, induced, pullback_columns,
    },
    save::SaveKind,
    utils::query_module_only,
};
use fp::matrix::SparseWorkspace;
use maybe_rayon::prelude::*;

/// The rows of one map on towers: each source tower id with the target tower ids.
type Rows = Vec<(u32, Vec<u32>)>;

/// Writes `prefix,source,targets` for the rows with a nonzero image.
fn write_rows(w: &mut impl Write, prefix: &str, rows: &Rows) -> anyhow::Result<()> {
    for (source, targets) in rows {
        if !targets.is_empty() {
            let t: Vec<String> = targets.iter().map(u32::to_string).collect();
            writeln!(w, "{prefix}{source},{}", t.join(" "))?;
        }
    }
    Ok(())
}

fn csv(dir: &std::path::Path, name: &str, header: &str) -> anyhow::Result<BufWriter<File>> {
    let mut w = BufWriter::new(File::create(dir.join(name))?);
    writeln!(w, "{header}")?;
    Ok(w)
}

/// The peak resident memory of this process so far, in GB.
fn peak_gb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1e6)
}

/// Finds the tower generators of `ctau` of weight at most `bound` that are not sums of products
/// of generators of lower weight and multiples of `h_0`, and writes them and their products
/// with every tower generator.
///
/// A product by `β` goes through a cocycle of `β` on the bar resolution, which needs the
/// resolution itself in the degrees of `β`: `res` is loaded that far, and no further.
fn products(
    dir: &std::path::Path,
    res: &ext::utils::QueryModuleResolution,
    ops: &OpTable,
    q: &QAlgebra,
    bb: &StreamedBackbone,
    ctau: &ScaledExt,
    bound: i32,
) -> anyhow::Result<()> {
    use std::collections::BTreeSet;

    use sseq::coordinates::Bidegree;
    let start = Instant::now();
    let store = res.save_dir().store().expect("a saved resolution");
    for s in 1..=bound + 1 {
        for n in 0..=bound {
            anyhow::ensure!(
                store.exists(SaveKind::NassauDifferential, Bidegree::n_s(n, s)),
                "the save lacks ({n}, {s}); refusing to resolve"
            );
        }
    }
    res.compute_through_stem(Bidegree::n_s(bound, bound + 1));
    let cmp = BarComparison::new(res, ops);
    // landing[(a, u)]: the products that land there, by level
    let mut landing: std::collections::HashMap<(i32, i32), Vec<(u16, Vec<u32>)>> =
        Default::default();
    let mut wg = csv(dir, "Ctau_generators.csv", "id,stem,filtration,weight")?;
    let mut wp = csv(dir, "Ctau_products.csv", "generator,source,targets")?;
    let (mut count, mut nonzero) = (0, 0);
    for weight in 1..=bound {
        let keys: Vec<(i32, i32)> = ctau
            .keys()
            .iter()
            .copied()
            .filter(|key| key.1 == weight)
            .collect();
        for (a, u) in keys {
            let red = ctau.reduced(a, u).unwrap();
            let space = ctau.space(q, bb, a, u);
            // the span, at each level, of the products and of the generators chosen so far,
            // modulo the towers born below the level, as vectors with distinct largest entries
            let mut spans: std::collections::HashMap<u16, Vec<BTreeSet<u32>>> = Default::default();
            let birth_of = |id: u32| red.tower((id - ctau.tower_id(a, u, 0)) as usize).0;
            let reduce = |span: &[BTreeSet<u32>], mut v: BTreeSet<u32>| {
                while let Some(&top) = v.last() {
                    match span.iter().find(|b| b.last() == Some(&top)) {
                        Some(b) => v = v.symmetric_difference(b).copied().collect(),
                        None => break,
                    }
                }
                v
            };
            for (level, ids) in landing.remove(&(a, u)).unwrap_or_default() {
                let v: BTreeSet<u32> = ids.into_iter().filter(|&t| birth_of(t) == level).collect();
                let span = spans.entry(level).or_default();
                let v = reduce(span, v);
                if !v.is_empty() {
                    span.push(v);
                }
            }
            for i in 0..red.towers.len() {
                let (birth, _) = red.tower(i);
                let id = ctau.tower_id(a, u, i as u32);
                let span = spans.entry(birth).or_default();
                let v = reduce(span, BTreeSet::from([id]));
                if v.is_empty() {
                    continue;
                }
                span.push(v);
                writeln!(wg, "{id},{},{},{u}", 2 * u - a, a + birth as i32)?;
                count += 1;
                let mut g: rustc_hash::FxHashMap<(i32, u32), QPoly> = Default::default();
                for &pos in red.generator(i) {
                    let (_, t, x, m, _) = space.decode(q, pos);
                    g.entry((t as i32, x)).or_default().push(m);
                }
                let gamma = cmp.cocycle_of(q, Tri::new(a, u, birth as i32), &g);
                let rows: Vec<(Rows, (i32, i32), Vec<(u16, Vec<u32>)>)> = ctau
                    .keys()
                    .to_vec()
                    .into_maybe_par_iter()
                    .map(|(a0, u0)| {
                        let (a2, u2) = (a0 + a, u0 + u);
                        let from = ctau.reduced(a0, u0).unwrap();
                        let Some(to) = ctau.reduced(a2, u2) else {
                            return (Vec::new(), (a2, u2), Vec::new());
                        };
                        if from.towers.is_empty() {
                            return (Vec::new(), (a2, u2), Vec::new());
                        }
                        let src = ctau.space(q, bb, a0, u0);
                        let tgt = ctau.space(q, bb, a2, u2);
                        let mut ws = SparseWorkspace::new(0);
                        let mut rows = Vec::new();
                        let mut landed = Vec::new();
                        for j in 0..from.towers.len() {
                            let entries = bar_cup(q, bb, &gamma, &src, &tgt, from.generator(j));
                            let level = from.tower(j).0 + birth;
                            let targets: Vec<u32> = to
                                .class(&mut ws, entries, level)
                                .into_iter()
                                .map(|t| ctau.tower_id(a2, u2, t))
                                .collect();
                            if !targets.is_empty() {
                                landed.push((level, targets.clone()));
                            }
                            rows.push((ctau.tower_id(a0, u0, j as u32), targets));
                        }
                        (rows, (a2, u2), landed)
                    })
                    .collect();
                for (rows, key, landed) in rows {
                    nonzero += rows.iter().filter(|r| !r.1.is_empty()).count();
                    write_rows(&mut wp, &format!("{id},"), &rows)?;
                    if key.1 <= bound && key.1 > weight {
                        landing.entry(key).or_default().extend(landed);
                    }
                }
            }
        }
        q.clear_memo();
    }
    eprintln!(
        "products: {count} generators of weight <= {bound}, {nonzero} nonzero products: {:.1} s, \
         peak {:.1} GB",
        start.elapsed().as_secs_f64(),
        peak_gb()
    );
    Ok(())
}

/// Reads the range and the save, and writes the export.
///
/// Nothing is resolved or lifted, and the resolution is not loaded: its differentials are read
/// off the save as sparse terms.
///
/// A class is `h_0^i g` for a tower generator `g` and `i` below the tower's length. Files, per
/// complex `<C>` (`Ctau`, and `C2`, `Ceta`, `Cnu`, `Csigma` for Cτ smashed with the cofiber of
/// `2`, `η`, `ν`, `σ`):
/// * `<C>_towers.csv`: `id,stem,filtration,weight,length` of each generator (`inf` for a tower
///   that never ends);
/// * `<C>_h.csv`: `j,source,targets` for `h_j`, `j = 1, ..., 4`: `h_j` times the generator
///   `source` is the sum of the classes of the towers `targets` in the tridegree of the product.
///   Multiplication by `h_0` is the tower structure;
/// * `<C>_istar.csv`, `<C>_pstar.csv`: the same for the bottom cell `Ctau → C` (sources are
///   towers of `Ctau`) and the top cell `C → Σ Ctau` (targets are towers of `Ctau`);
/// * `Ctau_generators.csv`, `Ctau_products.csv`: the algebra generators up to the product bound
///   (see [`products`]) and `generator,source,targets` for their products.
///
/// Every map commutes with `h_0`, so its value on `h_0^i g` is the sum of the classes of the
/// same targets, those whose tower still holds a class in that tridegree.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let res = query_module_only("Module", Some(AlgebraType::Milnor), false)?;
    let dir: PathBuf = query::raw("Output directory", |s| {
        Ok::<_, anyhow::Error>(PathBuf::from(s))
    });
    let max_n: i32 = query::with_default("Max motivic stem", "60", str::parse);
    let specs: String = query::with_default("Complexes (- for none)", "C2,Ceta,Cnu,Csigma", |s| {
        Ok::<_, anyhow::Error>(s.to_string())
    });
    let product_weight: i32 =
        query::with_default("Product bound (weight of β, 0 for none)", "0", str::parse);
    std::fs::create_dir_all(&dir)?;
    let store = res.save_dir().store().expect("a saved resolution");
    let algebra = res.algebra();

    let begin = Instant::now();
    let ops = OpTable::new(algebra.clone());
    let (max_a, max_t) = ScaledExt::backbone_range(max_n);
    let bb = StreamedBackbone::read(store, &algebra, &ops, max_a, &max_t)?;
    eprintln!(
        "backbone: {} terms: {:.1} s, peak {:.1} GB",
        bb.terms(),
        begin.elapsed().as_secs_f64(),
        peak_gb()
    );
    let q_degree = max_n / 2 + 2;
    let mut q = QAlgebra::new(q_degree, q_degree as u32);
    q.set_memo_limit(1 << 31);
    let start = Instant::now();
    // the shifts are those of h_1, ..., h_4, the cell actions of the Hopf cofibers, and their sums
    let tables = OpTables::new(
        &q,
        &bb,
        &[1, 2, 3, 4, 5, 6, 8, 9, 10, 12],
        q_degree,
        ScaledExt::reach(max_n),
    );
    q.clear_memo();
    eprintln!(
        "operation tables: {} entries: {:.1} s, peak {:.1} GB",
        tables.entries(),
        start.elapsed().as_secs_f64(),
        peak_gb()
    );

    /// Reduces `cx` and writes its towers and the action of `h_1, ..., h_4`.
    fn export(
        dir: &std::path::Path,
        q: &QAlgebra,
        bb: &StreamedBackbone,
        tables: &OpTables,
        cx: Complex,
        max_n: i32,
    ) -> anyhow::Result<ScaledExt> {
        let start = Instant::now();
        let name = cx.name.clone();
        let ext = ScaledExt::new(q, bb, tables, cx, max_n);
        q.clear_memo();
        let s = ext.stats;
        eprintln!(
            "{name}: {} towers; {} cochains (largest space {}), {} coboundary entries, {} reduced \
             entries (longest column {}), {} entries added: {:.1} s, peak {:.1} GB",
            ext.num_towers(),
            s.cochains,
            s.largest,
            s.input_entries,
            s.reduced_entries,
            s.longest,
            s.additions,
            start.elapsed().as_secs_f64(),
            peak_gb()
        );
        let mut w = csv(
            dir,
            &format!("{name}_towers.csv"),
            "id,stem,filtration,weight,length",
        )?;
        for &(a, u) in ext.keys() {
            let red = ext.reduced(a, u).unwrap();
            for i in 0..red.towers.len() {
                let (birth, length) = red.tower(i);
                let length = if length == NEVER {
                    "inf".to_string()
                } else {
                    length.to_string()
                };
                writeln!(
                    w,
                    "{},{},{},{u},{length}",
                    ext.tower_id(a, u, i as u32),
                    2 * u - a,
                    a + birth as i32
                )?;
            }
        }
        let start = Instant::now();
        let rows: Vec<Vec<(u32, Rows)>> = ext
            .keys()
            .to_vec()
            .into_maybe_par_iter()
            .map(|(a, u)| {
                let mut out = Vec::new();
                if ext.reduced(a, u).unwrap().towers.is_empty() {
                    return out;
                }
                let src = ext.space(q, bb, a, u);
                let mut ws = SparseWorkspace::new(0);
                for j in 0..4u32 {
                    let (a2, u2) = (a + 1, u + (1 << j));
                    if ext.reduced(a2, u2).is_none() {
                        continue;
                    }
                    let tgt = ext.space(q, bb, a2, u2);
                    // the whole cochain map, then its value on each generator
                    let red = ext.reduced(a, u).unwrap();
                    let mut wanted = vec![false; src.dim()];
                    for i in 0..red.towers.len() {
                        for &pos in red.generator(i) {
                            wanted[pos as usize] = true;
                        }
                    }
                    let mut columns: Vec<Vec<u32>> = vec![Vec::new(); src.dim()];
                    for (pos, column) in
                        pullback_columns(q, bb, tables, &ext.cx, &src, &tgt, 1 << j, |pos| {
                            !wanted[pos as usize]
                        })
                    {
                        columns[pos as usize] = column;
                    }
                    let images = induced(&ext, &ext, (a, u), (a2, u2), 0, &mut ws, |c, out| {
                        for &pos in c {
                            out.extend_from_slice(&columns[pos as usize]);
                        }
                    })
                    .unwrap();
                    let rows = images
                        .into_iter()
                        .enumerate()
                        .map(|(i, t)| (ext.tower_id(a, u, i as u32), t))
                        .collect();
                    out.push((j + 1, rows));
                }
                out
            })
            .collect();
        let mut w = csv(dir, &format!("{name}_h.csv"), "j,source,targets")?;
        for (j, rows) in rows.into_iter().flatten() {
            write_rows(&mut w, &format!("{j},"), &rows)?;
        }
        q.clear_memo();
        eprintln!(
            "{name} h_j: {:.1} s, peak {:.1} GB",
            start.elapsed().as_secs_f64(),
            peak_gb()
        );
        Ok(ext)
    }

    let ctau = export(&dir, &q, &bb, &tables, Complex::sphere(), max_n)?;

    for name in specs.split(',').filter(|s| !s.is_empty() && *s != "-") {
        let cx = match name {
            "C2" => Complex::moore(),
            "Ceta" => Complex::hopf(name, 0),
            "Cnu" => Complex::hopf(name, 1),
            "Csigma" => Complex::hopf(name, 2),
            _ => anyhow::bail!("unknown complex {name}"),
        };
        ctau.clear_spaces();
        let cof = export(&dir, &q, &bb, &tables, cx, max_n)?;
        let start = Instant::now();
        let top = cof.cx.cells.len() - 1;
        let [da, du, dk] = cof.cx.cells[top];
        // bottom cell: the same cochain on cell 0
        let rows: Vec<Rows> = ctau
            .keys()
            .to_vec()
            .into_maybe_par_iter()
            .map(|(a, u)| {
                if ctau.reduced(a, u).unwrap().towers.is_empty() || cof.reduced(a, u).is_none() {
                    return Vec::new();
                }
                let src = ctau.space(&q, &bb, a, u);
                let tgt = cof.space(&q, &bb, a, u);
                let mut ws = SparseWorkspace::new(0);
                induced(&ctau, &cof, (a, u), (a, u), 0, &mut ws, |c, out| {
                    for &pos in c {
                        let (_, t, i, m, _) = src.decode(&q, pos);
                        out.push(tgt.pos(&q, 0, t, i, &m));
                    }
                })
                .unwrap()
                .into_iter()
                .enumerate()
                .map(|(i, t)| (ctau.tower_id(a, u, i as u32), t))
                .collect()
            })
            .collect();
        let mut w = csv(&dir, &format!("{name}_istar.csv"), "source,targets")?;
        for r in &rows {
            write_rows(&mut w, "", r)?;
        }
        // top cell: the coordinates on the top cell, which are a cocycle of Ctau
        let rows: Vec<Rows> = cof
            .keys()
            .to_vec()
            .into_maybe_par_iter()
            .map(|(a, u)| {
                let (a2, u2) = (a + da, u - du);
                if cof.reduced(a, u).unwrap().towers.is_empty() || ctau.reduced(a2, u2).is_none() {
                    return Vec::new();
                }
                let src = cof.space(&q, &bb, a, u);
                let tgt = ctau.space(&q, &bb, a2, u2);
                let mut ws = SparseWorkspace::new(0);
                induced(&cof, &ctau, (a, u), (a2, u2), -dk, &mut ws, |c, out| {
                    for &pos in c {
                        let (cell, t, i, m, _) = src.decode(&q, pos);
                        if cell == top {
                            out.push(tgt.pos(&q, 0, t, i, &m));
                        }
                    }
                })
                .unwrap()
                .into_iter()
                .enumerate()
                .map(|(i, t)| (cof.tower_id(a, u, i as u32), t))
                .collect()
            })
            .collect();
        let mut w = csv(&dir, &format!("{name}_pstar.csv"), "source,targets")?;
        for r in &rows {
            write_rows(&mut w, "", r)?;
        }
        eprintln!(
            "{name} cells: {:.1} s, peak {:.1} GB",
            start.elapsed().as_secs_f64(),
            peak_gb()
        );
    }
    if product_weight > 0 {
        products(&dir, &res, &ops, &q, &bb, &ctau, product_weight)?;
    }
    eprintln!(
        "total {:.1} s, peak {:.1} GB",
        begin.elapsed().as_secs_f64(),
        peak_gb()
    );
    Ok(())
}
