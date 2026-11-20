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
    ctau::{OpTable, QAlgebra},
    ctau_scale::{Complex, NEVER, ScaledExt, Space, StreamedBackbone, induced, pullback_column},
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
///   towers of `Ctau`) and the top cell `C → Σ Ctau` (targets are towers of `Ctau`).
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
    q.set_memo_limit(1 << 26);

    /// Reduces `cx` and writes its towers and the action of `h_1, ..., h_4`.
    fn export(
        dir: &std::path::Path,
        q: &QAlgebra,
        bb: &StreamedBackbone,
        cx: Complex,
        max_n: i32,
    ) -> anyhow::Result<ScaledExt> {
        let start = Instant::now();
        let name = cx.name.clone();
        let ext = ScaledExt::new(q, bb, cx, max_n);
        q.clear_memo();
        let s = ext.stats;
        eprintln!(
            "{name}: {} towers; {} cochains (largest space {}), {} coboundary entries, {} reduced \
             entries (longest column {}): {:.1} s, peak {:.1} GB",
            ext.num_towers(),
            s.cochains,
            s.largest,
            s.input_entries,
            s.reduced_entries,
            s.longest,
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
                let src = Space::new(bb, q, &ext.cx, a, u);
                let mut ws = SparseWorkspace::new(0);
                for j in 0..4u32 {
                    let (a2, u2) = (a + 1, u + (1 << j));
                    if ext.reduced(a2, u2).is_none() {
                        continue;
                    }
                    let tgt = Space::new(bb, q, &ext.cx, a2, u2);
                    let images = induced(&ext, &ext, (a, u), (a2, u2), 0, &mut ws, |pos, out| {
                        pullback_column(q, bb, &ext.cx, &src, &tgt, pos, 1 << j, out)
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

    let ctau = export(&dir, &q, &bb, Complex::sphere(), max_n)?;

    for name in specs.split(',').filter(|s| !s.is_empty() && *s != "-") {
        let cx = match name {
            "C2" => Complex::moore(),
            "Ceta" => Complex::hopf(name, 0),
            "Cnu" => Complex::hopf(name, 1),
            "Csigma" => Complex::hopf(name, 2),
            _ => anyhow::bail!("unknown complex {name}"),
        };
        let cof = export(&dir, &q, &bb, cx, max_n)?;
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
                let src = Space::new(&bb, &q, &ctau.cx, a, u);
                let tgt = Space::new(&bb, &q, &cof.cx, a, u);
                let mut ws = SparseWorkspace::new(0);
                induced(&ctau, &cof, (a, u), (a, u), 0, &mut ws, |pos, out| {
                    let (_, t, i, m, _) = src.decode(&q, pos);
                    out.push(tgt.pos(&q, 0, t, i, &m));
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
                let src = Space::new(&bb, &q, &cof.cx, a, u);
                let tgt = Space::new(&bb, &q, &ctau.cx, a2, u2);
                let mut ws = SparseWorkspace::new(0);
                induced(&cof, &ctau, (a, u), (a2, u2), -dk, &mut ws, |pos, out| {
                    let (cell, t, i, m, _) = src.decode(&q, pos);
                    if cell == top {
                        out.push(tgt.pos(&q, 0, t, i, &m));
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
    eprintln!(
        "total {:.1} s, peak {:.1} GB",
        begin.elapsed().as_secs_f64(),
        peak_gb()
    );
    Ok(())
}
