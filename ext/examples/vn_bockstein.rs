//! Runs the $v_n$-Bockstein spectral sequences $Q^{(0)} \to Q^{(1)} \to \dots$ and writes each page.
//!
//! Usage: `vn_bockstein <max_n> <max_s> <max_stage> <out_dir>`, with `CTAU_BACKBONE=<dir>` to
//! load or save the resolution of $\mathbb{F}_2$ and `MIN_STAGE=<n>` to skip the stages below
//! `n`. For each stage `n` it prints the size of the complex, of $E_1$ and of $E_\infty$ and the
//! number of $d_r$ for each `r`, and writes `<out_dir>/e1_<n>.csv`, `einf_<n>.csv` and
//! `d_<n>.csv` as `stem,s,u,dim` (with `r` in front
//! for the differentials, keyed by their source).
use std::{collections::BTreeMap, io::Write, sync::Arc, time::Instant};

use ext::{
    algebraic_novikov::{unit_module, untwisted_backbone_range},
    vn_bockstein::{page_dims, vn_bockstein},
};

/// Writes `rows` as CSV lines to `path`.
fn write_csv(path: &std::path::Path, rows: impl IntoIterator<Item = String>) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for row in rows {
        writeln!(f, "{row}").unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: i32 = args[1].parse().unwrap();
    let max_s: i32 = args[2].parse().unwrap();
    let max_stage: usize = args[3].parse().unwrap();
    let out = std::path::PathBuf::from(&args[4]);
    std::fs::create_dir_all(&out).unwrap();

    let save = std::env::var_os("CTAU_BACKBONE").map(std::path::PathBuf::from);
    let res = ext::nassau::Resolution::new_with_save(Arc::new(unit_module()), save).unwrap();
    res.compute_through_stem(untwisted_backbone_range(max_n, max_s));

    let min_stage: usize = std::env::var("MIN_STAGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for n in min_stage..=max_stage {
        let start = Instant::now();
        let pages = vn_bockstein(&res, n, max_n, max_s);
        let e1 = page_dims(&pages, 1, max_n, max_s);
        let einf = page_dims(&pages, i32::MAX, max_n, max_s);
        let mut by_r: BTreeMap<i32, usize> = BTreeMap::new();
        let mut d_rows = Vec::new();
        let mut cochains = 0;
        for &(k, u, a) in pages.chains.keys() {
            let stem = 2 * u - a;
            if !(0..=max_n).contains(&stem) || a + k > max_s {
                continue;
            }
            cochains += pages.chains[&(k, u, a)];
            for (r, c) in pages.differentials((k, u, a)) {
                if r > 0 {
                    *by_r.entry(r).or_insert(0) += c;
                    d_rows.push(format!("{r},{stem},{},{u},{c}", a + k));
                }
            }
        }
        let total = |m: &BTreeMap<(i32, i32, i32), usize>| m.values().sum::<usize>();
        println!(
            "stage {n}: cochains {cochains}, E1 {}, E∞ {}, d_r {:?} ({:?})",
            total(&e1),
            total(&einf),
            by_r,
            start.elapsed()
        );
        let csv = |m: &BTreeMap<(i32, i32, i32), usize>| {
            m.iter()
                .map(|(&(n, s, u), d)| format!("{n},{s},{u},{d}"))
                .collect::<Vec<_>>()
        };
        write_csv(&out.join(format!("e1_{n}.csv")), csv(&e1));
        write_csv(&out.join(format!("einf_{n}.csv")), csv(&einf));
        write_csv(&out.join(format!("d_{n}.csv")), d_rows);
    }
}
