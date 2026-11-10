//! Writes the shape of a cocycle for every class of each stage $\mathrm{Ext}(Q^{(n)})$.
//!
//! Usage: `cycle_shapes <max_n> <max_s> <min_stage> <max_stage> <out_dir>`, with
//! `CTAU_BACKBONE=<dir>` to load or save the resolution of $\mathbb{F}_2$. Writes
//! `<out_dir>/shapes_<n>.csv` with the fields of [`ext::vn_bockstein::CycleShape`] in order.
use std::{io::Write, sync::Arc, time::Instant};

use ext::{
    algebraic_novikov::{unit_module, untwisted_backbone_range},
    vn_bockstein::{Layer, cycle_shapes},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_n: i32 = args[1].parse().unwrap();
    let max_s: i32 = args[2].parse().unwrap();
    let min_stage: usize = args[3].parse().unwrap();
    let max_stage: usize = args[4].parse().unwrap();
    let out = std::path::PathBuf::from(&args[5]);
    std::fs::create_dir_all(&out).unwrap();

    let save = std::env::var_os("CTAU_BACKBONE").map(std::path::PathBuf::from);
    let res = ext::nassau::Resolution::new_with_save(Arc::new(unit_module()), save).unwrap();
    res.compute_through_stem(untwisted_backbone_range(max_n, max_s));

    for n in min_stage..=max_stage {
        let start = Instant::now();
        let shapes = cycle_shapes(&res, Layer::Adjoin(n), max_n, max_s);
        let path = out.join(format!("shapes_{n}.csv"));
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for c in &shapes {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{}",
                c.stem,
                c.s,
                c.u,
                c.filtration,
                c.depth,
                c.support,
                c.generators,
                c.dim,
                c.dim_generators
            )
            .unwrap();
        }
        println!(
            "stage {n}: {} classes ({:?})",
            shapes.len(),
            start.elapsed()
        );
    }
}
