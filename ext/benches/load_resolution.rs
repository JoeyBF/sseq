use std::path::Path;

use algebra::Algebra;
use ext::{chain_complex::ChainComplex, utils::construct};
use sseq::coordinates::Bidegree;

fn compute_or_load_resolution(tempdir: &Path) -> std::time::Duration {
    let resolution = construct("S_2@milnor", Some(tempdir.join("S_2_milnor"))).unwrap();
    resolution.algebra().compute_basis(100);
    let start = std::time::Instant::now();
    resolution.compute_through_bidegree(Bidegree::s_t(50, 100));
    start.elapsed()
}

fn main() {
    let tempdir = tempfile::tempdir().unwrap();
    let to_compute = compute_or_load_resolution(tempdir.path());
    println!("Time to compute to (s, t) = (50, 100): {:?}", to_compute);

    let to_load = compute_or_load_resolution(tempdir.path());
    println!("Time to reload: {:?}", to_load);
}
