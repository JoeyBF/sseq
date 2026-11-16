//! Thread scaling of the Milnor primitives behind the d2 composite pass.

use std::time::Instant;

use algebra::{Algebra, MilnorAlgebra, pair_algebra::PairAlgebra};
use fp::{prime::TWO, vector::FpVector};
use maybe_rayon::prelude::*;

/// Usage: `sigma_scaling <idx|mul|sigma> <degree>`; the thread count comes from
/// `RAYON_NUM_THREADS`.
///
/// Runs the chosen primitive on every pair of basis elements of total degree `degree`, in chunks
/// that each own their result buffer, and prints the wall time.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args[1].as_str();
    let degree: i32 = args[2].parse().unwrap();

    let algebra = MilnorAlgebra::new(TWO, false);
    algebra.compute_basis(degree + 1);

    let pairs: Vec<(i32, usize, i32, usize)> = (1..degree)
        .flat_map(|a| {
            let (da, db) = (algebra.dimension(a), algebra.dimension(degree - a));
            (0..da).flat_map(move |i| (0..db).map(move |j| (a, i, degree - a, j)))
        })
        .collect();
    const CHUNK: usize = 256;
    let chunks = pairs.len().div_ceil(CHUNK);

    let start = Instant::now();
    let sink: usize = (0..chunks)
        .into_maybe_par_iter()
        .map(|c| {
            let chunk = &pairs[c * CHUNK..pairs.len().min((c + 1) * CHUNK)];
            match mode {
                "idx" => chunk
                    .iter()
                    .map(|&(a, i, _, _)| {
                        let elt = algebra.basis_element_from_index(a, i);
                        algebra.basis_element_to_index(&elt)
                    })
                    .sum(),
                "mul" => {
                    let mut result = FpVector::new(TWO, algebra.dimension(degree));
                    for &(a, i, b, j) in chunk {
                        algebra.multiply_basis_elements(result.as_slice_mut(), 1, a, i, b, j);
                    }
                    result.iter_nonzero().count()
                }
                "sigma" => {
                    let mut result = algebra.new_pair_element(degree);
                    for &(a, i, b, j) in chunk {
                        algebra.sigma_multiply_basis(&mut result, 1, a, i, b, j);
                    }
                    usize::from(!MilnorAlgebra::element_is_zero(&result))
                }
                _ => panic!("mode is idx, mul or sigma"),
            }
        })
        .sum();
    println!(
        "SCALING mode={mode} degree={degree} pairs={} threads={} secs={:.3} sink={sink}",
        pairs.len(),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_default(),
        start.elapsed().as_secs_f64()
    );
}
