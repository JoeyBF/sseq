//! Smoke test for the `fp-cuda` matmul kernel.
//!
//! Multiplies one pair of small F_2 matrices on the GPU and verifies the result against
//! `fp::blas`. Run with `cargo oxide run -p fp-cuda --example matmul_b1_demo`.

use fp::{matrix::Matrix, prime::TWO};
use fp_cuda::{GpuContext, matmul_b1};
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = GpuContext::new(0)?;
    let (major, minor) = gpu.compute_capability()?;
    println!("=== fp-cuda matmul_b1 demo ===");
    println!("GPU compute capability: sm_{major}{minor}");

    let mut rng = rand::rng();
    let mut make = |rows: usize, cols: usize| {
        let data: Vec<u64> = (0..rows * cols.div_ceil(64))
            .map(|_| rng.random())
            .collect();
        Matrix::from_data(TWO, rows, cols, data)
    };

    for &(m, k, n) in &[
        (64, 64, 64),
        (128, 128, 128),
        (256, 192, 320),
        (512, 512, 512),
    ] {
        let a = make(m, k);
        let b = make(k, n);
        let cpu = &a * &b;
        let gpu_out = matmul_b1(&gpu, &a, &b)?;
        let ok = cpu == gpu_out;
        println!(
            "  {m}x{k} * {k}x{n}: {}",
            if ok { "OK" } else { "MISMATCH" }
        );
        assert!(
            ok,
            "GPU result disagrees with CPU for shape {m}x{k}*{k}x{n}"
        );
    }

    println!("All shapes matched. ✓");
    Ok(())
}
