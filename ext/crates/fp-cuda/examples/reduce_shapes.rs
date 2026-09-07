//! Device vs CPU reduction at the shapes the resolution ACTUALLY produces, which are wide, not
//! square.
//!
//! `reduce_timing` measures half-rank squares, and that is what calibrated
//! `DEFAULT_RR_THRESHOLD = 8192` (GPU 0.57× at n=4096, 1.57× at n=8192, so "the crossover sits just
//! below 8192"). But the GPU row reduce is gated on `min(rows, cols)`, and in the 0→300 census the
//! wall-weighted aspect ratio `cols / rows-per-block` is 86× at the median and 3623× at p90. The
//! twenty most expensive bidegrees reduce matrices like 3055 × 1_770_153 — 645 MB, and ~258
//! Gword-ops of elimination against the 8.6 Gword-ops of the 8192² square the GPU won.
//!
//! So the gate rejects reductions carrying far more work than the case it was tuned to accept, on
//! the strength of a short side. Whether that rejection is wrong depends on something no bench here
//! measures: how the device does when there are few pivot rows and enormous row length. That is a
//! different parallelism structure from square, and it is the point of this example.
//!
//! The CPU baseline is `row_reduce_blas3`, which is CONSERVATIVE on purpose: production's actual
//! fallback above the threshold is single-threaded M4RI, which is slower still. A device win over
//! blas3 is therefore a lower bound on the device win over what production really does.
//!
//! ```sh
//! cargo run --release -p fp-cuda --example reduce_shapes            # census shapes + controls
//! cargo run --release -p fp-cuda --example reduce_shapes -- 2877x1622037
//! ```

use std::time::Instant;

use fp::{matrix::Matrix, prime::TWO};
use fp_cuda::GpuContext;
use rand::Rng;

mod common;
use common::upload_matrix;

/// Random `rows × cols` built straight into limbs. Going through `Vec<Vec<u32>>` costs one u32 per
/// BIT — 10.8 GB for a 1527 × 1_770_153 operand — where the packed form is 338 MB.
fn random_matrix(rows: usize, cols: usize) -> Matrix {
    let stride = cols.div_ceil(64);
    let mut rng = rand::rng();
    let mut limbs = vec![0u64; rows * stride];
    for l in limbs.iter_mut() {
        *l = rng.random();
    }
    // Bits past `cols` in the final limb of each row must be zero or the matrix is malformed.
    let tail = cols % 64;
    if tail != 0 {
        let mask = (1u64 << tail) - 1;
        for r in 0..rows {
            limbs[r * stride + stride - 1] &= mask;
        }
    }
    Matrix::from_data(TWO, rows, cols, limbs)
}

/// Half-rank `rows × cols`, matching the construction the square calibration used.
fn half_rank(rows: usize, cols: usize) -> Matrix {
    let rank = (rows / 2).max(1);
    &random_matrix(rows, rank) * &random_matrix(rank, cols)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let shapes: Vec<(usize, usize, &'static str)> = if args.is_empty() {
        vec![
            // Controls: reproduce the calibration points on this machine.
            (4096, 4096, "square control (recorded 0.57x = LOSS)"),
            (8192, 8192, "square control (recorded 1.57x = WIN)"),
            // Real census shapes, cheapest first.
            (587, 1_524_934, "b=(287,5)  0.60 core-h  aspect 2599x"),
            (657, 1_692_321, "b=(293,5)  0.71 core-h  aspect 2576x"),
            (1131, 611_461, "median REJECTED reduce  12.2 Gword-ops"),
            (1676, 1_686_395, "b=(266,7)  0.63 core-h  aspect 1006x"),
            (2877, 1_622_037, "b=(253,10) 0.66 core-h  556 MB"),
            (3055, 1_770_153, "b=(262,8)  0.69 core-h  645 MB, 258 Gword-ops"),
        ]
    } else {
        args.iter()
            .filter_map(|a| {
                let (r, c) = a.split_once('x')?;
                Some((r.parse().ok()?, c.parse().ok()?, "user"))
            })
            .collect()
    };

    let gpu = GpuContext::new(0)?;
    println!("=== device vs CPU blas3 reduction, half-rank, REAL shapes ===");
    println!(
        "  gate today is min(rows,cols) >= 8192; CPU baseline is blas3, so a device win here is a\n  \
         LOWER bound on the win over production's single-threaded M4RI fallback.\n"
    );
    println!(
        "  {:>6} {:>10} {:>8} {:>8} {:>10} {:>10} {:>9} {:>7}  {}",
        "rows", "cols", "aspect", "MB", "device", "cpu-blas3", "speedup", "gated?", "shape"
    );

    for (rows, cols, label) in shapes {
        let mm = half_rank(rows, cols);
        let mb = rows as f64 * cols as f64 / 8.0 / (1 << 20) as f64;
        let aspect = cols as f64 / rows as f64;
        let gated = if rows.min(cols) >= 8192 { "GPU" } else { "CPU" };

        // Device: upload + full reduce + sync, excluding download (matches reduce_timing).
        let t0 = Instant::now();
        let mut dm = upload_matrix(&gpu, &mm)?;
        let (_perm, r, _piv) = gpu.row_reduce_dev(&mut dm)?;
        let dev = t0.elapsed().as_secs_f64();

        let mut cpu = mm.clone();
        let t1 = Instant::now();
        let cpu_rank = cpu.row_reduce_blas3();
        let cpu_s = t1.elapsed().as_secs_f64();

        // Rank agreement is the cheap invariant. Materialising the full device RREF for a 645 MB
        // matrix would cost more than the measurement; `reduce_timing` does the full compare on
        // squares, and a rank mismatch is what a broken reduce actually produces.
        let ok = r == cpu_rank;

        println!(
            "  {rows:>6} {cols:>10} {aspect:>7.0}x {mb:>8.1} {dev:>9.3}s {cpu_s:>9.3}s \
             {:>8.2}x {gated:>7}  {label}{}",
            cpu_s / dev,
            if ok { "" } else { "   *** RANK MISMATCH ***" }
        );
        if !ok {
            eprintln!("rank mismatch: device {r} vs cpu {cpu_rank} at {rows}x{cols}");
            std::process::exit(1);
        }
    }
    println!(
        "\n  Read the `gated?` column against `speedup`: any row marked CPU with a speedup above\n  \
         1.00x is work the current threshold sends to the slower path."
    );
    Ok(())
}
