//! GPU counterpart to `nassau_milnor.rs`: benchmarks the **batched Milnor multiply on the GPU**
//! ([`algebra::milnor_gpu::multiply_batch_on_gpu`]) — the kernel + resident-master + cubecl-allocator
//! path Nassau's `S_2` resolution drives, WITHOUT the surrounding row-reduction or resolution
//! bookkeeping. This is the "hammer the GPU with multiplications" harness.
//!
//! Why a bench (not just an example):
//!  - **Perf + regression tracking.** `cargo bench --bench nassau_milnor_gpu -- --save-baseline pre`
//!    then `--baseline pre` across cubecl commits measures the multiply-throughput delta directly —
//!    the "how much does the cubecl backend cost us" number, and a guard against kernel regressions.
//!  - **Isolation.** A crash here indicts the multiply / cubecl allocator alone (not RREF, which runs
//!    on the separate `fp-cuda` runtime, nor Nassau bookkeeping).
//!
//! The output-degree sweep doubles as the memory axis: larger `out_degree` → larger resident master
//! (the shared segmented device buffer warms once and is reused across iterations, exactly as in a
//! real resolution). Requires `--features gpu` and `CUDA_PATH` (see `ext/gpu_prep`); without `gpu`
//! it compiles to a no-op `main`.
//!
//! Scope note: this is the fixed-scale THROUGHPUT bench. The ~100-stream concurrency + unbounded
//! resident-master GROWTH soak that reproduces the cubecl uninit-handle crash
//! (tracel-ai/cubecl#1401) belongs in a `#[test]`, not here — a criterion measurement loop is the
//! wrong shape for a memory-growth soak with correctness assertions.

#[cfg(feature = "gpu")]
mod gpu {
    use algebra::{
        Algebra, MilnorAlgebra,
        milnor_gpu::{GpuProduct, multiply_batch_on_gpu},
    };
    use criterion::{Criterion, Throughput, black_box, criterion_group};
    use fp::prime::TWO;

    /// Output degrees to sweep — the cost/memory axis. Chosen around Nassau's hot band
    /// (out ≈ 40–52; see `nassau_milnor.rs`'s `REGIME`), plus a cheap and an expensive anchor.
    const OUT_DEGREES: &[i32] = &[24, 32, 40, 48];
    /// Output rows per batch. Products round-robin across rows so each launch fills a real matrix
    /// rather than a single-row strip.
    const NUM_ROWS: usize = 32;

    /// Build one batched `get_partial_matrix`-shaped build at `out_degree`: every non-empty `R` of
    /// degree `1..out_degree` times a dense complementary element, round-robin across `NUM_ROWS`
    /// rows, single generator block (`out_offset = 0`, `num_cols = dim(out_degree)`). Mirrors the
    /// construction in `multiply_batch_matches_reference`, so it hits the same kernel path Nassau does.
    fn build_batch(algebra: &MilnorAlgebra, out_degree: i32) -> (usize, Vec<GpuProduct>) {
        let num_cols = algebra.dimension(out_degree);
        let mut products = Vec::new();
        for r_degree in 1..out_degree {
            let s_degree = out_degree - r_degree;
            let s_dim = algebra.dimension(s_degree);
            if s_dim == 0 {
                continue;
            }
            let r_dim = algebra.dimension(r_degree);
            for r_idx in 0..r_dim {
                if algebra
                    .basis_element_from_index(r_degree, r_idx)
                    .p_part
                    .is_empty()
                {
                    continue;
                }
                let row = products.len() % NUM_ROWS;
                products.push(GpuProduct {
                    r_degree,
                    r_idx,
                    s_degree,
                    term_indices: (0..s_dim).collect(),
                    row,
                    out_offset: 0,
                });
            }
        }
        (num_cols, products)
    }

    pub fn nassau_milnor_gpu(c: &mut Criterion) {
        // Exactly the algebra Nassau uses: the full Milnor algebra at p=2, stable (not unstable).
        use std::sync::Arc;
        let algebra = Arc::new(MilnorAlgebra::new(TWO, false));
        let mut g = c.benchmark_group("nassau_milnor_gpu");

        for &out_degree in OUT_DEGREES {
            // `compute_basis` is cumulative; seqno tables are what the GPU path indexes by.
            algebra.compute_basis(out_degree);
            algebra.compute_seqno_tables(out_degree);
            let (num_cols, products) = build_batch(&algebra, out_degree);
            if products.is_empty() || num_cols == 0 {
                continue;
            }

            // One "element" = one `Sq(R)·s` product fused into the launch.
            g.throughput(Throughput::Elements(products.len() as u64));
            g.bench_function(format!("multiply_batch/out{out_degree}"), |b| {
                b.iter(|| {
                    black_box(multiply_batch_on_gpu(
                        &algebra,
                        num_cols,
                        NUM_ROWS,
                        black_box(&products),
                    ));
                });
            });
        }

        g.finish();
    }

    criterion_group! {
        name = benches;
        config = Criterion::default()
            .measurement_time(std::time::Duration::from_secs(5))
            .sample_size(30);
        targets = nassau_milnor_gpu
    }
}

#[cfg(feature = "gpu")]
criterion::criterion_main!(gpu::benches);

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("nassau_milnor_gpu bench requires --features gpu (and CUDA_PATH; see ext/gpu_prep)");
}
