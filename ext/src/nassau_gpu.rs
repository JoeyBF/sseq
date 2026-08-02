//! GPU-accelerated `get_partial_matrix` for Nassau's Milnor differentials.
//!
//! A Nassau differential is a `FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>`;
//! building its (partial) matrix applies the differential to each input basis element,
//! whose dominant cost is the Milnor multiply `Sq(R) · s` in [`FreeModule::act`]. This
//! module batches every such multiply of one matrix build into a single GPU launch via
//! [`algebra::milnor_gpu::multiply_batch_on_gpu`]. See `benches/milnor_gpu_ab.rs` for a
//! CPU/GPU A/B of that launch.
//!
//! Only the *multiply* work is offloaded. Identity operations (`Sq(∅) = 1`, i.e.
//! `operation_degree == 0`) are plain copies with no admissible-matrix work, so they
//! are left to the CPU `apply_to_basis_element` per row. The output F₂ bits the kernel
//! returns are XORed into the matrix rows (bit `i` → `add_basis_element(i, 1)`), the
//! same layout the CPU path produces.
//!
//! Gated behind the `gpu` feature. Callers must ensure
//! [`MilnorAlgebra::gpu_multiply_applicable`] (`p = 2`, trivial profile, stable) — the
//! Nassau `S_2` regime — and that the algebra's basis and seqno tables reach `degree`.

use algebra::{
    MilnorAlgebra,
    milnor_gpu::{GpuProduct, multiply_batch_on_gpu},
    module::{
        FreeModule, Module,
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
    },
};
use fp::matrix::Matrix;

type NassauDifferential = FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>;

/// Whether the GPU `get_partial_matrix` path applies to this differential — the
/// seqno-table regime (`p = 2`, trivial profile, stable), i.e. Nassau `S_2`. The
/// (cheap, idempotent) seqno tables are built on demand in [`get_partial_matrix`].
pub fn applicable(hom: &NassauDifferential) -> bool {
    hom.target().algebra().gpu_multiply_applicable()
}

/// GPU analogue of [`ModuleHomomorphism::get_partial_matrix`] for a Nassau differential.
///
/// Produces the same matrix as the CPU path: rows are the differential applied to
/// `inputs[i]`, columns the target basis at `degree`. The multiply work of every input
/// is fused into one GPU launch; identity operations are filled in on the CPU.
///
/// Callers must ensure [`applicable`]; this mirrors the extraction the CPU
/// [`FreeModuleHomomorphism::apply_to_basis_element`] performs.
pub fn get_partial_matrix(hom: &NassauDifferential, degree: i32, inputs: &[usize]) -> Matrix {
    let (mut matrix, products) = extract(hom, degree, inputs);
    if !products.is_empty() {
        let target = hom.target();
        let algebra = target.algebra();
        // Idempotent + cheap (O(degree · width)); returns immediately once built.
        algebra.compute_seqno_tables(degree);
        let num_cols = target.dimension(degree);
        let out = multiply_batch_on_gpu(&algebra, num_cols, inputs.len(), &products);
        for (row, limbs) in out.iter_rows().enumerate() {
            let mut target_row = matrix.row_mut(row);
            for (limb_idx, &limb) in limbs.iter().enumerate() {
                let mut bits = limb;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    target_row.add_basis_element(limb_idx * 32 + b, 1);
                    bits &= bits - 1;
                }
            }
        }
    }
    matrix
}

/// Extract the non-identity Milnor multiplies of one matrix build as [`GpuProduct`]s,
/// mirroring `apply_to_basis_element` → [`FreeModule::act`]. Returns the matrix with its
/// identity (`Sq(∅)`) and out-of-range rows already filled, plus the products whose GPU
/// (or CPU-reference) output completes it.
fn extract(hom: &NassauDifferential, degree: i32, inputs: &[usize]) -> (Matrix, Vec<GpuProduct>) {
    let p = hom.prime();
    let source = hom.source();
    let target = hom.target();
    let shift = hom.degree_shift();
    let out_dim = target.dimension(degree);
    let mut matrix = Matrix::new(p, inputs.len(), out_dim);
    let mut products: Vec<GpuProduct> = Vec::new();

    if out_dim == 0 {
        return (matrix, products);
    }

    for (row, &input_index) in inputs.iter().enumerate() {
        let ogp = source.index_to_op_gen(degree, input_index);
        if ogp.generator_degree < hom.min_degree() {
            continue;
        }
        if ogp.operation_degree == 0 {
            // Sq(∅) · s = s: let the CPU copy it directly into this row.
            hom.apply_to_basis_element(matrix.row_mut(row), 1, degree, input_index);
            continue;
        }
        let out_on_gen = hom.output(ogp.generator_degree, ogp.generator_index);
        let act_input_degree = ogp.generator_degree - shift;
        for gd in target
            .iter_gen_offsets::<2>([act_input_degree, act_input_degree + ogp.operation_degree])
        {
            let (input_start, input_end) = (gd.start[0], gd.end[0]);
            if input_start >= out_on_gen.len() {
                break;
            }
            let s_slice = out_on_gen.as_slice().restrict(input_start, input_end);
            if s_slice.is_zero() {
                continue;
            }
            products.push(GpuProduct {
                r_degree: ogp.operation_degree,
                r_idx: ogp.operation_index,
                s_degree: act_input_degree - gd.gen_deg,
                term_indices: s_slice.iter_nonzero().map(|(i, _)| i).collect(),
                row,
                // `Sq(R) · s` lands in this target generator's block of the output row,
                // which starts at gd.start[1] (the offsets at the *output* degree).
                out_offset: gd.start[1],
            });
        }
    }
    (matrix, products)
}

/// Build the matrix both ways and assert they agree; returns the CPU matrix. For the
/// `NASSAU_GPU_VERIFY` env gate — validates the GPU path over a real resolution before
/// it is trusted unconditionally.
pub fn get_partial_matrix_verified(
    hom: &NassauDifferential,
    degree: i32,
    inputs: &[usize],
) -> Matrix {
    let gpu = get_partial_matrix(hom, degree, inputs);
    let cpu = hom.get_partial_matrix(degree, inputs);
    for row in 0..inputs.len() {
        let g: Vec<usize> = gpu.row(row).iter_nonzero().map(|(i, _)| i).collect();
        let c: Vec<usize> = cpu.row(row).iter_nonzero().map(|(i, _)| i).collect();
        assert_eq!(
            g, c,
            "GPU/CPU get_partial_matrix mismatch at degree {degree}, row {row} (input {})",
            inputs[row],
        );
    }
    cpu
}

/// Restricted GPU analogue of `Resolution::restricted_partial_matrix` (PR #272): builds the
/// differential matrix on `inputs` but with the output truncated to the first `target_dim` basis
/// elements of the target — the frozen (degree-restricted) prefix that Nassau's relaxed dependency
/// graph reads. Valid by minimality: a generator's differential lands in the radical, so nothing is
/// dropped that would be nonzero.
///
/// Differs from [`get_partial_matrix`] only in the output width: the matrix has `target_dim`
/// columns, identity (`Sq(∅)`) rows are filled via
/// [`FreeModuleHomomorphism::apply_to_basis_element_restricted`], products whose output block starts
/// at/after `target_dim` are dropped (blocks are generator-major and contiguous, and `target_dim`
/// falls on a generator boundary, so the whole block is outside), and the kernel is launched with
/// `num_cols = target_dim`. Any returned bit `>= target_dim` is masked out defensively.
/// Rows per GPU multiply batch, chosen so the dense readback (`rows × ceil(cols/32) × 4` bytes)
/// stays under `NASSAU_GPU_MAX_READBACK_MB` (default 1024). Bounds the transient host memory of one
/// build regardless of how many rows the bidegree has; ≥ 1. `0` MB disables batching (one call).
fn gpu_rows_per_batch(cols: usize, num_rows: usize) -> usize {
    static CAP_BYTES: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("NASSAU_GPU_MAX_READBACK_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024)
            * (1 << 20)
    });
    if *CAP_BYTES == 0 {
        return num_rows.max(1);
    }
    let bytes_per_row = cols.div_ceil(32) * 4; // one row's readback in bytes
    (*CAP_BYTES / bytes_per_row.max(1)).clamp(1, num_rows.max(1))
}

pub fn get_partial_matrix_restricted(
    hom: &NassauDifferential,
    degree: i32,
    inputs: &[usize],
    target_dim: usize,
) -> Matrix {
    // Spanned because the 279 s stalls land somewhere in this function *before* the multiply
    // (which itself measured 40 ms), and neither this build nor the pair pre-pass inside
    // `multiply_batch_on_gpu` was previously visible to the log.
    let (mut matrix, mut products) =
        tracing::info_span!("extract_restricted", inputs = inputs.len(), target_dim)
            .in_scope(|| extract_restricted(hom, degree, inputs, target_dim));
    if !products.is_empty() {
        let target = hom.target();
        let algebra = target.algebra();
        // Idempotent + cheap (O(degree · width)); returns immediately once built.
        algebra.compute_seqno_tables(degree);
        // The multiply must run at the *full* output width: a product's `out_offset + seqno`
        // indexes the full output-degree basis (a kept block can start below `target_dim` yet its
        // seqno spans past it), and `num_cols` sets the kernel's per-row limb count / bit layout.
        // Passing `target_dim` there would truncate `num_limbs` and corrupt the row layout. We
        // truncate afterwards by masking bits `>= target_dim` when XORing into the matrix.
        let full_cols = target.dimension(degree);
        // Cap how large a single multiply we hand the GPU: the dense readback (num_rows × num_limbs
        // u32) plus the matrix would otherwise both be held for the whole all-rows / zero-signature
        // build (~12 GB dense regions at stem 180). Process the rows in batches of ≤ `rows_per_batch`
        // so the readback stays bounded and is freed between batches. `products` is built in row
        // order (`extract_restricted`), so each batch's products are a contiguous slice; we remap
        // their `row` to batch-local (0-based) for the kernel and write back to the global rows.
        let rows_per_batch = gpu_rows_per_batch(full_cols, inputs.len());
        let mut p0 = 0usize;
        let mut r0 = 0usize;
        while r0 < inputs.len() {
            let r1 = (r0 + rows_per_batch).min(inputs.len());
            let mut p1 = p0;
            while p1 < products.len() && products[p1].row < r1 {
                p1 += 1;
            }
            if p1 > p0 {
                for pr in &mut products[p0..p1] {
                    pr.row -= r0; // batch-local row index for the kernel's output layout
                }
                let out = multiply_batch_on_gpu(&algebra, full_cols, r1 - r0, &products[p0..p1]);
                for (bi, limbs) in out.iter_rows().enumerate() {
                    let mut target_row = matrix.row_mut(r0 + bi);
                    for (limb_idx, &limb) in limbs.iter().enumerate() {
                        let mut bits = limb;
                        while bits != 0 {
                            let b = bits.trailing_zeros() as usize;
                            let col = limb_idx * 32 + b;
                            // Minimality should keep every bit within the restricted prefix, but mask
                            // defensively so a stray high bit can never write out of bounds.
                            if col < target_dim {
                                target_row.add_basis_element(col, 1);
                            }
                            bits &= bits - 1;
                        }
                    }
                }
            }
            p0 = p1;
            r0 = r1;
        }
    }
    matrix
}

/// Restricted analogue of [`extract`]: the matrix has `target_dim` columns, identity rows are
/// filled via the restricted apply, and products landing outside the restricted prefix are dropped.
fn extract_restricted(
    hom: &NassauDifferential,
    degree: i32,
    inputs: &[usize],
    target_dim: usize,
) -> (Matrix, Vec<GpuProduct>) {
    let p = hom.prime();
    let source = hom.source();
    let target = hom.target();
    let shift = hom.degree_shift();
    let mut matrix = Matrix::new(p, inputs.len(), target_dim);
    let mut products: Vec<GpuProduct> = Vec::new();

    if target_dim == 0 {
        return (matrix, products);
    }

    for (row, &input_index) in inputs.iter().enumerate() {
        let ogp = source.index_to_op_gen(degree, input_index);
        if ogp.generator_degree < hom.min_degree() {
            continue;
        }
        if ogp.operation_degree == 0 {
            // Sq(∅) · s = s: let the CPU copy it directly into this (restricted-width) row.
            hom.apply_to_basis_element_restricted(matrix.row_mut(row), 1, degree, input_index);
            continue;
        }
        let out_on_gen = hom.output(ogp.generator_degree, ogp.generator_index);
        let act_input_degree = ogp.generator_degree - shift;
        for gd in target
            .iter_gen_offsets::<2>([act_input_degree, act_input_degree + ogp.operation_degree])
        {
            let (input_start, input_end) = (gd.start[0], gd.end[0]);
            if input_start >= out_on_gen.len() {
                break;
            }
            // The product lands in the output block starting at `gd.start[1]`. If that is at or
            // beyond the restricted prefix, the whole (contiguous, generator-major) block is
            // outside `target_dim` and is dropped.
            if gd.start[1] >= target_dim {
                continue;
            }
            let s_slice = out_on_gen.as_slice().restrict(input_start, input_end);
            if s_slice.is_zero() {
                continue;
            }
            products.push(GpuProduct {
                r_degree: ogp.operation_degree,
                r_idx: ogp.operation_index,
                s_degree: act_input_degree - gd.gen_deg,
                term_indices: s_slice.iter_nonzero().map(|(i, _)| i).collect(),
                row,
                out_offset: gd.start[1],
            });
        }
    }
    (matrix, products)
}

/// Build the restricted matrix both ways and assert they agree; returns the CPU matrix. For the
/// `NASSAU_GPU_VERIFY` gate. The CPU reference mirrors `Resolution::restricted_partial_matrix`
/// (each row is the restricted apply of the corresponding input).
pub fn get_partial_matrix_restricted_verified(
    hom: &NassauDifferential,
    degree: i32,
    inputs: &[usize],
    target_dim: usize,
) -> Matrix {
    let gpu = get_partial_matrix_restricted(hom, degree, inputs, target_dim);
    let mut cpu = Matrix::new(hom.prime(), inputs.len(), target_dim);
    if target_dim > 0 {
        for (i, &input) in inputs.iter().enumerate() {
            hom.apply_to_basis_element_restricted(cpu.row_mut(i), 1, degree, input);
        }
    }
    for row in 0..inputs.len() {
        let g: Vec<usize> = gpu.row(row).iter_nonzero().map(|(i, _)| i).collect();
        let c: Vec<usize> = cpu.row(row).iter_nonzero().map(|(i, _)| i).collect();
        assert_eq!(
            g,
            c,
            "GPU/CPU restricted get_partial_matrix mismatch at degree {degree}, row {row} (input \
             {}, target_dim {target_dim}, num_rows {})",
            inputs[row],
            inputs.len(),
        );
    }
    cpu
}
