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
//! same layout the CPU path produces — as a limb-wise XOR, since the kernel's little-endian `u32`
//! limbs are byte-identical to `fp`'s `u64` limbs.
//!
//! Gated behind the `gpu` feature. Callers must ensure
//! [`MilnorAlgebra::gpu_multiply_applicable`] (`p = 2`, trivial profile, stable) — the
//! Nassau `S_2` regime — and that the algebra's basis and seqno tables reach `degree`.

use algebra::{
    MilnorAlgebra,
    milnor_gpu::{COL_MAP_DROP, GpuProduct, multiply_batch_on_gpu, multiply_batch_on_gpu_masked},
    module::{
        FreeModule, Module,
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
    },
};
use fp::{matrix::Matrix, vector::FpVector};

type NassauDifferential = FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>;

/// Reinterpret a GPU output row's `u32` limbs as their little-endian bytes.
///
/// The kernel's `u32` limbs and `fp`'s `u64` limbs are the same bit-vector in the same byte order,
/// so this is a view, not a conversion. `fp`'s own `limb::from_bytes`/`to_bytes` take exactly this
/// shortcut under the same `cfg`; the fallback keeps a big-endian target correct rather than
/// silently wrong.
///
/// One bulk `memcpy` per row. The first cut wrote `w.to_le_bytes()` into `buf` one `u32` at a time,
/// which a call-graph profile of an uncapped stem-150 run showed as 12.15% of ALL user cycles under
/// `copy_from_slice` — a bounds-checked 4-byte copy per limb, thousands per row, for a region that
/// is already byte-identical.
fn fill_limb_bytes(buf: &mut [u8], limbs: &[u32]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: `u32` has no padding or invalid bit patterns, `u8` has alignment 1 (so a `u32`
        // pointer is suitably aligned), and the length is the same region measured in bytes. The
        // view borrows `limbs` and does not outlive it.
        let src: &[u8] = unsafe {
            std::slice::from_raw_parts(limbs.as_ptr().cast::<u8>(), std::mem::size_of_val(limbs))
        };
        let n = src.len().min(buf.len());
        buf[..n].copy_from_slice(&src[..n]);
        buf[n..].fill(0);
    }
    #[cfg(not(target_endian = "little"))]
    {
        buf.fill(0);
        for (k, &w) in limbs.iter().enumerate() {
            let o = k * size_of::<u32>();
            if o >= buf.len() {
                break;
            }
            let n = size_of::<u32>().min(buf.len() - o);
            buf[o..o + n].copy_from_slice(&w.to_le_bytes()[..n]);
        }
    }
}

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
    // An empty signature mask means a zero-column matrix: `extract_restricted` has already returned
    // one of the right shape, and there is nothing to multiply into it. Guarding here also keeps
    // `nbytes` below non-zero -- it is `num_limbs(row_cols) * 8`, and the readback's
    // `nbytes - size_of::<u64>()` underflows when the mask is empty. Reachable in practice: the
    // verification run panicked here within seconds. The unmasked path cannot hit it, because
    // `extract_restricted` already returns early on `target_dim == 0`.
    let row_cols_nonzero = col_mask.map_or(target_dim > 0, |m| !m.is_empty());
    if !products.is_empty() && row_cols_nonzero {
        let p = hom.prime();
        let target = hom.target();
        let algebra = target.algebra();
        // Idempotent + cheap (O(degree · width)); returns immediately once built.
        algebra.compute_seqno_tables(degree);
        let num_cols = target.dimension(degree);
        let out = multiply_batch_on_gpu(&algebra, num_cols, inputs.len(), &products);
        // Limb-wise readback; see the equivalent (truncating) loop in
        // [`get_partial_matrix_restricted`] for why the byte copy is valid. Here the widths already
        // agree, so only the partial final limb needs masking.
        let num_limbs = FpVector::num_limbs(p, num_cols);
        let nbytes = num_limbs * size_of::<u64>();
        let mut scratch = FpVector::new(p, num_cols);
        let mut buf: Vec<u8> = vec![0; nbytes];
        let tail_mask: u64 = match num_cols % 64 {
            0 => u64::MAX,
            r => (1u64 << r) - 1,
        };
        for (row, limbs) in out.iter_rows().enumerate() {
            fill_limb_bytes(&mut buf, limbs);
            let last = nbytes - size_of::<u64>();
            let masked = u64::from_le_bytes(buf[last..].try_into().unwrap()) & tail_mask;
            buf[last..].copy_from_slice(&masked.to_le_bytes());
            scratch
                .update_from_bytes(&mut &buf[..])
                .expect("readback scratch is exactly num_limbs * 8 bytes");
            matrix.row_mut(row).add(scratch.as_slice(), 1);
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
    build_restricted(hom, degree, inputs, target_dim, None)
}

/// Shared body of the full-width and column-masked restricted builds.
///
/// `col_mask == None` reproduces the historical behaviour exactly. With `Some(mask)` the returned
/// matrix has `mask.len()` columns and no full-width matrix is ever allocated -- the gather happens
/// per row, against the readback scratch that already existed. The multiply itself is unchanged:
/// it still runs at the full output width because a product's `out_offset + seqno` indexes the full
/// output-degree basis.
fn build_restricted(
    hom: &NassauDifferential,
    degree: i32,
    inputs: &[usize],
    target_dim: usize,
    col_mask: Option<&[usize]>,
) -> Matrix {
    // Spanned because the 279 s stalls land somewhere in this function *before* the multiply
    // (which itself measured 40 ms), and neither this build nor the pair pre-pass inside
    // `multiply_batch_on_gpu` was previously visible to the log.
    let (mut matrix, mut products) =
        tracing::info_span!("extract_restricted", inputs = inputs.len(), target_dim)
            .in_scope(|| extract_restricted(hom, degree, inputs, target_dim, col_mask));
    if !products.is_empty() {
        let p = hom.prime();
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
        // COLUMN RESTRICTION. Under a mask the kernel writes masked positions directly, so `out` is
        // `mask.len()` wide instead of `full_cols` -- ~24k against ~1.15M at a stem-290 A(2)
        // bidegree. That shrinks the device buffer and the readback by the same ratio, and because
        // `gpu_rows_per_batch` sizes its cap off this width, it also stops a build being chopped
        // into batches it never needed (~7.5k rows per batch at the full width, vs the whole build
        // in one at the masked width).
        //
        // The map is indexed by FULL column because that is what `out_offset + seqno` produces
        // inside the kernel; entries outside the mask hold `COL_MAP_DROP`.
        //
        // The unmasked path is deliberately left exactly as it was, launching at `full_cols` and
        // truncating on readback: the kernel's row layout is keyed to the width it is given, and
        // narrowing that is a separate change from restricting the columns.
        let col_map: Option<std::sync::Arc<[u32]>> = col_mask.map(|mask| {
            let mut m = vec![COL_MAP_DROP; full_cols];
            for (j, &c) in mask.iter().enumerate() {
                if c < full_cols {
                    m[c] = j as u32;
                }
            }
            m.into()
        });
        // Width the kernel writes at, and therefore the width of every readback row.
        let kernel_cols = col_mask.map_or(full_cols, <[usize]>::len);
        // Width the host-side scratch row carries. Masked: the row is already the matrix's width.
        // Unmasked: the restricted prefix, as before.
        let row_cols = col_mask.map_or(target_dim, <[usize]>::len);
        // Cap how large a single multiply we hand the GPU: the dense readback (num_rows × num_limbs
        // u32) plus the matrix would otherwise both be held for the whole all-rows / zero-signature
        // build (~12 GB dense regions at stem 180). Process the rows in batches of ≤ `rows_per_batch`
        // so the readback stays bounded and is freed between batches. `products` is built in row
        // order (`extract_restricted`), so each batch's products are a contiguous slice; we remap
        // their `row` to batch-local (0-based) for the kernel and write back to the global rows.
        let rows_per_batch = gpu_rows_per_batch(kernel_cols, inputs.len());
        // Readback scratch, allocated once for the whole call (`target_dim` is fixed): the GPU's
        // per-row output is XORed into the matrix through a limb-wise `add` rather than bit by bit.
        //
        // Both sides are little-endian packed F_2 bitvectors with bit `i` = column `i`, so four of
        // the kernel's `u32` limbs ARE one of `fp`'s `u64` limbs, byte for byte — no transposition,
        // just a truncating copy. `update_from_bytes` fills the existing limbs in place (no
        // allocation, no resize), and `read_exact` demands exactly `num_limbs * 8` bytes, which is
        // why `buf` is sized once and refilled rather than sliced per row.
        //
        // The bit-at-a-time loop this replaces called `add_basis_element` once per set bit: at the
        // logged ~26% density that is ~0.26 * cols read-modify-writes per row against cols/64 limb
        // XORs here, and each one was a bounds-checked entry write rather than a word XOR.
        let num_limbs = FpVector::num_limbs(p, row_cols);
        let nbytes = num_limbs * size_of::<u64>();
        let mut scratch = FpVector::new(p, row_cols);
        let mut buf: Vec<u8> = vec![0; nbytes];
        // Bits at or past `target_dim` inside the final limb must not survive into the vector —
        // `FpVector` requires them zero, and dropping them is exactly what the old `col < target_dim`
        // guard did. Whole limbs past the end are dropped by `buf` being only `nbytes` long.
        let tail_mask: u64 = match row_cols % 64 {
            0 => u64::MAX,
            r => (1u64 << r) - 1,
        };
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
                let out = multiply_batch_on_gpu_masked(
                    &algebra,
                    kernel_cols,
                    col_map.clone(),
                    r1 - r0,
                    &products[p0..p1],
                );
                let _scatter = tracing::info_span!("gpu_readback", rows = r1 - r0).entered();
                for (bi, limbs) in out.iter_rows().enumerate() {
                    // Reinterpret this row's `u32` limbs as the vector's little-endian limb bytes,
                    // truncated at `target_dim` (both directions: partial final limb, and whole
                    // limbs past the restricted prefix).
                    fill_limb_bytes(&mut buf, limbs);
                    let last = nbytes - size_of::<u64>();
                    let masked = u64::from_le_bytes(buf[last..].try_into().unwrap()) & tail_mask;
                    buf[last..].copy_from_slice(&masked.to_le_bytes());
                    scratch
                        .update_from_bytes(&mut &buf[..])
                        .expect("readback scratch is exactly num_limbs * 8 bytes");
                    // The scratch is full width either way -- it is one ROW, not one matrix. Only
                    // the destination narrows, which is where the ~49x allocation saving is.
                    // The gather that `add_masked` used to do now happens on the device, so the
                    // row arrives already in masked coordinates and goes straight in.
                    matrix.row_mut(r0 + bi).add(scratch.as_slice(), 1);
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
    col_mask: Option<&[usize]>,
) -> (Matrix, Vec<GpuProduct>) {
    let p = hom.prime();
    let source = hom.source();
    let target = hom.target();
    let shift = hom.degree_shift();
    let out_cols = col_mask.map_or(target_dim, <[usize]>::len);
    let mut matrix = Matrix::new(p, inputs.len(), out_cols);
    let mut products: Vec<GpuProduct> = Vec::new();
    // The `operation_degree == 0` rows below are written by the CPU apply, which produces a
    // full-width row. Under a mask they need a scratch to land in first; allocated once here
    // rather than per row, and only when masking is actually on.
    let mut cpu_scratch = col_mask.map(|_| FpVector::new(p, target_dim));

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
            match (&mut cpu_scratch, col_mask) {
                (Some(scratch), Some(mask)) => {
                    scratch.set_to_zero();
                    hom.apply_to_basis_element_restricted(
                        scratch.as_slice_mut(),
                        1,
                        degree,
                        input_index,
                    );
                    matrix.row_mut(row).add_masked(scratch.as_slice(), 1, mask);
                }
                _ => {
                    hom.apply_to_basis_element_restricted(
                        matrix.row_mut(row),
                        1,
                        degree,
                        input_index,
                    );
                }
            }
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
/// As [`get_partial_matrix_restricted`], but the returned matrix has one column per entry of
/// `col_mask` instead of `target_dim` columns: `out[i][j] == full[i][col_mask[j]]`.
///
/// # Why this exists
///
/// The consumer of a restricted partial matrix immediately masks its columns
/// (`add_masked(row, 1, &next_mask)`), and the mask keeps ~2% of them. Materialising the full
/// width first is the single largest live allocation in the run: measured at the stem-285 frontier,
/// 37703 x 1914035 is **9.0 GB** against 183 MB for the masked form, and 143 GB across the
/// frontier's bidegrees against 7.5 GB used -- which matches, to 0.3%, the 142.9 GB a heap dump
/// attributes to in-flight matrices.
///
/// Only the *destination* narrows here. The multiply still runs at the full output width, because
/// a product's `out_offset + seqno` indexes the full output-degree basis and `num_cols` sets the
/// kernel's limb layout (see [`get_partial_matrix_restricted`]); pushing the mask into the kernel's
/// output indexing is a further, larger change. What this removes is the wide *host* matrix and the
/// separate `add_masked` pass over it.
///
/// The per-row readback scratch stays full width (one row, not one matrix) and the gather replaces
/// the limb-wise copy with `add_masked`. That trades `target_dim/64` limb XORs per row for
/// `col_mask.len()` masked reads -- at the frontier, 29907 against 38867, i.e. comparable, while
/// the allocation drops ~49x.
pub fn get_partial_matrix_restricted_masked(
    hom: &NassauDifferential,
    degree: i32,
    inputs: &[usize],
    target_dim: usize,
    col_mask: &[usize],
) -> Matrix {
    build_restricted(hom, degree, inputs, target_dim, Some(col_mask))
}

/// `NASSAU_GPU_VERIFY` gate for the masked build.
///
/// Checks the property the caller actually depends on: the masked matrix equals the full matrix
/// with `col_mask` applied. That catches a wrong gather as well as a wrong multiply, which
/// comparing two masked builds against each other would not.
pub fn get_partial_matrix_restricted_masked_verified(
    hom: &NassauDifferential,
    degree: i32,
    inputs: &[usize],
    target_dim: usize,
    col_mask: &[usize],
) -> Matrix {
    let masked = build_restricted(hom, degree, inputs, target_dim, Some(col_mask));
    // Reference: the CPU restricted apply at FULL width, then the mask. No GPU, no batching.
    let p = hom.prime();
    let mut full = Matrix::new(p, inputs.len(), target_dim);
    if target_dim > 0 {
        for (i, &input) in inputs.iter().enumerate() {
            hom.apply_to_basis_element_restricted(full.row_mut(i), 1, degree, input);
        }
    }
    let mut want = Matrix::new(p, inputs.len(), col_mask.len());
    for i in 0..inputs.len() {
        want.row_mut(i).add_masked(full.row(i), 1, col_mask);
    }
    for row in 0..inputs.len() {
        let g: Vec<usize> = masked.row(row).iter_nonzero().map(|(i, _)| i).collect();
        let c: Vec<usize> = want.row(row).iter_nonzero().map(|(i, _)| i).collect();
        assert_eq!(
            g,
            c,
            "masked restricted get_partial_matrix mismatch at degree {degree}, row {row} (input \
             {}, target_dim {target_dim}, mask_len {}, num_rows {})",
            inputs[row],
            col_mask.len(),
            inputs.len(),
        );
    }
    masked
}

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
