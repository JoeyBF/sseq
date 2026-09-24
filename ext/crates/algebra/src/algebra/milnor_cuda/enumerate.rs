//! Host side of the in-kernel admissible-matrix enumeration.
//!
//! Two passes, and the first one is the point. `emit = 0` runs the odometer and writes only the
//! per-`R` matrix COUNTS; the host prefix-sums those to decide where each `R`'s block goes, then
//! `emit = 1` fills it. So the sizes that decide the allocation come from the device, and the host
//! never enumerates anything -- which is the whole reason this kernel exists. Doing it the other
//! way round would mean a full CPU enumeration of every `R` just to learn how big its output is,
//! and that CPU enumeration is exactly the cost being avoided.

use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg, sys};

use super::{CudaError, MilnorCuda, Result, params};
use crate::algebra::milnor_algebra::PPart;

/// The kernel source, compiled at runtime by NVRTC.
const SRC: &str = include_str!("enumerate.cu");

fn module_key() -> String {
    let mut key = String::from("enumerate_admissible");
    for (name, value) in params::enum_defines() {
        key.push_str(&format!(":{name}={value}"));
    }
    key
}

/// One `R`'s dimensions, as the kernel needs them.
///
/// `cols` is the widest BIT-LENGTH of any entry, not the entry count -- the odometer's digits are
/// bit positions. Getting this wrong sizes every per-thread array wrongly.
pub(super) fn r_dims(p_part: &PPart) -> (usize, usize) {
    let rows = p_part.len();
    let cols = p_part
        .iter()
        .map(|x| (u32::BITS - x.leading_zeros()) as usize)
        .max()
        .unwrap_or(1)
        .max(1);
    (rows, cols)
}

/// Flattened, zero-padded inputs for a batch of `R`s: `(width, p_parts, rows, cols)`.
pub(super) fn layout(p_parts: &[PPart]) -> (usize, Vec<u32>, Vec<u32>, Vec<u32>) {
    let n_r = p_parts.len();
    let width = p_parts.iter().map(|p| p.len()).max().unwrap_or(1).max(1);
    let mut flat = vec![0u32; n_r * width];
    let mut r_rows = vec![0u32; n_r];
    let mut r_cols = vec![0u32; n_r];
    for (i, pp) in p_parts.iter().enumerate() {
        let (rows, cols) = r_dims(pp);
        for (slot, v) in flat[i * width..i * width + rows].iter_mut().zip(pp.iter()) {
            *slot = v;
        }
        r_rows[i] = rows as u32;
        r_cols[i] = cols as u32;
    }
    (width, flat, r_rows, r_cols)
}

/// Run the enumeration kernel, writing into buffers the CALLER owns.
///
/// This is what lets the resident master be built without the enumerated matrices ever touching the
/// host. The count pass sizes each `R`'s block, the caller grows its `GrowBuf`s and decides where
/// each block goes, and the emit pass writes there directly -- no staging buffer, no
/// device-to-device copy, no host round trip of a structure that reaches tens of GB.
///
/// `cs_ptr`/`mk_ptr` are raw device pointers; `cs_offsets`/`mk_offsets` are ELEMENT offsets into
/// them, one per `R`. With `emit = false` the pointers are never dereferenced and the offsets are
/// ignored, so a counting caller may pass anything valid to bind.
///
/// # Safety-adjacent
///
/// The caller must have committed enough backing store for every offset plus its block. Nothing
/// here can check that: the kernel writes where it is told. The two-pass discipline is what makes
/// it sound -- the offsets come from the counts the count pass produced, for the same `R`s in the
/// same order.
pub(super) fn enumerate_into(
    rt: &Arc<MilnorCuda>,
    p_parts: &[PPart],
    emit: bool,
    cs_ptr: sys::CUdeviceptr,
    mk_ptr: sys::CUdeviceptr,
    cs_offsets: &[u64],
    mk_offsets: &[u64],
) -> Result<Vec<u32>> {
    let n_r = p_parts.len();
    if n_r == 0 {
        return Ok(Vec::new());
    }
    assert_eq!(cs_offsets.len(), n_r, "one cs offset per R");
    assert_eq!(mk_offsets.len(), n_r, "one mk offset per R");

    let (width, flat, r_rows, r_cols) = layout(p_parts);
    check_caps(&r_rows, &r_cols)?;

    let module = rt.module(&module_key(), SRC, &params::enum_defines())?;
    let f = module
        .load_function("enumerate_admissible")
        .map_err(|e| CudaError::Compile(format!("load enumerate_admissible: {e:?}")))?;
    let stream = rt.context().default_stream();

    macro_rules! up {
        ($v:expr) => {
            stream
                .memcpy_stod(&$v)
                .map_err(|e| CudaError::Compile(format!("upload {}: {e:?}", stringify!($v))))?
        };
    }
    let d_pp = up!(flat);
    let d_rr = up!(r_rows);
    let d_rc = up!(r_cols);
    // `to_vec`: cudarc uploads from an owned slice type, and these arrive as borrowed slices.
    let cso = cs_offsets.to_vec();
    let mko = mk_offsets.to_vec();
    let d_cso = up!(cso);
    let d_mko = up!(mko);
    let mut d_counts = stream
        .alloc_zeros::<u32>(n_r)
        .map_err(|e| CudaError::Compile(format!("alloc counts: {e:?}")))?;

    let threads = params::ENUM_THREADS as u32;
    let cfg = LaunchConfig {
        grid_dim: ((n_r as u32).div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let width_u = width as u32;
    let n_r_u = n_r as u32;
    let emit_u = u32::from(emit);
    let cs_p = cs_ptr;
    let mk_p = mk_ptr;

    let mut b = stream.launch_builder(&f);
    b.arg(&d_pp)
        .arg(&d_rr)
        .arg(&d_rc)
        .arg(&d_cso)
        .arg(&d_mko)
        .arg(&cs_p)
        .arg(&mk_p)
        .arg(&mut d_counts)
        .arg(&width_u)
        .arg(&n_r_u)
        .arg(&emit_u);
    unsafe { b.launch(cfg) }
        .map_err(|e| CudaError::Compile(format!("launch enumerate (emit={emit}): {e:?}")))?;

    stream
        .clone_dtoh(&d_counts)
        .map_err(|e| CudaError::Compile(format!("readback counts: {e:?}")))
}

/// Guard the per-thread caps on the HOST, where it is an error message rather than a silent
/// out-of-bounds write into a neighbouring local array inside the kernel.
fn check_caps(r_rows: &[u32], r_cols: &[u32]) -> Result<()> {
    for (i, (&rows, &cols)) in r_rows.iter().zip(r_cols).enumerate() {
        if rows as usize > params::ENUM_ROW_CAP || cols as usize > params::ENUM_COL_CAP {
            return Err(CudaError::Compile(format!(
                "R #{i} is {rows}x{cols}, past the per-thread caps {}x{}; \
                 the kernel's local arrays would overflow",
                params::ENUM_ROW_CAP,
                params::ENUM_COL_CAP,
            )));
        }
    }
    Ok(())
}

/// What one enumeration launch produced.
pub struct Enumerated {
    /// Matrices per `R`, in the order given.
    pub num_mats: Vec<u32>,
    /// Per-`R` element offsets into `cs`/`mk`, and the per-matrix row lengths.
    pub cs_offset: Vec<u64>,
    pub mk_offset: Vec<u64>,
    pub cs_len: Vec<u32>,
    pub mk_len: Vec<u32>,
    /// The enumerated master, in the same layout the multiply kernel reads.
    pub cs: Vec<u16>,
    pub mk: Vec<u16>,
}

/// Enumerate every admissible matrix of each `R` on the device.
///
/// Returns the master in the same layout `resident::r_tables` produces on the host, so the two are
/// directly comparable -- which is how this is validated.
pub fn cuda_enumerate(rt: &Arc<MilnorCuda>, p_parts: &[PPart]) -> Result<Enumerated> {
    let n_r = p_parts.len();
    if n_r == 0 {
        return Ok(Enumerated {
            num_mats: Vec::new(),
            cs_offset: Vec::new(),
            mk_offset: Vec::new(),
            cs_len: Vec::new(),
            mk_len: Vec::new(),
            cs: Vec::new(),
            mk: Vec::new(),
        });
    }
    if let Some(bad) = p_parts.iter().position(|p| p.is_empty()) {
        return Err(CudaError::Compile(format!(
            "p_parts[{bad}] is empty; Sq(1) has no admissible matrices and is the caller's job"
        )));
    }

    let (width, flat, r_rows, r_cols) = layout(p_parts);
    let cs_len: Vec<u32> = r_cols.iter().map(|&c| c - 1).collect();
    let mk_len: Vec<u32> = r_rows
        .iter()
        .zip(&r_cols)
        .map(|(&r, &c)| r + c - 1)
        .collect();

    check_caps(&r_rows, &r_cols)?;

    let module = rt.module(&module_key(), SRC, &params::enum_defines())?;
    let f = module
        .load_function("enumerate_admissible")
        .map_err(|e| CudaError::Compile(format!("load enumerate_admissible: {e:?}")))?;
    let stream = rt.context().default_stream();

    macro_rules! up {
        ($v:expr) => {
            stream
                .memcpy_stod(&$v)
                .map_err(|e| CudaError::Compile(format!("upload {}: {e:?}", stringify!($v))))?
        };
    }
    let d_pp = up!(flat);
    let d_rr = up!(r_rows);
    let d_rc = up!(r_cols);

    let threads = params::ENUM_THREADS as u32;
    let cfg = LaunchConfig {
        grid_dim: ((n_r as u32).div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let width_u = width as u32;
    let n_r_u = n_r as u32;

    // PASS 1 -- counts only. The output buffers must still bind (the arguments are not optional),
    // so a one-element dummy stands in; the kernel never writes it with `emit = 0`.
    let zero_offsets = vec![0u64; n_r];
    let d_cso0 = up!(zero_offsets);
    let d_mko0 = stream
        .memcpy_stod(&vec![0u64; n_r])
        .map_err(|e| CudaError::Compile(format!("upload mk offsets: {e:?}")))?;
    let mut d_dummy_cs = stream
        .alloc_zeros::<u16>(1)
        .map_err(|e| CudaError::Compile(format!("alloc dummy: {e:?}")))?;
    let mut d_dummy_mk = stream
        .alloc_zeros::<u16>(1)
        .map_err(|e| CudaError::Compile(format!("alloc dummy: {e:?}")))?;
    let mut d_counts = stream
        .alloc_zeros::<u32>(n_r)
        .map_err(|e| CudaError::Compile(format!("alloc counts: {e:?}")))?;

    let emit0 = 0u32;
    let mut b = stream.launch_builder(&f);
    b.arg(&d_pp)
        .arg(&d_rr)
        .arg(&d_rc)
        .arg(&d_cso0)
        .arg(&d_mko0)
        .arg(&mut d_dummy_cs)
        .arg(&mut d_dummy_mk)
        .arg(&mut d_counts)
        .arg(&width_u)
        .arg(&n_r_u)
        .arg(&emit0);
    unsafe { b.launch(cfg) }
        .map_err(|e| CudaError::Compile(format!("launch enumerate (count): {e:?}")))?;

    let num_mats = stream
        .clone_dtoh(&d_counts)
        .map_err(|e| CudaError::Compile(format!("readback counts: {e:?}")))?;

    // Prefix-sum the device's own counts into per-`R` offsets.
    let mut cs_offset = Vec::with_capacity(n_r);
    let mut mk_offset = Vec::with_capacity(n_r);
    let (mut cs_at, mut mk_at) = (0u64, 0u64);
    for i in 0..n_r {
        cs_offset.push(cs_at);
        mk_offset.push(mk_at);
        cs_at += num_mats[i] as u64 * cs_len[i] as u64;
        mk_at += num_mats[i] as u64 * mk_len[i] as u64;
    }

    // PASS 2 -- fill.
    let d_cso = up!(cs_offset);
    let d_mko = up!(mk_offset);
    let mut d_cs = stream
        .alloc_zeros::<u16>(cs_at as usize)
        .map_err(|e| CudaError::Compile(format!("alloc cs scratch: {e:?}")))?;
    let mut d_mk = stream
        .alloc_zeros::<u16>(mk_at as usize)
        .map_err(|e| CudaError::Compile(format!("alloc mk scratch: {e:?}")))?;
    let mut d_counts2 = stream
        .alloc_zeros::<u32>(n_r)
        .map_err(|e| CudaError::Compile(format!("alloc counts: {e:?}")))?;

    let emit1 = 1u32;
    let mut b = stream.launch_builder(&f);
    b.arg(&d_pp)
        .arg(&d_rr)
        .arg(&d_rc)
        .arg(&d_cso)
        .arg(&d_mko)
        .arg(&mut d_cs)
        .arg(&mut d_mk)
        .arg(&mut d_counts2)
        .arg(&width_u)
        .arg(&n_r_u)
        .arg(&emit1);
    unsafe { b.launch(cfg) }
        .map_err(|e| CudaError::Compile(format!("launch enumerate (emit): {e:?}")))?;

    let cs = stream
        .clone_dtoh(&d_cs)
        .map_err(|e| CudaError::Compile(format!("readback cs: {e:?}")))?;
    let mk = stream
        .clone_dtoh(&d_mk)
        .map_err(|e| CudaError::Compile(format!("readback mk: {e:?}")))?;
    let recount = stream
        .clone_dtoh(&d_counts2)
        .map_err(|e| CudaError::Compile(format!("readback counts: {e:?}")))?;

    // The two passes must agree. They run the same odometer, so a disagreement means the emit pass
    // wrote outside the space the count pass sized -- i.e. the master is already corrupt, and the
    // multiply that consumes it would return a plausible wrong answer rather than fail.
    if recount != num_mats {
        return Err(CudaError::Compile(
            "the count and emit passes disagree on the matrix count; the master is not trustworthy"
                .to_owned(),
        ));
    }

    Ok(Enumerated {
        num_mats,
        cs_offset,
        mk_offset,
        cs_len,
        mk_len,
        cs,
        mk,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fp::prime::ValidPrime;

    use super::*;
    use crate::algebra::{Algebra, MilnorAlgebra, milnor_cuda::resident::r_tables};

    /// Every `R` up to `max_degree`, deduplicated by p-part.
    fn all_r(algebra: &MilnorAlgebra, max_degree: i32) -> Vec<PPart> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for d in 1..=max_degree {
            for i in 0..algebra.dimension(d) {
                let p = algebra.basis_element_from_index(d, i).p_part.clone();
                if !p.is_empty() && seen.insert(p.clone()) {
                    out.push(p);
                }
            }
        }
        out
    }

    /// The device enumeration must reproduce `admissible_matrices` EXACTLY, for every `R`.
    ///
    /// Not a digest comparison: this checks each `R`'s block element by element, because the
    /// failure that matters is one `R` landing at the wrong offset, which a whole-buffer digest
    /// reports as "everything is wrong" with no way in.
    ///
    /// The counts are checked first and separately. They come from the count pass, and they are
    /// what sizes the allocation -- a wrong count is not a wrong answer, it is an out-of-bounds
    /// write into the next `R`'s block.
    #[test]
    #[ignore = "needs a CUDA device; run explicitly with --ignored"]
    fn cuda_enumerate_matches_cpu_reference() {
        let max_degree = 50;
        let algebra = Arc::new(MilnorAlgebra::new(ValidPrime::new(2), false));
        algebra.compute_basis(max_degree);
        let rs = all_r(&algebra, max_degree);
        assert!(rs.len() > 100, "too few distinct R to be a real check");

        let rt = super::super::runtime(0).expect("open device 0");
        let got = cuda_enumerate(&rt, &rs).expect("device enumeration");

        let mut total_mats = 0u64;
        for (i, p) in rs.iter().enumerate() {
            let (cs_len, mk_len, num_mats, cs, mk) = r_tables(&algebra, p).expect("cpu tables");
            assert_eq!(
                got.num_mats[i] as usize, num_mats,
                "R #{i} ({p:?}): device counted {} matrices, CPU {num_mats}",
                got.num_mats[i]
            );
            assert_eq!(got.cs_len[i] as usize, cs_len, "R #{i}: cs_len");
            assert_eq!(got.mk_len[i] as usize, mk_len, "R #{i}: mk_len");

            let cs_at = got.cs_offset[i] as usize;
            let mk_at = got.mk_offset[i] as usize;
            assert_eq!(
                &got.cs[cs_at..cs_at + cs.len()],
                &cs[..],
                "R #{i} ({p:?}): col_sums differ"
            );
            assert_eq!(
                &got.mk[mk_at..mk_at + mk.len()],
                &mk[..],
                "R #{i} ({p:?}): masks differ"
            );
            total_mats += num_mats as u64;
        }
        eprintln!(
            "[enum] {} distinct R, {total_mats} matrices, cs={} mk={} u16",
            rs.len(),
            got.cs.len(),
            got.mk.len()
        );
        assert!(total_mats > 1000, "too few matrices to be a real check");
    }

    /// `cols` is a BIT-LENGTH, and the per-thread caps depend on it. A real `R` that exceeds them
    /// would overflow a local array in the kernel, so check the bound against actual `R`s rather
    /// than trusting the derivation.
    #[test]
    fn enum_caps_bound_real_rs() {
        let max_degree = 60;
        let algebra = Arc::new(MilnorAlgebra::new(ValidPrime::new(2), false));
        algebra.compute_basis(max_degree);
        let rs = all_r(&algebra, max_degree);
        assert!(!rs.is_empty());
        for p in &rs {
            let (rows, cols) = r_dims(p);
            assert!(
                rows <= params::ENUM_ROW_CAP,
                "{p:?} has {rows} rows, past ENUM_ROW_CAP {}",
                params::ENUM_ROW_CAP
            );
            assert!(
                cols <= params::ENUM_COL_CAP,
                "{p:?} needs {cols} bit-columns, past ENUM_COL_CAP {}",
                params::ENUM_COL_CAP
            );
        }
    }
}
