//! Host side of the batched Milnor multiply: marshal a `[GpuProduct]` into the buffers
//! [`multiply.cu`](../multiply.cu) reads, launch, and read the limbs back.
//!
//! Like the kernel, this is the SIMPLEST form that is correct. Everything the cubecl host path
//! does to avoid re-uploading data -- the process-shared resident master, the append-only resident
//! basis, the per-device segment stores, the row batching and the un-awaited readback -- is
//! deliberately absent. Those exist to make repeated launches cheap; none of them changes the
//! answer, and each is far easier to add back against a path already known to agree with
//! [`cpu_multiply_batch`] than to debug alongside a fresh kernel.
//!
//! So this marshals everything per call. It is the right shape for validation and the wrong shape
//! for production, and the next commits close that gap one piece at a time.

use std::collections::HashMap;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaError, MilnorCuda, Result, params};
use crate::algebra::{
    Algebra, MilnorAlgebra,
    combinatorics::xi_degrees,
    milnor_batch::{BatchOutput, GpuProduct},
};

/// The kernel source, compiled at runtime by NVRTC.
const SRC: &str = include_str!("multiply.cu");

/// Module cache key. Must change whenever the defines do, which they do only when `params.rs`
/// changes -- so the constants are folded into the key rather than trusted to stay put.
fn module_key() -> String {
    let mut key = String::from("multiply_batch");
    for (name, value) in params::defines() {
        key.push_str(&format!(":{name}={value}"));
    }
    key
}

/// Everything one launch needs, in device layout.
///
/// Built by [`marshal`] and kept separate from the launch so a test can inspect it, and so the
/// eventual resident-master version has an obvious seam to replace: only the `cs`/`mk`/`pp`/`ln`
/// fields become long-lived, and the rest stays per launch.
struct Marshalled {
    /// Admissible-matrix master, concatenated over distinct `R`s.
    cs: Vec<u16>,
    mk: Vec<u16>,
    /// Width-padded Milnor basis and true p-part lengths, indexed by global element index.
    pp: Vec<u16>,
    ln: Vec<u32>,
    /// Per distinct `R`.
    r_cs_offset: Vec<u64>,
    r_mk_offset: Vec<u64>,
    r_cs_len: Vec<u32>,
    r_mk_len: Vec<u32>,
    r_num_mats: Vec<u32>,
    /// Per product, in the order given.
    term_gei: Vec<u32>,
    prod_r_index: Vec<u32>,
    prod_term_start: Vec<u32>,
    prod_num_terms: Vec<u32>,
    prod_row_base: Vec<u32>,
    prod_out_offset: Vec<u32>,
    /// Prefix sum of pair counts, length `num_products + 1`.
    prod_pair_start: Vec<u64>,
    /// Shared stride of the `g` table and the padded basis.
    width: usize,
    g: Vec<u32>,
    xi: Vec<u32>,
}

impl Marshalled {
    fn num_products(&self) -> usize {
        self.prod_r_index.len()
    }

    fn total_pairs(&self) -> u64 {
        *self.prod_pair_start.last().unwrap_or(&0)
    }
}

/// `col_sums`/`masks` arrive as `u32` but are stored as `u16` on the device, halving the master.
///
/// Checked rather than truncated. The one production bug this port must not reintroduce was
/// exactly an unchecked narrowing: the masks master crossing `2^32` was read back through a `u16`
/// index and silently truncated, which surfaced stems later as a non-zero differential at (180,92).
fn narrow(values: &[u32], what: &str) -> Result<Vec<u16>> {
    values
        .iter()
        .map(|&v| {
            u16::try_from(v).map_err(|_| {
                CudaError::Compile(format!(
                    "{what} entry {v} does not fit u16; master layout wrong"
                ))
            })
        })
        .collect()
}

/// Build the device-side inputs for `products`.
///
/// `algebra` must have its seqno tables built through the highest output degree any product
/// reaches; the caller owns that because building them is degree-monotone and shared.
fn marshal(
    algebra: &MilnorAlgebra,
    num_rows: usize,
    num_limbs: usize,
    products: &[GpuProduct],
) -> Result<Marshalled> {
    let (width, g) = algebra.seqno_table_u32();

    // The basis has to cover every term's degree, and `g`/`width` fix the padding stride.
    let max_s_degree = products.iter().map(|p| p.s_degree).max().unwrap_or(0);
    let mut pp: Vec<u16> = Vec::new();
    let mut ln: Vec<u32> = Vec::new();
    // `global_base[d]` is the number of basis elements in degrees < d, so a term `(s_degree, ti)`
    // maps to `gei = global_base[s_degree] + ti`. The concatenation order fixes every element's
    // index, exactly as the resident basis does.
    let mut global_base: Vec<u32> = vec![0];
    for d in 0..=max_s_degree {
        let dim = algebra.dimension(d);
        for i in 0..dim {
            let elt = algebra.basis_element_from_index(d, i);
            ln.push(elt.p_part.len() as u32);
            let base = pp.len();
            pp.resize(base + width, 0);
            for (slot, v) in pp[base..base + width].iter_mut().zip(elt.p_part.iter()) {
                *slot = u16::try_from(v).map_err(|_| {
                    CudaError::Compile(format!("basis p-part entry {v} does not fit u16"))
                })?;
            }
        }
        global_base.push(ln.len() as u32);
    }

    // Distinct `R`s, deduplicated: an `R` shared across many rows is enumerated and uploaded once.
    let mut r_index: HashMap<(i32, usize), u32> = HashMap::new();
    let mut cs: Vec<u16> = Vec::new();
    let mut mk: Vec<u16> = Vec::new();
    let (mut r_cs_offset, mut r_mk_offset) = (Vec::new(), Vec::new());
    let (mut r_cs_len, mut r_mk_len, mut r_num_mats) = (Vec::new(), Vec::new(), Vec::new());

    let mut term_gei: Vec<u32> = Vec::new();
    let mut prod_r_index: Vec<u32> = Vec::new();
    let mut prod_term_start: Vec<u32> = Vec::new();
    let mut prod_num_terms: Vec<u32> = Vec::new();
    let mut prod_row_base: Vec<u32> = Vec::new();
    let mut prod_out_offset: Vec<u32> = Vec::new();
    let mut prod_pair_start: Vec<u64> = vec![0];

    for prod in products {
        // A product whose output degree is empty contributes nothing; the CPU reference skips it
        // and so must this, or its `num_mats * num_terms` pairs would all reject at a cost.
        if algebra.dimension(prod.r_degree + prod.s_degree) == 0 || prod.term_indices.is_empty() {
            continue;
        }
        let key = (prod.r_degree, prod.r_idx);
        let ri = match r_index.get(&key) {
            Some(&ri) => ri,
            None => {
                let r_p_part = algebra
                    .basis_element_from_index(prod.r_degree, prod.r_idx)
                    .p_part
                    .clone();
                // `Sq(empty) = 1` has no admissible matrices and is the caller's job, exactly as
                // in `AdmissibleMatrix::new`. It cannot reach a batch: the resolution never emits
                // a degree-0 operation as a product.
                if r_p_part.is_empty() {
                    return Err(CudaError::Compile(format!(
                        "product at r_degree {} has an empty R; Sq(1) is not a batch product",
                        prod.r_degree
                    )));
                }
                let (cs_len, mk_len, cs_v, mk_v) = algebra.admissible_matrices(r_p_part);
                let num_mats = if cs_len > 0 {
                    cs_v.len() / cs_len
                } else {
                    mk_v.len() / mk_len.max(1)
                };
                let ri = r_cs_len.len() as u32;
                r_cs_offset.push(cs.len() as u64);
                r_mk_offset.push(mk.len() as u64);
                r_cs_len.push(cs_len as u32);
                r_mk_len.push(mk_len as u32);
                r_num_mats.push(num_mats as u32);
                cs.extend(narrow(&cs_v, "col_sums")?);
                mk.extend(narrow(&mk_v, "masks")?);
                r_index.insert(key, ri);
                ri
            }
        };

        let base = global_base[prod.s_degree as usize];
        prod_term_start.push(term_gei.len() as u32);
        prod_num_terms.push(prod.term_indices.len() as u32);
        for &ti in prod.term_indices.iter() {
            term_gei.push(base + ti as u32);
        }
        prod_r_index.push(ri);
        prod_row_base.push((prod.row * num_limbs) as u32);
        prod_out_offset.push(prod.out_offset as u32);
        let pairs = r_num_mats[ri as usize] as u64 * prod.term_indices.len() as u64;
        prod_pair_start.push(prod_pair_start.last().unwrap() + pairs);
    }

    debug_assert!(
        products.iter().all(|p| p.row < num_rows),
        "row out of range"
    );

    // `xi` is indexed to `PPART_MAX_LEN` by the seqno loop regardless of how long the assembled
    // p-part is, since the entries past it are zero and contribute `0 * xi`. Pad so those reads
    // stay in bounds.
    let mut xi: Vec<u32> = xi_degrees(algebra.prime())
        .iter()
        .map(|&d| d as u32)
        .collect();
    xi.resize(xi.len().max(params::PPART_MAX_LEN), 0);

    Ok(Marshalled {
        cs,
        mk,
        pp,
        ln,
        r_cs_offset,
        r_mk_offset,
        r_cs_len,
        r_mk_len,
        r_num_mats,
        term_gei,
        prod_r_index,
        prod_term_start,
        prod_num_terms,
        prod_row_base,
        prod_out_offset,
        prod_pair_start,
        width,
        g,
        xi,
    })
}

/// Run a batch on the device and return the limbs, matching [`cpu_multiply_batch`] bit for bit.
///
/// `col_map` restricts the output to the masked columns, exactly as `cpu_multiply_batch_masked`
/// does: `num_cols` is then the MASKED width and `col_map.len()` the full one.
///
/// [`cpu_multiply_batch`]: crate::algebra::milnor_batch::cpu_multiply_batch
pub fn cuda_multiply_batch(
    rt: &MilnorCuda,
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
    col_map: Option<&[u32]>,
) -> Result<BatchOutput> {
    let num_limbs = num_cols.div_ceil(32).max(1);
    let m = marshal(algebra, num_rows, num_limbs, products)?;
    let out_len = num_rows * num_limbs;

    if m.total_pairs() == 0 {
        return Ok(BatchOutput::from_limbs(vec![0u32; out_len], num_limbs));
    }

    let module = rt.module(&module_key(), SRC, &params::defines())?;
    let f = module
        .load_function("multiply_batch")
        .map_err(|e| CudaError::Compile(format!("load multiply_batch: {e:?}")))?;
    let stream = rt.context().default_stream();

    macro_rules! up {
        ($v:expr) => {
            stream
                .memcpy_stod(&$v)
                .map_err(|e| CudaError::Compile(format!("upload {}: {e:?}", stringify!($v))))?
        };
    }
    let d_cs = up!(m.cs);
    let d_mk = up!(m.mk);
    let d_pp = up!(m.pp);
    let d_ln = up!(m.ln);
    let d_tg = up!(m.term_gei);
    let d_g = up!(m.g);
    let d_xi = up!(m.xi);
    let d_rco = up!(m.r_cs_offset);
    let d_rmo = up!(m.r_mk_offset);
    let d_rcl = up!(m.r_cs_len);
    let d_rml = up!(m.r_mk_len);
    let d_rnm = up!(m.r_num_mats);
    let d_pri = up!(m.prod_r_index);
    let d_pts = up!(m.prod_term_start);
    let d_pnt = up!(m.prod_num_terms);
    let d_prb = up!(m.prod_row_base);
    let d_poo = up!(m.prod_out_offset);
    let d_pps = up!(m.prod_pair_start);
    // The argument is not optional, so an unrestricted launch binds a one-element dummy the kernel
    // never reads.
    let cm: Vec<u32> = col_map.map_or_else(|| vec![0u32], <[u32]>::to_vec);
    let d_cm = up!(cm);

    let mut d_out = stream
        .alloc_zeros::<u32>(out_len)
        .map_err(|e| CudaError::Compile(format!("alloc out: {e:?}")))?;

    // Scalars need bindings: `arg` borrows, so a temporary would be dropped before the launch.
    let num_products = m.num_products() as u32;
    let col_map_len = col_map.map_or(0u32, |c| c.len() as u32);
    let use_col_map = u32::from(col_map.is_some());
    let width = m.width as u32;
    let num_limbs_u = num_limbs as u32;
    let out_len_u = out_len as u64;

    // The pair space is walked in pieces that each fit the 32-bit thread index, so a single
    // oversized row cannot overflow the grid.
    let threads = params::THREADS as u32;
    let chunk = (u32::MAX as u64 / threads as u64) * threads as u64;
    let total = m.total_pairs();
    let mut done: u64 = 0;
    while done < total {
        let n = (total - done).min(chunk);
        let pair_offset = done;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&f);
        b.arg(&d_cs)
            .arg(&d_mk)
            .arg(&d_pp)
            .arg(&d_ln)
            .arg(&d_tg)
            .arg(&d_g)
            .arg(&d_xi)
            .arg(&mut d_out)
            .arg(&d_cm)
            .arg(&col_map_len)
            .arg(&use_col_map)
            .arg(&d_rco)
            .arg(&d_rmo)
            .arg(&d_rcl)
            .arg(&d_rml)
            .arg(&d_rnm)
            .arg(&d_pri)
            .arg(&d_pts)
            .arg(&d_pnt)
            .arg(&d_prb)
            .arg(&d_poo)
            .arg(&d_pps)
            .arg(&num_products)
            .arg(&pair_offset)
            .arg(&width)
            .arg(&num_limbs_u)
            .arg(&out_len_u);
        unsafe { b.launch(cfg) }
            .map_err(|e| CudaError::Compile(format!("launch multiply_batch: {e:?}")))?;
        done += n;
    }

    let limbs = stream
        .clone_dtoh(&d_out)
        .map_err(|e| CudaError::Compile(format!("readback: {e:?}")))?;
    Ok(BatchOutput::from_limbs(limbs, num_limbs))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fp::prime::ValidPrime;

    use super::*;
    use crate::algebra::milnor_batch::{
        COL_MAP_DROP, cpu_multiply_batch, cpu_multiply_batch_masked,
    };

    /// A deterministic pseudo-random stream. No `rand` dependency, and the same sequence on every
    /// machine -- a failing case has to be reproducible from the test name alone.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // SplitMix64.
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Build a batch of products spread over many `(R, s)` pairs, rows and output offsets.
    ///
    /// Deliberately NOT one product per row: the whole point of the output being XOR-accumulated is
    /// that several products land in the same row, and a bug that drops or double-counts a pair is
    /// invisible when every row has exactly one contributor.
    fn sample_products(
        algebra: &MilnorAlgebra,
        max_degree: i32,
        num_rows: usize,
        count: usize,
        seed: u64,
    ) -> Vec<GpuProduct> {
        let mut rng = Rng(seed);
        let mut products = Vec::new();
        while products.len() < count {
            let r_degree = 1 + rng.below(12) as i32;
            let r_dim = algebra.dimension(r_degree);
            if r_dim == 0 {
                continue;
            }
            let r_idx = rng.below(r_dim);
            if algebra
                .basis_element_from_index(r_degree, r_idx)
                .p_part
                .is_empty()
            {
                continue; // Sq(empty) = 1 is not a batch product
            }
            let s_degree = 1 + rng.below((max_degree - r_degree) as usize) as i32;
            let s_dim = algebra.dimension(s_degree);
            if s_dim == 0 || algebra.dimension(r_degree + s_degree) == 0 {
                continue;
            }
            // A random non-empty subset of the source basis, so terms cancel against each other.
            let mut term_indices: Vec<usize> = (0..s_dim).filter(|_| rng.next() % 2 == 0).collect();
            if term_indices.is_empty() {
                term_indices.push(rng.below(s_dim));
            }
            products.push(GpuProduct {
                r_degree,
                r_idx,
                s_degree,
                term_indices: term_indices.into(),
                row: rng.below(num_rows),
                out_offset: 0,
            });
        }
        products
    }

    /// Output width that every product's `out_offset + seqno` fits inside.
    fn full_width(algebra: &MilnorAlgebra, products: &[GpuProduct]) -> usize {
        products
            .iter()
            .map(|p| p.out_offset + algebra.dimension(p.r_degree + p.s_degree))
            .max()
            .unwrap_or(1)
            .max(1)
    }

    fn algebra_to(max_degree: i32) -> Arc<MilnorAlgebra> {
        let algebra = Arc::new(MilnorAlgebra::new(ValidPrime::new(2), false));
        algebra.compute_basis(max_degree);
        algebra.compute_seqno_tables(max_degree);
        algebra
    }

    /// THE PORT'S ACCEPTANCE TEST: the device batch must equal the CPU reference bit for bit.
    ///
    /// The CPU path owes nothing to any framework or card, so this pins the right answer
    /// permanently. Every optimisation re-introduced into the kernel has to keep it passing.
    #[test]
    #[ignore = "needs a CUDA device; run explicitly with --ignored"]
    fn cuda_batch_matches_cpu_reference() {
        let max_degree = 40;
        let algebra = algebra_to(max_degree);
        let num_rows = 24;
        let products = sample_products(&algebra, max_degree, num_rows, 400, 0x5eed);
        let num_cols = full_width(&algebra, &products);

        let want = cpu_multiply_batch(&algebra, num_cols, num_rows, &products);
        let rt = super::super::runtime(0).expect("open device 0");
        let got = cuda_multiply_batch(&rt, &algebra, num_cols, num_rows, &products, None)
            .expect("device batch");

        assert_eq!(got.num_limbs(), want.num_limbs(), "limb count");
        assert!(
            want.iter_rows().flatten().any(|&w| w != 0),
            "the reference is all zero, so the comparison proves nothing"
        );
        for (r, (a, b)) in got.iter_rows().zip(want.iter_rows()).enumerate() {
            assert_eq!(a, b, "row {r} differs from the CPU reference");
        }
    }

    /// The same, with the column restriction on: `out` is allocated at the MASKED width and the
    /// kernel applies `col_map` itself, which is what lets the frontier's ~98%-discarded columns
    /// never be allocated at all. A map that is off by one silently drops bits, so it is checked
    /// against the CPU's compute-wide-then-gather rather than against the unmasked device run.
    #[test]
    #[ignore = "needs a CUDA device; run explicitly with --ignored"]
    fn cuda_batch_matches_cpu_reference_masked() {
        let max_degree = 40;
        let algebra = algebra_to(max_degree);
        let num_rows = 16;
        let products = sample_products(&algebra, max_degree, num_rows, 200, 0xc0ffee);
        let full = full_width(&algebra, &products);

        // Keep every third column, exactly as a signature mask keeps a sparse subset.
        let mut col_map = vec![COL_MAP_DROP; full];
        let mut kept = 0u32;
        for (i, slot) in col_map.iter_mut().enumerate() {
            if i % 3 == 0 {
                *slot = kept;
                kept += 1;
            }
        }
        let out_cols = kept as usize;

        let want = cpu_multiply_batch_masked(
            &algebra,
            out_cols,
            Some(col_map.clone().into()),
            num_rows,
            &products,
        );
        let rt = super::super::runtime(0).expect("open device 0");
        let got = cuda_multiply_batch(&rt, &algebra, out_cols, num_rows, &products, Some(&col_map))
            .expect("device batch");

        assert_eq!(got.num_limbs(), want.num_limbs(), "limb count");
        assert!(
            want.iter_rows().flatten().any(|&w| w != 0),
            "the reference is all zero, so the comparison proves nothing"
        );
        for (r, (a, b)) in got.iter_rows().zip(want.iter_rows()).enumerate() {
            assert_eq!(a, b, "row {r} differs from the masked CPU reference");
        }
    }

    /// A CPU walk of EXACTLY what the kernel does: the same pair decode out of the same marshalled
    /// buffers, the same per-column rule, the same seqno, the same emit.
    ///
    /// This exists to split the port's two possible failure modes apart. If this disagrees with
    /// `cpu_multiply_batch`, the bug is in [`marshal`] -- an offset, a dedup key, a prefix sum -- and
    /// no GPU is needed to find it. If this agrees and the device does not, the bug is in the CUDA C
    /// or the launch. Debugging both at once against one red test is what makes a kernel port drag.
    ///
    /// It also runs WITHOUT a card, so a marshalling regression is caught by an ordinary
    /// `cargo test --features cuda` rather than sitting unnoticed behind an `#[ignore]`.
    fn simulate(m: &Marshalled, num_rows: usize, num_limbs: usize, col_map: Option<&[u32]>) -> Vec<u32> {
        let mut out = vec![0u32; num_rows * num_limbs];
        let total = m.total_pairs();
        let num_products = m.num_products();
        for k in 0..total {
            // Largest p with prod_pair_start[p] <= k.
            let (mut lo, mut hi) = (0usize, num_products);
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if m.prod_pair_start[mid] <= k {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let p = lo;
            let ri = m.prod_r_index[p] as usize;
            let local = k - m.prod_pair_start[p];
            let num_mats = m.r_num_mats[ri] as u64;
            let mi = (local % num_mats) as usize;
            let t = (local / num_mats) as usize;

            let cs_len = m.r_cs_len[ri] as usize;
            let mk_len = m.r_mk_len[ri] as usize;
            let cs_base = m.r_cs_offset[ri] as usize + mi * cs_len;
            let mk_base = m.r_mk_offset[ri] as usize + mi * mk_len;

            let gei = m.term_gei[m.prod_term_start[p] as usize + t] as usize;
            let term_len = m.ln[gei] as usize;
            let b_base = gei * m.width;

            let cols = cs_len.max(mk_len).max(term_len);
            let low = term_len.min(cs_len);
            let mut working = [0u32; params::PPART_MAX_LEN];
            let mut rejected = false;
            for j in 0..cols {
                let b = if j < term_len { m.pp[b_base + j] as u32 } else { 0 };
                let c = if j < cs_len { m.cs[cs_base + j] as u32 } else { 0 };
                let msk = if j < mk_len { m.mk[mk_base + j] as u32 } else { 0 };
                // The same uniform per-position rule as `pair_col` in multiply.cu.
                let val = if j < low {
                    if c > b || ((b - c) & msk) != 0 {
                        None
                    } else {
                        Some((b - c) | msk)
                    }
                } else if c > 0 || (b & msk) != 0 {
                    None
                } else {
                    Some(b | msk)
                };
                match val {
                    None => rejected = true,
                    Some(v) => {
                        if j < params::PPART_MAX_LEN {
                            working[j] = v;
                        }
                    }
                }
            }
            if rejected {
                continue;
            }

            // seqno_core.
            let mut cur_d = 0u32;
            for h in 0..params::PPART_MAX_LEN {
                cur_d += working[h] * m.xi[h];
            }
            let mut rank = 0u32;
            for hh in 1..params::PPART_MAX_LEN {
                let h = params::PPART_MAX_LEN - hh;
                let r = working[h];
                if r != 0 {
                    let below = cur_d - r * m.xi[h];
                    rank += m.g[cur_d as usize * m.width + h] - m.g[below as usize * m.width + h];
                    cur_d = below;
                }
            }

            let mut bit_pos = m.prod_out_offset[p] as usize + rank as usize;
            if let Some(map) = col_map {
                if bit_pos >= map.len() {
                    continue;
                }
                match map[bit_pos] {
                    COL_MAP_DROP => continue,
                    mapped => bit_pos = mapped as usize,
                }
            }
            let limb = bit_pos / 32;
            if limb >= num_limbs {
                continue;
            }
            let word = m.prod_row_base[p] as usize + limb;
            if word >= out.len() {
                continue;
            }
            out[word] ^= 1u32 << (bit_pos % 32);
        }
        out
    }

    /// The marshalled buffers, walked the kernel's way, reproduce the CPU reference.
    ///
    /// No device needed: this is the marshalling half of the port under test on its own.
    #[test]
    fn marshalled_walk_matches_cpu_reference() {
        let max_degree = 40;
        let algebra = algebra_to(max_degree);
        let num_rows = 24;
        let products = sample_products(&algebra, max_degree, num_rows, 400, 0x5eed);
        let num_cols = full_width(&algebra, &products);
        let num_limbs = num_cols.div_ceil(32).max(1);

        let m = marshal(&algebra, num_rows, num_limbs, &products).expect("marshal");
        let got = simulate(&m, num_rows, num_limbs, None);
        let want = cpu_multiply_batch(&algebra, num_cols, num_rows, &products);

        assert_eq!(got.len() / num_limbs, num_rows);
        // An all-zero result agreeing with an all-zero reference is the failure mode that cost this
        // project weeks: cubecl swallowed a failed allocation and returned a zeroed buffer at
        // exit 0. Every comparison here first insists the reference carries bits.
        assert!(
            want.iter_rows().flatten().any(|&w| w != 0),
            "the reference is all zero, so the comparison proves nothing"
        );
        for (r, want_row) in want.iter_rows().enumerate() {
            let got_row = &got[r * num_limbs..(r + 1) * num_limbs];
            assert_eq!(got_row, want_row, "row {r}: marshalled walk != CPU reference");
        }
    }

    /// The same for a restricted output, so an off-by-one in the `col_map` application is caught
    /// without a card too.
    #[test]
    fn marshalled_walk_matches_cpu_reference_masked() {
        let max_degree = 40;
        let algebra = algebra_to(max_degree);
        let num_rows = 16;
        let products = sample_products(&algebra, max_degree, num_rows, 200, 0xc0ffee);
        let full = full_width(&algebra, &products);

        let mut col_map = vec![COL_MAP_DROP; full];
        let mut kept = 0u32;
        for (i, slot) in col_map.iter_mut().enumerate() {
            if i % 3 == 0 {
                *slot = kept;
                kept += 1;
            }
        }
        let out_cols = kept as usize;
        let num_limbs = out_cols.div_ceil(32).max(1);

        let m = marshal(&algebra, num_rows, num_limbs, &products).expect("marshal");
        let got = simulate(&m, num_rows, num_limbs, Some(&col_map));
        let want = cpu_multiply_batch_masked(
            &algebra,
            out_cols,
            Some(col_map.clone().into()),
            num_rows,
            &products,
        );

        assert!(
            want.iter_rows().flatten().any(|&w| w != 0),
            "the masked reference is all zero, so the comparison proves nothing"
        );
        for (r, want_row) in want.iter_rows().enumerate() {
            let got_row = &got[r * num_limbs..(r + 1) * num_limbs];
            assert_eq!(got_row, want_row, "row {r}: masked walk != CPU reference");
        }
    }

}
