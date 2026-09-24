//! Host side of the batched Milnor multiply: marshal a `[GpuProduct]` into what
//! [`multiply.cu`](./multiply.cu) reads, launch, and read the limbs back.
//!
//! The bulk data -- the admissible-matrix master, the Milnor basis, the seqno tables -- lives in
//! [`super::resident`] and is uploaded once. What this module builds per launch is only the small
//! per-product bookkeeping: which `R` each product uses, where its terms are, where its output
//! lands, and the prefix sum that lets a thread decode its own pair. Everything here is
//! proportional to the PRODUCT COUNT, not to the master: a frontier batch of ~540k products is a
//! few MB of this against tens of GB of resident matrices, and that ratio is the whole reason the
//! master is resident.
//!
//! Still absent, and still deliberate: row batching, an un-awaited readback, and per-device
//! streams. Those make repeated launches overlap; none of them changes the answer, so they follow
//! with the digest already pinned.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{
    CudaError, MilnorCuda, Result, params,
    resident::{BasisLayout, MasterLayout, RInfo, r_tables, resident},
};
use crate::algebra::{
    Algebra, MilnorAlgebra,
    milnor_algebra::PPart,
    milnor_batch::{BatchOutput, GpuProduct, LimbBlock},
};

/// The kernel source, compiled at runtime by NVRTC.
const SRC: &str = include_str!("multiply.cu");

/// Module cache key. Must change whenever the defines do -- which they do only when `params.rs`
/// changes -- so the constants are folded into the key rather than trusted to stay put.
fn module_key() -> String {
    let mut key = String::from("multiply_batch");
    for (name, value) in params::defines() {
        key.push_str(&format!(":{name}={value}"));
    }
    key
}

/// Enumerate admissible matrices on the CPU instead of on the device.
///
/// `NASSAU_CUDA_CPU_ENUM=1`. Off by default -- host enumeration is the cost the enum kernel exists
/// to remove -- but kept reachable, because the two paths must produce identical masters and being
/// able to flip between them in one process is how that gets checked on real input rather than on
/// a test fixture.
fn cpu_enumeration() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NASSAU_CUDA_CPU_ENUM").is_ok_and(|v| v != "0" && !v.is_empty())
    })
}

/// The distinct `R`s of a batch, in first-seen order, plus the map from a product's key to that
/// `R`'s position in the list.
///
/// ONE HASH PASS OVER THE PRODUCTS, not two. Making the `R`s resident and marshalling the arrays
/// both need to know which `R` a product uses, and each was building its own map -- 264,646
/// lookups each on the profile batch, on a launch where the store is already warm and the answer
/// is the same both times. The position in this list is also the launch-local `R` index the kernel
/// indexes `r_*` with, so the second map had nothing extra to compute.
///
/// Keyed by `(r_degree, r_idx)` rather than the p-part: the two identify the same thing, and this
/// one hashes two words instead of ten. The p-part is materialised only on a miss.
pub(super) fn plan_rs(algebra: &MilnorAlgebra, products: &[GpuProduct]) -> (Vec<PPart>, Vec<u32>) {
    let mut order: Vec<PPart> = Vec::new();
    // FxHashMap, not the std default. The key is two words and the map is consulted once per
    // product -- 426k times on the profile batch -- so SipHash's per-key setup dominates a lookup
    // that decides almost nothing. rustc-hash is already a dependency of this crate.
    let mut local: FxHashMap<(i32, usize), u32> = FxHashMap::default();
    // The products of one extract loop arrive PARTLY grouped by `R`, so the previous answer is
    // often the next one and a two-word compare in front of the map skips the hash on a hit.
    // Measured worth 9.3ms -> 8.3ms of the resident phase, i.e. a useful minority of lookups --
    // not the overwhelming majority the "grouped" reading would predict.
    let mut last: Option<((i32, usize), u32)> = None;
    // POSITIONAL, one entry per product, `SKIP` where the product contributes nothing. Returning
    // the map instead would make `marshal` hash all 426k products a second time to ask a question
    // already answered here -- and re-evaluate the skip condition, which costs an `algebra
    // .dimension` call of its own. A 1.7 MB vector buys both away.
    let mut per_product = vec![SKIP; products.len()];
    for (slot, prod) in per_product.iter_mut().zip(products) {
        // A product whose output degree is empty contributes nothing; the CPU reference skips it
        // and so must this, or its `R` would be enumerated for no reason.
        if algebra.dimension(prod.r_degree + prod.s_degree) == 0 || prod.term_indices.is_empty() {
            continue;
        }
        let key = (prod.r_degree, prod.r_idx);
        if let Some((k, ri)) = last {
            if k == key {
                *slot = ri;
                continue;
            }
        }
        let ri = *local.entry(key).or_insert_with(|| {
            let ri = order.len() as u32;
            order.push(
                algebra
                    .basis_element_from_index(prod.r_degree, prod.r_idx)
                    .p_part,
            );
            ri
        });
        *slot = ri;
        last = Some((key, ri));
    }
    (order, per_product)
}

/// Marks a product that contributes nothing, in the positional index [`plan_rs`] returns.
///
/// `u32::MAX` is safe as a sentinel: it would need that many distinct `R`s in ONE batch, against a
/// few thousand in practice, and the launch-local `R` count is bounded by the product count.
pub(super) const SKIP: u32 = u32::MAX;

/// The per-launch arrays, in device layout.
pub(super) struct LaunchArrays {
    /// Per distinct `R` in THIS launch, pointing into the resident master.
    r_cs_offset: Vec<u64>,
    r_mk_offset: Vec<u64>,
    r_cs_len: Vec<u32>,
    r_mk_len: Vec<u32>,
    r_num_mats: Vec<u32>,
    /// Term global basis indices, run-concatenated per product.
    term_gei: Vec<u32>,
    prod_r_index: Vec<u32>,
    prod_term_start: Vec<u32>,
    prod_num_terms: Vec<u32>,
    prod_row_base: Vec<u32>,
    prod_out_offset: Vec<u32>,
    /// Prefix sum of pair counts, length `num_products + 1`.
    prod_pair_start: Vec<u64>,
    /// Coarse index over the pair space; see `build_coarse`.
    prod_coarse: Vec<u32>,
}

impl LaunchArrays {
    fn num_products(&self) -> usize {
        self.prod_r_index.len()
    }

    /// Total THREADS this launch needs, i.e. tiles, not pairs.
    fn total_pairs(&self) -> u64 {
        *self.prod_pair_start.last().unwrap_or(&0)
    }
}

/// Build the per-launch arrays.
///
/// `r_of` resolves an `R`'s p-part to its place in the master, appending it if it is new; `gei_of`
/// maps a term to its resident basis index. Both are passed IN rather than taken from a `Resident`
/// so the device-free walk can drive this exact function against host-side layouts. The offsets and
/// the prefix sum are the part most worth testing, and testing a reimplementation of them would
/// prove nothing.
pub(super) fn marshal(
    algebra: &MilnorAlgebra,
    num_limbs: usize,
    products: &[GpuProduct],
    r_index: &[u32],
    r_infos: &[RInfo],
    // Generic, not `&dyn Fn`: this used to be a virtual call PER TERM, 4.15M of them on the
    // profile batch. Monomorphised it inlines to a vector index and an add.
    gei_of: impl Fn(i32, usize) -> u32,
    // First row of the block this launch covers. Output rows are numbered from it, so a block's
    // buffer is only as tall as the block.
    row0: usize,
) -> Result<LaunchArrays> {
    // Sized up front. These are per-product arrays over a batch with hundreds of thousands of
    // products, so growing them from empty is a run of reallocations and memcpys on the critical
    // path of a phase that already dominates the launch.
    let n = products.len();
    let terms: usize = products.iter().map(|p| p.term_indices.len()).sum();
    let mut pps = Vec::with_capacity(n + 1);
    pps.push(0);
    // The per-`R` tables are just `r_infos` transposed -- every distinct `R` of this batch is
    // already known and in kernel index order, so there is nothing to discover per product.
    let mut a = LaunchArrays {
        r_cs_offset: r_infos.iter().map(|i| i.cs_offset).collect(),
        r_mk_offset: r_infos.iter().map(|i| i.mk_offset).collect(),
        r_cs_len: r_infos.iter().map(|i| i.cs_len).collect(),
        r_mk_len: r_infos.iter().map(|i| i.mk_len).collect(),
        r_num_mats: r_infos.iter().map(|i| i.num_mats).collect(),
        term_gei: Vec::with_capacity(terms),
        prod_r_index: Vec::with_capacity(n),
        prod_term_start: Vec::with_capacity(n),
        prod_num_terms: Vec::with_capacity(n),
        prod_row_base: Vec::with_capacity(n),
        prod_out_offset: Vec::with_capacity(n),
        prod_pair_start: pps,
        prod_coarse: Vec::new(),
    };

    for (&ri, prod) in r_index.iter().zip(products) {
        // The skip decision and the launch-local `R` index were both settled by `plan_rs`; asking
        // again here would mean a second hash of every product and a second `dimension` call.
        if ri == SKIP {
            continue;
        }

        a.prod_term_start.push(a.term_gei.len() as u32);
        a.prod_num_terms.push(prod.term_indices.len() as u32);
        // The degree's base is the same for every term of a product, so resolve it ONCE per
        // product rather than once per term -- `term_gei` is the longest array built here
        // (4.15M entries on the profile batch) and this is its inner loop.
        let base = gei_of(prod.s_degree, 0);
        a.term_gei
            .extend(prod.term_indices.iter().map(|&ti| base + ti as u32));
        a.prod_r_index.push(ri);
        a.prod_row_base.push(((prod.row - row0) * num_limbs) as u32);
        a.prod_out_offset.push(prod.out_offset as u32);
        // TILES, not pairs: a thread covers MATRIX_GROUP x TERM_GROUP of them, so this prefix sum
        // -- and therefore the coarse index built from it -- is over threads. The ragged edge of a
        // partial tile is dropped inside the kernel, not here.
        let mg = (a.r_num_mats[ri as usize] as u64).div_ceil(params::MATRIX_GROUP as u64);
        let tg = (prod.term_indices.len() as u64).div_ceil(params::TERM_GROUP as u64);
        a.prod_pair_start
            .push(a.prod_pair_start.last().unwrap() + mg * tg);
    }
    a.prod_coarse = build_coarse(&a.prod_pair_start);
    Ok(a)
}

/// Bucket the pair space so the kernel's product search starts from a narrow bracket.
///
/// `coarse[ci]` is the largest product whose pair range starts at or before `ci << COARSE_LOG`, so
/// the owner of any pair in that bucket lies in `[coarse[ci], coarse[ci + 1]]`. Built in one pass
/// over the products rather than by searching per bucket.
///
/// Length is one past the last bucket, because the kernel reads `ci + 1` unconditionally -- a
/// thread in the final bucket would otherwise read off the end, which is an out-of-bounds load that
/// yields a plausible bracket and therefore a plausible wrong answer.
fn build_coarse(prod_pair_start: &[u64]) -> Vec<u32> {
    let num_products = prod_pair_start.len().saturating_sub(1);
    if num_products == 0 {
        return vec![0, 0];
    }
    let total = *prod_pair_start.last().unwrap();
    let buckets = (total >> params::COARSE_LOG) as usize + 2;
    let mut coarse = vec![0u32; buckets];
    let mut p = 0usize;
    for (ci, slot) in coarse.iter_mut().enumerate() {
        let target = (ci as u64) << params::COARSE_LOG;
        while p + 1 < num_products && prod_pair_start[p + 1] <= target {
            p += 1;
        }
        *slot = p as u32;
    }
    coarse
}

/// Where a launch's wall time actually went.
///
/// THIS EXISTS BECAUSE WALL TIME LIES ABOUT KERNELS. The first kernel optimisation re-introduced
/// into this port -- the coarse index on the product search, worth ~12% of cubecl's KERNEL time --
/// moved the end-to-end warm figure by 0.000s, because the kernel is a minority of it. This
/// codebase has a harness on record where 87% of the measured wall time was overhead and every
/// kernel change therefore measured as 0%; attributing that to "the optimisation does not work"
/// would be exactly the wrong conclusion.
///
/// Each phase is separated by a stream SYNCHRONISE, so the numbers add up to the wall time and
/// none of them hides behind another. That serialisation is the cost of being able to attribute,
/// and it is why the timed entry point is separate from the plain one.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaunchTiming {
    /// Making the batch's `R`s, basis degrees and seqno tables resident, including enumerating
    /// the new `R`s on the device. Separated from `marshal` because the two have completely
    /// different fixes: this one shrinks as the store warms, that one is per-product host work
    /// that never goes away.
    pub resident: f64,
    /// Building the per-launch arrays on the host.
    pub marshal: f64,
    /// Host-to-device of those arrays plus zeroing the output.
    pub upload: f64,
    /// The multiply kernel itself.
    pub kernel: f64,
    /// Device-to-host of the limbs.
    pub readback: f64,
}

impl LaunchTiming {
    pub fn total(&self) -> f64 {
        self.resident + self.marshal + self.upload + self.kernel + self.readback
    }
}

/// Output bytes a single launch may allocate, before it is split into row blocks.
///
/// The cudarc path used to launch every batch whole, which is fine until it is not: the output is
/// `num_rows * num_limbs * 4` bytes and both factors grow with the frontier, so an unbounded launch
/// eventually asks the device for more than it has. cubecl splits for exactly this reason, and part
/// of its per-launch cost buys that safety.
///
/// The default is deliberately GENEROUS (1 GiB): a 426k-product stem-170 batch needs 338 MB and
/// stays whole, so the bound costs nothing until it is actually needed.
fn block_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("NASSAU_CUDA_BLOCK_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024)
            << 20
    })
}

/// Split `products` into contiguous row blocks whose outputs each fit `block_bytes`.
///
/// Returns `(row0, row1, p0, p1)` per block. Rows are INDEPENDENT -- every product writes only its
/// own row -- so concatenating the blocks' outputs in row order reproduces the single-launch result
/// exactly. That is what lets `BatchOutput` hold one landing buffer per block instead of joining
/// them.
///
/// A single row whose output exceeds the budget becomes a block of its own and overshoots: rows
/// cannot be split without splitting the column space too, which is a different (and much larger)
/// change. Saying so is better than silently pretending the bound holds.
fn row_blocks(
    products: &[GpuProduct],
    num_rows: usize,
    num_limbs: usize,
    budget_bytes: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let row_bytes = num_limbs * size_of::<u32>();
    let max_rows = (budget_bytes / row_bytes.max(1)).max(1);
    if num_rows <= max_rows {
        return vec![(0, num_rows, 0, products.len())];
    }
    let mut blocks = Vec::new();
    let (mut row0, mut p0) = (0usize, 0usize);
    for (i, prod) in products.iter().enumerate() {
        if prod.row >= row0 + max_rows {
            // Close at this product: everything before it belongs to rows < prod.row.
            blocks.push((row0, prod.row, p0, i));
            row0 = prod.row;
            p0 = i;
        }
    }
    blocks.push((row0, num_rows, p0, products.len()));
    blocks
}

/// Run a batch on the device and return the limbs, matching [`cpu_multiply_batch`] bit for bit.
///
/// `col_map` restricts the output to the masked columns exactly as `cpu_multiply_batch_masked`
/// does: `num_cols` is then the MASKED width and `col_map.len()` the full one.
///
/// The algebra must have its basis and seqno tables built through every product's output degree.
///
/// [`cpu_multiply_batch`]: crate::algebra::milnor_batch::cpu_multiply_batch
pub fn cuda_multiply_batch(
    rt: &Arc<MilnorCuda>,
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
    col_map: Option<&[u32]>,
) -> Result<BatchOutput> {
    run(rt, algebra, num_cols, num_rows, products, col_map, false).map(|(o, _)| o)
}

/// [`cuda_multiply_batch`], also reporting where the time went.
///
/// Separate entry point because the attribution costs a stream synchronise per phase. Production
/// calls the plain one.
pub fn cuda_multiply_batch_timed(
    rt: &Arc<MilnorCuda>,
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
    col_map: Option<&[u32]>,
) -> Result<(BatchOutput, LaunchTiming)> {
    run(rt, algebra, num_cols, num_rows, products, col_map, true)
}

/// The launch, with phase attribution optional.
///
/// `timed` controls the intermediate stream synchronises. They exist ONLY so the phases add up:
/// everything here runs on one stream and is therefore already ordered, so the sole synchronise
/// production needs is the one before the readback buffer is handed to the caller. Leaving the
/// others in would make every production launch pay for instrumentation, and would serialise
/// upload against the previous block's kernel in a multi-block launch -- the exact overlap row
/// batching creates the opportunity for.
#[allow(clippy::too_many_arguments)]
fn run(
    rt: &Arc<MilnorCuda>,
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
    col_map: Option<&[u32]>,
    timed: bool,
) -> Result<(BatchOutput, LaunchTiming)> {
    use std::time::Instant;
    let mut timing = LaunchTiming::default();
    let t_resident = Instant::now();
    let num_limbs = num_cols.div_ceil(32).max(1);
    let out_len = num_rows * num_limbs;

    let max_out_degree = products
        .iter()
        .map(|p| p.r_degree + p.s_degree)
        .max()
        .unwrap_or(0)
        .max(1);
    let max_s_degree = products.iter().map(|p| p.s_degree).max().unwrap_or(0);

    // OUTSIDE THE LOCK. One pass over the products decides both which `R`s to make resident and
    // what the kernel's launch-local `R` index is for each product -- and it needs nothing from
    // the store, so there is no reason for other threads to wait through it.
    let (needed, r_index) = plan_rs(algebra, products);

    // THE CRITICAL SECTION IS ONLY WHAT MUTATES THE STORE, plus the snapshot taken out of it.
    //
    // Everything after this -- marshalling, the launches, the readback -- runs on locals. That is
    // sound because the store is APPEND-ONLY: its device pointers are stable for the life of the
    // process, and a concurrent append can add data but can never move or free what this launch
    // has already resolved.
    //
    // Holding the lock across the whole launch instead measured a 0.95x "speedup" for four
    // concurrent threads: full serialisation, plus the cost of contending for it. That matters
    // more than any single launch's cost, because a frontier run has ~18 threads inside GPU calls
    // at once.
    let (r_infos, bases, p_cs, p_mk, p_pp, p_ln, p_g, p_xi, width) = {
        let store = resident(rt)?;
        let t_lock = Instant::now();
        let mut store = store.lock().unwrap();
        if std::env::var_os("NASSAU_CUDA_LOCK_INFO").is_some() {
            eprintln!("[lock] waited {:.4}s", t_lock.elapsed().as_secs_f64());
        }
        // `ensure_seqno` FIRST: it establishes `width`, the stride `ensure_basis` pads to.
        store.ensure_seqno(algebra, max_out_degree)?;
        store.ensure_basis(algebra, max_s_degree)?;
        // Every `R` this batch needs, made resident in ONE enumeration launch. Per-`R` on demand
        // would be one launch each, and the enum kernel's duration is set by its longest single
        // `R` rather than by how many it carries -- so batching turns a sum into a max.
        let r_infos = if cpu_enumeration() {
            needed
                .iter()
                .map(|p| store.ensure_r(algebra, p))
                .collect::<Result<Vec<_>>>()?
        } else {
            store.ensure_rs(rt, &needed)?
        };
        // The basis is fully built above, so snapshotting each degree's base here is exact.
        let bases: Vec<u32> = (0..=max_s_degree.max(0))
            .map(|d| store.basis().gei(d, 0))
            .collect();
        (
            r_infos,
            bases,
            store.cs_ptr(),
            store.mk_ptr(),
            store.pp_ptr(),
            store.ln_ptr(),
            store.g_ptr(),
            store.xi_ptr(),
            store.width() as u32,
        )
    };

    timing.resident = t_resident.elapsed().as_secs_f64();
    let t_marshal = Instant::now();

    // Shared by every block: the compiled module, the snapshotted resident pointers and the column
    // map. Only the per-product arrays and the output buffer are per block.
    let gei_of = move |d: i32, ti: usize| bases[d as usize] + ti as u32;

    let module = rt.module(&module_key(), SRC, &params::defines())?;
    let f = module
        .load_function("multiply_batch")
        .map_err(|e| CudaError::Compile(format!("load multiply_batch: {e:?}")))?;
    // `NASSAU_CUDA_KERNEL_INFO=1`: registers, local memory and the block ceiling ptxas settled on.
    //
    // Worth having rather than inferring. The block-size sweep showed a CLIFF between 128 and 160
    // threads -- flat at ~64.5e9 pairs/s to 128, ~61 above -- which is the shape of an occupancy
    // boundary, and the register count is what decides where that boundary sits. Guessing at it is
    // how effort gets spent shrinking state that was never the limit.
    if std::env::var_os("NASSAU_CUDA_KERNEL_INFO").is_some() {
        use cudarc::driver::sys::CUfunction_attribute_enum as A;
        let get = |a| f.get_attribute(a).unwrap_or(-1);
        eprintln!(
            "[kernel] multiply_batch: regs/thread={} local={}B shared={}B max_threads/block={}",
            get(A::CU_FUNC_ATTRIBUTE_NUM_REGS),
            get(A::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
            get(A::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
            get(A::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        );
    }
    let stream = rt.context().default_stream();

    macro_rules! up {
        ($v:expr) => {
            stream
                .memcpy_stod(&$v)
                .map_err(|e| CudaError::Compile(format!("upload {}: {e:?}", stringify!($v))))?
        };
    }
    // The argument is not optional, so an unrestricted launch binds a one-element dummy the kernel
    // never reads. Uploaded ONCE: the map is indexed by full output column, which no block changes.
    let cm: Vec<u32> = col_map.map_or_else(|| vec![0u32], <[u32]>::to_vec);
    let d_cm = up!(cm);

    // Scalars need bindings: `arg` borrows, so a temporary would be dropped before the launch.
    let col_map_len = col_map.map_or(0u32, |c| c.len() as u32);
    let use_col_map = u32::from(col_map.is_some());
    let num_limbs_u = num_limbs as u32;
    let threads = params::THREADS as u32;
    // Walk the pair space in pieces that each fit the 32-bit thread index, so one oversized row
    // cannot overflow the grid.
    let chunk = (u32::MAX as u64 / threads as u64) * threads as u64;

    let plan = row_blocks(products, num_rows, num_limbs, block_bytes());
    let mut out_blocks: Vec<Box<dyn LimbBlock>> = Vec::with_capacity(plan.len());

    for &(row0, row1, p0, p1) in &plan {
        let t_marshal = Instant::now();
        let arrays = marshal(
            algebra,
            num_limbs,
            &products[p0..p1],
            &r_index[p0..p1],
            &r_infos,
            &gei_of,
            row0,
        )?;
        timing.marshal += t_marshal.elapsed().as_secs_f64();

        let blk_len = (row1 - row0) * num_limbs;
        if arrays.total_pairs() == 0 {
            // The rows still exist and still have to appear in the output, they are just empty.
            out_blocks.push(Box::new(vec![0u32; blk_len]));
            continue;
        }

        let t_upload = Instant::now();
        let d_tg = up!(arrays.term_gei);
        let d_rco = up!(arrays.r_cs_offset);
        let d_rmo = up!(arrays.r_mk_offset);
        let d_rcl = up!(arrays.r_cs_len);
        let d_rml = up!(arrays.r_mk_len);
        let d_rnm = up!(arrays.r_num_mats);
        let d_pri = up!(arrays.prod_r_index);
        let d_pts = up!(arrays.prod_term_start);
        let d_pnt = up!(arrays.prod_num_terms);
        let d_prb = up!(arrays.prod_row_base);
        let d_poo = up!(arrays.prod_out_offset);
        let d_pps = up!(arrays.prod_pair_start);
        let d_pc = up!(arrays.prod_coarse);
        let mut d_out = stream
            .alloc_zeros::<u32>(blk_len)
            .map_err(|e| CudaError::Compile(format!("alloc out: {e:?}")))?;
        if timed {
            stream
                .synchronize()
                .map_err(|e| CudaError::Compile(format!("sync after upload: {e:?}")))?;
            timing.upload += t_upload.elapsed().as_secs_f64();
        }

        let t_kernel = Instant::now();
        let num_products = arrays.num_products() as u32;
        let out_len_u = blk_len as u64;
        let total = arrays.total_pairs();
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
            b.arg(&p_cs)
                .arg(&p_mk)
                .arg(&p_pp)
                .arg(&p_ln)
                .arg(&d_tg)
                .arg(&p_g)
                .arg(&p_xi)
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
                .arg(&d_pc)
                .arg(&num_products)
                .arg(&pair_offset)
                .arg(&width)
                .arg(&num_limbs_u)
                .arg(&out_len_u);
            unsafe { b.launch(cfg) }
                .map_err(|e| CudaError::Compile(format!("launch multiply_batch: {e:?}")))?;
            done += n;
        }
        if timed {
            stream
                .synchronize()
                .map_err(|e| CudaError::Compile(format!("sync after kernel: {e:?}")))?;
            timing.kernel += t_kernel.elapsed().as_secs_f64();
        }

        // Land the readback in CACHED page-locked memory and hand that buffer straight to the
        // caller. A device-to-host copy into pageable memory is staged by the driver through its
        // own bounce buffer; into page-locked memory it is a direct DMA. And because `BatchOutput`
        // stores `Box<dyn LimbBlock>`, the landing buffer IS the result -- no second copy into a
        // `Vec`, which at frontier sizes would be gigabytes of memcpy for nothing.
        //
        // From the POOL: page-locking is charged per allocation and dominated this phase outright
        // (69.4 ms of 76.8 ms for 338 MB), while the transfer it enables runs at 45.7 GB/s.
        let t_readback = Instant::now();
        let mut pinned = super::pinned_pool(rt.device()).take(rt.context(), blk_len)?;
        stream
            .memcpy_dtoh(&d_out, pinned.as_mut_slice())
            .map_err(|e| CudaError::Compile(format!("readback: {e:?}")))?;
        // ALWAYS: the caller is about to read this buffer.
        stream
            .synchronize()
            .map_err(|e| CudaError::Compile(format!("sync after readback: {e:?}")))?;
        timing.readback += t_readback.elapsed().as_secs_f64();
        out_blocks.push(Box::new(pinned));
    }

    Ok((BatchOutput::from_blocks(out_blocks, num_limbs), timing))
}

/// Device memory currently COMMITTED by the resident store, for diagnostics.
///
/// Committed, not reserved: the reservation is address space, which is free, and reporting it would
/// make an idle process look like it is holding 100+ GB.
pub fn resident_committed_bytes(rt: &Arc<MilnorCuda>) -> Result<usize> {
    Ok(resident(rt)?.lock().unwrap().committed_bytes())
}

#[cfg(test)]
mod tests {
    use fp::prime::ValidPrime;

    use super::{
        super::resident::{basis_tables, xi_table},
        *,
    };
    use crate::algebra::milnor_batch::{
        COL_MAP_DROP, cpu_multiply_batch, cpu_multiply_batch_masked, load_captured_batch,
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

    /// Build a batch of products spread over many `(R, s)` pairs and rows.
    ///
    /// Deliberately NOT one product per row: the output is XOR-accumulated, so several products
    /// landing in the same row is the interesting case, and a bug that drops or double-counts a
    /// pair is invisible when every row has exactly one contributor.
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

    /// A host-only mirror of the resident store.
    ///
    /// Uses the SAME [`MasterLayout`] / [`BasisLayout`] / [`r_tables`] / [`basis_tables`] the
    /// device path uses, so the offsets it produces are the offsets the device would produce. Only
    /// the upload is replaced by a `Vec` append. That is what makes the walk below a real test of
    /// the bookkeeping rather than a test of a second implementation of it.
    struct HostStore {
        cs: Vec<u16>,
        mk: Vec<u16>,
        pp: Vec<u64>,
        ln: Vec<u32>,
        g: Vec<u32>,
        xi: Vec<u32>,
        pp_shift: Vec<u32>,
        pp_mask: Vec<u32>,
        width: usize,
        master: MasterLayout,
        basis: BasisLayout,
    }

    impl HostStore {
        fn new(algebra: &MilnorAlgebra, max_s_degree: i32) -> Self {
            let (width, g) = algebra.seqno_table_u32();
            let mut basis = BasisLayout::default();
            let (pp, ln, counts) = basis_tables(algebra, 0, max_s_degree).expect("basis tables");
            basis.extend(&counts);
            Self {
                cs: Vec::new(),
                mk: Vec::new(),
                pp,
                ln,
                g,
                xi: xi_table(algebra),
                pp_shift: params::pp_shifts(),
                pp_mask: params::pp_masks(),
                width,
                master: MasterLayout::default(),
                basis,
            }
        }

        fn ensure_r(&mut self, algebra: &MilnorAlgebra, p: &PPart) -> Result<RInfo> {
            if let Some(info) = self.master.get(p) {
                return Ok(info);
            }
            let (cs_len, mk_len, num_mats, cs_u, mk_u) = r_tables(algebra, p)?;
            self.cs.extend_from_slice(&cs_u);
            self.mk.extend_from_slice(&mk_u);
            Ok(self
                .master
                .place(p, cs_len, mk_len, num_mats, cs_u.len(), mk_u.len()))
        }
    }

    /// A CPU walk of EXACTLY what the kernel does: the same pair decode out of the same marshalled
    /// arrays and the same store layout, the same per-column rule, the same seqno, the same emit.
    ///
    /// This splits the port's two failure modes apart. If this disagrees with `cpu_multiply_batch`,
    /// the bug is in the marshalling or the store layout -- an offset, a `gei`, a prefix sum -- and
    /// no GPU is needed to find it. If this agrees and the device does not, the bug is in the CUDA
    /// C or the launch. Debugging both at once against one red test is what makes a port drag.
    ///
    /// It also runs WITHOUT a card, so a marshalling regression is caught by an ordinary
    /// `cargo test --features cuda` rather than sitting unnoticed behind an `#[ignore]`.
    ///
    /// WHAT IT DOES NOT MODEL: the kernel's loop SEGMENTATION. The column loop is split at
    /// `COL_SPLIT_32` into a 32-bit and a 64-bit accumulate, and this walks one uniform loop
    /// instead. The two compute the same thing by construction -- that is the split's whole
    /// premise -- but it means a bug in the segmentation shows up only on a device, so the
    /// `#[ignore]`d tests are load-bearing for that and not merely a faster check.
    fn simulate(
        s: &HostStore,
        a: &LaunchArrays,
        num_rows: usize,
        num_limbs: usize,
        col_map: Option<&[u32]>,
    ) -> Vec<u32> {
        let mut out = vec![0u32; num_rows * num_limbs];
        let total = a.total_pairs();
        let num_products = a.num_products();
        for k in 0..total {
            // Largest p with prod_pair_start[p] <= k, from the SAME coarse bracket the kernel
            // uses. Walking it unbracketed here would leave `build_coarse` untested, and a wrong
            // bracket does not crash -- it silently attributes a pair to the wrong product.
            let ci = (k >> params::COARSE_LOG) as usize;
            let mut lo = a.prod_coarse[ci] as usize;
            let mut hi = (a.prod_coarse[ci + 1] as usize + 1).min(num_products);
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if a.prod_pair_start[mid] <= k {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let p = lo;
            let ri = a.prod_r_index[p] as usize;
            let local = k - a.prod_pair_start[p];
            let num_mats = a.r_num_mats[ri] as usize;
            let nt = a.prod_num_terms[p] as usize;
            // The kernel's tile decode, reproduced exactly: matrix fastest, ragged lanes dropped.
            let mg_count = num_mats.div_ceil(params::MATRIX_GROUP) as u64;
            let m_base = (local % mg_count) as usize * params::MATRIX_GROUP;
            let t_base = (local / mg_count) as usize * params::TERM_GROUP;

            let cs_len = a.r_cs_len[ri] as usize;
            let mk_len = a.r_mk_len[ri] as usize;

            for (mm, tt) in (0..params::MATRIX_GROUP)
                .flat_map(|mm| (0..params::TERM_GROUP).map(move |tt| (mm, tt)))
            {
                if m_base + mm >= num_mats || t_base + tt >= nt {
                    continue;
                }
                let mi = m_base + mm;
                let cs_base = a.r_cs_offset[ri] as usize + mi * cs_len;
                let mk_base = a.r_mk_offset[ri] as usize + mi * mk_len;

                let gei = a.term_gei[a.prod_term_start[p] as usize + t_base + tt] as usize;
                let term_len = s.ln[gei] as usize;
                // The whole p-part in one word, exactly as the kernel holds it.
                let b_bits = s.pp[gei];

                // Capped at PPART_MAX_LEN: past it both `b` and `cs` are zero, so the column can
                // neither reject nor accumulate. See the kernel for the bound that makes this
                // exact rather than a truncation.
                let cols = cs_len.max(mk_len).max(term_len).min(params::PPART_MAX_LEN);
                let low = term_len.min(cs_len);
                // Packed exactly as the kernel packs it, so the walk keeps testing what the kernel
                // does rather than an equivalent-but-different assembly.
                let mut working: u64 = 0;
                let mut rejected = false;
                #[allow(clippy::needless_range_loop)]
                for j in 0..cols {
                    // No load: the digit comes out of the packed word, exactly as the kernel
                    // takes it out of a register.
                    let b = if j < params::PPART_MAX_LEN {
                        ((b_bits >> s.pp_shift[j]) & u64::from(s.pp_mask[j])) as u32
                    } else {
                        0
                    };
                    let c = if j < cs_len {
                        s.cs[cs_base + j] as u32
                    } else {
                        0
                    };
                    let msk = if j < mk_len {
                        s.mk[mk_base + j] as u32
                    } else {
                        0
                    };
                    // The TWO-ARMED reference rule, deliberately not the collapsed one the
                    // kernel uses. The kernel drops the `j < low` split on the argument that the
                    // arms coincide past `low`; keeping the original form here means the digest
                    // comparison is a check OF that argument rather than a restatement of it.
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
                                working |= u64::from(v & s.pp_mask[j]) << s.pp_shift[j];
                            }
                        }
                    }
                }
                if rejected {
                    continue;
                }

                // seqno_core.
                let digit =
                    |h: usize| ((working >> s.pp_shift[h]) & u64::from(s.pp_mask[h])) as u32;
                let mut cur_d = 0u32;
                for h in 0..params::PPART_MAX_LEN {
                    cur_d += digit(h) * s.xi[h];
                }
                let mut rank = 0u32;
                for hh in 1..params::PPART_MAX_LEN {
                    let h = params::PPART_MAX_LEN - hh;
                    let r = digit(h);
                    if r != 0 {
                        let below = cur_d - r * s.xi[h];
                        rank +=
                            s.g[cur_d as usize * s.width + h] - s.g[below as usize * s.width + h];
                        cur_d = below;
                    }
                }

                let mut bit_pos = a.prod_out_offset[p] as usize + rank as usize;
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
                let word = a.prod_row_base[p] as usize + limb;
                if word >= out.len() {
                    continue;
                }
                out[word] ^= 1u32 << (bit_pos % 32);
            }
        }
        out
    }

    /// Drive `marshal` against a host store and walk the result.
    fn walk(
        algebra: &MilnorAlgebra,
        products: &[GpuProduct],
        num_rows: usize,
        num_limbs: usize,
        col_map: Option<&[u32]>,
    ) -> Vec<u32> {
        let max_s_degree = products.iter().map(|p| p.s_degree).max().unwrap_or(0);
        let mut store = HostStore::new(algebra, max_s_degree);
        let (needed, r_index) = plan_rs(algebra, products);
        let r_infos: Vec<RInfo> = needed
            .iter()
            .map(|p| store.ensure_r(algebra, p).expect("host tables"))
            .collect();
        let arrays = {
            let bases: Vec<u32> = (0..=max_s_degree.max(0))
                .map(|d| store.basis.gei(d, 0))
                .collect();
            let gei_of = move |d: i32, ti: usize| bases[d as usize] + ti as u32;
            marshal(algebra, num_limbs, products, &r_index, &r_infos, &gei_of, 0).expect("marshal")
        };
        simulate(&store, &arrays, num_rows, num_limbs, col_map)
    }

    /// Drive `marshal` and the walk BLOCK BY BLOCK, as the device path does.
    fn walk_blocked(
        algebra: &MilnorAlgebra,
        products: &[GpuProduct],
        num_rows: usize,
        num_limbs: usize,
        rows_per_block: usize,
    ) -> Vec<u32> {
        let max_s_degree = products.iter().map(|p| p.s_degree).max().unwrap_or(0);
        let mut store = HostStore::new(algebra, max_s_degree);
        let (needed, r_index) = plan_rs(algebra, products);
        let r_infos: Vec<RInfo> = needed
            .iter()
            .map(|p| store.ensure_r(algebra, p).expect("host tables"))
            .collect();
        let bases: Vec<u32> = (0..=max_s_degree.max(0))
            .map(|d| store.basis.gei(d, 0))
            .collect();
        let gei_of = move |d: i32, ti: usize| bases[d as usize] + ti as u32;

        let plan = row_blocks(
            products,
            num_rows,
            num_limbs,
            rows_per_block * num_limbs * size_of::<u32>(),
        );
        let mut out = Vec::with_capacity(num_rows * num_limbs);
        for &(row0, row1, p0, p1) in &plan {
            let arrays = marshal(
                algebra,
                num_limbs,
                &products[p0..p1],
                &r_index[p0..p1],
                &r_infos,
                &gei_of,
                row0,
            )
            .expect("marshal");
            out.extend(simulate(&store, &arrays, row1 - row0, num_limbs, None));
        }
        out
    }

    /// The block plan must partition BOTH the rows and the products, exactly and in order.
    ///
    /// This is the property the whole split rests on: rows are independent, so concatenating the
    /// blocks reproduces the single-launch result -- but only if every row appears in exactly one
    /// block and every product goes with its row. An overlap would double-XOR a row, silently
    /// cancelling it; a gap would drop one. Neither crashes, and both give a plausible wrong
    /// answer, which is why this is checked directly rather than inferred from a digest.
    #[test]
    fn row_blocks_partition_rows_and_products() {
        let max_degree = 40;
        let algebra = algebra_to(max_degree);
        let num_rows = 97;
        let mut products = sample_products(&algebra, max_degree, num_rows, 500, 0xb10c);
        // The planner requires row-sorted products, which is what the extract loops emit.
        products.sort_by_key(|p| p.row);
        let num_limbs = 64;

        for rows_per_block in [1usize, 2, 7, 64, 1 << 20] {
            let budget = rows_per_block * num_limbs * size_of::<u32>();
            let plan = row_blocks(&products, num_rows, num_limbs, budget);
            assert!(!plan.is_empty(), "{rows_per_block}: empty plan");
            let (mut next_row, mut next_p) = (0usize, 0usize);
            for &(r0, r1, p0, p1) in &plan {
                assert_eq!(r0, next_row, "{rows_per_block}: rows are not contiguous");
                assert_eq!(p0, next_p, "{rows_per_block}: products are not contiguous");
                assert!(r1 > r0, "{rows_per_block}: empty row range");
                for prod in &products[p0..p1] {
                    assert!(
                        (r0..r1).contains(&prod.row),
                        "{rows_per_block}: product row {} outside block {r0}..{r1}",
                        prod.row
                    );
                }
                next_row = r1;
                next_p = p1;
            }
            assert_eq!(next_row, num_rows, "{rows_per_block}: rows not covered");
            assert_eq!(
                next_p,
                products.len(),
                "{rows_per_block}: products not covered"
            );
        }
    }

    /// The marshalled walk, run block by block, reproduces the CPU reference.
    ///
    /// The device replay shows the digest unchanged from a 1 GiB budget down to 4 MB (one block to
    /// ~85), but that needs a card. This covers the same property with none -- including the
    /// per-block `row0` rebasing of `prod_row_base`, which is exactly where a split goes wrong.
    #[test]
    fn blocked_walk_matches_cpu_reference() {
        let max_degree = 40;
        let algebra = algebra_to(max_degree);
        let num_rows = 48;
        let mut products = sample_products(&algebra, max_degree, num_rows, 300, 0x5917);
        products.sort_by_key(|p| p.row);
        let num_cols = full_width(&algebra, &products);
        let num_limbs = num_cols.div_ceil(32).max(1);

        let want = cpu_multiply_batch(&algebra, num_cols, num_rows, &products);
        assert!(
            want.iter_rows().flatten().any(|&w| w != 0),
            "the reference is all zero, so the comparison proves nothing"
        );

        for rows_per_block in [1usize, 5, 48] {
            let got = walk_blocked(&algebra, &products, num_rows, num_limbs, rows_per_block);
            for (r, want_row) in want.iter_rows().enumerate() {
                let got_row = &got[r * num_limbs..(r + 1) * num_limbs];
                assert_eq!(
                    got_row, want_row,
                    "rows_per_block {rows_per_block}, row {r}: blocked walk != CPU reference"
                );
            }
        }
    }

    /// The marshalled arrays, walked the kernel's way, reproduce the CPU reference.
    #[test]
    fn marshalled_walk_matches_cpu_reference() {
        let max_degree = 40;
        let algebra = algebra_to(max_degree);
        let num_rows = 24;
        let products = sample_products(&algebra, max_degree, num_rows, 400, 0x5eed);
        let num_cols = full_width(&algebra, &products);
        let num_limbs = num_cols.div_ceil(32).max(1);

        let got = walk(&algebra, &products, num_rows, num_limbs, None);
        let want = cpu_multiply_batch(&algebra, num_cols, num_rows, &products);

        // An all-zero result agreeing with an all-zero reference is the failure mode that cost this
        // project weeks: cubecl swallowed a failed allocation and returned a zeroed buffer at
        // exit 0. Every comparison here first insists the reference carries bits.
        assert!(
            want.iter_rows().flatten().any(|&w| w != 0),
            "the reference is all zero, so the comparison proves nothing"
        );
        for (r, want_row) in want.iter_rows().enumerate() {
            let got_row = &got[r * num_limbs..(r + 1) * num_limbs];
            assert_eq!(
                got_row, want_row,
                "row {r}: marshalled walk != CPU reference"
            );
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
        let num_limbs = out_cols.div_ceil(32).max(1);

        let got = walk(&algebra, &products, num_rows, num_limbs, Some(&col_map));
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

    /// Appending an `R` must not disturb where the previous ones live.
    ///
    /// This is the property the whole VMM design turns on -- growth in place -- and it is cheap to
    /// assert directly on the layout, with no device involved.
    #[test]
    fn master_layout_is_append_only() {
        let algebra = algebra_to(30);
        let mut layout = MasterLayout::default();
        let mut seen: Vec<(PPart, RInfo)> = Vec::new();
        for d in 1..=10 {
            for i in 0..algebra.dimension(d) {
                let p = algebra.basis_element_from_index(d, i).p_part.clone();
                if p.is_empty() || layout.get(&p).is_some() {
                    continue;
                }
                let (cs_len, mk_len, num_mats, cs, mk) = r_tables(&algebra, &p).expect("tables");
                let info = layout.place(&p, cs_len, mk_len, num_mats, cs.len(), mk.len());
                // A run of `num_mats` rectangular matrices, and nothing past the fill mark.
                assert_eq!(cs.len(), num_mats * cs_len, "col_sums is not rectangular");
                assert_eq!(mk.len(), num_mats * mk_len, "masks is not rectangular");
                assert!(info.cs_offset as usize + cs.len() <= layout.cs_elems());
                seen.push((p, info));
            }
        }
        assert!(seen.len() > 10, "too few distinct R to be a real check");
        for (p, info) in &seen {
            assert_eq!(
                layout.get(p),
                Some(*info),
                "an earlier R MOVED as later ones were appended"
            );
        }
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
    /// never be allocated at all.
    #[test]
    #[ignore = "needs a CUDA device; run explicitly with --ignored"]
    fn cuda_batch_matches_cpu_reference_masked() {
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

    /// Do concurrent launches on one device OVERLAP, or does the resident store serialise them?
    ///
    /// This matters more than any single launch's cost. A production frontier run has roughly 18
    /// threads inside GPU calls at once against 3 devices; if they queue behind one mutex, every
    /// per-launch optimisation in this module is competing for a lane that is already full.
    ///
    /// Reports the speedup of N threads against the same work done serially. Perfect overlap is
    /// bounded well below N -- the device is shared and the kernel is most of the launch -- so the
    /// number to watch is whether it is ABOVE 1.0 at all.
    #[test]
    #[ignore = "needs a CUDA device and captured batches; run explicitly with --ignored"]
    fn cuda_concurrent_launches_overlap() {
        use std::time::Instant;

        let Ok(dir) = std::env::var("NASSAU_REPLAY_PRODUCTS") else {
            panic!("NASSAU_REPLAY_PRODUCTS is unset; nothing to replay");
        };
        let mut files: Vec<String> = std::fs::read_dir(&dir)
            .expect("read capture dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path().display().to_string())
            .filter(|p| p.ends_with(".bin"))
            .collect();
        files.sort();
        assert!(files.len() >= 2, "need at least two captures to overlap");

        let loaded: Vec<_> = files
            .iter()
            .map(|f| load_captured_batch(f).expect("load"))
            .collect();
        let max_degree = loaded
            .iter()
            .flat_map(|(_, _, _, prods)| prods.iter().map(|p| p.r_degree + p.s_degree))
            .max()
            .unwrap_or(1)
            .max(1);
        let algebra = algebra_to(max_degree);
        let rt = super::super::runtime(0).expect("open device 0");

        // Warm the store so the measurement is about the launch, not first-touch enumeration.
        for (rows, cols, cm, prods) in &loaded {
            cuda_multiply_batch(&rt, &algebra, *cols, *rows, prods, cm.as_deref()).expect("warm");
        }

        let reps = 3usize;
        // The timed closure LAUNCHES AND DROPS. It must not digest: that is ~50ms of host work on
        // a 338 MB output, it happens outside the store's lock, and letting it into the timed
        // region makes the parallel arm look like it overlapped launches when what overlapped was
        // the hashing. That confound read as a 1.73x speedup while the lock data showed 0.94s of
        // genuine serialisation.
        let run_one = |i: usize| {
            let (rows, cols, cm, prods) = &loaded[i % loaded.len()];
            let out = cuda_multiply_batch(&rt, &algebra, *cols, *rows, prods, cm.as_deref())
                .expect("device batch");
            out.rows()
        };
        // Correctness is checked once, untimed, comparing a serial pass against a concurrent one.
        let digest_one = |i: usize| {
            let (rows, cols, cm, prods) = &loaded[i % loaded.len()];
            cuda_multiply_batch(&rt, &algebra, *cols, *rows, prods, cm.as_deref())
                .expect("device batch")
                .digest()
        };
        let serial_digests: Vec<_> = (0..loaded.len()).map(digest_one).collect();
        let concurrent_digests: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..loaded.len())
                .map(|i| {
                    let digest_one = &digest_one;
                    scope.spawn(move || digest_one(i))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(
            serial_digests, concurrent_digests,
            "concurrent launches disagreed with serial ones"
        );
        assert!(
            serial_digests.iter().all(|&(_, ones)| ones > 0),
            "all-zero outputs digest fine; this measured nothing"
        );

        // The serial arm must do the SAME work in the SAME grouping as the parallel one: each
        // batch `reps` times in a row. Cycling the batches instead would give the parallel arm a
        // locality advantage -- each of its threads repeats one batch -- and the measurement would
        // credit that to overlap.
        // Both arms do the SAME work in the SAME grouping: each batch `reps` times in a row.
        let t = Instant::now();
        for i in 0..loaded.len() {
            for _ in 0..reps {
                std::hint::black_box(run_one(i));
            }
        }
        let serial = t.elapsed().as_secs_f64();

        let t = Instant::now();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..loaded.len())
                .map(|i| {
                    let run_one = &run_one;
                    scope.spawn(move || {
                        for _ in 0..reps {
                            std::hint::black_box(run_one(i));
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
        let parallel = t.elapsed().as_secs_f64();

        eprintln!(
            "[concurrency] {} threads x {reps} reps: serial={serial:.3}s parallel={parallel:.3}s \
             speedup={:.2}x",
            loaded.len(),
            serial / parallel
        );
    }

    /// Replay REAL captured batches through the device and against the CPU reference.
    ///
    /// The generated products above span many `(R, s)` pairs but give every product a similar
    /// shape. Real batches do not: the `R` distribution is steeply skewed, term counts vary per
    /// product, and the row/offset layout is whatever the resolution emitted. A synthetic bench has
    /// already been caught ranking a 4x2 tile at 0.706x where the truth was 1.16x -- the WRONG SIGN
    /// against a 0.4% noise floor -- so "agrees on generated input" is a weaker claim than it looks.
    ///
    /// Running several batches in one process also exercises the RESIDENT store across launches,
    /// which a single batch cannot: the later batches must reuse the `R`s and basis degrees the
    /// earlier ones appended, and get identical answers out of them.
    ///
    /// Point `NASSAU_REPLAY_PRODUCTS` at a file or a directory of `batch_*.bin`. Capture with:
    ///
    /// ```text
    /// NASSAU_CAPTURE_PRODUCTS=<dir> NASSAU_CAPTURE_NTH=40 NASSAU_CAPTURE_COUNT=6 \
    ///   ./resolve_through_stem   # then feed "S_2", "", 80, 30
    /// ```
    #[test]
    #[ignore = "needs a CUDA device and a captured batch; run explicitly with --ignored"]
    fn cuda_replay_matches_cpu_reference() {
        let Ok(path) = std::env::var("NASSAU_REPLAY_PRODUCTS") else {
            panic!(
                "NASSAU_REPLAY_PRODUCTS is unset. A replay test that silently passes with nothing \
                 to replay is worse than no test: set it to a capture file or directory."
            );
        };
        let mut files: Vec<String> = if std::path::Path::new(&path).is_dir() {
            let mut v: Vec<String> = std::fs::read_dir(&path)
                .expect("read capture dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path().display().to_string())
                .filter(|p| p.ends_with(".bin"))
                .collect();
            v.sort();
            v
        } else {
            vec![path]
        };
        assert!(!files.is_empty(), "no .bin captures found");
        files.truncate(8);

        let rt = super::super::runtime(0).expect("open device 0");
        for file in &files {
            let (rows, cols, cm, prods) =
                load_captured_batch(file).expect("failed to load the captured batch");
            let max_degree = prods
                .iter()
                .map(|p| p.r_degree + p.s_degree)
                .max()
                .unwrap_or(1)
                .max(1);
            let algebra = algebra_to(max_degree);

            let want = cpu_multiply_batch_masked(&algebra, cols, cm.clone(), rows, &prods);
            let got = cuda_multiply_batch(&rt, &algebra, cols, rows, &prods, cm.as_deref())
                .expect("device batch");

            let (wh, wones) = want.digest();
            let (gh, gones) = got.digest();
            assert!(
                wones > 0,
                "{file}: the CPU reference is all zero, so this comparison proves nothing"
            );
            eprintln!(
                "[replay] {file}: products={} rows={rows} cols={cols} masked={} \
                 ones={wones} digest={wh:016x} resident={:.1}MB",
                prods.len(),
                cm.is_some(),
                resident_committed_bytes(&rt).unwrap_or(0) as f64 / 1e6,
            );
            assert_eq!(
                (gh, gones),
                (wh, wones),
                "{file}: the device disagrees with the CPU reference"
            );
        }
    }

    /// Time the cudarc path on a captured batch: cold, warm, and the gap between them.
    ///
    /// Reported separately because they answer different questions. COLD includes the CUDA context,
    /// the NVRTC compile, and the first enumeration of every `R`; WARM is the launch alone against
    /// an already-resident master. An edge worker that starts a process per unit of work pays cold
    /// every time, which is the measurement that decides whether workers can be transient -- the
    /// cubecl path measured ~68s cold against ~21s warm, i.e. ~47s of CUDA context plus JIT.
    ///
    /// The digest is checked ACROSS reps and must be identical. Without that, a warm timing could
    /// be measuring a different (or empty) computation and look excellent doing it -- this
    /// codebase has a crashed arm winning a sweep on record, because crashed runs are fast.
    ///
    /// ```text
    /// NASSAU_REPLAY_PRODUCTS=<dir-or-file> NASSAU_REPLAY_REPS=5 \
    ///   cargo test --release -p algebra --features cuda --lib cuda_replay_profile \
    ///   -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "perf profile: needs a CUDA device and a captured batch; run explicitly"]
    fn cuda_replay_profile() {
        use std::time::Instant;

        let Ok(path) = std::env::var("NASSAU_REPLAY_PRODUCTS") else {
            panic!("NASSAU_REPLAY_PRODUCTS is unset; there is nothing to profile");
        };
        let file = if std::path::Path::new(&path).is_dir() {
            let mut v: Vec<String> = std::fs::read_dir(&path)
                .expect("read capture dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path().display().to_string())
                .filter(|p| p.ends_with(".bin"))
                .collect();
            v.sort();
            // The LARGEST batch, not the first: a small one measures launch overhead, and this
            // codebase has already been burned by a harness where 87% of the wall time was
            // overhead and every kernel change measured as 0%.
            v.into_iter()
                .max_by_key(|f| std::fs::metadata(f).map(|m| m.len()).unwrap_or(0))
                .expect("no .bin captures found")
        } else {
            path
        };
        let reps: usize = std::env::var("NASSAU_REPLAY_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        let (rows, cols, cm, prods) =
            load_captured_batch(&file).expect("failed to load the captured batch");
        let max_degree = prods
            .iter()
            .map(|p| p.r_degree + p.s_degree)
            .max()
            .unwrap_or(1)
            .max(1);
        let t_setup = Instant::now();
        let algebra = algebra_to(max_degree);
        let setup = t_setup.elapsed().as_secs_f64();

        let rt = super::super::runtime(0).expect("open device 0");
        let mut times = Vec::new();
        let mut digests = Vec::new();
        let mut phases: Vec<LaunchTiming> = Vec::new();
        for _ in 0..reps.max(2) {
            let t = Instant::now();
            let (out, tm) =
                cuda_multiply_batch_timed(&rt, &algebra, cols, rows, &prods, cm.as_deref())
                    .expect("device batch");
            times.push(t.elapsed().as_secs_f64());
            phases.push(tm);
            let (h, ones) = out.digest();
            assert!(
                ones > 0,
                "an all-zero output digests fine; this rep computed NOTHING"
            );
            digests.push(h);
        }
        assert!(
            digests.iter().all(|d| *d == digests[0]),
            "reps disagree, so the warm timings measure a DIFFERENT computation: {digests:?}"
        );

        let cold = times[0];
        let warm: f64 = times[1..].iter().sum::<f64>() / (times.len() - 1) as f64;
        let pairs: u64 = prods.iter().map(|p| p.term_indices.len() as u64).sum();
        eprintln!(
            "[cuda-profile] {file}\n  products={} terms={pairs} rows={rows} cols={cols} \
             masked={}\n  algebra_setup={setup:.2}s cold={cold:.3}s warm={warm:.3}s \
             startup={:.3}s resident={:.1}MB digest={:016x}",
            prods.len(),
            cm.is_some(),
            (cold - warm).max(0.0),
            resident_committed_bytes(&super::super::runtime(0).expect("dev 0")).unwrap_or(0) as f64
                / 1e6,
            digests[0],
        );
        // The WARM phase split, averaged over the warm reps. Wall time is not kernel time: a
        // kernel optimisation can be real and still move the line above by nothing, and this is
        // what says which of those happened.
        let n = (phases.len() - 1) as f64;
        let avg = |f: fn(&LaunchTiming) -> f64| phases[1..].iter().map(f).sum::<f64>() / n;
        let (res, m, u, k, r) = (
            avg(|t| t.resident),
            avg(|t| t.marshal),
            avg(|t| t.upload),
            avg(|t| t.kernel),
            avg(|t| t.readback),
        );
        let tot = res + m + u + k + r;
        // Pair throughput, so this batch can be calibrated against the frontier rather than
        // assumed representative of it. `pairs` counts (matrix, term) pairs -- kernel THREADS --
        // which needs the resident matrix counts, not just the term counts.
        let store = resident(&super::super::runtime(0).expect("dev 0")).expect("resident");
        let store = store.lock().unwrap();
        let mut pairs: u64 = 0;
        for prod in &prods {
            let pp = algebra
                .basis_element_from_index(prod.r_degree, prod.r_idx)
                .p_part
                .clone();
            if let Some(info) = store.master_get(&pp) {
                pairs += info.num_mats as u64 * prod.term_indices.len() as u64;
            }
        }
        drop(store);
        eprintln!(
            "  pairs={pairs} -> {:.3}e9 pairs/s in the kernel",
            pairs as f64 / k / 1e9
        );
        eprintln!(
            "  warm phases: resident={res:.4}s ({:.1}%) marshal={m:.4}s ({:.1}%) \
             upload={u:.4}s ({:.1}%) kernel={k:.4}s ({:.1}%) readback={r:.4}s ({:.1}%)",
            100.0 * res / tot,
            100.0 * m / tot,
            100.0 * u / tot,
            100.0 * k / tot,
            100.0 * r / tot,
        );
    }
}
