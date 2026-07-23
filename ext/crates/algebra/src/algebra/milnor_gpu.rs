//! GPU offload for the Milnor multiply at `p = 2`, built on [CubeCL].
//!
//! Runs the admissible-matrix multiply
//! ([`super::milnor_algebra::MilnorAlgebra::multiply_basis_element_by_element_2`]) and the
//! hash-free `seqno` index as CubeCL kernels, batched per `get_partial_matrix` launch.
//!
//! The batched kernel `multiply_batch_kernel` fuses all `(R, s)` products of one
//! `get_partial_matrix` into a single launch — one thread per `(product, matrix, term)` pair,
//! decoded on-device from a prefix-sum over per-product pair counts. Admissible-matrix data is
//! deduplicated by distinct `R`, so a launch uploads compact per-`R`/per-product tables rather
//! than a per-pair table (which at scale would be gigabytes of almost-entirely-redundant data).
//! Its building blocks are the F₂ XOR accumulation (`xor_f2`), the on-device `seqno` index
//! (`seqno_core`/`seqno_kernel`, porting [`MilnorAlgebra::seqno`] as integer arithmetic over the
//! flat `g` table), and the single-`R` product (`multiply_pair`).
//!
//! Gated behind the `gpu` feature. Running needs the CUDA toolkit on `CUDA_PATH` /
//! `LD_LIBRARY_PATH` (the `gpu` dev shell in `ext/flake.nix` sets both) and a live
//! device; `cargo check`/`build` need neither (cudarc dlopens at runtime).
//!
//! [CubeCL]: https://github.com/tracel-ai/cubecl

use cubecl::{
    cuda::{CudaDevice, CudaRuntime},
    prelude::*,
};
use cubecl_common::stream_id::StreamId;
// Only the `#[cfg(test)]` standalone `seqno_kernel` sizes its working array by this bound; the
// production kernels use `WORKING_CAP`.
#[cfg(test)]
use crate::algebra::combinatorics::MAX_XI_TAU;
use crate::algebra::{Algebra, MilnorAlgebra, combinatorics::xi_degrees};

/// Comptime capacity for the per-thread `working` p_part in the multiply kernel.
/// The assembled p_part has length `max(term_len, mk_len)` before trimming, where
/// `mk_len = rows + cols − 1 ≤ MAX_XI_TAU + ⌈log2⌉`; 32 covers every in-range case.
const WORKING_CAP: usize = 32;

/// Target `(product, matrix, term)` thread-pairs per GPU launch. The batch multiply indexes
/// threads by CubeCL's `ABSOLUTE_POS` (a `u32`), so one launch can address at most `2^32` threads;
/// a single all-rows reuse build reaches ~4.4e9 pairs at stem ~145, past that limit. The row-block
/// splitter in [`multiply_batch_on_gpu`] closes a block once its pair count would pass this target
/// (alongside the [`gpu_block_bytes`] output budget). `1 << 30` (~1.07e9) leaves >3x headroom
/// under `2^32` even when a lone over-budget row overshoots it, and keeps the grid
/// (`total_pairs / 256` cubes) well under CUDA's `2^31 - 1` grid-dimension limit.
const GPU_PAIR_CHUNK: usize = 1 << 30;

/// Per-launch output-buffer budget in bytes (`NASSAU_GPU_BLOCK_MB`, default 512 MiB).
///
/// A launch's transient footprint — host marshal buffers, pinned staging, device buffers, and
/// each stream's retained pool pages — scales with its output size, and with the device mutex
/// gone many workers hold such transients simultaneously; at record stems an unbounded all-rows
/// reuse build multiplies to >100 GB on both host and device. [`multiply_batch_on_gpu`] therefore
/// splits large builds into row blocks whose output buffer stays under this budget. Rows of
/// distinct products are independent (each product writes only its own row), so blocks simply
/// concatenate — the same in-between as the old per-signature builds, but with blocks big enough
/// to keep the launch amortization. Together with [`GPU_PERMITS`] this makes peak transient
/// memory a configured constant (≈ permits × budget) instead of a function of the frontier size.
fn gpu_block_bytes() -> usize {
    static BYTES: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("NASSAU_GPU_BLOCK_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(512)
            << 20
    });
    *BYTES
}

/// Counting semaphore bounding how many workers may be inside the layout + device section at
/// once (`NASSAU_GPU_CONCURRENCY`, default 8). With per-thread streams every worker can launch
/// concurrently, which is the throughput win — but each concurrent section holds up to
/// [`gpu_block_bytes`] of transient host and device memory, so the count must be capped.
///
/// SAFETY INVARIANT: a permit must never be held across a rayon parallel section. A par_iter's
/// chunks execute on other threads, which do not carry the holder's thread-local
/// `ParallelGuard` flag and so can steal a resolution-step job mid-chunk; that job would park
/// on [`GpuPermit::acquire`] while the holder's permit waits on the never-finishing join —
/// a cycle (observed as a full stall on H200). [`multiply_batch_block`] therefore acquires its
/// permit only after the parallel marshal, guarding a strictly sequential section: every holder
/// makes progress, so parked acquirers always wake (priority inversion at worst, never
/// deadlock).
struct GpuPermits {
    free: Mutex<Vec<usize>>,
    freed: Condvar,
}

static GPU_PERMITS: LazyLock<GpuPermits> = LazyLock::new(|| {
    let max = std::env::var("NASSAU_GPU_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(8);
    GpuPermits {
        free: Mutex::new((0..max).rev().collect()),
        freed: Condvar::new(),
    }
});

/// RAII permit from [`GPU_PERMITS`]; blocks (parked, not spinning) until one frees.
///
/// A permit is also a *stream slot*: the holder runs its device section under
/// `StreamId { value: slot }`, so the process only ever touches `NASSAU_GPU_CONCURRENCY`
/// CUDA streams. Without the pin every worker thread gets its own default stream, and each
/// stream's memory pool *retains* its freed slabs (`memory_cleanup` only trims the calling
/// stream, at its next launch) — ~100 worker streams each retaining ~1 GB of out/staging
/// buffers filled the whole 143 GB H200. Pinning bounds device retention to
/// ≈ permits × [`gpu_block_bytes`].
struct GpuPermit {
    slot: usize,
}

impl GpuPermit {
    fn acquire() -> Self {
        let permits = &*GPU_PERMITS;
        let mut free = permits.free.lock().unwrap();
        loop {
            if let Some(slot) = free.pop() {
                return Self { slot };
            }
            free = permits.freed.wait(free).unwrap();
        }
    }
}

impl Drop for GpuPermit {
    fn drop(&mut self) {
        let permits = &*GPU_PERMITS;
        permits.free.lock().unwrap().push(self.slot);
        permits.freed.notify_one();
    }
}

/// Narrow an admissible-matrix / p-part entry to the `u16` the GPU buffers use, failing loudly
/// instead of silently wrapping. Every entry is well within `u16` for the stem ranges this path
/// targets; a panic here means that assumption was pushed past its limit, which must not ship
/// truncated data to the device.
fn narrow_u16(v: u32) -> u16 {
    u16::try_from(v).expect("admissible/term entry exceeds u16")
}

use std::sync::atomic::{AtomicU64, Ordering};

/// Aggregate [`multiply_batch_on_gpu`] counters across all launches (call count, host
/// marshal µs, device µs, total pairs), for splitting a whole resolution's GPU overhead.
static BATCH_CALLS: AtomicU64 = AtomicU64::new(0);
static BATCH_MARSHAL_US: AtomicU64 = AtomicU64::new(0);
static BATCH_DEVICE_US: AtomicU64 = AtomicU64::new(0);
static BATCH_PAIRS: AtomicU64 = AtomicU64::new(0);

/// Read and reset the aggregate batch counters: `(calls, marshal_us, device_us, pairs)`.
pub fn take_batch_stats() -> (u64, u64, u64, u64) {
    (
        BATCH_CALLS.swap(0, Ordering::Relaxed),
        BATCH_MARSHAL_US.swap(0, Ordering::Relaxed),
        BATCH_DEVICE_US.swap(0, Ordering::Relaxed),
        BATCH_PAIRS.swap(0, Ordering::Relaxed),
    )
}

use std::{
    collections::HashMap,
    sync::{Condvar, LazyLock, Mutex, RwLock},
};

use cubecl::server::Handle;

use crate::algebra::milnor_algebra::PPartEntry;

/// Where one `R`'s admissible-matrix data lives inside the resident master buffers.
#[derive(Clone, Copy)]
struct RInfo {
    cs_off: u32,
    mk_off: u32,
    cs_len: u32,
    mk_len: u32,
    num_mats: u32,
}

/// Process-shared host master of admissible-matrix data.
///
/// Admissible-matrix enumeration is a pure function of `R`'s p-part and the same
/// low-degree `R`s recur in essentially every bidegree, so the host master (`col_sums` /
/// `masks`, append-only, keyed by p-part in `index`) is enumerated once per distinct `R`
/// and never recomputed.
///
/// SHARED, not per-thread: the master reaches many GB at record stems (it grows with the
/// degree), so a thread-local copy per rayon worker multiplies it by the worker count —
/// measured at ~137 GB host / a full 143 GB H200 with 16 workers at stem 150. One copy
/// behind an `RwLock` restores the old shared-mutex footprint: lookups (the overwhelmingly
/// common case once the `R`s saturate) take the read lock, and only a first-sight append
/// takes the write lock — with the enumeration itself done *outside* the lock, so readers
/// never stall behind it.
#[derive(Default)]
struct ResidentHost {
    col_sums: Vec<u16>,
    masks: Vec<u16>,
    index: HashMap<Vec<PPartEntry>, RInfo>,
}

static RESIDENT_HOST: LazyLock<RwLock<ResidentHost>> =
    LazyLock::new(|| RwLock::new(ResidentHost::default()));

/// Process-shared device mirror of the host master: one upload for the whole process,
/// re-uploaded only when the master grew. The handles are shared across worker threads and
/// stream slots — safe under cubecl 0.10's per-device runner (which serializes all server
/// access), with cross-stream reuse event-synced via the handle's origin-stream stamp. The
/// mutex guards only the grew-check + upload, held briefly per launch.
#[derive(Default)]
struct ResidentDev {
    cs_handle: Option<Handle>,
    mk_handle: Option<Handle>,
    cs_uploaded: usize,
    mk_uploaded: usize,
}

static RESIDENT_DEV: LazyLock<Mutex<ResidentDev>> =
    LazyLock::new(|| Mutex::new(ResidentDev::default()));

/// Global offsets/lengths of `R`'s admissible matrices in the shared host master (see
/// [`ResidentHost`]), enumerating and appending them on first sight (the append order fixes
/// the offsets forever). The enumeration runs outside any lock; on a first-sight race the
/// loser rechecks under the write lock and discards its duplicate.
fn resident_info(algebra: &MilnorAlgebra, p_part: &[PPartEntry]) -> RInfo {
    if let Some(info) = RESIDENT_HOST.read().unwrap().index.get(p_part) {
        return *info;
    }
    let (cs_len, mk_len, cs, mk) = algebra.admissible_matrices(p_part);
    let mut host = RESIDENT_HOST.write().unwrap();
    if let Some(info) = host.index.get(p_part) {
        return *info;
    }
    let info = RInfo {
        cs_off: host.col_sums.len() as u32,
        mk_off: host.masks.len() as u32,
        cs_len: cs_len as u32,
        mk_len: mk_len as u32,
        num_mats: (mk.len() / mk_len) as u32,
    };
    host.col_sums.extend(cs.iter().map(|&v| narrow_u16(v)));
    host.masks.extend(mk.iter().map(|&v| narrow_u16(v)));
    host.index.insert(p_part.to_vec(), info);
    info
}

/// Elementwise F₂ addition of two bit-packed vectors: `out[i] = a[i] ^ b[i]`.
///
/// One thread per `u32` limb. F₂ addition is XOR of the packed limbs, so this is
/// the output primitive the multiply kernels accumulate with.
#[cfg(test)]
#[cube(launch)]
fn xor_f2(a: &Array<u32>, b: &Array<u32>, out: &mut Array<u32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] ^ b[ABSOLUTE_POS];
    }
}

/// Compute `a ^ b` limb-wise on the default CUDA device.
///
/// Host-side driver for `xor_f2`: uploads both operands, launches one thread per
/// limb, and reads the result back. Panics if the operands differ in length.
#[cfg(test)]
pub fn xor_f2_on_gpu(a: &[u32], b: &[u32]) -> Vec<u32> {
    assert_eq!(a.len(), b.len(), "operands must have equal limb counts");
    let n = a.len();
    let client = CudaRuntime::client(&CudaDevice::default());

    let a_handle = client.create_from_slice(u32::as_bytes(a));
    let b_handle = client.create_from_slice(u32::as_bytes(b));
    let out_handle = client.empty(std::mem::size_of_val(a));

    // One 1-D block of `THREADS` units, enough blocks to cover every limb.
    const THREADS: u32 = 256;
    let cubes = (n as u32).div_ceil(THREADS);
    unsafe {
        xor_f2::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(THREADS),
            ArrayArg::from_raw_parts(a_handle, n),
            ArrayArg::from_raw_parts(b_handle, n),
            ArrayArg::from_raw_parts(out_handle.clone(), n),
        );
    }

    let bytes = client.read_one(out_handle).unwrap();
    u32::from_bytes(&bytes).to_vec()
}

/// Device port of [`MilnorAlgebra::seqno`]: the index of `P(working)` in the Milnor
/// basis of its degree, from the flat `g` table with no hashing. `working` holds the
/// (trimmed) p_part in its first `wlen` entries; `g` has row width `width`, entry
/// `(e, h)` at `g[e*width + h]`; `xi` are the ξ-degrees.
///
/// Thread/array indices are `usize`; p_part and table *values* are `u32`. A degree
/// (`cur_d`) is a value computed from `u32`s but also indexes `g`, so it is cast to
/// `usize` at the index sites. Shared by `seqno_kernel` and
/// `multiply_single_r_kernel` so both index outputs identically.
#[cube]
fn seqno_core(
    g: &Array<u32>,
    xi: &Array<u32>,
    working: &Array<u32>,
    wlen: usize,
    width: usize,
) -> u32 {
    // cur_d = Σ working[h] · xi[h].
    let mut cur_d = 0u32;
    for h in 0..wlen {
        cur_d += working[h] * xi[h];
    }

    // Rank by consuming positions from high to low; position 0 contributes nothing.
    let mut rank = 0u32;
    for hh in 1..wlen {
        let h = wlen - hh; // wlen-1 down to 1
        let r = working[h];
        if r != 0 {
            let below = cur_d - r * xi[h];
            let cur_row = usize::cast_from(cur_d) * width + h;
            let below_row = usize::cast_from(below) * width + h;
            rank += g[cur_row] - g[below_row];
            cur_d = below;
        }
    }
    rank
}

/// One thread per padded p_part: `out[i] = seqno(p_parts[i])`. `p_parts` is
/// `n × width` row-major, each row a p_part zero-padded to `width` (padding entries
/// are zero and skipped, so `wlen == width` matches the CPU's trimmed loop).
#[cfg(test)]
#[cube(launch)]
fn seqno_kernel(
    g: &Array<u32>,
    xi: &Array<u32>,
    p_parts: &Array<u32>,
    out: &mut Array<u32>,
    width: usize,
) {
    let idx = ABSOLUTE_POS;
    if idx >= out.len() {
        terminate!();
    }
    let base = idx * width;

    let mut working = Array::<u32>::new(MAX_XI_TAU);
    for h in 0..width {
        working[h] = p_parts[base + h];
    }
    out[idx] = seqno_core(g, xi, &working, width, width);
}

/// Run `seqno_kernel` over `n` padded p_parts and return their seqno indices.
///
/// `g`/`xi` come from `MilnorAlgebra::seqno_table_u32` and
/// [`crate::algebra::combinatorics::xi_degrees`]; `p_parts` is `n × width` row-major,
/// each row a p_part zero-padded to `width`.
#[cfg(test)]
pub fn seqno_batch_on_gpu(
    width: usize,
    xi: &[u32],
    g: &[u32],
    p_parts: &[u32],
    n: usize,
) -> Vec<u32> {
    assert_eq!(xi.len(), width, "xi must have `width` entries");
    assert_eq!(p_parts.len(), n * width, "p_parts must be n × width");
    let client = CudaRuntime::client(&CudaDevice::default());

    let g_h = client.create_from_slice(u32::as_bytes(g));
    let xi_h = client.create_from_slice(u32::as_bytes(xi));
    let pp_h = client.create_from_slice(u32::as_bytes(p_parts));
    let out_h = client.empty(n * size_of::<u32>());

    const THREADS: u32 = 256;
    let cubes = (n as u32).div_ceil(THREADS);
    unsafe {
        seqno_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(THREADS),
            ArrayArg::from_raw_parts(g_h, g.len()),
            ArrayArg::from_raw_parts(xi_h, xi.len()),
            ArrayArg::from_raw_parts(pp_h, p_parts.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            width,
        );
    }

    let bytes = client.read_one(out_h).unwrap();
    u32::from_bytes(&bytes).to_vec()
}

/// Assemble one `(admissible matrix, term)` product and XOR its F₂ output bit into
/// `out` at `row_base + idx`. The whole per-term test + output assembly of
/// [`MilnorAlgebra::multiply_basis_element_by_element_2`] lives here; both the
/// single-`R` and batch kernels call it with per-pair offsets.
///
/// The reference's three tail branches collapse into one uniform per-position rule:
/// for column `j`, with `b`, `cs`, `mk` the term / `col_sums` / `masks` entries (zero
/// outside their lengths) and `low = min(term_len, cs_len)` —
/// - `j < low`: reject if `cs > b` or `(b−cs) & mk`; else `working[j] = (b−cs) | mk`.
/// - `j ≥ low`: reject if `cs > 0` or `b & mk`; else `working[j] = b | mk`.
///
/// (For `j ≥ low` at most one of `b`, `cs` is in range, so this reproduces every
/// branch.) `seqno_core` gives the output index; the F₂ bit is XORed atomically
/// (collisions cancel mod 2). No explicit trailing-zero trim is needed — `seqno_core`
/// skips zero entries and `working` beyond the assembled length is zero, so the full
/// `WORKING_CAP` length is equivalent to the CPU's trimmed p_part (`xi` is host-padded
/// to `WORKING_CAP` so the `cur_d` sum stays in bounds; the extra terms are `0 · xi`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn multiply_pair(
    col_sums: &Array<u16>,
    masks: &Array<u16>,
    term_pparts: &Array<u16>,
    g: &Array<u32>,
    xi: &Array<u32>,
    out: &mut Array<Atomic<u32>>,
    cs_base: usize,
    mk_base: usize,
    b_base: usize,
    term_len: usize,
    cs_len: usize,
    mk_len: usize,
    row_base: usize,
    out_offset: usize,
    width: usize,
) {
    let mut low = cs_len;
    if term_len < cs_len {
        low = term_len;
    }

    let mut working = Array::<u32>::new(WORKING_CAP);
    let mut rejected = false;

    for j in 0..WORKING_CAP {
        let mut b = 0u32;
        if j < term_len {
            b = u32::cast_from(term_pparts[b_base + j]);
        }
        let mut cs = 0u32;
        if j < cs_len {
            cs = u32::cast_from(col_sums[cs_base + j]);
        }
        let mut mk = 0u32;
        if j < mk_len {
            mk = u32::cast_from(masks[mk_base + j]);
        }

        let mut val = 0u32;
        if j < low {
            if cs > b {
                rejected = true;
            } else {
                let diff = b - cs;
                if (diff & mk) != 0u32 {
                    rejected = true;
                } else {
                    val = diff | mk;
                }
            }
        } else {
            if cs > 0u32 {
                rejected = true;
            }
            if (b & mk) != 0u32 {
                rejected = true;
            }
            val = b | mk;
        }
        working[j] = val;
    }

    if !rejected {
        // `seqno` indexes the algebra basis of the output degree; `out_offset` shifts it
        // to this product's target-generator block within the row (0 for a single-block
        // output). Both are bit offsets, added before splitting into (limb, bit).
        let idx = seqno_core(g, xi, &working, WORKING_CAP, width);
        let global_bit = out_offset + usize::cast_from(idx);
        let word = row_base + global_bit / 32;
        let bit = u32::cast_from(global_bit % 32);
        out[word].fetch_xor(1u32 << bit);
    }
}

/// Multiply `Sq(R) · s` for a single fixed operation `R` into one F₂ output vector.
/// One thread per `(matrix, term)` pair; delegates the assembly to `multiply_pair`.
#[cfg(test)]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn multiply_single_r_kernel(
    col_sums: &Array<u16>,
    masks: &Array<u16>,
    term_pparts: &Array<u16>,
    term_lens: &Array<u32>,
    g: &Array<u32>,
    xi: &Array<u32>,
    out: &mut Array<Atomic<u32>>,
    num_terms: usize,
    num_matrices: usize,
    cs_len: usize,
    mk_len: usize,
    width: usize,
) {
    let pair = ABSOLUTE_POS;
    if pair >= num_matrices * num_terms {
        terminate!();
    }
    let m = pair / num_terms;
    let t = pair % num_terms;
    let term_len = usize::cast_from(term_lens[t]);
    multiply_pair(
        col_sums,
        masks,
        term_pparts,
        g,
        xi,
        out,
        m * cs_len,
        m * mk_len,
        t * width,
        term_len,
        cs_len,
        mk_len,
        0,
        0,
        width,
    );
}

/// Batched multiply: one launch covering all `(R, s)` products of (e.g.) a
/// `get_partial_matrix` call. One thread per `(product, matrix, term)` pair.
///
/// Rather than a per-pair table (7 arrays × total-pairs — up to gigabytes at scale,
/// almost all redundant), the pair a thread handles is *decoded* from compact data:
/// - `prod_pair_start` is a prefix-sum of each product's pair count (`num_matrices ×
///   num_terms`), length `num_products + 1`. A binary search finds the product `p`
///   owning thread `k`, then `local = k − prod_pair_start[p]` splits into matrix
///   `m = local / num_terms` and term `t = local % num_terms`.
/// - Admissible-matrix data (`col_sums`/`masks`) is deduplicated by distinct `R`:
///   `prod_r_index[p]` indexes the per-`R` `r_*` tables, so an `R` shared across many
///   rows is stored (and uploaded) once.
///
/// Output is `num_rows` F₂ vectors of `num_limbs` `u32` limbs, row `r` at
/// `out[r*num_limbs ..]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn multiply_batch_kernel(
    col_sums: &Array<u16>,
    masks: &Array<u16>,
    term_pparts: &Array<u16>,
    term_lens: &Array<u32>,
    g: &Array<u32>,
    xi: &Array<u32>,
    out: &mut Array<Atomic<u32>>,
    r_cs_offset: &Array<u32>,
    r_mk_offset: &Array<u32>,
    r_cs_len: &Array<u32>,
    r_mk_len: &Array<u32>,
    prod_r_index: &Array<u32>,
    prod_term_start: &Array<u32>,
    prod_num_terms: &Array<u32>,
    prod_row_base: &Array<u32>,
    prod_out_offset: &Array<u32>,
    prod_pair_start: &Array<u32>,
    width: usize,
) {
    let k = ABSOLUTE_POS;
    let num_products = prod_pair_start.len() - 1;
    if k >= usize::cast_from(prod_pair_start[num_products]) {
        terminate!();
    }

    // Largest product `p` with `prod_pair_start[p] <= k` (every product owns ≥ 1 pair,
    // so `prod_pair_start` is strictly increasing and `p` is unique). 32 iterations
    // cover any realistic product count; once `hi = lo + 1` the update is idempotent.
    let mut lo = 0usize;
    let mut hi = num_products;
    for _ in 0..32 {
        if hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if usize::cast_from(prod_pair_start[mid]) <= k {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    let p = lo;

    let ri = usize::cast_from(prod_r_index[p]);
    let nt = usize::cast_from(prod_num_terms[p]);
    let local = k - usize::cast_from(prod_pair_start[p]);
    let m = local / nt;
    let t = local % nt;

    let cs_len = usize::cast_from(r_cs_len[ri]);
    let mk_len = usize::cast_from(r_mk_len[ri]);
    let term_slot = usize::cast_from(prod_term_start[p]) + t;
    multiply_pair(
        col_sums,
        masks,
        term_pparts,
        g,
        xi,
        out,
        usize::cast_from(r_cs_offset[ri]) + m * cs_len,
        usize::cast_from(r_mk_offset[ri]) + m * mk_len,
        term_slot * width,
        usize::cast_from(term_lens[term_slot]),
        cs_len,
        mk_len,
        usize::cast_from(prod_row_base[p]),
        usize::cast_from(prod_out_offset[p]),
        width,
    );
}

/// Compute `Sq(R) · s` on the GPU for a single operation `R = (r_degree, r_idx)`,
/// returning the F₂ result as bit-packed `u32` limbs (bit `i` = basis index `i`).
///
/// `term_indices` are the nonzero indices of `s` in the degree-`s_degree` basis.
/// `R` must be non-empty (`Sq(∅) = 1` is the trivial identity the caller handles).
/// Requires the algebra's basis and seqno tables built through `r_degree + s_degree`.
#[cfg(test)]
pub fn multiply_single_r_on_gpu(
    algebra: &MilnorAlgebra,
    r_degree: i32,
    r_idx: usize,
    s_degree: i32,
    term_indices: &[usize],
) -> Vec<u32> {
    let (width, g) = algebra.seqno_table_u32();
    // Pad `xi` to `WORKING_CAP` so the kernel's `cur_d` sum (which runs to the full
    // working capacity) never reads out of bounds; padding entries multiply zero.
    let mut xi: Vec<u32> = xi_degrees(algebra.prime())
        .iter()
        .map(|&x| x as u32)
        .collect();
    xi.resize(WORKING_CAP, 0);

    let r = algebra.basis_element_from_index(r_degree, r_idx);
    assert!(
        !r.p_part.is_empty(),
        "R must be non-empty (Sq(∅) = 1 is the identity)"
    );
    let (cs_len, mk_len, cs32, mk32) = algebra.admissible_matrices(&r.p_part);
    // Ship admissible-matrix / term data as u16 (see `multiply_batch_on_gpu`).
    let mut col_sums: Vec<u16> = cs32.iter().map(|&v| narrow_u16(v)).collect();
    let masks: Vec<u16> = mk32.iter().map(|&v| narrow_u16(v)).collect();
    let num_matrices = masks.len() / mk_len;

    // Terms of s, each p_part padded to `width`, with their true (trimmed) lengths.
    let num_terms = term_indices.len();
    let mut term_pparts = vec![0u16; num_terms * width];
    let mut term_lens = vec![0u32; num_terms];
    for (t, &ti) in term_indices.iter().enumerate() {
        let elt = algebra.basis_element_from_index(s_degree, ti);
        term_lens[t] = elt.p_part.len() as u32;
        for (slot, &v) in term_pparts[t * width..(t + 1) * width]
            .iter_mut()
            .zip(&elt.p_part)
        {
            *slot = narrow_u16(v);
        }
    }

    let out_degree = r_degree + s_degree;
    let dim = algebra.dimension(out_degree);
    let num_limbs = dim.div_ceil(32).max(1);

    // Device buffers must be non-empty; `cs_len == 0` (R's max entry is 1) leaves
    // `col_sums` empty. The kernel never reads past the real lengths.
    if col_sums.is_empty() {
        col_sums.push(0);
    }

    let client = CudaRuntime::client(&CudaDevice::default());
    let cs_h = client.create_from_slice(u16::as_bytes(&col_sums));
    let mk_h = client.create_from_slice(u16::as_bytes(&masks));
    let tp_h = client.create_from_slice(u16::as_bytes(&term_pparts));
    let tl_h = client.create_from_slice(u32::as_bytes(&term_lens));
    let g_h = client.create_from_slice(u32::as_bytes(&g));
    let xi_h = client.create_from_slice(u32::as_bytes(&xi));
    let zeros = vec![0u32; num_limbs];
    let out_h = client.create_from_slice(u32::as_bytes(&zeros));

    let total_pairs = num_matrices * num_terms;
    const THREADS: u32 = 256;
    let cubes = (total_pairs as u32).div_ceil(THREADS).max(1);
    unsafe {
        multiply_single_r_kernel::launch::<CudaRuntime>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(THREADS),
            ArrayArg::from_raw_parts(cs_h, col_sums.len()),
            ArrayArg::from_raw_parts(mk_h, masks.len()),
            ArrayArg::from_raw_parts(tp_h, term_pparts.len()),
            ArrayArg::from_raw_parts(tl_h, term_lens.len()),
            ArrayArg::from_raw_parts(g_h, g.len()),
            ArrayArg::from_raw_parts(xi_h, xi.len()),
            ArrayArg::from_raw_parts(out_h.clone(), num_limbs),
            num_terms,
            num_matrices,
            cs_len,
            mk_len,
            width,
        );
    }

    let bytes = client.read_one(out_h).unwrap();
    u32::from_bytes(&bytes).to_vec()
}

/// One `Sq(R) · s` product of a batched launch, written into output row `row` at bit
/// offset `out_offset`.
///
/// `term_indices` are the nonzero indices of `s` in the degree-`s_degree` basis.
/// Multiple products may target the same `row` (their F₂ contributions XOR together),
/// mirroring how `get_partial_matrix` accumulates a row over generator blocks. The
/// product's `seqno` output indexes the algebra basis of the output degree; `out_offset`
/// is the start of the target-generator block that basis maps into within the row (0 when
/// the whole row is a single algebra element, as in the single-generator tests).
pub struct GpuProduct {
    pub r_degree: i32,
    pub r_idx: usize,
    pub s_degree: i32,
    pub term_indices: Vec<usize>,
    pub row: usize,
    pub out_offset: usize,
}

/// Compute a whole batch of `Sq(R) · s` products on the GPU — the batched unit of one
/// `get_partial_matrix` call, split into row blocks of at most [`gpu_block_bytes`] of output
/// each (see [`multiply_batch_block`]). `R`s may differ (each contributes its own admissible
/// matrices). Returns `num_rows` F₂ vectors, each `⌈num_cols/32⌉` bit-packed `u32` limbs.
///
/// `num_cols` is the *row* width — for a module row that is the module dimension (a sum
/// over generator blocks, generally larger than any single algebra degree's dimension),
/// with each product's `out_offset` selecting its block. Every product's
/// `out_offset + index` must be `< num_cols`. Every `R` must be non-empty; the algebra's
/// basis and seqno tables must reach each product's output degree (`r_degree + s_degree`).
pub fn multiply_batch_on_gpu(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
) -> Vec<Vec<u32>> {
    let num_limbs = num_cols.div_ceil(32).max(1);
    let max_block_rows = (gpu_block_bytes() / (num_limbs * 4)).max(1);
    // Products arrive row-major (the extract loops emit them per input row, in order), so each
    // block is a contiguous product slice. Rows are independent — every product writes only its
    // own row — so concatenating block outputs reproduces the single-launch result exactly.
    debug_assert!(products.windows(2).all(|w| w[0].row <= w[1].row));
    // Per-product `(matrix, term)` pair counts, i.e. kernel threads. The kernel indexes threads
    // by `ABSOLUTE_POS`, a `u32`, so a block must also stay under `2^32` pairs — output bytes
    // alone don't bound this (pairs per row grow with the degree; an unbounded all-rows build
    // reaches ~4.4e9 pairs by stem ~145). This pre-pass also warms the shared resident master,
    // so every block's layout lookups below are read-lock cache hits.
    let prod_pairs: Vec<usize> = products
        .iter()
        .map(|prod| {
            let r = algebra.basis_element_from_index(prod.r_degree, prod.r_idx);
            resident_info(algebra, &r.p_part).num_mats as usize * prod.term_indices.len()
        })
        .collect();
    let mut result: Vec<Vec<u32>> = Vec::with_capacity(num_rows);
    let (mut r0, mut p0) = (0, 0);
    while r0 < num_rows {
        // Grow the block row by row until the next row would break either budget — output bytes
        // ([`gpu_block_bytes`]) or kernel threads ([`GPU_PAIR_CHUNK`]) — always taking at least
        // one row (a lone over-budget row still fits the kernel's `u32` limit, asserted in the
        // block).
        let (mut r1, mut p1) = (r0, p0);
        let mut pairs = 0usize;
        while r1 < num_rows && r1 - r0 < max_block_rows {
            let q = p1 + products[p1..].partition_point(|p| p.row <= r1);
            let row_pairs: usize = prod_pairs[p1..q].iter().sum();
            if r1 > r0 && pairs + row_pairs > GPU_PAIR_CHUNK {
                break;
            }
            pairs += row_pairs;
            (r1, p1) = (r1 + 1, q);
        }
        result.extend(multiply_batch_block(
            algebra,
            num_cols,
            r0,
            r1 - r0,
            &products[p0..p1],
        ));
        (r0, p0) = (r1, p1);
    }
    result
}

/// One bounded launch of [`multiply_batch_on_gpu`]: rows `row_base..row_base + num_rows` of the
/// full build, with `products` the (contiguous, row-major) slice landing in those rows. Holds a
/// [`GpuPermit`] for its sequential layout + device section (acquired only after the parallel
/// marshal — see [`GpuPermits`]), so at most `NASSAU_GPU_CONCURRENCY` device sections run at
/// once across all worker threads.
fn multiply_batch_block(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    row_base: usize,
    num_rows: usize,
    products: &[GpuProduct],
) -> Vec<Vec<u32>> {
    let (width, g) = algebra.seqno_table_u32();
    let mut xi: Vec<u32> = xi_degrees(algebra.prime())
        .iter()
        .map(|&x| x as u32)
        .collect();
    xi.resize(WORKING_CAP, 0);

    let num_limbs = num_cols.div_ceil(32).max(1);

    let t_marshal = std::time::Instant::now();

    // The two heavy parts of marshalling — enumerating each distinct `R`'s admissible
    // matrices, and looking up + padding every term's p-part — are independent per item,
    // so they run in parallel (rayon via `concurrent`; serial otherwise). The cheap
    // sequential glue (interning `R`s, concatenation, prefix sums) stays on one thread.
    use maybe_rayon::prelude::*;

    // Intern distinct `R`s in first-seen order (cheap, sequential); record each product's
    // `R` index. Admissible-matrix data is thus deduplicated: an `R` shared across many
    // rows is enumerated and uploaded once.
    let mut r_index: std::collections::HashMap<(i32, usize), u32> =
        std::collections::HashMap::new();
    let mut distinct_r: Vec<(i32, usize)> = Vec::new();
    let mut prod_r_index: Vec<u32> = Vec::with_capacity(products.len());
    for prod in products {
        let ri = *r_index
            .entry((prod.r_degree, prod.r_idx))
            .or_insert_with(|| {
                let i = distinct_r.len() as u32;
                distinct_r.push((prod.r_degree, prod.r_idx));
                i
            });
        prod_r_index.push(ri);
    }

    // Admissible-matrix data (`col_sums`/`masks` + per-`R` offsets) is resident (built in the
    // thread-local `RESIDENT` store below), so nothing to enumerate or lay out here.

    // Parallel: each product's term p-parts (padded to `width`) and lengths.
    let per_prod: Vec<(Vec<u16>, Vec<u32>)> = (0..products.len())
        .into_maybe_par_iter()
        .map(|pi| {
            let prod = &products[pi];
            let nt = prod.term_indices.len();
            let mut tp = vec![0u16; nt * width];
            let mut tl = Vec::with_capacity(nt);
            for (k, &ti) in prod.term_indices.iter().enumerate() {
                let elt = algebra.basis_element_from_index(prod.s_degree, ti);
                tl.push(elt.p_part.len() as u32);
                for (slot, &v) in tp[k * width..(k + 1) * width].iter_mut().zip(&elt.p_part) {
                    *slot = narrow_u16(v);
                }
            }
            (tp, tl)
        })
        .collect();

    // Take the concurrency permit only now, with every rayon parallel section behind us: holding
    // it across the `per_prod` par_iter above deadlocks, because that par_iter's chunks execute on
    // *other* threads, which do not carry this thread's `ParallelGuard` flag and so can steal a
    // bidegree job mid-chunk; the stolen job parks on `GpuPermit::acquire` while this thread's
    // permit waits on the never-finishing join (observed on H200). Everything from here on is
    // strictly sequential — the `ensure` calls below are cache hits (the caller's pair-count
    // pre-pass already enumerated every `R`), and the device section never enters rayon — so
    // every permit holder makes progress and stolen jobs waiting for a permit wake in finite
    // time (priority inversion at worst, never deadlock).
    let permit = GpuPermit::acquire();
    // Per-`R` offsets into the shared resident master (see [`ResidentHost`]). All read-lock
    // cache hits: the caller's pair-count pre-pass already enumerated every `R` in this block.
    // `need_cs`/`need_mk` track the furthest master offset this block dereferences, so the
    // device section can skip the (multi-GB, mutex-serialized) master re-upload whenever the
    // already-uploaded prefix covers it.
    let mut r_cs_offset: Vec<u32> = Vec::with_capacity(distinct_r.len());
    let mut r_mk_offset: Vec<u32> = Vec::with_capacity(distinct_r.len());
    let mut r_cs_len: Vec<u32> = Vec::with_capacity(distinct_r.len());
    let mut r_mk_len: Vec<u32> = Vec::with_capacity(distinct_r.len());
    let mut r_num_matrices: Vec<usize> = Vec::with_capacity(distinct_r.len());
    let mut need_cs: usize = 0;
    let mut need_mk: usize = 0;
    for &(rd, ridx) in &distinct_r {
        let r = algebra.basis_element_from_index(rd, ridx);
        assert!(!r.p_part.is_empty(), "each R must be non-empty");
        let info = resident_info(algebra, &r.p_part);
        r_cs_offset.push(info.cs_off);
        r_mk_offset.push(info.mk_off);
        r_cs_len.push(info.cs_len);
        r_mk_len.push(info.mk_len);
        r_num_matrices.push(info.num_mats as usize);
        need_cs = need_cs.max(info.cs_off as usize + info.num_mats as usize * info.cs_len as usize);
        need_mk = need_mk.max(info.mk_off as usize + info.num_mats as usize * info.mk_len as usize);
    }

    // Lay out per-product term data + records + the pair-count prefix sum (sequential).
    let mut term_pparts: Vec<u16> = Vec::new();
    let mut term_lens: Vec<u32> = Vec::new();
    let mut prod_term_start: Vec<u32> = Vec::with_capacity(products.len());
    let mut prod_num_terms: Vec<u32> = Vec::with_capacity(products.len());
    let mut prod_row_base: Vec<u32> = Vec::with_capacity(products.len());
    let mut prod_out_offset: Vec<u32> = Vec::with_capacity(products.len());
    // The pair prefix sum: entry `pi` is the number of `(matrix, term)` pairs before product
    // `pi`, with the sentinel total at the end — the kernel binary-searches it to decode its
    // thread index. The caller splits blocks near [`GPU_PAIR_CHUNK`], so every entry fits
    // `u32` (a lone over-budget row can exceed the target but stays far below the kernel's
    // `2^32` `ABSOLUTE_POS` limit; asserted below before the values are used).
    let mut pps: Vec<u32> = Vec::with_capacity(products.len() + 1);
    let mut pair_acc: usize = 0;
    for (pi, (tp, tl)) in per_prod.iter().enumerate() {
        let prod = &products[pi];
        let ri = prod_r_index[pi];
        prod_term_start.push(term_lens.len() as u32);
        term_lens.extend_from_slice(tl);
        term_pparts.extend_from_slice(tp);
        pps.push(pair_acc as u32);
        pair_acc += r_num_matrices[ri as usize] * prod.term_indices.len();
        prod_num_terms.push(prod.term_indices.len() as u32);
        prod_row_base.push(((prod.row - row_base) * num_limbs) as u32);
        prod_out_offset.push(prod.out_offset as u32);
    }

    let total_pairs = pair_acc;
    assert!(
        u32::try_from(total_pairs).is_ok(),
        "block pair count {total_pairs} exceeds the kernel's u32 thread limit"
    );
    pps.push(total_pairs as u32);
    let out_len = num_rows * num_limbs;
    if std::env::var_os("NASSAU_GPU_DEBUG").is_some() {
        eprintln!(
            "[gpu-batch] row_base={row_base} num_rows={num_rows} num_cols={num_cols} \
             num_limbs={num_limbs} products={} total_pairs={total_pairs} out_len={out_len}",
            products.len(),
        );
    }
    if total_pairs == 0 {
        return vec![vec![0u32; num_limbs]; num_rows];
    }

    // The resident `col_sums`/`masks` are non-empty once any `R` is present (guaranteed
    // here, since `total_pairs > 0`); only `term_pparts` needs the non-empty guard.
    if term_pparts.is_empty() {
        term_pparts.push(0);
    }

    let marshal_ms = t_marshal.elapsed().as_secs_f64() * 1e3;

    let t_device = std::time::Instant::now();

    // Device section pinned to this permit's stream slot: up to `NASSAU_GPU_CONCURRENCY`
    // launches overlap on distinct streams, but no more streams (and hence retained pools)
    // than that ever exist — see [`GpuPermit`]. Cubecl's per-device runner serializes the
    // actual server access; cross-slot/cross-thread reuse of the shared resident handles is
    // event-synced by cubecl. `memory_cleanup` below trims only this slot's own pool.
    let result = StreamId {
        value: permit.slot as u64,
    }
    .executes(|| {
        let client = CudaRuntime::client(&CudaDevice::default());
        // Shared resident admissible buffers (see [`ResidentDev`]): re-upload the master ONLY
        // when this block dereferences past the uploaded prefix (`need_cs` / `need_mk`). The
        // master grows continually at the frontier, so re-uploading on mere growth ships
        // multi-GB uploads under this mutex on nearly every launch (measured 1.5x wall
        // regression); most launches touch only long-uploaded low-degree `R`s and reuse the
        // stale handle at its uploaded length. When an upload does fire it captures the full
        // current master, amortizing all growth since the last one. Lock order is DEV.lock
        // then HOST.read, and nothing under either lock blocks on rayon or a permit. The
        // master is append-only, so the uploaded prefix is always a prefix of the current
        // host master and every offset `< uploaded` is final.
        let (cs_h, mk_h, cs_len_master, mk_len_master) = {
            let mut dev = RESIDENT_DEV.lock().unwrap();
            if dev.cs_handle.is_none() || dev.cs_uploaded < need_cs {
                let host = RESIDENT_HOST.read().unwrap();
                dev.cs_handle = Some(client.create_from_slice(u16::as_bytes(&host.col_sums)));
                dev.cs_uploaded = host.col_sums.len();
            }
            if dev.mk_handle.is_none() || dev.mk_uploaded < need_mk {
                let host = RESIDENT_HOST.read().unwrap();
                dev.mk_handle = Some(client.create_from_slice(u16::as_bytes(&host.masks)));
                dev.mk_uploaded = host.masks.len();
            }
            (
                dev.cs_handle.clone().unwrap(),
                dev.mk_handle.clone().unwrap(),
                dev.cs_uploaded,
                dev.mk_uploaded,
            )
        };
        // Upload the block's data — term data, seqno/xi tables, per-`R` offsets, per-product
        // records, the pair prefix sum, and the (zeroed) output buffer — and launch once: the
        // caller has already bounded this block's pair count and output size.
        let tp_h = client.create_from_slice(u16::as_bytes(&term_pparts));
        let tl_h = client.create_from_slice(u32::as_bytes(&term_lens));
        let g_h = client.create_from_slice(u32::as_bytes(&g));
        let xi_h = client.create_from_slice(u32::as_bytes(&xi));
        let rco_h = client.create_from_slice(u32::as_bytes(&r_cs_offset));
        let rmo_h = client.create_from_slice(u32::as_bytes(&r_mk_offset));
        let rcl_h = client.create_from_slice(u32::as_bytes(&r_cs_len));
        let rml_h = client.create_from_slice(u32::as_bytes(&r_mk_len));
        let zeros = vec![0u32; out_len];
        let out_h = client.create_from_slice(u32::as_bytes(&zeros));
        const THREADS: u32 = 256;

        let pri_h = client.create_from_slice(u32::as_bytes(&prod_r_index));
        let pts_h = client.create_from_slice(u32::as_bytes(&prod_term_start));
        let pnt_h = client.create_from_slice(u32::as_bytes(&prod_num_terms));
        let prb_h = client.create_from_slice(u32::as_bytes(&prod_row_base));
        let poo_h = client.create_from_slice(u32::as_bytes(&prod_out_offset));
        let pps_h = client.create_from_slice(u32::as_bytes(&pps));
        let cubes = (total_pairs as u32).div_ceil(THREADS).max(1);
        unsafe {
            multiply_batch_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(cs_h, cs_len_master),
                ArrayArg::from_raw_parts(mk_h, mk_len_master),
                ArrayArg::from_raw_parts(tp_h, term_pparts.len()),
                ArrayArg::from_raw_parts(tl_h, term_lens.len()),
                ArrayArg::from_raw_parts(g_h, g.len()),
                ArrayArg::from_raw_parts(xi_h, xi.len()),
                ArrayArg::from_raw_parts(out_h.clone(), out_len),
                ArrayArg::from_raw_parts(rco_h, r_cs_offset.len()),
                ArrayArg::from_raw_parts(rmo_h, r_mk_offset.len()),
                ArrayArg::from_raw_parts(rcl_h, r_cs_len.len()),
                ArrayArg::from_raw_parts(rml_h, r_mk_len.len()),
                ArrayArg::from_raw_parts(pri_h, products.len()),
                ArrayArg::from_raw_parts(pts_h, products.len()),
                ArrayArg::from_raw_parts(pnt_h, products.len()),
                ArrayArg::from_raw_parts(prb_h, products.len()),
                ArrayArg::from_raw_parts(poo_h, products.len()),
                ArrayArg::from_raw_parts(pps_h, pps.len()),
                width,
            );
        }

        let bytes = client.read_one(out_h).unwrap();
        let flat = u32::from_bytes(&bytes);
        let result: Vec<Vec<u32>> = (0..num_rows)
            .map(|r| flat[r * num_limbs..(r + 1) * num_limbs].to_vec())
            .collect();

        // `out_h` alone is `num_rows × num_limbs` u32 — hundreds of MB at record degrees.
        // It (and the small per-launch buffers, now dropped) varies in size launch to
        // launch, so CubeCL's pool cannot reuse the slab and would accumulate them until
        // the 4 GB card OOMs. Return the freed memory to the driver each launch; the
        // resident admissible handles stay alive (refcount > 0) so cleanup skips them.
        client.memory_cleanup();

        result
    });

    // Aggregate marshal/device totals across every launch (cheap, always on) so a whole
    // resolution's GPU overhead can be split host-vs-device via [`take_batch_stats`].
    let device_ms = t_device.elapsed().as_secs_f64() * 1e3;
    BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    BATCH_MARSHAL_US.fetch_add(
        (marshal_ms * 1e3) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    BATCH_DEVICE_US.fetch_add(
        (device_ms * 1e3) as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    BATCH_PAIRS.fetch_add(total_pairs as u64, std::sync::atomic::Ordering::Relaxed);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test proving the CubeCL `cuda` runtime launches and returns correct
    /// results. Requires a live GPU + the CUDA toolkit env (run under the `gpu`
    /// dev shell, unsandboxed).
    #[test]
    fn xor_f2_matches_host() {
        let a: Vec<u32> = (0..1000u32).map(|i| i.wrapping_mul(2654435761)).collect();
        let b: Vec<u32> = (0..1000u32).map(|i| i.wrapping_mul(40503)).collect();
        let expected: Vec<u32> = a.iter().zip(&b).map(|(x, y)| x ^ y).collect();
        assert_eq!(xor_f2_on_gpu(&a, &b), expected);
    }

    /// The device `seqno` must reproduce the CPU basis order exactly: for every
    /// basis element of every degree, `seqno(elt.p_part) == index`. Mirrors the CPU
    /// `seqno_matches_enumeration_order` test on-device. Requires a live GPU + the
    /// CUDA toolkit env (run under the `gpu` dev shell, unsandboxed).
    #[test]
    fn seqno_matches_index_on_gpu() {
        use fp::prime::ValidPrime;

        let p = ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);
        let max_degree = 60;
        algebra.compute_basis(max_degree);
        algebra.compute_seqno_tables(max_degree);

        let (width, g) = algebra.seqno_table_u32();
        assert_eq!(width, MAX_XI_TAU);
        let xi: Vec<u32> = xi_degrees(p).iter().map(|&x| x as u32).collect();

        // Marshal every basis element, padded to `width`; the expected seqno is the
        // element's own index (the identity permutation the CPU proves).
        let mut p_parts = Vec::new();
        let mut expected = Vec::new();
        for d in 0..=max_degree {
            let dim = algebra.dimension(d);
            for i in 0..dim {
                let elt = algebra.basis_element_from_index(d, i);
                let mut row = vec![0u32; width];
                for (slot, &v) in row.iter_mut().zip(&elt.p_part) {
                    *slot = v;
                }
                p_parts.extend_from_slice(&row);
                expected.push(i as u32);
            }
        }

        let n = expected.len();
        let got = seqno_batch_on_gpu(width, &xi, &g, &p_parts, n);
        assert_eq!(got, expected, "device seqno diverged from CPU basis order");
    }

    /// The single-`R` multiply kernel must match the CPU reference
    /// `multiply_basis_element_by_element_2` bit-for-bit. For many `(R, s)` with `R`
    /// non-empty and `s` the dense (all-ones) element — exercising the admissible
    /// path and mod-2 cancellation — compare the GPU's packed F₂ output to the CPU's.
    /// Requires a live GPU + the CUDA toolkit env (run under the `gpu` dev shell).
    #[test]
    fn multiply_single_r_matches_reference() {
        use fp::{prime::ValidPrime, vector::FpVector};

        let p = ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);
        let max_degree = 40;
        algebra.compute_basis(max_degree);
        algebra.compute_seqno_tables(max_degree);

        let mut checked = 0usize;
        for r_degree in 1..=12 {
            let r_dim = algebra.dimension(r_degree);
            for r_idx in 0..r_dim {
                if algebra
                    .basis_element_from_index(r_degree, r_idx)
                    .p_part
                    .is_empty()
                {
                    continue; // Sq(∅) = 1 is handled separately
                }
                for s_degree in 1..=(max_degree - r_degree) {
                    let s_dim = algebra.dimension(s_degree);
                    if s_dim == 0 {
                        continue;
                    }
                    let out_dim = algebra.dimension(r_degree + s_degree);

                    // s = dense (all basis elements): multi-term, mod-2 cancellation.
                    let mut s = FpVector::new(p, s_dim);
                    for j in 0..s_dim {
                        s.set_entry(j, 1);
                    }
                    let mut cpu = FpVector::new(p, out_dim);
                    algebra.multiply_basis_element_by_element_2(
                        cpu.as_slice_mut(),
                        1,
                        r_degree,
                        r_idx,
                        s_degree,
                        s.as_slice(),
                    );
                    let num_limbs = out_dim.div_ceil(32).max(1);
                    let mut golden = vec![0u32; num_limbs];
                    for (i, _) in cpu.iter_nonzero() {
                        golden[i / 32] ^= 1u32 << (i % 32);
                    }

                    let term_indices: Vec<usize> = (0..s_dim).collect();
                    let got = multiply_single_r_on_gpu(
                        &algebra,
                        r_degree,
                        r_idx,
                        s_degree,
                        &term_indices,
                    );
                    assert_eq!(
                        got, golden,
                        "GPU multiply diverged from reference: R(deg {r_degree}, idx {r_idx}) * \
                         dense s(deg {s_degree})",
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no (R, s) cases exercised");
        eprintln!("multiply_single_r: {checked} (R, s) cases matched reference");
    }

    /// The batched kernel must reproduce a whole output matrix: many heterogeneous
    /// `(R, s)` products, several accumulating into the same row (XOR), computed in a
    /// single launch, must equal the CPU reference matrix built product-by-product.
    /// Requires a live GPU + the CUDA toolkit env (run under the `gpu` dev shell).
    #[test]
    fn multiply_batch_matches_reference() {
        use fp::{prime::ValidPrime, vector::FpVector};

        let p = ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);
        let max_degree = 40;
        algebra.compute_basis(max_degree);
        algebra.compute_seqno_tables(max_degree);

        let out_degree = 24;
        let out_dim = algebra.dimension(out_degree);
        let num_rows = 8;

        // Products: every non-empty R of degree 1..out_degree, s dense at the
        // complementary degree, assigned round-robin to rows so rows accumulate.
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
                let row = products.len() % num_rows;
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

        // CPU golden matrix: accumulate each product into its row.
        let mut cpu_rows: Vec<FpVector> =
            (0..num_rows).map(|_| FpVector::new(p, out_dim)).collect();
        for prod in &products {
            let s_dim = algebra.dimension(prod.s_degree);
            let mut s = FpVector::new(p, s_dim);
            for &ti in &prod.term_indices {
                s.set_entry(ti, 1);
            }
            let mut tmp = FpVector::new(p, out_dim);
            algebra.multiply_basis_element_by_element_2(
                tmp.as_slice_mut(),
                1,
                prod.r_degree,
                prod.r_idx,
                prod.s_degree,
                s.as_slice(),
            );
            cpu_rows[prod.row].add(&tmp, 1);
        }
        let num_limbs = out_dim.div_ceil(32).max(1);
        let golden: Vec<Vec<u32>> = cpu_rows
            .iter()
            .map(|row| {
                let mut packed = vec![0u32; num_limbs];
                for (i, _) in row.iter_nonzero() {
                    packed[i / 32] ^= 1u32 << (i % 32);
                }
                packed
            })
            .collect();

        let got = multiply_batch_on_gpu(&algebra, out_dim, num_rows, &products);
        assert_eq!(
            got, golden,
            "batched GPU multiply diverged from reference matrix"
        );
        eprintln!(
            "multiply_batch: {} products across {num_rows} rows matched reference",
            products.len()
        );
    }
}
