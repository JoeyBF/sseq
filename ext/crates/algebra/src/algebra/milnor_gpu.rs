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

/// Target `(product, matrix, term)` thread-pairs per GPU launch. The batch multiply indexes threads
/// by CubeCL's `ABSOLUTE_POS` (a `u32`), so one launch can address at most `2^32` threads; a single
/// all-rows reuse build reaches ~4.4e9 pairs at stem ~145, past that limit. The row-block splitter
/// in [`multiply_batch_on_gpu`] closes a block once its pair count would pass this target (alongside
/// the [`gpu_block_bytes`] output budget).
///
/// Set close to the `2^32` ceiling, not far below it: every extra split is a whole extra launch
/// (upload + kernel + blocking readback), and at record stems this — not the byte budget — is the
/// binding constraint, so a conservative value chops each giant multiply into several
/// otherwise-unnecessary launches (measured: `1 << 30` pegged the giants at ~1.07e9 pairs, ~4
/// launches each, while their output is only ~350 MB, well under `gpu_block_bytes`). `3.9e9` leaves
/// ~0.39e9 of headroom under `2^32` for a lone over-budget row (the splitter always takes ≥1 row,
/// and a single row past `2^32` still trips the per-block `u32::try_from` assert), and keeps the
/// grid (`pairs / 256` cubes ≈ 1.5e7) far under CUDA's `2^31 - 1` grid-dimension limit.
const GPU_PAIR_CHUNK: usize = 3_900_000_000;

/// Per-launch output-buffer budget in bytes (`NASSAU_GPU_BLOCK_MB`, default 512 MiB).
///
/// A launch's transient footprint — host marshal buffers, pinned staging, device buffers, and
/// each stream's retained pool pages — scales with its output size, and with the device mutex
/// gone many workers hold such transients simultaneously; at record stems an unbounded all-rows
/// reuse build multiplies to >100 GB on both host and device. [`multiply_batch_on_gpu`] therefore
/// splits large builds into row blocks whose output buffer stays under this budget. Rows of
/// distinct products are independent (each product writes only its own row), so blocks simply
/// concatenate — the same in-between as the old per-signature builds, but with blocks big enough
/// to keep the launch amortization. Together with [`GPU_BUDGET`] this makes peak transient
/// memory a configured constant (≈ the byte budget) instead of a function of the frontier size.
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

/// Byte-weighted budget bounding the total *output size* of in-flight device sections
/// (`NASSAU_GPU_MEM_BUDGET_MB`, default 4096).
///
/// A count-based cap (formerly `NASSAU_GPU_CONCURRENCY` = 8 sections) throttled exactly the
/// wrong region: low-stem launches are a few MB each and were capped at 8 concurrent (measured
/// 4x slowdown vs the uncapped code at stem 130), while the cap only exists for the
/// multi-hundred-MB frontier blocks. Weighting admission by output bytes admits dozens of
/// small launches concurrently and still bounds the frontier to ~budget / [`gpu_block_bytes`]
/// in flight. A launch heavier than the whole budget is admitted alone (when nothing else is
/// in flight), so progress is always possible. Waiters have heterogeneous weights, so release
/// notifies all.
///
/// SAFETY INVARIANT: a permit must never be held across a rayon parallel section. A par_iter's
/// chunks execute on other threads, which do not carry the holder's thread-local
/// `ParallelGuard` flag and so can steal a resolution-step job mid-chunk; that job would park
/// on [`GpuPermit::acquire`] while the holder's permit waits on the never-finishing join —
/// a cycle (observed as a full stall on H200). [`multiply_batch_block`] therefore acquires its
/// permit only after the parallel marshal, guarding a strictly sequential section: every holder
/// makes progress, so parked acquirers always wake (priority inversion at worst, never
/// deadlock).
struct GpuBudget {
    budget: usize,
    used: Mutex<usize>,
    freed: Condvar,
}

static GPU_BUDGET: LazyLock<GpuBudget> = LazyLock::new(|| GpuBudget {
    budget: std::env::var("NASSAU_GPU_MEM_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&mb| mb > 0)
        .unwrap_or(4096)
        << 20,
    used: Mutex::new(0),
    freed: Condvar::new(),
});

/// Number of distinct CUDA streams to spread device work over (`NASSAU_GPU_STREAMS`, default 1).
/// Each worker thread gets a stable per-thread stream id (see [`thread_stream_id`]), and this caps
/// how many distinct streams — hence retained memory pools — exist; `1` forces the single-stream
/// mode (all threads → stream 0).
///
/// **Default 1 (single stream) because multi-stream is a MEMORY disaster for little gain.** cubecl
/// gives each CUDA stream its own device AND page-locked host (pinned) memory pool, and those pools
/// are never trimmed (`memory_cleanup` frees only the GPU pool). Measured at stem 180: single-stream
/// holds pinned host memory at ~4 GB, but ≥2 streams balloons it to 140–240 GB (each stream retains
/// its own varying-size readback/staging pages) — the dominant term in the ~500 GB cgroup OOM. And
/// the payoff is ~nil: cubecl's server is single-threaded (one runner behind a channel), so extra
/// streams buy no CPU-side concurrency, only GPU kernel overlap that the big saturating multiplies
/// barely benefit from. Raise it only on a dedicated large-RAM node that wants that overlap.
fn gpu_stream_slots() -> u64 {
    static SLOTS: LazyLock<u64> = LazyLock::new(|| {
        std::env::var("NASSAU_GPU_STREAMS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1)
    });
    *SLOTS
}

/// A CUDA stream id UNIQUE to the calling worker thread (never shared or reused across threads —
/// cubecl's per-stream state assumes one driver thread per stream). Returns 0 for all threads when
/// `NASSAU_GPU_STREAMS == 1` (the single-stream fallback). With the resident master/basis now stable
/// device buffers grown IN PLACE (no handle churn — see [`ResidentDev`]), cross-stream reads of those
/// shared globals are the ordinary case cubecl supports, so distinct per-thread streams are safe and
/// give real device concurrency for the wide record-run wavefront.
fn thread_stream_id() -> u64 {
    if gpu_stream_slots() == 1 {
        return 0;
    }
    thread_local! {
        static ID: u64 = {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            NEXT.fetch_add(1, Ordering::Relaxed) + 1 // +1: id 0 is reserved for the single-stream mode
        };
    }
    ID.with(|&id| id)
}

/// A/B diagnostic toggle (`NASSAU_GPU_BASIS_PASSTHROUGH=1`): when set, the batched multiply
/// marshals each term's p-part per launch and binds those buffers as the "basis" with an
/// identity index map, reproducing the pre-resident-basis behaviour through the same kernel.
/// Lets a single binary isolate a kernel-signature bug from a resident-basis host/upload bug.
fn basis_passthrough() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("NASSAU_GPU_BASIS_PASSTHROUGH").is_some());
    *ON
}

/// RAII reservation of `weight` output bytes from [`GPU_BUDGET`]; blocks (parked, not spinning)
/// until the budget admits it. The stream is chosen per worker thread (see [`thread_stream_id`]),
/// so the permit no longer carries a slot.
struct GpuPermit {
    weight: usize,
}

impl GpuPermit {
    fn acquire(weight: usize) -> Self {
        let b = &*GPU_BUDGET;
        let mut used = b.used.lock().unwrap();
        while !(*used == 0 || *used + weight <= b.budget) {
            used = b.freed.wait(used).unwrap();
        }
        *used += weight;
        Self { weight }
    }
}

impl Drop for GpuPermit {
    fn drop(&mut self) {
        let b = &*GPU_BUDGET;
        *b.used.lock().unwrap() -= self.weight;
        b.freed.notify_all();
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

/// Diagnostic (see `NASSAU_MEM_REPORT`): resident-master HOST-side heap bytes — the not-yet-uploaded
/// `col_sums`/`masks` tails, the width-padded basis `pparts`/`lens`, and the per-`R` `index` map
/// (its `Vec<PPartEntry>` keys). The bulk `col_sums`/`masks` are no longer retained (freed after
/// upload — see [`ResidentHost`]); only the pending tail + the `index` persist. Returns `(master, basis)`.
pub fn resident_host_bytes() -> (usize, usize) {
    let h = RESIDENT_HOST.read().unwrap();
    let master = h.cs_pending.capacity() * 2
        + h.mk_pending.capacity() * 2
        + h.index.capacity()
            * (std::mem::size_of::<RInfo>()
                + std::mem::size_of::<Vec<PPartEntry>>()
                + 4 * std::mem::size_of::<PPartEntry>());
    let b = RESIDENT_BASIS_HOST.read().unwrap();
    let basis = b.pparts.capacity() * 2 + b.lens.capacity() * 4 + b.global_base.capacity() * 4;
    (master, basis)
}

/// Diagnostic (see `NASSAU_MEM_REPORT`): DEVICE-side bytes of the resident master (`col_sums`+`masks`,
/// u16) and basis (`pparts` u16 + `lens` u32) — the persistent GPU buffers, from their uploaded
/// element counts. Returns `(master_bytes, basis_bytes)`.
pub fn resident_dev_bytes() -> (usize, usize) {
    let d = RESIDENT_DEV.read().unwrap();
    let master = d.cs.uploaded * 2 + d.mk.uploaded * 2;
    let b = RESIDENT_BASIS_DEV.read().unwrap();
    let basis = b.pp.uploaded * 2 + b.ln.uploaded * 4;
    (master, basis)
}

/// Diagnostic (see `NASSAU_MEM_REPORT`): the cubecl CUDA memory pool's device usage on the default
/// device, `(bytes_in_use, bytes_reserved)`. This is the batched-multiply pool; the fp-cuda RREF runs
/// on a separate cudarc context, so `nvidia-smi total − resident_dev − reserved` estimates the RREF
/// pool. Returns `(0, 0)` if the query fails.
pub fn cubecl_device_usage() -> (u64, u64) {
    let client = CudaRuntime::client(&CudaDevice::default());
    match client.memory_usage() {
        Ok(u) => (u.bytes_in_use, u.bytes_reserved),
        Err(_) => (0, 0),
    }
}

use std::{
    collections::HashMap,
    sync::{Arc, Condvar, LazyLock, Mutex, RwLock},
};

use cubecl::server::Handle;

use crate::algebra::milnor_algebra::PPartEntry;

/// Where one `R`'s admissible-matrix data lives inside the resident master buffers.
#[derive(Clone, Copy)]
struct RInfo {
    /// Global element offsets into the shared master (`u64`: the master exceeds `u32::MAX`
    /// u16 elements at high stems, so the offset itself must be 64-bit — `multiply_batch_kernel`
    /// reads `col_sums`/`masks` at these offsets under 64-bit `address_type`).
    cs_off: u64,
    mk_off: u64,
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
/// The host master keeps ONLY the not-yet-uploaded tail (`*_pending`) plus a logical length
/// (`*_len`), never the full `col_sums`/`masks`. Once an `R`'s admissible data is copied to the
/// device it is dropped host-side — it is provably never read again (offsets come from `index`;
/// growth uploads only the pending tail; a capacity realloc copies the old *device* buffer, not the
/// host). This removes the multi-GB host↔device duplicate that dominated the resolver's anon RSS
/// (~27 GB at stem 130, growing). Invariant maintained by [`resident_dev_handle`]:
/// `RESIDENT_DEV.$buf.uploaded == $len - $pending.len()`, i.e. `$pending == master[uploaded..$len]`.
#[derive(Default)]
struct ResidentHost {
    cs_pending: Vec<u16>,
    mk_pending: Vec<u16>,
    cs_len: usize,
    mk_len: usize,
    index: HashMap<Vec<PPartEntry>, RInfo>,
}

static RESIDENT_HOST: LazyLock<RwLock<ResidentHost>> =
    LazyLock::new(|| RwLock::new(ResidentHost::default()));

/// A resident device buffer that GROWS IN PLACE: a single stable handle, allocated once
/// (persistent, so per-launch cleanup never reindexes it) with headroom and written to at its
/// append offset as the host master grows — never re-`create_from_slice`d. The handle changes only
/// on a rare capacity doubling. `uploaded` is how many elements have been written; `cap` is the
/// allocated element capacity.
#[derive(Default)]
struct GrowBuf {
    handle: Option<Handle>,
    cap: usize,
    uploaded: usize,
}

/// Process-shared device mirror of the host master. Each buffer is a STABLE handle grown in place
/// (see [`GrowBuf`], [`resident_dev_handle`]). This is what makes the master safe to share across
/// streams: the churny "re-upload a new handle on every growth" it replaced broke cubecl's
/// per-handle cross-stream sync (crash rate tracked re-upload frequency); a fixed handle written in
/// place is the ordinary shared-global (model-weights) pattern. Reads go through `RESIDENT_DEV.read()`
/// (lock-free fan-out); growth runs outside that lock, serialized only by `RESIDENT_UPLOAD`.
#[derive(Default)]
struct ResidentDev {
    cs: GrowBuf,
    mk: GrowBuf,
}

static RESIDENT_DEV: LazyLock<RwLock<ResidentDev>> =
    LazyLock::new(|| RwLock::new(ResidentDev::default()));

/// Initial element capacity of a resident buffer. Doubling to grow past it is barrier-protected
/// (see [`RESIDENT_REALLOC`]) so it is correct, but each doubling briefly quiesces the device, so
/// the initial size carries headroom to keep them few (~a handful per run). u16 → ~512 MiB.
const RESIDENT_INIT_CAP: usize = 1 << 28;

/// Serializes master device *uploads* only — never handle reads. A launch that must grow the
/// device master takes this before uploading, so at a growth point at most one multi-GB
/// `create_from_slice` runs (others re-check and find it already done) instead of every launch
/// piling redundant copies. Reads go lock-free through `RESIDENT_DEV.read()`, so the upload no
/// longer blocks other bidegrees' device sections (the old single mutex held across the copy
/// collapsed the whole wavefront to one memcpy-ing thread).
static RESIDENT_UPLOAD: Mutex<()> = Mutex::new(());

/// Guards the rare capacity-doubling REALLOC (the only time a resident handle changes) against
/// concurrent readers. A device section holds the read lock while its multiply kernel is reading the
/// resident buffers; a realloc takes the write lock, so it waits for every in-flight reader to drain
/// and blocks new ones — the swap + old-buffer free then happens with the device quiesced, which the
/// churny path could not guarantee (residual ~1/10 crash at the readback / a stale multiply). The
/// common in-place delta write does NOT take this lock (it's stable-handle, proven race-free), so the
/// hot path stays lock-free; reallocs happen only a handful of times per run, so the barrier is free.
static RESIDENT_REALLOC: RwLock<()> = RwLock::new(());

/// Host-side cache of cold (degree > [`resident_degree_cap`]) `R`s' admissible matrices, narrowed to
/// `u16` and ready to upload. Cold `R`s are deliberately kept OFF the device master (that is the whole
/// point of eviction — it bounds the 143 GB device), but `admissible_matrices` is the *expensive* part
/// for exactly these high-degree `R`s (big matrices) and they recur across many bidegrees. Recomputing
/// per launch collapsed throughput (2× wall at stem 180, bench 2026-07-27). Caching the result
/// host-side (host has ~755 GB — never the constraint) makes the recompute one-shot, like the resident
/// path, while the *device* copy stays transient (uploaded per launch into a block-local buffer, freed
/// after). Grows to roughly the evicted tail of the master (tens of GB), well within host RAM.
struct ColdEntry {
    cs_len: u32,
    mk_len: u32,
    num_mats: u32,
    cs: Arc<Vec<u16>>,
    mk: Arc<Vec<u16>>,
}
static COLD_HOST: LazyLock<RwLock<HashMap<Vec<PPartEntry>, ColdEntry>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Cold-`R` admissible data, from the [`COLD_HOST`] cache (computed + narrowed once on first use).
/// Returns `(cs_len, mk_len, num_mats, cs, mk)`; the `Arc`s make the cache-hit path a cheap refcount
/// bump, no copy. Layout matches [`resident_info`]'s so the kernel indexes both identically.
fn cold_host_entry(
    algebra: &MilnorAlgebra,
    p_part: &[PPartEntry],
) -> (u32, u32, u32, Arc<Vec<u16>>, Arc<Vec<u16>>) {
    if let Some(e) = COLD_HOST.read().unwrap().get(p_part) {
        return (e.cs_len, e.mk_len, e.num_mats, e.cs.clone(), e.mk.clone());
    }
    let (cs_len, mk_len, cs, mk) = algebra.admissible_matrices(p_part);
    let num_mats = (mk.len() / mk_len) as u32;
    let entry = ColdEntry {
        cs_len: cs_len as u32,
        mk_len: mk_len as u32,
        num_mats,
        cs: Arc::new(cs.iter().map(|&v| narrow_u16(v)).collect()),
        mk: Arc::new(mk.iter().map(|&v| narrow_u16(v)).collect()),
    };
    let mut w = COLD_HOST.write().unwrap();
    let e = w.entry(p_part.to_vec()).or_insert(entry);
    (e.cs_len, e.mk_len, e.num_mats, e.cs.clone(), e.mk.clone())
}

/// Fetch a resident device-master handle, uploading the current master prefix only when this
/// block dereferences past what is already on the device (`need`). The upload runs OUTSIDE
/// `RESIDENT_DEV` (readers stay lock-free; only concurrent uploaders serialize, on
/// `RESIDENT_UPLOAD`, and a re-check coalesces a burst of growth-needing launches into one
/// upload). The master is append-only, so any handle with `uploaded >= need` is valid.
macro_rules! resident_dev_handle {
    ($client:expr, $need:expr, $buf:ident, $pending:ident, $len:ident) => {{
        // Lock-free fast path: the stable handle already covers `$need`.
        let read_current = || {
            let dev = RESIDENT_DEV.read().unwrap();
            match (dev.$buf.uploaded >= $need, dev.$buf.handle.clone()) {
                (true, Some(h)) => Some((h, dev.$buf.uploaded)),
                _ => None,
            }
        };
        match read_current() {
            Some(hu) => hu,
            None => {
                let _upload_guard = RESIDENT_UPLOAD.lock().unwrap();
                match read_current() {
                    Some(hu) => hu, // another grower already covered our need
                    None => {
                        let (old_handle, cap, uploaded) = {
                            let dev = RESIDENT_DEV.read().unwrap();
                            (dev.$buf.handle.clone(), dev.$buf.cap, dev.$buf.uploaded)
                        };
                        // Take the not-yet-uploaded tail and the logical length together. By the
                        // invariant (see [`ResidentHost`]) the tail is exactly `master[uploaded..len]`,
                        // so `uploaded + batch.len() == host_len`. `mem::take` FREES it host-side —
                        // once on the device it is never read again. Appends after this point land in
                        // a fresh `$pending` and upload on the next growth. `RESIDENT_UPLOAD` (held)
                        // makes this the sole grower, so `uploaded` is stable here.
                        let (batch, host_len) = {
                            let mut host = RESIDENT_HOST.write().unwrap();
                            (std::mem::take(&mut host.$pending), host.$len)
                        };
                        debug_assert_eq!(uploaded + batch.len(), host_len);

                        // (Re)allocate a stable persistent buffer on first use or capacity overflow
                        // (rare: `RESIDENT_INIT_CAP` has headroom and the master saturates early),
                        // copying existing device data on-device. Only THIS handle-swap needs a sync,
                        // so a fresh-handle cross-stream read never sees an incomplete copy.
                        let (handle, cap) = if old_handle.is_none() || host_len > cap {
                            // Quiesce readers for the handle swap (see [`RESIDENT_REALLOC`]).
                            let _realloc_w = RESIDENT_REALLOC.write().unwrap();
                            // No `u32` cap: the kernels bind these buffers with 64-bit addressing
                            // (`multiply_batch_kernel` reads with static `u64`; the grow copy with
                            // dynamic), so the length/offsets are never truncated past `u32::MAX`.
                            let new_cap = host_len.max(cap * 2).max(RESIDENT_INIT_CAP);
                            let new_handle = $client.empty(new_cap * ::core::mem::size_of::<u16>());
                            if let Some(oh) = &old_handle {
                                if uploaded > 0 {
                                    copy_chunked!(
                                        $client, copy_into_u16, oh, uploaded, 0usize,
                                        new_handle, new_cap, 0usize, uploaded
                                    );
                                }
                            }
                            let _ = cubecl_common::reader::read_sync($client.sync());
                            (new_handle, new_cap)
                        } else {
                            (old_handle.unwrap(), cap)
                        };

                        // Write the tail `batch` (== master[uploaded..host_len]) into the STABLE
                        // buffer at its append offset through a BOUNDED pinned staging (see
                        // [`STAGE_CHUNK`]), syncing each chunk. The sync also serves the cross-stream
                        // ordering the resident buffers need: cubecl does not order a *kernel write* to
                        // a shared buffer against reads on another stream the way it does
                        // `create_from_slice`, so blocking here makes the new prefix physically
                        // resident before any reader observes the bumped `uploaded`.
                        let mut done = 0usize;
                        while done < batch.len() {
                            let m = (batch.len() - done).min(STAGE_CHUNK);
                            let lo = uploaded + done;
                            let scratch =
                                $client.create_from_slice(u16::as_bytes(&batch[done..done + m]));
                            copy_chunked!(
                                $client, copy_into_u16, scratch, m, 0usize, handle, cap, lo, m
                            );
                            let _ = cubecl_common::reader::read_sync($client.sync());
                            done += m;
                        }

                        {
                            let mut dev = RESIDENT_DEV.write().unwrap();
                            dev.$buf.handle = Some(handle.clone());
                            dev.$buf.cap = cap;
                            dev.$buf.uploaded = host_len;
                        }
                        (handle, host_len)
                    }
                }
            }
        }
    }};
}

/// Global offsets/lengths of `R`'s admissible matrices in the shared host master (see
/// [`ResidentHost`]), enumerating and appending them on first sight (the append order fixes
/// the offsets forever). The enumeration runs outside any lock; on a first-sight race the
/// loser rechecks under the write lock and discards its duplicate.
/// Per-`R` access statistics for the eviction probe (`NASSAU_R_STATS`): how often each distinct `R`
/// is referenced (a block that uses it counts once), its degree, and the first/last reference "time"
/// (a `BATCH_CALLS` tick). Dumped by [`dump_r_stats`] to reveal the hot/cold structure that a device
/// working-set cache would exploit.
#[derive(Clone)]
struct RStat {
    count: u64,
    degree: i32,
    first: u64,
    last: u64,
}

static R_STATS: LazyLock<Option<Mutex<HashMap<Vec<PPartEntry>, RStat>>>> = LazyLock::new(|| {
    std::env::var_os("NASSAU_R_STATS").map(|_| Mutex::new(HashMap::new()))
});

/// Internal degree of `R` from its p-part: `Σ p_part[i] · deg(ξ_{i+1})`.
fn ppart_degree(p_part: &[PPartEntry]) -> i32 {
    let xi = xi_degrees(fp::prime::ValidPrime::new(2));
    p_part
        .iter()
        .zip(xi.iter())
        .map(|(&e, &d)| e as i32 * d as i32)
        .sum()
}

/// Operations `R` whose internal degree exceeds this stay OUT of the resident device master and are
/// instead recomputed and uploaded to a throwaway per-launch buffer (see [`MasterMode`]). Default
/// `i32::MAX` keeps every `R` resident — byte-identical to the pre-eviction path (the caller takes a
/// fast path that never touches the transient code). The `NASSAU_R_STATS` probe found a *degree*
/// threshold, not LRU, is the right policy: low-degree `R`s are the stable hot core (reference span
/// 0.99 of the run), high-degree `R`s are scattered-recurring (0.81) *and* the biggest matrices, so
/// excluding them saves more device bytes than their count fraction. On S_2 (150,75): θ≤100 keeps
/// 17% of distinct `R`s resident and recomputes 14% of references; θ≤125 keeps 43% / recomputes 4%.
/// The resident set saturates with degree, so this bounds the master at any stem (the stem-300 lever).
fn resident_degree_cap() -> i32 {
    static CAP: LazyLock<i32> = LazyLock::new(|| {
        std::env::var("NASSAU_GPU_RESIDENT_MAX_DEGREE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(i32::MAX)
    });
    *CAP
}

/// Where a launch's `col_sums`/`masks` come from. A given output row's products all share one `R`
/// (the operation of its input basis element), so rows split cleanly by `R`-degree into a resident
/// group and a transient group with no row overlap — the two launches write disjoint rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MasterMode {
    /// Persist each `R`'s admissible matrices in the shared append-only device master ([`ResidentDev`]).
    Resident,
    /// Recompute this block's `R`s ([`MilnorAlgebra::admissible_matrices`]) into a per-block buffer,
    /// uploaded fresh and freed with the launch. Keeps the resident master bounded across stems.
    Transient,
}

fn record_r_use(p_part: &[PPartEntry]) {
    if let Some(m) = R_STATS.as_ref() {
        let now = BATCH_CALLS.load(Ordering::Relaxed);
        let mut map = m.lock().unwrap();
        let e = map.entry(p_part.to_vec()).or_insert(RStat {
            count: 0,
            degree: ppart_degree(p_part),
            first: now,
            last: now,
        });
        e.count += 1;
        e.last = now;
    }
}

/// Dump the `R`-access distribution gathered under `NASSAU_R_STATS` (see [`RStat`]) — the data that
/// decides whether/what device eviction policy helps. Prints: reference skew (top-k coverage),
/// degree-vs-frequency correlation, and reference-lifetime spans. No-op unless the probe is enabled.
pub fn dump_r_stats() {
    let Some(m) = R_STATS.as_ref() else { return };
    let map = m.lock().unwrap();
    if map.is_empty() {
        return;
    }
    let n = map.len();
    let total_refs: u64 = map.values().map(|s| s.count).sum();
    let now = BATCH_CALLS.load(Ordering::Relaxed).max(1);
    let mut v: Vec<&RStat> = map.values().collect();
    // Coverage: sort by count desc, cumulative fraction of references from the top-k Rs.
    v.sort_by(|a, b| b.count.cmp(&a.count));
    let cov = |frac: f64| -> f64 {
        let k = ((n as f64 * frac).ceil() as usize).max(1).min(n);
        let hit: u64 = v[..k].iter().map(|s| s.count).sum();
        hit as f64 / total_refs as f64 * 100.0
    };
    // used-once fraction (pure cold), and how many Rs cover 90% of refs.
    let once = v.iter().filter(|s| s.count == 1).count();
    let mut acc = 0u64;
    let mut k90 = 0usize;
    for s in &v {
        acc += s.count;
        k90 += 1;
        if acc as f64 >= total_refs as f64 * 0.90 {
            break;
        }
    }
    // Degree vs frequency: avg degree of the hottest decile vs the coldest decile.
    let dec = (n / 10).max(1);
    let avg_deg = |slice: &[&RStat]| -> f64 {
        slice.iter().map(|s| s.degree as f64).sum::<f64>() / slice.len().max(1) as f64
    };
    let hot_deg = avg_deg(&v[..dec]);
    let cold_deg = avg_deg(&v[n - dec..]);
    // Reference lifetime: span (last-first)/now for the hot decile (are they used throughout, or windowed?).
    let hot_span = v[..dec]
        .iter()
        .map(|s| (s.last - s.first) as f64 / now as f64)
        .sum::<f64>()
        / dec as f64;
    let cold_span = v[n - dec..]
        .iter()
        .map(|s| (s.last - s.first) as f64 / now as f64)
        .sum::<f64>()
        / dec as f64;
    let max_deg = v.iter().map(|s| s.degree).max().unwrap_or(0);
    let min_deg = v.iter().map(|s| s.degree).min().unwrap_or(0);
    // Degree-threshold sizing: for a resident cache that keeps Rs with degree <= θ, what fraction of
    // distinct Rs it holds and what fraction of references hit it (miss rate = 100 − ref%).
    let mut deg_table = String::new();
    for theta in [50, 75, 100, 125] {
        let held = v.iter().filter(|s| s.degree <= theta).count();
        let refs: u64 = v.iter().filter(|s| s.degree <= theta).map(|s| s.count).sum();
        deg_table += &format!(
            " θ≤{theta}:[{:.0}%Rs,{:.0}%refs]",
            held as f64 / n as f64 * 100.0,
            refs as f64 / total_refs as f64 * 100.0
        );
    }
    eprintln!(
        "[R-STATS] distinct_R={n} total_refs={total_refs} used_once={once} ({:.0}%) \
         k_for_90%_refs={k90} ({:.1}% of Rs) | coverage top1%={:.0}% top5%={:.0}% top10%={:.0}% top25%={:.0}% \
         | degree hot_decile_avg={:.0} cold_decile_avg={:.0} range=[{min_deg},{max_deg}] \
         | ref_span hot={:.2} cold={:.2} (of run) | degree-threshold cache sizing:{}",
        once as f64 / n as f64 * 100.0,
        k90 as f64 / n as f64 * 100.0,
        cov(0.01),
        cov(0.05),
        cov(0.10),
        cov(0.25),
        hot_deg,
        cold_deg,
        hot_span,
        cold_span,
        deg_table,
    );
}

fn resident_info(algebra: &MilnorAlgebra, p_part: &[PPartEntry]) -> RInfo {
    record_r_use(p_part);
    if let Some(info) = RESIDENT_HOST.read().unwrap().index.get(p_part) {
        return *info;
    }
    let (cs_len, mk_len, cs, mk) = algebra.admissible_matrices(p_part);
    let mut host = RESIDENT_HOST.write().unwrap();
    if let Some(info) = host.index.get(p_part) {
        return *info;
    }
    // Offsets are the running LOGICAL lengths (`*_len`), not the pending-buffer lengths — the
    // uploaded prefix has been freed but the logical numbering is permanent (see [`ResidentHost`]).
    let info = RInfo {
        cs_off: host.cs_len as u64,
        mk_off: host.mk_len as u64,
        cs_len: cs_len as u32,
        mk_len: mk_len as u32,
        num_mats: (mk.len() / mk_len) as u32,
    };
    host.cs_pending.extend(cs.iter().map(|&v| narrow_u16(v)));
    host.mk_pending.extend(mk.iter().map(|&v| narrow_u16(v)));
    host.cs_len += cs.len();
    host.mk_len += mk.len();
    host.index.insert(p_part.to_vec(), info);
    info
}

/// Process-shared host master of the Milnor basis itself, laid out for the device.
///
/// Every basis element's p-part is stored zero-padded to `width` at `pparts[gei*width ..]`,
/// where `gei` is the element's *global* index (elements concatenated in degree order:
/// all of degree 0, then degree 1, …). `lens[gei]` is its true (trimmed) p-part length,
/// and `global_base[d]` is the number of elements in degrees `< d`, so a term `(s_degree,
/// ti)` maps to `gei = global_base[s_degree] + ti`.
///
/// This exists so a launch uploads only the small per-term *index* array (`term_gei`)
/// rather than re-gathering and re-uploading every term's padded p-part every launch — the
/// dominant per-launch H2D transfer. The basis is append-only and grows only when a higher
/// degree first appears, so it is uploaded to the device once and re-uploaded only on growth
/// (mirroring [`ResidentHost`]). `built_degree` is the highest degree fully appended.
#[derive(Default)]
struct ResidentBasisHost {
    pparts: Vec<u16>,
    lens: Vec<u32>,
    global_base: Vec<u32>,
    built_degree: i32,
    width: usize,
}

static RESIDENT_BASIS_HOST: LazyLock<RwLock<ResidentBasisHost>> =
    LazyLock::new(|| RwLock::new(ResidentBasisHost::default()));

/// Device mirror of [`ResidentBasisHost`], both buffers grown IN PLACE (see [`GrowBuf`],
/// [`resident_dev_handle`]). `pp` holds the width-padded p-parts (`elems * width` u16), `ln` the
/// lengths (`elems` u32); the basis element count is `ln.uploaded`.
#[derive(Default)]
struct ResidentBasisDev {
    pp: GrowBuf,
    ln: GrowBuf,
}

static RESIDENT_BASIS_DEV: LazyLock<RwLock<ResidentBasisDev>> =
    LazyLock::new(|| RwLock::new(ResidentBasisDev::default()));

/// Serializes basis device *uploads* only (never handle reads); see [`RESIDENT_UPLOAD`].
static RESIDENT_BASIS_UPLOAD: Mutex<()> = Mutex::new(());

/// Ensure the resident basis is built through `max_degree` and return a snapshot of
/// `global_base` (so callers compute `gei = global_base[s_degree] + ti` without holding the
/// lock during the parallel marshal). `width` is the fixed p-part padding stride.
///
/// Append-only: only the first sight of each new degree takes the write lock, and the append
/// order fixes every element's `gei` forever. Basis enumeration is a pure function of the
/// algebra, so a first-sight race just recomputes identical bytes (the loser rechecks under
/// the write lock and appends nothing already present, since we extend strictly past
/// `built_degree`).
fn ensure_basis(algebra: &MilnorAlgebra, width: usize, max_degree: i32) -> Vec<u32> {
    {
        let host = RESIDENT_BASIS_HOST.read().unwrap();
        // `width != 0` distinguishes an initialized store from the derived-`Default` zero state
        // (where `built_degree == 0` would spuriously claim degree 0 is already built).
        if host.width != 0 && host.built_degree >= max_degree {
            return host.global_base.clone();
        }
    }
    let mut host = RESIDENT_BASIS_HOST.write().unwrap();
    if host.width == 0 {
        host.width = width;
        host.built_degree = -1; // nothing built yet; the loop below starts at degree 0
        host.global_base.push(0); // global_base[0] = 0 elements before degree 0
    }
    debug_assert_eq!(host.width, width, "basis padding width must be stable");
    for d in (host.built_degree + 1)..=max_degree {
        let dim = algebra.dimension(d);
        for i in 0..dim {
            let elt = algebra.basis_element_from_index(d, i);
            host.lens.push(elt.p_part.len() as u32);
            let base = host.pparts.len();
            host.pparts.resize(base + width, 0);
            for (slot, &v) in host.pparts[base..base + width].iter_mut().zip(&elt.p_part) {
                *slot = narrow_u16(v);
            }
        }
        // global_base[d+1] = total elements in degrees ≤ d.
        let total = host.lens.len() as u32;
        host.global_base.push(total);
    }
    host.built_degree = max_degree;
    host.global_base.clone()
}

/// Fetch the resident basis device handles `(pparts, lens)`, uploading the current host basis
/// only when the device copy does not yet cover `$need` elements. Upload runs outside
/// `RESIDENT_BASIS_DEV` (readers stay lock-free), serialized by `RESIDENT_BASIS_UPLOAD` with a
/// re-check to coalesce a burst of growth. The basis is append-only, so any handle with
/// `uploaded >= $need` is valid. A macro (not a fn) so the cubecl client type stays inferred,
/// exactly as [`resident_dev_handle`].
macro_rules! basis_dev_handles {
    ($client:expr, $need:expr) => {{
        // Lock-free fast path: the stable buffers already cover `$need` basis elements
        // (`ln.uploaded` is the element count).
        let read_current = || {
            let dev = RESIDENT_BASIS_DEV.read().unwrap();
            match (dev.ln.uploaded >= $need, dev.pp.handle.clone(), dev.ln.handle.clone()) {
                (true, Some(pp), Some(ln)) => Some((pp, ln)),
                _ => None,
            }
        };
        match read_current() {
            Some(h) => h,
            None => {
                let _guard = RESIDENT_BASIS_UPLOAD.lock().unwrap();
                match read_current() {
                    Some(h) => h, // another grower already covered our need
                    None => {
                        let width = RESIDENT_BASIS_HOST.read().unwrap().width;
                        let elems = RESIDENT_BASIS_HOST.read().unwrap().lens.len();
                        let pp_len = elems * width; // u16 count
                        let ln_len = elems; // u32 count

                        // Grow the width-padded p-parts buffer (u16) in place.
                        let (old_pp, pp_cap, pp_up) = {
                            let d = RESIDENT_BASIS_DEV.read().unwrap();
                            (d.pp.handle.clone(), d.pp.cap, d.pp.uploaded)
                        };
                        let (pp_h, pp_cap) = if old_pp.is_none() || pp_len > pp_cap {
                            let _realloc_w = RESIDENT_REALLOC.write().unwrap();
                            let new_cap = pp_len.max(pp_cap * 2).max(RESIDENT_INIT_CAP);
                            let nh = $client.empty(new_cap * ::core::mem::size_of::<u16>());
                            if let Some(oh) = &old_pp {
                                if pp_up > 0 {
                                    copy_chunked!(
                                        $client, copy_into_u16, oh, pp_up, 0usize,
                                        nh, new_cap, 0usize, pp_up
                                    );
                                }
                            }
                            let _ = cubecl_common::reader::read_sync($client.sync());
                            (nh, new_cap)
                        } else {
                            (old_pp.unwrap(), pp_cap)
                        };
                        if pp_len > pp_up {
                            let n = pp_len - pp_up;
                            stage_upload!(
                                $client, copy_into_u16, u16::as_bytes,
                                RESIDENT_BASIS_HOST.read().unwrap(), pparts,
                                pp_h, pp_cap, pp_up, n
                            );
                        }

                        // Grow the lengths buffer (u32) in place.
                        let (old_ln, ln_cap, ln_up) = {
                            let d = RESIDENT_BASIS_DEV.read().unwrap();
                            (d.ln.handle.clone(), d.ln.cap, d.ln.uploaded)
                        };
                        let (ln_h, ln_cap) = if old_ln.is_none() || ln_len > ln_cap {
                            let _realloc_w = RESIDENT_REALLOC.write().unwrap();
                            let new_cap = ln_len.max(ln_cap * 2).max(RESIDENT_INIT_CAP);
                            let nh = $client.empty(new_cap * ::core::mem::size_of::<u32>());
                            if let Some(oh) = &old_ln {
                                if ln_up > 0 {
                                    copy_chunked!(
                                        $client, copy_into_u32, oh, ln_up, 0usize,
                                        nh, new_cap, 0usize, ln_up
                                    );
                                }
                            }
                            let _ = cubecl_common::reader::read_sync($client.sync());
                            (nh, new_cap)
                        } else {
                            (old_ln.unwrap(), ln_cap)
                        };
                        if ln_len > ln_up {
                            let n = ln_len - ln_up;
                            stage_upload!(
                                $client, copy_into_u32, u32::as_bytes,
                                RESIDENT_BASIS_HOST.read().unwrap(), lens,
                                ln_h, ln_cap, ln_up, n
                            );
                        }

                        // `stage_upload!` already synced each chunk, so both grown buffers are
                        // physically resident before publishing (see [`resident_dev_handle`]):
                        // a reader on another worker's stream must never observe the bumped element
                        // count before the copy is done.

                        {
                            let mut dev = RESIDENT_BASIS_DEV.write().unwrap();
                            dev.pp.handle = Some(pp_h.clone());
                            dev.pp.cap = pp_cap;
                            dev.pp.uploaded = pp_len;
                            dev.ln.handle = Some(ln_h.clone());
                            dev.ln.cap = ln_cap;
                            dev.ln.uploaded = ln_len;
                        }
                        (pp_h, ln_h)
                    }
                }
            }
        }
    }};
}

/// Zero a device `u32` buffer on-device: `out[i] = 0`, one thread per limb.
///
/// Initializes the batched multiply's XOR accumulator without allocating and uploading a host
/// zero buffer. Profiling (stem 145) showed the per-launch `create_from_slice` of a
/// hundreds-of-MB `out_h` zero vec — a host `memset` + non-pinned host→device `memcpy`, both on
/// the calling rayon worker — was the dominant serial marshaling cost, stalling the wavefront.
/// On-device zeroing is memory-bound (microseconds on an H200) and same-stream ordered before
/// the multiply kernel, so no host allocation, upload, or extra sync is needed.
#[cube(launch)]
fn zero_u32(out: &mut Array<u32>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = 0u32;
    }
}

/// Copy `count` elements `src[src_off + i] -> dst[dst_off + i]`, one thread per element. Used to grow
/// the resident master/basis IN PLACE — new data is uploaded to a scratch buffer and copied into the
/// stable resident buffer at its append offset, so the resident device handle never changes (no
/// re-`create_from_slice` churn that would break cross-stream sync).
///
/// Offsets are `usize` (64-bit on device under the launch's `address_type = "dynamic"`, so both the
/// `dst_off` append offset and the buffer length are safe past `u32::MAX`), and `count` bounds this
/// launch so the caller can split a copy larger than the `u32` grid/`ABSOLUTE_POS` thread limit into
/// chunks (a resident buffer exceeds 2^32 u16 elements around stem 150). See [`copy_chunked`].
// `launch_unchecked` + dynamic addressing: `dst_off`/buffer length exceed `u32` once the resident
// master/basis passes 2^32 elements, needing 64-bit `usize`; and cubecl's checked bounds clamp emits
// `min(u64, u64)` (ambiguous for NVRTC) under u64. The `ABSOLUTE_POS < count` guard keeps it in-bounds.
#[cube(launch_unchecked, address_type = "dynamic")]
fn copy_into_u16(src: &Array<u16>, dst: &mut Array<u16>, src_off: usize, dst_off: usize, count: u32) {
    if ABSOLUTE_POS < usize::cast_from(count) {
        dst[dst_off + ABSOLUTE_POS] = src[src_off + ABSOLUTE_POS];
    }
}

/// `u32` sibling of [`copy_into_u16`] (for the resident basis `lens`).
#[cube(launch_unchecked, address_type = "dynamic")]
fn copy_into_u32(src: &Array<u32>, dst: &mut Array<u32>, src_off: usize, dst_off: usize, count: u32) {
    if ABSOLUTE_POS < usize::cast_from(count) {
        dst[dst_off + ABSOLUTE_POS] = src[src_off + ABSOLUTE_POS];
    }
}

/// Elements per copy-kernel launch: below the kernel's `u32` `ABSOLUTE_POS` thread limit, so copies
/// of multi-billion-element resident buffers are split into this many at a time.
const COPY_CHUNK: usize = 1 << 30;

/// Elements per pinned host-staging chunk when uploading resident growth (see [`stage_upload`]).
/// Bounds the page-locked host buffer cubecl reserves per `create_from_slice`: those pinned pages
/// are pooled PER CUDA STREAM and never trimmed, so a single full-master `create_from_slice` (the
/// tail can be many GB) would pin that whole size on every stream — measured ~240 GB shmem at stem
/// 180 with 8 streams, the OOM driver. Staging in `STAGE_CHUNK` pieces with a sync between them
/// caps the live pinned staging at ~one chunk. 64 Mi × u16 = 128 MiB (× u32 = 256 MiB).
const STAGE_CHUNK: usize = 1 << 26;

/// Upload `host_slice[0..n]` into resident device `dst[dst_off..dst_off+n]` through a BOUNDED pinned
/// staging buffer, re-locking `$host_lock` to reslice each chunk. `$host_lock` is an expression
/// evaluating to a read guard whose `$field` is the source `Vec` (re-evaluated per chunk so the
/// guard is not held across the sync). Syncs after each chunk so the pinned pool reuses one page
/// instead of accumulating the whole tail (see [`STAGE_CHUNK`]). `$copy` is `copy_into_u16`/`_u32`,
/// `$as_bytes` the matching `u16`/`u32` `as_bytes`.
macro_rules! stage_upload {
    ($client:expr, $copy:ident, $as_bytes:path, $host_lock:expr, $field:ident,
     $dst:expr, $dst_cap:expr, $dst_off:expr, $n:expr) => {{
        let mut done = 0usize;
        while done < $n {
            let m = ($n - done).min(STAGE_CHUNK);
            let lo = $dst_off + done;
            let scratch = {
                let host = $host_lock;
                $client.create_from_slice($as_bytes(&host.$field[lo..lo + m]))
            };
            copy_chunked!($client, $copy, scratch, m, 0usize, $dst, $dst_cap, lo, m);
            // Drain the H2D copy so this chunk's pinned staging is freed (and its pool page reused)
            // before the next `create_from_slice`, keeping live pinned memory at ~one chunk.
            let _ = cubecl_common::reader::read_sync($client.sync());
            done += m;
        }
    }};
}

/// Copy `count` elements `src[src_off..] -> dst[dst_off..]` with `$kernel` (`copy_into_u16`/`_u32`),
/// splitting into [`COPY_CHUNK`]-element launches so counts past the `u32` thread limit are handled.
/// `$src_len`/`$dst_len` are the logical array lengths passed to the kernel (must cover the ranges).
macro_rules! copy_chunked {
    ($client:expr, $kernel:ident, $src:expr, $src_len:expr, $src_off:expr,
     $dst:expr, $dst_len:expr, $dst_off:expr, $count:expr) => {{
        const CT: u32 = 256;
        let mut done: usize = 0;
        while done < $count {
            let n = ($count - done).min(COPY_CHUNK);
            unsafe {
                $kernel::launch_unchecked::<CudaRuntime>(
                    &$client,
                    CubeCount::Static((n as u32).div_ceil(CT), 1, 1),
                    CubeDim::new_1d(CT),
                    // The resident dst offset ($dst_off) and buffer length exceed u32 at high stems.
                    AddressType::from_len(($src_len).max($dst_len).max($dst_off + $count)),
                    ArrayArg::from_raw_parts($src.clone(), $src_len),
                    ArrayArg::from_raw_parts($dst.clone(), $dst_len),
                    $src_off + done,
                    $dst_off + done,
                    n as u32,
                );
            }
            done += n;
        }
    }};
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
// 64-bit addressing (`address_type = "u64"`): the admissible masters (`col_sums`/`masks`) and the
// width-padded basis exceed `u32::MAX` elements at high stems, and the per-`R` offsets in
// `r_cs_offset`/`r_mk_offset` (u64) index into them. Static u64 (not "dynamic") because dynamic
// would pick 32-bit `usize` for small blocks and then *narrow* the u64 offset arrays on read
// (cubecl `usize::cast_from(u64)` under a u32 address type), corrupting results — the (180,92)
// `dx != 0`. `launch_unchecked` because cubecl's checked-mode bounds clamp emits `min(u64, u64)`,
// which NVRTC rejects as an ambiguous overload; every access here is already in-bounds by
// construction (the `need_*` prefix covers every offset and the per-column guards bound `j`).
#[cube(launch_unchecked, address_type = "u64")]
#[allow(clippy::too_many_arguments)]
fn multiply_batch_kernel(
    col_sums: &Array<u16>,
    masks: &Array<u16>,
    basis_pparts: &Array<u16>,
    basis_lens: &Array<u32>,
    term_gei: &Array<u32>,
    g: &Array<u32>,
    xi: &Array<u32>,
    out: &mut Array<Atomic<u32>>,
    r_cs_offset: &Array<u64>,
    r_mk_offset: &Array<u64>,
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
    // `term_gei[term_slot]` is the term's *global* basis-element index (across all degrees):
    // its (width-padded) p-part lives at `basis_pparts[gei*width ..]`, length `basis_lens[gei]`.
    // The basis is resident on the device (uploaded once, grown incrementally), so a launch
    // uploads only these indices instead of re-gathering every term's p-part — see
    // [`ResidentBasisHost`].
    let gei = usize::cast_from(term_gei[term_slot]);
    multiply_pair(
        col_sums,
        masks,
        basis_pparts,
        g,
        xi,
        out,
        usize::cast_from(r_cs_offset[ri]) + m * cs_len,
        usize::cast_from(r_mk_offset[ri]) + m * mk_len,
        gei * width,
        usize::cast_from(basis_lens[gei]),
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
#[derive(Clone)]
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
    let cap = resident_degree_cap();
    // Fast path (default, `cap == i32::MAX`, and any run whose `R`s are all under the cap): a single
    // resident-master pass, byte-identical to the pre-eviction code. No cloning, no second launch.
    if cap == i32::MAX || products.iter().all(|p| p.r_degree <= cap) {
        return multiply_batch_grouped(algebra, num_cols, num_rows, products, MasterMode::Resident);
    }
    // Eviction active. Each output row's products all share one `R` (see [`MasterMode`]), so the
    // hot (degree ≤ cap) and cold row sets are DISJOINT. Run each group on its own rows only,
    // **compacted** to a dense `0..k` range so each pass reads back just its own rows — total
    // readback stays `num_rows`, not 2× (critical in the intended high-θ regime, where the cold
    // set is a small tail and a full-height cold readback would be almost all zeros). Results
    // scatter back to the original row indices; a row in neither group stays zero (its content, if
    // any, comes from the caller's CPU identity path).
    let num_limbs = num_cols.div_ceil(32).max(1);
    let mut result = vec![vec![0u32; num_limbs]; num_rows];
    for mode in [MasterMode::Resident, MasterMode::Transient] {
        let is_group = |d: i32| match mode {
            MasterMode::Resident => d <= cap,
            MasterMode::Transient => d > cap,
        };
        // Distinct rows this group touches, in order (products are row-major, so already sorted).
        let mut rows: Vec<usize> = products
            .iter()
            .filter(|p| is_group(p.r_degree))
            .map(|p| p.row)
            .collect();
        rows.dedup();
        if rows.is_empty() {
            continue;
        }
        let remap: HashMap<usize, usize> =
            rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        let compact: Vec<GpuProduct> = products
            .iter()
            .filter(|p| is_group(p.r_degree))
            .map(|p| {
                let mut q = p.clone();
                q.row = remap[&p.row];
                q
            })
            .collect();
        let sub = multiply_batch_grouped(algebra, num_cols, rows.len(), &compact, mode);
        for (i, &orig) in rows.iter().enumerate() {
            for (a, b) in result[orig].iter_mut().zip(&sub[i]) {
                *a ^= *b;
            }
        }
    }
    result
}

fn multiply_batch_grouped(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
    mode: MasterMode,
) -> Vec<Vec<u32>> {
    let num_limbs = num_cols.div_ceil(32).max(1);
    let max_block_rows = (gpu_block_bytes() / (num_limbs * 4)).max(1);
    // Products arrive row-major (the extract loops emit them per input row, in order; the hot/cold
    // filter above preserves that order), so each block is a contiguous product slice. Rows are
    // independent — every product writes only its own row — so concatenating block outputs
    // reproduces the single-launch result exactly.
    debug_assert!(products.windows(2).all(|w| w[0].row <= w[1].row));
    // Per-product `(matrix, term)` pair counts, i.e. kernel threads. The kernel indexes threads
    // by `ABSOLUTE_POS`, a `u32`, so a block must also stay under `2^32` pairs — output bytes
    // alone don't bound this (pairs per row grow with the degree; an unbounded all-rows build
    // reaches ~4.4e9 pairs by stem ~145). For `Resident` this pre-pass also warms the shared
    // resident master, so every block's layout lookups below are read-lock cache hits; for
    // `Transient` it warms the host-side [`COLD_HOST`] cache the same way (no per-block recompute).
    let prod_pairs: Vec<usize> = products
        .iter()
        .map(|prod| {
            let r = algebra.basis_element_from_index(prod.r_degree, prod.r_idx);
            let num_mats = match mode {
                MasterMode::Resident => resident_info(algebra, &r.p_part).num_mats as usize,
                MasterMode::Transient => cold_host_entry(algebra, &r.p_part).2 as usize,
            };
            num_mats * prod.term_indices.len()
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
            mode,
        ));
        (r0, p0) = (r1, p1);
    }
    result
}

/// One bounded launch of [`multiply_batch_on_gpu`]: rows `row_base..row_base + num_rows` of the
/// full build, with `products` the (contiguous, row-major) slice landing in those rows. Holds a
/// [`GpuPermit`] for its sequential layout + device section (acquired only after the parallel
/// marshal — see [`GpuBudget`]), so the total output size of concurrent device sections
/// stays under `NASSAU_GPU_MEM_BUDGET_MB` across all worker threads.
fn multiply_batch_block(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    row_base: usize,
    num_rows: usize,
    products: &[GpuProduct],
    mode: MasterMode,
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

    // Per-product term p-parts (padded to `width`) and lengths, filled directly into two flat
    // buffers rather than one `(Vec, Vec)` per product. At the frontier a launch has ~10^5-10^6
    // products; the old per-product `Vec` pair (plus the later concat-copy) was ~10^6 tiny
    // allocations per launch — a dominant chunk of the marshal cost. `term_off` is the prefix sum
    // of term counts, so each product owns a disjoint output range and the fill stays parallel.
    let term_off: Vec<usize> = {
        let mut off = Vec::with_capacity(products.len() + 1);
        let mut acc = 0usize;
        for prod in products {
            off.push(acc);
            acc += prod.term_indices.len();
        }
        off.push(acc);
        off
    };
    let total_terms = *term_off.last().unwrap();

    // Resident-basis path (the default): a term's p-part is not marshalled at all — it lives on
    // the device (built once, grown incrementally). We upload only `term_gei[slot]`, the term's
    // *global* basis-element index `global_base[s_degree] + ti`. Ensure the basis covers every
    // `s_degree` in this block, then snapshot `global_base` so the parallel fill needs no lock.
    let max_s_degree = products.iter().map(|p| p.s_degree).max().unwrap_or(0);
    let global_base = ensure_basis(algebra, width, max_s_degree);
    let mut term_gei: Vec<u32> = vec![0u32; total_terms];
    {
        let tg_base = term_gei.as_mut_ptr() as usize;
        let global_base = &global_base;
        (0..products.len()).into_maybe_par_iter().for_each(|pi| {
            let prod = &products[pi];
            let (off, nt) = (term_off[pi], prod.term_indices.len());
            // SAFETY: products write disjoint `[off, off + nt)` ranges (from the prefix sum),
            // each within the allocated buffer, so no two tasks alias any element. The `usize`
            // base is re-formed into a pointer here because raw pointers are not `Send`.
            let tg = unsafe { std::slice::from_raw_parts_mut((tg_base as *mut u32).add(off), nt) };
            let base = global_base[prod.s_degree as usize];
            for (k, &ti) in prod.term_indices.iter().enumerate() {
                tg[k] = base + ti as u32;
            }
        });
    }
    // Device need: the largest `gei` any term dereferences is `< global_base[max_s_degree + 1]`
    // (all elements through degree `max_s_degree`), so uploading that many covers the block.
    let need_basis_elems = global_base[max_s_degree as usize + 1] as usize;

    // A/B diagnostic (`NASSAU_GPU_BASIS_PASSTHROUGH=1`): bind the *per-launch* term buffers as the
    // "basis" and set `term_gei` to the identity, so the new kernel reproduces the old behaviour
    // bit-for-bit. If passthrough matches the CPU but the resident path does not, the bug is in
    // the resident host/upload logic, not the kernel signature — and vice-versa.
    let passthrough = basis_passthrough();
    let (mut term_pparts, mut term_lens): (Vec<u16>, Vec<u32>) = if passthrough {
        let mut tp: Vec<u16> = vec![0u16; total_terms * width];
        let mut tl: Vec<u32> = vec![0u32; total_terms];
        let tp_base = tp.as_mut_ptr() as usize;
        let tl_base = tl.as_mut_ptr() as usize;
        (0..products.len()).into_maybe_par_iter().for_each(|pi| {
            let prod = &products[pi];
            let (off, nt) = (term_off[pi], prod.term_indices.len());
            // SAFETY: disjoint per-product ranges, as above.
            let tpp = unsafe {
                std::slice::from_raw_parts_mut((tp_base as *mut u16).add(off * width), nt * width)
            };
            let tll = unsafe { std::slice::from_raw_parts_mut((tl_base as *mut u32).add(off), nt) };
            for (k, &ti) in prod.term_indices.iter().enumerate() {
                let elt = algebra.basis_element_from_index(prod.s_degree, ti);
                tll[k] = elt.p_part.len() as u32;
                for (slot, &v) in tpp[k * width..(k + 1) * width].iter_mut().zip(&elt.p_part) {
                    *slot = narrow_u16(v);
                }
            }
        });
        // Identity indices, so the kernel's `gei*width` / `basis_lens[gei]` hit slot `term_slot`.
        for (i, g) in term_gei.iter_mut().enumerate() {
            *g = i as u32;
        }
        (tp, tl)
    } else {
        (Vec::new(), Vec::new())
    };

    // Take the concurrency permit only now, with every rayon parallel section behind us: holding
    // it across the `per_prod` par_iter above deadlocks, because that par_iter's chunks execute on
    // *other* threads, which do not carry this thread's `ParallelGuard` flag and so can steal a
    // bidegree job mid-chunk; the stolen job parks on `GpuPermit::acquire` while this thread's
    // permit waits on the never-finishing join (observed on H200). Everything from here on is
    // strictly sequential — the `ensure` calls below are cache hits (the caller's pair-count
    // pre-pass already enumerated every `R`), and the device section never enters rayon — so
    // every permit holder makes progress and stolen jobs waiting for a permit wake in finite
    // time (priority inversion at worst, never deadlock).
    // Held for the device section (RAII): bounds total in-flight output bytes across workers. The
    // stream is now chosen per-thread (see [`thread_stream_id`]), not from the permit.
    let _permit = GpuPermit::acquire(num_rows * num_limbs * 4);
    // Per-`R` offsets into the shared resident master (see [`ResidentHost`]). All read-lock
    // cache hits: the caller's pair-count pre-pass already enumerated every `R` in this block.
    // `need_cs`/`need_mk` track the furthest master offset this block dereferences, so the
    // device section can skip the (multi-GB, mutex-serialized) master re-upload whenever the
    // already-uploaded prefix covers it.
    let mut r_cs_offset: Vec<u64> = Vec::with_capacity(distinct_r.len());
    let mut r_mk_offset: Vec<u64> = Vec::with_capacity(distinct_r.len());
    let mut r_cs_len: Vec<u32> = Vec::with_capacity(distinct_r.len());
    let mut r_mk_len: Vec<u32> = Vec::with_capacity(distinct_r.len());
    let mut r_num_matrices: Vec<usize> = Vec::with_capacity(distinct_r.len());
    let mut need_cs: usize = 0;
    let mut need_mk: usize = 0;
    // `Transient`: this block's own `col_sums`/`masks`, packed contiguously with block-local
    // offsets (mirrors the resident master's layout, but built fresh here and freed after the
    // launch instead of persisting). Empty / unused under `Resident`.
    let mut cs_local: Vec<u16> = Vec::new();
    let mut mk_local: Vec<u16> = Vec::new();
    for &(rd, ridx) in &distinct_r {
        let r = algebra.basis_element_from_index(rd, ridx);
        assert!(!r.p_part.is_empty(), "each R must be non-empty");
        match mode {
            MasterMode::Resident => {
                let info = resident_info(algebra, &r.p_part);
                r_cs_offset.push(info.cs_off);
                r_mk_offset.push(info.mk_off);
                r_cs_len.push(info.cs_len);
                r_mk_len.push(info.mk_len);
                r_num_matrices.push(info.num_mats as usize);
                need_cs = need_cs
                    .max(info.cs_off as usize + info.num_mats as usize * info.cs_len as usize);
                need_mk = need_mk
                    .max(info.mk_off as usize + info.num_mats as usize * info.mk_len as usize);
            }
            MasterMode::Transient => {
                let (cs_len, mk_len, num_mats, cs, mk) = cold_host_entry(algebra, &r.p_part);
                r_cs_offset.push(cs_local.len() as u64);
                r_mk_offset.push(mk_local.len() as u64);
                r_cs_len.push(cs_len);
                r_mk_len.push(mk_len);
                r_num_matrices.push(num_mats as usize);
                cs_local.extend_from_slice(&cs);
                mk_local.extend_from_slice(&mk);
                need_cs = cs_local.len();
                need_mk = mk_local.len();
            }
        }
    }

    // Lay out per-product records + the pair-count prefix sum (sequential). Term data is already
    // in `term_pparts`/`term_lens` (filled in parallel above); `term_off` gives each product's
    // start, so nothing is copied here.
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
    for (pi, prod) in products.iter().enumerate() {
        let ri = prod_r_index[pi];
        prod_term_start.push(term_off[pi] as u32);
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
    // Output offsets (`prod_out_offset`/`prod_row_base`) are `u32` values indexing `out_h`; the
    // row-block splitter caps `out_len` well under `u32::MAX` (its output-byte budget is far below
    // 16 GiB), so these never truncate. Assert it loudly rather than silently corrupt if a future
    // budget is set absurdly high. (The `out_h` *length* itself is bound with dynamic addressing.)
    assert!(
        u32::try_from(out_len).is_ok(),
        "block output length {out_len} exceeds u32; lower NASSAU_GPU_BLOCK_MB / row-block budget"
    );
    if std::env::var_os("NASSAU_GPU_DEBUG").is_some() {
        let kb = |n: usize, sz: usize| n * sz / 1024;
        eprintln!(
            "[gpu-batch] rows={num_rows} cols={num_cols} products={} total_pairs={total_pairs} \
             out_len={out_len} | UPLOAD-KB: g={} xi={} term_gei={} \
             prod_arrays={} pps={} | resident cs={} mk={} basis_elems={need_basis_elems}",
            products.len(),
            kb(g.len(), 4),
            kb(xi.len(), 4),
            kb(term_gei.len(), 4),
            kb(products.len() * 5, 4),
            kb(pps.len(), 4),
            kb(need_cs, 2),
            kb(need_mk, 2),
        );
    }
    if total_pairs == 0 {
        return vec![vec![0u32; num_limbs]; num_rows];
    }

    // The resident `col_sums`/`masks` and basis are non-empty once any `R`/term is present
    // (guaranteed here, since `total_pairs > 0`); only `term_gei` (and, in passthrough, the
    // per-launch term buffers) needs the non-empty guard `create_from_slice` requires.
    if term_gei.is_empty() {
        term_gei.push(0);
    }
    if passthrough && term_pparts.is_empty() {
        term_pparts.push(0);
        term_lens.push(0);
    }

    let marshal_ms = t_marshal.elapsed().as_secs_f64() * 1e3;

    let t_device = std::time::Instant::now();

    // Device section on this worker's own stable stream (see [`thread_stream_id`]): up to
    // `gpu_stream_slots()` distinct streams run concurrently. Cubecl's per-device runner serializes
    // the actual server access; the shared resident master/basis (stable in-place-grown buffers) are
    // read cross-stream safely. `memory_cleanup` below trims only this stream's own transient pool.
    let result = StreamId {
        value: thread_stream_id(),
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
        // `Transient` (degree > cap `R`s): upload this block's freshly-built master and free it
        // with the launch — no persistence, no cross-stream sharing, so no realloc guard below.
        let (cs_h, cs_len_master, mk_h, mk_len_master) = match mode {
            MasterMode::Resident => {
                let (cs_h, cs_len_master) =
                    resident_dev_handle!(client, need_cs, cs, cs_pending, cs_len);
                let (mk_h, mk_len_master) =
                    resident_dev_handle!(client, need_mk, mk, mk_pending, mk_len);
                (cs_h, cs_len_master, mk_h, mk_len_master)
            }
            MasterMode::Transient => (
                client.create_from_slice(u16::as_bytes(&cs_local)),
                cs_local.len(),
                client.create_from_slice(u16::as_bytes(&mk_local)),
                mk_local.len(),
            ),
        };
        // Upload the block's data — term data, seqno/xi tables, per-`R` offsets, per-product
        // records, the pair prefix sum, and the (zeroed) output buffer — and launch once: the
        // caller has already bounded this block's pair count and output size.
        // Resident basis handles (default) or per-launch passthrough buffers (A/B diagnostic).
        // `bp_len`/`bl_len` are the logical array lengths the kernel sees; every `gei` a thread
        // dereferences is `< need_basis_elems`, so `need_basis_elems*width` / `need_basis_elems`
        // cover it (the resident buffer may be larger — append-only — which is fine).
        let (bp_h, bl_h, bp_len, bl_len) = if passthrough {
            (
                client.create_from_slice(u16::as_bytes(&term_pparts)),
                client.create_from_slice(u32::as_bytes(&term_lens)),
                term_pparts.len(),
                term_lens.len(),
            )
        } else {
            let (pp, ln) = basis_dev_handles!(client, need_basis_elems);
            (pp, ln, need_basis_elems * width, need_basis_elems)
        };
        let tg_h = client.create_from_slice(u32::as_bytes(&term_gei));
        let g_h = client.create_from_slice(u32::as_bytes(&g));
        let xi_h = client.create_from_slice(u32::as_bytes(&xi));
        let rco_h = client.create_from_slice(u64::as_bytes(&r_cs_offset));
        let rmo_h = client.create_from_slice(u64::as_bytes(&r_mk_offset));
        let rcl_h = client.create_from_slice(u32::as_bytes(&r_cs_len));
        let rml_h = client.create_from_slice(u32::as_bytes(&r_mk_len));
        const THREADS: u32 = 256;
        // Hold the resident READ lock across the multiply that dereferences the resident buffers,
        // through the readback that syncs it (see [`RESIDENT_REALLOC`]). All resident growth for this
        // block is already done above; a concurrent capacity realloc on another thread waits here for
        // this multiply to finish, so the resident handle can never be swapped mid-kernel. Held to
        // the end of the device section (dropped after `read_one`). `Transient` buffers are
        // block-local (never reallocated by another thread), so they need no such guard.
        let _realloc_guard =
            (mode == MasterMode::Resident).then(|| RESIDENT_REALLOC.read().unwrap());
        // Allocate the XOR accumulator uninitialized and zero it on-device (see [`zero_u32`]),
        // instead of uploading a hundreds-of-MB host zero buffer — the former dominant serial
        // marshaling cost. Same stream as the multiply below, so it is ordered before it.
        let out_h = client.empty(out_len * size_of::<u32>());
        unsafe {
            zero_u32::launch::<CudaRuntime>(
                &client,
                CubeCount::Static((out_len as u32).div_ceil(THREADS).max(1), 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(out_h.clone(), out_len),
            );
        }

        let pri_h = client.create_from_slice(u32::as_bytes(&prod_r_index));
        let pts_h = client.create_from_slice(u32::as_bytes(&prod_term_start));
        let pnt_h = client.create_from_slice(u32::as_bytes(&prod_num_terms));
        let prb_h = client.create_from_slice(u32::as_bytes(&prod_row_base));
        let poo_h = client.create_from_slice(u32::as_bytes(&prod_out_offset));
        let pps_h = client.create_from_slice(u32::as_bytes(&pps));
        let cubes = (total_pairs as u32).div_ceil(THREADS).max(1);
        // SAFETY: `launch_unchecked` — see the kernel's `address_type = "u64"` note. Every device
        // read is in-bounds by construction (uploaded `need_*` prefix + per-column `j` guards).
        unsafe {
            multiply_batch_kernel::launch_unchecked::<CudaRuntime>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(THREADS),
                ArrayArg::from_raw_parts(cs_h, cs_len_master),
                ArrayArg::from_raw_parts(mk_h, mk_len_master),
                ArrayArg::from_raw_parts(bp_h, bp_len),
                ArrayArg::from_raw_parts(bl_h, bl_len),
                ArrayArg::from_raw_parts(tg_h, term_gei.len()),
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
