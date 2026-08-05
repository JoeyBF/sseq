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
use cubecl_common::bytes::Bytes;

// Bounds the per-thread enumeration state ([`ENUM_ROW_CAP`]) and the `#[cfg(test)]` `seqno_kernel`'s
// working array; the multiply kernel uses `WORKING_CAP`.
use crate::algebra::combinatorics::MAX_XI_TAU;
use crate::algebra::{Algebra, MilnorAlgebra, combinatorics::xi_degrees, milnor_algebra::PPart};

/// Comptime capacity for the per-thread `working` p_part in the multiply kernel.
/// The assembled p_part has length `max(term_len, mk_len)` before trimming, where
/// `mk_len = rows + cols − 1 ≤ MAX_XI_TAU + ⌈log2⌉`; 32 covers every in-range case.
const WORKING_CAP: usize = 32;

/// `PPart::MAX_LEN` — the number of entries the packed exponent sequence can hold.
///
/// Positions beyond it are unreachable, not merely unused: entry `r_n` multiplies
/// `deg(xi_n) = 2^n - 1`, so length 11 requires degree >= 2047 while `PPart::MAX_DEGREE` is 2045.
/// The multiply kernel's `working` accumulator therefore packs into one `u64` with no loss.
const PPART_MAX_LEN: usize = 10;

/// Bit offset and value mask of each packed p-part field, uploaded per launch (80 bytes) so the
/// kernel can unpack `working` without a per-thread array. Mirrors `PPart`'s private tables.
fn ppart_shift_mask() -> (Vec<u32>, Vec<u32>) {
    let shift: Vec<u32> = (0..PPART_MAX_LEN).map(|i| PPart::shift(i)).collect();
    let mask: Vec<u32> = (0..PPART_MAX_LEN)
        .map(|i| {
            let w = PPart::shift(i + 1) - PPart::shift(i);
            ((1u64 << w) - 1) as u32
        })
        .collect();
    (shift, mask)
}

/// Per-thread local caps for the in-kernel admissible enumeration ([`enumerate_admissible_kernel`]).
/// Each `R` has `rows = |p_part| ≤ MAX_XI_TAU` and `cols ≤ WORKING_CAP` (max bit-length of an entry),
/// so the enumeration's `matrix` is `rows*cols`, `col_sums` is `cols−1`, and `masks` is `rows+cols−1`.
/// These bound the fixed-size local `Array`s the kernel allocates per thread.
const ENUM_ROW_CAP: usize = MAX_XI_TAU;
const ENUM_COL_CAP: usize = WORKING_CAP;
const ENUM_MATRIX_CAP: usize = ENUM_ROW_CAP * ENUM_COL_CAP;
const ENUM_MASK_CAP: usize = ENUM_ROW_CAP + ENUM_COL_CAP;

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

/// Chunk size (log2) of the multiply kernel's coarse product index. One entry per `2^COARSE_LOG`
/// pairs, so a launch of billions of pairs needs a table of a few thousand `u32` — negligible to
/// build and upload, and it turns the per-thread product lookup from a full binary search over
/// every product into a scan bounded by how many products one chunk spans.
const COARSE_LOG: usize = 20;

/// Terms one multiply thread handles against a single matrix. `col_sums`/`masks` depend only on the
/// matrix, so a group amortises those reads (and their address arithmetic) across `TERM_GROUP`
/// terms: loads per pair fall from 3 per column to `2/TERM_GROUP + 1`.
///
/// Tuned, and the optimum is NOT monotonic — bigger groups amortise more but waste more lanes on
/// the ragged tail, since a product's `nt` terms need `ceil(nt / TERM_GROUP)` groups and the last
/// one is usually partial. Two interleaved rounds at the measured `nt ~ 5`:
///
/// | TERM_GROUP | pairs/s        | idle lanes at nt=5 |
/// |------------|----------------|--------------------|
/// | 2          | 1.09 / 1.06e10 | 1 of 6  (17%)      |
/// | **3**      | **1.31 / 1.28e10** | 1 of 6  (17%)  |
/// | 4          | 1.20 / 1.18e10 | 3 of 8  (37%)      |
/// | 6          | 1.27 / 1.25e10 | 1 of 6  (17%)      |
/// | 8          | 1.06 / 1.07e10 | 3 of 8  (37%)      |
///
/// Every 17%-waste value beats every 37%-waste value, so tail waste dominates the choice; among
/// those, 3 amortises more than 2 and holds more registers than 6 does not need. Retune if the
/// terms-per-product regime moves: this is fitted to `nt ~ 5`, and a workload with a different
/// average would want a different divisor.
const TERM_GROUP: usize = 3;

/// Matrices one multiply thread handles, the second axis of the tile alongside [`TERM_GROUP`].
/// `col_sums`/`masks` are per-matrix, so an `M x T` tile costs `2M + T` loads per column for `M*T`
/// pairs. Unlike terms there is no meaningful ragged tail here -- `num_mats` runs to ~20 000, so a
/// partial tile idles a couple of lanes out of thousands.
const MATRIX_GROUP: usize = 2;

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

/// The single OS thread that owns the CUDA stream, and the queue that feeds it.
///
/// # Why a dedicated thread
///
/// Every worker used to run its own device section on the *shared* stream 0 (see
/// stream 0, so ~7 concurrent workers raced for cubecl's per-device submission path.
/// That path is not FIFO-fair, and under sustained contention a worker could be passed over for
/// minutes: measured on a stem-200 run, one `step` blocked for **370 s** inside the multiply while
/// six peers each stayed 100 % busy completing 200–576 steps at 0.6–1.6 s apiece. The same
/// `(bidegree, signature)` — bit-identical work — took 0.04 s in one run and 235 s in another; 428
/// step pairs differed by more than 100× across two complete runs, wasting ~110–150 min of thread
/// time each. The work is uniform (within a bidegree the spread is ~5×); the outliers were purely
/// service-order artifacts.
///
/// Funnelling every device section through one thread fed by an `mpsc` channel makes service order
/// FIFO by construction, so a worker's wait is bounded by the jobs enqueued ahead of it and the
/// starvation case cannot arise. Total device serialisation is unchanged — stream 0 already
/// serialised everything — but it is now *fair*. Workers still marshal in parallel; only the
/// (already sequential) device section moves.
///
/// # Invariants
///
/// - The receive loop runs inside a single `StreamId::executes`, so the stream is bound once and
///   has exactly one driver thread for the process's lifetime — what cubecl's per-stream state
///   assumes.
/// - Tasks must not need a worker thread. Nothing here enters rayon, and every resident-store lock
///   ([`resident_info`], `ensure_basis`) is taken and released *inside* one task, never held across
///   a submission — so a blocked worker can never hold a lock this thread waits on.
/// - Panics are caught per task and forwarded to the waiting worker, which resumes the unwind. A
///   panic that killed this thread would instead hang every future submission forever.
mod gpu_thread {
    use std::{
        sync::{
            OnceLock,
            atomic::{AtomicU64, Ordering},
            mpsc,
        },
        time::Instant,
    };

    use crossbeam_channel::{Sender, unbounded};
    use cubecl_common::stream_id::StreamId;

    /// Jobs enqueued but not yet started: the shared queue's depth when a worker joins it.
    static DEPTH: AtomicU64 = AtomicU64::new(0);

    type Task = Box<dyn FnOnce() + Send + 'static>;

    /// How long a submission waited in the queue, and how long it then took on the device.
    pub(super) struct Timing {
        /// Enqueue → task start. Under FIFO this is the work queued ahead of this job.
        pub queue_ms: f64,
        /// Task start → task end: the device section proper.
        pub exec_ms: f64,
        /// Queue depth observed at enqueue (including this job).
        pub depth: u64,
    }

    /// One queue PER DEVICE. A shared pull-queue would balance better — a worker takes the next job
    /// the instant it frees up, with no need to predict job size — but the sharded master makes work
    /// device-AFFINE: an `R`'s rows live on exactly one device, so its products can only run there.
    /// Balance therefore comes from spreading `R`s evenly (round-robin at first sight), not from
    /// letting idle workers steal.
    ///
    /// Each worker owns its device, its stream, and (via the thread-local set here) its own replica
    /// of the resident master/basis, so a device handle can never reach another device's client.
    ///
    /// `crossbeam-channel` because this is genuinely multi-consumer: std's `mpsc` has a single
    /// receiver, so one shared queue there would mean wrapping it in a `Mutex` and serialising every
    /// pop behind a lock held across a blocking `recv`.
    fn senders() -> &'static Vec<Sender<Task>> {
        static QUEUES: OnceLock<Vec<Sender<Task>>> = std::sync::OnceLock::new();
        QUEUES.get_or_init(|| {
            let mut txs = Vec::with_capacity(super::gpu_count());
            for dev in 0..super::gpu_count() {
                let (tx, rx) = unbounded::<Task>();
                txs.push(tx);
                std::thread::Builder::new()
                    .name(format!("nassau-gpu{dev}"))
                    .spawn(move || {
                        super::CUR_DEVICE.with(|c| c.set(dev));
                        // Bind the stream once for the whole loop: one stream per driver thread, a
                        // distinct id per device so the runtime keeps them independent.
                        StreamId { value: dev as u64 }.executes(|| {
                            while let Ok(task) = rx.recv() {
                                task();
                            }
                        });
                    })
                    .expect("failed to spawn a nassau-gpu thread");
            }
            txs
        })
    }

    /// Run `f` on the GPU thread, blocking until it returns. Panics propagate to the caller.
    /// A submitted job that has not been waited on yet.
    pub(super) struct Pending<T> {
        rx: mpsc::Receiver<(std::thread::Result<T>, f64, f64)>,
        depth: u64,
    }

    impl<T> Pending<T> {
        /// Block for the result. Panics in the job propagate to the caller.
        pub(super) fn wait(self) -> (T, Timing) {
            let (out, queue_ms, exec_ms) =
                self.rx.recv().expect("the nassau-gpu thread died mid-task");
            match out {
                Ok(v) => (
                    v,
                    Timing {
                        queue_ms,
                        exec_ms,
                        depth: self.depth,
                    },
                ),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
    }

    /// Submit `f` to `dev`'s worker WITHOUT blocking. The sharded fan-out needs every device in
    /// flight at once; blocking per device would serialise exactly what the shard split parallelises,
    /// and spawning a thread per device per block (the first cut) churned hundreds of OS threads a
    /// second — visible as `ThreadId(867540)` in the logs.
    pub(super) fn submit_on<T, F>(dev: usize, f: F) -> Pending<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = mpsc::sync_channel::<(std::thread::Result<T>, f64, f64)>(1);
        let depth = DEPTH.fetch_add(1, Ordering::Relaxed) + 1;
        let enqueued = Instant::now();
        senders()[dev]
            .send(Box::new(move || {
                let queue_ms = enqueued.elapsed().as_secs_f64() * 1e3;
                DEPTH.fetch_sub(1, Ordering::Relaxed);
                let started = Instant::now();
                // `AssertUnwindSafe`: on a panic the payload is forwarded and the worker resumes
                // the unwind, so no state observed after the catch is reused here.
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
                let exec_ms = started.elapsed().as_secs_f64() * 1e3;
                // A send error means the worker vanished (itself panicking); drop the result.
                let _ = tx.send((out, queue_ms, exec_ms));
            }))
            .expect("the nassau-gpu thread died");
        Pending { rx, depth }
    }

    /// Submit to `dev` and block for the result.
    pub(super) fn run_on<T, F>(dev: usize, f: F) -> (T, Timing)
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        submit_on(dev, f).wait()
    }
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
/// until the budget admits it. The device section runs on the shared GPU thread (see [`gpu_thread`]),
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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Set once the cubecl CUDA context has failed irrecoverably (a `CUDA_ERROR_LAUNCH_FAILED` /
/// `ServerUnhealthy` surfacing the unresolved cubecl uninit-handle bug — see
/// `~/cubecl-uninit-handle-followup.md`, tracel-ai/cubecl#1401). Such a failure **poisons the whole
/// CUDA context**: every later launch on the shared client fails too, so there is no per-call retry.
/// [`multiply_batch_on_gpu`] latches this on the first failure and then *propagates the panic* — the
/// run dies at the fault rather than silently finishing on the CPU, so a crash cannot masquerade as a
/// slow success. The flag exists purely so in-process observers (the soak test) can distinguish a
/// context death from an ordinary assertion failure. NOTE: this covers only the cubecl **multiply**;
/// the RREF path runs on a separate `fp-cuda` runtime and is not gated by this flag.
static GPU_DISABLED: AtomicBool = AtomicBool::new(false);

/// Whether the GPU multiply has been disabled for the rest of the process (see [`GPU_DISABLED`]).
pub fn gpu_disabled() -> bool {
    GPU_DISABLED.load(Ordering::Relaxed)
}

/// Global launch counter for throttling per-launch [`memory_cleanup`] (see [`cleanup_every`]).
static CLEANUP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// How often to call `client.memory_cleanup()` after a multiply launch, via
/// `NASSAU_GPU_CLEANUP_EVERY` (default `1` = every launch). `N` cleans every Nth launch; `0` never
/// cleans. DIAGNOSTIC: the residual `CUDA_ERROR_LAUNCH_FAILED` at high stems is consistent with a
/// cross-stream pool reclaim (one stream's cleanup reclaiming a resident-master page still in flight
/// on another stream — tracel-ai/cubecl#1401). Throttling this drastically cuts that reclaim rate; if
/// the crash disappears or moves much later, cleanup is confirmed as the trigger. The tradeoff is
/// device-memory growth, since freed pages linger — watch `nvidia-smi`.
fn cleanup_every() -> u64 {
    use std::sync::OnceLock;
    static EVERY: OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("NASSAU_GPU_CLEANUP_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    })
}

/// Aggregate [`multiply_batch_on_gpu`] counters across all launches (call count, host
/// marshal µs, device µs, total pairs), for splitting a whole resolution's GPU overhead.
static BATCH_CALLS: AtomicU64 = AtomicU64::new(0);
/// First-sight `R`s that forced an `admissible_matrices` enumeration + a `RESIDENT_HOST` write
/// lock (see [`resident_info`]). Diffed around the pair pre-pass to attribute its cost.
static RESIDENT_MISSES: AtomicU64 = AtomicU64::new(0);
static BATCH_MARSHAL_US: AtomicU64 = AtomicU64::new(0);
static BATCH_DEVICE_US: AtomicU64 = AtomicU64::new(0);
static BATCH_PAIRS: AtomicU64 = AtomicU64::new(0);
/// `BATCH_MARSHAL_US` split: host CPU work before any blocking, and time parked on the
/// [`GpuPermit`] / [`fp::gpu_lock`] acquisition. Conflating them hid which one dominates.
static BATCH_PREP_US: AtomicU64 = AtomicU64::new(0);
static BATCH_WAIT_US: AtomicU64 = AtomicU64::new(0);
/// `BATCH_WAIT_US` split again: the pre-existing [`GpuPermit`] (bounds in-flight output bytes)
/// versus the cross-runtime [`fp::gpu_lock`] arbitration. They have different owners and fixes.
static BATCH_PERMIT_US: AtomicU64 = AtomicU64::new(0);
static BATCH_LOCK_US: AtomicU64 = AtomicU64::new(0);
/// `BATCH_DEVICE_US` split in two by the dedicated GPU thread (see [`gpu_thread`]): time spent
/// waiting in the submission FIFO versus time the device section actually ran. The single
/// `device` figure could not tell "queued behind other work" from "computing", which is exactly
/// the distinction the 370 s stalls turned on.
static BATCH_QUEUE_US: AtomicU64 = AtomicU64::new(0);
static BATCH_EXEC_US: AtomicU64 = AtomicU64::new(0);
/// Queue depth summed over launches (÷ calls = mean depth) and its high-water mark.
static BATCH_DEPTH_SUM: AtomicU64 = AtomicU64::new(0);
static BATCH_DEPTH_MAX: AtomicU64 = AtomicU64::new(0);
/// The caller's wait, split at the point the GPU worker hands back the readback future: time until
/// this block was *launched* versus time waiting on its completion fence. `launch` counts jobs
/// queued ahead of this one on the worker, `fence` counts the device actually working — so
/// `fence >> launch` means the pipeline is full and `launch >> fence` means it is starved. Kept
/// separate from `queue`/`exec`, which measure the same run from the worker's side.
static BATCH_LAUNCH_US: AtomicU64 = AtomicU64::new(0);
static BATCH_FENCE_US: AtomicU64 = AtomicU64::new(0);
/// `prep` split three ways, to decide whether marshalling is worth a representation change: the
/// per-launch `R` intern, the `term_off`/`ensure_basis` middle, and the `term_gei` fill. Each has a
/// different fix, and the single `prep` number could not distinguish them.
static BATCH_INTERN_US: AtomicU64 = AtomicU64::new(0);
static BATCH_BASIS_US: AtomicU64 = AtomicU64::new(0);
static BATCH_TGEI_US: AtomicU64 = AtomicU64::new(0);

/// Read and reset the aggregate batch counters: `(calls, marshal_us, device_us, pairs)`.
pub fn take_batch_stats() -> (u64, u64, u64, u64) {
    (
        BATCH_CALLS.swap(0, Ordering::Relaxed),
        BATCH_MARSHAL_US.swap(0, Ordering::Relaxed),
        BATCH_DEVICE_US.swap(0, Ordering::Relaxed),
        BATCH_PAIRS.swap(0, Ordering::Relaxed),
    )
}

/// Where multiply time goes, as microsecond totals: host prep, then the GPU-thread split of
/// queue wait versus device execution, then queue depth (summed, max). See [`gpu_thread`].
///
/// Separate from [`take_batch_stats`] because the queue/exec split is the measurement that
/// distinguishes "waiting behind other workers" from "computing" — collapsing them into one
/// `device` figure is what made multi-minute submission stalls read as kernel time.
pub fn take_gpu_timing() -> (u64, u64, u64, u64, u64) {
    (
        BATCH_PREP_US.swap(0, Ordering::Relaxed),
        BATCH_QUEUE_US.swap(0, Ordering::Relaxed),
        BATCH_EXEC_US.swap(0, Ordering::Relaxed),
        BATCH_DEPTH_SUM.swap(0, Ordering::Relaxed),
        BATCH_DEPTH_MAX.swap(0, Ordering::Relaxed),
    )
}

/// Diagnostic (see `NASSAU_MEM_REPORT`): resident-master HOST-side heap bytes — the not-yet-uploaded
/// `col_sums`/`masks` tails, the width-padded basis `pparts`/`lens`, and the per-`R` `index` map
/// (its `Vec<PPartEntry>` keys). The bulk `col_sums`/`masks` are no longer retained (freed after
/// upload — see [`ResidentHost`]); only the pending tail + the `index` persist. Returns `(master, basis)`.
pub fn resident_host_bytes() -> (usize, usize) {
    let h = RESIDENT_HOST.read().unwrap();
    let master = h.cs_pending.iter().map(|p| p.capacity()).sum::<usize>() * 2
        + h.mk_pending.iter().map(|p| p.capacity()).sum::<usize>() * 2
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
    // Summed over every device: each holds its own replica of the resident master and basis.
    let master: usize = RESIDENT_DEV
        .iter()
        .map(|d| {
            let d = d.read().unwrap();
            d.cs.uploaded * 2 + d.mk.uploaded * 2
        })
        .sum();
    let basis: usize = RESIDENT_BASIS_DEV
        .iter()
        .map(|b| {
            let b = b.read().unwrap();
            b.pp.uploaded * 2 + b.ln.uploaded * 4
        })
        .sum();
    (master, basis)
}

/// Diagnostic (see `NASSAU_MEM_REPORT`): the cubecl CUDA memory pool's device usage on the default
/// device, `(bytes_in_use, bytes_reserved)`. This is the batched-multiply pool; the fp-cuda RREF runs
/// on a separate cudarc context, so `nvidia-smi total − resident_dev − reserved` estimates the RREF
/// pool. Returns `(0, 0)` if the query fails.
pub fn cubecl_device_usage() -> (u64, u64) {
    let client = gpu_client();
    match client.memory_usage() {
        Ok(u) => (u.bytes_in_use, u.bytes_reserved),
        Err(_) => (0, 0),
    }
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
    /// Global element offsets into the shared master (`u64`: the master exceeds `u32::MAX`
    /// u16 elements at high stems, so the offset itself must be 64-bit — `multiply_batch_kernel`
    /// reads `col_sums`/`masks` at these offsets under 64-bit `address_type`).
    cs_off: u64,
    mk_off: u64,
    cs_len: u32,
    mk_len: u32,
    num_mats: u32,
    /// Which device holds this `R`'s rows. The master is SHARDED, not replicated: each `R` lives on
    /// exactly one device, so total device memory is one master spread over `gpu_count()` cards
    /// rather than a full copy on each. That is what lifts the memory ceiling (aggregate VRAM
    /// instead of per-card VRAM) and keeps the upload cost at 1x rather than Nx.
    ///
    /// The offsets above are therefore into THIS DEVICE's master, not a global one.
    dev: u8,
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
/// (~27 GB at stem 130, growing). Invariant maintained by [`seg_grow`]:
/// `RESIDENT_DEV.$buf.uploaded == $len - $pending.len()`, i.e. `$pending == master[uploaded..$len]`.
/// `*_pending` is per DEVICE. The uploaded prefix is dropped host-side, so what remains is only the
/// tail each device has yet to consume; with several devices each needs its own copy of that tail,
/// because a device replicates the master rather than sharing it. This multiplies the TAIL, not the
/// master: the multi-GB host-side duplicate this design exists to avoid stays gone.
struct ResidentHost {
    cs_pending: Vec<Vec<u16>>,
    mk_pending: Vec<Vec<u16>>,
    /// Per-device logical master lengths; an `R` extends only its own device's.
    cs_len: Vec<usize>,
    mk_len: Vec<usize>,
    /// Accumulated `num_mats` per device — the work proxy the shard assignment balances.
    ///
    /// A launch's work on a device is the sum over its products of `num_mats(R) * ceil(nt/T)`, so
    /// with `R`s used at broadly similar rates the device's share is set by the `num_mats` it owns.
    /// Master bytes are `num_mats * (cs_len + mk_len)`, so balancing this also tracks memory.
    dev_load: Vec<u64>,
    index: HashMap<PPart, RInfo>,
}

static RESIDENT_HOST: LazyLock<RwLock<ResidentHost>> = LazyLock::new(|| {
    RwLock::new(ResidentHost {
        cs_pending: (0..gpu_count()).map(|_| Vec::new()).collect(),
        mk_pending: (0..gpu_count()).map(|_| Vec::new()).collect(),
        cs_len: vec![0; gpu_count()],
        mk_len: vec![0; gpu_count()],
        dev_load: vec![0; gpu_count()],
        index: HashMap::new(),
    })
});

/// Hard cap on devices, so the per-device tables below are a fixed, cheap allocation.
const MAX_GPUS: usize = 8;

/// How many CUDA devices the multiply path spreads work over. `NASSAU_GPU_DEVICES` overrides;
/// otherwise every device the driver exposes is used.
///
/// Multi-GPU is worth it here because the single-device run is GPU-bound, not host-bound: whole-run
/// accounting on stem 200 measured 3302 s of device execution against 3931 s wall (84% duty), so
/// eliminating *all* host work would cap out at 1.19x while `629 + 3302/N` predicts 1.72x at N = 2
/// and 2.70x at N = 4.
fn gpu_count() -> usize {
    static N: LazyLock<usize> = LazyLock::new(|| {
        let detected = std::fs::read_dir("/proc/driver/nvidia/gpus")
            .map(|d| d.filter_map(|e| e.ok()).count())
            .unwrap_or(0)
            .max(1);
        std::env::var("NASSAU_GPU_DEVICES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(detected)
            .clamp(1, MAX_GPUS)
    });
    *N
}

thread_local! {
    /// Which device the current thread's GPU work belongs to. Set once per GPU worker thread; every
    /// other thread sees 0 and never touches device state directly.
    static CUR_DEVICE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The device this thread's GPU work runs on. Device handles are NOT interchangeable across
/// devices, so every resident-state accessor and every client is keyed by this.
fn cur_device() -> usize {
    CUR_DEVICE.with(|c| c.get())
}

/// The cubecl client for this thread's device, borrowed for the process's lifetime.
///
/// `'static` on purpose: [`ComputeClient::read_async`] returns a future that borrows the client
/// (edition 2024 RPIT captures `&self`), and that future must outlive the GPU worker's task so the
/// readback can be awaited by the *caller* rather than on the worker — see [`multiply_batch_block`].
/// A per-call clone would make the future borrow a local and pin the wait to the worker thread,
/// which is exactly the one-kernel-deep pipeline this indirection removes. Constructing each
/// device's client once also drops a `CudaRuntime::client` lookup from every launch.
fn gpu_client() -> &'static cubecl::prelude::ComputeClient<CudaRuntime> {
    static CLIENTS: std::sync::OnceLock<Vec<cubecl::prelude::ComputeClient<CudaRuntime>>> =
        std::sync::OnceLock::new();
    &CLIENTS.get_or_init(|| {
        (0..gpu_count())
            .map(|d| CudaRuntime::client(&CudaDevice::new(d)))
            .collect()
    })[cur_device()]
}

/// Compile-time cap on the number of fixed-size segments a resident device buffer may hold. It
/// bounds both the multiply kernel's per-buffer argument count and the [`seg_read_u16`] /
/// [`seg_read_u32`] branch depth, so it must be a constant. With `master_seg_elems()` at its default
/// `2^31`, this holds `16 × 4 GiB = 64 GiB` of u16 master per buffer — well past what fits resident
/// on one H200. Raise it (and extend the two `seg_read_*` / the kernel binding) for larger buffers.
const MASTER_MAX_SEG: usize = 16;

/// Element count per resident segment (see [`SegBuf`]); env `NASSAU_GPU_MASTER_SEG_ELEMS`. Default
/// `2^31` — 4 GiB per u16 segment, 8 GiB per u32 — and deliberately `< u32::MAX`, so a single
/// segment's length never overflows cubecl's 32-bit array-length metadata (the truncation class of
/// bug the u64 offset addressing already guards against). Tests set it tiny to exercise many-segment
/// gathers at low degree; production leaves it large so a run needs only a handful of segments.
fn master_seg_elems() -> usize {
    static N: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("NASSAU_GPU_MASTER_SEG_ELEMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1usize << 31)
    });
    *N
}

/// A resident device buffer grown by APPENDING fixed-size segments — the existing segments are never
/// reallocated or copied, so the device peak is `live + one_segment`, not the `~2×` realloc-doubling
/// transient that pushed cubecl into its silent memory-corruption regime (the stem-140+ `dx != 0`).
/// Each segment holds exactly `master_seg_elems()` elements (allocated full; the last is only
/// partially written); the multiply kernel selects the segment for a global offset `o` by a static
/// branch (`o / seg_elems`), so at most [`MASTER_MAX_SEG`] segments exist. Append-only and stable —
/// a segment handle, once allocated and written, never changes identity and is never freed — which
/// is the ordinary shared-global (model-weights) pattern cubecl syncs correctly across streams; the
/// churny "swap the whole buffer on every growth" it replaced broke that sync. `uploaded` is how
/// many elements are physically resident across all segments.
#[derive(Default)]
struct SegBuf {
    segs: Vec<Handle>,
    uploaded: usize,
}

/// Process-shared device mirror of the host master. Each buffer is a segmented, append-only
/// no-copy-growth store (see [`SegBuf`], [`seg_grow`]). This is what makes the master safe to share
/// across streams: the churny "re-upload a new handle on every growth" it replaced broke cubecl's
/// per-handle cross-stream sync (crash rate tracked re-upload frequency); a stable segment written in
/// place is the ordinary shared-global (model-weights) pattern. Reads go through `RESIDENT_DEV.read()`
/// (lock-free fan-out); growth runs outside that lock, serialized only by `RESIDENT_UPLOAD`.
#[derive(Default)]
struct ResidentDev {
    cs: SegBuf,
    mk: SegBuf,
}

/// One per device: a `Handle` allocated on device `i` is meaningless on device `j`, so the resident
/// master is replicated rather than shared. The HOST master ([`RESIDENT_HOST`]) stays single-copy,
/// which is what keeps the multi-GB host-side duplicate from multiplying by `gpu_count()`.
static RESIDENT_DEV: LazyLock<Vec<RwLock<ResidentDev>>> = LazyLock::new(|| {
    (0..gpu_count())
        .map(|_| RwLock::new(ResidentDev::default()))
        .collect()
});

fn resident_dev() -> &'static RwLock<ResidentDev> {
    &RESIDENT_DEV[cur_device()]
}

/// Serializes master device *uploads* only — never segment reads. A launch that must grow the
/// device master takes this before uploading, so at a growth point at most one grower runs (others
/// re-check and find it already done) instead of every launch piling redundant copies. Reads go
/// lock-free through `RESIDENT_DEV.read()`, so the upload no longer blocks other bidegrees' device
/// sections (the old single mutex held across the copy collapsed the whole wavefront to one
/// memcpy-ing thread).
static RESIDENT_UPLOAD: LazyLock<Vec<Mutex<()>>> =
    LazyLock::new(|| (0..gpu_count()).map(|_| Mutex::new(())).collect());

fn resident_upload() -> &'static Mutex<()> {
    &RESIDENT_UPLOAD[cur_device()]
}

/// Shared resident device copies of the read-only seqno table `g` and the (constant) `xi` degrees.
/// These are identical across every launch at a given built degree, so re-uploading them per launch
/// (a `create_from_slice` each) was pure churn — one of the per-launch allocation/copy streams that
/// pushed cubecl's allocator into its `CUDA_ERROR_LAUNCH_FAILED` (719) failure at scale. Uploaded
/// once and re-uploaded only when `g` grows to a new max degree. Keyed by `g.len()`: `g` is a
/// deterministic function of the built degree, so equal length ⇒ identical bytes. Read-only and
/// shared cross-stream exactly like the resident master.
struct SeqnoDev {
    g_len: usize,
    g: Handle,
    xi: Handle,
}
static RESIDENT_SEQNO: LazyLock<Vec<RwLock<Option<SeqnoDev>>>> =
    LazyLock::new(|| (0..gpu_count()).map(|_| RwLock::new(None)).collect());

fn resident_seqno() -> &'static RwLock<Option<SeqnoDev>> {
    &RESIDENT_SEQNO[cur_device()]
}
/// Serializes seqno-table uploads only (never reads); see [`RESIDENT_UPLOAD`].
static RESIDENT_SEQNO_UPLOAD: LazyLock<Vec<Mutex<()>>> =
    LazyLock::new(|| (0..gpu_count()).map(|_| Mutex::new(())).collect());

fn resident_seqno_upload() -> &'static Mutex<()> {
    &RESIDENT_SEQNO_UPLOAD[cur_device()]
}

/// Fetch the shared resident `(g, xi)` device handles, uploading only when the cached table's length
/// differs from `$g` (i.e. the built degree changed). Lock-free fast path; a burst of first-sight
/// launches coalesces behind `RESIDENT_SEQNO_UPLOAD`. The upload is synced before publishing so a
/// cross-stream reader never observes the handles before their H2D copy completes.
macro_rules! resident_seqno {
    ($client:expr, $g:expr, $xi:expr) => {{
        let read_current = || {
            let s = resident_seqno().read().unwrap();
            match &*s {
                Some(d) if d.g_len == $g.len() => Some((d.g.clone(), d.xi.clone())),
                _ => None,
            }
        };
        match read_current() {
            Some(h) => h,
            None => {
                let _upload_guard = resident_seqno_upload().lock().unwrap();
                match read_current() {
                    Some(h) => h,
                    None => {
                        let gh = $client.create_from_slice(u32::as_bytes(&$g));
                        let xh = $client.create_from_slice(u32::as_bytes(&$xi));
                        // Make the copies physically resident before publishing (cross-stream reads).
                        let _ = cubecl_common::reader::read_sync($client.sync());
                        *resident_seqno().write().unwrap() = Some(SeqnoDev {
                            g_len: $g.len(),
                            g: gh.clone(),
                            xi: xh.clone(),
                        });
                        (gh, xh)
                    }
                }
            }
        }
    }};
}

/// Host-side cache of cold (degree > [`resident_degree_cap`]) `R`s' admissible-matrix *shape* only —
/// `(cs_len, mk_len, num_mats)`, twelve bytes per `R`. With [in-kernel enumeration](enumerate_admissible_kernel)
/// the cold `col_sums`/`masks` are generated ON the device into transient scratch, so the host never
/// stores (nor uploads) the arrays themselves — only their sizes, needed up front to lay out the
/// scratch offsets and the pair-count prefix sum before the launch. This is the memory win over the
/// old array cache: the evicted tail of the master (tens of GB) lives neither on the device nor the
/// host. The count is computed once per distinct `R` (via `admissible_matrices`, whose arrays are
/// dropped immediately) and memoized, so the per-launch cost is an `O(1)` lookup.
static COLD_COUNT: LazyLock<RwLock<HashMap<PPart, (u32, u32, u32)>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Cold-`R` admissible-matrix shape `(cs_len, mk_len, num_mats)` from the [`COLD_COUNT`] cache. On a
/// miss it runs `admissible_matrices` purely to *count* (the returned arrays are dropped, not kept —
/// the device enumerates them), then memoizes the triple. Layout matches [`resident_info`]'s so the
/// kernel indexes the on-device-enumerated scratch identically to the resident master.
fn cold_count(algebra: &MilnorAlgebra, p_part: PPart) -> (u32, u32, u32) {
    if let Some(&e) = COLD_COUNT.read().unwrap().get(&p_part) {
        return e;
    }
    let (cs_len, mk_len, _cs, mk) = algebra.admissible_matrices(p_part);
    let e = (cs_len as u32, mk_len as u32, (mk.len() / mk_len) as u32);
    COLD_COUNT.write().unwrap().entry(p_part).or_insert(e);
    e
}

/// Grow a segmented resident device buffer ([`SegBuf`]) so it covers `$need` elements, and return
/// `(segments, uploaded)` — the per-segment handles (each `master_seg_elems()` elements) and the
/// logical resident length. NO-COPY growth: existing segments are never reallocated or copied; a
/// growth only allocates the new segment(s) it needs and stage-writes the not-yet-uploaded tail into
/// them. This replaces the `~2×` realloc-doubling transient (old+new buffer both live) that pushed
/// cubecl into memory corruption — the stem-140+ `dx != 0` — with a `live + one_segment` peak.
///
/// Concurrency mirrors the old in-place path: a lock-free fast path returns the current segments when
/// they already cover `$need`; otherwise `$upload` (a `Mutex`) serializes growers, with a re-check
/// coalescing a burst. The long stage-write runs on a CLONE of the segment vector (segment handles
/// are refcounted, so cloning shares the buffers) and is published under a brief `$dev` write lock,
/// so readers taking `$dev.read()` always see a consistent `(segs, uploaded)` pair. Segments are
/// append-only and never freed, so a reader's cloned handle stays valid for its whole kernel with no
/// realloc barrier. `$tail` is `|uploaded| -> (Vec<$elem>, new_len)`: the owned tail
/// `master[uploaded..new_len]` (the master frees it host-side via `mem::take`; the basis copies it
/// out of its retained store) plus the new logical length.
macro_rules! seg_grow {
    ($client:expr, $dev:expr, $field:ident, $upload:expr, $need:expr,
     $copy:ident, $as_bytes:path, $elem:ty, $tail:expr) => {{
        let seg_elems = master_seg_elems();
        // Lock-free fast path: current segments already cover `$need`.
        let read_current = || {
            let dev = $dev.read().unwrap();
            if dev.$field.uploaded >= $need {
                Some((dev.$field.segs.clone(), dev.$field.uploaded))
            } else {
                None
            }
        };
        match read_current() {
            Some(su) => su,
            None => {
                let _upload_guard = $upload.lock().unwrap();
                match read_current() {
                    Some(su) => su, // another grower already covered our need
                    None => {
                        // Snapshot the current segments + logical length. We extend a CLONE and
                        // publish it atomically, so a concurrent reader sees either the whole old
                        // state or the whole new one — never a half-grown vector.
                        let (mut segs, uploaded): (Vec<Handle>, usize) = {
                            let dev = $dev.read().unwrap();
                            (dev.$field.segs.clone(), dev.$field.uploaded)
                        };
                        let (tail, new_len): (Vec<$elem>, usize) = ($tail)(uploaded);
                        debug_assert_eq!(uploaded + tail.len(), new_len);
                        assert!(
                            new_len.div_ceil(seg_elems.max(1)) <= MASTER_MAX_SEG,
                            "resident buffer needs {} segments (> MASTER_MAX_SEG={}); raise \
                             MASTER_MAX_SEG or NASSAU_GPU_MASTER_SEG_ELEMS",
                            new_len.div_ceil(seg_elems.max(1)),
                            MASTER_MAX_SEG
                        );
                        // Allocate (no copy) full-size segments until they cover `new_len`. The last
                        // one is allocated full even if only partially written; reads only touch
                        // written locals (`< uploaded`), so its uninitialized tail is never read.
                        while segs.len() * seg_elems < new_len {
                            segs.push($client.empty(seg_elems * ::core::mem::size_of::<$elem>()));
                        }
                        // Stage-write the tail `master[uploaded..new_len]` into its segments, split at
                        // segment boundaries and [`STAGE_CHUNK`], syncing each chunk. Existing
                        // segments (including the partially-filled last one) are appended into, never
                        // copied. The sync makes each chunk physically resident before the bumped
                        // `uploaded` is published, so a cross-stream reader never observes a gap
                        // (cubecl does not order a kernel write to a shared buffer against another
                        // stream's read the way `create_from_slice` does).
                        let mut pos = uploaded;
                        let mut done = 0usize;
                        while pos < new_len {
                            let seg = pos / seg_elems;
                            let local = pos % seg_elems;
                            let m = (new_len - pos).min(seg_elems - local).min(STAGE_CHUNK);
                            let scratch =
                                $client.create_from_slice($as_bytes(&tail[done..done + m]));
                            copy_chunked!(
                                $client, $copy, scratch, m, 0usize, segs[seg], seg_elems, local, m
                            );
                            let _ = cubecl_common::reader::read_sync($client.sync());
                            pos += m;
                            done += m;
                        }
                        {
                            let mut dev = $dev.write().unwrap();
                            dev.$field.segs = segs.clone();
                            dev.$field.uploaded = new_len;
                        }
                        (segs, new_len)
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

static R_STATS: LazyLock<Option<Mutex<HashMap<PPart, RStat>>>> =
    LazyLock::new(|| std::env::var_os("NASSAU_R_STATS").map(|_| Mutex::new(HashMap::new())));

/// Internal degree of `R` from its p-part: `Σ p_part[i] · deg(ξ_{i+1})`.
fn ppart_degree(p_part: PPart) -> i32 {
    let xi = xi_degrees(fp::prime::ValidPrime::new(2));
    p_part
        .iter()
        .zip(xi.iter())
        .map(|(e, &d)| e as i32 * d as i32)
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

fn record_r_use(p_part: PPart) {
    if let Some(m) = R_STATS.as_ref() {
        let now = BATCH_CALLS.load(Ordering::Relaxed);
        let mut map = m.lock().unwrap();
        let e = map.entry(p_part).or_insert(RStat {
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
        let refs: u64 = v
            .iter()
            .filter(|s| s.degree <= theta)
            .map(|s| s.count)
            .sum();
        deg_table += &format!(
            " θ≤{theta}:[{:.0}%Rs,{:.0}%refs]",
            held as f64 / n as f64 * 100.0,
            refs as f64 / total_refs as f64 * 100.0
        );
    }
    eprintln!(
        "[R-STATS] distinct_R={n} total_refs={total_refs} used_once={once} ({:.0}%) \
         k_for_90%_refs={k90} ({:.1}% of Rs) | coverage top1%={:.0}% top5%={:.0}% top10%={:.0}% \
         top25%={:.0}% | degree hot_decile_avg={:.0} cold_decile_avg={:.0} \
         range=[{min_deg},{max_deg}] | ref_span hot={:.2} cold={:.2} (of run) | degree-threshold \
         cache sizing:{}",
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

fn resident_info(algebra: &MilnorAlgebra, p_part: PPart) -> RInfo {
    record_r_use(p_part);
    if let Some(info) = RESIDENT_HOST.read().unwrap().index.get(&p_part) {
        return *info;
    }
    // Miss: enumerate the admissible matrices (expensive, CPU) and then append under the WRITE
    // lock. Counted so [`multiply_batch_grouped`]'s pre-pass can report how many misses it paid
    // for — the pre-pass cost is entirely a function of first-sight `R`s, which makes it depend on
    // what *other* bidegrees warmed earlier, i.e. on run order rather than on this step's work.
    RESIDENT_MISSES.fetch_add(1, Ordering::Relaxed);
    let (cs_len, mk_len, cs, mk) = algebra.admissible_matrices(p_part);
    let mut host = RESIDENT_HOST.write().unwrap();
    if let Some(info) = host.index.get(&p_part) {
        return *info;
    }
    // Offsets are the running LOGICAL lengths (`*_len`), not the pending-buffer lengths — the
    // uploaded prefix has been freed but the logical numbering is permanent (see [`ResidentHost`]).
    // Assign this `R` to the least-loaded device. Its rows go there and nowhere else, so a launch
    // must route products to the device owning their `R`.
    //
    // Round-robin over first-sight order was the first cut and balances COUNT, not work — `num_mats`
    // varies by orders of magnitude between `R`s, so equal counts left the devices badly uneven
    // (batch-stats: queue 67% / exec 32%, mean depth 8.4, i.e. devices waiting while others ran).
    // Greedy least-loaded is the standard fix and needs no lookahead.
    let num_mats = (mk.len() / mk_len) as u64;
    let dev = host
        .dev_load
        .iter()
        .enumerate()
        .min_by_key(|&(i, &load)| (load, i))
        .map(|(i, _)| i)
        .expect("at least one device");
    host.dev_load[dev] += num_mats;
    let info = RInfo {
        cs_off: host.cs_len[dev] as u64,
        mk_off: host.mk_len[dev] as u64,
        cs_len: cs_len as u32,
        mk_len: mk_len as u32,
        num_mats: num_mats as u32,
        dev: dev as u8,
    };
    host.cs_pending[dev].extend(cs.iter().map(|&v| narrow_u16(v)));
    host.mk_pending[dev].extend(mk.iter().map(|&v| narrow_u16(v)));
    host.cs_len[dev] += cs.len();
    host.mk_len[dev] += mk.len();
    host.index.insert(p_part, info);
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

/// Device mirror of [`ResidentBasisHost`], both buffers segmented no-copy-growth stores (see
/// [`SegBuf`], [`seg_grow`]). `pp` holds the width-padded p-parts (`elems * width` u16), `ln` the
/// lengths (`elems` u32); the basis element count is `ln.uploaded`.
#[derive(Default)]
struct ResidentBasisDev {
    pp: SegBuf,
    ln: SegBuf,
}

static RESIDENT_BASIS_DEV: LazyLock<Vec<RwLock<ResidentBasisDev>>> = LazyLock::new(|| {
    (0..gpu_count())
        .map(|_| RwLock::new(ResidentBasisDev::default()))
        .collect()
});

fn resident_basis_dev() -> &'static RwLock<ResidentBasisDev> {
    &RESIDENT_BASIS_DEV[cur_device()]
}

/// Serializes basis device *uploads* only (never handle reads); see [`RESIDENT_UPLOAD`].
static RESIDENT_BASIS_UPLOAD: LazyLock<Vec<Mutex<()>>> =
    LazyLock::new(|| (0..gpu_count()).map(|_| Mutex::new(())).collect());

fn resident_basis_upload() -> &'static Mutex<()> {
    &RESIDENT_BASIS_UPLOAD[cur_device()]
}

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
            for (slot, v) in host.pparts[base..base + width]
                .iter_mut()
                .zip(elt.p_part.iter())
            {
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

/// Zero a device `u32` buffer on-device: `out[i] = 0`, one thread per limb.
///
/// Initializes the batched multiply's XOR accumulator without allocating and uploading a host
/// zero buffer. Profiling (stem 145) showed the per-launch `create_from_slice` of a
/// hundreds-of-MB `out_h` zero vec — a host `memset` + non-pinned host→device `memcpy`, both on
/// the calling rayon worker — was the dominant serial marshaling cost, stalling the wavefront.
/// On-device zeroing is memory-bound (microseconds on an H200) and same-stream ordered before
/// the multiply kernel, so no host allocation, upload, or extra sync is needed.
#[cube(launch)]
fn zero_u32(out: &mut [u32]) {
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
fn copy_into_u16(src: &[u16], dst: &mut [u16], src_off: usize, dst_off: usize, count: u32) {
    if ABSOLUTE_POS < usize::cast_from(count) {
        dst[dst_off + ABSOLUTE_POS] = src[src_off + ABSOLUTE_POS];
    }
}

/// `u32` sibling of [`copy_into_u16`] (for the resident basis `lens`).
#[cube(launch_unchecked, address_type = "dynamic")]
fn copy_into_u32(src: &[u32], dst: &mut [u32], src_off: usize, dst_off: usize, count: u32) {
    if ABSOLUTE_POS < usize::cast_from(count) {
        dst[dst_off + ABSOLUTE_POS] = src[src_off + ABSOLUTE_POS];
    }
}

/// Elements per copy-kernel launch: below the kernel's `u32` `ABSOLUTE_POS` thread limit, so copies
/// of multi-billion-element resident buffers are split into this many at a time.
const COPY_CHUNK: usize = 1 << 30;

/// Elements per pinned host-staging chunk when uploading resident growth (see [`seg_grow`]).
/// Bounds the page-locked host buffer cubecl reserves per `create_from_slice`: those pinned pages
/// are pooled PER CUDA STREAM and never trimmed, so a single full-master `create_from_slice` (the
/// tail can be many GB) would pin that whole size on every stream — measured ~240 GB shmem at stem
/// 180 with 8 streams, the OOM driver. Staging in `STAGE_CHUNK` pieces with a sync between them
/// caps the live pinned staging at ~one chunk. 64 Mi × u16 = 128 MiB (× u32 = 256 MiB).
const STAGE_CHUNK: usize = 1 << 26;

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
                    BufferArg::from_raw_parts($src.clone(), $src_len),
                    BufferArg::from_raw_parts($dst.clone(), $dst_len),
                    $src_off + done,
                    $dst_off + done,
                    n as u32,
                );
            }
            done += n;
        }
    }};
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
fn seqno_core_packed(
    g: &[u32],
    xi: &[u32],
    pp_shift: &[u32],
    pp_mask: &[u32],
    working: u64,
    wlen: usize,
    width: usize,
) -> u32 {
    // cur_d = Σ working[h] · xi[h], reading entries out of the packed word.
    let mut cur_d = 0u32;
    for h in 0..wlen {
        let e =
            u32::cast_from((working >> u64::cast_from(pp_shift[h])) & u64::cast_from(pp_mask[h]));
        cur_d += e * xi[h];
    }

    // Rank by consuming positions from high to low; position 0 contributes nothing.
    let mut rank = 0u32;
    for hh in 1..wlen {
        let h = wlen - hh; // wlen-1 down to 1
        let r =
            u32::cast_from((working >> u64::cast_from(pp_shift[h])) & u64::cast_from(pp_mask[h]));
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
/// How many columns one `(matrix, term)` pair actually has to visit: past the longest of the three
/// inputs, `b`, `cs` and `mk` are all zero, so [`pair_col`] returns 0 — no rejection and nothing
/// added to `working`. Stopping there is exact, not a truncation.
///
/// This is per THREAD. The loop used to run to the launch's comptime `work_cap`, which is the max
/// `mk_len` over every `R` in the block, so one long `R` made every thread in the launch pay for
/// columns its own data does not have: measured `work_cap = 16` against a mean `mk_len` of 9.9.
/// Ablation put ~59% of kernel time in this loop (shortening it 16 -> 4 was +80%), so the waste was
/// the single largest item in the kernel.
#[cube]
fn pair_cols(term_len: usize, cs_len: usize, mk_len: usize) -> usize {
    let mut cols = cs_len;
    if term_len > cols {
        cols = term_len;
    }
    if mk_len > cols {
        cols = mk_len;
    }
    cols
}

/// Bit [`pair_col`] sets to report that a column rejects the whole product, above the 16 bits the
/// value itself occupies (`diff | mk` and `b | mk` are all widened from `u16`).
///
/// Packing the flag into the return value rather than signalling it out of band keeps the caller
/// branchless: it ORs the flag into an accumulator and shifts the low half into `working`
/// unconditionally, exactly as the pre-refactor code did. Guarding the accumulate on a rejection
/// test instead costs ~11% — twelve divergent branches per thread, one per column.
const PAIR_COL_REJECT: u32 = 1 << 16;

/// The per-column rule of [`multiply_pair`], factored out so a caller that reads `b`/`cs`/`mk`
/// from somewhere other than three contiguous slices can reuse it verbatim. Returns the assembled
/// `working[j]` in the low 16 bits, with [`PAIR_COL_REJECT`] set if this column kills the product
/// (in which case the value half is meaningless — the caller discards the whole product).
#[cube]
fn pair_col(j: usize, low: usize, b: u32, cs: u32, mk: u32) -> u32 {
    let mut val = 0u32;
    if j < low {
        if cs > b {
            val = PAIR_COL_REJECT;
        } else {
            let diff = b - cs;
            if (diff & mk) != 0u32 {
                val = PAIR_COL_REJECT;
            } else {
                val = diff | mk;
            }
        }
    } else {
        if cs > 0u32 {
            val = PAIR_COL_REJECT;
        } else if (b & mk) != 0u32 {
            val = PAIR_COL_REJECT;
        } else {
            val = b | mk;
        }
    }
    val
}

/// Tail of [`multiply_pair`] for an accepted product: index the assembled p-part and XOR its
/// F₂ bit into `out`. Split out alongside [`pair_col`] so both callers share it.
#[cube]
#[allow(clippy::too_many_arguments)]
fn pair_emit(
    g: &[u32],
    xi: &[u32],
    out: &mut [Atomic<u32>],
    working: u64,
    row_base: usize,
    out_offset: usize,
    width: usize,
    num_limbs: usize,
    #[comptime] sq_len: usize,
    pp_shift: &[u32],
    pp_mask: &[u32],
) {
    // `seqno` indexes the algebra basis of the output degree; `out_offset` shifts it
    // to this product's target-generator block within the row (0 for a single-block
    // output). Both are bit offsets, added before splitting into (limb, bit).
    // Only the first `PPART_MAX_LEN` positions can be non-zero (see `multiply_pair`'s
    // accumulator), so the rank loop stops at `sq_len = min(work_cap, PPART_MAX_LEN)`, computed on
    // the host (comptime arithmetic does not lower inside a `#[cube]` fn).
    let idx = seqno_core_packed(g, xi, pp_shift, pp_mask, working, sq_len, width);
    let global_bit = out_offset + usize::cast_from(idx);
    let limb = global_bit / 32;
    // Device-side mirror of the host's defensive mask: `nassau_gpu::get_partial_matrix_restricted`
    // launches at the full output width but masks bits `>= target_dim` on readback because a kept
    // block's `out_offset + seqno` can span past it. Skip writes past this row's `num_limbs` — they
    // would otherwise overrun into the next row (silent corruption) or past the buffer (an OOB
    // atomic; compute-sanitizer confirmed `Invalid __global__ atomic ... out of bounds`).
    // Two independent bounds, both required: `limb < num_limbs` keeps the write inside this row
    // (out_offset + seqno can span past it), and `word < out.len()` guards the row itself — a
    // `row_base` that overruns the buffer (compute-sanitizer caught this as a second OOB atomic at
    // higher degree, distinct from the intra-row overflow) would otherwise write past the end.
    if limb < num_limbs {
        let word = row_base + limb;
        if word < out.len() {
            let bit = u32::cast_from(global_bit % 32);
            out[word].fetch_xor(1u32 << bit);
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn multiply_pair(
    col_sums: &[u16],
    masks: &[u16],
    term_pparts: &[u16],
    g: &[u32],
    xi: &[u32],
    out: &mut [Atomic<u32>],
    cs_base: usize,
    mk_base: usize,
    b_base: usize,
    term_len: usize,
    cs_len: usize,
    mk_len: usize,
    row_base: usize,
    out_offset: usize,
    width: usize,
    num_limbs: usize,
    #[comptime] sq_len: usize,
    pp_shift: &[u32],
    pp_mask: &[u32],
) {
    let mut low = cs_len;
    if term_len < cs_len {
        low = term_len;
    }

    // Packed accumulator instead of `Array::<u32>::new(work_cap)`: the single largest slice of
    // per-thread state (work_cap x u32 ~= 16 registers of the measured 78, and registers are what
    // caps occupancy at 3 blocks/SM = 37.5%).
    //
    // Entries at index >= PPART_MAX_LEN cannot exist, so stopping there is exact rather than a
    // truncation, and it holds by the degree bound rather than by observation: at p = 2 the entry
    // r_n multiplies deg(xi_n) = 2^n - 1, so a p-part of length 11 needs degree >= 2^11 - 1 = 2047,
    // while `PPart::MAX_DEGREE` is 2045. `MAX_LEN = 10` is therefore forced by that bound, not a
    // cap something could exceed — which is also why `PPart::set` can assert `i < MAX_LEN`.
    let mut working = 0u64;
    let mut rejected = 0u32;

    for j in 0..pair_cols(term_len, cs_len, mk_len) {
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

        let val = pair_col(j, low, b, cs, mk);
        rejected |= val & PAIR_COL_REJECT;
        if j < PPART_MAX_LEN {
            working |= u64::cast_from(val & 0xffffu32) << u64::cast_from(pp_shift[j]);
        }
    }

    if rejected == 0u32 {
        pair_emit(
            g, xi, out, working, row_base, out_offset, width, num_limbs, sq_len, pp_shift, pp_mask,
        );
    }
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
// 64-bit addressing (`address_type = "u64"`): the admissible master and width-padded basis exceed
// `u32::MAX` elements at high stems, and the per-`R` offsets in `r_cs_offset`/`r_mk_offset` (u64)
// index the global (across-segment) master offsets. Static u64 (not "dynamic") because dynamic would
// pick 32-bit `usize` for small blocks and then *narrow* the u64 offset arrays on read (cubecl
// `usize::cast_from(u64)` under a u32 address type), corrupting results — the (180,92) `dx != 0`.
// `launch_unchecked` because cubecl's checked-mode bounds clamp emits `min(u64, u64)`, which NVRTC
// rejects as an ambiguous overload; every access here is in-bounds by construction (the `need_*`
// prefix covers every offset, `seg_read_*` selects the owning segment, and the per-column `j` guards).
//
// The master (`cs*`/`mk*`) and basis (`pp*`/`ln*`) are each a segmented, no-copy-growth store bound as
// [`MASTER_MAX_SEG`] separate segment `Array`s (cubecl has no array-of-buffers). A thread reads its
// one matrix's `col_sums`/`masks` and its term's p-part out of the segments via `seg_read_*` (correct
// for any offset, straddle or not — no layout padding needed), one column at a time, straight into
// the arithmetic. `seg_elems` is the segment element count (`o / seg_elems` picks the segment).
//
// OCCUPANCY: this kernel was register-bound, and the register count is essentially a function of how
// much per-thread state it holds. Two successive removals took it from 78 registers (3 blocks/SM,
// 37.5% — matching a sampled 36.8% "Compute Warps in Flight") to 48 (packing `working` into a `u64`)
// to 40 (fusing the `cs_local`/`mk_local`/`term_local` gather into the column loop), i.e. 6
// blocks/SM = 75%, with zero spill at every tier. Nothing of `work_cap` length lives in a thread any
// more, so occupancy no longer scales with the internal degree either. Measured +5.6% on the
// stem-200 bench (6.43 -> 6.79 e9 pairs/s, four paired rounds, arms non-overlapping).
//
// That retired a planned upstream cubecl change: cubecl emits only `__launch_bounds__(<threads>)`,
// and forcing the second argument (`~/cubecl-min-blocks-per-sm.patch`, mirroring `cluster_dim`) was
// worth +6.2% back when ptxas settled at 78 registers. It is moot now — ptxas picks 6 blocks/SM on
// its own, past the 4 the patch would have asked for, so the floor it sets would never bind.
//
// Occupancy is no longer the constraint, and pushing it further is not obviously the next move: an
// intermediate variant reached 35 registers (7 blocks/SM) by branching on the rejection test instead
// of accumulating it, and measured ~11% SLOWER — twelve divergent branches per thread cost more than
// the extra resident warp bought. Unrolling the column loop (`#[unroll]`) was throughput-neutral and
// quadrupled the code, so it is deliberately not used.
#[cube(launch_unchecked, address_type = "u64")]
#[allow(clippy::too_many_arguments)]
fn multiply_batch_kernel(
    cs0: &[u16],
    cs1: &[u16],
    cs2: &[u16],
    cs3: &[u16],
    cs4: &[u16],
    cs5: &[u16],
    cs6: &[u16],
    cs7: &[u16],
    cs8: &[u16],
    cs9: &[u16],
    cs10: &[u16],
    cs11: &[u16],
    cs12: &[u16],
    cs13: &[u16],
    cs14: &[u16],
    cs15: &[u16],
    mk0: &[u16],
    mk1: &[u16],
    mk2: &[u16],
    mk3: &[u16],
    mk4: &[u16],
    mk5: &[u16],
    mk6: &[u16],
    mk7: &[u16],
    mk8: &[u16],
    mk9: &[u16],
    mk10: &[u16],
    mk11: &[u16],
    mk12: &[u16],
    mk13: &[u16],
    mk14: &[u16],
    mk15: &[u16],
    pp0: &[u16],
    pp1: &[u16],
    pp2: &[u16],
    pp3: &[u16],
    pp4: &[u16],
    pp5: &[u16],
    pp6: &[u16],
    pp7: &[u16],
    pp8: &[u16],
    pp9: &[u16],
    pp10: &[u16],
    pp11: &[u16],
    pp12: &[u16],
    pp13: &[u16],
    pp14: &[u16],
    pp15: &[u16],
    ln0: &[u32],
    ln1: &[u32],
    ln2: &[u32],
    ln3: &[u32],
    ln4: &[u32],
    ln5: &[u32],
    ln6: &[u32],
    ln7: &[u32],
    ln8: &[u32],
    ln9: &[u32],
    ln10: &[u32],
    ln11: &[u32],
    ln12: &[u32],
    ln13: &[u32],
    ln14: &[u32],
    ln15: &[u32],
    term_gei: &[u32],
    g: &[u32],
    xi: &[u32],
    out: &mut [Atomic<u32>],
    r_cs_offset: &[u64],
    r_mk_offset: &[u64],
    r_cs_len: &[u32],
    r_mk_len: &[u32],
    r_num_mats: &[u32],
    prod_r_index: &[u32],
    prod_term_start: &[u32],
    prod_num_terms: &[u32],
    prod_row_base: &[u32],
    prod_out_offset: &[u32],
    prod_pair_start: &[u32],
    prod_coarse: &[u32],
    width: usize,
    seg_elems: usize,
    num_limbs: usize,
    // Runtime scalar, deliberately NOT `#[comptime]`: it differs block to block, so specialising on
    // it forces an NVRTC recompile per distinct value (measured: bench spread widened from 0.7% to
    // 5.6%). With the coarse index below it is `ceil(log2(chunk span))`, typically a couple of
    // steps rather than the ~15 a full search over every product needed.
    search_iters: usize,
    // Comptime is right here: at most `MASTER_MAX_SEG` distinct values, so the select chain folds
    // to the segments that exist without a recompile storm.
    #[comptime] num_segs: usize,
    pp_shift: &[u32],
    pp_mask: &[u32],
    // `min(work_cap, PPART_MAX_LEN)`: how far the packed rank loop runs.
    #[comptime] sq_len: usize,
    // Per-thread column count, specialised per launch to what THIS block actually needs
    // (`max(mk_len, term_len)`, rounded up), not the global worst case.
    //
    // This used to be the kernel's dominant cost: every thread held four arrays of this length
    // (`working` u32 plus `cs_local`/`mk_local`/`term_local` u16), so the constant set register
    // pressure and hence occupancy — measured 36% compute warps in flight with 64% of warp slots
    // unallocated at the old fixed 32. Shrinking it to what the data needs measured +28%
    // (5.97 -> 7.64 e9 pairs/s). All four arrays are gone now (`working` packs into a `u64`, the
    // other three were fused away into the loop below), so what is left is the trip count of a
    // loop over scalars; keeping it tight still shortens that loop, but it no longer gates
    // occupancy. It stays comptime so the trip count is a literal rather than a loaded scalar.
    //
    // It must NOT be hardcoded. `mk_len = rows + cols - 1` grows with internal degree: 16 suffices
    // to t~510, but a 9th xi appears at t>=511 and it becomes 17, then 18 past 1023. A fixed 16
    // would silently truncate at stem 300 — wrong answers, no error. Deriving it per launch keeps
    // the occupancy win at every degree, and the host asserts it fits [`WORKING_CAP`].
) {
    let k = ABSOLUTE_POS;
    let num_products = prod_pair_start.len() - 1;
    if k >= usize::cast_from(prod_pair_start[num_products]) {
        terminate!();
    }

    // Largest product `p` with `prod_pair_start[p] <= k` (every product owns ≥ 1 pair,
    // so `prod_pair_start` is strictly increasing and `p` is unique).
    //
    // `prod_coarse` brackets the answer before the search starts: entry `ci` is the product owning
    // pair `ci << COARSE_LOG`, so `p` is in `prod_coarse[ci] ..= prod_coarse[ci + 1]`. Products are
    // ordered and every product owns >= 1 pair, so the bracket is valid; the sentinel entry keeps
    // `ci + 1` readable for the final chunk.
    //
    // Each search step is a DEPENDENT global load of `prod_pair_start[mid]` — a full latency stall
    // before a thread can touch its own data — and ablation put the unbracketed search at ~12% of
    // kernel time. Two cheap loads replace ~15 dependent ones.
    let ci = k >> COARSE_LOG;
    let mut lo = usize::cast_from(prod_coarse[ci]);
    let mut hi = usize::cast_from(prod_coarse[ci + 1]) + 1;
    if hi > num_products {
        hi = num_products;
    }
    for _ in 0..search_iters {
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
    let p_start = usize::cast_from(prod_pair_start[p]);
    let local = k - p_start;
    // MATRIX varies fastest, term slowest. The obvious decode (`m = local / nt`, `t = local % nt`)
    // has the opposite order, and it is the kernel's dominant cost: consecutive threads then share a
    // matrix but each takes a DIFFERENT term, whose p-part sits at `gei * width` for an arbitrary
    // basis index -- a 32-way scatter across a multi-GB resident basis, 2 useful bytes per 32-byte
    // sector fetched. Ablation measured the whole kernel at +79% with the scatter removed (every
    // thread reading offset 0) and +0.3% with the entire seqno rank loop deleted, so locality is
    // essentially all of the remaining time.
    //
    // With `m` fastest, a warp shares one term -- its p-part read becomes a broadcast -- and the
    // `col_sums`/`masks` reads become `m * cs_len` apart, i.e. one contiguous fully-used run per
    // warp instead of one wasted sector per lane.
    //
    // This is a permutation of the same `(m, t)` set: every thread still owns exactly one pair, and
    // the output is XOR-accumulated, so the result is unchanged.
    // `num_mats` by lookup, not `(pair_span) / nt`: one load instead of an emulated integer
    // division (I2F/MUFU.RCP/F2I plus fixups). Only ONE division remains in the decode -- `t` and
    // `m` share it, since ptxas emits a single divide plus an IMAD for the remainder.
    let num_mats = usize::cast_from(r_num_mats[ri]);
    let nt = usize::cast_from(prod_num_terms[p]);

    // A thread covers a TILE of `MATRIX_GROUP` matrices x `TERM_GROUP` terms.
    //
    // `col_sums`/`masks` depend only on the matrix and a term's p-part only on the term, so a
    // MxT tile reads `2M + T` values per column to evaluate `M*T` pairs -- 1.17 loads per pair at
    // 2x3, against 1.67 at 1x3 and 3 at 1x1. The kernel is issue-limited on integer work (ncu:
    // 75% SM vs 1.9% DRAM, ALU top pipeline), so fewer loads and fewer addresses is the lever.
    //
    // The two axes are NOT symmetric. Terms are few (`nt ~ 5`), so the ragged tail dominates the
    // choice of `TERM_GROUP` -- see its doc comment, where 4 loses to 3 purely on wasted lanes.
    // Matrices are many (`num_mats ~ 20 000`), so a partial matrix tile costs a few idle lanes out
    // of thousands and `MATRIX_GROUP` is free to follow the load arithmetic instead.
    let mg_count = num_mats.div_ceil(MATRIX_GROUP);
    let mg = local % mg_count;
    let tg = local / mg_count;
    let m_base = mg * MATRIX_GROUP;
    let t_base = tg * TERM_GROUP;

    let cs_len = usize::cast_from(r_cs_len[ri]);
    let mk_len = usize::cast_from(r_mk_len[ri]);
    let cs_base = usize::cast_from(r_cs_offset[ri]);
    let mk_base = usize::cast_from(r_mk_offset[ri]);
    let ts_base = usize::cast_from(prod_term_start[p]) + t_base;

    // Per-term p-part offsets and lengths. Lanes past `nt` carry `term_len = 0` and are excluded
    // from the output below: an all-zero term against an all-zero column does NOT reject, so they
    // would otherwise emit a spurious `seqno(0)` bit.
    let mut pp_off = Array::<u64>::new(TERM_GROUP);
    let mut term_len = Array::<u32>::new(TERM_GROUP);
    let mut cols = cs_len;
    if mk_len > cols {
        cols = mk_len;
    }
    #[unroll]
    for tt in 0..TERM_GROUP {
        let mut po = 0u64;
        let mut tl = 0u32;
        if t_base + tt < nt {
            let gei = usize::cast_from(term_gei[ts_base + tt]);
            po = u64::cast_from(gei * width);
            tl = seg_read_u32(
                ln0, ln1, ln2, ln3, ln4, ln5, ln6, ln7, ln8, ln9, ln10, ln11, ln12, ln13, ln14,
                ln15, gei, seg_elems, num_segs,
            );
        }
        pp_off[tt] = po;
        term_len[tt] = tl;
        if usize::cast_from(tl) > cols {
            cols = usize::cast_from(tl);
        }
    }

    let mut working = Array::<u64>::new(MATRIX_GROUP * TERM_GROUP);
    let mut rejected = Array::<u32>::new(MATRIX_GROUP * TERM_GROUP);
    #[unroll]
    for i in 0..MATRIX_GROUP * TERM_GROUP {
        working[i] = 0u64;
        rejected[i] = 0u32;
    }

    for j in 0..cols {
        // One `col_sums`/`masks` pair per matrix in the tile, shared by every term.
        let mut cs = Array::<u32>::new(MATRIX_GROUP);
        let mut mk = Array::<u32>::new(MATRIX_GROUP);
        #[unroll]
        for mm in 0..MATRIX_GROUP {
            let mut c = 0u32;
            let mut k = 0u32;
            if m_base + mm < num_mats {
                if j < cs_len {
                    c = u32::cast_from(seg_read_u16(
                        cs0,
                        cs1,
                        cs2,
                        cs3,
                        cs4,
                        cs5,
                        cs6,
                        cs7,
                        cs8,
                        cs9,
                        cs10,
                        cs11,
                        cs12,
                        cs13,
                        cs14,
                        cs15,
                        cs_base + (m_base + mm) * cs_len + j,
                        seg_elems,
                        num_segs,
                    ));
                }
                if j < mk_len {
                    k = u32::cast_from(seg_read_u16(
                        mk0,
                        mk1,
                        mk2,
                        mk3,
                        mk4,
                        mk5,
                        mk6,
                        mk7,
                        mk8,
                        mk9,
                        mk10,
                        mk11,
                        mk12,
                        mk13,
                        mk14,
                        mk15,
                        mk_base + (m_base + mm) * mk_len + j,
                        seg_elems,
                        num_segs,
                    ));
                }
            }
            cs[mm] = c;
            mk[mm] = k;
        }

        // One p-part read per term in the tile, shared by every matrix.
        #[unroll]
        for tt in 0..TERM_GROUP {
            let tl = usize::cast_from(term_len[tt]);
            let mut b = 0u32;
            if j < tl {
                b = u32::cast_from(seg_read_u16(
                    pp0,
                    pp1,
                    pp2,
                    pp3,
                    pp4,
                    pp5,
                    pp6,
                    pp7,
                    pp8,
                    pp9,
                    pp10,
                    pp11,
                    pp12,
                    pp13,
                    pp14,
                    pp15,
                    usize::cast_from(pp_off[tt]) + j,
                    seg_elems,
                    num_segs,
                ));
            }
            let mut low = cs_len;
            if tl < cs_len {
                low = tl;
            }
            #[unroll]
            for mm in 0..MATRIX_GROUP {
                let val = pair_col(j, low, b, cs[mm], mk[mm]);
                let i = mm * TERM_GROUP + tt;
                rejected[i] |= val & PAIR_COL_REJECT;
                if j < PPART_MAX_LEN {
                    working[i] |= u64::cast_from(val & 0xffffu32) << u64::cast_from(pp_shift[j]);
                }
            }
        }
    }

    #[unroll]
    for mm in 0..MATRIX_GROUP {
        #[unroll]
        for tt in 0..TERM_GROUP {
            let i = mm * TERM_GROUP + tt;
            if m_base + mm < num_mats {
                if t_base + tt < nt {
                    if rejected[i] == 0u32 {
                        pair_emit(
                            g,
                            xi,
                            out,
                            working[i],
                            usize::cast_from(prod_row_base[p]),
                            usize::cast_from(prod_out_offset[p]),
                            width,
                            num_limbs,
                            sq_len,
                            pp_shift,
                            pp_mask,
                        );
                    }
                }
            }
        }
    }
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
/// One batch multiply's result, held as the D2H landing buffers themselves.
///
/// The device write has to land somewhere; everything after that is waste. The original form
/// allocated a fresh `Vec` per row and copied the whole output into freshly-mapped pages right
/// after the device had written it — ~32 M allocations over a stem-200 resolution. The measured
/// best-case launch cost scaled linearly with output bytes at only 2.1-5.0 GB/s (256 MiB in
/// 130 ms), far under PCIe 5.0 x16, with the GPU idle throughout.
///
/// So this keeps cubecl's [`Bytes`] (which may already be pinned — see `AllocationProperty`) and
/// hands out row slices as views. One block per bounded launch, in row order; the owned
/// constructor covers the eviction merge and the CPU oracle, which must accumulate.
pub struct BatchOutput {
    /// One landing buffer per row-block, in row order.
    blocks: Vec<Bytes>,
    num_limbs: usize,
}

impl BatchOutput {
    /// Wrap the per-block landing buffers (zero copy).
    fn from_blocks(blocks: Vec<Bytes>, num_limbs: usize) -> Self {
        Self { blocks, num_limbs }
    }

    /// Wrap owned row-major limbs (eviction merge, CPU oracle).
    pub fn from_limbs(limbs: Vec<u32>, num_limbs: usize) -> Self {
        Self {
            blocks: vec![Bytes::from_elems(limbs)],
            num_limbs,
        }
    }

    /// Build from per-row limb vectors (test/reference helper).
    pub fn from_rows(rows: &[Vec<u32>], num_limbs: usize) -> Self {
        Self::from_limbs(rows.concat(), num_limbs)
    }

    /// Limbs per row.
    pub fn num_limbs(&self) -> usize {
        self.num_limbs
    }

    /// Number of rows across all blocks.
    pub fn rows(&self) -> usize {
        if self.num_limbs == 0 {
            return 0;
        }
        self.blocks.iter().map(|b| b.len() / 4).sum::<usize>() / self.num_limbs
    }

    /// Row limb-slices in row order, as views into the landing buffers.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[u32]> {
        let n = self.num_limbs;
        self.blocks
            .iter()
            .flat_map(move |b| u32::from_bytes(b).chunks_exact(n))
    }
}

impl PartialEq for BatchOutput {
    fn eq(&self, other: &Self) -> bool {
        self.num_limbs == other.num_limbs && self.iter_rows().eq(other.iter_rows())
    }
}

impl Eq for BatchOutput {}

impl std::fmt::Debug for BatchOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchOutput")
            .field("rows", &self.rows())
            .field("num_limbs", &self.num_limbs)
            .finish()
    }
}

pub fn multiply_batch_on_gpu(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
) -> BatchOutput {
    // The CPU fallback that used to live here (catch the launch failure, latch [`GPU_DISABLED`],
    // finish the run on the CPU) was removed deliberately: it turned a hard GPU fault into a silent
    // ~100x slowdown, so a crashing run still reported "completed" and every A/B measurement had to
    // be reconstructed by grepping stderr. A context death is now loud — the panic propagates and
    // the run dies at the fault. [`GPU_DISABLED`] is still latched first so in-process observers
    // (the soak test) can tell a context death from an ordinary assertion failure.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        multiply_batch_gpu_inner(algebra, num_cols, num_rows, products)
    })) {
        Ok(out) => out,
        Err(payload) => {
            // compare_exchange so exactly one thread (of the ~100 that may fail together on the
            // shared poisoned context) prints the notice; the rest just resume unwinding.
            if GPU_DISABLED
                .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                eprintln!(
                    "[nassau-gpu] GPU milnor multiply failed (CUDA context poisoned); failing the \
                     run instead of falling back to the CPU. RREF (separate fp-cuda runtime) is \
                     unaffected by this flag."
                );
            }
            std::panic::resume_unwind(payload)
        }
    }
}

/// CPU reference for one [`multiply_batch_on_gpu`] batch: the exact same `num_rows × ⌈num_cols/32⌉`
/// bit-packed F₂ matrix the GPU produces, computed with [`MilnorAlgebra::multiply_basis_element_by_element_2`].
/// Used as the stopgap fallback when the GPU context dies mid-run, and as a correctness oracle for the
/// GPU stress bench. Each product's `Sq(R)·s` lands in row `prod.row` at column offset `prod.out_offset`.
pub fn cpu_multiply_batch(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
) -> BatchOutput {
    use fp::vector::FpVector;
    let p = algebra.prime();
    let num_limbs = num_cols.div_ceil(32).max(1);
    let mut rows = vec![vec![0u32; num_limbs]; num_rows];
    // (built per row, then flattened to the shared BatchOutput layout below)
    for prod in products {
        let out_degree = prod.r_degree + prod.s_degree;
        let block_dim = algebra.dimension(out_degree);
        if block_dim == 0 {
            continue;
        }
        let s_dim = algebra.dimension(prod.s_degree);
        let mut s = FpVector::new(p, s_dim);
        for &ti in &prod.term_indices {
            s.set_entry(ti, 1);
        }
        let mut tmp = FpVector::new(p, block_dim);
        algebra.multiply_basis_element_by_element_2(
            tmp.as_slice_mut(),
            1,
            prod.r_degree,
            prod.r_idx,
            prod.s_degree,
            s.as_slice(),
        );
        for (i, _) in tmp.iter_nonzero() {
            let col = prod.out_offset + i;
            rows[prod.row][col / 32] ^= 1u32 << (col % 32);
        }
    }
    BatchOutput::from_limbs(rows.concat(), num_limbs)
}

fn multiply_batch_gpu_inner(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
) -> BatchOutput {
    let cap = resident_degree_cap();
    // Fast path (default, `cap == i32::MAX`, and any run whose `R`s are all under the cap): a single
    // resident-master pass, byte-identical to the pre-eviction code. No cloning, no second launch.
    let num_limbs_all = num_cols.div_ceil(32).max(1);
    if cap == i32::MAX || products.iter().all(|p| p.r_degree <= cap) {
        return BatchOutput::from_blocks(
            multiply_batch_grouped(algebra, num_cols, num_rows, products, MasterMode::Resident),
            num_limbs_all,
        );
    }
    // Eviction active. Each output row's products all share one `R` (see [`MasterMode`]), so the
    // hot (degree ≤ cap) and cold row sets are DISJOINT. Run each group on its own rows only,
    // **compacted** to a dense `0..k` range so each pass reads back just its own rows — total
    // readback stays `num_rows`, not 2× (critical in the intended high-θ regime, where the cold
    // set is a small tail and a full-height cold readback would be almost all zeros). Results
    // scatter back to the original row indices; a row in neither group stays zero (its content, if
    // any, comes from the caller's CPU identity path).
    let num_limbs = num_cols.div_ceil(32).max(1);
    let mut result = vec![0u32; num_rows * num_limbs];
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
        let remap: HashMap<usize, usize> = rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        let compact: Vec<GpuProduct> = products
            .iter()
            .filter(|p| is_group(p.r_degree))
            .map(|p| {
                let mut q = p.clone();
                q.row = remap[&p.row];
                q
            })
            .collect();
        let sub_blocks = multiply_batch_grouped(algebra, num_cols, rows.len(), &compact, mode);
        let sub: Vec<u32> = sub_blocks
            .iter()
            .flat_map(|b| u32::from_bytes(b).iter().copied())
            .collect();
        for (i, &orig) in rows.iter().enumerate() {
            let (dst, src) = (orig * num_limbs, i * num_limbs);
            for k in 0..num_limbs {
                result[dst + k] ^= sub[src + k];
            }
        }
    }
    BatchOutput::from_limbs(result, num_limbs)
}

fn multiply_batch_grouped(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
    mode: MasterMode,
) -> Vec<Bytes> {
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
    // `Transient` it warms the host-side [`COLD_COUNT`] shape cache the same way (no per-block recount).
    // This pre-pass is where multi-minute stalls hide: it is strictly sequential, it calls
    // `admissible_matrices` for every first-sight `R`, and it serialises on the `RESIDENT_HOST`
    // write lock while appending multi-GB pending buffers — all of it previously outside every
    // span and every timer, so a worker parked here logged nothing at all. `new_r` distinguishes
    // "paid to warm the master" from "waited for someone else's warm-up".
    let prepass = tracing::debug_span!(
        "pair_prepass",
        products = products.len(),
        new_r = tracing::field::Empty,
    );
    let misses_before = RESIDENT_MISSES.load(std::sync::atomic::Ordering::Relaxed);
    let prod_pairs: Vec<usize> = prepass.in_scope(|| {
        products
            .iter()
            .map(|prod| {
                let r = algebra.basis_element_from_index(prod.r_degree, prod.r_idx);
                let num_mats = match mode {
                    MasterMode::Resident => resident_info(algebra, r.p_part).num_mats as usize,
                    MasterMode::Transient => cold_count(algebra, r.p_part).2 as usize,
                };
                // Threads, not pairs: one per (MATRIX_GROUP x TERM_GROUP tile).
                num_mats.div_ceil(MATRIX_GROUP) * prod.term_indices.len().div_ceil(TERM_GROUP)
            })
            .collect()
    });
    prepass.record(
        "new_r",
        RESIDENT_MISSES.load(std::sync::atomic::Ordering::Relaxed) - misses_before,
    );
    drop(prepass);
    let mut result: Vec<Bytes> = Vec::new();
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
        // Fan the row-block out across devices. The master is sharded, so a product can only run
        // where its `R` lives; each device evaluates its own subset over the SAME rows and the
        // partial outputs are XORed. That is exact, not an approximation: every product contributes
        // by `fetch_xor` into the output limbs, so the contributions commute and split freely.
        let block = &products[p0..p1];
        let mut by_dev: Vec<Vec<GpuProduct>> = vec![Vec::new(); gpu_count()];
        for (pi, prod) in block.iter().enumerate() {
            let d = match mode {
                // Transient blocks enumerate their own master into per-launch scratch, so they are
                // device-agnostic; spread them round-robin instead of piling onto device 0.
                MasterMode::Transient => pi % gpu_count(),
                MasterMode::Resident => {
                    let r = algebra.basis_element_from_index(prod.r_degree, prod.r_idx);
                    resident_info(algebra, r.p_part).dev as usize
                }
            };
            by_dev[d].push(prod.clone());
        }
        // Marshal + submit every device's share first, THEN wait. `multiply_batch_block` returns
        // the wait rather than performing it, so device `d + 1`'s marshalling overlaps device `d`'s
        // execution and all shards are in flight together. The first cut used `std::thread::scope`
        // here, which spawned an OS thread per device per block — hundreds a second, and thread ids
        // into the hundreds of thousands in the logs.
        // One thread, asynchronous submissions to every device. No fan-out threads and no pool:
        // `submit_on` does not block, so this marshals a shard, hands it to its device, and moves to
        // the next while that device is already running. Only the final wait blocks.
        //
        // Sharding SPLITS the products, so marshalling the shards one after another is the same
        // total host work as marshalling one unsharded block — there is nothing to parallelise here.
        // And with ~32 resolution workers each submitting `gpu_count()` jobs, ~128 launches are in
        // flight, which is the queue depth the devices need; a fan-out pool could only cap that
        // (sized at 64 it halved depth 3.8 -> 2.2 and cost 47% end to end).
        // Three phases, and the split points are load-bearing rather than stylistic.
        //
        //  1. MARSHAL every shard. Parallel inside (rayon), and no permit is held by this thread, so
        //     a stolen resolution-step job that parks on `GpuPermit::acquire` cannot wedge the join
        //     it needs to finish — [`GpuBudget`]'s invariant, which the previous fused shape broke
        //     by holding device `d`'s permit while marshalling device `d + 1`.
        //  2. SUBMIT every shard. Sequential and rayon-free: takes each permit and hands the block
        //     to its device's worker without blocking, so all `gpu_count()` shards are executing
        //     together rather than one after another.
        //  3. WAIT on each. `multiply_batch_block`'s readback is issued but not awaited on the
        //     worker, so the devices stay busy with later blocks while this thread sits on fences.
        //
        // In-flight work is therefore bounded by `NASSAU_GPU_MEM_BUDGET_MB` — memory, not threads.
        // Phase 1 runs the shards' marshalling CONCURRENTLY, which is safe here precisely because
        // no permit is taken until phase 2 — the split is what makes this legal, and it is also
        // where the time is. Sharding divides the products, so marshalling shards one after another
        // costs the same TOTAL host work as one unsharded block, and an earlier comment concluded
        // from that there was nothing to parallelise. That is true of total work and false of the
        // critical path: a row block is not done until its slowest shard is, so serial marshalling
        // makes each block's latency the SUM over devices rather than the max. Measured end to end
        // at stem 200: parallel marshal 2551 s, serial marshal 3136 s and 3140 s across two
        // independent runs, with `marshal` itself totalling 3444 s.
        use maybe_rayon::prelude::*;
        let nonempty: Vec<(usize, &Vec<GpuProduct>)> = by_dev
            .iter()
            .enumerate()
            .filter(|(_, ps)| !ps.is_empty())
            .collect();
        let submits: Vec<BlockSubmit<'_>> = nonempty
            .into_maybe_par_iter()
            .map(|(d, ps)| multiply_batch_block(algebra, num_cols, r0, r1 - r0, ps, mode, d))
            .collect();
        let waits: Vec<BlockWait> = submits.into_iter().map(|s| s()).collect();
        let partials: Vec<Bytes> = waits.into_iter().map(|w| w()).collect();
        let mut it = partials.into_iter();
        let mut acc = it
            .next()
            .expect("a non-empty row block has at least one device's products");
        for part in it {
            // Byte-wise XOR is identical to limb-wise here and needs no typed view.
            for (x, y) in acc.iter_mut().zip(part.iter()) {
                *x ^= *y;
            }
        }
        result.push(acc);
        (r0, p0) = (r1, p1);
    }
    result
}

/// The wait for one launched block: blocks on its completion fence and yields its output rows.
type BlockWait = Box<dyn FnOnce() -> Bytes + Send>;

/// A block that has been *marshalled* but not yet admitted to the device. Calling it takes the
/// [`GpuPermit`] and submits, returning the [`BlockWait`].
///
/// The two phases are separate so that no permit is ever held across a rayon parallel section —
/// [`GpuBudget`]'s safety invariant. Marshalling is parallel; permit acquisition and submission are
/// not. Fusing them (as this did until now) meant a caller fanning out over `gpu_count()` devices
/// held device `d`'s permit while marshalling device `d + 1`, so a par_iter chunk could steal a
/// resolution-step job that parked on `acquire` while this thread waited on a join those very
/// workers had to finish — the H200 stall the invariant exists to prevent.
type BlockSubmit<'a> = Box<dyn FnOnce() -> BlockWait + Send + 'a>;

/// One bounded launch of [`multiply_batch_on_gpu`]: rows `row_base..row_base + num_rows` of the
/// full build, with `products` the (contiguous, row-major) slice landing in those rows. Marshals
/// on the calling thread (in parallel), then hands back the submit step; the [`GpuPermit`] taken
/// there is held until the readback completes, so the total output size of in-flight launches
/// stays under `NASSAU_GPU_MEM_BUDGET_MB` across all worker threads.
fn multiply_batch_block<'a>(
    algebra: &'a MilnorAlgebra,
    num_cols: usize,
    row_base: usize,
    num_rows: usize,
    products: &'a [GpuProduct],
    mode: MasterMode,
    dev: usize,
) -> BlockSubmit<'a> {
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
    // Breakdown of the `prep` figure, which was only ever a single number and so could not say
    // whether marshalling is worth restructuring or merely worth accepting. `intern` is the
    // per-launch `HashMap<(i32, usize), u32>` above — pure representation tax, since every `R`
    // already has a resident device identity ([`RInfo`]) the caller could have carried instead of
    // an algebra-local `(degree, index)` pair. If it dominates, deleting it needs no new data
    // structure, only a different id in `GpuProduct`.
    let intern_ms = t_marshal.elapsed().as_secs_f64() * 1e3;
    let t_basis = std::time::Instant::now();

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
    // `ensure_basis` takes the basis WRITE lock on a first-sight degree; spanned separately from
    // the fill below so a wait on that lock is not misread as marshalling work.
    let global_base = tracing::debug_span!("ensure_basis", max_s_degree)
        .in_scope(|| ensure_basis(algebra, width, max_s_degree));
    // Everything from the intern to here: `term_off` prefix sums, the `max_s_degree` scan, and
    // `ensure_basis` (which may upload). Separated from the fill because the two have different
    // fixes — this shrinks by keeping the basis warm, the fill by not rebuilding indices.
    let basis_ms = t_basis.elapsed().as_secs_f64() * 1e3;
    let t_tgei = std::time::Instant::now();
    let mut term_gei: Vec<u32> = vec![0u32; total_terms];
    // The ONLY rayon construct inside the guarded region, hence the only place a worker can block
    // at a join and enter the steal loop. The multi-minute stalls sit somewhere in this guarded
    // region and every other part of it is now spanned and bounded (`extract_restricted` ≤4.8 s,
    // `pair_prepass` ≤5.1 s, `gpu_submit` 40 ms), so this is what remains. `prep` was only ever
    // reported as a sum, which cannot separate a few 100 s outliers from many small costs.
    {
        // Scoped to the fill alone: entering at function level would leave the span open across
        // the permit wait and the GPU submission and attribute their time here.
        let _marshal_span = tracing::debug_span!(
            "marshal_terms",
            products = products.len(),
            terms = total_terms
        )
        .entered();
        // SEQUENTIAL, deliberately. The body is one add and one store per term, so at the largest
        // observed size (478 972 products / 2 300 621 terms) the whole fill is a few milliseconds
        // of memory-bandwidth-bound work — rayon cannot speed that up past its own split/join
        // overhead.
        //
        // What parallelising it DID buy was a join, and therefore rayon's steal loop, and therefore
        // exposure to starvation: instrumenting this span measured a single fill at **146 s** (p99
        // 0.09 s — 99 % are fast, only the tail explodes), roughly 30 000x the work involved. That
        // was the multi-hundred-second "signature step" stall, which had been misattributed in turn
        // to GPU submission ordering, to the resident-master pre-pass, and to the kernel itself.
        //
        // With this sequential there is no join anywhere inside a bidegree's guarded region, so a
        // worker cannot be parked here at all.
        let tg_all = &mut term_gei[..];
        for (pi, prod) in products.iter().enumerate() {
            let (off, nt) = (term_off[pi], prod.term_indices.len());
            let base = global_base[prod.s_degree as usize];
            for (slot, &ti) in tg_all[off..off + nt].iter_mut().zip(&prod.term_indices) {
                *slot = base + ti as u32;
            }
        }
    }
    // The `term_gei` fill: one add and one store per term. This is the part that becomes a SLICE
    // rather than a build if products are held struct-of-arrays with global basis indices, so its
    // share is the ceiling on what that refactor can return.
    let tgei_ms = t_tgei.elapsed().as_secs_f64() * 1e3;
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
                for (slot, v) in tpp[k * width..(k + 1) * width]
                    .iter_mut()
                    .zip(elt.p_part.iter())
                {
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
    // Held for the device section (RAII): bounds total in-flight output bytes across workers. It is
    // MOVED INTO THE WAIT CLOSURE below and dropped only once the readback has completed, because
    // that is when the output buffer (device page + pinned host landing) is actually free. A plain
    // local here drops at this function's return — i.e. straight after `submit_on`, which does not
    // block — so between `1639982877` (when this function started returning its wait instead of
    // performing it) and now, the budget admitted every launch immediately and bounded nothing.
    // With the readback no longer serialising the worker, this permit is the ONLY thing bounding
    // in-flight memory, so its scope is load-bearing rather than belt-and-braces.
    // Split the "marshal" figure at the point where this thread stops doing CPU work and starts
    // waiting. `t_marshal` spans both, so the 80/20 marshal-vs-device headline it produced cannot
    // distinguish host marshalling from time parked on our own permit / arbitration lock — and the
    // two call for opposite fixes.
    let prep_ms = t_marshal.elapsed().as_secs_f64() * 1e3;

    // Everything below is the submit phase: strictly sequential (no rayon), so the permit it takes
    // satisfies [`GpuBudget`]'s invariant. The caller runs it only once every shard has marshalled.
    Box::new(move || {
        let t_wait = std::time::Instant::now();
        let permit = GpuPermit::acquire(num_rows * num_limbs * 4);
        let permit_ms = t_wait.elapsed().as_secs_f64() * 1e3;
        let t_lock = std::time::Instant::now();
        // Shared side of the cross-runtime GPU arbitration, taken here for the same reason as the
        // permit above and never earlier: multiplies overlap each other freely but yield while an
        // `fp-cuda` row reduction holds the device, so the reduction's thousands of tiny sequential
        // relaunches are not stuck behind these saturating kernels (~10 000× when they are — see
        // [`fp::gpu_lock`]). Taking it at function entry deadlocks exactly as described above: the
        // marshalling `par_iter` runs chunks on other workers, which steal another bidegree's
        // multiply, block acquiring the shared side behind a waiting reduction, and never let this
        // thread's join finish (observed on H200).
        // The arbitration's shared side is now taken by the GPU thread itself, around the device
        // section it owns (see [`gpu_thread`]). Taking it here instead put ~100 workers through a
        // writer-preferring lock to reach a stage only one of them could occupy anyway — measured at
        // 10% of multiply time, pure convoy. With one submitter it is a 1-vs-1 handshake against the
        // `fp-cuda` reduction, which is all the arbitration ever needed to be.
        let lock_ms = t_lock.elapsed().as_secs_f64() * 1e3;
        let wait_ms = t_wait.elapsed().as_secs_f64() * 1e3;
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
        // `Transient`: per-cold-`R` inputs for the on-device enumeration ([`enumerate_admissible_kernel`]).
        // Instead of building this block's `col_sums`/`masks` on the host and uploading them (the H2D
        // cost the eviction bench exposed), we upload only each cold `R`'s p-part + dimensions and
        // generate the arrays into device scratch at the block-local `r_cs_offset`/`r_mk_offset`. Empty
        // under `Resident`.
        let mut enum_pp_rows: Vec<Vec<u32>> = Vec::new();
        let mut enum_rows: Vec<u32> = Vec::new();
        let mut enum_cols: Vec<u32> = Vec::new();
        for &(rd, ridx) in &distinct_r {
            let r = algebra.basis_element_from_index(rd, ridx);
            assert!(!r.p_part.is_empty(), "each R must be non-empty");
            match mode {
                MasterMode::Resident => {
                    let info = resident_info(algebra, r.p_part);
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
                    let (cs_len, mk_len, num_mats) = cold_count(algebra, r.p_part);
                    r_cs_offset.push(need_cs as u64);
                    r_mk_offset.push(need_mk as u64);
                    r_cs_len.push(cs_len);
                    r_mk_len.push(mk_len);
                    r_num_matrices.push(num_mats as usize);
                    // `cols` = max bit-length of any entry, exactly as the enumeration kernel derives it;
                    // `cs_len == cols-1`, `mk_len == rows+cols-1` (asserted equal to `cold_count`'s below).
                    let cols = r
                        .p_part
                        .iter()
                        .map(|x| u32::BITS - x.leading_zeros())
                        .max()
                        .unwrap();
                    debug_assert_eq!(
                        (cs_len, mk_len),
                        (cols - 1, r.p_part.len() as u32 + cols - 1)
                    );
                    enum_rows.push(r.p_part.len() as u32);
                    enum_cols.push(cols);
                    enum_pp_rows.push(r.p_part.iter().collect::<Vec<_>>());
                    need_cs += num_mats as usize * cs_len as usize;
                    need_mk += num_mats as usize * mk_len as usize;
                }
            }
        }

        // (Transient) Flatten the cold p-parts (padded to the widest) for the enumeration kernel. The
        // per-`R` scratch offsets it writes at are `r_cs_offset`/`r_mk_offset` themselves (u64), passed
        // straight through — no u32 narrowing, so a big block's multi-GB scratch is addressed safely.
        let (enum_pp, enum_width) = if mode == MasterMode::Transient {
            let w = enum_rows.iter().copied().max().unwrap_or(1) as usize;
            let mut pp = vec![0u32; enum_pp_rows.len() * w];
            for (i, row) in enum_pp_rows.iter().enumerate() {
                for (slot, &v) in pp[i * w..i * w + row.len()].iter_mut().zip(row) {
                    *slot = v;
                }
            }
            (pp, w)
        } else {
            (Vec::new(), 1usize)
        };

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
        let mut real_pairs: usize = 0;
        for (pi, prod) in products.iter().enumerate() {
            let ri = prod_r_index[pi];
            prod_term_start.push(term_off[pi] as u32);
            pps.push(pair_acc as u32);
            // One thread per (matrix, TERM_GROUP-sized term group), not per (matrix, term). `pair_acc`
            // sizes the grid, so it counts THREADS; `real_pairs` stays the count of `(matrix, term)`
            // products actually evaluated, which is what the throughput stat must report.
            prod_num_terms.push(prod.term_indices.len() as u32);
            pair_acc += r_num_matrices[ri as usize].div_ceil(MATRIX_GROUP)
                * prod.term_indices.len().div_ceil(TERM_GROUP);
            real_pairs += r_num_matrices[ri as usize] * prod.term_indices.len();
            prod_row_base.push(((prod.row - row_base) * num_limbs) as u32);
            prod_out_offset.push(prod.out_offset as u32);
        }

        let total_pairs = pair_acc;
        assert!(
            u32::try_from(total_pairs).is_ok(),
            "block pair count {total_pairs} exceeds the kernel's u32 thread limit"
        );
        pps.push(total_pairs as u32);

        // Coarse index over the pair space: `coarse[i]` is the product owning pair `i << COARSE_LOG`,
        // so the product for a thread at pair `k` lies in `coarse[ci] ..= coarse[ci + 1]` for
        // `ci = k >> COARSE_LOG`. Ablation put the unaided binary search at ~12% of kernel time, and it
        // is the worst kind of work: `ceil(log2(num_products))` *dependent* global loads, each a full
        // latency stall, before a thread can touch any of its own data.
        let mut coarse: Vec<u32> = Vec::with_capacity((total_pairs >> COARSE_LOG) + 2);
        {
            let mut pi = 0usize;
            let mut k = 0usize;
            while k <= total_pairs {
                while pi + 1 < products.len() && (pps[pi + 1] as usize) <= k {
                    pi += 1;
                }
                coarse.push(pi as u32);
                k += 1 << COARSE_LOG;
            }
            // Sentinel: `ci + 1` must be readable for threads in the final chunk.
            coarse.push(products.len().saturating_sub(1) as u32);
        }
        // Widest product span any chunk covers, so the in-kernel scan has a static iteration bound.
        let coarse_span = coarse
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);

        let out_len = num_rows * num_limbs;
        // Output offsets (`prod_out_offset`/`prod_row_base`) are `u32` values indexing `out_h`; the
        // row-block splitter caps `out_len` well under `u32::MAX` (its output-byte budget is far below
        // 16 GiB), so these never truncate. Assert it loudly rather than silently corrupt if a future
        // budget is set absurdly high. (The `out_h` *length* itself is bound with dynamic addressing.)
        assert!(
            u32::try_from(out_len).is_ok(),
            "block output length {out_len} exceeds u32; lower NASSAU_GPU_BLOCK_MB / row-block \
             budget"
        );
        if std::env::var_os("NASSAU_GPU_DEBUG").is_some() {
            let kb = |n: usize, sz: usize| n * sz / 1024;
            eprintln!(
                "[gpu-batch] rows={num_rows} cols={num_cols} products={} \
                 total_pairs={total_pairs} out_len={out_len} | UPLOAD-KB: g={} xi={} term_gei={} \
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
            // Nothing to launch: hand back a wait that yields the zero block, so the caller's
            // submit-then-wait shape is uniform.
            let empty = Bytes::from_elems(vec![0u32; num_rows * num_limbs]);
            return Box::new(move || empty) as BlockWait;
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

        let term_gei_len = term_gei.len();
        let pps_len = pps.len();
        let marshal_ms = t_marshal.elapsed().as_secs_f64() * 1e3;

        let t_device = std::time::Instant::now();

        // `products` is borrowed; the device section only needs its length, and everything else it
        // touches is owned, so hoisting this makes the closure `'static` and thus sendable.
        let num_products = products.len();

        // Hand the whole device section to the single GPU thread (see [`gpu_thread`]) and block for the
        // result. FIFO service order bounds this wait by the work already queued, replacing the
        // unbounded starvation that the shared-stream free-for-all allowed (370 s observed).
        //
        // The `gpu_submit` span makes that wait *visible*: a worker stuck here previously logged
        // nothing at all for the whole stall, which is why the multi-minute steps looked like compute.
        // `dev` is the point of this field: the span is entered on the SUBMITTING (rayon) thread, not
        // inside the `nassau-gpu<dev>` worker, so neither the thread id nor its name says which device a
        // job went to. Without it there is no way to check shard balance from a log.
        let submit_span = tracing::debug_span!(
            "gpu_submit",
            dev = dev,
            rows = num_rows,
            pairs = total_pairs,
            out = out_len
        );
        // Submit and return the wait: the caller launches every device's share before blocking on any
        // of them, so the shards actually overlap.
        let pending = submit_span.in_scope(|| {
            gpu_thread::submit_on(dev, move || {
                // Arbitrate against the `fp-cuda` row reduction from the one thread that submits (see the
                // note where the permit is taken). Dropped at the end of this task.
                let _shared = fp::gpu_lock::shared();
                let client = gpu_client();
                // Bind the segmented resident master/basis (see [`SegBuf`], [`seg_grow`]). Each store is
                // `MASTER_MAX_SEG` segment handles padded with a never-indexed 1-element dummy; a
                // single-buffer store (transient enum scratch or the passthrough diagnostic) is bound as
                // segment 0, which the kernel resolves correctly because `seg_elems` exceeds its length so
                // every offset lands in segment 0. `seg_grow!` re-uploads only the tail past the resident
                // prefix (`need_*`), never copying existing segments — the no-`~2×`-spike growth that keeps
                // cubecl out of its memory-corruption regime.
                let seg_elems = master_seg_elems();
                let dummy16 = client.create_from_slice(u16::as_bytes(&[0u16]));
                let dummy32 = client.create_from_slice(u32::as_bytes(&[0u32]));
                let pad_u16 = |mut v: Vec<(Handle, usize)>| -> Vec<(Handle, usize)> {
                    assert!(
                        v.len() <= MASTER_MAX_SEG,
                        "segment count exceeds MASTER_MAX_SEG"
                    );
                    while v.len() < MASTER_MAX_SEG {
                        v.push((dummy16.clone(), 1));
                    }
                    v
                };
                let pad_u32 = |mut v: Vec<(Handle, usize)>| -> Vec<(Handle, usize)> {
                    assert!(
                        v.len() <= MASTER_MAX_SEG,
                        "segment count exceeds MASTER_MAX_SEG"
                    );
                    while v.len() < MASTER_MAX_SEG {
                        v.push((dummy32.clone(), 1));
                    }
                    v
                };
                let full = |segs: Vec<Handle>| -> Vec<(Handle, usize)> {
                    segs.into_iter().map(|h| (h, seg_elems)).collect()
                };

                // `Transient` (degree > cap `R`s): enumerate this block's cold master ON the device into
                // scratch, freed with the launch. `Resident` (default): grow + reuse the shared master.
                let (cs_seg, mk_seg) = match mode {
                    MasterMode::Resident => {
                        let (cs_segs, _) = seg_grow!(
                            client,
                            resident_dev(),
                            cs,
                            resident_upload(),
                            need_cs,
                            copy_into_u16,
                            u16::as_bytes,
                            u16,
                            |_up: usize| {
                                let mut h = RESIDENT_HOST.write().unwrap();
                                let dev = cur_device();
                                let nl = h.cs_len[dev];
                                (std::mem::take(&mut h.cs_pending[dev]), nl)
                            }
                        );
                        let (mk_segs, _) = seg_grow!(
                            client,
                            resident_dev(),
                            mk,
                            resident_upload(),
                            need_mk,
                            copy_into_u16,
                            u16::as_bytes,
                            u16,
                            |_up: usize| {
                                let mut h = RESIDENT_HOST.write().unwrap();
                                let dev = cur_device();
                                let nl = h.mk_len[dev];
                                (std::mem::take(&mut h.mk_pending[dev]), nl)
                            }
                        );
                        (pad_u16(full(cs_segs)), pad_u16(full(mk_segs)))
                    }
                    MasterMode::Transient => {
                        // The enumeration launch is issued before the multiply on this same stream, so the
                        // scratch is fully written when the multiply reads it (one-stream launches are
                        // ordered, as with `zero_u32` below).
                        const ENUM_THREADS: u32 = 256;
                        let n_cold = enum_rows.len();
                        let cs_cap = need_cs.max(1);
                        let mk_cap = need_mk.max(1);
                        assert!(
                            cs_cap <= seg_elems && mk_cap <= seg_elems,
                            "transient scratch ({cs_cap}/{mk_cap} u16) exceeds one segment \
                             ({seg_elems}); raise NASSAU_GPU_MASTER_SEG_ELEMS"
                        );
                        let cs_scratch = client.empty(cs_cap * size_of::<u16>());
                        let mk_scratch = client.empty(mk_cap * size_of::<u16>());
                        let cnt_scratch = client.empty(n_cold.max(1) * size_of::<u32>());
                        let epp_h = client.create_from_slice(u32::as_bytes(&enum_pp));
                        let er_h = client.create_from_slice(u32::as_bytes(&enum_rows));
                        let ec_h = client.create_from_slice(u32::as_bytes(&enum_cols));
                        let eco_h = client.create_from_slice(u64::as_bytes(&r_cs_offset));
                        let emo_h = client.create_from_slice(u64::as_bytes(&r_mk_offset));
                        unsafe {
                            enumerate_admissible_kernel::launch_unchecked::<CudaRuntime>(
                                &client,
                                CubeCount::Static(
                                    (n_cold as u32).div_ceil(ENUM_THREADS).max(1),
                                    1,
                                    1,
                                ),
                                CubeDim::new_1d(ENUM_THREADS),
                                BufferArg::from_raw_parts(epp_h, enum_pp.len()),
                                BufferArg::from_raw_parts(er_h, n_cold),
                                BufferArg::from_raw_parts(ec_h, n_cold),
                                BufferArg::from_raw_parts(eco_h, n_cold),
                                BufferArg::from_raw_parts(emo_h, n_cold),
                                BufferArg::from_raw_parts(cs_scratch.clone(), cs_cap),
                                BufferArg::from_raw_parts(mk_scratch.clone(), mk_cap),
                                BufferArg::from_raw_parts(cnt_scratch, n_cold.max(1)),
                                enum_width,
                                n_cold,
                            );
                        }
                        (
                            pad_u16(vec![(cs_scratch, cs_cap)]),
                            pad_u16(vec![(mk_scratch, mk_cap)]),
                        )
                    }
                };
                // Resident basis segments (default) or per-launch passthrough buffers (A/B diagnostic) bound
                // as segment 0. Every `gei` a thread dereferences is `< need_basis_elems`, so growing the
                // basis to `need_basis_elems` (pp: `× width`) covers it.
                let (pp_seg, ln_seg) = if passthrough {
                    assert!(
                        term_pparts.len() <= seg_elems && term_lens.len() <= seg_elems,
                        "passthrough basis exceeds one segment; raise NASSAU_GPU_MASTER_SEG_ELEMS"
                    );
                    let bp = client.create_from_slice(u16::as_bytes(&term_pparts));
                    let bl = client.create_from_slice(u32::as_bytes(&term_lens));
                    (
                        pad_u16(vec![(bp, term_pparts.len())]),
                        pad_u32(vec![(bl, term_lens.len())]),
                    )
                } else {
                    let (pp_segs, _) = seg_grow!(
                        client,
                        resident_basis_dev(),
                        pp,
                        resident_basis_upload(),
                        need_basis_elems * width,
                        copy_into_u16,
                        u16::as_bytes,
                        u16,
                        |up: usize| {
                            let h = RESIDENT_BASIS_HOST.read().unwrap();
                            let nl = h.lens.len() * h.width;
                            (h.pparts[up..nl].to_vec(), nl)
                        }
                    );
                    let (ln_segs, _) = seg_grow!(
                        client,
                        resident_basis_dev(),
                        ln,
                        resident_basis_upload(),
                        need_basis_elems,
                        copy_into_u32,
                        u32::as_bytes,
                        u32,
                        |up: usize| {
                            let h = RESIDENT_BASIS_HOST.read().unwrap();
                            let nl = h.lens.len();
                            (h.lens[up..nl].to_vec(), nl)
                        }
                    );
                    (pad_u16(full(pp_segs)), pad_u32(full(ln_segs)))
                };
                // Upload the block's data — term data, seqno/xi tables, per-`R` offsets, per-product
                // records, the pair prefix sum, and the (zeroed) output buffer — and launch once: the
                // caller has already bounded this block's pair count and output size.
                // Hand the marshalled buffers over (`create`) rather than have cubecl copy out of a
                // borrowed slice (`create_from_slice`): the marshal already built exactly the bytes the
                // upload wants, so the extra staging copy is pure waste. Mirrors what [`BatchOutput`]
                // does on the way back. NOT using `client.staging()` to pin these: it consumes the
                // `Bytes` by value (so a buffer cannot be pinned once and reused across launches) and
                // its own docs note it blocks the compute queue.
                let tg_h = client.create(Bytes::from_elems(term_gei));
                // `g`/`xi` are identical every launch at this degree: fetch the shared resident copies
                // (uploaded once, re-uploaded only on a degree bump) instead of re-uploading them here.
                let (g_h, xi_h) = resident_seqno!(client, g, xi);
                let rco_h = client.create_from_slice(u64::as_bytes(&r_cs_offset));
                let rmo_h = client.create_from_slice(u64::as_bytes(&r_mk_offset));
                let rcl_h = client.create_from_slice(u32::as_bytes(&r_cs_len));
                let rml_h = client.create_from_slice(u32::as_bytes(&r_mk_len));
                // Per-`R` matrix count, so the kernel reads it instead of dividing for it. Integer
                // division by a runtime value is emulated on the GPU (I2F/MUFU.RCP/F2I plus fixups,
                // ~20 instructions), and this kernel is issue-limited on integer work.
                let r_num_mats_u32: Vec<u32> = r_num_matrices.iter().map(|&n| n as u32).collect();
                let rnm_h = client.create_from_slice(u32::as_bytes(&r_num_mats_u32));
                const THREADS: u32 = 256;
                // No realloc barrier needed: the resident master/basis are append-only segmented stores whose
                // segments, once allocated and written, never change identity and are never freed (see
                // [`seg_grow`]). This block cloned their segment handles above, so each stays alive (refcount
                // > 0) for the whole kernel even if another thread grows the store concurrently by appending
                // a new segment — the churny whole-buffer swap that needed quiescing is gone.
                // Allocate the XOR accumulator uninitialized and zero it on-device (see [`zero_u32`]),
                // instead of uploading a hundreds-of-MB host zero buffer — the former dominant serial
                // marshaling cost. Bounded by the caller's row-batching (see `get_partial_matrix`), so it
                // stays small and is returned to the pool by `memory_cleanup` below. Same stream as the
                // multiply, so the zero is ordered before it.
                let out_h = client.empty(out_len * size_of::<u32>());
                unsafe {
                    zero_u32::launch::<CudaRuntime>(
                        &client,
                        CubeCount::Static((out_len as u32).div_ceil(THREADS).max(1), 1, 1),
                        CubeDim::new_1d(THREADS),
                        BufferArg::from_raw_parts(out_h.clone(), out_len),
                    );
                }

                let pri_h = client.create(Bytes::from_elems(prod_r_index));
                let pts_h = client.create(Bytes::from_elems(prod_term_start));
                let pnt_h = client.create(Bytes::from_elems(prod_num_terms));
                let prb_h = client.create(Bytes::from_elems(prod_row_base));
                let poo_h = client.create(Bytes::from_elems(prod_out_offset));
                let pps_h = client.create(Bytes::from_elems(pps));
                let coarse_len = coarse.len();
                let coarse_h = client.create(Bytes::from_elems(coarse));
                let cubes = (total_pairs as u32).div_ceil(THREADS).max(1);
                // Search depth over a single coarse chunk's product span, not over every product: the
                // coarse index brackets the answer first, so this is `ceil(log2(span))` rather than
                // `ceil(log2(num_products))`.
                let search_iters =
                    usize::BITS as usize - (coarse_span + 1).max(1).leading_zeros() as usize;
                // 80 bytes per launch; lets the kernel unpack `working` without a per-thread array.
                let (pp_shift_h, pp_mask_h) = ppart_shift_mask();
                let pp_shift_len = pp_shift_h.len();
                let psh_h = client.create(Bytes::from_elems(pp_shift_h));
                let pms_h = client.create(Bytes::from_elems(pp_mask_h));
                // Segments actually populated across the four segmented stores; the rest are the
                // never-indexed 1-element dummies. Passed bare so the kernel's select chain specialises to
                // this many arms instead of all `MASTER_MAX_SEG`.
                let num_segs = need_cs
                    .max(need_mk)
                    .max(need_basis_elems * width)
                    .div_ceil(seg_elems)
                    .max(1);
                // Per-thread working size this block actually needs. `mk_len` bounds the assembled
                // p-part, and a term's own p-part is at most `MAX_XI_TAU` long. Rounded to a multiple
                // of 4 so the number of distinct comptime values (hence NVRTC recompiles) stays small
                // while still tracking the degree — a hardcoded 16 would be right to t~510 and
                // silently truncate past stem ~300.
                let work_cap = (r_mk_len
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0)
                    .max(MAX_XI_TAU as u32) as usize)
                    .div_ceil(4)
                    * 4;
                assert!(
                    work_cap <= WORKING_CAP,
                    "block needs a working array of {work_cap} > WORKING_CAP {WORKING_CAP}; raise \
                     the cap (it also bounds the host-side `xi` padding)"
                );
                if launch_log_enabled() {
                    let mk_max = r_mk_len.iter().copied().max().unwrap_or(0);
                    let mk_sum: u64 = r_mk_len.iter().map(|&x| x as u64).sum();
                    eprintln!(
                        "[launch] work_cap={work_cap} mk_max={mk_max} mk_mean={:.1} n_r={} \
                         products={} pairs={} cubes={cubes}",
                        mk_sum as f64 / r_mk_len.len().max(1) as f64,
                        r_mk_len.len(),
                        num_products,
                        total_pairs,
                    );
                }
                // Bind one `BufferArg` per `(segment vector, index)` — the `.0` handle, `.1` element length.
                macro_rules! sa {
                    ($v:expr, $i:expr) => {
                        BufferArg::from_raw_parts($v[$i].0.clone(), $v[$i].1)
                    };
                }
                // SAFETY: `launch_unchecked` — see the kernel's `address_type = "u64"` note. Every device
                // read is in-bounds by construction (uploaded `need_*` prefix, per-segment select, `j` guards).
                unsafe {
                    multiply_batch_kernel::launch_unchecked::<CudaRuntime>(
                        &client,
                        CubeCount::Static(cubes, 1, 1),
                        CubeDim::new_1d(THREADS),
                        sa!(cs_seg, 0),
                        sa!(cs_seg, 1),
                        sa!(cs_seg, 2),
                        sa!(cs_seg, 3),
                        sa!(cs_seg, 4),
                        sa!(cs_seg, 5),
                        sa!(cs_seg, 6),
                        sa!(cs_seg, 7),
                        sa!(cs_seg, 8),
                        sa!(cs_seg, 9),
                        sa!(cs_seg, 10),
                        sa!(cs_seg, 11),
                        sa!(cs_seg, 12),
                        sa!(cs_seg, 13),
                        sa!(cs_seg, 14),
                        sa!(cs_seg, 15),
                        sa!(mk_seg, 0),
                        sa!(mk_seg, 1),
                        sa!(mk_seg, 2),
                        sa!(mk_seg, 3),
                        sa!(mk_seg, 4),
                        sa!(mk_seg, 5),
                        sa!(mk_seg, 6),
                        sa!(mk_seg, 7),
                        sa!(mk_seg, 8),
                        sa!(mk_seg, 9),
                        sa!(mk_seg, 10),
                        sa!(mk_seg, 11),
                        sa!(mk_seg, 12),
                        sa!(mk_seg, 13),
                        sa!(mk_seg, 14),
                        sa!(mk_seg, 15),
                        sa!(pp_seg, 0),
                        sa!(pp_seg, 1),
                        sa!(pp_seg, 2),
                        sa!(pp_seg, 3),
                        sa!(pp_seg, 4),
                        sa!(pp_seg, 5),
                        sa!(pp_seg, 6),
                        sa!(pp_seg, 7),
                        sa!(pp_seg, 8),
                        sa!(pp_seg, 9),
                        sa!(pp_seg, 10),
                        sa!(pp_seg, 11),
                        sa!(pp_seg, 12),
                        sa!(pp_seg, 13),
                        sa!(pp_seg, 14),
                        sa!(pp_seg, 15),
                        sa!(ln_seg, 0),
                        sa!(ln_seg, 1),
                        sa!(ln_seg, 2),
                        sa!(ln_seg, 3),
                        sa!(ln_seg, 4),
                        sa!(ln_seg, 5),
                        sa!(ln_seg, 6),
                        sa!(ln_seg, 7),
                        sa!(ln_seg, 8),
                        sa!(ln_seg, 9),
                        sa!(ln_seg, 10),
                        sa!(ln_seg, 11),
                        sa!(ln_seg, 12),
                        sa!(ln_seg, 13),
                        sa!(ln_seg, 14),
                        sa!(ln_seg, 15),
                        BufferArg::from_raw_parts(tg_h, term_gei_len),
                        BufferArg::from_raw_parts(g_h, g.len()),
                        BufferArg::from_raw_parts(xi_h, xi.len()),
                        BufferArg::from_raw_parts(out_h.clone(), out_len),
                        BufferArg::from_raw_parts(rco_h, r_cs_offset.len()),
                        BufferArg::from_raw_parts(rmo_h, r_mk_offset.len()),
                        BufferArg::from_raw_parts(rcl_h, r_cs_len.len()),
                        BufferArg::from_raw_parts(rml_h, r_mk_len.len()),
                        BufferArg::from_raw_parts(rnm_h, r_num_mats_u32.len()),
                        BufferArg::from_raw_parts(pri_h, num_products),
                        BufferArg::from_raw_parts(pts_h, num_products),
                        BufferArg::from_raw_parts(pnt_h, num_products),
                        BufferArg::from_raw_parts(prb_h, num_products),
                        BufferArg::from_raw_parts(poo_h, num_products),
                        BufferArg::from_raw_parts(pps_h, pps_len),
                        BufferArg::from_raw_parts(coarse_h, coarse_len),
                        width,
                        seg_elems,
                        num_limbs,
                        search_iters,
                        num_segs,
                        BufferArg::from_raw_parts(psh_h, pp_shift_len),
                        BufferArg::from_raw_parts(pms_h, pp_shift_len),
                        work_cap.min(PPART_MAX_LEN),
                    );
                }

                // KNOWN BUG, NOT YET FIXED — a full stem-200 crashes with
            // `CUDA_ERROR_LAUNCH_FAILED, "unspecified launch failure"` (observed at `max_t=304`
            // after 2673 s; NOT the cleanup race, `NASSAU_GPU_CLEANUP_EVERY=0` was set).
            //
            // Because this closure no longer blocks, it RETURNS AND DROPS ITS INPUT HANDLES while
            // the kernel may still be running, so cubecl can hand those pages to a later allocation
            // that the running kernel is still reading. `read_one` blocked, which kept them alive
            // until the kernel retired. `BufferArg::from_raw_parts` consuming the handles does NOT
            // save this: `cs_seg`/`mk_seg` are `Vec<(Handle, usize)>` CLONES with their own
            // lifetimes, as are the dummies and the `r*_h` group.
            //
            // Fix: return `(DynFut, Vec<Handle>)` and drop the vec after `block_on`, exactly as the
            // permit now does — clone each handle into it immediately before `launch_unchecked`.
            //
            // Issue the readback but DO NOT wait for it. `read_async` enqueues the device→host copy
                // into pinned memory and records a CUDA event, then hands back a future whose entire body
                // is that event's wait (`cubecl-cuda` `command.rs`, `Fence::wait_sync`). Returning it
                // un-awaited is what makes the pipeline deeper than one kernel: this worker goes straight
                // back to `rx.recv()` and launches the next block while this one is still executing,
                // whereas the previous `read_one` (= `block_on(read_async(..))`) pinned the worker here
                // until the kernel retired, so each device ran exactly ONE launch at a time no matter how
                // many callers were queued behind it.
                //
                // The caller awaits it in the wait closure below — cubecl's own `Fence` doc names this
                // the intended pattern ("allows the server to continue accepting other tasks"). No
                // executor is involved: the future never yields `Pending` (it wraps a blocking
                // `cuEventSynchronize`), so `block_on` polls it exactly once. The buffer itself is still
                // handed back with no copy (see [`BatchOutput`]); `out_h` stays alive inside the future.
                let result = client.read_async(vec![out_h]);

                // Trim this stream's transient pool. Historically this per-launch cleanup RENUMBERED the
                // exclusive pool's page indices (`update_page`), which under ~100-way concurrency corrupted
                // cached page handles on other streams → `ManagedMemoryDescriptor` id-mismatch /
                // `CUDA_ERROR_LAUNCH_FAILED` at high stems (tracel-ai/cubecl#1401). The generational-slot pool
                // fix (JoeyBF/cubecl@claude/pool-slot-map-v0.10.0) gives pages stable ids so cleanup no longer
                // renumbers, making this safe again — and it keeps the retained pool bounded (freed pages
                // returned to the driver) so device memory tracks the working set instead of ratcheting.
                // Throttled by `NASSAU_GPU_CLEANUP_EVERY` (see [`cleanup_every`]) to probe whether the residual
                // high-stem `LAUNCH_FAILED` is a cross-stream cleanup-reclaim race.
                //
                // This now runs with this launch's work still IN FLIGHT (the readback above is not
                // awaited). That is safe for the output buffer specifically — the future holds a `Handle`
                // clone of `out_h`, so its page cannot be reclaimed — and it does not change the input
                // buffers' exposure, which the launch already consumed and dropped before any wait even
                // in the blocking version. Production runs set `NASSAU_GPU_CLEANUP_EVERY=0` regardless.
                let every = cleanup_every();
                if every != 0 && CLEANUP_COUNTER.fetch_add(1, Ordering::Relaxed) % every == 0 {
                    client.memory_cleanup();
                }

                result
            })
        });

        Box::new(move || {
            // Two waits, deliberately measured apart. `pending.wait()` returns as soon as the worker has
            // *launched* this block and issued its readback; `block_on` then waits for the device to
            // finish. Splitting them is how a log can tell a full pipeline from an empty one: if
            // `launch_ms` is most of the total the worker is the bottleneck (jobs queued behind other
            // launches), whereas if `fence_ms` dominates the device is genuinely busy, which is the
            // regime we want. Conflated into one figure — as they were when the worker did the readback
            // — the two are indistinguishable, which is how the one-kernel-deep pipeline stayed hidden
            // through three separate fan-out rewrites.
            let t_launch = std::time::Instant::now();
            let (fut, timing) = pending.wait();
            let launch_ms = t_launch.elapsed().as_secs_f64() * 1e3;

            let t_fence = std::time::Instant::now();
            let result = cubecl_common::future::block_on(fut)
                .expect("GPU readback failed")
                .remove(0);
            let fence_ms = t_fence.elapsed().as_secs_f64() * 1e3;

            // Only now is the output buffer free — device page and pinned host landing both — so this is
            // where the byte budget must be released. Explicit rather than implicit: the whole point of
            // moving it here is that dropping it earlier silently unbounds memory (see its acquisition).
            drop(permit);

            BATCH_LAUNCH_US.fetch_add(
                (launch_ms * 1e3) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BATCH_FENCE_US.fetch_add(
                (fence_ms * 1e3) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BATCH_INTERN_US.fetch_add(
                (intern_ms * 1e3) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BATCH_BASIS_US.fetch_add(
                (basis_ms * 1e3) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BATCH_TGEI_US.fetch_add((tgei_ms * 1e3) as u64, std::sync::atomic::Ordering::Relaxed);

            // Aggregate marshal/device totals across every launch (cheap, always on) so a whole
            // resolution's GPU overhead can be split host-vs-device via [`take_batch_stats`].
            let device_ms = t_device.elapsed().as_secs_f64() * 1e3;
            // Keep the value this call was assigned: with ~100 workers incrementing, a separate `load`
            // races past exact multiples, so a `% every == 0` test on it can fire never (observed: zero
            // reports over 12 minutes). `fetch_add` returns a unique ticket per call, so exactly one
            // caller sees each multiple.
            let call_no = BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            BATCH_MARSHAL_US.fetch_add(
                (marshal_ms * 1e3) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BATCH_DEVICE_US.fetch_add(
                (device_ms * 1e3) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BATCH_PAIRS.fetch_add(real_pairs as u64, std::sync::atomic::Ordering::Relaxed);
            BATCH_PREP_US.fetch_add((prep_ms * 1e3) as u64, std::sync::atomic::Ordering::Relaxed);
            BATCH_WAIT_US.fetch_add((wait_ms * 1e3) as u64, std::sync::atomic::Ordering::Relaxed);
            BATCH_PERMIT_US.fetch_add(
                (permit_ms * 1e3) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BATCH_LOCK_US.fetch_add((lock_ms * 1e3) as u64, std::sync::atomic::Ordering::Relaxed);
            BATCH_QUEUE_US.fetch_add(
                (timing.queue_ms * 1e3) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BATCH_EXEC_US.fetch_add(
                (timing.exec_ms * 1e3) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            BATCH_DEPTH_SUM.fetch_add(timing.depth, std::sync::atomic::Ordering::Relaxed);
            BATCH_DEPTH_MAX.fetch_max(timing.depth, std::sync::atomic::Ordering::Relaxed);

            // Periodic split of where multiply time actually goes. The counters above were being collected
            // and never read ([`take_batch_stats`] had no callers), which left the dominant cost of a
            // resolution unattributed: profiling a stem-200 run showed ~96% of the slow bidegrees' time
            // inside the per-signature parallel section (row reduction was ~2%), but nothing said whether
            // that is host marshalling or device execution. Non-resetting reads so the totals stay
            // cumulative; `NASSAU_BATCH_REPORT_EVERY=0` disables.
            let every = batch_report_every();
            if every != 0 && call_no % every == 0 {
                let calls = call_no;
                let marshal_s =
                    BATCH_MARSHAL_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let device_s =
                    BATCH_DEVICE_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let pairs = BATCH_PAIRS.load(std::sync::atomic::Ordering::Relaxed);
                let prep_s = BATCH_PREP_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let wait_s = BATCH_WAIT_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let permit_s =
                    BATCH_PERMIT_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let lock_s = BATCH_LOCK_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let queue_s =
                    BATCH_QUEUE_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let exec_s = BATCH_EXEC_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let depth_sum = BATCH_DEPTH_SUM.load(std::sync::atomic::Ordering::Relaxed);
                let depth_max = BATCH_DEPTH_MAX.load(std::sync::atomic::Ordering::Relaxed);
                let launch_s =
                    BATCH_LAUNCH_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let fence_s =
                    BATCH_FENCE_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
                let total = (prep_s + wait_s + device_s).max(1e-9);
                eprintln!(
                    "[batch-stats] calls={calls} prep={prep_s:.1}s permit={permit_s:.1}s \
                     lock={lock_s:.1}s device={device_s:.1}s | prep={:.0}% permit={:.0}% \
                     lock={:.0}% device={:.0}% pairs={pairs} (marshal={marshal_s:.1}s \
                     wait={wait_s:.1}s) queue={queue_s:.1}s exec={exec_s:.1}s | queue={:.0}% \
                     exec={:.0}% depth mean={:.1} max={depth_max} | launch={launch_s:.1}s \
                     fence={fence_s:.1}s pipeline={:.0}% | intern={:.1}s basis={:.1}s tgei={:.1}s",
                    100.0 * prep_s / total,
                    100.0 * permit_s / total,
                    100.0 * lock_s / total,
                    100.0 * device_s / total,
                    100.0 * queue_s / total,
                    100.0 * exec_s / total,
                    depth_sum as f64 / calls as f64,
                    // Share of the caller's wait spent on the device rather than queued behind other
                    // launches. ~100% is a full pipeline; the pre-change one-kernel-deep behaviour
                    // drives this toward 0 as callers pile up.
                    100.0 * fence_s / (launch_s + fence_s).max(1e-9),
                    BATCH_INTERN_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
                    BATCH_BASIS_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
                    BATCH_TGEI_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
                );
            }

            result
        }) as BlockWait
    })
}

/// How often [`multiply_batch_block`] prints the cumulative marshal/device split, in launches.
/// `NASSAU_BATCH_REPORT_EVERY` (default 2000; `0` disables).
/// `NASSAU_GPU_LAUNCH_LOG=1` dumps each multiply launch's shape (`work_cap`, the `mk_len` spread,
/// product count, pair count, grid size) to stderr.
///
/// This exists because launch shape is otherwise invisible, and inferring it from aggregates is how
/// several wrong conclusions got made here. It settled one directly: two bench processes each report
/// ~7x the solo `pairs/s`, which looked like enormous headroom, and the log showed both were issuing
/// *identical* launches (same `work_cap`, same ~2.5e9 pairs, same ~9.6e6 blocks). Same launch, same
/// device, "7x faster" — so the throughput figure, not the kernel, was the thing that changed. The
/// bench's fixed-work warm-up (4 jobs, ~190 s regardless of co-tenancy) confirmed real throughput is
/// flat. Read `pairs/s` from a contended run as meaningless, not as headroom.
fn launch_log_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NASSAU_GPU_LAUNCH_LOG").is_some())
}

fn batch_report_every() -> u64 {
    use std::sync::OnceLock;
    static EVERY: OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("NASSAU_BATCH_REPORT_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000)
    })
}

/// Read `data[o]` from a master split into up to [`MASTER_MAX_SEG`] fixed-size segments of
/// `seg_elems` elements each: segment `o / seg_elems`, local index `o % seg_elems`. This is the
/// no-copy-growth replacement for a single contiguous `Array<u16>` — appending a segment never
/// reallocates/copies the existing ones, so the device peak is `live_size + one_segment` instead of
/// the `~2×` realloc-doubling transient that pushes cubecl into its memory-corruption regime. The
/// per-segment `Array`s are separate kernel args because cubecl has no array-of-buffers; the branch
/// is the price of staying inside cubecl (vs raw CUDA VMM). `seg_elems` is runtime so tests can use
/// tiny segments; production sets it large. Keep the branch chain length equal to [`MASTER_MAX_SEG`].
#[cube]
#[allow(clippy::too_many_arguments)]
fn seg_read_u16(
    s0: &[u16],
    s1: &[u16],
    s2: &[u16],
    s3: &[u16],
    s4: &[u16],
    s5: &[u16],
    s6: &[u16],
    s7: &[u16],
    s8: &[u16],
    s9: &[u16],
    s10: &[u16],
    s11: &[u16],
    s12: &[u16],
    s13: &[u16],
    s14: &[u16],
    s15: &[u16],
    o: usize,
    seg_elems: usize,
    #[comptime] num_segs: usize,
) -> u16 {
    // `num_segs` is passed bare, so cubecl bakes it in as a compile-time constant and every
    // `num_segs > k` below folds away — the chain specialises to the segments that actually exist
    // instead of always testing all [`MASTER_MAX_SEG`] of them. This is the hot path: the gather
    // loop calls a `seg_read` 3 x WORKING_CAP = 96 times per thread, so a 16-way compare/select
    // chain per call dominated the instruction stream. Measured on an H200 with the kernel
    // resident: SM Active 100%, SM Issue 68%, DRAM read 1.1% of peak — pure instruction cost, not
    // bandwidth. The single-segment case skips the division too.
    let mut v = 0u16;
    if num_segs == 1 {
        v = s0[o];
    } else {
        let seg = o / seg_elems;
        let local = o % seg_elems;
        if seg == 0 {
            v = s0[local];
        } else if num_segs > 1 && seg == 1 {
            v = s1[local];
        } else if num_segs > 2 && seg == 2 {
            v = s2[local];
        } else if num_segs > 3 && seg == 3 {
            v = s3[local];
        } else if num_segs > 4 && seg == 4 {
            v = s4[local];
        } else if num_segs > 5 && seg == 5 {
            v = s5[local];
        } else if num_segs > 6 && seg == 6 {
            v = s6[local];
        } else if num_segs > 7 && seg == 7 {
            v = s7[local];
        } else if num_segs > 8 && seg == 8 {
            v = s8[local];
        } else if num_segs > 9 && seg == 9 {
            v = s9[local];
        } else if num_segs > 10 && seg == 10 {
            v = s10[local];
        } else if num_segs > 11 && seg == 11 {
            v = s11[local];
        } else if num_segs > 12 && seg == 12 {
            v = s12[local];
        } else if num_segs > 13 && seg == 13 {
            v = s13[local];
        } else if num_segs > 14 && seg == 14 {
            v = s14[local];
        } else {
            v = s15[local];
        }
    }
    v
}

/// `u32` sibling of [`seg_read_u16`] (the resident basis `lens` are u32). Same static-branch segment
/// select; see [`seg_read_u16`] for the layout and rationale.
#[cube]
#[allow(clippy::too_many_arguments)]
fn seg_read_u32(
    s0: &[u32],
    s1: &[u32],
    s2: &[u32],
    s3: &[u32],
    s4: &[u32],
    s5: &[u32],
    s6: &[u32],
    s7: &[u32],
    s8: &[u32],
    s9: &[u32],
    s10: &[u32],
    s11: &[u32],
    s12: &[u32],
    s13: &[u32],
    s14: &[u32],
    s15: &[u32],
    o: usize,
    seg_elems: usize,
    #[comptime] num_segs: usize,
) -> u32 {
    // `num_segs` is passed bare, so cubecl bakes it in as a compile-time constant and every
    // `num_segs > k` below folds away — the chain specialises to the segments that actually exist
    // instead of always testing all [`MASTER_MAX_SEG`] of them. This is the hot path: the gather
    // loop calls a `seg_read` 3 x WORKING_CAP = 96 times per thread, so a 16-way compare/select
    // chain per call dominated the instruction stream. Measured on an H200 with the kernel
    // resident: SM Active 100%, SM Issue 68%, DRAM read 1.1% of peak — pure instruction cost, not
    // bandwidth. The single-segment case skips the division too.
    let mut v = 0u32;
    if num_segs == 1 {
        v = s0[o];
    } else {
        let seg = o / seg_elems;
        let local = o % seg_elems;
        if seg == 0 {
            v = s0[local];
        } else if num_segs > 1 && seg == 1 {
            v = s1[local];
        } else if num_segs > 2 && seg == 2 {
            v = s2[local];
        } else if num_segs > 3 && seg == 3 {
            v = s3[local];
        } else if num_segs > 4 && seg == 4 {
            v = s4[local];
        } else if num_segs > 5 && seg == 5 {
            v = s5[local];
        } else if num_segs > 6 && seg == 6 {
            v = s6[local];
        } else if num_segs > 7 && seg == 7 {
            v = s7[local];
        } else if num_segs > 8 && seg == 8 {
            v = s8[local];
        } else if num_segs > 9 && seg == 9 {
            v = s9[local];
        } else if num_segs > 10 && seg == 10 {
            v = s10[local];
        } else if num_segs > 11 && seg == 11 {
            v = s11[local];
        } else if num_segs > 12 && seg == 12 {
            v = s12[local];
        } else if num_segs > 13 && seg == 13 {
            v = s13[local];
        } else if num_segs > 14 && seg == 14 {
            v = s14[local];
        } else {
            v = s15[local];
        }
    }
    v
}

/// In-kernel admissible-matrix enumeration: one thread per distinct `R`, generating that `R`'s
/// `col_sums`/`masks` for *every* admissible matrix directly into device scratch — the on-GPU
/// replacement for the resident/uploaded master (the stem-300 memory wall + the eviction re-upload
/// cost). This is the cubecl transcription of [`enumerate_admissible_ref`] (validated bit-exact on
/// the CPU), using only flag-guarded control flow — `while … && !found` in place of the odometer's
/// `break`/early `return`, `handled` in place of its `continue`. All per-thread state lives in
/// fixed-size local `Array`s ([`ENUM_MATRIX_CAP`] etc.); the values are small (≤ `u16`), stored as
/// `u16` exactly like the uploaded master so the multiply kernel reads them unchanged.
///
/// Inputs are per-`R`: `p_parts` (`n_r × width`, zero-padded), `r_rows`/`r_cols` (its dimensions),
/// and `r_cs_out`/`r_mk_out` (its base offset, in `u16` units, into the shared `out_cs`/`out_mk`
/// scratch — a host prefix-sum of `num_mats × cs_len` / `num_mats × mk_len`). `out_counts[ri]`
/// receives the number of matrices the thread emitted, so a count-only pre-pass can drive the
/// prefix-sum without any host enumeration. Runtime-agnostic: the same kernel lowers to CUDA (the
/// H200 path) and to the `cpu` backend (used by `admissible_enum_gpu_matches` to cross-check the
/// device lowering against `enumerate_admissible_ref` without a GPU).
//
// 64-bit addressing (`address_type = "u64"`, like `multiply_batch_kernel`): a big all-rows block's
// `out_cs`/`out_mk` scratch reaches multiple GB at high stems, so the flat element index (and the
// byte offset cubecl derives from it) overflows the default `u32` address type — the write lands at
// a wild address → `CUDA_ERROR_LAUNCH_FAILED`. `launch_unchecked` because checked u64 mode emits a
// `min(u64, u64)` NVRTC rejects; every access here is in-bounds by construction (the write offset is
// `r_cs_out[ri] + mat*cs_len + j < need_cs`, the scratch length, and reads are bounded by `n_r`).
#[cube(launch_unchecked, address_type = "u64")]
#[allow(clippy::too_many_arguments)]
fn enumerate_admissible_kernel(
    p_parts: &[u32],
    r_rows: &[u32],
    r_cols: &[u32],
    // u64: the scratch offsets index buffers that reach billions of elements in a big block, past
    // `u32::MAX` (same reason the multiply's `r_cs_offset`/`r_mk_offset` are u64 — these ARE those).
    r_cs_out: &[u64],
    r_mk_out: &[u64],
    out_cs: &mut [u16],
    out_mk: &mut [u16],
    out_counts: &mut [u32],
    width: usize,
    n_r: usize,
) {
    let ri = ABSOLUTE_POS;
    if ri >= n_r {
        terminate!();
    }
    let rows = usize::cast_from(r_rows[ri]);
    let cols = usize::cast_from(r_cols[ri]);
    let cs_len = cols - 1;
    let mk_len = rows + cols - 1;
    let pbase = ri * width;
    let cs_base = usize::cast_from(r_cs_out[ri]);
    let mk_base = usize::cast_from(r_mk_out[ri]);

    // Per-thread local state, mirroring `AdmissibleMatrix` / `enumerate_admissible_ref`. CUDA local
    // arrays are uninitialized, so every slot up to the comptime cap is explicitly zeroed first.
    let mut matrix = Array::<u32>::new(ENUM_MATRIX_CAP);
    let mut totals = Array::<u32>::new(ENUM_ROW_CAP);
    let mut col_sums = Array::<u32>::new(ENUM_COL_CAP);
    let mut masks = Array::<u32>::new(ENUM_MASK_CAP);
    for i in 0..ENUM_MATRIX_CAP {
        matrix[i] = 0u32;
    }
    for i in 0..ENUM_ROW_CAP {
        totals[i] = 0u32;
    }
    for i in 0..ENUM_COL_CAP {
        col_sums[i] = 0u32;
    }
    for i in 0..ENUM_MASK_CAP {
        masks[i] = 0u32;
    }
    // Column 0 of the matrix (and the initial masks) is the padded p_part.
    for i in 0..rows {
        let x = p_parts[pbase + i];
        matrix[i * cols] = x;
        masks[i] = x;
    }

    let mut mat = 0usize;
    let mut more = true;
    while more {
        // Emit the current matrix's col_sums/masks into this R's scratch slot.
        let co = cs_base + mat * cs_len;
        for j in 0..cs_len {
            out_cs[co + j] = u16::cast_from(col_sums[j]);
        }
        let mo = mk_base + mat * mk_len;
        for j in 0..mk_len {
            out_mk[mo + j] = u16::cast_from(masks[j]);
        }
        mat += 1;

        // One `next()` step: `found` = produced a new matrix (the ref's `return true`); `handled`
        // = this column already updated `totals` (the ref's `continue`). Loops guard on `!found`.
        let mut found = false;
        let mut row = 0usize;
        while row < rows && !found {
            let mut p_to_the_j = 1u32;
            totals[row] = matrix[row * cols];
            let mut col = 1usize;
            while col < cols && !found {
                p_to_the_j *= 2u32;
                let mut handled = false;
                if p_to_the_j <= totals[row] {
                    // Bitsum along the anti-diagonal to the bottom-left (saturating start index).
                    let mut d = 0u32;
                    let mut c = 0usize;
                    if row + col + 1 > rows {
                        c = row + col + 1 - rows;
                    }
                    while c < col {
                        d |= matrix[(row + col - c) * cols + c];
                        c += 1;
                    }
                    let cur = matrix[row * cols + col];
                    let new_entry = ((cur | d) + 1u32) & !d;
                    let inc = new_entry - cur;
                    let sub = inc * p_to_the_j;
                    if totals[row] < sub {
                        totals[row] += p_to_the_j * cur;
                        handled = true;
                    } else {
                        matrix[row * cols] = totals[row] - sub;
                        masks[row] = matrix[row * cols];
                        col_sums[col - 1] += inc;
                        let mut j = 1usize;
                        while j < col {
                            masks[row + j] &= !matrix[row * cols + j];
                            col_sums[j - 1] -= matrix[row * cols + j];
                            matrix[row * cols + j] = 0u32;
                            j += 1;
                        }
                        matrix[row * cols + col] = new_entry;
                        let mut i = 0usize;
                        while i < row {
                            matrix[i * cols] = totals[i];
                            masks[i] = totals[i];
                            let mut j2 = 1usize;
                            while j2 < cols {
                                if i + j2 > row {
                                    masks[i + j2] &= !matrix[i * cols + j2];
                                }
                                col_sums[j2 - 1] -= matrix[i * cols + j2];
                                matrix[i * cols + j2] = 0u32;
                                j2 += 1;
                            }
                            i += 1;
                        }
                        masks[row + col] = d | new_entry;
                        found = true;
                        handled = true;
                    }
                }
                if !handled {
                    totals[row] += p_to_the_j * matrix[row * cols + col];
                }
                col += 1;
            }
            row += 1;
        }
        more = found;
    }

    out_counts[ri] = u32::cast_from(mat);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Elementwise F₂ addition of two bit-packed vectors: `out[i] = a[i] ^ b[i]`.
    ///
    /// One thread per `u32` limb. F₂ addition is XOR of the packed limbs, so this is
    /// the output primitive the multiply kernels accumulate with.
    #[cube(launch)]
    fn xor_f2(a: &[u32], b: &[u32], out: &mut [u32]) {
        if ABSOLUTE_POS < out.len() {
            out[ABSOLUTE_POS] = a[ABSOLUTE_POS] ^ b[ABSOLUTE_POS];
        }
    }

    /// Compute `a ^ b` limb-wise on the default CUDA device.
    ///
    /// Host-side driver for `xor_f2`: uploads both operands, launches one thread per
    /// limb, and reads the result back. Panics if the operands differ in length.
    pub fn xor_f2_on_gpu(a: &[u32], b: &[u32]) -> Vec<u32> {
        assert_eq!(a.len(), b.len(), "operands must have equal limb counts");
        let n = a.len();
        let client = gpu_client();

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
                BufferArg::from_raw_parts(a_handle, n),
                BufferArg::from_raw_parts(b_handle, n),
                BufferArg::from_raw_parts(out_handle.clone(), n),
            );
        }

        let bytes = client.read_one(out_handle).unwrap();
        u32::from_bytes(&bytes).to_vec()
    }

    /// One thread per padded p_part: `out[i] = seqno(p_parts[i])`. `p_parts` is
    /// `n × width` row-major, each row a p_part zero-padded to `width` (padding entries
    /// are zero and skipped, so `wlen == width` matches the CPU's trimmed loop).
    #[cube(launch)]
    fn seqno_kernel(
        g: &[u32],
        xi: &[u32],
        p_parts: &[u32],
        out: &mut [u32],
        width: usize,
        pp_shift: &[u32],
        pp_mask: &[u32],
        #[comptime] sq_len: usize,
    ) {
        let idx = ABSOLUTE_POS;
        if idx >= out.len() {
            terminate!();
        }
        let base = idx * width;

        let mut working = 0u64;
        for h in 0..PPART_MAX_LEN {
            if h < width {
                working |= u64::cast_from(p_parts[base + h]) << u64::cast_from(pp_shift[h]);
            }
        }
        out[idx] = seqno_core_packed(g, xi, pp_shift, pp_mask, working, sq_len, width);
    }

    /// Run `seqno_kernel` over `n` padded p_parts and return their seqno indices.
    ///
    /// `g`/`xi` come from `MilnorAlgebra::seqno_table_u32` and
    /// [`crate::algebra::combinatorics::xi_degrees`]; `p_parts` is `n × width` row-major,
    /// each row a p_part zero-padded to `width`.
    pub fn seqno_batch_on_gpu(
        width: usize,
        xi: &[u32],
        g: &[u32],
        p_parts: &[u32],
        n: usize,
    ) -> Vec<u32> {
        assert_eq!(xi.len(), width, "xi must have `width` entries");
        assert_eq!(p_parts.len(), n * width, "p_parts must be n × width");
        let client = gpu_client();

        let g_h = client.create_from_slice(u32::as_bytes(g));
        let xi_h = client.create_from_slice(u32::as_bytes(xi));
        let pp_h = client.create_from_slice(u32::as_bytes(p_parts));
        let out_h = client.empty(n * size_of::<u32>());

        let (psh, pms) = ppart_shift_mask();
        let pp_len = psh.len();
        let psh_h = client.create(Bytes::from_elems(psh));
        let pms_h = client.create(Bytes::from_elems(pms));
        const THREADS: u32 = 256;
        let cubes = (n as u32).div_ceil(THREADS);
        unsafe {
            seqno_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(THREADS),
                BufferArg::from_raw_parts(g_h, g.len()),
                BufferArg::from_raw_parts(xi_h, xi.len()),
                BufferArg::from_raw_parts(pp_h, p_parts.len()),
                BufferArg::from_raw_parts(out_h.clone(), n),
                width,
                BufferArg::from_raw_parts(psh_h, pp_len),
                BufferArg::from_raw_parts(pms_h, pp_len),
                width.min(PPART_MAX_LEN),
            );
        }

        let bytes = client.read_one(out_h).unwrap();
        u32::from_bytes(&bytes).to_vec()
    }

    /// Multiply `Sq(R) · s` for a single fixed operation `R` into one F₂ output vector.
    /// One thread per `(matrix, term)` pair; delegates the assembly to `multiply_pair`.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    fn multiply_single_r_kernel(
        col_sums: &[u16],
        masks: &[u16],
        term_pparts: &[u16],
        term_lens: &[u32],
        g: &[u32],
        xi: &[u32],
        out: &mut [Atomic<u32>],
        num_terms: usize,
        num_matrices: usize,
        cs_len: usize,
        mk_len: usize,
        width: usize,
        num_limbs: usize,
        pp_shift: &[u32],
        pp_mask: &[u32],
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
            num_limbs,
            PPART_MAX_LEN,
            pp_shift,
            pp_mask,
        );
    }

    /// Compute `Sq(R) · s` on the GPU for a single operation `R = (r_degree, r_idx)`,
    /// returning the F₂ result as bit-packed `u32` limbs (bit `i` = basis index `i`).
    ///
    /// `term_indices` are the nonzero indices of `s` in the degree-`s_degree` basis.
    /// `R` must be non-empty (`Sq(∅) = 1` is the trivial identity the caller handles).
    /// Requires the algebra's basis and seqno tables built through `r_degree + s_degree`.
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
        let (cs_len, mk_len, cs32, mk32) = algebra.admissible_matrices(r.p_part);
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
            for (slot, v) in term_pparts[t * width..(t + 1) * width]
                .iter_mut()
                .zip(elt.p_part.iter())
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

        let client = gpu_client();
        let cs_h = client.create_from_slice(u16::as_bytes(&col_sums));
        let mk_h = client.create_from_slice(u16::as_bytes(&masks));
        let tp_h = client.create_from_slice(u16::as_bytes(&term_pparts));
        let tl_h = client.create_from_slice(u32::as_bytes(&term_lens));
        let g_h = client.create_from_slice(u32::as_bytes(&g));
        let xi_h = client.create_from_slice(u32::as_bytes(&xi));
        let zeros = vec![0u32; num_limbs];
        let out_h = client.create_from_slice(u32::as_bytes(&zeros));

        let (psh, pms) = ppart_shift_mask();
        let pp_len = psh.len();
        let psh_h = client.create(Bytes::from_elems(psh));
        let pms_h = client.create(Bytes::from_elems(pms));
        let total_pairs = num_matrices * num_terms;
        const THREADS: u32 = 256;
        let cubes = (total_pairs as u32).div_ceil(THREADS).max(1);
        unsafe {
            multiply_single_r_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(THREADS),
                BufferArg::from_raw_parts(cs_h, col_sums.len()),
                BufferArg::from_raw_parts(mk_h, masks.len()),
                BufferArg::from_raw_parts(tp_h, term_pparts.len()),
                BufferArg::from_raw_parts(tl_h, term_lens.len()),
                BufferArg::from_raw_parts(g_h, g.len()),
                BufferArg::from_raw_parts(xi_h, xi.len()),
                BufferArg::from_raw_parts(out_h.clone(), num_limbs),
                num_terms,
                num_matrices,
                cs_len,
                mk_len,
                width,
                num_limbs,
                BufferArg::from_raw_parts(psh_h, pp_len),
                BufferArg::from_raw_parts(pms_h, pp_len),
            );
        }

        let bytes = client.read_one(out_h).unwrap();
        u32::from_bytes(&bytes).to_vec()
    }

    /// Validation kernel for [`seg_read_u16`]: `out[i] = segmented[idx[i]]`. Lets a test assert the
    /// segmented read reproduces a contiguous buffer bit-for-bit (see `seg_read_matches_contiguous`).
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    fn seg_gather_kernel(
        s0: &[u16],
        s1: &[u16],
        s2: &[u16],
        s3: &[u16],
        s4: &[u16],
        s5: &[u16],
        s6: &[u16],
        s7: &[u16],
        s8: &[u16],
        s9: &[u16],
        s10: &[u16],
        s11: &[u16],
        s12: &[u16],
        s13: &[u16],
        s14: &[u16],
        s15: &[u16],
        idx: &[u32],
        out: &mut [u16],
        seg_elems: usize,
        #[comptime] num_segs: usize,
    ) {
        let i = ABSOLUTE_POS;
        if i >= out.len() {
            terminate!();
        }
        out[i] = seg_read_u16(
            s0,
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
            s8,
            s9,
            s10,
            s11,
            s12,
            s13,
            s14,
            s15,
            usize::cast_from(idx[i]),
            seg_elems,
            num_segs,
        );
    }

    /// Backend-agnostic host driver for [`enumerate_admissible_kernel`]. Lays out each `R`'s scratch
    /// slot from the supplied per-`R` `num_mats` (a prefix-sum of `num_mats·cs_len` / `num_mats·mk_len`),
    /// uploads the compact per-`R` inputs, launches one thread per `R` on `device`, and reads back the
    /// packed `(out_cs, out_mk, counts)`. Generic over [`Runtime`] so the *same* kernel can be run on
    /// CUDA (the H200 path) and on the `cpu` backend — the cross-lowering check the caller uses to
    /// confirm the device semantics match [`enumerate_admissible_ref`] without needing a GPU.
    fn enumerate_admissible_on_runtime<R: Runtime>(
        device: &R::Device,
        p_parts: &[Vec<u32>],
        num_mats: &[u32],
    ) -> (Vec<u16>, Vec<u16>, Vec<u32>) {
        let n_r = p_parts.len();
        let width = p_parts.iter().map(Vec::len).max().unwrap();

        let mut pp_flat = vec![0u32; n_r * width];
        let mut r_rows = vec![0u32; n_r];
        let mut r_cols = vec![0u32; n_r];
        let mut r_cs_out = vec![0u64; n_r];
        let mut r_mk_out = vec![0u64; n_r];
        let mut cs_total = 0u64;
        let mut mk_total = 0u64;
        for (i, pp) in p_parts.iter().enumerate() {
            let rows = pp.len();
            let cols = pp
                .iter()
                .map(|&x| (u32::BITS - x.leading_zeros()) as usize)
                .max()
                .unwrap();
            let cs_len = (cols - 1) as u64;
            let mk_len = (rows + cols - 1) as u64;
            for (slot, &v) in pp_flat[i * width..i * width + rows].iter_mut().zip(pp) {
                *slot = v;
            }
            r_rows[i] = rows as u32;
            r_cols[i] = cols as u32;
            r_cs_out[i] = cs_total;
            r_mk_out[i] = mk_total;
            cs_total += num_mats[i] as u64 * cs_len;
            mk_total += num_mats[i] as u64 * mk_len;
        }

        let client = R::client(device);
        let pp_h = client.create_from_slice(u32::as_bytes(&pp_flat));
        let rr_h = client.create_from_slice(u32::as_bytes(&r_rows));
        let rc_h = client.create_from_slice(u32::as_bytes(&r_cols));
        let rco_h = client.create_from_slice(u64::as_bytes(&r_cs_out));
        let rmo_h = client.create_from_slice(u64::as_bytes(&r_mk_out));
        // `empty` needs a non-zero size even when a batch happens to have no matrices.
        let cs_cap = (cs_total.max(1)) as usize;
        let mk_cap = (mk_total.max(1)) as usize;
        let ocs_h = client.empty(cs_cap * size_of::<u16>());
        let omk_h = client.empty(mk_cap * size_of::<u16>());
        let cnt_h = client.empty(n_r * size_of::<u32>());

        const THREADS: u32 = 64;
        let cubes = (n_r as u32).div_ceil(THREADS);
        unsafe {
            enumerate_admissible_kernel::launch_unchecked::<R>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(THREADS),
                BufferArg::from_raw_parts(pp_h, pp_flat.len()),
                BufferArg::from_raw_parts(rr_h, n_r),
                BufferArg::from_raw_parts(rc_h, n_r),
                BufferArg::from_raw_parts(rco_h, n_r),
                BufferArg::from_raw_parts(rmo_h, n_r),
                BufferArg::from_raw_parts(ocs_h.clone(), cs_cap),
                BufferArg::from_raw_parts(omk_h.clone(), mk_cap),
                BufferArg::from_raw_parts(cnt_h.clone(), n_r),
                width,
                n_r,
            );
        }
        // Truncate off the `max(1)` padding element present when a batch has zero col_sums / masks (an
        // all-ones p_part gives `cs_len == 0`); the caller compares against the exact packed reference.
        let mut cs = u16::from_bytes(&client.read_one(ocs_h).unwrap()).to_vec();
        let mut mk = u16::from_bytes(&client.read_one(omk_h).unwrap()).to_vec();
        cs.truncate(cs_total as usize);
        mk.truncate(mk_total as usize);
        let counts = u32::from_bytes(&client.read_one(cnt_h).unwrap()).to_vec();
        (cs, mk, counts)
    }

    /// CPU reference for the planned *in-kernel* admissible-matrix enumeration — the direction that
    /// replaces the resident/uploaded master (the stem-300 memory wall + the eviction re-upload cost)
    /// by generating each `R`'s `col_sums`/`masks` ON THE GPU into a transient scratch buffer, never
    /// storing or uploading them. This reimplements [`MilnorAlgebra::admissible_matrices`] /
    /// `AdmissibleMatrix` using ONLY flag-guarded control flow — no `break`, `continue`, or early
    /// `return` — because that is the subset the cubecl DSL compiles cleanly (cf. `multiply_pair`,
    /// which tracks `rejected` rather than breaking). The eventual `#[cube]` kernel is then a mechanical
    /// transcription of this function onto per-thread local `Array`s (state is tiny: `rows = |p_part|`,
    /// `cols ≤ 32`). Returns the same `(cs_len, mk_len, col_sums, masks)` row-major flattening as
    /// `admissible_matrices`; `admissible_enum_ref_matches` asserts bit-exact equivalence over every
    /// real `R` up to degree 60, validating the flag-based restructuring before the hard-to-debug port.
    fn enumerate_admissible_ref(p_part: &[u32]) -> (usize, usize, Vec<u32>, Vec<u32>) {
        let rows = p_part.len();
        let cols = p_part
            .iter()
            .map(|&x| (u32::BITS - x.leading_zeros()) as usize)
            .max()
            .unwrap();
        let cs_len = cols - 1;
        let mk_len = rows + cols - 1;

        // State mirrors `AdmissibleMatrix`: `matrix` row-major `rows*cols` (column 0 = `p_part`),
        // `totals[rows]`, `col_sums[cs_len]`, `masks[mk_len]` (masks starts as the padded `p_part`).
        let mut matrix = vec![0u32; rows * cols];
        for (i, &x) in p_part.iter().enumerate() {
            matrix[i * cols] = x;
        }
        let mut totals = vec![0u32; rows];
        let mut col_sums = vec![0u32; cs_len];
        let mut masks = vec![0u32; mk_len];
        for (i, &x) in p_part.iter().enumerate() {
            masks[i] = x;
        }

        let mut out_cs: Vec<u32> = Vec::new();
        let mut out_mk: Vec<u32> = Vec::new();

        // Emit the current matrix, then advance; `more` is `AdmissibleMatrix::next`'s return value.
        let mut more = true;
        while more {
            out_cs.extend_from_slice(&col_sums);
            out_mk.extend_from_slice(&masks);

            // One `next()` step, flag-based: `found` = "produced a new matrix" (the original's
            // `return true`); `handled` = "this column already updated `totals`" (the original's
            // `continue 'mid`, which skips the trailing add). Loops are guarded by `!found` instead
            // of breaking.
            let mut found = false;
            let mut row = 0;
            while row < rows && !found {
                let mut p_to_the_j: u32 = 1;
                totals[row] = matrix[row * cols]; // get(row, 0)
                let mut col = 1;
                while col < cols && !found {
                    p_to_the_j *= 2;
                    let mut handled = false;
                    if p_to_the_j <= totals[row] {
                        // Bitsum along the anti-diagonal to the bottom-left.
                        let mut d = 0u32;
                        let mut c = (row + col + 1).saturating_sub(rows);
                        while c < col {
                            d |= matrix[(row + col - c) * cols + c];
                            c += 1;
                        }
                        let cur = matrix[row * cols + col];
                        let new_entry = ((cur | d) + 1) & !d;
                        let inc = new_entry - cur;
                        let sub = inc * p_to_the_j;
                        if totals[row] < sub {
                            totals[row] += p_to_the_j * cur;
                            handled = true;
                        } else {
                            matrix[row * cols] = totals[row] - sub; // set(row, 0, ..)
                            masks[row] = matrix[row * cols];
                            col_sums[col - 1] += inc;
                            let mut j = 1;
                            while j < col {
                                masks[row + j] &= !matrix[row * cols + j];
                                col_sums[j - 1] -= matrix[row * cols + j];
                                matrix[row * cols + j] = 0;
                                j += 1;
                            }
                            matrix[row * cols + col] = new_entry;
                            let mut i = 0;
                            while i < row {
                                matrix[i * cols] = totals[i];
                                masks[i] = totals[i];
                                let mut j = 1;
                                while j < cols {
                                    if i + j > row {
                                        masks[i + j] &= !matrix[i * cols + j];
                                    }
                                    col_sums[j - 1] -= matrix[i * cols + j];
                                    matrix[i * cols + j] = 0;
                                    j += 1;
                                }
                                i += 1;
                            }
                            masks[row + col] = d | new_entry;
                            found = true;
                            handled = true;
                        }
                    }
                    if !handled {
                        totals[row] += p_to_the_j * matrix[row * cols + col];
                    }
                    col += 1;
                }
                row += 1;
            }
            more = found;
        }

        (cs_len, mk_len, out_cs, out_mk)
    }

    /// Host driver for [`seg_gather_kernel`]: splits `data` into ≤ [`MASTER_MAX_SEG`] segments of
    /// `seg_elems`, uploads each as its own device buffer (no contiguous copy — the whole point), and
    /// returns `data[idx[i]]` gathered through the segmented read. Unused segments get a 1-element dummy
    /// (never indexed). Proves the segmented master reads identically to a contiguous one.
    fn seg_gather_on_gpu(data: &[u16], seg_elems: usize, indices: &[u32]) -> Vec<u16> {
        let n = data.len();
        let nseg = n.div_ceil(seg_elems).max(1);
        assert!(
            nseg <= MASTER_MAX_SEG,
            "prototype caps at {MASTER_MAX_SEG} segments"
        );
        let client = gpu_client();

        // One handle per segment slot; real segments hold their slice, unused slots a 1-elem dummy.
        let dummy = [0u16];
        let mut handles = Vec::with_capacity(MASTER_MAX_SEG);
        let mut lens = Vec::with_capacity(MASTER_MAX_SEG);
        for s in 0..MASTER_MAX_SEG {
            let lo = s * seg_elems;
            if lo < n {
                let hi = (lo + seg_elems).min(n);
                handles.push(client.create_from_slice(u16::as_bytes(&data[lo..hi])));
                lens.push(hi - lo);
            } else {
                handles.push(client.create_from_slice(u16::as_bytes(&dummy)));
                lens.push(1);
            }
        }
        let idx_h = client.create_from_slice(u32::as_bytes(indices));
        let out_h = client.empty(indices.len() * size_of::<u16>());

        const THREADS: u32 = 256;
        let cubes = (indices.len() as u32).div_ceil(THREADS).max(1);
        let arg = |i: usize| unsafe { BufferArg::from_raw_parts(handles[i].clone(), lens[i]) };
        unsafe {
            seg_gather_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(THREADS),
                arg(0),
                arg(1),
                arg(2),
                arg(3),
                arg(4),
                arg(5),
                arg(6),
                arg(7),
                arg(8),
                arg(9),
                arg(10),
                arg(11),
                arg(12),
                arg(13),
                arg(14),
                arg(15),
                BufferArg::from_raw_parts(idx_h, indices.len()),
                BufferArg::from_raw_parts(out_h.clone(), indices.len()),
                seg_elems,
                nseg,
            );
        }
        u16::from_bytes(&client.read_one(out_h).unwrap()).to_vec()
    }

    /// Shared body for the per-backend enumeration tests: builds a batch of every real `R` up to
    /// `max_degree`, computes the expected packed `col_sums`/`masks` (and per-`R` `num_mats`) from the
    /// CPU-validated [`enumerate_admissible_ref`], runs [`enumerate_admissible_kernel`] on `R`'s
    /// `device`, and asserts the device output is bit-exact — values *and* per-`R` counts. Generic so
    /// CUDA (H200) and the `cpu` backend run the identical kernel through it.
    fn check_enum_backend<Rt: Runtime>(device: &Rt::Device, max_degree: i32) {
        use fp::prime::ValidPrime;

        let p = ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);
        algebra.compute_basis(max_degree);

        // Process one degree per launch. At high degree the full master is tens of GB, so batching every
        // R together would OOM the host; per-degree keeps the expected arrays bounded AND pinpoints the
        // exact degree if the device lowering ever diverges from the CPU reference.
        let mut total_r = 0usize;
        let mut total_mats = 0u64;
        for deg in 1..=max_degree {
            let mut p_parts: Vec<Vec<u32>> = Vec::new();
            let mut num_mats: Vec<u32> = Vec::new();
            let mut exp_cs: Vec<u16> = Vec::new();
            let mut exp_mk: Vec<u16> = Vec::new();
            for idx in 0..algebra.dimension(deg) {
                let pp: Vec<u32> = algebra
                    .basis_element_from_index(deg, idx)
                    .p_part
                    .iter()
                    .collect();
                if pp.is_empty() {
                    continue;
                }
                let (_cs_len, mk_len, cs, mk) = enumerate_admissible_ref(&pp);
                // `mk_len = rows+cols-1 ≥ 1` always, so it recovers the matrix count even when
                // `cs_len == 0` (an all-ones p_part contributes no col_sums).
                num_mats.push((mk.len() / mk_len) as u32);
                exp_cs.extend(cs.iter().map(|&v| narrow_u16(v)));
                exp_mk.extend(mk.iter().map(|&v| narrow_u16(v)));
                p_parts.push(pp);
            }
            if p_parts.is_empty() {
                continue;
            }
            let (got_cs, got_mk, counts) =
                enumerate_admissible_on_runtime::<Rt>(device, &p_parts, &num_mats);
            assert_eq!(
                counts, num_mats,
                "per-R matrix counts diverged at degree {deg}"
            );
            assert_eq!(got_cs, exp_cs, "device col_sums diverged at degree {deg}");
            assert_eq!(got_mk, exp_mk, "device masks diverged at degree {deg}");
            total_r += p_parts.len();
            total_mats += num_mats.iter().map(|&m| m as u64).sum::<u64>();
        }
        assert!(total_r > 0, "no R's exercised");
        eprintln!(
            "enum backend: {total_r} R's, {total_mats} matrices bit-exact vs \
             enumerate_admissible_ref (degrees 1..={max_degree})"
        );
    }

    /// Time one enumeration launch on `device`, split into (marshal+upload, kernel, full readback).
    /// `kernel` reads only the tiny `counts` buffer to force a stream sync (so it captures kernel wall
    /// time without the big transfer); `readback` then pulls the full `col_sums`/`masks`. Used by
    /// `bench_admissible_cpu_vs_gpu` — production never reads the arrays back (the multiply consumes the
    /// scratch on-device), so `kernel` is the production-relevant cost and `readback` is bench-only.
    fn enum_launch_timed<R: Runtime>(
        device: &R::Device,
        p_parts: &[Vec<u32>],
        num_mats: &[u32],
    ) -> (f64, f64, f64) {
        use std::time::Instant;
        let n_r = p_parts.len();
        let width = p_parts.iter().map(Vec::len).max().unwrap();

        let t_marshal = Instant::now();
        let mut pp_flat = vec![0u32; n_r * width];
        let mut r_rows = vec![0u32; n_r];
        let mut r_cols = vec![0u32; n_r];
        let mut r_cs_out = vec![0u64; n_r];
        let mut r_mk_out = vec![0u64; n_r];
        let (mut cs_total, mut mk_total) = (0u64, 0u64);
        for (i, pp) in p_parts.iter().enumerate() {
            let rows = pp.len();
            let cols = pp
                .iter()
                .map(|&x| (u32::BITS - x.leading_zeros()) as usize)
                .max()
                .unwrap();
            for (slot, &v) in pp_flat[i * width..i * width + rows].iter_mut().zip(pp) {
                *slot = v;
            }
            r_rows[i] = rows as u32;
            r_cols[i] = cols as u32;
            r_cs_out[i] = cs_total;
            r_mk_out[i] = mk_total;
            cs_total += num_mats[i] as u64 * (cols - 1) as u64;
            mk_total += num_mats[i] as u64 * (rows + cols - 1) as u64;
        }
        let client = R::client(device);
        let pp_h = client.create_from_slice(u32::as_bytes(&pp_flat));
        let rr_h = client.create_from_slice(u32::as_bytes(&r_rows));
        let rc_h = client.create_from_slice(u32::as_bytes(&r_cols));
        let rco_h = client.create_from_slice(u64::as_bytes(&r_cs_out));
        let rmo_h = client.create_from_slice(u64::as_bytes(&r_mk_out));
        let cs_cap = cs_total.max(1) as usize;
        let mk_cap = mk_total.max(1) as usize;
        let ocs_h = client.empty(cs_cap * size_of::<u16>());
        let omk_h = client.empty(mk_cap * size_of::<u16>());
        let cnt_h = client.empty(n_r * size_of::<u32>());
        let marshal_s = t_marshal.elapsed().as_secs_f64();

        const THREADS: u32 = 64;
        let cubes = (n_r as u32).div_ceil(THREADS);
        let t_kernel = Instant::now();
        unsafe {
            enumerate_admissible_kernel::launch_unchecked::<R>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(THREADS),
                BufferArg::from_raw_parts(pp_h, pp_flat.len()),
                BufferArg::from_raw_parts(rr_h, n_r),
                BufferArg::from_raw_parts(rc_h, n_r),
                BufferArg::from_raw_parts(rco_h, n_r),
                BufferArg::from_raw_parts(rmo_h, n_r),
                BufferArg::from_raw_parts(ocs_h.clone(), cs_cap),
                BufferArg::from_raw_parts(omk_h.clone(), mk_cap),
                BufferArg::from_raw_parts(cnt_h.clone(), n_r),
                width,
                n_r,
            );
        }
        // Reading the tiny counts buffer blocks until the kernel completes: kernel wall time, ~no transfer.
        let _ = client.read_one(cnt_h).unwrap();
        let kernel_s = t_kernel.elapsed().as_secs_f64();

        let t_read = Instant::now();
        let _ = client.read_one(ocs_h).unwrap();
        let _ = client.read_one(omk_h).unwrap();
        let readback_s = t_read.elapsed().as_secs_f64();

        (marshal_s, kernel_s, readback_s)
    }

    /// The in-kernel [`enumerate_admissible_kernel`], run on the CUDA backend, must reproduce the
    /// CPU-validated [`enumerate_admissible_ref`] bit-for-bit over every real `R` up to degree 145 —
    /// validating the cubecl lowering of the flag-based enumeration (local arrays, bitops, u16
    /// stores) across the FULL degree range the eviction path exercises (cold R's reach ~144 at
    /// stem 150), not just the low degrees. Requires a live GPU + the CUDA toolkit env.
    #[test]
    fn admissible_enum_gpu_matches() {
        check_enum_backend::<CudaRuntime>(&CudaDevice::default(), 145);
    }

    /// The segmented master read ([`seg_read_u16`]) must reproduce a contiguous buffer bit-for-bit,
    /// including reads that land in every one of the [`MASTER_MAX_SEG`] segments and at segment
    /// boundaries. This validates the no-copy-growth mechanic before it is wired into the multiply's
    /// hot path. Requires a live GPU + the CUDA toolkit env.
    #[test]
    fn seg_read_matches_contiguous() {
        // 1000 elements over seg_elems=137 → 8 segments (0..137, 137..274, …, 959..1000), so every
        // segment slot is exercised, including the ragged last one and the boundaries between them.
        let data: Vec<u16> = (0..1000u16).collect();
        let seg_elems = 137usize;
        // Gather in a scrambled order so a segment-selection bug can't hide behind sequential access.
        let indices: Vec<u32> = (0..1000u32).map(|i| (i * 613) % 1000).collect();
        let got = seg_gather_on_gpu(&data, seg_elems, &indices);
        let want: Vec<u16> = indices.iter().map(|&i| data[i as usize]).collect();
        assert_eq!(
            got, want,
            "segmented read diverged from contiguous indexing"
        );
    }

    /// Throughput comparison, CPU `admissible_matrices` vs the in-kernel [`enumerate_admissible_kernel`],
    /// for enumerating every `R`'s admissible matrices up to a degree. Reports GPU kernel-only time
    /// (the production-relevant cost — the multiply consumes the scratch on-device, no readback) and
    /// the full-readback time separately. Run with `--nocapture --ignored`; needs a live GPU.
    #[test]
    #[ignore = "benchmark, not a correctness check; run explicitly with --ignored --nocapture"]
    fn bench_admissible_cpu_vs_gpu() {
        use std::time::Instant;

        use fp::prime::ValidPrime;

        let p = ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);
        let max_degree = 130;
        algebra.compute_basis(max_degree);

        // Gather every non-empty R, grouped by degree (per-degree GPU launches keep host arrays bounded).
        // Per degree keep the packed u16 col_sums/masks (what upload-based eviction uploads H->D),
        // so we can time that upload directly and compare it against on-device enumeration. Both are
        // production paths: the matrices end up on-device either way, so NEITHER counts a readback.
        let mut by_degree: Vec<(Vec<Vec<u32>>, Vec<u32>, Vec<u16>, Vec<u16>)> = Vec::new();
        let mut cpu_secs = 0.0f64;
        let mut total_r = 0usize;
        let mut total_mats = 0u64;
        for deg in 1..=max_degree {
            let (mut pps, mut nms, mut cs_all, mut mk_all) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for idx in 0..algebra.dimension(deg) {
                let pp: Vec<u32> = algebra
                    .basis_element_from_index(deg, idx)
                    .p_part
                    .iter()
                    .collect();
                if pp.is_empty() {
                    continue;
                }
                // Time the CPU enumeration (`admissible_matrices`, the call the CPU multiply makes).
                let t = Instant::now();
                let (_cs_len, mk_len, cs, mk) =
                    algebra.admissible_matrices(PPart::try_from_slice(&pp).unwrap());
                cpu_secs += t.elapsed().as_secs_f64();
                let nm = (mk.len() / mk_len) as u32;
                nms.push(nm);
                total_mats += nm as u64;
                cs_all.extend(cs.iter().map(|&v| narrow_u16(v)));
                mk_all.extend(mk.iter().map(|&v| narrow_u16(v)));
                pps.push(pp);
            }
            total_r += pps.len();
            if !pps.is_empty() {
                by_degree.push((pps, nms, cs_all, mk_all));
            }
        }

        let device = CudaDevice::default();
        let client = CudaRuntime::client(&device);
        // Warm up the runtime/JIT so the first degree's compile doesn't skew the GPU timing.
        {
            let (pps, nms, _, _) = &by_degree[0];
            let _ = enum_launch_timed::<CudaRuntime>(&device, pps, nms);
        }
        let mut g_kernel = 0.0f64;
        let mut upload_secs = 0.0f64;
        for (pps, nms, cs_all, mk_all) in &by_degree {
            // (a) On-device enumeration, kernel only (no readback — the multiply consumes the scratch).
            let (_m, k, _r) = enum_launch_timed::<CudaRuntime>(&device, pps, nms);
            g_kernel += k;
            // (b) What upload-based eviction does instead: upload the host-built arrays H->D. Force the
            // (possibly async) copies to complete by syncing the stream via a tiny throwaway readback
            // (4 bytes back, negligible) — NOT by reading the big arrays back, so this is upload-only.
            let t = Instant::now();
            let ch = client.create_from_slice(u16::as_bytes(cs_all));
            let mh = client.create_from_slice(u16::as_bytes(mk_all));
            let sync = client.empty(size_of::<u32>());
            let _ = client.read_one(sync).unwrap();
            upload_secs += t.elapsed().as_secs_f64();
            drop((ch, mh));
        }

        eprintln!(
            "\n=== admissible matrices onto the device: enumerate vs upload (degrees \
             1..={max_degree}) ===\nR's: {total_r}   matrices: {total_mats}   (both paths leave \
             the arrays ON-DEVICE, no readback)\nCPU  admissible_matrices (enumerate, 1 core) : \
             {cpu_secs:.3} s\nGPU  enumerate in-kernel                     : {g_kernel:.3} \
             s\nH->D upload of host-built arrays             : {upload_secs:.3} s\n--> in-kernel \
             enum is {:.2}x the cost of just uploading the same arrays",
            g_kernel / upload_secs,
        );
    }

    /// The flag-based [`enumerate_admissible_ref`] must reproduce `admissible_matrices` bit-for-bit
    /// on every real `R` — this validates the no-break/continue/return restructuring (the tricky
    /// part of the future cubecl in-kernel port) purely on the CPU, where it is fast to debug.
    /// Pure CPU: no GPU needed.
    #[test]
    fn admissible_enum_ref_matches() {
        use fp::prime::ValidPrime;

        let p = ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);
        // To 150: the eviction bench faults on cold (high-degree) R's at internal degree ~144, above
        // the degree-40/60 originally checked — extend the CPU reference to that range.
        let max_degree = 150;
        algebra.compute_basis(max_degree);
        let mut checked = 0usize;
        for deg in 1..=max_degree {
            for idx in 0..algebra.dimension(deg) {
                let p_part: Vec<u32> = algebra
                    .basis_element_from_index(deg, idx)
                    .p_part
                    .iter()
                    .collect();
                if p_part.is_empty() {
                    continue;
                }
                let want = algebra.admissible_matrices(PPart::try_from_slice(&p_part).unwrap());
                let got = enumerate_admissible_ref(&p_part);
                assert_eq!(got, want, "R degree {deg} idx {idx} p_part {p_part:?}");
                checked += 1;
            }
        }
        assert!(checked > 0, "no R's exercised");
        eprintln!("admissible_enum_ref: {checked} R's matched admissible_matrices");
    }

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
                for (slot, v) in row.iter_mut().zip(elt.p_part.iter()) {
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
        let golden = BatchOutput::from_rows(&golden, num_limbs);

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

    /// The stopgap CPU fallback ([`cpu_multiply_batch`]) must produce byte-identical output to the GPU
    /// batch multiply — otherwise a mid-run GPU-context death would silently corrupt the resolution.
    /// Uses TWO generator blocks at distinct `out_offset`s in one wide row, the module-row layout the
    /// single-block `multiply_batch_matches_reference` does not exercise.
    #[test]
    fn cpu_multiply_batch_matches_gpu() {
        use fp::prime::ValidPrime;

        let p = ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);
        let max_degree = 44;
        algebra.compute_basis(max_degree);
        algebra.compute_seqno_tables(max_degree);

        let num_rows = 6;
        // Block A (degree 24) at offset 0, block B (degree 20) immediately after — a two-generator
        // row of width dim(A) + dim(B), so products carry a nonzero `out_offset`.
        let (deg_a, deg_b) = (24, 20);
        let (dim_a, dim_b) = (algebra.dimension(deg_a), algebra.dimension(deg_b));
        let num_cols = dim_a + dim_b;

        let mut products = Vec::new();
        for (out_deg, out_offset) in [(deg_a, 0usize), (deg_b, dim_a)] {
            for r_degree in 1..out_deg {
                let s_degree = out_deg - r_degree;
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
                        out_offset,
                    });
                }
            }
        }

        let gpu = multiply_batch_on_gpu(&algebra, num_cols, num_rows, &products);
        let cpu = cpu_multiply_batch(&algebra, num_cols, num_rows, &products);
        assert_eq!(
            gpu, cpu,
            "cpu_multiply_batch diverged from the GPU batch multiply (out_offset path)"
        );
        eprintln!(
            "cpu_multiply_batch matches GPU: {} products, {num_rows} rows, num_cols={num_cols}",
            products.len()
        );
    }

    /// Drive several batched multiplies at INCREASING output degree so the shared segmented resident
    /// master (see [`SegBuf`], [`seg_grow`]) grows ACROSS launches — each later call appends into the
    /// partially-filled last segment and, at a small `NASSAU_GPU_MASTER_SEG_ELEMS`, allocates fresh
    /// segments. Run with e.g. `NASSAU_GPU_MASTER_SEG_ELEMS=64` to force many segments and exercise
    /// the cross-launch append (the one growth sub-path the single-shot test above does not reach);
    /// with the default large segment it still checks correctness on the single-segment path.
    #[test]
    fn multiply_batch_incremental_growth() {
        use fp::{prime::ValidPrime, vector::FpVector};

        let p = ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);
        let max_degree = 48;
        algebra.compute_basis(max_degree);
        algebra.compute_seqno_tables(max_degree);
        let num_rows = 8;

        // One batched multiply at `out_degree`, checked against the CPU reference. Reused across a
        // sequence of growing degrees; the resident master persists (process-global) and grows
        // monotonically between calls.
        let check = |out_degree: i32| {
            let out_dim = algebra.dimension(out_degree);
            let mut products = Vec::new();
            for r_degree in 1..out_degree {
                let s_degree = out_degree - r_degree;
                let s_dim = algebra.dimension(s_degree);
                if s_dim == 0 {
                    continue;
                }
                for r_idx in 0..algebra.dimension(r_degree) {
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
            let mut cpu_rows: Vec<FpVector> =
                (0..num_rows).map(|_| FpVector::new(p, out_dim)).collect();
            for prod in &products {
                let mut s = FpVector::new(p, algebra.dimension(prod.s_degree));
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
            let golden = BatchOutput::from_rows(&golden, num_limbs);
            let got = multiply_batch_on_gpu(&algebra, out_dim, num_rows, &products);
            assert_eq!(
                got, golden,
                "incremental-growth GPU multiply diverged from reference at \
                 out_degree={out_degree}"
            );
        };

        // Strictly increasing so every call grows the resident master past the previous one.
        for out_degree in [14, 20, 26, 32, 38] {
            check(out_degree);
        }
        eprintln!(
            "multiply_batch_incremental_growth: 5 growing launches matched reference \
             (seg_elems={})",
            master_seg_elems()
        );
    }

    /// Concurrency + growth soak on the GPU Milnor multiply: many worker threads hammer
    /// [`multiply_batch_on_gpu`] against ONE shared resident master while it grows, across many
    /// streams with per-launch `memory_cleanup` on — the same access pattern that trips the cubecl
    /// cross-stream pool-reclaim race (tracel-ai/cubecl#1401) in the stem-200 resolution (one
    /// stream's cleanup reclaiming a pool page another stream's in-flight launch still reads).
    ///
    /// It is BOTH a fast correctness guard and a fast #1401 reproducer:
    /// - **Correctness:** every GPU result is compared against the bit-identical
    ///   [`cpu_multiply_batch`] oracle (up to `verify_max`), catching cross-stream renumber/identity
    ///   races; any mid-soak context death also flips `GPU_DISABLED` and fails the final assert.
    /// - **#1401 (historic):** this used to reproduce the cross-stream pool-reclaim race — at
    ///   `max_degree=160` with `NASSAU_GPU_STREAMS=48` the `never initialized` / `ServerUnhealthy`
    ///   cascade fired within ~45 s at only ~28 GB host / ~22 GB GPU, a low-memory stand-in for the
    ///   40-min stem-200 crash. The dedicated-GPU-thread redesign (see [`gpu_thread`]) deleted the
    ///   multi-stream mode outright: every device section now runs on one thread on stream 0, so
    ///   there are no cross-stream reclaims left to race. This config must now be GREEN, and this
    ///   test is the gate that says so.
    ///
    /// Ignored by default (needs a CUDA device). Run the ex-reproducer config with:
    /// ```text
    /// NASSAU_GPU_CLEANUP_EVERY=1 NASSAU_SOAK_MAX_DEGREE=160 \
    ///   cargo test -p algebra --release --features gpu -- --ignored --nocapture concurrent_growth_soak
    /// ```
    /// Tunables (env): `NASSAU_SOAK_THREADS` (64), `NASSAU_SOAK_SECS` (60), `NASSAU_SOAK_MAX_DEGREE`
    /// (60), `NASSAU_SOAK_VERIFY_MAX` (44, the degree ceiling for the CPU-oracle correctness check).
    #[test]
    #[ignore = "GPU concurrency soak: needs a CUDA device; run explicitly"]
    fn concurrent_growth_soak() {
        use std::{
            sync::{
                Arc,
                atomic::{AtomicU64, Ordering},
            },
            time::{Duration, Instant},
        };

        let env_num = |key: &str, default: u64| -> u64 {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let threads = env_num("NASSAU_SOAK_THREADS", 64) as usize;
        let secs = env_num("NASSAU_SOAK_SECS", 60);
        let max_degree = env_num("NASSAU_SOAK_MAX_DEGREE", 60) as i32;
        // Correctness is checked only up to this degree — the CPU-oracle precompute
        // ([`cpu_multiply_batch`]) explodes past ~degree 48, so keep it modest while letting
        // `max_degree` run far higher to widen the resident master (the race window scales with
        // buffer size). Degrees above the cap still launch on the GPU for the stability/#1401 axis;
        // their output just isn't compared.
        let verify_max = env_num("NASSAU_SOAK_VERIFY_MAX", 44) as i32;
        let num_rows = 32usize;

        let p = fp::prime::ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);
        algebra.compute_basis(max_degree);
        algebra.compute_seqno_tables(max_degree);

        // One `get_partial_matrix`-shaped batch per output degree: every non-empty R of degree
        // 1..out_degree times a dense complementary element, round-robin across rows, single block.
        // Mirrors `multiply_batch_incremental_growth` / the throughput bench, so it hits the exact
        // kernel + resident-master path Nassau drives.
        let build_batch = |out_degree: i32| -> (usize, Vec<GpuProduct>) {
            let num_cols = algebra.dimension(out_degree);
            let mut products = Vec::new();
            for r_degree in 1..out_degree {
                let s_degree = out_degree - r_degree;
                let s_dim = algebra.dimension(s_degree);
                if s_dim == 0 {
                    continue;
                }
                for r_idx in 0..algebra.dimension(r_degree) {
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
            (num_cols, products)
        };

        // Ascending degrees so the sweep drives resident-master growth; precompute each batch and
        // its CPU golden once (shared, read-only) so worker threads only launch + compare.
        struct Job {
            num_cols: usize,
            products: Vec<GpuProduct>,
            golden: Option<BatchOutput>,
        }
        let jobs: Arc<Vec<Job>> = Arc::new(
            (12..=max_degree)
                .step_by(2)
                .filter_map(|d| {
                    let (num_cols, products) = build_batch(d);
                    if products.is_empty() {
                        return None;
                    }
                    let golden = (d <= verify_max)
                        .then(|| cpu_multiply_batch(&algebra, num_cols, num_rows, &products));
                    Some(Job {
                        num_cols,
                        products,
                        golden,
                    })
                })
                .collect(),
        );
        assert!(
            !jobs.is_empty(),
            "no non-empty batches built up to degree {max_degree}"
        );

        let launches = AtomicU64::new(0);
        let mismatches = AtomicU64::new(0);
        let started = Instant::now();
        let deadline = started + Duration::from_secs(secs);

        std::thread::scope(|scope| {
            for t in 0..threads {
                let jobs = Arc::clone(&jobs);
                let algebra = &algebra;
                let launches = &launches;
                let mismatches = &mismatches;
                scope.spawn(move || {
                    // Desynchronize threads across the degree sweep so some GROW the master (first
                    // touch of a high degree) while others READ lower resident pages + cleanup.
                    let mut i = t % jobs.len();
                    while Instant::now() < deadline && !gpu_disabled() {
                        let job = &jobs[i];
                        let got =
                            multiply_batch_on_gpu(algebra, job.num_cols, num_rows, &job.products);
                        launches.fetch_add(1, Ordering::Relaxed);
                        // A mismatch while the GPU is still enabled is a real concurrency bug (a
                        // cross-stream renumber/identity race). Once disabled, results come from the
                        // bit-identical CPU oracle, so they still match — no false alarm. Degrees
                        // above `verify_max` have no golden and only exercise stability.
                        if let Some(golden) = &job.golden {
                            if got != *golden && !gpu_disabled() {
                                mismatches.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        i = (i + 1) % jobs.len();
                    }
                });
            }
        });

        let elapsed = started.elapsed();
        let n = launches.load(Ordering::Relaxed);
        let mm = mismatches.load(Ordering::Relaxed);
        eprintln!(
            "[soak] {threads} threads × {secs}s, 1 gpu thread: {n} launches ({:.0}/s over {} \
             degrees, verified ≤{verify_max}), {mm} correctness mismatches, gpu_disabled={}",
            n as f64 / elapsed.as_secs_f64().max(1e-3),
            jobs.len(),
            gpu_disabled(),
        );
        assert_eq!(
            mm, 0,
            "GPU multiply diverged from the CPU oracle under concurrency (renumber/identity race)"
        );
        assert!(
            !gpu_disabled(),
            "cubecl GPU multiply was disabled mid-soak — the cross-stream pool-reclaim race \
             (tracel-ai/cubecl#1401) fired. This is the crash the submission-thread redesign \
             closes."
        );
    }

    /// Benchmark of the **hard stem-200 regime**: the GPU submission path under the contention
    /// shape a record-stem Nassau resolution actually produces.
    ///
    /// # Why this exists
    ///
    /// Every change to the GPU path was previously validated by a ~3 h stem-200 resolution, so the
    /// iteration loop was measured in hours and each answer arrived with run-to-run variance mixed
    /// in. This reproduces the regime in minutes.
    ///
    /// # Calibration
    ///
    /// The shape below is not invented; it is the measured distribution of `gpu_submit` spans from
    /// a complete stem-200 run, restricted to the hard tail (stems ≥ 190, n = 11 287 launches):
    ///
    /// ```text
    ///            p10        p50        p90         max
    /// rows       122        158        260      83 702
    /// pairs   190 962  1 963 056 14 770 242 632 386 884
    /// out_u32 326 106    379 516    665 860  42 269 510
    /// ```
    ///
    /// Two properties matter as much as the sizes:
    /// - **Worker count ≈ 7**, the real wavefront (time-weighted mean 6.6, and 87 % of the run sits
    ///   at 6–7 bidegrees in flight). The soak's 64 threads are deliberately *wrong* here: queue
    ///   contention is the thing under test, so the number of contenders must match the resolution.
    /// - **Steady state, not growth.** The timed phase runs after a warm-up sweep has grown the
    ///   resident master, because at stem 200 the master is long since built; timing the growth
    ///   transient would measure a phase the hard regime is not in.
    ///
    /// `num_cols` is chosen by searching for the output degree whose dimension is closest to
    /// `NASSAU_BENCH_COLS`, rather than hard-coding a degree — the degree that yields a given
    /// matrix width is an artifact of the algebra, and pinning the *width* is what keeps this
    /// comparable to the measured run.
    ///
    /// Reports launches/s, pairs/s and the prep / queue / exec split with queue depth, plus the
    /// achieved workload distribution so drift from the calibration above is visible rather than
    /// silent.
    ///
    /// Ignored by default (needs a CUDA device). Run with:
    /// ```text
    /// cargo test -p algebra --release --features gpu -- --ignored --nocapture stem200_regime_bench
    /// ```
    /// Tunables (env): `NASSAU_BENCH_WORKERS` (7), `NASSAU_BENCH_ROWS` (158),
    /// `NASSAU_BENCH_COLS` (77 000), `NASSAU_BENCH_SECS` (60), `NASSAU_BENCH_SPREAD` (4).
    #[test]
    #[ignore = "GPU perf bench: needs a CUDA device; run explicitly"]
    fn stem200_regime_bench() {
        use std::{
            sync::{
                Arc, Mutex,
                atomic::{AtomicU64, Ordering},
            },
            time::{Duration, Instant},
        };

        let env_num = |key: &str, default: u64| -> u64 {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let workers = env_num("NASSAU_BENCH_WORKERS", 7) as usize;
        let num_rows = env_num("NASSAU_BENCH_ROWS", 158) as usize;
        let target_cols = env_num("NASSAU_BENCH_COLS", 77_000) as usize;
        let secs = env_num("NASSAU_BENCH_SECS", 60);
        // How many neighbouring degrees to sweep. A single degree would let every launch reuse one
        // resident-master prefix; the real run interleaves several bidegrees at once.
        let spread = env_num("NASSAU_BENCH_SPREAD", 4) as i32;

        let p = fp::prime::ValidPrime::new(2);
        let algebra = MilnorAlgebra::new(p, false);

        // Grow the basis until it brackets `target_cols`, then take the closest degree. Doubling
        // the probe keeps this from computing a far larger basis than the bench needs.
        let mut probe = 32;
        loop {
            algebra.compute_basis(probe);
            if algebra.dimension(probe) >= target_cols || probe > 512 {
                break;
            }
            probe *= 2;
        }
        let out_degree = (1..=probe)
            .min_by_key(|&d| algebra.dimension(d).abs_diff(target_cols))
            .expect("non-empty degree range");
        let max_degree = out_degree;
        algebra.compute_basis(max_degree);
        algebra.compute_seqno_tables(max_degree);
        eprintln!(
            "[bench] target_cols={target_cols} -> out_degree={out_degree} (num_cols={}), \
             workers={workers} rows={num_rows} spread={spread} secs={secs}",
            algebra.dimension(out_degree),
        );

        // Same `get_partial_matrix`-shaped batch the soak builds, but with the product count
        // BOUNDED.
        //
        // Taking every `(R, s)` pair the way the soak does is fine to degree ~160 and impossible
        // above it: the batch holds `sum_r dim(r) * dim(out_degree - r)` term indices, which at the
        // degree that yields stem-200-scale matrices is astronomically large — the first version of
        // this bench died in setup there, never reaching a launch. Real batches are bounded too
        // (measured p50 ~24k products, max ~479k), so sampling `R`s on a stride reproduces the
        // regime while the exhaustive build merely runs out of memory.
        let max_products = env_num("NASSAU_BENCH_PRODUCTS", 24_000) as usize;
        // Terms per product. Real products carry the nonzeros of a sparse vector, not a whole
        // basis: run I measured `products=478972 terms=2300621`, i.e. ~4.8 terms each. Using the
        // full s-degree basis (as the low-degree soak does) makes every product hundreds of
        // thousands of terms here, which blows past the kernel's u32 pair limit before it can run.
        let terms_per_product = env_num("NASSAU_BENCH_TERMS", 5) as usize;
        let build_batch = |out_degree: i32| -> (usize, Vec<GpuProduct>) {
            let num_cols = algebra.dimension(out_degree);
            // Count the candidate `R`s first so the stride spreads the sample over the whole
            // degree range instead of truncating at low `r_degree` (which would bias every launch
            // toward small, cheap operations).
            let mut candidates: Vec<(i32, usize, i32)> = Vec::new();
            for r_degree in 1..out_degree {
                let s_degree = out_degree - r_degree;
                if algebra.dimension(s_degree) == 0 {
                    continue;
                }
                for r_idx in 0..algebra.dimension(r_degree) {
                    if algebra
                        .basis_element_from_index(r_degree, r_idx)
                        .p_part
                        .is_empty()
                    {
                        continue;
                    }
                    candidates.push((r_degree, r_idx, s_degree));
                }
            }
            let stride = candidates.len().div_ceil(max_products.max(1)).max(1);
            let mut products = Vec::new();
            for (r_degree, r_idx, s_degree) in candidates.into_iter().step_by(stride) {
                let s_dim = algebra.dimension(s_degree);
                let nt = terms_per_product.min(s_dim);
                let t_stride = s_dim.div_ceil(nt.max(1)).max(1);
                let row = products.len() % num_rows;
                products.push(GpuProduct {
                    r_degree,
                    r_idx,
                    s_degree,
                    term_indices: (0..s_dim).step_by(t_stride).take(nt).collect(),
                    row,
                    out_offset: 0,
                });
            }
            (num_cols, products)
        };

        struct Job {
            num_cols: usize,
            products: Vec<GpuProduct>,
        }
        let jobs: Arc<Vec<Job>> = Arc::new(
            (out_degree - spread + 1..=out_degree)
                .filter(|&d| d > 1)
                .filter_map(|d| {
                    let (num_cols, products) = build_batch(d);
                    (!products.is_empty()).then_some(Job { num_cols, products })
                })
                .collect(),
        );
        assert!(
            !jobs.is_empty(),
            "no non-empty batches at degree {out_degree}"
        );

        // Warm-up: one pass per job grows the resident master to its steady-state extent, so the
        // timed phase below measures the regime rather than the growth transient.
        let warm = Instant::now();
        for job in jobs.iter() {
            let _ = multiply_batch_on_gpu(&algebra, job.num_cols, num_rows, &job.products);
        }
        eprintln!(
            "[bench] warm-up: {} jobs in {:.1}s",
            jobs.len(),
            warm.elapsed().as_secs_f64()
        );
        assert!(!gpu_disabled(), "GPU died during warm-up");

        // Discard warm-up from the counters; the timed phase starts from zero.
        let _ = take_batch_stats();
        let _ = take_gpu_timing();

        let launches = AtomicU64::new(0);
        // Per-launch wall time: the tail is the starvation signal the aggregate mean hides.
        let waits: Mutex<Vec<f64>> = Mutex::new(Vec::new());
        let started = Instant::now();
        let deadline = started + Duration::from_secs(secs);

        std::thread::scope(|scope| {
            for t in 0..workers {
                let jobs = Arc::clone(&jobs);
                let algebra = &algebra;
                let launches = &launches;
                let waits = &waits;
                scope.spawn(move || {
                    let mut local = Vec::new();
                    let mut i = t % jobs.len();
                    while Instant::now() < deadline && !gpu_disabled() {
                        let job = &jobs[i];
                        let t0 = Instant::now();
                        let _ =
                            multiply_batch_on_gpu(algebra, job.num_cols, num_rows, &job.products);
                        local.push(t0.elapsed().as_secs_f64());
                        launches.fetch_add(1, Ordering::Relaxed);
                        i = (i + 1) % jobs.len();
                    }
                    waits.lock().unwrap().extend(local);
                });
            }
        });

        let elapsed = started.elapsed().as_secs_f64().max(1e-3);
        let n = launches.load(Ordering::Relaxed);
        let (calls, _marshal_us, device_us, pairs) = take_batch_stats();
        let (prep_us, queue_us, exec_us, depth_sum, depth_max) = take_gpu_timing();
        let us = |v: u64| v as f64 / 1e6;
        let total = (us(prep_us) + us(device_us)).max(1e-9);

        let mut w = waits.into_inner().unwrap();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |pc: usize| w[(w.len() * pc / 100).min(w.len().saturating_sub(1))];

        eprintln!(
            "[bench] {n} calls ({calls} blocks) in {elapsed:.1}s: {:.1} calls/s, {:.2e} pairs/s",
            n as f64 / elapsed,
            pairs as f64 / elapsed,
        );
        eprintln!(
            "[bench] prep={:.1}s queue={:.1}s exec={:.1}s | prep={:.0}% queue={:.0}% exec={:.0}% \
             | depth mean={:.1} max={depth_max}",
            us(prep_us),
            us(queue_us),
            us(exec_us),
            100.0 * us(prep_us) / total,
            100.0 * us(queue_us) / total,
            100.0 * us(exec_us) / total,
            depth_sum as f64 / calls.max(1) as f64,
        );
        eprintln!(
            "[bench] per-call wall: p50={:.3}s p90={:.3}s p99={:.3}s max={:.3}s (ratio \
             max/p50={:.0}x)",
            q(50),
            q(90),
            q(99),
            w[w.len() - 1],
            w[w.len() - 1] / q(50).max(1e-9),
        );
        eprintln!(
            "[bench] gpu-thread duty cycle: {:.0}% ({:.1}s exec of {elapsed:.1}s wall)",
            100.0 * us(exec_us) / elapsed,
            us(exec_us),
        );

        assert!(!gpu_disabled(), "GPU context died during the bench");
        assert!(n > 0, "no launches completed");
    }
}
