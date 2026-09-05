//! GPU dispatch for F₂ matrix multiplication (Hopper `wgmma.b1`).
//!
//! Compiled only under the `gpu` feature. [`try_mul`] is consulted by
//! `<&Matrix as Mul>::mul` before the CPU BLAS path: for large enough `p = 2`
//! products it converts the operands to the raw row-major limb layout
//! `fp-cuda` expects, runs the kernel, and rebuilds a [`Matrix`]. Anything that
//! makes the GPU path unavailable or unsuitable — no device, a launch error, or
//! a below-threshold size — returns `None`, and the caller falls back to the
//! (bit-identical) CPU kernel.
//!
//! Tuning knobs (environment variables, read once):
//! - `FP_CUDA_DISABLE` — set to any value to force the CPU path.
//! - `FP_CUDA_THRESHOLD` — minimum of `m`, `k`, `n` (in bits) below which the
//!   CPU path is used. Defaults to 2048; the GPU only wins once the kernel work
//!   dwarfs the H2D/D2H + TMA-layout marshalling, which dominates small sizes.

use std::sync::OnceLock;

use fp_cuda::GpuContext;

use crate::{matrix::Matrix, prime::TWO};

/// Smallest `min(m, k, n)` for which we attempt the GPU matmul. Below this the
/// host marshalling (bit-repack into TMA tiles + copies) costs more than it saves.
const DEFAULT_THRESHOLD: usize = 2048;

/// Smallest `min(rows, cols)` for which we attempt the GPU row reduction. Higher
/// than the matmul threshold: a full reduction is many dependent panel steps, not
/// one GEMM, so its CPU crossover is later. Re-validated on an H200 post-
/// optimization (half-rank square, device incl. upload/reduce vs M4RI
/// `row_reduce`): GPU is 0.57× at n=4096 (a loss) and 1.57× at n=8192 (a win),
/// so the crossover sits just below 8192. The small-n crossover is bound by fixed
/// launch/transfer overhead, not the trailing GEMM, so the recent throughput wins
/// (which scale with n²) did not move it. Measured against single-thread M4RI;
/// the concurrent CPU path is faster, which only pushes the crossover up — so
/// 8192 is the safe floor. Override with `FP_CUDA_RR_THRESHOLD`.
const DEFAULT_RR_THRESHOLD: usize = 8192;

fn threshold() -> usize {
    std::env::var("FP_CUDA_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_THRESHOLD)
}

pub(crate) fn rr_threshold() -> usize {
    std::env::var("FP_CUDA_RR_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_RR_THRESHOLD)
}

/// The process-wide GPU context, created lazily on first use. `None` if no
/// usable device is present (no driver, no Hopper GPU, or the kernel PTX is the
/// nvcc-absent build stub). Shared as `&'static` — no lock: `GpuContext` is
/// `Send + Sync` (its cudarc handles are), and every submission goes through a
/// per-thread stream ([`GpuContext::stream`]), so concurrent rayon workers run
/// on independent streams (overlapping transfers + kernels) instead of
/// serializing on one mutex. Buffers are per-call and thread-local, so there is
/// no shared mutable device state to guard.
fn context() -> Option<&'static GpuContext> {
    static GPU: OnceLock<Option<GpuContext>> = OnceLock::new();
    GPU.get_or_init(|| {
        if std::env::var_os("FP_CUDA_DISABLE").is_some() {
            return None;
        }
        // `FP_CUDA_DEVICE` puts the row reduction on its own GPU. On a single device the two CUDA
        // consumers contend: the reduction's thousands of tiny sequential relaunches queue behind
        // the multiply's saturating kernels (1.8-9.7 ms standalone vs 8.6-96.8 s co-running), which
        // is why [`crate::gpu_lock`] exists at all — and that arbitration then costs ~47% of
        // multiply time. Separate devices remove the contention by construction, so the lock
        // becomes a no-op (see [`crate::gpu_lock::set_devices_shared`]).
        let device = std::env::var("FP_CUDA_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mult_devices = multiply_devices();
        let shared = device < mult_devices;
        crate::gpu_lock::set_devices_shared(shared);
        // Log it: whether arbitration is live decides whether the reduction's thousands of tiny
        // launches overlap the multiply's saturating ones, and getting it wrong is invisible in a
        // normal run until something fails far away. `[batch-stats] lock=` alone cannot distinguish
        // "arbitration off" from "arbitration on but uncontended".
        eprintln!(
            "[fp-cuda] row reduction on device {device}; multiply spans devices \
             0..{mult_devices}; multiply yields to reductions: {}; reductions serialize against \
             each other: always",
            if shared {
                "yes"
            } else {
                "no (separate devices)"
            }
        );
        GpuContext::new(device).ok()
    })
    .as_ref()
}

/// How many GPUs the cubecl Milnor multiply spreads over — it shards across ALL visible devices, so
/// the row reduction shares a device with it whenever `FP_CUDA_DEVICE < multiply_devices()`.
///
/// This used to ask which single device the multiply ran on, reading `NASSAU_GPU_DEVICE` — a
/// variable nothing in `algebra` reads any more, left behind when the multiply became multi-GPU. It
/// therefore answered "device 0" no matter how many GPUs the multiply was actually saturating, and
/// `FP_CUDA_DEVICE=2` on a 4-GPU node would silently conclude "separate devices, no arbitration
/// needed" while the multiply was hammering device 2 as well. Arbitration exists to keep the
/// reduction's thousands of tiny sequential relaunches from queueing behind saturating multiply
/// kernels (1.8-9.7 ms standalone vs 8.6-96.8 s co-running); losing it is not a small regression.
///
/// Mirrors `algebra::algebra::milnor_gpu::gpu_count` — `fp` cannot call it (`algebra` depends on
/// `fp`, not the reverse), so the two must be kept in step. Both honour `CUDA_VISIBLE_DEVICES`,
/// since CUDA renumbers the visible subset to `0..n`.
fn multiply_devices() -> usize {
    const MAX_GPUS: usize = 8;
    let physical = std::fs::read_dir("/proc/driver/nvidia/gpus")
        .map(|d| d.filter_map(|e| e.ok()).count())
        .unwrap_or(0)
        .max(1);
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok().map(|v| {
        v.split(',')
            .take_while(|e| {
                e.trim()
                    .parse::<usize>()
                    .is_ok_and(|ord| ord < physical.max(MAX_GPUS))
            })
            .count()
    });
    std::env::var("NASSAU_GPU_DEVICES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| visible.unwrap_or(physical).max(1))
        .clamp(1, MAX_GPUS)
}

/// The single thread every `fp-cuda` submission goes through, so this process has exactly one
/// owner of the reduction GPU.
///
/// # Why a thread and not a lock
///
/// Both `fp-cuda` entry points launch **persistent whole-device grids** (`num_ctas = occupancy x
/// SMs`, cluster-aligned): the row reduction's trailing GEMM and the standalone [`try_mul`]. Two of
/// those cannot be placed at once, and on Hopper the loser does not queue — it fails with a bare
/// `CUDA_ERROR_LAUNCH_FAILED` that compute-sanitizer cannot attribute (0 invalid accesses across a
/// whole run: it was never a memory bug). The cooperative reduction path fails worse, spinning
/// forever at a grid-wide barrier for CTAs that were never scheduled.
///
/// A lock could serialize this, and [`crate::gpu_lock::exclusive`] did for `row_reduce` — but
/// `try_mul` was deliberately lock-free, so the device still had two independent whole-device
/// consumers and a dedicated GPU was not actually owned by anything. Routing *both* through one
/// thread makes single-ownership structural rather than a discipline every new call site has to
/// remember.
///
/// # Completion, not submission
///
/// The job runs to completion on this thread, and both jobs end in a device-to-host download, which
/// synchronizes. That is the property that matters: serializing *submission* is not enough, because
/// kernels outlive the call that launched them — the mistake `gpu_lock::shared` still makes on the
/// multiply side (it is taken inside the submit closure and dropped when submission returns, which
/// is why `[batch-stats] lock=` reads 0.0s in every run).
///
/// # Not yet done
///
/// Transfers are serialized along with compute. They need not be: copy engines do not consume SMs,
/// so the next job's H2D upload could overlap the current job's kernels without touching
/// co-residency. That requires splitting each job into upload / compute / download stages on
/// separate streams and pipelining them here; the correctness property above does not depend on it.
/// # Widening the pool: `FP_CUDA_DRIVER_THREADS`
///
/// The single-owner argument above rests on whole-device persistent grids that cannot be
/// co-resident. That hazard belongs to the launch paths that are opt-in and off by default:
/// [`fp_cuda::rr_coop`] is false, so reduction kernels synchronise at kernel boundaries rather than
/// launching cooperatively, and the GEMM's cluster path is likewise off. A probe of 192 concurrent
/// `row_reduce_dev` calls across K = 1/2/4/8 on an idle H200 produced zero launch failures and zero
/// wrong ranks.
///
/// Whether concurrency WINS is regime-dependent and unsettled. On an idle device K=4 measured 1.50x
/// over K=1, because a reduction is a chain of ~33k tiny dependent launches and the device drains
/// between them, so concurrent reductions fill each other's gaps. In a full resolution the Milnor
/// multiply shares the device and may already fill those gaps: an end-to-end A/B at `max_n=300`
/// found progress flat and reduce throughput ~10% worse at K=4. Re-measure in the target regime
/// before changing the default.
///
/// Defaults to 1, exactly the historical single-owner behaviour, so this is inert unless asked for.
/// All threads share the one [`GpuContext`], which is `Send + Sync` and hands each thread its own
/// stream. Sharding across DEVICES is a separate change needing one context per device.
mod driver {
    use std::sync::{Arc, Mutex, OnceLock, mpsc};

    type Job = Box<dyn FnOnce() + Send + 'static>;

    /// How many jobs may be in flight on the reduction device at once; 1 is the historical
    /// single-owner behaviour.
    pub(super) fn threads() -> usize {
        static N: OnceLock<usize> = OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("FP_CUDA_DRIVER_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(1)
                .clamp(1, 32)
        })
    }

    fn sender() -> &'static Mutex<mpsc::Sender<Job>> {
        static TX: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
        TX.get_or_init(|| {
            let (tx, rx) = mpsc::channel::<Job>();
            let n = threads();
            if n > 1 {
                // One shared receiver, so a long job never blocks a short one behind it the way
                // per-worker queues would. The mutex is held only across `recv`, never across
                // `job()` -- holding it while running would serialize the pool back down to one.
                let rx = Arc::new(Mutex::new(rx));
                for i in 0..n {
                    let rx = Arc::clone(&rx);
                    std::thread::Builder::new()
                        .name(format!("fp-cuda-driver-{i}"))
                        .spawn(move || {
                            loop {
                                let job = {
                                    let guard = rx.lock().unwrap_or_else(|e| e.into_inner());
                                    guard.recv()
                                };
                                match job {
                                    Ok(job) => job(),
                                    Err(_) => break, // all senders dropped; shutting down
                                }
                            }
                        })
                        .expect("failed to spawn an fp-cuda driver thread");
                }
                eprintln!("[fp-cuda] driver pool: {n} threads (concurrent reductions enabled)");
                return Mutex::new(tx);
            }
            std::thread::Builder::new()
                .name("fp-cuda-driver".into())
                .spawn(move || {
                    for job in rx {
                        // NO `gpu_lock::exclusive()` here. Serialization among fp-cuda jobs is
                        // already structural — this is the only thread that submits them — so the
                        // guard would be redundant, and taking it deadlocked the run: it waits for
                        // the multiply's readers to drain while worker threads block on `run`
                        // waiting for this loop. Yielding to the multiply on a SHARED device has to
                        // be arranged without a guard held across a blocking job.
                        job();
                    }
                })
                .expect("failed to spawn the fp-cuda driver thread");
            Mutex::new(tx)
        })
    }

    /// Run `f` on the driver thread and block for its result. `f` owns everything it touches (both
    /// call sites have already marshalled to owned limb buffers), so nothing borrows across threads.
    pub(super) fn run<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = mpsc::channel();
        sender()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send(Box::new(move || {
                // A send failure means the caller gave up; the job still ran, so just drop it.
                let _ = tx.send(f());
            }))
            .expect("the fp-cuda driver thread died");
        rx.recv().expect("the fp-cuda driver thread dropped a job")
    }
}

/// Row-major, K-major `u64` limbs — the exact layout `fp_cuda::matmul_b1_raw`
/// expects (`rows × columns.div_ceil(64)` limbs, no inter-row padding). Uses
/// `Matrix::to_bytes`, which already strips the physical row stride.
/// A bounded pool of reusable host buffers for the row-reduce marshalling.
///
/// A heap profile of a stem-320 run found the host-side plumbing around the GPU reduction to be the
/// single largest consumer in the process: 201-222 GiB live across ~190 objects, in
/// `Matrix::from_data` under `try_row_reduce`. Two things caused it. Each reduction allocated four
/// full-size host copies (`to_bytes` -> `Vec<u8>`, collect -> `Vec<u64>`, `download` -> `Vec<u64>`,
/// and `out` -> `Matrix::from_data`), and the FIRST of those was made on the calling thread BEFORE
/// `driver::run` queued the job -- so every thread waiting behind the serialized driver was holding
/// a complete copy of its matrix. Live memory scaled with queue depth, not with concurrency on the
/// device.
///
/// The copy cannot simply move inside the closure: `driver::run` requires `'static`, so it cannot
/// borrow the matrix. Instead the buffers are OWNED, taken from this fixed-size pool, moved into the
/// closure, and handed back out again. At most `capacity()` of them exist regardless of how many
/// threads are queued.
///
/// `acquire` falls back to allocating after a timeout rather than blocking forever. A panic inside
/// the driver closure would drop a buffer without returning it, and permanently losing a permit
/// would deadlock every later reduction -- degrading to an extra allocation is much the lesser
/// failure.
mod marshal {
    use std::{
        sync::{Condvar, LazyLock, Mutex},
        time::Duration,
    };

    struct Pool {
        free: Vec<Vec<u64>>,
        checked_out: usize,
    }

    static POOL: LazyLock<(Mutex<Pool>, Condvar)> = LazyLock::new(|| {
        (
            Mutex::new(Pool {
                free: Vec::new(),
                checked_out: 0,
            }),
            Condvar::new(),
        )
    });

    /// How many marshalling buffers may exist at once (`FP_CUDA_MARSHAL_BUFFERS`).
    ///
    /// Two per driver thread by default: one being filled while another is in flight, which keeps
    /// the device fed without letting the queue multiply memory.
    fn capacity() -> usize {
        static CAP: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("FP_CUDA_MARSHAL_BUFFERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| (2 * super::driver::threads()).max(2))
        });
        *CAP
    }

    pub(super) fn acquire() -> Vec<u64> {
        let (lock, cv) = &*POOL;
        let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = Duration::from_secs(30);
        loop {
            if let Some(b) = g.free.pop() {
                g.checked_out += 1;
                return b;
            }
            if g.checked_out < capacity() {
                g.checked_out += 1;
                return Vec::new();
            }
            let (ng, timeout) = cv
                .wait_timeout(g, deadline)
                .unwrap_or_else(|e| e.into_inner());
            g = ng;
            if timeout.timed_out() {
                // A permit was lost (a panicked closure never returned its buffer). Allocate rather
                // than hang; the pool self-heals as live buffers come back.
                g.checked_out += 1;
                return Vec::new();
            }
        }
    }

    pub(super) fn release(mut b: Vec<u64>) {
        let (lock, cv) = &*POOL;
        let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
        g.checked_out = g.checked_out.saturating_sub(1);
        if g.free.len() < capacity() {
            b.clear();
            g.free.push(b);
        }
        cv.notify_one();
    }
}

/// Pack `m`'s rows into `buf` in the tight row-major layout `upload` expects.
///
/// Replaces a `u64 -> little-endian bytes -> u64` round trip, which was two full-size copies to
/// perform the identity on a little-endian machine. When the matrix's own stride already matches
/// the packed one -- the common case, since `Matrix::new` sets `columns_capacity == columns` -- this
/// is a single `extend_from_slice` of the whole buffer.
fn fill_limbs(m: &Matrix, buf: &mut Vec<u64>) {
    let packed = m.columns().div_ceil(64);
    let ms = m.stride();
    let rows = m.rows();
    buf.clear();
    buf.reserve(rows * packed);
    let data = m.data();
    if ms == packed {
        buf.extend_from_slice(&data[..rows * packed]);
    } else {
        for i in 0..rows {
            buf.extend_from_slice(&data[i * ms..i * ms + packed]);
        }
    }
}

fn to_limbs(m: &Matrix) -> Vec<u64> {
    let stride = m.columns().div_ceil(64);
    let mut bytes = Vec::with_capacity(m.rows() * stride * 8);
    m.to_bytes(&mut bytes).expect("Vec writes never fail");
    let (chunks, _) = bytes.as_chunks::<8>();
    chunks.iter().map(|&c| u64::from_le_bytes(c)).collect()
}

/// Try to compute `a · b` on the GPU. Returns `None` (and the caller uses the
/// CPU path) if the GPU is unavailable, the product is below the size
/// threshold, or the launch fails. The result is bit-identical to the CPU path.
///
/// Assumes `a.prime() == b.prime() == 2` and `a.columns() == b.rows()` — the
/// same preconditions the caller has already checked.
pub(super) fn try_mul(a: &Matrix, b: &Matrix) -> Option<Matrix> {
    debug_assert_eq!(a.prime(), TWO);
    debug_assert_eq!(b.prime(), TWO);
    debug_assert_eq!(a.columns(), b.rows());

    let (m, k, n) = (a.rows(), a.columns(), b.columns());
    let t = threshold();
    if m < t || k < t || n < t {
        return None;
    }

    let ctx = context()?;
    let a_limbs = to_limbs(a);
    let b_limbs = to_limbs(b);

    // Through the driver: this is a persistent whole-device grid, so "concurrent callers do not
    // interfere" was wrong — two at once cannot both be placed (see [`driver`]).
    // `.ok()` inside the closure: the error is a `Box<dyn Error>`, which is not `Send`, so it
    // cannot cross back from the driver thread. The caller only distinguishes success from
    // fall-back-to-CPU anyway.
    let c = driver::run(move || fp_cuda::matmul_b1_raw(ctx, &a_limbs, m, k, &b_limbs, n).ok())?;
    Some(Matrix::from_data(TWO, m, n, c))
}

/// Try to row-reduce `m` to RREF on the GPU, in place. Returns `Some(rank)` and
/// leaves `m` in the same canonical reduced form `Matrix::row_reduce` produces
/// (pivot rows at the top in column order, zeros below, `pivots` set); returns
/// `None` — and the caller uses the CPU M4RI path — if the GPU is unavailable,
/// below threshold, or a launch fails. The result is bit-identical to the CPU
/// path (validated in `fp-cuda`'s `row_reduce_demo`).
///
/// Assumes `m.prime() == 2` (the caller has checked).
pub(crate) fn try_row_reduce(m: &mut Matrix) -> Option<usize> {
    debug_assert_eq!(m.prime(), TWO);
    let (rows, cols) = (m.rows(), m.columns());
    let t = rr_threshold();
    if rows < t || cols < t {
        return None;
    }
    let ctx = context()?;

    let stride = cols.div_ceil(64);

    // Lock-free, per-thread stream (see [`context`]): the default row-reduce is composable (no
    // cooperative launch) and allocates its device buffers per call, so concurrent rayon workers
    // reduce different matrices on independent streams — overlapping instead of serializing.
    //
    // The claim that this "needs no cross-runtime exclusion against the cubecl multiply" is exactly
    // backwards. Composability (no cooperative launch) means this path *can* overlap other GPU work
    // without deadlocking — not that it should. This reduction is a chain of thousands of tiny
    // sequential per-column relaunches, so overlapping it with the multiply's saturating kernels
    // makes every launch queue: 1.8–9.7 ms standalone becomes 8.6–96.8 s co-running. Take the
    // device exclusively for the duration; see [`fp::gpu_lock`] for the measurements and the cost
    // (~5 s of multiply pause across a whole stem-200 resolution).
    // The exclusive guard now lives on the driver thread, which holds it for the whole job — see
    // [`driver`]. Taking it here as well would deadlock: the driver would wait on a guard this
    // thread holds while this thread waits on the driver.
    // Both buffers come from the bounded pool and are handed back by the closure whether or not the
    // reduction succeeded -- dropping one inside would lose a permit permanently (see `marshal`).
    let mut in_buf = marshal::acquire();
    fill_limbs(m, &mut in_buf);
    let mut out_buf = marshal::acquire();
    out_buf.clear();
    out_buf.resize(rows * stride, 0);

    let (in_buf, mut out_buf, res) = driver::run(move || {
        let mut outcome = None;
        if let Ok(mut dm) = ctx.upload(&in_buf, rows, cols)
            && let Ok((perm_dev, r, pivot_cols)) = ctx.row_reduce_dev(&mut dm)
            && ctx.download_into(&dm, &mut out_buf).is_ok()
            && let Ok(perm) = ctx.download_u32(&perm_dev)
        {
            outcome = Some((perm, r, pivot_cols));
        }
        (in_buf, out_buf, outcome)
    });
    marshal::release(in_buf);
    let Some((perm, r, pivot_cols)) = res else {
        marshal::release(out_buf);
        return None;
    };

    // Materialize the canonical RREF IN PLACE: pivot k (column pivot_cols[k], ascending) at row k,
    // taken from device row perm[k]; rows [r, rows) zero. Writing into `m`'s existing storage avoids
    // a fourth full-size allocation, and it also preserves `m`'s row/column CAPACITY, which
    // `Matrix::from_data` silently discarded -- callers such as `extend_image` then `add_row` into
    // it.
    let ms = m.stride();
    {
        let data = m.data_mut();
        data.fill(0);
        for k in 0..r {
            let src = perm[k] as usize * stride;
            data[k * ms..k * ms + stride].copy_from_slice(&out_buf[src..src + stride]);
        }
    }
    marshal::release(out_buf);
    m.initialize_pivots();
    let piv = m.pivots_mut();
    for (k, &q) in pivot_cols.iter().enumerate() {
        piv[q] = k as isize;
    }
    Some(r)
}
