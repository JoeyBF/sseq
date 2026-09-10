//! GPU dispatch for F₂ matrix multiplication (Hopper `wgmma.b1`).

use std::sync::OnceLock;

use fp_cuda::GpuContext;

use crate::{matrix::Matrix, prime::TWO};

/// Smallest `min(m, k, n)` for which we attempt the GPU.
///
/// Below this the host marshalling (bit-repack into TMA tiles + copies) costs more than it saves.
const DEFAULT_THRESHOLD: usize = 2048;

/// Smallest problem size, in bits, for which we attempt the GPU row reduction.
///
/// Size rather than a short side, because that is what the crossover tracks: the device needs
/// enough total work, not a fat short side. This is the first size that wins outright. See
/// `crates/fp-cuda/EXPERIMENTS.md` for the shapes it was measured on, and for the short-side
/// floor it replaced. Override with `FP_CUDA_RR_MIN_BITS`.
const DEFAULT_RR_MIN_BITS: u64 = 1 << 22;

/// Legacy minimum on the short side, `FP_CUDA_RR_THRESHOLD`.
///
/// Inert at its default of 0: [`DEFAULT_RR_MIN_BITS`] decides. Kept so that scripts setting it keep
/// working, and so `FP_CUDA_RR_THRESHOLD=8192` restores the old behaviour exactly.
const DEFAULT_RR_THRESHOLD: usize = 0;

/// The matmul threshold in use, overridable via the `FP_CUDA_THRESHOLD` environment variable.
fn threshold() -> usize {
    std::env::var("FP_CUDA_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_THRESHOLD)
}

/// The legacy short-side floor in use, overridable via `FP_CUDA_RR_THRESHOLD`.
fn rr_threshold() -> usize {
    std::env::var("FP_CUDA_RR_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_RR_THRESHOLD)
}

/// The row-reduction size floor in use, overridable via `FP_CUDA_RR_MIN_BITS`.
fn rr_min_bits() -> u64 {
    std::env::var("FP_CUDA_RR_MIN_BITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_RR_MIN_BITS)
}

/// Is this reduction worth the device?
///
/// Size decides; the legacy short-side floor applies only when explicitly set.
pub(crate) fn rr_worth_gpu(rows: usize, cols: usize) -> bool {
    let t = rr_threshold();
    if rows < t || cols < t {
        return false;
    }
    (rows as u64).saturating_mul(cols as u64) >= rr_min_bits()
}

/// The process-wide GPU context, created lazily on first use.
///
/// `None` if no usable device is present (no driver, no Hopper GPU, or the kernel PTX is the
/// nvcc-absent build stub), or if `FP_CUDA_DISABLE` is set.
///
/// Shared as `&'static` with no lock: `GpuContext` is `Send + Sync`, every submission goes through
/// a per-thread stream ([`GpuContext::stream`]) so concurrent callers overlap instead of
/// serializing, and device buffers are per-call, so there is no shared state to guard.
fn context() -> Option<&'static GpuContext> {
    static GPU: OnceLock<Option<GpuContext>> = OnceLock::new();
    GPU.get_or_init(|| {
        if std::env::var_os("FP_CUDA_DISABLE").is_some() {
            return None;
        }
        // `FP_CUDA_DEVICE` selects the GPU the row reduction runs on.
        let device = std::env::var("FP_CUDA_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        GpuContext::new(device).ok()
    })
    .as_ref()
}

/// The single thread every `fp-cuda` submission goes through.
///
/// This is what gives the process exactly one owner of the reduction GPU.
///
/// Both entry points — the row reduction's trailing GEMM and the standalone [`try_mul`] — launch
/// grids sized to fill the machine. Co-scheduled they do not fail, they *queue*, and the reduction
/// is a chain of thousands of dependent relaunches, so that queueing lands on a serial critical
/// path. A lock would cover only the call sites that remember to take it; one thread makes single
/// ownership structural.
///
/// Jobs run here to *completion*, not just submission: kernels outlive the call that launched them,
/// and both jobs end in a synchronizing device-to-host download.
mod driver {
    use std::sync::{Mutex, OnceLock, mpsc};

    type Job = Box<dyn FnOnce() + Send + 'static>;

    /// The driver thread's job channel, spawning the thread on first use.
    fn sender() -> &'static Mutex<mpsc::Sender<Job>> {
        static TX: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
        TX.get_or_init(|| {
            let (tx, rx) = mpsc::channel::<Job>();
            std::thread::Builder::new()
                .name("fp-cuda-driver".into())
                .spawn(move || {
                    for job in rx {
                        job();
                    }
                })
                .expect("failed to spawn the fp-cuda driver thread");
            Mutex::new(tx)
        })
    }

    /// Run `f` on the driver thread and block for its result.
    ///
    /// `f` owns everything it touches (both call sites have already marshalled to owned limb
    /// buffers), so nothing borrows across threads.
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

/// A small pool of reusable host buffers for marshalling matrices to and from the device.
///
/// The reduction cannot borrow the matrix it is reducing: [`driver::run`] requires `'static`, so
/// the limbs have to be owned and moved into the closure. That copy is made on the *calling*
/// thread, so allocating one per call makes live memory scale with the driver's queue depth rather
/// than with device concurrency — which at frontier sizes dominated the process.
///
/// Buffers are taken here, moved in, and handed back by the closure whether or not the reduction
/// succeeded: dropping one inside would lose a permit permanently.
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

    /// How many buffers may exist at once, overridable via `FP_CUDA_MARSHAL_BUFFERS`.
    ///
    /// Enough for one buffer being filled while another is in flight. The right bound is really
    /// bytes rather than a count — at frontier sizes a single buffer is many GiB — so raising the
    /// count is not the way to serve more concurrent marshalling.
    fn capacity() -> usize {
        static CAP: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("FP_CUDA_MARSHAL_BUFFERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2)
        });
        *CAP
    }

    /// How long to wait for a free buffer before allocating one instead.
    ///
    /// Short deliberately, because it is hit constantly rather than rarely:
    /// [`super::try_row_reduce`] acquires twice, so one reduction in flight consumes the whole pool
    /// while marshalling stays concurrent across every worker. Waiting bounds nothing either — the
    /// buffer is allocated on timeout regardless — so a long deadline can only ever lose.
    const ACQUIRE_WAIT: Duration = Duration::from_millis(200);

    /// Take a buffer from the pool, or a fresh one if none is free within [`ACQUIRE_WAIT`].
    pub(super) fn acquire() -> Vec<u64> {
        let (lock, cv) = &*POOL;
        let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
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
                .wait_timeout(g, ACQUIRE_WAIT)
                .unwrap_or_else(|e| e.into_inner());
            g = ng;
            if timeout.timed_out() {
                // Either the pool is simply busy — the common case — or a permit was lost when a
                // panicking closure failed to return its buffer. Both are handled the same way:
                // allocate, and let the pool self-heal as live buffers come back.
                g.checked_out += 1;
                return Vec::new();
            }
        }
    }

    /// Hand a buffer back, waking one waiter.
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

/// Pack `m`'s rows into `buf` in the tight row-major layout `GpuContext::upload` expects.
///
/// Replaces a `u64` → little-endian bytes → `u64` round trip, which was two full-size copies to
/// perform the identity on a little-endian machine. When the matrix's own stride already matches
/// the packed one — the common case, since `Matrix::new` sets `columns_capacity == columns` — this
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

/// Row-major, K-major `u64` limbs — the exact layout `fp_cuda::matmul_b1_raw` expects.
///
/// That is `rows × columns.div_ceil(64)` limbs with no inter-row padding. Uses `Matrix::to_bytes`,
/// which already strips the physical row stride.
fn to_limbs(m: &Matrix) -> Vec<u64> {
    let stride = m.columns().div_ceil(64);
    let mut bytes = Vec::with_capacity(m.rows() * stride * 8);
    m.to_bytes(&mut bytes).expect("Vec writes never fail");
    let (chunks, _) = bytes.as_chunks::<8>();
    chunks.iter().map(|&c| u64::from_le_bytes(c)).collect()
}

/// Try to compute `a · b` on the GPU.
///
/// This is consulted by `<&Matrix as Mul>::mul` before the CPU BLAS path: for large enough
/// `p = 2` products it converts the operands to the raw row-major limb layout `fp-cuda` expects,
/// runs the kernel, and rebuilds a [`Matrix`]. Anything that makes the GPU path unavailable or
/// unsuitable — no device, a launch error, or a below-threshold size — returns `None`.
///
/// Assumes `a.prime() == b.prime() == 2` and `a.columns() == b.rows()`.
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

/// Try to row-reduce `m` to RREF on the GPU, in place.
///
/// Returns `Some(rank)`, leaving `m` in the same canonical reduced form `Matrix::row_reduce`
/// produces — pivot rows at the top in column order, zeros below, `pivots` set — and bit-identical
/// to it, which `fp-cuda`'s `row_reduce_demo` validates. Returns `None` if the GPU is unavailable,
/// the matrix is below threshold, or a launch fails; the caller then takes the CPU M4RI path.
pub(crate) fn try_row_reduce(m: &mut Matrix) -> Option<usize> {
    debug_assert_eq!(m.prime(), TWO);
    let (rows, cols) = (m.rows(), m.columns());
    if !rr_worth_gpu(rows, cols) {
        return None;
    }
    let ctx = context()?;

    let stride = cols.div_ceil(64);

    // The default row-reduce is composable (no cooperative launch) and allocates its device
    // buffers per call, so it needs no exclusion of its own; [`driver`] is what keeps this process
    // to a single GPU owner.
    //
    // Both buffers come from [`marshal`] and are handed back by the closure on every path, since
    // dropping one inside would lose a permit permanently.
    let mut in_buf = marshal::acquire();
    fill_limbs(m, &mut in_buf);
    let mut out_buf = marshal::acquire();
    out_buf.clear();
    out_buf.resize(rows * stride, 0);

    let (in_buf, out_buf, res) = driver::run(move || {
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

    // Materialize the canonical RREF in place: pivot k (column pivot_cols[k], ascending) at row k,
    // taken from device row perm[k]; rows [r, rows) zero. Writing into `m`'s existing storage saves
    // a full-size allocation and preserves `m`'s row and column capacity, which `Matrix::from_data`
    // silently discarded — callers such as `extend_image` then `add_row` into it.
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
