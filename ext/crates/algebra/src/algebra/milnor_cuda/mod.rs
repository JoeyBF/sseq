//! The CUDA backend for the Milnor multiply: cudarc driver API + NVRTC + virtual memory.
//!
//! Named for what it is rather than `cuda_rt`/`cuda`, which conventionally means the CUDA
//! RUNTIME API (`libcudart.so`). cudarc binds the DRIVER API, so that name said the opposite
//! of what this does.
//!
//! This is the foundation for moving the Milnor GPU path off cubecl. It is deliberately
//! self-contained and separately feature-gated (`cuda`), so both backends can be built at once
//! and digest-compared on identical input before anything is switched over.
//!
//! # Why leave cubecl
//!
//! Two problems, both in its memory manager rather than its codegen.
//!
//! **It hides allocation failures.** cubecl sizes its pools from *total device memory* and raises
//! `BufferTooBig` when no pool accepts a size — then swallows it, and the dead handle is used
//! anyway ("Memory location was never initialized"). The output buffer is never written, so the
//! multiply returns an ALL-ZERO product at `exit 0`, quickly, because no work happened. Measured on
//! an 11 GB card: the pool refused 960,000,000 bytes while plain `cuMemAlloc` on the same card
//! served 1/2/4/8 GiB and held 10,496 MiB at once. Every CPU-reference unit test passes there,
//! because none is large enough to reach the cap.
//!
//! **It has no VMM, which is why the master is segmented.** Growing a device buffer means
//! reallocating and copying, so `milnor_gpu` grows the resident master by appending fixed-size
//! segments — and that brings `MASTER_MAX_SEG = 16`, sixteen segment kernel arguments, the
//! `seg_read_*` branch chains, and a two-sided squeeze on the segment size: too large exceeds the
//! pool cap, too small needs more than sixteen segments. At full frontier batch size on an 11 GB
//! card that window is empty.
//!
//! [`GrowBuf`] removes the cause. One reserved virtual address range is grown by *mapping* physical
//! chunks into it, so the pointer never moves and nothing is ever copied — the kernel sees a single
//! contiguous array however many chunks back it.
//!
//! # Kernels are compiled at runtime
//!
//! NVRTC, not nvcc-at-build. Build-time PTX is what produced the silent stub failure in `fp-cuda`:
//! with no toolchain present the build emitted a 210-byte stub carrying no kernel, every dispatch
//! declined, and the banner still printed. Compiling at runtime also specialises per device
//! architecture and lets `#[comptime]`-style folding become `-D` defines.

pub mod enumerate;
pub mod multiply;
pub mod params;
pub mod resident;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::driver::{CudaContext, CudaModule, sys};

/// Errors from this backend.
///
/// Every variant is *returned*, never swallowed. That is the point: the failure this backend
/// exists to eliminate was an allocation error that became an all-zero result.
#[derive(Debug)]
pub enum CudaError {
    /// A driver call failed, with the call site and the raw `CUresult`.
    Driver(&'static str, sys::CUresult),
    /// NVRTC failed to compile a kernel; carries the compiler log.
    Compile(String),
    /// A kernel or module name was not valid C.
    BadName(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(what, r) => write!(f, "CUDA driver call {what} failed: {r:?}"),
            Self::Compile(log) => write!(f, "NVRTC compilation failed:\n{log}"),
            Self::BadName(n) => write!(f, "invalid kernel or module name: {n}"),
        }
    }
}

impl std::error::Error for CudaError {}

type Result<T> = std::result::Result<T, CudaError>;

/// Check a raw driver result, naming the call for the error message.
fn ck(what: &'static str, r: sys::CUresult) -> Result<()> {
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::Driver(what, r))
    }
}

/// A device buffer that GROWS WITHOUT MOVING, backed by CUDA virtual memory.
///
/// A large virtual range is reserved up front — reserving address space costs no physical memory —
/// and physical chunks are committed into it as the buffer grows. The device pointer is stable for
/// the buffer's whole life, so kernels take one pointer and one length regardless of how many
/// chunks back it.
///
/// This is what makes the segmented master unnecessary. Compare the cubecl path, which cannot grow
/// an allocation in place and therefore appends fixed-size segments, capped at `MASTER_MAX_SEG`
/// and passed to kernels as sixteen separate arguments read through a branch chain.
///
/// Chunks are never unmapped on growth, so previously written data stays valid and no copy occurs.
pub struct GrowBuf {
    /// Base of the reserved virtual range. Stable for the life of the buffer.
    base: sys::CUdeviceptr,
    /// Bytes of address space reserved (an upper bound on growth).
    reserved: usize,
    /// Bytes actually backed by physical memory.
    committed: usize,
    /// Allocation granularity; every commit is a multiple of this.
    granularity: usize,
    /// Physical handles, kept so they can be released in `Drop`.
    handles: Vec<sys::CUmemGenericAllocationHandle>,
    device: i32,
}

// The buffer owns its mapping; the raw pointer is not aliased outside it.
unsafe impl Send for GrowBuf {}
unsafe impl Sync for GrowBuf {}

impl GrowBuf {
    /// Reserve address space on the device a [`MilnorCuda`] already has bound.
    ///
    /// Prefer this over [`GrowBuf::reserve`] off the runtime's thread: the raw VMM entry points are
    /// `sys::` calls, which do not bind a context by themselves.
    pub fn reserve_on(rt: &MilnorCuda, reserve_bytes: usize) -> Result<Self> {
        rt.context().bind_to_thread().map_err(|_| {
            CudaError::Driver("bind_to_thread", sys::CUresult::CUDA_ERROR_INVALID_CONTEXT)
        })?;
        Self::reserve(rt.device(), reserve_bytes)
    }

    /// Reserve `reserve_bytes` of device address space, committing nothing yet.
    ///
    /// Reserve generously: address space is free, and the reservation is the only hard ceiling on
    /// later growth. This is precisely the knob that does NOT exist under cubecl, where the
    /// equivalent ceiling is `MASTER_MAX_SEG * seg_elems` and has to be tuned per card.
    pub fn reserve(device: i32, reserve_bytes: usize) -> Result<Self> {
        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;

        let mut granularity: usize = 0;
        unsafe {
            ck(
                "cuMemGetAllocationGranularity",
                sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                ),
            )?;
        }
        let granularity = granularity.max(1);
        let reserved = reserve_bytes.div_ceil(granularity) * granularity;

        let mut base: sys::CUdeviceptr = 0;
        unsafe {
            ck(
                "cuMemAddressReserve",
                sys::cuMemAddressReserve(&mut base, reserved, granularity, 0, 0),
            )?;
        }
        Ok(Self {
            base,
            reserved,
            committed: 0,
            granularity,
            handles: Vec::new(),
            device,
        })
    }

    /// Device pointer to the start of the buffer. Stable across every `grow_to`.
    pub fn ptr(&self) -> sys::CUdeviceptr {
        self.base
    }

    /// Bytes currently backed by physical memory.
    pub fn committed(&self) -> usize {
        self.committed
    }

    /// Bytes of address space reserved.
    pub fn reserved(&self) -> usize {
        self.reserved
    }

    /// Ensure at least `bytes` are physically backed, committing more if needed.
    ///
    /// Existing chunks are left mapped, so data already written stays valid and nothing is copied.
    /// Returns an error rather than silently under-committing — the failure mode this backend
    /// exists to avoid is exactly a short buffer that reads as zeros.
    pub fn grow_to(&mut self, bytes: usize) -> Result<()> {
        if bytes <= self.committed {
            return Ok(());
        }
        if bytes > self.reserved {
            // Out of reserved address space: a real limit, surfaced rather than hidden.
            return Err(CudaError::Driver(
                "grow_to beyond reserved address space",
                sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            ));
        }
        let target = bytes.div_ceil(self.granularity) * self.granularity;
        let extra = target - self.committed;

        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = self.device;

        let mut handle: sys::CUmemGenericAllocationHandle = 0;
        unsafe {
            ck(
                "cuMemCreate",
                sys::cuMemCreate(&mut handle, extra, &prop, 0),
            )?;
            let at = self.base + self.committed as sys::CUdeviceptr;
            if let Err(e) = ck("cuMemMap", sys::cuMemMap(at, extra, 0, handle, 0)) {
                let _ = sys::cuMemRelease(handle);
                return Err(e);
            }
            let access = sys::CUmemAccessDesc {
                location: sys::CUmemLocation {
                    type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                    id: self.device,
                },
                flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            };
            if let Err(e) = ck("cuMemSetAccess", sys::cuMemSetAccess(at, extra, &access, 1)) {
                let _ = sys::cuMemUnmap(at, extra);
                let _ = sys::cuMemRelease(handle);
                return Err(e);
            }
        }
        self.handles.push(handle);
        self.committed = target;
        Ok(())
    }

    /// Grow as needed and copy `data` in at `offset_bytes`.
    ///
    /// This is the whole append path for the resident master, and it is worth noticing how little
    /// there is of it. Under cubecl the same operation needed a segment table, a staging chunk size
    /// (`STAGE_CHUNK`, because `create_from_slice` pins host pages per stream and never trims them
    /// -- measured ~240 GB of pinned shmem at stem 180 across 8 streams), a copy kernel to move the
    /// old contents into the new allocation, and a `MASTER_MAX_SEG` ceiling. Growth in place needs
    /// none of it: the old bytes never move, so there is nothing to copy and nothing to stage.
    ///
    /// Synchronous, and the caller must already have the context bound -- `cuMemcpyHtoD_v2` is a
    /// raw `sys::` entry point and does not bind one.
    pub fn write_at<T: Copy>(&mut self, offset_bytes: usize, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(());
        }
        self.grow_to(offset_bytes + bytes)?;
        unsafe {
            ck(
                "cuMemcpyHtoD",
                sys::cuMemcpyHtoD_v2(
                    self.base + offset_bytes as sys::CUdeviceptr,
                    data.as_ptr().cast(),
                    bytes,
                ),
            )
        }
    }
}

impl Drop for GrowBuf {
    fn drop(&mut self) {
        unsafe {
            if self.committed > 0 {
                let _ = sys::cuMemUnmap(self.base, self.committed);
            }
            for &h in &self.handles {
                let _ = sys::cuMemRelease(h);
            }
            if self.reserved > 0 {
                let _ = sys::cuMemAddressFree(self.base, self.reserved);
            }
        }
    }
}

/// The NVRTC `--gpu-architecture` string for a compute capability.
///
/// A table rather than a formatted `String`, because `CompileOptions::arch` is
/// `Option<&'static str>`. fp-cuda gets away with a single `const ARCH` since its wgmma kernel is
/// sm_90a-only; the Milnor kernels must run on both Ada and Hopper, so the arch is chosen per
/// device and still has to be `'static`.
fn arch_str(major: u32, minor: u32) -> Option<&'static str> {
    Some(match (major, minor) {
        (7, 0) => "compute_70",
        (7, 5) => "compute_75",
        (8, 0) => "compute_80",
        (8, 6) => "compute_86",
        (8, 9) => "compute_89",
        (9, 0) => "compute_90",
        (10, 0) => "compute_100",
        (12, 0) => "compute_120",
        // Unknown capability: let NVRTC pick its default rather than guess a wrong one.
        _ => return None,
    })
}

/// Compile CUDA C to PTX at runtime for this device's architecture.
///
/// Follows the convention PR 298 established for fp-cuda:
/// * probe `is_culib_present` FIRST -- it has to precede any other nvrtc call -- and say plainly
///   that the toolkit is missing. Without it an absent `libnvrtc` surfaces as an opaque failure
///   deep inside compilation.
/// * tuning constants live in Rust and reach the kernel as `-D` options, so there is ONE source of
///   truth. The kernel should define none itself and `#error` on a missing one, which turns a
///   forgotten define into a compile error instead of a silently wrong default.
///
/// That `-D` mechanism is also how cubecl's `#[comptime]` folding carries over: cubecl folds at
/// macro-expansion time, NVRTC folds at compile time from these.
pub fn compile_ptx(
    src: &str,
    name: &str,
    arch: (u32, u32),
    defines: &[(&str, String)],
) -> Result<String> {
    use cudarc::nvrtc::safe::{CompileOptions, compile_ptx_with_opts};

    // Must come before any other nvrtc call.
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(CudaError::Compile(format!(
            "{name}: libnvrtc was not found, so the CUDA kernel cannot be compiled. Install the \
             CUDA toolkit (module load cuda/12.4) or put libnvrtc on the library path."
        )));
    }

    let opts = CompileOptions {
        arch: arch_str(arch.0, arch.1),
        options: defines
            .iter()
            .map(|(k, v)| format!("-D{k}={v}"))
            .collect::<Vec<_>>(),
        ..Default::default()
    };
    compile_ptx_with_opts(src, opts)
        .map(|p| p.to_src())
        .map_err(|e| CudaError::Compile(format!("{name}: {e:?}")))
}

/// Per-device context plus a cache of compiled modules.
pub struct MilnorCuda {
    ctx: Arc<CudaContext>,
    device: i32,
    arch: (u32, u32),
    modules: Mutex<HashMap<String, Arc<CudaModule>>>,
}

impl MilnorCuda {
    /// Open a device.
    ///
    /// `CudaContext::new` PANICS rather than returning `Err` when no driver is present, so probing
    /// for a GPU must go through `catch_unwind` at the call site; do not install a panic hook to
    /// hide it.
    pub fn new(device: i32) -> Result<Self> {
        let ctx = CudaContext::new(device as usize)
            .map_err(|e| CudaError::Compile(format!("CudaContext::new({device}): {e:?}")))?;
        let major = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0) as u32;
        let minor = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0) as u32;
        Ok(Self {
            ctx,
            device,
            arch: (major, minor),
            modules: Mutex::new(HashMap::new()),
        })
    }

    /// Compute capability, as used for the NVRTC `--gpu-architecture` flag.
    pub fn arch(&self) -> (u32, u32) {
        self.arch
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Compile (or fetch from cache) a module. Keyed by `key`, which must cover the defines.
    pub fn module(
        &self,
        key: &str,
        src: &str,
        defines: &[(&str, String)],
    ) -> Result<Arc<CudaModule>> {
        if let Some(m) = self.modules.lock().unwrap().get(key) {
            return Ok(m.clone());
        }
        let ptx = compile_ptx(src, key, self.arch, defines)?;
        let module = self
            .ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx))
            .map_err(|e| CudaError::Compile(format!("load_module({key}): {e:?}")))?;
        self.modules
            .lock()
            .unwrap()
            .insert(key.to_owned(), module.clone());
        Ok(module)
    }

    /// Largest single allocation the DRIVER will serve, probed by bisection.
    ///
    /// Recorded because the cubecl failure looked like a hardware limit and was not: the pool
    /// refused 960,000,000 bytes on a card that `cuMemAlloc`'d 8 GiB. Anything claiming a device
    /// cannot allocate should be checked against this first.
    pub fn probe_max_alloc(&self) -> usize {
        // Raw `sys::` calls do not bind the context; cudarc's own wrappers do. Without this the
        // probe reports 0 GiB on a perfectly healthy card -- which is exactly the sort of bogus
        // "the device cannot allocate" reading this function exists to refute.
        if self.ctx.bind_to_thread().is_err() {
            return 0;
        }
        let mut lo = 0usize;
        let mut hi = 64usize << 30;
        while lo + (1 << 20) < hi {
            let mid = lo + (hi - lo) / 2;
            let mut p: sys::CUdeviceptr = 0;
            let ok = unsafe { sys::cuMemAlloc_v2(&mut p, mid) } == sys::CUresult::CUDA_SUCCESS;
            if ok {
                unsafe { sys::cuMemFree_v2(p) };
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Process-wide contexts, one per device.
static RUNTIMES: OnceLock<Mutex<HashMap<i32, Arc<MilnorCuda>>>> = OnceLock::new();

/// The runtime for `device`, opening it on first use.
pub fn runtime(device: i32) -> Result<Arc<MilnorCuda>> {
    let map = RUNTIMES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(rt) = map.lock().unwrap().get(&device) {
        return Ok(rt.clone());
    }
    let rt = Arc::new(MilnorCuda::new(device)?);
    map.lock().unwrap().insert(device, rt.clone());
    Ok(rt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NVRTC compiles and the module loads and launches. Proves the runtime-compilation path
    /// end-to-end, including that `-D` defines reach the kernel — which is how the cubecl
    /// `#[comptime]` specialisation will carry over.
    #[test]
    #[ignore = "needs a CUDA device; run explicitly with --ignored"]
    fn nvrtc_compiles_and_launches() {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        const SRC: &str = r#"
extern "C" __global__ void axpy(float *out, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = SCALE * x[i];
}
"#;
        let rt = runtime(0).expect("open device 0");
        let module = rt
            .module("axpy_scale3", SRC, &[("SCALE", "3.0f".to_string())])
            .expect("compile axpy");
        let f = module.load_function("axpy").expect("load axpy");

        let stream = rt.context().default_stream();
        let n = 1024usize;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dx = stream.memcpy_stod(&x).expect("upload");
        let mut dout = stream.alloc_zeros::<f32>(n).expect("alloc out");

        let cfg = LaunchConfig::for_num_elems(n as u32);
        // `n` needs a binding: `arg` borrows, so a temporary would be dropped before the launch.
        let n_arg = n as i32;
        let mut b = stream.launch_builder(&f);
        b.arg(&mut dout).arg(&dx).arg(&n_arg);
        unsafe { b.launch(cfg) }.expect("launch");

        let got = stream.memcpy_dtov(&dout).expect("readback");
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, 3.0 * i as f32, "axpy mismatch at {i}");
        }
    }

    /// A VMM buffer grows WITHOUT MOVING and without losing what was already written.
    ///
    /// This is the property the segmented master is a workaround for, so it is worth asserting
    /// directly: the pointer is unchanged across growth, and data written before a grow is still
    /// readable at the same offsets afterwards.
    #[test]
    #[ignore = "needs a CUDA device; run explicitly with --ignored"]
    fn grow_buf_keeps_pointer_and_contents() {
        let rt = runtime(0).expect("open device 0");
        let stream = rt.context().default_stream();

        // Reserve 1 GiB of ADDRESS SPACE (free), commit 4 MiB.
        //
        // `reserve_on`, not `reserve`: this test drives the VMM and the copies through raw `sys::`
        // entry points, and those do NOT bind a CUDA context by themselves. Run alone the test
        // happened to inherit a context another call had left current on the thread; run alongside
        // its siblings in one process it got `CUDA_ERROR_INVALID_CONTEXT` on the first copy. That
        // is the same trap that once made `probe_max_alloc` report 0 GiB on a healthy card.
        let mut buf = GrowBuf::reserve_on(&rt, 1 << 30).expect("reserve");
        buf.grow_to(4 << 20).expect("first commit");
        let ptr0 = buf.ptr();

        let first: Vec<u32> = (0..(1 << 20)).collect();
        unsafe {
            ck(
                "cuMemcpyHtoD first",
                sys::cuMemcpyHtoD_v2(
                    buf.ptr(),
                    first.as_ptr() as *const _,
                    first.len() * size_of::<u32>(),
                ),
            )
            .expect("upload first");
        }

        // Grow past the first commit several times; each crosses a fresh physical chunk.
        for mb in [16usize, 64, 192] {
            buf.grow_to(mb << 20).expect("grow");
            assert_eq!(buf.ptr(), ptr0, "VMM buffer MOVED on growth at {mb} MiB");
        }

        // Everything written before the growth must still be there, at the same offsets.
        let mut back = vec![0u32; first.len()];
        unsafe {
            ck(
                "cuMemcpyDtoH",
                sys::cuMemcpyDtoH_v2(
                    back.as_mut_ptr() as *mut _,
                    buf.ptr(),
                    back.len() * size_of::<u32>(),
                ),
            )
            .expect("readback");
        }
        assert_eq!(back, first, "contents changed across growth");

        // And the newly committed tail is writable through the same base pointer.
        let tail_off = (128usize << 20) / size_of::<u32>();
        let tail: Vec<u32> = (0..4096u32).map(|i| i ^ 0xa5a5).collect();
        unsafe {
            ck(
                "cuMemcpyHtoD tail",
                sys::cuMemcpyHtoD_v2(
                    buf.ptr() + (tail_off * size_of::<u32>()) as sys::CUdeviceptr,
                    tail.as_ptr() as *const _,
                    tail.len() * size_of::<u32>(),
                ),
            )
            .expect("upload tail");
        }
        let mut tail_back = vec![0u32; tail.len()];
        unsafe {
            ck(
                "cuMemcpyDtoH tail",
                sys::cuMemcpyDtoH_v2(
                    tail_back.as_mut_ptr() as *mut _,
                    buf.ptr() + (tail_off * size_of::<u32>()) as sys::CUdeviceptr,
                    tail_back.len() * size_of::<u32>(),
                ),
            )
            .expect("readback tail");
        }
        assert_eq!(
            tail_back, tail,
            "tail write/read through grown region failed"
        );
        drop(stream);
    }

    /// The driver's real allocation ceiling, for comparison against what cubecl claims.
    #[test]
    #[ignore = "needs a CUDA device; diagnostic, run explicitly with --ignored --nocapture"]
    fn report_max_alloc() {
        let rt = runtime(0).expect("open device 0");
        let max = rt.probe_max_alloc();
        let (maj, min) = rt.arch();
        eprintln!(
            "[milnor-cuda] device {} sm_{maj}{min}: largest single cuMemAlloc = {:.2} GiB",
            rt.device(),
            max as f64 / (1u64 << 30) as f64
        );
        assert!(
            max > 0,
            "driver would not allocate anything at all -- is the context bound on this thread?"
        );
    }
}
