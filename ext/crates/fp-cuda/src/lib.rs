//! CUDA backend for `fp::blas` F_2 matrix multiplication on Hopper.
//!
//! The kernel lives in `cuda_kernels/matmul_b1.cu` and is compiled to PTX by
//! `build.rs` (via nvcc with `-arch=sm_90a`). It uses the Hopper memory
//! pipeline end-to-end: TMA bulk loads (`cp.async.bulk.tensor.2d`) +
//! `mbarrier` sync + `wgmma.mma_async.sync.aligned.m64n64k256.row.col.s32.b1.b1.s32.and.popc`.
//!
//! This Rust side uses [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)'s
//! `cuda-core` crate for the host driver-API surface (context, stream, device
//! buffers, untyped module + function loading, raw kernel launch). No
//! Rust-to-PTX path is involved — the kernel is C++ all the way down.
//!
//! The host side builds two `CUtensorMap` descriptors per matmul (one for A,
//! one for B) via the raw `sys::cuTensorMapEncodeTiled` binding and passes
//! them to the kernel as `__grid_constant__` parameters.

use std::{ffi::c_void, mem::MaybeUninit, sync::Arc};

use cuda_core::{
    CudaContext, CudaFunction, CudaModule, CudaStream, DeviceBuffer, launch_kernel_on_stream,
    sys::{
        CUdeviceptr, CUresult, CUtensorMap, cuTensorMapEncodeTiled,
        // bindgen emits CUDA enum variants prefixed with the original C enum name
        // (e.g. `CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT64`). Alias
        // them to readable names at the import site so the call site stays clean.
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT64 as CU_TENSOR_MAP_DATA_TYPE_UINT64,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE as CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE as CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE as CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE as CU_TENSOR_MAP_SWIZZLE_NONE,
        cudaError_enum_CUDA_SUCCESS as CUDA_SUCCESS,
    },
};
use fp::{matrix::Matrix, prime::TWO};

/// PTX image emitted by `build.rs` from `cuda_kernels/matmul_b1.cu`.
static PTX_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_b1.ptx"));

/// Kernel symbol name (must match the `extern "C"` function in matmul_b1.cu).
const KERNEL_NAME: &str = "matmul_b1_kernel";

/// CTA-level tile shape. Must match the constants in matmul_b1.cu.
const TILE_M: u32 = 64;
const TILE_K: u32 = 256;
const TILE_N_BITS: u32 = 64;
const THREADS_PER_CTA: u32 = 128;

/// TMA box dimensions in elements (u64), matching the kernel's `boxDim`.
const BOX_A_X: u32 = TILE_K / 64; // 4 u64s along K
const BOX_A_Y: u32 = TILE_M; // 64 rows along M
const BOX_B_X: u32 = 1; // 1 u64 along N (one column-limb per CTA)
const BOX_B_Y: u32 = TILE_K; // 256 rows along K

/// Long-lived CUDA handle bundling context + loaded module + function pointer.
pub struct GpuContext {
    ctx: Arc<CudaContext>,
    #[allow(dead_code)] // module must outlive the function
    module: Arc<CudaModule>,
    kernel: CudaFunction,
}

impl GpuContext {
    pub fn new(device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(device_id)?;
        let module = ctx.load_module_from_image(PTX_IMAGE)?;
        let kernel = module.load_function(KERNEL_NAME)?;
        Ok(Self {
            ctx,
            module,
            kernel,
        })
    }

    pub fn compute_capability(&self) -> Result<(i32, i32), Box<dyn std::error::Error>> {
        Ok(self.ctx.compute_capability()?)
    }

    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.ctx.default_stream()
    }
}

/// Multiply two F_2 matrices on the GPU and return the product.
///
/// Asserts `a.prime() == b.prime() == 2` and `a.columns() == b.rows()`. The
/// inputs are bridged to GPU memory via `Matrix::to_bytes`; the output comes
/// back via `Matrix::from_data`.
pub fn matmul_b1(
    gpu: &GpuContext,
    a: &Matrix,
    b: &Matrix,
) -> Result<Matrix, Box<dyn std::error::Error>> {
    assert_eq!(a.prime(), TWO, "fp-cuda matmul_b1 requires prime 2");
    assert_eq!(b.prime(), TWO, "fp-cuda matmul_b1 requires prime 2");
    assert_eq!(
        a.columns(),
        b.rows(),
        "shape mismatch: a.columns()={}, b.rows()={}",
        a.columns(),
        b.rows()
    );

    let m_usize = a.rows();
    let k_usize = a.columns();
    let n_usize = b.columns();
    let stride_a_usize = k_usize.div_ceil(64);
    let stride_b_usize = n_usize.div_ceil(64);
    let n_lim_usize = stride_b_usize;

    let stream = gpu.ctx.default_stream();

    let a_limbs = matrix_to_u64s(a);
    let b_limbs = matrix_to_u64s(b);
    debug_assert_eq!(a_limbs.len(), m_usize * stride_a_usize);
    debug_assert_eq!(b_limbs.len(), k_usize * stride_b_usize);

    let a_dev = DeviceBuffer::from_host(&stream, &a_limbs)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_limbs)?;
    let c_dev = DeviceBuffer::<u64>::zeroed(&stream, m_usize * n_lim_usize)?;

    // Build CUtensorMap descriptors for A and B.
    let tensor_map_a = encode_tensor_map_2d(
        a_dev.cu_deviceptr(),
        stride_a_usize as u64,     // innermost (X) dim length in u64s
        m_usize as u64,            // outermost (Y) dim length in rows
        stride_a_usize as u64 * 8, // row stride in bytes (Y stride)
        BOX_A_X,
        BOX_A_Y,
    )?;
    let tensor_map_b = encode_tensor_map_2d(
        b_dev.cu_deviceptr(),
        stride_b_usize as u64,
        k_usize as u64,
        stride_b_usize as u64 * 8,
        BOX_B_X,
        BOX_B_Y,
    )?;

    // Pack kernel parameters. Order matches the C++ signature:
    //   (const __grid_constant__ CUtensorMap tensor_map_a,
    //    const __grid_constant__ CUtensorMap tensor_map_b,
    //    uint32_t m, uint32_t k, uint32_t n_lim,
    //    uint64_t* c)
    let mut tma_a_storage = tensor_map_a;
    let mut tma_b_storage = tensor_map_b;
    let mut m_val: u32 = m_usize as u32;
    let mut k_val: u32 = k_usize as u32;
    let mut n_lim_val: u32 = n_lim_usize as u32;
    let mut c_ptr: CUdeviceptr = c_dev.cu_deviceptr();

    let mut params: [*mut c_void; 6] = [
        &mut tma_a_storage as *mut _ as *mut c_void,
        &mut tma_b_storage as *mut _ as *mut c_void,
        &mut m_val as *mut _ as *mut c_void,
        &mut k_val as *mut _ as *mut c_void,
        &mut n_lim_val as *mut _ as *mut c_void,
        &mut c_ptr as *mut _ as *mut c_void,
    ];

    let grid_x = n_lim_val;
    let grid_y = m_val.div_ceil(TILE_M);
    let _ = (TILE_K, TILE_N_BITS); // referenced by the kernel; pinned in const above

    // SAFETY: PTX was compiled from a kernel matching this signature; tensor
    // maps were just encoded with valid global addresses owned by a_dev/b_dev;
    // params[i] points to a value with matching size/alignment; buffers and
    // tensor-map storage outlive the call because we synchronize before
    // returning.
    unsafe {
        launch_kernel_on_stream(
            &gpu.kernel,
            (grid_x, grid_y, 1),
            (THREADS_PER_CTA, 1, 1),
            0,
            &stream,
            &mut params,
        )?;
    }
    stream.synchronize()?;

    let c_limbs = c_dev.to_host_vec(&stream)?;
    Ok(Matrix::from_data(TWO, m_usize, n_usize, c_limbs))
}

/// Build a 2D `CUtensorMap` for a u64-packed binary tile, no swizzle.
///
/// `global_addr` is the device pointer to the tensor's element-zero. The
/// tensor is interpreted as 2D row-major with `global_dim = (dim_x, dim_y)`
/// u64 elements; `row_stride_bytes` is the byte distance from one Y row to
/// the next (= `dim_x * 8` for tightly packed data). `box_x` × `box_y` is
/// the tile size the kernel pulls per TMA load.
fn encode_tensor_map_2d(
    global_addr: CUdeviceptr,
    dim_x: u64,
    dim_y: u64,
    row_stride_bytes: u64,
    box_x: u32,
    box_y: u32,
) -> Result<CUtensorMap, Box<dyn std::error::Error>> {
    let mut tmap: MaybeUninit<CUtensorMap> = MaybeUninit::uninit();

    let global_dim: [u64; 2] = [dim_x, dim_y];
    // `globalStrides` has length `tensorRank - 1` and gives the byte stride
    // between successive elements along the OUTER dimension(s). For 2D:
    // a single entry = bytes per row.
    let global_strides: [u64; 1] = [row_stride_bytes];
    let box_dim: [u32; 2] = [box_x, box_y];
    let element_strides: [u32; 2] = [1, 1];

    // SAFETY: `cuTensorMapEncodeTiled` writes into `tmap`. All pointers are
    // valid for the duration of the call; the rank is 2 (matches both array
    // lengths); the data type is UINT64 (8 bytes per element). The function
    // is host-only and does not require an active context.
    let res: CUresult = unsafe {
        cuTensorMapEncodeTiled(
            tmap.as_mut_ptr(),
            CU_TENSOR_MAP_DATA_TYPE_UINT64,
            2,
            global_addr as *mut c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if res != CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled failed: {res:?}").into());
    }
    // SAFETY: the call returned CUDA_SUCCESS, so tmap is now fully initialised.
    Ok(unsafe { tmap.assume_init() })
}

/// Serialize a `Matrix` into row-major bit-packed `u64` limbs.
///
/// Round-trips through `Matrix::to_bytes` because the underlying limb-slice
/// accessor on `Matrix` is `pub(crate)`. Little-endian byte order of
/// `to_bytes` matches the natural `u64` interpretation, so limb values are
/// preserved verbatim.
fn matrix_to_u64s(m: &Matrix) -> Vec<u64> {
    let stride = m.columns().div_ceil(64);
    let len = m.rows() * stride;
    let mut bytes = Vec::with_capacity(len * 8);
    m.to_bytes(&mut bytes).expect("Vec writes never fail");
    debug_assert_eq!(bytes.len(), len * 8);
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}
