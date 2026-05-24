// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Hopper wgmma.b1 F_2 GEMM kernel with TMA bulk loads.
//
// One CTA = one warpgroup (128 threads). One CTA computes a 64-row by
// 64-col bit tile of C (i.e. one u64 column-limb of 64 consecutive output
// rows). One wgmma.mma_async.sync.aligned.m64n64k256.row.col.s32.b1.b1.s32
// is issued per 256-bit K chunk; the s32 accumulator is reduced to F_2 by
// taking the LSB after the K loop, then bit-packed back into u64 limbs.
//
// Memory pipeline (Phase 1, TMA without swizzle):
//   - A and B tiles are pulled into SMEM via cp.async.bulk.tensor.2d.
//     Tensor descriptors (`CUtensorMap`) are built on the host with
//     `cuTensorMapEncodeTiled` (`CU_TENSOR_MAP_SWIZZLE_NONE` for now —
//     Phase 1.5 tuning will move to `CU_TENSOR_MAP_SWIZZLE_128B` and the
//     matching swizzle mode in the wgmma SMEM descriptor to eliminate bank
//     conflicts on the wgmma reads).
//   - An mbarrier tracks TMA completion per K iteration.
//   - The B tile lands row-major in SMEM, then a warp-level
//     `__ballot_sync` transpose builds the col-major tile wgmma expects.
//
// Layout assumptions (must match the host side in src/lib.rs):
//   A: row-major, bit-packed u64 limbs, row stride = stride_a u64s.
//   B: row-major, bit-packed u64 limbs, row stride = stride_b u64s.
//      Each CTA reads the slice b[:, col_limb] — a single u64 column.
//   C: row-major, bit-packed u64 limbs, row stride = n_lim u64s.
//   LSB-first bit ordering within each u64 (column j is bit j%64 of the
//   limb at index j/64 within the row).
//
// HARDWARE-VALIDATION NOTES (read before running on a Hopper GPU):
//   1. The SMEM descriptor `leading_dim_bytes` / `stride_bytes` passed to
//      `make_smem_desc` use swizzle=0 conventions; once we move to 128B
//      swizzle the values change (CUTLASS `cute::SM90_64x64x256_S32_TN_B1B1`
//      atom is the canonical reference).
//   2. The per-thread accumulator → output bit mapping is derived from the
//      PTX manual's "Matrix Fragments for WGMMA" table for `m64n64.s32`;
//      verify with a 64×64 identity-matrix product before benching.
//   3. The mbarrier transaction-count must equal the total bytes loaded by
//      TMA per iteration (here 2048 for A + 2048 for B = 4096).
//
// References:
//   - NVIDIA PTX ISA 8.0+, §9.7.13 (SMEM descriptors)
//   - NVIDIA PTX ISA 8.0+, §9.7.14 (wgmma)
//   - NVIDIA PTX ISA 8.0+, §9.7.9  (cp.async.bulk.tensor)
//   - NVIDIA PTX ISA 8.0+, §9.7.10 (mbarrier)
//   - NVIDIA Hopper Tuning Guide, §1.4.6.2 (warpgroup MMA)
//   - CUDA Programming Guide, Appendix on TMA + tensor maps

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>  // for CUtensorMap

// ============================================================================
// SMEM descriptor encoder (swizzle mode 0 / "no swizzle")
// ============================================================================

// Bit layout per PTX manual §9.7.13.2:
//   [13:0]   start address >> 4
//   [29:16]  leading-dim byte offset >> 4
//   [45:32]  stride byte offset >> 4
//   [49:46]  base offset within 128-byte chunk (0 for 128B-aligned)
//   [52:50]  swizzle mode (0 = no swizzle, 1 = 128B, 2 = 64B, 3 = 32B —
//            verify mapping against latest PTX manual; older versions use
//            different ordering)
//   [63:53]  must be 0
__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr,
    uint32_t leading_dim_bytes,
    uint32_t stride_bytes
) {
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (static_cast<uint64_t>(shared_addr) >> 4) & 0x3FFFULL;
    desc |= (static_cast<uint64_t>(leading_dim_bytes >> 4) & 0x3FFFULL) << 16;
    desc |= (static_cast<uint64_t>(stride_bytes >> 4) & 0x3FFFULL) << 32;
    // swizzle = 0 (no swizzle); base offset = 0
    return desc;
}

// ============================================================================
// TMA bulk load + mbarrier helpers
// ============================================================================

__device__ __forceinline__ void mbarrier_init(uint64_t* bar_smem, uint32_t arrive_count) {
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_smem));
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_addr), "r"(arrive_count)
    );
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar_smem, uint32_t tx_count) {
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_smem));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_addr), "r"(tx_count) : "memory"
    );
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar_smem, uint32_t phase) {
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_smem));
    asm volatile(
        "{ .reg .pred P;                                                              \n"
        "  WAITLOOP:                                                                  \n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;                      \n"
        "  @P bra DONE;                                                               \n"
        "  bra WAITLOOP;                                                              \n"
        "  DONE: }                                                                    \n"
        :: "r"(bar_addr), "r"(phase) : "memory"
    );
}

// Issue a 2D TMA bulk load into SMEM. `tensor_map_ptr` is the address of the
// CUtensorMap (passed by value into the kernel and addressed via .param).
__device__ __forceinline__ void tma_load_2d(
    void* smem_dst,
    const CUtensorMap* tensor_map_ptr,
    int32_t coord_x,
    int32_t coord_y,
    uint64_t* bar_smem
) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_smem));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dst_addr),
           "l"(reinterpret_cast<uint64_t>(tensor_map_ptr)),
           "r"(coord_x), "r"(coord_y),
           "r"(bar_addr)
        : "memory"
    );
}

// ============================================================================
// wgmma.b1 m64n64k256 wrapper (one warpgroup, 32 s32 acc regs per thread)
// ============================================================================

// scale_d = 1 (always accumulate); we initialize acc to 0 before the K loop.
// imm_scale_a = imm_scale_b = 1 (no negation; meaningless for binary).
__device__ __forceinline__ void wgmma_b1_m64n64k256_and(
    int32_t acc[32],
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k256.row.col.s32.b1.b1.s32.and.popc "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, 1, 1, 1;\n"
        : "+r"(acc[0]),  "+r"(acc[1]),  "+r"(acc[2]),  "+r"(acc[3]),
          "+r"(acc[4]),  "+r"(acc[5]),  "+r"(acc[6]),  "+r"(acc[7]),
          "+r"(acc[8]),  "+r"(acc[9]),  "+r"(acc[10]), "+r"(acc[11]),
          "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]),
          "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]),
          "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]),
          "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]),
          "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        : "l"(desc_a), "l"(desc_b)
    );
}

__device__ __forceinline__ void wgmma_fence()        { asm volatile("wgmma.fence.sync.aligned;\n"        ::: "memory"); }
__device__ __forceinline__ void wgmma_commit_group() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait_zero()    { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

// ============================================================================
// Kernel
// ============================================================================

// Tile shape (per CTA): 64 rows × 64 cols of C = 64 × 1 u64 limbs.
constexpr uint32_t TILE_M = 64;
constexpr uint32_t TILE_N_BITS = 64;
constexpr uint32_t TILE_K = 256;
constexpr uint32_t K_LIMBS_PER_CHUNK = TILE_K / 64;  // 4 u64s along K per A row
constexpr uint32_t TMA_BYTES_A = TILE_M * K_LIMBS_PER_CHUNK * 8;   // 64*4*8 = 2048
constexpr uint32_t TMA_BYTES_B = TILE_K * 1 * 8;                   // 256*1*8 = 2048

// Bit transpose helpers
// =====================
//
// After TMA: smem_b_raw[k_row] holds B[k_row][col_limb] as a u64 (64 N bits).
// wgmma needs smem_b_swz[j * K_LIMBS_PER_CHUNK + (k_row/64)] with bit
// (k_row % 64) of column j. We build it via warp-level __ballot_sync.
//
// Each warp owns 32 K-rows. For each j in [0, 64), all 32 lanes test bit j
// of their k_row's u64; __ballot_sync packs 32 lanes into a u32. Two warps
// (covering 64 K-rows) combine to fill one u64. With 4 warps × 32 K-rows =
// 128 K-rows per pass, we run 2 passes to cover 256 K-rows.
__device__ __forceinline__ void transpose_b_warp(
    const uint64_t* smem_b_raw,   // [256] u64s, row-major
    uint64_t*       smem_b_swz    // [64*4] u64s, col-major (j * 4 + k/64)
) {
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane    = tid & 31;

    // Two passes: pass 0 handles K-rows [warp_id*32, warp_id*32+32),
    // pass 1 handles K-rows [128 + warp_id*32, 128 + warp_id*32+32).
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        uint32_t k_base = pass * 128 + warp_id * 32;
        uint64_t my_row = smem_b_raw[k_base + lane];

        // For each output column j, ballot the warp to get 32 bits.
        // The thread whose lane == j collects the result.
        #pragma unroll
        for (int j = 0; j < 64; ++j) {
            uint32_t bit = static_cast<uint32_t>((my_row >> j) & 1ULL);
            uint32_t mask = __ballot_sync(0xFFFFFFFFu, bit);

            // Lane "j mod 32" gathers; for j >= 32 we'd need a second pass
            // along lane. Simpler: every lane stores its own slot.
            // The result mask is the same across all lanes (ballot is
            // warp-uniform), so any lane can store. Use lane 0 to avoid
            // contention.
            if (lane == 0) {
                // Output bit position within the column's 4-limb stride:
                //   column j, K-bit-position k_base+0..k_base+31
                //   target u64 = smem_b_swz[j * 4 + (k_base / 64)]
                //   bit-shift within that u64 = k_base % 64
                uint32_t dst_idx = j * K_LIMBS_PER_CHUNK + (k_base / 64);
                uint64_t shift   = k_base & 63;
                // Clear the 32-bit slot we're writing (atomic-free because
                // pass+warp_id uniquely selects (dst_idx, shift) ranges,
                // but warp_id 0 and warp_id 1 share the same dst_idx in
                // pass 0 — they have shifts 0 and 32 respectively, so OR
                // is safe).
                atomicOr(reinterpret_cast<unsigned long long*>(&smem_b_swz[dst_idx]),
                         (static_cast<unsigned long long>(mask) << shift));
            }
        }
    }
}

// Kernel signature: tensor maps passed by value (driver copies into kernel
// param region). The data pointers are *inside* the tensor maps; we don't
// take a/b as separate args.
extern "C" __global__ void matmul_b1_kernel(
    const __grid_constant__ CUtensorMap tensor_map_a,
    const __grid_constant__ CUtensorMap tensor_map_b,
    uint32_t m,
    uint32_t k,
    uint32_t n_lim,
    uint64_t* __restrict__ c
) {
    // SMEM layout (all 128-byte aligned for TMA):
    //   smem_a:     TILE_M * K_LIMBS_PER_CHUNK u64s = 256 = 2048 bytes
    //   smem_b_raw: TILE_K * 1                u64s = 256 = 2048 bytes
    //   smem_b_swz: TILE_N_BITS * K_LIMBS_PER_CHUNK u64s = 64*4 = 2048 bytes
    //   smem_c:     TILE_M u64s = 64 = 512 bytes
    //   bar:        1 u64
    __shared__ alignas(128) uint64_t smem_a[TILE_M * K_LIMBS_PER_CHUNK];
    __shared__ alignas(128) uint64_t smem_b_raw[TILE_K];
    __shared__ alignas(128) uint64_t smem_b_swz[TILE_N_BITS * K_LIMBS_PER_CHUNK];
    __shared__ uint64_t smem_c[TILE_M];
    __shared__ uint64_t bar[1];

    const uint32_t bi  = blockIdx.y;
    const uint32_t bj  = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    const uint32_t row_base = bi * TILE_M;
    const uint32_t col_limb = bj;

    if (row_base >= m || col_limb >= n_lim) return;

    // Initialize accumulator, output scratch, mbarrier.
    int32_t acc[32];
    #pragma unroll
    for (int r = 0; r < 32; ++r) acc[r] = 0;
    if (tid < TILE_M) smem_c[tid] = 0;
    if (tid == 0) mbarrier_init(&bar[0], 1);
    __syncthreads();

    const uint32_t k_chunks = (k + TILE_K - 1) / TILE_K;
    uint32_t phase = 0;

    // ------------------------------------------------------------------------
    // Outer K loop
    // ------------------------------------------------------------------------
    for (uint32_t kk = 0; kk < k_chunks; ++kk) {
        const uint32_t k_offset_limbs = kk * K_LIMBS_PER_CHUNK;
        const int32_t  coord_a_x = static_cast<int32_t>(k_offset_limbs);
        const int32_t  coord_a_y = static_cast<int32_t>(row_base);
        const int32_t  coord_b_x = static_cast<int32_t>(col_limb);
        const int32_t  coord_b_y = static_cast<int32_t>(kk * TILE_K);

        // Issue TMA loads from thread 0; expect transaction bytes.
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&bar[0], TMA_BYTES_A + TMA_BYTES_B);
            tma_load_2d(smem_a,     &tensor_map_a, coord_a_x, coord_a_y, &bar[0]);
            tma_load_2d(smem_b_raw, &tensor_map_b, coord_b_x, coord_b_y, &bar[0]);
        }

        // Wait for TMA to complete.
        mbarrier_wait(&bar[0], phase);
        phase ^= 1;

        // Clear the swizzled-B SMEM region (transpose uses atomicOr).
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            uint32_t idx = tid * 2 + s;  // 0..255
            smem_b_swz[idx] = 0;
        }
        __syncthreads();

        transpose_b_warp(smem_b_raw, smem_b_swz);
        __syncthreads();

        // wgmma SMEM descriptors. With swizzle=0, the values below are
        // "row stride in bytes" for both leading and stride fields; this
        // is a starting estimate, validate against PTX manual.
        uint64_t desc_a = make_smem_desc(smem_a,     /*leading*/ 32, /*stride*/ 32);
        uint64_t desc_b = make_smem_desc(smem_b_swz, /*leading*/ 32, /*stride*/ 32);

        wgmma_fence();
        wgmma_b1_m64n64k256_and(acc, desc_a, desc_b);
        wgmma_commit_group();
        wgmma_wait_zero();

        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Reduce s32 accumulators to F_2 (LSB) and pack into output u64s.
    //
    // Per-thread layout for wgmma m64n64 .row.col .s32 output (PTX manual,
    // "Matrix Fragments for WGMMA"):
    //   warp_id  = tid / 32       (0..3, owns 16-row band)
    //   lane     = tid % 32
    //   row_base = warp_id*16 + lane/4
    //   col_base = (lane & 3) * 2
    //   For col-group g in 0..7 (across N=64 in steps of 8):
    //     acc[g*4+0]: (row_base,     col_base + g*8 + 0)
    //     acc[g*4+1]: (row_base,     col_base + g*8 + 1)
    //     acc[g*4+2]: (row_base + 8, col_base + g*8 + 0)
    //     acc[g*4+3]: (row_base + 8, col_base + g*8 + 1)
    // ------------------------------------------------------------------------
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane    = tid & 31;
    const uint32_t r_base  = warp_id * 16 + (lane >> 2);
    const uint32_t c_base  = (lane & 3) * 2;

    uint64_t bits_r0 = 0, bits_r8 = 0;
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        uint32_t c0 = c_base + g * 8 + 0;
        uint32_t c1 = c_base + g * 8 + 1;
        bits_r0 |= (static_cast<uint64_t>(acc[g*4 + 0] & 1)) << c0;
        bits_r0 |= (static_cast<uint64_t>(acc[g*4 + 1] & 1)) << c1;
        bits_r8 |= (static_cast<uint64_t>(acc[g*4 + 2] & 1)) << c0;
        bits_r8 |= (static_cast<uint64_t>(acc[g*4 + 3] & 1)) << c1;
    }
    atomicXor(reinterpret_cast<unsigned long long*>(&smem_c[r_base]),
              static_cast<unsigned long long>(bits_r0));
    atomicXor(reinterpret_cast<unsigned long long*>(&smem_c[r_base + 8]),
              static_cast<unsigned long long>(bits_r8));

    __syncthreads();

    if (tid < TILE_M) {
        uint32_t out_row = bi * TILE_M + tid;
        if (out_row < m && col_limb < n_lim) {
            c[out_row * n_lim + col_limb] = smem_c[tid];
        }
    }
}
