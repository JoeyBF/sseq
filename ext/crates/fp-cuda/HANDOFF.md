# fp-cuda handoff — H100 validation of the Phase 3–7 batch

> **RESOLVED 2026-06-15 (H100 NVL, sm_90, CUDA 13.0 driver / 12.8 toolkit).**
> The whole batch is **hardware-validated and bit-exact.** PTX JITs at load; the
> dynamic-SMEM opt-in (~122 KB at STAGES=3) and all three TMA descriptors are
> accepted. `matmul_b1_demo` (64…8192) and `bench_kernel_only` (4096…32768, incl.
> a full 32768³ CPU cross-check) all match the CPU path bit-for-bit, so the
> swizzle (`5c40830`), n256 bit-pack (`af251a5`), and TMA store (`0dcd0ff`) paths
> are all sound — no bisect needed. Kernel-only throughput peaks at ~5,800 binary
> TOPS @16384 (~50–58× the ~100-TOPS pre-swizzle baseline; see README table).
> The >16384 throughput drop is diagnosed (not power/compute-bound: 136 W of
> 310 W, SM 0–12 %): the kernel is HBM-bandwidth bound on **L2 residency of B**
> — each B column-panel is reused across every M-tile, so once `K*N/8` > 50 MB
> L2 (true at 32768²: 134 MB) B is re-streamed from HBM per M-tile.
> `examples/bench_shapes` proves it with equal-FLOPs shapes. This is exactly
> what the next rungs fix: persistent kernel + tile rasterization (keep the
> active working set in L2) and clusters + TMA multicast (one HBM read per
> B-panel per cluster). The validation steps below are kept for the record.
>
> Server-box setup notes for next time: the host layer now uses `cudarc`
> (stable Rust, dynamic-loads the driver), so **stable** rustc works — no
> nightly, no `cuda-core`, no `libclang-dev`/bindgen needed. `nvcc` is still
> required at build time (PTX) and a CUDA driver at runtime. The shared
> `target/` dir can throw a transient `Stale file handle (os error 116)` — just
> retry the cargo command.

Note to the Claude instance running this on the GPU server. This branch
(`fp_cuda_hopper`) has a batch of kernel optimizations that were written
**code-only, with no GPU available** — they compile (nvcc → PTX + rustc) but
have **not been validated on hardware**. Your job is to validate (and tune).

## What's in the batch (newest first)

| Commit | Change | If correctness breaks, suspect this for… |
|--------|--------|------------------------------------------|
| `0dcd0ff` Phase 7 | TMA bulk output store (S2G); `sC` row-major; C padded to whole 256-col groups; kernel takes a C tensor map instead of a raw ptr | garbage in the rightmost columns → `n_padded_lim` trim/readback or store coords |
| `d66beea` Phase 6 | `STAGES = 3` deeper K-pipeline | (low risk; logic is depth-agnostic) |
| `77e4345` Phase 5 | `setmaxnreg` dec(40)/inc(216) producer/consumer | launch failure "too many registers" → lower CONSUMER_REGS |
| `af251a5` Phase 4 | widen to one `m64n256k256` per k-step (was 4× `m64n64k256`) | wrong bits everywhere → the n256 fragment→limb bit-pack |
| `5c40830` Phase 3 | 128B swizzle on both operands + pipelined wgmmas (one commit/wait per stage, scale-D=1 accumulate) | wrong bits everywhere → swizzle desc constants |

Everything before `5c40830` (`e38263b` "Step (d)") is the last hardware-validated
state (~100 binary TOPS). `git bisect` against that if needed.

## Validate in this order (each gates the next)

1. `cargo build -p fp-cuda` — first real check that the **PTX JITs on the
   device**. setmaxnreg reg budget, the dynamic-SMEM opt-in
   (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`, ~122 KB at STAGES=3),
   and all three TMA descriptors are only checked at module-load / launch — not
   at compile. If load fails here, read the driver error; it's usually the SMEM
   attribute or a TMA descriptor field.
2. **64×256×64 identity product** — the one fast falsifier. Exercises one A
   tile, one 256-col B tile, one `m64n256k256`, and the TMA store. Use the
   existing `examples/matmul_b1_demo.rs` 64-case, or add a tiny identity test.
   If it fails, bisect with the table above — the two prime suspects are the
   swizzle constants (`5c40830`) and the n256 bit-pack (`af251a5`).
3. `cargo run -p fp-cuda --example matmul_b1_demo` — full bit-exact CPU↔GPU
   sweep (64…8192).
4. `cargo bench -p fp-cuda` — TOPS vs the ~100 baseline; confirm outputs stay
   bit-equal.

## Key facts / where the risky constants live

- All in `cuda_kernels/matmul_b1.cu` and `src/lib.rs`. `STAGES` is duplicated in
  both and **must match**.
- wgmma 128B K-major descriptor (kernel `make_desc`, constants `DESC_LBO=16`,
  `DESC_SBO=1024`, `DESC_SWIZ=1`) and the +32 B (`c*KSUB_U64`) per-k256 advance
  are derived from CUTLASS `make_gmma_desc<Major::K>` / `LayoutType::B128`
  (`include/cute/atom/mma_traits_sm90_gmma.hpp`). **Not hardware-checked.** If
  the identity test fails, re-derive against that file / the device's actual
  swizzle, possibly by printing a small SMEM tile.
- n256 bit-pack assumes the `m64n256` accumulator fragment is the `m64n64`
  layout tiled along N (register group `gi` in 0..32 → output cols
  `[gi*8, gi*8+8)`). Verify against the PTX ISA "Matrix Fragments for wgmma"
  for `m64n256.s32` if bits are scrambled in a structured way.
- TMA: A box = 64 rows, B box = 256 rows, both inner = 128 B (the swizzle
  width). C store box = `[NG*2 UINT32, 64]`, swizzle NONE. Dynamic SMEM base is
  `__align__(128)` — required for TMA; if loads fault, double-check that holds.

## Tuning knobs (once correct)

- `STAGES` (2/3/4) — latency vs occupancy. 2 → 2 CTAs/SM (82 KB), 3 → 1 CTA/SM
  (122 KB). Sweep it.
- `PRODUCER_REGS` / `CONSUMER_REGS` in the kernel — must be multiples of 8 in
  [24,256]; `128*(prod+cons) ≤ 65536`.

## Environment / workflow gotchas (from the dev box; may differ on server)

- Building writes to a shared target dir outside the sandbox, so `cargo build`
  needed `dangerouslyDisableSandbox`. **Do not** override `CARGO_TARGET_DIR`.
- Run `cargo fmt` after editing any Rust (project convention).
- This crate is excluded from workspace `default-members`; always use `-p fp-cuda`.

## Not yet done (remaining worklog rungs)

Thread-block clusters + TMA multicast, then a persistent kernel + tile
scheduler. Deferred deliberately until this batch is validated and we have
throughput numbers to guide the design choices (cluster size, schedule).

— Report results back to the user; if the identity test passes, the swizzle +
n256 + TMA-store path is sound and we can proceed to clusters.
