# Handoff: CubeCL `get_partial_matrix` kernel for Nassau's Milnor multiply (p = 2)

This document is a self-contained brief for building a GPU kernel that offloads the Milnor
multiplication work in Nassau's algorithm (`S` at the prime 2). It was written from a CPU-only
container that **cannot run GPU code**, so all the design, measurements, and the exact CPU reference
to port are captured here for a machine that *can* run a GPU (yours).

Everything below is on branch **`claude/algebra-benchmarking-suite-zj5sgm`** of `JoeyBF/sseq`
(`ext/` workspace). Pull that branch first.

---

## 0. TL;DR — what to build

A CubeCL kernel that computes, in one launch, all the `Sq(R) · s` products issued by a single
`ModuleHomomorphism::get_partial_matrix` call — the natural, provably-maximal GPU batch (see §5).
Port the CPU **admissible-matrix** algorithm in `MilnorAlgebra::multiply_basis_element_by_element_2`
(one workgroup per operation `R`: enumerate `Sq(R)`'s admissible matrices once, test every element
term against them) and index outputs with the hash-free **`seqno`** table instead of a hashmap
(a GPU can't carry a `HashMap`). Accumulate into bit-packed F₂ output vectors with **atomic XOR**.

Validate against the CPU reference bit-for-bit (§8), then A/B end-to-end against the current CPU
per-term path with `nassau_e2e` (§8).

---

## 1. Environment reality (important)

- **Confirmed compiling** here: `cubecl = "0.10.0"` with the **`wgpu`** feature. `cargo check` builds
  clean (wgpu 29, naga, etc.). The wgpu runtime needs a device only at *runtime*, so it compiles on a
  headless box. A trivial `#[cube]` kernel using `Array<u32>` and `^` (XOR) type-checks.
- **`cpu` runtime (`cubecl-cpu`)**: pulls `tracel-llvm-bundler`, which **downloads a prebuilt LLVM/MLIR
  bundle from GitHub releases**. That was blocked here (HTTP 403 through the sandbox proxy). On your
  machine with normal network it should fetch fine, and gives you a **CPU execution backend for
  correctness tests without a GPU**. Worth enabling for CI-style validation.
- On your machine pick the runtime that matches your hardware: `wgpu` (portable — Metal on macOS,
  Vulkan/DX12 elsewhere), `cuda` (NVIDIA), or `hip` (AMD). `wgpu` is the safe default.
- (Container-only noise you can ignore: `-C target-cpu=native` currently SIGILLs rustc/LLVM on the
  `algebra` release build *in the sandbox*. Not a code issue; native builds are fine on real hardware.
  Mentioned only so you don't chase it.)

---

## 2. Where things live (branch `claude/algebra-benchmarking-suite-zj5sgm`)

Key file: **`ext/crates/algebra/src/algebra/milnor_algebra.rs`**

| Item | Symbol | Line (approx.) |
|---|---|---|
| CPU reference to port | `MilnorAlgebra::multiply_basis_element_by_element_2` (`pub`) | ~1650 |
| Admissible-matrix enumerator | `struct AdmissibleMatrix`, `::new`, `::next` | ~1890 |
| Hash-free index (build) | `MilnorAlgebra::compute_seqno_tables` (`pub`) | ~1368 |
| Hash-free index (lookup) | `MilnorAlgebra::seqno` (`pub`) | ~1433 |
| Flat table storage | `struct SeqnoTables { max_degree, width, g }` | ~664 |
| Applicability gate | `MilnorAlgebra::seqno_applicable` | ~1354 |
| Basis enumeration order | `compute_ppart` (~1298), `generate_basis_2` (~1503) | — |
| Active CPU path (per-term) | `MilnorAlgebra::multiply_basis_element_by_element` | ~959 |
| Correctness oracle (test) | `admissible_multiply_agrees_with_reference` | tests mod |
| Occupancy probe | `mod profile` (`scope_*`, `record_call`, `report`) | ~14–256 |

Call-path files:
- `ext/crates/algebra/src/module/free_module.rs` — `FreeModule::act` (line ~180) → per generator
  block calls `multiply_basis_element_by_element_unstable`.
- `ext/crates/algebra/src/module/homomorphism/free_module_homomorphism.rs` — `apply_to_basis_element`
  (line ~55) → `target.act(...)`.
- `ext/crates/algebra/src/module/homomorphism/mod.rs` — `get_matrix` (~122), `get_partial_matrix`
  (~150). **This is the launch boundary.**
- `ext/src/nassau.rs` — `get_partial_matrix` call sites (lines ~621, ~691).
- `ext/examples/nassau_e2e.rs` — end-to-end timing harness (see §8).

Constants/types: `PPartEntry = u32`; a p_part is `PPart = Vec<u32>`; row width for tables/matrices is
`MAX_XI_TAU` (= `MAX_MULTINOMIAL_LEN`, in `combinatorics.rs`), the max number of ξ-entries. At p = 2,
`xi_degrees(2)` = Mersenne numbers `[1, 3, 7, 15, 31, …]`, length `MAX_XI_TAU`.

Session commits (context for how we got here): `0a41291` reapply admissible multiply, `ba2a858` trim
p=2 hot path, `28ecb04` profiling, `c4995bd`/`5344984` seqno + flat storage, `08928d8` revert
admissible off CPU (kept as GPU model), `327c67f`/`9b2ed14` realizable occupancy probe.

---

## 3. The exact call chain get_partial_matrix drives

```
get_partial_matrix(degree, inputs)              // homomorphism/mod.rs:150  — LAUNCH SCOPE
  └ (parallel over rows) apply_to_basis_element(row, 1, degree, inputs[i])   // free_module_homomorphism.rs:55
       │  decompose source basis elt → (op_degree, op_index, gen_degree, gen_index)
       └ target.act(result_row, 1, op_degree, op_index, gen_degree - shift, d(gen))  // free_module.rs:180
            └ for each nonzero generator block of d(gen):
                 multiply_basis_element_by_element_unstable(out_slice, 1,
                     op_degree, op_index,            //  == R  (fixed within this act)
                     block_degree, block_slice,      //  == s  (a general algebra element)
                     gen_deg)                          // excess; U=false ⇒ ignored
                 └ (U=false shim) → multiply_basis_element_by_element(out, 1, R_deg, R_idx, s_deg, s)
                      └ [CPU default now] per-term PPartMultiplier sweep
                      └ [GPU model]       multiply_basis_element_by_element_2  ← PORT THIS
```

So the fundamental kernel unit is: **fixed operation `R = (op_degree, op_index)` × a general element
`s` (a sum of `Sq(S_j)` terms) → add `Sq(R)·s` to an output F₂ slice.** Within one launch there are
many such `(R, s)` products; `R` repeats across rows that are (same operation)×(different generator).

---

## 4. The math to port (`multiply_basis_element_by_element_2`, p = 2)

`Sq(R) · Sq(S)` at p = 2 is a sum of `Sq(T)` basis elements, one per **admissible matrix** `X` of
`Sq(R)` compatible with `S`. The CPU reference enumerates all admissible matrices of `Sq(R)` **once**
and, for each, tests every term `Sq(S_j)` of `s`. This is the amortization that suits a GPU workgroup.

**Per (matrix `X`, term `S`) test and output assembly** (from the reference, lines ~1729–1782):
- Matrix `X` exposes, per column `j`: `col_sums[j]` (Σ down the column) and `masks[j]` (the anti-
  diagonal bit pattern). Rows are indexed by entries of `R`.
- Compatibility test for column `j` against `S`'s entry `basis[j]`:
  - reject if `col_sums[j] > basis[j]` (not enough), **or**
  - reject if `(basis[j] - col_sums[j]) & masks[j] != 0` (bit conflict — this is the mod-2
    multinomial coefficient being even, i.e. the term vanishes).
- If all columns pass, the output p_part entry at position `j` is
  `(basis[j] - col_sums[j]) | masks[j]` (disjoint bits, so `|` == `+`). Handle the tails where
  `basis` and the matrix have different lengths exactly as the reference does (lines ~1748–1773),
  then trim trailing zeros (~1775).
- Look up the resulting `working.p_part`'s index and `result.add_basis_element(idx, 1)` — the `+ 1`
  is **XOR** in F₂, so repeated hits on the same output index cancel mod 2. **On GPU this must be an
  atomic XOR into the bit-packed output** (multiple `(X, S)` pairs, possibly across threads, can land
  on the same output index).

**Admissible matrix enumeration** — `AdmissibleMatrix::new(&R.p_part)` then loop `while matrix.next()`
(lines ~1899–1987). `new` sizes `cols = max over entries of bit-length(entry)`, `rows = len(R)`,
seeds column 0 with `R`. `next` is a carry-style increment over the matrix cells maintaining
`col_sums` and `masks` incrementally; it returns `false` when exhausted. This is inherently sequential
per `R` — **run it once per workgroup (one workgroup ⇒ one `R`), materialize the matrices (or stream
them) into shared memory, and parallelize the term tests across the workgroup's threads.** The count
of matrices per `R` is small-to-moderate; the parallel width comes from the *terms* (see §5).

**Special cases to keep:**
- `Sq(∅) = 1` (empty `R.p_part`): output is `s` unchanged (~1678). Common — the identity operation.
- Sparse fallback: the CPU reference peeks the first two terms and, for `< 2` terms, uses the per-term
  path (single-term amortizes nothing). On GPU you'll batch many `(R, s)` at once so this specific
  gate may not apply, but note that **~31% of `s` are single-term and ~44% are ≤ 2 terms** — plan the
  work layout so tiny `s` don't each get a full workgroup idling.

---

## 5. Batching / occupancy facts (measured, p = 2, real resolutions)

Batch **per `get_partial_matrix`**. This was measured with the `MILNOR_PROFILE=1` occupancy probe (§7).

- The multiply is the bottleneck (~66% of resolution time; **not** linear-algebra bound).
- Regime densifies with scale: mean terms per `s` grows 6.0 (stem 100) → 9.0 (stem 120), but stays
  bimodal (big single-term head + dense tail).
- **Realizable occupancy** (grouping work by `R` *within one `get_partial_matrix` launch*, which is
  all a kernel can fuse without buffering across the streaming algorithm) at stem (100,52): mean
  **3,917 element-terms/launch** (max 86k). Fraction of term-work in `R`s reaching ≥ W terms in one
  launch: **W=64 → 82.8%, W=128 → 74.9%, W=256 → 59.8%, W=1024 → 25.7%**. It climbs with scale
  (W=256 was 0% at stem 40, 31% at stem 80, 60% at stem 100).
- **Widening the scope does NOT help.** Merging all per-signature builds of a differential at one
  bidegree gave 8.5× bigger launches but *bit-identical* coverage: an operation `Sq(R)`'s subalgebra
  signature is essentially a function of `R`, so all of an `R`'s work already lands in one signature's
  `get_partial_matrix`. Corollary: the per-signature builds are **disjoint, not redundant** — "compute
  the full matrix once and slice" buys nothing. So `get_partial_matrix` is the maximal, non-redundant
  batch. Don't build cross-launch buffering.

**Design implication:** one workgroup per distinct `R` in the launch; threads within it stream the
`R`'s element terms across the (shared) admissible matrices. ~60% of term-work fills a 256-wide
workgroup on same-`R` amortization alone; the remaining smaller-`R` work is covered by the sheer count
of independent `(R, s)` products per launch (thousands) — assign leftover threads/warps across
different `R`s so nothing idles.

---

## 6. Data layout for the device

Upload once per algebra (rebuilt as `compute_basis`/`compute_seqno_tables` grow):

- **seqno flat table** — `SeqnoTables { width, g: Vec<usize> }`, row-major `g[e * width + h]`.
  `seqno(p_part)` (line ~1433) is: `cur_d = Σ p_part[i]·xi[i]`; then for `h` from high to low with
  `p_part[h] = r > 0`: `rank += g[cur_d*width + h] - g[(cur_d - r·xi[h])*width + h]; cur_d -= r·xi[h]`.
  Pure integer arithmetic + array reads — **direct GPU port, no hashing.** Applies only when
  `seqno_applicable()` (p=2, trivial profile, stable), which is exactly Nassau's case.
  `xi_degrees(2)` (the Mersenne `[1,3,7,…]`) is a tiny constant array — upload or hard-code.
- **Per-launch input buffers**: for each row/product, the operation `R` (its `p_part`, fixed width
  `MAX_XI_TAU`) and the element `s` as a list of term `p_part`s (or indices + a shared basis table).
  A p_part is ≤ `MAX_XI_TAU` `u32`s; pad to fixed width for coalesced access.
- **Output**: F₂ vectors bit-packed into `u32` (or `Atomic<u32>`) arrays, one per matrix row; write via
  `atomic_xor(word, 1 << bit)` at `idx = seqno(working_p_part)`. Match the crate's `FpVector` p=2 bit
  order (LSB-first within a `u64`/`u32` limb — verify against `fp`).

Note `basis_element_from_index`/`basis_element_to_index` are `pub` on `MilnorAlgebra`; the basis table
for a degree is contiguous and in `seqno` order (that's the whole point — `seqno(elt.p_part) == index`,
proven by the `seqno_matches_enumeration_order` test to degree 100).

---

## 7. The occupancy probe (how to regenerate the numbers)

Build any driver with the env var set, run **serially** (the probe's scope accounting assumes it):

```
MILNOR_PROFILE=1 cargo run --release --example nassau_e2e -- 100 52 1
```

Prints three views at the end: global R-occupancy, **realizable per-launch**, and **merged
per-bidegree**. The probe is `#[cfg(milnor_profile)]` (build.rs reads `MILNOR_PROFILE`), zero cost
otherwise. Recording is wired onto the active multiply path and scope boundaries are in
`get_matrix`/`get_partial_matrix`. Use it to size workgroups and to see where a new resolution's mass
sits. (`report()` is called at the end of `nassau_e2e`.)

---

## 8. Validation & benchmarking (do this on the GPU box)

**Correctness oracle** — `admissible_multiply_agrees_with_reference` (tests module of
`milnor_algebra.rs`) already checks `multiply_basis_element_by_element_2` bit-for-bit against the
`PPartMultiplier` reference (`multiply_basis_elements`) for single- and multi-term `s`, including mod-2
cancellation, to degree 32. **Mirror it for the kernel**: for every `(R, s)` in a degree range, compare
the kernel's output F₂ vector to the CPU `multiply_basis_element_by_element_2` output. If you enable
the `cpu` cubecl runtime, this runs without a GPU.

**End-to-end A/B** — the CPU baseline is the current default (per-term sweep). `ext/examples/nassau_e2e.rs`
times a full `construct_nassau(("S_2","milnor"))` resolution to `(stem, filtration)` and reports the
best of N runs. Wire the GPU path behind a feature/flag, then compare `best=` at e.g. `100 52 3` and
`120 62 3`. The GPU has to beat the CPU per-term path *including* host↔device transfer and the seqno
uploads — so keep buffers resident across the resolution and only restage what grew.

Reference CPU timings (this sandbox, serial, no GPU): stem (80,42) ≈ 13.4 s, (100,52) ≈ 115 s. Your
box will differ; use it only to sanity-check the harness, not as an absolute target.

---

## 9. Suggested staging (each stage compiles + validates before the next)

1. **Toolchain slice.** Add `cubecl` behind a new `gpu` feature on the `algebra` crate
   (`cubecl = { version = "0.10", optional = true, features = ["wgpu"] }`, `gpu = ["dep:cubecl"]`).
   Land a trivial `#[cube]` kernel (F₂ XOR of two arrays) + a host launch + a unit test. Proves the
   runtime end-to-end on your hardware. (This exact skeleton compiled in the sandbox.)
2. **seqno on device.** Port `seqno` as a `#[cube]` function; test it against the CPU `seqno` over a
   full degree's basis (must be the identity permutation `seqno(elt.p_part) == index`).
3. **Single-`R` multiply.** One workgroup: enumerate `Sq(R)`'s admissible matrices on the host (or a
   single device thread) into a buffer, then parallelize the term tests + atomic-XOR output across the
   workgroup. Validate against `multiply_basis_element_by_element_2` for one `(R, s)`.
4. **Batched launch = one `get_partial_matrix`.** Marshal all `(R, s)` products of a real
   `get_partial_matrix` call, one workgroup per distinct `R`, leftover warps mopping up small `R`s.
   Validate the whole output matrix against the CPU `get_partial_matrix`.
5. **Wire into Nassau + A/B.** Feature-flag the GPU path in `FreeModule::act`/`get_partial_matrix`,
   keep device buffers resident, and A/B with `nassau_e2e`. Iterate on transfer/occupancy using §7.

---

## 10. Gotchas

- **F₂ is XOR.** `add_basis_element(idx, 1)` = XOR the bit. Outputs collide across `(matrix, term)`
  pairs and cancel mod 2 — the device accumulator must be atomic XOR, not add.
- **Trailing-zero canonicalization.** `working.p_part` is trimmed of trailing zeros before indexing
  (line ~1775); `seqno` assumes a canonical (trimmed) p_part. Don't feed it non-canonical inputs
  (this bit us once — `try_basis_element_to_index` rejects trailing-zero p_parts).
- **`seqno` only applies in Nassau's regime** (`seqno_applicable`: p=2, trivial profile, stable). Good,
  because that's exactly this workload — but assert it before uploading the table.
- **Fixed widths.** p_part width ≤ `MAX_XI_TAU`; the seqno table row width is `xi_degrees(2).len()`.
  Pad device buffers to these constants for coalesced access.
- **`AdmissibleMatrix::next` is sequential.** Enumerate per `R` once; the parallelism is over element
  terms, not over matrices. Don't try to parallelize `next` itself.
- **The admissible path is a CPU regression (3–8%)** but the correct GPU shape — that's *why* it's
  kept `pub` and tested rather than deleted. Don't "fix" the CPU default back to it.
- **Keep buffers resident.** The seqno table and basis tables are rebuilt monotonically as degree
  grows; upload deltas, don't restage every launch, or transfer will eat the win.

---

*Written from a sandbox without GPU access; the CubeCL `wgpu` compile path and the trivial kernel were
verified to build, but nothing here was executed on a GPU. Validate stage-by-stage on real hardware.*
