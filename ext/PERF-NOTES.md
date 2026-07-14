# Signature engine — performance investigation

## TL;DR (corrected)

**There is no per-operation performance gap between the generic
`SignatureResolution` and master's hand-tuned `nassau.rs`.** The apparent 14×
was a benchmark artifact: master's `compute_through_stem` computes a **stem
region** while ours computes a full **rectangle** (~4× more bidegrees). On the
*same* region they are on par.

## Evidence (classical p=2 sphere, `nassau_bench -- 2 60 30`, serial, release)

Instrumented the shared `ModuleHomomorphism::get_partial_matrix` (GPM) primitive
so all engines are measured identically.

**Master computing a stem region vs ours computing a rectangle** (the original,
unfair comparison):

| engine | region | time | GPM calls | GPM inputs |
|---|---|---:|---:|---:|
| master nassau | **stem** | 1.36 s | 10 500 | 320 k |
| signature (ours) | **rectangle** | 18.9 s | 20 377 | 1.24 M |

**Both computing the rectangle** (fair):

| engine | time | GPM time (share) | calls | inputs | µs/input |
|---|---:|---:|---:|---:|---:|
| master nassau | 18.1 s | 15.1 s (83%) | 22 682 | 1.24 M | 12.1 |
| signature (ours) | 18.9 s | 14.0 s (74%) | 20 377 | **1.24 M** | 11.3 |
| generic | 31.4 s | — (doesn't use GPM) | 0 | 0 | — |

Identical input counts, matching per-input cost, matching GPM share. Ours is even
marginally cheaper in GPM. And on the same region the signature engine is
**~1.65× faster than the generic engine** (18.9 s vs 31.4 s) — the shrink
shortcut is a real win over the generic path, and on par with master.

## So what actually to improve

1. **Compute a stem region, not a rectangle (the real ~4× win).** Our
   `SignatureResolution::compute_through_stem(max_s, max_t)` loops the full
   rectangle `{s ≤ max_s, t ≤ max_t}`; master computes only the stem region
   `{t − s ≤ n}` (plus dependency slack), which for a chart is ~4× fewer
   bidegrees. Matching that recovers essentially all of the apparent gap — it is
   about computing the *right region*, not making the engine faster. (Mind the
   dependency: `(s,t)` needs `(s−1,t)`, i.e. stem `n+1` — master's wavefront
   handles the staircase / prunes above the vanishing edge.)

2. **Allocation-free signatures (~10%, real).** `signature_mask` still does
   79.9M `basis_element_signature` calls, each allocating a `Vec` for
   `MilnorAlgebra`. Bucketing basis elements by signature once per (module,
   degree) + a packed `u64/u128` signature key removes these. Worth doing but
   it is ~10%, not the headline.

3. GPM itself (74%) is inherent — master spends the same share there. No change.

## Instrumentation (TEMP — remove after)

- `algebra::module::homomorphism::{GPM_CALLS, GPM_INPUTS, GPM_NANOS}` on the
  shared `get_partial_matrix` (in `crates/algebra/.../homomorphism/mod.rs`).
- `ext::motivic_nassau::prof` (per-phase timers in `signature_mask` /
  `step_general`).
- The `[5]`/`[6]` blocks in `examples/nassau_bench.rs`.

All add per-call `Instant::now` overhead — **remove once the region change lands**.
The `assert!(d²=0)` + the `tmf_*`/`classical_*`/`odd_primary_*` rank cross-checks
guard correctness through any rewrite.
