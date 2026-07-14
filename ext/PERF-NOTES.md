# Signature engine — performance investigation

## TL;DR (fix landed)

**There is no per-operation performance gap between the generic
`SignatureResolution` and master's hand-tuned `nassau.rs`.** The apparent 14×
was a benchmark artifact: master's `compute_through_stem` computes a **stem
region** while ours computed a full **rectangle** (~4–11× more bidegrees). On
the *same* region they are on par.

**Fixed:** `SignatureResolution::compute_through_stem` now computes the `(n,s)`
stem rectangle `{n = t−s ≤ max_n, s ≤ max_s}` (gate `t − s > max_n` → skip),
matching master. Measured on the classical sphere (`nassau_bench`, release,
`target-cpu=native`, serial):

| box | signature (before) | signature (after) | generic | master |
|---|---:|---:|---:|---:|
| `2 60 30` | 18.9 s | **1.66 s** | 1.77 s | 1.43 s |
| `2 80 40` | — | **15.6 s** | 23.3 s | 13.8 s |

Identical GPM input counts to master (320 150 / 1 372 178). The signature engine
now beats the generic engine and sits within ~15% of master; the residual is the
`signature_mask` allocations (item 2 below).

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

1. **DONE — compute the `(n,s)` rectangle, not the constant-`t` diagonal (the
   real ~4–11× win).** Our `compute_through_stem(max_s, max_t)` looped `{s ≤
   max_s, t ≤ max_t}`. In `(n = t−s, s)` coordinates that is *not* a rectangle:
   `t ≤ max_t` is the **slope-(−1) line** `n + s = max_t`, so the region fanned
   out to stem `max_n + max_s` at `s = 0`. Master's `compute_through_stem`
   instead gates on `stem ≤ max_n` (`distance = max.n() − b.n()` in `nassau.rs`),
   giving the **`(n,s)` rectangle** `{n ≤ max_n, s ≤ max_s}` — a **vertical** cut
   at `n = max_n`.

   The extra region is the low-`s`, high-`n` triangle `{n > max_n, n+s ≤
   max_t}`. It is only ~25% more *bidegrees* but ~4–11× more *work*, because
   `dim(A)` grows (≈exponentially) with degree: `C_1` at stem ~89 is enormous
   even though it has few generators. That corner is pure waste for a chart that
   only wants stems ≤ `max_n`.

   **Fix (landed):** gate the loop on `t − s > max_n → continue`. The dependency
   `(s,t) ← (s−1,t)` reaches one stem higher (`n+1`); at the top boundary that
   neighbour is skipped, but its degree-`t` generators map to *nonzero* images so
   they are never in `ker d_{s−1}` — the partial matrix over the smaller target
   is still correct, and the skipped cell lies at stem `n+1 > max_n`, outside the
   reported region. This is exactly the phantom-boundary read master's parallel
   wavefront makes at its `distance == 1` cells (it `send`s the token without a
   full step). No wavefront needed in the serial engine: the plain `for t { for s
   }` order already satisfies the true minimal dependency `(s,t) ← (s,t−1),
   (s−1,t−1)`, both within the region.

2. **Allocation-free signatures (~10%, real).** `signature_mask` still does tens
   of millions of `basis_element_signature` calls (62M on `2 80 40`), each
   allocating a `Vec` for `MilnorAlgebra`. Bucketing basis elements by signature
   once per (module, degree) + a packed `u64/u128` signature key removes these.
   Worth doing but it is ~10%, not the headline — this is what remains of the gap
   to master (15.6 s vs 13.8 s on `2 80 40`).

3. GPM itself (73%) is inherent — master spends the same share there. No change.

## Instrumentation (TEMP — remove after)

- `algebra::module::homomorphism::{GPM_CALLS, GPM_INPUTS, GPM_NANOS}` on the
  shared `get_partial_matrix` (in `crates/algebra/.../homomorphism/mod.rs`).
- `ext::motivic_nassau::prof` (per-phase timers in `signature_mask` /
  `step_general`).
- The `[5]`/`[6]` blocks in `examples/nassau_bench.rs`.

All add per-call `Instant::now` overhead — **remove once the region change lands**.
The `assert!(d²=0)` + the `tmf_*`/`classical_*`/`odd_primary_*` rank cross-checks
guard correctness through any rewrite.
