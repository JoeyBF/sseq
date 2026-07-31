# Motivic pipeline perf — measurements (2026-07-31)

Logs in this dir (worktree `.claude/worktrees/motivic-seqsee/ext/perf/`):

- `product-scaling.log` — `motivic_perf` example: one h₂ product lift per box, with
  correction/product-op/cache counters. Reproduce: `cargo run --release --features
  concurrent --example motivic_perf`.
- `chart-n50-full.log` — full `RUST_LOG=info` trace of `chart_motivic_seqsee` at n=50.
- `chart-n50-phases.log` — the phase-level spans grepped out of the full trace.

## The scaling of a single h₂ product lift

| n  | resolve | lift_product | corrections | product_ops | hit%  | µs/corr | ops/round |
|----|---------|--------------|-------------|-------------|-------|---------|-----------|
| 40 | 0.43 s  | 0.38 s       | 16 329      | 0.30 M      | 98.9% | 23.2    | 18        |
| 50 | 2.00 s  | 1.57 s       | 39 722      | 2.31 M      | 99.7% | 39.6    | 58        |
| 60 | 7.20 s  | 4.58 s       | 75 270      | 16.5 M      | 99.9% | 60.8    | 219       |
| 70 | 28.9 s  | 15.1 s       | 154 829     | 89.0 M      | 100%  | 97.2    | 575       |

Growth per +10 stems: **corrections ~2.0×, product_ops ~7×, ops/round ~3×.**
Empirically product_ops ~ n^9.7; extrapolates to ~4–5 billion ops for one lift at
n=100 (matches the observed 633 s in `n100.log`).

## Diagnosis

The dominant cost is the **volume of A_C product operations in the τ-adic correction
loop** — and, crucially:

1. **Not cache misses.** Hit rate is ~100%. The A_C Milnor product cache
   (`crates/algebra/.../milnor.rs`: `RwLock` on block lookup, lock-free `OnceLock`
   reads) is doing its job. Products are cheap *lookups* (~170 ns each). There are just
   an astronomical number of them.
2. **Not correction depth.** `corrections` (order-by-order rounds) grow only ~n^3.7.
3. **It's the work *per* round.** `ops/round` explodes 18 → 575 (≈3× per +10 stems):
   each correction round's `compose_into` walks |support| × |differential support| ×
   |product-term expansion| cached lookups, and all three grow with internal degree.
   That product is the n^9 term.

So: we build φ_a as a full A_C chain map over the **entire padded box, every
filtration, to full τ-convergence**, then read a thin slice (the 1⊗b augmentation
coefficient at in-box product targets). The waste is doing exploding per-round support
walks for cells/orders the chart never reads.

Ruled out: faster products (already ~100% cached), cross-lift parallelism (cores
already saturated, ~96 threads), lock contention (lock-free reads).

## Levers, ranked

1. **Restrict the lift to what in-box products need** (cone-pruning + weight-order
   truncation). Cutting cells cuts rounds; cutting to lower degree shrinks the
   per-round support walk — attacks the n^9 term at both factors. Highest leverage,
   needs a correctness argument (truncated φ_a is no longer a full chain map beyond the
   cut, but must still give correct in-box products).
2. **Shrink the per-round `compose_into` cost.** It re-walks operation×operation pairs
   each round; batching/memoizing the composite structure at a coarser grain than
   single products could cut the constant. Medium leverage, safe-ish.
3. **Prune trivially-zero cells** (empty target bidegree, zero augmentation seed).
   Safe, moderate.
4. **Persist product lifts to disk** (like the differential lift). Amortizes re-runs.

## Chart context (n=50)

Assembly phases (`chart-n50-phases.log`): differential `lift` 1.40 s, the three h₀/h₁/h₂
product lifts 2.06 + 1.74 + 1.56 = **5.4 s** (the bulk), SS build ~10 ms, `chart_dots`
sub-ms. Products dominate, and the gap widens with n.
