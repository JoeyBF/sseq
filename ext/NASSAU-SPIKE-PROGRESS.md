# Motivic Nassau spike — progress log

Companion to `NASSAU-SPIKE-PLAN.md`. Records the baseline, per-milestone exit
notes, and final numbers. Everything is built/run with
`RUSTFLAGS="-C target-cpu=x86-64-v3"`; touched files formatted with
`rustfmt +nightly --edition 2024`.

## M0 — Baseline + harness  ✅

- Base branch: `claude/nassau-spike-impl-8q3cj6`, cut from `claude/nassau-spike`
  (which cherry-picked `ctau.rs` onto the PR1 algebra engine). `cargo build` /
  `cargo test` clean under the required RUSTFLAGS.
- Harness: `ext/examples/motivic_ctau_baseline.rs` resolves the trivial module
  `k = F₂` over `A_C/τ` (`CTauAlgebra`) with the **generic** engine
  (`Resolution<FiniteChainComplex<FDModule<CTauAlgebra>>>` +
  `compute_through_stem`) and prints per-`(s, t)` generator ranks.
- **Output contract** the new engine must match: the set of
  `number_of_gens_in_bidegree(s, t)` over the box. This is what M5 compares.
- Golden fixture: `ext/tests/fixtures/motivic_ctau_golden_ranks.txt`, box
  **n ≤ 40, s ≤ 22** (per request, larger than the plan's n≤30/s≤16 "to be
  sure"). **789 total generators.**
- Baseline resolution-phase wall-clock on this box: **~1.83 s** (generic engine,
  release, single run; the whole box, not just one bidegree).

## M1 — Signature combinatorics  ⏳ (in progress)

## M2 — B-trivial signature-graded multiplication

## M3 — MVP inductive step (B = A(0))

## M4 — General Algorithm 2 + generic fallback

## M5 — Validation & benchmark
