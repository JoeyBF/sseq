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

## M1 — Signature combinatorics  ✅

- `ext/src/motivic_nassau.rs` (feature `sig-nassau`): [`MotivicSubalgebra`] —
  a finite admissible `B ⊂ A_C/τ` as a **polynomial profile** `p_profile[i]`
  (bound on `ξ_{i+1}`) plus an **exterior length** `q_len` (`Q_0…Q_{q_len-1} ∈
  B`). So `A(n) = { p_profile: [n, n-1, …, 1], q_len: n+1 }`, e.g.
  `A(0) = E(Q_0)`.
- Signatures carry **both** parts (plan subtlety #1): `sig_q = E ∩
  {Q_0…Q_{q_len-1}}`, `sig_p[i] = r_{i+1} mod 2^{p_profile[i]}` — the classical
  low-bit convention on `Q(E)P(R)`.
- Unit tests: `dim A(0)=2`, `dim A(1)=8`, `dim A(2)=64`, signature
  well-definedness / partition, ordering strictly increasing.

## M2 — B-trivial signature-graded action  ✅

- `signature_mask` / `signature_matrix` restrict the ordinary `A_C/τ` product to
  a fixed signature over `FreeModule` (the "B-trivial" action, exactly as the
  classical engine does — masking, not a separate matrix enumeration).
- **Key finding (handedness).** The diagnostic `ext/examples/sig_diag.rs` sweeps
  signature conventions against the *real* `A_C/τ` product and checks
  acyclicity. Over `A_C/τ` no independent per-coordinate mod-`2^p` signature is
  right-A-stable, but with the **classical low-bit** signature the **left**
  multiplication *does* preserve `sig ≥ R` (the *right* factor is the floor) —
  the **opposite handedness** to the classical polynomial case, and
  degree-monotone. Result: `A(0)`, `A(1)`, `E(Q_0)`, `E(Q_0,Q_1)` are ACYCLIC
  with signature-degree order.
- Consequence: the engine resolves over the **opposite algebra**
  [`CTauOpAlgebra`] (`a ·ᵒᵖ b = b·a`), so the free-module differential's source
  operation becomes the right factor and the floor holds. The antipode gives
  `(A_C/τ)^op ≅ A_C/τ`, so `Ext` ranks are **identical** — verified:
  opposite-algebra generic resolution reproduces the golden total (130 on the
  n≤20/s≤12 box).
- Correction order: **signature degree ascending** (validated linear extension).
- Tests: `product_raises_signature` (right-factor floor over the real product),
  `signatures_sum_to_full_dimension`, `opposite_algebra_matches_golden_total`.

## M3 — MVP inductive step (B = A(0))  ⏳ (in progress)

## M4 — General Algorithm 2 + generic fallback

## M5 — Validation & benchmark
