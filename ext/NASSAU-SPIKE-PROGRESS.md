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

## M3 / M4 — SignatureResolution engine + generic fallback  ✅

- [`SignatureResolution`] (in `motivic_nassau.rs`): a **standalone** minimal
  resolution of `k` over `A_C/τ` by Algorithm 2, carrying its own
  `FreeModule<CTauOpAlgebra>` modules + `FreeModuleHomomorphism` differentials
  (decoupled from the `ChainComplex` pipeline, per M4). Ports the classical
  `step0`/`step1`/`step_resolution_with_subalgebra` with the motivic signature
  masking of M1/M2.
- `choose_B` = [`MotivicSubalgebra::optimal_for`] (below-line `A(n)` vanishing
  bound, Thm 3.1). Where no `A(n)` qualifies it uses `B = F₂`, whose
  zero-signature block **is** the whole module — i.e. the ordinary generic step.
  So every non-applicable bidegree is covered and the composite is **always
  correct** — no separate fallback code path needed.
- Extra safety: the step returns `false` if the correction sweep leaves
  `d² ≠ 0` (signature order insufficient), and the caller redoes that bidegree
  with `B = F₂`. In practice this never fires on the validated box (the
  vanishing-line `B` is always sufficient).
- All kernel/image/lift linear algebra reuses `fp` (`Matrix`,
  `AugmentedMatrix`, `compute_kernel`, `compute_quasi_inverse`) — no hand-rolled
  linear algebra.
- **Exit:** `SignatureResolution` resolves `k` and its ranks match the generic
  engine bidegree-for-bidegree on `s ≤ 12, t ≤ 32` (unit test
  `signature_resolution_matches_generic`), with `d² = 0` throughout.

## M5 — Validation & benchmark  ✅

Harness: `ext/examples/motivic_nassau_validate.rs` (feature `sig-nassau`) resolves
the same `(max_s, max_t)` rectangle with both engines and reports correctness,
shrink, timing, and shortcut/fallback counts. On the **n ≤ 40, s ≤ 22** box
(`max_t = 62`), release, `RUSTFLAGS="-C target-cpu=x86-64-v3"`:

- **[1] Correctness (basis-independent).** Signature-engine generator ranks equal
  the generic engine's at **every** bidegree (1106 generators over the full
  rectangle). Homology `= k` / `d² = 0` holds (checked inside every step).
- **[2] Shrink `dim C / dim E₀C`.** Aggregate **5.1×** (Σ dim C = 309 548, Σ dim
  E₀ = 60 660), median 2.0×, **max 48.2×** — in line with Nassau's ≈50× on the
  polynomial case. Largest single step: `(s=6, t=62)`, `B = A(1)`,
  dim C = 2213 → dim E₀C = 339.
- **[3] Timing (same box).** Signature engine **12.1 s** vs generic engine
  **33.3 s** — a **≈2.75× resolution-phase speedup** (both serial, same
  rectangle).
- **[4] Steps.** 1008 signature-shortcut steps, 315 plain fallbacks (low stems
  below every vanishing line).

Both engines are runnable side by side; the signature engine is fully
feature-gated (`sig-nassau`) and additive.

### Scaling (s ≤ 22, increasing stem)

| box (n) | gens | median shrink | max shrink | signature | generic | speedup |
|--------:|-----:|--------------:|-----------:|----------:|--------:|--------:|
| 40      | 1106 | 2.0×          | 48.2×      | 12.1 s    | 33.3 s  | 2.75×   |
| 50      | 1730 | 2.0×          | 50.9×      | 57.9 s    | 177.4 s | 3.06×   |
| 60      | 2449 | 5.9×          | 50.9×      | 265 s     | 763 s   | 2.88×   |

Rank-for-rank correctness and `d²=0` hold at **every** box. At n=60 the
vanishing-line `B` promotes the deepest bidegrees to `A(2)`: the largest single
step is `(s=7, t=82)`, `B=A(2)`, dim C = 7128 → dim E₀C = 201 (**35.5×**).

**Speedup is sub-linear in the shrink** (and dips slightly at n=60) by design:
promoting a bidegree from `A(1)` to `A(2)` shrinks its `E₀` homology but runs the
correction sweep over ~63 signatures instead of ~7, and this spike deliberately
does **not** cache masks/partial matrices across signatures or parallelize the
`E_R` solves (the classical engine does both). So the measured result is a
*proof of shrink* with a ~3× end-to-end resolution-phase win; closing the gap to
a shrink-proportional speedup is an optimization task, not a correctness one.

## Optimal `B`-selection ranges

`optimal_for(s, t)` picks the applicable `B` of **largest dimension** (smallest
`E₀ ≈ C/\dim B`, Nassau's heuristic) over the `A(n)` ladder plus the
pure-exterior `E(Q_0..Q_k)` intermediates, each admitted by its own **sound**
below-line bound `t > (2^{q_len}-1)(s+1) + \tau_B` (slope set by the largest
Bockstein, Lemma 2.6 / Thm 3.1) — never below a `B`'s vanishing line, so ranks
stay correct rather than fallback-corrected.

The `B` histogram on n≤40 is `A(0)×583, A(1)×359, A(2)×33, E(Q_0..Q_1)×33`. The
finer ladder upgrades only 33 bidegrees (A(0)→E(Q_0,Q_1)), moving aggregate
shrink 5.1→5.2×. This is not a tuning failure but a **structural** fact:
admissibility couples the two parts — `Q_0·ξ_1 ∋ Q_1` forces any `B ∋ ξ_1, Q_0`
to contain `Q_1` — so no sound `B` has `A(0)`'s slope-1 line with better shrink,
and the large `A(0)` region is irreducible. The `A(n)` below-line family is
therefore already near-optimal; richer families (e.g. the above-line `F(n)`)
were judged not worth the complexity for the achievable gain.

## `PAlgebra` trait — unifying the ordinary and motivic engines

The resolution engine is generic: `SignatureResolution<A: PAlgebra>`, where
`PAlgebra` abstracts exactly the data Nassau's sweep needs (profiles,
signatures, `signature_mask`, `optimal_profile`, dims/names). Two implementations:

- `impl PAlgebra for CTauOpAlgebra` — the motivic case, over the opposite
  algebra so the filtration is right-stable (`MotivicSubalgebra` profiles).
- `impl PAlgebra for MilnorAlgebra` — the **classical** case, reusing the proven
  `MilnorSubalgebra` machinery from `crate::nassau` (now `pub(crate)`); the
  standard Milnor basis is already right-stable, so no opposite is needed.

One engine drives both. Validated: `SignatureResolution<MilnorAlgebra>`
reproduces the classical Adams `E_2` rank-for-rank against the generic engine
over `MilnorAlgebra` (`classical_signature_resolution_matches_generic`), while
`SignatureResolution<CTauOpAlgebra>` reproduces the motivic (unchanged). The
handedness — the only real difference between the two cases — is confined to the
trait `impl` (which algebra, direct vs. opposite), per the note
`notes/opposite-algebra-nassau.tex`.

Margolis's general `P`-algebras fit the same trait: any graded connected
$\mathbb{F}_2$-Hopf algebra with a profile/signature structure and a
right-stable (or opposite-right-stable) filtration is a `PAlgebra` and is
resolvable by this engine unchanged.

## M6 — Signature-accelerated τ-lift (Algorithm 3)

**Not attempted in this workspace.** M6 routes the `TauLift`'s inner `solve`
through the signature filtration, but the τ-lift + deformation pipeline
(`ext/src/motivic/`, `TauLift`, `chart_motivic`) is **big-branch only**
(`origin/claude/awesome-davinci-khkr8x`); it does not exist on this PR1-based
branch. M5 is green, so per the plan M6 is the next step — to be run on the
big-branch checkout, reusing `MotivicSubalgebra` + the signature masking here.

## Definition of done — status

- ✅ Rank-for-rank identical to the generic engine, exact (`d²=0`, `H=k`), on the
  box.
- ✅ Generic engine untouched; new engine feature-gated; generic fallback (trivial
  `B`) covers every non-applicable bidegree.
- ✅ Measured `dim C / dim E₀C` shrink table + resolution-phase timing.
- ⏳ M6 (τ-lift) deferred to the big branch (pipeline not present here).
- ✅ Builds/tests green under the required RUSTFLAGS; touched files nightly-fmt'd.
- ✅ Committed with this progress log; no PR; no changes to the generic engine.
