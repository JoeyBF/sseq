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

## M3 / M4 — SignatureResolution engine (trivial-B plain steps, no fallback)  ✅

- [`SignatureResolution`] (in `motivic_nassau.rs`): a **standalone** minimal
  resolution of `k` over `A_C/τ` by Algorithm 2, carrying its own
  `FreeModule<CTauOpAlgebra>` modules + `FreeModuleHomomorphism` differentials
  (decoupled from the `ChainComplex` pipeline, per M4). Ports the classical
  `step0`/`step1`/`step_resolution_with_subalgebra` with the motivic signature
  masking of M1/M2.
- `choose_B` = [`MotivicSubalgebra::optimal_for`] (below-line `A(n)` vanishing
  bound, Thm 3.1). Where no `A(n)` qualifies it uses the trivial `B = A(-1) = k`,
  whose zero-signature block **is** the whole module — i.e. the ordinary generic
  step, which always works. So `optimal_profile` returns only a
  provably-applicable `B` or the trivial one, both of which the sweep converges
  on: there is **no runtime fallback**.
- The `d² = 0` identity is enforced by an `assert!` at the end of each step
  (rather than a silent retry). It held at every bidegree of every validated box
  — motivic to n=60, classical, odd-primary p=3/5 — so a future failure would be
  a genuine bug (bad vanishing line or signature order), surfaced rather than
  masked. (An earlier version retried with the trivial `B` on `d² ≠ 0`;
  instrumenting that path showed it never fired, so it was removed.)
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
- **[4] Steps.** 1008 signature-shortcut steps, 315 plain steps (trivial `B`,
  low stems below every vanishing line — not a fallback, the correct choice
  there).

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

### Performance: stem region + allocation-free signatures  ✅

Two engine-level fixes (see `PERF-NOTES.md` for the full investigation), both
landed with the `d²=0` assert and the rank cross-checks guarding correctness:

1. **Stem region, not the constant-`t` diagonal.**
   `SignatureResolution::compute_through_stem(max_s, max_t)` looped `{s ≤ max_s,
   t ≤ max_t}`. On the `(n, s)` grid that is the slope-(−1) diagonal `n + s =
   max_t`, fanning out to stem `max_t` at `s = 0` — the low-`s`, high-`n` corner,
   which is the *expensive* corner because `dim A` grows with degree. Now the loop
   gates `t − s > max_n → continue`, computing the `(n, s)` stem rectangle `{n ≤
   max_n, s ≤ max_s}` master already computed. The top boundary reads its `(s−1)`
   neighbour one stem higher (skipped), but that neighbour's degree-`t` generators
   map to nonzero images so they never enter the kernel we hit — the same
   phantom-boundary read master makes at its `distance == 1` cells. On the
   classical sphere this cut the `2 60 30` box from **18.9 s → 1.66 s**.

2. **Allocation-free coset check.** `signature_mask`'s inner loop decided coset
   membership by allocating a whole signature (a `Vec`-carrying struct) per basis
   element — 62M times on `2 80 40` — comparing with `==` and dropping it. Added
   `PAlgebra::basis_element_has_signature`, a field-by-field check that allocates
   nothing (both algebras override it). `signature_mask` 1639 ms → 579 ms.

On the classical sphere the signature engine is now on par with master's
hand-tuned `nassau.rs` (`2 80 40`: **14.8 s** vs 14.1 s) and well ahead of the
generic engine (23.7 s).

### Motivic scaling after the region fix (stem region, `s ≈ n/2`)

| region (n, s) | gens | shrink (agg) | median | max | signature | generic | speedup |
|--------------:|-----:|-------------:|-------:|----:|----------:|--------:|--------:|
| n ≤ 60, s ≤ 30 | 2776 | 4.5× | 2.0× | 48.2× | 21.0 s  | 56.4 s   | 2.68× |
| n ≤ 70, s ≤ 35 | 4925 | 5.0× | 2.0× | 50.9× | 104.0 s | 294.0 s  | 2.83× |
| n ≤ 80, s ≤ 40 | 8045 | 5.4× | 2.0× | 50.9× | 479.1 s | 1369.2 s | 2.86× |

Rank-for-rank correct and `d²=0` at every box. The end-to-end speedup over the
generic engine holds at ~2.7–2.9× and creeps up with box size as the deeper
bidegrees promote to larger `B` (the `A(2)` count grows 36 → 80 → 140). These are
the true stem-region times: both engines now compute the `(n, s)` rectangle, so
the low-`s` waste corner is gone from *both* — the ratio is the honest shrink win,
not a region artifact.

## Optimal `B`-selection ranges

`optimal_for(s, t)` picks the applicable `B` of **largest dimension** (smallest
`E₀ ≈ C/\dim B`, Nassau's heuristic) over the `A(n)` ladder plus the
pure-exterior `E(Q_0..Q_k)` intermediates, each admitted by its own **sound**
below-line bound `t > (2^{q_len}-1)(s+1) + \tau_B` (slope set by the largest
Bockstein, Lemma 2.6 / Thm 3.1) — never below a `B`'s vanishing line, so ranks
stay correct — provably applicable, not fallback-corrected.

The `B` histogram on n≤40 is `A(0)×583, A(1)×359, A(2)×33, E(Q_0..Q_1)×33`. The
finer ladder upgrades only 33 bidegrees (A(0)→E(Q_0,Q_1)), moving aggregate
shrink 5.1→5.2×. This is not a tuning failure but a **structural** fact:
admissibility couples the two parts — `Q_0·ξ_1 ∋ Q_1` forces any `B ∋ ξ_1, Q_0`
to contain `Q_1` — so no sound `B` has `A(0)`'s slope-1 line with better shrink,
and the large `A(0)` region is irreducible. The `A(n)` below-line family is
therefore already near-optimal; richer families (e.g. the above-line `F(n)`)
were judged not worth the complexity for the achievable gain.

## `PAlgebra` trait — unifying the ordinary and motivic engines

The resolution engine is generic: `SignatureResolution<A: PAlgebra>`. The
abstraction is split into two traits, mapping onto the mathematics of a
Margolis P-algebra:

- **`Profile`** — the coset combinatorics of one finite sub-Hopf-algebra `B`,
  i.e. its signature space `S_B ≅ B\\A`: `zero_signature`, `iter_signatures`,
  `dimension`, `name`. No basis, no module. (These are the methods that were
  wrongly bundled into `PAlgebra` — they're operations on signatures, not the
  algebra.)
- **`PAlgebra`** — the algebra `A`: its profile family (`optimal_profile` from
  the vanishing lines) and the *single* basis-dependent primitive, the coset
  projection `basis_element_signature = sig_B`. `signature_mask` and
  `signature_matrix` are now derived from it generically — one implementation,
  not one per algebra.

Correspondence to the definition: a P-algebra is a graded connected `F_2`-Hopf
algebra free over each finite sub-Hopf-algebra `B` (Milnor–Moore); that freeness
`A ≅ B ⊗ (B\\A)` is what makes `sig_B` well-defined and `S_B` a set of cosets.
`Profile` packages `S_B`; `PAlgebra` provides `sig_B` and the basis. Two honest
caveats: (1) the trait requires only `Algebra` — the Hopf/freeness axioms are a
precondition the impl is trusted to meet (guarded by the `d²=0` assert + rank
check), so `PAlgebra` is the *computational shadow* of a P-algebra; (2)
right-stability is an extra requirement beyond "P-algebra" (the filtration by
right ideals), always arrangeable for a cocommutative P-algebra via the antipode
— the opposite-algebra move. Odd primes are out of scope (the engine is `F_2`).

Two implementations:

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
$\mathbb{F}_p$-Hopf algebra with a profile/signature structure and a
right-stable (or opposite-right-stable) filtration is a `PAlgebra` and is
resolvable by this engine unchanged.

### Odd-primary generalization

The engine is now $\mathbb{F}_p$-generic, not $\mathbb{F}_2$-only:

- All matrices/vectors are built over `algebra.prime()`, and the
  signature-correction sweep runs through `QuasiInverse::apply` with coefficient
  $-1 \equiv p-1$, which does the honest $\mathbb{F}_p$ arithmetic (scale each
  preimage row by its coefficient; `x -= f`, `dx -= d(f)`). At $p = 2$ this is
  identical to the old 0/1 logic, so the classical/motivic results are unchanged.
- `impl PAlgebra for MilnorAlgebra` now uses [`SteenrodProfile`], a **prime-aware**
  profile in the standard Milnor basis $Q(E)P(R)$: at $p = 2$ the exterior part is
  vacuous and it reduces to the classical polynomial profile; at odd $p$ it
  carries both the polynomial truncation ($\xi_{i+1} < p^{p\_profile[i]}$) and the
  exterior $Q_i \in B$. Degrees come from `combinatorics::{xi,tau}_degrees(p)`;
  `A(n)` and the vanishing slope $\rho$ branch on $p$.

Validated: `SignatureResolution<MilnorAlgebra>` reproduces the odd-primary Adams
`E_2` rank-for-rank against the generic engine at **$p = 3$ and $p = 5$**, and the
signature shortcut *fires* (`sig_steps > 0`) — confirming (a) the standard odd-$p$
Milnor basis is **right-stable** (used directly, no opposite, unlike the
conjugate-basis motivic case), and (b) the $\mathbb{F}_p$ correction arithmetic is
correct. The $p = 2$ classical and motivic validations are unchanged.

One engine now spans: motivic $A_C/\tau$ ($p=2$, opposite), classical Steenrod
($p=2$, direct), odd-primary Steenrod ($p=3,5$, direct).

### Finite ambients — tmf over `A(2)`

The engine also resolves over a **finite** sub-Hopf-algebra as the ground ring.
`Ext_{A(2)}(F_2, F_2)` is the tmf Adams `E_2`, so resolving `k` over `A(2)`
(`MilnorAlgebra::new_with_profile`, `p_part = [3,2,1]`, dim 64) computes tmf.

`optimal_profile` uses the `A(n)`/vanishing-line ladder **capped at the ambient**
(`B ⊆ self.profile()`). The vanishing line `t > ρ(s+1) + τ_B` is **intrinsic to
`B`** — it comes from `Ext_B` vanishing above `B`'s own top-Bockstein line
(Lemma 2.6) — so it applies over a finite ambient exactly as over the full
algebra. The *only* ambient-dependence is the cap: a valid `B` must actually be a
sub-Hopf-algebra of the ground ring (`A(3) ⊄ A(2)`, so it is excluded). When
`A(2)` itself is the applicable `B` (high `t`), that is above its top line where
`Ext_{A(2)} = 0`, so those bidegrees resolve to zero essentially for free. (An
earlier version conservatively used plain steps for finite ambients; that was
unnecessary — the missing piece was the cap, not the vanishing lines.)

Validated (`tmf_resolution_over_a2_matches_generic`, `examples/tmf.rs`): the
`A(2)` resolution matches the generic engine **rank-for-rank** — on stem ≤ 50,
s ≤ 24 in the example — and the signature shortcut fires (**1357** shortcut vs
368 plain steps there; `B ⊆ A(2)`). The chart shows the tmf `E_2` — the
`h_0`-tower in stem 0, `h_1`/`h_2`, the `c_0` family, the onset of Δ-periodicity
around stem 24 — and crucially **no `h_3`** (`Sq⁸ ∉ A(2)`), which distinguishes
`A(2)` from the full Steenrod algebra.

### Finite modules and chain complexes

The signature machinery (`step_general`, `s ≥ 2`) is intrinsic to the free
`A`-modules and their `A`-linear differentials — it never inspects what is being
resolved. So the engine resolves not just `k` but any bounded module `M`, and any
bounded **finite chain complex** `C_*`.

- `SignatureResolution::from_module` / `from_chain_complex` seed from an arbitrary
  target. The single module-specific ingredient is the **vanishing-line offset**:
  `Ext_B(M, k)` is a subquotient of `⊕ Σ^{n_i} Ext_B(k, k)`, so it vanishes only
  above `k`'s line shifted by `top(M)`. `optimal_profile` is called at
  `t − target_top`; without it the engine picks `B` below its true line and the
  sweep fails (`Cnu`, cofiber of `ν = Sq⁴`, panicked with `d² ≠ 0` at `(2,5)`
  before the offset).
- Chain complexes need the full Cartan–Eilenberg step. `plain_step` ports the
  generic engine's C–E augmentation over the target complex `C_{s,·}`; the
  previous augmented kernel is *recomputed* (`plain_kernel`) rather than stored,
  so it composes with `step_general` across the stem boundary with no cross-step
  state. Dispatch: a nontrivial `B` is only ever chosen when `t > target_top`,
  where `C_{·,t} = 0` and the C–E step degenerates to exactly the pure free-module
  step — so the shortcut region and the augmentation provably never overlap.

Validated rank-for-rank against the generic engine: finite modules `C2v14, Ceta,
Cnu, Joker, Csigma, RP4, DA1` (`examples/finite_module_validate.rs`), and the
Yoneda **cofibers** named by the `cofiber` attribute — `C4` (3-term), `Ceta2`
(3-term), `C2v14` (5-term) — built exactly as `utils::construct_standard` does
(`examples/cofiber_validate.rs`). Unit tests: `finite_module_resolution_matches_generic`
(incl. the offset-exercising `Cnu`) and `cofiber_chain_complex_matches_generic`
(Yoneda of `h₀²`, 3-term, shortcut steps firing).

**Speedup (C4 cofiber, classical `p = 2`, vs the generic engine — master's Nassau
is `k`-only, so it cannot resolve these at all):**

| box (n, s) | gens | signature | generic | speedup |
|-----------:|-----:|----------:|--------:|--------:|
| 50, 26     | 459  | 0.45 s    | 0.43 s  | 0.96×   |
| 60, 30     | 647  | 1.61 s    | 1.84 s  | 1.14×   |
| 70, 35     | 1020 | 5.54 s    | 7.32 s  | 1.32×   |
| 80, 40     | 1453 | 18.05 s   | 31.93 s | 1.77×   |

The ratio climbs with stem exactly as classical `k` does (also ~1.6–1.8× at
n=80): the shrink converts to wall-clock as the generic engine goes cubic. So
the signature engine gives Nassau-style acceleration for finite chain complexes —
a capability the hand-tuned `nassau.rs` does not have.

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
- ✅ Generic engine untouched; new engine feature-gated; the trivial `B` (a plain
  step) covers every non-applicable bidegree — no runtime fallback needed.
- ✅ Measured `dim C / dim E₀C` shrink table + resolution-phase timing.
- ⏳ M6 (τ-lift) deferred to the big branch (pipeline not present here).
- ✅ Builds/tests green under the required RUSTFLAGS; touched files nightly-fmt'd.
- ✅ Committed with this progress log; no PR; no changes to the generic engine.
