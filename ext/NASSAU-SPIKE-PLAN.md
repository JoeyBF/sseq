# Prototyping Plan — "Motivic Nassau": a signature-filtration resolution engine for A_C/τ

**Audience:** an autonomous coding agent working in the `spectralsequences/sseq` (a.k.a.
`joeybf/sseq`) Rust workspace.
**Goal of the spike:** prototype Christian Nassau's *signature-filtration shortcut*
(arXiv:1910.04063, "Computing a minimal resolution over the Steenrod algebra",
Algorithm 2) as an **alternative engine** for resolving the trivial module over
`A_C/τ` (the mod-τ C-motivic Steenrod algebra), and measure the speedup against the
current generic minimal-resolution engine. This is the phase that dominates the
motivic Adams E₂ wall-clock (~340s of ~524s on a mid-size box).

**This is a spike, not a product.** Optimize for a *validated, flag-gated proof of
speedup*, decoupled from the existing pipeline. Correctness is validated against the
existing engine; the existing engine is never modified or removed.

---

## 0. Guardrails — READ BEFORE TOUCHING ANYTHING

- **CPU / build flags (mandatory).** The build config pins `target-cpu=native`, but
  this VM advertises AVX-512 in `/proc/cpuinfo` and then **SIGILLs** executing it.
  Build and run *everything* with:
  ```
  RUSTFLAGS="-C target-cpu=x86-64-v3" cargo ...
  ```
  Do **not** edit the committed `.cargo/config.toml` (it is correct for real hardware).
- **Formatting.** `rustfmt.toml` uses unstable options → nightly only, and only on the
  files you touch: `rustfmt +nightly --edition 2024 <files>`. Never run
  `cargo +nightly fmt` (it reformats the whole workspace and creates collateral churn).
- **Branch / scope — base off the minimum PR branch, NOT the big dev branch.** The
  signature engine is fundamentally an algebra-engine feature, and its only hard
  dependency is the `A_C/τ` product (`MotivicMilnorAlgebra`), which lives on **PR1 =
  `claude/motivic-algebra-engine`**. So:
  - Cut a dedicated experiment branch (e.g. `claude/nassau-spike`) from
    `origin/claude/motivic-algebra-engine`, so if it ever becomes a PR it stacks cleanly
    on PR1 rather than the sprawling big branch.
  - **PR1 does not contain `ctau.rs`** (the `CTauAlgebra` F₂-view) or the
    `src/motivic/` resolution pipeline — those are big-branch only. For the baseline and
    rank/exactness validation you need `CTauAlgebra`, so **cherry-pick the single file
    `ext/crates/algebra/src/algebra/motivic/ctau.rs`** from
    `origin/claude/awesome-davinci-khkr8x` (resolve any small compile deps; it is a thin
    wrapper over `MotivicMilnorAlgebra`, already present on PR1).
  - The **end-to-end chart validation (M5 step 3)** needs the full deformation pipeline,
    which is big-branch only. Run *that step* against `origin/claude/awesome-davinci-khkr8x`
    as a separate confirmation; the engine's primary correctness (M5 steps 1–2: ranks +
    exactness) runs on the minimal PR1-based branch.
  - Keep the new engine **additive and feature-gated** (e.g. a `sig-nassau` cargo feature
    or an env toggle). **Do not** modify PR1/PR3 branches, and do not open a PR unless
    explicitly asked. Commit locally with descriptive messages; push only when asked.
- **Cost discipline.** Resolving high stems is genuinely expensive (a single stem-74
  resolution can exceed 10 minutes). **Keep every validation/benchmark box small**
  (start `n ≤ 30, s ≤ 16`). Use `MOT_SAVE=<dir>` to cache the resolution + lift between
  runs. Do not launch large runs "to be sure."
- **Correctness posture.** The prototype must be **always correct, opportunistically
  fast**: wherever Nassau's shortcut is not provably applicable at a bidegree, **fall
  back to the existing generic algorithm at that bidegree**. Never trade correctness
  for coverage.

---

## 1. Codebase map (verified pointers)

Resolution pipeline:
- `ext/src/motivic/mod.rs` — `MotivicResolution`. The mod-τ resolution type is
  `pub type CTauResolution = Resolution<FiniteChainComplex<FDModule<CTauAlgebra>>>`.
  `MotivicResolution::new` / `with_save` builds it with the **ordinary engine**
  (`Resolution::new_with_save`, `compute_through_stem`). **This is the call the spike
  replaces/shadows.**
- `ext/examples/resolve_motivic_ctau.rs` — standalone baseline: resolves `k` over
  `A_C/τ`. Use this as the harness for M0/M5.
- `ext/examples/chart_motivic.rs` — renders the downstream deformation-SS chart; use as
  a **basis-independent** end-to-end validation.

The algebra:
- `ext/crates/algebra/src/algebra/motivic/ctau.rs` — `CTauAlgebra`: `A_C/τ ≅
  F₂[ξ₁,ξ₂,…] ⊗ E(τ₀,τ₁,…)`, a thin F₂ view over `MotivicMilnorAlgebra` presenting the
  **τ⁰ part** of the product. Basis is `{Q(E)P(R)}`; each element carries a **weight**.
- `ext/crates/algebra/src/algebra/motivic/milnor.rs` — `MotivicMilnorAlgebra`. The
  **Milnor-matrix product** already lives here: `multiply(a, b)` at ~L559 enumerates the
  `X`/`Y` matrices with row/column/diagonal-sum constraints and tracks weight/τ. This is
  the primitive for Nassau's Lemma 2.5. **No profile restriction yet** — that is what M2
  adds.

**Reuse — do not reinvent** (classical `MilnorAlgebra`, `ext/crates/algebra/src/algebra/milnor_algebra.rs`):
- `MilnorProfile { p_part: Vec<u32>, q_part: u32, truncated: bool }` (~L14–100). Its
  `is_valid()` (~L43) already checks Nassau's admissibility condition
  `p(i+j) ≥ p(i) − j`; `is_an()` (~L72) detects `A(n)`; `get_p_part(i)` returns the
  truncation exponent (`∞` when unrestricted). This IS Nassau's profile function.
- `MilnorAlgebra::new_with_profile` (~L284) + the profile-pruned matrix multiplication
  (`multiply_basis_elements` ~L481, matrix iteration ~L1344, profile exclusion e.g.
  `q_part & !self.profile.q_part` ~L914). The classical engine can already multiply
  *inside* a sub-Hopf-algebra B; the motivic engine cannot yet.

**Takeaway:** the profile type + the "restrict the matrix enumeration to B" idea are
already implemented once (classically). The spike's genuinely new content is (a) the
**signature decomposition** `A = ⊕_R E_R A` and its induced action, and (b) **Algorithm
2's inductive step** driving the resolution off `E₀C`.

---

## 2. The math you must implement (self-contained; source = the paper, PDF also at
`/root/.claude/uploads/3a332657-b335-57fa-be5f-c01f7cbf325b/0df8cafb-1910.04063v1.pdf`)

Work at `p = 2` but **on the odd-primary-shaped algebra** `A_C/τ` (= Nassau's "EA, the
odd-primary Steenrod algebra for p=2", §7). So a basis element is `Q(E)·P(R)`:
exterior part `E = (ε₀, ε₁, …), εᵢ ∈ {0,1}` (the `q_part` bitmask) and polynomial part
`R = (r₁, r₂, …)` (the `p_part`). **Everything below must carry both parts and the
weight** — do not specialize to the polynomial-only `Sq(R)` body of §2/§3.

### 2.1 Signatures (§2, p.5)
Fix a **finite admissible** sub-Hopf-algebra `B ⊂ A_C/τ` via a profile `p`. Basis of B:
the `Q(E)P(R)` with `εⱼ` allowed by `q_part` and `0 ≤ rⱼ < 2^{p(j)}`.
- Signature set `S_B = { basis elements of B }`.
- Same B-signature: `Sⱼ ≡ Rⱼ (mod 2^{p(j)})` for all j (and equal exterior parts within
  B's exterior range). `sig_B(x)` = the unique representative in `S_B`.
- **Signature decomposition** `A = ⊕_{R∈S_B} E_R A`, `E_R A = F₂{ x : sig_B(x) = R }`.
- **Signature filtration** `F_R A = Σ_{S ≥ R} E_S A`; each `F_R A` is a **right
  A-submodule**, and `E_R A ≅ B\\A` (up to degree shift `|R|`).

### 2.2 Admissible ordering (Lemma 2.4)
Require the profile to satisfy `p(i+j) ≥ p(i) − j` (already checked by
`MilnorProfile::is_valid`). Let `P = { Pˢₜ ∈ B }`, ordered `Pˢₜ < Pˢ'ₜ'` when `t < t'`.
For a signature R, binary-decompose each `rⱼ = Σ_{tₖ=j} 2^{sₖ} εₖ`; order signatures by
the **lexicographic order of the bit-vector** `R̃ = (ε₁, …, εₙ)`. (Include the exterior
`Q` bits in the ordering analogously.)

### 2.3 Signature-graded multiplication (Lemma 2.5 — the "B-trivial" rule)
Milnor's product `Sq(R)·Sq(S) = Σ_X β_{R,S,X} Sq(T)`, sum over matrices `X = (xᵢⱼ)` with
row-weighted sums `rᵢ = Σⱼ 2ʲ xᵢⱼ`, column sums `sⱼ = Σᵢ xᵢⱼ`, diagonal sums
`tₖ = Σ_{i+j=k} xᵢⱼ`; `β ≠ 0` iff the diagonal decomposition is **bitwise disjoint**.
Call X **B-trivial** if `xᵢⱼ ≡ 0 (mod 2^{p(i)−j})` whenever `j ≤ p(i)`. Then
```
Sq(R)·Sq(S) = Σ_{B-trivial X} β_{R,S,X} Sq(T)  +  (terms of signature > R).
```
So **right-multiplication on `E_R A` enumerates only the B-trivial matrices** — this is
the per-signature action the resolver needs, and it is exactly a *profile-restricted*
version of the motivic `multiply` matrix enumeration. (The exterior `Q(E)` part
multiplies by the usual exterior rule with its own profile mask.)

### 2.4 Vanishing lines (Lemma 2.6) & applicability (Thm 3.1 / 4.1)
Let `Qₛ = P⁰_{s+1}` (Milnor primitives). For finite `B` with smallest/largest Bockstein
`Q_{smin}, Q_{smax}` in B: `Ext_B^{s,t}(k) = 0` if `t < s·|Q_{smin}|` **or**
`t > s·|Q_{smax}|`.
- **Below the line (Thm 3.1):** `B ⊆ A(n)`, `ρₙ = 2^{n+1} − 1`, `τ_B = max dim` in B.
  Algorithm 2 valid at `(s,t)` when `t > ρₙ(s+1) + τ_B`.
- **Above the line (Thm 4.1):** `B ⊆ F(n)` (kill the first n polynomial generators).
  Valid when `t < (2^{n+1} − 1)s`. Odd-primary/EA `F(n)`, `F'(n)` are as in §4:
  `F(n) = {Q(ε)P(R) : r₁=…=rₙ=0, ε₀=…=ε_{n−1}=0}`, `F'(n)` also `ε_n = 0`.
- **All degrees carry the extra weight grading** — it only refines the gradings (finer ⇒
  smaller graded pieces), but signatures, vanishing cones, and the ordering must all be
  computed weight-aware.

### 2.5 Algorithm 2 (the inductive step at bidegree (s,t))  — reproduce faithfully
```
Input:  partial resolution below (s,t); admissible B with its ordering of S_B
Output: extension of the resolution to (s,t)

M ← matrix of d : E₀C_{s,t} → E₀C_{s-1,t}
K ← kernel basis of M
N ← matrix of d : E₀C_{s+1,t} → E₀C_{s,t}
Q ← basis of K / im N            # = H(E₀C)_{s,t}
if Q nonempty:
    for qᵢ ∈ Q:                   # seed approximate boundaries
        xᵢ ← representative of qᵢ in E₀C_{s,t}   # becomes d(gᵢ)
        dᵢ ← d(xᵢ)                # represents d²(gᵢ); must reach 0
    for R ∈ S_B, R ≠ 0 in signature order:      # signature-ordered corrections
        M ← matrix of d : E_R C_{s,t} → E_R C_{s-1,t}
        for qᵢ ∈ Q:
            eᵢ ← summands of dᵢ in E_R C_{s-1,t}
            fᵢ ← solve M·fᵢ = eᵢ
            xᵢ ← xᵢ − fᵢ ;  dᵢ ← dᵢ − d(fᵢ)
    for qᵢ ∈ Q:
        assert dᵢ == 0
        introduce generator gᵢ ∈ C_{s+1,t} with d(gᵢ) = xᵢ
```
The win: the homology work happens in `E₀C` (Nassau's tables: `dim E₀Cs ≈ dim Cs / 50`,
e.g. 23997 → 586), and the corrections are cheap lifting solves in the small `E_R C`.

### 2.6 (Stretch) Algorithm 3 — lifting a cycle via the same filtration (§5)
The τ-lift's inner solves (`TauLift`/`ProductCells::solve` calling
`d.quasi_inverse(...)`) are exactly "given cycle `z ∈ C_{s,t}`, find `w` with `d(w)=z`",
which Algorithm 3 accelerates with the signature filtration. Nassau (§7) notes the τ-
correction pass "requires at each step to solve a lifting problem … for which our theory
is again applicable." Only attempt after M5 validates the resolution engine.

---

## 3. Milestones (each ends with a concrete, checkable deliverable)

### M0 — Baseline + harness  *(no new engine yet)*
- **Set up the base branch first** (see Guardrails): experiment branch off
  `origin/claude/motivic-algebra-engine`, cherry-pick `ctau.rs` so `CTauAlgebra` compiles
  and the generic engine can resolve `A_C/τ`. Confirm a clean `cargo build`/`cargo test`
  under the required `RUSTFLAGS` before writing any new code.
- Build with the required `RUSTFLAGS`. Run `resolve_motivic_ctau` (or `MotivicResolution`)
  on a small box (`n ≤ 30, s ≤ 16`) with `MOT_SAVE`. Capture, as a **golden fixture**:
  the per-`(s,t,w)` generator ranks, and the downstream `chart_motivic` output.
- Record baseline wall-clock for the resolution phase alone (add a coarse timer or reuse
  existing tracing spans; the resolution span already exists).
- Write down the exact function boundary where `A_C/τ` is resolved, so the new engine can
  slot behind the same output contract (a set of `FreeModuleHomomorphism` differentials,
  or an equivalent standalone structure — see M4).
- **Exit:** golden ranks + golden chart committed under `tests/`/fixtures; baseline
  timing noted in the plan's progress log.

### M1 — Signature combinatorics  *(pure, unit-tested in isolation)*
- Introduce a `signature` module. Represent B via the existing `MilnorProfile` (reuse it;
  do **not** define a new profile type). Implement: enumerate `S_B`; the admissible
  bit-vector ordering (2.2); `sig_B(basis_element) → S_B` respecting both `p_part` and
  `q_part`; the decomposition index `E_R`.
- Unit tests against hand-computable cases from the paper: `B = A(0) = E(Q₀)`,
  `B = E(Q₀,Q₁)`, `B = A(1)`. Check `|S_B|`, `dim B`, ordering monotonicity, and
  `sig_B` well-definedness. Cross-check `dim B` against known values.
- **Exit:** `signature` module with green unit tests, zero dependence on the resolver.

### M2 — Signature-graded ("B-trivial") multiplication over A_C/τ
- Add a profile-restricted variant of the motivic `multiply` (`motivic/milnor.rs`) that
  enumerates only **B-trivial matrices** (2.3), yielding right-multiplication on `E_R A`.
  Mirror the classical `MilnorAlgebra` profile-pruning; keep weight tracking intact.
- Validate: (i) summing over all signatures reproduces the full τ⁰ product; (ii)
  `dim E_R A = dim B\\A` in a range; (iii) the induced product is insensitive to the
  first factor's B-signature (Lemma 2.5) — spot-check.
- **Exit:** an `E_R`-graded action callable per signature, with the three checks green.

### M3 — MVP inductive step for `B = A(0) = E(Q₀)`  *(the Sq¹/Q₀ halving, Lemmas 2.1–2.2)*
- Implement Algorithm 2's step for the single-Bockstein case: `E₀C` and one `E_R` piece,
  one lifting correction. This exercises the whole scaffold (decomposition → E₀ homology →
  one signature correction → new generator) on the smallest nontrivial B.
- Drive it to extend the resolution a few bidegrees where `A(0)` is applicable (2.4), and
  assert results agree with the generic engine there.
- **Exit:** the Q₀-halving reproduces generic ranks/differentials on a small applicable
  region; `d² = 0` holds; a regression test pins it.

### M4 — General Algorithm 2 with B-selection + generic fallback
- Implement full 2.5 with signature-ordered corrections over general admissible B.
- Implement `choose_B(s, t, w) → Option<MilnorProfile>` from the vanishing lines (2.4):
  below-line `A(n)` and above-line `F(n)/F'(n)`; pick the B minimizing `dim E₀C` (Nassau's
  practical heuristic: order candidate B's, evaluate the first piece `E₀C`). When no B is
  applicable, **return `None` and fall back to the generic step** at that bidegree.
- Package the result as a **standalone `SignatureResolution`** (its own ranks +
  differential matrices), *not yet* wired into `MotivicResolution`. Decoupling keeps the
  spike from having to satisfy the full `ChainComplex`/`FreeModuleHomomorphism` contract.
- Reuse `fp` (`FpVector`, matrices, `quasi_inverse`) for all kernel/lift linear algebra —
  do not hand-roll linear algebra.
- **Exit:** `SignatureResolution` computes `k`-resolution over `A_C/τ` on the M0 box.

### M5 — Validation & benchmark  *(the payoff measurement)*
- **Correctness (basis-independent):** minimal resolutions are unique only up to
  iso, so compare invariants, not raw differentials:
  1. per-`(s,t,w)` **generator ranks** must equal the M0 golden **exactly**;
  2. **self-consistency**: `d∘d = 0` and homology of the new resolution `= k` (exactness);
  3. **end-to-end** *(big-branch step)*: the lift + deformation SS + `chart_motivic` are
     big-branch only, so run this confirmation against `origin/claude/awesome-davinci-khkr8x`
     — drop the new engine in there, feed its resolution through the existing pipeline, and
     assert the `chart_motivic` output matches the M0 golden chart. Steps 1–2 remain the
     primary gate on the minimal PR1-based branch; step 3 is the higher-level confirmation.
- **Performance:** report `dim E₀C / dim C` shrink factor per bidegree (expect ≫1) and the
  resolution-phase wall-clock vs. M0 baseline on the same small box. Do **not** scale the
  box up to chase a bigger number — the shrink factor + a mid-box timing is the result.
- **Exit:** a short validation report (ranks match, chart matches, exactness holds) + a
  speedup table. Flag-gate so both engines are runnable side by side.

### M6 — Signature-accelerated τ-lift (Algorithm 3)  *(proceed automatically once M5 is green)*
- **Do not stop after M5** — if and only if M5 passes (engine validated), continue into M6
  without checking in. If M5 fails, stop and report instead.
- Route `TauLift`'s inner `solve` (`ProductCells::solve` → `d.quasi_inverse`) through the
  signature filtration (2.6 / Algorithm 3): decompose the lifting problem by signature and
  solve in the small `E_R C` pieces. This is big-branch work (the lift lives there), so it
  runs on the big-branch validation checkout alongside M5 step 3.
- Validate identically: the motivic products / Massey products and the final chart must be
  **unchanged**; measure the lift-phase speedup (the lift was ~184s of the baseline).

---

## 4. The three subtleties most likely to cause silent wrongness

1. **Odd-primary shape (highest risk).** `A_C/τ` basis is `Q(E)P(R)`, **not** `Sq(R)`.
   Nassau's §2/§3 body is written polynomial-only; the correct reference for the exterior
   generators is §4 (`F(n)/F'(n)`, `Q(ε)P(R)`) and the odd-primary sources [6]/[7]. Every
   signature, product, and vanishing cone must carry the exterior `q_part`. A prototype
   that drops the `τᵢ`/`Qᵢ` part will look plausible and be wrong.
2. **Weight trigrading.** `A_C/τ` is trigraded `(stem, filtration, weight)`. Thread weight
   through signatures, the B-trivial product, `choose_B`, and every matrix. It only refines
   (helps), but mis-bookkeeping it corrupts the E₀ decomposition.
3. **Basis-dependence of differentials.** Do not compare raw `d(gᵢ)` against the golden —
   generator choices differ. Compare **ranks + exactness + downstream chart** (M5). This is
   also why M4 is decoupled: you're validating the functor's output, not its internals.

---

## 5. Definition of done (spike)

- `SignatureResolution` computes the `A_C/τ` resolution on `n ≤ 30, s ≤ 16`, **rank-for-rank
  identical** to the generic engine, exact (`d²=0`, `H = k`), and produces a
  `chart_motivic` output byte-identical to baseline.
- Generic engine untouched; new engine feature-gated; generic fallback covers every
  non-applicable bidegree so the composite is always correct.
- A measured `dim E₀C / dim C` shrink table and a resolution-phase timing comparison on the
  small box.
- **If M5 was green, M6 is also done**: the τ-lift routed through the signature filtration,
  products/chart unchanged, lift-phase timing reported. (If M5 failed, stop at M5 and report.)
- Everything builds/tests green under `RUSTFLAGS="-C target-cpu=x86-64-v3"`; touched files
  formatted with `rustfmt +nightly --edition 2024`.
- Committed to the experiment branch with a written progress log (baseline, per-milestone
  exit notes, final numbers). No PR, no changes to PR1/PR3.

## 6. Explicitly out of scope for the spike
- Productionizing behind the `ChainComplex` trait / replacing the default engine.
- General P-algebras beyond `A_C/τ` (R-motivic, C₂-equivariant). The design should not
  *preclude* them — keep B/profile/vanishing-line inputs data-driven — but do not
  implement them.
- Any performance run beyond the small validation box.
