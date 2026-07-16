# Handoff: chromatic (Morava stabilizer) computations

**Status.** Two validated, self-contained deliverables:
1. the height-1 algebra `gr S(1)` and a smoke test reproducing `H^*(S(1))` at `p = 2` (the
   *small-prime* reference; §5), and
2. **the finite Chevalley–Eilenberg tool of §3 — now implemented and validated.** It reproduces the
   whole known validation ladder `dim H^*(L(n,n)) = 2, 12, 152` at `n = 1, 2, 3`, and **independently
   confirms `dim H^*(L(4,4)) = 3440`** — the near-term prize (Salch's `n = 4` case, by a method
   orthogonal to his). See §3 for the delivered code and what it settled.

This brief lays out the mathematics and the remaining open direction (`n = 5`). Read it top to bottom
before extending the tool; the math is subtle and there is a documented trap (see §6).

> **Update since first handoff.** The subscript ambiguity flagged in §2 is now *resolved by the tool
> itself*: the green-book reading reproduces the known `n = 3` answer `152`; the swapped ("Salch eq.
> (8)") reading gives `128`, so it is the wrong transcription. The green-book reading is the
> constructor default. See §2 and the `validation_ladder_n3` test.

**One-paragraph orientation.** The algebraic input to `K(n)`-local (chromatic) homotopy is the
cohomology of the height-`n` Morava stabilizer group `S_n`. There are two genuinely different
computational regimes, and *they want different tools*:

- **Large primes (`p > n+1`): `H^*(S(n)) = H^*(L(n,n))`, a *finite* Chevalley–Eilenberg
  computation** over an `n^2`-dimensional Lie algebra. This is where the *open, checkable* frontier
  is (`n = 5` is unknown; see §3). The right tool is direct finite-dimensional Lie-algebra cohomology
  — pure `F_p` linear algebra — **not** the resolution/field-trick machinery.
- **Small primes (`p = 2, 3`): the May spectral sequence does not collapse**, the object is infinite
  (sphere-like), and the natural tool *is* a minimal free resolution / the field trick. This is what
  the validated `gr S(1)` reference in this branch exercises (§5).

The recommendation (§3) is to build the finite-CE tool and aim it at **independently verifying
`H^*(L(4,4))`** (Salch is still writing that case up) and then **attacking the open `n = 5`**.

---

## 1. What is in this branch

| File | What it is |
|---|---|
| `crates/algebra/src/algebra/morava_stabilizer.rs` | `MoravaStabilizerAlgebra`: `gr S(1) = u(L(1)) = Λ(x_1) ⊗ F_2[x_2]` as an `Algebra + Bialgebra`. Validated, unit-tested. *(small-prime reference, §5)* |
| `examples/chromatic_grs1.rs` | Resolves `F_2` over `gr S(1)`, checks the chart equals `F_2[h_1] ⊗ Λ(ρ_1)` (Ravenel 6.3.21(a)). Self-contained (no field-trick machinery). |
| `crates/algebra/src/lie/morava_lie.rs` | `MoravaLie`: the finite Lie algebra `L(n,n)` (green book Thm 6.3.3), structure constants, bracket. Unit-tested for antisymmetry + Jacobi at `n = 2, 3, 4`. *(large-prime tool, §3)* |
| `crates/algebra/src/lie/cohomology.rs` | `chevalley_eilenberg_cohomology`: builds `Λ^•(L^*)`, ranks the CE differential over `F_p` (block-decomposed by `i`-weight), returns total dim + Betti numbers. Includes a `d² = 0` gate and the `2, 12, 152` validation tests. |
| `examples/chromatic_finite_ce.rs` | Runs the validation ladder `n = 1..4`, printing `dim H^*(L(n,n))` and the (Poincaré-self-dual) `H^k` Betti profile against the reference table. |

Run the small-prime reference: `cargo run --example chromatic_grs1 -- 32 18`.
Run the large-prime ladder: `cargo run --release --example chromatic_finite_ce -- 4`.
Tests: `cargo test -p algebra morava` and `cargo test -p algebra --lib lie`.

The `gr S(1)` files are *restricted*/small-prime data (§5): a validated worked example — proof that the
codebase accepts a new Morava algebra and returns the right answer, and a reference for the exact
height-1 structure — not the base for the finite-CE tool, which is a different construction (the `lie`
module).

**`chromatic_finite_ce` output** (`n = 4`, primes `p = 3, 5, 5, 7`):

```text
  n   p   dim L       dim H^*      expected   status
  1   3       1             2             2   OK
  2   5       4            12            12   OK
  3   5       9           152           152   OK
  4   7      16          3440          3440   OK   <- independent confirmation of Salch's n=4
```

The field-trick machinery (`field_resolution_ext`, `tensor_resolution.rs`, ≈5k lines) lives on the
branch `claude/cohomology-field-resolution-cmcu0i` and its descendants, **not** on `master`. See §7.

---

## 2. The mathematics, pinned to the sources

Notation from Ravenel, *Complex Cobordism and Stable Homotopy Groups of Spheres* ("green book"),
ch. 6 (`§6.2–6.3` are the ones to implement from), cross-checked against Salch, *Ravenel's May
spectral sequence collapses immediately at large primes* (arXiv:2312.17185).

**The associated graded Hopf algebra.** `E_0 S(n) ≅ F_p[t_{i,j} : i ≥ 1, j ∈ Z/n] / (t_{i,j}^p)`,
with `t_{i,j}` in internal (topological) degree `2(p^i − 1) p^j`. It is the linear dual of a
primitively generated Hopf algebra `E_0 S(n)^* = u(L(n))`, the restricted enveloping algebra of a
restricted Lie algebra `L(n)`.

**The restricted Lie algebra `L(n)` (green book Thm 6.3.3).** `F_p`-basis `{x_{i,j} : i ≥ 1,
j ∈ Z/n}`, `x_{i,j}` dual to `t_{i,j}`. With `m = ⌊pn/(p−1)⌋` and `δ^s_t = 1` iff `s ≡ t (mod n)`
(else 0):

```
                  ⎧ δ^l_{i+j} x_{i+k,j}  −  δ^j_{k+l} x_{i+k,l}     if i + k ≤ m
[x_{i,j}, x_{k,l}] = ⎨
                  ⎩ 0                                              otherwise
```

Restriction (`p`-th power) map:

```
              ⎧ x_{i+n, j+1}              if i > n/(p−1),  or  i = n/(p−1) and p > 2
ξ(x_{i,j})  = ⎨ x_{2n, j} + x_{2n, j+1}   if i = n and p = 2
              ⎩ 0                         if i < n/(p−1)
```

> **⚠ Subscript ambiguity — RESOLVED by computation.** My OCR of green book 6.3.3 reads
> `δ^l_{i+j} x_{i+k,j} − δ^j_{k+l} x_{i+k,l}`, while Salch's eq. (8) reads
> `δ^l_{i+j} x_{i+k,l} − δ^j_{k+l} x_{i+k,j}` — the two `x` second-subscripts (`j` vs `l`) appear
> swapped between transcriptions. This is irrelevant at `n = 1` (all `δ = 1`, both terms are
> `x_{i+k,0}`, and the bracket vanishes) and, as it turns out, the two also *coincide* at `n = 2`
> (both give `12`) — but they **diverge at `n = 3`**: the green-book reading reproduces the known
> `dim H^*(L(3,3)) = 152`, while the swapped reading gives `128`. **So the green-book reading is
> correct**, and it is the [`BracketConvention::GreenBook`] constructor default; the other is kept
> behind `with_convention` purely to keep the disambiguation checkable. Both readings pass
> antisymmetry, Jacobi, and `d² = 0` (they are genuinely different Lie algebras), so only a *known
> cohomology answer* distinguishes them — which is exactly what the `n = 3` rung provides. (Note also
> that Ravenel corrected the restriction formula between editions — use a current printing.)

**`L(n,m)` (Salch, following Ravenel Thm 1.4).** The finite quotient of `L(n)` by the span of
`{x_{i,j} : i > m}`. For **large primes `p > n+1`**, `m = ⌊pn/(p−1)⌋ = n`, so `L(n,n)` has basis
`{x_{i,j} : 1 ≤ i ≤ n, j ∈ Z/n}` — exactly `n^2`-dimensional — and the restriction is **trivial on
`L(n,n)`** (`ξ` sends `x_{i,j}` to `x_{i+n,·}`, which is `0` in the quotient). Hence at large primes
`H^*(L(n,n))` is *ordinary* Lie-algebra cohomology, and the collapse `H^*(S(n)) = H^*(L(n,n))` holds
(Salch's main theorem).

---

## 3. The finite Chevalley–Eilenberg cohomology tool — DELIVERED

**Delivered** in `crates/algebra/src/lie/` (`morava_lie.rs` + `cohomology.rs`) with the driver
`examples/chromatic_finite_ce.rs`. It executes the plan below and clears the correctness gate
(`2, 12, 152` at `n = 1, 2, 3`) plus the near-term prize (`3440` at `n = 4`). The "Concrete first
steps" checklist at the end of this section is done through step 4; step 5 (`n = 5`) is left open and
is discussed under "Feasibility gradient". What follows is the design as built (and the source for
anyone extending it to `n = 5`).

**Goal.** Compute `H^*(L(n,n); F_p)` for `p > n+1` by direct Lie-algebra cohomology: build the
`n^2`-dimensional Lie algebra `L(n,n)` from §2, form the Chevalley–Eilenberg complex, and take
cohomology via `F_p` linear algebra. This is **new, self-contained code** (a Koszul complex + rank
computations) — it does not use the resolution engine or the field trick, and it needs the `fp` crate
at **odd** primes (the existing chromatic code is `p = 2`-only, but `fp` supports odd `p`).

**The Chevalley–Eilenberg complex.** Let `V = L(n,n)^*` (dimension `n^2`). The complex is the
exterior algebra `C^• = Λ^•(V)`, `C^k = Λ^k(V)`, `k = 0, …, n^2`, with the CE differential
`d : Λ^k(V) → Λ^{k+1}(V)` dual to the bracket:

```
(dω)(v_0, …, v_k) = Σ_{i<j} (−1)^{i+j} ω([v_i, v_j], v_0, …, v̂_i, …, v̂_j, …, v_k)
```

(trivial coefficients). Concretely `d` is the derivation extending, on `V`,
`d(x_a^*) = − Σ_{b<c} c^a_{bc} · x_b^* ∧ x_c^*`, where `[x_b, x_c] = Σ_a c^a_{bc} x_a`. Then
`H^k = ker(d_k) / im(d_{k−1})`, and

```
dim_{F_p} H^*(L(n,n)) = 2^{n^2} − 2 · Σ_k rank(d_k).
```

**Validation ladder (this is the whole point — each rung is checkable):**

| n | dim `H^*(L(n,n))` | dim CE complex `2^{n^2}` | status of the answer |
|---|---|---|---|
| 1 | 2 | 2 | trivial (`= E(h_{1,0})`) |
| 2 | 12 | 16 | known, all p |
| 3 | 152 | 512 | known `p ≥ 5` |
| 4 | 3440 | 65536 | **✓ CONFIRMED by this tool** (`p = 7`); independent check of Salch's in-progress `n = 4`. Betti profile `[1,5,18,55,129,249,409,551,606,551,409,249,129,55,18,5,1]` |
| 5 | 128512 (conjectured, from a generating function) | 2^25 ≈ 33.5M | **open — no one has it** |
| 6 | 7621888 (conjectured) | 2^36 | open |

**Crucial simplification for validation:** the *total* `F_p`-dimension of `H^*(L(n,n))` is
grading-independent, so you can match the table above **without solving the grading puzzle** (§6) —
just build the Lie algebra, form `Λ^•`, and rank the differentials. Get the `(s,t,u)`-graded chart
later; the total dimension is the honest first milestone.

**Feasibility gradient** (status as built).
- `n ≤ 3` (complex ≤ 512): **done, gate passing.** Reproduces 2, 12, 152.
- `n = 4` (complex 65536): **done.** The complex is block-decomposed by `i`-weight (not internal
  degree — see the note below), so the largest matrix ranked is a few hundred rows, never
  65536-square; the whole ladder runs in well under a second. This was the highest-value target: an
  independent machine confirmation of Salch's in-progress `H^*(L(4,4)) = 3440`, by a method orthogonal
  to his (he derives it via deformations/spectral sequences, not computation). Given the field's track
  record — Ravenel's own first-edition `H^*(S(2))` at `p=3` was wrong until Henn caught it — this
  independent check has real value.
- `n = 5` (complex 2^25 ≈ 33.5M): **still open — deliberately not attempted by this pass.** The
  current implementation enumerates all `2^N` monomials densely (bucketing by `i`-weight, then dense
  `row_reduce` per block) and refuses `N > 20` (`MAX_DIM` in `cohomology.rs`) to avoid OOM. At `n = 5`
  even a single mid-weight block has millions of monomials, so a dense pass is out regardless. Getting
  `n = 5` needs (a) a *streaming* subset enumerator that generates each `(weight, degree)` block by
  subset-sum rather than materializing all `2^{25}`, and (b) sparse rank over `F_p` (or borrowing
  Salch's height-shifting). Partial data — specific `i`-weights, or the low/high cohomological corners
  — is the reachable and genuinely novel next step. A confirmed total of `128512` would be a real
  result.

  > **Block-grading note (what actually splits the complex).** The differential is *not* homogeneous
  > for the topological internal degree `2(p^i−1)p^j` (this is the §6 trap). It *is* homogeneous for
  > the **`i`-weight** `w = Σ i` over the wedge factors, because the bracket sends weight `i` and `k`
  > to weight `i + k`. The implementation decomposes `Λ^•` by `(i-weight, cohomological degree)`; that
  > is what keeps the blocks small. A finer split by an honest May/RMSS internal degree (§6) would
  > shrink `n = 5` blocks further and is the natural refinement.

**Concrete first steps** (checklist — done through step 4).
1. ✓ `crates/algebra/src/lie/morava_lie.rs`: `MoravaLie { n, p, gens, convention }`, fixed index order
   `index(i,j) = (i−1)n + j`, structure constants from the §2 bracket (`m = n`). Bracket unit-tested
   for antisymmetry + Jacobi at `n = 2, 3, 4`.
2. ✓ `crates/algebra/src/lie/cohomology.rs`: `Λ^•(V)` as bitmasks, CE differential built as `F_p`
   matrices (`fp::matrix::Matrix`), each `d_k` ranked via `row_reduce`; `d² = 0` is unit-tested.
3. ✓ `dim H^* = 2^N − 2 Σ rank d_k`; asserts 2, 12, 152 at `n = 1, 2, 3`. **Gate passing.**
4. ✓ `n = 4` confirmed `3440` (`examples/chromatic_finite_ce.rs`).
5. ☐ `n = 5` — open; see the feasibility note above for the two upgrades it needs.

---

## 4. Why not the field trick / resolution engine for this

The resolution engine computes `Ext_A(F_p, F_p)` = *restricted*/comodule cohomology (infinite),
which is the right thing at **small** primes where the May SS does not collapse. At **large** primes
the answer is *ordinary* finite Lie cohomology — a different complex (Koszul, not a free resolution),
and cheaper. Using the infinite machinery for the finite large-`p` answer is the wrong tool. The
field trick's genuine niches are (a) `p = 2` `E_1` pages and (b) module amortization (`Ext` of the
lattice of type-`n` modules `V(k)`, Smith–Toda, `E_n/I_k` over `gr S(n)`) — see §7.

---

## 5. The validated reference: `gr S(1)` at `p = 2`

`p = 2` is a *small* prime at height 1 (`p = n+1`, not `> n+1`), so this is the non-collapsed regime
and the object is infinite. From §2 at `n = 1`: `L(1)` is abelian (the bracket vanishes — both `δ`
are 1 and the two terms cancel), with restriction `ξ(x_1) = 0`, `ξ(x_i) = x_{i+1}` for `i ≥ 2` (from
`ξ(x_1) = x_{2,0} + x_{2,1} = 2x_2 = 0` and the `i > n/(p−1)` case). The restricted enveloping algebra
therefore collapses to

```
gr S(1) = Λ(x_1) ⊗ F_2[x_2],    |x_1| = 2 (exterior),    |x_2| = 6 (polynomial),
```

and resolving `F_2` gives (Ravenel Thm 6.3.21(a); the May SS collapses at height 1, so this is the
genuine `H^*(S(1))`):

```
Ext_{gr S(1)}(F_2, F_2) = P(h_1) ⊗ E(ρ_1) = F_2[h_1] ⊗ Λ(ρ_1),
    h_1 = h_{1,0} = [t_1] ∈ (s,t) = (1,2),    ρ_1 = h_{2,0} = [t_2] ∈ (s,t) = (1,6).
```

The example reproduces exactly this. Note the large-`p` answer is different and smaller
(`H^*(S(1)) = E(h_{1,0})`, dim 2, green book 6.3.21(b)) — that is the one the finite-CE tool (§3)
computes and the one in the validation table.

---

## 6. The grading trap (learned the hard way)

The first `gr S(1)` implementation graded every Lie generator `x_i` by its topological degree
`2(2^i − 1)` and treated them all as independent primitives → the exterior algebra `Λ(x_1, x_2, …)` →
the answer `F_2[h_1, h_2, …]`. **That is wrong.** Under the topological grading the restriction
`ξ(x_i) = x_{i+1}` and the coproduct cross terms are *inhomogeneous* and silently vanish; the reader
happily returns a plausible-looking but incorrect chart. The fix required the restriction (which
forces `|x_{i+1}| = p·|x_i|`, so for `i ≥ 2` the `x_i` are the *powers* of one polynomial generator,
not independent).

**Lesson for the finite-CE tool.** Two safeguards: (i) for validation, use the *total* cohomology
dimension, which is grading-independent — sidestep the trap entirely at first; (ii) for the graded
chart, note that the bracket `[x_{i,j}, x_{k,l}] ~ x_{i+k,·}` is **not** homogeneous in the
topological degree `2(p^i−1)p^j` (it is homogeneous in the Ravenel/May filtration). Pin the grading
against green book 6.3 *before* trusting any graded output; the reader will not catch a wrong grading
for you.

---

## 7. The field-trick lane (alternative, for small primes and modules)

Preserved on `claude/cohomology-field-resolution-cmcu0i` (and `…/chromatic-computations-feasibility`):
a full, generic implementation of Nassau's tensor/field trick over any `Bialgebra` — antipode `χ`,
closed-form tensor differential `δ_Q`, `field_resolution_ext`/`_products`, disk caching. Validated at
height 1 there (both a direct resolution and the field trick reproduce `F_2[h_1] ⊗ Λ(ρ_1)`). Its
`docs/chromatic-field-trick.md` details the integration surface (implement `Algebra + Bialgebra`;
`MuAlgebra<false>` is a blanket impl; no `enum_dispatch` registration needed).

Use this lane if the goal is **small-prime (`p = 2, 3`) `E_1` pages** at `n = 2, 3` (the crate's
infinite-resolution forte) or **module Ext amortization** — resolve `gr S(n)` once, then get
`Ext_{gr S(n)}(M, F_p)` for a whole lattice of type-`n` modules cheaply (the green book itself does
such a module computation: `Ext_{Σ(1)}(K(1)_*, K(1)_*(T(m))) = K(1)_*[u_2,…,u_{m+1}] ⊗ E(h_{m+1,0})`).
For `n ≥ 2` this lane still requires the §2 bracket implemented correctly (§3a of the field-trick
doc) — the same substantive work.

---

## 8. Honest limits

- The finite-CE tool computes the **large-prime** answer (`p > n+1`). Small primes (`p = 2, 3`), where
  much of the homotopy-theoretic interest lives, need the field-trick lane plus the May/RMSS
  differentials, which are **not** automated anywhere here.
- Neither tool touches the genuinely hard chromatic content beyond the `E_1`: the May/Ravenel
  differentials reassembling `H^*(gr S(n)) ⇒ H^*(S(n))` at small primes, and the lift `K(n) → E_n`
  carrying the integral information. This is an `E_1`-and-verification engine, not a
  chromatic-answer machine.
- `n ≥ 5` at small primes, and full `n ≥ 6` even at large primes, are likely computationally
  prohibitive by brute force. Manage expectations: the realistic, defensible win is `n = 4`
  verification and *partial* `n = 5` data.

## 9. References

- **D. Ravenel, *Complex Cobordism and Stable Homotopy Groups of Spheres* (green book), ch. 6** —
  the primary source. `E_0 S(n)` structure and Thm 6.3.3 (`L(n)`, bracket, restriction); Lemma 6.3.11
  (`ζ_n = Σ_j h_{n,j}`, `ρ_n = Σ_j h_{2n,j}` at `p = 2`); Thm 6.3.21 (`H^*(S(1))`); Thm 6.3.22 / 6.3.24
  (`H^*(S(2))` for `p > 3` / `p = 3`, the latter Henn-corrected). Use a current printing (the
  restriction formula was fixed between editions).
- **A. Salch, *Ravenel's May spectral sequence collapses immediately at large primes*
  (arXiv:2312.17185)** — `L(n,n)`, the collapse for `p > n+1`, the dimension table (2, 12, 152, 3440;
  conjectural 128512, 7621888), and the `n = 4` computation ([15] therein, the height-shifting method).
- Shimomura–Yabe, Shimomura–Wang (`H^*(S(2))`, `p ≥ 5` / `p = 3`); GHMR, Beaudry, Beaudry–Goerss–Henn
  (height 2 at `p = 2, 3`); Devinatz–Hopkins (`H^*_c(G_n; (E_n)_*) ⇒ π_* L_{K(n)} S`, the target this
  feeds).
