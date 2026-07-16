# Handoff: chromatic (Morava stabilizer) computations

**Status.** Clean branch off `master`. Contains one validated, self-contained deliverable — the
height-1 algebra `gr S(1)` and a smoke test reproducing `H^*(S(1))` at `p = 2` — plus this brief,
which lays out the recommended next direction and the mathematics to execute it. Read this top to
bottom before writing code; the math is subtle and there is a documented trap (see §6).

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
| `crates/algebra/src/algebra/morava_stabilizer.rs` | `MoravaStabilizerAlgebra`: `gr S(1) = u(L(1)) = Λ(x_1) ⊗ F_2[x_2]` as an `Algebra + Bialgebra`. Validated, unit-tested. |
| `examples/chromatic_grs1.rs` | Resolves `F_2` over `gr S(1)`, checks the chart equals `F_2[h_1] ⊗ Λ(ρ_1)` (Ravenel 6.3.21(a)). Self-contained (no field-trick machinery). |

Run it: `cargo run --example chromatic_grs1 -- 32 18`. Tests: `cargo test -p algebra morava`.

This is *restricted*/small-prime data (§5). It is here as a validated worked example — proof that the
codebase accepts a new Morava algebra and returns the right answer, and a reference for the exact
height-1 structure — not as the base for the finite-CE tool, which is a different construction.

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

> **⚠ Verify the bracket's second subscripts against the source before coding.** My OCR of green
> book 6.3.3 reads `δ^l_{i+j} x_{i+k,j} − δ^j_{k+l} x_{i+k,l}`, while Salch's eq. (8) reads
> `δ^l_{i+j} x_{i+k,l} − δ^j_{k+l} x_{i+k,j}` — the two `x` second-subscripts (`j` vs `l`) appear
> swapped between transcriptions. This is irrelevant at `n = 1` (all `δ = 1`, both terms are
> `x_{i+k,0}`, and the bracket vanishes) but decisive for `n ≥ 2`. Resolve it directly against a
> clean copy of Thm 6.3.3 / Ravenel Thm 1.4 before trusting any `n ≥ 2` output. (Note also that
> Ravenel corrected the restriction formula between editions — use a current printing.)

**`L(n,m)` (Salch, following Ravenel Thm 1.4).** The finite quotient of `L(n)` by the span of
`{x_{i,j} : i > m}`. For **large primes `p > n+1`**, `m = ⌊pn/(p−1)⌋ = n`, so `L(n,n)` has basis
`{x_{i,j} : 1 ≤ i ≤ n, j ∈ Z/n}` — exactly `n^2`-dimensional — and the restriction is **trivial on
`L(n,n)`** (`ξ` sends `x_{i,j}` to `x_{i+n,·}`, which is `0` in the quotient). Hence at large primes
`H^*(L(n,n))` is *ordinary* Lie-algebra cohomology, and the collapse `H^*(S(n)) = H^*(L(n,n))` holds
(Salch's main theorem).

---

## 3. Recommended mission: a finite Chevalley–Eilenberg cohomology tool

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
| 4 | 3440 | 65536 | **being written up now (Salch); verifying this independently is the near-term prize** |
| 5 | 128512 (conjectured, from a generating function) | 2^25 ≈ 33.5M | **open — no one has it** |
| 6 | 7621888 (conjectured) | 2^36 | open |

**Crucial simplification for validation:** the *total* `F_p`-dimension of `H^*(L(n,n))` is
grading-independent, so you can match the table above **without solving the grading puzzle** (§6) —
just build the Lie algebra, form `Λ^•`, and rank the differentials. Get the `(s,t,u)`-graded chart
later; the total dimension is the honest first milestone.

**Feasibility gradient.**
- `n ≤ 3` (complex ≤ 512): trivial. Your correctness gate — must reproduce 2, 12, 152.
- `n = 4` (complex 65536): very reachable. The complex splits by internal degree into small blocks,
  so you never form a 65536-square matrix. **This is the highest-value target**: an independent
  machine confirmation of Salch's in-progress `H^*(L(4,4)) = 3440`, by a method orthogonal to his
  (he derives it via deformations/spectral sequences, not computation). Given the field's track
  record — Ravenel's own first-edition `H^*(S(2))` at `p=3` was wrong until Henn caught it — an
  independent check has real value, and this is a natural collaboration.
- `n = 5` (complex 2^25 ≈ 33.5M): the moonshot. Brute force is likely what pushed Salch to a smarter
  "height-shifting" method. Grading decomposition helps enormously (compute per internal degree), and
  **partial `n = 5` data — specific internal degrees, or the low/high cohomological corners — is
  reachable and genuinely novel**. A confirmed total dimension of 128512 (matching the conjectural
  generating function) would be a real result; full brute force may need serious sparse linear
  algebra or borrowing height-shifting.

**Concrete first steps.**
1. New module, e.g. `crates/algebra/src/lie/morava_lie.rs` (or a small standalone crate): a struct
   holding `n`, `p`, the basis `{x_{i,j} : 1 ≤ i ≤ n, j ∈ Z/n}` with a fixed index order, and the
   structure constants `c^a_{bc}` from the §2 bracket (with `m = n`, large `p`). Unit-test the bracket
   (antisymmetry, Jacobi) at `n = 2, 3`.
2. Build `Λ^•(V)` (basis = subsets of the `n^2` generators = bitmasks) and the CE differential as
   sparse `F_p` matrices (`fp::matrix`). Rank each `d_k`.
3. Sum to `dim H^*`; assert 2, 12, 152 at `n = 1, 2, 3`. **Gate.**
4. Run `n = 4`; confirm 3440. Compare notes with Salch.
5. Attempt `n = 5` (total dim, then per-internal-degree pieces as far as compute allows).

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
