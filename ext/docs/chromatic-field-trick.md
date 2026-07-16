# Handoff: chromatic (Morava stabilizer) computations via the field trick

**Status:** phase 1 (§6) done — see the update box below. This document hands off a
self-contained investigation: whether Nassau's tensor/field trick — already implemented in this
crate for the mod‑2 Steenrod algebra — can be pointed at the **Morava stabilizer algebra** to
compute the algebraic input to K(n)‑local (chromatic) homotopy.

> **Update (phase 1 complete): height 1 at p = 2 implemented and validated against Ravenel 6.3.21(a).**
>
> `gr S(1) = u(L(1)) = E_0 S(1)^*` is implemented as a standalone `Algebra + Bialgebra` in
> `crates/algebra/src/algebra/morava_stabilizer.rs` (`MoravaStabilizerAlgebra`). No `enum_dispatch`
> registration was needed: the resolution engine is fully generic over `CC::Algebra`, and
> `MuAlgebra<false>` is a blanket impl over `Algebra`.
>
> **Get the algebra, not just the grading, right.** `L(1)` is abelian (green book Thm 6.3.3), but its
> restriction is *not* trivial: `ξ(x_1) = 0` and `ξ(x_i) = x_{i+1}` for `i ≥ 2`. So the restricted
> enveloping algebra collapses to
>
> ```text
> gr S(1) = Λ(x_1) ⊗ F_2[x_2],   |x_1| = 2 (exterior),   |x_2| = 6 (polynomial),
> ```
> — **not** the exterior algebra `Λ(x_1, x_2, …)` on all the Lie generators. Grading the `x_i` by the
> topological degree `2(2^i−1)` and treating them all as independent primitives makes the restriction
> and the coproduct cross terms inhomogeneous; they silently vanish and you get the degenerate
> `F_2[h_1, h_2, …]`. That is precisely the "wrong but plausible-looking chart" §5 warns about — the
> first implementation fell into it and had to be corrected once the reference (green book 6.3) was in
> hand. The correct answer (Ravenel Thm 6.3.21(a); at height 1 the May SS collapses, so this is
> already `H^*(S(1))`) is
>
> ```text
> Ext_{gr S(1)}(F_2, F_2) = P(h_1) ⊗ E(ρ_1) = F_2[h_1] ⊗ Λ(ρ_1),
>     h_1 = h_{1,0} = [t_1] ∈ (s,t) = (1, 2),   ρ_1 = h_{2,0} = [t_2] ∈ (s,t) = (1, 6).
> ```
>
> The example `ext/examples/chromatic_grs1.rs` resolves `F_2` and confirms this at every computed
> bidegree, and *also* re-derives the same chart through `field_resolution_ext` (`M = F_2`), so the
> full field-trick stack (antipode `χ`, closed-form `δ_Q`) is confirmed to accept this new
> `Bialgebra`. Unit tests live alongside the algebra.
>
> **Lesson for height ≥ 2.** Pin conventions against green book 6.3 *first*. The reader that computes
> Ext is agnostic to the grading you hand it and will happily return a wrong chart; correctness comes
> from (i) the restriction map (Thm 6.3.3) and (ii) a grading in which both the bracket and the
> restriction `ξ(x) = x^{[p]}` (degree `p·|x|`) are homogeneous. For `n ≥ 2` the bracket is genuinely
> non-abelian — the substantive §3a product work.
>
> Not yet done: the `HopfAlgebra` refactor of §3e (optional; the generic `Antipode` already works),
> odd primes, and height ≥ 2.

The one-line thesis: *the field-trick stack is generic over `Bialgebra`, and the associated graded
of the Morava stabilizer algebra is a connected, finite-type, cocommutative `Bialgebra` over
`F_p` with an explicit coproduct/antipode — so the port is "implement a new algebra," not a
rewrite.*

Read this top to bottom before touching code. The math section tells you what you're computing and
why the naive thing fails; the codebase section tells you exactly which traits to implement and what
you get for free; the honest-limits section tells you what this does **not** solve.

---

## 1. What we're computing, and why the field trick is even a candidate

### The object
At prime `p`, height `n`, the chromatic analog of "Ext over the Steenrod algebra" is Ext over the
**Morava stabilizer algebra**:

```
H*_c(G_n; (E_n)_*)  ==>  pi_* L_{K(n)} S
```

`G_n = S_n ⋊ Gal` is the automorphism group of the height‑`n` Honda formal group; `S_n` is the group
of units in the maximal order of the division algebra `D_{1/n}` over `Q_p`, a compact `p`‑adic Lie
group of dimension `n²`. This E2 *is* the K(n)‑local sphere. Chromatic homotopy is largely the
project of computing it and its module variants.

### Why the naive field trick stalls
Over Morava E‑theory `E_n`, `(E_n)_* = W F_{p^n}[[u_1,…,u_{n-1}]][u^±]` is a **complete local ring,
not a field**, and `E_n_* E_n` is a Hopf **algebroid** (distinct left/right units). The field trick
needs a Hopf **algebra** (the "free ⊗ M is free, untwist by the antipode" step is algebra-specific).
Dead end at the `E_n` level.

### Why it becomes a candidate at the residue field
Drop to `K(n)`. `K(n)_* = F_p[v_n^±]` is a **graded field**. Ravenel's *Morava change of rings*
(green book, ch. 6) replaces `K(n)_* K(n)` by an honest `F_p`‑**Hopf algebra** `S(n)` with

```
Ext_{S(n)}(F_p, F_p) = H*(S_n).
```

`S(n)` itself is profinite (not finite type), but its **May-associated-graded is exactly the object
the field trick wants**:

```
gr S(n) = u(L(n)),  the restricted enveloping algebra of a graded restricted Lie algebra L(n),
```

- **connected, finite type** (finite-dimensional in each internal degree),
- **cocommutative** (so the antipode is `χ = -1` on primitives — even simpler than Steenrod),
- with an **explicit coproduct and antipode read off the formal group law** (Ravenel 6.3).

That explicitness is the whole reason this is worth trying. Contrast the *secondary* Steenrod algebra
(the d2 machinery in this crate): a computable secondary coproduct is essentially unavailable, which
is the wall for lifting the trick to the secondary level. For `gr S(n)` there is no such wall — the
Hopf structure is a clean combinatorial formula. `H*(gr S(n))` is the E1 of a May spectral sequence
(a Koszul / restricted-Lie-algebra-cohomology computation), and this is where the tensor trick lives.

### Where the amortization would actually pay off
The field trick's payoff is: resolve the base once, get every module's Ext as a closed-form add-on.
That needs *many* modules over one Hopf algebra. Chromatic land supplies them, unlike ko/tmf:
- the algebraic **chromatic spectral sequence** pieces (`E_n_*/I_k` towers, the `M_n^i`),
- **Smith–Toda complexes** `V(k)` and generalized Moore spectra `S/(p^{i_0}, v_1^{i_1}, …)` — a whole
  lattice of type‑`n` finite complexes and their smashes,

all `gr S(n)`‑modules, all wanting their K(n)‑local Ext. This is the regime the amortization was
built for.

---

## 2. What already exists in this crate (all generic over `Bialgebra`)

The field-trick stack does **not** know anything Steenrod-specific. Concrete pointers:

| Piece | Location | Genericity |
|---|---|---|
| Antipode `χ`, computed from `decompose` + `coproduct`, memoized | `src/ext_algebra/tensor_resolution.rs` (`struct Antipode<A: Bialgebra>`) | any `A: Bialgebra` (see §3e — promote to a `HopfAlgebra` trait) |
| Closed-form coboundary `δ_Q` of `Q• = P• ⊗ M` | `src/ext_algebra/tensor_resolution.rs` (`TensorResolutionDifferential`) | any `CC: FreeChainComplex`, `CC::Algebra: Bialgebra` |
| Additive Ext of a module by the trick | `field_resolution_ext[_with_save_dir]` (same file) | ditto |
| Products + Massey (closed-form cup) | `field_resolution_products[_with_save_dir]` (same file); `src/ext_algebra/massey.rs` | ditto |
| The `ExtAlgebra` (cohomology, products, Massey, cohomology transport) | `src/ext_algebra/mod.rs` | generic |
| Disk caching (`SaveKind::TensorDifferential`, chain-map `prod_*`) | `src/save.rs`, wired in `tensor_resolution.rs` + `resolution_homomorphism.rs` | generic |
| Minimal resolution engine | `src/resolution.rs` (`MuResolution<const U: bool, CC: ChainComplex>`) | generic over `CC`, needs `CC::Algebra: MuAlgebra<U>` |

The **only** algebra-specific inputs the whole pipeline consumes are the `Algebra` +
`Bialgebra` (+ `MuAlgebra<false>`) trait methods. Implement those for `gr S(n)` and everything above
runs unchanged, including the disk cache and the amortization we just validated for C2 at stem 100.

---

## 3. The integration surface — exactly what to implement

### 3a. `Algebra` (`crates/algebra/src/algebra/algebra_trait.rs`)
The core methods (mirror `MilnorAlgebra` in `crates/algebra/src/algebra/milnor_algebra.rs`):
- `prime()`, `magic()` (pick a fresh magic u32 for the save headers), `prefix()`
- `compute_basis(degree)` — build the basis up to `degree`
- `dimension(degree)` — number of basis elements in that internal degree
- `multiply_basis_elements(result, coeff, r_deg, r_idx, s_deg, s_idx)` — the algebra multiplication
- `basis_element_to_string` / `basis_element_from_string`
- unstable methods: implement the **stable** specialization (`MuAlgebra<false>`, see below); the
  unstable variants degenerate.

For `gr S(n) = u(L(n))`: pick a PBW/monomial basis in the Lie generators `x_{i,j}` (Ravenel 6.3.1;
generators indexed by `i ≥ 1` and `j ∈ Z/n`, in specific internal degrees). Multiplication is the
restricted-enveloping-algebra product (PBW straightening + the restriction `x^{[p]}`). This is the
substantive piece of work — it is a concrete but nontrivial combinatorial algebra.

### 3b. `Bialgebra` (`crates/algebra/src/algebra/bialgebra_trait.rs`)
Only two methods:
```rust
fn coproduct(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize, i32, usize)>; // Δ(x) = Σ A_ij ⊗ B_ij
fn decompose(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize)>;             // x = product of "easy-coproduct" atoms
```
`Antipode` only ever calls `coproduct` on the atoms returned by `decompose`, and uses
`χ(ab) = χ(b)χ(a)`. For a **primitively generated** Hopf algebra (which `gr S(n)` is):
- `decompose(monomial)` = its list of generator factors,
- `coproduct(generator x)` = `x ⊗ 1 + 1 ⊗ x` (primitive),
- so `χ(x) = -x` on generators, extended by the anti-homomorphism property.

This makes the `Bialgebra` impl for `gr S(n)` **dramatically simpler than Milnor's** (which is the
reference in `milnor_algebra.rs:1724`). The hard part is 3a (the product), not 3b.

### 3c. `MuAlgebra<false>` (`crates/algebra/src/algebra/algebra_trait.rs:317`)
Required by `MuResolution`/`MuFreeModule` (the resolution machinery). For the stable case (`U=false`)
this is `Algebra` plus unstable methods that reduce to the stable ones. Mirror how `MilnorAlgebra`
gets its `MuAlgebra<false>` impl.

### 3d. Building a resolution and invoking the trick
`construct_standard`/`construct_nassau` (`src/utils.rs`) are Steenrod-specific; you won't use them.
Instead, mirror their body with your algebra:
1. `algebra = Arc::new(GrMoravaStabilizer::new(p, n))`
2. trivial module `k` over it (an `FDModule`/`ZeroModule`-style unit), chain complex `ccdz`, then
   `MuResolution::new(...)` and `compute_through_stem(...)` — this is `P•`, the minimal resolution of
   `F_p` over `gr S(n)`. (Because `gr S(n)` is *not* finite, this resolution is genuinely infinite,
   like the sphere — so Nassau-style efficiency matters; but the standard resolver already works,
   and you can add save dirs exactly as the sphere workflow does.)
3. Build the module `M` (a `gr S(n)`-module: `E_n_*/I_k`, `V(k)`, etc.) over the same algebra.
4. `field_resolution_products_with_save_dir(P•, M, Some(dir))` → an `ExtAlgebra` whose cohomology is
   `Ext_{gr S(n)}(M, F_p)`, with products and Massey, all closed form and disk-cached.

Everything from stem-100 C2 (products, Massey, `prod_*` sharing, δ_Q disk cache) then applies
verbatim.

### 3e. Recommended refactor: promote the antipode to a `HopfAlgebra` trait

Today the antipode lives in a standalone `Antipode<A: Bialgebra>` struct (`tensor_resolution.rs`) that
derives `χ` generically (Milnor's recursion on the `decompose` atoms, `χ(ab)=χ(b)χ(a)`), memoized in
its own `DashMap`. That is correct — for a *connected graded* bialgebra the antipode is uniquely
determined by the coproduct, so `Bialgebra` already suffices and no `HopfAlgebra` trait is needed for
correctness. But `gr S(n)` is the first algebra where the antipode is *trivial* (cocommutative,
`χ = −1` on primitives), and the generic recursion is pointless there. That is the motivation to add:

```rust
// crates/algebra — enum_dispatched like Bialgebra
pub trait HopfAlgebra: Bialgebra {
    fn antipode(&self, deg: i32, idx: usize) -> FpVector;
}
```

**Cache placement — the one design decision.** The memoization must be keyed *per algebra instance*
(the `(deg, idx)` key is only meaningful relative to one algebra; a process-global `static` cache
would collide across algebras — Milnor vs Adem vs a profiled subalgebra vs `gr S(n)` all have
different basis elements at the same `(deg, idx)`, and would cross-contaminate tests in one binary).
So the cache lives **on the algebra**, exactly like the existing multiplication/basis caches
(`&self` interior mutability). Factor the generic recursion into a reusable component that holds only
the cache:

```rust
pub struct AntipodeCache { cache: DashMap<(i32, usize), FpVector> }
impl AntipodeCache {
    // memoized generic recursion; recursive sub-calls route back through `get`, so mid-recursion
    // memoization is preserved. Does NOT hold Arc<A> — it lives on the algebra, takes it as a param.
    pub fn get<A: Bialgebra>(&self, alg: &A, deg: i32, idx: usize) -> FpVector { /* … */ }
}

struct MilnorAlgebra { /* … */ antipode_cache: AntipodeCache }
impl HopfAlgebra for MilnorAlgebra {
    fn antipode(&self, d: i32, i: usize) -> FpVector { self.antipode_cache.get(self, d, i) }
}
```

`gr S(n)` skips `AntipodeCache` entirely and returns `−x` directly (no state, cocommutative). `ext`
then deletes its `Antipode` struct, `TensorResolutionDifferential` calls `algebra.antipode(deg, idx)`,
and the field-trick bounds go `CC::Algebra: Bialgebra → HopfAlgebra`.

**Timing.** This is *pure prep* — it changes nothing measurable for the Steenrod computations that
exist today (the antipode is already cached and off the hot path; a closed-form Steenrod antipode
would **not** save time, so do not bother writing one). Land this refactor **in the same pass that
adds `gr S(n)`**, with the new algebra as the immediate second `HopfAlgebra` impl that justifies the
abstraction — not as speculative surgery on the shared `algebra` crate beforehand.

---

## 4. Suggested phasing

1. **Height 1 smoke test.** `gr S(1) = u(L(1))` is tiny; `H*(S_1) = H*(Z_p^×)` reproduces the
   image-of-J / `α`-family pattern. Implement the algebra, resolve `F_p`, check
   `Ext_{gr S(1)}(F_p, F_p)` against the known answer. This validates the `Algebra`/`Bialgebra` impl
   end-to-end on something where the answer is on paper.
2. **Height 2, low degree.** `gr S(2)` and its `H*` are computed in the literature (Ravenel 6.3;
   Shimomura–Yabe at `p ≥ 5`; GHMR / Beaudry at `p = 2,3`). Cross-check the E1 (May) page in a
   modest range — dims, and a few named classes. This is the real correctness gate.
3. **A module, not just the base.** Field-trick a genuine `gr S(2)`-module (start with something
   whose answer is known, e.g. a Moore-spectrum reduction), exercising `field_resolution_products`
   and the amortization.
4. **Explore.** Height ≥ 3, where even the E1 page is prohibitive by hand — this is the only place
   the automation might tell you something new.

## 5. Honest limits — read before promising anything

- **The trick automates the resolution (E1-level Ext). It does not touch the hard chromatic
  content:** the May/Ravenel differentials reassembling `H*(gr S(n)) ⇒ H*(S(n))`, the lift from
  `K(n)` back to `E_n` (the algebroid again — the genuinely harder direction, carrying the integral
  information), and the chromatic reassembly. None of that is here.
- **Consequently:** height 1 is trivial, height 2 is already done by hand, height ≥ 3 is open
  *precisely because the objects are enormous* — which is the one place an amortizing engine could
  help, and also the place where "enormous" may defeat it. Manage expectations accordingly.
- **`gr S(n)` is infinite** (like the Steenrod algebra), so resolving `F_p` over it is the expensive,
  once-per-height artifact — the analog of the days-long stem-256 sphere. Budget for that.
- **Grading/convention pitfalls.** The internal grading on `gr S(n)` (the May weight vs the topological
  degree vs the `v_n`-grading) needs to be pinned down carefully; a wrong grading silently produces a
  wrong (but plausible-looking) chart. Fix conventions against Ravenel 6.3 *before* trusting any
  output, and validate every stage against a known answer (phase 1–2) rather than eyeballing.
- **`p = 2` first.** The rest of this crate is `p = 2`-specialized (the antipode notes say so). Odd
  primes are extra work; the interesting height‑2 computations at `p = 2,3` may force it eventually,
  but start at `p = 2` to reuse the stack.

## 6. First concrete task for the agent

Implement `gr S(1)` (`u(L(1))`, the height‑1 restricted enveloping algebra over `F_2`) as an
`Algebra + Bialgebra + MuAlgebra<false>`, resolve `F_2` over it with the existing `MuResolution`
machinery, and reproduce `Ext_{gr S(1)}(F_2, F_2)`. Success = the additive chart matches the known
height‑1 answer in a small range. That single deliverable de-risks the entire integration (it proves
the generic stack accepts a new `Bialgebra`), and everything after it is "bigger algebra, same
pipeline."

## 7. References
- D. Ravenel, *Complex Cobordism and Stable Homotopy Groups of Spheres* ("green book"), ch. 6 — the
  Morava stabilizer algebra `S(n)`, its Hopf structure, `gr S(n) = u(L(n))`, the change-of-rings, and
  low-height `Ext` computations. **This is the primary source; §6.2–6.3 are the ones to implement
  from.**
- Shimomura–Yabe, and Shimomura–Wang — height 2 `H*` at `p ≥ 5` / `p = 3`.
- Goerss–Henn–Mahowald–Rezk; Beaudry; Beaudry–Goerss–Henn — height 2 at `p = 2, 3`.
- Devinatz–Hopkins — `H*_c(G_n; (E_n)_*) ⇒ π_* L_{K(n)} S` (the descent/E-theory picture, i.e. the
  integral target this residue-field computation feeds into).

## 8. In-repo starting points (verified)
- `crates/algebra/src/algebra/bialgebra_trait.rs` — the 2-method trait to implement.
- `crates/algebra/src/algebra/algebra_trait.rs` — `Algebra` (core) and `MuAlgebra` (`:317`).
- `crates/algebra/src/algebra/milnor_algebra.rs` — full reference `Algebra`/`Bialgebra`/`MuAlgebra`
  impl to mirror (Bialgebra at `:1724`).
- `src/ext_algebra/tensor_resolution.rs` — `Antipode`, `TensorResolutionDifferential`,
  `field_resolution_ext[_with_save_dir]`, `field_resolution_products[_with_save_dir]`; module docs at
  the top explain the untwisting in detail.
- `src/ext_algebra/mod.rs` — `ExtAlgebra`, `CochainCup`, `ExtDifferential`, the cohomology transport.
- `src/ext_algebra/massey.rs` — closed-form Massey (cochain DGA).
- `src/utils.rs` — `construct_standard` / `construct_nassau` as the template for standing up a
  resolution over a new algebra.
- `src/save.rs` — add a `SaveKind` variant if you want a distinct on-disk namespace; the field-trick
  δ_Q cache (`SaveKind::TensorDifferential`) shows the pattern.
- `ext/examples/field_product_structure.rs` — the end-to-end resolve→save→products→module workflow to
  copy for a chromatic module run.
