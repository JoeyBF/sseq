# The Adams E2 page of Cτ from the classical resolution

Design and results for `src/ctau.rs`, `examples/ctau_e2.rs`, `ctau_check.py` and `ctau_plan.py`.
Everything here is 2-primary and C-motivic. Numbers were measured on `sfx7` against a copy of the
stem ≤ 100 subset of the stem-400 resolution, on 24 threads at `nice 19`, unless stated otherwise.

## 1. Summary

* `Ext_{A_mot/τ}(F_2, F_2) ≅ Ext_A(F_2, Q)` with `Q = F_2[q_0, q_1, ...]`, an algebra over the
  classical Steenrod algebra `A` in halved degrees. It is computed as the cohomology of the
  untwisted cochains `Hom_A(P, Q)` on the saved minimal resolution `P` of the classical sphere. No
  new resolution is needed, and none is computed: the example refuses to run if the save lacks a
  bidegree it would read.
* Motivic stem `N` reads the backbone through classical stem `(N + 1) / 2` rounded down, so the
  stem-400 resolution reaches motivic stem 800 as far as data goes.
* The ring structure is computed from the backbone alone. Multiplication by `h_0` is
  multiplication by `q_0`; by `h_{j+1}` it is a Cartan term of the differential; by any class `β`
  it is a sum over chains of differentials weighted by a cocycle of `β` on the bar resolution,
  which costs a little linear algebra in degrees up to that of `β` only. Lin's saved chain maps
  give the same products where they exist, but each was saved only on the corner Lin's algorithm
  read, so they are a cross-check rather than the source.
* Two-cell complexes `Cτ ∧ C(2)`, `Cτ ∧ C(η)`, `Cτ ∧ C(ν)`, `Cτ ∧ C(σ)` are cones on the same
  cochains, with the maps of the bottom and top cell and the action of the `h_j`.
* Against Isaksen's tables through stem 70: 64 of 65 generator tridegrees, and all 70 of the
  `h_0`, `h_1`, `h_2` extensions. The one miss is a misprint in the paper (section 5).
* The limit is linear algebra, not data. The complete E2 page costs about 17 CPU-hours through
  stem 300 and about 2000 through stem 400; at stem 798 the largest coboundary is 2.5e7 square,
  far out of reach of dense elimination. What is in reach near stem 800 is the part of bounded
  Chow degree `k = stem + filtration - 2 weight`, which contains everything of Adams filtration at
  most `k`: about 225 CPU-hours for `k ≤ 10` and 1e4 for `k ≤ 15` (section 7).

## 2. The isomorphism

### Statement

Write `A_mot` for the C-motivic Steenrod algebra over `M_2 = F_2[τ]`. Its dual is
`M_2[τ_0, τ_1, ..., ξ_1, ξ_2, ...] / (τ_i^2 = τ ξ_{i+1})` with (degree, weight)
`|τ_i| = (2^{i+1} - 1, 2^i - 1)` and `|ξ_i| = (2^{i+1} - 2, 2^i - 1)`, and

    Δ ξ_n = Σ_i ξ_{n-i}^{2^i} ⊗ ξ_i,        Δ τ_n = τ_n ⊗ 1 + Σ_i ξ_{n-i}^{2^i} ⊗ τ_i.

Modulo `τ` the relation becomes `τ_i^2 = 0`, so `(A_mot/τ)_* = P_* ⊗ E_*` as an algebra, with
`P_* = F_2[ξ_1, ξ_2, ...]` a sub Hopf algebra and `E_* = E[τ_0, τ_1, ...]` the quotient Hopf
algebra, primitively generated. `P_*` is the classical dual Steenrod algebra with every degree
doubled and weight equal to half the degree.

`Ext_{E_*}(F_2, F_2) = F_2[q_0, q_1, ...] =: Q` with `q_i = [τ_i]` in cohomological degree 1, and
the coaction of `P_*` on `Q` is multiplicative with

    q_n ↦ Σ_i ξ_{n-i}^{2^i} ⊗ q_i,

read off `Δ τ_n`. `Q` is graded by polynomial degree `k`, and `Q = ⊕_k Q_k` as comodules.

**Claim.** `Ext_{A_mot/τ}^{s, t, w}(F_2, F_2) ≅ Ext_{P_*}^{s - k, *}(F_2, Q_k)` with
`k = t - 2w`, as rings.

Proof. The extension `P_* → (A_mot/τ)_* → E_*` gives a Cartan–Eilenberg spectral sequence
`E_2^{a, k} = Ext_{P_*}^a(F_2, Ext_{E_*}^k(F_2, F_2)) ⇒ Ext_{A_mot/τ}^{a + k}`. Every `ξ_i` has
`t - 2w = 0` and every `τ_i` has `t - 2w = 1`, so a class of `E_2^{a, k}` has `t - 2w = k`. The
differential `d_r` lowers `k` by `r - 1` and preserves `t` and `w`, so it vanishes for `r ≥ 2`,
and since `k` is determined by `(t, w)` the filtration is a grading: there are no extensions,
additive or multiplicative. ∎

By Gheorghe–Wang–Xu this is also the E2 page of the algebraic Novikov spectral sequence, with
`Q` the associated graded of `BP_*` for the powers of `I = (2, v_1, ...)`, `q_i ↔ v_i`, and the
coaction that of `BP_*BP` modulo `I^2`. Nothing here uses that identification.

### Halving

A `P_*`-comodule in even degrees is an `A_*`-comodule after halving degrees. `Q_k` is not in even
degrees, but `Σ^{-k} Q_k` is (each `q_i` has degree `2^{i+1} - 1`), so halve after removing one
from the degree of each `q_i`: then `|q_i| = 2^i - 1`, the degree of `ξ_i`, and `q_0` has degree
0. With `u` the halved internal degree,

    Ext_{A_mot/τ}^{s, t, w} = Ext_A^{a, u}(F_2, Q_k),   a = s - k,  u = w,  t = 2u + k,

    motivic (stem, filtration, weight) = (2u - a, a + k, u).

So `a = 2 weight - stem` is the Novikov filtration, and `k = stem + filtration - 2 weight` is the
Chow degree. `k = 0` is `Ext_A(F_2, F_2)` itself: the classical E2 page sits in `Cτ` at twice the
stem plus the filtration, `h_j ↦ h_{j+1}`. The motivic `h_0` is `q_0`.

### The module `Q`

The code holds `Q` as a left `A`-module, which is what cochains need. A left comodule is a right
module, and turning it into a left module with the antipode gives

    P(R) · m = Σ ⟨P(R), χ c⟩ m'   over the terms c ⊗ m' of the coaction on m,

so the coefficients that appear are the conjugates `ζ_{n-i}^{2^i}`. This action lowers the degree
by `|P(R)|`, satisfies the Cartan formula (the coaction is multiplicative and `A` is
cocommutative), and never lengthens a monomial: writing a monomial of `Q_k` as `q_0^{k - l} m`
with `m` free of `q_0` and of length `l ≤ k`, the action sends `m` to monomials of length at most
`l`. The basis of `Q_k` is therefore the `q_0`-free monomials of length at most `k`, the action
does not depend on `k`, and multiplication by `q_0` is the inclusion `Q_k ⊂ Q_{k+1}`.

### What was verified, and how

* The module axiom `P(R)(P(S)m) = (P(R)P(S))m` against the Milnor product, the Cartan formula,
  and the degree and length statements: unit tests in `src/ctau.rs`, through degree 14.
* `δ² = 0` and that every product of cocycles is a cocycle: asserted on every class the example
  touches (`CtauExt::class` panics otherwise).
* Convergence to classical Ext. Forgetting the weight, `⊕_w Ext(Cτ)` is the E_2 page of the
  classical Cartan–Eilenberg spectral sequence of `E → A → A//E` (the τ-Bockstein spectral
  sequence with τ inverted), which converges to `Ext_A^{s, t}` with differentials that raise `s`
  by one at fixed `t`. So for each `t` the Euler characteristic in `s` must match that of
  classical Ext, and each rank must be at least the classical one. Both hold for every `t ≤ 36`
  (all that is complete in the stem-70 export). This tests the coaction and the linear algebra
  independently of anything motivic.
* The identification with `A_mot/τ`, weights included: Isaksen's tables (section 5).
* Not re-run here: branch `ctau-plan-n400` has a test comparing the bigraded ranks with a minimal
  resolution over `A_mot/τ` through (stem 30, filtration 15). Running it resolves over `A_mot/τ`,
  which the rules for this work forbid without asking.

## 3. The computation

`P_• → F_2` is the saved minimal resolution over `A`, with generators `x_i` of `P_a` in degrees
`t_i`. A cochain of filtration `a` and weight `u` is an `A`-linear map `c: P_a → Q` lowering
degree by `u`, determined by the values `c(x_i) ∈ Q` of degree `u - t_i`. The cochains have basis
`x_i ↦ m`, sorted by the length of `m` so that those of `Q_k` are a prefix. The coboundary is

    (δ c)(z) = Σ θ · c(x)        over the terms θ x of d z.

This is the untwisted tensor resolution of sseq PR #279 for the module `M = Q_k^∨`: `Hom_A(P ⊗
M, F_2) ≅ Hom_A(P, DM)`, with the antipode absorbed into the left action on `DM = Q_k`.

Everything else is the same formula with `d` replaced by another map. For a map `φ` on
generators and an exponent sequence `S`, write

    (φ^*_S c)(z) = Σ P(R - S) · c(x)     over the terms P(R) x of φ(z) with R ≥ S,

implemented once as `pullback`. `δ = d^*_0`.

**h_0.** Multiplication by `q_0`: the same cocycle, read in `Q_{k+1}`.

**h_{j+1}.** The classical `h_j` is the extension `C(h_j)` with cells `b`, `t` and `Sq^{2^j} b =
t`. Its connecting map on `Hom_A(P, Q ⊗ DC(h_j))` is, by the Cartan formula and `Δ P(R) =
Σ P(R') ⊗ P(R'')`, the part of `δ` in which `Sq^{2^j} = P(2^j)` falls on the cell:

    h_j · c = d^*_{(2^j)} c.

No chain map is needed. For `Q_0 = F_2` this is the usual rule that `h_j` products are the
coefficients of `Sq^{2^j}` in the differential.

**Classes with a saved chain map.** If `f: P_{•+s} → P_•` lifts a classical class `g`, then
`g · c = f^*_0 c = c ∘ f`. `ChainMapTerms::read` reads the saved steps of Lin's maps
(`products/lin_ring/x_(n, s, i)`) and lifts nothing; a product needing a step that was not saved
is reported in `Ctau_unknown.csv`.

**Any class.** Let `β ∈ Ext^{a'}(Q_{k'})` and let `Γ: B_{a'}(A) → Q_{k'}` be a cocycle for it on
the normalised bar resolution `B_j(A) = A ⊗ Ā^{⊗ j}`. For a cocycle `c: P_a → Q_k`,

    (β · c)(z) = Σ Γ[θ'_{a'} | ... | θ'_1] · (θ''_{a'} ··· θ''_1 c(z_0))

over the chains `z = z_{a'} → z_{a'-1} → ... → z_0`, where `θ_j z_{j-1}` is a term of `d z_j` and
`z_0` is a generator of `P_a`, and over the splittings `θ_j ↦ θ'_j ⊗ θ''_j` of each coproduct
with `θ'_j ≠ 1`. In the Milnor basis a splitting of `P(R)` is a choice of `R' ≤ R`.

Proof. `B_•(A) ⊗ Q_k` with the diagonal action is a free resolution of `Q_k`, with contracting
homotopy `s ⊗ 1`, `s(θ[w]) = 1[θ | w]`. Lifting `c` along it gives `F_j: P_{a+j} → B_j ⊗ Q_k`,
`F_0(x) = 1[] ⊗ c(x)` and `F_j(z) = (s ⊗ 1) F_{j-1}(dz)` on generators, which unwinds to the sum
above without `Γ`. Then `μ (Γ ⊗ 1) F_{a'}` represents the Yoneda composite of `β` after `c ⊗ 1`,
which is the product. ∎

`Γ` is `g ∘ ψ` for the cocycle `g: P_{a'} → Q_{k'}` of `β` on the backbone and the comparison
`ψ: B_•(A) → P_•`, `ψ_j = σ_j ψ_{j-1} d`, where `σ_j` is a section of `d: P_j → P_{j-1}` on its
image. That is the only linear algebra the product needs, and it lives in filtrations up to
`a'` and degrees up to the degree `u_β` of `β`, however large the class being multiplied is
(`BarComparison`). The sum is pruned to the words on which `Γ` is nonzero (`BarCocycle::cup`).
`h_j` is the case `a' = 1`, `Γ[θ] = 1` exactly for `θ = Sq^{2^j}`.

**Two-cell complexes.** `Cells` lists the cells of the classical complex `M` being doubled and
the action on its dual; cochains are maps `P_a → Q ⊗ DM` and `pullback` adds the Cartan terms in
which part of an operation moves a cell. For `C(h_j)` this is the cone of `h_j`:
`δ(c·t) = (δ c)·t + (h_j c)·b`. The attaching map 2 is `q_0`, which is not a classical class, so
`Cτ ∧ C(2)` is instead `Q/q_0`: the monomials of length exactly `k`, a window of the sorted
basis. The bottom cell is the inclusion of cochains, the top cell reads the coordinates on the
top cell (for `C(2)`, the Bockstein: the coboundary of a lift, which is shorter).

`C(h_j)` here is the cofiber of the map detected by the motivic `h_{j+1}`: `C(η)`, `C(ν)`,
`C(σ)` for `j = 0, 1, 2`.

## 4. Range

Stems `0..=N` and filtrations `0..=S` of `Cτ` read `P_a` for `a ≤ S + 1` in degrees `t ≤ (N + a
+ 1) / 2`, that is in classical stems up to `(N - a + 1) / 2` (`CtauExt::backbone_range`). The
resolution through classical stem 400 therefore serves motivic stems through 800, with no
restriction on the filtration: above its vanishing line the backbone is zero, not missing.
`ctau_plan.py` accepts stem 800 and refuses 802 on the ranks of the stem-400 resolution.

The `q_0` direction costs no backbone. A monomial of degree `e` free of `q_0` has length at most
`e`, so at weight `u` the cochains of `Q_k` stop growing at `k = u`, and beyond that every class
is a `q_0`-multiple. Only stem 0 has infinite `h_0`-towers.

Each polynomial degree `k` is its own complex. Bounding `k` is therefore exact, not a
truncation: it computes precisely the classes of Chow degree at most `k`, and those include
every class of Adams filtration at most `k` because `k = s - a ≤ s`. Products add Chow degrees.

## 5. Validation against Isaksen

`python3 -I ctau_check.py <export> <tables>` on the export through stem 70, filtration 36, with
products by every class of weight at most 18:

| check | result |
|---|---|
| generators of the Cτ E2 page (`ss_Ctau_E2.csv`): nonzero group at the tridegree | 64 of 65 |
| `h_0`, `h_1`, `h_2` extensions (`ss_Ctau_h{0,1,2}.csv`): nonzero product into the listed tridegree | 70 of 70 |
| Euler characteristic and rank bound against classical Ext | all 37 complete degrees |
| `h_1`, `h_2`, `h_3` from the differential against their bar cocycles | 4588 products, 0 differ |
| Lin's saved maps against bar cocycles (the doubles of `c_0`, `Ph_1`, `Ph_2`, `h_4`, `d_0`) | 177 products, 0 differ |
| commutativity `β·α = α·β` | 76937 ordered pairs, 0 differ |
| associativity `β(γα) = (βγ)α` | 4158365 triples, 0 differ |
| exactness of the cofiber sequences for `C(2)`, `C(η)`, `C(ν)`, `C(σ)` (ranks of the complex and of both cell maps) | 18429 tridegrees, 0 inexact |

The miss is `\overline{τ h_2^2 g^3}`, which the generator table of "Stable stems" puts at
`(67, 13, 39)`. The same paper's `h_0` and `h_2` extension tables put it at `(67, 13, 38)`, which
is where `τ h_2^2 g^3 ∈ (66, 14, 39)` forces it, and where this computation has a class with the
listed `h_0` and `h_2` extensions and none at weight 39. The generator table has a misprint.

The tables state the `h_i` extensions as hidden extensions between named classes. The check is
that the product is nonzero between the two tridegrees; it does not match names to basis
vectors.

## 6. Measured costs

The export of ranks and `h_0, ..., h_4` on `Cτ` (no cofibers, no products):

| stem `N`, filtration `N/2` | classes | largest cochain space | wall | peak memory |
|---|---|---|---|---|
| 70, 36 | 5040 | 309 | 0.3 s | 0.2 GB |
| 100 | 19317 | 926 | 3.5 s | 0.5 GB |
| 120 | 41779 | 1674 | 7.3 s | 2.0 GB |
| 140 | 82967 | 2838 | 29 s | 7.4 GB |
| 160 | 151788 | 4665 | 105 s | 22 GB |
| 180 | 261315 | 7732 | 452 s | 67 GB |
| 200 | 431613 | 12346 | 2241 s | 207 GB |

The coboundaries are very sparse: through stem 120 there are 248668 cochains (over every
filtration and weight held) and 2090328 nonzero coboundary entries, 8.4 per cochain, and 41779
classes.

The stem-70 export with all four cofibers and the products by the 363 classes of weight at most
18 takes about 30 s and 0.7 GB, nearly all of it in the products.

`src/ctau.rs` is written for checking, not for scale: it holds every coboundary densely, and
reduces one matrix per polynomial degree `k` where one per (filtration, weight) serves every `k`
(columns are sorted by length, so the rank for `Q_k` is the number of pivots in a prefix; branch
`ctau-plan-n400` does this for ranks). Memory and time in the table are dominated by that
redundancy, which grows with the number of polynomial degrees.

Dense row reduction in `fp` on one core takes `1.5e-12 s × rows × cols × min(rows, cols)`
(measured at 8000, 16000 and 32000 square, full rank). `ctau_plan.py` computes that product for
every coboundary from the ranks of the backbone alone:

| range | largest coboundary | dense size | CPU-hours, one reduction per coboundary |
|---|---|---|---|
| stem ≤ 200 | 12908 × 11670 | 18 MB | 0.03 |
| stem ≤ 300 | 84822 × 83572 | 0.8 GiB | 17 |
| stem ≤ 400 | 364695 × 369937 | 15.7 GiB | 2.2e3 |
| stem ≤ 500 | 1364411 × 1290783 | 205 GiB | 1.4e5 |
| stem ≤ 600 | 4.07e6 × 3.98e6 | 1.9 TiB | 4.8e6 |
| stem ≤ 798 | 2.49e7 × 2.43e7 | 70 TiB | 1.8e9 |
| stem ≤ 798, `k ≤ 4` | 4611 × 4587 | 3 MB | 0.04 |
| stem ≤ 798, `k ≤ 6` | 15842 × 15946 | 30 MB | 1.6 |
| stem ≤ 798, `k ≤ 8` | 41069 × 40826 | 0.2 GiB | 26 |
| stem ≤ 798, `k ≤ 10` | 88071 × 86836 | 0.9 GiB | 225 |
| stem ≤ 798, `k ≤ 15` | 342730 × 333649 | 13.3 GiB | 1.1e4 |
| stem ≤ 798, `k ≤ 20` | 826820 × 823257 | 79 GiB | 1.4e5 |
| stem ≤ 798, `k ≤ 30` | 2.50e6 × 2.50e6 | 729 GiB | 3.3e6 |

The CPU-hours assume full-rank dense matrices and are an upper bound for this method; the
coboundaries are sparse and of low rank relative to their size, and the GPU row reduction used
for the stem-400 resolution applies to every matrix here that fits a card. They do not include
reading the backbone: `nassau_differential` of the stem-400 snapshot is 104 GB on disk.

## 7. Reaching stem 800

**What the saved data covers.** The resolution: everything, through motivic stem 800. Lin's
chain maps: not needed. They were saved only where Lin's ring algorithm reads them (the map of
`c_0` has steps for input stems 8 to 23 and filtrations below 8, for instance), which decides the
classical ring but leaves most of the action on `Q` unknown: through stem 70, 408 nonzero
products are decided by saved steps and 978 pairs are not. The bar cocycle products need no
chain maps.

**What limits the stem.** The size of the coboundaries, which is the number of generators of the
backbone times the number of monomials of `Q`. The complete page is in reach through stem 400
(2e3 CPU-hours, or far less on GPUs: the largest matrix is 16 GiB) and not beyond about stem 500
by dense elimination on any hardware here.

**Proposal.**

1. Stem 400, complete, all `k`. Needs the single-pass reduction and a backbone reader that streams
   one filtration at a time instead of loading the resolution (the loader holds every
   differential densely). About 2e3 CPU-hours of reduction at most.
2. Stem 798 in bounded Chow degree, raising `k` while the cost allows: `k ≤ 10` is about 225
   CPU-hours and contains every class of filtration at most 10; `k ≤ 15` about 1e4 CPU-hours (or
   GPU: 13 GiB matrices); `k ≤ 20` wants GPUs throughout. A bound on `k` also thins the backbone
   that has to be read, because an operation `P(R)` acts on monomials of length at most `k`
   through products of at most `k` coefficients of the coaction, and most of the Milnor basis
   acts by zero.
3. Products at scale: `h_0` free, `h_{j+1}` one extra pass over each differential, any class of
   low degree by its bar cocycle. The cost of a product by `β` grows with the number of chains of
   length `a'(β)` in the backbone, so classes of Novikov filtration 1 and 2 (the `h_j`, the
   image of J, `c_0`, `d_0`, `g`) are cheap everywhere, and a class of large `a'` is better
   written as a product.
4. Beyond dense elimination: the coboundary is strictly triangular for the filtration by the
   degree in `Q` (the algebraic Atiyah–Hirzebruch filtration, `E_1 = Ext_A ⊗ Q`), with `d_1 = Σ
   h_j ⊗ P(2^j)`. Elimination in that order, or the chain of Bockstein spectral sequences
   `Ext(Q/(q_0, ..., q_n)) ⊗ F_2[q_n] ⇒ Ext(Q/(q_0, ..., q_{n-1}))` starting from classical Ext,
   keeps intermediate pages near the size of the answer. Neither is implemented or measured, and
   the complete page at stem 800 depends on one of them working.

A resolution over `A_mot/τ` directly is not proposed: its ranks are the answer, but its modules
have the dimensions of the classical Steenrod algebra in degrees up to 800 and more, orders of
magnitude past the stem-400 computation.

## 8. Not done here

* The E2 page of the motivic sphere, and with it the maps `S → Cτ → S^{1,-1}` on E2 pages. The
  sphere's Ext acts on `Ext(Cτ)` through the unit map, whose image is the classes that survive
  the τ-Bockstein spectral sequence; the action of that image is covered by the products above,
  but the image itself is not identified. Branch `ctau-plan-n400` computes τ-Bockstein differentials from
  the backbone as ranks (`koszul_lift`, `tau_lift`); they would have to be redone on the bases of
  this export.
* The Adams differentials of `Cτ` (the algebraic Novikov differentials). `novikov_lift` on the
  same branch has the first one, again as ranks.
* Products on the cofibers by classes other than the `h_j`, and two-cell complexes on other
  attaching maps. Both are the formulas of section 3 with more cells.
* Names. Classes are indexed by basis vectors; nothing matches them to Isaksen's names.
* Massey products, by request.

## 9. Running it

    cargo build --release --no-default-features --features concurrent,logging,nassau --example ctau_e2
    printf 'S_2\n<copy of the save>\n<out>\n70\n36\nC2,Ceta,Cnu,Csigma\n18\ny\n' | target/release/examples/ctau_e2
    python3 -I ctau_check.py <out> <stemlit>/out/tables
    python3 -I ctau_plan.py <rank.csv> 798 10

The prompts are the module, its save directory (a copy: loaders can write), the output directory,
the largest motivic stem and filtration, the complexes, the largest weight of a class `β` to
multiply by (0 for none), and whether to read Lin's maps. The files are described at the top of
`examples/ctau_e2.rs`.
