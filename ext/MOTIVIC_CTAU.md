# The Adams E2 page of Cτ from the classical resolution

Design and results for `src/ctau.rs` and `examples/ctau_e2.rs` (the dense reference),
`src/ctau_scale.rs`, `src/ctau_lazy.rs`, `src/ctau_dual.rs`, `examples/ctau_scale.rs`,
`examples/ctau_orders.rs` and `fp::matrix::SparseEchelon` (the sparse implementations), and the
scripts `ctau_check.py`, `ctau_compare.py`, `ctau_rank_compare.py`, `ctau_legacy.py`,
`ctau_towers_check.py`, `ctau_quotient_check.py`, `ctau_plan.py` and `ctau_plan_quotients.py`.
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
* A native Nassau resolution over `A_C/τ` (upstream master) agrees on every rank by tridegree
  and on the ranks of `h_0, ..., h_3` through stem 140, filtration 70: 82967 classes, no
  mismatch (section 8).
* `src/ctau_scale.rs` is the implementation that scales: a sparse reduction per (filtration,
  weight) ordered by polynomial degree, which yields the `h_0`-towers directly, on a backbone
  streamed from the save. Stem 200 for `Cτ` reduces in 2 s where the dense reference took 1725
  s and 207 GB. Exports are archived through stem 280 for `Cτ` and its four two-cell complexes,
  through stem 320 for `Cτ` with `h_1..h_4`, and through stem 350 for the towers of `Cτ`
  (sections 7, 9, 10). The transposed export (section 11) adds `Cτ` and `Cτ ∧ C(η)`
  with `h_1..h_4` and both cell maps through stem 350, and `Cτ ∧ C(2)` through stem 350.
* The cost of the reduction by columns was its elimination, which adds 5 to 65 times the
  input and grows like the 15th power of the stem. **Reducing the transposed coboundaries from
  the top filtration down finds the same towers with additions of about the size of the
  input**: 17, 36, 71 and 105 times fewer at stems 160, 200, 240 and 280, four rows in five
  reduced as they stand. `src/ctau_dual.rs` is the export on it, with `h_1..h_4` and both cell
  maps computed on functionals instead of cocycles; it agrees with the reduction by columns
  and with the native resolution on every rank and on the rank of every map (section 11).
  `Cτ` with `h_1..h_4` takes 42 s through stem 280, 3 minutes through 320 and under 15
  through 350 (plus 3, 12 and 26 minutes of tables), where the reduction by columns took
  405 s, 32 minutes and 140 GB, and 75 minutes and 210 GB without the `h_j`. What is left is
  building the coboundary, which grows like the 11th to 13th power.
* Each polynomial degree is its own complex, so a bound on the Chow degree `k = stem +
  filtration - 2 weight` is exact. `k <= 4`, `8`, `12` through stem 400 take 81 s, 243 s and
  1330 s for `Cτ` and `Cτ ∧ C(η)` together, and their cost grows like `N^8` to `N^9`: a bound
  buys two orders of magnitude at a given stem and two to four in the power. At stem 798,
  `k <= 4` was stopped by the memory of a session here and is a batch job (section 12).
* The quotients `Q/(q_0, ..., q_{n-1})` cost little. `n = 1` is `Cτ ∧ C(2)`, which as the cone
  of `q_0` was the most expensive complex and as the quotient is the cheapest: 17 million
  classes through stem 350 in 169 s. `n = 4` is done through stem 798, the first of these
  computations at the end of the range: 3.7 million classes in 81 minutes (section 12).
* The complete page has about `N^5.3` towers through stem `N`, about 7e8 at stem 800: the
  output is not the obstacle. The entries built for the coboundaries grow like `N^11` to
  `N^13`, which is what the transposed export costs, and the operation tables like `N^9`.
  Section 13 has the cost models: the complete page of `Cτ` is under an hour at stem 400 and
  half a day to a day at 500; stem 800 is node-years as the code stands, and 96% of what is
  built there lies beyond what the pairing reads, which is the next thing to remove.
* General products are validated but not yet at scale: algebra generators of weight at most 16
  and their products are archived through stem 160.

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
* A native minimal resolution over `A_C/τ`: section 8.

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

## 6. The dense reference: costs, and the size of the problem

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

## 7. The sparse implementation

`src/ctau.rs` is the reference: dense matrices, one cohomology per polynomial degree, written to
be checked. `src/ctau_scale.rs` and `examples/ctau_scale.rs` are what runs at scale, and they
agree with it on every rank of every complex through stem 70 and on the ranks of `Cτ` through
stem 120.

**Towers, one reduction per (filtration, weight).** The coboundary never lengthens a monomial,
so the cochains of a filtration and weight are filtered by polynomial degree and
`⊕_k Ext^{a,u}(Q_k)` is a module over `F_2[q_0] = F_2[h_0]`. Number the cochains by the level at
which they first appear and reduce the coboundary by columns, taking sources in order of level
and pivoting at the latest row. That is the matrix reduction of persistent homology: a reduced
column is a cocycle born at the level of its pivot that becomes a coboundary at the level of its
source, so it generates the tower `g, h_0 g, ..., h_0^{r-1} g` with `r` the difference. No
polynomial degree is reduced twice, and no kernel is computed: the complex is acyclic in the
limit `k → ∞` (above the lowest filtration), so every cocycle is a combination of reduced
columns. The class of a cocycle at level `k` is the set of columns its reduction used that are
still alive at `k`. Every map that commutes with `q_0` is determined by its values on tower
generators, and the export is in that form.

A cochain that is the pivot of a coboundary into its own filtration is skipped (its coboundary
reduces to zero), which halves the work.

`Cτ ∧ C(2)` is the cone of `q_0` here, not `Q/q_0`: the top cell carries the cochains of `Cτ`
one filtration down and one level up, with the identity as attaching term. Its towers all have
length 1, as they must since `h_0` acts by zero.

**`fp::matrix::SparseEchelon`.** A column echelon form over `F_2` with columns stored as sorted
row indices, the pivot of a column its largest index, and reduction only at the leading entry.
The caller owns both orders, rows by numbering and columns by insertion, which is all the
persistence reduction needs, and it has no other dependency on this problem. The working column
is a bit set scanned downward from its last pivot (`SparseWorkspace`).

**Pivot order.** The level must come first. Within a level the order is free, and it decides
the amount of elimination. The coboundary raises the degree of the generator, so it is strictly
triangular for the algebraic Atiyah–Hirzebruch filtration (by the degree in `Q`), with `E_1 =
Ext_A ⊗ Q` and `d_1 = Σ h_j ⊗ P(2^j)`. Rows are numbered so that, within a level, the generators
of lowest degree come last: the pivot of a coboundary is then its shortest Atiyah–Hirzebruch
differential, and the short differentials, which cancel most of `Ext_A ⊗ Q`, are cleared first
with short columns. Measured on `Cτ` through stem 160, in entries of the columns added during
elimination:

| order within a level | entries added | reduced entries kept |
|---|---|---|
| pivot at the highest generator degree | 4.8e8 | 2.0e7 |
| pivot at the lowest generator degree (used) | 1.0e8 | 1.2e7 |
| the same, sources in the opposite order | 2.6e8 | 1.3e7 |

A further ordering by Chow degree is already there (it is the level), and the Bockstein tower in
`q_n` was not tried: the measurements below say elimination is no longer the first cost.

**Bit set against heap.** Finding the largest entry of the working column through a binary heap
of its entries made the C2 reduction at stem 200 take 107 s; a bit set scanned from the top takes
6.4 s for the same 7.3e9 added entries. Columns here are short (tens to thousands of entries in
spaces of tens of thousands of rows), which is the regime where toggling one bit per entry beats
both a heap and dense rows.

**Building the coboundary.** After the changes above this was the whole cost: for each cochain
`x ↦ m`, the operations landing on `x` have to be matched with the operations that act
nontrivially on `m`, and both lists run to thousands while the matches are tens. `OpTables`
computes the coaction on every monomial once and files it by operation, keeping only the
operations that occur in the backbone and only in the degrees where a generator they land on
carries monomials. A block of cochains (one generator, all monomials of one degree) is then built
by walking the generator's operations once, at a cost of the entries produced. The same tables
give the `h_j` as whole cochain maps applied to the tower generators.

**Streaming the backbone.** `StreamedBackbone::read` reads each bidegree's differential off the
save, keeps its nonzero terms, filed by the generator they land on, and drops the vector. The
resolution is never loaded. Stem 0 is synthesised (`d x_a = Sq^1 x_{a-1}`): the motivic
`h_1`-tower needs it up to filtration `N`, far above what a save holds, and nothing between
stem 0 and the vanishing line is read. Through classical stem 120 the backbone is 2.1e6 terms;
the part of the stem-400 snapshot through classical stem 207 is 155 MB on disk (the snapshot's
104 GB is nearly all in stems above 300).

**Products.** `h_0` is the tower structure and `h_1, ..., h_4` come with every complex. For
other classes, `products` in the example finds algebra generators weight by weight (a tower
generator is one unless it is a sum of products already computed and multiples of `h_0`) and
multiplies by each through its bar cocycle. Through stem 70 it finds 27 generators of weight at
most 16 and agrees with the `h_j` from the differential (5625 products) and with commutativity
(351 pairs). It is not at scale: `bar_cup` still matches operations one cochain at a time, and
at stem 200 it had done 5 generators in ten minutes when it was stopped. The fix is the one
that worked for the coboundary: a bar word is a composite of the maps `c ↦ d^*_{(R)} c`, one
per letter, and each is a table-driven block map.

## 8. Reference: a native resolution over A_C/τ

`ctau_compare.py` compares an export with the generators of a minimal resolution of `F_2` over
`A_C/τ`: ranks by tridegree (the weight of a generator is read off its differential, which is
homogeneous), and the ranks of `h_0, ..., h_3` out of each tridegree (the coefficient of
`Sq^{2^j}`). The resolution is Nassau's algorithm on upstream master (`construct_nassau("Ctau")`,
which works since "Run Nassau's algorithm over any Milnor algebra"; the tree of #288 itself
still refuses a non-classical algebra in `construct_nassau`), built with
`concurrent,logging,nassau` and run with `RUST_LOG=info EXT_NASSAU_NO_SAVE_QI=1` and no save
directory. Bounded by stem.

| box | tridegrees | classes | `h_0, h_1, h_2, h_3` total rank | mismatches | reference cost |
|---|---|---|---|---|---|
| stem ≤ 100, filtration ≤ 50 | 9157 | 19317 | 1380, 16529, 1066, 451 | 0 | 236 s, 1.5 GB (Nassau, 16 threads) |
| stem ≤ 140, filtration ≤ 70 | 25661 | 82967 | 4285, 73820, 3709, 1648 | 0 | 12889 s, 34 GB (Nassau, 16 threads, while the exports ran) |

The classical algorithm on the #288 tree gives the same ranks through stem 100 (3865 s, 39.5
GB). The export of the same box takes under a second.

## 9. Measured costs

Sparse export of `Cτ`, `C(2)`, `C(η)`, `C(ν)`, `C(σ)` with `h_1..h_4` and both cell maps, on
`sfx7` at `nice 19` (24 to 28 threads, a shared node, other jobs of mine running in some rows):

| stem | `Cτ` cochains | coboundary entries | entries added | reduced kept | `Cτ` reduction | all five complexes | peak memory | dense prototype, `Cτ` only |
|---|---|---|---|---|---|---|---|---|
| 120 | 2.6e5 | 1.9e6 | 3.8e6 | 1.4e6 | < 0.1 s | 2 s | 0.2 GB | 7 s, 2 GB |
| 160 | 1.1e6 | 2.2e7 | 1.0e8 | 1.2e7 | 0.3 s | 13 s | 1.1 GB | 105 s, 22 GB |
| 200 | 3.5e6 | 2.0e8 | 1.9e9 | 8.3e7 | 1.9 s | 70 s | 5.4 GB | 2241 s, 207 GB |
| 240 | 9.8e6 | 1.4e9 | 2.4e10 | 4.6e8 | 24 s | 745 s | 26 GB | |
| 280 | 2.4e7 | 7.7e9 | 2.2e11 | 2.1e9 | 153 s | 5630 s | 96 GB | |
| 320 | 5.6e7 | 4.1e10 | 1.9e12 | 8.9e9 | 948 s | `Cτ` with `h_j`: 1930 s; killed in `C(2)` | 140 GB for `Cτ` | |
| 350 | 9.9e7 | 1.3e11 | 8.6e12 | 2.5e10 | 4474 s | killed in the `h_j` of `Cτ` | 210 GB after the reduction | |

The runs at 320 and 350 were killed by the kernel at 344 GB: sessions on `sfx7` here live in the
memory cgroup of a Slurm job (`nassau_coord_cpu`, 400 GB, shared with the production
coordinator), not in the node's 1.5 TB. Three runs hit that limit before I saw it (stem 350 for
`Cτ`, stem 320 in `C(2)`, stem 350 again); twice the kernel also killed one other small
process in the cgroup. The job itself is still running. Nothing above about 150 GB should be run
from such a session again; it belongs in its own batch job.

At stem 350 the operation tables hold 6.0e9 entries and take 1000 s and 134 GB to build (the
coactions they are filtered from are the peak); at stem 280, 7.6e8 entries and 110 s. The
products by the 27 algebra generators of weight at most 16 take 35 s through stem 120 and 1177 s
through stem 160, on 6 threads.

The density of the coboundaries is 0.4% to 0.7% throughout (7.5, 20, 58, 140, 315, 1330 entries
per cochain where the largest space has 1.7e3, 4.9e3, 1.3e4, 3.0e4, 6.1e4, 1.8e5), so they are
sparse in proportion but not in absolute terms: entries grow like the square of the dimension.
Elimination adds 2, 5, 9, 17, 29, 65 times the input and keeps 0.7, 0.5, 0.4, 0.34, 0.28, 0.19
of it: there is no fill-in, and the cost is chains of short additions. Per 40 stems, the
entries grow about 5 times, the additions 8 to 9 times and the reduced columns 4 to 5 times.

**Stem 350 and 400 by columns.** Not run with the lazy driver: at the measured rates `Cτ`
alone is about 6 hours at stem 350 and the four cofibers 15 times that, and stem 400 is about
12 times stem 350. Section 11 replaces the reduction, and these estimates with it.

**Bounded memory (`src/ctau_lazy.rs`, the spill directory prompt of `ctau_scale`).** The
in-memory driver keeps every reduced column, the whole backbone and the coactions the tables
are filtered from. The lazy driver goes filtration by filtration instead:

* the differentials of one filtration of the backbone are read, used and dropped (two when a
  cell is shifted in filtration);
* each weight's coboundary out of that filtration is reduced, the maps into the cocycles it
  yields (`h_j`, bottom cell, top cell) are computed while they are in memory, and they are
  written to one append-only spill file as byte-coded gaps and dropped. A later map reads back
  only the tower generators of its source, and the top cell reads back the sphere's cocycles of
  one filtration and weight;
* between filtrations only one bit per cochain stays (the pivots, which the next filtration
  skips);
* the coaction on a monomial is computed, filtered and dropped (no memo), and a table entry is
  two 16-bit positions.

Same machine, all five complexes with `h_1..h_4` and both cell maps, 24 threads:

| stem | peak, in memory | peak, lazy | wall, in memory | wall, lazy | weights in flight | spill written | spill read |
|---|---|---|---|---|---|---|---|
| 200 | 5.4 GB | 0.9 GB | 70 s | 159 s | 24 | 0.2 GB | 0.5 GB |
| 240 | 26 GB | 3.6 GB | 745 s | 1457 s | 24 | 0.9 GB | 2.5 GB |
| 280 | 96 GB | 12.5 GB | 5630 s | 12244 s | 12 | 4.0 GB | 11.2 GB |

The towers at stem 200 are those of the in-memory driver, and at stem 70 the ranks of every
complex, the exactness of the cofiber sequences and the comparison with the native reference
all hold as before. Disk traffic is small because only reduced columns are spilled, they are
0.2 to 0.3 of the input, and gaps take little more than a byte each: 2.5 GB for the 2.1e9
reduced entries of `Cτ` at stem 280.

What is still held throughout: the operation tables (3 GB at stem 280, 24 GB expected at 350,
about 120 GB at 400), which every filtration reads in every degree, so dropping a degree means
rebuilding it for each filtration or spilling the unreduced coboundaries instead (hundreds of
GB at stem 350); and, transiently, the whole unreduced coboundary of the weights in flight. The
lazy driver is about twice as slow, because the weights of one filtration are few and each
reduction is sequential: it used 4 to 8 of 24 cores.

**The reduction by columns in parallel.** A profile of the lazy driver at stem 280 has 97% of
its samples in the reduction of one column against the columns before it. Two ways of
spreading that over threads were built. Both give the result of taking the sources one by
one, bit for bit (a column is always reduced at its leading entry by the one earlier column
with that pivot, so only the order in which the additions happen changes):

* batches: 4096 sources are reduced in parallel against the columns before the batch, then
  in order against the batch itself. The profile above is of this one: nearly every addition
  is by a column of the same batch, so the parallel step does 3% of the work;
* halves (`Elimination::solve`, in use): solve the first half of the sources, reduce the
  second half against it in parallel, solve the second half, down to leaves of 16 sources.
  Additions by a column of the same leaf are all that is left in order: 18%, 16% and 13% of
  them for `Cτ` at stems 200, 240, 280, and 4% to 6% for `Cτ ∧ C(η)`.

Wall time of the lazy driver with `h_1..h_4`, 24 threads, 12 weights in flight (the node was
shared with other jobs of mine, more heavily in the later columns):

| stem | complex | one column at a time | batches | halves |
|---|---|---|---|---|
| 200 | `Cτ` | 11.6 s | 8.9 s | 9.6 s |
| 200 | `Cτ ∧ C(η)` | 20.7 s | 17.6 s | 15.9 s |
| 240 | `Cτ` | 100 s | 70 s | 58 s |
| 240 | `Cτ ∧ C(η)` | 209 s | 154 s | 114 s |
| 280 | `Cτ` | 717 s | 576 s | 405 s |
| 280 | `Cτ ∧ C(η)` | 1686 s | 1455 s | not run |

So 1.8 times at stem 280, not the 6 that the idle cores promised. The clocks on the phases
say why: at stem 280 the tasks of `Cτ` spend 439 s building coboundaries, 423 s reducing them
(90 s of it in order) and 1107 s on the maps `h_j` (368 s building them, the rest taking
classes, which is one more reduction per tower generator and map). With the reduction
parallel, the classes are the cost, and they are sequential per class and long. This line was
not pushed further, because the transposed reduction (section 11) removes the additions
themselves.

## 10. Results and where they are

Under `/rs/rs_grp_csht/resolutions/ctau_e2/`:

| archive | contents |
|---|---|
| `towers_n240.tar.zst` | stem ≤ 240: towers, `h_1..h_4`, cell maps for `Cτ`, `C(2)`, `C(η)`, `C(ν)`, `C(σ)` |
| `towers_n280.tar.zst` | the same through stem 280 |
| `ctau_n320.tar.zst` | stem ≤ 320: towers and `h_1..h_4` of `Cτ`, and the towers of `C(2)` without its maps (the run was killed there) |
| `ctau_n350.tar.zst` | stem ≤ 350: the towers of `Cτ` only (the run was killed in `h_j`) |
| `products_n120.tar.zst`, `products_n160.tar.zst` | the five complexes through stem 120 and 160, with `Ctau_generators.csv` and `Ctau_products.csv` (weight ≤ 16) |

Through stem 350, `Cτ` has 8540409 towers (5272324 through stem 320, 2605517 through 280).

Each archive is one `tar | zstd` file; nothing is unpacked on panfs. The scratch copies under
`/dev/shm` are not kept.

From the transposed export (section 11), whose bases are dual to the towers and so are not
those of the archives above, tower for tower as they may correspond:

| archive | contents |
|---|---|
| `transposed_n320.tar.zst` | stem ≤ 320: towers and `h_1..h_4` of `Cτ` and `Cτ ∧ C(η)`, and both cell maps |
| `transposed_n350.tar.zst` | the same through stem 350 |
| `transposed_c2_n350.tar.zst` | stem ≤ 350: `Cτ ∧ C(2)` as `Q/q_0` (files named `Ctau_*`; every tower has length one), with `h_1..h_4` |
| `quotient4_n798.tar.zst` | stem ≤ 798: `Cτ/I_4`, with `h_1..h_4` and multiplication by `q_4`, by columns |

Through stem 350, `Cτ ∧ C(η)` has 476050 towers (289276 through stem 320) and `Cτ ∧ C(2)` has
16949321 classes.

## 11. The transposed reduction

### The measurement

The towers are the pairs of the persistence reduction (section 7), and the pairs do not depend
on how the matrix is reduced: by columns with the pivot at the last row, or by rows with the
pivot at the first column, taking the rows from the last to the first. The work does. In the
second form the pivots of one coboundary are the rows that the coboundary one filtration
*down* need not reduce, so the filtrations are taken from the top. `examples/ctau_orders.rs`
runs both on the same matrices, one column or row at a time, and counts the entries of what
is added:

| stem | complex | by columns: entries added | transposed: entries added | ratio | rows reduced as they stand |
|---|---|---|---|---|---|
| 160 | `Cτ` | 1.02e8 | 6.0e6 | 17 | 86% |
| 200 | `Cτ` | 1.90e9 | 5.3e7 | 36 | 84% |
| 240 | `Cτ` | 2.38e10 | 3.4e8 | 71 | 82% |
| 280 | `Cτ` | 2.21e11 | 2.1e9 | 105 | 81% |
| 200 | `Cτ ∧ C(η)` | 5.29e9 | 1.17e8 | 45 | 90% |
| 240 | `Cτ ∧ C(η)` | 6.28e10 | 7.7e8 | 81 | 89% |
| 280 | `Cτ ∧ C(η)` | 5.84e11 | 4.9e9 | 119 | 88% |

By columns the additions are 9, 17 and 29 times the entries built at stems 200, 240, 280 and
the factor grows; transposed they are 0.6, 0.7, 1.0, 1.4 and 1.7 times the entries of the rows
reduced at stems 200 to 350. The pairs agree in number at every stem, and as multisets of
(stem, filtration, weight, length) through stem 280 (2605517 towers).

The transposition does not make the matrix smaller. Building it is reading `P(R) · m ∋ m'`
off the tables for every term `P(R) x` of a differential, and the entries built that way
repeat: 3.2, 4.0 and 4.9 are built for each one left at stems 200, 240, 280, the rest
cancelling in pairs, by columns and by rows alike. What changes is that building is nearly all
that is left to do.

This is the choice between the homology and the cohomology algorithm of persistent homology,
where the difference is as large and as hard to predict; for these complexes the columns were
the wrong side. The same example compares ten orders of the cochains within a level for the
reduction by columns (by generator degree, by the exponent of `q_1`, by the exponents of
`q_1, q_2, ...`, each way and in either precedence): at stem 160 they add between 1.01e8 and
4.8e8 entries, and the order in use (section 7) is within 1% of the best. No order within a
level changes the growth; the transposition does.

### What it computes

Write `M_a: C^a → C^{a+1}` for the coboundary at one weight, with the cochains numbered by
level as before. Row `τ` of `M_a` is the functional `e_τ^* ∘ δ` on `C^a`. The reduction keeps,
for each cochain `τ` of `C^{a+1}` that is not skipped,

    R_τ = ψ_τ ∘ δ,      ψ_τ = e_τ^* + (coordinates of cochains after τ),

with the first cochain `σ` of `R_τ` its pivot, distinct for distinct `τ`. Then:

* `(σ, τ)` is a pair: a tower in `C^{a+1}` born at the level of `τ` that dies at the level of
  `σ`, a class in the polynomial degrees in between. A row that reduces to zero is a class
  that never dies (in the lowest filtration, or with a bound on the Chow degree);
* `ψ_τ` is the tower as a **functional**. `R_τ` starts at `σ`, so it vanishes on the cochains
  of level below that of `σ`: for `k` from the level of `τ` up to there, `ψ_τ` vanishes on the
  coboundaries of `Q_k` and is a functional on `H^{a+1}(Q_k)`. The towers alive at `k` are a
  basis of the dual of `H^{a+1}(Q_k)`;
* the cochains `σ` are skipped as rows of `M_{a-1}`: every cochain of `C^a` is either the
  pivot of a row out of `C^a` or starts a functional `ψ` on `C^a`, so the two families
  together are a basis of the dual of `C^a`, triangular for the order of the cochains.

No cocycle is produced. What replaces "the class of a cocycle" is "the value of the tower
functionals on it", and what replaces a cochain map `F` applied to a generator is `F`
transposed applied to a tower: for a tower `τ'` of the target, `y = ψ_{τ'} ∘ F` is written in
the triangular basis of the source space,

    y = Σ α R + Σ β_τ ψ_τ,

and `β_τ` is the coefficient of the tower `τ'` in `F` of the tower `τ`, in the bases of
cohomology dual to the towers. Two facts make this cheap and correct:

* *Only the levels where `τ'` lives matter.* The coefficients on cochains of level at most
  `k` depend on `y` restricted to those cochains, and nothing is wanted beyond the level at
  which `τ'` dies, so `y` and every row added to it are cut there. Most towers have length
  one, so this is one level of a space that has many. At stem 200, `h_1` on `Cτ` costs 7.3e8
  added entries without the cut and 2.2e6 with it.
* *No dead functional appears.* Below the level at which `τ'` dies, `y` is a functional on
  cohomology: `y ∘ δ = R_{τ'} ∘ F`, which starts at or after the level of the pivot of
  `R_{τ'}`. The functionals of towers that have died by level `k` are independent on the
  coboundaries of `Q_k`, and those alive vanish there, so `β_τ = 0` for every `τ` that dies
  before `τ'` does. The export checks this on every coefficient it computes
  (`Step::finish`), which is a test of the whole construction on every product.

The two steps of writing `y` are done a filtration apart. The rows out of `C^a` are in memory
when `M_a` has been reduced, and that is when `y` is reduced by them, which leaves a few
cochains at which functionals of `C^a` start (`Step::pull`); the functionals themselves are
found when `M_{a-1}` is reduced, and finish the job (`Step::finish`). So the reduced rows,
which are the bulk (as large as the matrix), are never stored: they live while their weight
is in flight. The functionals are nearly coordinates: 1.4 to 2 cochains each on average.

`h_{j+1}` is `F = d^*_{(2^j)}` from `C^a` at weight `u` to `C^{a+1}` at weight `u + 2^j`,
built by rows (`pullback_rows`: the differential of the generator of a block of targets is
walked once) and only at the rows the towers use. Weights are taken from the highest down
within a filtration, so that the towers a product lands on are there.

The cell maps of a two-cell complex are maps within a filtration, which need the rows out of
their source and the towers of their target; those are found a step apart in two exports. So
the top cell is pulled back while the cofiber is reduced, from the towers of the sphere
(exported first), and the bottom cell while the sphere is reduced a second time, from the
towers of the cofiber. The second reduction of the sphere costs a sixth to a third of the
cofiber.

### What it does not give

* **Cocycles.** The basis of a tridegree is the one dual to the towers, not a list of
  cocycles, and it is not the basis of the reduction by columns (the two are related by a
  triangular matrix; the towers correspond). Everything linear is unaffected: ranks, `h_0`
  (the tower structure), `h_1..h_4`, cell maps, and their composites are matrices in these
  bases. What needs a cocycle is the product by a general class through its bar cocycle
  (section 3), and anything downstream that lifts a class to cochains. A cocycle dual to one
  tower is one triangular solve against the rows and functionals of its space; that is not
  written, and it is the price of using this export for general products.
* **The cell maps of `Cτ ∧ C(2)`.** The export takes cells that are not shifted in filtration
  or polynomial degree, which is the Hopf cofibers; `C(2)` as the cone of `q_0` is shifted in
  both. As the quotient `Q/q_0` it is exported, and cheaply (section 12), with `h_1..h_4` and
  without the maps to and from `Cτ`: reduction modulo `q_0` and the Bockstein.

### Validation

`ctau_rank_compare.py` compares two exports up to a change of basis: every rank, and the rank
of every `h_j`, of every composite `h_i h_j` and of both cell maps out of every tridegree.

| range | complexes | result |
|---|---|---|
| stem ≤ 100 | `Cτ`, `C(η)`, `C(ν)`, `C(σ)` against the reduction by columns | 91105 classes; 43473 `h_j`, 45111 composites, 36358 cell maps: no rank differs |
| stem ≤ 160 | `Cτ`, `C(η)` against the reduction by columns | 176776 classes; 68852 `h_j`, 79451 composites, 18173 cell maps: no rank differs |
| stem ≤ 140, filtration ≤ 70 | `Cτ` against the native resolution (section 8) | 82967 classes, `h_0, ..., h_3` of total rank 4285, 73820, 3709, 1648: no mismatch |
| stem ≤ 200, Chow degree ≤ 8 | `Cτ`, `C(η)` against the reduction by columns | 67607 classes, with `h_j`, composites and cell maps: no rank differs |
| stem ≤ 120, quotient by `q_0, q_1` | `Cτ`, `C(η)` against the reduction by columns | 13986 classes: no rank differs |
| stem ≤ 280 | `Cτ` against the reduction by columns | the 2605517 towers agree as a multiset |

### Measured

The export with `h_1..h_4` and the cell maps, 24 threads, 12 weights in flight through stem
280, 8 at 320 and 6 at 350, on a node that other jobs of mine loaded to between one and two
times its cores, so that the same run varies by half from one time to the next. "Built"
counts entries before the repeated ones cancel, for the coboundary and for the maps; the rows
reduced are counted after.

| stem | complex | built | rows reduced, their entries | entries added | by columns (section 9) | transposed | peak of the run |
|---|---|---|---|---|---|---|---|
| 200 | `Cτ` | 2.8e8, 4.8e6 | 1.7e6, 8.8e7 | 5.3e7 | 9.6 s | 2.3 s | 0.5 GB |
| 200 | `Cτ ∧ C(η)`, top cell | 7.4e8, 1.3e7 | 3.3e6, 2.4e8 | 1.2e8 | 15.9 s | 2.9 s | |
| 200 | the bottom cell | 2.8e8 | | | | 1.6 s | |
| 240 | `Cτ` | 1.9e9, 2.0e7 | 4.8e6, 4.7e8 | 3.4e8 | 58 s | 7.2 s | 1.7 GB |
| 240 | `Cτ ∧ C(η)`, top cell | 5.0e9, 5.3e7 | 9.4e6, 1.3e9 | 7.7e8 | 114 s | 13.6 s | |
| 240 | the bottom cell | 1.9e9 | | | | 5.1 s | |
| 280 | `Cτ` | 1.0e10, 1.3e8 | 1.2e7, 2.1e9 | 2.1e9 | 405 s | 42 s | 6.2 GB |
| 280 | `Cτ ∧ C(η)`, top cell | 2.8e10, 3.8e8 | 2.4e7, 6.0e9 | 4.9e9 | 1455 s | 105 s | |
| 280 | the bottom cell | 1.0e10 | | | | 33 s | |
| 320 | `Cτ` | 5.5e10, 7.1e8 | 2.8e7, 8.7e9 | 1.2e10 | 1930 s in memory, 140 GB | 180 s | 22 GB |
| 320 | `Cτ ∧ C(η)`, top cell | 1.5e11, 2.1e9 | 5.4e7, 2.5e10 | 2.7e10 | never finished | 428 s | |
| 320 | the bottom cell | 5.5e10 | | | | 176 s | |
| 350 | `Cτ` | 1.7e11, 1.3e11 | 4.9e7, 2.4e10 | 4.0e10 | 4474 s for the reduction alone, 210 GB | 878 s (earlier builder) | 52 GB |
| 350 | `Cτ ∧ C(η)`, top cell | 4.8e11, 2.9e11 | 9.6e7, 6.9e10 | 8.9e10 | not attempted | 2413 s | |
| 350 | the bottom cell | 1.7e11 | | | | 459 s | |

The operation tables are extra and shared: 9 s, 36 s, 3 min, 12 min and 26 min at stems 200
to 350.

The rows at stem 350 were run before three changes to the builder, which the other rows
have. At stem 320 they took `Cτ` from 460 s to 180 s, `Cτ ∧ C(η)` from 944 s to 428 s and the
peak from 55 GB to 22 GB (the last by shrinking each row to its length):

* a row of a map is read only as deep as the towers that use it live (the sources of a row
  are listed in order of level): the entries built for the `h_j` of `Cτ` at stem 280 go from
  6.8e9 to 1.3e8;
* the tables are sorted by row, and a list much longer than the rows wanted is searched for
  each of them;
* the terms of a differential are resolved once per filtration and shift to the operation's
  number in the tables, or dropped if it never acts; before, every visit of a term (one per
  weight) looked it up by its exponents.

So the times at stem 350 are high by a factor of two to three for what the code now does.

At stem 280 the tasks of `Cτ` now spend 314 s building rows, 21 s reducing them and 90 s on
the `h_j`: the elimination is 5% of the work, and what is left is reading the coboundary off
the tables. The additions grow faster than the rows (about `N^13` against `N^11`), but an
added entry costs a thirtieth of an entry built, so they do not matter before stem 600.

At stem 320 the whole export, `Cτ` and `Cτ ∧ C(η)` with `h_1..h_4` and both cell maps, takes
25 minutes, 12 of them the tables, and 22 GB; by columns `Cτ` alone took 32 minutes and 140 GB
in memory, and `Cτ ∧ C(η)` had not been computed at this stem. At stem 350, with the earlier
builder, `Cτ` with `h_1..h_4` took 15 minutes after the tables, where the reduction by columns
took 75 minutes and was killed at 210 GB before the `h_j`; `Cτ ∧ C(η)` with its top cell took
40 minutes and the bottom cell 8, and the whole run 90 minutes and 52 GB. Through stem 350,
`Cτ ∧ C(η)` has 476050 towers against the 8540409 of `Cτ`.

## 12. Bounded Chow degree, and the quotients of `Q`

### Bounded Chow degree

With a bound `K` on the polynomial degree the monomials longer than `K` do not exist and the
export is of `⊕_{k ≤ K} Ext_A(F_2, Q_k)`, exactly (section 4). The complex is no longer
acyclic, so a cocycle need not be a coboundary: by columns the others are found one filtration
later, as the sources whose coboundary reduces to zero, and transposed as the rows that reduce
to zero. A tower still standing at the bound is exported with length `K + 1 - k+`.

Differentials: the Adams `d_r` of `Cτ` (the algebraic Novikov `d_r`) raises the Novikov
filtration by one and the Chow degree by `r - 1`. So the part of Chow degree at most `K` is a
quotient of the spectral sequence, in which a class of Chow degree `k` has its `d_r` for
`r ≤ K - k + 1`; it is the algebraic Novikov spectral sequence of `BP_*/I^{K+1}`. It contains
every class of Adams filtration at most `K`.

`Cτ` and `Cτ ∧ C(η)` with `h_1..h_4` and cell maps, by columns, 24 threads, on the copy of
the backbone through classical stem 207 (wall time includes reading it); for each bound the
largest space, the coboundary entries of `Cτ`, and the time for both complexes:

| stem | `K = 4` | `K = 8` | `K = 12` | all `k`, `Cτ` |
|---|---|---|---|---|
| 200 | 154, 1.2e6, 1.4 s | 806, 1.4e7, 2.4 s | 1934, 4.4e7, 4.4 s | 12908, 2.0e8 |
| 280 | 312, 1.2e7, 7.3 s | 1677, 1.9e8, 15 s | 4732, 8.0e8, 41 s | 61006, 7.7e9 |
| 360 | 542, 7.9e7, 37 s | 3683, 1.8e9, 107 s | 10691, 9.6e9, 422 s | about 2.1e5, 1.9e11 |
| 400 | 643, 1.6e8, 81 s | 4739, 4.2e9, 243 s | 14906, 2.5e10, 1330 s | about 3.6e5, 6e11 |

At stem 400, `Cτ` has 157797, 542374 and 1206295 towers within the three bounds, and the peak
memory is 5 GB, 5 GB and 15.6 GB. The spaces are small and grow slowly (like `N^2` to `N^3`),
but the entries grow like `N^7` to `N^8`, `N^8.3` and `N^9.2`: a bound makes the computation
at stem 400 smaller by 4000, 140 and 24 times and lowers the power of the growth by two to
four, not more. The coboundary gets denser at a fixed bound too (13, 151 and 767 entries per
cochain for `K = 4` at stems 200, 400, 600), because the differentials of the backbone get
longer whatever the monomials are.

How much of the page a bound holds, from the towers through stem 280 (the share of classes of
Chow degree at most `K` among those of the stems in the band):

| stems | `Cτ`: `K = 4` | `8` | `12` | `20` | `30` | `50` | `Cτ ∧ C(η)`: `K = 4` | `8` | `12` | `20` | `30` | `50` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 to 100 | 15% | 36% | 56% | 82% | 96% | 100% | 26% | 45% | 60% | 80% | 94% | 100% |
| 101 to 200 | 3.3% | 11% | 22% | 47% | 71% | 93% | 14% | 27% | 39% | 57% | 74% | 92% |
| 201 to 280 | 1.5% | 5.1% | 11% | 28% | 51% | 81% | 10% | 23% | 34% | 50% | 65% | 84% |

The transposed export takes a bound as well (validated, section 11) but gains less there: a
tower standing at the bound never dies, so its products are computed on every level, and with
few monomials the time goes to walking the differentials rather than to the entries. At stem
400 with `K = 8`, `Cτ` takes 178 s transposed on 8 threads against 78 s by columns on 24:
about the same work. What it gains is memory on the whole backbone (below).

**The whole backbone.** Beyond stem 414 the backbone is the whole snapshot, 104 GB on local
disk, read one filtration at a time with the terms filtered as they are read
(`BackboneSource::slice`). The filter needs the operations that act on some monomial held,
and listing the operations of the backbone first, as the unbounded export does, is not
possible there (the differentials are too large to hold unfiltered). So the tables are built
from the monomials alone, as for the quotients: with `K = 4` and `8` at stem 400 that is
3.3e5 and 2.4e7 entries where the tables filtered by the backbone have 2.4e5 and 1.6e7.
A bound selects that.

`K = 4` on the whole backbone, `Cτ` with `h_1..h_4`:

| stem | export | towers | largest space | entries built | time | peak |
|---|---|---|---|---|---|---|
| 600 | by columns, 12 threads | 809498 | 1981 | 4.4e9 | 22 min | 17.7 GB |
| 798 | by columns, 16 threads | | | | stopped after 20 min | 67 GB and growing |
| 798 | transposed, 24 threads | 2464904 through Novikov filtration 30 | | | stopped after 24 min | 57 GB and growing |

Stem 798 at `K = 4` was not finished. By columns the terms of the differentials are filed by
the generator they land on, two filtrations at a time, and that alone held 67 GB after 20
minutes. The transposed export got through the Novikov filtrations from 399 down to 30 in 24
minutes, and was stopped at 57 GB, growing by 3 GB a minute, with the 30 filtrations that
hold four fifths of the differentials still to come: it would have passed the 100 GB allowed
to a run from a session here. What it holds is the terms of one filtration that act on some
monomial, once for each of the five shifts of the coboundary and the `h_j`, 12 bytes each; at
stem 798 that is billions of terms even for monomials of length four. It is a batch job of
about 250 GB and a few hours as it stands, and a leaner one if the terms are held once and
the shifts read off them.

Getting that far took four changes, all in the reading of the backbone, which was the whole
of the cost there:

* the terms of a differential are resolved to the operation's number in the tables once per
  filtration, or dropped, in place of a lookup by exponents on every visit (section 11);
* a term is tested by the degree and index of its operation, as the save has them, before
  it is unpacked into exponents: with few monomials nearly none acts (`OpFilter`). Unpacking
  and looking up every term was 82% of the run;
* the differentials are read as the save holds them, by generator, and resolved by the
  thread that read each degree, so that a filtration is held once
  (`BackboneSource::resolved`);
* the memo of unpacked operations is read without its lock.

The first pass of these took `K = 4` through stem 400 from 59 s to 32 s on 8 threads, and the
filtrations from 399 to 37 at stem 798 from 106 minutes to 17.

### The quotients `Q/(q_0, ..., q_{n-1})`

`q_n` is invariant modulo `q_0, ..., q_{n-1}`, so the quotient `Q/I_n` is a comodule algebra
and `Ext_A(F_2, Q/I_n)` is the E2 page of `Cτ` smashed with the algebraic `S/(2, v_1, ...,
v_{n-1})`: `n = 1` is `Cτ ∧ C(2)`. In a quotient no cochain is shorter than its source, every
tower has length one, and multiplication by `q_n` is a map of exports (`Ctau_q.csv`).
`ctau_quotient_check.py` checks consecutive quotients against each other through the long
exact sequence of `q_n`.

`Cτ/I_n` and `Cτ/I_n ∧ C(η)`, by columns, tables from the monomials:

| stem | `n = 2`: classes, time | `n = 3` | `n = 4` |
|---|---|---|---|
| 200 | 76616, 1.0 s | 15749, 0.9 s | 5249, 0.9 s |
| 280 | 373144, 3.8 s | 65376, 2.8 s | 18884, 2.4 s |
| 360 | 1394169, 14 s | 212386, 8.3 s | 54821, 6.6 s |
| 400 | 2508868, 28 s | 356054, 16 s | 86861, 13 s |
| 500 (whole backbone) | | | 254760, 45 s |
| 600 | | | 691587, 249 s |
| 798 | | | 3682138, 81 min |

The last row is the first of these computations at the end of the range: the E2 page of
`Cτ/I_4` through motivic stem 798, on the whole stem-400 backbone, in 81 minutes and 16 GB.
Its coboundaries have 1.4e7 entries in all; the time is the pass over the 104 GB of
differentials, of which nearly every term is read and dropped.

The coboundaries of `Q/I_2` have 7 entries per cochain at stem 400 where those of `Q` have a
thousand: the density of the full complex is in `q_0` and `q_1`.

**`Cτ ∧ C(2)` is `n = 1`.** As the cone of `q_0` it was the most expensive of the five
complexes (3191 s at stem 280 by columns, twice the cochains of `Cτ`, never finished at stem
320). As `Q/q_0`, transposed, it is the cheapest by far: the coboundary of `Q/q_0` is the part
of the coboundary of `Q` that keeps the length of the monomial, a twentieth of it and less.
With `h_1..h_4`, 12 threads, tables included:

| stem | classes | entries built | time | peak |
|---|---|---|---|---|
| 200 | 949306 | 1.5e7 | 3.3 s | 0.1 GB |
| 240 | 2336379 | 7.9e7 | 6.5 s | 0.3 GB |
| 280 | 5162929 | 3.5e8 | 35 s | 0.7 GB |
| 320 | 10457233 | 1.5e9 | 75 s | 1.5 GB |
| 350 | 16949321 | 4.2e9 | 169 s | 2.7 GB |

The classes at stems 200, 240, 280 are the towers the cone had there, and the ranks of the
two agree at every tridegree through stem 120 (27039 tridegrees, 77515 classes). The entries
grow like `N^11` as for `Cτ`, from a twenty-fifth of the size: the same model puts stem 400
at ten minutes and stem 600 at a day on one node.

## 13. Growth, cost models, and stem 800

### How things grow

Fitted on the ladders of sections 9 and 11 (stems 200 to 350), as powers of the stem `N`.
The powers are not constant: these are counts of monomials and of Milnor basis elements, and
every one of the powers below rises slowly with the stem, so the last column is an order of
magnitude at best.

| quantity | growth, 200 to 350 | stem 280 | stem 350 | stem 400 | stem 800 |
|---|---|---|---|---|---|
| towers of `Cτ` | `N^5.3` | 2.6e6 | 8.5e6 | 1.7e7 | 7e8 |
| cochains of `Cτ` | `N^6` | 2.4e7 | 9.9e7 | 2e8 | 1.3e10 |
| largest cochain space (from the ranks, `ctau_plan.py`) | | 6.1e4 | 1.8e5 | 3.6e5 | 2.5e7 |
| coboundary entries built, by columns | `N^11.6` | 7.7e9 | 1.3e11 | 6e11 | 1.5e15 |
| entries added, by columns | `N^15` | 2.2e11 | 8.6e12 | 5e13 | 1.5e18 |
| coboundary entries built, transposed | `N^10.4`, `11.1`, `12.5`, `12.8` from rung to rung | 1.0e10 | 1.7e11 | 1e12 | 1e15 to 1e16 |
| entries of the rows reduced, after cancelling | `N^9.2` rising to `N^11.2` | 2.1e9 | 2.4e10 | 1e11 | 2e14 to 1e15 |
| entries added, transposed | `N^13` | 2.1e9 | 4.0e10 | 2e11 | 2e15 |
| operation tables | `N^9` | 7.6e8 | 6.0e9 | 2e10 | 1e13 |

"Eight times per 40 stems" was the reduction by columns at stems 240 to 350; as a power it is
the 15th. The output is the smallest line of the table by far: a page of 7e8 towers at stem
800 is a file of tens of GB. What stands between is the density of the coboundary in the
basis of monomials and generators (a thousand entries per cochain at stem 350, growing as
fast as the cochains do), and that most of what is built cancels or is never read; not the
size of the answer.

### Cost model: the complete page, transposed

The time is the entries built: 3.3e-9 s each on one node of 24 threads at stem 320 (180 s for
5.5e10), with the node shared. `Cτ ∧ C(η)` with its top cell takes 2.4 times `Cτ` and the
bottom cell as much as `Cτ` again, so the three together 4.4 times. With the powers of the
table, the lower figure for the 11th and the higher for the 13th:

| stem | `Cτ` | `Cτ`, `Cτ ∧ C(η)` and both cell maps | tables | memory |
|---|---|---|---|---|
| 280 (measured) | 42 s | 3 min | 7.6e8 entries, 3 min | 6 GB |
| 320 (measured) | 3 min | 13 min | 2.6e9 entries, 12 min | 22 GB |
| 350 | 8 to 10 min (15 measured, earlier builder) | 40 min | 6.0e9 entries, 26 min | 30 to 50 GB |
| 400 | 35 to 55 min | 2.5 to 4 h | 2e10 entries, 1.5 h | tables 80 GB (110 GB while they are built) and 30 to 100 GB in flight: a batch job of 250 GB |
| 500 | 7 to 18 h | 1.5 to 3.5 days | 1.6e11 entries | tables by degree; 100 to 200 GB in flight |
| 600 | 2 to 10 days | 1 to 6 weeks | 8e11 entries | the same, and the largest coboundaries out of core |
| 800 | 2 months to 2 years | 1 to 9 years | 1e13 entries | one coboundary is 3e12 entries: rows built when asked for, not stored |

The lines at 500 and beyond assume the tables are restructured. They are times on one node,
and the weights of a filtration are independent tasks, so a cluster divides them. As the code
stands, stem 500 is a batch job, stem 600 a campaign, and stem 800 out of reach; what would
change that is below.

**The next factor: rows built only as deep as their pair.** The entries of a row are at
cochains of its own level and of every level above, and its pivot is its first entry. A row
that is reduced as it stands is read no further than its pivot, and three pairs in four
have both ends at one level. `ctau_orders` counts the entries by the number of levels between
a row and the source:

| stem | built | at the level of the row | up to the level of its pivot | one, two, ..., five levels up | six and more | within the longest tower of the space |
|---|---|---|---|---|---|---|
| 160 | 2.9e7 | 2.5e6, 8.4% | 2.7e6, 9.2% | 2.1e6, 2.6e6, 1.9e6, 2.7e6, 1.8e6 | 1.6e7 | 8.7e6, 30% |
| 200 | 2.8e8 | 1.5e7, 5.4% | 1.6e7, 5.7% | 1.6e7, 2.0e7, 1.7e7, 2.2e7, 1.7e7 | 1.7e8 | 6.8e7, 24% |
| 240 | 1.9e9 | 7.9e7, 4.2% | 8.1e7, 4.3% | 9.7e7, 1.2e8, 1.1e8, 1.3e8, 1.1e8 | 1.2e9 | 4.2e8, 22% |

So 91% to 96% of what is built lies beyond the pivot of its row, and the share that does not
shrinks as the stem grows (about `N^-1.7`). The level of the row alone is the complex of
`Q/q_0`, which is why `Cτ ∧ C(2)` as that quotient costs so little (section 12): the towers
of `Cτ` are the Bockstein spectral sequence of `q_0` on it, and nearly all of the coboundary
is read to compute differentials that turn out short. An export that builds each row level by
level, as far as its pair and the rows that use it need, would build a twentieth of the
entries at stem 240 and less beyond; one that builds every row of a space as deep as the
longest tower there, which is simpler, a fifth. It needs the tables sorted by the number of
levels dropped (a prefix of each list is then a depth), rows built again one at a time, and a
reduction that knows how far each row is exact. It is the first thing to build next, and it
is not started.

Three things stand between this and stem 800, in order:

1. **Tables.** They are global, `N^9`, and every filtration reads every degree. Past stem
   400 they have to be built per degree when a block needs them and dropped, or replaced by
   the coaction computed per block. The row builder reads them in the same pattern as the
   column builder, so either works for both.
2. **Memory of one coboundary.** The reduced rows of a weight in flight are as large as its
   matrix: 1 GB at stem 350, 5 GB at 400, terabytes at 800. Four rows in five are never
   changed, and a row is needed again only when a later row or a product lands on its pivot:
   a third of the rows are (30% and 34% of them, with 35% and 40% of the entries, at stems
   200 and 240; `ctau_orders` counts them), and the products add a twentieth of what the
   reduction does. So the rows need not be kept: keep the pivot of each and the functionals
   that are not coordinates, and build a row again from the differential of its generator
   when it is asked for, as Ripser does with its apparent pairs. The rows added are about as
   many as the rows, so that costs about one more build of the matrix, and it leaves a few
   words per cochain in memory.
3. **Distribution.** Per weight, the chain of filtrations from the top down is one task with
   a small state between steps (a bit per cochain, and the products pulled back). Stem `N +
   40` continues every chain of stem `N` downward and adds new weights, so nothing is redone
   when the range grows.

### Cost model: bounded Chow degree at stem 800

From the ladder of section 12 and the entries built, `Cτ` and `Cτ ∧ C(η)` together on one
node:

| bound | entries of `Cτ` at stem 400 | growth | at stem 600 | at stem 800 | time at stem 800 | largest space at 800 |
|---|---|---|---|---|---|---|
| `K ≤ 4` | 1.6e8 | `N^8` | 4.4e9 (measured) | 4.5e10 | half a day | 4.6e3 |
| `K ≤ 8` | 4.2e9 | `N^8.3` | 1.2e11 | 1.3e12 | one to two days | 4.1e4 |
| `K ≤ 12` | 2.5e10 | `N^9.2` | 1e12 | 1.5e13 | one to two weeks | 1.6e5 |
| all `k` | 6e11 | `N^11` to `N^13` | 1e14 | 1e15 to 1e16 | node-years | 2.5e7 |

The times follow the entries. The times of the ladder itself grow more slowly (`N^5.8`,
`N^6.7`, `N^8.2`), because reading the backbone dominates its low rungs, and would put the
last column at 1.3 hours, 7 hours and 4.5 days; the one point beyond the ladder, `K = 4` at
stem 600, took 22 minutes for `Cτ` alone on 12 threads where those powers give 14 minutes
for both complexes on 24, so the entries are the ones to trust. Each complex also costs one
pass over the backbone, which took 80 minutes at stem 798 for the quotient by `q_0, ..., q_3`
(the last row of the quotients in section 12 is little else). The row for `K ≤ 4` at stem
800 was tried and not finished (section 12): reading the backbone, not the entries, is what
a small bound costs there, and the memory of one filtration of it, more than 60 GB as the
code stands, is what stopped the run.

### What else was considered

* **Orders within a level** (section 11): ten tried, no gain.
* **The Bockstein tower.** `Ext(Q/I_{n+1})[q_n] ⇒ Ext(Q/I_n)` from the cheap quotients
  down to `Q`. The top of it is cheap (seconds), but the last two steps are where everything
  is: `Ext(Q/I_2)[q_1]` has about as many classes as `Q/q_0` has cochains (`q_1` has degree one), so
  the first page of that step is no smaller than the complex it replaces, and its
  differentials need the cochains of `Q/q_0` anyway. Transferring the differential to
  cohomology at each step (homological perturbation) gives a complex 3 to 10 times smaller
  whose differentials each cost a reduction in the larger one: a constant, not a power, and
  the transposition already took more.
* **The structure of `Q` as a comodule.** `Q_k` is the dual of the free unstable module
  `F(k)` desuspended `k` times (the monomials of length at most `k` in the `ζ_i`), so the
  page is `⊕_k Ext_A(Σ^{-k} F(k), F_2)` and the towers are the spectral sequence of the
  length filtration of the dual Steenrod algebra, which converges to `F_2`: that is the
  acyclicity the reduction uses. Multiplication by `q_n` is linear only modulo `q_0, ...,
  q_{n-1}`. That linearity is not used for `n ≥ 1`: for `Cτ ∧ C(2) = Q/q_0`, all weights are
  one persistence module over `F_2[q_1]` on the cochains of `Q/I_2`, 24 times fewer than
  those of `Q/q_0` at stem 280. That would make `C(2)` cheaper again than the quotient export
  of section 12, which is already cheap enough that it is not the place to work. For `Q`
  itself only `q_0` is invariant, and its towers are what is computed.
* **Change of rings.** `Q/q_0` is a comodule over the double of the Steenrod algebra, and
  the Cartan–Eilenberg spectral sequence over it has `E_2 = Ext_A(F_2, Q ⊗ Q)` in halved
  degrees again, with differentials: a larger coefficient module and a spectral sequence in
  exchange for half the stem. Not pursued.
* **Periodicity and known towers.** Above the lowest filtration the complex is acyclic, and
  both reductions already skip every cochain that is known to be a pivot; four rows in five
  are forced in the transposed form. The `h_1`-periodic region is at high Novikov filtration,
  where the spaces are small. Nothing left to skip is large.
* **`Cτ ∧ C(η)` as the target.** It has 18 times fewer towers than `Cτ` (141763 against
  2605517 at stem 280) and costs 3 times as much to compute, in either reduction: the complex
  is the cone, twice the cochains. With a minimal resolution of the classical Moore module in
  place of the backbone it would be one copy; that resolution is not on `/rs` through stem
  400 and may not be computed here.

## 14. Fitting together with the native route (#288, #289, #290)

The upstream stack resolves over `A_C/τ` natively (#288), lifts the differential to `A_C` so
that it becomes `d + δ` with `δ` divisible by `τ`, runs the τ-Bockstein (deformation) spectral
sequence with `E_1 = Ext_{A_C/τ}` by weight (#289), and takes the cohomology of `δ` with
products (#290). The two routes compute the same `E_1`; section 8 is that comparison.

* **Scale.** The native resolution costs like a classical resolution to the same stem (the
  algebra has the same dimensions): 236 s to stem 100 with Nassau's algorithm, and stem 400
  would be a computation of the size of the classical stem-400 one. The untwisted route reads
  that range off the classical resolution through stem 200 in minutes. So the native route is
  the reference at low stems and the untwisted one the source at high stems.
* **What the untwisted complex is.** `Hom_A(P, Q)` is `Hom_{A_C/τ}(K, F_2)` for the free,
  non-minimal resolution `K = (A_C/τ) ⊗_E Koszul ⊗ P`-style complex whose generators are
  (backbone generator, dual monomial). #289 and #290 are written against a minimal resolution,
  where `E_1` is the generators. To run them here they need two changes: take `E_1` as the
  cohomology of the mod-τ complex with chosen cocycles (the tower generators of this export)
  instead of the generators themselves, and take `δ` as a lift of the differential of `K`.
* **The lift exists in this repository.** Branch `ctau-plan-n400` (`koszul_lift`, `tau_lift`)
  lifts exactly that differential to `A_C` in a Koszul gauge and gets the τ-Bockstein
  differentials at every τ-order from the backbone, as ranks. It needs products in `A_C` in the
  degrees of the operations only, no new resolution. Redone as a cochain map on the spaces of
  `ctau_scale` (it has the shape of `pullback_columns` with a different table), it gives `d_1`
  on tower generators, and the zig-zag of #289 gives the higher `d_r`.
* **Proposal.** Make #289's page-turning and #290's `F_2[τ]`-module and product code generic
  over "a complex over `A_C/τ` with weights, chosen cocycles and a lifted differential", with
  the native resolution as one instance and the Koszul-on-backbone complex as the other; check
  the second against the first through stem 100 to 140; then run the second at scale. I have
  not read #289 and #290 closely enough to say how invasive that is.

## 15. Not done here

* General products at scale (section 7), and products on the cofibers beyond `h_1..h_4`. With
  the transposed export they need one more piece: a cocycle for each algebra generator, or the
  bar construction transposed (section 11).
* The maps between `Cτ` and `Cτ ∧ C(2)` in the transposed export, and `Q/q_0` as one module
  over `F_2[q_1]` (sections 11, 13).
* Operation tables by degree, rows built again when asked for instead of kept, and chains of
  filtrations per weight as separate tasks: what the complete page needs past stem 400 (section
  13).
* The E2 page of the motivic sphere and the maps `S → Cτ → S^{1,-1}` on E2 pages (section 14);
  the Adams differentials of `Cτ`; names. Not started, as agreed.
* Massey products, by request.

## 16. Running it

    cargo build --release --no-default-features --features concurrent,logging,nassau \
        --example ctau_scale --example ctau_e2
    printf 'S_2\n<copy of the save>\n<out>\n240\nC2,Ceta,Cnu,Csigma\n0\n' | target/release/examples/ctau_scale
    python3 -I ctau_legacy.py <out> <out>/legacy 100 50     # per-tridegree format, small ranges
    python3 -I ctau_compare.py <out>/legacy <reference>     # against a native resolution
    python3 -I ctau_towers_check.py <out>                   # products, when asked for
    python3 -I ctau_check.py <dense or legacy export> <stemlit>/out/tables
    python3 -I ctau_plan.py <rank.csv> 798 10

`ctau_scale` asks for the module, its save directory (a copy: it opens the store), the output
directory, the largest motivic stem, the complexes and the largest weight of an algebra generator
to multiply by (0 for none); then the spill directory (`-` for the in-memory driver, which is
the one with general products), the weights in flight, the largest Chow degree (`-` for all),
the `n` of a quotient by `q_0, ..., q_{n-1}` (0 for none), a file of ranks of the backbone (`-`
to read them off the save) and whether to reduce the transposed coboundaries (`y` is
`src/ctau_dual.rs`, which takes the Hopf cofibers and not `C2`). `ctau_e2` is the dense
reference and also reads Lin's maps. The file formats are described on `main` in each example.

    # the transposed export of Cτ and Cτ ∧ C(η) through stem 280
    printf 'S_2\n<copy of the save>\n<out>\n280\nCeta\n0\n<spill dir>\n12\n-\n0\n-\ny\n' | target/release/examples/ctau_scale
    python3 -I ctau_rank_compare.py <legacy of one export> <legacy of another> Ctau Ceta
    printf 'S_2\n<copy of the save>\n200\nCtau\n10\n' | target/release/examples/ctau_orders
