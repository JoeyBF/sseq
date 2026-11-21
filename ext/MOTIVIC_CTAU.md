# The Adams E2 page of Cτ from the classical resolution

Design and results for `src/ctau.rs` and `examples/ctau_e2.rs` (the dense reference),
`src/ctau_scale.rs`, `examples/ctau_scale.rs` and `fp::matrix::SparseEchelon` (the sparse
implementation), and the scripts `ctau_check.py`, `ctau_compare.py`, `ctau_legacy.py`,
`ctau_towers_check.py` and `ctau_plan.py`.
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
  (sections 7, 9, 10).
* The coboundaries are 0.4% to 0.7% dense at every size, so the cost is the size of the output,
  which grows about five times per 40 stems. Stem 400 is about a day per complex after two
  storage changes and a batch job of its own; the complete page at stem 800 is out of reach of
  any elimination, and the
  part of Chow degree `k = stem + filtration - 2 weight` at most 10 to 15 at stem 800 (which
  contains everything of Adams filtration at most `k`) is the size of the stem 350 to 400
  computations (section 11).
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

**Stem 400.** Not run. From stem 350, 50 stems further is about 8 times the entries (1e12), 14
times the additions (1e14, about a day of reduction at the measured rate) and 6 times the
reduced columns (1.4e11 entries, 570 GB as stored now), with operation tables of about 3e10
entries. That is over a terabyte in the present layout, so the driver script refuses it. It fits with two changes that are bookkeeping, not research: store reduced
columns as byte-coded gaps (they are sorted indices, a fourfold saving), and drop a weight's
columns once the maps into it are computed. With those, `Cτ` through stem 400 is about a day
on one node, and each cofiber three to five times that.

## 10. Results and where they are

Under `/rs/rs_grp_csht/resolutions/ctau_e2/`:

| archive | contents |
|---|---|
| `towers_n240.tar.zst` | stem ≤ 240: towers, `h_1..h_4`, cell maps for `Cτ`, `C(2)`, `C(η)`, `C(ν)`, `C(σ)` |
| `towers_n280.tar.zst` | the same through stem 280 |
| `ctau_n320.tar.zst` | stem ≤ 320: towers and `h_1..h_4` of `Cτ`, and the towers of `C(2)` without its maps (the run was killed there) |
| `ctau_n350.tar.zst` | stem ≤ 350: the towers of `Cτ` only (the run was killed in `h_j`) |
| `products_n120.tar.zst`, `products_n160.tar.zst` | the five complexes through stem 120 and 160, with `Ctau_generators.csv` and `Ctau_products.csv` (weight ≤ 16) |

Through stem 350, `Cτ` has 8540409 towers (5272324 through stem 320, 2605517 through 280). Stem 400 was not run; see
"Stem 400" in the section before.

Each archive is one `tar | zstd` file; nothing is unpacked on panfs. The scratch copies under
`/dev/shm` are not kept.

## 11. Reaching stem 800

**Data.** The backbone through classical stem 400 covers motivic stem 800 (section 4). It is 104
GB on disk and, by the count through stem 120, of the order of 1e10 to 1e11 terms: it has to be
streamed one filtration at a time, which `StreamedBackbone` does not do yet (it keeps every
term).

**Size.** With a density of 0.7% and the dimensions of `ctau_plan.py` (the sum of rows times
columns over the coboundaries), rounded to a power of ten or so:

| range | largest coboundary | coboundary entries |
|---|---|---|
| stem ≤ 350, complete (measured) | 1.8e5 square | 1.3e11 |
| stem ≤ 400, complete | 3.6e5 square | 1e12 |
| stem ≤ 800, `k ≤ 10` | 8.8e4 square | 1e11 |
| stem ≤ 800, `k ≤ 15` | 3.4e5 square | 1e12 |
| stem ≤ 800, `k ≤ 20` | 8.3e5 square | 6e12 |
| stem ≤ 800, complete | 2.5e7 square | 2e15 |

The same recipe gives 2e11 for stem 400 where the measured trend gives 1e12, so read the table as low by up to five times. The density at bounded `k` has not been measured and may differ. The complete page at stem 800
is petabytes of coboundary: sparse elimination does not change that, because the matrices are
not sparse enough. What it changes is everything up to there. The part of bounded Chow degree
`k` at stem 800 (which contains everything of Adams filtration at most `k`) is, for `k ≤ 10`
and `k ≤ 15`, the size of the stem-350 and stem-400 computations.

**What is needed for stem 800 at bounded `k`**, in order:

1. A `max_k` in `Space`: levels above the bound are simply not built. Each `k` is its own
   complex, so this is exact.
2. Streaming the backbone by filtration, and keeping a weight's reduced columns only while the
   maps into that weight are computed. Reduced columns are the memory: 0.34 of the input and
   growing 5 to 6 times per 40 stems.
3. Operation tables by degree on demand. They grow 5 to 6 times per 20 degrees and at degree 200
   the full coaction has 1e5 terms per monomial; at bounded `k` only monomials of length at most
   `k` are needed, which is a different and much smaller table.
4. Products through tables (section 7).

**Complete page beyond stem 400.** The limit is the output. A structured reduction that never
forms the cochains (the Bockstein tower `Ext(Q/(q_0..q_n))[q_n] ⇒ Ext(Q/(q_0..q_{n-1}))`
starting from classical Ext) is the only route I see, and it is not started.

**`Cτ ∧ C(η)` as the target.** Most of `Ext(Cτ)` is `h_1`-towers, and they collapse in the
cofiber of `h_1`: 64172 towers against 1180528 at stem 240, 141763 against 2605517 at stem 280,
eighteen times fewer at both. The reduction does not get cheaper, because the complex here is
the cone of `h_1` on the cochains of `Cτ`, twice the cochains:

| stem | complex | cochains | coboundary entries | entries added | reduced kept | reduction | `h_j` |
|---|---|---|---|---|---|---|---|
| 200 | `Cτ` | 3.5e6 | 2.0e8 | 1.9e9 | 8.3e7 | 1.9 s | 2.0 s |
| 200 | `Cτ ∧ C(η)` | 6.8e6 | 6.5e8 | 5.3e9 | 2.3e8 | 8.6 s | 2.0 s |
| 240 | `Cτ` | 9.8e6 | 1.4e9 | 2.4e10 | 4.6e8 | 24 s | 17 s |
| 240 | `Cτ ∧ C(η)` | 1.9e7 | 4.4e9 | 6.3e10 | 1.3e9 | 60 s | 12 s |
| 280 | `Cτ` | 2.4e7 | 7.7e9 | 2.2e11 | 2.1e9 | 153 s | 130 s |
| 280 | `Cτ ∧ C(η)` | 4.8e7 | 2.5e10 | 5.8e11 | 6.0e9 | 727 s | 113 s |

So the cone costs 3 times the input, 2.6 to 2.8 times the additions, 2.9 times the memory and 2.5
to 4.8 times the reduction time of `Cτ`; only what comes after the reduction (the `h_j`, the
export, everything downstream) gets the eighteenfold saving. The engine can use `Cτ ∧ C(η)` as a
node in its own right, with `S ∧ C(η) → Cτ ∧ C(η)` for naturality, and never needs `Cτ`
rebuilt from it; the `Cτ` reduction then need not be kept in memory, which removes one of the
two echelons the cell maps hold.

Can the input shrink too? Three things were considered, none measured.

* Resolving the cofiber instead of coning it. `h_1` is the classical `h_0`, so `Cτ ∧ C(η)` is
  `Ext_A(F_2, Q ⊗ D M)` for the classical mod 2 Moore module `M`, and with a minimal resolution
  `P'` of `M` the cochains are `Hom_A(P', Q)`: one copy, with the generators of `Ext_A(M)` in
  place of those of `Ext_A(F_2)`. That is about the size of the `Cτ` complex instead of twice
  it, a gain of 2 to 3, not of 18: the cochains are large because `Q` is, not because of the
  `h_1`-towers. Only the block on the stem-0 generators `h_0^a` goes away, and the large spaces
  sit at Novikov filtration 5 to 11 where it is a small part. It also needs a minimal resolution
  of the classical Moore spectrum through stem 400, which is not on `/rs` as far as I know and
  which the rules do not let me compute.
* `h_1`-periodicity. Above a line the multiplication by `h_1` is an isomorphism, so `Ext(Cτ)`
  there is determined by one column. In these coordinates that region is high Novikov
  filtration, where the cochain spaces are already small; the cost is elsewhere.
* The `h_1`-local closed form (Guillou–Isaksen). It gives the infinite `h_1`-towers without
  computing, which is a check on the tower counts of `Cτ` and a way to skip exporting them, but
  for the same reason it does not touch the large spaces.

So the plan for stem 800 with `Cτ ∧ C(η)` as the target is the plan above with every cost
multiplied by about 3 on the input side and divided by 18 on the output side: at bounded Chow
degree, `k ≤ 10` is about three stem-350-sized reductions and `k ≤ 15` about three of stem 400;
the complete page stays out of reach, since its obstacle is the input. With a minimal resolution
of the classical Moore spectrum the factor 3 becomes about 1.

## 12. Fitting together with the native route (#288, #289, #290)

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

## 13. Not done here

* General products at scale (section 7), and products on the cofibers beyond `h_1..h_4`.
* The E2 page of the motivic sphere and the maps `S → Cτ → S^{1,-1}` on E2 pages (section 12);
  the Adams differentials of `Cτ`; names. Not started, as agreed.
* Massey products, by request.

## 14. Running it

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
to multiply by (0 for none). `ctau_e2` is the dense reference and also reads Lin's maps. The
file formats are described on `main` in each example.
