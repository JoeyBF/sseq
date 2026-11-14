# Comodules and Cotor: design

Status: experimental, on the branch `comodules`. The [plan](#plan) is implemented except where
it says otherwise: `algebra::comodule`, `algebra::MilnorQuotientAlgebra` and `ext::cotor`.

## Goal

Add comodules over the dual Steenrod algebra and its quotient Hopf algebras to the `algebra`
crate, as first-class objects alongside the implementors of `algebra::module::Module`, and compute

$$\mathrm{Cotor}_\Gamma(M, \mathbb{F}_p) \cong \mathrm{Ext}_A(\mathbb{F}_p, M)$$

for bounded-below right comodules $M$ of finite type, in `ext`.

The motivating case is $\mathrm{Ext}_P(Q)$ with $Q = \mathbb{F}_2[v_0, v_1, \dots]$, which by
Nassau's untwisting is the $E_2$ page for $C\tau$ (and through the $v_0$-Bockstein, much of the
$\mathbb{C}$-motivic $E_2$ page). Today that computation lives in `ext/src/algebraic_novikov.rs`,
`vn_bockstein.rs`, `v0_transfer.rs` and `q1_products.rs`, built on types that are comodules in
all but name: `Coaction`, `QDual`, `QDown`, `Psi`, `Layer`. The point of this work is to
name that structure, decide what an efficient comodule interface looks like, and make the Cotor
machinery generic over the comodule, so that the next comodule (a quotient $Q/I_n$, $A/\!/A(n)_*$,
a tensor product for a Cartan–Eilenberg spectral sequence) is an implementor and not a fork.

## Mathematics and conventions

### Comodules as lowering modules

Let $\Gamma$ be $A_*$, a quotient Hopf algebra of it, or a sub-Hopf algebra of it, and
$A = \Gamma^\vee$. A right $\Gamma$-comodule $M$ has a coaction $\psi\colon M \to M \otimes \Gamma$.
When $M$ is bounded below and of finite type, it is the same thing as a *locally finite* left
$A$-module, through

$$a \cdot m = \sum m_i \langle a, \gamma_i \rangle \qquad \text{where } \psi(m) = \sum m_i \otimes \gamma_i.$$

Coassociativity, $(\psi \otimes 1)\psi = (1 \otimes \Delta)\psi$, makes this a left action because
$\Delta$ is dual to the product of $A$. A *left* comodule gives a right $A$-module the same way,
and becomes a left one only through the antipode; this is the source of the $\xi$ versus
$\chi\xi$ confusion recorded below. The `Comodule` trait therefore presents right comodules, and
the Cotor computed is $\mathrm{Cotor}_\Gamma(M, \mathbb{F}_p)$.

This action **lowers** degree: $A_n \otimes M_d \to M_{d-n}$. That is the reason a comodule is not
a `Module`. A `Module` is bounded below and its operations raise degree; the dual $M^\vee$ of $Q$
is bounded *above* and unbounded below, so it cannot satisfy `Module::min_degree`, and every
algorithm in `ext` that walks degrees upwards from `min_degree` would be wrong for it.

### Cotor through a resolution of the unit

With $C \to \mathbb{F}_p$ the minimal free $A$-resolution that `ext` already computes,

$$\mathrm{Cotor}^{s,u}_\Gamma(M, \mathbb{F}_p) = H^s\big(\mathrm{Hom}_A(C, M)\big)_u,$$

where a cochain of weight $u$ assigns to each generator $x$ of $C_s$ an element
$f(x) \in M_{u - |x|}$, and the coboundary is

$$(\delta f)(z) = f(dz) = \sum_{P(R)\,x \,\in\, dz} P(R) \cdot f(x) \in M_{u - |z|}.$$

This is the untwisted complex. Everything is finite in each $(s, u)$ because $M$ is bounded below
and $C_s$ is of finite type. Each $(s, u)$ is independent of every other, so the computation
parallelises completely. The same backbone $C$ serves every comodule $M$.

The other textbook route, a minimal resolution of $M$ by extended comodules $\Gamma \otimes V$, is
out of scope for now: it is dual to resolving $M^\vee$, it shares no backbone between comodules,
and it gives up the independence of bidegrees.

### Conventions to pin down by test, not by prose

- **Basis duality.** The Milnor basis $P(R)$ of `MilnorAlgebra` is dual to the monomial basis
  $\xi^R$ of $A_*$: $\langle P(R), \xi^S \rangle = \delta_{RS}$. A quotient Hopf algebra given by a
  `MilnorProfile` is dual to the subalgebra with that profile, with the same indexing.
- **$\xi$ versus $\zeta = \chi\xi$.** The cross-check of the Cartan–Eilenberg spectral sequence
  showed that the existing `Convention::Conjugate` of `QDown` corresponds to the comodule whose
  coaction is honestly written with $\xi$. An implementor states its coaction in one convention
  and the tests fix which one the induced action uses. Every comodule test must include a
  non-self-conjugate case, because the two conventions agree on small examples.
- **Grading.** At $p = 2$ the $C\tau$ work uses $P$ with degrees halved, which is $A_*$ again.
  Whether $P$ is named as the quotient of $A$ dual to $\mathbb{F}_2[\xi_i^2]$ or as $A$ with a
  halved grading is a choice for the first user; the two must give the same Cotor.

## Measurements that shape the design

From $\mathrm{Ext}_P(Q)$ through `v0_transfer`, on a Zarr-saved backbone, four threads. "Target" is
the stem of $\mathrm{Ext}_P(Q)$; the backbone goes to roughly half of it.

| target | distinct $P(R)$ in $C$ | monomials kept | $(m, R)$ pairs | $(m, R, \gamma)$ triples | table | peak RSS |
|---|---|---|---|---|---|---|
| 160 | 6.6k | 12k | 4.6M | 7.7M | ~0.07 GB | 0.21 GB |
| 200 | 15.4k | 29k | 24M | 45M | ~0.38 GB | 0.94 GB |
| 240 | 30.8k | 63k | 97M | 204M | ~1.6 GB | 3.4 GB |

- The number of distinct operations doubles every 40 target stems. That is the backbone's
  exponential growth (it doubles every 16 classical stems), and it is the only exponential in
  the table: the monomials of $Q$ grow polynomially.
- The table is the action of every operation in $C$ on every monomial it can meet. It grows about
  fourfold per 40 target stems, which predicts roughly 400 GB at target 400, matching an
  out-of-memory failure there at 400 GB.
- Peak memory is about twice the table. The rest is the walk that builds it: intermediate
  coactions keep every $\xi^R$ below a cap, in 128-bit keys, on many branches at once.
- Taking the weights in bands barely helps. For a single weight $u$ the table needs
  $\mathrm{ops}(u - d)$ in each degree $d$, and the operations on a generator of low degree are
  nearly all operations, so even one weight in the middle needs about 60% of the table. With one
  band per weight, peak fell from 2.5 GB to 1.5 GB and the time rose 34-fold.
- The per-weight working set is small next to the table (tens of MB in the middle weights at
  target 240), though each extra thread still costs about 0.3 GB of peak there.

The conclusion that drives everything below: **the expensive object is the action of a set of
operations on a range of degrees, and who asks for it, for which set, and for how long it is kept
must be explicit in the interface**, not hidden inside a table that one implementor builds for
everything up front.

## Interface

### The coalgebra is named by its dual

Everything here is of finite type, so $\Gamma$ is the graded dual of an `Algebra`, and the
structure of $\Gamma$ is that algebra's:

- the coproduct of $\Gamma$ is dual to `Algebra::multiply_basis_elements`;
- the product of $\Gamma$ is dual to `Bialgebra::coproduct`, which carries coefficients and which
  `MilnorAlgebra` implements at every prime and for both shapes: it splits $Q(E) P(R)$ into
  $\pm Q(E') P(S) \otimes Q(E'') P(T)$ over $E = E' \sqcup E''$ and $R = S + T$, the sign that of
  the shuffle of the $\tau$s, so the product of the dual adds exponents;
- the basis of $\Gamma$ is the dual of the algebra's basis, with the same `(degree, index)`.

So a comodule names its coalgebra by its dual algebra, as a `Module` names its algebra, and
`TensorComodule` uses `Bialgebra` exactly as `TensorModule` does. A comodule over a subalgebra
of $A_*$, that is over a quotient algebra of $A$, needs that quotient as an `Algebra` (see
[Quotient algebras](#quotient-algebras-of-a)).

```rust
/// An operation of the dual algebra, as the resolution's differentials name it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpId { pub degree: i32, pub index: u32 }
```

**Why `OpId` is (degree, index) and not a packed exponent vector.** The consumer of a comodule is
a free resolution, whose differentials already name operations as `(operation_degree,
operation_index)`. Packing (`PackedXi` today) is an implementation detail of the coaction walk,
fast for adding exponents but bounded in each exponent. It stays inside the implementor that
needs it, which converts at the boundary with the algebra's `basis_element_from_index`.

### Comodule

The trait, in `algebra::comodule`, mirrors `Module`:

- `type Algebra: Algebra`, the dual algebra; only constructions that need the coproduct, such as
  tensor products, ask for `Bialgebra`.
- `min_degree`, `compute_basis`, `max_computed_degree`, `dimension`, `basis_element_to_string`,
  `max_degree`, as for `Module`.
- `act_on_basis(result, coeff, op_degree, op_index, degree, idx)`, with `result` in degree
  `degree - op_degree`, and `act` on a general element.
- `act_batch(&OpSet, degree) -> ActionBlock`: the action of every operation of a set on every
  basis element of one degree. This is the unit of work: an implementor shares work across the
  operations and across the basis, and the caller decides how long the block lives. The default
  calls `act_on_basis` once per pair.
- `act_batches(&OpSet, max_degree) -> Vec<ActionBlock>`: the same for every degree at once, for
  an implementor that produces them together; `PolynomialComodule` does it in one walk over its
  monomials (`walk_coactions`, which also serves callers that want the raw coaction).

- `OpSet` is a set of operations, sorted by degree, with a dense local numbering so that an
  `ActionBlock` can be indexed by `(local op, basis index)` without hashing. Its down-closure
  (below), for implementors that prune, is still to come.
- `ActionBlock` is the sparse matrix of each operation in the set on $M_d$, stored as one
  compressed row per basis element, in the layout of `Psi<u32>` (keys, starts, targets). At $p > 2$ it carries coefficients; at $p = 2$ it does not.
- The full coaction $\psi(m)$ is deliberately not a required method. For $Q$ it involves every
  $\xi^R$ below the degree of $m$, which is far more than any consumer reads. A comodule that can
  produce it cheaply (a finite one) offers it as an inherent method.

### Filtrations

```rust
/// A comodule with an increasing filtration preserved by the action.
pub trait FilteredComodule: Comodule {
    fn filtration(&self, degree: i32, idx: usize) -> i32;
}
```

This is what `Layer::filtration` is today. Sub and quotient comodules cut out by a filtration
value are then generic (below), and the Bockstein and Cartan–Eilenberg spectral sequences become
consumers of a `FilteredComodule` rather than of $Q$ specifically.

### Quotient algebras of $A$

Sub-Hopf algebras of $A$ are dual to quotient Hopf algebras of $A_*$; `MilnorProfile` covers them,
and they are already resolvable. Quotient Hopf algebras $B$ of $A$ are dual to sub-Hopf algebras
$B_* \subset A_*$, and $\mathrm{Ext}_B(\mathbb{F}_2, \mathbb{F}_2) = \mathrm{Cotor}_{B_*}(\mathbb{F}_2,
\mathbb{F}_2)$. Examples:

- $P = \mathbb{F}_2[\xi_i^2]$, dual to $A /\!/ E(Q_0, Q_1, \dots)$;
- $\mathbb{F}_2[\xi_1, \dots, \xi_n]$, closed under the coproduct because $\Delta\xi_k$ involves
  only $\xi_j$ with $j \le k$;
- generally $\mathbb{F}_2[\xi_i^{2^{e_i}}]$ for exponents satisfying the closure condition dual to
  that of a profile.

Computing Cotor over $B_*$ by the route above needs a resolution over $B$, so $B$ must be an
`Algebra`. Reusing the resolution over $A$ does not work: $B \otimes_A C$ is free over $B$ but not
exact, since $\mathrm{Tor}^A(B, \mathbb{F}_2) \neq \mathbb{F}_2$, and passing from $A$ to $B$
gives a Cartan–Eilenberg spectral sequence rather than an isomorphism.

For the monomial family $B$ is cheap. The kernel of $A \to B$ is spanned by the $P(R)$ with
$\xi^R \notin B_*$, because the Milnor basis is dual to the monomials. So $B$ is the Milnor basis
restricted to the admissible $R$, with the Milnor product and the inadmissible terms dropped:
a second kind of profile, restricting which exponents occur where `MilnorProfile` truncates them.
It implements `Bialgebra` by the same splitting of exponents, so tensor products work over it too.

## Implementors

Mirroring `algebra::module`:

| implementor | contents | replaces |
|---|---|---|
| `FiniteComodule` | dense tables of the action of every operation; `coregular` gives $\Gamma$ truncated | — |
| `PolynomialComodule` | a polynomial comodule algebra given by its generators and their coactions | `Coaction`, `QDual`, `QDown` |
| `TensorComodule` | diagonal coaction, through `Bialgebra::coproduct` | the tensor modules in the CESS check |
| `SubComodule`, `QuotientComodule` | filtration pieces of a `FilteredComodule` | `Layer::Adjoin`, `Layer::Quotient` |
| `SuspensionComodule` | a shift of degree | — |
| `ExtendedComodule` | $\Gamma \otimes V$; covered for now by `TensorComodule` with `FiniteComodule::coregular` | — |

### `PolynomialComodule` and the walk

$Q$, $Q/I_n$, $A/\!/A(n)_* = \mathbb{F}_2[\zeta_1^{2^{n+1}}, \zeta_2^{2^n}, \dots]$ and
$H_*BP$ are all polynomial comodule algebras: $\psi$ is a ring map, so it is determined by
$\psi$ on generators. `PolynomialComodule` takes the generators, their degrees, and their
coactions as sums of $\xi^S \otimes (\text{monomial})$, and computes `act_batch` by the walk
`QDown` uses: each monomial is a child of a smaller one with one binary chunk $x^{2^j}$ peeled
off, and its coaction is the product of the parent's with the Frobenius twist of the chunk's.

Two changes from today's walk, both forced by the measurements:

1. **Prune to the down-closure of the request.** Along the walk the exponent vector $R$ of each
   term only grows, because the product of $A_*$ adds exponents. So a term whose $R$ is not
   below some element of the requested `OpSet` can never contribute and is dropped as soon as it
   appears. Today the walk prunes only to the componentwise maximum of all operations, which is
   why its intermediates are larger than the table.
2. **The walk serves one batch, not the whole range.** It runs for the degrees and operations
   asked for and returns; no global table is built inside the comodule.

The existing ordering of monomials by length then exponents, which makes each $Q_k$ (monomials
of length at most $k$) a prefix of every degree, becomes the basis ordering of the
`FilteredComodule` by length, so `SubComodule` by length is a prefix and costs nothing.

## Cotor in `ext`

```rust
pub struct CotorComplex<CC: FreeChainComplex, M: Comodule> { backbone: Arc<CC>, comodule: Arc<M>, ... }

impl<CC, M> CotorComplex<CC, M> {
    /// The cochains of filtration s and weight u, with their coboundary.
    pub fn coboundary(&self, s: i32, u: i32, cache: &ActionCache<M>) -> (Vec<Cochain>, SparseMatrix);
    /// The dimension of Cotor^{s,u}.
    pub fn cotor(&self, s: i32, u: i32) -> usize;
}
```

The coboundary at $(s, u)$ needs, for each generator degree $t$, the action of
$\mathrm{ops}(t)$, the operations on generators of $C$ of degree $t$, on $M_{u-t}$. `CotorComplex`
owns the policy for those `ActionBlock`s through an `ActionCache`, with three policies to
compare:

- **Global**: one block per degree for the union of all operations, kept for the whole run. This
  is today's table: fastest, and memory as measured above.
- **Per weight**: blocks for one weight's $(t, u - t)$ pairs, computed on demand and dropped after
  the weight. Memory is one weight's need per thread; time is multiplied by how many weights
  share a block.
- **Spilled**: the global table written once to disk (the backbone's Zarr store is the natural
  place) and memory-mapped. The table is immutable and read in degree order, so this trades
  memory for disk at little cost in time, and it is the only one of the three that plausibly
  reaches target 400 on one node without recomputation.

The coboundary itself is assembled per weight and never cached, as today.

The $v_0$-transfer, the $v_n$-Bocksteins and the products of `q1_products` then take a
`FilteredComodule` and a `CotorComplex` instead of building `Stage` and `QDown` themselves.

## Tests

Implemented:

- **Coproduct**: multiplicative, with Koszul signs, at $p = 2, 3, 5$ for both Milnor shapes.
- **Coregular**: $\Gamma$ truncated is a left module (associativity checked on every pair of
  operations), its batched action agrees with the single one, and its Cotor is the ground field
  in degree zero.
- **Unit**: Cotor of the ground field against `ext`'s Ext of the sphere.
- **Change of rings**: the dual of $A(1) \backslash A$, built by `ext::cotor::coinduced` as a
  `FiniteComodule`, against $\mathrm{Ext}_{A(1)}(\mathbb{F}_2, \mathbb{F}_2)$ from a resolution
  over the profile algebra.

To come:

- **Change of rings for `PolynomialComodule`**: $A/\!/A(1)_*$ and $A/\!/E(0)_*$ presented by
  generators, against the same answers; their coactions are not self-conjugate.
- **Finite comodules from modules**: the dual of an `FDModule` needs the antipode to become a
  left module, so this waits for one.
- **Regression**: $\mathrm{Ext}_P(Q)$ and $\mathrm{Ext}_P(Q/I_1)$ through `CotorComplex` against
  `ctau_e2_untwisted_on` and the saved CSVs, at small stems in the test suite and at large stems
  by hand.
- **Policies**: the three `ActionCache` policies give identical coboundaries.

## Plan

1. *Done.* `OpId`; `Comodule`, `OpSet`, `ActionBlock`; `FiniteComodule`; the coproduct with
   coefficients at every prime. Cotor in `ext` with the global policy.
2. *Done.* `PolynomialComodule`, with `dual_steenrod_subalgebra` for the change-of-rings tests,
   and `ext::algebraic_novikov::q_comodule`; $\mathrm{Ext}_P(Q)$ through `CotorComplex` agrees
   with `ctau_e2_untwisted_on`.
3. *Done in part.* `FilteredComodule`, `SubComodule`, `QuotientComodule`;
   `CotorComplex::cotor_by_filtration`, and `q_mod_v0_comodule`, whose pieces are every $Q_k$
   at once. The transfer's coaction table (`Stage` for the layers that set $v_0 = 1$) comes from
   the comodule walk. `Stage`, `QDown` and the transfer itself are not yet generic over a
   comodule, and the quotients by $v_0$ keep the `QDown` walk.
4. *Global policy only.* `CotorComplex::precompute_actions` fills the cache in one walk. The
   per-weight and spilled policies are not written: with the walk, the action is no longer what
   limits the direct computation (see below), and for the transfer its table is the same size
   as before.
5. *Done in part.* `TensorComodule`, checked by associativity at $p = 2, 3$ (the Koszul sign is
   needed at 3) and by the cofree comodule $M \otimes \Gamma$. The Cartan–Eilenberg check is
   not written.
6. *Done.* `MilnorQuotientAlgebra`, the quotient dual to $\mathbb{F}_p[\xi_k^{p^{e_k}}]$,
   resolved by `ext` as it stands: the quotient dual to $\mathbb{F}_2[\xi_k^2]$ gives
   $\mathrm{Ext}_A$ doubled, and the one dual to $\mathbb{F}_2[\xi_1]$ gives
   $\mathbb{F}_2[h_{1j}]$.

### Measurements of the implementation

$\mathrm{Ext}_P(Q)$ by the direct method, `cotor_q` against `ctau_e2_untwisted_on`, four threads,
same classes:

| target | action | total | old total |
|---|---|---|---|
| 80 | 0.02 s | 0.04 s | 0.53 s |
| 120 | 0.33 s | 0.85 s | 18.4 s |
| 140 | 1.2 s | 3.1 s | 100.7 s |
| 160 | 3.4 s | 11.6 s | |
| 200 | 24.7 s | 122 s | |

Most of the old time was the action, recomputed per pair of operation and monomial through a
memo. The walk computes each monomial's coaction once, from its parent's. At 200 the reductions
dominate, and the $v_0$-transfer, which avoids most of them, is the faster method.

For the transfer, at target 200 the comodule walk and the `QDown` walk give the same classes in
70 s and 74 s, both peaking at 649 MB: the two walks are the same algorithm, so this gains
generality, not speed.

### What is left

- **Make the transfer generic.** `Stage` builds its coboundaries from `QDown` bases and the
  `Psi` table; written against a `FilteredComodule` and `ActionBlock`s, the transfer and the
  Bocksteins would run on any filtered comodule, and `QDown` and `QDual` could go.
- **Find what sets the transfer's peak.** Pruning the walk to the exact down-closure of the
  requested operations, instead of their componentwise maximum, was tried at target 200 and
  changed neither the peak (650 MB) nor the time, so the walk's intermediates are not it there.
- **Spill the table to the backbone's store** for targets where it does not fit in memory.
- **The Cartan–Eilenberg check**, $\mathrm{Ext}_P(Q \otimes M) \Rightarrow \mathrm{Ext}_P(\Phi_* M)$,
  through `TensorComodule`.
- **An antipode** for `MilnorAlgebra`, to make a left comodule (the dual of an `FDModule`) a
  right one.

## Open questions

- **Odd primes.** Nothing above is specific to $p = 2$ except the coefficient-free
  `ActionBlock`. Supporting $p > 2$ from the start costs a coefficient vector per block; deferring
  it risks baking $p = 2$ into `PolynomialComodule`.
- **Motivic and multigraded coalgebras.** The motivic $A_*$ is bigraded and has
  $\tau$-torsion-free but non-polynomial structure. `OpId` with a single degree does not cover it;
  the multigrade work may settle what a degree is before this needs to.
- **Left comodules.** The trait presents right comodules. A left comodule, such as $A/\!/A(1)_*$
  as usually written, becomes a right one through the antipode, which `MilnorAlgebra` does not
  have yet. Whether that is a wrapper comodule or a constructor can wait for the first user;
  `ext::cotor::coinduced` sidesteps it by building the right comodule directly.
- **Whether `act_on_basis` earns its place.** Every consumer so far wants batches. It is kept
  because it makes `FiniteComodule` and the tests simple, and because it is the method a
  newcomer reaches for; if it invites slow code it should go.
