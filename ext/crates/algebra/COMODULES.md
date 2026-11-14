# Comodules and Cotor: design

Status: proposal, on the experimental branch `comodules`. Nothing here is implemented yet.

## Goal

Add comodules over the dual Steenrod algebra and its quotient Hopf algebras to the `algebra`
crate, as first-class objects alongside the implementors of `algebra::module::Module`, and compute

$$\mathrm{Cotor}_\Gamma(\mathbb{F}_p, M) \cong \mathrm{Ext}_A(\mathbb{F}_p, M)$$

for bounded-below comodules $M$ of finite type, in `ext`.

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

Let $\Gamma$ be $A_*$ or a quotient Hopf algebra of it, and $A = \Gamma^\vee$. A left
$\Gamma$-comodule $M$ has a coaction $\psi\colon M \to \Gamma \otimes M$. When $M$ is bounded below
and of finite type, it is the same thing as a *locally finite* left $A$-module, through

$$a \cdot m = \sum \langle a, \gamma_i \rangle\, m_i \qquad \text{where } \psi(m) = \sum \gamma_i \otimes m_i.$$

This action **lowers** degree: $A_n \otimes M_d \to M_{d-n}$. That is the reason a comodule is not
a `Module`. A `Module` is bounded below and its operations raise degree; the dual $M^\vee$ of $Q$
is bounded *above* and unbounded below, so it cannot satisfy `Module::min_degree`, and every
algorithm in `ext` that walks degrees upwards from `min_degree` would be wrong for it.

### Cotor through a resolution of the unit

With $C \to \mathbb{F}_p$ the minimal free $A$-resolution that `ext` already computes,

$$\mathrm{Cotor}^{s,u}_\Gamma(\mathbb{F}_p, M) = H^s\big(\mathrm{Hom}_A(C, M)\big)_u,$$

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
- **Grading.** At $p = 2$ the $C\tau$ work uses $P$ with degrees halved. The coalgebra carries its
  own degree function; a halved $P$ is a coalgebra in its own right, not a flag on $A_*$.

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

### Coalgebra

```rust
/// A graded connected Hopf algebra over F_p, the dual of an `Algebra` in the same basis.
pub trait Coalgebra: Send + Sync + 'static {
    type Dual: Algebra;
    fn dual(&self) -> Arc<Self::Dual>;
    fn prime(&self) -> ValidPrime;
    fn compute_basis(&self, degree: i32);
    fn dimension(&self, degree: i32) -> usize;
    /// The coproduct of a basis element, as pairs of basis elements.
    fn coproduct_on_basis(&self, degree: i32, idx: usize) -> Vec<(OpId, OpId)>;
    /// The product of two basis elements, dual to the coproduct of `Dual`.
    fn multiply_basis_elements(&self, a: OpId, b: OpId) -> Option<OpId>;
    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String;
}

/// A basis element of a coalgebra: its degree and its index in the basis of the dual algebra.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpId { pub degree: i32, pub index: u32 }
```

The first implementor is `MilnorCoalgebra`, wrapping an `Arc<MilnorAlgebra>` (any profile, so
$A(n)_*$ and $P$ come for free). It reuses the algebra's basis and indexing, so pairing is the
identity on indices. The product of $A_*$ is polynomial, so `multiply_basis_elements` adds
exponents and is single-valued at $p = 2$ (hence `Option`, for exceeding a profile).

**Why `OpId` is (degree, index) and not a packed exponent vector.** The consumer of a comodule is
a free resolution, whose differentials already name operations as `(operation_degree,
operation_index)`. Packing (`PackedXi` today) is an implementation detail of the coaction walk,
fast for adding exponents but bounded in each exponent. It stays inside the implementor that
needs it, which converts at the boundary with the algebra's `basis_element_from_index`.

### Comodule

```rust
pub trait Comodule: std::fmt::Display + Send + Sync + 'static {
    type Coalgebra: Coalgebra;
    fn coalgebra(&self) -> Arc<Self::Coalgebra>;
    fn min_degree(&self) -> i32;
    fn compute_basis(&self, degree: i32);
    fn max_computed_degree(&self) -> i32;
    fn dimension(&self, degree: i32) -> usize;
    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String;

    /// The lowering action of one operation on one basis element, added into `result`, which
    /// lives in degree `degree - op.degree`.
    fn act_on_basis(&self, result: FpSliceMut, coeff: u32, op: OpId, degree: i32, idx: usize);

    /// The action of every operation in `ops` on every basis element of `degree`.
    ///
    /// The batch is the unit of work: implementors share work across the operations and across
    /// the basis elements of a degree, and the caller decides how long the result lives.
    fn act_batch(&self, ops: &OpSet, degree: i32) -> ActionBlock {
        // Default: one `act_on_basis` per pair.
    }
}
```

- `OpSet` is a set of operations, sorted by degree, with a dense local numbering so that an
  `ActionBlock` can be indexed by `(local op, basis index)` without hashing. It also exposes its
  down-closure (below), which implementors may use to prune.
- `ActionBlock` is the sparse matrix of each operation in the set on $M_d$, stored as one
  compressed row per basis element: the existing `Psi<u32>` layout (keys, starts, targets),
  promoted to a public type. At $p > 2$ it carries coefficients; at $p = 2$ it does not.
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

## Implementors

Mirroring `algebra::module`:

| implementor | contents | replaces |
|---|---|---|
| `FiniteComodule` | dense tables of the action, read from JSON like `FDModule` | — |
| `PolynomialComodule` | a polynomial comodule algebra given by its generators and their coactions | `Coaction`, `QDual`, `QDown` |
| `TensorComodule` | diagonal coaction, through `Coalgebra::multiply_basis_elements` | the tensor modules in the CESS check |
| `SubComodule`, `QuotientComodule` | filtration pieces of a `FilteredComodule` | `Layer::Adjoin`, `Layer::Quotient` |
| `SuspensionComodule` | a shift of degree | — |
| `ExtendedComodule` | $\Gamma \otimes V$, for change-of-rings tests | — |

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

- **Unit**: $\mathrm{Cotor}_{A_*}(\mathbb{F}_2, \mathbb{F}_2)$ against `ext`'s Ext of the sphere.
- **Change of rings**: $\mathrm{Cotor}_{A_*}(\mathbb{F}_2, A/\!/A(1)_*)$ against
  $\mathrm{Ext}_{A(1)}(\mathbb{F}_2, \mathbb{F}_2)$ from a resolution over the profile algebra,
  and $A/\!/E(0)_*$ against $\mathbb{F}_2[h_0]$. These exercise `PolynomialComodule` on a
  comodule other than $Q$, with a coaction that is not self-conjugate.
- **Finite**: `FiniteComodule` from the dual of an existing `FDModule` (for example $C\eta$)
  against `ext`'s Ext of that module, with the grading reversed.
- **Regression**: $\mathrm{Ext}_P(Q)$ and $\mathrm{Ext}_P(Q/I_1)$ through `CotorComplex` against
  `ctau_e2_untwisted_on` and the saved CSVs, at small stems in the test suite and at large stems
  by hand.
- **Policies**: the three `ActionCache` policies give identical coboundaries.

## Plan

1. `Coalgebra`, `OpId`, `MilnorCoalgebra`; `Comodule`, `OpSet`, `ActionBlock`; `FiniteComodule`.
   Cotor of the unit and of a finite comodule in `ext`, with the global policy only.
2. `PolynomialComodule` with the pruned walk; the change-of-rings tests; $\mathrm{Ext}_P(Q)$
   regression against the existing code.
3. `FilteredComodule`, sub and quotient comodules; port `vn_bockstein` and `v0_transfer` onto
   `CotorComplex`, and retire `QDown`, `QDual` and `Stage`.
4. The per-weight and spilled policies, measured at targets 240 and 320.
5. `TensorComodule` and the Cartan–Eilenberg check.

## Open questions

- **Odd primes.** Nothing above is specific to $p = 2$ except the coefficient-free
  `ActionBlock`. Supporting $p > 2$ from the start costs a coefficient vector per block; deferring
  it risks baking $p = 2$ into `PolynomialComodule`.
- **Motivic and multigraded coalgebras.** The motivic $A_*$ is bigraded and has
  $\tau$-torsion-free but non-polynomial structure. `OpId` with a single degree does not cover it;
  the multigrade work may settle what a degree is before this needs to.
- **Right comodules and the opposite convention.** $\mathrm{Cotor}(M, \mathbb{F}_p)$ and two-sided
  Cotor are not needed yet. A right comodule is a left one over $\Gamma$ with the conjugate
  coproduct; whether that is a wrapper or a parameter can wait for the first user.
- **Whether `act_on_basis` earns its place.** Every consumer so far wants batches. It is kept
  because it makes `FiniteComodule` and the tests simple, and because it is the method a
  newcomer reaches for; if it invites slow code it should go.
