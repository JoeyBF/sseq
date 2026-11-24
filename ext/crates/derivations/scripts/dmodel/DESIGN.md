# The D model: a principled, complete deduction engine for Adams differentials

Criterion: every equation is a theorem about the Adams spectral sequences in the diagram (sound),
and every consequence of those equations is extracted (complete relative to them).

## 1. Where the current engine loses information

The engine works page by page on bounds for E_r:
- certain cycles modulo certain boundaries for sources;
- classes modulo all possible boundaries for targets.

It loses information in three places:
1. A Leibniz or naturality term is dropped when its target has boundary uncertainty.
2. A module pair is dropped when a product is unknown.
3. The domain of d_r (survival to E_r) enters only where it is certain, so every case-split
   deduction is missed.

Items 1 and 3 have one cause: the domain of d_r depends on d_2, …, d_{r−1}, so the system is
disjunctive, not linear, when it is written on pages.

## 2. The D model

For a spectrum X, write V = E2(X), with basis and bigrading (stem n, filtration f).

The spectral sequence from E2 on is that of a filtered complex (V, D):
- D = D2 + D3 + … with D_r: V_{n,f} → V_{n−1,f+r};
- D² = 0.

Every page is a function of D: E_r = Z_r / B_r, and d_r is read through correction terms. D is
unique up to the gauge D ↦ g D g⁻¹ with g = 1 + (filtration-raising).

A morphism of spectral sequences from a map f is a filtered map F = f₀ + F₁ + … with D F = F D.

### Soundness: which statements are theorems

- **(T1)** A filtered complex (V, D) realizing a given spectral sequence of finite-type vector
  spaces exists: pick a basis adapted to every Z_r and B_r (interval / barcode decomposition).
  Theorem.
- **(T2)** Naturality. A map of spectral sequences is realized by a filtered map between
  barcode-adapted models. To be proved here; it is a statement about morphisms of persistence
  modules.
- **(T3)** Leibniz with D a derivation of a filtered product μ = μ₀ + μ₁ + …, where μ₁ is given by
  the λ-secondary products. This is a modelling claim, not known to me as a theorem for the Adams
  spectral sequence over F2: an F2-linear multiplicative model of a synthetic ring may be obstructed,
  since νS is not F2-linear.

  Until it is proved, the engine does not impose D(xy) = Dx·y + x·Dy unconditionally. It imposes
  the theorem-level version, Leibniz on each E_r (T3′), in the D-model encoding of § 3.

## 3. Encoding the conditional theorems exactly

Gauge-fix D in barcode normal form. There is a basis of V in which D pairs basis elements
x ↦ y (d_r x = y on E_r, r = filtration gap); the other basis elements are permanent.

A statement about pages is then a statement about this pairing. The unknown is the pairing
together with the adapted basis (a flag of subspaces per bidegree), not a linear object. The
lower-order part is already pinned:
- D2 is exact (module d2 from the formula + carry; the sphere's d2 from files);
- the residual is a few free directions.

So the structure is almost fixed, and what remains is genuinely finite.

The engine solves:
- **Linear core.** For each r, given the barcode below r (fixed where it is known), the theorems
  at page r are linear in D_r, exactly as the current r = 3 system is once d2 is exact.
- **Exact disjunction.** Where the lower barcode is not determined, each consistent choice of the
  undetermined lower pairings is a branch. Every branch runs the linear core of all higher pages,
  across all spectra jointly. A statement is forced iff it holds in every branch that is
  consistent through r_max. Inconsistent branches are pruned by the higher-page theorems, which is
  exactly the information the current engine leaves on the table.

The number of branches is the obstacle. The lower uncertainty is small and structured (a handful
of free directions per uncertain bidegree), so branching is per connected component of the
dependency graph between uncertain choices, not global. Components are solved independently and
their forced facts combined.

## 4. Theorems imposed (all spectra jointly)

- E2 data: the products μ₀, module actions (now including h_i × lifts), i_*, p_*, maps of filtration
  0.
- d2: exact for every spectrum (sphere: files; modules: the formula + Z/4 carry).
- Page Leibniz (ring and module) and naturality on every E_r, r ≥ 3.
- Secondary-product coupling: [x][y] = λL with xy = 0 in E2 gives d_r constraints on L (as the
  engine's coupling does), and the λ-products of h_i with cycles.
- Known differentials (seeds) only when the run is allowed seeds; never in validation.

## 5. Plan

1. Component analysis. Build the dependency graph of undetermined lower-page data at stem 110 and
   stem 200: which undetermined d_r values share equations. Measure component sizes; this decides
   feasibility.
2. A prototype in the engine: an exact branch over the uncertain choices of a component, with the
   existing page machinery as the linear core per branch. Compare with the current engine; every
   current fact must stay forced.
3. Separately, settle T2 (naturality, provable) and T3 (the multiplicative model). If T3 holds,
   Leibniz becomes unconditional and linear in D, and most branching disappears.
