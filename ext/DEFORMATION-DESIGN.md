# Deformed Ext: one abstraction, two instances

*Design note. Prose only — no committed API yet. The point is to write down the
framing before we start folding code into it, so the interface is forced by two
real examples rather than guessed from one.*

## Thesis

The **secondary** ($d_2$) machinery and the **motivic** ($\delta$) machinery are
not two unrelated features that happen to share a differential. They are two
instances of a single deformation-theoretic object. The objects we actually want
are

- `DeformedResolution`
- `DeformedResolutionHomomorphism`
- `DeformedExt`

with the deformation itself as the parameter, and **synthetic-$\mathbb{F}_2$** and
**$\mathbb{C}$-motivic** as the two instances.

The connecting differential of a `DeformedExt` is already realised — it is the
`ExtDifferential` trait (`ext_algebra`), and both the Adams $d_2$
(`SecondaryCoboundary`) and the motivic $\delta$ (`MotivicCoboundary`) implement
it. That trait is the first stone of this abstraction; the rest of the stack has
the same two-instance shape.

## The deformation picture

A deformation here is a graded/filtered lift of a homological-algebra object over
a base ring $R$ carrying a distinguished parameter:

- **special fibre** (mod the parameter) is an *algebraic* object we can resolve
  directly;
- **generic fibre** (invert the parameter) is the thing we actually want;
- the **connecting map** (a Bockstein) is the differential that measures the
  deformation;
- $\Ext$ over the deformation is an **$R$-module**, and its parameter-torsion is
  the interesting output.

| | base $R$ | special fibre (mod parameter) | generic fibre | connecting map |
|---|---|---|---|---|
| **synthetic $\mathbb{F}_2$** | $\mathbb{F}_2[\lambda]$ | primary $\Ext_{\mathcal{A}}$ (Adams $E_2$) | the $\mathbb{F}_2$-Adams target | $\lambda$-Bockstein $= d_2, d_3, \dots$ |
| **$\mathbb{C}$-motivic** | $\mathbb{F}_2[\tau]$ | $\Ext_{\mathcal{A}_{\mathbb C}/\tau}$ (algebraic Novikov $E_1$) | motivic Adams $E_2$ | $\tau$-Bockstein $= d_r$ family |

Both rows are **polynomial** $R$-module deformations. This is the key correction
to an easy misreading (below): the synthetic side is *not* intrinsically
square-zero.

## Square-zero is a truncation, not a species

It is tempting to file the secondary machinery as a *square-zero* deformation
($\lambda^2 = 0$, working in $\mathrm{Mod}_{C\lambda^2}$) and the motivic one as
polynomial, and to treat those as two different kinds of object. That is wrong.

$\lambda$ is a polynomial deformation parameter, exactly like the motivic $\tau$.
Synthetic-$\mathbb{F}_2$ $\Ext$ wants to be a full $\mathbb{F}_2[\lambda]$-module
with a whole $\lambda$-Bockstein tower $d_2, d_3, \dots$. The code stops at
$C\lambda^2$ for a concrete reason: **$\mathbb{F}_2$-synthetic data is not
algebraic in general.** We have Baues' secondary Steenrod algebra as an honest
algebraic model of the *first obstruction only* (the second layer of the
$\lambda$-adic filtration), which yields $d_2$ — and nothing algebraic past it.

On the motivic side we are luckier: $\mathcal{A}_{\mathbb C}$ over
$\mathbb{F}_2[\tau]$ is a genuine algebra with a closed-form product (Kong–Lin,
specialised to $\rho = 0$), so the *entire* tower is computable today.

So the difference between the instances is **not** square-zero vs. polynomial. It
is *how much of the same tower is currently modelled*:

- synthetic $\mathbb{F}_2$: the $\mathbb{F}_2[\lambda]$ tower **truncated at
  $\lambda^2$** (second layer) $\Rightarrow$ $d_2$;
- motivic $\mathbb{C}$: the full $\mathbb{F}_2[\tau]$ tower $\Rightarrow$ the whole
  Bockstein family.

"Support fully synthetic computations eventually" then means *raise the truncation
layer* — model $C\lambda^3, C\lambda^4, \dots$ as the synthetic Steenrod structure
becomes available — not *write a third machine*. The synthetic side is meant to
land on the same rails as the motivic side.

## Why the existing machinery already generalises

Nothing in the shared cohomology path is motivic-specific; it is the general
$R$-module machinery, and the secondary layer is a collapsed special case of it.

- **`graded_dimension` / `matrix_capped`** — the $R$-adic weight cap. Sweeping the
  cap walks up the tower and exposes $R$-torsion. The motivic instance uses the
  full range; the secondary layer is the same thing with the cap range collapsed
  to $\{0, 1\}$, which is *why* it currently looks ungraded. The synthetic side
  will use these verbatim once more layers are modelled.
- **`ExtDifferential`** — one connecting map. The general object is a *family*
  $d_2, d_3, \dots$ indexed by $R$-adic layer (roughly, $R/\mathfrak{m}^n$ realises
  $d_2 \dots d_n$). Secondary $= R/\lambda^2 \Rightarrow d_2$ only; motivic $=$ full
  $\mathbb{F}_2[\tau] \Rightarrow$ all of them.
- **`cohomology_subquotient`** — computes the page (kernel mod image) for either
  differential, unchanged.

## The `Deformed*` family

| `Deformed*` object | synthetic-$\mathbb{F}_2$ instance | motivic-$\mathbb{C}$ instance |
|---|---|---|
| `DeformedResolution` | `SecondaryResolution` (minimal res + secondary homotopies) | `MotivicResolution` (mod-$\tau$ res + lifted $\mathcal{A}_{\mathbb C}$ differentials + weights) |
| `DeformedResolutionHomomorphism` | `SecondaryResolutionHomomorphism` (pair-algebra lift of a chain map) | `TauLift` / `lift_product` ($\mathcal{A}_{\mathbb C}$ lift over $\mathbb{F}_2[\tau]$) |
| `DeformedExt` | `SecondaryExtAlgebra` $\to H(d_2) = E_3$ | `Deformation` / `ext` $\to H(\delta) =$ motivic Adams $E_2$ |
| connecting differential(s) | `SecondaryCoboundary` ($d_2$) | `MotivicCoboundary` ($\delta$) |

The generic interface the family must expose is a **lift-data accessor up to layer
$r$**: `DeformedResolution` returns the $R$-adic lift datum at each layer (the
secondary instance stores the $\lambda^1$ homotopy — one layer, giving $d_2$; the
motivic instance stores the whole $\tau$-tower). Everything downstream —
`DeformedExt`, its cohomology, and its products — is generic over that accessor.

`DeformedResolutionHomomorphism` is what a plain `ResolutionHomomorphism` cannot
be: the lift of a chain map to the deformation, carrying the parameter-part of the
product (the $\lambda$-part / $\tau$-power) that the special-fibre product drops.
This is genuine chain-level data, and it is the one piece that must live in a
parameter-bounded ($\mathtt{PairAlgebra}$, or the $\mathcal{A}_{\mathbb C}$ engine)
object — whether that object is a named struct or a type-erased capability slot on
`ExtAlgebra` is an implementation choice, not a mathematical one.

## A built-in consistency check

Under the $C\tau$-philosophy (Gheorghe–Isaksen–Wang–Xu), $C\tau$-modules match the
algebraic Novikov world and the Adams $d_2$ corresponds to the algebraic-Novikov
$d_1$, i.e. the $\tau$-Bockstein. So the secondary $d_2$ and the motivic $\delta$
are not merely two instances of one interface — at the sphere they are *the same
connecting map viewed two ways*. `DeformedExt`-synthetic and `DeformedExt`-motivic
should therefore produce **matching $d_2$'s on $S$**, computed by two entirely
different engines. That is a regression test worth writing once both instances
sit on the shared abstraction.

## Scope discipline

Build the `Deformed*` interface **after both instances are in the tree**, so the
tower / $R$-module shape is forced by two real polynomial examples rather than
guessed from one. Abstracting from the single available example (or the
one-and-a-half we have while synthetic is truncated at $\lambda^2$) risks baking
the $\lambda^2$-truncation into the interface, which the synthetic-depth work then
has to fight.

`ExtDifferential` is the correct-sized down-payment now: it is the one piece that
already spans both instances *without* committing to either's depth. The rest of
the family waits until the motivic instance and the secondary instance are both
upstream and can be quotiented by their common shape.
