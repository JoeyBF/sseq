//! The C-motivic (over $\mathbb{C}$, prime 2) Steenrod algebra and its mod-$\tau$
//! reduction.
//!
//! This layer implements the *deformation* view of the C-motivic Adams $E_2$: the
//! C-motivic dual Steenrod algebra $A_C$ over $\mathbb{F}_2[\tau]$, its mod-$\tau$
//! reduction $A_C/\tau$ (a connected finite-type $\mathbb{F}_2$-algebra), and the
//! coefficient ring $\mathbb{F}_2[\tau]$ itself.
//!
//! The three algebras (see `MOTIVIC_PLAN.md`):
//!
//! - [`MotivicMilnorAlgebra`] — $A_C$, over $\mathbb{F}_2[\tau]$. The product
//!   engine ([`milnor`]); resolving $\mathbb{F}_2[\tau]$ over it gives the motivic
//!   Adams $E_2$.
//! - [`CTauAlgebra`] — $A_C/\tau$, an ordinary $\mathbb{F}_2$-[`Algebra`](crate::algebra::Algebra)
//!   ([`ctau`]); its $\mathrm{Ext}$ is the algebraic Novikov $E_2$. This is the
//!   object the existing resolution engine resolves, **unchanged**, so the
//!   classical path stays bit-identical.
//! - the classical Steenrod algebra is a further collapse (invert $\tau$), which
//!   the codebase already handles.
//!
//! The heavy combinatorics live over the field $\mathbb{F}_2$ (via `CTauAlgebra`);
//! the $\tau$-tower is carried by the small [`Tau`] scalar ([`tau`]) and the
//! Phase 2 lift, never threaded through the resolution engine.

pub mod tau;
pub use tau::Tau;

pub mod milnor;
pub use milnor::MotivicMilnorAlgebra;

pub mod ctau;
pub use ctau::CTauAlgebra;
