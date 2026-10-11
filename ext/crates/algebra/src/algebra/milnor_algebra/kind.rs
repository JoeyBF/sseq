//! The dual Steenrod algebras that a [`MilnorAlgebra`] can be a sub-Hopf-algebra of.

use fp::prime::ValidPrime;

use super::{Exterior, MilnorAlgebra, MilnorAlgebraInner, MilnorProfile, NoExterior};
use crate::algebra::Algebra;

/// Which Hopf algebra a [`MilnorAlgebra`] is a sub-Hopf-algebra of.
///
/// Each kind exists at some primes only, and fixes the [shape](super::MilnorShape) of the basis
/// there. Every pair of a prime and a shape is exactly one kind, so the kind names what the shape
/// encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MilnorKind {
    /// The Steenrod algebra $A$, at any prime.
    Classical,
    /// The mod-$\tau$ $\mathbb{C}$-motivic Steenrod algebra $A^{\mathbb{C}}/\tau$, at `p = 2`.
    MotivicModTau,
    /// The algebra of reduced powers $P = A/\!/E(Q_0, Q_1, \dots)$, at odd primes, whose dual is
    /// $\mathbb{F}_p[\xi_1, \xi_2, \dots]$.
    ReducedPowers,
}

impl MilnorKind {
    /// Whether this kind's basis has an exterior part at `p`, or `None` if it does not exist at
    /// `p`.
    pub fn has_exterior(self, p: ValidPrime) -> Option<bool> {
        match (self, p == 2) {
            (Self::Classical, two) => Some(!two),
            (Self::MotivicModTau, true) => Some(true),
            (Self::ReducedPowers, false) => Some(false),
            _ => None,
        }
    }

    /// The kind with the given prime and shape.
    pub fn from_shape(p: ValidPrime, has_exterior: bool) -> Self {
        match (p == 2, has_exterior) {
            (true, false) | (false, true) => Self::Classical,
            (true, true) => Self::MotivicModTau,
            (false, false) => Self::ReducedPowers,
        }
    }
}

impl std::fmt::Display for MilnorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Classical => "A",
            Self::MotivicModTau => "A_C/tau",
            Self::ReducedPowers => "P",
        })
    }
}

impl MilnorAlgebra {
    /// The sub-Hopf-algebra of `kind` at `p` given by `profile`, or `None` if `kind` does not
    /// exist at `p`. The default profile gives all of `kind`.
    pub fn of_kind(
        kind: MilnorKind,
        p: ValidPrime,
        profile: MilnorProfile,
        unstable_enabled: bool,
    ) -> Option<Self> {
        Some(if kind.has_exterior(p)? {
            MilnorAlgebraInner::<Exterior>::new_with_profile(p, profile, unstable_enabled).into()
        } else {
            MilnorAlgebraInner::<NoExterior>::new_with_profile(p, profile, unstable_enabled).into()
        })
    }

    /// The Hopf algebra this is a sub-Hopf-algebra of.
    pub fn kind(&self) -> MilnorKind {
        MilnorKind::from_shape(self.prime(), self.has_exterior())
    }
}
