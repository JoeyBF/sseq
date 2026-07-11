//! $A_C/\tau$: the mod-$\tau$ reduction of the C-motivic Steenrod algebra,
//! presented to the resolution engine as an ordinary $\mathbb{F}_2$-algebra.
//!
//! Setting $\tau = 0$ in $A_C$ collapses the defining relation $\tau_i^2 =
//! \tau\xi_{i+1}$ to $\tau_i^2 = 0$, so
//! $$A_C/\tau \;\cong\; \mathbb{F}_2[\xi_1, \xi_2, \dots] \otimes E(\tau_0, \tau_1, \dots).$$
//! This is a **connected, finite-type $\mathbb{F}_2$-algebra** — once $\tau$ is
//! gone every generator has positive stem — and it is *strictly bigger* than the
//! classical dual Steenrod algebra $\mathbb{F}_2[\xi_i]$ (it carries the extra
//! exterior generators $\tau_i$). Its cohomology $\mathrm{Ext}_{A_C/\tau}$ is the
//! **algebraic Novikov $E_2$**, equivalently the motivic Adams $E_2$ of $C\tau$
//! (Gheorghe–Wang–Xu).
//!
//! Because $A_C$ is a free $\mathbb{F}_2[\tau]$-module on the Milnor basis
//! $\{Q(E)P(R)\}$, the reduction $A_C/\tau = A_C \otimes_{\mathbb{F}_2[\tau]}
//! \mathbb{F}_2$ has *the same basis*, and its product is the $\tau^0$ part of the
//! $A_C$ product: reduce each structure constant $\tau^k$ mod $\tau$, keeping only
//! $k = 0$. So this type is a thin $\mathbb{F}_2$ view over the
//! [`MotivicMilnorAlgebra`] engine, and the ordinary ($\mathbb{F}_p$-only)
//! resolution engine can resolve over it unchanged.
//!
//! Each basis element additionally carries a **motivic weight** (via
//! [`CTauAlgebra::weight`]); the algebra is graded by the single topological
//! degree `t`, with weight a bounded per-basis-element label that the Phase 2
//! lift consumes.

use fp::{
    prime::{TWO, ValidPrime},
    vector::FpSliceMut,
};

use super::MotivicMilnorAlgebra;
use crate::algebra::Algebra;

/// $A_C/\tau$, the mod-$\tau$ C-motivic Steenrod algebra as an $\mathbb{F}_2$-algebra.
///
/// Wraps the [`MotivicMilnorAlgebra`] $A_C$ engine and presents its $\tau^0$
/// product to the resolution engine. Resolving the trivial module over this
/// algebra yields the algebraic Novikov $E_2$.
#[derive(Default)]
pub struct CTauAlgebra {
    ac: MotivicMilnorAlgebra,
}

impl CTauAlgebra {
    pub fn new() -> Self {
        Self {
            ac: MotivicMilnorAlgebra::new(),
        }
    }

    /// The underlying $A_C$ product engine (over $\mathbb{F}_2[\tau]$), shared so
    /// that the Phase 2 lift can recover the $\tau$-powers this $\mathbb{F}_2$ view
    /// discards.
    pub fn engine(&self) -> &MotivicMilnorAlgebra {
        &self.ac
    }

    /// The motivic weight of the `idx`-th basis element in topological degree `t`.
    ///
    /// $A_C/\tau$ is graded by the single degree `t`; the weight is an extra label
    /// carried per basis element (inherited from the $A_C$ bigrading, where
    /// reducing mod $\tau$ does not change weights).
    pub fn weight(&self, t: i32, idx: usize) -> i32 {
        self.ac.bidegree(t, idx).1
    }
}

impl std::fmt::Display for CTauAlgebra {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "CTauAlgebra(A_C/τ, p=2)")
    }
}

impl Algebra for CTauAlgebra {
    fn prefix(&self) -> &str {
        "motivic_ctau"
    }

    fn magic(&self) -> u32 {
        // Distinct from the classical Milnor/Adem magics so motivic save files can
        // never be cross-loaded.
        0x004D_0003
    }

    fn prime(&self) -> ValidPrime {
        TWO
    }

    fn compute_basis(&self, degree: i32) {
        self.ac.compute_basis(degree);
    }

    fn dimension(&self, degree: i32) -> usize {
        self.ac.dimension(degree)
    }

    fn multiply_basis_elements(
        &self,
        mut result: FpSliceMut,
        coeff: u32,
        r_degree: i32,
        r_idx: usize,
        s_degree: i32,
        s_idx: usize,
    ) {
        // The A_C/τ product is the τ^0 part of the A_C product: keep the structure
        // constants of τ-valuation 0, drop everything divisible by τ.
        for (tau, idx) in self.ac.product_indexed(r_degree, r_idx, s_degree, s_idx) {
            if tau.valuation() == Some(0) {
                result.add_basis_element(idx, coeff);
            }
        }
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        self.ac.basis_element_to_string(degree, idx)
    }

    fn basis_element_from_string(&self, _elt: &str) -> Option<(i32, usize)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use fp::vector::FpVector;

    use super::*;

    #[test]
    fn test_ctau_dimensions_match_ac_basis() {
        // A_C/τ has the same basis as A_C (same Milnor monomials per degree), so its
        // F_2-dimensions are the A_C F_2[τ]-ranks. Spot-check low degrees:
        //   t=0: 1 (unit); t=1: Q_0; t=2: P(ξ_1); t=3: Q_1, Q_0 P(ξ_1) → dim 2.
        let alg = CTauAlgebra::new();
        alg.compute_basis(6);
        assert_eq!(alg.dimension(0), 1);
        assert_eq!(alg.dimension(1), 1);
        assert_eq!(alg.dimension(2), 1);
        assert_eq!(alg.dimension(3), 2);
    }

    #[test]
    fn test_ctau_weight_labels() {
        // The weight label is the A_C bigrading's weight coordinate, carried through the
        // mod-τ reduction unchanged. In this presentation the algebra weight is the
        // *negative* of the dual monomial's weight (so products stay homogeneous with τ,
        // of weight −1): Q_0 = Sq^1 has weight 0; the ξ_1-dual (t = 2) has weight −1.
        // (The standard-sign motivic weight is recovered by a single global negation when
        // the Phase 1 chart is compared to published data.)
        let alg = CTauAlgebra::new();
        alg.compute_basis(2);
        assert_eq!(alg.weight(1, 0), alg.engine().bidegree(1, 0).1);
        assert_eq!(alg.weight(1, 0), 0);
        assert_eq!(alg.weight(2, 0), -1);
    }

    #[test]
    fn test_ctau_product_is_tau0_part_of_ac() {
        // The defining contract: the A_C/τ product of two basis elements is exactly the
        // τ^0 subset of the A_C (F_2[τ]) product. Verify this over a range, and confirm
        // that some product genuinely has a τ-divisible term that gets dropped (so the
        // reduction is doing real work, not a no-op).
        use std::collections::BTreeSet;

        let alg = CTauAlgebra::new();
        alg.compute_basis(10);
        let mut saw_drop = false;
        for t1 in 0..=5 {
            for idx1 in 0..alg.dimension(t1) {
                for t2 in 0..=(5 - t1) {
                    for idx2 in 0..alg.dimension(t2) {
                        let t = t1 + t2;
                        let raw = alg.engine().product_indexed(t1, idx1, t2, idx2);
                        let expected: BTreeSet<usize> = raw
                            .iter()
                            .filter(|(tau, _)| tau.valuation() == Some(0))
                            .map(|(_, i)| *i)
                            .collect();
                        saw_drop |= raw.iter().any(|(tau, _)| tau.valuation() != Some(0));

                        let mut result = FpVector::new(TWO, alg.dimension(t));
                        alg.multiply_basis_elements(result.as_slice_mut(), 1, t1, idx1, t2, idx2);
                        let got: BTreeSet<usize> = result.iter_nonzero().map(|(i, _)| i).collect();
                        assert_eq!(got, expected, "mod-τ product ≠ τ^0 part at ({t1},{idx1})·({t2},{idx2})");
                    }
                }
            }
        }
        assert!(saw_drop, "no product in range dropped a τ-divisible term");
    }

    #[test]
    fn test_ctau_unit_acts_trivially() {
        // 1 · x = x for the unit in degree 0.
        let alg = CTauAlgebra::new();
        alg.compute_basis(3);
        for t in 0..=3 {
            for idx in 0..alg.dimension(t) {
                let mut result = FpVector::new(TWO, alg.dimension(t));
                alg.multiply_basis_elements(result.as_slice_mut(), 1, 0, 0, t, idx);
                let mut expected = FpVector::new(TWO, alg.dimension(t));
                expected.add_basis_element(idx, 1);
                assert_eq!(result, expected, "unit did not act as identity on ({t}, {idx})");
            }
        }
    }
}
