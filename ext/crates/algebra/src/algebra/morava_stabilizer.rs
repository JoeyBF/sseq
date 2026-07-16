//! The associated graded of the Morava stabilizer algebra, `gr S(n) = u(L(n))`.
//!
//! This is the algebraic input to `K(n)`-local (chromatic) homotopy: the `E_1` page of the May
//! spectral sequence computing `H^*(S_n)`, the continuous cohomology of the height-`n` Morava
//! stabilizer group. See `ext/docs/chromatic-field-trick.md` for the wider context and
//! Ravenel's *Complex Cobordism and Stable Homotopy Groups of Spheres* ch. 6 for the mathematics.
//!
//! # Height 1 at `p = 2` (the only case implemented so far)
//!
//! At height 1 the Morava stabilizer group `S_1 = Z_p^x` is abelian, so its Lie algebra `L(1)` is
//! abelian. Passing to the associated graded of the augmentation-ideal filtration kills both the
//! restriction (`p`-th power) map and every cross term of the coproduct, leaving a *primitively
//! generated* exterior Hopf algebra
//!
//! ```text
//! gr S(1) = Lambda(x_1, x_2, x_3, ...),   |x_i| = 2 (2^i - 1) = 2, 6, 14, 30, ...
//! ```
//!
//! with each `x_i` primitive (`Delta(x_i) = x_i (x) 1 + 1 (x) x_i`), all brackets zero and every
//! `x_i^2 = 0`. Consequently
//!
//! ```text
//! Ext_{gr S(1)}(F_2, F_2) = F_2[h_1, h_2, h_3, ...],   h_i in (s, t) = (1, 2 (2^i - 1)).
//! ```
//!
//! This polynomial algebra is the algebraic `E_1` (May) page for the height-1 stabilizer; the May
//! differentials that cut it down to the genuine `H^*(S_1)` are a separate computation and are not
//! performed here.
//!
//! ## Grading convention
//!
//! We grade `x_i` by the topological degree `2 (2^i - 1)` inherited from `t_i in BP_* BP`. This is a
//! genuine connected, finite-type `Z`-grading (the degrees are distinct and increasing), directly
//! analogous to the topological grading used elsewhere in this crate. Ravenel additionally records a
//! May *weight* for each generator; that finer bigrading is not needed to pin down the additive
//! answer and is omitted.
//!
//! ## Representation
//!
//! A monomial in the exterior algebra is squarefree, so it is exactly a finite subset of the
//! generator indices `{1, 2, 3, ...}`. We encode such a subset as a `u64` bitmask, with bit `i - 1`
//! set iff `x_i` divides the monomial.

use std::collections::HashMap;

use fp::{
    prime::{Prime, ValidPrime},
    vector::FpSliceMut,
};
use once::OnceVec;

use crate::algebra::{Algebra, Bialgebra};

/// `gr S(n) = u(L(n))`, the associated graded of the Morava stabilizer algebra.
///
/// Only height `n = 1` at `p = 2` is currently implemented; see the module documentation.
pub struct MoravaStabilizerAlgebra {
    prime: ValidPrime,
    height: u32,
    /// `basis_table[d]` lists the basis elements of internal degree `d`, each encoded as a bitmask
    /// over the exterior generators. Indexed by degree.
    basis_table: OnceVec<Vec<u64>>,
    /// `index_table[d]` maps a basis element's bitmask to its index within degree `d`.
    index_table: OnceVec<HashMap<u64, usize>>,
}

impl MoravaStabilizerAlgebra {
    /// Creates `gr S(height)` over `p`. Currently panics unless `p = 2` and `height = 1`.
    pub fn new(prime: ValidPrime, height: u32) -> Self {
        assert_eq!(
            prime.as_u32(),
            2,
            "gr S(n) is currently only implemented at p = 2"
        );
        assert_eq!(
            height, 1,
            "gr S(n) is currently only implemented at height 1"
        );
        Self {
            prime,
            height,
            basis_table: OnceVec::new(),
            index_table: OnceVec::new(),
        }
    }

    /// The internal degree of the exterior generator `x_i` (for `i >= 1`): `2 (2^i - 1)`.
    fn generator_degree(i: u32) -> i32 {
        debug_assert!((1..=62).contains(&i), "generator index out of range");
        (2 * ((1i64 << i) - 1)) as i32
    }

    /// The total internal degree of the monomial encoded by `mask`.
    fn mask_degree(mask: u64) -> i32 {
        let mut degree = 0;
        let mut m = mask;
        while m != 0 {
            let bit = m.trailing_zeros();
            degree += Self::generator_degree(bit + 1);
            m &= m - 1;
        }
        degree
    }

    /// Enumerates every squarefree monomial (bitmask) of total degree exactly `degree`.
    fn monomials_of_degree(degree: i32) -> Vec<u64> {
        fn recurse(i: u32, remaining: i32, mask: u64, out: &mut Vec<u64>) {
            if remaining == 0 {
                out.push(mask);
                return;
            }
            let gen_deg = MoravaStabilizerAlgebra::generator_degree(i);
            // Generator degrees strictly increase in `i`, so once `x_i` overshoots the remaining
            // degree no later generator can contribute either.
            if gen_deg > remaining {
                return;
            }
            // Include `x_i`, then exclude it.
            recurse(i + 1, remaining - gen_deg, mask | (1u64 << (i - 1)), out);
            recurse(i + 1, remaining, mask, out);
        }

        let mut out = Vec::new();
        if degree >= 0 {
            recurse(1, degree, 0, &mut out);
        }
        // Sort for a deterministic, stable basis order (indices must be reproducible).
        out.sort_unstable();
        out
    }

    /// The index of the basis element encoded by `mask` within its degree.
    fn index_of(&self, mask: u64) -> usize {
        let degree = Self::mask_degree(mask);
        self.compute_basis(degree);
        self.index_table[degree as usize][&mask]
    }

    /// The `(degree, index)` of the primitive generator `x_i`.
    fn generator(&self, i: u32) -> (i32, usize) {
        let degree = Self::generator_degree(i);
        (degree, self.index_of(1u64 << (i - 1)))
    }
}

impl std::fmt::Display for MoravaStabilizerAlgebra {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "gr S({}) at p = {}", self.height, self.prime)
    }
}

impl Algebra for MoravaStabilizerAlgebra {
    fn prefix(&self) -> &str {
        "grmorava"
    }

    fn magic(&self) -> u32 {
        // Distinct from the Steenrod magics so save files cannot be confused across algebras.
        0x6d6f7231 // "mor1"
    }

    fn prime(&self) -> ValidPrime {
        self.prime
    }

    fn compute_basis(&self, max_degree: i32) {
        if max_degree < 0 {
            return;
        }
        let target = max_degree as usize;
        self.basis_table
            .extend(target, |d| Self::monomials_of_degree(d as i32));
        self.index_table.extend(target, |d| {
            self.basis_table[d]
                .iter()
                .enumerate()
                .map(|(idx, &mask)| (mask, idx))
                .collect()
        });
    }

    fn dimension(&self, degree: i32) -> usize {
        if degree < 0 {
            0
        } else {
            self.basis_table[degree as usize].len()
        }
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
        let left = self.basis_table[r_degree as usize][r_idx];
        let right = self.basis_table[s_degree as usize][s_idx];
        // Exterior product: a repeated generator squares to zero.
        if left & right != 0 {
            return;
        }
        let product = left | right;
        let target_degree = r_degree + s_degree;
        self.compute_basis(target_degree);
        let idx = self.index_table[target_degree as usize][&product];
        result.add_basis_element(idx, coeff);
    }

    fn default_filtration_one_products(&self) -> Vec<(String, i32, usize)> {
        // The filtration-one classes h_i in Ext^{1, |x_i|} are dual to the primitive generators x_i.
        // We list a modest range; the generators continue for all i >= 1.
        (1..=6)
            .map(|i| {
                let (degree, idx) = self.generator(i);
                (format!("h_{i}"), degree, idx)
            })
            .collect()
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        let mask = self.basis_table[degree as usize][idx];
        if mask == 0 {
            return "1".to_string();
        }
        let mut factors = Vec::new();
        let mut m = mask;
        while m != 0 {
            let bit = m.trailing_zeros();
            factors.push(format!("x_{}", bit + 1));
            m &= m - 1;
        }
        factors.join(" ")
    }

    fn basis_element_from_string(&self, elt: &str) -> Option<(i32, usize)> {
        let trimmed = elt.trim();
        if trimmed.is_empty() || trimmed == "1" {
            return Some((0, 0));
        }
        let mut mask = 0u64;
        for token in trimmed.split_whitespace() {
            let i: u32 = token.strip_prefix("x_")?.parse().ok()?;
            if !(1..=62).contains(&i) {
                return None;
            }
            let bit = 1u64 << (i - 1);
            if mask & bit != 0 {
                return None; // x_i^2 = 0 is not a basis element
            }
            mask |= bit;
        }
        let degree = Self::mask_degree(mask);
        self.compute_basis(degree);
        let idx = *self.index_table[degree as usize].get(&mask)?;
        Some((degree, idx))
    }
}

impl Bialgebra for MoravaStabilizerAlgebra {
    fn coproduct(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize, i32, usize)> {
        // gr S(1) is primitively generated, so for a monomial x_S the coproduct is
        //   Delta(x_S) = sum_{T subset S} x_T (x) x_{S \ T},
        // all coefficients 1 at p = 2. We iterate over sub-bitmasks T of the monomial's mask.
        let mask = self.basis_table[op_deg as usize][op_idx];
        let mut terms = Vec::new();
        let mut t = mask;
        loop {
            let left = t;
            let right = mask ^ t;
            let left_deg = Self::mask_degree(left);
            let right_deg = op_deg - left_deg;
            let left_idx = self.index_table[left_deg as usize][&left];
            let right_idx = self.index_table[right_deg as usize][&right];
            terms.push((left_deg, left_idx, right_deg, right_idx));
            if t == 0 {
                break;
            }
            t = (t - 1) & mask;
        }
        terms
    }

    fn decompose(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize)> {
        // The full coproduct above is available directly on any basis element, so each element is
        // its own single "atom" (mirroring the Milnor algebra).
        vec![(op_deg, op_idx)]
    }
}

#[cfg(test)]
mod tests {
    use fp::{prime::TWO, vector::FpVector};

    use super::*;

    fn algebra() -> MoravaStabilizerAlgebra {
        MoravaStabilizerAlgebra::new(TWO, 1)
    }

    #[test]
    fn dimensions() {
        let a = algebra();
        a.compute_basis(16);
        // Generators x_1, x_2, x_3 sit in degrees 2, 6, 14. The squarefree monomials are:
        //   deg 0: 1;  deg 2: x_1;  deg 6: x_2;  deg 8: x_1 x_2;  deg 14: x_3;  deg 16: x_1 x_3.
        // Degrees with no monomial are 0-dimensional.
        assert_eq!(a.dimension(0), 1);
        assert_eq!(a.dimension(2), 1);
        assert_eq!(a.dimension(4), 0);
        assert_eq!(a.dimension(6), 1);
        assert_eq!(a.dimension(8), 1);
        assert_eq!(a.dimension(10), 0);
        assert_eq!(a.dimension(14), 1);
        assert_eq!(a.dimension(16), 1);
    }

    #[test]
    fn strings_round_trip() {
        let a = algebra();
        a.compute_basis(8);
        assert_eq!(a.basis_element_to_string(0, 0), "1");
        assert_eq!(a.basis_element_to_string(2, 0), "x_1");
        assert_eq!(a.basis_element_to_string(8, 0), "x_1 x_2");

        assert_eq!(a.basis_element_from_string("1"), Some((0, 0)));
        assert_eq!(a.basis_element_from_string("x_1"), Some((2, 0)));
        assert_eq!(a.basis_element_from_string("x_1 x_2"), Some((8, 0)));
    }

    #[test]
    fn exterior_multiplication() {
        let a = algebra();
        a.compute_basis(8);

        // x_1 * x_1 = 0 (exterior square).
        let mut result = FpVector::new(TWO, a.dimension(4));
        a.multiply_basis_elements(result.as_slice_mut(), 1, 2, 0, 2, 0);
        assert!(result.is_zero());

        // x_1 * x_2 = x_1 x_2 (the unique basis element of degree 8).
        let mut result = FpVector::new(TWO, a.dimension(8));
        a.multiply_basis_elements(result.as_slice_mut(), 1, 2, 0, 6, 0);
        assert_eq!(result.entry(0), 1);

        // x_2 * x_1 = x_1 x_2 as well (commutative at p = 2).
        let mut result = FpVector::new(TWO, a.dimension(8));
        a.multiply_basis_elements(result.as_slice_mut(), 1, 6, 0, 2, 0);
        assert_eq!(result.entry(0), 1);
    }

    #[test]
    fn primitive_and_full_coproduct() {
        let a = algebra();
        a.compute_basis(8);

        // x_1 is primitive: Delta(x_1) = x_1 (x) 1 + 1 (x) x_1.
        let mut coprod = a.coproduct(2, 0);
        coprod.sort_unstable();
        assert_eq!(coprod, vec![(0, 0, 2, 0), (2, 0, 0, 0)]);

        // Delta(x_1 x_2) = sum over subsets: 1(x)x_1x_2 + x_1(x)x_2 + x_2(x)x_1 + x_1x_2(x)1.
        let mut coprod = a.coproduct(8, 0);
        coprod.sort_unstable();
        let mut expected = vec![(0, 0, 8, 0), (2, 0, 6, 0), (6, 0, 2, 0), (8, 0, 0, 0)];
        expected.sort_unstable();
        assert_eq!(coprod, expected);
    }
}
