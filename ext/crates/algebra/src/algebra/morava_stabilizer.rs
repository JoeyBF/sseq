//! The associated graded of the Morava stabilizer algebra, `gr S(n) = u(L(n)) = E_0 S(n)^*`.
//!
//! This is the algebraic input to `K(n)`-local (chromatic) homotopy: resolving `F_p` over it
//! computes `Cotor_{E_0 S(n)}(F_p, F_p) = H^*(L(n))`, the `E_1` page of Ravenel's May spectral
//! sequence for `H^*(S(n))`, the continuous cohomology of the height-`n` Morava stabilizer group.
//! See `ext/docs/chromatic-computations.md` for context and Ravenel, *Complex Cobordism and Stable
//! Homotopy Groups of Spheres*, ch. 6 (green book) for the mathematics.
//!
//! # Height 1 at `p = 2` (the only case implemented so far)
//!
//! At height 1 the group `S_1 = Z_p^x` is abelian, so the restricted Lie algebra `L(1)` is abelian
//! (green book Thm 6.3.3: the bracket `[x_{i,0}, x_{k,0}]` vanishes because both Kronecker deltas
//! coincide). The restriction (`p`-th power) map at `n = 1, p = 2` is
//!
//! ```text
//! xi(x_1) = 0,        xi(x_i) = x_{i+1}   for i >= 2
//! ```
//!
//! (Thm 6.3.3, using `xi(x_1) = x_{2,0} + x_{2,1} = 2 x_2 = 0`). Hence the restricted enveloping
//! algebra collapses to a tensor product of an exterior and a polynomial factor:
//!
//! ```text
//! gr S(1) = V(L(1)) = Lambda(x_1) (x) F_2[x_2],
//! ```
//!
//! where `x_1` is exterior (`x_1^2 = xi(x_1) = 0`) and `x_2` is a polynomial generator (its powers
//! `x_2, x_2^2 = x_3, x_2^4 = x_4, ...` are the higher Lie generators, never truncating). Both are
//! primitive. We grade them by the topological internal degrees of the cohomology classes they carry
//! (green book 6.3.11, where `zeta_1 = h_{1,0} = [t_1]` and `rho_1 = h_{2,0} = [t_2]`):
//!
//! ```text
//! |x_1| = 2 (2^1 - 1) = 2,    |x_2| = 2 (2^2 - 1) = 6.
//! ```
//!
//! Resolving `F_2` therefore yields (Ravenel, Thm 6.3.21(a)):
//!
//! ```text
//! Ext_{gr S(1)}(F_2, F_2) = P(h_{1,0}) (x) E(rho_1) = F_2[h_1] (x) Lambda(rho_1),
//! ```
//!
//! with `h_1 in (s, t) = (1, 2)` (the exterior generator `x_1` dualizes to a *polynomial* class) and
//! `rho_1 in (s, t) = (1, 6)` (the polynomial generator `x_2` dualizes to an *exterior* class). At
//! height 1 the May spectral sequence collapses, so this is already the genuine `H^*(S(1))`.
//!
//! ## Grading pitfall (why this is not the exterior algebra on all the `x_i`)
//!
//! It is tempting to treat every Lie generator `x_i` (`i >= 1`, degree `2(2^i - 1)`) as an
//! independent exterior generator. That is wrong: it silently drops the restriction and the
//! coproduct cross terms (which are inhomogeneous in the topological grading) and produces the
//! degenerate answer `F_2[h_1, h_2, ...]`. The restriction `xi(x_i) = x_{i+1}` forces `|x_{i+1}| =
//! 2 |x_i|`, so for `i >= 2` the `x_i` are the powers of the single polynomial generator `x_2`, not
//! independent generators.
//!
//! ## Representation
//!
//! A basis element is a monomial `x_1^{e_1} x_2^{e_2}` with `e_1 in {0, 1}` (exterior) and
//! `e_2 >= 0` (polynomial), stored as the pair `(e_1, e_2)`. Its internal degree is `2 e_1 + 6 e_2`.

use std::collections::HashMap;

use fp::{
    prime::{Prime, ValidPrime},
    vector::FpSliceMut,
};
use once::OnceVec;

use crate::algebra::{Algebra, Bialgebra};

/// The internal degree of the exterior generator `x_1`.
const DEG_X1: i32 = 2;
/// The internal degree of the polynomial generator `x_2`.
const DEG_X2: i32 = 6;

/// `gr S(n) = u(L(n))`, the associated graded of the Morava stabilizer algebra.
///
/// Only height `n = 1` at `p = 2` is currently implemented; see the module documentation.
pub struct MoravaStabilizerAlgebra {
    prime: ValidPrime,
    height: u32,
    /// `basis_table[d]` lists the basis elements of internal degree `d`, each a monomial
    /// `x_1^{e_1} x_2^{e_2}` encoded as `(e_1, e_2)`. Indexed by degree.
    basis_table: OnceVec<Vec<(u32, u32)>>,
    /// `index_table[d]` maps a basis element `(e_1, e_2)` to its index within degree `d`.
    index_table: OnceVec<HashMap<(u32, u32), usize>>,
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

    /// The internal degree of the monomial `(e_1, e_2)`.
    fn monomial_degree(e1: u32, e2: u32) -> i32 {
        e1 as i32 * DEG_X1 + e2 as i32 * DEG_X2
    }

    /// Enumerates the basis of internal degree `degree`: the monomials `x_1^{e_1} x_2^{e_2}` with
    /// `2 e_1 + 6 e_2 = degree`, `e_1 in {0, 1}`, `e_2 >= 0`.
    fn monomials_of_degree(degree: i32) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        if degree < 0 {
            return out;
        }
        for e1 in 0..=1u32 {
            let rest = degree - e1 as i32 * DEG_X1;
            if rest >= 0 && rest % DEG_X2 == 0 {
                out.push((e1, (rest / DEG_X2) as u32));
            }
        }
        // Deterministic order (by e_1 then e_2) so indices are reproducible.
        out.sort_unstable();
        out
    }

    /// `binom(n, k) mod 2`, i.e. 1 iff `k` is a submask of `n` in binary (Lucas' theorem at `p = 2`).
    fn binom_mod2(n: u32, k: u32) -> bool {
        (n & k) == k
    }

    /// The index of the monomial `(e_1, e_2)` within its degree.
    fn index_of(&self, e1: u32, e2: u32) -> usize {
        let degree = Self::monomial_degree(e1, e2);
        self.compute_basis(degree);
        self.index_table[degree as usize][&(e1, e2)]
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
                .map(|(idx, &mono)| (mono, idx))
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
        let (e1, e2) = self.basis_table[r_degree as usize][r_idx];
        let (f1, f2) = self.basis_table[s_degree as usize][s_idx];
        // x_1 is exterior: x_1^2 = 0.
        if e1 + f1 > 1 {
            return;
        }
        let product = (e1 + f1, e2 + f2);
        let target_degree = r_degree + s_degree;
        self.compute_basis(target_degree);
        let idx = self.index_table[target_degree as usize][&product];
        result.add_basis_element(idx, coeff);
    }

    fn default_filtration_one_products(&self) -> Vec<(String, i32, usize)> {
        // The filtration-one classes are dual to the two primitive generators:
        //   h_1 = h_{1,0} in Ext^{1, 2}  (dual to the exterior generator x_1), and
        //   rho_1 = h_{2,0} in Ext^{1, 6}  (dual to the polynomial generator x_2).
        vec![
            ("h_1".to_string(), DEG_X1, self.index_of(1, 0)),
            ("rho_1".to_string(), DEG_X2, self.index_of(0, 1)),
        ]
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        let (e1, e2) = self.basis_table[degree as usize][idx];
        let mut factors = Vec::new();
        if e1 == 1 {
            factors.push("x_1".to_string());
        }
        match e2 {
            0 => {}
            1 => factors.push("x_2".to_string()),
            _ => factors.push(format!("x_2^{e2}")),
        }
        if factors.is_empty() {
            "1".to_string()
        } else {
            factors.join(" ")
        }
    }

    fn basis_element_from_string(&self, elt: &str) -> Option<(i32, usize)> {
        let trimmed = elt.trim();
        if trimmed.is_empty() || trimmed == "1" {
            return Some((0, 0));
        }
        let mut e1 = 0u32;
        let mut e2 = 0u32;
        for token in trimmed.split_whitespace() {
            if token == "x_1" {
                if e1 != 0 {
                    return None; // x_1^2 = 0 is not a basis element
                }
                e1 = 1;
            } else if let Some(rest) = token.strip_prefix("x_2") {
                let power: u32 = if rest.is_empty() {
                    1
                } else {
                    rest.strip_prefix('^')?.parse().ok()?
                };
                e2 = e2.checked_add(power)?;
            } else {
                return None;
            }
        }
        Some((Self::monomial_degree(e1, e2), self.index_of(e1, e2)))
    }
}

impl Bialgebra for MoravaStabilizerAlgebra {
    fn coproduct(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize, i32, usize)> {
        // Both generators are primitive, so the coproduct of x_1^{e_1} x_2^{e_2} splits each factor:
        //   Delta(x_1^{e_1}) = sum_{i<=e_1} C(e_1, i) x_1^i (x) x_1^{e_1-i},
        //   Delta(x_2^{e_2}) = sum_{k<=e_2} C(e_2, k) x_2^k (x) x_2^{e_2-k},
        // with binomial coefficients taken mod 2.
        let (e1, e2) = self.basis_table[op_deg as usize][op_idx];
        let mut terms = Vec::new();
        for i in 0..=e1 {
            if !Self::binom_mod2(e1, i) {
                continue;
            }
            for k in 0..=e2 {
                if !Self::binom_mod2(e2, k) {
                    continue;
                }
                let left = (i, k);
                let right = (e1 - i, e2 - k);
                let left_deg = Self::monomial_degree(left.0, left.1);
                let right_deg = Self::monomial_degree(right.0, right.1);
                let left_idx = self.index_table[left_deg as usize][&left];
                let right_idx = self.index_table[right_deg as usize][&right];
                terms.push((left_deg, left_idx, right_deg, right_idx));
            }
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
        a.compute_basis(20);
        // Basis: x_1^{e_1} x_2^{e_2}, e_1 in {0,1}, e_2 >= 0, degree 2 e_1 + 6 e_2.
        //   deg 0: 1;  deg 2: x_1;  deg 6: x_2;  deg 8: x_1 x_2;  deg 12: x_2^2;  deg 14: x_1 x_2^2.
        // Every occupied degree is 1-dimensional; degrees == 4 mod 6 and odd degrees are empty.
        assert_eq!(a.dimension(0), 1);
        assert_eq!(a.dimension(2), 1);
        assert_eq!(a.dimension(4), 0);
        assert_eq!(a.dimension(6), 1);
        assert_eq!(a.dimension(8), 1);
        assert_eq!(a.dimension(10), 0);
        assert_eq!(a.dimension(12), 1);
        assert_eq!(a.dimension(14), 1);
    }

    #[test]
    fn strings_round_trip() {
        let a = algebra();
        a.compute_basis(14);
        assert_eq!(a.basis_element_to_string(0, 0), "1");
        assert_eq!(a.basis_element_to_string(2, 0), "x_1");
        assert_eq!(a.basis_element_to_string(6, 0), "x_2");
        assert_eq!(a.basis_element_to_string(8, 0), "x_1 x_2");
        assert_eq!(a.basis_element_to_string(14, 0), "x_1 x_2^2");

        assert_eq!(a.basis_element_from_string("1"), Some((0, 0)));
        assert_eq!(a.basis_element_from_string("x_1"), Some((2, 0)));
        assert_eq!(a.basis_element_from_string("x_2"), Some((6, 0)));
        assert_eq!(a.basis_element_from_string("x_1 x_2"), Some((8, 0)));
        assert_eq!(a.basis_element_from_string("x_1 x_2^2"), Some((14, 0)));
    }

    #[test]
    fn multiplication() {
        let a = algebra();
        a.compute_basis(16);

        // x_1 * x_1 = 0 (exterior).
        let mut result = FpVector::new(TWO, a.dimension(4));
        a.multiply_basis_elements(result.as_slice_mut(), 1, 2, 0, 2, 0);
        assert!(result.is_zero());

        // x_2 * x_2 = x_2^2 (polynomial, nonzero): the unique basis element of degree 12.
        let mut result = FpVector::new(TWO, a.dimension(12));
        a.multiply_basis_elements(result.as_slice_mut(), 1, 6, 0, 6, 0);
        assert_eq!(result.entry(0), 1);

        // x_1 * x_2 = x_1 x_2 (degree 8).
        let mut result = FpVector::new(TWO, a.dimension(8));
        a.multiply_basis_elements(result.as_slice_mut(), 1, 2, 0, 6, 0);
        assert_eq!(result.entry(0), 1);
    }

    #[test]
    fn coproducts() {
        let a = algebra();
        a.compute_basis(12);

        // x_1 is primitive.
        let mut coprod = a.coproduct(2, 0);
        coprod.sort_unstable();
        assert_eq!(coprod, vec![(0, 0, 2, 0), (2, 0, 0, 0)]);

        // x_2 is primitive.
        let mut coprod = a.coproduct(6, 0);
        coprod.sort_unstable();
        assert_eq!(coprod, vec![(0, 0, 6, 0), (6, 0, 0, 0)]);

        // Delta(x_2^2) = x_2^2 (x) 1 + 1 (x) x_2^2 (the cross term has C(2,1) = 0 mod 2).
        let mut coprod = a.coproduct(12, 0);
        coprod.sort_unstable();
        assert_eq!(coprod, vec![(0, 0, 12, 0), (12, 0, 0, 0)]);
    }
}
