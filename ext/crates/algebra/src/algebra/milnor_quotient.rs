use fp::{
    prime::{Prime, ValidPrime},
    vector::{FpSliceMut, FpVector},
};
use once::OnceVec;

use crate::algebra::{Algebra, Bialgebra, MilnorAlgebra};

/// The quotient Hopf algebra of the Milnor algebra dual to a monomial sub-Hopf algebra of the
/// polynomial part of the dual, $\mathbb{F}_p[\xi_k^{p^{e_k}}]$, with $\xi_k$ left out where $e_k$
/// is `None`.
///
/// The Milnor basis is dual to the monomials, so the kernel of the quotient map is spanned by the
/// $Q(E) P(R)$ with $E$ nonempty or $\xi^R$ outside the subalgebra. The quotient has the
/// remaining Milnor basis elements, and its product is the Milnor product with the other terms
/// dropped. Its Ext is the Cotor of the subalgebra; with every $e_k = 1$, for example, it is the
/// polynomial part of the Milnor algebra with degrees multiplied by $p$.
pub struct MilnorQuotientAlgebra {
    inner: MilnorAlgebra,
    /// $e_k$ for $k = 1, 2, \dots$; generators past the end are left out.
    exponents: Vec<Option<u32>>,
    /// The index in the Milnor algebra of each basis element, by degree.
    to_inner: OnceVec<Vec<usize>>,
    /// The index here of each Milnor basis element, if it is kept, by degree.
    from_inner: OnceVec<Vec<Option<usize>>>,
}

impl MilnorQuotientAlgebra {
    /// The quotient of `inner` dual to $\mathbb{F}_p[\xi_k^{p^{e_k}}]$, $e_k$ = `exponents[k - 1]`.
    ///
    /// The exponents must make the subalgebra closed under the coproduct
    /// $\Delta\xi_k = \sum_i \xi_{k-i}^{p^i} \otimes \xi_i$: if $\xi_k$ is kept, so is every
    /// $\xi_{k-i}$, with $e_{k-i} \leq e_k + i$, and so is every $\xi_i$, with $e_i \leq e_k$.
    pub fn new(inner: MilnorAlgebra, exponents: Vec<Option<u32>>) -> Self {
        for (k, &e) in exponents.iter().enumerate() {
            let Some(e) = e else { continue };
            for (i, &lower) in exponents.iter().enumerate().take(k) {
                let lower = lower.expect("a kept xi_k needs every xi_i below it");
                assert!(
                    lower <= e + (k - i) as u32 && lower <= e,
                    "the exponents do not give a sub-Hopf algebra at xi_{}",
                    k + 1
                );
            }
        }
        Self {
            inner,
            exponents,
            to_inner: OnceVec::new(),
            from_inner: OnceVec::new(),
        }
    }

    /// Whether the Milnor basis element at `(degree, idx)` survives to the quotient.
    fn admissible(&self, degree: i32, idx: usize) -> bool {
        let p = self.prime().as_u32();
        let elt = self.inner.basis_element_from_index(degree, idx);
        elt.q_part == 0
            && elt.p_part.iter().enumerate().all(|(k, r)| {
                r == 0
                    || self
                        .exponents
                        .get(k)
                        .copied()
                        .flatten()
                        .is_some_and(|e| r.is_multiple_of(p.pow(e)))
            })
    }

    /// The Milnor algebra this is a quotient of.
    pub fn inner(&self) -> &MilnorAlgebra {
        &self.inner
    }
}

impl std::fmt::Display for MilnorQuotientAlgebra {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "quotient of {} by exponents {:?}",
            self.inner, self.exponents
        )
    }
}

impl Algebra for MilnorQuotientAlgebra {
    fn prime(&self) -> ValidPrime {
        self.inner.prime()
    }

    fn compute_basis(&self, degree: i32) {
        self.inner.compute_basis(degree);
        self.to_inner.extend(degree.max(-1) as usize, |d| {
            let d = d as i32;
            (0..self.inner.dimension(d))
                .filter(|&i| self.admissible(d, i))
                .collect()
        });
        self.from_inner.extend(degree.max(-1) as usize, |d| {
            let mut out = vec![None; self.inner.dimension(d as i32)];
            for (here, &there) in self.to_inner[d].iter().enumerate() {
                out[there] = Some(here);
            }
            out
        });
    }

    fn dimension(&self, degree: i32) -> usize {
        if degree < 0 {
            return 0;
        }
        self.to_inner[degree as usize].len()
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
        let degree = r_degree + s_degree;
        let mut full = FpVector::new(self.prime(), self.inner.dimension(degree));
        self.inner.multiply_basis_elements(
            full.as_slice_mut(),
            coeff,
            r_degree,
            self.to_inner[r_degree as usize][r_idx],
            s_degree,
            self.to_inner[s_degree as usize][s_idx],
        );
        let from = &self.from_inner[degree as usize];
        for (i, c) in full.iter_nonzero() {
            if let Some(j) = from[i] {
                result.add_basis_element(j, c);
            }
        }
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        self.inner
            .basis_element_to_string(degree, self.to_inner[degree as usize][idx])
    }

    fn basis_element_from_string(&self, elt: &str) -> Option<(i32, usize)> {
        let (d, i) = self.inner.basis_element_from_string(elt)?;
        self.compute_basis(d);
        Some((d, self.from_inner[d as usize][i]?))
    }
}

/// The coproduct of the Milnor algebra with the terms outside the quotient dropped, dual to the
/// product of the subalgebra.
impl Bialgebra for MilnorQuotientAlgebra {
    fn coproduct(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize, i32, usize, u32)> {
        self.inner
            .coproduct(op_deg, self.to_inner[op_deg as usize][op_idx])
            .into_iter()
            .filter_map(|(ld, li, rd, ri, c)| {
                Some((
                    ld,
                    self.from_inner[ld as usize][li]?,
                    rd,
                    self.from_inner[rd as usize][ri]?,
                    c,
                ))
            })
            .collect()
    }

    fn decompose(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize)> {
        vec![(op_deg, op_idx)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quotient is associative: the kernel of the projection is an ideal.
    #[test]
    fn quotients_are_associative() {
        for exponents in [
            vec![Some(1), Some(1), Some(1), Some(1)],
            vec![Some(0), Some(0)],
            vec![Some(0), Some(1), Some(2)],
        ] {
            let q = MilnorQuotientAlgebra::new(
                MilnorAlgebra::new(ValidPrime::new(2), false),
                exponents.clone(),
            );
            let max = 24;
            q.compute_basis(max);
            let p = q.prime();
            for a in 0..=max {
                for b in 0..=max - a {
                    for c in 0..=max - a - b {
                        for ia in 0..q.dimension(a) {
                            for ib in 0..q.dimension(b) {
                                let mut ab = FpVector::new(p, q.dimension(a + b));
                                q.multiply_basis_elements(ab.as_slice_mut(), 1, a, ia, b, ib);
                                for ic in 0..q.dimension(c) {
                                    let mut bc = FpVector::new(p, q.dimension(b + c));
                                    q.multiply_basis_elements(bc.as_slice_mut(), 1, b, ib, c, ic);
                                    let mut lhs = FpVector::new(p, q.dimension(a + b + c));
                                    let mut rhs = FpVector::new(p, q.dimension(a + b + c));
                                    q.multiply_element_by_basis_element(
                                        lhs.as_slice_mut(),
                                        1,
                                        a + b,
                                        ab.as_slice(),
                                        c,
                                        ic,
                                    );
                                    q.multiply_basis_element_by_element(
                                        rhs.as_slice_mut(),
                                        1,
                                        a,
                                        ia,
                                        b + c,
                                        bc.as_slice(),
                                    );
                                    assert_eq!(lhs, rhs, "{exponents:?}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
