//! Primary Massey products in $\Ext$.
//!
//! [`ExtAlgebra::massey`] computes a triple Massey product $\langle a, b, c\rangle$, and
//! [`ExtAlgebra::massey_family`] computes $\langle a, b, -\rangle$ for fixed $a, b$ and every valid
//! third factor at once (the optimized form). Both wrap [`ChainHomotopy`]: for each candidate `c`
//! we lift the multiplication map, build the null-homotopy of the composite with `b`, and read off
//! the bracket by pairing with `a`. The valid `c` (those with `b · c = 0`) are exactly the kernel
//! of multiplication by `b`.
//!
//! The result is a coset representative together with the indeterminacy
//! $a \cdot \Ext + \Ext \cdot c$. The indeterminacy is computed when `M == k` (so both terms live
//! in the same algebra); otherwise it is reported as the zero subspace. This matches (and reuses
//! the logic of) the `massey` example, which computes the products up to a sign.

use std::sync::Arc;

use fp::{
    matrix::{AugmentedMatrix, Matrix, Subspace},
    vector::{FpSlice, FpVector},
};
use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

use super::ExtAlgebra;
use crate::{
    chain_complex::{AugmentedChainComplex, ChainHomotopy, FreeChainComplex},
    resolution_homomorphism::ResolutionHomomorphism,
};

/// The result of a Massey product computation: a coset representative together with the
/// indeterminacy subspace.
pub struct MasseyResult {
    /// A representative of the Massey product, in bidegree
    /// `a.degree() + b.degree() + c.degree() - (1, 0)`.
    pub representative: BidegreeElement,
    /// The indeterminacy $a \cdot \Ext + \Ext \cdot c$, as a subspace of the bracket's bidegree.
    /// Only populated when `M == k`; the zero subspace otherwise.
    pub indeterminacy: Subspace,
}

impl<CC> ExtAlgebra<CC>
where
    CC: FreeChainComplex + AugmentedChainComplex,
{
    /// The bidegree shift of $\langle a, b, -\rangle$: a class `c` produces a bracket in bidegree
    /// `c.degree() + a.degree() + b.degree() - (1, 0)`.
    fn massey_shift(a: &BidegreeElement, b: &BidegreeElement) -> Bidegree {
        a.degree() + b.degree() - Bidegree::s_t(1, 0)
    }

    /// The multiplication-by-`b` chain map (in the unit), extended far enough for brackets landing
    /// at `shift`.
    fn massey_b_hom(
        &self,
        b: &BidegreeElement,
        shift: Bidegree,
    ) -> Arc<ResolutionHomomorphism<CC, CC>> {
        let b_coords: Vec<u32> = b.vec().iter().collect();
        let hom = Arc::new(ResolutionHomomorphism::from_class(
            String::new(),
            Arc::clone(self.unit()),
            Arc::clone(self.unit()),
            b.degree(),
            &b_coords,
        ));
        hom.extend_through_stem(shift);
        hom
    }

    /// Compute, for a single multiplicand bidegree `c_deg`, the per-generator bracket values
    /// `answers[gen][i]` and the kernel of multiplication by `b` (the valid third factors). Returns
    /// `None` if the bracket bidegree is empty or uncomputed.
    fn massey_at(
        &self,
        a: &BidegreeElement,
        b: &BidegreeElement,
        b_hom: &Arc<ResolutionHomomorphism<CC, CC>>,
        shift: Bidegree,
        offset_a: usize,
        c_deg: Bidegree,
    ) -> Option<(Vec<Vec<u32>>, Subspace, Bidegree)> {
        let p = self.prime();
        let resolution = self.resolution();
        let unit = self.unit();

        if !resolution.has_computed_bidegree(c_deg + shift) {
            return None;
        }
        let tot = c_deg + shift;

        let num_gens = resolution.number_of_gens_in_bidegree(c_deg);
        let product_num_gens = resolution.number_of_gens_in_bidegree(b.degree() + c_deg);
        let target_num_gens = resolution.number_of_gens_in_bidegree(tot);
        if target_num_gens == 0 {
            return None;
        }

        let a_coords: Vec<u32> = a.vec().iter().collect();
        let b_coords: Vec<u32> = b.vec().iter().collect();

        let mut answers = vec![vec![0u32; target_num_gens]; num_gens];
        let mut product = AugmentedMatrix::<2>::new(p, num_gens, [product_num_gens, num_gens]);
        product.segment(1, 1).add_identity();

        let mut matrix = Matrix::new(p, num_gens, 1);
        for (idx, answer_row) in answers.iter_mut().enumerate() {
            let hom = Arc::new(ResolutionHomomorphism::new(
                String::new(),
                Arc::clone(resolution),
                Arc::clone(unit),
                c_deg,
            ));

            matrix.row_mut(idx).set_entry(0, 1);
            hom.extend_step(c_deg, Some(&matrix));
            matrix.row_mut(idx).set_entry(0, 0);

            hom.extend_through_stem(tot);

            let homotopy = ChainHomotopy::new(Arc::clone(&hom), Arc::clone(b_hom));
            homotopy.extend(tot);

            let last = homotopy.homotopy(tot.s());
            for (i, answer) in answer_row.iter_mut().enumerate() {
                let output = last.output(tot.t(), i);
                for (k, &val) in a_coords.iter().enumerate() {
                    if val != 0 {
                        *answer += val * output.entry(offset_a + k);
                    }
                }
            }

            for (k, &val) in b_coords.iter().enumerate() {
                if val != 0 {
                    let g = BidegreeGenerator::new(b.degree(), k);
                    hom.act(product.row_mut(idx).slice_mut(0, product_num_gens), val, g);
                }
            }
        }
        product.row_reduce();
        let kernel = product.compute_kernel();

        Some((answers, kernel, tot))
    }

    /// Contract the per-generator bracket values `answers` against the coordinates `row` of a
    /// third factor, producing the bracket representative at `tot`.
    fn massey_contract(
        &self,
        answers: &[Vec<u32>],
        row: FpSlice,
        tot: Bidegree,
    ) -> BidegreeElement {
        let p = self.prime();
        let target_num_gens = answers.first().map_or(0, Vec::len);
        let mut v = FpVector::new(p, target_num_gens);
        for (j, val) in row.iter().enumerate() {
            if val != 0 {
                for (i, a) in answers[j].iter().enumerate() {
                    v.add_basis_element(i, val * a);
                }
            }
        }
        BidegreeElement::new(tot, v)
    }

    /// The indeterminacy $a \cdot \Ext^{|b| + |c| - (1,0)} + \Ext^{|a| + |b| - (1,0)} \cdot c$ at
    /// the bracket bidegree `tot`. Computed only when `M == k`; otherwise the zero subspace.
    fn massey_indeterminacy(
        &self,
        a: &BidegreeElement,
        c: &BidegreeElement,
        tot: Bidegree,
    ) -> Subspace {
        let mut sub = Subspace::new(self.prime(), self.dimension(tot));
        if !self.is_unit() {
            return sub;
        }

        // a · Ext^{tot - a.degree()}
        for y in self.basis(tot - a.degree()) {
            if let Some(prod) = self.try_multiply(a, &self.generator(y)) {
                sub.add_vector(prod.vec());
            }
        }
        // Ext^{tot - c.degree()} · c
        for x in self.basis(tot - c.degree()) {
            if let Some(prod) = self.try_multiply(&self.generator(x), c) {
                sub.add_vector(prod.vec());
            }
        }
        sub
    }

    /// Compute the family of Massey products $\langle a, b, -\rangle$ for fixed `a` and `b` and
    /// every valid third factor `c` (those with `b · c = 0`), across all computed bidegrees.
    ///
    /// `a` and `b` are taken in $\Ext(k, k)$; the third factor ranges over $\Ext(M, k)$. The
    /// caller must have resolved `M` and the unit far enough. This assumes `a · b = 0` (so that the
    /// bracket is defined); it is not verified.
    pub fn massey_family(
        &self,
        a: &BidegreeElement,
        b: &BidegreeElement,
    ) -> Vec<(BidegreeElement, MasseyResult)> {
        let shift = Self::massey_shift(a, b);
        let offset_a =
            self.unit()
                .module(a.degree().s())
                .generator_offset(a.degree().t(), a.degree().t(), 0);
        let b_hom = self.massey_b_hom(b, shift);

        let mut results = Vec::new();
        for c_deg in self.resolution().iter_nonzero_stem() {
            let Some((answers, kernel, tot)) = self.massey_at(a, b, &b_hom, shift, offset_a, c_deg)
            else {
                continue;
            };
            for row in kernel.iter() {
                let c = BidegreeElement::new(c_deg, row.to_owned());
                let representative = self.massey_contract(&answers, row, tot);
                let indeterminacy = self.massey_indeterminacy(a, &c, tot);
                results.push((
                    c,
                    MasseyResult {
                        representative,
                        indeterminacy,
                    },
                ));
            }
        }
        results
    }

    /// Compute the triple Massey product $\langle a, b, c\rangle$.
    ///
    /// `a` and `b` are taken in $\Ext(k, k)$ and `c` in $\Ext(M, k)$. Returns `None` if
    /// `b · c != 0` (so the bracket is undefined). This assumes `a · b = 0`; it is not verified.
    pub fn massey(
        &self,
        a: &BidegreeElement,
        b: &BidegreeElement,
        c: &BidegreeElement,
    ) -> Option<MasseyResult> {
        let shift = Self::massey_shift(a, b);
        let offset_a =
            self.unit()
                .module(a.degree().s())
                .generator_offset(a.degree().t(), a.degree().t(), 0);
        let b_hom = self.massey_b_hom(b, shift);

        let (answers, kernel, tot) = self.massey_at(a, b, &b_hom, shift, offset_a, c.degree())?;

        // The bracket is defined exactly when b · c = 0, i.e. c lies in the kernel of (· b).
        let mut reduced = c.vec().to_owned();
        kernel.reduce(reduced.as_slice_mut());
        if !reduced.is_zero() {
            return None;
        }

        let representative = self.massey_contract(&answers, c.vec(), tot);
        let indeterminacy = self.massey_indeterminacy(a, c, tot);
        Some(MasseyResult {
            representative,
            indeterminacy,
        })
    }
}

#[cfg(test)]
mod tests {
    use sseq::coordinates::BidegreeGenerator;

    use super::*;
    use crate::utils::construct_standard;

    #[test]
    fn test_sphere_massey() {
        let res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        res.compute_through_stem(Bidegree::n_s(6, 5));
        let alg = ExtAlgebra::new(Arc::clone(&res), res, true);

        let h0 = alg.generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));
        let h1 = alg.generator(BidegreeGenerator::new(Bidegree::n_s(1, 1), 0));

        // The classic relation <h0, h1, h0> = h1^2, the generator of Ext^{2,4} at (2, 2).
        let bracket = alg
            .massey(&h0, &h1, &h0)
            .expect("<h0, h1, h0> should be defined");
        assert_eq!(bracket.representative.degree(), Bidegree::n_s(2, 2));
        assert_eq!(alg.dimension(Bidegree::n_s(2, 2)), 1);
        assert!(
            !bracket.representative.vec().is_zero(),
            "<h0, h1, h0> = h1^2 should be nonzero"
        );

        let h1_sq = alg.multiply(&h1, &h1);
        assert_eq!(
            bracket.representative.vec().iter().collect::<Vec<_>>(),
            h1_sq.vec().iter().collect::<Vec<_>>(),
            "<h0, h1, h0> should equal h1^2"
        );
        // The indeterminacy a·Ext + Ext·c vanishes here, so the bracket is a single class.
        assert_eq!(bracket.indeterminacy.dimension(), 0);

        // <h0, h1, h1> is undefined: h1 · h1 = h1^2 != 0, so h1 is not a valid third factor.
        assert!(
            alg.massey(&h0, &h1, &h1).is_none(),
            "<h0, h1, h1> should be undefined since h1^2 != 0"
        );
    }
}
