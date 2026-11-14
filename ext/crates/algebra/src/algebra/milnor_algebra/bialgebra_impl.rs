use fp::prime::{ValidPrime, iter::BitflagIterator, minus_one_to_the_n};

use super::{MilnorAlgebraInner, MilnorBasisElement, MilnorShape, PPart};
use crate::algebra::{Algebra, Bialgebra};

impl<F: MilnorShape> MilnorAlgebraInner<F> {
    /// Advance `element` to the next p-part bounded entrywise by `max`, in odometer order.
    ///
    /// Returns `true` once the odometer wraps, i.e. when `element` was already `max`.
    ///
    /// This carries *before* incrementing rather than after. The two orders enumerate the same
    /// sequence, but incrementing first would transiently store `max[i] + 1`, which need not fit
    /// in a packed field whose width is exactly saturated by `max[i]`.
    fn increment_p_part(element: &mut PPart, max: PPart) -> bool {
        for i in 0..max.len() {
            let entry = element.get(i);
            if entry < max.get(i) {
                element.set(i, entry + 1);
                return false;
            }
            element.set(i, 0);
        }
        true
    }
}

/// The Milnor basis is dual to the monomial basis $\tau^E \xi^R$ of the dual algebra, whose
/// product adds exponents, so the coproduct of $Q(E) P(R)$ is the sum over splittings
/// $E = E' \sqcup E''$ and $R = S + T$ of $Q(E') P(S) \otimes Q(E'') P(T)$, signed by
/// [`shuffle_sign`].
impl<F: MilnorShape> Bialgebra for MilnorAlgebraInner<F> {
    fn coproduct(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize, i32, usize, u32)> {
        let p = self.prime();
        let elt = self.basis_element_from_index(op_deg, op_idx);
        let (p_part, q_part) = (elt.p_part, elt.q_part);
        let len: usize =
            p_part.iter().map(|i| i as usize + 1).product::<usize>() << q_part.count_ones();
        let mut result = Vec::with_capacity(len);

        let mut left_p = PPart::zero();
        loop {
            let mut right_p = PPart::zero();
            for i in 0..p_part.len() {
                right_p.set(i, p_part.get(i) - left_p.get(i));
            }
            // The subsets of `q_part`, by the standard trick of decrementing within a mask.
            let mut left_q = q_part;
            loop {
                let right_q = q_part & !left_q;
                let mut left = MilnorBasisElement {
                    q_part: left_q,
                    p_part: left_p,
                    degree: 0,
                };
                let mut right = MilnorBasisElement {
                    q_part: right_q,
                    p_part: right_p,
                    degree: 0,
                };
                self.compute_degree(&mut left);
                self.compute_degree(&mut right);
                result.push((
                    left.degree,
                    self.basis_element_to_index(&left),
                    right.degree,
                    self.basis_element_to_index(&right),
                    shuffle_sign(p, left_q, right_q),
                ));
                if left_q == 0 {
                    break;
                }
                left_q = (left_q - 1) & q_part;
            }
            if Self::increment_p_part(&mut left_p, p_part) {
                break;
            }
        }
        result
    }

    fn decompose(&self, op_deg: i32, op_idx: usize) -> Vec<(i32, usize)> {
        vec![(op_deg, op_idx)]
    }
}

/// The sign, as a coefficient mod `p`, of $\tau^{E'} \tau^{E''} = \pm \tau^{E' \sqcup E''}$ in the
/// exterior algebra on the $\tau_k$ written in increasing order: one transposition for each
/// $k' \in E'$ and $k'' \in E''$ with $k' > k''$.
fn shuffle_sign(p: ValidPrime, left: u32, right: u32) -> u32 {
    let inversions: u32 = BitflagIterator::set_bit_iterator(left as u64)
        .map(|k| (right & ((1 << k) - 1)).count_ones())
        .sum();
    minus_one_to_the_n(p, inversions as i32)
}

#[cfg(test)]
mod tests {
    use fp::{prime::Prime, vector::FpVector};
    use rstest::rstest;
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::algebra::milnor_algebra::{Exterior, MilnorProfile, NoExterior};

    /// An element of $A \otimes A$, by pairs of basis elements, with no zero coefficients.
    type Tensor = FxHashMap<(i32, usize, i32, usize), u32>;

    /// Adds `c` times the product of basis elements `(d1, i1)` and `(d2, i2)` to `out` as a vector.
    fn product<A: Algebra>(alg: &A, d1: i32, i1: usize, d2: i32, i2: usize) -> FpVector {
        let mut out = FpVector::new(alg.prime(), alg.dimension(d1 + d2));
        alg.multiply_basis_elements(out.as_slice_mut(), 1, d1, i1, d2, i2);
        out
    }

    /// Adds `c * x` to `t`, dropping zero coefficients.
    fn add(t: &mut Tensor, p: ValidPrime, key: (i32, usize, i32, usize), c: u32) {
        let e = t.entry(key).or_insert(0);
        *e = (*e + c) % p.as_u32();
        if *e == 0 {
            t.remove(&key);
        }
    }

    /// Checks that the coproduct is multiplicative, $\Delta(ab) = \Delta(a) \Delta(b)$ with the
    /// Koszul sign $(a' \otimes a'')(b' \otimes b'') = (-1)^{|a''||b'|} a'b' \otimes a''b''$, on
    /// every pair of basis elements of total degree at most `max_degree`.
    fn check_multiplicative<A: Algebra + Bialgebra>(alg: &A, max_degree: i32) {
        let p = alg.prime();
        alg.compute_basis(max_degree);
        for d1 in 0..=max_degree {
            for d2 in 0..=max_degree - d1 {
                for i1 in 0..alg.dimension(d1) {
                    for i2 in 0..alg.dimension(d2) {
                        let mut lhs = Tensor::default();
                        for (k, c) in product(alg, d1, i1, d2, i2).iter_nonzero() {
                            for (l0, l1, r0, r1, e) in alg.coproduct(d1 + d2, k) {
                                add(&mut lhs, p, (l0, l1, r0, r1), c * e);
                            }
                        }
                        let mut rhs = Tensor::default();
                        for (a0, a1, a2, a3, ca) in alg.coproduct(d1, i1) {
                            for (b0, b1, b2, b3, cb) in alg.coproduct(d2, i2) {
                                let sign = minus_one_to_the_n(p, a2 * b0);
                                let c = ca * cb % p.as_u32() * sign % p.as_u32();
                                let left = product(alg, a0, a1, b0, b1);
                                let right = product(alg, a2, a3, b2, b3);
                                for (x, cx) in left.iter_nonzero() {
                                    for (y, cy) in right.iter_nonzero() {
                                        let k = (a0 + b0, x, a2 + b2, y);
                                        add(&mut rhs, p, k, c * cx % p.as_u32() * cy);
                                    }
                                }
                            }
                        }
                        assert_eq!(lhs, rhs, "p = {p}, ({d1}, {i1}) * ({d2}, {i2})");
                    }
                }
            }
        }
    }

    #[rstest]
    #[case(2, 24)]
    #[case(3, 40)]
    #[case(5, 50)]
    fn coproduct_is_multiplicative_without_exterior(#[case] p: u32, #[case] max_degree: i32) {
        let p = ValidPrime::new(p);
        let alg =
            MilnorAlgebraInner::<NoExterior>::new_with_profile(p, MilnorProfile::default(), false);
        check_multiplicative(&alg, max_degree);
    }

    #[rstest]
    #[case(2, 20)]
    #[case(3, 40)]
    #[case(5, 50)]
    fn coproduct_is_multiplicative_with_exterior(#[case] p: u32, #[case] max_degree: i32) {
        let p = ValidPrime::new(p);
        let alg =
            MilnorAlgebraInner::<Exterior>::new_with_profile(p, MilnorProfile::default(), false);
        check_multiplicative(&alg, max_degree);
    }

    /// `increment_p_part` walks up to and including `max`, whose top entry may saturate its field.
    /// Incrementing before carrying would overflow there.
    #[test]
    fn ppart_odometer_handles_saturated_field() {
        let top = PPart::MAX_LEN - 1;
        let mut max = PPart::from_slice(&[2]);
        max.set(top, PPart::max_entry(top));

        let mut count = 0;
        let mut cur = PPart::zero();
        loop {
            count += 1;
            if MilnorAlgebraInner::<NoExterior>::increment_p_part(&mut cur, max) {
                break;
            }
        }
        assert_eq!(count, 3 * (PPart::max_entry(top) as usize + 1));
        // Wrapping leaves the odometer back at zero.
        assert_eq!(cur, PPart::zero());
    }
}
