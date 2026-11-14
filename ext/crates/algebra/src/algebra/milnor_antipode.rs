use std::sync::Arc;

use dashmap::DashMap;
use fp::{prime::Prime, vector::FpVector};
use rustc_hash::FxBuildHasher;

use crate::algebra::{
    Algebra, MilnorAlgebra,
    combinatorics::xi_degrees,
    milnor_algebra::{MilnorBasisElement, PPart},
};

/// A sum of Milnor basis elements $P(R)$, by $R$, with nonzero coefficients.
pub type PPartSum = Vec<(PPart, u32)>;

/// The antipode $\chi$ of the Milnor algebra on the $P(R)$, memoised.
///
/// It comes from $\sum_{S + T = R} \chi(P(S))\, P(T) = 0$ for $R \neq 0$, the antipode axiom with
/// the coproduct $P(R) \mapsto \sum P(S) \otimes P(T)$. The $P(R)$ span a subalgebra closed
/// under the coproduct, so their antipodes stay among them.
pub struct MilnorAntipode {
    algebra: Arc<MilnorAlgebra>,
    /// $\chi(P(R))$ for every $R$ met, shared by every thread.
    memo: DashMap<PPart, Arc<PPartSum>, FxBuildHasher>,
}

impl MilnorAntipode {
    /// The antipode of `algebra`.
    pub fn new(algebra: Arc<MilnorAlgebra>) -> Self {
        Self {
            algebra,
            memo: DashMap::default(),
        }
    }

    /// The degree of $P(R)$.
    pub fn degree(&self, r: PPart) -> i32 {
        let xi = xi_degrees(self.algebra.prime());
        self.algebra.q() * r.iter().zip(xi).map(|(e, &d)| e as i32 * d).sum::<i32>()
    }

    /// The index of $P(R)$ in its degree.
    fn index(&self, r: PPart) -> usize {
        self.algebra.basis_element_to_index(&MilnorBasisElement {
            q_part: 0,
            p_part: r,
            degree: self.degree(r),
        })
    }

    /// $\chi(P(R))$.
    pub fn of(&self, r: PPart) -> Arc<PPartSum> {
        if r.is_empty() {
            return Arc::new(vec![(r, 1)]);
        }
        if let Some(v) = self.memo.get(&r) {
            return Arc::clone(&v);
        }
        let p = self.algebra.prime();
        let degree = self.degree(r);
        self.algebra.compute_basis(degree);
        let mut acc = FpVector::new(p, self.algebra.dimension(degree));
        // chi(P(R)) = - sum over nonzero T <= R of chi(P(R - T)) P(T).
        let mut t = PPart::zero();
        while !increment(&mut t, r) {
            let mut s = PPart::zero();
            for i in 0..r.len() {
                s.set(i, r.get(i) - t.get(i));
            }
            let (t_deg, t_idx) = (self.degree(t), self.index(t));
            for &(m, c) in self.of(s).iter() {
                self.algebra.multiply_basis_elements(
                    acc.as_slice_mut(),
                    (p.as_u32() - c) % p.as_u32(),
                    self.degree(m),
                    self.index(m),
                    t_deg,
                    t_idx,
                );
            }
        }
        let out: PPartSum = acc
            .iter_nonzero()
            .map(|(i, c)| (self.algebra.basis_element_from_index(degree, i).p_part, c))
            .collect();
        let out = Arc::new(out);
        // Another thread may have finished the same R; keep whichever landed first.
        Arc::clone(self.memo.entry(r).or_insert(out).value())
    }
}

/// Steps `t` through the p-parts bounded by `max`, in odometer order; returns `true` on wrapping.
fn increment(t: &mut PPart, max: PPart) -> bool {
    for i in 0..max.len() {
        if t.get(i) < max.get(i) {
            t.set(i, t.get(i) + 1);
            return false;
        }
        t.set(i, 0);
    }
    true
}

#[cfg(test)]
mod tests {
    use fp::prime::ValidPrime;

    use super::*;

    /// $\chi$ is an involution and reverses products, on every pair of degree at most a bound.
    #[test]
    fn antipode_is_an_anti_involution() {
        for (p, max) in [(2, 30), (3, 48)] {
            let alg = Arc::new(MilnorAlgebra::new(ValidPrime::new(p), false));
            alg.compute_basis(max);
            let chi = MilnorAntipode::new(Arc::clone(&alg));
            let as_vec = |d: i32, sum: &PPartSum| {
                let mut v = FpVector::new(alg.prime(), alg.dimension(d));
                for &(r, c) in sum {
                    v.add_basis_element(chi.index(r), c);
                }
                v
            };
            // chi applied to a vector.
            let chi_vec = |d: i32, v: &FpVector| {
                let mut out = FpVector::new(alg.prime(), alg.dimension(d));
                for (i, c) in v.iter_nonzero() {
                    let elt = alg.basis_element_from_index(d, i);
                    if elt.q_part != 0 {
                        continue;
                    }
                    out.add(&as_vec(d, &chi.of(elt.p_part)), c);
                }
                out
            };
            let ppart = |d: i32, i: usize| {
                let e = alg.basis_element_from_index(d, i);
                (e.q_part == 0).then_some(e.p_part)
            };
            for d in 0..=max {
                for i in 0..alg.dimension(d) {
                    let Some(r) = ppart(d, i) else { continue };
                    let twice = chi_vec(d, &as_vec(d, &chi.of(r)));
                    let mut one = FpVector::new(alg.prime(), alg.dimension(d));
                    one.set_entry(i, 1);
                    assert_eq!(twice, one, "chi^2 at {r:?}");
                }
            }
            for a in 0..=max {
                for b in 0..=max - a {
                    for i in 0..alg.dimension(a) {
                        let Some(r) = ppart(a, i) else { continue };
                        for j in 0..alg.dimension(b) {
                            let Some(s) = ppart(b, j) else { continue };
                            let mut prod = FpVector::new(alg.prime(), alg.dimension(a + b));
                            alg.multiply_basis_elements(prod.as_slice_mut(), 1, a, i, b, j);
                            let lhs = chi_vec(a + b, &prod);
                            let mut rhs = FpVector::new(alg.prime(), alg.dimension(a + b));
                            for &(x, cx) in chi.of(s).iter() {
                                for &(y, cy) in chi.of(r).iter() {
                                    alg.multiply_basis_elements(
                                        rhs.as_slice_mut(),
                                        cx * cy % p,
                                        b,
                                        chi.index(x),
                                        a,
                                        chi.index(y),
                                    );
                                }
                            }
                            assert_eq!(lhs, rhs, "chi(P{r:?} P{s:?})");
                        }
                    }
                }
            }
        }
    }
}
