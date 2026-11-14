use fp::vector::FpVector;

use super::Comodule;
use crate::algebra::Algebra;

/// Checks that the action is associative, $a \cdot (b \cdot m) = (ab) \cdot m$, on every pair of
/// basis operations of positive degree and every basis element of degree at most `max_degree`.
pub fn check_associativity<M: Comodule>(m: &M, max_degree: i32) -> Result<(), String> {
    let p = m.prime();
    let alg = m.algebra();
    let min = m.min_degree();
    m.compute_basis(max_degree);
    alg.compute_basis(max_degree - min);
    for d in min..=max_degree {
        for idx in 0..m.dimension(d) {
            for nb in 1..=d - min {
                for na in 1..=d - min - nb {
                    let out_dim = m.dimension(d - na - nb);
                    if out_dim == 0 {
                        continue;
                    }
                    for b in 0..alg.dimension(nb) {
                        let mut bm = FpVector::new(p, m.dimension(d - nb));
                        m.act_on_basis(bm.as_slice_mut(), 1, nb, b, d, idx);
                        for a in 0..alg.dimension(na) {
                            let mut lhs = FpVector::new(p, out_dim);
                            m.act(lhs.as_slice_mut(), 1, na, a, d - nb, bm.as_slice());
                            let mut ab = FpVector::new(p, alg.dimension(na + nb));
                            alg.multiply_basis_elements(ab.as_slice_mut(), 1, na, a, nb, b);
                            let mut rhs = FpVector::new(p, out_dim);
                            for (k, c) in ab.iter_nonzero() {
                                m.act_on_basis(rhs.as_slice_mut(), c, na + nb, k, d, idx);
                            }
                            if lhs != rhs {
                                return Err(format!(
                                    "a = ({na}, {a}), b = ({nb}, {b}) on ({d}, {idx}): {lhs} != {rhs}"
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
