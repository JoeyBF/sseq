//! The unit-level obstruction cochain `J` of the cofiber homotopy of `C(h_j)` on the bar resolution,
//! computed exactly in the Milnor pair algebra, against the 3-cochain `O_j` the formula solves.
//!
//! On `g = [a1|a2|a3]` (total degree `2^j + 1`) the only term of `d g` with a nonunit coefficient
//! is `a1 [a2|a3]`, and `A(1, -) = 0`, `θ(1) = 0`, so
//!
//! `J(a1, a2, a3) = A(a1, Λ) + A(θ a1, Γ) + θ A(a1, Γ)`, the unit coefficient, where
//! `Γ = σ(a2)σ(a3) + σ(a2 a3)` and `Λ = σ(a2)σ(θ a3) + σ(θ a2)σ(a3) + σ(θ(a2 a3))`
//! are the composite of `[a2|a3]` and its `θ`-twist, and `θ` caps with `ξ_1^{2^j}`.
//!
//! # Output
//! For each `j`: the triples where `J` is nonzero, and whether `J = O_j = [ξ_1 | ξ_1^h | ξ_1^h]`
//! (`h = 2^{j-1}`, `j ≥ 2`; `O_j = 0` for `j ≤ 1`).

use algebra::{Algebra, MilnorAlgebra, pair_algebra::PairAlgebra};
use fp::{prime::TWO, vector::FpVector};

/// `θ` on a Milnor vector of degree `deg`: `Sq(R) ↦ Sq(R - k e_1)`.
fn theta(alg: &MilnorAlgebra, k: i32, deg: i32, v: &FpVector) -> FpVector {
    let mut out = FpVector::new(TWO, if deg >= k { alg.dimension(deg - k) } else { 0 });
    if deg < k {
        return out;
    }
    for (i, _) in v.iter_nonzero() {
        let mut e = alg.basis_element_from_index(deg, i);
        let r1 = e.p_part.get(0);
        if (r1 as i32) < k {
            continue;
        }
        e.p_part.set(0, r1 - k as u32);
        e.degree -= k;
        out.add_basis_element(alg.basis_element_to_index(&e), 1);
    }
    out
}

/// A basis vector.
fn unit(alg: &MilnorAlgebra, deg: i32, i: usize) -> FpVector {
    let mut v = FpVector::new(TWO, alg.dimension(deg));
    v.set_entry(i, 1);
    v
}

/// The product of two Milnor vectors.
fn mul(alg: &MilnorAlgebra, d1: i32, x: &FpVector, d2: i32, y: &FpVector) -> FpVector {
    let mut out = FpVector::new(TWO, alg.dimension(d1 + d2));
    alg.multiply_element_by_element(out.as_slice_mut(), 1, d1, x.as_slice(), d2, y.as_slice());
    out
}

fn main() {
    let alg = MilnorAlgebra::new(TWO, false);
    alg.compute_basis(20);
    let one = unit(&alg, 0, 0);
    for j in 0..=3 {
        let k: i32 = 1 << j;
        let n = k + 1;
        let (mut nonzero, mut differ) = (vec![], 0usize);
        for d1 in 1..n {
            for d2 in 1..n - d1 {
                let d3 = n - d1 - d2;
                for i1 in 0..alg.dimension(d1) {
                    for i2 in 0..alg.dimension(d2) {
                        for i3 in 0..alg.dimension(d3) {
                            let (a1, a2, a3) =
                                (unit(&alg, d1, i1), unit(&alg, d2, i2), unit(&alg, d3, i3));
                            let a23 = mul(&alg, d2, &a2, d3, &a3);
                            let dg = d2 + d3;
                            // Γ = σ(a2)σ(a3) + σ(1)σ(a2 a3)
                            let mut gamma = alg.new_pair_element(dg);
                            alg.sigma_multiply_basis(&mut gamma, 1, d2, i2, d3, i3);
                            alg.sigma_multiply(&mut gamma, 1, 0, one.as_slice(), dg, a23.as_slice());
                            MilnorAlgebra::finalize_element(&mut gamma);
                            // Λ = σ(a2)σ(θa3) + σ(θa2)σ(a3) + σ(1)σ(θ(a2 a3))
                            let ld = dg - k;
                            let mut lam_terms = false;
                            let mut lam = alg.new_pair_element(ld.max(0));
                            if ld >= 0 {
                                let ta3 = theta(&alg, k, d3, &a3);
                                let ta2 = theta(&alg, k, d2, &a2);
                                let ta23 = theta(&alg, k, dg, &a23);
                                if d3 >= k && !ta3.is_zero() {
                                    alg.sigma_multiply(&mut lam, 1, d2, a2.as_slice(), d3 - k, ta3.as_slice());
                                    lam_terms = true;
                                }
                                if d2 >= k && !ta2.is_zero() {
                                    alg.sigma_multiply(&mut lam, 1, d2 - k, ta2.as_slice(), d3, a3.as_slice());
                                    lam_terms = true;
                                }
                                if !ta23.is_zero() {
                                    alg.sigma_multiply(&mut lam, 1, 0, one.as_slice(), ld, ta23.as_slice());
                                    lam_terms = true;
                                }
                                if lam_terms {
                                    MilnorAlgebra::finalize_element(&mut lam);
                                }
                            }
                            // J (degree 0)
                            let mut j0 = 0u32;
                            if lam_terms {
                                let mut r = FpVector::new(TWO, 1);
                                alg.a_multiply(r.as_slice_mut(), 1, d1, a1.as_slice(), ld, &lam);
                                j0 ^= r.entry(0);
                            }
                            let ta1 = theta(&alg, k, d1, &a1);
                            if d1 >= k && !ta1.is_zero() {
                                let mut r = FpVector::new(TWO, alg.dimension(d1 - k + dg - 1));
                                alg.a_multiply(r.as_slice_mut(), 1, d1 - k, ta1.as_slice(), dg, &gamma);
                                if r.len() > 0 {
                                    j0 ^= r.entry(0);
                                }
                            }
                            let mut ag = FpVector::new(TWO, alg.dimension(d1 + dg - 1));
                            alg.a_multiply(ag.as_slice_mut(), 1, d1, a1.as_slice(), dg, &gamma);
                            let tag = theta(&alg, k, d1 + dg - 1, &ag);
                            if tag.len() > 0 {
                                j0 ^= tag.entry(0);
                            }
                            // O_j
                            let h = k / 2;
                            let is = |d: i32, i: usize, r: i32| {
                                let e = alg.basis_element_from_index(d, i);
                                d == r && e.p_part.get(0) as i32 == r
                            };
                            let o = (j >= 2 && is(d1, i1, 1) && is(d2, i2, h) && is(d3, i3, h)) as u32;
                            let name = |d: i32, i: usize| alg.basis_element_to_string(d, i);
                            if j0 != 0 {
                                nonzero.push(format!("[{}|{}|{}]", name(d1, i1), name(d2, i2), name(d3, i3)));
                            }
                            if j0 != o {
                                differ += 1;
                                println!(
                                    "j={j}: J != O_j at [{}|{}|{}]: J={j0} O={o}",
                                    name(d1, i1),
                                    name(d2, i2),
                                    name(d3, i3)
                                );
                            }
                        }
                    }
                }
            }
        }
        println!("j={j}: J nonzero on {nonzero:?}; {differ} triples where J != O_j");
    }
}
