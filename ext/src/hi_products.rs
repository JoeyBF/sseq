//! Multiplication by `h_i`, read off the differentials of a minimal resolution. No chain map is
//! lifted.
//!
//! In a minimal resolution `F` of `F_2`, the class `h_i` is dual to the generator of `F_1` with
//! `d = Sq(2^i) ι`, and for `x ∈ Ext^{s,t}` dual to the generator `g_x` of `F_s`,
//!
//! ```text
//!     h_i x = Σ_j <Sq(2^i) g_x, d g_j> g_j^*      (g_j the generators of F_{s+1} in degree t + 2^i)
//! ```
//!
//! i.e. the coefficient of the Milnor basis element `Sq(2^i)` on `g_x` in `d g_j`. (`Sq(2^i)` is
//! the only Milnor basis element of degree `2^i` pairing non-trivially with `ξ_1^{2^i}`, the cobar
//! representative of `h_i`; in the admissible basis this fails.) That is
//! [`FreeChainComplex::try_filtration_one_product`]; [`hi_products`] sweeps it over a range.

use algebra::Algebra;
use anyhow::{Context, ensure};
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

use crate::chain_complex::FreeChainComplex;

/// One block of the `h_k` multiplication table: `prod[i]` lists the generators `j` of `Ext` at
/// `source + (1, 2^k)` whose sum is `h_k` times generator `i` of `Ext` at `source`.
#[derive(Clone, Debug)]
pub struct HiBlock {
    pub k: u32,
    pub source: Bidegree,
    pub prod: Vec<Vec<usize>>,
}

/// Every block of `h_k`, `k ∈ ks`, whose source and target both lie in `n <= max_n`,
/// `s <= max_s` and are non-zero, sorted by `(k, s, n)`. `res` must be a resolution over the
/// Milnor basis at the prime 2, computed through `(max_n, max_s)`.
pub fn hi_products<CC>(res: &CC, max_n: i32, max_s: i32, ks: &[u32]) -> anyhow::Result<Vec<HiBlock>>
where
    CC: FreeChainComplex,
{
    let algebra = res.algebra();
    ensure!(
        res.prime() == 2 && algebra.prefix() == "milnor",
        "h_i products are read off the differentials only over the Milnor basis at p = 2"
    );
    let mut jobs = vec![];
    for &k in ks {
        let deg = 1i32 << k;
        let (_, op_idx) = algebra
            .basis_element_from_string(&format!("Sq{deg}"))
            .with_context(|| format!("Sq({deg}) is not a Milnor basis element"))?;
        for s in 0..max_s {
            for n in 0..=(max_n + 1 - deg) {
                let source = Bidegree::n_s(n, s);
                let target = source + Bidegree::s_t(1, deg);
                if res.number_of_gens_in_bidegree(source) > 0
                    && res.number_of_gens_in_bidegree(target) > 0
                {
                    jobs.push((k, deg, op_idx, source));
                }
            }
        }
    }
    jobs.into_maybe_par_iter()
        .map(|(k, deg, op_idx, source)| {
            let m = res.try_filtration_one_product(deg, op_idx, source)?;
            Ok(HiBlock {
                k,
                source,
                prod: m
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter(|(_, c)| **c != 0)
                            .map(|(j, _)| j)
                            .collect()
                    })
                    .collect(),
            })
        })
        .collect()
}
