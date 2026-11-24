//! Every local feature of the sphere's secondary resolution that the unit coefficient of the
//! cofiber homotopy `W'` of `C(h_j)` can depend on, for fitting and checking formulas.
//!
//! For `y ∈ P_s` and `g ∈ P_{s+2}` with `|g| - |y| = 2^j + 1` the features are: `H:b`, the
//! coefficient of `b y` in `H(g)` (`|b| = 2^j`); `d:b|c`, the number mod 2 of `d ∘ d` paths
//! `g → e → y` with coefficients `b` then `c`; and `G:t:i`, `G:y:k:l:i`, the coordinates of the
//! stored composite `Γ(g)` on `y` (its twos and its `Y_{k,l}` part).
//!
//! # Usage
//! Prompts as `secondary`; env `CF_J`, `CF_MAX_N`, `CF_MAX_S`.
//!
//! # Output (stdout)
//! `feature,stem,Adams_filtration,index,values`: the row of `y` over the generators `g` at
//! `(n_y + 2^j - 1, s_y + 2)`, for every feature with a nonzero row. stderr counts the pairs whose
//! homotopy or composite was not computed.

use std::{collections::BTreeMap, sync::Arc};

use algebra::{Algebra, module::Module, pair_algebra::PairAlgebra};
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::{ExtModule, SecondaryExtAlgebra},
    utils::query_module,
};
use fp::vector::FpVector;
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let env = |k: &str| std::env::var(k).map_err(|_| anyhow::anyhow!("missing env {k}"));
    let j: i32 = env("CF_J")?.parse()?;
    let max_n: i32 = env("CF_MAX_N")?.parse()?;
    let max_s: i32 = env("CF_MAX_S")?.parse()?;
    let k = 1 << j;

    let res = Arc::new(query_module(Some(algebra::AlgebraType::Milnor), true)?);
    let e2 = Arc::new(ExtModule::from_resolution(Arc::clone(&res))?);
    let sec_e2 = Arc::new(SecondaryExtAlgebra::new(Arc::clone(&e2)));
    sec_e2.extend_all();
    let res_lift = Arc::clone(sec_e2.res_lift());
    let algebra = res.algebra();
    let p = res.prime();
    algebra.compute_basis(k + 2);

    let name = |d: i32, i: usize| algebra.basis_element_to_string(d, i);
    // The layout of a pair algebra element of degree k + 1 (see `element_from_bytes`).
    let deg = k + 1;
    let log2 = |x: usize| usize::BITS as usize - 1 - x.leading_zeros() as usize;
    let mut ylayout = vec![];
    for kk in 0..log2(deg as usize) + 1 {
        let rem = deg as usize + 1 - (1 << kk);
        for l in 0..log2(rem) + 1 {
            ylayout.push((kk, l, (rem - (1 << l)) as i32));
        }
    }

    println!("feature,stem,Adams_filtration,index,values");
    let mut missing = 0usize;
    for s in 0..=max_s {
        for n in 0..=max_n {
            let by = Bidegree::n_s(n, s);
            let bg = Bidegree::n_s(n + k - 1, s + 2);
            if !res.has_computed_bidegree(by) || !res.has_computed_bidegree(bg) {
                continue;
            }
            let (ydim, gdim) = (
                res.number_of_gens_in_bidegree(by),
                res.number_of_gens_in_bidegree(bg),
            );
            if ydim == 0 || gdim == 0 {
                continue;
            }
            let (ty, tg) = (by.t(), bg.t());
            // feature -> [y][g]
            let mut feat: BTreeMap<String, Vec<Vec<u32>>> = BTreeMap::new();
            let mut put = |f: String, y: usize, g: usize| {
                feat.entry(f).or_insert_with(|| vec![vec![0; gdim]; ydim])[y][g] ^= 1;
            };
            let hom = res_lift.homotopy(s + 2);
            let p_s1 = res.module(s + 1);
            let p_s = res.module(s);
            let d2 = res.differential(s + 2);
            let d1 = res.differential(s + 1);
            for g in 0..gdim {
                // H:b
                if hom.homotopies.next_degree() > tg {
                    let hg = hom.homotopies.output(tg, g);
                    for y in 0..ydim {
                        let off = p_s.generator_offset(tg - 1, ty, y);
                        for b in 0..algebra.dimension(k) {
                            if hg.entry(off + b) != 0 {
                                put(format!("H:{}", name(k, b)), y, g);
                            }
                        }
                    }
                } else {
                    missing += 1;
                }
                // d:b|c
                let dg = d2.output(tg, g);
                for (te, ei) in p_s1.iter_gens(tg - 1) {
                    if te <= ty || te >= tg {
                        continue;
                    }
                    let (bd, cd) = (tg - te, te - ty);
                    let off = p_s1.generator_offset(tg, te, ei);
                    let de = d1.output(te, ei);
                    for b in 0..algebra.dimension(bd) {
                        if dg.entry(off + b) == 0 {
                            continue;
                        }
                        for y in 0..ydim {
                            let offy = p_s.generator_offset(te, ty, y);
                            for c in 0..algebra.dimension(cd) {
                                if de.entry(offy + c) != 0 {
                                    put(format!("d:{}|{}", name(bd, b), name(cd, c)), y, g);
                                }
                            }
                        }
                    }
                }
                // G: the composite's coefficient on y
                if hom.composites.len() > tg {
                    let comp = hom.composite(tg, g);
                    for y in 0..ydim {
                        let Some(elt) = comp.coefficient(ty, y) else {
                            continue;
                        };
                        let mut buf = vec![];
                        algebra.element_to_bytes(elt, &mut buf)?;
                        let mut cur = &buf[..];
                        let twos = FpVector::from_bytes(p, algebra.dimension(deg), &mut cur)?;
                        for (i, _) in twos.iter_nonzero() {
                            put(format!("G:t:{}", name(deg, i)), y, g);
                        }
                        for &(kk, l, d) in &ylayout {
                            let v = FpVector::from_bytes(p, algebra.dimension(d), &mut cur)?;
                            for (i, _) in v.iter_nonzero() {
                                put(format!("G:y:{kk}:{l}:{}", name(d, i)), y, g);
                            }
                        }
                    }
                } else {
                    missing += 1;
                }
            }
            for (f, m) in feat {
                for (y, row) in m.iter().enumerate() {
                    if row.iter().any(|&v| v != 0) {
                        println!("{f},{n},{s},{y},\"{row:?}\"");
                    }
                }
            }
        }
    }
    eprintln!("missing homotopies or composites: {missing}");
    Ok(())
}
