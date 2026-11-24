//! Checks the Z/4 carry of the d2 read off the non-minimal `P ⊗ M` against `M`'s own minimal
//! resolution: the rank of d2 per bidegree from the minimal resolution (no carry there), from
//! `TensorResolutionSecondary::d2` with the carry, and without it (`TRS_NO_CARRY`).
//!
//! # Usage
//! Prompts as `secondary` for the sphere (S_2, save dir, max n, max s); env `CC_MODULES`
//! (comma separated module names), `CC_MAX_N`, `CC_MAX_S`.

use std::{collections::BTreeMap, sync::Arc};

use algebra::{
    Algebra, MilnorAlgebra,
    module::{FDModule, Module},
};
use ext::{
    chain_complex::ChainComplex,
    ext_algebra::tensor_resolution::TensorResolutionSecondary,
    secondary::{SecondaryLift, SecondaryResolution},
    utils::{load_module_json, query_module},
};
use sseq::coordinates::{Bidegree, BidegreeElement};

/// Rank over F2 of a list of rows.
fn rank(rows: &[Vec<u32>]) -> usize {
    let mut piv: Vec<Vec<u32>> = vec![];
    let mut r = 0;
    for row in rows {
        let mut v = row.clone();
        for p in &piv {
            let lead = p.iter().position(|&x| x == 1).unwrap();
            if v[lead] == 1 {
                for (a, b) in v.iter_mut().zip(p) {
                    *a ^= b;
                }
            }
        }
        if v.iter().any(|&x| x == 1) {
            piv.push(v);
            r += 1;
        }
    }
    r
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let res = Arc::new(query_module(Some(algebra::AlgebraType::Milnor), false)?);
    let max_n: i32 = std::env::var("CC_MAX_N").map_or(40, |v| v.parse().unwrap());
    let max_s: i32 = std::env::var("CC_MAX_S").map_or(12, |v| v.parse().unwrap());
    let algebra = res.algebra();
    algebra.compute_basis(max_n + max_s + 24);
    for name in std::env::var("CC_MODULES")?.split(',') {
        let json = load_module_json(name)?;
        let m = FDModule::from_json(Arc::clone(&algebra), &json)?;
        m.compute_basis(max_n + max_s + 20);
        let module = Arc::new(m);
        let top = module.max_degree().unwrap_or(0);

        // The minimal resolution of M with sseq's default (zero) secondary lift.
        let c = Arc::new(ext::resolution::Resolution::new(Arc::new(
            ext::chain_complex::FiniteChainComplex::<
                FDModule<MilnorAlgebra>,
                algebra::module::homomorphism::FullModuleHomomorphism<
                    FDModule<MilnorAlgebra>,
                    FDModule<MilnorAlgebra>,
                >,
            >::ccdz(Arc::clone(&module)),
        )));
        c.compute_through_stem(Bidegree::n_s(max_n + 1, max_s + 3));
        let sec = Arc::new(SecondaryResolution::new(Arc::clone(&c)));
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sec.extend_all())).is_ok();
        let mut minimal = BTreeMap::new();
        if ok {
            for b in c.iter_stem() {
                if b.n() > max_n || b.s() > max_s || b.t() <= 0 {
                    continue;
                }
                if !c.has_computed_bidegree(b + Bidegree::n_s(-1, 2)) {
                    continue;
                }
                let r = rank(&sec.homotopy(b.s() + 2).homotopies.hom_k(b.t()));
                if r > 0 {
                    minimal.insert((b.n(), b.s()), r);
                }
            }
        }

        // P ⊗ M.
        let trs = TensorResolutionSecondary::new(Arc::clone(&res), Arc::clone(&module));
        trs.resolution().set_pstem_max(max_n + top + 2);
        trs.compute_through_stem(Bidegree::n_s(max_n, max_s));
        let mut with = BTreeMap::new();
        let mut without = BTreeMap::new();
        for s in 0..=max_s {
            for n in 0..=max_n {
                let b = Bidegree::n_s(n, s);
                let Some(dim) = trs.cohomology_dimension(b) else {
                    continue;
                };
                if dim == 0 {
                    continue;
                }
                for (map, carry) in [(&mut with, true), (&mut without, false)] {
                    if carry {
                        // SAFETY: single-threaded here; the variable only toggles the carry.
                        unsafe { std::env::remove_var("TRS_NO_CARRY") };
                    } else {
                        unsafe { std::env::set_var("TRS_NO_CARRY", "1") };
                    }
                    let mut rows = vec![];
                    let mut known = true;
                    for i in 0..dim {
                        let mut v = vec![0u32; dim];
                        v[i] = 1;
                        let x = BidegreeElement::new(b, fp::vector::FpVector::from_slice(fp::prime::TWO, &v));
                        match trs.d2(&x) {
                            Some(y) => rows.push(y.vec().iter().collect::<Vec<u32>>()),
                            None => known = false,
                        }
                    }
                    if known {
                        let r = rank(&rows);
                        if r > 0 {
                            map.insert((n, s), r);
                        }
                    }
                }
            }
        }
        let keys: std::collections::BTreeSet<_> =
            minimal.keys().chain(with.keys()).chain(without.keys()).copied().collect();
        let (mut ok_with, mut ok_without) = (0, 0);
        let mut bad_with = vec![];
        let mut bad_without = vec![];
        for k in keys {
            let (a, b, w) = (minimal.get(&k), with.get(&k), without.get(&k));
            if a == b { ok_with += 1 } else { bad_with.push((k, a, b)) }
            if a == w { ok_without += 1 } else { bad_without.push((k, a, w)) }
        }
        println!(
            "{name}: minimal lift {}; with carry {ok_with} agree, {} differ {:?}; without carry {ok_without} agree, {} differ {:?}",
            if ok { "OK" } else { "FAILED" },
            bad_with.len(),
            &bad_with[..bad_with.len().min(6)],
            bad_without.len(),
            &bad_without[..bad_without.len().min(6)]
        );
    }
    Ok(())
}
