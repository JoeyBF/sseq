//! The bottom-cell block of the secondary homotopy of `C(h_j)`, read from the sphere's secondary
//! chain map of `h_j`.
//!
//! The cone `R = P ⊕ Σ^{2^j} P` of the chain map `φ` of `h_j` (`d_R = [[d, φ], [0, d]]`) is a
//! resolution of `H^* C(h_j)`, and `[[H, Φ], [0, H]]` is a secondary homotopy for it, where `H` is
//! the sphere's and `Φ` the secondary homotopy of `φ` (`SecondaryResolutionHomomorphism`): the
//! cone's recursion `d_R H_R = I_R + H_R d_R` is, block by block, the sphere's recursion on the
//! diagonal and the secondary chain map recursion off it, with the same zero base. So the unit
//! coefficients of `Φ` are exactly the `(2^j, 0)` block of `module_d2`'s output, with no formula.
//!
//! With `CD_MAP=theta` the chain map is not lifted but given: `δ'(g) = θ(d g)`, where `θ` caps the
//! coefficients with `ξ_1^{2^j}` (`Sq(R) ↦ Sq(R - 2^j e_1)`). Its cone is `P ⊗ H^* C(h_j)` itself
//! (`D(g ⊗ x_0) = d g ⊗ x_0 + δ'(g) ⊗ x_1`), so the output is in the basis of the cocycles `y ⊗ x_1`
//! that `cofiber_slots` uses; the cone of a lifted chain map of `h_j` differs from it by a chain
//! homotopy, which moves the lifts by `i_*` terms. `CD_VERIFY=1` checks `d Φ = I` on every
//! generator of the first step, the one step whose lift the recursion does not check itself.
//!
//! # Usage
//! Prompts as `secondary` (S_2, save dir, max n, max s); env `CD_J` (the `j` of `h_j`),
//! `CD_NAME` (module name in the output), `CD_MAX_N`, `CD_MAX_S` (range of the class `y`).
//!
//! # Output (stdout)
//! `module_d2`'s format, the `(2^j, 0)` block only:
//! `module,stem,Adams_filtration,index,c_deg,c_idx,b_deg,b_idx,values`, the row of `y` over the
//! generators `g` at `(n_y + 2^j - 1, s_y + 2)`. stderr reports any bidegree of `y` whose block was
//! not computed, so a missing block is never read as zero.

use std::sync::Arc;

use algebra::module::{Module, homomorphism::ModuleHomomorphism};
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::{ExtModule, SecondaryExtAlgebra},
    resolution_homomorphism::ResolutionHomomorphism,
    secondary::{SecondaryLift, SecondaryResolutionHomomorphism},
    utils::query_module,
};
use fp::vector::FpVector;
use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let t0 = std::time::Instant::now();
    let phase = |name: &str| eprintln!("PHASE {name} at {:.1}s", t0.elapsed().as_secs_f64());
    let env = |k: &str| std::env::var(k).map_err(|_| anyhow::anyhow!("missing env {k}"));

    let j: i32 = env("CD_J")?.parse()?;
    let name = env("CD_NAME")?;
    let max_n: i32 = env("CD_MAX_N")?.parse()?;
    let max_s: i32 = env("CD_MAX_S")?.parse()?;
    let k = 1 << j;

    let res = Arc::new(query_module(Some(algebra::AlgebraType::Milnor), true)?);
    let e2 = Arc::new(ExtModule::from_resolution(Arc::clone(&res))?);
    assert!(e2.is_unit(), "this driver is for the sphere");
    let sec_e2 = Arc::new(SecondaryExtAlgebra::new(Arc::clone(&e2)));
    sec_e2.extend_all();
    phase("sphere secondary resolution extended");

    let theta = std::env::var("CD_MAP").is_ok_and(|v| v == "theta");
    let lift = if theta {
        let algebra = res.algebra();
        let p = res.prime();
        let phi = Arc::new(ResolutionHomomorphism::new(
            String::new(),
            Arc::clone(&res),
            Arc::clone(&res),
            Bidegree::s_t(1, k),
        ));
        let mut nonzero = 0usize;
        for s in 1..res.next_homological_degree() {
            let (source, target) = (res.module(s), res.module(s - 1));
            let d = res.differential(s);
            for t in std::cmp::max(source.min_degree(), target.min_degree() + k)..=source.max_computed_degree() {
                let dim = if t - k >= target.min_degree() { target.dimension(t - k) } else { 0 };
                let rows: Vec<FpVector> = (0..source.number_of_gens_in_degree(t))
                    .map(|g| {
                        let mut row = FpVector::new(p, dim);
                        let dg = d.output(t, g);
                        for (gen_deg, gen_idx, op_deg, slice) in target.iter_slices(t, dg.as_slice()) {
                            if op_deg < k {
                                continue;
                            }
                            for (idx, _) in slice.iter_nonzero() {
                                let mut elt = algebra.basis_element_from_index(op_deg, idx);
                                let r1 = elt.p_part.get(0);
                                if (r1 as i32) < k {
                                    continue;
                                }
                                elt.p_part.set(0, r1 - k as algebra::milnor_algebra::PPartEntry);
                                elt.degree -= k;
                                let new = algebra.basis_element_to_index(&elt);
                                row.add_basis_element(
                                    target.generator_offset(t - k, gen_deg, gen_idx) + new,
                                    1,
                                );
                                nonzero += 1;
                            }
                        }
                        row
                    })
                    .collect();
                phi.set_map_rows(s, t, rows);
            }
        }
        phase(&format!("delta' built ({nonzero} terms)"));
        Arc::new(SecondaryResolutionHomomorphism::new(
            Arc::clone(sec_e2.res_lift()),
            Arc::clone(sec_e2.res_lift()),
            phi,
        ))
    } else {
        // h_j is the only generator of (2^j - 1, 1) for j >= 0.
        let hj = Bidegree::n_s(k - 1, 1);
        assert_eq!(e2.dimension(hj), 1, "h_{j} is not alone in its bidegree");
        let x: BidegreeElement = e2.element(hj, &[1]);
        let lift = sec_e2.secondary_product_lift(&x);
        lift.underlying().extend_all();
        lift
    };
    phase("chain map of h_j ready");
    lift.extend_all();
    phase("secondary chain map of h_j extended");

    if std::env::var("CD_VERIFY").is_ok() {
        // The first step: d Φ(g) must equal the intermediate exactly (zero base).
        let s = lift.shift().s() + 1;
        let hom = &lift.homotopies()[s].homotopies;
        let target = res.module(s - 2);
        let dt = res.differential(s - 2);
        let (mut ok, mut bad) = (0usize, 0usize);
        for t in std::cmp::max(res.module(s).min_degree(), hom.min_degree())..hom.next_degree() {
            for i in 0..res.module(s).number_of_gens_in_degree(t) {
                let g = BidegreeGenerator::s_t(s, t, i);
                let mut v = lift.compute_intermediate(g);
                let out = hom.output(t, i);
                let tt = t - lift.shift().t() - 1;
                if tt >= target.min_degree() {
                    dt.apply(v.as_slice_mut(), 1, tt, out.as_slice());
                }
                if v.is_zero() { ok += 1 } else { bad += 1; eprintln!("FIRST STEP FAILS at {g}") }
            }
        }
        eprintln!("first step check: {ok} generators exact, {bad} failing");
    }

    let homotopies = lift.homotopies();
    println!("module,stem,Adams_filtration,index,c_deg,c_idx,b_deg,b_idx,values");
    let (mut rows, mut missing) = (0usize, 0usize);
    for s_y in 0..=max_s {
        let s_g = s_y + 2;
        let Some(hom) = homotopies.get(s_g) else {
            continue;
        };
        let hom = &hom.homotopies;
        for n_y in 0..=max_n {
            let by = Bidegree::n_s(n_y, s_y);
            let bg = Bidegree::n_s(n_y + k - 1, s_g);
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
            assert_eq!(by.t() + hom.degree_shift(), bg.t(), "unexpected shift");
            if hom.next_degree() <= bg.t() {
                missing += 1;
                eprintln!("MISSING block at y = ({n_y}, {s_y})");
                continue;
            }
            let m = hom.hom_k(by.t());
            for (yi, row) in m.iter().enumerate() {
                if row.iter().any(|&v| v != 0) {
                    println!("{name},{n_y},{s_y},{yi},{k},0,0,0,\"{row:?}\"");
                    rows += 1;
                }
            }
        }
    }
    eprintln!("{name}: {rows} nonzero rows, {missing} missing blocks");
    phase("done");
    Ok(())
}
