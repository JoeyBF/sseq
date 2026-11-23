//! Predicts the unit-coefficient blocks of the secondary homotopy of `P ⊗ M` from the sphere alone.
//!
//! Reads a module spec written by `twisted_w` (`SPEC_OUT`): per pair of cells `(c, b)` a sum of
//! operators on the sphere's resolution, `H:e` (the coefficient of `e` in the secondary homotopy
//! `H g` at `x`, with `H:1` the unit coefficient, i.e. the sphere's `d₂`) and `d:a|a'` (the sum over
//! `d∘d` paths `g → y → x` of `[a](dg)_y [a'](dy)_x`). No lifting and no tensor resolution: only
//! coefficient reads of `d` and `H`.
//!
//! # Usage
//! Prompts as `secondary` (S_2, save dir, max n, max s); env `MD_SPEC`, `MD_MAX_N`, `MD_MAX_S`.
//!
//! # Output (stdout)
//! `cell_truth`'s format: `module,stem,Adams_filtration,index,c_deg,c_idx,b_deg,b_idx,values`.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

use algebra::{Algebra, module::Module};
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::{ExtModule, SecondaryExtAlgebra},
    utils::query_module,
};
use sseq::coordinates::Bidegree;

/// One operator of a block, with its algebra elements resolved to `(degree, index)`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Op {
    /// `H:e`; the unit coefficient when the degree is zero.
    H(i32, usize),
    /// `d:a|a'`.
    Path(i32, usize, i32, usize),
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let res = Arc::new(query_module(Some(algebra::AlgebraType::Milnor), true)?);
    let e2 = Arc::new(ExtModule::from_resolution(Arc::clone(&res))?);
    let sec_e2 = Arc::new(SecondaryExtAlgebra::new(Arc::clone(&e2)));
    sec_e2.extend_all();
    let res_lift = Arc::clone(sec_e2.res_lift());
    let algebra = res.algebra();
    let max_n: i32 = std::env::var("MD_MAX_N").map_or(60, |v| v.parse().unwrap());
    let max_s: i32 = std::env::var("MD_MAX_S").map_or(28, |v| v.parse().unwrap());

    // the spec
    let spec = std::fs::read_to_string(std::env::var("MD_SPEC")?)?;
    let mut name = String::new();
    // (c_deg, c_idx, b_deg, b_idx) -> operator names
    let mut blocks: Vec<((i32, usize, i32, usize), Vec<String>)> = vec![];
    for line in spec.lines() {
        let f: Vec<&str> = line.splitn(6, ',').collect();
        match f[0] {
            "module" => name = f[1].to_string(),
            "block" => blocks.push((
                (f[1].parse()?, f[2].parse()?, f[3].parse()?, f[4].parse()?),
                f[5].split('+').map(str::to_string).collect(),
            )),
            _ => {}
        }
    }
    let max_delta = blocks.iter().map(|((c, _, b, _), _)| c - b).max().unwrap_or(0);
    algebra.compute_basis(max_delta + 2);
    // basis element names -> (degree, index)
    let mut by_name: HashMap<String, (i32, usize)> = HashMap::new();
    for d in 0..=max_delta + 1 {
        for i in 0..algebra.dimension(d) {
            by_name.insert(algebra.basis_element_to_string(d, i), (d, i));
        }
    }
    let parse = |op: &str| -> Op {
        if op == "H:1" {
            return Op::H(0, 0);
        }
        if let Some(e) = op.strip_prefix("H:") {
            let (d, i) = by_name[e];
            return Op::H(d, i);
        }
        let (a, b) = op.strip_prefix("d:").unwrap().split_once('|').unwrap();
        let (ad, ai) = by_name[a];
        let (bd, bi) = by_name[b];
        Op::Path(ad, ai, bd, bi)
    };
    let blocks: Vec<((i32, usize, i32, usize), Vec<Op>)> = blocks
        .into_iter()
        .map(|(k, ops)| (k, ops.iter().map(|o| parse(o)).collect()))
        .collect();
    let deltas: BTreeSet<i32> = blocks.iter().map(|((c, _, b, _), _)| c - b).collect();

    println!("module,stem,Adams_filtration,index,c_deg,c_idx,b_deg,b_idx,values");
    for s in 0..=max_s {
        for n in 0..=max_n {
            let b0 = Bidegree::n_s(n, s);
            if !res.has_computed_bidegree(b0) {
                continue;
            }
            let dim = res.number_of_gens_in_bidegree(b0);
            if dim == 0 {
                continue;
            }
            let tz = b0.t();
            for &delta in &deltas {
                let tb = Bidegree::n_s(n + delta - 1, s + 2);
                if !res.has_computed_bidegree(tb) {
                    continue;
                }
                let tdim = res.number_of_gens_in_bidegree(tb);
                if tdim == 0 {
                    continue;
                }
                let tg = tz + delta + 1;
                let needed: BTreeSet<Op> = blocks
                    .iter()
                    .filter(|((c, _, b, _), _)| c - b == delta)
                    .flat_map(|(_, ops)| ops.iter().copied())
                    .collect();
                // operator -> matrix [x][g]
                let mut val: BTreeMap<Op, Vec<Vec<u32>>> = BTreeMap::new();
                for &op in &needed {
                    val.insert(op, vec![vec![0; tdim]; dim]);
                }
                let p_s1 = res.module(s + 1);
                let p_s = res.module(s);
                let d2 = res.differential(s + 2);
                let d1 = res.differential(s + 1);
                let hom = &res_lift.homotopy(s + 2).homotopies;
                let has_paths = needed.iter().any(|o| matches!(o, Op::Path(..)));
                for g in 0..tdim {
                    if has_paths {
                        let dg = d2.output(tg, g);
                        for (ty, yi) in p_s1.iter_gens(tg - 1) {
                            if ty <= tz {
                                continue;
                            }
                            let a = tg - ty;
                            let b = ty - tz;
                            let off = p_s1.generator_offset(tg, ty, yi);
                            let dy = d1.output(ty, yi);
                            let offx = p_s.generator_offset(ty, tz, 0);
                            for e1 in 0..algebra.dimension(a) {
                                if dg.entry(off + e1) == 0 {
                                    continue;
                                }
                                for x in 0..dim {
                                    for e2 in 0..algebra.dimension(b) {
                                        if dy.entry(offx + x * algebra.dimension(b) + e2) == 0 {
                                            continue;
                                        }
                                        if let Some(m) = val.get_mut(&Op::Path(a, e1, b, e2)) {
                                            m[x][g] ^= 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if hom.next_degree() > tg {
                        let hg = hom.output(tg, g);
                        if hg.len() == p_s.dimension(tg - 1) {
                            let offx = p_s.generator_offset(tg - 1, tz, 0);
                            let w = algebra.dimension(delta);
                            for &op in &needed {
                                if let Op::H(d, e) = op {
                                    debug_assert_eq!(d, delta);
                                    let m = val.get_mut(&op).unwrap();
                                    for x in 0..dim {
                                        m[x][g] ^= hg.entry(offx + x * w + e);
                                    }
                                }
                            }
                        }
                    }
                }
                for ((cd, ci, bd, bi), ops) in &blocks {
                    if cd - bd != delta {
                        continue;
                    }
                    for x in 0..dim {
                        let mut row = vec![0u32; tdim];
                        for op in ops {
                            for (r, v) in row.iter_mut().zip(&val[op][x]) {
                                *r ^= v;
                            }
                        }
                        if row.iter().any(|&v| v != 0) {
                            println!("{name},{n},{s},{x},{cd},{ci},{bd},{bi},\"{row:?}\"");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
