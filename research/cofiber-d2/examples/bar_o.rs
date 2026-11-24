//! The universal cross term: sseq's secondary algorithm on the bar resolution `B` of `F_2` and on
//! `B ⊗ M`; prints `U − χ-shear(H_B)` on the generators `[a1|a2|a3] ⊗ b → [a] ⊗ c`.
//!
//! Usage: `bar_w <module> [t_max] [s_max]`.

use std::{collections::HashMap, sync::Arc};

use algebra::{
    Algebra, MilnorAlgebra,
    module::{
        FDModule, FreeModule, Module,
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
    },
};
use ext::{
    chain_complex::ChainComplex,
    ext_algebra::tensor_resolution::{Antipode, TensorResolutionSecondary},
    secondary::SecondaryLift,
    utils::load_module_json,
};
use fp::vector::FpVector;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

type Word = Vec<(i32, usize)>;

/// The reduced bar resolution of `F_2`, through internal degree `t_max` and `s ≤ s_max`.
struct Bar {
    algebra: Arc<MilnorAlgebra>,
    modules: Vec<Arc<FreeModule<MilnorAlgebra>>>,
    diffs: Vec<Arc<FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>>>,
    zero: Arc<FreeModule<MilnorAlgebra>>,
    /// words[s][t]: the generators of B_s in degree t
    words: Vec<Vec<Vec<Word>>>,
    index: Vec<HashMap<Word, usize>>,
    t_max: i32,
    s_max: i32,
}

impl Bar {
    /// Builds `B` through `(s_max, t_max)` with its differentials' quasi-inverses.
    fn new(algebra: Arc<MilnorAlgebra>, t_max: i32, s_max: i32) -> Self {
        algebra.compute_basis(t_max + 1);
        let mut words: Vec<Vec<Vec<Word>>> = vec![];
        for s in 0..=s_max {
            let mut by_t: Vec<Vec<Word>> = vec![vec![]; (t_max + 1) as usize];
            if s == 0 {
                by_t[0].push(vec![]);
            } else {
                for t in 1..=t_max {
                    for w in &words[(s - 1) as usize] {
                        let _ = w;
                    }
                    let _ = t;
                }
                // extend words of s-1 by one more letter at the end
                for t0 in 0..=t_max {
                    for w in &words[(s - 1) as usize][t0 as usize] {
                        for d in 1..=t_max - t0 {
                            for i in 0..algebra.dimension(d) {
                                let mut w2 = w.clone();
                                w2.push((d, i));
                                by_t[(t0 + d) as usize].push(w2);
                            }
                        }
                    }
                }
            }
            words.push(by_t);
        }
        let index: Vec<HashMap<Word, usize>> = words
            .iter()
            .map(|by_t| {
                let mut h = HashMap::new();
                for ws in by_t {
                    for (i, w) in ws.iter().enumerate() {
                        h.insert(w.clone(), i);
                    }
                }
                h
            })
            .collect();
        let zero = Arc::new(FreeModule::new(Arc::clone(&algebra), "0".to_string(), 0));
        zero.compute_basis(t_max);
        let modules: Vec<Arc<FreeModule<MilnorAlgebra>>> = (0..=s_max)
            .map(|s| {
                let m = Arc::new(FreeModule::new(Arc::clone(&algebra), format!("B_{s}"), 0));
                m.compute_basis(t_max);
                for t in 0..=t_max {
                    m.add_generators(t, words[s as usize][t as usize].len(), None);
                }
                m
            })
            .collect();
        let mut diffs = vec![];
        for s in 0..=s_max {
            let target = if s == 0 { Arc::clone(&zero) } else { Arc::clone(&modules[(s - 1) as usize]) };
            let d = FreeModuleHomomorphism::new(Arc::clone(&modules[s as usize]), target, 0);
            if s == 0 {
                d.extend_by_zero(t_max);
            } else {
                let tgt = &modules[(s - 1) as usize];
                for t in 0..=t_max {
                    let mut rows = vec![];
                    for w in &words[s as usize][t as usize] {
                        let mut out = FpVector::new(fp::prime::TWO, tgt.dimension(t));
                        // a1 · [a2 | ... ]
                        let (a1d, a1i) = w[0];
                        let rest: Word = w[1..].to_vec();
                        let gi = index[(s - 1) as usize][&rest];
                        out.add_basis_element(tgt.generator_offset(t, t - a1d, gi) + a1i, 1);
                        // [ ... | a_i a_{i+1} | ... ]
                        for i in 0..w.len() - 1 {
                            let (d1, i1) = w[i];
                            let (d2, i2) = w[i + 1];
                            let mut prod = FpVector::new(fp::prime::TWO, algebra.dimension(d1 + d2));
                            algebra.multiply_basis_elements(prod.as_slice_mut(), 1, d1, i1, d2, i2);
                            for (e, _) in prod.iter_nonzero() {
                                let mut w2: Word = w[..i].to_vec();
                                w2.push((d1 + d2, e));
                                w2.extend_from_slice(&w[i + 2..]);
                                let gi = index[(s - 1) as usize][&w2];
                                out.add_basis_element(tgt.generator_offset(t, t, gi), 1);
                            }
                        }
                        rows.push(out);
                    }
                    d.add_generators_from_rows(t, rows);
                }
            }
            d.compute_auxiliary_data_through_degree(t_max);
            diffs.push(Arc::new(d));
        }
        Self { algebra, modules, diffs, zero, words, index, t_max, s_max }
    }
}

impl ChainComplex for Bar {
    type Algebra = MilnorAlgebra;
    type Homomorphism = FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>;
    type Module = FreeModule<MilnorAlgebra>;

    fn algebra(&self) -> Arc<MilnorAlgebra> {
        Arc::clone(&self.algebra)
    }

    fn min_degree(&self) -> i32 {
        0
    }

    fn zero_module(&self) -> Arc<Self::Module> {
        Arc::clone(&self.zero)
    }

    fn module(&self, s: i32) -> Arc<Self::Module> {
        Arc::clone(&self.modules[s as usize])
    }

    fn differential(&self, s: i32) -> Arc<Self::Homomorphism> {
        Arc::clone(&self.diffs[s as usize])
    }

    fn has_computed_bidegree(&self, b: Bidegree) -> bool {
        b.s() >= 0 && b.s() <= self.s_max && b.t() <= self.t_max
    }

    fn compute_through_bidegree(&self, _b: Bidegree) {}

    fn next_homological_degree(&self) -> i32 {
        self.s_max + 1
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let name = &args[1];
    let t_max: i32 = args.get(2).map_or(11, |v| v.parse().unwrap());
    let algebra = Arc::new(MilnorAlgebra::new(fp::prime::TWO, false));
    let bar = Arc::new(Bar::new(Arc::clone(&algebra), t_max, 5));
    let m = FDModule::from_json(Arc::clone(&algebra), &load_module_json(name)?)?;
    m.compute_basis(t_max + 2);
    let module = Arc::new(m);
    let top = module.max_degree().unwrap_or(0);

    // B ⊗ M: composites and the s = 3 A-terms only (H_2 is the unknown)
    let trs = TensorResolutionSecondary::new(Arc::clone(&bar), Arc::clone(&module));
    let q = trs.resolution();
    q.compute_through_bidegree(Bidegree::s_t(4, t_max));
    let qsec = trs.secondary();
    qsec.initialize_homotopies();
    qsec.compute_composites();
    let s = |d: i32, i: usize| algebra.basis_element_to_string(d, i);
    let ws = |w: &Word| w.iter().map(|&(d, i)| s(d, i)).collect::<Vec<_>>().join("|");
    let q0 = q.module(0);
    // ε(A_Q([a1|a2|a3] ⊗ b)) ∈ M
    println!("word,b_deg,b_idx,c_deg,c_idx");
    for d in 3..=t_max {
        for (wi, w) in bar.words[3][d as usize].iter().enumerate() {
            for bd in module.min_degree()..=top {
                let t = d + bd;
                if t > t_max || t - 1 > top {
                    continue;
                }
                for bi in 0..module.dimension(bd) {
                    let Some(gi) = q.q_gen_index(3, t, d, wi, bi) else { continue };
                    let inter = qsec.compute_intermediate(BidegreeGenerator::s_t(3, t, gi));
                    let mut out = FpVector::new(fp::prime::TWO, module.dimension(t - 1));
                    for (i, c) in inter.iter_nonzero() {
                        let og = q0.index_to_op_gen(t - 1, i);
                        // Q_0's generators in degree e are [] ⊗ m, m ∈ M_e
                        if og.operation_degree == 0 {
                            out.add_basis_element(og.generator_index, c);
                        } else {
                            module.act_on_basis(
                                out.as_slice_mut(),
                                c,
                                og.operation_degree,
                                og.operation_index,
                                og.generator_degree,
                                og.generator_index,
                            );
                        }
                    }
                    for (ci, _) in out.iter_nonzero() {
                        println!("{},{bd},{bi},{},{ci}", ws(w), t - 1);
                    }
                }
            }
        }
    }
    Ok(())
}

