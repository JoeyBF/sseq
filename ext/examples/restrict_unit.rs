//! The map on Adams E2 pages induced by a unit map `S -> E` whose cohomology is induced up from a
//! sub-Hopf algebra `B = A(n)` of the Steenrod algebra: `H^*(E) = A ⊗_B M`.
//!
//! The unit induces `A ⊗_B M -> F_2`, adjoint to the `B`-module map `φ: M -> F_2` that sends the
//! bottom class to 1. On `E_2` it is
//! ```text
//!   Ext_A(F_2, F_2) --restrict--> Ext_B(F_2, F_2) --φ^*--> Ext_B(M, F_2) = Ext_A(H^*E, F_2).
//! ```
//! At chain level: `P` is the minimal `A`-resolution of `F_2` (a Nassau save, loaded only), `R` is
//! the minimal `B`-resolution of `M` (a Nassau save over `B`, read raw). `P` restricted to `B` is a
//! free `B`-resolution, so `φ` lifts to a `B`-chain map `g: R -> P`, built generator by generator:
//! `g_0(r) = φ(ε(r)) · e_0`, and for `r ∈ R_s`, `g(r)` solves `d_P g(r) = g(d_R r)` (the `B`-basis
//! monomials of `d_R r` read as the same Milnor monomials in `A`). Minimality of both resolutions
//! makes Hom into `F_2` have zero differential, so for `x` dual to the generator `e` of `P_s`,
//! `ι(x)(r) = ` coefficient of `1·e` in `g(r)`.
//!
//! For `tmf`, `B = A(2)` and `M = F_2`; for `j`, `B = A(3)` and `M` is the module of the save.
//!
//! Usage: `restrict_unit <P save> <R save> <max n> <max s> <out prefix> [products]`. Writes
//! `<out>.csv` (stem,Adams_filtration,index,image), `<out>_basis.csv` (the generators of `R`),
//! `<out>_timing.csv`, and prints the `h_i`-linearity checks. With `products`, also writes the
//! Yoneda products in `Ext_B(M, F_2)` by `Ext_B(F_2, F_2)` classes listed in `RESTRICT_PRODUCTS`
//! (only for `M = F_2`, where `R` is also the target).
//!
//! `restrict_unit ring <B save> <max n> <max s> <out dir>` exports `Ext_B(F_2, F_2)` as a ring.
//! `restrict_unit cofibers <B save> <max n> <max s> <ring dir> <out dir> [C2:1,Ceta:2,Cnu:4]
//! [box n] [box s]` exports `Ext_B(H^*C)` for the two-cell complexes `C = S ∪_{h_j} e^{k+1}` as
//! modules over that ring export, in its basis and in the format of `cofiber_export`; the untwisted
//! tensor complex over the minimal `R` needs only the chain maps of the ring's indecomposables.
//! Check the result with `restrict_unit_validate_cofibers.py` (long exact sequence) and
//! `restrict_unit_validate_cofibers_br.py` (Bruner-Rognes Tables 1.5-1.7).

use std::{
    collections::BTreeMap,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use algebra::{
    Algebra, AlgebraType, GeneratedAlgebra, MilnorAlgebra,
    milnor_algebra::MilnorProfile,
    module::{
        FDModule, FreeModule, Module,
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
    },
};
use anyhow::{Context, ensure};
use byteorder::{LittleEndian, ReadBytesExt};
use ext::{
    chain_complex::ChainComplex,
    save::{SaveKind, ZarrSaveStore},
    utils::{QueryModuleResolution, construct},
};
use fp::{matrix::Matrix, prime::TWO, vector::FpVector};
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

type FM = FreeModule<MilnorAlgebra>;

/// Nanoseconds spent building `g(d r)` (multiplication), solving (quasi-inverse), and verifying.
static NS_MULT: AtomicU64 = AtomicU64::new(0);
static NS_SOLVE: AtomicU64 = AtomicU64::new(0);
static NS_VERIFY: AtomicU64 = AtomicU64::new(0);

fn add_ns(c: &AtomicU64, since: Instant) {
    c.fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
}
type Hom = FreeModuleHomomorphism<FM>;

/// A minimal resolution over a sub-algebra `B`, read from a Nassau save store.
struct SubRes {
    alg: Arc<MilnorAlgebra>,
    modules: Vec<Arc<FM>>,
    /// `d[s]: modules[s] -> modules[s - 1]`; `d[0]` is unused.
    d: Vec<Arc<Hom>>,
    /// For `R_0` generators: the value of `φ ∘ ε` (1 on the generator over the bottom class).
    phi: BTreeMap<(i32, usize), u32>,
    max_n: i32,
}

/// A complex we can lift into.
trait Target: Sync {
    fn module(&self, s: i32) -> Arc<FM>;
    fn alg(&self) -> Arc<MilnorAlgebra>;
    /// For each input `z` in `module(s - 1)` of degree `t`, some `y` in `module(s)` with `d y = z`.
    fn solve(&self, s: i32, t: i32, inputs: &[FpVector]) -> Vec<FpVector>;
    fn d(&self, s: i32) -> Arc<Hom>;
}

struct PTarget(Arc<QueryModuleResolution>);

impl Target for PTarget {
    fn module(&self, s: i32) -> Arc<FM> {
        self.0.module(s)
    }

    fn alg(&self) -> Arc<MilnorAlgebra> {
        self.0.algebra()
    }

    fn solve(&self, s: i32, t: i32, inputs: &[FpVector]) -> Vec<FpVector> {
        let dim = self.0.module(s).dimension(t);
        let mut results = vec![FpVector::new(TWO, dim); inputs.len()];
        assert!(
            self.0
                .apply_quasi_inverse(&mut results, Bidegree::s_t(s, t), inputs),
            "no quasi-inverse at (s, t) = ({s}, {t})"
        );
        results
    }

    fn d(&self, s: i32) -> Arc<Hom> {
        self.0.differential(s)
    }
}

impl Target for SubRes {
    fn module(&self, s: i32) -> Arc<FM> {
        Arc::clone(&self.modules[s as usize])
    }

    fn alg(&self) -> Arc<MilnorAlgebra> {
        Arc::clone(&self.alg)
    }

    fn solve(&self, s: i32, t: i32, inputs: &[FpVector]) -> Vec<FpVector> {
        let d = &self.d[s as usize];
        let dim = self.modules[s as usize].dimension(t);
        inputs
            .iter()
            .map(|z| {
                let mut y = FpVector::new(TWO, dim);
                assert!(d.apply_quasi_inverse(y.as_slice_mut(), t, z.as_slice()));
                y
            })
            .collect()
    }

    fn d(&self, s: i32) -> Arc<Hom> {
        Arc::clone(&self.d[s as usize])
    }
}

fn profile_of(store_path: &std::path::Path) -> anyhow::Result<(MilnorProfile, serde_json::Value)> {
    let attrs = ZarrSaveStore::read_root_attributes(store_path)?;
    let spec = attrs
        .get("module_spec")
        .cloned()
        .context("store has no module_spec")?;
    let prof = &spec["algebra"]["profile"];
    let p_part: Vec<_> = prof["p_part"]
        .as_array()
        .context("no p_part in profile")?
        .iter()
        .map(|v| v.as_u64().unwrap() as _)
        .collect();
    let profile = MilnorProfile {
        truncated: prof["truncated"].as_bool().unwrap_or(false),
        q_part: !0,
        p_part,
    };
    Ok((profile, spec))
}

/// Number of `B`-module generators of `M` in each degree, and the index of the one over the bottom
/// class (degree 0, which must be one-dimensional).
fn module_generators(
    alg: &Arc<MilnorAlgebra>,
    spec: &serde_json::Value,
) -> anyhow::Result<Vec<usize>> {
    let m = FDModule::from_json(Arc::clone(alg), spec)?;
    let top = m.max_degree().context("module is not bounded")?;
    ensure!(
        m.min_degree() == 0 && m.dimension(0) == 1,
        "M must be F_2 in degree 0"
    );
    alg.compute_basis(top);
    let mut counts = Vec::new();
    for t in 0..=top {
        let dim = m.dimension(t);
        let mut rows = Vec::new();
        for d in 1..=t {
            for op in alg.generators(d) {
                for i in 0..m.dimension(t - d) {
                    let mut v = FpVector::new(TWO, dim);
                    m.act_on_basis(v.as_slice_mut(), 1, d, op, t - d, i);
                    rows.push(v);
                }
            }
        }
        let mut mat = Matrix::new(TWO, rows.len(), dim);
        for (i, r) in rows.iter().enumerate() {
            mat.row_mut(i).assign(r.as_slice());
        }
        let rank = if rows.is_empty() { 0 } else { mat.row_reduce() };
        counts.push(dim - rank);
    }
    Ok(counts)
}

impl SubRes {
    /// Load a resolution over `B` through stem `max_n` and filtration `max_s`.
    fn load(path: &std::path::Path, max_n: i32, max_s: i32) -> anyhow::Result<Self> {
        let (profile, spec) = profile_of(path)?;
        let alg = Arc::new(MilnorAlgebra::new_with_profile(TWO, profile, false));
        let max_t = max_n + max_s + 1;
        alg.compute_basis(max_t);

        let gens0: Vec<usize> = if spec["gens"].as_object().map(|g| g.len()) == Some(1) {
            vec![1]
        } else {
            module_generators(&alg, &spec)?
        };
        eprintln!(
            "R_0 generators by degree: {:?}",
            gens0
                .iter()
                .enumerate()
                .filter(|(_, c)| **c > 0)
                .collect::<Vec<_>>()
        );

        let store = ZarrSaveStore::create(path)?;
        let mut modules: Vec<Arc<FM>> = Vec::new();
        let mut d: Vec<Arc<Hom>> = Vec::new();
        let mut phi = BTreeMap::new();

        let m0 = Arc::new(FM::new(Arc::clone(&alg), "R0".into(), 0));
        for t in 0..=max_n {
            m0.add_generators(t, gens0.get(t as usize).copied().unwrap_or(0), None);
        }
        m0.compute_basis(max_n + 1);
        phi.insert((0, 0), 1);
        modules.push(Arc::clone(&m0));
        d.push(Arc::new(Hom::new(Arc::clone(&m0), Arc::clone(&m0), 0)));

        for s in 1..=max_s {
            let prev = Arc::clone(&modules[(s - 1) as usize]);
            let m = Arc::new(FM::new(Arc::clone(&alg), format!("R{s}"), 0));
            let hom = Arc::new(Hom::new(Arc::clone(&m), Arc::clone(&prev), 0));
            for t in 0..=(max_n + s) {
                prev.compute_basis(t);
                let target_dim = prev.dimension(t);
                let mut rows = Vec::new();
                if t >= s {
                    let data = store
                        .read(SaveKind::NassauDifferential, Bidegree::s_t(s, t))?
                        .with_context(|| format!("R save lacks (s, t) = ({s}, {t})"))?;
                    let mut f = std::io::Cursor::new(data);
                    let num = f.read_u64::<LittleEndian>()? as usize;
                    let saved_dim = f.read_u64::<LittleEndian>()? as usize;
                    for _ in 0..num {
                        let raw = FpVector::from_bytes(TWO, saved_dim, &mut f)?;
                        let mut row = FpVector::new(TWO, target_dim);
                        for (i, c) in raw.iter_nonzero() {
                            ensure!(
                                i < target_dim,
                                "row entry {i} beyond R_{} dim {target_dim} in degree {t}",
                                s - 1
                            );
                            row.set_entry(i, c);
                        }
                        rows.push(row);
                    }
                }
                m.add_generators(t, rows.len(), None);
                m.compute_basis(t);
                hom.add_generators_from_rows(t, rows);
            }
            m.compute_basis(max_n + s + 1);
            modules.push(m);
            d.push(hom);
        }
        // No generators are added from here on, so higher degrees are safe to lay out.
        alg.compute_basis(max_t + 2);
        for m in &modules {
            m.compute_basis(max_t + 2);
        }
        Ok(Self {
            alg,
            modules,
            d,
            phi,
            max_n,
        })
    }

    fn gens(&self, s: i32, t: i32) -> usize {
        self.modules[s as usize].number_of_gens_in_degree(t)
    }

    /// Check `d d = 0` on every generator.
    fn check_dd(&self) -> usize {
        let mut checked = 0;
        for s in 2..self.modules.len() as i32 {
            let m = &self.modules[s as usize];
            for t in s..=(self.max_n + s) {
                for i in 0..m.number_of_gens_in_degree(t) {
                    let dx = self.d[s as usize].output(t, i);
                    let mut ddx = FpVector::new(TWO, self.modules[(s - 2) as usize].dimension(t));
                    self.d[(s - 1) as usize].apply(ddx.as_slice_mut(), 1, t, dx.as_slice());
                    assert!(ddx.is_zero(), "d^2 != 0 at s={s} t={t} gen {i}");
                    checked += 1;
                }
            }
        }
        checked
    }

    /// Compute quasi-inverses (classically: these are small) so that we can lift into `self`.
    fn prepare_target(&self, max_s: i32) {
        (1..=max_s).into_maybe_par_iter().for_each(|s| {
            self.d[s as usize].compute_auxiliary_data_through_degree(self.max_n + s);
        });
    }
}

/// Coefficient of each generator `1·e_i` of `module` in degree `t` in `v`.
fn gen_coeffs(module: &FM, t: i32, v: &FpVector) -> Vec<u32> {
    (0..module.number_of_gens_in_degree(t))
        .map(|i| v.entry(module.generator_offset(t, t, i)))
        .collect()
}

struct LiftOut {
    /// `(source s, source t) -> per source generator, coefficients of the target generators at
    /// (s - s_shift, t - t_shift)`.
    coeffs: BTreeMap<(i32, i32), Vec<Vec<u32>>>,
    /// `(source s, source t) -> per entry of `op_degs`, per source generator, coefficients of
    /// `Sq(d) e` for the target generators `e` of degree `t - t_shift - d`.
    ops: BTreeMap<(i32, i32), Vec<Vec<Vec<u32>>>>,
    /// stem of the source -> summed seconds of bidegree work.
    secs: BTreeMap<i32, f64>,
}

/// Lift a chain map `g: src_{s + s_shift} -> tgt_s` lowering internal degree by `t_shift`, starting
/// from `init(t, i)` on the generators of `src_{s_shift}` (a vector in `tgt_0` of degree
/// `t - t_shift`), over source stems `<= max_n` and target filtrations `<= max_s`.
fn lift(
    src: &SubRes,
    tgt: &dyn Target,
    s_shift: i32,
    t_shift: i32,
    init: &(dyn Fn(i32, usize) -> FpVector + Sync),
    max_n: i32,
    max_s: i32,
    verify: bool,
    op_degs: &[i32],
) -> LiftOut {
    let tgt_alg = tgt.alg();
    // the Milnor index of Sq(d), for each d in `op_degs`
    let op_idx: Vec<usize> = op_degs
        .iter()
        .map(|&d| {
            tgt_alg.compute_basis(d);
            (0..tgt_alg.dimension(d))
                .find(|&i| {
                    let e = tgt_alg.basis_element_from_index(d, i);
                    let p: Vec<u32> = e.p_part.iter().collect();
                    e.q_part == 0 && p.first() == Some(&(d as u32)) && p[1..].iter().all(|&x| x == 0)
                })
                .unwrap_or_else(|| panic!("no Sq({d}) in the target algebra"))
        })
        .collect();
    let mut ops = BTreeMap::new();
    let src_alg = &src.alg;
    let mut coeffs = BTreeMap::new();
    let secs: Mutex<BTreeMap<i32, f64>> = Mutex::new(BTreeMap::new());
    let max_s = max_s.min(src.modules.len() as i32 - 1 - s_shift);

    // `prev[t]` = images of the source generators of degree t at the previous level.
    let mut prev: Vec<Vec<FpVector>> = Vec::new();
    for s in 0..=max_s {
        let ss = s + s_shift;
        let src_m = &src.modules[ss as usize];
        let t_range = ss..=(max_n + ss);
        let level: Vec<(i32, Vec<FpVector>, f64)> = t_range
            .clone()
            .into_maybe_par_iter()
            .map(|t| {
                let start = Instant::now();
                let n = src_m.number_of_gens_in_degree(t);
                let tt = t - t_shift;
                if n == 0 || tt < 0 {
                    return (t, vec![FpVector::new(TWO, 0); n], 0.0);
                }
                if s == 0 {
                    let v = (0..n).map(|i| init(t, i)).collect();
                    return (t, v, start.elapsed().as_secs_f64());
                }
                let t_mult = Instant::now();
                let tgt_prev = tgt.module(s - 1);
                let src_prev = &src.modules[(ss - 1) as usize];
                let zs: Vec<FpVector> = (0..n)
                    .map(|i| {
                        let mut z = FpVector::new(TWO, tgt_prev.dimension(tt));
                        let dx = src.d[ss as usize].output(t, i);
                        for (idx, _) in dx.iter_nonzero() {
                            let og = src_prev.index_to_op_gen(t, idx);
                            let gimg = &prev[og.generator_degree as usize][og.generator_index];
                            if gimg.is_zero() {
                                continue;
                            }
                            let elt = src_alg
                                .basis_element_from_index(og.operation_degree, og.operation_index);
                            let op = tgt_alg.basis_element_to_index(&elt);
                            tgt_prev.act(
                                z.as_slice_mut(),
                                1,
                                og.operation_degree,
                                op,
                                og.generator_degree - t_shift,
                                gimg.as_slice(),
                            );
                        }
                        z
                    })
                    .collect();
                add_ns(&NS_MULT, t_mult);
                let t_solve = Instant::now();
                let ys = tgt.solve(s, tt, &zs);
                add_ns(&NS_SOLVE, t_solve);
                let t_verify = Instant::now();
                if verify {
                    let d = tgt.d(s);
                    for (y, z) in ys.iter().zip(&zs) {
                        let mut dy = FpVector::new(TWO, z.len());
                        d.apply(dy.as_slice_mut(), 1, tt, y.as_slice());
                        assert_eq!(&dy, z, "lift failed at target (s, t) = ({s}, {tt})");
                    }
                }
                add_ns(&NS_VERIFY, t_verify);
                (t, ys, start.elapsed().as_secs_f64())
            })
            .collect();

        let mut next: Vec<Vec<FpVector>> = vec![Vec::new(); (max_n + ss + 1) as usize];
        for (t, ys, sec) in level {
            *secs.lock().unwrap().entry(t - ss).or_insert(0.0) += sec;
            let tt = t - t_shift;
            let tgt_m = tgt.module(s);
            let c: Vec<Vec<u32>> = ys
                .iter()
                .map(|y| {
                    if y.is_empty() {
                        vec![
                            0;
                            if tt >= 0 {
                                tgt_m.number_of_gens_in_degree(tt)
                            } else {
                                0
                            }
                        ]
                    } else {
                        gen_coeffs(&tgt_m, tt, y)
                    }
                })
                .collect();
            coeffs.insert((ss, t), c);
            if !op_degs.is_empty() {
                let o: Vec<Vec<Vec<u32>>> = op_degs
                    .iter()
                    .zip(&op_idx)
                    .map(|(&d, &oi)| {
                        let gd = tt - d;
                        let ng = if gd >= 0 {
                            tgt_m.number_of_gens_in_degree(gd)
                        } else {
                            0
                        };
                        ys.iter()
                            .map(|y| {
                                (0..ng)
                                    .map(|i| {
                                        if y.is_empty() {
                                            0
                                        } else {
                                            y.entry(tgt_m.operation_generator_to_index(d, oi, gd, i))
                                        }
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                ops.insert((ss, t), o);
            }
            next[t as usize] = ys;
        }
        prev = next;
        eprintln!("level s = {ss} done");
    }
    LiftOut {
        coeffs,
        ops,
        secs: secs.into_inner().unwrap(),
    }
}

/// `h_i` multiplication `Ext^{s,t} -> Ext^{s+1,t+2^i}` read off a minimal resolution's
/// differential: `h_i x_k = Σ_y c y*` where `c` is the coefficient of `Sq^{2^i} e_k` in `d y`.
fn h_matrix(alg: &MilnorAlgebra, m_s: &FM, m_s1: &FM, d: &Hom, i: u32, t: i32) -> Vec<Vec<u32>> {
    let deg = 1 << i;
    let op = alg.generators(deg);
    let rows = m_s.number_of_gens_in_degree(t);
    let cols = m_s1.number_of_gens_in_degree(t + deg);
    let mut out = vec![vec![0; cols]; rows];
    if op.is_empty() {
        return out;
    }
    for (y, row) in (0..cols).map(|y| (y, d.output(t + deg, y))) {
        for (k, o) in out.iter_mut().enumerate() {
            let idx = m_s.operation_generator_to_index(deg, op[0], t, k);
            if idx < row.len() {
                o[y] = row.entry(idx);
            }
        }
    }
    out
}

fn matmul(a: &[Vec<u32>], b: &[Vec<u32>], inner: usize, cols: usize) -> Vec<Vec<u32>> {
    a.iter()
        .map(|r| {
            (0..cols)
                .map(|j| (0..inner).map(|k| r[k] * b[k][j]).sum::<u32>() % 2)
                .collect()
        })
        .collect()
}

fn fmt_vec(v: &[u32]) -> String {
    format!(
        "\"[{}]\"",
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// A vector in the ring export's notation: `[1]`, or `"[1, 0]"` (quoted when it has a comma).
fn fmt_ring_vec(v: &[u32]) -> String {
    let inner = v
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if v.len() > 1 {
        format!("\"[{inner}]\"")
    } else {
        format!("[{inner}]")
    }
}

/// Names of the Bruner-Rognes generators of Ext_{A(2)}, by (stem, s).
fn br_name(n: i32, s: i32) -> Option<&'static str> {
    Some(match (n, s) {
        (0, 1) => "h0",
        (1, 1) => "h1",
        (3, 1) => "h2",
        (8, 3) => "c0",
        (8, 4) => "w1",
        (12, 3) => "alpha",
        (15, 3) => "beta",
        (14, 4) => "d0",
        (17, 4) => "e0",
        (20, 4) => "g",
        (25, 5) => "gamma",
        (32, 7) => "delta",
        (48, 8) => "w2",
        _ => return None,
    })
}

type Prods = BTreeMap<(i32, i32, usize), Vec<u32>>;

/// Products y * x for `y = x_(yn, ys, yi)` and every basis element `x`, by one chain map on the
/// resolution: `(xn, xs, xk) -> product vector`, zero products omitted.
fn products_by(r: &SubRes, (yn, ys, yi): (i32, i32, usize), max_n: i32, max_s: i32) -> Prods {
    let yt = yn + ys;
    let q0 = Arc::clone(&r.modules[0]);
    let init = |t: i32, i: usize| -> FpVector {
        let mut v = FpVector::new(TWO, q0.dimension(t - yt));
        if t == yt && i == yi {
            v.set_entry(q0.generator_offset(0, 0, 0), 1);
        }
        v
    };
    let prod = lift(r, r, ys, yt, &init, max_n, max_s - ys, false, &[]);
    let mut out = BTreeMap::new();
    for ((ss, t), per_q) in &prod.coeffs {
        let s = ss - ys;
        let tx = t - yt;
        if tx < s {
            continue;
        }
        for k in 0..r.gens(s, tx) {
            let row: Vec<u32> = per_q.iter().map(|c| c[k]).collect();
            if row.iter().any(|&c| c != 0) {
                out.insert((tx - s, s, k), row);
            }
        }
    }
    out
}

fn rank_of(rows: &[Vec<u32>], d: usize) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let mut m = Matrix::new(TWO, rows.len(), d);
    for (i, v) in rows.iter().enumerate() {
        for (j, &c) in v.iter().enumerate() {
            m.row_mut(i).set_entry(j, c);
        }
    }
    m.row_reduce()
}

/// Export Ext_B(F_2, F_2) as a ring: rank.csv, indecomposables.csv, generator_products.csv.
fn ring_export(r: &SubRes, max_n: i32, max_s: i32, dir: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let start = Instant::now();
    r.prepare_target(max_s);
    eprintln!("quasi-inverses in {:.1}s", start.elapsed().as_secs_f64());

    let mut w = BufWriter::new(std::fs::File::create(dir.join("rank.csv"))?);
    writeln!(w, "stem,Adams_filtration,rank")?;
    let mut max_rank = 0;
    for n in 0..=max_n {
        for s in 0..=max_s {
            let d = r.gens(s, n + s);
            if d > 0 {
                max_rank = max_rank.max(d);
                writeln!(w, "{n},{s},{d}")?;
            }
        }
    }
    w.flush()?;
    eprintln!("max rank {max_rank}");

    // Indecomposables by (stem, s): basis elements completing the decomposables.
    let mut gens: Vec<((i32, i32, usize), String, Prods)> = Vec::new();
    for n in 0..=max_n {
        for s in 1..=max_s {
            let d = r.gens(s, n + s);
            if d == 0 {
                continue;
            }
            let mut rows: Vec<Vec<u32>> = Vec::new();
            for ((yn, ys, _), _, pr) in &gens {
                let (xn, xs) = (n - yn, s - ys);
                if xs < 1 || xn < 0 {
                    continue;
                }
                for k in 0..r.gens(xs, xn + xs) {
                    if let Some(v) = pr.get(&(xn, xs, k)) {
                        rows.push(v.clone());
                    }
                }
            }
            let dec = rank_of(&rows, d);
            let mut cur_rank = dec;
            // For delta, prefer the basis element with h0 x != 0 and h1 x != 0 (BR's delta).
            let mut order: Vec<usize> = (0..d).collect();
            if br_name(n, s) == Some("delta") {
                let hx = |hn: i32, k: usize| -> bool {
                    gens.iter()
                        .find(|(y, _, _)| *y == (hn, 1, 0))
                        .is_some_and(|(_, _, pr)| pr.contains_key(&(n, s, k)))
                };
                order.sort_by_key(|&k| !(hx(0, k) && hx(1, k)));
            }
            let mut new = Vec::new();
            for k in order {
                if cur_rank >= d {
                    break;
                }
                let mut e = vec![0; d];
                e[k] = 1;
                rows.push(e);
                let rk = rank_of(&rows, d);
                if rk > cur_rank {
                    cur_rank = rk;
                    new.push(k);
                } else {
                    rows.pop();
                }
            }
            for k in new {
                let name = match br_name(n, s) {
                    Some(nm) => nm.to_string(),
                    None => {
                        eprintln!("WARNING: unexpected indecomposable at ({n},{s}) index {k}");
                        String::new()
                    }
                };
                let t0 = Instant::now();
                let pr = products_by(r, (n, s, k), max_n, max_s);
                eprintln!(
                    "indecomposable x_({n},{s},{k}) {name}: decomposables rank {dec} of {d}; {} \
                     nonzero products in {:.1}s",
                    pr.len(),
                    t0.elapsed().as_secs_f64()
                );
                gens.push(((n, s, k), name, pr));
            }
        }
    }

    let mut w = BufWriter::new(std::fs::File::create(dir.join("indecomposables.csv"))?);
    writeln!(w, "stem,Adams_filtration,index,name")?;
    for ((n, s, k), name, _) in &gens {
        writeln!(w, "{n},{s},{k},{name}")?;
    }
    w.flush()?;

    let mut w = BufWriter::new(std::fs::File::create(dir.join("generator_products.csv"))?);
    writeln!(
        w,
        "stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,product"
    )?;
    let mut count = 0;
    for ((n, s, k), _, pr) in &gens {
        for ((xn, xs, xk), v) in pr {
            writeln!(w, "{n},{s},{k},{xn},{xs},{xk},{}", fmt_ring_vec(v))?;
            count += 1;
        }
    }
    w.flush()?;
    eprintln!(
        "ring export: {} indecomposables, {count} products, {:.1}s",
        gens.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// A vector over `F_2` of dimension at most 64, coordinate `i` in bit `i`.
type Bits = u64;

fn bits_of(v: &[u32]) -> Bits {
    assert!(v.len() <= 64);
    v.iter()
        .enumerate()
        .fold(0, |acc, (i, &c)| acc | (((c & 1) as u64) << i))
}

/// Parses `[1]` or `"[0, 1, 1]"` into bits.
fn parse_bits(s: &str) -> Bits {
    let s = s
        .trim()
        .trim_matches('"')
        .trim_start_matches('[')
        .trim_end_matches(']');
    s.split(',')
        .enumerate()
        .fold(0, |acc, (i, e)| acc | (u64::from(e.trim() == "1") << i))
}

/// `[1]`, or `"[1, 0]"` when the vector has more than one entry (the cofiber export notation).
fn fmt_bits(v: Bits, dim: usize) -> String {
    let e: Vec<u32> = (0..dim).map(|i| ((v >> i) & 1) as u32).collect();
    fmt_ring_vec(&e)
}

/// Fully reduced echelon form of the span of `rows`: `(pivot bit, row)`, the pivot being the
/// row's lowest set bit and clear in every other row.
fn rref_bits(rows: impl IntoIterator<Item = Bits>) -> Vec<(u32, Bits)> {
    let mut out: Vec<(u32, Bits)> = Vec::new();
    for mut r in rows {
        for &(p, b) in &out {
            if (r >> p) & 1 == 1 {
                r ^= b;
            }
        }
        if r == 0 {
            continue;
        }
        let p = r.trailing_zeros();
        for (_, b) in out.iter_mut() {
            if (*b >> p) & 1 == 1 {
                *b ^= r;
            }
        }
        out.push((p, r));
    }
    out.sort_unstable();
    out
}

/// The kernel of the map sending basis vector `i` to `images[i]`.
fn kernel_bits(images: &[Bits]) -> Vec<Bits> {
    let mut kernel = Vec::new();
    // (pivot, image, source), eliminated on the image part
    let mut done: Vec<(u32, Bits, Bits)> = Vec::new();
    for (i, &f0) in images.iter().enumerate() {
        let (mut f, mut src) = (f0, 1 << i);
        for &(p, g, s) in &done {
            if (f >> p) & 1 == 1 {
                f ^= g;
                src ^= s;
            }
        }
        if f == 0 {
            kernel.push(src);
        } else {
            done.push((f.trailing_zeros(), f, src));
        }
    }
    kernel
}

/// XOR of `m[i]` over the set bits `i` of `x`.
fn apply_bits(m: Option<&Vec<Bits>>, x: Bits, what: &str) -> Bits {
    let mut out = 0;
    let mut x = x;
    while x != 0 {
        let i = x.trailing_zeros() as usize;
        x &= x - 1;
        out ^= m.unwrap_or_else(|| panic!("missing {what}"))[i];
    }
    out
}

/// The chain map `g_y: R_{s + y_s} -> R_s` of a class `y` of `Ext_B(F_2, F_2)`, read as the
/// product `x -> y x` and, for each `k` asked for, the `Sq(k)` part of `g_y` that a two-cell
/// module with top class in degree `k` sees.
struct YMap {
    y: (i32, i32, usize),
    /// `(n, s)` of `x` -> per basis element of `Ext^{n,s}`, `y x` over `Ext^{n + y_n, s + y_s}`.
    prod: BTreeMap<(i32, i32), Vec<Bits>>,
    /// `(k, n, s)` -> per generator `e_l` of `R_s` in degree `n + s`, the coefficients of
    /// `Sq(k) e_l` in `g_y(e')` over the generators `e'` of `Ext^{n + y_n + k, s + y_s}`.
    cross: BTreeMap<(i32, i32, i32), Vec<Bits>>,
}

fn ymap(r: &SubRes, y: (i32, i32, usize), ks: &[i32], max_n: i32, max_s: i32) -> YMap {
    let (yn, ys, yi) = y;
    let yt = yn + ys;
    let q0 = Arc::clone(&r.modules[0]);
    let init = |t: i32, i: usize| -> FpVector {
        let mut v = FpVector::new(TWO, q0.dimension(t - yt));
        if t == yt && i == yi {
            v.set_entry(q0.generator_offset(0, 0, 0), 1);
        }
        v
    };
    let out = lift(r, r, ys, yt, &init, max_n, max_s - ys, false, ks);
    let mut prod = BTreeMap::new();
    for ((ss, t), per_q) in &out.coeffs {
        let s = ss - ys;
        let tx = t - yt;
        if tx < s {
            continue;
        }
        let v: Vec<Bits> = (0..r.gens(s, tx))
            .map(|k| {
                per_q
                    .iter()
                    .enumerate()
                    .fold(0, |a, (q, c)| a | ((c[k] as u64) << q))
            })
            .collect();
        prod.insert((tx - s, s), v);
    }
    let mut cross = BTreeMap::new();
    for ((ss, t), per_op) in &out.ops {
        let s = ss - ys;
        for (oi, &k) in ks.iter().enumerate() {
            let tv = t - yt - k;
            if tv < s {
                continue;
            }
            let per_q = &per_op[oi];
            let v: Vec<Bits> = (0..r.gens(s, tv))
                .map(|l| {
                    per_q
                        .iter()
                        .enumerate()
                        .fold(0, |a, (q, c)| a | ((c[l] as u64) << q))
                })
                .collect();
            cross.insert((k, tv - s, s), v);
        }
    }
    YMap { y, prod, cross }
}

/// One bidegree of `Ext_B(H^*C)` for `C = S ∪_{h_j} e^{k+1}` (cohomology `x_0`, `x_k = Sq^k x_0`).
///
/// `R` is minimal, so the untwisted tensor complex `Hom_B(R ⊗ H^*C, F_2)` is `Ext(S) ⊕ Ext(S)[k]`
/// (cochains on `(e, x_0)` and on `(e, x_k)`) with the one differential `(e, x_k)^* ↦ h_j e^*`.
/// The basis: first the bottom-cell classes `e_p^*` for the positions `p` outside the pivots of
/// `im h_j` (in increasing `p`), then the top-cell classes `(κ, x_k)` for the reduced echelon basis
/// `κ` of `ker h_j` (in increasing pivot).
struct CBid {
    img: Vec<(u32, Bits)>,
    free: Vec<u32>,
    ker: Vec<(u32, Bits)>,
}

impl CBid {
    fn dim(&self) -> usize {
        self.free.len() + self.ker.len()
    }

    /// The cocycle `(u on x_0, v on x_k)` of basis element `c`.
    fn rep(&self, c: usize) -> (Bits, Bits) {
        if c < self.free.len() {
            (1 << self.free[c], 0)
        } else {
            (0, self.ker[c - self.free.len()].1)
        }
    }

    /// The class of the cocycle `(u, v)`.
    fn project(&self, mut u: Bits, mut v: Bits) -> Bits {
        let nf = self.free.len();
        let mut out = 0;
        for (i, &(p, kv)) in self.ker.iter().enumerate() {
            if (v >> p) & 1 == 1 {
                v ^= kv;
                out |= 1 << (nf + i);
            }
        }
        assert_eq!(v, 0, "not a cocycle: top-cell part outside ker h_j");
        for &(p, b) in &self.img {
            if (u >> p) & 1 == 1 {
                u ^= b;
            }
        }
        for (c, &p) in self.free.iter().enumerate() {
            if (u >> p) & 1 == 1 {
                out |= 1 << c;
            }
        }
        out
    }
}

/// `y` times the cocycle `(u, v)` at `(n, s)` of the two-cell module with top degree `k`, as a
/// cocycle at `(n + y_n, s + y_s)`: `(y ψ)(e', m) = Σ_l ψ(e_l, χ(b_l) m)` for `g_y(e') = Σ b_l e_l`,
/// and `χ(b) x_0` is the `Sq(k)`-coefficient of `b` times `x_k` (for `k = 4`, `χ(Sq(4)) = Sq(4) +
/// Sq(1, 1)` and `Sq(1, 1) x_0 = 0`).
fn act_cocycle(y: &YMap, k: i32, n: i32, s: i32, (u, v): (Bits, Bits)) -> (Bits, Bits) {
    let u2 = apply_bits(y.prod.get(&(n, s)), u, "product (bottom)")
        ^ apply_bits(y.cross.get(&(k, n - k, s)), v, "Sq(k) cross term");
    let v2 = apply_bits(y.prod.get(&(n - k, s)), v, "product (top)");
    (u2, v2)
}

/// Export `Ext_B(H^*C)` for the two-cell complexes `C = S ∪_{h_j} e^{k+1}` in the format of ext's
/// `cofiber_export` (`<C>_rank.csv`, `<C>_indaction.csv`, `<C>_istar.csv`, `<C>_pstar.csv`,
/// `<C>_unknown.csv`), in the basis of the ring export in `ring_dir` (same save, same generators).
/// With `box_n, box_s > 0` it also writes `<C>_action.csv`: the action of every class of
/// `Ext_B(F_2, F_2)` in stems `<= box_n` and filtrations `<= box_s`, each by its own chain map
/// (zero products included), for the diagram engine's action check.
#[allow(clippy::too_many_arguments)]
fn cofiber_export(
    r: &SubRes,
    max_n: i32,
    max_s: i32,
    ring_dir: &std::path::Path,
    dir: &std::path::Path,
    specs: &[(String, i32)],
    box_n: i32,
    box_s: i32,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let start = Instant::now();
    r.prepare_target(max_s);
    eprintln!("quasi-inverses in {:.1}s", start.elapsed().as_secs_f64());
    let dim_s = |n: i32, s: i32| -> usize {
        if n < 0 || s < 0 || n > max_n || s > max_s + 1 {
            0
        } else {
            r.gens(s, n + s)
        }
    };

    // Same resolution as the ring export: the ranks agree.
    let mut ring_rank = BTreeMap::new();
    for l in std::fs::read_to_string(ring_dir.join("rank.csv"))?
        .lines()
        .skip(1)
    {
        let v: Vec<i32> = l.split(',').map(|x| x.parse().unwrap()).collect();
        ring_rank.insert((v[0], v[1]), v[2] as usize);
    }
    let mut bad = 0;
    for n in 0..=max_n {
        for s in 0..=max_s {
            if ring_rank.get(&(n, s)).copied().unwrap_or(0) != dim_s(n, s) {
                bad += 1;
            }
        }
    }
    ensure!(bad == 0, "{bad} ranks differ from the ring export");

    let inds: Vec<(i32, i32, usize)> =
        std::fs::read_to_string(ring_dir.join("indecomposables.csv"))?
            .lines()
            .skip(1)
            .map(|l| {
                let v: Vec<&str> = l.split(',').collect();
                (
                    v[0].parse().unwrap(),
                    v[1].parse().unwrap(),
                    v[2].parse().unwrap(),
                )
            })
            .collect();
    let ks: Vec<i32> = specs.iter().map(|(_, k)| *k).collect();
    let t0 = Instant::now();
    let ymaps: Vec<YMap> = inds
        .iter()
        .map(|&y| {
            let t = Instant::now();
            let m = ymap(r, y, &ks, max_n, max_s);
            eprintln!("chain map of {y:?} in {:.1}s", t.elapsed().as_secs_f64());
            m
        })
        .collect();
    eprintln!(
        "indecomposable chain maps in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // Our products agree with the ring export's generator products (same basis).
    {
        let mut ring: BTreeMap<((i32, i32, usize), (i32, i32, usize)), Bits> = BTreeMap::new();
        for l in std::fs::read_to_string(ring_dir.join("generator_products.csv"))?
            .lines()
            .skip(1)
        {
            let v: Vec<&str> = l.splitn(7, ',').collect();
            let p = |i: usize| v[i].parse::<i32>().unwrap();
            ring.insert(
                ((p(0), p(1), p(2) as usize), (p(3), p(4), p(5) as usize)),
                parse_bits(v[6]),
            );
        }
        let (mut same, mut diff) = (0, 0);
        for ym in &ymaps {
            for (&(n, s), per_x) in &ym.prod {
                if n + ym.y.0 > max_n || s + ym.y.1 > max_s {
                    continue;
                }
                for (k, &p) in per_x.iter().enumerate() {
                    let theirs = ring.get(&(ym.y, (n, s, k))).copied().unwrap_or(0);
                    if theirs == p {
                        same += 1;
                    } else {
                        diff += 1;
                        if diff < 10 {
                            eprintln!(
                                "product {:?} * ({n},{s},{k}): ours {p:b} ring {theirs:b}",
                                ym.y
                            );
                        }
                    }
                }
            }
        }
        eprintln!("products vs ring export: {same} agree, {diff} differ");
        ensure!(diff == 0, "products differ from the ring export");
    }

    // h_i: Ext^{n,s} -> Ext^{n + 2^i - 1, s + 1}, from the differential
    let hmat = |i: u32, n: i32, s: i32| -> Vec<Bits> {
        if dim_s(n, s) == 0 {
            return vec![];
        }
        h_matrix(
            &r.alg,
            &r.modules[s as usize],
            &r.modules[(s + 1) as usize],
            &r.d[(s + 1) as usize],
            i,
            n + s,
        )
        .iter()
        .map(|row| bits_of(row))
        .collect()
    };

    for (name, k) in specs {
        let k = *k;
        let j = k.trailing_zeros();
        ensure!(k == 1 << j && j <= 2, "{name}: k = {k} is not 1, 2 or 4");
        let t1 = Instant::now();
        let mut cb: BTreeMap<(i32, i32), CBid> = BTreeMap::new();
        for n in 0..=max_n {
            for s in 0..=max_s {
                let d = dim_s(n, s);
                let img = if s >= 1 && n - k + 1 >= 0 {
                    rref_bits(hmat(j, n - k + 1, s - 1))
                } else {
                    vec![]
                };
                let free: Vec<u32> = (0..d as u32)
                    .filter(|&p| img.iter().all(|&(q, _)| q != p))
                    .collect();
                let ker = if n - k >= 0 {
                    rref_bits(kernel_bits(&hmat(j, n - k, s)))
                } else {
                    vec![]
                };
                let c = CBid { img, free, ker };
                if c.dim() > 0 {
                    ensure!(c.dim() <= 64, "{name}: rank {} at ({n},{s})", c.dim());
                    cb.insert((n, s), c);
                }
            }
        }
        let csv = |suffix: &str, header: &str| -> anyhow::Result<BufWriter<std::fs::File>> {
            let mut w = BufWriter::new(std::fs::File::create(
                dir.join(format!("{name}_{suffix}.csv")),
            )?);
            writeln!(w, "{header}")?;
            Ok(w)
        };
        let mut w = csv("rank", "stem,Adams_filtration,rank")?;
        let mut total = 0;
        for (&(n, s), c) in &cb {
            writeln!(w, "{n},{s},{}", c.dim())?;
            total += c.dim();
        }
        w.flush()?;

        let mut w = csv("istar", "stem,Adams_filtration,index,image")?;
        for (&(n, s), c) in &cb {
            for g in 0..dim_s(n, s) {
                let img = c.project(1 << g, 0);
                if img != 0 {
                    writeln!(w, "{n},{s},{g},{}", fmt_bits(img, c.dim()))?;
                }
            }
        }
        w.flush()?;
        let mut w = csv("pstar", "stem,Adams_filtration,index,image")?;
        for (&(n, s), c) in &cb {
            for x in 0..c.dim() {
                let (_, v) = c.rep(x);
                if v != 0 {
                    writeln!(w, "{n},{s},{x},{}", fmt_bits(v, dim_s(n - k, s)))?;
                }
            }
        }
        w.flush()?;

        // `y` times the module vector `v` at `(n, s)`, if the target is in range
        let act = |y: &YMap, n: i32, s: i32, v: Bits| -> Option<Bits> {
            let (tn, ts) = (n + y.y.0, s + y.y.1);
            if tn > max_n || ts > max_s {
                return None;
            }
            let src = cb.get(&(n, s))?;
            let tgt = cb.get(&(tn, ts))?;
            let (mut u, mut w) = (0, 0);
            for x in 0..src.dim() {
                if (v >> x) & 1 == 1 {
                    let (a, b) = act_cocycle(y, k, n, s, src.rep(x));
                    u ^= a;
                    w ^= b;
                }
            }
            Some(tgt.project(u, w))
        };

        let header = "stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,product";
        let mut w = csv("indaction", header)?;
        let mut rows = 0;
        for y in &ymaps {
            for (&(n, s), c) in &cb {
                for x in 0..c.dim() {
                    if let Some(p) = act(y, n, s, 1 << x)
                        && p != 0
                    {
                        let tdim = cb[&(n + y.y.0, s + y.y.1)].dim();
                        writeln!(
                            w,
                            "{},{},{},{n},{s},{x},{}",
                            y.y.0,
                            y.y.1,
                            y.y.2,
                            fmt_bits(p, tdim)
                        )?;
                        rows += 1;
                    }
                }
            }
        }
        w.flush()?;
        // every product in range is known
        let mut w = csv(
            "unknown",
            "stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,reason",
        )?;
        w.flush()?;

        // The action is a module action of a commutative ring: y (z x) = z (y x).
        let (mut ok, mut bad) = (0usize, 0usize);
        for (a, y) in ymaps.iter().enumerate() {
            for z in &ymaps[a + 1..] {
                for (&(n, s), c) in &cb {
                    for x in 0..c.dim() {
                        let yz =
                            act(z, n, s, 1 << x).and_then(|v| act(y, n + z.y.0, s + z.y.1, v));
                        let zy =
                            act(y, n, s, 1 << x).and_then(|v| act(z, n + y.y.0, s + y.y.1, v));
                        match (yz, zy) {
                            (Some(p), Some(q)) if p == q => ok += 1,
                            (Some(_), Some(_)) => {
                                bad += 1;
                                if bad < 10 {
                                    eprintln!(
                                        "{name}: {:?} {:?} ({n},{s},{x}) not commutative",
                                        y.y, z.y
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        eprintln!("{name}: commutativity y(zx) = z(yx): {ok} agree, {bad} differ");
        ensure!(bad == 0, "{name}: action not commutative");
        eprintln!(
            "{name}: {} bidegrees, dim {total}, {rows} indecomposable action rows, {:.1}s",
            cb.len(),
            t1.elapsed().as_secs_f64()
        );

        if box_n > 0 && box_s > 0 {
            let t2 = Instant::now();
            let mut w = csv("action", header)?;
            let ys: Vec<(i32, i32, usize)> = (0..=box_n)
                .flat_map(|n| (1..=box_s).map(move |s| (n, s)))
                .flat_map(|(n, s)| (0..dim_s(n, s)).map(move |i| (n, s, i)))
                .collect();
            let maps: Vec<YMap> = ys
                .into_maybe_par_iter()
                .map(|y| ymap(r, y, &[k], box_n, box_s))
                .collect();
            let mut rows = 0;
            for y in &maps {
                for (&(n, s), c) in &cb {
                    if n + y.y.0 > box_n || s + y.y.1 > box_s {
                        continue;
                    }
                    for x in 0..c.dim() {
                        if let Some(p) = act(y, n, s, 1 << x) {
                            let tdim = cb[&(n + y.y.0, s + y.y.1)].dim();
                            writeln!(
                                w,
                                "{},{},{},{n},{s},{x},{}",
                                y.y.0,
                                y.y.1,
                                y.y.2,
                                fmt_bits(p, tdim)
                            )?;
                            rows += 1;
                        }
                    }
                }
            }
            w.flush()?;
            eprintln!(
                "{name}: action of {} classes in stems <= {box_n}, filtrations <= {box_s}: {rows} \
                 rows, {:.1}s",
                maps.len(),
                t2.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "cofibers") {
        // restrict_unit cofibers <B save> <max n> <max s> <ring dir> <out dir> [name:k,...]
        // [box n] [box s]
        ensure!(
            args.len() >= 7,
            "usage: restrict_unit cofibers <B save> <max n> <max s> <ring dir> <out dir> \
             [name:k,...] [box n] [box s]"
        );
        let max_n: i32 = args[3].parse()?;
        let max_s: i32 = args[4].parse()?;
        let specs: Vec<(String, i32)> = args
            .get(7)
            .map_or("C2:1,Ceta:2,Cnu:4", |s| s.as_str())
            .split(',')
            .map(|sp| {
                let (a, b) = sp.split_once(':').expect("name:k");
                (a.to_string(), b.parse().expect("k"))
            })
            .collect();
        let box_n: i32 = args.get(8).map_or(Ok(0), |s| s.parse())?;
        let box_s: i32 = args.get(9).map_or(Ok(0), |s| s.parse())?;
        // one filtration more, for ker h_j at the top
        let r = SubRes::load(std::path::Path::new(&args[2]), max_n, max_s + 1)?;
        eprintln!("d^2 = 0 on {} generators", r.check_dd());
        return cofiber_export(
            &r,
            max_n,
            max_s,
            std::path::Path::new(&args[5]),
            std::path::Path::new(&args[6]),
            &specs,
            box_n,
            box_s,
        );
    }
    if args.get(1).is_some_and(|a| a == "ring") {
        // restrict_unit ring <B save> <max n> <max s> <out dir>
        ensure!(
            args.len() >= 6,
            "usage: restrict_unit ring <B save> <max n> <max s> <out dir>"
        );
        let max_n: i32 = args[3].parse()?;
        let max_s: i32 = args[4].parse()?;
        let r = SubRes::load(std::path::Path::new(&args[2]), max_n, max_s)?;
        eprintln!("d^2 = 0 on {} generators", r.check_dd());
        return ring_export(&r, max_n, max_s, std::path::Path::new(&args[5]));
    }
    ensure!(
        args.len() >= 6,
        "usage: restrict_unit <P save> <R save> <max n> <max s> <out prefix> [products]"
    );
    let p_path = PathBuf::from(&args[1]);
    let r_path = PathBuf::from(&args[2]);
    let max_n: i32 = args[3].parse()?;
    let max_s: i32 = args[4].parse()?;
    let out = PathBuf::from(&args[5]);
    let products = args.get(6).is_some_and(|a| a == "products");
    let verify = std::env::var_os("RESTRICT_VERIFY").is_some();

    // Refuse to compute anything over A: every differential in range must be saved.
    {
        let store = ZarrSaveStore::create(&p_path)?;
        for s in 1..=max_s {
            for n in 0..=max_n {
                ensure!(
                    store.exists(SaveKind::NassauDifferential, Bidegree::n_s(n, s)),
                    "P save lacks (n, s) = ({n}, {s}); refusing to resolve over A"
                );
            }
        }
    }

    let t0 = Instant::now();
    let res = construct(("S_2", AlgebraType::Milnor), Some(p_path.clone()))?;
    res.compute_through_stem(Bidegree::n_s(max_n, max_s));
    let res = Arc::new(res);
    eprintln!("P loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    let r = SubRes::load(&r_path, max_n, max_s)?;
    eprintln!(
        "R loaded in {:.1}s; d^2 = 0 on {} generators",
        t1.elapsed().as_secs_f64(),
        r.check_dd()
    );

    // ι: g_0(r) = φ(ε r) e_0.
    let p0 = res.module(0);
    let init = |t: i32, i: usize| -> FpVector {
        let mut v = FpVector::new(TWO, p0.dimension(t));
        if t == 0 && r.phi.get(&(t, i)).copied().unwrap_or(0) == 1 {
            v.set_entry(p0.generator_offset(0, 0, 0), 1);
        }
        v
    };
    let target = PTarget(Arc::clone(&res));
    let t2 = Instant::now();
    let iota = lift(&r, &target, 0, 0, &init, max_n, max_s, verify, &[]);
    let lift_secs = t2.elapsed().as_secs_f64();
    eprintln!(
        "lift in {lift_secs:.1}s wall; thread-seconds: multiply {:.1}, solve {:.1}, verify {:.1}",
        NS_MULT.load(Ordering::Relaxed) as f64 * 1e-9,
        NS_SOLVE.load(Ordering::Relaxed) as f64 * 1e-9,
        NS_VERIFY.load(Ordering::Relaxed) as f64 * 1e-9
    );

    // ι as rows over the P generators: iota_rows[(s,t)][k][q].
    let mut csv = BufWriter::new(std::fs::File::create(out.with_extension("csv"))?);
    writeln!(csv, "stem,Adams_filtration,index,image")?;
    let mut iota_rows: BTreeMap<(i32, i32), Vec<Vec<u32>>> = BTreeMap::new();
    let mut nonzero = 0;
    for n in 0..=max_n {
        for s in 0..=max_s {
            let t = n + s;
            let pk = res.module(s).number_of_gens_in_degree(t);
            let cols = r.gens(s, t);
            let per_q = iota.coeffs.get(&(s, t)).cloned().unwrap_or_default();
            let rows: Vec<Vec<u32>> = (0..pk)
                .map(|k| (0..cols).map(|q| per_q[q][k]).collect())
                .collect();
            for (k, row) in rows.iter().enumerate() {
                if row.iter().any(|&c| c != 0) {
                    nonzero += 1;
                }
                writeln!(csv, "{n},{s},{k},{}", fmt_vec(row))?;
            }
            iota_rows.insert((s, t), rows);
        }
    }
    csv.flush()?;

    let mut basis = BufWriter::new(std::fs::File::create(out.with_file_name(format!(
        "{}_basis.csv",
        out.file_name().unwrap().to_string_lossy()
    )))?);
    writeln!(basis, "stem,Adams_filtration,index,t,differential")?;
    for n in 0..=max_n {
        for s in 0..=max_s {
            let t = n + s;
            for i in 0..r.gens(s, t) {
                let dx = if s == 0 {
                    String::new()
                } else {
                    // d(x) as a sum of `Sq(R) x_(n', s-1, i')`.
                    let prev = &r.modules[(s - 1) as usize];
                    r.d[s as usize]
                        .output(t, i)
                        .iter_nonzero()
                        .map(|(idx, _)| {
                            let og = prev.index_to_op_gen(t, idx);
                            let e = r
                                .alg
                                .basis_element_from_index(og.operation_degree, og.operation_index);
                            format!(
                                "Sq{:?} x_({},{},{})",
                                e.p_part.iter().collect::<Vec<_>>(),
                                og.generator_degree - (s - 1),
                                s - 1,
                                og.generator_index
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" + ")
                };
                writeln!(basis, "{n},{s},{i},{t},\"{dx}\"")?;
            }
        }
    }
    basis.flush()?;

    let mut timing = BufWriter::new(std::fs::File::create(out.with_file_name(format!(
        "{}_timing.csv",
        out.file_name().unwrap().to_string_lossy()
    )))?);
    writeln!(timing, "stem,bidegree_seconds")?;
    for (n, sec) in &iota.secs {
        writeln!(timing, "{n},{sec:.3}")?;
    }
    timing.flush()?;

    // h_i products on both sides, as rows "i,stem,s,index,h_i x" (over the basis at (s+1, t+2^i)).
    for (side, name) in [(0, "hA"), (1, "hB")] {
        let mut w = BufWriter::new(std::fs::File::create(out.with_file_name(format!(
            "{}_{name}.csv",
            out.file_name().unwrap().to_string_lossy()
        )))?);
        writeln!(w, "i,stem,Adams_filtration,index,product")?;
        for i in 0..4u32 {
            let deg = 1 << i;
            for s in 0..max_s {
                for n in 0..=max_n {
                    let t = n + s;
                    if t + deg - (s + 1) > max_n {
                        continue;
                    }
                    let h = if side == 0 {
                        h_matrix(
                            &res.algebra(),
                            &res.module(s),
                            &res.module(s + 1),
                            &res.differential(s + 1),
                            i,
                            t,
                        )
                    } else {
                        h_matrix(
                            &r.alg,
                            &r.modules[s as usize],
                            &r.modules[(s + 1) as usize],
                            &r.d[(s + 1) as usize],
                            i,
                            t,
                        )
                    };
                    for (k, row) in h.iter().enumerate() {
                        writeln!(w, "{i},{n},{s},{k},{}", fmt_vec(row))?;
                    }
                }
            }
        }
        w.flush()?;
    }

    // h_i-linearity: ι(h_i x) = h_i ι(x).
    let checks = AtomicU64::new(0);
    let mut failures = 0;
    for i in 0..4u32 {
        let deg = 1 << i;
        for s in 0..max_s {
            for n in 0..=max_n {
                let t = n + s;
                if t + deg - (s + 1) > max_n {
                    continue;
                }
                let ha = h_matrix(
                    &res.algebra(),
                    &res.module(s),
                    &res.module(s + 1),
                    &res.differential(s + 1),
                    i,
                    t,
                );
                let hb = if s + 1 < r.modules.len() as i32 {
                    h_matrix(
                        &r.alg,
                        &r.modules[s as usize],
                        &r.modules[(s + 1) as usize],
                        &r.d[(s + 1) as usize],
                        i,
                        t,
                    )
                } else {
                    continue;
                };
                let i_st = &iota_rows[&(s, t)];
                let i_up = &iota_rows[&(s + 1, t + deg)];
                let lhs = matmul(
                    &ha,
                    i_up,
                    res.module(s + 1).number_of_gens_in_degree(t + deg),
                    r.gens(s + 1, t + deg),
                );
                let rhs = matmul(i_st, &hb, r.gens(s, t), r.gens(s + 1, t + deg));
                checks.fetch_add(ha.len() as u64, Ordering::Relaxed);
                if lhs != rhs {
                    failures += 1;
                    eprintln!("h{i}-linearity FAILS at (n, s) = ({n}, {s})");
                }
            }
        }
    }
    eprintln!(
        "h_i-linearity (i = 0..3): {} classes checked, {failures} failing bidegrees",
        checks.load(Ordering::Relaxed)
    );
    eprintln!("{nonzero} classes with nonzero image");

    if products {
        // Yoneda products y * - on Ext_B(F_2, F_2) for the classes y = x_(n, s, i) listed as
        // "n,s,i;..." in RESTRICT_PRODUCTS. Output: for each y, rows (x, x*y) over R's basis.
        let list = std::env::var("RESTRICT_PRODUCTS").unwrap_or_default();
        r.prepare_target(max_s);
        let mut pcsv = BufWriter::new(std::fs::File::create(out.with_file_name(format!(
            "{}_products.csv",
            out.file_name().unwrap().to_string_lossy()
        )))?);
        writeln!(pcsv, "y_stem,y_s,y_index,x_stem,x_s,x_index,product")?;
        for item in list.split(';').filter(|x| !x.is_empty()) {
            let v: Vec<i32> = item.split(',').map(|x| x.trim().parse().unwrap()).collect();
            let (yn, ys, yi) = (v[0], v[1], v[2] as usize);
            let yt = yn + ys;
            let q0 = Arc::clone(&r.modules[0]);
            let init = |t: i32, i: usize| -> FpVector {
                let mut v = FpVector::new(TWO, q0.dimension(t - yt));
                if t == yt && i == yi {
                    v.set_entry(q0.generator_offset(0, 0, 0), 1);
                }
                v
            };
            let start = Instant::now();
            let prod = lift(&r, &r, ys, yt, &init, max_n, max_s - ys, verify, &[]);
            for ((ss, t), per_q) in &prod.coeffs {
                let s = ss - ys;
                let tx = t - yt;
                if tx < s {
                    continue;
                }
                let xk = r.gens(s, tx);
                for k in 0..xk {
                    let row: Vec<u32> = per_q.iter().map(|c| c[k]).collect();
                    writeln!(pcsv, "{yn},{ys},{yi},{},{s},{k},{}", tx - s, fmt_vec(&row))?;
                }
            }
            eprintln!(
                "products by x_({yn},{ys},{yi}) in {:.1}s",
                start.elapsed().as_secs_f64()
            );
        }
        pcsv.flush()?;
    }

    Ok(())
}
