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
) -> LiftOut {
    let tgt_alg = tgt.alg();
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
            next[t as usize] = ys;
        }
        prev = next;
        eprintln!("level s = {ss} done");
    }
    LiftOut {
        coeffs,
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

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
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
    let iota = lift(&r, &target, 0, 0, &init, max_n, max_s, verify);
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
            let prod = lift(&r, &r, ys, yt, &init, max_n, max_s - ys, verify);
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
