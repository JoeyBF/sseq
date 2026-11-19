//! Exports the Adams E2 data of the sphere and of two-cell complexes, for differential propagation.
//!
//! Everything is computed from one minimal resolution `P` of `F_2`, so all bases agree: `Ext(S)` is
//! `P`'s generators, and `Ext(M)` for a finite module `M` is the cohomology of `Hom_A(P ⊗ M, F_2)`
//! (the untwisted tensor resolution, [`tensor_resolution_products`]).
//!
//! For each two-cell complex `C = S ∪ e^{k+1}` (cohomology `x_0` in degree 0, `x_k` in degree `k`)
//! it writes the action of `Ext(S)` on `Ext(C)` and the maps induced by the bottom-cell inclusion
//! `i: S → C` and the top-cell collapse `p: C → S^{k+1}`. The untwisting `Q(P ⊗ M) ≅ ⊕ Σ^{d_i} M`
//! is natural in `M`, so a module map `φ` acts on cochains generator by generator: `i_*` sends the
//! generator cochain `g^*` to `(g, x_0)^*`, and `p_*` reads the `(g, x_k)` coordinate of a cocycle.
//!
//! Files (bidegrees are `stem,Adams_filtration`; vectors as in the stem 256 data set):
//! * `rank.csv`, `products.csv`: the sphere, in the ring-export format.
//! * `<M>_rank.csv`; `<M>_action.csv`: `S-class,M-class,product in M`;
//!   `<M>_istar.csv`: `S-class,image in M`; `<M>_pstar.csv`: `M-class,image in S (stem - k)`.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use algebra::{
    Algebra,
    module::{FDModule, Module},
};
use anyhow::Context;
use ext::{
    chain_complex::ChainComplex,
    ext_algebra::{Cochain, ExtAlgebra, tensor_resolution_products},
    utils::{construct_standard, load_module_json},
};
use fp::vector::FpVector;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

/// A vector of `F_2` coefficients as in the data set.
fn vector(v: impl Iterator<Item = u32>) -> String {
    let e: Vec<String> = v.map(|c| c.to_string()).collect();
    if e.len() == 1 {
        format!("[{}]", e[0])
    } else {
        format!("\"[{}]\"", e.join(", "))
    }
}

fn csv(dir: &std::path::Path, name: &str, header: &str) -> anyhow::Result<BufWriter<File>> {
    let mut w = BufWriter::new(File::create(dir.join(name))?);
    writeln!(w, "{header}")?;
    Ok(w)
}

/// The position of the cochain `(g, α)` in the cochain basis of `Hom_A(P_s ⊗ M, k)_t`, which lists
/// the generators of `P_s` by degree and then the `M`-basis in the complementary degree (the order
/// of `TensorResolutionDifferential`).
fn cochain_index<M: Module>(
    gens: impl Iterator<Item = (i32, usize)>,
    module: &M,
    b: Bidegree,
    g: (i32, usize),
    alpha: usize,
) -> usize {
    let mut idx = 0;
    for (d, i) in gens {
        let m = module.dimension(b.t() - d);
        if (d, i) == g {
            assert!(alpha < m);
            return idx + alpha;
        }
        idx += m;
    }
    panic!("generator {g:?} not in P_{} through t = {}", b.s(), b.t())
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let dir: PathBuf = query::raw("Output directory", |s| Ok::<_, anyhow::Error>(s.into()));
    let max_n: i32 = query::with_default("Max stem", "60", str::parse);
    let max_s: i32 = query::with_default("Max filtration", "30", str::parse);
    let specs: String = query::with_default(
        "Two-cell complexes (name:k)",
        "C2:1,Ceta:2,Cnu:4,Csigma:8",
        |s| Ok::<_, anyhow::Error>(s.to_string()),
    );
    // The sphere's self-maps are shared between the Ext algebras through this directory (each
    // algebra keeps its own in-memory cache, so without it every complex rebuilds them).
    let save: String = query::with_default("Save directory (- for none)", "-", |s| {
        Ok::<_, anyhow::Error>(s.to_string())
    });
    let save = (save != "-").then(|| PathBuf::from(save));
    // With the sphere's indecomposables (a ring export's `indecomposables.csv`), only their action
    // is computed and the sphere's own products are left to the ring export.
    let inds_path: String = query::with_default("S indecomposables (- for all products)", "-", |s| {
        Ok::<_, anyhow::Error>(s.to_string())
    });
    std::fs::create_dir_all(&dir)?;

    let t_max = max_n + max_s + 2;
    let res = Arc::new(construct_standard::<false, _, _>("S_2", save)?);
    res.compute_through_bidegree(Bidegree::s_t(max_s + 2, t_max));
    res.algebra().compute_basis(t_max + 1);
    let algebra = res.algebra();

    let in_range = |b: Bidegree| b.n() >= 0 && b.s() >= 0 && b.n() <= max_n && b.s() <= max_s;
    let bidegrees: Vec<Bidegree> = (0..=max_s)
        .flat_map(|s| (0..=max_n).map(move |n| Bidegree::n_s(n, s)))
        .collect();

    let load = |name: &str| -> anyhow::Result<Arc<FDModule<_>>> {
        let json = load_module_json(name).with_context(|| format!("loading {name}"))?;
        let m = FDModule::from_json(Arc::clone(&algebra), &json)?;
        m.compute_basis(t_max);
        Ok(Arc::new(m))
    };

    // The sphere, through the same engine with M = k so its basis is P's generators.
    let ext_s = tensor_resolution_products(Arc::clone(&res), load("S_2")?);
    let dim_s = |b: Bidegree| ext_s.cohomology_dimension(b).unwrap_or(0);
    let mut w = csv(&dir, "rank.csv", "stem,Adams_filtration,rank")?;
    for &b in &bidegrees {
        if dim_s(b) > 0 {
            writeln!(w, "{},{},{}", b.n(), b.s(), dim_s(b))?;
        }
    }
    let header = "stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,product";

    /// Writes the action of `Ext(S)` on `ext` (rows: S-class, M-class, product), in parallel over
    /// the bidegrees of `ext`.
    fn action<CC: ext::chain_complex::FreeChainComplex + ext::chain_complex::AugmentedChainComplex>(
        ext: &ExtAlgebra<CC>,
        bidegrees: &[Bidegree],
        dim_s: &(dyn Fn(Bidegree) -> usize + Sync),
        in_range: &(dyn Fn(Bidegree) -> bool + Sync),
        w: &mut BufWriter<File>,
    ) -> anyhow::Result<usize> {
        use maybe_rayon::prelude::*;
        // Extend every generator's self-map to the whole range first, each map on its own thread:
        // extending one map from two threads at once races in its `OnceVec`s.
        let top = bidegrees.iter().copied().max_by_key(|b| (b.s(), b.n())).unwrap();
        let gens: Vec<BidegreeGenerator> = bidegrees
            .iter()
            .flat_map(|&b| (0..dim_s(b)).map(move |j| BidegreeGenerator::new(b, j)))
            .collect();
        gens.into_maybe_par_iter().for_each(|g| {
            ext.unit_self_map(g).extend_through_stem(top);
        });
        let chunks: Vec<Vec<String>> = bidegrees
            .to_vec()
            .into_maybe_par_iter()
            .map(|c| {
                let mut out = vec![];
                let dc = ext.cohomology_dimension(c).unwrap_or(0);
                for x in 0..dc {
                    let xe = ext.generator(BidegreeGenerator::new(c, x));
                    for &b in bidegrees {
                        let t = b + c;
                        if dim_s(b) == 0
                            || !in_range(t)
                            || ext.cohomology_dimension(t).unwrap_or(0) == 0
                        {
                            continue;
                        }
                        let Some(m) = ext.multiply_into(&xe, b) else { continue };
                        for j in 0..m.rows() {
                            let row = m.row(j);
                            if row.iter().any(|v| v != 0) {
                                out.push(format!(
                                    "{},{},{j},{},{},{x},{}",
                                    b.n(),
                                    b.s(),
                                    c.n(),
                                    c.s(),
                                    vector(row.iter())
                                ));
                            }
                        }
                    }
                }
                out
            })
            .collect();
        let mut rows = 0;
        for chunk in chunks {
            for l in chunk {
                writeln!(w, "{l}")?;
                rows += 1;
            }
        }
        Ok(rows)
    }

    let top = Bidegree::n_s(max_n, max_s);
    let inds: Option<Vec<BidegreeGenerator>> = (inds_path != "-").then(|| {
        std::fs::read_to_string(&inds_path)
            .expect("indecomposables file")
            .lines()
            .skip(1)
            .filter_map(|l| {
                let v: Vec<&str> = l.split(',').collect();
                let (n, s, i): (i32, i32, usize) =
                    (v[0].parse().ok()?, v[1].parse().ok()?, v[2].parse().ok()?);
                (n <= max_n && s <= max_s && i < dim_s(Bidegree::n_s(n, s)))
                    .then(|| BidegreeGenerator::new(Bidegree::n_s(n, s), i))
            })
            .collect()
    });
    if inds.is_none() {
        let mut w = csv(&dir, "products.csv", header)?;
        let n = action(&ext_s, &bidegrees, &dim_s, &in_range, &mut w)?;
        eprintln!("S: {n} products");
    }

    /// Writes the action of the given sphere classes (indecomposables) on `ext`, by cupping each
    /// cocycle with the classes' self-maps.
    fn ind_action<
        CC: ext::chain_complex::FreeChainComplex + ext::chain_complex::AugmentedChainComplex,
    >(
        ext: &ExtAlgebra<CC>,
        inds: &[BidegreeGenerator],
        bidegrees: &[Bidegree],
        in_range: &(dyn Fn(Bidegree) -> bool + Sync),
        top: Bidegree,
        w: &mut BufWriter<File>,
    ) -> anyhow::Result<usize> {
        use maybe_rayon::prelude::*;
        let cup = std::sync::Arc::clone(ext.cup().expect("a cup product"));
        // one thread per self-map: extending one map from two threads races in its `OnceVec`s
        inds.to_vec().into_maybe_par_iter().for_each(|g| {
            ext.unit_self_map(g).extend_through_stem(top);
        });
        let dim = |b: Bidegree| ext.cohomology_dimension(b).unwrap_or(0);
        let chunks: Vec<Vec<String>> = bidegrees
            .to_vec()
            .into_maybe_par_iter()
            .map(|c| {
                let mut out = vec![];
                let dc = dim(c);
                if dc == 0 {
                    return out;
                }
                let lifts: Vec<Cochain> = (0..dc)
                    .map(|x| ext.lift(&ext.generator(BidegreeGenerator::new(c, x))))
                    .collect();
                for g in inds {
                    let t = c + g.degree();
                    if !in_range(t) || dim(t) == 0 {
                        continue;
                    }
                    let f = ext.unit_self_map(*g);
                    let mat = cup.cup_matrix(&f, c.s(), c.t(), g.degree());
                    for (x, l) in lifts.iter().enumerate() {
                        let mut v = FpVector::new(ext.prime(), ext.cochain_dimension(t));
                        mat.apply(v.as_slice_mut(), 1, l.vec());
                        let cls = ext.project(&Cochain::new(t, v));
                        if !cls.vec().is_zero() {
                            out.push(format!(
                                "{},{},{},{},{},{x},{}",
                                g.degree().n(),
                                g.degree().s(),
                                g.idx(),
                                c.n(),
                                c.s(),
                                vector(cls.vec().iter())
                            ));
                        }
                    }
                }
                out
            })
            .collect();
        let mut rows = 0;
        for chunk in chunks {
            for l in chunk {
                writeln!(w, "{l}")?;
                rows += 1;
            }
        }
        Ok(rows)
    }

    for spec in specs.split(',') {
        let (name, k) = spec.split_once(':').context("name:k")?;
        let k: i32 = k.parse()?;
        let module = load(name)?;
        assert_eq!((module.dimension(0), module.dimension(k)), (1, 1), "{name} is not S ∪ e^{k}");
        let ext_m = tensor_resolution_products(Arc::clone(&res), Arc::clone(&module));
        let dim_m = |b: Bidegree| ext_m.cohomology_dimension(b).unwrap_or(0);

        let mut w = csv(&dir, &format!("{name}_rank.csv"), "stem,Adams_filtration,rank")?;
        for &b in &bidegrees {
            if dim_m(b) > 0 {
                writeln!(w, "{},{},{}", b.n(), b.s(), dim_m(b))?;
            }
        }
        let n = if let Some(inds) = &inds {
            let mut w = csv(&dir, &format!("{name}_indaction.csv"), header)?;
            ind_action(&ext_m, inds, &bidegrees, &in_range, top, &mut w)?
        } else {
            let mut w = csv(&dir, &format!("{name}_action.csv"), header)?;
            action(&ext_m, &bidegrees, &dim_s, &in_range, &mut w)?
        };

        // i_*: g^* ↦ (g, x_0)^*, then project
        let mut wi = csv(&dir, &format!("{name}_istar.csv"), "stem,Adams_filtration,index,image")?;
        // p_*: read the (g, x_k) coordinates of a cocycle representative
        let mut wp = csv(&dir, &format!("{name}_pstar.csv"), "stem,Adams_filtration,index,image")?;
        for &b in &bidegrees {
            let (ds, dm) = (dim_s(b), dim_m(b));
            if ds > 0 && dm > 0 {
                let gens: Vec<(i32, usize)> = (0..ds).map(|i| (b.t(), i)).collect();
                for (i, &g) in gens.iter().enumerate() {
                    let mut v = FpVector::new(ext_m.prime(), ext_m.cochain_dimension(b));
                    v.set_entry(cochain_index(res.module(b.s()).iter_gens(b.t()), &*module, b, g, 0), 1);
                    let img = ext_m.project(&Cochain::new(b, v));
                    if !img.vec().is_zero() {
                        writeln!(wi, "{},{},{i},{}", b.n(), b.s(), vector(img.vec().iter()))?;
                    }
                }
            }
            let bt = Bidegree::n_s(b.n() - k, b.s());
            if dm > 0 && in_range(bt) && dim_s(bt) > 0 {
                for x in 0..dm {
                    let cocycle = ext_m.lift(&ext_m.generator(BidegreeGenerator::new(b, x)));
                    let mut img = FpVector::new(ext_s.prime(), dim_s(bt));
                    for j in 0..dim_s(bt) {
                        // the S generator j at bt is the P_s generator (bt.t(), j); its top-cell
                        // partner in Hom(P ⊗ M) sits at internal degree bt.t() + k = b.t()
                        let idx = cochain_index(
                            res.module(b.s()).iter_gens(b.t()),
                            &*module,
                            b,
                            (bt.t(), j),
                            module.dimension(k) - 1,
                        );
                        img.set_entry(j, cocycle.vec().entry(idx));
                    }
                    if !img.is_zero() {
                        writeln!(wp, "{},{},{x},{}", b.n(), b.s(), vector(img.iter()))?;
                    }
                }
            }
        }
        eprintln!("{name}: {n} action rows");
    }
    Ok(())
}
