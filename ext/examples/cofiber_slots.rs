//! The E2 page of the two-cell complexes `C_j = S ∪_{h_j} e^{2^j}` as modules over the sphere's,
//! read from a saved resolution and W. Lin's saved chain maps; nothing is lifted.
//!
//! On the minimal resolution `P`, `Hom(P ⊗ H^*C_j) = Hom(P)·bottom ⊕ Hom(P)·top` with coboundary
//! `z·top ↦ (h_j z)·bottom`, so `Ext(C_j) = coker(h_j) ⊕ ker(h_j)` with no choices. The cup product
//! with the chain map `f_g` of a sphere class `g` is diagonal on the cells (ordinary products, taken
//! from the ring export) except for one term,
//!
//! `g·(z·top) = (z g)·top + (z ⋆ g)·bottom`, `(z ⋆ g)_l = Σ_i z_i [Sq(2^j)] (f_g(e_l))_{x_i}`,
//!
//! the `Sq(2^j)` coefficient on `z`'s generators of `f_g`, read from `f_g`'s saved steps (`χ` fixes
//! that coefficient). A pair whose step was never lifted (outside Lin's corner, or `g = h_i`, which
//! has no map) is reported as unknown, never as zero.
//!
//! Files per complex `<C>`, in the conventions of `derivations --modules`: `<C>_rank.csv`,
//! `<C>_indaction.csv`, `<C>_unknown.csv` (pairs whose product is unknown), `<C>_istar.csv`,
//! `<C>_pstar.csv`; the basis of each bidegree is the cokernel part (unit vectors at the non-pivot
//! columns of the image of `h_j`) followed by the kernel part (its reduced basis).

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use algebra::{Algebra, AlgebraType, module::Module};
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::h_product,
    save::SaveKind,
    utils::query_module,
};
use sseq::coordinates::Bidegree;

/// A bitmask over a basis of at most 64 elements.
type V = u64;

/// Echelon form of a list of vectors (pivot = lowest set bit), reduced.
struct Echelon {
    rows: Vec<V>,
}

impl Echelon {
    fn new(vs: impl IntoIterator<Item = V>) -> Self {
        let mut rows: Vec<V> = vec![];
        for mut v in vs {
            for r in &rows {
                if v >> r.trailing_zeros() & 1 == 1 {
                    v ^= r;
                }
            }
            if v != 0 {
                let p = v.trailing_zeros();
                for r in rows.iter_mut() {
                    if *r >> p & 1 == 1 {
                        *r ^= v;
                    }
                }
                rows.push(v);
            }
        }
        Self { rows }
    }
    fn pivots(&self) -> V {
        self.rows.iter().fold(0, |a, r| a | (1 << r.trailing_zeros()))
    }
    fn reduce(&self, mut v: V) -> V {
        for r in &self.rows {
            if v >> r.trailing_zeros() & 1 == 1 {
                v ^= r;
            }
        }
        v
    }
}

/// The kernel of a matrix given by its rows (`rows[i]` = image of basis vector `i`).
fn kernel(rows: &[V]) -> Vec<V> {
    // augmented rows: image | identity
    let n = rows.len();
    let mut aug: Vec<(V, V)> = rows.iter().enumerate().map(|(i, &r)| (r, 1 << i)).collect();
    let mut out = vec![];
    let mut used = vec![false; n];
    for bit in 0..64 {
        let Some(p) = (0..n).find(|&i| !used[i] && aug[i].0 >> bit & 1 == 1) else { continue };
        used[p] = true;
        let pr = aug[p];
        for (i, a) in aug.iter_mut().enumerate() {
            if i != p && a.0 >> bit & 1 == 1 {
                a.0 ^= pr.0;
                a.1 ^= pr.1;
            }
        }
    }
    for (i, a) in aug.iter().enumerate() {
        if !used[i] && a.0 == 0 {
            out.push(a.1);
        }
    }
    // reduced echelon basis of the kernel
    Echelon::new(out).rows
}

fn vector(v: V, len: usize) -> String {
    let e: Vec<String> = (0..len).map(|k| (v >> k & 1).to_string()).collect();
    if len == 1 {
        format!("[{}]", e[0])
    } else {
        format!("\"[{}]\"", e.join(", "))
    }
}

fn parse_vec(s: &str) -> V {
    let s = s.trim().trim_matches('"').trim_start_matches('[').trim_end_matches(']');
    s.split(',')
        .enumerate()
        .filter(|(_, e)| e.trim() == "1")
        .fold(0, |a, (k, _)| a | (1 << k))
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let res = Arc::new(query_module(Some(AlgebraType::Milnor), true)?);
    let ring: PathBuf = query::raw("Ring export directory", |s| {
        Ok::<_, anyhow::Error>(PathBuf::from(s))
    });
    let dir: PathBuf =
        query::raw("Output directory", |s| Ok::<_, anyhow::Error>(PathBuf::from(s)));
    let max_n: i32 = query::with_default("Max stem", "100", str::parse);
    let max_s: i32 = query::with_default("Max filtration", "50", str::parse);
    let specs: String = query::with_default("Complexes (name:j)", "C2:0,Ceta:1,Cnu:2,Csigma:3", |s| {
        Ok::<_, anyhow::Error>(s.to_string())
    });
    std::fs::create_dir_all(&dir)?;
    let algebra = res.algebra();
    let store = res.save_dir().store().expect("a saved resolution").clone();

    let in_range = |b: Bidegree| b.s() >= 0 && b.n() >= 0 && b.n() <= max_n && b.s() <= max_s;
    let gens = |b: Bidegree| {
        if b.s() < 0 || b.t() < b.s() || !res.has_computed_bidegree(b) {
            0
        } else {
            res.number_of_gens_in_bidegree(b)
        }
    };

    // The ring export: indecomposables and their products with basis elements.
    let mut inds: Vec<(Bidegree, usize)> = vec![];
    for l in BufReader::new(File::open(ring.join("indecomposables.csv"))?).lines().skip(1) {
        let l = l?;
        let v: Vec<&str> = l.split(',').collect();
        let b = Bidegree::n_s(v[0].parse()?, v[1].parse()?);
        if in_range(b) {
            inds.push((b, v[2].parse()?));
        }
    }
    let mut prod: HashMap<((Bidegree, usize), (Bidegree, usize)), V> = HashMap::new();
    for l in BufReader::new(File::open(ring.join("generator_products.csv"))?).lines().skip(1) {
        let l = l?;
        let p: Vec<&str> = l.splitn(7, ',').collect();
        let a = Bidegree::n_s(p[0].parse()?, p[1].parse()?);
        let b = Bidegree::n_s(p[3].parse()?, p[4].parse()?);
        if !in_range(a + b) {
            continue;
        }
        prod.insert(((a, p[2].parse()?), (b, p[5].parse()?)), parse_vec(p[6]));
    }
    // g · z for a vector z at `b`
    let mul = |g: (Bidegree, usize), b: Bidegree, z: V| {
        (0..64)
            .filter(|&i| z >> i & 1 == 1)
            .fold(0, |a, i| a ^ prod.get(&(g, (b, i))).copied().unwrap_or(0))
    };
    eprintln!("ring: {} indecomposables, {} products", inds.len(), prod.len());

    let bidegrees: Vec<Bidegree> = (0..=max_s)
        .flat_map(|s| (0..=max_n).map(move |n| Bidegree::n_s(n, s)))
        .collect();

    for spec in specs.split(',') {
        let (name, j) = spec.split_once(':').unwrap();
        let j: u32 = j.parse()?;
        let k = 1i32 << j;
        let (_, op_idx) = algebra.basis_element_from_string(&format!("Sq{k}")).unwrap();
        // h_j: rows at b, columns at b + (k - 1, 1)
        let hj = |b: Bidegree| -> Vec<V> {
            let rows = gens(b);
            let tgt = b + Bidegree::n_s(k - 1, 1);
            if rows == 0 {
                return vec![];
            }
            if gens(tgt) == 0 || !res.has_computed_bidegree(tgt) {
                return vec![0; rows];
            }
            let m = h_product(&*res, j, b).expect("h_j product");
            (0..rows)
                .map(|r| m.row(r).iter().enumerate().fold(0, |a, (c, x)| a | ((x as V) << c)))
                .collect()
        };
        // the basis of Ext(C_j) at B: (cokernel echelon, cokernel columns, kernel basis)
        struct Part {
            image: Echelon,
            coker: Vec<usize>,
            ker: Vec<V>,
        }
        let mut parts: HashMap<Bidegree, Part> = HashMap::new();
        let mut rank = BufWriter::new(File::create(dir.join(format!("{name}_rank.csv")))?);
        writeln!(rank, "stem,Adams_filtration,rank")?;
        for &b in &bidegrees {
            let below = Bidegree::n_s(b.n() - k + 1, b.s() - 1);
            let image = Echelon::new(if below.s() >= 0 && below.n() >= 0 { hj(below) } else { vec![] });
            let piv = image.pivots();
            let coker: Vec<usize> = (0..gens(b)).filter(|&c| piv >> c & 1 == 0).collect();
            let top = Bidegree::n_s(b.n() - k, b.s());
            let ker = if top.n() >= 0 { kernel(&hj(top)) } else { vec![] };
            let d = coker.len() + ker.len();
            assert!(d <= 64);
            if d > 0 {
                writeln!(rank, "{},{},{d}", b.n(), b.s())?;
            }
            parts.insert(b, Part { image, coker, ker });
        }
        // coordinates in Ext(C_j) at B of (bottom v at B, top w at B - (k, 0))
        let coords = |b: Bidegree, v: V, w: V| -> V {
            let p = &parts[&b];
            let v = p.image.reduce(v);
            let mut out = 0;
            for (i, &c) in p.coker.iter().enumerate() {
                out |= (v >> c & 1) << i;
            }
            let off = p.coker.len();
            for (i, &r) in p.ker.iter().enumerate() {
                out |= (w >> r.trailing_zeros() & 1) << (off + i);
            }
            out
        };
        let dim = |b: Bidegree| parts.get(&b).map_or(0, |p| p.coker.len() + p.ker.len());

        // The Sq(2^j) slot of the saved map `ms` (of a class in bidegree `mb`) at the top class
        // `z ⊗ top`, `z ∈ ker(h_j)` at `zb`: the bottom part of the product, or `None` if that step
        // was never lifted.
        let slot = |ms: &ext::save::ZarrSaveStore,
                    mb: Bidegree,
                    zb: Bidegree,
                    z: V|
         -> anyhow::Result<Option<V>> {
            let input = Bidegree::s_t(zb.s() + mb.s(), zb.t() + mb.t() + k);
            if gens(input) == 0 {
                return Ok(Some(0));
            }
            Ok(ms.read(SaveKind::ChainMap, input)?.map(|data| {
                let src = res.module(zb.s());
                let fx_dim = src.dimension(zb.t() + k);
                let mut cur = &data[..];
                let mut star: V = 0;
                for l in 0..gens(input) {
                    let fv = fp::vector::FpVector::from_bytes(res.prime(), fx_dim, &mut cur).unwrap();
                    let mut c = 0;
                    for i in 0..gens(zb) {
                        if z >> i & 1 == 1 {
                            c ^= fv.entry(src.operation_generator_to_index(k, op_idx, zb.t(), i));
                        }
                    }
                    star |= (c as V) << l;
                }
                star
            }))
        };
        type Cls = (Bidegree, usize);
        // x[(g, u)]: the indecomposable g times the basis element u of Ext(C_j)
        use maybe_rayon::prelude::*;
        let mut x: HashMap<(Cls, Cls), V> = HashMap::new();
        let mut unknown: HashMap<(Cls, Cls), &str> = HashMap::new();
        let per_ind: Vec<anyhow::Result<(Vec<((Cls, Cls), V)>, Vec<((Cls, Cls), &str)>)>> = inds
            .clone()
            .into_maybe_par_iter()
            .map(|(gb, gi)| {
            let mut x = vec![];
            let mut unknown = vec![];
            let g = (gb, gi);
            let map = format!("products/lin_ring/x_({}, {}, {})", gb.n(), gb.s(), gi);
            let has_map = store.path().join(&map).join("zarr.json").exists();
            let mstore = has_map.then(|| store.subgroup(&map)).transpose()?;
            for &b in &bidegrees {
                let t = b + gb;
                if dim(b) == 0 || !in_range(t) || dim(t) == 0 {
                    continue;
                }
                let p = &parts[&b];
                for (xc, &c) in p.coker.iter().enumerate() {
                    x.push(((g, (b, xc)), coords(t, mul(g, b, 1 << c), 0)));
                }
                let zb = Bidegree::n_s(b.n() - k, b.s());
                for (xi, &z) in p.ker.iter().enumerate() {
                    let u = (b, p.coker.len() + xi);
                    let star = match &mstore {
                        Some(ms) => slot(ms, gb, zb, z)?,
                        None => None,
                    };
                    match star {
                        Some(star) => {
                            x.push(((g, u), coords(t, star, mul(g, zb, z))));
                        }
                        None => {
                            unknown.push(((g, u), if has_map { "no step" } else { "no map" }));
                        }
                    }
                }
            }
            Ok((x, unknown))
            })
            .collect();
        for r in per_ind {
            let (xs, us) = r?;
            x.extend(xs);
            unknown.extend(us);
        }
        let direct_unknown = unknown.len();
        eprintln!("{name}: indecomposables read, {} known, {direct_unknown} unknown", x.len());
        // Every saved monomial map m = f_1 ⋯ f_r gives m·(z ⊗ top) on its corner. The top part is the
        // ring product z·f_1⋯f_r, the bottom part the slot.
        let lin = store.path().join("products/lin_ring");
        let names: Vec<String> = std::fs::read_dir(&lin)?
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        let per_map: Vec<anyhow::Result<Vec<(Vec<Cls>, Cls, V)>>> = names
            .into_maybe_par_iter()
            .map(|name| {
            let mut mons = vec![];
            let factors: Vec<Cls> = name
                .split('.')
                .filter_map(|f| {
                    let v: Vec<i32> = f
                        .trim_start_matches("x_(")
                        .trim_end_matches(')')
                        .split(", ")
                        .map(|t| t.parse().ok())
                        .collect::<Option<_>>()?;
                    Some((Bidegree::n_s(v[0], v[1]), v[2] as usize))
                })
                .collect();
            if factors.len() < 2 {
                return Ok(mons);
            }
            let mb = factors.iter().fold(Bidegree::n_s(0, 0), |a, f| a + f.0);
            if !in_range(mb) {
                return Ok(mons);
            }
            let ms = store.subgroup(&format!("products/lin_ring/{name}"))?;
            for &b in &bidegrees {
                let t = b + mb;
                if dim(b) == 0 || !in_range(t) || dim(t) == 0 {
                    continue;
                }
                let p = &parts[&b];
                let zb = Bidegree::n_s(b.n() - k, b.s());
                for (xi, &z) in p.ker.iter().enumerate() {
                    let Some(star) = slot(&ms, mb, zb, z)? else { continue };
                    let (mut w, mut wb) = (z, zb);
                    for &f in &factors {
                        w = mul(f, wb, w);
                        wb = wb + f.0;
                    }
                    mons.push((factors.clone(), (b, p.coker.len() + xi), coords(t, star, w)));
                }
            }
            Ok(mons)
            })
            .collect();
        let mut mons: Vec<(Vec<Cls>, Cls, V)> = vec![];
        for r in per_map {
            mons.extend(r?);
        }
        eprintln!("{name}: {} monomial products read", mons.len());
        // Associativity closure: m·c = f·((m/f)·c) for each factor f is linear in the unknown x(f, ·).
        let bits = |v: V| (0..64).filter(move |&i| v >> i & 1 == 1);
        let (mut rounds, mut consistent, mut inconsistent) = (0, 0usize, 0usize);
        loop {
            rounds += 1;
            let mut groups: HashMap<(Cls, Bidegree), Vec<(V, V)>> = HashMap::new();
            consistent = 0;
            inconsistent = 0;
            for (factors, (b, xc), val) in &mons {
                for fi in 0..factors.len() {
                    let f = factors[fi];
                    let (mut v, mut vb, mut ok) = (1 << xc, *b, true);
                    for (gi, &g) in factors.iter().enumerate() {
                        if gi == fi {
                            continue;
                        }
                        let mut nv = 0;
                        for u in bits(v) {
                            match x.get(&(g, (vb, u))) {
                                Some(p) => nv ^= p,
                                None => ok = false,
                            }
                        }
                        if !ok {
                            break;
                        }
                        v = nv;
                        vb = vb + g.0;
                    }
                    if !ok {
                        continue;
                    }
                    let (mut mask, mut rhs) = (0, *val);
                    for u in bits(v) {
                        match x.get(&(f, (vb, u))) {
                            Some(p) => rhs ^= p,
                            None => mask |= 1 << u,
                        }
                    }
                    if mask == 0 {
                        if rhs == 0 { consistent += 1 } else { inconsistent += 1 }
                    } else {
                        groups.entry((f, vb)).or_default().push((mask, rhs));
                    }
                }
            }
            let mut progress = false;
            for ((f, vb), rows) in groups {
                let mut ech: Vec<(V, V)> = vec![];
                for (mut m, mut r) in rows {
                    for &(em, er) in &ech {
                        if m >> em.trailing_zeros() & 1 == 1 {
                            m ^= em;
                            r ^= er;
                        }
                    }
                    if m == 0 {
                        if r != 0 {
                            inconsistent += 1;
                        }
                        continue;
                    }
                    let p = m.trailing_zeros();
                    for e in ech.iter_mut() {
                        if e.0 >> p & 1 == 1 {
                            e.0 ^= m;
                            e.1 ^= r;
                        }
                    }
                    ech.push((m, r));
                }
                for (m, r) in ech {
                    if m.count_ones() == 1 {
                        let u = (vb, m.trailing_zeros() as usize);
                        if unknown.remove(&(f, u)).is_some() {
                            x.insert((f, u), r);
                            progress = true;
                        }
                    }
                }
            }
            if !progress {
                break;
            }
        }
        let mut ia = BufWriter::new(File::create(dir.join(format!("{name}_indaction.csv")))?);
        writeln!(ia, "stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,product")?;
        for (&((gb, gi), (b, xc)), &v) in &x {
            if v != 0 {
                writeln!(ia, "{},{},{gi},{},{},{xc},{}", gb.n(), gb.s(), b.n(), b.s(), vector(v, dim(b + gb)))?;
            }
        }
        let mut un = BufWriter::new(File::create(dir.join(format!("{name}_unknown.csv")))?);
        writeln!(un, "stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,reason")?;
        let (mut no_map, mut no_step) = (0, 0);
        for (&((gb, gi), (b, xc)), &reason) in &unknown {
            if reason == "no map" { no_map += 1 } else { no_step += 1 }
            writeln!(un, "{},{},{gi},{},{},{xc},{reason}", gb.n(), gb.s(), b.n(), b.s())?;
        }
        let known = x.len();
        // i_*: bottom; p_*: the kernel part
        let mut wi = BufWriter::new(File::create(dir.join(format!("{name}_istar.csv")))?);
        writeln!(wi, "stem,Adams_filtration,index,image")?;
        let mut wp = BufWriter::new(File::create(dir.join(format!("{name}_pstar.csv")))?);
        writeln!(wp, "stem,Adams_filtration,index,image")?;
        for &b in &bidegrees {
            let p = &parts[&b];
            for i in 0..gens(b) {
                let out = coords(b, 1 << i, 0);
                if out != 0 {
                    writeln!(wi, "{},{},{i},{}", b.n(), b.s(), vector(out, dim(b)))?;
                }
            }
            let zb = Bidegree::n_s(b.n() - k, b.s());
            for (xi, &z) in p.ker.iter().enumerate() {
                writeln!(wp, "{},{},{},{}", b.n(), b.s(), p.coker.len() + xi, vector(z, gens(zb)))?;
            }
        }
        eprintln!(
            "{name}: products known {known}, unknown {} (no map {no_map}, step not lifted {no_step}); \
             {} unknown before closure; {} monomial products, closure {rounds} rounds, {consistent} \
             equations consistent, {inconsistent} inconsistent",
            no_map + no_step,
            direct_unknown,
            mons.len()
        );
    }
    Ok(())
}
