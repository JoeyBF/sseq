//! The E2 page of any finite spectrum `X` from a saved minimal resolution `P` of the sphere.
//!
//! `P ⊗ H^*X` with the diagonal action is a free resolution of `H^*X` (free on `g ⊗ m`), so
//! `Ext(X)` is the cohomology of `Hom(P) ⊗ (H^*X)^*` with coboundary
//!
//! `δφ(g' ⊗ m') = Σ_{g, a} [a, g](d g') · φ(g ⊗ χ(a) m')`,
//!
//! the sum over Milnor basis elements `a` of positive degree, `[a, g](d g')` the coefficient of
//! `a g` in the saved differential. Nothing is lifted: only coefficients of `d` of degree below the
//! top cell of `X` are read.
//!
//! Prompts: the sphere's module and save as for `resolve`, then the module json of `X`, its name
//! and the output directory. Files (cochain coordinates `cell:generator` over the cells sorted by
//! degree, generator = index among the sphere's generators in bidegree `(n - |cell|, s)`):
//! - `<X>_cells.csv`: `index,cell,degree`, the coordinate order of the cells;
//! - `<X>_rank.csv`: `stem,Adams_filtration,rank`;
//! - `<X>_basis.csv`: `stem,Adams_filtration,index,coords`, the reduced echelon basis of cocycles
//!   modulo boundaries (pivot = first coordinate, bottom cell first). For a two-cell complex this
//!   is `cofiber_slots`' basis (cokernel part, then kernel part);
//! - `<X>_bound.csv`: `stem,Adams_filtration,index,coords`, the boundaries in reduced echelon form.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use algebra::{
    Algebra, AlgebraType, Bialgebra, MilnorAlgebra,
    module::{FDModule, Module},
};
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    utils::query_module,
};
use sseq::coordinates::Bidegree;

/// A bit vector over cochain coordinates.
#[derive(Clone, PartialEq, Eq)]
struct Bits(Vec<u64>);

impl Bits {
    fn zero(n: usize) -> Self {
        Self(vec![0; n.div_ceil(64)])
    }
    fn get(&self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 == 1
    }
    fn flip(&mut self, i: usize) {
        self.0[i / 64] ^= 1 << (i % 64);
    }
    fn xor(&mut self, o: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&o.0) {
            *a ^= b;
        }
    }
    fn first(&self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .find(|(_, w)| **w != 0)
            .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
    }
    fn is_zero(&self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }
    fn ones(&self, n: usize) -> impl Iterator<Item = usize> + '_ {
        (0..n).filter(|&i| self.get(i))
    }
}

/// Fully reduced echelon form, pivot = first set bit, rows sorted by pivot.
fn rref(vs: impl IntoIterator<Item = Bits>) -> Vec<Bits> {
    let mut rows: Vec<(usize, Bits)> = vec![];
    for mut v in vs {
        for (p, r) in &rows {
            if v.get(*p) {
                v.xor(r);
            }
        }
        if let Some(p) = v.first() {
            for (_, r) in rows.iter_mut() {
                if r.get(p) {
                    r.xor(&v);
                }
            }
            rows.push((p, v));
        }
    }
    rows.sort_by_key(|r| r.0);
    rows.into_iter().map(|r| r.1).collect()
}

/// Reduce `v` by fully reduced echelon rows.
fn reduce(rows: &[Bits], mut v: Bits) -> Bits {
    for r in rows {
        let p = r.first().unwrap();
        if v.get(p) {
            v.xor(r);
        }
    }
    v
}

/// The kernel of the map sending source coordinate `i` to `images[i]` (target length `tn`).
fn kernel(images: &[Bits], n: usize, tn: usize) -> Vec<Bits> {
    let mut aug: Vec<(Bits, Bits)> = images
        .iter()
        .enumerate()
        .map(|(i, im)| {
            let mut id = Bits::zero(n);
            id.flip(i);
            (im.clone(), id)
        })
        .collect();
    let mut used = vec![false; n];
    for bit in 0..tn {
        let Some(p) = (0..n).find(|&i| !used[i] && aug[i].0.get(bit)) else {
            continue;
        };
        used[p] = true;
        let pr = aug[p].clone();
        for (i, a) in aug.iter_mut().enumerate() {
            if i != p && a.0.get(bit) {
                a.0.xor(&pr.0);
                a.1.xor(&pr.1);
            }
        }
    }
    aug.into_iter()
        .zip(used)
        .filter(|((im, _), u)| !u && im.is_zero())
        .map(|((_, id), _)| id)
        .collect()
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let res = Arc::new(query_module(Some(AlgebraType::Milnor), true)?);
    let mpath: PathBuf = query::raw("Module json of X", |s| Ok::<_, anyhow::Error>(PathBuf::from(s)));
    let name: String = query::raw("Name of X", |s| Ok::<_, anyhow::Error>(s.to_string()));
    let dir: PathBuf = query::raw("Output directory", |s| Ok::<_, anyhow::Error>(PathBuf::from(s)));
    let max_n: i32 = query::with_default("Max stem", "100", str::parse);
    let max_s: i32 = query::with_default("Max filtration", "50", str::parse);
    std::fs::create_dir_all(&dir)?;

    let algebra = res.algebra();
    let milnor: &MilnorAlgebra = (&*algebra).try_into().expect("Milnor basis");
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&mpath)?)?;
    let m = FDModule::from_json(Arc::clone(&algebra), &json)?;
    let (lo, hi) = (m.min_degree(), m.max_degree().expect("finite module"));
    algebra.compute_basis(hi - lo + 1);
    // cells sorted by degree: (degree, index in that degree)
    let cells: Vec<(i32, usize)> = (lo..=hi)
        .flat_map(|d| (0..m.dimension(d)).map(move |i| (d, i)))
        .collect();
    let cell_of = |d: i32, i: usize| cells.iter().position(|&c| c == (d, i)).unwrap();
    let cname = |c: usize| m.basis_element_to_string(cells[c].0, cells[c].1);

    // chi(a) on H^*X for every Milnor basis element a of degree <= top - bottom, as a matrix:
    // chi[deg][idx][cell] = set of cells of chi(a) cell. chi(a) = sum_{a' a'' = Delta a, |a'| > 0}
    // a' chi(a'').
    let act = |od: i32, oi: usize, c: usize| -> Vec<usize> {
        let (d, i) = cells[c];
        if m.dimension(d + od) == 0 {
            return vec![];
        }
        let mut v = fp::vector::FpVector::new(m.prime(), m.dimension(d + od));
        m.act_on_basis(v.as_slice_mut(), 1, od, oi, d, i);
        v.iter_nonzero().map(|(j, _)| cell_of(d + od, j)).collect()
    };
    let span = hi - lo;
    let mut chi: Vec<Vec<Vec<Vec<usize>>>> = vec![vec![(0..cells.len()).map(|c| vec![c]).collect()]];
    for od in 1..=span {
        let mut this = vec![];
        for oi in 0..algebra.dimension(od) {
            let mut row: Vec<Vec<usize>> = vec![vec![]; cells.len()];
            for (d1, i1, d2, i2) in milnor.coproduct(od, oi) {
                if d1 == 0 {
                    continue;
                }
                for c in 0..cells.len() {
                    for &c2 in &chi[d2 as usize][i2][c] {
                        for c3 in act(d1, i1, c2) {
                            if let Some(p) = row[c].iter().position(|&x| x == c3) {
                                row[c].swap_remove(p);
                            } else {
                                row[c].push(c3);
                            }
                        }
                    }
                }
            }
            this.push(row);
        }
        chi.push(this);
    }

    let gens = |b: Bidegree| {
        if b.s() < 0 || b.n() < 0 || b.t() < b.s() || !res.has_computed_bidegree(b) {
            0
        } else {
            res.number_of_gens_in_bidegree(b)
        }
    };
    // coordinate layout at (n, s): offsets per cell
    let layout = |n: i32, s: i32| -> (Vec<usize>, usize) {
        let mut off = vec![];
        let mut tot = 0;
        for &(d, _) in &cells {
            off.push(tot);
            tot += gens(Bidegree::n_s(n - (d - lo), s));
        }
        (off, tot)
    };
    // the coboundary from (n, s) to (n - 1, s + 1), as images of source coordinates
    let delta = |n: i32, s: i32| -> Vec<Bits> {
        let (so, sn) = layout(n, s);
        let (to, tn) = layout(n - 1, s + 1);
        let mut images = vec![Bits::zero(tn); sn];
        if tn == 0 || sn == 0 {
            return images;
        }
        let t = n + s; // internal degree of the cochains, cell degrees relative to lo
        let src = res.module(s);
        let dif = res.differential(s + 1);
        for (c2, &(d2, _)) in cells.iter().enumerate() {
            let tb = Bidegree::n_s(n - 1 - (d2 - lo), s + 1);
            let tdeg = t - (d2 - lo);
            for g2 in 0..gens(tb) {
                let dg = dif.output(tdeg, g2);
                for (c, &(d, _)) in cells.iter().enumerate() {
                    let od = d - d2;
                    if od <= 0 {
                        continue;
                    }
                    let sb = Bidegree::n_s(n - (d - lo), s);
                    for oi in 0..algebra.dimension(od) {
                        if !chi[od as usize][oi][c2].contains(&c) {
                            continue;
                        }
                        for g in 0..gens(sb) {
                            let idx = src.operation_generator_to_index(od, oi, sb.t(), g);
                            if dg.entry(idx) == 1 {
                                images[so[c] + g].flip(to[c2] + g2);
                            }
                        }
                    }
                }
            }
        }
        images
    };

    let fmt = |n: i32, s: i32, v: &Bits, len: usize| -> String {
        let (off, _) = layout(n, s);
        v.ones(len)
            .map(|i| {
                let c = off.iter().rposition(|&o| o <= i).unwrap();
                format!("{}:{}", cname(c), i - off[c])
            })
            .collect::<Vec<_>>()
            .join(";")
    };
    let mut cf = BufWriter::new(File::create(dir.join(format!("{name}_cells.csv")))?);
    writeln!(cf, "index,cell,degree")?;
    for (c, &(d, _)) in cells.iter().enumerate() {
        writeln!(cf, "{c},{},{}", cname(c), d - lo)?;
    }
    drop(cf);
    let mut rank = BufWriter::new(File::create(dir.join(format!("{name}_rank.csv")))?);
    let mut basis = BufWriter::new(File::create(dir.join(format!("{name}_basis.csv")))?);
    let mut bound = BufWriter::new(File::create(dir.join(format!("{name}_bound.csv")))?);
    writeln!(rank, "stem,Adams_filtration,rank")?;
    writeln!(basis, "stem,Adams_filtration,index,coords")?;
    writeln!(bound, "stem,Adams_filtration,index,coords")?;
    let mut ext_at: std::collections::HashMap<(i32, i32), (Vec<Bits>, Vec<Bits>)> =
        Default::default();
    for s in 0..=max_s {
        for n in 0..=max_n {
            let (_, len) = layout(n, s);
            if len == 0 {
                continue;
            }
            let out = delta(n, s);
            let z = kernel(&out, len, layout(n - 1, s + 1).1);
            let b = if s > 0 { rref(delta(n + 1, s - 1)) } else { vec![] };
            let reps = rref(z.into_iter().map(|v| reduce(&b, v)).filter(|v| !v.is_zero()));
            if !reps.is_empty() {
                writeln!(rank, "{n},{s},{}", reps.len())?;
            }
            for (i, v) in reps.iter().enumerate() {
                writeln!(basis, "{n},{s},{i},{}", fmt(n, s, v, len))?;
            }
            for (i, v) in b.iter().enumerate() {
                writeln!(bound, "{n},{s},{i},{}", fmt(n, s, v, len))?;
            }
            ext_at.insert((n, s), (reps, b));
        }
    }
    drop((rank, basis, bound));

    // h_i acts through the chain map g ↦ θ_i(dg) lifting h_i, θ_i(Sq(R)) = Sq(R - 2^i e_1):
    // (h_i φ)(g' ⊗ m') = Σ [Sq(R_b + 2^i e_1), g](dg') φ(g ⊗ χ(b) m') over b of degree |m| - |m'|.
    let mut hact = BufWriter::new(File::create(dir.join(format!("{name}_hact.csv")))?);
    writeln!(
        hact,
        "stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,product"
    )?;
    let max_i = 7;
    algebra.compute_basis(span + (1 << max_i));
    // coordinates of a cocycle in the basis at (n, s)
    let coords = |n: i32, s: i32, v: Bits| -> Option<Vec<bool>> {
        let (reps, b) = ext_at.get(&(n, s))?;
        let mut v = reduce(b, v);
        let mut out = vec![false; reps.len()];
        for (i, r) in reps.iter().enumerate() {
            if v.get(r.first().unwrap()) {
                v.xor(r);
                out[i] = true;
            }
        }
        assert!(v.is_zero(), "h_i product is not a cocycle at ({n}, {s})");
        Some(out)
    };
    for i in 0..=max_i {
        let k = 1i32 << i;
        for s in 0..max_s {
            for n in 0..=max_n {
                let (tn_, ts) = (n + k - 1, s + 1);
                if tn_ > max_n {
                    continue;
                }
                let Some((reps, _)) = ext_at.get(&(n, s)) else {
                    continue;
                };
                if reps.is_empty() || !ext_at.contains_key(&(tn_, ts)) {
                    continue;
                }
                let (so, sn) = layout(n, s);
                let (to, tn) = layout(tn_, ts);
                let t = n + s;
                let src = res.module(s);
                let dif = res.differential(ts);
                // image of each source coordinate
                let mut images = vec![Bits::zero(tn); sn];
                for (c2, &(d2, _)) in cells.iter().enumerate() {
                    let tb = Bidegree::n_s(tn_ - (d2 - lo), ts);
                    let tdeg = t + k - (d2 - lo);
                    for g2 in 0..gens(tb) {
                        let dg = dif.output(tdeg, g2);
                        for (c, &(d, _)) in cells.iter().enumerate() {
                            let od = d - d2;
                            if od < 0 {
                                continue;
                            }
                            let sb = Bidegree::n_s(n - (d - lo), s);
                            if gens(sb) == 0 {
                                continue;
                            }
                            for oi in 0..algebra.dimension(od) {
                                if !chi[od as usize][oi][c2].contains(&c) {
                                    continue;
                                }
                                let mut e = milnor.basis_element_from_index(od, oi);
                                e.p_part.set(0, e.p_part.get(0) + k as u32);
                                e.degree += k;
                                let ai = milnor.basis_element_to_index(&e);
                                for g in 0..gens(sb) {
                                    let idx = src.operation_generator_to_index(od + k, ai, sb.t(), g);
                                    if dg.entry(idx) == 1 {
                                        images[so[c] + g].flip(to[c2] + g2);
                                    }
                                }
                            }
                        }
                    }
                }
                for (xi, r) in reps.iter().enumerate() {
                    let mut im = Bits::zero(tn);
                    for j in r.ones(sn) {
                        im.xor(&images[j]);
                    }
                    let v = coords(tn_, ts, im).unwrap();
                    let body: Vec<String> = v.iter().map(|&b| (b as u8).to_string()).collect();
                    let vs = if v.len() == 1 {
                        format!("[{}]", body[0])
                    } else {
                        format!("\"[{}]\"", body.join(", "))
                    };
                    writeln!(hact, "{},1,0,{n},{s},{xi},{vs}", k - 1)?;
                }
            }
        }
    }
    Ok(())
}
