//! A ring map on E2 (`--solve-map`), extended from low stems by multiplicativity.

use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};

use crate::{Ext, Mask, build_products, check_products, compare::load_iota, eliminate, load};

/// The image of a sphere basis class: known, or the first of its unknown coordinates.
#[derive(Clone, Copy)]
enum Image {
    Known(Mask),
    Unknown(u32),
}

/// Solves for a ring map `iota: Ext(S) -> Ext(R)` on E2 above the stems where it is given.
///
/// `--solve-map <ring dir> <seed csv> <seed max stem> <out prefix>`: the seed (in the format of
/// `compare::load_iota`) is trusted through `seed max stem`; above it every coordinate of
/// `iota(x)` is an unknown. Each product `g x` of a sphere indecomposable `g` and a basis class `x`
/// gives `iota(g x) = iota(g) iota(x)`, linear whenever `iota(g)` or `iota(x)` is known, which is
/// every product below twice the seed's range. Writes `<out prefix>.csv` (determined images, in
/// the seed's format, seed rows included) and `<out prefix>_status.csv` (every class above the
/// seed: `determined` or `open`, with the particular solution).
pub(crate) fn run(ext: &Ext, args: &[String]) {
    let t = Instant::now();
    let i = args.iter().position(|a| a == "--solve-map").unwrap();
    let (dir, seed_path, out) = (&args[i + 1], &args[i + 2], &args[i + 4]);
    let seed_max: i32 = args[i + 3].parse().unwrap();
    let rb = load(dir, ext.n_max, ext.f_max);
    let pb = build_products(&rb);
    check_products(&rb, &pb);
    let seed = load_iota(seed_path);

    // the image of every sphere basis class whose R bidegree is non-zero
    let nb = ext.deg.len();
    let mut image: Vec<Option<Image>> = vec![None; ext.off[nb - 1] as usize + ext.dim[nb - 1]];
    let mut nvars = 0u32;
    let mut unknown: Vec<(u32, usize, u32)> = vec![]; // (bidegree, index, R bidegree)
    for b in 0..nb as u32 {
        let (n, f) = ext.deg[b as usize];
        let Some(tb) = rb.get(n, f) else { continue };
        for k in 0..ext.dim[b as usize] {
            let g = ext.gid(b, k) as usize;
            image[g] = Some(if n <= seed_max {
                Image::Known(seed.get(&(n, f, k)).copied().unwrap_or(0))
            } else {
                unknown.push((b, k, tb));
                nvars += rb.dim[tb as usize] as u32;
                Image::Unknown(nvars - rb.dim[tb as usize] as u32)
            });
        }
    }
    let one = nvars;
    eprintln!(
        "map: {} classes known through stem {seed_max}, {} unknown classes, {nvars} unknowns",
        image
            .iter()
            .flatten()
            .filter(|x| matches!(x, Image::Known(_)))
            .count(),
        unknown.len()
    );

    // R product of a known vector at bidegree a with basis vector j at bidegree c
    let mul_r = |a: u32, u: Mask, c: u32, j: usize| -> Mask {
        let mut out = 0;
        for i in 0..rb.dim[a as usize] {
            if u >> i & 1 == 1 {
                out ^= pb.mul(&rb, a, i, c, j);
            }
        }
        out
    };
    let mut prop = crate::propagate::Propagator::new(nvars);
    // iota(g x) = iota(g) iota(x) per target coordinate; None when both images are unknown
    let equation = |g: u32, x: u32, m: Mask| -> Option<Vec<Vec<u32>>> {
        let (bg, bx) = (ext.basis_bid[g as usize], ext.basis_bid[x as usize]);
        // out of range, or a zero group: no equation
        let Some(bt) = ext.sum(bg, bx) else {
            return Some(vec![]);
        };
        let (n, f) = ext.deg[bt as usize];
        let Some(rt) = rb.get(n, f) else {
            return Some(vec![]);
        };
        let dt = rb.dim[rt as usize];
        let mut terms: Vec<Vec<u32>> = vec![vec![]; dt];
        let push_mask = |terms: &mut Vec<Vec<u32>>, mask: Mask| {
            for (r, tr) in terms.iter_mut().enumerate() {
                if mask >> r & 1 == 1 {
                    tr.push(one);
                }
            }
        };
        let r_of = |b: u32| {
            let (n, f) = ext.deg[b as usize];
            rb.get(n, f)
        };
        if let (Some(rg), Some(rx)) = (r_of(bg), r_of(bx)) {
            match (image[g as usize].unwrap(), image[x as usize].unwrap()) {
                (Image::Known(u), Image::Known(w)) => {
                    let mut prod = 0;
                    for j in 0..rb.dim[rx as usize] {
                        if w >> j & 1 == 1 {
                            prod ^= mul_r(rg, u, rx, j);
                        }
                    }
                    push_mask(&mut terms, prod);
                }
                (Image::Known(u), Image::Unknown(v)) => {
                    for j in 0..rb.dim[rx as usize] {
                        let p = mul_r(rg, u, rx, j);
                        for (r, tr) in terms.iter_mut().enumerate() {
                            if p >> r & 1 == 1 {
                                tr.push(v + j as u32);
                            }
                        }
                    }
                }
                (Image::Unknown(v), Image::Known(w)) => {
                    for j in 0..rb.dim[rg as usize] {
                        let p = mul_r(rx, w, rg, j);
                        for (r, tr) in terms.iter_mut().enumerate() {
                            if p >> r & 1 == 1 {
                                tr.push(v + j as u32);
                            }
                        }
                    }
                }
                (Image::Unknown(_), Image::Unknown(_)) => return None,
            }
        }
        // + iota(g x)
        for c in (0..ext.dim[bt as usize]).filter(|&c| m >> c & 1 == 1) {
            match image[ext.gid(bt, c) as usize].unwrap() {
                Image::Known(w) => push_mask(&mut terms, w),
                Image::Unknown(v) => {
                    for (r, tr) in terms.iter_mut().enumerate() {
                        tr.push(v + r as u32);
                    }
                }
            }
        }
        Some(
            terms
                .into_iter()
                .map(|mut tr| {
                    tr.sort_unstable();
                    let mut e: Vec<u32> = vec![];
                    for v in tr {
                        if e.last() == Some(&v) {
                            e.pop();
                        } else {
                            e.push(v);
                        }
                    }
                    e
                })
                .filter(|e| !e.is_empty())
                .collect(),
        )
    };
    let in_r = |gid: u32| {
        let (n, f) = ext.deg[ext.basis_bid[gid as usize] as usize];
        rb.get(n, f).is_some()
    };
    let mut bilinear = 0usize;
    let mut eqs: Vec<Vec<u32>> = vec![];
    let mut add = |e: Option<Vec<Vec<u32>>>, eqs: &mut Vec<Vec<u32>>| match e {
        Some(e) => eqs.extend(e),
        None => bilinear += 1,
    };
    // non-zero products where iota(g) iota(x) vanishes because one image lies in a zero group
    for (&(g, x), &m) in &ext.gen_prod {
        if ext.indec[g as usize] && !(in_r(g) && in_r(x)) {
            add(equation(g, x, m), &mut eqs);
        }
    }
    // every product of an indecomposable and a class whose images both may be non-zero, zero
    // products included
    let classes: Vec<u32> = (0..image.len() as u32).filter(|&x| in_r(x)).collect();
    for &g in classes.iter().filter(|&&g| ext.indec[g as usize]) {
        for &x in &classes {
            let m = ext.gen_prod.get(&(g, x)).copied().unwrap_or(0);
            add(equation(g, x, m), &mut eqs);
        }
        if eqs.len() >= 1 << 16 {
            prop.absorb(&eqs);
            eqs.clear();
        }
    }
    prop.absorb(&eqs);
    eprintln!("map: {bilinear} products with both images unknown skipped");
    let (sys, short) = prop.finish();
    let (pivot, rows, _, contra) = eliminate(nvars, &sys, false);
    println!("map: {} contradictions", contra.len() + short);
    let queries: Vec<Vec<u32>> = (0..nvars).map(|v| vec![v]).collect();
    let vals = crate::page::evaluate_with(nvars, &pivot, &rows, &queries, |_| false);

    let mut w = BufWriter::new(File::create(format!("{out}.csv")).unwrap());
    let mut ws = BufWriter::new(File::create(format!("{out}_status.csv")).unwrap());
    writeln!(w, "stem,Adams_filtration,index,image").unwrap();
    writeln!(ws, "stem,Adams_filtration,index,status,image").unwrap();
    let fmt = |m: Mask, d: usize| {
        let v: Vec<String> = (0..d).map(|r| (m >> r & 1).to_string()).collect();
        format!("\"[{}]\"", v.join(", "))
    };
    let (mut det, mut open, mut nonzero) = (0, 0, 0);
    for b in 0..nb as u32 {
        let (n, f) = ext.deg[b as usize];
        let Some(tb) = rb.get(n, f) else { continue };
        let d = rb.dim[tb as usize];
        for k in 0..ext.dim[b as usize] {
            match image[ext.gid(b, k) as usize].unwrap() {
                Image::Known(m) => {
                    if m != 0 {
                        writeln!(w, "{n},{f},{k},{}", fmt(m, d)).unwrap();
                    }
                }
                Image::Unknown(v) => {
                    let mut m: Mask = 0;
                    let mut all = true;
                    for r in 0..d {
                        let (val, is_det, _) = &vals[(v + r as u32) as usize];
                        all &= *is_det;
                        m |= (*val as Mask) << r;
                    }
                    let status = if all { "determined" } else { "open" };
                    writeln!(ws, "{n},{f},{k},{status},{}", fmt(m, d)).unwrap();
                    if all {
                        det += 1;
                        nonzero += (m != 0) as usize;
                        if m != 0 {
                            writeln!(w, "{n},{f},{k},{}", fmt(m, d)).unwrap();
                        }
                    } else {
                        open += 1;
                    }
                }
            }
        }
    }
    println!(
        "map: {} unknown classes: {det} determined ({nonzero} non-zero), {open} open, {:.1}s",
        unknown.len(),
        t.elapsed().as_secs_f64()
    );
}
