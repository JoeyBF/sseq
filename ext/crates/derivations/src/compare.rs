//! d2 on the sphere and on a comparison ring spectrum (tmf, ...) solved jointly, coupled by the
//! map on E2 induced by a ring map S -> R (`--compare <ring dir> <iota csv>`).

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use rustc_hash::FxHashMap;

use crate::{
    Ext, Mask, Vars, build_products, check_products, eliminate, equations, load, parse_vec, var,
    vars,
};

/// The map on E2, `(stem, filtration, index)` of a sphere basis element -> its image over the
/// comparison basis of the same bidegree. Missing rows are zero.
fn load_iota(path: &str) -> FxHashMap<(i32, i32, usize), Mask> {
    let mut out = FxHashMap::default();
    for line in BufReader::new(File::open(path).unwrap()).lines().skip(1) {
        let line = line.unwrap();
        let mut it = line.splitn(4, ',');
        let n: i32 = it.next().unwrap().parse().unwrap();
        let f: i32 = it.next().unwrap().parse().unwrap();
        let i: usize = it.next().unwrap().parse().unwrap();
        out.insert((n, f, i), parse_vec(it.next().unwrap()));
    }
    out
}

/// `eqs` are the sphere's equations (Leibniz and seeds) with the constant column `v.n`.
pub(crate) fn run(ext: &Ext, v: &Vars, eqs: Vec<Vec<u32>>, gens_only: bool, args: &[String]) {
    let i = args.iter().position(|a| a == "--compare").unwrap();
    let (dir, iota_path) = (&args[i + 1], &args[i + 2]);
    let ext_b = load(dir, ext.n_max, ext.f_max);
    let p_b = build_products(&ext_b);
    check_products(&ext_b, &p_b);
    let v_b = vars(&ext_b, &[]);
    let (eqs_b, _) = equations(&ext_b, &p_b, &v_b, gens_only);
    let shift = v.n;
    let joint = v.n + v_b.n;
    let mut all: Vec<Vec<u32>> = eqs
        .into_iter()
        .map(|e| {
            e.into_iter()
                .map(|x| if x == v.n { joint } else { x })
                .collect()
        })
        .collect();
    let n_sphere = all.len();
    all.extend(
        eqs_b
            .into_iter()
            .map(|e| e.into_iter().map(|x| x + shift).collect::<Vec<u32>>()),
    );
    let n_b = all.len() - n_sphere;

    // naturality: d2(iota x) = iota(d2 x) for every sphere basis element x
    let iota = load_iota(iota_path);
    let im = |n: i32, f: i32, i: usize| iota.get(&(n, f, i)).copied().unwrap_or(0);
    let mut n_nat = 0;
    for c in 0..ext.deg.len() as u32 {
        let Some(cp) = ext.prime(c) else { continue };
        let (n, f) = ext.deg[c as usize];
        let Some(bp) = ext_b.get(n - 1, f + 2) else {
            continue;
        };
        let b = ext_b.get(n, f);
        for x in 0..ext.dim[c as usize] {
            let ix = im(n, f, x);
            for r in 0..ext_b.dim[bp as usize] {
                let mut e: Vec<u32> = vec![];
                if let Some(b) = b {
                    for j in (0..ext_b.dim[b as usize]).filter(|&j| ix >> j & 1 == 1) {
                        e.push(var(&ext_b, &v_b, b, j, r) + shift);
                    }
                }
                for k in 0..ext.dim[cp as usize] {
                    if im(n - 1, f + 2, k) >> r & 1 == 1 {
                        e.push(var(ext, v, c, x, k));
                    }
                }
                e.sort_unstable();
                if !e.is_empty() {
                    all.push(e);
                    n_nat += 1;
                }
            }
        }
    }
    eprintln!(
        "compare: {} comparison variables, {n_b} comparison Leibniz and {n_nat} naturality \
         equations",
        v_b.n
    );
    let (pivot, rows, _, contradictions) = eliminate(joint, &all, false);
    eprintln!("joint: {} contradictions", contradictions.len());

    // what is forced of d2 on the comparison spectrum
    let mut queries = vec![];
    let mut classes = vec![];
    for b in 0..ext_b.deg.len() as u32 {
        let Some(bp) = ext_b.prime(b) else { continue };
        for j in 0..ext_b.dim[b as usize] {
            classes.push((b, j));
            for r in 0..ext_b.dim[bp as usize] {
                queries.push(vec![var(&ext_b, &v_b, b, j, r) + shift]);
            }
        }
    }
    let vals = crate::page::evaluate(joint, &pivot, &rows, &queries);
    let out = args
        .iter()
        .position(|a| a == "--compare-solution")
        .map(|k| BufWriter::new(File::create(&args[k + 1]).unwrap()));
    let mut w = out;
    if let Some(w) = w.as_mut() {
        writeln!(w, "stem,Adams_filtration,index,status,target").unwrap();
    }
    let mut summary: BTreeMap<i32, (usize, usize, usize)> = BTreeMap::new();
    let mut qi = 0;
    for (b, j) in classes {
        let m = ext_b.dim[ext_b.prime(b).unwrap() as usize];
        let vs = &vals[qi..qi + m];
        qi += m;
        let det = vs.iter().all(|(_, d)| d.is_empty());
        let nz = vs.iter().any(|(v, _)| *v);
        let (n, f) = ext_b.deg[b as usize];
        let s = summary.entry(n).or_default();
        s.0 += 1;
        if det {
            s.1 += 1;
            if nz {
                s.2 += 1;
            }
        }
        if let Some(w) = w.as_mut() {
            let t: Vec<String> = vs.iter().map(|(v, _)| (*v as u8).to_string()).collect();
            let st = if det { "determined" } else { "open" };
            writeln!(w, "{n},{f},{j},{st},\"[{}]\"", t.join(", ")).unwrap();
        }
    }
    let (tc, td, tz) = summary
        .values()
        .fold((0, 0, 0), |a, s| (a.0 + s.0, a.1 + s.1, a.2 + s.2));
    println!(
        "comparison d2: {tc} classes, {td} determined ({tz} nonzero), {} open",
        tc - td
    );
}
