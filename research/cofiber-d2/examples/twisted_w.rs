//! Solve the twisted cobar equation for the path weights W of an arbitrary finite module M:
//! for cells c, b with delta = |c| - |b| >= 1 and Milnor basis triples (a1, a2, a3) (a1 the top
//! step) with |a1| + |a2| + |a3| = delta + 1,
//!
//!   W_cb(a1 a2, a3) + W_cb(a1, a2 a3)
//!     + sum_b' D_cb'(a3) W_b'b(a1, a2) + sum_b' W_cb'(a2, a3) D_b'b(a1)  =  O_cb(a1, a2, a3),
//!
//! where D_cb'(a) = <c, chi(a) b'> (sseq's delta_Q) and O_cb = <c, X(A(a1, sigma(a2) sigma(a3))) b>
//! with X = chi (OBST=chi, default) or the identity (OBST=plain); A is the pair algebra's A-map.
//! Prints a solution per block as a cocfit spec `c_deg>b_deg:d:P(..)|P(..)+...`, or reports the
//! equations inconsistent.
//!
//! Usage: `twisted_w <module>`.

use std::{collections::BTreeMap, sync::Arc};

use algebra::{
    Algebra, MilnorAlgebra,
    module::{FDModule, Module},
    pair_algebra::PairAlgebra,
};
use ext::{ext_algebra::tensor_resolution::Antipode, utils::load_module_json};
use fp::vector::FpVector;

type Cell = (i32, usize);

fn main() -> anyhow::Result<()> {
    let name = std::env::args().nth(1).unwrap();
    let obst_chi = std::env::var("OBST").map_or(true, |v| v != "plain");
    let no_twist = std::env::var("NO_TWIST").is_ok();
    let algebra = Arc::new(MilnorAlgebra::new(fp::prime::TWO, false));
    let m = FDModule::from_json(Arc::clone(&algebra), &load_module_json(&name)?)?;
    let top = m.max_degree().unwrap();
    let bot = m.min_degree();
    algebra.compute_basis(top - bot + 4);
    m.compute_basis(top + 2);
    let chi = Antipode::new(Arc::clone(&algebra));
    let cells: Vec<Cell> = (bot..=top).flat_map(|d| (0..m.dimension(d)).map(move |i| (d, i))).collect();
    // <c, x b> for x a vector in degree |c| - |b|
    let act = |x: &FpVector, b: Cell, c: Cell| -> u32 {
        let d = c.0 - b.0;
        let mut out = FpVector::new(fp::prime::TWO, m.dimension(c.0));
        for (i, v) in x.iter_nonzero() {
            m.act_on_basis(out.as_slice_mut(), v, d, i, b.0, b.1);
        }
        out.entry(c.1)
    };
    let basis_vec = |d: i32, i: usize| {
        let mut v = FpVector::new(fp::prime::TWO, algebra.dimension(d));
        v.set_entry(i, 1);
        v
    };
    // D_{c -> b'}(a) = <c, chi(a) b'>
    let dmap = |a: (i32, usize), b: Cell, c: Cell| -> u32 {
        if c.0 - b.0 != a.0 {
            return 0;
        }
        act(&chi.apply(a.0, a.1), b, c)
    };
    let s = |d: i32, i: usize| algebra.basis_element_to_string(d, i);
    // The module's Z/4 carry. N(a; b→c): how many Milnor terms of χ(a) carry b to c (an integer,
    // where δ_Q only sees it mod 2).
    let nint = |a: (i32, usize), b: Cell, c: Cell| -> i64 {
        if c.0 - b.0 != a.0 || a.0 < 0 {
            return 0;
        }
        if a.0 == 0 {
            return (b == c) as i64;
        }
        chi.apply(a.0, a.1)
            .iter_nonzero()
            .map(|(o, _)| act(&basis_vec(a.0, o), b, c) as i64)
            .sum()
    };
    // u(a2, a3; m→c) = (Σ_m' N(a2; m→m') N(a3; m'→c) − Σ_e [σ(a2)σ(a3)]_e N(e; m→c)) / 2 mod 2
    let carry = |a2: (i32, usize), a3: (i32, usize), mm: Cell, c: Cell| -> u32 {
        let mut r: i64 = 0;
        for &mp in &cells {
            if mp.0 == mm.0 + a2.0 {
                r += nint(a2, mm, mp) * nint(a3, mp, c);
            }
        }
        let mut e = algebra.new_pair_element(a2.0 + a3.0);
        algebra.sigma_multiply_basis(&mut e, 1, a2.0, a2.1, a3.0, a3.1);
        let mut sv: i64 = 0;
        for i in 0..algebra.dimension(a2.0 + a3.0) {
            let v = e.ones().entry(i) as i64 + 2 * e.twos().entry(i) as i64;
            if v != 0 {
                sv += v * nint((a2.0 + a3.0, i), mm, c);
            }
        }
        let diff = (r - sv).rem_euclid(4);
        if diff % 2 != 0 {
            eprintln!("odd module carry at {mm:?}->{c:?}");
        }
        ((diff / 2) & 1) as u32
    };
    // κ(a1) as a vector: A(a1, 2·1)
    let kappa = |a1: (i32, usize)| -> FpVector {
        let mut out = FpVector::new(fp::prime::TWO, algebra.dimension(a1.0 - 1));
        let mut two = algebra.new_pair_element(0);
        two.twos_mut().set_entry(0, 1);
        <MilnorAlgebra as PairAlgebra>::finalize_element(&mut two);
        algebra.a_multiply(out.as_slice_mut(), 1, a1.0, basis_vec(a1.0, a1.1).as_slice(), 0, &two);
        out
    };
    let corr_mode = std::env::var("CORR").unwrap_or_default();
    // EXACT=<file>: the obstruction ε(A_Q([a1|a2|a3] ⊗ b)) from bar_o, and the twist of the exact
    // equation (every coproduct split of a1)
    let exact: Option<std::collections::HashSet<(String, Cell, Cell)>> =
        std::env::var("EXACT").ok().map(|path| {
            std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .skip(1)
                .map(|l| {
                    let f: Vec<&str> = l.rsplitn(5, ',').collect();
                    let p = |i: usize| f[i].parse::<i32>().unwrap();
                    (f[4].to_string(), (p(3), p(2) as usize), (p(1), p(0) as usize))
                })
                .collect()
        });
    let cop = ext::ext_algebra::tensor_resolution::FullCoproduct::new(Arc::clone(&algebra));
    let corr = |a1: (i32, usize), a2: (i32, usize), a3: (i32, usize), b: Cell, c: Cell| -> u32 {
        let mut t = 0;
        match corr_mode.as_str() {
            // κ(a1) acts first (through χ), then the module carry of (a2, a3)
            "1" => {
                let k = kappa(a1);
                for &mm in &cells {
                    if mm.0 != b.0 + a1.0 - 1 {
                        continue;
                    }
                    let mut chik = FpVector::new(fp::prime::TWO, algebra.dimension(a1.0 - 1));
                    for (i, v) in k.iter_nonzero() {
                        if a1.0 - 1 == 0 {
                            chik.add_basis_element(i, v);
                        } else {
                            chik.add(&chi.apply(a1.0 - 1, i), v);
                        }
                    }
                    let first = if a1.0 == 1 { (b == mm) as u32 } else { act(&chik, b, mm) };
                    t ^= first & carry(a2, a3, mm, c);
                }
            }
            // only a1 = Sq1, carry of (a2, a3) from b
            "3" => {
                if a1.0 == 1 {
                    t = carry(a2, a3, b, c);
                }
            }
            // the carry of (a1, a2) first, then κ(a3)
            "2" => {
                for &mm in &cells {
                    if mm.0 != b.0 + a1.0 + a2.0 {
                        continue;
                    }
                    let first = carry(a1, a2, b, mm);
                    if first == 0 {
                        continue;
                    }
                    let k = kappa(a3);
                    let second = if a3.0 == 1 {
                        (mm == c) as u32
                    } else {
                        let mut chik = FpVector::new(fp::prime::TWO, algebra.dimension(a3.0 - 1));
                        for (i, v) in k.iter_nonzero() {
                            chik.add(&chi.apply(a3.0 - 1, i), v);
                        }
                        act(&chik, mm, c)
                    };
                    t ^= second;
                }
            }
            _ => {}
        }
        t
    };
    // unknowns
    let blocks: Vec<(Cell, Cell)> = cells
        .iter()
        .flat_map(|&c| cells.iter().filter(move |&&b| c.0 > b.0).map(move |&b| (c, b)))
        .collect();
    let mut unk: BTreeMap<(Cell, Cell, i32, usize, i32, usize), usize> = BTreeMap::new();
    let mut unk_list = vec![];
    for &(c, b) in &blocks {
        let delta = c.0 - b.0;
        for d1 in 1..=delta {
            let d2 = delta + 1 - d1;
            for i1 in 0..algebra.dimension(d1) {
                for i2 in 0..algebra.dimension(d2) {
                    unk.insert((c, b, d1, i1, d2, i2), unk_list.len());
                    unk_list.push((c, b, d1, i1, d2, i2));
                }
            }
        }
    }
    let n = unk_list.len();
    let words = n / 64 + 2;
    let mut rows: Vec<Vec<u64>> = vec![];
    let mut labels: Vec<String> = vec![];
    let set = |row: &mut Vec<u64>, k: usize| row[k / 64] ^= 1 << (k % 64);
    let mut n_obst = 0;
    for &(c, b) in &blocks {
        let delta = c.0 - b.0;
        for d1 in 1..=delta {
            for d2 in 1..=delta {
                let d3 = delta + 1 - d1 - d2;
                if d3 < 1 {
                    continue;
                }
                for i1 in 0..algebra.dimension(d1) {
                    for i2 in 0..algebra.dimension(d2) {
                        let mut p12 = FpVector::new(fp::prime::TWO, algebra.dimension(d1 + d2));
                        algebra.multiply_basis_elements(p12.as_slice_mut(), 1, d1, i1, d2, i2);
                        for i3 in 0..algebra.dimension(d3) {
                            let mut row = vec![0u64; words];
                            for (e, v) in p12.iter_nonzero() {
                                if v != 0 {
                                    set(&mut row, unk[&(c, b, d1 + d2, e, d3, i3)]);
                                }
                            }
                            let mut p23 = FpVector::new(fp::prime::TWO, algebra.dimension(d2 + d3));
                            algebra.multiply_basis_elements(p23.as_slice_mut(), 1, d2, i2, d3, i3);
                            for (e, v) in p23.iter_nonzero() {
                                if v != 0 {
                                    set(&mut row, unk[&(c, b, d1, i1, d2 + d3, e)]);
                                }
                            }
                            if exact.is_some() {
                                // Σ_{(a1)} a1'·W(a2, a3)(χ(a1'') b)
                                for &(ld, li, rd, ri) in cop.terms(d1, i1).iter() {
                                    for &mm in &cells {
                                        if mm.0 != b.0 + rd {
                                            continue;
                                        }
                                        let al = if rd == 0 {
                                            (mm == b) as u32
                                        } else {
                                            act(&chi.apply(rd, ri), b, mm)
                                        };
                                        if al == 0 {
                                            continue;
                                        }
                                        for &cp in &cells {
                                            if cp.0 != c.0 - ld {
                                                continue;
                                            }
                                            let be = if ld == 0 {
                                                (cp == c) as u32
                                            } else {
                                                act(&basis_vec(ld, li), cp, c)
                                            };
                                            if be == 0 {
                                                continue;
                                            }
                                            if let Some(&k) = unk.get(&(cp, mm, d2, i2, d3, i3)) {
                                                set(&mut row, k);
                                            }
                                        }
                                    }
                                }
                            } else if !no_twist {
                                for &bp in &cells {
                                    // D_{c -> b'}(a3) W_{b' -> b}(a1, a2)
                                    if bp.0 > b.0 && c.0 - bp.0 == d3 && dmap((d3, i3), bp, c) != 0 {
                                        set(&mut row, unk[&(bp, b, d1, i1, d2, i2)]);
                                    }
                                    // W_{c -> b'}(a2, a3) D_{b' -> b}(a1)
                                    if bp.0 < c.0 && bp.0 - b.0 == d1 && dmap((d1, i1), b, bp) != 0 {
                                        set(&mut row, unk[&(c, bp, d2, i2, d3, i3)]);
                                    }
                                }
                            }
                            // the obstruction
                            let mut e = algebra.new_pair_element(d2 + d3);
                            algebra.sigma_multiply_basis(&mut e, 1, d2, i2, d3, i3);
                            let mut tw = algebra.new_pair_element(d2 + d3);
                            tw.twos_mut().add(e.twos(), 1);
                            for (k, r) in e.ys().iter().enumerate() {
                                for (l, v) in r.iter().enumerate() {
                                    tw.ys_mut()[k][l].add(v, 1);
                                }
                            }
                            <MilnorAlgebra as PairAlgebra>::finalize_element(&mut tw);
                            let mut out = FpVector::new(fp::prime::TWO, algebra.dimension(delta));
                            algebra.a_multiply(
                                out.as_slice_mut(),
                                1,
                                d1,
                                basis_vec(d1, i1).as_slice(),
                                d2 + d3,
                                &tw,
                            );
                            let x = if obst_chi {
                                let mut x = FpVector::new(fp::prime::TWO, algebra.dimension(delta));
                                for (i, v) in out.iter_nonzero() {
                                    x.add(&chi.apply(delta, i), v);
                                }
                                x
                            } else {
                                out
                            };
                            let o = if let Some(ex) = &exact {
                                ex.contains(&(format!("{}|{}|{}", s(d1, i1), s(d2, i2), s(d3, i3)), b, c)) as u32
                            } else {
                                act(&x, b, c) ^ corr((d1, i1), (d2, i2), (d3, i3), b, c)
                            };
                            n_obst += o as usize;
                            if o != 0 {
                                row[n / 64 + 1] |= 1 << 63; // rhs flag
                            }
                            labels.push(format!("{c:?}->{b:?} {}|{}|{} obst={o}", s(d1, i1), s(d2, i2), s(d3, i3)));
                            let lab = format!("{c:?}->{b:?} {}|{}|{}", s(d1, i1), s(d2, i2), s(d3, i3));
                            if std::env::var("DROP").is_ok_and(|d| d == lab) {
                                labels.pop();
                                continue;
                            }
                            rows.push(row);
                        }
                    }
                }
            }
        }
    }
    // NORM=nopath: W vanishes on blocks (c, b) with c not in A·b
    if std::env::var("NORM").is_ok_and(|v| v == "nopath") {
        for &(c, b) in &blocks {
            let delta = c.0 - b.0;
            let connected = (0..algebra.dimension(delta)).any(|i| act(&basis_vec(delta, i), b, c) != 0);
            if connected {
                continue;
            }
            for (k, u) in unk_list.iter().enumerate() {
                if (u.0, u.1) == (c, b) {
                    let mut row = vec![0u64; words];
                    set(&mut row, k);
                    labels.push(format!("norm {c:?}->{b:?}"));
                    rows.push(row);
                }
            }
        }
    }
    // Gaussian elimination, rhs in the last bit
    let rhs = |r: &Vec<u64>| r[n / 64 + 1] >> 63 & 1;
    let top_bit = |r: &Vec<u64>| -> Option<usize> {
        (0..n).rev().find(|&k| r[k / 64] >> (k % 64) & 1 != 0)
    };
    let mut piv: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    let mut inconsistent = 0;
    let rows_copy = rows.clone();
    let nr = rows.len();
    let mut combs: BTreeMap<usize, Vec<bool>> = BTreeMap::new();
    for (ri, mut r) in rows.into_iter().enumerate() {
        let mut comb = vec![false; nr];
        comb[ri] = true;
        loop {
            match top_bit(&r) {
                None => {
                    if rhs(&r) != 0 {
                        inconsistent += 1;
                        eprintln!("contradiction from:");
                        for (k, &t) in comb.iter().enumerate() {
                            if t {
                                eprintln!("  {}", labels[k]);
                            }
                        }
                    }
                    break;
                }
                Some(p) => match piv.get(&p) {
                    Some(pr) => {
                        for w in 0..words {
                            r[w] ^= pr[w];
                        }
                        let pc = &combs[&p];
                        for k in 0..nr {
                            comb[k] ^= pc[k];
                        }
                    }
                    None => {
                        piv.insert(p, r);
                        combs.insert(p, comb);
                        break;
                    }
                },
            }
        }
    }
    eprintln!(
        "{name}: {n} unknowns, rank {}, obstruction entries {n_obst}, {}",
        piv.len(),
        if inconsistent == 0 { "CONSISTENT".to_string() } else { format!("INCONSISTENT ({inconsistent})") }
    );
    // back substitution, free variables 0
    let mut sol = vec![0u64; words];
    for (&p, r) in &piv {
        let mut v = rhs(r);
        for k in 0..p {
            if r[k / 64] >> (k % 64) & 1 != 0 && sol[k / 64] >> (k % 64) & 1 != 0 {
                v ^= 1;
            }
        }
        if v != 0 {
            sol[p / 64] |= 1 << (p % 64);
        }
    }
    if std::env::var("RESID").is_ok() {
        for (k, r) in rows_copy.iter().enumerate() {
            let mut v = rhs(r);
            for w in 0..=n / 64 {
                let mask = if w == n / 64 { (1u64 << (n % 64)) - 1 } else { !0 };
                v ^= ((r[w] & sol[w] & mask).count_ones() & 1) as u64;
            }
            if v != 0 {
                eprintln!("violated: {}", labels[k]);
            }
        }
    }
    let mut spec: BTreeMap<(i32, i32), Vec<String>> = BTreeMap::new();
    for &(c, b) in &blocks {
        spec.entry((c.0, b.0)).or_default();
    }
    for (k, &(c, b, d1, i1, d2, i2)) in unk_list.iter().enumerate() {
        if sol[k / 64] >> (k % 64) & 1 != 0 {
            spec.get_mut(&(c.0, b.0)).unwrap().push(format!("d:{}|{}", s(d1, i1), s(d2, i2)));
        }
    }
    for ((cd, bd), v) in spec {
        println!("{cd}>{bd}:{}", if v.is_empty() { "0".to_string() } else { v.join("+") });
    }
    // SPEC_OUT=<file>: everything module_d2 needs, per pair of cells: `block,c_deg,c_idx,b_deg,
    // b_idx,op+op` (the antipode shear `H:e`, the path weight `d:a|a'`, and `H:1` on the diagonal)
    if let Ok(path) = std::env::var("SPEC_OUT") {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "module,{name}")?;
        for &c in &cells {
            writeln!(f, "block,{},{},{},{},H:1", c.0, c.1, c.0, c.1)?;
        }
        for &(c, b) in &blocks {
            let delta = c.0 - b.0;
            let mut ops = vec![];
            for e in 0..algebra.dimension(delta) {
                if act(&chi.apply(delta, e), b, c) != 0 {
                    ops.push(format!("H:{}", s(delta, e)));
                }
            }
            for (k, &(cc, bb, d1, i1, d2, i2)) in unk_list.iter().enumerate() {
                if (cc, bb) == (c, b) && sol[k / 64] >> (k % 64) & 1 != 0 {
                    ops.push(format!("d:{}|{}", s(d1, i1), s(d2, i2)));
                }
            }
            if !ops.is_empty() {
                writeln!(f, "block,{},{},{},{},{}", c.0, c.1, b.0, b.1, ops.join("+"))?;
            }
        }
    }
    // NULL_OUT=<file>: a basis of the homogeneous solutions, one per free variable, as
    // `name|c>b:op+op|c>b:op...`
    if let Ok(path) = std::env::var("NULL_OUT") {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        for free in (0..n).filter(|k| !piv.contains_key(k)) {
            let mut sol = vec![0u64; words];
            sol[free / 64] |= 1 << (free % 64);
            for (&p, r) in &piv {
                let mut v = 0;
                for k in 0..p {
                    if r[k / 64] >> (k % 64) & 1 != 0 && sol[k / 64] >> (k % 64) & 1 != 0 {
                        v ^= 1;
                    }
                }
                if v != 0 {
                    sol[p / 64] |= 1 << (p % 64);
                }
            }
            let mut by: BTreeMap<(i32, i32), Vec<String>> = BTreeMap::new();
            for (k, &(c, b, d1, i1, d2, i2)) in unk_list.iter().enumerate() {
                if sol[k / 64] >> (k % 64) & 1 != 0 {
                    by.entry((c.0, b.0)).or_default().push(format!("d:{}|{}", s(d1, i1), s(d2, i2)));
                }
            }
            let (c, b, d1, i1, d2, i2) = unk_list[free];
            let name = format!("{}>{}:{}|{}", c.0, b.0, s(d1, i1), s(d2, i2));
            let blocks: Vec<String> =
                by.iter().map(|((cd, bd), v)| format!("{cd}>{bd}:{}", v.join("+"))).collect();
            writeln!(f, "{name}#{}", blocks.join("#"))?;
        }
    }
    Ok(())
}
