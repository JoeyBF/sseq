//! Certificates: the equations whose sum proves a forced value, and pools for minimising them.

use std::{fs::File, io::Write, time::Instant};

use rayon::prelude::*;

use crate::{implied, xor_sorted};

/// A set of equations whose sum (over F2) is the relation expressing the variable `x` through the
/// free variables, shrunk greedily (longest equations first) to a minimal subset that still implies
/// it. `prov[r]` lists the equation a row of the echelon form came from and the rows reduced into
/// it; `one` is the constant column. Returns the relation and the equations.
pub(crate) fn certificate(
    eqs: &[Vec<u32>],
    pivot: &[u32],
    rows: &[Vec<u32>],
    prov: &[(u32, Vec<u32>)],
    x: u32,
    one: u32,
) -> (Vec<u32>, Vec<u32>) {
    let t = Instant::now();
    assert_ne!(pivot[x as usize], u32::MAX, "the variable is free");
    // back-substitute: x + (free variables), as a sum of rows
    let mut rp = vec![false; rows.len()];
    let mut e = rows[pivot[x as usize] as usize].clone();
    rp[pivot[x as usize] as usize] = true;
    loop {
        // the constant column is never a pivot
        let Some(&c) = e
            .iter()
            .find(|&&c| c != x && c != one && pivot[c as usize] != u32::MAX)
        else {
            break;
        };
        let r = pivot[c as usize] as usize;
        e = xor_sorted(&e, &rows[r]);
        rp[r] ^= true;
    }
    // expand rows into equations, latest row first
    let mut ep = vec![false; eqs.len()];
    for r in (0..rows.len()).rev() {
        if rp[r] {
            ep[prov[r].0 as usize] ^= true;
            for &u in &prov[r].1 {
                rp[u as usize] ^= true;
            }
        }
    }
    let mut set: Vec<u32> = (0..eqs.len() as u32).filter(|&i| ep[i as usize]).collect();
    let sum = set.iter().fold(vec![], |acc: Vec<u32>, &i| {
        xor_sorted(&acc, &eqs[i as usize])
    });
    assert_eq!(sum, e, "certificate does not sum to the relation");
    eprintln!(
        "certificate: {} equations ({:.1}s)",
        set.len(),
        t.elapsed().as_secs_f64()
    );
    // greedy shrink, longest equations first; quadratic, so only for small sets
    if set.len() <= 20_000 {
        set.sort_by_key(|&i| std::cmp::Reverse(eqs[i as usize].len()));
        let mut k = 0;
        while k < set.len() {
            let trial: Vec<u32> = set
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != k)
                .map(|(_, &i)| i)
                .collect();
            if implied(eqs, &trial, &e) {
                set = trial;
            } else {
                k += 1;
            }
        }
        eprintln!(
            "shrunk: {} equations ({:.1}s)",
            set.len(),
            t.elapsed().as_secs_f64()
        );
    }
    (e, set)
}

/// Writes the pool `scripts/min_certificate.py` minimises over: the relation, a name per variable,
/// and every equation sharing a variable with the certificate `set` (with `closure`, every
/// equation inside the variables those touch).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dump_pool(
    nvars: u32,
    eqs: &[Vec<u32>],
    rel: &[u32],
    set: &[u32],
    path: &str,
    closure: bool,
    name: &dyn Fn(u32) -> String,
    origin: &dyn Fn(u32) -> String,
) {
    let mut v0 = vec![false; nvars as usize + 1];
    for &i in set {
        for &y in &eqs[i as usize] {
            v0[y as usize] = true;
        }
    }
    let touching: Vec<u32> = (0..eqs.len() as u32)
        .into_par_iter()
        .filter(|&i| eqs[i as usize].iter().any(|&y| v0[y as usize]))
        .collect();
    let pool: Vec<u32> = if closure {
        let mut v1 = v0.clone();
        for &i in &touching {
            for &y in &eqs[i as usize] {
                v1[y as usize] = true;
            }
        }
        (0..eqs.len() as u32)
            .into_par_iter()
            .filter(|&i| eqs[i as usize].iter().all(|&y| v1[y as usize]))
            .collect()
    } else {
        touching.clone()
    };
    let mut w = std::io::BufWriter::new(File::create(path).unwrap());
    let rel_s: Vec<String> = rel.iter().map(|y| y.to_string()).collect();
    writeln!(w, "relation {}", rel_s.join(" ")).unwrap();
    let mut named = vec![false; nvars as usize + 1];
    for &i in &pool {
        for &y in &eqs[i as usize] {
            if !named[y as usize] {
                named[y as usize] = true;
                writeln!(w, "name {y} {}", name(y)).unwrap();
            }
        }
    }
    for &i in &pool {
        let vs: Vec<String> = eqs[i as usize].iter().map(|y| y.to_string()).collect();
        writeln!(w, "eq {i} {} : {}", origin(i), vs.join(" ")).unwrap();
    }
    eprintln!(
        "pool: {} touching, {} in the pool, written to {path}",
        touching.len(),
        pool.len()
    );
}
