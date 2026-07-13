//! Signature-order diagnostic: which signature convention gives a right/left-stable
//! (acyclic) filtration over A_C/τ, and on which multiplication side.
use std::collections::{BTreeMap, BTreeSet};

use algebra::{Algebra, motivic::CTauAlgebra};
use fp::{prime::TWO, vector::FpVector};

type Sig = (u32, Vec<u32>);

fn sig_of(e: u32, r: &[u32], qlen: usize, prof: &[u8], ext_high: bool, poly_high: bool) -> Sig {
    let qmask = if qlen >= 32 { !0 } else { (1u32 << qlen) - 1 };
    let q = if ext_high { e & !qmask } else { e & qmask };
    let p = prof
        .iter()
        .enumerate()
        .map(|(i, &pp)| {
            let rk = r.get(i + 1).copied().unwrap_or(0);
            if poly_high { rk >> pp } else { rk & ((1 << pp) - 1) }
        })
        .collect();
    (q, p)
}

fn acyclic(edges: &BTreeSet<(Sig, Sig)>, nodes: &BTreeSet<Sig>) -> bool {
    let mut indeg: BTreeMap<Sig, usize> = nodes.iter().map(|n| (n.clone(), 0)).collect();
    let mut adj: BTreeMap<Sig, Vec<Sig>> = BTreeMap::new();
    for (a, b) in edges {
        adj.entry(a.clone()).or_default().push(b.clone());
        *indeg.get_mut(b).unwrap() += 1;
    }
    let mut q: Vec<Sig> = indeg
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(n, _)| n.clone())
        .collect();
    let mut count = 0;
    while let Some(n) = q.pop() {
        count += 1;
        if let Some(ns) = adj.get(&n) {
            for m in ns.clone() {
                let d = indeg.get_mut(&m).unwrap();
                *d -= 1;
                if *d == 0 {
                    q.push(m);
                }
            }
        }
    }
    count == nodes.len()
}

fn test(name: &str, qlen: usize, prof: &[u8], eh: bool, ph: bool, right: bool) {
    let alg = CTauAlgebra::new();
    alg.compute_basis(28);
    let mut edges = BTreeSet::new();
    let mut nodes = BTreeSet::new();
    for ta in 0..=12 {
        for ia in 0..alg.dimension(ta) {
            for tb in 0..=(12 - ta) {
                for ib in 0..alg.dimension(tb) {
                    let (ef, rf) = if right {
                        alg.engine().basis_element(tb, ib).clone()
                    } else {
                        alg.engine().basis_element(ta, ia).clone()
                    };
                    let sf = sig_of(ef, &rf, qlen, prof, eh, ph);
                    nodes.insert(sf.clone());
                    let mut out = FpVector::new(TWO, alg.dimension(ta + tb));
                    alg.multiply_basis_elements(out.as_slice_mut(), 1, ta, ia, tb, ib);
                    for (idx, _) in out.iter_nonzero() {
                        let (e, r) = alg.engine().basis_element(ta + tb, idx);
                        let so = sig_of(*e, r, qlen, prof, eh, ph);
                        nodes.insert(so.clone());
                        if so != sf {
                            edges.insert((sf.clone(), so.clone()));
                        }
                    }
                }
            }
        }
    }
    // Signature degree in the low-bit (classical) convention.
    let sigdeg = |s: &Sig| -> i32 {
        let mut d = 0i32;
        for i in 0..qlen {
            if (s.0 >> i) & 1 != 0 {
                d += (1 << (i + 1)) - 1;
            }
        }
        for (i, &v) in s.1.iter().enumerate() {
            d += v as i32 * ((1 << (i + 2)) - 2);
        }
        d
    };
    let ac = acyclic(&edges, &nodes);
    // Is "sort by signature degree ascending" a valid linear extension (every edge
    // non-decreasing in signature degree, strict edges where degrees tie handled)?
    let mut deg_monotone = true;
    for (a, b) in &edges {
        if sigdeg(b) < sigdeg(a) {
            deg_monotone = false;
        }
    }
    println!(
        "{name} qlen={qlen} prof={prof:?} eh={eh} ph={ph} right={right}: {} deg_monotone={deg_monotone} ({} nodes, {} edges)",
        if ac { "ACYCLIC" } else { "CYCLE" },
        nodes.len(),
        edges.len()
    );
}

fn main() {
    println!("=== pure exterior E(Q_0) ===");
    for eh in [false, true] {
        for rt in [false, true] {
            test("E(Q0)", 1, &[], eh, false, rt);
        }
    }
    println!("=== pure exterior E(Q_0,Q_1) ===");
    for eh in [false, true] {
        for rt in [false, true] {
            test("E(Q0,Q1)", 2, &[], eh, false, rt);
        }
    }
    println!("=== pure poly xi_1<2 ===");
    for ph in [false, true] {
        for rt in [false, true] {
            test("poly[1]", 0, &[1], false, ph, rt);
        }
    }
    println!("=== A(1) all ===");
    for eh in [false, true] {
        for ph in [false, true] {
            for rt in [false, true] {
                test("A(1)", 2, &[1], eh, ph, rt);
            }
        }
    }
}
