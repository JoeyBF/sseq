//! Union-find propagation of the short F_2 equations before sparse elimination.
//!
//! Equations are sets of variables summing to zero; the constant column `nvars` stands for the
//! value 1. One-term equations fix a variable and two-term ones identify two variables (up to the
//! constant), which a union-find with parity absorbs without any elimination. Batches are
//! rewritten in parallel against a frozen union-find, then their short results are applied in
//! order. The output is an equivalent system: the remaining long equations, rewritten in terms of
//! class representatives, plus one row `x + root (+ 1)` per merged variable, so that for a fixed
//! column order elimination finds the same pivots and free columns as on the original system.

use rayon::prelude::*;

/// Equations per batch: large enough that the parallel rewrite amortises its overhead, small
/// enough that the union-find a batch is rewritten against is never far behind.
const BATCH: usize = 1024;

struct UnionFind {
    parent: Vec<u32>,
    /// value(x) = value(parent) + parity
    parity: Vec<u8>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            parity: vec![0; n],
            rank: vec![0; n],
        }
    }

    /// Root and parity of `x`, without path compression (callable from parallel readers).
    fn find(&self, mut x: u32) -> (u32, u8) {
        let mut p = 0;
        while self.parent[x as usize] != x {
            p ^= self.parity[x as usize];
            x = self.parent[x as usize];
        }
        (x, p)
    }

    /// Imposes value(a) + value(b) = c; returns false on a contradiction.
    fn union(&mut self, a: u32, b: u32, c: u8) -> bool {
        let (ra, pa) = self.find(a);
        let (rb, pb) = self.find(b);
        let c = c ^ pa ^ pb;
        if ra == rb {
            return c == 0;
        }
        // the constant (the last node) always stays a root, so "parity 1" keeps meaning "+ 1"
        let one = self.parent.len() as u32 - 1;
        let (child, root) = if ra == one {
            (rb, ra)
        } else if rb == one || self.rank[ra as usize] < self.rank[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        if self.rank[ra as usize] == self.rank[rb as usize] {
            self.rank[root as usize] += 1;
        }
        self.parent[child as usize] = root;
        self.parity[child as usize] = c;
        true
    }

    /// Rewrites an equation in representatives (the constant `one` is itself a representative).
    fn rewrite(&self, eq: &[u32], one: u32) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::with_capacity(eq.len() + 1);
        let mut constant = 0;
        for &x in eq {
            let (r, p) = self.find(x);
            out.push(r);
            constant ^= p;
        }
        if constant == 1 {
            out.push(one);
        }
        out.sort_unstable();
        let mut dedup: Vec<u32> = Vec::with_capacity(out.len());
        for v in out {
            if dedup.last() == Some(&v) {
                dedup.pop();
            } else {
                dedup.push(v);
            }
        }
        dedup
    }
}

/// Short-equation outcome of a rewritten equation; `None` means it is long and must be kept.
fn absorb(uf: &mut UnionFind, eq: &[u32], one: u32, contradictions: &mut usize) -> bool {
    match eq {
        [] => true,
        [a] if *a == one => {
            *contradictions += 1;
            true
        }
        [a] => {
            if !uf.union(*a, one, 1) {
                *contradictions += 1;
            }
            true
        }
        [a, b] => {
            if !uf.union(*a, *b, 0) {
                *contradictions += 1;
            }
            true
        }
        _ => false,
    }
}

/// Absorbs equations batch by batch, keeping only the union-find and the long remainder, so
/// equations can be generated, absorbed and dropped without ever being stored together.
pub(crate) struct Propagator {
    uf: UnionFind,
    one: u32,
    long: Vec<Vec<u32>>,
    contradictions: usize,
    absorbed: usize,
    start: std::time::Instant,
}

impl Propagator {
    pub(crate) fn new(nvars: u32) -> Self {
        Self {
            uf: UnionFind::new(nvars as usize + 1),
            one: nvars,
            long: vec![],
            contradictions: 0,
            absorbed: 0,
            start: std::time::Instant::now(),
        }
    }

    pub(crate) fn absorb(&mut self, eqs: &[Vec<u32>]) {
        self.absorbed += eqs.len();
        for batch in eqs.chunks(BATCH) {
            let uf = &self.uf;
            let one = self.one;
            let rewritten: Vec<Vec<u32>> = batch.par_iter().map(|e| uf.rewrite(e, one)).collect();
            self.settle(rewritten);
        }
    }

    /// Applies rewritten equations in order: short ones to the union-find, long ones kept.
    fn settle(&mut self, rewritten: Vec<Vec<u32>>) {
        for eq in rewritten {
            // rewrite again: earlier equations of the batch may have merged its variables
            let eq = if eq.len() <= 2 {
                self.uf.rewrite(&eq, self.one)
            } else {
                eq
            };
            if !absorb(&mut self.uf, &eq, self.one, &mut self.contradictions) {
                self.long.push(eq);
            }
        }
    }

    /// The equivalent system (long equations plus one row per merged variable) and the number
    /// of contradictions found.
    pub(crate) fn finish(mut self) -> (Vec<Vec<u32>>, usize) {
        // the long equations were rewritten against older states: repeat until nothing shrinks
        loop {
            let long = std::mem::take(&mut self.long);
            let before = long.len();
            let (uf, one) = (&self.uf, self.one);
            let rewritten: Vec<Vec<u32>> = long.par_iter().map(|e| uf.rewrite(e, one)).collect();
            drop(long);
            self.settle(rewritten);
            if self.long.len() == before {
                break;
            }
        }
        let (uf, one) = (&self.uf, self.one);
        let merged: Vec<u32> = (0..one).filter(|&x| uf.parent[x as usize] != x).collect();
        // one row per merged variable: x + root (+ 1) = 0
        let mut out: Vec<Vec<u32>> = merged
            .iter()
            .map(|&x| {
                let (r, p) = uf.find(x);
                let mut e = vec![x, r];
                if p == 1 {
                    e.push(one);
                }
                e.sort_unstable();
                let mut d: Vec<u32> = vec![];
                for v in e {
                    if d.last() == Some(&v) {
                        d.pop();
                    } else {
                        d.push(v);
                    }
                }
                d
            })
            .collect();
        let n_long = self.long.len();
        out.append(&mut self.long);
        eprintln!(
            "propagate: {} equations -> {} merged variables + {n_long} long equations, {} \
             contradictions, {:.1}s",
            self.absorbed,
            merged.len(),
            self.contradictions,
            self.start.elapsed().as_secs_f64()
        );
        (out, self.contradictions)
    }
}

/// An equivalent system for `nvars` variables (constant column `nvars`), with every short
/// equation absorbed. Also returns the number of contradictions found while absorbing.
pub(crate) fn propagate(nvars: u32, eqs: &[Vec<u32>]) -> (Vec<Vec<u32>>, usize) {
    let mut p = Propagator::new(nvars);
    p.absorb(eqs);
    p.finish()
}
