//! Size check: the sphere's Adams spectral sequence as an `Sseq<3, SyntheticAdams>`.
//!
//! Usage: `sseq_size <rank.csv> <d2 solution csv> <max stem> <max filtration> <pages>`. Loads every
//! E2 dimension, every determined d2, and a zero d_r on every class for r < `pages` (a stand-in for
//! the full depth of page data), then prints time and peak memory.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};

use fp::{prime::ValidPrime, vector::FpVector};
use sseq::{
    Adams, Sseq, SseqProfile, SyntheticAdams,
    coordinates::{degree::MultiDegree, element::MultiDegreeElement},
};

/// Peak resident memory of this process, in MB.
fn peak_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let kb: f64 = status
        .lines()
        .find(|l| l.starts_with("VmHWM"))
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap()
        .parse()
        .unwrap();
    kb / 1024.0
}

fn main() {
    // SSEQ_N=2: the same data as a bigraded `Sseq<2, Adams>`, for comparison
    if std::env::var("SSEQ_N").is_ok_and(|v| v == "2") {
        run::<2, Adams>(|n, s| MultiDegree::new([n, s]));
    } else {
        run::<3, SyntheticAdams>(|n, s| MultiDegree::new([n, s, 0]));
    }
}

/// Builds the spectral sequence with `deg` placing stem and filtration in `N` gradings.
fn run<const N: usize, P: SseqProfile<N>>(deg: impl Fn(i32, i32) -> MultiDegree<N>) {
    let a: Vec<String> = std::env::args().collect();
    let (n_max, f_max, pages): (i32, i32, i32) = (
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
        a[5].parse().unwrap(),
    );
    let t = Instant::now();
    let p = ValidPrime::new(2);
    let mut dims = std::collections::HashMap::new();
    for line in BufReader::new(File::open(&a[1]).unwrap()).lines().skip(1) {
        let line = line.unwrap();
        let w: Vec<i32> = line.split(',').map(|x| x.trim().parse().unwrap()).collect();
        dims.insert((w[0], w[1]), w[2] as usize);
    }
    let mut sseq = Sseq::<N, P>::new(p);
    let mut classes = 0;
    for n in -1..=n_max + 1 {
        for s in 0..=f_max + pages + 1 {
            let d = if n >= 0 && n <= n_max && s <= f_max {
                dims.get(&(n, s)).copied().unwrap_or(0)
            } else {
                0
            };
            classes += d;
            sseq.set_dimension(deg(n, s), d);
        }
    }
    let t_dims = t.elapsed().as_secs_f64();
    let mut d2s = 0;
    for line in BufReader::new(File::open(&a[2]).unwrap()).lines().skip(1) {
        let line = line.unwrap();
        let mut it = line.splitn(5, ',');
        let n: i32 = it.next().unwrap().parse().unwrap();
        let s: i32 = it.next().unwrap().parse().unwrap();
        let i: usize = it.next().unwrap().parse().unwrap();
        if it.next().unwrap() != "determined" || n > n_max || s > f_max {
            continue;
        }
        let tgt: Vec<u32> = it
            .next()
            .unwrap()
            .trim_matches(|c| c == '"' || c == '[' || c == ']')
            .split(',')
            .filter(|x| !x.trim().is_empty())
            .map(|x| x.trim().parse().unwrap())
            .collect();
        let b = deg(n, s);
        let tb = P::profile(2, b);
        if sseq.dimension(tb) != tgt.len() {
            continue;
        }
        let mut v = FpVector::new(p, sseq.dimension(b));
        v.set_entry(i, 1);
        sseq.add_differential(
            2,
            &MultiDegreeElement::new(b, v),
            FpVector::from_slice(p, &tgt).as_slice(),
        );
        d2s += 1;
    }
    let t_d2 = t.elapsed().as_secs_f64();
    let mut zeros = 0;
    for r in 3..pages {
        for n in 0..=n_max {
            for s in 0..=f_max {
                let b = deg(n, s);
                let tb = P::profile(r, b);
                for i in 0..sseq.dimension(b) {
                    let mut v = FpVector::new(p, sseq.dimension(b));
                    v.set_entry(i, 1);
                    let z = FpVector::new(p, sseq.dimension(tb));
                    sseq.add_differential(r, &MultiDegreeElement::new(b, v), z.as_slice());
                    zeros += 1;
                }
            }
        }
    }
    let t_add = t.elapsed().as_secs_f64();
    sseq.update();
    println!(
        "{classes} classes, {d2s} d2, {zeros} stand-in zero d_r (r < {pages}); dims {t_dims:.1}s, \
         d2 {:.1}s, zeros {:.1}s, update {:.1}s; peak {:.0} MB",
        t_d2 - t_dims,
        t_add - t_d2,
        t.elapsed().as_secs_f64() - t_add,
        peak_mb()
    );
}
