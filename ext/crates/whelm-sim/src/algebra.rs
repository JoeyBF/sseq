//! The Milnor-subalgebra combinatorics that shape Nassau's job DAG, ported from `ext::nassau`.

use std::collections::HashMap;

use whelm::DagTemplate;

/// Bits per Milnor exponent in the packed representation (`PPart::WIDTHS` in `algebra`).
const WIDTHS: [u32; 16] = [11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 0, 0, 0, 0, 0, 0];

/// Degree of `xi_{i+1}`.
fn xi_degree(i: usize) -> i32 {
    (1 << (i + 1)) - 1
}

/// The sequence of subalgebras `optimal_for` walks: `[1]`, `[1,1]`, `[2,1]`, `[2,1,1]`, `[2,2,1]`,
/// `[3,2,1]`, ... (`SubalgebraIterator` in `ext::nassau`).
pub fn subalgebra_sequence() -> impl Iterator<Item = Vec<u8>> {
    let mut current: Vec<u8> = Vec::new();
    std::iter::from_fn(move || {
        if current.is_empty() || current[0] as usize == current.len() {
            current.push(1);
        } else if let Some((_, e)) = current
            .iter_mut()
            .rev()
            .enumerate()
            .find(|(idx, e)| **e as usize == *idx)
        {
            *e += 1;
        }
        Some(current.clone())
    })
}

/// Top degree of the subalgebra with this profile.
pub fn top_degree(profile: &[u8]) -> i32 {
    profile
        .iter()
        .enumerate()
        .map(|(i, &p)| xi_degree(i) * ((1 << p) - 1))
        .sum()
}

/// The profile Nassau uses at `(s, t)` (like everything in this module, a pure function of the
/// bidegree, which is why the whole run's DAG can be built before any computation): the last subalgebra of the sequence whose vanishing line
/// `t >= (2^len - 1)(s + 1) + top_degree` holds, at most `max_len` entries long (the
/// `NASSAU_MAX_SUBALGEBRA` cap is `k + 1` for `A(k)`). Empty means `F_2`.
pub fn optimal_profile(s: i32, t: i32, max_len: usize) -> Vec<u8> {
    subalgebra_sequence()
        .take_while(|p| {
            let coeff = (1i32 << p.len()) - 1;
            t >= coeff * (s + 1) + top_degree(p)
        })
        .take_while(|p| p.len() <= max_len)
        .last()
        .unwrap_or_default()
}

/// Smallest positive degree carrying the zero signature (`zero_sig_floor`): how far back in its
/// own row a bidegree reads.
pub fn zero_sig_floor(profile: &[u8]) -> i32 {
    profile
        .iter()
        .enumerate()
        .map(|(i, &p)| (1i32 << (p as u32).min(WIDTHS[i])) * xi_degree(i))
        .min()
        .unwrap_or(1)
}

/// Number of signatures of the profile (including the zero signature).
pub fn signature_count(profile: &[u8]) -> usize {
    profile.iter().map(|&p| 1usize << p).product()
}

/// Index of a signature in mixed-radix order (`sig_dag::index`).
pub fn sig_index(sig: &[u32], profile: &[u8]) -> usize {
    let mut k = 0;
    let mut mul = 1;
    for (i, &p) in profile.iter().enumerate() {
        let radix = 1usize << p;
        k += (sig.get(i).copied().unwrap_or(0) as usize % radix) * mul;
        mul *= radix;
    }
    k
}

/// The signature with this index, and its degree.
pub fn signature(index: usize, profile: &[u8]) -> (Vec<u32>, i32) {
    let mut k = index;
    let mut sig = Vec::with_capacity(profile.len());
    let mut deg = 0;
    for (i, &p) in profile.iter().enumerate() {
        let radix = 1usize << p;
        let v = (k % radix) as u32;
        k /= radix;
        deg += xi_degree(i) * v as i32;
        sig.push(v);
    }
    (sig, deg)
}

/// The non-zero signatures of degree at most `degree`, by index (`iter_signatures`).
pub fn active_signatures(profile: &[u8], degree: i32) -> Vec<usize> {
    (1..signature_count(profile))
        .filter(|&i| signature(i, profile).1 <= degree)
        .collect()
}

/// Support of `Sq(R) * Sq(s1)`: the exponent tuples with odd coefficient (`support_single`).
fn support_single(r: &[u32], s1: u32) -> Vec<Vec<u32>> {
    let m = r.len();
    let mut counts: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut x = vec![0u32; m];
    /// Enumerate the free column of the admissible matrix, entry by entry.
    fn rec(
        i: usize,
        r: &[u32],
        s1: u32,
        used: u32,
        x: &mut [u32],
        counts: &mut HashMap<Vec<u32>, u32>,
    ) {
        let m = r.len();
        if i == m {
            // Antidiagonals must have pairwise disjoint binary supports (the coefficient is 1).
            let mut anti = vec![0u32; m + 2];
            let mut push = |k: usize, v: u32| -> bool {
                if v == 0 {
                    return true;
                }
                if anti[k] & v != 0 {
                    return false;
                }
                anti[k] |= v;
                true
            };
            let mut ok = push(1, s1 - used);
            for (i2, &ri) in r.iter().enumerate() {
                if !ok {
                    break;
                }
                ok &= push(i2 + 1, ri - 2 * x[i2]);
                ok &= push(i2 + 2, x[i2]);
            }
            if ok {
                *counts.entry(anti[1..].to_vec()).or_insert(0) += 1;
            }
            return;
        }
        let mut v = 0;
        while 2 * v <= r[i] && used + v <= s1 {
            x[i] = v;
            rec(i + 1, r, s1, used + v, x, counts);
            v += 1;
        }
        x[i] = 0;
    }
    rec(0, r, s1, 0, &mut x, &mut counts);
    counts
        .into_iter()
        .filter(|(_, c)| c % 2 == 1)
        .map(|(t, _)| t)
        .collect()
}

/// The signature DAG's direct edges (`sig_dag::direct`): `a -> b` when some `Sq(R)` of signature
/// `a` times a generator `Sq(2^k)` has a term of signature `b`. Node 0 (the zero signature) has no
/// edges; it is the zero step, which precedes the whole walk.
pub fn signature_dag(profile: &[u8]) -> DagTemplate {
    let n = signature_count(profile);
    let radices: Vec<usize> = profile.iter().map(|&p| 1usize << p).collect();
    let cap = 2 * radices.first().copied().unwrap_or(1);
    let ss: Vec<u32> = (0..)
        .map(|k| 1u32 << k)
        .take_while(|&v| (v as usize) < cap * 2)
        .collect();
    let bounds: Vec<u32> = radices.iter().map(|&r| 2 * r as u32).collect();
    let total: u64 = bounds.iter().map(|&b| b as u64).product();
    let mut r = vec![0u32; profile.len()];
    let mut edges = Vec::new();
    for idx in 0..total {
        let mut q = idx;
        for (i, &b) in bounds.iter().enumerate() {
            r[i] = (q % b as u64) as u32;
            q /= b as u64;
        }
        let a = sig_index(&r, profile);
        if a == 0 {
            continue;
        }
        for &s1 in &ss {
            for t in support_single(&r, s1) {
                let b = sig_index(&t, profile);
                if b != a && b != 0 {
                    edges.push((a as u32, b as u32));
                }
            }
        }
    }
    DagTemplate::new(n, edges).expect("the signature DAG is acyclic")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sequence and the vanishing lines match `ext::nassau`.
    #[test]
    fn subalgebras() {
        let seq: Vec<Vec<u8>> = subalgebra_sequence().take(6).collect();
        assert_eq!(
            seq,
            vec![
                vec![1],
                vec![1, 1],
                vec![2, 1],
                vec![2, 1, 1],
                vec![2, 2, 1],
                vec![3, 2, 1]
            ]
        );
        assert_eq!(top_degree(&[4, 3, 2, 1]), 15 + 21 + 21 + 15);
        assert_eq!(top_degree(&[5, 4, 3, 2, 1]), 201);
        assert!(optimal_profile(0, 0, usize::MAX).is_empty());
        // The zero-signature floors measured in `ext::nassau`: A(0)=2 ... A(4)=32.
        for (k, floor) in [(0usize, 2), (1, 4), (2, 8), (3, 16), (4, 32)] {
            let profile: Vec<u8> = (0..=k).map(|i| (k + 1 - i) as u8).collect();
            assert_eq!(zero_sig_floor(&profile), floor, "A({k})");
        }
    }

    /// The A(3) signature order has 4028 covering relations (its transitive reduction).
    #[test]
    fn signature_dag_reduces_to_its_covers() {
        assert_eq!(
            signature_dag(&[4, 3, 2, 1])
                .transitive_reduction()
                .edge_count(),
            4028
        );
    }

    /// Edge counts printed by `ext::nassau` for the transitively closed DAG are 938 at A(2) and
    /// 137,081 at A(3); the direct DAG must close to exactly those.
    #[test]
    fn signature_dag_closes_to_the_verified_counts() {
        for (profile, closed) in [(vec![3u8, 2, 1], 938u64), (vec![4, 3, 2, 1], 137_081)] {
            let t = signature_dag(&profile);
            let n = t.len();
            let mut reach = vec![vec![false; n]; n];
            // Closure in reverse topological order: process nodes with no unvisited successors.
            let mut order: Vec<usize> = (0..n).collect();
            let mut done = vec![false; n];
            let mut stack = Vec::new();
            for &s in &order.clone() {
                if done[s] {
                    continue;
                }
                stack.push((s, 0));
                while let Some((v, i)) = stack.pop() {
                    let succ = t.successors(v);
                    if i < succ.len() {
                        stack.push((v, i + 1));
                        let c = succ[i] as usize;
                        if !done[c] {
                            stack.push((c, 0));
                        }
                    } else if !done[v] {
                        done[v] = true;
                        for &c in succ {
                            let c = c as usize;
                            reach[v][c] = true;
                            let row = reach[c].clone();
                            for (k, &b) in row.iter().enumerate() {
                                reach[v][k] |= b;
                            }
                        }
                    }
                }
            }
            order.clear();
            let edges: u64 = reach
                .iter()
                .map(|r| r.iter().filter(|&&b| b).count() as u64)
                .sum();
            assert_eq!(edges, closed, "profile {profile:?}");
        }
    }
}
