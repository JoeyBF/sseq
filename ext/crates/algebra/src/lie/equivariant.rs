//! `Z/n`-equivariant (character) splitting of the Chevalley–Eilenberg complex of `L(n, n)`.
//!
//! The rotation `g : x_{i, j} -> x_{i, j+1}` is a Lie-algebra automorphism of `L(n, n)` (the bracket
//! structure constants depend on `j` only through Kronecker deltas mod `n`), so `g` generates a
//! `Z/n`-action on the complex `/\^•(L^*)` commuting with the differential `d`. When the ground field
//! contains a primitive `n`-th root of unity `ω` — i.e. `p ≡ 1 (mod n)` — the group algebra
//! `F_p[Z/n]` splits into `n` characters and the complex splits as a direct sum
//!
//! ```text
//!   /\^•(L^*) = ⊕_{c = 0}^{n-1} (/\^•)_c,     d preserves each (/\^•)_c,
//! ```
//!
//! so `rank(d_k) = Σ_c rank(d_k restricted to character c)`. Each character block has ~`1/n` the
//! dimension of the full block, which is what makes the large `n = 5` blocks (up to `565k` square)
//! reachable: each shrinks to `~113k`. All arithmetic stays in `F_p` (no field extension) because
//! `ω ∈ F_p`.
//!
//! # Reduced matrix
//!
//! Group the degree-`k` monomials into `g`-orbits. A size-`ℓ` orbit with representative `R` (the
//! minimal mask) has elements `R_t = g^t R` with signs `σ^R_t` (`g^t e_R = σ^R_t e_{R_t}`), and it
//! carries the character `c` iff `ω^{cℓ} = s_R`, where `s_R = σ^R_ℓ` is the sign of `g^ℓ` on `R`.
//! When it does, its character-`c` eigenvector is `v_c(R) = Σ_{t<ℓ} ω^{-ct} σ^R_t e_{R_t}`. Writing
//! `d(e_{R_t}) = Σ γ(R_t, ·) e_·` (from [`differential_terms`]), the reduced differential on the
//! orbit-representative basis is (dropping the harmless global `1/ℓ` from the inverse DFT, which does
//! not affect rank)
//!
//! ```text
//!   D_c[O, U] = Σ_{u < ℓ_U} σ^U_u ω^{cu} · ( Σ_{t < ℓ_O} ω^{-ct} σ^O_t γ(O_t, U_u) ),
//! ```
//!
//! for domain/codomain orbits `O, U` both carrying `c`. `rank(D_c)` is taken with the sparse solver.
//!
//! The whole construction is validated by reproducing the known totals `12, 152, 3440` at
//! `n = 2, 3, 4` (see the tests) before it is trusted on `n = 5`.

use std::collections::HashMap;

use fp::prime::Prime;
use maybe_rayon::prelude::*;

use crate::lie::{
    cohomology::{differential_terms, sparse_rank},
    morava_lie::MoravaLie,
};

/// A single `g`-orbit of monomials (subsets), all of one cohomological degree.
struct Orbit {
    /// Length `ℓ` of the orbit (divides `n`).
    len: u32,
    /// Sign `s = σ^R_ℓ` of `g^ℓ` acting on the representative (determines which characters occur).
    s: i32,
    /// The elements `R_0 = R, R_1 = gR, ...` with their signs `σ^R_t` (`g^t e_R = σ^R_t e_{R_t}`).
    elements: Vec<(u32, i32)>,
}

/// The orbit decomposition of a set of monomials: the orbits, plus a lookup `mask -> (orbit index,
/// position u within the orbit)`.
struct OrbitData {
    orbits: Vec<Orbit>,
    lookup: HashMap<u32, (usize, u32)>,
}

/// Applies the rotation `g` to a monomial `mask`, returning `(g·mask, sign)` where
/// `g e_mask = sign · e_{g·mask}` (`sign` is the parity of sorting the permuted generator indices).
fn g_apply(mask: u32, perm: &[u32]) -> (u32, i32) {
    let mut seq: Vec<u32> = Vec::with_capacity(mask.count_ones() as usize);
    let mut bits = mask;
    let mut new_mask = 0u32;
    while bits != 0 {
        let a = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        seq.push(perm[a]);
        new_mask |= 1u32 << perm[a];
    }
    let mut inversions = 0usize;
    for i in 0..seq.len() {
        for j in (i + 1)..seq.len() {
            if seq[i] > seq[j] {
                inversions += 1;
            }
        }
    }
    let sign = if inversions.is_multiple_of(2) { 1 } else { -1 };
    (new_mask, sign)
}

/// Decomposes `masks` (all of one degree) into `g`-orbits, walking each orbit from its minimal mask.
fn orbit_data(masks: &[u32], perm: &[u32]) -> OrbitData {
    let mut lookup: HashMap<u32, (usize, u32)> = HashMap::new();
    let mut orbits: Vec<Orbit> = Vec::new();
    for &m in masks {
        if lookup.contains_key(&m) {
            continue;
        }
        // Find the orbit's element set, then take the minimal mask as representative.
        let mut cycle = vec![m];
        let (mut cur, _) = g_apply(m, perm);
        while cur != m {
            cycle.push(cur);
            cur = g_apply(cur, perm).0;
        }
        let rep = *cycle.iter().min().unwrap();
        // Walk from the representative, accumulating signs.
        let orbit_idx = orbits.len();
        let mut elements: Vec<(u32, i32)> = Vec::new();
        let mut sign = 1i32;
        let mut node = rep;
        let mut u = 0u32;
        loop {
            elements.push((node, sign));
            lookup.insert(node, (orbit_idx, u));
            let (next, s) = g_apply(node, perm);
            sign *= s;
            node = next;
            u += 1;
            if node == rep {
                break;
            }
        }
        let len = elements.len() as u32;
        let s = sign; // = σ^R_ℓ, the sign of g^ℓ on the representative
        orbits.push(Orbit { len, s, elements });
    }
    OrbitData { orbits, lookup }
}

/// The generator-index permutation induced by `g : x_{i,j} -> x_{i,j+1}`.
fn rotation_perm(lie: &MoravaLie) -> Vec<u32> {
    let n = lie.height();
    let dim = lie.dim();
    let mut perm = vec![0u32; dim];
    for (a, slot) in perm.iter_mut().enumerate() {
        let (i, j) = lie.generator(a);
        // index(i, j) = (i - 1) * n + j; rotate j -> (j + 1) mod n.
        let jn = (j + 1) % n;
        *slot = (i - 1) * n + jn;
    }
    perm
}

/// A primitive `n`-th root of unity in `F_p`. Panics if `p !≡ 1 (mod n)` (no such root exists).
fn primitive_root(n: u32, p: u32) -> u32 {
    assert!(
        (p - 1).is_multiple_of(n),
        "equivariant splitting needs p ≡ 1 (mod n): p = {p}, n = {n}"
    );
    for x in 2..p {
        // order(x) == n ?
        let mut v = x % p;
        let mut ord = 1u32;
        while v != 1 {
            v = v * x % p;
            ord += 1;
        }
        if ord == n {
            return x;
        }
    }
    panic!("no primitive {n}-th root of unity mod {p}");
}

/// `dim_{F_p} H^*(L(n, n))` via the `Z/n`-character splitting. Requires `p ≡ 1 (mod n)`.
///
/// Ranks each `(weight, degree, character)` sub-block in parallel; the character blocks are `~1/n`
/// the size of the full blocks, so this reaches heights the plain streaming path cannot.
pub fn equivariant_total_dim(lie: &MoravaLie) -> usize {
    let n = lie.height();
    let p = lie.prime().as_u32();
    let n_dim = lie.dim();
    let cobracket = lie.cobracket();
    let perm = rotation_perm(lie);
    let weights: Vec<u32> = (0..n_dim).map(|a| lie.weight(a)).collect();
    let max_weight: u32 = weights.iter().sum();

    let omega = primitive_root(n, p);
    // pow_omega[e] = ω^e for e in 0..n (ω^n = 1).
    let mut pow_omega = vec![1u32; n as usize];
    for e in 1..n as usize {
        pow_omega[e] = pow_omega[e - 1] * omega % p;
    }
    let om = |e: i64| -> u32 {
        let e = e.rem_euclid(n as i64) as usize;
        pow_omega[e]
    };
    let s_to_fp = |s: i32| -> u32 { s.rem_euclid(p as i32) as u32 };

    // Collect (weight, degree) blocks as tasks; rank each block's character pieces.
    let mut tasks: Vec<(Vec<u32>, Vec<u32>)> = Vec::new(); // (domain masks, codomain masks)
    for w in 0..=max_weight {
        let mut by_k: Vec<Vec<u32>> = vec![Vec::new(); n_dim + 1];
        crate::lie::cohomology::enumerate_weight(&weights, w, &mut |mask| {
            by_k[mask.count_ones() as usize].push(mask);
        });
        for k in 0..n_dim {
            if by_k[k].is_empty() || by_k[k + 1].is_empty() {
                continue;
            }
            tasks.push((by_k[k].clone(), by_k[k + 1].clone()));
        }
    }

    let cobracket_ref = &cobracket;
    let perm_ref = &perm;
    let total_rank: usize = tasks
        .into_maybe_par_iter()
        .map(|(domain, codomain)| {
            block_rank_equivariant(
                &domain,
                &codomain,
                cobracket_ref,
                perm_ref,
                n,
                p,
                &om,
                &s_to_fp,
            )
        })
        .sum();

    let total: usize = 1usize << n_dim;
    total - 2 * total_rank
}

/// Ranks one `(weight, degree)` differential block by summing the ranks of its `n` character pieces.
#[allow(clippy::too_many_arguments)]
fn block_rank_equivariant(
    domain: &[u32],
    codomain: &[u32],
    cobracket: &[Vec<(usize, usize, i32)>],
    perm: &[u32],
    n: u32,
    p: u32,
    om: &impl Fn(i64) -> u32,
    s_to_fp: &impl Fn(i32) -> u32,
) -> usize {
    let dom = orbit_data(domain, perm);
    let cod = orbit_data(codomain, perm);

    let mut total = 0usize;
    for c in 0..n as i64 {
        // Rows of the reduced matrix D_c: one per domain orbit carrying character c.
        let mut rows: Vec<Vec<(u32, u32)>> = Vec::new();
        for o in &dom.orbits {
            // Orbit carries c iff ω^{c·ℓ} = s.
            if om(c * o.len as i64) != s_to_fp(o.s) {
                continue;
            }
            // Accumulate D_c[O, U] over codomain orbits U (keyed by orbit index).
            let mut row: HashMap<u32, u32> = HashMap::new();
            for (t, &(mask_t, sigma_o_t)) in o.elements.iter().enumerate() {
                // ω^{-c t} σ^O_t
                let phase_t = om(-c * t as i64) * s_to_fp(sigma_o_t) % p;
                if phase_t == 0 {
                    continue;
                }
                for (target, gamma) in differential_terms(mask_t, cobracket, p) {
                    let &(u_orbit, u_pos) = cod.lookup.get(&target).expect("target in codomain");
                    let u = &cod.orbits[u_orbit];
                    // U must also carry c.
                    if om(c * u.len as i64) != s_to_fp(u.s) {
                        continue;
                    }
                    let sigma_u_u = u.elements[u_pos as usize].1;
                    // σ^U_u ω^{cu} · (phase_t · γ)
                    let coeff = s_to_fp(sigma_u_u)
                        * om(c * u_pos as i64) % p
                        * phase_t % p
                        * (gamma % p) % p;
                    if coeff == 0 {
                        continue;
                    }
                    let e = row.entry(u_orbit as u32).or_insert(0);
                    *e = (*e + coeff) % p;
                }
            }
            let row: Vec<(u32, u32)> = row.into_iter().filter(|&(_, v)| v != 0).collect();
            rows.push(row);
        }
        total += sparse_rank(rows, p);
    }
    total
}

#[cfg(test)]
mod tests {
    use fp::prime::ValidPrime;

    use super::*;

    #[test]
    fn matches_streaming_totals_n2_n3_n4() {
        // The equivariant path must reproduce the known totals through the SAME general orbit code
        // that will be used at n = 5. Primes chosen with p ≡ 1 (mod n) and p > n + 1.
        for (n, p, expected) in [(2u32, 5u32, 12usize), (3, 7, 152), (4, 13, 3440)] {
            let lie = MoravaLie::new(ValidPrime::new(p), n);
            assert_eq!(
                equivariant_total_dim(&lie),
                expected,
                "equivariant total wrong at n = {n}, p = {p}"
            );
        }
    }
}
