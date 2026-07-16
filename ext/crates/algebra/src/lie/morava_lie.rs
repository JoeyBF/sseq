//! The height-`n` Morava Lie algebra `L(n, n)` at large primes.
//!
//! This is the `n^2`-dimensional restricted Lie algebra from Ravenel, *Complex Cobordism and Stable
//! Homotopy Groups of Spheres* (green book) Thm 6.3.3, restricted to the finite quotient `L(n, n)`
//! (Salch, arXiv:2312.17185, following Ravenel Thm 1.4). For `p > n + 1` this quotient carries
//! *trivial* restriction (the `p`-th power map lands in `x_{i + n, ·}`, which is zero in the
//! quotient), so `H^*(L(n, n); F_p)` is *ordinary* Lie-algebra cohomology, and Salch's collapse
//! theorem identifies it with `H^*(S(n))`, the continuous cohomology of the height-`n` Morava
//! stabilizer group. See `ext/docs/chromatic-computations.md`.
//!
//! # The Lie algebra
//!
//! `L(n, n)` has `F_p`-basis `{x_{i, j} : 1 <= i <= n, j in Z/n}` (so dimension `n^2`), with bracket
//! (green book Thm 6.3.3, with `m = n` since `p > n + 1`, and `delta^s_t = 1` iff `s == t (mod n)`):
//!
//! ```text
//!                       / delta^l_{i+j} x_{i+k, j}  -  delta^j_{k+l} x_{i+k, l}   if i + k <= n
//! [x_{i,j}, x_{k,l}] = <
//!                       \ 0                                                       otherwise
//! ```
//!
//! The `i`-index (`1..=n`) is a "height"/May-weight grading: the bracket sends weight `i` and weight
//! `k` to weight `i + k`, so the total `i`-weight is preserved by the bracket and hence by the
//! Chevalley–Eilenberg differential. This grading is what makes the cohomology computation split
//! into small blocks (see [`crate::lie::cohomology`]).
//!
//! ## The subscript-ordering caveat (green book vs Salch)
//!
//! Two transcriptions of Thm 6.3.3 disagree on which second subscript pairs with which Kronecker
//! delta: one reads `delta^l_{i+j} x_{i+k, j} - delta^j_{k+l} x_{i+k, l}` and the other swaps the
//! `x`-subscripts `j <-> l`. The difference is invisible at `n = 1` (all deltas are 1 and both terms
//! are `x_{i+k, 0}`, so the bracket vanishes), and the two happen to coincide at `n = 2`, but they
//! *diverge* at `n = 3`: the first reading reproduces the known `dim H^*(L(3,3)) = 152`, while the
//! swapped reading gives `128`. **So the first reading ([`BracketConvention::GreenBook`], the
//! constructor default) is the correct one** — the finite-CE cohomology tool itself resolves the
//! ambiguity the handoff flagged. [`MoravaLie::with_convention`] still exposes the other reading so
//! the disambiguation stays checkable (see the `validation_ladder_n3` test).

use fp::prime::{Prime, ValidPrime};

/// Which transcription of the green-book Thm 6.3.3 bracket to use for the `x`-subscripts. See the
/// module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BracketConvention {
    /// `[x_{i,j}, x_{k,l}] = delta^l_{i+j} x_{i+k, j} - delta^j_{k+l} x_{i+k, l}` (green-book OCR).
    GreenBook,
    /// `[x_{i,j}, x_{k,l}] = delta^l_{i+j} x_{i+k, l} - delta^j_{k+l} x_{i+k, j}` (Salch eq. (8)).
    Salch,
}

/// The finite Morava Lie algebra `L(n, n)` over `F_p`, `p > n + 1`.
///
/// Generators `x_{i, j}` (`1 <= i <= n`, `0 <= j < n`) are indexed in the fixed order
/// `index(i, j) = (i - 1) * n + j`, so index `0..n^2`. Coefficients are kept as signed `i32`
/// (always `+1`, `-1`, or `0`) and reduced mod `p` only at the point of use.
#[derive(Clone, Debug)]
pub struct MoravaLie {
    prime: ValidPrime,
    height: u32,
    /// `gens[a] = (i, j)` for the generator at index `a`.
    gens: Vec<(u32, u32)>,
    convention: BracketConvention,
}

impl MoravaLie {
    /// Builds `L(n, n)` over `F_p` with the green-book bracket convention.
    ///
    /// Panics if `height == 0`. Emits a warning to the log (but does not panic) if `p <= n + 1`,
    /// since outside the large-prime range the restriction is nontrivial and this *ordinary* Lie
    /// cohomology is no longer the whole story (see the module docs and §8 of the handoff).
    pub fn new(prime: ValidPrime, height: u32) -> Self {
        Self::with_convention(prime, height, BracketConvention::GreenBook)
    }

    /// Builds `L(n, n)` with an explicit [`BracketConvention`].
    pub fn with_convention(
        prime: ValidPrime,
        height: u32,
        convention: BracketConvention,
    ) -> Self {
        assert!(height >= 1, "L(n, n) requires height n >= 1");
        if prime.as_u32() <= height + 1 {
            eprintln!(
                "warning: MoravaLie: p = {} is not > n + 1 = {}; H^*(L(n,n)) is the *large-prime* \
                 answer only for p > n + 1 (restriction is nontrivial otherwise).",
                prime,
                height + 1
            );
        }
        let n = height;
        let mut gens = Vec::with_capacity((n * n) as usize);
        for i in 1..=n {
            for j in 0..n {
                gens.push((i, j));
            }
        }
        Self {
            prime,
            height,
            gens,
            convention,
        }
    }

    /// The prime `p`.
    pub fn prime(&self) -> ValidPrime {
        self.prime
    }

    /// The height `n`.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The dimension of the Lie algebra, `n^2`.
    pub fn dim(&self) -> usize {
        self.gens.len()
    }

    /// The `(i, j)` labelling the generator at index `a`.
    pub fn generator(&self, a: usize) -> (u32, u32) {
        self.gens[a]
    }

    /// The `i`-weight (height grading) of generator `a`; the bracket adds weights.
    pub fn weight(&self, a: usize) -> u32 {
        self.gens[a].0
    }

    /// The modulus `p^n - 1` of the internal grading (see [`Self::internal_degree`]).
    pub fn internal_modulus(&self) -> u64 {
        (self.prime.as_u32() as u64).pow(self.height) - 1
    }

    /// The internal (topological) degree of generator `a`, valued in `Z/(p^n - 1)`:
    ///
    /// ```text
    /// t_{i,j} = p^j (p^i - 1)  ==  p^{i+j} - p^j   (mod p^n - 1).
    /// ```
    ///
    /// This is the *periodic* ("chromatic") refinement of Ravenel's topological degree
    /// `2(p^i - 1)p^j`: because `L(n, n)` identifies `j` mod `n` and `p^n ≡ 1 (mod p^n - 1)`, the
    /// honest topological degree is only well-defined mod `p^n - 1`, and it is exactly for *this*
    /// grading — not the naïve `Z`-valued one (the §6 trap) — that the bracket is homogeneous:
    /// `t_{i,j} + t_{k,l} = t_{i+k, ·}` for both bracket terms. Hence the Chevalley–Eilenberg
    /// differential preserves it and `H^*(L(n,n))` is bigraded by `(s, t)`, `t ∈ Z/(p^n - 1)`. (We
    /// drop Ravenel's uniform factor of `2`, which only rescales `t`.)
    pub fn internal_degree(&self, a: usize) -> u64 {
        let (i, j) = self.gens[a];
        let m = self.internal_modulus();
        let p = self.prime.as_u32() as u64;
        let pj = p.pow(j) % m;
        let pi_minus_1 = (p.pow(i) % m + m - 1) % m;
        pj * pi_minus_1 % m
    }

    /// The index of the generator `x_{i, j}` (`1 <= i <= n`, `j` taken mod `n`), or `None` if
    /// `i` is out of range `1..=n`.
    fn index_of(&self, i: u32, j: u32) -> Option<usize> {
        let n = self.height;
        if i < 1 || i > n {
            return None;
        }
        let j = j % n;
        Some(((i - 1) * n + j) as usize)
    }

    /// The bracket `[x_a, x_b]` as a list of `(generator index, signed coefficient)` terms. The
    /// coefficients are `+1` or `-1`; terms that cancel or land outside the algebra are omitted.
    pub fn bracket(&self, a: usize, b: usize) -> Vec<(usize, i32)> {
        let n = self.height;
        let (i, j) = self.gens[a];
        let (k, l) = self.gens[b];
        // The bracket lands in weight i + k, which must remain <= m = n.
        if i + k > n {
            return Vec::new();
        }
        // delta^s_t = 1 iff s == t (mod n).
        let delta = |s: u32, t: u32| (s % n) == (t % n);
        // (plus_target_j, minus_target_j) — which second subscript the +/- term carries.
        let (plus_j, minus_j) = match self.convention {
            BracketConvention::GreenBook => (j, l),
            BracketConvention::Salch => (l, j),
        };
        let mut terms: Vec<(usize, i32)> = Vec::new();
        let push = |terms: &mut Vec<(usize, i32)>, tj: u32, coeff: i32| {
            if let Some(idx) = self.index_of(i + k, tj) {
                if let Some(slot) = terms.iter_mut().find(|(t, _)| *t == idx) {
                    slot.1 += coeff;
                } else {
                    terms.push((idx, coeff));
                }
            }
        };
        // + delta^l_{i+j} x_{i+k, plus_j}
        if delta(l, i + j) {
            push(&mut terms, plus_j, 1);
        }
        // - delta^j_{k+l} x_{i+k, minus_j}
        if delta(j, k + l) {
            push(&mut terms, minus_j, -1);
        }
        terms.retain(|(_, c)| *c != 0);
        terms
    }

    /// The "cobracket" table used by the Chevalley–Eilenberg differential: `cobracket()[a]` lists
    /// every `(b, c, coeff)` with `b < c` such that `x_a` appears with nonzero coefficient `coeff`
    /// in `[x_b, x_c]`. Equivalently, `d(x_a^*) = - sum coeff * x_b^* /\ x_c^*` on generators.
    pub fn cobracket(&self) -> Vec<Vec<(usize, usize, i32)>> {
        let dim = self.dim();
        let mut table = vec![Vec::new(); dim];
        for b in 0..dim {
            for c in (b + 1)..dim {
                for (a, coeff) in self.bracket(b, c) {
                    table[a].push((b, c, coeff));
                }
            }
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use fp::prime::ValidPrime;

    use super::*;

    fn p(v: u32) -> ValidPrime {
        ValidPrime::new(v)
    }

    /// Reduce a signed coefficient mod p to `0..p`.
    fn redp(c: i32, prime: u32) -> u32 {
        c.rem_euclid(prime as i32) as u32
    }

    /// Antisymmetry `[x_a, x_b] = -[x_b, x_a]` and `[x_a, x_a] = 0`, checked over `F_p`.
    fn check_antisymmetry(lie: &MoravaLie) {
        let dim = lie.dim();
        let prime = lie.prime().as_u32();
        for a in 0..dim {
            assert!(lie.bracket(a, a).is_empty(), "diagonal bracket nonzero");
            for b in 0..dim {
                let ab = to_vec(&lie.bracket(a, b), dim, prime);
                let ba = to_vec(&lie.bracket(b, a), dim, prime);
                for t in 0..dim {
                    assert_eq!(
                        redp(ab[t] as i32 + ba[t] as i32, prime),
                        0,
                        "antisymmetry failed at ({a},{b}) target {t}"
                    );
                }
            }
        }
    }

    fn to_vec(terms: &[(usize, i32)], dim: usize, prime: u32) -> Vec<u32> {
        let mut v = vec![0u32; dim];
        for &(t, c) in terms {
            v[t] = redp(v[t] as i32 + c, prime);
        }
        v
    }

    /// Jacobi identity `[[a,b],c] + [[b,c],a] + [[c,a],b] = 0` over `F_p`.
    fn check_jacobi(lie: &MoravaLie) {
        let dim = lie.dim();
        let prime = lie.prime().as_u32();
        // bracket of a vector with a generator.
        let bracket_vec_gen = |v: &[u32], c: usize| -> Vec<u32> {
            let mut out = vec![0i64; dim];
            for (a, &coeff) in v.iter().enumerate() {
                if coeff == 0 {
                    continue;
                }
                for (t, bc) in lie.bracket(a, c) {
                    out[t] += coeff as i64 * bc as i64;
                }
            }
            out.into_iter()
                .map(|x| x.rem_euclid(prime as i64) as u32)
                .collect()
        };
        for a in 0..dim {
            for b in 0..dim {
                for c in 0..dim {
                    let ab = to_vec(&lie.bracket(a, b), dim, prime);
                    let bc = to_vec(&lie.bracket(b, c), dim, prime);
                    let ca = to_vec(&lie.bracket(c, a), dim, prime);
                    let t1 = bracket_vec_gen(&ab, c);
                    let t2 = bracket_vec_gen(&bc, a);
                    let t3 = bracket_vec_gen(&ca, b);
                    for t in 0..dim {
                        let s = redp(t1[t] as i32 + t2[t] as i32 + t3[t] as i32, prime);
                        assert_eq!(s, 0, "Jacobi failed at ({a},{b},{c}) target {t}");
                    }
                }
            }
        }
    }

    #[test]
    fn height1_is_abelian() {
        let lie = MoravaLie::new(p(3), 1);
        assert_eq!(lie.dim(), 1);
        assert!(lie.bracket(0, 0).is_empty());
    }

    #[test]
    fn lie_axioms_n2() {
        for conv in [BracketConvention::GreenBook, BracketConvention::Salch] {
            let lie = MoravaLie::with_convention(p(5), 2, conv);
            assert_eq!(lie.dim(), 4);
            check_antisymmetry(&lie);
            check_jacobi(&lie);
        }
    }

    #[test]
    fn lie_axioms_n3() {
        for conv in [BracketConvention::GreenBook, BracketConvention::Salch] {
            let lie = MoravaLie::with_convention(p(7), 3, conv);
            assert_eq!(lie.dim(), 9);
            check_antisymmetry(&lie);
            check_jacobi(&lie);
        }
    }

    #[test]
    fn lie_axioms_n4() {
        let lie = MoravaLie::new(p(7), 4);
        assert_eq!(lie.dim(), 16);
        check_antisymmetry(&lie);
        check_jacobi(&lie);
    }
}
