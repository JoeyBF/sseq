//! Dense linear algebra over the PID $\mathbb{F}_2[\tau]$, with polynomials packed
//! as [`Poly`] — an arbitrary-length bitset of exponents (bit $i$ = coefficient of
//! $\tau^i$), stored as `u64` limbs.
//!
//! Used to read the motivic Adams $E_2$ as an $\mathbb{F}_2[\tau]$-module: the
//! $\tau$-torsion orders are the non-unit invariant factors of $\delta$
//! ([`invariant_factors`], Smith normal form), and Massey-product coset membership
//! is decided by reducing modulo a submodule ([`reduce_mod`]).
//!
//! The representation was `u128` (exponents capped at 127); it is now a fixed
//! [`LIMBS`]-limb bitset ([`Poly`]) — stack-allocated and `Copy`, with capacity
//! `64 * LIMBS` exponents. That comfortably exceeds any $\tau$-tower length or
//! hidden-extension power in the resolved range; an exponent past capacity panics
//! (a bounds check) rather than silently wrapping as the `u128` shift did.

use std::ops::BitXorAssign;

/// Number of `u64` limbs backing a [`Poly`]: `64 * LIMBS = 512` bits of exponent
/// capacity. Real motivic $\tau$-powers/torsion orders stay far below this even at
/// stems past 100; larger would only need a bigger `LIMBS`.
const LIMBS: usize = 8;

/// A polynomial over $\mathbb{F}_2[\tau]$: a bitset of exponents, bit $i$ of limb
/// $w$ being the coefficient of $\tau^{64w+i}$. Fixed-size and `Copy`, so it lives
/// on the stack with no allocation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Poly([u64; LIMBS]);

impl Poly {
    /// The zero polynomial.
    pub fn zero() -> Self {
        Poly([0; LIMBS])
    }

    /// The monomial $\tau^{\mathrm{exp}}$.
    pub fn tau_pow(exp: u32) -> Self {
        let mut p = Poly::zero();
        p.toggle(exp);
        p
    }

    /// Whether this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// Add $\tau^{\mathrm{exp}}$ (toggle the coefficient of $\tau^{\mathrm{exp}}$).
    /// Panics (bounds check) if `exp >= 64 * LIMBS`.
    pub fn toggle(&mut self, exp: u32) {
        self.0[(exp / 64) as usize] ^= 1u64 << (exp % 64);
    }

    /// Multiply by $\tau^k$ (shift the bitset left by `k`). Panics (bounds check)
    /// if any set bit would land at exponent `>= 64 * LIMBS`.
    fn shl(&self, k: u32) -> Poly {
        if self.is_zero() {
            return Poly::zero();
        }
        let limb_shift = (k / 64) as usize;
        let bit_shift = k % 64;
        let mut out = [0u64; LIMBS];
        for i in (0..LIMBS).rev() {
            let w = self.0[i];
            if w == 0 {
                continue;
            }
            out[i + limb_shift] ^= w << bit_shift;
            if bit_shift > 0 {
                let carry = w >> (64 - bit_shift);
                if carry != 0 {
                    out[i + limb_shift + 1] ^= carry;
                }
            }
        }
        Poly(out)
    }

    /// The set exponents, low to high.
    fn exponents(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter().enumerate().flat_map(|(i, &w)| {
            (0..64u32).filter_map(move |b| (w & (1u64 << b) != 0).then_some(i as u32 * 64 + b))
        })
    }
}

impl BitXorAssign<&Poly> for Poly {
    fn bitxor_assign(&mut self, rhs: &Poly) {
        for i in 0..LIMBS {
            self.0[i] ^= rhs.0[i];
        }
    }
}

impl BitXorAssign<Poly> for Poly {
    fn bitxor_assign(&mut self, rhs: Poly) {
        *self ^= &rhs;
    }
}

/// Degree of `a` (`-1` for the zero polynomial).
pub fn deg(a: &Poly) -> i32 {
    for i in (0..LIMBS).rev() {
        if a.0[i] != 0 {
            return (i as i32) * 64 + (63 - a.0[i].leading_zeros() as i32);
        }
    }
    -1
}

/// Polynomial product over $\mathbb{F}_2$ (carryless multiply).
fn mul(a: &Poly, b: &Poly) -> Poly {
    let mut r = Poly::zero();
    for i in a.exponents() {
        r ^= b.shl(i);
    }
    r
}

/// Quotient $\lfloor a / b \rfloor$ over $\mathbb{F}_2[\tau]$ (the remainder is
/// dropped). `b` must be nonzero.
fn div(a: &Poly, b: &Poly) -> Poly {
    let db = deg(b);
    let (mut q, mut rem) = (Poly::zero(), a.clone());
    while deg(&rem) >= db {
        let sh = (deg(&rem) - db) as u32;
        q.toggle(sh);
        rem ^= b.shl(sh);
    }
    q
}

/// Reduce the vector `target` modulo the $\mathbb{F}_2[\tau]$-submodule spanned by
/// `rows`. Row-reduces `rows` to an echelon form (a minimal-degree pivot per
/// column, cleared below via Euclidean division), then reduces `target` against
/// the pivots. The returned remainder is a canonical coset representative — zero
/// iff `target` lies in the submodule.
#[allow(clippy::needless_range_loop)]
pub fn reduce_mod(mut rows: Vec<Vec<Poly>>, mut target: Vec<Poly>) -> Vec<Poly> {
    let ncols = target.len();
    let nrows = rows.len();
    let mut r = 0; // next pivot row
    for col in 0..ncols {
        if r >= nrows {
            break;
        }
        // Euclidean-reduce column `col` among rows[r..] until one row carries the
        // gcd there and the rest are zero in that column.
        loop {
            let mut piv = None;
            for i in r..nrows {
                if !rows[i][col].is_zero()
                    && piv.is_none_or(|p: usize| deg(&rows[i][col]) < deg(&rows[p][col]))
                {
                    piv = Some(i);
                }
            }
            let Some(pi) = piv else { break };
            rows.swap(r, pi);
            let p = rows[r][col].clone();
            let mut changed = false;
            for i in 0..nrows {
                if i != r && !rows[i][col].is_zero() {
                    let q = div(&rows[i][col], &p);
                    if !q.is_zero() {
                        for j in 0..ncols {
                            let t = mul(&q, &rows[r][j]);
                            rows[i][j] ^= t;
                        }
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if !rows[r][col].is_zero() {
            // Reduce target's entry in this pivot column modulo the pivot.
            let q = div(&target[col], &rows[r][col]);
            if !q.is_zero() {
                for j in 0..ncols {
                    let t = mul(&q, &rows[r][j]);
                    target[j] ^= t;
                }
            }
            r += 1;
        }
    }
    target
}

/// The non-unit invariant factors (degree $\ge 1$) of `m` over
/// $\mathbb{F}_2[\tau]$, by Smith normal form. `m` is consumed.
///
/// Standard Euclidean SNF: pivot on the minimum-degree nonzero entry, clear its
/// row and column by division, and pull any lower-degree residual back into the
/// pivot until the pivot divides the remaining block; then recurse on the
/// complementary submatrix.
// The row/column clears index two rows (or columns) at once — `m[i][j]` against
// `m[r0][j]` — so index loops are clearer than split-borrow iterator gymnastics.
#[allow(clippy::needless_range_loop)]
pub fn invariant_factors(mut m: Vec<Vec<Poly>>) -> Vec<Poly> {
    let rows = m.len();
    let cols = m.first().map_or(0, Vec::len);
    let mut factors = Vec::new();
    let (mut r0, mut c0) = (0, 0);
    while r0 < rows && c0 < cols {
        // Pivot = minimum-degree nonzero entry of the active submatrix.
        let mut piv: Option<(usize, usize)> = None;
        for i in r0..rows {
            for j in c0..cols {
                if !m[i][j].is_zero()
                    && piv.is_none_or(|(pi, pj)| deg(&m[i][j]) < deg(&m[pi][pj]))
                {
                    piv = Some((i, j));
                }
            }
        }
        let Some((pi, pj)) = piv else { break };
        m.swap(r0, pi);
        for row in &mut m {
            row.swap(c0, pj);
        }

        loop {
            let mut changed = false;
            let p = m[r0][c0].clone();
            for i in 0..rows {
                if i != r0 && !m[i][c0].is_zero() {
                    let q = div(&m[i][c0], &p);
                    if !q.is_zero() {
                        for j in 0..cols {
                            let t = mul(&q, &m[r0][j]);
                            m[i][j] ^= t;
                        }
                        changed = true;
                    }
                }
            }
            let p = m[r0][c0].clone();
            for j in 0..cols {
                if j != c0 && !m[r0][j].is_zero() {
                    let q = div(&m[r0][j], &p);
                    if !q.is_zero() {
                        for i in 0..rows {
                            let t = mul(&q, &m[i][c0]);
                            m[i][j] ^= t;
                        }
                        changed = true;
                    }
                }
            }
            // A nonzero residual left in the pivot row/column (degree below the
            // pivot): swap it in and keep reducing.
            let mut resid = None;
            for i in r0 + 1..rows {
                if !m[i][c0].is_zero() {
                    resid = Some((i, c0));
                }
            }
            for j in c0 + 1..cols {
                if !m[r0][j].is_zero() {
                    resid = Some((r0, j));
                }
            }
            if let Some((i, j)) = resid {
                m.swap(r0, i);
                if j != c0 {
                    for row in &mut m {
                        row.swap(c0, j);
                    }
                }
                changed = true;
            }
            if !changed {
                break;
            }
        }

        if deg(&m[r0][c0]) >= 1 {
            factors.push(m[r0][c0].clone());
        }
        r0 += 1;
        c0 += 1;
    }
    factors
}
