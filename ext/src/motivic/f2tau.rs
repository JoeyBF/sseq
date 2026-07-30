//! Dense linear algebra over the PID $\mathbb{F}_2[\tau]$, with polynomials packed
//! as [`Poly`] — an arbitrary-length bitset of exponents (bit $i$ = coefficient of
//! $\tau^i$), stored as `u64` limbs.
//!
//! Used to read the motivic Adams $E_2$ as an $\mathbb{F}_2[\tau]$-module: the
//! $\tau$-torsion orders are the non-unit invariant factors of $\delta$
//! ([`smith`], Smith normal form), and Massey-product coset membership
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

    /// Whether this is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// Add $\tau^{\mathrm{exp}}$ (toggle the coefficient of $\tau^{\mathrm{exp}}$).
    /// Panics (bounds check) if `exp >= 64 * LIMBS`.
    pub fn toggle(&mut self, exp: u32) {
        self.0[(exp / 64) as usize] ^= 1u64 << (exp % 64);
    }

    /// The coefficient of $\tau^{\mathrm{exp}}$ (bit `exp`): `true` iff set. Panics
    /// (bounds check) if `exp >= 64 * LIMBS`.
    pub fn coeff(&self, exp: u32) -> bool {
        self.0[(exp / 64) as usize] & (1u64 << (exp % 64)) != 0
    }

    /// Exact division by $\tau^k$ (shift the bitset right by `k`), dropping any bits
    /// below `k`. When `self` is divisible by $\tau^k$ (no set bit `< k`) this is the
    /// exact quotient.
    fn shr(&self, k: u32) -> Poly {
        if self.is_zero() {
            return Poly::zero();
        }
        let limb_shift = (k / 64) as usize;
        let bit_shift = k % 64;
        let mut out = [0u64; LIMBS];
        for i in limb_shift..LIMBS {
            let w = self.0[i];
            if w == 0 {
                continue;
            }
            out[i - limb_shift] ^= w >> bit_shift;
            if bit_shift > 0 && i - limb_shift >= 1 {
                out[i - limb_shift - 1] ^= w << (64 - bit_shift);
            }
        }
        Poly(out)
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

/// The Smith normal form of `m` over $\mathbb{F}_2[\tau]$: its `rank` (number of
/// nonzero invariant factors = the rank over the fraction field $\mathbb{F}_2(\tau)$,
/// i.e. inverting $\tau$) and the non-unit invariant `factors` (degree $\ge 1$, the
/// $\tau$-torsion orders). `m` is consumed.
///
/// One source of truth for the graded δ: over the PID $\mathbb{F}_2[\tau]$ an
/// invariant factor $\tau^r$ is exactly a length-$r$ differential $d_r$ of the
/// τ-Bockstein SS, and `rank` (invert $\tau$) is what the $E_\infty$ free part
/// counts. So [`crate::motivic::MotivicResolution::tau_module`] reads both its free
/// rank and its torsion off this one call, while [`smith_with_sources`] reads the
/// *same* SNF for the SS's differential vectors.
pub fn smith(m: Vec<Vec<Poly>>) -> (usize, Vec<Poly>) {
    let (rank, diag, _) = snf_core(m, false);
    let factors = diag.into_iter().filter(|p| deg(p) >= 1).collect();
    (rank, factors)
}

/// One differential of the τ-Bockstein SS, read off the graded SNF of the outgoing
/// δ: `(order, source, target)` with `δ(source) = τ^{order} · target` over
/// $\mathbb{F}_2[\tau]$. `source` is a combination of the original **rows** (source
/// generators) and `target` of the original **columns** (target generators); each is
/// weight-homogeneous, so its $\tau^0$ part is the weight-pure $E_1$/SS
/// representative (the lowest-weight term of the homogeneous combination).
pub struct SmithDifferential {
    /// The differential length $r$ = the invariant factor $\tau^r$.
    pub order: u32,
    /// The source combination over the original rows (source generators).
    pub source: Vec<Poly>,
    /// The target combination over the original columns: $\delta(\text{source}) / \tau^r$.
    pub target: Vec<Poly>,
}

/// The **same** SNF as [`smith`], but tracking the change of basis so each non-unit
/// pivot $\tau^r$ is returned as a [`SmithDifferential`] — the source/target vectors
/// of the corresponding τ-Bockstein $d_r$. Over $\mathbb{F}_2[\tau]$ every δ entry is
/// τ-divisible (δ raises the weight), so every pivot is non-unit and yields a
/// differential; the caller projects each homogeneous combination onto its $\tau^0$
/// (weight-pure) part to get the Sseq source and target.
pub fn smith_with_sources(m: Vec<Vec<Poly>>) -> Vec<SmithDifferential> {
    let cols = m.first().map_or(0, Vec::len);
    // Keep the original δ to reconstruct each pivot's target from its source combo.
    let m0 = m.clone();
    let (rank, diag, srow) = snf_core(m, true);
    let mut out = Vec::new();
    for k in 0..rank {
        let order = deg(&diag[k]);
        if order < 1 {
            continue; // a unit pivot carries no differential (does not occur for δ)
        }
        let source = srow[k].clone();
        // target_full[j] = Σᵢ source[i]·m0[i][j] = δ(source), divisible by τ^order.
        let mut target = vec![Poly::zero(); cols];
        for (i, si) in source.iter().enumerate() {
            if si.is_zero() {
                continue;
            }
            for (j, tj) in target.iter_mut().enumerate() {
                *tj ^= mul(si, &m0[i][j]);
            }
        }
        for tj in &mut target {
            debug_assert!(
                (0..order as u32).all(|e| !tj.coeff(e)),
                "δ(source) not divisible by the pivot τ^order"
            );
            *tj = tj.shr(order as u32);
        }
        out.push(SmithDifferential {
            order: order as u32,
            source,
            target,
        });
    }
    out
}

/// The shared SNF engine behind [`smith`] and [`smith_with_sources`]. Returns the
/// `rank` (pivot count), the diagonal `diag` entry of every pivot, and — when
/// `track_sources` — the source combination `srow[k]` (over the original rows) that
/// the $k$-th pivot represents. `m` is consumed.
///
/// Standard Euclidean SNF: pivot on the minimum-degree nonzero entry, clear its
/// row and column by division, and pull any lower-degree residual back into the
/// pivot until the pivot divides the remaining block; then recurse on the
/// complementary submatrix. Row operations (and row swaps) are mirrored into `srow`
/// so `srow[k]` accumulates the change of source basis; column operations do not
/// touch the source and are ignored there. Because δ is graded (each entry a monomial
/// $\tau^{w_j-w_i}$), every operation is weight-homogeneous, so each `srow[k]` comes
/// out weight-homogeneous.
// The row/column clears index two rows (or columns) at once — `m[i][j]` against
// `m[r0][j]` — so index loops are clearer than split-borrow iterator gymnastics.
#[allow(clippy::needless_range_loop)]
fn snf_core(mut m: Vec<Vec<Poly>>, track_sources: bool) -> (usize, Vec<Poly>, Vec<Vec<Poly>>) {
    let rows = m.len();
    let cols = m.first().map_or(0, Vec::len);
    let mut diag = Vec::new();
    // srow[i] = the combination of original rows currently sitting in row i (the
    // identity to start). Only maintained when tracking sources.
    let mut srow: Vec<Vec<Poly>> = if track_sources {
        (0..rows)
            .map(|i| {
                let mut v = vec![Poly::zero(); rows];
                v[i].toggle(0);
                v
            })
            .collect()
    } else {
        Vec::new()
    };
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
        if track_sources {
            srow.swap(r0, pi);
        }
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
                        if track_sources {
                            for k in 0..rows {
                                let t = mul(&q, &srow[r0][k]);
                                srow[i][k] ^= t;
                            }
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
                if track_sources && i != r0 {
                    srow.swap(r0, i);
                }
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

        diag.push(m[r0][c0].clone());
        r0 += 1;
        c0 += 1;
    }
    (r0, diag, srow) // r0 = number of pivots = rank over F₂(τ)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The monomial $\tau^e$.
    fn tau(e: u32) -> Poly {
        let mut p = Poly::zero();
        p.toggle(e);
        p
    }

    /// `Σⱼ v[j]·rows[·][j]`-style matrix/vector product used to reconstruct
    /// $\delta(\text{source})$ from a source combination over rows.
    fn apply_delta(m: &[Vec<Poly>], source: &[Poly]) -> Vec<Poly> {
        let cols = m.first().map_or(0, Vec::len);
        let mut out = vec![Poly::zero(); cols];
        for (i, si) in source.iter().enumerate() {
            for (j, tj) in out.iter_mut().enumerate() {
                *tj ^= mul(si, &m[i][j]);
            }
        }
        out
    }

    #[test]
    fn shr_is_exact_inverse_of_shl() {
        for e in [0u32, 1, 5, 63, 64, 65, 130] {
            let p = tau(e);
            for k in 0..=e {
                assert_eq!(p.shr(k), tau(e - k), "τ^{e} / τ^{k}");
            }
        }
        // A sum divides cleanly when every term clears the shift.
        let mut p = tau(3);
        p ^= tau(5);
        let q = p.shr(3);
        assert!(q.coeff(0) && q.coeff(2) && !q.coeff(1), "(τ³+τ⁵)/τ³ = 1+τ²");
    }

    #[test]
    fn smith_with_sources_matches_smith_and_reconstructs() {
        // A graded δ: sources e0,e1 (weight 0), targets f0 (weight 1), f1 (weight 2).
        // δ(e0) = τ·f0 + τ²·f1, δ(e1) = τ²·f1. Entry (i,j) = τ^{w_j - w_i}.
        let m = vec![vec![tau(1), tau(2)], vec![Poly::zero(), tau(2)]];

        let (rank, factors) = smith(m.clone());
        assert_eq!(rank, 2, "both sources map nontrivially");
        let mut orders: Vec<i32> = factors.iter().map(deg).collect();
        orders.sort_unstable();
        assert_eq!(orders, vec![1, 2], "invariant factors τ¹, τ²");

        let diffs = smith_with_sources(m.clone());
        let mut got: Vec<u32> = diffs.iter().map(|d| d.order).collect();
        got.sort_unstable();
        assert_eq!(got, vec![1, 2], "smith_with_sources agrees on the orders");

        // Each differential satisfies δ(source) = τ^order · target exactly.
        for d in &diffs {
            let lhs = apply_delta(&m, &d.source);
            let rhs: Vec<Poly> = d.target.iter().map(|t| mul(t, &tau(d.order))).collect();
            assert_eq!(lhs, rhs, "δ(source) = τ^{} · target", d.order);
            // The τ⁰ (weight-pure) source part is nonzero — a valid E₁ representative.
            assert!(
                d.source.iter().any(|p| p.coeff(0)),
                "source has a τ⁰ (weight-pure) part"
            );
        }
    }

    #[test]
    fn smith_with_sources_weight_pure_parts_are_homogeneous() {
        // Weights: sources e0,e1,e2 at 0,0,1; targets f0,f1 at 1,2.
        // δ(e0)=τ·f0, δ(e1)=τ²·f1, δ(e2)=τ·f1 (all entries τ^{w_j-w_i}, positive).
        let weights_src = [0i32, 0, 1];
        let weights_tgt = [1i32, 2];
        let m = vec![
            vec![tau(1), Poly::zero()],
            vec![Poly::zero(), tau(2)],
            vec![Poly::zero(), tau(1)],
        ];
        for d in smith_with_sources(m) {
            // All τ⁰ source generators share one weight w; all τ⁰ target generators
            // share weight w + order.
            let src_w: Vec<i32> = (0..weights_src.len())
                .filter(|&i| d.source[i].coeff(0))
                .map(|i| weights_src[i])
                .collect();
            let tgt_w: Vec<i32> = (0..weights_tgt.len())
                .filter(|&j| d.target[j].coeff(0))
                .map(|j| weights_tgt[j])
                .collect();
            assert!(!src_w.is_empty(), "nonempty weight-pure source");
            let w = src_w[0];
            assert!(src_w.iter().all(|&x| x == w), "source weight-pure");
            assert!(
                tgt_w.iter().all(|&x| x == w + d.order as i32),
                "target at weight w + order"
            );
        }
    }
}
