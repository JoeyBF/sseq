//! The Hopf algebroid $(BP_*, BP_*BP)$ at $p = 2$, through a fixed degree.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

/// A polynomial with rational coefficients: exponent vectors, all of one length, to coefficients.
pub type Poly = BTreeMap<Vec<u32>, BigRational>;

/// $|v_n| = |t_n| = 2(2^n - 1)$.
pub fn generator_degree(n: usize) -> i32 {
    2 * ((1 << n) - 1)
}

/// Polynomial arithmetic in variables of fixed degrees, dropping terms above a maximum degree.
pub struct Ring {
    /// The degree of each variable.
    degrees: Vec<i32>,
    /// Terms of higher degree are dropped.
    max_degree: i32,
}

impl Ring {
    /// The ring on variables of degrees `degrees`, truncated above `max_degree`.
    pub fn new(degrees: Vec<i32>, max_degree: i32) -> Self {
        Self {
            degrees,
            max_degree,
        }
    }

    /// The number of variables.
    pub fn len(&self) -> usize {
        self.degrees.len()
    }

    /// Whether there are no variables.
    pub fn is_empty(&self) -> bool {
        self.degrees.is_empty()
    }

    /// The degree of a monomial.
    pub fn degree(&self, m: &[u32]) -> i32 {
        m.iter()
            .zip(&self.degrees)
            .map(|(&e, &d)| e as i32 * d)
            .sum()
    }

    /// The constant `c`.
    pub fn constant(&self, c: BigRational) -> Poly {
        let mut p = Poly::new();
        if !c.is_zero() {
            p.insert(vec![0; self.len()], c);
        }
        p
    }

    /// The constant 1.
    pub fn one(&self) -> Poly {
        self.constant(BigRational::one())
    }

    /// The variable `i`, or zero if it lies above the maximum degree.
    pub fn var(&self, i: usize) -> Poly {
        let mut m = vec![0; self.len()];
        m[i] = 1;
        let mut p = Poly::new();
        if self.degree(&m) <= self.max_degree {
            p.insert(m, BigRational::one());
        }
        p
    }

    /// Adds `c · b` to `a`.
    pub fn add_assign(&self, a: &mut Poly, b: &Poly, c: &BigRational) {
        for (m, x) in b {
            let e = a.entry(m.clone()).or_insert_with(BigRational::zero);
            *e += x * c;
            if e.is_zero() {
                a.remove(m);
            }
        }
    }

    /// `a + c · b`.
    pub fn add(&self, a: &Poly, b: &Poly, c: &BigRational) -> Poly {
        let mut r = a.clone();
        self.add_assign(&mut r, b, c);
        r
    }

    /// `c · a`.
    pub fn scale(&self, a: &Poly, c: &BigRational) -> Poly {
        if c.is_zero() {
            return Poly::new();
        }
        a.iter().map(|(m, x)| (m.clone(), x * c)).collect()
    }

    /// `a · b`, truncated.
    pub fn mul(&self, a: &Poly, b: &Poly) -> Poly {
        let mut r = Poly::new();
        for (ma, xa) in a {
            let da = self.degree(ma);
            for (mb, xb) in b {
                if da + self.degree(mb) > self.max_degree {
                    continue;
                }
                let m: Vec<u32> = ma.iter().zip(mb).map(|(x, y)| x + y).collect();
                let e = r.entry(m).or_insert_with(BigRational::zero);
                *e += xa * xb;
            }
        }
        r.retain(|_, x| !x.is_zero());
        r
    }

    /// `a^(2^i)`, truncated.
    pub fn pow2(&self, a: &Poly, i: u32) -> Poly {
        let mut r = a.clone();
        for _ in 0..i {
            r = self.mul(&r, &r);
        }
        r
    }

    /// The image of `a` under the ring map sending variable `j` to `images[j]`.
    pub fn substitute(&self, a: &Poly, images: &[Poly], target: &Self) -> Poly {
        let mut r = Poly::new();
        for (m, x) in a {
            let mut term = target.constant(x.clone());
            for (j, &e) in m.iter().enumerate() {
                for _ in 0..e {
                    term = target.mul(&term, &images[j]);
                }
            }
            target.add_assign(&mut r, &term, &BigRational::one());
        }
        r
    }
}

/// The 2-adic valuation of a nonzero rational.
pub fn valuation(x: &BigRational) -> i64 {
    let two = |n: &BigInt| n.trailing_zeros().unwrap_or(0) as i64;
    two(x.numer()) - two(x.denom())
}

/// The Hazewinkel structure of $BP_*BP$ through degree `max_degree`.
///
/// Variables are $v_1, \dots, v_N$ and then $t_1, \dots, t_N$, for the largest $N$ with
/// $|v_N|$ at most `max_degree`; for $\Delta$, a second set $t'_1, \dots, t'_N$ follows, the right
/// tensor factor.
pub struct BpHopfAlgebroid {
    /// The number of generators of each kind.
    pub n: usize,
    /// $BP_*BP$, on the $v$s and $t$s.
    pub bpbp: Ring,
    /// $BP_*BP \otimes_{BP_*} BP_*BP$, on the $v$s, $t$s and $t'$s.
    pub bpbp2: Ring,
    /// $\ell_0, \dots, \ell_N$ in terms of the $v$s, as elements of `bpbp`.
    pub log: Vec<Poly>,
    /// $\eta_R(v_0), \dots, \eta_R(v_N)$, with $v_0 = 2$.
    pub eta_r: Vec<Poly>,
    /// $\Delta(t_0), \dots, \Delta(t_N)$, as elements of `bpbp2`.
    pub delta: Vec<Poly>,
}

impl BpHopfAlgebroid {
    /// The structure maps through degree `max_degree`.
    pub fn new(max_degree: i32) -> Self {
        let n = (1..)
            .take_while(|&i| generator_degree(i) <= max_degree)
            .last()
            .unwrap_or(0);
        let degs: Vec<i32> = (1..=n).map(generator_degree).collect();
        let bpbp = Ring::new([degs.clone(), degs.clone()].concat(), max_degree);
        let bpbp2 = Ring::new([degs.clone(), degs.clone(), degs].concat(), max_degree);
        let v = |i: usize| bpbp.var(i - 1);
        let t = |i: usize| {
            if i == 0 {
                bpbp.one()
            } else {
                bpbp.var(n + i - 1)
            }
        };
        let half = BigRational::new(1.into(), 2.into());
        let one = BigRational::one();
        let two = BigRational::from_integer(2.into());

        // 2 ℓ_k = Σ_{i < k} ℓ_i v_{k-i}^{2^i}.
        let mut log = vec![bpbp.one()];
        for k in 1..=n {
            let mut sum = Poly::new();
            for (i, l) in log.iter().enumerate() {
                bpbp.add_assign(
                    &mut sum,
                    &bpbp.mul(l, &bpbp.pow2(&v(k - i), i as u32)),
                    &one,
                );
            }
            log.push(bpbp.scale(&sum, &half));
        }

        // η_R(ℓ_k) = Σ_{i <= k} ℓ_i t_{k-i}^{2^i}.
        let eta_log: Vec<Poly> = (0..=n)
            .map(|k| {
                let mut sum = Poly::new();
                for (i, l) in log.iter().enumerate().take(k + 1) {
                    bpbp.add_assign(
                        &mut sum,
                        &bpbp.mul(l, &bpbp.pow2(&t(k - i), i as u32)),
                        &one,
                    );
                }
                sum
            })
            .collect();

        // v_k = 2 ℓ_k - Σ_{0 < i < k} ℓ_i v_{k-i}^{2^i}, and η_R is a ring map.
        let mut eta_r = vec![bpbp.constant(two.clone())];
        for k in 1..=n {
            let mut e = bpbp.scale(&eta_log[k], &two);
            for i in 1..k {
                let term = bpbp.mul(&eta_log[i], &bpbp.pow2(&eta_r[k - i], i as u32));
                bpbp.add_assign(&mut e, &term, &-one.clone());
            }
            eta_r.push(e);
        }

        // Σ_{i+j=k} ℓ_i Δ(t_j)^{2^i} = Σ_{i+j+l=k} ℓ_i t_j^{2^i} ⊗ t'_l^{2^{i+j}}.
        let lift = |p: &Poly| -> Poly {
            p.iter()
                .map(|(m, x)| {
                    let mut m2 = m.clone();
                    m2.resize(3 * n, 0);
                    (m2, x.clone())
                })
                .collect()
        };
        let log2: Vec<Poly> = log.iter().map(lift).collect();
        let t2 = |i: usize| {
            if i == 0 {
                bpbp2.one()
            } else {
                bpbp2.var(n + i - 1)
            }
        };
        let tp = |i: usize| {
            if i == 0 {
                bpbp2.one()
            } else {
                bpbp2.var(2 * n + i - 1)
            }
        };
        let mut delta = vec![bpbp2.one()];
        for k in 1..=n {
            let mut d = Poly::new();
            for (i, log_i) in log2.iter().enumerate().take(k + 1) {
                for j in 0..=k - i {
                    let l = k - i - j;
                    let term = bpbp2.mul(
                        &bpbp2.mul(log_i, &bpbp2.pow2(&t2(j), i as u32)),
                        &bpbp2.pow2(&tp(l), (i + j) as u32),
                    );
                    bpbp2.add_assign(&mut d, &term, &one);
                }
            }
            for i in 1..=k {
                let term = bpbp2.mul(&log2[i], &bpbp2.pow2(&delta[k - i], i as u32));
                bpbp2.add_assign(&mut d, &term, &-one.clone());
            }
            delta.push(d);
        }

        Self {
            n,
            bpbp,
            bpbp2,
            log,
            eta_r,
            delta,
        }
    }

    /// The $I$-adic filtration of a term $c\, v^I t^J$: the 2-adic valuation of `c` plus $|I|$.
    pub fn adic_filtration(&self, m: &[u32], c: &BigRational) -> i64 {
        valuation(c) + m[..self.n].iter().map(|&e| e as i64).sum::<i64>()
    }
}

/// A rational in Wang's notation `c(e)`, meaning $c \cdot 2^e$.
pub fn wang_coefficient(x: &BigRational) -> String {
    let e = valuation(x);
    let c = if e >= 0 {
        x / BigRational::from_integer(BigInt::from(2).pow(e as u32))
    } else {
        x * BigRational::from_integer(BigInt::from(2).pow((-e) as u32))
    };
    debug_assert!(c.is_integer());
    format!("{}({e})", c.to_integer())
}

/// Whether every coefficient is a 2-adic integer.
pub fn is_integral(p: &Poly) -> bool {
    p.values().all(|x| valuation(x) >= 0 || x.is_zero())
}

/// The largest numerator size, for diagnostics.
pub fn max_bits(p: &Poly) -> u64 {
    p.values()
        .map(|x| x.numer().abs().bits())
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The terms of a polynomial in Wang's notation, keyed by monomial text, for comparison with
    /// the formulas his code prints.
    fn wang_terms(h: &BpHopfAlgebroid, p: &Poly) -> BTreeMap<String, String> {
        let n = h.n;
        p.iter()
            .map(|(m, x)| {
                let named = |name: &str, exps: &[u32]| -> String {
                    exps.iter()
                        .enumerate()
                        .filter(|&(_, &e)| e > 0)
                        .map(|(i, e)| format!("{name}_{}^{e}", i + 1))
                        .collect()
                };
                let (vs, ts) = (named("v", &m[..n]), named("t", &m[n..2 * n]));
                (format!("{vs}[{ts}]"), wang_coefficient(x))
            })
            .collect()
    }

    /// Parses `c(e)v_..[t_..]` terms of one of Wang's printed formulas.
    fn parse_wang(s: &str) -> BTreeMap<String, String> {
        let body = s.split_once("= o+").unwrap().1;
        let mut out = BTreeMap::new();
        let mut rest = body.trim();
        while !rest.is_empty() {
            // A coefficient `c(e)` ends at `)`; the monomial runs to the next `+`/`-` that starts
            // a new coefficient, i.e. one followed by a digit.
            let close = rest.find(')').unwrap();
            let coef = &rest[..=close];
            let after = &rest[close + 1..];
            let mut end = after.len();
            for (i, ch) in after.char_indices() {
                if ch == '+' {
                    end = i;
                    break;
                }
            }
            out.insert(after[..end].to_string(), coef.to_string());
            rest = after[end..].trim_start_matches('+');
        }
        out
    }

    /// η_R(v_1), …, η_R(v_4) as printed by Wang's `BPtab 25`.
    const WANG_ETA_R: [&str; 4] = [
        "etaR(v1) = o+1(1)[t_1^1]+1(0)v_1^1[]",
        "etaR(v2) = o+-1(2)[t_1^3]+1(1)[t_2^1]+-5(0)v_1^1[t_1^2]+-3(0)v_1^2[t_1^1]+1(0)v_2^1[]",
        "etaR(v3) = o+-1(4)[t_1^7]+-1(2)[t_1^1t_2^2]+1(1)[t_3^1]+-7(3)v_1^1[t_1^6]+-1(2)v_1^1[t_1^3t_2^1]+-1(0)v_1^1[t_2^2]+-85(0)v_1^2[t_1^5]+-1(1)v_1^2[t_1^2t_2^1]+-35(1)v_1^3[t_1^4]+-1(1)v_1^3[t_1^1t_2^1]+-9(2)v_1^4[t_1^3]+-1(0)v_1^4[t_2^1]+-11(0)v_1^5[t_1^2]+-1(1)v_1^6[t_1^1]+1(0)v_2^1[t_1^4]+-1(2)v_2^1[t_1^1t_2^1]+-1(1)v_1^1v_2^1[t_1^3]+-1(1)v_1^1v_2^1[t_2^1]+-1(0)v_1^2v_2^1[t_1^2]+-1(0)v_1^3v_2^1[t_1^1]+-1(0)v_2^2[t_1^1]+1(0)v_3^1[]",
        "etaR(v4) = o+-1(8)[t_1^15]+-1(8)[t_1^12t_2^1]+3(7)[t_1^9t_2^2]+-3(7)[t_1^6t_2^3]+7(4)[t_1^3t_2^4]+-1(4)[t_2^5]+-3(6)[t_1^8t_3^1]+1(4)[t_1^2t_2^2t_3^1]+-1(2)[t_1^1t_3^2]+1(1)[t_4^1]+-1(11)v_1^1[t_1^14]+-9(7)v_1^1[t_1^11t_2^1]+33(5)v_1^1[t_1^8t_2^2]+-29(5)v_1^1[t_1^5t_2^3]+17(3)v_1^1[t_1^2t_2^4]+-3(8)v_1^1[t_1^7t_3^1]+1(4)v_1^1[t_1^4t_2^1t_3^1]+3(2)v_1^1[t_1^1t_2^2t_3^1]+-1(0)v_1^1[t_3^2]+-231(5)v_1^2[t_1^13]+-87(5)v_1^2[t_1^10t_2^1]+189(3)v_1^2[t_1^7t_2^2]+-71(4)v_1^2[t_1^4t_2^3]+91(0)v_1^2[t_1^1t_2^4]+-335(2)v_1^2[t_1^6t_3^1]+1(4)v_1^2[t_1^3t_2^1t_3^1]+1(1)v_1^2[t_2^2t_3^1]+-1025(4)v_1^3[t_1^12]+-525(3)v_1^3[t_1^9t_2^1]+575(1)v_1^3[t_1^6t_2^2]+-21(5)v_1^3[t_1^3t_2^3]+-1(2)v_1^3[t_2^4]+-671(1)v_1^3[t_1^5t_3^1]+3(2)v_1^3[t_1^2t_2^1t_3^1]+-25113(0)v_1^4[t_1^11]+-4309(0)v_1^4[t_1^8t_2^1]+347(0)v_1^4[t_1^5t_2^2]+-99(1)v_1^4[t_1^2t_2^3]+-209(2)v_1^4[t_1^4t_3^1]+1(3)v_1^4[t_1^1t_2^1t_3^1]+-28177(0)v_1^5[t_1^10]+-1519(1)v_1^5[t_1^7t_2^1]+-141(1)v_1^5[t_1^4t_2^2]+1(4)v_1^5[t_1^1t_2^3]+-83(2)v_1^5[t_1^3t_3^1]+1(1)v_1^5[t_2^1t_3^1]+-5989(2)v_1^6[t_1^9]+-183(3)v_1^6[t_1^6t_2^1]+-161(1)v_1^6[t_1^3t_2^2]+-1(0)v_1^6[t_2^3]+-41(1)v_1^6[t_1^2t_3^1]+-15767(0)v_1^7[t_1^8]+-451(0)v_1^7[t_1^5t_2^1]+-141(0)v_1^7[t_1^2t_2^2]+-3(2)v_1^7[t_1^1t_3^1]+-1021(3)v_1^8[t_1^7]+-107(0)v_1^8[t_1^4t_2^1]+-13(0)v_1^8[t_1^1t_2^2]+-1(0)v_1^8[t_3^1]+-1699(1)v_1^9[t_1^6]+-19(1)v_1^9[t_1^3t_2^1]+-1(0)v_1^9[t_2^2]+-291(2)v_1^10[t_1^5]+-19(0)v_1^10[t_1^2t_2^1]+-337(0)v_1^11[t_1^4]+-1(1)v_1^11[t_1^1t_2^1]+-41(1)v_1^12[t_1^3]+-1(4)v_1^13[t_1^2]+-1(1)v_1^14[t_1^1]+-7(5)v_2^1[t_1^12]+3(7)v_2^1[t_1^9t_2^1]+-71(3)v_2^1[t_1^6t_2^2]+7(5)v_2^1[t_1^3t_2^3]+-39(0)v_2^1[t_2^4]+-1(2)v_2^1[t_1^5t_3^1]+1(4)v_2^1[t_1^2t_2^1t_3^1]+-15(6)v_1^1v_2^1[t_1^11]+145(3)v_1^1v_2^1[t_1^8t_2^1]+-693(1)v_1^1v_2^1[t_1^5t_2^2]+33(3)v_1^1v_2^1[t_1^2t_2^3]+3(1)v_1^1v_2^1[t_1^4t_3^1]+1(4)v_1^1v_2^1[t_1^1t_2^1t_3^1]+-1031(1)v_1^2v_2^1[t_1^10]+119(4)v_1^2v_2^1[t_1^7t_2^1]+-1711(0)v_1^2v_2^1[t_1^4t_2^2]+11(4)v_1^2v_2^1[t_1^1t_2^3]+1(3)v_1^2v_2^1[t_1^3t_3^1]+1(2)v_1^2v_2^1[t_2^1t_3^1]+-2771(0)v_1^3v_2^1[t_1^9]+913(1)v_1^3v_2^1[t_1^6t_2^1]+-127(3)v_1^3v_2^1[t_1^3t_2^2]+-5(1)v_1^3v_2^1[t_2^3]+3(1)v_1^3v_2^1[t_1^2t_3^1]+-655(2)v_1^4v_2^1[t_1^8]+511(1)v_1^4v_2^1[t_1^5t_2^1]+-299(0)v_1^4v_2^1[t_1^2t_2^2]+1(1)v_1^4v_2^1[t_1^1t_3^1]+-1877(0)v_1^5v_2^1[t_1^7]+141(0)v_1^5v_2^1[t_1^4t_2^1]+23(0)v_1^5v_2^1[t_1^1t_2^2]+-277(2)v_1^6v_2^1[t_1^6]+-19(3)v_1^6v_2^1[t_1^3t_2^1]+-1(1)v_1^6v_2^1[t_2^2]+-567(0)v_1^7v_2^1[t_1^5]+-97(0)v_1^7v_2^1[t_1^2t_2^1]+-263(0)v_1^8v_2^1[t_1^4]+-5(0)v_1^8v_2^1[t_1^1t_2^1]+-51(1)v_1^9v_2^1[t_1^3]+-15(1)v_1^10v_2^1[t_1^2]+-1(2)v_1^11v_2^1[t_1^1]+95(0)v_2^2[t_1^9]+-35(3)v_2^2[t_1^6t_2^1]+21(3)v_2^2[t_1^3t_2^2]+-5(3)v_2^2[t_2^3]+1(2)v_2^2[t_1^2t_3^1]+61(2)v_1^1v_2^2[t_1^8]+-43(4)v_1^1v_2^2[t_1^5t_2^1]+99(1)v_1^1v_2^2[t_1^2t_2^2]+1(1)v_1^1v_2^2[t_1^1t_3^1]+143(1)v_1^2v_2^2[t_1^7]+-213(2)v_1^2v_2^2[t_1^4t_2^1]+131(0)v_1^2v_2^2[t_1^1t_2^2]+123(0)v_1^3v_2^2[t_1^6]+-253(1)v_1^3v_2^2[t_1^3t_2^1]+-1(3)v_1^3v_2^2[t_2^2]+-5(4)v_1^4v_2^2[t_1^5]+-73(1)v_1^4v_2^2[t_1^2t_2^1]+-173(0)v_1^5v_2^2[t_1^4]+15(0)v_1^5v_2^2[t_1^1t_2^1]+-15(3)v_1^6v_2^2[t_1^3]+-11(2)v_1^7v_2^2[t_1^2]+-1(2)v_1^8v_2^2[t_1^1]+-23(1)v_2^3[t_1^6]+7(3)v_2^3[t_1^3t_2^1]+-5(2)v_2^3[t_2^2]+-115(0)v_1^1v_2^3[t_1^5]+17(2)v_1^1v_2^3[t_1^2t_2^1]+-141(0)v_1^2v_2^3[t_1^4]+23(1)v_1^2v_2^3[t_1^1t_2^1]+-83(0)v_1^3v_2^3[t_1^3]+-1(1)v_1^3v_2^3[t_2^1]+-23(0)v_1^4v_2^3[t_1^2]+3(0)v_1^5v_2^3[t_1^1]+7(0)v_2^4[t_1^3]+-5(0)v_2^4[t_2^1]+9(0)v_1^1v_2^4[t_1^2]+3(1)v_1^2v_2^4[t_1^1]+-95(0)v_3^1[t_1^8]+1(3)v_3^1[t_1^2t_2^2]+-1(2)v_3^1[t_1^1t_3^1]+-3(7)v_1^1v_3^1[t_1^7]+1(3)v_1^1v_3^1[t_1^4t_2^1]+3(1)v_1^1v_3^1[t_1^1t_2^2]+-1(1)v_1^1v_3^1[t_3^1]+-335(1)v_1^2v_3^1[t_1^6]+1(3)v_1^2v_3^1[t_1^3t_2^1]+1(0)v_1^2v_3^1[t_2^2]+-671(0)v_1^3v_3^1[t_1^5]+3(1)v_1^3v_3^1[t_1^2t_2^1]+-209(1)v_1^4v_3^1[t_1^4]+1(2)v_1^4v_3^1[t_1^1t_2^1]+-83(1)v_1^5v_3^1[t_1^3]+1(0)v_1^5v_3^1[t_2^1]+-41(0)v_1^6v_3^1[t_1^2]+-3(1)v_1^7v_3^1[t_1^1]+-1(1)v_2^1v_3^1[t_1^5]+1(3)v_2^1v_3^1[t_1^2t_2^1]+3(0)v_1^1v_2^1v_3^1[t_1^4]+1(3)v_1^1v_2^1v_3^1[t_1^1t_2^1]+1(2)v_1^2v_2^1v_3^1[t_1^3]+1(1)v_1^2v_2^1v_3^1[t_2^1]+3(0)v_1^3v_2^1v_3^1[t_1^2]+1(0)v_1^4v_2^1v_3^1[t_1^1]+1(1)v_2^2v_3^1[t_1^2]+1(0)v_1^1v_2^2v_3^1[t_1^1]+-1(0)v_3^2[t_1^1]+1(0)v_4^1[]",
    ];

    #[test]
    fn eta_r_matches_wang() {
        let h = BpHopfAlgebroid::new(50);
        for (k, s) in WANG_ETA_R.iter().enumerate() {
            assert_eq!(
                wang_terms(&h, &h.eta_r[k + 1]),
                parse_wang(s),
                "η_R(v{})",
                k + 1
            );
        }
    }

    #[test]
    fn structure_maps_are_integral() {
        let h = BpHopfAlgebroid::new(70);
        for (k, e) in h.eta_r.iter().enumerate() {
            assert!(is_integral(e), "η_R(v{k})");
        }
        for (k, d) in h.delta.iter().enumerate() {
            assert!(is_integral(d), "Δ(t{k})");
        }
    }

    /// Modulo $I^2$, $\eta_R(v_n) = \sum_i v_i t_{n-i}^{2^i}$: the coaction on $Q$.
    #[test]
    fn eta_r_is_the_q_coaction_mod_i_squared() {
        let h = BpHopfAlgebroid::new(70);
        let n = h.n;
        for k in 1..=n {
            let mut linear = BTreeMap::new();
            for (m, x) in &h.eta_r[k] {
                let f = h.adic_filtration(m, x);
                assert!(f >= 1, "η_R(v{k}) has a term of filtration {f}");
                if f == 1 {
                    linear.insert(m.clone(), ());
                }
            }
            let mut expected = BTreeMap::new();
            for i in 0..=k {
                let mut m = vec![0; 2 * n];
                if i > 0 {
                    m[i - 1] = 1;
                }
                if k > i {
                    m[n + k - i - 1] = 1 << i;
                }
                // v_0 t_k with v_0 = 2 is 2 t_k: the same monomial, filtration from the 2.
                if h.bpbp.degree(&m) <= 70 {
                    expected.insert(m, ());
                }
            }
            assert_eq!(linear, expected, "η_R(v{k}) mod I^2");
        }
    }

    /// $(\epsilon \otimes 1)\Delta = 1 = (1 \otimes \epsilon)\Delta$.
    #[test]
    fn delta_is_counital() {
        let h = BpHopfAlgebroid::new(70);
        let n = h.n;
        for k in 1..=n {
            let keep = |range: std::ops::Range<usize>| -> Poly {
                h.delta[k]
                    .iter()
                    .filter(|(m, _)| range.clone().all(|i| m[i] == 0))
                    .map(|(m, x)| (m.clone(), x.clone()))
                    .collect()
            };
            let mut right = vec![0; 3 * n];
            right[2 * n + k - 1] = 1;
            let mut left = vec![0; 3 * n];
            left[n + k - 1] = 1;
            let one = BigRational::one();
            assert_eq!(
                keep(n..2 * n),
                [(right, one.clone())].into(),
                "t = 0 in Δ(t{k})"
            );
            assert_eq!(
                keep(2 * n..3 * n),
                [(left, one)].into(),
                "t' = 0 in Δ(t{k})"
            );
        }
    }
}
