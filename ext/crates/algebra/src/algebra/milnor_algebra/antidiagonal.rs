//! Enumeration of Milnor products at the prime 2, one anti-diagonal at a time.

use super::PPart;

/// The number of interior entries (not in row or column 0) on anti-diagonals of index at most
/// [`PPart::MAX_LEN`].
const MAX_CELLS: usize = PPart::MAX_LEN * (PPart::MAX_LEN - 1) / 2;

/// Call `emit` with every p-part $T$ such that $P(T)$ appears in $P(r) P(s)$ at the prime 2, with
/// multiplicity: equal terms may be emitted several times and should be summed mod 2.
///
/// $P(r) P(s)$ is the sum, over matrices $X$ with $\sum_j 2^j x_{ij} = r_i$ for each row and
/// $\sum_i x_{ij} = s_j$ for each column, of $P(T(X))$, where $T_d$ is the sum of the $d$th
/// anti-diagonal. Mod 2 a matrix contributes exactly when the entries of each anti-diagonal have
/// pairwise disjoint binary digits.
///
/// We fill the interior entries one anti-diagonal at a time, from the last one down. The other
/// entries of row $d$ and column $d$ lie on later anti-diagonals, so on reaching anti-diagonal $d$
/// the slacks $x_{d0}$ and $x_{0d}$ are known. The anti-diagonal starts from their bitwise or, and
/// each interior entry only takes values disjoint from the bits used so far, so every completed
/// matrix contributes.
///
/// `degree` is an upper bound for the polynomial degree of the product. Entries on anti-diagonals
/// $d$ with $2^d - 1 >$ `degree` must vanish, so we never visit them.
pub(super) fn for_each_term(degree: i32, r: PPart, s: PPart, mut emit: impl FnMut(PPart)) {
    let rows = r.len();
    let cols = s.len();
    if rows == 0 || cols == 0 {
        emit(if rows == 0 { s } else { r });
        return;
    }
    let mut max_diag = (rows + cols).min(PPart::MAX_LEN);
    while max_diag > 1 && (1i64 << max_diag) - 1 > i64::from(degree) {
        max_diag -= 1;
    }

    // What is left of `r` and `s`, indexed from 1. Once row (resp. column) `d` is filled, this is
    // its entry in column (resp. row) 0.
    let mut r_left = [0u32; PPart::MAX_LEN + 1];
    let mut s_left = [0u32; PPart::MAX_LEN + 1];
    for i in 0..rows {
        r_left[i + 1] = r.get(i);
    }
    for j in 0..cols {
        s_left[j + 1] = s.get(j);
    }

    // The interior cells in fill order, and whether each is the last of its anti-diagonal.
    let mut row = [0u8; MAX_CELLS];
    let mut col = [0u8; MAX_CELLS];
    let mut last = [false; MAX_CELLS];
    let mut num_cells = 0;
    for d in (2..=max_diag).rev() {
        for i in d.saturating_sub(cols).max(1)..=rows.min(d - 1) {
            row[num_cells] = i as u8;
            col[num_cells] = (d - i) as u8;
            num_cells += 1;
        }
        last[num_cells - 1] = true;
    }

    let (a, b) = (r_left[max_diag], s_left[max_diag]);
    if a & b != 0 {
        return;
    }
    if max_diag == 1 {
        emit(PPart::from_bits(u64::from(a | b) << PPart::shift(0)));
        return;
    }

    // `value[k]` is the current entry of cell `k`, `bound[k]` the most it may be given the entries
    // before it, and `before[k]` the bits used on its anti-diagonal by the cells before it.
    let mut value = [0u32; MAX_CELLS];
    let mut bound = [0u32; MAX_CELLS];
    let mut before = [0u32; MAX_CELLS];
    // The packed value of $T_2, \ldots$, filled in as anti-diagonals are completed.
    let mut t_bits = 0u64;

    let mut k = 0;
    let mut tally = a | b;
    'enter: loop {
        // Value 0 never overlaps.
        let (i, j) = (row[k] as usize, col[k] as usize);
        before[k] = tally;
        bound[k] = (r_left[i] >> j).min(s_left[j]);
        value[k] = 0;
        'advance: loop {
            let used = before[k] | value[k];
            if !last[k] {
                k += 1;
                tally = used;
                continue 'enter;
            }
            // Anti-diagonal `d` is complete, and the slacks of `d - 1` are now known.
            let d = (row[k] + col[k]) as usize;
            let shift = PPart::shift(d - 1);
            t_bits = (t_bits & !(u64::from(PPart::max_entry(d - 1)) << shift))
                | (u64::from(used) << shift);
            let (a, b) = (r_left[d - 1], s_left[d - 1]);
            if a & b == 0 {
                if d == 2 {
                    emit(PPart::from_bits(
                        t_bits | u64::from(a | b) << PPart::shift(0),
                    ));
                } else {
                    k += 1;
                    tally = a | b;
                    continue 'enter;
                }
            }
            // Backtrack to the latest cell that can still be increased.
            loop {
                let (i, j) = (row[k] as usize, col[k] as usize);
                let v = value[k];
                let mask = before[k];
                let next = ((v | mask) + 1) & !mask;
                if next <= bound[k] {
                    r_left[i] -= (next - v) << j;
                    s_left[j] -= next - v;
                    value[k] = next;
                    continue 'advance;
                }
                r_left[i] += v << j;
                s_left[j] += v;
                if k == 0 {
                    return;
                }
                k -= 1;
            }
        }
    }
}

/// The running state of an anti-diagonal's multinomial coefficient mod 4, as its entries are added
/// one at a time.
///
/// The multinomial coefficient of entries $e_1, \ldots, e_k$ is the product of the binomial
/// coefficients $\binom{e_1 + \cdots + e_i}{e_i}$. By Kummer's theorem $\binom{n}{m}$ has 2-adic
/// valuation the number of carries in adding $m$ to $n - m$, so a product of them is nonzero mod 4
/// exactly when at most one carry happens in all, and is then 2 mod 4. With no carry, each factor
/// is $(-1)^{\#\{i : n_{i+1} = n_i = 1,\ m_{i+1} \ne m_i\}}$ mod 4, so the product is $\pm 1$
/// according to the parity of the total count.
#[derive(Clone, Copy, Default)]
struct Mod4 {
    /// The sum of the entries added so far.
    sum: u32,
    /// Whether the single carry the coefficient can afford has been spent.
    carried: bool,
    /// The parity of the sign count, meaningful only while `carried` is false.
    odd: bool,
}

impl Mod4 {
    /// The state after adding `v`, or `None` if the coefficient becomes 0 mod 4.
    #[inline]
    fn add(self, v: u32) -> Option<Self> {
        let n = self.sum + v;
        let carries = self.sum.count_ones() + v.count_ones() - n.count_ones();
        match (carries, self.carried) {
            (0, _) => Some(Self {
                sum: n,
                carried: self.carried,
                odd: self.odd ^ ((n & (n >> 1) & (v ^ (v >> 1))).count_ones() & 1 == 1),
            }),
            (1, false) => Some(Self {
                sum: n,
                carried: true,
                odd: false,
            }),
            _ => None,
        }
    }

    /// The coefficient mod 4, which is never 0.
    fn coefficient(self) -> u32 {
        if self.carried {
            2
        } else if self.odd {
            3
        } else {
            1
        }
    }

    /// The least `v > cur` that can be added without making the coefficient 0 mod 4, or
    /// `u64::MAX` if none fits in a `u32`.
    #[inline]
    fn next(self, cur: u32) -> u64 {
        let s = u64::from(self.sum);
        let mut best = next_with(u64::from(cur), s, 0);
        if !self.carried {
            // One carry: `v` shares exactly one bit `i` with the sum and neither has bit `i + 1`,
            // so the carry stops there.
            let mut bits = s & !(s >> 1);
            while bits != 0 {
                let i = bits.trailing_zeros();
                bits &= bits - 1;
                let forbid = (s & !(1 << i)) | (1 << (i + 1));
                best = best.min(next_with(u64::from(cur), forbid, 1 << i));
            }
        }
        best
    }
}

/// The least `x > cur` with no bit of `forbid` and every bit of `force`, which must be disjoint, or
/// `u64::MAX` if it does not fit in a `u32`.
#[inline]
fn next_with(cur: u64, forbid: u64, force: u64) -> u64 {
    let mut x = cur + 1;
    loop {
        let bad = x & forbid;
        if bad != 0 {
            // Every number from `x` up to `x` with all bits below the top bad bit set has that
            // bit too: step past them.
            let h = 63 - bad.leading_zeros();
            x = (x | ((1 << h) - 1)) + 1;
            continue;
        }
        let missing = force & !x;
        if missing != 0 {
            // The least number at least `x` with the top missing bit is `x` with that bit set and
            // the bits below it cleared.
            let h = 63 - missing.leading_zeros();
            x = ((x >> h) | 1) << h;
            continue;
        }
        return if x > u64::from(u32::MAX) { u64::MAX } else { x };
    }
}

/// Call `emit` with every p-part $T$ such that $P(T)$ appears in $P(r) P(s)$ over $\mathbb{Z}/4$,
/// with its coefficient (1, 2 or 3). Equal terms may be emitted several times and should be summed
/// mod 4.
///
/// This is [`for_each_term`] with each matrix's coefficient taken mod 4 (see [`Mod4`]): an entry
/// may share one bit with its anti-diagonal's running sum, once per matrix, so every completed
/// matrix still contributes. The carry budget and the sign run across anti-diagonals, while the sum
/// restarts at each from its two slacks.
pub(crate) fn for_each_term_mod4(
    degree: i32,
    r: PPart,
    s: PPart,
    mut emit: impl FnMut(PPart, u32),
) {
    let rows = r.len();
    let cols = s.len();
    if rows == 0 || cols == 0 {
        emit(if rows == 0 { s } else { r }, 1);
        return;
    }
    let mut max_diag = (rows + cols).min(PPart::MAX_LEN);
    while max_diag > 1 && (1i64 << max_diag) - 1 > i64::from(degree) {
        max_diag -= 1;
    }

    let mut r_left = [0u32; PPart::MAX_LEN + 1];
    let mut s_left = [0u32; PPart::MAX_LEN + 1];
    for i in 0..rows {
        r_left[i + 1] = r.get(i);
    }
    for j in 0..cols {
        s_left[j + 1] = s.get(j);
    }

    let mut row = [0u8; MAX_CELLS];
    let mut col = [0u8; MAX_CELLS];
    let mut last = [false; MAX_CELLS];
    let mut num_cells = 0;
    for d in (2..=max_diag).rev() {
        for i in d.saturating_sub(cols).max(1)..=rows.min(d - 1) {
            row[num_cells] = i as u8;
            col[num_cells] = (d - i) as u8;
            num_cells += 1;
        }
        last[num_cells - 1] = true;
    }

    let start = |prev: Mod4, a: u32, b: u32| Mod4 { sum: a, ..prev }.add(b);

    let Some(first) = start(Mod4::default(), r_left[max_diag], s_left[max_diag]) else {
        return;
    };
    if max_diag == 1 {
        emit(
            PPart::from_bits(u64::from(first.sum) << PPart::shift(0)),
            first.coefficient(),
        );
        return;
    }

    // `value[k]` is the current entry of cell `k`, `bound[k]` the most it may be given the entries
    // before it, and `before[k]` the state of its anti-diagonal before it.
    let mut value = [0u32; MAX_CELLS];
    let mut bound = [0u32; MAX_CELLS];
    let mut before = [Mod4::default(); MAX_CELLS];
    let mut t_bits = 0u64;

    let mut k = 0;
    let mut state = first;
    'enter: loop {
        let (i, j) = (row[k] as usize, col[k] as usize);
        before[k] = state;
        bound[k] = (r_left[i] >> j).min(s_left[j]);
        value[k] = 0;
        // Adding 0 changes nothing.
        let mut after = state;
        'advance: loop {
            if !last[k] {
                k += 1;
                state = after;
                continue 'enter;
            }
            let d = (row[k] + col[k]) as usize;
            let shift = PPart::shift(d - 1);
            t_bits = (t_bits & !(u64::from(PPart::max_entry(d - 1)) << shift))
                | (u64::from(after.sum) << shift);
            if let Some(next) = start(after, r_left[d - 1], s_left[d - 1]) {
                if d == 2 {
                    emit(
                        PPart::from_bits(t_bits | u64::from(next.sum) << PPart::shift(0)),
                        next.coefficient(),
                    );
                } else {
                    k += 1;
                    state = next;
                    continue 'enter;
                }
            }
            // Backtrack to the latest cell that can still be increased.
            loop {
                let (i, j) = (row[k] as usize, col[k] as usize);
                let v = value[k];
                let next = before[k].next(v);
                if next <= u64::from(bound[k]) {
                    let next = next as u32;
                    r_left[i] -= (next - v) << j;
                    s_left[j] -= next - v;
                    value[k] = next;
                    after = before[k]
                        .add(next)
                        .expect("Mod4::next returns only addable values");
                    continue 'advance;
                }
                r_left[i] += v << j;
                s_left[j] += v;
                if k == 0 {
                    return;
                }
                k -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::milnor_algebra::{PPartAllocation, PPartMultiplier};

    /// The polynomial degree of $P(x)$.
    fn degree(x: PPart) -> i32 {
        x.iter()
            .enumerate()
            .map(|(i, e)| ((1 << (i + 1)) - 1) * e as i32)
            .sum()
    }

    /// The terms of `P(r) P(s)` that survive mod 2, sorted, by anti-diagonal enumeration and by
    /// [`PPartMultiplier`].
    fn both(r: PPart, s: PPart) -> (Vec<PPart>, Vec<PPart>) {
        let degree = degree(r) + degree(s);
        let reduce = |mut terms: Vec<PPart>| {
            terms.sort_unstable();
            let mut odd = Vec::new();
            for chunk in terms.chunk_by(|a, b| a == b) {
                if chunk.len() % 2 == 1 {
                    odd.push(chunk[0]);
                }
            }
            odd
        };
        let mut new = Vec::new();
        for_each_term(degree, r, s, |t| new.push(t));
        let mut multiplier = PPartMultiplier::<false>::new_from_allocation(
            fp::prime::TWO,
            r,
            s,
            PPartAllocation::default(),
            0,
            degree,
        );
        let mut old = Vec::new();
        while let Some(c) = multiplier.next() {
            if c % 2 == 1 {
                old.push(multiplier.ans.p_part);
            }
        }
        (reduce(new), reduce(old))
    }

    #[test]
    fn small_products() {
        let p = PPart::from_slice;
        // Sq(1) Sq(1) = 0, Sq(1) Sq(2) = Sq(3), Sq(2) Sq(1) = Sq(3) + Sq(0, 1).
        assert_eq!(both(p(&[1]), p(&[1])).0, vec![]);
        assert_eq!(both(p(&[1]), p(&[2])).0, vec![p(&[3])]);
        let (new, old) = both(p(&[2]), p(&[1]));
        assert_eq!(new, old);
        assert_eq!(new.len(), 2);
        // Sq(4, 2) Sq(4, 1) = Sq(4, 2, 1) + Sq(7, 1, 1) + Sq(1, 3, 1).
        let mut want = vec![p(&[4, 2, 1]), p(&[7, 1, 1]), p(&[1, 3, 1])];
        want.sort_unstable();
        assert_eq!(both(p(&[4, 2]), p(&[4, 1])).0, want);
    }

    /// Every pair of p-parts of length at most 4 with small entries.
    #[test]
    fn all_small_products_match() {
        const BOUND: [u32; 4] = [12, 6, 3, 2];
        let mut parts = vec![PPart::zero()];
        for len in 1..=BOUND.len() {
            let mut e = vec![0u32; len];
            loop {
                if e[len - 1] != 0 {
                    parts.push(PPart::from_slice(&e));
                }
                let Some(i) = (0..len).find(|&i| e[i] < BOUND[i]) else {
                    break;
                };
                e[..i].fill(0);
                e[i] += 1;
            }
        }
        let mut nonzero = 0;
        for &r in &parts {
            for &s in &parts {
                let (new, old) = both(r, s);
                assert_eq!(new, old, "P{r:?} P{s:?}");
                nonzero += usize::from(!new.is_empty());
            }
        }
        assert!(nonzero > 0);
    }

    /// Long p-parts, up to the largest length a [`PPart`] can hold.
    #[test]
    fn long_products_match() {
        let p = PPart::from_slice;
        for (r, s) in [
            (p(&[60, 30, 8, 2, 1]), p(&[20, 30, 20, 4, 1, 2])),
            (p(&[35, 12, 20, 14, 1, 3]), p(&[60, 30, 0, 2, 1])),
            (p(&[3, 1, 0, 0, 0, 0, 0, 0, 0, 1]), p(&[1, 2, 1])),
            (
                p(&[0, 0, 0, 0, 0, 0, 0, 0, 1]),
                p(&[5, 0, 0, 0, 0, 0, 0, 0, 1]),
            ),
        ] {
            let (new, old) = both(r, s);
            assert_eq!(new, old, "P{r:?} P{s:?}");
            let (new, old) = both4(r, s);
            assert_eq!(new, old, "mod 4: P{r:?} P{s:?}");
        }
    }

    /// The terms of `P(r) P(s)` over Z/4 with their nonzero coefficients, sorted, by anti-diagonal
    /// enumeration and by [`PPartMultiplier`].
    fn both4(r: PPart, s: PPart) -> (Vec<(PPart, u32)>, Vec<(PPart, u32)>) {
        let degree = degree(r) + degree(s);
        let reduce = |mut terms: Vec<(PPart, u32)>| {
            terms.sort_unstable();
            terms
                .chunk_by(|a, b| a.0 == b.0)
                .map(|c| (c[0].0, c.iter().map(|t| t.1).sum::<u32>() % 4))
                .filter(|t| t.1 != 0)
                .collect::<Vec<_>>()
        };
        let mut new = Vec::new();
        for_each_term_mod4(degree, r, s, |t, c| new.push((t, c)));
        let mut multiplier = PPartMultiplier::<true>::new_from_allocation(
            fp::prime::TWO,
            r,
            s,
            PPartAllocation::default(),
            0,
            degree,
        );
        let mut old = Vec::new();
        while let Some(c) = multiplier.next() {
            old.push((multiplier.ans.p_part, c));
        }
        (reduce(new), reduce(old))
    }

    /// Every p-part of degree at most `max`.
    fn pparts_up_to(max: i32) -> Vec<PPart> {
        let mut out = vec![PPart::zero()];
        let mut stack = vec![(Vec::<u32>::new(), 0i32)];
        while let Some((e, deg)) = stack.pop() {
            let i = e.len();
            let w = (1i32 << (i + 1)) - 1;
            if w > max - deg || i == PPart::MAX_LEN {
                continue;
            }
            for v in 0..=((max - deg) / w) as u32 {
                let mut e = e.clone();
                e.push(v);
                if v != 0 {
                    out.push(PPart::from_slice(&e));
                }
                stack.push((e, deg + v as i32 * w));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    #[test]
    fn mod4_small_products() {
        let p = PPart::from_slice;
        // Sq(1) Sq(1) = 2 Sq(2) over the integral lift: (1 + 1 choose 1) = 2.
        assert_eq!(both4(p(&[1]), p(&[1])).0, vec![(p(&[2]), 2)]);
        let (new, old) = both4(p(&[4, 2]), p(&[4, 1]));
        assert_eq!(new, old);
    }

    /// Every pair of p-parts of length at most 4 with small entries, mod 4.
    #[test]
    fn mod4_all_small_products_match() {
        const BOUND: [u32; 4] = [12, 6, 3, 2];
        let mut parts = vec![PPart::zero()];
        for len in 1..=BOUND.len() {
            let mut e = vec![0u32; len];
            loop {
                if e[len - 1] != 0 {
                    parts.push(PPart::from_slice(&e));
                }
                let Some(i) = (0..len).find(|&i| e[i] < BOUND[i]) else {
                    break;
                };
                e[..i].fill(0);
                e[i] += 1;
            }
        }
        let mut twos = 0;
        for &r in &parts {
            for &s in &parts {
                let (new, old) = both4(r, s);
                assert_eq!(new, old, "P{r:?} P{s:?}");
                twos += new.iter().filter(|t| t.1 == 2).count();
            }
        }
        assert!(twos > 0);
    }

    /// Every pair of p-parts of total degree in `MOD4_BENCH_DEGREES` (default 40..=60), by both
    /// enumerators; prints the time of each. Run with `--release --ignored --nocapture`.
    #[test]
    #[ignore]
    fn mod4_bench() {
        let range = std::env::var("MOD4_BENCH_DEGREES").unwrap_or_else(|_| "40..=60".into());
        let (lo, hi) = range.split_once("..=").expect("lo..=hi");
        let (lo, hi): (i32, i32) = (lo.parse().unwrap(), hi.parse().unwrap());
        let parts = pparts_up_to(hi);
        let mut by_degree = vec![Vec::new(); hi as usize + 1];
        for &x in &parts {
            by_degree[degree(x) as usize].push(x);
        }
        let check = std::env::var_os("MOD4_BENCH_CHECK").is_some();
        for total in lo..=hi {
            let pairs: Vec<(PPart, PPart)> = (0..=total)
                .flat_map(|a| {
                    let (ra, sa) = (&by_degree[a as usize], &by_degree[(total - a) as usize]);
                    ra.iter()
                        .flat_map(move |&r| sa.iter().map(move |&s| (r, s)))
                })
                .collect();

            let t0 = std::time::Instant::now();
            let mut new_terms = 0u64;
            let mut new_sum = 0u64;
            for &(r, s) in &pairs {
                for_each_term_mod4(total, r, s, |t, c| {
                    new_terms += 1;
                    new_sum = new_sum.wrapping_add(t.bits() ^ u64::from(c));
                });
            }
            let new_s = t0.elapsed().as_secs_f64();

            let t0 = std::time::Instant::now();
            let mut old_terms = 0u64;
            let mut old_sum = 0u64;
            let mut allocation = PPartAllocation::default();
            for &(r, s) in &pairs {
                let mut m = PPartMultiplier::<true>::new_from_allocation(
                    fp::prime::TWO,
                    r,
                    s,
                    allocation,
                    0,
                    total,
                );
                while let Some(c) = m.next() {
                    if c != 0 {
                        old_terms += 1;
                        old_sum = old_sum.wrapping_add(m.ans.p_part.bits() ^ u64::from(c));
                    }
                }
                allocation = m.into_allocation();
            }
            let old_s = t0.elapsed().as_secs_f64();

            if check {
                for &(r, s) in &pairs {
                    let (new, old) = both4(r, s);
                    assert_eq!(new, old, "P{r:?} P{s:?}");
                }
            }
            println!(
                "MOD4BENCH degree={total} pairs={} terms new={new_terms} old={old_terms} \
                 new_s={new_s:.4} old_s={old_s:.4} speedup={:.2} digest_eq={}",
                pairs.len(),
                old_s / new_s,
                new_sum == old_sum,
            );
        }
    }
}
