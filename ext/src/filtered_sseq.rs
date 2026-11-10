//! The spectral sequence of a filtered cochain complex over $\mathbb{F}_2$, read off persistence pairs.

use std::collections::BTreeMap;

/// One coboundary $\delta \colon C^a \to C^{a+1}$ of a complex with an increasing filtration
/// $F_j$ by subcomplexes, with every basis element in a single $F_j$.
#[derive(Debug, Clone, Default)]
pub struct FilteredMap {
    /// The filtration of each source basis element.
    pub source: Vec<i32>,
    /// The filtration of each target basis element.
    pub target: Vec<i32>,
    /// The target indices of the nonzero entries of $\delta$ of each source basis element.
    pub columns: Vec<Vec<u32>>,
}

/// A pair cancelled by $d_r$ of the spectral sequence of a [`FilteredMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pair {
    /// The source basis element whose class supports the differential.
    pub source: usize,
    /// The target basis element whose class it hits.
    pub target: usize,
    /// The page: the filtration of the source minus that of the target.
    pub r: i32,
}

/// The symmetric difference of two sorted lists.
fn xor(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::with_capacity(a.len() + b.len());
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

impl FilteredMap {
    /// The persistence pairs of the sources in `keep`: the differentials of the spectral
    /// sequence, one per pair, with $d_0$ the pairs of gap zero.
    ///
    /// Sources are reduced in increasing filtration, each against the earlier ones, and paired
    /// with the target of highest filtration left in it. Only earlier sources are added, so
    /// each reduced source stays in its filtration, and the pairing is that of the spectral
    /// sequence whatever order breaks ties.
    pub fn pairs(&self, keep: impl Fn(usize) -> bool) -> Vec<Pair> {
        // Targets renumbered so that the highest filtration comes first.
        let mut order: Vec<u32> = (0..self.target.len() as u32).collect();
        order.sort_by_key(|&t| (std::cmp::Reverse(self.target[t as usize]), t));
        let mut rank = vec![0u32; order.len()];
        for (new, &old) in order.iter().enumerate() {
            rank[old as usize] = new as u32;
        }
        let mut sources: Vec<usize> = (0..self.source.len()).filter(|&s| keep(s)).collect();
        sources.sort_by_key(|&s| (self.source[s], s));

        let mut reduced: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        let mut pairs = Vec::new();
        for s in sources {
            let mut col: Vec<u32> = self.columns[s].iter().map(|&t| rank[t as usize]).collect();
            col.sort_unstable();
            while let Some(&low) = col.first() {
                match reduced.get(&low) {
                    Some(prev) => col = xor(&col, prev),
                    None => break,
                }
            }
            if let Some(&low) = col.first() {
                let target = order[low as usize] as usize;
                pairs.push(Pair {
                    source: s,
                    target,
                    r: self.source[s] - self.target[target],
                });
                reduced.insert(low, col);
            }
        }
        pairs
    }
}

/// The pages of a spectral sequence, as the dimension of each degree on $E_r$ for every $r$
/// at which a differential occurs, from the persistence pairs into and out of each degree.
#[derive(Debug, Clone, Default)]
pub struct Pages<D: Ord + Copy> {
    /// The dimension of the complex in each degree.
    pub chains: BTreeMap<D, usize>,
    /// The pages of the differentials out of each degree.
    pub out_of: BTreeMap<D, Vec<i32>>,
    /// The pages of the differentials into each degree.
    pub into: BTreeMap<D, Vec<i32>>,
}

impl<D: Ord + Copy> Pages<D> {
    /// Records the chains of degree `d` and the pairs of the coboundary from `d` to `next`.
    pub fn add(&mut self, d: D, dim: usize, next: D, pairs: &[Pair]) {
        self.chains.insert(d, dim);
        for p in pairs {
            self.out_of.entry(d).or_default().push(p.r);
            self.into.entry(next).or_default().push(p.r);
        }
    }

    /// The dimension of $E_r$ in degree `d`.
    pub fn dim(&self, d: D, r: i32) -> usize {
        let cancelled = |m: &BTreeMap<D, Vec<i32>>| {
            m.get(&d)
                .map_or(0, |rs| rs.iter().filter(|&&q| q < r).count())
        };
        self.chains.get(&d).copied().unwrap_or(0) - cancelled(&self.out_of) - cancelled(&self.into)
    }

    /// The number of $d_r$ out of degree `d`, by `r`.
    pub fn differentials(&self, d: D) -> BTreeMap<i32, usize> {
        let mut out = BTreeMap::new();
        for &r in self.out_of.get(&d).into_iter().flatten() {
            *out.entry(r).or_insert(0) += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// $\mathbb{F}_2[x]/x^4$ filtered by powers of $x$, with $\delta(x^j) = x^{j-r}$ on the
    /// even ones: a $d_r$ for each pair.
    #[test]
    fn gap_is_page() {
        for r in 0..3 {
            let map = FilteredMap {
                source: vec![r, r + 2],
                target: vec![0, 2],
                columns: vec![vec![0], vec![0, 1]],
            };
            let pairs = map.pairs(|_| true);
            let mut gaps: Vec<i32> = pairs.iter().map(|p| p.r).collect();
            gaps.sort();
            assert_eq!(gaps, vec![r, r]);
        }
    }

    /// Reducing against an earlier source lowers the page of a later one.
    #[test]
    fn reduction_lowers_page() {
        // δ(a) = t0 + t1, δ(b) = t1 with a in filtration 1, b in 2, t_j in j: a pairs with t1
        // (d_0), then b reduces to t1 + (t0 + t1) = t0 and pairs with it (d_2).
        let map = FilteredMap {
            source: vec![1, 2],
            target: vec![0, 1],
            columns: vec![vec![0, 1], vec![1]],
        };
        let pairs = map.pairs(|_| true);
        assert_eq!(
            pairs,
            vec![
                Pair {
                    source: 0,
                    target: 1,
                    r: 0
                },
                Pair {
                    source: 1,
                    target: 0,
                    r: 2
                }
            ]
        );
    }
}
