//! Progress tracking for sweeps that complete work in dependency order.
//!
//! Some computations fill a sequence in an order dictated by a dependency relation rather than
//! left to right. A sweep over a bidegree grid, for example, may reach `(s, 7)` before `(s, 5)`,
//! because the dependencies of `(s, 7)` happened to be satisfied first. The sequence still ends up
//! gap-free; it is only *transiently* full of holes, for the duration of the sweep.
//!
//! [`Frontier`] tracks that. It records which indices are done and reports how far the gap-free
//! prefix reaches. The interesting part is what [`Frontier::complete`] returns: a [`Claim`], which
//! is not a description of the frontier but a statement of *responsibility*.

use std::{collections::BTreeSet, ops::Range};

/// Indices that a completion just made available, and that its caller alone is responsible for.
///
/// When a sweep runs in parallel, many threads complete indices concurrently, but each advance of
/// the frontier is observed by exactly one of them. The claims handed out over the life of a
/// [`Frontier`] are disjoint and cover every index the frontier passes — so if the caller that
/// receives a claim does not act on it, nothing else will.
///
/// This is why a claim is a range rather than a single index. The caller learns not just that the
/// frontier moved, but precisely which indices *it* is on the hook for, without having to compare
/// against a previous reading — which would race.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "a claim names work that no other caller will do; dropping it strands that work"]
pub enum Claim<I = usize> {
    /// The frontier did not move: this index landed in a hole above it and is now pending. Some
    /// later completion will claim it, along with the rest of its gap.
    Nothing,
    /// These indices just became part of the gap-free prefix, observed by this caller only.
    Advanced(Range<I>),
}

impl<I> Claim<I> {
    /// The claimed indices, if any.
    pub fn advanced(self) -> Option<Range<I>> {
        match self {
            Self::Nothing => None,
            Self::Advanced(range) => Some(range),
        }
    }

    pub const fn is_nothing(&self) -> bool {
        matches!(self, Self::Nothing)
    }

    /// Reinterpret the claimed indices in another coordinate system.
    ///
    /// Used by [`OnceBiVec`](crate::OnceBiVec) to shift a claim from storage indices into degrees.
    pub fn map<J>(self, f: impl Fn(I) -> J) -> Claim<J> {
        match self {
            Self::Nothing => Claim::Nothing,
            Self::Advanced(range) => Claim::Advanced(f(range.start)..f(range.end)),
        }
    }
}

/// How far a dependency-ordered sweep has progressed.
///
/// Indices may be completed in any order. The *frontier* is the exclusive upper bound of the
/// longest gap-free prefix; indices completed above it are held as *pending* until the gap below
/// them closes.
///
/// A sweep is expected to leave this [settled](Self::is_settled) — pending empty — when it
/// finishes. Holes are a property of a sweep in progress, not of the data it produces.
///
/// # Example
///
/// ```
/// # use once::{Claim, Frontier};
/// let mut f = Frontier::new();
///
/// // 2 arrives early and has to wait: the gap at 0..2 is not closed.
/// assert_eq!(f.complete(2), Claim::Nothing);
/// assert_eq!(f.get(), 0);
/// assert!(!f.is_settled());
///
/// // 0 closes nothing beyond itself.
/// assert_eq!(f.complete(0), Claim::Advanced(0..1));
///
/// // 1 closes the gap, so it claims 2 as well — including the index it did not complete.
/// assert_eq!(f.complete(1), Claim::Advanced(1..3));
/// assert_eq!(f.get(), 3);
/// assert!(f.is_settled());
/// ```
#[derive(Clone, Debug, Default)]
pub struct Frontier {
    /// Exclusive upper bound of the gap-free prefix.
    frontier: usize,
    /// Completed indices above the frontier, waiting on gaps below them.
    pending: BTreeSet<usize>,
}

impl Frontier {
    pub const fn new() -> Self {
        Self {
            frontier: 0,
            pending: BTreeSet::new(),
        }
    }

    /// A frontier that already covers `0..frontier`, for data that starts out populated.
    pub const fn settled_through(frontier: usize) -> Self {
        Self {
            frontier,
            pending: BTreeSet::new(),
        }
    }

    /// The exclusive upper bound of the gap-free prefix.
    pub const fn get(&self) -> usize {
        self.frontier
    }

    /// Whether there are no completed-but-unreachable indices above the frontier.
    pub fn is_settled(&self) -> bool {
        self.pending.is_empty()
    }

    /// Completed indices still waiting on a gap below them, in increasing order.
    pub fn pending(&self) -> impl Iterator<Item = usize> + '_ {
        self.pending.iter().copied()
    }

    /// Record `index` as complete and return the indices this caller is now responsible for.
    ///
    /// # Panics
    ///
    /// Panics if `index` is below the frontier or has already been completed. Either means two
    /// callers believe they own the same work.
    pub fn complete(&mut self, index: usize) -> Claim {
        assert!(
            index >= self.frontier,
            "index {index} is already below the frontier ({})",
            self.frontier
        );
        assert!(
            self.pending.insert(index),
            "index {index} was already completed"
        );

        if index != self.frontier {
            // Lands in a hole; whoever closes the gap below will claim it.
            return Claim::Nothing;
        }

        let start = self.frontier;
        let mut end = start;
        while self.pending.remove(&end) {
            end += 1;
        }
        self.frontier = end;
        Claim::Advanced(start..end)
    }

    /// Move the frontier to `frontier` wholesale, for bulk fills that are already in order.
    ///
    /// # Panics
    ///
    /// Panics if there are pending indices, or if this would move the frontier backwards.
    pub fn advance_to(&mut self, frontier: usize) {
        assert!(
            self.is_settled(),
            "cannot bulk-advance past pending indices: {:?}",
            self.pending
        );
        assert!(
            frontier >= self.frontier,
            "cannot move the frontier backwards, from {} to {frontier}",
            self.frontier
        );
        self.frontier = frontier;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_order_claims_one_at_a_time() {
        let mut f = Frontier::new();
        for i in 0..5 {
            assert_eq!(f.complete(i), Claim::Advanced(i..i + 1));
        }
        assert_eq!(f.get(), 5);
        assert!(f.is_settled());
    }

    #[test]
    fn gap_is_claimed_by_whoever_closes_it() {
        let mut f = Frontier::new();
        for i in [1, 2, 3] {
            assert_eq!(f.complete(i), Claim::Nothing);
        }
        assert_eq!(f.get(), 0);
        // 0 closes the gap and claims everything that was waiting on it.
        assert_eq!(f.complete(0), Claim::Advanced(0..4));
        assert!(f.is_settled());
    }

    #[test]
    fn claims_are_disjoint_and_cover_everything() {
        // Whatever order indices arrive in, the claims partition 0..n exactly once.
        for seed in 0..64usize {
            let n = 12;
            let mut order: Vec<usize> = (0..n).collect();
            // Cheap deterministic shuffle, varied by seed.
            order.sort_by_key(|i| (i * 7 + seed * 5) % n);

            let mut f = Frontier::new();
            let mut covered = vec![0usize; n];
            for i in order {
                if let Claim::Advanced(range) = f.complete(i) {
                    for j in range {
                        covered[j] += 1;
                    }
                }
            }
            assert!(f.is_settled());
            assert_eq!(f.get(), n);
            assert!(
                covered.iter().all(|&c| c == 1),
                "seed {seed}: each index must be claimed exactly once, got {covered:?}"
            );
        }
    }

    #[test]
    fn pending_lists_unreachable_indices() {
        let mut f = Frontier::new();
        let _ = f.complete(4);
        let _ = f.complete(2);
        assert_eq!(f.pending().collect::<Vec<_>>(), vec![2, 4]);
        assert!(!f.is_settled());
    }

    #[test]
    #[should_panic(expected = "already completed")]
    fn double_completion_panics() {
        let mut f = Frontier::new();
        let _ = f.complete(3);
        let _ = f.complete(3);
    }

    #[test]
    #[should_panic(expected = "already below the frontier")]
    fn completing_settled_index_panics() {
        let mut f = Frontier::new();
        let _ = f.complete(0);
        let _ = f.complete(0);
    }

    #[test]
    fn settled_through_starts_populated() {
        let mut f = Frontier::settled_through(3);
        assert_eq!(f.get(), 3);
        assert_eq!(f.complete(3), Claim::Advanced(3..4));
    }

    #[test]
    fn claim_maps_coordinates() {
        let c: Claim<usize> = Claim::Advanced(2..5);
        assert_eq!(c.map(|i| i as i32 - 3), Claim::Advanced(-1..2));
        assert_eq!(Claim::<usize>::Nothing.map(|i| i as i32), Claim::Nothing);
    }
}
