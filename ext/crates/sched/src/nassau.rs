//! Helpers for driving a Nassau resolution (bidegrees `(s, t)`) with this crate.

/// The group id of bidegree `(s, t)` (homological degree `s`, internal degree `t`), for
/// [`JobSpec::group`](crate::JobSpec::group) with [`GroupOrder::Id`](crate::GroupOrder::Id):
/// bidegrees ordered by `s`, then by `t`. It depends only on the bidegree, so the order survives
/// a coordinator restart, unlike [`GroupOrder::Arrival`](crate::GroupOrder::Arrival).
///
/// `(s, t)` is the restart-stable key closest to arrival order: in the whole-run simulation it
/// matches arrival's makespan with a tenth of its bidegree latency at the 90th percentile, where
/// `(t, s)` and `(t - s, s)` are 5-8% slower (see RESULTS.md).
pub fn group(s: u32, t: u32) -> u64 {
    (u64::from(s) << 32) | u64::from(t)
}

/// The bidegree of a [`group`] id: `(s, t)`.
pub fn bidegree(group: u64) -> (u32, u32) {
    ((group >> 32) as u32, group as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids order by `s`, then `t`, and round-trip.
    #[test]
    fn order_and_round_trip() {
        assert!(group(0, 400) < group(1, 1));
        assert!(group(3, 10) < group(3, 11));
        assert_eq!(bidegree(group(7, 300)), (7, 300));
    }
}
