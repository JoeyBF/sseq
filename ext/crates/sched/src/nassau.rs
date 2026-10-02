//! Helpers for driving a Nassau resolution (bidegrees `(s, t)`) with this crate.

/// The group id of bidegree `(s, t)` (homological degree `s`, internal degree `t`), for
/// [`JobSpec::group`](crate::JobSpec::group) with [`GroupOrder::Id`](crate::GroupOrder::Id):
/// bidegrees ordered by internal degree `t`, then by `s`. It depends only on the bidegree, so the
/// order survives a coordinator restart, unlike [`GroupOrder::Arrival`](crate::GroupOrder::Arrival).
pub fn group(s: u32, t: u32) -> u64 {
    (u64::from(t) << 32) | u64::from(s)
}

/// The bidegree of a [`group`] id: `(s, t)`.
pub fn bidegree(group: u64) -> (u32, u32) {
    (group as u32, (group >> 32) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids order by `t`, then `s`, and round-trip.
    #[test]
    fn order_and_round_trip() {
        assert!(group(5, 10) < group(0, 11));
        assert!(group(3, 10) < group(4, 10));
        assert_eq!(bidegree(group(7, 300)), (7, 300));
    }
}
