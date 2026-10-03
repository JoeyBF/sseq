//! Helpers for driving a Nassau resolution (bidegrees `(s, t)`) with this crate.
//!
//! A resolution's work comes in bidegrees, and a bidegree's tasks are best kept together: make
//! each bidegree a [group](crate::JobSpec::group) with [`group`], order groups by id with
//! [`GroupOrder::Id`](crate::GroupOrder::Id), and read a group back with [`bidegree`].
//!
//! The two jobs below are submitted out of order, but the one in the lower homological degree
//! starts first: group ids order by `s`, then by `t`.
//!
//! ```
//! use whelm::{
//!     Config, GroupOrder, Input, JobSpec, Output, Policy, Resources, Scheduler, Time,
//!     WorkerState,
//!     nassau::{bidegree, group},
//! };
//!
//! let mut config = Config::default();
//! config.group_order = GroupOrder::Id;
//! let mut p = Scheduler::new(config);
//! let submit = |id, g| {
//!     Input::Submit(JobSpec {
//!         id,
//!         demand: Resources::mem_gb(1.0),
//!         group: g,
//!         ..Default::default()
//!     })
//! };
//! p.handle(submit(1, group(3, 20)), Time::ORIGIN);
//! p.handle(submit(2, group(2, 40)), Time::ORIGIN);
//! let worker = WorkerState {
//!     id: 1,
//!     budget: Resources::mem_gb(8.0),
//!     ..Default::default()
//! };
//! p.handle(Input::Worker(worker), Time::ORIGIN);
//! let start = Output::Start {
//!     job: 2,
//!     attempt: 1,
//!     worker: 1,
//! };
//! assert_eq!(p.poll(Time::ORIGIN), [start]);
//! assert_eq!(bidegree(group(2, 40)), (2, 40));
//! ```

/// The group id of bidegree `(s, t)` (homological degree `s`, internal degree `t`), for
/// [`JobSpec::group`](crate::JobSpec::group) with [`GroupOrder::Id`](crate::GroupOrder::Id):
/// bidegrees ordered by `s`, then by `t`. It depends only on the bidegree, so the order survives
/// a coordinator restart, unlike [`GroupOrder::Arrival`](crate::GroupOrder::Arrival).
///
/// Every bidegree in a lower homological degree sorts first, however large its `t`:
///
/// ```
/// use whelm::nassau::group;
///
/// assert!(group(0, 400) < group(1, 1));
/// assert!(group(3, 10) < group(3, 11));
/// ```
///
/// Of the restart-stable keys, `(s, t)` comes closest to arrival order's makespan, and `(t, s)`
/// and `(t - s, s)` are slower: see `whelm-sim`'s RESULTS.md, "Restart-stable order".
pub fn group(s: u32, t: u32) -> u64 {
    (u64::from(s) << 32) | u64::from(t)
}

/// The bidegree of a [`group`] id: `(s, t)`.
///
/// It inverts [`group`], so a group id (e.g. one quoted by [`explain`](crate::Policy::explain))
/// reads back as a bidegree:
///
/// ```
/// use whelm::nassau::{bidegree, group};
///
/// assert_eq!(bidegree(group(7, 300)), (7, 300));
/// assert_eq!(bidegree(group(0, u32::MAX)), (0, u32::MAX));
/// ```
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
