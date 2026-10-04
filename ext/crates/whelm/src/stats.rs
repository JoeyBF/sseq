//! What a policy reports about its state.

use std::time::Duration;

#[cfg(doc)]
use crate::{Defer, Policy, Reservations, Timing, WorkerView};
use crate::{JobId, Resources, Time, WorkerId};

/// A reservation: `worker` admits no job other than `job` until `job` is placed.
///
/// Listed by [`PolicyStats::reservations`]; the crate's [Time](crate#time) chapter shows one made
/// and paid off.
///
/// # Examples
///
/// A job that fits nowhere reserves a worker once it has waited
/// [`reserve_after`](Reservations::reserve_after).
///
/// ```
/// use whelm::{
///     Config, Input, JobSpec, Policy, ReservationInfo, Reservations, Resources, Scheduler, Time,
///     WorkerState,
/// };
///
/// let mut p = Scheduler::new(Config::default());
/// p.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         capacity: Resources::mem_gb(10.0).with_slots(2),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// p.handle(
///     Input::Submit(JobSpec {
///         id: 1,
///         demand: Resources::mem_gb(6.0),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// p.handle(
///     Input::Submit(JobSpec {
///         id: 2,
///         demand: Resources::mem_gb(6.0),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// p.poll(Time::ORIGIN);
/// let t = Time::ORIGIN + Reservations::default().reserve_after;
/// p.poll(t);
/// assert_eq!(
///     p.stats().reservations,
///     [ReservationInfo {
///         job: 2,
///         worker: 1,
///         since: t
///     }]
/// );
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ReservationInfo {
    /// The job holding the reservation.
    pub job: JobId,
    /// The reserved worker.
    pub worker: WorkerId,
    /// When the reservation was made.
    pub since: Time,
}

/// The library's view of one worker's load.
///
/// # Examples
///
/// ```
/// use whelm::{Config, Input, JobSpec, Policy, Resources, Scheduler, Time, WorkerState};
///
/// let mut p = Scheduler::new(Config::default());
/// p.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         capacity: Resources::mem_gb(10.0).with_slots(4),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// p.handle(
///     Input::Submit(JobSpec {
///         id: 1,
///         demand: Resources::mem_gb(3.0),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// p.poll(Time::ORIGIN);
/// let load = &p.stats().workers[0];
/// assert_eq!((load.id, load.running, load.reserved_for), (1, 1, None));
/// assert_eq!(load.headroom, [Some(7_000_000_000), None, Some(3)]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerLoad {
    /// The worker.
    pub id: WorkerId,
    /// Its class.
    pub class: String,
    /// Its capacity.
    pub capacity: Resources,
    /// Live attempts on it (a job speculated onto it counts here and on its other worker).
    pub running: usize,
    /// Sum of the demands of those attempts.
    pub placed: Resources,
    /// Headroom per declared resource as admission sees it ([`WorkerView::headroom`]); `None`
    /// where the capacity is unknown.
    pub headroom: Vec<Option<i64>>,
    /// The job this worker is reserved for, if any.
    pub reserved_for: Option<JobId>,
    /// Its speed for a job of no particular kind ([`Timing`]); under [`Timing::Unrelated`], a
    /// kind's speed is this times the kind's factor on the worker's class.
    pub speed: f64,
}

/// A snapshot of a policy's state, for logs and metrics.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{Config, Input, JobSpec, Policy, Resources, Scheduler, Time, WorkerState};
///
/// let mut p = Scheduler::new(Config::default());
/// p.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         capacity: Resources::ZERO.with_slots(1),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// for id in 1..=3 {
///     p.handle(
///         Input::Submit(JobSpec {
///             id,
///             ..Default::default()
///         }),
///         Time::ORIGIN,
///     );
/// }
/// p.poll(Time::ORIGIN);
/// p.poll(Time(Duration::from_secs(25)));
/// let stats = p.stats();
/// assert_eq!(
///     (stats.now, stats.waiting, stats.running),
///     (Time(Duration::from_secs(25)), 2, 1)
/// );
/// assert_eq!(stats.longest_wait, Some((2, Duration::from_secs(25))));
/// assert_eq!(stats.placements_total, 1);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PolicyStats {
    /// The latest `now` seen in any event.
    pub now: Time,
    /// Number of waiting jobs.
    pub waiting: usize,
    /// Number of running jobs (with at least one live attempt), each counted once.
    pub running: usize,
    /// The waiting job that has waited longest, and for how long.
    pub longest_wait: Option<(JobId, Duration)>,
    /// Current reservations.
    pub reservations: Vec<ReservationInfo>,
    /// Total attempts started since creation (retries and speculative attempts included).
    pub placements_total: u64,
    /// Total reservations made since creation.
    pub reservations_total: u64,
    /// Per-worker load, ordered by worker id.
    pub workers: Vec<WorkerLoad>,
    /// Jobs the last [`Policy::poll`] placed on the worker they had reserved (a reservation paying
    /// off), in placement order.
    pub last_dispatch_holders: Vec<JobId>,
    /// Jobs the last [`Policy::poll`] deliberately left waiting, at some point of its scan, for a
    /// faster worker that was busy ([`Defer`]), and did not place afterwards: `(job, worker it
    /// waits for, expected start there)`. Less urgent jobs may have taken slower workers
    /// meanwhile.
    pub deferred: Vec<(JobId, WorkerId, Time)>,
    /// Every job that deferred at some point of the last [`Policy::poll`]'s scan, including those
    /// placed later in it (after a released reservation restarted the scan): while deferring, a
    /// job leaves the slower workers it declined to less urgent jobs.
    pub deferred_any: Vec<JobId>,
}
