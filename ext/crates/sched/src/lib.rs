//! Pure, deterministic, resource-aware job placement.
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod admission;
mod config;
mod dag;
pub mod log;
pub use log::EventSink;
pub mod nassau;
mod scheduler;
mod shared;
#[cfg(feature = "sim")]
pub mod sim;
mod speed;

pub use admission::{Admission, ProductionAdmission, WorkerView};
pub use config::{
    Config, DEFAULT_AGE_LIMIT, Defer, Fit, GroupOrder, Order, Reservations, SpeedConfig,
    SpeedPolicy, Spoliation,
};
#[cfg(feature = "serde")]
pub use dag::DagSnapshot;
pub use dag::{
    Dag, DagConfig, DagError, DagJob, DagScheduler, DagStats, DagTemplate, InstanceSpec, NodeLabel,
};
pub use scheduler::Scheduler;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use shared::{Attempt, FailKind, FailOutcome, Lease, Placement, RetryConfig, SharedPolicy};
pub use speed::{Learn, Sharing, SpeedEstimator};

/// A job identifier, chosen by the caller. Must be unique among live (waiting or running) jobs.
pub type JobId = u64;

/// A worker identifier. The caller maps its own ids (e.g. `"host:port"`) to these.
pub type WorkerId = u64;

/// A point in time, in seconds, on the caller's clock. Only differences and ordering are used; the
/// caller should pass non-decreasing values.
pub type Instant = f64;

/// Number of resource dimensions in a [`Resources`] vector.
pub const DIMS: usize = 2;
/// The host-memory dimension of a [`Resources`] vector, in bytes.
pub const MEM: usize = 0;
/// The device-memory dimension of a [`Resources`] vector, in bytes.
pub const DEV: usize = 1;

/// An additive resource vector, one component per dimension ([`MEM`], [`DEV`]).
///
/// Comparisons between vectors are component-wise ([`Resources::fits_within`]). As a capacity
/// ([`WorkerState::budget`]), a zero component means that dimension's capacity is unknown and is
/// not enforced (see [`ProductionAdmission`]); as a demand, a zero component means none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Resources(pub [u64; DIMS]);

/// Bytes in a gigabyte (10^9), rounding to the nearest byte.
fn gb_bytes(gb: f64) -> u64 {
    (gb.max(0.0) * 1e9).round() as u64
}

impl Resources {
    /// The largest representable vector (an "unbounded" capacity).
    pub const MAX: Self = Self([u64::MAX; DIMS]);
    /// No resources.
    pub const ZERO: Self = Self([0; DIMS]);

    /// A vector with `bytes` of host memory and nothing else.
    pub const fn mem(bytes: u64) -> Self {
        let mut r = Self::ZERO;
        r.0[MEM] = bytes;
        r
    }

    /// A vector with `gb` gigabytes (10^9 bytes) of host memory, rounded to the nearest byte.
    pub fn mem_gb(gb: f64) -> Self {
        Self::mem(gb_bytes(gb))
    }

    /// This vector with `bytes` of device memory.
    pub const fn with_dev(mut self, bytes: u64) -> Self {
        self.0[DEV] = bytes;
        self
    }

    /// This vector with `gb` gigabytes of device memory.
    pub fn with_dev_gb(self, gb: f64) -> Self {
        self.with_dev(gb_bytes(gb))
    }

    /// Whether every component of `self` is at most the matching component of `cap`.
    pub fn fits_within(&self, cap: &Self) -> bool {
        (0..DIMS).all(|d| self[d] <= cap[d])
    }

    /// The vector `f(self[d], other[d])` for every dimension `d`.
    fn zip(self, other: Self, f: impl Fn(u64, u64) -> u64) -> Self {
        Self(std::array::from_fn(|d| f(self[d], other[d])))
    }

    /// Component-wise maximum.
    pub fn max(self, other: Self) -> Self {
        self.zip(other, u64::max)
    }

    /// Component-wise saturating addition.
    pub fn saturating_add(self, other: Self) -> Self {
        self.zip(other, u64::saturating_add)
    }

    /// Component-wise saturating subtraction.
    pub fn saturating_sub(self, other: Self) -> Self {
        self.zip(other, u64::saturating_sub)
    }

    /// Every component multiplied by `n`, saturating.
    pub fn saturating_mul(self, n: u64) -> Self {
        Self(self.0.map(|x| x.saturating_mul(n)))
    }
}

impl std::ops::Index<usize> for Resources {
    type Output = u64;

    /// The component of dimension `d` ([`MEM`], [`DEV`]).
    fn index(&self, d: usize) -> &u64 {
        &self.0[d]
    }
}

impl std::ops::IndexMut<usize> for Resources {
    /// The component of dimension `d` ([`MEM`], [`DEV`]).
    fn index_mut(&mut self, d: usize) -> &mut u64 {
        &mut self.0[d]
    }
}

impl std::ops::Add for Resources {
    type Output = Self;

    /// Saturating, like [`Resources::saturating_add`].
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl std::ops::AddAssign for Resources {
    /// Saturating, like [`Resources::saturating_add`].
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::Sub for Resources {
    type Output = Self;

    /// Saturating: bookkeeping never goes below zero.
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
}

impl std::ops::SubAssign for Resources {
    /// Saturating, like [`Resources::saturating_sub`].
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

/// A job, as submitted to a [`Policy`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JobSpec {
    /// The job's id.
    pub id: JobId,
    /// What the job is expected to use while running (an estimate; it may be pessimistic).
    pub demand: Resources,
    /// Priority group, e.g. the bidegree a job belongs to. Groups are ordered by first arrival: the
    /// default priority is "oldest group first, then FIFO within a group".
    pub group: u64,
    /// Explicit priority overriding group order; smaller is more urgent. Jobs without one count as
    /// [`Order::Priority`]'s `default_priority` (0 by default), so negative values jump ahead of
    /// unprioritised jobs and positive values fall behind them.
    pub priority: Option<i64>,
    /// Soft placement preference (cache affinity): workers to try first. Never required.
    pub prefer: Vec<WorkerId>,
    /// Workers the job must not run on (e.g. ones it already failed on). Hard unless
    /// [`avoid_soft`](Self::avoid_soft): a job that avoids every live worker it could run on
    /// waits until one it does not avoid joins.
    pub avoid: Vec<WorkerId>,
    /// Make `avoid` soft: avoided workers become eligible while no other live worker (one with
    /// slots, of the job's class) exists. "Live", not "free": a retry still waits for a busy
    /// healthy worker rather than returning to the one it failed on.
    #[cfg_attr(feature = "serde", serde(default))]
    pub avoid_soft: bool,
    /// If set, the job runs only on workers of this class (a hard constraint).
    pub class: Option<String>,
    /// Estimated work, in seconds on a worker of [`WorkerState::speed`] 1.0. Used by
    /// [`SpeedPolicy::EarliestFinish`] (and filled in from the DAG layer's estimate when unset).
    #[cfg_attr(feature = "serde", serde(default))]
    pub work: Option<f64>,
}

impl JobSpec {
    /// A job with the given id, demand and group, and no priority, preference, avoid list or class.
    pub fn new(id: JobId, demand: Resources, group: u64) -> Self {
        Self {
            id,
            demand,
            group,
            priority: None,
            prefer: Vec::new(),
            avoid: Vec::new(),
            avoid_soft: false,
            class: None,
            work: None,
        }
    }
}

/// A worker's declared capacity and last reported usage.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkerState {
    /// The worker's id.
    pub id: WorkerId,
    /// The worker's class (e.g. GPU type), used by class pins ([`JobSpec::class`]) and per-class
    /// reservations.
    pub class: String,
    /// Maximum number of concurrent jobs.
    pub slots: usize,
    /// Capacity: host memory, and device memory (the pool jobs' device allocations come from). A
    /// zero component is unknown and not enforced.
    pub budget: Resources,
    /// What one job of this worker is expected to take at least, learned by the worker (e.g. the
    /// typical device launch request); zero components are unknown. Each job counts for at least
    /// this much against `budget` in every dimension, whatever its own [`JobSpec::demand`] says.
    #[cfg_attr(feature = "serde", serde(default))]
    pub per_task: Resources,
    /// Last reported resident usage (from a heartbeat; may lag by seconds).
    pub reported_used: Resources,
    /// The part of `reported_used` not attributable to jobs (caches, runtime).
    pub reported_baseline: Resources,
    /// How fast a job runs here, relative to a reference worker (1.0): a job with
    /// [`JobSpec::work`] `w` takes `w / speed` seconds. Used by speed-aware placement
    /// ([`SpeedPolicy`]); ignored otherwise. Default 1.0.
    #[cfg_attr(feature = "serde", serde(default = "unit_speed"))]
    pub speed: f64,
}

/// The default [`WorkerState::speed`].
#[cfg(feature = "serde")]
fn unit_speed() -> f64 {
    1.0
}

impl WorkerState {
    /// A worker with the given capacity and nothing reported yet.
    pub fn new(id: WorkerId, class: impl Into<String>, slots: usize, budget: Resources) -> Self {
        Self {
            id,
            class: class.into(),
            slots,
            budget,
            reported_used: Resources::ZERO,
            reported_baseline: Resources::ZERO,
            speed: 1.0,
            per_task: Resources::ZERO,
        }
    }
}

/// A reservation: `worker` admits no job other than `job` until `job` is placed.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservationInfo {
    /// The job holding the reservation.
    pub job: JobId,
    /// The reserved worker.
    pub worker: WorkerId,
    /// When the reservation was made.
    pub since: Instant,
}

/// The library's view of one worker's load.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerLoad {
    /// The worker.
    pub id: WorkerId,
    /// Its class.
    pub class: String,
    /// Its slot count.
    pub slots: usize,
    /// Jobs placed on it and not yet completed.
    pub running: usize,
    /// Sum of the demands of those jobs.
    pub placed: Resources,
    /// Headroom per dimension as admission sees it ([`WorkerView::headroom`]); `None` where the
    /// capacity is unknown.
    pub headroom: [Option<i64>; DIMS],
    /// The job this worker is reserved for, if any.
    pub reserved_for: Option<JobId>,
    /// Its speed.
    pub speed: f64,
}

/// A snapshot of a policy's state, for logs and metrics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PolicyStats {
    /// The latest `now` seen in any event.
    pub now: Instant,
    /// Number of waiting jobs.
    pub waiting: usize,
    /// Number of running (placed, not completed) jobs.
    pub running: usize,
    /// The waiting job that has waited longest, and for how long (seconds).
    pub longest_wait: Option<(JobId, f64)>,
    /// Current reservations.
    pub reservations: Vec<ReservationInfo>,
    /// Total placements made since creation.
    pub placements_total: u64,
    /// Total reservations made since creation.
    pub reservations_total: u64,
    /// Per-worker load, ordered by worker id.
    pub workers: Vec<WorkerLoad>,
    /// Jobs the last `dispatch` placed on the worker they had reserved (a reservation paying
    /// off), in placement order.
    pub last_dispatch_holders: Vec<JobId>,
    /// Jobs the last `dispatch` deliberately left waiting, at some point of its scan, for a faster
    /// worker that was busy ([`Defer`]), and did not place afterwards: `(job, worker it waits
    /// for, expected start there)`. Less urgent jobs may have taken slower workers meanwhile.
    pub deferred: Vec<(JobId, WorkerId, Instant)>,
    /// Every job that deferred at some point of the last `dispatch`'s scan, including those placed
    /// later in it (after a released reservation restarted the scan): while deferring, a job
    /// leaves the slower workers it declined to less urgent jobs.
    pub deferred_any: Vec<JobId>,
}

/// A placement policy, driven by events.
///
/// Every method is deterministic: the same sequence of calls (with the same `now`s) produces the
/// same placements. The caller calls [`Policy::dispatch`] after every event (submission,
/// completion, heartbeat) and sends each returned job to its worker, treating it as running.
pub trait Policy {
    /// A job became ready. Submitting an id that is already waiting or running is ignored.
    fn submit(&mut self, job: JobSpec, now: Instant);
    /// The job is no longer wanted. A waiting job is dropped; a running job's resources are
    /// released as if it completed. Unknown ids are ignored.
    fn cancel(&mut self, job: JobId);
    /// A worker joined, or reported a heartbeat. Its running jobs and placed demand are kept.
    fn worker_update(&mut self, w: WorkerState, now: Instant);
    /// A worker left. Its running jobs are forgotten; the caller resubmits them if they should run.
    fn worker_gone(&mut self, w: WorkerId, now: Instant);
    /// A running job finished; its resources are released (and its duration may teach the
    /// policy its worker's speed).
    fn completed(&mut self, job: JobId, now: Instant);
    /// A running job failed: its resources are released, nothing is learned from its duration.
    /// The default cancels it; `why` is for logs ([`log::Logged`]).
    fn failed(&mut self, job: JobId, now: Instant, why: &str) {
        let _ = (now, why);
        self.cancel(job);
    }
    /// The placements to make now.
    fn dispatch(&mut self, now: Instant) -> Vec<(JobId, WorkerId)>;
    /// Why a waiting job is not placed, in words (for logs). `None` for unknown jobs.
    fn explain(&self, job: JobId) -> Option<String>;
    /// Counters and current state.
    fn stats(&self) -> PolicyStats;
    /// [`dispatch`](Self::dispatch) plus preemptions ([`Spoliation`]): running jobs to restart
    /// on a faster worker. The caller kills each preempted job's instance on `from` and starts it
    /// again on `to`; the policy already counts it on `to` only. If the old instance completes
    /// before the kill lands, report `completed(job)` (that releases `to`) and kill the new one.
    /// The default never preempts.
    fn dispatch_full(&mut self, now: Instant) -> Dispatch {
        Dispatch {
            start: self.dispatch(now),
            preempt: Vec::new(),
        }
    }
    /// The next time `dispatch` should be called even if no event arrives: a job's voluntary wait
    /// for a faster worker ([`Defer`]) expires then. `None` if nothing is timed. Callers with
    /// frequent events may ignore it at the cost of that much extra waiting.
    fn next_wakeup(&self) -> Option<Instant> {
        None
    }
}

impl<P: Policy + ?Sized> Policy for Box<P> {
    /// Forwarded to the boxed policy.
    fn submit(&mut self, job: JobSpec, now: Instant) {
        (**self).submit(job, now)
    }

    /// Forwarded to the boxed policy.
    fn cancel(&mut self, job: JobId) {
        (**self).cancel(job)
    }

    /// Forwarded to the boxed policy.
    fn worker_update(&mut self, w: WorkerState, now: Instant) {
        (**self).worker_update(w, now)
    }

    /// Forwarded to the boxed policy.
    fn worker_gone(&mut self, w: WorkerId, now: Instant) {
        (**self).worker_gone(w, now)
    }

    /// Forwarded to the boxed policy.
    fn completed(&mut self, job: JobId, now: Instant) {
        (**self).completed(job, now)
    }

    /// Forwarded to the boxed policy.
    fn failed(&mut self, job: JobId, now: Instant, why: &str) {
        (**self).failed(job, now, why)
    }

    /// Forwarded to the boxed policy.
    fn dispatch(&mut self, now: Instant) -> Vec<(JobId, WorkerId)> {
        (**self).dispatch(now)
    }

    /// Forwarded to the boxed policy.
    fn explain(&self, job: JobId) -> Option<String> {
        (**self).explain(job)
    }

    /// Forwarded to the boxed policy.
    fn stats(&self) -> PolicyStats {
        (**self).stats()
    }

    /// Forwarded.
    fn next_wakeup(&self) -> Option<Instant> {
        (**self).next_wakeup()
    }

    /// Forwarded.
    fn dispatch_full(&mut self, now: Instant) -> Dispatch {
        (**self).dispatch_full(now)
    }
}

/// What [`Policy::dispatch_full`] decided: placements of waiting jobs, and preemptions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dispatch {
    /// Waiting jobs to start: `(job, worker)`.
    pub start: Vec<(JobId, WorkerId)>,
    /// Running jobs to restart elsewhere.
    pub preempt: Vec<Preemption>,
}

/// A running job moved to a faster worker (see [`Policy::dispatch_full`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preemption {
    /// The job.
    pub job: JobId,
    /// Where it was running (kill it there).
    pub from: WorkerId,
    /// Where it runs now (start it again there).
    pub to: WorkerId,
}
