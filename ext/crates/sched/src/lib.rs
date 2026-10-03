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
mod speed;

pub use admission::{Admission, ProductionAdmission, WorkerView};
pub use config::{
    Config, DEFAULT_AGE_LIMIT, Defer, GroupOrder, OrderTerm, Reservations, RetryConfig, ScoreTerm,
    Speculate, SpeedConfig,
};
#[cfg(feature = "serde")]
pub use dag::DagSnapshot;
pub use dag::{
    DagConfig, DagError, DagJob, DagScheduler, DagStats, DagTemplate, NodeSource, TemplateNode,
    Unit,
};
pub use scheduler::Scheduler;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use shared::{Lease, SharedPolicy};
pub use speed::{Learn, Sharing, SpeedEstimator};

/// A job identifier, chosen by the caller. Must be unique among live (waiting or running) jobs.
pub type JobId = u64;

/// A worker identifier. The caller maps its own ids (e.g. `"host:port"`) to these.
pub type WorkerId = u64;

/// A point in time, in seconds, on the caller's clock. Only differences and ordering are used; the
/// caller should pass non-decreasing values.
pub type Instant = f64;

/// Number of resource dimensions in a [`Resources`] vector.
pub const DIMS: usize = 3;
/// The host-memory dimension of a [`Resources`] vector, in bytes.
pub const MEM: usize = 0;
/// The device-memory dimension of a [`Resources`] vector, in bytes.
pub const DEV: usize = 1;
/// The execution-slot dimension of a [`Resources`] vector. Every job demands one slot: the
/// scheduler sets that component of [`JobSpec::demand`] at submission, whatever the caller wrote.
pub const SLOTS: usize = 2;

/// Which dimensions are hard. A hard dimension is always enforced, a zero capacity included, and
/// has no escape hatch; a soft one is enforced only where its capacity is known (nonzero), and a
/// job alone on a worker ignores it (see [`ProductionAdmission`]). Slots are counted exactly, so
/// they are hard; memory figures are estimates, so they are soft.
pub const HARD: [bool; DIMS] = {
    let mut hard = [false; DIMS];
    hard[SLOTS] = true;
    hard
};

/// An additive resource vector, one component per dimension ([`MEM`], [`DEV`], [`SLOTS`]).
///
/// Comparisons between vectors are component-wise ([`Resources::fits_within`]). As a capacity
/// ([`WorkerState::budget`]), a zero component of a soft dimension means its capacity is unknown
/// and not enforced, and a zero component of a [`HARD`] one means none (see
/// [`ProductionAdmission`]); as a demand, a zero component means none.
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

    /// The component of dimension `d` ([`MEM`], [`DEV`], [`SLOTS`]).
    fn index(&self, d: usize) -> &u64 {
        &self.0[d]
    }
}

impl std::ops::IndexMut<usize> for Resources {
    /// The component of dimension `d` ([`MEM`], [`DEV`], [`SLOTS`]).
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

/// What a [`Constraint`] is about: one worker, or every worker of a class.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Selector {
    /// The worker with this id.
    Worker(WorkerId),
    /// The workers of this class ([`WorkerState::class`]).
    Class(String),
}

impl Selector {
    /// Whether worker `w` is selected.
    pub fn matches(&self, w: &WorkerState) -> bool {
        match self {
            Self::Worker(id) => *id == w.id,
            Self::Class(class) => *class == w.class,
        }
    }
}

/// How a [`Constraint`] binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Strength {
    /// Run only on a selected worker. Requires of the same kind (all on workers, or all on
    /// classes) are alternatives, since a worker has one id and one class: the job runs on a
    /// worker matching at least one Require of each kind it has.
    Require,
    /// Never run on a selected worker.
    Forbid,
    /// Run on a selected worker only while no live worker (one with slots) that the hard
    /// constraints allow is free of every Avoid. That depends on the set of workers, not on their
    /// load, so a retry waits for a busy healthy worker rather than returning to the one it failed
    /// on.
    Avoid,
    /// Favour a selected worker where [`ScoreTerm::Preferred`] ranks workers (cache affinity).
    Prefer,
}

/// One placement constraint of a job ([`JobSpec::constraints`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Constraint {
    /// The workers it is about.
    pub on: Selector,
    /// How it binds.
    pub strength: Strength,
}

/// A job, as submitted to a [`Policy`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JobSpec {
    /// The job's id.
    pub id: JobId,
    /// What the job is expected to use while running (an estimate; it may be pessimistic). The
    /// [`SLOTS`] component is the scheduler's to set.
    pub demand: Resources,
    /// Priority group, e.g. the bidegree a job belongs to ([`OrderTerm::Group`]).
    pub group: u64,
    /// Explicit priority, the hook for an external planner ([`OrderTerm::Priority`]); smaller is
    /// more urgent. Jobs without one count as [`Config::default_priority`].
    pub priority: Option<i64>,
    /// Upward rank: the job's work plus the longest chain of work below it. Set by the DAG layer
    /// ([`DagConfig::rank_priority`]); larger is more urgent ([`OrderTerm::Rank`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rank: Option<f64>,
    /// Weight in a weighted objective ([`OrderTerm::Wspt`]). Default 1.
    #[cfg_attr(feature = "serde", serde(default = "unit"))]
    pub weight: f64,
    /// Due date, on the policy's clock ([`OrderTerm::Edd`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub due: Option<Instant>,
    /// Estimated work, in seconds on a worker of [`WorkerState::speed`] 1.0. Used by
    /// [`OrderTerm::Wspt`], [`Defer`], shadow backfill and [`Speculate`] (and filled in from the
    /// DAG layer's estimate when unset).
    #[cfg_attr(feature = "serde", serde(default))]
    pub work: Option<f64>,
    /// Where the job may, should and should not run. A retried job also avoids, softly, the
    /// workers its failed attempts ran on.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub constraints: Vec<Constraint>,
}

impl JobSpec {
    /// A job with the given id, demand and group, weight 1 and nothing else.
    pub fn new(id: JobId, demand: Resources, group: u64) -> Self {
        Self {
            id,
            demand,
            group,
            priority: None,
            rank: None,
            weight: 1.0,
            due: None,
            work: None,
            constraints: Vec::new(),
        }
    }

    /// This job with one more constraint.
    pub fn constrain(mut self, strength: Strength, on: Selector) -> Self {
        self.constraints.push(Constraint { on, strength });
        self
    }

    /// This job, run only on workers of `class` (or of another required class).
    pub fn require_class(self, class: impl Into<String>) -> Self {
        self.constrain(Strength::Require, Selector::Class(class.into()))
    }

    /// This job, never run on worker `w`.
    pub fn forbid_worker(self, w: WorkerId) -> Self {
        self.constrain(Strength::Forbid, Selector::Worker(w))
    }

    /// This job, softly avoiding worker `w`.
    pub fn avoid_worker(self, w: WorkerId) -> Self {
        self.constrain(Strength::Avoid, Selector::Worker(w))
    }

    /// This job, preferring worker `w`.
    pub fn prefer_worker(self, w: WorkerId) -> Self {
        self.constrain(Strength::Prefer, Selector::Worker(w))
    }
}

/// A worker's declared capacity and last reported usage.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkerState {
    /// The worker's id.
    pub id: WorkerId,
    /// The worker's class (e.g. GPU type), used by [`Selector::Class`] and per-class reservations.
    pub class: String,
    /// Concurrent jobs at most; zero admits nothing.
    pub slots: usize,
    /// Memory capacity: host memory, and device memory (the pool jobs' device allocations come
    /// from). A zero component is unknown and not enforced. Its [`SLOTS`] component is ignored:
    /// [`capacity`](Self::capacity) takes it from `slots`.
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
    /// [`JobSpec::work`] `w` takes `w / speed` seconds. Used by [`ScoreTerm::Speed`], [`Defer`]
    /// and [`Speculate`]. Default 1.0.
    #[cfg_attr(feature = "serde", serde(default = "unit"))]
    pub speed: f64,
}

/// The default [`WorkerState::speed`] and [`JobSpec::weight`].
#[cfg(feature = "serde")]
fn unit() -> f64 {
    1.0
}

impl WorkerState {
    /// A worker with `slots` execution slots, the memory capacity in `budget`, and nothing
    /// reported yet.
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

    /// The capacity admission enforces: `budget` in the memory dimensions, `slots` in
    /// [`SLOTS`].
    pub fn capacity(&self) -> Resources {
        let mut c = self.budget;
        c[SLOTS] = self.slots as u64;
        c
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
    /// Live attempts on it (a job speculated onto it counts here and on its other worker).
    pub running: usize,
    /// Sum of the demands of those attempts.
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
    /// Number of running jobs (with at least one live attempt), each counted once.
    pub running: usize,
    /// The waiting job that has waited longest, and for how long (seconds).
    pub longest_wait: Option<(JobId, f64)>,
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
    /// Jobs the last [`Policy::poll`] deliberately left waiting, at some point of its scan, for a faster
    /// worker that was busy ([`Defer`]), and did not place afterwards: `(job, worker it waits
    /// for, expected start there)`. Less urgent jobs may have taken slower workers meanwhile.
    pub deferred: Vec<(JobId, WorkerId, Instant)>,
    /// Every job that deferred at some point of the last [`Policy::poll`]'s scan, including those placed
    /// later in it (after a released reservation restarted the scan): while deferring, a job
    /// leaves the slower workers it declined to less urgent jobs.
    pub deferred_any: Vec<JobId>,
}

/// The number of a job's attempt: 1 for its first start, counting retries and speculative
/// attempts. The DAG layer's local jobs use 0 (see [`DagScheduler`]).
pub type Attempt = u32;

/// Why an attempt failed (the caller classifies; the policy records it in [`Tried`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FailKind {
    /// The worker ran out of device memory.
    DeviceOom,
    /// The connection to the worker died ([`Input::WorkerGone`] fails attempts with this).
    LinkDied,
    /// The worker refused the job.
    Rejected,
    /// The job took too long.
    Timeout,
    /// Anything else.
    Other,
}

/// One failed attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Tried {
    /// Where it ran.
    pub worker: WorkerId,
    /// How it failed.
    pub kind: FailKind,
    /// The caller's description.
    pub why: String,
}

/// A job the policy stopped retrying ([`RetryConfig::max_attempts`] rounds): it is forgotten.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GaveUp {
    /// The job.
    pub job: JobId,
    /// Every failed attempt, in order.
    pub tried: Vec<Tried>,
    /// Every attempt failed with [`FailKind::DeviceOom`]: the job might fit later, elsewhere, or
    /// split.
    pub retryable: bool,
}

/// An event a [`Policy`] reacts to.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Input {
    /// A job became ready. An id that is already waiting or running is ignored.
    Submit(JobSpec),
    /// An attempt finished: the job is complete, and every other live attempt of it is stopped
    /// ([`Output::Stop`]). Ignored unless `attempt` is live.
    Done {
        /// The job.
        job: JobId,
        /// The attempt that finished.
        attempt: Attempt,
    },
    /// An attempt failed. If no other attempt of the job is live, the job is retried (it keeps its
    /// place in the queue and its age, and softly avoids the workers it failed on) or, after
    /// [`RetryConfig::max_attempts`] rounds, given up ([`Output::GaveUp`]). Ignored unless
    /// `attempt` is live.
    Failed {
        /// The job.
        job: JobId,
        /// The attempt that failed.
        attempt: Attempt,
        /// How.
        kind: FailKind,
        /// The caller's description, kept in [`Tried`].
        why: String,
    },
    /// The job is no longer wanted: dropped if waiting, its live attempts stopped
    /// ([`Output::Stop`]) if running. Unknown ids are ignored.
    Cancel(JobId),
    /// A worker joined, or reported a heartbeat. Its live attempts are kept.
    Worker(WorkerState),
    /// A worker left: each live attempt on it fails with [`FailKind::LinkDied`], as if reported
    /// by [`Input::Failed`].
    WorkerGone(WorkerId),
}

/// What a [`Policy`] asks its caller to do.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Output {
    /// Start this attempt of the job on the worker. Report its end with [`Input::Done`] or
    /// [`Input::Failed`].
    Start {
        /// The job.
        job: JobId,
        /// The attempt's number.
        attempt: Attempt,
        /// Where to run it.
        worker: WorkerId,
    },
    /// Stop this attempt: another attempt won, or the job was cancelled. Its result is not wanted,
    /// and the policy has already released its resources.
    Stop {
        /// The job.
        job: JobId,
        /// The attempt.
        attempt: Attempt,
        /// Where it runs.
        worker: WorkerId,
    },
    /// The job failed too often and is forgotten.
    GaveUp(GaveUp),
    /// A [`DagJob::local`] job is ready: run it on the caller and report it with
    /// [`Input::Done`] and attempt 0.
    RunLocal {
        /// The job.
        job: JobId,
    },
    /// A job is ready and held (without [`DagConfig::auto_submit`]): submit it with
    /// [`DagScheduler::release`] when it is sendable.
    Ready {
        /// The job.
        job: JobId,
    },
    /// A passthrough job (or an instance's `done` job) completed (with
    /// [`DagConfig::record_passthrough`]).
    Passed {
        /// The job.
        job: JobId,
    },
}

/// A placement policy, driven by messages.
///
/// Jobs are idempotent: running one twice is harmless and the first completion wins. That is the
/// caller's side of the contract; it lets the policy retry failed attempts, start a speculative
/// second attempt ([`Speculate`]) and ignore late messages about attempts it no longer tracks.
///
/// The caller feeds every event to [`handle`](Self::handle), then calls [`poll`](Self::poll) and
/// acts on each [`Output`]. Every method is deterministic: the same sequence of calls (with the
/// same `now`s) produces the same outputs, which is what [`log::replay`] relies on.
pub trait Policy {
    /// Take in one event at time `now` (non-decreasing across calls). Outputs it causes (stops,
    /// give-ups) are returned by the next `poll`.
    fn handle(&mut self, input: Input, now: Instant);
    /// Place what can be placed now, and return every output since the last call, in order.
    fn poll(&mut self, now: Instant) -> Vec<Output>;
    /// The next time `poll` should be called even if no event arrives: a hold lapses (e.g. a
    /// [`Defer`] wait for a faster worker), a job ages, or a job may reserve. `None` if nothing
    /// is timed. Callers with frequent events may ignore it at the cost of that much extra
    /// waiting.
    fn next_wakeup(&self) -> Option<Instant>;
    /// Why a job is not running, in words (for logs). `None` for unknown jobs.
    fn explain(&self, job: JobId) -> Option<String>;
    /// Counters and current state.
    fn stats(&self) -> PolicyStats;
}

impl<P: Policy + ?Sized> Policy for Box<P> {
    /// Forwarded to the boxed policy.
    fn handle(&mut self, input: Input, now: Instant) {
        (**self).handle(input, now)
    }

    /// Forwarded to the boxed policy.
    fn poll(&mut self, now: Instant) -> Vec<Output> {
        (**self).poll(now)
    }

    /// Forwarded to the boxed policy.
    fn next_wakeup(&self) -> Option<Instant> {
        (**self).next_wakeup()
    }

    /// Forwarded to the boxed policy.
    fn explain(&self, job: JobId) -> Option<String> {
        (**self).explain(job)
    }

    /// Forwarded to the boxed policy.
    fn stats(&self) -> PolicyStats {
        (**self).stats()
    }
}
