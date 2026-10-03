//! Job descriptions and their placement constraints.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{
    Config, DagConfig, DagScheduler, Defer, OrderTerm, Policy, SLOTS, ScoreTerm, Speculate, Timing,
};
use crate::{Instant, Resources, WorkerId, WorkerState};

/// A job identifier, chosen by the caller. Must be unique among live (waiting or running) jobs.
///
/// A completed or given-up job's id may be reused; under a [`DagScheduler`], ids also name units
/// and are remembered once complete, so they should not be.
pub type JobId = u64;

/// What a [`Constraint`] is about: one worker, or every worker of a class.
///
/// # Examples
///
/// ```
/// use whelm::{Resources, Selector, WorkerState};
///
/// let w = WorkerState::new(7, "gpu", 4, Resources::ZERO);
/// assert!(Selector::Worker(7).matches(&w));
/// assert!(Selector::Class("gpu".into()).matches(&w));
/// assert!(!Selector::Class("cpu".into()).matches(&w));
/// ```
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
///
/// `Require` and `Forbid` are hard, `Avoid` is soft and `Prefer` only ranks workers. The crate's
/// [Constraints](crate#constraints) chapter shows each one placing jobs.
///
/// # Examples
///
/// Requires of one kind are alternatives: this job may run on either class.
///
/// ```
/// use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
///
/// let mut p = Scheduler::new(Config::default());
/// for (id, class) in [(1, "cpu"), (2, "a100"), (3, "h100")] {
///     p.handle(
///         Input::Worker(WorkerState::new(id, class, 1, Resources::ZERO)),
///         0.0,
///     );
/// }
/// for id in 1..=3 {
///     let job = JobSpec::new(id, Resources::ZERO, 0)
///         .require_class("a100")
///         .require_class("h100");
///     p.handle(Input::Submit(job), 0.0);
/// }
/// assert_eq!(
///     p.poll(0.0),
///     [
///         Output::Start {
///             job: 1,
///             attempt: 1,
///             worker: 2
///         },
///         Output::Start {
///             job: 2,
///             attempt: 1,
///             worker: 3
///         },
///     ]
/// );
/// ```
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
///
/// # Examples
///
/// The [`JobSpec`] builders push constraints; [`JobSpec::constrain`] takes any combination.
///
/// ```
/// use whelm::{Constraint, JobSpec, Resources, Selector, Strength};
///
/// let job = JobSpec::new(1, Resources::ZERO, 0).avoid_worker(3);
/// let avoid = Constraint {
///     on: Selector::Worker(3),
///     strength: Strength::Avoid,
/// };
/// assert_eq!(job.constraints, [avoid]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Constraint {
    /// The workers it is about.
    pub on: Selector,
    /// How it binds.
    pub strength: Strength,
}

/// A job, as submitted to a [`Policy`].
///
/// Only the id, demand and group are needed; every other field refines ordering, placement or
/// timing, and is read only by the [`Config`] terms and features that use it.
///
/// # Examples
///
/// Start from [`JobSpec::new`] and fill in the rest with struct update syntax or the builders.
///
/// ```
/// use whelm::{JobSpec, Resources};
///
/// let job = JobSpec {
///     priority: Some(-1),
///     work: Some(120.0),
///     ..JobSpec::new(42, Resources::mem_gb(6.0), 3)
/// }
/// .with_kind("sig")
/// .require_class("gpu");
/// assert_eq!((job.id, job.group, job.weight), (42, 3, 1.0));
/// assert_eq!(job.kind.as_deref(), Some("sig"));
/// assert_eq!(job.constraints.len(), 1);
/// ```
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
    /// ([`DagConfig::track_ranks`]); larger is more urgent ([`OrderTerm::Rank`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rank: Option<f64>,
    /// Weight in a weighted objective ([`OrderTerm::Wspt`]).
    #[cfg_attr(feature = "serde", serde(default = "crate::worker::unit"))]
    pub weight: f64,
    /// Due date, on the policy's clock ([`OrderTerm::Edd`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub due: Option<Instant>,
    /// Estimated work, in seconds on a worker of [`WorkerState::speed`] 1.0. Used by
    /// [`OrderTerm::Wspt`], [`Defer`], shadow backfill and [`Speculate`] (and filled in from the
    /// DAG layer's estimate when unset).
    #[cfg_attr(feature = "serde", serde(default))]
    pub work: Option<f64>,
    /// What kind of job it is, for [`Timing::Unrelated`], which learns each kind's speed on each
    /// worker class; other timings ignore it.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub kind: Option<String>,
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
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{JobSpec, Resources};
    ///
    /// let job = JobSpec::new(1, Resources::mem_gb(2.0), 0);
    /// assert_eq!((job.priority, job.work, job.weight), (None, None, 1.0));
    /// assert!(job.constraints.is_empty());
    /// ```
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
            kind: None,
            constraints: Vec::new(),
        }
    }

    /// This job, of kind `kind` ([`JobSpec::kind`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{JobSpec, Resources};
    ///
    /// let job = JobSpec::new(1, Resources::ZERO, 0).with_kind("zero");
    /// assert_eq!(job.kind.as_deref(), Some("zero"));
    /// ```
    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = Some(kind.into());
        self
    }

    /// This job with one more constraint.
    ///
    /// # Examples
    ///
    /// Constraints the shorthand builders do not cover, such as avoiding a whole class.
    ///
    /// ```
    /// use whelm::{JobSpec, Resources, Selector, Strength};
    ///
    /// let job = JobSpec::new(1, Resources::ZERO, 0)
    ///     .constrain(Strength::Avoid, Selector::Class("flaky".into()));
    /// assert_eq!(job.constraints[0].strength, Strength::Avoid);
    /// ```
    pub fn constrain(mut self, strength: Strength, on: Selector) -> Self {
        self.constraints.push(Constraint { on, strength });
        self
    }

    /// This job, run only on workers of `class` (or of another required class).
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
    ///
    /// let mut p = Scheduler::new(Config::default());
    /// p.handle(
    ///     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)),
    ///     0.0,
    /// );
    /// p.handle(
    ///     Input::Worker(WorkerState::new(2, "gpu", 1, Resources::ZERO)),
    ///     0.0,
    /// );
    /// p.handle(
    ///     Input::Submit(JobSpec::new(1, Resources::ZERO, 0).require_class("gpu")),
    ///     0.0,
    /// );
    /// assert_eq!(
    ///     p.poll(0.0),
    ///     [Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 2
    ///     }]
    /// );
    /// ```
    pub fn require_class(self, class: impl Into<String>) -> Self {
        self.constrain(Strength::Require, Selector::Class(class.into()))
    }

    /// This job, never run on worker `w`.
    ///
    /// # Examples
    ///
    /// Unlike an avoided worker, a forbidden one is never used, even when it is the only one.
    ///
    /// ```
    /// use whelm::{Config, Input, JobSpec, Policy, Resources, Scheduler, WorkerState};
    ///
    /// let mut p = Scheduler::new(Config::default());
    /// p.handle(
    ///     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)),
    ///     0.0,
    /// );
    /// p.handle(
    ///     Input::Submit(JobSpec::new(1, Resources::ZERO, 0).forbid_worker(1)),
    ///     0.0,
    /// );
    /// assert!(p.poll(0.0).is_empty());
    /// ```
    pub fn forbid_worker(self, w: WorkerId) -> Self {
        self.constrain(Strength::Forbid, Selector::Worker(w))
    }

    /// This job, softly avoiding worker `w`.
    ///
    /// # Examples
    ///
    /// With no other live worker, the avoided one is used after all (see [`Strength::Avoid`]).
    ///
    /// ```
    /// use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
    ///
    /// let mut p = Scheduler::new(Config::default());
    /// p.handle(
    ///     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)),
    ///     0.0,
    /// );
    /// p.handle(
    ///     Input::Submit(JobSpec::new(1, Resources::ZERO, 0).avoid_worker(1)),
    ///     0.0,
    /// );
    /// assert_eq!(
    ///     p.poll(0.0),
    ///     [Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn avoid_worker(self, w: WorkerId) -> Self {
        self.constrain(Strength::Avoid, Selector::Worker(w))
    }

    /// This job, preferring worker `w`.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
    ///
    /// let mut p = Scheduler::new(Config::default());
    /// p.handle(
    ///     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)),
    ///     0.0,
    /// );
    /// p.handle(
    ///     Input::Worker(WorkerState::new(2, "cpu", 1, Resources::ZERO)),
    ///     0.0,
    /// );
    /// p.handle(
    ///     Input::Submit(JobSpec::new(1, Resources::ZERO, 0).prefer_worker(2)),
    ///     0.0,
    /// );
    /// assert_eq!(
    ///     p.poll(0.0),
    ///     [Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 2
    ///     }]
    /// );
    /// ```
    pub fn prefer_worker(self, w: WorkerId) -> Self {
        self.constrain(Strength::Prefer, Selector::Worker(w))
    }
}
