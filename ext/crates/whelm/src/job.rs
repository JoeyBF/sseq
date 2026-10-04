//! Job descriptions and their placement constraints.

use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{
    Config, DagConfig, DagScheduler, Defer, Input, OrderTerm, Output, Policy, Resource, ScoreTerm,
    Speculate, Timing,
};
use crate::{Resources, Time, WorkerId, WorkerState};

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
/// use whelm::{Resources, SLOTS, Selector, WorkerState};
///
/// let w = WorkerState {
///     id: 7,
///     class: "gpu".into(),
///     capacity: Resources::new().with(SLOTS, 1),
///     ..Default::default()
/// };
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
/// use whelm::{
///     Config, Constraint, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
///     WorkerState,
/// };
///
/// let mut p = Scheduler::new(Config::default());
/// for (id, class) in [(1, "cpu"), (2, "a100"), (3, "h100")] {
///     let w = WorkerState {
///         id,
///         class: class.into(),
///         capacity: Resources::new().with(SLOTS, 1),
///         ..Default::default()
///     };
///     p.handle(Input::Worker(w), Time::ORIGIN);
/// }
/// for id in 1..=3 {
///     let spec = JobSpec {
///         constraints: vec![
///             Constraint::require_class("a100"),
///             Constraint::require_class("h100"),
///         ],
///         ..Default::default()
///     };
///     p.handle(Input::Submit { job: id, spec }, Time::ORIGIN);
/// }
/// assert_eq!(
///     p.poll(Time::ORIGIN),
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
    /// Run on a selected worker only while no live worker (one that would admit some job if it
    /// were empty) that the hard constraints allow is free of every Avoid. That depends on the set
    /// of workers, not on their load, so a retry waits for a busy healthy worker rather than
    /// returning to the one it failed on.
    Avoid,
    /// Favour a selected worker where [`ScoreTerm::Preferred`] ranks workers (cache affinity).
    Prefer,
}

/// One placement constraint of a job ([`JobSpec::constraints`]).
///
/// The named constructors cover each [`Strength`] on one worker or one class; the literal says the
/// same thing at more length.
///
/// # Examples
///
/// ```
/// use whelm::{Constraint, JobSpec, Selector, Strength};
///
/// let job = JobSpec {
///     constraints: vec![Constraint::avoid_worker(3)],
///     ..Default::default()
/// };
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

impl Constraint {
    /// Run only on worker `w` (or on another required worker).
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{Constraint, Selector, Strength};
    ///
    /// let c = Constraint::require_worker(2);
    /// assert_eq!((c.on, c.strength), (Selector::Worker(2), Strength::Require));
    /// ```
    pub fn require_worker(w: WorkerId) -> Self {
        Self {
            on: Selector::Worker(w),
            strength: Strength::Require,
        }
    }

    /// Run only on workers of `class` (or of another required class).
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{
    ///     Config, Constraint, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
    ///     WorkerState,
    /// };
    ///
    /// let mut p = Scheduler::new(Config::default());
    /// for (id, class) in [(1, "cpu"), (2, "gpu")] {
    ///     let w = WorkerState {
    ///         id,
    ///         class: class.into(),
    ///         capacity: Resources::new().with(SLOTS, 1),
    ///         ..Default::default()
    ///     };
    ///     p.handle(Input::Worker(w), Time::ORIGIN);
    /// }
    /// let spec = JobSpec {
    ///     constraints: vec![Constraint::require_class("gpu")],
    ///     ..Default::default()
    /// };
    /// p.handle(Input::Submit { job: 1, spec }, Time::ORIGIN);
    /// assert_eq!(
    ///     p.poll(Time::ORIGIN),
    ///     [Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 2
    ///     }]
    /// );
    /// ```
    pub fn require_class(class: impl Into<String>) -> Self {
        Self {
            on: Selector::Class(class.into()),
            strength: Strength::Require,
        }
    }

    /// Never run on worker `w`.
    ///
    /// # Examples
    ///
    /// Unlike an avoided worker, a forbidden one is never used, even when it is the only one.
    ///
    /// ```
    /// use whelm::{
    ///     Config, Constraint, Input, JobSpec, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
    /// };
    ///
    /// let mut p = Scheduler::new(Config::default());
    /// p.handle(
    ///     Input::Worker(WorkerState {
    ///         id: 1,
    ///         capacity: Resources::new().with(SLOTS, 1),
    ///         ..Default::default()
    ///     }),
    ///     Time::ORIGIN,
    /// );
    /// let spec = JobSpec {
    ///     constraints: vec![Constraint::forbid_worker(1)],
    ///     ..Default::default()
    /// };
    /// p.handle(Input::Submit { job: 1, spec }, Time::ORIGIN);
    /// assert!(p.poll(Time::ORIGIN).is_empty());
    /// ```
    pub fn forbid_worker(w: WorkerId) -> Self {
        Self {
            on: Selector::Worker(w),
            strength: Strength::Forbid,
        }
    }

    /// Never run on a worker of `class`.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{Constraint, Selector, Strength};
    ///
    /// let c = Constraint::forbid_class("cpu");
    /// assert_eq!(
    ///     (c.on, c.strength),
    ///     (Selector::Class("cpu".into()), Strength::Forbid)
    /// );
    /// ```
    pub fn forbid_class(class: impl Into<String>) -> Self {
        Self {
            on: Selector::Class(class.into()),
            strength: Strength::Forbid,
        }
    }

    /// Softly avoid worker `w`.
    ///
    /// # Examples
    ///
    /// With no other live worker, the avoided one is used after all (see [`Strength::Avoid`]).
    ///
    /// ```
    /// use whelm::{
    ///     Config, Constraint, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
    ///     WorkerState,
    /// };
    ///
    /// let mut p = Scheduler::new(Config::default());
    /// p.handle(
    ///     Input::Worker(WorkerState {
    ///         id: 1,
    ///         capacity: Resources::new().with(SLOTS, 1),
    ///         ..Default::default()
    ///     }),
    ///     Time::ORIGIN,
    /// );
    /// let spec = JobSpec {
    ///     constraints: vec![Constraint::avoid_worker(1)],
    ///     ..Default::default()
    /// };
    /// p.handle(Input::Submit { job: 1, spec }, Time::ORIGIN);
    /// assert_eq!(
    ///     p.poll(Time::ORIGIN),
    ///     [Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn avoid_worker(w: WorkerId) -> Self {
        Self {
            on: Selector::Worker(w),
            strength: Strength::Avoid,
        }
    }

    /// Softly avoid the workers of `class`.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{Constraint, Selector, Strength};
    ///
    /// let c = Constraint::avoid_class("flaky");
    /// assert_eq!(
    ///     (c.on, c.strength),
    ///     (Selector::Class("flaky".into()), Strength::Avoid)
    /// );
    /// ```
    pub fn avoid_class(class: impl Into<String>) -> Self {
        Self {
            on: Selector::Class(class.into()),
            strength: Strength::Avoid,
        }
    }

    /// Prefer worker `w`.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{
    ///     Config, Constraint, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
    ///     WorkerState,
    /// };
    ///
    /// let mut p = Scheduler::new(Config::default());
    /// for id in [1, 2] {
    ///     p.handle(
    ///         Input::Worker(WorkerState {
    ///             id,
    ///             capacity: Resources::new().with(SLOTS, 1),
    ///             ..Default::default()
    ///         }),
    ///         Time::ORIGIN,
    ///     );
    /// }
    /// let spec = JobSpec {
    ///     constraints: vec![Constraint::prefer_worker(2)],
    ///     ..Default::default()
    /// };
    /// p.handle(Input::Submit { job: 1, spec }, Time::ORIGIN);
    /// assert_eq!(
    ///     p.poll(Time::ORIGIN),
    ///     [Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 2
    ///     }]
    /// );
    /// ```
    pub fn prefer_worker(w: WorkerId) -> Self {
        Self {
            on: Selector::Worker(w),
            strength: Strength::Prefer,
        }
    }

    /// Prefer the workers of `class`.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{Constraint, Selector, Strength};
    ///
    /// let c = Constraint::prefer_class("l40s");
    /// assert_eq!(
    ///     (c.on, c.strength),
    ///     (Selector::Class("l40s".into()), Strength::Prefer)
    /// );
    /// ```
    pub fn prefer_class(class: impl Into<String>) -> Self {
        Self {
            on: Selector::Class(class.into()),
            strength: Strength::Prefer,
        }
    }
}

/// What a job is, submitted to a [`Policy`] beside its id ([`Input::Submit`]).
///
/// Only the demand and group are needed; every other field refines ordering, placement or timing,
/// and is read only by the [`Config`] terms and features that use it.
///
/// # Examples
///
/// Name the fields that matter and take the rest from [`Default`].
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{Constraint, JobSpec, MEMORY, Resources, gb};
///
/// let job = JobSpec {
///     demand: Resources::new().with(MEMORY, gb(6.0)),
///     group: 3,
///     priority: Some(-1),
///     work: Some(Duration::from_secs(120)),
///     kind: Some("sig".into()),
///     constraints: vec![Constraint::require_class("gpu")],
///     ..Default::default()
/// };
/// assert_eq!((job.weight, job.due), (1.0, None));
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JobSpec {
    /// What the job is expected to use while running (an estimate; it may be pessimistic). The
    /// scheduler fills in each declared resource it leaves out with that resource's
    /// [`default_demand`](Resource::default_demand) (one slot, under the default declaration), and
    /// [rejects](Output::Rejected) the job if it names a resource [`Config::resources`] does not
    /// declare.
    pub demand: Resources,
    /// Priority group, e.g. the bidegree a job belongs to ([`OrderTerm::Group`]).
    pub group: u64,
    /// Explicit priority, the hook for an external planner ([`OrderTerm::Priority`]); smaller is
    /// more urgent. Jobs without one count as [`Config::default_priority`].
    pub priority: Option<i64>,
    /// Upward rank: the job's work plus the longest chain of work below it. Set by the DAG layer
    /// ([`DagConfig::track_ranks`]); larger is more urgent ([`OrderTerm::Rank`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rank: Option<Duration>,
    /// Weight in a weighted objective ([`OrderTerm::Wspt`]).
    #[cfg_attr(feature = "serde", serde(default = "crate::worker::unit"))]
    pub weight: f64,
    /// Due date, on the policy's clock ([`OrderTerm::Edd`]).
    #[cfg_attr(feature = "serde", serde(default))]
    pub due: Option<Time>,
    /// Estimated work: the run time on a worker of [`WorkerState::speed`] 1.0. Used by
    /// [`OrderTerm::Wspt`], [`Defer`], shadow backfill and [`Speculate`], and by the DAG layer's
    /// ranks, which fill it in when unset.
    #[cfg_attr(feature = "serde", serde(default))]
    pub work: Option<Duration>,
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

impl Default for JobSpec {
    /// A job weighing 1, so that a weighted objective counts jobs it is not told about alike.
    fn default() -> Self {
        Self {
            demand: Resources::new(),
            group: 0,
            priority: None,
            rank: None,
            weight: 1.0,
            due: None,
            work: None,
            kind: None,
            constraints: Vec::new(),
        }
    }
}
