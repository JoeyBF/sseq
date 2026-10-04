//! Job descriptions and their placement constraints.
//!
//! A [`JobSpec`] is what the caller says about a job when it submits it: what it is expected to use
//! ([`demand`](JobSpec::demand), see [`resources`](crate::resources)), the group it belongs to, and
//! the optional fields that the [ordering](crate::config) and [speed](crate::speed) features read.
//! This page is about the one part of a spec that restricts where the job may run: its constraints.
//!
//! # Constraints
//!
//! A job's [`constraints`](JobSpec::constraints) restrict where it runs. Each names workers with a
//! [`Selector`] (one worker, or a class of workers) and binds with a [`Strength`]: `Require` and
//! `Forbid` are hard, `Avoid` is soft, and `Prefer` only ranks the workers that admit the job. The
//! constructors on [`Constraint`] cover each strength on one worker or one class.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! # use whelm::job::Constraint;
//! let mut p = Scheduler::new(Config::default());
//! for (id, class) in [(1, "cpu"), (2, "gpu"), (3, "gpu")] {
//!     let worker = WorkerState {
//!         id,
//!         class: class.into(),
//!         capacity: Resources::new().with(SLOTS, 4),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Worker(worker), Time::ORIGIN);
//! }
//! let constrained = |constraints| JobSpec {
//!     constraints,
//!     ..Default::default()
//! };
//! let gpu = Constraint::require_class("gpu");
//! let spec = constrained(vec![gpu.clone()]);
//! p.handle(Input::Submit { job: 1, spec }, Time::ORIGIN);
//! let not_2 = Constraint::forbid_worker(2);
//! let spec = constrained(vec![gpu, not_2]);
//! p.handle(Input::Submit { job: 2, spec }, Time::ORIGIN);
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [
//!         Output::Start {
//!             job: 1,
//!             attempt: 1,
//!             worker: 2
//!         },
//!         Output::Start {
//!             job: 2,
//!             attempt: 1,
//!             worker: 3
//!         },
//!     ]
//! );
//!
//! // No worker has the class: the job waits, and says why.
//! let tpu = Constraint::require_class("tpu");
//! let spec = constrained(vec![tpu]);
//! p.handle(Input::Submit { job: 3, spec }, Time(Duration::from_secs(1)));
//! assert!(p.poll(Time(Duration::from_secs(1))).is_empty());
//! let why = p.explain(3).unwrap();
//! let verdicts = &why.waiting().unwrap().workers;
//! assert!(
//!     verdicts
//!         .iter()
//!         .all(|(_, v)| *v == whelm::explain::Verdict::Ineligible)
//! );
//! assert!(
//!     why.to_string()
//!         .ends_with("; 3 worker(s) excluded by its constraints")
//! );
//! ```
//!
//! # Preferring a worker
//!
//! A preferred worker wins over a less loaded one: the default score ranks
//! [`Preferred`](ScoreTerm::Preferred) before [`Load`](ScoreTerm::Load) (see
//! [choosing a worker](crate::config#choosing-a-worker)). Use it for cache affinity.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! # use whelm::job::Constraint;
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id,
//!             capacity: Resources::new().with(SLOTS, 4),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! p.handle(
//!     Input::Submit {
//!         job: 1,
//!         spec: JobSpec::default(),
//!     },
//!     Time::ORIGIN,
//! );
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // Worker 1 is busier, but job 2 prefers it; job 3 has no preference.
//! let fond = JobSpec {
//!     constraints: vec![Constraint::prefer_worker(1)],
//!     ..Default::default()
//! };
//! let later = Time(Duration::from_secs(1));
//! p.handle(Input::Submit { job: 2, spec: fond }, later);
//! p.handle(
//!     Input::Submit {
//!         job: 3,
//!         spec: JobSpec::default(),
//!     },
//!     later,
//! );
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(1))),
//!     [
//!         Output::Start {
//!             job: 2,
//!             attempt: 1,
//!             worker: 1
//!         },
//!         Output::Start {
//!             job: 3,
//!             attempt: 1,
//!             worker: 2
//!         },
//!     ]
//! );
//! ```
//!
//! # Avoiding a worker
//!
//! An avoided worker is used only while no live worker (one with slots) that the hard constraints
//! allow is free of every `Avoid`. That depends on which workers exist, not on how loaded they are,
//! so the job waits for a busy acceptable worker; it is the same rule a
//! [retry](crate#failures-and-retries) applies to the workers it failed on. Here job 2 avoids
//! worker 1 and waits for worker 2, until worker 2 is drained (its slot count set to zero by a
//! heartbeat).
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! # use whelm::job::Constraint;
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id,
//!             capacity: Resources::new().with(SLOTS, 1),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! let constrained = |constraint| JobSpec {
//!     constraints: vec![constraint],
//!     ..Default::default()
//! };
//! let spec = constrained(Constraint::prefer_worker(2));
//! p.handle(Input::Submit { job: 1, spec }, Time::ORIGIN);
//! let spec = constrained(Constraint::avoid_worker(1));
//! p.handle(Input::Submit { job: 2, spec }, Time::ORIGIN);
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 2
//!     }]
//! );
//! let why = p.explain(2).unwrap();
//! assert_eq!(
//!     why.waiting().unwrap().workers[0],
//!     (1, whelm::explain::Verdict::Ineligible)
//! );
//!
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 2,
//!         capacity: Resources::new().with(SLOTS, 0),
//!         ..Default::default()
//!     }),
//!     Time(Duration::from_secs(1)),
//! );
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(1))),
//!     [Output::Start {
//!         job: 2,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! ```

use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{
    config::{Config, Defer, OrderTerm, ScoreTerm, Speculate},
    dag::{DagConfig, DagScheduler},
    policy::{Input, Output, Policy},
    resources::Resource,
    speed::Timing,
};
use crate::{
    resources::Resources,
    time::Time,
    worker::{WorkerId, WorkerState},
};

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
/// use whelm::{job::Selector, prelude::*};
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
/// `Require` and `Forbid` are hard, `Avoid` is soft and `Prefer` only ranks workers. The module's
/// [Constraints](crate::job#constraints) chapter shows each one placing jobs.
///
/// # Examples
///
/// Requires of one kind are alternatives: this job may run on either class.
///
/// ```
/// use whelm::{job::Constraint, prelude::*};
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
/// use whelm::{
///     job::{Constraint, Selector, Strength},
///     prelude::*,
/// };
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
    /// use whelm::job::{Constraint, Selector, Strength};
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
    /// use whelm::{job::Constraint, prelude::*};
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
    /// use whelm::{job::Constraint, prelude::*};
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
    /// use whelm::job::{Constraint, Selector, Strength};
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
    /// use whelm::{job::Constraint, prelude::*};
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
    /// use whelm::job::{Constraint, Selector, Strength};
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
    /// use whelm::{job::Constraint, prelude::*};
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
    /// use whelm::job::{Constraint, Selector, Strength};
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
/// use whelm::{job::Constraint, prelude::*};
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
