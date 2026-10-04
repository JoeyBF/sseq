//! The DAG layer's configuration and its errors.

use std::time::Duration;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::JobId;
#[cfg(doc)]
use crate::{DagJob, DagScheduler, JobSpec, Output, TemplateSpec};

/// Configuration for [`DagScheduler`].
///
/// Without [`track_ranks`](Self::track_ranks), a job's rank covers its own unit only; with it,
/// the chain of dependents counts too. [`default_work`](Self::default_work) sizes jobs declared
/// without a [`work`](JobSpec::work).
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{Config, DagConfig, DagJob, DagScheduler, Scheduler, Time};
///
/// let ranks = |config| {
///     let mut dag = DagScheduler::new(config, Scheduler::new(Config::fifo()));
///     dag.declare(
///         [
///             DagJob {
///                 id: 1,
///                 ..Default::default()
///             },
///             DagJob {
///                 id: 2,
///                 deps: vec![1],
///                 ..Default::default()
///             },
///         ],
///         Time::ORIGIN,
///     )
///     .unwrap();
///     (dag.rank(1), dag.rank(2))
/// };
/// let config = DagConfig {
///     default_work: Duration::from_secs(2),
///     ..DagConfig::default()
/// };
/// let (two, four) = (Duration::from_secs(2), Duration::from_secs(4));
/// assert_eq!(ranks(config.clone()), (Some(four), Some(two)));
/// assert_eq!(
///     ranks(DagConfig {
///         track_ranks: false,
///         ..config
///     }),
///     (Some(two), Some(two))
/// );
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DagConfig {
    /// The work of a [`DagJob`] whose spec has no [`work`](JobSpec::work).
    ///
    /// A [`Unit`](crate::Unit) declared without a scale takes this, in seconds, as its scale. A
    /// plain job is a unit of a one-node template of one second of work, as are the nodes of
    /// [`TemplateSpec::jobs`], so each such job gets `default_work`.
    pub default_work: Duration,
    /// Ranks between units are maintained approximately: a rank increase smaller than this
    /// fraction is not propagated to the unit's dependencies. Larger makes growing the graph
    /// cheaper and ranks between units less exact; ranks within a unit are exact.
    pub rank_epsilon: f64,
    /// Submit jobs to the policy as soon as they are ready. When false, ready jobs are held and
    /// announced by [`Output::Ready`]; the caller submits each with [`DagScheduler::release`] when
    /// it is actually sendable (e.g. after coordinator-side preparation).
    pub auto_submit: bool,
    /// Announce passthrough leaves, and units other than plain jobs, as they complete, with
    /// [`Output::Passed`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub record_passthrough: bool,
    /// Maintain units' ranks as the graph grows and work changes, and submit each job with its
    /// upward rank (the critical path below it) as [`JobSpec::rank`] unless it has one. Whether
    /// ranks order anything is up to the policy ([`OrderTerm::Rank`](crate::OrderTerm::Rank)).
    /// Without them, declaring and re-estimating skip all rank propagation, which on long
    /// dependency chains is most of the cost.
    #[cfg_attr(feature = "serde", serde(default = "yes"))]
    pub track_ranks: bool,
}

/// `true`, for serde defaults.
#[cfg(feature = "serde")]
fn yes() -> bool {
    true
}

impl Default for DagConfig {
    /// Ranks tracked, jobs submitted as soon as they are ready, and nothing announced but local
    /// jobs.
    fn default() -> Self {
        Self {
            default_work: Duration::from_secs(1),
            rank_epsilon: 0.01,
            auto_submit: true,
            record_passthrough: false,
            track_ranks: true,
        }
    }
}

/// Errors from [`DagScheduler::declare`], [`DagScheduler::close`] and [`TemplateSpec::build`].
///
/// A failed declaration changes nothing. Each variant, as it is returned:
///
/// ```
/// use std::sync::Arc;
///
/// use whelm::{
///     Config, DagConfig, DagError, DagJob, DagScheduler, Scheduler, TemplateSpec, Time, Unit,
/// };
///
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
/// // Unit 99 owns ids 10, 11 and 12.
/// let unit = Unit {
///     id: 99,
///     base: 10,
///     template: Arc::new(TemplateSpec::jobs(3).build().unwrap()),
///     ..Default::default()
/// };
/// dag.declare([unit], Time::ORIGIN).unwrap();
///
/// let cycle = [
///     DagJob {
///         id: 1,
///         deps: vec![2],
///         ..Default::default()
///     },
///     DagJob {
///         id: 2,
///         deps: vec![1],
///         ..Default::default()
///     },
/// ];
/// assert_eq!(
///     dag.declare(cycle, Time::ORIGIN),
///     Err(DagError::Cycle { job: 1 })
/// );
/// assert_eq!(
///     TemplateSpec {
///         edges: vec![(0, 1), (1, 0)],
///         ..TemplateSpec::jobs(2)
///     }
///     .build()
///     .unwrap_err(),
///     DagError::Cycle { job: 0 }
/// );
/// assert_eq!(
///     dag.declare(
///         [DagJob {
///             id: 99,
///             ..Default::default()
///         }],
///         Time::ORIGIN
///     ),
///     Err(DagError::Duplicate(99))
/// );
/// assert_eq!(
///     dag.declare(
///         [DagJob {
///             id: 11,
///             ..Default::default()
///         }],
///         Time::ORIGIN
///     ),
///     Err(DagError::Overlap(11))
/// );
/// assert_eq!(
///     dag.declare(
///         [DagJob {
///             id: 5,
///             deps: vec![11],
///             ..Default::default()
///         }],
///         Time::ORIGIN
///     ),
///     Err(DagError::Overlap(11))
/// );
/// assert_eq!(dag.close(42, Time::ORIGIN), Err(DagError::NotFound(42)));
/// assert_eq!(
///     DagError::NotFound(42).to_string(),
///     "no live unit contains job 42"
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DagError {
    /// The declaration would close a dependency cycle through this unit (or template node).
    ///
    /// The cycle may run through units declared earlier: a forward reference can close one.
    Cycle {
        /// A unit on the cycle.
        job: JobId,
    },
    /// The unit is already declared (or completed), or appears twice in the batch.
    Duplicate(JobId),
    /// This id is both a unit's id or dependency and a leaf of another unit.
    ///
    /// Dependencies name units, not the jobs inside them, so naming another unit's leaf is
    /// refused rather than read as a dependency on the whole unit.
    Overlap(JobId),
    /// No live unit has this id or leaf ([`DagScheduler::close`]).
    NotFound(JobId),
}

impl std::fmt::Display for DagError {
    /// A one-line description of the error.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cycle { job } => {
                write!(f, "declaring unit {job} would create a dependency cycle")
            }
            Self::Duplicate(job) => write!(f, "unit {job} is already declared"),
            Self::Overlap(job) => write!(f, "id {job} is already a leaf of another unit"),
            Self::NotFound(job) => write!(f, "no live unit contains job {job}"),
        }
    }
}

impl std::error::Error for DagError {}
