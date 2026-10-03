//! The DAG layer's configuration and its errors.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::JobId;
#[cfg(doc)]
use crate::{DagJob, DagScheduler, JobSpec, Output, TemplateSpec};

/// Configuration for [`DagScheduler`].
///
/// Without [`track_ranks`](Self::track_ranks), a job's rank covers its own unit only; with it,
/// the chain of dependents counts too. [`default_work`](Self::default_work) sizes jobs declared
/// without an estimate.
///
/// ```
/// use whelm::{Config, DagConfig, DagJob, DagScheduler, JobSpec, Scheduler};
///
/// let job = |id, deps| DagJob {
///     spec: JobSpec {
///         id,
///         ..Default::default()
///     },
///     deps,
///     ..Default::default()
/// };
/// let ranks = |config| {
///     let mut dag = DagScheduler::new(config, Scheduler::new(Config::fifo()));
///     dag.declare([job(1, vec![]), job(2, vec![1])], 0.0).unwrap();
///     (dag.rank(1), dag.rank(2))
/// };
/// let config = DagConfig {
///     default_work: 2.0,
///     ..DagConfig::default()
/// };
/// assert_eq!(ranks(config.clone()), (Some(4.0), Some(2.0)));
/// assert_eq!(
///     ranks(DagConfig {
///         track_ranks: false,
///         ..config
///     }),
///     (Some(2.0), Some(2.0))
/// );
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DagConfig {
    /// Scale of a unit declared without one, i.e. the work of a [`DagJob`] without an estimate.
    pub default_work: f64,
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
            default_work: 1.0,
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
///     Config, DagConfig, DagError, DagJob, DagScheduler, JobSpec, Scheduler, TemplateSpec, Unit,
/// };
///
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
/// let job = |id, deps| DagJob {
///     spec: JobSpec {
///         id,
///         ..Default::default()
///     },
///     deps,
///     ..Default::default()
/// };
/// // Unit 99 owns ids 10, 11 and 12.
/// let unit = Unit {
///     id: 99,
///     base: 10,
///     template: Arc::new(TemplateSpec::jobs(3).build().unwrap()),
///     ..Default::default()
/// };
/// dag.declare([unit], 0.0).unwrap();
///
/// let cycle = [job(1, vec![2]), job(2, vec![1])];
/// assert_eq!(dag.declare(cycle, 0.0), Err(DagError::Cycle { job: 1 }));
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
///     dag.declare([job(99, vec![])], 0.0),
///     Err(DagError::Duplicate(99))
/// );
/// assert_eq!(
///     dag.declare([job(11, vec![])], 0.0),
///     Err(DagError::Overlap(11))
/// );
/// assert_eq!(
///     dag.declare([job(5, vec![11])], 0.0),
///     Err(DagError::Overlap(11))
/// );
/// assert_eq!(dag.close(42, 0.0), Err(DagError::NotFound(42)));
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
