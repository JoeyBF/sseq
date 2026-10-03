//! The optional dependency layer in front of a [`Policy`].
//!
//! [`DagScheduler`] wraps any [`Policy`] and is one itself. It holds jobs back until their
//! dependencies complete, then submits them to the inner policy, which places them as it places
//! any job. Everything below is deterministic, like the policy it wraps.
//!
//! # Units over templates
//!
//! The graph has two levels. A [`DagTemplate`] is a dependency structure over nodes, each a
//! [`TemplateNode`]: a job run on a worker, a job run on the caller, a passthrough (a
//! synchronisation point that runs nothing), or a unit of another template substituted for the
//! node. A [`Unit`] is one instance of a template, with its own id and dependencies on other units;
//! the units form the *coarse graph*. A unit's jobs are its template's *leaves*, numbered in node
//! order from the unit's [`base`](Unit::base). Units share their template through an `Arc`, so
//! declaring one costs the same however many jobs it holds. A [`DagJob`] is the degenerate case: a
//! unit of a one-node template whose id is its job's.
//!
//! # Readiness
//!
//! A unit is *entered* once every unit it depends on has completed; a dependency may name a unit
//! declared later (a forward reference). The template's sources are then ready, and any other node
//! is ready once its predecessors in the template have completed. A ready node is
//!
//! - a worker job ([`TemplateNode::Job`]): submitted to the inner policy, or, without
//!   [`DagConfig::auto_submit`], held and announced by [`Output::Ready`] until the caller
//!   [`release`](DagScheduler::release)s it;
//! - a local job ([`TemplateNode::Local`]): held and announced by [`Output::RunLocal`]; the caller
//!   runs it and reports [`Input::Done`] with attempt 0;
//! - a passthrough ([`TemplateNode::Pass`]): complete at once, announced by [`Output::Passed`] with
//!   [`DagConfig::record_passthrough`];
//! - a substituted unit ([`TemplateNode::Unit`]): entered in turn.
//!
//! A unit completes once all of its nodes have, which may release its dependents.
//!
//! # Lazy materialisation
//!
//! A declared unit keeps only what its ranks need. Its per-node state (a counter of unmet
//! dependencies for each template node) is allocated when it is entered, a substituted unit's when
//! its node is ready, and freed when it completes, so memory follows the frontier rather than the
//! declared graph ([`DagStats::frames`], [`DagStats::node_bytes`]). This affects memory only: which
//! jobs are ready when, and with what ranks, is the same as if every unit were materialised at
//! declaration.
//!
//! # Ranks
//!
//! With [`DagConfig::track_ranks`], each job is submitted with its upward rank as
//! [`JobSpec::rank`]: its work plus the longest chain of work below it, through its own unit and
//! on through the units depending on it. A policy orders by it with
//! [`OrderTerm::Rank`](crate::OrderTerm::Rank), which runs the longest remaining chain first.
//! Ranks within a unit are exact; ranks between units are maintained approximately
//! ([`DagConfig::rank_epsilon`]) as the graph grows and [`DagScheduler::update_work`] revises
//! estimates. [`DagScheduler::rank`] reads them.
//!
//! # Outputs
//!
//! [`poll`](Policy::poll) returns this layer's announcements ([`Output::RunLocal`],
//! [`Output::Ready`], [`Output::Passed`]) and then the inner policy's outputs.
//! [`DagScheduler::announcements`] drains the announcements alone, so the caller can act on them
//! before anything is placed. A job the inner policy gives up on ([`Output::GaveUp`]) is held
//! again rather than forgotten: its dependents wait until it is released or cancelled.
//!
//! # Resuming and closing
//!
//! A unit declared with leaves already complete ([`Unit::with_completed`]) runs only the rest, so
//! a run restarted from a checkpoint redoes nothing. [`DagScheduler::close`] completes a unit early
//! when its remaining jobs are known to be no-ops, and [`DagScheduler::cancel`] removes a unit and
//! everything depending on it.
//!
//! # Snapshots
//!
//! With the `serde` feature, `DagScheduler::snapshot` captures the declared graph and the
//! materialised state as a `DagSnapshot`, and `DagScheduler::restore` rebuilds the layer in front
//! of a fresh policy, resubmitting the jobs the old policy had.
//!
//! # Example
//!
//! Two units of one template, the second after the first: each loads its input on the caller,
//! then runs two jobs. One single-slot worker runs everything; the second unit is materialised
//! only once the first completes.
//!
//! ```
//! use std::sync::Arc;
//!
//! use whelm::{
//!     Config, DagConfig, DagScheduler, DagTemplate, Input, Output, Policy, Resources, Scheduler,
//!     TemplateNode, Unit, WorkerState,
//! };
//!
//! // Node 0 loads (on the caller), nodes 1 and 2 compute after it.
//! let template = Arc::new(
//!     DagTemplate::with_nodes(
//!         vec![
//!             TemplateNode::Local(1.0),
//!             TemplateNode::Job(2.0),
//!             TemplateNode::Job(2.0),
//!         ],
//!         [(0, 1), (0, 2)],
//!     )
//!     .unwrap(),
//! );
//! let spec = whelm::JobSpec::new(0, Resources::mem(1), 0);
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
//! dag.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))),
//!     0.0,
//! );
//!
//! // Unit 1000 has jobs 0..3, unit 2000 jobs 10..13 and waits for unit 1000.
//! dag.declare(
//!     [
//!         Unit::new(1000, 0, template.clone(), spec.clone(), vec![]),
//!         Unit::new(2000, 10, template, spec, vec![1000]),
//!     ],
//!     0.0,
//! )
//! .unwrap();
//! // Ranks run through both units: 1 + 2 in each.
//! assert_eq!(
//!     (dag.rank(0), dag.rank(1), dag.rank(10)),
//!     (Some(6.0), Some(5.0), Some(3.0))
//! );
//! let stats = dag.dag_stats();
//! assert_eq!((stats.units, stats.open, stats.frames), (2, 1, 1));
//!
//! assert_eq!(dag.poll(0.0), vec![Output::RunLocal { job: 0 }]);
//! dag.handle(Input::Done { job: 0, attempt: 0 }, 1.0);
//! let start = |job| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker: 1,
//! };
//! assert_eq!(dag.poll(1.0), vec![start(1)]);
//! dag.handle(Input::Done { job: 1, attempt: 1 }, 3.0);
//! assert_eq!(dag.poll(3.0), vec![start(2)]);
//! // The last job of unit 1000 completes it, which enters unit 2000.
//! dag.handle(Input::Done { job: 2, attempt: 1 }, 5.0);
//! assert_eq!(dag.poll(5.0), vec![Output::RunLocal { job: 10 }]);
//! let stats = dag.dag_stats();
//! assert_eq!((stats.units, stats.open, stats.frames), (1, 1, 1));
//! ```

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    sync::Arc,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Attempt, Input, Instant, JobId, JobSpec, Output, Policy, PolicyStats, WorkerId};

mod frame;
#[cfg(feature = "serde")]
mod snapshot;
mod template;

use frame::{COMPLETE, Frame, HELD, SUBMITTED, Work};
#[cfg(feature = "serde")]
pub use snapshot::DagSnapshot;
pub use template::{DagTemplate, TemplateNode};

/// A plain job with dependencies.
///
/// It is a [`Unit`] of a one-node template, whose one leaf is the job, so other units depend on
/// it by the job's id. Its constructors make the three kinds of node a job can be: a worker job
/// ([`new`](Self::new)), a passthrough ([`passthrough`](Self::passthrough)) and a local job
/// ([`local`](Self::local)).
///
/// A diamond of plain jobs on two single-slot workers: jobs 2 and 3 run side by side once job 1
/// is done, and job 4 waits for both.
///
/// ```
/// use whelm::{
///     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources,
///     Scheduler, WorkerState,
/// };
///
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
/// for w in [1, 2] {
///     dag.handle(
///         Input::Worker(WorkerState::new(w, "cpu", 1, Resources::mem(100))),
///         0.0,
///     );
/// }
/// let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
/// dag.declare(
///     [
///         job(1, vec![]),
///         job(2, vec![1]),
///         job(3, vec![1]),
///         job(4, vec![2, 3]),
///     ],
///     0.0,
/// )
/// .unwrap();
///
/// let start = |job, worker| Output::Start {
///     job,
///     attempt: 1,
///     worker,
/// };
/// assert_eq!(dag.poll(0.0), vec![start(1, 1)]);
/// dag.handle(Input::Done { job: 1, attempt: 1 }, 1.0);
/// assert_eq!(dag.poll(1.0), vec![start(2, 1), start(3, 2)]);
/// dag.handle(Input::Done { job: 2, attempt: 1 }, 2.0);
/// assert!(dag.poll(2.0).is_empty());
/// dag.handle(Input::Done { job: 3, attempt: 1 }, 3.0);
/// assert_eq!(dag.poll(3.0), vec![start(4, 1)]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DagJob {
    /// The job, as it will be submitted to the policy.
    pub spec: JobSpec,
    /// Units that must all complete before this one is ready (see [`Unit::deps`]).
    pub deps: Vec<JobId>,
    /// Relative cost, for ranks. `None` uses [`DagConfig::default_work`].
    pub work_estimate: Option<f64>,
    /// A pure synchronisation point ("group G is done"): when ready it completes by itself
    /// instead of being submitted to the policy. Its `work_estimate` still counts in ranks.
    pub passthrough: bool,
    /// Runs on the caller, not on a worker (registration, loading, commit steps): when ready it is
    /// held, never submitted to the policy, and announced by [`Output::RunLocal`]; report its
    /// completion with [`Input::Done`] and attempt 0. It runs exactly once: it is not retried.
    pub local: bool,
}

impl DagJob {
    /// A job to run on a worker after `deps`, of [`DagConfig::default_work`].
    ///
    /// Job 2 starts once job 1 is done; job 1's rank includes job 2's work.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, WorkerState};
    /// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// dag.handle(
    ///     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))),
    ///     0.0,
    /// );
    /// let spec = |id| JobSpec::new(id, Resources::mem(1), 0);
    /// dag.declare(
    ///     [DagJob::new(spec(1), vec![]), DagJob::new(spec(2), vec![1])],
    ///     0.0,
    /// )
    /// .unwrap();
    /// assert_eq!((dag.rank(1), dag.rank(2)), (Some(2.0), Some(1.0)));
    ///
    /// assert_eq!(
    ///     dag.poll(0.0),
    ///     vec![Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// dag.handle(Input::Done { job: 1, attempt: 1 }, 1.0);
    /// assert_eq!(
    ///     dag.poll(1.0),
    ///     vec![Output::Start {
    ///         job: 2,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn new(spec: JobSpec, deps: Vec<JobId>) -> Self {
        Self {
            spec,
            deps,
            work_estimate: None,
            passthrough: false,
            local: false,
        }
    }

    /// A passthrough job (see [`passthrough`](field@DagJob::passthrough)) of group `group`, worth
    /// `work` in ranks.
    ///
    /// A barrier: job 3 stands for "jobs 1 and 2 are done", so that job 4 can name one dependency
    /// instead of every job before it. It completes without reaching the policy, and with
    /// [`DagConfig::record_passthrough`] it is announced as it does.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, WorkerState};
    /// let config = DagConfig {
    ///     record_passthrough: true,
    ///     ..DagConfig::default()
    /// };
    /// let mut dag = DagScheduler::new(config, Scheduler::new(Config::fifo()));
    /// dag.handle(
    ///     Input::Worker(WorkerState::new(1, "cpu", 2, Resources::mem(100))),
    ///     0.0,
    /// );
    /// let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// let barrier = DagJob::passthrough(3, 0, vec![1, 2], 0.0);
    /// dag.declare(
    ///     [job(1, vec![]), job(2, vec![]), barrier, job(4, vec![3])],
    ///     0.0,
    /// )
    /// .unwrap();
    /// assert_eq!(dag.poll(0.0).len(), 2);
    /// dag.handle(Input::Done { job: 1, attempt: 1 }, 1.0);
    /// dag.handle(Input::Done { job: 2, attempt: 1 }, 1.0);
    /// assert_eq!(
    ///     dag.poll(1.0),
    ///     vec![
    ///         Output::Passed { job: 3 },
    ///         Output::Start {
    ///             job: 4,
    ///             attempt: 1,
    ///             worker: 1
    ///         }
    ///     ]
    /// );
    /// ```
    pub fn passthrough(id: JobId, group: u64, deps: Vec<JobId>, work: f64) -> Self {
        Self {
            spec: JobSpec::new(id, crate::Resources::ZERO, group),
            deps,
            work_estimate: Some(work),
            passthrough: true,
            local: false,
        }
    }

    /// Make it a local job (see [`local`](field@DagJob::local)).
    ///
    /// Job 1 registers something on the caller before job 2 runs on a worker. The caller reports
    /// it done with attempt 0; a worker attempt's number does not complete it.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// dag.declare([job(1, vec![]).local(), job(2, vec![1])], 0.0)
    ///     .unwrap();
    /// assert_eq!(dag.poll(0.0), vec![Output::RunLocal { job: 1 }]);
    /// assert_eq!(dag.stats().waiting, 0);
    ///
    /// dag.handle(Input::Done { job: 1, attempt: 1 }, 1.0);
    /// assert!(dag.poll(1.0).is_empty());
    /// dag.handle(Input::Done { job: 1, attempt: 0 }, 1.0);
    /// assert_eq!(
    ///     dag.poll(1.0),
    ///     vec![Output::Start {
    ///         job: 2,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn local(mut self) -> Self {
        self.local = true;
        self
    }

    /// Set the work estimate.
    ///
    /// The estimate counts in ranks and becomes the submitted [`JobSpec::work`] unless that is set.
    /// Job 1 leads a chain of work 5 then 1; job 3, independent and of default work, ranks below
    /// it.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// dag.declare(
    ///     [
    ///         job(1, vec![]).with_work(5.0),
    ///         job(2, vec![1]),
    ///         job(3, vec![]),
    ///     ],
    ///     0.0,
    /// )
    /// .unwrap();
    /// assert_eq!(
    ///     (dag.rank(1), dag.rank(2), dag.rank(3)),
    ///     (Some(6.0), Some(1.0), Some(1.0))
    /// );
    /// ```
    pub fn with_work(mut self, work: f64) -> Self {
        self.work_estimate = Some(work);
        self
    }
}

/// A node of the coarse graph: an instance of a [`DagTemplate`], declared with
/// [`DagScheduler::declare`].
///
/// Its leaf `k` (see [`DagTemplate`]) is the job `base + k`. Its template's sources wait for every
/// unit in `deps`; the unit completes once all of its dependencies have and every node of its
/// template has. Other units depend on it by naming `id`. A one-node unit with `id == base` is a
/// plain job: its id is its leaf's.
///
/// Everything ranks need is known at declaration; the per-node state is materialised only once the
/// dependencies complete, and dropped when the unit does.
///
/// A unit whose template nests another: node 1 of the outer template is a two-job chain, whose
/// leaves take ids 1 and 2 between the outer jobs 0 and 3. The nested unit gets its own frame of
/// state only while it runs.
///
/// ```
/// use std::sync::Arc;
///
/// use whelm::{
///     Config, DagConfig, DagScheduler, DagTemplate, Input, JobSpec, Output, Policy, Resources,
///     Scheduler, TemplateNode, Unit, WorkerState,
/// };
///
/// let chain = Arc::new(DagTemplate::new(2, [(0, 1)]).unwrap());
/// let outer = DagTemplate::with_nodes(
///     vec![
///         TemplateNode::Job(1.0),
///         TemplateNode::Unit(chain),
///         TemplateNode::Job(1.0),
///     ],
///     [(0, 1), (1, 2)],
/// )
/// .unwrap();
/// assert_eq!(outer.leaves(), 4);
///
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
/// dag.handle(
///     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))),
///     0.0,
/// );
/// let spec = JobSpec::new(0, Resources::mem(1), 0);
/// dag.declare([Unit::new(50, 0, Arc::new(outer), spec, vec![])], 0.0)
///     .unwrap();
/// assert_eq!(
///     [0, 1, 2, 3].map(|j| dag.rank(j).unwrap()),
///     [4.0, 3.0, 2.0, 1.0]
/// );
///
/// let mut order = Vec::new();
/// let mut t = 0.0;
/// loop {
///     let out = dag.poll(t);
///     let [Output::Start { job, attempt, .. }] = out[..] else {
///         break;
///     };
///     order.push((job, dag.dag_stats().frames));
///     t += 1.0;
///     dag.handle(Input::Done { job, attempt }, t);
/// }
/// assert_eq!(order, [(0, 1), (1, 2), (2, 2), (3, 1)]);
/// assert_eq!(dag.dag_stats().units, 0);
/// ```
#[derive(Clone, Debug)]
pub struct Unit {
    /// The name dependents use. Unless the unit is a plain job, it must lie outside
    /// `base..base + leaves`.
    pub id: JobId,
    /// The id of leaf 0.
    pub base: JobId,
    /// The shared structure.
    pub template: Arc<DagTemplate>,
    /// Units that must complete before the template's sources are ready. A dependency may name a
    /// unit not declared yet (a forward reference): it is pending until declared and completed.
    /// A dependency on a unit that already completed is satisfied.
    pub deps: Vec<JobId>,
    /// Every leaf's spec, with `id` replaced by the leaf's and `work`, if unset, by its work.
    pub spec: JobSpec,
    /// Multiplies every leaf's work. `None` uses [`DagConfig::default_work`].
    pub scale: Option<f64>,
    /// Leaves' work, spec and label come from the scheduler's [`NodeSource`] rather than from the
    /// template and `spec` alone.
    pub sourced: bool,
    /// Leaves already complete (e.g. restored from a checkpoint): they never run, and their
    /// successors start with those dependencies met. Need not be closed under predecessors: an
    /// incomplete predecessor of a complete leaf still runs, and its completion does not touch the
    /// complete leaf.
    pub completed: Vec<u32>,
}

impl Unit {
    /// A unit of `template` named `id`, its leaves at `base..`, after `deps`, at scale 1.
    ///
    /// Unit 10 runs three independent jobs as 100, 101 and 102; job 5 depends on the unit as a
    /// whole and starts once all three are done.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, DagTemplate, Input, JobSpec, Output,
    /// #     Policy, Resources, Scheduler, Unit, WorkerState};
    /// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// dag.handle(Input::Worker(WorkerState::new(1, "cpu", 3, Resources::mem(100))), 0.0);
    /// let spec = |id| JobSpec::new(id, Resources::mem(1), 0);
    /// let three = Arc::new(DagTemplate::new(3, []).unwrap());
    /// dag.declare([Unit::new(10, 100, three, spec(0), vec![])], 0.0).unwrap();
    /// dag.declare([DagJob::new(spec(5), vec![10])], 0.0).unwrap();
    ///
    /// let start = |job| Output::Start { job, attempt: 1, worker: 1 };
    /// assert_eq!(dag.poll(0.0), vec![start(100), start(101), start(102)]);
    /// for job in [100, 101, 102] {
    ///     dag.handle(Input::Done { job, attempt: 1 }, 1.0);
    /// }
    /// assert_eq!(dag.poll(1.0), vec![start(5)]);
    /// ```
    pub fn new(
        id: JobId,
        base: JobId,
        template: Arc<DagTemplate>,
        spec: JobSpec,
        deps: Vec<JobId>,
    ) -> Self {
        Self {
            id,
            base,
            template,
            deps,
            spec,
            scale: Some(1.0),
            sourced: false,
            completed: Vec::new(),
        }
    }

    /// This unit with its leaves' work multiplied by `scale`.
    ///
    /// One template serves units of different sizes: here a two-job chain at three times its
    /// template's work. The scale reaches ranks and each submitted job's [`JobSpec::work`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagScheduler, DagTemplate, JobSpec, Resources, Scheduler,
    /// #     Unit};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// let chain = Arc::new(DagTemplate::new(2, [(0, 1)]).unwrap());
    /// let spec = JobSpec::new(0, Resources::mem(1), 0);
    /// dag.declare([Unit::new(10, 100, chain, spec, vec![]).with_scale(3.0)], 0.0).unwrap();
    /// assert_eq!((dag.rank(100), dag.rank(101), dag.rank(10)), (Some(6.0), Some(3.0), Some(6.0)));
    /// ```
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    /// This unit with its leaves described by the scheduler's [`NodeSource`].
    ///
    /// The source here gives leaf `k` work `k + 1`, which the template alone could not say for
    /// every unit; the [`NodeSource`] docs show the rest of the trait.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagScheduler, DagTemplate, JobId, JobSpec, NodeSource,
    /// #     Resources, Scheduler, Unit};
    /// struct Growing;
    /// impl NodeSource for Growing {
    ///     fn work(&self, _unit: JobId, leaf: u32) -> f64 {
    ///         f64::from(leaf + 1)
    ///     }
    /// }
    ///
    /// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()))
    ///     .with_source(Arc::new(Growing));
    /// let three = Arc::new(DagTemplate::new(3, []).unwrap());
    /// let spec = JobSpec::new(0, Resources::mem(1), 0);
    /// dag.declare([Unit::new(10, 100, three, spec, vec![]).sourced()], 0.0)
    ///     .unwrap();
    /// assert_eq!(
    ///     [100, 101, 102].map(|j| dag.rank(j).unwrap()),
    ///     [1.0, 2.0, 3.0]
    /// );
    /// ```
    pub fn sourced(mut self) -> Self {
        self.sourced = true;
        self
    }

    /// This unit with leaves `completed` already complete.
    ///
    /// Resuming a three-job chain whose first job finished before a restart: the run picks up at
    /// leaf 1, job 101.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagScheduler, DagTemplate, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, Unit, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// let chain = Arc::new(DagTemplate::new(3, [(0, 1), (1, 2)]).unwrap());
    /// let spec = JobSpec::new(0, Resources::mem(1), 0);
    /// let unit = Unit::new(10, 100, chain, spec, vec![]).with_completed(vec![0]);
    /// dag.declare([unit], 0.0).unwrap();
    /// assert_eq!(dag.poll(0.0), vec![Output::Start { job: 101, attempt: 1, worker: 1 }]);
    /// assert_eq!(dag.dag_stats().pending, 1);
    /// ```
    pub fn with_completed(mut self, completed: Vec<u32>) -> Self {
        self.completed = completed;
        self
    }
}

impl From<DagJob> for Unit {
    /// A unit of the one-node template of the job's kind.
    fn from(j: DagJob) -> Self {
        let template = if j.passthrough {
            &template::PASS
        } else if j.local {
            &template::LOCAL
        } else {
            &template::JOB
        };
        Self {
            id: j.spec.id,
            base: j.spec.id,
            template: Arc::clone(template),
            deps: j.deps,
            spec: j.spec,
            scale: j.work_estimate,
            sourced: false,
            completed: Vec::new(),
        }
    }
}

/// Per-leaf data of [`sourced`](Unit::sourced) units, computed on demand rather than stored.
///
/// A unit costs the same however many leaves it has until it materialises. Units of one template
/// can then differ leaf by leaf (work, demand, which leaves are no-ops) without a template each.
///
/// A source for a three-step chain in which step 1 is a no-op for this unit and step 2 needs a
/// GPU worker. Step 1 completes by itself, the label names the steps in
/// [`explain`](Policy::explain), and the rank of step 0 is its work plus step 2's.
///
/// ```
/// use std::sync::Arc;
///
/// use whelm::{
///     Config, DagConfig, DagScheduler, DagTemplate, Input, JobId, JobSpec, NodeSource, Output,
///     Policy, Resources, Scheduler, Unit, WorkerState,
/// };
///
/// struct Steps;
/// impl NodeSource for Steps {
///     fn work(&self, _unit: JobId, leaf: u32) -> f64 {
///         f64::from(leaf + 1)
///     }
///
///     fn passthrough(&self, _unit: JobId, leaf: u32) -> bool {
///         leaf == 1
///     }
///
///     fn spec(&self, _unit: JobId, leaf: u32, spec: &mut JobSpec) {
///         if leaf == 2 {
///             *spec = spec.clone().require_class("gpu");
///         }
///     }
///
///     fn label(&self, unit: JobId, leaf: u32) -> Option<String> {
///         Some(format!("unit {unit} step {leaf}"))
///     }
/// }
///
/// let config = DagConfig {
///     record_passthrough: true,
///     ..DagConfig::default()
/// };
/// let mut dag =
///     DagScheduler::new(config, Scheduler::new(Config::fifo())).with_source(Arc::new(Steps));
/// dag.handle(
///     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))),
///     0.0,
/// );
/// dag.handle(
///     Input::Worker(WorkerState::new(2, "gpu", 1, Resources::mem(100))),
///     0.0,
/// );
/// let chain = Arc::new(DagTemplate::new(3, [(0, 1), (1, 2)]).unwrap());
/// let spec = JobSpec::new(0, Resources::mem(1), 0);
/// dag.declare([Unit::new(10, 100, chain, spec, vec![]).sourced()], 0.0)
///     .unwrap();
/// assert_eq!((dag.rank(100), dag.rank(102)), (Some(4.0), Some(3.0)));
///
/// assert_eq!(
///     dag.poll(0.0),
///     vec![Output::Start {
///         job: 100,
///         attempt: 1,
///         worker: 1
///     }]
/// );
/// assert_eq!(
///     dag.explain(102).unwrap(),
///     "[unit 10 step 2] job 102 waits for 1 dependency within its unit"
/// );
/// dag.handle(
///     Input::Done {
///         job: 100,
///         attempt: 1,
///     },
///     1.0,
/// );
/// assert_eq!(
///     dag.poll(1.0),
///     vec![
///         Output::Passed { job: 101 },
///         Output::Start {
///             job: 102,
///             attempt: 1,
///             worker: 2
///         }
///     ]
/// );
/// ```
pub trait NodeSource: Send + Sync {
    /// The work of leaf `leaf` of unit `unit`, before the unit's scale. Read when the unit is
    /// declared (its critical path), and again as it materialises and submits leaves, so it must
    /// not change meanwhile.
    fn work(&self, unit: JobId, leaf: u32) -> f64;

    /// Whether leaf `leaf` of unit `unit` does nothing in that unit, though other units of the
    /// template may run it: it then acts as a [`TemplateNode::Pass`] of no work, completing by
    /// itself once ready (announced by [`Output::Passed`] with
    /// [`DagConfig::record_passthrough`]), and `work` is not read for it. Read when `work` is, so
    /// it must not change meanwhile either. Default: no leaf.
    fn passthrough(&self, _unit: JobId, _leaf: u32) -> bool {
        false
    }

    /// Finish the spec of leaf `leaf` of unit `unit` before it is submitted (e.g. its demand). It
    /// arrives as the unit's spec with the leaf's id, work and rank. Default: unchanged.
    fn spec(&self, _unit: JobId, _leaf: u32, _spec: &mut JobSpec) {}

    /// The leaf's name in [`explain`](Policy::explain) messages. Default: none.
    fn label(&self, _unit: JobId, _leaf: u32) -> Option<String> {
        None
    }
}

/// The scheduler's [`NodeSource`], opaque to `Debug`.
#[derive(Clone)]
struct Source(Arc<dyn NodeSource>);

impl std::fmt::Debug for Source {
    /// The source is opaque.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NodeSource(..)")
    }
}

/// Errors from [`DagScheduler::declare`], [`DagScheduler::close`] and building a
/// [`DagTemplate`].
///
/// A failed declaration changes nothing. Each variant, as it is returned:
///
/// ```
/// use std::sync::Arc;
///
/// use whelm::{
///     Config, DagConfig, DagError, DagJob, DagScheduler, DagTemplate, JobSpec, Resources,
///     Scheduler, Unit,
/// };
///
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
/// let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
/// let three = Arc::new(DagTemplate::new(3, []).unwrap());
/// let spec = JobSpec::new(0, Resources::mem(1), 0);
/// // Unit 99 owns ids 10, 11 and 12.
/// dag.declare([Unit::new(99, 10, three, spec, vec![])], 0.0)
///     .unwrap();
///
/// let cycle = [job(1, vec![2]), job(2, vec![1])];
/// assert_eq!(dag.declare(cycle, 0.0), Err(DagError::Cycle { job: 1 }));
/// assert_eq!(
///     DagTemplate::new(2, [(0, 1), (1, 0)]).unwrap_err(),
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

/// Configuration for [`DagScheduler`].
///
/// Without [`track_ranks`](Self::track_ranks), a job's rank covers its own unit only; with it,
/// the chain of dependents counts too. [`default_work`](Self::default_work) sizes jobs declared
/// without an estimate.
///
/// ```
/// use whelm::{Config, DagConfig, DagJob, DagScheduler, JobSpec, Resources, Scheduler};
///
/// let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
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

/// Where a unit is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnitState {
    /// Named as a dependency, not declared yet.
    Undeclared,
    /// Declared, some dependency not complete.
    Pending,
    /// Entered: its per-node state is materialised.
    Open,
}

/// A live unit.
#[derive(Clone, Debug)]
struct UnitRec {
    id: JobId,
    base: JobId,
    state: UnitState,
    /// Closed early before it was entered: it completes as soon as it is.
    closed: bool,
    sourced: bool,
    /// `None` while undeclared.
    template: Option<Arc<DagTemplate>>,
    spec: JobSpec,
    scale: f64,
    /// The critical path of the leaves' work before `scale`.
    span: f64,
    /// The longest rank among dependents (approximate): the rank below the unit.
    tail: f64,
    /// Dependencies not complete.
    preds: Vec<u32>,
    /// Dependents.
    succs: Vec<u32>,
    /// Leaves already complete, sorted.
    completed: Box<[u32]>,
    /// The materialised top frame, while open.
    frame: u32,
}

impl UnitRec {
    /// A forward reference.
    fn undeclared(id: JobId) -> Self {
        Self {
            id,
            base: id,
            state: UnitState::Undeclared,
            closed: false,
            sourced: false,
            template: None,
            spec: JobSpec::new(id, crate::Resources::ZERO, 0),
            scale: 0.0,
            span: 0.0,
            tail: 0.0,
            preds: Vec::new(),
            succs: Vec::new(),
            completed: Box::new([]),
            frame: 0,
        }
    }

    /// Upward rank: the unit's critical path plus the rank below it.
    fn top(&self) -> f64 {
        self.scale * self.span + self.tail
    }

    /// Number of leaves (one for a forward reference, whose id is reserved).
    fn leaves(&self) -> u64 {
        self.template.as_ref().map_or(1, |t| t.leaves() as u64)
    }

    /// Whether the unit is a plain job: its id is its single leaf's.
    fn plain(&self) -> bool {
        self.id == self.base && self.template.as_ref().is_some_and(|t| t.leaves() == 1)
    }

    /// Whether leaf `leaf` was declared complete.
    fn leaf_completed(&self, leaf: u32) -> bool {
        self.completed.binary_search(&leaf).is_ok()
    }

    /// Number of leaves in `range` declared complete.
    fn completed_in(&self, range: std::ops::Range<u32>) -> usize {
        let lo = self.completed.partition_point(|&c| c < range.start);
        let hi = self.completed.partition_point(|&c| c < range.end);
        hi - lo
    }
}

/// What a job id names.
#[derive(Clone, Copy, Debug)]
enum Loc {
    /// A unit, by its id (other than a plain job).
    Unit(u32),
    /// Leaf `leaf` of a unit.
    Leaf { unit: u32, leaf: u32 },
}

impl Loc {
    /// The unit.
    fn unit(self) -> u32 {
        match self {
            Self::Unit(u) | Self::Leaf { unit: u, .. } => u,
        }
    }
}

/// Counters describing the DAG layer's state, from [`DagScheduler::dag_stats`].
///
/// Job 2 is declared after job 1 before job 1 is: job 1 is an undeclared forward reference, and
/// job 2 a pending unit with no materialised state. Declaring job 1 enters and submits it.
///
/// ```
/// use whelm::{Config, DagConfig, DagJob, DagScheduler, DagStats, JobSpec, Resources, Scheduler};
///
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
/// let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
/// dag.declare([job(2, vec![1])], 0.0).unwrap();
/// let s = dag.dag_stats();
/// assert_eq!(
///     (s.units, s.undeclared, s.edges, s.pending, s.frames),
///     (1, 1, 1, 1, 0)
/// );
///
/// dag.declare([job(1, vec![])], 0.0).unwrap();
/// let s = dag.dag_stats();
/// assert_eq!(
///     (s.units, s.open, s.undeclared, s.pending, s.submitted),
///     (2, 1, 0, 1, 1)
/// );
/// assert_eq!((s.frames, s.nodes), (1, 1));
/// assert!(s.node_bytes > 0);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DagStats {
    /// Declared units not complete.
    pub units: usize,
    /// Of those, the entered ones, whose per-node state is materialised.
    pub open: usize,
    /// Units named as dependencies but not declared yet.
    pub undeclared: usize,
    /// Dependency edges between live units.
    pub edges: usize,
    /// Leaves whose dependencies are not all complete, materialised or not.
    pub pending: usize,
    /// Ready jobs waiting for `release` or for the caller to run them locally.
    pub held: usize,
    /// Jobs handed to the policy and not complete.
    pub submitted: usize,
    /// Completed unit ids remembered so that later dependencies on them are satisfied.
    pub completed_remembered: usize,
    /// Materialised units and substituted units within them.
    pub frames: usize,
    /// Template nodes with materialised state.
    pub nodes: usize,
    /// Approximate memory of the materialised state, bytes.
    pub node_bytes: usize,
}

/// A dependency layer in front of a [`Policy`], itself a [`Policy`].
///
/// The graph is a coarse DAG of [`Unit`]s, each an instance of a [`DagTemplate`] whose nodes are
/// jobs or, recursively, units of other templates. Units are declared with their dependencies,
/// possibly long before they are ready; a unit's per-node state (a counter per template node) is
/// allocated when its dependencies complete and freed when it completes, so memory follows the
/// frontier, while ranks are computed from the whole declared graph. A job is submitted to the
/// inner policy when its last dependency completes. Inputs go to the inner policy; an
/// [`Input::Done`] of a live attempt also completes the job here. [`Policy::poll`] returns the
/// inner policy's outputs and this layer's own: [`Output::RunLocal`], [`Output::Ready`] and
/// [`Output::Passed`], which [`announcements`](Self::announcements) drains alone.
///
/// A job the inner policy gives up on ([`Output::GaveUp`], passed through) is held again: its
/// dependents stay pending until the caller [`release`](Self::release)s it (another round of
/// attempts) or [`cancel`](Self::cancel)s it.
///
/// With one attempt per job, job 1's failure is a give-up; releasing it runs it again, and its
/// dependent follows.
///
/// ```
/// use whelm::{
///     Config, DagConfig, DagJob, DagScheduler, FailKind, GaveUp, Input, JobSpec, Output, Policy,
///     Resources, RetryConfig, Scheduler, WorkerState,
/// };
///
/// let config = Config {
///     retry: RetryConfig { max_attempts: 1 },
///     ..Config::fifo()
/// };
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(config));
/// dag.handle(
///     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))),
///     0.0,
/// );
/// let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
/// dag.declare([job(1, vec![]), job(2, vec![1])], 0.0).unwrap();
/// assert_eq!(
///     dag.poll(0.0),
///     vec![Output::Start {
///         job: 1,
///         attempt: 1,
///         worker: 1
///     }]
/// );
///
/// let why = "segfault".to_string();
/// dag.handle(
///     Input::Failed {
///         job: 1,
///         attempt: 1,
///         kind: FailKind::Other,
///         why,
///     },
///     1.0,
/// );
/// let out = dag.poll(1.0);
/// assert!(matches!(out[..], [Output::GaveUp(GaveUp { job: 1, .. })]));
/// assert_eq!(
///     dag.explain(1).unwrap(),
///     "job 1 is ready and held until release"
/// );
///
/// assert!(dag.release(1, 2.0));
/// assert_eq!(
///     dag.poll(2.0),
///     vec![Output::Start {
///         job: 1,
///         attempt: 1,
///         worker: 1
///     }]
/// );
/// dag.handle(Input::Done { job: 1, attempt: 1 }, 3.0);
/// assert_eq!(
///     dag.poll(3.0),
///     vec![Output::Start {
///         job: 2,
///         attempt: 1,
///         worker: 1
///     }]
/// );
/// ```
#[derive(Clone, Debug)]
pub struct DagScheduler<P> {
    config: DagConfig,
    policy: P,
    source: Option<Source>,
    units: Vec<Option<UnitRec>>,
    free_units: Vec<u32>,
    /// Unit id -> slot, forward references included.
    ids: BTreeMap<JobId, u32>,
    /// Leaf 0's id -> slot, for declared units other than plain jobs.
    ranges: BTreeMap<JobId, u32>,
    frames: Vec<Option<Frame>>,
    free_frames: Vec<u32>,
    /// Readiness events not processed yet.
    work: VecDeque<Work>,
    completed: HashSet<JobId>,
    completed_floor: JobId,
    /// This layer's outputs not yet returned by `poll`.
    outbox: Vec<Output>,
    /// Live attempts of the inner policy's jobs, from its outputs and the inputs: an
    /// [`Input::Done`] completes a job here only if its attempt is live.
    live: HashMap<JobId, Vec<(Attempt, WorkerId)>>,
    /// Running jobs of units closed early: their completion only frees their resources.
    ignored: HashSet<JobId>,
    /// Stops the inner policy emits for attempts the caller has already reported (an ignored
    /// job's failure, turned into a cancellation): not passed on.
    quiet: HashSet<(JobId, Attempt)>,
    now: Instant,
}

impl<P: Policy> DagScheduler<P> {
    /// A DAG layer in front of `policy`.
    ///
    /// Jobs without dependencies need not be declared: an [`Input::Submit`] goes straight to the
    /// inner policy.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, WorkerState};
    /// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// dag.handle(
    ///     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))),
    ///     0.0,
    /// );
    /// dag.handle(Input::Submit(JobSpec::new(7, Resources::mem(1), 0)), 0.0);
    /// assert_eq!(
    ///     dag.poll(0.0),
    ///     vec![Output::Start {
    ///         job: 7,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// assert_eq!(dag.rank(7), None);
    /// ```
    pub fn new(config: DagConfig, policy: P) -> Self {
        Self {
            config,
            policy,
            source: None,
            units: Vec::new(),
            free_units: Vec::new(),
            ids: BTreeMap::new(),
            ranges: BTreeMap::new(),
            frames: Vec::new(),
            free_frames: Vec::new(),
            work: VecDeque::new(),
            completed: HashSet::new(),
            completed_floor: 0,
            outbox: Vec::new(),
            live: HashMap::new(),
            ignored: HashSet::new(),
            quiet: HashSet::new(),
            now: 0.0,
        }
    }

    /// This scheduler, describing [`sourced`](Unit::sourced) units' leaves with `source`.
    ///
    /// Declaring a sourced unit without one panics. See [`NodeSource`] for an example.
    pub fn with_source(mut self, source: Arc<dyn NodeSource>) -> Self {
        self.source = Some(Source(source));
        self
    }

    /// The wrapped policy.
    ///
    /// Its own view of the jobs submitted so far: here job 2 is not there, since it waits for
    /// job 1.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// # let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// dag.declare([job(1, vec![]), job(2, vec![1])], 0.0).unwrap();
    /// assert_eq!(dag.policy().stats().waiting, 1);
    /// ```
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// The wrapped policy, mutably. Inputs given to it directly bypass this layer's bookkeeping;
    /// jobs outside the DAG are submitted with [`Input::Submit`] through this layer instead.
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    /// The live unit in slot `u`.
    fn unit(&self, u: u32) -> &UnitRec {
        self.units[u as usize].as_ref().expect("a live unit")
    }

    /// The live unit in slot `u`, mutably.
    fn unit_mut(&mut self, u: u32) -> &mut UnitRec {
        self.units[u as usize].as_mut().expect("a live unit")
    }

    /// The node source; panics without one.
    fn src(&self) -> &dyn NodeSource {
        &*self
            .source
            .as_ref()
            .expect("a sourced unit needs DagScheduler::with_source")
            .0
    }

    /// Whether `job` is known to have completed as a unit (remembered, or below the floor).
    fn is_completed(&self, job: JobId) -> bool {
        job < self.completed_floor || self.completed.contains(&job)
    }

    /// What `job` names among live units.
    fn locate(&self, job: JobId) -> Option<Loc> {
        if let Some(&u) = self.ids.get(&job) {
            return Some(if self.unit(u).plain() {
                Loc::Leaf { unit: u, leaf: 0 }
            } else {
                Loc::Unit(u)
            });
        }
        let (&base, &u) = self.ranges.range(..=job).next_back()?;
        (job - base < self.unit(u).leaves()).then(|| Loc::Leaf {
            unit: u,
            leaf: (job - base) as u32,
        })
    }

    /// The live unit containing `job` as a leaf or named `job`.
    fn unit_of(&self, job: JobId) -> Option<u32> {
        self.locate(job).map(Loc::unit)
    }

    /// Put a unit in a free slot.
    fn alloc_unit(&mut self, rec: UnitRec) -> u32 {
        match self.free_units.pop() {
            Some(u) => {
                self.units[u as usize] = Some(rec);
                u
            }
            None => {
                self.units.push(Some(rec));
                (self.units.len() - 1) as u32
            }
        }
    }

    /// Drop unit `u` from the maps and free its slot.
    fn free_unit(&mut self, u: u32) -> UnitRec {
        let rec = self.units[u as usize].take().expect("a live unit");
        self.ids.remove(&rec.id);
        if self.ranges.get(&rec.base) == Some(&u) {
            self.ranges.remove(&rec.base);
        }
        self.free_units.push(u);
        rec
    }

    /// Check a unit against the live ones, and enter it: fill its forward reference or take a new
    /// slot. Returns the slot and whether it was a forward reference.
    fn admit(&mut self, unit: &Unit) -> Result<(u32, bool), DagError> {
        let (id, base) = (unit.id, unit.base);
        let leaves = unit.template.leaves() as u64;
        let end = base.checked_add(leaves).expect("unit id range overflows");
        let existing = self.ids.get(&id).copied();
        if self.is_completed(id)
            || existing.is_some_and(|u| self.unit(u).state != UnitState::Undeclared)
        {
            return Err(DagError::Duplicate(id));
        }
        if let Some((&b, &r)) = self.ranges.range(..=id).next_back()
            && id - b < self.unit(r).leaves()
        {
            return Err(DagError::Overlap(id));
        }
        let plain = id == base && leaves == 1;
        if !plain && leaves > 0 {
            if (base..end).contains(&id) {
                return Err(DagError::Overlap(id));
            }
            if let Some((&b, &r)) = self.ranges.range(..end).next_back()
                && b + self.unit(r).leaves() > base
            {
                return Err(DagError::Overlap(b.max(base)));
            }
            if let Some((&j, _)) = self.ids.range(base..end).next() {
                return Err(DagError::Overlap(j));
            }
        }
        let mut completed = unit.completed.clone();
        completed.sort_unstable();
        completed.dedup();
        assert!(
            completed.last().is_none_or(|&c| u64::from(c) < leaves),
            "completed leaf out of range"
        );
        let template = unit.template.clone();
        let span = if unit.sourced {
            frame::sourced_bottom_levels(self.src(), id, &template, 0)
                .into_iter()
                .fold(0.0, f64::max)
        } else {
            template.span()
        };
        let u = match existing {
            Some(u) => u,
            None => {
                let u = self.alloc_unit(UnitRec::undeclared(id));
                self.ids.insert(id, u);
                u
            }
        };
        let scale = unit.scale.unwrap_or(self.config.default_work);
        let rec = self.unit_mut(u);
        rec.base = base;
        rec.state = UnitState::Pending;
        rec.sourced = unit.sourced;
        rec.template = Some(template);
        rec.spec = unit.spec.clone();
        rec.scale = scale;
        rec.span = span;
        rec.completed = completed.into_boxed_slice();
        if !plain && leaves > 0 {
            self.ranges.insert(base, u);
        }
        Ok((u, existing.is_some()))
    }

    /// Undo a rejected declaration: its edges, its units and the forward references it created.
    fn rollback(&mut self, batch: &[(u32, bool)], created: &[u32]) {
        for &(u, _) in batch {
            for p in std::mem::take(&mut self.unit_mut(u).preds) {
                self.unit_mut(p).succs.retain(|&s| s != u);
            }
        }
        for &c in created {
            self.free_unit(c);
        }
        for &(u, was_reference) in batch {
            if was_reference {
                let base = self.unit(u).base;
                if self.ranges.get(&base) == Some(&u) {
                    self.ranges.remove(&base);
                }
                let rec = self.unit_mut(u);
                let fresh = UnitRec::undeclared(rec.id);
                rec.base = fresh.base;
                rec.state = fresh.state;
                rec.sourced = false;
                rec.template = None;
                rec.spec = fresh.spec;
                rec.scale = 0.0;
                rec.span = 0.0;
                rec.completed = fresh.completed;
            } else {
                self.free_unit(u);
            }
        }
    }

    /// A unit on a cycle reachable from `starts`, if any (iterative three-colour DFS along
    /// dependent edges; only the part of the graph reachable from the new units is visited).
    fn find_cycle(&self, starts: &[(u32, bool)]) -> Option<JobId> {
        // 1 = on the DFS stack, 2 = finished.
        let mut colour: HashMap<u32, u8> = HashMap::new();
        for &(s, _) in starts {
            if colour.contains_key(&s) {
                continue;
            }
            colour.insert(s, 1);
            let mut stack = vec![(s, self.unit(s).succs.clone())];
            while let Some((node, children)) = stack.last_mut() {
                match children.pop() {
                    Some(c) => match colour.get(&c) {
                        Some(1) => return Some(self.unit(c).id),
                        Some(_) => {}
                        None => {
                            colour.insert(c, 1);
                            stack.push((c, self.unit(c).succs.clone()));
                        }
                    },
                    None => {
                        colour.insert(*node, 2);
                        stack.pop();
                    }
                }
            }
        }
        None
    }

    /// Raise dependencies' ranks after `start`'s rank grew.
    fn propagate_rank(&mut self, start: u32) {
        let eps = self.config.rank_epsilon.max(0.0);
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            let top = self.unit(n).top();
            for i in 0..self.unit(n).preds.len() {
                let p = self.unit(n).preds[i];
                let rec = self.unit_mut(p);
                let old = rec.top();
                let cand = rec.scale * rec.span + top;
                if cand > old * (1.0 + eps) && cand > old {
                    rec.tail = top;
                    stack.push(p);
                }
            }
        }
    }

    /// Declare units (or plain [`DagJob`]s).
    ///
    /// The graph grows during the run; dependencies may be forward references. Rejects (and
    /// leaves no trace of) a batch that would create a cycle, redeclare a unit or overlap another
    /// unit's ids. Units whose dependencies are all complete are entered at once.
    ///
    /// Job 3 names job 2 before it exists; declaring job 2 after job 3 would close a cycle, so
    /// it is refused and the graph is as it was. A dependency on a completed job is met.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagError, DagJob, DagScheduler, Input, JobSpec, Output,
    /// #     Policy, Resources, Scheduler, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// # let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// dag.declare([job(1, vec![]), job(3, vec![2])], 0.0).unwrap();
    /// assert_eq!(dag.explain(3).unwrap(), "job 3 waits for 1 dependency [2]");
    /// assert_eq!(
    ///     dag.explain(2).unwrap(),
    ///     "unit 2 is not declared yet (named as a dependency of 1 unit(s))"
    /// );
    /// let before = dag.dag_stats();
    /// assert_eq!(
    ///     dag.declare([job(2, vec![3])], 0.0),
    ///     Err(DagError::Cycle { job: 2 })
    /// );
    /// assert_eq!(dag.dag_stats(), before);
    ///
    /// assert_eq!(
    ///     dag.poll(0.0),
    ///     vec![Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// dag.handle(Input::Done { job: 1, attempt: 1 }, 1.0);
    /// dag.declare([job(2, vec![1])], 1.0).unwrap();
    /// assert_eq!(
    ///     dag.poll(1.0),
    ///     vec![Output::Start {
    ///         job: 2,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn declare<U: Into<Unit>>(
        &mut self,
        units: impl IntoIterator<Item = U>,
        now: Instant,
    ) -> Result<(), DagError> {
        self.now = now;
        let units: Vec<Unit> = units.into_iter().map(Into::into).collect();
        let mut batch: Vec<(u32, bool)> = Vec::with_capacity(units.len());
        let mut created = Vec::new();
        for unit in &units {
            match self.admit(unit) {
                Ok(entry) => batch.push(entry),
                Err(e) => {
                    self.rollback(&batch, &created);
                    return Err(e);
                }
            }
        }
        for (unit, &(u, _)) in units.iter().zip(&batch) {
            let mut deps = unit.deps.clone();
            deps.sort_unstable();
            deps.dedup();
            for d in deps {
                if self.is_completed(d) {
                    continue;
                }
                let p = match self.locate(d) {
                    Some(Loc::Unit(p)) => p,
                    Some(Loc::Leaf { unit: p, .. }) if self.unit(p).plain() => p,
                    Some(Loc::Leaf { .. }) => {
                        self.rollback(&batch, &created);
                        return Err(DagError::Overlap(d));
                    }
                    None => {
                        let p = self.alloc_unit(UnitRec::undeclared(d));
                        self.ids.insert(d, p);
                        created.push(p);
                        p
                    }
                };
                self.unit_mut(p).succs.push(u);
                self.unit_mut(u).preds.push(p);
            }
        }
        if let Some(job) = self.find_cycle(&batch) {
            self.rollback(&batch, &created);
            return Err(DagError::Cycle { job });
        }
        if self.config.track_ranks {
            for &(u, _) in &batch {
                self.propagate_rank(u);
            }
        }
        for &(u, _) in &batch {
            if self.unit(u).preds.is_empty() {
                self.work.push_back(Work::Open(u));
            }
        }
        self.settle(now);
        Ok(())
    }

    /// Drain this layer's own announcements without polling the inner policy.
    ///
    /// These are [`Output::RunLocal`], [`Output::Ready`] and [`Output::Passed`]; draining them
    /// lets the caller act on them (release, declare, close) before anything is placed in the
    /// same instant. The next [`poll`](Policy::poll) returns the announcements made since, in
    /// order, ahead of the inner policy's outputs.
    ///
    /// Jobs 1 and 2 are ready together; the caller releases job 2 first, so it takes the one
    /// slot.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, WorkerState};
    /// # let config = DagConfig { auto_submit: false, ..DagConfig::default() };
    /// # let mut dag = DagScheduler::new(config, Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// # let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// dag.declare([job(1, vec![]), job(2, vec![])], 0.0).unwrap();
    /// assert_eq!(
    ///     dag.announcements(),
    ///     vec![Output::Ready { job: 1 }, Output::Ready { job: 2 }]
    /// );
    /// assert!(dag.announcements().is_empty());
    /// assert!(dag.release(2, 0.0) && dag.release(1, 0.0));
    /// assert_eq!(
    ///     dag.poll(0.0),
    ///     vec![Output::Start {
    ///         job: 2,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn announcements(&mut self) -> Vec<Output> {
        std::mem::take(&mut self.outbox)
    }

    /// Submit a ready, held job to the policy.
    ///
    /// Jobs are held when ready without [`DagConfig::auto_submit`], and after the policy gives up
    /// on them (see [`DagScheduler`]). Returns false if the job is not held, so a second release
    /// does nothing.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, WorkerState};
    /// # let config = DagConfig { auto_submit: false, ..DagConfig::default() };
    /// # let mut dag = DagScheduler::new(config, Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// # let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// dag.declare([job(1, vec![]), job(2, vec![1])], 0.0).unwrap();
    /// assert_eq!(dag.poll(0.0), vec![Output::Ready { job: 1 }]);
    /// assert!(!dag.release(2, 0.0));
    /// assert!(dag.release(1, 0.0));
    /// assert!(!dag.release(1, 0.0));
    /// assert_eq!(
    ///     dag.poll(0.0),
    ///     vec![Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// dag.handle(Input::Done { job: 1, attempt: 1 }, 1.0);
    /// assert_eq!(dag.poll(1.0), vec![Output::Ready { job: 2 }]);
    /// ```
    pub fn release(&mut self, job: JobId, now: Instant) -> bool {
        self.now = now;
        match self.leaf_node(job) {
            Some((f, i))
                if self.frame(f).counter[i] == HELD
                    && matches!(self.frame(f).template.node(i), TemplateNode::Job(_)) =>
            {
                self.submit(f, i, now);
                true
            }
            _ => false,
        }
    }

    /// Forget remembered completed ids below `floor`, and treat every id below `floor` as
    /// completed from now on.
    ///
    /// Use when ids are allocated increasingly and everything below `floor` is known to be done,
    /// to keep memory proportional to the live frontier. A dependency on an id below `floor` is
    /// met, and declaring one is a [`DagError::Duplicate`].
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagError, DagJob, DagScheduler, Input, JobSpec, Output,
    /// #     Policy, Resources, Scheduler, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// # let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// dag.declare([job(1, vec![])], 0.0).unwrap();
    /// dag.poll(0.0);
    /// dag.handle(Input::Done { job: 1, attempt: 1 }, 1.0);
    /// assert_eq!(dag.dag_stats().completed_remembered, 1);
    ///
    /// dag.forget_completed_below(10);
    /// assert_eq!(dag.dag_stats().completed_remembered, 0);
    /// assert_eq!(
    ///     dag.declare([job(5, vec![])], 1.0),
    ///     Err(DagError::Duplicate(5))
    /// );
    /// dag.declare([job(10, vec![1, 9])], 1.0).unwrap();
    /// assert_eq!(
    ///     dag.poll(1.0),
    ///     vec![Output::Start {
    ///         job: 10,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn forget_completed_below(&mut self, floor: JobId) {
        if floor > self.completed_floor {
            self.completed_floor = floor;
            self.completed.retain(|&j| j >= floor);
        }
    }

    /// The upward rank of a unit or of a job.
    ///
    /// A unit's, named by its id, is its critical path plus the rank below it; a job's (a leaf)
    /// is its work plus the longest chain of work below it, through the enclosing units and their
    /// dependents. `None` for ids this layer does not know, completed ones included.
    ///
    /// A unit of two independent jobs of work 1 and 3, followed by a job of work 2:
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, DagTemplate, JobSpec, Resources,
    /// #     Scheduler, TemplateNode, Unit};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// let nodes = vec![TemplateNode::Job(1.0), TemplateNode::Job(3.0)];
    /// let pair = DagTemplate::with_nodes(nodes, []).unwrap();
    /// let spec = JobSpec::new(0, Resources::mem(1), 0);
    /// dag.declare(
    ///     [
    ///         Unit::new(10, 100, Arc::new(pair), spec.clone(), vec![]),
    ///         DagJob::new(JobSpec { id: 20, ..spec }, vec![10])
    ///             .with_work(2.0)
    ///             .into(),
    ///     ],
    ///     0.0,
    /// )
    /// .unwrap();
    /// assert_eq!(
    ///     (dag.rank(10), dag.rank(100), dag.rank(101)),
    ///     (Some(5.0), Some(3.0), Some(5.0))
    /// );
    /// assert_eq!((dag.rank(20), dag.rank(99)), (Some(2.0), None));
    /// ```
    pub fn rank(&self, job: JobId) -> Option<f64> {
        match self.locate(job)? {
            Loc::Unit(u) => Some(self.unit(u).top()),
            Loc::Leaf { unit, leaf } => Some(self.leaf_rank(unit, leaf)),
        }
    }

    /// Change a unit's scale (a plain job's work, e.g. once its real size is known) and re-rank
    /// it and its dependencies, up or down.
    ///
    /// Returns false for unknown ids and leaves of other units. Jobs already handed to the policy
    /// keep the rank they were submitted with.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Resources, Scheduler, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// # let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// dag.declare([job(1, vec![]), job(2, vec![1])], 0.0).unwrap();
    /// assert_eq!(dag.rank(1), Some(2.0));
    /// assert!(dag.update_work(2, 5.0));
    /// assert_eq!(dag.rank(1), Some(6.0));
    /// assert!(dag.update_work(2, 0.5));
    /// assert_eq!(dag.rank(1), Some(1.5));
    /// assert!(!dag.update_work(3, 1.0));
    /// ```
    pub fn update_work(&mut self, job: JobId, scale: f64) -> bool {
        let Some(&n) = self.ids.get(&job) else {
            return false;
        };
        if self.unit(n).state == UnitState::Undeclared {
            return false;
        }
        self.unit_mut(n).scale = scale;
        if !self.config.track_ranks {
            return true;
        }
        let eps = self.config.rank_epsilon.max(0.0);
        let mut stack = vec![n];
        let mut first = true;
        while let Some(m) = stack.pop() {
            let tail = self
                .unit(m)
                .succs
                .iter()
                .map(|&c| self.unit(c).top())
                .fold(0.0, f64::max);
            let rec = self.unit(m);
            let old = if first { f64::NAN } else { rec.top() };
            let new = rec.scale * rec.span + tail;
            if first || (new - old).abs() > eps * old.abs().max(new.abs()) {
                self.unit_mut(m).tail = tail;
                stack.extend(self.unit(m).preds.iter().copied());
            }
            first = false;
        }
        true
    }

    /// Counters describing this layer's state; see [`DagStats`].
    pub fn dag_stats(&self) -> DagStats {
        let mut s = DagStats {
            completed_remembered: self.completed.len(),
            ..DagStats::default()
        };
        for rec in self.units.iter().flatten() {
            s.edges += rec.preds.len();
            match rec.state {
                UnitState::Undeclared => s.undeclared += 1,
                UnitState::Pending => {
                    s.units += 1;
                    s.pending += rec.leaves() as usize - rec.completed.len();
                }
                UnitState::Open => {
                    s.units += 1;
                    s.open += 1;
                }
            }
        }
        for f in self.frames.iter().flatten() {
            s.frames += 1;
            s.nodes += f.counter.len();
            s.node_bytes += f.bytes();
            let rec = self.unit(f.unit);
            for (i, &c) in f.counter.iter().enumerate() {
                match c {
                    SUBMITTED => s.submitted += 1,
                    HELD => s.held += 1,
                    COMPLETE | frame::OPEN => {}
                    _ => {
                        let first = f.leaf0 + f.template.leaf_offset(i) as u32;
                        let next = f.leaf0 + f.template.leaf_offset(i + 1) as u32;
                        s.pending += (next - first) as usize - rec.completed_in(first..next);
                    }
                }
            }
        }
        s
    }

    /// Drop announcements not yet polled ([`Output::Ready`], [`Output::RunLocal`]) of the jobs
    /// for which `gone` holds.
    fn unannounce(&mut self, gone: impl Fn(JobId) -> bool) {
        self.outbox.retain(
            |o| !matches!(*o, Output::Ready { job } | Output::RunLocal { job } if gone(job)),
        );
    }

    /// Cancel a unit and, transitively, every unit depending on it (they can never run).
    ///
    /// The unit is named by its id or any of its jobs. Returns the cancelled units' ids (a plain
    /// job's is the job's). Each of their jobs the inner policy has is cancelled there (its live
    /// attempts are stopped). An id this layer does not know is cancelled in the inner policy.
    /// [`Input::Cancel`] does the same.
    ///
    /// Cancelling running job 1 takes its dependents 2 and 3 with it, and stops its attempt:
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, DagStats, Input, JobSpec, Output,
    /// #     Policy, Resources, Scheduler, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// # let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// dag.declare([job(1, vec![]), job(2, vec![1]), job(3, vec![2])], 0.0)
    ///     .unwrap();
    /// dag.poll(0.0);
    /// assert_eq!(dag.cancel(1), vec![1, 2, 3]);
    /// assert_eq!(
    ///     dag.poll(1.0),
    ///     vec![Output::Stop {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// assert_eq!(dag.dag_stats(), DagStats::default());
    /// ```
    pub fn cancel(&mut self, job: JobId) -> Vec<JobId> {
        let Some(start) = self.unit_of(job) else {
            self.policy.handle(Input::Cancel(job), self.now);
            return Vec::new();
        };
        let mut doomed = BTreeSet::new();
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            if doomed.insert((self.unit(u).id, u)) {
                stack.extend(self.unit(u).succs.iter().copied());
            }
        }
        let slots: HashSet<u32> = doomed.iter().map(|d| d.1).collect();
        let mut cancelled = Vec::with_capacity(doomed.len());
        let mut leaves = HashSet::new();
        let mut orphan_candidates = Vec::new();
        for &(id, u) in &doomed {
            if self.unit(u).state == UnitState::Open {
                let (submitted, held) = self.drop_frames(u);
                for leaf in submitted {
                    self.policy.handle(Input::Cancel(leaf), self.now);
                    leaves.insert(leaf);
                }
                leaves.extend(held);
            }
            if self.unit(u).state != UnitState::Undeclared {
                cancelled.push(id);
            }
            for &p in &self.unit(u).preds {
                if !slots.contains(&p) {
                    orphan_candidates.push(p);
                }
            }
        }
        self.unannounce(|j| leaves.contains(&j));
        for &(_, u) in &doomed {
            for p in std::mem::take(&mut self.unit_mut(u).preds) {
                if !slots.contains(&p) {
                    self.unit_mut(p).succs.retain(|&s| s != u);
                }
            }
            self.free_unit(u);
        }
        // Forward references kept alive only by the cancelled units are no longer needed.
        for p in orphan_candidates {
            let orphan = self.units[p as usize]
                .as_ref()
                .is_some_and(|r| r.state == UnitState::Undeclared && r.succs.is_empty());
            if orphan {
                self.free_unit(p);
            }
        }
        for leaf in &leaves {
            self.live.remove(leaf);
            self.ignored.remove(leaf);
        }
        cancelled
    }

    /// Close a unit early (named by its id or any of its jobs).
    ///
    /// For when its remaining jobs are known to be no-ops. If it is open, its jobs that have not
    /// started complete as no-ops (waiting ones are withdrawn from the policy) and the unit
    /// completes; it returns the jobs already running, whose workers keep their resources until
    /// each one's attempt ends ([`Input::Done`], [`Input::Failed`], or its worker leaving), which
    /// then changes nothing else; such a job is not retried. A unit not entered yet completes as
    /// soon as its dependencies do, without running anything.
    ///
    /// Unit 200 has jobs 100 and 101 and job 2 waits for it. Closing it while job 100 runs
    /// withdraws job 101 and completes the unit, so job 2 is submitted; job 100's worker is busy
    /// until its attempt ends. Unit 300, closed before it is entered, never runs anything.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, DagTemplate, Input, JobSpec, Output,
    /// #     Policy, Resources, Scheduler, Unit, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::mem(100))), 0.0);
    /// # let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps);
    /// let pair = Arc::new(DagTemplate::new(2, []).unwrap());
    /// let spec = JobSpec::new(0, Resources::mem(1), 0);
    /// dag.declare(
    ///     [
    ///         Unit::new(200, 100, pair.clone(), spec.clone(), vec![]),
    ///         Unit::new(300, 110, pair, spec, vec![2]),
    ///         job(2, vec![200]).into(),
    ///     ],
    ///     0.0,
    /// )
    /// .unwrap();
    /// assert_eq!(dag.poll(0.0), vec![Output::Start { job: 100, attempt: 1, worker: 1 }]);
    /// assert_eq!(dag.close(300, 0.0), Ok(vec![]));
    /// assert_eq!(dag.close(200, 1.0), Ok(vec![100]));
    /// assert_eq!((dag.stats().waiting, dag.stats().running), (1, 1));
    ///
    /// dag.handle(Input::Done { job: 100, attempt: 1 }, 2.0);
    /// assert_eq!(dag.poll(2.0), vec![Output::Start { job: 2, attempt: 1, worker: 1 }]);
    /// dag.handle(Input::Done { job: 2, attempt: 1 }, 3.0);
    /// assert!(dag.poll(3.0).is_empty());
    /// assert_eq!(dag.dag_stats().units, 0);
    /// ```
    pub fn close(&mut self, job: JobId, now: Instant) -> Result<Vec<JobId>, DagError> {
        self.now = now;
        let u = self.unit_of(job).ok_or(DagError::NotFound(job))?;
        match self.unit(u).state {
            UnitState::Undeclared => Err(DagError::NotFound(job)),
            UnitState::Pending => {
                self.unit_mut(u).closed = true;
                Ok(Vec::new())
            }
            UnitState::Open => {
                let (submitted, held) = self.drop_frames(u);
                let mut running = Vec::new();
                for leaf in submitted {
                    if self.live.contains_key(&leaf) {
                        self.ignored.insert(leaf);
                        running.push(leaf);
                    } else {
                        self.policy.handle(Input::Cancel(leaf), now);
                    }
                }
                let held: HashSet<JobId> = held.into_iter().collect();
                self.unannounce(|j| held.contains(&j));
                self.finish_unit(u);
                self.settle(now);
                running.sort_unstable();
                Ok(running)
            }
        }
    }

    /// Forget a live attempt; whether it was one.
    fn drop_attempt(&mut self, job: JobId, attempt: Attempt) -> bool {
        let Some(v) = self.live.get_mut(&job) else {
            return false;
        };
        let before = v.len();
        v.retain(|a| a.0 != attempt);
        let dropped = v.len() < before;
        if v.is_empty() {
            self.live.remove(&job);
        }
        dropped
    }

    /// An attempt (or, with attempt 0, a local job) finished.
    fn done(&mut self, job: JobId, attempt: Attempt, now: Instant) {
        if attempt == 0 {
            if let Some((f, i)) = self.leaf_node(job)
                && self.frame(f).counter[i] == HELD
                && matches!(self.frame(f).template.node(i), TemplateNode::Local(_))
            {
                self.complete_node(f, i);
                self.settle(now);
            }
            return;
        }
        let accepted = self
            .live
            .get(&job)
            .is_some_and(|v| v.iter().any(|a| a.0 == attempt));
        self.policy.handle(Input::Done { job, attempt }, now);
        if !accepted {
            return;
        }
        self.live.remove(&job);
        if self.ignored.remove(&job) {
            return;
        }
        if let Some((f, i)) = self.leaf_node(job)
            && self.frame(f).counter[i] == SUBMITTED
        {
            self.complete_node(f, i);
            self.settle(now);
        }
    }

    /// An attempt failed. A job of a unit closed early is not retried: its last failure cancels
    /// it.
    fn failed(&mut self, job: JobId, attempt: Attempt, kind: crate::FailKind, why: String) {
        let was_live = self.drop_attempt(job, attempt);
        if was_live && self.ignored.contains(&job) && !self.live.contains_key(&job) {
            self.ignored.remove(&job);
            self.quiet.insert((job, attempt));
            self.policy.handle(Input::Cancel(job), self.now);
        } else {
            let input = Input::Failed {
                job,
                attempt,
                kind,
                why,
            };
            self.policy.handle(input, self.now);
        }
    }

    /// A worker left: its attempts are no longer live. Jobs of units closed early that ran only
    /// there are cancelled rather than retried.
    fn worker_gone(&mut self, w: WorkerId, now: Instant) {
        let mut orphaned = Vec::new();
        self.live.retain(|&job, v| {
            let lost: Vec<Attempt> = v.iter().filter(|a| a.1 == w).map(|a| a.0).collect();
            v.retain(|a| a.1 != w);
            if v.is_empty() && !lost.is_empty() && self.ignored.contains(&job) {
                orphaned.push((job, lost));
            }
            !v.is_empty()
        });
        orphaned.sort_unstable();
        for (job, lost) in orphaned {
            self.ignored.remove(&job);
            self.quiet.extend(lost.into_iter().map(|a| (job, a)));
            self.policy.handle(Input::Cancel(job), now);
        }
        self.policy.handle(Input::WorkerGone(w), now);
    }

    /// The inner policy gave up on a job: hold it again, for `release` or `cancel`.
    fn hold_again(&mut self, job: JobId) {
        if let Some((f, i)) = self.leaf_node(job) {
            let frame = self.frame_mut(f);
            if frame.counter[i] == SUBMITTED {
                frame.counter[i] = HELD;
            }
        }
    }

    /// A unit's state in words.
    fn explain_unit(&self, u: u32, what: &str) -> String {
        let rec = self.unit(u);
        let id = rec.id;
        match rec.state {
            UnitState::Undeclared => format!(
                "{what} {id} is not declared yet (named as a dependency of {} unit(s))",
                rec.succs.len()
            ),
            UnitState::Pending => {
                let mut deps: Vec<JobId> = rec.preds.iter().map(|&d| self.unit(d).id).collect();
                deps.sort_unstable();
                let shown: Vec<_> = deps.iter().take(8).collect();
                format!(
                    "{what} {id} {}waits for {} dependenc{} {shown:?}{}",
                    if rec.closed { "is closed and " } else { "" },
                    deps.len(),
                    if deps.len() == 1 { "y" } else { "ies" },
                    if deps.len() > 8 { " ..." } else { "" }
                )
            }
            UnitState::Open => format!("{what} {id} is open"),
        }
    }

    /// `explain` for leaf `leaf` of unit `u`.
    fn explain_leaf(&self, u: u32, leaf: u32, job: JobId) -> Option<String> {
        let rec = self.unit(u);
        let msg = if rec.plain() && rec.state == UnitState::Pending {
            Some(self.explain_unit(u, "job"))
        } else if rec.leaf_completed(leaf) {
            Some(format!("job {job} completed"))
        } else if rec.state == UnitState::Pending {
            Some(format!(
                "job {job} waits for its unit: {}",
                self.explain_unit(u, "unit")
            ))
        } else {
            match self.leaf_node(job) {
                None => Some(format!(
                    "job {job} waits for its part of unit {} to be entered",
                    rec.id
                )),
                Some((f, i)) => match self.frame(f).counter[i] {
                    SUBMITTED => self.policy.explain(job),
                    HELD => Some(format!("job {job} is ready and held until release")),
                    COMPLETE => Some(format!("job {job} completed")),
                    k => Some(format!(
                        "job {job} waits for {k} dependenc{} within its unit",
                        if k == 1 { "y" } else { "ies" }
                    )),
                },
            }
        };
        match rec.sourced {
            true => match self.src().label(rec.id, leaf) {
                Some(l) => msg.map(|m| format!("[{l}] {m}")),
                None => msg,
            },
            false => msg,
        }
    }
}

impl<P: Policy> Policy for DagScheduler<P> {
    /// Forwarded to the inner policy, with this layer's bookkeeping: a done attempt completes its
    /// job here (attempt 0 completes a local job, which the inner policy never sees), and a
    /// cancellation cascades to the job's dependents ([`DagScheduler::cancel`]).
    fn handle(&mut self, input: Input, now: Instant) {
        self.now = now;
        match input {
            Input::Done { job, attempt } => self.done(job, attempt, now),
            Input::Failed {
                job,
                attempt,
                kind,
                why,
            } => self.failed(job, attempt, kind, why),
            Input::Cancel(job) => {
                self.cancel(job);
            }
            Input::WorkerGone(w) => self.worker_gone(w, now),
            input @ (Input::Submit(_) | Input::Worker(_)) => self.policy.handle(input, now),
        }
    }

    /// This layer's announcements not yet drained, then the inner policy's outputs.
    fn poll(&mut self, now: Instant) -> Vec<Output> {
        self.now = now;
        let inner = self.policy.poll(now);
        let mut out = std::mem::take(&mut self.outbox);
        for o in inner {
            match &o {
                Output::Start {
                    job,
                    attempt,
                    worker,
                } => self.live.entry(*job).or_default().push((*attempt, *worker)),
                Output::Stop { job, attempt, .. } => {
                    self.drop_attempt(*job, *attempt);
                    if self.quiet.remove(&(*job, *attempt)) {
                        continue;
                    }
                }
                Output::GaveUp(g) => {
                    self.live.remove(&g.job);
                    self.hold_again(g.job);
                }
                _ => {}
            }
            out.push(o);
        }
        out
    }

    /// Forwarded to the inner policy.
    fn next_wakeup(&self) -> Option<Instant> {
        self.policy.next_wakeup()
    }

    /// The DAG's reason while the job is not submitted, the inner policy's afterwards. Units are
    /// explained by their id.
    fn explain(&self, job: JobId) -> Option<String> {
        match self.locate(job) {
            Some(Loc::Unit(u)) => Some(self.explain_unit(u, "unit")),
            Some(Loc::Leaf { unit, leaf }) => self.explain_leaf(unit, leaf, job),
            None if self.is_completed(job) => Some(format!("job {job} completed")),
            None => self.policy.explain(job),
        }
    }

    /// Forwarded to the inner policy.
    fn stats(&self) -> PolicyStats {
        self.policy.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, FailKind, GaveUp, Resources, RetryConfig, Scheduler, WorkerState};

    /// A DAG over a FIFO scheduler with one one-slot worker, and `max_attempts` attempts per job.
    fn dag(config: DagConfig, max_attempts: u32) -> DagScheduler<Scheduler> {
        let mut d = DagScheduler::new(
            config,
            Scheduler::new(Config {
                retry: RetryConfig { max_attempts },
                ..Config::fifo()
            }),
        );
        d.handle(
            Input::Worker(WorkerState::new(1, "x", 1, Resources::mem(100))),
            0.0,
        );
        d
    }

    /// A job of group 0 with the given dependencies.
    fn job(id: JobId, deps: &[JobId]) -> DagJob {
        DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps.to_vec())
    }

    /// The start of attempt `attempt` of `job` on worker 1.
    fn start(job: JobId, attempt: Attempt) -> Output {
        Output::Start {
            job,
            attempt,
            worker: 1,
        }
    }

    /// Handle `inputs` at `t`, then poll.
    fn feed(
        d: &mut DagScheduler<Scheduler>,
        t: Instant,
        inputs: impl IntoIterator<Item = Input>,
    ) -> Vec<Output> {
        for i in inputs {
            d.handle(i, t);
        }
        d.poll(t)
    }

    /// A done message.
    fn done(job: JobId, attempt: Attempt) -> Input {
        Input::Done { job, attempt }
    }

    /// A failure message.
    fn fail(job: JobId, attempt: Attempt) -> Input {
        Input::Failed {
            job,
            attempt,
            kind: FailKind::Other,
            why: "test".into(),
        }
    }

    /// A unit `id` of a template of `len` independent jobs at `base`, after `deps`.
    fn unit(id: JobId, base: JobId, len: usize, deps: &[JobId]) -> Unit {
        Unit::new(
            id,
            base,
            Arc::new(DagTemplate::new(len, []).unwrap()),
            JobSpec::new(0, Resources::mem(1), 0),
            deps.to_vec(),
        )
    }

    /// The inner policy's starts come out of the DAG's poll, and a done attempt releases the
    /// dependents.
    #[test]
    fn done_releases_dependents() {
        let mut d = dag(DagConfig::default(), 4);
        d.declare(vec![job(1, &[]), job(2, &[1])], 0.0).unwrap();
        assert_eq!(d.poll(0.0), vec![start(1, 1)]);
        assert_eq!(feed(&mut d, 1.0, [done(1, 1)]), vec![start(2, 1)]);
        assert!(feed(&mut d, 2.0, [done(2, 1)]).is_empty());
        let st = d.dag_stats();
        assert_eq!(
            (st.pending, st.submitted, st.completed_remembered),
            (0, 0, 2)
        );
        assert_eq!(d.explain(2), Some("job 2 completed".into()));
    }

    /// A retried job's stale report does not complete it here either.
    #[test]
    fn stale_done_does_not_complete() {
        let mut d = dag(DagConfig::default(), 4);
        d.declare(vec![job(1, &[]), job(2, &[1])], 0.0).unwrap();
        d.poll(0.0);
        assert_eq!(feed(&mut d, 1.0, [fail(1, 1)]), vec![start(1, 2)]);
        assert!(feed(&mut d, 2.0, [done(1, 1)]).is_empty());
        assert_eq!(d.dag_stats().pending, 1);
        assert_eq!(feed(&mut d, 3.0, [done(1, 2)]), vec![start(2, 1)]);
    }

    /// A worker leaving fails its attempts in the inner policy, which retries them: a late report
    /// from the departed worker completes nothing.
    #[test]
    fn worker_gone_retries() {
        let mut d = dag(DagConfig::default(), 4);
        d.declare(vec![job(1, &[]), job(2, &[1])], 0.0).unwrap();
        d.poll(0.0);
        assert!(feed(&mut d, 1.0, [Input::WorkerGone(1)]).is_empty());
        assert!(feed(&mut d, 2.0, [done(1, 1)]).is_empty());
        let w = WorkerState::new(1, "x", 1, Resources::mem(100));
        assert_eq!(feed(&mut d, 3.0, [Input::Worker(w)]), vec![start(1, 2)]);
        assert_eq!(feed(&mut d, 4.0, [done(1, 2)]), vec![start(2, 1)]);
    }

    /// Local jobs are announced, never submitted, and completed with attempt 0.
    #[test]
    fn local_jobs_run_on_the_caller() {
        let mut d = dag(DagConfig::default(), 4);
        d.declare(vec![job(1, &[]).local(), job(2, &[1])], 0.0)
            .unwrap();
        assert_eq!(d.poll(0.0), vec![Output::RunLocal { job: 1 }]);
        assert_eq!(d.stats().waiting, 0);
        assert!(!d.release(1, 0.0));
        // Attempt numbers of workers' jobs do not complete a local job.
        assert!(feed(&mut d, 1.0, [done(1, 1)]).is_empty());
        assert_eq!(feed(&mut d, 1.0, [done(1, 0)]), vec![start(2, 1)]);
    }

    /// Without `auto_submit`, ready jobs are announced and held until released.
    #[test]
    fn held_jobs_are_announced() {
        let config = DagConfig {
            auto_submit: false,
            ..DagConfig::default()
        };
        let mut d = dag(config, 4);
        d.declare(vec![job(1, &[]), job(2, &[1]), job(3, &[])], 0.0)
            .unwrap();
        assert_eq!(
            d.poll(0.0),
            vec![Output::Ready { job: 1 }, Output::Ready { job: 3 }]
        );
        assert!(d.release(1, 1.0));
        assert!(!d.release(1, 1.0));
        assert_eq!(d.poll(1.0), vec![start(1, 1)]);
        assert_eq!(
            feed(&mut d, 2.0, [done(1, 1)]),
            vec![Output::Ready { job: 2 }]
        );
        // Cancelling withdraws an announcement not yet polled.
        d.declare(vec![job(4, &[])], 3.0).unwrap();
        assert_eq!(d.cancel(4), vec![4]);
        assert!(d.poll(3.0).is_empty());
    }

    /// `announcements` drains the layer's own outputs without placing anything; the next poll
    /// places, and returns only what was announced since.
    #[test]
    fn announcements_come_before_placement() {
        let config = DagConfig {
            auto_submit: false,
            ..DagConfig::default()
        };
        let mut d = dag(config, 4);
        d.declare(vec![job(1, &[]), job(2, &[]), job(3, &[1])], 0.0)
            .unwrap();
        assert_eq!(
            d.announcements(),
            vec![Output::Ready { job: 1 }, Output::Ready { job: 2 }]
        );
        assert!(d.announcements().is_empty());
        // Job 2 is released first, so it takes the one slot.
        assert!(d.release(2, 0.0) && d.release(1, 0.0));
        assert_eq!(d.poll(0.0), vec![start(2, 1)]);
        d.handle(done(2, 1), 1.0);
        d.declare(vec![job(4, &[])], 1.0).unwrap();
        assert_eq!(d.poll(1.0), vec![Output::Ready { job: 4 }, start(1, 1)]);
    }

    /// Passthrough jobs and units are announced when recorded; a plain job's unit is not.
    #[test]
    fn passthroughs_are_announced() {
        let config = DagConfig {
            record_passthrough: true,
            ..DagConfig::default()
        };
        let mut d = dag(config, 4);
        d.declare(
            vec![job(1, &[]), DagJob::passthrough(2, 0, vec![1], 0.0)],
            0.0,
        )
        .unwrap();
        assert_eq!(d.poll(0.0), vec![start(1, 1)]);
        assert_eq!(
            feed(&mut d, 1.0, [done(1, 1)]),
            vec![Output::Passed { job: 2 }]
        );
        d.declare([unit(200, 100, 1, &[2])], 2.0).unwrap();
        assert_eq!(d.poll(2.0), vec![start(100, 1)]);
        assert_eq!(
            feed(&mut d, 3.0, [done(100, 1)]),
            vec![Output::Passed { job: 200 }]
        );
    }

    /// A give-up passes through and holds the job again; releasing it starts another round.
    #[test]
    fn give_up_holds_the_job() {
        let mut d = dag(DagConfig::default(), 1);
        d.declare(vec![job(1, &[]), job(2, &[1])], 0.0).unwrap();
        d.poll(0.0);
        let out = feed(&mut d, 1.0, [fail(1, 1)]);
        let [Output::GaveUp(GaveUp { job: 1, .. })] = out[..] else {
            panic!("expected a give-up, got {out:?}");
        };
        assert_eq!(d.dag_stats().held, 1);
        assert!(d.explain(1).unwrap().contains("held until release"));
        assert!(d.release(1, 2.0));
        assert_eq!(d.poll(2.0), vec![start(1, 1)]);
        assert_eq!(feed(&mut d, 3.0, [done(1, 1)]), vec![start(2, 1)]);
    }

    /// `Input::Cancel` cascades to dependents and stops running attempts.
    #[test]
    fn cancel_input_cascades() {
        let mut d = dag(DagConfig::default(), 4);
        d.declare(vec![job(1, &[]), job(2, &[1]), job(3, &[2])], 0.0)
            .unwrap();
        d.poll(0.0);
        let out = feed(&mut d, 1.0, [Input::Cancel(1)]);
        assert_eq!(
            out,
            vec![Output::Stop {
                job: 1,
                attempt: 1,
                worker: 1
            }]
        );
        assert_eq!(d.dag_stats(), DagStats::default());
        assert_eq!(d.stats().running, 0);
    }

    /// A two-job unit after job 1, and job 2 after it; job 1 done, leaf 100 running, and the unit
    /// closed early.
    fn closed_unit() -> DagScheduler<Scheduler> {
        let mut d = dag(DagConfig::default(), 4);
        d.declare(vec![job(1, &[]), job(2, &[200])], 0.0).unwrap();
        d.declare([unit(200, 100, 2, &[1])], 0.0).unwrap();
        assert_eq!(d.poll(0.0), vec![start(1, 1)]);
        assert_eq!(feed(&mut d, 1.0, [done(1, 1)]), vec![start(100, 1)]);
        assert_eq!(d.close(200, 2.0), Ok(vec![100]));
        // Leaf 101 was withdrawn; the unit completed, releasing job 2 behind the running leaf.
        let st = d.stats();
        assert_eq!((st.waiting, st.running), (1, 1));
        assert!(d.poll(2.0).is_empty());
        d
    }

    /// A running job of a unit closed early keeps its resources until its attempt ends; a
    /// failure then is neither retried nor reported.
    #[test]
    fn closed_unit_job_fails() {
        let mut d = closed_unit();
        assert_eq!(feed(&mut d, 3.0, [fail(100, 1)]), vec![start(2, 1)]);
        assert_eq!(d.stats().running, 1);
        assert_eq!(d.explain(100), None);
    }

    /// The same when the job's worker leaves.
    #[test]
    fn closed_unit_job_worker_gone() {
        let mut d = closed_unit();
        assert!(feed(&mut d, 3.0, [Input::WorkerGone(1)]).is_empty());
        let st = d.stats();
        assert_eq!((st.waiting, st.running), (1, 0));
        let w = WorkerState::new(1, "x", 1, Resources::mem(100));
        assert_eq!(feed(&mut d, 4.0, [Input::Worker(w)]), vec![start(2, 1)]);
    }

    /// Its completion only frees its resources.
    #[test]
    fn closed_unit_job_done() {
        let mut d = closed_unit();
        assert_eq!(feed(&mut d, 3.0, [done(100, 1)]), vec![start(2, 1)]);
    }

    /// A snapshot restores held jobs as announcements and submitted ones as fresh submissions.
    #[cfg(feature = "serde")]
    #[test]
    fn restore_announces_held_jobs() {
        let config = DagConfig {
            auto_submit: false,
            ..DagConfig::default()
        };
        let mut d = dag(config, 4);
        d.declare(
            vec![job(1, &[]), job(2, &[]), job(3, &[1]), job(4, &[]).local()],
            0.0,
        )
        .unwrap();
        d.poll(0.0);
        assert!(d.release(1, 0.0));
        assert_eq!(d.poll(0.0), vec![start(1, 1)]);
        let snap = d.snapshot();
        let mut r = DagScheduler::restore(snap, Scheduler::new(Config::fifo()), None, 5.0);
        let w = WorkerState::new(7, "x", 1, Resources::mem(100));
        assert_eq!(
            feed(&mut r, 5.0, [Input::Worker(w)]),
            vec![
                Output::RunLocal { job: 4 },
                Output::Ready { job: 2 },
                Output::Start {
                    job: 1,
                    attempt: 1,
                    worker: 7
                }
            ]
        );
    }

    /// Ids of a unit may not collide with another's.
    #[test]
    fn overlapping_ids_are_refused() {
        let mut d = dag(DagConfig::default(), 4);
        d.declare([unit(99, 10, 3, &[])], 0.0).unwrap();
        assert_eq!(
            d.declare([unit(98, 12, 3, &[])], 0.0).unwrap_err(),
            DagError::Overlap(12)
        );
        assert_eq!(
            d.declare([job(11, &[])], 0.0).unwrap_err(),
            DagError::Overlap(11)
        );
        assert_eq!(
            d.declare([job(5, &[11])], 0.0).unwrap_err(),
            DagError::Overlap(11)
        );
        assert_eq!(
            d.declare([unit(11, 20, 3, &[])], 0.0).unwrap_err(),
            DagError::Overlap(11)
        );
        assert_eq!(
            d.declare([unit(30, 29, 3, &[])], 0.0).unwrap_err(),
            DagError::Overlap(30)
        );
        assert_eq!(
            d.declare([unit(97, 0, 30, &[])], 0.0).unwrap_err(),
            DagError::Overlap(10)
        );
        assert_eq!(
            d.declare([unit(99, 50, 3, &[])], 0.0).unwrap_err(),
            DagError::Duplicate(99)
        );
        d.declare([unit(98, 13, 3, &[99])], 0.0).unwrap();
    }
}
