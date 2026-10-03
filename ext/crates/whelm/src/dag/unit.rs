//! What the caller declares: plain jobs, units over templates, and the source of their leaves.

use std::sync::Arc;

use super::{DagTemplate, template};
#[cfg(doc)]
use crate::{DagConfig, DagScheduler, Input, Output, Policy, TemplateNode, TemplateSpec};
use crate::{JobId, JobSpec};

/// A plain job with dependencies.
///
/// It is a [`Unit`] of a one-node template, whose one leaf is the job, so other units depend on
/// it by the job's id. A job runs on a worker unless it is a
/// [`passthrough`](field@Self::passthrough) or a [`local`](field@Self::local) job, the other two
/// kinds of node a job can be.
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
/// for id in [1, 2] {
///     let w = WorkerState {
///         id,
///         budget: Resources::mem(100),
///         ..Default::default()
///     };
///     dag.handle(Input::Worker(w), 0.0);
/// }
/// let job = |id, deps| DagJob {
///     spec: JobSpec {
///         id,
///         demand: Resources::mem(1),
///         ..Default::default()
///     },
///     deps,
///     ..Default::default()
/// };
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
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DagJob {
    /// The job, as it will be submitted to the policy.
    pub spec: JobSpec,
    /// Units that must all complete before this one is ready (see [`Unit::deps`]).
    pub deps: Vec<JobId>,
    /// Relative cost, for ranks. `None` uses [`DagConfig::default_work`].
    ///
    /// The estimate becomes the submitted [`JobSpec::work`] unless that is set. Job 1 leads a
    /// chain of work 5 then 1; job 3, independent and of default work, ranks below it.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, JobSpec, Scheduler};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// let job = |id, deps| DagJob {
    ///     spec: JobSpec {
    ///         id,
    ///         ..Default::default()
    ///     },
    ///     deps,
    ///     ..Default::default()
    /// };
    /// let lead = DagJob {
    ///     work_estimate: Some(5.0),
    ///     ..job(1, vec![])
    /// };
    /// dag.declare([lead, job(2, vec![1]), job(3, vec![])], 0.0)
    ///     .unwrap();
    /// assert_eq!(
    ///     (dag.rank(1), dag.rank(2), dag.rank(3)),
    ///     (Some(6.0), Some(1.0), Some(1.0))
    /// );
    /// ```
    pub work_estimate: Option<f64>,
    /// A pure synchronisation point ("group G is done"): when ready it completes by itself
    /// instead of being submitted to the policy. Its `work_estimate` still counts in ranks.
    ///
    /// A barrier: job 3 stands for "jobs 1 and 2 are done", so that job 4 can name one dependency
    /// instead of every job before it. It completes without reaching the policy, and with
    /// [`DagConfig::record_passthrough`] it is announced as it does.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Scheduler, WorkerState};
    /// let config = DagConfig {
    ///     record_passthrough: true,
    ///     ..DagConfig::default()
    /// };
    /// let mut dag = DagScheduler::new(config, Scheduler::new(Config::fifo()));
    /// let w = WorkerState {
    ///     id: 1,
    ///     slots: 2,
    ///     ..Default::default()
    /// };
    /// dag.handle(Input::Worker(w), 0.0);
    /// let job = |id, deps| DagJob {
    ///     spec: JobSpec {
    ///         id,
    ///         ..Default::default()
    ///     },
    ///     deps,
    ///     ..Default::default()
    /// };
    /// let barrier = DagJob {
    ///     passthrough: true,
    ///     work_estimate: Some(0.0),
    ///     ..job(3, vec![1, 2])
    /// };
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
    pub passthrough: bool,
    /// Runs on the caller, not on a worker (registration, loading, commit steps): when ready it is
    /// held, never submitted to the policy, and announced by [`Output::RunLocal`]; report its
    /// completion with [`Input::Done`] and attempt 0. It runs exactly once: it is not retried.
    ///
    /// Job 1 registers something on the caller before job 2 runs on a worker. The caller reports
    /// it done with attempt 0; a worker attempt's number does not complete it.
    ///
    /// ```
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Scheduler, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), 0.0);
    /// let job = |id, deps| DagJob {
    ///     spec: JobSpec {
    ///         id,
    ///         ..Default::default()
    ///     },
    ///     deps,
    ///     ..Default::default()
    /// };
    /// let register = DagJob {
    ///     local: true,
    ///     ..job(1, vec![])
    /// };
    /// dag.declare([register, job(2, vec![1])], 0.0).unwrap();
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
    pub local: bool,
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
///     Config, DagConfig, DagScheduler, Input, Output, Policy, Scheduler, TemplateNode,
///     TemplateSpec, Unit, WorkerState,
/// };
///
/// let chain = TemplateSpec {
///     edges: vec![(0, 1)],
///     ..TemplateSpec::jobs(2)
/// };
/// let outer = TemplateSpec {
///     nodes: vec![
///         TemplateNode::Job(1.0),
///         TemplateNode::Unit(Arc::new(chain.build().unwrap())),
///         TemplateNode::Job(1.0),
///     ],
///     edges: vec![(0, 1), (1, 2)],
/// }
/// .build()
/// .unwrap();
/// assert_eq!(outer.leaves(), 4);
///
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
/// dag.handle(Input::Worker(WorkerState::default()), 0.0);
/// let unit = Unit {
///     id: 50,
///     base: 0,
///     template: Arc::new(outer),
///     ..Default::default()
/// };
/// dag.declare([unit], 0.0).unwrap();
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
    ///
    /// Unit 10 runs three independent jobs as 100, 101 and 102; job 5 depends on the unit as a
    /// whole and starts once all three are done.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Scheduler, TemplateSpec, Unit, WorkerState};
    /// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// let w = WorkerState {
    ///     id: 1,
    ///     slots: 3,
    ///     ..Default::default()
    /// };
    /// dag.handle(Input::Worker(w), 0.0);
    /// let unit = Unit {
    ///     id: 10,
    ///     base: 100,
    ///     template: Arc::new(TemplateSpec::jobs(3).build().unwrap()),
    ///     ..Default::default()
    /// };
    /// let after = DagJob {
    ///     spec: JobSpec {
    ///         id: 5,
    ///         ..Default::default()
    ///     },
    ///     deps: vec![10],
    ///     ..Default::default()
    /// };
    /// dag.declare([unit], 0.0).unwrap();
    /// dag.declare([after], 0.0).unwrap();
    ///
    /// let start = |job| Output::Start {
    ///     job,
    ///     attempt: 1,
    ///     worker: 1,
    /// };
    /// assert_eq!(dag.poll(0.0), vec![start(100), start(101), start(102)]);
    /// for job in [100, 101, 102] {
    ///     dag.handle(Input::Done { job, attempt: 1 }, 1.0);
    /// }
    /// assert_eq!(dag.poll(1.0), vec![start(5)]);
    /// ```
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
    ///
    /// One template serves units of different sizes: here a two-job chain at three times its
    /// template's work. The scale reaches ranks and each submitted job's [`JobSpec::work`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagScheduler, Scheduler, TemplateSpec, Unit};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// let chain = TemplateSpec {
    ///     edges: vec![(0, 1)],
    ///     ..TemplateSpec::jobs(2)
    /// };
    /// let unit = Unit {
    ///     id: 10,
    ///     base: 100,
    ///     template: Arc::new(chain.build().unwrap()),
    ///     scale: Some(3.0),
    ///     ..Default::default()
    /// };
    /// dag.declare([unit], 0.0).unwrap();
    /// assert_eq!(
    ///     (dag.rank(100), dag.rank(101), dag.rank(10)),
    ///     (Some(6.0), Some(3.0), Some(6.0))
    /// );
    /// ```
    pub scale: Option<f64>,
    /// Leaves' work, spec and label come from the scheduler's [`NodeSource`] rather than from the
    /// template and `spec` alone.
    ///
    /// The source here gives leaf `k` work `k + 1`, which the template alone could not say for
    /// every unit; the [`NodeSource`] docs show the rest of the trait.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagScheduler, JobId, NodeSource, Scheduler, TemplateSpec,
    /// #     Unit};
    /// struct Growing;
    /// impl NodeSource for Growing {
    ///     fn work(&self, _unit: JobId, leaf: u32) -> f64 {
    ///         f64::from(leaf + 1)
    ///     }
    /// }
    ///
    /// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()))
    ///     .with_source(Arc::new(Growing));
    /// let unit = Unit {
    ///     id: 10,
    ///     base: 100,
    ///     template: Arc::new(TemplateSpec::jobs(3).build().unwrap()),
    ///     sourced: true,
    ///     ..Default::default()
    /// };
    /// dag.declare([unit], 0.0).unwrap();
    /// assert_eq!(
    ///     [100, 101, 102].map(|j| dag.rank(j).unwrap()),
    ///     [1.0, 2.0, 3.0]
    /// );
    /// ```
    pub sourced: bool,
    /// Leaves already complete (e.g. restored from a checkpoint): they never run, and their
    /// successors start with those dependencies met. Need not be closed under predecessors: an
    /// incomplete predecessor of a complete leaf still runs, and its completion does not touch the
    /// complete leaf.
    ///
    /// Resuming a three-job chain whose first job finished before a restart: the run picks up at
    /// leaf 1, job 101.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use whelm::{Config, DagConfig, DagScheduler, Input, Output, Policy, Scheduler,
    /// #     TemplateSpec, Unit, WorkerState};
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), 0.0);
    /// let chain = TemplateSpec {
    ///     edges: vec![(0, 1), (1, 2)],
    ///     ..TemplateSpec::jobs(3)
    /// };
    /// let unit = Unit {
    ///     id: 10,
    ///     base: 100,
    ///     template: Arc::new(chain.build().unwrap()),
    ///     completed: vec![0],
    ///     ..Default::default()
    /// };
    /// dag.declare([unit], 0.0).unwrap();
    /// assert_eq!(
    ///     dag.poll(0.0),
    ///     vec![Output::Start {
    ///         job: 101,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// assert_eq!(dag.dag_stats().pending, 1);
    /// ```
    pub completed: Vec<u32>,
}

impl Default for Unit {
    /// A plain job: a unit of the one-node template of a worker job, with no dependency.
    fn default() -> Self {
        DagJob::default().into()
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

/// Per-leaf data of [`sourced`](field@Unit::sourced) units, computed on demand rather than stored.
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
///     Config, Constraint, DagConfig, DagScheduler, Input, JobId, JobSpec, NodeSource, Output,
///     Policy, Scheduler, TemplateSpec, Unit, WorkerState,
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
///             spec.constraints.push(Constraint::require_class("gpu"));
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
/// for (id, class) in [(1, "cpu"), (2, "gpu")] {
///     let w = WorkerState {
///         id,
///         class: class.into(),
///         ..Default::default()
///     };
///     dag.handle(Input::Worker(w), 0.0);
/// }
/// let chain = TemplateSpec {
///     edges: vec![(0, 1), (1, 2)],
///     ..TemplateSpec::jobs(3)
/// };
/// let unit = Unit {
///     id: 10,
///     base: 100,
///     template: Arc::new(chain.build().unwrap()),
///     sourced: true,
///     ..Default::default()
/// };
/// dag.declare([unit], 0.0).unwrap();
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
