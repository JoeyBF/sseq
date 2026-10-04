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
//! before anything is placed. A job the inner policy gives up on ([`Output::GaveUp`]) or rejects
//! ([`Output::Rejected`]) is held again rather than forgotten: its dependents wait until it is
//! released or cancelled.
//!
//! # Resuming and closing
//!
//! A unit declared with leaves already complete ([`Unit::completed`]) runs only the rest, so
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
//! use std::{sync::Arc, time::Duration};
//!
//! use whelm::{
//!     Config, DagConfig, DagScheduler, Input, Output, Policy, Resources, SLOTS, Scheduler,
//!     TemplateNode, TemplateSpec, Time, Unit, WorkerState,
//! };
//!
//! // Node 0 loads (on the caller), nodes 1 and 2 compute after it.
//! let template = TemplateSpec {
//!     nodes: vec![
//!         TemplateNode::Local(Duration::from_secs(1)),
//!         TemplateNode::Job(Duration::from_secs(2)),
//!         TemplateNode::Job(Duration::from_secs(2)),
//!     ],
//!     edges: vec![(0, 1), (0, 2)],
//! };
//! let template = Arc::new(template.build().unwrap());
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
//! dag.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//!
//! // Unit 1000 has jobs 0..3, unit 2000 jobs 10..13 and waits for unit 1000.
//! let first = Unit {
//!     id: 1000,
//!     base: 0,
//!     template: template.clone(),
//!     ..Default::default()
//! };
//! let second = Unit {
//!     id: 2000,
//!     base: 10,
//!     template,
//!     deps: vec![1000],
//!     ..Default::default()
//! };
//! dag.declare([first, second], Time::ORIGIN).unwrap();
//! // Ranks run through both units: 1 + 2 in each.
//! assert_eq!(
//!     [0, 1, 10].map(|j| dag.rank(j)),
//!     [6, 5, 3].map(|s| Some(Duration::from_secs(s)))
//! );
//! let stats = dag.dag_stats();
//! assert_eq!((stats.units, stats.open, stats.frames), (2, 1, 1));
//!
//! assert_eq!(dag.poll(Time::ORIGIN), vec![Output::RunLocal { job: 0 }]);
//! dag.handle(
//!     Input::Done { job: 0, attempt: 0 },
//!     Time(Duration::from_secs(1)),
//! );
//! let start = |job| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker: 1,
//! };
//! assert_eq!(dag.poll(Time(Duration::from_secs(1))), vec![start(1)]);
//! dag.handle(
//!     Input::Done { job: 1, attempt: 1 },
//!     Time(Duration::from_secs(3)),
//! );
//! assert_eq!(dag.poll(Time(Duration::from_secs(3))), vec![start(2)]);
//! // The last job of unit 1000 completes it, which enters unit 2000.
//! dag.handle(
//!     Input::Done { job: 2, attempt: 1 },
//!     Time(Duration::from_secs(5)),
//! );
//! assert_eq!(
//!     dag.poll(Time(Duration::from_secs(5))),
//!     vec![Output::RunLocal { job: 10 }]
//! );
//! let stats = dag.dag_stats();
//! assert_eq!((stats.units, stats.open, stats.frames), (1, 1, 1));
//! ```

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    sync::Arc,
};

use crate::{
    Attempt, Explanation, GaveUp, Input, JobId, JobSpec, Output, Policy, PolicyStats, Status, Time,
    WorkerId,
};

mod config;
mod declare;
mod frame;
mod lifecycle;
mod rank;
mod report;
#[cfg(feature = "serde")]
mod snapshot;
mod template;
#[cfg(test)]
mod tests;
mod unit;

pub use config::{DagConfig, DagError};
use frame::{Frame, Work};
pub use report::DagStats;
#[cfg(feature = "serde")]
pub use snapshot::DagSnapshot;
pub use template::{DagTemplate, TemplateNode, TemplateSpec};
pub use unit::{DagJob, NodeSource, Unit};

/// The scheduler's [`NodeSource`], opaque to `Debug`.
#[derive(Clone)]
struct Source(Arc<dyn NodeSource>);

impl std::fmt::Debug for Source {
    /// The source is opaque.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NodeSource(..)")
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
            spec: JobSpec {
                id,
                ..Default::default()
            },
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
/// A job the inner policy gives up on ([`Output::GaveUp`]) or rejects ([`Output::Rejected`]),
/// both passed through, is held again: its dependents stay pending until the caller
/// [`release`](Self::release)s it (another round of attempts) or [`cancel`](Self::cancel)s it.
///
/// With one attempt per job, job 1's failure is a give-up; releasing it runs it again, and its
/// dependent follows.
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{
///     Config, DagConfig, DagJob, DagScheduler, FailKind, GaveUp, Input, JobSpec, Output, Policy,
///     Resources, RetryConfig, SLOTS, Scheduler, Time, WorkerState,
/// };
///
/// let config = Config {
///     retry: RetryConfig { max_attempts: 1 },
///     ..Config::fifo()
/// };
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(config));
/// dag.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         capacity: Resources::new().with(SLOTS, 1),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// let job = |id, deps| DagJob {
///     spec: JobSpec {
///         id,
///         ..Default::default()
///     },
///     deps,
///     ..Default::default()
/// };
/// dag.declare([job(1, vec![]), job(2, vec![1])], Time::ORIGIN)
///     .unwrap();
/// assert_eq!(
///     dag.poll(Time::ORIGIN),
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
///     Time(Duration::from_secs(1)),
/// );
/// let out = dag.poll(Time(Duration::from_secs(1)));
/// assert!(matches!(out[..], [Output::GaveUp(GaveUp { job: 1, .. })]));
/// assert_eq!(dag.explain(1).unwrap().status, whelm::Status::Held);
///
/// assert!(dag.release(1, Time(Duration::from_secs(2))));
/// assert_eq!(
///     dag.poll(Time(Duration::from_secs(2))),
///     vec![Output::Start {
///         job: 1,
///         attempt: 1,
///         worker: 1
///     }]
/// );
/// dag.handle(
///     Input::Done { job: 1, attempt: 1 },
///     Time(Duration::from_secs(3)),
/// );
/// assert_eq!(
///     dag.poll(Time(Duration::from_secs(3))),
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
    now: Time,
}

impl<P: Policy> DagScheduler<P> {
    /// A DAG layer in front of `policy`.
    ///
    /// Jobs without dependencies need not be declared: an [`Input::Submit`] goes straight to the
    /// inner policy.
    ///
    /// ```
    /// # use whelm::{
    /// #     Config, DagConfig, DagScheduler, Input, JobSpec, Output, Policy, Resources, SLOTS,
    /// #     Scheduler, Time, WorkerState,
    /// # };
    /// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// dag.handle(
    ///     Input::Worker(WorkerState {
    ///         id: 1,
    ///         capacity: Resources::new().with(SLOTS, 1),
    ///         ..Default::default()
    ///     }),
    ///     Time::ORIGIN,
    /// );
    /// let job = JobSpec {
    ///     id: 7,
    ///     ..Default::default()
    /// };
    /// dag.handle(Input::Submit(job), Time::ORIGIN);
    /// assert_eq!(
    ///     dag.poll(Time::ORIGIN),
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
            now: Time::ORIGIN,
        }
    }

    /// This scheduler, describing [`sourced`](field@Unit::sourced) units' leaves with `source`.
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
    /// # use whelm::{
    /// #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources,
    /// #     SLOTS, Scheduler, Time, WorkerState,
    /// # };
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # let capacity = Resources::new().with(SLOTS, 1);
    /// # let worker = WorkerState { id: 1, capacity, ..Default::default() };
    /// # dag.handle(Input::Worker(worker), Time::ORIGIN);
    /// # let job = |id, deps| DagJob {
    /// #     spec: JobSpec { id, ..Default::default() },
    /// #     deps,
    /// #     ..Default::default()
    /// # };
    /// dag.declare([job(1, vec![]), job(2, vec![1])], Time::ORIGIN)
    ///     .unwrap();
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
}

impl<P: Policy> Policy for DagScheduler<P> {
    /// Forwarded to the inner policy, with this layer's bookkeeping: a done attempt completes its
    /// job here (attempt 0 completes a local job, which the inner policy never sees), and a
    /// cancellation cascades to the job's dependents ([`DagScheduler::cancel`]).
    fn handle(&mut self, input: Input, now: Time) {
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
    fn poll(&mut self, now: Time) -> Vec<Output> {
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
                Output::GaveUp(GaveUp { job, .. }) | Output::Rejected { job, .. } => {
                    self.live.remove(job);
                    self.hold_again(*job);
                }
                _ => {}
            }
            out.push(o);
        }
        out
    }

    /// Forwarded to the inner policy.
    fn next_wakeup(&self) -> Option<Time> {
        self.policy.next_wakeup()
    }

    /// The DAG's reason while the job is not submitted, the inner policy's afterwards. Units are
    /// explained by their id.
    fn explain(&self, job: JobId) -> Option<Explanation> {
        match self.locate(job) {
            Some(Loc::Unit(u)) => Some(Explanation {
                unit: true,
                ..Explanation::new(job, self.unit_status(u))
            }),
            Some(Loc::Leaf { unit, leaf }) => self.explain_leaf(unit, leaf, job),
            None if self.is_completed(job) => Some(Explanation::new(job, Status::Completed)),
            None => self.policy.explain(job),
        }
    }

    /// Forwarded to the inner policy.
    fn stats(&self) -> PolicyStats {
        self.policy.stats()
    }
}
