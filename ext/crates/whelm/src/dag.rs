//! The optional dependency layer in front of a [`Policy`].

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    sync::Arc,
};

use crate::{Attempt, Input, Instant, JobId, JobSpec, Output, Policy, PolicyStats, WorkerId};

mod frame;
#[cfg(feature = "serde")]
mod snapshot;
mod template;

use frame::{COMPLETE, Frame, HELD, SUBMITTED, Work};
#[cfg(feature = "serde")]
pub use snapshot::DagSnapshot;
pub use template::{DagTemplate, TemplateNode};

/// A plain job with dependencies: a [`Unit`] of a one-node template, whose one leaf is the job.
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
    /// A job to run, with default work.
    pub fn new(spec: JobSpec, deps: Vec<JobId>) -> Self {
        Self {
            spec,
            deps,
            work_estimate: None,
            passthrough: false,
            local: false,
        }
    }

    /// A passthrough job (see [`DagJob::passthrough`]) of group `group`, worth `work` in ranks.
    pub fn passthrough(id: JobId, group: u64, deps: Vec<JobId>, work: f64) -> Self {
        Self {
            spec: JobSpec::new(id, crate::Resources::ZERO, group),
            deps,
            work_estimate: Some(work),
            passthrough: true,
            local: false,
        }
    }

    /// Make it a local job (see [`DagJob::local`]).
    pub fn local(mut self) -> Self {
        self.local = true;
        self
    }

    /// Set the work estimate.
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
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    /// This unit with its leaves described by the scheduler's [`NodeSource`].
    pub fn sourced(mut self) -> Self {
        self.sourced = true;
        self
    }

    /// This unit with leaves `completed` already complete.
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

/// Per-leaf data of [`sourced`](Unit::sourced) units, computed on demand rather than stored: a
/// unit costs the same however many leaves it has until it materialises.
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

/// Errors from [`DagScheduler::declare`] and [`DagScheduler::close`]. A failed declaration
/// changes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DagError {
    /// The declaration would close a dependency cycle through this unit (or template node).
    Cycle {
        /// A unit on the cycle.
        job: JobId,
    },
    /// The unit is already declared (or completed), or appears twice in the batch.
    Duplicate(JobId),
    /// This id is both a unit's id or dependency and a leaf of another unit.
    Overlap(JobId),
    /// No live unit has this id or leaf.
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
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DagConfig {
    /// Scale of a unit declared without one, i.e. the work of a [`DagJob`] without an estimate.
    /// Default 1.
    pub default_work: f64,
    /// Ranks between units are maintained approximately: a rank increase smaller than this
    /// fraction is not propagated to the unit's dependencies. Bounds the cost of growing the
    /// graph; ranks within a unit are exact. Default 0.01.
    pub rank_epsilon: f64,
    /// Submit jobs to the policy as soon as they are ready (the default). When false, ready jobs
    /// are held and announced by [`Output::Ready`]; the caller submits each with
    /// [`DagScheduler::release`] when it is actually sendable (e.g. after coordinator-side
    /// preparation).
    pub auto_submit: bool,
    /// Announce passthrough leaves, and units other than plain jobs, as they complete, with
    /// [`Output::Passed`]. Default false.
    #[cfg_attr(feature = "serde", serde(default))]
    pub record_passthrough: bool,
    /// Maintain units' ranks as the graph grows and work changes, and submit each job with its
    /// upward rank (the critical path below it) as [`JobSpec::rank`] unless it has one. Whether
    /// ranks order anything is up to the policy ([`OrderTerm::Rank`](crate::OrderTerm::Rank)).
    /// Without them, declaring and re-estimating skip all rank propagation, which on long
    /// dependency chains is most of the cost. Default true.
    #[cfg_attr(feature = "serde", serde(default = "yes"))]
    pub track_ranks: bool,
}

/// `true`, for serde defaults.
#[cfg(feature = "serde")]
fn yes() -> bool {
    true
}

impl Default for DagConfig {
    /// The defaults documented on each field.
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

/// Counters describing the DAG layer's state.
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
    pub fn with_source(mut self, source: Arc<dyn NodeSource>) -> Self {
        self.source = Some(Source(source));
        self
    }

    /// The wrapped policy.
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

    /// Declare units (or plain [`DagJob`]s). The graph grows during the run; dependencies may be
    /// forward references. Rejects (and leaves no trace of) a batch that would create a cycle,
    /// redeclare a unit or overlap another unit's ids. Units whose dependencies are all complete
    /// are entered at once.
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

    /// Drain this layer's own announcements ([`Output::RunLocal`], [`Output::Ready`] and
    /// [`Output::Passed`]) without polling the inner policy, so the caller can act on them
    /// (release, declare, close) before anything is placed in the same instant. The next
    /// [`poll`](Policy::poll) returns the announcements made since, in order, ahead of the inner
    /// policy's outputs.
    pub fn announcements(&mut self) -> Vec<Output> {
        std::mem::take(&mut self.outbox)
    }

    /// Submit a ready, held job to the policy (only meaningful without `auto_submit`). Returns
    /// false if the job is not held.
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
    /// completed from now on. Use when ids are allocated increasingly and everything below `floor`
    /// is known to be done, to keep memory proportional to the live frontier.
    pub fn forget_completed_below(&mut self, floor: JobId) {
        if floor > self.completed_floor {
            self.completed_floor = floor;
            self.completed.retain(|&j| j >= floor);
        }
    }

    /// The upward rank of a unit (named by its id: its critical path plus the rank below it) or of
    /// a job (a leaf: its work plus the longest chain of work below it, through the enclosing
    /// units and their dependents). `None` for unknown ids.
    pub fn rank(&self, job: JobId) -> Option<f64> {
        match self.locate(job)? {
            Loc::Unit(u) => Some(self.unit(u).top()),
            Loc::Leaf { unit, leaf } => Some(self.leaf_rank(unit, leaf)),
        }
    }

    /// Change a unit's scale (a plain job's work, e.g. once its real size is known) and re-rank
    /// it and its dependencies, up or down. Returns false for unknown ids and leaves of other
    /// units. Jobs already handed to the policy keep the rank they were submitted with.
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

    /// Counters.
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

    /// Cancel a unit (named by its id or any of its jobs) and, transitively, every unit depending
    /// on it (they can never run). Returns the cancelled units' ids (a plain job's is the job's).
    /// Each of their jobs the inner policy has is cancelled there (its live attempts are stopped).
    /// An id this layer does not know is cancelled in the inner policy. [`Input::Cancel`] does the
    /// same.
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

    /// Close a unit early (named by its id or any of its jobs), e.g. when its remaining jobs are
    /// known to be no-ops. If it is open, its jobs that have not started complete as no-ops
    /// (waiting ones are withdrawn from the policy) and the unit completes; it returns the jobs
    /// already running, whose workers keep their resources until each one's attempt ends
    /// ([`Input::Done`], [`Input::Failed`], or its worker leaving), which then changes nothing
    /// else; such a job is not retried. A unit not entered yet completes as soon as its
    /// dependencies do, without running anything.
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

    /// The first start of `job` on worker 1.
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
