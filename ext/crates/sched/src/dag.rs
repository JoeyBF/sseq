//! The optional dependency layer in front of a [`Policy`].

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use petgraph::{
    Direction::{Incoming, Outgoing},
    stable_graph::{NodeIndex, StableDiGraph},
    visit::EdgeRef,
};

use crate::{Instant, JobId, JobSpec, Policy, PolicyStats, WorkerId, WorkerState};

/// A job with dependencies.
#[derive(Clone, Debug, PartialEq)]
pub struct DagJob {
    /// The job, as it will be submitted to the policy.
    pub spec: JobSpec,
    /// Jobs that must all complete before this one is ready. A dependency may name a job that is
    /// not declared yet (a forward reference): it is pending until declared and completed. A
    /// dependency on a job that already completed is satisfied.
    pub deps: Vec<JobId>,
    /// Relative cost, for ranking. `None` uses [`DagConfig::default_work`].
    pub work_estimate: Option<f64>,
    /// A pure synchronisation point ("group G is done"): when ready it completes by itself
    /// instead of being submitted to the policy. Its `work_estimate` still counts in ranks, which
    /// lets a placeholder stand for work that is not expanded yet (lower it later with
    /// [`DagScheduler::update_work`]).
    pub passthrough: bool,
}

impl DagJob {
    /// A job to run, with default work.
    pub fn new(spec: JobSpec, deps: Vec<JobId>) -> Self {
        Self {
            spec,
            deps,
            work_estimate: None,
            passthrough: false,
        }
    }

    /// A passthrough job (see [`DagJob::passthrough`]) of group `group`, worth `work` in ranks.
    pub fn passthrough(id: JobId, group: u64, deps: Vec<JobId>, work: f64) -> Self {
        Self {
            spec: JobSpec::new(id, crate::Resources::ZERO, group),
            deps,
            work_estimate: Some(work),
            passthrough: true,
        }
    }

    /// Set the work estimate.
    pub fn with_work(mut self, work: f64) -> Self {
        self.work_estimate = Some(work);
        self
    }
}

/// A reusable dependency structure over nodes `0..len`, instantiated once per group with
/// [`DagScheduler::declare_template`] (e.g. one signature DAG per subalgebra profile, shared by
/// every bidegree with that profile). Building it checks acyclicity once.
#[derive(Clone, Debug)]
pub struct DagTemplate {
    succ: Vec<Vec<u32>>,
    pred: Vec<Vec<u32>>,
    /// A topological order.
    topo: Vec<u32>,
}

impl DagTemplate {
    /// A template with `len` nodes and the given edges `(from, to)`: `to` depends on `from`.
    /// Duplicate edges are merged; out-of-range nodes panic; a cycle is an error naming a node on
    /// it.
    pub fn new(len: usize, edges: impl IntoIterator<Item = (u32, u32)>) -> Result<Self, DagError> {
        let mut succ: Vec<Vec<u32>> = vec![Vec::new(); len];
        for (a, b) in edges {
            assert!(
                (a as usize) < len && (b as usize) < len,
                "edge ({a}, {b}) out of range"
            );
            succ[a as usize].push(b);
        }
        let mut pred: Vec<Vec<u32>> = vec![Vec::new(); len];
        for (a, row) in succ.iter_mut().enumerate() {
            row.sort_unstable();
            row.dedup();
            for &b in row.iter() {
                pred[b as usize].push(a as u32);
            }
        }
        // Kahn's algorithm.
        let mut indeg: Vec<usize> = pred.iter().map(Vec::len).collect();
        let mut topo: Vec<u32> = (0..len as u32)
            .filter(|&i| indeg[i as usize] == 0)
            .collect();
        let mut i = 0;
        while i < topo.len() {
            for &b in &succ[topo[i] as usize] {
                indeg[b as usize] -= 1;
                if indeg[b as usize] == 0 {
                    topo.push(b);
                }
            }
            i += 1;
        }
        if topo.len() < len {
            let job = indeg.iter().position(|&d| d > 0).unwrap() as JobId;
            return Err(DagError::Cycle { job });
        }
        Ok(Self { succ, pred, topo })
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.succ.len()
    }

    /// Whether the template has no nodes.
    pub fn is_empty(&self) -> bool {
        self.succ.is_empty()
    }

    /// Number of (deduplicated) edges.
    pub fn edge_count(&self) -> usize {
        self.succ.iter().map(Vec::len).sum()
    }

    /// The nodes in a topological order (every node after all its predecessors).
    pub fn topological_order(&self) -> &[u32] {
        &self.topo
    }

    /// The nodes depending directly on `node`.
    pub fn successors(&self, node: usize) -> &[u32] {
        &self.succ[node]
    }

    /// The nodes `node` depends on directly.
    pub fn predecessors(&self, node: usize) -> &[u32] {
        &self.pred[node]
    }

    /// Nodes with no dependency inside the template (they receive a group's entry dependencies).
    pub fn sources(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).filter(|&i| self.pred[i].is_empty())
    }

    /// Nodes nothing in the template depends on (a group is done when they are).
    pub fn sinks(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).filter(|&i| self.succ[i].is_empty())
    }

    /// The same partial order with every implied edge removed (an edge `a -> c` is implied when
    /// `c` is reachable from another successor of `a`). Readiness and critical paths are
    /// unchanged; instances get far fewer edges. Takes `O(len^2 / 8)` bytes of scratch.
    pub fn transitive_reduction(&self) -> DagTemplate {
        let n = self.len();
        let words = n.div_ceil(64);
        let mut position = vec![0usize; n];
        for (p, &v) in self.topo.iter().enumerate() {
            position[v as usize] = p;
        }
        // reach[v]: the strict descendants of v, filled in reverse topological order.
        let mut reach = vec![0u64; n * words];
        let mut edges = Vec::new();
        let mut acc = vec![0u64; words];
        for &v in self.topo.iter().rev() {
            let v = v as usize;
            let mut succ = self.succ[v].clone();
            // Earlier successors (in topological order) are the only ones that can reach later
            // ones, so a successor already covered by the earlier ones is implied.
            succ.sort_unstable_by_key(|&c| position[c as usize]);
            acc.fill(0);
            for c in succ {
                let c = c as usize;
                if acc[c / 64] >> (c % 64) & 1 == 1 {
                    continue;
                }
                edges.push((v as u32, c as u32));
                acc[c / 64] |= 1 << (c % 64);
                for (a, r) in acc.iter_mut().zip(&reach[c * words..(c + 1) * words]) {
                    *a |= r;
                }
            }
            reach[v * words..(v + 1) * words].copy_from_slice(&acc);
        }
        DagTemplate::new(n, edges).expect("a sub-DAG of a DAG is acyclic")
    }

    /// Nodes on a longest chain of `work` (CPOP's critical nodes): those whose longest path
    /// from a source plus longest path to a sink equals the critical path, within a relative
    /// tolerance `tol`.
    pub fn critical_nodes(&self, work: impl Fn(usize) -> f64, tol: f64) -> Vec<bool> {
        let n = self.len();
        let w: Vec<f64> = (0..n).map(&work).collect();
        let mut below = vec![0.0f64; n];
        for &v in self.topo.iter().rev() {
            let v = v as usize;
            let tail = self.succ[v]
                .iter()
                .map(|&c| below[c as usize])
                .fold(0.0, f64::max);
            below[v] = w[v] + tail;
        }
        let mut above = vec![0.0f64; n]; // longest path ending just before v
        for &v in &self.topo {
            let v = v as usize;
            above[v] = self.pred[v]
                .iter()
                .map(|&p| above[p as usize] + w[p as usize])
                .fold(0.0, f64::max);
        }
        let cp = below.iter().copied().fold(0.0, f64::max);
        (0..n)
            .map(|v| cp > 0.0 && above[v] + below[v] >= cp * (1.0 - tol))
            .collect()
    }

    /// The longest chain of `work` through the template: a group's duration with unlimited
    /// workers, i.e. the cost to give its placeholder.
    pub fn critical_path(&self, work: impl Fn(usize) -> f64) -> f64 {
        let mut below = vec![0.0f64; self.len()];
        let mut best = 0.0f64;
        for &n in self.topo.iter().rev() {
            let n = n as usize;
            let tail = self.succ[n]
                .iter()
                .map(|&c| below[c as usize])
                .fold(0.0, f64::max);
            below[n] = work(n) + tail;
            best = best.max(below[n]);
        }
        best
    }
}

/// Errors from [`Dag::declare`]. A failed declaration changes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DagError {
    /// The declaration would close a dependency cycle through this job.
    Cycle {
        /// A job on the cycle.
        job: JobId,
    },
    /// The job is already declared (or completed), or appears twice in the batch.
    Duplicate(JobId),
}

impl std::fmt::Display for DagError {
    /// A one-line description of the error.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cycle { job } => write!(f, "declaring job {job} would create a dependency cycle"),
            Self::Duplicate(job) => write!(f, "job {job} is already declared"),
        }
    }
}

impl std::error::Error for DagError {}

/// Configuration for [`DagScheduler`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DagConfig {
    /// Rank jobs by the critical path below them instead of "oldest group first": a job without an
    /// explicit [`JobSpec::priority`] is submitted with priority `-(rank * rank_scale)`, so longer
    /// remaining chains are more urgent. Default false.
    pub rank_priority: bool,
    /// Converts ranks (in work units) to integer priorities. Default 1000.
    pub rank_scale: f64,
    /// Work of a job declared without an estimate. Default 1.
    pub default_work: f64,
    /// Ranks are maintained approximately: a rank increase smaller than this fraction is not
    /// propagated to the job's dependencies. Bounds the cost of growing the graph. Default 0.01.
    pub rank_epsilon: f64,
    /// Submit jobs to the policy as soon as they are ready (the default). When false, ready jobs
    /// are queued; the caller collects them with [`DagScheduler::take_ready`] and submits each
    /// with [`DagScheduler::release`] when it is actually sendable (e.g. after coordinator-side
    /// preparation).
    pub auto_submit: bool,
    /// Record passthrough jobs as they complete, for [`DagScheduler::take_passed`]. Default false
    /// (a caller that never drains the list would grow it without bound).
    #[cfg_attr(feature = "serde", serde(default))]
    pub record_passthrough: bool,
}

impl Default for DagConfig {
    /// The defaults documented on each field.
    fn default() -> Self {
        Self {
            rank_priority: false,
            rank_scale: 1000.0,
            default_work: 1.0,
            rank_epsilon: 0.01,
            auto_submit: true,
            record_passthrough: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum State {
    /// Referenced as a dependency, not declared yet.
    Undeclared,
    /// Declared, some dependency not completed.
    Pending,
    /// All dependencies completed, waiting for [`DagScheduler::release`].
    Held,
    /// Handed to the policy (waiting or running there).
    Submitted,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Node {
    id: JobId,
    state: State,
    /// `None` while undeclared.
    spec: Option<JobSpec>,
    /// Dependencies not completed yet (= in-degree).
    unmet: u32,
    work: f64,
    /// Upward rank: work plus the longest chain of work among descendants (approximate).
    rank: f64,
    /// Completes by itself when ready (see [`DagJob::passthrough`]).
    #[cfg_attr(feature = "serde", serde(default))]
    passthrough: bool,
}

/// Counters describing the DAG layer's state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DagStats {
    /// Declared jobs whose dependencies are not all complete.
    pub pending: usize,
    /// Jobs referenced as dependencies but not declared yet.
    pub undeclared: usize,
    /// Ready jobs waiting for `release` (only without `auto_submit`).
    pub held: usize,
    /// Jobs handed to the policy and not completed.
    pub submitted: usize,
    /// Dependency edges between live jobs.
    pub edges: usize,
    /// Completed job ids remembered so that later dependencies on them are satisfied.
    pub completed_remembered: usize,
    /// Declared group placeholders.
    pub placeholders: usize,
}

/// The dependency layer's interface. [`DagScheduler`] implements it over any [`Policy`].
pub trait Dag {
    /// Declare jobs. The graph grows during the run; dependencies may be forward references.
    /// Rejects (and leaves no trace of) a batch that would create a cycle or redeclare a job.
    /// Jobs whose dependencies are all complete become ready immediately.
    fn declare(&mut self, jobs: Vec<DagJob>, now: Instant) -> Result<(), DagError>;
    /// Declare work known to come but not yet expandable: group `group` will exist after the
    /// groups `after_groups` complete and costs about `cost`. Only ranks use this. Redeclaring a
    /// group replaces its placeholder.
    fn declare_group_placeholder(&mut self, group: u64, after_groups: Vec<u64>, cost: f64);
    /// A job completed: its dependents may become ready. Forwards to the policy.
    fn completed(&mut self, job: JobId, now: Instant);
    /// Cancel a job and, transitively, every job depending on it (they can never run). Returns the
    /// cancelled ids. Forwards each submitted one to the policy.
    fn cancel(&mut self, job: JobId) -> Vec<JobId>;
    /// Passthrough of [`Policy::worker_update`].
    fn worker_update(&mut self, w: WorkerState, now: Instant);
    /// Passthrough of [`Policy::worker_gone`]. Resubmit its jobs with
    /// [`DagScheduler::resubmit`].
    fn worker_gone(&mut self, w: WorkerId, now: Instant);
    /// Passthrough of [`Policy::dispatch`].
    fn dispatch(&mut self, now: Instant) -> Vec<(JobId, WorkerId)>;
    /// Why a job is not running: unmet dependencies, held, or the policy's explanation.
    fn explain(&self, job: JobId) -> Option<String>;
    /// Passthrough of [`Policy::stats`].
    fn stats(&self) -> PolicyStats;
}

/// A [`Dag`] in front of a [`Policy`].
///
/// Jobs are declared with their dependencies, possibly long before they are ready; a job is
/// submitted to the policy when its last dependency completes. The graph lives in a petgraph
/// [`StableGraph`](petgraph::stable_graph::StableGraph) (edges point from a dependency to its
/// dependent); completed jobs are removed from it, so its size tracks the live frontier rather
/// than the whole run.
#[derive(Clone, Debug)]
pub struct DagScheduler<P> {
    config: DagConfig,
    policy: P,
    graph: StableDiGraph<Node, ()>,
    index: HashMap<JobId, NodeIndex>,
    completed: HashSet<JobId>,
    completed_floor: JobId,
    /// Ready ids not yet returned by `take_ready`, in readiness order.
    newly_ready: Vec<JobId>,
    placeholders: BTreeMap<u64, (Vec<u64>, f64)>,
    /// group -> placeholder groups that wait for it.
    waiters: HashMap<u64, Vec<u64>>,
    tail_memo: HashMap<u64, f64>,
    /// Ready passthrough nodes waiting for `drain_passthrough`.
    passing: Vec<NodeIndex>,
    /// Completed passthrough ids, for `take_passed`.
    passed: Vec<JobId>,
    now: Instant,
}

impl<P: Policy> DagScheduler<P> {
    /// A DAG layer in front of `policy`.
    pub fn new(config: DagConfig, policy: P) -> Self {
        Self {
            config,
            policy,
            graph: StableDiGraph::default(),
            index: HashMap::new(),
            completed: HashSet::new(),
            completed_floor: 0,
            newly_ready: Vec::new(),
            placeholders: BTreeMap::new(),
            waiters: HashMap::new(),
            tail_memo: HashMap::new(),
            passing: Vec::new(),
            passed: Vec::new(),
            now: 0.0,
        }
    }

    /// The wrapped policy.
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// The wrapped policy, mutably (e.g. to call [`Policy::submit`] for jobs outside the DAG).
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    /// Ids that became ready since the last call, in readiness order. With `auto_submit` they are
    /// already submitted; otherwise submit each with [`release`](Self::release).
    pub fn take_ready(&mut self) -> Vec<JobId> {
        std::mem::take(&mut self.newly_ready)
    }

    /// Submit a ready, held job to the policy (only meaningful without `auto_submit`). Returns
    /// false if the job is not held.
    pub fn release(&mut self, job: JobId, now: Instant) -> bool {
        self.now = now;
        match self.index.get(&job) {
            Some(&n) if self.graph[n].state == State::Held => {
                self.submit_node(n, now);
                true
            }
            _ => false,
        }
    }

    /// Submit a job again (e.g. after its worker left). Returns false unless the job was submitted
    /// and has not completed.
    pub fn resubmit(&mut self, job: JobId, now: Instant) -> bool {
        self.now = now;
        match self.index.get(&job) {
            Some(&n) if self.graph[n].state == State::Submitted => {
                self.submit_node(n, now);
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

    /// Remove a group placeholder.
    pub fn remove_group_placeholder(&mut self, group: u64) {
        if let Some((after, _)) = self.placeholders.remove(&group) {
            for h in after {
                if let Some(v) = self.waiters.get_mut(&h) {
                    v.retain(|&g| g != group);
                }
            }
            self.tail_memo.clear();
        }
    }

    /// The job's current upward rank (its work plus the longest chain of descendants' work), plus
    /// the estimated cost of placeholder groups waiting on its group. `None` for unknown jobs.
    pub fn rank(&mut self, job: JobId) -> Option<f64> {
        let &n = self.index.get(&job)?;
        let group = self.graph[n].spec.as_ref().map(|s| s.group);
        Some(self.graph[n].rank + group.map_or(0.0, |g| self.group_tail(g)))
    }

    /// Counters.
    pub fn dag_stats(&self) -> DagStats {
        let mut s = DagStats {
            edges: self.graph.edge_count(),
            completed_remembered: self.completed.len(),
            placeholders: self.placeholders.len(),
            ..DagStats::default()
        };
        for n in self.graph.node_weights() {
            match n.state {
                State::Undeclared => s.undeclared += 1,
                State::Pending => s.pending += 1,
                State::Held => s.held += 1,
                State::Submitted => s.submitted += 1,
            }
        }
        s
    }

    /// Whether `job` is known to have completed (remembered, or below the floor).
    fn is_completed(&self, job: JobId) -> bool {
        job < self.completed_floor || self.completed.contains(&job)
    }

    /// A placeholder node for a job named as a dependency before being declared.
    fn undeclared(&mut self, id: JobId) -> NodeIndex {
        let n = self.graph.add_node(Node {
            id,
            state: State::Undeclared,
            spec: None,
            unmet: 0,
            work: 0.0,
            rank: 0.0,
            passthrough: false,
        });
        self.index.insert(id, n);
        n
    }

    /// A job on a cycle reachable from `starts`, if any (iterative three-colour DFS along
    /// dependent edges; only the part of the graph reachable from the new jobs is visited).
    fn find_cycle(&self, starts: &[NodeIndex]) -> Option<JobId> {
        // 1 = on the DFS stack, 2 = finished.
        let mut colour: HashMap<NodeIndex, u8> = HashMap::new();
        for &s in starts {
            if colour.contains_key(&s) {
                continue;
            }
            colour.insert(s, 1);
            let mut stack = vec![(
                s,
                self.graph
                    .neighbors_directed(s, Outgoing)
                    .collect::<Vec<_>>(),
            )];
            while let Some((node, children)) = stack.last_mut() {
                match children.pop() {
                    Some(c) => match colour.get(&c) {
                        Some(1) => return Some(self.graph[c].id),
                        Some(_) => {}
                        None => {
                            colour.insert(c, 1);
                            let next = self.graph.neighbors_directed(c, Outgoing).collect();
                            stack.push((c, next));
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

    /// Raise ancestors' ranks after `start`'s rank grew.
    fn propagate_rank(&mut self, start: NodeIndex) {
        let eps = self.config.rank_epsilon.max(0.0);
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            let r = self.graph[n].rank;
            let preds: Vec<_> = self.graph.neighbors_directed(n, Incoming).collect();
            for p in preds {
                let cand = self.graph[p].work + r;
                let old = self.graph[p].rank;
                if cand > old * (1.0 + eps) && cand > old {
                    self.graph[p].rank = cand;
                    stack.push(p);
                }
            }
        }
    }

    /// The longest chain of placeholder cost waiting on `group`, memoised until placeholders change.
    fn group_tail(&mut self, group: u64) -> f64 {
        /// Memoised depth-first longest path through the placeholder waiters of `g`.
        fn go(
            g: u64,
            placeholders: &BTreeMap<u64, (Vec<u64>, f64)>,
            waiters: &HashMap<u64, Vec<u64>>,
            memo: &mut HashMap<u64, f64>,
            visiting: &mut HashSet<u64>,
        ) -> f64 {
            if let Some(&t) = memo.get(&g) {
                return t;
            }
            if !visiting.insert(g) {
                return 0.0; // placeholder cycle: ignore the back edge
            }
            let mut best = 0.0f64;
            for &p in waiters.get(&g).map_or(&[][..], |v| v) {
                let cost = placeholders.get(&p).map_or(0.0, |x| x.1);
                best = best.max(cost + go(p, placeholders, waiters, memo, visiting));
            }
            visiting.remove(&g);
            memo.insert(g, best);
            best
        }
        if self.placeholders.is_empty() {
            return 0.0;
        }
        go(
            group,
            &self.placeholders,
            &self.waiters,
            &mut self.tail_memo,
            &mut HashSet::new(),
        )
    }

    /// Hand a ready job to the policy, with its rank as priority if configured.
    fn submit_node(&mut self, n: NodeIndex, now: Instant) {
        self.graph[n].state = State::Submitted;
        let mut spec = self.graph[n]
            .spec
            .clone()
            .expect("submitting an undeclared job");
        if spec.work.is_none() {
            // The DAG layer's estimate feeds speed-aware placement (earliest finish).
            spec.work = Some(self.graph[n].work);
        }
        if self.config.rank_priority && spec.priority.is_none() {
            let rank = self.graph[n].rank + self.group_tail(spec.group);
            let p = -(rank * self.config.rank_scale).round();
            spec.priority = Some(p.clamp(i64::MIN as f64, i64::MAX as f64) as i64);
        }
        self.policy.submit(spec, now);
    }

    /// Record that a job has no unmet dependencies left, and submit or hold it.
    fn make_ready(&mut self, n: NodeIndex, now: Instant) {
        if self.graph[n].passthrough {
            self.graph[n].state = State::Held;
            self.passing.push(n);
            return;
        }
        self.newly_ready.push(self.graph[n].id);
        if self.config.auto_submit {
            self.submit_node(n, now);
        } else {
            self.graph[n].state = State::Held;
        }
    }

    /// Undo a rejected declaration: its edges, its new nodes and its filled-in forward references.
    fn rollback(&mut self, batch: &[NodeIndex], created: &[NodeIndex]) {
        for &n in batch {
            let edges: Vec<_> = self
                .graph
                .edges_directed(n, Incoming)
                .map(|e| e.id())
                .collect();
            for e in edges {
                self.graph.remove_edge(e);
            }
            let node = &mut self.graph[n];
            node.state = State::Undeclared;
            node.spec = None;
            node.unmet = 0;
            node.work = 0.0;
            node.passthrough = false;
        }
        for &n in created {
            if let Some(node) = self.graph.remove_node(n) {
                self.index.remove(&node.id);
            }
        }
    }

    /// Release `job`'s dependents, drop its node and remember it as completed.
    fn finish(&mut self, job: JobId, now: Instant) {
        if let Some(n) = self.index.remove(&job) {
            let mut dependents: Vec<_> = self
                .graph
                .neighbors_directed(n, Outgoing)
                .map(|c| (self.graph[c].id, c))
                .collect();
            dependents.sort_unstable();
            self.graph.remove_node(n);
            for (_, c) in dependents {
                let node = &mut self.graph[c];
                node.unmet -= 1;
                if node.unmet == 0 && node.state == State::Pending {
                    self.make_ready(c, now);
                }
            }
        }
        if job >= self.completed_floor {
            self.completed.insert(job);
        }
    }

    /// Complete every ready passthrough job, and the ones that become ready as a result (a
    /// worklist, so long chains of them do not recurse).
    fn drain_passthrough(&mut self, now: Instant) {
        while let Some(n) = self.passing.pop() {
            let id = self.graph[n].id;
            if self.config.record_passthrough {
                self.passed.push(id);
            }
            self.finish(id, now);
        }
    }

    /// Passthrough jobs completed since the last call, in completion order (only with
    /// [`DagConfig::record_passthrough`]).
    pub fn take_passed(&mut self) -> Vec<JobId> {
        std::mem::take(&mut self.passed)
    }

    /// Change a declared job's work estimate (e.g. once its real size is known) and re-rank it
    /// and its ancestors, up or down. Returns false for unknown or completed jobs. Jobs already
    /// handed to the policy keep the priority they were submitted with.
    pub fn update_work(&mut self, job: JobId, work: f64) -> bool {
        let Some(&n) = self.index.get(&job) else {
            return false;
        };
        self.graph[n].work = work;
        let eps = self.config.rank_epsilon.max(0.0);
        let mut stack = vec![n];
        let mut first = true;
        while let Some(m) = stack.pop() {
            let below = self
                .graph
                .neighbors_directed(m, Outgoing)
                .map(|c| self.graph[c].rank)
                .fold(0.0, f64::max);
            let new = self.graph[m].work + below;
            let old = self.graph[m].rank;
            if first || (new - old).abs() > eps * old.abs().max(new.abs()) {
                self.graph[m].rank = new;
                stack.extend(self.graph.neighbors_directed(m, Incoming));
            }
            first = false;
        }
        true
    }

    /// Instantiate `template` as jobs: node `i` becomes job `id(i)`, built by `node(i)` (which
    /// supplies the spec, work and whether it is a passthrough; its id and `deps` are replaced).
    /// Template edges become dependencies, and every source node also depends on `entry`. Edges
    /// to and from existing jobs are still checked for cycles.
    pub fn declare_template(
        &mut self,
        template: &DagTemplate,
        id: impl Fn(usize) -> JobId,
        mut node: impl FnMut(usize) -> DagJob,
        entry: &[JobId],
        now: Instant,
    ) -> Result<(), DagError> {
        let jobs = (0..template.len())
            .map(|i| {
                let mut j = node(i);
                j.spec.id = id(i);
                j.deps = template.pred[i].iter().map(|&p| id(p as usize)).collect();
                if j.deps.is_empty() {
                    j.deps.extend_from_slice(entry);
                }
                j
            })
            .collect();
        self.declare(jobs, now)
    }

    /// Restore a scheduler from a snapshot, in front of a fresh `policy`. Jobs that were submitted
    /// (waiting or running in the old policy) are submitted again at `now`; held jobs stay held and
    /// are returned again by [`take_ready`](Self::take_ready).
    #[cfg(feature = "serde")]
    pub fn restore(snapshot: DagSnapshot, policy: P, now: Instant) -> Self {
        let mut s = Self::new(snapshot.config, policy);
        s.graph = snapshot.graph;
        s.completed = snapshot.completed.into_iter().collect();
        s.completed_floor = snapshot.completed_floor;
        for (g, (after, cost)) in snapshot.placeholders {
            s.declare_group_placeholder(g, after, cost);
        }
        let mut submitted = Vec::new();
        let mut held = Vec::new();
        for n in s.graph.node_indices() {
            let node = &s.graph[n];
            s.index.insert(node.id, n);
            match node.state {
                State::Submitted => submitted.push((node.id, n)),
                State::Held => held.push(node.id),
                _ => {}
            }
        }
        submitted.sort_unstable();
        for (_, n) in submitted {
            s.submit_node(n, now);
        }
        held.sort_unstable();
        s.newly_ready = held;
        s.now = now;
        s
    }

    /// A serialisable snapshot of the declared DAG (not of the policy): pending, held and
    /// submitted jobs with their edges, remembered completions and placeholders.
    #[cfg(feature = "serde")]
    pub fn snapshot(&self) -> DagSnapshot {
        let mut completed: Vec<_> = self.completed.iter().copied().collect();
        completed.sort_unstable();
        DagSnapshot {
            config: self.config.clone(),
            graph: self.graph.clone(),
            completed,
            completed_floor: self.completed_floor,
            placeholders: self.placeholders.clone(),
        }
    }
}

/// A serialisable snapshot of a [`DagScheduler`]'s declared graph; see
/// [`DagScheduler::snapshot`] and [`DagScheduler::restore`].
#[cfg(feature = "serde")]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DagSnapshot {
    config: DagConfig,
    graph: StableDiGraph<Node, ()>,
    completed: Vec<JobId>,
    completed_floor: JobId,
    placeholders: BTreeMap<u64, (Vec<u64>, f64)>,
}

impl<P: Policy> Dag for DagScheduler<P> {
    /// Validate ids, insert nodes and edges, reject cycles, update ranks, then release ready jobs.
    fn declare(&mut self, jobs: Vec<DagJob>, now: Instant) -> Result<(), DagError> {
        self.now = now;
        // Validate ids before touching anything.
        let mut seen = HashSet::with_capacity(jobs.len());
        for j in &jobs {
            let id = j.spec.id;
            let redeclared = self
                .index
                .get(&id)
                .is_some_and(|&n| self.graph[n].state != State::Undeclared);
            if !seen.insert(id) || redeclared || self.is_completed(id) {
                return Err(DagError::Duplicate(id));
            }
        }
        let mut batch = Vec::with_capacity(jobs.len());
        let mut created = Vec::new();
        for j in &jobs {
            let n = match self.index.get(&j.spec.id) {
                Some(&n) => n,
                None => {
                    let n = self.undeclared(j.spec.id);
                    created.push(n);
                    n
                }
            };
            let node = &mut self.graph[n];
            node.state = State::Pending;
            node.spec = Some(j.spec.clone());
            node.work = j.work_estimate.unwrap_or(self.config.default_work);
            node.passthrough = j.passthrough;
            batch.push(n);
        }
        for (j, &n) in jobs.iter().zip(&batch) {
            let mut deps = j.deps.clone();
            deps.sort_unstable();
            deps.dedup();
            for d in deps {
                if self.is_completed(d) {
                    continue;
                }
                let dn = match self.index.get(&d) {
                    Some(&dn) => dn,
                    None => {
                        let dn = self.undeclared(d);
                        created.push(dn);
                        dn
                    }
                };
                self.graph.add_edge(dn, n, ());
                self.graph[n].unmet += 1;
            }
        }
        if let Some(job) = self.find_cycle(&batch) {
            self.rollback(&batch, &created);
            return Err(DagError::Cycle { job });
        }
        for &n in &batch {
            let below = self
                .graph
                .neighbors_directed(n, Outgoing)
                .map(|c| self.graph[c].rank)
                .fold(0.0, f64::max);
            self.graph[n].rank = self.graph[n].work + below;
            self.propagate_rank(n);
        }
        for &n in &batch {
            if self.graph[n].unmet == 0 {
                self.make_ready(n, now);
            }
        }
        self.drain_passthrough(now);
        Ok(())
    }

    /// Replace the group's placeholder and invalidate the memoised tails.
    fn declare_group_placeholder(&mut self, group: u64, after_groups: Vec<u64>, cost: f64) {
        self.remove_group_placeholder(group);
        for &h in &after_groups {
            self.waiters.entry(h).or_default().push(group);
        }
        self.placeholders.insert(group, (after_groups, cost));
        self.tail_memo.clear();
    }

    /// Release the job's dependents, drop its node and remember it as completed.
    fn completed(&mut self, job: JobId, now: Instant) {
        self.now = now;
        self.policy.completed(job, now);
        self.finish(job, now);
        self.drain_passthrough(now);
    }

    /// Remove the job and its descendants, then any forward references only they needed.
    fn cancel(&mut self, job: JobId) -> Vec<JobId> {
        let Some(&start) = self.index.get(&job) else {
            self.policy.cancel(job);
            return Vec::new();
        };
        let mut doomed = BTreeSet::new();
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            if doomed.insert((self.graph[n].id, n)) {
                stack.extend(self.graph.neighbors_directed(n, Outgoing));
            }
        }
        let mut orphan_candidates = Vec::new();
        let mut cancelled = Vec::with_capacity(doomed.len());
        for &(id, n) in &doomed {
            orphan_candidates.extend(self.graph.neighbors_directed(n, Incoming));
            if self.graph[n].state == State::Submitted {
                self.policy.cancel(id);
            }
            if self.graph[n].state != State::Undeclared {
                cancelled.push(id);
            }
        }
        for &(id, n) in &doomed {
            self.graph.remove_node(n);
            self.index.remove(&id);
        }
        self.newly_ready
            .retain(|j| !doomed.iter().any(|d| d.0 == *j));
        // Forward references kept alive only by the cancelled jobs are no longer needed.
        for n in orphan_candidates {
            let orphan = self
                .graph
                .node_weight(n)
                .is_some_and(|w| w.state == State::Undeclared)
                && self.graph.neighbors_directed(n, Outgoing).next().is_none();
            if orphan && let Some(w) = self.graph.remove_node(n) {
                self.index.remove(&w.id);
            }
        }
        cancelled
    }

    /// Forwarded to the policy.
    fn worker_update(&mut self, w: WorkerState, now: Instant) {
        self.now = now;
        self.policy.worker_update(w, now);
    }

    /// Forwarded to the policy.
    fn worker_gone(&mut self, w: WorkerId, now: Instant) {
        self.now = now;
        self.policy.worker_gone(w, now);
    }

    /// Forwarded to the policy.
    fn dispatch(&mut self, now: Instant) -> Vec<(JobId, WorkerId)> {
        self.now = now;
        self.policy.dispatch(now)
    }

    /// The DAG's reason while the job is not submitted, the policy's afterwards.
    fn explain(&self, job: JobId) -> Option<String> {
        let Some(&n) = self.index.get(&job) else {
            return if self.is_completed(job) {
                Some(format!("job {job} completed"))
            } else {
                self.policy.explain(job)
            };
        };
        let node = &self.graph[n];
        match node.state {
            State::Undeclared => Some(format!(
                "job {job} is not declared yet (named as a dependency of {} job(s))",
                self.graph.neighbors_directed(n, Outgoing).count()
            )),
            State::Pending => {
                let mut deps: Vec<JobId> = self
                    .graph
                    .neighbors_directed(n, Incoming)
                    .map(|d| self.graph[d].id)
                    .collect();
                deps.sort_unstable();
                let shown: Vec<_> = deps.iter().take(8).collect();
                Some(format!(
                    "job {job} waits for {} dependenc{} {shown:?}{}",
                    deps.len(),
                    if deps.len() == 1 { "y" } else { "ies" },
                    if deps.len() > 8 { " ..." } else { "" }
                ))
            }
            State::Held => Some(format!("job {job} is ready and held until release")),
            State::Submitted => self.policy.explain(job),
        }
    }

    /// Forwarded to the policy.
    fn stats(&self) -> PolicyStats {
        self.policy.stats()
    }
}
