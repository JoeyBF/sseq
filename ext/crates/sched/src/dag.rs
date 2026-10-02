//! The optional dependency layer in front of a [`Policy`].

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use petgraph::{
    Direction::{Incoming, Outgoing},
    stable_graph::{NodeIndex, StableDiGraph},
    visit::EdgeRef,
};

use crate::{Attempt, Input, Instant, JobId, JobSpec, Output, Policy, PolicyStats, WorkerId};

mod instance;
pub use instance::{InstanceSpec, NodeLabel};

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

/// A reusable dependency structure over nodes `0..len`, instantiated once per group with
/// [`DagScheduler::declare_template`] (e.g. one signature DAG per subalgebra profile, shared by
/// every bidegree with that profile). Building it checks acyclicity once.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        self.bottom_levels(work).into_iter().fold(0.0, f64::max)
    }

    /// Each node's bottom level: its work plus the longest chain of work below it (its upward
    /// rank within the template).
    pub fn bottom_levels(&self, work: impl Fn(usize) -> f64) -> Vec<f64> {
        let mut below = vec![0.0f64; self.len()];
        for &n in self.topo.iter().rev() {
            let n = n as usize;
            let tail = self.succ[n]
                .iter()
                .map(|&c| below[c as usize])
                .fold(0.0, f64::max);
            below[n] = work(n) + tail;
        }
        below
    }
}

/// Errors from [`DagScheduler::declare`]. A failed declaration changes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DagError {
    /// The declaration would close a dependency cycle through this job.
    Cycle {
        /// A job on the cycle.
        job: JobId,
    },
    /// The job is already declared (or completed), or appears twice in the batch.
    Duplicate(JobId),
    /// No open instance has this job as a node or as its `done` job.
    NotFound(JobId),
}

impl std::fmt::Display for DagError {
    /// A one-line description of the error.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cycle { job } => write!(f, "declaring job {job} would create a dependency cycle"),
            Self::Duplicate(job) => write!(f, "job {job} is already declared"),
            Self::NotFound(job) => write!(f, "no open instance contains job {job}"),
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
    /// are held and announced by [`Output::Ready`]; the caller submits each with
    /// [`DagScheduler::release`] when it is actually sendable (e.g. after coordinator-side
    /// preparation).
    pub auto_submit: bool,
    /// Announce passthrough jobs (and instances' `done` jobs) as they complete, with
    /// [`Output::Passed`]. Default false.
    #[cfg_attr(feature = "serde", serde(default))]
    pub record_passthrough: bool,
    /// At most this many implicit instances may be open (entry completed, nodes releasable) at
    /// once; further ones wait, in the order their entries completed, until one closes. Bounds the
    /// coordinator's frontier state. `None` (the default): unbounded. At least 1 is enforced.
    #[cfg_attr(feature = "serde", serde(default))]
    pub max_open_instances: Option<usize>,
    /// Maintain upward ranks as the graph grows and work estimates change. Ranks are needed by
    /// `rank_priority`, [`DagScheduler::rank`] and placeholders; without them, declaring and
    /// re-estimating skip all rank propagation, which on long dependency chains is most of the
    /// cost. Default true.
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
            rank_priority: false,
            rank_scale: 1000.0,
            default_work: 1.0,
            rank_epsilon: 0.01,
            auto_submit: true,
            record_passthrough: false,
            max_open_instances: None,
            track_ranks: true,
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
    /// All dependencies completed, waiting for [`DagScheduler::release`] (or to run locally, or
    /// to complete as a passthrough). A job the policy gave up on returns here.
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
    /// Rank contributed by implicit instances this job is the entry of: their critical path plus
    /// their `done` job's rank (they have no edges for ranks to flow along).
    #[cfg_attr(feature = "serde", serde(default))]
    implicit_below: f64,
    /// Runs on the caller (see [`DagJob::local`]).
    #[cfg_attr(feature = "serde", serde(default))]
    local: bool,
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
    /// Implicit instances ([`InstanceSpec`]) declared and not finished; their nodes count in
    /// `pending`, `held` and `submitted`.
    pub instances: usize,
    /// Of those, the ones whose entry completed and that are not throttled.
    pub instances_open: usize,
    /// Approximate memory held by instances (counters, work, flags), bytes.
    pub instance_bytes: usize,
}

/// A dependency layer in front of a [`Policy`], itself a [`Policy`].
///
/// Jobs are declared with their dependencies, possibly long before they are ready; a job is
/// submitted to the inner policy when its last dependency completes. Inputs go to the inner
/// policy; an [`Input::Done`] of a live attempt also completes the job here, and its dependents
/// may become ready. [`Policy::poll`] returns the inner policy's outputs and this layer's own:
/// [`Output::RunLocal`], [`Output::Ready`] and [`Output::Passed`].
///
/// A job the inner policy gives up on ([`Output::GaveUp`], passed through) is held again: its
/// dependents stay pending until the caller [`release`](Self::release)s it (another round of
/// attempts) or [`cancel`](Self::cancel)s it. The graph lives in a petgraph
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
    /// This layer's outputs not yet returned by `poll`.
    outbox: Vec<Output>,
    /// Live attempts of the inner policy's jobs, from its outputs and the inputs: an
    /// [`Input::Done`] completes a job here only if its attempt is live.
    live: HashMap<JobId, Vec<(Attempt, WorkerId)>>,
    /// Running jobs of instances closed early: their completion only frees their resources.
    ignored: HashSet<JobId>,
    /// Stops the inner policy emits for attempts the caller has already reported (an ignored
    /// job's failure, turned into a cancellation): not passed on.
    quiet: HashSet<(JobId, Attempt)>,
    placeholders: BTreeMap<u64, (Vec<u64>, f64)>,
    /// group -> placeholder groups that wait for it.
    waiters: HashMap<u64, Vec<u64>>,
    tail_memo: HashMap<u64, f64>,
    /// Ready passthrough nodes waiting for `drain_passthrough`.
    passing: Vec<NodeIndex>,
    /// Implicit template instances (see [`InstanceSpec`]); `None` for free slots.
    instances: Vec<Option<instance::Instance>>,
    free_instances: Vec<usize>,
    /// Instance base id -> slot.
    by_base: BTreeMap<JobId, usize>,
    /// Instances whose entry completed while `max_open_instances` was reached, oldest first.
    waiting_to_open: std::collections::VecDeque<usize>,
    /// Entry job -> instances whose sources wait for it.
    by_entry: HashMap<JobId, Vec<usize>>,
    /// An instance's `done` job -> (its entry, its critical path), for rank propagation.
    by_done: HashMap<JobId, Vec<(JobId, f64)>>,
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
            outbox: Vec::new(),
            live: HashMap::new(),
            ignored: HashSet::new(),
            quiet: HashSet::new(),
            placeholders: BTreeMap::new(),
            waiters: HashMap::new(),
            tail_memo: HashMap::new(),
            passing: Vec::new(),
            instances: Vec::new(),
            free_instances: Vec::new(),
            by_base: BTreeMap::new(),
            waiting_to_open: std::collections::VecDeque::new(),
            by_entry: HashMap::new(),
            by_done: HashMap::new(),
            now: 0.0,
        }
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

    /// Submit a ready, held job to the policy (only meaningful without `auto_submit`). Returns
    /// false if the job is not held.
    pub fn release(&mut self, job: JobId, now: Instant) -> bool {
        self.now = now;
        if self.instance_of(job).is_some() {
            return self.instance_release(job, now);
        }
        match self.index.get(&job) {
            Some(&n) if self.graph[n].state == State::Held && !self.graph[n].local => {
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
        let (p, h, sub) = self.instance_counts();
        s.pending += p;
        s.held += h;
        s.submitted += sub;
        s.instances = self.instances.iter().flatten().count();
        s.instances_open = self.instances.iter().flatten().filter(|i| i.open).count();
        s.instance_bytes = self.instance_bytes();
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
            implicit_below: 0.0,
            local: false,
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
            // Across implicit instances: `n` may be the `done` job of instances whose entries
            // must see its rank through the instance's critical path.
            let links = self
                .by_done
                .get(&self.graph[n].id)
                .cloned()
                .unwrap_or_default();
            for (entry, cp) in links {
                let Some(&e) = self.index.get(&entry) else {
                    continue;
                };
                let below = cp + r;
                if below > self.graph[e].implicit_below {
                    self.graph[e].implicit_below = below;
                    let cand = self.graph[e].work + below;
                    let old = self.graph[e].rank;
                    if cand > old * (1.0 + eps) && cand > old {
                        self.graph[e].rank = cand;
                        stack.push(e);
                    }
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
        self.policy.handle(Input::Submit(spec), now);
    }

    /// Record that a job has no unmet dependencies left, and submit or hold it.
    fn make_ready(&mut self, n: NodeIndex, now: Instant) {
        if self.graph[n].passthrough {
            self.graph[n].state = State::Held;
            self.passing.push(n);
            return;
        }
        let job = self.graph[n].id;
        if self.graph[n].local {
            self.graph[n].state = State::Held;
            self.outbox.push(Output::RunLocal { job });
            return;
        }
        if self.config.auto_submit {
            self.submit_node(n, now);
        } else {
            self.graph[n].state = State::Held;
            self.outbox.push(Output::Ready { job });
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
            node.local = false;
        }
        for &n in created {
            if let Some(node) = self.graph.remove_node(n) {
                self.index.remove(&node.id);
            }
        }
    }

    /// `cancel` for an explicit job (and its explicit descendants).
    fn cancel_explicit(&mut self, job: JobId) -> Vec<JobId> {
        let Some(&start) = self.index.get(&job) else {
            self.policy.handle(Input::Cancel(job), self.now);
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
                self.policy.handle(Input::Cancel(id), self.now);
            }
            if self.graph[n].state != State::Undeclared {
                cancelled.push(id);
            }
        }
        for &(id, n) in &doomed {
            self.graph.remove_node(n);
            self.index.remove(&id);
        }
        self.unannounce(|j| doomed.iter().any(|d| d.0 == j));
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

    /// Release `job`'s dependents, drop its node and remember it as completed.
    fn finish(&mut self, job: JobId, now: Instant) {
        if let Some((slot, i)) = self.instance_of(job) {
            self.finish_instance_node(slot, i, now);
            return;
        }
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
        if let Some(slots) = self.by_entry.remove(&job) {
            for slot in slots {
                if self.instances[slot].is_some() {
                    self.open_sources(slot, now);
                }
            }
        }
    }

    /// Complete every ready passthrough job, and the ones that become ready as a result (a
    /// worklist, so long chains of them do not recurse).
    fn drain_passthrough(&mut self, now: Instant) {
        while let Some(n) = self.passing.pop() {
            let job = self.graph[n].id;
            if self.config.record_passthrough {
                self.outbox.push(Output::Passed { job });
            }
            self.finish(job, now);
        }
    }

    /// Change a declared job's work estimate (e.g. once its real size is known) and re-rank it
    /// and its ancestors, up or down. Returns false for unknown or completed jobs. Jobs already
    /// handed to the policy keep the priority they were submitted with.
    pub fn update_work(&mut self, job: JobId, work: f64) -> bool {
        let Some(&n) = self.index.get(&job) else {
            return false;
        };
        self.graph[n].work = work;
        if !self.config.track_ranks {
            return true;
        }
        let eps = self.config.rank_epsilon.max(0.0);
        let mut stack = vec![n];
        let mut first = true;
        while let Some(m) = stack.pop() {
            let below = self
                .graph
                .neighbors_directed(m, Outgoing)
                .map(|c| self.graph[c].rank)
                .fold(0.0, f64::max);
            let new = self.graph[m].work + below.max(self.graph[m].implicit_below);
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
    /// are announced again ([`Output::Ready`], [`Output::RunLocal`]). Open implicit instances are restored
    /// with their templates (shared again among instances that shared them).
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
        let mut local = Vec::new();
        for n in s.graph.node_indices() {
            let node = &s.graph[n];
            s.index.insert(node.id, n);
            match node.state {
                State::Submitted => submitted.push((node.id, n)),
                State::Held if node.local => local.push(node.id),
                State::Held => held.push(node.id),
                _ => {}
            }
        }
        local.sort_unstable();
        s.outbox
            .extend(local.into_iter().map(|job| Output::RunLocal { job }));
        submitted.sort_unstable();
        for (_, n) in submitted {
            s.submit_node(n, now);
        }
        held.sort_unstable();
        s.outbox
            .extend(held.into_iter().map(|job| Output::Ready { job }));
        s.restore_instances(snapshot.templates, snapshot.instances, now);
        s.now = now;
        s
    }

    /// A serialisable snapshot of the declared DAG (not of the policy): pending, held and
    /// submitted jobs with their edges, remembered completions and placeholders.
    #[cfg(feature = "serde")]
    pub fn snapshot(&self) -> DagSnapshot {
        let mut completed: Vec<_> = self.completed.iter().copied().collect();
        completed.sort_unstable();
        let (templates, instances) = self.snapshot_instances();
        DagSnapshot {
            config: self.config.clone(),
            graph: self.graph.clone(),
            completed,
            completed_floor: self.completed_floor,
            placeholders: self.placeholders.clone(),
            templates,
            instances,
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
    /// Templates of open instances, each once.
    #[serde(default)]
    templates: Vec<DagTemplate>,
    /// Open instances.
    #[serde(default)]
    instances: Vec<instance::InstanceSnapshot>,
}

impl<P: Policy> DagScheduler<P> {
    /// [`cancel`](Self::cancel), before the cancelled jobs' attempts are forgotten.
    fn cancel_inner(&mut self, job: JobId) -> Vec<JobId> {
        if let Some((slot, _)) = self.instance_of(job) {
            let done = self.inst(slot).done;
            let mut ids = self.cancel_instance(slot);
            ids.extend(self.cancel_inner(done));
            return ids;
        }
        let mut ids = self.cancel_explicit(job);
        // Instances waiting on a cancelled job can never open; their `done` dependents go too.
        let mut k = 0;
        while k < ids.len() {
            let entry = ids[k];
            for slot in self.by_entry.remove(&entry).unwrap_or_default() {
                if self.instances[slot].is_some() {
                    let done = self.inst(slot).done;
                    ids.extend(self.cancel_instance(slot));
                    ids.extend(self.cancel_explicit(done));
                }
            }
            k += 1;
        }
        ids
    }

    /// Drop announcements not yet polled ([`Output::Ready`], [`Output::RunLocal`]) of the jobs
    /// for which `gone` holds.
    fn unannounce(&mut self, gone: impl Fn(JobId) -> bool) {
        self.outbox.retain(
            |o| !matches!(*o, Output::Ready { job } | Output::RunLocal { job } if gone(job)),
        );
    }

    /// Declare jobs. The graph grows during the run; dependencies may be forward references.
    /// Rejects (and leaves no trace of) a batch that would create a cycle or redeclare a job.
    /// Jobs whose dependencies are all complete become ready immediately.
    pub fn declare(&mut self, jobs: Vec<DagJob>, now: Instant) -> Result<(), DagError> {
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
            node.local = j.local;
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
            self.graph[n].rank = self.graph[n].work + below.max(self.graph[n].implicit_below);
            if self.config.track_ranks {
                self.propagate_rank(n);
            }
        }
        for &n in &batch {
            if self.graph[n].unmet == 0 {
                self.make_ready(n, now);
            }
        }
        self.drain_passthrough(now);
        Ok(())
    }

    /// Declare work known to come but not yet expandable: group `group` will exist after the
    /// groups `after_groups` complete and costs about `cost`. Only ranks use this. Redeclaring a
    /// group replaces its placeholder.
    pub fn declare_group_placeholder(&mut self, group: u64, after_groups: Vec<u64>, cost: f64) {
        self.remove_group_placeholder(group);
        for &h in &after_groups {
            self.waiters.entry(h).or_default().push(group);
        }
        self.placeholders.insert(group, (after_groups, cost));
        self.tail_memo.clear();
    }

    /// Cancel a job and, transitively, every job depending on it (they can never run). Returns
    /// the cancelled ids. Each one the inner policy has is cancelled there (its live attempts are
    /// stopped). [`Input::Cancel`] does the same.
    pub fn cancel(&mut self, job: JobId) -> Vec<JobId> {
        let ids = self.cancel_inner(job);
        for id in &ids {
            self.live.remove(id);
            self.ignored.remove(id);
        }
        ids
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
            let local = self.index.get(&job).is_some_and(|&n| {
                let node = &self.graph[n];
                node.local && node.state == State::Held
            });
            if local {
                self.finish(job, now);
                self.drain_passthrough(now);
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
        self.finish(job, now);
        self.drain_passthrough(now);
    }

    /// An attempt failed. A job of a closed instance is not retried: its last failure cancels it.
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

    /// A worker left: its attempts are no longer live. Jobs of closed instances that ran only
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
        if let Some((slot, i)) = self.instance_of(job) {
            let inst = self.instances[slot].as_mut().unwrap();
            if inst.counter[i] == instance::SUBMITTED {
                inst.counter[i] = instance::HELD;
            }
        } else if let Some(&n) = self.index.get(&job)
            && self.graph[n].state == State::Submitted
        {
            self.graph[n].state = State::Held;
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

    /// This layer's announcements, then the inner policy's outputs.
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

    /// The DAG's reason while the job is not submitted, the inner policy's afterwards.
    fn explain(&self, job: JobId) -> Option<String> {
        if let Some((slot, i)) = self.instance_of(job) {
            return self.explain_instance_node(slot, i, job);
        }
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

    /// Forwarded to the inner policy.
    fn stats(&self) -> PolicyStats {
        self.policy.stats()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

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

    /// Passthrough jobs and instances' `done` jobs are announced when recorded.
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
        let spec = InstanceSpec {
            template: Arc::new(DagTemplate::new(1, []).unwrap()),
            base: 100,
            entry: 2,
            done: 200,
            proto: JobSpec::new(0, Resources::mem(1), 0),
            work: vec![1.0],
            passthrough: vec![false],
            demand: None,
            label: None,
            completed: Vec::new(),
        };
        d.open_instance(spec, 2.0).unwrap();
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

    /// A two-node instance after job 1, and job 2 after it; job 1 done, node 100 running, and
    /// the instance closed early.
    fn closed_instance() -> DagScheduler<Scheduler> {
        let mut d = dag(DagConfig::default(), 4);
        let spec = InstanceSpec {
            template: Arc::new(DagTemplate::new(2, []).unwrap()),
            base: 100,
            entry: 1,
            done: 200,
            proto: JobSpec::new(0, Resources::mem(1), 0),
            work: vec![1.0; 2],
            passthrough: vec![false; 2],
            demand: None,
            label: None,
            completed: Vec::new(),
        };
        d.declare(vec![job(1, &[]), job(2, &[200])], 0.0).unwrap();
        d.open_instance(spec, 0.0).unwrap();
        assert_eq!(d.poll(0.0), vec![start(1, 1)]);
        assert_eq!(feed(&mut d, 1.0, [done(1, 1)]), vec![start(100, 1)]);
        assert_eq!(d.close_instance(200, 2.0), Ok(vec![100]));
        // Node 101 was withdrawn; `done` completed, releasing job 2 behind the running node.
        let st = d.stats();
        assert_eq!((st.waiting, st.running), (1, 1));
        assert!(d.poll(2.0).is_empty());
        d
    }

    /// A running node of an instance closed early keeps its resources until its attempt ends; a
    /// failure then is neither retried nor reported.
    #[test]
    fn closed_instance_node_fails() {
        let mut d = closed_instance();
        assert_eq!(feed(&mut d, 3.0, [fail(100, 1)]), vec![start(2, 1)]);
        assert_eq!(d.stats().running, 1);
        assert_eq!(d.explain(100), None);
    }

    /// The same when the node's worker leaves.
    #[test]
    fn closed_instance_node_worker_gone() {
        let mut d = closed_instance();
        assert!(feed(&mut d, 3.0, [Input::WorkerGone(1)]).is_empty());
        let st = d.stats();
        assert_eq!((st.waiting, st.running), (1, 0));
        let w = WorkerState::new(1, "x", 1, Resources::mem(100));
        assert_eq!(feed(&mut d, 4.0, [Input::Worker(w)]), vec![start(2, 1)]);
    }

    /// Its completion only frees its resources.
    #[test]
    fn closed_instance_node_done() {
        let mut d = closed_instance();
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
        let mut r = DagScheduler::restore(snap, Scheduler::new(Config::fifo()), 5.0);
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
}
