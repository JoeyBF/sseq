//! Templates: the dependency structure units share.
//!
//! A [`DagTemplate`] is built from a [`TemplateSpec`] once per shape of work, and shared by every
//! [`Unit`](super::Unit) of that shape. Building it checks acyclicity and precomputes the bottom
//! levels ranks need, so declaring a unit of it costs nothing per node. Its nodes are
//! [`TemplateNode`]s; a node may itself be a unit of another template, which gives the hierarchy
//! its depth: the leaves of a substituted unit are numbered among the enclosing template's, in
//! node order.
//!
//! The analysis methods ([`bottom_levels`](DagTemplate::bottom_levels),
//! [`critical_path`](DagTemplate::critical_path), [`critical_nodes`](DagTemplate::critical_nodes))
//! take the work per node as a function, so one template answers for units of any size or per-leaf
//! cost.
//!
//! A three-stage pipeline whose middle stage is a nested fan-out of two jobs:
//!
//! ```
//! use std::{sync::Arc, time::Duration};
//!
//! use whelm::dag::{TemplateNode, TemplateSpec};
//!
//! let fan = Arc::new(TemplateSpec::jobs(2).build().unwrap());
//! let pipeline = TemplateSpec {
//!     nodes: vec![
//!         TemplateNode::Local(Duration::from_secs(1)),
//!         TemplateNode::Unit(fan),
//!         TemplateNode::Pass(Duration::ZERO),
//!     ],
//!     edges: vec![(0, 1), (1, 2)],
//! }
//! .build()
//! .unwrap();
//! // Leaves: the load is leaf 0, the fan-out's jobs leaves 1 and 2, the barrier leaf 3.
//! assert_eq!((pipeline.len(), pipeline.leaves()), (3, 4));
//! assert_eq!([0, 1, 2].map(|i| pipeline.leaf_offset(i)), [0, 1, 3]);
//! // The fan-out weighs its own span, 1: the pipeline's critical path is 1 + 1 + 0.
//! assert_eq!(pipeline.span(), Duration::from_secs(2));
//! ```

use std::{
    sync::{Arc, LazyLock},
    time::Duration,
};

use super::{DagError, frame::SENTINEL};
use crate::{job::JobId, time::secs};

/// One node of a [`DagTemplate`].
///
/// A leaf's [`Duration`] is its work before the unit's scale; a [`Unit`](Self::Unit) node weighs
/// its template's [`span`](DagTemplate::span) instead.
#[derive(Clone, Debug)]
pub enum TemplateNode {
    /// A job run on a worker, of this much work (before the unit's
    /// [`scale`](super::Unit::scale)).
    Job(Duration),
    /// A job run on the caller ([`Output::RunLocal`](crate::policy::Output::RunLocal)), of this
    /// much work.
    Local(Duration),
    /// A synchronisation point, of this much work: it completes by itself once ready.
    Pass(Duration),
    /// A unit of another template substituted for the node: its sources wait for the node's
    /// predecessors, and the node is complete once all of its nodes are. Its leaves take the next
    /// ids of the enclosing unit (see [`DagTemplate::leaf_offset`]).
    Unit(Arc<DagTemplate>),
}

/// A dependency structure over nodes `0..len`, shared by every [`Unit`](super::Unit) built on it.
///
/// For example, one signature DAG per subalgebra profile, shared by every bidegree with that
/// profile. [`TemplateSpec::build`] builds one.
///
/// Building it checks acyclicity once and computes what ranks need: each node's bottom level (its
/// work plus the longest chain of work below it, a substituted unit weighing its template's
/// [`span`](Self::span)) and the template's span. Its *leaves* are its `Job`, `Local` and `Pass`
/// nodes, those of substituted units included, numbered in node order; a unit's leaf `k` is job
/// `base + k`.
///
/// A diamond: node 0 before nodes 1 and 2, both before node 3.
///
/// ```
/// use std::time::Duration;
///
/// use whelm::dag::TemplateSpec;
///
/// let diamond = TemplateSpec {
///     edges: vec![(0, 1), (0, 2), (1, 3), (2, 3)],
///     ..TemplateSpec::jobs(4)
/// }
/// .build()
/// .unwrap();
/// assert_eq!(diamond.sources().collect::<Vec<_>>(), [0]);
/// assert_eq!(diamond.sinks().collect::<Vec<_>>(), [3]);
/// assert_eq!(
///     (diamond.successors(0), diamond.predecessors(3)),
///     (&[1, 2][..], &[1, 2][..])
/// );
/// assert_eq!(diamond.topological_order(), [0, 1, 2, 3]);
/// assert_eq!(
///     (diamond.edge_count(), diamond.span()),
///     (4, Duration::from_secs(3))
/// );
/// ```
#[derive(Clone, Debug)]
pub struct DagTemplate {
    nodes: Vec<TemplateNode>,
    succ: Vec<Vec<u32>>,
    pred: Vec<Vec<u32>>,
    /// A topological order.
    topo: Vec<u32>,
    /// Each node's first leaf, then the number of leaves.
    offset: Vec<u32>,
    /// Each node's bottom level with the nodes' own work, in seconds (rank arithmetic is in
    /// seconds).
    bl: Vec<f64>,
    span: Duration,
}

/// The one-node templates of plain jobs ([`DagJob`](super::DagJob)), of one second of work, so
/// that a unit's scale is the job's work in seconds.
pub(super) static JOB: LazyLock<Arc<DagTemplate>> = LazyLock::new(|| single(TemplateNode::Job));
/// See [`JOB`].
pub(super) static LOCAL: LazyLock<Arc<DagTemplate>> = LazyLock::new(|| single(TemplateNode::Local));
/// See [`JOB`].
pub(super) static PASS: LazyLock<Arc<DagTemplate>> = LazyLock::new(|| single(TemplateNode::Pass));

/// A one-node template of one second of work.
fn single(node: fn(Duration) -> TemplateNode) -> Arc<DagTemplate> {
    let node = node(Duration::from_secs(1));
    Arc::new(DagTemplate::build(vec![node], []).expect("one node is acyclic"))
}

/// What a [`DagTemplate`] is built from: its nodes, and its edges `(from, to)`, `to` depending on
/// `from`.
///
/// [`build`](Self::build) checks the structure and analyses it. [`jobs`](Self::jobs) gives the
/// commonest nodes, worker jobs of unit work, to which a literal adds the edges.
///
/// # Examples
///
/// Nodes of each kind, one of them a nested template: the two-job chain substituted for node 1
/// counts as its span, 2, in the outer template's critical path.
///
/// ```
/// use std::{sync::Arc, time::Duration};
///
/// use whelm::dag::{TemplateNode, TemplateSpec};
///
/// let chain = TemplateSpec {
///     edges: vec![(0, 1)],
///     ..TemplateSpec::jobs(2)
/// };
/// let t = TemplateSpec {
///     nodes: vec![
///         TemplateNode::Local(Duration::from_millis(500)),
///         TemplateNode::Unit(Arc::new(chain.build().unwrap())),
///         TemplateNode::Job(Duration::from_secs(4)),
///         TemplateNode::Pass(Duration::ZERO),
///     ],
///     edges: vec![(0, 1), (0, 2), (1, 3), (2, 3)],
/// }
/// .build()
/// .unwrap();
/// assert_eq!(t.leaves(), 5);
/// assert!(matches!(t.node(1), TemplateNode::Unit(sub) if sub.leaves() == 2));
/// assert_eq!(t.span(), Duration::from_millis(4500));
/// ```
#[derive(Clone, Debug, Default)]
pub struct TemplateSpec {
    /// The nodes, node `i` at index `i`.
    pub nodes: Vec<TemplateNode>,
    /// The edges `(from, to)`: node `to` depends on node `from`. Duplicates are merged.
    pub edges: Vec<(u32, u32)>,
}

impl TemplateSpec {
    /// `n` worker jobs of one second of work each and no edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::dag::TemplateSpec;
    ///
    /// let three = TemplateSpec::jobs(3).build().unwrap();
    /// assert_eq!(
    ///     (three.len(), three.edge_count(), three.span()),
    ///     (3, 0, Duration::from_secs(1))
    /// );
    /// ```
    pub fn jobs(n: usize) -> Self {
        Self {
            nodes: vec![TemplateNode::Job(Duration::from_secs(1)); n],
            edges: Vec::new(),
        }
    }

    /// The template of these nodes and edges.
    ///
    /// A cycle is an error naming a node on it; an edge naming a node out of range panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::dag::{DagError, TemplateSpec};
    ///
    /// let chain = TemplateSpec {
    ///     edges: vec![(0, 1), (1, 2), (0, 1)],
    ///     ..TemplateSpec::jobs(3)
    /// };
    /// let chain = chain.build().unwrap();
    /// assert_eq!(
    ///     (chain.len(), chain.edge_count(), chain.span()),
    ///     (3, 2, Duration::from_secs(3))
    /// );
    /// let cycle = TemplateSpec {
    ///     edges: vec![(0, 1), (1, 2), (2, 1)],
    ///     ..TemplateSpec::jobs(3)
    /// };
    /// assert_eq!(cycle.build().unwrap_err(), DagError::Cycle { job: 1 });
    /// ```
    pub fn build(self) -> Result<DagTemplate, DagError> {
        DagTemplate::build(self.nodes, self.edges)
    }
}

impl DagTemplate {
    /// The template of `nodes` and `edges` (see [`TemplateSpec::build`]).
    fn build(
        nodes: Vec<TemplateNode>,
        edges: impl IntoIterator<Item = (u32, u32)>,
    ) -> Result<Self, DagError> {
        let len = nodes.len();
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
        assert!(
            pred.iter().all(|p| p.len() < SENTINEL as usize),
            "template in-degree too large"
        );
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
        let mut offset = Vec::with_capacity(len + 1);
        let mut leaves = 0u32;
        for n in &nodes {
            offset.push(leaves);
            let k = match n {
                TemplateNode::Unit(t) => t.leaves() as u32,
                _ => 1,
            };
            leaves = leaves.checked_add(k).expect("too many leaves");
        }
        offset.push(leaves);
        let mut t = Self {
            nodes,
            succ,
            pred,
            topo,
            offset,
            bl: Vec::new(),
            span: Duration::ZERO,
        };
        t.span = t.critical_path(|i| t.own_work(i));
        t.bl = t.levels(|i| t.own_work(i).as_secs_f64());
        Ok(t)
    }

    /// Node `i`'s own work: a leaf's work, or a substituted unit's span.
    fn own_work(&self, i: usize) -> Duration {
        match &self.nodes[i] {
            TemplateNode::Job(w) | TemplateNode::Local(w) | TemplateNode::Pass(w) => *w,
            TemplateNode::Unit(t) => t.span,
        }
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the template has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Node `i`.
    pub fn node(&self, i: usize) -> &TemplateNode {
        &self.nodes[i]
    }

    /// Number of leaves, those of substituted units included: a unit's ids span
    /// `base..base + leaves()`.
    pub fn leaves(&self) -> usize {
        self.offset[self.len()] as usize
    }

    /// Node `i`'s first leaf: a leaf node is leaf `leaf_offset(i)`, a substituted unit's leaves
    /// follow from there.
    pub fn leaf_offset(&self, i: usize) -> usize {
        self.offset[i] as usize
    }

    /// The node holding leaf `leaf` (which must be below [`leaves`](Self::leaves)).
    pub(super) fn node_of(&self, leaf: u32) -> usize {
        self.offset[..self.len()].partition_point(|&o| o <= leaf) - 1
    }

    /// The longest chain of the nodes' own work, a substituted unit weighing its template's span:
    /// a unit's duration with unlimited workers, at scale 1.
    pub fn span(&self) -> Duration {
        self.span
    }

    /// Node `i`'s bottom level with the nodes' own work (see [`span`](Self::span)), seconds.
    pub(super) fn own_bottom_level(&self, i: usize) -> f64 {
        self.bl[i]
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

    /// Nodes with no dependency inside the template (they wait for the unit's dependencies).
    pub fn sources(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).filter(|&i| self.pred[i].is_empty())
    }

    /// Nodes nothing in the template depends on.
    pub fn sinks(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).filter(|&i| self.succ[i].is_empty())
    }

    /// The same partial order over the same nodes with every implied edge removed.
    ///
    /// An edge `a -> c` is implied when `c` is reachable from another successor of `a`. Readiness
    /// and critical paths are unchanged; materialised units count down fewer edges. Takes
    /// `O(len^2 / 8)` bytes of scratch.
    ///
    /// ```
    /// use whelm::dag::TemplateSpec;
    ///
    /// // 0 -> 2 is implied by 0 -> 1 -> 2.
    /// let t = TemplateSpec {
    ///     edges: vec![(0, 1), (1, 2), (0, 2)],
    ///     ..TemplateSpec::jobs(3)
    /// }
    /// .build()
    /// .unwrap();
    /// let r = t.transitive_reduction();
    /// assert_eq!((t.edge_count(), r.edge_count()), (3, 2));
    /// assert_eq!(r.successors(0), [1]);
    /// assert_eq!(r.span(), t.span());
    /// ```
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
        DagTemplate::build(self.nodes.clone(), edges).expect("a sub-DAG of a DAG is acyclic")
    }

    /// Nodes on a longest chain of `work` (CPOP's critical nodes).
    ///
    /// These are the nodes whose longest path from a source plus longest path to a sink equals
    /// the critical path, within a relative tolerance `tol`.
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::dag::TemplateSpec;
    ///
    /// // Node 0 before nodes 1 and 2; node 2 is the heavier branch.
    /// let fork = TemplateSpec {
    ///     edges: vec![(0, 1), (0, 2)],
    ///     ..TemplateSpec::jobs(3)
    /// }
    /// .build()
    /// .unwrap();
    /// let work = [1, 1, 5].map(Duration::from_secs);
    /// assert_eq!(fork.critical_nodes(|i| work[i], 0.0), [true, false, true]);
    /// ```
    pub fn critical_nodes(&self, work: impl Fn(usize) -> Duration, tol: f64) -> Vec<bool> {
        let n = self.len();
        let w: Vec<Duration> = (0..n).map(&work).collect();
        let below = self.bottom_levels(|i| w[i]);
        let mut above = vec![Duration::ZERO; n]; // longest path ending just before v
        for &v in &self.topo {
            let v = v as usize;
            above[v] = (self.pred[v].iter())
                .map(|&p| above[p as usize].saturating_add(w[p as usize]))
                .max()
                .unwrap_or_default();
        }
        let cp = below.iter().copied().max().unwrap_or_default();
        let bar = secs(cp.as_secs_f64() * (1.0 - tol));
        (0..n)
            .map(|v| !cp.is_zero() && above[v] + below[v] >= bar)
            .collect()
    }

    /// The longest chain of `work` through the template.
    ///
    /// With the template's own work this is its [`span`](Self::span); any other `work` (per-leaf
    /// costs of one unit, say) gives that unit's critical path.
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::dag::TemplateSpec;
    ///
    /// let fork = TemplateSpec {
    ///     edges: vec![(0, 1), (0, 2)],
    ///     ..TemplateSpec::jobs(3)
    /// }
    /// .build()
    /// .unwrap();
    /// assert_eq!(fork.critical_path(|_| Duration::from_secs(1)), fork.span());
    /// let work = [1, 1, 5].map(Duration::from_secs);
    /// assert_eq!(fork.critical_path(|i| work[i]), Duration::from_secs(6));
    /// ```
    pub fn critical_path(&self, work: impl Fn(usize) -> Duration) -> Duration {
        (self.bottom_levels(work).into_iter().max()).unwrap_or_default()
    }

    /// Each node's bottom level under `work`: its work plus the longest chain of work below it.
    ///
    /// This is its upward rank within the template; the DAG layer adds the scale and the rank
    /// below the unit.
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::dag::TemplateSpec;
    ///
    /// let fork = TemplateSpec {
    ///     edges: vec![(0, 1), (0, 2)],
    ///     ..TemplateSpec::jobs(3)
    /// }
    /// .build()
    /// .unwrap();
    /// let work = [1, 1, 5].map(Duration::from_secs);
    /// assert_eq!(
    ///     fork.bottom_levels(|i| work[i]),
    ///     [6, 1, 5].map(Duration::from_secs)
    /// );
    /// ```
    pub fn bottom_levels(&self, work: impl Fn(usize) -> Duration) -> Vec<Duration> {
        let mut below = vec![Duration::ZERO; self.len()];
        for &n in self.topo.iter().rev() {
            let n = n as usize;
            let tail = (self.succ[n].iter().map(|&c| below[c as usize]))
                .max()
                .unwrap_or_default();
            below[n] = work(n).saturating_add(tail);
        }
        below
    }

    /// [`bottom_levels`](Self::bottom_levels) in seconds, for the DAG layer's rank arithmetic.
    pub(super) fn levels(&self, work: impl Fn(usize) -> f64) -> Vec<f64> {
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
