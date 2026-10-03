//! Templates: the dependency structure units share.

use std::sync::{Arc, LazyLock};

use super::{DagError, frame::SENTINEL};
use crate::JobId;

/// One node of a [`DagTemplate`].
#[derive(Clone, Debug)]
pub enum TemplateNode {
    /// A job run on a worker, of this much work (before the unit's
    /// [`scale`](super::Unit::scale)).
    Job(f64),
    /// A job run on the caller ([`Output::RunLocal`](crate::Output::RunLocal)), of this much work.
    Local(f64),
    /// A synchronisation point, of this much work: it completes by itself once ready.
    Pass(f64),
    /// A unit of another template substituted for the node: its sources wait for the node's
    /// predecessors, and the node is complete once all of its nodes are. Its leaves take the next
    /// ids of the enclosing unit (see [`DagTemplate::leaf_offset`]).
    Unit(Arc<DagTemplate>),
}

/// A dependency structure over nodes `0..len`, shared by every [`Unit`](super::Unit) built on it
/// (e.g. one signature DAG per subalgebra profile, shared by every bidegree with that profile).
///
/// Building it checks acyclicity once and computes what ranks need: each node's bottom level (its
/// work plus the longest chain of work below it, a substituted unit weighing its template's
/// [`span`](Self::span)) and the template's span. Its *leaves* are its `Job`, `Local` and `Pass`
/// nodes, those of substituted units included, numbered in node order; a unit's leaf `k` is job
/// `base + k`.
#[derive(Clone, Debug)]
pub struct DagTemplate {
    nodes: Vec<TemplateNode>,
    succ: Vec<Vec<u32>>,
    pred: Vec<Vec<u32>>,
    /// A topological order.
    topo: Vec<u32>,
    /// Each node's first leaf, then the number of leaves.
    offset: Vec<u32>,
    /// Each node's bottom level with the nodes' own work.
    bl: Vec<f64>,
    span: f64,
}

/// The one-node templates of plain jobs ([`DagJob`](super::DagJob)), of unit work.
pub(super) static JOB: LazyLock<Arc<DagTemplate>> = LazyLock::new(|| single(TemplateNode::Job));
/// See [`JOB`].
pub(super) static LOCAL: LazyLock<Arc<DagTemplate>> = LazyLock::new(|| single(TemplateNode::Local));
/// See [`JOB`].
pub(super) static PASS: LazyLock<Arc<DagTemplate>> = LazyLock::new(|| single(TemplateNode::Pass));

/// A one-node template of unit work.
fn single(node: fn(f64) -> TemplateNode) -> Arc<DagTemplate> {
    Arc::new(DagTemplate::with_nodes(vec![node(1.0)], []).expect("one node is acyclic"))
}

impl DagTemplate {
    /// A template of `len` worker jobs of unit work, with the given edges `(from, to)`: `to`
    /// depends on `from`. Duplicate edges are merged; out-of-range nodes panic; a cycle is an
    /// error naming a node on it.
    pub fn new(len: usize, edges: impl IntoIterator<Item = (u32, u32)>) -> Result<Self, DagError> {
        Self::with_nodes(vec![TemplateNode::Job(1.0); len], edges)
    }

    /// A template of the given nodes and edges (as for [`new`](Self::new)).
    pub fn with_nodes(
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
            span: 0.0,
        };
        t.bl = t.bottom_levels(|i| t.own_work(i));
        t.span = t.bl.iter().copied().fold(0.0, f64::max);
        Ok(t)
    }

    /// Node `i`'s own work: a leaf's work, or a substituted unit's span.
    fn own_work(&self, i: usize) -> f64 {
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
    pub fn span(&self) -> f64 {
        self.span
    }

    /// Node `i`'s bottom level with the nodes' own work (see [`span`](Self::span)).
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

    /// The same partial order over the same nodes with every implied edge removed (an edge
    /// `a -> c` is implied when `c` is reachable from another successor of `a`). Readiness and
    /// critical paths are unchanged; materialised units count down fewer edges. Takes
    /// `O(len^2 / 8)` bytes of scratch.
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
        DagTemplate::with_nodes(self.nodes.clone(), edges).expect("a sub-DAG of a DAG is acyclic")
    }

    /// Nodes on a longest chain of `work` (CPOP's critical nodes): those whose longest path
    /// from a source plus longest path to a sink equals the critical path, within a relative
    /// tolerance `tol`.
    pub fn critical_nodes(&self, work: impl Fn(usize) -> f64, tol: f64) -> Vec<bool> {
        let n = self.len();
        let w: Vec<f64> = (0..n).map(&work).collect();
        let below = self.bottom_levels(|i| w[i]);
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

    /// The longest chain of `work` through the template.
    pub fn critical_path(&self, work: impl Fn(usize) -> f64) -> f64 {
        self.bottom_levels(work).into_iter().fold(0.0, f64::max)
    }

    /// Each node's bottom level under `work`: its work plus the longest chain of work below it
    /// (its upward rank within the template).
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
