//! What the DAG layer reports: [`DagStats`] and its part of [`Policy::explain`].

use super::{
    DagScheduler, UnitState,
    frame::{self, COMPLETE, HELD, SUBMITTED},
};
use crate::{
    explain::{Explanation, Status},
    job::JobId,
    policy::Policy,
};

/// Counters describing the DAG layer's state, from [`DagScheduler::dag_stats`].
///
/// Job 2 is declared after job 1 before job 1 is: job 1 is an undeclared forward reference, and
/// job 2 a pending unit with no materialised state. Declaring job 1 enters and submits it.
///
/// ```
/// use whelm::{
///     dag::{DagConfig, DagJob, DagScheduler, DagStats},
///     prelude::*,
/// };
///
/// let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
/// dag.declare(
///     [DagJob {
///         id: 2,
///         deps: vec![1],
///         ..Default::default()
///     }],
///     Time::ORIGIN,
/// )
/// .unwrap();
/// let s = dag.dag_stats();
/// assert_eq!(
///     (s.units, s.undeclared, s.edges, s.pending, s.frames),
///     (1, 1, 1, 1, 0)
/// );
///
/// dag.declare(
///     [DagJob {
///         id: 1,
///         ..Default::default()
///     }],
///     Time::ORIGIN,
/// )
/// .unwrap();
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

impl<P: Policy> DagScheduler<P> {
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

    /// The status of unit `u` itself.
    pub(super) fn unit_status(&self, u: u32) -> Status {
        let rec = self.unit(u);
        match rec.state {
            UnitState::Undeclared => Status::Undeclared {
                dependents: rec.succs.len(),
            },
            UnitState::Pending => {
                let mut unmet: Vec<JobId> = rec.preds.iter().map(|&d| self.unit(d).id).collect();
                unmet.sort_unstable();
                Status::Pending {
                    unit: rec.id,
                    closed: rec.closed,
                    unmet,
                }
            }
            UnitState::Open => Status::Open,
        }
    }

    /// `explain` for leaf `leaf` of unit `u`.
    pub(super) fn explain_leaf(&self, u: u32, leaf: u32, job: JobId) -> Option<Explanation> {
        let rec = self.unit(u);
        let pending = rec.state == UnitState::Pending;
        let status = if pending && (rec.plain() || !rec.leaf_completed(leaf)) {
            self.unit_status(u)
        } else if rec.leaf_completed(leaf) {
            Status::Completed
        } else {
            match self.leaf_node(job) {
                None => Status::Unentered { unit: rec.id },
                Some((f, i)) => match self.frame(f).counter[i] {
                    SUBMITTED => return self.labelled(u, leaf, self.policy.explain(job)?),
                    HELD => Status::Held,
                    COMPLETE => Status::Completed,
                    k => Status::PendingWithin { unmet: k.into() },
                },
            }
        };
        self.labelled(u, leaf, Explanation::new(job, status))
    }

    /// `e` with the label of leaf `leaf` of unit `u`, if its source gives one.
    fn labelled(&self, u: u32, leaf: u32, e: Explanation) -> Option<Explanation> {
        let rec = self.unit(u);
        let label = if rec.sourced {
            self.src().label(rec.id, leaf)
        } else {
            None
        };
        Some(Explanation {
            label: label.or(e.label),
            ..e
        })
    }
}
