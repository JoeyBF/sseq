//! What the DAG layer reports: [`DagStats`] and its part of [`Policy::explain`].

use super::{
    DagScheduler, UnitState,
    frame::{self, COMPLETE, HELD, SUBMITTED},
};
use crate::{JobId, Policy};

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

    /// A unit's state in words.
    pub(super) fn explain_unit(&self, u: u32, what: &str) -> String {
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
    pub(super) fn explain_leaf(&self, u: u32, leaf: u32, job: JobId) -> Option<String> {
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
