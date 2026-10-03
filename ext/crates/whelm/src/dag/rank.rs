//! Upward ranks: each unit's critical path to the end of the graph.

use super::{DagScheduler, Loc, UnitState};
use crate::{JobId, Policy};

impl<P: Policy> DagScheduler<P> {
    /// Raise dependencies' ranks after `start`'s rank grew.
    pub(super) fn propagate_rank(&mut self, start: u32) {
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
}
