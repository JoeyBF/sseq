//! Upward ranks: each unit's critical path to the end of the graph.

use std::time::Duration;

use super::{DagScheduler, Loc, UnitState};
use crate::{JobId, Policy, time::secs};

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
    /// # use std::{sync::Arc, time::Duration};
    /// # use whelm::{
    /// #     Config, DagConfig, DagJob, DagScheduler, JobSpec, Scheduler, TemplateNode,
    /// #     TemplateSpec, Time, Unit,
    /// # };
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// let pair = TemplateSpec {
    ///     nodes: vec![
    ///         TemplateNode::Job(Duration::from_secs(1)),
    ///         TemplateNode::Job(Duration::from_secs(3)),
    ///     ],
    ///     ..Default::default()
    /// };
    /// let unit = Unit {
    ///     id: 10,
    ///     base: 100,
    ///     template: Arc::new(pair.build().unwrap()),
    ///     ..Default::default()
    /// };
    /// let after = DagJob {
    ///     id: 20,
    ///     deps: vec![10],
    ///     spec: JobSpec {
    ///         work: Some(Duration::from_secs(2)),
    ///         ..Default::default()
    ///     },
    ///     ..Default::default()
    /// };
    /// dag.declare([unit, after.into()], Time::ORIGIN).unwrap();
    /// assert_eq!(
    ///     [10, 100, 101, 20].map(|j| dag.rank(j)),
    ///     [5, 3, 5, 2].map(|s| Some(Duration::from_secs(s)))
    /// );
    /// assert_eq!(dag.rank(99), None);
    /// ```
    pub fn rank(&self, job: JobId) -> Option<Duration> {
        let rank = match self.locate(job)? {
            Loc::Unit(u) => self.unit(u).top(),
            Loc::Leaf { unit, leaf } => self.leaf_rank(unit, leaf),
        };
        Some(secs(rank))
    }

    /// Change a unit's scale (a plain job's work, e.g. once its real size is known) and re-rank
    /// it and its dependencies, up or down.
    ///
    /// Returns false for unknown ids and leaves of other units. Jobs already handed to the policy
    /// keep the rank they were submitted with.
    ///
    /// ```
    /// # use std::time::Duration;
    /// #
    /// # use whelm::{
    /// #     Config, DagConfig, DagJob, DagScheduler, Input, Output, Policy, Resources,
    /// #     SLOTS, Scheduler, Time, WorkerState,
    /// # };
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # let capacity = Resources::new().with(SLOTS, 1);
    /// # let worker = WorkerState { id: 1, capacity, ..Default::default() };
    /// # dag.handle(Input::Worker(worker), Time::ORIGIN);
    /// dag.declare(
    ///     [
    ///         DagJob {
    ///             id: 1,
    ///             ..Default::default()
    ///         },
    ///         DagJob {
    ///             id: 2,
    ///             deps: vec![1],
    ///             ..Default::default()
    ///         },
    ///     ],
    ///     Time::ORIGIN,
    /// )
    /// .unwrap();
    /// assert_eq!(dag.rank(1), Some(Duration::from_secs(2)));
    /// assert!(dag.update_work(2, 5.0));
    /// assert_eq!(dag.rank(1), Some(Duration::from_secs(6)));
    /// assert!(dag.update_work(2, 0.5));
    /// assert_eq!(dag.rank(1), Some(Duration::from_millis(1500)));
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
