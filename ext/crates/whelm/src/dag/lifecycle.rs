//! Progress of declared work: release, completion, failure, cancellation and closing.

use std::collections::{BTreeSet, HashSet};

use super::{
    DagError, DagScheduler, TemplateNode, UnitState,
    frame::{HELD, SUBMITTED},
};
#[cfg(doc)]
use crate::DagConfig;
use crate::{Attempt, Input, JobId, Output, Policy, Time, WorkerId};

impl<P: Policy> DagScheduler<P> {
    /// Submit a ready, held job to the policy.
    ///
    /// Jobs are held when ready without [`DagConfig::auto_submit`], and after the policy gives up
    /// on them (see [`DagScheduler`]). Returns false if the job is not held, so a second release
    /// does nothing.
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use whelm::{
    /// #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Scheduler,
    /// #     Time, WorkerState,
    /// # };
    /// # let config = DagConfig { auto_submit: false, ..DagConfig::default() };
    /// # let mut dag = DagScheduler::new(config, Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
    /// # let job = |id, deps| DagJob {
    /// #     spec: JobSpec { id, ..Default::default() },
    /// #     deps,
    /// #     ..Default::default()
    /// # };
    /// dag.declare([job(1, vec![]), job(2, vec![1])], Time::ORIGIN)
    ///     .unwrap();
    /// assert_eq!(dag.poll(Time::ORIGIN), vec![Output::Ready { job: 1 }]);
    /// assert!(!dag.release(2, Time::ORIGIN));
    /// assert!(dag.release(1, Time::ORIGIN));
    /// assert!(!dag.release(1, Time::ORIGIN));
    /// assert_eq!(
    ///     dag.poll(Time::ORIGIN),
    ///     vec![Output::Start {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// dag.handle(
    ///     Input::Done { job: 1, attempt: 1 },
    ///     Time(Duration::from_secs(1)),
    /// );
    /// assert_eq!(
    ///     dag.poll(Time(Duration::from_secs(1))),
    ///     vec![Output::Ready { job: 2 }]
    /// );
    /// ```
    pub fn release(&mut self, job: JobId, now: Time) -> bool {
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
    /// completed from now on.
    ///
    /// Use when ids are allocated increasingly and everything below `floor` is known to be done,
    /// to keep memory proportional to the live frontier. A dependency on an id below `floor` is
    /// met, and declaring one is a [`DagError::Duplicate`].
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use whelm::{
    /// #     Config, DagConfig, DagError, DagJob, DagScheduler, Input, JobSpec, Output, Policy,
    /// #     Scheduler, Time, WorkerState,
    /// # };
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
    /// # let job = |id, deps| DagJob {
    /// #     spec: JobSpec { id, ..Default::default() },
    /// #     deps,
    /// #     ..Default::default()
    /// # };
    /// dag.declare([job(1, vec![])], Time::ORIGIN).unwrap();
    /// dag.poll(Time::ORIGIN);
    /// dag.handle(
    ///     Input::Done { job: 1, attempt: 1 },
    ///     Time(Duration::from_secs(1)),
    /// );
    /// assert_eq!(dag.dag_stats().completed_remembered, 1);
    ///
    /// dag.forget_completed_below(10);
    /// assert_eq!(dag.dag_stats().completed_remembered, 0);
    /// assert_eq!(
    ///     dag.declare([job(5, vec![])], Time(Duration::from_secs(1))),
    ///     Err(DagError::Duplicate(5))
    /// );
    /// dag.declare([job(10, vec![1, 9])], Time(Duration::from_secs(1)))
    ///     .unwrap();
    /// assert_eq!(
    ///     dag.poll(Time(Duration::from_secs(1))),
    ///     vec![Output::Start {
    ///         job: 10,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn forget_completed_below(&mut self, floor: JobId) {
        if floor > self.completed_floor {
            self.completed_floor = floor;
            self.completed.retain(|&j| j >= floor);
        }
    }

    /// Drop announcements not yet polled ([`Output::Ready`], [`Output::RunLocal`]) of the jobs
    /// for which `gone` holds.
    fn unannounce(&mut self, gone: impl Fn(JobId) -> bool) {
        self.outbox.retain(
            |o| !matches!(*o, Output::Ready { job } | Output::RunLocal { job } if gone(job)),
        );
    }

    /// Cancel a unit and, transitively, every unit depending on it (they can never run).
    ///
    /// The unit is named by its id or any of its jobs. Returns the cancelled units' ids (a plain
    /// job's is the job's). Each of their jobs the inner policy has is cancelled there (its live
    /// attempts are stopped). An id this layer does not know is cancelled in the inner policy.
    /// [`Input::Cancel`] does the same.
    ///
    /// Cancelling running job 1 takes its dependents 2 and 3 with it, and stops its attempt:
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use whelm::{
    /// #     Config, DagConfig, DagJob, DagScheduler, DagStats, Input, JobSpec, Output, Policy,
    /// #     Scheduler, Time, WorkerState,
    /// # };
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
    /// # let job = |id, deps| DagJob {
    /// #     spec: JobSpec { id, ..Default::default() },
    /// #     deps,
    /// #     ..Default::default()
    /// # };
    /// dag.declare(
    ///     [job(1, vec![]), job(2, vec![1]), job(3, vec![2])],
    ///     Time::ORIGIN,
    /// )
    /// .unwrap();
    /// dag.poll(Time::ORIGIN);
    /// assert_eq!(dag.cancel(1), vec![1, 2, 3]);
    /// assert_eq!(
    ///     dag.poll(Time(Duration::from_secs(1))),
    ///     vec![Output::Stop {
    ///         job: 1,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// assert_eq!(dag.dag_stats(), DagStats::default());
    /// ```
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

    /// Close a unit early (named by its id or any of its jobs).
    ///
    /// For when its remaining jobs are known to be no-ops. If it is open, its jobs that have not
    /// started complete as no-ops (waiting ones are withdrawn from the policy) and the unit
    /// completes; it returns the jobs already running, whose workers keep their resources until
    /// each one's attempt ends ([`Input::Done`], [`Input::Failed`], or its worker leaving), which
    /// then changes nothing else; such a job is not retried. A unit not entered yet completes as
    /// soon as its dependencies do, without running anything.
    ///
    /// Unit 200 has jobs 100 and 101 and job 2 waits for it. Closing it while job 100 runs
    /// withdraws job 101 and completes the unit, so job 2 is submitted; job 100's worker is busy
    /// until its attempt ends. Unit 300, closed before it is entered, never runs anything.
    ///
    /// ```
    /// # use std::{sync::Arc, time::Duration};
    /// # use whelm::{
    /// #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Scheduler,
    /// #     TemplateSpec, Time, Unit, WorkerState,
    /// # };
    /// # let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::fifo()));
    /// # dag.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
    /// # let job = |id, deps| DagJob {
    /// #     spec: JobSpec { id, ..Default::default() },
    /// #     deps,
    /// #     ..Default::default()
    /// # };
    /// let pair = Arc::new(TemplateSpec::jobs(2).build().unwrap());
    /// let first = Unit {
    ///     id: 200,
    ///     base: 100,
    ///     template: pair.clone(),
    ///     ..Default::default()
    /// };
    /// let second = Unit {
    ///     id: 300,
    ///     base: 110,
    ///     template: pair,
    ///     deps: vec![2],
    ///     ..Default::default()
    /// };
    /// dag.declare([first, second, job(2, vec![200]).into()], Time::ORIGIN)
    ///     .unwrap();
    /// assert_eq!(
    ///     dag.poll(Time::ORIGIN),
    ///     vec![Output::Start {
    ///         job: 100,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// assert_eq!(dag.close(300, Time::ORIGIN), Ok(vec![]));
    /// assert_eq!(dag.close(200, Time(Duration::from_secs(1))), Ok(vec![100]));
    /// assert_eq!((dag.stats().waiting, dag.stats().running), (1, 1));
    ///
    /// dag.handle(
    ///     Input::Done {
    ///         job: 100,
    ///         attempt: 1,
    ///     },
    ///     Time(Duration::from_secs(2)),
    /// );
    /// assert_eq!(
    ///     dag.poll(Time(Duration::from_secs(2))),
    ///     vec![Output::Start {
    ///         job: 2,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// dag.handle(
    ///     Input::Done { job: 2, attempt: 1 },
    ///     Time(Duration::from_secs(3)),
    /// );
    /// assert!(dag.poll(Time(Duration::from_secs(3))).is_empty());
    /// assert_eq!(dag.dag_stats().units, 0);
    /// ```
    pub fn close(&mut self, job: JobId, now: Time) -> Result<Vec<JobId>, DagError> {
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
    pub(super) fn drop_attempt(&mut self, job: JobId, attempt: Attempt) -> bool {
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
    pub(super) fn done(&mut self, job: JobId, attempt: Attempt, now: Time) {
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
    pub(super) fn failed(
        &mut self,
        job: JobId,
        attempt: Attempt,
        kind: crate::FailKind,
        why: String,
    ) {
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
    pub(super) fn worker_gone(&mut self, w: WorkerId, now: Time) {
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
    pub(super) fn hold_again(&mut self, job: JobId) {
        if let Some((f, i)) = self.leaf_node(job) {
            let frame = self.frame_mut(f);
            if frame.counter[i] == SUBMITTED {
                frame.counter[i] = HELD;
            }
        }
    }
}
