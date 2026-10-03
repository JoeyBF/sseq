//! Declaring units: admission into the graph, cycle checks, and announcing held jobs.

use std::collections::HashMap;

use super::{DagError, DagScheduler, Loc, Unit, UnitRec, UnitState, frame, frame::Work};
#[cfg(doc)]
use crate::DagJob;
use crate::{JobId, Output, Policy, Time};

impl<P: Policy> DagScheduler<P> {
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
            template.span().as_secs_f64()
        };
        let u = match existing {
            Some(u) => u,
            None => {
                let u = self.alloc_unit(UnitRec::undeclared(id));
                self.ids.insert(id, u);
                u
            }
        };
        let scale = (unit.scale).unwrap_or(self.config.default_work.as_secs_f64());
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

    /// Declare units (or plain [`DagJob`]s).
    ///
    /// The graph grows during the run; dependencies may be forward references. Rejects (and
    /// leaves no trace of) a batch that would create a cycle, redeclare a unit or overlap another
    /// unit's ids. Units whose dependencies are all complete are entered at once.
    ///
    /// Job 3 names job 2 before it exists; declaring job 2 after job 3 would close a cycle, so
    /// it is refused and the graph is as it was. A dependency on a completed job is met.
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
    /// dag.declare([job(1, vec![]), job(3, vec![2])], Time::ORIGIN)
    ///     .unwrap();
    /// assert_eq!(dag.explain(3).unwrap(), "job 3 waits for 1 dependency [2]");
    /// assert_eq!(
    ///     dag.explain(2).unwrap(),
    ///     "unit 2 is not declared yet (named as a dependency of 1 unit(s))"
    /// );
    /// let before = dag.dag_stats();
    /// assert_eq!(
    ///     dag.declare([job(2, vec![3])], Time::ORIGIN),
    ///     Err(DagError::Cycle { job: 2 })
    /// );
    /// assert_eq!(dag.dag_stats(), before);
    ///
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
    /// dag.declare([job(2, vec![1])], Time(Duration::from_secs(1)))
    ///     .unwrap();
    /// assert_eq!(
    ///     dag.poll(Time(Duration::from_secs(1))),
    ///     vec![Output::Start {
    ///         job: 2,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn declare<U: Into<Unit>>(
        &mut self,
        units: impl IntoIterator<Item = U>,
        now: Time,
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

    /// Drain this layer's own announcements without polling the inner policy.
    ///
    /// These are [`Output::RunLocal`], [`Output::Ready`] and [`Output::Passed`]; draining them
    /// lets the caller act on them (release, declare, close) before anything is placed in the
    /// same instant. The next [`poll`](Policy::poll) returns the announcements made since, in
    /// order, ahead of the inner policy's outputs.
    ///
    /// Jobs 1 and 2 are ready together; the caller releases job 2 first, so it takes the one
    /// slot.
    ///
    /// ```
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
    /// dag.declare([job(1, vec![]), job(2, vec![])], Time::ORIGIN)
    ///     .unwrap();
    /// assert_eq!(
    ///     dag.announcements(),
    ///     vec![Output::Ready { job: 1 }, Output::Ready { job: 2 }]
    /// );
    /// assert!(dag.announcements().is_empty());
    /// assert!(dag.release(2, Time::ORIGIN) && dag.release(1, Time::ORIGIN));
    /// assert_eq!(
    ///     dag.poll(Time::ORIGIN),
    ///     vec![Output::Start {
    ///         job: 2,
    ///         attempt: 1,
    ///         worker: 1
    ///     }]
    /// );
    /// ```
    pub fn announcements(&mut self) -> Vec<Output> {
        std::mem::take(&mut self.outbox)
    }
}
