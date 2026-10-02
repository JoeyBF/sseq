//! Implicit template instances: a group's sub-DAG as counters over a shared template.

#[cfg(feature = "serde")]
use std::collections::HashMap;
use std::sync::Arc;

use super::{DagError, DagScheduler, DagTemplate, State};
use crate::{Instant, JobId, JobSpec, Policy, Resources};

/// Counter sentinels (template in-degrees stay below them).
pub(super) const COMPLETE: u16 = u16::MAX;
/// See [`COMPLETE`].
pub(super) const SUBMITTED: u16 = u16::MAX - 1;
/// See [`COMPLETE`].
pub(super) const HELD: u16 = u16::MAX - 2;
/// The smallest sentinel: real counters are below it.
pub(super) const SENTINEL: u16 = HELD;

/// A group's sub-DAG, opened as an implicit instance of a shared [`DagTemplate`] with
/// [`DagScheduler::open_instance`].
///
/// Node `i` is job `base + i`. Instead of graph nodes and edges, the instance keeps one counter
/// per node (PaRSEC's parameterised task graphs do the same): about 11 bytes per node against
/// hundreds for an explicit node and its edges, which matters for templates of tens of
/// thousands of nodes. A node's [`JobSpec`] is built from `proto` only when it becomes ready.
#[derive(Clone, Debug)]
pub struct InstanceSpec {
    /// The shared structure.
    pub template: Arc<DagTemplate>,
    /// Node `i` is job `base + i`. The range must not overlap other jobs or instances.
    pub base: JobId,
    /// The template's sources become ready when this job completes (any job: explicit, from
    /// another instance's `done`, or already completed).
    pub entry: JobId,
    /// Completed (as a job id) once every node of the instance has: explicit jobs depend on the
    /// instance by naming it. It must not be declared as a job itself.
    pub done: JobId,
    /// Each node's spec, with `id` and `work` replaced.
    pub proto: JobSpec,
    /// Each node's work estimate (ranks, and [`JobSpec::work`]).
    pub work: Vec<f64>,
    /// Nodes that are synchronisation points only: they complete by themselves when ready.
    pub passthrough: Vec<bool>,
    /// Each node's demand, overriding `proto.demand` (one per template node).
    pub demand: Option<Arc<[Resources]>>,
    /// Each node's name in [`explain`](crate::Dag::explain) messages (not kept by snapshots).
    pub label: Option<NodeLabel>,
    /// Nodes already complete (e.g. restored from a checkpoint): they never run, and their
    /// successors start with those dependencies met. Need not be closed under predecessors: an
    /// incomplete predecessor of a complete node still runs, and its completion does not touch
    /// the complete node.
    pub completed: Vec<u32>,
}

/// A node's name for an instance: `label(i)` names node `i`.
#[derive(Clone)]
pub struct NodeLabel(pub Arc<dyn Fn(usize) -> String + Send + Sync>);

impl std::fmt::Debug for NodeLabel {
    /// The closure is opaque.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NodeLabel(..)")
    }
}

/// An open instance.
#[derive(Clone, Debug)]
pub(super) struct Instance {
    pub(super) template: Arc<DagTemplate>,
    pub(super) base: JobId,
    pub(super) entry: JobId,
    pub(super) done: JobId,
    pub(super) proto: JobSpec,
    pub(super) work: Vec<f64>,
    pub(super) pass: Vec<u64>,
    /// Unmet predecessors, or a sentinel.
    pub(super) counter: Vec<u16>,
    /// Nodes not complete.
    pub(super) remaining: u32,
    /// Whether `entry` has completed.
    pub(super) open: bool,
    /// Bottom levels (only with `rank_priority`).
    pub(super) below: Vec<f64>,
    pub(super) demand: Option<Arc<[Resources]>>,
    pub(super) label: Option<NodeLabel>,
}

impl Instance {
    /// Whether node `i` is a passthrough.
    fn is_pass(&self, i: usize) -> bool {
        self.pass[i / 64] >> (i % 64) & 1 == 1
    }

    /// Number of nodes.
    fn len(&self) -> usize {
        self.counter.len()
    }
}

impl<P: Policy> DagScheduler<P> {
    /// Open an implicit instance of a template (see [`InstanceSpec`]). Its sources become ready
    /// when `entry` completes, at once if it already has.
    pub fn open_instance(&mut self, spec: InstanceSpec, now: Instant) -> Result<(), DagError> {
        self.now = now;
        let n = spec.template.len();
        assert_eq!(spec.work.len(), n, "one work estimate per template node");
        assert_eq!(
            spec.passthrough.len(),
            n,
            "one passthrough flag per template node"
        );
        assert!(
            (0..n).all(|i| spec.template.predecessors(i).len() < SENTINEL as usize),
            "template in-degree too large for implicit instances"
        );
        if let Some(d) = &spec.demand {
            assert_eq!(d.len(), n, "one demand per template node");
        }
        let end = spec
            .base
            .checked_add(n as u64)
            .expect("instance id range overflows");
        if let Some((&b, &slot)) = self.by_base.range(..end).next_back()
            && b + self.instances[slot].as_ref().unwrap().len() as u64 > spec.base
        {
            return Err(DagError::Duplicate(spec.base.max(b)));
        }
        if let Some(id) = (spec.base..end).find(|id| self.index.contains_key(id)) {
            return Err(DagError::Duplicate(id));
        }
        let done_declared = self
            .index
            .get(&spec.done)
            .is_some_and(|&d| self.graph[d].state != State::Undeclared);
        if done_declared || self.is_completed(spec.done) || (spec.base..end).contains(&spec.done) {
            return Err(DagError::Duplicate(spec.done));
        }
        let mut pass = vec![0u64; n.div_ceil(64)];
        for (i, &p) in spec.passthrough.iter().enumerate() {
            if p {
                pass[i / 64] |= 1 << (i % 64);
            }
        }
        // The entry's rank must see the instance's critical path, as it would through edges.
        let cp = spec.template.critical_path(|i| spec.work[i]);
        self.by_done
            .entry(spec.done)
            .or_default()
            .push((spec.entry, cp));
        let done_rank = self
            .index
            .get(&spec.done)
            .map_or(0.0, |&d| self.graph[d].rank);
        if self.config.track_ranks
            && let Some(&e) = self.index.get(&spec.entry)
        {
            let below = cp + done_rank;
            if below > self.graph[e].implicit_below {
                self.graph[e].implicit_below = below;
                let rank = self.graph[e].work + below;
                if rank > self.graph[e].rank {
                    self.graph[e].rank = rank;
                    self.propagate_rank(e);
                }
            }
        }
        let below = if self.config.rank_priority {
            spec.template.bottom_levels(|i| spec.work[i])
        } else {
            Vec::new()
        };
        let mut counter: Vec<u16> = (0..n)
            .map(|i| spec.template.predecessors(i).len() as u16)
            .collect();
        let mut remaining = n as u32;
        for &i in &spec.completed {
            if counter[i as usize] != COMPLETE {
                counter[i as usize] = COMPLETE;
                remaining -= 1;
            }
        }
        for &i in &spec.completed {
            for &c in spec.template.successors(i as usize) {
                if counter[c as usize] < SENTINEL {
                    counter[c as usize] -= 1;
                }
            }
        }
        let inst = Instance {
            counter,
            template: spec.template,
            base: spec.base,
            entry: spec.entry,
            done: spec.done,
            proto: spec.proto,
            work: spec.work,
            pass,
            remaining,
            open: false,
            below,
            demand: spec.demand,
            label: spec.label,
        };
        let slot = match self.free_instances.pop() {
            Some(s) => {
                self.instances[s] = Some(inst);
                s
            }
            None => {
                self.instances.push(Some(inst));
                self.instances.len() - 1
            }
        };
        self.by_base.insert(spec.base, slot);
        if remaining == 0 {
            self.close_slot(slot, now);
        } else if self.is_completed(spec.entry) {
            self.open_sources(slot, now);
        } else {
            self.by_entry.entry(spec.entry).or_default().push(slot);
        }
        self.drain_passthrough(now);
        Ok(())
    }

    /// The instance slot and node index of a job id, if it belongs to an open instance.
    pub(super) fn instance_of(&self, job: JobId) -> Option<(usize, usize)> {
        let (&base, &slot) = self.by_base.range(..=job).next_back()?;
        let i = (job - base) as usize;
        (i < self.instances[slot].as_ref()?.len()).then_some((slot, i))
    }

    /// The instance in a slot.
    pub(super) fn inst(&self, slot: usize) -> &Instance {
        self.instances[slot].as_ref().expect("an open instance")
    }

    /// The entry completed: release the sources, or queue the instance if the open budget is
    /// spent.
    pub(super) fn open_sources(&mut self, slot: usize, now: Instant) {
        if let Some(max) = self.config.max_open_instances {
            let open = self.instances.iter().flatten().filter(|i| i.open).count();
            if open >= max.max(1) {
                self.waiting_to_open.push_back(slot);
                return;
            }
        }
        let inst = self.instances[slot].as_mut().unwrap();
        inst.open = true;
        let ready: Vec<usize> = (0..inst.len()).filter(|&i| inst.counter[i] == 0).collect();
        self.instance_ready(slot, ready, now);
    }

    /// Nodes whose last predecessor completed: passthroughs complete at once (a worklist), the
    /// rest are submitted or held.
    fn instance_ready(&mut self, slot: usize, ready: Vec<usize>, now: Instant) {
        // Before the entry completes, nodes stay at zero; `open_sources` releases them.
        if !self.inst(slot).open {
            return;
        }
        let mut work: Vec<usize> = ready.into_iter().rev().collect();
        while let Some(i) = work.pop() {
            if self.inst(slot).is_pass(i) {
                let mut next = self.complete_node(slot, i);
                if self.instances[slot].is_none() {
                    return; // the instance closed
                }
                next.reverse();
                work.extend(next);
                continue;
            }
            let id = self.inst(slot).base + i as u64;
            self.newly_ready.push(id);
            if self.config.auto_submit {
                self.submit_instance_node(slot, i, now);
            } else {
                self.instances[slot].as_mut().unwrap().counter[i] = HELD;
            }
        }
        if self.instances[slot]
            .as_ref()
            .is_some_and(|x| x.remaining == 0)
        {
            self.close_slot(slot, now);
        }
    }

    /// Mark node `i` complete; return its successors that became ready, in index order.
    fn complete_node(&mut self, slot: usize, i: usize) -> Vec<usize> {
        let inst = self.instances[slot].as_mut().unwrap();
        inst.counter[i] = COMPLETE;
        inst.remaining -= 1;
        let mut ready = Vec::new();
        for &c in inst.template.successors(i) {
            let c = c as usize;
            // A successor already complete (opened so, see `InstanceSpec::completed`) stays so.
            if inst.counter[c] >= SENTINEL {
                continue;
            }
            inst.counter[c] -= 1;
            if inst.counter[c] == 0 {
                ready.push(c);
            }
        }
        ready
    }

    /// A submitted instance node completed.
    pub(super) fn finish_instance_node(&mut self, slot: usize, i: usize, now: Instant) {
        if self.inst(slot).counter[i] == COMPLETE {
            return;
        }
        let ready = self.complete_node(slot, i);
        self.instance_ready(slot, ready, now);
        if self.instances[slot]
            .as_ref()
            .is_some_and(|x| x.remaining == 0)
        {
            self.close_slot(slot, now);
        }
    }

    /// Close an instance early (e.g. its remaining nodes are known to be no-ops): its nodes that
    /// have not started complete as no-ops (waiting ones are withdrawn from the policy) and its
    /// `done` job completes. `job` is the instance's `done` job or any of its nodes. Returns the
    /// nodes already running: their workers keep their resources until each one's
    /// [`completed`](crate::Dag::completed) (or [`resubmit`](Self::resubmit) after its worker
    /// left), which then changes nothing else.
    pub fn close_instance(&mut self, job: JobId, now: Instant) -> Result<Vec<JobId>, DagError> {
        self.now = now;
        let slot = match self.instance_of(job) {
            Some((slot, _)) => slot,
            None => self
                .instances
                .iter()
                .position(|i| i.as_ref().is_some_and(|i| i.done == job))
                .ok_or(DagError::NotFound(job))?,
        };
        let (base, len, entry) = {
            let i = self.inst(slot);
            (i.base, i.len(), i.entry)
        };
        let mut running = Vec::new();
        for i in 0..len {
            let id = base + i as u64;
            if self.inst(slot).counter[i] == SUBMITTED {
                if self.running.remove(&id) {
                    self.ignored.insert(id);
                    running.push(id);
                } else {
                    self.policy.cancel(id);
                }
            }
        }
        self.newly_ready
            .retain(|j| !(base..base + len as u64).contains(j));
        if let Some(v) = self.by_entry.get_mut(&entry) {
            v.retain(|&s| s != slot);
        }
        self.waiting_to_open.retain(|&s| s != slot);
        let inst = self.instances[slot].as_mut().unwrap();
        inst.counter.fill(COMPLETE);
        inst.remaining = 0;
        self.close_slot(slot, now);
        self.drain_passthrough(now);
        Ok(running)
    }

    /// Every node completed: drop the instance and complete its `done` job.
    fn close_slot(&mut self, slot: usize, now: Instant) {
        let Some(inst) = self.instances[slot].take() else {
            return;
        };
        self.by_base.remove(&inst.base);
        self.unlink_done(&inst);
        self.free_instances.push(slot);
        if self.config.record_passthrough {
            // An instance's `done` is a synchronisation point like a passthrough job.
            self.passed.push(inst.done);
        }
        // A throttled instance may open now.
        while let Some(next) = self.waiting_to_open.pop_front() {
            if self.instances[next].is_some() {
                self.open_sources(next, now);
                break;
            }
        }
        self.finish(inst.done, now);
    }

    /// Hand a ready instance node to the policy.
    pub(super) fn submit_instance_node(&mut self, slot: usize, i: usize, now: Instant) {
        let inst = self.instances[slot].as_mut().unwrap();
        inst.counter[i] = SUBMITTED;
        let mut spec = inst.proto.clone();
        spec.id = inst.base + i as u64;
        spec.work = Some(inst.work[i]);
        if let Some(d) = &inst.demand {
            spec.demand = d[i];
        }
        if self.config.rank_priority && spec.priority.is_none() {
            // The rank below the instance, now (it may have grown since the instance opened).
            let done_rank = self
                .index
                .get(&inst.done)
                .map_or(0.0, |&d| self.graph[d].rank);
            let below = inst.below[i];
            let group = spec.group;
            let rank = below + done_rank + self.group_tail(group);
            let p = -(rank * self.config.rank_scale).round();
            spec.priority = Some(p.clamp(i64::MIN as f64, i64::MAX as f64) as i64);
        }
        self.policy.submit(spec, now);
    }

    /// Cancel a whole instance (its pending, held and submitted nodes); returns their ids.
    pub(super) fn cancel_instance(&mut self, slot: usize) -> Vec<JobId> {
        let Some(inst) = self.instances[slot].take() else {
            return Vec::new();
        };
        self.by_base.remove(&inst.base);
        self.unlink_done(&inst);
        self.free_instances.push(slot);
        if let Some(v) = self.by_entry.get_mut(&inst.entry) {
            v.retain(|&s| s != slot);
        }
        self.waiting_to_open.retain(|&s| s != slot);
        let mut ids = Vec::new();
        for (i, &c) in inst.counter.iter().enumerate() {
            if c == COMPLETE {
                continue;
            }
            let id = inst.base + i as u64;
            if c == SUBMITTED {
                self.policy.cancel(id);
            }
            ids.push(id);
        }
        self.newly_ready
            .retain(|j| !(inst.base..inst.base + inst.len() as u64).contains(j));
        ids
    }

    /// `explain` for an instance node.
    pub(super) fn explain_instance_node(
        &self,
        slot: usize,
        i: usize,
        job: JobId,
    ) -> Option<String> {
        let inst = self.inst(slot);
        let msg = match inst.counter[i] {
            SUBMITTED => self.policy.explain(job),
            HELD => Some(format!("job {job} is ready and held until release")),
            COMPLETE => Some(format!("job {job} completed")),
            _ if !inst.open => Some(format!(
                "job {job} waits for its group's entry job {}",
                inst.entry
            )),
            k => Some(format!(
                "job {job} waits for {k} dependenc{} within its group",
                if k == 1 { "y" } else { "ies" }
            )),
        };
        match &inst.label {
            Some(l) => msg.map(|m| format!("[{}] {m}", (l.0)(i))),
            None => msg,
        }
    }

    /// Release a held instance node, or resubmit a submitted one; false if neither applies.
    pub(super) fn instance_release(&mut self, job: JobId, now: Instant, want: u16) -> bool {
        match self.instance_of(job) {
            Some((slot, i)) if self.inst(slot).counter[i] == want => {
                self.submit_instance_node(slot, i, now);
                true
            }
            _ => false,
        }
    }

    /// Forget an instance's rank link from its `done` job to its entry.
    fn unlink_done(&mut self, inst: &Instance) {
        if let Some(v) = self.by_done.get_mut(&inst.done) {
            v.retain(|&(e, _)| e != inst.entry);
            if v.is_empty() {
                self.by_done.remove(&inst.done);
            }
        }
    }

    /// Approximate bytes held by instances.
    pub(super) fn instance_bytes(&self) -> usize {
        self.instances
            .iter()
            .flatten()
            .map(|i| {
                std::mem::size_of::<Instance>()
                    + i.counter.len() * 2
                    + i.work.len() * 8
                    + i.pass.len() * 8
                    + i.below.len() * 8
            })
            .sum()
    }

    /// Counts of open instances' nodes: (pending, held, submitted).
    pub(super) fn instance_counts(&self) -> (usize, usize, usize) {
        let (mut p, mut h, mut s) = (0, 0, 0);
        for inst in self.instances.iter().flatten() {
            for &c in &inst.counter {
                match c {
                    COMPLETE => {}
                    SUBMITTED => s += 1,
                    HELD => h += 1,
                    _ => p += 1,
                }
            }
        }
        (p, h, s)
    }
}

/// An open instance in a [`DagSnapshot`](super::DagSnapshot): its template by index.
#[cfg(feature = "serde")]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(super) struct InstanceSnapshot {
    template: usize,
    base: JobId,
    entry: JobId,
    done: JobId,
    proto: JobSpec,
    work: Vec<f64>,
    pass: Vec<u64>,
    counter: Vec<u16>,
    remaining: u32,
    open: bool,
    #[serde(default)]
    demand: Option<Vec<Resources>>,
}

#[cfg(feature = "serde")]
impl<P: Policy> DagScheduler<P> {
    /// Open instances for a snapshot, with their templates deduplicated.
    pub(super) fn snapshot_instances(&self) -> (Vec<DagTemplate>, Vec<InstanceSnapshot>) {
        let mut index: HashMap<*const DagTemplate, usize> = HashMap::new();
        let mut templates = Vec::new();
        let mut out = Vec::new();
        for inst in self.instances.iter().flatten() {
            let t = *index.entry(Arc::as_ptr(&inst.template)).or_insert_with(|| {
                templates.push((*inst.template).clone());
                templates.len() - 1
            });
            out.push(InstanceSnapshot {
                template: t,
                base: inst.base,
                entry: inst.entry,
                done: inst.done,
                proto: inst.proto.clone(),
                work: inst.work.clone(),
                pass: inst.pass.clone(),
                counter: inst.counter.clone(),
                remaining: inst.remaining,
                open: inst.open,
                demand: inst.demand.as_ref().map(|d| d.to_vec()),
            });
        }
        out.sort_by_key(|i| i.base);
        (templates, out)
    }

    /// Reopen snapshotted instances: submitted nodes are submitted again, held ones held again.
    pub(super) fn restore_instances(
        &mut self,
        templates: Vec<DagTemplate>,
        instances: Vec<InstanceSnapshot>,
        now: Instant,
    ) {
        let templates: Vec<Arc<DagTemplate>> = templates.into_iter().map(Arc::new).collect();
        for snap in instances {
            let template = templates[snap.template].clone();
            let cp = template.critical_path(|i| snap.work[i]);
            self.by_done
                .entry(snap.done)
                .or_default()
                .push((snap.entry, cp));
            let below = if self.config.rank_priority {
                template.bottom_levels(|i| snap.work[i])
            } else {
                Vec::new()
            };
            let inst = Instance {
                template,
                base: snap.base,
                entry: snap.entry,
                done: snap.done,
                proto: snap.proto,
                work: snap.work,
                pass: snap.pass,
                counter: snap.counter,
                remaining: snap.remaining,
                open: snap.open,
                below,
                demand: snap.demand.map(Arc::from),
                label: None,
            };
            let slot = self.instances.len();
            if !inst.open {
                self.by_entry.entry(inst.entry).or_default().push(slot);
            }
            self.by_base.insert(inst.base, slot);
            let held: Vec<usize> = (0..inst.len())
                .filter(|&i| inst.counter[i] == HELD)
                .collect();
            let submitted: Vec<usize> = (0..inst.len())
                .filter(|&i| inst.counter[i] == SUBMITTED)
                .collect();
            let base = inst.base;
            self.instances.push(Some(inst));
            for i in submitted {
                self.submit_instance_node(slot, i, now);
            }
            self.newly_ready
                .extend(held.into_iter().map(|i| base + i as u64));
        }
    }
}
