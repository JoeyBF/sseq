//! Materialised units: a counter per template node, from entry to completion.
//!
//! An entered unit gets a [`Frame`]: one `u16` per node of its template, counting the node's unmet
//! predecessors, or once the node is past waiting a sentinel saying where it is ([`HELD`],
//! [`SUBMITTED`], [`OPEN`] for a substituted unit with a frame of its own, [`COMPLETE`]). A
//! substituted unit's frame is a child of its node's and is made only when that node is ready.
//! Leaves declared complete start as [`COMPLETE`] with their successors' counts reduced.
//!
//! Readiness is processed as a queue of [`Work`] events drained by `settle`, so completing one
//! node can release a chain of passthroughs, nested frames and units without recursion. A frame
//! is freed as soon as its last node completes, and the unit's record when its top frame is.
//!
//! Ranks do not depend on frames: a leaf's rank comes from the template's bottom levels, the
//! unit's scale and the rank below the unit, whether or not the leaf is materialised
//! (`leaf_rank`).

use std::sync::Arc;

use super::{DagScheduler, DagTemplate, Loc, NodeSource, TemplateNode, UnitState};
use crate::{Input, Instant, JobId, Output, Policy};

/// Counter sentinels (template in-degrees stay below them).
pub(super) const COMPLETE: u16 = u16::MAX;
/// See [`COMPLETE`].
pub(super) const SUBMITTED: u16 = u16::MAX - 1;
/// See [`COMPLETE`]. Also a ready local job.
pub(super) const HELD: u16 = u16::MAX - 2;
/// A substituted unit with its own frame. See [`COMPLETE`].
pub(super) const OPEN: u16 = u16::MAX - 3;
/// The smallest sentinel: real counters are below it.
pub(super) const SENTINEL: u16 = OPEN;

/// A readiness event.
#[derive(Clone, Copy, Debug)]
pub(super) enum Work {
    /// A unit's dependencies completed.
    Open(u32),
    /// A frame's node has no unmet dependency left.
    Node(u32, u32),
    /// Every node of a frame completed.
    FrameDone(u32),
}

/// The materialised state of an entered unit, or of a unit substituted for a node of one.
#[derive(Clone, Debug)]
pub(super) struct Frame {
    /// The top-level unit's slot.
    pub(super) unit: u32,
    /// The template this frame instantiates: the unit's, or a substituted node's.
    pub(super) template: Arc<DagTemplate>,
    /// The id of the frame's first leaf.
    pub(super) base: JobId,
    /// The index of the frame's first leaf among the unit's.
    pub(super) leaf0: u32,
    /// The enclosing frame and the node this one stands for.
    pub(super) parent: Option<(u32, u32)>,
    /// Rank below the frame within its unit, before the unit's scale and tail.
    pub(super) tail: f64,
    /// Each node's bottom level before the unit's scale, for sourced units (the template's
    /// otherwise).
    pub(super) bl: Option<Box<[f64]>>,
    /// Unmet dependencies of each node, or a sentinel.
    pub(super) counter: Box<[u16]>,
    /// Nodes not complete.
    pub(super) remaining: u32,
    /// Frames of the substituted units entered so far: `(node, frame)`.
    pub(super) children: Vec<(u32, u32)>,
}

impl Frame {
    /// Node `i`'s bottom level before the unit's scale.
    pub(super) fn bottom_level(&self, i: usize) -> f64 {
        match &self.bl {
            Some(bl) => bl[i],
            None => self.template.own_bottom_level(i),
        }
    }

    /// Approximate memory, bytes.
    pub(super) fn bytes(&self) -> usize {
        size_of::<Option<Frame>>()
            + self.counter.len() * size_of::<u16>()
            + self.bl.as_ref().map_or(0, |b| b.len() * size_of::<f64>())
            + self.children.capacity() * size_of::<(u32, u32)>()
    }
}

/// Bottom levels of `t`'s nodes placed at leaf `leaf0` of sourced unit `unit`, before the unit's
/// scale: a substituted unit weighs its own critical path, a leaf the source makes a passthrough
/// nothing.
pub(super) fn sourced_bottom_levels(
    src: &dyn NodeSource,
    unit: JobId,
    t: &DagTemplate,
    leaf0: u32,
) -> Vec<f64> {
    t.bottom_levels(|i| {
        let leaf = leaf0 + t.leaf_offset(i) as u32;
        match t.node(i) {
            TemplateNode::Unit(sub) => sourced_bottom_levels(src, unit, sub, leaf)
                .into_iter()
                .fold(0.0, f64::max),
            _ if src.passthrough(unit, leaf) => 0.0,
            _ => src.work(unit, leaf),
        }
    })
}

impl<P: Policy> DagScheduler<P> {
    /// The frame in slot `f`.
    pub(super) fn frame(&self, f: u32) -> &Frame {
        self.frames[f as usize].as_ref().expect("a live frame")
    }

    /// The frame in slot `f`, mutably.
    pub(super) fn frame_mut(&mut self, f: u32) -> &mut Frame {
        self.frames[f as usize].as_mut().expect("a live frame")
    }

    /// The materialised frame and node of job `job`, if it is a leaf with materialised state.
    pub(super) fn leaf_node(&self, job: JobId) -> Option<(u32, usize)> {
        let Loc::Leaf { unit, leaf } = self.locate(job)? else {
            return None;
        };
        if self.unit(unit).state != UnitState::Open {
            return None;
        }
        let mut f = self.unit(unit).frame;
        loop {
            let frame = self.frame(f);
            let i = frame.template.node_of(leaf - frame.leaf0);
            match frame.template.node(i) {
                TemplateNode::Unit(_) => {
                    f = frame.children.iter().find(|c| c.0 as usize == i)?.1;
                }
                _ => return Some((f, i)),
            }
        }
    }

    /// The rank of node `i` of frame `f`.
    fn node_rank(&self, f: u32, i: usize) -> f64 {
        let frame = self.frame(f);
        let rec = self.unit(frame.unit);
        rec.scale * (frame.bottom_level(i) + frame.tail) + rec.tail
    }

    /// The rank of leaf `leaf` of unit `u`, whether or not it is materialised.
    pub(super) fn leaf_rank(&self, u: u32, leaf: u32) -> f64 {
        let rec = self.unit(u);
        if let Some((f, i)) = self.leaf_node(rec.base + JobId::from(leaf)) {
            return self.node_rank(f, i);
        }
        let mut t = rec.template.clone().expect("a declared unit");
        let (mut leaf0, mut tail) = (0u32, 0.0);
        loop {
            let bl = rec
                .sourced
                .then(|| sourced_bottom_levels(self.src(), rec.id, &t, leaf0));
            let level = |i: usize| bl.as_ref().map_or(t.own_bottom_level(i), |b| b[i]);
            let i = t.node_of(leaf - leaf0);
            let TemplateNode::Unit(sub) = t.node(i) else {
                return rec.scale * (level(i) + tail) + rec.tail;
            };
            let first = leaf0 + t.leaf_offset(i) as u32;
            let span = match rec.sourced {
                true => sourced_bottom_levels(self.src(), rec.id, sub, first)
                    .into_iter()
                    .fold(0.0, f64::max),
                false => sub.span(),
            };
            tail += level(i) - span;
            leaf0 = first;
            t = sub.clone();
        }
    }

    /// Process readiness events until none is left.
    pub(super) fn settle(&mut self, now: Instant) {
        while let Some(w) = self.work.pop_front() {
            match w {
                Work::Open(u) => self.enter(u),
                Work::Node(f, i) => self.node_ready(f, i as usize, now),
                Work::FrameDone(f) => self.frame_done(f),
            }
        }
    }

    /// Unit `u`'s dependencies completed: materialise it (or, closed early, complete it).
    fn enter(&mut self, u: u32) {
        let rec = self.unit_mut(u);
        rec.state = UnitState::Open;
        if rec.closed {
            self.finish_unit(u);
            return;
        }
        let (template, base) = (rec.template.clone().expect("declared"), rec.base);
        let f = self.make_frame(u, template, base, 0, None, 0.0);
        self.unit_mut(u).frame = f;
    }

    /// Materialise a frame: count each node's dependencies, those of complete leaves already
    /// met, and queue the ready nodes.
    fn make_frame(
        &mut self,
        u: u32,
        template: Arc<DagTemplate>,
        base: JobId,
        leaf0: u32,
        parent: Option<(u32, u32)>,
        tail: f64,
    ) -> u32 {
        let rec = self.unit(u);
        let n = template.len();
        let mut counter: Vec<u16> = (0..n)
            .map(|i| template.predecessors(i).len() as u16)
            .collect();
        let mut remaining = n as u32;
        for (i, c) in counter.iter_mut().enumerate() {
            let leaf = !matches!(template.node(i), TemplateNode::Unit(_));
            if leaf && rec.leaf_completed(leaf0 + template.leaf_offset(i) as u32) {
                *c = COMPLETE;
                remaining -= 1;
            }
        }
        for i in 0..n {
            if counter[i] == COMPLETE {
                for &c in template.successors(i) {
                    if counter[c as usize] < SENTINEL {
                        counter[c as usize] -= 1;
                    }
                }
            }
        }
        let bl = rec.sourced.then(|| {
            sourced_bottom_levels(self.src(), rec.id, &template, leaf0).into_boxed_slice()
        });
        let frame = Frame {
            unit: u,
            template,
            base,
            leaf0,
            parent,
            tail,
            bl,
            counter: counter.into_boxed_slice(),
            remaining,
            children: Vec::new(),
        };
        let f = self.alloc_frame(frame);
        self.queue_frame(f);
        f
    }

    /// Queue a new frame's ready nodes, or its completion if nothing is left to do.
    pub(super) fn queue_frame(&mut self, f: u32) {
        let frame = self.frame(f);
        if frame.remaining == 0 {
            self.work.push_back(Work::FrameDone(f));
            return;
        }
        let ready: Vec<u32> = (0..frame.counter.len() as u32)
            .filter(|&i| frame.counter[i as usize] == 0)
            .collect();
        self.work
            .extend(ready.into_iter().map(|i| Work::Node(f, i)));
    }

    /// Put a frame in a free slot.
    pub(super) fn alloc_frame(&mut self, frame: Frame) -> u32 {
        match self.free_frames.pop() {
            Some(f) => {
                self.frames[f as usize] = Some(frame);
                f
            }
            None => {
                self.frames.push(Some(frame));
                (self.frames.len() - 1) as u32
            }
        }
    }

    /// The frame for node `i` of frame `f`, a substituted unit: its sources wait for nothing more.
    pub(super) fn child_frame(&mut self, f: u32, i: usize) -> (Arc<DagTemplate>, JobId, u32, f64) {
        let frame = self.frame(f);
        let TemplateNode::Unit(sub) = frame.template.node(i) else {
            unreachable!("node {i} is not a unit");
        };
        let offset = frame.template.leaf_offset(i);
        let leaf0 = frame.leaf0 + offset as u32;
        let rec = self.unit(frame.unit);
        let span = match rec.sourced {
            true => sourced_bottom_levels(self.src(), rec.id, sub, leaf0)
                .into_iter()
                .fold(0.0, f64::max),
            false => sub.span(),
        };
        let tail = frame.tail + frame.bottom_level(i) - span;
        (sub.clone(), frame.base + offset as JobId, leaf0, tail)
    }

    /// Node `i` of frame `f` has no unmet dependency left: submit, hold or announce a job,
    /// complete a passthrough, or enter a substituted unit.
    fn node_ready(&mut self, f: u32, i: usize, now: Instant) {
        let frame = self.frame(f);
        let offset = frame.template.leaf_offset(i);
        let job = frame.base + offset as JobId;
        let rec = self.unit(frame.unit);
        match frame.template.node(i) {
            TemplateNode::Job(_) | TemplateNode::Local(_)
                if rec.sourced && self.src().passthrough(rec.id, frame.leaf0 + offset as u32) =>
            {
                self.pass(f, i, job)
            }
            TemplateNode::Job(_) if self.config.auto_submit => self.submit(f, i, now),
            TemplateNode::Job(_) => {
                self.frame_mut(f).counter[i] = HELD;
                self.outbox.push(Output::Ready { job });
            }
            TemplateNode::Local(_) => {
                self.frame_mut(f).counter[i] = HELD;
                self.outbox.push(Output::RunLocal { job });
            }
            TemplateNode::Pass(_) => self.pass(f, i, job),
            TemplateNode::Unit(_) => {
                let (template, base, leaf0, tail) = self.child_frame(f, i);
                self.frame_mut(f).counter[i] = OPEN;
                let u = self.frame(f).unit;
                let child = self.make_frame(u, template, base, leaf0, Some((f, i as u32)), tail);
                self.frame_mut(f).children.push((i as u32, child));
            }
        }
    }

    /// Complete node `i` of frame `f`, job `job`, a ready passthrough.
    fn pass(&mut self, f: u32, i: usize, job: JobId) {
        if self.config.record_passthrough {
            self.outbox.push(Output::Passed { job });
        }
        self.complete_node(f, i);
    }

    /// Mark node `i` of frame `f` complete and queue what that releases.
    pub(super) fn complete_node(&mut self, f: u32, i: usize) {
        let frame = self.frames[f as usize].as_mut().expect("a live frame");
        frame.counter[i] = COMPLETE;
        frame.remaining -= 1;
        for &c in frame.template.successors(i) {
            let c = c as usize;
            // A successor already complete (declared so) stays so.
            if frame.counter[c] >= SENTINEL {
                continue;
            }
            frame.counter[c] -= 1;
            if frame.counter[c] == 0 {
                self.work.push_back(Work::Node(f, c as u32));
            }
        }
        if frame.remaining == 0 {
            self.work.push_back(Work::FrameDone(f));
        }
    }

    /// Every node of frame `f` completed: free it, and complete what it stands for.
    fn frame_done(&mut self, f: u32) {
        let frame = self.frames[f as usize].take().expect("a live frame");
        self.free_frames.push(f);
        match frame.parent {
            Some((p, i)) => {
                self.frame_mut(p).children.retain(|c| c.1 != f);
                self.complete_node(p, i as usize);
            }
            None => self.finish_unit(frame.unit),
        }
    }

    /// Unit `u` completed: release its dependents, drop it and remember it as completed.
    pub(super) fn finish_unit(&mut self, u: u32) {
        let rec = self.free_unit(u);
        if self.config.record_passthrough && !rec.plain() {
            self.outbox.push(Output::Passed { job: rec.id });
        }
        for &v in &rec.succs {
            let dependent = self.unit_mut(v);
            dependent.preds.retain(|&p| p != u);
            if dependent.preds.is_empty() && dependent.state == UnitState::Pending {
                self.work.push_back(Work::Open(v));
            }
        }
        if rec.id >= self.completed_floor {
            self.completed.insert(rec.id);
        }
    }

    /// Hand node `i` of frame `f`, a ready job, to the policy.
    pub(super) fn submit(&mut self, f: u32, i: usize, now: Instant) {
        self.frame_mut(f).counter[i] = SUBMITTED;
        let frame = self.frame(f);
        let rec = self.unit(frame.unit);
        let offset = frame.template.leaf_offset(i);
        let leaf = frame.leaf0 + offset as u32;
        let mut spec = rec.spec.clone();
        spec.id = frame.base + offset as JobId;
        if spec.work.is_none() {
            let own = match (rec.sourced, frame.template.node(i)) {
                (true, _) => self.src().work(rec.id, leaf),
                (false, TemplateNode::Job(w)) => *w,
                (false, _) => unreachable!("only worker jobs are submitted"),
            };
            spec.work = Some(rec.scale * own);
        }
        if self.config.track_ranks && spec.rank.is_none() {
            spec.rank = Some(self.node_rank(f, i));
        }
        if rec.sourced {
            self.src().spec(rec.id, leaf, &mut spec);
        }
        self.policy.handle(Input::Submit(spec), now);
    }

    /// Free every frame of open unit `u`; returns its submitted and its held jobs.
    pub(super) fn drop_frames(&mut self, u: u32) -> (Vec<JobId>, Vec<JobId>) {
        let (mut submitted, mut held) = (Vec::new(), Vec::new());
        let mut stack = vec![self.unit(u).frame];
        while let Some(f) = stack.pop() {
            let frame = self.frames[f as usize].take().expect("a live frame");
            self.free_frames.push(f);
            for (i, &c) in frame.counter.iter().enumerate() {
                let job = frame.base + frame.template.leaf_offset(i) as JobId;
                match c {
                    SUBMITTED => submitted.push(job),
                    HELD => held.push(job),
                    _ => {}
                }
            }
            stack.extend(frame.children.iter().map(|c| c.1));
        }
        (submitted, held)
    }
}
