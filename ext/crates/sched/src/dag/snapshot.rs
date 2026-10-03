//! Snapshots of the coarse graph and the materialised units.

use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};

use super::{
    DagConfig, DagScheduler, DagTemplate, HELD, NodeSource, SUBMITTED, Source, TemplateNode,
    UnitRec, UnitState,
    frame::{Frame, sourced_bottom_levels},
};
use crate::{Instant, JobId, JobSpec, Output, Policy};

/// A serialisable snapshot of a [`DagScheduler`]'s declared graph (not of its policy); see
/// [`DagScheduler::snapshot`] and [`DagScheduler::restore`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagSnapshot {
    config: DagConfig,
    /// Every template in use, each once, those substituted in another before it.
    templates: Vec<TemplateSnapshot>,
    /// Live units and forward references, by id.
    units: Vec<UnitSnapshot>,
    /// Materialised frames, each after its parent.
    frames: Vec<FrameSnapshot>,
    completed: Vec<JobId>,
    completed_floor: JobId,
}

/// A template, its substituted templates by index.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TemplateSnapshot {
    nodes: Vec<NodeSnapshot>,
    edges: Vec<(u32, u32)>,
}

/// A template node, a substituted template by index.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NodeSnapshot {
    /// [`TemplateNode::Job`].
    Job(f64),
    /// [`TemplateNode::Local`].
    Local(f64),
    /// [`TemplateNode::Pass`].
    Pass(f64),
    /// [`TemplateNode::Unit`].
    Unit(usize),
}

/// A unit; `template: None` for a forward reference.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct UnitSnapshot {
    id: JobId,
    base: JobId,
    template: Option<usize>,
    spec: JobSpec,
    scale: f64,
    sourced: bool,
    closed: bool,
    tail: f64,
    preds: Vec<JobId>,
    completed: Vec<u32>,
}

/// A frame: its unit's top frame, or the frame of node `parent.1` of frame `parent.0`.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct FrameSnapshot {
    unit: JobId,
    parent: Option<(usize, u32)>,
    counter: Vec<u16>,
    remaining: u32,
}

/// Templates by index, deduplicated by identity.
#[derive(Default)]
struct Templates {
    index: HashMap<*const DagTemplate, usize>,
    out: Vec<TemplateSnapshot>,
}

impl Templates {
    /// The index of `t`, after its substituted templates'.
    fn add(&mut self, t: &Arc<DagTemplate>) -> usize {
        if let Some(&i) = self.index.get(&Arc::as_ptr(t)) {
            return i;
        }
        let nodes = (0..t.len())
            .map(|i| match t.node(i) {
                TemplateNode::Job(w) => NodeSnapshot::Job(*w),
                TemplateNode::Local(w) => NodeSnapshot::Local(*w),
                TemplateNode::Pass(w) => NodeSnapshot::Pass(*w),
                TemplateNode::Unit(sub) => NodeSnapshot::Unit(self.add(sub)),
            })
            .collect();
        let edges = (0..t.len())
            .flat_map(|a| t.successors(a).iter().map(move |&b| (a as u32, b)))
            .collect();
        self.out.push(TemplateSnapshot { nodes, edges });
        self.index.insert(Arc::as_ptr(t), self.out.len() - 1);
        self.out.len() - 1
    }
}

impl<P: Policy> DagScheduler<P> {
    /// A serialisable snapshot of the declared graph (not of the policy): the coarse graph of
    /// units with their ranks, the materialised state of entered ones, and remembered
    /// completions. Running jobs of units closed early are not kept.
    pub fn snapshot(&self) -> DagSnapshot {
        let mut templates = Templates::default();
        let mut live: Vec<&UnitRec> = self.units.iter().flatten().collect();
        live.sort_unstable_by_key(|r| r.id);
        let mut units = Vec::with_capacity(live.len());
        let mut frames = Vec::new();
        for rec in live {
            let mut preds: Vec<JobId> = rec.preds.iter().map(|&p| self.unit(p).id).collect();
            preds.sort_unstable();
            units.push(UnitSnapshot {
                id: rec.id,
                base: rec.base,
                template: rec.template.as_ref().map(|t| templates.add(t)),
                spec: rec.spec.clone(),
                scale: rec.scale,
                sourced: rec.sourced,
                closed: rec.closed,
                tail: rec.tail,
                preds,
                completed: rec.completed.to_vec(),
            });
            if rec.state != UnitState::Open {
                continue;
            }
            let mut stack = vec![(rec.frame, None)];
            while let Some((f, parent)) = stack.pop() {
                let frame = self.frame(f);
                let at = frames.len();
                frames.push(FrameSnapshot {
                    unit: rec.id,
                    parent,
                    counter: frame.counter.to_vec(),
                    remaining: frame.remaining,
                });
                stack.extend(frame.children.iter().map(|&(i, c)| (c, Some((at, i)))));
            }
        }
        let mut completed: Vec<_> = self.completed.iter().copied().collect();
        completed.sort_unstable();
        DagSnapshot {
            config: self.config.clone(),
            templates: templates.out,
            units,
            frames,
            completed,
            completed_floor: self.completed_floor,
        }
    }

    /// Restore a scheduler from a snapshot, in front of a fresh `policy`, with the [`NodeSource`]
    /// its sourced units need. Jobs that were submitted (waiting or running in the old policy)
    /// are submitted again at `now`; held jobs stay held and are announced again
    /// ([`Output::RunLocal`], then [`Output::Ready`]).
    pub fn restore(
        snapshot: DagSnapshot,
        policy: P,
        source: Option<Arc<dyn NodeSource>>,
        now: Instant,
    ) -> Self {
        let mut s = Self::new(snapshot.config, policy);
        s.source = source.map(Source);
        s.completed = snapshot.completed.into_iter().collect();
        s.completed_floor = snapshot.completed_floor;
        let mut templates: Vec<Arc<DagTemplate>> = Vec::with_capacity(snapshot.templates.len());
        for t in snapshot.templates {
            let nodes = t
                .nodes
                .into_iter()
                .map(|n| match n {
                    NodeSnapshot::Job(w) => TemplateNode::Job(w),
                    NodeSnapshot::Local(w) => TemplateNode::Local(w),
                    NodeSnapshot::Pass(w) => TemplateNode::Pass(w),
                    NodeSnapshot::Unit(i) => TemplateNode::Unit(templates[i].clone()),
                })
                .collect();
            let t = DagTemplate::with_nodes(nodes, t.edges).expect("a snapshot's template");
            templates.push(Arc::new(t));
        }
        let mut preds = Vec::with_capacity(snapshot.units.len());
        for snap in snapshot.units {
            let mut rec = UnitRec::undeclared(snap.id);
            if let Some(t) = snap.template {
                let template = templates[t].clone();
                rec.span = match snap.sourced {
                    true => sourced_bottom_levels(s.src(), snap.id, &template, 0)
                        .into_iter()
                        .fold(0.0, f64::max),
                    false => template.span(),
                };
                rec.state = UnitState::Pending;
                rec.template = Some(template);
            }
            rec.base = snap.base;
            rec.spec = snap.spec;
            rec.scale = snap.scale;
            rec.sourced = snap.sourced;
            rec.closed = snap.closed;
            rec.tail = snap.tail;
            rec.completed = snap.completed.into_boxed_slice();
            let ranged = rec.template.is_some() && !rec.plain() && rec.leaves() > 0;
            let (id, base) = (rec.id, rec.base);
            let u = s.alloc_unit(rec);
            s.ids.insert(id, u);
            if ranged {
                s.ranges.insert(base, u);
            }
            preds.push((u, snap.preds));
        }
        for (u, ps) in preds {
            for p in ps {
                let p = s.ids[&p];
                s.unit_mut(p).succs.push(u);
                s.unit_mut(u).preds.push(p);
            }
        }
        let mut slots: Vec<u32> = Vec::with_capacity(snapshot.frames.len());
        for snap in snapshot.frames {
            let u = s.ids[&snap.unit];
            let (template, base, leaf0, tail) = match snap.parent {
                None => {
                    let rec = s.unit(u);
                    (rec.template.clone().expect("declared"), rec.base, 0, 0.0)
                }
                Some((p, i)) => s.child_frame(slots[p], i as usize),
            };
            let rec = s.unit(u);
            let bl = rec.sourced.then(|| {
                sourced_bottom_levels(s.src(), rec.id, &template, leaf0).into_boxed_slice()
            });
            let f = s.alloc_frame(Frame {
                unit: u,
                template,
                base,
                leaf0,
                parent: snap.parent.map(|(p, i)| (slots[p], i)),
                tail,
                bl,
                counter: snap.counter.into_boxed_slice(),
                remaining: snap.remaining,
                children: Vec::new(),
            });
            match snap.parent {
                None => {
                    let rec = s.unit_mut(u);
                    rec.state = UnitState::Open;
                    rec.frame = f;
                }
                Some((p, i)) => s.frame_mut(slots[p]).children.push((i, f)),
            }
            slots.push(f);
        }
        let (mut local, mut held, mut submitted) = (Vec::new(), Vec::new(), Vec::new());
        for &f in &slots {
            let frame = s.frame(f);
            for (i, &c) in frame.counter.iter().enumerate() {
                let job = frame.base + frame.template.leaf_offset(i) as JobId;
                match (c, frame.template.node(i)) {
                    (HELD, TemplateNode::Local(_)) => local.push(job),
                    (HELD, _) => held.push(job),
                    (SUBMITTED, _) => submitted.push((job, f, i)),
                    _ => {}
                }
            }
        }
        local.sort_unstable();
        held.sort_unstable();
        submitted.sort_unstable();
        s.outbox
            .extend(local.into_iter().map(|job| Output::RunLocal { job }));
        for (_, f, i) in submitted {
            s.submit(f, i, now);
        }
        s.outbox
            .extend(held.into_iter().map(|job| Output::Ready { job }));
        s.now = now;
        s
    }
}
