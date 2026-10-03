//! Lazily materialised hierarchical units behave like the fully expanded graph they stand for.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use proptest::prelude::*;
use sched::{
    Config, DagConfig, DagJob, DagScheduler, DagTemplate, Input, JobId, JobSpec, NodeSource,
    Output, Policy, Resources, Scheduler, TemplateNode, Unit, WorkerState,
};

/// A small deterministic generator (splitmix64).
struct Rng(u64);

impl Rng {
    /// The next 64 random bits.
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// True with probability `p`.
    fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }
}

/// Leaf work of sourced units: a fixed function of the unit and the leaf.
struct Works;

impl NodeSource for Works {
    /// Between 0 and 2.
    fn work(&self, unit: JobId, leaf: u32) -> f64 {
        ((unit * 31 + u64::from(leaf) * 7) % 5) as f64 * 0.5
    }
}

/// A unit of the random world.
#[derive(Clone, Debug)]
struct UnitDecl {
    id: JobId,
    base: JobId,
    template: Arc<DagTemplate>,
    deps: Vec<JobId>,
    scale: f64,
    sourced: bool,
    completed: Vec<u32>,
}

impl UnitDecl {
    /// Whether it is a plain job.
    fn plain(&self) -> bool {
        self.id == self.base
    }

    /// The unit to declare.
    fn unit(&self) -> Unit {
        if self.plain() {
            let spec = JobSpec::new(self.id, Resources::ZERO, 0);
            let mut j = DagJob::new(spec, self.deps.clone()).with_work(self.scale);
            match self.template.node(0) {
                TemplateNode::Local(_) => j = j.local(),
                TemplateNode::Pass(_) => j.passthrough = true,
                _ => {}
            }
            return j.into();
        }
        let spec = JobSpec::new(0, Resources::ZERO, 0);
        let mut u = Unit::new(
            self.id,
            self.base,
            self.template.clone(),
            spec,
            self.deps.clone(),
        )
        .with_scale(self.scale)
        .with_completed(self.completed.clone());
        u.sourced = self.sourced;
        u
    }

    /// Leaf `leaf`'s work.
    fn work(&self, leaf: u32, own: f64) -> f64 {
        self.scale
            * if self.sourced {
                Works.work(self.id, leaf)
            } else {
                own
            }
    }
}

/// A random leaf.
fn leaf(rng: &mut Rng) -> TemplateNode {
    let work = rng.below(4) as f64;
    match rng.below(8) {
        0 => TemplateNode::Pass(work),
        1 => TemplateNode::Local(work),
        _ => TemplateNode::Job(work),
    }
}

/// Random forward edges over `n` nodes.
fn edges(rng: &mut Rng, n: usize) -> Vec<(u32, u32)> {
    (0..rng.below(2 * n + 1))
        .map(|_| (rng.below(n.max(1)) as u32, rng.below(n.max(1)) as u32))
        .filter(|&(a, b)| a < b)
        .collect()
}

/// A random world: templates over three levels of substitution, then units of them (and plain
/// jobs), each depending on earlier ones.
fn world(rng: &mut Rng) -> Vec<UnitDecl> {
    let mut templates: Vec<Arc<DagTemplate>> = Vec::new();
    for level in 0..3 {
        for _ in 0..1 + rng.below(2) {
            let n = rng.below(if level == 0 { 7 } else { 5 });
            let nodes = (0..n)
                .map(|_| {
                    if !templates.is_empty() && rng.chance(0.4) {
                        TemplateNode::Unit(templates[rng.below(templates.len())].clone())
                    } else {
                        leaf(rng)
                    }
                })
                .collect();
            let e = edges(rng, n);
            templates.push(Arc::new(DagTemplate::with_nodes(nodes, e).unwrap()));
        }
    }
    let mut units: Vec<UnitDecl> = Vec::new();
    for k in 0..1 + rng.below(8) {
        let deps: BTreeSet<JobId> = (0..rng.below(3))
            .filter(|_| k > 0)
            .map(|_| units[rng.below(k)].id)
            .collect();
        let scale = [0.5, 1.0, 2.0][rng.below(3)];
        let base = 1000 * (k as JobId + 1);
        let u = if rng.chance(0.3) {
            // A plain job's work is its scale.
            let node = match leaf(rng) {
                TemplateNode::Local(_) => TemplateNode::Local(1.0),
                TemplateNode::Pass(_) => TemplateNode::Pass(1.0),
                _ => TemplateNode::Job(1.0),
            };
            let t = Arc::new(DagTemplate::with_nodes(vec![node], []).unwrap());
            UnitDecl {
                id: base,
                base,
                template: t,
                deps: deps.into_iter().collect(),
                scale,
                sourced: false,
                completed: Vec::new(),
            }
        } else {
            let t = templates[rng.below(templates.len())].clone();
            let completed = (0..t.leaves() as u32)
                .filter(|_| rng.chance(0.15))
                .collect();
            UnitDecl {
                id: 900_000 + k as JobId,
                base,
                template: t,
                deps: deps.into_iter().collect(),
                scale,
                sourced: rng.chance(0.4),
                completed,
            }
        };
        units.push(u);
    }
    units
}

/// What a node of the expanded graph is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// A worker job.
    Job,
    /// A local job.
    Local,
    /// A passthrough leaf.
    Pass,
    /// A frame's entry (all its leaves wait for it), or its exit (it waits for every node).
    Sync,
}

/// A node of the expanded graph.
#[derive(Clone, Debug)]
struct Node {
    kind: Kind,
    /// The job id for leaves; for a unit's exit, its id if it is announced.
    id: Option<JobId>,
    work: f64,
    deps: Vec<usize>,
    /// The unit it belongs to.
    unit: usize,
}

/// A unit fully expanded into leaves and synchronisation nodes.
struct Reference {
    nodes: Vec<Node>,
    /// Each unit's entry node.
    enter: Vec<usize>,
    declared: Vec<bool>,
    done: Vec<bool>,
    announced: Vec<bool>,
    ranks: Vec<f64>,
    /// Leaf id -> node.
    by_id: BTreeMap<JobId, usize>,
}

impl Reference {
    /// Expand every unit.
    fn new(units: &[UnitDecl]) -> Self {
        let mut r = Reference {
            nodes: Vec::new(),
            enter: Vec::new(),
            declared: vec![false; units.len()],
            done: Vec::new(),
            announced: Vec::new(),
            ranks: Vec::new(),
            by_id: BTreeMap::new(),
        };
        let index: BTreeMap<JobId, usize> =
            units.iter().enumerate().map(|(k, u)| (u.id, k)).collect();
        let mut exits = Vec::new();
        for (k, u) in units.iter().enumerate() {
            let enter = r.push(Kind::Sync, None, 0.0, Vec::new(), k);
            r.enter.push(enter);
            let exit = r.expand(u, k, &u.template, 0, enter);
            if !u.plain() {
                r.nodes[exit].id = Some(u.id);
            }
            exits.push(exit);
        }
        for (k, u) in units.iter().enumerate() {
            r.nodes[r.enter[k]].deps = u.deps.iter().map(|d| exits[index[d]]).collect();
        }
        r.done = vec![false; r.nodes.len()];
        r.announced = vec![false; r.nodes.len()];
        for u in units {
            for &c in &u.completed {
                r.done[r.by_id[&(u.base + JobId::from(c))]] = true;
            }
        }
        r.ranks = r.longest_paths();
        r
    }

    /// Add a node.
    fn push(
        &mut self,
        kind: Kind,
        id: Option<JobId>,
        work: f64,
        deps: Vec<usize>,
        unit: usize,
    ) -> usize {
        self.nodes.push(Node {
            kind,
            id,
            work,
            deps,
            unit,
        });
        self.nodes.len() - 1
    }

    /// Expand `t` placed at leaf `leaf0` of unit `u`, entered by node `enter`; returns its exit.
    fn expand(
        &mut self,
        u: &UnitDecl,
        k: usize,
        t: &DagTemplate,
        leaf0: u32,
        enter: usize,
    ) -> usize {
        let mut exit = vec![usize::MAX; t.len()];
        for &i in t.topological_order() {
            let i = i as usize;
            let mut deps: Vec<usize> = t
                .predecessors(i)
                .iter()
                .map(|&p| exit[p as usize])
                .collect();
            deps.push(enter);
            let leaf = leaf0 + t.leaf_offset(i) as u32;
            let id = u.base + JobId::from(leaf);
            exit[i] = match t.node(i) {
                TemplateNode::Unit(sub) => {
                    let inner = self.push(Kind::Sync, None, 0.0, deps, k);
                    self.expand(u, k, sub, leaf, inner)
                }
                n => {
                    let (kind, own) = match *n {
                        TemplateNode::Job(w) => (Kind::Job, w),
                        TemplateNode::Local(w) => (Kind::Local, w),
                        TemplateNode::Pass(w) => (Kind::Pass, w),
                        TemplateNode::Unit(_) => unreachable!(),
                    };
                    let node = self.push(kind, Some(id), u.work(leaf, own), deps, k);
                    self.by_id.insert(id, node);
                    node
                }
            };
        }
        let mut deps = exit;
        deps.push(enter);
        self.push(Kind::Sync, None, 0.0, deps, k)
    }

    /// Every node's longest path of work to the end of the graph, itself included.
    fn longest_paths(&self) -> Vec<f64> {
        let n = self.nodes.len();
        let mut succ = vec![Vec::new(); n];
        for (v, node) in self.nodes.iter().enumerate() {
            for &d in &node.deps {
                succ[d].push(v);
            }
        }
        let mut memo: Vec<Option<f64>> = vec![None; n];
        /// Memoised longest path from `v`.
        fn go(v: usize, nodes: &[Node], succ: &[Vec<usize>], memo: &mut Vec<Option<f64>>) -> f64 {
            if let Some(r) = memo[v] {
                return r;
            }
            let below = succ[v]
                .iter()
                .map(|&c| go(c, nodes, succ, memo))
                .fold(0.0, f64::max);
            let r = nodes[v].work + below;
            memo[v] = Some(r);
            r
        }
        (0..n)
            .map(|v| go(v, &self.nodes, &succ, &mut memo))
            .collect()
    }

    /// Run synchronisation nodes and passthroughs to a fixpoint; returns what is announced.
    fn advance(&mut self) -> BTreeSet<Ann> {
        let mut out = BTreeSet::new();
        loop {
            let mut changed = false;
            for v in 0..self.nodes.len() {
                let node = &self.nodes[v];
                if self.done[v] || self.announced[v] || !node.deps.iter().all(|&d| self.done[d]) {
                    continue;
                }
                if node.kind == Kind::Sync && self.enter.contains(&v) && !self.declared[node.unit] {
                    continue;
                }
                changed = true;
                match node.kind {
                    Kind::Job => {
                        self.announced[v] = true;
                        out.insert(Ann::Ready(node.id.unwrap()));
                    }
                    Kind::Local => {
                        self.announced[v] = true;
                        out.insert(Ann::Local(node.id.unwrap()));
                    }
                    Kind::Pass | Kind::Sync => {
                        self.done[v] = true;
                        if let Some(job) = node.id {
                            out.insert(Ann::Passed(job));
                        }
                    }
                }
            }
            if !changed {
                return out;
            }
        }
    }

    /// Unit `k`'s leaves are all done, without announcing anything.
    fn close(&mut self, k: usize) {
        for v in 0..self.nodes.len() {
            if self.nodes[v].unit == k
                && matches!(self.nodes[v].kind, Kind::Job | Kind::Local | Kind::Pass)
            {
                self.done[v] = true;
            }
        }
    }
}

/// An announcement: a job ready to release, to run locally, or passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Ann {
    /// [`Output::Ready`].
    Ready(JobId),
    /// [`Output::RunLocal`].
    Local(JobId),
    /// [`Output::Passed`].
    Passed(JobId),
}

impl Ann {
    /// The job.
    fn job(self) -> JobId {
        match self {
            Self::Ready(j) | Self::Local(j) | Self::Passed(j) => j,
        }
    }
}

/// A poll's announcements, which must be all it returns.
fn set(out: Vec<Output>) -> BTreeSet<Ann> {
    out.into_iter()
        .map(|o| match o {
            Output::Ready { job } => Ann::Ready(job),
            Output::RunLocal { job } => Ann::Local(job),
            Output::Passed { job } => Ann::Passed(job),
            o => panic!("unexpected output {o:?}"),
        })
        .collect()
}

/// A scheduler over one worker, announcing ready jobs.
fn scheduler(eps: f64) -> DagScheduler<Scheduler> {
    let config = DagConfig {
        auto_submit: false,
        rank_priority: true,
        record_passthrough: true,
        rank_epsilon: eps,
        ..DagConfig::default()
    };
    let mut d =
        DagScheduler::new(config, Scheduler::new(Config::default())).with_source(Arc::new(Works));
    join(&mut d, 0.0);
    d
}

/// The worker joins.
fn join(d: &mut DagScheduler<Scheduler>, now: f64) {
    d.handle(
        Input::Worker(WorkerState::new(0, "x", 1, Resources::ZERO)),
        now,
    );
}

/// Run ready job `j` to completion; returns the outputs that follow.
fn run(d: &mut DagScheduler<Scheduler>, j: Ann, now: f64) -> BTreeSet<Ann> {
    match j {
        Ann::Ready(job) => {
            assert!(d.release(job, now), "job {job} is not held");
            assert_eq!(
                d.poll(now),
                vec![Output::Start {
                    job,
                    attempt: 1,
                    worker: 0
                }]
            );
            d.handle(Input::Done { job, attempt: 1 }, now);
        }
        Ann::Local(job) => d.handle(Input::Done { job, attempt: 0 }, now),
        Ann::Passed(job) => panic!("job {job} is not ready"),
    }
    set(d.poll(now))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Random hierarchies of units (templates substituted up to three deep, sourced and scaled
    /// work, leaves declared complete, plain jobs among them), declared in random batches and
    /// orders (so forward references abound), then run in a random order with a unit closed early
    /// and a snapshot restored at random points: every event announces exactly what the fully
    /// expanded graph makes ready, and every announced job's rank is its longest path there
    /// (exactly, or within the compounded `rank_epsilon` above).
    #[test]
    fn units_match_the_expanded_graph(
        seed in any::<u64>(),
        snap_at in 0usize..40,
        close_at in 0usize..40,
        approximate in any::<bool>(),
    ) {
        let mut rng = Rng(seed);
        let units = world(&mut rng);
        let mut r = Reference::new(&units);
        let eps = if approximate { 0.05 } else { 0.0 };
        let mut d = scheduler(eps);
        let mut order: Vec<usize> = (0..units.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut ready: BTreeSet<Ann> = BTreeSet::new();
        let mut at = 0;
        while at < order.len() {
            let batch = &order[at..(at + 1 + rng.below(3)).min(order.len())];
            at += batch.len();
            d.declare(batch.iter().map(|&k| units[k].unit()), 0.0).unwrap();
            for &k in batch {
                r.declared[k] = true;
            }
            let (got, want) = (set(d.poll(0.0)), r.advance());
            prop_assert_eq!(&got, &want, "after declaring {:?}", batch);
            ready.extend(got.into_iter().filter(|o| !matches!(o, Ann::Passed(_))));
        }
        let mut steps = 0;
        // A unit's rank lags by less than `eps` of itself per level of units below it, and its
        // jobs' ranks by as much.
        let longest = r.ranks.iter().copied().fold(0.0, f64::max);
        let slack = ((1.0 + eps).powi(units.len() as i32) - 1.0) * longest + 1e-9;
        loop {
            let t = steps as f64;
            // Materialised or not.
            for (&job, &v) in r.by_id.iter().filter(|&(_, &v)| !r.done[v]) {
                let (got, want) = (d.rank(job).unwrap(), r.ranks[v]);
                prop_assert!(
                    got <= want + 1e-9 && got >= want - slack,
                    "job {}: rank {} vs {}", job, got, want
                );
            }
            if steps == snap_at {
                let json = serde_json::to_string(&d.snapshot()).unwrap();
                d = DagScheduler::restore(
                    serde_json::from_str(&json).unwrap(),
                    Scheduler::new(Config::default()),
                    Some(Arc::new(Works)),
                    t,
                );
                join(&mut d, t);
                prop_assert_eq!(&set(d.poll(t)), &ready, "restored announcements");
            }
            if steps == close_at {
                let k = rng.below(units.len());
                let u = &units[k];
                let closed = d.close(u.id, t);
                if closed.is_ok() {
                    prop_assert_eq!(closed.unwrap(), Vec::<JobId>::new());
                    r.close(k);
                    let end = u.base + u.template.leaves() as JobId;
                    ready.retain(|o| !(u.base..end).contains(&o.job()));
                    let (got, want) = (set(d.poll(t)), r.advance());
                    prop_assert_eq!(&got, &want, "after closing unit {}", u.id);
                    ready.extend(got.into_iter().filter(|o| !matches!(o, Ann::Passed(_))));
                }
            }
            let Some(&j) = ready.iter().nth(rng.below(ready.len().max(1))) else {
                break;
            };
            ready.remove(&j);
            r.done[r.by_id[&j.job()]] = true;
            let (got, want) = (run(&mut d, j, t), r.advance());
            prop_assert_eq!(&got, &want, "after running {:?}", j);
            ready.extend(got.into_iter().filter(|o| !matches!(o, Ann::Passed(_))));
            steps += 1;
        }
        // Everything ran or was closed.
        prop_assert!(r.done.iter().all(|&x| x), "the reference did not finish");
        let s = d.dag_stats();
        prop_assert_eq!((s.units, s.frames, s.pending, s.held, s.submitted), (0, 0, 0, 0, 0));
    }
}
