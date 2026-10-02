//! Small flat scheduling instances, their simulation, generators and perturbations (for PISA).

use std::{cmp::Ordering, collections::BinaryHeap};

use serde::Serialize;

use super::whole::{mix, normal, uniform};
use crate::{
    Config, Dag, DagConfig, DagJob, DagScheduler, GroupOrder, JobSpec, Policy, Resources,
    Scheduler, SpeedConfig, WorkerState,
};

/// What a task is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Kind {
    /// A group's first task.
    Zero,
    /// A task of a group's walk.
    Sig,
    /// A zero-work join ("group registered"): completes when its dependencies have.
    Join,
}

/// One task. Dependencies always name lower indices, so every instance is acyclic.
#[derive(Clone, Debug, Serialize)]
pub struct SmallTask {
    /// Its group (priority label).
    pub group: u32,
    /// The group's row and column in the coarse grid (for perturbations and reports).
    pub row: u32,
    /// See `row`.
    pub col: u32,
    /// What it is.
    pub kind: Kind,
    /// True work (seconds at speed 1).
    pub work: f64,
    /// The estimate ranks and placement see.
    pub est: f64,
    /// Lower-indexed tasks it depends on.
    pub deps: Vec<u32>,
}

/// A worker class.
#[derive(Clone, Debug, Serialize)]
pub struct Class {
    /// Its name.
    pub name: String,
    /// Speed of each worker.
    pub speed: f64,
    /// Number of workers.
    pub workers: u32,
    /// Slots per worker.
    pub slots: u32,
}

/// An instance: tasks and a fleet.
#[derive(Clone, Debug, Serialize)]
pub struct SmallInstance {
    /// Tasks, in topological order.
    pub tasks: Vec<SmallTask>,
    /// The fleet.
    pub classes: Vec<Class>,
}

/// How ready jobs are ordered.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum Order {
    /// Oldest group first, FIFO within.
    Group,
    /// Upward rank (estimated, or true costs when `oracle`).
    Rank {
        /// Rank on true costs.
        oracle: bool,
    },
    /// Oldest group first, rank within.
    GroupRank {
        /// Rank on true costs.
        oracle: bool,
    },
}

/// A dispatch plan for small instances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmallPlan {
    /// Job order.
    pub order: Order,
    /// `Config::age_limit`.
    pub age_limit: Option<f64>,
    /// Speed-aware placement.
    pub speed: SpeedConfig,
}

/// One simulation's outcome and features.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct SmallResult {
    /// Last completion.
    pub makespan: f64,
    /// Critical path with every task at the fastest speed.
    pub d_fast: f64,
    /// Total work over total throughput.
    pub w_over_p: f64,
    /// Fraction of the makespan with ready jobs and no free slot (when order matters).
    pub contention: f64,
    /// Fraction of the makespan with ready jobs and a free slot (voluntary idling).
    pub idle_ready: f64,
    /// Fraction of the makespan spent running realised-critical-chain tasks on slow workers.
    pub slow_on_crit: f64,
    /// Jobs restarted on a faster worker.
    pub preemptions: u64,
}

impl SmallInstance {
    /// Total work over total throughput, and the critical path at the fastest speed.
    pub fn bounds(&self) -> (f64, f64) {
        let fast = self.classes.iter().map(|c| c.speed).fold(0.0, f64::max);
        let cap: f64 = self
            .classes
            .iter()
            .map(|c| c.speed * (c.workers * c.slots) as f64)
            .sum();
        let mut fin = vec![0.0f64; self.tasks.len()];
        for (i, t) in self.tasks.iter().enumerate() {
            let start = t.deps.iter().map(|&d| fin[d as usize]).fold(0.0, f64::max);
            fin[i] = start
                + if t.kind == Kind::Join {
                    0.0
                } else {
                    t.work / fast
                };
        }
        let work: f64 = self
            .tasks
            .iter()
            .filter(|t| t.kind != Kind::Join)
            .map(|t| t.work)
            .sum();
        (
            work / cap.max(1e-12),
            fin.iter().copied().fold(0.0, f64::max),
        )
    }

    /// Number of non-join tasks.
    pub fn jobs(&self) -> usize {
        self.tasks.iter().filter(|t| t.kind != Kind::Join).count()
    }
}

/// A completion event (earliest first).
#[derive(PartialEq)]
struct Ev(f64, u64, Option<(usize, u32)>);

impl Eq for Ev {}

impl PartialOrd for Ev {
    /// The total order of [`Ord`].
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Ev {
    /// Reversed for a min-heap; the sequence number breaks ties.
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// Simulate `inst` under `plan` through the real [`DagScheduler`] and policy engine. Workers run
/// each job at their speed (exclusive slots, i.e. linear processor sharing).
pub fn simulate_small(inst: &SmallInstance, plan: &SmallPlan) -> SmallResult {
    let (rank, oracle, group_first) = match plan.order {
        Order::Group => (false, false, false),
        Order::Rank { oracle } => (true, oracle, false),
        Order::GroupRank { oracle } => (true, oracle, true),
    };
    let policy = Scheduler::new(Config {
        order: crate::Order::Priority {
            default_priority: 0,
            group_order: GroupOrder::Arrival,
            group_first,
        },
        age_limit: plan.age_limit,
        speed: plan.speed,
        ..Config::default()
    });
    let mut dag = DagScheduler::new(
        DagConfig {
            rank_priority: rank,
            rank_scale: 1e6,
            default_work: 0.0,
            rank_epsilon: 0.0,
            ..DagConfig::default()
        },
        policy,
    );
    let mut speed = Vec::new();
    let mut slow = Vec::new();
    let fastest = inst.classes.iter().map(|c| c.speed).fold(0.0, f64::max);
    for c in &inst.classes {
        for _ in 0..c.workers {
            let id = speed.len() as u64;
            dag.worker_update(
                WorkerState {
                    speed: c.speed,
                    ..WorkerState::new(
                        id,
                        c.name.clone(),
                        c.slots as usize,
                        Resources::mem(1 << 60),
                    )
                },
                0.0,
            );
            speed.push(c.speed);
            slow.push(c.speed < fastest);
        }
    }
    let slots_total: usize = inst
        .classes
        .iter()
        .map(|c| (c.workers * c.slots) as usize)
        .sum();
    let jobs = inst
        .tasks
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let deps = t.deps.iter().map(|&d| d as u64).collect();
            if t.kind == Kind::Join {
                DagJob::passthrough(i as u64, t.group as u64, deps, 0.0)
            } else {
                DagJob::new(
                    JobSpec::new(i as u64, Resources::ZERO, t.group as u64),
                    deps,
                )
                .with_work(if oracle { t.work } else { t.est })
            }
        })
        .collect();
    dag.declare(jobs, 0.0).expect("small instances are acyclic");

    let n = inst.tasks.len();
    let mut start = vec![f64::NAN; n];
    let mut finish = vec![f64::NAN; n];
    let mut on = vec![usize::MAX; n];
    let mut heap = BinaryHeap::new();
    let mut seq = 0u64;
    let mut running = 0usize;
    let (mut now, mut contention, mut idle_ready) = (0.0f64, 0.0f64, 0.0f64);
    let mut wake = f64::NAN;
    let mut epoch = vec![0u32; n];
    let mut preemptions = 0u64;
    loop {
        // Dispatch until nothing moves: a preemption frees a slot on a slower worker.
        loop {
            let d = dag.dispatch_full(now);
            for (j, w) in d.start {
                let (j, w) = (j as usize, w as usize);
                start[j] = now;
                on[j] = w;
                running += 1;
                seq += 1;
                heap.push(Ev(
                    now + inst.tasks[j].work / speed[w],
                    seq,
                    Some((j, epoch[j])),
                ));
            }
            if d.preempt.is_empty() {
                break;
            }
            for pr in d.preempt {
                // Kill on `from` (its pending completion goes stale), restart on `to`.
                let (j, w) = (pr.job as usize, pr.to as usize);
                epoch[j] += 1;
                start[j] = now;
                on[j] = w;
                preemptions += 1;
                seq += 1;
                heap.push(Ev(
                    now + inst.tasks[j].work / speed[w],
                    seq,
                    Some((j, epoch[j])),
                ));
            }
        }
        if let Some(t) = dag.policy().next_wakeup()
            && t != wake
        {
            wake = t;
            seq += 1;
            heap.push(Ev(t, seq, None));
        }
        let Some(Ev(t, _, ev)) = heap.pop() else {
            break;
        };
        let waiting = dag.stats().waiting;
        if waiting > 0 {
            if running >= slots_total {
                contention += t - now;
            } else {
                idle_ready += t - now;
            }
        }
        now = t;
        // Apply every event of this instant before dispatching again.
        let mut events = vec![ev];
        while heap.peek().is_some_and(|e| e.0 == t) {
            events.push(heap.pop().unwrap().2);
        }
        for (j, e) in events.into_iter().flatten() {
            if e != epoch[j] {
                continue; // a preempted instance
            }
            running -= 1;
            finish[j] = now;
            dag.completed(j as u64, now);
        }
    }
    // Joins finish with their last dependency.
    for (i, t) in inst.tasks.iter().enumerate() {
        if t.kind == Kind::Join {
            finish[i] = t
                .deps
                .iter()
                .map(|&d| finish[d as usize])
                .fold(0.0, f64::max);
        }
    }
    let makespan = finish.iter().copied().fold(0.0, f64::max);
    assert!(finish.iter().all(|f| f.is_finite()), "a task never ran");
    // The realised critical chain: back from the last task through its latest-finishing dep.
    let mut slow_time = 0.0;
    let mut cur = (0..n).max_by(|&a, &b| finish[a].total_cmp(&finish[b]));
    while let Some(i) = cur {
        if inst.tasks[i].kind != Kind::Join && slow[on[i]] {
            slow_time += finish[i] - start[i];
        }
        cur = inst.tasks[i]
            .deps
            .iter()
            .map(|&d| d as usize)
            .max_by(|&a, &b| finish[a].total_cmp(&finish[b]));
    }
    let (w_over_p, d_fast) = inst.bounds();
    let m = makespan.max(1e-12);
    SmallResult {
        makespan,
        d_fast,
        w_over_p,
        contention: contention / m,
        idle_ready: idle_ready / m,
        slow_on_crit: slow_time / m,
        preemptions,
    }
}

/// Parameters of a mini-Nassau grid instance.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct GridParams {
    /// Rows (homological degrees).
    pub rows: u32,
    /// Columns (stems).
    pub cols: u32,
    /// Same-row read-back distance (the zero-signature floor).
    pub floor: u32,
    /// Walk template: layers and width.
    pub depth: u32,
    /// See `depth`.
    pub width: u32,
    /// Fraction of groups with a walk.
    pub live: f64,
    /// Work growth per column, and decay per row.
    pub growth: f64,
    /// See `growth`.
    pub row_decay: f64,
    /// Per-task and per-group log-normal noise of true work around the estimate.
    pub sigma_task: f64,
    /// See `sigma_task`.
    pub sigma_group: f64,
    /// Fleet: fast class speed (slow is 1), workers per class, slots per worker.
    pub fast_speed: f64,
    /// See `fast_speed`.
    pub fast_workers: u32,
    /// See `fast_speed`.
    pub slow_workers: u32,
    /// See `fast_speed`.
    pub slots: u32,
    /// Seed of all random draws.
    pub seed: u64,
}

impl GridParams {
    /// Random parameters around our production regime.
    pub fn random(seed: u64) -> Self {
        let u = |k: u64| uniform(mix(seed) ^ k);
        let pick =
            |k: u64, lo: u32, hi: u32| lo + ((u(k) * (hi - lo + 1) as f64) as u32).min(hi - lo);
        GridParams {
            rows: pick(1, 2, 6),
            cols: pick(2, 6, 24),
            floor: pick(3, 1, 3),
            depth: pick(4, 2, 6),
            width: pick(5, 2, 8),
            live: 0.2 + 0.4 * u(6),
            growth: 1.0 + 0.3 * u(7),
            row_decay: 0.5 + 0.5 * u(8),
            sigma_task: 0.8 * u(9),
            sigma_group: 0.4 * u(10),
            fast_speed: 1.0 + 3.0 * u(11),
            fast_workers: pick(12, 1, 4),
            slow_workers: pick(13, 1, 4),
            slots: pick(14, 1, 8),
            seed,
        }
    }
}

/// Build a mini-Nassau instance: a grid of groups with `depgraph`'s edges, each group a zero
/// task, an optional layered walk, and a join.
pub fn grid(p: &GridParams) -> SmallInstance {
    let key = |a: u64, b: u64| mix(p.seed ^ mix(a ^ mix(b)));
    let mut tasks: Vec<SmallTask> = Vec::new();
    let (rows, cols) = (p.rows as usize, p.cols as usize);
    let mut join = vec![vec![u32::MAX; cols]; rows];
    for col in 0..cols {
        for row in 0..rows {
            let g = (col * rows + row) as u32;
            let base = p.growth.powi(col as i32) * p.row_decay.powi(row as i32);
            let gnoise = (p.sigma_group * normal(key(g as u64, 1))).exp();
            let mut deps = Vec::new();
            if col >= p.floor as usize {
                deps.push(join[row][col - p.floor as usize]);
            }
            if row > 0 && col > 0 {
                deps.push(join[row - 1][col - 1]);
            }
            let zero = tasks.len() as u32;
            let zw = 0.02 * base;
            tasks.push(SmallTask {
                group: g,
                row: row as u32,
                col: col as u32,
                kind: Kind::Zero,
                work: zw * gnoise,
                est: zw,
                deps,
            });
            let mut sinks = vec![zero];
            if uniform(key(g as u64, 2)) < p.live {
                let mut prev: Vec<u32> = vec![zero];
                for layer in 0..p.depth {
                    let share = 1.0 / ((layer + 1) as f64).powf(0.6) / p.width as f64;
                    let mut cur = Vec::new();
                    for k in 0..p.width {
                        let i = tasks.len() as u32;
                        let mut deps = vec![prev[(key(i as u64, 3) % prev.len() as u64) as usize]];
                        if prev.len() > 1 && uniform(key(i as u64, 4)) < 0.5 {
                            let d = prev[(key(i as u64, 5) % prev.len() as u64) as usize];
                            if !deps.contains(&d) {
                                deps.push(d);
                            }
                        }
                        let m = base * share;
                        tasks.push(SmallTask {
                            group: g,
                            row: row as u32,
                            col: col as u32,
                            kind: Kind::Sig,
                            work: m
                                * gnoise
                                * (p.sigma_task * normal(key(i as u64, 6 + k as u64))).exp(),
                            est: m,
                            deps,
                        });
                        cur.push(i);
                    }
                    prev = cur;
                }
                sinks = prev;
            }
            let mut deps = sinks;
            deps.push(zero);
            if col > 0 {
                deps.push(join[row][col - 1]);
            }
            deps.sort_unstable();
            deps.dedup();
            join[row][col] = tasks.len() as u32;
            tasks.push(SmallTask {
                group: g,
                row: row as u32,
                col: col as u32,
                kind: Kind::Join,
                work: 0.0,
                est: 0.0,
                deps,
            });
        }
    }
    SmallInstance {
        tasks,
        classes: vec![
            Class {
                name: "slow".into(),
                speed: 1.0,
                workers: p.slow_workers,
                slots: p.slots,
            },
            Class {
                name: "fast".into(),
                speed: p.fast_speed,
                workers: p.fast_workers,
                slots: p.slots,
            },
        ],
    }
}

/// A random multiplicative perturbation of `inst` (work, estimates, walk edges, fleet, or a
/// whole row or column), keyed by `key`. `structural` allows edge and fleet changes.
pub fn perturb(inst: &SmallInstance, key: u64, structural: bool) -> SmallInstance {
    let mut x = inst.clone();
    let u = |k: u64| uniform(mix(key) ^ k);
    let n = x.tasks.len();
    let pick_task = |k: u64, f: &dyn Fn(&SmallTask) -> bool, x: &SmallInstance| -> Option<usize> {
        let cands: Vec<usize> = (0..n).filter(|&i| f(&x.tasks[i])).collect();
        (!cands.is_empty()).then(|| cands[(u(k) * cands.len() as f64) as usize % cands.len()])
    };
    let factor = |k: u64| (u(k) - 0.5).mul_add(1.0, 0.0).exp();
    let op = (u(1) * if structural { 6.0 } else { 3.0 }) as u32;
    match op {
        0 => {
            if let Some(i) = pick_task(2, &|t| t.kind != Kind::Join, &x) {
                x.tasks[i].work = (x.tasks[i].work * factor(3)).max(1e-9);
            }
        }
        1 => {
            if let Some(i) = pick_task(2, &|t| t.kind != Kind::Join, &x) {
                x.tasks[i].est = (x.tasks[i].est * factor(3)).max(1e-9);
            }
        }
        2 => {
            // Scale a row or a column (moves the critical path between rows).
            let by_row = u(2) < 0.5;
            let maxv = x
                .tasks
                .iter()
                .map(|t| if by_row { t.row } else { t.col })
                .max()
                .unwrap_or(0);
            let v = (u(3) * (maxv + 1) as f64) as u32;
            let f = factor(4);
            for t in &mut x.tasks {
                if (if by_row { t.row } else { t.col }) == v {
                    t.work *= f;
                    t.est *= f;
                }
            }
        }
        3 => {
            // Add a walk edge between two signatures of one group.
            if let Some(j) = pick_task(2, &|t| t.kind == Kind::Sig, &x) {
                let g = x.tasks[j].group;
                let earlier: Vec<u32> = (0..j as u32)
                    .filter(|&i| {
                        x.tasks[i as usize].group == g && x.tasks[i as usize].kind == Kind::Sig
                    })
                    .collect();
                if !earlier.is_empty() {
                    let i = earlier[(u(3) * earlier.len() as f64) as usize % earlier.len()];
                    if !x.tasks[j].deps.contains(&i) {
                        x.tasks[j].deps.push(i);
                    }
                }
            }
        }
        4 => {
            // Remove a walk edge (a signature keeps at least one dependency).
            if let Some(j) = pick_task(2, &|t| t.kind == Kind::Sig && t.deps.len() > 1, &x) {
                let k = (u(3) * x.tasks[j].deps.len() as f64) as usize % x.tasks[j].deps.len();
                x.tasks[j].deps.remove(k);
            }
        }
        _ => {
            let c = (u(2) * x.classes.len() as f64) as usize % x.classes.len();
            match (u(3) * 3.0) as u32 {
                0 if x.classes[c].speed > 1.0 => {
                    x.classes[c].speed =
                        (x.classes[c].speed * (0.25 * (u(4) - 0.5)).exp()).clamp(1.0, 6.0)
                }
                1 => {
                    let w = x.classes[c].workers as i64 + if u(4) < 0.5 { -1 } else { 1 };
                    x.classes[c].workers = w.clamp(1, 8) as u32;
                }
                _ => {
                    let s = x.classes[c].slots as i64 + if u(4) < 0.5 { -1 } else { 1 };
                    x.classes[c].slots = s.clamp(1, 16) as u32;
                }
            }
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpeedPolicy;

    /// The driver respects the bounds, and a single slot equals total work.
    #[test]
    fn small_simulation_is_sound() {
        for seed in 0..40 {
            let inst = grid(&GridParams::random(seed));
            let (wp, d) = inst.bounds();
            for order in [
                Order::Group,
                Order::Rank { oracle: true },
                Order::GroupRank { oracle: false },
            ] {
                for policy in [SpeedPolicy::Oblivious, SpeedPolicy::FastestFirst] {
                    let plan = SmallPlan {
                        order,
                        age_limit: None,
                        speed: SpeedConfig {
                            policy,
                            learn: None,
                            spoliation: None,
                        },
                    };
                    let r = simulate_small(&inst, &plan);
                    assert!(r.makespan >= wp.max(d) * (1.0 - 1e-9), "seed {seed}: {r:?}");
                    // Graham: a greedy schedule on related machines is within W/P_slowest + D_slowest.
                    let slow = inst
                        .classes
                        .iter()
                        .map(|c| c.speed)
                        .fold(f64::INFINITY, f64::min);
                    let fast = inst.classes.iter().map(|c| c.speed).fold(0.0, f64::max);
                    assert!(
                        r.makespan <= (wp + d) * fast / slow * (1.0 + 1e-9),
                        "seed {seed}: {r:?}"
                    );
                }
            }
        }
        let mut one = grid(&GridParams::random(7));
        one.classes = vec![Class {
            name: "x".into(),
            speed: 2.0,
            workers: 1,
            slots: 1,
        }];
        let r = simulate_small(
            &one,
            &SmallPlan {
                order: Order::Group,
                age_limit: None,
                speed: SpeedConfig::default(),
            },
        );
        let work: f64 = one.tasks.iter().map(|t| t.work).sum();
        assert!((r.makespan - work / 2.0).abs() < 1e-9 * work);
    }

    /// Perturbations keep instances acyclic and well formed.
    #[test]
    fn perturbations_stay_valid() {
        let mut inst = grid(&GridParams::random(3));
        for k in 0..2000 {
            inst = perturb(&inst, k, true);
            for (i, t) in inst.tasks.iter().enumerate() {
                assert!(t.deps.iter().all(|&d| (d as usize) < i));
                assert!(t.work.is_finite() && t.work >= 0.0);
            }
        }
        simulate_small(
            &inst,
            &SmallPlan {
                order: Order::Group,
                age_limit: None,
                speed: SpeedConfig::default(),
            },
        );
    }
}
