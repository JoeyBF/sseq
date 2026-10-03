//! Small flat scheduling instances, their simulation, generators and perturbations (for PISA).

use std::{collections::HashMap, time::Duration};

use serde::Serialize;
use whelm::{
    Attempt, Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, OrderTerm, Output, Policy,
    Resources, Scheduler, Time, WorkerState,
};

use crate::{
    engine::Queue,
    heft,
    plan::SpeedPlan,
    whole::{mix, normal, uniform},
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
    /// An offline plan's order: a job's start in a [`heft`] schedule is its priority.
    ///
    /// The start becomes [`JobSpec::priority`]; the plan uses estimated costs, or true costs when
    /// `oracle`.
    Heft {
        /// Plan on true costs.
        oracle: bool,
    },
    /// Smith's rule ([`Config::weighted_completion`]): shortest estimated work first.
    ///
    /// All weights are 1.
    Wspt,
}

impl Order {
    /// The scheduler's order terms, and whether ranks and plans see true costs.
    fn terms(self) -> (Vec<OrderTerm>, bool) {
        use OrderTerm::{Group, Priority, Rank};
        match self {
            Self::Group => (vec![Group], false),
            Self::Rank { oracle } => (vec![Rank, Group], oracle),
            Self::GroupRank { oracle } => (vec![Group, Rank], oracle),
            Self::Heft { oracle } => (vec![Priority, Group], oracle),
            Self::Wspt => (Config::weighted_completion().order, false),
        }
    }
}

/// A dispatch plan for small instances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmallPlan {
    /// Job order.
    pub order: Order,
    /// `Config::age_limit`.
    pub age_limit: Option<Duration>,
    /// Speed-aware placement and the machine model.
    pub speed: SpeedPlan,
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
    /// Speculative attempts started on a faster worker ([`Speculate`](whelm::Speculate)).
    pub speculations: u64,
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

    /// The speed of every slot of every worker: the machines of the offline models.
    ///
    /// They match [`simulate_small`]'s exclusive slots.
    pub fn machines(&self) -> Vec<f64> {
        (self.classes.iter())
            .flat_map(|c| std::iter::repeat_n(c.speed, (c.workers * c.slots) as usize))
            .collect()
    }
}

/// Simulate `inst` under `plan` through the real [`DagScheduler`] and policy engine.
///
/// Workers run each job at their speed (exclusive slots, i.e. linear processor sharing). A
/// speculative attempt holds its own slot until the first attempt of its job finishes and the
/// other is stopped.
pub fn simulate_small(inst: &SmallInstance, plan: &SmallPlan) -> SmallResult {
    let (order, oracle) = plan.order.terms();
    let policy = Scheduler::new(Config {
        order,
        age_limit: plan.age_limit,
        score: plan.speed.score(),
        speed: plan.speed.config,
        ..Config::default()
    });
    let mut dag = DagScheduler::new(
        DagConfig {
            default_work: Duration::ZERO,
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
            let state = WorkerState {
                id,
                class: c.name.clone(),
                slots: c.slots as usize,
                budget: Resources::mem(1 << 60),
                speed: if plan.speed.learned() { 1.0 } else { c.speed },
                ..Default::default()
            };
            dag.handle(Input::Worker(state), Time::ZERO);
            speed.push(c.speed);
            slow.push(c.speed < fastest);
        }
    }
    let slots_total: usize = inst
        .classes
        .iter()
        .map(|c| (c.workers * c.slots) as usize)
        .sum();
    let priority = matches!(plan.order, Order::Heft { .. })
        .then(|| heft::priorities(&heft::heft(inst, oracle)));
    let jobs: Vec<DagJob> = inst
        .tasks
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let deps = t.deps.iter().map(|&d| d as u64).collect();
            if t.kind == Kind::Join {
                DagJob {
                    spec: JobSpec {
                        id: i as u64,
                        group: t.group as u64,
                        ..Default::default()
                    },
                    deps,
                    work_estimate: Some(Duration::ZERO),
                    passthrough: true,
                    ..Default::default()
                }
            } else {
                let kind = if t.kind == Kind::Zero { "zero" } else { "sig" };
                DagJob {
                    spec: JobSpec {
                        id: i as u64,
                        group: t.group as u64,
                        priority: priority.as_ref().map(|p| p[i]),
                        kind: Some(kind.into()),
                        ..Default::default()
                    },
                    deps,
                    work_estimate: Some(Duration::from_secs_f64(if oracle {
                        t.work
                    } else {
                        t.est
                    })),
                    ..Default::default()
                }
            }
        })
        .collect();
    dag.declare(jobs, Time::ZERO)
        .expect("small instances are acyclic");

    let n = inst.tasks.len();
    let mut start = vec![f64::NAN; n];
    let mut finish = vec![f64::NAN; n];
    let mut on = vec![usize::MAX; n];
    // Completion events: `Some((job, attempt))`, or `None` for a wakeup.
    let mut queue: Queue<Option<(usize, Attempt)>> = Queue::new();
    // Running attempts: (start, worker).
    let mut live: HashMap<(usize, Attempt), (f64, usize)> = HashMap::new();
    let (mut now, mut contention, mut idle_ready) = (0.0f64, 0.0f64, 0.0f64);
    let mut wake = None;
    let mut speculations = 0u64;
    loop {
        for o in dag.poll(Time::from_secs_f64(now)) {
            match o {
                Output::Start {
                    job,
                    attempt,
                    worker,
                } => {
                    let (j, w) = (job as usize, worker as usize);
                    live.insert((j, attempt), (now, w));
                    speculations += u64::from(attempt > 1);
                    queue.push(now + inst.tasks[j].work / speed[w], Some((j, attempt)));
                }
                // Its pending completion goes stale.
                Output::Stop { job, attempt, .. } => {
                    live.remove(&(job as usize, attempt));
                }
                Output::GaveUp(g) => unreachable!("job {} failed, but no attempt fails", g.job),
                Output::RunLocal { .. } | Output::Ready { .. } | Output::Passed { .. } => {}
            }
        }
        if let Some(t) = dag.next_wakeup()
            && Some(t) != wake
        {
            wake = Some(t);
            queue.push(t.as_secs_f64(), None);
        }
        let Some((t, events)) = queue.pop_instant() else {
            break;
        };
        if dag.stats().waiting > 0 {
            if live.len() >= slots_total {
                contention += t - now;
            } else {
                idle_ready += t - now;
            }
        }
        now = t;
        // Every event of this instant is applied before polling again.
        for (j, attempt) in events.into_iter().flatten() {
            let Some((s, w)) = live.remove(&(j, attempt)) else {
                continue; // a stopped attempt
            };
            // The other attempt finished at the same instant; its stop is on its way.
            if finish[j].is_finite() {
                continue;
            }
            start[j] = s;
            on[j] = w;
            finish[j] = now;
            dag.handle(
                Input::Done {
                    job: j as u64,
                    attempt,
                },
                Time::from_secs_f64(now),
            );
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
        speculations,
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

/// Build a mini-Nassau instance from `p`.
///
/// It is a grid of groups with `depgraph`'s edges, each group a zero task, an optional layered
/// walk, and a join.
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

/// A random multiplicative perturbation of `inst`, keyed by `key`.
///
/// It changes work, estimates, walk edges, the fleet, or a whole row or column; `structural`
/// allows edge and fleet changes.
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

/// Jobs in a [`tiny`] instance.
///
/// Few enough for [`exact::solve`](crate::exact::solve) to prove optimality quickly.
pub const TINY_JOBS: std::ops::RangeInclusive<usize> = 8..=20;

/// A tiny mini-Nassau instance of [`TINY_JOBS`] jobs, for comparing plans against the optimum.
///
/// A small grid with short narrow walks on a small two-class fleet. Parameters are drawn from
/// `seed`, redrawn deterministically until the job count fits.
pub fn tiny(seed: u64) -> SmallInstance {
    for attempt in 0u64.. {
        let key = mix(seed ^ mix(attempt));
        let u = |k: u64| uniform(key ^ k);
        let pick =
            |k: u64, lo: u32, hi: u32| lo + ((u(k) * (hi - lo + 1) as f64) as u32).min(hi - lo);
        let p = GridParams {
            rows: pick(1, 1, 2),
            cols: pick(2, 2, 4),
            floor: pick(3, 1, 2),
            depth: pick(4, 1, 2),
            width: pick(5, 1, 3),
            live: 0.5 + 0.5 * u(6),
            growth: 1.0 + 0.3 * u(7),
            row_decay: 0.5 + 0.5 * u(8),
            sigma_task: 0.8 * u(9),
            sigma_group: 0.4 * u(10),
            fast_speed: 1.0 + 3.0 * u(11),
            fast_workers: 1,
            slow_workers: pick(13, 1, 2),
            slots: pick(14, 1, 2),
            seed: key,
        };
        let inst = grid(&p);
        if TINY_JOBS.contains(&inst.jobs()) {
            return inst;
        }
    }
    unreachable!("the attempts are unbounded")
}

#[cfg(test)]
mod tests {
    use whelm::{SpeedConfig, Timing};

    use super::*;

    /// A plan with the given order and speed-awareness and nothing else.
    fn plan(order: Order, fast: bool) -> SmallPlan {
        SmallPlan {
            order,
            age_limit: None,
            speed: SpeedPlan {
                fast,
                config: SpeedConfig::default(),
            },
        }
    }

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
                Order::Heft { oracle: false },
                Order::Wspt,
            ] {
                for fast in [false, true] {
                    let r = simulate_small(&inst, &plan(order, fast));
                    assert!(r.makespan >= wp.max(d) * (1.0 - 1e-9), "seed {seed}: {r:?}");
                    // Graham: a greedy schedule on related machines is within
                    // W/P_slowest + D_slowest.
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
        let r = simulate_small(&one, &plan(Order::Group, false));
        let work: f64 = one.tasks.iter().map(|t| t.work).sum();
        assert!((r.makespan - work / 2.0).abs() < 1e-9 * work);
    }

    /// Speculation on a mixed fleet starts second attempts, and no schedule beats the bounds.
    ///
    /// Every job finishes once; the simulator asserts it.
    #[test]
    fn speculation_runs_second_attempts() {
        let mut p = plan(Order::Group, true);
        p.speed.config.speculate = Some(whelm::Speculate::default());
        let mut speculations = 0;
        for seed in 0..40 {
            let inst = grid(&GridParams::random(seed));
            let (wp, d) = inst.bounds();
            let r = simulate_small(&inst, &p);
            assert!(r.makespan >= wp.max(d) * (1.0 - 1e-9), "seed {seed}: {r:?}");
            speculations += r.speculations;
        }
        assert!(speculations > 0);
    }

    /// Every machine model runs every instance to completion within the bounds.
    #[test]
    fn every_timing_finishes() {
        for timing in [
            Timing::Identical,
            Timing::default(),
            Timing::learned(),
            Timing::unrelated(),
        ] {
            let mut p = plan(Order::Rank { oracle: false }, true);
            p.speed.config.timing = timing;
            for seed in 0..10 {
                let inst = grid(&GridParams::random(seed));
                let (wp, d) = inst.bounds();
                let r = simulate_small(&inst, &p);
                assert!(r.makespan >= wp.max(d) * (1.0 - 1e-9), "{timing:?}: {r:?}");
            }
        }
    }

    /// Tiny instances have the advertised size and stay valid.
    #[test]
    fn tiny_instances_fit() {
        for seed in 0..50 {
            let inst = tiny(seed);
            assert!(TINY_JOBS.contains(&inst.jobs()), "seed {seed}");
            for (i, t) in inst.tasks.iter().enumerate() {
                assert!(t.deps.iter().all(|&d| (d as usize) < i));
            }
        }
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
        simulate_small(&inst, &plan(Order::Group, false));
    }
}
