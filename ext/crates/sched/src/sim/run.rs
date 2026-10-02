//! The event-driven replay of a trace against a policy, and its metrics.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
};

use serde::Serialize;

use super::{model::ServiceModel, trace::Trace};
use crate::{
    Dag, DagConfig, DagJob, DagScheduler, JobSpec, Policy, PolicyStats, Resources, WorkerState,
};

/// How the replay derives each worker's `reported_baseline`.
#[derive(Clone, Debug)]
pub enum Baseline {
    /// Production's rule: the rolling RSS floor (see
    /// [`Trace::floor_baseline`](super::trace::Trace::floor_baseline)), replayed from the trace.
    RollingFloor {
        /// Window, seconds.
        window_s: f64,
        /// GB subtracted from the replayed floor. The trace samples RSS once a minute while the
        /// worker's gate samples it more often and so sees lower dips; this calibrates for that.
        offset_gb: f64,
    },
    /// A constant per worker, GB.
    PerWorker(Vec<f64>),
}

/// A policy usable from a simulation thread.
pub type BoxPolicy = Box<dyn Policy + Send>;

/// Inputs shared by all runs.
pub struct SimSetup<'a> {
    /// The trace.
    pub trace: &'a Trace,
    /// Each task's work (from [`fit`](super::model::fit)).
    pub work: &'a [f64],
    /// The service model.
    pub model: &'a dyn ServiceModel,
    /// How `reported_baseline` is derived.
    pub baseline: Baseline,
    /// Replay the trace's RSS samples as `reported_used` (otherwise report the baseline only).
    pub replay_rss: bool,
    /// Heartbeat period, seconds.
    pub heartbeat_s: f64,
    /// Closed-loop arrivals through a [`DagScheduler`] with this configuration.
    pub closed_loop: Option<DagConfig>,
    /// Jobs whose estimate exceeds this many GB are "big" in the metrics.
    pub big_gb: f64,
    /// Print the policy's `explain` for this request to stderr every 10 simulated minutes while
    /// it waits.
    pub explain: Option<u64>,
}

/// Distribution summary.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Quantiles {
    /// Sample size.
    pub n: usize,
    /// Mean.
    pub mean: f64,
    /// Median.
    pub p50: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Maximum.
    pub max: f64,
}

impl Quantiles {
    /// Summarise `v` (nearest-rank quantiles).
    pub fn of(mut v: Vec<f64>) -> Self {
        if v.is_empty() {
            return Self::default();
        }
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        Self {
            n: v.len(),
            mean: v.iter().sum::<f64>() / v.len() as f64,
            p50: q(0.5),
            p90: q(0.9),
            p99: q(0.99),
            max: v[v.len() - 1],
        }
    }
}

/// Per-class results.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ClassMetrics {
    /// Jobs completed on this class.
    pub jobs: usize,
    /// Work completed on this class.
    pub work: f64,
    /// Work per hour of makespan.
    pub work_per_h: f64,
    /// Busy slot-time over available slot-time.
    pub slot_util: f64,
    /// Placed estimate-GB-time over budget-GB-time.
    pub mem_util: f64,
}

/// One run's results.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Metrics {
    /// Policy name.
    pub policy: String,
    /// `"open"`, `"closed"` or `"production"`.
    pub arrivals: String,
    /// Jobs in the trace.
    pub jobs: usize,
    /// Jobs completed.
    pub completed: usize,
    /// Last completion minus first arrival, hours.
    pub makespan_h: f64,
    /// Total work completed.
    pub work: f64,
    /// Work per hour of makespan.
    pub work_per_h: f64,
    /// Per worker class.
    pub per_class: BTreeMap<String, ClassMetrics>,
    /// Busy slot-time over available slot-time.
    pub slot_util: f64,
    /// Placed estimate-GB-time over budget-GB-time.
    pub mem_util: f64,
    /// Wait (placement minus arrival), seconds, all jobs.
    pub wait: Quantiles,
    /// Wait for big jobs.
    pub wait_big: Quantiles,
    /// Group latency (last completion minus first arrival of the group), seconds.
    pub group_latency: Quantiles,
    /// Reservations made.
    pub reservations: u64,
    /// Free slot-hours on reserved workers while they were reserved (capacity idled draining).
    pub reserved_idle_slot_h: f64,
    /// That, as a fraction of all available slot-time.
    pub reserved_idle_frac: f64,
    /// Wall time of `dispatch` calls, microseconds.
    pub dispatch_us: Quantiles,
    /// The longest waits: `(req, est_gb, bidegree, arrival_s, wait_s)`.
    pub worst: Vec<(u64, f64, (i64, i64), f64, f64)>,
}

#[derive(Clone, Copy, Debug)]
enum Ev {
    Join(usize),
    Heartbeat(usize),
    Arrive(usize),
    Done(usize, u64),
}

struct Item {
    t: f64,
    seq: u64,
    ev: Ev,
}

impl PartialEq for Item {
    /// Equal when [`Ord`] says so.
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

impl Eq for Item {}

impl PartialOrd for Item {
    /// Total, from [`Ord`].
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Item {
    /// Reversed: `BinaryHeap` is a max-heap and we want the earliest event.
    fn cmp(&self, o: &Self) -> Ordering {
        o.t.total_cmp(&self.t).then(o.seq.cmp(&self.seq))
    }
}

#[derive(Default)]
struct WSim {
    running: Vec<(usize, f64)>,
    last: f64,
    version: u64,
    joined: bool,
    sample: usize,
    busy: f64,
    mem: f64,
}

enum Driver {
    Open(BoxPolicy),
    Closed(Box<DagScheduler<BoxPolicy>>),
}

impl Driver {
    /// Forwarded to the policy or the DAG layer.
    fn worker_update(&mut self, w: WorkerState, t: f64) {
        match self {
            Self::Open(p) => p.worker_update(w, t),
            Self::Closed(d) => d.worker_update(w, t),
        }
    }

    /// Forwarded to the policy or the DAG layer.
    fn completed(&mut self, j: usize, t: f64) {
        match self {
            Self::Open(p) => p.completed(j as u64, t),
            Self::Closed(d) => d.completed(j as u64, t),
        }
    }

    /// Forwarded to the policy or the DAG layer.
    fn dispatch(&mut self, t: f64) -> Vec<(u64, u64)> {
        match self {
            Self::Open(p) => p.dispatch(t),
            Self::Closed(d) => d.dispatch(t),
        }
    }

    /// Forwarded to the policy or the DAG layer.
    fn stats(&self) -> PolicyStats {
        match self {
            Self::Open(p) => p.stats(),
            Self::Closed(d) => d.stats(),
        }
    }

    /// Forwarded to the policy or the DAG layer.
    fn explain(&self, j: usize) -> Option<String> {
        match self {
            Self::Open(p) => p.explain(j as u64),
            Self::Closed(d) => d.explain(j as u64),
        }
    }
}

/// The job submitted for task `j`: its estimate as demand, its bidegree as group.
fn spec(trace: &Trace, j: usize) -> JobSpec {
    let t = &trace.tasks[j];
    JobSpec::new(j as u64, Resources::mem_gb(t.est_gb), t.group)
}

/// The closed-loop dependency lists (as task indices) and gaps.
fn closed_loop_inputs(trace: &Trace) -> (Vec<Vec<usize>>, Vec<f64>) {
    let idx: HashMap<u64, usize> = trace
        .tasks
        .iter()
        .enumerate()
        .map(|(i, t)| (t.req, i))
        .collect();
    let mut by_group: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, t) in trace.tasks.iter().enumerate() {
        by_group.entry(t.group).or_default().push(i);
    }
    // Sinks of a group: its tasks no other task of the group depends on.
    let mut has_dependent = vec![false; trace.tasks.len()];
    for t in &trace.tasks {
        for d in &t.deps {
            if let Some(&di) = idx.get(d)
                && trace.tasks[di].group == t.group
            {
                has_dependent[di] = true;
            }
        }
    }
    let deps: Vec<Vec<usize>> = trace
        .tasks
        .iter()
        .map(|t| {
            let mut v: Vec<usize> = t.deps.iter().filter_map(|d| idx.get(d).copied()).collect();
            for g in &t.after_groups {
                if let Some(members) = by_group.get(g) {
                    v.extend(members.iter().copied().filter(|&m| !has_dependent[m]));
                }
            }
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();
    let gaps = trace
        .tasks
        .iter()
        .zip(&deps)
        .map(|(t, d)| {
            let last = d.iter().map(|&i| trace.tasks[i].done_s).fold(0.0, f64::max);
            (t.ready_s - last).max(0.0)
        })
        .collect();
    (deps, gaps)
}

/// Replay the trace against `policy`.
///
/// Workers join at their trace join time and stay until the end. Every `heartbeat_s` each worker
/// reports `reported_used` = the trace's resident-memory sample at that time and
/// `reported_baseline` per [`Baseline`]; both are exogenous (replayed from the trace, not
/// responsive to the simulated placements). Jobs run under the
/// processor-sharing [`ServiceModel`]. The policy's `dispatch` is called after every event.
///
/// Arrivals are either **open-loop** (each job arrives at its trace `ready_s`) or **closed-loop**:
/// all jobs are declared to a [`DagScheduler`] up front with the trace's dependencies, and a job
/// arrives a fixed gap after its last dependency completes *in the simulation*, the gap being the
/// one measured in the trace (`ready_s - max(dep done_s)`, at least 0). Zero tasks' group
/// dependencies (`after_groups`) become dependencies on the sinks of those groups.
pub fn simulate(setup: &SimSetup, name: &str, policy: BoxPolicy) -> Metrics {
    let trace = setup.trace;
    let n = trace.tasks.len();
    let nw = trace.workers.len();
    let mut heap = BinaryHeap::new();
    let mut seq = 0u64;
    let mut push = |heap: &mut BinaryHeap<Item>, t: f64, ev: Ev| {
        seq += 1;
        heap.push(Item { t, seq, ev });
    };
    for (w, tw) in trace.workers.iter().enumerate() {
        push(&mut heap, tw.join_s, Ev::Join(w));
    }
    let (mut driver, gaps) = match &setup.closed_loop {
        None => {
            for (j, t) in trace.tasks.iter().enumerate() {
                push(&mut heap, t.ready_s, Ev::Arrive(j));
            }
            (Driver::Open(policy), Vec::new())
        }
        Some(cfg) => {
            let (deps, gaps) = closed_loop_inputs(trace);
            let mut dag = DagScheduler::new(
                DagConfig {
                    auto_submit: false,
                    ..cfg.clone()
                },
                policy,
            );
            let jobs = deps
                .into_iter()
                .enumerate()
                .map(|(j, d)| DagJob {
                    spec: spec(trace, j),
                    deps: d.into_iter().map(|x| x as u64).collect(),
                    work_estimate: Some(setup.work[j]),
                })
                .collect();
            dag.declare(jobs, 0.0)
                .expect("the trace's dependencies are acyclic");
            for j in dag.take_ready() {
                push(&mut heap, gaps[j as usize], Ev::Arrive(j as usize));
            }
            (Driver::Closed(Box::new(dag)), gaps)
        }
    };

    let mut ws: Vec<WSim> = (0..nw).map(|_| WSim::default()).collect();
    let mut arrival = vec![f64::NAN; n];
    let mut placed = vec![f64::NAN; n];
    let mut done = vec![f64::NAN; n];
    let mut ran_on: Vec<Option<usize>> = vec![None; n];
    let mut completed = 0usize;
    let mut reserved: Vec<usize> = Vec::new();
    let mut reserved_idle = 0.0;
    let mut prev_t = 0.0;
    let mut dispatch_us = Vec::new();
    let horizon = trace.tasks.iter().map(|t| t.done_s).fold(0.0, f64::max) * 50.0 + 1e6;
    let watched = setup
        .explain
        .and_then(|r| trace.tasks.iter().position(|t| t.req == r));
    let mut last_explain = f64::NEG_INFINITY;
    let rate = |w: usize, k: usize| setup.model.throughput(&trace.workers[w].class, k) / k as f64;

    let advance = |s: &mut WSim, w: usize, t: f64| {
        let dt = t - s.last;
        let k = s.running.len();
        if dt > 0.0 && k > 0 {
            let r = rate(w, k);
            for x in &mut s.running {
                x.1 -= r * dt;
            }
            s.busy += k as f64 * dt;
            s.mem += s
                .running
                .iter()
                .map(|x| trace.tasks[x.0].est_gb)
                .sum::<f64>()
                * dt;
        }
        s.last = t;
    };
    let state = |w: usize, s: &mut WSim, t: f64| -> WorkerState {
        let tw = &trace.workers[w];
        while s.sample + 1 < tw.samples.len() && tw.samples[s.sample + 1].t_s <= t {
            s.sample += 1;
        }
        let baseline = match &setup.baseline {
            Baseline::RollingFloor {
                window_s,
                offset_gb,
            } => (trace.floor_baseline(w, t, *window_s).unwrap_or(0.0) - offset_gb).max(0.0),
            Baseline::PerWorker(v) => v[w],
        };
        let rss = match tw.samples.get(s.sample) {
            Some(x) if setup.replay_rss && x.t_s <= t => x.rss_gb,
            _ => baseline,
        };
        WorkerState {
            reported_used: Resources::mem_gb(rss),
            reported_baseline: Resources::mem_gb(baseline),
            ..WorkerState::new(
                w as u64,
                tw.class.clone(),
                tw.slots,
                Resources::mem_gb(tw.budget_gb),
            )
        }
    };

    while let Some(Item { t, ev, .. }) = heap.pop() {
        for &w in &reserved {
            let free = trace.workers[w].slots.saturating_sub(ws[w].running.len());
            reserved_idle += free as f64 * (t - prev_t);
        }
        prev_t = t;
        let mut dirty: Vec<usize> = Vec::new();
        match ev {
            Ev::Join(w) => {
                ws[w].joined = true;
                ws[w].last = t;
                let st = state(w, &mut ws[w], t);
                driver.worker_update(st, t);
                push(&mut heap, t + setup.heartbeat_s, Ev::Heartbeat(w));
            }
            Ev::Heartbeat(w) => {
                let st = state(w, &mut ws[w], t);
                driver.worker_update(st, t);
                if completed < n && t < horizon {
                    push(&mut heap, t + setup.heartbeat_s, Ev::Heartbeat(w));
                }
            }
            Ev::Arrive(j) => {
                arrival[j] = t;
                match &mut driver {
                    Driver::Open(p) => p.submit(spec(trace, j), t),
                    Driver::Closed(d) => {
                        d.release(j as u64, t);
                    }
                }
            }
            Ev::Done(w, v) => {
                if v != ws[w].version {
                    continue;
                }
                advance(&mut ws[w], w, t);
                let mut finished = Vec::new();
                ws[w].running.retain(|&(j, rem)| {
                    let fin = rem <= 1e-7;
                    if fin {
                        finished.push(j);
                    }
                    !fin
                });
                finished.sort_unstable();
                for j in finished {
                    done[j] = t;
                    completed += 1;
                    driver.completed(j, t);
                }
                if let Driver::Closed(d) = &mut driver {
                    for r in d.take_ready() {
                        push(&mut heap, t + gaps[r as usize], Ev::Arrive(r as usize));
                    }
                }
                dirty.push(w);
            }
        }
        let clock = std::time::Instant::now();
        let out = driver.dispatch(t);
        dispatch_us.push(clock.elapsed().as_secs_f64() * 1e6);
        for (j, w) in out {
            let (j, w) = (j as usize, w as usize);
            advance(&mut ws[w], w, t);
            ws[w].running.push((j, setup.work[j]));
            placed[j] = t;
            ran_on[j] = Some(w);
            dirty.push(w);
        }
        dirty.sort_unstable();
        dirty.dedup();
        for w in dirty {
            let s = &mut ws[w];
            s.version += 1;
            if !s.running.is_empty() {
                let r = rate(w, s.running.len());
                let min = s
                    .running
                    .iter()
                    .map(|x| x.1)
                    .fold(f64::INFINITY, f64::min)
                    .max(0.0);
                let v = s.version;
                push(&mut heap, t + min / r, Ev::Done(w, v));
            }
        }
        reserved = driver
            .stats()
            .reservations
            .iter()
            .map(|r| r.worker as usize)
            .collect();
        if let Some(j) = watched
            && arrival[j].is_finite()
            && !placed[j].is_finite()
            && t - last_explain >= 600.0
        {
            last_explain = t;
            eprintln!(
                "[{name} t={t:.0}] {}",
                driver.explain(j).unwrap_or_default()
            );
        }
    }

    let stats = driver.stats();
    let jobs_done: Vec<usize> = (0..n).filter(|&j| done[j].is_finite()).collect();
    let mut m = summarize(
        setup,
        name,
        if setup.closed_loop.is_some() {
            "closed"
        } else {
            "open"
        },
        &jobs_done,
        &arrival,
        &placed,
        &done,
        &|w| ws[w].busy,
        &|w| ws[w].mem,
    );
    for (j, w) in ran_on.iter_mut().enumerate() {
        if !done[j].is_finite() {
            *w = None;
        }
    }
    attribute_work(setup, &mut m, &ran_on);
    m.reservations = stats.reservations_total;
    m.reserved_idle_slot_h = reserved_idle / 3600.0;
    let slot_time = slot_time(setup.trace, &done);
    m.reserved_idle_frac = reserved_idle / slot_time.max(1e-9);
    m.dispatch_us = Quantiles::of(dispatch_us);
    m
}

/// Available slot-seconds: each worker's slots from its join to the last completion.
fn slot_time(trace: &Trace, done: &[f64]) -> f64 {
    let end = done
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(0.0, f64::max);
    trace
        .workers
        .iter()
        .map(|w| w.slots as f64 * (end - w.join_s).max(0.0))
        .sum()
}

/// Metrics from per-job arrival, placement and completion times and per-worker integrals.
#[allow(clippy::too_many_arguments)]
fn summarize(
    setup: &SimSetup,
    policy: &str,
    arrivals: &str,
    jobs_done: &[usize],
    arrival: &[f64],
    placed: &[f64],
    done: &[f64],
    busy: &dyn Fn(usize) -> f64,
    mem: &dyn Fn(usize) -> f64,
) -> Metrics {
    let trace = setup.trace;
    let start = arrival
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::INFINITY, f64::min);
    let end = jobs_done.iter().map(|&j| done[j]).fold(0.0, f64::max);
    let makespan_h = ((end - start) / 3600.0).max(1e-9);
    let mut per_class: BTreeMap<String, ClassMetrics> = BTreeMap::new();
    let mut avail: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for (w, tw) in trace.workers.iter().enumerate() {
        let span = (end - tw.join_s).max(0.0);
        let a = avail.entry(tw.class.clone()).or_default();
        a.0 += tw.slots as f64 * span;
        a.1 += tw.budget_gb * span;
        let c = per_class.entry(tw.class.clone()).or_default();
        c.slot_util += busy(w);
        c.mem_util += mem(w);
    }
    let mut total_work = 0.0;
    for &j in jobs_done {
        total_work += setup.work[j];
    }
    let (mut busy_all, mut mem_all, mut slot_all, mut budget_all) = (0.0, 0.0, 0.0, 0.0);
    for (class, c) in per_class.iter_mut() {
        let (s, b) = avail[class];
        busy_all += c.slot_util;
        mem_all += c.mem_util;
        slot_all += s;
        budget_all += b;
        c.slot_util /= s.max(1e-9);
        c.mem_util /= b.max(1e-9);
    }
    let wait: Vec<f64> = jobs_done.iter().map(|&j| placed[j] - arrival[j]).collect();
    let mut order: Vec<usize> = (0..jobs_done.len()).collect();
    order.sort_by(|&a, &b| wait[b].total_cmp(&wait[a]));
    let worst = order
        .iter()
        .take(5)
        .map(|&i| {
            let t = &trace.tasks[jobs_done[i]];
            (t.req, t.est_gb, t.bidegree, arrival[jobs_done[i]], wait[i])
        })
        .collect();
    let wait_big: Vec<f64> = jobs_done
        .iter()
        .filter(|&&j| trace.tasks[j].est_gb > setup.big_gb)
        .map(|&j| placed[j] - arrival[j])
        .collect();
    let mut groups: HashMap<u64, (f64, f64)> = HashMap::new();
    let done_set: HashSet<usize> = jobs_done.iter().copied().collect();
    for (j, t) in trace.tasks.iter().enumerate() {
        let g = groups
            .entry(t.group)
            .or_insert((f64::INFINITY, f64::NEG_INFINITY));
        if arrival[j].is_finite() {
            g.0 = g.0.min(arrival[j]);
        }
        g.1 = g.1.max(if done_set.contains(&j) {
            done[j]
        } else {
            f64::INFINITY
        });
    }
    let group_latency = groups
        .values()
        .filter(|g| g.1.is_finite() && g.0.is_finite())
        .map(|g| g.1 - g.0)
        .collect();
    Metrics {
        policy: policy.to_string(),
        arrivals: arrivals.to_string(),
        jobs: trace.tasks.len(),
        completed: jobs_done.len(),
        makespan_h,
        work: total_work,
        work_per_h: total_work / makespan_h,
        per_class,
        slot_util: busy_all / slot_all.max(1e-9),
        mem_util: mem_all / budget_all.max(1e-9),
        wait: Quantiles::of(wait),
        wait_big: Quantiles::of(wait_big),
        group_latency: Quantiles::of(group_latency),
        worst,
        ..Metrics::default()
    }
}

/// Fill per-class work figures, given the worker each job ran on.
pub fn attribute_work(setup: &SimSetup, m: &mut Metrics, worker_of: &[Option<usize>]) {
    for c in m.per_class.values_mut() {
        c.jobs = 0;
        c.work = 0.0;
    }
    for (j, w) in worker_of.iter().enumerate() {
        if let Some(w) = w {
            let c = m.per_class.get_mut(&setup.trace.workers[*w].class).unwrap();
            c.jobs += 1;
            c.work += setup.work[j];
        }
    }
    for c in m.per_class.values_mut() {
        c.work_per_h = c.work / m.makespan_h;
    }
}

/// The metrics of what production actually did (from the trace's placements).
pub fn production(setup: &SimSetup) -> Metrics {
    let trace = setup.trace;
    let n = trace.tasks.len();
    let arrival: Vec<f64> = trace.tasks.iter().map(|t| t.ready_s).collect();
    let placed: Vec<f64> = trace.tasks.iter().map(|t| t.placed_s).collect();
    let done: Vec<f64> = trace.tasks.iter().map(|t| t.done_s).collect();
    let mut busy = vec![0.0; trace.workers.len()];
    let mut mem = vec![0.0; trace.workers.len()];
    for t in &trace.tasks {
        busy[t.worker] += t.done_s - t.placed_s;
        mem[t.worker] += t.est_gb * (t.done_s - t.placed_s);
    }
    let all: Vec<usize> = (0..n).collect();
    let mut m = summarize(
        setup,
        "production",
        "production",
        &all,
        &arrival,
        &placed,
        &done,
        &|w| busy[w],
        &|w| mem[w],
    );
    let worker_of: Vec<Option<usize>> = trace.tasks.iter().map(|t| Some(t.worker)).collect();
    attribute_work(setup, &mut m, &worker_of);
    m
}
