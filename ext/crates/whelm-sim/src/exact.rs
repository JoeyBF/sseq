//! An exact branch-and-bound oracle for tiny instances.

use serde::Serialize;

use crate::{
    heft,
    small::{Kind, SmallInstance},
};

/// How long [`solve`] may search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// Search nodes at most.
    pub nodes: u64,
    /// Wall-clock seconds at most.
    pub seconds: f64,
}

impl Default for Limits {
    /// Limits that [`TINY_JOBS`](crate::small::TINY_JOBS)-sized instances usually stay far below.
    fn default() -> Self {
        Self {
            nodes: 10_000_000,
            seconds: 10.0,
        }
    }
}

/// What [`solve`] found.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Solution {
    /// The best makespan found.
    pub makespan: f64,
    /// Whether the search finished, so that `makespan` is optimal.
    ///
    /// Otherwise `makespan` is the best found within the [`Limits`].
    pub optimal: bool,
    /// A lower bound on the optimum: `makespan` when optimal, else the root's bound.
    pub lower_bound: f64,
    /// Search nodes visited.
    pub nodes: u64,
    /// Wall-clock seconds spent.
    pub seconds: f64,
}

/// The scheduling problem behind an instance: its jobs and its machines.
///
/// Joins are contracted into dependencies.
#[derive(Clone, Debug)]
struct Problem {
    /// Each job's true work.
    work: Vec<f64>,
    /// Each job's job predecessors, through chains of joins.
    preds: Vec<Vec<usize>>,
    /// Each job's work at the fastest speed plus the longest such chain below it.
    tail: Vec<f64>,
    /// Machine speeds.
    speeds: Vec<f64>,
    /// The fastest speed.
    fast: f64,
}

impl Problem {
    /// The jobs of `inst` on the machines of [`SmallInstance::machines`].
    fn new(inst: &SmallInstance) -> Self {
        let n = inst.tasks.len();
        // Each task's job index, and the job predecessors a successor of it inherits: itself if it
        // is a job, its own job predecessors if it is a join.
        let mut index = vec![usize::MAX; n];
        let mut through: Vec<Vec<usize>> = vec![Vec::new(); n];
        let (mut work, mut preds) = (Vec::new(), Vec::new());
        for (i, t) in inst.tasks.iter().enumerate() {
            let mut p: Vec<usize> = t
                .deps
                .iter()
                .flat_map(|&d| through[d as usize].iter().copied())
                .collect();
            p.sort_unstable();
            p.dedup();
            if t.kind == Kind::Join {
                through[i] = p;
            } else {
                index[i] = work.len();
                through[i] = vec![work.len()];
                work.push(t.work);
                preds.push(p);
            }
        }
        let speeds = inst.machines();
        let fast = speeds.iter().copied().fold(0.0, f64::max);
        let mut tail: Vec<f64> = work.iter().map(|w| w / fast).collect();
        // Job indices follow task indices, so reverse index order is reverse topological.
        for j in (0..work.len()).rev() {
            for &p in &preds[j] {
                tail[p] = tail[p].max(work[p] / fast + tail[j]);
            }
        }
        Self {
            work,
            preds,
            tail,
            speeds,
            fast,
        }
    }
}

/// The search's state: a partial schedule built in nondecreasing start order.
struct Search<'a> {
    pb: &'a Problem,
    limits: Limits,
    clock: std::time::Instant,
    /// When each machine is next free.
    free: Vec<f64>,
    /// Each job's finish, `NAN` while unscheduled.
    finish: Vec<f64>,
    /// Unscheduled predecessors of each job.
    unmet: Vec<u32>,
    /// Successors of each job.
    succs: Vec<Vec<usize>>,
    /// Work of the unscheduled jobs.
    remaining: f64,
    /// The incumbent makespan.
    best: f64,
    nodes: u64,
    aborted: bool,
    /// Scratch for [`bound`](Self::bound).
    est: Vec<f64>,
}

impl Search<'_> {
    /// A lower bound on every completion of a partial schedule.
    ///
    /// The partial schedule's last job started at `t0` (every job left starts no earlier, by the
    /// search's order) and its jobs finish by `cur_max`. The bound is the largest of:
    ///
    /// - per job: its earliest start (from its predecessors, at the fastest speed for unscheduled
    ///   ones), its earliest finish on any machine from there, and the rest of its chain at the
    ///   fastest speed;
    /// - load: the time by which the machines, from when each is free (and not before `t0`), can
    ///   have done the remaining work.
    fn bound(&mut self, t0: f64, cur_max: f64) -> f64 {
        let pb = self.pb;
        let mut lb = cur_max;
        for j in 0..pb.work.len() {
            if !self.finish[j].is_nan() {
                continue;
            }
            let mut e = t0;
            for &p in &pb.preds[j] {
                let f = if self.finish[p].is_nan() {
                    self.est[p] + pb.work[p] / pb.fast
                } else {
                    self.finish[p]
                };
                e = e.max(f);
            }
            self.est[j] = e;
            let first = (pb.speeds.iter().zip(&self.free))
                .map(|(s, &free)| e.max(free) + pb.work[j] / s)
                .fold(f64::INFINITY, f64::min);
            lb = lb.max(first + pb.tail[j] - pb.work[j] / pb.fast);
        }
        let mut avail: Vec<(f64, f64)> = (self.free.iter())
            .zip(&pb.speeds)
            .map(|(&f, &s)| (f.max(t0), s))
            .collect();
        avail.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Capacity grows piecewise linearly as machines become free.
        let (mut done, mut rate, mut t) = (0.0f64, 0.0f64, avail[0].0);
        for k in 0..avail.len() {
            rate += avail[k].1;
            let next = avail.get(k + 1).map_or(f64::INFINITY, |a| a.0);
            if done + rate * (next - t) >= self.remaining {
                t += (self.remaining - done) / rate;
                break;
            }
            done += rate * (next - t);
            t = next;
        }
        lb.max(t)
    }

    /// Extend the partial schedule in every canonical way that may beat the incumbent.
    ///
    /// It has `scheduled` jobs, the last being job `last.1`, started at `last.0`.
    fn dfs(&mut self, last: (f64, usize), cur_max: f64, scheduled: usize) {
        let pb = self.pb;
        let n = pb.work.len();
        if scheduled == n {
            self.best = self.best.min(cur_max);
            return;
        }
        self.nodes += 1;
        if self.nodes >= self.limits.nodes
            || (self.nodes.is_multiple_of(4096)
                && self.clock.elapsed().as_secs_f64() >= self.limits.seconds)
        {
            self.aborted = true;
        }
        if self.aborted || self.bound(last.0, cur_max) >= self.best * (1.0 - 1e-12) {
            return;
        }
        // (start, -tail, job, machine, finish)
        let mut children: Vec<(f64, f64, usize, usize, f64)> = Vec::new();
        for j in 0..n {
            if !self.finish[j].is_nan() || self.unmet[j] > 0 {
                continue;
            }
            let ready = (pb.preds[j].iter())
                .map(|&p| self.finish[p])
                .fold(0.0, f64::max);
            let mut tried: Vec<(f64, f64)> = Vec::new();
            for (m, &s) in pb.speeds.iter().enumerate() {
                // Machines of equal speed that are free at the same time are interchangeable.
                if tried.contains(&(s, self.free[m])) {
                    continue;
                }
                tried.push((s, self.free[m]));
                let start = ready.max(self.free[m]);
                // Canonical order: by start, then by job index.
                if start < last.0 || (start == last.0 && j < last.1) {
                    continue;
                }
                children.push((start, -pb.tail[j], j, m, start + pb.work[j] / s));
            }
        }
        children.sort_by(|a, b| {
            (a.0.total_cmp(&b.0))
                .then(a.1.total_cmp(&b.1))
                .then(a.4.total_cmp(&b.4))
        });
        for (start, _, j, m, finish) in children {
            let free = self.free[m];
            self.free[m] = finish;
            self.finish[j] = finish;
            self.remaining -= pb.work[j];
            for &s in &self.succs[j] {
                self.unmet[s] -= 1;
            }
            self.dfs((start, j), cur_max.max(finish), scheduled + 1);
            for &s in &self.succs[j] {
                self.unmet[s] += 1;
            }
            self.remaining += pb.work[j];
            self.finish[j] = f64::NAN;
            self.free[m] = free;
            if self.aborted {
                return;
            }
        }
    }
}

/// The optimal makespan of `inst` given its true work, or the best found within `limits`.
///
/// The problem is Q|prec|Cmax: each worker slot is a machine at its worker's speed, as
/// [`simulate_small`](crate::small::simulate_small) runs them, and joins take no time.
///
/// Depth-first branch and bound over semi-active schedules (every job starts as soon as its
/// predecessors and its machine allow), each generated once: jobs are appended in nondecreasing
/// start order, ties by job index, and of several machines with the same speed and the same free
/// time only one is tried. Left-shifting any schedule gives a semi-active one with no larger
/// makespan, and listing a semi-active schedule's jobs by start reproduces it, so the search is
/// complete. The incumbent starts as HEFT's schedule on true costs; a lower bound per node
/// prunes.
///
/// The number of semi-active schedules grows like `jobs! * machines^jobs` before pruning, so this
/// is for [`tiny`](crate::small::tiny) instances only.
pub fn solve(inst: &SmallInstance, limits: Limits) -> Solution {
    let clock = std::time::Instant::now();
    let pb = Problem::new(inst);
    let n = pb.work.len();
    if n == 0 {
        return Solution {
            makespan: 0.0,
            optimal: true,
            lower_bound: 0.0,
            nodes: 0,
            seconds: 0.0,
        };
    }
    let mut succs = vec![Vec::new(); n];
    for (j, p) in pb.preds.iter().enumerate() {
        for &q in p {
            succs[q].push(j);
        }
    }
    let mut s = Search {
        pb: &pb,
        limits,
        clock,
        free: vec![0.0; pb.speeds.len()],
        finish: vec![f64::NAN; n],
        unmet: pb.preds.iter().map(|p| p.len() as u32).collect(),
        succs,
        remaining: pb.work.iter().sum(),
        best: heft::heft(inst, true).makespan,
        nodes: 0,
        aborted: false,
        est: vec![0.0; n],
    };
    let root = s.bound(0.0, 0.0);
    s.dfs((f64::NEG_INFINITY, 0), 0.0, 0);
    let optimal = !s.aborted;
    Solution {
        makespan: s.best,
        optimal,
        lower_bound: if optimal { s.best } else { root.min(s.best) },
        nodes: s.nodes,
        seconds: clock.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plan::SpeedPlan,
        small::{Class, Order, SmallPlan, SmallTask, simulate_small, tiny},
        whole::{mix, uniform},
    };

    /// Every semi-active schedule, without pruning, symmetry or ordering rules.
    fn brute(pb: &Problem, free: &mut [f64], finish: &mut [f64]) -> f64 {
        let n = pb.work.len();
        let mut best = f64::INFINITY;
        let mut leaf = true;
        for j in 0..n {
            if !finish[j].is_nan() || pb.preds[j].iter().any(|&p| finish[p].is_nan()) {
                continue;
            }
            leaf = false;
            let ready = pb.preds[j].iter().map(|&p| finish[p]).fold(0.0, f64::max);
            for m in 0..pb.speeds.len() {
                let old = free[m];
                let f = ready.max(old) + pb.work[j] / pb.speeds[m];
                free[m] = f;
                finish[j] = f;
                best = best.min(brute(pb, free, finish));
                finish[j] = f64::NAN;
                free[m] = old;
            }
        }
        if leaf {
            finish.iter().copied().fold(0.0, f64::max)
        } else {
            best
        }
    }

    /// A random instance of `jobs` jobs with joins among them.
    ///
    /// The fleet is a slow worker of `slots` slots and a one-slot worker that is faster unless
    /// `identical`.
    fn random(seed: u64, jobs: usize, slots: u32, identical: bool) -> SmallInstance {
        let u = |k: u64| uniform(mix(seed) ^ k);
        let mut tasks: Vec<SmallTask> = Vec::new();
        let mut k = 0;
        while tasks.iter().filter(|t| t.kind != Kind::Join).count() < jobs {
            k += 1;
            let i = tasks.len();
            let join = i > 0 && u(k * 7) < 0.2;
            let deps: Vec<u32> = (0..i as u32)
                .filter(|&d| u(k * 7 + 1 + d as u64 * 13) < 0.3)
                .collect();
            let work = if join { 0.0 } else { 0.5 + 4.0 * u(k * 7 + 2) };
            tasks.push(SmallTask {
                group: 0,
                row: 0,
                col: 0,
                kind: if join { Kind::Join } else { Kind::Sig },
                work,
                est: work,
                deps,
            });
        }
        let class = |name: &str, speed, slots| Class {
            name: name.into(),
            speed,
            workers: 1,
            slots,
        };
        SmallInstance {
            tasks,
            classes: vec![
                class("slow", 1.0, slots),
                class("fast", if identical { 1.0 } else { 2.5 }, 1),
            ],
        }
    }

    /// Branch and bound agrees with exhaustive enumeration on very small instances.
    #[test]
    fn branch_and_bound_matches_brute_force() {
        for seed in 0..60 {
            let jobs = 3 + (seed % 4) as usize;
            let inst = random(seed, jobs, 1 + (seed % 2) as u32, seed % 5 == 0);
            let pb = Problem::new(&inst);
            let want = brute(
                &pb,
                &mut vec![0.0; pb.speeds.len()],
                &mut vec![f64::NAN; pb.work.len()],
            );
            let got = solve(&inst, Limits::default());
            assert!(got.optimal, "seed {seed}");
            assert!(
                (got.makespan - want).abs() <= 1e-9 * want,
                "seed {seed}: {} vs {want}",
                got.makespan
            );
        }
    }

    /// On tiny instances the optimum is proved, no bound exceeds it and no schedule beats it.
    ///
    /// The lower bounds stay below it; HEFT's offline schedule and every online plan's makespan
    /// stay above it.
    #[test]
    fn optimum_bounds_the_plans() {
        for seed in 0..12 {
            let inst = tiny(seed);
            let opt = solve(&inst, Limits::default());
            assert!(opt.optimal, "seed {seed}: {opt:?}");
            let (wp, d) = inst.bounds();
            assert!(opt.makespan >= wp.max(d) * (1.0 - 1e-9));
            assert!(opt.makespan <= heft::heft(&inst, true).makespan * (1.0 + 1e-12));
            for order in [Order::Group, Order::Rank { oracle: true }] {
                let plan = SmallPlan {
                    order,
                    age_limit: None,
                    speed: SpeedPlan::default(),
                };
                let r = simulate_small(&inst, &plan);
                assert!(r.makespan >= opt.makespan * (1.0 - 1e-9), "seed {seed}");
            }
        }
    }

    /// A node limit stops the search with the best schedule found and a valid lower bound.
    #[test]
    fn limits_report_best_found() {
        let (inst, full) = (0..)
            .map(|seed| {
                let inst = tiny(seed);
                let full = solve(&inst, Limits::default());
                (inst, full)
            })
            .find(|(_, full)| full.nodes > 50)
            .unwrap();
        let cut = solve(
            &inst,
            Limits {
                nodes: 5,
                seconds: 10.0,
            },
        );
        assert!(!cut.optimal);
        assert!(cut.makespan >= full.makespan * (1.0 - 1e-12));
        assert!(cut.lower_bound <= full.makespan * (1.0 + 1e-12));
    }
}
