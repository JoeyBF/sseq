//! Offline HEFT planning on small instances.

use crate::small::{Kind, SmallInstance};

/// A static schedule: every task's planned start, finish and machine (an index into
/// [`SmallInstance::machines`]; `None` for joins, which take no time).
#[derive(Clone, Debug)]
pub struct Schedule {
    /// Planned start of each task.
    pub start: Vec<f64>,
    /// Planned finish of each task.
    pub finish: Vec<f64>,
    /// Machine of each task.
    pub machine: Vec<Option<usize>>,
    /// The last planned finish.
    pub makespan: f64,
}

/// Each task's upward rank at the machines' mean speed: its cost plus the longest chain of cost
/// below it, both over that speed. Joins cost nothing.
pub fn upward_ranks(inst: &SmallInstance, cost: &[f64]) -> Vec<f64> {
    let speeds = inst.machines();
    let mean = speeds.iter().sum::<f64>() / speeds.len().max(1) as f64;
    let n = inst.tasks.len();
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, t) in inst.tasks.iter().enumerate() {
        for &d in &t.deps {
            succs[d as usize].push(i);
        }
    }
    let mut rank = vec![0.0f64; n];
    // Dependencies name lower indices, so reverse index order is a reverse topological order.
    for i in (0..n).rev() {
        let below = succs[i].iter().map(|&s| rank[s]).fold(0.0, f64::max);
        rank[i] = cost[i] / mean.max(1e-12) + below;
    }
    rank
}

/// HEFT (Topcuoglu, Hariri and Wu): tasks in decreasing upward rank at the mean speed, each put on
/// the machine where it finishes earliest, inserted into the first idle gap long enough for it.
/// Costs are the estimates, or the true work when `oracle`; the schedule's times are in those
/// costs.
///
/// Every predecessor ranks at least as high as its successors and has a lower index, so breaking
/// rank ties by index keeps the order topological.
pub fn heft(inst: &SmallInstance, oracle: bool) -> Schedule {
    let n = inst.tasks.len();
    let cost: Vec<f64> = (inst.tasks.iter())
        .map(|t| match (t.kind, oracle) {
            (Kind::Join, _) => 0.0,
            (_, true) => t.work,
            (_, false) => t.est,
        })
        .collect();
    let rank = upward_ranks(inst, &cost);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| rank[b].total_cmp(&rank[a]).then(a.cmp(&b)));
    let speeds = inst.machines();
    // Each machine's busy intervals, sorted by start.
    let mut busy: Vec<Vec<(f64, f64)>> = vec![Vec::new(); speeds.len()];
    let mut start = vec![0.0f64; n];
    let mut finish = vec![0.0f64; n];
    let mut machine = vec![None; n];
    for i in order {
        let t = &inst.tasks[i];
        let ready = t
            .deps
            .iter()
            .map(|&d| finish[d as usize])
            .fold(0.0, f64::max);
        if t.kind == Kind::Join {
            (start[i], finish[i]) = (ready, ready);
            continue;
        }
        let mut best: Option<(f64, f64, usize, usize)> = None;
        for (m, &speed) in speeds.iter().enumerate() {
            let d = cost[i] / speed;
            let (s, slot) = earliest_gap(&busy[m], ready, d);
            if best.is_none_or(|b| s + d < b.1) {
                best = Some((s, s + d, m, slot));
            }
        }
        let (s, f, m, slot) = best.expect("an instance has at least one machine");
        busy[m].insert(slot, (s, f));
        (start[i], finish[i], machine[i]) = (s, f, Some(m));
    }
    let makespan = finish.iter().copied().fold(0.0, f64::max);
    Schedule {
        start,
        finish,
        machine,
        makespan,
    }
}

/// The earliest start at or after `ready` of a task of duration `d` among `busy` intervals, and
/// the position to insert it at.
fn earliest_gap(busy: &[(f64, f64)], ready: f64, d: f64) -> (f64, usize) {
    let mut s = ready;
    for (k, &(a, b)) in busy.iter().enumerate() {
        if s + d <= a {
            return (s, k);
        }
        s = s.max(b);
    }
    (s, busy.len())
}

/// The schedule's order as [`JobSpec::priority`](whelm::JobSpec::priority)s: each task's position
/// when sorted by planned start, ties by planned finish, then index. (HEFT's selection order is
/// the upward-rank order, which the rank plans already use; the start order also carries what its
/// machine choices implied.)
pub fn priorities(plan: &Schedule) -> Vec<i64> {
    let n = plan.start.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        (plan.start[a].total_cmp(&plan.start[b]))
            .then(plan.finish[a].total_cmp(&plan.finish[b]))
            .then(a.cmp(&b))
    });
    let mut p = vec![0i64; n];
    for (pos, i) in order.into_iter().enumerate() {
        p[i] = pos as i64;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::small::{GridParams, grid};

    /// A HEFT schedule respects dependencies, never overlaps two tasks on a machine, and with true
    /// costs is a feasible schedule no shorter than the lower bounds.
    #[test]
    fn heft_schedules_are_feasible() {
        for seed in 0..30 {
            let inst = grid(&GridParams::random(seed));
            let (wp, d) = inst.bounds();
            let s = heft(&inst, true);
            let speeds = inst.machines();
            for (i, t) in inst.tasks.iter().enumerate() {
                for &dep in &t.deps {
                    assert!(s.start[i] >= s.finish[dep as usize] - 1e-9);
                }
                if let Some(m) = s.machine[i] {
                    assert!((s.finish[i] - s.start[i] - t.work / speeds[m]).abs() < 1e-9);
                }
            }
            for m in 0..speeds.len() {
                let mut iv: Vec<(f64, f64)> = (0..inst.tasks.len())
                    .filter(|&i| s.machine[i] == Some(m))
                    .map(|i| (s.start[i], s.finish[i]))
                    .collect();
                iv.sort_by(|a, b| a.0.total_cmp(&b.0));
                assert!(iv.windows(2).all(|w| w[0].1 <= w[1].0 + 1e-9));
            }
            assert!(s.makespan >= wp.max(d) * (1.0 - 1e-9));
        }
    }

    /// Priorities are a permutation that lists every predecessor first.
    #[test]
    fn priorities_are_topological() {
        let inst = grid(&GridParams::random(5));
        let p = priorities(&heft(&inst, false));
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..p.len() as i64).collect::<Vec<_>>());
        for (i, t) in inst.tasks.iter().enumerate() {
            assert!(t.deps.iter().all(|&d| p[d as usize] < p[i]));
        }
    }
}
