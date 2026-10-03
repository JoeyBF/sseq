//! What the scheduler reports: [`Policy::stats`] and [`Policy::explain`].

use super::{Scheduler, Worker, holds::Hold, placement::Refusal};
#[cfg(doc)]
use crate::Policy;
use crate::{
    DEV, DIMS, JobId, MEM, PolicyStats, ReservationInfo, Resources, SLOTS, WorkerId, WorkerLoad,
};

/// Bytes per gigabyte, for `explain`.
pub(super) const GB: f64 = 1e9;

/// Each dimension's name in `explain`, indexed by dimension.
const DIM_NAMES: [&str; DIMS] = ["memory", "device memory", "slots"];

/// A resource vector in words for `explain`: host memory always, device memory when nonzero.
fn gb_list(r: &Resources) -> String {
    let mut parts = vec![format!("{:.2} GB", r[MEM] as f64 / GB)];
    if r[DEV] > 0 {
        parts.push(format!("{:.2} GB device", r[DEV] as f64 / GB));
    }
    parts.join(" + ")
}

impl Scheduler {
    /// A worker's load as reported in the stats.
    pub(super) fn load(&self, w: &Worker) -> WorkerLoad {
        WorkerLoad {
            id: w.state.id,
            class: w.state.class.clone(),
            slots: w.state.slots,
            running: w.running(),
            placed: w.placed,
            headroom: w.view().headroom(),
            reserved_for: w.reserved_for,
            speed: w.speed,
        }
    }

    /// Counters and per-worker load.
    pub(super) fn stats(&self) -> PolicyStats {
        let mut reservations: Vec<(u64, ReservationInfo)> = Vec::new();
        let mut deferred = Vec::new();
        for (&job, h) in &self.holds {
            match *h {
                Hold::Reserve {
                    worker,
                    since,
                    order,
                    ..
                } => reservations.push((order, ReservationInfo { job, worker, since })),
                Hold::Defer { worker, at, .. } => {
                    deferred.push((self.urgency(&self.waiting[&job]), (job, worker, at)))
                }
            }
        }
        reservations.sort_by_key(|r| r.0);
        deferred.sort_by_key(|d| d.0);
        PolicyStats {
            now: self.now,
            waiting: self.waiting.len(),
            running: self.running.len(),
            longest_wait: self
                .by_age
                .values()
                .next()
                .map(|j| (*j, self.now - self.waiting[j].since)),
            reservations: reservations.into_iter().map(|r| r.1).collect(),
            placements_total: self.placements_total,
            reservations_total: self.reservations_total,
            workers: self.workers.values().map(|w| self.load(w)).collect(),
            last_dispatch_holders: self.last_dispatch_holders.clone(),
            deferred: deferred.into_iter().map(|d| d.1).collect(),
            deferred_any: self.deferred_any.clone(),
        }
    }

    /// Classify every worker's reason to refuse the job, and summarise.
    pub(super) fn explain(&self, job: JobId) -> Option<String> {
        if let Some(r) = self.running.get(&job) {
            let runs: Vec<String> = r
                .live
                .iter()
                .map(|run| format!("attempt {} on worker {}", run.attempt, run.worker))
                .collect();
            return Some(format!("job {job} is running: {}", runs.join(", ")));
        }
        let j = self.waiting.get(&job)?;
        let ahead = self.queue.range(..j.key).count();
        let mut msg = format!(
            "job {job} (demand {}, group {}) waiting {:.0}s, {ahead} more urgent job(s) waiting",
            gb_list(&j.spec.demand),
            j.spec.group,
            (self.now - j.since).as_secs_f64()
        );
        if let Some(last) = j.tried.last() {
            msg += &format!(
                "; failed {} time(s), last on worker {} ({:?}: {})",
                j.tried.len(),
                last.worker,
                last.kind,
                last.why
            );
        }
        match self.holds.get(&job) {
            Some(Hold::Reserve { worker, .. }) => {
                let w = &self.workers[worker];
                msg += &format!(
                    "; holds the reservation on worker {} (draining: {}/{} running, used {})",
                    w.state.id,
                    w.running(),
                    w.state.slots,
                    gb_list(&w.view().used())
                );
            }
            Some(Hold::Defer { worker, at, .. }) => {
                msg += &format!(
                    "; waiting for faster worker {worker} (expected free at t={:.0})",
                    at.as_secs_f64()
                );
            }
            None => {}
        }
        if let (Some(kind), Some(name)) = (j.kind, &j.spec.kind) {
            let factors: Vec<String> = (self.speeds.kind_factors(kind).into_iter())
                .map(|(class, f)| format!("{f:.2}x on class {class}"))
                .collect();
            if !factors.is_empty() {
                msg += &format!("; kind {name} runs {}", factors.join(", "));
            }
        }
        let (mut full, mut excluded) = (0, 0);
        // Per dimension: workers short of it, and the one with the most headroom there.
        let mut short = [0usize; DIMS];
        let mut best_short: [Option<(i64, WorkerId)>; DIMS] = [None; DIMS];
        let mut reserved = Vec::new();
        let mut takers = Vec::new();
        for (&id, w) in &self.workers {
            match self.refusal(j, w) {
                None => takers.push(id),
                Some(Refusal::Ineligible) => excluded += 1,
                Some(Refusal::Held(h, Hold::Reserve { .. })) => {
                    reserved.push(format!("worker {id} for job {h}"))
                }
                Some(Refusal::Held(_, Hold::Defer { .. })) => {}
                Some(Refusal::Admission) => {
                    let view = w.view();
                    let headroom = view.headroom();
                    let lacking: Vec<usize> = view.short(&j.spec.demand).collect();
                    if lacking.contains(&SLOTS) {
                        full += 1;
                        continue;
                    }
                    for d in lacking {
                        short[d] += 1;
                        let h = headroom[d].unwrap_or(i64::MAX);
                        if best_short[d].is_none_or(|(b, _)| h > b) {
                            best_short[d] = Some((h, id));
                        }
                    }
                }
            }
        }
        if self.workers.is_empty() {
            msg += "; no workers";
        }
        if full > 0 {
            msg += &format!("; slots full on {full} worker(s)");
        }
        for d in 0..DIMS {
            if let Some((h, w)) = best_short[d] {
                msg += &format!(
                    "; {} short on {} worker(s) (best headroom {:.2} GB on worker {w})",
                    DIM_NAMES[d],
                    short[d],
                    h as f64 / GB
                );
            }
        }
        if excluded > 0 {
            msg += &format!("; {excluded} worker(s) excluded by its constraints");
        }
        if !reserved.is_empty() {
            msg += &format!("; reserved: {}", reserved.join(", "));
        }
        if !takers.is_empty() {
            msg += &format!(
                "; admitted on worker(s) {takers:?} (placed at the next poll unless a more urgent \
                 job takes the slot)"
            );
        }
        Some(msg)
    }
}
