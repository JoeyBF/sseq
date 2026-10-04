//! What the scheduler reports: [`Policy::stats`] and [`Policy::explain`].

use super::{Job, Scheduler, Worker, holds::Hold, placement::Refusal};
#[cfg(doc)]
use crate::Policy;
use crate::{
    Explanation, Holding, JobId, PolicyStats, ReservationInfo, Status, Verdict, Waiting, WorkerLoad,
};

impl Scheduler {
    /// A worker's load as reported in the stats.
    pub(super) fn load(&self, w: &Worker) -> WorkerLoad {
        WorkerLoad {
            id: w.state.id,
            class: w.state.class.clone(),
            capacity: w.state.capacity.clone(),
            running: w.running(),
            placed: w.placed.clone(),
            headroom: w.view(&self.config.resources).headroom(),
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

    /// The job's live attempts if it runs, otherwise its wait and every worker's verdict.
    pub(super) fn explain(&self, job: JobId) -> Option<Explanation> {
        if let Some(r) = self.running.get(&job) {
            let attempts = r.live.iter().map(|run| (run.attempt, run.worker)).collect();
            return Some(Explanation::new(job, Status::Running { attempts }));
        }
        let j = self.waiting.get(&job)?;
        let hold = self.holds.get(&job).map(|h| match *h {
            Hold::Reserve {
                worker,
                since,
                shadow,
                ..
            } => {
                let w = &self.workers[&worker];
                Holding::Reservation {
                    worker,
                    since,
                    shadow,
                    running: w.running(),
                    capacity: w.state.capacity.clone(),
                    used: w.view(&self.config.resources).used(),
                }
            }
            Hold::Defer { worker, at, until } => Holding::Deferral {
                worker,
                expected_free: at,
                until,
            },
        });
        let mut kind_factors: Vec<(String, f64)> = match j.kind {
            Some(kind) => (self.speeds.kind_factors(kind).into_iter())
                .map(|(class, f)| (class.to_owned(), f))
                .collect(),
            None => Vec::new(),
        };
        kind_factors.sort_by(|a, b| a.0.cmp(&b.0));
        let workers = (self.workers.iter())
            .map(|(&id, w)| (id, self.verdict(j, w)))
            .collect();
        let waiting = Waiting {
            demand: j.spec.demand.clone(),
            resources: self.config.resources.clone(),
            group: j.spec.group,
            since: j.since,
            waited: self.now - j.since,
            ahead: self.queue.range(..j.key).count(),
            aged: self.aged(j),
            tried: j.tried.clone(),
            hold,
            kind: j.spec.kind.clone(),
            kind_factors,
            workers,
        };
        Some(Explanation::new(job, Status::Waiting(Box::new(waiting))))
    }

    /// Whether `w` takes `job`, or why not, from its [`Refusal`].
    fn verdict(&self, job: &Job, w: &Worker) -> Verdict {
        match self.refusal(job, w) {
            None => Verdict::Takes,
            Some(Refusal::Ineligible) => Verdict::Ineligible,
            Some(Refusal::Held(by, Hold::Reserve { .. })) => Verdict::Reserved { by },
            Some(Refusal::Held(_, Hold::Defer { .. })) => Verdict::Deferred,
            Some(Refusal::Admission) => {
                let resources = &self.config.resources;
                let view = w.view(resources);
                let (hard, soft): (Vec<_>, Vec<_>) =
                    (view.short(&job.spec.demand)).partition(|d| resources[d.0].hard);
                if hard.is_empty() {
                    Verdict::Short {
                        dims: soft,
                        headroom: view.headroom(),
                    }
                } else {
                    Verdict::Full { dims: hard }
                }
            }
        }
    }
}
