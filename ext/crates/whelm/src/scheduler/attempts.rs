//! Attempts: starting, finishing, failing and stopping them, worker loss, and speculation.

use std::collections::BTreeMap;

use super::{Job, Run, Running, Scheduler, Worker, tick_occ};
use crate::{
    Attempt, FailKind, GaveUp, JobId, Output, Time, Tried, WorkerId, WorkerState,
    admission::WorkerAmounts,
    resources::{add, sub},
    time::secs,
};

impl Scheduler {
    /// Move a waiting job onto a worker as its next attempt.
    pub(super) fn place(&mut self, job: JobId, worker: WorkerId) {
        if self.reserved(job) == Some(worker) {
            self.last_dispatch_holders.push(job);
        }
        let j = self
            .remove_waiting(job)
            .expect("placing a job that is not waiting");
        self.running.insert(
            job,
            Running {
                job: j,
                live: Vec::new(),
            },
        );
        self.start(job, worker);
    }

    /// Start a new attempt of a running job on `worker` and emit it.
    pub(super) fn start(&mut self, job: JobId, worker: WorkerId) {
        let r = self.running.get_mut(&job).unwrap();
        r.job.attempts += 1;
        let attempt = r.job.attempts;
        let w = self
            .workers
            .get_mut(&worker)
            .expect("placing on an unknown worker");
        tick_occ(w, self.now);
        add(&mut w.placed, &r.job.demand);
        w.jobs.insert(job, attempt);
        r.live.push(Run {
            attempt,
            worker,
            started: self.now,
            occ0: w.occ,
        });
        self.placements_total += 1;
        self.outbox.push(Output::Start {
            job,
            attempt,
            worker,
        });
    }

    /// Release an attempt's demand on its worker (if the worker is still known).
    fn release_run(
        workers: &mut BTreeMap<WorkerId, Worker>,
        now: Time,
        job: JobId,
        demand: &[u64],
        run: &Run,
    ) {
        if let Some(w) = workers.get_mut(&run.worker) {
            tick_occ(w, now);
            sub(&mut w.placed, demand);
            w.jobs.remove(&job);
        }
    }

    /// Release every live attempt of a running job, emitting a stop for each one except
    /// `except`; returns the job.
    fn stop_running(&mut self, job: JobId, except: Option<Attempt>) -> Option<Job> {
        let r = self.running.remove(&job)?;
        for run in &r.live {
            Self::release_run(&mut self.workers, self.now, job, &r.job.demand, run);
            if Some(run.attempt) != except {
                self.outbox.push(Output::Stop {
                    job,
                    attempt: run.attempt,
                    worker: run.worker,
                });
            }
        }
        Some(r.job)
    }

    /// Drop a waiting job, or stop a running one.
    pub(super) fn cancel(&mut self, job: JobId) {
        if self.remove_waiting(job).is_none() {
            self.stop_running(job, None);
        }
    }

    /// The index of `attempt` among `job`'s live attempts, if it is one.
    pub(super) fn live(&self, job: JobId, attempt: Attempt) -> Option<usize> {
        self.running
            .get(&job)?
            .live
            .iter()
            .position(|r| r.attempt == attempt)
    }

    /// An attempt finished: the job is complete; its other attempts are stopped.
    pub(super) fn done(&mut self, job: JobId, attempt: Attempt) {
        let Some(i) = self.live(job, attempt) else {
            return;
        };
        self.learn_from(job, i);
        self.stop_running(job, Some(attempt));
    }

    /// An attempt failed: record it, then retry or give up once no attempt is live.
    pub(super) fn failed(&mut self, job: JobId, attempt: Attempt, kind: FailKind, why: String) {
        let Some(i) = self.live(job, attempt) else {
            return;
        };
        let r = self.running.get_mut(&job).unwrap();
        let run = r.live.remove(i);
        r.job.tried.push(Tried {
            worker: run.worker,
            kind,
            why,
        });
        if !r.job.retry_avoid.contains(&run.worker) {
            r.job.retry_avoid.push(run.worker);
        }
        let idle = r.live.is_empty();
        Self::release_run(&mut self.workers, self.now, job, &r.job.demand, &run);
        if !idle {
            return;
        }
        let j = self.running.remove(&job).unwrap().job;
        // Speculative attempts are extra tries within a round, not rounds of their own.
        if j.attempts - j.speculated < self.config.retry.max_attempts.max(1) {
            self.enqueue(j);
        } else {
            let retryable = j.tried.iter().all(|t| t.kind == FailKind::DeviceOom);
            self.outbox.push(Output::GaveUp(GaveUp {
                job,
                tried: j.tried,
                retryable,
            }));
        }
    }

    /// Add a worker or replace its reported state, keeping its placements.
    ///
    /// # Panics
    ///
    /// If `state` names a resource that is not declared.
    pub(super) fn worker_update(&mut self, state: WorkerState, now: Time) {
        let amounts = WorkerAmounts::new(&self.config.resources, &state);
        let class = self.speeds.class(&state.class);
        let speed = self.speeds.worker_speed(state.id, class, state.speed);
        match self.workers.get_mut(&state.id) {
            Some(w) => {
                // A reservation counted against the old class (per-class limits) must not move to
                // the new one; its holder reserves again at the next dispatch.
                let moved = (w.state.class != state.class)
                    .then_some(w.reserved_for)
                    .flatten();
                w.state = state;
                w.amounts = amounts;
                w.class = class;
                w.speed = speed;
                if let Some(holder) = moved {
                    self.release_hold(holder);
                }
            }
            None => {
                let id = state.id;
                self.workers.insert(
                    id,
                    Worker {
                        state,
                        amounts,
                        placed: self.zero.clone(),
                        jobs: BTreeMap::new(),
                        reserved_for: None,
                        class,
                        speed,
                        occ: 0.0,
                        occ_at: now,
                    },
                );
            }
        }
    }

    /// Forget a worker and every hold on it; each live attempt there fails with
    /// [`FailKind::LinkDied`].
    pub(super) fn worker_gone(&mut self, id: WorkerId) {
        let Some(w) = self.workers.remove(&id) else {
            return;
        };
        self.holds.retain(|_, h| h.worker() != id);
        for (job, attempt) in w.jobs {
            self.failed(
                job,
                attempt,
                FailKind::LinkDied,
                format!("worker {id} left"),
            );
        }
    }

    /// Start speculative attempts on workers left with room (see
    /// [`Speculate`](crate::Speculate)).
    pub(super) fn speculate(&mut self) {
        let Some(cfg) = self.config.speed.speculate else {
            return;
        };
        let now = self.now;
        let ids: Vec<WorkerId> = self.workers.keys().copied().collect();
        for to in ids {
            loop {
                let w = &self.workers[&to];
                if !self.has_room(w) || w.reserved_for.is_some() {
                    break;
                }
                // Candidates: run time known, every live attempt on a worker slower for the job,
                // allowed and admitted here.
                let mut best: Option<(Time, JobId)> = None;
                for (&job, r) in &self.running {
                    let rank = self.speed_rank(&r.job, w);
                    let slower = r.live.iter().all(|run| {
                        self.workers
                            .get(&run.worker)
                            .is_some_and(|v| self.speed_rank(&r.job, v) > rank)
                    });
                    if !slower
                        || r.job.speculated >= cfg.max_per_job
                        || !self.eligible(&r.job, w)
                        || !(self.admission)
                            .admits(&self.demand(&r.job), &w.view(&self.config.resources))
                    {
                        continue;
                    }
                    let (Some(end), Some(run)) = (self.expected_end(r), self.eta(&r.job, w)) else {
                        continue;
                    };
                    let end_here = now + run + cfg.restart_overhead;
                    if end <= end_here || end - end_here < secs(run.as_secs_f64() * cfg.min_gain) {
                        continue;
                    }
                    if best.is_none_or(|(e, j)| end > e || (end == e && job < j)) {
                        best = Some((end, job));
                    }
                }
                let Some((_, job)) = best else { break };
                self.running.get_mut(&job).unwrap().job.speculated += 1;
                self.start(job, to);
            }
        }
    }
}
