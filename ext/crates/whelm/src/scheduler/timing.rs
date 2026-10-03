//! Run-time estimates from the machine model, and learning from finished attempts.

use std::time::Duration;

use super::{Job, Running, Scheduler, Worker, order::ordered, tick_occ};
use crate::{JobId, Time, time::secs};

impl Scheduler {
    /// How fast `job` runs on `w`: the worker's speed times the job kind's factor on its class.
    /// Every speed a decision reads goes through here.
    pub(super) fn speed(&self, job: &Job, w: &Worker) -> f64 {
        w.speed * self.speeds.factor(job.kind, w.class)
    }

    /// The expected run time of `job` on `w`: its work over its [`speed`](Self::speed) there,
    /// `None` without a work estimate. Every run-time estimate goes through here.
    pub(super) fn eta(&self, job: &Job, w: &Worker) -> Option<Duration> {
        (job.spec.work).map(|work| secs(work.as_secs_f64() / self.speed(job, w)))
    }

    /// When a running job is expected to end: the earliest expected end of its live attempts,
    /// each its start plus its [`eta`](Self::eta), or, once that has passed, as far beyond now as
    /// it has run (an overrunning attempt is assumed half done, StarPU's rule). `None` if its run
    /// time is unknown.
    pub(super) fn expected_end(&self, r: &Running) -> Option<Time> {
        r.live
            .iter()
            .filter_map(|run| {
                let end = run.started + self.eta(&r.job, self.workers.get(&run.worker)?)?;
                Some(if end > self.now {
                    end
                } else {
                    self.now + (self.now - run.started)
                })
            })
            .min()
    }

    /// Live attempt `i` of a job finished: learn its speed on its worker from its duration and the
    /// worker's mean concurrency meanwhile.
    pub(super) fn learn_from(&mut self, job: JobId, i: usize) {
        let now = self.now;
        let r = &self.running[&job];
        let run = &r.live[i];
        let kind = r.job.kind;
        let (Some(work), Some(w)) = (r.job.spec.work, self.workers.get_mut(&run.worker)) else {
            return;
        };
        tick_occ(w, now);
        let dt = now - run.started;
        let k = if dt > Duration::ZERO {
            (w.occ - run.occ0) / dt.as_secs_f64()
        } else {
            1.0
        };
        let (id, class) = (w.state.id, w.class);
        if !self.speeds.observe(id, class, kind, work, dt, k) {
            return;
        }
        // The class estimate moved too: refresh every worker of the class.
        for w in self.workers.values_mut().filter(|w| w.class == class) {
            w.speed = self.speeds.worker_speed(w.state.id, class, w.state.speed);
        }
    }

    /// `job`'s speed on `w` as an ordering key (more negative is faster). With learning and a
    /// resolution, speeds within one resolution step of each other compare equal, so per-worker
    /// noise does not override load.
    pub(super) fn speed_rank(&self, job: &Job, w: &Worker) -> i64 {
        let speed = self.speed(job, w);
        let res = self.speeds.resolution();
        if res > 0.0 {
            -(speed.ln() / res.ln_1p()).round() as i64
        } else {
            ordered(-speed)
        }
    }
}
