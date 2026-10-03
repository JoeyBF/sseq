//! Holds: reservations, their shadow backfill, and deferral to a faster worker.

use super::{Job, Scheduler, Worker, order::ordered, placement::Score};
use crate::{JobId, Resources, Time, WorkerId, WorkerState, WorkerView};

/// A worker kept from a job on purpose although it might admit it: the one notion behind
/// reservations, their shadow backfill and deferral. Each waiting job has at most one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Hold {
    /// The job reserves `worker`: no other job may take it, except one expected to finish by
    /// `shadow` (shadow backfill). Lasts until the job is placed or the reservation is released.
    Reserve {
        worker: WorkerId,
        since: Time,
        /// Position among the reservations, for reporting them in creation order.
        order: u64,
        /// The holder's shadow time, fixed once known (so that an overrun can exceed it).
        shadow: Option<Time>,
    },
    /// The job waits for the faster, busy `worker`, expected to free at `at`, and declines every
    /// other worker until `until`. Decided afresh whenever `dispatch` reaches the job.
    Defer {
        worker: WorkerId,
        at: Time,
        until: Time,
    },
}

impl Hold {
    /// The worker the hold is about.
    pub(super) fn worker(&self) -> WorkerId {
        match *self {
            Hold::Reserve { worker, .. } | Hold::Defer { worker, .. } => worker,
        }
    }

    /// When the hold lapses by itself, if it does.
    pub(super) fn until(&self) -> Option<Time> {
        match *self {
            Hold::Reserve { .. } => None,
            Hold::Defer { until, .. } => Some(until),
        }
    }
}

impl Scheduler {
    /// The worker `job` reserves, if any.
    pub(super) fn reserved(&self, job: JobId) -> Option<WorkerId> {
        match self.holds.get(&job)? {
            Hold::Reserve { worker, .. } => Some(*worker),
            Hold::Defer { .. } => None,
        }
    }

    /// Drop the hold `job` has, if any, on both sides.
    pub(super) fn release_hold(&mut self, job: JobId) {
        if let Some(Hold::Reserve { worker, .. }) = self.holds.remove(&job)
            && let Some(w) = self.workers.get_mut(&worker)
        {
            w.reserved_for = None;
        }
    }

    /// Give `job` a reservation on `worker`, in position `order`.
    fn reserve(&mut self, job: JobId, worker: WorkerId, order: u64) {
        self.holds.insert(
            job,
            Hold::Reserve {
                worker,
                since: self.now,
                order,
                shadow: None,
            },
        );
        self.workers.get_mut(&worker).unwrap().reserved_for = Some(job);
        self.reservations_total += 1;
    }

    /// The hold that keeps `w` from `job`, and the job that has it: a reservation of `w` by
    /// another job that `job` cannot backfill, or `job`'s own deferral to another worker. This is
    /// the one place holds are enforced.
    pub(super) fn held(&self, job: &Job, w: &Worker) -> Option<(JobId, Hold)> {
        if let Some(holder) = w.reserved_for
            && holder != job.spec.id
            && !self.shadow_backfills(job, w)
        {
            return Some((holder, self.holds[&holder]));
        }
        match self.holds.get(&job.spec.id) {
            Some(&h @ Hold::Defer { worker, .. }) if worker != w.state.id => Some((job.spec.id, h)),
            _ => None,
        }
    }

    /// The shadow time of the reservation on `w`, once known.
    pub(super) fn shadow(&self, w: &Worker) -> Option<Time> {
        match self.holds.get(&w.reserved_for?)? {
            Hold::Reserve { shadow, .. } => *shadow,
            Hold::Defer { .. } => None,
        }
    }

    /// Whether `job` may backfill reserved worker `w`: it is expected to finish before the
    /// holder's shadow time.
    fn shadow_backfills(&self, job: &Job, w: &Worker) -> bool {
        let (Some(t), Some(run)) = (self.shadow(w), self.eta(job, w)) else {
            return false;
        };
        self.now + run <= t
    }

    /// The holder's shadow time on reserved worker `w`: the expected end of the running job
    /// whose release lets the configured admission rule admit the holder (now, if it already
    /// does). `None` if some end is unknown or no release suffices.
    fn shadow_time(&self, w: &Worker, holder: JobId) -> Option<Time> {
        let demand = self.waiting.get(&holder)?.spec.demand;
        let mut ends: Vec<(Time, Resources)> = Vec::with_capacity(w.jobs.len());
        for j in w.jobs.keys() {
            let r = &self.running[j];
            ends.push((self.expected_end(r)?, r.job.spec.demand));
        }
        ends.sort_by_key(|e| e.0);
        // The projection assumes a released job frees what it was placed with; the reported
        // usage cannot be predicted, so the hypothetical worker reports none above its baseline.
        let state = WorkerState {
            reported_used: Resources::ZERO,
            ..w.state.clone()
        };
        let fits = |placed: Resources| {
            let view = WorkerView {
                state: &state,
                placed,
            };
            self.admission.admits(&demand, &view)
        };
        let mut placed = w.placed;
        if fits(placed) {
            return Some(self.now);
        }
        for (end, d) in ends {
            placed -= d;
            if fits(placed) {
                return Some(end);
            }
        }
        None
    }

    /// Compute the shadow time of each reservation that has none yet.
    pub(super) fn refresh_shadows(&mut self) {
        if self.config.reservations.is_none_or(|c| !c.shadow_backfill) {
            return;
        }
        let pending: Vec<(JobId, WorkerId)> = self
            .holds
            .iter()
            .filter_map(|(&job, h)| match *h {
                Hold::Reserve {
                    worker,
                    shadow: None,
                    ..
                } => Some((job, worker)),
                _ => None,
            })
            .collect();
        for (job, worker) in pending {
            let t = self.shadow_time(&self.workers[&worker], job);
            if let Some(Hold::Reserve { shadow, .. }) = self.holds.get_mut(&job) {
                *shadow = t;
            }
        }
    }

    /// If `job` qualifies (waited long enough, no reservation yet), reserve a worker for it: the
    /// one with the most headroom if reservations are left, otherwise take over the reservation
    /// of the least urgent holder that is less urgent than `job` (its worker has been draining
    /// already). Returns whether `job` now holds a reservation.
    pub(super) fn try_reserve(&mut self, job: JobId) -> bool {
        let Some(cfg) = self.config.reservations else {
            return false;
        };
        let j = &self.waiting[&job];
        if self.reserved(job).is_some() || self.now - j.since < cfg.reserve_after {
            return false;
        }
        let reservations: Vec<(JobId, WorkerId, u64)> = self
            .holds
            .iter()
            .filter_map(|(&h, hold)| match *hold {
                Hold::Reserve { worker, order, .. } => Some((h, worker, order)),
                Hold::Defer { .. } => None,
            })
            .collect();
        let class_full = |class: &str| {
            let n = reservations
                .iter()
                .filter(|r| !cfg.per_class || self.workers[&r.1].state.class == class)
                .count();
            n >= cfg.max
        };
        // Most headroom; then the configured score; then smallest id.
        let mut best: Option<((i64, Score), WorkerId)> = None;
        for (&id, w) in &self.workers {
            if w.reserved_for.is_some()
                || !w.live()
                || class_full(&w.state.class)
                || !self.eligible(j, w)
            {
                continue;
            }
            let score = (
                ordered(-w.view().free_share(&Resources::ZERO)),
                self.score(j, w),
            );
            if best.as_ref().is_none_or(|(b, _)| score < *b) {
                best = Some((score, id));
            }
        }
        if let Some((_, w)) = best {
            let order = self.next_reservation;
            self.next_reservation += 1;
            self.reserve(job, w, order);
            return true;
        }
        let mine = self.urgency(j);
        let victim = reservations
            .iter()
            .filter(|r| self.eligible(j, &self.workers[&r.1]))
            .map(|r| (self.urgency(&self.waiting[&r.0]), r.0, r.1, r.2))
            .filter(|(u, ..)| *u > mine)
            .max();
        let Some((_, victim, w, order)) = victim else {
            return false;
        };
        self.release_hold(victim);
        self.reserve(job, w, order);
        true
    }
}
