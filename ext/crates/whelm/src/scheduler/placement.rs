//! The scan: eligibility, admission and scoring of each worker for each waiting job.

use std::collections::{BTreeSet, HashMap};

#[cfg(doc)]
use super::order::Key;
use super::{
    Job, Scheduler, Worker,
    holds::Hold,
    order::{Cursor, ordered},
};
#[cfg(doc)]
use crate::Config;
use crate::{
    JobId, JobSpec, Resources, SLOTS, ScoreTerm, Selector, Strength, Time, WorkerId, WorkerState,
    WorkerView, time::secs,
};

/// The most terms a [`Config::score`] has once repeats are dropped: one per [`ScoreTerm`].
const SCORE_TERMS: usize = 5;

/// A worker's rank for a job under [`Config::score`]: smaller is better, like [`Key`].
pub(super) type Score = [i64; SCORE_TERMS];

/// Projected slot free times of busy workers during one `dispatch` (for deferral): per worker,
/// the expected end of each running or deferred job, smallest first.
type Projection = HashMap<WorkerId, Vec<Option<Time>>>;

/// What `choose` decided for a job.
enum Pick {
    Place(WorkerId),
    /// Wait for this worker, expected to start there at this time.
    Defer(WorkerId, Time),
    Nothing,
}

/// Why a worker does not take a job, for [`Verdict`](crate::Verdict).
pub(super) enum Refusal {
    Ineligible,
    /// A hold, and the job that has it.
    Held(JobId, Hold),
    Admission,
}

/// Whether `job`'s hard constraints allow `w`: no [`Strength::Forbid`] selects it, and for each
/// kind of selector the job requires on, some [`Strength::Require`] of that kind selects it.
fn allows(job: &JobSpec, w: &WorkerState) -> bool {
    // Per selector kind (worker, class): whether the job has a Require of it, and one selects w.
    let (mut required, mut met) = ([false; 2], [false; 2]);
    for c in &job.constraints {
        let selects = c.on.matches(w);
        match c.strength {
            Strength::Forbid if selects => return false,
            Strength::Require => {
                let kind = matches!(c.on, Selector::Class(_)) as usize;
                required[kind] = true;
                met[kind] |= selects;
            }
            _ => {}
        }
    }
    required == met
}

/// Whether a [`Strength::Avoid`] of `job`, or one of its failed attempts, selects `w`.
fn avoids(job: &Job, w: &WorkerState) -> bool {
    job.retry_avoid.contains(&w.id)
        || job
            .spec
            .constraints
            .iter()
            .any(|c| c.strength == Strength::Avoid && c.on.matches(w))
}

/// Whether a [`Strength::Prefer`] of `job` selects `w`.
fn prefers(job: &JobSpec, w: &WorkerState) -> bool {
    job.constraints
        .iter()
        .any(|c| c.strength == Strength::Prefer && c.on.matches(w))
}

/// Whether `job` can run on some worker of `classes` as far as its class Requires go.
fn class_possible(job: &JobSpec, classes: &BTreeSet<String>) -> bool {
    let mut required = (job.constraints.iter())
        .filter_map(|c| match (&c.on, c.strength) {
            (Selector::Class(class), Strength::Require) => Some(class),
            _ => None,
        })
        .peekable();
    required.peek().is_none() || required.any(|c| classes.contains(c))
}

impl Scheduler {
    /// Whether `job`'s constraints allow `w` at all. Avoidance (its [`Strength::Avoid`]s and the
    /// workers of failed attempts) lapses while no live worker that the hard constraints allow is
    /// free of it; that depends on the worker set only, not on load, so admission stays monotone.
    pub(super) fn eligible(&self, job: &Job, w: &Worker) -> bool {
        let spec = &job.spec;
        allows(spec, &w.state)
            && (!avoids(job, &w.state)
                || !self
                    .workers
                    .values()
                    .any(|o| o.live() && allows(spec, &o.state) && !avoids(job, &o.state)))
    }

    /// Whether `w` takes the job, or why not.
    pub(super) fn refusal(&self, job: &Job, w: &Worker) -> Option<Refusal> {
        if !self.eligible(job, w) {
            return Some(Refusal::Ineligible);
        }
        if let Some((by, hold)) = self.held(job, w) {
            return Some(Refusal::Held(by, hold));
        }
        if !self.admission.admits(&job.spec.demand, &w.view()) {
            return Some(Refusal::Admission);
        }
        None
    }

    /// When a busy worker's next slot is expected to free: the earliest projected end, or `None`
    /// if a running job's end is unknown on every slot.
    fn next_free(&self, w: &Worker, proj: &mut Projection) -> Option<Time> {
        let ends = proj.entry(w.state.id).or_insert_with(|| {
            let mut ends: Vec<Option<Time>> = (w.jobs.keys())
                .map(|j| self.expected_end(&self.running[j]))
                .collect();
            ends.sort_by_key(|e| (e.is_none(), *e));
            ends
        });
        ends.first().copied().flatten()
    }

    /// Worker `w`'s rank for `job` under [`Config::score`].
    pub(super) fn score(&self, job: &Job, w: &Worker) -> Score {
        let mut score = [0; SCORE_TERMS];
        for (slot, term) in score.iter_mut().zip(&self.config.score) {
            *slot = match term {
                ScoreTerm::Speed => self.speed_rank(job, w),
                ScoreTerm::Tightest => ordered(w.view().free_share(&job.spec.demand)),
                ScoreTerm::Loosest => ordered(-w.view().free_share(&job.spec.demand)),
                ScoreTerm::Preferred => !prefers(&job.spec, &w.state) as i64,
                ScoreTerm::Load => w.running() as i64,
            };
        }
        score
    }

    /// The best worker that takes `job`, or a busy faster worker to wait for, if any.
    fn choose(&self, job: &Job, proj: &mut Projection) -> Pick {
        // Smallest score wins; the worker id makes the order total (determinism).
        let mut best: Option<(Score, WorkerId)> = None;
        for (&id, w) in &self.workers {
            if self.refusal(job, w).is_some() {
                continue;
            }
            let score = self.score(job, w);
            if best.as_ref().is_none_or(|(b, _)| score < *b) {
                best = Some((score, id));
            }
        }
        let Some((_, place)) = best else {
            return Pick::Nothing;
        };
        if let Some(defer) = self.config.speed.defer
            && let Some(run_here) = self.eta(job, &self.workers[&place])
            && self.reserved(job.spec.id).is_none()
            && !self.aged(job)
            && self.now - job.since < defer.max_wait
        {
            let work = job.spec.work.unwrap_or_default();
            let here = self.now + run_here;
            let speed_here = self.speed(job, &self.workers[&place]);
            let mut wait: Option<(Time, WorkerId)> = None;
            for (&id, w) in &self.workers {
                let slots = w.state.slots;
                // Only full workers, and only if they would admit the job with one slot free.
                if self.speed(job, w) <= speed_here
                    || slots == 0
                    || w.running() < slots
                    || !self.eligible(job, w)
                    || w.reserved_for.is_some_and(|h| h != job.spec.id)
                {
                    continue;
                }
                let mut placed = w.placed;
                placed[SLOTS] = slots as u64 - 1;
                let view = WorkerView {
                    state: &w.state,
                    placed,
                };
                if !self.admission.admits(&job.spec.demand, &view) {
                    continue;
                }
                let (Some(start), Some(run)) = (self.next_free(w, proj), self.eta(job, w)) else {
                    continue;
                };
                let eft = start.max(self.now) + run;
                if wait.is_none_or(|(e, _)| eft < e) {
                    wait = Some((eft, id));
                }
            }
            if let Some((eft, id)) = wait
                && eft + secs(work.as_secs_f64() * defer.min_gain) < here
            {
                // Book the slot so that later deferrals in this scan see it taken.
                let start = self.next_free(&self.workers[&id], proj).unwrap();
                let ends = proj.get_mut(&id).unwrap();
                ends.remove(0);
                let at = ends.partition_point(|&e| e.is_some_and(|e| e < eft));
                ends.insert(at, Some(eft));
                return Pick::Defer(id, start.max(self.now));
            }
        }
        Pick::Place(place)
    }

    /// Whether a worker takes jobs other than its holder's: unreserved, or backfillable.
    fn open(&self, w: &Worker) -> bool {
        w.reserved_for.is_none() || self.shadow(w).is_some()
    }

    /// Whether worker `w` admits anything at all, by the admission bound.
    pub(super) fn has_room(&self, w: &Worker) -> bool {
        self.admission.bound(&w.view()).is_some()
    }

    /// Classes with an open worker that has room.
    fn open_classes(&self) -> BTreeSet<String> {
        self.workers
            .values()
            .filter(|w| self.open(w) && self.has_room(w))
            .map(|w| w.state.class.clone())
            .collect()
    }

    /// Component-wise maximum admission bound over open workers, or `None` if no open worker can
    /// take anything.
    fn open_bound(&self) -> Option<Resources> {
        self.workers
            .values()
            .filter(|w| self.open(w))
            .filter_map(|w| self.admission.bound(&w.view()))
            .reduce(Resources::max)
    }

    /// Scan waiting jobs in order, placing each where it is admitted, reserving for the starving.
    pub(super) fn dispatch(&mut self) {
        self.last_dispatch_holders.clear();
        self.deferred_any.clear();
        // Deferrals are decided afresh by this scan. A reservation on a worker that can never run
        // anything again is dead weight, and so is one its holder may no longer use (a soft avoid
        // list that lapsed when it reserved holds again once another worker joins).
        self.holds.retain(|_, h| matches!(h, Hold::Reserve { .. }));
        let dead: Vec<JobId> = self
            .holds
            .iter()
            .filter(|(job, h)| {
                let w = &self.workers[&h.worker()];
                !w.live() || !self.eligible(&self.waiting[job], w)
            })
            .map(|(&job, _)| job)
            .collect();
        for job in dead {
            self.release_hold(job);
        }
        let mut proj = Projection::new();
        'scan: loop {
            if !self.workers.values().any(|w| self.has_room(w)) {
                break;
            }
            let mut bound = self.open_bound();
            let mut classes = self.open_classes();
            let mut cursor = Cursor::Aged(None);
            proj.clear();
            self.refresh_shadows();
            loop {
                let Some(job) = self.next_job(&mut cursor) else {
                    break 'scan;
                };
                if matches!(self.holds.get(&job), Some(Hold::Defer { .. })) {
                    self.holds.remove(&job);
                }
                let j = &self.waiting[&job];
                let holder = self.reserved(job).is_some();
                // Cheap pruning: the admission bound, and required classes without room.
                let hopeful = holder
                    || (bound.is_some_and(|b| j.spec.demand.fits_within(&b))
                        && class_possible(&j.spec, &classes));
                let pick = if hopeful {
                    self.choose(j, &mut proj)
                } else {
                    Pick::Nothing
                };
                match pick {
                    Pick::Defer(worker, at) => {
                        let Some(d) = self.config.speed.defer else {
                            unreachable!("deferral without a Defer config")
                        };
                        let until = j.since + d.max_wait;
                        self.holds.insert(job, Hold::Defer { worker, at, until });
                        if !self.deferred_any.contains(&job) {
                            self.deferred_any.push(job);
                        }
                    }
                    Pick::Place(w) => {
                        self.place(job, w);
                        if holder {
                            // Its worker is open again: more urgent jobs refused there because of
                            // the reservation must get the first look.
                            continue 'scan;
                        }
                        if !self.workers.values().any(|w| self.has_room(w)) {
                            break 'scan;
                        }
                        bound = self.open_bound();
                        classes = self.open_classes();
                    }
                    Pick::Nothing => {
                        if self.try_reserve(job) {
                            self.refresh_shadows();
                            // A taken-over worker may admit the job now that it is the holder.
                            if let Pick::Place(w) = self.choose(&self.waiting[&job], &mut proj) {
                                self.place(job, w);
                                continue 'scan;
                            }
                            bound = self.open_bound();
                            classes = self.open_classes();
                        }
                    }
                }
            }
        }
    }
}
