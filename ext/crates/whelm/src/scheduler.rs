//! The one placement policy, [`Scheduler`].

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    ops::Bound,
};

use crate::{
    Admission, Attempt, Config, DEV, DIMS, FailKind, GaveUp, GroupOrder, Input, Instant, JobId,
    JobSpec, MEM, OrderTerm, Output, Policy, PolicyStats, ProductionAdmission, ReservationInfo,
    Resources, SLOTS, ScoreTerm, Selector, Strength, Tried, WorkerId, WorkerLoad, WorkerState,
    WorkerView,
    speed::{ClassId, KindId, Speeds},
};

/// The most terms a [`Config::order`] has once repeats are dropped: one per [`OrderTerm`].
const ORDER_TERMS: usize = 5;

/// The most terms a [`Config::score`] has once repeats are dropped: one per [`ScoreTerm`].
const SCORE_TERMS: usize = 5;

/// Urgency: smaller is more urgent. `terms[i]` is the key of `config.order[i]`; the rest are 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    terms: [i64; ORDER_TERMS],
    seq: u64,
}

/// A worker's rank for a job under [`Config::score`]: smaller is better, like [`Key`].
type Score = [i64; SCORE_TERMS];

/// A worker's reported state with the scheduler's bookkeeping of it.
#[derive(Clone, Debug)]
struct Worker {
    state: WorkerState,
    /// The demands of the live attempts here, one slot each.
    placed: Resources,
    /// The jobs with a live attempt here, and that attempt (a job has at most one per worker).
    jobs: BTreeMap<JobId, Attempt>,
    /// The job holding a [`Hold::Reserve`] on this worker, if any.
    reserved_for: Option<JobId>,
    /// `state.class`, interned.
    class: ClassId,
    /// Speed for a job of no particular kind: learned, as reported, or 1 ([`crate::Timing`]).
    speed: f64,
    /// `∫ running dt` up to `occ_at` (mean concurrency over a job's run, for learning).
    occ: f64,
    occ_at: Instant,
}

impl Worker {
    /// The worker as the admission rule sees it.
    fn view(&self) -> WorkerView<'_> {
        WorkerView {
            state: &self.state,
            placed: self.placed,
        }
    }

    /// Live attempts here.
    fn running(&self) -> usize {
        self.placed[SLOTS] as usize
    }

    /// Whether the worker can ever run anything: it has slots.
    fn live(&self) -> bool {
        self.state.slots > 0
    }
}

/// A job across its attempts. A retry requeues it with the same `key` and `since`, so it keeps
/// its place and its age.
#[derive(Clone, Debug)]
struct Job {
    spec: JobSpec,
    /// `spec.kind`, interned if the timing distinguishes kinds.
    kind: Option<KindId>,
    key: Key,
    since: Instant,
    /// Attempts started so far: the last attempt's number.
    attempts: Attempt,
    /// Failed attempts.
    tried: Vec<Tried>,
    /// Workers its failed attempts ran on, avoided softly like a [`Strength::Avoid`].
    retry_avoid: Vec<WorkerId>,
    /// Speculative attempts started.
    speculated: u32,
}

/// One live attempt of a running job.
#[derive(Clone, Debug)]
struct Run {
    attempt: Attempt,
    worker: WorkerId,
    started: Instant,
    /// The worker's `occ` when it started.
    occ0: f64,
}

/// A job with at least one live attempt.
#[derive(Clone, Debug)]
struct Running {
    job: Job,
    live: Vec<Run>,
}

/// A worker kept from a job on purpose although it might admit it: the one notion behind
/// reservations, their shadow backfill and deferral. Each waiting job has at most one.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Hold {
    /// The job reserves `worker`: no other job may take it, except one expected to finish by
    /// `shadow` (shadow backfill). Lasts until the job is placed or the reservation is released.
    Reserve {
        worker: WorkerId,
        since: Instant,
        /// Position among the reservations, for reporting them in creation order.
        order: u64,
        /// The holder's shadow time, fixed once known (so that an overrun can exceed it).
        shadow: Option<Instant>,
    },
    /// The job waits for the faster, busy `worker`, expected to free at `at`, and declines every
    /// other worker until `until`. Decided afresh whenever `dispatch` reaches the job.
    Defer {
        worker: WorkerId,
        at: Instant,
        until: Instant,
    },
}

impl Hold {
    /// The worker the hold is about.
    fn worker(&self) -> WorkerId {
        match *self {
            Hold::Reserve { worker, .. } | Hold::Defer { worker, .. } => worker,
        }
    }

    /// When the hold lapses by itself, if it does.
    fn until(&self) -> Option<Instant> {
        match *self {
            Hold::Reserve { .. } => None,
            Hold::Defer { until, .. } => Some(until),
        }
    }
}

/// Projected slot free times of busy workers during one `dispatch` (for deferral): per worker,
/// the expected end of each running or deferred job, smallest first.
type Projection = HashMap<WorkerId, Vec<f64>>;

/// What `choose` decided for a job.
enum Pick {
    Place(WorkerId),
    /// Wait for this worker, expected to start there at this time.
    Defer(WorkerId, f64),
    Nothing,
}

/// Why a worker does not take a job, for `explain`.
enum Refusal {
    Ineligible,
    /// A hold, and the job that has it.
    Held(JobId, Hold),
    Admission,
}

/// Position of `dispatch`'s scan: first the aged jobs by age, then the rest by urgency.
#[derive(Clone, Copy, Debug)]
enum Cursor {
    Aged(Option<u64>),
    Queue(Option<Key>),
}

/// Bytes per gigabyte, for `explain`.
const GB: f64 = 1e9;

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

/// A float as a totally ordered integer key (for order and score keys).
fn ordered(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    b ^ (((b >> 63) as u64) >> 1) as i64
}

/// Bring a worker's concurrency integral up to `now`.
fn tick_occ(w: &mut Worker, now: Instant) {
    if now > w.occ_at {
        w.occ += w.running() as f64 * (now - w.occ_at);
        w.occ_at = now;
    }
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

/// `terms` without repeats, keeping first occurrences.
fn dedup<T: PartialEq + Copy>(terms: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(terms.len());
    for &t in terms {
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

/// The placement policy: list scheduling with admission, reservations and backfill, configured
/// by a [`Config`].
///
/// - Jobs are considered in [`Config::order`], aged jobs first ([`Config::age_limit`]).
/// - A job takes a worker only if no more urgent waiting job is admitted there; among the
///   workers that admit it, the one [`Config::score`] ranks first.
/// - [`Config::reservations`] drain a worker for a starving job; every other worker keeps
///   admitting less urgent jobs.
///
/// - Failed attempts are retried ([`Config::retry`]) and, with
///   [`SpeedConfig::speculate`](crate::SpeedConfig::speculate), idle fast workers run second
///   attempts of jobs on slow ones.
///
/// Each [`poll`](Policy::poll) scans waiting jobs in order and gives each one a worker if any
/// admits it. Because the scan is in urgency order and admission is monotone in load, a job is
/// placed on a worker only if every more urgent waiting job was refused there -- the priority
/// invariant -- without any explicit check. The one event that can make an already-refused worker
/// admissible mid-scan is the release of a hold (a reservation whose holder is placed); the scan
/// restarts from the top when that happens. Speculative attempts come after the scan, on workers
/// every waiting job was refused on.
pub struct Scheduler {
    config: Config,
    admission: Box<dyn Admission + Send>,
    workers: BTreeMap<WorkerId, Worker>,
    queue: BTreeMap<Key, JobId>,
    by_age: BTreeMap<u64, JobId>,
    waiting: HashMap<JobId, Job>,
    running: HashMap<JobId, Running>,
    /// Outputs not yet returned by `poll`.
    outbox: Vec<Output>,
    /// Group -> sequence number of its first arrival.
    groups: HashMap<u64, u64>,
    /// Holds by job.
    holds: BTreeMap<JobId, Hold>,
    next_seq: u64,
    /// The next [`Hold::Reserve`] order.
    next_reservation: u64,
    now: Instant,
    placements_total: u64,
    reservations_total: u64,
    last_dispatch_holders: Vec<JobId>,
    /// Every job that deferred at some point of the last dispatch, placed later or not.
    deferred_any: Vec<JobId>,
    /// The machine model's state ([`SpeedConfig::timing`](crate::SpeedConfig::timing)).
    speeds: Speeds,
}

impl fmt::Debug for Scheduler {
    /// The configuration and the job and worker counts.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scheduler")
            .field("config", &self.config)
            .field("workers", &self.workers.len())
            .field("waiting", &self.waiting.len())
            .field("running", &self.running.len())
            .finish_non_exhaustive()
    }
}

impl Scheduler {
    /// A scheduler with the production admission rule.
    pub fn new(config: Config) -> Self {
        Self::with_admission(config, ProductionAdmission)
    }

    /// A scheduler with a custom admission rule.
    pub fn with_admission(mut config: Config, admission: impl Admission + Send + 'static) -> Self {
        config.order = dedup(&config.order);
        config.score = dedup(&config.score);
        Self {
            speeds: Speeds::new(config.speed.timing),
            config,
            admission: Box::new(admission),
            workers: BTreeMap::new(),
            queue: BTreeMap::new(),
            by_age: BTreeMap::new(),
            waiting: HashMap::new(),
            running: HashMap::new(),
            outbox: Vec::new(),
            groups: HashMap::new(),
            holds: BTreeMap::new(),
            next_seq: 0,
            next_reservation: 0,
            now: 0.0,
            placements_total: 0,
            reservations_total: 0,
            last_dispatch_holders: Vec::new(),
            deferred_any: Vec::new(),
        }
    }

    /// Forget a group's first-arrival time. A later job of that group then counts as a new group.
    /// Use it when a group is known to be finished, to bound memory.
    pub fn forget_group(&mut self, group: u64) {
        self.groups.remove(&group);
    }

    /// How fast `job` runs on `w`: the worker's speed times the job kind's factor on its class.
    /// Every speed a decision reads goes through here.
    fn speed(&self, job: &Job, w: &Worker) -> f64 {
        w.speed * self.speeds.factor(job.kind, w.class)
    }

    /// The expected run time of `job` on `w`: its work over its [`speed`](Self::speed) there,
    /// `None` without a work estimate. Every run-time estimate goes through here.
    fn eta(&self, job: &Job, w: &Worker) -> Option<f64> {
        job.spec.work.map(|work| work / self.speed(job, w))
    }

    /// When a running job is expected to end: the earliest expected end of its live attempts,
    /// each its start plus its [`eta`](Self::eta), or, once that has passed, as far beyond now as
    /// it has run (an overrunning attempt is assumed half done, StarPU's rule). `None` if its run
    /// time is unknown.
    fn expected_end(&self, r: &Running) -> Option<Instant> {
        r.live
            .iter()
            .filter_map(|run| {
                let end = run.started + self.eta(&r.job, self.workers.get(&run.worker)?)?;
                Some(if end > self.now {
                    end
                } else {
                    self.now + (self.now - run.started).max(0.0)
                })
            })
            .reduce(f64::min)
    }

    /// Whether `job`'s constraints allow `w` at all. Avoidance (its [`Strength::Avoid`]s and the
    /// workers of failed attempts) lapses while no live worker that the hard constraints allow is
    /// free of it; that depends on the worker set only, not on load, so admission stays monotone.
    fn eligible(&self, job: &Job, w: &Worker) -> bool {
        let spec = &job.spec;
        allows(spec, &w.state)
            && (!avoids(job, &w.state)
                || !self
                    .workers
                    .values()
                    .any(|o| o.live() && allows(spec, &o.state) && !avoids(job, &o.state)))
    }

    /// The urgency key of a job submitted as number `seq`, recording its group's first arrival
    /// if groups are ordered by it.
    fn key(&mut self, spec: &JobSpec, seq: u64) -> Key {
        let mut terms = [0; ORDER_TERMS];
        for (slot, term) in terms.iter_mut().zip(&self.config.order) {
            *slot = match term {
                OrderTerm::Priority => spec.priority.unwrap_or(self.config.default_priority),
                OrderTerm::Rank => spec.rank.map_or(i64::MAX, |r| ordered(-r)),
                OrderTerm::Group => match self.config.group_order {
                    GroupOrder::Arrival => *self.groups.entry(spec.group).or_insert(seq) as i64,
                    // Order-preserving from u64 to i64.
                    GroupOrder::Id => spec.group as i64 ^ i64::MIN,
                },
                OrderTerm::Wspt => spec.work.map_or(i64::MAX, |w| ordered(-(spec.weight / w))),
                OrderTerm::Edd => spec.due.map_or(i64::MAX, ordered),
            };
        }
        Key { terms, seq }
    }

    /// Queue a new job under its urgency key, demanding one slot.
    fn submit(&mut self, mut spec: JobSpec, now: Instant) {
        if self.waiting.contains_key(&spec.id) || self.running.contains_key(&spec.id) {
            return;
        }
        spec.demand[SLOTS] = 1;
        let seq = self.next_seq;
        self.next_seq += 1;
        let key = self.key(&spec, seq);
        let kind = self.speeds.kind(spec.kind.as_deref());
        self.enqueue(Job {
            kind,
            spec,
            key,
            since: now,
            attempts: 0,
            tried: Vec::new(),
            retry_avoid: Vec::new(),
            speculated: 0,
        });
    }

    /// Put a job in the waiting indexes under its key.
    fn enqueue(&mut self, job: Job) {
        self.queue.insert(job.key, job.spec.id);
        self.by_age.insert(job.key.seq, job.spec.id);
        self.waiting.insert(job.spec.id, job);
    }

    /// The worker `job` reserves, if any.
    fn reserved(&self, job: JobId) -> Option<WorkerId> {
        match self.holds.get(&job)? {
            Hold::Reserve { worker, .. } => Some(*worker),
            Hold::Defer { .. } => None,
        }
    }

    /// Drop the hold `job` has, if any, on both sides.
    fn release_hold(&mut self, job: JobId) {
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

    /// Take a job out of the waiting indexes, releasing its hold.
    fn remove_waiting(&mut self, job: JobId) -> Option<Job> {
        self.release_hold(job);
        let j = self.waiting.remove(&job)?;
        self.queue.remove(&j.key);
        self.by_age.remove(&j.key.seq);
        Some(j)
    }

    /// Release an attempt's slot and demand on its worker (if the worker is still known).
    fn release_run(&mut self, job: JobId, demand: Resources, run: &Run) {
        if let Some(w) = self.workers.get_mut(&run.worker) {
            tick_occ(w, self.now);
            w.placed -= demand;
            w.jobs.remove(&job);
        }
    }

    /// Release every live attempt of a running job, emitting a stop for each one except
    /// `except`; returns the job.
    fn stop_running(&mut self, job: JobId, except: Option<Attempt>) -> Option<Job> {
        let r = self.running.remove(&job)?;
        for run in &r.live {
            self.release_run(job, r.job.spec.demand, run);
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
    fn cancel(&mut self, job: JobId) {
        if self.remove_waiting(job).is_none() {
            self.stop_running(job, None);
        }
    }

    /// The index of `attempt` among `job`'s live attempts, if it is one.
    fn live(&self, job: JobId, attempt: Attempt) -> Option<usize> {
        self.running
            .get(&job)?
            .live
            .iter()
            .position(|r| r.attempt == attempt)
    }

    /// An attempt finished: the job is complete; its other attempts are stopped.
    fn done(&mut self, job: JobId, attempt: Attempt) {
        let Some(i) = self.live(job, attempt) else {
            return;
        };
        self.learn_from(job, i);
        self.stop_running(job, Some(attempt));
    }

    /// An attempt failed: record it, then retry or give up once no attempt is live.
    fn failed(&mut self, job: JobId, attempt: Attempt, kind: FailKind, why: String) {
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
        let demand = r.job.spec.demand;
        let idle = r.live.is_empty();
        self.release_run(job, demand, &run);
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

    /// Live attempt `i` of a job finished: learn its speed on its worker from its duration and the
    /// worker's mean concurrency meanwhile.
    fn learn_from(&mut self, job: JobId, i: usize) {
        let now = self.now;
        let r = &self.running[&job];
        let run = &r.live[i];
        let kind = r.job.kind;
        let (Some(work), Some(w)) = (r.job.spec.work, self.workers.get_mut(&run.worker)) else {
            return;
        };
        tick_occ(w, now);
        let dt = now - run.started;
        let k = if dt > 0.0 {
            (w.occ - run.occ0) / dt
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
    fn speed_rank(&self, job: &Job, w: &Worker) -> i64 {
        let speed = self.speed(job, w);
        let res = self.speeds.resolution();
        if res > 0.0 {
            -(speed.ln() / res.ln_1p()).round() as i64
        } else {
            ordered(-speed)
        }
    }

    /// Add a worker or replace its reported state, keeping its placements.
    fn worker_update(&mut self, state: WorkerState, now: Instant) {
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
                        placed: Resources::ZERO,
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
    fn worker_gone(&mut self, id: WorkerId) {
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

    /// The hold that keeps `w` from `job`, and the job that has it: a reservation of `w` by
    /// another job that `job` cannot backfill, or `job`'s own deferral to another worker. This is
    /// the one place holds are enforced.
    fn held(&self, job: &Job, w: &Worker) -> Option<(JobId, Hold)> {
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

    /// Whether `w` takes the job, or why not.
    fn refusal(&self, job: &Job, w: &Worker) -> Option<Refusal> {
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

    /// The shadow time of the reservation on `w`, once known.
    fn shadow(&self, w: &Worker) -> Option<Instant> {
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
    fn shadow_time(&self, w: &Worker, holder: JobId) -> Option<Instant> {
        let demand = self.waiting.get(&holder)?.spec.demand;
        let mut ends: Vec<(f64, Resources)> = Vec::with_capacity(w.jobs.len());
        for j in w.jobs.keys() {
            let r = &self.running[j];
            ends.push((self.expected_end(r)?, r.job.spec.demand));
        }
        ends.sort_by(|a, b| a.0.total_cmp(&b.0));
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
    fn refresh_shadows(&mut self) {
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

    /// Whether `job` has waited past the age limit.
    fn aged(&self, job: &Job) -> bool {
        self.config
            .age_limit
            .is_some_and(|a| self.now - job.since >= a)
    }

    /// The next job in scan order: aged jobs oldest first, then everything else by urgency.
    fn next_job(&self, cursor: &mut Cursor) -> Option<JobId> {
        loop {
            match *cursor {
                Cursor::Aged(after) => {
                    if self.config.age_limit.is_some() {
                        let lower = after.map_or(Bound::Unbounded, Bound::Excluded);
                        // `by_age` is in submission order, so the aged jobs are a prefix of it.
                        if let Some((&seq, &job)) =
                            self.by_age.range((lower, Bound::Unbounded)).next()
                            && self.aged(&self.waiting[&job])
                        {
                            *cursor = Cursor::Aged(Some(seq));
                            return Some(job);
                        }
                    }
                    *cursor = Cursor::Queue(None);
                }
                Cursor::Queue(after) => {
                    let lower = after.map_or(Bound::Unbounded, Bound::Excluded);
                    let (&key, &job) = self.queue.range((lower, Bound::Unbounded)).next()?;
                    *cursor = Cursor::Queue(Some(key));
                    if !self.aged(&self.waiting[&job]) {
                        return Some(job);
                    }
                }
            }
        }
    }

    /// When a busy worker's next slot is expected to free: the earliest projected end, or `None`
    /// if a running job's end is unknown on every slot.
    fn next_free(&self, w: &Worker, proj: &mut Projection) -> Option<f64> {
        let ends = proj.entry(w.state.id).or_insert_with(|| {
            let mut ends: Vec<f64> = w
                .jobs
                .keys()
                .map(|j| self.expected_end(&self.running[j]).unwrap_or(f64::INFINITY))
                .collect();
            ends.sort_by(f64::total_cmp);
            ends
        });
        ends.first().copied().filter(|e| e.is_finite())
    }

    /// Worker `w`'s rank for `job` under [`Config::score`].
    fn score(&self, job: &Job, w: &Worker) -> Score {
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
            let mut wait: Option<(f64, WorkerId)> = None;
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
                && eft < here - defer.min_gain * work
            {
                // Book the slot so that later deferrals in this scan see it taken.
                let start = self.next_free(&self.workers[&id], proj).unwrap();
                let ends = proj.get_mut(&id).unwrap();
                ends.remove(0);
                let at = ends.partition_point(|&e| e < eft);
                ends.insert(at, eft);
                return Pick::Defer(id, start.max(self.now));
            }
        }
        Pick::Place(place)
    }

    /// Move a waiting job onto a worker as its next attempt.
    fn place(&mut self, job: JobId, worker: WorkerId) {
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
    fn start(&mut self, job: JobId, worker: WorkerId) {
        let r = self.running.get_mut(&job).unwrap();
        r.job.attempts += 1;
        let attempt = r.job.attempts;
        let w = self
            .workers
            .get_mut(&worker)
            .expect("placing on an unknown worker");
        tick_occ(w, self.now);
        w.placed += r.job.spec.demand;
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

    /// Scan order as a sortable value: aged jobs first by age, then the rest by urgency.
    fn urgency(&self, job: &Job) -> (bool, Key) {
        if self.aged(job) {
            (
                false,
                Key {
                    terms: [0; ORDER_TERMS],
                    seq: job.key.seq,
                },
            )
        } else {
            (true, job.key)
        }
    }

    /// If `job` qualifies (waited long enough, no reservation yet), reserve a worker for it: the
    /// one with the most headroom if reservations are left, otherwise take over the reservation
    /// of the least urgent holder that is less urgent than `job` (its worker has been draining
    /// already). Returns whether `job` now holds a reservation.
    fn try_reserve(&mut self, job: JobId) -> bool {
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

    /// Whether a worker takes jobs other than its holder's: unreserved, or backfillable.
    fn open(&self, w: &Worker) -> bool {
        w.reserved_for.is_none() || self.shadow(w).is_some()
    }

    /// Whether worker `w` admits anything at all, by the admission bound.
    fn has_room(&self, w: &Worker) -> bool {
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
    fn dispatch(&mut self) {
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

    /// Start speculative attempts on workers left with a free slot (see
    /// [`Speculate`](crate::Speculate)).
    fn speculate(&mut self) {
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
                let mut best: Option<(f64, JobId)> = None;
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
                        || !self.admission.admits(&r.job.spec.demand, &w.view())
                    {
                        continue;
                    }
                    let (Some(end), Some(run)) = (self.expected_end(r), self.eta(&r.job, w)) else {
                        continue;
                    };
                    let end_here = now + run + cfg.restart_overhead;
                    if end - end_here < cfg.min_gain * run || end <= end_here {
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

    /// The next time, after now, that the passing of time alone changes what `dispatch` may do:
    /// a hold lapses, a job ages, or a job waits long enough to reserve.
    fn next_wakeup(&self) -> Option<Instant> {
        let now = self.now;
        // `by_age` is in submission order, so the first job whose deadline is still ahead has
        // the earliest one.
        let first_after = |wait: f64| {
            self.by_age
                .values()
                .map(|job| self.waiting[job].since + wait)
                .find(|&t| t > now)
        };
        let aging = self.config.age_limit.and_then(first_after);
        let reserving =
            (self.config.reservations.as_ref()).and_then(|c| first_after(c.reserve_after));
        self.holds
            .values()
            .filter_map(Hold::until)
            .filter(|&t| t > now)
            .chain(aging)
            .chain(reserving)
            .reduce(f64::min)
    }

    /// A worker's load as reported in the stats.
    fn load(&self, w: &Worker) -> WorkerLoad {
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
    fn stats(&self) -> PolicyStats {
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
    fn explain(&self, job: JobId) -> Option<String> {
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
            self.now - j.since
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
                msg +=
                    &format!("; waiting for faster worker {worker} (expected free at t={at:.0})");
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

impl Policy for Scheduler {
    /// Applied at once; stops and give-ups wait in the outbox for `poll`.
    fn handle(&mut self, input: Input, now: Instant) {
        self.now = now;
        match input {
            Input::Submit(spec) => self.submit(spec, now),
            Input::Done { job, attempt } => self.done(job, attempt),
            Input::Failed {
                job,
                attempt,
                kind,
                why,
            } => self.failed(job, attempt, kind, why),
            Input::Cancel(job) => self.cancel(job),
            Input::Worker(w) => self.worker_update(w, now),
            Input::WorkerGone(w) => self.worker_gone(w),
        }
    }

    /// One scan in urgency order, then speculation; the outbox, then the new starts.
    fn poll(&mut self, now: Instant) -> Vec<Output> {
        self.now = now;
        self.dispatch();
        self.speculate();
        std::mem::take(&mut self.outbox)
    }

    /// When a hold lapses, a job ages or a job may reserve, whichever is first.
    fn next_wakeup(&self) -> Option<Instant> {
        Scheduler::next_wakeup(self)
    }

    /// Every worker's reason to refuse it, summarised.
    fn explain(&self, job: JobId) -> Option<String> {
        Scheduler::explain(self, job)
    }

    /// Counters, holds and per-worker load.
    fn stats(&self) -> PolicyStats {
        Scheduler::stats(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RetryConfig, Speculate};

    const GB: u64 = 1_000_000_000;

    /// A worker of class "x" with a budget in GB.
    fn worker(id: WorkerId, slots: usize, budget_gb: u64) -> WorkerState {
        WorkerState::new(id, "x", slots, Resources::mem(budget_gb * GB))
    }

    /// A job with a demand in GB.
    fn job(id: JobId, gb: u64, group: u64) -> JobSpec {
        JobSpec::new(id, Resources::mem(gb * GB), group)
    }

    /// Handle `inputs` at `t`, then poll.
    fn feed(p: &mut Scheduler, t: Instant, inputs: impl IntoIterator<Item = Input>) -> Vec<Output> {
        for i in inputs {
            p.handle(i, t);
        }
        p.poll(t)
    }

    /// The `(job, worker)` of each start.
    fn starts(out: &[Output]) -> Vec<(JobId, WorkerId)> {
        out.iter()
            .filter_map(|o| match *o {
                Output::Start { job, worker, .. } => Some((job, worker)),
                _ => None,
            })
            .collect()
    }

    /// A done message.
    fn done(job: JobId, attempt: Attempt) -> Input {
        Input::Done { job, attempt }
    }

    /// A failure message of kind `kind`.
    fn fail(job: JobId, attempt: Attempt, kind: FailKind) -> Input {
        Input::Failed {
            job,
            attempt,
            kind,
            why: "test".into(),
        }
    }

    /// FIFO spreads jobs over the least loaded workers in arrival order.
    #[test]
    fn fifo_fills_least_loaded_first() {
        let mut p = Scheduler::new(Config::fifo());
        let mut inputs = vec![
            Input::Worker(worker(1, 4, 100)),
            Input::Worker(worker(2, 4, 100)),
        ];
        inputs.extend((0..4).map(|i| Input::Submit(job(i, 10, 0))));
        let out = feed(&mut p, 0.0, inputs);
        assert_eq!(starts(&out), vec![(0, 1), (1, 2), (2, 1), (3, 2)]);
        assert!(
            out.iter()
                .all(|o| matches!(o, Output::Start { attempt: 1, .. }))
        );
    }

    /// A preferred worker is chosen over a less loaded one.
    #[test]
    fn preference_wins_over_load() {
        let mut p = Scheduler::new(Config::default());
        let inputs = [
            Input::Worker(worker(1, 4, 100)),
            Input::Worker(worker(2, 4, 100)),
            Input::Submit(job(0, 10, 0)),
        ];
        feed(&mut p, 0.0, inputs);
        let j = job(1, 10, 0).prefer_worker(1);
        assert_eq!(starts(&feed(&mut p, 0.0, [Input::Submit(j)])), vec![(1, 1)]);
    }

    /// Explicit priority first, then the oldest group, then FIFO.
    #[test]
    fn priority_order_is_group_arrival_then_fifo() {
        let mut p = Scheduler::new(Config::default());
        p.handle(Input::Submit(job(10, 1, 7)), 0.0); // group 7 arrives first
        p.handle(Input::Submit(job(11, 1, 3)), 1.0);
        p.handle(Input::Submit(job(12, 1, 7)), 2.0);
        let mut urgent = job(13, 1, 3);
        urgent.priority = Some(-1);
        p.handle(Input::Submit(urgent), 3.0);
        p.handle(Input::Worker(worker(1, 1, 100)), 4.0);
        let mut order = Vec::new();
        let mut finished = Vec::new();
        for t in 0..4 {
            let out = starts(&feed(&mut p, 5.0 + t as f64, finished.drain(..)));
            assert_eq!(out.len(), 1);
            order.push(out[0].0);
            finished.push(done(out[0].0, 1));
        }
        assert_eq!(order, vec![13, 10, 12, 11]);
    }

    /// Best fit picks the worker left with the smallest share of its capacity free.
    #[test]
    fn best_fit_packs_tightly() {
        let mut p = Scheduler::new(Config::best_fit());
        let inputs = [
            Input::Worker(worker(1, 4, 100)),
            Input::Worker(worker(2, 4, 50)),
            Input::Submit(job(0, 1, 0)),
            Input::Submit(job(1, 1, 0)),
        ];
        // Both empty workers admit; the smaller one is the tighter fit.
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 2), (1, 2)]);
        let inputs = [Input::Submit(job(2, 50, 0)), Input::Submit(job(3, 1, 0))];
        // Then worker 1 has 49 GB free (49%), worker 2 has 47 GB (94%): worker 1 is fuller.
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(2, 1), (3, 1)]);
    }

    /// The tightest fit compares the bottleneck dimension.
    #[test]
    fn best_fit_ranks_by_the_bottleneck() {
        let mut p = Scheduler::new(Config::best_fit());
        // Worker 1 has most of its memory free but 20% of its device pool; worker 2 has 40% of
        // its memory free and all of its device pool.
        let w1 = WorkerState {
            reported_used: Resources::ZERO.with_dev(8 * GB),
            ..WorkerState::new(1, "x", 4, Resources::mem(100 * GB).with_dev(10 * GB))
        };
        let w2 = WorkerState {
            reported_used: Resources::mem(59 * GB),
            ..WorkerState::new(2, "x", 4, Resources::mem(100 * GB).with_dev(100 * GB))
        };
        let inputs = [
            Input::Worker(w1),
            Input::Worker(w2),
            Input::Submit(job(0, 1, 0)),
        ];
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 1)]);
    }

    /// A failed job is retried before less urgent jobs submitted after it, and keeps its age.
    #[test]
    fn retry_keeps_place_and_age() {
        let mut p = Scheduler::new(Config::fifo());
        let inputs = [
            Input::Worker(worker(1, 1, 100)),
            Input::Submit(job(0, 1, 0)),
            Input::Submit(job(1, 1, 0)),
        ];
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 1)]);
        assert_eq!(p.stats().longest_wait, Some((1, 0.0)));
        // Job 0 avoids worker 1 softly; it is the only live worker, so job 0 goes back there
        // ahead of job 1.
        let out = feed(&mut p, 5.0, [fail(0, 1, FailKind::Other)]);
        assert_eq!(
            out,
            vec![Output::Start {
                job: 0,
                attempt: 2,
                worker: 1
            }]
        );
        // Between the failure and the restart, job 0 counted as waiting since its submission.
        let mut q = Scheduler::new(Config::fifo());
        let inputs = [
            Input::Worker(worker(1, 1, 100)),
            Input::Worker(worker(2, 1, 100)),
            Input::Submit(job(0, 1, 0)),
            Input::Submit(job(1, 1, 0)),
            Input::Submit(job(2, 1, 0)),
        ];
        assert_eq!(starts(&feed(&mut q, 0.0, inputs)), vec![(0, 1), (1, 2)]);
        // Worker 1 frees, but job 0 avoids it while worker 2 lives: job 2 backfills it.
        let out = feed(&mut q, 5.0, [fail(0, 1, FailKind::Other)]);
        assert_eq!(starts(&out), vec![(2, 1)]);
        assert_eq!(q.stats().longest_wait, Some((0, 5.0)));
        let msg = q.explain(0).unwrap();
        assert!(msg.contains("failed 1 time(s), last on worker 1"), "{msg}");
        let out = feed(&mut q, 6.0, [done(1, 1)]);
        assert_eq!(
            out,
            vec![Output::Start {
                job: 0,
                attempt: 2,
                worker: 2
            }]
        );
    }

    /// Each failure adds its worker to the avoid list; once every live worker is avoided the soft
    /// list lapses; after `max_attempts` failures the job is given up.
    #[test]
    fn avoid_grows_then_gives_up() {
        let mut p = Scheduler::new(Config {
            retry: RetryConfig { max_attempts: 4 },
            ..Config::fifo()
        });
        let mut inputs: Vec<Input> = (1..=3).map(|w| Input::Worker(worker(w, 1, 100))).collect();
        inputs.push(Input::Submit(job(0, 1, 0)));
        let mut out = feed(&mut p, 0.0, inputs);
        let mut seen = Vec::new();
        for attempt in 1..=4 {
            let [
                Output::Start {
                    job: 0,
                    attempt: a,
                    worker,
                },
            ] = out[..]
            else {
                panic!("expected one start, got {out:?}");
            };
            assert_eq!(a, attempt);
            seen.push(worker);
            out = feed(&mut p, attempt as f64, [fail(0, a, FailKind::DeviceOom)]);
        }
        // Three distinct workers, then the lapsed soft list allows any (the first).
        assert_eq!(seen, vec![1, 2, 3, 1]);
        let [Output::GaveUp(g)] = &out[..] else {
            panic!("expected a give-up, got {out:?}");
        };
        assert_eq!(g.job, 0);
        assert_eq!(g.tried.len(), 4);
        assert!(g.retryable);
        assert_eq!(p.stats().waiting + p.stats().running, 0);
        assert_eq!(p.explain(0), None);
    }

    /// Retries avoid the workers tried softly, but the caller's Forbid stays hard.
    #[test]
    fn forbid_survives_retries() {
        let mut p = Scheduler::new(Config::fifo());
        let j = job(0, 1, 0).forbid_worker(1);
        let inputs = [
            Input::Worker(worker(1, 1, 100)),
            Input::Worker(worker(2, 1, 100)),
            Input::Submit(j),
        ];
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 2)]);
        // Worker 2 is now avoided softly; the only other live worker is forbidden, so the
        // avoidance lapses and the retry goes back to 2.
        let out = feed(&mut p, 1.0, [fail(0, 1, FailKind::Other)]);
        assert_eq!(
            out,
            vec![Output::Start {
                job: 0,
                attempt: 2,
                worker: 2
            }]
        );
    }

    /// The next wakeup covers aging and reservation deadlines, not only deferrals.
    #[test]
    fn next_wakeup_reports_aging_and_reserving() {
        let cfg = Config {
            age_limit: Some(100.0),
            ..Config::default()
        };
        let reserve_after = cfg.reservations.as_ref().unwrap().reserve_after;
        let mut p = Scheduler::new(cfg);
        assert_eq!(p.next_wakeup(), None);
        feed(&mut p, 10.0, [Input::Submit(job(0, 1, 0))]);
        assert_eq!(p.next_wakeup(), Some(10.0 + reserve_after.min(100.0)));
        let later = 10.0 + reserve_after.max(100.0);
        p.poll(10.0 + reserve_after.min(100.0));
        assert_eq!(p.next_wakeup(), (reserve_after != 100.0).then_some(later));
        p.poll(later);
        assert_eq!(p.next_wakeup(), None);
    }

    /// A give-up is retryable only if every attempt ran out of device memory.
    #[test]
    fn give_up_is_retryable_only_for_device_oom() {
        let mut p = Scheduler::new(Config {
            retry: RetryConfig { max_attempts: 2 },
            ..Config::fifo()
        });
        let inputs = [
            Input::Worker(worker(1, 1, 100)),
            Input::Submit(job(0, 1, 0)),
        ];
        feed(&mut p, 0.0, inputs);
        feed(&mut p, 1.0, [fail(0, 1, FailKind::DeviceOom)]);
        let out = feed(&mut p, 2.0, [fail(0, 2, FailKind::Timeout)]);
        let [Output::GaveUp(g)] = &out[..] else {
            panic!("expected a give-up, got {out:?}");
        };
        assert!(!g.retryable);
        assert_eq!(
            g.tried.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![FailKind::DeviceOom, FailKind::Timeout]
        );
    }

    /// A departing worker fails its attempts with `LinkDied`: the jobs are retried elsewhere
    /// without the caller resubmitting, or given up when out of attempts.
    #[test]
    fn worker_gone_fails_and_requeues() {
        let mut p = Scheduler::new(Config {
            retry: RetryConfig { max_attempts: 2 },
            ..Config::fifo()
        });
        let inputs = [
            Input::Worker(worker(1, 2, 100)),
            Input::Submit(job(0, 1, 0)),
            Input::Submit(job(1, 1, 0)),
        ];
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 1), (1, 1)]);
        let out = feed(&mut p, 1.0, [Input::WorkerGone(1)]);
        assert!(out.is_empty());
        assert_eq!((p.stats().waiting, p.stats().running), (2, 0));
        let out = feed(&mut p, 2.0, [Input::Worker(worker(2, 2, 100))]);
        assert_eq!(starts(&out), vec![(0, 2), (1, 2)]);
        assert!(
            out.iter()
                .all(|o| matches!(o, Output::Start { attempt: 2, .. }))
        );
        // The second loss exhausts both jobs' attempts.
        let out = feed(&mut p, 3.0, [Input::WorkerGone(2)]);
        assert_eq!(out.len(), 2);
        for o in &out {
            let Output::GaveUp(g) = o else {
                panic!("expected give-ups, got {out:?}")
            };
            assert!(g.tried.iter().all(|t| t.kind == FailKind::LinkDied));
            assert_eq!(g.tried[0].why, "worker 1 left");
        }
    }

    /// Messages about attempts that are not live change nothing.
    #[test]
    fn stale_messages_are_ignored() {
        let mut p = Scheduler::new(Config::fifo());
        let inputs = [
            Input::Worker(worker(1, 1, 100)),
            Input::Submit(job(0, 1, 0)),
        ];
        feed(&mut p, 0.0, inputs);
        feed(&mut p, 1.0, [fail(0, 1, FailKind::Other)]);
        // Attempt 2 runs; reports about attempt 1, unknown attempts and unknown jobs are stale.
        let stale = [
            done(0, 1),
            fail(0, 1, FailKind::Other),
            done(0, 7),
            done(9, 1),
            fail(9, 1, FailKind::Other),
            Input::Cancel(9),
            Input::WorkerGone(9),
        ];
        assert!(feed(&mut p, 2.0, stale).is_empty());
        let st = p.stats();
        assert_eq!((st.waiting, st.running, st.workers[0].running), (0, 1, 1));
        assert!(feed(&mut p, 3.0, [done(0, 2)]).is_empty());
        let st = p.stats();
        assert_eq!((st.running, st.workers[0].placed), (0, Resources::ZERO));
        // A late duplicate of the winning report is stale too.
        assert!(feed(&mut p, 4.0, [done(0, 2)]).is_empty());
        assert_eq!(p.explain(0), None);
    }

    /// Cancelling a running job stops its attempt; a waiting one is dropped silently.
    #[test]
    fn cancel_stops_running_attempts() {
        let mut p = Scheduler::new(Config::fifo());
        let inputs = [
            Input::Worker(worker(1, 1, 100)),
            Input::Submit(job(0, 1, 0)),
            Input::Submit(job(1, 1, 0)),
        ];
        feed(&mut p, 0.0, inputs);
        assert!(feed(&mut p, 1.0, [Input::Cancel(1)]).is_empty());
        let out = feed(&mut p, 2.0, [Input::Cancel(0)]);
        assert_eq!(
            out,
            vec![Output::Stop {
                job: 0,
                attempt: 1,
                worker: 1
            }]
        );
        let st = p.stats();
        assert_eq!((st.waiting, st.running, st.workers[0].running), (0, 0, 0));
    }

    /// A slow worker (speed 1) and, later, an idle fast one (speed 4), with speculation.
    fn speculating() -> Scheduler {
        let mut cfg = Config::fifo();
        cfg.speed.speculate = Some(Speculate::default());
        let mut p = Scheduler::new(cfg);
        let mut j = job(0, 1, 0);
        j.work = Some(100.0);
        let inputs = [Input::Worker(worker(1, 1, 100)), Input::Submit(j)];
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 1)]);
        let fast = WorkerState {
            speed: 4.0,
            ..worker(2, 1, 100)
        };
        let out = feed(&mut p, 1.0, [Input::Worker(fast)]);
        assert_eq!(
            out,
            vec![Output::Start {
                job: 0,
                attempt: 2,
                worker: 2
            }]
        );
        let st = p.stats();
        assert_eq!(st.running, 1);
        assert_eq!(
            st.workers.iter().map(|w| w.running).collect::<Vec<_>>(),
            vec![1, 1]
        );
        // At most one speculative attempt per job.
        assert!(p.poll(2.0).is_empty());
        p
    }

    /// An idle fast worker starts a second attempt of a job on a slow one; the first to finish
    /// wins and the other is stopped.
    #[test]
    fn speculation_first_done_wins() {
        let mut p = speculating();
        let out = feed(&mut p, 26.0, [done(0, 2)]);
        assert_eq!(
            out,
            vec![Output::Stop {
                job: 0,
                attempt: 1,
                worker: 1
            }]
        );
        assert!(p.stats().workers.iter().all(|w| w.running == 0));
        assert!(feed(&mut p, 27.0, [done(0, 1)]).is_empty());

        let mut p = speculating();
        let out = feed(&mut p, 10.0, [done(0, 1)]);
        assert_eq!(
            out,
            vec![Output::Stop {
                job: 0,
                attempt: 2,
                worker: 2
            }]
        );
    }

    /// A failing speculative attempt leaves the original running; cancelling stops both.
    #[test]
    fn speculation_failure_and_cancel() {
        let mut p = speculating();
        assert!(feed(&mut p, 5.0, [fail(0, 2, FailKind::Other)]).is_empty());
        assert_eq!(p.stats().running, 1);
        let out = feed(&mut p, 6.0, [Input::Cancel(0)]);
        assert_eq!(
            out,
            vec![Output::Stop {
                job: 0,
                attempt: 1,
                worker: 1
            }]
        );

        let mut p = speculating();
        let out = feed(&mut p, 5.0, [Input::Cancel(0)]);
        assert_eq!(starts(&out), vec![]);
        assert_eq!(out.len(), 2);
    }

    /// A failed speculative attempt does not count against the retry limit: with two rounds
    /// allowed, the original's failure after it is retried rather than given up.
    #[test]
    fn speculative_failure_is_not_a_round() {
        let mut p = speculating();
        p.config.retry.max_attempts = 2;
        assert!(feed(&mut p, 5.0, [fail(0, 2, FailKind::Other)]).is_empty());
        let out = feed(&mut p, 6.0, [fail(0, 1, FailKind::Other)]);
        assert!(
            matches!(
                out[..],
                [Output::Start {
                    job: 0,
                    attempt: 3,
                    ..
                }]
            ),
            "{out:?}"
        );
        let out = feed(&mut p, 7.0, [fail(0, 3, FailKind::Other)]);
        assert!(
            matches!(&out[..], [Output::GaveUp(g)] if g.tried.len() == 3),
            "{out:?}"
        );
    }

    /// The order in which a one-slot worker runs `jobs`, all submitted before it joins.
    fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
        let mut p = Scheduler::new(config);
        let n = jobs.len();
        feed(&mut p, 0.0, jobs.into_iter().map(Input::Submit));
        p.handle(Input::Worker(worker(1, 1, 100)), 0.0);
        let mut order = Vec::new();
        for t in 0..n {
            let out = starts(&p.poll(t as f64));
            assert_eq!(out.len(), 1, "{out:?}");
            order.push(out[0].0);
            p.handle(done(out[0].0, 1), t as f64 + 0.5);
        }
        order
    }

    /// Smith's rule: largest weight over work first; jobs without work last, in arrival order.
    #[test]
    fn wspt_orders_by_weight_over_work() {
        let spec = |id, weight, work| JobSpec {
            weight,
            work,
            ..job(id, 1, 0)
        };
        let jobs = vec![
            spec(0, 1.0, None),
            spec(1, 1.0, Some(10.0)),
            spec(2, 3.0, Some(10.0)),
            spec(3, 1.0, Some(2.0)),
            spec(4, 1.0, None),
        ];
        let order = run_order(Config::weighted_completion(), jobs);
        assert_eq!(order, vec![3, 2, 1, 0, 4]);
    }

    /// Jackson's rule: earliest due date first; jobs without one last.
    #[test]
    fn edd_orders_by_due_date() {
        let spec = |id, due| JobSpec {
            due,
            ..job(id, 1, 0)
        };
        let jobs = vec![spec(0, None), spec(1, Some(50.0)), spec(2, Some(-3.0))];
        assert_eq!(run_order(Config::lateness(), jobs), vec![2, 1, 0]);
    }

    /// The default order: explicit priority, then group arrival, ranks ignored; with
    /// [`OrderTerm::Rank`] between them, largest rank first. A repeated term changes nothing.
    #[test]
    fn default_order_is_priority_group() {
        let spec = |id, group, priority, rank| JobSpec {
            priority,
            rank,
            ..job(id, 1, group)
        };
        let jobs = || {
            vec![
                spec(0, 5, None, None),
                spec(1, 6, None, Some(2.0)),
                spec(2, 6, None, Some(9.0)),
                spec(3, 5, Some(-1), None),
            ]
        };
        assert_eq!(run_order(Config::default(), jobs()), vec![3, 0, 1, 2]);
        let mut repeated = Config::default();
        repeated
            .order
            .extend([OrderTerm::Priority, OrderTerm::Group]);
        assert_eq!(run_order(repeated, jobs()), vec![3, 0, 1, 2]);
        let ranked = Config {
            order: vec![OrderTerm::Priority, OrderTerm::Rank, OrderTerm::Group],
            ..Config::default()
        };
        assert_eq!(run_order(ranked, jobs()), vec![3, 2, 1, 0]);
        // Group before rank: the older group first.
        let group_first = Config {
            order: vec![OrderTerm::Group, OrderTerm::Rank],
            ..Config::default()
        };
        assert_eq!(run_order(group_first, jobs()), vec![0, 3, 2, 1]);
    }

    /// Requires of one kind are alternatives; of different kinds, all must hold. Forbid wins.
    #[test]
    fn requires_and_forbids() {
        let mut p = Scheduler::new(Config::fifo());
        let mut inputs: Vec<Input> = [(1, "a"), (2, "b"), (3, "c")]
            .into_iter()
            .map(|(id, class)| {
                Input::Worker(WorkerState::new(id, class, 4, Resources::mem(100 * GB)))
            })
            .collect();
        let either = job(0, 1, 0).require_class("c").require_class("b");
        let both = job(1, 1, 0)
            .require_class("a")
            .constrain(Strength::Require, Selector::Worker(2));
        let forbidden = job(2, 1, 0)
            .require_class("a")
            .constrain(Strength::Forbid, Selector::Class("a".into()));
        let worker_or = job(3, 1, 0)
            .constrain(Strength::Require, Selector::Worker(3))
            .constrain(Strength::Require, Selector::Worker(1))
            .forbid_worker(1);
        inputs.extend([either, both, forbidden, worker_or].map(Input::Submit));
        // Job 0 takes the less loaded of b and c; jobs 1 and 2 match nothing.
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 2), (3, 3)]);
        let msg = p.explain(1).unwrap();
        assert!(
            msg.contains("3 worker(s) excluded by its constraints"),
            "{msg}"
        );
    }

    /// A Prefer on a class ranks the whole class first; Loosest picks the emptiest worker.
    #[test]
    fn prefer_class_and_loosest() {
        let mut p = Scheduler::new(Config::default());
        let inputs = [
            Input::Worker(WorkerState::new(1, "a", 4, Resources::mem(100 * GB))),
            Input::Worker(WorkerState::new(2, "b", 4, Resources::mem(100 * GB))),
            Input::Submit(job(0, 1, 0).constrain(Strength::Prefer, Selector::Class("b".into()))),
        ];
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 2)]);
        let mut p = Scheduler::new(Config {
            score: vec![ScoreTerm::Loosest],
            ..Config::default()
        });
        let inputs = [
            Input::Worker(worker(1, 4, 100)),
            Input::Worker(worker(2, 4, 50)),
            Input::Submit(job(0, 1, 0)),
        ];
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 1)]);
    }

    /// Slots are a resource dimension: every job takes exactly one, whatever the caller wrote,
    /// and a full worker is explained as such.
    #[test]
    fn slots_are_a_hard_dimension() {
        let mut p = Scheduler::new(Config::fifo());
        let mut greedy = job(0, 1, 0);
        greedy.demand[SLOTS] = 5;
        let inputs = [
            Input::Worker(worker(1, 2, 100)),
            Input::Submit(greedy),
            Input::Submit(job(1, 1, 0)),
            Input::Submit(job(2, 1, 0)),
        ];
        assert_eq!(starts(&feed(&mut p, 0.0, inputs)), vec![(0, 1), (1, 1)]);
        let load = &p.stats().workers[0];
        assert_eq!(
            (load.running, load.placed[SLOTS], load.headroom[SLOTS]),
            (2, 2, Some(0))
        );
        let msg = p.explain(2).unwrap();
        assert!(msg.contains("slots full on 1 worker(s)"), "{msg}");
    }

    /// Without enough gain, nothing is speculated.
    #[test]
    fn speculation_needs_gain() {
        let mut cfg = Config::fifo();
        cfg.speed.speculate = Some(Speculate::default());
        let mut p = Scheduler::new(cfg);
        let mut j = job(0, 1, 0);
        j.work = Some(100.0);
        let inputs = [Input::Worker(worker(1, 1, 100)), Input::Submit(j)];
        feed(&mut p, 0.0, inputs);
        // At t=90 the slow attempt is expected to end at 100; a fresh one on a worker twice as
        // fast would end at 140.
        let fast = WorkerState {
            speed: 2.0,
            ..worker(2, 1, 100)
        };
        assert!(feed(&mut p, 90.0, [Input::Worker(fast)]).is_empty());
    }
}
