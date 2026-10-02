//! The one placement policy, [`Scheduler`].

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    ops::Bound,
};

use crate::{
    Admission, Config, DEV, DIMS, Fit, GroupOrder, Instant, JobId, JobSpec, MEM, Order, Policy,
    PolicyStats, ProductionAdmission, ReservationInfo, Resources, SpeedEstimator, SpeedPolicy,
    WorkerId, WorkerLoad, WorkerState, WorkerView,
};

/// Urgency: smaller is more urgent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    priority: i64,
    group: u64,
    /// Priority within the group (only with `group_first`).
    within: i64,
    seq: u64,
}

#[derive(Clone, Debug)]
struct Worker {
    state: WorkerState,
    running: usize,
    placed: Resources,
    jobs: BTreeSet<JobId>,
    /// The job holding a [`Hold::Reserve`] on this worker, if any.
    reserved_for: Option<JobId>,
    /// Effective speed: learned, or as reported.
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
            running: self.running,
            placed: self.placed,
        }
    }
}

#[derive(Clone, Debug)]
struct Waiting {
    spec: JobSpec,
    key: Key,
    since: Instant,
}

#[derive(Clone, Debug)]
struct Running {
    worker: WorkerId,
    started: Instant,
    spec: JobSpec,
    /// The worker's `occ` when it started.
    occ0: f64,
    /// Times it was preempted.
    preemptions: u32,
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
const DIM_NAMES: [&str; DIMS] = ["memory", "device memory"];

/// A resource vector in words for `explain`: host memory always, device memory when nonzero.
fn gb_list(r: &Resources) -> String {
    let mut parts = vec![format!("{:.2} GB", r[MEM] as f64 / GB)];
    if r[DEV] > 0 {
        parts.push(format!("{:.2} GB device", r[DEV] as f64 / GB));
    }
    parts.join(" + ")
}

/// A float as a totally ordered integer key (for score tuples).
fn ordered(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    b ^ (((b >> 63) as u64) >> 1) as i64
}

/// Bring a worker's concurrency integral up to `now`.
fn tick_occ(w: &mut Worker, now: Instant) {
    if now > w.occ_at {
        w.occ += w.running as f64 * (now - w.occ_at);
        w.occ_at = now;
    }
}

/// Whether `job`'s class pin allows `w`.
fn class_allows(job: &JobSpec, w: &Worker) -> bool {
    job.class.as_ref().is_none_or(|c| *c == w.state.class)
}

/// The placement policy: list scheduling with admission, reservations and backfill, configured
/// by a [`Config`].
///
/// - Jobs are considered in [`Config::order`], aged jobs first ([`Config::age_limit`]).
/// - A job takes a worker only if no more urgent waiting job is admitted there; among the
///   workers that admit it, the one [`Config::fit`] and [`Config::speed`] rank first.
/// - [`Config::reservations`] drain a worker for a starving job; every other worker keeps
///   admitting less urgent jobs.
///
/// `dispatch` scans waiting jobs in order and gives each one a worker if any admits it. Because the
/// scan is in urgency order and admission is monotone in load, a job is placed on a worker only if
/// every more urgent waiting job was refused there -- the priority invariant -- without any
/// explicit check. The one event that can make an already-refused worker admissible mid-scan is
/// the release of a hold (a reservation whose holder is placed); the scan restarts from the top
/// when that happens.
pub struct Scheduler {
    config: Config,
    admission: Box<dyn Admission + Send>,
    workers: BTreeMap<WorkerId, Worker>,
    queue: BTreeMap<Key, JobId>,
    by_age: BTreeMap<u64, JobId>,
    waiting: HashMap<JobId, Waiting>,
    running: HashMap<JobId, Running>,
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
    /// Learned speeds (with [`SpeedConfig::learn`](crate::SpeedConfig::learn)).
    learned: Option<SpeedEstimator>,
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
    pub fn with_admission(config: Config, admission: impl Admission + Send + 'static) -> Self {
        Self {
            learned: config.speed.learn.map(SpeedEstimator::new),
            config,
            admission: Box::new(admission),
            workers: BTreeMap::new(),
            queue: BTreeMap::new(),
            by_age: BTreeMap::new(),
            waiting: HashMap::new(),
            running: HashMap::new(),
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

    /// The expected run time of `job` on `w`: its work over the worker's speed, `None` without a
    /// work estimate. Every run-time estimate goes through here.
    fn eta(&self, job: &JobSpec, w: &Worker) -> Option<f64> {
        job.work.map(|work| work / w.speed)
    }

    /// When a running job is expected to end: its start plus its [`eta`](Self::eta), or, once
    /// that has passed, as far beyond now as it has run (an overrunning job is assumed half done,
    /// StarPU's rule). `None` if its run time is unknown.
    fn expected_end(&self, r: &Running) -> Option<Instant> {
        let end = r.started + self.eta(&r.spec, self.workers.get(&r.worker)?)?;
        Some(if end > self.now {
            end
        } else {
            self.now + (self.now - r.started).max(0.0)
        })
    }

    /// Whether `job`'s constraints (class, avoid list) allow `w` at all. A soft avoid list
    /// lapses while every live worker of the job's class is on it; that depends on the worker
    /// set only, not on load, so admission stays monotone.
    fn eligible(&self, job: &JobSpec, w: &Worker) -> bool {
        if !class_allows(job, w) {
            return false;
        }
        if !job.avoid.contains(&w.state.id) {
            return true;
        }
        job.avoid_soft
            && !self.workers.values().any(|o| {
                o.state.slots > 0 && class_allows(job, o) && !job.avoid.contains(&o.state.id)
            })
    }

    /// Queue a job under its urgency key, recording its group's first arrival.
    fn submit(&mut self, spec: JobSpec, now: Instant) {
        self.now = now;
        if self.waiting.contains_key(&spec.id) || self.running.contains_key(&spec.id) {
            return;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        let key = match self.config.order {
            Order::Fifo => Key {
                priority: 0,
                group: 0,
                within: 0,
                seq,
            },
            Order::Priority {
                default_priority,
                group_order,
                group_first,
            } => {
                let group = match group_order {
                    GroupOrder::Arrival => *self.groups.entry(spec.group).or_insert(seq),
                    GroupOrder::Id => spec.group,
                };
                let priority = spec.priority.unwrap_or(default_priority);
                if group_first {
                    Key {
                        priority: 0,
                        group,
                        within: priority,
                        seq,
                    }
                } else {
                    Key {
                        priority,
                        group,
                        within: 0,
                        seq,
                    }
                }
            }
        };
        self.queue.insert(key, spec.id);
        self.by_age.insert(seq, spec.id);
        self.waiting.insert(
            spec.id,
            Waiting {
                spec,
                key,
                since: now,
            },
        );
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
    fn remove_waiting(&mut self, job: JobId) -> Option<Waiting> {
        self.release_hold(job);
        let j = self.waiting.remove(&job)?;
        self.queue.remove(&j.key);
        self.by_age.remove(&j.key.seq);
        Some(j)
    }

    /// Release a running job's slot and demand; false if it was not running.
    fn finish_running(&mut self, job: JobId) -> bool {
        let Some(r) = self.running.remove(&job) else {
            return false;
        };
        if let Some(w) = self.workers.get_mut(&r.worker) {
            tick_occ(w, self.now);
            w.running -= 1;
            w.placed -= r.spec.demand;
            w.jobs.remove(&job);
        }
        true
    }

    /// Drop a waiting job, or release a running one.
    fn cancel(&mut self, job: JobId) {
        if self.remove_waiting(job).is_none() {
            self.finish_running(job);
        }
    }

    /// Release a running job (or drop it, if it was still waiting).
    fn completed(&mut self, job: JobId, now: Instant) {
        self.now = now;
        self.learn_from(job, now);
        if !self.finish_running(job) {
            self.remove_waiting(job);
        }
    }

    /// The speed to use for worker `id` of `class` that reports `reported`: learned, once there
    /// are enough samples, else as reported.
    fn worker_speed(&mut self, id: WorkerId, class: &str, reported: f64) -> f64 {
        match self.learned.as_mut() {
            Some(e) => e.speed(id, class, reported),
            None => crate::speed::sane(reported),
        }
    }

    /// A job finished: learn its worker's speed from its duration and the worker's mean
    /// concurrency meanwhile.
    fn learn_from(&mut self, job: JobId, now: Instant) {
        if self.learned.is_none() {
            return;
        }
        let Some(r) = self.running.get(&job) else {
            return;
        };
        let (Some(work), Some(w)) = (r.spec.work, self.workers.get_mut(&r.worker)) else {
            return;
        };
        tick_occ(w, now);
        let dt = now - r.started;
        let k = if dt > 0.0 { (w.occ - r.occ0) / dt } else { 1.0 };
        let (id, class) = (w.state.id, w.state.class.clone());
        if !self
            .learned
            .as_mut()
            .unwrap()
            .observe(id, &class, work, dt, k)
        {
            return;
        }
        // The class estimate moved too: refresh every worker of the class.
        let ids: Vec<WorkerId> = self
            .workers
            .values()
            .filter(|w| w.state.class == class)
            .map(|w| w.state.id)
            .collect();
        for id in ids {
            let reported = self.workers[&id].state.speed;
            let speed = self.worker_speed(id, &class, reported);
            self.workers.get_mut(&id).unwrap().speed = speed;
        }
    }

    /// Speed as an ordering key (more negative is faster). With learning and a resolution, speeds
    /// within one resolution step of each other compare equal, so per-worker noise does not
    /// override load.
    fn speed_rank(&self, w: &Worker) -> i64 {
        let res = self.learned.as_ref().map_or(0.0, |e| e.config().resolution);
        if res > 0.0 {
            -(w.speed.ln() / res.ln_1p()).round() as i64
        } else {
            ordered(-w.speed)
        }
    }

    /// Add a worker or replace its reported state, keeping its placements.
    fn worker_update(&mut self, state: WorkerState, now: Instant) {
        self.now = now;
        let speed = self.worker_speed(state.id, &state.class, state.speed);
        match self.workers.get_mut(&state.id) {
            Some(w) => {
                // A reservation counted against the old class (per-class limits) must not move to
                // the new one; its holder reserves again at the next dispatch.
                let moved = (w.state.class != state.class)
                    .then_some(w.reserved_for)
                    .flatten();
                w.state = state;
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
                        running: 0,
                        placed: Resources::ZERO,
                        jobs: BTreeSet::new(),
                        reserved_for: None,
                        speed,
                        occ: 0.0,
                        occ_at: now,
                    },
                );
            }
        }
    }

    /// Forget a worker, its running jobs and every hold on it.
    fn worker_gone(&mut self, id: WorkerId, now: Instant) {
        self.now = now;
        let Some(w) = self.workers.remove(&id) else {
            return;
        };
        for job in &w.jobs {
            self.running.remove(job);
        }
        self.holds.retain(|_, h| h.worker() != id);
    }

    /// The hold that keeps `w` from `job`, and the job that has it: a reservation of `w` by
    /// another job that `job` cannot backfill, or `job`'s own deferral to another worker. This is
    /// the one place holds are enforced.
    fn held(&self, job: &JobSpec, w: &Worker) -> Option<(JobId, Hold)> {
        if let Some(holder) = w.reserved_for
            && holder != job.id
            && !self.shadow_backfills(job, w)
        {
            return Some((holder, self.holds[&holder]));
        }
        match self.holds.get(&job.id) {
            Some(&h @ Hold::Defer { worker, .. }) if worker != w.state.id => Some((job.id, h)),
            _ => None,
        }
    }

    /// Whether `w` takes the job, or why not.
    fn refusal(&self, job: &JobSpec, w: &Worker) -> Option<Refusal> {
        if !self.eligible(job, w) {
            return Some(Refusal::Ineligible);
        }
        if let Some((by, hold)) = self.held(job, w) {
            return Some(Refusal::Held(by, hold));
        }
        if !self.admission.admits(&job.demand, &w.view()) {
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
    fn shadow_backfills(&self, job: &JobSpec, w: &Worker) -> bool {
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
        for j in &w.jobs {
            let r = &self.running[j];
            ends.push((self.expected_end(r)?, r.spec.demand));
        }
        ends.sort_by(|a, b| a.0.total_cmp(&b.0));
        // The projection assumes a released job frees what it was placed with; the reported
        // usage cannot be predicted, so the hypothetical worker reports none above its baseline.
        let state = WorkerState {
            reported_used: Resources::ZERO,
            ..w.state.clone()
        };
        let fits = |running: usize, placed: Resources| {
            let view = WorkerView {
                state: &state,
                running,
                placed,
            };
            self.admission.admits(&demand, &view)
        };
        let (mut running, mut placed) = (w.running, w.placed);
        if fits(running, placed) {
            return Some(self.now);
        }
        for (end, d) in ends {
            running -= 1;
            placed -= d;
            if fits(running, placed) {
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
    fn aged(&self, job: &Waiting) -> bool {
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
                .iter()
                .map(|j| self.expected_end(&self.running[j]).unwrap_or(f64::INFINITY))
                .collect();
            ends.sort_by(f64::total_cmp);
            ends
        });
        ends.first().copied().filter(|e| e.is_finite())
    }

    /// The best worker that takes `job`, or a busy faster worker to wait for, if any.
    fn choose(&self, job: &Waiting, proj: &mut Projection) -> Pick {
        let speed_first = self.config.speed.policy != SpeedPolicy::Oblivious;
        // Smallest tuple wins; the worker id makes the order total (determinism).
        let mut best: Option<((i64, i64, bool, usize), WorkerId)> = None;
        for (&id, w) in &self.workers {
            if self.refusal(&job.spec, w).is_some() {
                continue;
            }
            let preferred = job.spec.prefer.contains(&id);
            let speed_key = if speed_first { self.speed_rank(w) } else { 0 };
            let fit = match self.config.fit {
                Fit::LeastLoaded => 0,
                Fit::Tightest { prefer_penalty } => {
                    let after = w.view().free_share(&job.spec.demand);
                    ordered(after - if preferred { prefer_penalty } else { 0.0 })
                }
            };
            let score = (speed_key, fit, !preferred, w.running);
            if best.as_ref().is_none_or(|(b, _)| score < *b) {
                best = Some((score, id));
            }
        }
        let Some((_, place)) = best else {
            return Pick::Nothing;
        };
        if let SpeedPolicy::EarliestFinish(Some(defer)) = self.config.speed.policy
            && let Some(run_here) = self.eta(&job.spec, &self.workers[&place])
            && self.reserved(job.spec.id).is_none()
            && !self.aged(job)
            && self.now - job.since < defer.max_wait
        {
            let work = job.spec.work.unwrap_or_default();
            let here = self.now + run_here;
            let mut wait: Option<(f64, WorkerId)> = None;
            for (&id, w) in &self.workers {
                // Only workers that refuse for want of a slot, and would admit with one free.
                if w.speed <= self.workers[&place].speed
                    || w.running < w.state.slots
                    || !self.eligible(&job.spec, w)
                    || w.reserved_for.is_some_and(|h| h != job.spec.id)
                {
                    continue;
                }
                let view = WorkerView {
                    running: w.state.slots.saturating_sub(1),
                    ..w.view()
                };
                if w.state.slots == 0 || !self.admission.admits(&job.spec.demand, &view) {
                    continue;
                }
                let (Some(start), Some(run)) = (self.next_free(w, proj), self.eta(&job.spec, w))
                else {
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

    /// Move a waiting job onto a worker and record the placement.
    fn place(&mut self, job: JobId, worker: WorkerId, out: &mut Vec<(JobId, WorkerId)>) {
        if self.reserved(job) == Some(worker) {
            self.last_dispatch_holders.push(job);
        }
        let j = self
            .remove_waiting(job)
            .expect("placing a job that is not waiting");
        let w = self
            .workers
            .get_mut(&worker)
            .expect("placing on an unknown worker");
        tick_occ(w, self.now);
        w.running += 1;
        w.placed += j.spec.demand;
        w.jobs.insert(job);
        let occ0 = w.occ;
        self.running.insert(
            job,
            Running {
                worker,
                started: self.now,
                spec: j.spec,
                preemptions: 0,
                occ0,
            },
        );
        self.placements_total += 1;
        out.push((job, worker));
    }

    /// Scan order as a sortable value: aged jobs first by age, then the rest by urgency.
    fn urgency(&self, job: &Waiting) -> (bool, Key) {
        if self.aged(job) {
            (
                false,
                Key {
                    priority: 0,
                    group: 0,
                    within: 0,
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
        // Most headroom; then fastest (when speed-aware); then preferred; then fewest running;
        // then smallest id.
        let speed_first = self.config.speed.policy != SpeedPolicy::Oblivious;
        let mut best: Option<((i64, i64, bool, usize), WorkerId)> = None;
        for (&id, w) in &self.workers {
            if w.reserved_for.is_some()
                || w.state.slots == 0
                || class_full(&w.state.class)
                || !self.eligible(&j.spec, w)
            {
                continue;
            }
            let score = (
                ordered(-w.view().free_share(&Resources::ZERO)),
                if speed_first { self.speed_rank(w) } else { 0 },
                !j.spec.prefer.contains(&id),
                w.running,
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
            .filter(|r| self.eligible(&j.spec, &self.workers[&r.1]))
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

    /// Classes with an open worker that has a free slot.
    fn open_classes(&self) -> BTreeSet<String> {
        self.workers
            .values()
            .filter(|w| w.running < w.state.slots && self.open(w))
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
    fn dispatch(&mut self, now: Instant) -> Vec<(JobId, WorkerId)> {
        self.now = now;
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
                w.state.slots == 0 || !self.eligible(&self.waiting[job].spec, w)
            })
            .map(|(&job, _)| job)
            .collect();
        for job in dead {
            self.release_hold(job);
        }
        let mut proj = Projection::new();
        let mut out = Vec::new();
        'scan: loop {
            if !self.workers.values().any(|w| w.running < w.state.slots) {
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
                // Cheap pruning: the admission bound, and a class pin with no free slot of its
                // class.
                let hopeful = holder
                    || (bound.is_some_and(|b| j.spec.demand.fits_within(&b))
                        && j.spec.class.as_ref().is_none_or(|c| classes.contains(c)));
                let pick = if hopeful {
                    self.choose(j, &mut proj)
                } else {
                    Pick::Nothing
                };
                match pick {
                    Pick::Defer(worker, at) => {
                        let SpeedPolicy::EarliestFinish(Some(d)) = self.config.speed.policy else {
                            unreachable!("deferral without a Defer config")
                        };
                        let until = j.since + d.max_wait;
                        self.holds.insert(job, Hold::Defer { worker, at, until });
                        if !self.deferred_any.contains(&job) {
                            self.deferred_any.push(job);
                        }
                    }
                    Pick::Place(w) => {
                        self.place(job, w, &mut out);
                        if holder {
                            // Its worker is open again: more urgent jobs refused there because of
                            // the reservation must get the first look.
                            continue 'scan;
                        }
                        if !self.workers.values().any(|w| w.running < w.state.slots) {
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
                                self.place(job, w, &mut out);
                                continue 'scan;
                            }
                            bound = self.open_bound();
                            classes = self.open_classes();
                        }
                    }
                }
            }
        }
        out
    }

    /// Placements, then spoliation (see [`Spoliation`](crate::Spoliation)).
    fn dispatch_full(&mut self, now: Instant) -> crate::Dispatch {
        let start = self.dispatch(now);
        let preempt = self.spoliate();
        crate::Dispatch { start, preempt }
    }

    /// Move running jobs from slower workers to faster ones with free slots left after dispatch.
    fn spoliate(&mut self) -> Vec<crate::Preemption> {
        let Some(cfg) = self.config.speed.spoliation else {
            return Vec::new();
        };
        let now = self.now;
        let mut out = Vec::new();
        let ids: Vec<WorkerId> = self.workers.keys().copied().collect();
        for to in ids {
            loop {
                let w = &self.workers[&to];
                if w.running >= w.state.slots || w.reserved_for.is_some() {
                    break;
                }
                // Candidates: run time known, on a slower worker, allowed and admitted here.
                let mut best: Option<(f64, JobId)> = None;
                for (&job, r) in &self.running {
                    let Some(v) = self.workers.get(&r.worker) else {
                        continue;
                    };
                    if self.speed_rank(v) <= self.speed_rank(w)
                        || r.preemptions >= cfg.max_per_job
                        || !self.eligible(&r.spec, w)
                        || !self.admission.admits(&r.spec.demand, &w.view())
                    {
                        continue;
                    }
                    let (Some(end_v), Some(run)) = (self.expected_end(r), self.eta(&r.spec, w))
                    else {
                        continue;
                    };
                    let end_w = now + run + cfg.restart_overhead;
                    if end_v - end_w < cfg.min_gain * run || end_v <= end_w {
                        continue;
                    }
                    if best.is_none_or(|(e, j)| end_v > e || (end_v == e && job < j)) {
                        best = Some((end_v, job));
                    }
                }
                let Some((_, job)) = best else { break };
                let r = self.running.get_mut(&job).unwrap();
                let from = r.worker;
                r.worker = to;
                r.started = now;
                r.preemptions += 1;
                let demand = r.spec.demand;
                let v = self.workers.get_mut(&from).unwrap();
                tick_occ(v, now);
                v.running -= 1;
                v.placed -= demand;
                v.jobs.remove(&job);
                let w = self.workers.get_mut(&to).unwrap();
                tick_occ(w, now);
                w.running += 1;
                w.placed += demand;
                w.jobs.insert(job);
                let occ0 = w.occ;
                self.running.get_mut(&job).unwrap().occ0 = occ0;
                out.push(crate::Preemption { job, from, to });
            }
        }
        out
    }

    /// The next time a hold lapses by itself, after now.
    fn next_wakeup(&self) -> Option<Instant> {
        self.holds
            .values()
            .filter_map(Hold::until)
            .filter(|&t| t > self.now)
            .reduce(f64::min)
    }

    /// A worker's load as reported in the stats.
    fn load(&self, w: &Worker) -> WorkerLoad {
        WorkerLoad {
            id: w.state.id,
            class: w.state.class.clone(),
            slots: w.state.slots,
            running: w.running,
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
            return Some(format!("job {job} is running on worker {}", r.worker));
        }
        let j = self.waiting.get(&job)?;
        let ahead = self.queue.range(..j.key).count();
        let mut msg = format!(
            "job {job} (demand {}, group {}) waiting {:.0}s, {ahead} more urgent job(s) waiting",
            gb_list(&j.spec.demand),
            j.spec.group,
            self.now - j.since
        );
        match self.holds.get(&job) {
            Some(Hold::Reserve { worker, .. }) => {
                let w = &self.workers[worker];
                msg += &format!(
                    "; holds the reservation on worker {} (draining: {}/{} running, used {})",
                    w.state.id,
                    w.running,
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
        let (mut full, mut excluded) = (0, 0);
        // Per dimension: workers short of it, and the one with the most headroom there.
        let mut short = [0usize; DIMS];
        let mut best_short: [Option<(i64, WorkerId)>; DIMS] = [None; DIMS];
        let mut reserved = Vec::new();
        let mut takers = Vec::new();
        for (&id, w) in &self.workers {
            match self.refusal(&j.spec, w) {
                None => takers.push(id),
                Some(Refusal::Ineligible) => excluded += 1,
                Some(Refusal::Held(h, Hold::Reserve { .. })) => {
                    reserved.push(format!("worker {id} for job {h}"))
                }
                Some(Refusal::Held(_, Hold::Defer { .. })) => {}
                Some(Refusal::Admission) if w.running >= w.state.slots => full += 1,
                Some(Refusal::Admission) => {
                    let view = w.view();
                    let headroom = view.headroom();
                    for d in view.short(&j.spec.demand) {
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
            msg += &format!("; {excluded} worker(s) excluded by its class or avoid list");
        }
        if !reserved.is_empty() {
            msg += &format!("; reserved: {}", reserved.join(", "));
        }
        if !takers.is_empty() {
            msg += &format!(
                "; admitted on worker(s) {takers:?} (placed at the next dispatch unless a more \
                 urgent job takes the slot)"
            );
        }
        Some(msg)
    }
}

impl Policy for Scheduler {
    /// Queued under its urgency key.
    fn submit(&mut self, job: JobSpec, now: Instant) {
        Scheduler::submit(self, job, now)
    }

    /// Dropped, or released, with its hold.
    fn cancel(&mut self, job: JobId) {
        Scheduler::cancel(self, job)
    }

    /// Added or updated, keeping its placements.
    fn worker_update(&mut self, w: WorkerState, now: Instant) {
        Scheduler::worker_update(self, w, now)
    }

    /// Forgotten with its running jobs and the holds on it.
    fn worker_gone(&mut self, w: WorkerId, now: Instant) {
        Scheduler::worker_gone(self, w, now)
    }

    /// Released, learning from its duration.
    fn completed(&mut self, job: JobId, now: Instant) {
        Scheduler::completed(self, job, now)
    }

    /// One scan in urgency order.
    fn dispatch(&mut self, now: Instant) -> Vec<(JobId, WorkerId)> {
        Scheduler::dispatch(self, now)
    }

    /// Every worker's reason to refuse it, summarised.
    fn explain(&self, job: JobId) -> Option<String> {
        Scheduler::explain(self, job)
    }

    /// Counters, holds and per-worker load.
    fn stats(&self) -> PolicyStats {
        Scheduler::stats(self)
    }

    /// When the earliest deferral lapses.
    fn next_wakeup(&self) -> Option<Instant> {
        Scheduler::next_wakeup(self)
    }

    /// One scan, then spoliation.
    fn dispatch_full(&mut self, now: Instant) -> crate::Dispatch {
        Scheduler::dispatch_full(self, now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    /// A worker of class "x" with a budget in GB.
    fn worker(id: WorkerId, slots: usize, budget_gb: u64) -> WorkerState {
        WorkerState::new(id, "x", slots, Resources::mem(budget_gb * GB))
    }

    /// A job with a demand in GB.
    fn job(id: JobId, gb: u64, group: u64) -> JobSpec {
        JobSpec::new(id, Resources::mem(gb * GB), group)
    }

    /// FIFO spreads jobs over the least loaded workers in arrival order.
    #[test]
    fn fifo_fills_least_loaded_first() {
        let mut p = Scheduler::new(Config::fifo());
        p.worker_update(worker(1, 4, 100), 0.0);
        p.worker_update(worker(2, 4, 100), 0.0);
        for i in 0..4 {
            p.submit(job(i, 10, 0), 0.0);
        }
        let out = p.dispatch(0.0);
        assert_eq!(out, vec![(0, 1), (1, 2), (2, 1), (3, 2)]);
    }

    /// A preferred worker is chosen over a less loaded one.
    #[test]
    fn preference_wins_over_load() {
        let mut p = Scheduler::new(Config::default());
        p.worker_update(worker(1, 4, 100), 0.0);
        p.worker_update(worker(2, 4, 100), 0.0);
        p.submit(job(0, 10, 0), 0.0);
        p.dispatch(0.0);
        let mut j = job(1, 10, 0);
        j.prefer = vec![1];
        p.submit(j, 0.0);
        assert_eq!(p.dispatch(0.0), vec![(1, 1)]);
    }

    /// Explicit priority first, then the oldest group, then FIFO.
    #[test]
    fn priority_order_is_group_arrival_then_fifo() {
        let mut p = Scheduler::new(Config::default());
        p.submit(job(10, 1, 7), 0.0); // group 7 arrives first
        p.submit(job(11, 1, 3), 1.0);
        p.submit(job(12, 1, 7), 2.0);
        let mut urgent = job(13, 1, 3);
        urgent.priority = Some(-1);
        p.submit(urgent, 3.0);
        p.worker_update(worker(1, 1, 100), 4.0);
        let mut order = Vec::new();
        for t in 0..4 {
            let out = p.dispatch(5.0 + t as f64);
            assert_eq!(out.len(), 1);
            order.push(out[0].0);
            p.completed(out[0].0, 5.5 + t as f64);
        }
        assert_eq!(order, vec![13, 10, 12, 11]);
    }

    /// Best fit picks the worker left with the smallest share of its capacity free.
    #[test]
    fn best_fit_packs_tightly() {
        let mut p = Scheduler::new(Config::best_fit());
        p.worker_update(worker(1, 4, 100), 0.0);
        p.worker_update(worker(2, 4, 50), 0.0);
        p.submit(job(0, 1, 0), 0.0);
        p.submit(job(1, 1, 0), 0.0);
        // Both empty workers admit; the smaller one is the tighter fit.
        assert_eq!(p.dispatch(0.0), vec![(0, 2), (1, 2)]);
        p.submit(job(2, 50, 0), 0.0);
        p.submit(job(3, 1, 0), 0.0);
        // Then worker 1 has 49 GB free (49%), worker 2 has 47 GB (94%): worker 1 is fuller.
        assert_eq!(p.dispatch(0.0), vec![(2, 1), (3, 1)]);
    }

    /// The tightest fit compares the bottleneck dimension.
    #[test]
    fn best_fit_ranks_by_the_bottleneck() {
        let mut p = Scheduler::new(Config::best_fit());
        // Worker 1 has most of its memory free but 20% of its device pool; worker 2 has 40% of
        // its memory free and all of its device pool.
        p.worker_update(
            WorkerState {
                reported_used: Resources::ZERO.with_dev(8 * GB),
                ..WorkerState::new(1, "x", 4, Resources::mem(100 * GB).with_dev(10 * GB))
            },
            0.0,
        );
        p.worker_update(
            WorkerState {
                reported_used: Resources::mem(59 * GB),
                ..WorkerState::new(2, "x", 4, Resources::mem(100 * GB).with_dev(100 * GB))
            },
            0.0,
        );
        p.submit(job(0, 1, 0), 0.0);
        assert_eq!(p.dispatch(0.0), vec![(0, 1)]);
    }
}
