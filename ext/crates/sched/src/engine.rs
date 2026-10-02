//! The placement engine shared by every policy, and the policy types themselves.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Bound,
};

use crate::{
    Admission, Instant, JobId, JobSpec, Policy, PolicyStats, ProductionAdmission, ReservationInfo,
    Resources, WorkerId, WorkerLoad, WorkerState, WorkerView,
};

/// Configuration for [`Greedy`]. It has no parameters.
#[derive(Clone, Debug, Default)]
pub struct GreedyConfig {}

/// Configuration for [`PriorityBackfill`] (and the backfill part of [`BestFit`] and [`Lanes`]).
#[derive(Clone, Debug)]
pub struct BackfillConfig {
    /// A job that has waited at least this long (seconds) and is admitted nowhere may reserve a
    /// worker. Default 60.
    pub reserve_after: f64,
    /// Maximum number of simultaneous reservations (per worker class if
    /// `per_class_reservations`). Default 1. Zero disables reservations, which turns the policy
    /// into plain priority order with backfill (and allows starvation).
    pub max_reservations: usize,
    /// Count `max_reservations` per worker class instead of globally. Default false.
    pub per_class_reservations: bool,
    /// The priority of jobs whose [`JobSpec::priority`] is `None`. Default 0.
    pub default_priority: i64,
    /// Aging: a job that has waited at least this long (seconds) becomes more urgent than every
    /// job that has not, oldest first. Strict priority starves a job for as long as more urgent
    /// jobs keep arriving; with group-arrival order that is bounded by the older groups' work,
    /// but with arbitrary priorities (e.g. DAG ranks) it is not, and this bounds it. Default
    /// `None` (strict priority).
    pub age_limit: Option<f64>,
}

impl Default for BackfillConfig {
    /// The defaults documented on each field.
    fn default() -> Self {
        Self {
            reserve_after: 60.0,
            max_reservations: 1,
            per_class_reservations: false,
            default_priority: 0,
            age_limit: None,
        }
    }
}

/// Configuration for [`BestFit`].
#[derive(Clone, Debug, Default)]
pub struct BestFitConfig {
    /// Priority, reservation and backfill parameters.
    pub backfill: BackfillConfig,
    /// How much a preferred worker ([`JobSpec::prefer`]) is favoured, in bytes of headroom: a
    /// preferred worker competes as if its headroom after placement were this much smaller. 0
    /// (the default) makes preference a pure tie-breaker.
    pub prefer_penalty: u64,
}

/// Which workers are big lanes, for [`Lanes`].
#[derive(Clone, Debug, PartialEq)]
pub enum LaneSet {
    /// Every worker of these classes.
    Classes(Vec<String>),
    /// These workers.
    Workers(Vec<WorkerId>),
}

/// Configuration for [`Lanes`].
#[derive(Clone, Debug)]
pub struct LanesConfig {
    /// Priority, reservation and backfill parameters.
    pub backfill: BackfillConfig,
    /// The big lanes.
    pub lanes: LaneSet,
    /// A job is big if its demand does not fit within this. Big jobs try lanes first.
    pub big_threshold: Resources,
    /// A small job may join a busy lane only if the lane keeps at least this much memory headroom
    /// after the placement. (A job alone on a worker always goes: the escape hatch.)
    pub lane_reserve: Resources,
}

impl Default for LanesConfig {
    /// The defaults documented on each field; no lanes until configured.
    fn default() -> Self {
        Self {
            backfill: BackfillConfig::default(),
            lanes: LaneSet::Classes(Vec::new()),
            big_threshold: Resources::mem_gb(7.5),
            lane_reserve: Resources::mem_gb(12.0),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    LeastLoaded,
    Tightest { prefer_penalty: u64 },
}

#[derive(Clone, Debug)]
struct Mode {
    priority_order: bool,
    reservations: Option<BackfillConfig>,
    default_priority: i64,
    age_limit: Option<f64>,
    choice: Choice,
    lanes: Option<LanesConfig>,
}

/// Urgency: smaller is more urgent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    priority: i64,
    group: u64,
    seq: u64,
}

#[derive(Clone, Debug)]
struct Worker {
    state: WorkerState,
    running: usize,
    placed: Resources,
    jobs: BTreeSet<JobId>,
    reserved_for: Option<JobId>,
    lane: bool,
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
    reserved: Option<WorkerId>,
}

#[derive(Clone, Debug)]
struct Running {
    worker: WorkerId,
    demand: Resources,
}

/// Whether `job`'s hard constraints (class, avoid list) allow `w` at all.
fn eligible(job: &JobSpec, w: &Worker) -> bool {
    job.class.as_ref().is_none_or(|c| *c == w.state.class) && !job.avoid.contains(&w.state.id)
}

/// Position of `dispatch`'s scan: first the aged jobs by age, then the rest by urgency.
#[derive(Clone, Copy, Debug)]
enum Cursor {
    Aged(Option<u64>),
    Queue(Option<Key>),
}

/// Why a worker does not take a job, for `explain`.
enum Refusal {
    Ineligible,
    ReservedFor(JobId),
    Lane,
    Admission,
}

/// The placement engine shared by every policy.
///
/// One engine implements all policies; they differ in three switches:
///
/// - **order**: FIFO (greedy) or priority (`priority`, then group first arrival, then FIFO);
/// - **reservations**: none (greedy) or one worker drained for the most urgent starving job;
/// - **choice**: among admitting workers, the least loaded (greedy, backfill) or the tightest fit
///   (best fit), optionally with big-lane routing (lanes).
///
/// `dispatch` scans waiting jobs in order and gives each one a worker if any admits it. Because the
/// scan is in urgency order and admission is monotone in load, a job is placed on a worker only if
/// every more urgent waiting job was refused there -- the priority invariant -- without any
/// explicit check. The one event that can make an already-refused worker admissible mid-scan is
/// the release of a reservation; the scan restarts from the top when that happens.
#[derive(Clone, Debug)]
struct Engine<A> {
    mode: Mode,
    admission: A,
    workers: BTreeMap<WorkerId, Worker>,
    queue: BTreeMap<Key, JobId>,
    by_age: BTreeMap<u64, JobId>,
    waiting: HashMap<JobId, Waiting>,
    running: HashMap<JobId, Running>,
    /// Group -> sequence number of its first arrival.
    groups: HashMap<u64, u64>,
    /// Current reservations, in creation order: (holder, worker, since).
    reservations: Vec<(JobId, WorkerId, Instant)>,
    next_seq: u64,
    now: Instant,
    placements_total: u64,
    reservations_total: u64,
    last_dispatch_holders: Vec<JobId>,
}

impl<A: Admission> Engine<A> {
    /// An engine with no workers and no jobs.
    fn new(mode: Mode, admission: A) -> Self {
        Self {
            mode,
            admission,
            workers: BTreeMap::new(),
            queue: BTreeMap::new(),
            by_age: BTreeMap::new(),
            waiting: HashMap::new(),
            running: HashMap::new(),
            groups: HashMap::new(),
            reservations: Vec::new(),
            next_seq: 0,
            now: 0.0,
            placements_total: 0,
            reservations_total: 0,
            last_dispatch_holders: Vec::new(),
        }
    }

    /// Whether a worker is one of the configured big lanes.
    fn is_lane(&self, state: &WorkerState) -> bool {
        match self.mode.lanes.as_ref().map(|l| &l.lanes) {
            Some(LaneSet::Classes(c)) => c.contains(&state.class),
            Some(LaneSet::Workers(ids)) => ids.contains(&state.id),
            None => false,
        }
    }

    /// Whether a demand counts as big for lane routing (never, without lanes).
    fn is_big(&self, demand: &Resources) -> bool {
        self.mode
            .lanes
            .as_ref()
            .is_some_and(|l| !demand.fits_within(&l.big_threshold))
    }

    /// Queue a job under its urgency key, recording its group's first arrival.
    fn submit(&mut self, spec: JobSpec, now: Instant) {
        self.now = now;
        if self.waiting.contains_key(&spec.id) || self.running.contains_key(&spec.id) {
            return;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        let group_seq = *self.groups.entry(spec.group).or_insert(seq);
        let key = if self.mode.priority_order {
            Key {
                priority: spec.priority.unwrap_or(self.mode.default_priority),
                group: group_seq,
                seq,
            }
        } else {
            Key {
                priority: 0,
                group: 0,
                seq,
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
                reserved: None,
            },
        );
    }

    /// Drop the reservation `job` holds, if any, on both sides.
    fn release_reservation_of(&mut self, job: JobId) {
        if let Some(i) = self.reservations.iter().position(|r| r.0 == job) {
            let (_, w, _) = self.reservations.remove(i);
            if let Some(w) = self.workers.get_mut(&w) {
                w.reserved_for = None;
            }
            if let Some(j) = self.waiting.get_mut(&job) {
                j.reserved = None;
            }
        }
    }

    /// Take a job out of the waiting indexes, releasing its reservation.
    fn remove_waiting(&mut self, job: JobId) -> Option<Waiting> {
        self.release_reservation_of(job);
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
            w.running -= 1;
            w.placed -= r.demand;
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
        if !self.finish_running(job) {
            self.remove_waiting(job);
        }
    }

    /// Add a worker or replace its reported state, keeping its placements.
    fn worker_update(&mut self, state: WorkerState, now: Instant) {
        self.now = now;
        let lane = self.is_lane(&state);
        match self.workers.get_mut(&state.id) {
            Some(w) => {
                // A reservation counted against the old class (per-class limits) must not move to
                // the new one; its holder reserves again at the next dispatch.
                let moved = (w.state.class != state.class)
                    .then_some(w.reserved_for)
                    .flatten();
                w.state = state;
                w.lane = lane;
                if let Some(holder) = moved {
                    self.release_reservation_of(holder);
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
                        lane,
                    },
                );
            }
        }
    }

    /// Forget a worker, its running jobs and any reservation on it.
    fn worker_gone(&mut self, id: WorkerId, now: Instant) {
        self.now = now;
        let Some(w) = self.workers.remove(&id) else {
            return;
        };
        for job in &w.jobs {
            self.running.remove(job);
        }
        if let Some(holder) = w.reserved_for {
            self.release_reservation_of(holder);
        }
    }

    /// Whether `w` takes the job, or why not. Reservations and lanes are checked here; everything
    /// else is the admission rule.
    fn refusal(&self, job: &Waiting, w: &Worker) -> Option<Refusal> {
        if !eligible(&job.spec, w) {
            return Some(Refusal::Ineligible);
        }
        if let Some(holder) = w.reserved_for
            && holder != job.spec.id
        {
            return Some(Refusal::ReservedFor(holder));
        }
        let view = w.view();
        if !self.admission.admits(&job.spec.demand, &view) {
            return Some(Refusal::Admission);
        }
        if let Some(l) = &self.mode.lanes
            && w.lane
            && w.running > 0
            && !self.is_big(&job.spec.demand)
            && view.headroom() - (job.spec.demand.mem as i64) < l.lane_reserve.mem as i64
        {
            return Some(Refusal::Lane);
        }
        None
    }

    /// Whether `job` has waited past the age limit.
    fn aged(&self, job: &Waiting) -> bool {
        self.mode
            .age_limit
            .is_some_and(|a| self.now - job.since >= a)
    }

    /// The next job in scan order: aged jobs oldest first, then everything else by urgency.
    fn next_job(&self, cursor: &mut Cursor) -> Option<JobId> {
        loop {
            match *cursor {
                Cursor::Aged(after) => {
                    if self.mode.priority_order && self.mode.age_limit.is_some() {
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

    /// The best worker that takes `job`, if any.
    fn choose(&self, job: &Waiting) -> Option<WorkerId> {
        let big = self.is_big(&job.spec.demand);
        let demand = job.spec.demand.mem as i128;
        // Smallest tuple wins; the worker id makes the order total (determinism).
        let mut best: Option<((u8, i128, bool, usize), WorkerId)> = None;
        for (&id, w) in &self.workers {
            if self.refusal(job, w).is_some() {
                continue;
            }
            let preferred = job.spec.prefer.contains(&id);
            let lane_rank = u8::from(big && !w.lane);
            let fit = match self.mode.choice {
                Choice::LeastLoaded => 0,
                Choice::Tightest { prefer_penalty } => {
                    let after = w.view().headroom() as i128 - demand;
                    after - if preferred { prefer_penalty as i128 } else { 0 }
                }
            };
            let score = (lane_rank, fit, !preferred, w.running);
            if best.as_ref().is_none_or(|(b, _)| score < *b) {
                best = Some((score, id));
            }
        }
        best.map(|(_, id)| id)
    }

    /// Move a waiting job onto a worker and record the placement.
    fn place(&mut self, job: JobId, worker: WorkerId, out: &mut Vec<(JobId, WorkerId)>) {
        if self.waiting.get(&job).and_then(|j| j.reserved) == Some(worker) {
            self.last_dispatch_holders.push(job);
        }
        let j = self
            .remove_waiting(job)
            .expect("placing a job that is not waiting");
        let w = self
            .workers
            .get_mut(&worker)
            .expect("placing on an unknown worker");
        w.running += 1;
        w.placed += j.spec.demand;
        w.jobs.insert(job);
        self.running.insert(
            job,
            Running {
                worker,
                demand: j.spec.demand,
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
        let Some(cfg) = &self.mode.reservations else {
            return false;
        };
        let j = &self.waiting[&job];
        if j.reserved.is_some() || self.now - j.since < cfg.reserve_after {
            return false;
        }
        let class_full = |class: &str| {
            let n = self
                .reservations
                .iter()
                .filter(|r| !cfg.per_class_reservations || self.workers[&r.1].state.class == class)
                .count();
            n >= cfg.max_reservations
        };
        // Most headroom; then preferred; then fewest running; then smallest id.
        let mut best: Option<((i64, bool, usize), WorkerId)> = None;
        for (&id, w) in &self.workers {
            if w.reserved_for.is_some()
                || w.state.slots == 0
                || class_full(&w.state.class)
                || !eligible(&j.spec, w)
            {
                continue;
            }
            let score = (
                -w.view().headroom(),
                !j.spec.prefer.contains(&id),
                w.running,
            );
            if best.as_ref().is_none_or(|(b, _)| score < *b) {
                best = Some((score, id));
            }
        }
        if let Some((_, w)) = best {
            self.workers.get_mut(&w).unwrap().reserved_for = Some(job);
            self.waiting.get_mut(&job).unwrap().reserved = Some(w);
            self.reservations.push((job, w, self.now));
            self.reservations_total += 1;
            return true;
        }
        let mine = self.urgency(j);
        let victim = self
            .reservations
            .iter()
            .filter(|r| eligible(&j.spec, &self.workers[&r.1]))
            .map(|r| (self.urgency(&self.waiting[&r.0]), r.0))
            .filter(|(u, _)| *u > mine)
            .max();
        let Some((_, victim)) = victim else {
            return false;
        };
        let i = self
            .reservations
            .iter()
            .position(|r| r.0 == victim)
            .unwrap();
        let w = self.reservations[i].1;
        self.reservations[i] = (job, w, self.now);
        self.waiting.get_mut(&victim).unwrap().reserved = None;
        self.waiting.get_mut(&job).unwrap().reserved = Some(w);
        self.workers.get_mut(&w).unwrap().reserved_for = Some(job);
        self.reservations_total += 1;
        true
    }

    /// Component-wise maximum admission bound over unreserved workers with a free slot, or `None`
    /// if no unreserved worker can take anything.
    fn open_bound(&self) -> Option<Resources> {
        let mut acc: Option<Resources> = None;
        for w in self.workers.values() {
            if w.reserved_for.is_some() {
                continue;
            }
            if let Some(b) = self.admission.bound(&w.view()) {
                acc = Some(acc.map_or(b, |a| a.max(b)));
            }
        }
        acc
    }

    /// Scan waiting jobs in order, placing each where it is admitted, reserving for the starving.
    fn dispatch(&mut self, now: Instant) -> Vec<(JobId, WorkerId)> {
        self.now = now;
        self.last_dispatch_holders.clear();
        let mut out = Vec::new();
        // A reservation on a worker that can never run anything again is dead weight.
        let dead: Vec<JobId> = self
            .reservations
            .iter()
            .filter(|r| self.workers[&r.1].state.slots == 0)
            .map(|r| r.0)
            .collect();
        for job in dead {
            self.release_reservation_of(job);
        }
        'scan: loop {
            if !self.workers.values().any(|w| w.running < w.state.slots) {
                break;
            }
            let mut bound = self.open_bound();
            let mut cursor = Cursor::Aged(None);
            loop {
                let Some(job) = self.next_job(&mut cursor) else {
                    break 'scan;
                };
                let j = &self.waiting[&job];
                let holder = j.reserved.is_some();
                let hopeful = holder || bound.is_some_and(|b| j.spec.demand.fits_within(&b));
                match hopeful.then(|| self.choose(j)).flatten() {
                    Some(w) => {
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
                    }
                    None => {
                        if self.try_reserve(job) {
                            // A taken-over worker may admit the job now that it is the holder.
                            if let Some(w) = self.choose(&self.waiting[&job]) {
                                self.place(job, w, &mut out);
                                continue 'scan;
                            }
                            bound = self.open_bound();
                        }
                    }
                }
            }
        }
        out
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
        }
    }

    /// Counters and per-worker load.
    fn stats(&self) -> PolicyStats {
        PolicyStats {
            now: self.now,
            waiting: self.waiting.len(),
            running: self.running.len(),
            longest_wait: self
                .by_age
                .values()
                .next()
                .map(|j| (*j, self.now - self.waiting[j].since)),
            reservations: self
                .reservations
                .iter()
                .map(|&(job, worker, since)| ReservationInfo { job, worker, since })
                .collect(),
            placements_total: self.placements_total,
            reservations_total: self.reservations_total,
            workers: self.workers.values().map(|w| self.load(w)).collect(),
            last_dispatch_holders: self.last_dispatch_holders.clone(),
        }
    }

    /// Classify every worker's reason to refuse the job, and summarise.
    fn explain(&self, job: JobId) -> Option<String> {
        const GB: f64 = 1e9;
        if let Some(r) = self.running.get(&job) {
            return Some(format!("job {job} is running on worker {}", r.worker));
        }
        let j = self.waiting.get(&job)?;
        let ahead = self.queue.range(..j.key).count();
        let mut msg = format!(
            "job {job} (demand {:.2} GB, group {}) waiting {:.0}s, {ahead} more urgent job(s) \
             waiting",
            j.spec.demand.mem as f64 / GB,
            j.spec.group,
            self.now - j.since
        );
        if let Some(w) = j.reserved {
            let w = &self.workers[&w];
            msg += &format!(
                "; holds the reservation on worker {} (draining: {}/{} running, headroom {:.2} GB)",
                w.state.id,
                w.running,
                w.state.slots,
                w.view().headroom() as f64 / GB
            );
        }
        let (mut full, mut short, mut lane, mut excluded) = (0, 0, 0, 0);
        let mut best_short: Option<(i64, WorkerId)> = None;
        let mut reserved = Vec::new();
        let mut takers = Vec::new();
        for (&id, w) in &self.workers {
            match self.refusal(j, w) {
                None => takers.push(id),
                Some(Refusal::Ineligible) => excluded += 1,
                Some(Refusal::ReservedFor(h)) => reserved.push(format!("worker {id} for job {h}")),
                Some(Refusal::Lane) => lane += 1,
                Some(Refusal::Admission) if w.running >= w.state.slots => full += 1,
                Some(Refusal::Admission) => {
                    short += 1;
                    let h = w.view().headroom();
                    if best_short.is_none_or(|(b, _)| h > b) {
                        best_short = Some((h, id));
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
        if let Some((h, w)) = best_short {
            msg += &format!(
                "; memory short on {short} worker(s) (best headroom {:.2} GB on worker {w})",
                h as f64 / GB
            );
        }
        if lane > 0 {
            msg += &format!("; {lane} big lane(s) keep their reserve headroom");
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

macro_rules! policy {
    ($(#[$doc:meta])* $name:ident, $cfg:ty, |$c:ident| $mode:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        pub struct $name<A = ProductionAdmission>(Engine<A>);

        impl $name<ProductionAdmission> {
            /// The policy with the production admission rule.
            pub fn new(config: $cfg) -> Self {
                Self::with_admission(config, ProductionAdmission)
            }
        }

        impl<A: Admission> $name<A> {
            /// The policy with a custom admission rule.
            pub fn with_admission($c: $cfg, admission: A) -> Self {
                Self(Engine::new($mode, admission))
            }

            /// Forget a group's first-arrival time. A later job of that group then counts as a
            /// new group. Use it when a group is known to be finished, to bound memory.
            pub fn forget_group(&mut self, group: u64) {
                self.0.groups.remove(&group);
            }
        }

        impl<A: Admission> Policy for $name<A> {
            /// Forwarded to the engine.
            fn submit(&mut self, job: JobSpec, now: Instant) {
                self.0.submit(job, now)
            }

            /// Forwarded to the engine.
            fn cancel(&mut self, job: JobId) {
                self.0.cancel(job)
            }

            /// Forwarded to the engine.
            fn worker_update(&mut self, w: WorkerState, now: Instant) {
                self.0.worker_update(w, now)
            }

            /// Forwarded to the engine.
            fn worker_gone(&mut self, w: WorkerId, now: Instant) {
                self.0.worker_gone(w, now)
            }

            /// Forwarded to the engine.
            fn completed(&mut self, job: JobId, now: Instant) {
                self.0.completed(job, now)
            }

            /// Forwarded to the engine.
            fn dispatch(&mut self, now: Instant) -> Vec<(JobId, WorkerId)> {
                self.0.dispatch(now)
            }

            /// Forwarded to the engine.
            fn explain(&self, job: JobId) -> Option<String> {
                self.0.explain(job)
            }

            /// Forwarded to the engine.
            fn stats(&self) -> PolicyStats {
                self.0.stats()
            }
        }
    };
}

policy!(
    /// The historical behaviour, kept as a baseline: jobs are considered in arrival order and each
    /// takes any worker that admits it -- preferred workers first, then the least loaded. No
    /// priority, no reservations, so jobs larger than the typical headroom starve.
    Greedy,
    GreedyConfig,
    |_c| Mode {
        priority_order: false,
        reservations: None,
        default_priority: 0,
        age_limit: None,
        choice: Choice::LeastLoaded,
        lanes: None,
    }
);

policy!(
    /// Strict priority with one reservation and backfill.
    ///
    /// - Jobs are ordered by `priority` (default [`BackfillConfig::default_priority`]), then by
    ///   their group's first arrival, then FIFO.
    /// - A job takes a worker only if no more urgent waiting job is admitted there; among the
    ///   workers that admit it, preferred workers first, then the least loaded.
    /// - **Reservation**: the most urgent job that has waited at least `reserve_after` and is
    ///   admitted nowhere reserves the worker with the most headroom (ties: preferred, then fewest
    ///   running). No other job is admitted there until the holder is placed -- at the latest when
    ///   the worker empties, by the escape hatch. A reservation is released when its holder is
    ///   placed (anywhere) or cancelled, or its worker leaves or changes class (the holder may then
    ///   reserve again). When all reservations are taken, a more urgent qualifying job takes over
    ///   the least urgent holder's reservation, worker included (it has been draining already).
    /// - **Backfill**: every other worker keeps admitting less urgent jobs.
    ///
    /// No starvation: the most urgent waiting job is placed within `reserve_after` plus the
    /// longest running time of the jobs on the worker it reserves (nothing new is admitted there
    /// once it holds the reservation, and nobody more urgent can take the reservation over).
    PriorityBackfill,
    BackfillConfig,
    |c| Mode {
        priority_order: true,
        default_priority: c.default_priority,
        age_limit: c.age_limit,
        reservations: Some(c),
        choice: Choice::LeastLoaded,
        lanes: None,
    }
);

policy!(
    /// [`PriorityBackfill`] that places each job on the admitting worker whose headroom after
    /// placement is smallest, packing small jobs tightly and keeping big holes open.
    /// [`BestFitConfig::prefer_penalty`] decides how much cache affinity counts.
    BestFit,
    BestFitConfig,
    |c| Mode {
        priority_order: true,
        default_priority: c.backfill.default_priority,
        age_limit: c.backfill.age_limit,
        reservations: Some(c.backfill),
        choice: Choice::Tightest { prefer_penalty: c.prefer_penalty },
        lanes: None,
    }
);

policy!(
    /// [`BestFit`] with big lanes: designated workers admit small jobs only while keeping
    /// [`LanesConfig::lane_reserve`] headroom free, and big jobs try the lanes first.
    Lanes,
    LanesConfig,
    |c| Mode {
        priority_order: true,
        default_priority: c.backfill.default_priority,
        age_limit: c.backfill.age_limit,
        reservations: Some(c.backfill.clone()),
        choice: Choice::Tightest { prefer_penalty: 0 },
        lanes: Some(c),
    }
);

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

    /// Greedy spreads jobs over the least loaded workers in arrival order.
    #[test]
    fn greedy_fills_least_loaded_first() {
        let mut p = Greedy::new(GreedyConfig {});
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
        let mut p = PriorityBackfill::new(BackfillConfig::default());
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
        let mut p = PriorityBackfill::new(BackfillConfig::default());
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

    /// Best fit picks the worker left with the least headroom.
    #[test]
    fn best_fit_packs_tightly() {
        let mut p = BestFit::new(BestFitConfig::default());
        p.worker_update(worker(1, 4, 100), 0.0);
        p.worker_update(worker(2, 4, 50), 0.0);
        p.submit(job(0, 1, 0), 0.0);
        p.submit(job(1, 1, 0), 0.0);
        // Both empty workers admit; the smaller one is the tighter fit.
        assert_eq!(p.dispatch(0.0), vec![(0, 2), (1, 2)]);
    }

    /// Small jobs stop at the lane reserve; big jobs go to the lane.
    #[test]
    fn lanes_keep_headroom_for_big_jobs() {
        let mut p = Lanes::new(LanesConfig {
            lanes: LaneSet::Workers(vec![1]),
            big_threshold: Resources::mem(8 * GB),
            lane_reserve: Resources::mem(30 * GB),
            ..LanesConfig::default()
        });
        p.worker_update(worker(1, 16, 100), 0.0);
        p.worker_update(worker(2, 0, 100), 0.0);
        // Seed the lane so it is busy (the escape hatch would admit anything on an empty lane).
        p.submit(job(0, 5, 0), 0.0);
        assert_eq!(p.dispatch(0.0), vec![(0, 1)]);
        p.worker_update(worker(2, 16, 100), 0.0);
        // The lane is the tighter fit (95 GB vs 100 GB free), so small jobs go there first -- until
        // it would drop below 30 GB of headroom: 95 - 9 * 7 = 32, 95 - 10 * 7 = 25.
        for i in 1..=20 {
            p.submit(job(i, 7, 0), 0.0);
        }
        let out = p.dispatch(0.0);
        let on_lane = out.iter().filter(|(_, w)| *w == 1).count();
        let lane = p.stats().workers.into_iter().find(|w| w.id == 1).unwrap();
        assert_eq!(on_lane, 9);
        assert_eq!(lane.headroom, 32 * GB as i64);
        // A big job goes to the lane.
        p.submit(job(100, 20, 0), 0.0);
        assert_eq!(p.dispatch(0.0), vec![(100, 1)]);
    }
}
