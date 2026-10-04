//! The one placement policy, [`Scheduler`].
//!
//! A [`Scheduler`] is a [`Policy`]: [`handle`](Policy::handle) applies an event to its state at
//! once, and [`poll`](Policy::poll) decides placements and returns every [`Output`] since the last
//! poll. A poll scans the waiting jobs in urgency order ([`Config::order`], with aged jobs first)
//! and for each job looks at every worker in turn:
//!
//! 1. the job's constraints must allow the worker ([`Strength`]);
//! 2. no hold may keep the worker from the job: a [reservation](crate::Reservations) by another
//!    job, or the job's own [deferral](crate::Defer) to a faster worker;
//! 3. the [`Admission`] rule must admit the job's demand there;
//!
//! and among the workers left, [`Config::score`] picks one. A job no worker takes may reserve one
//! instead. After the scan, idle fast workers may start [speculative](crate::Speculate) second
//! attempts of jobs running on slow ones. Failed attempts come back to the queue
//! ([`Config::retry`]).
//!
//! Two one-slot workers and three jobs: two start at once, the third when a slot frees.
//!
//! ```
//! use std::time::Duration;
//!
//! use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, Time, WorkerState};
//!
//! let mut s = Scheduler::new(Config::default());
//! for w in [1, 2] {
//!     let budget = Resources::mem_gb(8.0);
//!     s.handle(
//!         Input::Worker(WorkerState {
//!             id: w,
//!             class: "cpu".into(),
//!             budget,
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! for id in 0..3 {
//!     s.handle(
//!         Input::Submit(JobSpec {
//!             id,
//!             demand: Resources::mem_gb(2.0),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! let start = |job, worker| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker,
//! };
//! assert_eq!(s.poll(Time::ORIGIN), [start(0, 1), start(1, 2)]);
//! let why = s.explain(2).unwrap();
//! assert!(
//!     why.to_string().contains("slots full on 2 worker(s)"),
//!     "{why}"
//! );
//! s.handle(
//!     Input::Done { job: 0, attempt: 1 },
//!     Time(Duration::from_secs(10)),
//! );
//! assert_eq!(s.poll(Time(Duration::from_secs(10))), [start(2, 1)]);
//! ```

mod attempts;
mod holds;
mod order;
mod placement;
mod report;
#[cfg(test)]
mod tests;
mod timing;

use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    time::Duration,
};

use holds::Hold;
use order::{Key, dedup};

use crate::{
    Admission, Attempt, Config, Explanation, Input, JobId, JobSpec, Output, Policy, PolicyStats,
    ProductionAdmission, Resources, SLOTS, Time, Tried, WorkerId, WorkerState, WorkerView,
    speed::{ClassId, KindId, Speeds},
};
#[cfg(doc)]
use crate::{GroupOrder, Strength};

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
    /// `∫ running dt`, in seconds, up to `occ_at` (mean concurrency over a job's run, for
    /// learning).
    occ: f64,
    occ_at: Time,
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
    since: Time,
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
    started: Time,
    /// The worker's `occ` when it started.
    occ0: f64,
}

/// A job with at least one live attempt.
#[derive(Clone, Debug)]
struct Running {
    job: Job,
    live: Vec<Run>,
}

/// Bring a worker's concurrency integral up to `now`.
fn tick_occ(w: &mut Worker, now: Time) {
    if now > w.occ_at {
        w.occ += w.running() as f64 * (now - w.occ_at).as_secs_f64();
        w.occ_at = now;
    }
}

/// The placement policy: list scheduling with admission, reservations and backfill, configured
/// by a [`Config`].
///
/// - Jobs are considered in [`Config::order`], aged jobs first ([`Config::age_limit`]).
/// - A job takes a worker only if no more urgent waiting job is admitted there; among the
///   workers that admit it, the one [`Config::score`] ranks first.
/// - [`Config::reservations`] drain a worker for a starving job; every other worker keeps
///   admitting less urgent jobs.
/// - Failed attempts are retried ([`Config::retry`]) and, with
///   [`SpeedConfig::speculate`](crate::SpeedConfig::speculate), idle fast workers run second
///   attempts of jobs on slow ones.
///
/// The examples below share the hidden helpers `worker(id, slots, bytes)`, `job(id, bytes)` and
/// `start(job, attempt, worker)`, which build a [`WorkerState`], a [`JobSpec`] and an
/// [`Output::Start`].
///
/// # The scan
///
/// Each [`poll`](Policy::poll) scans waiting jobs in order and gives each one a worker if any
/// admits it. Because the scan is in urgency order and admission is monotone in load, a job is
/// placed on a worker only if every more urgent waiting job was refused there -- the priority
/// invariant -- without any explicit check. The one event that can make an already-refused worker
/// admissible mid-scan is the release of a hold (a reservation whose holder is placed); the scan
/// restarts from the top when that happens. Speculative attempts come after the scan, on workers
/// every waiting job was refused on.
///
/// A less urgent job that fits backfills the room a more urgent one cannot use:
///
/// ```
/// # use std::time::Duration;
/// # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, Time, WorkerState};
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     slots,
/// #     budget: Resources::mem(bytes),
/// #     ..Default::default()
/// # };
/// # let job = |id, bytes| JobSpec { id, demand: Resources::mem(bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// let mut s = Scheduler::new(Config::default());
/// s.handle(Input::Worker(worker(1, 4, 100)), Time::ORIGIN);
/// s.handle(Input::Submit(job(0, 70)), Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(0, 1, 1)]);
/// let urgent = JobSpec {
///     priority: Some(-1),
///     ..job(1, 50)
/// };
/// s.handle(Input::Submit(urgent), Time(Duration::from_secs(1)));
/// s.handle(Input::Submit(job(2, 20)), Time(Duration::from_secs(1)));
/// assert_eq!(s.poll(Time(Duration::from_secs(1))), [start(2, 1, 1)]);
/// let why = s.explain(1).unwrap();
/// assert!(why.to_string().contains("memory short on 1 worker(s)"), "{why}");
/// ```
///
/// # Retries
///
/// A failed attempt sends its job back to the queue with its place and age; the retry avoids,
/// softly, the workers it failed on. Reports about attempts that are no longer live are ignored,
/// and cancelling a running job stops it.
///
/// ```
/// # use std::time::Duration;
/// # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, Time, WorkerState};
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     slots,
/// #     budget: Resources::mem(bytes),
/// #     ..Default::default()
/// # };
/// # let job = |id, bytes| JobSpec { id, demand: Resources::mem(bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// use whelm::FailKind;
///
/// let mut s = Scheduler::new(Config::default());
/// s.handle(Input::Worker(worker(1, 1, 100)), Time::ORIGIN);
/// s.handle(Input::Worker(worker(2, 1, 100)), Time::ORIGIN);
/// s.handle(Input::Submit(job(0, 10)), Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(0, 1, 1)]);
/// let failed = Input::Failed {
///     job: 0,
///     attempt: 1,
///     kind: FailKind::Other,
///     why: "segfault".into(),
/// };
/// s.handle(failed, Time(Duration::from_secs(1)));
/// assert_eq!(s.poll(Time(Duration::from_secs(1))), [start(0, 2, 2)]);
/// s.handle(Input::Done { job: 0, attempt: 1 }, Time(Duration::from_secs(2))); // stale
/// assert_eq!((s.poll(Time(Duration::from_secs(2))), s.stats().running), (vec![], 1));
/// s.handle(Input::Cancel(0), Time(Duration::from_secs(3)));
/// let stop = Output::Stop {
///     job: 0,
///     attempt: 2,
///     worker: 2,
/// };
/// assert_eq!(s.poll(Time(Duration::from_secs(3))), [stop]);
/// ```
///
/// [`RetryConfig`](crate::RetryConfig) shows a job given up.
///
/// # Holds: reservations and deferral
///
/// A hold keeps a worker from a job on purpose although it might admit it. A
/// [reservation](crate::Reservations) keeps a worker from every job but its holder, so that it
/// drains for a job that fits nowhere. Here two workers each run a 60-byte job, a 50-byte job
/// reserves worker 1 once it has waited `reserve_after`, a small job keeps to worker 2, and the
/// holder starts when worker 1 drains:
///
/// ```
/// # use std::time::Duration;
/// # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, Time, WorkerState};
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     slots,
/// #     budget: Resources::mem(bytes),
/// #     ..Default::default()
/// # };
/// # let job = |id, bytes| JobSpec { id, demand: Resources::mem(bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// let mut s = Scheduler::new(Config::default());
/// s.handle(Input::Worker(worker(1, 4, 100)), Time::ORIGIN);
/// s.handle(Input::Worker(worker(2, 4, 100)), Time::ORIGIN);
/// s.handle(Input::Submit(job(10, 60)), Time::ORIGIN);
/// s.handle(Input::Submit(job(11, 60)), Time::ORIGIN);
/// s.handle(Input::Submit(job(1, 50)), Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(10, 1, 1), start(11, 1, 2)]);
/// let reserve_after = Config::default().reservations.unwrap().reserve_after;
/// let reserve_at = Time::ORIGIN + reserve_after;
/// assert_eq!(s.next_wakeup(), Some(reserve_at));
/// assert_eq!(s.poll(reserve_at), []);
/// let reservation = &s.stats().reservations[0];
/// assert_eq!((reservation.job, reservation.worker), (1, 1));
/// s.handle(Input::Submit(job(20, 5)), Time(Duration::from_secs(70)));
/// assert_eq!(s.poll(Time(Duration::from_secs(70))), [start(20, 1, 2)]);
/// assert!(matches!(
///     s.explain(1).unwrap().waiting().unwrap().hold,
///     Some(whelm::Holding::Reservation { worker: 1, .. })
/// ));
/// s.handle(Input::Done { job: 10, attempt: 1 }, Time(Duration::from_secs(100)));
/// assert_eq!(s.poll(Time(Duration::from_secs(100))), [start(1, 1, 1)]);
/// assert_eq!(s.stats().last_dispatch_holders, [1]);
/// ```
///
/// A [deferral](crate::Defer) is a job's own hold: it declines a slow free worker to wait for a
/// fast busy one, when it expects to finish sooner that way. Here job 1, of 40 s of work, waits
/// 2.5 s for the worker four times as fast instead of starting on the slow one:
///
/// ```
/// # use std::time::Duration;
/// #
/// # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, Time, WorkerState};
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     slots,
/// #     budget: Resources::mem(bytes),
/// #     ..Default::default()
/// # };
/// # let job = |id, bytes| JobSpec { id, demand: Resources::mem(bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// use whelm::{Defer, SpeedConfig};
///
/// let mut s = Scheduler::new(Config {
///     speed: SpeedConfig {
///         defer: Some(Defer::default()),
///         ..SpeedConfig::default()
///     },
///     ..Config::default()
/// });
/// let fast = WorkerState {
///     speed: 4.0,
///     ..worker(2, 1, 100)
/// };
/// s.handle(Input::Worker(worker(1, 1, 100)), Time::ORIGIN);
/// s.handle(Input::Worker(fast), Time::ORIGIN);
/// let work = |id, work| JobSpec {
///     work: Some(Duration::from_secs(work)),
///     ..job(id, 10)
/// };
/// s.handle(Input::Submit(work(0, 10)), Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(0, 1, 2)]);
/// s.handle(Input::Submit(work(1, 40)), Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), []);
/// assert_eq!(s.stats().deferred, [(1, 2, Time(Duration::from_millis(2500)))]);
/// assert!(matches!(
///     s.explain(1).unwrap().waiting().unwrap().hold,
///     Some(whelm::Holding::Deferral { worker: 2, .. })
/// ));
/// s.handle(Input::Done { job: 0, attempt: 1 }, Time(Duration::from_millis(2500)));
/// assert_eq!(s.poll(Time(Duration::from_millis(2500))), [start(1, 1, 2)]);
/// ```
///
/// # Speculation
///
/// With [`Speculate`](crate::Speculate), a fast worker left idle after the scan starts a second
/// attempt of a long job running on a slow worker. Both run; the first to finish wins, and the
/// other is stopped:
///
/// ```
/// # use std::time::Duration;
/// #
/// # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, Time, WorkerState};
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     slots,
/// #     budget: Resources::mem(bytes),
/// #     ..Default::default()
/// # };
/// # let job = |id, bytes| JobSpec { id, demand: Resources::mem(bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// use whelm::{Speculate, SpeedConfig};
///
/// let mut s = Scheduler::new(Config {
///     speed: SpeedConfig {
///         speculate: Some(Speculate::default()),
///         ..SpeedConfig::default()
///     },
///     ..Config::default()
/// });
/// let fast = WorkerState {
///     speed: 4.0,
///     ..worker(2, 1, 100)
/// };
/// s.handle(Input::Worker(worker(1, 1, 100)), Time::ORIGIN);
/// s.handle(Input::Worker(fast), Time::ORIGIN);
/// let work = |id, work| JobSpec {
///     work: Some(Duration::from_secs(work)),
///     ..job(id, 10)
/// };
/// s.handle(Input::Submit(work(0, 4)), Time::ORIGIN);
/// s.handle(Input::Submit(work(1, 100)), Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(0, 1, 2), start(1, 1, 1)]);
/// // The fast worker frees at 1 s; job 1 would end there at 26 s instead of 100 s.
/// s.handle(Input::Done { job: 0, attempt: 1 }, Time(Duration::from_secs(1)));
/// assert_eq!(s.poll(Time(Duration::from_secs(1))), [start(1, 2, 2)]);
/// s.handle(Input::Done { job: 1, attempt: 2 }, Time(Duration::from_secs(26)));
/// let stop = Output::Stop {
///     job: 1,
///     attempt: 1,
///     worker: 1,
/// };
/// assert_eq!(s.poll(Time(Duration::from_secs(26))), [stop]);
/// ```
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
    now: Time,
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
    /// A scheduler with the production admission rule ([`ProductionAdmission`]).
    ///
    /// It starts with no workers and no jobs, so nothing is timed:
    ///
    /// ```
    /// use whelm::{Config, Policy, Scheduler};
    ///
    /// let s = Scheduler::new(Config::default());
    /// assert_eq!((s.stats().waiting, s.next_wakeup()), (0, None));
    /// ```
    pub fn new(config: Config) -> Self {
        Self::with_admission(config, ProductionAdmission)
    }

    /// A scheduler with a custom admission rule.
    ///
    /// The rule must be monotone in load ([`Admission`] has the contract and another example).
    /// Here the production rule is kept, and workers of class "draining" take nothing new:
    ///
    /// ```
    /// use whelm::{
    ///     Admission, Config, Input, JobSpec, Output, Policy, ProductionAdmission, Resources,
    ///     Scheduler, Time, WorkerState, WorkerView,
    /// };
    ///
    /// struct SkipDraining;
    ///
    /// impl Admission for SkipDraining {
    ///     fn admits(&self, demand: &Resources, w: &WorkerView) -> bool {
    ///         w.state.class != "draining" && ProductionAdmission.admits(demand, w)
    ///     }
    /// }
    ///
    /// let mut s = Scheduler::with_admission(Config::default(), SkipDraining);
    /// s.handle(
    ///     Input::Worker(WorkerState {
    ///         id: 1,
    ///         class: "draining".into(),
    ///         slots: 4,
    ///         ..Default::default()
    ///     }),
    ///     Time::ORIGIN,
    /// );
    /// s.handle(
    ///     Input::Worker(WorkerState {
    ///         id: 2,
    ///         class: "cpu".into(),
    ///         slots: 4,
    ///         ..Default::default()
    ///     }),
    ///     Time::ORIGIN,
    /// );
    /// s.handle(
    ///     Input::Submit(JobSpec {
    ///         id: 0,
    ///         ..Default::default()
    ///     }),
    ///     Time::ORIGIN,
    /// );
    /// assert_eq!(
    ///     s.poll(Time::ORIGIN),
    ///     [Output::Start {
    ///         job: 0,
    ///         attempt: 1,
    ///         worker: 2
    ///     }]
    /// );
    /// ```
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
            now: Time::ORIGIN,
            placements_total: 0,
            reservations_total: 0,
            last_dispatch_holders: Vec::new(),
            deferred_any: Vec::new(),
        }
    }

    /// Forget a group's first-arrival time. A later job of that group then counts as a new group.
    /// Use it when a group is known to be finished, to bound memory. Only
    /// [`GroupOrder::Arrival`] records arrivals.
    ///
    /// Group 1 arrived first, so a new job of it would run before one of group 2; once group 1
    /// is forgotten, the new job counts as arriving after group 2:
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::{Config, Input, JobSpec, Output, Policy, Scheduler, Time, WorkerState};
    ///
    /// let next = |forget| {
    ///     let mut s = Scheduler::new(Config::default());
    ///     s.handle(
    ///         Input::Worker(WorkerState {
    ///             id: 1,
    ///             ..Default::default()
    ///         }),
    ///         Time::ORIGIN,
    ///     );
    ///     s.handle(
    ///         Input::Submit(JobSpec {
    ///             id: 0,
    ///             group: 1,
    ///             ..Default::default()
    ///         }),
    ///         Time::ORIGIN,
    ///     );
    ///     s.poll(Time::ORIGIN); // job 0, the last of group 1 for now, takes the slot
    ///     if forget {
    ///         s.forget_group(1);
    ///     }
    ///     s.handle(
    ///         Input::Submit(JobSpec {
    ///             id: 1,
    ///             group: 2,
    ///             ..Default::default()
    ///         }),
    ///         Time(Duration::from_secs(1)),
    ///     );
    ///     s.handle(
    ///         Input::Submit(JobSpec {
    ///             id: 2,
    ///             group: 1,
    ///             ..Default::default()
    ///         }),
    ///         Time(Duration::from_secs(1)),
    ///     );
    ///     s.handle(
    ///         Input::Done { job: 0, attempt: 1 },
    ///         Time(Duration::from_secs(2)),
    ///     );
    ///     match s.poll(Time(Duration::from_secs(2)))[..] {
    ///         [Output::Start { job, .. }] => job,
    ///         ref out => panic!("{out:?}"),
    ///     }
    /// };
    /// assert_eq!(next(false), 2);
    /// assert_eq!(next(true), 1);
    /// ```
    pub fn forget_group(&mut self, group: u64) {
        self.groups.remove(&group);
    }

    /// The next time, after now, that the passing of time alone changes what `dispatch` may do:
    /// a hold lapses, a job ages, or a job waits long enough to reserve.
    fn next_wakeup(&self) -> Option<Time> {
        let now = self.now;
        // `by_age` is in submission order, so the first job whose deadline is still ahead has
        // the earliest one.
        let first_after = |wait: Duration| {
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
            .min()
    }
}

impl Policy for Scheduler {
    /// Applied at once; stops and give-ups wait in the outbox for `poll`.
    fn handle(&mut self, input: Input, now: Time) {
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
    fn poll(&mut self, now: Time) -> Vec<Output> {
        self.now = now;
        self.dispatch();
        self.speculate();
        std::mem::take(&mut self.outbox)
    }

    /// When a hold lapses, a job ages or a job may reserve, whichever is first.
    fn next_wakeup(&self) -> Option<Time> {
        Scheduler::next_wakeup(self)
    }

    /// Every worker's reason to refuse it, summarised.
    fn explain(&self, job: JobId) -> Option<Explanation> {
        Scheduler::explain(self, job)
    }

    /// Counters, holds and per-worker load.
    fn stats(&self) -> PolicyStats {
        Scheduler::stats(self)
    }
}
