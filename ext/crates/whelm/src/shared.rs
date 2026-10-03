//! A thread-safe, blocking front end over a [`Policy`], for callers with a thread per task.
//!
//! A [`Policy`] is driven by one event loop. A caller that runs each task on its own thread
//! instead wraps the policy in a [`SharedPolicy`]: the thread calls
//! [`lease`](SharedPolicy::lease), which blocks until the policy starts its job and returns a
//! [`Lease`] naming the worker, then reports the outcome on the lease. Worker heartbeats and
//! departures come from whichever thread sees them.
//!
//! Two threads share a worker with one slot; the second blocks until the first lease ends:
//!
//! ```
//! use std::{sync::Arc, thread};
//!
//! use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
//!
//! let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::fifo()), || 0.0));
//! shared.worker_update(WorkerState::new(1, "x", 1, Resources::mem_gb(8.0)));
//! let job = |id| JobSpec::new(id, Resources::mem_gb(1.0), 0);
//!
//! let first = shared.lease(job(1));
//! let second = thread::spawn({
//!     let shared = shared.clone();
//!     move || {
//!         let lease = shared.lease(job(2));
//!         let worker = lease.worker();
//!         lease.complete();
//!         worker
//!     }
//! });
//! while shared.waiting() == 0 {
//!     thread::yield_now();
//! }
//! assert!(
//!     shared
//!         .explain(2)
//!         .unwrap()
//!         .ends_with("slots full on 1 worker(s)")
//! );
//! first.complete();
//! assert_eq!(second.join().unwrap(), 1);
//! assert_eq!(shared.stats().placements_total, 2);
//! ```

use std::{
    collections::HashMap,
    sync::{Arc, Condvar, Mutex, MutexGuard, Weak},
    thread::JoinHandle,
    time::Duration,
};

use crate::{
    Attempt, FailKind, GaveUp, Input, Instant, JobId, JobSpec, Output, Policy, PolicyStats,
    WorkerId, WorkerState,
};

/// One leased job's mailbox, from its first submission until the lease ends.
struct Slot {
    /// Wakes the job's thread.
    cv: Arc<Condvar>,
    /// The job as submitted, returned if a timed lease expires.
    spec: JobSpec,
    /// When the thread started waiting for the next attempt (lease or failure).
    asked: Instant,
    /// The attempt the thread holds, while it holds one.
    held: Option<(Attempt, WorkerId)>,
    /// The held attempt was failed by its worker leaving: the policy treats it as stale.
    lost: bool,
    /// A stop arrived for the held attempt.
    stopped: bool,
    /// The next attempt, not yet picked up by the thread.
    started: Option<(Attempt, WorkerId)>,
    /// The policy gave the job up, not yet picked up by the thread.
    gave_up: Option<GaveUp>,
}

/// The state behind the lock.
struct State<P> {
    policy: P,
    now: Instant,
    /// Jobs with a lease, held or awaited.
    jobs: HashMap<JobId, Slot>,
    /// Tickers spawned before this generation exit.
    ticker_generation: u64,
}

/// Why waiting for a start ended without one.
enum NoStart {
    GaveUp(GaveUp),
    Timeout(Box<JobSpec>),
}

/// A thread-safe front end over a [`Policy`]: each task thread calls [`lease`](Self::lease),
/// which submits its job and blocks until the policy starts it, then ends the lease with
/// [`Lease::complete`] or [`Lease::fail`].
///
/// `poll` runs after every mutating call and on [`tick`](Self::tick) (aging, reservations and
/// voluntary waits need time to pass without events; [`spawn_ticker`](Self::spawn_ticker) does it
/// on a thread). Each start wakes exactly the thread waiting for that job.
///
/// A thread runs one attempt at a time, so speculation ([`Speculate`](crate::Speculate)) should be
/// off: a start for a job whose lease holds an attempt is answered with [`FailKind::Rejected`],
/// which counts as a failed attempt. Outputs of a [`DagScheduler`](crate::DagScheduler) other
/// than starts, stops and give-ups are not delivered: wrap a flat policy.
///
/// The policy stays deterministic; only the clock and the interleaving of callers are not.
pub struct SharedPolicy<P> {
    state: Mutex<State<P>>,
    clock: Box<dyn Fn() -> Instant + Send + Sync>,
}

impl<P: Policy> SharedPolicy<P> {
    /// A front end over `policy`, reading time from `clock` (seconds; made non-decreasing here).
    ///
    /// A clock the caller sets makes time-dependent behaviour reproducible. Here the second job
    /// waits while the clock moves to 7, and a clock going back to 3 is read as 7:
    ///
    /// ```
    /// use std::{
    ///     sync::{Arc, Mutex},
    ///     thread,
    /// };
    ///
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
    ///
    /// let time = Arc::new(Mutex::new(0.0));
    /// let clock = {
    ///     let time = time.clone();
    ///     move || *time.lock().unwrap()
    /// };
    /// let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::default()), clock));
    /// shared.worker_update(WorkerState::new(1, "x", 1, Resources::mem_gb(8.0)));
    /// let job = |id| JobSpec::new(id, Resources::mem_gb(1.0), 0);
    ///
    /// let first = shared.lease(job(1));
    /// let second = thread::spawn({
    ///     let shared = shared.clone();
    ///     move || {
    ///         let lease = shared.lease(job(2));
    ///         let waited = lease.waited();
    ///         lease.complete();
    ///         waited
    ///     }
    /// });
    /// while shared.waiting() == 0 {
    ///     thread::yield_now();
    /// }
    /// *time.lock().unwrap() = 7.0;
    /// shared.tick();
    /// *time.lock().unwrap() = 3.0;
    /// first.complete();
    /// assert_eq!(second.join().unwrap(), 7.0);
    /// assert_eq!(shared.stats().now, 7.0);
    /// ```
    pub fn new(policy: P, clock: impl Fn() -> Instant + Send + Sync + 'static) -> Self {
        Self {
            state: Mutex::new(State {
                policy,
                now: f64::NEG_INFINITY,
                jobs: HashMap::new(),
                ticker_generation: 0,
            }),
            clock: Box::new(clock),
        }
    }

    /// A front end whose clock is seconds since its creation.
    ///
    /// ```
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
    ///
    /// let shared = SharedPolicy::with_system_clock(Scheduler::new(Config::default()));
    /// shared.worker_update(WorkerState::new(1, "x", 4, Resources::mem_gb(8.0)));
    /// let lease = shared.lease(JobSpec::new(1, Resources::mem_gb(1.0), 0));
    /// assert!(lease.waited() >= 0.0);
    /// lease.complete();
    /// ```
    pub fn with_system_clock(policy: P) -> Self {
        let start = std::time::Instant::now();
        Self::new(policy, move || start.elapsed().as_secs_f64())
    }

    /// Lock the state and advance its clock. A poisoned lock (a caller panicked inside the
    /// policy) is taken over: the policy's own state is updated atomically per call.
    fn lock(&self) -> (MutexGuard<'_, State<P>>, Instant) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let now = (self.clock)().max(s.now);
        s.now = now;
        (s, now)
    }

    /// Poll, and deliver each output to its job's slot. Rejecting a speculative start, or
    /// cancelling a start nobody waits for, frees a slot, so polling repeats until neither
    /// happens.
    fn pump(s: &mut State<P>, now: Instant) {
        loop {
            let mut again = false;
            for o in s.policy.poll(now) {
                match o {
                    Output::Start {
                        job,
                        attempt,
                        worker,
                    } => match s.jobs.get_mut(&job) {
                        Some(slot) if slot.held.is_some() && !slot.lost => {
                            let why = "a thread-per-task caller runs one attempt at a time".into();
                            let input = Input::Failed {
                                job,
                                attempt,
                                kind: FailKind::Rejected,
                                why,
                            };
                            s.policy.handle(input, now);
                            again = true;
                        }
                        Some(slot) => {
                            slot.started = Some((attempt, worker));
                            slot.cv.notify_one();
                        }
                        // Nobody waits (a job submitted through `with`): keep it consistent.
                        None => {
                            s.policy.handle(Input::Cancel(job), now);
                            again = true;
                        }
                    },
                    Output::Stop { job, attempt, .. } => {
                        if let Some(slot) = s.jobs.get_mut(&job)
                            && slot.held.is_some_and(|h| h.0 == attempt)
                        {
                            slot.stopped = true;
                        }
                    }
                    Output::GaveUp(g) => {
                        if let Some(slot) = s.jobs.get_mut(&g.job) {
                            slot.gave_up = Some(g);
                            slot.cv.notify_one();
                        }
                    }
                    Output::RunLocal { .. } | Output::Ready { .. } | Output::Passed { .. } => {}
                }
            }
            if !again {
                return;
            }
        }
    }

    /// Block until job `id` starts again or is given up, at most until `deadline` (of
    /// `std::time`, not of the policy clock); on timeout the job is cancelled.
    fn wait<'a>(
        &'a self,
        mut s: MutexGuard<'a, State<P>>,
        id: JobId,
        deadline: Option<std::time::Instant>,
    ) -> Result<Lease<'a, P>, NoStart> {
        loop {
            let now = s.now;
            let slot = s
                .jobs
                .get_mut(&id)
                .expect("waiting for a job without a slot");
            if let Some((attempt, worker)) = slot.started.take() {
                slot.held = Some((attempt, worker));
                slot.lost = false;
                slot.stopped = false;
                return Ok(Lease {
                    shared: self,
                    job: id,
                    attempt,
                    worker,
                    waited: now - slot.asked,
                    open: true,
                });
            }
            if let Some(g) = slot.gave_up.take() {
                s.jobs.remove(&id);
                return Err(NoStart::GaveUp(g));
            }
            let cv = slot.cv.clone();
            match deadline {
                None => s = cv.wait(s).unwrap_or_else(|e| e.into_inner()),
                Some(d) => {
                    let left = d.saturating_duration_since(std::time::Instant::now());
                    if left.is_zero() {
                        let slot = s.jobs.remove(&id).unwrap();
                        s.policy.handle(Input::Cancel(id), now);
                        Self::pump(&mut s, now);
                        return Err(NoStart::Timeout(Box::new(slot.spec)));
                    }
                    s = cv
                        .wait_timeout(s, left)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
            }
        }
    }

    /// Submit `job` and wait for its start, at most until `deadline`.
    fn lease_until(
        &self,
        job: JobSpec,
        deadline: Option<std::time::Instant>,
    ) -> Result<Lease<'_, P>, Box<JobSpec>> {
        let id = job.id;
        let (mut s, now) = self.lock();
        assert!(!s.jobs.contains_key(&id), "job {id} is already leased");
        s.jobs.insert(
            id,
            Slot {
                cv: Arc::new(Condvar::new()),
                spec: job.clone(),
                asked: now,
                held: None,
                lost: false,
                stopped: false,
                started: None,
                gave_up: None,
            },
        );
        s.policy.handle(Input::Submit(job), now);
        Self::pump(&mut s, now);
        match self.wait(s, id, deadline) {
            Ok(lease) => Ok(lease),
            Err(NoStart::Timeout(job)) => Err(job),
            Err(NoStart::GaveUp(_)) => unreachable!("a job is given up only after failing"),
        }
    }

    /// Submit `job` and block until the policy starts it. Dropping the lease without
    /// [`complete`](Lease::complete) or [`fail`](Lease::fail) (e.g. when the caller unwinds)
    /// cancels the job.
    ///
    /// ```
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::default()), || 0.0);
    /// shared.worker_update(WorkerState::new(1, "x", 1, Resources::mem_gb(8.0)));
    /// let lease = shared.lease(JobSpec::new(1, Resources::mem_gb(1.0), 0));
    /// assert_eq!((lease.worker(), lease.attempt()), (1, 1));
    /// assert_eq!(shared.stats().running, 1);
    /// drop(lease);
    /// assert_eq!(shared.stats().running, 0);
    /// ```
    ///
    /// # Panics
    ///
    /// If a job with this id is already leased here.
    pub fn lease(&self, job: JobSpec) -> Lease<'_, P> {
        self.lease_until(job, None)
            .unwrap_or_else(|_| unreachable!("no deadline"))
    }

    /// [`lease`](Self::lease), giving up after `timeout`: the job is then withdrawn from the
    /// policy and returned. A start made concurrently with the timeout is still returned.
    ///
    /// With no worker, the job never starts:
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::default()), || 0.0);
    /// let job = JobSpec::new(1, Resources::mem_gb(1.0), 0);
    /// let back = shared
    ///     .lease_timeout(job, Duration::from_millis(10))
    ///     .err()
    ///     .unwrap();
    /// assert_eq!(back.id, 1);
    /// assert_eq!(shared.stats().waiting, 0);
    /// ```
    pub fn lease_timeout(
        &self,
        job: JobSpec,
        timeout: Duration,
    ) -> Result<Lease<'_, P>, Box<JobSpec>> {
        self.lease_until(job, Some(std::time::Instant::now() + timeout))
    }

    /// A worker joined or reported a heartbeat.
    ///
    /// A worker joining wakes the threads whose jobs it starts:
    ///
    /// ```
    /// use std::{sync::Arc, thread};
    ///
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
    ///
    /// let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::default()), || 0.0));
    /// let task = thread::spawn({
    ///     let shared = shared.clone();
    ///     move || {
    ///         let lease = shared.lease(JobSpec::new(1, Resources::mem_gb(1.0), 0));
    ///         let worker = lease.worker();
    ///         lease.complete();
    ///         worker
    ///     }
    /// });
    /// while shared.waiting() == 0 {
    ///     thread::yield_now();
    /// }
    /// shared.worker_update(WorkerState::new(5, "x", 1, Resources::mem_gb(8.0)));
    /// assert_eq!(task.join().unwrap(), 5);
    /// ```
    pub fn worker_update(&self, w: WorkerState) {
        let (mut s, now) = self.lock();
        s.policy.handle(Input::Worker(w), now);
        Self::pump(&mut s, now);
    }

    /// A worker left: the attempts there fail with [`FailKind::LinkDied`] and are retried (or
    /// given up) at once. Returns the leased jobs whose attempt ran there, by id: each thread's
    /// later [`Lease::fail`] returns the job's next attempt (or its give-up), and its
    /// [`Lease::complete`] cancels the retry.
    ///
    /// ```
    /// use whelm::{Config, FailKind, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::fifo()), || 0.0);
    /// for w in [1, 2] {
    ///     shared.worker_update(WorkerState::new(w, "x", 1, Resources::mem_gb(8.0)));
    /// }
    /// let lease = shared.lease(JobSpec::new(7, Resources::mem_gb(1.0), 0));
    /// assert_eq!(lease.worker(), 1);
    /// assert_eq!(shared.worker_gone(1), [7]);
    /// // The thread learns of it from its own link, and gets the retry the policy already made.
    /// let retry = lease.fail(FailKind::LinkDied, "connection reset").unwrap();
    /// assert_eq!((retry.worker(), retry.attempt()), (2, 2));
    /// retry.complete();
    /// ```
    pub fn worker_gone(&self, w: WorkerId) -> Vec<JobId> {
        let (mut s, now) = self.lock();
        let mut hit = Vec::new();
        for (&id, slot) in &mut s.jobs {
            if slot.held.is_some_and(|h| h.1 == w) {
                slot.lost = true;
                hit.push(id);
            }
            // A retry started there and not yet picked up fails with the worker; the thread
            // waits for the one after it.
            if slot.started.is_some_and(|h| h.1 == w) {
                slot.started = None;
            }
        }
        hit.sort_unstable();
        s.policy.handle(Input::WorkerGone(w), now);
        Self::pump(&mut s, now);
        hit
    }

    /// Poll now (time has passed).
    ///
    /// Every other call polls too; `tick` is for time passing without events, at
    /// [`next_wakeup`](Self::next_wakeup) or on a [ticker](Self::spawn_ticker).
    pub fn tick(&self) {
        let (mut s, now) = self.lock();
        Self::pump(&mut s, now);
    }

    /// When the policy next needs a poll without events, on the policy clock.
    ///
    /// A job too big for a busy worker's headroom reserves it once it has waited
    /// [`Reservations::reserve_after`](crate::Reservations::reserve_after); ticking then makes
    /// the reservation:
    ///
    /// ```
    /// use std::{
    ///     sync::{Arc, Mutex},
    ///     thread,
    /// };
    ///
    /// use whelm::{Config, JobSpec, Reservations, Resources, Scheduler, SharedPolicy, WorkerState};
    ///
    /// let time = Arc::new(Mutex::new(0.0));
    /// let clock = {
    ///     let time = time.clone();
    ///     move || *time.lock().unwrap()
    /// };
    /// let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::default()), clock));
    /// shared.worker_update(WorkerState::new(1, "x", 4, Resources::mem_gb(8.0)));
    /// let first = shared.lease(JobSpec::new(1, Resources::mem_gb(4.0), 0));
    /// let big = thread::spawn({
    ///     let shared = shared.clone();
    ///     move || {
    ///         shared
    ///             .lease(JobSpec::new(2, Resources::mem_gb(6.0), 0))
    ///             .complete()
    ///     }
    /// });
    /// while shared.waiting() == 0 {
    ///     thread::yield_now();
    /// }
    ///
    /// let wake = shared.next_wakeup().unwrap();
    /// assert_eq!(wake, Reservations::default().reserve_after);
    /// *time.lock().unwrap() = wake;
    /// shared.tick();
    /// assert_eq!(shared.stats().reservations[0].job, 2);
    /// assert!(
    ///     shared
    ///         .explain(2)
    ///         .unwrap()
    ///         .contains("holds the reservation on worker 1")
    /// );
    /// first.complete();
    /// big.join().unwrap();
    /// ```
    pub fn next_wakeup(&self) -> Option<Instant> {
        self.lock().0.policy.next_wakeup()
    }

    /// Why a job is not running ([`Policy::explain`]); see the [module example](self).
    pub fn explain(&self, job: JobId) -> Option<String> {
        self.lock().0.policy.explain(job)
    }

    /// The policy's counters ([`Policy::stats`]).
    pub fn stats(&self) -> PolicyStats {
        self.lock().0.policy.stats()
    }

    /// Run `f` on the policy under the lock (e.g. to forget a group), then poll. Starts of jobs
    /// submitted through `f` are cancelled: nobody waits for them.
    ///
    /// ```
    /// use whelm::{Config, Input, JobSpec, Policy, Resources, Scheduler, SharedPolicy};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::default()), || 0.0);
    /// shared.worker_update(whelm::WorkerState::new(1, "x", 1, Resources::mem_gb(8.0)));
    /// let job = JobSpec::new(1, Resources::mem_gb(1.0), 0);
    /// shared.with(|p, now| p.handle(Input::Submit(job), now));
    /// let stats = shared.stats();
    /// assert_eq!((stats.placements_total, stats.running), (1, 0));
    /// ```
    pub fn with<R>(&self, f: impl FnOnce(&mut P, Instant) -> R) -> R {
        let (mut s, now) = self.lock();
        let r = f(&mut s.policy, now);
        Self::pump(&mut s, now);
        r
    }

    /// Threads blocked waiting for a start: in [`lease`](Self::lease), or in [`Lease::fail`] for
    /// a retry.
    pub fn waiting(&self) -> usize {
        self.lock()
            .0
            .jobs
            .values()
            .filter(|s| s.held.is_none())
            .count()
    }
}

impl<P: Policy + Send + 'static> SharedPolicy<P> {
    /// A thread that calls [`tick`](Self::tick) every `period`, or sooner when the policy asks
    /// ([`Policy::next_wakeup`]). It stops when the front end is dropped or
    /// [`stop_ticker`](Self::stop_ticker) is called.
    ///
    /// With a ticker, the reservation of the [`next_wakeup`](Self::next_wakeup) example is made
    /// without calling [`tick`](Self::tick):
    ///
    /// ```
    /// # use std::{
    /// #     sync::{Arc, Mutex},
    /// #     thread,
    /// #     time::Duration,
    /// # };
    /// # use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
    /// # let time = Arc::new(Mutex::new(0.0));
    /// # let clock = {
    /// #     let time = time.clone();
    /// #     move || *time.lock().unwrap()
    /// # };
    /// # let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::default()), clock));
    /// # shared.worker_update(WorkerState::new(1, "x", 4, Resources::mem_gb(8.0)));
    /// # let first = shared.lease(JobSpec::new(1, Resources::mem_gb(4.0), 0));
    /// # let big = thread::spawn({
    /// #     let shared = shared.clone();
    /// #     move || shared.lease(JobSpec::new(2, Resources::mem_gb(6.0), 0)).complete()
    /// # });
    /// # while shared.waiting() == 0 {
    /// #     thread::yield_now();
    /// # }
    /// let ticker = shared.spawn_ticker(Duration::from_millis(1));
    /// *time.lock().unwrap() = shared.next_wakeup().unwrap();
    /// while shared.stats().reservations.is_empty() {
    ///     thread::yield_now();
    /// }
    /// shared.stop_ticker();
    /// ticker.join().unwrap();
    /// # first.complete();
    /// # big.join().unwrap();
    /// ```
    pub fn spawn_ticker(self: &Arc<Self>, period: Duration) -> JoinHandle<()> {
        let weak: Weak<Self> = Arc::downgrade(self);
        let generation = self.lock().0.ticker_generation;
        std::thread::Builder::new()
            .name("whelm-ticker".into())
            .spawn(move || {
                loop {
                    let sleep = {
                        let Some(me) = weak.upgrade() else { return };
                        let (mut s, now) = me.lock();
                        if s.ticker_generation != generation {
                            return;
                        }
                        Self::pump(&mut s, now);
                        match s.policy.next_wakeup() {
                            Some(t) if t > now => period.min(Duration::from_secs_f64(t - now)),
                            _ => period,
                        }
                    };
                    std::thread::sleep(sleep.max(Duration::from_millis(1)));
                }
            })
            .expect("spawning the ticker thread")
    }

    /// Stop every ticker started by [`spawn_ticker`](Self::spawn_ticker) so far (each exits
    /// within a period). Tickers spawned afterwards run normally.
    pub fn stop_ticker(&self) {
        self.lock().0.ticker_generation += 1;
    }
}

/// A started attempt of a leased job (see [`SharedPolicy::lease`]). Dropping it without
/// [`complete`](Self::complete) or [`fail`](Self::fail) cancels the job.
///
/// A lease is a loop: run the attempt on [`worker`](Self::worker), and on failure take the lease
/// [`fail`](Self::fail) returns for the next attempt, until the job completes or is given up:
///
/// ```
/// use whelm::{
///     Config, FailKind, JobSpec, Resources, RetryConfig, Scheduler, SharedPolicy, WorkerState,
/// };
///
/// let config = Config {
///     retry: RetryConfig { max_attempts: 3 },
///     ..Config::fifo()
/// };
/// let shared = SharedPolicy::new(Scheduler::new(config), || 0.0);
/// for w in [1, 2] {
///     shared.worker_update(WorkerState::new(w, "x", 1, Resources::mem_gb(8.0)));
/// }
/// // Every attempt runs out of device memory.
/// let mut lease = shared.lease(JobSpec::new(7, Resources::mem_gb(1.0), 0));
/// let mut workers = Vec::new();
/// let gave_up = loop {
///     workers.push(lease.worker());
///     match lease.fail(FailKind::DeviceOom, "out of memory") {
///         Ok(retry) => lease = retry,
///         Err(gave_up) => break gave_up,
///     }
/// };
/// // Each retry softly avoids the workers already tried.
/// assert_eq!(workers, [1, 2, 1]);
/// assert_eq!(
///     (gave_up.job, gave_up.tried.len(), gave_up.retryable),
///     (7, 3, true)
/// );
/// ```
pub struct Lease<'a, P: Policy> {
    shared: &'a SharedPolicy<P>,
    job: JobId,
    attempt: Attempt,
    worker: WorkerId,
    waited: f64,
    open: bool,
}

impl<P: Policy> std::fmt::Debug for Lease<'_, P> {
    /// The attempt the lease holds; the front end it belongs to is left out.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lease")
            .field("job", &self.job)
            .field("attempt", &self.attempt)
            .field("worker", &self.worker)
            .field("waited", &self.waited)
            .finish()
    }
}

impl<'a, P: Policy> Lease<'a, P> {
    /// The worker to send the job to.
    pub fn worker(&self) -> WorkerId {
        self.worker
    }

    /// This attempt's number (1 for the first).
    pub fn attempt(&self) -> Attempt {
        self.attempt
    }

    /// Seconds (policy clock) from the request to this start: from the lease for the first
    /// attempt, from the failure for a retry.
    pub fn waited(&self) -> f64 {
        self.waited
    }

    /// Whether the policy stopped this attempt (the job was cancelled through
    /// [`SharedPolicy::with`]): its result is not wanted.
    ///
    /// ```
    /// use whelm::{Config, Input, JobSpec, Policy, Resources, Scheduler, SharedPolicy};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::default()), || 0.0);
    /// shared.worker_update(whelm::WorkerState::new(1, "x", 1, Resources::mem_gb(8.0)));
    /// let lease = shared.lease(JobSpec::new(1, Resources::mem_gb(1.0), 0));
    /// assert!(!lease.stopped());
    /// shared.with(|p, now| p.handle(Input::Cancel(1), now));
    /// assert!(lease.stopped());
    /// lease.complete();
    /// ```
    pub fn stopped(&self) -> bool {
        let (s, _) = self.shared.lock();
        s.jobs.get(&self.job).is_some_and(|slot| slot.stopped)
    }

    /// The job finished. If its worker left meanwhile ([`SharedPolicy::worker_gone`]), the
    /// policy's retry is cancelled instead: the result is in hand.
    ///
    /// ```
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::fifo()), || 0.0);
    /// for w in [1, 2] {
    ///     shared.worker_update(WorkerState::new(w, "x", 1, Resources::mem_gb(8.0)));
    /// }
    /// let lease = shared.lease(JobSpec::new(7, Resources::mem_gb(1.0), 0));
    /// shared.worker_gone(lease.worker());
    /// // The policy is running a retry on worker 2, but the result arrived anyway.
    /// assert_eq!(shared.stats().running, 1);
    /// lease.complete();
    /// assert_eq!(shared.stats().running, 0);
    /// ```
    pub fn complete(mut self) {
        self.open = false;
        let (mut s, now) = self.shared.lock();
        let lost = s.jobs.remove(&self.job).is_some_and(|slot| slot.lost);
        let input = if lost {
            Input::Cancel(self.job)
        } else {
            Input::Done {
                job: self.job,
                attempt: self.attempt,
            }
        };
        s.policy.handle(input, now);
        SharedPolicy::pump(&mut s, now);
    }

    /// The attempt failed: block until the policy starts the job again (the returned lease) or
    /// gives it up. After the worker left, the policy has already retried the job, and this
    /// returns that retry's start (or the give-up). The [type's example](Lease) runs this to
    /// give-up.
    pub fn fail(mut self, kind: FailKind, why: &str) -> Result<Lease<'a, P>, GaveUp> {
        self.open = false;
        let shared = self.shared;
        let (mut s, now) = shared.lock();
        let slot = s.jobs.get_mut(&self.job).expect("a leased job has a slot");
        slot.held = None;
        slot.lost = false;
        slot.stopped = false;
        slot.asked = now;
        let input = Input::Failed {
            job: self.job,
            attempt: self.attempt,
            kind,
            why: why.to_string(),
        };
        s.policy.handle(input, now);
        SharedPolicy::pump(&mut s, now);
        match shared.wait(s, self.job, None) {
            Ok(lease) => Ok(lease),
            Err(NoStart::GaveUp(g)) => Err(g),
            Err(NoStart::Timeout(_)) => unreachable!("no deadline"),
        }
    }
}

impl<P: Policy> Drop for Lease<'_, P> {
    /// Cancel the job if the lease was neither completed nor failed.
    fn drop(&mut self) {
        if self.open {
            let (mut s, now) = self.shared.lock();
            s.jobs.remove(&self.job);
            s.policy.handle(Input::Cancel(self.job), now);
            SharedPolicy::pump(&mut s, now);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;
    use crate::{Config, Resources, RetryConfig, Scheduler, Speculate};

    /// A one-slot worker of class "x".
    fn worker(id: WorkerId) -> WorkerState {
        WorkerState::new(id, "x", 1, Resources::mem(100))
    }

    /// A shared FIFO scheduler with `max_attempts` attempts per job, on a manual clock of 0.
    fn shared(max_attempts: u32) -> SharedPolicy<Scheduler> {
        SharedPolicy::new(
            Scheduler::new(Config {
                retry: RetryConfig { max_attempts },
                ..Config::fifo()
            }),
            || 0.0,
        )
    }

    /// Stopping the tickers ends those running; one spawned afterwards keeps running until it is
    /// stopped in turn.
    #[test]
    fn ticker_restarts_after_stop() {
        let s = Arc::new(shared(1));
        let first = s.spawn_ticker(Duration::from_millis(1));
        s.stop_ticker();
        first.join().unwrap();
        let second = s.spawn_ticker(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(20));
        assert!(!second.is_finished());
        s.stop_ticker();
        second.join().unwrap();
    }

    /// A job.
    fn job(id: JobId) -> JobSpec {
        JobSpec::new(id, Resources::mem(1), 0)
    }

    /// A failed lease is retried on another worker, then completes.
    #[test]
    fn fail_returns_the_retry() {
        let s = shared(4);
        s.worker_update(worker(1));
        s.worker_update(worker(2));
        let l = s.lease(job(7));
        assert_eq!((l.worker(), l.attempt()), (1, 1));
        let l = l.fail(FailKind::Other, "boom").unwrap();
        assert_eq!((l.worker(), l.attempt()), (2, 2));
        assert!(!l.stopped());
        l.complete();
        let st = s.stats();
        assert_eq!((st.waiting, st.running), (0, 0));
    }

    /// The last allowed failure returns the give-up.
    #[test]
    fn fail_gives_up() {
        let s = shared(2);
        s.worker_update(worker(1));
        let l = s.lease(job(7)).fail(FailKind::DeviceOom, "oom").unwrap();
        let g = l.fail(FailKind::DeviceOom, "oom").err().unwrap();
        assert_eq!((g.job, g.tried.len(), g.retryable), (7, 2, true));
        assert_eq!(s.stats().running, 0);
        // The id can be leased afresh.
        s.lease(job(7)).complete();
    }

    /// A thread blocked in `lease` is woken by a worker joining; one whose worker leaves gets the
    /// policy's retry when it reports the lost link, even though the policy failed that attempt
    /// first.
    #[test]
    fn worker_gone_then_fail_link_died() {
        let s = Arc::new(shared(4));
        let (tx, rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let t = {
            let s = s.clone();
            std::thread::spawn(move || {
                let l = s.lease(job(7));
                tx.send((l.worker(), l.attempt())).unwrap();
                go_rx.recv().unwrap();
                let l = l.fail(FailKind::LinkDied, "connection reset").unwrap();
                tx.send((l.worker(), l.attempt())).unwrap();
                l.complete();
            })
        };
        while s.waiting() == 0 {
            std::thread::yield_now();
        }
        s.worker_update(worker(1));
        assert_eq!(rx.recv().unwrap(), (1, 1));
        s.worker_update(worker(2));
        assert_eq!(s.worker_gone(1), vec![7]);
        // The policy already retried the job on worker 2.
        assert_eq!(s.stats().running, 1);
        go_tx.send(()).unwrap();
        assert_eq!(rx.recv().unwrap(), (2, 2));
        t.join().unwrap();
        let st = s.stats();
        assert_eq!((st.waiting, st.running), (0, 0));
        assert_eq!(st.placements_total, 2);
    }

    /// Completing an attempt whose worker left cancels the policy's retry: the result is in hand.
    #[test]
    fn complete_after_worker_gone_cancels_the_retry() {
        let s = shared(4);
        s.worker_update(worker(1));
        s.worker_update(worker(2));
        let l = s.lease(job(7));
        s.worker_gone(l.worker());
        assert_eq!(s.stats().running, 1);
        l.complete();
        let st = s.stats();
        assert_eq!((st.waiting, st.running), (0, 0));
        assert!(st.workers.iter().all(|w| w.running == 0));
    }

    /// A speculative second attempt is rejected: the lease keeps its attempt and completes it.
    #[test]
    fn speculative_start_is_rejected() {
        let mut cfg = Config::fifo();
        cfg.speed.speculate = Some(Speculate::default());
        let s = SharedPolicy::new(Scheduler::new(cfg), || 0.0);
        s.worker_update(worker(1));
        let mut j = job(7);
        j.work = Some(100.0);
        let l = s.lease(j);
        s.worker_update(WorkerState {
            speed: 4.0,
            ..worker(2)
        });
        let st = s.stats();
        assert_eq!(st.placements_total, 2);
        assert!(st.workers.iter().all(|w| w.running == (w.id == 1) as usize));
        assert_eq!(l.attempt(), 1);
        l.complete();
        assert_eq!(s.stats().running, 0);
    }

    /// Dropping a lease cancels its job; a timed lease returns the job when it expires.
    #[test]
    fn drop_and_timeout_cancel() {
        let s = shared(4);
        s.worker_update(worker(1));
        drop(s.lease(job(1)));
        assert_eq!(s.stats().running, 0);
        let held = s.lease(job(2));
        let back = s.lease_timeout(job(3), Duration::from_millis(10)).err();
        assert_eq!(back.map(|j| j.id), Some(3));
        assert_eq!(s.stats().waiting, 0);
        held.complete();
    }

    /// A lease whose job is cancelled through `with` sees the stop.
    #[test]
    fn stop_is_visible() {
        let s = shared(4);
        s.worker_update(worker(1));
        let l = s.lease(job(1));
        s.with(|p, now| p.handle(Input::Cancel(1), now));
        assert!(l.stopped());
        l.complete();
        assert_eq!(s.stats().running, 0);
    }
}
