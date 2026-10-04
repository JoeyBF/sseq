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
//! use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, Time, WorkerState};
//!
//! let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::fifo()), || {
//!     Time::ORIGIN
//! }));
//! shared.worker_update(WorkerState {
//!     id: 1,
//!     capacity: Resources::mem_gb(8.0).with_slots(1),
//!     ..Default::default()
//! });
//! let job = |id| JobSpec {
//!     id,
//!     demand: Resources::mem_gb(1.0),
//!     ..Default::default()
//! };
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
//! assert_eq!(
//!     shared.explain(2).unwrap().waiting().unwrap().workers,
//!     [(
//!         1,
//!         whelm::Verdict::Full {
//!             dims: vec![whelm::ResourceId::SLOTS]
//!         }
//!     )]
//! );
//! first.complete();
//! assert_eq!(second.join().unwrap(), 1);
//! assert_eq!(shared.stats().placements_total, 2);
//! ```

mod lease;
mod mailbox;
#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
    thread::JoinHandle,
    time::Duration,
};

pub use lease::Lease;
use mailbox::State;

#[cfg(doc)]
use crate::FailKind;
use crate::{Explanation, Input, JobId, JobSpec, Policy, PolicyStats, Time, WorkerId, WorkerState};

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
    clock: Box<dyn Fn() -> Time + Send + Sync>,
}

impl<P: Policy> SharedPolicy<P> {
    /// A front end over `policy`, reading time from `clock` (any origin; made non-decreasing here).
    ///
    /// A clock the caller sets makes time-dependent behaviour reproducible. Here the second job
    /// waits while the clock moves to 7 s, and a clock going back to 3 s is read as 7 s:
    ///
    /// ```
    /// use std::{
    ///     sync::{Arc, Mutex},
    ///     thread,
    ///     time::Duration,
    /// };
    ///
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, Time, WorkerState};
    ///
    /// let time = Arc::new(Mutex::new(Time::ORIGIN));
    /// let clock = {
    ///     let time = time.clone();
    ///     move || *time.lock().unwrap()
    /// };
    /// let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::default()), clock));
    /// shared.worker_update(WorkerState {
    ///     id: 1,
    ///     capacity: Resources::mem_gb(8.0).with_slots(1),
    ///     ..Default::default()
    /// });
    /// let job = |id| JobSpec {
    ///     id,
    ///     demand: Resources::mem_gb(1.0),
    ///     ..Default::default()
    /// };
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
    /// *time.lock().unwrap() = Time(Duration::from_secs(7));
    /// shared.tick();
    /// *time.lock().unwrap() = Time(Duration::from_secs(3));
    /// first.complete();
    /// assert_eq!(second.join().unwrap(), Duration::from_secs(7));
    /// assert_eq!(shared.stats().now, Time(Duration::from_secs(7)));
    /// ```
    pub fn new(policy: P, clock: impl Fn() -> Time + Send + Sync + 'static) -> Self {
        Self {
            state: Mutex::new(State {
                policy,
                now: Time::ORIGIN,
                jobs: HashMap::new(),
                ticker_generation: 0,
            }),
            clock: Box::new(clock),
        }
    }

    /// A front end whose clock reads the time since its creation.
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
    ///
    /// let shared = SharedPolicy::with_system_clock(Scheduler::new(Config::default()));
    /// shared.worker_update(WorkerState {
    ///     id: 1,
    ///     capacity: Resources::mem_gb(8.0).with_slots(4),
    ///     ..Default::default()
    /// });
    /// let lease = shared.lease(JobSpec {
    ///     id: 1,
    ///     demand: Resources::mem_gb(1.0),
    ///     ..Default::default()
    /// });
    /// assert!(lease.waited() < Duration::from_secs(60)); // the worker was free
    /// lease.complete();
    /// ```
    pub fn with_system_clock(policy: P) -> Self {
        let start = std::time::Instant::now();
        Self::new(policy, move || Time::ORIGIN + start.elapsed())
    }

    /// Submit `job` and block until the policy starts it. Dropping the lease without
    /// [`complete`](Lease::complete) or [`fail`](Lease::fail) (e.g. when the caller unwinds)
    /// cancels the job.
    ///
    /// ```
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, Time, WorkerState};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::default()), || Time::ORIGIN);
    /// shared.worker_update(WorkerState {
    ///     id: 1,
    ///     capacity: Resources::mem_gb(8.0).with_slots(1),
    ///     ..Default::default()
    /// });
    /// let lease = shared.lease(JobSpec {
    ///     id: 1,
    ///     demand: Resources::mem_gb(1.0),
    ///     ..Default::default()
    /// });
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
    /// policy and returned. A start made concurrently with the timeout is still returned. A
    /// `timeout` too long for an [`Instant`](std::time::Instant) never expires.
    ///
    /// With no worker, the job never starts:
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, Time};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::default()), || Time::ORIGIN);
    /// let job = JobSpec {
    ///     id: 1,
    ///     demand: Resources::mem_gb(1.0),
    ///     ..Default::default()
    /// };
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
        self.lease_until(job, std::time::Instant::now().checked_add(timeout))
    }

    /// A worker joined or reported a heartbeat.
    ///
    /// A worker joining wakes the threads whose jobs it starts:
    ///
    /// ```
    /// use std::{sync::Arc, thread};
    ///
    /// use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, Time, WorkerState};
    ///
    /// let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::default()), || {
    ///     Time::ORIGIN
    /// }));
    /// let task = thread::spawn({
    ///     let shared = shared.clone();
    ///     move || {
    ///         let lease = shared.lease(JobSpec {
    ///             id: 1,
    ///             demand: Resources::mem_gb(1.0),
    ///             ..Default::default()
    ///         });
    ///         let worker = lease.worker();
    ///         lease.complete();
    ///         worker
    ///     }
    /// });
    /// while shared.waiting() == 0 {
    ///     thread::yield_now();
    /// }
    /// shared.worker_update(WorkerState {
    ///     id: 5,
    ///     capacity: Resources::mem_gb(8.0).with_slots(1),
    ///     ..Default::default()
    /// });
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
    /// use whelm::{Config, FailKind, JobSpec, Resources, Scheduler, SharedPolicy, Time, WorkerState};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::fifo()), || Time::ORIGIN);
    /// for w in [1, 2] {
    ///     shared.worker_update(WorkerState {
    ///         id: w,
    ///         capacity: Resources::mem_gb(8.0).with_slots(1),
    ///         ..Default::default()
    ///     });
    /// }
    /// let lease = shared.lease(JobSpec {
    ///     id: 7,
    ///     demand: Resources::mem_gb(1.0),
    ///     ..Default::default()
    /// });
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
    /// use whelm::{
    ///     Config, JobSpec, Reservations, Resources, Scheduler, SharedPolicy, Time, WorkerState,
    /// };
    ///
    /// let time = Arc::new(Mutex::new(Time::ORIGIN));
    /// let clock = {
    ///     let time = time.clone();
    ///     move || *time.lock().unwrap()
    /// };
    /// let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::default()), clock));
    /// shared.worker_update(WorkerState {
    ///     id: 1,
    ///     capacity: Resources::mem_gb(8.0).with_slots(4),
    ///     ..Default::default()
    /// });
    /// let first = shared.lease(JobSpec {
    ///     id: 1,
    ///     demand: Resources::mem_gb(4.0),
    ///     ..Default::default()
    /// });
    /// let big = thread::spawn({
    ///     let shared = shared.clone();
    ///     move || {
    ///         shared
    ///             .lease(JobSpec {
    ///                 id: 2,
    ///                 demand: Resources::mem_gb(6.0),
    ///                 ..Default::default()
    ///             })
    ///             .complete()
    ///     }
    /// });
    /// while shared.waiting() == 0 {
    ///     thread::yield_now();
    /// }
    ///
    /// let wake = shared.next_wakeup().unwrap();
    /// assert_eq!(wake, Time::ORIGIN + Reservations::default().reserve_after);
    /// *time.lock().unwrap() = wake;
    /// shared.tick();
    /// assert_eq!(shared.stats().reservations[0].job, 2);
    /// assert!(matches!(
    ///     shared.explain(2).unwrap().waiting().unwrap().hold,
    ///     Some(whelm::Holding::Reservation { worker: 1, .. })
    /// ));
    /// first.complete();
    /// big.join().unwrap();
    /// ```
    pub fn next_wakeup(&self) -> Option<Time> {
        self.lock().0.policy.next_wakeup()
    }

    /// Why a job is not running ([`Policy::explain`]); see the [module example](self).
    pub fn explain(&self, job: JobId) -> Option<Explanation> {
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
    /// use whelm::{Config, Input, JobSpec, Policy, Resources, Scheduler, SharedPolicy, Time};
    ///
    /// let shared = SharedPolicy::new(Scheduler::new(Config::default()), || Time::ORIGIN);
    /// shared.worker_update(whelm::WorkerState {
    ///     id: 1,
    ///     capacity: Resources::mem_gb(8.0).with_slots(1),
    ///     ..Default::default()
    /// });
    /// let job = JobSpec {
    ///     id: 1,
    ///     demand: Resources::mem_gb(1.0),
    ///     ..Default::default()
    /// };
    /// shared.with(|p, now| p.handle(Input::Submit(job), now));
    /// let stats = shared.stats();
    /// assert_eq!((stats.placements_total, stats.running), (1, 0));
    /// ```
    pub fn with<R>(&self, f: impl FnOnce(&mut P, Time) -> R) -> R {
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
    /// # use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, Time, WorkerState};
    /// # let time = Arc::new(Mutex::new(Time::ORIGIN));
    /// # let clock = {
    /// #     let time = time.clone();
    /// #     move || *time.lock().unwrap()
    /// # };
    /// # let shared = Arc::new(SharedPolicy::new(Scheduler::new(Config::default()), clock));
    /// # let job = |id, gb| JobSpec { id, demand: Resources::mem_gb(gb), ..Default::default() };
    /// # shared.worker_update(WorkerState {
    /// #     id: 1,
    /// #     capacity: Resources::mem_gb(8.0).with_slots(4),
    /// #     ..Default::default()
    /// # });
    /// # let first = shared.lease(job(1, 4.0));
    /// # let big = thread::spawn({
    /// #     let shared = shared.clone();
    /// #     move || shared.lease(job(2, 6.0)).complete()
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
                            Some(t) if t > now => period.min(t - now),
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
