//! A thread-safe, blocking front end over a [`Policy`], for callers with a thread per task.

use std::{
    collections::HashMap,
    sync::{Arc, Condvar, Mutex, MutexGuard, Weak},
    thread::JoinHandle,
    time::Duration,
};

use crate::{Instant, JobId, JobSpec, Policy, PolicyStats, WorkerId, WorkerState};

/// Why a placed job failed (the caller classifies; the policy only counts).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FailKind {
    /// The worker ran out of device memory.
    DeviceOom,
    /// The connection to the worker died.
    LinkDied,
    /// The worker refused the job.
    Rejected,
    /// The job took too long.
    Timeout,
    /// Anything else.
    Other,
}

/// Retry settings for [`SharedPolicy::failed`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryConfig {
    /// Attempts per job, the first included. Default 4.
    pub max_attempts: u32,
}

impl Default for RetryConfig {
    /// Four attempts.
    fn default() -> Self {
        Self { max_attempts: 4 }
    }
}

/// One failed attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attempt {
    /// Where it ran.
    pub worker: WorkerId,
    /// How it failed.
    pub kind: FailKind,
    /// The caller's description.
    pub why: String,
}

/// What to do after a failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailOutcome {
    /// Place the job again: the next [`SharedPolicy::place`] of the same id avoids `avoid` (the
    /// workers tried so far), softly: they are used only while no other live worker exists.
    Retry {
        /// Attempts made so far.
        attempts: u32,
        /// Workers tried so far.
        avoid: Vec<WorkerId>,
    },
    /// `max_attempts` reached; the job's attempt history is forgotten.
    GiveUp {
        /// Every attempt, in order.
        tried: Vec<Attempt>,
        /// Every attempt failed with [`FailKind::DeviceOom`]: the job might fit later, or
        /// elsewhere, or split.
        retryable: bool,
    },
}

/// Where a job was placed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// The worker to send it to.
    pub worker: WorkerId,
    /// This is attempt number `attempt` (1 for the first).
    pub attempt: u32,
    /// Seconds from submission to placement.
    pub waited: f64,
}

/// The state behind the lock.
struct State<P> {
    policy: P,
    now: Instant,
    /// Jobs submitted by a `place` call that has not returned: their wake handle.
    waiters: HashMap<JobId, Arc<Condvar>>,
    /// Placements not yet picked up by their waiter.
    ready: HashMap<JobId, WorkerId>,
    /// Submission times of waiting jobs.
    since: HashMap<JobId, Instant>,
    /// Jobs placed and not yet completed, failed or abandoned, with their worker (kept after the
    /// worker leaves, until the caller reports the outcome).
    placed: HashMap<JobId, WorkerId>,
    /// Failed attempts of jobs that may be retried.
    attempts: HashMap<JobId, Vec<Attempt>>,
    retry: RetryConfig,
    stopped: bool,
}

/// A thread-safe front end over a [`Policy`]: each task thread calls [`place`](Self::place),
/// which submits its job and blocks until the policy places it, then reports the outcome with
/// [`completed`](Self::completed) or [`failed`](Self::failed).
///
/// `dispatch` runs after every mutating call and on [`tick`](Self::tick) (aging, reservations and
/// voluntary waits need time to pass without events; [`spawn_ticker`](Self::spawn_ticker) does it
/// on a thread). Each placement wakes exactly the thread waiting for that job. `dispatch` is used,
/// never `dispatch_full`: a thread-per-task caller cannot move a running job.
///
/// The policy stays deterministic; only the clock and the interleaving of callers are not.
pub struct SharedPolicy<P> {
    state: Mutex<State<P>>,
    clock: Box<dyn Fn() -> Instant + Send + Sync>,
}

impl<P: Policy> SharedPolicy<P> {
    /// A front end over `policy`, reading time from `clock` (seconds; made non-decreasing here).
    pub fn new(policy: P, clock: impl Fn() -> Instant + Send + Sync + 'static) -> Self {
        Self::with_retry(policy, clock, RetryConfig::default())
    }

    /// [`new`](Self::new) with retry settings.
    pub fn with_retry(
        policy: P,
        clock: impl Fn() -> Instant + Send + Sync + 'static,
        retry: RetryConfig,
    ) -> Self {
        Self {
            state: Mutex::new(State {
                policy,
                now: f64::NEG_INFINITY,
                waiters: HashMap::new(),
                ready: HashMap::new(),
                since: HashMap::new(),
                placed: HashMap::new(),
                attempts: HashMap::new(),
                retry,
                stopped: false,
            }),
            clock: Box::new(clock),
        }
    }

    /// A front end whose clock is seconds since its creation.
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

    /// Dispatch and hand placements to their waiters.
    fn dispatch(s: &mut State<P>, now: Instant) {
        for (job, worker) in s.policy.dispatch(now) {
            s.placed.insert(job, worker);
            match s.waiters.get(&job) {
                Some(cv) => {
                    s.ready.insert(job, worker);
                    cv.notify_one();
                }
                // Nobody waits (cannot happen through this front end); keep it consistent.
                None => {
                    s.placed.remove(&job);
                    s.policy.cancel(job);
                }
            }
        }
    }

    /// Submit `job` and wait for its placement, at most until `deadline` (seconds of
    /// `std::time`, not of the policy clock).
    fn place_until(
        &self,
        mut job: JobSpec,
        deadline: Option<std::time::Instant>,
    ) -> Result<Placement, Box<JobSpec>> {
        let id = job.id;
        let (mut s, now) = self.lock();
        assert!(
            !s.waiters.contains_key(&id) && !s.placed.contains_key(&id),
            "job {id} is already waiting or running"
        );
        let attempt = s.attempts.get(&id).map_or(0, Vec::len) as u32 + 1;
        if let Some(tried) = s.attempts.get(&id) {
            for a in tried {
                if !job.avoid.contains(&a.worker) {
                    job.avoid.push(a.worker);
                }
            }
            job.avoid_soft = true;
        }
        let cv = Arc::new(Condvar::new());
        s.waiters.insert(id, cv.clone());
        s.since.insert(id, now);
        s.policy.submit(job.clone(), now);
        Self::dispatch(&mut s, now);
        loop {
            if let Some(worker) = s.ready.remove(&id) {
                s.waiters.remove(&id);
                let since = s.since.remove(&id).unwrap_or(s.now);
                return Ok(Placement {
                    worker,
                    attempt,
                    waited: s.now - since,
                });
            }
            match deadline {
                None => s = cv.wait(s).unwrap_or_else(|e| e.into_inner()),
                Some(d) => {
                    let left = d.saturating_duration_since(std::time::Instant::now());
                    if left.is_zero() {
                        s.waiters.remove(&id);
                        s.since.remove(&id);
                        s.policy.cancel(id);
                        return Err(Box::new(job));
                    }
                    s = cv
                        .wait_timeout(s, left)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
            }
        }
    }

    /// Submit `job` and block until it is placed. If the job failed before (see
    /// [`failed`](Self::failed)), it avoids the workers it was tried on, softly.
    ///
    /// # Panics
    ///
    /// If a job with this id is already waiting or running here.
    pub fn place(&self, job: JobSpec) -> Placement {
        self.place_until(job, None)
            .unwrap_or_else(|_| unreachable!("no deadline"))
    }

    /// [`place`](Self::place), giving up after `timeout`: the job is then withdrawn from the
    /// policy and returned. A placement made concurrently with the timeout is still returned.
    pub fn place_timeout(
        &self,
        job: JobSpec,
        timeout: Duration,
    ) -> Result<Placement, Box<JobSpec>> {
        self.place_until(job, Some(std::time::Instant::now() + timeout))
    }

    /// [`place`](Self::place), returning a guard that abandons the job (releasing its resources)
    /// if dropped before [`Lease::complete`] or [`Lease::fail`], e.g. when the caller unwinds.
    pub fn lease(&self, job: JobSpec) -> Lease<'_, P> {
        let id = job.id;
        let placement = self.place(job);
        Lease {
            shared: self,
            job: id,
            placement,
            open: true,
        }
    }

    /// A placed job finished successfully: its resources are released and its attempt history
    /// forgotten.
    pub fn completed(&self, job: JobId) {
        let (mut s, now) = self.lock();
        s.placed.remove(&job);
        s.attempts.remove(&job);
        s.policy.completed(job, now);
        Self::dispatch(&mut s, now);
    }

    /// A placed job failed. Its resources are released (without learning a speed from it) and
    /// the attempt is recorded against its worker.
    pub fn failed(&self, job: JobId, kind: FailKind, why: &str) -> FailOutcome {
        let (mut s, now) = self.lock();
        let worker = s.placed.remove(&job);
        s.policy.failed(job, now, &format!("{kind:?}: {why}"));
        let max = s.retry.max_attempts.max(1);
        let tried = s.attempts.entry(job).or_default();
        if let Some(worker) = worker {
            tried.push(Attempt {
                worker,
                kind,
                why: why.to_string(),
            });
        }
        let n = tried.len() as u32;
        let out = if n >= max {
            let tried = s.attempts.remove(&job).unwrap_or_default();
            let retryable =
                !tried.is_empty() && tried.iter().all(|a| a.kind == FailKind::DeviceOom);
            FailOutcome::GiveUp { tried, retryable }
        } else {
            let mut avoid: Vec<WorkerId> = tried.iter().map(|a| a.worker).collect();
            avoid.dedup();
            FailOutcome::Retry { attempts: n, avoid }
        };
        Self::dispatch(&mut s, now);
        out
    }

    /// Forget a placed job without recording an attempt (the caller gave up on it): its
    /// resources are released and its attempt history dropped.
    pub fn abandon(&self, job: JobId) {
        let (mut s, now) = self.lock();
        s.placed.remove(&job);
        s.attempts.remove(&job);
        s.policy.cancel(job);
        Self::dispatch(&mut s, now);
    }

    /// A worker joined or reported a heartbeat.
    pub fn worker_update(&self, w: WorkerState) {
        let (mut s, now) = self.lock();
        s.policy.worker_update(w, now);
        Self::dispatch(&mut s, now);
    }

    /// A worker left. Returns the jobs placed there whose outcome the caller has not reported
    /// (they will typically fail with [`FailKind::LinkDied`]); report each with `failed` or
    /// `completed` as usual.
    pub fn worker_gone(&self, w: WorkerId) -> Vec<JobId> {
        let (mut s, now) = self.lock();
        s.policy.worker_gone(w, now);
        let mut jobs: Vec<JobId> = s
            .placed
            .iter()
            .filter(|&(_, &x)| x == w)
            .map(|(&j, _)| j)
            .collect();
        jobs.sort_unstable();
        Self::dispatch(&mut s, now);
        jobs
    }

    /// Dispatch now (time has passed).
    pub fn tick(&self) {
        let (mut s, now) = self.lock();
        Self::dispatch(&mut s, now);
    }

    /// When the policy next needs a dispatch without events, on the policy clock.
    pub fn next_wakeup(&self) -> Option<Instant> {
        self.lock().0.policy.next_wakeup()
    }

    /// Why a waiting job is not placed.
    pub fn explain(&self, job: JobId) -> Option<String> {
        self.lock().0.policy.explain(job)
    }

    /// The policy's counters.
    pub fn stats(&self) -> PolicyStats {
        self.lock().0.policy.stats()
    }

    /// Run `f` on the policy under the lock (e.g. to forget a group), then dispatch.
    pub fn with<R>(&self, f: impl FnOnce(&mut P, Instant) -> R) -> R {
        let (mut s, now) = self.lock();
        let r = f(&mut s.policy, now);
        Self::dispatch(&mut s, now);
        r
    }

    /// Jobs blocked in `place`.
    pub fn waiting(&self) -> usize {
        self.lock().0.waiters.len()
    }
}

impl<P: Policy + Send + 'static> SharedPolicy<P> {
    /// A thread that calls [`tick`](Self::tick) every `period`, or sooner when the policy asks
    /// ([`Policy::next_wakeup`]). It stops when the front end is dropped or
    /// [`stop_ticker`](Self::stop_ticker) is called.
    pub fn spawn_ticker(self: &Arc<Self>, period: Duration) -> JoinHandle<()> {
        let weak: Weak<Self> = Arc::downgrade(self);
        std::thread::Builder::new()
            .name("sched-ticker".into())
            .spawn(move || {
                loop {
                    let sleep = {
                        let Some(me) = weak.upgrade() else { return };
                        let (mut s, now) = me.lock();
                        if s.stopped {
                            return;
                        }
                        Self::dispatch(&mut s, now);
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

    /// Stop a ticker started by [`spawn_ticker`](Self::spawn_ticker) (it exits within a period).
    pub fn stop_ticker(&self) {
        self.lock().0.stopped = true;
    }
}

/// A placed job that is abandoned unless completed or failed (see [`SharedPolicy::lease`]).
pub struct Lease<'a, P: Policy> {
    shared: &'a SharedPolicy<P>,
    job: JobId,
    placement: Placement,
    open: bool,
}

impl<P: Policy> Lease<'_, P> {
    /// Where the job was placed.
    pub fn placement(&self) -> Placement {
        self.placement
    }

    /// The worker to send the job to.
    pub fn worker(&self) -> WorkerId {
        self.placement.worker
    }

    /// The job finished: see [`SharedPolicy::completed`].
    pub fn complete(mut self) {
        self.open = false;
        self.shared.completed(self.job);
    }

    /// The job failed: see [`SharedPolicy::failed`].
    pub fn fail(mut self, kind: FailKind, why: &str) -> FailOutcome {
        self.open = false;
        self.shared.failed(self.job, kind, why)
    }
}

impl<P: Policy> Drop for Lease<'_, P> {
    /// Abandon the job if it was neither completed nor failed.
    fn drop(&mut self) {
        if self.open {
            self.shared.abandon(self.job);
        }
    }
}
