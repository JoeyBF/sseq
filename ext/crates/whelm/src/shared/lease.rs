//! A leased attempt, held by the thread running it.

use super::{SharedPolicy, mailbox::NoStart};
use crate::{Attempt, FailKind, GaveUp, Input, JobId, Policy, WorkerId};

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
///     shared.worker_update(WorkerState {
///         id: w,
///         budget: Resources::mem_gb(8.0),
///         ..Default::default()
///     });
/// }
/// // Every attempt runs out of device memory.
/// let mut lease = shared.lease(JobSpec {
///     id: 7,
///     demand: Resources::mem_gb(1.0),
///     ..Default::default()
/// });
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
    pub(super) shared: &'a SharedPolicy<P>,
    pub(super) job: JobId,
    pub(super) attempt: Attempt,
    pub(super) worker: WorkerId,
    pub(super) waited: f64,
    pub(super) open: bool,
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
    /// shared.worker_update(whelm::WorkerState {
    ///     id: 1,
    ///     budget: Resources::mem_gb(8.0),
    ///     ..Default::default()
    /// });
    /// let lease = shared.lease(JobSpec {
    ///     id: 1,
    ///     demand: Resources::mem_gb(1.0),
    ///     ..Default::default()
    /// });
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
    ///     shared.worker_update(WorkerState {
    ///         id: w,
    ///         budget: Resources::mem_gb(8.0),
    ///         ..Default::default()
    ///     });
    /// }
    /// let lease = shared.lease(JobSpec {
    ///     id: 7,
    ///     demand: Resources::mem_gb(1.0),
    ///     ..Default::default()
    /// });
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
