//! The state behind the lock: one mailbox per leased job, and delivering the policy's outputs
//! to them.

use std::{
    collections::HashMap,
    sync::{Arc, Condvar, MutexGuard},
};

use super::{SharedPolicy, lease::Lease};
use crate::{
    job::{JobId, JobSpec},
    policy::{Attempt, FailKind, GaveUp, Input, Output, Policy, Rejection},
    time::Time,
    worker::WorkerId,
};

/// One leased job's mailbox, from its first submission until the lease ends.
pub(super) struct Slot {
    /// Wakes the job's thread.
    pub(super) cv: Arc<Condvar>,
    /// The job as submitted, returned if a timed lease expires.
    pub(super) spec: JobSpec,
    /// When the thread started waiting for the next attempt (lease or failure).
    pub(super) asked: Time,
    /// The attempt the thread holds, while it holds one.
    pub(super) held: Option<(Attempt, WorkerId)>,
    /// The held attempt was failed by its worker leaving: the policy treats it as stale.
    pub(super) lost: bool,
    /// A stop arrived for the held attempt.
    pub(super) stopped: bool,
    /// The next attempt, not yet picked up by the thread.
    pub(super) started: Option<(Attempt, WorkerId)>,
    /// The policy gave the job up, not yet picked up by the thread.
    pub(super) gave_up: Option<GaveUp>,
    /// The policy rejected the job, not yet picked up by the thread.
    pub(super) rejected: Option<Rejection>,
}

/// The state behind the lock.
pub(super) struct State<P> {
    pub(super) policy: P,
    /// The latest time the clock gave; `Time::ORIGIN`, the earliest, until the first lock.
    pub(super) now: Time,
    /// Jobs with a lease, held or awaited.
    pub(super) jobs: HashMap<JobId, Slot>,
    /// Tickers spawned before this generation exit.
    pub(super) ticker_generation: u64,
}

/// Why waiting for a start ended without one.
pub(super) enum NoStart {
    GaveUp(GaveUp),
    Rejected(Rejection),
    Timeout(Box<JobSpec>),
}

impl<P: Policy> SharedPolicy<P> {
    /// Lock the state and advance its clock. A poisoned lock (a caller panicked inside the
    /// policy) is taken over: the policy's own state is updated atomically per call.
    pub(super) fn lock(&self) -> (MutexGuard<'_, State<P>>, Time) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let now = (self.clock)().max(s.now);
        s.now = now;
        (s, now)
    }

    /// Poll, and deliver each output to its job's slot. Rejecting a speculative start, or
    /// cancelling a start nobody waits for, frees a slot, so polling repeats until neither
    /// happens.
    pub(super) fn pump(s: &mut State<P>, now: Time) {
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
                    Output::Rejected { job, reason } => {
                        if let Some(slot) = s.jobs.get_mut(&job) {
                            slot.rejected = Some(reason);
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
    pub(super) fn wait<'a>(
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
            if let Some(reason) = slot.rejected.take() {
                s.jobs.remove(&id);
                return Err(NoStart::Rejected(reason));
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

    /// Submit job `id` and wait for its start, at most until `deadline`.
    pub(super) fn lease_until(
        &self,
        id: JobId,
        spec: JobSpec,
        deadline: Option<std::time::Instant>,
    ) -> Result<Lease<'_, P>, Box<JobSpec>> {
        let (mut s, now) = self.lock();
        assert!(!s.jobs.contains_key(&id), "job {id} is already leased");
        s.jobs.insert(
            id,
            Slot {
                cv: Arc::new(Condvar::new()),
                spec: spec.clone(),
                asked: now,
                held: None,
                lost: false,
                stopped: false,
                started: None,
                gave_up: None,
                rejected: None,
            },
        );
        s.policy.handle(Input::Submit { job: id, spec }, now);
        Self::pump(&mut s, now);
        match self.wait(s, id, deadline) {
            Ok(lease) => Ok(lease),
            Err(NoStart::Timeout(spec)) => Err(spec),
            Err(NoStart::GaveUp(_)) => unreachable!("a job is given up only after failing"),
            Err(NoStart::Rejected(reason)) => panic!("job {id} was rejected: {reason}"),
        }
    }
}
