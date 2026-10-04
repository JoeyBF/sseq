//! The messages between a caller and a [`Policy`], and the trait itself.

use std::{borrow::Cow, fmt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{
    Config, DagConfig, DagJob, DagScheduler, Defer, RetryConfig, Scheduler, Speculate, log,
};
use crate::{Explanation, JobId, JobSpec, PolicyStats, Time, WorkerId, WorkerState};

/// The number of a job's attempt: 1 for its first start, counting retries and speculative
/// attempts. The DAG layer's local jobs use 0 (see [`DagScheduler`]).
///
/// [`Input::Done`] and [`Input::Failed`] name the attempt they report, and are ignored unless it
/// is live, so a late report about a superseded attempt changes nothing.
pub type Attempt = u32;

/// Why an attempt failed (the caller classifies; the policy records it in [`Tried`]).
///
/// Only [`DeviceOom`](Self::DeviceOom) changes what the policy does: it makes a give-up
/// [`retryable`](GaveUp::retryable).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FailKind {
    /// The worker ran out of device memory.
    DeviceOom,
    /// The connection to the worker died ([`Input::WorkerGone`] fails attempts with this).
    LinkDied,
    /// The worker refused the job.
    Rejected,
    /// The job took too long.
    Timeout,
    /// Anything else.
    Other,
}

/// One failed attempt.
///
/// # Examples
///
/// A worker leaving is recorded like a reported failure.
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{
///     Config, FailKind, GaveUp, Input, JobSpec, Output, Policy, Resources, RetryConfig, SLOTS,
///     Scheduler, Time, Tried, WorkerState,
/// };
///
/// let config = Config {
///     retry: RetryConfig { max_attempts: 1 },
///     ..Config::default()
/// };
/// let mut p = Scheduler::new(config);
/// p.handle(
///     Input::Worker(WorkerState {
///         id: 3,
///         class: "cpu".into(),
///         capacity: Resources::new().with(SLOTS, 1),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// p.handle(
///     Input::Submit(JobSpec {
///         id: 1,
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// p.poll(Time::ORIGIN);
/// p.handle(Input::WorkerGone(3), Time(Duration::from_secs(1)));
/// let tried = Tried {
///     worker: 3,
///     kind: FailKind::LinkDied,
///     why: "worker 3 left".into(),
/// };
/// assert_eq!(
///     p.poll(Time(Duration::from_secs(1))),
///     [Output::GaveUp(GaveUp {
///         job: 1,
///         tried: vec![tried],
///         retryable: false
///     })]
/// );
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Tried {
    /// Where it ran.
    pub worker: WorkerId,
    /// How it failed.
    pub kind: FailKind,
    /// The caller's description.
    pub why: String,
}

/// A job the policy stopped retrying ([`RetryConfig::max_attempts`] rounds): it is forgotten.
///
/// The crate's [Messages and attempts](crate#messages-and-attempts) chapter shows a job retried
/// and given up; [`Tried`] has a short example.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GaveUp {
    /// The job.
    pub job: JobId,
    /// Every failed attempt, in order.
    pub tried: Vec<Tried>,
    /// Every attempt failed with [`FailKind::DeviceOom`]: the job might fit later, elsewhere, or
    /// split.
    pub retryable: bool,
}

/// Why a policy refused a job at submission ([`Output::Rejected`]).
///
/// # Examples
///
/// A job demanding GPUs from a scheduler that does not declare them:
///
/// ```
/// use whelm::{
///     Config, Input, JobSpec, Output, Policy, Rejection, Resource, Resources, Scheduler, Time,
/// };
///
/// const GPUS: Resource = Resource::new("gpus").hard();
///
/// let mut p = Scheduler::new(Config::default());
/// let job = JobSpec {
///     id: 1,
///     demand: Resources::new().with(GPUS, 1),
///     ..Default::default()
/// };
/// p.handle(Input::Submit(job), Time::ORIGIN);
/// let reason = Rejection::Undeclared {
///     resource: "gpus".into(),
/// };
/// assert_eq!(
///     reason.to_string(),
///     "its demand names resource \"gpus\", which the configuration does not declare"
/// );
/// assert_eq!(p.poll(Time::ORIGIN), [Output::Rejected { job: 1, reason }]);
/// assert_eq!(p.explain(1), None);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Rejection {
    /// The job's demand names a resource that [`Config::resources`] does not declare: the first
    /// such name, in name order.
    Undeclared {
        /// The resource's name.
        resource: Cow<'static, str>,
    },
}

impl fmt::Display for Rejection {
    /// The reason as a clause about the job.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Undeclared { resource } => write!(
                f,
                "its demand names resource {resource:?}, which the configuration does not declare"
            ),
        }
    }
}

/// An event a [`Policy`] reacts to.
///
/// Every input is applied by [`Policy::handle`] at once and is safe to repeat or deliver late:
/// what no longer applies (an unknown job, an attempt that is not live) is ignored.
///
/// # Examples
///
/// Inputs serialise (feature `serde`) as externally tagged snake-case variants, the form logs use.
///
/// ```
/// # #[cfg(feature = "serde")]
/// # fn main() {
/// use whelm::Input;
///
/// let json = serde_json::to_string(&Input::Done { job: 7, attempt: 1 }).unwrap();
/// assert_eq!(json, r#"{"done":{"job":7,"attempt":1}}"#);
/// assert_eq!(
///     serde_json::from_str::<Input>(&json).unwrap(),
///     Input::Done { job: 7, attempt: 1 }
/// );
/// # }
/// # #[cfg(not(feature = "serde"))]
/// # fn main() {}
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Input {
    /// A job became ready. An id that is already waiting or running is ignored, and a job whose
    /// demand names a resource the policy does not declare is [rejected](Output::Rejected).
    Submit(JobSpec),
    /// An attempt finished: the job is complete, and every other live attempt of it is stopped
    /// ([`Output::Stop`]). Ignored unless `attempt` is live.
    Done {
        /// The job.
        job: JobId,
        /// The attempt that finished.
        attempt: Attempt,
    },
    /// An attempt failed. If no other attempt of the job is live, the job is retried (it keeps its
    /// place in the queue and its age, and softly avoids the workers it failed on) or, after
    /// [`RetryConfig::max_attempts`] rounds, given up ([`Output::GaveUp`]). Ignored unless
    /// `attempt` is live.
    Failed {
        /// The job.
        job: JobId,
        /// The attempt that failed.
        attempt: Attempt,
        /// How.
        kind: FailKind,
        /// The caller's description, kept in [`Tried`].
        why: String,
    },
    /// The job is no longer wanted: dropped if waiting, its live attempts stopped
    /// ([`Output::Stop`]) if running. Unknown ids are ignored.
    Cancel(JobId),
    /// A worker joined, or reported a heartbeat. Its live attempts are kept.
    ///
    /// A [`Scheduler`] panics on a state that names a resource [`Config::resources`] does not
    /// declare: a worker's resources are the caller's to match with the configuration, and one
    /// left out would be ignored.
    Worker(WorkerState),
    /// A worker left: each live attempt on it fails with [`FailKind::LinkDied`], as if reported
    /// by [`Input::Failed`].
    WorkerGone(WorkerId),
}

/// What a [`Policy`] asks its caller to do.
///
/// A flat policy such as [`Scheduler`] emits only [`Start`](Self::Start), [`Stop`](Self::Stop),
/// [`GaveUp`](Self::GaveUp) and [`Rejected`](Self::Rejected); the other variants come from the
/// [`DagScheduler`].
///
/// # Examples
///
/// A caller's dispatch over the outputs of one poll.
///
/// ```
/// use whelm::{
///     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
/// };
///
/// let mut p = Scheduler::new(Config::default());
/// p.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         capacity: Resources::new().with(SLOTS, 2),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// p.handle(
///     Input::Submit(JobSpec {
///         id: 1,
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// let mut sent = Vec::new();
/// for out in p.poll(Time::ORIGIN) {
///     match out {
///         Output::Start {
///             job,
///             attempt,
///             worker,
///         } => sent.push((job, attempt, worker)),
///         Output::Stop { .. } => {} // tell the worker to drop that attempt
///         Output::GaveUp(_) => {}   // report the job as failed
///         Output::Rejected { .. } => {} // report the job as malformed
///         Output::RunLocal { .. } | Output::Ready { .. } | Output::Passed { .. } => {}
///     }
/// }
/// assert_eq!(sent, [(1, 1, 1)]);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Output {
    /// Start this attempt of the job on the worker. Report its end with [`Input::Done`] or
    /// [`Input::Failed`].
    Start {
        /// The job.
        job: JobId,
        /// The attempt's number.
        attempt: Attempt,
        /// Where to run it.
        worker: WorkerId,
    },
    /// Stop this attempt: another attempt won, or the job was cancelled. Its result is not wanted,
    /// and the policy has already released its resources.
    Stop {
        /// The job.
        job: JobId,
        /// The attempt.
        attempt: Attempt,
        /// Where it runs.
        worker: WorkerId,
    },
    /// The job failed too often and is forgotten.
    GaveUp(GaveUp),
    /// The job was refused at submission and is forgotten.
    Rejected {
        /// The job.
        job: JobId,
        /// Why.
        reason: Rejection,
    },
    /// A [local](field@DagJob::local) job is ready: run it on the caller and report it with
    /// [`Input::Done`] and attempt 0.
    RunLocal {
        /// The job.
        job: JobId,
    },
    /// A job is ready and held (without [`DagConfig::auto_submit`]): submit it with
    /// [`DagScheduler::release`] when it is sendable.
    Ready {
        /// The job.
        job: JobId,
    },
    /// A passthrough job, or a unit other than a plain job, completed (with
    /// [`DagConfig::record_passthrough`]).
    Passed {
        /// The job.
        job: JobId,
    },
}

/// A placement policy, driven by messages.
///
/// Jobs are idempotent: running one twice is harmless and the first completion wins. That is the
/// caller's side of the contract; it lets the policy retry failed attempts, start a speculative
/// second attempt ([`Speculate`]) and ignore late messages about attempts it no longer tracks.
///
/// The caller feeds every event to [`handle`](Self::handle), then calls [`poll`](Self::poll) and
/// acts on each [`Output`]. Every method is deterministic: the same sequence of calls (with the
/// same `now`s) produces the same outputs, which is what [`log::replay`] relies on.
///
/// The policies compose: [`DagScheduler`], [`log::Logged`] and `Box<P>` are policies wrapping a
/// policy, so a driver written against `&mut dyn Policy` runs any of them.
///
/// # Examples
///
/// A wrapper that counts the starts of the policy it wraps.
///
/// ```
/// use whelm::{
///     Config, Explanation, Input, JobId, JobSpec, Output, Policy, PolicyStats, Resources, SLOTS,
///     Scheduler, Time, WorkerState,
/// };
///
/// /// Counts every start the inner policy emits.
/// struct Counting<P> {
///     inner: P,
///     starts: usize,
/// }
///
/// impl<P: Policy> Policy for Counting<P> {
///     fn handle(&mut self, input: Input, now: Time) {
///         self.inner.handle(input, now)
///     }
///
///     fn poll(&mut self, now: Time) -> Vec<Output> {
///         let out = self.inner.poll(now);
///         self.starts += out
///             .iter()
///             .filter(|o| matches!(o, Output::Start { .. }))
///             .count();
///         out
///     }
///
///     fn next_wakeup(&self) -> Option<Time> {
///         self.inner.next_wakeup()
///     }
///
///     fn explain(&self, job: JobId) -> Option<Explanation> {
///         self.inner.explain(job)
///     }
///
///     fn stats(&self) -> PolicyStats {
///         self.inner.stats()
///     }
/// }
///
/// let mut p = Counting {
///     inner: Scheduler::new(Config::default()),
///     starts: 0,
/// };
/// p.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         capacity: Resources::new().with(SLOTS, 4),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// for id in 1..=3 {
///     p.handle(
///         Input::Submit(JobSpec {
///             id,
///             ..Default::default()
///         }),
///         Time::ORIGIN,
///     );
/// }
/// p.poll(Time::ORIGIN);
/// assert_eq!(p.starts, 3);
/// ```
pub trait Policy {
    /// Take in one event at time `now` (non-decreasing across calls). Outputs it causes (stops,
    /// give-ups) are returned by the next `poll`.
    fn handle(&mut self, input: Input, now: Time);
    /// Place what can be placed now, and return every output since the last call, in order.
    fn poll(&mut self, now: Time) -> Vec<Output>;
    /// The next time `poll` should be called even if no event arrives: a hold lapses (e.g. a
    /// [`Defer`] wait for a faster worker), a job ages, or a job may reserve. `None` if nothing
    /// is timed. Callers with frequent events may ignore it at the cost of that much extra
    /// waiting.
    fn next_wakeup(&self) -> Option<Time>;
    /// Why a job is (not) running; its [`Display`](std::fmt::Display) form is one line for logs.
    /// `None` for unknown jobs.
    fn explain(&self, job: JobId) -> Option<Explanation>;
    /// Counters and current state.
    fn stats(&self) -> PolicyStats;
}

impl<P: Policy + ?Sized> Policy for Box<P> {
    /// Forwarded to the boxed policy.
    fn handle(&mut self, input: Input, now: Time) {
        (**self).handle(input, now)
    }

    /// Forwarded to the boxed policy.
    fn poll(&mut self, now: Time) -> Vec<Output> {
        (**self).poll(now)
    }

    /// Forwarded to the boxed policy.
    fn next_wakeup(&self) -> Option<Time> {
        (**self).next_wakeup()
    }

    /// Forwarded to the boxed policy.
    fn explain(&self, job: JobId) -> Option<Explanation> {
        (**self).explain(job)
    }

    /// Forwarded to the boxed policy.
    fn stats(&self) -> PolicyStats {
        (**self).stats()
    }
}
