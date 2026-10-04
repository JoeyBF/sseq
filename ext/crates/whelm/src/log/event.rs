//! The events of a log.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use super::{Logged, replay};
use crate::{Input, JobId, Output, Time};

/// What a job is, for the simulator (optional; Nassau's vocabulary). Without it a logged job
/// replays as a signature task of its group.
///
/// The policy never reads it: [`Logged::annotate`] attaches it to the job's submission in the log.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TaskInfo {
    /// `"zero"` or `"sig"`.
    pub kind: String,
    /// The bidegree `(n, s)`.
    pub bidegree: (i64, i64),
    /// Size covariate of the service-model fit (target dimension).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub target: Option<f64>,
    /// Size covariate of the service-model fit (next dimension).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub next: Option<f64>,
    /// Jobs that had to complete first.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub deps: Vec<JobId>,
    /// Bidegrees `(n, s)` that had to complete first (zero tasks).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub after_groups: Vec<(i64, i64)>,
    /// The signature's Milnor exponents.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub sig: Vec<u32>,
}

/// One logged event. Times are on the policy's clock ([`Time`]).
///
/// The [`Input`](Event::Input) and [`Poll`](Event::Poll) records are the whole run: feeding them
/// back with [`replay`] reproduces every output, reservations included. [`Sample`](Event::Sample)
/// summarises a worker's state for the trace reader, with its id written as a string (the trace
/// format names workers).
///
/// Events can be written by hand, e.g. to script a run for [`replay`]:
///
/// ```
/// use whelm::{Input, JobSpec, MEMORY, Resources, Time, gb, log::Event};
///
/// let submit = Event::Input {
///     t: Time::ORIGIN,
///     input: Input::Submit(JobSpec {
///         id: 1,
///         demand: Resources::new().with(MEMORY, gb(1.0)),
///         ..Default::default()
///     }),
///     info: None,
/// };
/// let poll = Event::Poll {
///     t: Time::ORIGIN,
///     out: Vec::new(),
/// };
/// # let _ = (submit, poll);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "lowercase"))]
pub enum Event {
    /// An input, exactly as the policy handled it.
    Input {
        /// When.
        t: Time,
        /// The input.
        input: Input,
        /// What a submitted job is ([`Logged::annotate`]).
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        info: Option<Box<TaskInfo>>,
    },
    /// A poll, and what it returned.
    Poll {
        /// When.
        t: Time,
        /// Its outputs, in order.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Vec::is_empty")
        )]
        out: Vec<Output>,
    },
    /// A heartbeat (rate-limited): reported resident memory and the policy's own bookkeeping.
    Sample {
        /// When.
        t: Time,
        /// The worker.
        worker: String,
        /// Reported resident memory, GB.
        rss_gb: f64,
        /// Reported baseline, GB.
        baseline_gb: f64,
        /// Sum of the demands placed there, GB.
        reserved_gb: f64,
        /// Live attempts there.
        running: usize,
        /// The worker's learned device memory per job, GB (0: unknown).
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_zero"))]
        dev_per_task_gb: f64,
    },
}

/// Whether a logged quantity is zero (left out of the line).
#[cfg(feature = "serde")]
fn is_zero(x: &f64) -> bool {
    *x == 0.0
}
