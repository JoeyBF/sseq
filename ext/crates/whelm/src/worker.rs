//! Worker descriptions.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Resources;
#[cfg(doc)]
use crate::{Config, Resource, SLOTS};
#[cfg(doc)]
use crate::{Defer, Input, JobSpec, ScoreTerm, Selector, Speculate, Timing};

/// A worker identifier. The caller maps its own ids (e.g. `"host:port"`) to these.
pub type WorkerId = u64;

/// A worker's declared capacity and last reported usage.
///
/// The caller sends one with [`Input::Worker`] when the worker joins and again with every
/// heartbeat; the policy keeps the latest and its own bookkeeping of what it placed there.
///
/// # Examples
///
/// A GPU worker with 120 GB of host memory, 20 GB of device memory and 16 slots, reporting 30 GB
/// resident of which 12 GB is its runtime, and running at 1.4 times the reference speed.
///
/// ```
/// use whelm::{DEVICE_MEMORY, MEMORY, Resources, SLOTS, WorkerState, gb};
///
/// let w = WorkerState {
///     id: 7,
///     class: "l40s".into(),
///     capacity: Resources::new()
///         .with(MEMORY, gb(120.0))
///         .with(DEVICE_MEMORY, gb(20.0))
///         .with(SLOTS, 16),
///     reported_used: Resources::new().with(MEMORY, gb(30.0)),
///     reported_baseline: Resources::new().with(MEMORY, gb(12.0)),
///     speed: 1.4,
///     ..Default::default()
/// };
/// assert_eq!(w.capacity.get(SLOTS), 16);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkerState {
    /// The worker's id.
    pub id: WorkerId,
    /// The worker's class (e.g. GPU type), used by [`Selector::Class`] and per-class reservations.
    pub class: String,
    /// What the worker has of each [declared resource](Config::resources). A zero amount of a
    /// [hard](Resource::hard) resource is none of it, and of a soft one, an unknown amount that is
    /// not enforced. Under the default declaration, the worker runs nothing without [`SLOTS`].
    pub capacity: Resources,
    /// What one job of this worker is expected to take at least, learned by the worker (e.g. the
    /// typical device launch request); zero amounts are unknown. Each job counts for at least
    /// this much against `capacity` in every resource, whatever its own [`JobSpec::demand`] says.
    #[cfg_attr(feature = "serde", serde(default))]
    pub per_task: Resources,
    /// Last reported resident usage (from a heartbeat; may lag by seconds).
    pub reported_used: Resources,
    /// The part of `reported_used` not attributable to jobs (caches, runtime).
    pub reported_baseline: Resources,
    /// How fast a job runs here, relative to a reference worker (1.0): a job with
    /// [`JobSpec::work`] `w` takes `w / speed`. Used by [`ScoreTerm::Speed`], [`Defer`]
    /// and [`Speculate`] as the [`Timing`] says: ignored by identical machines, a prior for
    /// learned ones.
    #[cfg_attr(feature = "serde", serde(default = "unit"))]
    pub speed: f64,
}

/// The default [`WorkerState::speed`] and [`JobSpec::weight`].
#[cfg(feature = "serde")]
pub(crate) fn unit() -> f64 {
    1.0
}

impl Default for WorkerState {
    /// A worker of zero capacity, at the reference speed.
    ///
    /// Under the default declaration it runs nothing until its capacity has slots: a worker
    /// writes its capacity whole, slots included. A job no worker has slots for is
    /// [explained](crate::Policy::explain) as slots full there.
    fn default() -> Self {
        Self {
            id: 0,
            class: String::new(),
            capacity: Resources::new(),
            per_task: Resources::new(),
            reported_used: Resources::new(),
            reported_baseline: Resources::new(),
            speed: 1.0,
        }
    }
}
