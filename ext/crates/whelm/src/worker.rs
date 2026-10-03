//! Worker descriptions.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{Defer, Input, JobSpec, ScoreTerm, Selector, Speculate, Timing};
use crate::{Resources, SLOTS};

/// A worker identifier. The caller maps its own ids (e.g. `"host:port"`) to these.
pub type WorkerId = u64;

/// A worker's declared capacity and last reported usage.
///
/// The caller sends one with [`Input::Worker`] when the worker joins and again with every
/// heartbeat; the policy keeps the latest and its own bookkeeping of what it placed there.
///
/// # Examples
///
/// A GPU worker with 16 slots, 120 GB of host memory and 20 GB of device memory, reporting 30 GB
/// resident of which 12 GB is its runtime, and running at 1.4 times the reference speed.
///
/// ```
/// use whelm::{Resources, SLOTS, WorkerState};
///
/// let w = WorkerState {
///     id: 7,
///     class: "l40s".into(),
///     slots: 16,
///     budget: Resources::mem_gb(120.0).with_dev_gb(20.0),
///     reported_used: Resources::mem_gb(30.0),
///     reported_baseline: Resources::mem_gb(12.0),
///     speed: 1.4,
///     ..Default::default()
/// };
/// assert_eq!(w.capacity()[SLOTS], 16);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkerState {
    /// The worker's id.
    pub id: WorkerId,
    /// The worker's class (e.g. GPU type), used by [`Selector::Class`] and per-class reservations.
    pub class: String,
    /// Concurrent jobs at most; zero admits nothing.
    pub slots: usize,
    /// Memory capacity: host memory, and device memory (the pool jobs' device allocations come
    /// from). A zero component is unknown and not enforced. Its [`SLOTS`] component is ignored:
    /// [`capacity`](Self::capacity) takes it from `slots`.
    pub budget: Resources,
    /// What one job of this worker is expected to take at least, learned by the worker (e.g. the
    /// typical device launch request); zero components are unknown. Each job counts for at least
    /// this much against `budget` in every dimension, whatever its own [`JobSpec::demand`] says.
    #[cfg_attr(feature = "serde", serde(default))]
    pub per_task: Resources,
    /// Last reported resident usage (from a heartbeat; may lag by seconds).
    pub reported_used: Resources,
    /// The part of `reported_used` not attributable to jobs (caches, runtime).
    pub reported_baseline: Resources,
    /// How fast a job runs here, relative to a reference worker (1.0): a job with
    /// [`JobSpec::work`] `w` takes `w / speed` seconds. Used by [`ScoreTerm::Speed`], [`Defer`]
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
    /// A worker that runs one job at a time, of unknown memory, at the reference speed.
    ///
    /// A worker of zero slots admits nothing, so a default of zero would make a worker that
    /// silently never runs a job; with one, a worker that leaves `slots` out still runs its jobs,
    /// in turn.
    fn default() -> Self {
        Self {
            id: 0,
            class: String::new(),
            slots: 1,
            budget: Resources::ZERO,
            per_task: Resources::ZERO,
            reported_used: Resources::ZERO,
            reported_baseline: Resources::ZERO,
            speed: 1.0,
        }
    }
}

impl WorkerState {
    /// The capacity admission enforces: `budget` in the memory dimensions, `slots` in
    /// [`SLOTS`].
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{MEM, Resources, SLOTS, WorkerState};
    ///
    /// let w = WorkerState {
    ///     slots: 8,
    ///     budget: Resources::mem(100),
    ///     ..Default::default()
    /// };
    /// let cap = w.capacity();
    /// assert_eq!((cap[MEM], cap[SLOTS]), (100, 8));
    /// ```
    pub fn capacity(&self) -> Resources {
        let mut c = self.budget;
        c[SLOTS] = self.slots as u64;
        c
    }
}
