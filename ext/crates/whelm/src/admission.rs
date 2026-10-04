//! The admission rule: whether a worker accepts a job right now.
//!
//! The [`Scheduler`](crate::Scheduler) places a job only where its [`Admission`] rule admits it,
//! so the rule alone enforces capacity, slots included. The rule sees a worker as a
//! [`WorkerView`]: the resources the configuration declares, the worker's last reported
//! [`WorkerState`], and the demands the scheduler has placed there itself. [`ProductionAdmission`]
//! is the rule [`Scheduler::new`] uses: one inequality per declared resource, with an escape hatch
//! for a job alone on a worker in the soft ones. Any other rule goes to
//! [`Scheduler::with_admission`].
//!
//! A worker with 100 bytes of memory and two slots, running one 60-byte job, takes a 40-byte job
//! beside it but not a 41-byte one. The scheduler asks with demands that take one slot each
//! ([`Resources::with_slots`]), as the `job` closure builds them:
//!
//! ```
//! use whelm::{Admission, Config, ProductionAdmission, Resources, WorkerState, WorkerView};
//!
//! let job = |bytes| Resources::mem(bytes).with_slots(1);
//! let state = WorkerState {
//!     id: 1,
//!     class: "cpu".into(),
//!     capacity: Resources::mem(100).with_slots(2),
//!     ..Default::default()
//! };
//! let view = WorkerView {
//!     resources: &Config::default().resources,
//!     state: &state,
//!     placed: &job(60),
//!     running: 1,
//! };
//! assert!(ProductionAdmission.admits(&job(40), &view));
//! assert!(!ProductionAdmission.admits(&job(41), &view));
//! ```
//!
//! [`Scheduler::new`]: crate::Scheduler::new
//! [`Scheduler::with_admission`]: crate::Scheduler::with_admission

#[cfg(doc)]
use crate::{Config, JobSpec};
use crate::{Resource, ResourceId, Resources, WorkerState};

/// A worker as an [`Admission`] rule sees it: its state plus the library's own bookkeeping.
///
/// The methods are the vocabulary [`ProductionAdmission`] is written in, and a custom rule may
/// use them too; [`free_share`](Self::free_share) is also what
/// [`ScoreTerm::Tightest`](crate::ScoreTerm::Tightest) and reservations rank workers by.
///
/// A worker with 100 bytes of memory, unknown device memory and four slots reports 30 bytes in
/// use, 10 of them its own baseline, and runs one job placed with 50 bytes. The examples on the
/// methods continue from this one.
///
/// ```
/// use whelm::{Config, ResourceId, Resources, WorkerState, WorkerView};
///
/// // A demand as the scheduler sends it: one slot per job.
/// let job = |bytes| Resources::mem(bytes).with_slots(1);
/// let config = Config::default();
/// let state = WorkerState {
///     id: 1,
///     class: "cpu".into(),
///     capacity: Resources::mem(100).with_slots(4),
///     reported_used: Resources::mem(30),
///     reported_baseline: Resources::mem(10),
///     ..Default::default()
/// };
/// let view = WorkerView {
///     resources: &config.resources,
///     state: &state,
///     placed: &job(50),
///     running: 1,
/// };
/// // Memory and slots are enforced; device memory, with a zero capacity, is not.
/// assert!(view.enforced(ResourceId::MEM) && view.enforced(ResourceId::SLOTS));
/// assert!(!view.enforced(ResourceId::DEV));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct WorkerView<'a> {
    /// The declared resources ([`Config::resources`]): the dimensions the rule looks at.
    pub resources: &'a [Resource],
    /// The worker's last reported state.
    pub state: &'a WorkerState,
    /// Sum of the demands of the attempts placed on the worker and not yet ended, with their
    /// [default demands](Resource::default_demand) filled in.
    pub placed: &'a Resources,
    /// Attempts placed on the worker and not yet ended.
    pub running: usize,
}

impl WorkerView<'_> {
    /// The declared resources, in order.
    pub fn ids(&self) -> impl Iterator<Item = ResourceId> + use<> {
        (0..self.resources.len()).map(ResourceId)
    }

    /// The declaration of resource `d`.
    pub fn resource(&self, d: ResourceId) -> &Resource {
        &self.resources[d.0]
    }

    /// Whether resource `d` is enforced: always if it is [hard](Resource::hard), else if its
    /// capacity is known (nonzero).
    pub fn enforced(&self, d: ResourceId) -> bool {
        self.resource(d).hard || self.state.capacity[d] > 0
    }

    /// What a job of demand `demand` counts for here: at least the worker's
    /// [`per_task`](WorkerState::per_task) in every resource.
    ///
    /// On a worker whose jobs each take at least 30 bytes of device memory, a job declaring 5
    /// counts for 30:
    ///
    /// ```
    /// use whelm::{Config, Resources, WorkerState, WorkerView};
    ///
    /// let state = WorkerState {
    ///     id: 1,
    ///     class: "gpu".into(),
    ///     capacity: Resources::mem(100).with_dev(100).with_slots(4),
    ///     per_task: Resources::ZERO.with_dev(30),
    ///     ..Default::default()
    /// };
    /// let view = WorkerView {
    ///     resources: &Config::default().resources,
    ///     state: &state,
    ///     placed: &Resources::ZERO,
    ///     running: 0,
    /// };
    /// let demand = Resources::mem(5).with_dev(5);
    /// assert_eq!(view.charge(&demand), Resources::mem(5).with_dev(30));
    /// ```
    pub fn charge(&self, demand: &Resources) -> Resources {
        demand.clone().max(self.state.per_task.clone())
    }

    /// [`charge`](Self::charge) in resource `d` alone.
    fn charge_in(&self, demand: &Resources, d: ResourceId) -> u64 {
        demand[d].max(self.state.per_task[d])
    }

    /// Usage as the production rule counts it, per declared resource: `max(reported_used,
    /// reported_baseline + max(placed, running * per_task))`.
    ///
    /// The reported figure lags (heartbeats); the placed sum is exact but only an estimate of what
    /// the jobs use. Taking the maximum is conservative in both directions.
    ///
    /// For the worker of the [type-level example](WorkerView), the baseline plus the placed 50
    /// bytes outweighs the 30 reported:
    ///
    /// ```
    /// # use whelm::{Config, Resources, WorkerState, WorkerView};
    /// # let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// # let config = Config::default();
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     capacity: Resources::mem(100).with_slots(4),
    /// #     reported_used: Resources::mem(30),
    /// #     reported_baseline: Resources::mem(10),
    /// #     ..Default::default()
    /// # };
    /// # let view = WorkerView {
    /// #     resources: &config.resources,
    /// #     state: &state,
    /// #     placed: &job(50),
    /// #     running: 1,
    /// # };
    /// assert_eq!(view.used(), job(60)); // max(30, 10 + 50) bytes, and one slot
    /// ```
    pub fn used(&self) -> Resources {
        Resources::of(self.ids().map(|d| (d, self.used_in(d))))
    }

    /// [`used`](Self::used) in resource `d` alone.
    fn used_in(&self, d: ResourceId) -> u64 {
        let s = self.state;
        let floor = s.per_task[d].saturating_mul(self.running as u64);
        (s.reported_used[d]).max(s.reported_baseline[d].saturating_add(self.placed[d].max(floor)))
    }

    /// Headroom per declared resource, `capacity - used`; negative when over-committed, `None`
    /// where the resource is not [`enforced`](Self::enforced).
    ///
    /// For the worker of the [type-level example](WorkerView):
    ///
    /// ```
    /// # use whelm::{Config, Resources, WorkerState, WorkerView};
    /// # let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// # let config = Config::default();
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     capacity: Resources::mem(100).with_slots(4),
    /// #     reported_used: Resources::mem(30),
    /// #     reported_baseline: Resources::mem(10),
    /// #     ..Default::default()
    /// # };
    /// # let view = WorkerView {
    /// #     resources: &config.resources,
    /// #     state: &state,
    /// #     placed: &job(50),
    /// #     running: 1,
    /// # };
    /// // 100 - 60 bytes of memory, device memory unknown, 4 - 1 slots.
    /// assert_eq!(view.headroom(), [Some(40), None, Some(3)]);
    /// ```
    pub fn headroom(&self) -> Vec<Option<i64>> {
        self.ids()
            .map(|d| {
                self.enforced(d).then(|| {
                    (self.state.capacity[d] as i128 - self.used_in(d) as i128)
                        .clamp(i64::MIN as i128, i64::MAX as i128) as i64
                })
            })
            .collect()
    }

    /// The enforced resources in which a job of demand `demand` does not fit beside the jobs
    /// already here: `used + charge > capacity`. It ignores the escape hatch, so it names what a
    /// refusal is about rather than deciding one.
    ///
    /// For the worker of the [type-level example](WorkerView):
    ///
    /// ```
    /// # use whelm::{Config, Resources, WorkerState, WorkerView};
    /// # let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// # let config = Config::default();
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     capacity: Resources::mem(100).with_slots(4),
    /// #     reported_used: Resources::mem(30),
    /// #     reported_baseline: Resources::mem(10),
    /// #     ..Default::default()
    /// # };
    /// # let view = WorkerView {
    /// #     resources: &config.resources,
    /// #     state: &state,
    /// #     placed: &job(50),
    /// #     running: 1,
    /// # };
    /// use whelm::ResourceId;
    ///
    /// assert_eq!(view.short(&job(40)).count(), 0);
    /// assert_eq!(view.short(&job(41)).collect::<Vec<_>>(), [ResourceId::MEM]);
    /// ```
    pub fn short<'b>(&'b self, demand: &'b Resources) -> impl Iterator<Item = ResourceId> + 'b {
        self.ids().filter(move |&d| {
            self.enforced(d)
                && self.used_in(d).saturating_add(self.charge_in(demand, d))
                    > self.state.capacity[d]
        })
    }

    /// The fraction of capacity left after placing `demand`, in the bottleneck resource: `min_d
    /// (capacity - used - charge) / capacity` over the enforced soft resources (one minus the
    /// dominant share, as in dominant-resource fairness), infinite when none is enforced.
    ///
    /// This is the one scalar workers are compared by: smallest for the tightest fit, largest
    /// (with a zero demand) for the most headroom. Hard resources are counts rather than capacity
    /// to pack into, and are left to [`ScoreTerm::Load`](crate::ScoreTerm::Load).
    ///
    /// For the worker of the [type-level example](WorkerView), 60 of 100 bytes in use:
    ///
    /// ```
    /// # use whelm::{Config, Resources, WorkerState, WorkerView};
    /// # let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// # let config = Config::default();
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     capacity: Resources::mem(100).with_slots(4),
    /// #     reported_used: Resources::mem(30),
    /// #     reported_baseline: Resources::mem(10),
    /// #     ..Default::default()
    /// # };
    /// # let view = WorkerView {
    /// #     resources: &config.resources,
    /// #     state: &state,
    /// #     placed: &job(50),
    /// #     running: 1,
    /// # };
    /// assert_eq!(view.free_share(&Resources::ZERO), 0.4);
    /// assert_eq!(view.free_share(&job(20)), 0.2);
    /// // A worker with no memory capacity enforces no soft resource.
    /// let unknown = WorkerState {
    ///     id: 2,
    ///     class: "cpu".into(),
    ///     capacity: Resources::ZERO.with_slots(4),
    ///     ..Default::default()
    /// };
    /// let view = WorkerView {
    ///     state: &unknown,
    ///     placed: &Resources::ZERO,
    ///     running: 0,
    ///     ..view
    /// };
    /// assert_eq!(view.free_share(&job(20)), f64::INFINITY);
    /// ```
    pub fn free_share(&self, demand: &Resources) -> f64 {
        self.ids()
            .filter(|&d| !self.resource(d).hard && self.enforced(d))
            .map(|d| {
                let cap = self.state.capacity[d] as f64;
                (cap - self.used_in(d) as f64 - self.charge_in(demand, d) as f64) / cap
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Whether a worker admits a job.
///
/// # Contract
///
/// Implementations must be **monotone in load**: if a job is refused by a worker, it stays refused
/// after more jobs are placed on that worker (with no completion or heartbeat in between). The
/// scheduler relies on this to guarantee the priority invariant within one
/// [`poll`](crate::Policy::poll).
///
/// The rule alone enforces capacity, slots included: the scheduler places a job wherever the rule
/// admits it. It sees demands as the scheduler holds them: truncated to the declared resources,
/// with [default demands](Resource::default_demand) filled in.
///
/// # Example
///
/// A rule that counts slots and nothing else: two 80-byte jobs share a 100-byte worker, where
/// [`ProductionAdmission`] would run them one at a time.
///
/// ```
/// use whelm::{
///     Admission, Config, Input, JobSpec, Policy, ResourceId, Resources, Scheduler, Time,
///     WorkerState, WorkerView,
/// };
///
/// /// Admits while a slot is free, whatever the memory.
/// struct SlotsOnly;
///
/// impl Admission for SlotsOnly {
///     fn admits(&self, _demand: &Resources, w: &WorkerView) -> bool {
///         (w.running as u64) < w.state.capacity[ResourceId::SLOTS]
///     }
/// }
///
/// let mut s = Scheduler::with_admission(Config::fifo(), SlotsOnly);
/// s.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         capacity: Resources::mem(100).with_slots(2),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// for id in 0..3 {
///     s.handle(
///         Input::Submit(JobSpec {
///             id,
///             demand: Resources::mem(80),
///             ..Default::default()
///         }),
///         Time::ORIGIN,
///     );
/// }
/// assert_eq!(s.poll(Time::ORIGIN).len(), 2);
/// assert_eq!(
///     s.explain(2).unwrap().waiting().unwrap().workers,
///     [(
///         1,
///         whelm::Verdict::Full {
///             dims: vec![ResourceId::SLOTS]
///         }
///     )]
/// );
/// ```
pub trait Admission {
    /// Whether `w` admits a job with demand `demand`.
    fn admits(&self, demand: &Resources, w: &WorkerView) -> bool;

    /// An upper bound on admitted demands, used only to skip hopeless checks quickly: if
    /// `admits(d, w)` then `d.fits_within(bound(w))`. `None` means `w` admits nothing. The default
    /// is no bound in any declared resource. [`ProductionAdmission::bound`] has an example.
    fn bound(&self, w: &WorkerView) -> Option<Resources> {
        Some(Resources::repeat(u64::MAX, w.resources.len()))
    }
}

/// The production admission rule, one inequality per enforced declared resource `d`:
///
/// ```text
/// admit(job on w) iff for every enforced d:
///                       max(reported_used[d],
///                           reported_baseline[d] + max(placed[d], running * per_task[d]))
///                         + max(demand[d], per_task[d]) <= capacity[d]
///                     or (d is soft and running == 0)      // escape hatch: a job alone goes
/// ```
///
/// [Hard](Resource::hard) resources (slots) are always enforced and have no escape hatch; a soft
/// one is enforced where its capacity is nonzero. With a device `per_task` alone (jobs without
/// device demands) the device inequality is the per-worker count `(running + 1) * per_task <=
/// cap`; with per-job device demands it is their sum.
///
/// `reported_baseline` must exclude the running jobs (a worker's `baseline_excl`: its rolling RSS
/// floor minus their estimates). A floor that contains them counts them twice, once in it and
/// once in `placed`, and keeps a busy worker a few GB short of its capacity.
///
/// # Examples
///
/// The escape hatch: an empty worker takes a job larger than its memory capacity, a busy one does
/// not. Slots have no escape hatch, so a worker without slots takes nothing.
///
/// ```
/// use whelm::{Admission, Config, ProductionAdmission, Resources, WorkerState, WorkerView};
///
/// let config = Config::default();
/// let job = |bytes| Resources::mem(bytes).with_slots(1);
/// let state = WorkerState {
///     id: 1,
///     class: "cpu".into(),
///     capacity: Resources::mem(100).with_slots(2),
///     ..Default::default()
/// };
/// // Whether `state` admits a job of `bytes` beside `running` jobs that took `placed`.
/// let admits = |state: &WorkerState, bytes, placed: &Resources, running| {
///     let view = WorkerView {
///         resources: &config.resources,
///         state,
///         placed,
///         running,
///     };
///     ProductionAdmission.admits(&job(bytes), &view)
/// };
/// assert!(admits(&state, 1000, &Resources::ZERO, 0));
/// assert!(!admits(&state, 1000, &job(1), 1));
///
/// let no_slots = WorkerState {
///     id: 2,
///     class: "cpu".into(),
///     capacity: Resources::mem(100),
///     ..Default::default()
/// };
/// assert!(!admits(&no_slots, 1, &Resources::ZERO, 0));
/// ```
///
/// A device `per_task` alone caps the jobs per worker: with 100 bytes of device memory and 30 per
/// job, three jobs fit and a fourth does not, whatever their own (zero) device demands.
///
/// ```
/// # use whelm::{Admission, Config, ProductionAdmission, Resources, WorkerState, WorkerView};
/// let config = Config::default();
/// let state = WorkerState {
///     id: 1,
///     class: "gpu".into(),
///     capacity: Resources::ZERO.with_dev(100).with_slots(8),
///     per_task: Resources::ZERO.with_dev(30),
///     ..Default::default()
/// };
/// let admits_with = |running| {
///     let view = WorkerView {
///         resources: &config.resources,
///         state: &state,
///         placed: &Resources::ZERO.with_slots(running),
///         running: running as usize,
///     };
///     ProductionAdmission.admits(&Resources::ZERO.with_slots(1), &view)
/// };
/// assert!(admits_with(2));
/// assert!(!admits_with(3));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct ProductionAdmission;

impl Admission for ProductionAdmission {
    /// The production rule itself.
    fn admits(&self, demand: &Resources, w: &WorkerView) -> bool {
        let empty = w.running == 0;
        w.short(demand).all(|d| !w.resource(d).hard && empty)
    }

    /// The headroom left under the production rule (unbounded in the resources not enforced, and
    /// in the soft ones when the worker is empty).
    ///
    /// ```
    /// use whelm::{
    ///     Admission, Config, ProductionAdmission, ResourceId, Resources, WorkerState, WorkerView,
    /// };
    ///
    /// let config = Config::default();
    /// let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// let state = WorkerState {
    ///     id: 1,
    ///     class: "cpu".into(),
    ///     capacity: Resources::mem(100).with_slots(2),
    ///     ..Default::default()
    /// };
    /// let bound = |placed: &Resources, running| {
    ///     let view = WorkerView {
    ///         resources: &config.resources,
    ///         state: &state,
    ///         placed,
    ///         running,
    ///     };
    ///     ProductionAdmission.bound(&view)
    /// };
    /// let (mem, slots) = (ResourceId::MEM, ResourceId::SLOTS);
    /// // Empty: memory unbounded (the escape hatch), two slots.
    /// let empty = bound(&Resources::ZERO, 0).unwrap();
    /// assert_eq!((empty[mem], empty[slots]), (u64::MAX, 2));
    /// // One 60-byte job: 40 bytes and one slot left.
    /// let busy = bound(&job(60), 1).unwrap();
    /// assert_eq!((busy[mem], busy[slots]), (40, 1));
    /// // Both slots taken: nothing is admitted.
    /// assert_eq!(bound(&Resources::mem(60).with_slots(2), 2), None);
    /// ```
    fn bound(&self, w: &WorkerView) -> Option<Resources> {
        let empty = w.running == 0;
        let mut bound = Resources::repeat(u64::MAX, w.resources.len());
        for d in w.ids() {
            let r = w.resource(d);
            if !w.enforced(d) || (!r.hard && empty) {
                continue;
            }
            let room = w.state.capacity[d].checked_sub(w.used_in(d))?;
            // Every job counts for at least `per_task`, and for one at least where a default
            // demand fills in its zeros, so less room than that admits nothing.
            let least = w.state.per_task[d].max((r.default_demand > 0) as u64);
            if room < least {
                return None;
            }
            bound[d] = room;
        }
        Some(bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, ResourceUnit};

    const MEM: ResourceId = ResourceId::MEM;
    const DEV: ResourceId = ResourceId::DEV;
    const SLOTS: ResourceId = ResourceId::SLOTS;

    /// The default declaration.
    fn standard() -> Vec<Resource> {
        Config::default().resources
    }

    /// A worker state with the given capacity and reported usage.
    fn worker(slots: u64, capacity: u64, used: u64, baseline: u64) -> WorkerState {
        WorkerState {
            id: 1,
            capacity: Resources::mem(capacity).with_slots(slots),
            reported_used: Resources::mem(used),
            reported_baseline: Resources::mem(baseline),
            ..Default::default()
        }
    }

    /// `running` attempts' worth of slots on top of `r`.
    fn with_running(r: Resources, running: u64) -> Resources {
        r.with_slots(running)
    }

    /// The rule, its escape hatch and its bound, at the boundaries.
    #[test]
    fn production_rule() {
        let a = ProductionAdmission;
        let decl = standard();
        let s = worker(2, 100, 30, 10);
        let placed = |running, placed| with_running(Resources::mem(placed), running);
        let (empty, half, little, full) = (placed(0, 0), placed(1, 50), placed(1, 5), placed(2, 0));
        let view = |placed| WorkerView {
            resources: &decl,
            state: &s,
            placed,
            running: placed[SLOTS] as usize,
        };
        let job = |bytes| with_running(Resources::mem(bytes), 1);
        // Escape hatch: alone, anything goes, even beyond the capacity.
        assert!(a.admits(&job(1000), &view(&empty)));
        // max(30, 10 + 50) + 40 = 100 <= 100.
        assert!(a.admits(&job(40), &view(&half)));
        assert!(!a.admits(&job(41), &view(&half)));
        // Reported usage dominates: max(30, 10 + 5) + 70 = 100.
        assert!(a.admits(&job(70), &view(&little)));
        assert!(!a.admits(&job(71), &view(&little)));
        // Slots full: no escape hatch for a hard resource.
        assert!(!a.admits(&job(0), &view(&full)));
        assert_eq!(a.bound(&view(&full)), None);
        assert_eq!(
            a.bound(&view(&half)),
            Some(with_running(Resources::mem(40).with_dev(u64::MAX), 1))
        );
        assert_eq!(view(&half).headroom(), [Some(40), None, Some(1)]);
    }

    /// `baseline_excl` (the rolling RSS floor minus the estimates running) as `reported_baseline`
    /// gives `max(rss, baseline_excl + Σ placed) + demand <= capacity`: the floor does not count
    /// the running jobs twice.
    #[test]
    fn baseline_excl_removes_the_double_count() {
        let decl = standard();
        // RSS 60 with 40 of estimates running; the floor (50) contains those jobs.
        let (rss, floor, placed, capacity) = (60, 50, 40, 100);
        let placed = with_running(Resources::mem(placed), 4);
        let view = |s| WorkerView {
            resources: &decl,
            state: s,
            placed: &placed,
            running: 4,
        };
        let job = |bytes| with_running(Resources::mem(bytes), 1);
        let with_floor = worker(16, capacity, rss, floor);
        let with_excl = worker(16, capacity, rss, floor - 40);
        // Floor: max(60, 50 + 40) + 15 = 105 > 100. Excl: max(60, 10 + 40) + 40 = 100.
        assert!(!ProductionAdmission.admits(&job(15), &view(&with_floor)));
        assert!(ProductionAdmission.admits(&job(40), &view(&with_excl)));
        assert!(!ProductionAdmission.admits(&job(41), &view(&with_excl)));
        // RSS stays the safety term: estimates that undercount cannot admit past it.
        let undercount = worker(16, capacity, 95, 0);
        assert!(!ProductionAdmission.admits(&job(6), &view(&undercount)));
    }

    /// Every resource follows the same inequality, with `per_task` as a per-job floor; a zero
    /// capacity component of a soft resource is not enforced, host memory included.
    #[test]
    fn one_rule_per_resource() {
        let a = ProductionAdmission;
        let decl = standard();
        let s = WorkerState {
            id: 1,
            capacity: Resources::mem(100).with_dev(100).with_slots(8),
            per_task: Resources::mem(20).with_dev(30),
            ..Default::default()
        };
        let view = |placed| WorkerView {
            resources: &decl,
            state: &s,
            placed,
            running: placed[SLOTS] as usize,
        };
        // Device: max(10, 2 * 30) + max(5, 30) = 90 <= 100, and 60 + 41 > 100.
        let placed = with_running(Resources::mem(40).with_dev(10), 2);
        let v = view(&placed);
        assert!(a.admits(&Resources::mem(10).with_dev(5).with_slots(1), &v));
        assert!(!a.admits(&Resources::mem(10).with_dev(41).with_slots(1), &v));
        assert_eq!(v.short(&Resources::mem(70)).collect::<Vec<_>>(), [MEM]);
        let placed = with_running(Resources::mem(40), 3);
        let v = view(&placed);
        assert_eq!(v.short(&Resources::ZERO).collect::<Vec<_>>(), [DEV]);
        // Host: the floor counts 3 * 20 = 60 against the 40 placed.
        assert_eq!(v.headroom(), [Some(40), Some(10), Some(5)]);
        assert_eq!(a.bound(&v), None);
        // A zero memory capacity leaves its resource unenforced; slots stay enforced.
        let free = WorkerState {
            id: 2,
            capacity: Resources::ZERO.with_slots(8),
            ..Default::default()
        };
        let placed = with_running(Resources::mem(1 << 40), 7);
        let v = WorkerView {
            state: &free,
            placed: &placed,
            running: 7,
            ..view(&placed)
        };
        let huge = Resources::repeat(u64::MAX, 3).with_slots(1);
        assert!(a.admits(&huge, &v));
        assert!(!a.admits(&Resources::repeat(u64::MAX, 3), &v));
        assert_eq!(v.headroom(), [None, None, Some(1)]);
        assert_eq!(v.free_share(&huge), f64::INFINITY);
    }

    /// The scalar comparison is the bottleneck soft resource's free fraction; slots do not
    /// enter it.
    #[test]
    fn free_share_is_the_bottleneck() {
        let decl = standard();
        let s = WorkerState {
            id: 1,
            capacity: Resources::mem(100).with_dev(10).with_slots(2),
            ..Default::default()
        };
        let placed = with_running(Resources::mem(50).with_dev(2), 1);
        let v = WorkerView {
            resources: &decl,
            state: &s,
            placed: &placed,
            running: 1,
        };
        // Host 1 - 60/100 = 0.4, device 1 - 4/10 = 0.6; slots would be 1 - 2/2 = 0.
        let share = v.free_share(&with_running(Resources::mem(10).with_dev(2), 1));
        assert!((share - 0.4).abs() < 1e-12);
    }

    /// Slots are hard: without a free one, not even the escape hatch admits.
    #[test]
    fn zero_slots_admit_nothing() {
        let decl = standard();
        let s = worker(0, 100, 0, 0);
        let v = WorkerView {
            resources: &decl,
            state: &s,
            placed: &Resources::ZERO,
            running: 0,
        };
        let one_slot = with_running(Resources::ZERO, 1);
        assert!(!ProductionAdmission.admits(&one_slot, &v));
        assert_eq!(ProductionAdmission.bound(&v), None);
    }

    /// A declaration of its own: the rule reads hardness and default demands from it, and
    /// ignores components beyond it.
    #[test]
    fn custom_declaration() {
        let gpus = ResourceId(0);
        let scratch = ResourceId(1);
        let decl = [
            Resource {
                name: "gpus".into(),
                hard: true,
                ..Default::default()
            },
            Resource {
                name: "scratch".into(),
                default_demand: 5,
                unit: ResourceUnit::Bytes,
                ..Default::default()
            },
        ];
        let s = WorkerState {
            id: 1,
            capacity: Resources::of([(gpus, 2), (scratch, 10)]),
            ..Default::default()
        };
        let (zero, one) = (Resources::ZERO, Resources::of([(scratch, 5)]));
        let busy = Resources::of([(gpus, 2), (scratch, 5)]);
        let (six, ten) = (
            Resources::of([(scratch, 6)]),
            Resources::of([(scratch, 10)]),
        );
        let view = |placed, running| WorkerView {
            resources: &decl,
            state: &s,
            placed,
            running,
        };
        let a = ProductionAdmission;
        let two_gpus = Resources::of([(gpus, 2), (scratch, 5)]);
        // Hard: three GPUs never fit, not even alone.
        assert!(!a.admits(&two_gpus.clone().with(gpus, 3), &view(&zero, 0)));
        assert!(a.admits(&two_gpus, &view(&zero, 0)));
        // Soft: alone, any amount of scratch goes; beside another job, 5 + 6 > 10 does not.
        assert!(a.admits(&Resources::of([(scratch, 50)]), &view(&zero, 0)));
        assert!(!a.admits(&six, &view(&one, 1)));
        // A job without GPUs fits beside two GPUs' worth; the slots component is no resource
        // here, so a huge one changes nothing.
        assert!(a.admits(&one.clone().with_slots(99), &view(&busy, 1)));
        assert_eq!(view(&busy, 1).short(&two_gpus).collect::<Vec<_>>(), [gpus]);
        // The bound: no GPU left, 5 of scratch. With a default demand every job takes some
        // scratch, so a worker with none left admits nothing.
        let b = a.bound(&view(&busy, 1)).unwrap();
        assert_eq!((b[gpus], b[scratch]), (0, 5));
        assert_eq!(a.bound(&view(&ten, 1)), None);
    }
}
