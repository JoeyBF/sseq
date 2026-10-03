//! The admission rule: whether a worker accepts a job right now.
//!
//! The [`Scheduler`](crate::Scheduler) places a job only where its [`Admission`] rule admits it,
//! so the rule alone enforces capacity, slots included. The rule sees a worker as a
//! [`WorkerView`]: the worker's last reported [`WorkerState`] plus the demands the scheduler has
//! placed there itself. [`ProductionAdmission`] is the rule [`Scheduler::new`] uses: one
//! inequality per resource dimension, with an escape hatch for a job alone on a worker. Any other
//! rule goes to [`Scheduler::with_admission`].
//!
//! A worker with 100 bytes of memory and two slots, running one 60-byte job, takes a 40-byte job
//! beside it but not a 41-byte one. The scheduler asks with demands that take one [`SLOTS`]
//! each ([`Resources::with_slots`]), as the `job` closure builds them:
//!
//! ```
//! use whelm::{Admission, ProductionAdmission, Resources, WorkerState, WorkerView};
//!
//! let job = |bytes| Resources::mem(bytes).with_slots(1);
//! let state = WorkerState {
//!     id: 1,
//!     class: "cpu".into(),
//!     slots: 2,
//!     budget: Resources::mem(100),
//!     ..Default::default()
//! };
//! let view = WorkerView {
//!     state: &state,
//!     placed: job(60),
//! };
//! assert!(ProductionAdmission.admits(&job(40), &view));
//! assert!(!ProductionAdmission.admits(&job(41), &view));
//! ```
//!
//! [`Scheduler::new`]: crate::Scheduler::new
//! [`Scheduler::with_admission`]: crate::Scheduler::with_admission

use crate::{DIMS, HARD, Resources, SLOTS, WorkerState};

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
/// use whelm::{DEV, MEM, Resources, SLOTS, WorkerState, WorkerView};
///
/// // A demand as the scheduler sends it: one slot per job.
/// let job = |bytes| Resources::mem(bytes).with_slots(1);
/// let state = WorkerState {
///     id: 1,
///     class: "cpu".into(),
///     slots: 4,
///     budget: Resources::mem(100),
///     reported_used: Resources::mem(30),
///     reported_baseline: Resources::mem(10),
///     ..Default::default()
/// };
/// let view = WorkerView {
///     state: &state,
///     placed: job(50),
/// };
/// assert_eq!(view.running(), 1);
/// // Memory and slots are enforced; device memory, with a zero budget, is not.
/// assert!(view.enforced(MEM) && view.enforced(SLOTS) && !view.enforced(DEV));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct WorkerView<'a> {
    /// The worker's last reported state.
    pub state: &'a WorkerState,
    /// Sum of the demands of the attempts placed on the worker and not yet ended, each counting
    /// one [`SLOTS`].
    pub placed: Resources,
}

impl WorkerView<'_> {
    /// Attempts placed here: `placed[SLOTS]`, since each demands one slot.
    pub fn running(&self) -> u64 {
        self.placed[SLOTS]
    }

    /// Whether dimension `d` is enforced: always if it is [`HARD`], else if its capacity is known
    /// (nonzero).
    pub fn enforced(&self, d: usize) -> bool {
        HARD[d] || self.state.capacity()[d] > 0
    }

    /// What a job of demand `demand` counts for here: at least the worker's
    /// [`per_task`](WorkerState::per_task) in every dimension.
    ///
    /// On a worker whose jobs each take at least 30 bytes of device memory, a job declaring 5
    /// counts for 30:
    ///
    /// ```
    /// use whelm::{Resources, WorkerState, WorkerView};
    ///
    /// let state = WorkerState {
    ///     id: 1,
    ///     class: "gpu".into(),
    ///     slots: 4,
    ///     budget: Resources::mem(100).with_dev(100),
    ///     per_task: Resources::ZERO.with_dev(30),
    ///     ..Default::default()
    /// };
    /// let view = WorkerView {
    ///     state: &state,
    ///     placed: Resources::ZERO,
    /// };
    /// let demand = Resources::mem(5).with_dev(5);
    /// assert_eq!(view.charge(&demand), Resources::mem(5).with_dev(30));
    /// ```
    pub fn charge(&self, demand: &Resources) -> Resources {
        demand.max(self.state.per_task)
    }

    /// Usage as the production rule counts it, per dimension: `max(reported_used,
    /// reported_baseline + max(placed, running * per_task))`.
    ///
    /// The reported figure lags (heartbeats); the placed sum is exact but only an estimate of what
    /// the jobs use. Taking the maximum is conservative in both directions.
    ///
    /// For the worker of the [type-level example](WorkerView), the baseline plus the placed 50
    /// bytes outweighs the 30 reported:
    ///
    /// ```
    /// # use whelm::{Resources, WorkerState, WorkerView};
    /// # let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     slots: 4,
    /// #     budget: Resources::mem(100),
    /// #     reported_used: Resources::mem(30),
    /// #     reported_baseline: Resources::mem(10),
    /// #     ..Default::default()
    /// # };
    /// # let view = WorkerView {
    /// #     state: &state,
    /// #     placed: job(50),
    /// # };
    /// assert_eq!(view.used(), job(60)); // max(30, 10 + 50) bytes, and one slot
    /// ```
    pub fn used(&self) -> Resources {
        let floor = self.state.per_task.saturating_mul(self.running());
        self.state
            .reported_used
            .max(self.state.reported_baseline + self.placed.max(floor))
    }

    /// Headroom per dimension, `budget - used`; negative when over-committed, `None` where the
    /// dimension is not [`enforced`](Self::enforced).
    ///
    /// For the worker of the [type-level example](WorkerView):
    ///
    /// ```
    /// # use whelm::{Resources, WorkerState, WorkerView};
    /// # let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     slots: 4,
    /// #     budget: Resources::mem(100),
    /// #     reported_used: Resources::mem(30),
    /// #     reported_baseline: Resources::mem(10),
    /// #     ..Default::default()
    /// # };
    /// # let view = WorkerView {
    /// #     state: &state,
    /// #     placed: job(50),
    /// # };
    /// // 100 - 60 bytes of memory, device memory unknown, 4 - 1 slots.
    /// assert_eq!(view.headroom(), [Some(40), None, Some(3)]);
    /// ```
    pub fn headroom(&self) -> [Option<i64>; DIMS] {
        let used = self.used();
        std::array::from_fn(|d| {
            self.enforced(d).then(|| {
                (self.state.capacity()[d] as i128 - used[d] as i128)
                    .clamp(i64::MIN as i128, i64::MAX as i128) as i64
            })
        })
    }

    /// The enforced dimensions in which a job of demand `demand` does not fit beside the jobs
    /// already here: `used + charge > budget`. It ignores the escape hatch, so it names what a
    /// refusal is about rather than deciding one.
    ///
    /// For the worker of the [type-level example](WorkerView):
    ///
    /// ```
    /// # use whelm::{Resources, WorkerState, WorkerView};
    /// # let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     slots: 4,
    /// #     budget: Resources::mem(100),
    /// #     reported_used: Resources::mem(30),
    /// #     reported_baseline: Resources::mem(10),
    /// #     ..Default::default()
    /// # };
    /// # let view = WorkerView {
    /// #     state: &state,
    /// #     placed: job(50),
    /// # };
    /// # use whelm::MEM;
    /// assert_eq!(view.short(&job(40)).count(), 0);
    /// assert_eq!(view.short(&job(41)).collect::<Vec<_>>(), vec![MEM]);
    /// ```
    pub fn short(&self, demand: &Resources) -> impl Iterator<Item = usize> + '_ {
        let (used, charge) = (self.used(), self.charge(demand));
        (0..DIMS).filter(move |&d| {
            self.enforced(d) && used[d].saturating_add(charge[d]) > self.state.capacity()[d]
        })
    }

    /// The fraction of capacity left after placing `demand`, in the bottleneck dimension: `min_d
    /// (budget - used - charge) / budget` over the enforced soft dimensions (one minus the
    /// dominant share, as in dominant-resource fairness), infinite when none is enforced.
    ///
    /// This is the one scalar workers are compared by: smallest for the tightest fit, largest
    /// (with a zero demand) for the most headroom. [`HARD`] dimensions are counts rather than
    /// capacity to pack into, and are left to [`ScoreTerm::Load`](crate::ScoreTerm::Load).
    ///
    /// For the worker of the [type-level example](WorkerView), 60 of 100 bytes in use:
    ///
    /// ```
    /// # use whelm::{Resources, WorkerState, WorkerView};
    /// # let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     slots: 4,
    /// #     budget: Resources::mem(100),
    /// #     reported_used: Resources::mem(30),
    /// #     reported_baseline: Resources::mem(10),
    /// #     ..Default::default()
    /// # };
    /// # let view = WorkerView {
    /// #     state: &state,
    /// #     placed: job(50),
    /// # };
    /// assert_eq!(view.free_share(&Resources::ZERO), 0.4);
    /// assert_eq!(view.free_share(&job(20)), 0.2);
    /// // A worker with no memory budget enforces no soft dimension.
    /// let unknown = WorkerState {
    ///     id: 2,
    ///     class: "cpu".into(),
    ///     slots: 4,
    ///     ..Default::default()
    /// };
    /// let view = WorkerView {
    ///     state: &unknown,
    ///     placed: Resources::ZERO,
    /// };
    /// assert_eq!(view.free_share(&job(20)), f64::INFINITY);
    /// ```
    pub fn free_share(&self, demand: &Resources) -> f64 {
        let (used, charge) = (self.used(), self.charge(demand));
        (0..DIMS)
            .filter(|&d| !HARD[d] && self.enforced(d))
            .map(|d| {
                let cap = self.state.capacity()[d] as f64;
                (cap - used[d] as f64 - charge[d] as f64) / cap
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
/// admits it.
///
/// # Example
///
/// A rule that counts slots and nothing else: two 80-byte jobs share a 100-byte worker, where
/// [`ProductionAdmission`] would run them one at a time.
///
/// ```
/// use whelm::{
///     Admission, Config, Input, JobSpec, Policy, Resources, Scheduler, Time, WorkerState,
///     WorkerView,
/// };
///
/// /// Admits while a slot is free, whatever the memory.
/// struct SlotsOnly;
///
/// impl Admission for SlotsOnly {
///     fn admits(&self, _demand: &Resources, w: &WorkerView) -> bool {
///         w.running() < w.state.slots as u64
///     }
/// }
///
/// let mut s = Scheduler::with_admission(Config::fifo(), SlotsOnly);
/// s.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         slots: 2,
///         budget: Resources::mem(100),
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
/// assert!(s.explain(2).unwrap().contains("slots full on 1 worker(s)"));
/// ```
pub trait Admission {
    /// Whether `w` admits a job with demand `demand`.
    fn admits(&self, demand: &Resources, w: &WorkerView) -> bool;

    /// An upper bound on admitted demands, used only to skip hopeless checks quickly: if
    /// `admits(d, w)` then `d.fits_within(bound(w))`. `None` means `w` admits nothing. The default
    /// is no bound. [`ProductionAdmission::bound`] has an example.
    fn bound(&self, _w: &WorkerView) -> Option<Resources> {
        Some(Resources::MAX)
    }
}

/// The production admission rule, one inequality per enforced resource dimension `d`:
///
/// ```text
/// admit(job on w) iff for every enforced d:
///                       max(reported_used[d],
///                           reported_baseline[d] + max(placed[d], running * per_task[d]))
///                         + max(demand[d], per_task[d]) <= budget[d]
///                     or (d is soft and running == 0)      // escape hatch: a job alone goes
/// ```
///
/// [`HARD`] dimensions (slots) are always enforced and have no escape hatch; a soft dimension is
/// enforced where its capacity is nonzero. With a device `per_task` alone (jobs without device
/// demands) the device inequality is the per-worker count `(running + 1) * per_task <= cap`; with
/// per-job device demands it is their sum.
///
/// `reported_baseline` must exclude the running jobs (a worker's `baseline_excl`: its rolling RSS
/// floor minus their estimates). A floor that contains them counts them twice, once in it and
/// once in `placed`, and keeps a busy worker a few GB short of its budget.
///
/// # Examples
///
/// The escape hatch: an empty worker takes a job larger than its memory budget, a busy one does
/// not. Slots have no escape hatch, so a worker without slots takes nothing.
///
/// ```
/// use whelm::{Admission, ProductionAdmission, Resources, WorkerState, WorkerView};
///
/// let job = |bytes| Resources::mem(bytes).with_slots(1);
/// let state = WorkerState {
///     id: 1,
///     class: "cpu".into(),
///     slots: 2,
///     budget: Resources::mem(100),
///     ..Default::default()
/// };
/// let view = |placed| WorkerView {
///     state: &state,
///     placed,
/// };
/// assert!(ProductionAdmission.admits(&job(1000), &view(Resources::ZERO)));
/// assert!(!ProductionAdmission.admits(&job(1000), &view(job(1))));
///
/// let no_slots = WorkerState {
///     id: 2,
///     class: "cpu".into(),
///     slots: 0,
///     budget: Resources::mem(100),
///     ..Default::default()
/// };
/// let view = WorkerView {
///     state: &no_slots,
///     placed: Resources::ZERO,
/// };
/// assert!(!ProductionAdmission.admits(&job(1), &view));
/// ```
///
/// A device `per_task` alone caps the jobs per worker: with 100 bytes of device memory and 30 per
/// job, three jobs fit and a fourth does not, whatever their own (zero) device demands.
///
/// ```
/// # use whelm::{Admission, ProductionAdmission, Resources, WorkerState, WorkerView};
/// let state = WorkerState {
///     id: 1,
///     class: "gpu".into(),
///     slots: 8,
///     budget: Resources::ZERO.with_dev(100),
///     per_task: Resources::ZERO.with_dev(30),
///     ..Default::default()
/// };
/// let admits_with = |running| {
///     let view = WorkerView {
///         state: &state,
///         placed: Resources::ZERO.with_slots(running),
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
        let empty = w.running() == 0;
        w.short(demand).all(|d| !HARD[d] && empty)
    }

    /// The headroom left under the production rule (unbounded in the dimensions not enforced, and
    /// in the soft ones when the worker is empty).
    ///
    /// ```
    /// use whelm::{Admission, MEM, ProductionAdmission, Resources, SLOTS, WorkerState, WorkerView};
    ///
    /// let job = |bytes| Resources::mem(bytes).with_slots(1);
    /// let state = WorkerState {
    ///     id: 1,
    ///     class: "cpu".into(),
    ///     slots: 2,
    ///     budget: Resources::mem(100),
    ///     ..Default::default()
    /// };
    /// let view = |placed| WorkerView {
    ///     state: &state,
    ///     placed,
    /// };
    /// // Empty: memory unbounded (the escape hatch), two slots.
    /// let empty = ProductionAdmission.bound(&view(Resources::ZERO)).unwrap();
    /// assert_eq!((empty[MEM], empty[SLOTS]), (u64::MAX, 2));
    /// // One 60-byte job: 40 bytes and one slot left.
    /// let busy = ProductionAdmission.bound(&view(job(60))).unwrap();
    /// assert_eq!((busy[MEM], busy[SLOTS]), (40, 1));
    /// // Both slots taken: nothing is admitted.
    /// let full = Resources::mem(60).with_slots(2);
    /// assert_eq!(ProductionAdmission.bound(&view(full)), None);
    /// ```
    fn bound(&self, w: &WorkerView) -> Option<Resources> {
        let (used, empty) = (w.used(), w.running() == 0);
        let mut bound = Resources::MAX;
        for d in (0..DIMS).filter(|&d| w.enforced(d) && (HARD[d] || !empty)) {
            let room = w.state.capacity()[d].checked_sub(used[d])?;
            // Every job counts for at least `per_task` and takes a slot, so less room than that
            // admits nothing.
            let least = w.state.per_task[d].max((d == SLOTS) as u64);
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
    use crate::{DEV, MEM};

    /// A worker state with the given capacity and reported usage.
    fn worker(slots: usize, budget: u64, used: u64, baseline: u64) -> WorkerState {
        WorkerState {
            id: 1,
            slots,
            budget: Resources::mem(budget),
            reported_used: Resources::mem(used),
            reported_baseline: Resources::mem(baseline),
            ..Default::default()
        }
    }

    /// `running` attempts' worth of slots on top of `r`.
    fn with_running(mut r: Resources, running: u64) -> Resources {
        r[SLOTS] = running;
        r
    }

    /// The rule, its escape hatch and its bound, at the boundaries.
    #[test]
    fn production_rule() {
        let a = ProductionAdmission;
        let s = worker(2, 100, 30, 10);
        let view = |running, placed| WorkerView {
            state: &s,
            placed: with_running(Resources::mem(placed), running),
        };
        // Escape hatch: alone, anything goes, even beyond the budget.
        assert!(a.admits(&Resources::mem(1000), &view(0, 0)));
        // max(30, 10 + 50) + 40 = 100 <= 100.
        assert!(a.admits(&Resources::mem(40), &view(1, 50)));
        assert!(!a.admits(&Resources::mem(41), &view(1, 50)));
        // Reported usage dominates: max(30, 10 + 5) + 70 = 100.
        assert!(a.admits(&Resources::mem(70), &view(1, 5)));
        assert!(!a.admits(&Resources::mem(71), &view(1, 5)));
        // Slots full: no escape hatch for a hard dimension.
        assert!(!a.admits(&with_running(Resources::ZERO, 1), &view(2, 0)));
        assert_eq!(a.bound(&view(2, 0)), None);
        assert_eq!(
            a.bound(&view(1, 50)),
            Some(with_running(Resources::mem(40).with_dev(u64::MAX), 1))
        );
        assert_eq!(view(1, 50).headroom(), [Some(40), None, Some(1)]);
    }

    /// `baseline_excl` (the rolling RSS floor minus the estimates running) as `reported_baseline`
    /// gives `max(rss, baseline_excl + Σ placed) + demand <= budget`: the floor does not count the
    /// running jobs twice.
    #[test]
    fn baseline_excl_removes_the_double_count() {
        // RSS 60 with 40 of estimates running; the floor (50) contains those jobs.
        let (rss, floor, placed, budget) = (60, 50, 40, 100);
        let view = |s| WorkerView {
            state: s,
            placed: with_running(Resources::mem(placed), 4),
        };
        let with_floor = worker(16, budget, rss, floor);
        let with_excl = worker(16, budget, rss, floor - placed);
        // Floor: max(60, 50 + 40) + 15 = 105 > 100. Excl: max(60, 10 + 40) + 40 = 100.
        assert!(!ProductionAdmission.admits(&Resources::mem(15), &view(&with_floor)));
        assert!(ProductionAdmission.admits(&Resources::mem(40), &view(&with_excl)));
        assert!(!ProductionAdmission.admits(&Resources::mem(41), &view(&with_excl)));
        // RSS stays the safety term: estimates that undercount cannot admit past it.
        let undercount = worker(16, budget, 95, 0);
        assert!(!ProductionAdmission.admits(&Resources::mem(6), &view(&undercount)));
    }

    /// Every dimension follows the same inequality, with `per_task` as a per-job floor; a zero
    /// capacity component of a soft dimension is not enforced, host memory included.
    #[test]
    fn one_rule_per_dimension() {
        let a = ProductionAdmission;
        let s = WorkerState {
            id: 1,
            slots: 8,
            budget: Resources::mem(100).with_dev(100),
            per_task: Resources::mem(20).with_dev(30),
            ..Default::default()
        };
        let view = |running, placed| WorkerView {
            state: &s,
            placed: with_running(placed, running),
        };
        // Device: max(10, 2 * 30) + max(5, 30) = 90 <= 100, and 60 + 41 > 100.
        let v = view(2, Resources::mem(40).with_dev(10));
        assert!(a.admits(&Resources::mem(10).with_dev(5), &v));
        assert!(!a.admits(&Resources::mem(10).with_dev(41), &v));
        assert_eq!(v.short(&Resources::mem(70)).collect::<Vec<_>>(), vec![MEM]);
        let v = view(3, Resources::mem(40));
        assert_eq!(v.short(&Resources::ZERO).collect::<Vec<_>>(), vec![DEV]);
        // Host: the floor counts 3 * 20 = 60 against the 40 placed.
        assert_eq!(v.headroom(), [Some(40), Some(10), Some(5)]);
        assert_eq!(a.bound(&v), None);
        // A zero memory budget leaves its dimension unenforced; slots stay enforced.
        let free = WorkerState {
            id: 2,
            slots: 8,
            ..Default::default()
        };
        let v = WorkerView {
            state: &free,
            placed: with_running(Resources::mem(1 << 40), 7),
        };
        let mut huge = Resources::MAX;
        huge[SLOTS] = 1;
        assert!(a.admits(&huge, &v));
        assert!(!a.admits(&Resources::MAX, &v));
        assert_eq!(v.headroom(), [None, None, Some(1)]);
        assert_eq!(v.free_share(&Resources::MAX), f64::INFINITY);
    }

    /// The scalar comparison is the bottleneck memory dimension's free fraction; slots do not
    /// enter it.
    #[test]
    fn free_share_is_the_bottleneck() {
        let s = WorkerState {
            id: 1,
            slots: 2,
            budget: Resources::mem(100).with_dev(10),
            ..Default::default()
        };
        let v = WorkerView {
            state: &s,
            placed: with_running(Resources::mem(50).with_dev(2), 1),
        };
        // Host 1 - 60/100 = 0.4, device 1 - 4/10 = 0.6; slots would be 1 - 2/2 = 0.
        let share = v.free_share(&with_running(Resources::mem(10).with_dev(2), 1));
        assert!((share - 0.4).abs() < 1e-12);
    }

    /// Slots are hard: without a free one, not even the escape hatch admits.
    #[test]
    fn zero_slots_admit_nothing() {
        let s = worker(0, 100, 0, 0);
        let v = WorkerView {
            state: &s,
            placed: Resources::ZERO,
        };
        let one_slot = with_running(Resources::ZERO, 1);
        assert!(!ProductionAdmission.admits(&one_slot, &v));
        assert_eq!(ProductionAdmission.bound(&v), None);
    }
}
