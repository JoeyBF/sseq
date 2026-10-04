//! The admission rule: whether a worker accepts a job right now.
//!
//! The [`Scheduler`](crate::Scheduler) places a job only where its [`Admission`] rule admits it,
//! so the rule alone enforces capacity, slots included. The rule sees a worker as a
//! [`WorkerView`]: the resources the configuration declares, the worker's last reported
//! [`WorkerState`], and the demands the scheduler has placed there itself; and a job's demand as
//! [`Amounts`] of the declared resources. [`ProductionAdmission`] is the rule [`Scheduler::new`]
//! uses: one inequality per declared resource, with an escape hatch for a job alone on a worker in
//! the soft ones. Any other rule goes to [`Scheduler::with_admission`].
//!
//! A worker with 100 bytes of memory and two slots, running one 60-byte job, takes a 40-byte job
//! beside it but not a 41-byte one. [`WorkerView::new`] builds the view the scheduler would pass,
//! and [`WorkerView::demand`] a demand as the scheduler holds it, with the slot every job takes
//! filled in:
//!
//! ```
//! use whelm::{
//!     Admission, Config, MEMORY, ProductionAdmission, Resources, SLOTS, WorkerState, WorkerView,
//! };
//!
//! let config = Config::default();
//! let state = WorkerState {
//!     id: 1,
//!     class: "cpu".into(),
//!     capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 2),
//!     ..Default::default()
//! };
//! let placed = Resources::new().with(MEMORY, 60).with(SLOTS, 1);
//! let view = WorkerView::new(&config.resources, &state, &placed, 1);
//! let job = |bytes| view.demand(&Resources::new().with(MEMORY, bytes));
//! assert!(ProductionAdmission.admits(&job(40), &view));
//! assert!(!ProductionAdmission.admits(&job(41), &view));
//! ```
//!
//! [`Scheduler::new`]: crate::Scheduler::new
//! [`Scheduler::with_admission`]: crate::Scheduler::with_admission

use std::borrow::Cow;

#[cfg(doc)]
use crate::{Config, JobSpec, ScoreTerm};
use crate::{
    Resource, Resources, WorkerState,
    resources::{Dense, dense, named, position},
};

/// A worker's reported amounts over a declaration: the scheduler's working form of a
/// [`WorkerState`], converted once per report.
#[derive(Clone, Debug)]
pub(crate) struct WorkerAmounts {
    pub(crate) capacity: Dense,
    pub(crate) per_task: Dense,
    pub(crate) reported_used: Dense,
    pub(crate) reported_baseline: Dense,
}

impl WorkerAmounts {
    /// `state`'s amounts over the declaration `resources`.
    ///
    /// # Panics
    ///
    /// If `state` names a resource that `resources` does not declare.
    pub(crate) fn new(resources: &[Resource], state: &WorkerState) -> Self {
        let convert = |field: &str, r: &Resources| {
            dense(resources, r).unwrap_or_else(|name| {
                panic!(
                    "worker {} reports {field} of resource {name:?}, which Config::resources does \
                     not declare",
                    state.id
                )
            })
        };
        Self {
            capacity: convert("capacity", &state.capacity),
            per_task: convert("per_task", &state.per_task),
            reported_used: convert("reported_used", &state.reported_used),
            reported_baseline: convert("reported_baseline", &state.reported_baseline),
        }
    }
}

/// An amount of each resource of a declaration: a job's demand as an [`Admission`] rule sees it,
/// or a rule's [bound](Admission::bound).
///
/// A [`WorkerView`] makes them ([`demand`](WorkerView::demand),
/// [`unbounded`](WorkerView::unbounded)), and the scheduler passes the demands it holds, with
/// [default demands](Resource::default_demand) filled in. Resources are looked up by name, as in
/// [`Resources`]; a resource the declaration leaves out reads as zero.
///
/// # Examples
///
/// ```
/// use whelm::{Config, MEMORY, Resource, Resources, SLOTS, WorkerState, WorkerView};
///
/// let config = Config::default();
/// let state = WorkerState::default();
/// let view = WorkerView::new(&config.resources, &state, &Resources::new(), 0);
/// let mut demand = view.demand(&Resources::new().with(MEMORY, 5));
/// assert_eq!((demand.get(MEMORY), demand.get(SLOTS)), (5, 1));
/// demand.set(SLOTS, 2);
/// let names: Vec<(&str, u64)> = demand.iter().map(|(r, x)| (&*r.name, x)).collect();
/// assert_eq!(names, [("memory", 5), ("device memory", 0), ("slots", 2)]);
/// assert_eq!(demand.get(Resource::new("gpus")), 0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Amounts<'a> {
    resources: &'a [Resource],
    values: Cow<'a, [u64]>,
}

impl<'a> Amounts<'a> {
    /// The amounts `values` of the declaration `resources`, by position.
    pub(crate) fn new(resources: &'a [Resource], values: impl Into<Cow<'a, [u64]>>) -> Self {
        Self {
            resources,
            values: values.into(),
        }
    }

    /// The amounts by position in the declaration.
    pub(crate) fn values(&self) -> &[u64] {
        &self.values
    }

    /// The amount of `resource`; zero if it is not declared.
    pub fn get(&self, resource: impl AsRef<Resource>) -> u64 {
        position(self.resources, &resource.as_ref().name).map_or(0, |d| self.values[d])
    }

    /// Set the amount of `resource`.
    ///
    /// # Panics
    ///
    /// If `resource` is not declared.
    pub fn set(&mut self, resource: impl AsRef<Resource>, amount: u64) {
        let name = &resource.as_ref().name;
        let d = position(self.resources, name)
            .unwrap_or_else(|| panic!("resource {name:?} is not declared"));
        self.values.to_mut()[d] = amount;
    }

    /// Each declared resource and its amount, in declaration order.
    pub fn iter(&self) -> impl Iterator<Item = (&'a Resource, u64)> + '_ {
        self.resources.iter().zip(self.values.iter().copied())
    }

    /// The amounts keyed by name.
    pub fn to_resources(&self) -> Resources {
        named(self.resources, &self.values)
    }
}

/// One declared resource on a worker, as [`WorkerView::resources`] lists them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Usage<'a> {
    /// The resource.
    pub resource: &'a Resource,
    /// The worker's capacity of it.
    pub capacity: u64,
    /// The worker's [`per_task`](WorkerState::per_task) floor in it.
    pub per_task: u64,
    /// The sum of the demands placed on the worker ([`WorkerView::placed`]).
    pub placed: u64,
    /// Usage as the production rule counts it ([`WorkerView::used`]).
    pub used: u64,
    /// `capacity - used`, or `None` where the resource is not enforced
    /// ([`WorkerView::headroom`]).
    pub headroom: Option<i64>,
}

/// A worker as an [`Admission`] rule sees it: its state plus the library's own bookkeeping.
///
/// The methods are the vocabulary [`ProductionAdmission`] is written in, and a custom rule may
/// use them too; [`free_share`](Self::free_share) is also what [`ScoreTerm::Tightest`] and
/// reservations rank workers by. Each takes a [`Resource`] and looks it up by name in the
/// declaration, with a linear scan; a resource the declaration leaves out has zero of
/// everything and is not enforced.
///
/// A worker with 100 bytes of memory, unknown device memory and four slots reports 30 bytes in
/// use, 10 of them its own baseline, and runs one job placed with 50 bytes. The examples on the
/// methods continue from this one.
///
/// ```
/// use whelm::{Config, DEVICE_MEMORY, MEMORY, Resources, SLOTS, WorkerState, WorkerView};
///
/// let config = Config::default();
/// let state = WorkerState {
///     id: 1,
///     class: "cpu".into(),
///     capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 4),
///     reported_used: Resources::new().with(MEMORY, 30),
///     reported_baseline: Resources::new().with(MEMORY, 10),
///     ..Default::default()
/// };
/// let placed = Resources::new().with(MEMORY, 50).with(SLOTS, 1);
/// let view = WorkerView::new(&config.resources, &state, &placed, 1);
/// // Memory and slots are enforced; device memory, with a zero capacity, is not.
/// assert!(view.enforced(MEMORY) && view.enforced(SLOTS));
/// assert!(!view.enforced(DEVICE_MEMORY));
/// assert_eq!(
///     (view.capacity(SLOTS), view.placed(SLOTS), view.running()),
///     (4, 1, 1)
/// );
/// ```
#[derive(Clone, Debug)]
pub struct WorkerView<'a> {
    resources: &'a [Resource],
    state: &'a WorkerState,
    amounts: Cow<'a, WorkerAmounts>,
    placed: Cow<'a, [u64]>,
    running: usize,
}

impl<'a> WorkerView<'a> {
    /// The view of a worker in state `state`, under the declaration `resources`, running
    /// `running` attempts whose demands sum to `placed`: what a [`Scheduler`](crate::Scheduler)
    /// would pass a rule, for calling one directly.
    ///
    /// `placed` is taken as it is, so it should include the default demands the scheduler fills
    /// in: under the default declaration, one slot per running attempt.
    ///
    /// # Panics
    ///
    /// If `state` or `placed` names a resource that `resources` does not declare.
    pub fn new(
        resources: &'a [Resource],
        state: &'a WorkerState,
        placed: &Resources,
        running: usize,
    ) -> Self {
        let placed = dense(resources, placed)
            .unwrap_or_else(|name| panic!("placed names resource {name:?}, which is not declared"));
        Self::from_parts(
            resources,
            state,
            Cow::Owned(WorkerAmounts::new(resources, state)),
            Cow::Owned(placed),
            running,
        )
    }

    /// The view from the scheduler's own records.
    pub(crate) fn from_parts(
        resources: &'a [Resource],
        state: &'a WorkerState,
        amounts: Cow<'a, WorkerAmounts>,
        placed: Cow<'a, [u64]>,
        running: usize,
    ) -> Self {
        Self {
            resources,
            state,
            amounts,
            placed,
            running,
        }
    }

    /// This view with another load: `running` attempts whose demands sum to `placed`.
    pub(crate) fn with_load(self, placed: impl Into<Cow<'a, [u64]>>, running: usize) -> Self {
        Self {
            placed: placed.into(),
            running,
            ..self
        }
    }

    /// This view with the worker reporting none in use beyond its baseline.
    pub(crate) fn without_reported_use(mut self) -> Self {
        self.amounts.to_mut().reported_used.fill(0);
        self
    }

    /// The worker's last reported state.
    pub fn state(&self) -> &'a WorkerState {
        self.state
    }

    /// Attempts placed on the worker and not yet ended.
    pub fn running(&self) -> usize {
        self.running
    }

    /// `demand` as the scheduler holds a job's demand: over the declared resources, with
    /// [default demands](Resource::default_demand) filled in.
    ///
    /// # Panics
    ///
    /// If `demand` names a resource that is not declared (the scheduler rejects such a job).
    pub fn demand(&self, demand: &Resources) -> Amounts<'a> {
        let mut values = dense(self.resources, demand).unwrap_or_else(|name| {
            panic!("the demand names resource {name:?}, which is not declared")
        });
        fill_defaults(self.resources, &mut values);
        Amounts::new(self.resources, values)
    }

    /// No bound in any declared resource: the default [`Admission::bound`].
    pub fn unbounded(&self) -> Amounts<'a> {
        Amounts::new(self.resources, vec![u64::MAX; self.resources.len()])
    }

    /// Each declared resource on the worker, in declaration order.
    ///
    /// For the worker of the [type-level example](WorkerView):
    ///
    /// ```
    /// # use whelm::{Config, MEMORY, Resources, SLOTS, WorkerState, WorkerView};
    /// # let config = Config::default();
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 4),
    /// #     reported_used: Resources::new().with(MEMORY, 30),
    /// #     reported_baseline: Resources::new().with(MEMORY, 10),
    /// #     ..Default::default()
    /// # };
    /// # let placed = Resources::new().with(MEMORY, 50).with(SLOTS, 1);
    /// # let view = WorkerView::new(&config.resources, &state, &placed, 1);
    /// let rows: Vec<(&str, u64, Option<i64>)> = view
    ///     .resources()
    ///     .map(|u| (&*u.resource.name, u.used, u.headroom))
    ///     .collect();
    /// assert_eq!(
    ///     rows,
    ///     [
    ///         ("memory", 60, Some(40)),
    ///         ("device memory", 0, None),
    ///         ("slots", 1, Some(3))
    ///     ]
    /// );
    /// ```
    pub fn resources(&self) -> impl Iterator<Item = Usage<'a>> + '_ {
        self.resources
            .iter()
            .enumerate()
            .map(|(d, resource)| Usage {
                resource,
                capacity: self.amounts.capacity[d],
                per_task: self.amounts.per_task[d],
                placed: self.placed[d],
                used: self.used_at(d),
                headroom: self.headroom_at(d),
            })
    }

    /// The worker's capacity of `resource`.
    pub fn capacity(&self, resource: impl AsRef<Resource>) -> u64 {
        self.index(resource.as_ref())
            .map_or(0, |d| self.amounts.capacity[d])
    }

    /// The sum of the demands of the attempts placed on the worker and not yet ended, in
    /// `resource`, with their [default demands](Resource::default_demand) filled in.
    pub fn placed(&self, resource: impl AsRef<Resource>) -> u64 {
        self.index(resource.as_ref()).map_or(0, |d| self.placed[d])
    }

    /// Usage of `resource` as the production rule counts it: `max(reported_used,
    /// reported_baseline + max(placed, running * per_task))`.
    ///
    /// The reported figure lags (heartbeats); the placed sum is exact but only an estimate of what
    /// the jobs use. Taking the maximum is conservative in both directions.
    ///
    /// For the worker of the [type-level example](WorkerView), the baseline plus the placed 50
    /// bytes outweighs the 30 reported:
    ///
    /// ```
    /// # use whelm::{Config, MEMORY, Resources, SLOTS, WorkerState, WorkerView};
    /// # let config = Config::default();
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 4),
    /// #     reported_used: Resources::new().with(MEMORY, 30),
    /// #     reported_baseline: Resources::new().with(MEMORY, 10),
    /// #     ..Default::default()
    /// # };
    /// # let placed = Resources::new().with(MEMORY, 50).with(SLOTS, 1);
    /// # let view = WorkerView::new(&config.resources, &state, &placed, 1);
    /// assert_eq!(view.used(MEMORY), 60); // max(30, 10 + 50)
    /// ```
    pub fn used(&self, resource: impl AsRef<Resource>) -> u64 {
        self.index(resource.as_ref()).map_or(0, |d| self.used_at(d))
    }

    /// Whether `resource` is enforced: always if it is [hard](Resource::hard), else if its
    /// capacity is known (nonzero).
    pub fn enforced(&self, resource: impl AsRef<Resource>) -> bool {
        self.index(resource.as_ref())
            .is_some_and(|d| self.enforced_at(d))
    }

    /// Headroom in `resource`, `capacity - used`; negative when over-committed, `None` where the
    /// resource is not [`enforced`](Self::enforced).
    ///
    /// For the worker of the [type-level example](WorkerView):
    ///
    /// ```
    /// # use whelm::{Config, DEVICE_MEMORY, MEMORY, Resources, SLOTS, WorkerState, WorkerView};
    /// # let config = Config::default();
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 4),
    /// #     reported_used: Resources::new().with(MEMORY, 30),
    /// #     reported_baseline: Resources::new().with(MEMORY, 10),
    /// #     ..Default::default()
    /// # };
    /// # let placed = Resources::new().with(MEMORY, 50).with(SLOTS, 1);
    /// # let view = WorkerView::new(&config.resources, &state, &placed, 1);
    /// // 100 - 60 bytes of memory, device memory unknown, 4 - 1 slots.
    /// assert_eq!(view.headroom(MEMORY), Some(40));
    /// assert_eq!(view.headroom(DEVICE_MEMORY), None);
    /// assert_eq!(view.headroom(SLOTS), Some(3));
    /// ```
    pub fn headroom(&self, resource: impl AsRef<Resource>) -> Option<i64> {
        self.headroom_at(self.index(resource.as_ref())?)
    }

    /// The enforced resources in which a job of demand `demand` does not fit beside the jobs
    /// already here, in declaration order: `used + max(demand, per_task) > capacity`. It ignores
    /// the escape hatch, so it names what a refusal is about rather than deciding one.
    ///
    /// For the worker of the [type-level example](WorkerView):
    ///
    /// ```
    /// # use whelm::{Config, MEMORY, Resources, SLOTS, WorkerState, WorkerView};
    /// # let config = Config::default();
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 4),
    /// #     reported_used: Resources::new().with(MEMORY, 30),
    /// #     reported_baseline: Resources::new().with(MEMORY, 10),
    /// #     ..Default::default()
    /// # };
    /// # let placed = Resources::new().with(MEMORY, 50).with(SLOTS, 1);
    /// # let view = WorkerView::new(&config.resources, &state, &placed, 1);
    /// let job = |bytes| view.demand(&Resources::new().with(MEMORY, bytes));
    /// assert_eq!(view.short(&job(40)).count(), 0);
    /// assert_eq!(view.short(&job(41)).collect::<Vec<_>>(), [&MEMORY]);
    /// ```
    pub fn short<'b>(&'b self, demand: &'b Amounts) -> impl Iterator<Item = &'a Resource> + 'b {
        self.short_at(demand.values()).map(|d| &self.resources[d])
    }

    /// The fraction of capacity left after placing `demand`, in the bottleneck resource: `min_d
    /// (capacity - used - max(demand, per_task)) / capacity` over the enforced soft resources (one
    /// minus the dominant share, as in dominant-resource fairness), infinite when none is
    /// enforced.
    ///
    /// This is the one scalar workers are compared by: smallest for the tightest fit, largest
    /// (with a zero demand) for the most headroom. Hard resources are counts rather than capacity
    /// to pack into, and are left to [`ScoreTerm::Load`].
    ///
    /// For the worker of the [type-level example](WorkerView), 60 of 100 bytes in use:
    ///
    /// ```
    /// # use whelm::{Config, MEMORY, Resources, SLOTS, WorkerState, WorkerView};
    /// # let config = Config::default();
    /// # let state = WorkerState {
    /// #     id: 1,
    /// #     class: "cpu".into(),
    /// #     capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 4),
    /// #     reported_used: Resources::new().with(MEMORY, 30),
    /// #     reported_baseline: Resources::new().with(MEMORY, 10),
    /// #     ..Default::default()
    /// # };
    /// # let placed = Resources::new().with(MEMORY, 50).with(SLOTS, 1);
    /// # let view = WorkerView::new(&config.resources, &state, &placed, 1);
    /// let job = |bytes| view.demand(&Resources::new().with(MEMORY, bytes));
    /// assert_eq!(view.free_share(&job(0)), 0.4);
    /// assert_eq!(view.free_share(&job(20)), 0.2);
    /// // A worker with no memory capacity enforces no soft resource.
    /// let unknown = WorkerState {
    ///     id: 2,
    ///     class: "cpu".into(),
    ///     capacity: Resources::new().with(SLOTS, 4),
    ///     ..Default::default()
    /// };
    /// let view = WorkerView::new(&config.resources, &unknown, &Resources::new(), 0);
    /// assert_eq!(view.free_share(&job(20)), f64::INFINITY);
    /// ```
    pub fn free_share(&self, demand: &Amounts) -> f64 {
        self.free_share_at(demand.values())
    }

    /// The declared resources.
    pub(crate) fn declared(&self) -> &'a [Resource] {
        self.resources
    }

    /// The position of `resource` in the declaration.
    fn index(&self, resource: &Resource) -> Option<usize> {
        position(self.resources, &resource.name)
    }

    /// [`enforced`](Self::enforced) by position.
    pub(crate) fn enforced_at(&self, d: usize) -> bool {
        self.resources[d].hard || self.amounts.capacity[d] > 0
    }

    /// What a job of demand `demand` counts for in resource `d`: at least the worker's
    /// [`per_task`](WorkerState::per_task).
    fn charge_at(&self, demand: &[u64], d: usize) -> u64 {
        demand[d].max(self.amounts.per_task[d])
    }

    /// [`used`](Self::used) by position.
    pub(crate) fn used_at(&self, d: usize) -> u64 {
        let a = &*self.amounts;
        let floor = a.per_task[d].saturating_mul(self.running as u64);
        (a.reported_used[d]).max(a.reported_baseline[d].saturating_add(self.placed[d].max(floor)))
    }

    /// [`used`](Self::used) in every declared resource.
    pub(crate) fn used_all(&self) -> Dense {
        (0..self.resources.len()).map(|d| self.used_at(d)).collect()
    }

    /// [`headroom`](Self::headroom) by position.
    pub(crate) fn headroom_at(&self, d: usize) -> Option<i64> {
        self.enforced_at(d).then(|| {
            (self.amounts.capacity[d] as i128 - self.used_at(d) as i128)
                .clamp(i64::MIN as i128, i64::MAX as i128) as i64
        })
    }

    /// [`short`](Self::short) by position.
    pub(crate) fn short_at<'b>(&'b self, demand: &'b [u64]) -> impl Iterator<Item = usize> + 'b {
        (0..self.resources.len()).filter(move |&d| {
            self.enforced_at(d)
                && self.used_at(d).saturating_add(self.charge_at(demand, d))
                    > self.amounts.capacity[d]
        })
    }

    /// [`free_share`](Self::free_share) of a demand by position.
    pub(crate) fn free_share_at(&self, demand: &[u64]) -> f64 {
        (0..self.resources.len())
            .filter(|&d| !self.resources[d].hard && self.enforced_at(d))
            .map(|d| {
                let cap = self.amounts.capacity[d] as f64;
                (cap - self.used_at(d) as f64 - self.charge_at(demand, d) as f64) / cap
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Replace each zero amount of `values` by its resource's default demand.
pub(crate) fn fill_defaults(resources: &[Resource], values: &mut [u64]) {
    for (x, r) in values.iter_mut().zip(resources) {
        if *x == 0 {
            *x = r.default_demand;
        }
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
/// admits it. It sees demands as the scheduler holds them: over the declared resources, with
/// [default demands](Resource::default_demand) filled in.
///
/// # Example
///
/// A rule that counts slots and nothing else: two 80-byte jobs share a 100-byte worker, where
/// [`ProductionAdmission`] would run them one at a time.
///
/// ```
/// use whelm::{
///     Admission, Amounts, Config, Input, JobSpec, MEMORY, Policy, Resources, SLOTS, Scheduler,
///     Time, Verdict, WorkerState, WorkerView,
/// };
///
/// /// Admits while a slot is free, whatever the memory.
/// struct SlotsOnly;
///
/// impl Admission for SlotsOnly {
///     fn admits(&self, _demand: &Amounts, w: &WorkerView) -> bool {
///         (w.running() as u64) < w.capacity(SLOTS)
///     }
/// }
///
/// let mut s = Scheduler::with_admission(Config::fifo(), SlotsOnly);
/// s.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 2),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// for id in 0..3 {
///     s.handle(
///         Input::Submit(JobSpec {
///             id,
///             demand: Resources::new().with(MEMORY, 80),
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
///         Verdict::Full {
///             dims: vec![SLOTS.name]
///         }
///     )]
/// );
/// ```
pub trait Admission {
    /// Whether `w` admits a job with demand `demand`.
    fn admits(&self, demand: &Amounts, w: &WorkerView) -> bool;

    /// An upper bound on admitted demands, used only to skip hopeless checks quickly: if
    /// `admits(d, w)` then every amount of `d` is at most the bound's. `None` means `w` admits
    /// nothing. The default is [no bound](WorkerView::unbounded).
    /// [`ProductionAdmission::bound`] has an example.
    fn bound<'a>(&self, w: &WorkerView<'a>) -> Option<Amounts<'a>> {
        Some(w.unbounded())
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
/// use whelm::{
///     Admission, Config, MEMORY, ProductionAdmission, Resources, SLOTS, WorkerState, WorkerView,
/// };
///
/// let config = Config::default();
/// let mem = |bytes| Resources::new().with(MEMORY, bytes);
/// let state = WorkerState {
///     id: 1,
///     class: "cpu".into(),
///     capacity: mem(100).with(SLOTS, 2),
///     ..Default::default()
/// };
/// // Whether `state` admits a job of `bytes` beside `running` jobs that took `placed`.
/// let admits = |state: &WorkerState, bytes, placed: &Resources, running| {
///     let view = WorkerView::new(&config.resources, state, placed, running);
///     ProductionAdmission.admits(&view.demand(&mem(bytes)), &view)
/// };
/// assert!(admits(&state, 1000, &Resources::new(), 0));
/// assert!(!admits(&state, 1000, &mem(1).with(SLOTS, 1), 1));
///
/// let no_slots = WorkerState {
///     id: 2,
///     class: "cpu".into(),
///     capacity: mem(100),
///     ..Default::default()
/// };
/// assert!(!admits(&no_slots, 1, &Resources::new(), 0));
/// ```
///
/// A device `per_task` alone caps the jobs per worker: with 100 bytes of device memory and 30 per
/// job, three jobs fit and a fourth does not, whatever their own (zero) device demands.
///
/// ```
/// use whelm::{
///     Admission, Config, DEVICE_MEMORY, ProductionAdmission, Resources, SLOTS, WorkerState,
///     WorkerView,
/// };
///
/// let config = Config::default();
/// let state = WorkerState {
///     id: 1,
///     class: "gpu".into(),
///     capacity: Resources::new().with(DEVICE_MEMORY, 100).with(SLOTS, 8),
///     per_task: Resources::new().with(DEVICE_MEMORY, 30),
///     ..Default::default()
/// };
/// let admits_with = |running| {
///     let placed = Resources::new().with(SLOTS, running);
///     let view = WorkerView::new(&config.resources, &state, &placed, running as usize);
///     ProductionAdmission.admits(&view.demand(&Resources::new()), &view)
/// };
/// assert!(admits_with(2));
/// assert!(!admits_with(3));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct ProductionAdmission;

impl Admission for ProductionAdmission {
    /// The production rule itself.
    fn admits(&self, demand: &Amounts, w: &WorkerView) -> bool {
        let empty = w.running == 0;
        w.short_at(demand.values())
            .all(|d| !w.resources[d].hard && empty)
    }

    /// The headroom left under the production rule (unbounded in the resources not enforced, and
    /// in the soft ones when the worker is empty).
    ///
    /// ```
    /// use whelm::{
    ///     Admission, Config, MEMORY, ProductionAdmission, Resources, SLOTS, WorkerState, WorkerView,
    /// };
    ///
    /// let config = Config::default();
    /// let state = WorkerState {
    ///     id: 1,
    ///     class: "cpu".into(),
    ///     capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 2),
    ///     ..Default::default()
    /// };
    /// let bound = |placed: &Resources, running| {
    ///     let view = WorkerView::new(&config.resources, &state, placed, running);
    ///     let b = ProductionAdmission.bound(&view)?;
    ///     Some((b.get(MEMORY), b.get(SLOTS)))
    /// };
    /// let busy = |bytes, running| Resources::new().with(MEMORY, bytes).with(SLOTS, running);
    /// // Empty: memory unbounded (the escape hatch), two slots.
    /// assert_eq!(bound(&Resources::new(), 0), Some((u64::MAX, 2)));
    /// // One 60-byte job: 40 bytes and one slot left.
    /// assert_eq!(bound(&busy(60, 1), 1), Some((40, 1)));
    /// // Both slots taken: nothing is admitted.
    /// assert_eq!(bound(&busy(60, 2), 2), None);
    /// ```
    fn bound<'a>(&self, w: &WorkerView<'a>) -> Option<Amounts<'a>> {
        let empty = w.running == 0;
        let mut bound = vec![u64::MAX; w.resources.len()];
        for (d, r) in w.resources.iter().enumerate() {
            if !w.enforced_at(d) || (!r.hard && empty) {
                continue;
            }
            let room = w.amounts.capacity[d].checked_sub(w.used_at(d))?;
            // Every job counts for at least `per_task`, and for one at least where a default
            // demand fills in its zeros, so less room than that admits nothing.
            let least = w.amounts.per_task[d].max((r.default_demand > 0) as u64);
            if room < least {
                return None;
            }
            bound[d] = room;
        }
        Some(Amounts::new(w.resources, bound))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, DEVICE_MEMORY, MEMORY, ResourceUnit, SLOTS};

    /// `bytes` of memory.
    fn mem(bytes: u64) -> Resources {
        Resources::new().with(MEMORY, bytes)
    }

    /// A worker state with the given capacity and reported usage.
    fn worker(slots: u64, capacity: u64, used: u64, baseline: u64) -> WorkerState {
        WorkerState {
            id: 1,
            capacity: mem(capacity).with(SLOTS, slots),
            reported_used: mem(used),
            reported_baseline: mem(baseline),
            ..Default::default()
        }
    }

    /// The rule, its escape hatch and its bound, at the boundaries.
    #[test]
    fn production_rule() {
        let a = ProductionAdmission;
        let decl = Config::default().resources;
        let s = worker(2, 100, 30, 10);
        let placed = |running, bytes| mem(bytes).with(SLOTS, running);
        let (empty, half, little, full) = (placed(0, 0), placed(1, 50), placed(1, 5), placed(2, 0));
        let view =
            |placed: &Resources| WorkerView::new(&decl, &s, placed, placed.get(SLOTS) as usize);
        let job = |v: &WorkerView, bytes| ProductionAdmission.admits(&v.demand(&mem(bytes)), v);
        // Escape hatch: alone, anything goes, even beyond the capacity.
        assert!(job(&view(&empty), 1000));
        // max(30, 10 + 50) + 40 = 100 <= 100.
        assert!(job(&view(&half), 40));
        assert!(!job(&view(&half), 41));
        // Reported usage dominates: max(30, 10 + 5) + 70 = 100.
        assert!(job(&view(&little), 70));
        assert!(!job(&view(&little), 71));
        // Slots full: no escape hatch for a hard resource.
        assert!(!job(&view(&full), 0));
        assert_eq!(a.bound(&view(&full)), None);
        let b = a.bound(&view(&half)).unwrap();
        assert_eq!(b.values(), [40, u64::MAX, 1]);
        let headroom: Vec<_> = view(&half).resources().map(|u| u.headroom).collect();
        assert_eq!(headroom, [Some(40), None, Some(1)]);
    }

    /// `baseline_excl` (the rolling RSS floor minus the estimates running) as `reported_baseline`
    /// gives `max(rss, baseline_excl + Σ placed) + demand <= capacity`: the floor does not count
    /// the running jobs twice.
    #[test]
    fn baseline_excl_removes_the_double_count() {
        let decl = Config::default().resources;
        // RSS 60 with 40 of estimates running; the floor (50) contains those jobs.
        let (rss, floor, placed, capacity) = (60, 50, 40, 100);
        let placed = mem(placed).with(SLOTS, 4);
        let admits = |s: &WorkerState, bytes| {
            let v = WorkerView::new(&decl, s, &placed, 4);
            ProductionAdmission.admits(&v.demand(&mem(bytes)), &v)
        };
        let with_floor = worker(16, capacity, rss, floor);
        let with_excl = worker(16, capacity, rss, floor - 40);
        // Floor: max(60, 50 + 40) + 15 = 105 > 100. Excl: max(60, 10 + 40) + 40 = 100.
        assert!(!admits(&with_floor, 15));
        assert!(admits(&with_excl, 40));
        assert!(!admits(&with_excl, 41));
        // RSS stays the safety term: estimates that undercount cannot admit past it.
        let undercount = worker(16, capacity, 95, 0);
        assert!(!admits(&undercount, 6));
    }

    /// Every resource follows the same inequality, with `per_task` as a per-job floor; a zero
    /// capacity of a soft resource is not enforced, host memory included.
    #[test]
    fn one_rule_per_resource() {
        let a = ProductionAdmission;
        let decl = Config::default().resources;
        let s = WorkerState {
            id: 1,
            capacity: mem(100).with(DEVICE_MEMORY, 100).with(SLOTS, 8),
            per_task: mem(20).with(DEVICE_MEMORY, 30),
            ..Default::default()
        };
        let view =
            |placed: &Resources| WorkerView::new(&decl, &s, placed, placed.get(SLOTS) as usize);
        // Device: max(10, 2 * 30) + max(5, 30) = 90 <= 100, and 60 + 41 > 100.
        let v = view(&mem(40).with(DEVICE_MEMORY, 10).with(SLOTS, 2));
        assert!(a.admits(&v.demand(&mem(10).with(DEVICE_MEMORY, 5)), &v));
        assert!(!a.admits(&v.demand(&mem(10).with(DEVICE_MEMORY, 41)), &v));
        assert_eq!(v.short(&v.demand(&mem(70))).collect::<Vec<_>>(), [&MEMORY]);
        let v = view(&mem(40).with(SLOTS, 3));
        assert_eq!(
            v.short(&v.demand(&Resources::new())).collect::<Vec<_>>(),
            [&DEVICE_MEMORY]
        );
        // Host: the floor counts 3 * 20 = 60 against the 40 placed.
        let headroom: Vec<_> = v.resources().map(|u| u.headroom).collect();
        assert_eq!(headroom, [Some(40), Some(10), Some(5)]);
        assert_eq!(a.bound(&v), None);
        // A zero memory capacity leaves its resource unenforced; slots stay enforced.
        let free = WorkerState {
            id: 2,
            capacity: Resources::new().with(SLOTS, 8),
            ..Default::default()
        };
        let v = WorkerView::new(&decl, &free, &mem(1 << 40).with(SLOTS, 7), 7);
        let mut huge = v.unbounded();
        huge.set(SLOTS, 1);
        assert!(a.admits(&huge, &v));
        assert!(!a.admits(&v.unbounded(), &v));
        let headroom: Vec<_> = v.resources().map(|u| u.headroom).collect();
        assert_eq!(headroom, [None, None, Some(1)]);
        assert_eq!(v.free_share(&huge), f64::INFINITY);
    }

    /// The scalar comparison is the bottleneck soft resource's free fraction; slots do not
    /// enter it.
    #[test]
    fn free_share_is_the_bottleneck() {
        let decl = Config::default().resources;
        let s = WorkerState {
            id: 1,
            capacity: mem(100).with(DEVICE_MEMORY, 10).with(SLOTS, 2),
            ..Default::default()
        };
        let v = WorkerView::new(&decl, &s, &mem(50).with(DEVICE_MEMORY, 2).with(SLOTS, 1), 1);
        // Host 1 - 60/100 = 0.4, device 1 - 4/10 = 0.6; slots would be 1 - 2/2 = 0.
        let share = v.free_share(&v.demand(&mem(10).with(DEVICE_MEMORY, 2)));
        assert!((share - 0.4).abs() < 1e-12);
    }

    /// Slots are hard: without a free one, not even the escape hatch admits.
    #[test]
    fn zero_slots_admit_nothing() {
        let decl = Config::default().resources;
        let s = worker(0, 100, 0, 0);
        let v = WorkerView::new(&decl, &s, &Resources::new(), 0);
        assert!(!ProductionAdmission.admits(&v.demand(&Resources::new()), &v));
        assert_eq!(ProductionAdmission.bound(&v), None);
    }

    /// A declaration of its own: the rule reads hardness and default demands from it, whatever
    /// the constants the amounts were built with say.
    #[test]
    fn custom_declaration() {
        const GPUS: Resource = Resource::new("gpus").hard();
        const SCRATCH: Resource = Resource::new("scratch")
            .default_demand(5)
            .unit(ResourceUnit::Bytes);
        let decl = [GPUS, SCRATCH];
        let s = WorkerState {
            id: 1,
            capacity: Resources::new().with(GPUS, 2).with(SCRATCH, 10),
            ..Default::default()
        };
        let scratch = |x| Resources::new().with(SCRATCH, x);
        let busy = scratch(5).with(GPUS, 2);
        let view = |placed: &Resources, running| WorkerView::new(&decl, &s, placed, running);
        let a = ProductionAdmission;
        let admits = |demand: &Resources, v: &WorkerView| a.admits(&v.demand(demand), v);
        let empty = view(&Resources::new(), 0);
        // Hard: three GPUs never fit, not even alone.
        assert!(!admits(&Resources::new().with(GPUS, 3), &empty));
        assert!(admits(&Resources::new().with(GPUS, 2), &empty));
        // Soft: alone, any amount of scratch goes; beside another job, 5 + 6 > 10 does not.
        assert!(admits(&scratch(50), &empty));
        assert!(!admits(&scratch(6), &view(&scratch(5), 1)));
        // The same name with other rules is the same resource, under the declared rules: a soft
        // "gpus" constant does not make the escape hatch apply.
        let soft_gpus = Resource::new("gpus");
        assert!(!admits(&Resources::new().with(soft_gpus, 3), &empty));
        // A job without GPUs fits beside two GPUs' worth.
        let v = view(&busy, 1);
        assert!(admits(&Resources::new(), &v));
        let two = v.demand(&Resources::new().with(GPUS, 2));
        assert_eq!(v.short(&two).collect::<Vec<_>>(), [&GPUS]);
        // The bound: no GPU left, 5 of scratch. With a default demand every job takes some
        // scratch, so a worker with none left admits nothing.
        let b = a.bound(&v).unwrap();
        assert_eq!((b.get(GPUS), b.get(SCRATCH)), (0, 5));
        assert_eq!(a.bound(&view(&scratch(10), 1)), None);
    }

    /// A worker state naming an undeclared resource is a caller bug.
    #[test]
    #[should_panic(expected = "worker 1 reports capacity of resource \"gpus\"")]
    fn undeclared_capacity_panics() {
        let s = WorkerState {
            id: 1,
            capacity: Resources::new().with(Resource::new("gpus"), 1),
            ..Default::default()
        };
        WorkerAmounts::new(&Config::default().resources, &s);
    }
}
