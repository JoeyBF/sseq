//! The [`Scheduler`](crate::Scheduler)'s configuration.
//!
//! A [`Config`] is plain data. Its parts, in the order a poll uses them:
//!
//! - [`Config::order`] (with [`GroupOrder`] and [`Config::default_priority`]) ranks the waiting
//!   jobs, as a list of [`OrderTerm`]s; [`Config::age_limit`] lets jobs that waited long jump
//!   that order.
//! - [`Config::score`] ranks the workers that admit a job, as a list of [`ScoreTerm`]s.
//! - [`Config::reservations`] ([`Reservations`]) drains a worker for a job that fits nowhere.
//! - [`Config::speed`] ([`SpeedConfig`]) chooses the machine model ([`Timing`]), and whether a job
//!   may wait for a faster busy worker ([`Defer`]) or run twice ([`Speculate`]).
//! - [`Config::retry`] ([`RetryConfig`]) bounds the retries of failed attempts.
//!
//! The presets on [`Config`] cover the common objectives; anything else is a preset with some
//! fields replaced:
//!
//! ```
//! use whelm::{Config, OrderTerm, Reservations, Scheduler};
//!
//! // Best fit, with the DAG layer's ranks breaking ties between equal priorities, and two
//! // reservations at a time.
//! let config = Config {
//!     order: vec![OrderTerm::Priority, OrderTerm::Rank, OrderTerm::Group],
//!     reservations: Some(Reservations {
//!         max: 2,
//!         ..Reservations::default()
//!     }),
//!     ..Config::best_fit()
//! };
//! assert_eq!(config.score, Config::best_fit().score);
//! let scheduler = Scheduler::new(config);
//! # let _ = scheduler;
//! ```

use std::time::Duration;

use crate::Timing;

/// How a [`Scheduler`](crate::Scheduler) behaves: plain data, with presets.
///
/// A configuration is a list-scheduling rule: [`order`](Self::order) says which waiting job goes
/// first and [`score`](Self::score) which of the workers that admit it it goes to. The presets map
/// objectives to rules. On one machine some of these rules are optimal (Smith's rule for weighted
/// completion time, Jackson's rule for maximum lateness); on many machines, with resources and
/// online arrivals, every one of them is a heuristic.
///
/// - [`Default`]: makespan with bounded latency.
/// - [`Config::fifo`]: a baseline.
/// - [`Config::best_fit`]: makespan, packing each job into the tightest worker.
/// - [`Config::weighted_completion`]: weighted completion time.
/// - [`Config::lateness`]: maximum lateness.
///
/// # Example
///
/// The same four jobs, waiting for one single-slot worker, run in a different order under each
/// preset. Job 3 has an explicit priority, which every preset but [`fifo`](Config::fifo)
/// honours first; after it, the default runs group 5 (the older group) before group 6,
/// [`weighted_completion`](Config::weighted_completion) the shortest jobs first, and
/// [`lateness`](Config::lateness) the earliest due dates first. (`run_order`, hidden, submits the
/// jobs, then runs them one by one on the worker.)
///
/// ```
/// # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Scheduler, Time, WorkerState};
/// # /// The order a one-slot worker runs `jobs` in, all submitted before it joins.
/// # fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
/// #     let mut s = Scheduler::new(config);
/// #     for j in jobs {
/// #         s.handle(Input::Submit(j), Time::ORIGIN);
/// #     }
/// #     s.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
/// #     let (mut order, mut t) = (Vec::new(), Time::ORIGIN);
/// #     while let [Output::Start { job, attempt, .. }] = s.poll(t)[..] {
/// #         order.push(job);
/// #         t += std::time::Duration::from_secs(1);
/// #         s.handle(Input::Done { job, attempt }, t);
/// #     }
/// #     order
/// # }
/// use std::time::Duration;
///
/// let spec = |id, group, work, due: Option<u64>| JobSpec {
///     id,
///     group,
///     work: Some(Duration::from_secs(work)),
///     due: due.map(|s| Time(Duration::from_secs(s))),
///     ..Default::default()
/// };
/// let jobs = || {
///     vec![
///         spec(0, 5, 100, Some(50)),
///         spec(1, 6, 10, Some(30)),
///         spec(2, 5, 50, None),
///         JobSpec {
///             priority: Some(-1),
///             ..spec(3, 6, 20, Some(10))
///         },
///     ]
/// };
/// assert_eq!(run_order(Config::fifo(), jobs()), [0, 1, 2, 3]);
/// assert_eq!(run_order(Config::default(), jobs()), [3, 0, 2, 1]);
/// assert_eq!(
///     run_order(Config::weighted_completion(), jobs()),
///     [3, 1, 2, 0]
/// );
/// assert_eq!(run_order(Config::lateness(), jobs()), [3, 1, 0, 2]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// The order waiting jobs are considered in: lexicographic over these terms, then submission
    /// order. A term listed twice adds nothing; the repeat is ignored.
    pub order: Vec<OrderTerm>,
    /// How [`OrderTerm::Group`] orders groups.
    pub group_order: GroupOrder,
    /// The priority of jobs whose [`JobSpec::priority`](crate::JobSpec::priority) is `None`, for
    /// [`OrderTerm::Priority`]: priorities below it jump ahead of unprioritised jobs, and those
    /// above fall behind them.
    pub default_priority: i64,
    /// Aging: a job that has waited at least this long becomes more urgent than every
    /// job that has not, oldest first, whatever [`order`](Self::order) says. Strict priority
    /// starves a job for as long as more urgent jobs keep arriving (a young group behind a wide
    /// old one), and this bounds it. Shorter bounds the worst wait more tightly but overrides
    /// `order` for more jobs. Default [`DEFAULT_AGE_LIMIT`]; `None` is strict priority.
    pub age_limit: Option<Duration>,
    /// Workers drained for starving jobs. `None` allows starvation of jobs larger than the
    /// typical headroom.
    pub reservations: Option<Reservations>,
    /// Which of the workers that admit a job it goes to: lexicographic over these terms, then the
    /// smallest worker id. A term listed twice adds nothing; the repeat is ignored.
    pub score: Vec<ScoreTerm>,
    /// The machine model, deferral and speculation.
    pub speed: SpeedConfig,
    /// Retries of failed attempts.
    pub retry: RetryConfig,
}

impl Default for Config {
    /// Makespan with bounded latency: explicit priority, then group, then arrival; aging and a
    /// reservation against starvation; the fastest, preferred, then least loaded worker.
    ///
    /// Group 7 arrives first, so its jobs run before group 3's, except the one with an explicit
    /// priority below [`default_priority`](Config::default_priority):
    ///
    /// ```
    /// # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Scheduler, Time, WorkerState};
    /// # /// The order a one-slot worker runs `jobs` in, all submitted before it joins.
    /// # fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
    /// #     let mut s = Scheduler::new(config);
    /// #     for j in jobs {
    /// #         s.handle(Input::Submit(j), Time::ORIGIN);
    /// #     }
    /// #     s.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
    /// #     let (mut order, mut t) = (Vec::new(), Time::ORIGIN);
    /// #     while let [Output::Start { job, attempt, .. }] = s.poll(t)[..] {
    /// #         order.push(job);
    /// #         t += std::time::Duration::from_secs(1);
    /// #         s.handle(Input::Done { job, attempt }, t);
    /// #     }
    /// #     order
    /// # }
    /// let job = |id, group| JobSpec {
    ///     id,
    ///     group,
    ///     ..Default::default()
    /// };
    /// let urgent = JobSpec {
    ///     priority: Some(-1),
    ///     ..job(13, 3)
    /// };
    /// let jobs = vec![job(10, 7), job(11, 3), job(12, 7), urgent];
    /// assert_eq!(run_order(Config::default(), jobs), [13, 10, 12, 11]);
    /// ```
    fn default() -> Self {
        Self {
            order: vec![OrderTerm::Priority, OrderTerm::Group],
            group_order: GroupOrder::Arrival,
            default_priority: 0,
            age_limit: Some(DEFAULT_AGE_LIMIT),
            reservations: Some(Reservations::default()),
            score: vec![ScoreTerm::Speed, ScoreTerm::Preferred, ScoreTerm::Load],
            speed: SpeedConfig::default(),
            retry: RetryConfig::default(),
        }
    }
}

impl Config {
    /// Arrival order, no aging, no reservations, score `[Preferred, Load]`: each job takes any
    /// worker that admits it, and jobs larger than the typical headroom starve. A baseline.
    ///
    /// Priorities and groups are ignored:
    ///
    /// ```
    /// # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Scheduler, Time, WorkerState};
    /// # /// The order a one-slot worker runs `jobs` in, all submitted before it joins.
    /// # fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
    /// #     let mut s = Scheduler::new(config);
    /// #     for j in jobs {
    /// #         s.handle(Input::Submit(j), Time::ORIGIN);
    /// #     }
    /// #     s.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
    /// #     let (mut order, mut t) = (Vec::new(), Time::ORIGIN);
    /// #     while let [Output::Start { job, attempt, .. }] = s.poll(t)[..] {
    /// #         order.push(job);
    /// #         t += std::time::Duration::from_secs(1);
    /// #         s.handle(Input::Done { job, attempt }, t);
    /// #     }
    /// #     order
    /// # }
    /// let jobs = || {
    ///     vec![
    ///         JobSpec {
    ///             id: 0,
    ///             group: 9,
    ///             ..Default::default()
    ///         },
    ///         JobSpec {
    ///             id: 1,
    ///             group: 1,
    ///             priority: Some(-5),
    ///             ..Default::default()
    ///         },
    ///     ]
    /// };
    /// assert_eq!(run_order(Config::fifo(), jobs()), [0, 1]);
    /// assert_eq!(run_order(Config::default(), jobs()), [1, 0]);
    /// ```
    pub fn fifo() -> Self {
        Self {
            order: Vec::new(),
            age_limit: None,
            reservations: None,
            score: vec![ScoreTerm::Preferred, ScoreTerm::Load],
            ..Self::default()
        }
    }

    /// The default with score `[Speed, Tightest, Preferred, Load]`: packs small jobs tightly and
    /// keeps big holes open.
    ///
    /// The order is the default's; placement differs. Of two empty workers, a small job goes to
    /// the smaller one, which it fills more, leaving the large one whole for a large job; the
    /// default's least-loaded rule breaks the tie by worker id instead.
    ///
    /// ```
    /// use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, Time, WorkerState};
    ///
    /// let place = |config| {
    ///     let mut s = Scheduler::new(config);
    ///     s.handle(
    ///         Input::Worker(WorkerState {
    ///             id: 1,
    ///             slots: 4,
    ///             budget: Resources::mem(100),
    ///             ..Default::default()
    ///         }),
    ///         Time::ORIGIN,
    ///     );
    ///     s.handle(
    ///         Input::Worker(WorkerState {
    ///             id: 2,
    ///             slots: 4,
    ///             budget: Resources::mem(50),
    ///             ..Default::default()
    ///         }),
    ///         Time::ORIGIN,
    ///     );
    ///     s.handle(
    ///         Input::Submit(JobSpec {
    ///             id: 0,
    ///             demand: Resources::mem(1),
    ///             ..Default::default()
    ///         }),
    ///         Time::ORIGIN,
    ///     );
    ///     s.poll(Time::ORIGIN)
    /// };
    /// let on = |worker| {
    ///     vec![Output::Start {
    ///         job: 0,
    ///         attempt: 1,
    ///         worker,
    ///     }]
    /// };
    /// assert_eq!(place(Config::best_fit()), on(2));
    /// assert_eq!(place(Config::default()), on(1));
    /// ```
    pub fn best_fit() -> Self {
        Self {
            score: vec![
                ScoreTerm::Speed,
                ScoreTerm::Tightest,
                ScoreTerm::Preferred,
                ScoreTerm::Load,
            ],
            ..Self::default()
        }
    }

    /// The default with order `[Priority, Wspt]`, for the sum of weighted completion times:
    /// Smith's rule, optimal on one machine.
    ///
    /// Largest [`weight`](crate::JobSpec::weight) over [`work`](crate::JobSpec::work) first; jobs
    /// without a work estimate last, in arrival order:
    ///
    /// ```
    /// # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Scheduler, Time, WorkerState};
    /// # /// The order a one-slot worker runs `jobs` in, all submitted before it joins.
    /// # fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
    /// #     let mut s = Scheduler::new(config);
    /// #     for j in jobs {
    /// #         s.handle(Input::Submit(j), Time::ORIGIN);
    /// #     }
    /// #     s.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
    /// #     let (mut order, mut t) = (Vec::new(), Time::ORIGIN);
    /// #     while let [Output::Start { job, attempt, .. }] = s.poll(t)[..] {
    /// #         order.push(job);
    /// #         t += std::time::Duration::from_secs(1);
    /// #         s.handle(Input::Done { job, attempt }, t);
    /// #     }
    /// #     order
    /// # }
    /// use std::time::Duration;
    ///
    /// let spec = |id, weight, work: Option<u64>| JobSpec {
    ///     id,
    ///     weight,
    ///     work: work.map(Duration::from_secs),
    ///     ..Default::default()
    /// };
    /// let jobs = vec![
    ///     spec(0, 1.0, None),
    ///     spec(1, 1.0, Some(10)), // 0.1 per second
    ///     spec(2, 3.0, Some(10)), // 0.3
    ///     spec(3, 1.0, Some(2)),  // 0.5
    ///     spec(4, 1.0, None),
    /// ];
    /// assert_eq!(
    ///     run_order(Config::weighted_completion(), jobs),
    ///     [3, 2, 1, 0, 4]
    /// );
    /// ```
    pub fn weighted_completion() -> Self {
        Self {
            order: vec![OrderTerm::Priority, OrderTerm::Wspt],
            ..Self::default()
        }
    }

    /// The default with order `[Priority, Edd]`, for maximum lateness: Jackson's rule, optimal on
    /// one machine.
    ///
    /// Earliest [`due`](crate::JobSpec::due) date first; jobs without one last:
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Scheduler, Time, WorkerState};
    /// # /// The order a one-slot worker runs `jobs` in, all submitted before it joins.
    /// # fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
    /// #     let mut s = Scheduler::new(config);
    /// #     for j in jobs {
    /// #         s.handle(Input::Submit(j), Time::ORIGIN);
    /// #     }
    /// #     s.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
    /// #     let (mut order, mut t) = (Vec::new(), Time::ORIGIN);
    /// #     while let [Output::Start { job, attempt, .. }] = s.poll(t)[..] {
    /// #         order.push(job);
    /// #         t += Duration::from_secs(1);
    /// #         s.handle(Input::Done { job, attempt }, t);
    /// #     }
    /// #     order
    /// # }
    /// let spec = |id, due| JobSpec {
    ///     id,
    ///     due,
    ///     ..Default::default()
    /// };
    /// let jobs = vec![
    ///     spec(0, None),
    ///     spec(1, Some(Time(Duration::from_secs(50)))),
    ///     spec(2, Some(Time(Duration::from_secs(3)))),
    /// ];
    /// assert_eq!(run_order(Config::lateness(), jobs), [2, 1, 0]);
    /// ```
    pub fn lateness() -> Self {
        Self {
            order: vec![OrderTerm::Priority, OrderTerm::Edd],
            ..Self::default()
        }
    }
}

/// [`Config::age_limit`]'s default. Shorter bounds the worst wait more tightly but lets
/// aged FIFO override [`Config::order`] for more jobs; its effect on the trace replay's waits and
/// throughput is in `whelm-sim`'s RESULTS.md, "Headline".
///
/// # Example
///
/// A low-priority job waits behind a busy worker; a more urgent one arrives later. Once the first
/// has waited `DEFAULT_AGE_LIMIT`, it goes first; without aging the urgent one does.
/// Reservations are off, since the waiting job would otherwise reserve the worker first.
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{
///     Config, DEFAULT_AGE_LIMIT, Input, JobSpec, Output, Policy, Scheduler, Time, WorkerState,
/// };
///
/// let next = |age_limit| {
///     let mut s = Scheduler::new(Config {
///         age_limit,
///         reservations: None,
///         ..Config::default()
///     });
///     s.handle(
///         Input::Worker(WorkerState {
///             id: 1,
///             ..Default::default()
///         }),
///         Time::ORIGIN,
///     );
///     s.handle(
///         Input::Submit(JobSpec {
///             id: 0,
///             ..Default::default()
///         }),
///         Time::ORIGIN,
///     );
///     s.poll(Time::ORIGIN); // job 0 takes the slot
///     let low = JobSpec {
///         id: 1,
///         priority: Some(5),
///         ..Default::default()
///     };
///     s.handle(Input::Submit(low), Time::ORIGIN);
///     // Aging is timed: the scheduler asks to be polled when job 1 ages.
///     assert_eq!(s.next_wakeup(), age_limit.map(|a| Time::ORIGIN + a));
///     let urgent = JobSpec {
///         id: 2,
///         ..Default::default()
///     };
///     let aged = Time::ORIGIN + DEFAULT_AGE_LIMIT;
///     s.handle(Input::Submit(urgent), aged - Duration::from_secs(1));
///     s.handle(Input::Done { job: 0, attempt: 1 }, aged);
///     match s.poll(aged)[..] {
///         [Output::Start { job, .. }] => job,
///         ref out => panic!("{out:?}"),
///     }
/// };
/// assert_eq!(next(Some(DEFAULT_AGE_LIMIT)), 1);
/// assert_eq!(next(None), 2);
/// ```
pub const DEFAULT_AGE_LIMIT: Duration = Duration::from_secs(1800);

/// One term of [`Config::order`]. Every key is computed once, at submission; a job that lacks
/// what a term reads sorts after every job that has it, within that term.
///
/// Terms compare lexicographically, so their order in the list matters. With the same jobs,
/// group first runs the older group 5 before anything else, and rank first runs the longest
/// chain (job 2) first:
///
/// ```
/// # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Scheduler, Time, WorkerState};
/// # /// The order a one-slot worker runs `jobs` in, all submitted before it joins.
/// # fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
/// #     let mut s = Scheduler::new(config);
/// #     for j in jobs {
/// #         s.handle(Input::Submit(j), Time::ORIGIN);
/// #     }
/// #     s.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
/// #     let (mut order, mut t) = (Vec::new(), Time::ORIGIN);
/// #     while let [Output::Start { job, attempt, .. }] = s.poll(t)[..] {
/// #         order.push(job);
/// #         t += std::time::Duration::from_secs(1);
/// #         s.handle(Input::Done { job, attempt }, t);
/// #     }
/// #     order
/// # }
/// use std::time::Duration;
///
/// use whelm::OrderTerm::{Group, Rank};
///
/// let spec = |id, group, rank| JobSpec {
///     id,
///     group,
///     rank,
///     ..Default::default()
/// };
/// let jobs = || {
///     vec![
///         spec(0, 5, None),
///         spec(1, 6, Some(Duration::from_secs(2))),
///         spec(2, 6, Some(Duration::from_secs(9))),
///     ]
/// };
/// let order = |order| Config {
///     order,
///     ..Config::default()
/// };
/// assert_eq!(run_order(order(vec![Group, Rank]), jobs()), [0, 2, 1]);
/// assert_eq!(run_order(order(vec![Rank, Group]), jobs()), [2, 1, 0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrderTerm {
    /// [`JobSpec::priority`](crate::JobSpec::priority), smallest first; unset counts as
    /// [`Config::default_priority`].
    ///
    /// Raising the default priority moves a job of priority 1 from behind the unprioritised job
    /// to ahead of it:
    ///
    /// ```
    /// # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Scheduler, Time, WorkerState};
    /// # /// The order a one-slot worker runs `jobs` in, all submitted before it joins.
    /// # fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
    /// #     let mut s = Scheduler::new(config);
    /// #     for j in jobs {
    /// #         s.handle(Input::Submit(j), Time::ORIGIN);
    /// #     }
    /// #     s.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
    /// #     let (mut order, mut t) = (Vec::new(), Time::ORIGIN);
    /// #     while let [Output::Start { job, attempt, .. }] = s.poll(t)[..] {
    /// #         order.push(job);
    /// #         t += std::time::Duration::from_secs(1);
    /// #         s.handle(Input::Done { job, attempt }, t);
    /// #     }
    /// #     order
    /// # }
    /// let jobs = || {
    ///     vec![
    ///         JobSpec {
    ///             id: 0,
    ///             ..Default::default()
    ///         },
    ///         JobSpec {
    ///             id: 1,
    ///             priority: Some(1),
    ///             ..Default::default()
    ///         },
    ///     ]
    /// };
    /// let default_priority = |default_priority| Config {
    ///     default_priority,
    ///     ..Config::default()
    /// };
    /// assert_eq!(run_order(default_priority(0), jobs()), [0, 1]);
    /// assert_eq!(run_order(default_priority(2), jobs()), [1, 0]);
    /// ```
    Priority,
    /// [`JobSpec::rank`](crate::JobSpec::rank), largest first: the longest remaining chain first,
    /// as in HEFT. Set by the DAG layer; unset sorts last.
    Rank,
    /// [`JobSpec::group`](crate::JobSpec::group), in [`Config::group_order`] ([`GroupOrder`] has
    /// an example).
    Group,
    /// Weighted shortest processing time, Smith's rule: largest
    /// [`weight`](crate::JobSpec::weight) over [`work`](crate::JobSpec::work) first. Jobs without
    /// a work estimate sort last. [`Config::weighted_completion`] has an example.
    Wspt,
    /// Earliest due date, Jackson's rule: smallest [`due`](crate::JobSpec::due) first. Jobs
    /// without one sort last. [`Config::lateness`] has an example.
    Edd,
}

/// One term of [`Config::score`], ranking the workers that admit a job.
///
/// Like [`OrderTerm`]s, terms compare lexicographically: an earlier term decides, and a later
/// one only breaks its ties. The examples on the variants place one job on a set of workers with
/// a hidden helper, `first_worker(score, workers, job)`, which returns where it went.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScoreTerm {
    /// The fastest for the job first, as [`SpeedConfig::timing`] has it: under
    /// [`Timing::Unrelated`] a worker can rank first for one kind of job and last for another.
    /// With [`Learn`](crate::Learn) and a resolution, speeds within one resolution step of each
    /// other tie, so per-worker noise does not override the later terms.
    ///
    /// A worker reporting twice the speed wins over a lower id:
    ///
    /// ```
    /// # use whelm::{
    /// #     Config, Input, JobSpec, Output, Policy, Scheduler, ScoreTerm, Time, WorkerId,
    /// #     WorkerState,
    /// # };
    /// # /// Where `job` goes among `workers`, ranked by `score`.
    /// # fn first_worker(
    /// #     score: Vec<ScoreTerm>,
    /// #     workers: Vec<WorkerState>,
    /// #     job: JobSpec,
    /// # ) -> WorkerId {
    /// #     let mut s = Scheduler::new(Config { score, ..Config::default() });
    /// #     for w in workers {
    /// #         s.handle(Input::Worker(w), Time::ORIGIN);
    /// #     }
    /// #     s.handle(Input::Submit(job), Time::ORIGIN);
    /// #     match s.poll(Time::ORIGIN)[..] {
    /// #         [Output::Start { worker, .. }] => worker,
    /// #         ref out => panic!("{out:?}"),
    /// #     }
    /// # }
    /// let fast = WorkerState {
    ///     id: 2,
    ///     slots: 4,
    ///     speed: 2.0,
    ///     ..Default::default()
    /// };
    /// let workers = || {
    ///     vec![
    ///         WorkerState {
    ///             id: 1,
    ///             slots: 4,
    ///             ..Default::default()
    ///         },
    ///         fast.clone(),
    ///     ]
    /// };
    /// let job = || JobSpec {
    ///     id: 0,
    ///     ..Default::default()
    /// };
    /// assert_eq!(first_worker(vec![ScoreTerm::Speed], workers(), job()), 2);
    /// assert_eq!(first_worker(vec![ScoreTerm::Load], workers(), job()), 1);
    /// ```
    Speed,
    /// The tightest fit: the smallest [`WorkerView::free_share`](crate::WorkerView::free_share)
    /// after placement. [`Config::best_fit`] has an example.
    Tightest,
    /// The loosest fit: the largest free share after placement.
    ///
    /// ```
    /// # use whelm::{
    /// #     Config, Input, JobSpec, Output, Policy, Resources, Scheduler, ScoreTerm, Time,
    /// #     WorkerId, WorkerState,
    /// # };
    /// # /// Where `job` goes among `workers`, ranked by `score`.
    /// # fn first_worker(
    /// #     score: Vec<ScoreTerm>,
    /// #     workers: Vec<WorkerState>,
    /// #     job: JobSpec,
    /// # ) -> WorkerId {
    /// #     let mut s = Scheduler::new(Config { score, ..Config::default() });
    /// #     for w in workers {
    /// #         s.handle(Input::Worker(w), Time::ORIGIN);
    /// #     }
    /// #     s.handle(Input::Submit(job), Time::ORIGIN);
    /// #     match s.poll(Time::ORIGIN)[..] {
    /// #         [Output::Start { worker, .. }] => worker,
    /// #         ref out => panic!("{out:?}"),
    /// #     }
    /// # }
    /// let workers = vec![
    ///     WorkerState {
    ///         id: 1,
    ///         slots: 4,
    ///         budget: Resources::mem(50),
    ///         ..Default::default()
    ///     },
    ///     WorkerState {
    ///         id: 2,
    ///         slots: 4,
    ///         budget: Resources::mem(100),
    ///         ..Default::default()
    ///     },
    /// ];
    /// let job = JobSpec {
    ///     id: 0,
    ///     demand: Resources::mem(10),
    ///     ..Default::default()
    /// };
    /// assert_eq!(first_worker(vec![ScoreTerm::Loosest], workers, job), 2);
    /// ```
    Loosest,
    /// Workers a [`Strength::Prefer`](crate::Strength::Prefer) constraint selects first.
    ///
    /// ```
    /// # use whelm::{
    /// #     Config, Constraint, Input, JobSpec, Output, Policy, Scheduler, ScoreTerm, Time,
    /// #     WorkerId, WorkerState,
    /// # };
    /// # /// Where `job` goes among `workers`, ranked by `score`.
    /// # fn first_worker(
    /// #     score: Vec<ScoreTerm>,
    /// #     workers: Vec<WorkerState>,
    /// #     job: JobSpec,
    /// # ) -> WorkerId {
    /// #     let mut s = Scheduler::new(Config { score, ..Config::default() });
    /// #     for w in workers {
    /// #         s.handle(Input::Worker(w), Time::ORIGIN);
    /// #     }
    /// #     s.handle(Input::Submit(job), Time::ORIGIN);
    /// #     match s.poll(Time::ORIGIN)[..] {
    /// #         [Output::Start { worker, .. }] => worker,
    /// #         ref out => panic!("{out:?}"),
    /// #     }
    /// # }
    /// let workers = vec![
    ///     WorkerState {
    ///         id: 1,
    ///         slots: 4,
    ///         ..Default::default()
    ///     },
    ///     WorkerState {
    ///         id: 2,
    ///         slots: 4,
    ///         ..Default::default()
    ///     },
    /// ];
    /// let job = JobSpec {
    ///     id: 0,
    ///     constraints: vec![Constraint::prefer_worker(2)],
    ///     ..Default::default()
    /// };
    /// assert_eq!(first_worker(vec![ScoreTerm::Preferred], workers, job), 2);
    /// ```
    Preferred,
    /// The fewest live attempts.
    ///
    /// Jobs spread over equal workers:
    ///
    /// ```
    /// use whelm::{Config, Input, JobSpec, Output, Policy, Scheduler, ScoreTerm, Time, WorkerState};
    ///
    /// let mut s = Scheduler::new(Config {
    ///     score: vec![ScoreTerm::Load],
    ///     ..Config::default()
    /// });
    /// for w in [1, 2] {
    ///     s.handle(
    ///         Input::Worker(WorkerState {
    ///             id: w,
    ///             slots: 4,
    ///             ..Default::default()
    ///         }),
    ///         Time::ORIGIN,
    ///     );
    /// }
    /// for id in 0..3 {
    ///     s.handle(
    ///         Input::Submit(JobSpec {
    ///             id,
    ///             ..Default::default()
    ///         }),
    ///         Time::ORIGIN,
    ///     );
    /// }
    /// let workers: Vec<_> = (s.poll(Time::ORIGIN).into_iter())
    ///     .map(|o| match o {
    ///         Output::Start { worker, .. } => worker,
    ///         _ => unreachable!(),
    ///     })
    ///     .collect();
    /// assert_eq!(workers, [1, 2, 1]);
    /// ```
    Load,
}

/// How [`OrderTerm::Group`] orders [`JobSpec::group`](crate::JobSpec::group)s.
///
/// Group 9 is submitted before group 2:
///
/// ```
/// # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Scheduler, Time, WorkerState};
/// # /// The order a one-slot worker runs `jobs` in, all submitted before it joins.
/// # fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
/// #     let mut s = Scheduler::new(config);
/// #     for j in jobs {
/// #         s.handle(Input::Submit(j), Time::ORIGIN);
/// #     }
/// #     s.handle(Input::Worker(WorkerState { id: 1, ..Default::default() }), Time::ORIGIN);
/// #     let (mut order, mut t) = (Vec::new(), Time::ORIGIN);
/// #     while let [Output::Start { job, attempt, .. }] = s.poll(t)[..] {
/// #         order.push(job);
/// #         t += std::time::Duration::from_secs(1);
/// #         s.handle(Input::Done { job, attempt }, t);
/// #     }
/// #     order
/// # }
/// use whelm::GroupOrder;
///
/// let jobs = || {
///     vec![
///         JobSpec {
///             id: 0,
///             group: 9,
///             ..Default::default()
///         },
///         JobSpec {
///             id: 1,
///             group: 2,
///             ..Default::default()
///         },
///     ]
/// };
/// let config = |group_order| Config {
///     group_order,
///     ..Config::default()
/// };
/// assert_eq!(run_order(config(GroupOrder::Arrival), jobs()), [0, 1]);
/// assert_eq!(run_order(config(GroupOrder::Id), jobs()), [1, 0]);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GroupOrder {
    /// By the group's first submission: "oldest group first". Depends on the order the caller
    /// happens to submit in, so a restarted caller that resubmits in another order reorders
    /// the groups.
    #[default]
    Arrival,
    /// By the group id itself, smallest first: restart-stable when the caller derives ids from
    /// the work (e.g. [`nassau::group`](crate::nassau::group)).
    Id,
}

/// Reservations: the most urgent job that has waited at least `reserve_after` and is admitted
/// nowhere reserves the worker with the most headroom
/// ([`WorkerView::free_share`](crate::WorkerView::free_share)), which admits no other job until
/// the holder is placed -- at the latest when the worker empties, by the escape hatch.
///
/// A reservation is released when its holder is placed (anywhere) or cancelled, or its worker
/// leaves or changes class (the holder may then reserve again). When all reservations are taken,
/// a more urgent qualifying job takes over the least urgent holder's reservation, worker included
/// (it has been draining already). Every other worker keeps admitting less urgent jobs
/// (backfill).
///
/// No starvation: the most urgent waiting job is placed within `reserve_after` plus the longest
/// running time of the jobs on the worker it reserves (nothing new is admitted there once it holds
/// the reservation, and nobody more urgent can take the reservation over).
///
/// [`Scheduler`](crate::Scheduler) shows a reservation from start to finish.
///
/// # Example
///
/// Shadow backfill. Two workers each run a 60-byte job expected to end at 100 s; a 50-byte job
/// fits on neither and reserves worker 1 once it has waited `reserve_after`, and a long job takes
/// the rest of worker 2. Then a job that would end at 81 s backfills the reserved worker, and one
/// that would end at 111 s does not.
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{
///     Config, Input, JobSpec, Output, Policy, Reservations, Resources, Scheduler, Time,
///     WorkerState,
/// };
///
/// let mut s = Scheduler::new(Config {
///     reservations: Some(Reservations {
///         shadow_backfill: true,
///         ..Reservations::default()
///     }),
///     ..Config::default()
/// });
/// let job = |id, bytes, work| JobSpec {
///     id,
///     demand: Resources::mem(bytes),
///     work: Some(Duration::from_secs(work)),
///     ..Default::default()
/// };
/// for w in [1, 2] {
///     s.handle(
///         Input::Worker(WorkerState {
///             id: w,
///             slots: 4,
///             budget: Resources::mem(100),
///             ..Default::default()
///         }),
///         Time::ORIGIN,
///     );
/// }
/// s.handle(Input::Submit(job(10, 60, 100)), Time::ORIGIN);
/// s.handle(Input::Submit(job(11, 60, 100)), Time::ORIGIN);
/// s.handle(Input::Submit(job(1, 50, 10)), Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN).len(), 2);
/// assert_eq!(s.poll(Time(Duration::from_secs(60))), []);
/// assert_eq!(s.stats().reservations[0].worker, 1);
/// s.handle(
///     Input::Submit(job(30, 40, 1_000_000)),
///     Time(Duration::from_secs(60)),
/// );
/// s.handle(Input::Submit(job(20, 5, 50)), Time(Duration::from_secs(61)));
/// s.handle(Input::Submit(job(21, 5, 20)), Time(Duration::from_secs(61)));
/// let start = |job, worker| Output::Start {
///     job,
///     attempt: 1,
///     worker,
/// };
/// assert_eq!(
///     s.poll(Time(Duration::from_secs(61))),
///     [start(30, 2), start(21, 1)]
/// );
/// assert!(
///     s.explain(20)
///         .unwrap()
///         .contains("reserved: worker 1 for job 1")
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reservations {
    /// A job that has waited at least this long and is admitted nowhere may reserve a worker.
    /// Shorter bounds starvation more tightly but drains workers, idling their slots, more often.
    pub reserve_after: Duration,
    /// Maximum number of simultaneous reservations (per worker class if `per_class`). Zero makes
    /// no reservations. More drain more workers at once, idling more slots; `whelm-sim`'s
    /// RESULTS.md, "Headline", has their cost on the trace replay.
    pub max: usize,
    /// Count `max` per worker class instead of globally.
    pub per_class: bool,
    /// EASY-style backfill on a reserved worker: less urgent jobs may still run there if they
    /// are expected to finish before the holder could start, its *shadow time*. The shadow time is
    /// computed once per reservation, from the running jobs' expected ends, predicting usage from
    /// placed demands (heartbeat usage cannot be predicted); an unknown end means no backfill.
    /// Once the shadow time passes nothing can finish before it, so the worker drains strictly
    /// from then on: the holder waits at most for the jobs running at the shadow time. Off, the
    /// worker drains strictly from the start.
    pub shadow_backfill: bool,
}

impl Default for Reservations {
    /// One reservation at a time, drained strictly: more only add idle slot time (see
    /// [`max`](Self::max)).
    fn default() -> Self {
        Self {
            reserve_after: Duration::from_secs(60),
            max: 1,
            per_class: false,
            shadow_backfill: false,
        }
    }
}

/// When a job may wait for a busy worker faster than the one [`Config::score`] picked, instead of
/// starting there: earliest finish time with a deferral window (HEFT's processor choice, online;
/// StarPU's dmda). A job with [`JobSpec::work`](crate::JobSpec::work) waits for the full, faster
/// worker on which it is expected to finish earliest, if that beats starting now by enough.
///
/// A deferral is a hold that lapses by itself; [`Scheduler`](crate::Scheduler) shows one paying
/// off.
///
/// # Example
///
/// The fast worker (ten times the speed) is busy with a job expected to end at 20 s. A job of 100
/// s of work waits for it, expecting to finish at 30 s rather than at 100 s on the idle slow
/// worker. The fast job overruns, so at `max_wait` the waiting job gives up and starts on the
/// slow worker.
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{
///     Config, Defer, Input, JobSpec, Output, Policy, Scheduler, SpeedConfig, Time, WorkerState,
/// };
///
/// let mut s = Scheduler::new(Config {
///     speed: SpeedConfig {
///         defer: Some(Defer {
///             max_wait: Duration::from_secs(30),
///             min_gain: 0.0,
///         }),
///         ..SpeedConfig::default()
///     },
///     ..Config::default()
/// });
/// let worker = |id, speed| WorkerState {
///     id,
///     speed,
///     ..Default::default()
/// };
/// let job = |id, work| JobSpec {
///     id,
///     work: Some(Duration::from_secs(work)),
///     ..Default::default()
/// };
/// s.handle(Input::Worker(worker(1, 1.0)), Time::ORIGIN);
/// s.handle(Input::Worker(worker(2, 10.0)), Time::ORIGIN);
/// s.handle(Input::Submit(job(0, 200)), Time::ORIGIN);
/// s.poll(Time::ORIGIN); // job 0 on the fast worker
/// s.handle(Input::Submit(job(1, 100)), Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), []);
/// assert_eq!(s.stats().deferred, [(1, 2, Time(Duration::from_secs(20)))]);
/// assert_eq!(s.next_wakeup(), Some(Time(Duration::from_secs(30))));
/// assert_eq!(s.poll(Time(Duration::from_secs(29))), []);
/// assert_eq!(
///     s.poll(Time(Duration::from_secs(30))),
///     [Output::Start {
///         job: 1,
///         attempt: 1,
///         worker: 1
///     }]
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Defer {
    /// A job that has waited this long no longer defers. Bounds the extra waiting;
    /// expiry is reported by [`Policy::next_wakeup`](crate::Policy::next_wakeup).
    pub max_wait: Duration,
    /// Defer only if the expected finish improves by at least this fraction of the job's work.
    /// Lower defers more often, also for a barely faster, scarce class, where waiting backfires;
    /// higher gives up more of waiting's benefit. `whelm-sim`'s RESULTS.md, "Ordering and
    /// placement", has both effects.
    pub min_gain: f64,
}

impl Default for Defer {
    /// Bounded waiting, and only for a substantial gain (see [`min_gain`](Self::min_gain)).
    fn default() -> Self {
        Self {
            max_wait: Duration::from_secs(3600),
            min_gain: 0.25,
        }
    }
}

/// Speed-aware settings. How speed ranks workers is [`ScoreTerm::Speed`]'s place in
/// [`Config::score`].
///
/// The default trusts reported speeds and neither defers nor speculates:
///
/// ```
/// use whelm::{SpeedConfig, Timing};
///
/// let speed = SpeedConfig::default();
/// assert_eq!(speed.timing, Timing::Related { learn: None });
/// assert_eq!((speed.defer, speed.speculate), (None, None));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeedConfig {
    /// The machine model: a job's speed on each worker, reported or learned.
    pub timing: Timing,
    /// Wait for a faster busy worker when it pays.
    pub defer: Option<Defer>,
    /// Start a second attempt of a running job on a faster worker that would otherwise stay idle.
    pub speculate: Option<Speculate>,
}

/// Speculative execution (HeteroPrio's spoliation, without the kill): after a poll's placements,
/// a worker with a free slot that no waiting job took starts another attempt of the running job,
/// on slower workers only, that it would finish soonest relative to where it runs (the one with
/// the latest expected end among those that gain), if the new attempt is expected to finish at
/// least `min_gain` of its run time earlier. The original keeps running; the first to finish
/// wins and the other is stopped ([`Output::Stop`](crate::Output::Stop)). Needs
/// [`JobSpec::work`](crate::JobSpec::work).
///
/// [`Scheduler`](crate::Scheduler) shows a speculative attempt winning.
///
/// # Example
///
/// No speculation without enough gain: at 90 s a job of 100 s of work on a slow worker is
/// expected to end at 100 s, and a fresh attempt on a newly joined worker twice as fast would
/// end at 140 s.
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{
///     Config, Input, JobSpec, Policy, Scheduler, Speculate, SpeedConfig, Time, WorkerState,
/// };
///
/// let mut s = Scheduler::new(Config {
///     speed: SpeedConfig {
///         speculate: Some(Speculate::default()),
///         ..SpeedConfig::default()
///     },
///     ..Config::fifo()
/// });
/// s.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// let job = JobSpec {
///     id: 0,
///     work: Some(Duration::from_secs(100)),
///     ..Default::default()
/// };
/// s.handle(Input::Submit(job), Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN).len(), 1);
/// let fast = WorkerState {
///     id: 2,
///     speed: 2.0,
///     ..Default::default()
/// };
/// s.handle(Input::Worker(fast), Time(Duration::from_secs(90)));
/// assert_eq!(s.poll(Time(Duration::from_secs(90))), []);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Speculate {
    /// Minimum gain, as a fraction of the new attempt's run time.
    pub min_gain: f64,
    /// What a new attempt costs on top of its run time (it starts from scratch).
    pub restart_overhead: Duration,
    /// A job gets at most this many speculative attempts.
    pub max_per_job: u32,
}

impl Default for Speculate {
    /// Speculate only for a substantial gain, assuming no restart cost, with few extra attempts
    /// per job.
    fn default() -> Self {
        Self {
            min_gain: 0.25,
            restart_overhead: Duration::ZERO,
            max_per_job: 1,
        }
    }
}

/// How often a failed job is retried.
///
/// [`Scheduler`](crate::Scheduler) shows a retry. With two rounds, the second failure gives the
/// job up; it is [`retryable`](crate::GaveUp::retryable) only if every attempt ran out of device
/// memory.
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{
///     Config, FailKind, Input, JobSpec, Output, Policy, RetryConfig, Scheduler, Time, WorkerState,
/// };
///
/// let mut s = Scheduler::new(Config {
///     retry: RetryConfig { max_attempts: 2 },
///     ..Config::default()
/// });
/// s.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// s.handle(
///     Input::Submit(JobSpec {
///         id: 0,
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// s.poll(Time::ORIGIN);
/// let fail = |attempt, kind| Input::Failed {
///     job: 0,
///     attempt,
///     kind,
///     why: "boom".into(),
/// };
/// s.handle(fail(1, FailKind::DeviceOom), Time(Duration::from_secs(1)));
/// assert_eq!(
///     s.poll(Time(Duration::from_secs(1))),
///     [Output::Start {
///         job: 0,
///         attempt: 2,
///         worker: 1
///     }]
/// );
/// s.handle(fail(2, FailKind::Timeout), Time(Duration::from_secs(2)));
/// let [Output::GaveUp(gave_up)] = &s.poll(Time(Duration::from_secs(2)))[..] else {
///     panic!("expected a give-up");
/// };
/// assert_eq!(gave_up.tried.len(), 2);
/// assert!(!gave_up.retryable);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryConfig {
    /// Rounds per job before it is given up ([`Output::GaveUp`](crate::Output::GaveUp)): a round
    /// is an attempt started from the queue, with any speculative attempts made alongside it, and
    /// it fails when its last live attempt does. 0 counts as 1.
    pub max_attempts: u32,
}

impl Default for RetryConfig {
    /// A few rounds before giving up.
    fn default() -> Self {
        Self { max_attempts: 4 }
    }
}
