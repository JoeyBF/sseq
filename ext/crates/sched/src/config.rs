//! The [`Scheduler`](crate::Scheduler)'s configuration.

use crate::Learn;

/// How a [`Scheduler`](crate::Scheduler) behaves: plain data, with presets.
///
/// [`Default`] is priority order with aging, one reservation and least-loaded placement;
/// [`Config::fifo`] and [`Config::best_fit`] are the other presets. Every field is independent of
/// the others.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// The order waiting jobs are considered in. Default [`Order::default`].
    pub order: Order,
    /// Aging: a job that has waited at least this long (seconds) becomes more urgent than every
    /// job that has not, oldest first. Strict priority starves a job for as long as more urgent
    /// jobs keep arriving (a young group behind a wide old one), and this bounds it. Default
    /// [`DEFAULT_AGE_LIMIT`]; `None` is strict priority.
    pub age_limit: Option<f64>,
    /// Workers drained for starving jobs. Default [`Reservations::default`]; `None` allows
    /// starvation of jobs larger than the typical headroom.
    pub reservations: Option<Reservations>,
    /// Which of the workers that admit a job it goes to. Default [`Fit::LeastLoaded`].
    pub fit: Fit,
    /// Speed-aware placement. Default: oblivious.
    pub speed: SpeedConfig,
    /// Retries of failed attempts. Default [`RetryConfig::default`].
    pub retry: RetryConfig,
}

impl Default for Config {
    /// Priority order, [`DEFAULT_AGE_LIMIT`], one reservation, least loaded, speed-oblivious,
    /// default retries.
    fn default() -> Self {
        Self {
            order: Order::default(),
            age_limit: Some(DEFAULT_AGE_LIMIT),
            reservations: Some(Reservations::default()),
            fit: Fit::LeastLoaded,
            speed: SpeedConfig::default(),
            retry: RetryConfig::default(),
        }
    }
}

impl Config {
    /// Arrival order, no aging, no reservations, least loaded: each job takes any worker that
    /// admits it, and jobs larger than the typical headroom starve. A baseline.
    pub fn fifo() -> Self {
        Self {
            order: Order::Fifo,
            age_limit: None,
            reservations: None,
            fit: Fit::LeastLoaded,
            speed: SpeedConfig::default(),
            retry: RetryConfig::default(),
        }
    }

    /// The default with [`Fit::Tightest`] and no preference penalty: packs small jobs tightly and
    /// keeps big holes open.
    pub fn best_fit() -> Self {
        Self {
            fit: Fit::Tightest {
                prefer_penalty: 0.0,
            },
            ..Self::default()
        }
    }
}

/// [`Config::age_limit`]'s default, seconds: in the trace replay, 30 minutes cut the maximum wait
/// 8x at no throughput cost.
pub const DEFAULT_AGE_LIMIT: f64 = 1800.0;

/// The order waiting jobs are considered in (after aged jobs, see [`Config::age_limit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Submission order.
    Fifo,
    /// By [`JobSpec::priority`](crate::JobSpec::priority), then group, then submission order.
    Priority {
        /// The priority of jobs whose [`JobSpec::priority`](crate::JobSpec::priority) is `None`.
        default_priority: i64,
        /// How groups are ordered against each other.
        group_order: GroupOrder,
        /// Order by group, then by priority within the group (instead of priority first). With
        /// DAG-rank priorities this is "oldest group first, critical path within it".
        group_first: bool,
    },
}

impl Default for Order {
    /// Priority 0 by default, groups by arrival, priority before group.
    fn default() -> Self {
        Self::Priority {
            default_priority: 0,
            group_order: GroupOrder::Arrival,
            group_first: false,
        }
    }
}

/// How [`JobSpec::group`](crate::JobSpec::group)s are ordered against each other.
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reservations {
    /// A job that has waited at least this long (seconds) and is admitted nowhere may reserve a
    /// worker. Default 60.
    pub reserve_after: f64,
    /// Maximum number of simultaneous reservations (per worker class if `per_class`). Default 1.
    /// Zero makes no reservations.
    pub max: usize,
    /// Count `max` per worker class instead of globally. Default false.
    pub per_class: bool,
    /// EASY-style backfill on a reserved worker: less urgent jobs may still run there if they
    /// are expected to finish before the holder could start, its *shadow time*. The shadow time is
    /// computed once per reservation, from the running jobs' expected ends, predicting usage from
    /// placed demands (heartbeat usage cannot be predicted); an unknown end means no backfill.
    /// Once the shadow time passes nothing can finish before it, so the worker drains strictly
    /// from then on: the holder waits at most for the jobs running at the shadow time. Default
    /// false (strict draining from the start).
    pub shadow_backfill: bool,
}

impl Default for Reservations {
    /// The defaults documented on each field.
    fn default() -> Self {
        Self {
            reserve_after: 60.0,
            max: 1,
            per_class: false,
            shadow_backfill: false,
        }
    }
}

/// Which of the workers that admit a job it goes to. Preferred workers
/// ([`JobSpec::prefer`](crate::JobSpec::prefer)) and then the fewest running jobs break ties.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Fit {
    /// The least loaded (fewest running jobs), preferred workers first.
    #[default]
    LeastLoaded,
    /// The tightest fit: the smallest [`WorkerView::free_share`](crate::WorkerView::free_share)
    /// after placement.
    Tightest {
        /// How much a preferred worker is favoured, as a fraction of capacity: it competes as if
        /// its free share after placement were this much smaller. 0 makes preference a pure
        /// tie-breaker.
        prefer_penalty: f64,
    },
}

/// How worker speed ([`WorkerState::speed`](crate::WorkerState::speed)) enters placement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SpeedPolicy {
    /// Speed is ignored (the default).
    #[default]
    Oblivious,
    /// Among the workers that admit a job, the fastest first; load and fit break ties within a
    /// speed. On a span-bound run this is the single largest placement lever.
    FastestFirst,
    /// Earliest expected finish: like `FastestFirst` among workers free now, and, with
    /// [`Defer`], a job with [`JobSpec::work`](crate::JobSpec::work) may wait for a busy faster
    /// worker whose slot is expected to free soon enough that it would still finish earlier there
    /// (HEFT's processor choice, online; StarPU's dmda with a deferral window).
    EarliestFinish(Option<Defer>),
}

/// When a job may wait for a faster, busy worker instead of starting on a slower free one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Defer {
    /// A job that has waited this long (seconds) no longer defers. Bounds the extra waiting;
    /// expiry is reported by [`Policy::next_wakeup`](crate::Policy::next_wakeup).
    pub max_wait: f64,
    /// Defer only if the expected finish improves by at least this fraction of the job's work.
    pub min_gain: f64,
}

impl Default for Defer {
    /// Wait at most an hour, and only for at least a quarter of the job's work in gain: in
    /// simulation that keeps most of waiting's benefit while halving the cases where it backfires
    /// (a barely faster, scarce class).
    fn default() -> Self {
        Self {
            max_wait: 3600.0,
            min_gain: 0.25,
        }
    }
}

/// Speed-aware placement settings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeedConfig {
    /// How speed orders the candidate workers.
    pub policy: SpeedPolicy,
    /// Learn each worker class's speed from completion times instead of trusting
    /// [`WorkerState::speed`](crate::WorkerState::speed).
    pub learn: Option<Learn>,
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Speculate {
    /// Minimum gain, as a fraction of the new attempt's run time.
    pub min_gain: f64,
    /// Seconds a new attempt costs on top of its run time (it starts from scratch).
    pub restart_overhead: f64,
    /// A job gets at most this many speculative attempts.
    pub max_per_job: u32,
}

impl Default for Speculate {
    /// At least a quarter of the run time gained, no overhead, at most one extra attempt per job.
    fn default() -> Self {
        Self {
            min_gain: 0.25,
            restart_overhead: 0.0,
            max_per_job: 1,
        }
    }
}

/// How often a failed job is retried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryConfig {
    /// Failed attempts per job before it is given up
    /// ([`Output::GaveUp`](crate::Output::GaveUp)). Default 4; 0 counts as 1.
    pub max_attempts: u32,
}

impl Default for RetryConfig {
    /// Four attempts.
    fn default() -> Self {
        Self { max_attempts: 4 }
    }
}
