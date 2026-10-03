//! The [`Scheduler`](crate::Scheduler)'s configuration.

use crate::Learn;

/// How a [`Scheduler`](crate::Scheduler) behaves: plain data, with presets.
///
/// A configuration is a list-scheduling rule: [`order`](Self::order) says which waiting job goes
/// first and [`score`](Self::score) which of the workers that admit it it goes to. The presets map
/// objectives to rules. On one machine some of these rules are optimal (Smith's rule for weighted
/// completion time, Jackson's rule for maximum lateness); on many machines, with resources and
/// online arrivals, every one of them is a heuristic.
///
/// - [`Default`]: makespan with bounded latency. Explicit priority, rank, group, then arrival;
///   aging and one reservation; fastest, preferred, then least loaded worker.
/// - [`Config::fifo`]: arrival order, no aging or reservations. A baseline.
/// - [`Config::best_fit`]: the default, packing each job into the tightest worker.
/// - [`Config::weighted_completion`]: weighted completion time ([`OrderTerm::Wspt`]).
/// - [`Config::lateness`]: maximum lateness ([`OrderTerm::Edd`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// The order waiting jobs are considered in: lexicographic over these terms, then submission
    /// order. A term listed twice adds nothing; the repeat is ignored.
    pub order: Vec<OrderTerm>,
    /// How [`OrderTerm::Group`] orders groups. Default [`GroupOrder::Arrival`].
    pub group_order: GroupOrder,
    /// The priority of jobs whose [`JobSpec::priority`](crate::JobSpec::priority) is `None`, for
    /// [`OrderTerm::Priority`]. Default 0, so negative priorities jump ahead of unprioritised jobs
    /// and positive ones fall behind them.
    pub default_priority: i64,
    /// Aging: a job that has waited at least this long (seconds) becomes more urgent than every
    /// job that has not, oldest first, whatever [`order`](Self::order) says. Strict priority
    /// starves a job for as long as more urgent jobs keep arriving (a young group behind a wide
    /// old one), and this bounds it. Default [`DEFAULT_AGE_LIMIT`]; `None` is strict priority.
    pub age_limit: Option<f64>,
    /// Workers drained for starving jobs. Default [`Reservations::default`]; `None` allows
    /// starvation of jobs larger than the typical headroom.
    pub reservations: Option<Reservations>,
    /// Which of the workers that admit a job it goes to: lexicographic over these terms, then the
    /// smallest worker id. A term listed twice adds nothing; the repeat is ignored.
    pub score: Vec<ScoreTerm>,
    /// Speed learning, deferral and speculation. Default: none of them.
    pub speed: SpeedConfig,
    /// Retries of failed attempts. Default [`RetryConfig::default`].
    pub retry: RetryConfig,
}

impl Default for Config {
    /// Order `[Priority, Rank, Group]`, [`DEFAULT_AGE_LIMIT`], one reservation, score `[Speed,
    /// Preferred, Load]`, default retries.
    fn default() -> Self {
        Self {
            order: vec![OrderTerm::Priority, OrderTerm::Rank, OrderTerm::Group],
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
    pub fn weighted_completion() -> Self {
        Self {
            order: vec![OrderTerm::Priority, OrderTerm::Wspt],
            ..Self::default()
        }
    }

    /// The default with order `[Priority, Edd]`, for maximum lateness: Jackson's rule, optimal on
    /// one machine.
    pub fn lateness() -> Self {
        Self {
            order: vec![OrderTerm::Priority, OrderTerm::Edd],
            ..Self::default()
        }
    }
}

/// [`Config::age_limit`]'s default, seconds: in the trace replay, 30 minutes cut the maximum wait
/// 8x at no throughput cost.
pub const DEFAULT_AGE_LIMIT: f64 = 1800.0;

/// One term of [`Config::order`]. Every key is computed once, at submission; a job that lacks
/// what a term reads sorts after every job that has it, within that term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrderTerm {
    /// [`JobSpec::priority`](crate::JobSpec::priority), smallest first; unset counts as
    /// [`Config::default_priority`].
    Priority,
    /// [`JobSpec::rank`](crate::JobSpec::rank), largest first: the longest remaining chain first,
    /// as in HEFT. Set by the DAG layer; unset sorts last.
    Rank,
    /// [`JobSpec::group`](crate::JobSpec::group), in [`Config::group_order`].
    Group,
    /// Weighted shortest processing time, Smith's rule: largest
    /// [`weight`](crate::JobSpec::weight) over [`work`](crate::JobSpec::work) first. Jobs without
    /// a work estimate sort last.
    Wspt,
    /// Earliest due date, Jackson's rule: smallest [`due`](crate::JobSpec::due) first. Jobs
    /// without one sort last.
    Edd,
}

/// One term of [`Config::score`], ranking the workers that admit a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScoreTerm {
    /// The fastest first ([`WorkerState::speed`](crate::WorkerState::speed), or learned). With
    /// [`Learn`] and a resolution, speeds within one resolution step of each other tie, so
    /// per-worker noise does not override the later terms.
    Speed,
    /// The tightest fit: the smallest [`WorkerView::free_share`](crate::WorkerView::free_share)
    /// after placement.
    Tightest,
    /// The loosest fit: the largest free share after placement.
    Loosest,
    /// Workers a [`Strength::Prefer`](crate::Strength::Prefer) constraint selects first.
    Preferred,
    /// The fewest live attempts.
    Load,
}

/// How [`OrderTerm::Group`] orders [`JobSpec::group`](crate::JobSpec::group)s.
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

/// When a job may wait for a busy worker faster than the one [`Config::score`] picked, instead of
/// starting there: earliest finish time with a deferral window (HEFT's processor choice, online;
/// StarPU's dmda). A job with [`JobSpec::work`](crate::JobSpec::work) waits for the full, faster
/// worker on which it is expected to finish earliest, if that beats starting now by enough.
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

/// Speed-aware settings. How speed ranks workers is [`ScoreTerm::Speed`]'s place in
/// [`Config::score`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeedConfig {
    /// Learn each worker class's speed from completion times instead of trusting
    /// [`WorkerState::speed`](crate::WorkerState::speed).
    pub learn: Option<Learn>,
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
    /// Rounds per job before it is given up ([`Output::GaveUp`](crate::Output::GaveUp)): a round
    /// is an attempt started from the queue, with any speculative attempts made alongside it, and
    /// it fails when its last live attempt does. Default 4; 0 counts as 1.
    pub max_attempts: u32,
}

impl Default for RetryConfig {
    /// Four attempts.
    fn default() -> Self {
        Self { max_attempts: 4 }
    }
}
