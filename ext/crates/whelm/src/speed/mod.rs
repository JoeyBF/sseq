//! Machine models: how fast a job runs on a worker, learned from completion times.
//!
//! [`Timing`] is the machine model a [`Scheduler`](crate::Scheduler) uses
//! ([`SpeedConfig::timing`](crate::SpeedConfig::timing)): identical machines, machines with one
//! speed each, or speeds that also depend on the kind of job. Speeds are reported by the workers
//! ([`WorkerState::speed`](crate::WorkerState::speed)) or learned from completion times as
//! [`Learn`] configures, with [`Sharing`] correcting for the worker's load. The learner is a
//! [`SpeedEstimator`], which a caller may also use on its own.
//!
//! An estimator learns a class's speed from completions: work of 10 s at speed 1 that took 5 s
//! is a sample of speed 2. Until the class has [`Learn::min_samples`] samples, its workers get the
//! speed they report.
//!
//! ```
//! use whelm::{Learn, SpeedEstimator};
//!
//! let mut e = SpeedEstimator::new(Learn::default());
//! for _ in 1..Learn::default().min_samples {
//!     e.observe(1, "gpu", 10.0, 5.0, 1.0);
//! }
//! assert_eq!(e.speed(2, "gpu", 1.5), 1.5);
//! e.observe(1, "gpu", 10.0, 5.0, 1.0);
//! assert!((e.speed(2, "gpu", 1.5) - 2.0).abs() < 1e-9);
//! ```

mod estimator;
mod model;

pub use estimator::SpeedEstimator;
pub(crate) use model::{ClassId, KindId, Speeds};

/// The machine model: how a job's speed depends on the worker it runs on.
///
/// A job's run time is its [`JobSpec::work`](crate::JobSpec::work) over that speed;
/// [`ScoreTerm::Speed`](crate::ScoreTerm::Speed) ranks workers by it, and
/// [`Defer`](crate::Defer), shadow backfill and [`Speculate`](crate::Speculate) estimate run times
/// with it. What is learned lives in the scheduler, so replaying a log rebuilds it.
///
/// The examples on the variants share two hidden helpers: `two_classes(timing)`, a scheduler with
/// one one-slot worker of class "x" (1) and one of class "y" (2), both reporting speed 1; and
/// `place(scheduler, job, now)`, which submits a job alone, returns the worker it starts on and
/// reports it done.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Timing {
    /// Identical machines (P): every job runs at speed 1 everywhere. Reported speeds are ignored,
    /// so the speed term ties, nothing defers and nothing is speculated.
    ///
    /// A worker reporting speed 4 gets no preference, and the statistics show speed 1:
    ///
    /// ```
    /// # use whelm::{
    /// #     Config, Input, JobSpec, Output, Policy, Resources, Scheduler, SpeedConfig, Timing,
    /// #     WorkerId, WorkerState,
    /// # };
    /// # /// A scheduler with `timing` and two one-slot workers: 1 of class "x" and 2 of class "y",
    /// # /// each reporting speed 1.
    /// # fn two_classes(timing: Timing) -> Scheduler {
    /// #     let mut s = Scheduler::new(Config {
    /// #         speed: SpeedConfig { timing, ..SpeedConfig::default() },
    /// #         ..Config::default()
    /// #     });
    /// #     for (id, class) in [(1, "x"), (2, "y")] {
    /// #         s.handle(Input::Worker(WorkerState::new(id, class, 1, Resources::ZERO)), 0.0);
    /// #     }
    /// #     s
    /// # }
    /// # /// Where a lone job goes, both workers free, and that it then finishes at once.
    /// # fn place(s: &mut Scheduler, job: JobSpec, now: f64) -> WorkerId {
    /// #     let id = job.id;
    /// #     s.handle(Input::Submit(job), now);
    /// #     let [Output::Start { worker, .. }] = s.poll(now)[..] else { panic!() };
    /// #     s.handle(Input::Done { job: id, attempt: 1 }, now);
    /// #     worker
    /// # }
    /// let fast = WorkerState {
    ///     speed: 4.0,
    ///     ..WorkerState::new(2, "y", 1, Resources::ZERO)
    /// };
    /// let mut s = two_classes(Timing::Identical);
    /// s.handle(Input::Worker(fast.clone()), 0.0);
    /// assert_eq!(place(&mut s, JobSpec::new(0, Resources::ZERO, 0), 0.0), 1);
    /// assert_eq!(s.stats().workers[1].speed, 1.0);
    /// // The default, related machines at their reported speeds, prefers it.
    /// let mut s = two_classes(Timing::default());
    /// s.handle(Input::Worker(fast), 0.0);
    /// assert_eq!(place(&mut s, JobSpec::new(0, Resources::ZERO, 0), 0.0), 2);
    /// assert_eq!(s.stats().workers[1].speed, 4.0);
    /// ```
    Identical,
    /// Uniformly related machines (Q): every job runs at its worker's speed, the reported
    /// [`WorkerState::speed`](crate::WorkerState::speed) or, with `learn`, an estimate learned per
    /// worker ([`SpeedEstimator`]: its class as prior, the reported speed as the class's prior).
    /// [`Timing::Identical`] shows reported speeds, and [`Timing::learned`] learned ones.
    Related {
        /// Learn speeds instead of trusting the reported ones.
        learn: Option<Learn>,
    },
    /// Unrelated machines (R): a job's speed depends on its [`kind`](crate::JobSpec::kind) as well
    /// as its worker. On worker `w` it is `w`'s speed as [`Timing::Related`] learns it, from jobs
    /// of every kind, times the kind's factor on `w`'s class: how much faster the kind runs there
    /// than the class's average job, learned per (kind, class) and shrunk towards 1. A new kind,
    /// or a job without one, runs at the related speed; a worker slow for its class is slow for
    /// every kind. Kinds are interned for good, so they should be a small set (the job's
    /// algorithm, not its size).
    ///
    /// Kind "a" runs four times as fast on class "x" as on "y", and kind "b" twice as fast on "y"
    /// as on "x". After enough of each kind on each class, each kind goes to its own class; the
    /// related model, one speed per worker, sends both to "x", whose average is higher.
    ///
    /// ```
    /// # use whelm::{
    /// #     Config, Input, JobSpec, Output, Policy, Resources, Scheduler, SpeedConfig, Timing,
    /// #     WorkerId, WorkerState,
    /// # };
    /// # /// A scheduler with `timing` and two one-slot workers: 1 of class "x" and 2 of class "y",
    /// # /// each reporting speed 1.
    /// # fn two_classes(timing: Timing) -> Scheduler {
    /// #     let mut s = Scheduler::new(Config {
    /// #         speed: SpeedConfig { timing, ..SpeedConfig::default() },
    /// #         ..Config::default()
    /// #     });
    /// #     for (id, class) in [(1, "x"), (2, "y")] {
    /// #         s.handle(Input::Worker(WorkerState::new(id, class, 1, Resources::ZERO)), 0.0);
    /// #     }
    /// #     s
    /// # }
    /// # /// Where a lone job goes, both workers free, and that it then finishes at once.
    /// # fn place(s: &mut Scheduler, job: JobSpec, now: f64) -> WorkerId {
    /// #     let id = job.id;
    /// #     s.handle(Input::Submit(job), now);
    /// #     let [Output::Start { worker, .. }] = s.poll(now)[..] else { panic!() };
    /// #     s.handle(Input::Done { job: id, attempt: 1 }, now);
    /// #     worker
    /// # }
    /// let truth = |kind, worker| match (kind, worker) {
    ///     ("a", 1) => 4.0,
    ///     ("b", 2) => 2.0,
    ///     _ => 1.0,
    /// };
    /// let train = |s: &mut Scheduler| {
    ///     let (mut now, mut id) = (0.0, 0);
    ///     for _ in 0..20 {
    ///         for kind in ["a", "b"] {
    ///             for (w, class) in [(1, "x"), (2, "y")] {
    ///                 let job = JobSpec {
    ///                     work: Some(8.0),
    ///                     ..JobSpec::new(id, Resources::ZERO, 0)
    ///                 };
    ///                 s.handle(Input::Submit(job.with_kind(kind).require_class(class)), now);
    ///                 assert_eq!(s.poll(now).len(), 1);
    ///                 now += 8.0 / truth(kind, w);
    ///                 s.handle(Input::Done { job: id, attempt: 1 }, now);
    ///                 id += 1;
    ///             }
    ///         }
    ///     }
    ///     now
    /// };
    /// let kind = |id, kind| JobSpec {
    ///     work: Some(8.0),
    ///     ..JobSpec::new(id, Resources::ZERO, 0)
    /// }
    /// .with_kind(kind);
    ///
    /// let mut s = two_classes(Timing::unrelated());
    /// let now = train(&mut s);
    /// assert_eq!(place(&mut s, kind(100, "a"), now), 1);
    /// assert_eq!(place(&mut s, kind(101, "b"), now), 2);
    /// s.handle(Input::Submit(kind(102, "a")), now);
    /// let why = s.explain(102).unwrap();
    /// assert!(why.contains("kind a runs") && why.contains("on class y"), "{why}");
    ///
    /// let mut s = two_classes(Timing::learned());
    /// let now = train(&mut s);
    /// assert_eq!(place(&mut s, kind(100, "a"), now), 1);
    /// assert_eq!(place(&mut s, kind(101, "b"), now), 1);
    /// ```
    Unrelated {
        /// The learning of the related speeds, whose moving-average weight, hysteresis and
        /// concurrency correction the kind factors share.
        learn: Learn,
        /// How many samples the class's average counts for in a kind's factor on the class: a
        /// kind with `n` samples there gets weight `min(n, 1 / learn.weight) / (that +
        /// kind_prior)`. Larger trusts a kind's first runs on a class less.
        kind_prior: f64,
    },
}

impl Default for Timing {
    /// Related machines at their reported speeds.
    fn default() -> Self {
        Self::Related { learn: None }
    }
}

impl Timing {
    /// Related machines, speeds learned with [`Learn::default`].
    ///
    /// Both workers report speed 1, but jobs on worker 2 take half as long. Once its class has
    /// [`Learn::min_samples`] samples, it is known to run at speed 2, and an unconstrained job
    /// goes there:
    ///
    /// ```
    /// # use whelm::{
    /// #     Config, Input, JobSpec, Output, Policy, Resources, Scheduler, SpeedConfig, Timing,
    /// #     WorkerId, WorkerState,
    /// # };
    /// # /// A scheduler with `timing` and two one-slot workers: 1 of class "x" and 2 of class "y",
    /// # /// each reporting speed 1.
    /// # fn two_classes(timing: Timing) -> Scheduler {
    /// #     let mut s = Scheduler::new(Config {
    /// #         speed: SpeedConfig { timing, ..SpeedConfig::default() },
    /// #         ..Config::default()
    /// #     });
    /// #     for (id, class) in [(1, "x"), (2, "y")] {
    /// #         s.handle(Input::Worker(WorkerState::new(id, class, 1, Resources::ZERO)), 0.0);
    /// #     }
    /// #     s
    /// # }
    /// # /// Where a lone job goes, both workers free, and that it then finishes at once.
    /// # fn place(s: &mut Scheduler, job: JobSpec, now: f64) -> WorkerId {
    /// #     let id = job.id;
    /// #     s.handle(Input::Submit(job), now);
    /// #     let [Output::Start { worker, .. }] = s.poll(now)[..] else { panic!() };
    /// #     s.handle(Input::Done { job: id, attempt: 1 }, now);
    /// #     worker
    /// # }
    /// let mut s = two_classes(Timing::learned());
    /// let mut now = 0.0;
    /// for id in 0..u64::from(Learn::default().min_samples) {
    ///     let job = JobSpec {
    ///         work: Some(10.0),
    ///         ..JobSpec::new(id, Resources::ZERO, 0)
    ///     };
    ///     s.handle(Input::Submit(job.require_class("y")), now);
    ///     s.poll(now);
    ///     now += 5.0;
    ///     s.handle(Input::Done { job: id, attempt: 1 }, now);
    /// }
    /// assert!((s.stats().workers[1].speed - 2.0).abs() < 1e-9);
    /// assert_eq!(place(&mut s, JobSpec::new(1000, Resources::ZERO, 0), now), 2);
    /// # use whelm::Learn;
    /// ```
    pub fn learned() -> Self {
        Self::Related {
            learn: Some(Learn::default()),
        }
    }

    /// Unrelated machines learned with [`Learn::default`], the class's average counting for as
    /// many samples against a kind as against a worker ([`Learn::worker_prior`]).
    /// [`Timing::Unrelated`] has an example.
    pub fn unrelated() -> Self {
        let learn = Learn::default();
        Self::Unrelated {
            kind_prior: learn.worker_prior,
            learn,
        }
    }

    /// How speeds are learned, if they are.
    ///
    /// ```
    /// use whelm::{Learn, Timing};
    ///
    /// assert_eq!(Timing::Identical.learn(), None);
    /// assert_eq!(Timing::default().learn(), None);
    /// assert_eq!(Timing::unrelated().learn(), Some(&Learn::default()));
    /// ```
    pub fn learn(&self) -> Option<&Learn> {
        match self {
            Self::Identical => None,
            Self::Related { learn } => learn.as_ref(),
            Self::Unrelated { learn, .. } => Some(learn),
        }
    }
}

/// Online speed learning: each completed job with work `w` (seconds at speed 1) that ran `d`
/// seconds is a sample `ln(w / d)` of its worker's speed, averaged in log space (durations are
/// log-normal). Unlike StarPU's history models there is no outlier filter: per-job noise is wide
/// enough that a "50% off the mean" filter would discard most samples (see `whelm-sim`'s
/// RESULTS.md, "The DAG", for the fitted spread).
///
/// A class's estimate replaces the reported speed once the class has `min_samples` samples. With
/// `per_worker`, each worker's estimate is its own mean shrunk towards its class's estimate, so
/// that a clock-capped card shows up as a slower member of its class.
///
/// Per worker, a member of a class at speed 1 that runs at half speed is learned slower than its
/// class, but not as slow as its own samples say: its class is its prior.
///
/// ```
/// use whelm::{Learn, SpeedEstimator};
///
/// let mut e = SpeedEstimator::new(Learn::default());
/// for _ in 0..100 {
///     e.observe(1, "gpu", 10.0, 10.0, 1.0);
///     e.observe(2, "gpu", 10.0, 10.0, 1.0);
///     e.observe(3, "gpu", 10.0, 20.0, 1.0); // half speed
/// }
/// let (healthy, capped) = (e.speed(1, "gpu", 1.0), e.speed(3, "gpu", 1.0));
/// assert!(
///     capped < 0.9 * healthy && capped > 0.5,
///     "{capped} vs {healthy}"
/// );
/// // Without `per_worker`, every member gets the class's speed.
/// let mut e = SpeedEstimator::new(Learn {
///     per_worker: false,
///     ..Learn::default()
/// });
/// for _ in 0..100 {
///     e.observe(1, "gpu", 10.0, 10.0, 1.0);
///     e.observe(3, "gpu", 10.0, 20.0, 1.0);
/// }
/// assert_eq!(e.speed(1, "gpu", 1.0), e.speed(3, "gpu", 1.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Learn {
    /// Weight of a new sample once warmed up (an exponential moving average; plain averaging
    /// until then). It also caps how many samples a worker's own mean counts for, `1 / weight`.
    /// Larger follows a changing speed sooner, with noisier estimates.
    pub weight: f64,
    /// Samples a class needs before its learned speed replaces the reported one. Fewer trusts a
    /// noisier estimate sooner.
    pub min_samples: u32,
    /// Learn each worker's speed (shrunk towards its class), not only each class's.
    pub per_worker: bool,
    /// How many samples the class estimate counts for in a worker's estimate: a worker's own
    /// mean of `n` samples gets weight `min(n, 1 / weight) / (that + worker_prior)`.
    pub worker_prior: f64,
    /// Hysteresis: a worker's published speed changes only when its estimate moves by more than
    /// this fraction; speed-ordered placement also treats speeds within one such step as equal,
    /// so load still breaks ties within a class. 0 disables both.
    pub resolution: f64,
    /// How a worker's per-job speed depends on its concurrency, to normalise samples taken at
    /// different loads. `None`: per-job speed does not depend on load (the fitted service model,
    /// up to the slot count; see `whelm-sim`'s RESULTS.md).
    pub sharing: Option<Sharing>,
}

impl Default for Learn {
    /// Per worker with its class as prior, smoothed and with hysteresis against per-job noise;
    /// load-independent per-job speed. Its effect on a whole run is in `whelm-sim`'s RESULTS.md,
    /// "Restart-stable order, learned speeds".
    fn default() -> Self {
        Self {
            weight: 0.05,
            min_samples: 20,
            per_worker: true,
            worker_prior: 5.0,
            resolution: 0.1,
            sharing: None,
        }
    }
}

/// Processor sharing with saturation: a worker running `k` jobs delivers a total throughput
/// `speed * min(k, k_sat)^alpha`, split equally, so each job runs at
/// `speed * min(k, k_sat)^alpha / k`.
///
/// A worker that gains nothing from running several jobs at once (`k_sat` 1) runs each of 4
/// concurrent jobs at a quarter of its speed: a job of 10 s of work that took 40 s at concurrency
/// 4 is a sample of speed 1, not 0.25.
///
/// ```
/// use whelm::{Learn, Sharing, SpeedEstimator};
///
/// let learned = |sharing| {
///     let mut e = SpeedEstimator::new(Learn {
///         sharing,
///         min_samples: 1,
///         ..Learn::default()
///     });
///     e.observe(1, "cpu", 10.0, 40.0, 4.0);
///     e.speed(1, "cpu", 1.0)
/// };
/// let no_gain = Sharing {
///     k_sat: 1.0,
///     alpha: 1.0,
/// };
/// assert!((learned(Some(no_gain)) - 1.0).abs() < 1e-9);
/// assert!((learned(None) - 0.25).abs() < 1e-9);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sharing {
    /// Concurrency beyond which throughput stops growing.
    pub k_sat: f64,
    /// Growth exponent below saturation (1: linear, each job at full speed).
    pub alpha: f64,
}

impl Sharing {
    /// `ln` of the factor that turns an observed per-job rate at mean concurrency `k` into the
    /// worker's `speed`.
    fn log_correction(&self, k: f64) -> f64 {
        let k = k.max(1.0);
        k.ln() - self.alpha * k.min(self.k_sat.max(1.0)).ln()
    }
}
