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

use std::collections::{BTreeMap, HashMap};

use crate::WorkerId;

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

/// A running mean of `ln speed`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Stat {
    mean: f64,
    n: u32,
}

impl Stat {
    /// Add a sample: plain averaging, then an exponential moving average of weight `weight`.
    fn add(&mut self, x: f64, weight: f64) {
        let a = weight.max(1.0 / f64::from(self.n + 1));
        self.mean += a * (x - self.mean);
        self.n = self.n.saturating_add(1);
    }
}

/// One worker's state.
#[derive(Clone, Debug, Default, PartialEq)]
struct WorkerStat {
    stat: Stat,
    /// The speed last handed out (hysteresis).
    published: Option<f64>,
}

/// Online estimates of worker speeds (see [`Learn`]), usable on its own to set
/// [`WorkerState::speed`](crate::WorkerState::speed), and what the scheduler learns related
/// speeds with ([`Timing`]). Deterministic.
///
/// Used on its own, the estimator turns completions into the speed a worker reports in its next
/// heartbeat; the scheduler then trusts it under the default [`Timing`].
///
/// ```
/// use whelm::{Input, Learn, Resources, SpeedEstimator, WorkerState};
///
/// let mut e = SpeedEstimator::new(Learn::default());
/// for _ in 0..Learn::default().min_samples {
///     e.observe(7, "h100", 30.0, 10.0, 1.0);
/// }
/// let state = WorkerState {
///     speed: e.speed(7, "h100", 1.0),
///     ..WorkerState::new(7, "h100", 4, Resources::mem_gb(80.0))
/// };
/// assert!((state.speed - 3.0).abs() < 1e-9);
/// let heartbeat = Input::Worker(state);
/// # let _ = heartbeat;
/// ```
#[derive(Clone, Debug)]
pub struct SpeedEstimator {
    cfg: Learn,
    classes: BTreeMap<String, Stat>,
    workers: BTreeMap<WorkerId, WorkerStat>,
}

impl SpeedEstimator {
    /// An estimator with no samples.
    pub fn new(cfg: Learn) -> Self {
        Self {
            cfg,
            classes: BTreeMap::new(),
            workers: BTreeMap::new(),
        }
    }

    /// Its configuration.
    pub fn config(&self) -> &Learn {
        &self.cfg
    }

    /// Record a completion on `worker` (of `class`): `work` seconds at speed 1 took `duration`
    /// seconds while the worker ran `concurrency` jobs on average (this one included). Samples
    /// with a non-positive or non-finite work or duration are ignored. Returns whether it was used.
    ///
    /// ```
    /// use whelm::{Learn, SpeedEstimator};
    ///
    /// let mut e = SpeedEstimator::new(Learn::default());
    /// assert!(e.observe(1, "cpu", 10.0, 5.0, 1.0));
    /// assert!(!e.observe(1, "cpu", 10.0, 0.0, 1.0));
    /// assert_eq!((e.class_samples("cpu"), e.worker_samples(1)), (1, 1));
    /// ```
    pub fn observe(
        &mut self,
        worker: WorkerId,
        class: &str,
        work: f64,
        duration: f64,
        concurrency: f64,
    ) -> bool {
        let Some(x) = self.sample(work, duration, concurrency) else {
            return false;
        };
        self.add(worker, class, x, x);
        true
    }

    /// The `ln speed` sample of a completion (see [`observe`](Self::observe)), corrected for
    /// concurrency; `None` if the work or duration is unusable.
    pub(crate) fn sample(&self, work: f64, duration: f64, concurrency: f64) -> Option<f64> {
        if !(work > 0.0 && duration > 0.0 && work.is_finite() && duration.is_finite()) {
            return None;
        }
        let mut x = (work / duration).ln();
        if let Some(s) = self.cfg.sharing
            && concurrency.is_finite()
        {
            x += s.log_correction(concurrency);
        }
        Some(x)
    }

    /// Record the sample `class_x` for `class` and `worker_x` for `worker`: the same sample,
    /// unless the worker's has a job kind's deviation taken out ([`Timing::Unrelated`]).
    pub(crate) fn add(&mut self, worker: WorkerId, class: &str, class_x: f64, worker_x: f64) {
        let w = self.cfg.weight;
        match self.classes.get_mut(class) {
            Some(c) => c.add(class_x, w),
            None => {
                let mut c = Stat::default();
                c.add(class_x, w);
                self.classes.insert(class.to_string(), c);
            }
        }
        self.workers
            .entry(worker)
            .or_default()
            .stat
            .add(worker_x, w);
    }

    /// The mean `ln speed` of a class's samples, if it has any, warmed up or not.
    pub(crate) fn class_mean(&self, class: &str) -> Option<f64> {
        self.classes.get(class).map(|c| c.mean)
    }

    /// `ln` of a class's speed: learned once warmed up, else the prior.
    fn class_log(&self, class: &str, prior: f64) -> f64 {
        match self.classes.get(class) {
            Some(c) if c.n >= self.cfg.min_samples => c.mean,
            _ => prior.ln(),
        }
    }

    /// The raw estimate of `worker`'s speed, without hysteresis. `prior` is the speed the worker
    /// reports, which stands for its class's until the class has [`Learn::min_samples`].
    ///
    /// A worker never seen gets its class's estimate, and a class never seen keeps the prior:
    ///
    /// ```
    /// use whelm::{Learn, SpeedEstimator};
    ///
    /// let mut e = SpeedEstimator::new(Learn::default());
    /// for _ in 0..Learn::default().min_samples {
    ///     e.observe(1, "gpu", 10.0, 5.0, 1.0);
    /// }
    /// assert!((e.estimate(1, "gpu", 1.0) - 2.0).abs() < 1e-9);
    /// assert!((e.estimate(2, "gpu", 1.0) - 2.0).abs() < 1e-9);
    /// assert_eq!(e.estimate(3, "cpu", 0.5), 0.5);
    /// ```
    pub fn estimate(&self, worker: WorkerId, class: &str, prior: f64) -> f64 {
        let prior = sane(prior);
        let class_log = self.class_log(class, prior);
        let log = match self.workers.get(&worker) {
            Some(ws) if self.cfg.per_worker && ws.stat.n > 0 => {
                let n = f64::from(ws.stat.n).min(1.0 / self.cfg.weight.max(1e-9));
                let k = self.cfg.worker_prior.max(0.0);
                (n * ws.stat.mean + k * class_log) / (n + k)
            }
            _ => class_log,
        };
        log.exp()
    }

    /// The speed to use for `worker` (of `class`, reporting `prior`): the estimate, moved only
    /// when it leaves the hysteresis band around the last value handed out
    /// ([`Learn::resolution`]).
    ///
    /// One job three times slower than usual moves the estimate but not the speed:
    ///
    /// ```
    /// use whelm::{Learn, SpeedEstimator};
    ///
    /// let mut e = SpeedEstimator::new(Learn::default());
    /// for _ in 0..100 {
    ///     e.observe(1, "cpu", 1.0, 1.0, 1.0);
    /// }
    /// let before = e.speed(1, "cpu", 1.0);
    /// e.observe(1, "cpu", 1.0, 3.0, 1.0);
    /// assert!(e.estimate(1, "cpu", 1.0) < before);
    /// assert_eq!(e.speed(1, "cpu", 1.0), before);
    /// ```
    pub fn speed(&mut self, worker: WorkerId, class: &str, prior: f64) -> f64 {
        let est = self.estimate(worker, class, prior);
        let band = self.cfg.resolution.max(0.0).ln_1p();
        let ws = self.workers.entry(worker).or_default();
        match ws.published {
            Some(p) if (est.ln() - p.ln()).abs() <= band => p,
            _ => {
                ws.published = Some(est);
                est
            }
        }
    }

    /// Samples recorded for a class.
    pub fn class_samples(&self, class: &str) -> u32 {
        self.classes.get(class).map_or(0, |c| c.n)
    }

    /// Samples recorded for a worker.
    pub fn worker_samples(&self, worker: WorkerId) -> u32 {
        self.workers.get(&worker).map_or(0, |w| w.stat.n)
    }
}

/// An interned worker class.
pub(crate) type ClassId = u32;

/// An interned job kind.
pub(crate) type KindId = u32;

/// Interned names: a dense id per distinct string, in order of first sight.
#[derive(Clone, Debug, Default)]
struct Names {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

impl Names {
    /// The id of `name`, interning it if new.
    fn id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }

    /// The name behind an id.
    fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }
}

/// A kind's speed on one class relative to the class's average ([`Timing::Unrelated`]).
#[derive(Clone, Copy, Debug)]
struct KindStat {
    /// The kind's own `ln speed` samples on the class.
    stat: Stat,
    /// `ln` of the factor last handed out (hysteresis).
    published: Option<f64>,
    /// `published.exp()`, or 1 before anything is published.
    factor: f64,
}

/// The scheduler's side of a [`Timing`]: speeds per worker and per (job kind, worker class), and
/// what they learn from completions.
#[derive(Clone, Debug)]
pub(crate) struct Speeds {
    timing: Timing,
    /// The related speeds' learner, when they are learned.
    estimator: Option<SpeedEstimator>,
    classes: Names,
    kinds: Names,
    /// [`Timing::Unrelated`]: every kind's factor on every class it has run on.
    per_kind: BTreeMap<(ClassId, KindId), KindStat>,
}

impl Speeds {
    /// No samples yet.
    pub(crate) fn new(timing: Timing) -> Self {
        Self {
            estimator: timing.learn().copied().map(SpeedEstimator::new),
            timing,
            classes: Names::default(),
            kinds: Names::default(),
            per_kind: BTreeMap::new(),
        }
    }

    /// Speeds within one step of this fraction of each other rank equal (0: no rounding).
    pub(crate) fn resolution(&self) -> f64 {
        self.timing.learn().map_or(0.0, |l| l.resolution)
    }

    /// The id of a worker class.
    pub(crate) fn class(&mut self, name: &str) -> ClassId {
        self.classes.id(name)
    }

    /// The id of a job kind, if the timing distinguishes kinds.
    pub(crate) fn kind(&mut self, name: Option<&str>) -> Option<KindId> {
        match self.timing {
            Timing::Unrelated { .. } => name.map(|n| self.kinds.id(n)),
            _ => None,
        }
    }

    /// The speed of worker `id` of `class` that reports `reported`, for a job of no particular
    /// kind: 1 for identical machines, else learned once there are enough samples, else as
    /// reported.
    pub(crate) fn worker_speed(&mut self, id: WorkerId, class: ClassId, reported: f64) -> f64 {
        match (&self.timing, self.estimator.as_mut()) {
            (Timing::Identical, _) => 1.0,
            (_, Some(e)) => e.speed(id, self.classes.name(class), reported),
            (_, None) => sane(reported),
        }
    }

    /// The factor a job of `kind` runs at on `class` relative to the worker's speed.
    pub(crate) fn factor(&self, kind: Option<KindId>, class: ClassId) -> f64 {
        kind.and_then(|k| self.per_kind.get(&(class, k)))
            .map_or(1.0, |s| s.factor)
    }

    /// Every class `kind` has a factor on, and that factor, by class id.
    pub(crate) fn kind_factors(&self, kind: KindId) -> Vec<(&str, f64)> {
        (self.per_kind.iter())
            .filter(|((_, k), _)| *k == kind)
            .map(|(&(c, _), s)| (self.classes.name(c), s.factor))
            .collect()
    }

    /// `ln` of `kind`'s factor on `class` from its samples, without hysteresis: its mean shrunk
    /// towards the class's.
    fn raw_log(&self, class: ClassId, kind: KindId) -> f64 {
        let (Some(e), Some(s), Timing::Unrelated { learn, kind_prior }) = (
            self.estimator.as_ref(),
            self.per_kind.get(&(class, kind)),
            self.timing,
        ) else {
            return 0.0;
        };
        let Some(c) = e.class_mean(self.classes.name(class)) else {
            return 0.0;
        };
        let n = f64::from(s.stat.n).min(1.0 / learn.weight.max(1e-9));
        n * (s.stat.mean - c) / (n + kind_prior.max(0.0))
    }

    /// Learn from a completion on `worker` of `class` of a job of `kind` (see
    /// [`SpeedEstimator::observe`]). Returns whether the sample was used; if so, every worker of
    /// the class needs its speed refreshed ([`worker_speed`](Self::worker_speed)), and every
    /// kind's factor on the class has been.
    pub(crate) fn observe(
        &mut self,
        worker: WorkerId,
        class: ClassId,
        kind: Option<KindId>,
        work: f64,
        duration: f64,
        concurrency: f64,
    ) -> bool {
        let Some(x) = (self.estimator.as_ref()).and_then(|e| e.sample(work, duration, concurrency))
        else {
            return false;
        };
        let (Timing::Unrelated { learn, .. }, Some(kind)) = (self.timing, kind) else {
            let e = self.estimator.as_mut().unwrap();
            e.add(worker, self.classes.name(class), x, x);
            return true;
        };
        // The worker learns the class's average job: this kind's deviation is taken out.
        let deviation = self.raw_log(class, kind);
        let e = self.estimator.as_mut().unwrap();
        e.add(worker, self.classes.name(class), x, x - deviation);
        (self.per_kind.entry((class, kind)))
            .or_insert(KindStat {
                stat: Stat::default(),
                published: None,
                factor: 1.0,
            })
            .stat
            .add(x, learn.weight);
        // The class's average moved, so every kind's factor on it did.
        let band = learn.resolution.max(0.0).ln_1p();
        let kinds: Vec<KindId> = (self.per_kind.range((class, 0)..=(class, KindId::MAX)))
            .map(|(&(_, k), _)| k)
            .collect();
        for k in kinds {
            let raw = self.raw_log(class, k);
            let s = self.per_kind.get_mut(&(class, k)).unwrap();
            match s.published {
                Some(p) if (raw - p).abs() <= band => {}
                _ => {
                    s.published = Some(raw);
                    s.factor = raw.exp();
                }
            }
        }
        true
    }
}

/// A reported speed, guarded against nonsense.
pub(crate) fn sane(x: f64) -> f64 {
    if x > 0.0 && x.is_finite() { x } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Noise-free samples recover each class's speed exactly.
    #[test]
    fn exact_recovery() {
        let mut e = SpeedEstimator::new(Learn::default());
        for i in 0..40 {
            e.observe(1 + i % 2, "a", 10.0, 5.0, 1.0);
            e.observe(10, "b", 10.0, 10.0, 1.0);
        }
        assert!((e.speed(1, "a", 1.0) - 2.0).abs() < 1e-9);
        assert!((e.speed(10, "b", 7.0) - 1.0).abs() < 1e-9);
        // A worker never seen gets its class's estimate.
        assert!((e.speed(3, "a", 1.0) - 2.0).abs() < 1e-9);
        // An unknown class keeps its prior.
        assert!((e.speed(4, "c", 3.0) - 3.0).abs() < 1e-9);
    }

    /// A slow member of a class is estimated slower than its peers, but not below its own mean.
    #[test]
    fn capped_card_shows_up() {
        let mut e = SpeedEstimator::new(Learn::default());
        for _ in 0..200 {
            for w in 1..=5 {
                e.observe(w, "h200", 100.0, 100.0, 4.0);
            }
            e.observe(6, "h200", 76.5, 100.0, 4.0);
        }
        let normal = e.speed(1, "h200", 1.0);
        let capped = e.speed(6, "h200", 1.0);
        assert!(capped < 0.85 * normal, "{capped} vs {normal}");
        assert!(capped > 0.765 * 0.99);
    }

    /// Identical machines report speed 1 whatever the worker says; related ones without learning
    /// trust it; neither distinguishes kinds.
    #[test]
    fn identical_and_reported() {
        let mut p = Speeds::new(Timing::Identical);
        let x = p.class("x");
        assert_eq!(p.worker_speed(1, x, 4.0), 1.0);
        assert_eq!(p.kind(Some("a")), None);
        assert!(!p.observe(1, x, None, 1.0, 1.0, 1.0));
        let mut q = Speeds::new(Timing::default());
        let x = q.class("x");
        assert_eq!(q.worker_speed(1, x, 4.0), 4.0);
        assert_eq!(q.kind(Some("a")), None);
    }

    /// A kind's factor is its speed on the class over the class's average, shrunk by
    /// `kind_prior`; a kind never seen on the class runs at the worker's speed.
    #[test]
    fn kind_factors() {
        let timing = Timing::Unrelated {
            learn: Learn {
                resolution: 0.0,
                ..Learn::default()
            },
            kind_prior: 5.0,
        };
        let mut s = Speeds::new(timing);
        let x = s.class("x");
        let (a, b, new) = (s.kind(Some("a")), s.kind(Some("b")), s.kind(Some("new")));
        // Kind a alone: it is the class's average, so its factor is 1 and the worker learns 4.
        for _ in 0..40 {
            s.observe(1, x, a, 10.0, 2.5, 1.0);
        }
        assert_eq!(s.factor(a, x), 1.0);
        assert!((s.worker_speed(1, x, 1.0) - 4.0).abs() < 1e-9);
        // Kind b at speed 1 drags the class average down; a's factor rises above 1, b's is
        // below, and each sits between 1 and its unshrunk ratio to the class mean.
        for _ in 0..40 {
            s.observe(1, x, b, 10.0, 10.0, 1.0);
            s.observe(1, x, a, 10.0, 2.5, 1.0);
        }
        let (fa, fb) = (s.factor(a, x), s.factor(b, x));
        assert!(fa > 1.0 && fb < 1.0, "{fa} {fb}");
        let n = 1.0 / Learn::default().weight;
        let shrink = n / (n + 5.0);
        assert!((fa.ln() / fb.ln() + 1.0).abs() < 0.2, "{fa} {fb}");
        assert!(fa.ln() / 2f64.ln() < shrink + 0.1, "{fa}");
        assert_eq!(s.factor(new, x), 1.0);
        assert_eq!(s.factor(None, x), 1.0);
        assert_eq!(s.kind_factors(a.unwrap()), vec![("x", fa)]);
    }

    /// One 3x-slow job does not move a warmed-up worker's published speed.
    #[test]
    fn hysteresis_absorbs_outliers() {
        let mut e = SpeedEstimator::new(Learn::default());
        for _ in 0..100 {
            e.observe(1, "a", 1.0, 1.0, 1.0);
        }
        let before = e.speed(1, "a", 1.0);
        e.observe(1, "a", 1.0, 3.0, 1.0);
        assert_eq!(e.speed(1, "a", 1.0), before);
        assert!(e.estimate(1, "a", 1.0) < before);
    }

    /// With linear sharing up to 16, concurrency does not matter; with no gain from sharing, a
    /// job at concurrency 4 took 4x longer and the correction recovers the speed.
    #[test]
    fn concurrency_correction() {
        let flat = Learn {
            sharing: Some(Sharing {
                k_sat: 1.0,
                alpha: 1.0,
            }),
            resolution: 0.0,
            ..Learn::default()
        };
        let mut e = SpeedEstimator::new(flat);
        for _ in 0..30 {
            e.observe(1, "a", 10.0, 40.0, 4.0);
        }
        assert!((e.speed(1, "a", 1.0) - 1.0).abs() < 1e-9);
        let linear = Learn {
            sharing: Some(Sharing {
                k_sat: 16.0,
                alpha: 1.0,
            }),
            resolution: 0.0,
            ..Learn::default()
        };
        let mut e = SpeedEstimator::new(linear);
        for _ in 0..30 {
            e.observe(1, "a", 10.0, 10.0, 4.0);
        }
        assert!((e.speed(1, "a", 1.0) - 1.0).abs() < 1e-9);
    }
}
