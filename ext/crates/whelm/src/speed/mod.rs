//! Machine models: how fast a job runs on a worker, reported or learned, and what speed drives.
//!
//! Workers differ in speed. A job's [`work`](JobSpec::work) is its run time on a worker of speed
//! 1, so on a worker of speed `s` it is expected to take `work / s`. The machine model,
//! [`Timing`] in [`SpeedConfig::timing`], says where speeds come from:
//!
//! - [`Timing::Identical`]: every worker runs at speed 1, whatever it reports.
//! - [`Timing::Related`] (the default): each worker has one speed, as reported in
//!   [`WorkerState::speed`] or, with [`Learn`], learned from completion times.
//! - [`Timing::Unrelated`]: a job's speed also depends on its [`kind`](JobSpec::kind), learned per
//!   kind and worker class.
//!
//! With [`ScoreTerm::Speed`] in the score, a worker reporting speed 3 wins a job under the default
//! model, and counts as any other worker under `Identical`:
//!
//! ```
//! # use whelm::prelude::*;
//! # use whelm::config::SpeedConfig;
//! # use whelm::speed::Timing;
//! /// The worker a job goes to, given a reference worker 1 and a worker 2 reporting speed 3.
//! fn place(timing: Timing) -> u64 {
//!     let mut p = Scheduler::new(Config {
//!         speed: SpeedConfig {
//!             timing,
//!             ..SpeedConfig::default()
//!         },
//!         ..Config::default()
//!     });
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id: 1,
//!             class: "a".into(),
//!             capacity: Resources::new().with(SLOTS, 4),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//!     let fast = WorkerState {
//!         id: 2,
//!         class: "b".into(),
//!         capacity: Resources::new().with(SLOTS, 4),
//!         speed: 3.0,
//!         ..Default::default()
//!     };
//!     p.handle(Input::Worker(fast), Time::ORIGIN);
//!     p.handle(
//!         Input::Submit {
//!             job: 1,
//!             spec: JobSpec::default(),
//!         },
//!         Time::ORIGIN,
//!     );
//!     match p.poll(Time::ORIGIN)[..] {
//!         [Output::Start { worker, .. }] => worker,
//!         ref out => panic!("{out:?}"),
//!     }
//! }
//! assert_eq!(place(Timing::default()), 2);
//! assert_eq!(place(Timing::Identical), 1);
//! ```
//!
//! # Learning speeds
//!
//! With [`Timing::learned`], each completion of a job with a work estimate is a sample of its
//! worker's speed, and a class's estimate replaces the reported speed once it has
//! [`min_samples`](Learn::min_samples); [`Sharing`] corrects each sample for the worker's load.
//! Here both workers report speed 1, but worker 2 really runs three times faster;
//! [`WorkerLoad::speed`] shows what the policy has learned.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! # use whelm::config::SpeedConfig;
//! # use whelm::job::Constraint;
//! # use whelm::speed::{Learn, Timing};
//! let mut p = Scheduler::new(Config {
//!     speed: SpeedConfig {
//!         timing: Timing::learned(),
//!         ..SpeedConfig::default()
//!     },
//!     ..Config::default()
//! });
//! for (id, class) in [(1, "a"), (2, "b")] {
//!     let worker = WorkerState {
//!         id,
//!         class: class.into(),
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Worker(worker), Time::ORIGIN);
//! }
//!
//! // Run jobs of 30 s of work on each worker in turn, pinned there by class.
//! let work = Duration::from_secs(30);
//! let (mut now, mut job) = (Time::ORIGIN, 0);
//! for _ in 0..Learn::default().min_samples {
//!     for (class, true_speed) in [("a", 1.0), ("b", 3.0)] {
//!         let spec = JobSpec {
//!             work: Some(work),
//!             constraints: vec![Constraint::require_class(class)],
//!             ..Default::default()
//!         };
//!         p.handle(Input::Submit { job, spec }, now);
//!         p.poll(now);
//!         now += work.div_f64(true_speed);
//!         p.handle(Input::Done { job, attempt: 1 }, now);
//!         job += 1;
//!     }
//! }
//! let speeds: Vec<f64> = p.stats().workers.iter().map(|w| w.speed).collect();
//! assert!(
//!     (speeds[0] - 1.0).abs() < 1e-9 && (speeds[1] - 3.0).abs() < 1e-9,
//!     "{speeds:?}"
//! );
//! ```
//!
//! Under [`Timing::unrelated`], the same samples are also split by job kind: a kind that runs
//! unusually fast on one class learns a factor there. Two kinds that favour different classes then
//! go to different workers, where one speed per worker would send both to the faster one on
//! average.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! # use whelm::config::SpeedConfig;
//! # use whelm::job::Constraint;
//! # use whelm::speed::Timing;
//! /// Workers 1 (class x) and 2 (class y) after training: kind "a" runs four times faster on x,
//! /// kind "b" twice as fast on y. Returns the policy and the time.
//! fn trained(timing: Timing) -> (Scheduler, Time) {
//!     let mut p = Scheduler::new(Config {
//!         speed: SpeedConfig {
//!             timing,
//!             ..SpeedConfig::default()
//!         },
//!         ..Config::default()
//!     });
//!     for (id, class) in [(1, "x"), (2, "y")] {
//!         let worker = WorkerState {
//!             id,
//!             class: class.into(),
//!             capacity: Resources::new().with(SLOTS, 1),
//!             ..Default::default()
//!         };
//!         p.handle(Input::Worker(worker), Time::ORIGIN);
//!     }
//!     let work = Duration::from_secs(8);
//!     let (mut now, mut job) = (Time::ORIGIN, 0);
//!     for _ in 0..20 {
//!         let runs = [
//!             ("a", "x", 4.0),
//!             ("a", "y", 1.0),
//!             ("b", "x", 1.0),
//!             ("b", "y", 2.0),
//!         ];
//!         for (kind, class, true_speed) in runs {
//!             let spec = JobSpec {
//!                 work: Some(work),
//!                 kind: Some(kind.into()),
//!                 constraints: vec![Constraint::require_class(class)],
//!                 ..Default::default()
//!             };
//!             p.handle(Input::Submit { job, spec }, now);
//!             p.poll(now);
//!             now += work.div_f64(true_speed);
//!             p.handle(Input::Done { job, attempt: 1 }, now);
//!             job += 1;
//!         }
//!     }
//!     (p, now)
//! }
//!
//! /// Where a lone job of `kind` goes once trained.
//! fn place(timing: Timing, kind: &str) -> u64 {
//!     let (mut p, now) = trained(timing);
//!     let spec = JobSpec {
//!         work: Some(Duration::from_secs(8)),
//!         kind: Some(kind.into()),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit { job: 1000, spec }, now);
//!     match p.poll(now)[..] {
//!         [Output::Start { worker, .. }] => worker,
//!         ref out => panic!("{out:?}"),
//!     }
//! }
//! assert_eq!(
//!     (
//!         place(Timing::unrelated(), "a"),
//!         place(Timing::unrelated(), "b")
//!     ),
//!     (1, 2)
//! );
//! assert_eq!(
//!     (place(Timing::learned(), "a"), place(Timing::learned(), "b")),
//!     (1, 1)
//! );
//! ```
//!
//! The learner is a [`SpeedEstimator`], which a caller may also use on its own. It learns a class's
//! speed from completions: work of 10 s at speed 1 that took 5 s is a sample of speed 2. Until the
//! class has [`Learn::min_samples`] samples, its workers get the speed they report.
//!
//! ```
//! use std::time::Duration;
//!
//! use whelm::speed::{Learn, SpeedEstimator};
//!
//! let mut e = SpeedEstimator::new(Learn::default());
//! for _ in 1..Learn::default().min_samples {
//!     e.observe(
//!         1,
//!         "gpu",
//!         Duration::from_secs(10),
//!         Duration::from_secs(5),
//!         1.0,
//!     );
//! }
//! assert_eq!(e.speed(2, "gpu", 1.5), 1.5);
//! e.observe(
//!     1,
//!     "gpu",
//!     Duration::from_secs(10),
//!     Duration::from_secs(5),
//!     1.0,
//! );
//! assert!((e.speed(2, "gpu", 1.5) - 2.0).abs() < 1e-9);
//! ```
//!
//! # Waiting for a faster worker
//!
//! The score picks the best worker that admits a job *now*. With [`SpeedConfig::defer`], a job may
//! instead wait for a busy, faster worker on which it would finish sooner (earliest finish time, as
//! in HEFT). The wait is a [hold](crate::scheduler#holds-and-wakeups): it shows in
//! [`PolicyStats::deferred`] and as an `explain`'s [`Holding::Deferral`], and it lapses after
//! [`max_wait`](Defer::max_wait).
//!
//! ```
//! # use std::time::Duration;
//! #
//! # use whelm::prelude::*;
//! # use whelm::config::{Defer, SpeedConfig};
//! let mut p = Scheduler::new(Config {
//!     speed: SpeedConfig {
//!         defer: Some(Defer::default()),
//!         ..SpeedConfig::default()
//!     },
//!     ..Config::default()
//! });
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         class: "slow".into(),
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! let fast = WorkerState {
//!     id: 2,
//!     class: "fast".into(),
//!     speed: 4.0,
//!     capacity: Resources::new().with(SLOTS, 1),
//!     ..Default::default()
//! };
//! p.handle(Input::Worker(fast), Time::ORIGIN);
//! let spec = |work| JobSpec {
//!     work: Some(Duration::from_secs(work)),
//!     ..Default::default()
//! };
//!
//! // Job 1 takes the fast worker until 10 / 4 = 2.5 s.
//! p.handle(
//!     Input::Submit {
//!         job: 1,
//!         spec: spec(10),
//!     },
//!     Time::ORIGIN,
//! );
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 2
//!     }]
//! );
//!
//! // Job 2 would take 40 s on the slow worker, or 2.5 + 10 s on the fast one: it waits.
//! p.handle(
//!     Input::Submit {
//!         job: 2,
//!         spec: spec(40),
//!     },
//!     Time::ORIGIN,
//! );
//! assert!(p.poll(Time::ORIGIN).is_empty());
//! assert_eq!(
//!     p.stats().deferred,
//!     [(2, 2, Time(Duration::from_millis(2500)))]
//! );
//! assert!(matches!(
//!     p.explain(2).unwrap().waiting().unwrap().hold,
//!     Some(whelm::explain::Holding::Deferral { worker: 2, .. })
//! ));
//!
//! p.handle(
//!     Input::Done { job: 1, attempt: 1 },
//!     Time(Duration::from_millis(2500)),
//! );
//! assert_eq!(
//!     p.poll(Time(Duration::from_millis(2500))),
//!     [Output::Start {
//!         job: 2,
//!         attempt: 1,
//!         worker: 2
//!     }]
//! );
//! ```
//!
//! # Speculative attempts
//!
//! With [`SpeedConfig::speculate`], a worker left idle after a poll starts a second attempt of a
//! job running on a slower worker, when it would finish sufficiently sooner. Both attempts run; the
//! first to finish completes the job and the other is stopped.
//! [Idempotence](crate#messages-and-attempts) is what makes this safe.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! # use whelm::config::{Speculate, SpeedConfig};
//! let mut p = Scheduler::new(Config {
//!     speed: SpeedConfig {
//!         speculate: Some(Speculate::default()),
//!         ..SpeedConfig::default()
//!     },
//!     ..Config::default()
//! });
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         class: "slow".into(),
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! let spec = JobSpec {
//!     work: Some(Duration::from_secs(40)),
//!     ..Default::default()
//! };
//! p.handle(Input::Submit { job: 1, spec }, Time::ORIGIN);
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // A worker four times faster joins at 1 s: done at 11 s rather than 40 s.
//! let fast = WorkerState {
//!     id: 2,
//!     class: "fast".into(),
//!     speed: 4.0,
//!     capacity: Resources::new().with(SLOTS, 1),
//!     ..Default::default()
//! };
//! p.handle(Input::Worker(fast), Time(Duration::from_secs(1)));
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(1))),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 2,
//!         worker: 2
//!     }]
//! );
//!
//! // The second attempt wins; the first is stopped.
//! p.handle(
//!     Input::Done { job: 1, attempt: 2 },
//!     Time(Duration::from_secs(11)),
//! );
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(11))),
//!     [Output::Stop {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! ```

#[cfg(doc)]
use crate::{
    config::{Defer, ScoreTerm, Speculate, SpeedConfig},
    explain::Holding,
    job::JobSpec,
    stats::{PolicyStats, WorkerLoad},
    worker::WorkerState,
};

mod estimator;
mod model;

pub use estimator::SpeedEstimator;
pub(crate) use model::{ClassId, KindId, Speeds};

/// The machine model: how a job's speed depends on the worker it runs on.
///
/// A job's run time is its [`JobSpec::work`](crate::job::JobSpec::work) over that speed;
/// [`ScoreTerm::Speed`] ranks workers by it, and
/// [`Defer`], shadow backfill and [`Speculate`] estimate run times
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
    /// # use whelm::prelude::*;
    /// # use whelm::config::SpeedConfig;
    /// # use whelm::job::JobId;
    /// # use whelm::speed::Timing;
    /// # use whelm::worker::WorkerId;
    /// # /// A scheduler with `timing` and two one-slot workers: 1 of class "x" and 2 of class "y",
    /// # /// each reporting speed 1.
    /// # fn two_classes(timing: Timing) -> Scheduler {
    /// #     let mut s = Scheduler::new(Config {
    /// #         speed: SpeedConfig { timing, ..SpeedConfig::default() },
    /// #         ..Config::default()
    /// #     });
    /// #     for (id, class) in [(1, "x"), (2, "y")] {
    /// #         let capacity = Resources::new().with(SLOTS, 1);
    /// #         let w = WorkerState { id, class: class.into(), capacity, ..Default::default() };
    /// #         s.handle(Input::Worker(w), Time::ORIGIN);
    /// #     }
    /// #     s
    /// # }
    /// # /// Where a lone job goes, both workers free, and that it then finishes at once.
    /// # fn place(s: &mut Scheduler, job: JobId, spec: JobSpec, now: Time) -> WorkerId {
    /// #     s.handle(Input::Submit { job, spec }, now);
    /// #     let [Output::Start { worker, .. }] = s.poll(now)[..] else { panic!() };
    /// #     s.handle(Input::Done { job, attempt: 1 }, now);
    /// #     worker
    /// # }
    /// let fast = WorkerState {
    ///     id: 2,
    ///     class: "y".into(),
    ///     speed: 4.0,
    ///     capacity: Resources::new().with(SLOTS, 1),
    ///     ..Default::default()
    /// };
    /// let mut s = two_classes(Timing::Identical);
    /// s.handle(Input::Worker(fast.clone()), Time::ORIGIN);
    /// assert_eq!(place(&mut s, 0, JobSpec::default(), Time::ORIGIN), 1);
    /// assert_eq!(s.stats().workers[1].speed, 1.0);
    /// // The default, related machines at their reported speeds, prefers it.
    /// let mut s = two_classes(Timing::default());
    /// s.handle(Input::Worker(fast), Time::ORIGIN);
    /// assert_eq!(place(&mut s, 0, JobSpec::default(), Time::ORIGIN), 2);
    /// assert_eq!(s.stats().workers[1].speed, 4.0);
    /// ```
    Identical,
    /// Uniformly related machines (Q): every job runs at its worker's speed, the reported
    /// [`WorkerState::speed`](crate::worker::WorkerState::speed) or, with `learn`, an estimate
    /// learned per worker ([`SpeedEstimator`]: its class as prior, the reported speed as the
    /// class's prior). [`Timing::Identical`] shows reported speeds, and [`Timing::learned`] learned
    /// ones.
    Related {
        /// Learn speeds instead of trusting the reported ones.
        learn: Option<Learn>,
    },
    /// Unrelated machines (R): a job's speed depends on its [`kind`](crate::job::JobSpec::kind) as
    /// well as its worker. On worker `w` it is `w`'s speed as [`Timing::Related`] learns it, from
    /// jobs of every kind, times the kind's factor on `w`'s class: how much faster the kind runs
    /// there than the class's average job, learned per (kind, class) and shrunk towards 1. A new
    /// kind, or a job without one, runs at the related speed; a worker slow for its class is slow
    /// for every kind. Kinds are interned for good, so they should be a small set (the job's
    /// algorithm, not its size).
    ///
    /// Kind "a" runs four times as fast on class "x" as on "y", and kind "b" twice as fast on "y"
    /// as on "x". After enough of each kind on each class, each kind goes to its own class; the
    /// related model, one speed per worker, sends both to "x", whose average is higher.
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use whelm::prelude::*;
    /// # use whelm::config::SpeedConfig;
    /// # use whelm::job::{Constraint, JobId};
    /// # use whelm::speed::Timing;
    /// # use whelm::worker::WorkerId;
    /// # /// A scheduler with `timing` and two one-slot workers: 1 of class "x" and 2 of class "y",
    /// # /// each reporting speed 1.
    /// # fn two_classes(timing: Timing) -> Scheduler {
    /// #     let mut s = Scheduler::new(Config {
    /// #         speed: SpeedConfig { timing, ..SpeedConfig::default() },
    /// #         ..Config::default()
    /// #     });
    /// #     for (id, class) in [(1, "x"), (2, "y")] {
    /// #         let capacity = Resources::new().with(SLOTS, 1);
    /// #         let w = WorkerState { id, class: class.into(), capacity, ..Default::default() };
    /// #         s.handle(Input::Worker(w), Time::ORIGIN);
    /// #     }
    /// #     s
    /// # }
    /// # /// Where a lone job goes, both workers free, and that it then finishes at once.
    /// # fn place(s: &mut Scheduler, job: JobId, spec: JobSpec, now: Time) -> WorkerId {
    /// #     s.handle(Input::Submit { job, spec }, now);
    /// #     let [Output::Start { worker, .. }] = s.poll(now)[..] else { panic!() };
    /// #     s.handle(Input::Done { job, attempt: 1 }, now);
    /// #     worker
    /// # }
    /// let truth = |kind, worker| match (kind, worker) {
    ///     ("a", 1) => 4.0,
    ///     ("b", 2) => 2.0,
    ///     _ => 1.0,
    /// };
    /// let train = |s: &mut Scheduler| {
    ///     let (mut now, mut id) = (Time::ORIGIN, 0);
    ///     for _ in 0..20 {
    ///         for kind in ["a", "b"] {
    ///             for (w, class) in [(1, "x"), (2, "y")] {
    ///                 let spec = JobSpec {
    ///                     work: Some(Duration::from_secs(8)),
    ///                     kind: Some(kind.into()),
    ///                     constraints: vec![Constraint::require_class(class)],
    ///                     ..Default::default()
    ///                 };
    ///                 s.handle(Input::Submit { job: id, spec }, now);
    ///                 assert_eq!(s.poll(now).len(), 1);
    ///                 now += Duration::from_secs(8).div_f64(truth(kind, w));
    ///                 s.handle(Input::Done { job: id, attempt: 1 }, now);
    ///                 id += 1;
    ///             }
    ///         }
    ///     }
    ///     now
    /// };
    /// let kind = |kind: &str| JobSpec {
    ///     work: Some(Duration::from_secs(8)),
    ///     kind: Some(kind.into()),
    ///     ..Default::default()
    /// };
    ///
    /// let mut s = two_classes(Timing::unrelated());
    /// let now = train(&mut s);
    /// assert_eq!(place(&mut s, 100, kind("a"), now), 1);
    /// assert_eq!(place(&mut s, 101, kind("b"), now), 2);
    /// s.handle(Input::Submit { job: 102, spec: kind("a") }, now);
    /// let why = s.explain(102).unwrap();
    /// let classes: Vec<&str> = (why.waiting().unwrap().kind_factors.iter())
    ///     .map(|(class, _)| class.as_str())
    ///     .collect();
    /// assert_eq!(classes, ["x", "y"], "{why}");
    ///
    /// let mut s = two_classes(Timing::learned());
    /// let now = train(&mut s);
    /// assert_eq!(place(&mut s, 100, kind("a"), now), 1);
    /// assert_eq!(place(&mut s, 101, kind("b"), now), 1);
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
    /// # use std::time::Duration;
    /// # use whelm::prelude::*;
    /// # use whelm::config::SpeedConfig;
    /// # use whelm::job::{Constraint, JobId};
    /// # use whelm::speed::Timing;
    /// # use whelm::worker::WorkerId;
    /// # /// A scheduler with `timing` and two one-slot workers: 1 of class "x" and 2 of class "y",
    /// # /// each reporting speed 1.
    /// # fn two_classes(timing: Timing) -> Scheduler {
    /// #     let mut s = Scheduler::new(Config {
    /// #         speed: SpeedConfig { timing, ..SpeedConfig::default() },
    /// #         ..Config::default()
    /// #     });
    /// #     for (id, class) in [(1, "x"), (2, "y")] {
    /// #         let capacity = Resources::new().with(SLOTS, 1);
    /// #         let w = WorkerState { id, class: class.into(), capacity, ..Default::default() };
    /// #         s.handle(Input::Worker(w), Time::ORIGIN);
    /// #     }
    /// #     s
    /// # }
    /// # /// Where a lone job goes, both workers free, and that it then finishes at once.
    /// # fn place(s: &mut Scheduler, job: JobId, spec: JobSpec, now: Time) -> WorkerId {
    /// #     s.handle(Input::Submit { job, spec }, now);
    /// #     let [Output::Start { worker, .. }] = s.poll(now)[..] else { panic!() };
    /// #     s.handle(Input::Done { job, attempt: 1 }, now);
    /// #     worker
    /// # }
    /// let mut s = two_classes(Timing::learned());
    /// let mut now = Time::ORIGIN;
    /// let spec = JobSpec {
    ///     work: Some(Duration::from_secs(10)),
    ///     constraints: vec![Constraint::require_class("y")],
    ///     ..Default::default()
    /// };
    /// for job in 0..u64::from(Learn::default().min_samples) {
    ///     s.handle(Input::Submit { job, spec: spec.clone() }, now);
    ///     s.poll(now);
    ///     now += Duration::from_secs(5);
    ///     s.handle(Input::Done { job, attempt: 1 }, now);
    /// }
    /// assert!((s.stats().workers[1].speed - 2.0).abs() < 1e-9);
    /// assert_eq!(place(&mut s, 1000, JobSpec::default(), now), 2);
    /// # use whelm::speed::Learn;
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
    /// use whelm::speed::{Learn, Timing};
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

/// Online speed learning: each completed job with work `w` (its run time at speed 1) that ran for
/// `d` is a sample `ln(w / d)` of its worker's speed, averaged in log space (durations are
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
/// use std::time::Duration;
///
/// use whelm::speed::{Learn, SpeedEstimator};
///
/// let mut e = SpeedEstimator::new(Learn::default());
/// for _ in 0..100 {
///     e.observe(
///         1,
///         "gpu",
///         Duration::from_secs(10),
///         Duration::from_secs(10),
///         1.0,
///     );
///     e.observe(
///         2,
///         "gpu",
///         Duration::from_secs(10),
///         Duration::from_secs(10),
///         1.0,
///     );
///     e.observe(
///         3,
///         "gpu",
///         Duration::from_secs(10),
///         Duration::from_secs(20),
///         1.0,
///     ); // half speed
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
///     e.observe(
///         1,
///         "gpu",
///         Duration::from_secs(10),
///         Duration::from_secs(10),
///         1.0,
///     );
///     e.observe(
///         3,
///         "gpu",
///         Duration::from_secs(10),
///         Duration::from_secs(20),
///         1.0,
///     );
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
/// use std::time::Duration;
///
/// use whelm::speed::{Learn, Sharing, SpeedEstimator};
///
/// let learned = |sharing| {
///     let mut e = SpeedEstimator::new(Learn {
///         sharing,
///         min_samples: 1,
///         ..Learn::default()
///     });
///     e.observe(
///         1,
///         "cpu",
///         Duration::from_secs(10),
///         Duration::from_secs(40),
///         4.0,
///     );
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

impl Default for Sharing {
    /// Throughput linear in concurrency, never saturating: each job runs at the worker's full
    /// speed, the same as no [`Learn::sharing`].
    ///
    /// It corrects nothing, so a literal that sets one parameter models that one alone.
    fn default() -> Self {
        Self {
            k_sat: f64::INFINITY,
            alpha: 1.0,
        }
    }
}

impl Sharing {
    /// `ln` of the factor that turns an observed per-job rate at mean concurrency `k` into the
    /// worker's `speed`.
    fn log_correction(&self, k: f64) -> f64 {
        let k = k.max(1.0);
        k.ln() - self.alpha * k.min(self.k_sat.max(1.0)).ln()
    }
}
