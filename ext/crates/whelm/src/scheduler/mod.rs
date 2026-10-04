//! The one placement policy, [`Scheduler`], and how it bounds a job's wait.
//!
//! A [`Scheduler`] is a [`Policy`]: [`handle`](Policy::handle) applies an event to its state at
//! once, and [`poll`](Policy::poll) decides placements and returns every [`Output`] since the last
//! poll. A poll scans the waiting jobs in urgency order ([`Config::order`], with aged jobs first)
//! and for each job looks at every worker in turn:
//!
//! 1. the job's constraints must allow the worker ([`Strength`]);
//! 2. no hold may keep the worker from the job: a [reservation](#reservations) by another job, or
//!    the job's own [deferral](crate::speed#waiting-for-a-faster-worker) to a faster worker;
//! 3. the [`Admission`] rule must admit the job's demand there;
//!
//! and among the workers left, [`Config::score`] picks one. A job no worker takes may reserve one
//! instead. After the scan, idle fast workers may start
//! [speculative](crate::speed#speculative-attempts) second attempts of jobs running on slow ones.
//! Failed attempts come back to the queue ([`Config::retry`]).
//!
//! Two one-slot workers and three jobs: two start at once, the third when a slot frees.
//!
//! ```
//! use std::time::Duration;
//!
//! use whelm::prelude::*;
//!
//! let mut s = Scheduler::new(Config::default());
//! for w in [1, 2] {
//!     s.handle(
//!         Input::Worker(WorkerState {
//!             id: w,
//!             class: "cpu".into(),
//!             capacity: Resources::new().with(MEMORY, gb(8.0)).with(SLOTS, 1),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! let spec = JobSpec {
//!     demand: Resources::new().with(MEMORY, gb(2.0)),
//!     ..Default::default()
//! };
//! for job in 0..3 {
//!     s.handle(
//!         Input::Submit {
//!             job,
//!             spec: spec.clone(),
//!         },
//!         Time::ORIGIN,
//!     );
//! }
//! let start = |job, worker| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker,
//! };
//! assert_eq!(s.poll(Time::ORIGIN), [start(0, 1), start(1, 2)]);
//! let why = s.explain(2).unwrap();
//! assert!(
//!     why.to_string().contains("slots full on 2 worker(s)"),
//!     "{why}"
//! );
//! s.handle(
//!     Input::Done { job: 0, attempt: 1 },
//!     Time(Duration::from_secs(10)),
//! );
//! assert_eq!(s.poll(Time(Duration::from_secs(10))), [start(2, 1)]);
//! ```
//!
//! # Time
//!
//! The policy reads time only from the `now` passed with each call, and some of its decisions
//! depend on how long a job has waited. Two mechanisms bound waiting under strict priority order:
//! aging and reservations.
//!
//! ## Aging
//!
//! [`Config::age_limit`]: a job that has waited that long becomes more urgent than every job that
//! has not, oldest first. Below, job 2 waits behind a running job; a more urgent job 3 arrives
//! later. When the worker frees at 150 s, job 2 has waited past the 100 s limit and goes first;
//! without aging, job 3 would.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! # use whelm::job::JobId;
//! /// The job that runs after job 1, under an age limit.
//! fn second(age_limit: Option<Duration>) -> JobId {
//!     let mut p = Scheduler::new(Config {
//!         age_limit,
//!         reservations: None,
//!         ..Config::default()
//!     });
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id: 1,
//!             capacity: Resources::new().with(SLOTS, 1),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//!     for job in [1, 2] {
//!         p.handle(
//!             Input::Submit {
//!                 job,
//!                 spec: JobSpec::default(),
//!             },
//!             Time::ORIGIN,
//!         );
//!     }
//!     p.poll(Time::ORIGIN);
//!     let urgent = JobSpec {
//!         priority: Some(-1),
//!         ..Default::default()
//!     };
//!     p.handle(
//!         Input::Submit {
//!             job: 3,
//!             spec: urgent,
//!         },
//!         Time(Duration::from_secs(50)),
//!     );
//!     p.handle(
//!         Input::Done { job: 1, attempt: 1 },
//!         Time(Duration::from_secs(150)),
//!     );
//!     match p.poll(Time(Duration::from_secs(150)))[..] {
//!         [Output::Start { job, .. }] => job,
//!         ref out => panic!("{out:?}"),
//!     }
//! }
//! assert_eq!(second(Some(Duration::from_secs(100))), 2);
//! assert_eq!(second(None), 3);
//! ```
//!
//! ## Reservations
//!
//! [`Config::reservations`]: a large job can starve while smaller ones keep filling the space it
//! needs. The most urgent job that has waited [`reserve_after`](Reservations::reserve_after) and is
//! admitted nowhere reserves a worker, which then takes no other job until the holder is placed.
//! Every other worker keeps taking less urgent jobs (backfill).
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! # use whelm::explain::{Holding, Verdict};
//! # use whelm::stats::ReservationInfo;
//! let mut p = Scheduler::new(Config::default()); // reserve after 60 s
//! let capacity = Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 3);
//! let worker = WorkerState {
//!     id: 1,
//!     capacity,
//!     ..Default::default()
//! };
//! p.handle(Input::Worker(worker), Time::ORIGIN);
//! let job = |job, size| {
//!     let demand = Resources::new().with(MEMORY, gb(size));
//!     Input::Submit {
//!         job,
//!         spec: JobSpec {
//!             demand,
//!             ..Default::default()
//!         },
//!     }
//! };
//! let start = |job| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker: 1,
//! };
//! let done = |job| Input::Done { job, attempt: 1 };
//!
//! // Big job 9 needs 8 GB; small jobs 1 and 2 get in first.
//! for input in [job(1, 4.0), job(9, 8.0), job(2, 4.0)] {
//!     p.handle(input, Time::ORIGIN);
//! }
//! assert_eq!(p.poll(Time::ORIGIN), [start(1), start(2)]);
//!
//! // A small job frees 4 GB; that is not enough for job 9, and another small job takes it.
//! p.handle(done(1), Time(Duration::from_secs(30)));
//! p.handle(job(3, 4.0), Time(Duration::from_secs(30)));
//! assert_eq!(p.poll(Time(Duration::from_secs(30))), [start(3)]);
//!
//! // At 60 s, job 9 reserves the worker.
//! assert!(p.poll(Time(Duration::from_secs(60))).is_empty());
//! let reservation = ReservationInfo {
//!     job: 9,
//!     worker: 1,
//!     since: Time(Duration::from_secs(60)),
//! };
//! assert_eq!(p.stats().reservations, [reservation]);
//! let hold = p.explain(9).unwrap().waiting().unwrap().hold.clone();
//! assert!(matches!(hold, Some(Holding::Reservation { worker: 1, .. })));
//!
//! // Small jobs no longer get in, although they would fit.
//! p.handle(done(2), Time(Duration::from_secs(70)));
//! p.handle(job(4, 4.0), Time(Duration::from_secs(70)));
//! assert!(p.poll(Time(Duration::from_secs(70))).is_empty());
//! assert_eq!(
//!     p.explain(4).unwrap().waiting().unwrap().workers,
//!     [(1, Verdict::Reserved { by: 9 })]
//! );
//!
//! // Once enough has drained, the holder runs.
//! p.handle(done(3), Time(Duration::from_secs(90)));
//! assert_eq!(p.poll(Time(Duration::from_secs(90))), [start(9)]);
//! assert_eq!(p.stats().last_dispatch_holders, [9]);
//! ```
//!
//! A drained worker idles slots. With [`shadow_backfill`](Reservations::shadow_backfill), the
//! reserved worker still takes jobs expected to finish before the holder could start (its *shadow
//! time*, from the running jobs' [`work`](JobSpec::work) estimates), which costs the holder
//! nothing. Here the holder can start when job 1 ends at 100 s: job 3 would end at 90 s and
//! backfills, job 2 would end at 110 s and does not.
//!
//! ```
//! # use std::time::Duration;
//! #
//! # use whelm::prelude::*;
//! # use whelm::config::Reservations;
//! let reservations = Reservations {
//!     shadow_backfill: true,
//!     ..Reservations::default()
//! };
//! let mut p = Scheduler::new(Config {
//!     reservations: Some(reservations),
//!     ..Config::default()
//! });
//! let worker = WorkerState {
//!     id: 1,
//!     capacity: Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 3),
//!     ..Default::default()
//! };
//! p.handle(Input::Worker(worker), Time::ORIGIN);
//! let spec = |size, work| JobSpec {
//!     demand: Resources::new().with(MEMORY, gb(size)),
//!     work: Some(Duration::from_secs(work)),
//!     ..Default::default()
//! };
//!
//! p.handle(
//!     Input::Submit {
//!         job: 1,
//!         spec: spec(4.0, 100),
//!     },
//!     Time::ORIGIN,
//! );
//! p.handle(
//!     Input::Submit {
//!         job: 9,
//!         spec: spec(8.0, 100),
//!     },
//!     Time::ORIGIN,
//! );
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! let t = Time(Duration::from_secs(60));
//! p.handle(
//!     Input::Submit {
//!         job: 2,
//!         spec: spec(4.0, 50),
//!     },
//!     t,
//! );
//! p.handle(
//!     Input::Submit {
//!         job: 3,
//!         spec: spec(4.0, 30),
//!     },
//!     t,
//! );
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(60))),
//!     [Output::Start {
//!         job: 3,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! assert_eq!(p.stats().reservations[0].job, 9);
//! ```
//!
//! ## Holds and wakeups
//!
//! Reservations and voluntary waits for a faster worker ([`Defer`], in the
//! [`speed`](crate::speed#waiting-for-a-faster-worker) chapter) are both *holds*: a worker kept
//! from a job that it might admit. Holds, aging and reservation thresholds make the passing of time
//! matter even when no event arrives, so the policy says when it next needs a poll:
//! [`next_wakeup`](Policy::next_wakeup). A caller sleeps until the earlier of its next event and
//! that time. Here a job that fits nowhere has two deadlines: it may reserve at
//! [`reserve_after`](Reservations::reserve_after) and it ages at [`DEFAULT_AGE_LIMIT`].
//!
//! ```
//! # use whelm::prelude::*;
//! # use whelm::config::{DEFAULT_AGE_LIMIT, Reservations};
//! let mut p = Scheduler::new(Config::default());
//! let capacity = Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 2);
//! let worker = WorkerState {
//!     id: 1,
//!     capacity,
//!     ..Default::default()
//! };
//! p.handle(Input::Worker(worker), Time::ORIGIN);
//! let spec = JobSpec {
//!     demand: Resources::new().with(MEMORY, gb(6.0)),
//!     ..Default::default()
//! };
//! p.handle(
//!     Input::Submit {
//!         job: 1,
//!         spec: spec.clone(),
//!     },
//!     Time::ORIGIN,
//! ); // runs for ever
//! p.handle(Input::Submit { job: 2, spec }, Time::ORIGIN);
//! p.poll(Time::ORIGIN);
//!
//! // No events arrive: poll whenever the policy asks to.
//! let mut wakeups = Vec::new();
//! while let Some(t) = p.next_wakeup() {
//!     wakeups.push(t);
//!     p.poll(t);
//! }
//! let reserve_after = Reservations::default().reserve_after;
//! assert_eq!(
//!     wakeups,
//!     [
//!         Time::ORIGIN + reserve_after,
//!         Time::ORIGIN + DEFAULT_AGE_LIMIT
//!     ]
//! );
//! assert_eq!(p.stats().reservations[0].job, 2);
//! ```

mod attempts;
mod holds;
mod order;
mod placement;
mod report;
#[cfg(test)]
mod tests;
mod timing;

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fmt,
    time::Duration,
};

use holds::Hold;
use order::{Key, dedup};

use crate::{
    admission::{Admission, ProductionAdmission, WorkerAmounts, WorkerView},
    config::Config,
    explain::Explanation,
    job::{JobId, JobSpec},
    policy::{Attempt, Input, Output, Policy, Tried},
    resources::{Dense, Resource},
    speed::{ClassId, KindId, Speeds},
    stats::PolicyStats,
    time::Time,
    worker::{WorkerId, WorkerState},
};
#[cfg(doc)]
use crate::{
    config::{DEFAULT_AGE_LIMIT, Defer, GroupOrder, Reservations},
    job::Strength,
};

/// A worker's reported state with the scheduler's bookkeeping of it.
#[derive(Clone, Debug)]
struct Worker {
    state: WorkerState,
    /// `state`'s amounts over the declared resources.
    amounts: WorkerAmounts,
    /// The demands of the live attempts here.
    placed: Dense,
    /// The jobs with a live attempt here, and that attempt (a job has at most one per worker).
    jobs: BTreeMap<JobId, Attempt>,
    /// The job holding a [`Hold::Reserve`] on this worker, if any.
    reserved_for: Option<JobId>,
    /// `state.class`, interned.
    class: ClassId,
    /// Speed for a job of no particular kind: learned, as reported, or 1
    /// ([`crate::speed::Timing`]).
    speed: f64,
    /// `∫ running dt`, in seconds, up to `occ_at` (mean concurrency over a job's run, for
    /// learning).
    occ: f64,
    occ_at: Time,
}

impl Worker {
    /// The worker as the admission rule sees it, under the declaration `resources`.
    fn view<'a>(&'a self, resources: &'a [Resource]) -> WorkerView<'a> {
        WorkerView::from_parts(
            resources,
            &self.state,
            Cow::Borrowed(&self.amounts),
            Cow::Borrowed(&self.placed),
            self.running(),
        )
    }

    /// Live attempts here.
    fn running(&self) -> usize {
        self.jobs.len()
    }
}

/// A job across its attempts. A retry requeues it with the same `key` and `since`, so it keeps
/// its place and its age.
#[derive(Clone, Debug)]
struct Job {
    id: JobId,
    spec: JobSpec,
    /// `spec.demand` over the declared resources, with the default demands filled in.
    demand: Dense,
    /// `spec.kind`, interned if the timing distinguishes kinds.
    kind: Option<KindId>,
    key: Key,
    since: Time,
    /// Attempts started so far: the last attempt's number.
    attempts: Attempt,
    /// Failed attempts.
    tried: Vec<Tried>,
    /// Workers its failed attempts ran on, avoided softly like a [`Strength::Avoid`].
    retry_avoid: Vec<WorkerId>,
    /// Speculative attempts started.
    speculated: u32,
}

/// One live attempt of a running job.
#[derive(Clone, Debug)]
struct Run {
    attempt: Attempt,
    worker: WorkerId,
    started: Time,
    /// The worker's `occ` when it started.
    occ0: f64,
}

/// A job with at least one live attempt.
#[derive(Clone, Debug)]
struct Running {
    job: Job,
    live: Vec<Run>,
}

/// Bring a worker's concurrency integral up to `now`.
fn tick_occ(w: &mut Worker, now: Time) {
    if now > w.occ_at {
        w.occ += w.running() as f64 * (now - w.occ_at).as_secs_f64();
        w.occ_at = now;
    }
}

/// The placement policy: list scheduling with admission, reservations and backfill, configured
/// by a [`Config`].
///
/// - Jobs are considered in [`Config::order`], aged jobs first ([`Config::age_limit`]).
/// - A job takes a worker only if no more urgent waiting job is admitted there; among the
///   workers that admit it, the one [`Config::score`] ranks first.
/// - [`Config::reservations`] drain a worker for a starving job; every other worker keeps
///   admitting less urgent jobs.
/// - Failed attempts are retried ([`Config::retry`]) and, with
///   [`SpeedConfig::speculate`](crate::config::SpeedConfig::speculate), idle fast workers run
///   second attempts of jobs on slow ones.
///
/// The examples below share the hidden helpers `worker(id, slots, bytes)`, `job(bytes)` and
/// `start(job, attempt, worker)`, which build a [`WorkerState`], a [`JobSpec`] demanding `bytes`
/// of memory and an [`Output::Start`].
///
/// # The scan
///
/// Each [`poll`](Policy::poll) scans waiting jobs in order and gives each one a worker if any
/// admits it. Because the scan is in urgency order and admission is monotone in load, a job is
/// placed on a worker only if every more urgent waiting job was refused there -- the priority
/// invariant -- without any explicit check. The one event that can make an already-refused worker
/// admissible mid-scan is the release of a hold (a reservation whose holder is placed); the scan
/// restarts from the top when that happens. Speculative attempts come after the scan, on workers
/// every waiting job was refused on.
///
/// A less urgent job that fits backfills the room a more urgent one cannot use:
///
/// ```
/// # use std::time::Duration;
/// # use whelm::prelude::*;
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     capacity: Resources::new().with(MEMORY, bytes).with(SLOTS, slots),
/// #     ..Default::default()
/// # };
/// # let job = |bytes| JobSpec { demand: Resources::new().with(MEMORY, bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// let mut s = Scheduler::new(Config::default());
/// s.handle(Input::Worker(worker(1, 4, 100)), Time::ORIGIN);
/// s.handle(Input::Submit { job: 0, spec: job(70) }, Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(0, 1, 1)]);
/// let urgent = JobSpec {
///     priority: Some(-1),
///     ..job(50)
/// };
/// s.handle(Input::Submit { job: 1, spec: urgent }, Time(Duration::from_secs(1)));
/// s.handle(Input::Submit { job: 2, spec: job(20) }, Time(Duration::from_secs(1)));
/// assert_eq!(s.poll(Time(Duration::from_secs(1))), [start(2, 1, 1)]);
/// let why = s.explain(1).unwrap();
/// assert!(why.to_string().contains("memory short on 1 worker(s)"), "{why}");
/// ```
///
/// # Retries
///
/// A failed attempt sends its job back to the queue with its place and age; the retry avoids,
/// softly, the workers it failed on. Reports about attempts that are no longer live are ignored,
/// and cancelling a running job stops it.
///
/// ```
/// # use std::time::Duration;
/// # use whelm::prelude::*;
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     capacity: Resources::new().with(MEMORY, bytes).with(SLOTS, slots),
/// #     ..Default::default()
/// # };
/// # let job = |bytes| JobSpec { demand: Resources::new().with(MEMORY, bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// use whelm::policy::FailKind;
///
/// let mut s = Scheduler::new(Config::default());
/// s.handle(Input::Worker(worker(1, 1, 100)), Time::ORIGIN);
/// s.handle(Input::Worker(worker(2, 1, 100)), Time::ORIGIN);
/// s.handle(Input::Submit { job: 0, spec: job(10) }, Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(0, 1, 1)]);
/// let failed = Input::Failed {
///     job: 0,
///     attempt: 1,
///     kind: FailKind::Other,
///     why: "segfault".into(),
/// };
/// s.handle(failed, Time(Duration::from_secs(1)));
/// assert_eq!(s.poll(Time(Duration::from_secs(1))), [start(0, 2, 2)]);
/// s.handle(Input::Done { job: 0, attempt: 1 }, Time(Duration::from_secs(2))); // stale
/// assert_eq!((s.poll(Time(Duration::from_secs(2))), s.stats().running), (vec![], 1));
/// s.handle(Input::Cancel(0), Time(Duration::from_secs(3)));
/// let stop = Output::Stop {
///     job: 0,
///     attempt: 2,
///     worker: 2,
/// };
/// assert_eq!(s.poll(Time(Duration::from_secs(3))), [stop]);
/// ```
///
/// [`RetryConfig`](crate::config::RetryConfig) shows a job given up.
///
/// # Holds: reservations and deferral
///
/// A hold keeps a worker from a job on purpose although it might admit it. A
/// [reservation](crate::config::Reservations) keeps a worker from every job but its holder, so that
/// it drains for a job that fits nowhere. Here two workers each run a 60-byte job, a 50-byte job
/// reserves worker 1 once it has waited `reserve_after`, a small job keeps to worker 2, and the
/// holder starts when worker 1 drains:
///
/// ```
/// # use std::time::Duration;
/// # use whelm::prelude::*;
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     capacity: Resources::new().with(MEMORY, bytes).with(SLOTS, slots),
/// #     ..Default::default()
/// # };
/// # let job = |bytes| JobSpec { demand: Resources::new().with(MEMORY, bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// let mut s = Scheduler::new(Config::default());
/// s.handle(Input::Worker(worker(1, 4, 100)), Time::ORIGIN);
/// s.handle(Input::Worker(worker(2, 4, 100)), Time::ORIGIN);
/// s.handle(Input::Submit { job: 10, spec: job(60) }, Time::ORIGIN);
/// s.handle(Input::Submit { job: 11, spec: job(60) }, Time::ORIGIN);
/// s.handle(Input::Submit { job: 1, spec: job(50) }, Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(10, 1, 1), start(11, 1, 2)]);
/// let reserve_after = Config::default().reservations.unwrap().reserve_after;
/// let reserve_at = Time::ORIGIN + reserve_after;
/// assert_eq!(s.next_wakeup(), Some(reserve_at));
/// assert_eq!(s.poll(reserve_at), []);
/// let reservation = &s.stats().reservations[0];
/// assert_eq!((reservation.job, reservation.worker), (1, 1));
/// s.handle(Input::Submit { job: 20, spec: job(5) }, Time(Duration::from_secs(70)));
/// assert_eq!(s.poll(Time(Duration::from_secs(70))), [start(20, 1, 2)]);
/// assert!(matches!(
///     s.explain(1).unwrap().waiting().unwrap().hold,
///     Some(whelm::explain::Holding::Reservation { worker: 1, .. })
/// ));
/// s.handle(Input::Done { job: 10, attempt: 1 }, Time(Duration::from_secs(100)));
/// assert_eq!(s.poll(Time(Duration::from_secs(100))), [start(1, 1, 1)]);
/// assert_eq!(s.stats().last_dispatch_holders, [1]);
/// ```
///
/// A [deferral](crate::config::Defer) is a job's own hold: it declines a slow free worker to wait
/// for a fast busy one, when it expects to finish sooner that way. Here job 1, of 40 s of work,
/// waits 2.5 s for the worker four times as fast instead of starting on the slow one:
///
/// ```
/// # use std::time::Duration;
/// #
/// # use whelm::prelude::*;
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     capacity: Resources::new().with(MEMORY, bytes).with(SLOTS, slots),
/// #     ..Default::default()
/// # };
/// # let job = |bytes| JobSpec { demand: Resources::new().with(MEMORY, bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// use whelm::config::{Defer, SpeedConfig};
///
/// let mut s = Scheduler::new(Config {
///     speed: SpeedConfig {
///         defer: Some(Defer::default()),
///         ..SpeedConfig::default()
///     },
///     ..Config::default()
/// });
/// let fast = WorkerState {
///     speed: 4.0,
///     ..worker(2, 1, 100)
/// };
/// s.handle(Input::Worker(worker(1, 1, 100)), Time::ORIGIN);
/// s.handle(Input::Worker(fast), Time::ORIGIN);
/// let work = |work| JobSpec {
///     work: Some(Duration::from_secs(work)),
///     ..job(10)
/// };
/// s.handle(Input::Submit { job: 0, spec: work(10) }, Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(0, 1, 2)]);
/// s.handle(Input::Submit { job: 1, spec: work(40) }, Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), []);
/// assert_eq!(s.stats().deferred, [(1, 2, Time(Duration::from_millis(2500)))]);
/// assert!(matches!(
///     s.explain(1).unwrap().waiting().unwrap().hold,
///     Some(whelm::explain::Holding::Deferral { worker: 2, .. })
/// ));
/// s.handle(Input::Done { job: 0, attempt: 1 }, Time(Duration::from_millis(2500)));
/// assert_eq!(s.poll(Time(Duration::from_millis(2500))), [start(1, 1, 2)]);
/// ```
///
/// # Speculation
///
/// With [`Speculate`](crate::config::Speculate), a fast worker left idle after the scan starts a
/// second attempt of a long job running on a slow worker. Both run; the first to finish wins, and
/// the other is stopped:
///
/// ```
/// # use std::time::Duration;
/// #
/// # use whelm::prelude::*;
/// # let worker = |id, slots, bytes| WorkerState {
/// #     id,
/// #     capacity: Resources::new().with(MEMORY, bytes).with(SLOTS, slots),
/// #     ..Default::default()
/// # };
/// # let job = |bytes| JobSpec { demand: Resources::new().with(MEMORY, bytes), ..Default::default() };
/// # let start = |job, attempt, worker| Output::Start { job, attempt, worker };
/// use whelm::config::{Speculate, SpeedConfig};
///
/// let mut s = Scheduler::new(Config {
///     speed: SpeedConfig {
///         speculate: Some(Speculate::default()),
///         ..SpeedConfig::default()
///     },
///     ..Config::default()
/// });
/// let fast = WorkerState {
///     speed: 4.0,
///     ..worker(2, 1, 100)
/// };
/// s.handle(Input::Worker(worker(1, 1, 100)), Time::ORIGIN);
/// s.handle(Input::Worker(fast), Time::ORIGIN);
/// let work = |work| JobSpec {
///     work: Some(Duration::from_secs(work)),
///     ..job(10)
/// };
/// s.handle(Input::Submit { job: 0, spec: work(4) }, Time::ORIGIN);
/// s.handle(Input::Submit { job: 1, spec: work(100) }, Time::ORIGIN);
/// assert_eq!(s.poll(Time::ORIGIN), [start(0, 1, 2), start(1, 1, 1)]);
/// // The fast worker frees at 1 s; job 1 would end there at 26 s instead of 100 s.
/// s.handle(Input::Done { job: 0, attempt: 1 }, Time(Duration::from_secs(1)));
/// assert_eq!(s.poll(Time(Duration::from_secs(1))), [start(1, 2, 2)]);
/// s.handle(Input::Done { job: 1, attempt: 2 }, Time(Duration::from_secs(26)));
/// let stop = Output::Stop {
///     job: 1,
///     attempt: 1,
///     worker: 1,
/// };
/// assert_eq!(s.poll(Time(Duration::from_secs(26))), [stop]);
/// ```
pub struct Scheduler {
    config: Config,
    admission: Box<dyn Admission + Send>,
    workers: BTreeMap<WorkerId, Worker>,
    queue: BTreeMap<Key, JobId>,
    by_age: BTreeMap<u64, JobId>,
    waiting: HashMap<JobId, Job>,
    running: HashMap<JobId, Running>,
    /// Outputs not yet returned by `poll`.
    outbox: Vec<Output>,
    /// Group -> sequence number of its first arrival.
    groups: HashMap<u64, u64>,
    /// Holds by job.
    holds: BTreeMap<JobId, Hold>,
    next_seq: u64,
    /// The next [`Hold::Reserve`] order.
    next_reservation: u64,
    now: Time,
    placements_total: u64,
    reservations_total: u64,
    last_dispatch_holders: Vec<JobId>,
    /// Every job that deferred at some point of the last dispatch, placed later or not.
    deferred_any: Vec<JobId>,
    /// The machine model's state ([`SpeedConfig::timing`](crate::config::SpeedConfig::timing)).
    speeds: Speeds,
    /// A zero amount of every declared resource: an empty worker's load.
    zero: Dense,
}

impl fmt::Debug for Scheduler {
    /// The configuration and the job and worker counts.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scheduler")
            .field("config", &self.config)
            .field("workers", &self.workers.len())
            .field("waiting", &self.waiting.len())
            .field("running", &self.running.len())
            .finish_non_exhaustive()
    }
}

impl Scheduler {
    /// A scheduler with the production admission rule ([`ProductionAdmission`]).
    ///
    /// It starts with no workers and no jobs, so nothing is timed:
    ///
    /// ```
    /// use whelm::prelude::*;
    ///
    /// let s = Scheduler::new(Config::default());
    /// assert_eq!((s.stats().waiting, s.next_wakeup()), (0, None));
    /// ```
    ///
    /// # Panics
    ///
    /// If [`Config::resources`] declares a name twice.
    pub fn new(config: Config) -> Self {
        Self::with_admission(config, ProductionAdmission)
    }

    /// A scheduler with a custom admission rule.
    ///
    /// The rule must be monotone in load ([`Admission`] has the contract and another example).
    /// It panics as [`new`](Self::new) does. Here the production rule is kept, and workers of
    /// class "draining" take nothing new:
    ///
    /// ```
    /// use whelm::{
    ///     admission::{Admission, Amounts, ProductionAdmission, WorkerView},
    ///     prelude::*,
    /// };
    ///
    /// struct SkipDraining;
    ///
    /// impl Admission for SkipDraining {
    ///     fn admits(&self, demand: &Amounts, w: &WorkerView) -> bool {
    ///         w.state().class != "draining" && ProductionAdmission.admits(demand, w)
    ///     }
    /// }
    ///
    /// let mut s = Scheduler::with_admission(Config::default(), SkipDraining);
    /// s.handle(
    ///     Input::Worker(WorkerState {
    ///         id: 1,
    ///         class: "draining".into(),
    ///         capacity: Resources::new().with(SLOTS, 4),
    ///         ..Default::default()
    ///     }),
    ///     Time::ORIGIN,
    /// );
    /// s.handle(
    ///     Input::Worker(WorkerState {
    ///         id: 2,
    ///         class: "cpu".into(),
    ///         capacity: Resources::new().with(SLOTS, 4),
    ///         ..Default::default()
    ///     }),
    ///     Time::ORIGIN,
    /// );
    /// s.handle(
    ///     Input::Submit {
    ///         job: 0,
    ///         spec: JobSpec::default(),
    ///     },
    ///     Time::ORIGIN,
    /// );
    /// assert_eq!(
    ///     s.poll(Time::ORIGIN),
    ///     [Output::Start {
    ///         job: 0,
    ///         attempt: 1,
    ///         worker: 2
    ///     }]
    /// );
    /// ```
    pub fn with_admission(mut config: Config, admission: impl Admission + Send + 'static) -> Self {
        let resources = &config.resources;
        for (i, r) in resources.iter().enumerate() {
            assert!(
                resources[..i].iter().all(|o| o.name != r.name),
                "Config::resources declares resource {:?} twice",
                r.name
            );
        }
        config.order = dedup(&config.order);
        config.score = dedup(&config.score);
        Self {
            zero: vec![0; config.resources.len()],
            speeds: Speeds::new(config.speed.timing),
            config,
            admission: Box::new(admission),
            workers: BTreeMap::new(),
            queue: BTreeMap::new(),
            by_age: BTreeMap::new(),
            waiting: HashMap::new(),
            running: HashMap::new(),
            outbox: Vec::new(),
            groups: HashMap::new(),
            holds: BTreeMap::new(),
            next_seq: 0,
            next_reservation: 0,
            now: Time::ORIGIN,
            placements_total: 0,
            reservations_total: 0,
            last_dispatch_holders: Vec::new(),
            deferred_any: Vec::new(),
        }
    }

    /// Forget a group's first-arrival time. A later job of that group then counts as a new group.
    /// Use it when a group is known to be finished, to bound memory. Only
    /// [`GroupOrder::Arrival`] records arrivals.
    ///
    /// Group 1 arrived first, so a new job of it would run before one of group 2; once group 1
    /// is forgotten, the new job counts as arriving after group 2:
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::prelude::*;
    ///
    /// let next = |forget| {
    ///     let mut s = Scheduler::new(Config::default());
    ///     s.handle(
    ///         Input::Worker(WorkerState {
    ///             id: 1,
    ///             capacity: Resources::new().with(SLOTS, 1),
    ///             ..Default::default()
    ///         }),
    ///         Time::ORIGIN,
    ///     );
    ///     let spec = JobSpec {
    ///         group: 1,
    ///         ..Default::default()
    ///     };
    ///     s.handle(Input::Submit { job: 0, spec }, Time::ORIGIN);
    ///     s.poll(Time::ORIGIN); // job 0, the last of group 1 for now, takes the slot
    ///     if forget {
    ///         s.forget_group(1);
    ///     }
    ///     let spec = JobSpec {
    ///         group: 2,
    ///         ..Default::default()
    ///     };
    ///     s.handle(Input::Submit { job: 1, spec }, Time(Duration::from_secs(1)));
    ///     let spec = JobSpec {
    ///         group: 1,
    ///         ..Default::default()
    ///     };
    ///     s.handle(Input::Submit { job: 2, spec }, Time(Duration::from_secs(1)));
    ///     s.handle(
    ///         Input::Done { job: 0, attempt: 1 },
    ///         Time(Duration::from_secs(2)),
    ///     );
    ///     match s.poll(Time(Duration::from_secs(2)))[..] {
    ///         [Output::Start { job, .. }] => job,
    ///         ref out => panic!("{out:?}"),
    ///     }
    /// };
    /// assert_eq!(next(false), 2);
    /// assert_eq!(next(true), 1);
    /// ```
    pub fn forget_group(&mut self, group: u64) {
        self.groups.remove(&group);
    }

    /// The next time, after now, that the passing of time alone changes what `dispatch` may do:
    /// a hold lapses, a job ages, or a job waits long enough to reserve.
    fn next_wakeup(&self) -> Option<Time> {
        let now = self.now;
        // `by_age` is in submission order, so the first job whose deadline is still ahead has
        // the earliest one.
        let first_after = |wait: Duration| {
            self.by_age
                .values()
                .map(|job| self.waiting[job].since + wait)
                .find(|&t| t > now)
        };
        let aging = self.config.age_limit.and_then(first_after);
        let reserving =
            (self.config.reservations.as_ref()).and_then(|c| first_after(c.reserve_after));
        self.holds
            .values()
            .filter_map(Hold::until)
            .filter(|&t| t > now)
            .chain(aging)
            .chain(reserving)
            .min()
    }
}

impl Policy for Scheduler {
    /// Applied at once; stops and give-ups wait in the outbox for `poll`.
    fn handle(&mut self, input: Input, now: Time) {
        self.now = now;
        match input {
            Input::Submit { job, spec } => self.submit(job, spec, now),
            Input::Done { job, attempt } => self.done(job, attempt),
            Input::Failed {
                job,
                attempt,
                kind,
                why,
            } => self.failed(job, attempt, kind, why),
            Input::Cancel(job) => self.cancel(job),
            Input::Worker(w) => self.worker_update(w, now),
            Input::WorkerGone(w) => self.worker_gone(w),
        }
    }

    /// One scan in urgency order, then speculation; the outbox, then the new starts.
    fn poll(&mut self, now: Time) -> Vec<Output> {
        self.now = now;
        self.dispatch();
        self.speculate();
        std::mem::take(&mut self.outbox)
    }

    /// When a hold lapses, a job ages or a job may reserve, whichever is first.
    fn next_wakeup(&self) -> Option<Time> {
        Scheduler::next_wakeup(self)
    }

    /// Every worker's reason to refuse it, summarised.
    fn explain(&self, job: JobId) -> Option<Explanation> {
        Scheduler::explain(self, job)
    }

    /// Counters, holds and per-worker load.
    fn stats(&self) -> PolicyStats {
        Scheduler::stats(self)
    }
}
