//! Pure, deterministic, resource-aware job placement.
//!
//! This page is a guided tour of the crate, meant to be read top to bottom: each chapter builds on
//! the previous one, and every example is a test that asserts what the crate really does. The items
//! it links to hold the full semantics.
//!
//! 1. [What whelm is](#what-whelm-is)
//! 2. [Quick start](#quick-start)
//! 3. [Messages and attempts](#messages-and-attempts)
//! 4. [Resources and admission](#resources-and-admission)
//! 5. [Constraints](#constraints)
//! 6. [Ordering](#ordering)
//! 7. [Time](#time)
//! 8. [Speed](#speed)
//! 9. [Dependencies](#dependencies)
//! 10. [Threads](#threads)
//! 11. [Logging and replay](#logging-and-replay)
//! 12. [Testing your integration](#testing-your-integration)
//! 13. [Where to look next](#where-to-look-next)
//!
//! # What whelm is
//!
//! whelm decides which job runs on which worker, and when, so as to neither overwhelm the machines
//! (too many jobs, or too much memory, on one worker) nor underwhelm them (idle slots while jobs
//! wait). Workers come and go, report their capacity and usage, and differ in speed; jobs arrive
//! online with resource estimates, constraints and, optionally, dependencies.
//!
//! The core is a deterministic state machine. It does no I/O, starts no threads and reads no clock:
//! the caller feeds it [`Input`]s (a worker joined, a job is ready, an attempt finished), each
//! stamped with the caller's notion of "now", and acts on the [`Output`]s it returns (start this
//! job on that worker, stop that attempt, this job failed for good). The same inputs at the same
//! times always produce the same outputs, which makes a run replayable from a log and every
//! behaviour testable without a cluster. The [`Policy`] trait is that state machine's interface,
//! and [`Scheduler`] is the one implementation of placement; everything else in the crate wraps a
//! policy:
//!
//! - [`DagScheduler`] adds dependencies between jobs,
//! - [`log::Logged`] records every input and output for replay,
//! - [`SharedPolicy`] puts a policy behind a lock for callers with a thread per task.
//!
//! # Quick start
//!
//! A worker joins, a job is submitted, and [`poll`](Policy::poll) says where to run it. Reporting
//! the attempt [`Done`](Input::Done) frees its slot and the scheduler forgets the job.
//!
//! ```
//! use std::time::Duration;
//!
//! use whelm::{
//!     Config, Input, JobSpec, MEMORY, Output, Policy, Resources, SLOTS, Scheduler, Time,
//!     WorkerState, gb,
//! };
//!
//! let mut policy = Scheduler::new(Config::default());
//!
//! // Worker 1 joins: class "cpu", 4 execution slots, 16 GB of host memory.
//! let worker = WorkerState {
//!     id: 1,
//!     class: "cpu".into(),
//!     capacity: Resources::new().with(MEMORY, gb(16.0)).with(SLOTS, 4),
//!     ..Default::default()
//! };
//! policy.handle(Input::Worker(worker), Time::ORIGIN);
//!
//! // Job 7 becomes ready: it expects to use 2 GB and belongs to group 3.
//! let job = JobSpec {
//!     id: 7,
//!     demand: Resources::new().with(MEMORY, gb(2.0)),
//!     group: 3,
//!     ..Default::default()
//! };
//! policy.handle(Input::Submit(job), Time::ORIGIN);
//!
//! // After a batch of inputs, poll: start attempt 1 of job 7 on worker 1.
//! assert_eq!(
//!     policy.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 7,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // Thirty seconds later the worker reports success. Nothing is left to do.
//! let later = Time::ORIGIN + Duration::from_secs(30);
//! policy.handle(Input::Done { job: 7, attempt: 1 }, later);
//! assert!(policy.poll(later).is_empty());
//! assert_eq!(policy.explain(7), None); // the job is forgotten
//! assert_eq!(policy.stats().placements_total, 1);
//! ```
//!
//! Every input and poll carries the current [`Time`]: a point on the caller's clock, the
//! [`Duration`](std::time::Duration) since an origin the caller picks (here the start of the run,
//! [`Time::ORIGIN`]). A point is written `Time::ORIGIN + span`, or `Time(span)` as the examples
//! below do, and its `.0` is that span again. The policy only compares times and measures the
//! spans between them, which are `Duration`s, as are all the spans in the crate (work estimates,
//! waiting bounds). Moving a `Time` by a `Duration` and subtracting two `Time`s saturate rather
//! than panic.
//!
//! Workers and jobs, like configurations, are plain structs written as literals: name the fields
//! that matter and take the rest from [`Default`]. Each type's `Default` documents what the
//! omitted fields mean; a [`JobSpec`] needs little more than an id, a [`WorkerState`] an id and
//! its capacity.
//!
//! That is the whole protocol: [`handle`](Policy::handle) every event as it happens, then
//! [`poll`](Policy::poll) and act on each output. `handle` applies an input at once but never
//! places anything; placement happens in `poll`, which also returns the outputs earlier inputs
//! caused. A real caller matches on the outputs; the next example sends each start to a stand-in
//! for the workers and reports completions back.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
//! # };
//! let mut policy = Scheduler::new(Config::default());
//! let worker = WorkerState {
//!     id: 1,
//!     capacity: Resources::new().with(SLOTS, 2),
//!     ..Default::default()
//! };
//! policy.handle(Input::Worker(worker), Time::ORIGIN);
//! for id in 1..=3 {
//!     let job = JobSpec {
//!         id,
//!         ..Default::default()
//!     };
//!     policy.handle(Input::Submit(job), Time::ORIGIN);
//! }
//!
//! // The caller's side: attempts sent to workers and not yet answered.
//! let mut in_flight = Vec::new();
//! let mut now = Time::ORIGIN;
//! while now < Time(Duration::from_secs(10)) {
//!     for out in policy.poll(now) {
//!         match out {
//!             Output::Start {
//!                 job,
//!                 attempt,
//!                 worker,
//!             } => in_flight.push((job, attempt, worker)),
//!             Output::Stop { job, attempt, .. } => {
//!                 in_flight.retain(|r| (r.0, r.1) != (job, attempt))
//!             }
//!             Output::GaveUp(g) => panic!("job {} failed too often", g.job),
//!             _ => unreachable!("only the DAG layer emits the other outputs"),
//!         }
//!     }
//!     // Each second, the oldest attempt in flight finishes.
//!     now += Duration::from_secs(1);
//!     if !in_flight.is_empty() {
//!         let (job, attempt, _) = in_flight.remove(0);
//!         policy.handle(Input::Done { job, attempt }, now);
//!     }
//! }
//! // Two slots: jobs 1 and 2 started at once, job 3 when job 1 finished.
//! assert_eq!(policy.stats().placements_total, 3);
//! assert!(in_flight.is_empty());
//! ```
//!
//! # Messages and attempts
//!
//! The contract between caller and policy rests on one assumption, stated on [`Policy`]: **jobs are
//! idempotent**. Running a job twice is harmless, and its first completion wins. That frees the
//! policy to retry a failed job elsewhere, to run a second, speculative copy of a slow one, and to
//! ignore any message about an attempt it no longer tracks.
//!
//! Every start carries an [`Attempt`] number, 1 for the first, and [`Input::Done`] and
//! [`Input::Failed`] name the attempt they report. Messages are therefore safe to repeat or to
//! deliver late: a duplicate submission, or a report about an attempt that is no longer live, is
//! ignored.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 4),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! let job = JobSpec {
//!     id: 1,
//!     ..Default::default()
//! };
//!
//! p.handle(Input::Submit(job.clone()), Time::ORIGIN);
//! p.handle(Input::Submit(job.clone()), Time::ORIGIN); // already waiting: ignored
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! p.handle(Input::Submit(job.clone()), Time(Duration::from_secs(1))); // already running: ignored
//! assert!(p.poll(Time(Duration::from_secs(1))).is_empty());
//!
//! p.handle(
//!     Input::Done { job: 1, attempt: 1 },
//!     Time(Duration::from_secs(2)),
//! );
//! p.handle(
//!     Input::Done { job: 1, attempt: 1 },
//!     Time(Duration::from_secs(2)),
//! ); // no longer live: ignored
//! assert!(p.poll(Time(Duration::from_secs(2))).is_empty());
//!
//! // Once complete, the id is free again: a new submission is a new job.
//! p.handle(Input::Submit(job), Time(Duration::from_secs(3)));
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(3))),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! ```
//!
//! ## Failures and retries
//!
//! A failed attempt puts its job back in the queue, in the place and with the age it had, and the
//! retry softly avoids the workers the job failed on: they are used only while no other live worker
//! the job may run on remains. Here the retry goes to worker 2, then waits for it rather than
//! returning to worker 1, and goes back to worker 1 only once it has failed on both.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, FailKind, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id,
//!             capacity: Resources::new().with(SLOTS, 1),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! let fail = |job, attempt, why: &str| Input::Failed {
//!     job,
//!     attempt,
//!     kind: FailKind::Other,
//!     why: why.into(),
//! };
//!
//! for id in [5, 6] {
//!     let job = JobSpec {
//!         id,
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(job), Time::ORIGIN);
//! }
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [
//!         Output::Start {
//!             job: 5,
//!             attempt: 1,
//!             worker: 1
//!         },
//!         Output::Start {
//!             job: 6,
//!             attempt: 1,
//!             worker: 2
//!         },
//!     ]
//! );
//!
//! // Job 5 fails on worker 1. Worker 2 is busy, but job 5 waits for it.
//! p.handle(fail(5, 1, "segfault"), Time(Duration::from_secs(10)));
//! assert!(p.poll(Time(Duration::from_secs(10))).is_empty());
//! assert_eq!(
//!     p.explain(5).unwrap().to_string(),
//!     "job 5 (demand [slots 1], group 0) waiting 10s, 0 more urgent job(s) waiting; failed 1 \
//!      time(s), last on worker 1 (Other: segfault); slots full on 1 worker(s); 1 worker(s) \
//!      excluded by its constraints"
//! );
//!
//! // Worker 2 frees up: attempt 2 runs there.
//! p.handle(Input::Done { job: 6, attempt: 1 }, Time(Duration::from_secs(20)));
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(20))),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 2,
//!         worker: 2
//!     }]
//! );
//!
//! // It fails there too. Every worker is now avoided, so the avoidance lapses.
//! p.handle(fail(5, 2, "segfault"), Time(Duration::from_secs(30)));
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(30))),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 3,
//!         worker: 1
//!     }]
//! );
//! ```
//!
//! After [`RetryConfig::max_attempts`] failed rounds the policy gives up: it emits
//! [`Output::GaveUp`] with every failed attempt ([`Tried`]) and forgets the job. A give-up is
//! [`retryable`](GaveUp::retryable) when every attempt ran out of device memory, the one failure
//! that a smaller job, or a later attempt on a less loaded worker, may avoid.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, FailKind, GaveUp, Input, JobSpec, Output, Policy, Resources, RetryConfig, SLOTS,
//! #     Scheduler, Time, Tried, WorkerState,
//! # };
//! let config = Config {
//!     retry: RetryConfig { max_attempts: 2 },
//!     ..Config::default()
//! };
//! let mut p = Scheduler::new(config);
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         class: "gpu".into(),
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 5,
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//!
//! let oom = |attempt| Input::Failed {
//!     job: 5,
//!     attempt,
//!     kind: FailKind::DeviceOom,
//!     why: "out of device memory".into(),
//! };
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! p.handle(oom(1), Time(Duration::from_secs(1)));
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(1))),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 2,
//!         worker: 1
//!     }]
//! );
//! p.handle(oom(2), Time(Duration::from_secs(2)));
//!
//! let tried = Tried {
//!     worker: 1,
//!     kind: FailKind::DeviceOom,
//!     why: "out of device memory".into(),
//! };
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(2))),
//!     [Output::GaveUp(GaveUp {
//!         job: 5,
//!         tried: vec![tried.clone(), tried],
//!         retryable: true
//!     })]
//! );
//! assert_eq!(p.explain(5), None);
//! ```
//!
//! ## Cancellation, stale reports and lost workers
//!
//! [`Input::Cancel`] drops a waiting job silently and stops a running one: the policy releases the
//! attempt's resources at once and asks the caller to [`Stop`](Output::Stop) it.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! let job = |id| JobSpec {
//!     id,
//!     ..Default::default()
//! };
//! p.handle(Input::Submit(job(1)), Time::ORIGIN);
//! p.handle(Input::Submit(job(2)), Time::ORIGIN);
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! p.handle(Input::Cancel(2), Time(Duration::from_secs(1))); // waiting: dropped
//! p.handle(Input::Cancel(1), Time(Duration::from_secs(1))); // running: stopped
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(1))),
//!     [Output::Stop {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! let stats = p.stats();
//! assert_eq!((stats.waiting, stats.running), (0, 0));
//! ```
//!
//! A report about an attempt the policy has already written off is ignored. Here the caller times
//! attempt 1 out and the retry starts; when attempt 1's success arrives after all, the job is still
//! running attempt 2.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, FailKind, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id,
//!             capacity: Resources::new().with(SLOTS, 1),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 5,
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! let why = "no reply".into();
//! let timeout = Input::Failed {
//!     job: 5,
//!     attempt: 1,
//!     kind: FailKind::Timeout,
//!     why,
//! };
//! p.handle(timeout, Time(Duration::from_secs(60)));
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(60))),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 2,
//!         worker: 2
//!     }]
//! );
//!
//! p.handle(Input::Done { job: 5, attempt: 1 }, Time(Duration::from_secs(61))); // stale: ignored
//! assert!(p.poll(Time(Duration::from_secs(61))).is_empty());
//! assert_eq!(
//!     p.explain(5).unwrap().status,
//!     whelm::Status::Running {
//!         attempts: vec![(2, 2)]
//!     }
//! );
//! ```
//!
//! When a worker leaves ([`Input::WorkerGone`]), each attempt on it fails as if reported with
//! [`FailKind::LinkDied`], and is retried like any other failure. The caller does not resubmit.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id,
//!             capacity: Resources::new().with(SLOTS, 1),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 5,
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! p.handle(Input::WorkerGone(1), Time(Duration::from_secs(5)));
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(5))),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 2,
//!         worker: 2
//!     }]
//! );
//! assert_eq!(p.stats().workers.len(), 1);
//! ```
//!
//! # Resources and admission
//!
//! A configuration declares the resources its workers have and its jobs use, as a list of
//! [`Resource`]s in [`Config::resources`]. A `Resource` describes one kind of resource and is
//! identified by its name; it is usually a constant, and the default declaration is three of them:
//! host memory ([`MEMORY`]) and device memory ([`DEVICE_MEMORY`]), in bytes, and execution slots
//! ([`SLOTS`]). A [`Resources`] value holds amounts of resources keyed by name, built up with
//! [`with`](Resources::with): a job's [`demand`](JobSpec::demand) is what it is expected to use,
//! and a worker's [`capacity`](WorkerState::capacity) is what it has. A resource it leaves out has
//! amount zero, and [`gb`] turns gigabytes into bytes.
//!
//! ```
//! use whelm::{DEVICE_MEMORY, MEMORY, Resources, SLOTS, gb};
//!
//! let demand = Resources::new()
//!     .with(MEMORY, gb(6.0))
//!     .with(DEVICE_MEMORY, gb(2.0));
//! assert_eq!(demand.get(MEMORY), 6_000_000_000);
//! assert_eq!(demand.get(SLOTS), 0);
//! // Setting a resource again replaces its amount.
//! assert_eq!(demand.with(MEMORY, 1).get(MEMORY), 1);
//! ```
//!
//! A resource's [`default_demand`](Resource::default_demand) is what a job takes of it when its
//! demand leaves it out: one slot, so every job takes a slot unless it asks for more. A worker
//! states its capacity whole, so its slots too; one without slots runs nothing.
//!
//! Whether a worker takes a job is up to an [`Admission`] rule; the default,
//! [`ProductionAdmission`], admits a job if, in every declared resource, what the worker already
//! uses plus the job's demand fits its capacity. A job that fits nowhere waits, and
//! [`explain`](Policy::explain) says why: its [`Explanation`] gives the job's [`Status`], and for a
//! job [`Waiting`] for a worker, one [`Verdict`] per worker. Displayed, it is one line for a log.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobSpec, MEMORY, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState, gb,
//! # };
//! use whelm::{Status, Verdict};
//!
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 2),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 1,
//!         demand: Resources::new().with(MEMORY, gb(6.0)),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 2,
//!         demand: Resources::new().with(MEMORY, gb(6.0)),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//!
//! // A slot is free, but 6 + 6 GB exceeds the 10 GB of memory.
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! let why = p.explain(2).unwrap();
//! let Status::Waiting(waiting) = &why.status else {
//!     panic!("job 2 is not waiting");
//! };
//! for (worker, verdict) in &waiting.workers {
//!     match verdict {
//!         Verdict::Short { dims, headroom } => {
//!             // Short of memory only, with 4 GB left; resources are named.
//!             assert_eq!((*worker, &dims[..]), (1, &[MEMORY.name][..]));
//!             assert_eq!(headroom[0], (MEMORY.name, Some(4_000_000_000)));
//!         }
//!         other => panic!("worker {worker}: {other:?}"),
//!     }
//! }
//! assert_eq!(
//!     why.to_string(),
//!     "job 2 (demand [memory 6.00 GB, slots 1], group 0) waiting 0s, 0 more urgent job(s) \
//!      waiting; memory short on 1 worker(s) (best headroom 4.00 GB on worker 1)"
//! );
//!
//! p.handle(
//!     Input::Done { job: 1, attempt: 1 },
//!     Time(Duration::from_secs(50)),
//! );
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(50))),
//!     [Output::Start {
//!         job: 2,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! ```
//!
//! Demands are estimates, so the worker's own reports count too. A heartbeat ([`Input::Worker`]
//! again, with the same id) carries the worker's [`reported_used`](WorkerState::reported_used)
//! memory and the [`reported_baseline`](WorkerState::reported_baseline) not due to any job; the
//! rule takes the larger of the reported usage and the baseline plus the placed demands. Here the
//! running job uses more than it said, and the heartbeat keeps a second job off the worker.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobSpec, MEMORY, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState, gb,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! let worker = WorkerState {
//!     id: 1,
//!     capacity: Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 4),
//!     ..Default::default()
//! };
//! p.handle(Input::Worker(worker.clone()), Time::ORIGIN);
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 1,
//!         demand: Resources::new().with(MEMORY, gb(3.0)),
//!         ..Default::default()
//!     }),
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
//! // Heartbeat: 8 GB resident, 1 GB of it the worker's own runtime.
//! let heartbeat = WorkerState {
//!     reported_used: Resources::new().with(MEMORY, gb(8.0)),
//!     reported_baseline: Resources::new().with(MEMORY, gb(1.0)),
//!     ..worker
//! };
//! p.handle(Input::Worker(heartbeat), Time(Duration::from_secs(5)));
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 2,
//!         demand: Resources::new().with(MEMORY, gb(3.0)),
//!         ..Default::default()
//!     }),
//!     Time(Duration::from_secs(5)),
//! );
//! // max(8, 1 + 3) + 3 = 11 GB > 10 GB.
//! assert!(p.poll(Time(Duration::from_secs(5))).is_empty());
//! ```
//!
//! Memory is a *soft* resource. A job alone on a worker always runs, whatever its estimate (the
//! escape hatch, so that every job can run somewhere), and a zero memory capacity means "unknown"
//! and is not enforced. Slots are [*hard*](Resource::hard): always enforced, with no escape hatch.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobSpec, MEMORY, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState, gb,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 4),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 1,
//!         demand: Resources::new().with(MEMORY, gb(50.0)),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 2,
//!         demand: Resources::new().with(MEMORY, gb(1.0)),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! // Job 1 is five times the memory but runs alone; job 2 must wait for it.
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // A worker of unknown memory: only its two slots limit it.
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 2,
//!         capacity: Resources::new().with(SLOTS, 2),
//!         ..Default::default()
//!     }),
//!     Time(Duration::from_secs(1)),
//! );
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 3,
//!         demand: Resources::new().with(MEMORY, gb(500.0)),
//!         ..Default::default()
//!     }),
//!     Time(Duration::from_secs(1)),
//! );
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(1))),
//!     [
//!         Output::Start {
//!             job: 2,
//!             attempt: 1,
//!             worker: 2
//!         },
//!         Output::Start {
//!             job: 3,
//!             attempt: 1,
//!             worker: 2
//!         },
//!     ]
//! );
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 4,
//!         ..Default::default()
//!     }),
//!     Time(Duration::from_secs(2)),
//! );
//! assert!(p.poll(Time(Duration::from_secs(2))).is_empty());
//! assert!((p.explain(4).unwrap().to_string()).contains("slots full on 1 worker(s)"));
//! ```
//!
//! A worker can also declare a [`per_task`](WorkerState::per_task) floor: what any one job takes
//! there at least, whatever its demand says. With a device-memory floor and jobs that declare no
//! device demand, the device memory simply counts jobs. [`PolicyStats::workers`] shows each
//! worker's load and headroom as admission sees it.
//!
//! ```
//! # use whelm::{
//! #     Config, DEVICE_MEMORY, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState, gb,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! // 8 slots, unknown host memory, 10 GB of device memory, at least 4 GB of it per job.
//! let gpu = WorkerState {
//!     per_task: Resources::new().with(DEVICE_MEMORY, gb(4.0)),
//!     id: 1,
//!     class: "gpu".into(),
//!     capacity: Resources::new().with(DEVICE_MEMORY, gb(10.0)).with(SLOTS, 8),
//!     ..Default::default()
//! };
//! p.handle(Input::Worker(gpu), Time::ORIGIN);
//! for id in 1..=3 {
//!     let job = JobSpec {
//!         id,
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(job), Time::ORIGIN);
//! }
//! // (running + 1) * 4 GB <= 10 GB admits two jobs.
//! assert_eq!(p.poll(Time::ORIGIN).len(), 2);
//! let load = &p.stats().workers[0];
//! assert_eq!(load.running, 2);
//! let headroom: Vec<_> = load.headroom.iter().map(|(_, h)| *h).collect();
//! assert_eq!(headroom, [None, Some(2_000_000_000), Some(6)]); // memory, device memory, slots
//! ```
//!
//! Any other resource is a declaration away: a constant, made with [`Resource::new`] and its
//! `const` builder methods, added to [`Config::resources`]. Here workers count their GPUs, a hard
//! resource that jobs take none of unless they say so: the GPU jobs share the one worker that has
//! GPUs, the others go anywhere, and a job no worker has a GPU left for is explained by name. A
//! license pool per worker, or a scratch disk (soft, like memory), is declared the same way.
//!
//! A resource is its name. The scheduler reads a resource's rules (hard or soft, default demand,
//! unit) from its declaration alone, so amounts built with another constant of the same name mean
//! the declared resource. A name the declaration lacks is a mistake the scheduler reports rather
//! than ignores: a job demanding it is [rejected](Output::Rejected) and forgotten, and a worker
//! state naming it is a panic ([`Input::Worker`]).
//!
//! ```
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
//! # };
//! use whelm::{Rejection, Resource};
//!
//! const GPUS: Resource = Resource::new("gpus").hard();
//!
//! let mut config = Config::default();
//! config.resources.push(GPUS);
//! let mut p = Scheduler::new(config);
//!
//! // Worker 1 has no GPU, worker 2 has two; both have eight slots.
//! for (id, n) in [(1, 0), (2, 2)] {
//!     let worker = WorkerState {
//!         id,
//!         capacity: Resources::new().with(SLOTS, 8).with(GPUS, n),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Worker(worker), Time::ORIGIN);
//! }
//! // Jobs 1 to 3 need a GPU each; job 4 needs none.
//! for id in 1..=4 {
//!     let job = JobSpec {
//!         id,
//!         demand: Resources::new().with(GPUS, (id < 4).into()),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(job), Time::ORIGIN);
//! }
//! // Job 5 needs a TPU, which this configuration does not declare.
//! let tpu_job = JobSpec {
//!     id: 5,
//!     demand: Resources::new().with(Resource::new("tpus"), 1),
//!     ..Default::default()
//! };
//! p.handle(Input::Submit(tpu_job), Time::ORIGIN);
//!
//! let start = |job, worker| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker,
//! };
//! let rejected = Output::Rejected {
//!     job: 5,
//!     reason: Rejection::Undeclared {
//!         resource: "tpus".into(),
//!     },
//! };
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [rejected, start(1, 2), start(2, 2), start(4, 1)]
//! );
//! assert_eq!(
//!     p.explain(3).unwrap().to_string(),
//!     "job 3 (demand [slots 1, gpus 1], group 0) waiting 0s, 0 more urgent job(s) waiting; gpus \
//!      full on 2 worker(s)"
//! );
//! assert_eq!(p.explain(5), None);
//! ```
//!
//! The rule can be called directly on a [`WorkerView`]: the declared resources, a worker's state,
//! and what the scheduler has placed on it, as [`WorkerView::new`] builds it. A demand goes in as
//! the scheduler holds it, with the default demands filled in ([`WorkerView::demand`]). The view's
//! methods give the pieces of the rule per resource, such as the
//! [`headroom`](WorkerView::headroom) and the [`free_share`](WorkerView::free_share) that scores
//! compare workers by.
//!
//! ```
//! use whelm::{
//!     Admission, Config, MEMORY, ProductionAdmission, Resources, SLOTS, WorkerState, WorkerView,
//!     gb,
//! };
//!
//! let config = Config::default();
//! let state = WorkerState {
//!     id: 1,
//!     capacity: Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 4),
//!     ..Default::default()
//! };
//! // One job running, which took 6 GB and a slot.
//! let placed = Resources::new().with(MEMORY, gb(6.0)).with(SLOTS, 1);
//! let view = WorkerView::new(&config.resources, &state, &placed, 1);
//!
//! let job = |x| view.demand(&Resources::new().with(MEMORY, gb(x)));
//! assert!(ProductionAdmission.admits(&job(4.0), &view));
//! assert!(!ProductionAdmission.admits(&job(5.0), &view));
//! assert_eq!(view.headroom(MEMORY), Some(4_000_000_000));
//! assert_eq!(view.headroom(SLOTS), Some(3));
//! // After placing 2 GB more, a fifth of the memory would be left.
//! assert_eq!(view.free_share(&job(2.0)), 0.2);
//! ```
//!
//! A different rule plugs in with [`Scheduler::with_admission`]. It must be monotone in load (see
//! [`Admission`]) and it alone enforces capacity, slots included. This one trusts no memory figure
//! and counts slots only.
//!
//! ```
//! use whelm::{
//!     Admission, Amounts, Config, Input, JobSpec, MEMORY, Policy, Resources, SLOTS, Scheduler,
//!     Time, WorkerState, WorkerView, gb,
//! };
//!
//! /// Admits while a slot is free, whatever the memory figures say.
//! struct SlotsOnly;
//!
//! impl Admission for SlotsOnly {
//!     fn admits(&self, _demand: &Amounts, w: &WorkerView) -> bool {
//!         (w.running() as u64) < w.capacity(SLOTS)
//!     }
//! }
//!
//! let mut p = Scheduler::with_admission(Config::default(), SlotsOnly);
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 2),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! for id in 1..=3 {
//!     let job = JobSpec {
//!         id,
//!         demand: Resources::new().with(MEMORY, gb(50.0)),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(job), Time::ORIGIN);
//! }
//! assert_eq!(p.poll(Time::ORIGIN).len(), 2);
//! ```
//!
//! # Constraints
//!
//! A job's [`constraints`](JobSpec::constraints) restrict where it runs. Each names workers with a
//! [`Selector`] (one worker, or a class of workers) and binds with a [`Strength`]: `Require` and
//! `Forbid` are hard, `Avoid` is soft, and `Prefer` only ranks the workers that admit the job. The
//! constructors on [`Constraint`] cover each strength on one worker or one class.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Constraint, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! for (id, class) in [(1, "cpu"), (2, "gpu"), (3, "gpu")] {
//!     let worker = WorkerState {
//!         id,
//!         class: class.into(),
//!         capacity: Resources::new().with(SLOTS, 4),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Worker(worker), Time::ORIGIN);
//! }
//! let job = |id, constraints| JobSpec {
//!     id,
//!     constraints,
//!     ..Default::default()
//! };
//! let gpu = Constraint::require_class("gpu");
//! p.handle(Input::Submit(job(1, vec![gpu.clone()])), Time::ORIGIN);
//! let not_2 = Constraint::forbid_worker(2);
//! p.handle(Input::Submit(job(2, vec![gpu, not_2])), Time::ORIGIN);
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [
//!         Output::Start {
//!             job: 1,
//!             attempt: 1,
//!             worker: 2
//!         },
//!         Output::Start {
//!             job: 2,
//!             attempt: 1,
//!             worker: 3
//!         },
//!     ]
//! );
//!
//! // No worker has the class: the job waits, and says why.
//! let tpu = Constraint::require_class("tpu");
//! p.handle(
//!     Input::Submit(job(3, vec![tpu])),
//!     Time(Duration::from_secs(1)),
//! );
//! assert!(p.poll(Time(Duration::from_secs(1))).is_empty());
//! let why = p.explain(3).unwrap();
//! let verdicts = &why.waiting().unwrap().workers;
//! assert!(
//!     verdicts
//!         .iter()
//!         .all(|(_, v)| *v == whelm::Verdict::Ineligible)
//! );
//! assert!(
//!     why.to_string()
//!         .ends_with("; 3 worker(s) excluded by its constraints")
//! );
//! ```
//!
//! A preferred worker wins over a less loaded one: the default score ranks
//! [`Preferred`](ScoreTerm::Preferred) before [`Load`](ScoreTerm::Load) (see
//! [Ordering](#ordering)). Use it for cache affinity.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Constraint, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id,
//!             capacity: Resources::new().with(SLOTS, 4),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! let job = |id| JobSpec {
//!     id,
//!     ..Default::default()
//! };
//! p.handle(Input::Submit(job(1)), Time::ORIGIN);
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // Worker 1 is busier, but job 2 prefers it; job 3 has no preference.
//! let fond = JobSpec {
//!     id: 2,
//!     constraints: vec![Constraint::prefer_worker(1)],
//!     ..Default::default()
//! };
//! p.handle(Input::Submit(fond), Time(Duration::from_secs(1)));
//! p.handle(Input::Submit(job(3)), Time(Duration::from_secs(1)));
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(1))),
//!     [
//!         Output::Start {
//!             job: 2,
//!             attempt: 1,
//!             worker: 1
//!         },
//!         Output::Start {
//!             job: 3,
//!             attempt: 1,
//!             worker: 2
//!         },
//!     ]
//! );
//! ```
//!
//! An avoided worker is used only while no live worker (one with slots) that the hard constraints
//! allow is free of every `Avoid`. That depends on which workers exist, not on how loaded they are,
//! so the job waits for a busy acceptable worker; it is the same rule a retry applies to the
//! workers it failed on. Here job 2 avoids worker 1 and waits for worker 2, until worker 2 is
//! drained (its slot count set to zero by a heartbeat).
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Constraint, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id,
//!             capacity: Resources::new().with(SLOTS, 1),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//! }
//! let job = |id, constraint| JobSpec {
//!     id,
//!     constraints: vec![constraint],
//!     ..Default::default()
//! };
//! p.handle(
//!     Input::Submit(job(1, Constraint::prefer_worker(2))),
//!     Time::ORIGIN,
//! );
//! p.handle(
//!     Input::Submit(job(2, Constraint::avoid_worker(1))),
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
//! let why = p.explain(2).unwrap();
//! assert_eq!(
//!     why.waiting().unwrap().workers[0],
//!     (1, whelm::Verdict::Ineligible)
//! );
//!
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 2,
//!         capacity: Resources::new().with(SLOTS, 0),
//!         ..Default::default()
//!     }),
//!     Time(Duration::from_secs(1)),
//! );
//! assert_eq!(
//!     p.poll(Time(Duration::from_secs(1))),
//!     [Output::Start {
//!         job: 2,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! ```
//!
//! # Ordering
//!
//! A [`Config`] is a list-scheduling rule in two parts. Its [`order`](Config::order) decides which
//! waiting job is considered first: a lexicographic list of [`OrderTerm`]s, ties broken by arrival.
//! Its [`score`](Config::score) decides which of the workers that admit the job it goes to: a list
//! of [`ScoreTerm`]s, ties broken by the smallest worker id. Each [`poll`](Policy::poll) scans the
//! waiting jobs in order and places each one on its best admitting worker, so a job takes a worker
//! only if every more urgent job was refused there.
//!
//! The presets map objectives to rules. The same three jobs, run one at a time on one worker, start
//! in a different order under each:
//!
//! ```
//! use std::time::Duration;
//!
//! use whelm::{
//!     Config, GroupOrder, Input, JobId, JobSpec, Output, Policy, Resources, SLOTS, Scheduler,
//!     Time, WorkerState,
//! };
//!
//! /// The order one single-slot worker runs three jobs in under `config`.
//! fn run_order(config: Config) -> Vec<JobId> {
//!     let mut p = Scheduler::new(config);
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id: 1,
//!             capacity: Resources::new().with(SLOTS, 1),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//!     // (id, group, weight, work, due)
//!     let jobs = [(1, 5, 1.0, 10, 100), (2, 3, 1.0, 1, 50), (3, 5, 4.0, 5, 20)];
//!     for (id, group, weight, work, due) in jobs {
//!         let spec = JobSpec {
//!             id,
//!             group,
//!             weight,
//!             work: Some(Duration::from_secs(work)),
//!             due: Some(Time(Duration::from_secs(due))),
//!             ..Default::default()
//!         };
//!         p.handle(Input::Submit(spec), Time::ORIGIN);
//!     }
//!     let (mut order, mut now) = (Vec::new(), Time::ORIGIN);
//!     while order.len() < 3 {
//!         for out in p.poll(now) {
//!             if let Output::Start { job, attempt, .. } = out {
//!                 order.push(job);
//!                 now += Duration::from_secs(1);
//!                 p.handle(Input::Done { job, attempt }, now);
//!             }
//!         }
//!     }
//!     order
//! }
//!
//! // Groups in order of first arrival (group 5 first), then arrival within a group.
//! assert_eq!(run_order(Config::default()), [1, 3, 2]);
//! // Arrival only.
//! assert_eq!(run_order(Config::fifo()), [1, 2, 3]);
//! // Largest weight / work first (Smith's rule): 1.0, 0.8, 0.1 per second.
//! assert_eq!(run_order(Config::weighted_completion()), [2, 3, 1]);
//! // Earliest due date first (Jackson's rule).
//! assert_eq!(run_order(Config::lateness()), [3, 2, 1]);
//! // Groups by id rather than by arrival.
//! assert_eq!(
//!     run_order(Config {
//!         group_order: GroupOrder::Id,
//!         ..Config::default()
//!     }),
//!     [2, 1, 3]
//! );
//! ```
//!
//! [Groups](JobSpec::group) gather related jobs (Nassau's bidegrees, through [`nassau::group`]) so
//! that [`OrderTerm::Group`] finishes one group before starting the next. [`GroupOrder::Arrival`]
//! depends on the order the caller happened to submit in; [`GroupOrder::Id`] does not, so it
//! survives a caller restart. [`Scheduler::forget_group`] drops a finished group's arrival record.
//!
//! ## Explicit priorities
//!
//! [`OrderTerm::Priority`], first in every preset but FIFO, is the hook for an external planner:
//! smaller [`priority`](JobSpec::priority) values go first, and jobs without one count as
//! [`Config::default_priority`], so a planner can pull some jobs ahead of the rest and push others
//! behind.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobId, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! for (id, priority) in [(1, Some(1)), (2, None), (3, Some(-1))] {
//!     let spec = JobSpec {
//!         id,
//!         priority,
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(spec), Time::ORIGIN);
//! }
//! let mut order: Vec<JobId> = Vec::new();
//! for now in [0, 1, 2].map(|s| Time(Duration::from_secs(s))) {
//!     for out in p.poll(now) {
//!         if let Output::Start { job, attempt, .. } = out {
//!             order.push(job);
//!             p.handle(
//!                 Input::Done { job, attempt },
//!                 now + Duration::from_millis(500),
//!             );
//!         }
//!     }
//! }
//! assert_eq!(order, [3, 2, 1]);
//! ```
//!
//! ## Choosing a worker
//!
//! The default score is [`Speed`](ScoreTerm::Speed), [`Preferred`](ScoreTerm::Preferred),
//! [`Load`](ScoreTerm::Load): the fastest admitting worker, then a preferred one, then the least
//! loaded. [`Config::best_fit`] inserts [`Tightest`](ScoreTerm::Tightest) after speed, packing each
//! job where it leaves the least room and keeping large holes open for large jobs.
//!
//! ```
//! # use whelm::{
//! #     Config, Input, JobSpec, MEMORY, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState, gb,
//! # };
//! /// The worker a 10 GB job goes to, given a 100 GB worker 1 and a 20 GB worker 2.
//! fn place(config: Config) -> u64 {
//!     let mut p = Scheduler::new(config);
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id: 1,
//!             class: "big".into(),
//!             capacity: Resources::new().with(MEMORY, gb(100.0)).with(SLOTS, 4),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//!     p.handle(
//!         Input::Worker(WorkerState {
//!             id: 2,
//!             class: "small".into(),
//!             capacity: Resources::new().with(MEMORY, gb(20.0)).with(SLOTS, 4),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//!     let job = JobSpec {
//!         id: 1,
//!         demand: Resources::new().with(MEMORY, gb(10.0)),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(job), Time::ORIGIN);
//!     match p.poll(Time::ORIGIN)[..] {
//!         [Output::Start { worker, .. }] => worker,
//!         ref out => panic!("{out:?}"),
//!     }
//! }
//! assert_eq!(place(Config::default()), 1); // equally loaded: the smaller id
//! assert_eq!(place(Config::best_fit()), 2); // the tighter fit
//! ```
//!
//! Workers report a [`speed`](WorkerState::speed) relative to a reference worker; with
//! [`ScoreTerm::Speed`] in the score the fastest admitting worker wins. [Speed](#speed) covers
//! where speeds come from and what else they drive.
//!
//! ```
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         class: "old".into(),
//!         capacity: Resources::new().with(SLOTS, 4),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! let fast = WorkerState {
//!     id: 2,
//!     class: "new".into(),
//!     capacity: Resources::new().with(SLOTS, 4),
//!     speed: 2.5,
//!     ..Default::default()
//! };
//! p.handle(Input::Worker(fast), Time::ORIGIN);
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 1,
//!         ..Default::default()
//!     }),
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
//! ```
//!
//! # Time
//!
//! The policy reads time only from the `now` passed with each call, and some of its decisions
//! depend on how long a job has waited. Two mechanisms bound waiting under strict priority order.
//!
//! **Aging** ([`Config::age_limit`]): a job that has waited that long becomes more urgent than
//! every job that has not, oldest first. Below, job 2 waits behind a running job; a more urgent job
//! 3 arrives later. When the worker frees at 150 s, job 2 has waited past the 100 s limit and goes
//! first; without aging, job 3 would.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobId, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time,
//! #     WorkerState,
//! # };
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
//!     let job = |id| JobSpec {
//!         id,
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(job(1)), Time::ORIGIN);
//!     p.handle(Input::Submit(job(2)), Time::ORIGIN);
//!     p.poll(Time::ORIGIN);
//!     let urgent = JobSpec {
//!         id: 3,
//!         priority: Some(-1),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(urgent), Time(Duration::from_secs(50)));
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
//! **Reservations** ([`Config::reservations`]): a large job can starve while smaller ones keep
//! filling the space it needs. The most urgent job that has waited
//! [`reserve_after`](Reservations::reserve_after) and is admitted nowhere reserves a worker, which
//! then takes no other job until the holder is placed. Every other worker keeps taking less urgent
//! jobs (backfill).
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Holding, Input, JobSpec, MEMORY, Output, Policy, ReservationInfo, Resources,
//! #     SLOTS, Scheduler, Time, Verdict, WorkerState, gb,
//! # };
//! let mut p = Scheduler::new(Config::default()); // reserve after 60 s
//! let capacity = Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 3);
//! let worker = WorkerState { id: 1, capacity, ..Default::default() };
//! p.handle(Input::Worker(worker), Time::ORIGIN);
//! let job = |id, size| {
//!     Input::Submit(JobSpec { id, demand: Resources::new().with(MEMORY, gb(size)), ..Default::default() })
//! };
//! let start = |job| Output::Start { job, attempt: 1, worker: 1 };
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
//! let reservation = ReservationInfo { job: 9, worker: 1, since: Time(Duration::from_secs(60)) };
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
//! # use whelm::{
//! #     Config, Input, JobSpec, MEMORY, Output, Policy, Reservations, Resources, SLOTS, Scheduler,
//! #     Time, WorkerState, gb,
//! # };
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
//! let job = |id, size, work| JobSpec {
//!     id,
//!     demand: Resources::new().with(MEMORY, gb(size)),
//!     work: Some(Duration::from_secs(work)),
//!     ..Default::default()
//! };
//!
//! p.handle(Input::Submit(job(1, 4.0, 100)), Time::ORIGIN);
//! p.handle(Input::Submit(job(9, 8.0, 100)), Time::ORIGIN);
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! p.handle(
//!     Input::Submit(job(2, 4.0, 50)),
//!     Time(Duration::from_secs(60)),
//! );
//! p.handle(
//!     Input::Submit(job(3, 4.0, 30)),
//!     Time(Duration::from_secs(60)),
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
//! Reservations and voluntary waits for a faster worker ([`Defer`], in the next chapter) are both
//! *holds*: a worker kept from a job that it might admit. Holds, aging and reservation thresholds
//! make the passing of time matter even when no event arrives, so the policy says when it next
//! needs a poll: [`next_wakeup`](Policy::next_wakeup). A caller sleeps until the earlier of its
//! next event and that time. Here a job that fits nowhere has two deadlines: it may reserve at
//! [`reserve_after`](Reservations::reserve_after) and it ages at [`DEFAULT_AGE_LIMIT`].
//!
//! ```
//! # use whelm::{
//! #     Config, DEFAULT_AGE_LIMIT, Input, JobSpec, MEMORY, Policy, Reservations, Resources, SLOTS,
//! #     Scheduler, Time, WorkerState, gb,
//! # };
//! let mut p = Scheduler::new(Config::default());
//! let capacity = Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 2);
//! let worker = WorkerState { id: 1, capacity, ..Default::default() };
//! p.handle(Input::Worker(worker), Time::ORIGIN);
//! let job = |id| JobSpec { id, demand: Resources::new().with(MEMORY, gb(6.0)), ..Default::default() };
//! p.handle(Input::Submit(job(1)), Time::ORIGIN); // runs for ever
//! p.handle(Input::Submit(job(2)), Time::ORIGIN);
//! p.poll(Time::ORIGIN);
//!
//! // No events arrive: poll whenever the policy asks to.
//! let mut wakeups = Vec::new();
//! while let Some(t) = p.next_wakeup() {
//!     wakeups.push(t);
//!     p.poll(t);
//! }
//! let reserve_after = Reservations::default().reserve_after;
//! assert_eq!(wakeups, [Time::ORIGIN + reserve_after, Time::ORIGIN + DEFAULT_AGE_LIMIT]);
//! assert_eq!(p.stats().reservations[0].job, 2);
//! ```
//!
//! # Speed
//!
//! Workers differ in speed. A job's [`work`](JobSpec::work) is its run time on a worker of speed
//! 1, so on a worker of speed `s` it is expected to take `work / s`. The machine model,
//! [`SpeedConfig::timing`], says where speeds come from:
//!
//! - [`Timing::Identical`]: every worker runs at speed 1, whatever it reports.
//! - [`Timing::Related`] (the default): each worker has one speed, as reported in
//!   [`WorkerState::speed`] or, with [`Learn`], learned from completion times.
//! - [`Timing::Unrelated`]: a job's speed also depends on its [`kind`](JobSpec::kind), learned per
//!   kind and worker class.
//!
//! ```
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, SpeedConfig, Time,
//! #     Timing, WorkerState,
//! # };
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
//!     let job = JobSpec {
//!         id: 1,
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(job), Time::ORIGIN);
//!     match p.poll(Time::ORIGIN)[..] {
//!         [Output::Start { worker, .. }] => worker,
//!         ref out => panic!("{out:?}"),
//!     }
//! }
//! assert_eq!(place(Timing::default()), 2);
//! assert_eq!(place(Timing::Identical), 1);
//! ```
//!
//! ## Learning speeds
//!
//! With [`Timing::learned`], each completion of a job with a work estimate is a sample of its
//! worker's speed, and a class's estimate replaces the reported speed once it has
//! [`min_samples`](Learn::min_samples). Here both workers report speed 1, but worker 2 really runs
//! three times faster; [`WorkerLoad::speed`] shows what the policy has learned.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Constraint, Input, JobSpec, Learn, Policy, Resources, SLOTS, Scheduler,
//! #     SpeedConfig, Time, Timing, WorkerState,
//! # };
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
//! let (mut now, mut id) = (Time::ORIGIN, 0);
//! for _ in 0..Learn::default().min_samples {
//!     for (class, true_speed) in [("a", 1.0), ("b", 3.0)] {
//!         let spec = JobSpec {
//!             id,
//!             work: Some(work),
//!             constraints: vec![Constraint::require_class(class)],
//!             ..Default::default()
//!         };
//!         p.handle(Input::Submit(spec), now);
//!         p.poll(now);
//!         now += work.div_f64(true_speed);
//!         p.handle(
//!             Input::Done {
//!                 job: id,
//!                 attempt: 1,
//!             },
//!             now,
//!         );
//!         id += 1;
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
//! # use whelm::{
//! #     Config, Constraint, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler,
//! #     SpeedConfig, Time, Timing, WorkerState,
//! # };
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
//!     let (mut now, mut id) = (Time::ORIGIN, 0);
//!     for _ in 0..20 {
//!         let runs = [
//!             ("a", "x", 4.0),
//!             ("a", "y", 1.0),
//!             ("b", "x", 1.0),
//!             ("b", "y", 2.0),
//!         ];
//!         for (kind, class, true_speed) in runs {
//!             let spec = JobSpec {
//!                 id,
//!                 work: Some(work),
//!                 kind: Some(kind.into()),
//!                 constraints: vec![Constraint::require_class(class)],
//!                 ..Default::default()
//!             };
//!             p.handle(Input::Submit(spec), now);
//!             p.poll(now);
//!             now += work.div_f64(true_speed);
//!             p.handle(
//!                 Input::Done {
//!                     job: id,
//!                     attempt: 1,
//!                 },
//!                 now,
//!             );
//!             id += 1;
//!         }
//!     }
//!     (p, now)
//! }
//!
//! /// Where a lone job of `kind` goes once trained.
//! fn place(timing: Timing, kind: &str) -> u64 {
//!     let (mut p, now) = trained(timing);
//!     let spec = JobSpec {
//!         id: 1000,
//!         work: Some(Duration::from_secs(8)),
//!         kind: Some(kind.into()),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Submit(spec), now);
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
//! ## Waiting for a faster worker
//!
//! The score picks the best worker that admits a job *now*. With [`SpeedConfig::defer`], a job may
//! instead wait for a busy, faster worker on which it would finish sooner (earliest finish time, as
//! in HEFT). The wait is a hold: it shows in [`PolicyStats::deferred`] and as an `explain`'s [`Holding::Deferral`], and it
//! lapses after [`max_wait`](Defer::max_wait).
//!
//! ```
//! # use std::time::Duration;
//! #
//! # use whelm::{
//! #     Config, Defer, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, SpeedConfig,
//! #     Time, WorkerState,
//! # };
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
//! let job = |id, work| JobSpec {
//!     id,
//!     work: Some(Duration::from_secs(work)),
//!     ..Default::default()
//! };
//!
//! // Job 1 takes the fast worker until 10 / 4 = 2.5 s.
//! p.handle(Input::Submit(job(1, 10)), Time::ORIGIN);
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
//! p.handle(Input::Submit(job(2, 40)), Time::ORIGIN);
//! assert!(p.poll(Time::ORIGIN).is_empty());
//! assert_eq!(p.stats().deferred, [(2, 2, Time(Duration::from_millis(2500)))]);
//! assert!(matches!(
//!     p.explain(2).unwrap().waiting().unwrap().hold,
//!     Some(whelm::Holding::Deferral { worker: 2, .. })
//! ));
//!
//! p.handle(Input::Done { job: 1, attempt: 1 }, Time(Duration::from_millis(2500)));
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
//! ## Speculative attempts
//!
//! With [`SpeedConfig::speculate`], a worker left idle after a poll starts a second attempt of a
//! job running on a slower worker, when it would finish sufficiently sooner. Both attempts run; the
//! first to finish completes the job and the other is stopped. Idempotence is what makes this safe.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Speculate,
//! #     SpeedConfig, Time, WorkerState,
//! # };
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
//!     id: 1,
//!     work: Some(Duration::from_secs(40)),
//!     ..Default::default()
//! };
//! p.handle(Input::Submit(spec), Time::ORIGIN);
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
//!
//! # Dependencies
//!
//! [`DagScheduler`] puts a dependency layer in front of any policy and is itself a [`Policy`]. Jobs
//! are *declared* with their dependencies, possibly long before they can run; each is submitted to
//! the inner policy when its last dependency completes. Everything else (workers, reports,
//! explanations) goes through as before.
//!
//! ```
//! use std::time::Duration;
//!
//! use whelm::{
//!     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources, SLOTS,
//!     Scheduler, Time, WorkerState,
//! };
//!
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
//! dag.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 4),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! // A job and the jobs it waits for.
//! let job = |id, deps| DagJob {
//!     spec: JobSpec {
//!         id,
//!         ..Default::default()
//!     },
//!     deps,
//!     ..Default::default()
//! };
//! let start = |job| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker: 1,
//! };
//!
//! // A diamond: 1 -> {2, 3} -> 4.
//! let jobs = [
//!     job(1, vec![]),
//!     job(2, vec![1]),
//!     job(3, vec![1]),
//!     job(4, vec![2, 3]),
//! ];
//! dag.declare(jobs, Time::ORIGIN).unwrap();
//! assert_eq!(dag.poll(Time::ORIGIN), [start(1)]);
//! assert_eq!(
//!     dag.explain(4).unwrap().status,
//!     whelm::Status::Pending {
//!         unit: 4,
//!         closed: false,
//!         unmet: vec![2, 3]
//!     }
//! );
//!
//! dag.handle(
//!     Input::Done { job: 1, attempt: 1 },
//!     Time(Duration::from_secs(1)),
//! );
//! assert_eq!(dag.poll(Time(Duration::from_secs(1))), [start(2), start(3)]);
//! dag.handle(
//!     Input::Done { job: 2, attempt: 1 },
//!     Time(Duration::from_secs(2)),
//! );
//! dag.handle(
//!     Input::Done { job: 3, attempt: 1 },
//!     Time(Duration::from_secs(2)),
//! );
//! assert_eq!(dag.poll(Time(Duration::from_secs(2))), [start(4)]);
//! ```
//!
//! A declaration that would close a cycle, or redeclare a job, is rejected with a [`DagError`] and
//! leaves no trace. A dependency may name a job not declared yet; it is pending until declared and
//! completed.
//!
//! ## The layer's own outputs
//!
//! Not every job runs on a worker. A [local](field@DagJob::local) job runs on the caller: it is
//! announced by [`Output::RunLocal`] and reported with [`Input::Done`] and attempt 0. A
//! [passthrough](field@DagJob::passthrough) is a pure synchronisation point that completes by
//! itself, and with [`DagConfig::record_passthrough`] is announced by [`Output::Passed`].
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources, SLOTS,
//! #     Scheduler, Time, WorkerState,
//! # };
//! let config = DagConfig {
//!     record_passthrough: true,
//!     ..DagConfig::default()
//! };
//! let mut dag = DagScheduler::new(config, Scheduler::new(Config::default()));
//! dag.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 4),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! let job = |id, deps| DagJob {
//!     spec: JobSpec {
//!         id,
//!         ..Default::default()
//!     },
//!     deps,
//!     ..Default::default()
//! };
//!
//! // Load locally (1), then a barrier (2), then compute (3).
//! let jobs = [
//!     DagJob {
//!         local: true,
//!         ..job(1, vec![])
//!     },
//!     DagJob {
//!         passthrough: true,
//!         ..job(2, vec![1])
//!     },
//!     job(3, vec![2]),
//! ];
//! dag.declare(jobs, Time::ORIGIN).unwrap();
//! assert_eq!(dag.poll(Time::ORIGIN), [Output::RunLocal { job: 1 }]);
//!
//! dag.handle(
//!     Input::Done { job: 1, attempt: 0 },
//!     Time(Duration::from_secs(1)),
//! );
//! assert_eq!(
//!     dag.poll(Time(Duration::from_secs(1))),
//!     [
//!         Output::Passed { job: 2 },
//!         Output::Start {
//!             job: 3,
//!             attempt: 1,
//!             worker: 1
//!         }
//!     ]
//! );
//! ```
//!
//! Without [`DagConfig::auto_submit`], a ready job is held and announced by [`Output::Ready`]; the
//! caller submits it with [`release`](DagScheduler::release) once it is actually sendable.
//! [`announcements`](DagScheduler::announcements) drains the layer's announcements without polling
//! the inner policy, so the caller can react (release, declare, close) before anything is placed.
//!
//! ```
//! # use whelm::{
//! #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources, SLOTS,
//! #     Scheduler, Time, WorkerState,
//! # };
//! let config = DagConfig {
//!     auto_submit: false,
//!     ..DagConfig::default()
//! };
//! let mut dag = DagScheduler::new(config, Scheduler::new(Config::default()));
//! dag.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 4),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//!
//! let job = DagJob {
//!     spec: JobSpec {
//!         id: 1,
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! dag.declare([job], Time::ORIGIN).unwrap();
//! assert_eq!(dag.announcements(), [Output::Ready { job: 1 }]);
//! assert_eq!(dag.explain(1).unwrap().status, whelm::Status::Held);
//!
//! // ... prepare the job's inputs, then hand it over.
//! assert!(dag.release(1, Time::ORIGIN));
//! assert_eq!(
//!     dag.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! ```
//!
//! ## Templates and units
//!
//! Large graphs repeat a structure. A [`DagTemplate`] is a dependency structure checked once and
//! shared; a [`Unit`] is one instance of it, whose leaf `k` is job `base + k` and which other units
//! depend on by its own id. A [`TemplateNode::Unit`] substitutes a whole template for one node, so
//! templates nest. A plain [`DagJob`] is a unit of a one-node template.
//!
//! ```
//! use std::{sync::Arc, time::Duration};
//!
//! use whelm::{
//!     Config, DagConfig, DagScheduler, Input, Output, Policy, Resources, SLOTS, Scheduler,
//!     TemplateNode, TemplateSpec, Time, Unit, WorkerState,
//! };
//!
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
//! dag.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 4),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! let start = |job| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker: 1,
//! };
//! let done = |job| Input::Done { job, attempt: 1 };
//!
//! // A job, then a pair of independent jobs, then a job: four leaves.
//! let pair = Arc::new(TemplateSpec::jobs(2).build().unwrap());
//! let shape = TemplateSpec {
//!     nodes: vec![
//!         TemplateNode::Job(Duration::from_secs(1)),
//!         TemplateNode::Unit(pair),
//!         TemplateNode::Job(Duration::from_secs(1)),
//!     ],
//!     edges: vec![(0, 1), (1, 2)],
//! }
//! .build()
//! .unwrap();
//! assert_eq!((shape.len(), shape.leaves()), (3, 4));
//! let shape = Arc::new(shape);
//!
//! // Unit 10 is jobs 100..104; unit 20, jobs 200..204, runs after it. Each leaf is submitted
//! // with the unit's `spec`, here the default one, under the leaf's id.
//! let first = Unit {
//!     id: 10,
//!     base: 100,
//!     template: shape.clone(),
//!     ..Default::default()
//! };
//! let second = Unit {
//!     id: 20,
//!     base: 200,
//!     template: shape,
//!     deps: vec![10],
//!     ..Default::default()
//! };
//! dag.declare([first, second], Time::ORIGIN).unwrap();
//! assert_eq!(dag.poll(Time::ORIGIN), [start(100)]);
//! assert_eq!(
//!     dag.explain(20).unwrap().to_string(),
//!     "unit 20 waits for 1 dependency [10]"
//! );
//!
//! dag.handle(done(100), Time(Duration::from_secs(1)));
//! assert_eq!(
//!     dag.poll(Time(Duration::from_secs(1))),
//!     [start(101), start(102)]
//! );
//! dag.handle(done(101), Time(Duration::from_secs(2)));
//! dag.handle(done(102), Time(Duration::from_secs(2)));
//! assert_eq!(dag.poll(Time(Duration::from_secs(2))), [start(103)]);
//! dag.handle(done(103), Time(Duration::from_secs(3)));
//! assert_eq!(dag.poll(Time(Duration::from_secs(3))), [start(200)]);
//! ```
//!
//! A unit costs a fixed amount of memory until its dependencies complete; only then is its per-node
//! state allocated, and it is freed when the unit completes ([`DagStats`] counts both). Per-leaf
//! data that differs between units of one template comes from a [`NodeSource`] rather than being
//! stored: a [`sourced`](field@Unit::sourced) unit asks the scheduler's source for each leaf's
//! work, final spec and label, on demand.
//!
//! ```
//! # use std::{sync::Arc, time::Duration};
//! # use whelm::{
//! #     Config, DagConfig, DagScheduler, Input, JobSpec, MEMORY, Output, Policy, Resources, SLOTS,
//! #     Scheduler, TemplateSpec, Time, Unit, WorkerState, gb,
//! # };
//! use whelm::{JobId, NodeSource};
//!
//! /// Leaf `k` has work `10 (k + 1)` seconds and needs `k + 1` GB.
//! struct Growing;
//!
//! impl NodeSource for Growing {
//!     fn work(&self, _unit: JobId, leaf: u32) -> Duration {
//!         Duration::from_secs(10 * (u64::from(leaf) + 1))
//!     }
//!     fn spec(&self, _unit: JobId, leaf: u32, spec: &mut JobSpec) {
//!         spec.demand = Resources::new().with(MEMORY, gb(f64::from(leaf + 1)));
//!     }
//!     fn label(&self, unit: JobId, leaf: u32) -> Option<String> {
//!         Some(format!("unit {unit} step {leaf}"))
//!     }
//! }
//!
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()))
//!     .with_source(Arc::new(Growing));
//! let capacity = Resources::new().with(MEMORY, gb(100.0)).with(SLOTS, 4);
//! let worker = WorkerState { id: 1, capacity, ..Default::default() };
//! dag.handle(Input::Worker(worker), Time::ORIGIN);
//! let chain = TemplateSpec { edges: vec![(0, 1), (1, 2)], ..TemplateSpec::jobs(3) };
//! let unit = Unit {
//!     id: 10,
//!     base: 100,
//!     template: Arc::new(chain.build().unwrap()),
//!     sourced: true,
//!     ..Default::default()
//! };
//! dag.declare([unit], Time::ORIGIN).unwrap();
//!
//! assert_eq!(dag.poll(Time::ORIGIN), [Output::Start { job: 100, attempt: 1, worker: 1 }]);
//! assert_eq!(dag.stats().workers[0].placed.get(MEMORY), 1_000_000_000);
//! assert_eq!(
//!     dag.explain(102).unwrap().to_string(),
//!     "[unit 10 step 2] job 102 waits for 1 dependency within its unit"
//! );
//! ```
//!
//! ## Ranks
//!
//! The layer computes each job's *upward rank*: its work plus the longest chain of work below it,
//! through its unit and the units that depend on it ([`DagScheduler::rank`]). Jobs are submitted
//! with it as [`JobSpec::rank`], and [`OrderTerm::Rank`] orders by it, longest remaining chain
//! first, as HEFT does. That keeps the critical path moving.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources, SLOTS,
//! #     Scheduler, Time, WorkerState,
//! # };
//! use whelm::OrderTerm;
//!
//! /// The first job one single-slot worker starts: a lone job 1, or the head of chain 2 -> 3 -> 4.
//! fn first(config: Config) -> u64 {
//!     let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(config));
//!     dag.handle(
//!         Input::Worker(WorkerState {
//!             id: 1,
//!             capacity: Resources::new().with(SLOTS, 1),
//!             ..Default::default()
//!         }),
//!         Time::ORIGIN,
//!     );
//!     let job = |id, deps| DagJob {
//!         spec: JobSpec {
//!             id,
//!             ..Default::default()
//!         },
//!         deps,
//!         ..Default::default()
//!     };
//!     dag.declare(
//!         [
//!             job(1, vec![]),
//!             job(2, vec![]),
//!             job(3, vec![2]),
//!             job(4, vec![3]),
//!         ],
//!         Time::ORIGIN,
//!     )
//!     .unwrap();
//!     // Every job has the default work of a second.
//!     let secs = |s| Some(Duration::from_secs(s));
//!     assert_eq!((dag.rank(1), dag.rank(2)), (secs(1), secs(3)));
//!     match dag.poll(Time::ORIGIN)[..] {
//!         [Output::Start { job, .. }] => job,
//!         ref out => panic!("{out:?}"),
//!     }
//! }
//! assert_eq!(first(Config::default()), 1);
//! assert_eq!(
//!     first(Config {
//!         order: vec![OrderTerm::Rank],
//!         ..Config::default()
//!     }),
//!     2
//! );
//! ```
//!
//! ## Resuming and closing units
//!
//! A unit can be declared with some leaves already complete ([`Unit::completed`], e.g. from a
//! checkpoint), and closed early ([`close`](DagScheduler::close)) when its remaining jobs are known
//! to be no-ops. Closing completes the unit at once: its unstarted jobs are dropped, and the jobs
//! already running are returned, keeping their resources until their attempts end.
//!
//! ```
//! # use std::{sync::Arc, time::Duration};
//! # use whelm::{
//! #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources, SLOTS,
//! #     Scheduler, TemplateSpec, Time, Unit, WorkerState,
//! # };
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
//! dag.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//!
//! // Unit 10: four independent jobs 100..104, the first two done before a restart. Job 20 follows.
//! let unit = Unit {
//!     id: 10,
//!     base: 100,
//!     template: Arc::new(TemplateSpec::jobs(4).build().unwrap()),
//!     completed: vec![0, 1],
//!     ..Default::default()
//! };
//! let after = DagJob {
//!     spec: JobSpec {
//!         id: 20,
//!         ..Default::default()
//!     },
//!     deps: vec![10],
//!     ..Default::default()
//! };
//! dag.declare([unit, after.into()], Time::ORIGIN).unwrap();
//! assert_eq!(
//!     dag.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 102,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // Job 103 turns out to be unnecessary: close the unit. Job 102 is still running.
//! assert_eq!(dag.close(10, Time(Duration::from_secs(1))), Ok(vec![102]));
//! assert!(dag.poll(Time(Duration::from_secs(1))).is_empty()); // job 20 is ready, but 102 holds the slot
//! dag.handle(
//!     Input::Done {
//!         job: 102,
//!         attempt: 1,
//!     },
//!     Time(Duration::from_secs(2)),
//! );
//! assert_eq!(
//!     dag.poll(Time(Duration::from_secs(2))),
//!     [Output::Start {
//!         job: 20,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! ```
//!
//! ## Snapshots
//!
//! With the `serde` feature, `DagScheduler::snapshot` saves the declared graph (not the inner
//! policy), and `DagScheduler::restore` rebuilds the layer in front of a fresh policy. Jobs that
//! were submitted are submitted again, so a restarted coordinator only loses the attempts in
//! flight.
//!
//! ```
//! # #[cfg(feature = "serde")]
//! # fn main() {
//! # use std::time::Duration;
//! # use whelm::{
//! #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources, SLOTS,
//! #     Scheduler, Time, WorkerState,
//! # };
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
//! let worker = WorkerState {
//!     id: 1,
//!     capacity: Resources::new().with(SLOTS, 1),
//!     ..Default::default()
//! };
//! dag.handle(Input::Worker(worker.clone()), Time::ORIGIN);
//! let job = |id, deps| DagJob {
//!     spec: JobSpec {
//!         id,
//!         ..Default::default()
//!     },
//!     deps,
//!     ..Default::default()
//! };
//! dag.declare([job(1, vec![]), job(2, vec![1])], Time::ORIGIN)
//!     .unwrap();
//! dag.poll(Time::ORIGIN);
//! dag.handle(
//!     Input::Done { job: 1, attempt: 1 },
//!     Time(Duration::from_secs(1)),
//! );
//! assert_eq!(
//!     dag.poll(Time(Duration::from_secs(1))),
//!     [Output::Start {
//!         job: 2,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // The coordinator saves its graph, restarts, and its workers reconnect.
//! let json = serde_json::to_string(&dag.snapshot()).unwrap();
//! let snapshot = serde_json::from_str(&json).unwrap();
//! let mut dag = DagScheduler::restore(
//!     snapshot,
//!     Scheduler::new(Config::default()),
//!     None,
//!     Time(Duration::from_secs(5)),
//! );
//! dag.handle(Input::Worker(worker), Time(Duration::from_secs(5)));
//! assert_eq!(
//!     dag.poll(Time(Duration::from_secs(5))),
//!     [Output::Start {
//!         job: 2,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! # }
//! # #[cfg(not(feature = "serde"))]
//! # fn main() {}
//! ```
//!
//! # Threads
//!
//! Some callers have a thread per task rather than an event loop. [`SharedPolicy`] puts a policy
//! behind a lock: [`lease`](SharedPolicy::lease) submits a job and blocks until the policy starts
//! it, and the returned [`Lease`] names the worker and ends with [`complete`](Lease::complete) or
//! [`fail`](Lease::fail). It polls after every call; [`spawn_ticker`](SharedPolicy::spawn_ticker)
//! also polls as time passes, for holds and aging. Here three threads share a one-slot worker and
//! run one after another.
//!
//! ```
//! use whelm::{Config, JobSpec, Resources, SLOTS, Scheduler, SharedPolicy, WorkerState};
//!
//! let shared = SharedPolicy::with_system_clock(Scheduler::new(Config::default()));
//! shared.worker_update(WorkerState {
//!     id: 1,
//!     capacity: Resources::new().with(SLOTS, 1),
//!     ..Default::default()
//! });
//!
//! std::thread::scope(|s| {
//!     for id in 1..=3 {
//!         let shared = &shared;
//!         s.spawn(move || {
//!             let job = JobSpec {
//!                 id,
//!                 ..Default::default()
//!             };
//!             let lease = shared.lease(job); // blocks
//!             assert_eq!((lease.worker(), lease.attempt()), (1, 1));
//!             // ... send the task to lease.worker() and wait for its reply ...
//!             lease.complete();
//!         });
//!     }
//! });
//! let stats = shared.stats();
//! assert_eq!(
//!     (stats.placements_total, stats.waiting, stats.running),
//!     (3, 0, 0)
//! );
//! ```
//!
//! A failed lease blocks for the job's retry, on another worker if there is one, or returns the
//! [`GaveUp`] once the policy stops retrying.
//!
//! ```
//! # use whelm::{
//! #     Config, FailKind, JobSpec, Resources, RetryConfig, SLOTS, Scheduler, SharedPolicy,
//! #     WorkerState,
//! # };
//! let config = Config {
//!     retry: RetryConfig { max_attempts: 2 },
//!     ..Config::default()
//! };
//! let shared = SharedPolicy::with_system_clock(Scheduler::new(config));
//! for id in [1, 2] {
//!     shared.worker_update(WorkerState {
//!         id,
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     });
//! }
//!
//! let lease = shared.lease(JobSpec {
//!     id: 1,
//!     ..Default::default()
//! });
//! assert_eq!(lease.worker(), 1);
//! let retry = lease.fail(FailKind::Timeout, "no reply").unwrap();
//! assert_eq!((retry.worker(), retry.attempt()), (2, 2));
//! // `Lease` is not `Debug`, so `unwrap_err` is unavailable.
//! let Err(gave_up) = retry.fail(FailKind::Timeout, "no reply") else {
//!     panic!("a third attempt");
//! };
//! assert_eq!(gave_up.tried.len(), 2);
//! ```
//!
//! # Logging and replay
//!
//! Because the policy is deterministic, its inputs are the whole story of a run. [`log::Logged`]
//! wraps a policy and records every input and every poll's outputs to an [`EventSink`];
//! [`log::replay`] feeds such a log to a fresh policy and returns what each poll returned, which
//! must equal what was logged. A production run can thus be reproduced, inspected with `explain`,
//! or replayed against another configuration.
//!
//! ```
//! use std::{
//!     sync::{Arc, Mutex},
//!     time::Duration,
//! };
//!
//! use whelm::{
//!     Config, Input, JobSpec, Output, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
//!     log::{self, Event, Logged},
//! };
//!
//! // An in-memory sink; `log::JsonlSink` (feature `log`) writes compressed JSON lines instead.
//! let events = Arc::new(Mutex::new(Vec::<Event>::new()));
//! let mut p = Logged::new(Scheduler::new(Config::default()), events.clone());
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! let job = |id| JobSpec {
//!     id,
//!     ..Default::default()
//! };
//! p.handle(Input::Submit(job(1)), Time::ORIGIN);
//! p.handle(Input::Submit(job(2)), Time::ORIGIN);
//! p.poll(Time::ORIGIN);
//! p.handle(
//!     Input::Done { job: 1, attempt: 1 },
//!     Time(Duration::from_secs(5)),
//! );
//! p.poll(Time(Duration::from_secs(5)));
//!
//! let events = events.lock().unwrap().clone();
//! let polls = log::polls(&events);
//! assert_eq!(
//!     polls[1],
//!     (
//!         Time(Duration::from_secs(5)),
//!         vec![Output::Start {
//!             job: 2,
//!             attempt: 1,
//!             worker: 1
//!         }]
//!     )
//! );
//! let mut fresh = Scheduler::new(Config::default());
//! assert_eq!(log::replay(&mut fresh, events), polls);
//! ```
//!
//! To log a run driven through the DAG layer, log the inner policy:
//! `DagScheduler<Logged<Scheduler>>`. The DAG's own operations are method calls, not inputs.
//!
//! # Testing your integration
//!
//! Determinism makes the policy easy to test against: no clock, threads or network are involved, so
//! a test can script inputs at chosen times and assert the exact outputs. A small harness that
//! applies the inputs of each scripted time and then polls covers most needs, and works for any
//! [`Policy`], boxed or not.
//!
//! ```
//! use std::time::Duration;
//!
//! use whelm::{
//!     Config, DagConfig, DagScheduler, FailKind, Input, JobSpec, MEMORY, Output, Policy,
//!     Resources, SLOTS, Scheduler, Time, WorkerState, gb,
//! };
//!
//! /// Apply the inputs of each time, then poll; every poll's outputs, with its time.
//! fn run(p: &mut dyn Policy, script: Vec<(Time, Vec<Input>)>) -> Vec<(Time, Vec<Output>)> {
//!     let mut polls = Vec::new();
//!     for (now, inputs) in script {
//!         for input in inputs {
//!             p.handle(input, now);
//!         }
//!         polls.push((now, p.poll(now)));
//!     }
//!     polls
//! }
//!
//! let script = || {
//!     let job = |id| {
//!         Input::Submit(JobSpec {
//!             id,
//!             demand: Resources::new().with(MEMORY, gb(4.0)),
//!             ..Default::default()
//!         })
//!     };
//!     let fail = Input::Failed {
//!         job: 1,
//!         attempt: 1,
//!         kind: FailKind::Other,
//!         why: "test".into(),
//!     };
//!     let worker = WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(MEMORY, gb(8.0)).with(SLOTS, 2),
//!         ..Default::default()
//!     };
//!     vec![
//!         (Time::ORIGIN, vec![Input::Worker(worker)]),
//!         (Time(Duration::from_secs(1)), vec![job(1), job(2), job(3)]),
//!         (Time(Duration::from_secs(2)), vec![fail]),
//!         (
//!             Time(Duration::from_secs(3)),
//!             vec![Input::Done { job: 2, attempt: 1 }],
//!         ),
//!     ]
//! };
//!
//! // The same script gives the same outputs, whichever policy wrapper runs it.
//! let mut flat = Scheduler::new(Config::default());
//! let mut boxed: Box<dyn Policy> = Box::new(DagScheduler::new(
//!     DagConfig::default(),
//!     Scheduler::new(Config::default()),
//! ));
//! let polls = run(&mut flat, script());
//! assert_eq!(run(&mut boxed, script()), polls);
//!
//! let start = |job, attempt| Output::Start {
//!     job,
//!     attempt,
//!     worker: 1,
//! };
//! assert_eq!(polls[1].1, [start(1, 1), start(2, 1)]);
//! // The retry keeps job 1's place ahead of job 3.
//! assert_eq!(polls[2], (Time(Duration::from_secs(2)), vec![start(1, 2)]));
//! assert_eq!(polls[3].1, [start(3, 1)]);
//!
//! // Inspect the end state.
//! let stats = flat.stats();
//! assert_eq!((stats.running, stats.placements_total), (2, 4));
//! let memory = &stats.workers[0].headroom[0];
//! assert_eq!(memory, &(MEMORY.name, Some(0)));
//! assert_eq!(
//!     flat.explain(1).unwrap().status,
//!     whelm::Status::Running {
//!         attempts: vec![(2, 1)]
//!     }
//! );
//! ```
//!
//! [`Policy::explain`] and [`Policy::stats`] are the windows into a running policy: the first says
//! why a job is not running (fit, constraints, holds, past failures), as an [`Explanation`] that
//! prints as one line, and the second counts jobs, reservations, placements and every worker's
//! load. Both are cheap enough to log.
//!
//! # Where to look next
//!
//! Each module's page tells the full story of its part; the items it defines are also exported at
//! the crate root.
//!
//! - [`scheduler`]: [`Scheduler`], the placement policy, and how a poll scans and places.
//! - [`config`]: [`Config`], its presets, and every order and score term, retry, speed and
//!   reservation setting.
//! - [`resources`]: resource kinds ([`Resource`], the standard [`MEMORY`], [`DEVICE_MEMORY`] and
//!   [`SLOTS`]) and the amounts of them, [`Resources`].
//! - [`admission`]: the [`Admission`] contract, [`ProductionAdmission`], and the [`WorkerView`]
//!   and [`Amounts`] a rule reads.
//! - [`speed`]: machine models ([`Timing`]) and speed learning ([`Learn`], [`SpeedEstimator`], the
//!   last usable on its own).
//! - [`dag`]: [`DagScheduler`], templates, units and [`NodeSource`].
//! - [`shared`]: [`SharedPolicy`], the blocking front end for a thread per task.
//! - [`explain`]: what [`Policy::explain`] reports: [`Explanation`] and the per-worker
//!   [`Verdict`]s.
//! - [`log`]: event logs, sinks and replay.
//! - [`time`]: [`Time`], the points on the caller's clock that every call carries.
//! - [`nassau`]: helpers for driving a Nassau resolution.
//! - This page: the message types ([`Input`], [`Output`], [`Policy`]) and the job and worker
//!   descriptions ([`JobSpec`], [`WorkerState`]).
//!
//! The repository's README gives the scheduling problem in α|β|γ notation and the plan for
//! integrating with Nassau's coordinator. The sibling crate `whelm-sim` replays logged traces
//! against this crate's policies, holds the simulators that model whole runs, and records the
//! measurements behind the defaults in its RESULTS.md.
#![warn(missing_docs)]

pub mod admission;
pub mod config;
pub mod dag;
pub mod explain;
pub mod job;
pub mod log;
pub mod message;
pub mod nassau;
pub mod resources;
pub mod scheduler;
pub mod shared;
pub mod speed;
pub mod stats;
pub mod time;
pub mod worker;

pub use admission::{Admission, Amounts, ProductionAdmission, Usage, WorkerView};
pub use config::{
    Config, DEFAULT_AGE_LIMIT, Defer, GroupOrder, OrderTerm, Reservations, RetryConfig, ScoreTerm,
    Speculate, SpeedConfig,
};
#[cfg(feature = "serde")]
pub use dag::DagSnapshot;
pub use dag::{
    DagConfig, DagError, DagJob, DagScheduler, DagStats, DagTemplate, NodeSource, TemplateNode,
    TemplateSpec, Unit,
};
#[doc(inline)]
pub use explain::{Explanation, Holding, Status, Verdict, Waiting};
#[doc(inline)]
pub use job::{Constraint, JobId, JobSpec, Selector, Strength};
pub use log::EventSink;
#[doc(inline)]
pub use message::{Attempt, FailKind, GaveUp, Input, Output, Policy, Rejection, Tried};
#[doc(inline)]
pub use resources::{DEVICE_MEMORY, MEMORY, Resource, ResourceUnit, Resources, SLOTS, gb};
pub use scheduler::Scheduler;
pub use shared::{Lease, SharedPolicy};
pub use speed::{Learn, Sharing, SpeedEstimator, Timing};
#[doc(inline)]
pub use stats::{PolicyStats, ReservationInfo, WorkerLoad};
pub use time::Time;
#[doc(inline)]
pub use worker::{WorkerId, WorkerState};

/// The README's examples, compiled and run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
