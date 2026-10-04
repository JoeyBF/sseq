//! Pure, deterministic, resource-aware job placement.
//!
//! This page introduces the crate: what it is, the event loop that drives it, and the messages that
//! loop exchanges. Each module's page is then a chapter on one part, and the
//! [reading guide](#reading-guide) at the end lists them in order. Every example is a test that
//! asserts what the crate really does.
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
//! A worker joins, a job is submitted, and [`poll`](policy::Policy::poll) says where to run it.
//! Reporting the attempt [`Done`](policy::Input::Done) frees its slot and the scheduler forgets the
//! job. [`whelm::prelude`](prelude) holds the names this loop uses; everything else is imported
//! from its module.
//!
//! ```
//! use std::time::Duration;
//!
//! use whelm::prelude::*;
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
//! let spec = JobSpec {
//!     demand: Resources::new().with(MEMORY, gb(2.0)),
//!     group: 3,
//!     ..Default::default()
//! };
//! policy.handle(Input::Submit { job: 7, spec }, Time::ORIGIN);
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
//! Every input and poll carries the current [`Time`]: the [`Duration`](std::time::Duration) since
//! an origin the caller picks, here the start of the run, [`Time::ORIGIN`]. The examples write a
//! point as `Time::ORIGIN + span` or `Time(span)`; spans, such as work estimates and waiting
//! bounds, are `Duration`s.
//!
//! Workers and jobs, like configurations, are plain structs written as literals: name the fields
//! that matter and take the rest from [`Default`]. Each type's `Default` documents what the
//! omitted fields mean; a [`WorkerState`] needs little more than an id and its capacity, and a
//! [`JobSpec`] often nothing at all. A job's id is not part of its spec: the caller picks it and
//! passes it beside the spec in [`Input::Submit`], as every other message about the job names it.
//!
//! That is the whole protocol: [`handle`](policy::Policy::handle) every event as it happens, then
//! [`poll`](policy::Policy::poll) and act on each output. `handle` applies an input at once but
//! never places anything; placement happens in `poll`, which also returns the outputs earlier
//! inputs caused. A real caller matches on the outputs; the next example sends each start to a
//! stand-in for the workers and reports completions back.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! let mut policy = Scheduler::new(Config::default());
//! let worker = WorkerState {
//!     id: 1,
//!     capacity: Resources::new().with(SLOTS, 2),
//!     ..Default::default()
//! };
//! policy.handle(Input::Worker(worker), Time::ORIGIN);
//! for job in 1..=3 {
//!     let spec = JobSpec::default();
//!     policy.handle(Input::Submit { job, spec }, Time::ORIGIN);
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
//! # use whelm::prelude::*;
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 4),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! let submit = Input::Submit {
//!     job: 1,
//!     spec: JobSpec::default(),
//! };
//!
//! p.handle(submit.clone(), Time::ORIGIN);
//! p.handle(submit.clone(), Time::ORIGIN); // already waiting: ignored
//! assert_eq!(
//!     p.poll(Time::ORIGIN),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! p.handle(submit.clone(), Time(Duration::from_secs(1))); // already running: ignored
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
//! p.handle(submit, Time(Duration::from_secs(3)));
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
//! # use whelm::prelude::*;
//! # use whelm::policy::FailKind;
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
//! for job in [5, 6] {
//!     let spec = JobSpec::default();
//!     p.handle(Input::Submit { job, spec }, Time::ORIGIN);
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
//! p.handle(
//!     Input::Done { job: 6, attempt: 1 },
//!     Time(Duration::from_secs(20)),
//! );
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
//! [`retryable`](policy::GaveUp::retryable) when every attempt ran out of device memory, the one
//! failure that a smaller job, or a later attempt on a less loaded worker, may avoid.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! # use whelm::config::RetryConfig;
//! # use whelm::policy::{FailKind, GaveUp, Tried};
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
//!     Input::Submit {
//!         job: 5,
//!         spec: JobSpec::default(),
//!     },
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
//! attempt's resources at once and asks the caller to [`Stop`](policy::Output::Stop) it.
//!
//! ```
//! # use std::time::Duration;
//! # use whelm::prelude::*;
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState {
//!         id: 1,
//!         capacity: Resources::new().with(SLOTS, 1),
//!         ..Default::default()
//!     }),
//!     Time::ORIGIN,
//! );
//! for job in [1, 2] {
//!     p.handle(
//!         Input::Submit {
//!             job,
//!             spec: JobSpec::default(),
//!         },
//!         Time::ORIGIN,
//!     );
//! }
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
//! # use whelm::prelude::*;
//! # use whelm::policy::FailKind;
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
//!     Input::Submit {
//!         job: 5,
//!         spec: JobSpec::default(),
//!     },
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
//! p.handle(
//!     Input::Done { job: 5, attempt: 1 },
//!     Time(Duration::from_secs(61)),
//! ); // stale: ignored
//! assert!(p.poll(Time(Duration::from_secs(61))).is_empty());
//! assert_eq!(
//!     p.explain(5).unwrap().status,
//!     whelm::explain::Status::Running {
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
//! # use whelm::prelude::*;
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
//!     Input::Submit {
//!         job: 5,
//!         spec: JobSpec::default(),
//!     },
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
//!     dag::{DagConfig, DagScheduler},
//!     policy::FailKind,
//!     prelude::*,
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
//!     let spec = JobSpec {
//!         demand: Resources::new().with(MEMORY, gb(4.0)),
//!         ..Default::default()
//!     };
//!     let submit = |job| Input::Submit {
//!         job,
//!         spec: spec.clone(),
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
//!         (
//!             Time(Duration::from_secs(1)),
//!             vec![submit(1), submit(2), submit(3)],
//!         ),
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
//!     whelm::explain::Status::Running {
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
//! # Reading guide
//!
//! Each module's page is a chapter on one part of the crate, best read in this order.
//!
//! 1. [`prelude`]: the names the event loop uses, for a glob import.
//! 2. [`policy`]: the [`Policy`] trait and its messages, whose protocol this page describes.
//! 3. [`time`]: [`Time`], the point on the caller's clock that every call carries.
//! 4. [`job`] and [`worker`]: what the caller says about a job ([`JobSpec`], with the constraints
//!    on where it may run) and about a worker ([`WorkerState`]).
//! 5. [`resources`]: declaring the resources workers have and jobs use, and amounts of them.
//! 6. [`admission`]: whether a worker takes a job, as the default rule decides it or a rule of your
//!    own.
//! 7. [`explain`]: why a job is where it is, as [`Policy::explain`] reports it.
//! 8. [`config`]: which waiting job goes first and to which worker, with the presets for common
//!    objectives.
//! 9. [`scheduler`]: [`Scheduler`], the placement policy: how a poll places jobs, and how aging,
//!    reservations and other holds bound the wait of a job that keeps losing.
//! 10. [`speed`]: workers of different speeds, learned or reported, and waiting for or racing on a
//!     faster one.
//! 11. [`stats`]: the counts and per-worker loads [`Policy::stats`] reports.
//! 12. [`dag`]: dependencies between jobs, in a layer in front of any policy.
//! 13. [`shared`]: a blocking front end for callers with a thread per task.
//! 14. [`log`]: recording a run and replaying it.
//! 15. [`nassau`]: helpers for driving a Nassau resolution.
//!
//! The repository's README gives the scheduling problem in α|β|γ notation and the plan for
//! integrating with Nassau's coordinator. The sibling crate `whelm-sim` replays logged traces
//! against this crate's policies, holds the simulators that model whole runs, and records the
//! measurements behind the defaults in its RESULTS.md.
//!
//! [`Attempt`]: policy::Attempt
//! [`DagScheduler`]: dag::DagScheduler
//! [`Explanation`]: explain::Explanation
//! [`FailKind::LinkDied`]: policy::FailKind::LinkDied
//! [`GaveUp`]: policy::GaveUp
//! [`Input`]: policy::Input
//! [`Input::Cancel`]: policy::Input::Cancel
//! [`Input::Done`]: policy::Input::Done
//! [`Input::Failed`]: policy::Input::Failed
//! [`Input::Submit`]: policy::Input::Submit
//! [`Input::WorkerGone`]: policy::Input::WorkerGone
//! [`JobSpec`]: job::JobSpec
//! [`Output`]: policy::Output
//! [`Output::GaveUp`]: policy::Output::GaveUp
//! [`Policy`]: policy::Policy
//! [`Policy::explain`]: policy::Policy::explain
//! [`Policy::stats`]: policy::Policy::stats
//! [`RetryConfig::max_attempts`]: config::RetryConfig::max_attempts
//! [`Scheduler`]: scheduler::Scheduler
//! [`SharedPolicy`]: shared::SharedPolicy
//! [`Time`]: time::Time
//! [`Time::ORIGIN`]: time::Time::ORIGIN
//! [`Tried`]: policy::Tried
//! [`WorkerState`]: worker::WorkerState
#![warn(missing_docs)]

pub mod admission;
pub mod config;
pub mod dag;
pub mod explain;
pub mod job;
pub mod log;
pub mod nassau;
pub mod policy;
pub mod prelude;
pub mod resources;
pub mod scheduler;
pub mod shared;
pub mod speed;
pub mod stats;
pub mod time;
pub mod worker;

/// The README's examples, compiled and run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
