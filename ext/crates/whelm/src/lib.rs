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
//! use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//!
//! let mut policy = Scheduler::new(Config::default());
//!
//! // Worker 1 joins: class "cpu", 4 execution slots, 16 GB of host memory.
//! let worker = WorkerState::new(1, "cpu", 4, Resources::mem_gb(16.0));
//! policy.handle(Input::Worker(worker), 0.0);
//!
//! // Job 7 becomes ready: it expects to use 2 GB and belongs to group 0.
//! policy.handle(
//!     Input::Submit(JobSpec::new(7, Resources::mem_gb(2.0), 0)),
//!     0.0,
//! );
//!
//! // After a batch of inputs, poll: start attempt 1 of job 7 on worker 1.
//! assert_eq!(
//!     policy.poll(0.0),
//!     [Output::Start {
//!         job: 7,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // Thirty seconds later the worker reports success. Nothing is left to do.
//! policy.handle(Input::Done { job: 7, attempt: 1 }, 30.0);
//! assert!(policy.poll(30.0).is_empty());
//! assert_eq!(policy.explain(7), None); // the job is forgotten
//! assert_eq!(policy.stats().placements_total, 1);
//! ```
//!
//! That is the whole protocol: [`handle`](Policy::handle) every event as it happens, then
//! [`poll`](Policy::poll) and act on each output. `handle` applies an input at once but never
//! places anything; placement happens in `poll`, which also returns the outputs earlier inputs
//! caused. A real caller matches on the outputs; the next example sends each start to a stand-in
//! for the workers and reports completions back.
//!
//! ```
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut policy = Scheduler::new(Config::default());
//! policy.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 2, Resources::ZERO)),
//!     0.0,
//! );
//! for id in 1..=3 {
//!     policy.handle(Input::Submit(JobSpec::new(id, Resources::ZERO, 0)), 0.0);
//! }
//!
//! // The caller's side: attempts sent to workers and not yet answered.
//! let mut in_flight = Vec::new();
//! let mut now = 0.0;
//! while now < 10.0 {
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
//!     now += 1.0;
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
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 4, Resources::ZERO)),
//!     0.0,
//! );
//! let job = JobSpec::new(1, Resources::ZERO, 0);
//!
//! p.handle(Input::Submit(job.clone()), 0.0);
//! p.handle(Input::Submit(job.clone()), 0.0); // already waiting: ignored
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! p.handle(Input::Submit(job.clone()), 1.0); // already running: ignored
//! assert!(p.poll(1.0).is_empty());
//!
//! p.handle(Input::Done { job: 1, attempt: 1 }, 2.0);
//! p.handle(Input::Done { job: 1, attempt: 1 }, 2.0); // no longer live: ignored
//! assert!(p.poll(2.0).is_empty());
//!
//! // Once complete, the id is free again: a new submission is a new job.
//! p.handle(Input::Submit(job), 3.0);
//! assert_eq!(
//!     p.poll(3.0),
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
//! # use whelm::{
//! #     Config, FailKind, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState
//! # };
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState::new(id, "cpu", 1, Resources::ZERO)),
//!         0.0,
//!     );
//! }
//! let fail = |job, attempt, why: &str| Input::Failed {
//!     job,
//!     attempt,
//!     kind: FailKind::Other,
//!     why: why.into(),
//! };
//!
//! p.handle(Input::Submit(JobSpec::new(5, Resources::ZERO, 0)), 0.0);
//! p.handle(Input::Submit(JobSpec::new(6, Resources::ZERO, 0)), 0.0);
//! assert_eq!(
//!     p.poll(0.0),
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
//! p.handle(fail(5, 1, "segfault"), 10.0);
//! assert!(p.poll(10.0).is_empty());
//! assert_eq!(
//!     p.explain(5).unwrap(),
//!     "job 5 (demand 0.00 GB, group 0) waiting 10s, 0 more urgent job(s) waiting; failed 1 \
//!      time(s), last on worker 1 (Other: segfault); slots full on 1 worker(s); 1 worker(s) \
//!      excluded by its constraints"
//! );
//!
//! // Worker 2 frees up: attempt 2 runs there.
//! p.handle(Input::Done { job: 6, attempt: 1 }, 20.0);
//! assert_eq!(
//!     p.poll(20.0),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 2,
//!         worker: 2
//!     }]
//! );
//!
//! // It fails there too. Every worker is now avoided, so the avoidance lapses.
//! p.handle(fail(5, 2, "segfault"), 30.0);
//! assert_eq!(
//!     p.poll(30.0),
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
//! # use whelm::{
//! #     Config, FailKind, GaveUp, Input, JobSpec, Output, Policy, Resources, RetryConfig,
//! #     Scheduler, Tried, WorkerState,
//! # };
//! let config = Config {
//!     retry: RetryConfig { max_attempts: 2 },
//!     ..Config::default()
//! };
//! let mut p = Scheduler::new(config);
//! p.handle(
//!     Input::Worker(WorkerState::new(1, "gpu", 1, Resources::ZERO)),
//!     0.0,
//! );
//! p.handle(Input::Submit(JobSpec::new(5, Resources::ZERO, 0)), 0.0);
//!
//! let oom = |attempt| Input::Failed {
//!     job: 5,
//!     attempt,
//!     kind: FailKind::DeviceOom,
//!     why: "out of device memory".into(),
//! };
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! p.handle(oom(1), 1.0);
//! assert_eq!(
//!     p.poll(1.0),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 2,
//!         worker: 1
//!     }]
//! );
//! p.handle(oom(2), 2.0);
//!
//! let tried = Tried {
//!     worker: 1,
//!     kind: FailKind::DeviceOom,
//!     why: "out of device memory".into(),
//! };
//! assert_eq!(
//!     p.poll(2.0),
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
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)),
//!     0.0,
//! );
//! p.handle(Input::Submit(JobSpec::new(1, Resources::ZERO, 0)), 0.0);
//! p.handle(Input::Submit(JobSpec::new(2, Resources::ZERO, 0)), 0.0);
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! p.handle(Input::Cancel(2), 1.0); // waiting: dropped
//! p.handle(Input::Cancel(1), 1.0); // running: stopped
//! assert_eq!(
//!     p.poll(1.0),
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
//! # use whelm::{
//! #     Config, FailKind, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState
//! # };
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState::new(id, "cpu", 1, Resources::ZERO)),
//!         0.0,
//!     );
//! }
//! p.handle(Input::Submit(JobSpec::new(5, Resources::ZERO, 0)), 0.0);
//! assert_eq!(
//!     p.poll(0.0),
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
//! p.handle(timeout, 60.0);
//! assert_eq!(
//!     p.poll(60.0),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 2,
//!         worker: 2
//!     }]
//! );
//!
//! p.handle(Input::Done { job: 5, attempt: 1 }, 61.0); // stale: ignored
//! assert!(p.poll(61.0).is_empty());
//! assert_eq!(
//!     p.explain(5).unwrap(),
//!     "job 5 is running: attempt 2 on worker 2"
//! );
//! ```
//!
//! When a worker leaves ([`Input::WorkerGone`]), each attempt on it fails as if reported with
//! [`FailKind::LinkDied`], and is retried like any other failure. The caller does not resubmit.
//!
//! ```
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState::new(id, "cpu", 1, Resources::ZERO)),
//!         0.0,
//!     );
//! }
//! p.handle(Input::Submit(JobSpec::new(5, Resources::ZERO, 0)), 0.0);
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 5,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! p.handle(Input::WorkerGone(1), 5.0);
//! assert_eq!(
//!     p.poll(5.0),
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
//! A [`Resources`] vector has one component per dimension: host memory ([`MEM`]) and device memory
//! ([`DEV`]) in bytes, and execution slots ([`SLOTS`]). A job's [`demand`](JobSpec::demand) is what
//! it is expected to use; a worker's [`budget`](WorkerState::budget) and slot count are what it
//! has. Every job takes exactly one slot, which the scheduler fills in itself.
//!
//! ```
//! use whelm::{DEV, MEM, Resources, SLOTS};
//!
//! let demand = Resources::mem_gb(6.0).with_dev_gb(2.0);
//! assert_eq!(
//!     (demand[MEM], demand[DEV], demand[SLOTS]),
//!     (6_000_000_000, 2_000_000_000, 0)
//! );
//! assert!(demand.fits_within(&Resources::mem_gb(8.0).with_dev_gb(2.0)));
//! // Arithmetic saturates: bookkeeping never goes below zero.
//! assert_eq!(
//!     demand - Resources::mem_gb(10.0),
//!     Resources::ZERO.with_dev_gb(2.0)
//! );
//! ```
//!
//! Whether a worker takes a job is up to an [`Admission`] rule; the default,
//! [`ProductionAdmission`], admits a job if, in every dimension, what the worker already uses plus
//! the job's demand fits its capacity. A job that fits nowhere waits, and
//! [`explain`](Policy::explain) says why.
//!
//! ```
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 2, Resources::mem_gb(10.0))),
//!     0.0,
//! );
//! p.handle(
//!     Input::Submit(JobSpec::new(1, Resources::mem_gb(6.0), 0)),
//!     0.0,
//! );
//! p.handle(
//!     Input::Submit(JobSpec::new(2, Resources::mem_gb(6.0), 0)),
//!     0.0,
//! );
//!
//! // A slot is free, but 6 + 6 GB exceeds the 10 GB budget.
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! assert_eq!(
//!     p.explain(2).unwrap(),
//!     "job 2 (demand 6.00 GB, group 0) waiting 0s, 0 more urgent job(s) waiting; memory short \
//!      on 1 worker(s) (best headroom 4.00 GB on worker 1)"
//! );
//!
//! p.handle(Input::Done { job: 1, attempt: 1 }, 50.0);
//! assert_eq!(
//!     p.poll(50.0),
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
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! let worker = WorkerState::new(1, "cpu", 4, Resources::mem_gb(10.0));
//! p.handle(Input::Worker(worker.clone()), 0.0);
//! p.handle(
//!     Input::Submit(JobSpec::new(1, Resources::mem_gb(3.0), 0)),
//!     0.0,
//! );
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // Heartbeat: 8 GB resident, 1 GB of it the worker's own runtime.
//! let heartbeat = WorkerState {
//!     reported_used: Resources::mem_gb(8.0),
//!     reported_baseline: Resources::mem_gb(1.0),
//!     ..worker
//! };
//! p.handle(Input::Worker(heartbeat), 5.0);
//! p.handle(
//!     Input::Submit(JobSpec::new(2, Resources::mem_gb(3.0), 0)),
//!     5.0,
//! );
//! // max(8, 1 + 3) + 3 = 11 GB > 10 GB.
//! assert!(p.poll(5.0).is_empty());
//! ```
//!
//! Memory is a *soft* dimension. A job alone on a worker always runs, whatever its estimate (the
//! escape hatch, so that every job can run somewhere), and a zero memory capacity means "unknown"
//! and is not enforced. Slots are [`HARD`]: always enforced, with no escape hatch.
//!
//! ```
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 4, Resources::mem_gb(10.0))),
//!     0.0,
//! );
//! p.handle(
//!     Input::Submit(JobSpec::new(1, Resources::mem_gb(50.0), 0)),
//!     0.0,
//! );
//! p.handle(
//!     Input::Submit(JobSpec::new(2, Resources::mem_gb(1.0), 0)),
//!     0.0,
//! );
//! // Job 1 is five times the budget but runs alone; job 2 must wait for it.
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // A worker of unknown memory: only its two slots limit it.
//! p.handle(
//!     Input::Worker(WorkerState::new(2, "cpu", 2, Resources::ZERO)),
//!     1.0,
//! );
//! p.handle(
//!     Input::Submit(JobSpec::new(3, Resources::mem_gb(500.0), 0)),
//!     1.0,
//! );
//! assert_eq!(
//!     p.poll(1.0),
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
//! p.handle(Input::Submit(JobSpec::new(4, Resources::ZERO, 0)), 2.0);
//! assert!(p.poll(2.0).is_empty());
//! assert!(p.explain(4).unwrap().contains("slots full on 1 worker(s)"));
//! ```
//!
//! A worker can also declare a [`per_task`](WorkerState::per_task) floor: what any one job takes
//! there at least, whatever its demand says. With a device-memory floor and jobs that declare no
//! device demand, the device budget simply counts jobs. [`PolicyStats::workers`] shows each
//! worker's load and headroom as admission sees it.
//!
//! ```
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! // 8 slots, unknown host memory, 10 GB of device memory, at least 4 GB of it per job.
//! let gpu = WorkerState {
//!     per_task: Resources::ZERO.with_dev_gb(4.0),
//!     ..WorkerState::new(1, "gpu", 8, Resources::ZERO.with_dev_gb(10.0))
//! };
//! p.handle(Input::Worker(gpu), 0.0);
//! for id in 1..=3 {
//!     p.handle(Input::Submit(JobSpec::new(id, Resources::ZERO, 0)), 0.0);
//! }
//! // (running + 1) * 4 GB <= 10 GB admits two jobs.
//! assert_eq!(p.poll(0.0).len(), 2);
//! let load = &p.stats().workers[0];
//! assert_eq!(load.running, 2);
//! assert_eq!(load.headroom, [None, Some(2_000_000_000), Some(6)]);
//! ```
//!
//! The rule can be called directly on a [`WorkerView`]: a worker's state plus what the scheduler
//! has placed on it. Its methods give the pieces of the rule, such as the
//! [`headroom`](WorkerView::headroom) per dimension and the [`free_share`](WorkerView::free_share)
//! that scores compare workers by.
//!
//! ```
//! use whelm::{Admission, ProductionAdmission, Resources, SLOTS, WorkerState, WorkerView};
//!
//! let state = WorkerState::new(1, "cpu", 4, Resources::mem_gb(10.0));
//! let mut placed = Resources::mem_gb(6.0);
//! placed[SLOTS] = 1; // one job running
//! let view = WorkerView {
//!     state: &state,
//!     placed,
//! };
//!
//! assert!(ProductionAdmission.admits(&Resources::mem_gb(4.0), &view));
//! assert!(!ProductionAdmission.admits(&Resources::mem_gb(5.0), &view));
//! assert_eq!(view.headroom(), [Some(4_000_000_000), None, Some(3)]);
//! // After placing 2 GB more, a fifth of the memory would be left.
//! assert_eq!(view.free_share(&Resources::mem_gb(2.0)), 0.2);
//! ```
//!
//! A different rule plugs in with [`Scheduler::with_admission`]. It must be monotone in load (see
//! [`Admission`]) and it alone enforces capacity, slots included. This one trusts no memory figure
//! and counts slots only.
//!
//! ```
//! use whelm::{
//!     Admission, Config, Input, JobSpec, Policy, Resources, Scheduler, WorkerState, WorkerView,
//! };
//!
//! /// Admits while a slot is free, whatever the memory figures say.
//! struct SlotsOnly;
//!
//! impl Admission for SlotsOnly {
//!     fn admits(&self, _demand: &Resources, w: &WorkerView) -> bool {
//!         w.running() < w.state.slots as u64
//!     }
//! }
//!
//! let mut p = Scheduler::with_admission(Config::default(), SlotsOnly);
//! p.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 2, Resources::mem_gb(10.0))),
//!     0.0,
//! );
//! for id in 1..=3 {
//!     p.handle(
//!         Input::Submit(JobSpec::new(id, Resources::mem_gb(50.0), 0)),
//!         0.0,
//!     );
//! }
//! assert_eq!(p.poll(0.0).len(), 2);
//! ```
//!
//! # Constraints
//!
//! A job's [`constraints`](JobSpec::constraints) restrict where it runs. Each names workers with a
//! [`Selector`] (one worker, or a class of workers) and binds with a [`Strength`]: `Require` and
//! `Forbid` are hard, `Avoid` is soft, and `Prefer` only ranks the workers that admit the job. The
//! builders on [`JobSpec`] cover the common cases.
//!
//! ```
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! for (id, class) in [(1, "cpu"), (2, "gpu"), (3, "gpu")] {
//!     p.handle(
//!         Input::Worker(WorkerState::new(id, class, 4, Resources::ZERO)),
//!         0.0,
//!     );
//! }
//! let job = |id| JobSpec::new(id, Resources::ZERO, 0);
//! p.handle(Input::Submit(job(1).require_class("gpu")), 0.0);
//! p.handle(
//!     Input::Submit(job(2).require_class("gpu").forbid_worker(2)),
//!     0.0,
//! );
//! assert_eq!(
//!     p.poll(0.0),
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
//! p.handle(Input::Submit(job(3).require_class("tpu")), 1.0);
//! assert!(p.poll(1.0).is_empty());
//! assert!(
//!     p.explain(3)
//!         .unwrap()
//!         .ends_with("; 3 worker(s) excluded by its constraints")
//! );
//! ```
//!
//! A preferred worker wins over a less loaded one: the default score ranks
//! [`Preferred`](ScoreTerm::Preferred) before [`Load`](ScoreTerm::Load) (see
//! [Ordering](#ordering)). Use it for cache affinity.
//!
//! ```
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState::new(id, "cpu", 4, Resources::ZERO)),
//!         0.0,
//!     );
//! }
//! p.handle(Input::Submit(JobSpec::new(1, Resources::ZERO, 0)), 0.0);
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//!
//! // Worker 1 is busier, but job 2 prefers it; job 3 has no preference.
//! p.handle(
//!     Input::Submit(JobSpec::new(2, Resources::ZERO, 0).prefer_worker(1)),
//!     1.0,
//! );
//! p.handle(Input::Submit(JobSpec::new(3, Resources::ZERO, 0)), 1.0);
//! assert_eq!(
//!     p.poll(1.0),
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
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! for id in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState::new(id, "cpu", 1, Resources::ZERO)),
//!         0.0,
//!     );
//! }
//! p.handle(
//!     Input::Submit(JobSpec::new(1, Resources::ZERO, 0).prefer_worker(2)),
//!     0.0,
//! );
//! p.handle(
//!     Input::Submit(JobSpec::new(2, Resources::ZERO, 0).avoid_worker(1)),
//!     0.0,
//! );
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 2
//!     }]
//! );
//! assert!(
//!     p.explain(2)
//!         .unwrap()
//!         .contains("1 worker(s) excluded by its constraints")
//! );
//!
//! p.handle(
//!     Input::Worker(WorkerState::new(2, "cpu", 0, Resources::ZERO)),
//!     1.0,
//! );
//! assert_eq!(
//!     p.poll(1.0),
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
//! use whelm::{
//!     Config, GroupOrder, Input, JobId, JobSpec, Output, Policy, Resources, Scheduler,
//!     WorkerState,
//! };
//!
//! /// The order one single-slot worker runs three jobs in under `config`.
//! fn run_order(config: Config) -> Vec<JobId> {
//!     let mut p = Scheduler::new(config);
//!     p.handle(
//!         Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)),
//!         0.0,
//!     );
//!     // (id, group, weight, work, due)
//!     let jobs = [
//!         (1, 5, 1.0, 10.0, 100.0),
//!         (2, 3, 1.0, 1.0, 50.0),
//!         (3, 5, 4.0, 5.0, 20.0),
//!     ];
//!     for (id, group, weight, work, due) in jobs {
//!         let spec = JobSpec {
//!             weight,
//!             work: Some(work),
//!             due: Some(due),
//!             ..JobSpec::new(id, Resources::ZERO, group)
//!         };
//!         p.handle(Input::Submit(spec), 0.0);
//!     }
//!     let (mut order, mut now) = (Vec::new(), 0.0);
//!     while order.len() < 3 {
//!         for out in p.poll(now) {
//!             if let Output::Start { job, attempt, .. } = out {
//!                 order.push(job);
//!                 now += 1.0;
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
//! // Largest weight / work first (Smith's rule): 1.0, 0.8, 0.1.
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
//! # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! p.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)), 0.0);
//! for (id, priority) in [(1, Some(1)), (2, None), (3, Some(-1))] {
//!     let spec = JobSpec { priority, ..JobSpec::new(id, Resources::ZERO, 0) };
//!     p.handle(Input::Submit(spec), 0.0);
//! }
//! let mut order: Vec<JobId> = Vec::new();
//! for now in 0..3 {
//!     for out in p.poll(now as f64) {
//!         if let Output::Start { job, attempt, .. } = out {
//!             order.push(job);
//!             p.handle(Input::Done { job, attempt }, now as f64 + 0.5);
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
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! /// The worker a 10 GB job goes to, given a 100 GB worker 1 and a 20 GB worker 2.
//! fn place(config: Config) -> u64 {
//!     let mut p = Scheduler::new(config);
//!     p.handle(
//!         Input::Worker(WorkerState::new(1, "big", 4, Resources::mem_gb(100.0))),
//!         0.0,
//!     );
//!     p.handle(
//!         Input::Worker(WorkerState::new(2, "small", 4, Resources::mem_gb(20.0))),
//!         0.0,
//!     );
//!     p.handle(
//!         Input::Submit(JobSpec::new(1, Resources::mem_gb(10.0), 0)),
//!         0.0,
//!     );
//!     match p.poll(0.0)[..] {
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
//! # use whelm::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! let mut p = Scheduler::new(Config::default());
//! p.handle(
//!     Input::Worker(WorkerState::new(1, "old", 4, Resources::ZERO)),
//!     0.0,
//! );
//! let fast = WorkerState {
//!     speed: 2.5,
//!     ..WorkerState::new(2, "new", 4, Resources::ZERO)
//! };
//! p.handle(Input::Worker(fast), 0.0);
//! p.handle(Input::Submit(JobSpec::new(1, Resources::ZERO, 0)), 0.0);
//! assert_eq!(
//!     p.poll(0.0),
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
//! # use whelm::{Config, Input, JobId, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};
//! /// The job that runs after job 1, under an age limit.
//! fn second(age_limit: Option<f64>) -> JobId {
//!     let mut p = Scheduler::new(Config { age_limit, reservations: None, ..Config::default() });
//!     p.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)), 0.0);
//!     p.handle(Input::Submit(JobSpec::new(1, Resources::ZERO, 0)), 0.0);
//!     p.handle(Input::Submit(JobSpec::new(2, Resources::ZERO, 0)), 0.0);
//!     p.poll(0.0);
//!     let urgent = JobSpec { priority: Some(-1), ..JobSpec::new(3, Resources::ZERO, 0) };
//!     p.handle(Input::Submit(urgent), 50.0);
//!     p.handle(Input::Done { job: 1, attempt: 1 }, 150.0);
//!     match p.poll(150.0)[..] {
//!         [Output::Start { job, .. }] => job,
//!         ref out => panic!("{out:?}"),
//!     }
//! }
//! assert_eq!(second(Some(100.0)), 2);
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
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, ReservationInfo, Resources, Scheduler, WorkerState
//! # };
//! let mut p = Scheduler::new(Config::default()); // reserve after 60 s
//! p.handle(Input::Worker(WorkerState::new(1, "cpu", 3, Resources::mem_gb(10.0))), 0.0);
//! let job = |id, gb| Input::Submit(JobSpec::new(id, Resources::mem_gb(gb), 0));
//! let start = |job| Output::Start { job, attempt: 1, worker: 1 };
//! let done = |job| Input::Done { job, attempt: 1 };
//!
//! // Big job 9 needs 8 GB; small jobs 1 and 2 get in first.
//! for input in [job(1, 4.0), job(9, 8.0), job(2, 4.0)] {
//!     p.handle(input, 0.0);
//! }
//! assert_eq!(p.poll(0.0), [start(1), start(2)]);
//!
//! // A small job frees 4 GB; that is not enough for job 9, and another small job takes it.
//! p.handle(done(1), 30.0);
//! p.handle(job(3, 4.0), 30.0);
//! assert_eq!(p.poll(30.0), [start(3)]);
//!
//! // At 60 s, job 9 reserves the worker.
//! assert!(p.poll(60.0).is_empty());
//! let reservation = ReservationInfo { job: 9, worker: 1, since: 60.0 };
//! assert_eq!(p.stats().reservations, [reservation]);
//! assert!(p.explain(9).unwrap().contains("holds the reservation on worker 1"));
//!
//! // Small jobs no longer get in, although they would fit.
//! p.handle(done(2), 70.0);
//! p.handle(job(4, 4.0), 70.0);
//! assert!(p.poll(70.0).is_empty());
//! assert!(p.explain(4).unwrap().ends_with("; reserved: worker 1 for job 9"));
//!
//! // Once enough has drained, the holder runs.
//! p.handle(done(3), 90.0);
//! assert_eq!(p.poll(90.0), [start(9)]);
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
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Reservations, Resources, Scheduler, WorkerState
//! # };
//! let reservations = Reservations { shadow_backfill: true, ..Reservations::default() };
//! let mut p = Scheduler::new(Config { reservations: Some(reservations), ..Config::default() });
//! p.handle(Input::Worker(WorkerState::new(1, "cpu", 3, Resources::mem_gb(10.0))), 0.0);
//! let job = |id, gb, work| JobSpec {
//!     work: Some(work),
//!     ..JobSpec::new(id, Resources::mem_gb(gb), 0)
//! };
//!
//! p.handle(Input::Submit(job(1, 4.0, 100.0)), 0.0);
//! p.handle(Input::Submit(job(9, 8.0, 100.0)), 0.0);
//! assert_eq!(p.poll(0.0), [Output::Start { job: 1, attempt: 1, worker: 1 }]);
//!
//! p.handle(Input::Submit(job(2, 4.0, 50.0)), 60.0);
//! p.handle(Input::Submit(job(3, 4.0, 30.0)), 60.0);
//! assert_eq!(p.poll(60.0), [Output::Start { job: 3, attempt: 1, worker: 1 }]);
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
//! #     Config, DEFAULT_AGE_LIMIT, Input, JobSpec, Policy, Reservations, Resources, Scheduler,
//! #     WorkerState
//! # };
//! let mut p = Scheduler::new(Config::default());
//! p.handle(Input::Worker(WorkerState::new(1, "cpu", 2, Resources::mem_gb(10.0))), 0.0);
//! p.handle(Input::Submit(JobSpec::new(1, Resources::mem_gb(6.0), 0)), 0.0); // runs for ever
//! p.handle(Input::Submit(JobSpec::new(2, Resources::mem_gb(6.0), 0)), 0.0);
//! p.poll(0.0);
//!
//! // No events arrive: poll whenever the policy asks to.
//! let mut wakeups = Vec::new();
//! while let Some(t) = p.next_wakeup() {
//!     wakeups.push(t);
//!     p.poll(t);
//! }
//! assert_eq!(wakeups, [Reservations::default().reserve_after, DEFAULT_AGE_LIMIT]);
//! assert_eq!(p.stats().reservations[0].job, 2);
//! ```
//!
//! # Speed
//!
//! Workers differ in speed. A job's [`work`](JobSpec::work) is its run time in seconds on a worker
//! of speed 1, so on a worker of speed `s` it is expected to take `work / s`. The machine model,
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
//! #     Config, Input, JobSpec, Output, Policy, Resources, Scheduler, SpeedConfig, Timing,
//! #     WorkerState
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
//!         Input::Worker(WorkerState::new(1, "a", 4, Resources::ZERO)),
//!         0.0,
//!     );
//!     let fast = WorkerState {
//!         speed: 3.0,
//!         ..WorkerState::new(2, "b", 4, Resources::ZERO)
//!     };
//!     p.handle(Input::Worker(fast), 0.0);
//!     p.handle(Input::Submit(JobSpec::new(1, Resources::ZERO, 0)), 0.0);
//!     match p.poll(0.0)[..] {
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
//! # use whelm::{
//! #     Config, Input, JobSpec, Learn, Policy, Resources, Scheduler, SpeedConfig, Timing,
//! #     WorkerState
//! # };
//! let mut p = Scheduler::new(Config {
//!     speed: SpeedConfig {
//!         timing: Timing::learned(),
//!         ..SpeedConfig::default()
//!     },
//!     ..Config::default()
//! });
//! for (id, class) in [(1, "a"), (2, "b")] {
//!     p.handle(
//!         Input::Worker(WorkerState::new(id, class, 1, Resources::ZERO)),
//!         0.0,
//!     );
//! }
//!
//! // Run jobs of 30 s of work on each worker in turn, pinned there by class.
//! let (mut now, mut id) = (0.0, 0);
//! for _ in 0..Learn::default().min_samples {
//!     for (class, true_speed) in [("a", 1.0), ("b", 3.0)] {
//!         let spec = JobSpec {
//!             work: Some(30.0),
//!             ..JobSpec::new(id, Resources::ZERO, 0)
//!         };
//!         p.handle(Input::Submit(spec.require_class(class)), now);
//!         p.poll(now);
//!         now += 30.0 / true_speed;
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
//! # use whelm::{
//! #     Config, Input, JobId, JobSpec, Output, Policy, Resources, Scheduler, SpeedConfig, Timing,
//! #     WorkerState
//! # };
//! /// Workers 1 (class x) and 2 (class y) after training: kind "a" runs four times faster on x,
//! /// kind "b" twice as fast on y. Returns the policy and the time.
//! fn trained(timing: Timing) -> (Scheduler, f64) {
//!     let mut p = Scheduler::new(Config {
//!         speed: SpeedConfig { timing, ..SpeedConfig::default() },
//!         ..Config::default()
//!     });
//!     for (id, class) in [(1, "x"), (2, "y")] {
//!         p.handle(Input::Worker(WorkerState::new(id, class, 1, Resources::ZERO)), 0.0);
//!     }
//!     let (mut now, mut id) = (0.0, 0);
//!     for _ in 0..20 {
//!         let runs = [("a", "x", 4.0), ("a", "y", 1.0), ("b", "x", 1.0), ("b", "y", 2.0)];
//!         for (kind, class, true_speed) in runs {
//!             let spec = JobSpec { work: Some(8.0), ..JobSpec::new(id, Resources::ZERO, 0) };
//!             p.handle(Input::Submit(spec.with_kind(kind).require_class(class)), now);
//!             p.poll(now);
//!             now += 8.0 / true_speed;
//!             p.handle(Input::Done { job: id, attempt: 1 }, now);
//!             id += 1;
//!         }
//!     }
//!     (p, now)
//! }
//!
//! /// Where a lone job of `kind` goes once trained.
//! fn place(timing: Timing, kind: &str) -> u64 {
//!     let (mut p, now) = trained(timing);
//!     let spec = JobSpec { work: Some(8.0), ..JobSpec::new(1000, Resources::ZERO, 0) };
//!     p.handle(Input::Submit(spec.with_kind(kind)), now);
//!     match p.poll(now)[..] {
//!         [Output::Start { worker, .. }] => worker,
//!         ref out => panic!("{out:?}"),
//!     }
//! }
//! assert_eq!((place(Timing::unrelated(), "a"), place(Timing::unrelated(), "b")), (1, 2));
//! assert_eq!((place(Timing::learned(), "a"), place(Timing::learned(), "b")), (1, 1));
//! ```
//!
//! ## Waiting for a faster worker
//!
//! The score picks the best worker that admits a job *now*. With [`SpeedConfig::defer`], a job may
//! instead wait for a busy, faster worker on which it would finish sooner (earliest finish time, as
//! in HEFT). The wait is a hold: it shows in [`PolicyStats::deferred`] and in `explain`, and it
//! lapses after [`max_wait`](Defer::max_wait).
//!
//! ```
//! # use whelm::{
//! #     Config, Defer, Input, JobSpec, Output, Policy, Resources, Scheduler, SpeedConfig,
//! #     WorkerState
//! # };
//! let mut p = Scheduler::new(Config {
//!     speed: SpeedConfig {
//!         defer: Some(Defer::default()),
//!         ..SpeedConfig::default()
//!     },
//!     ..Config::default()
//! });
//! p.handle(
//!     Input::Worker(WorkerState::new(1, "slow", 1, Resources::ZERO)),
//!     0.0,
//! );
//! let fast = WorkerState {
//!     speed: 4.0,
//!     ..WorkerState::new(2, "fast", 1, Resources::ZERO)
//! };
//! p.handle(Input::Worker(fast), 0.0);
//! let job = |id, work| JobSpec {
//!     work: Some(work),
//!     ..JobSpec::new(id, Resources::ZERO, 0)
//! };
//!
//! // Job 1 takes the fast worker until 10 / 4 = 2.5 s.
//! p.handle(Input::Submit(job(1, 10.0)), 0.0);
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 1,
//!         attempt: 1,
//!         worker: 2
//!     }]
//! );
//!
//! // Job 2 would take 40 s on the slow worker, or 2.5 + 10 s on the fast one: it waits.
//! p.handle(Input::Submit(job(2, 40.0)), 0.0);
//! assert!(p.poll(0.0).is_empty());
//! assert_eq!(p.stats().deferred, [(2, 2, 2.5)]);
//! assert!(
//!     p.explain(2)
//!         .unwrap()
//!         .contains("waiting for faster worker 2")
//! );
//!
//! p.handle(Input::Done { job: 1, attempt: 1 }, 2.5);
//! assert_eq!(
//!     p.poll(2.5),
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
//! # use whelm::{
//! #     Config, Input, JobSpec, Output, Policy, Resources, Scheduler, Speculate, SpeedConfig,
//! #     WorkerState
//! # };
//! let mut p = Scheduler::new(Config {
//!     speed: SpeedConfig { speculate: Some(Speculate::default()), ..SpeedConfig::default() },
//!     ..Config::default()
//! });
//! p.handle(Input::Worker(WorkerState::new(1, "slow", 1, Resources::ZERO)), 0.0);
//! let spec = JobSpec { work: Some(40.0), ..JobSpec::new(1, Resources::ZERO, 0) };
//! p.handle(Input::Submit(spec), 0.0);
//! assert_eq!(p.poll(0.0), [Output::Start { job: 1, attempt: 1, worker: 1 }]);
//!
//! // A worker four times faster joins at 1 s: done at 11 s rather than 40 s.
//! let fast = WorkerState { speed: 4.0, ..WorkerState::new(2, "fast", 1, Resources::ZERO) };
//! p.handle(Input::Worker(fast), 1.0);
//! assert_eq!(p.poll(1.0), [Output::Start { job: 1, attempt: 2, worker: 2 }]);
//!
//! // The second attempt wins; the first is stopped.
//! p.handle(Input::Done { job: 1, attempt: 2 }, 11.0);
//! assert_eq!(p.poll(11.0), [Output::Stop { job: 1, attempt: 1, worker: 1 }]);
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
//! use whelm::{
//!     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources,
//!     Scheduler, WorkerState,
//! };
//!
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
//! dag.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 4, Resources::ZERO)),
//!     0.0,
//! );
//! let job = |id| JobSpec::new(id, Resources::ZERO, 0);
//! let start = |job| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker: 1,
//! };
//!
//! // A diamond: 1 -> {2, 3} -> 4.
//! let jobs = [
//!     DagJob::new(job(1), vec![]),
//!     DagJob::new(job(2), vec![1]),
//!     DagJob::new(job(3), vec![1]),
//!     DagJob::new(job(4), vec![2, 3]),
//! ];
//! dag.declare(jobs, 0.0).unwrap();
//! assert_eq!(dag.poll(0.0), [start(1)]);
//! assert_eq!(
//!     dag.explain(4).unwrap(),
//!     "job 4 waits for 2 dependencies [2, 3]"
//! );
//!
//! dag.handle(Input::Done { job: 1, attempt: 1 }, 1.0);
//! assert_eq!(dag.poll(1.0), [start(2), start(3)]);
//! dag.handle(Input::Done { job: 2, attempt: 1 }, 2.0);
//! dag.handle(Input::Done { job: 3, attempt: 1 }, 2.0);
//! assert_eq!(dag.poll(2.0), [start(4)]);
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
//! # use whelm::{
//! #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources,
//! #     Scheduler, WorkerState
//! # };
//! let config = DagConfig {
//!     record_passthrough: true,
//!     ..DagConfig::default()
//! };
//! let mut dag = DagScheduler::new(config, Scheduler::new(Config::default()));
//! dag.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 4, Resources::ZERO)),
//!     0.0,
//! );
//! let job = |id| JobSpec::new(id, Resources::ZERO, 0);
//!
//! // Load locally (1), then a barrier (2), then compute (3).
//! let jobs = [
//!     DagJob::new(job(1), vec![]).local(),
//!     DagJob::passthrough(2, 0, vec![1], 0.0),
//!     DagJob::new(job(3), vec![2]),
//! ];
//! dag.declare(jobs, 0.0).unwrap();
//! assert_eq!(dag.poll(0.0), [Output::RunLocal { job: 1 }]);
//!
//! dag.handle(Input::Done { job: 1, attempt: 0 }, 1.0);
//! assert_eq!(
//!     dag.poll(1.0),
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
//! #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources,
//! #     Scheduler, WorkerState
//! # };
//! let config = DagConfig {
//!     auto_submit: false,
//!     ..DagConfig::default()
//! };
//! let mut dag = DagScheduler::new(config, Scheduler::new(Config::default()));
//! dag.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 4, Resources::ZERO)),
//!     0.0,
//! );
//!
//! dag.declare(
//!     [DagJob::new(JobSpec::new(1, Resources::ZERO, 0), vec![])],
//!     0.0,
//! )
//! .unwrap();
//! assert_eq!(dag.announcements(), [Output::Ready { job: 1 }]);
//! assert_eq!(
//!     dag.explain(1).unwrap(),
//!     "job 1 is ready and held until release"
//! );
//!
//! // ... prepare the job's inputs, then hand it over.
//! assert!(dag.release(1, 0.0));
//! assert_eq!(
//!     dag.poll(0.0),
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
//! use std::sync::Arc;
//!
//! use whelm::{
//!     Config, DagConfig, DagScheduler, DagTemplate, Input, JobSpec, Output, Policy, Resources,
//!     Scheduler, TemplateNode, Unit, WorkerState,
//! };
//!
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
//! dag.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 4, Resources::ZERO)),
//!     0.0,
//! );
//! let start = |job| Output::Start {
//!     job,
//!     attempt: 1,
//!     worker: 1,
//! };
//! let done = |job| Input::Done { job, attempt: 1 };
//!
//! // A job, then a pair of independent jobs, then a job: four leaves.
//! let pair = Arc::new(DagTemplate::new(2, []).unwrap());
//! let nodes = vec![
//!     TemplateNode::Job(1.0),
//!     TemplateNode::Unit(pair),
//!     TemplateNode::Job(1.0),
//! ];
//! let shape = DagTemplate::with_nodes(nodes, [(0, 1), (1, 2)]).unwrap();
//! assert_eq!((shape.len(), shape.leaves()), (3, 4));
//! let shape = Arc::new(shape);
//!
//! // Unit 10 is jobs 100..104; unit 20, jobs 200..204, runs after it.
//! let spec = JobSpec::new(0, Resources::ZERO, 0); // each leaf's, with the leaf's id
//! dag.declare(
//!     [
//!         Unit::new(10, 100, shape.clone(), spec.clone(), vec![]),
//!         Unit::new(20, 200, shape, spec, vec![10]),
//!     ],
//!     0.0,
//! )
//! .unwrap();
//! assert_eq!(dag.poll(0.0), [start(100)]);
//! assert_eq!(
//!     dag.explain(20).unwrap(),
//!     "unit 20 waits for 1 dependency [10]"
//! );
//!
//! dag.handle(done(100), 1.0);
//! assert_eq!(dag.poll(1.0), [start(101), start(102)]);
//! dag.handle(done(101), 2.0);
//! dag.handle(done(102), 2.0);
//! assert_eq!(dag.poll(2.0), [start(103)]);
//! dag.handle(done(103), 3.0);
//! assert_eq!(dag.poll(3.0), [start(200)]);
//! ```
//!
//! A unit costs a fixed amount of memory until its dependencies complete; only then is its per-node
//! state allocated, and it is freed when the unit completes ([`DagStats`] counts both). Per-leaf
//! data that differs between units of one template comes from a [`NodeSource`] rather than being
//! stored: a [`sourced`](Unit::sourced()) unit asks the scheduler's source for each leaf's work,
//! final spec and label, on demand.
//!
//! ```
//! # use std::sync::Arc;
//! # use whelm::{
//! #     Config, DagConfig, DagScheduler, DagTemplate, Input, JobSpec, Output, Policy, Resources,
//! #     Scheduler, Unit, WorkerState,
//! # };
//! use whelm::{JobId, MEM, NodeSource};
//!
//! /// Leaf `k` has work `10 (k + 1)` and needs `k + 1` GB.
//! struct Growing;
//!
//! impl NodeSource for Growing {
//!     fn work(&self, _unit: JobId, leaf: u32) -> f64 {
//!         10.0 * f64::from(leaf + 1)
//!     }
//!     fn spec(&self, _unit: JobId, leaf: u32, spec: &mut JobSpec) {
//!         spec.demand = Resources::mem_gb(f64::from(leaf + 1));
//!     }
//!     fn label(&self, unit: JobId, leaf: u32) -> Option<String> {
//!         Some(format!("unit {unit} step {leaf}"))
//!     }
//! }
//!
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()))
//!     .with_source(Arc::new(Growing));
//! dag.handle(Input::Worker(WorkerState::new(1, "cpu", 4, Resources::mem_gb(100.0))), 0.0);
//! let chain = Arc::new(DagTemplate::new(3, [(0, 1), (1, 2)]).unwrap());
//! let unit = Unit::new(10, 100, chain, JobSpec::new(0, Resources::ZERO, 0), vec![]).sourced();
//! dag.declare([unit], 0.0).unwrap();
//!
//! assert_eq!(dag.poll(0.0), [Output::Start { job: 100, attempt: 1, worker: 1 }]);
//! assert_eq!(dag.stats().workers[0].placed[MEM], 1_000_000_000);
//! assert_eq!(
//!     dag.explain(102).unwrap(),
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
//! # use whelm::{
//! #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources,
//! #     Scheduler, WorkerState
//! # };
//! use whelm::OrderTerm;
//!
//! /// The first job one single-slot worker starts: a lone job 1, or the head of chain 2 -> 3 -> 4.
//! fn first(config: Config) -> u64 {
//!     let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(config));
//!     dag.handle(
//!         Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)),
//!         0.0,
//!     );
//!     let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::ZERO, 0), deps);
//!     dag.declare(
//!         [
//!             job(1, vec![]),
//!             job(2, vec![]),
//!             job(3, vec![2]),
//!             job(4, vec![3]),
//!         ],
//!         0.0,
//!     )
//!     .unwrap();
//!     assert_eq!((dag.rank(1), dag.rank(2)), (Some(1.0), Some(3.0)));
//!     match dag.poll(0.0)[..] {
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
//! A unit can be declared with some leaves already complete ([`Unit::with_completed`], e.g. from a
//! checkpoint), and closed early ([`close`](DagScheduler::close)) when its remaining jobs are known
//! to be no-ops. Closing completes the unit at once: its unstarted jobs are dropped, and the jobs
//! already running are returned, keeping their resources until their attempts end.
//!
//! ```
//! # use std::sync::Arc;
//! # use whelm::{
//! #     Config, DagConfig, DagJob, DagScheduler, DagTemplate, Input, JobSpec, Output, Policy,
//! #     Resources, Scheduler, Unit, WorkerState
//! # };
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
//! dag.handle(Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)), 0.0);
//! let spec = |id| JobSpec::new(id, Resources::ZERO, 0);
//!
//! // Unit 10: four independent jobs 100..104, the first two done before a restart. Job 20 follows.
//! let four = Arc::new(DagTemplate::new(4, []).unwrap());
//! let unit = Unit::new(10, 100, four, spec(0), vec![]).with_completed(vec![0, 1]);
//! dag.declare([unit, DagJob::new(spec(20), vec![10]).into()], 0.0).unwrap();
//! assert_eq!(dag.poll(0.0), [Output::Start { job: 102, attempt: 1, worker: 1 }]);
//!
//! // Job 103 turns out to be unnecessary: close the unit. Job 102 is still running.
//! assert_eq!(dag.close(10, 1.0), Ok(vec![102]));
//! assert!(dag.poll(1.0).is_empty()); // job 20 is ready, but 102 holds the slot
//! dag.handle(Input::Done { job: 102, attempt: 1 }, 2.0);
//! assert_eq!(dag.poll(2.0), [Output::Start { job: 20, attempt: 1, worker: 1 }]);
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
//! # use whelm::{
//! #     Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources,
//! #     Scheduler, WorkerState
//! # };
//! let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
//! let worker = WorkerState::new(1, "cpu", 1, Resources::ZERO);
//! dag.handle(Input::Worker(worker.clone()), 0.0);
//! let job = |id, deps| DagJob::new(JobSpec::new(id, Resources::ZERO, 0), deps);
//! dag.declare([job(1, vec![]), job(2, vec![1])], 0.0).unwrap();
//! dag.poll(0.0);
//! dag.handle(Input::Done { job: 1, attempt: 1 }, 1.0);
//! assert_eq!(
//!     dag.poll(1.0),
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
//! let mut dag = DagScheduler::restore(snapshot, Scheduler::new(Config::default()), None, 5.0);
//! dag.handle(Input::Worker(worker), 5.0);
//! assert_eq!(
//!     dag.poll(5.0),
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
//! use whelm::{Config, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};
//!
//! let shared = SharedPolicy::with_system_clock(Scheduler::new(Config::default()));
//! shared.worker_update(WorkerState::new(1, "cpu", 1, Resources::ZERO));
//!
//! std::thread::scope(|s| {
//!     for id in 1..=3 {
//!         let shared = &shared;
//!         s.spawn(move || {
//!             let lease = shared.lease(JobSpec::new(id, Resources::ZERO, 0)); // blocks
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
//! #     Config, FailKind, JobSpec, Resources, RetryConfig, Scheduler, SharedPolicy, WorkerState
//! # };
//! let config = Config { retry: RetryConfig { max_attempts: 2 }, ..Config::default() };
//! let shared = SharedPolicy::with_system_clock(Scheduler::new(config));
//! for id in [1, 2] {
//!     shared.worker_update(WorkerState::new(id, "cpu", 1, Resources::ZERO));
//! }
//!
//! let lease = shared.lease(JobSpec::new(1, Resources::ZERO, 0));
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
//! use std::sync::{Arc, Mutex};
//!
//! use whelm::{
//!     Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState,
//!     log::{self, Event, Logged},
//! };
//!
//! // An in-memory sink; `log::JsonlSink` (feature `log`) writes compressed JSON lines instead.
//! let events = Arc::new(Mutex::new(Vec::<Event>::new()));
//! let mut p = Logged::new(Scheduler::new(Config::default()), events.clone());
//! p.handle(
//!     Input::Worker(WorkerState::new(1, "cpu", 1, Resources::ZERO)),
//!     0.0,
//! );
//! p.handle(Input::Submit(JobSpec::new(1, Resources::ZERO, 0)), 0.0);
//! p.handle(Input::Submit(JobSpec::new(2, Resources::ZERO, 0)), 0.0);
//! p.poll(0.0);
//! p.handle(Input::Done { job: 1, attempt: 1 }, 5.0);
//! p.poll(5.0);
//!
//! let events = events.lock().unwrap().clone();
//! let polls = log::polls(&events);
//! assert_eq!(
//!     polls[1],
//!     (
//!         5.0,
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
//! applies each instant's inputs and then polls covers most needs, and works for any [`Policy`],
//! boxed or not.
//!
//! ```
//! use whelm::{
//!     Config, DagConfig, DagScheduler, FailKind, Input, Instant, JobSpec, Output, Policy,
//!     Resources, Scheduler, WorkerState,
//! };
//!
//! /// Apply each instant's inputs, then poll; every poll's outputs, with its time.
//! fn run(p: &mut dyn Policy, script: Vec<(Instant, Vec<Input>)>) -> Vec<(Instant, Vec<Output>)> {
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
//!     let job = |id| Input::Submit(JobSpec::new(id, Resources::mem_gb(4.0), 0));
//!     let fail = Input::Failed {
//!         job: 1,
//!         attempt: 1,
//!         kind: FailKind::Other,
//!         why: "test".into(),
//!     };
//!     vec![
//!         (
//!             0.0,
//!             vec![Input::Worker(WorkerState::new(
//!                 1,
//!                 "cpu",
//!                 2,
//!                 Resources::mem_gb(8.0),
//!             ))],
//!         ),
//!         (1.0, vec![job(1), job(2), job(3)]),
//!         (2.0, vec![fail]),
//!         (3.0, vec![Input::Done { job: 2, attempt: 1 }]),
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
//! assert_eq!(polls[1], (1.0, vec![start(1, 1), start(2, 1)]));
//! // The retry keeps job 1's place ahead of job 3.
//! assert_eq!(polls[2], (2.0, vec![start(1, 2)]));
//! assert_eq!(polls[3], (3.0, vec![start(3, 1)]));
//!
//! // Inspect the end state.
//! let stats = flat.stats();
//! assert_eq!((stats.running, stats.placements_total), (2, 4));
//! assert_eq!(stats.workers[0].headroom[whelm::MEM], Some(0));
//! assert!(
//!     flat.explain(1)
//!         .unwrap()
//!         .starts_with("job 1 is running: attempt 2")
//! );
//! ```
//!
//! [`Policy::explain`] and [`Policy::stats`] are the windows into a running policy: the first says,
//! in words, why a job is not running (fit, constraints, holds, past failures), and the second
//! counts jobs, reservations, placements and every worker's load. Both are cheap enough to log.
//!
//! # Where to look next
//!
//! Each module's page tells the full story of its part; the items it defines are also exported at
//! the crate root.
//!
//! - [`scheduler`]: [`Scheduler`], the placement policy, and how a poll scans and places.
//! - [`config`]: [`Config`], its presets, and every order and score term, retry, speed and
//!   reservation setting.
//! - [`admission`]: the [`Admission`] contract, [`ProductionAdmission`] and [`WorkerView`].
//! - [`speed`]: machine models ([`Timing`]) and speed learning ([`Learn`], [`SpeedEstimator`], the
//!   last usable on its own).
//! - [`dag`]: [`DagScheduler`], templates, units and [`NodeSource`].
//! - [`shared`]: [`SharedPolicy`], the blocking front end for a thread per task.
//! - [`log`]: event logs, sinks and replay.
//! - [`nassau`]: helpers for driving a Nassau resolution.
//! - This page: the message types ([`Input`], [`Output`], [`Policy`]) and the job and worker
//!   descriptions ([`JobSpec`], [`WorkerState`], [`Resources`]).
//!
//! The repository's README gives the scheduling problem in α|β|γ notation and the plan for
//! integrating with Nassau's coordinator. The sibling crate `whelm-sim` replays logged traces
//! against this crate's policies, holds the simulators that model whole runs, and records the
//! measurements behind the defaults in its RESULTS.md.
#![warn(missing_docs)]

pub mod admission;
pub mod config;
pub mod dag;
pub mod job;
pub mod log;
pub mod message;
pub mod nassau;
pub mod resources;
pub mod scheduler;
pub mod shared;
pub mod speed;
pub mod stats;
pub mod worker;

pub use admission::{Admission, ProductionAdmission, WorkerView};
pub use config::{
    Config, DEFAULT_AGE_LIMIT, Defer, GroupOrder, OrderTerm, Reservations, RetryConfig, ScoreTerm,
    Speculate, SpeedConfig,
};
#[cfg(feature = "serde")]
pub use dag::DagSnapshot;
pub use dag::{
    DagConfig, DagError, DagJob, DagScheduler, DagStats, DagTemplate, NodeSource, TemplateNode,
    Unit,
};
#[doc(inline)]
pub use job::{Constraint, JobId, JobSpec, Selector, Strength};
pub use log::EventSink;
#[doc(inline)]
pub use message::{Attempt, FailKind, GaveUp, Input, Instant, Output, Policy, Tried};
#[doc(inline)]
pub use resources::{DEV, DIMS, HARD, MEM, Resources, SLOTS};
pub use scheduler::Scheduler;
pub use shared::{Lease, SharedPolicy};
pub use speed::{Learn, Sharing, SpeedEstimator, Timing};
#[doc(inline)]
pub use stats::{PolicyStats, ReservationInfo, WorkerLoad};
#[doc(inline)]
pub use worker::{WorkerId, WorkerState};

/// The README's examples, compiled and run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
