# sched

A pure, deterministic, resource-aware job-placement library. Given a stream of jobs that each
declare a resource demand and a changing pool of workers that each have a capacity and a number of
execution slots, it decides **which job runs on which worker, and when**, and what to do when an
attempt fails or a worker leaves. The core is sans-IO and message-driven: the caller feeds it
[`Input`]s, each with the caller's "now", and acts on the [`Output`]s it returns. It has no
networking, threads, clocks or persistence, and the same inputs produce the same outputs. Around it:
a dependency layer ([`DagScheduler`]), an event log that replays ([`log`]), and a blocking front end
for callers with a thread per task ([`SharedPolicy`]).

## The problem

In α|β|γ terms: Q (uniform machines, with speeds and classes) that join and leave online; jobs with
vector resource demands, eligibility constraints (required classes, forbidden and avoided
workers), precedence constraints
(the DAG layer) and work estimates rather than known processing times; objective makespan, with
bounded per-group latency (no starvation).

The policy is list scheduling: at every poll, waiting jobs are taken in a configured order and
each goes to the best-scored worker that admits it. With total work `W`, total throughput `P` and critical path `D`,
every schedule needs at least `max(W/P, D)`, and every *greedy* one -- never idle while a job is
ready and admissible -- needs at most `W/P + D` (Graham; Brent). Placement is greedy within
admission, apart from two deliberate holds (below) that keep a worker from a job to bound
starvation or to wait for a faster worker. The order is a list of terms: explicit priority, upward
rank (critical path below a job, as in HEFT), group, weighted shortest processing time, earliest
due date; reservations with backfill are EASY-style backfilling; aging bounds starvation under any
order.

## Idempotent jobs

Jobs are idempotent: running one twice is harmless and the first completion wins. This is the
caller's side of the [`Policy`] contract, and what lets the policy retry failed attempts, run a
speculative second attempt ([`Speculate`]) and ignore late messages about attempts it no longer
tracks. Every start carries an [`Attempt`] number (1 for the first); [`Input::Done`] and
[`Input::Failed`] name the attempt they report, and are ignored unless it is live. The one
exception is the DAG layer's local jobs ([`DagJob::local`]), which the caller runs exactly once.

## Event loop

```rust
use sched::{Config, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState};

let mut policy = Scheduler::new(Config::default());

// A worker joins (and later heartbeats): 16 slots, 120 GB, 20 GB used by its runtime.
let mut w = WorkerState::new(1, "l40s", 16, Resources::mem_gb(120.0));
w.reported_used = Resources::mem_gb(20.0);
w.reported_baseline = Resources::mem_gb(20.0);
policy.handle(Input::Worker(w), 0.0);

// Jobs become ready; `group` orders them (oldest group first), then FIFO.
policy.handle(Input::Submit(JobSpec::new(7, Resources::mem_gb(6.0), /* group */ 3)), 1.0);
policy.handle(Input::Submit(JobSpec::new(8, Resources::mem_gb(30.0), 3)), 1.0);

// After every batch of inputs, poll and act on each output.
let mut started = Vec::new();
for out in policy.poll(1.0) {
    match out {
        Output::Start { job, attempt, worker } => started.push((job, attempt, worker)),
        Output::Stop { .. } => {}   // cancel that attempt: its result is not wanted
        Output::GaveUp(_) => {}     // the job failed `max_attempts` times and is forgotten
        _ => unreachable!("DAG layer only"),
    }
}
assert_eq!(started, [(7, 1, 1), (8, 1, 1)]);

// Report each attempt's end; it frees its slot and demand.
policy.handle(Input::Done { job: 7, attempt: 1 }, 95.0);
assert!(policy.poll(95.0).is_empty());
println!("{:?}", policy.explain(8)); // why a job is (not) running, for logs
```

[`Policy::handle`] applies an input at once; outputs it causes (stops, give-ups) come out of the
next [`Policy::poll`], which also places what can be placed. [`Policy::next_wakeup`] is when the
policy next needs a poll without any input: a deferral lapses, a job ages, or a job has waited long
enough to reserve. A poll typically takes microseconds.

## Model

- A **job** ([`JobSpec`]) has a demand ([`Resources`]: a vector over [`DIMS`] dimensions, [`MEM`]
  for host memory, [`DEV`] for device memory and [`SLOTS`], which the scheduler sets to one), a
  priority group, an optional explicit priority, rank, weight and due date, an optional work
  estimate ([`JobSpec::work`], seconds at speed 1), and [`Constraint`]s: a [`Selector`] (a worker
  or a class) with a [`Strength`]. Require and Forbid are hard; Avoid is soft (avoided workers are
  used while no other live worker the hard constraints allow exists); Prefer is a score term.
- A **worker** ([`WorkerState`]) has a class, a speed, a budget per dimension (slots included), a
  per-job floor ([`WorkerState::per_task`], e.g. the typical device launch request), and its last
  reported usage and baseline. The scheduler keeps its own sum of the demands it placed on each
  worker; heartbeats only update the reported figures.

### Admission

An [`Admission`] rule decides whether a worker takes a job. [`ProductionAdmission`] applies one
inequality to every dimension:

```text
admit iff for every enforced dimension d:
            max(reported_used[d],
                reported_baseline[d] + max(placed[d], running * per_task[d]))
              + max(demand[d], per_task[d]) <= budget[d]
          or (d is soft and running == 0)                 // escape hatch
```

- [`HARD`] dimensions (slots) are always enforced and have no escape hatch. A zero budget
  component of a soft dimension (memory) is unknown and not enforced.
- `per_task` is a floor: each job counts for at least that much. With a device `per_task` alone the
  device inequality is a per-worker count, `(running + 1) * per_task <= budget`; with per-job device
  demands it is their sum. A floor should be near the mean job, not a high quantile: a sum of jobs
  concentrates near its mean.
- The escape hatch guarantees that every job can run somewhere: a job alone on a worker with a slot
  always goes.
- `reported_baseline` must exclude the running jobs' usage (the worker's resident floor minus their
  estimates); a floor that contains them counts them twice.

Where one number must rank workers (tightest fit, most headroom), it is the free fraction of
capacity in the bottleneck memory dimension, [`WorkerView::free_share`]. A custom rule goes in through
[`Scheduler::with_admission`]; it must be monotone in load (see [`Admission`]).

## The scheduler

[`Scheduler`] implements [`Policy`]; a plain [`Config`] decides its behaviour. Its
[`order`](Config::order) ([`OrderTerm`]s, then arrival) and [`score`](Config::score)
([`ScoreTerm`]s, then worker id) are lexicographic lists; the presets map objectives to them:

| preset | objective | order | score | reservations |
|---|---|---|---|---|
| [`Config::fifo`] | baseline | arrival | preferred, load | none (big jobs starve) |
| `Config::default()` | makespan, bounded latency | priority, rank, group; aging | speed, preferred, load | yes |
| [`Config::best_fit`] | as above, packing | as above | speed, tightest, preferred, load | yes |
| [`Config::weighted_completion`] | Σ w·C | priority, WSPT; aging | as default | yes |
| [`Config::lateness`] | max lateness | priority, EDD; aging | as default | yes |

WSPT and EDD are optimal on one machine only; here, as every rule, they are heuristics.

**Order and backfill.** Each poll scans waiting jobs in urgency order, so a job takes a worker
only if every more urgent waiting job was refused there. Keys are computed once, at submission.
Groups are ordered by first arrival ([`GroupOrder::Arrival`]) or by id ([`GroupOrder::Id`]), which
survives a restart that resubmits in another order ([`nassau::group`] gives Nassau's bidegrees such
ids). [`Scheduler::forget_group`] bounds the memory of group arrivals.

**Aging** ([`Config::age_limit`], default [`DEFAULT_AGE_LIMIT`], `None` for strict priority) puts
jobs that have waited that long ahead of everything else, oldest first.

**Reservations** ([`Reservations`]): the most urgent job that has waited at least `reserve_after`
and is admitted nowhere reserves the worker with the most headroom; nothing else is admitted there
until it is placed (at the latest when the worker empties). The most urgent waiting job is
therefore placed within `reserve_after` plus the longest running time of the jobs on the worker it
reserves. With `shadow_backfill`, a reserved worker still takes jobs expected to finish before the
holder could start (EASY backfilling), without weakening that bound.

**Speed** ([`SpeedConfig`]). A job's expected run time on a worker is its work over the worker's
speed.

- [`ScoreTerm::Speed`]: among the workers that admit a job, the fastest.
- [`Defer`]: earliest finish, HEFT's processor choice, online. A job may wait for a busy worker
  faster than the one the score picked when it would still finish earlier there (by at least
  `min_gain` of its work, for at most `max_wait`).
- [`Learn`]: learn speeds from completion times, per worker with its class as prior, corrected for
  concurrency, with hysteresis; speed-ordered placement treats speeds within one `resolution` step
  as equal, so load still balances a class. [`SpeedEstimator`] is the same estimator on its own.
- [`Speculate`]: a worker left with a free slot after the scan starts a second attempt of a job
  running on a slower worker, if that attempt is expected to finish at least `min_gain` of its run
  time earlier. Both run; the first to finish wins and the other gets an [`Output::Stop`].

**Holds.** A reservation and a deferral are the same thing to the scheduler: a worker kept from a
job that it might admit, by a hold that lapses when its holder is placed (or, for a deferral, at a
deadline). Holds are enforced in one place, reported by [`Policy::explain`], and their deadlines
give [`Policy::next_wakeup`]. Releasing a hold mid-scan restarts the scan, which keeps the priority
invariant.

**Retries and worker loss** ([`RetryConfig`]). A failed attempt, with no other attempt of the job
live, requeues the job with its original place and age, softly avoiding every worker it failed on
(the caller's Forbids stay hard). After [`RetryConfig::max_attempts`] rounds -- a
speculative attempt is an extra try within a round, not a round -- the policy emits [`Output::GaveUp`] with every
[`Tried`] attempt, `retryable` when all were [`FailKind::DeviceOom`]. [`Input::WorkerGone`] fails
each live attempt on the worker with [`FailKind::LinkDied`]; the caller never resubmits.
[`Input::Cancel`] drops a waiting job or stops a running one's attempts.

## Thread-per-task callers: `SharedPolicy`

[`SharedPolicy`] wraps a flat policy for a caller that runs each task on its own thread.
[`lease`](SharedPolicy::lease) submits a job and blocks until it starts (a wake handle per job, no
polling); the [`Lease`] names the worker and attempt, and ends with [`Lease::complete`] or
[`Lease::fail`], which blocks for the retry's start or returns the [`GaveUp`]. Dropping a lease
cancels its job; [`lease_timeout`](SharedPolicy::lease_timeout) withdraws the job after a timeout.
It polls after every call, and [`spawn_ticker`](SharedPolicy::spawn_ticker) polls as time passes.

A thread runs one attempt at a time, so speculation should be off here: a speculative start for a
leased job is answered with [`FailKind::Rejected`]. When a worker leaves, the thread's later
`fail(LinkDied)` still receives the job's retry or give-up, and its `complete` cancels the retry.

```rust
use std::sync::Arc;
use sched::{Config, FailKind, JobSpec, Resources, Scheduler, SharedPolicy, WorkerState};

let shared = Arc::new(SharedPolicy::with_system_clock(Scheduler::new(Config::default())));
shared.worker_update(WorkerState::new(1, "l40s", 16, Resources::mem_gb(120.0)));
let mut lease = shared.lease(JobSpec::new(42, Resources::mem_gb(6.0), 3)); // blocks
loop {
    // Send the task to `lease.worker()` and wait for the reply.
    let reply: Result<(), (FailKind, String)> = Ok(());
    match reply {
        Ok(()) => break lease.complete(),
        Err((kind, why)) => match lease.fail(kind, &why) {
            Ok(retry) => lease = retry, // another worker, softly avoiding the ones tried
            Err(gave_up) => break eprintln!("job 42 failed: {:?}", gave_up.tried),
        },
    }
}
```

## Event log and replay

[`log::Logged`] wraps a policy and records every input it handles (submissions with an optional
[`log::TaskInfo`]) and every poll's outputs to an [`EventSink`], with rate-limited heartbeat
samples for the trace reader. [`log::replay`] feeds a log back into a fresh policy
with the same configuration and reproduces every poll:

```rust
use std::sync::{Arc, Mutex};
use sched::{
    Config, Input, JobSpec, Policy, Resources, Scheduler, WorkerState,
    log::{self, Event, Logged},
};

let events = Arc::new(Mutex::new(Vec::<Event>::new()));
let mut p = Logged::new(Scheduler::new(Config::default()), events.clone());
p.handle(Input::Worker(WorkerState::new(1, "x", 2, Resources::mem_gb(10.0))), 0.0);
p.handle(Input::Submit(JobSpec::new(1, Resources::mem_gb(4.0), 0)), 0.0);
p.poll(0.0);

let events = events.lock().unwrap().clone();
let mut fresh = Scheduler::new(Config::default());
assert_eq!(log::replay(&mut fresh, events.clone()), log::polls(&events));
```

`log::JsonlSink` (feature `log`) writes the events as gzip-compressed JSON lines, the trace format
of the simulator.

## Dependencies: the DAG layer

[`DagScheduler`] wraps any policy and is itself a [`Policy`]. The graph is a coarse DAG of
[`Unit`]s, each an instance of a [`DagTemplate`] whose nodes are jobs or, recursively, units of
other templates; a plain job ([`DagJob`]) is a unit of a one-node template. Units are declared with
their dependencies, possibly long before they are ready and possibly naming units not declared yet;
a job is submitted to the inner policy when its last dependency completes, and an [`Input::Done`]
of a live attempt completes it here. Cycles are rejected at declaration (the batch leaves no
trace). A unit's per-node state is materialised when its dependencies complete and freed when it
completes, so memory tracks the live frontier while ranks see the whole declared graph.

- **Ranks.** With [`DagConfig::rank_priority`], jobs are submitted with their upward rank
  ([`JobSpec::rank`]: their work plus the longest chain of work below them, through the enclosing
  units and their dependents), which [`OrderTerm::Rank`] orders by.
  [`DagScheduler::update_work`] rescales a unit later.
- **Units.** A [`DagTemplate`] is checked once and shared by every unit built on it. Per-leaf work,
  demands and labels can come from a [`NodeSource`] instead of being stored. A unit can be declared
  with leaves already complete (resuming from a checkpoint) and closed early
  ([`DagScheduler::close`]: unstarted jobs complete as no-ops, running ones only free their
  resources when they end).
- **Outputs.** A ready local job is announced by [`Output::RunLocal`] and reported with
  [`Input::Done`] and attempt 0. Without [`DagConfig::auto_submit`], ready jobs are announced by
  [`Output::Ready`] and submitted by [`DagScheduler::release`]. **Passthrough** jobs
  ([`DagJob::passthrough`]) are synchronisation points ("group G is done") that complete by
  themselves, announced by [`Output::Passed`] with [`DagConfig::record_passthrough`].
- **Give-ups and cancellation.** A job the inner policy gives up on is held again: its dependents
  stay pending until the caller releases it (another round of attempts) or cancels it.
  [`DagScheduler::cancel`] and [`Input::Cancel`] cascade to every dependent.
- **Snapshots** (feature `serde`): `DagScheduler::snapshot` and `DagScheduler::restore` save the
  coarse graph and the materialised units; submitted jobs are submitted again on restore.

```rust
use sched::{
    Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources, Scheduler,
    WorkerState,
};

let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
dag.handle(Input::Worker(WorkerState::new(1, "x", 4, Resources::ZERO)), 0.0);
let job = |id| JobSpec::new(id, Resources::ZERO, 0);
dag.declare(vec![DagJob::new(job(1), vec![]).local(), DagJob::new(job(2), vec![1])], 0.0)
    .unwrap();
assert_eq!(dag.poll(0.0), [Output::RunLocal { job: 1 }]);
dag.handle(Input::Done { job: 1, attempt: 0 }, 1.0);
assert_eq!(dag.poll(1.0), [Output::Start { job: 2, attempt: 1, worker: 1 }]);
```

To log a DAG-driven run, log the inner policy: `DagScheduler<Logged<Scheduler>>`.

## Features

- `serde` (default): serialisation of the message types and DAG snapshots.
- `log`: the JSONL event-log writer `log::JsonlSink` (adds `serde_json`, `flate2`).

Without features the crate has no dependencies.

## Simulator

The trace simulator, the whole-run Nassau model, their results and the literature notes live in
the sibling crate `ext/crates/sched-sim`, which replays `log::JsonlSink` traces against this
crate's policies.

## Integration notes (Nassau's coordinator)

Phase 1 keeps the thread per task and replaces the inside of `acquire`/`release` with a lease loop:

```rust,no_run
# #[cfg(feature = "log")]
# fn main() {
use std::{sync::Arc, time::Duration};
use sched::{
    Config, FailKind, GroupOrder, JobSpec, Resources, Scheduler, SharedPolicy, SpeedConfig, Timing,
    WorkerState,
    log::{JsonlSink, Logged, TaskInfo},
    nassau,
};

// Once: restart-stable bidegree order, aging at `DEFAULT_AGE_LIMIT`, fast workers first with
// speeds learned per worker, every input and poll logged.
let policy = Scheduler::new(Config {
    group_order: GroupOrder::Id,
    speed: SpeedConfig {
        timing: Timing::learned(),
        ..SpeedConfig::default()
    },
    ..Config::default()
});
let log = JsonlSink::create("sched_events.jsonl.gz".as_ref()).unwrap();
let shared = Arc::new(SharedPolicy::with_system_clock(Logged::new(policy, log)));
let _ticker = shared.spawn_ticker(Duration::from_secs(1));

// Every MemReport: rss as reported_used, baseline_excl (the rolling floor minus the estimates
// running) as reported_baseline, the class prior as speed (learning corrects it), the device
// launch pool as the device budget and the learned per-task device demand (0 = unknown).
let (id, class, rss, baseline_excl, dev_cap, dev_demand) = (7, "l40s", 40.0, 12.0, 19.5, 2.4);
shared.worker_update(WorkerState {
    reported_used: Resources::mem_gb(rss),
    reported_baseline: Resources::mem_gb(baseline_excl),
    speed: if class == "l40s" { 1.39 } else { 1.0 },
    per_task: Resources::ZERO.with_dev_gb(dev_demand),
    ..WorkerState::new(id, class, 16, Resources::mem_gb(123.7).with_dev_gb(dev_cap))
});

// acquire(res, key, est, b, what, avoid): one task, its bidegree (s, t), its estimate in GB and
// expected seconds on an H200 (learning needs a work estimate; any size proxy proportional to it
// works). A per-task device estimate, when known, goes in `spec.demand[DEV]` (`with_dev_gb`).
let (task, s, t, est_gb, work) = (123_456, 3, 200, 9.5, 600.0);
shared.with(|p, _| {
    p.annotate(task, TaskInfo { kind: "sig".into(), bidegree: (t - s, s), ..TaskInfo::default() })
});
let mut spec = JobSpec::new(task, Resources::mem_gb(est_gb), nassau::group(s as u32, t as u32));
spec.work = Some(work);
let mut lease = shared.lease(spec); // blocks; no polling
loop {
    let _worker = lease.worker(); // send over TCP, block on the reply
    let reply: Result<(), (FailKind, String)> = Ok(());
    match reply {
        Ok(()) => break lease.complete(), // release
        Err((kind, why)) => match lease.fail(kind, &why) {
            Ok(retry) => lease = retry, // avoids the workers tried, softly
            Err(gave_up) => {
                // `retryable`: every attempt was DeviceOom, so retry at the bidegree level.
                let _ = gave_up.retryable;
                break;
            }
        },
    }
}

// A worker left: its attempts are retried at once; each thread's `fail(LinkDied)` picks up the
// retry, and a reply already in hand completes the job.
let _lost = shared.worker_gone(id);
# }
# #[cfg(not(feature = "log"))]
# fn main() {}
```

Phase 2 drives the coordinator from a [`DagScheduler`] over a `Logged<Scheduler>`, in a single
event loop: bidegrees' zero steps, registrations and commits as local jobs ([`Output::RunLocal`]),
each walk an instance of its profile's signature template with per-node demands, opened with its
checkpointed nodes complete and closed early at the dead tail; `snapshot`/`restore` across
coordinator restarts. Worker loss needs no resubmission: the core
retries the lost attempts.

## License

MIT OR Apache-2.0.
