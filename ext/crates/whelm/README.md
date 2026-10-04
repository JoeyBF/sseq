# whelm

Pure, deterministic placement of jobs onto a changing fleet of workers: which job runs on which
worker, and when, so as to neither overwhelm nor underwhelm the machines. The core is sans-IO and
message-driven: the caller feeds it `Input`s, each stamped with the caller's "now" (a `Time`: the
`std::time::Duration` since an origin on the caller's clock; spans are plain `Duration`s), and acts
on the `Output`s it returns. It has no networking, threads, clocks or persistence, and the same
inputs produce the same outputs. Around the core sit a dependency layer (`DagScheduler`), a
replayable event log (`log`) and a blocking front end for callers with a thread per task
(`SharedPolicy`).

The crate documentation (`cargo doc --open`) is a guided tour of the API, chapter by chapter, with
an example of every behaviour; this page is the overview.

## The problem

In α|β|γ notation:

- **α, machines.** Workers (`WorkerState`) join, heartbeat and leave at any time. Each has a class,
  a capacity of each declared resource, and a speed. The timing model
  (`Timing`) is identical machines (P, `Timing::Identical`), uniformly related ones (Q,
  `Timing::Related`: one speed per worker) or unrelated ones (R, `Timing::Unrelated`: a speed per
  job kind and worker class). Speeds are reported, or learned online from completion times (`Learn`;
  `SpeedEstimator` is the same learner on its own).
- **β, jobs.** A job (`JobSpec`) has
  - a demand (`Resources`: amounts keyed by the names of the resources that `Config::resources`
    declares, by default `MEMORY`, `DEVICE_MEMORY` and `SLOTS`; a job naming another is rejected),
    checked by one per-resource admission rule (below);
  - `Constraint`s on a worker or a class (`Selector`) with a `Strength`: Require and Forbid are
    hard, Avoid is soft, Prefer only ranks workers;
  - dependencies, through the DAG layer;
  - estimates: its `work` (run time) at speed 1, and a `kind` for unrelated machines;
  - a `weight`, a `due` date, an explicit `priority` and a `group`.

  Jobs arrive online. There is no preemption or migration: a job runs to completion, or another
  attempt of it restarts from scratch (retries, speculation).
- **γ, objective.** Chosen by the configuration's `order` and `score` term lists, with presets for
  makespan with bounded latency, weighted completion time and maximum lateness (see [the
  scheduler](#the-scheduler)).

The algorithm is greedy list scheduling: at every poll, waiting jobs are taken in the configured
order and each goes to the best-scored worker that admits it. On identical machines with slots as
the only binding resource and no holds, that is a list schedule in Graham's sense, so its makespan
is within `2 - 1/m` of optimal, `m` the total number of slots, precedence constraints included. With
memory demands, speeds, holds or arrivals over time every rule here is a heuristic.

### Admission

An `Admission` rule decides whether a worker takes a job. `ProductionAdmission` applies one
inequality to every enforced resource:

```text
admit iff for every enforced resource d:
            max(reported_used[d],
                reported_baseline[d] + max(placed[d], running * per_task[d]))
              + max(demand[d], per_task[d]) <= capacity[d]
          or (d is soft and running == 0)                 // escape hatch
```

- The configuration declares each resource (`Resource`, identified by its name) hard or soft, with
  a default demand for jobs that leave it out. Hard resources (slots, a GPU count, a license pool) are always enforced and
  have no escape hatch; a worker with zero capacity of one runs no job that demands it. A soft
  resource (memory) is enforced only where its capacity is nonzero: a zero capacity is unknown.
- `per_task` is a per-job floor, so with a device floor and no per-job device demands the device
  inequality counts jobs, `(running + 1) * per_task <= capacity`.
- The escape hatch lets a job alone on a worker run whatever its soft estimates, so every job that
  fits the hard resources of some worker can run somewhere.
- `reported_baseline` must exclude the running jobs' usage; a floor that contains them counts them
  twice.

Where one number must compare workers (tightest fit, most headroom) it is the free fraction of
capacity in the bottleneck soft resource, `WorkerView::free_share`. A custom rule goes in through
`Scheduler::with_admission` and must be monotone in load (see `Admission`).

## Jobs, attempts and messages

Jobs are idempotent: running one twice is harmless and the first completion wins. That is the
caller's side of the `Policy` contract, and it lets the policy retry failed attempts, run
speculative second attempts and ignore late messages. Every start carries an `Attempt` number (1 for
the first); `Input::Done` and `Input::Failed` name the attempt they report and are ignored unless it
is live. The DAG layer's local jobs (`DagJob::local`) are the exception: the caller runs those
exactly once.

```rust
use std::time::Duration;

use whelm::{
    Config, Input, JobSpec, MEMORY, Output, Policy, Resources, SLOTS, Scheduler, Time, WorkerState,
    gb,
};

let mut policy = Scheduler::new(Config::default());
// Points on the caller's clock: here, seconds since the start of the run.
let at = |secs| Time::ORIGIN + Duration::from_secs(secs);

// A worker joins (and later heartbeats): 16 slots, 120 GB, 20 GB used by its runtime.
policy.handle(
    Input::Worker(WorkerState {
        id: 1,
        class: "l40s".into(),
        capacity: Resources::new().with(MEMORY, gb(120.0)).with(SLOTS, 16),
        reported_used: Resources::new().with(MEMORY, gb(20.0)),
        reported_baseline: Resources::new().with(MEMORY, gb(20.0)),
        ..Default::default()
    }),
    Time::ORIGIN,
);

// Jobs and workers are struct literals; `Default` fills in what they leave out.
let job = |id, size| JobSpec { id, demand: Resources::new().with(MEMORY, gb(size)), group: 3, ..Default::default() };
policy.handle(Input::Submit(job(7, 6.0)), at(1));
policy.handle(Input::Submit(job(8, 30.0)), at(1));

// After every batch of inputs, poll and act on each output.
let mut started = Vec::new();
for out in policy.poll(at(1)) {
    match out {
        Output::Start { job, attempt, worker } => started.push((job, attempt, worker)),
        Output::Stop { .. } => {}   // cancel that attempt: its result is not wanted
        Output::GaveUp(_) => {}     // the job failed too often and is forgotten
        Output::Rejected { .. } => {} // its demand names an undeclared resource
        _ => unreachable!("DAG layer only"),
    }
}
assert_eq!(started, [(7, 1, 1), (8, 1, 1)]);

// Report each attempt's end; it frees its slot and demand.
policy.handle(Input::Done { job: 7, attempt: 1 }, at(95));
assert!(policy.poll(at(95)).is_empty());
println!("{}", policy.explain(8).unwrap()); // why a job is (not) running, one line for logs
```

`Policy::handle` applies an input at once; the outputs it causes (stops, give-ups) come out of the
next `Policy::poll`, which also places what can be placed. `Policy::next_wakeup` is the next time a
poll is needed without any input.

## The scheduler

`Scheduler` implements `Policy`, configured by a plain `Config`. Its `order` (`OrderTerm`s, then
arrival) and `score` (`ScoreTerm`s, then worker id) are lexicographic lists, and the presets map
objectives to them:

| preset | objective | order | score | aging, reservations |
|---|---|---|---|---|
| `Config::default()` | makespan, bounded latency | priority, group | speed, preferred, load | yes |
| `Config::fifo` | baseline | arrival | preferred, load | no |
| `Config::best_fit` | packing | as default | speed, tightest, preferred, load | yes |
| `Config::weighted_completion` | Σ wC | priority, WSPT | as default | yes |
| `Config::lateness` | max lateness | priority, EDD | as default | yes |

WSPT (Smith's rule) and EDD (Jackson's rule) are optimal on one machine only. Keys are computed
once, at submission; a job lacking what a term reads (a rank, a work estimate, a due date) sorts
after the jobs that have it. `OrderTerm::Group` orders groups by first arrival
(`GroupOrder::Arrival`) or by id (`GroupOrder::Id`), which survives a restart that resubmits in
another order; `Scheduler::forget_group` bounds the memory of arrivals.

**Order and backfill.** Each poll scans waiting jobs in order, so a job takes a worker only if every
more urgent waiting job was refused there. That priority invariant follows from admission being
monotone in load, without an explicit check.

**Aging** (`Config::age_limit`, default `DEFAULT_AGE_LIMIT`; `None` is strict priority) puts jobs
that have waited that long ahead of everything else, oldest first.

**Reservations** (`Reservations`): the most urgent job that has waited `reserve_after` and is
admitted nowhere reserves the worker with the most headroom, which takes no other job until the
holder is placed (at the latest when the worker empties). The most urgent waiting job is therefore
placed within `reserve_after` plus the longest running time of the jobs on the worker it reserves.
With `shadow_backfill`, a reserved worker still takes jobs expected to finish before the holder
could start (EASY backfilling), without weakening that bound.

**Speed** (`SpeedConfig`). A job's expected run time on a worker is its work over its speed there,
as `SpeedConfig::timing` has it.

- `ScoreTerm::Speed` ranks the fastest admitting worker first. With learning, speeds within one
  `resolution` step tie, so load still balances a class.
- `Defer` (earliest finish, HEFT's processor choice, online): a job may wait for a busy worker
  faster than the one the score picked, when it would finish earlier there by at least `min_gain` of
  its work, for at most `max_wait`.
- `Speculate`: a worker left with a free slot after the scan starts a second attempt of a job
  running on a slower worker, if that attempt is expected to finish at least `min_gain` of its run
  time earlier. Both run; the first to finish wins and the other gets an `Output::Stop`.

**Holds.** A reservation and a deferral are one notion: a worker kept from a job it might admit,
lapsing when the holder is placed or, for a deferral, at a deadline. Holds are enforced in one place
and reported by `Policy::explain`; their deadlines, with the aging and reservation thresholds, give
`Policy::next_wakeup`. Releasing a hold mid-scan restarts the scan, which keeps the priority
invariant.

**Retries, worker loss, cancellation** (`RetryConfig`). A failed attempt with no other attempt of
the job live requeues the job with its original place and age, softly avoiding every worker it
failed on (the caller's Forbids stay hard). After `RetryConfig::max_attempts` rounds (a speculative
attempt is an extra try within a round) the policy emits `Output::GaveUp` with every `Tried`
attempt, `retryable` when all failed with `FailKind::DeviceOom`. `Input::WorkerGone` fails each live
attempt on the worker with `FailKind::LinkDied`; the caller never resubmits. `Input::Cancel` drops a
waiting job or stops a running one's attempts.

## Thread-per-task callers: `SharedPolicy`

`SharedPolicy` wraps a flat policy behind a lock. `lease` submits a job and blocks until it starts
(each start wakes only its job's thread); the `Lease` names the worker and attempt and ends with
`Lease::complete` or `Lease::fail`, which blocks for the retry's start or returns the `GaveUp`.
Dropping a lease cancels its job; `lease_timeout` withdraws it after a timeout. The front end polls
after every call, and `spawn_ticker` polls as time passes.

A thread runs one attempt at a time, so speculation should be off: a speculative start of a leased
job is answered with `FailKind::Rejected`. When a worker leaves, the thread's later `fail(LinkDied)`
still receives the job's retry or give-up, and its `complete` cancels the retry.

```rust
use std::sync::Arc;
use whelm::{
    Config, FailKind, JobSpec, MEMORY, Resources, SLOTS, Scheduler, SharedPolicy, WorkerState, gb,
};

let shared = Arc::new(SharedPolicy::with_system_clock(Scheduler::new(Config::default())));
shared.worker_update(WorkerState {
    id: 1,
    class: "l40s".into(),
    capacity: Resources::new().with(MEMORY, gb(120.0)).with(SLOTS, 16),
    ..Default::default()
});
let job = JobSpec { id: 42, demand: Resources::new().with(MEMORY, gb(6.0)), group: 3, ..Default::default() };
let mut lease = shared.lease(job); // blocks
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

`log::Logged` wraps a policy and records every input it handles (a submission with an optional
`log::TaskInfo`) and every poll's outputs to an `EventSink`, plus rate-limited heartbeat samples for
the trace reader. `log::replay` feeds a log into a fresh policy with the same configuration and
reproduces every poll; learned speeds live in the scheduler, so it relearns them.

```rust
use std::sync::{Arc, Mutex};
use whelm::{
    Config, Input, JobSpec, MEMORY, Policy, Resources, SLOTS, Scheduler, Time, WorkerState, gb,
    log::{self, Event, Logged},
};

let events = Arc::new(Mutex::new(Vec::<Event>::new()));
let mut p = Logged::new(Scheduler::new(Config::default()), events.clone());
let capacity = Resources::new().with(MEMORY, gb(10.0)).with(SLOTS, 2);
let worker = WorkerState { id: 1, capacity, ..Default::default() };
p.handle(Input::Worker(worker), Time::ORIGIN);
let job = JobSpec { id: 1, demand: Resources::new().with(MEMORY, gb(4.0)), ..Default::default() };
p.handle(Input::Submit(job), Time::ORIGIN);
p.poll(Time::ORIGIN);

let events = events.lock().unwrap().clone();
let mut fresh = Scheduler::new(Config::default());
assert_eq!(log::replay(&mut fresh, events.clone()), log::polls(&events));
```

`log::JsonlSink` (feature `log`) writes the events as gzip-compressed JSON lines, the simulator's
trace format.

## Dependencies: the DAG layer

`DagScheduler` wraps any policy and is itself a `Policy`. Its graph is a coarse DAG of `Unit`s. A
unit is an instance of a `DagTemplate`, a dependency structure checked once and shared by every unit
built on it, whose nodes (`TemplateNode`) are worker jobs, local jobs, passthroughs or, recursively,
units of other templates. A plain job (`DagJob`) is a unit of a one-node template. A unit's leaf `k`
is job `base + k`; other units depend on it by its id.

- **Declaration.** `declare` takes units with their dependencies, possibly long before they are
  ready and possibly naming units not declared yet. A batch that would close a cycle, redeclare a
  unit or overlap another unit's ids is rejected and leaves no trace.
- **Lazy materialisation.** A unit costs a fixed amount until its dependencies complete; then its
  per-node counters are allocated, and they are freed when it completes, so memory follows the live
  frontier. Materialisation is bookkeeping only and never changes the schedule: a job is submitted
  to the inner policy as soon as its last dependency completes.
- **Per-leaf data.** Leaf work, demands and labels come from the template and the unit's spec and
  scale or, for a `sourced` unit, from the scheduler's `NodeSource`, computed on demand rather than
  stored. The source can also make a leaf a no-op in one unit (`NodeSource::passthrough`), so units
  of one template differ in which leaves run.
- **Resume and close.** A unit can be declared with leaves already complete (`Unit::completed`,
  e.g. from a checkpoint) and closed early (`close`): unstarted jobs complete as no-ops, and running
  ones keep their resources until their attempt ends.
- **Ranks.** Upward ranks (a job's work plus the longest chain of work below it, through the
  enclosing units and their dependents) are exact within a unit and maintained between units to
  within `DagConfig::rank_epsilon`. Jobs are submitted with their `rank`, which orders them only
  where `Config::order` lists `OrderTerm::Rank`; `update_work` rescales a unit later.
- **Outputs.** A ready local job is announced by `Output::RunLocal` and reported with `Input::Done`
  and attempt 0. Without `DagConfig::auto_submit`, ready jobs are announced by `Output::Ready` and
  submitted by `release`. With `DagConfig::record_passthrough`, completed passthroughs and units
  other than plain jobs are announced by `Output::Passed`. `announcements` drains these alone, so
  the caller can react before anything is placed.
- **Give-ups and cancellation.** A job the inner policy gives up on is held again: its dependents
  wait until the caller releases it (another round of attempts) or cancels it. `cancel` and
  `Input::Cancel` cascade to every dependent.
- **Snapshots** (feature `serde`): `DagScheduler::snapshot` and `DagScheduler::restore` save the
  coarse graph and the materialised units; submitted jobs are submitted again on restore.

```rust
use std::{sync::Arc, time::Duration};

use whelm::{
    Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources, SLOTS,
    Scheduler, TemplateSpec, Time, Unit, WorkerState,
};

let mut dag = DagScheduler::new(DagConfig::default(), Scheduler::new(Config::default()));
let worker = WorkerState { id: 1, capacity: Resources::new().with(SLOTS, 4), ..Default::default() };
dag.handle(Input::Worker(worker), Time::ORIGIN);

// A local job 1, then unit 10: a chain of jobs 100 -> 101 -> 102, the first checkpointed.
let spec = JobSpec { id: 1, ..Default::default() };
let load = DagJob { spec, local: true, ..Default::default() };
let chain = TemplateSpec { edges: vec![(0, 1), (1, 2)], ..TemplateSpec::jobs(3) };
let unit = Unit {
    id: 10,
    base: 100,
    template: Arc::new(chain.build().unwrap()),
    deps: vec![1],
    completed: vec![0],
    ..Default::default()
};
dag.declare([load.into(), unit], Time::ORIGIN).unwrap();
assert_eq!(dag.poll(Time::ORIGIN), [Output::RunLocal { job: 1 }]);
dag.handle(Input::Done { job: 1, attempt: 0 }, Time(Duration::from_secs(1)));
assert_eq!(dag.poll(Time(Duration::from_secs(1))), [Output::Start { job: 101, attempt: 1, worker: 1 }]);
```

To log a DAG-driven run, log the inner policy: `DagScheduler<Logged<Scheduler>>`.

## Features

- `serde` (default): serialisation of the message types and DAG snapshots.
- `log`: the JSONL event-log writer `log::JsonlSink` (adds `serde_json` and `flate2`).

Without features the crate has no dependencies.

## Simulator

The sibling crate `ext/crates/whelm-sim` holds the simulators, their results and the literature
notes: `whelm-sim` replays `log::JsonlSink` traces against this crate's policies, and `whelm-whole`,
`whelm-device` and `whelm-pisa` model a whole Nassau run, device memory and synthetic instances.

## Integrating with Nassau's coordinator

**Phase 1** keeps the thread per task and replaces the inside of `acquire`/`release` with a lease
loop over a logged `SharedPolicy`. Each heartbeat maps onto a `WorkerState`: RSS as `reported_used`,
`baseline_excl` (the rolling RSS floor minus the estimates running) as `reported_baseline`, the
class prior as `speed` (learning corrects it), the device launch pool as the device capacity and the
learned per-task device demand as the device `per_task`. Bidegrees become groups through
`nassau::group` under `GroupOrder::Id`.

```rust,no_run
# #[cfg(feature = "log")]
# fn main() {
use std::{sync::Arc, time::Duration};
use whelm::{
    Config, DEVICE_MEMORY, FailKind, GroupOrder, JobSpec, MEMORY, Resources, SLOTS, Scheduler,
    SharedPolicy, SpeedConfig, Timing, WorkerState, gb, log::{JsonlSink, Logged, TaskInfo}, nassau,
};

// Once: restart-stable bidegree order, speeds learned per worker, every input and poll logged.
let policy = Scheduler::new(Config {
    group_order: GroupOrder::Id,
    speed: SpeedConfig {
        timing: Timing::learned(),
        ..SpeedConfig::default()
    },
    ..Config::default()
});
let log = JsonlSink::create("whelm_events.jsonl.gz".as_ref()).unwrap();
let shared = Arc::new(SharedPolicy::with_system_clock(Logged::new(policy, log)));
let _ticker = shared.spawn_ticker(Duration::from_secs(1));

// Every heartbeat (MemReport).
let (id, class, rss, baseline_excl, dev_cap, dev_per_task) = (7, "l40s", 40.0, 12.0, 19.5, 2.4);
let prior = if class == "l40s" { 1.39 } else { 1.0 };
shared.worker_update(WorkerState {
    id,
    class: class.into(),
    capacity: Resources::new()
        .with(MEMORY, gb(123.7))
        .with(DEVICE_MEMORY, gb(dev_cap))
        .with(SLOTS, 16),
    per_task: Resources::new().with(DEVICE_MEMORY, gb(dev_per_task)),
    reported_used: Resources::new().with(MEMORY, gb(rss)),
    reported_baseline: Resources::new().with(MEMORY, gb(baseline_excl)),
    speed: prior,
});

// acquire: one task of bidegree (s, t), its memory estimate and its expected run time at speed 1
// (any size proxy proportional to the run time works for learning).
let (task, s, t, est_gb, work) = (123_456, 3, 200, 9.5, Duration::from_secs(600));
shared.with(|p, _| {
    p.annotate(task, TaskInfo { kind: "sig".into(), bidegree: (t - s, s), ..TaskInfo::default() })
});
let spec = JobSpec {
    id: task,
    demand: Resources::new().with(MEMORY, gb(est_gb)),
    group: nassau::group(s as u32, t as u32),
    work: Some(work),
    ..Default::default()
};
let mut lease = shared.lease(spec);
loop {
    let _worker = lease.worker(); // send over TCP, block on the reply
    let reply: Result<(), (FailKind, String)> = Ok(());
    match reply {
        Ok(()) => break lease.complete(), // release
        Err((kind, why)) => match lease.fail(kind, &why) {
            Ok(retry) => lease = retry,
            Err(gave_up) => {
                // `retryable`: every attempt ran out of device memory; retry at the bidegree
                // level.
                let _ = gave_up.retryable;
                break;
            }
        },
    }
}

// A worker left: its attempts are retried at once, and each thread's `fail(LinkDied)` picks up
// the retry.
let _lost = shared.worker_gone(id);
# }
# #[cfg(not(feature = "log"))]
# fn main() {}
```

**Phase 2** drives the coordinator from one event loop over a `DagScheduler<Logged<Scheduler>>`.
Each bidegree is a `sourced` unit of its profile's signature template, its `NodeSource` supplying
per-node work and demands; it is declared with its checkpointed nodes complete
(`Unit::completed`) and `close`d at the dead tail. Zero steps, registrations and commits are
local jobs (`Output::RunLocal`). `snapshot`/`restore` carry the graph across coordinator restarts,
and worker loss needs no resubmission: the core retries the lost attempts.

## License

MIT OR Apache-2.0.
