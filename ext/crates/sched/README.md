# sched

A pure, deterministic, resource-aware job-placement library: given a stream of jobs that each
declare a resource demand, and a changing pool of workers that each have a capacity and a number
of execution slots, it decides **which waiting job goes to which worker, and when**. The core does
nothing else -- no networking, threads, clocks or persistence: every input is an event carrying the
caller's "now", and the same events produce the same placements. Around it: a blocking front end
for callers with a thread per task ([`SharedPolicy`]), an event log ([`log`]), and a dependency
layer ([`DagScheduler`]).

## The problem, in general terms

Online scheduling of a weighted DAG on heterogeneous machines, with resource constraints:

- **Graph.** Jobs form a directed acyclic graph, possibly declared long before it is ready, and
  possibly built by *substitution*: a coarse DAG whose nodes expand into copies of shared
  sub-DAGs ([`DagTemplate`]), expanded lazily. Zero-weight join nodes ([`DagJob::passthrough`])
  mark "group done".
- **Weights.** Each job has work that is unknown until it runs, with an estimate to rank by; a
  resource demand held while it runs.
- **Machines.** Workers have slots, a capacity, and a class (speed); they join and leave.
- **Policy.** At every event, choose which ready jobs start where, subject to admission, so as to
  finish soon without starving anyone.

Reference points: with total work `W`, total throughput `P` and critical path `D`, every
schedule needs at least `max(W/P, D)`, and every *greedy* one -- never idle while a job is ready
and admissible -- needs at most `W/P + D` (Graham; Brent), so greedy is within 2x. Policies here
are greedy by construction (subject to admission). The order among ready jobs is list scheduling:
group arrival, or upward rank (critical path below a job, as in HEFT). Memory-aware admission
with reservations and backfill is EASY-style backfilling; aging bounds starvation under
priorities.

## Event loop

```rust
use sched::{BackfillConfig, JobSpec, Policy, PriorityBackfill, Resources, WorkerState};

let gb = |x: u64| Resources::mem(x << 30);
let mut policy = PriorityBackfill::new(BackfillConfig::default());

// A worker joins (and later heartbeats): 16 slots, 120 GB, 20 GB used by its runtime.
let mut w = WorkerState::new(1, "l40s", 16, gb(120));
w.reported_used = gb(20);
w.reported_baseline = gb(20);
policy.worker_update(w, 0.0);

// Jobs become ready; `group` orders them (oldest group first), then FIFO.
policy.submit(JobSpec::new(7, gb(6), /* group */ 3), 1.0);
policy.submit(JobSpec::new(8, gb(30), 3), 1.0);

// After every event, place what can be placed and send each job to its worker.
for (job, worker) in policy.dispatch(1.0) {
    println!("send job {job} to worker {worker}");
}
policy.completed(7, 95.0); // frees its slot and memory
let _ = policy.dispatch(95.0);
println!("{:?}", policy.explain(8)); // why a job is (not) running, for logs
```

The caller calls `dispatch` after every event (submission, completion, heartbeat) from a single
thread; a typical call takes microseconds.

## Model

- A **job** ([`JobSpec`]) has a demand ([`Resources`]: host and device memory), a priority group, an optional
  explicit priority, optional preferred workers (cache affinity, never required), a required worker
  class, and workers to avoid (e.g. ones it failed on). The avoid list is hard, or soft
  ([`JobSpec::avoid_soft`]): then avoided workers are used while no other live worker of the class
  exists.
- A **worker** ([`WorkerState`]) has a class, slots, a budget (host memory, and its device pool;
  a zero device budget is unknown and not enforced), a learned device demand per job
  ([`WorkerState::dev_per_task`]), and its last reported usage and baseline. The library keeps its
  own sum of the demands it placed on each worker; heartbeats only update the reported figures.
- An [`Admission`] rule decides whether a worker takes a job. The default,
  [`ProductionAdmission`], is

  ```text
  admit iff running < slots
        and (running == 0                                   // escape hatch
             or (max(reported_used, reported_baseline + placed) + demand <= budget    // host
                 and (budget.dev == 0                                                  // device
                      or max(placed.dev, running * dev_per_task)
                           + max(demand.dev, dev_per_task) <= budget.dev)))
  ```

  With `dev_per_task` alone the device rule is a per-worker count, `(running + 1) * per_task <=
  pool`; with per-job device demands it is their sum. A per-task figure should be near the mean
  job, not a high quantile: a sum of jobs concentrates near its mean.

  The escape hatch guarantees that any job can run somewhere: a job alone on a worker always goes.
  `reported_baseline` must exclude the running jobs' memory (the worker's resident floor minus
  their estimates); a floor that contains them counts them twice.

## Policies

All implement [`Policy`]; parameters are plain config structs with defaults.

| policy | order | worker choice | reservations |
|---|---|---|---|
| [`Greedy`] | arrival | preferred, then least loaded | none (big jobs starve) |
| [`PriorityBackfill`] | priority, group arrival, FIFO | preferred, then least loaded | yes |
| [`BestFit`] | as above | smallest headroom left (preference: tie-break or penalty) | yes |
| [`Lanes`] | as above | big jobs to "lane" workers first; lanes keep headroom from small jobs | yes |

**Priority and backfill.** A job may take a worker only if no more urgent waiting job is admitted
there. **Reservation:** the most urgent job that has waited at least `reserve_after` and is
admitted nowhere reserves the worker with the most headroom; nothing else is admitted there until
it is placed (at the latest when the worker empties). A more urgent starving job takes over the
least urgent holder's reservation when none are left. Every other worker keeps admitting less
urgent jobs. **Aging** (`age_limit`, [`DEFAULT_AGE_LIMIT`] = 30 minutes by default, `None` for
strict priority) puts jobs that have waited that long ahead of everything else, oldest first:
a job waits behind work submitted after it for at most the age limit.

**No starvation.** The most urgent waiting job is placed within `reserve_after` plus the longest
running time of the jobs on the worker it reserves. With `shadow_backfill`, a reserved worker
still takes jobs expected to finish before the holder could start (EASY backfilling), without
weakening that bound.

**Group order.** Groups are ordered by first arrival ([`GroupOrder::Arrival`]) or by id
([`GroupOrder::Id`]), which survives a restart that resubmits in another order ([`nassau::group`]
gives Nassau's bidegrees an id order). `group_first` orders by group before priority (e.g. oldest
bidegree first, critical path within it).

## Speed-aware placement

Workers report a [`WorkerState::speed`] and jobs may carry a [`JobSpec::work`] estimate. Every
policy takes a [`SpeedConfig`]:

- [`SpeedPolicy::FastestFirst`]: among admitting workers, the fastest.
- [`SpeedPolicy::EarliestFinish`]: HEFT's processor choice online. With a [`Defer`], a job may
  *wait* for a busy faster worker when it would still finish earlier there (by at least
  `min_gain` of its work, at most `max_wait`); [`Policy::next_wakeup`] tells the caller when a wait
  expires. In simulation of a full Nassau run it trims makespan by about 1% and bidegree latency
  p90 by 2.7x over fastest-first (more on small, heavily contended instances).
- [`SlowGate`] (HeteroPrio): keep slow workers idle while the fast class can absorb the backlog.
- [`Learn`]: learn speeds online from completion times, per worker with its class as prior,
  corrected for concurrency, with hysteresis; speed-ordered placement treats speeds within one
  `resolution` step as equal, so load still balances a class. [`SpeedEstimator`] is the same
  estimator on its own. Samples need [`JobSpec::work`].
- [`Spoliation`] (HeteroPrio): restart a running job on a faster worker that would otherwise stay
  idle, through [`Policy::dispatch_full`]'s preemptions (the caller kills and restarts).

## Thread-per-task callers: `SharedPolicy`

[`SharedPolicy`] wraps any policy for a caller that runs each task on its own thread:
[`place`](SharedPolicy::place) submits a job and blocks until it is placed (a wake handle per job,
no polling), [`place_timeout`](SharedPolicy::place_timeout) gives up and withdraws it, and
[`lease`](SharedPolicy::lease) returns a guard that releases the job if the thread unwinds.
`dispatch` runs after every call, and [`spawn_ticker`](SharedPolicy::spawn_ticker) runs it when
time passes (aging, reservations, voluntary waits).

[`failed`](SharedPolicy::failed) frees a failed job's resources without learning from its duration
and returns [`FailOutcome::Retry`] (the next `place` of the same id avoids, softly, every worker
tried) or, after [`RetryConfig::max_attempts`], [`FailOutcome::GiveUp`] with every attempt and
whether all were device OOMs.

```rust
use std::sync::Arc;
use sched::{
    BackfillConfig, FailKind, FailOutcome, JobSpec, PriorityBackfill, Resources, SharedPolicy,
    WorkerState,
};

let shared = Arc::new(SharedPolicy::with_system_clock(PriorityBackfill::new(
    BackfillConfig::default(),
)));
shared.worker_update(WorkerState::new(1, "l40s", 16, Resources::mem_gb(120.0)));
let job = JobSpec::new(42, Resources::mem_gb(6.0), 3);
loop {
    let lease = shared.lease(job.clone()); // blocks until placed
    let ok = lease.worker() == 1; // send the task to lease.worker() and wait for the reply
    if ok {
        lease.complete();
        break;
    }
    if let FailOutcome::GiveUp { .. } = lease.fail(FailKind::LinkDied, "connection reset") {
        break;
    }
}
```

## Event log

[`log::Logged`] wraps a policy and records every event at its source -- submissions (with an
optional [`log::TaskInfo`]), placements, completions, failures, worker capacity, heartbeat samples
and reservations -- to an [`EventSink`]. `log::JsonlSink` (feature `log`) writes gzip-compressed
JSON lines in the format `sched-sim --trace` reads, so a logged run is a simulator input: replaying
it with the same policy reproduces its placements.

## Dependencies: the DAG layer

[`DagScheduler`] sits in front of any policy. Jobs are declared with their dependencies
([`DagJob`]), possibly long before they are ready and possibly naming jobs not declared yet; a job
is submitted to the policy when its last dependency completes. Cycles are rejected at declaration
(the batch leaves no trace). Readiness is incremental (constant work per dependency edge);
completed jobs are removed from the graph, which is a petgraph `StableGraph`. With
`rank_priority`, jobs are prioritised by their upward rank -- their work plus the longest chain of
work below them, plus group placeholders' costs -- instead of group arrival. Work estimates can be
refined later ([`DagScheduler::update_work`]), moving ranks up or down. **Passthrough** jobs are
pure synchronisation points ("group G is done") that complete by themselves, and a
[`DagTemplate`] is a dependency structure shared by many groups (e.g. one per algebra), checked
once and instantiated per group with [`DagScheduler::declare_template`]; its
[`critical_path`](DagTemplate::critical_path) gives an unexpanded group's rank weight.

**Implicit instances** ([`DagScheduler::open_instance`], after PaRSEC's parameterised task graphs)
keep a group's sub-DAG as dense counters over its shared template instead of graph nodes and
edges: about 11 bytes per node, a `JobSpec` built only when a node becomes ready, the group's
`done` job completed when its last node does, and ranks flowing across instances as along edges.
`DagConfig::max_open_instances` bounds how many are open at once (a frontier budget). An instance
can carry a demand and a label per node, open with nodes already complete (resuming from a
checkpoint: those never run, their successors start with them met), and be closed early
([`DagScheduler::close_instance`]: unstarted nodes complete as no-ops, running ones are returned and
their later completions only free resources).

**Local jobs** ([`DagJob::local`]) run on the caller (registration, loading, committing): when
ready they are held, never submitted, and returned by [`DagScheduler::take_local`].

With the `serde` feature (default) the declared graph, instances included, can be snapshotted and
restored; jobs that were submitted or running are submitted again on restore.

## Simulator

The `sim` feature builds `sched-sim`, which replays a JSONL trace (workers, tasks with dependencies,
per-minute memory samples) against the policies under a fitted processor-sharing service model and
reports throughput, utilisation, wait distributions, group latency and reservation cost:

```text
cargo run --release --features sim --bin sched-sim -- --trace sched_trace.jsonl.gz --json out.json
cargo run --release --features sim --bin sched-sim -- --trace ... --closed --rank --age-limit 1800
```

`--closed` derives arrivals from simulated dependency completions through the DAG layer instead of
the trace's ready times. Two more binaries:

- `sched-whole` builds a whole Nassau run as one DAG a priori (bidegrees, signature-DAG templates,
  census-fitted costs) and compares dispatch plans against its lower bounds; `--export-dslab`
  writes an instance for dslab-dag cross-validation.
- `sched-pisa` compares two plans on many small instances (random mini-grids, or perturbed replicas
  of real data), typically and adversarially (simulated annealing with witness minimisation).

See `RESULTS.md` for results, `research/` for what was taken from dslab-dag, StarPU, Batsim, SAGA
and PaRSEC, and `LITERATURE.md` for references.

## Features

- `serde` (default): DAG snapshots (`petgraph/serde-1`).
- `log`: the JSONL event-log writer (adds `serde_json`, `flate2`).
- `sim`: the simulator (`log`, plus `clap`).

Without features the only dependency is `petgraph`.

## Integration notes (Nassau's coordinator)

Phase 1 keeps the thread per task and replaces the inside of `acquire`/`release`:

```rust,no_run
# #[cfg(feature = "log")]
# fn main() {
use std::{sync::Arc, time::Duration};
use sched::{
    BackfillConfig, FailKind, FailOutcome, GroupOrder, JobSpec, Learn, PriorityBackfill,
    Resources, SharedPolicy, SpeedConfig, SpeedPolicy, WorkerState,
    log::{JsonlSink, Logged, TaskInfo},
    nassau,
};

// Once: restart-stable bidegree order, 30-minute aging (the default), fast workers first with
// speeds learned per worker, every decision logged.
let policy = PriorityBackfill::new(BackfillConfig {
    group_order: GroupOrder::Id,
    speed: SpeedConfig {
        policy: SpeedPolicy::FastestFirst,
        learn: Some(Learn::default()),
        ..SpeedConfig::default()
    },
    ..BackfillConfig::default()
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
    dev_per_task: (dev_demand * 1e9) as u64,
    ..WorkerState::new(id, class, 16, Resources::mem_gb(123.7).with_dev_gb(dev_cap))
});

// acquire(res, key, est, b, what, avoid): one task, its bidegree (s, t), its estimate in GB and
// expected seconds on an H200 (learning needs a work estimate; any size proxy proportional to it
// works). A per-task device estimate, when known, goes in `spec.demand.dev` (`with_dev_gb`).
let (task, s, t, est_gb, work) = (123_456, 3, 200, 9.5, 600.0);
shared.with(|p, _| {
    p.annotate(task, TaskInfo { kind: "sig".into(), bidegree: (t - s, s), ..TaskInfo::default() })
});
let mut spec = JobSpec::new(task, Resources::mem_gb(est_gb), nassau::group(s as u32, t as u32));
spec.work = Some(work);
loop {
    let lease = shared.lease(spec.clone()); // blocks; no polling
    let worker = lease.worker(); // send over TCP, block on the reply
    let reply: Result<(), (FailKind, String)> = Ok(());
    let _ = worker;
    match reply {
        Ok(()) => break lease.complete(), // release
        Err((kind, why)) => match lease.fail(kind, &why) {
            FailOutcome::Retry { .. } => continue, // avoids the workers tried, softly
            FailOutcome::GiveUp { retryable, .. } => {
                let _ = retryable; // all attempts were DeviceOom: retry at the bidegree level
                break;
            }
        },
    }
}

// A worker left: its tasks' threads will fail with LinkDied and retry.
let _lost = shared.worker_gone(id);
# }
# #[cfg(not(feature = "log"))]
# fn main() {}
```

Phase 2 drives the coordinator from a [`DagScheduler`]: bidegrees' zero steps, registrations and
commits as explicit (local) jobs, each walk an instance of its profile's signature template with
per-node demands, opened with its checkpointed nodes complete and closed early at the dead tail;
`snapshot`/`restore` across coordinator restarts.

## License

MIT OR Apache-2.0.
