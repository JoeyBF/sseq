# sched

A pure, deterministic, resource-aware job-placement library: given a stream of jobs that each
declare a resource demand, and a changing pool of workers that each have a capacity and a number
of execution slots, it decides **which waiting job goes to which worker, and when**. Nothing else
-- no networking, threads, clocks or persistence. Every input is an event carrying the caller's
"now", and the same events produce the same placements.

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

- A **job** ([`JobSpec`]) has a demand ([`Resources`], memory today), a priority group, an optional
  explicit priority, optional preferred workers (cache affinity, never required), and two hard
  constraints: workers to avoid (e.g. ones it failed on) and a required worker class.
- A **worker** ([`WorkerState`]) has a class, slots, a budget, and its last reported usage and
  baseline. The library keeps its own sum of the demands it placed on each worker; heartbeats only
  update the reported figures.
- An [`Admission`] rule decides whether a worker takes a job. The default,
  [`ProductionAdmission`], is

  ```text
  admit iff running < slots
        and (running == 0                                   // escape hatch
             or max(reported_used, reported_baseline + placed) + demand <= budget)
  ```

  The escape hatch guarantees that any job can run somewhere: a job alone on a worker always goes.

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
urgent jobs. Optional **aging** (`age_limit`) puts long-waiting jobs ahead of everything else,
which bounds waits even under priorities that are not arrival-ordered (such as DAG ranks).

**No starvation.** The most urgent waiting job is placed within `reserve_after` plus the longest
running time of the jobs on the worker it reserves.

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
[`critical_path`](DagTemplate::critical_path) gives an unexpanded group's rank weight. With the
`serde` feature (default) the declared graph can be snapshotted and restored.

## Simulator

The `sim` feature builds `sched-sim`, which replays a JSONL trace (workers, tasks with dependencies,
per-minute memory samples) against the policies under a fitted processor-sharing service model and
reports throughput, utilisation, wait distributions, group latency and reservation cost:

```text
cargo run --release --features sim --bin sched-sim -- --trace sched_trace.jsonl.gz --json out.json
cargo run --release --features sim --bin sched-sim -- --trace ... --closed --rank --age-limit 1800
```

`--closed` derives arrivals from simulated dependency completions through the DAG layer instead of
the trace's ready times. See `RESULTS.md` for the reference trace.

## Features

- `serde` (default): DAG snapshots (`petgraph/serde-1`).
- `sim`: the simulator (adds `serde_json`, `flate2`, `clap`).

Without features the only dependency is `petgraph`.

## License

MIT OR Apache-2.0.
