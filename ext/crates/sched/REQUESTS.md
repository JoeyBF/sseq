# Requests: what `sched` needs to drive Nassau

From the `ext` side, 2026-10-02. Read with `RESULTS.md` (yours) and the spec. Ordered by phase;
within a phase, by priority. Each item has the API we would like (adapt freely if you see a better
shape), the semantics we rely on, and what "done" means. Nothing here changes `ext`; the
integration is ours (section "Who does what").

## Context you need

**Today's coordinator** (`ext/src/nassau/remote.rs`, branch `sig-parallel-lift`, currently
`fbc57a1697` in production):
- **Placement:** a thread per waiting task calls `acquire(res, key, est, b, what, avoid)` and
  BLOCKS until it gets a worker. It then sends the task over TCP, blocks on the reply, and calls
  `release`.
  - The queue is in-process: FIFO by ready time, plus one reservation for a memory-blocked task.
  - About 900 tasks wait at the frontier, each polling every 50 ms.
- **Retries:** up to 4 attempts per task, skipping workers already tried while others are live.
  After the last one, the coordinator panics with the last cause; an OOM cause is classified as
  retryable at the bidegree level.
- **Workers:** about 21 processes (7 H200, 14 L40S), 16 slots each, joining and leaving whenever
  Slurm decides.
  - **Heartbeat** (`MemReport`): `rss, budget, reserved, running, waiting, baseline`.
    - `baseline` is the gate's rolling RSS floor; it contains running tasks (your double count).
    - Budgets are 123.7 GB (L40S) and 154.6 GB (H200).
  - **Measured speed** on this workload: per GPU, an L40S does about 1.39× the work of an H200.
    Your per-process fit says 2.41×.
- **Per-task memory estimate:** `mem::sig_task_bytes`, recalibrated today: median 1.9× the
  measured peak, never below 1.23×.
- **Bidegrees:** every open bidegree is a rayon thread parked in `thread::scope` for its whole
  walk, hence the 24–128 cap.
  - `depgraph` (`nassau.rs` around 7600) has the bidegree-level edges your whole-run model uses.
  - A walk dispatches each signature when its direct predecessors (`sig_dag::direct`) are done.
- **Live evidence for aging:** strict oldest-bidegree-first, with "oldest" taken from
  post-restart replay order, starved younger bidegrees for over 5 h (job 40506688,
  2026-10-02 04:46–11:08).

**Our integration model, phase 1:** we keep the thread-per-task structure and replace only the
inside of `acquire`/`release` with your policy. Phase 2 makes the coordinator event-driven.

---

## Phase 1: placement only (blocking us now)

### R1. A thread-safe blocking front end (`SharedPolicy`)

The pure `Policy` is right. We also need a small shim our task threads can block on, so `acquire`
becomes "submit, then wait for my placement".

```rust
pub struct SharedPolicy<P: Policy> { /* Mutex<P> + Condvar + dispatcher state */ }
impl<P: Policy> SharedPolicy<P> {
    pub fn new(policy: P, clock: impl Fn() -> Instant + Send + Sync + 'static) -> Self;
    /// Submit and block until the job is placed; returns the worker. Cancellation-safe.
    pub fn place(&self, job: JobSpec) -> Placement;            // Placement { worker, ... }
    pub fn place_timeout(&self, job: JobSpec, t: Duration) -> Result<Placement, JobSpec>;
    pub fn completed(&self, job: JobId);
    pub fn failed(&self, job: JobId, why: &str) -> FailOutcome; // see R2
    pub fn worker_update(&self, w: WorkerState);
    pub fn worker_gone(&self, w: WorkerId) -> Vec<JobId>;      // jobs that were running there
    pub fn explain(&self, job: JobId) -> Option<String>;
    pub fn stats(&self) -> PolicyStats;
}
```

- **Dispatch timing:** run `dispatch` after every mutating call, and on a periodic tick (aging and
  reservation deadlines need time to pass). Wake exactly the threads whose jobs were placed; no
  polling.
- **Load:** about 1,000 blocked threads, 100–200 events/s. `dispatch` p99 under 1 ms at that size
  (you report µs, so this is a regression guard).
- **Dropped or cancelled `place`** (the caller unwinds): the job is cancelled and nothing leaks.
- **Done:** a stress test with 1,000 threads, random completions and worker churn, checking for no
  lost wakeups and no double placement, plus a loom or model test of the wake logic if you have
  time.

### R2. Failure semantics: `failed(job, why)`

Distinct from `completed`.
- **Frees the job's resources** and records `why` against the worker.
- **Returns `FailOutcome::Retry { avoid }` or `GiveUp { tried, last_why }`.** It gives up after
  `max_attempts` (config, default 4).
- **Retries avoid the workers already tried,** using `JobSpec::avoid`, but only while some other
  eligible live worker exists. Your `avoid` is a hard constraint, and the policy, not the caller,
  should apply the "only if another worker exists" rule (see R3).
- **Classify the cause**, e.g. a small `enum FailKind { DeviceOom, LinkDied, Rejected, Timeout,
  Other }` that we pass in. Let the policy report `GiveUp.retryable = all attempts were DeviceOom`,
  so we stop re-deriving it from panic text.
- **Done:** property tests for attempt counting and the avoid rule; resources never leak through
  any sequence of failures and departures.

### R3. Soft avoid

Today `avoid` is hard: "a job that avoids every worker waits until one it does not avoid joins".
- **Add** `avoid_soft: bool` (or a separate `prefer_not`). With it, avoided workers are allowed
  when no other eligible live worker exists.
- **Retries from R2 use the soft form.**
- **Why:** on 2026-10-02 every attempt of one task landed on a single OOM'd card, which re-joined
  empty. A hard avoid would have parked the task forever if that card had been the only one live.

### R4. Admission without the double count

Your headline finding.
- **We'll change the worker** to report `baseline_excl` = its rolling RSS floor minus the
  estimates of the tasks it's running. A new heartbeat field, sent alongside the old one.
- **We need an `Admission` impl** that uses it:
  `admit iff running < slots && (running == 0 || max(rss, baseline_excl + Σ placed) + demand <= budget)`.
  - Keep `max(rss, …)` as the safety term, since estimates can still undercount.
  - Keep `ProductionAdmission` selectable for A/B.
- **Replay it on the trace:**
  - rebuild `baseline_excl` from the samples, as you did for the floor;
  - report waits and slot use against today's rule, as in your "idle baseline" row;
  - and, important for us, report how often `rss` alone would exceed `budget` (the overrun risk you
    flagged).
  - With the calibrated estimates (1.9× median instead of 6×) the double count is already much
    smaller. Please also replay with estimates scaled ×0.32 to mimic the recalibration
    (median-ratio scaling; not exact, but indicative).
- **Done:** the impl and a unit test with the formula above; a RESULTS.md row for the replay,
  including the overrun count.

### R5. Restart-stable priority, aging on by default

- **Helper:** `JobSpec` takes the priority; give us a documented way to express "oldest bidegree
  first" by DAG position, not arrival.
  - Suggested key: the bidegree's DAG release order, e.g. `(t, s)` or your s-major id; you know
    which one your whole-run results used, so use that.
  - A function `group_priority(s, t) -> i64` in a small `nassau` helper module is fine. Arrival
    order is what broke after our restart.
- **Defaults:** `age_limit` 1,800 s for PriorityBackfill and BestFit, as your recommendation says.
  Keep `None` reachable.
- **Done:** a test that a group's priority doesn't depend on submission order, and a starvation
  test with a young group behind a wide old group (our incident) showing waits bounded by
  `age_limit`.

### R6. Worker speed from completions

The caller sets `WorkerState::speed` today.
- **We'd like a helper,** `SpeedEstimator`, that learns per-class speed online from
  `(class, work_estimate, observed_time, concurrency)` at completion, with a prior per class.
  Priors: H200 1.0, L40S 2.4 from your fit; let the run correct them.
- **Hysteresis or EWMA,** so one slow task doesn't flip placement.
- **Clock-capped cards** (some H200s run at 1,365 MHz instead of 1,785) should show up as slower
  workers of the same class. Per-worker estimates with class priors, not per-class only.
- **Done:** unit tests; the sim using estimated instead of oracle speeds keeps most of the
  fast-first gain, reported in RESULTS.md.

### R7. Structured event log (closing the loop)

The user asked for DAGs and events to be stored explicitly from now on, not reconstructed from
logs (we had to rebuild this trace from logs, and it cost a day).
- **Add** an `EventSink` trait (or a JSONL writer behind a feature) that records `submit`
  (spec, ready time), `placed` (worker, time), `completed`/`failed`, worker join/update/leave,
  reservations and age promotions.
- **Format:** exactly what `sched-sim` reads, so every production run is a simulator input.
- **Size:** about 200k jobs a day plus 1-minute worker samples, gzip, under 50 MB a day.
- **Done:** round trip, meaning a simulated run's event log fed back to `sched-sim` reproduces its
  placements.

---

## Phase 2: the DAG drives the coordinator (we start this once phase 1 runs)

### R8. Per-node demand (and labels) in instances

`InstanceSpec` gives every node one `proto`, but a walk's signatures differ in memory: each has its
own estimate from its target and next sizes.
- **Add** `demand: Vec<Resources>` (or `Arc<dyn Fn(usize) -> Resources>`) and optionally
  `label: Arc<dyn Fn(usize) -> String>` for `explain`.
- **Memory:** the per-node cost stays small. 300 open walks × up to 4,096 nodes; A(4) is 32,768.

### R9. Opening an instance with nodes already complete

For resuming from our signature checkpoint (one Zarr store per bidegree, keyed by the signature's
lex index).
- **Add** `InstanceSpec::completed: Option<FixedBitSet>` (or `Vec<u32>`). Those nodes never
  dispatch; their successors' counters start decremented.
- The completed set need not be closed under predecessors. Our outputs are pure functions of their
  DAG predecessors, so a missing ancestor is recomputed. Just never re-run a completed node.
- **Done:** property test that opening with set S equals opening and then completing S in any
  topological order.

### R10. Closing an instance early

Our walks can stop early: once no remaining signature can contribute (the dead tail), the rest are
no-ops.
- **Add a public** `close_instance(group_or_done_id)`. It completes every not-started node as a
  no-op and fires `done`.
- **Nodes already running** finish, and their `completed` is ignored. Return their ids so we can
  drop the replies.
- **Done:** tests for no double `done`, and for resources of ignored completions being freed.

### R11. Coordinator-local jobs

Registration, loading a saved bidegree and the commit step run on the coordinator, not on workers.
- **Confirm** that `auto_submit = false` plus `take_ready`/`release` is the intended way to receive
  ready local jobs without placement. If so, document it with an example.
- **Otherwise, add a `Local` class** that the policy never places but the DAG layer tracks.

### R12. Restart and resume

- **`snapshot`/`restore` must round-trip** an open frontier: hundreds of instances, partial
  completions and placeholders.
- **On restore, jobs that were placed or running are returned as ready.** The workers are gone
  after a coordinator restart.
- **Done:** property test of snapshot after N random events, restore, continue, giving the same
  completions as uninterrupted (modulo placement).

---

## Not needed yet (but keep in mind)

- **Lookahead reservation (spec §4b):** your HEFT comparison says full lookahead is worth about
  15%. Later.
- **Lanes:** they lost on slot utilisation in your replay, so they're not needed for now.
- **Preempting onto faster workers:** keep it off by default. A preempted task loses its partial
  GPU work, and our tasks are minutes long.

## Who does what

- **You (crate):** R1–R7 first (phase 1), then R8–R12. Results in RESULTS.md, items in README.
  Same standards as so far: property tests, no runtime dependencies beyond petgraph, a standalone
  crate.
- **Us (`ext`):**
  - the worker-side `baseline_excl` heartbeat field (R4);
  - replacing `acquire`/`release` with `SharedPolicy` (phase 1);
  - the event-driven coordinator, signature checkpointing and the `depgraph` → `Dag` migration
    (phase 2);
  - byte-identical chart checks at stems 120–200 and the kill-and-resume emulation.
- **Hand-off signal:** tag the commit that completes R1–R7 (e.g. `sched-phase1`) and add a short
  "integration notes" section to README, with the exact calls our `acquire` should make.

---

## Status (from the crate side, 2026-10-02)

All of R1–R12 are implemented and tested on branch `worktree-sched-crate`; the tag
`sched-phase1` marks it. README "Integration notes" has the exact calls for `acquire`; RESULTS.md
has the replays.

| # | what you get | tests |
|---|---|---|
| R1 | `SharedPolicy::{place, place_timeout, lease, completed, failed, abandon, worker_update, worker_gone, tick, spawn_ticker, explain, stats, with}`; one wake handle per waiting job; `dispatch` after every call; the ticker sleeps until `next_wakeup` | 1,000-thread stress test with worker churn (no lost wakeup, no double placement, slots never exceeded); dispatch p99 < 1 ms at 21 x 16 slots and 1,000 waiting |
| R2 | `failed(job, FailKind, why) -> FailOutcome::{Retry { attempts, avoid }, GiveUp { tried, retryable }}`, `RetryConfig { max_attempts: 4 }`; `Policy::failed` frees without learning a speed | property test of attempt counting, the avoid rule and leaks through failures and departures |
| R3 | `JobSpec::avoid_soft`; retries from R2 use it automatically. "Live" = has slots and the job's class, not "free": a retry waits for a busy healthy worker | unit tests, plus the invariants property test (it found a stale-reservation bug, fixed) |
| R4 | No new `Admission`: send `baseline_excl` as `reported_baseline` and `ProductionAdmission` is your formula (unit test). Replay in RESULTS.md: wait p99 30 min -> 8 s; modelled overruns 0.008% of heartbeats with backfill, 0.19% with best fit | `sched-sim --baseline excl --est-scale X` |
| R5 | `GroupOrder::Id` + `nassau::group(s, t)` (ordered by `(s, t)`, the best restart-stable key in the whole-run simulation); `age_limit` defaults to `DEFAULT_AGE_LIMIT` = 1800 s | submission-order independence; the young-behind-wide-old-group incident bounded by the age limit |
| R6 | `SpeedConfig::learn: Some(Learn::default())` learns per worker (class as prior, concurrency-corrected, 10% hysteresis); `SpeedEstimator` on its own. Needs `JobSpec::work` | unit tests incl. a clock-capped H200; the whole run with learned speeds keeps the fast-first gain (+0.4%) |
| R7 | `log::Logged<P>` (wrap the policy inside `SharedPolicy`), `log::JsonlSink` (one gzip member per flush, so a killed process keeps its log up to the last flush), `Logged::annotate(job, TaskInfo)` for bidegree, kind and deps; `sched-sim --trace` reads the log | round trip: a logged run replays to identical placements |
| R8 | `InstanceSpec::{demand: Option<Arc<[Resources]>>, label: Option<NodeLabel>}` | unit test |
| R9 | `InstanceSpec::completed: Vec<u32>` | property test: opening with S equals opening, then completing S |
| R10 | `DagScheduler::close_instance(done_or_node, now) -> running ids` | no double `done`; ignored completions free resources |
| R11 | `DagJob::local()` + `DagScheduler::take_local()`: never submitted, never released | unit test |
| R12 | `snapshot`/`restore` (placed or running jobs come back submitted) | property test: random workloads, restarts at random points, every job completes exactly once |

Three things that differ from what you assumed:

1. **Speed ratio.** At your measured 1.39x (L40S/H200), fast-first is worth about 9% on top of
   uncapping (it was about 25% at the fitted 2.41x); uncapping is still about 21%. Today ->
   uncapped + fast first is 1.38x (1.65x at 2.41x). Per-worker learning finds the true ratio;
   start from your measured priors.
2. **Estimates on the reference trace are about 1.7x actual use, not 6x**: the median of
   (RSS - idle) / estimates running is 0.595. Scaling them by 0.32 would put them below actual use.
3. **Aging is FIFO by submission** among aged jobs: it bounds a job's wait behind work submitted
   after it, not behind work already queued. A restarted coordinator that resubmits everything at
   once should resubmit in `nassau::group` order.
