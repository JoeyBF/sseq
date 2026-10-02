# Batsim / batsched / pybatsim: what to extract into `sched`

## Sources and versions

All clones are under `/wsu/home/hd/hd72/hd7264/.claude/jobs/71ebdf0b/tmp/refs/`, shallow, from gitlab.inria.fr/batsim:

- `batsim/`: master. Its protocol is now the **batprotocol** (Flatbuffers or JSON, `docs/protocol.rst:76-98`).
- `batsched/`: master, with the C++ reference schedulers. It still speaks the **old JSON protocol**: `JOB_SUBMITTED`, `EXECUTE_JOB` and so on (`src/main.cpp:402-416`, `src/protocol.cpp:184-372`). Its CHANGELOG targets Batsim v2 to v4. **So batsched master and batsim master are not wire-compatible.** Running them together means pairing batsched with a Batsim v4.x release (not verified).
- `pybatsim/`: master (v4 beta). It is now a thin EDC library with only an FCFS example (`pybatsim-core/src/pybatsim/external_decision_components/fcfs.py`). The classic schedulers were removed.
- `pybatsim-v3/`: tag v3.2.1. It still has `schedulers/easyBackfill.py` and the `batsim.sched` framework with reservation-depth backfilling.

What I read in our crate:
- `README.md`, `RESULTS.md`, `LITERATURE.md` §5 and §9.
- `src/lib.rs`, `src/engine.rs`, `src/admission.rs`.
- `tests/{invariants,starvation,reservation}.rs`.
- Skimmed `src/sim/run.rs`, `src/sim/trace.rs` and `src/sim/model.rs`.

---

## 1. Their algorithms, exactly

### 1.0 Model they all share

- **Resources.** A platform is `nb_machines` identical hosts. A job requests `res` hosts (rigid, an integer) and holds them exclusively. Batsim itself lets several jobs share a host (`machines.cpp:471`, `jobs_being_computed` is a set), but no reference scheduler uses that.
- **No second resource.** None of them models memory or any other resource. The only "multi-resource" aspect is *which* hosts:
  - batsched's `ContiguousResourceSelector` (`main.cpp:220-223`);
  - pybatsim's consecutive-resources filter (`batsim/sched/algorithms/utils.py:195-209, 376-388`);
  - easyBackfill.py's "rectangles" (`schedulers/easyBackfill.py:1-4`).
- **Walltime is a hard upper bound.** Batsim kills a job when it reaches its walltime. `execute_job_process` runs the task with `remaining_time = walltime` and maps return code -1 to `COMPLETED_WALLTIME_REACHED` (`batsim/src/jobs_execution.cpp:169, 197-201`). The walltime is optional in the workload (`docs/input-workload.rst`, "walltime"; `src/jobs.cpp:374-387`).
- **batsched pads every walltime.** It adds `rjms_delay` (default 5 s, `main.cpp:104`) to every walltime (`json_workload.cpp:49, 62`).
- **Jobs without a walltime are rejected** by every backfilling variant:
  - `easy_bf.cpp:55-60`;
  - `easy_bf_fast.cpp:195-198`;
  - `conservative_bf.cpp:61-66`.
- **Jobs larger than the machine are rejected** too (`easy_bf.cpp:51-54`).
- **Consequence: no running job ever outlives its estimate.** Their "finish early" handling is just "release the rest of the reservation". They have no overrun path at all.
- **One exception.** pybatsim's `find_first_time_to_fit_walltime` treats an allocation whose kill is still in flight as busy until its real end (`batsim/sched/resource.py:320-336`).
- **Event batching.** A batsched decision call sees every event at the current timestamp together: `_jobs_released_recently`, `_jobs_ended_recently` and `_jobs_killed_recently` (`isalgorithm.hpp:156-158`). It then decides once (`make_decisions`). Our caller instead calls `dispatch` after each event, or after a batch at its choice.

### 1.1 batsched `easy_bf_fast`: classic EASY with shadow time and extra nodes

`src/algo/easy_bf_fast.cpp`. The file's own description (`:40-56`): FCFS only, one priority job, walltimes required, and it "will sometimes be a little more greedy" than `easy_bf`.

**State:**
- `_available_machines` and `_nb_available_machines`;
- `_pending_jobs`, the FCFS list without the priority job;
- `_horizons`, a sorted list of `(walltime end date, released machines)` for the running jobs;
- `_priority_job`;
- `_priority_job_expected_start_time`, the shadow time;
- `_remaining_resources_at_priority_job_start`, the extra nodes (`easy_bf_fast.hpp:338-355`).

```
make_decisions(now):
  for each job e that ended:                                   # :61-73
      free e's machines; erase e's horizon point
  if some job ended and P (priority job) exists:               # :76-181
      if P.res <= free:                                        # :83-99
          start P now; insert horizon (now+P.walltime, P.res); P := none
          for j in pending (FCFS):                             # :102-133
              if j.res <= free: start j now, add horizon
              else: P := j; recompute_shadow(); remove j from pending; break
      if free > 0:                                             # :137-180
          if P exists: recompute_shadow()                      # :141-142  (early finish moves shadow)
          for j in pending (FCFS):                             # :144-179
              if j.res <= free and
                 (now + j.walltime <= shadow  or  j.res <= extra):   # :149-151
                  start j; add horizon
                  if now + j.walltime > shadow: extra -= j.res        # :168-169
                  if free == 0: break
  for each newly released job n (in order):                    # :185-253
      reject if n.res > nb_machines or n has no walltime
      elif n.res <= free:
          if P is none or now+n.walltime <= shadow or n.res <= extra:   # :205-207
              start n; add horizon; if P and now+n.walltime > shadow: extra -= n.res   # :226-227
          else: pending.push_back(n)                           # :234
      else:
          if P is none: P := n; recompute_shadow()             # :241-246
          else: pending.push_back(n)

recompute_shadow():                                            # :256-275
  avail := free
  for (date, released) in horizons (ascending date):
      avail += released
      if avail >= P.res: shadow := date; extra := avail - P.res; return
  assert unreachable ("job will never be executable")
```

- **Reservation depth** is exactly 1. The reservation is a *count* of hosts at a *time*. No specific hosts are reserved.
- **Queue order:** FCFS only. `_pending_jobs` is never sorted.
- **Early finish:** handled implicitly. A finished job's horizon point is erased (`:71`) and the shadow is recomputed (`:141-142`).
- **Walltime overrun:** impossible, because Batsim kills the job at its walltime.

### 1.2 batsched `easy_bf`: EASY on a full 2-D schedule (machine-specific reservation)

`src/algo/easy_bf.cpp`, with the `Schedule` profile in `src/schedule.cpp`.

**`Schedule`.** A list of time slices (`begin`, `end`, `available_machines` as an IntervalSet, `allocated_jobs`), starting with one slice `[now, 1e19)` (`schedule.cpp:13-29`).

**`add_job_first_fit(job)`** (`:133-291`):
- It scans the slices for the first anchor slice with `nb_available >= res` (`:157-160`).
- It then intersects the available machine sets over the following slices until their total length reaches the walltime (`:217-233`).
- If the intersection keeps at least `res` machines, the selector picks the machines; `BasicResourceSelector` takes the lowest ids (`locality.cpp:29-39`).
- The job is then inserted into every slice it covers, splitting the slice at `begin + walltime` (`:236-265`).
- `started_in_first_slice` says whether the job starts now (`:187, 243`).

**`remove_job`** (`:1031-...`) gives the machines back in every slice and merges equal neighbours. That is how an early finish frees the rest of a reservation.

**`update_first_slice(now)`** moves the first slice's begin to `now` and asserts `now <= first slice end` (`:46-69`). This only holds because Batsim kills jobs at their walltime.

```
make_decisions(now):
  P_before := queue.first
  for each ended e: schedule.remove_job(e)                    # :42-43
  queue newly released jobs (reject oversize / no walltime)   # :46-66
  schedule.update_first_slice(now)                            # :69
  sort queue; P_after := queue.first                          # :150-162
  if P_after != P_before:                                     # :165-188
      remove P_before's reservation from the schedule if present
      loop: add_job_first_fit(P_after) (a reservation, maybe in the future)
            if it starts in the first slice: execute it, P_after := next head, repeat
  if no job ended:                                            # :76-101
      for each newly queued j != P_after with j.res <= free_now:
          alloc := add_job_first_fit(j)   # must avoid P's reserved machines over its window
          if alloc starts now: execute j  else remove j from the schedule
  else:                                                       # :103-146
      for j in queue order while free_now > 0:
          remove j from the schedule if present  (this also re-plans P: "compression")
          alloc := add_job_first_fit(j)
          if starts now: execute j; if j was P: P := new queue head
          else if j != P: remove j from the schedule   # only P keeps a reservation (depth 1)
```

How it differs from `easy_bf_fast`:
- The priority job's reservation is on **specific machines** in a specific window. A backfilled job must fit around those exact machines over its whole walltime, which is stricter than the count-based "extra nodes" test. The file comment `easy_bf_fast.cpp:52-56` acknowledges this.
- The queue order is pluggable (`-o`, `main.cpp:86, 200-215`):
  - `fcfs` and `lcfs` sort by release date, ties by job id **string** (`queue.cpp:43-72`);
  - `asc_size` and `desc_size`, and `asc_walltime` and `desc_walltime` (`queue.cpp:135-208`);
  - `desc_slowdown` and `desc_bounded_slowdown`.
- **The two slowdown orders collapse to FCFS.** `update_slowdown` sets `slowdown = now - release_date`, which is the *waiting time*, not a ratio (`queue.cpp:27-30`). `bounded_slowdown = (now - release)/min_job_length` with `min_job_length = 1` (`queue.cpp:32-35`, `main.cpp:204-205`). Both are monotone in release date, so both equal FCFS apart from ties.
- **With a non-FCFS order the head can change at every event.** The reservation then moves to the new head (`easy_bf.cpp:165-169`), which can starve large jobs; under `asc_walltime` the long ones starve. batsched has no aging.

### 1.3 batsched `conservative_bf`: a reservation for every queued job, plus compression

`src/algo/conservative_bf.cpp:43-137`.

```
make_decisions(now):
  for each ended e: schedule.remove_job(e)                     # :48-49
  queue newly released jobs (reject oversize / no walltime)    # :53-72
  schedule.update_first_slice(now); sort queue                 # :75-78
  if no job ended:                                             # :81-95
      for each newly queued j: alloc := add_job_first_fit(j)   # it KEEPS its reservation
          if it starts now: execute j, dequeue
  else:  # "compress the schedule"                             # :96-124
      for j in queue order:
          schedule.remove_job_if_exists(j); alloc := add_job_first_fit(j)
          if it starts now: execute j, dequeue
  answer waiting-time queries: schedule.query_wait(res, walltime)   # :128-133
```

- Every queued job keeps a reservation, so the reservation depth is unbounded.
- A new job is added after all existing reservations and can never delay them.
- On any completion each job is re-planned in queue order. Its old slot is free during the re-plan, so it can only move earlier (or stay).
- Later jobs keep their reservations while earlier ones re-plan, so the re-plan cannot push a later job back. This is the predictability property of Mu'alem and Feitelson.
- A by-product is a waiting-time prediction for each job (`query_wait`, `schedule.cpp:330-398`).

### 1.4 pybatsim v3: reservation depth k, and an SJF backfill order

**`batsim/sched/algorithms/backfilling.py`:**
- `backfilling_sched(reservation_depth=1, backfilling_sort=None, priority_sort=None, ...)` (`:80-153`) first runs `filler_sched(abort_on_first_nonfitting=True)`: it starts jobs in order until the first one that does not fit (`:139-144`).
- If jobs remain, `do_backfilling` (`:14-77`) runs:
  - `reserved = runnable[:k]`;
  - `remaining = runnable[k:]`, optionally re-sorted by `backfilling_sort` (`:58-67`).
- `schedulers/schedEasySjfBackfill.py:16-28` sets `reservation_depth` from the option `backfilling_reservation_depth` (default 1) and `backfilling_sort = requested_time`. That is EASY with **SJF order for the backfill candidates only**; the reserved head stays in queue order.

**`utils.py: find_resources_without_delaying_priority_jobs`** (`:15-110`). For **each** backfill candidate:
- It temporarily allocates each of the k reserved jobs at its earliest start (`find_with_earliest_start_time(..., allow_future_allocations=True)`, `:46-83`). This is sequential, so later reserved jobs see the earlier ones.
- It then asks whether the candidate fits **now** (`:90-91`), with an optional `check_func` veto (`:93-95`).
- Finally it frees the temporary reservations (`:101-103`).
- The reservations are therefore **recomputed from scratch for every candidate and every call**. Nothing persists between calls, unlike batsched's `conservative_bf`, and `k = ∞` here is a "no-guarantee" conservative.
- The cost is O(candidates × k × allocations).

**`resource.py: find_first_time_to_fit_walltime`** (`:297-367`) is first fit in time against `estimated_end_time = start + walltime` (`alloc.py:88-93`).

**`schedulers/easyBackfill.py`** is a separate EASY on a contiguous free-space list:
- the shadow time is `findAllocFuture` (`:323-337`), which releases running jobs in order of estimated finish until the head fits;
- the shortened free spaces have length `shadow - now` (`:339-411`);
- a candidate is backfilled if `res <= space.res and requested_time <= space.length` (`:300-308`).

It is EASY with the extra nodes expressed as the free spaces not shortened, plus contiguity.

### 1.5 How this differs from our reservation and backfill

| | batsched / pybatsim EASY | `sched::PriorityBackfill` (`engine.rs:518-655`) |
|---|---|---|
| Who reserves | The queue head, immediately | The most urgent job that has waited at least `reserve_after` **and is admitted nowhere** (`try_reserve`, `engine.rs:518-524`) |
| What is reserved | k hosts from time *shadow* on (count-based in fast, specific machines in easy_bf) | A **whole worker**, with no time (`engine.rs:553-558`) |
| Which machines | The earliest feasible start | The worker with the **most headroom**; ties: preferred, then fewest running (`engine.rs:534-552`) |
| Backfill on the reserved capacity | Yes: jobs that end before the shadow time, or fit in the extra capacity | **None**: `refusal` returns `ReservedFor` for every non-holder (`engine.rs:388-392`). The worker drains until the holder fits; at the latest it empties and the escape hatch admits the holder (`admission.rs:70-75`) |
| Backfill elsewhere | Every candidate not hindering the head | Every unreserved worker admits less urgent jobs, subject to the priority invariant |
| Runtime information | Walltime, required | None |
| Resource model | One integer per job, identical hosts, no slots | A memory vector plus slots per worker, heterogeneous workers, a reported-usage term, and an escape hatch |
| Depth | 1 (EASY), ∞ (conservative), k (pybatsim) | `max_reservations` (default 1, optionally per class) |
| Changing the head | The reservation moves (no protection) | Takeover only by a *more urgent starving* job (`engine.rs:560-582`); aging (`engine.rs:408-443`) bounds every wait |
| Overrun | Impossible (walltime kill) | Possible, and normal: estimates are predictions, nothing is killed |

So our policy is **EASY with T_shadow = ∞**. Without runtimes we cannot tell whether a candidate ends before the holder can start, so we forbid everything on the reserved worker. RESULTS.md puts the cost of that at **0.92–0.98% of slot time idled while draining** (open loop). It is also why more reservations "only add idle time" (RESULTS.md headline 4; "Reservation sweeps").

---

## 2. What to extract into `sched`, and what not to

### 2.1 Extract

**E1. Runtime estimates in the placement core.** Required by everything below.
- Add `JobSpec::runtime_estimate: Option<f64>`: seconds on a speed-1.0 worker, meant as an upper-ish estimate.
- Add `WorkerState::speed: f64` (default 1.0). An estimated run time on worker w is `runtime_estimate / speed_w`.
- The `speed` field is also what LITERATURE.md §9 items 1 and 3 (EFT placement) need, so it is not wasted.
- `JobSpec::new` and `WorkerState::new` set the defaults. Every struct literal in the crate already uses `..WorkerState::new(..)` (I grepped; there are no users outside `crates/sched`), so this is source-compatible inside the repo.
- Under `serde`, add `#[serde(default)]` so old snapshots still load. The `DagSnapshot` stores `JobSpec`s.

**E2. Track estimated ends for running jobs.**
- `Running` (`engine.rs:151-155`) gains `start: Instant` and `est_end: Option<Instant>` (`start + est/speed`).
- An **overrun policy** decides what an estimated end in the past means. Batsim has no equivalent, because it kills at the walltime; see §2.2.

**E3. EASY shadow-time backfill on reserved workers.** The core extraction, adapted to memory plus slots.

For the holder J of worker w (demand d_J), compute the shadow from the jobs running on w, ordered by `est_end` (ties broken by job id, for determinism). This is `easy_bf_fast.cpp:256-275` with "hosts" replaced by our admission rule:

```
T = first t in {now} ∪ {est_end of running jobs, ascending} such that
      count(t) < slots  and  (count(t) == 0  or  proj_used(t) + d_J <= budget)
proj_used(t) = max(reported_used − Σ_{ended by t} demand,  baseline + Σ_{running at t} demand)
extra_mem    = budget − proj_used(T) − d_J        (≥ 0, or 0 under the escape hatch)
extra_slots  = slots − count(T) − 1
```

If any running job on w has no usable estimate, then T = ∞ and the worker behaves as today (strict drain).

A non-holder B is admitted on reserved w iff:
- the admission rule admits B now, as today; and
- either `now + est_B/speed_w <= T` (it ends before the shadow, `easy_bf_fast.cpp:150`),
- or `d_B <= extra_mem and extra_slots >= 1` (it fits beside J, `:151`).

T and extra are recomputed from the running set on demand (≤ `slots` jobs, so cheap) instead of being decremented as at `:168-169`. A B accepted by the second rule stays in `Σ running at T`, so extra shrinks. **The rule is monotone in load**: T never moves earlier and extra never grows as jobs are placed. The priority invariant's argument (`engine.rs:186-190`) therefore still holds.

**E4. Choose the reserved worker by earliest shadow.** In `try_reserve`, when estimates exist, score workers by `(T_w, −headroom, !preferred, running, id)` instead of `(−headroom, …)` (`engine.rs:534-552`). This is EASY's "earliest start time" (`find_with_earliest_start_time`, `resource.py:625-651`; `findAllocFuture`, `easyBackfill.py:323-337`). With T = ∞ everywhere it reduces exactly to today's order.

**E5. A starvation safety valve against overruns** (no Batsim equivalent).
- Record `T0` when the reservation is made.
- Once `now > T0 + drain_grace`, stop shadow backfill on that worker and fall back to strict drain.
- This keeps a finite bound when estimates are wrong. Tsafrir et al. (LITERATURE.md §5) instead extend the estimate and re-plan; offer that as an overrun policy, but keep the valve.

**E6. Reservation depth.**
- Keep `max_reservations` as the depth k. With E3 a deeper reservation no longer idles a whole worker, so the sweep k ∈ {1, 2, 4} (LITERATURE.md §5(c), §9.5) becomes meaningful.
- **Add `reserve_after = 0` as the documented "EASY" setting.** It reserves for the head as soon as it is admitted nowhere.
- That is all of pybatsim's `reservation_depth` that maps cleanly. Theirs stacks the k reservations in time on one pool; with ours, each holder gets its own worker.

**E7. Conservative backfilling as a separate policy** (optional, later).
- Each waiting job, up to depth k, holds a planned `(worker, start)` in a per-worker capacity profile. That profile is the `Schedule` (`schedule.cpp`) generalised to (slots, mem) per worker.
- Re-plan in queue order on every completion (`conservative_bf.cpp:105-123`), and admit now only what fits without invalidating any plan.
- It needs estimates for everything. Jobs without one get a configured default.
- **It gives predicted start times for free.** Expose them as `explain()` text and a `PolicyStats` field, as Batsim does with `query_wait` (`conservative_bf.cpp:128-133`).

**E8. Simulator metrics.**
- **Bounded slowdown.** `bsld = max(1, (wait + run) / max(run, τ))`, τ = 10 s (the Feitelson convention, not from these repos), reported overall and for big jobs.
- **Stretch.** Batsim exports plain slowdown `turnaround / runtime` (`batsim/src/export.cpp:711-719`) and `stretch` in jobs.csv (`docs/output-jobs.rst`), with a runtime of 0 replaced by 1e-5 (`jobs_execution.cpp:215-219`). Report the mean and max of stretch, to match `schedule.json` (`docs/output-schedule.rst`).
- **Two new counters:**
  - shadow accuracy: holder start − T0;
  - jobs backfilled onto reserved workers.

**E9. Simulator estimate models.**
- `--estimates none|oracle|noisy:σ|quantile:q`, with work estimates drawn from `setup.work` (`run.rs:39`) times seeded log-normal noise.
- The fitted residual sd is 0.54 (RESULTS.md, "The fit is noisy"). So p80 is about ×1.57 of the median, which follows Mu'alem and Feitelson's "overestimates are fine".
- The worker speed comes from the model curve (`model.rs:19`).

### 2.2 Do NOT transfer

- **Rigid multi-host jobs** (`res` = number of hosts, held exclusively).
  - Our jobs occupy one slot on one worker and a memory amount there; they never span workers.
  - Their "free host count" is our (free slots, headroom) pair per worker, which does not pool across workers.
  - The counting arithmetic of `easy_bf_fast` transfers per worker (E3). The global pool does not.
  - Resource selectors and contiguity (`locality.cpp`, `utils.py:195-388`, the easyBackfill.py free spaces) are irrelevant.
- **Walltime kill and walltime-required rejection** (`jobs_execution.cpp:169, 197-201`; `easy_bf.cpp:55-60`).
  - Killing a Nassau task at its estimate destroys work, and our estimates are predictions with log-sd 0.54, not user promises.
  - The library must instead (a) accept jobs without estimates, as today's behaviour, and (b) survive overruns (E2/E5).
  - The same goes for `rjms_delay` padding (`json_workload.cpp:49`): if wanted, fold it into the estimate quantile.
- **Head-only reservation without takeover protection.**
  - batsched moves the reservation to whatever the new head is (`easy_bf.cpp:165-169`), so it starves under SJF or LCFS.
  - Our `try_reserve` takeover plus `age_limit` are strictly better. Keep them.
- **SJF backfill order** (`schedEasySjfBackfill.py:20-21`) and the other queue orders.
  - A backfill order that differs from the urgency order **violates our priority invariant**: B may be placed on w while a more urgent A is admitted there (`tests/invariants.rs:404-424`).
  - Callers who want SJF can already express it through `JobSpec::priority`, for example `priority = estimate bucket`.
  - batsched's `desc_*slowdown` orders are FCFS anyway (`queue.cpp:27-35`).
  - Recommendation: no new order types.
- **Energy and pstate schedulers, the protocol and ZMQ, and machine state events.** Our `worker_update` and `worker_gone` already cover joins and leaves. Batsim's `RESOURCE_STATE_CHANGED` and pstates are out of scope.
- **pybatsim's per-candidate re-planning cost model** (O(candidates × k) allocations per call). Too slow for our `dispatch` budget of µs to ms at 10⁴–10⁵ waiting jobs. Recompute only per reserved worker.

### 2.3 Expected value (to keep this honest)

- **The upside of E3/E4 on the production trace is small in throughput.** The thing it removes, draining idle, costs about 1% of slot time (RESULTS.md table: "idle draining" 0.92–0.98%).
- **The likelier gains:**
  - shorter waits for holders, because the reserved worker is the one that frees soonest rather than the one with the most headroom now;
  - deeper or per-class reservations without the 2–4% idle cost RESULTS.md measured.
- **Slots-only simulations gain nothing.** With unit demands and no memory, which is the whole-run sim (`sched-whole`, RESULTS.md "Memory is not modelled here"), every job fits any free slot, so no reservation is ever needed and EASY does nothing. Backfilling only matters under memory admission.

### 2.4 Engine vs simulator

**Engine (pure, deterministic, caller's `now`):**
- estimate fields;
- `est_end` bookkeeping;
- shadow/extra computation;
- the reserved-worker backfill rule;
- earliest-shadow reservation choice;
- the overrun policy and drain valve;
- (later) the conservative profile.

**Simulator only:**
- where estimates come from (noise, quantile);
- bounded slowdown and stretch;
- shadow-accuracy metrics;
- the Batsim export/import tooling;
- the oracle comparison harness.

The production coordinator would supply estimates from the census cost model (RESULTS.md "Costs").

---

## 3. Validation plan: same instance, their policy vs ours

### 3.1 The exact-equivalence instance: one worker, memory as hosts

On a single worker with:
- `reported_used = reported_baseline = 0`;
- `slots ≥ budget units`, so slots never bind;
- every demand an integer `d ≤ B` units,

our admission rule `running == 0 or placed + d ≤ B` (`admission.rs:70-75`) is **exactly** Batsim's "d ≤ free hosts" with B hosts. The escape hatch never fires a different decision, because running == 0 implies placed == 0.

Then the new EASY mode should match `easy_bf_fast` **decision for decision**, given:
- `PriorityBackfill` with `reserve_after = 0` and `max_reservations = 1`;
- each job in its own group, in increasing order, so the order is FCFS by seq;
- exact estimates equal to the walltime;
- no aging.

Why they should agree:
- Both start queued jobs in order until the first that does not fit, which becomes the holder or priority job (`easy_bf_fast.cpp:102-133` vs `engine.rs:615-652`).
- Both compute the shadow from estimated ends (`:256-275` vs E3).
- Both admit a later job iff it fits now and (it ends by the shadow, or fits in the extra) (`:149-151, 205-207`).

**Plan A, a pure-Rust oracle. Do this; no external dependencies.**
- Port `easy_bf_fast` (about 120 lines of logic) into `tests/easy_oracle.rs` as a reference implementation over hosts.
- Drive both with the same fixed-duration event loop. Do **not** use sched-sim's processor-sharing model.
- Process every event at a timestamp, then call `dispatch` once, mirroring batsched's batching (`isalgorithm.hpp:156-158`).
- proptest over random workloads:
  - B ∈ [4, 64];
  - d ∈ [1, B];
  - durations 1–200 s;
  - Poisson arrivals.
- Assert identical `(job, start time)` sequences.
- **Expected discrepancies, which the test must neutralise:**
  - tie order at equal timestamps: completions before releases in both; releases in id order;
  - `<=` vs `<` at the shadow: both use `<=`;
  - f64 rounding: use integer-valued times.
- A second oracle port of `conservative_bf` plus a `Schedule` profile (about 300 lines) validates the conservative policy (E7) the same way. Note that `easy_bf` (the 2-D variant) is **not** expected to match `easy_bf_fast`, by its own comment (`easy_bf_fast.cpp:52-56`). Compare against `_fast`.

**Plan B: actual Batsim, optional, environment-dependent.**
- Write `sched-sim export-batsim`. For the single-worker instances above it emits:
  - `workload.json`: `nb_res = B`; jobs `{id: zero-padded, subtime, res: d, walltime: est, profile}` with a `DelayProfile` whose delay is the true runtime (format in `docs/input-workload.rst`, example `workloads/example_workload_hpc_seed1_jobs250.json`);
  - a SimGrid platform with B compute hosts plus a master (`docs/input-platform.rst`).
- Run batsched `-v easy_bf_fast -o fcfs -d 0` (`main.cpp:104-110`; `-d 0` turns off walltime padding) against a **Batsim v4.x** binary. batsched master does not speak batsim master's batprotocol (§Sources).
- Compare `out/jobs.csv` `starting_time` per job against our placements, and `schedule.json` `mean_waiting_time`, `max_waiting_time` and `mean_slowdown`.
- **Expected discrepancies:**
  - batsched sorts FCFS ties by id *string* (`queue.cpp:47-48`), so ids must be zero-padded;
  - its `rjms_delay` default is 5 s;
  - Batsim's own decision delay, if configured;
  - floats vs batsched's `Rational`.
- **Feasibility on this HPC is uncertain.** Nix on panfs fails (memory note), the EL7 glibc is old, and SimGrid has to be built. Run it on another machine or in a container. Not verified.

### 3.2 On our real trace: a qualitative reference only

The trace cannot be mapped exactly:
- there are 21 workers with separate budgets, and Batsim pools hosts;
- durations follow processor sharing, not fixed delays;
- reported RSS is exogenous.

Two useful approximations:
1. **"Pooled" Batsim run.**
   - Hosts = Σ budgets in GB-units; jobs `res = ceil(est_gb)`, `walltime = estimate`, `delay = observed run time`; `easy_bf_fast` and `conservative_bf`.
   - This is a no-fragmentation, no-RSS reference.
   - Expected: much shorter waits than any of our policies. The gap measures the **cost of per-worker fragmentation plus the RSS term**, which is itself a useful number. It is not a correctness check.
2. **Our own sched-sim with E8/E9.**
   - Compare `PriorityBackfill` in today's mode against EASY mode (`reserve_after` 0 and 60) under estimates {none, oracle, noisy σ = 0.54, quantile p80}, at depth {1, 2, 4}.
   - Metrics: wait p50/p99/max, bsld, idle draining, shadow accuracy.
   - Expectations:
     - idle draining should fall from about 1% toward 0 with oracle estimates;
     - with noisy estimates, holder delays appear and the valve (E5) triggers;
     - makespan in the open loop does not move (RESULTS.md: open-loop makespan is fixed by the arrivals).

---

## 4. Implementation plan

Paths are relative to `ext/crates/sched/`. Effort is in focused engineer-days.

| # | Step | Files | Effort |
|---|---|---|---|
| 1 | Simulator metrics: bounded slowdown and stretch | `src/sim/run.rs` (`Metrics` :110, `summarize` :583, wait at :627), `src/bin` printing | 0.5 d |
| 2 | Estimate fields in the API | `src/lib.rs` (`JobSpec` :121-156, `WorkerState` :161-188), `src/dag.rs` snapshot defaults | 0.5 d |
| 3 | Running `est_end`; shadow computation; overrun policy | `src/engine.rs` (`Running` :151, `place` :473, new `fn shadow(&self, w, holder) -> Shadow`) | 1 d |
| 4 | Backfill on reserved workers, earliest-shadow reservation, drain valve | `src/engine.rs` (`refusal` :384-406, `try_reserve` :518-583, `explain` :694-766, `BackfillConfig` :19-50) | 1.5 d |
| 5 | Tests: unit, invariants, starvation, determinism | `tests/invariants.rs`, `tests/starvation.rs`, `tests/reservation.rs`, new `tests/easy.rs` | 2 d |
| 6 | Differential oracle against `easy_bf_fast` | new `tests/easy_oracle.rs` | 1–1.5 d |
| 7 | Simulator estimate models and sweeps; RESULTS section | `src/sim/run.rs` (`spec` :245, `state` :396-420), `src/bin/sched-sim`, `RESULTS.md` | 1.5 d |
| 8 | (Optional) `Conservative` policy with per-worker profiles plus its oracle | new `src/profile.rs`, `src/engine.rs` or a new policy module, `tests/conservative_oracle.rs` | 4–5 d |
| 9 | (Optional) Batsim export and a v4 run | `src/sim/batsim.rs`, a CLI flag | 1 d, plus environment (uncertain) |

### 4.1 Public API (proposed signatures)

```rust
// lib.rs
pub struct JobSpec {
    // ... existing fields ...
    /// Estimated run time in seconds on a worker of speed 1.0 (ideally an upper quantile).
    /// `None`: unknown; the job is never backfilled ahead of a reservation's shadow time.
    #[cfg_attr(feature = "serde", serde(default))]
    pub runtime_estimate: Option<f64>,
}
pub struct WorkerState {
    // ... existing fields ...
    /// Relative speed: a job's run time here is `runtime_estimate / speed`. Default 1.0.
    #[cfg_attr(feature = "serde", serde(default = "one"))]
    pub speed: f64,
}
pub struct ReservationInfo {           // gains:
    pub job: JobId, pub worker: WorkerId, pub since: Instant,
    /// Estimated time at which the holder becomes admissible (`None`: unknown, strict drain).
    pub shadow: Option<Instant>,
}

// engine.rs
/// What a running job's estimated end means once it has passed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Overrun {
    /// Treat the end as unknown: the worker falls back to strict drain (default; safest).
    Unknown,
    /// Extend the estimate to `elapsed * factor` (Tsafrir et al.); factor > 1.
    Extend { factor: f64 },
}

pub struct BackfillConfig {
    // existing: reserve_after, max_reservations, per_class_reservations, default_priority, age_limit
    /// Admit non-holders on a reserved worker when they end before the holder's shadow time, or
    /// fit beside it (EASY backfilling). Needs estimates; without them the worker drains.
    /// Default false (today's behaviour).
    pub shadow_backfill: bool,
    /// Once a reservation is this far past the shadow time computed when it was made, its
    /// worker stops shadow-backfilling and drains. Default 600 s.
    pub drain_grace: f64,
    /// Interpretation of overrun estimates. Default `Overrun::Unknown`.
    pub overrun: Overrun,
}
```

- `PolicyStats` gains `shadow_backfills_total: u64`.
- An "EASY" preset could be offered as `BackfillConfig::easy()`, which sets `reserve_after = 0.0` and `shadow_backfill = true`.
- (Step 8) `pub struct Conservative<A = ProductionAdmission>` with `ConservativeConfig { depth: usize, default_estimate: f64, backfill: BackfillConfig }`, plus `PolicyStats::predicted_start: Vec<(JobId, WorkerId, Instant)>`.

### 4.2 Engine details (step 3/4)

- `fn shadow(&self, w: &Worker, holder: &Waiting) -> Option<(Instant, Resources, usize)>`.
  - Collect `(est_end_eff, demand)` for `w.jobs` and sort with `total_cmp`, ties by job id.
  - Walk them per the formula in §2.1 E3. Return `None` if any `est_end_eff` is unknown.
  - Cost is O(slots log slots) per call. It is called only when refusing or admitting on reserved workers, so dispatch stays in µs.
- `refusal`: replace the unconditional `ReservedFor(holder)` (`engine.rs:388-392`) with:
  - if `shadow_backfill` and the reservation is not past its valve, and a shadow exists, admit when admission holds and (`ends_by` or `fits_extra`);
  - otherwise `ReservedFor`.
  - Add a `Refusal::Shadow(holder)` variant for `explain`.
- `open_bound` (`engine.rs:587-598`) currently skips reserved workers entirely. It must include a reserved worker's `extra_mem` bound, or keep skipping only when `shadow_backfill` is off. Otherwise `hopeful` (`engine.rs:627`) filters out jobs that could backfill there. The short-job case cannot be bounded by memory alone, so return the admission bound when `shadow_backfill` is on.
- `try_reserve`: score `(shadow_or_inf, -headroom, !preferred, running, id)`. Store `T0` in `reservations` (the tuple at `engine.rs:203` gets a field).
- **Determinism.** Use only f64 ordering via `total_cmp`, BTreeMap iteration and explicit id tie-breaks. Never use HashMap iteration order.

### 4.3 Tests (step 5/6). Every existing invariant must hold.

**No over-commit.** Unchanged: shadow backfill only *narrows* the set of refused jobs on reserved workers, and admission is still checked. `invariants.rs`'s `sh.admits` (`:184-192`) continues to apply.

**Escape hatch.** Unchanged (`invariants.rs:430-440`). An empty reserved worker still admits its holder. Add a case: a reserved worker that is empty but has its holder blocked by `avoid` must still be released. That is today's behaviour; re-verify it.

**Priority.**
- Extend the Shadow model in `invariants.rs` with estimates (an `Op::Submit` with an optional `est`, and `Op::Worker` with a speed).
- Add an independent re-implementation of the shadow and extra rule (`fn reserved_admits`).
- The priority check (`:404-424`) then accepts "refused" when A fails `reserved_admits` on a reserved w.
- The policy's `stats().reservations` must expose `shadow` so the test can check it against the shadow model.
- Add `Kind::Backfill { shadow: bool, overrun }` to the `kind()` strategy.

**No starvation.**
- (a) Existing tests stay as they are: no estimates means today's path.
- (b) New proptest: exact estimates, with the bound unchanged, `wait ≤ reserve_after + D + 2·tick` (`starvation.rs:124-127`).
- (c) Overrun proptest: estimates = true × U[0.5, 2] with `drain_grace = G`. Bound: `wait ≤ reserve_after + (T0 − t_res) + G + 2·D + 2·tick`.
  - The second D covers a job backfilled just before the valve.
  - **Mark (c) as the bound to prove carefully.** Under `Overrun::Extend`, T can move later; the valve uses `T0`, not the current T, precisely to keep this finite.

**Determinism.** The existing double run (`invariants.rs:452-460`) covers it once `kind()` includes the new options.

**Reservation edge cases** (`reservation.rs`):
- a holder whose worker has an overrunning job falls back to drain;
- a heartbeat raising `reported_used` makes `proj_used` refuse extra backfills;
- `worker_update` that changes `speed` changes est_end only for later placements. Decide whether running jobs keep their est_end; recommended: keep it, documented.

**Oracle** (step 6): §3.1 Plan A.

### 4.4 Risks

- **`reported_used` is exogenous.** E3 projects RSS by subtracting finished jobs' demands. If real RSS does not drop (RESULTS.md: RSS "barely tracks the running count"), J can be refused at T while "extra" jobs keep the worker busy. **The valve (E5) is what keeps the bound.** Consider a stricter projection, `proj_used(T) = max(reported_used, …)` with no subtraction, as a config option, and measure both.
- **The estimate quality is poor.** Within-group R² is 0.37. Expect frequent overruns at p50; use the p80 quantile (LITERATURE.md §5(a)) and measure shadow accuracy (E8).
- **`open_bound`/`hopeful` interaction** (§4.2). Getting this wrong silently disables backfill. The oracle test catches it.
- **A larger proof surface for the priority invariant.** The monotonicity argument (§2.1 E3) must be written into the `Engine` doc (`engine.rs:177-191`), and the restart-on-release logic (`engine.rs:631-635`) re-checked: a holder placed on its worker re-opens it fully.
- **The conservative policy (step 8) can blow the dispatch budget.** It needs (worker × time) profiles of up to `slots` events each, with a depth cap. Uncapped `k = ∞` over 10⁴ waiting jobs per event is too slow; benchmark before shipping.
- **Batsim Plan B** may be infeasible on this HPC (old glibc, nix on panfs). Plan A is self-sufficient.

### 4.5 Uncertainties

- Which Batsim release batsched master's JSON protocol matches is inferred from batsched's CHANGELOG link targets, not tested.
- I did not read `filling.py`, `alloc.py` beyond the end-time property, or the pybatsim v3 `Job` model.
- I did not read batsched's `energy_bf*`, `easy_bf_plot_liquid_load_horizon` or `queueing_theory_waiting_time_estimator`. They do not look relevant to placement.
- τ = 10 s for bounded slowdown is the literature convention. These repos do not compute bounded slowdown in their outputs; batsched only uses a degenerate version of it as a queue order.
