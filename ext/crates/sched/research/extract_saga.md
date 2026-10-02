# SAGA + PISA, read against `sched`

Source: `https://github.com/ANRGUSC/saga`, shallow clone at
`/wsu/home/hd/hd72/hd7264/.claude/jobs/71ebdf0b/tmp/refs/saga`, commit `5b20e8d31a` (2026-08-25).
SAGA paths below are relative to `src/saga/` unless they start with `scripts/`. `sched` paths are
relative to `ext/crates/sched/` in the `sched-crate` worktree.

Everything below comes from reading the code. Nothing was executed: this node has only Python 3.7,
and SAGA needs 3.12 or later (`README.md`, "Python Version"). Uncertain points are marked **[U]**.
Places where SAGA's code departs from the published algorithm are marked **[DEV]**. Most of those
judgements rely on my memory of the papers and should be checked against them.

---

## 1. Problem model: theirs vs ours

### 1.1 SAGA's model

**Network** (`__init__.py:19-124`)
- **Nodes:** each node has a scalar `speed` (`:25`).
- **Links:** every pair of nodes has a link with `speed` (bandwidth, `:41`).
- **Fully connected:** `Network.create` fills in every missing pair (`:110-114`).
  - A missing link between two distinct nodes defaults to speed **0.0**; a self-loop defaults to `inf` (`:96, :113`).
- **Machine model:** related machines. Execution time is `task.cost / node.speed` (`:817`). There is no per-task-per-node cost matrix.

**Task graph** (`:280-412`)
- **Tasks:** each has a `cost` (`:286`).
- **Edges:** each has a data `size` (`:302`).
- **Single source and sink:** `TaskGraph.create` adds a zero-cost `__super_source__` / `__super_sink__` when there are several sources or sinks (`:357-410`).
- **Communication:** `size / link.speed`. It is zero when both tasks run on the same node, because self-loops have `inf` speed (`:826`).

**Schedule** (`:597-937`)
- **Capacity:** a node runs at most one task at a time. `add_task` raises on overlap (`:885-891`).
- **Start time:** `get_earliest_start_time` (`:790-853`) gives the earliest start, with gap insertion or append-only.
- **Objectives:** makespan is the latest end (`:656-664`). Throughput is 1/bottleneck load (`:709-731`).
- **Constraints:** optional per-task node constraints (`:607-614`).

**Scheduler API** (`:940-965`)
- **Signature:** `schedule(network, task_graph, schedule=None, min_start_time=0) -> Schedule`.
- **Offline:** the whole DAG and all costs are known up front, and the result is a static timetable.

**Online layer** (`schedulers/online/`)
- **Environment:** an event loop that plays a schedule forward and lets a policy revise it at each event (`online/environment/__init__.py:113-280`).
- **Frontier variant:** `FrontierEnvironment` (`online/environment/frontier.py:20-136`) holds a ready frontier and commits tasks only as they become ready. This is the closest thing in SAGA to our engine.

**Stochastic layer** (`stochastic.py`, `schedulers/stochastic/`)
- **Model:** costs and speeds are random variables.
- **Schedulers:** they determinise each variable to an estimate (mean, or mean+std), then run a deterministic scheduler (`schedulers/stochastic/estimate_stochastic_scheduler.py:47, :87`).

### 1.2 Differences that matter

| aspect | SAGA | `sched` | consequence |
|---|---|---|---|
| time model | static, offline: full DAG and exact costs known at t=0; output is a timetable | online and event driven: `dispatch(now)` after every event (`src/lib.rs:245-268`); the DAG is declared incrementally and lazily expanded per group (`DagTemplate`, `src/dag.rs:63-200`; `instantiate` in `src/sim/whole.rs:1043-1078`) | HEFT-family "insertion into future gaps" has no direct online analogue; only greedy or "wait for a slot" placement does |
| machines | one task per node; related speeds | worker = `slots` × processor sharing: `f(k)=speed·min(k,k_sat)^α`, split equally (`src/sim/model.rs:40-60`, `src/sim/whole.rs:1217-1258`) | with α=1 and k ≤ k_sat, a worker is exactly `slots` independent machines of that speed (each job runs at `speed`). **Caveat:** the fitted H200 has `k_sat=15` with 16 slots (`RESULTS.md`, service model), so its 16th job slows everyone by 15/16. A cross-check needs k_sat = slots. |
| comm costs | first class (`size/link.speed`) | none | set every `size=0`. **Trap:** a link left at the default speed 0.0 gives `0/0` = Python `ZeroDivisionError` (`__init__.py:684, :826`; also `cpop.py:33`, `peft.py:51-54`), so links must be given speed > 0 explicitly |
| memory / admission | none (only optional node constraints) | `Admission` (`ProductionAdmission`), reservations, backfill, aging (`src/engine.rs:384-660`) | the whole-run sim already turns memory off (`Resources::mem(1<<60)`, `src/sim/whole.rs:965`), so SAGA and `sched-whole` agree there |
| costs at decision time | exact (deterministic schedulers) or a determinised estimate (stochastic ones) | estimate for ranks (`DagJob::with_work`, `update_work`), truth for execution (`World::sig_work(.., truth)`, `src/sim/whole.rs:663-681`) | this is SAGA's "MeanHEFT on the estimate, executed on the sample" (§2.13) |
| priority | per-algorithm rank | `Key {priority, group, seq}` (`src/engine.rs:116-121`): explicit priority (DAG rank: `-(rank·1000)`, `src/dag.rs:596-608`), then group first arrival, then FIFO; aging prefix (`:409-444`) | SAGA has nothing like group-first-arrival ordering, except the online FIFO frontier, which keys on *task* arrival time (`frontier.py:51-53`) |
| worker choice | per algorithm: EFT, EST, fastest, least-loaded… | `choose`: lane rank, then fit, then preferred, then fewest running, then id (`src/engine.rs:446-470`). "Fast first" = `prefer` = all fastest-class workers (`src/sim/whole.rs:978-993`) | there is no EFT or speed-aware choice in the engine: `WorkerState` has no speed and `JobSpec` has no work field (`src/lib.rs:121-174`) |
| size | tens to hundreds of tasks (`scripts/experiments/benchmarking/prepare.py`); PISA uses 3–5 tasks | 3.93M tasks, 81k groups | SAGA's code is O(V²·P) Python and cannot run our instance; it can run only small exports |

---

## 2. The schedulers: exact algorithm, complexity, relevance to us

"Our setting" means: online, lazily expanded, span-bound (`D = 883 h > W/P = 770 h`), two speed classes
(L40S ≈ 2.41× H200 per job), 16 slots per worker, zero communication.

- **Complexities** are for SAGA's code. V = tasks, E = edges, P = nodes, R = ready-set size.
- **Zero communication** collapses most comm-aware heuristics into simpler ones. Each entry says how.

### 2.1 HEFT (`schedulers/heft.py`, ranks in `schedulers/cpop.py:9-43`)

**Algorithm.**
- **Upward rank** (`cpop.py:20-41`):
  - In reverse topological order, `rank(t) = mean_p(cost_t/speed_p) + max_{children c}(rank(c) + mean_{all network edges}(size/edge.speed))`.
  - The comm mean runs over *all* edges, self-loops included. **[DEV, minor]**
- **Sort** (`heft.py:11-27`): by `(rank, index in reversed topological order)` descending.
- **Placement** (`:61-90`): for each task in that order, over all nodes, compute the insertion-based earliest start (`get_earliest_start_time(append_only=False)`). Finish = start + cost/speed. Take the strict minimum finish; the first node in `frozenset` order wins ties.

**Complexity.**
- Ranks: O(E·P²) (comm mean over P² edges per dependency).
- Placement: O(V·P·(V + indeg)) because insertion scans the node's task list.
- The paper's figure is O(E·P).

**Relevance to us.**
- **Order:** with zero comm and related machines, `rank = cost·mean(1/speed) + max child rank`. That is our upward rank (`src/dag.rs:290, :868-871`) times a constant, so the order is identical. We already ran it ("rank-oracle") and it loses to group order.
- **Placement:** EFT with insertion is what we lack.
  - **Online meaning:** finish = max(now, earliest time a slot of worker w is expected free) + work/speed_w. This lets a job wait for a fast slot.
  - **Mapping:** this is `LITERATURE.md` §9 item 3. The insertion part (back-filling *future* gaps) has no online meaning.
- **Verdict:** add the EFT half only. **Rank #1** (§2.17).

### 2.2 CPOP (`schedulers/cpop.py:46-211`)

**Algorithm.**
- **Downward rank** (`:56-78`): `down(t) = max_{parents p}(down(p) + mean comm + cost_t/mean(speed))`.
  - **[DEV]** In the paper, down(t) adds the *parent's* cost. This code adds the task's own cost, using the mean speed.
- **Priority** (`:93-98`): `up + down`.
- **Critical-path rank** (`:144`): `cp_rank` = the largest priority among entry tasks.
- **CP node** (`:148-155`): the node minimising Σ cost/speed over tasks whose priority is `isclose` to `cp_rank`. For related machines this is the fastest node.
- **Loop** (`:158-210`): a max-heap of ready tasks. A critical task (`isclose(prio, cp_rank)`) may use only the CP node (`:176-177`). Every other task takes the min-EFT node with insertion.

**Complexity:** as HEFT, plus O(V log V) for the heap.

**Relevance to us.**
- **Idea:** pin critical-path tasks to the fast class; let others go anywhere by EFT or fast-first.
  - This is `LITERATURE.md` §9 item 1 (critical-task speed-class assignment).
  - With 16 slots per worker and 14 fast workers, "the CP node" becomes "the fast class".
  - Pinning to one slot would be harmful.
- **Lazy expansion:** `down` needs predecessors' costs. Online, use the *actual* ready time instead: `now + bl(n)/s_fast` against a running estimate of the makespan, which is slack.
- **Verdict:** **Rank #3**, as a DAG-layer option (§5, step S4).

### 2.3 PEFT (`schedulers/peft.py`)

**Algorithm.**
- **OCT table** (`:6-65`), in reverse topological order: `OCT(t,p) = max_{children c} min_{p'}(OCT(c,p') + cost_c/speed_{p'} + [p≠p']·size/link(p,p'))`, and 0 for sinks.
  - Comment `:12-20`: this uses the *actual* link cost rather than the paper's average.
- **Rank** (`:81-84`): mean over p of `OCT(t,p)`. Sorted with a reverse-topological tie-break (`:85-92`).
- **Loop** (`:177-201`): repeatedly take the ready task with the largest rank, and place it on argmin_p of (insertion EFT + OCT(t,p)) (`:121-136`).
- The `schedule` argument is ignored: it is overwritten at `:169`.

**Complexity:** O(E·P²) for the OCT; O(V·(R + P·V)) for the loop.

**Relevance to us: degenerate.**
- **Zero comm:** OCT(t,p) is independent of p. It equals the longest downstream chain at the fastest speed, excluding t's own cost.
- **Placement:** minimising EFT + OCT is then plain EFT.
- **Order:** "upward rank at fastest speed minus own cost", which is nearly our rank.
- **Verdict:** no new plan. At most, try rank − own work as an ordering variant. **Low.**

### 2.4 MinMin (`schedulers/minmin.py`), MaxMin (`schedulers/maxmin.py`), Duplex (`schedulers/duplex.py`)

These are batch-mode heuristics for independent tasks (Braun et al.), run level by level.

**MinMin.**
- **Rounds** (`minmin.py:96-123`): each outer round collects `available_tasks`, those whose parents are all scheduled. Children enabled during a round wait for the next round.
- **Inner loop:** repeatedly pick the (task, node) pair with minimum ECT, append-only.
  - `ECT = cost/speed + max(EAT(node), FAT(task,node))`, where EAT = the node's last end and FAT = the latest input arrival.
  - Schedule it, then clear the caches.

**MaxMin.**
- **Selection** (`maxmin.py:104-115`): the same, but choose the task whose *minimum* ECT is largest, then its argmin node.

**Duplex.**
- **Selection** (`duplex.py:29-38`): run both and keep the shorter makespan.

**Complexity:** O(V·R·P) with per-pick cache clears, i.e. O(V²·P) worst case.

**Relevance to us.**
- **MinMin:** in the online two-class setting it reduces to shortest-job-first onto the fastest free slot. In a span-bound DAG that is anti-critical-path, so no.
- **MaxMin:** reduces to longest-job-first (LPT). Within a group it is a crude stand-in for rank.
- **Verdict:** no global plan. A "within-group longest first" tie-break is cheap but dominated by "within-group rank" (§2.17 #4). **Low.**

### 2.5 ETF (`schedulers/etf.py`)

**Algorithm.** An event-driven simulation (`:125-178`). At `current_moment`:
- **Ready nodes:** nodes idle by then.
- **Pairing:** for each ready task, the idle node with the earliest data arrival (`_get_start_times`, `:10-57`). Execution speed is **ignored**. Ties fall to set iteration order.
- **Commit:** the globally earliest-start task, if it starts at or before `next_moment`. Repeat while tasks and idle nodes remain, then advance to the next completion.
- **[DEV]** Original ETF breaks ties by static level; this code does not.

**Complexity:** O(V·R·P·indeg) overall.

**Relevance to us.**
- **Equivalence:** this *is* our speed-oblivious greedy (group order, no "fast first") with an arbitrary tie-break. It adds no new plan.
- **Use:** a cross-check target for event-driven list scheduling (§4).
- PISA treats ETF as needing homogeneous compute (`scripts/experiments/pisa/run.py:36`).

### 2.6 BIL (`schedulers/bil.py`)

**Algorithm.**
- **BIL table** (`:47-64`), in reverse topological order: `BIL(t,p) = cost/speed_p + max_c min(BIL(c,p), min_{p'} BIL(c,p') + size/link(p',p))`.
- **Task selection** (`:74-111`):
  - For each ready task, sort its `BIM(t,p) = EAT(p) + BIL(t,p)` values *descending*.
  - Pick the task with the largest j-th value, starting at j=0; on a tie, go to j+1.
- **Node selection** (`:116-157`):
  - Revised `BIM* = BIM + cost/speed_p · max(|ready|/|V| − 1, 0)`, then argmin. Ties go to the node maximising the others' sum.
  - **[DEV]** In the paper the factor is ready-count over *processor count* (k/m − 1). Here it is `len(ready_tasks)/len(tasks)` (`:127`), which is ≤ 1, so the term is always 0. Also, `ready_tasks` is never pruned of scheduled tasks.
- **Start** (`:162-181`): append-only.

**Complexity:** O(E·P²) for the table; O(V·R·P log P) for the loop.

**Relevance to us: degenerate.**
- **Zero comm:** `BIL(t,p) = cost/speed_p + (downstream critical path at fastest speed)`.
- **Result:** task selection is rank-like, and node selection is append-only EFT. So BIL is HEFT order + EFT placement without insertion: nothing beyond §2.1.
- **Verdict:** **Low.**

### 2.7 FastestNode (`schedulers/fastest_node.py`)

**Algorithm** (`:55-83`): in topological order, put every task on the fastest *allowed* node, append-only. Without constraints, everything runs serially on one node.

**Complexity:** O(V·P + E).

**Relevance to us.**
- **Online analogue:** "fast class only", a hard `JobSpec.class = Some(fastest)` (`src/lib.rs:140`; enforced in `eligible`, `src/engine.rs:158-160`).
- **Why it matters:** it is a cheap and informative baseline in a span-bound run.
  - **Bound:** fast-class capacity is 14 × 38.61 = 540.5 H200-units versus 645.5 for the whole fleet (f(16) values in `RESULTS.md`).
  - So fast-only W/P is about 770 × 645.5/540.5 ≈ **920 h**, against D = 883 h and today's best plan at 1,237 h.
  - Whether the slow class adds anything at all is unmeasured.
- **Verdict:** **Rank #2**, for its cost-to-information ratio. It also gives the reference that "EFT with waiting" should approach.

### 2.8 GDL / DLS (`schedulers/gdl.py`)

**Algorithm.**
- **Static level** (`:81-92`): median execution time + max over children of (SL(child) + `out_edge.size`). Raw size, not divided by link speed: **[DEV]**.
- **Speed advantage** (`:73-78`): `Δ(t,p) = median of all exec times − exec(t,p)`.
- **DL1** (`:122-127`): `SL(t) − max(data_available, node_available) + Δ(t,p)`.
- **DL2** (`:155-168`): adds a descendant term.
- **Selection** (`:174-191, :217`):
  - `preferred_node = argmin_p DL`; `cost = DL(pref) − max_p DL`; `GDL = DL(pref) + cost`.
  - The task with the *minimum* GDL is scheduled, append-only.
- **[DEV/likely bug]** Sih & Lee select the pair with the *largest* dynamic level. As far as I recall, the generalized form takes the best node by max DL and adds DL(best) − DL(second best). With `min`, SAGA picks the lowest-level pair. Verify against the paper before relying on its GDL numbers. **[U]**

**Complexity:** O(V·R·P·(indeg + P)).

**Relevance to us.**
- **Valid idea:** "level − start + speed advantage".
  - **When a fast slot frees,** give it to the ready job that gains most from speed and has the highest level: `bl + w·(1/s_slow − 1/s_fast)`.
  - **Contrast:** fast-first gives it to the first job in key order, not the one that gains most.
- **Engine fit:** our engine scans job-major (`src/engine.rs:601-656`), while DLS is pair-major.
- **Verdict:** **Rank #5**. Only as a variant of the EFT work, a "fast-slot key".

### 2.9 MCT (`schedulers/mct.py`)

**Algorithm** (`:75-90`): in (networkx) topological order, put each task on the node with the minimum completion time `max(EAT, FAT) + cost/speed`, append-only.

**Complexity:** O((V+E)·P).

**Relevance to us.**
- **Equivalence:** this is "arrival order + EFT placement", the cleanest online analogue of the placement lever.
- **Plan mapping:** `group+eft` once EFT exists. It is the same mechanism as HEFT's half, so it is covered by §2.1.

### 2.10 MET (`schedulers/met.py`)

**Algorithm** (`:68-83`): in topological order, put each task on argmin exec time, ignoring load; append-only. For related machines this sends everything to the single fastest node (ties go to the first in list order).

**Complexity:** O(V·P + E).

**Relevance to us:** identical to FastestNode for related machines; covered by §2.7.

### 2.11 OLB (`schedulers/olb.py`)

**Algorithm** (`:44-85`): in topological order, put each task on the node that becomes available earliest, ignoring speed; append-only.

**Complexity:** O(V·P + E).

**Relevance to us:** our least-loaded, speed-oblivious greedy, i.e. "group" without "+fast". It adds nothing; use it for cross-checks only.

### 2.12 Sufferage (`schedulers/sufferage.py`)

**Algorithm** (`:76-107`):
- **Score:** each round, compute every ready task's ECT on every node (append-only). sufferage = second-best ECT − best.
- **Pick:** schedule the task with the largest sufferage on its best node.
- **Ready set:** recomputed after each pick. Unlike MinMin, there are no level rounds.

**Complexity:** O(V·R·P·indeg).

**Relevance to us.**
- **Two classes, both free:** sufferage ∝ w·(1/s_slow − 1/s_fast), so the largest job goes to the fast slot first.
- **Busy fast slots:** sufferage becomes "how much worse is the slow slot than waiting".
- **Verdict:** only the "fast-slot key" idea of §2.8, folded into the EFT work (§2.17 #5). **Low** standalone.

### 2.13 WBA (`schedulers/wba.py`)

**Algorithm** (`:109-152`):
- **Rounds:** level rounds as in MinMin.
- **Pick:** for every ready (task, node) pair, the makespan increase is `max(ECT − cur_makespan, 0)`. Choose uniformly at random (`random.choice`, `:140`) among pairs with increase ≤ i_min + α·(i_max − i_min), α = 0.5. Append-only.

**Complexity:** O(V·R·P).

**Relevance to us.**
- **Randomized:** it cannot be a production plan, and it makes PISA's energy noisy.
- **Use:** as a *robustness probe*. Randomize ties in our key and measure the spread. It is a GRASP-style check of how much of a plan's makespan is luck.
- **Verdict:** **Low.**

### 2.14 FLB (`schedulers/flb.py`)

**Algorithm** (`:93-282`): Radulescu & van Gemund's Fast Load Balancing.
- **Enabling processor (EP):** the node from which the task's last message arrives (`:47-56`, using average comm speed).
- **Queues:**
  - Tasks whose `LMT ≥ PRT(EP)` go into the EP's EMT- and LMT-keyed heaps.
  - The others go into a global non-EP heap keyed by LMT.
- **Each step** (`:113-198`): compare (head EP task on the head active processor) with (head non-EP task on the earliest-idle processor). Schedule whichever has the smaller EST. Then update the lists (`:200-275`).
- **Notes:**
  - **[DEV]** The docstring (`:13-15`) says it "schedules to the fastest node whenever the original schedules to an arbitrary node". The code orders `all_procs` by `(PRT, name)`, not by speed.
  - **[U, possible bug]** It rebuilds `PriorityQueue.queue` lists by filtering (`:173-177, :213-217, :225-235, :268-273`) without re-heapifying, which can break the heap invariant.
  - PISA treats FLB as needing homogeneous compute and communication (`scripts/experiments/pisa/run.py:36-38`).

**Complexity:** O(V(log V + log P) + E) by design.

**Relevance to us.**
- **Zero comm:** the EP distinction vanishes and FLB becomes ETF with heaps.
- **Verdict:** no.

### 2.15 Stochastic variants (`schedulers/stochastic/`)

**Algorithms.**
- **MeanHEFT** (`mean_heft.py:9`): HEFT on mean costs.
- **SHEFT** (`sheft.py:10`): HEFT on mean+std.
- **Shared wrapper:** `EstimateStochasticScheduler` (`estimate_stochastic_scheduler.py:24-130`) determinises, then schedules.
- **OnlineHEFT** (`online/algorithms/online_heft.py`):
  - **Setup:** a `StochasticEnvironment` (`online/environment/stochastic.py`, marked "*** WIP ***" at `:32`).
  - **Plan:** it plans on means.
  - **Replanning:** it replans at every completion with `ReschedulePolicy`.
  - **Execution:** it plays forward on sampled "actual" costs.

**Relevance to us.**
- **Estimated rank:** our "rank (estimated)" plan is MeanHEFT-order, but on *medians*. Our truth is the estimate × lognormal(σ), so its mean is est·e^{σ²/2}.
- **SHEFT variant:** rank on est·e^{σ} (or a quantile).
  - Group noise σ 0.31 and per-signature σ 0.60 differ (`RESULTS.md`, costs). So the risk-adjusted rank re-weights zero steps against signatures.
- **Verdict:** a one-line variant, `rank-q` (rank on a quantile). **Low–medium**: it probes "noisy ranks" (`LITERATURE.md` §1.3 item 3).

### 2.16 Online FIFO vs FrontierHEFT (`schedulers/online/`): SAGA's version of our open question

**FIFO** (`online/algorithms/fifo.py:7-21`, `online/environment/frontier.py:20-136`, `online/policy/frontier_fill.py`)
- **Frontier:** a task enters the frontier when all its predecessors are *finished* (`ready_condition="p_complete"`).
- **Key:** `(current_time at entry, name)`, i.e. task arrival time (`frontier.py:51-53, :117-120`).
- **Fill:** at each event (`next_event`: every start or end), pop as many tasks as there are idle nodes (`ready_node_only=True`; `frontier_fill.py:48-55`). Insert each with **EST** at `min_start_time=now`.
- **Node choice:** EST ties go to the node with the smallest *name*, because candidates are name-sorted (`parametric/components.py:139`, strict `<` at `:176`).
- **Speed:** FIFO is speed-oblivious except through node naming.

**FrontierHEFT** (`online/algorithms/frontier_heft.py:13-45`)
- **Frontier:** a task enters once its predecessors are *committed* (`p_committed`), so it plans ahead of execution.
- **Key:** `(−upward rank, −topological index)`.
- **Fill:** it schedules the whole frontier each step (`ready_node_only=False`) with **EFT** insertion.

**Relevance to us.**
- FrontierHEFT differs from FIFO in **three** ways at once: order, EFT placement, and look-ahead commitment.
- Our rank-vs-group comparison changes only the order.
- `scripts/examples/frontier_heft_vs_fifo/main.py` compares the two on one Montage workflow but records no result. Its outcome would not isolate order either.
- For us this supports the reading in `LITERATURE.md` §1.2: the HEFT gain is mostly the EFT half.
- FIFO is also the best **exact** cross-check target for our simulator (§4).

### 2.17 Ranking: what to add, and the mapping to our API

**1. EFT placement, optionally waiting for a fast slot** (HEFT/MCT/BIL machine half; StarPU `dmda`)
- **Engine:** a new `Choice::EarliestFinish { wait: bool }`.
- **Inputs:** it needs `WorkerState.speed: f64` (or a `class → speed` map in `BackfillConfig`), `JobSpec.work: Option<f64>`, and a start time per running job (`Running` gets `started`, `src/engine.rs:152-156`).
- **Rule:** predicted finish on worker w = `max(now, next_free_w) + work/speed_w`, where `next_free_w` = now if `running < slots`, else the minimum over its running jobs of `started + work/speed`.
- **Waiting:** with `wait`, a job whose best finish is on a busy worker is *not* placed now. It books that future slot in a per-dispatch tentative table, so two waiting jobs do not count on the same slot. Less urgent jobs may still backfill the idle slow slots.
- **sched-whole:** suffixes `+eft` and `+eft-wait`, next to `+fast` (`src/bin/sched_whole.rs:202-210`).
- **Expectation:** "fast first" won 27–31%. EFT-wait additionally stops a critical job from starting on a slow slot when a fast slot frees within (t_slow − t_fast).

**2. Fast class only** (FastestNode/MET)
- **sched-whole:** a `+fastonly` suffix that sets `spec.class = Some(fastest class)` in the `spec` closure (`src/sim/whole.rs:989-993`).
- **Cost:** under an hour of work. The lower bound is ~920 h (fast-class W/P), against 1,237 h for the best plan today.

**3. CPOP-style critical pinning**
- **DAG layer:** `DagConfig.critical: Option<CriticalConfig { class: String, slack: f64 }>`.
- **Rule:** in `submit_node` (`src/dag.rs:596-608`), set `spec.class = Some(class)` when `rank + group_tail ≥ (1 − slack) · max_live_rank`.
- **Bookkeeping:** `max_live_rank` comes from a `BTreeMap<OrderedRank, count>` over pending, held and submitted nodes, updated wherever `rank` changes (`propagate_rank` `:541-557`, `update_work` `:694-720`, `finish` `:651-671`).
- **Combination:** works with any order; non-critical jobs use fast-first or EFT.
- **Plans:** `group+cp`, `rank+cp`.

**4. Group order, then rank within the group** (not in SAGA, but SAGA's mix-and-match "parametric" family suggests exactly this decomposition: `parametric/__init__.py`, `parametric/components.py:23-183`)
- **Key:** `Key { priority, group, sub, seq }` with `sub = −(rank·scale)`.
- **API:** add `JobSpec.tiebreak: Option<i64>`, or `DagConfig.rank_within_group: bool`, which fills `tiebreak` instead of `priority` (`src/dag.rs:602-605`).
- **Rationale:** it keeps the oldest-first wavefront, which wins, and applies critical-path order inside A(3)/A(4) walks, which have 4,028 and 195,579 covers.
- **Plans:** `group-rank`, `group-rank+fast`.

**5. Fast-slot key** (DLS Δ / Sufferage)
- **Rule:** when exactly one fast slot frees and several jobs want it, pick by `level + w·(1/s_slow − 1/s_fast)` rather than key order.
- **Scope:** a sub-option of #1. Defer until #1 is measured.

**6. Quantile ranks** (SHEFT): `DagConfig.rank_cost_quantile`, or just scale estimates in `whole.rs` `cost(..)` (`:996`). Low.

**Not worth adding:** MinMin, MaxMin and Duplex (anti-critical-path or dominated); ETF, OLB and FLB (= existing greedy); WBA (random); PEFT and BIL (degenerate to HEFT under zero comm).

---

## 3. PISA: how it works, and a PISA for `sched`

### 3.1 Exactly how PISA works (`pisa/simulated_annealing.py`, `pisa/changes.py`, `scripts/experiments/pisa/run.py`)

**Objective.**
- **Energy:** `makespan(scheduler)/makespan(base_scheduler)`, **maximised** (`simulated_annealing.py:157-162, :384, :417-419`).
- **Result:** the best iteration is the one with the largest `current_energy` (`:317-328`).

**State.**
- **Contents:** a `(Network, TaskGraph)` pair.
- **Fixed size:** task count and node count never change. Only weights and edges do.

**Perturbations** (`changes.py`; one change type is drawn uniformly per iteration, `simulated_annealing.py:404-407`). Constants: `MINVAL=0.1, MAXVAL=1.0, DELTA=0.1` (`changes.py:15-17`).

| change | what it does (`changes.py`) |
|---|---|
| `TaskGraphDeleteDependency` | removes a random edge that touches no super node (`:79-105`) |
| `TaskGraphAddDependency` | shuffles nodes; for the first source with a valid target (not itself, not an ancestor, not an existing successor), adds the edge with size U(0.1, 1) (`:119-156`) |
| `TaskGraphChangeDependencyWeight` | size += U(−0.1, 0.1), clamped to [0.1, 1] (`:169-202`) |
| `TaskGraphChangeTaskWeight` | cost += U(−0.1, 0.1), clamped to [0.1, 1] (`:214-241`) |
| `NetworkChangeEdgeWeight` | a non-self link's speed += U(−0.1, 0.1), clamped (`:252-280`) |
| `NetworkChangeNodeWeight` | a node's speed += U(−0.1, 0.1), clamped (`:290-312`) |

- **Additive clamps** cap heterogeneity at 10×.
- **Graph construction:** add and delete build `TaskGraph(...)` directly, not via `create`, so deleting edges can leave several sources or sinks.

**Search** (simulated annealing, `simulated_annealing.py:350-462`)
- **Temperature:** starts at `max_temp`, multiplied by `cooling_rate` after each iteration. The loop stops at `max_iterations` or when `temp ≤ min_temp`.
- **Acceptance** (`:422-426`): `ratio = E_new/E_cur`. If `ratio > 1`, always accept. Otherwise accept with probability `exp(−ratio/T)`.
  - **[DEV, looks wrong]** This is not Metropolis. At a fixed T, a *much worse* neighbour (small ratio) is accepted **more** often than a slightly worse one (ratio ≈ 1).
  - **Standard rule:** `exp((E_new − E_cur)/T)`, or `exp(ln(ratio)/T)`.
  - **Effect:** at low T it accepts almost nothing below 1 except very bad moves. The search is effectively a greedy hill-climb plus occasional large jumps.
- **Cost per iteration:** it re-runs both schedulers on the *current* instance too, only to log it (`:428-434`). That doubles cost and makes the logged energy of a randomized scheduler such as WBA noisy.

**Defaults and the paper's settings.**
- **Library defaults:** T 100 → 0.1, ×0.99, max 1,000 iterations, so ~688 iterations are run (`:86-102`).
- **Paper experiment** (`run.py:41-52, :272-311`), for every ordered pair of the 16 schedulers in `SCHEDULERS` (`simulated_annealing.py:53-70`):
  - **Restarts:** 10 random restarts.
  - **Temperature:** T 10 → 0.1, ×0.99. That is 459 iterations (ln 100/−ln 0.99 ≈ 458); the cap of 1,000 never binds.
  - **Initial instance:** a chain of 3–5 tasks with U(0.1, 1) weights on a complete network of 3–5 nodes (`run.py:106-123`).
  - **Output:** the best try per pair (`:150-188`).
- **Homogeneity restrictions** (`run.py:35-38, :110-117`):
  - **Fixed speeds:** for ETF, FCP and FLB (compute) and for BIL, GDL, FCP and FLB (communication), node or link speeds are fixed at 1.0, and the matching change type should be omitted.
  - **[Bug]** The appended change types are swapped. Heterogeneous compute appends `NetworkChangeEdgeWeight`, and heterogeneous communication appends `NetworkChangeNodeWeight`. So, for example, ETF runs get node-speed perturbations.
- **Optimal reference:** `BruteForceScheduler` (`schedulers/brute_force.py`) enumerates every mapping × every topological order, append-only. That is OPT for zero communication and non-preemptive tasks; it is usable as a base for 5–7 tasks.

### 3.2 Why naive PISA would not answer our open question

On 3–5-task instances, *any* two list orders show Graham anomalies, with ratios up to about 2 − 1/m in either direction.

- **What a raw run shows:** a PISA run "group vs rank" will find both "group ≫ rank" and "rank ≫ group" witnesses. That proves only that neither dominates, which we already know (`LITERATURE.md` §1.2).
- **Our question is narrower:** why group order wins **on our grid-shaped, template-expanded, noisy-estimate family** at 16 slots.
- **What the search must therefore do:**
  - (a) **Constrain the instance family** to that structure.
  - (b) **Seed from a scaled-down replica** of the real world.
  - (c) **Report typical-case distributions** alongside the adversarial maximum.
  - (d) **Minimise and featurise** witnesses so that a mechanism, not an instance, comes out.

### 3.3 Plan: `sched-pisa`, an adversarial search over our simulator

**New code**
- `src/sim/small.rs`: a small-instance model and a flat driver.
- `src/bin/sched_pisa.rs`: the search.
- **Feature:** `sim`.
- **Dependencies:** none new. `clap`, `serde` and `serde_json` are already present. Use the SplitMix64 helpers `mix`, `uniform` and `normal` from `src/sim/whole.rs:121-139`, moved to `src/sim/rng.rs`. `proptest` is already a dev-dependency for invariants.

**Instance**

```rust
pub struct SmallTask { pub group: u32, pub work: f64, pub est: f64, pub deps: Vec<u32> }
pub struct Class    { pub speed: f64, pub workers: u32, pub slots: u32 }
pub struct SmallInstance { pub tasks: Vec<SmallTask>, pub classes: Vec<Class> }
```

- **Acyclicity:** deps always point to lower indices, so every DAG is acyclic by construction and "add edge" is O(1).
- **Groups:** `group` is only a priority label (`JobSpec.group`).

**Driver.** `simulate_small(&inst, &SmallPlan) -> SmallResult { makespan, d_fast, w_over_p, contention_frac, slow_on_crit }`.
- **Setup:** declare all tasks at t = 0 through `DagScheduler` (`auto_submit: true`, `rank_priority` per plan, estimates via `with_work(est or work)`), on `PriorityBackfill` with unbounded memory.
- **Workers:** processor sharing with `ClassCurve { speed, k_sat: slots, alpha: 1.0 }`.
- **Event loop:** reuse `advance`/`schedule` from `whole.rs` (`:1217-1258`), factored into `src/sim/ps.rs`.
- **`SmallPlan`:** `{ order: Group | Rank{oracle, age} | GroupRank | Fifo, place: LeastLoaded | FastFirst | FastOnly | Eft{wait} }`.

**Generators** (initial states)
- **G1, flat** (classic PISA): 6–40 tasks; a random layered DAG, chain, fork-join or in/out-tree; group = task, or group = layer.
- **G2, mini-Nassau** (the family that matters):
  - **Coarse grid:** S ∈ [2, 6] rows × N ∈ [6, 30] columns of groups, with our `depgraph` edges:
    - `(s, t−f) → (s, t)` with floor f ∈ {1, 2, 3};
    - `(s−1, t−1) → (s, t)`;
    - the registration chain `(s, t−1) → (s, t)`.

    These mirror `compute_deps` (`whole.rs:704-722`) and the `reg` deps (`:1015-1022`).
  - **Group contents:** a zero step plus a walk template, drawn from chain, fork-join and layered(width ∈ [2, 8], depth ∈ [2, 6]).
  - **Liveness:** a fraction ρ ∈ [0.2, 0.6] of groups are live; the rest have no walk.
  - **Work:** grows by a per-column factor g ∈ [1.0, 1.3], with within-group shares and lognormal truth noise (σ_sig ∈ [0, 0.8], σ_group ∈ [0, 0.4]). Estimates = medians, as `World::sig_work`.
  - **Expansion:** lazy, as `simulate` does it (`instantiate` at zero-step readiness, `whole.rs:1043-1078`) when the plan flag `lazy` is set. Otherwise eager.
- **G3, replica seed:**
  - **Build:** the existing synthetic `World` (test census with s ≤ 4, n ≤ 40, `whole.rs:1262-1290`) or a real-census `World` cut at `max_n` 40–60, `max_s` 6–10.
  - **Export:** flatten to `SmallInstance` with an exporter `World::to_small(max_tasks)`. Truncate A(4) walks if needed.
  - **Search scope:** perturb only costs, estimates and fleet. This answers "which property of *our* instance makes group win".
- **Fleet:** two classes, speeds `{1, r}` with r ∈ [1, 4] (ours is 2.41), 1–6 workers each, 1–16 slots. Seed ratio fast:slow workers 2:1 as in the trace.

**Perturbations** (multiplicative, log scale, because our costs span orders of magnitude; PISA's additive [0.1, 1] clamp cannot express that)

| op | description | families |
|---|---|---|
| P1 | `work *= exp(U(−δ, δ))`, δ = 0.5; clamp to a 10⁴ dynamic range | all |
| P2 | `est *= exp(U(−δ, δ))`: estimate error only | all |
| P3 | add edge i→j (i < j); within a template in G2 | G1, G2-template |
| P4 | delete edge (G2: only template edges; grid edges fixed) | G1, G2-template |
| P5 | move a task to another group (relabel; changes only group order) | G1 |
| P6 | swap two groups' first-arrival order (G1) / change floor f per row (G2) | G1, G2 |
| P7 | fleet: r ± U(0, 0.25); workers ± 1; slots ± 1 (bounded) | all |
| P8 | add or remove a leaf task / template layer | G1, G2 |
| P9 | G2 only: scale one row's or one column's work (moves the critical path between rows) | G2 |

- **Validity:** after P3 and P8, recompute and keep the DAG.
- **G3 scope:** only P1, P2, P7 and P9.

**Objective and search**
- **Energy:** `E = makespan(A)/makespan(B)`.
- **Optional normalised variant:** `E' = makespan(A)/max(D_fast, W/P)`. This finds instances where A is bad in absolute terms, not just relative to B.
- **Simulated annealing** on `ℓ = ln E` with standard Metropolis:
  - **Acceptance:** always accept if Δℓ ≥ 0, else with probability `exp(Δℓ/T)`. This replaces PISA's `exp(−ratio/T)`.
  - **Schedule:** T₀ = 0.05 (a 5% worse move is accepted with probability 1/e), T_end = 5·10⁻⁴, geometric over N iterations.
  - **Per iteration:** evaluate only the neighbour. The current instance's energy is cached, unlike PISA (`:428-434`).
- **Ties and noise:** both plans are deterministic. For estimate-noise robustness, evaluate each state on K = 3 noise seeds (common random numbers) and use the mean of ln E.

**Pairs to run** (both directions each)
- `group+fast` vs `rank-oracle+fast`;
- `group+fast` vs `rank+fast` (estimated);
- `group` vs `rank-oracle` (no speed);
- `group+fast` vs `group-rank+fast`;
- later, `group+fast` vs `group+eft-wait`, and `fast-first` vs `fast-only`.

**Budget**
- **Cost per run:** a 40-task G1 instance simulates in about 10–50 µs (estimated from `dispatch` at a few µs per event; **[U]** until measured). A G2 instance with about 2k tasks takes about 1–5 ms.
- **G1:** N = 20,000 iterations × 32 restarts × 2 plans ≈ 1.3M runs, which is minutes on one thread.
- **G2:** N = 3,000 × 16 restarts ≈ 100k runs, about 5 min.
- **G3:** N = 1,000 × 8 restarts.
- **Node limits:** run with `ulimit -v 8000000` on one core. Memory is tiny (one `DagScheduler` per run).

**Outputs** (`--json`)
- **Best witness:** per pair, family and restart, with its energy.
- **Minimisation:** afterwards, greedily remove tasks and edges and snap weights to round values while E stays ≥ 0.95·E_best. This gives a small witness to read.
- **Typical case** (no annealing): sample 10k G2 instances and report the distribution of `ln E` (mean, p5, p95, P(E > 1)). The adversarial maximum alone is uninformative (§3.2).
- **Features** per state:
  - Kendall τ between group order and rank order over jobs that were simultaneously waiting;
  - fraction of time with ready > free slots (contention, `LITERATURE.md` §1.3 item 1);
  - critical-chain work run on slow slots;
  - rank staleness (submitted rank vs final rank);
  - aging activations.

  Then regress `ln E` on these features over all accepted states. This is the step that can explain the open question: e.g. "rank loses when contention is low and the critical chain starts on a slow slot", or "when estimate error exceeds X".

**Hypotheses the search can confirm or refute** (from `RESULTS.md` and `LITERATURE.md` §1.3)
- **H1:** group order ≈ 1DF along the wavefront; rank adds nothing on grids, and loses through tie or estimate noise.
  - **Test:** G3 with P2 only. Does E → 1 as estimate error → 0?
- **H2:** ranks are frozen at submission and the placeholder `cp_est` is stale.
  - **Test:** a `rank-refresh` variant that re-keys waiting jobs when their rank changes by more than ε.
- **H3:** rank interacts badly with speed-oblivious placement.
  - **Test:** E with `+fast` vs `+eft-wait`.
- **H4:** rank starves off-path groups until they become critical. This is the 348 h tail.
  - **Test:** sweep the aging parameter inside the search.

---

## 4. Cross-check against SAGA

### 4.1 Exporting our instances
- **Command:** `sched-pisa --export-saga out.json <instance>` writes:
  - `tasks: [{name, cost}]`: `cost` = **true** work in H200-seconds; names are zero-padded ids.
  - `dependencies: [{source, target, size: 0}]`, with passthroughs contracted (their in-edges × out-edges become edges).
  - `nodes`: **one SAGA node per slot**, `{name, speed}`. Named so that the fast class sorts first: `a_<class>_<worker>_<slot>`.
  - `edges`: every pair with `speed: 1.0`. Never omit them, because the 0.0 default divides 0/0 (`__init__.py:96, :684, :826`). Self-loops get `inf`.
- **Python side:** `TaskGraph.create(...)` (which adds super source and sink with cost 0) and `Network.create(nodes, edges)`.

### 4.2 What can be compared, and how exactly

**(a) Exact match: our `group+fast` vs SAGA `FIFOScheduler`** (`online/algorithms/fifo.py:290-301`)
- **Conditions:**
  - flat instances with group = task, so our key `(group first arrival, seq)` reduces to arrival order;
  - no passthroughs;
  - generic (tie-free) costs: with no simultaneous completions, our readiness order (dependents sorted by id per completion, `src/dag.rs:651-667`) matches FIFO's `(arrival time, name)` key;
  - `k_sat = slots`, α = 1;
  - fast slots named first: FIFO's EST tie-break picks the smallest-named idle node (`components.py:139, :176`), so it places on the fast class first.
- **Expectation:** equal makespans to 1e-9.
- **Exact match is only possible with fast-first:** within a class every slot is equivalent under α = 1, but our least-loaded choice (`src/engine.rs:446-470`) selects a *worker* by load, which a static name order cannot reproduce across classes. Without "+fast", compare single-class fleets only.
- **Single source:** with `create` there is exactly one source (`__super_source__`). That avoids FIFO's bootstrap, which commits *all* sources at reset in `frozenset` order (`frontier.py:84-92`).

**(b) Simulator validity**
- **Export:** our realised schedule as (task, slot, start, end), slots assigned post hoc by interval colouring per worker.
- **Load:** into `Schedule.add_task` in start order. It raises on any overlap (`__init__.py:885-891`).
- **Assert in Python:** each duration = cost/speed, and each start ≥ the latest parent end.
- **Also:** a Rust unit test can do the same without Python.

**(c) Bounds**
- Every SAGA makespan must be ≥ our `max(D_fast, W/P)` from `simulate_small`.
- `BruteForceScheduler` (instances with ≤ 6 tasks and ≤ 3 slot-nodes) gives **OPT**. Report `ms(plan)/OPT` for our plans on G1 witnesses: the true approximation ratio.

**(d) Reference points, not equality**
- **Run:** SAGA `HEFT`, `CPoP`, `MCT`, `MinMin`, `MaxMin`, `FastestNode`, `OLB`, `ETF`, `FrontierHeftScheduler` on the same exports.
- **Expected relationships:**
  - SAGA HEFT ≤ our `rank-oracle+eft` ≈ `rank-oracle+fast` (it has insertion and clairvoyance);
  - OLB ≈ our `group` on single-class fleets;
  - FastestNode ≥ our `+fastonly` (one slot vs the whole class).
- **Use:** a large gap between SAGA HEFT and our best online plan bounds what clairvoyant planning could add.

### 4.3 Reconciling the models
- **Processor sharing:** with α = 1 and k ≤ k_sat, PS is identical to per-slot machines. Otherwise there is no SAGA equivalent; use a cross-check fleet with `k_sat = slots`.
- **Communication:** sizes 0, links 1.0.
- **Memory:** off on both sides.
- **Lazy expansion:** export the fully expanded DAG. It cannot matter for (a) because our DAG layer releases jobs only when ready either way. It *does* matter for rank plans: unexpanded groups use the `cp_est` placeholder (`whole.rs:1023-1028`). For rank comparisons, therefore, export from an eager run.
- **Estimates:** SAGA has none. Export the true costs for HEFT; our `rank` plan uses estimates. Compare against `rank-oracle`.

### 4.4 Running SAGA here
- **Python:** the node has Python 3.7 only, and SAGA needs ≥ 3.12.
- **Setup:**
  - download the standalone `uv` binary into `/wsu/home/hd/hd72/hd7264/.claude/jobs/71ebdf0b/tmp/refs/` (or a similar scratch directory);
  - `uv venv --python 3.12` there;
  - `uv pip install -e saga`.

  This needs network access and about 300 MB. **[U]**: not attempted.
- **Run:** under `ulimit -v 8000000`, instances ≤ 200 tasks. SAGA's O(V²P) Python is fine there.
- **Location:** keep the script outside the crate, or as `ext/crates/sched/scripts/saga_crosscheck.py` with no Rust dependency.

---

## 5. Implementation plan

Effort is in focused days. The steps are ordered so that each one produces a usable result.

| # | step | files | new API / plans | tests | effort |
|---|---|---|---|---|---|
| S0 | factor the PS driver (`Wk`, `advance`, `schedule`, heap `Item`) and the SplitMix helpers out of `whole.rs` | `src/sim/ps.rs` (new), `src/sim/rng.rs` (new), `src/sim/whole.rs`, `src/sim/mod.rs` | none (internal) | existing `whole_run_plans_finish_within_bounds` unchanged; byte-identical `sched-whole` JSON on the test world | 0.5 |
| S1 | `+fastonly` plan | `src/sim/whole.rs` (`simulate` gets `place: Place` instead of `fast_first: bool`), `src/bin/sched_whole.rs` (suffix parsing `:202-210`) | `pub enum Place { LeastLoaded, FastFirst, FastOnly }` | plan finishes; makespan ≥ fast-only W/P | 0.25 |
| S2 | `simulate_small` + bounds + JSON I/O | `src/sim/small.rs` (new) | `SmallInstance`, `SmallPlan`, `simulate_small`, `small_bounds` | chain → Σw/s_fast under FastFirst; independent equal tasks on m identical slots → ⌈n/m⌉·w; determinism; proptest: random DAGs finish with makespan ≥ bounds and ≤ W/P + D (greedy bound, identical speeds) | 1 |
| S3 | group-then-rank ordering | `src/lib.rs` (`JobSpec.tiebreak: Option<i64>`), `src/engine.rs` (`Key` gets `sub`, `:116-121`, filled at `submit`, `:250-283`), `src/dag.rs` (`DagConfig.rank_within_group`, `submit_node` `:596-608`), `whole.rs` `Plan::GroupRank` | `JobSpec::tiebreak`, `DagConfig::rank_within_group`, plan `group-rank` | engine: within one group, a smaller tiebreak goes first; across groups, group order is unchanged; DAG test | 0.75 |
| S4 | `sched-pisa` binary: generators G1–G3, P1–P9, SA, minimisation, typical-case sampling, features, `--export-saga` | `src/bin/sched_pisa.rs` (new), `src/sim/small.rs`, `src/sim/whole.rs` (`World::to_small`), `Cargo.toml` (`[[bin]] sched-pisa`, `required-features = ["sim"]`) | CLI: `--family g1\|g2\|g3 --a <plan> --b <plan> --iters --restarts --seed --json --export-saga` | SA with A = B gives E = 1 everywhere; perturbations keep deps < index (proptest); seed reproducibility; G3 export matches `World::summary` task counts | 2 |
| S5 | SAGA cross-check | `scripts/saga_crosscheck.py` (or kept in scratch), exporter from S4 | none | FIFO exact match on 100 random G1 exports; schedule validity; BruteForce OPT ≤ all | 1 (+0.5 env) |
| S6 | EFT placement (± wait) | `src/lib.rs` (`WorkerState.speed: f64`, default 1.0 in `new`; `JobSpec.work: Option<f64>`), `src/engine.rs` (`Choice::EarliestFinish{wait}`, `Running.started`, per-dispatch booking, `explain` text), `src/dag.rs` (copy `work_estimate` into `spec.work` at `submit_node`), `whole.rs` + `small.rs` (`Place::Eft{wait}`) | `EftConfig` or a `BackfillConfig.choice` field; plans `+eft`, `+eft-wait` | never waits when the slow finish ≤ the fast finish; waits when a fast slot frees within (t_slow − t_fast); two waiting jobs never book the same slot; aging still bounds waits; dispatch stays O(jobs·workers); determinism | 2.5 |
| S7 | CPOP-style critical pinning | `src/dag.rs` (`DagConfig.critical: Option<CriticalConfig>`, live-rank multiset maintained in `declare`/`propagate_rank`/`update_work`/`finish`) | `CriticalConfig { class, slack }`; plans `+cp` | the critical chain only on the fast class; with slack = 0 only the max-rank jobs are pinned; multiset consistent under `cancel`/`snapshot`-`restore` | 1.5 |
| S8 | runs + write-up | `RESULTS.md` (new sections: SAGA cross-check; PISA findings; new plans on the whole run) | none | none | 1.5 |

Total: about 11 days.

**Order of value**
1. S0–S2 and S4 (the search) can start immediately and do not touch the library API.
2. S1 and S3 are cheap plan additions for the 8-minute whole-run sims.
3. S6 is the largest expected win and the largest change.

**Risks**
- **Uninformative adversarial maxima:** on tiny DAGs, both directions show Graham-anomaly ratios of about 2.
  - **Mitigation:** G2/G3 families, typical-case sampling, and feature regression (§3.2).
- **Breaking API change:** adding fields to the public, non-`#[non_exhaustive]` structs `JobSpec` and `WorkerState` (`src/lib.rs:119-188`) breaks struct-literal users.
  - **Mitigation:** add them with constructors and defaults. The crate is 0.1 and only in-tree.
- **EFT-wait under noisy estimates** (per-signature σ 0.6) can idle fast slots or wait for the wrong ones.
  - **Mitigation:** test under estimate noise in `sched-pisa` (P2), and cap waiting at a fraction of the job's slow-runtime advantage.
- **Rounding and ties:** `rank_scale 1000` rounding and `rank_epsilon 0.01` (`src/dag.rs:232-262`) create ties that differ from SAGA's float ranks. This affects only rank-plan comparisons; use `rank_epsilon = 0` in the cross-check.
- **H200 `k_sat = 15` with 16 slots:** the cross-check fleets must use k_sat = slots, or exact equality fails.
- **Python environment:** SAGA needs Python 3.12 and network access to install (§4.4). If that is unavailable, S5 is limited to Rust-side validity checks.
- **Simulation time:** whole-run sims take about 8 minutes per plan, and each new plan × {fast, eft, cp} multiplies that. Use the G3 replica to prune before running the full world.
- **Suspected SAGA bugs:** `[DEV]` items in GDL (min/max inversion), BIL (k/|V|), FLB (heap mutation), the PISA acceptance rule, and the swapped change types in `run.py`. Do not treat SAGA's numbers for those schedulers, or its PISA ratios, as ground truth without checking.
