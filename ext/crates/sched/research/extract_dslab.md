# dslab-dag: what to extract into `sched`

Clone: `refs/dslab` at a6c155e8db (2024-07-28), shallow. Unless stated otherwise, paths are relative to
`refs/dslab/crates/dslab-dag/`. `refs/` means `/wsu/home/hd/hd72/hd7264/.claude/jobs/71ebdf0b/tmp/refs/`.
Our crate is `ext/crates/sched` in the `sched-crate` worktree.

Built and run here: `refs/target/release/dag-demo` (`cargo build --release -p dag-demo`, 46 s).
Toy instances and scripts are in `refs/xval/`. Everything marked **[measured]** comes from those runs.

---

## 1. What the package is, and its exact models

dslab-dag is a discrete-event simulator for executing one DAG on a set of resources. A pluggable
`Scheduler` emits actions. It ships HEFT, PEFT, Lookahead and DLS (all static), plus `DynamicList` and
`Simple` (dynamic). It is a workflow-benchmarking tool, not a runtime library: everything is
`Rc<RefCell<…>>` on simcore's event loop, so none of its code can be linked into our engine. What
transfers is the algorithms.

### Task (`src/task.rs:51-64`, `src/dag.rs`)
- **Fields:** `flops` (Gflop), `memory` (MB, u64), `min_cores`/`max_cores`, `cores_dependency`,
  input and output data items, and an optional `ResourceRestriction::{Only, Except}` (task.rs:21-33).
- **Edges:** dependencies are *data items*. An item has one producer and many consumers
  (dag.rs:281-314). A task becomes Ready when all of its input items are Ready (dag.rs:332-354).
- **States:** Pending, Ready, Scheduled, Runnable, Running, Done (task.rs:5-18). A task can be
  scheduled before it is ready (static plans).

### Speed, cores and time (`../dslab-compute/src/multicore.rs`)
- **Execution time:** `flops / speed / speedup(cores)` (multicore.rs:475, and :529 on resume).
- **Speedup:** `Linear` = cores; `LinearWithFixed` = Amdahl `1/(f + (1-f)/cores)`; or `Custom`
  (multicore.rs:35-61).
- **Cores are exclusive (space sharing).** No processor sharing. The runner keeps one FIFO queue per
  core (runner.rs:75, 339-345) and starts a task when `need_cores` cores and the memory are free and
  its inputs are present (runner.rs:396-482). The core count is fixed when the task starts.
- **Time:** simcore's f64 seconds. **Makespan** = `sim.time()` after `step_until_no_events()`
  (src/experiment.rs:213-215; dag-demo prints the same).
  - It includes uploading the DAG outputs to the master (runner.rs:596-605).
  - Completion = all tasks Done and no transfers in flight (runner.rs:201-203).
  - If the run deadlocks, it still prints a "makespan", which is just the time of the last event,
    plus an error log line (runner.rs:206-222). See the HEFT deadlock below.

### Resource (`src/resource.rs:21-56`)
- **Fields:** `speed` (Gflop/s, per core), `cores`, `memory` (MB).
- **Memory is a hard capacity:** `Σ running task.memory ≤ memory` (runner.rs:417, 443-453). There is
  no reported usage, no baseline and no escape hatch.
- **Master:** `DagSimulation::init` adds a resource named `master` (speed 1, 1 core, 0 memory) if none
  exists (dag_simulation.rs:58-60).
  - It restricts every real task to `Except(master)`.
  - It adds `input` and `output` pseudo-tasks pinned to the master (dag_simulation.rs:137-161).
  - The master leaks into several formulas (see the quirks list).

### Network (`src/network.rs`, `src/data_item.rs`)
- **Models:** ConstantBandwidth, SharedBandwidth, or TopologyAware star/full mesh
  (network.rs:186-210). Bandwidth is in MB/s and latency in µs.
- **Transfer modes:** `ViaMasterNode`, `Direct` or `Manual` (data_item.rs:52-81).
  - The planning cost is `size × net_time`, with `net_time = 1/bw` (Direct) or the sum of the two hops
    (via master) (data_item.rs:66-81).
  - Lazy-strategy downloads add the latency (common.rs:134-157).
- **Zero-cost setting:** with sizes 0, latency 0 and a huge bandwidth, every transfer takes 0 s
  **[measured: the critical-path-only toy gives exactly the fast-speed critical path]**.

### Scheduler interface (`src/scheduler.rs:27-93`)
- **Callbacks:**
  - `start(dag, system, config, ctx) -> Vec<Action>`;
  - `on_task_state_changed(...)`, which the runner calls **only when a task completes**
    (runner.rs:607-620);
  - `is_static()`.
- **Actions:**
  - `ScheduleTask{task, resource, cores}`;
  - `ScheduleTaskOnCores{…, expected_span}`;
  - `TransferData`.
- **Dynamic schedulers** see the current `System` (`cores_available`, `memory_available`) and
  `dag.get_ready_tasks()` (a BTreeSet, so ascending id).

### Inputs and outputs
- **DAG files:**
  - YAML (yaml_parser.rs): `tasks[{name, flops, memory, min_cores, max_cores, cores_dependency,
    inputs[names], outputs[{name, size}]}]`, top-level `inputs`. Flops are taken as given, and
    `reference_speed` is unused here.
  - WfCommons JSON (runtime × reference_speed).
  - DAX XML.
  - DOT (node `size` in flops → Gflop; edge `size` in bytes → MB; dot_parser.rs:132,154).
- **System YAML:** `resources[{name, speed, cores, memory}]` plus `network{model, …}` (see
  `examples/dag-demo/systems/*.yaml`).
- **Drivers:**
  - `examples/dag-demo` runs 47 scheduler configs on one (dag, system) pair;
  - `Experiment` (experiment.rs) runs a dag × system × scheduler grid on a thread pool;
  - `examples/dag-portfolio` ranks the DynamicList grid on 8 WfCommons workflows × 4 platforms.
- **Lower bound** (lower_bound.rs:8-48): `max(critical path at each task's fastest resource,
  total flops / Σ speed·cores)`. The capacity sum includes the master's core.

### Algorithms (formulas, tie-breaks)

| scheduler | file:lines | rank / order | resource choice | ties |
|---|---|---|---|---|
| HEFT | heft.rs:39-125 | `rank_u = w·avg_flop_time + max_succ(rank_u(succ) + size·avg_net_time)` (common.rs:272-297) | min EFT with insertion (common.rs:60-178, 180-261) | rank: stable sort, so lower id first (heft.rs:49); resource: strict `>`, so lowest index (heft.rs:88) |
| PEFT | peft.rs:47-189 | `OCT(t,p) = max_succ min_p' [OCT(s,p') + w_s/speed_p' + size·c(p,p')]`; rank = mean over p of `OCT(t,p)` (peft.rs:55-92); always the highest-ranked *ready* task (peft.rs:109-121) | min `EFT(t,p) + OCT(t,p)` (peft.rs:148-150) | lowest index |
| DLS | dls.rs:39-145 | static level `SL` = rank_u at the *median* flop time over non-master resources (dls.rs:47-55) | joint pick: max over ready tasks × resources of `DL = SL(t) − EST(t,r) + w_t·(median_ft − 1/speed_r)` (dls.rs:106-108) | first found (rank order, then index) |
| Lookahead | lookahead.rs:65-284 | HEFT order | for each candidate resource, tentatively schedule all unscheduled tasks within `depth` by HEFT; keep the resource with the smallest resulting makespan (prunes at `≥ best`, :226) | lowest index |
| DynamicList | dynamic_list.rs | see the grid below | | |
| Simple | simple_scheduler.rs:23-55 | ready tasks by ascending id | first resource (index order) that fits | none |

Static schedulers emit the whole plan at `start`, sorted by planned start (heft.rs:123-124). The
runner then enforces the per-core order (runner.rs:396-482).

**The DynamicList grid** (dynamic_list.rs:28-74). It reruns at start and at every completion
(:436-463).

- **Task criteria**, for `schedule_by_task` (:110-295). Ready tasks are sorted descending by one of:
  - `CompSize` (flops);
  - `DataSize`;
  - `ChildrenCount` (actually counts *output items*, :180);
  - `BottomLevel` (rank_u at the average speed, recomputed over the whole DAG at every call, :134);
  - `Cores`, `Memory`, `Cores×Memory` (sum or product), `Cores×Flops`, `Memory×Flops`,
    `CoresMemoryFlops` (buggy comparator, :204-205).

  Ties go to ascending id, because the sort is stable over the BTreeSet.
- **Resource criteria.** Each task picks the `min_by` (so the first, i.e. lowest index, on ties) among
  resources that fit now:
  - `Speed` (max speed; no load term);
  - `TaskData` (most input bytes already there, then speed);
  - `Max/MinAvailableCores`, then speed;
  - `Max/MinAvailableMemory`, then speed;
  - `DotProduct` (`(avail_cores·task_cores/cores² + avail_mem·task_mem/mem²)/2`, :426-434), then
    speed;
  - `DotProductSpeed` (dot + normalised speed).

  A task that fits nowhere is skipped. That is backfill without reservation: big tasks can starve.
- **RankPack\*** (`schedule_by_resource`, :298-423): resources sorted by (speed desc,
  `cores_available` desc). Each one in turn repeatedly takes the ready task maximising
  `α·rank/max_rank + (1−α)·dot`, with α ∈ {0, .25, .5, .75, 1}, `Mult` (rank·dot), or `Sel` (best dot
  among the top 10 by rank).
- **Cores criterion:** `MaxCores` (everything available, clamped to `max_cores`) or
  `Efficiency90/50` (largest core count with `speedup/cores ≥ e`).

**Quirks and bugs to know before cross-validating.** I verified 1 and 2 by running them.
1. **HEFT and Lookahead deadlock when zero-flop tasks are not in topological id order**
   **[measured]**.
   - **Cause:** a zero-flop join ties on rank with its successor. The stable sort then plans the child
     first, and it is evaluated with its parent "unscheduled", i.e. start 0 (common.rs:106-108). The
     per-core FIFO then waits forever.
   - **Example:** `refs/xval/grid_rev.yaml` gives "189 Scheduled, 1 Runnable, 3 Done" and a bogus
     makespan of 4.78. The same DAG listed in topological order, or with 1e-6 flops on the joins,
     gives 100.42 (`grid_eps.yaml`, `grid_rev_eps.yaml`).
2. **Speed-oblivious first-fit is terrible on related machines.** `Simple` with the slow resource
   listed first: 236 vs 100.42 on the chain-bound toy **[measured]**. It is the same effect as our
   "fast first" finding.
3. **PEFT ignores restrictions in the OCT.** It takes the min over *all* resources (the master
   included) and averages them into the rank (peft.rs:62-92). Keep every real speed ≥ the master's 1.
4. **avg_flop_time assumes exactly one master.** It filters out the master but divides by
   `len − 1` (system.rs:18-26).
5. **The lower bound counts the master's core in capacity** (lower_bound.rs:22-29).
6. **Rank recomputation is O(V+E) per completion.**
   - `calc_rank` is recursive DFS (common.rs:272-284), so deep chains risk overflowing the stack.
   - DynamicList recomputes ranks over the whole DAG at every completion, for every task criterion
     (dynamic_list.rs:134, 333). So dynamic runs cost O(N·(V+E)).
   - PEFT and DLS selection is O(N²) (peft.rs:109-121; dls.rs:71-117). Full Lookahead is about
     O(N²·R²): 24.8 s for 896 tasks **[measured]**.
7. **The Treap uses `thread_rng`** (treap.rs:3,150). The tree shape is nondeterministic; the results
   are not.

**Reading verified against dslab's output.** I re-implemented `DynamicList[BottomLevel, Speed]` in
`refs/xval/dynsim.py`: exclusive cores, ready set at completions, stable rank sort, fastest free core,
lowest index on ties. It reproduces dslab exactly: 91.00 on no-comm test_4 and 289.03 on the wide toy
**[measured]**.

---

## 2. What to extract into `sched`

Overall:
- **Key structural fact:** with **no communication costs** and **related machines** (a single speed
  ratio for every task), all of dslab's ranks collapse to one order:
  - HEFT's average-speed rank, DLS's median-speed SL and our `DagScheduler` upward rank
    (dag.rs:541-556) are the same quantity times a constant.
  - PEFT's OCT stops depending on `p`: the `c(p,p')` term vanishes, so
    `OCT(t,·) = rank_fast(t) − w_t/s_fast`.
  - PEFT's choice therefore becomes HEFT's EFT choice, and its rank becomes "remaining critical path
    excluding self".
- **So for us, the whole value is in the resource choice** (EFT and its "wait for a fast slot"
  variant, DLS's speed delta). Ordering adds nothing new, which is consistent with our RESULTS.md
  finding that rank does not beat group order.

### E1. A native speed-aware worker choice ("fast first" = DynamicList `resource=Speed`) — ENGINE
- **Source:** dynamic_list.rs:233-234 (max speed); ties → speed for every other criterion
  (:239-261).
- **What it computes:** among admitting workers, the highest speed first. dslab breaks ties by lowest
  index; we keep our tuple.
- **Mapping:**
  - Add `pub speed: f64` to `WorkerState` (lib.rs:159-174; default 1.0 in `WorkerState::new`) and
    `speed` to `WorkerLoad`.
  - Add a public `SpeedPolicy` and put it *before* `fit` in `Engine::choose`'s score
    (engine.rs:446-470): `(lane_rank, speed_key, fit, !preferred, running)`.
- **Why it is needed:** today the simulator fakes it with `prefer = fast workers` (whole.rs:982-992).
  - For Greedy and PriorityBackfill, `fit = 0`, so prefer decides.
  - **For BestFit, `fit` comes before `!preferred`, so the faked fast-first is silently a
    tie-breaker only** (engine.rs:457-464). A native key fixes that, and fast-first + BestFit becomes
    "fastest class, tightest fit within it".
- **Processor-sharing adaptation:** dslab's `Speed` has no load term because its cores are exclusive.
  - **Under our PS model** (per-job rate `f(k)/k`), the right key is the per-job rate after placement,
    `f(k+1)/(k+1)`.
  - **With the fitted α = 1, k ≤ k_sat** (RESULTS.md: H200 k_sat 15, L40S 16), that equals `speed`,
    so the plain `speed` key is exact except H200 at k = 16.
  - **Optionally**, `WorkerState::rate_at: Option<Arc<[f64]>>` (per-job rate by running count) for
    sublinear classes. Not needed now.
- **Where:** engine (runtime). The coordinator knows each worker's class and can supply `speed` from
  config.

### E2. Earliest-finish-time choice with "wait for a faster slot" (HEFT's processor selection, online) — ENGINE + DAG layer
- **Source:**
  - `evaluate_assignment` (common.rs:60-178): `EFT(t,r) = EST(t,r) + w_t/speed_r/speedup`, with
    `EST` = the earliest gap of `need_cores` cores and enough memory over the whole interval, from
    `max(parent finish + transfer)` (find_earliest_slot, common.rs:180-261);
  - the choice `argmin EFT` (heft.rs:67-94).
- **What transfers online:** insertion into future gaps needs a full plan, which we do not have.
  The online residue is:
  - for each eligible worker `r`, `free_at(r)` = `now` if it admits, else the earliest expected
    completion among its running jobs (`started + work/speed`);
  - `EFT(r) = max(now, free_at(r)) + work/speed_r`;
  - **if the argmin is a busy worker** whose EFT beats the best admitting worker's, the job *waits*
    (skip, no reservation), and less urgent jobs may still take the slow slot (backfill);
  - **otherwise** place it on the argmin admitting worker.
- **Evidence:**
  - **No-comm test_4** **[measured/simulated]**: non-idling `BottomLevel×Speed` gives 91; rank + EFT
    with wait gives 78; static HEFT, DLS and PEFT give 74; lower bound 54.
  - **Wide toy:** 289.0 → 286.5, vs static HEFT 264.5. Waiting for the fast slot helps the job at
    hand, but most of HEFT's edge there comes from planning *future* critical tasks onto fast cores,
    which no online rule reproduces.
  - **Cost:** it gives up the Graham greedy W/P + D guarantee. Our RESULTS show the big lever is
    placement (−27–31%), so this is the principled next step after E1 (LITERATURE.md §9 item 3).
- **Mapping:**
  - `JobSpec.work: Option<f64>`, in reference-speed seconds.
  - The DAG layer fills it from the node's `work` in `submit_node` (dag.rs:596-606) when the caller
    has not.
  - The engine's `Running` (engine.rs:151-155) gains `started: Instant, work: Option<f64>`.
  - `SpeedPolicy::EarliestFinish { wait_for_faster: bool, max_wait: f64 }`.
  - The `dispatch` None-branch (engine.rs:641-650) must not call `try_reserve` for a job that is
    *waiting by choice* (it is admitted somewhere).
  - `explain` gains "waiting for worker W (est. free at T)".
- **Guards:**
  - Overdue running jobs (`started + work/speed < now`) count as "unknown", so no waiting on them.
  - `max_wait` bounds voluntary waiting.
  - Jobs without `work` behave as E1.

### E3. DLS dynamic level as a class-aware priority — ENGINE (optional) or SIMULATOR experiment
- **Source:** dls.rs:47-55 (median flop time), :106-108 (`Δ = w·(median_ft − 1/speed_r)`), :109
  (global argmax over ready × resources).
- **Online with free slots** (`EST = now`), it reduces to: when a slot of class `c` frees, give it to
  the ready job maximising `SL(t) + w_t·(1/s_med − 1/s_c)`.
  - **On a fast slot** the bonus favours big jobs.
  - **On a slow slot** big jobs are penalised.
  - This is the "long tasks fast, short tasks slow" rule that LITERATURE.md §1.2 derives from PEFT.
- **Mapping:**
  - Our engine has one queue with one key per job (engine.rs:115-120, 196). A per-class key needs one
    `BTreeMap<Key, JobId>` per speed class, built at submit from `priority − λ·work·δ_c`, and
    `dispatch` scanning the class queues of free workers.
  - **Effort:** 2-3 days. Only worth doing if E2 shows a residual gap.
  - **Cheaper first:** prototype in the simulator as a fixed per-job class *preference* (top-q jobs
    by `w·δ` prefer fast; the rest prefer slow and overflow) using `JobSpec.prefer` / `class`.
- **Where:** simulator first.

### E4. Rank variants — DAG layer: document, do not implement
- HEFT rank = our upward rank (dag.rs:541-556; DagTemplate::critical_path dag.rs:187), up to the
  constant `avg_flop_time`.
- DLS SL = the same at the median speed.
- PEFT rank (no comm) = rank − own work, at the fastest speed. It is a strictly different order only
  through the "exclude self" term.
  - **[measured]** PEFT (284.0) is worse than HEFT (264.5) on the wide toy, and ties on test_4.
- **dslab recomputes ranks at every completion** (dynamic_list.rs:134), but from static flops, so
  the recomputation is a no-op. It does not address our "frozen ranks" concern, which needs
  `update_work` on submitted jobs.
- **Action:** add a README note mapping our `rank_priority` to HEFT `rank_u` (and to DynamicList
  `BottomLevel`). No code.

### E5. Static insertion-based HEFT plan as an offline baseline — SIMULATOR only (optional)
- **Source:**
  - common.rs:180-261 (per-core `BTreeSet<ScheduledTask>` gaps, plus a memory treap for interval max,
    treap.rs);
  - heft.rs:62-121;
  - runner per-core FIFO execution (runner.rs:396-482).
- **Use:** on a materialised sub-region of the whole-run DAG, plan with estimated costs and execute
  with true costs, keeping the per-slot order. It reports "what perfect global planning would buy"
  next to our online plans. It is an upper reference for E2 (toy: 74 vs 78 vs 91).
- **Pitfalls:**
  - Rank ties must break topologically (quirk 1).
  - Execution-order rigidity under noise makes it optimistic only with oracle costs.
  - It needs the whole sub-DAG up front: no lazy templates.
  - Cost: O(N·R·cores) evaluations. A whole 3.9M-task run is borderline (about 1e9 gap probes), so
    restrict it to regions.

### E6. Metrics — SIMULATOR (trivial)
- **Source:** run_stats.rs:9-46, 109-157.
- **Metrics:**
  - `cpu_utilization_active`: busy over each worker's first-to-last use window;
  - `used_resource_count`;
  - `max_used_cores`.
- **Use:** cheap additions to `WholeMetrics` (whole.rs:863-884) that show whether slow workers idle
  at the tail under fast-first.

### What does NOT transfer, and why
- **Network, data items and transfer modes** (data_item.rs, network.rs, common.rs:89-157). Our edges
  carry no data cost: coordinator-mediated state, not modelled. With zero sizes, all of it vanishes,
  and PEFT degenerates as described in E4.
- **Static whole-DAG planning:** HEFT, PEFT, DLS and Lookahead are all `is_static() = true`
  (heft.rs:146).
  - They need every task and cost at `start`. Our DAG unfolds lazily (templates at bidegree open),
    costs are estimates with log-sd 0.6, and workers join and leave.
  - Executing a plan rigidly is brittle: the deadlock in quirk 1, and no recovery when a task overruns.
  - They do not scale: PEFT and DLS are O(N²); full Lookahead is about O(N²R²).
  - **Only their resource-choice rule transfers** (E2, E3).
- **Moldable tasks** (`CoresCriterion`, `cores_dependency`, `min/max_cores`). Our tasks take one slot.
  - The portfolio shows the core-count choice dominates their results: every `Efficiency*` variant
    is 1.9-4.9× worse than `MaxCores`. That is irrelevant to us.
- **Memory model:** a hard additive capacity (runner.rs:417, 443-453) with no baseline, reported RSS
  or escape hatch. Ours (`Admission`, admission.rs:46-89) is richer. dslab's `DotProduct` and
  `RankPack*` packing scores could be an alternative to `Tightest`, but the RESULTS show placement
  packing is not our bottleneck.
- **`TaskData` locality:** it already exists as `JobSpec::prefer`.
- **No arrival or FIFO order:** dslab has no group-arrival criterion at all. Our best order (oldest
  group first) has no dslab counterpart. `Simple`'s ascending-id order is the closest (see §3).

### Which DynamicList combinations matter for us
Our setting: span-bound (D = 883 h > W/P = 770 h), 2 speed classes (L40S 2.41× H200 per job), PS with
α = 1, one slot per task, memory not modelled in the whole-run sim.

| dslab combination | our equivalent | relevance |
|---|---|---|
| `Simple`, fast resources listed first | Greedy / group order + fast first (today: `prefer`) | **High.** Closest to our best plan (1,237 h); cross-validation anchor |
| `Simple`, slow resources first | speed-oblivious first-fit | Shows the speed effect (toy: 236 vs 100) |
| `BottomLevel × Speed` | `rank_priority` + fast first (no aging) | **High.** Our "DAG rank … fast first" plans |
| `CompSize × Speed` | `priority = −work` + fast first | Medium. A "big jobs first" baseline (best in dslab's own portfolio: 1.151 vs BottomLevel 1.184 avg ratio); cheap to try via `JobSpec.priority` |
| `* × MaxAvailableCores` | LeastLoaded (by free slots, then speed) | Medium. Our speed-oblivious arm, but dslab breaks ties by speed and we break them by id |
| `* × MinAvailableMemory` | BestFit (`Tightest`) | Low (memory off in whole-run) |
| `* × TaskData` | `prefer` affinity | None (no data) |
| `RankPack1` (by resource, fastest first) | the same as `BottomLevel × Speed` absent memory | Low: duplicate |
| `RankPack0/…`, `DotProduct*`, Cores/Memory criteria, `Efficiency*` | – | None (moldable / packing) |

In dslab's own portfolio (results.json, 32 instances) `Speed` and `MaxAvailableCores` tie exactly.
`MaxCores` makes every node all-or-nothing, so availability ties and the speed tie-break decides. Do
not read that as "speed does not matter".

---

## 3. Cross-validation plan

**Goal.**
- **Exact agreement** where the models coincide: this validates the DAG we build, the critical path
  and the work/speed arithmetic.
- **Bounded disagreement** where only tie-breaks differ.
- **A reference** for what static planning buys.

**Model alignment** (each needs only configuration, not semantics changes):
1. **Network.** Every data item has size 0; `ConstantBandwidth{bandwidth: 1e9, latency: 0}`;
   `DataTransferMode::Direct`.
2. **Speeds.**
   - One dslab resource per worker process, `cores = slots (16)`, `memory = 0`; every task has
     `memory = 0` and `min_cores = max_cores = 1`.
   - H200 speed `10`, L40S speed `24.13` (= 10 × 2.413); every task's `flops = 10 × true work`
     (H200-seconds). Durations are then in seconds, and every real speed stays above the auto-master's
     1 (quirk 3).
3. **Processor sharing.** Our sim's per-job rate is `f(k)/k` = speed for `k ≤ k_sat`. The fitted H200
   `k_sat = 15` differs from exclusive cores only at k = 16. For exact runs, pass `sched-whole` a
   `PsModel` with `k_sat = 16` for both classes. That needs a CLI switch (`--linear-ps`), since the
   model is fitted in `sched_whole.rs:93`.
4. **Joins.**
   - Our passthroughs take zero time and no slot. In dslab every node is a task that occupies a core.
   - **Plan:** fold "walk done" (4k+2) into "registered" (4k+1) and emit it as one join task
     `r_k` with `flops = 1e-6`. Emit dead signatures (sig_work = 0, which are passthroughs in
     whole.rs) as 1e-6 tasks too.
   - **Emit tasks in topological order** (bidegrees by (t, s), then template topological order),
     because of quirk 1.
   - **Residual difference:** under full contention a dslab join waits for a free core (up to one
     task duration), especially under `CompSize`. Expect dslab ≥ ours by that much.
5. **Edges, one data item per producer** consumed by all successors:
   - `compute_deps(k)` → `z_k` (whole.rs:704-722);
   - `z_k` → template sources (`declare_template(..., &[zero])`, whole.rs:1061);
   - template edges (`info.template.successors`, dag.rs:129);
   - template sinks and `z_k` → `r_k`;
   - `r_{(s,t−1)}` → `r_k`.

**Export.**
- **Function:** `sim::export::dslab(world, max_n, max_s, truth: bool) -> (String /*dag yaml*/,
  String /*system yaml*/)`, called from `sched-whole --export-dslab DIR --fleet …`.
- **Region:** pick it with `World::summary` so the sub-DAG has **≤ 5k tasks**. dslab's dynamic list is
  O(N·(V+E)), and PEFT/DLS are O(N²). Use Lookahead only at ≤ 1k tasks.
- **What to export:** use `truth = true` costs. dslab uses the same flops for planning and execution.
- **Plan-with-estimates variant:** run static schedulers on the estimate DAG and replay their actions
  on the truth DAG through a 30-line `ReplayScheduler` in a driver crate under `refs/` (outside our
  crate).

**Runs.**
- **dslab side:** `refs/target/release/dag-demo -d dag.yaml -s system.yaml` (prints all 47 configs).
  Add `DynamicList[task=RankPack1,resource=Speed]` via a small driver if wanted.
- **Our side:** `sched-whole --max-n N --max-s S --fleet l40s:2:16,h200:1:16 --linear-ps --plans
  group,group+fast,rank-oracle+fast`.
- **Fleet:** a scaled fleet (e.g. 2 L40S + 1 H200, 16 slots each) keeps contention comparable to
  production. Also run the production fleet.

**Comparisons and expected agreement.**

| check | ours | dslab | expected |
|---|---|---|---|
| G1: unlimited capacity (one L40S resource with 10⁵ cores; `--fleet l40s:1:100000`) | any plan with fast first | any DynamicList `×Speed`; HEFT | **Exact** (≤ 1e-9 rel.), both = D_fast, which also equals `World::bounds().0` and dslab's `makespan_lower_bound` critical-path term. Validates the DAG and the costs |
| G2: one worker, one slot | any | any | **Exact:** Σ work / speed (+ joins' 1e-6) |
| G3: no-comm test_4 ported as a unit test | rank + FastestFirst on 4 single-slot workers (speeds 1, 2, 4, 4) | `BottomLevel×Speed` = 91, `CompSize×Speed` = 90, HEFT = DLS = PEFT = 74, Lookahead 73 | **Exact 91** if ready-set tie order matches (id order; our DAG layer sorts dependents by id, dag.rs:658, and the rank must not round to ties under `rank_scale`). EFT+wait: 78 per my simulation |
| C1: contention | group + fast first | `Simple` with L40S resources listed first | Within a few % (2-5%). Our order is (group first arrival, FIFO seq) vs dslab's ascending id; our worker tie-break is least-loaded vs lowest index (makespan-neutral under α = 1). Closed-loop makespans move ±1.7% under 1 s perturbations (RESULTS.md), so compare over ≥ 5 seeds/perturbations |
| C2: contention | rank-oracle, no aging, fast first | `BottomLevel×Speed` | Within a few %; plus integer rounding of ranks (dag.rs:604) and `rank_epsilon` |
| C3: speed-oblivious | group order (LeastLoaded) | `MaxAvailableCores` (ties → speed) / `Simple` with H200 first | Only qualitative (the bracket around our number) |
| R1: static reference | – | HEFT / PEFT / DLS / Lookahead | 5-15% below the dynamic plans on oracle costs (toy: 264.5 vs 289.0). It is a target for E2, not an agreement check |

A mismatch in G1 or G2 means a DAG or export bug. A C1 or C2 gap above about 5% that persists over
seeds means a semantics difference to chase: join slots first, then rank ties.

---

## 4. Implementation plan

Ordered. Effort is focused developer time. Every step keeps `cargo test` (proptest is already a
dev-dependency) and `just lint` green. Simulator steps are behind the `sim` feature.

| # | step | files | effort |
|---|---|---|---|
| 1 | `WorkerState.speed` + `SpeedPolicy::FastestFirst` | lib.rs, engine.rs | 0.5 d |
| 2 | Sim uses native speed; delete the `prefer` hack; regression | sim/whole.rs, bin/sched_whole.rs | 0.5 d |
| 3 | `JobSpec.work` + DAG layer fills it + `Running.started/work` | lib.rs, dag.rs, engine.rs | 0.5 d |
| 4 | `SpeedPolicy::EarliestFinish{wait_for_faster, max_wait}` | engine.rs | 1.5-2 d |
| 5 | Sim plans `+eft`, `+eft-wait`, estimated vs oracle work; RESULTS rows | sim/whole.rs, bin/sched_whole.rs, RESULTS.md | 0.5 d + runs (≈8 min/plan) |
| 6 | dslab exporter + `--linear-ps` + driver; G1-G3, C1-C2 | sim/export.rs (new), sim/mod.rs, bin/sched_whole.rs; driver under refs/ | 1-1.5 d |
| 7 | (optional) DLS class-aware key: sim prototype, then engine | sim/whole.rs; engine.rs | 1 d sim; +2-3 d engine |
| 8 | (optional) offline insertion-HEFT baseline on sub-regions | sim/heft.rs (new) | 2-3 d |
| 9 | Docs: README model/policy table, HEFT/PEFT equivalence note | README.md, LITERATURE.md §7 | 0.25 d |

### Step 1: speed and FastestFirst
API:
```rust
// lib.rs
pub struct WorkerState { /* … */ pub speed: f64 }        // per-job speed relative to the reference; new() sets 1.0
pub struct WorkerLoad  { /* … */ pub speed: f64 }

// engine.rs
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SpeedPolicy {
    #[default] Oblivious,
    /// Fastest admitting worker first (dslab DynamicList `resource=Speed`); then the policy's own choice.
    FastestFirst,
    /// Online HEFT: least estimated finish time; optionally wait for a busy faster worker.
    EarliestFinish { wait_for_faster: bool, max_wait: f64 },
}
pub struct BackfillConfig { /* … */ pub speed: SpeedPolicy }   // BestFit and Lanes inherit it via .backfill
pub struct GreedyConfig   { pub speed: SpeedPolicy }
```
- **Engine:** `Mode.speed`; `choose` score `(lane_rank, speed_key, fit, !preferred, running)`, with
  `speed_key` a total-ordered `Reverse(speed)`. Use `f64::total_cmp` via a small newtype or bit trick.
  Never compare NaN.
- **Validation:** reject `speed ≤ 0` / NaN in `worker_update` by clamping to `f64::MIN_POSITIVE`, and
  document it.
- **Tests:**
  - **Unit:** a fast worker beats a less-loaded slow one; equal speeds fall back to least loaded; under
    BestFit, the tightest fit *within* the fastest class; lanes still route big jobs first.
  - **Property (proptest), random workers, speeds and jobs:** after `dispatch`, no placed job sits on a
    worker slower than some eligible worker that still admitted it at that moment. Re-check admission
    on a cloned engine.
  - **Property:** `Oblivious` reproduces today's placements bit for bit (an equivalence harness over
    random event sequences against a frozen copy of the old `choose`).
- **Risk:** adding a pub field breaks struct-literal construction of `WorkerState` downstream. It is
  pre-1.0, `..WorkerState::new` is the documented idiom, and the `serde` default is needed
  (`#[serde(default = "one")]`) so old snapshots still load.

### Step 2: simulator on native speed
- **Change:** `simulate` sets `WorkerState.speed = model.throughput(class, 1)` and uses
  `BackfillConfig.speed = FastestFirst` when `fast_first` (replacing whole.rs:982-992)
  (`sim/run.rs` has no fast-first arm: grep finds no "fast").
- **Regression:** the Greedy/PriorityBackfill makespans in RESULTS.md "whole run" must be
  **identical**. For two classes, `(speed desc, running)` equals `(!preferred, running)`. Run
  `today+fast`, `group+fast` and `rank+fast` at reduced `--max-n` in a test and compare to the
  prefer-based results computed in the same test.

### Step 3: work on jobs
- **API:** `JobSpec { …, pub work: Option<f64> }` ("estimated run time on a speed-1 worker, seconds";
  used only by `EarliestFinish`).
- **DAG layer:** `submit_node` (dag.rs:596-606) sets `spec.work.get_or_insert(node.work)`, except for
  passthroughs.
- **Engine:** `Running { worker, demand, started: Instant, work: Option<f64> }`.
- **Tests:** a DAG job's work reaches the policy; `update_work` before submission is reflected;
  snapshots keep it (serde default `None`).

### Step 4: EarliestFinish
- **Algorithm** (per job in the scan, inside `choose` and its caller):
  ```text
  eta(w)  = now                                    if w admits job
          = min over running j on w with work: max(now, started_j + work_j / speed_w)
            (skip w if any running job lacks work, or the min is overdue: started + work/speed < now)
  eft(w)  = eta(w) + job.work / speed_w            (job.work None → behave as FastestFirst)
  best_ad = argmin eft over admitting workers      (ties: existing tuple)
  best_all= argmin eft over eligible workers whose reservation/lane checks pass ignoring slots
  if wait_for_faster and best_all is busy and eft(best_all) < eft(best_ad)
       and now − since(job) < max_wait            → job waits (no placement, no try_reserve)
  else place on best_ad
  ```
- **Dispatch change** (engine.rs:628-650): distinguish "refused everywhere" (→ `try_reserve`) from
  "waiting by choice" (→ continue the scan). Keep the `hopeful`/`bound` fast path unchanged: it only
  skips jobs that are refused everywhere.
- **`explain`:** "waiting for faster worker W (est. free in X s, EFT Y vs Z on W')".
- **Stats:** `PolicyStats.waiting_for_faster: usize`.
- **Tests:**
  - **Unit:** the three cases (fast frees soon → wait; fast frees late → slow; overdue → slow).
  - **Unit:** `max_wait` expiry.
  - **Unit:** with `wait_for_faster = false`, identical to FastestFirst under α = 1.
  - **Golden:** the no-comm test_4 (step 6) gives 91 under FastestFirst + rank and 78 under EFT+wait
    (my python, `refs/xval/dynsim.py`). Encode it as a tiny event loop in the test.
  - **Property:** liveness. Every job is placed within `max_wait` + one running time after it becomes
    the most urgent.
  - **Property:** the priority invariant still holds for placements; waiting by choice is the only
    exception.
- **Risks:**
  - It loses the greedy W/P + D bound, so a bad `work` estimate can idle slow slots. `max_wait` and the
    overdue rule bound the damage.
  - Estimates have log-sd 0.6, so `eta` is noisy. Evaluate under estimated work, not just oracle.
  - Cost: O(workers × running) per job per scan. Cache per-worker `eta` once per `dispatch`, since
    placements only add jobs that start now.

### Step 5: simulator arms
- **Plans:** `group+eft`, `group+eft-wait`, `rank-oracle+eft-wait`, with work from estimates and
  from truth.
- **Expectation:** on the toys, wait recovers most of static HEFT's gain on small DAGs (91 → 78 vs 74)
  but little on wider ones (289.0 → 286.5 vs 264.5). The whole-run effect is unknown; hence this step.

### Step 6: cross-validation
- **Exporter:** as in §3, `pub fn dslab_yaml(world: &World, region: (i32, i32), truth: bool, fleet:
  &Fleet) -> (String, String)`.
  - It writes YAML by hand with `format!`, so no serde_yaml dependency.
  - It needs read access to `World`'s private `bideg`/`profiles`/`offsets`, so it lives in
    `sim::whole` or a sibling `pub(super)` module.
- **`--linear-ps`:** overrides the fitted `k_sat` to the slot count.
- **Tests (sim feature):**
  - **G1:** the export of a 2×3 toy world gives a task count of tasks+joins, and the DAG yaml
    critical path equals `World::bounds().0`. Compute it in Rust over the exported structure; no dslab
    is needed in CI.
  - dslab runs stay manual and out of CI, recorded in RESULTS.md.

### Risks overall
- **dslab is unmaintained-looking** (last commit 2024-07) and uses old simcore 0.1. It builds today
  with cargo 1.97 **[measured]**; pin `Cargo.lock` in the driver.
- **Join-slot semantics** (§3, item 4) are the main source of C1/C2 disagreement. If they dominate,
  give the driver a `sync` resource with `Only` restrictions (programmatic DAG, not YAML).
  - That breaks PEFT's OCT (min over all resources, quirk 3), so use it only for the dynamic and
    Simple runs.
- **Scope creep from static planners:** keep E5 optional. The online rule (E2) is what the
  coordinator can run.

### Uncertainties
- I did not read `sim/run.rs` or `sim/trace.rs` in detail (a grep shows no fast-first arm in run.rs, so
  step 2 should only touch whole.rs).
- I did not verify the A(3) template sizes for a ≤5k-task region; use `World::summary` to choose the
  region.
- The 78 (test_4) and 286.5 (wide toy) EFT-wait figures come from my python re-implementation
  (`refs/xval/dynsim.py`), not dslab. That same script reproduces dslab's non-idling numbers exactly
  (91.00, 289.03).
