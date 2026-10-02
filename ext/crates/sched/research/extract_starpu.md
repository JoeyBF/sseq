# StarPU: what to extract into `sched`

Source: shallow clone of https://github.com/starpu-runtime/starpu at
`61b42c39417473a3617e4b3edb5e31e423662ddf` (2026-10-01), in
`/wsu/home/hd/hd72/hd7264/.claude/jobs/71ebdf0b/tmp/refs/starpu`. All `file:line` references
below are relative to that root unless they start with `sched/`, which means
`ext/crates/sched/` in the worktree. The paper material (spoliation) comes from the survey
Beaumont et al., "Scheduling on Two Types of Resources: A Survey", arXiv:1909.11365, §5.1.3,
Algorithm 2, and §7.2.1. I extracted its text locally to `refs/survey.txt`.

Markers: **[code]** = read in StarPU source. **[paper]** = from the survey text.
**[U]** = uncertain or my judgement.

---

## 0. Headline findings

1. **StarPU does not implement spoliation.** No `spoli*`, preempt or abort path exists in
   `src/` (I grepped the whole tree). Heteroprio's "steal" (`src/sched_policies/heteroprio.c:3649-3707`)
   takes a *not-yet-started* task from the prefetch queue of another worker **of the same arch**.
   It never touches a running task. Spoliation exists only in the paper (Algorithm 2), so our API
   for it has to be designed from scratch (§2.5).
2. **The HeteroPrio part that maps onto our two-class fleet is the "slow factor" gate**
   (`heteroprio.c:3602-3606`). A worker of a slow arch takes a task from a bucket only if
   `bucket_ntasks / n_workers_of_fastest_arch >= slow_factor[arch]`. Put simply, the slow class
   joins only when the fast class could not drain the backlog within one slow-task time. This is
   a cheap, estimate-free replacement for "fast first, slow takes the overflow".
3. **HeteroPrio's acceleration buckets degenerate for us.** Our speed ratio is roughly uniform
   (2.41x for every task), so every task falls in one bucket (`component_heteroprio.c:33,425-437`).
   The CPU-front / GPU-back ordering then has nothing to sort. The same holds for auto-heteroprio's
   per-codelet scoring: there is one "codelet".
4. **dmda is EFT over per-worker queues, decided at submit time.** Its multi-slot formula
   (`component_sched.c:706-734`) and its "count only higher-priority queued work"
   (`deque_modeling_policy_data_aware.c:474-487`) translate directly into an `EarliestFinish`
   worker choice with a per-dispatch *virtual queue*. That queue is what makes "wait for a fast
   slot" safe from starving the slow class.
5. **The perf-model machinery is a running mean per (task footprint, arch).** It has a calibration
   minimum of 10 and *rejects samples that deviate by more than 50%*
   (`perfmodel_history.c:1945-1966`, `configure.ac:3480-3483`). With our per-signature log-sd of
   0.60 this outlier rule would discard most samples, so **do not copy it**. What transfers is a
   per-class speed learned in log space from `completed` durations, and optionally a per-group
   correction factor.

---

## 1. How the StarPU pieces work

### 1.1 dm / dmda / dmdap / dmdas / dmdasd / dmdar (`src/sched_policies/deque_modeling_policy_data_aware.c`)

**Structure.** There is one FIFO per worker, `struct starpu_st_fifo_taskq` (`fifo_queues.c:49-69`),
with these fields:
- `exp_start`: when the worker frees up.
- `exp_len`: predicted work queued, not yet started.
- `exp_end = exp_start + exp_len`.
- `pipeline_len`: predicted work already popped (transferring or running).
- `ntasks`, `pipeline_ntasks`.
- Optionally `exp_len_per_priority[]` and `ntasks_per_priority[]`.

The decision is made **once, at push time**: the task goes into one worker's queue and is never
reconsidered. Workers only pop locally (`_dmda_pop_task`, `:221-259`).

**Expected end per worker** (`compute_all_performance_predictions`, `:411-602`):
```
exp_start(w)   = isnan(fifo.exp_start) ? now + pipeline_len : max(fifo.exp_start, now)      :457
prev_exp_len   = fifo.exp_len                     (dm, dmda, dmdar, dmdas)                   :471
               = exp_len_per_priority[prio(task)] (dmdasd: only work of >= priority)         :474-481
               = Σ predicted(tasks ahead in the sorted list) + pipeline_len
                                                  (dmdasd without bounded prios,             :484
                                                   fifo_queues.c:171-222)
start(w)       = exp_start + prev_exp_len; with data awareness,
                 max(start, now + transfer(w))                                               :568-571
exp_end(w)     = start(w) + length(task, arch(w))                                            :573
min_exp_end    = min_w exp_end(w);   max_exp_end_of_workers = max_w (exp_start + prev_exp_len)  :489-491,575-580
```

**Fitness and choice** (`:652-710`):
```
fitness(w) = α·(exp_end(w) − min_exp_end) + β·transfer(w) + γ·energy(w)                      :680-682
           + γ·idle_power·(exp_end(w) − max_exp_end_of_workers)/1e6   if exp_end(w) > max_exp_end   :686-694
```
Without data awareness (`dm`), `fitness = exp_end − min_exp_end` (`:684`). The policy picks
argmin fitness. **Tie-break:** strict `<` over the worker iteration order, so the first worker in
collection order wins (`:696`).

Defaults are α=1, β=1, γ=1000, idle_power=0 (`:156-158`, `:864-873`). Environment variables
`STARPU_SCHED_ALPHA/BETA/GAMMA` and `STARPU_IDLE_POWER` override them, and runtime knobs multiply
them (`:54-63,133-139`).

**Uncalibrated fallback** (`:516-563`):
```
ntasks_end(w) = (fifo.ntasks + fifo.pipeline_ntasks) / relative_speedup(arch(w))
```
Here `relative_speedup = Σ_dev alpha(arch)·ncores` (`perfmodel.c:138-149`). If any worker's
prediction is NaN or 0 (`unknown`), the task is *forced* to the worker with the smallest
`ntasks_end`. A worker where the model is uncalibrated is preferred, so that it gets calibrated
(`:540,545`). That forced choice is `forced_best` (`:590,713-722`).

**Bookkeeping after push** (`push_task_on_best_worker`, `:271-408`):
- `exp_start = max(exp_start, now)`.
- The transfer counts only if it outlasts the queue: `predicted_transfer = max(0, now + tr − exp_end)` (`:309-320`).
- `exp_len += transfer + predicted` (`:322-346`), and the same is added to `exp_len_per_priority[0..=prio]`.
- At pop, transfer moves from `exp_len` to `pipeline_len` (`:161-179`). At pre-exec, the
  prediction moves too (`:182-205,939-957`). At post-exec, it leaves `pipeline_len` and
  `exp_start = max(now + pipeline_len, exp_start)` (`:208-219,1051-1059`).

**Variants** (`:1061-1167`):

| policy | push | worker queue | pop |
|---|---|---|---|
| `dm` | `da=0` | FIFO | local pop |
| `dmda` | `da=1` | FIFO | local pop |
| `dmdap` | `da=1` | sorted by priority (`fifo_queues.c:224-291`, larger = more urgent, FIFO among equals) | local pop |
| `dmdas` | as `dmdap` | sorted | `pop_first_ready_task`: the first task whose data is already local, within priority |
| `dmdasd` | as `dmdas`, plus `sorted_decision=1` (EFT counts only queued work of ≥ priority) | sorted | as `dmdas` |
| `dmdar` | `dmda` | FIFO | ready-first |

Priorities are normalised with `starpu_st_normalize_prio = ((num_prios−1)/(max−min))·(prio−min)`
(`fifo_queues.c:371-376`). **Note:** that is integer division, so with the defaults
`INT_MIN..INT_MAX` it collapses to a single level [code; I did not trace whether that is
intended].

### 1.2 Modular schedulers (component framework)

A tree of components exchanges `push_task` / `can_push` / `pull_task` and estimates through
`estimated_end` and `estimated_load`.

**Multi-slot expected end** (`component_sched.c:706-734`):
```
end_min_add(C, L) = min_i end_i + (L + Σ_i (end_i − min_i end_i)) / |workers(C)|
```
This answers "when could a group of k parallel servers with ends `end_i` and backlog `L` start
one more task?". A worker's own `estimated_end` is `max(now, exp_start) + exp_len`
(`component_worker.c:485-495`). For a prio queue it is `end_min_add(component, queue.exp_len)`
(`component_prio.c:42-47`).

**Per-worker queue caps** (`component_prio.c:97-140`):
- A push is refused when `ntasks >= ntasks_threshold` or `exp_len + predicted >= exp_len_threshold`.
- A refused task stays in the window (`FIFO_ABOVE`) and is retried on `can_push`, after a worker
  pops.
- Defaults (`modular_ez.c:49-51,312-342`): `ntasks_threshold = 30` for heft, mct and heteroprio
  ("need more queueing to allow CPUs to take some share of the work"); 2 for others; `UINT_MAX`
  with `NOLIMIT`, which is the modular dmda family (`modular_heft.c:80-180`).
  `exp_len_threshold = 1e9 µs`. Environment variables `STARPU_NTASKS_THRESHOLD` and
  `STARPU_EXP_LEN_THRESHOLD` override them.
- **This is how modular-heft "waits for a fast slot":** the decision is EFT, but the commitment
  is bounded by the queue cap.

**MCT** (`component_mct.c:27-117`, `helper_mct.c`):
- Per child: `estimated_end = max(child.estimated_end, now)` (`helper_mct.c:154-156`).
- `exp_end_with_task = end + max(0, transfer − (end − now)) + length` (`helper_mct.c:63-88`).
- `fitness = α(exp_end − min) + β·transfer + γ·energy` plus the same idle-power term
  (`helper_mct.c:90-110`). Argmin, first wins on ties (`:199-214`).
- `max_exp_end_of_workers` here is the max of the children's *current* ends (`:169-170`), which
  differs slightly from dmda.

**HEFT component** (`component_heft.c:41-166`):
- Pops the top task and up to `NTASKS = 5` more tasks of the *same* priority (`:32,50-62`).
- Computes min-EFT for each and schedules the one with the **smallest min-EFT** (`:115-127`).
- Pushes the others back to the front (`:131-136`), then applies the MCT choice to the chosen
  one (`:140`).

**Component heteroprio** (`component_heteroprio.c`):
- On push, `acceleration = max_arch(min_impl expected) / min_arch(...)` (`:383-426`).
- The task goes into the bucket whose accel is within ±10% (`APPROX=0.10`, `:33,433-437`).
  Otherwise a new bucket is inserted, keeping the buckets sorted by decreasing accel (`:440-477`).
- Tasks that not every arch can run go to `no_accel` and get plain MCT (`:489-498,242-321`).
- Progress (`:323-358`): CUDA, HIP and OpenCL pop from the *most* accelerated bucket (`front=1`);
  MPI-SC and CPU from the *least* (`front=0`).
- The popped task then goes through MCT. If MCT's best child has no worker of the popping arch,
  the task goes back (`:192-239`).
- `modular-heteroprio-heft` stacks this on top of heft (`modular_heteroprio_heft.c:22-46`).

### 1.3 Heteroprio, the monolithic policy (`src/sched_policies/heteroprio.c`)

**Data.**
- Up to `HETEROPRIO_MAX_PRIO = 100` buckets (`heteroprio.h:23`). A bucket holds a task list,
  `valid_archs`, `slow_factors_per_index[arch]` and `factor_base_arch_index` (the fastest arch)
  (`:151-170`).
- Per arch: `nb_prio_per_arch_index` and the mapping `prio_mapping_per_arch_index[arch][i] → bucket`
  (`:303-306`). Each arch visits buckets in its own order: this is the "one priority per
  processing-unit type" (`doc/doxygen/chapters/starpu_extensions/advanced_scheduling.doxy:201-208`).
- Counters `nb_remaining_tasks_per_arch_index` and `total_tasks_in_buckets` (`:307-310`).
- One worker = one slot. Each worker has a local prefetch deque of `STARPU_HETEROPRIO_MAX_PREFETCH = 2`
  (`include/schedulers/starpu_heteroprio.h:32`).

**Push** (`:3175-3450`). The bucket is `task->priority`, or the auto-priority (§1.4). The task is
pushed at the *front* of the bucket list (`:3381`) and the per-arch counters are incremented
(`:3387-3397`).

**Pop** (`:3452-3789`, non-LA path `:3571-3709`):
1. **Prefetch target** = `MAX_PREFETCH − local_queue`, capped by remaining tasks. If fewer
   remaining tasks than workers: take 1 only if the local queue is empty (`:3578-3591`).
2. Visit the arch's buckets in mapping order (`:3595-3637`). Pop from the **front** of the bucket,
   so the bucket is LIFO, since push is also at the front [code: both ends are front]. Pop only
   while this holds:
   ```
   factor_base_arch_index == 0                      // fastest arch is CPU (enum 0): no gate
   || worker.arch == factor_base_arch_index         // I am the fastest arch
   || bucket.ntasks / nb_workers[fastest arch] >= slow_factor[my arch]                     :3602-3606
   ```
   **Note:** the `== 0` short-circuit means a bucket whose fastest arch is CPU
   (`STARPU_CPU_WORKER = 0`, `include/starpu_worker.h:50`) is open to everyone. This is
   arch-index punning, but harmless there [code].
3. Run from the local deque. If it is empty, **steal** from a same-arch worker's prefetch deque,
   circularly, starting after self (`:3643-3708`).

**Slow factors.**
- Manual: `set_faster_arch(arch, bucket)` sets `factor_base` and `slow_factor[arch] = 0`;
  `set_arch_slow_factor` sets the others (`:549-579`).
- Auto: `autoheteroprio_update_slowdown_data` sets fastest = argmin estimated time, and
  `slow_factor[a] = time[a] / time[fastest]` (`:3122-3172`).
- Initial auto values: every non-CPU arch gets slow_factor 1.0, with CPU fastest (`:1566-1571`).

**Hooks** (`:3791-3847`): accumulate per-arch busy and free time and the per-(arch, bucket)
execution time (auto mode only).

### 1.4 Auto-heteroprio (`heteroprio.c`, auto mode is ON by default: `STARPU_HETEROPRIO_USE_AUTO_CALIBRATION=1`, `:1432`)

- **Bucket = codelet** (perfmodel symbol or name) (`:2676-2727`).
- **Per-push statistics** (only if `!freeze_data_gathering`), each a running mean capped at
  `AUTOHETEROPRIO_RELEVANT_TASK_LIFE = 256` samples (`heteroprio.h:30`, `:2931-2979`):
  - `NOD = Σ_{succ} 1/indeg(succ)`: "normalised number of dependents" (`:2736-2777`).
  - `NRT[arch] = Σ_{succ} P(succ runs on arch)·normtime(succ, arch)/indeg(succ)`: "normalised
    released time" (`:2780-2841`), whose running mean is URT.
  - Successors' best-time sum (`:2897-2929`).
  - Overall proportion of each codelet (`:3021-3038`).
- **Execution side** (post hook): per-(arch, codelet) mean time, and proportion executed per arch
  (`:3069-3090`).
- `normtime(p, a) = est(p, a) / Σ_q prop(q)·best_est(q)` (`:2268-2284`).
- Unknown time:
  - `FAIR_TIME = 1000` if the arch can run it (`:2215-2218`);
  - otherwise `LONG_TIME = 1e8` (policy 0), or the best other arch's time (policy 1)
    (`:2220-2248`).
- **Re-ordering** every `priority_ordering_interval = 32` pushes (`:1442,3219-3233`) through
  `order_priorities` (`:2369-2640`):
  - For each codelet p and arch a, compare with the "worst" arch, or with the second worst if a
    *is* the worst (`:2385-2454`):
    - `archDiff = otherTime − ownTime`, `archRelDiff = otherTime/ownTime`;
    - `need_own = 1 − busy_proportion(a)`;
    - `URT = URT_own·need_own + URT_other·need_other`.
  - **Default score** (`STARPU_HETEROPRIO_URT_DOT_DIFF_4`, `:1436`), with `and4pond = 1.0` (`:1492`):
    ```
    score = (1 + URT)·archDiff − and4pond·ownTime·reLU(−archNeedDiff)          :2529-2532
    ```
  - 27 other score formulas exist (`:2476-2599`).
  - **Exploration bonus:** `+99999999` while there are fewer than `AUTOHETEROPRIO_RELEVANT_SAMPLE_SIZE = 16`
    time samples on that arch (`:2601-2605`, `heteroprio.h:32`).
  - Sort descending per arch (`:2610-2613`, comparator `:2362-2367`); the score order becomes
    the arch's bucket visit order (`:2615-2623`).
- **Persisted across runs** in a data file (`:1045-1388`).

### 1.5 Spoliation (paper only) [paper]

From survey Algorithm 2 (refs/survey.txt, lines 941-967), for independent tasks:
```
L ← tasks sorted by non-decreasing acceleration α_j = p_j(CPU)/p_j(GPU)
loop: t ← first time a resource i is idle
  if L non-empty: CPU pops head, GPU pops tail; start on i at t
  else:
    S ← {tasks assigned to other resources that would finish earlier if started on i at t}
    if S non-empty: j ← task of S with the HIGHEST finish time; unassign j, start it on i at t
    else stop
```

- **Graph version** (survey §7.2.1, lines 2443-2453):
  - Each idle resource takes a ready task by the HeteroPrio rule.
  - "If no task is ready, an idle GPU is allowed to spoliate a task from one of the CPUs if it can
    finish it earlier."
  - Priorities (HEFT-like ranks) break ties within one acceleration and "decide which task a GPU
    spoliates" [paper; the exact tie rule is in ref. [8], which I have not read: **[U]**].
- **Bounds:** ratio in [2 + 2/√3, 2 + √2] for independent tasks. Spoliation is what makes it
  bounded. Spoliated work is discarded ("T2 (aborted)", Fig. 3).

### 1.6 Performance models (`src/core/perfmodel/`)

**Types** (`perfmodel.c:199-244`):
- `PER_ARCH`, `PER_WORKER`, `COMMON`: user cost function, with `COMMON` divided by
  `relative_speedup` (`:151-164`);
- `HISTORY_BASED`, `REGRESSION_BASED`, `NL_REGRESSION_BASED`, `MULTIPLE_REGRESSION_BASED`.

**History** (`perfmodel_history.c`):
- A hash keyed by `(arch combination, impl, footprint)`. The footprint is a hash of the buffer
  sizes (`datawizard/footprint.c:45`).
- Each entry holds `sum`, `sum2`, `nsample`, `mean`, `deviation` and `nerror`.
- **Update** (`:1855-1992`):
  - The first sample of a new footprint is *discarded* for `HISTORY_BASED` (`:1924-1932`).
  - `local_dev = measured/mean`. If `100·local_dev > 100+H` or `100/local_dev > 100+H`
    (H = `STARPU_HISTORY_MAX_ERROR`, default **50**: `configure.ac:3477-3483`), the sample is
    counted as an error and *not* added. When `nerror >= nsample`, the entry is flushed
    (`:1945-1966`).
  - Otherwise `mean = sum/n` and `deviation = sqrt(|sum2 − sum²/n|/n)` (`:1969-1975`).
- **Use** (`:1740-1825`): the mean, only once `nsample >= _starpu_calibration_minimum`
  (`STARPU_CALIBRATE_MINIMUM`, default 10, `:120`). Otherwise NaN, and the scheduler forces
  calibration (§1.1).
- **Feed:** the driver measures pure execution time. **Failed tasks are not recorded**
  (`drivers/driver_common/driver_common.c:315-326`).

**Regression** (`:1994-2026`):
- Online log-log OLS: `ln t = ln α + β ln size`, with running Σlnx, Σlnx², Σlny, Σlnx·lny.
- Valid when `nsample >= 10` and `minx < 0.9·maxx` (`:66-67`).
- Prediction `α·size^β`, only for `0.9·minx <= size <= 1.1·maxx` (`:1599-1600`).
- **NL regression:** `a·size^b + c` within the same range, else the history mean (`:1643-1663`).
  Its fit lives in `regression.c` [not read in detail].

**Averages.** `starpu_task_expected_length_average` is the **harmonic** mean over workers
(`perfmodel.c:282-321`). It is used only by recursive tasks (`core/jobs_recursive.c:73`).

### 1.7 Priority handling, summarised

- StarPU priorities: **larger = more urgent** (`fifo_queues.c:251`). Ours is the opposite.
- `prio_deque` pops the highest priority, FIFO within a level (`prio_deque.c:62-79`).
- Heteroprio interprets priority as a *bucket id* (0..99), not an urgency, and each arch has its
  own visit order.
- No StarPU scheduler computes graph ranks automatically, except the test policy
  `graph_test_policy.c`, which uses depth or descendants after a full submission
  (`common/graph.c:382-470`). The application sets priorities.

---

## 2. What to extract, item by item

Throughout, "speed" means per-job service rate of a class relative to the reference class (H200 =
1.0, L40S = 2.413, `sched/RESULTS.md` service model). Work is in reference-seconds.

### 2.1 Work estimate on `JobSpec` (prerequisite)

- **Why:** EFT, the slow gate, spoliation and learning all need a cost per job. Today only
  `DagJob::work_estimate` has one (`sched/src/dag.rs:22`), and the engine never sees it.
- **API:** `JobSpec.work: Option<f64>` (reference-seconds; `None` = unknown) with `#[serde(default)]`.
  `DagScheduler::submit_node` (`sched/src/dag.rs:596-608`) fills it from the node's `work` when
  the spec has none.
- **Where:** `lib.rs` and the DAG layer.

### 2.2 `PerfModel`: online per-class speed (from §1.6 and the dmda calibration logic)

- **Algorithm.** For each completed job with `work = Some(w)`, `w > 0`, on class c, with duration
  `d = now − started`:
  - observe `x = ln(w/d)`;
  - keep per class a windowed running mean of x with sample cap `N` (StarPU's capped running mean,
    `heteroprio.c:2965-2979`, `N = 256` default), plus the sample count;
  - `speed(c) = exp(mean_x)` once `count >= calibration_minimum` (default 10,
    `perfmodel_history.c:120`); before that, use the configured prior `prior[c]` (default 1.0).
- **Why log space, and no 50% outlier rejection:** our per-signature noise is log-normal with sd
  0.60 (`sched/RESULTS.md`, "Signature split"). StarPU's ±50% rule (`perfmodel_history.c:1945-1966`)
  would reject roughly 50% of the samples and then flush. The geometric mean of `w/d` is the
  maximum-likelihood speed under log-normal noise [U: assumes estimate errors are independent of
  class].
- **What is not recorded:**
  - failures, cancellations and preempted runs (as `driver_common.c:315`);
  - jobs with `work = None`.
- **Processor sharing:** the fitted model has per-job speed independent of k up to `k_sat`
  (`sched/RESULTS.md`: "A job's own speed does not depend on how many others share its worker"),
  so d is comparable across loads. Note that H200 `k_sat = 15 < 16` slots: a 16th job runs at
  15/16 [code: `sched/src/sim/model.rs` `ClassCurve::shape`]. Accept that bias, or record k at
  start [U].
- **Optional per-group correction** (the footprint idea, keyed by `group`):
  - `f_g = exp( n/(n+n0) · mean_g(ln(d·speed(c)/w)) )` with shrinkage `n0 = 5`.
  - The engine multiplies the remaining waiting jobs' w by `f_g` when computing EFT.
  - The DAG layer may apply it to ranks through `update_work` [U: worth it only if within-group
    errors correlate; the census level spread of sd 0.31 per bidegree suggests they do].
- **API:**
  ```rust
  pub struct PerfModelConfig { pub prior: BTreeMap<String, f64>, pub calibration_minimum: u32,
                               pub window: u32, pub learn: bool, pub group_shrinkage: Option<f64> }
  pub struct PerfModel { /* BTreeMap<String, ClassStat>, BTreeMap<u64, GroupStat> */ }
  impl PerfModel {
      pub fn new(cfg: PerfModelConfig) -> Self;
      pub fn speed(&self, class: &str) -> f64;              // prior until calibrated
      pub fn calibrated(&self, class: &str) -> bool;
      pub fn observe(&mut self, class: &str, group: u64, work: f64, duration: f64);
      pub fn expected(&self, class: &str, group: u64, work: f64) -> f64; // seconds on class
  }
  ```
  New trait method, provided so it does not break callers:
  `fn failed(&mut self, job: JobId, now: Instant) { self.completed(job, now) }`. Engine
  policies override it to skip learning. `PolicyStats` gains `speeds: Vec<(String, f64, u32)>`.
- **Where:** a new `src/perf.rs` (pure). The engine owns one instance (it knows each running
  job's start and class). `Running` gains `started: Instant`, `work: Option<f64>` and `class`
  (`sched/src/engine.rs:151-155`).

### 2.3 `Choice::EarliestFinish`: dmda/MCT worker choice with a multi-slot virtual queue

- **Algorithm** (from `deque_modeling_policy_data_aware.c:441-602`, `helper_mct.c:141-172`,
  `component_sched.c:706-734`). For job j and eligible worker w of class c, in the engine's
  existing scan:
  ```
  p_w        = work_j · f_group / speed(c)                       // expected run time on w
  end_i      = max(now + ε, started_i + p_i)                     // each running job; an overrunning
                                                                 // job gets now + κ·elapsed_i (κ=0.5) [U]
  if running(w) < slots(w) and admitted:    start_w = now
  else:  start_w = min_i end_i + (V_w + Σ_i(end_i − min_i end_i)) / slots(w)   // end_min_add
  eft_w      = start_w + p_w
  ```
  - `V_w` is the **virtual queue**: Σ p of jobs *deferred onto w earlier in this dispatch scan*.
  - Because the scan is in urgency order, `V_w` holds only more urgent work, so this is exactly
    dmdasd's `exp_len_per_priority` (`:474-481`) for free.
  - **Choice:** argmin over admitting workers of `(eft_w, !preferred, running, id)`, with
    `total_cmp` on floats and id last for determinism. dmda's α term is `eft − min_eft`, so argmin
    of that equals argmin of eft. β (transfer) and γ (energy) do not transfer (§2.8). A
    `prefer_penalty_s` (seconds) plays the cache-affinity role of β [U].
  - **Plain EFT, no deferral** (default): only workers that admit *now* are candidates. This is
    "speed-aware least-finish". It differs from fast-first only when a fast worker's free slot
    ties with a slow one. With uniform speeds and free slots, EFT always picks the fast class, so
    **without deferral EFT ≈ fast-first** [U, but follows from the formula].
- **Deferral** ("wait for a fast slot"; the "push refused, stays in window" mechanism of
  `component_prio.c:97-140`):
  - If the argmin over *all eligible workers*, including full ones, is a full worker w*, and
    `eft_{w*} < eft_best_admitting − min_gain·p_j`, and the job is not aged, and it has waited
    less than `max_defer`, the job is **not placed**.
  - Record `deferred[j] = (w*, eft_{w*})` and add `p_j` to `V_{w*}`.
  - The memory check for w* uses a new provided method
    `Admission::admits_if_slot_free(&self, d, w)`, whose default evaluates `admits` with
    `running` clamped to `slots − 1`.
- **Invariants:**
  - The **priority invariant becomes:** a job is placed on w only if every more urgent waiting job
    is refused on w **or deferred** with a strictly smaller EFT elsewhere.
  - The **work-conserving "empty worker leaves no eligible job waiting" check**
    (`sched/tests/invariants.rs:387-395`) must exempt deferred jobs, and only them.
  - The **escape hatch is unchanged.** A deferred job's EFT on an empty worker is `now + p`, so it
    only defers when `w*` is genuinely sooner.
- **Bounded waiting:** `max_defer` (seconds), and aged jobs (`age_limit`) never defer. This gives
  `wait <= max_defer` on top of the existing no-starvation bound. Deferral expiry is not an event,
  so the caller must be woken: new provided trait method
  `fn next_wakeup(&self) -> Option<Instant> { None }`. The engine returns the earliest
  `deferred_since + max_defer`.
- **API:**
  ```rust
  pub struct EftConfig { pub perf: PerfModelConfig, pub defer: Option<DeferConfig>,
                         pub overrun_factor: f64, pub prefer_penalty_s: f64 }
  pub struct DeferConfig { pub max_defer: f64, pub min_gain: f64 }
  pub struct EftPolicyConfig { pub backfill: BackfillConfig, pub eft: EftConfig,
                               pub slow_gate: Option<SlowGate>, pub spoliation: Option<SpoliationConfig> }
  policy!(EarliestFinish, EftPolicyConfig, |c| Mode { choice: Choice::EarliestFinish(..), .. });
  ```
  `explain` adds "deferred: expects worker W at t=…, EFT here …". `PolicyStats.deferred: Vec<(JobId, WorkerId, Instant)>`.
- **Where:** the engine (`sched/src/engine.rs` `choose` `:446-470`, `dispatch` `:601-655`).

### 2.4 `SlowGate`: HeteroPrio's slow-factor admission (`heteroprio.c:3602-3606`, auto slow factor `:3162-3168`)

- **Algorithm.** Classes are ranked by `speed`; `fast = argmax speed`, ties broken by smallest
  class name. A worker of class c ≠ fast admits the job being scanned only if
  ```
  backlog / fast_slots_total >= gate_factor · speed(fast)/speed(c)
  ```
  - `backlog` = waiting jobs eligible for the fast class, counted at the start of dispatch and
    decremented as jobs are placed.
  - `fast_slots_total` = Σ slots of live fast workers. StarPU counts workers, and each of ours is
    16 slots, so slots are the right analogue [U].
  - `gate_factor = 1` reproduces StarPU's auto `slow_factor = t_slow/t_fast`.
  - A **work-weighted variant** replaces the count by `Σ p_fast(waiting) / (fast_slots · p_fast(j))`.
    It is better when task sizes vary 10x+, as ours do [U].
- Aged jobs and reservation holders bypass the gate, which keeps the starvation bound. A job whose
  `class` constraint names the slow class bypasses it too.
- **Where:** the engine, as an extra `Refusal::SlowGate` in `refusal` (`sched/src/engine.rs:384-406`).
- **Monotonicity:** within one dispatch, placements only shrink the backlog, so a refused job
  stays refused. The Admission monotonicity contract (`sched/src/admission.rs:43-45`) is
  preserved for the scan.

### 2.5 Spoliation: preempt-and-restart (paper Algorithm 2; NOT in StarPU code)

- **Rule** (paper, adapted to two speed classes and multi-slot workers). After the normal scan,
  for each worker w with a free slot, in id order, *if no waiting job was placed on or is admitted
  on w*:
  ```
  S = { running job r on worker v :  speed(class v) < speed(class w)
                                      and eligible(r, w) and admits_on(w, r.demand)
                                      and r.preemptions < max_preemptions_per_job (default 1)
                                      and end_r > now + p_r(w) + restart_overhead
                                      and end_r − (now + p_r(w)) >= min_gain · p_r(w) }
  pick r* = argmax end_r      (paper: "highest finish time")
            ties → more urgent key, then job id
  ```
  - `end_r` uses the same overrun rule as §2.3.
  - **Variant `SpoliationOrder::MostUrgent`:** argmin urgency key among S, since the graph version
    uses priorities [paper §7.2.1, exact rule **[U]**].
  - **Engine effect:** release r on v (slot and demand), place r on w, set `r.started = now`,
    `r.preemptions += 1`, and emit `Preemption { job: r, from: v, to: w }`. Repeat while w has
    free slots and S is non-empty.
- **API (non-breaking):**
  ```rust
  pub struct Preemption { pub job: JobId, pub from: WorkerId, pub to: WorkerId }
  pub struct Dispatch { pub start: Vec<(JobId, WorkerId)>, pub preempt: Vec<Preemption> }
  pub trait Policy { ...
      /// Placements plus preemptions. Default: no preemptions.
      fn dispatch_full(&mut self, now: Instant) -> Dispatch {
          Dispatch { start: self.dispatch(now), preempt: Vec::new() }
      }
  }
  pub struct SpoliationConfig { pub min_gain: f64, pub restart_overhead: f64,
                                pub max_preemptions_per_job: u32, pub order: SpoliationOrder }
  ```
  For engine policies, `dispatch()` equals `dispatch_full().start` only when spoliation is off.
  With spoliation on, calling plain `dispatch` must **panic or debug_assert** to prevent silent
  loss of preemptions [design choice]. `Dag` gets `dispatch_full` as a passthrough.
- **Caller contract:**
  1. On `Preemption{job, from, to}`: kill the instance on `from`, start a fresh one on `to`. The
     library already counts the job as running on `to` only.
  2. **Race:** if the `from` instance reports success before the kill lands, the caller calls
     `completed(job)`. That releases the `to` placement, so the caller must also kill the `to`
     instance. Our tasks are deterministic and restartable (`sched/LITERATURE.md:144-147`), so
     either result is valid.
  3. The killed instance's failure or exit is **not** reported (no `failed`/`completed`). Its
     time is never fed to the perf model.
  4. `worker_gone(from)` after a preemption does not affect the job (it is no longer on `from`).
- **Invariants to test:**
  - (i) Each job has at most one live placement. `Σ running` per worker equals the shadow model's
    after applying both `start` and `preempt`.
  - (ii) No over-commit on `to`: admitted including the escape hatch, with slots.
  - (iii) `speed(to) > speed(from)`, and `to` had a free slot that no waiting job took (no
    preemption while a waiting job is admitted on `to`; this keeps priority above spoliation).
  - (iv) `preemptions(job) <= max`, which rules out ping-pong.
  - (v) Determinism, as for dispatch.
  - (vi) `completed(job)` after a preemption releases `to`, not `from`.
  - (vii) Work-conservation for waiting jobs still holds.
  - (viii) With `spoliation = None`, `dispatch_full().preempt` is always empty and placements are
    byte-identical to today's.
- **Where:** the engine (decision and bookkeeping), the DAG layer (passthrough), the simulator
  (kill and restart in the processor-sharing `Wk`, `sched/src/sim/whole.rs:918-924,1217-1258`).

### 2.6 HEFT window (`component_heft.c:41-166`): evaluate, probably reject

Among the ≤5 most urgent equal-priority jobs, schedule first the one with the smallest min-EFT.
This is shortest-first within a priority level. In a span-bound run, longest-first within a
bidegree is the usual better tie-break [U], and our scan already places every admissible job in
one dispatch. So the window only matters under contention.

If tried: a `Key` tie-break field `work_order: ShortestFirst | LongestFirst | Fifo` in the engine.
Low priority.

### 2.7 NOD / URT priorities (auto-heteroprio): DAG layer, optional

`NOD(j) = Σ_{s∈succ(j)} 1/indeg(s)` (`heteroprio.c:2736-2777`) is a cheap "how much this job
unlocks" score. It is computable in `DagScheduler` at submit time from `unmet` counts, using
`unmet` (live) rather than StarPU's total in-degree [U]. It could serve as a tie-break *within a
group* under `group` ordering. Given the finding that upward rank did not beat group order
(`sched/RESULTS.md` "What the gain is made of", item 3), this is low priority. The per-arch score
formulas need more than one task type to mean anything, so skip them.

### 2.8 What does NOT transfer

| StarPU feature | file:line | why not |
|---|---|---|
| β·transfer term, `predicted_transfer` overlap, prefetch, LA-heteroprio memory-node groups, `dmdar`/`dmdas` ready-first pop | `deque_modeling_policy_data_aware.c:309-320,568-571`; `heteroprio.c:1875-2160,3493-3570`; `fifo_queues.c:378+` | Our workers fetch inputs themselves; no data-handle model; transfer cost is inside the measured duration. |
| γ·energy, idle_power | `:680-694`; `helper_mct.c:100-107` | No energy objective. |
| CUDA streams, multiple implementations (`nimpl`), `best_impl` component, per-worker FIFO with pipelined exp_len | `:462-586`; `component_best_implementation.c` | One implementation per task; our "slot" concurrency is processor sharing inside one worker process, not device streams. |
| Acceleration buckets, per-arch bucket visit order, auto-heteroprio score zoo | `component_heteroprio.c:368-503`; `heteroprio.c:2369-2640` | Acceleration is a class constant for us (2.41 for all tasks): one bucket; ordering reduces to the existing priority. |
| Commit-at-push per-worker queues (dmda) | `:271-408` | Our engine commits only to free slots; the virtual queue (§2.3) gets the EFT effect without irrevocable queues, which would fight reservations, aging and memory admission. |
| 50% outlier flush, first-sample discard | `perfmodel_history.c:1924-1966` | Log-normal noise sd 0.60 makes most samples "outliers"; we have one sample per unique task. |
| Footprint-keyed history means | `perfmodel_history.c:1740-1825` | Our tasks are mostly unique; the only reusable keys are class and group (§2.2). |
| Same-arch prefetch stealing | `heteroprio.c:3649-3707` | We have no worker-local queues. |
| Persisted calibration files | `heteroprio.c:1045-1388`; `perfmodel_history.c:1216+` | The library is pure and IO-free; the caller can snapshot `PerfModel` (serde) if wanted. |

---

## 3. Evaluating each item in `sched-whole`

**Setting.** Fleet: 7 H200 + 14 L40S processes × 16 slots. L40S is the fast class: 224 of 336
slots and ~83% of throughput. The run is span-bound (W/P 770 h, D 883 h). The best plan so far is
group + uncapped + fast-first = 1,237 h (1.40× the bound) at 59.5% slot utilisation
(`sched/RESULTS.md`). The remaining 0.40× is mostly critical work running below the fastest speed
plus dependency latency [U].

**Simulator changes needed:**
- `simulate` gets a `PlacementMode` enum instead of `fast_first: bool`
  (`sched/src/sim/whole.rs:929-935`), with values `Oblivious`, `FastFirst`, `Eft`, `EftDefer`,
  `SlowGate`, plus a `spoliation` flag.
- Jobs carry `work` = estimated cost; truth stays in the `Wk` running list.
- Preemption is handled by `advance(from)`, removing the job, `schedule(from)`, pushing the job to
  `to` with full true work, and counting wasted slot-seconds.
- The engine uses `dispatch_full`. A deferral needs a timer: push `Ev::Wake` at `next_wakeup()`.

**Plans to add** (each on `group` uncapped, and on `rank+aging` as a check; ≥5 seeds of the cost
noise, compare paired):

| plan | what it tests | expected effect [all U, judgement] |
|---|---|---|
| `eft` (no defer, oracle speeds) | EFT ≈ fast-first sanity check | within ±1% of `group+fast`; if not, a bug |
| `eft+defer` sweep `max_defer ∈ {60, 600, 3600} s`, `min_gain ∈ {0, 0.1}` | "wait for a fast slot" | −0 to −8%: helps when a critical job would land on H200 while an L40S slot frees within 0.585·p; virtual queue keeps H200s busy when backlog is deep. Watch slot util and bidegree p90. |
| `slowgate` sweep `gate_factor ∈ {0.5, 1, 2}`, count vs work-weighted | HeteroPrio's estimate-free version of the same | similar sign to eft+defer, smaller; risk of idling H200s at narrow points |
| `+spoliate` on `group+fast` and on `eft+defer`; `order ∈ {LatestFinish, MostUrgent}`; `min_gain ∈ {0, 0.2}` | correcting misplacements in tails / narrow points | −2 to −10% on `group+fast` (utilisation ~60% means fast slots are often idle while H200s run); smaller on top of `eft+defer`. Report: preemptions, wasted H200-h, makespan, tail latency. |
| `learn` = EFT/spoliation with priors {1,1} and learning on vs oracle speeds | online perf model | within ~1% of oracle after warm-up (thousands of tasks per hour); report speeds over time |
| `learn+group` per-group correction | footprint idea | small on placement; mainly helps rank plans [U] |
| noise robustness: multiply estimate noise sd ×{0.5,1,2} | EFT/spoliation depend on estimates | spoliation degrades least (it uses elapsed time); deferral most |

**Metrics to add to `WholeMetrics`:**
- `preemptions`, `wasted_work_h`, `deferred_jobs`, `max_defer_wait`;
- class split of critical-path time [U: needs critical-path tracking; a cheaper proxy is "share
  of work on H200 among tasks in the final 10% of each bidegree's walk"].

**Go/no-go:** adopt an item if it gains ≥3% paired across seeds and does not raise bidegree
latency max by more than 20%.

---

## 4. Implementation plan

Ordered; each step is green on `just lint` / tests before the next. Effort: S ≤ ½ day,
M ≈ 1–2 days, L ≈ 3+ days.

1. **`JobSpec.work` and `Running.started`/`class`** (S).
   - Files: `src/lib.rs` (field, serde default, `JobSpec::with_work`), `src/engine.rs`
     (`Running`), `src/dag.rs` (`submit_node` fills work).
   - Tests: DAG fills work from the estimate; existing tests unchanged.
   - Risk: `JobSpec` struct literals in callers break. It is internal, so fix them.
2. **`src/perf.rs` PerfModel** (M). API as in §2.2, `Serialize` behind `serde`.
   - Unit tests:
     - prior used until `calibration_minimum`;
     - log-mean converges to the true speed under log-normal noise (seeded, proptest);
     - the window cap keeps tracking a speed change;
     - observing in a different order gives the same result *for the same multiset within a full
       window*, and is otherwise exactly deterministic.
   - Engine: provided `Policy::failed`; learning on `completed` only.
3. **`Choice::EarliestFinish` without deferral, plus the `EarliestFinish` policy** (M). Files:
   `src/engine.rs` (`choose`, `Mode`, new config structs), `src/lib.rs` (re-exports, `PolicyStats.speeds`).
   - Tests:
     - fast free slot beats slow free slot;
     - equal speeds reduce to LeastLoaded order;
     - add `Kind::Eft` to `tests/invariants.rs`: no over-commit, escape hatch, priority invariant
       unchanged (no deferral yet), determinism (run twice), bookkeeping.
4. **Deferral, virtual queue, `next_wakeup`** (M–L). Files: `src/engine.rs` (scan keeps `V_w`;
   `deferred` map; `explain`), `src/admission.rs` (`admits_if_slot_free`), `src/lib.rs`
   (`Policy::next_wakeup`, `PolicyStats.deferred`), `src/dag.rs` (passthrough).
   - Tests (proptest, extending `tests/invariants.rs`):
     - (a) relaxed priority invariant: a more urgent waiting job is refused, or deferred with
       `eft_target < eft_here`;
     - (b) work-conservation holds for non-deferred jobs;
     - (c) a deferred job is placed or un-deferred by `deferred_since + max_defer`
       (`tests/starvation.rs` style, driving `next_wakeup`);
     - (d) aged jobs never defer;
     - (e) the virtual queue makes a slow worker take the k-th job once
       `V_fast/slots > p·(1/s_slow − 1/s_fast)` (a handcrafted unit test);
     - (f) determinism.
   - Risks:
     - deferral breaks Graham's greedy guarantee; bounded only by `max_defer`;
     - interplay with reservations: a reservation holder must not defer (it bypasses);
     - float ties need `total_cmp` and id tie-breaks.
5. **SlowGate** (S–M). In `src/engine.rs` `refusal` and `explain`.
   - Tests:
     - gate closes at a short backlog and opens at a long one;
     - holders and aged jobs bypass;
     - monotone within a dispatch (proptest: a refused job is never later placed on the same
       worker in the same dispatch);
     - existing invariants still hold, with the priority invariant counting `SlowGate` as a
       refusal.
6. **Spoliation** (L). Files: `src/lib.rs` (`Preemption`, `Dispatch`, `Policy::dispatch_full`),
   `src/engine.rs` (post-scan pass, `preemptions` count per running job), `src/dag.rs`
   (`Dag::dispatch_full`), `PolicyStats.preemptions_total`.
   - Tests:
     - proptest with a shadow model that applies `start` and `preempt`, checking invariants
       (i)–(viii) of §2.5;
     - unit tests: the race (completion after preemption releases `to`), `worker_gone(from)`
       after preemption, no preemption to an equal or slower class, the `max_preemptions` cap.
   - Risks:
     - wasted work if estimates are bad: `min_gain` and the overrun rule;
     - production caller complexity (kill path, duplicate-instance race);
     - memory: the restart is admitted on `to` under normal admission, but `from`'s RSS drops only
       after the kill, so the reported `used` lags [U].
7. **Simulator** (M). Files: `src/sim/whole.rs` (`PlacementMode`, preemption handling, `Ev::Wake`,
   new metrics), `src/bin/sched_whole.rs` (`--placement eft|eft-defer|slowgate|fast|none`,
   `--spoliate`, `--learn`, sweeps). Optionally `src/sim/run.rs` for trace replay.
   - Tests: extend `whole_run_plans_finish_within_bounds` (`whole.rs:1331`) to the new modes;
     every plan finishes and makespan ≥ `max(W/P, D)`.
8. **Run the §3 plans, then write RESULTS.md and README** (M, mostly compute: ~8 min per plan per
   seed).
   - Update README's policy table and the trait section (`next_wakeup`, `dispatch_full`, `failed`).

**Order rationale:** steps 1–3 are cheap and give an EFT baseline equal to fast-first, which
validates the plumbing. Step 4 is the dmda/HEFT "wait for fast" lever, and LITERATURE.md §9
item 3 says to run it first as the baseline for spoliation. Steps 5–6 are HeteroPrio's two ideas.
Learning (step 2) is exercised from step 3 on, but evaluated last against oracle speeds.

**Cross-cutting risks:**
- f64 in decisions: keep determinism with `total_cmp` and id tie-breaks; never iterate a
  `HashMap` in decision paths. The engine already uses BTreeMap for workers.
- `dispatch` cost: EFT is O(workers × running per worker) per job. Cache per-worker
  `(min_end, Σ(end − min))` per dispatch, which gives O(workers) per job, like today.
- **The biggest uncertainty:** whether the gains survive memory admission, which `sched-whole`
  does not model (`sched/RESULTS.md` caveats). Re-check spoliation in `sched-sim` trace replay,
  where memory binds.
