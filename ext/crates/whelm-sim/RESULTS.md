# Results on the reference trace

Trace `sched_trace_40518773.jsonl.gz`: 164,114 jobs (174 zero, 163,940 signature), 174 bidegrees,
21 workers (6 H200, 15 L40S worker processes, 16 slots each, 123.7 / 154.6 GB budgets), 33 h,
812,950 dependency edges. Reproduce with `whelm-sim --trace <trace> [--closed [--rank]]
[--age-limit 1800] --json out.json` (release build, under 1 s per open-loop policy). "Big" means
`est_gb > 7.5`.

## Headline

1. **Production starved big jobs because of how it computes the baseline, not because of how it
   places jobs.** A worker's `reported_baseline` is its memory gate's rolling 5–10 min RSS floor
   (`MemGate::baseline`). On a worker that always has 8–16 jobs running, that floor already
   contains those jobs' memory, and admission then adds their estimates on top
   (`baseline + Σ placed`). The double count keeps every worker a few GB from its budget. Replaying
   the same placements with a task-free baseline (each worker's idle RSS) makes waits ≈ 0
   (p99 42 s, max 22 min, Greedy) at unchanged throughput. No placement policy comes close.
   Caveat: the replay keeps the trace's RSS samples, so it cannot show whether looser admission
   would push real RSS over the budget. The estimates are known to be pessimistic, which makes
   this plausible but unverified.
2. **Under the production baseline, PriorityBackfill and BestFit cut median wait 8x and max wait
   1.4x, at no measurable throughput cost.** With `age_limit = 1800 s` they cut max wait 8x
   (4.9 h → 36–38 min). The cluster is memory-saturated for hours at a time, so the long tail is
   queueing behind more urgent work, not starvation. Placement can reorder that queue but cannot
   shorten it. Aging bounds every job's wait; without it, strict priority lets a job in a young
   group wait behind older groups for hours.
3. **DAG-rank priority (closed loop) shortens the makespan by ~3.5%** (34.0–35.3 h vs 35.4–37.0 h
   for group order, over heartbeat perturbations). On its own it starves low-rank jobs (max wait
   29 h); with aging it does not (max 39 min at `age_limit = 1800 s`), keeping most of the gain.
4. Reservations cost about 1% of slot time idled while draining. More reservations (2, 4, per
   class) only add idle time (2–4%) without improving the tail. Lanes trade 3–5 points of slot
   utilisation for slightly better big-job p90/p99. `dispatch` takes a few µs on average and at
   most ~2 ms.

## Service model

Per worker process, total throughput `f(k) = speed · min(k, k_sat)^alpha` with `k` running jobs,
split equally between them (processor sharing). Fitted by OLS on `ln(service time)` with
bidegree × kind fixed effects plus `ln target`, `ln(next+1)`, `ln est_gb`, grid-searching
`(k_sat, alpha)` per class (method on `whelm::sim::model::fit`). 163,977 jobs, 344 fixed effects:

| class | speed (job alone, rel. H200) | k_sat | alpha | f(1) | f(8) | f(16) |
|---|---|---|---|---|---|---|
| h200 | 1.000 | 15 | 1.0 | 1.00 | 8.00 | 15.00 |
| l40s | 2.413 | 16 | 1.0 | 2.41 | 19.31 | 38.61 |

- **Throughput is linear in concurrency up to the 16 slots.** A job's own speed does not depend
  on how many others share its worker. GPU sharing is not the bottleneck at this concurrency.
  `alpha = 0` (no concurrency gain) fits worse: SSE 51,636 vs 48,565.
- **The fit is noisy:** within-group R² 0.37, residual sd 0.54 in log space. This does not weaken
  the replay, because each job's work is computed from its own observed run
  (`W = ∫ f(k)/k dt`), so replaying production's placements reproduces production's service
  times exactly. The regression only identifies the shape of `f`.
- **L40S ≈ 2.4x H200 per job.** This is per worker process; it is not comparable to the measured
  "L40S ≈ 1.39x H200 per GPU" without the process-to-GPU mapping, which the trace lacks. The
  bidegree fixed effects control for H200 workers getting different bidegrees, but not for
  within-bidegree selection beyond the size covariates.

## Memory model and validation

The replay feeds each worker, every 60 s, `reported_used` = the trace's RSS sample and
`reported_baseline` = the rolling RSS floor over two 300 s windows, rebuilt from the samples
minus a calibration offset. Both are exogenous: replayed from the trace, not responsive to
simulated placements. RSS in the trace is mostly a persistent footprint growing ~0.6 GB/h; it
barely tracks the running count, which supports replaying it.

The offset is needed because the trace samples RSS once a minute while the gate samples it more
often and sees lower dips. Checked against production's own placements, 60% of them violate the
floor rebuilt from per-minute samples, by a median of 0.5 GB and a p90 of 3.3 GB. Sweeping the
offset for the Greedy replay against what production did:

| floor offset | wait p50 | p90 | p99 | max | big p50 | big p90 | big max | makespan | slot util |
|---|---|---|---|---|---|---|---|---|---|
| production | 9.4 s | 1.9 m | 18.3 m | 18.9 h | 54.2 s | 7.9 m | 18.9 h | 33.01 h | 74.8% |
| 0 GB | 17.3 m | 37.4 m | 44.0 m | 9.2 h | 29.2 m | 43.1 m | 9.2 h | 33.63 h | 72.7% |
| 2 GB | 1.1 m | 3.9 m | 10.7 m | 4.9 h | 1.1 m | 3.2 m | 4.9 h | 33.03 h | 73.7% |
| **3 GB (default)** | 10.9 s | 2.1 m | 9.8 m | 4.9 h | 16.3 s | 1.3 m | 4.9 h | 33.23 h | 72.8% |
| 4 GB | 0.4 s | 54.2 s | 8.7 m | 1.4 h | 3.5 s | 41.5 s | 1.4 h | 33.10 h | 72.7% |

At 3 GB the Greedy replay matches production's bulk (p50, p90, makespan, utilisation, and the
H200/L40S split: 25k/139k jobs vs 30k/134k). Its tail is milder (max 4.9 h vs 18.9 h), so the
absolute tail figures below are optimistic. Compare policies against simulated Greedy, not
against production.

## Open loop (arrivals at the trace's `ready_s`)

| policy | makespan | work/h | slot util | wait p50 | p90 | p99 | max | big p50 | big p90 | big p99 | big max | reservations | idle draining |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production | 33.01 h | 1.561M | 74.8% | 9.4 s | 1.9 m | 18.3 m | 18.9 h | 54.2 s | 7.9 m | 52.5 m | 18.9 h | – | – |
| greedy | 33.23 h | 1.550M | 72.8% | 10.9 s | 2.1 m | 9.8 m | 4.9 h | 16.3 s | 1.3 m | 7.0 m | 4.9 h | 0 | 0 |
| backfill (no reservation) | 33.13 h | 1.555M | 73.2% | 0.9 s | 23.3 s | 9.9 m | 6.3 h | 5.4 s | 2.3 m | 34.4 m | 6.3 h | 0 | 0 |
| backfill | 33.23 h | 1.551M | 73.0% | 1.3 s | 36.3 s | 13.9 m | 3.5 h | 6.7 s | 2.4 m | 34.5 m | 3.5 h | 9,309 | 0.92% |
| bestfit | 33.24 h | 1.550M | 74.4% | 1.5 s | 40.3 s | 13.4 m | 3.5 h | 8.5 s | 2.6 m | 34.1 m | 3.5 h | 9,764 | 0.94% |
| lanes (H200 = lanes) | 33.15 h | 1.554M | 69.7% | 1.6 s | 39.8 s | 14.7 m | 3.6 h | 5.3 s | 1.7 m | 23.0 m | 3.6 h | 7,876 | 1.51% |
| backfill, aging 30 min | 33.11 h | 1.556M | 73.4% | 1.5 s | 57.2 s | 30.0 m | 37.6 m | 7.6 s | 3.7 m | 30.1 m | 37.6 m | 10,468 | 0.97% |
| bestfit, aging 30 min | 33.15 h | 1.554M | 74.6% | 1.6 s | 57.1 s | 30.0 m | 35.7 m | 8.7 s | 3.7 m | 30.1 m | 35.7 m | 10,713 | 0.98% |
| greedy, idle baseline | 33.12 h | 1.556M | 71.3% | 0.0 s | 0.0 s | 41.8 s | 22.2 m | 0.0 s | 0.0 s | 9.4 s | 22.2 m | 0 | 0 |
| bestfit, idle baseline | 33.00 h | 1.561M | 61.0% | 0.0 s | 0.0 s | 9.7 s | 38.9 m | 0.0 s | 0.0 s | 2.1 s | 38.9 m | 253 | 0 |

- Open-loop makespan is fixed by the arrivals (the last jobs become ready near the end), so
  throughput differences here are below 0.5% and not meaningful.
- Group completion latency is the same for every policy (p50 3.5 h, p90 5.6 h): a group is paced
  by its own dependency chain, which the open loop replays fixed.
- Strict priority raises big-job p99 (7 m → 34 m) because priority, not size, decides who waits.
  Big jobs of young groups queue behind older groups. Aging trades that tail for a bounded
  maximum: everything waits ≤ ~38 min, and p99 rises to the age limit.
- Reservation sweeps (`max_reservations` 2 and 4, one per class) leave max wait at 3.5–3.6 h and
  raise idle draining to 1.8–4.0%. One reservation is enough.

## Closed loop (arrivals = simulated dependency completion + the trace's measured gap)

All jobs are declared up front to the DAG layer (zero tasks depend on the sinks of their
`after_groups`); a job is released its measured coordinator gap after its last dependency
completes in the simulation. Closed-loop makespan is sensitive to small perturbations: ±0.6 h
when the heartbeat moves by 1 s. So only differences beyond ~1 h are meaningful; wait metrics are
stable.

| policy | makespan | wait p50 | p90 | p99 | max | big p99 | big max | group p50 | group p90 | group max |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 35.85 h | 2.7 m | 9.9 m | 16.2 m | 3.7 h | 16.1 m | 3.7 h | 3.8 h | 5.1 h | 9.0 h |
| backfill | 36.46 h | 1.4 s | 37.4 s | 9.8 m | 3.2 h | 59.0 m | 3.2 h | 3.7 h | 4.6 h | 6.8 h |
| bestfit | 35.41 h | 1.3 s | 36.5 s | 9.0 m | 3.2 h | 1.1 h | 3.2 h | 3.7 h | 4.7 h | 6.3 h |
| backfill, aging 30 min | 35.17 h | 1.5 s | 1.0 m | 30.1 m | 33.2 m | 30.5 m | 33.2 m | 3.7 h | 4.6 h | 6.1 h |
| backfill, DAG rank | 34.02 h | 9.2 s | 8.9 m | 34.4 m | **29.9 h** | 1.2 h | 29.9 h | 3.6 h | 6.4 h | 32.4 h |
| bestfit, DAG rank | 34.26 h | 8.4 s | 8.9 m | 34.1 m | **28.8 h** | 1.1 h | 28.8 h | 3.6 h | 6.5 h | 32.4 h |
| bestfit, DAG rank, aging 30 min | 34.46 h | 10.0 s | 14.9 m | 30.2 m | 38.5 m | 30.6 m | 38.5 m | 3.4 h | 6.1 h | 10.0 h |
| bestfit, DAG rank, aging 60 min | 34.19 h | 8.8 s | 9.0 m | 1.0 h | 1.3 h | 1.0 h | 1.3 h | 3.5 h | 6.0 h | 14.4 h |

- Priority with backfill cuts median wait from minutes to ~1 s and the worst group from 9.0 h to
  6.3–6.8 h; aging brings the worst group to 6.1 h.
- DAG-rank priority finishes the whole trace ~1.3 h (~3.5%) sooner, across perturbations. It
  works by running critical-path work first and letting off-path groups lag: group p90 rises to
  6.0–6.5 h. Without aging, a few low-rank jobs wait more than a day.
- **Recommended configuration:** BestFit or PriorityBackfill, one reservation,
  `reserve_after = 60 s`, `age_limit` 30–60 min. Add DAG-rank priority once the coordinator
  declares dependencies. Separately, and with a larger effect: exclude running tasks from the
  worker's reported baseline.

## Admission without the double count (`baseline_excl`)

The worker will report `baseline_excl`: its rolling RSS floor minus the estimates of the jobs it
runs. With it as `reported_baseline`, `ProductionAdmission` is exactly
`running < slots && (running == 0 || max(rss, baseline_excl + Σ placed) + demand <= budget)`.
Replayed by rebuilding `baseline_excl` from the samples (`max(0, rss - estimates running)`, same
rolling windows as the floor; `whelm-sim --baseline excl`), open loop, `age_limit = 1800 s`:

| baseline | estimates | policy | wait p90 / p99 / max | slot util | modelled heartbeats over budget (max excess) |
|---|---|---|---|---|---|
| floor (today) | as recorded | backfill | 57 s / 30.0 m / 37.6 m | 73.4% | 0% |
| floor (today) | as recorded | best fit | 57 s / 30.0 m / 35.7 m | 74.6% | 0.003% (4.9 GB) |
| **excl** | as recorded | **backfill** | **0 s / 7.8 s / 13.9 m** | 71.4% | **0.008% (5.7 GB)** |
| excl | as recorded | best fit | 0 s / 5.2 s / 13.0 m | 60.7% | 0.19% (34 GB) |
| idle (oracle) | as recorded | backfill | 0 s / 11.2 s / 37.2 m | 71.5% | 0.003% (0.1 GB) |
| floor | x0.32 | backfill | 0 s / 7.6 s / 13.8 m | 71.4% | 0.05% (17 GB) |
| excl | x0.32 | backfill | 0 s / 7.6 s / 13.8 m | 71.4% | 0.05% (17 GB) |
| excl | x0.32 | best fit | 0 s / 4.9 s / 13.0 m | 60.6% | 0.34% (34 GB) |

All arms finish the same work in the same time (W/h within 0.7%); slot utilisation drops because
jobs no longer queue, not because work is lost.

- **`baseline_excl` removes the double count**: waits collapse to seconds, as with the oracle idle
  baseline, at unchanged throughput.
- **Overrun risk.** The trace's RSS never exceeded a budget (0 of 37,521 samples), and it does not
  respond to simulated placements, so the risk is modelled: each job occupies its recorded
  estimate times a per-job fraction, log-normal (sd 0.5, an assumption) around the trace's median
  ratio of `(RSS - idle)` to estimates running, **0.595** (p10 0.38, p90 0.87, p99 1.11, over
  37,498 samples). With backfill and `baseline_excl` the modelled RSS exceeds the budget at 0.008%
  of heartbeats (about 3 minutes in 33 hours over 21 workers). Best fit packs the L40S tighter
  and raises that to 0.19%: prefer backfill with `baseline_excl`. Since admission still compares
  against reported RSS, real overruns would also be capped by that safety term, which the replay
  cannot show.
- **The recorded estimates are not 6x.** On this trace jobs occupy about 0.6 of their recorded
  estimate in aggregate (1.7x, not 6x), so the x0.32 arm puts estimates *below* actual use
  (0.54x): it admits more and overruns 6x more often (0.05%), and with x0.32 the baseline no
  longer matters (floor and excl agree). Scale estimates only as far as the measured peak allows.

## Limitations

- RSS and the baseline are replayed from the trace (overruns only through the usage model above),
  so the simulation cannot show a policy's effect on real memory, nor a drained worker's RSS
  dropping.
- Workers join at their trace join times and never leave, and job durations follow the
  processor-sharing model, so failures, retries and per-job variance beyond the model are absent.
- Closed loop uses each job's measured gap after its dependencies and oracle work estimates for
  ranks (the fitted `W`). Production's ranks would use estimates.
- Lookahead reservation (spec §4b, optional) is not implemented.

# Whole run, built a priori (`whelm-whole`)

The stem-400 resolution as one DAG, built before any computation. Reproduce with
`whelm-whole --trace <trace> --census <csv>... --max-n 400 --max-s 202 --max-profile-len 5
--plans today,today+fast,group,group+fast,group+eft,rank+eft --noise-seed N`. Each plan
simulates in about 5 minutes (rank plans and waiting plans longer). Raw results:
`/rs/rs_grp_csht/resolutions/sphere_gpu_n400/full_seed{1,2,3}.json`.

## The DAG

- **Bidegrees:** 81,403 (n ≤ 400, s ≤ 202).
  - **Edges:** `ext::nassau`'s `depgraph`:
    - `Register(s, t − floor) → Compute(s, t)`;
    - `Register(s−1, t−1) → Compute(s, t)`;
    - `Register(0, t) → Compute(1, t)`;
    - `Register(s, t−1) → Register(s, t)`.
  - **Subalgebra:** `optimal_for`'s rule, which reproduces 99.6% of 81,511 census rows at the
    production cap.
- **Walks:** each live bidegree (one with generators; 30,677 of them) expands its profile's
  signature DAG.
  - **Edges:** `sig_dag::direct`, which closes to the verified 938 (A(2)) and 137,081 (A(3))
    edges.
  - **Size:** transitively reduced, A(3)'s 51,859 direct edges shrink to its 4,028 covers and
    A(4)'s 11.5M to 195,579.
  - **Rows 0 and 1** run `step0`/`step1`, so they have no walk.
- **Tasks:** 3.93M signature tasks and 81k zero steps. Work is 495k H200-hours of signatures and
  2.4k of zero steps.
- **Costs** (H200-seconds):
  - **Scale across bidegrees** comes from the census, where 61k of the 81k bidegrees were
    measured: `ln wall ~ 0.55 ln tmd + 0.50 ln nd + 0.87 ln(signatures+1)`, R² 0.80, with a
    per-file effect.
  - **Level** is calibrated on the trace's 150 bidegrees, with per-bidegree spread sd 0.31.
  - **Zero step:** 0.5% of a bidegree's work.
  - **Signature split:** the rest goes to signatures with weight `deg^-0.59 · tmd(t − deg)^-0.16`
    (within-bidegree fit on 74,743 tasks, per-signature sd 0.60).
  - **"True" costs** are the estimates times seeded log-normal noise with those spreads.
- **Fleet:** the trace's 7 H200 and 14 L40S worker processes, 16 slots each. Throughput is linear
  in concurrency, as fitted, and an L40S runs a job 2.41× faster.

## Bounds and plans

Corrected results (the first version of this table was produced by a simulator that dispatched
after every individual event; see "Simulator artefacts" below). Lower bounds per noise seed:
critical path 817-877 h, capacity (W/P) 772-776 h, so the run is **span-bound**. Any greedy
schedule finishes within W/P + D (Graham/Brent) on identical machines; on mixed speeds that needs
the critical tasks on fast machines.

**The two levers** (noise seed 1, bound 869 h):

| plan | makespan | x bound | slot util | bidegree latency p90 / max |
|---|---|---|---|---|
| **today**: <= 24 open bidegrees, <= 32 walk tasks each, oldest first, speed-oblivious | 2,060 h | 2.37 | 37.9% | 13.5 h / 111 h |
| today + fast first | 1,482 h | 1.71 | 44.3% | 10.4 h / 75 h |
| oldest first, uncapped, speed-oblivious | 1,628 h | 1.87 | 48.0% | 5.9 h / 88 h |
| **oldest first, uncapped, fast first** | **1,249 h** | **1.44** | 59.4% | 5.0 h / 96 h |

- **Speed-aware placement: -23% to -28%.** In a span-bound run a critical task on a 2.4x slower
  worker extends the run directly.
- **Removing the caps: -16% to -21%.** Today's coordinator is not greedy: it idles slots while
  ready work waits behind the open-bidegree and walk-thread caps. Uncapped, about 260 bidegrees are
  open at the peak, with about 470k live DAG nodes.
- **Combined: 1.65x faster than today** (2,060 h -> 1,249 h), at 1.44x the lower bound.

**Refinements** (mean of 3 noise seeds, all uncapped; relative to oldest first + fast first on
the same seed):

| plan | makespan | vs oldest first + fast first | bidegree latency p90 / max | dispatch mean / max |
|---|---|---|---|---|
| oldest first, fast first | 1,226 h (1.44x) | -- | 4.3 h / 91 h | 1 us / 0.05 ms |
| oldest first, earliest finish with waiting | 1,212 h | -1.2% (-2.3 .. +0.3) | **1.6 h / 69 h** | 82 us / 5.9 ms |
| oldest first, fast first, slow gate | 1,218 h | -0.6% (-2.0 .. +0.4) | 7.1 h / 77 h | 64 us / 5.4 ms |
| oldest first, rank within, earliest finish | 1,216 h | -0.8% (-2.4 .. +0.5) | 1.6 h / 79 h | 76 us / 4.6 ms |
| DAG rank (true costs), fast first | 1,268 h | **+3.4%** (+3.2 .. +3.8) | 5.0 h / 76 h | 1 us / 13 ms |
| DAG rank (true costs), earliest finish | 1,276 h | **+4.1%** (+2.1 .. +5.8) | 5.0 h / 76 h | 54 us / 20 ms |
| DAG rank (estimated costs), earliest finish | 1,276 h | **+4.1%** (+2.5 .. +5.4) | 5.0 h / 69 h | 37 us / 5.7 ms |

- **At full scale, oldest-bidegree-first is the right order.** Critical-path rank priority loses
  3-4% on every seed, even with true costs and with the dispatch artefact fixed. It wins on small,
  heavily contended replicas (below), which do not represent the production regime.
- **Waiting for a fast slot** buys about 1% of makespan here but cuts bidegree latency p90 from
  4.3 h to 1.6 h and the worst bidegree from 91 h to 69 h, for a few tens of microseconds per
  dispatch.
- All online plans stay at 1.42-1.49x the bound. On a 3.5k-task region, static HEFT with true
  costs reaches the critical-path bound (see the cross-validation section): the remaining gap is
  what full lookahead would buy, not what a better online ordering could.
- Comparisons between builds of the simulator differ by up to about 1% for the same plan and seed
  (the release order of simultaneous jobs), the same tie-break noise as between plans on one
  instance; differences of that size are not conclusions.

## Caveats

- **Memory is not modelled here:** the whole-run simulation enforces slots only. The trace
  replay shows memory admission does bind in production.
- **The cost model is a model.**
  - 25% of bidegrees have extrapolated dims.
  - The census mixes code versions; a per-file effect absorbs that.
  - The within-bidegree fit explains only 24% of variance.
  - Absolute hours are therefore indicative. Ratios between plans are the robust output, since
    every plan sees identical costs.
- **The "today" model is simplified:** a cap on open bidegrees and a cap on walk tasks per
  bidegree, both read off the trace. Coordinator-local work, registration and log replay are
  zero-time.
- **The fleet is fixed:** no joins, leaves or failures.


## Restart-stable order, learned speeds, and the measured speed ratio

Three noise seeds each, uncapped, fast first unless stated; makespan relative to the first row of
each table. Raw results: `gk_seed*.json`, `age_seed*.json`, `r139_seed*.json`,
`age139_seed*.json` next to the other whole-run results.

**Group order** (fitted speeds, L40S 2.41x H200; bound about 854 h):

| order between bidegrees | aging | makespan | vs arrival | bidegree latency p90 / max |
|---|---|---|---|---|
| arrival (first submission) | none | 1,226 h | -- | 5.6 h / 88 h |
| `(s, t)` | none | 1,221 h | -0.4% (-1.2..+0.1) | 0.6 h / 132 h |
| `(t, s)` | none | 1,293 h | +5.5% (+5.2..+5.8) | 0.9 h / 217 h |
| `(t - s, s)` | none | 1,318 h | +7.6% (+6.1..+8.4) | 0.6 h / 253 h |
| arrival | 30 min | 1,233 h | -- | 4.7 h / 82 h |
| `(s, t)` | 30 min | 1,227 h | -0.5% (-1.4..-0.1) | 4.6 h / 83 h |
| `(s, t)`, earliest finish with waiting | 30 min | 1,207 h | -2.2% (-2.7..-1.6) | 4.2 h / 74 h |

- **`(s, t)` is the restart-stable order** (`nassau::group`): it matches arrival order's makespan.
  Stem or `t`-major orders cost 5-8%.
- Without aging, `(s, t)` cuts p90 bidegree latency ninefold but lets a few high-`s` bidegrees
  wait longer (worst 132 h vs 88 h). **With the default 30-minute aging the two orders are
  indistinguishable**: at full scale most jobs wait past the age limit, so aged FIFO decides, and
  the worst bidegree is bounded (83 h).

**Learned speeds.** Every worker reports speed 1 and `Learn::default()` (per worker, class prior,
hysteresis) learns from completions with estimated work: +0.4% (-0.3..+1.1) against oracle speeds,
-0.1% with `(s, t)` and aging. The estimator recovers the fast-first gain.

**The measured speed ratio.** The fit gives L40S 2.41x H200 per worker process; production
measures 1.39x per GPU. With `--class-speed l40s=1.39` (bound about 1,480 h):

| plan | makespan | vs today |
|---|---|---|
| today (24 open bidegrees, 32 walk tasks each, speed-oblivious) | 2,750 h | -- |
| uncapped, oldest first | 2,178 h | -20.8% |
| uncapped, fast first | 1,991 h | -27.6% |
| uncapped, earliest finish with waiting | 1,976 h | -28.2% |
| uncapped, fast first, `(s, t)`, 30 min aging | 2,012 h | -26.8% |
| uncapped, fast first, `(s, t)`, learned speeds | 1,987 h | -27.7% |

Removing the caps is worth about 21% at either ratio; fast-first placement is worth about 9% on
top at 1.39x, against about 25% at 2.41x. Today to uncapped fast-first is 1.38x at the measured
ratio (1.65x at the fitted one).

## Device memory (`whelm-device`, synthetic)

The trace has no device data, so `whelm-device` builds a scenario after job 40506688: 14 small
workers of 16 slots, a 19.5 GB device launch pool each, jobs needing about 2.4 GB (log-normal, sd
0.3: about 8 fit), 20,000 jobs ready at once, work log-normal (median 600 s, sd 0.6). Over the
pool, a job runs at `(C / S)^(1 + gamma)` of its speed for total demand `S`: `gamma = 0` means the
pool only serialises launches, and `gamma = 1.29` reproduces the live drop (16 jobs delivering 0.41x
of 8 jobs' throughput). Policy: backfill with the production rule, host AND device.

| gamma | device admission | makespan vs host only | jobs per worker | pool over-subscribed |
|---|---|---|---|---|
| 0 | none (host only) | -- (37.7 h) | 15.6 | 98% |
| 0 | per-task demand = p90 of jobs' | +53% | 5.0 | 0.1% |
| 0 | per-task demand = p75 | +28% | 5.9 | 2% |
| 0 | per-task demand = p50 | +2.5% | 7.9 | 60% |
| 0 | per-job demands, exact | +0.1% | 7.6 | 0% |
| 0 | per-job demands, error sd 0.3 | +10% | 7.4 | 44% |
| 1.29 | none (host only) | -- (94.9 h) | 15.9 | 99% |
| 1.29 | per-task demand = p90 | -39% | 5.0 | 0.1% |
| 1.29 | per-task demand = p75 | -49% | 5.9 | 2% |
| 1.29 | per-task demand = p50 | -56% | 7.9 | 63% |
| 1.29 | per-job demands, exact | **-60%** | 7.6 | 0% |
| 1.29 | per-job demands, error sd 0.3 | -52% | 7.5 | 49% |

- **Device-aware admission avoids the slowdown:** 39-60% shorter makespan at the live penalty.
- **Which per-task demand matters.** In the count form a high quantile is pessimistic for a sum:
  8 jobs' total concentrates near 8 x the mean, not 8 x the 90th percentile, so the p90 form fits
  only 5 jobs and, when over-subscription is harmless, costs 53%. Report a per-task demand near the
  mean, or per-job estimates (the sum form), which cost nothing when exact.

## Cross-validation against dslab-dag

`whelm-whole --export-dslab DIR` writes a region as dslab-dag input (one resource per worker,
speeds `10 x` single-job throughput, flops `10 x` work, zero-size data, both passthroughs of a
bidegree merged into one 1e-6-flop join, topological task order). On the n <= 60, s <= 12 region
(3,466 tasks), with `--linear-ps` (exclusive-core-equivalent throughput):

| check | ours | dslab-dag |
|---|---|---|
| G1: unlimited capacity (one L40S, 100,000 cores) | 25.389159 s, every plan, = critical path | 25.39 s, every scheduler |
| G2: one worker, one slot | 225.1285 s, every plan, = total work / speed | 225.13 s, all 46 configurations |
| C1: 2 L40S + 1 H200, 4 slots each: rank + fastest worker | 30.157 s (exact ranks) | 30.01 s (`DynamicList[BottomLevel, Speed]`) |
| C1: oldest-first + fast first / dslab FIFO + speed | 29.695 s | 32.89 s (`Simple`, id order) |
| C1: speed-oblivious | 37.810 s (oldest first, least loaded) | 37.15 s (`DynamicList[BottomLevel, MaxAvailableCores]`) |
| C1: static, full knowledge | -- | 25.39 s (HEFT, DLS, Lookahead[0]) = the critical path |

The two exact checks validate the DAG, the costs and the bounds. The dynamic plans agree within a
few percent where their semantics match (these C1 numbers predate the dispatch fix below). Static
HEFT with true costs reaching the critical path on this instance shows what full lookahead buys:
about 15% over any online rule.

## Simulator artefacts found and fixed

1. **Per-event dispatch.** `whelm-whole` dispatched after every single event, including the burst
   of same-instant releases when one completion readies many jobs, so whichever job happened to
   be released first took the free slots, whatever its priority. All events of an instant are now
   applied before dispatching (as a coordinator draining its queue would). With it fixed, the
   small driver and `whelm-whole` agree. It did not change the full-scale verdict: rank priority
   still loses 3-4% to oldest-first there (see "Bounds and plans").
2. **Makespan included stale wakeups.** Deferral deadlines that fired after the last completion
   moved "now" forward; makespan is now the last completion. Regression test added.
3. **`--linear-ps` capped linear throughput at the trace's 16 slots**, which slowed every job on
   a 100,000-core worker. Linear means uncapped now.
4. **Rank maintenance made non-rank plans slow.** With exact rank propagation every bidegree
   opening pushed rank updates up the whole grid, even for plans that never read ranks (5,300 s
   per plan instead of 300 s). `DagConfig::track_ranks` turns rank maintenance off.
5. **Tie-breaks matter at the ±5% level on any single instance.** The small driver and
   `whelm-whole` agree exactly once they break ties between simultaneously ready bidegrees the same
   way (s-major ids), and differ by 2-6% otherwise. Single-instance comparisons below that
   resolution are noise (Graham's anomalies); the conclusions below use many perturbed instances
   with common random numbers, and several noise seeds for the whole run.

## Ordering and placement, over many instances (`whelm-pisa`)

`whelm-pisa` simulates small instances through the real `DagScheduler` and engine: mini-Nassau
grids with random parameters, or replicas of the real world's first bidegrees (n <= 60, s <= 12,
3,466 tasks, 2 L40S + 1 H200 x 4 slots) with 20 random multiplicative perturbations of work,
estimates and whole rows/columns per sample. Each sample runs both plans on the same instance.

| A vs B | instances | mean ln(A/B) | A better in |
|---|---|---|---|
| oldest-first + fast first vs oldest-first, speed-oblivious | 200 replicas | -10.5% | 92% |
| rank (true costs) + fast first vs oldest-first + fast first | 500 grids | -6.9% | 68% (worse in 7%) |
| rank (true costs) + fast first vs oldest-first + fast first | 200 replicas | -4.2% | 74% |
| rank (estimated costs) + fast first vs oldest-first + fast first | 200 replicas | +0.8% | 41% |
| oldest-first, rank within (true costs) vs oldest-first, both fast first | 200 replicas | -1.9% | 61% |
| slow gate vs fast first (oldest-first) | 200 replicas | -1.9% | 64% |
| earliest finish, wait for >= 25% gain, vs fast first (oldest-first) | 500 grids / 150 replicas | -9.1% / -8.0% | 63% (worse 6%) / 96% |
| **rank (estimated costs) + earliest finish (>= 25% gain) vs oldest-first + fast first** | 150 replicas | **-14.4%** | **100%** |
| rank (true costs) + earliest finish (>= 25% gain) vs oldest-first + fast first | 150 replicas | -17.9% | 100% |

- **Rank priority alone is fragile:** with true costs it helps (4-7%), with realistic estimate
  error (sd 0.6) it does not. Annealing finds instances where rank + fast-first is 1.41x worse than
  oldest-first: rank fills the fast slots with high-rank work, and the job that turns out critical
  lands on a slow worker and stays there (63% of the realised critical chain on slow workers,
  against 0% for oldest-first).
- **Waiting for a fast slot fixes exactly that**, and is a large, consistent win: 7-9% with
  oldest-first, 14-18% with rank, which only pays together with it. Annealing also finds where
  waiting backfires (1.42x): a fast class barely faster than the slow one and holding a small share
  of capacity, where urgent jobs queue for scarce fast slots. A minimum gain of 25% halves those
  cases at no cost on average, hence `Defer::default()` (25%, at most an hour).
- **The slow gate** helps a little (2%) and costs nothing on average.
- **These small-instance results do not all transfer to the full run** (next section): there,
  rank priority loses 3-4% and waiting gains about 1% in makespan. The small replicas (3 workers,
  heavy contention throughout) exaggerate exactly the effects that ordering and waiting act on.

## Implicit template instances

`DagScheduler::open_instance` keeps a walk as dense counters over its shared template (PaRSEC's
parameterised task graphs): 2 B counter + 8 B work + 1 bit per node, against an explicit node,
its edges and its `JobSpec`. Readiness, entry ranks and snapshots are property-tested against
the explicit layer, and `whelm-whole` gave bit-identical makespans with explicit walks and with
instances on every plan tried. (That layer has since been replaced by lazily materialised units of
shared templates, and the `--explicit` and `--eager` flags with it. The frontier budget in open
walks, once `DagConfig::max_open_instances`, is `whelm-whole --max-open`, modelled by the simulated
coordinator.)
