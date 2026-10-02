# Results on the reference trace

Trace `sched_trace_40518773.jsonl.gz`: 164,114 jobs (174 zero, 163,940 signature), 174 bidegrees,
21 workers (6 H200, 15 L40S worker processes, 16 slots each, 123.7 / 154.6 GB budgets), 33 h,
812,950 dependency edges. Reproduce with `sched-sim --trace <trace> [--closed [--rank]]
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
`(k_sat, alpha)` per class (method on `sched::sim::model::fit`). 163,977 jobs, 344 fixed effects:

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

## Limitations

- RSS and the baseline are replayed from the trace, not modelled, so the simulation cannot show
  memory overruns caused by a policy, nor a drained worker's RSS dropping.
- Workers join at their trace join times and never leave, and job durations follow the
  processor-sharing model, so failures, retries and per-job variance beyond the model are absent.
- Closed loop uses each job's measured gap after its dependencies and oracle work estimates for
  ranks (the fitted `W`). Production's ranks would use estimates.
- Lookahead reservation (spec §4b, optional) is not implemented.
