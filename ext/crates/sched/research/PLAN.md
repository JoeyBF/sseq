# Extraction plan from reference implementations

This plan merges five per-package reports, each in this directory with file:line citations into
the reference sources:

- [`extract_dslab.md`](extract_dslab.md)
- [`extract_starpu.md`](extract_starpu.md)
- [`extract_batsim.md`](extract_batsim.md)
- [`extract_saga.md`](extract_saga.md)
- [`extract_parsec.md`](extract_parsec.md)

The effort figures are the reports' own estimates.

## What the five agree on

1. **Ordering: nothing new to take.** With zero communication cost and one speed ratio for every
   task, HEFT, PEFT, BIL and the dslab ranks all reduce to our upward rank, up to a constant
   (dslab and SAGA reports). Why rank loses to oldest-bidegree-first is a property of our DAG,
   not a missing algorithm, so it calls for an experiment (item 6), not an import.
2. **Placement is where the prior art helps, and all five point there:**
   - dslab's resource criterion `Speed`;
   - StarPU's dmda (expected finish time) and heteroprio's slow-worker gate;
   - SAGA's HEFT/MCT earliest-finish placement and FastestNode;
   - PaRSEC's `parsec_select_best_device`.

   Today speed-aware placement exists only in the simulator, faked through `JobSpec::prefer`. The
   dslab report notes that under `BestFit`, `prefer` is only a tie-breaker.
3. **None of them models memory admission with an escape hatch, or processor sharing.** Those
   stay ours.

## Prioritised plan

| # | Item | Source | Where | Effort | Expected |
|---|---|---|---|---|---|
| 1 | `WorkerState::speed`, and a speed-aware worker choice: `FastestFirst`, plus heteroprio's slow-worker gate (a slow worker takes a job only if backlog / fast workers ≥ a factor) | dslab E1; StarPU `heteroprio.c:3602-3606`; PaRSEC `device.c:107-288` | engine | 1–2 d | Makes the measured −27–31% a library feature |
| 2 | `JobSpec::work` (estimate) and an `EarliestFinish` worker choice from per-slot free times (StarPU's multi-slot formula, `component_sched.c:706-734`); then optionally *waiting* for a fast slot, via a `Policy::next_wakeup()` hook | dslab E2; StarPU dmda; SAGA HEFT/MCT | engine + DAG layer (fills `work`) | 2–3 d | Unproven: dslab's synthetic grid gained only ~1% from waiting. Measure in `sched-whole` before going further |
| 3 | New simulator plans and bounds: fast-class-only (bound about 920 h); CPOP-style pinning of critical tasks to fast; group order between bidegrees and rank within; Chekuri–Bender chain bound L; critical path on the realised (noisy) costs | SAGA; LITERATURE §8 | `sched-whole` | 2 d | Shows how much of the 1.40× gap is reachable |
| 4 | Learned per-class speed (log space, from `completed` durations), optionally a per-bidegree correction. Do **not** copy StarPU's ±50% outlier filter, which would discard most samples at our noise (sd 0.6) | StarPU perfmodel | engine (optional model fed by events) | 2 d | Removes a hand-set speed table |
| 5 | Cross-validation: export ≤ 5k-task sub-regions to dslab YAML; exact agreement expected with unlimited capacity (= critical path) and on dslab's 10-task test, 2–5% under contention. Exports must list tasks in topological order and give join nodes non-zero work (dslab's HEFT and Lookahead deadlock otherwise) | dslab | test tooling | 1–2 d | Independent check of our simulator |
| 6 | `sched-pisa`: adversarial search over our-shaped DAGs (mini grid plus templates, seeded from a scaled-down replica of the real run, multiplicative perturbations, standard annealing); shrink witnesses to minimal ones and regress the win/loss ratio on DAG features | SAGA PISA | new sim binary | 3–4 d | Settles rank vs oldest-first |
| 7 | Implicit template instances: per-node counters for an open bidegree instead of graph nodes and edges (estimated ~64 KiB vs ~13 MB per A(4) bidegree); "walk done" as a count of remaining sinks; no per-task completed-id set (fixes the ~4.1M-id leak in `sched-whole`); a frontier budget measured in state, with hysteresis; a stall detector | PaRSEC PTG and DTD | DAG layer | 6–8 d, **profile first** | Needed before driving the real coordinator at scale |
| 8 | EASY-style shadow time on the reserved worker (backfill jobs that end before it, or fit beside the holder); reserve the worker that frees soonest; fall back to full drain after overrunning by a grace period (keeps the starvation bound); bounded-slowdown and stretch metrics | Batsim / batsched `easy_bf_fast.cpp` | engine + `sched-sim` | 4–5 d (+4–5 for conservative backfilling) | Small: draining idles ~1% of slot time |
| 9 | Spoliation: `dispatch_full` returning starts plus preemptions `{job, from, to}` (the default returns none); a caller kill/restart contract; 8 invariants including the race where the old copy finishes first | StarPU report (designed from the paper; StarPU itself does not implement it) | engine API | 4–5 d | Agent's estimate 0–10%, mainly for the tail |

**Do not import:**
- SJF backfill ordering (breaks the priority invariant).
- Walltime kills, rigid multi-host jobs.
- Static whole-DAG planners (HEFT and the like plan everything up front; we are online and expand lazily).
- Network and data-transfer models, MPI and owner-computes.
- Heteroprio's acceleration buckets (they collapse at a single speed ratio).

## Suggested order

1. **Items 1, 3, 2, 5:** speed and earliest-finish placement as library features, the bounds that
   show what is left, and an external check. About 1.5 weeks, simulator-measurable.
2. **Item 6:** explain the ordering result before tuning ordering further.
3. **Item 7:** the DAG-layer rework that real coordinator integration needs.
4. **Items 4, 8, 9:** refinements, each gated on a simulated gain.

Each step keeps the existing invariants (no over-commit, escape hatch, priority, no-starvation
bound, determinism). Each step is accepted only if `sched-whole` and `sched-sim` show the
expected effect.

## Status (2026-10-02)

All nine items are implemented and tested; outcomes in `RESULTS.md`.

| # | item | status | outcome |
|---|---|---|---|
| 1 | `WorkerState::speed`, `FastestFirst`, slow gate | done | fast-first −10.5% on replicas; gate −1.9% |
| 2 | `EarliestFinish` with deferral, `next_wakeup` | done | −7..9% over fast-first; −14..18% with rank; `Defer::default()` = 25% gain, 1 h |
| 3 | plans and bounds: fast-only, CPOP, group-then-rank | done | see the whole-run section |
| 4 | learned per-class speed | done | exact recovery in tests; off by default |
| 5 | dslab-dag cross-check | done | exact on unlimited capacity and one slot; within ~1% on contention |
| 6 | `sched-pisa` | done | found a simulator artefact (per-event dispatch) and the mechanism (rank parks critical jobs on slow workers; waiting fixes it); but at full scale rank still loses 3-4% |
| 7 | implicit instances, frontier budget, snapshots | done | bit-identical makespans to the explicit layer |
| 8 | shadow backfill | done | negligible on the trace (idle 0.97% → 0.92%); off by default |
| 9 | spoliation | done | −4..8% over fast-first alone, nothing on top of waiting; off by default |
