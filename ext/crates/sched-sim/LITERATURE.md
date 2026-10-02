# Literature search: scheduling the Nassau whole-run DAG

Scope: precedence-constrained scheduling on related machines, uncertain durations, memory-bounded
DAG scheduling, task runtimes, backfilling, malleable work, and existing libraries. Searched
2026-10-02.

**How each citation was checked:**
- **[V]:** I opened the abstract or the paper (or a reliable index such as dblp, ACM DL or the
  publisher), and the claim attributed to it comes from that text.
- **[M]:** from memory. I am confident the paper exists, but I did not check the specific claim in
  this session.
- **[U]:** I could not verify it. Treat it as a lead only.

Context taken from `results_whole.md`:
- **Bounds:** D = 883 h is measured at the fastest single-job speed, W/P = 770 h, and the run is
  span-bound.
- **Fleet:** 7 H200 and 14 L40S workers × 16 slots. An L40S runs a job 2.41× faster.
- **Best plan so far:** group order, uncapped, fast first = 1,237 h (1.40× the lower bound, 59.5%
  slot utilisation).

---

## 1. Precedence-constrained scheduling on related or heterogeneous machines

### 1.1 Graham list scheduling and its related-machines generalisation

**Graham, "Bounds on multiprocessing timing anomalies", SIAM J. Appl. Math. 17(2), 1969** [M];
the 1966 BSTJ paper is the original.
- **Result:** any list ("greedy": never idle while a ready task exists) is within 2 − 1/m of
  optimal on identical machines, and this holds for **any** priority list.
- **Proof:** the proof is the useful part. Every instant of the schedule either has all machines
  busy or has a job of one specific chain running. That gives Cmax ≤ W/m + (length of that chain).
- **Anomalies:** the same paper shows timing anomalies. Shorter tasks, more machines or relaxed
  precedence can *increase* the makespan of a list schedule.
- **Cost knowledge:** the guarantee needs no knowledge of task durations, so it is
  non-clairvoyant.

**Liu & Liu (1974), cited in Chekuri–Bender below** [V, via Chekuri–Bender's text].
- **Result:** plain list scheduling on uniformly related machines (speeds s_i) has ratio
  1 + s_max/s_min − s_max/Σs_i.
- **For us:** that is about 3.4 for us, with s_max/s_min = 2.41. The bound depends on the speed
  ratio, and the reason is exactly ours: the chain covering the idle time may run on slow
  machines.

**Jaffe, "Efficient scheduling of tasks without full use of processor resources", TCS 26, 1980**
[V, via Chekuri–Bender].
- **Result:** list scheduling restricted to machines with speed ≥ s_max/√m is O(√m)-approximate.
- **Lesson:** the first speed-independent bound came from *not using* slow machines for some
  work.

**Chudak & Shmoys, "Approximation algorithms for precedence-constrained scheduling problems on
parallel machines that run at different speeds", SODA 1997; J. Algorithms 30, 1999** [V].
- **Result:** an O(log m) approximation for Q|prec|Cmax.
- **Reduction:** drop machines slower than s_max/m and round speeds to powers of 2, leaving
  K = O(log m) speed classes.
- **Algorithm:** an LP assigns each job to a **speed class**. "Speed-based list scheduling" then
  runs each job only on machines of its assigned class.
- **Ratio:** O(K), and K is 2 for us.
- **Relevance to us:** with two classes, their framework gives a **constant-factor** guarantee
  *if* jobs are assigned to classes so that no chain spends too long on slow machines.
  - **What to add:** this is the formal version of "pin critical work to the fast class".
  - **Algorithm to try:** a *class assignment* (critical → fast only; the rest → either), not
    just a preference.

**Chekuri & Bender, "An efficient approximation algorithm for minimizing makespan on uniformly
related machines", J. Algorithms 41(2):212–224, 2001; IPCO 1998** [V, read the PDF:
<https://chekuri.cs.illinois.edu/papers/related_jalg.pdf>].
- **Algorithm:** a combinatorial O(log m) algorithm with O(n³) running time, based on a **new
  lower bound**.
- **The bound:**
  - Take a *maximal chain decomposition* P1, P2, …: P1 is a longest chain, P2 a longest chain of
    what remains, and so on.
  - Sort the machine speeds s1 ≥ s2 ≥ ….
  - Then `C* ≥ max( AL, max_{j ≤ min(r,m)} Σ_{i≤j}|P_i| / Σ_{i≤j} s_i )`, where AL = W/Σs.
  - The bound is valid for preemptive schedules too.
- **What it generalises:** it includes D/s_max (j = 1) and W/Σs. In between, it captures "only so
  many long chains can each have a fast machine".
- **Precedent:** for disjoint chains, Horvath, Lam & Sethi (1977) showed it is the exact
  preemptive optimum.
- **Our algorithm is the same:** their algorithm is speed-based list scheduling after a
  chain-to-speed assignment.
- **Relevance to us:**
  - **(a) Report L next to W/P and D.**
    - **How:** treat each slot as a machine (112 fast, 224 slow). Compute the greedy maximal
      chain decomposition only for the first ~112–336 chains: repeated longest path with removal,
      cheap in a DAG.
    - **Why:** L is ≥ max(W/P, D) by construction and may close part of the 883 → 1,237 h gap.
    - **Caveat:** a full decomposition of 4M nodes is unnecessary. The max over j is usually
      reached at small j or at j = m.
  - **(b) A rule to try:** jobs on the top-k chains go to the fast class only. The chains are
    recomputed lazily per instantiated group.

**Shi Li, "Scheduling to minimize total weighted completion time via time-indexed linear
programming relaxations", FOCS 2017 / SICOMP; arXiv:1707.08039** [V via the abstract and a
citing paper].
- **Result:** improves Q|prec|Cmax to O(log m / log log m).
- **Hardness:** a super-constant hardness is known only under a structural hypothesis (Bazzi &
  Norouzi-Fard, ESA 2015, arXiv:1507.01906 [V]).
- **Relevance to us:** theory only. With two speed classes we are in the constant-factor regime,
  so these refinements do not matter. Cite them to explain why no strong worst-case guarantee
  exists in general.

**Beaumont, Canon, Eyraud-Dubois, Lucarelli, Marchal, Mommessin, Simon, Trystram, "Scheduling on
Two Types of Resources: A Survey", ACM Computing Surveys 53(3), 2020; arXiv:1909.11365** [V,
existence and scope].
- **Scope:** the survey for exactly our shape, two machine types (CPU/GPU). It includes unified
  implementations.
- **Relevance to us:** the first thing to read for two-class algorithms (HLP, QA, ER-LS,
  HeteroPrio).

**Canon, Marchal, Simon, Vivien, "Online Scheduling of Task Graphs on Hybrid Platforms",
Euro-Par 2018 (hal-01828301); extended as "Online scheduling of task graphs on heterogeneous
platforms", IEEE TPDS 2020 (hal-02291268)** [V for the Euro-Par abstract; I could not fetch the
TPDS PDF].
- **Model:** online DAG scheduling: a task is known when it becomes ready, and its times on both
  resource types are known then.
- **Results:**
  - A lower bound of √(m/k) for any online algorithm (m CPUs, k GPUs).
  - A (2√(m/k) + 1)-competitive algorithm.
  - A tunable mix of a system-oriented heuristic (EFT-like) and the competitive allocation rule,
    which is Θ(√(m/k))-competitive and "close to offline" in simulation.
- **Caveat:** their model is *unrelated* (task-dependent acceleration). Ours is closer to
  related, since 2.41× is roughly uniform.
- **Relevance to us:**
  - **Model match:** this is the right model for "templates instantiated when the coarse node is
    ready". It is online, and the DAG is revealed as you go.
  - **Their rule:** allocate to the fast class when the acceleration-weighted cost justifies it,
    otherwise use EFT. That is a principled replacement for "fast first".
  - **The tension it resolves:** pure EFT wastes the fast class on non-critical work, and pure
    fast-first lets slow slots idle.

**Beaumont, Eyraud-Dubois, Kumar, "Approximation proofs of a fast and efficient list scheduling
algorithm for task-based runtime systems on multicores and GPUs" (HeteroPrio), IPDPS 2017** [V].
- **Algorithm:**
  - Each resource type picks from the ready list in its own preference order: GPUs take tasks
    with the largest acceleration, CPUs the smallest.
  - **Spoliation:** when a resource goes idle, it may *abort* a task running elsewhere and
    restart it if it would finish earlier.
- **Bounds:** for independent tasks, ratio between 2 + 2/√3 and 2 + √2. Spoliation is what makes
  a bounded ratio possible.
- **Relevance to us:**
  - **Our tail:** in a span-bound run's tail, a critical task stuck on a
    slow-class slot while a fast slot is idle is pure loss.
  - **What to add:** simulate spoliation. Restart on the fast class when
    `remaining_est_on_slow > full_est_on_fast + restart_overhead`. Our tasks are restartable
    (deterministic), and work is discarded rather than corrupted.
  - **When it pays most:** when few tasks remain, at the end of the run and at narrow points of
    the grid.

### 1.2 Practical heuristics (HEFT family)

**Topcuoglu, Hariri, Wu, "Performance-effective and low-complexity task scheduling for
heterogeneous computing", IEEE TPDS 13(3):260–274, 2002** [M].
- **HEFT:** priority is the upward rank using *average* cost over machines. Each task goes to the
  machine with the **earliest finish time** (EFT), with insertion into gaps.
- **CPOP:** priority is upward + downward rank. Tasks on the critical path are all placed on the
  single "critical-path processor" (the one that minimises their total time), and others go by
  EFT.
- **Relevance to us:**
  - **Which half pays:** HEFT's gains come from both halves. Our results already show the
    machine-choice half (EFT/fast-first, −27–31%) is the one that pays.
  - **The rule to test:** CPOP's idea is the speed-class assignment above.
  - **Variant to test:** EFT that accounts for queueing, not just "prefer fast". If a fast slot
    frees up within (t_slow − t_fast), waiting for it beats starting on slow. This "wait for a
    fast slot" rule is something plain fast-first cannot do.

**Bittencourt, Sakellariou, Madeira, "DAG scheduling using a lookahead variant of the
Heterogeneous Earliest Finish Time algorithm", PDP 2010** [V, existence:
<http://www.cs.man.ac.uk/~rizos/papers/pdp10.pdf>].
- **Algorithm:** a machine choice for a task minimises the EFT of its *children*.

**Arabnejad & Barbosa, "List scheduling algorithm for heterogeneous systems by an optimistic
cost table" (PEFT), IEEE TPDS 25(3):682–694, 2014** [V].
- **Algorithm:** lookahead through an Optimistic Cost Table, OCT(t, p) = the optimistic remaining
  path cost if t runs on p. Rank is the average OCT, and the machine choice minimises EFT + OCT.
- **Complexity:** same as HEFT.
- **Relevance to us:**
  - **OCT is cheap here:** with two classes and roughly uniform speed, OCT(t, fast) vs
    OCT(t, slow) is close to "remaining critical path / speed".
  - **The resulting rule:** a task with long downstream work goes fast, and a task with little
    downstream work is fine on slow.
  - **Under lazy unfolding:** use a template-level OCT (precompute per template, plus a coarse
    remainder).

**Kwok & Ahmad, "Static scheduling algorithms for allocating directed task graphs to
multiprocessors", ACM Computing Surveys 31(4), 1999** [M]. The classic survey of list,
clustering and duplication heuristics.

**Coleman & Krishnamachari, "PISA: An adversarial approach to comparing task graph scheduling
algorithms", arXiv:2403.07120** [V].
- **Findings:** with their SAGA library, 15 algorithms on 16 datasets look similar on benchmarks,
  but each loses badly on adversarial instances.
- **Relevance to us:** a warning that "rank vs FIFO" rankings are instance-specific. Our grid DAG
  is the instance that matters.

### 1.3 Why critical-path priority can fail to beat FIFO (my synthesis, not a single citation)

1. **Graham's bound is list-independent.** Priority affects the makespan only during intervals
   when ready tasks exceed free slots.
   - **Our utilisation:** 46–60%. Contention is episodic, so the order matters only during those
     episodes.
   - **What decides the rest:** whether the covering chain's tasks start the moment they become
     ready, and on which speed. Both depend on placement and caps, not on order.
2. **On a regular grid DAG, "oldest group first" already approximates critical-path order.** The
   wavefront in (s, t) is the long path. Blelloch et al. (§3) also show that sequential-order
   (1DF) priority is time-optimal up to W/P + D *and* space-efficient. So FIFO-by-group is a
   principled choice, not a naive one.
3. **Graham anomalies plus noise.**
   - **Noisy ranks:** upward rank uses estimates with log-normal error (sd 0.6 per task), so it is
     a noisy ordering. Under noise, a list can do worse than another on the realised durations
     even if it is better on expected durations.
   - **Frozen ranks:** ranks frozen at submission (the results note says this) go stale.
   - **The robustness question:** Canon & Jeannot (§2) study how robust DAG schedules are under
     exactly this kind of noise.
4. **Rank starvation.** Without aging, rank defers low-rank tasks until they *become* critical.
   That is the 348 h worst group in the results.
   - **The fix in the literature:** dynamic slack, LST = latest start time minus now, recomputed.
     That is the RCPSP "minimum slack / LST" rule (§5), not a static rank.

### 1.4 Expected critical path under noise (bound to report)

**Canon & Jeannot, "Evaluation and optimization of the robustness of DAG schedules in
heterogeneous environments", IEEE TPDS 21(4), 2010** [V].
- **Related work:** Canon & Jeannot, "Correlation-aware heuristics for evaluating the
  distribution of the longest path length of a DAG with random weights", IEEE TPDS 2016 [V,
  existence].
- **The fact:** with random task durations, E[longest path] ≥ longest path of the expectations
  (Jensen; max is convex). With sd 0.6 per task and sd 0.3 per group, the realised critical path
  in each seeded run is longer than the nominal D, and its variance matters.
- **Relevance to us:** report, per seed, **D_realised** (the critical path on the drawn "true"
  costs at fast speed) alongside D_nominal.
  - **Why:** the honest lower bound for a seed is max(W_realised/P, D_realised, L_realised).
  - **What it may explain:** part of the 1.40× gap may be a gap to the *realised* D, which no
    scheduler can beat.

---

## 2. Uncertain durations, predictions, robustness

**Shmoys, Wein, Williamson, "Scheduling parallel machines on-line", SIAM J. Comput. 24(6), 1995**
[M].
- **Results:**
  - Online release, non-clairvoyant: a general doubling/batching conversion that loses a factor
    of about 2 vs offline.
  - List scheduling is non-clairvoyant by nature.
  - Chekuri–Bender note that the related-machines result extends to release dates through this
    paper [V].
- **Relevance to us:** a theoretical licence for making placement decisions without cost
  estimates. The risk is placement on the wrong speed.

**Möhring, Schulz, Uetz, "Approximation in stochastic scheduling: the power of LP-based priority
policies", JACM 46(6):924–942, 1999** [V, existence].
- **Scope:** stochastic durations with known distributions; the objective is the *sum of weighted
  completion times*, not makespan. Policies are list orders from LP relaxations.
- **Relevance to us:** limited. Our objective is makespan. Their analysis does justify fixed-list
  policies under stochastic durations.

**Purohit, Svitkina, Kumar, "Improving online algorithms via ML predictions", NeurIPS 2018**
[M].
- **Learning-augmented scheduling:** blends a prediction-trusting algorithm (shortest predicted
  first) with a robust non-clairvoyant one (round robin) with a trade-off parameter λ.
  - **Consistency:** near-optimal with good predictions.
  - **Robustness:** bounded loss with bad ones.
- **Relevance to us:** a design pattern. Run a *blend* of oldest-group-first (robust, needs no
  estimates) and estimate-based slack priority, and sweep λ under several noise levels (sd ×0.5,
  ×1, ×2).

**Lassota, Lindermayr, Megow, Schlöter, "Minimalistic predictions to schedule jobs with online
precedence constraints", ICML 2023 (PMLR 202:18563–18583), arXiv:2301.12863** [V].
- **Model:** non-clairvoyant scheduling where a job is revealed only once its predecessors finish.
  That is precisely lazy template instantiation.
- **Questions studied:** which *minimal* predictions help, e.g. the number of descendants or the
  chain length remaining, rather than full durations. They give bounds per precedence topology.
- **Objective:** I believe the main objective is weighted completion time [U on the makespan
  specifics].
- **Relevance to us:**
  - **Model match:** this is the closest modern formal model to our "templates unfold lazily".
  - **The takeaway:** a coarse prediction such as "remaining coarse-grid depth from this group",
    available before unfolding, may capture most of what rank offers.
  - **Experiment:** priority = remaining coarse critical path (bidegree-level only), compared
    against full task-level rank.

**Lindermayr & Megow, "Algorithms with Predictions" bibliography**
(<https://algorithms-with-predictions.github.io>) [M]. A maintained list of learning-augmented
scheduling papers.

**Graham anomalies** (see 1.1).
- **Relevance to us:** always compare policies on *common random numbers* (identical seeded
  noise) and on several seeds. A single-seed 4–7% difference (rank vs group) may lie inside the
  anomaly and noise band. Report mean ± sd over ≥ 10 seeds.

---

## 3. Memory- or frontier-bounded DAG scheduling

### 3.1 Space-efficient greedy scheduling (closest match to "bounded open groups")

**Blelloch, Gibbons, Matias, "Provably efficient scheduling for languages with fine-grained
parallelism", JACM 46(2), 1999** [V].
- **Result:** for any computation with work w, depth d and a sequential schedule using space s₁,
  a parallel schedule exists that takes fewer than w/p + d steps and less than s₁ + p·d space.
- **The schedule:** greedy with priority = *sequential (depth-first, 1DF) order*. A ready task
  earlier in the sequential order runs first.

**Narlikar & Blelloch, "Space-efficient scheduling of nested parallelism", ACM TOPLAS 21(1),
1999** [M].
- **Algorithms:** AsyncDF and DFDeques add a **memory threshold K**. A task allocating more than
  K is delayed or split.
- **Trade-off:** space s₁ + O(K·p·d), with locality and time traded through K.

**Blumofe & Leiserson, "Scheduling multithreaded computations by work stealing", JACM 46(5), 1999**
[M]. Work stealing gives T₁/P + O(T∞) expected time and S₁·P space. That is the "Cilk bound", and
its space is worse than 1DF.

**Relevance to us:**
- **Our policy already has this shape:** "oldest-group-first" is a 1DF-like priority (sequential
  order = group order).
- **The theory says:**
  - **(a)** a greedy 1DF schedule's open state is bounded by s₁ + p·d. So the frontier grows
    with *slots × depth*, not without limit, and **a hard cap is not needed for boundedness**.
  - **(b)** if a cap is needed, it should be a *space threshold* (bytes of open-group state, or
    an AsyncDF-style "premature node" budget), not a count of 24 groups.
- **Measure first:** the frontier under uncapped 1DF (results say about 280 open groups and about
  490k live nodes).
- **Then sweep:** a byte budget K, plotting makespan vs peak coordinator state as a Pareto curve.

### 3.2 Pebbling and memory-aware DAG scheduling

**Hong & Kung, "I/O complexity: the red-blue pebble game", STOC 1981** [M]; Sethi (1975) on black
pebbling register allocation [M].
- **Problem:** minimum memory (pebbles) to evaluate a DAG, which is NP-hard (PSPACE-complete for
  some variants).
- **Relevance to us:** framing only.
- **Implication:** our grid with group-state needs about O(width) pebbles under 1DF.
- **Optional bound to report:** the minimum frontier of a sequential traversal, which is s₁ in
  Blelloch's bound.

**Eyraud-Dubois, Marchal, Sinnen, Vivien, "Parallel scheduling of task trees with limited
memory", ACM TOPC 2(2), 2015 (hal-01160118)** [V].
- **Model:** trees with large edge data; a task runs only if its inputs and outputs fit in
  memory.
- **Results:** memory-aware list heuristics and the hardness of the makespan/memory bi-criteria.
- **Lineage:** Liu (1987) and Jacquelin, Marchal, Robert, Uçar (IPDPS 2011) on memory-optimal
  tree traversals [M].

**Marchal, Simon, Vivien, "Limiting the memory footprint when dynamically scheduling DAGs on
shared-memory platforms", JPDC 128:30–42, 2019** [V].
- **Idea:** add **fictitious "memory dependence" edges** to the DAG *before* execution, so that
  *any* dynamic schedule of the augmented DAG respects the memory bound.
- **Follow-up:** "Revisiting dynamic DAG scheduling under memory constraints for shared-memory
  platforms" (Gou, Marchal, Simon? [U on the author list], ICL UT 2020 report).
- **Relevance to us:** the conceptual alternative to admission control.
  - **Mechanism:** encode "group g+K can't open until group g closes" as edges in the coarse DAG.
  - **Benefit:** memory safety becomes a DAG property, and the scheduler stays pure greedy and
    simple.
  - **Fit:** our coarse DAG is known a priori, so this is feasible.
  - **Variant to try:** the cap as edges, using sequential order + K, vs the cap as an admission
    counter.

**Sbîrlea, Budimlić, Sarkar, "Bounded memory scheduling of dynamic task graphs", PACT 2014** [V].
- **Approach:** inspector/executor. A heuristic, with a fallback to an exact method, finds a
  schedule restriction that guarantees a peak-memory bound, or reports infeasibility before
  running.
- **Relevance to us:** the same "decide feasibility offline on the a priori DAG" idea. Our
  simulator *is* the inspector.

**Kayaaslan, Lambert, Marchal, Uçar, "Scheduling series-parallel task graphs to minimize peak
memory", TCS 2018** [M]. Polynomial peak-memory traversals for series-parallel graphs.
- **For us:** templates expanded by substitution are close to series-parallel at the coarse
  level, but not inside the signature DAGs.

---

## 4. Runtime systems: DAG discovery and priority schemes

**PaRSEC (Bosilca, Bouteiller, Danalis, Herault, Lemarinier, Dongarra), "DAGuE/PaRSEC", Parallel
Computing 2012 / IEEE CiSE 2013** [M].
- **PTG (Parameterized Task Graph):** the DAG is a symbolic, parameterised description (JDF).
  Successors are computed on the fly from task parameters, so the whole graph is never stored.
- **DTD:** a dynamic-discovery alternative, compared with PTG in Hoque, Herault, Bosilca,
  Dongarra, "Dynamic task discovery in PaRSEC", ScalA@SC 2017 [V].
- **Priorities:** JDF supports per-task-class priority expressions [M].
- **Relevance to us:**
  - **The match:** PTG is exactly our "coarse grid + templates" representation, a symbolic DAG
    unfolded lazily.
  - **What to copy:** keep priorities as *closed-form functions of (s, t, template position)*, as
    PTG priority expressions are, rather than computed ranks over an instantiated graph.

**StarPU (Augonnet, Thibault, Namyst, Wacrenier, CCPE 23(2), 2011)** [M]; scheduler list [V,
<https://files.inria.fr/starpu/doc/html/Scheduling.html>].
- **Schedulers:** eager, prio, ws, lws, dm, dmda (HEFT-like: completion time from
  history-calibrated performance models plus data transfer), dmdas (sorted by priority), heft,
  heteroprio, darts and others.
- **Performance models:** history-based, with calibration runs.
- **Simulation:** StarPU can run its *real* scheduler inside SimGrid [M].
- **Relevance to us:**
  - **Algorithm to add:** dmda is "EFT with per-worker queues", the natural next step beyond
    fast-first.
  - **What to copy:** the StarPU+SimGrid practice of one scheduler code path in both simulation
    and production is what we want from the `sched` crate.

**Agullo, Beaumont, Eyraud-Dubois, Kumar, "Are static schedules so bad? A case study on Cholesky
factorization", IPDPS 2016** [V].
- **Finding:** static, carefully computed priority and allocation (e.g. from a CP or LP solution)
  injected into a dynamic runtime can beat purely dynamic heuristics on CPU+GPU.
- **Relevance to us:** a hybrid is worth testing. Precompute a *class assignment* for the coarse
  grid offline, from our a priori DAG, and let dynamic greedy handle timing.

**Dask distributed** [V,
<https://distributed.dask.org/en/stable/scheduling-policies.html>, <https://docs.dask.org/en/stable/order.html>].
- **Priority tuple:**
  - **First element:** a submission-generation counter, i.e. FIFO across computations.
  - **Second element:** `dask.order`, which favours "small goals with big steps": depth-first
    toward completions that free memory, preferring critical paths.
- **Worker choice:** minimises estimated start time (queued runtime plus transfer), with ties
  broken by the least stored data.
- **Root-task queuing:** a `worker-saturation` factor (default 1.1) lets only about 1.1 × threads
  root tasks out, to keep memory low.
- **Relevance to us:** the same 1DF + memory philosophy.
  - **What to copy:** `worker-saturation` is the analogue of our in-flight caps, set *relative to
    slots* (1.1×), not as an absolute 24/32.
  - **Parameter to expose:** in-flight caps as multiples of free slots.

**Legion/Realm (Bauer, Treichler, Slaughter, Aiken, SC 2012)** [M].
- **Design:** a dynamic dependence analysis over logical regions, with a pluggable **Mapper**
  interface that holds placement and priority policy separately from the runtime.
- **What to copy:** the mapper interface is a good model for a "pure scheduling core" API.

**Ray (Moritz et al., OSDI 2018)** [M]. A bottom-up, locality-first distributed scheduler with no
global priority.
- **For us:** not a fit for a span-bound DAG.

**Taskflow (Huang, Lin, Lin, Wong, IEEE TPDS 33(6), 2022)** [M]. C++ static graphs plus
*dynamic subflows* (a task spawns a subgraph at run time), similar to template instantiation;
scheduled by work stealing.

**HPX** [M]. Futures and dataflow, with no notable priority theory.

**Cilk** [M]. See Blumofe–Leiserson in §3.

**Pure, deterministic scheduling core:**
- **Libraries that come close:**
  - dslab-dag (§7) exposes a `Scheduler` trait driven by a simulator.
  - Batsim (§5) separates the decision process behind an event protocol.
  - The Legion mapper.
  - StarPU's modular schedulers.
- **Gap:** I found no production runtime that advertises a deterministic, IO-free scheduling core
  usable unchanged in both simulation and production [U]. Our design appears uncommon.

---

## 5. Batch scheduling with memory or resource constraints

**Lifka, "The ANL/IBM SP scheduling system", JSSPP 1995, LNCS 949** [M].
- **EASY backfilling:** FCFS queue, a reservation for the *head* job only, and later jobs may
  start if they do not delay that reservation.

**Mu'alem & Feitelson, "Utilization, predictability, workloads, and user runtime estimates in
scheduling the IBM SP2 with backfilling", IEEE TPDS 12(6):529–543, 2001** [V].
- **Comparison:** EASY vs conservative backfilling (a reservation for *every* queued job).
- **Main finding:** **backfilling performs better when runtime estimates are overestimated**,
  counter-intuitively.
- **Also:** conservative backfilling gives predictability, and EASY gives better utilisation and
  response time in most workloads.

**Tsafrir, Etsion, Feitelson, "Backfilling using system-generated predictions rather than user
runtime estimates", IEEE TPDS 18(6):789–803, 2007** [V].
- **Approach:** use history-based predictions, and when a job outlives its prediction, extend the
  prediction and recompute the reservations.
- **Relevance to us:**
  - **Estimates:** our admission reservation needs end-time estimates for running tasks to
    compute the "shadow time".
  - **(a)** use an *upper* quantile of the log-normal (e.g. p80) for the shadow-time computation,
    following Mu'alem–Feitelson.
  - **(b)** extend the estimate on overrun, following Tsafrir.
  - **(c)** expose reservation depth k (EASY = 1, conservative = ∞) and sweep k ∈ {0, 1, 2, 4}.

**Srinivasan, Kettimuthu, Subramani, Sadayappan, "Characterization of backfilling strategies for
parallel job scheduling", ICPP Workshops 2002** [M].
- **Findings:** reservation depth, starvation, and selective reservation (reserve only after a job
  has waited past an expansion-factor threshold).
- **Relevance to us:** "selective reservation" is principled aging. Reserve memory only for a
  task whose wait exceeds X, instead of always reserving for the top task.

**Dutot, Mercier, Poquet, Richard, "Batsim: a realistic language-independent resources and jobs
management systems simulator", JSSPP 2016 (LNCS 10353, 2017)** [V].
- **Design:** an RJMS simulator on SimGrid with an event protocol, plus EASY and other reference
  schedulers.

**RCPSP (resource-constrained project scheduling):**
- **Kolisch, "Serial and parallel resource-constrained project scheduling methods revisited:
  theory and computation", EJOR 90(2), 1996** [M].
- **Kolisch & Hartmann, "Experimental investigation of heuristics for resource-constrained
  project scheduling: an update", EJOR 174(1), 2006** [M].
- **Findings:**
  - The **parallel schedule-generation scheme** (time-stepping, like our dispatcher) is a
    non-delay scheme.
  - **Priority rules:** LFT (latest finish time), LST and MSLK (minimum slack), and WCS/GRPW. In
    their experiments LFT and minimum slack are consistently among the best single-pass rules.
- **Stochastic RCPSP:** "preselective policies" (Möhring, Stork) handle random durations [M].
- **Relevance to us:**
  - **Rule to add:** dynamic **minimum-slack** priority, slack = LST − now, recomputed as tasks
    finish. It differs from frozen upward rank because it reflects realised progress.
  - **How to compute it:** LST needs a deadline. Use the current makespan estimate, i.e.
    D-from-now on the remaining DAG.

---

## 6. Malleable or moldable tasks and processor sharing

**Feldmann, Kao, Sgall, Teng, "Optimal on-line scheduling of parallel jobs with dependencies",
J. Combinatorial Optimization 1:393–411, 1998; STOC 1993** [V for existence and the
malleable/dependency model].
- **Model:** online malleable jobs whose dependencies are revealed as predecessors finish, with
  unknown running times.
- **Result:** I recall an optimal competitive ratio of 1 + φ ≈ 2.618 [U], achieved by giving each
  job a fixed fraction of processors.

**Also relevant:** "Improved online scheduling of moldable task graphs under common speedup
models" (arXiv:2304.14127, 2023) [V, existence]; "Multi-resource list scheduling of moldable
parallel jobs under precedence constraints" (arXiv:2106.07059) [V, existence].

**Relevance to us:**
- **No change while sharing is linear:** throughput is linear in concurrency up to 16, so a
  worker is exactly 16 independent slots and the malleable theory adds nothing. The advice
  changes only if per-job speed depends on co-runners (sublinear processor sharing, memory
  bandwidth).
  - **Then:** a critical task should run on a *lightly loaded* fast worker.
  - **Check:** do the trace data show per-job slowdown at high occupancy?
- **Multi-resource list scheduling:** memory + slots is a 2-resource problem. arXiv:2106.07059
  gives list-scheduling ratios that grow with the number of resource types d.
  - **For us:** with d = 2 the loss is modest. Memory admission is a second "machine dimension",
    not a reason to abandon greedy.

---

## 7. Existing libraries to compare against

| Library | Language | What | Verified |
|---|---|---|---|
| **dslab-dag** (<https://osukhoroslov.github.io/dslab/docs/dslab_dag/index.html>) | **Rust** | DAG execution simulator with built-in HEFT, Lookahead, PEFT, DLS and others; user-defined `Scheduler` trait; used in "Benchmarking DAG scheduling algorithms on scientific workflow instances" | [V] |
| SAGA (Coleman & Krishnamachari) | Python | 15+ task-graph schedulers (HEFT, CPOP, …) and the PISA adversarial comparison | [V] |
| Batsim / pybatsim | C++ / Python | Batch RJMS simulator with EASY backfilling reference schedulers | [V] |
| `peft` (github.com/mackncheesiest/peft) | Python | Small reference PEFT implementation | [V, existence] |
| StarPU + SimGrid | C | Real runtime schedulers (dmda, heteroprio) in simulation | [M] |
| WRENCH | C++ | Workflow simulation on SimGrid | [M] |
| Taskflow | C++ | Runtime, not a scheduler-comparison tool | [M] |

**Other Rust crates:**
- **What exists:** some Rust DAG executors exist (async_dag, dagrs), but they run tasks at
  maximal parallelism without heterogeneous list scheduling.
- **Backfilling:** I found no Rust crate implementing backfilling.
- **Relevance to us:** **dslab-dag is the comparison target.**
  - **Cross-check:** port a small instance, e.g. one coarse row with templates, to dslab-dag's
    format and run its HEFT/PEFT. This checks our simulator's makespans and our HEFT
    implementation against an independent one.
  - **Limitation:** dslab-dag does not model lazy unfolding or processor-sharing slots as we do
    [U on its resource model details].

---

## 8. Bounds to report alongside W/P and D

1. **W/P = 770 h and D_fast = 883 h** (already reported; D is at the fastest single-job speed,
   which is correct).
2. **Chekuri–Bender chain bound L.**
   - **Formula:** `max_j Σ_{i≤j}|P_i| / Σ_{i≤j} s_i` over a greedy maximal chain decomposition,
     with slots as machines sorted by speed.
   - **Order:** L ≥ max(W/P, D), and it is valid for preemptive schedules too.
   - **Cost:** compute the first few hundred chains by repeated longest-path-with-removal.
3. **Realised bounds per seed:** W_real/P, D_real and L_real on the drawn true costs. Report the
   makespan ratio against these. By Jensen, D_real ≥ D_nominal in expectation.
4. **Graham-style a posteriori certificate per simulated run.**
   - **Split:** time where all slots are busy (the load part) vs time where some slot idles.
   - **Covering chain:** for the idle part, extract the chain that covered it, and how much of it
     ran on slow slots or *waited ready*.
   - **What it says:** directly which lever remains: speed placement, caps, or order.
5. **Liu & Liu's speed-oblivious ratio** 1 + s_max/s_min − s_max/Σs ≈ 3.4. Use it as context for
   why speed-oblivious placement (today's plan, 2.40×) is not covered by the Graham 2× intuition.
6. **Optional:** Blelloch's space bound s₁ + p·d, to set the frontier cap scale. Measure s₁ as the
   max open groups/state of a sequential 1DF traversal.

---

## 9. Ranked: top 5 experiments or algorithms for the simulator

1. **Critical-task speed-class assignment plus spoliation** (Chudak–Shmoys, Chekuri–Bender,
   CPOP, HeteroPrio).
   - **Rule:** tasks with low slack (or on the top-k maximal chains) may run *only* on the fast
     class. Others prefer fast but take slow freely.
   - **Spoliation:** when a fast slot idles and a critical task is running on slow with
     `remaining_slow > est_fast`, restart it on fast.
   - **Why first:** placement is already the largest lever (−27–31%) and this is its principled
     form.
   - **Expected effect:** largest in the tail and at grid bottlenecks.
2. **Report L, D_real and the Graham certificate (§8).**
   - **Cost:** cheap, and needs no policy change.
   - **Why:** it tells whether 1.40× is mostly unavoidable (realised D, chain-on-fast-slot
     limits) or still recoverable, and so whether items 1, 3 and 4 can pay at all.
3. **EFT placement with "wait for a fast slot"** (HEFT/dmda/PEFT machine choice).
   - **Rule:** place on the class minimising estimated finish = max(now, earliest free slot of
     class) + cost/speed. If a fast slot frees soon, wait for it instead of starting on slow.
   - **Comparisons:** compare with crude fast-first, and with a PEFT-style OCT term (downstream
     critical path) in the choice.
   - **Robustness:** test with noisy estimates.
4. **Dynamic minimum-slack priority with a robust blend** (RCPSP MSLK/LFT; learning-augmented
   blending).
   - **Priority:** λ·(oldest-group order) + (1−λ)·(slack recomputed from realised progress at the
     coarse level, i.e. remaining bidegree-level critical path, per Lassota et al.'s "minimal
     predictions").
   - **Sweep:** λ, and estimate noise ×{0.5, 1, 2}.
   - **Method:** common random numbers, ≥ 10 seeds, because of Graham anomalies.
   - **Hypothesis:** in contention episodes only, slack beats FIFO, so gains are small but
     non-negative.
5. **Space-threshold cap instead of count caps; memory edges vs admission** (Blelloch/AsyncDF,
   Dask worker-saturation, Marchal–Simon–Vivien).
   - **Cap to sweep:** replace "≤24 groups, ≤32 tasks/group" with a byte or node budget K on open
     group state (1DF order, allow out-of-order groups only within K).
   - **Output:** a Pareto curve of makespan vs peak coordinator state.
   - **Alternative arm:** encode the cap as fictitious coarse edges (group g+K depends on group g
     closing).
   - **Memory admission:** with memory modelled, also sweep reservation depth k ∈ {0, 1, 2, ∞}
     (EASY vs conservative), shadow-time estimate quantile p50/p80, and selective reservation
     (reserve only after a wait > X).

**Caveats:**
- Items 1 and 3 overlap. Run 3 first as the baseline for 1.
- The expected sizes of effects are my judgement from the theory and the current results table,
  not literature numbers.
