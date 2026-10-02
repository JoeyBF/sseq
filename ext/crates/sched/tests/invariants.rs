//! Property tests of the policy invariants over random event streams.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use sched::{
    Attempt, Config, DIMS, Defer, FailKind, Fit, GaveUp, GroupOrder, Input, JobId, JobSpec, Order,
    Output, Policy, Reservations, Resources, RetryConfig, Scheduler, Speculate, SpeedConfig,
    SpeedPolicy, Tried, WorkerId, WorkerState,
};

#[derive(Clone, Debug)]
enum Kind {
    Fifo,
    Backfill {
        max_res: usize,
        per_class: bool,
        age: Option<f64>,
        group_first: bool,
    },
    BestFit {
        penalty: f64,
        age: Option<f64>,
    },
}

impl Kind {
    /// The policy under test.
    fn build(&self, speed: SpeedConfig, retry: RetryConfig) -> Box<dyn Policy> {
        let config = |max, per_class, age_limit, group_first, group_order| Config {
            order: Order::Priority {
                default_priority: 0,
                group_order,
                group_first,
            },
            reservations: Some(Reservations {
                reserve_after: 30.0,
                max,
                per_class,
                shadow_backfill: false,
            }),
            age_limit,
            speed,
            retry,
            ..Config::default()
        };
        Box::new(Scheduler::new(match *self {
            Kind::Fifo => Config {
                speed,
                retry,
                ..Config::fifo()
            },
            Kind::Backfill {
                max_res,
                per_class,
                age,
                group_first,
            } => config(max_res, per_class, age, group_first, GroupOrder::Arrival),
            Kind::BestFit { penalty, age } => Config {
                fit: Fit::Tightest {
                    prefer_penalty: penalty,
                },
                ..config(1, false, age, false, GroupOrder::Id)
            },
        }))
    }

    /// Whether the priority invariant applies.
    fn priority(&self) -> bool {
        !matches!(self, Kind::Fifo)
    }

    /// The configured age limit.
    fn age(&self) -> Option<f64> {
        match *self {
            Kind::Backfill { age, .. } | Kind::BestFit { age, .. } => age,
            _ => None,
        }
    }

    /// Whether groups are ordered by id (else by first arrival).
    fn by_id(&self) -> bool {
        matches!(self, Kind::BestFit { .. })
    }

    /// Whether groups come before priorities.
    fn group_first(&self) -> bool {
        matches!(
            self,
            Kind::Backfill {
                group_first: true,
                ..
            }
        )
    }

    /// The reservation limit and whether it is per class.
    fn max_reservations(&self) -> (usize, bool) {
        match *self {
            Kind::Fifo => (0, false),
            Kind::Backfill {
                max_res, per_class, ..
            } => (max_res, per_class),
            Kind::BestFit { .. } => (1, false),
        }
    }
}

#[derive(Clone, Debug)]
enum Op {
    Submit {
        demand: u64,
        group: u64,
        priority: Option<i64>,
        prefer: Option<WorkerId>,
        avoid: Option<WorkerId>,
        avoid_soft: bool,
        class: Option<u8>,
        work: Option<u32>,
        dev: Option<u64>,
    },
    /// A live attempt (running job, then attempt, by index) finishes.
    Complete(usize, usize),
    /// A live attempt fails.
    Fail(usize, usize, FailKind),
    /// A report about an attempt that is not live (finished, failed, stopped or never started).
    Stale {
        job: usize,
        attempt: usize,
        done: bool,
    },
    Cancel(usize),
    Worker {
        id: WorkerId,
        slots: usize,
        budget: u64,
        used: u64,
        baseline: u64,
        class: u8,
        dev_cap: u64,
        per_task: u64,
    },
    Gone(WorkerId),
    Tick(u32),
}

/// An index into a collection of up to `1 << 16` elements.
fn index() -> impl Strategy<Value = usize> {
    any::<prop::sample::Index>().prop_map(|i| i.index(1 << 16))
}

/// A random event.
fn op() -> impl Strategy<Value = Op> {
    let kind = prop_oneof![
        Just(FailKind::DeviceOom),
        Just(FailKind::Other),
        Just(FailKind::Timeout)
    ];
    prop_oneof![
        6 => (
            1u64..80,
            0u64..4,
            prop::option::weighted(0.2, -2i64..3),
            prop::option::of(0u64..4),
            prop::option::weighted(0.2, 0u64..4),
            any::<bool>(),
            prop::option::weighted(0.15, 0u8..2),
            prop::option::weighted(0.7, 1u32..120),
            prop::option::weighted(0.5, 1u64..40),
        )
            .prop_map(|(demand, group, priority, prefer, avoid, avoid_soft, class, work, dev)| {
                Op::Submit {
                    demand,
                    group,
                    priority,
                    prefer,
                    avoid,
                    avoid_soft,
                    class,
                    work,
                    dev,
                }
            }),
        4 => (index(), index()).prop_map(|(j, a)| Op::Complete(j, a)),
        2 => (index(), index(), kind).prop_map(|(j, a, k)| Op::Fail(j, a, k)),
        1 => (index(), index(), any::<bool>())
            .prop_map(|(job, attempt, done)| Op::Stale { job, attempt, done }),
        1 => index().prop_map(Op::Cancel),
        2 => (
            0u64..4,
            0usize..5,
            20u64..150,
            0u64..150,
            0u64..60,
            0u8..2,
            prop_oneof![Just(0u64), 20u64..100],
            prop_oneof![Just(0u64), 1u64..30],
        )
            .prop_map(|(id, slots, budget, used, baseline, class, dev_cap, per_task)| {
                Op::Worker { id, slots, budget, used, baseline, class, dev_cap, per_task }
            }),
        1 => (0u64..4).prop_map(Op::Gone),
        3 => (0u32..90).prop_map(Op::Tick),
    ]
}

/// Speed of a worker class in the tests: class 1 is the fast class.
fn class_speed(class: u8) -> f64 {
    if class == 1 { 2.5 } else { 1.0 }
}

/// A random speed configuration.
fn speed() -> impl Strategy<Value = SpeedConfig> {
    let policy =
        prop_oneof![
            Just(SpeedPolicy::Oblivious),
            Just(SpeedPolicy::FastestFirst),
            Just(SpeedPolicy::EarliestFinish(None)),
            (
                prop_oneof![Just(0.0), Just(0.2)],
                prop_oneof![Just(40.0), Just(500.0)]
            )
                .prop_map(|(min_gain, max_wait)| SpeedPolicy::EarliestFinish(Some(
                    Defer { max_wait, min_gain }
                ))),
        ];
    let speculate = prop::option::weighted(
        0.4,
        (prop_oneof![Just(0.0), Just(0.25)], 1u32..3).prop_map(|(min_gain, max_per_job)| {
            Speculate {
                min_gain,
                restart_overhead: 0.0,
                max_per_job,
            }
        }),
    );
    (policy, speculate).prop_map(|(policy, speculate)| SpeedConfig {
        policy,
        learn: None,
        speculate,
    })
}

/// A random policy configuration.
fn kind() -> impl Strategy<Value = Kind> {
    let age = prop::option::of(prop_oneof![Just(0.0), Just(45.0), Just(200.0)]);
    prop_oneof![
        Just(Kind::Fifo),
        (0usize..3, any::<bool>(), age.clone(), any::<bool>()).prop_map(
            |(max_res, per_class, age, group_first)| Kind::Backfill {
                max_res,
                per_class,
                age,
                group_first,
            }
        ),
        (prop_oneof![Just(0.0), Just(0.25)], age)
            .prop_map(|(penalty, age)| Kind::BestFit { penalty, age }),
    ]
}

/// A random retry limit (0 counts as 1).
fn retry() -> impl Strategy<Value = RetryConfig> {
    (0u32..5).prop_map(|max_attempts| RetryConfig { max_attempts })
}

/// A job as the model sees it, across its attempts.
#[derive(Clone, Debug)]
struct SJob {
    spec: JobSpec,
    since: f64,
    seq: u64,
    /// The last attempt number started.
    attempts: Attempt,
    tried: Vec<Tried>,
    speculated: u32,
}

/// A live attempt.
#[derive(Clone, Debug)]
struct Live {
    attempt: Attempt,
    worker: WorkerId,
    started: f64,
}

/// A running job and its live attempts, in start order.
#[derive(Clone, Debug)]
struct SRun {
    job: SJob,
    live: Vec<Live>,
}

#[derive(Default)]
struct Shadow {
    now: f64,
    workers: BTreeMap<WorkerId, WorkerState>,
    waiting: BTreeMap<JobId, SJob>,
    running: BTreeMap<JobId, SRun>,
    /// The last attempt number ever started per job, kept after the job ends.
    started: BTreeMap<JobId, Attempt>,
    groups: BTreeMap<u64, u64>,
    seq: u64,
    next_id: JobId,
    group_first: bool,
    by_id: bool,
    max_attempts: u32,
    /// Non-start outputs the next poll must return, in order.
    expect: Vec<Output>,
}

impl Shadow {
    /// Live attempts and their summed demand on a worker.
    fn load(&self, w: WorkerId) -> (usize, Resources) {
        self.running
            .values()
            .flat_map(|r| {
                r.live
                    .iter()
                    .filter(move |l| l.worker == w)
                    .map(move |_| r.job.spec.demand)
            })
            .fold((0, Resources::ZERO), |(n, m), d| (n + 1, m + d))
    }

    /// The production rule, written out again: in every dimension whose capacity is known
    /// (nonzero), each job counting at least `per_task`.
    fn admits(&self, demand: Resources, w: WorkerId) -> bool {
        let s = &self.workers[&w];
        let (running, placed) = self.load(w);
        if running >= s.slots {
            return false;
        }
        if running == 0 {
            return true;
        }
        (0..DIMS).all(|d| {
            let held = placed[d].max(running as u64 * s.per_task[d]);
            let used = s.reported_used[d].max(s.reported_baseline[d] + held);
            s.budget[d] == 0 || used + demand[d].max(s.per_task[d]) <= s.budget[d]
        })
    }

    /// The hard constraints (class, avoid list), written out again: a soft avoid list lapses
    /// while no live worker of the class is off it.
    fn eligible(&self, spec: &JobSpec, w: WorkerId) -> bool {
        let class_ok = |w: &WorkerState| spec.class.as_ref().is_none_or(|c| *c == w.class);
        if !class_ok(&self.workers[&w]) {
            return false;
        }
        !spec.avoid.contains(&w)
            || (spec.avoid_soft
                && !self
                    .workers
                    .values()
                    .any(|o| o.slots > 0 && class_ok(o) && !spec.avoid.contains(&o.id)))
    }

    /// Scan order: aged jobs by age, then priority, group arrival, FIFO (or group arrival before
    /// priority with `group_first`).
    fn urgency(&self, j: &SJob, age: Option<f64>) -> (bool, i64, u64, i64, u64) {
        let p = j.spec.priority.unwrap_or(0);
        let g = if self.by_id {
            j.spec.group
        } else {
            self.groups[&j.spec.group]
        };
        if age.is_some_and(|a| self.now - j.since >= a) {
            (false, 0, 0, 0, j.seq)
        } else if self.group_first {
            (true, 0, g, p, j.seq)
        } else {
            (true, p, g, 0, j.seq)
        }
    }

    /// The expected run time of `spec` on worker `w`.
    fn eta(&self, spec: &JobSpec, w: WorkerId) -> Option<f64> {
        Some(spec.work? / self.workers[&w].speed)
    }

    /// When a running job is expected to end: its earliest live attempt's expected end, an
    /// overrunning attempt counting as half done.
    fn expected_end(&self, r: &SRun) -> Option<f64> {
        r.live
            .iter()
            .map(|l| {
                let end = l.started + self.eta(&r.job.spec, l.worker)?;
                Some(if end > self.now {
                    end
                } else {
                    self.now + (self.now - l.started).max(0.0)
                })
            })
            .reduce(|a, b| Some(a?.min(b?)))
            .flatten()
    }

    /// The `i`-th running job and its `k`-th live attempt, if any job runs.
    fn pick_live(&self, i: usize, k: usize) -> Option<(JobId, usize, Attempt)> {
        if self.running.is_empty() {
            return None;
        }
        let (&j, r) = self.running.iter().nth(i % self.running.len()).unwrap();
        let k = k % r.live.len();
        Some((j, k, r.live[k].attempt))
    }

    /// Submit to both the model and the policy.
    fn submit(&mut self, spec: JobSpec, p: &mut dyn Policy) {
        let seq = self.seq;
        self.seq += 1;
        self.groups.entry(spec.group).or_insert(seq);
        self.waiting.insert(
            spec.id,
            SJob {
                spec: spec.clone(),
                since: self.now,
                seq,
                attempts: 0,
                tried: Vec::new(),
                speculated: 0,
            },
        );
        p.handle(Input::Submit(spec), self.now);
    }

    /// End a running job: every live attempt but `except` is stopped.
    fn stop_all(&mut self, job: JobId, except: Option<Attempt>) {
        let r = self.running.remove(&job).unwrap();
        for l in r.live {
            if Some(l.attempt) != except {
                self.expect.push(Output::Stop {
                    job,
                    attempt: l.attempt,
                    worker: l.worker,
                });
            }
        }
    }

    /// Live attempt `k` of `job` failed: once no attempt is live, the job is retried in its
    /// original place, softly avoiding every worker it failed on, or given up.
    fn fail(&mut self, job: JobId, k: usize, kind: FailKind, why: String) {
        let r = self.running.get_mut(&job).unwrap();
        let l = r.live.remove(k);
        r.job.tried.push(Tried {
            worker: l.worker,
            kind,
            why,
        });
        if !r.live.is_empty() {
            return;
        }
        let mut j = self.running.remove(&job).unwrap().job;
        if j.tried.len() < self.max_attempts.max(1) as usize {
            for t in &j.tried {
                if !j.spec.avoid.contains(&t.worker) {
                    j.spec.avoid.push(t.worker);
                }
            }
            j.spec.avoid_soft = true;
            self.waiting.insert(job, j);
        } else {
            let retryable = j.tried.iter().all(|t| t.kind == FailKind::DeviceOom);
            self.expect.push(Output::GaveUp(GaveUp {
                job,
                tried: j.tried,
                retryable,
            }));
        }
    }
}

/// Runs the stream, checking every invariant; returns the outputs and explanations made.
///
/// Every policy is driven by the same random stream while a model, written independently of the
/// engine, keeps its own bookkeeping of jobs and their attempts and re-checks each start as it is
/// made:
///
/// - **messages**: `Done` completes a job and stops its other attempts, `Failed` and `WorkerGone`
///   retry it in its original place (softly avoiding where it failed) or give it up, `Cancel`
///   stops every live attempt, and reports about attempts that are not live change nothing --
///   every non-start output is exactly what the model predicts;
/// - **attempts**: each start is the job's next attempt number; a job has two live attempts only
///   by speculation, and a job with a live attempt is never also waiting;
/// - **no over-commit**: the production admission rule held at the moment of each start, in every
///   dimension (no enforced capacity is exceeded, escape hatch aside);
/// - **hard constraints**: no attempt runs on a worker its class or avoid list excludes (a soft
///   avoid list only while some live worker of the class is off it);
/// - **escape hatch**: after a poll, no worker with a free slot is empty while a job that may run
///   there waits;
/// - **priority** (priority order): when B is placed on w, every more urgent waiting job was
///   refused by w at that moment (by the admission rule, or because w was B's reservation, or
///   because it chose to wait for a faster worker);
/// - **speculation**: a second attempt goes to a strictly faster, unreserved worker that no
///   waiting job wanted, is expected to gain at least `min_gain` of its run time, and respects
///   `max_per_job`;
/// - **bookkeeping**: per-worker slots and placed demand equal the sum over live attempts, job
///   counts and reservations agree with the model after every event, and everything is released
///   at the end;
/// - **determinism**: replaying the stream gives identical outputs and explanations.
fn run(
    kind: &Kind,
    speed: SpeedConfig,
    retry: RetryConfig,
    ops: &[Op],
) -> Result<Vec<String>, TestCaseError> {
    let mut p = kind.build(speed, retry);
    let mut sh = Shadow {
        group_first: kind.group_first(),
        by_id: kind.by_id(),
        max_attempts: retry.max_attempts,
        ..Shadow::default()
    };
    let mut log = Vec::new();
    for op in ops {
        match *op {
            Op::Submit {
                demand,
                group,
                priority,
                prefer,
                avoid,
                avoid_soft,
                class,
                work,
                dev,
            } => {
                let id = sh.next_id;
                sh.next_id += 1;
                let mut spec =
                    JobSpec::new(id, Resources::mem(demand).with_dev(dev.unwrap_or(0)), group);
                spec.priority = priority;
                spec.prefer = prefer.into_iter().collect();
                spec.avoid = avoid.into_iter().collect();
                spec.avoid_soft = avoid_soft;
                spec.class = class.map(|c| format!("c{c}"));
                spec.work = work.map(f64::from);
                sh.submit(spec, &mut *p);
            }
            Op::Complete(i, k) => {
                if let Some((job, _, attempt)) = sh.pick_live(i, k) {
                    sh.stop_all(job, Some(attempt));
                    p.handle(Input::Done { job, attempt }, sh.now);
                }
            }
            Op::Fail(i, k, kind) => {
                if let Some((job, k, attempt)) = sh.pick_live(i, k) {
                    let why = format!("failed at {}", sh.now);
                    sh.fail(job, k, kind, why.clone());
                    p.handle(
                        Input::Failed {
                            job,
                            attempt,
                            kind,
                            why,
                        },
                        sh.now,
                    );
                }
            }
            Op::Stale { job, attempt, done } => {
                if sh.next_id > 0 {
                    let job = job as JobId % sh.next_id;
                    let live: Vec<Attempt> = sh
                        .running
                        .get(&job)
                        .map(|r| r.live.iter().map(|l| l.attempt).collect())
                        .unwrap_or_default();
                    // Attempt 0 and the next attempt number were never started.
                    let last = sh.started.get(&job).copied().unwrap_or(0);
                    let stale: Vec<Attempt> =
                        (0..=last + 1).filter(|a| !live.contains(a)).collect();
                    let attempt = stale[attempt % stale.len()];
                    p.handle(
                        if done {
                            Input::Done { job, attempt }
                        } else {
                            Input::Failed {
                                job,
                                attempt,
                                kind: FailKind::Other,
                                why: "stale".into(),
                            }
                        },
                        sh.now,
                    );
                }
            }
            Op::Cancel(i) => {
                let live: Vec<JobId> = sh
                    .running
                    .keys()
                    .chain(sh.waiting.keys())
                    .copied()
                    .collect();
                if !live.is_empty() {
                    let j = live[i % live.len()];
                    if sh.running.contains_key(&j) {
                        sh.stop_all(j, None);
                    }
                    sh.waiting.remove(&j);
                    p.handle(Input::Cancel(j), sh.now);
                }
            }
            Op::Worker {
                id,
                slots,
                budget,
                used,
                baseline,
                class,
                dev_cap,
                per_task,
            } => {
                let s = WorkerState {
                    reported_used: Resources::mem(used),
                    reported_baseline: Resources::mem(baseline),
                    speed: class_speed(class),
                    per_task: Resources::ZERO.with_dev(per_task),
                    ..WorkerState::new(
                        id,
                        format!("c{class}"),
                        slots,
                        Resources::mem(budget).with_dev(dev_cap),
                    )
                };
                sh.workers.insert(id, s.clone());
                p.handle(Input::Worker(s), sh.now);
            }
            Op::Gone(w) => {
                if sh.workers.remove(&w).is_some() {
                    // Each live attempt there fails; the caller resubmits nothing.
                    let lost: Vec<(JobId, usize)> = sh
                        .running
                        .iter()
                        .filter_map(|(&j, r)| Some((j, r.live.iter().position(|l| l.worker == w)?)))
                        .collect();
                    for (j, k) in lost {
                        sh.fail(j, k, FailKind::LinkDied, format!("worker {w} left"));
                    }
                }
                p.handle(Input::WorkerGone(w), sh.now);
            }
            Op::Tick(dt) => sh.now += dt as f64,
        }

        let before = p.stats();
        let out = p.poll(sh.now);
        let after = p.stats();
        if std::env::var_os("SCHED_TRACE").is_some() {
            eprintln!("op {op:?} -> {out:?} deferred {:?}", after.deferred);
            for j in sh.waiting.keys() {
                eprintln!("  {}", p.explain(*j).unwrap_or_default());
            }
        }
        let mut starts = Vec::new();
        let mut other = Vec::new();
        for o in &out {
            match *o {
                Output::Start {
                    job,
                    attempt,
                    worker,
                } => starts.push((job, attempt, worker)),
                _ => other.push(o.clone()),
            }
        }
        prop_assert_eq!(&other, &std::mem::take(&mut sh.expect), "after {:?}", op);
        let holders: BTreeSet<JobId> = after.last_dispatch_holders.iter().copied().collect();
        let deferred: BTreeSet<JobId> = after.deferred.iter().map(|d| d.0).collect();
        let deferred_any: BTreeSet<JobId> = after.deferred_any.iter().copied().collect();
        prop_assert!(deferred.is_subset(&deferred_any));
        // Deferral is only for jobs with work, within their waiting window, onto a full, faster
        // worker; and its expiry is a wakeup.
        for &(j, w, at) in &after.deferred {
            let job = &sh.waiting[&j];
            let SpeedPolicy::EarliestFinish(Some(d)) = speed.policy else {
                prop_assert!(false, "deferral without a Defer config");
                unreachable!()
            };
            prop_assert!(job.spec.work.is_some() && sh.now - job.since < d.max_wait);
            prop_assert!(
                !kind.age().is_some_and(|a| sh.now - job.since >= a),
                "aged job deferred"
            );
            prop_assert!(at >= sh.now);
            prop_assert!(sh.workers.contains_key(&w));
            prop_assert!(
                p.next_wakeup()
                    .is_some_and(|t| t > sh.now && t <= job.since + d.max_wait)
            );
        }
        if let Some(t) = p.next_wakeup() {
            prop_assert!(t > sh.now, "wakeup {} not in the future", t);
        }
        for &(j, attempt, w) in &starts {
            prop_assert!(sh.workers.contains_key(&w), "started on unknown worker {w}");
            if let Some(job) = sh.waiting.get(&j).cloned() {
                prop_assert_eq!(attempt, job.attempts + 1, "job {} attempt", j);
                prop_assert!(
                    sh.eligible(&job.spec, w),
                    "{kind:?}: job {j} placed on excluded worker {w}"
                );
                // No over-commit (slots included).
                prop_assert!(
                    sh.admits(job.spec.demand, w),
                    "{kind:?}: job {j} over-commits worker {w}"
                );
                // Priority: every more urgent waiting job is refused here (unless this is a holder
                // taking its own reserved worker, which nobody else could take).
                if kind.priority() && !holders.contains(&j) {
                    let mine = sh.urgency(&job, kind.age());
                    for a in sh.waiting.values() {
                        if a.spec.id != j && sh.urgency(a, kind.age()) < mine {
                            let refused = deferred_any.contains(&a.spec.id)
                                || !sh.eligible(&a.spec, w)
                                || !sh.admits(a.spec.demand, w);
                            prop_assert!(
                                refused,
                                "{kind:?}: job {j} placed on {w} while more urgent job {} is \
                                 admitted there",
                                a.spec.id
                            );
                        }
                    }
                }
                sh.waiting.remove(&j);
                sh.running.insert(
                    j,
                    SRun {
                        job: SJob {
                            attempts: attempt,
                            ..job
                        },
                        live: vec![Live {
                            attempt,
                            worker: w,
                            started: sh.now,
                        }],
                    },
                );
            } else if let Some(r) = sh.running.get(&j) {
                // A second live attempt: only by speculation, onto a strictly faster worker that
                // admits it, whose free slot no waiting job wanted, at most `max_per_job` times,
                // when it gains enough.
                let Some(cfg) = speed.speculate else {
                    prop_assert!(false, "job {j} has two live attempts without speculation");
                    unreachable!()
                };
                prop_assert_eq!(attempt, r.job.attempts + 1, "job {} attempt", j);
                prop_assert!(
                    r.job.speculated < cfg.max_per_job,
                    "job {j} over-speculated"
                );
                let spec = r.job.spec.clone();
                prop_assert!(
                    r.live
                        .iter()
                        .all(|l| sh.workers[&l.worker].speed < sh.workers[&w].speed),
                    "job {j} speculated onto {w}, not faster than {:?}",
                    r.live
                );
                prop_assert!(
                    sh.eligible(&spec, w) && sh.admits(spec.demand, w),
                    "job {j} speculated onto {w}, which does not take it"
                );
                prop_assert!(
                    !after.reservations.iter().any(|r| r.worker == w),
                    "speculated onto a reserved worker"
                );
                let wanted = sh.waiting.values().find(|a| {
                    !deferred.contains(&a.spec.id)
                        && sh.eligible(&a.spec, w)
                        && sh.admits(a.spec.demand, w)
                });
                prop_assert!(
                    wanted.is_none(),
                    "speculated onto {w} while job {:?} waits for it",
                    wanted.map(|a| a.spec.id)
                );
                let end = sh.expected_end(r);
                let run = sh.eta(&spec, w);
                prop_assert!(
                    end.is_some() && run.is_some(),
                    "speculated without a run time"
                );
                let (end, run) = (end.unwrap(), run.unwrap());
                let end_here = sh.now + run + cfg.restart_overhead;
                prop_assert!(
                    end > end_here && end - end_here >= cfg.min_gain * run,
                    "job {j} speculated onto {w} for too little: {end} vs {end_here}"
                );
                let r = sh.running.get_mut(&j).unwrap();
                r.job.attempts = attempt;
                r.job.speculated += 1;
                r.live.push(Live {
                    attempt,
                    worker: w,
                    started: sh.now,
                });
            } else {
                prop_assert!(
                    false,
                    "started job {j}, which is neither waiting nor running"
                );
            }
            sh.started.insert(j, attempt);
        }
        let held: BTreeMap<JobId, WorkerId> = after
            .reservations
            .iter()
            .map(|r| (r.job, r.worker))
            .collect();
        // Escape hatch: an empty worker with a free slot leaves no job waiting that may run there,
        // except one waiting for a faster worker by choice.
        for (&w, s) in &sh.workers {
            if s.slots > 0 && sh.load(w).0 == 0 {
                let stuck = sh
                    .waiting
                    .values()
                    .find(|j| sh.eligible(&j.spec, w) && !deferred.contains(&j.spec.id));
                prop_assert!(
                    stuck.is_none(),
                    "{kind:?}: worker {w} empty while job {:?} waits",
                    stuck.map(|j| j.spec.id)
                );
            }
        }
        // Bookkeeping agrees with the model: job counts, live attempts per job, and per-worker
        // slots and placed demand as the sum over live attempts.
        prop_assert_eq!(after.waiting, sh.waiting.len());
        prop_assert_eq!(after.running, sh.running.len());
        prop_assert_eq!(after.workers.len(), sh.workers.len());
        for l in &after.workers {
            let (n, m) = sh.load(l.id);
            prop_assert_eq!((l.running, l.placed), (n, m), "worker {} load", l.id);
        }
        for (&j, r) in &sh.running {
            prop_assert!(
                speed.speculate.is_some() || r.live.len() == 1,
                "job {j} has live attempts {:?}",
                r.live
            );
            let runs: Vec<String> = r
                .live
                .iter()
                .map(|l| format!("attempt {} on worker {}", l.attempt, l.worker))
                .collect();
            prop_assert_eq!(
                p.explain(j),
                Some(format!("job {j} is running: {}", runs.join(", ")))
            );
        }
        for &j in sh.waiting.keys() {
            let e = p.explain(j).unwrap_or_default();
            prop_assert!(
                e.starts_with(&format!("job {j} (")),
                "waiting job {j} explained as {e}"
            );
        }
        let (max_res, per_class) = kind.max_reservations();
        let mut per: BTreeMap<String, usize> = BTreeMap::new();
        for (&job, &worker) in &held {
            prop_assert!(
                sh.waiting.contains_key(&job),
                "holder {} is not waiting",
                job
            );
            let l = after.workers.iter().find(|l| l.id == worker);
            prop_assert!(l.is_some_and(|l| l.reserved_for == Some(job)));
            *per.entry(if per_class {
                sh.workers[&worker].class.clone()
            } else {
                String::new()
            })
            .or_default() += 1;
        }
        prop_assert!(
            per.values().all(|&n| n <= max_res),
            "too many reservations: {per:?}"
        );
        prop_assert!(before.placements_total + starts.len() as u64 == after.placements_total);
        log.push(format!(
            "{out:?} {:?} {:?}",
            after.deferred,
            p.next_wakeup()
        ));
        for id in 0..sh.next_id {
            log.push(p.explain(id).unwrap_or_default());
        }
    }
    // Liveness of bookkeeping: completing and cancelling everything releases everything.
    for (j, r) in std::mem::take(&mut sh.running) {
        p.handle(
            Input::Done {
                job: j,
                attempt: r.live[0].attempt,
            },
            sh.now,
        );
    }
    for j in sh.waiting.keys() {
        p.handle(Input::Cancel(*j), sh.now);
    }
    let end = p.stats();
    prop_assert_eq!(end.waiting + end.running, 0);
    prop_assert!(end.reservations.is_empty());
    prop_assert!(
        end.workers
            .iter()
            .all(|l| l.running == 0 && l.placed == Resources::ZERO && l.reserved_for.is_none())
    );
    Ok(log)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(768))]

    /// Every invariant holds on random streams, and replays are identical.
    #[test]
    fn invariants_hold(
        kind in kind(),
        speed in speed(),
        retry in retry(),
        ops in prop::collection::vec(op(), 1..160),
    ) {
        let first = run(&kind, speed, retry, &ops)?;
        let second = run(&kind, speed, retry, &ops)?;
        prop_assert_eq!(first, second, "not deterministic");
    }
}
