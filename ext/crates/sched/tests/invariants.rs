//! Property tests of the policy invariants over random event streams.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use sched::{
    Config, DIMS, Defer, Fit, GroupOrder, JobId, JobSpec, Order, Policy, Reservations, Resources,
    Scheduler, SpeedConfig, SpeedPolicy, Spoliation, WorkerId, WorkerState,
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
    fn build(&self, speed: SpeedConfig) -> Box<dyn Policy> {
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
            ..Config::default()
        };
        Box::new(Scheduler::new(match *self {
            Kind::Fifo => Config {
                speed,
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
    Complete(usize),
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

/// A random event.
fn op() -> impl Strategy<Value = Op> {
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
        4 => any::<prop::sample::Index>().prop_map(|i| Op::Complete(i.index(1 << 16))),
        1 => any::<prop::sample::Index>().prop_map(|i| Op::Cancel(i.index(1 << 16))),
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
    let spoliation = prop::option::weighted(
        0.4,
        (prop_oneof![Just(0.0), Just(0.25)], 1u32..3).prop_map(|(min_gain, max_per_job)| {
            Spoliation {
                min_gain,
                restart_overhead: 0.0,
                max_per_job,
            }
        }),
    );
    (policy, spoliation).prop_map(|(policy, spoliation)| SpeedConfig {
        policy,
        learn: None,
        spoliation,
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

#[derive(Clone, Debug)]
struct SJob {
    spec: JobSpec,
    since: f64,
    seq: u64,
}

#[derive(Default)]
struct Shadow {
    now: f64,
    workers: BTreeMap<WorkerId, WorkerState>,
    running: BTreeMap<JobId, (WorkerId, Resources)>,
    /// Specs of running jobs, and how often each was preempted.
    specs: BTreeMap<JobId, (JobSpec, u32)>,
    waiting: BTreeMap<JobId, SJob>,
    groups: BTreeMap<u64, u64>,
    seq: u64,
    next_id: JobId,
    group_first: bool,
    by_id: bool,
}

impl Shadow {
    /// Running count and placed demand on a worker.
    fn load(&self, w: WorkerId) -> (usize, Resources) {
        self.running
            .values()
            .filter(|r| r.0 == w)
            .fold((0, Resources::ZERO), |(n, m), r| (n + 1, m + r.1))
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

    /// Submit to both the shadow and the policy.
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
            },
        );
        p.submit(spec, self.now);
    }
}

/// Runs the stream, checking every invariant; returns the placements and explanations made.
///
/// Every policy is driven by the same random stream while a shadow model, written independently
/// of the engine, keeps its own bookkeeping and re-checks each placement as it is made:
///
/// - **no over-commit**: the production admission rule held at the moment of each placement, in
///   every dimension (no enforced capacity is exceeded, escape hatch aside);
/// - **hard constraints**: no job runs on a worker its class or avoid list excludes (a soft avoid
///   list only while some live worker of the class is off it);
/// - **escape hatch**: after a dispatch, no worker with a free slot is empty while a job that may
///   run there waits;
/// - **priority** (priority order): when B is placed on w, every more urgent waiting job was
///   refused by w at that moment (by the admission rule, or because w was B's reservation, or
///   because it chose to wait for a faster worker);
/// - **bookkeeping**: running counts, placed demand and reservations agree with the shadow after
///   every event, and everything is released at the end;
/// - **determinism**: replaying the stream gives identical placements and explanations.
fn run(kind: &Kind, speed: SpeedConfig, ops: &[Op]) -> Result<Vec<String>, TestCaseError> {
    let mut p = kind.build(speed);
    let mut sh = Shadow {
        group_first: kind.group_first(),
        by_id: kind.by_id(),
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
            Op::Complete(i) => {
                if !sh.running.is_empty() {
                    let j = *sh.running.keys().nth(i % sh.running.len()).unwrap();
                    sh.running.remove(&j);
                    p.completed(j, sh.now);
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
                    sh.running.remove(&j);
                    sh.waiting.remove(&j);
                    p.cancel(j);
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
                p.worker_update(s, sh.now);
            }
            Op::Gone(w) => {
                if sh.workers.remove(&w).is_some() {
                    p.worker_gone(w, sh.now);
                    // The caller resubmits the jobs that were running there, avoiding that worker
                    // (it is gone, but a worker may rejoin under the same id).
                    let lost: Vec<JobId> = sh
                        .running
                        .iter()
                        .filter(|r| r.1.0 == w)
                        .map(|r| *r.0)
                        .collect();
                    for j in lost {
                        let (_, demand) = sh.running.remove(&j).unwrap();
                        let mut spec = JobSpec::new(j, demand, 0);
                        spec.avoid = vec![w];
                        sh.submit(spec, &mut *p);
                    }
                }
            }
            Op::Tick(dt) => sh.now += dt as f64,
        }

        let before = p.stats();
        let full = p.dispatch_full(sh.now);
        let out = full.start;
        let after = p.stats();
        if std::env::var_os("SCHED_TRACE").is_some() {
            eprintln!("op {op:?} -> {out:?} deferred {:?}", after.deferred);
            for j in sh.waiting.keys() {
                eprintln!("  {}", p.explain(*j).unwrap_or_default());
            }
        }
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
        for &(j, w) in &out {
            let job = sh.waiting.get(&j).cloned();
            prop_assert!(job.is_some(), "placed job {j} is not waiting");
            let job = job.unwrap();
            prop_assert!(sh.workers.contains_key(&w), "placed on unknown worker {w}");
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
                            "{kind:?}: job {j} placed on {w} while more urgent job {} is admitted \
                             there",
                            a.spec.id
                        );
                    }
                }
            }
            sh.waiting.remove(&j);
            sh.running.insert(j, (w, job.spec.demand));
            sh.specs.insert(j, (job.spec.clone(), 0));
        }
        let held: BTreeMap<JobId, WorkerId> = after
            .reservations
            .iter()
            .map(|r| (r.job, r.worker))
            .collect();
        // Spoliation: each preemption moves a running job, with work, from a strictly slower
        // worker to one that admits it, whose free slot no waiting job wanted, at most
        // `max_per_job` times.
        let mut vacated = BTreeSet::new();
        for pr in &full.preempt {
            let Some(cfg) = speed.spoliation else {
                prop_assert!(false, "preemption without spoliation");
                unreachable!()
            };
            prop_assert_eq!(
                sh.running.get(&pr.job).map(|r| r.0),
                Some(pr.from),
                "{:?}",
                pr
            );
            let (spec, n) = sh.specs[&pr.job].clone();
            prop_assert!(spec.work.is_some());
            prop_assert!(
                sh.workers[&pr.from].speed < sh.workers[&pr.to].speed,
                "{:?}",
                pr
            );
            prop_assert!(
                sh.eligible(&spec, pr.to) && sh.admits(spec.demand, pr.to),
                "{:?}",
                pr
            );
            prop_assert!(n < cfg.max_per_job, "ping-pong: {:?}", pr);
            prop_assert!(
                !held.contains_key(&pr.job)
                    && !after.reservations.iter().any(|r| r.worker == pr.to),
                "preempted onto a reserved worker"
            );
            let wanted = sh.waiting.values().find(|j| {
                !deferred.contains(&j.spec.id)
                    && sh.eligible(&j.spec, pr.to)
                    && sh.admits(j.spec.demand, pr.to)
            });
            prop_assert!(
                wanted.is_none(),
                "preempted onto {} while job {:?} waits for it",
                pr.to,
                wanted.map(|j| j.spec.id)
            );
            sh.running.insert(pr.job, (pr.to, spec.demand));
            sh.specs.insert(pr.job, (spec, n + 1));
            vacated.insert(pr.from);
        }
        // Escape hatch: an empty worker with a free slot leaves no job waiting that may run there,
        // except one waiting for a faster worker by choice.
        for (&w, s) in &sh.workers {
            // A worker that just gave up a preempted job is refilled at the next dispatch.
            if s.slots > 0 && sh.load(w).0 == 0 && !vacated.contains(&w) {
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
        // Bookkeeping agrees with the shadow.
        prop_assert_eq!(after.waiting, sh.waiting.len());
        prop_assert_eq!(after.running, sh.running.len());
        prop_assert_eq!(after.workers.len(), sh.workers.len());
        for l in &after.workers {
            let (n, m) = sh.load(l.id);
            prop_assert_eq!((l.running, l.placed), (n, m), "worker {} load", l.id);
        }
        let (max_res, per_class) = kind.max_reservations();
        let mut per: BTreeMap<String, usize> = BTreeMap::new();
        for r in &after.reservations {
            prop_assert!(
                sh.waiting.contains_key(&r.job),
                "holder {} is not waiting",
                r.job
            );
            let l = after.workers.iter().find(|l| l.id == r.worker);
            prop_assert!(l.is_some_and(|l| l.reserved_for == Some(r.job)));
            *per.entry(if per_class {
                sh.workers[&r.worker].class.clone()
            } else {
                String::new()
            })
            .or_default() += 1;
        }
        prop_assert!(
            per.values().all(|&n| n <= max_res),
            "too many reservations: {per:?}"
        );
        prop_assert!(before.placements_total + out.len() as u64 == after.placements_total);
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
    for j in sh.running.keys().copied().collect::<Vec<_>>() {
        p.completed(j, sh.now);
    }
    for j in sh.waiting.keys().copied().collect::<Vec<_>>() {
        p.cancel(j);
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
    fn invariants_hold(kind in kind(), speed in speed(), ops in prop::collection::vec(op(), 1..160)) {
        let first = run(&kind, speed, &ops)?;
        let second = run(&kind, speed, &ops)?;
        prop_assert_eq!(first, second, "not deterministic");
    }
}
