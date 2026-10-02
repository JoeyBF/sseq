//! Property tests of the policy invariants over random event streams.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use sched::{
    BackfillConfig, BestFit, BestFitConfig, Greedy, GreedyConfig, JobId, JobSpec, LaneSet, Lanes,
    LanesConfig, Policy, PriorityBackfill, Resources, WorkerId, WorkerState,
};

const LANE: WorkerId = 0;
const LANE_BIG: u64 = 30;
const LANE_RESERVE: u64 = 20;

#[derive(Clone, Debug)]
enum Kind {
    Greedy,
    Backfill {
        max_res: usize,
        per_class: bool,
        age: Option<f64>,
    },
    BestFit {
        penalty: u64,
        age: Option<f64>,
    },
    Lanes,
}

impl Kind {
    /// The policy under test.
    fn build(&self) -> Box<dyn Policy> {
        let bf = |max_res, per_class, age| BackfillConfig {
            reserve_after: 30.0,
            max_reservations: max_res,
            per_class_reservations: per_class,
            age_limit: age,
            ..BackfillConfig::default()
        };
        match *self {
            Kind::Greedy => Box::new(Greedy::new(GreedyConfig {})),
            Kind::Backfill {
                max_res,
                per_class,
                age,
            } => Box::new(PriorityBackfill::new(bf(max_res, per_class, age))),
            Kind::BestFit { penalty, age } => Box::new(BestFit::new(BestFitConfig {
                backfill: bf(1, false, age),
                prefer_penalty: penalty,
            })),
            Kind::Lanes => Box::new(Lanes::new(LanesConfig {
                backfill: bf(1, false, None),
                lanes: LaneSet::Workers(vec![LANE]),
                big_threshold: Resources::mem(LANE_BIG),
                lane_reserve: Resources::mem(LANE_RESERVE),
            })),
        }
    }

    /// Whether the priority invariant applies.
    fn priority(&self) -> bool {
        !matches!(self, Kind::Greedy)
    }

    /// The configured age limit.
    fn age(&self) -> Option<f64> {
        match *self {
            Kind::Backfill { age, .. } | Kind::BestFit { age, .. } => age,
            _ => None,
        }
    }

    /// The reservation limit and whether it is per class.
    fn max_reservations(&self) -> (usize, bool) {
        match *self {
            Kind::Greedy => (0, false),
            Kind::Backfill {
                max_res, per_class, ..
            } => (max_res, per_class),
            _ => (1, false),
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
        class: Option<u8>,
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
            prop::option::weighted(0.15, 0u8..2),
        )
            .prop_map(|(demand, group, priority, prefer, avoid, class)| Op::Submit {
                demand,
                group,
                priority,
                prefer,
                avoid,
                class,
            }),
        4 => any::<prop::sample::Index>().prop_map(|i| Op::Complete(i.index(1 << 16))),
        1 => any::<prop::sample::Index>().prop_map(|i| Op::Cancel(i.index(1 << 16))),
        2 => (0u64..4, 0usize..5, 20u64..150, 0u64..150, 0u64..60, 0u8..2).prop_map(
            |(id, slots, budget, used, baseline, class)| Op::Worker { id, slots, budget, used, baseline, class }
        ),
        1 => (0u64..4).prop_map(Op::Gone),
        3 => (0u32..90).prop_map(Op::Tick),
    ]
}

/// A random policy configuration.
fn kind() -> impl Strategy<Value = Kind> {
    let age = prop::option::of(prop_oneof![Just(0.0), Just(45.0), Just(200.0)]);
    prop_oneof![
        Just(Kind::Greedy),
        (0usize..3, any::<bool>(), age.clone()).prop_map(|(max_res, per_class, age)| {
            Kind::Backfill {
                max_res,
                per_class,
                age,
            }
        }),
        (prop_oneof![Just(0u64), Just(25)], age)
            .prop_map(|(penalty, age)| Kind::BestFit { penalty, age }),
        Just(Kind::Lanes),
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
    running: BTreeMap<JobId, (WorkerId, u64)>,
    waiting: BTreeMap<JobId, SJob>,
    groups: BTreeMap<u64, u64>,
    seq: u64,
    next_id: JobId,
}

impl Shadow {
    /// Running count and placed demand on a worker.
    fn load(&self, w: WorkerId) -> (usize, u64) {
        self.running
            .values()
            .filter(|r| r.0 == w)
            .fold((0, 0), |(n, m), r| (n + 1, m + r.1))
    }

    /// The production rule, written out again.
    fn admits(&self, demand: u64, w: WorkerId) -> bool {
        let s = &self.workers[&w];
        let (running, placed) = self.load(w);
        if running >= s.slots {
            return false;
        }
        running == 0
            || s.reported_used.mem.max(s.reported_baseline.mem + placed) + demand <= s.budget.mem
    }

    /// The hard constraints (class, avoid list), written out again.
    fn eligible(&self, spec: &JobSpec, w: WorkerId) -> bool {
        spec.class
            .as_ref()
            .is_none_or(|c| *c == self.workers[&w].class)
            && !spec.avoid.contains(&w)
    }

    /// The lane rule, written out again.
    fn lane_refuses(&self, demand: u64, w: WorkerId) -> bool {
        let s = &self.workers[&w];
        let (running, placed) = self.load(w);
        let used = s.reported_used.mem.max(s.reported_baseline.mem + placed);
        w == LANE
            && running > 0
            && demand <= LANE_BIG
            && (s.budget.mem as i64 - used as i64 - demand as i64) < LANE_RESERVE as i64
    }

    /// Scan order: aged jobs by age, then priority, group arrival, FIFO.
    fn urgency(&self, j: &SJob, age: Option<f64>) -> (bool, i64, u64, u64) {
        if age.is_some_and(|a| self.now - j.since >= a) {
            (false, 0, 0, j.seq)
        } else {
            (
                true,
                j.spec.priority.unwrap_or(0),
                self.groups[&j.spec.group],
                j.seq,
            )
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
/// - **no over-commit**: the production admission rule held at the moment of each placement;
/// - **hard constraints**: no job runs on a worker its class or avoid list excludes;
/// - **escape hatch**: after a dispatch, no worker with a free slot is empty while a job that may
///   run there waits;
/// - **priority** (priority policies): when B is placed on w, every more urgent waiting job was
///   refused by w at that moment (by the admission rule, or by a lane, or because w was B's
///   reservation);
/// - **bookkeeping**: running counts, placed demand and reservations agree with the shadow after
///   every event, and everything is released at the end;
/// - **determinism**: replaying the stream gives identical placements and explanations.
fn run(kind: &Kind, ops: &[Op]) -> Result<Vec<String>, TestCaseError> {
    let mut p = kind.build();
    let mut sh = Shadow::default();
    let mut log = Vec::new();
    for op in ops {
        match *op {
            Op::Submit {
                demand,
                group,
                priority,
                prefer,
                avoid,
                class,
            } => {
                let id = sh.next_id;
                sh.next_id += 1;
                let mut spec = JobSpec::new(id, Resources::mem(demand), group);
                spec.priority = priority;
                spec.prefer = prefer.into_iter().collect();
                spec.avoid = avoid.into_iter().collect();
                spec.class = class.map(|c| format!("c{c}"));
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
            } => {
                let s = WorkerState {
                    reported_used: Resources::mem(used),
                    reported_baseline: Resources::mem(baseline),
                    ..WorkerState::new(id, format!("c{class}"), slots, Resources::mem(budget))
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
                        let mut spec = JobSpec::new(j, Resources::mem(demand), 0);
                        spec.avoid = vec![w];
                        sh.submit(spec, &mut *p);
                    }
                }
            }
            Op::Tick(dt) => sh.now += dt as f64,
        }

        let before = p.stats();
        let out = p.dispatch(sh.now);
        let after = p.stats();
        let holders: BTreeSet<JobId> = after.last_dispatch_holders.iter().copied().collect();
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
                sh.admits(job.spec.demand.mem, w),
                "{kind:?}: job {j} over-commits worker {w}"
            );
            if matches!(kind, Kind::Lanes) {
                prop_assert!(
                    !sh.lane_refuses(job.spec.demand.mem, w),
                    "lane reserve broken"
                );
            }
            // Priority: every more urgent waiting job is refused here (unless this is a holder
            // taking its own reserved worker, which nobody else could take).
            if kind.priority() && !holders.contains(&j) {
                let mine = sh.urgency(&job, kind.age());
                for a in sh.waiting.values() {
                    if a.spec.id != j && sh.urgency(a, kind.age()) < mine {
                        let refused = !sh.eligible(&a.spec, w)
                            || !sh.admits(a.spec.demand.mem, w)
                            || (matches!(kind, Kind::Lanes)
                                && sh.lane_refuses(a.spec.demand.mem, w));
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
            sh.running.insert(j, (w, job.spec.demand.mem));
        }
        // Escape hatch: an empty worker with a free slot leaves no job waiting that may run there.
        for (&w, s) in &sh.workers {
            if s.slots > 0 && sh.load(w).0 == 0 {
                let stuck = sh.waiting.values().find(|j| sh.eligible(&j.spec, w));
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
            prop_assert_eq!((l.running, l.placed.mem), (n, m), "worker {} load", l.id);
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
        log.push(format!("{out:?}"));
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
            .all(|l| l.running == 0 && l.placed.mem == 0 && l.reserved_for.is_none())
    );
    Ok(log)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(768))]

    /// Every invariant holds on random streams, and replays are identical.
    #[test]
    fn invariants_hold(kind in kind(), ops in prop::collection::vec(op(), 1..160)) {
        let first = run(&kind, &ops)?;
        let second = run(&kind, &ops)?;
        prop_assert_eq!(first, second, "not deterministic");
    }
}
