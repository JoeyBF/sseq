//! The DAG layer as a coordinator drives it: REQUESTS.md's R8 to R12.

use std::{sync::Arc, time::Duration};

use proptest::prelude::*;
use whelm::{
    Attempt, Config, DagConfig, DagJob, DagScheduler, DagTemplate, Input, JobId, JobSpec, MEM,
    NodeSource, Output, Policy, Resources, Scheduler, TemplateSpec, Time, Unit, WorkerState,
};

/// A DAG layer over the default backfill policy with one worker of `slots` slots.
fn whelm(slots: usize, config: DagConfig) -> DagScheduler<Scheduler> {
    let mut d = DagScheduler::new(config, Scheduler::new(Config::default()));
    join(&mut d, slots, Time::ZERO);
    d
}

/// Worker 0, with `slots` slots, joins.
fn join(d: &mut DagScheduler<Scheduler>, slots: usize, now: Time) {
    let w = WorkerState {
        class: "x".into(),
        slots,
        budget: Resources::mem(1 << 40),
        ..Default::default()
    };
    d.handle(Input::Worker(w), now);
}

/// Report attempt `attempt` of `job` done (0 for a local job).
fn done(d: &mut DagScheduler<Scheduler>, job: JobId, attempt: Attempt, now: Time) {
    d.handle(Input::Done { job, attempt }, now);
}

/// The jobs a poll started (on worker 0, first attempts), in output order.
fn starts(out: &[Output]) -> Vec<JobId> {
    out.iter()
        .filter_map(|o| match *o {
            Output::Start {
                job,
                attempt: 1,
                worker: 0,
            } => Some(job),
            _ => None,
        })
        .collect()
}

/// The jobs a poll reported as passed.
fn passed(out: &[Output]) -> Vec<JobId> {
    out.iter()
        .filter_map(|o| match *o {
            Output::Passed { job } => Some(job),
            _ => None,
        })
        .collect()
}

/// A walk: a unit `done` of `template` at `base`, after `entry`.
fn walk(template: &Arc<DagTemplate>, base: JobId, entry: JobId, done: JobId) -> Unit {
    Unit {
        id: done,
        base,
        template: template.clone(),
        deps: vec![entry],
        spec: JobSpec {
            demand: Resources::mem(1),
            group: 7,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// A plain job of `demand` bytes in group `group`, after `deps`.
fn plain(id: JobId, demand: u64, group: u64, deps: Vec<JobId>) -> DagJob {
    DagJob {
        spec: JobSpec {
            id,
            demand: Resources::mem(demand),
            group,
            ..Default::default()
        },
        deps,
        ..Default::default()
    }
}

/// The template of `n` worker jobs with the given edges.
fn template_of(n: usize, edges: impl IntoIterator<Item = (u32, u32)>) -> Arc<DagTemplate> {
    let spec = TemplateSpec {
        edges: edges.into_iter().collect(),
        ..TemplateSpec::jobs(n)
    };
    Arc::new(spec.build().unwrap())
}

/// A local job in group 7 with no dependency.
fn local_entry(id: JobId) -> DagJob {
    DagJob {
        local: true,
        ..plain(id, 0, 7, vec![])
    }
}

/// A local entry job `1` (the zero step), already completed.
fn with_entry(d: &mut DagScheduler<Scheduler>) {
    d.declare(vec![local_entry(1)], Time::ZERO).unwrap();
    assert_eq!(d.poll(Time::ZERO), vec![Output::RunLocal { job: 1 }]);
    done(d, 1, 0, Time::ZERO);
}

/// Per-leaf demands and names, computed on demand.
struct Squares;

impl NodeSource for Squares {
    /// One second of work.
    fn work(&self, _unit: JobId, _leaf: u32) -> Duration {
        Duration::from_secs(1)
    }

    /// Leaf `i` demands `i^2 + 3` bytes.
    fn spec(&self, _unit: JobId, leaf: u32, spec: &mut JobSpec) {
        spec.demand = Resources::mem(u64::from(leaf * leaf + 3));
    }

    /// `Sq(i)`.
    fn label(&self, _unit: JobId, leaf: u32) -> Option<String> {
        Some(format!("Sq({leaf})"))
    }
}

/// R8: each leaf of a sourced unit is submitted with its own demand, and explained under its
/// label.
#[test]
fn per_node_demand_and_label() {
    let mut d = whelm(16, DagConfig::default()).with_source(Arc::new(Squares));
    let t = template_of(3, []);
    d.declare(vec![local_entry(1)], Time::ZERO).unwrap();
    let unit = Unit {
        sourced: true,
        ..walk(&t, 100, 1, 99)
    };
    d.declare([unit], Time::ZERO).unwrap();
    assert!(
        d.explain(101).unwrap().starts_with("[Sq(1)]"),
        "{:?}",
        d.explain(101)
    );
    assert_eq!(d.poll(Time::ZERO), vec![Output::RunLocal { job: 1 }]);
    done(&mut d, 1, 0, Time::ZERO);
    assert_eq!(starts(&d.poll(Time::ZERO)).len(), 3);
    assert_eq!(d.stats().workers[0].placed[MEM], 3 + 4 + 7);
}

/// The completion order of a walk driven to the end: every round, poll, then complete every
/// started job in id order; also returns how often `done` (99) fired.
fn drive(d: &mut DagScheduler<Scheduler>) -> (Vec<JobId>, usize) {
    let mut order = Vec::new();
    let mut fired = 0;
    for _ in 0..1000 {
        let out = d.poll(Time::ZERO);
        fired += passed(&out).iter().filter(|&&j| j == 99).count();
        let mut placed = starts(&out);
        if placed.is_empty() {
            break;
        }
        placed.sort_unstable();
        for j in placed {
            order.push(j);
            done(d, j, 1, Time::ZERO);
        }
    }
    (order, fired)
}

/// A random DAG on `n` nodes (edges from lower to higher index).
fn template() -> impl Strategy<Value = (usize, Vec<(u32, u32)>)> {
    (1usize..14).prop_flat_map(|n| {
        (
            Just(n),
            prop::collection::vec((0..n as u32, 0..n as u32), 0..30).prop_map(|e| {
                e.into_iter()
                    .filter(|(a, b)| a < b)
                    .collect::<Vec<(u32, u32)>>()
            }),
        )
    })
}

proptest! {
    /// R9: entering with a completed set S behaves exactly like the template without S, every
    /// edge out of S satisfied: the same nodes run afterwards, round by round (every ready node
    /// runs each round), `done` fires once, and no node of S ever runs.
    #[test]
    fn opening_with_completed_equals_completing(
        (n, edges) in template(),
        mask in prop::collection::vec(any::<bool>(), 14),
    ) {
        let t = template_of(n, edges);
        let s: Vec<u32> = t
            .topological_order()
            .iter()
            .copied()
            .filter(|&i| mask[i as usize])
            .collect();
        let cfg = DagConfig { record_passthrough: true, ..DagConfig::default() };
        let mut a = whelm(1000, cfg.clone());
        with_entry(&mut a);
        let unit = Unit { completed: s.clone(), ..walk(&t, 100, 1, 99) };
        a.declare([unit], Time::ZERO).unwrap();
        // The reference: the nodes outside S, declared as plain jobs after the entry, and `done`
        // as a passthrough after all of them and the entry.
        let mut b = whelm(1000, cfg);
        with_entry(&mut b);
        let rest: Vec<u32> = (0..n as u32).filter(|i| !s.contains(i)).collect();
        let mut jobs: Vec<DagJob> = rest
            .iter()
            .map(|&i| {
                let mut deps: Vec<JobId> = t
                    .predecessors(i as usize)
                    .iter()
                    .filter(|p| !s.contains(p))
                    .map(|&p| 100 + JobId::from(p))
                    .collect();
                deps.push(1);
                plain(100 + JobId::from(i), 1, 7, deps)
            })
            .collect();
        let mut nodes: Vec<JobId> = rest.iter().map(|&i| 100 + JobId::from(i)).collect();
        nodes.push(1);
        jobs.push(DagJob {
            passthrough: true,
            work_estimate: Some(Duration::ZERO),
            ..plain(99, 0, 7, nodes)
        });
        b.declare(jobs, Time::ZERO).unwrap();
        let (order_a, done_a) = drive(&mut a);
        let (order_b, done_b) = drive(&mut b);
        prop_assert_eq!(&order_a, &order_b);
        prop_assert_eq!(done_a, 1);
        prop_assert_eq!(done_b, 1);
        for &i in &s {
            prop_assert!(!order_a.contains(&(100 + u64::from(i))), "node {} of S ran", i);
        }
        prop_assert_eq!(order_a.len() + s.len(), n);
    }
}

/// R10: closing early completes the walk once, withdraws waiting leaves, and returns the running
/// ones, whose later completions only free their resources.
#[test]
fn close_walk_early() {
    let cfg = DagConfig {
        record_passthrough: true,
        ..DagConfig::default()
    };
    let mut d = whelm(2, cfg);
    with_entry(&mut d);
    // Four independent nodes, a chain after them; two slots.
    let t = template_of(6, [(0, 4), (1, 4), (4, 5)]);
    d.declare([walk(&t, 100, 1, 99)], Time::ZERO).unwrap();
    // Something after the walk.
    d.declare(vec![plain(200, 1, 8, vec![99])], Time::ZERO)
        .unwrap();
    assert_eq!(starts(&d.poll(Time::ZERO)), vec![100, 101]);
    let mut running = d.close(99, Time::from_secs(1)).unwrap();
    running.sort_unstable();
    assert_eq!(running, vec![100, 101]);
    // The walk passed, and its dependent is ready, but both slots are still taken.
    assert_eq!(d.poll(Time::from_secs(1)), vec![Output::Passed { job: 99 }]);
    // The waiting nodes were withdrawn.
    let st = d.stats();
    assert_eq!((st.waiting, st.running), (1, 2));
    assert!(d.close(99, Time::from_secs(1)).is_err(), "closed twice");
    // Ignored completions free their slots and change nothing else.
    done(&mut d, 100, 1, Time::from_secs(2));
    done(&mut d, 101, 1, Time::from_secs(2));
    let out = d.poll(Time::from_secs(2));
    assert!(passed(&out).is_empty(), "no second done");
    assert_eq!(starts(&out), vec![200]);
    done(&mut d, 200, 1, Time::from_secs(3));
    assert!(d.poll(Time::from_secs(3)).is_empty());
    let st = d.stats();
    assert_eq!((st.waiting, st.running), (0, 0));
    assert_eq!(st.workers[0].placed, Resources::ZERO);
}

/// R11: local jobs are never submitted; they are announced as [`Output::RunLocal`], not
/// [`Output::Ready`], `release` refuses them, and they are reported done with attempt 0.
#[test]
fn local_jobs_stay_on_the_caller() {
    let mut d = whelm(
        4,
        DagConfig {
            auto_submit: false,
            ..DagConfig::default()
        },
    );
    d.declare(
        vec![
            DagJob {
                local: true,
                ..plain(1, 0, 0, vec![])
            },
            plain(2, 1, 0, vec![1]),
            DagJob {
                local: true,
                ..plain(3, 0, 0, vec![2])
            },
        ],
        Time::ZERO,
    )
    .unwrap();
    assert_eq!(d.poll(Time::ZERO), vec![Output::RunLocal { job: 1 }]);
    assert!(!d.release(1, Time::ZERO));
    assert!(d.poll(Time::ZERO).is_empty());
    done(&mut d, 1, 0, Time::from_secs(1));
    assert_eq!(d.poll(Time::from_secs(1)), vec![Output::Ready { job: 2 }]);
    assert!(d.release(2, Time::from_secs(1)));
    assert_eq!(starts(&d.poll(Time::from_secs(1))), vec![2]);
    done(&mut d, 2, 1, Time::from_secs(2));
    assert_eq!(
        d.poll(Time::from_secs(2)),
        vec![Output::RunLocal { job: 3 }]
    );
    assert_eq!(d.stats().placements_total, 1);
}

/// R12: snapshot and restore of an open frontier.
#[cfg(feature = "serde")]
mod restart {
    use std::collections::{BTreeMap, BTreeSet, HashSet};

    use super::*;

    /// A random workload: plain jobs (some local) with dependencies on earlier ones, and walks
    /// (units) entered after a plain job and depended on by a later one.
    #[derive(Clone, Debug)]
    struct Workload {
        plain: Vec<(Vec<usize>, bool)>,
        walks: Vec<(usize, usize, Vec<(u32, u32)>, usize)>,
    }

    /// A random [`Workload`].
    fn workload() -> impl Strategy<Value = Workload> {
        (
            prop::collection::vec(
                (
                    prop::collection::vec(any::<prop::sample::Index>(), 0..3),
                    any::<bool>(),
                ),
                2..12,
            ),
            prop::collection::vec(
                (
                    any::<prop::sample::Index>(),
                    any::<prop::sample::Index>(),
                    1usize..10,
                    prop::collection::vec((0u32..10, 0u32..10), 0..15),
                ),
                0..4,
            ),
        )
            .prop_map(|(ex, walks)| {
                let n = ex.len();
                let plain = ex
                    .into_iter()
                    .enumerate()
                    .map(|(i, (deps, local))| {
                        let deps = if i == 0 {
                            Vec::new()
                        } else {
                            deps.iter().map(|d| d.index(i)).collect()
                        };
                        (deps, local)
                    })
                    .collect();
                let walks = walks
                    .into_iter()
                    .map(|(entry, after, len, edges)| {
                        let entry = entry.index(n);
                        let after = after.index(n);
                        let edges = edges
                            .into_iter()
                            .filter(|&(a, b)| a < b && (b as usize) < len)
                            .collect();
                        (entry, len, edges, after)
                    })
                    .collect();
                Workload { plain, walks }
            })
    }

    /// Leaf 0 of the first walk; later walks follow at a fixed stride.
    const WALK_BASE: JobId = 1000;
    /// Explicit job `i`.
    fn ex_id(i: usize) -> JobId {
        i as JobId
    }
    /// Walk `w`'s `done` job.
    fn done_id(w: usize) -> JobId {
        500 + w as JobId
    }

    /// Declare the workload. A walk's `done` is depended on by a fresh plain job (`600 + w`),
    /// which a later plain job `after` also waits for only if `after > entry` (no cycles).
    fn declare(d: &mut DagScheduler<Scheduler>, w: &Workload) -> BTreeSet<JobId> {
        let mut all = BTreeSet::new();
        let mut jobs = Vec::new();
        for (i, (deps, local)) in w.plain.iter().enumerate() {
            let mut deps: Vec<JobId> = deps.iter().map(|&j| ex_id(j)).collect();
            for (k, walk) in w.walks.iter().enumerate() {
                if walk.3 == i && walk.3 > walk.0 {
                    deps.push(600 + k as JobId);
                }
            }
            jobs.push(DagJob {
                local: *local,
                ..plain(ex_id(i), 1, i as u64, deps)
            });
            all.insert(ex_id(i));
        }
        for k in 0..w.walks.len() {
            jobs.push(plain(600 + k as JobId, 1, 50, vec![done_id(k)]));
            all.insert(600 + k as JobId);
        }
        d.declare(jobs, Time::ZERO).unwrap();
        for (k, (entry, len, edges, _)) in w.walks.iter().enumerate() {
            let t = template_of(*len, edges.iter().copied());
            let base = WALK_BASE + 100 * k as JobId;
            d.declare([walk(&t, base, ex_id(*entry), done_id(k))], Time::ZERO)
                .unwrap();
            all.extend((0..*len).map(|i| base + i as JobId));
        }
        all
    }

    /// Run to the end, snapshotting and restoring (onto a fresh policy, all running work lost) at
    /// each step in `restarts`. Returns, per job, how often it completed, and checks dependencies.
    fn run(
        w: &Workload,
        restarts: &BTreeSet<usize>,
    ) -> Result<BTreeMap<JobId, usize>, TestCaseError> {
        let mut d = whelm(3, DagConfig::default());
        let all = declare(&mut d, w);
        let mut completed: BTreeMap<JobId, usize> = BTreeMap::new();
        let mut running: Vec<(JobId, Attempt)> = Vec::new();
        let mut step = 0;
        while completed.len() < all.len() && step < 10_000 {
            let t = Time::from_secs(step as u64);
            if restarts.contains(&step) {
                let snap = serde_json::to_string(&d.snapshot()).unwrap();
                d = DagScheduler::restore(
                    serde_json::from_str(&snap).unwrap(),
                    Scheduler::new(Config::default()),
                    None,
                    t,
                );
                join(&mut d, 3, t);
                running.clear();
            }
            for o in d.poll(t) {
                match o {
                    Output::RunLocal { job } => {
                        *completed.entry(job).or_default() += 1;
                        done(&mut d, job, 0, t);
                    }
                    Output::Start { job, attempt, .. } => {
                        prop_assert!(
                            !completed.contains_key(&job),
                            "job {} placed after completing",
                            job
                        );
                        running.push((job, attempt));
                    }
                    o => prop_assert!(false, "unexpected output {:?}", o),
                }
            }
            // Complete the oldest running job.
            if !running.is_empty() {
                let (j, attempt) = running.remove(0);
                *completed.entry(j).or_default() += 1;
                done(&mut d, j, attempt, t);
            }
            step += 1;
        }
        let ids: BTreeSet<JobId> = completed.keys().copied().collect();
        prop_assert_eq!(ids, all);
        Ok(completed)
    }

    proptest! {
        /// R12: snapshot and restore at random points (any number of times) completes the same jobs
        /// as an uninterrupted run, each exactly once, never placing a completed job again.
        #[test]
        fn snapshot_restore_round_trip(
            w in workload(),
            restarts in prop::collection::btree_set(0usize..40, 0..4),
        ) {
            let straight = run(&w, &BTreeSet::new())?;
            let resumed = run(&w, &restarts)?;
            prop_assert!(straight.values().all(|&n| n == 1));
            prop_assert!(resumed.values().all(|&n| n == 1), "{:?}", resumed);
            let a: HashSet<_> = straight.keys().collect();
            let b: HashSet<_> = resumed.keys().collect();
            prop_assert_eq!(a, b);
        }
    }
}

/// Leaves completed before the walk's entry (a checkpoint replayed early) do not release their
/// successors until the entry completes.
#[test]
fn nothing_runs_before_the_entry() {
    let mut d = whelm(4, DagConfig::default());
    d.declare(vec![plain(1, 1, 7, vec![])], Time::ZERO).unwrap();
    let t = template_of(2, [(0, 1)]);
    let unit = Unit {
        completed: vec![0],
        ..walk(&t, 100, 1, 99)
    };
    d.declare([unit], Time::ZERO).unwrap();
    assert_eq!(
        starts(&d.poll(Time::ZERO)),
        vec![1],
        "node 101 ran before the entry"
    );
    done(&mut d, 1, 1, Time::from_secs(2));
    assert_eq!(starts(&d.poll(Time::from_secs(2))), vec![101]);
}
