//! Driving a coordinator from the DAG layer: per-node demands and labels, opening a walk with
//! nodes already complete, closing a walk early, coordinator-local jobs, and snapshot/restore of
//! an open frontier.

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    sync::Arc,
};

use proptest::prelude::*;
use sched::{
    Attempt, Config, DagConfig, DagJob, DagScheduler, DagTemplate, Input, InstanceSpec, JobId,
    JobSpec, NodeLabel, Output, Policy, Resources, Scheduler, WorkerState,
};

/// A DAG layer over the default backfill policy with one worker of `slots` slots.
fn sched(slots: usize, config: DagConfig) -> DagScheduler<Scheduler> {
    let mut d = DagScheduler::new(config, Scheduler::new(Config::default()));
    join(&mut d, slots, 0.0);
    d
}

/// Worker 0, with `slots` slots, joins.
fn join(d: &mut DagScheduler<Scheduler>, slots: usize, now: f64) {
    let w = WorkerState::new(0, "x", slots, Resources::mem(1 << 40));
    d.handle(Input::Worker(w), now);
}

/// Report attempt `attempt` of `job` done (0 for a local job).
fn done(d: &mut DagScheduler<Scheduler>, job: JobId, attempt: Attempt, now: f64) {
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

/// An instance of `template` at `base`, entered by `entry`, finished as `done`.
fn instance(template: &Arc<DagTemplate>, base: JobId, entry: JobId, done: JobId) -> InstanceSpec {
    let n = template.len();
    InstanceSpec {
        template: template.clone(),
        base,
        entry,
        done,
        proto: JobSpec::new(0, Resources::mem(1), 7),
        work: vec![1.0; n],
        passthrough: vec![false; n],
        demand: None,
        label: None,
        completed: Vec::new(),
    }
}

/// A local entry job `1` (the zero step), already completed.
fn with_entry(d: &mut DagScheduler<Scheduler>) {
    d.declare(
        vec![DagJob::new(JobSpec::new(1, Resources::ZERO, 7), vec![]).local()],
        0.0,
    )
    .unwrap();
    assert_eq!(d.poll(0.0), vec![Output::RunLocal { job: 1 }]);
    done(d, 1, 0, 0.0);
}

/// R8: each node is submitted with its own demand, and explained under its label.
#[test]
fn per_node_demand_and_label() {
    let mut d = sched(16, DagConfig::default());
    let t = Arc::new(DagTemplate::new(3, []).unwrap());
    d.declare(
        vec![DagJob::new(JobSpec::new(1, Resources::ZERO, 7), vec![]).local()],
        0.0,
    )
    .unwrap();
    d.open_instance(
        InstanceSpec {
            demand: Some(Arc::from(vec![
                Resources::mem(3),
                Resources::mem(5),
                Resources::mem(9),
            ])),
            label: Some(NodeLabel(Arc::new(|i| format!("Sq({i})")))),
            ..instance(&t, 100, 1, 99)
        },
        0.0,
    )
    .unwrap();
    assert!(
        d.explain(101).unwrap().starts_with("[Sq(1)]"),
        "{:?}",
        d.explain(101)
    );
    assert_eq!(d.poll(0.0), vec![Output::RunLocal { job: 1 }]);
    done(&mut d, 1, 0, 0.0);
    assert_eq!(starts(&d.poll(0.0)).len(), 3);
    assert_eq!(d.stats().workers[0].placed, Resources::mem(17));
}

/// The completion order of an instance driven to the end: every round, poll, then complete every
/// started job in id order; also returns how often `done` (99) fired.
fn drive(d: &mut DagScheduler<Scheduler>) -> (Vec<JobId>, usize) {
    let mut order = Vec::new();
    let mut fired = 0;
    for _ in 0..1000 {
        let out = d.poll(0.0);
        fired += passed(&out).iter().filter(|&&j| j == 99).count();
        let mut placed = starts(&out);
        if placed.is_empty() {
            break;
        }
        placed.sort_unstable();
        for j in placed {
            order.push(j);
            done(d, j, 1, 0.0);
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
    /// R9: opening with a completed set S behaves exactly like the template without S, every
    /// edge out of S satisfied: the same nodes run afterwards, round by round (every ready node
    /// runs each round), `done` fires once, and no node of S ever runs.
    #[test]
    fn opening_with_completed_equals_completing(
        (n, edges) in template(),
        mask in prop::collection::vec(any::<bool>(), 14),
    ) {
        let t = Arc::new(DagTemplate::new(n, edges).unwrap());
        let s: Vec<u32> = t
            .topological_order()
            .iter()
            .copied()
            .filter(|&i| mask[i as usize])
            .collect();
        let cfg = DagConfig { record_passthrough: true, ..DagConfig::default() };
        let mut a = sched(1000, cfg.clone());
        with_entry(&mut a);
        a.open_instance(InstanceSpec { completed: s.clone(), ..instance(&t, 100, 1, 99) }, 0.0)
            .unwrap();
        // The reference: the nodes outside S, declared explicitly after the entry, and `done`
        // as a passthrough after all of them.
        let mut b = sched(1000, cfg);
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
                DagJob::new(JobSpec::new(100 + JobId::from(i), Resources::mem(1), 7), deps)
            })
            .collect();
        let nodes = rest.iter().map(|&i| 100 + JobId::from(i)).collect();
        jobs.push(DagJob::passthrough(99, 7, nodes, 0.0));
        b.declare(jobs, 0.0).unwrap();
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

/// R10: closing early completes `done` once, withdraws waiting nodes, and returns the running
/// ones, whose later completions only free their resources.
#[test]
fn close_instance_early() {
    let cfg = DagConfig {
        record_passthrough: true,
        ..DagConfig::default()
    };
    let mut d = sched(2, cfg);
    with_entry(&mut d);
    // Four independent nodes, a chain after them; two slots.
    let t = Arc::new(DagTemplate::new(6, [(0, 4), (1, 4), (4, 5)]).unwrap());
    d.open_instance(instance(&t, 100, 1, 99), 0.0).unwrap();
    // Something after the walk.
    d.declare(
        vec![DagJob::new(
            JobSpec::new(200, Resources::mem(1), 8),
            vec![99],
        )],
        0.0,
    )
    .unwrap();
    assert_eq!(starts(&d.poll(0.0)), vec![100, 101]);
    let mut running = d.close_instance(99, 1.0).unwrap();
    running.sort_unstable();
    assert_eq!(running, vec![100, 101]);
    // `done` passed, and the walk's dependent is ready, but both slots are still taken.
    assert_eq!(d.poll(1.0), vec![Output::Passed { job: 99 }]);
    // The waiting nodes were withdrawn.
    let st = d.stats();
    assert_eq!((st.waiting, st.running), (1, 2));
    assert!(d.close_instance(99, 1.0).is_err(), "closed twice");
    // Ignored completions free their slots and change nothing else.
    done(&mut d, 100, 1, 2.0);
    done(&mut d, 101, 1, 2.0);
    let out = d.poll(2.0);
    assert!(passed(&out).is_empty(), "no second done");
    assert_eq!(starts(&out), vec![200]);
    done(&mut d, 200, 1, 3.0);
    assert!(d.poll(3.0).is_empty());
    let st = d.stats();
    assert_eq!((st.waiting, st.running), (0, 0));
    assert_eq!(st.workers[0].placed, Resources::ZERO);
}

/// R11: local jobs are never submitted; they are announced as [`Output::RunLocal`], not
/// [`Output::Ready`], `release` refuses them, and they are reported done with attempt 0.
#[test]
fn local_jobs_stay_on_the_caller() {
    let mut d = sched(
        4,
        DagConfig {
            auto_submit: false,
            ..DagConfig::default()
        },
    );
    d.declare(
        vec![
            DagJob::new(JobSpec::new(1, Resources::ZERO, 0), vec![]).local(),
            DagJob::new(JobSpec::new(2, Resources::mem(1), 0), vec![1]),
            DagJob::new(JobSpec::new(3, Resources::ZERO, 0), vec![2]).local(),
        ],
        0.0,
    )
    .unwrap();
    assert_eq!(d.poll(0.0), vec![Output::RunLocal { job: 1 }]);
    assert!(!d.release(1, 0.0));
    assert!(d.poll(0.0).is_empty());
    done(&mut d, 1, 0, 1.0);
    assert_eq!(d.poll(1.0), vec![Output::Ready { job: 2 }]);
    assert!(d.release(2, 1.0));
    assert_eq!(starts(&d.poll(1.0)), vec![2]);
    done(&mut d, 2, 1, 2.0);
    assert_eq!(d.poll(2.0), vec![Output::RunLocal { job: 3 }]);
    assert_eq!(d.stats().placements_total, 1);
}

/// A random workload: explicit jobs (some local) with dependencies on earlier ones, and walks
/// (instances) entered by an explicit job and depended on by a later one.
#[derive(Clone, Debug)]
struct Workload {
    explicit: Vec<(Vec<usize>, bool)>,
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
            let explicit = ex
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
            Workload { explicit, walks }
        })
}

/// Job ids of the workload.
const WALK_BASE: JobId = 1000;
/// Explicit job `i`.
fn ex_id(i: usize) -> JobId {
    i as JobId
}
/// Walk `w`'s `done` job.
fn done_id(w: usize) -> JobId {
    500 + w as JobId
}

/// Declare the workload. A walk's `done` is depended on by a fresh explicit job (`600 + w`),
/// which a later explicit job `after` also waits for only if `after > entry` (no cycles).
fn declare(d: &mut DagScheduler<Scheduler>, w: &Workload) -> BTreeSet<JobId> {
    let mut all = BTreeSet::new();
    let mut jobs = Vec::new();
    for (i, (deps, local)) in w.explicit.iter().enumerate() {
        let mut deps: Vec<JobId> = deps.iter().map(|&j| ex_id(j)).collect();
        for (k, walk) in w.walks.iter().enumerate() {
            if walk.3 == i && walk.3 > walk.0 {
                deps.push(600 + k as JobId);
            }
        }
        let mut j = DagJob::new(JobSpec::new(ex_id(i), Resources::mem(1), i as u64), deps);
        j.local = *local;
        jobs.push(j);
        all.insert(ex_id(i));
    }
    for k in 0..w.walks.len() {
        jobs.push(DagJob::new(
            JobSpec::new(600 + k as JobId, Resources::mem(1), 50),
            vec![done_id(k)],
        ));
        all.insert(600 + k as JobId);
    }
    d.declare(jobs, 0.0).unwrap();
    for (k, (entry, len, edges, _)) in w.walks.iter().enumerate() {
        let t = Arc::new(DagTemplate::new(*len, edges.iter().copied()).unwrap());
        let base = WALK_BASE + 100 * k as JobId;
        d.open_instance(instance(&t, base, ex_id(*entry), done_id(k)), 0.0)
            .unwrap();
        all.extend((0..*len).map(|i| base + i as JobId));
    }
    all
}

/// Run to the end, snapshotting and restoring (onto a fresh policy, all running work lost) at
/// each step in `restarts`. Returns, per job, how often it completed, and checks dependencies.
fn run(w: &Workload, restarts: &BTreeSet<usize>) -> Result<BTreeMap<JobId, usize>, TestCaseError> {
    let mut d = sched(3, DagConfig::default());
    let all = declare(&mut d, w);
    let mut completed: BTreeMap<JobId, usize> = BTreeMap::new();
    let mut running: Vec<(JobId, Attempt)> = Vec::new();
    let mut step = 0;
    while completed.len() < all.len() && step < 10_000 {
        let t = step as f64;
        if restarts.contains(&step) {
            let snap = serde_json::to_string(&d.snapshot()).unwrap();
            d = DagScheduler::restore(
                serde_json::from_str(&snap).unwrap(),
                Scheduler::new(Config::default()),
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

/// Nodes completed before the walk's entry (a checkpoint replayed early) do not release their
/// successors until the entry completes.
#[test]
fn nothing_runs_before_the_entry() {
    let mut d = sched(4, DagConfig::default());
    d.declare(
        vec![DagJob::new(JobSpec::new(1, Resources::mem(1), 7), vec![])],
        0.0,
    )
    .unwrap();
    let t = Arc::new(DagTemplate::new(2, [(0, 1)]).unwrap());
    let spec = InstanceSpec {
        completed: vec![0],
        ..instance(&t, 100, 1, 99)
    };
    d.open_instance(spec, 0.0).unwrap();
    assert_eq!(
        starts(&d.poll(0.0)),
        vec![1],
        "node 101 ran before the entry"
    );
    done(&mut d, 1, 1, 2.0);
    assert_eq!(starts(&d.poll(2.0)), vec![101]);
}
