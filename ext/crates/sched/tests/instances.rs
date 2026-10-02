//! Implicit template instances behave exactly like the explicit layer.

use std::{collections::BTreeSet, sync::Arc};

use proptest::prelude::*;
use sched::{
    BackfillConfig, Dag, DagConfig, DagJob, DagScheduler, DagTemplate, InstanceSpec, JobId,
    JobSpec, PriorityBackfill, Resources, WorkerState,
};

/// A random group structure: per group, its template index, its passthrough flags and the
/// earlier groups its entry waits for.
#[derive(Clone, Debug)]
struct World {
    templates: Vec<(usize, Vec<(u32, u32)>)>,
    groups: Vec<(usize, Vec<bool>, Vec<usize>)>,
}

/// Ids: group g's entry, its instance base and its done job.
fn entry(g: usize) -> JobId {
    900_000 + g as JobId
}

/// See [`entry`].
fn base(g: usize) -> JobId {
    1_000 * (g as JobId + 1)
}

/// See [`entry`].
fn done(g: usize) -> JobId {
    500_000 + g as JobId
}

/// A random world: 1-3 templates of 1-30 nodes, 1-12 groups.
fn world() -> impl Strategy<Value = World> {
    let template = (1usize..30).prop_flat_map(|n| {
        (
            Just(n),
            prop::collection::vec((0..n as u32, 0..n as u32), 0..3 * n),
        )
    });
    prop::collection::vec(template, 1..4).prop_flat_map(|templates| {
        let k = templates.len();
        let group = (
            0..k,
            prop::collection::vec(prop::bool::weighted(0.2), 30),
            prop::collection::vec(0usize..12, 0..3),
        );
        (Just(templates), prop::collection::vec(group, 1..12)).prop_map(|(templates, groups)| {
            World {
                templates: templates
                    .into_iter()
                    .map(|(n, e)| (n, e.into_iter().filter(|&(a, b)| a < b).collect()))
                    .collect(),
                groups: groups
                    .into_iter()
                    .enumerate()
                    .map(|(g, (t, pass, deps))| {
                        (t, pass, deps.into_iter().filter(|&h| h < g).collect())
                    })
                    .collect(),
            }
        })
    })
}

/// A scheduler holding `w`, built explicitly or with instances.
fn build(w: &World, implicit: bool) -> DagScheduler<PriorityBackfill> {
    build_with(w, implicit, None)
}

/// [`build`] with an open-instance budget.
fn build_with(w: &World, implicit: bool, budget: Option<usize>) -> DagScheduler<PriorityBackfill> {
    let mut d = DagScheduler::new(
        DagConfig {
            auto_submit: false,
            rank_priority: true,
            rank_epsilon: 0.0,
            max_open_instances: budget,
            ..DagConfig::default()
        },
        PriorityBackfill::new(BackfillConfig::default()),
    );
    d.worker_update(WorkerState::new(0, "x", 1, Resources::mem(1)), 0.0);
    let templates: Vec<Arc<DagTemplate>> = w
        .templates
        .iter()
        .map(|(n, e)| Arc::new(DagTemplate::new(*n, e.iter().copied()).unwrap()))
        .collect();
    for (g, (ti, pass, deps)) in w.groups.iter().enumerate() {
        let t = &templates[*ti];
        let n = t.len();
        let entry_deps: Vec<JobId> = deps.iter().map(|&h| done(h)).collect();
        d.declare(
            vec![DagJob::new(
                JobSpec::new(entry(g), Resources::ZERO, g as u64),
                entry_deps,
            )],
            0.0,
        )
        .unwrap();
        let pass: Vec<bool> = (0..n).map(|i| pass[i % pass.len()]).collect();
        let proto = JobSpec::new(0, Resources::ZERO, g as u64);
        if implicit {
            d.open_instance(
                InstanceSpec {
                    template: t.clone(),
                    base: base(g),
                    entry: entry(g),
                    done: done(g),
                    proto,
                    work: vec![1.0; n],
                    passthrough: pass,
                    demand: None,
                    label: None,
                    completed: Vec::new(),
                },
                0.0,
            )
            .unwrap();
        } else {
            d.declare_template(
                t,
                |i| base(g) + i as JobId,
                |i| {
                    if pass[i] {
                        DagJob::passthrough(0, g as u64, vec![], 1.0)
                    } else {
                        DagJob::new(proto.clone(), vec![]).with_work(1.0)
                    }
                },
                &[entry(g)],
                0.0,
            )
            .unwrap();
            let mut deps: Vec<JobId> = t.sinks().map(|i| base(g) + i as JobId).collect();
            deps.push(entry(g));
            d.declare(vec![DagJob::passthrough(done(g), g as u64, deps, 0.0)], 0.0)
                .unwrap();
        }
    }
    d
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Same readiness, event by event, under the same random completion order.
    #[test]
    fn implicit_instances_match_the_explicit_layer(w in world(), seed in any::<u64>(), snap_at in 0usize..40) {
        let mut x = build(&w, false);
        let mut y = build(&w, true);
        // Ranks flow across instances as they do along explicit edges.
        for g in 0..w.groups.len() {
            let (rx, ry) = (x.rank(entry(g)).unwrap(), y.rank(entry(g)).unwrap());
            prop_assert!((rx - ry).abs() < 1e-6 * rx.max(1.0), "entry {} rank {} vs {}", g, rx, ry);
        }
        let mut ready: BTreeSet<JobId> = BTreeSet::new();
        let (a, b): (BTreeSet<JobId>, BTreeSet<JobId>) = (x.take_ready().into_iter().collect(), y.take_ready().into_iter().collect());
        prop_assert_eq!(&a, &b);
        ready.extend(a);
        let mut rng = seed;
        let mut steps = 0;
        while !ready.is_empty() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if steps == snap_at {
                // Snapshot, round-trip through JSON, restore: held jobs are reported again.
                let json = serde_json::to_string(&y.snapshot()).unwrap();
                y = DagScheduler::restore(
                    serde_json::from_str(&json).unwrap(),
                    PriorityBackfill::new(BackfillConfig::default()),
                    steps as f64,
                );
                let again: BTreeSet<JobId> = y.take_ready().into_iter().collect();
                prop_assert_eq!(&again, &ready, "restored held set");
            }
            let j = *ready.iter().nth((rng >> 33) as usize % ready.len()).unwrap();
            ready.remove(&j);
            x.completed(j, steps as f64);
            y.completed(j, steps as f64);
            let (a, b): (BTreeSet<JobId>, BTreeSet<JobId>) = (x.take_ready().into_iter().collect(), y.take_ready().into_iter().collect());
            prop_assert_eq!(&a, &b, "after completing {}", j);
            ready.extend(a);
            let (sx, sy) = (x.dag_stats(), y.dag_stats());
            prop_assert_eq!(sx.held, sy.held);
            steps += 1;
        }
        // Everything ran: every entry and every non-passthrough node.
        let (sx, sy) = (x.dag_stats(), y.dag_stats());
        prop_assert_eq!(sx.pending + sx.held + sx.submitted, 0);
        prop_assert_eq!(sy.pending + sy.held + sy.submitted, 0);
        prop_assert_eq!(sy.instances, 0);
    }
}

/// A two-node instance with an explicit job after its `done`.
fn small() -> DagScheduler<PriorityBackfill> {
    let mut d = DagScheduler::new(
        DagConfig::default(),
        PriorityBackfill::new(BackfillConfig::default()),
    );
    d.worker_update(WorkerState::new(0, "x", 8, Resources::mem(10)), 0.0);
    d.declare(
        vec![DagJob::new(JobSpec::new(1, Resources::ZERO, 0), vec![])],
        0.0,
    )
    .unwrap();
    d.declare(
        vec![DagJob::new(JobSpec::new(2, Resources::ZERO, 0), vec![99])],
        0.0,
    )
    .unwrap();
    d.open_instance(
        InstanceSpec {
            template: Arc::new(DagTemplate::new(2, [(0, 1)]).unwrap()),
            base: 10,
            entry: 1,
            done: 99,
            proto: JobSpec::new(0, Resources::ZERO, 0),
            work: vec![3.0, 4.0],
            passthrough: vec![false, false],
            demand: None,
            label: None,
            completed: Vec::new(),
        },
        0.0,
    )
    .unwrap();
    d
}

/// Instance nodes run, explain themselves, and release explicit dependents of `done`.
#[test]
fn instance_lifecycle() {
    let mut d = small();
    assert!(d.explain(10).unwrap().contains("entry job 1"));
    assert_eq!(d.dispatch(0.0), vec![(1, 0)]);
    d.completed(1, 1.0);
    assert!(d.explain(11).unwrap().contains("1 dependency"));
    assert_eq!(d.dispatch(1.0), vec![(10, 0)]);
    assert_eq!(d.dag_stats().instances, 1);
    d.completed(10, 2.0);
    assert_eq!(d.dispatch(2.0), vec![(11, 0)]);
    d.completed(11, 3.0);
    assert_eq!(d.dag_stats().instances, 0, "closed on its last node");
    assert_eq!(
        d.dispatch(3.0),
        vec![(2, 0)],
        "done released the explicit dependent"
    );
}

/// Cancelling an instance node cancels the instance and what waits for its `done`; cancelling the
/// entry cancels the instance too.
#[test]
fn instance_cancellation() {
    let mut d = small();
    d.dispatch(0.0);
    d.completed(1, 1.0);
    assert_eq!(d.dispatch(1.0), vec![(10, 0)]);
    let mut c = d.cancel(11);
    c.sort_unstable();
    assert_eq!(c, vec![2, 10, 11]);
    assert_eq!(
        d.stats().running,
        0,
        "the submitted node was cancelled in the policy"
    );
    let s = d.dag_stats();
    assert_eq!((s.instances, s.pending, s.submitted), (0, 0, 0));

    let mut d = small();
    let mut c = d.cancel(1);
    c.sort_unstable();
    assert_eq!(c, vec![1, 2, 10, 11]);
}

/// Overlapping ids and a declared `done` are refused.
#[test]
fn instance_validation() {
    let mut d = small();
    let spec = |base, done| InstanceSpec {
        template: Arc::new(DagTemplate::new(3, []).unwrap()),
        base,
        entry: 1,
        done,
        proto: JobSpec::new(0, Resources::ZERO, 0),
        work: vec![1.0; 3],
        passthrough: vec![false; 3],
        demand: None,
        label: None,
        completed: Vec::new(),
    };
    assert!(
        d.open_instance(spec(9, 98), 0.0).is_err(),
        "overlaps 10..12"
    );
    assert!(
        d.open_instance(spec(0, 98), 0.0).is_err(),
        "overlaps explicit job 1"
    );
    assert!(
        d.open_instance(spec(20, 2), 0.0).is_err(),
        "done is a declared job"
    );
    assert!(d.open_instance(spec(20, 98), 0.0).is_ok());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// A frontier budget never deadlocks and is never exceeded.
    #[test]
    fn open_instance_budget(w in world(), seed in any::<u64>(), budget in 1usize..4) {
        let mut d = build_with(&w, true, Some(budget));
        let mut ready: BTreeSet<JobId> = d.take_ready().into_iter().collect();
        let mut rng = seed;
        let mut steps = 0;
        while !ready.is_empty() {
            prop_assert!(d.dag_stats().instances_open <= budget);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = *ready.iter().nth((rng >> 33) as usize % ready.len()).unwrap();
            ready.remove(&j);
            d.completed(j, steps as f64);
            ready.extend(d.take_ready());
            steps += 1;
        }
        let s = d.dag_stats();
        prop_assert_eq!(s.pending + s.held + s.submitted + s.instances, 0, "deadlocked: {:?}", s);
    }
}
