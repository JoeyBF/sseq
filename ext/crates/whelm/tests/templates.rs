//! Passthrough jobs, work updates, templates and substitution, and the avoid/class constraints.

use std::{collections::BTreeMap, sync::Arc};

use proptest::prelude::*;
use whelm::{
    Config, Constraint, DagConfig, DagError, DagJob, DagScheduler, DagTemplate, Input, JobId,
    JobSpec, NodeSource, Output, Policy, Resources, Scheduler, TemplateNode, TemplateSpec, Unit,
    WorkerId, WorkerState,
};

/// A DAG layer over one worker with many slots.
fn dag(config: DagConfig) -> DagScheduler<Scheduler> {
    let mut d = DagScheduler::new(config, Scheduler::new(Config::default()));
    let w = WorkerState {
        class: "x".into(),
        slots: 64,
        budget: Resources::mem(1000),
        ..Default::default()
    };
    d.handle(Input::Worker(w), 0.0);
    d
}

/// Report the first attempt of `job` done.
fn complete(p: &mut impl Policy, job: JobId, now: f64) {
    p.handle(Input::Done { job, attempt: 1 }, now);
}

/// The first attempts a poll started, as `(job, worker)`.
fn starts(p: &mut impl Policy, now: f64) -> Vec<(JobId, WorkerId)> {
    p.poll(now)
        .into_iter()
        .map(|o| match o {
            Output::Start {
                job,
                attempt: 1,
                worker,
            } => (job, worker),
            o => panic!("unexpected output {o:?}"),
        })
        .collect()
}

/// A unit job in group 0.
fn job(id: JobId, deps: &[JobId]) -> DagJob {
    DagJob {
        spec: JobSpec {
            id,
            demand: Resources::mem(1),
            ..Default::default()
        },
        deps: deps.to_vec(),
        ..Default::default()
    }
}

/// A unit job in group 0 with a work estimate.
fn worked(id: JobId, deps: &[JobId], work: f64) -> DagJob {
    DagJob {
        work_estimate: Some(work),
        ..job(id, deps)
    }
}

/// A passthrough in group 0 worth `work` in ranks.
fn passthrough(id: JobId, deps: &[JobId], work: f64) -> DagJob {
    DagJob {
        spec: JobSpec {
            id,
            ..Default::default()
        },
        passthrough: true,
        ..worked(id, deps, work)
    }
}

/// The template of `n` worker jobs of unit work with the given edges.
fn template_of(
    n: usize,
    edges: impl IntoIterator<Item = (u32, u32)>,
) -> Result<DagTemplate, DagError> {
    TemplateSpec {
        edges: edges.into_iter().collect(),
        ..TemplateSpec::jobs(n)
    }
    .build()
}

/// The ids started by a poll, sorted.
fn placed(d: &mut DagScheduler<Scheduler>, now: f64) -> Vec<JobId> {
    let mut v: Vec<JobId> = starts(d, now).into_iter().map(|p| p.0).collect();
    v.sort_unstable();
    v
}

/// A passthrough completes on readiness, counts in ranks and never reaches the policy.
#[test]
fn passthrough_jobs_complete_by_themselves() {
    let mut d = dag(DagConfig {
        record_passthrough: true,
        ..DagConfig::default()
    });
    // 1 -> done(2) -> 3: the passthrough never reaches the policy.
    d.declare(
        vec![job(1, &[]), passthrough(2, &[1], 5.0), job(3, &[2])],
        0.0,
    )
    .unwrap();
    assert_eq!(placed(&mut d, 0.0), vec![1]);
    assert_eq!(
        d.rank(1),
        Some(7.0),
        "the passthrough's work counts in ranks"
    );
    complete(&mut d, 1, 1.0);
    let start = Output::Start {
        job: 3,
        attempt: 1,
        worker: 0,
    };
    assert_eq!(d.poll(1.0), vec![Output::Passed { job: 2 }, start]);
    assert_eq!(d.stats().placements_total, 2);
}

/// Leaf `k` of a sourced unit weighs `k + 1`; leaf 1 of unit 10 does nothing.
struct SkipOne;

impl NodeSource for SkipOne {
    /// `leaf + 1`.
    fn work(&self, _unit: JobId, leaf: u32) -> f64 {
        f64::from(leaf) + 1.0
    }

    /// Leaf 1 of unit 10 only.
    fn passthrough(&self, unit: JobId, leaf: u32) -> bool {
        (unit, leaf) == (10, 1)
    }
}

/// A source makes a leaf a passthrough in one unit of a template and not in another, in ranks
/// and in what runs.
#[test]
fn a_source_makes_a_leaf_a_passthrough_in_one_unit() {
    let mut d = dag(DagConfig {
        record_passthrough: true,
        ..DagConfig::default()
    })
    .with_source(Arc::new(SkipOne));
    // A chain: job, local job, job.
    let nodes = vec![
        TemplateNode::Job(1.0),
        TemplateNode::Local(1.0),
        TemplateNode::Job(1.0),
    ];
    let t = TemplateSpec {
        nodes,
        edges: vec![(0, 1), (1, 2)],
    };
    let t = Arc::new(t.build().unwrap());
    let unit = |id, base| Unit {
        id,
        base,
        template: t.clone(),
        spec: JobSpec {
            demand: Resources::mem(1),
            ..Default::default()
        },
        sourced: true,
        ..Default::default()
    };
    d.declare([unit(10, 100), unit(20, 200)], 0.0).unwrap();
    assert_eq!(d.rank(100), Some(1.0 + 3.0), "the no-op weighs nothing");
    assert_eq!(d.rank(200), Some(1.0 + 2.0 + 3.0));
    assert_eq!(placed(&mut d, 0.0), vec![100, 200]);
    complete(&mut d, 100, 1.0);
    complete(&mut d, 200, 1.0);
    assert_eq!(
        d.poll(1.0),
        vec![
            Output::Passed { job: 101 },
            Output::RunLocal { job: 201 },
            Output::Start {
                job: 102,
                attempt: 1,
                worker: 0
            },
        ]
    );
}

/// A chain of `N` passthroughs completes without recursing.
#[test]
fn long_passthrough_chains_do_not_recurse() {
    let mut d = dag(DagConfig::default());
    const N: u64 = 200_000;
    let mut jobs = vec![job(0, &[])];
    jobs.extend((1..N).map(|i| passthrough(i, &[i - 1], 0.0)));
    jobs.push(job(N, &[N - 1]));
    d.declare(jobs, 0.0).unwrap();
    assert_eq!(placed(&mut d, 0.0), vec![0]);
    complete(&mut d, 0, 1.0);
    assert_eq!(placed(&mut d, 1.0), vec![N]);
    assert_eq!(d.dag_stats().pending, 0);
}

/// A passthrough with no dependencies completes during `declare`.
#[test]
fn a_ready_passthrough_completes_at_declaration() {
    let mut d = dag(DagConfig::default());
    d.declare(vec![passthrough(1, &[], 0.0), job(2, &[1])], 0.0)
        .unwrap();
    assert_eq!(placed(&mut d, 0.0), vec![2]);
}

/// Lowering and raising work moves ranks both ways, switching the critical path.
#[test]
fn update_work_raises_and_lowers_ranks() {
    let mut d = dag(DagConfig {
        rank_epsilon: 0.0,
        ..DagConfig::default()
    });
    // 1 -> 2 -> 4 and 1 -> 3 -> 4.
    d.declare(
        vec![
            worked(1, &[], 1.0),
            worked(2, &[1], 10.0),
            worked(3, &[1], 2.0),
            worked(4, &[2, 3], 1.0),
        ],
        0.0,
    )
    .unwrap();
    assert_eq!(d.rank(1), Some(12.0));
    assert!(d.update_work(2, 0.5));
    assert_eq!(
        d.rank(1),
        Some(4.0),
        "the other branch is now the critical path"
    );
    assert!(d.update_work(3, 20.0));
    assert_eq!(d.rank(1), Some(22.0));
    assert!(!d.update_work(99, 1.0));
}

/// Templates reject cycles, merge duplicate edges, and compute critical paths.
#[test]
fn template_basics() {
    assert_eq!(
        template_of(3, [(0, 1), (1, 2), (2, 0)]).unwrap_err(),
        DagError::Cycle { job: 0 }
    );
    let t = template_of(4, [(0, 1), (0, 2), (1, 3), (2, 3), (0, 1)]).unwrap();
    assert_eq!(t.len(), 4);
    assert_eq!(t.edge_count(), 4);
    assert_eq!(t.sources().collect::<Vec<_>>(), vec![0]);
    assert_eq!(t.sinks().collect::<Vec<_>>(), vec![3]);
    assert_eq!(t.critical_path(|i| [1.0, 5.0, 2.0, 1.0][i]), 7.0);
}

/// Critical nodes are exactly those on a longest chain.
#[test]
fn critical_nodes_lie_on_the_longest_chain() {
    // 0 -> 1 -> 3 (1 + 5 + 1) and 0 -> 2 -> 3 (1 + 2 + 1); 4 is isolated (3).
    let t = template_of(5, [(0, 1), (0, 2), (1, 3), (2, 3)]).unwrap();
    let w = [1.0, 5.0, 2.0, 1.0, 3.0];
    assert_eq!(
        t.critical_nodes(|i| w[i], 1e-9),
        vec![true, true, false, true, false]
    );
    // A tolerance wide enough to include the other branch (4 / 7 of the critical path).
    assert_eq!(
        t.critical_nodes(|i| w[i], 0.5),
        vec![true, true, true, true, false]
    );
}

/// Two units of a template chain through their dependencies.
#[test]
fn units_of_a_template() {
    // A diamond whose node 2 is a passthrough, twice; unit 2 waits for unit 1.
    let t = TemplateSpec {
        nodes: vec![
            TemplateNode::Job(1.0),
            TemplateNode::Job(1.0),
            TemplateNode::Pass(0.0),
            TemplateNode::Job(1.0),
        ],
        edges: vec![(0, 1), (0, 2), (1, 3), (2, 3)],
    };
    let t = Arc::new(t.build().unwrap());
    let mut d = dag(DagConfig::default());
    for g in 1..=2u64 {
        let deps: Vec<JobId> = if g == 2 { vec![101] } else { vec![] };
        let unit = Unit {
            id: 100 + g,
            base: g * 10,
            template: t.clone(),
            deps,
            spec: JobSpec {
                demand: Resources::mem(1),
                group: g,
                ..Default::default()
            },
            ..Default::default()
        };
        d.declare([unit], 0.0).unwrap();
    }
    let mut order = Vec::new();
    for step in 0..10 {
        let now = step as f64;
        let out = placed(&mut d, now);
        for &j in &out {
            complete(&mut d, j, now + 0.5);
        }
        order.push(out);
    }
    let flat: Vec<JobId> = order.into_iter().flatten().collect();
    assert_eq!(flat, vec![10, 11, 13, 20, 21, 23]);
}

/// A substituted unit numbers its leaves after the enclosing node's offset, weighs its span in
/// bottom levels, and runs in place of its node.
#[test]
fn substituted_units() {
    // inner: 0 -> 1 (work 2, 3); outer: job 0 -> inner -> job 2, and an isolated job 3.
    let inner = TemplateSpec {
        nodes: vec![TemplateNode::Job(2.0), TemplateNode::Job(3.0)],
        edges: vec![(0, 1)],
    };
    let inner = Arc::new(inner.build().unwrap());
    let outer = TemplateSpec {
        nodes: vec![
            TemplateNode::Job(1.0),
            TemplateNode::Unit(inner.clone()),
            TemplateNode::Job(4.0),
            TemplateNode::Job(1.0),
        ],
        edges: vec![(0, 1), (1, 2)],
    };
    let outer = Arc::new(outer.build().unwrap());
    assert_eq!(outer.leaves(), 5);
    assert_eq!(
        (0..4).map(|i| outer.leaf_offset(i)).collect::<Vec<_>>(),
        vec![0, 1, 3, 4]
    );
    assert_eq!(inner.span(), 5.0);
    assert_eq!(outer.span(), 10.0);
    let mut d = dag(DagConfig {
        rank_epsilon: 0.0,
        ..DagConfig::default()
    });
    let unit = Unit {
        id: 1,
        base: 10,
        template: outer,
        spec: JobSpec {
            demand: Resources::mem(1),
            ..Default::default()
        },
        scale: Some(2.0),
        ..Default::default()
    };
    d.declare([unit, worked(2, &[1], 7.0).into()], 0.0).unwrap();
    // Leaf 12 (inner node 1): 2 * (3 + 4) + 7.
    assert_eq!(d.rank(12), Some(21.0));
    assert_eq!(d.rank(1), Some(27.0));
    assert_eq!(placed(&mut d, 0.0), vec![10, 14]);
    assert!(
        d.explain(11).unwrap().contains("to be entered"),
        "{:?}",
        d.explain(11)
    );
    complete(&mut d, 10, 1.0);
    assert_eq!(placed(&mut d, 1.0), vec![11]);
    assert_eq!(d.dag_stats().frames, 2);
    complete(&mut d, 11, 2.0);
    assert_eq!(placed(&mut d, 2.0), vec![12]);
    complete(&mut d, 12, 3.0);
    assert_eq!(placed(&mut d, 3.0), vec![13]);
    complete(&mut d, 13, 4.0);
    complete(&mut d, 14, 4.0);
    assert_eq!(placed(&mut d, 4.0), vec![2]);
    assert_eq!(d.dag_stats().frames, 1, "only job 2's");
}

/// Forbids and required classes exclude workers; a job excluded everywhere waits and says so.
#[test]
fn forbid_and_class_are_hard_constraints() {
    let mut p = Scheduler::new(Config::default());
    let join = |p: &mut Scheduler, id, class, slots, now| {
        let w = WorkerState {
            id,
            class: String::from(class),
            slots,
            budget: Resources::mem(100),
            ..Default::default()
        };
        p.handle(Input::Worker(w), now);
    };
    join(&mut p, 1, "h200", 4, 0.0);
    join(&mut p, 2, "l40s", 4, 0.0);
    let spec = |id, constraints| JobSpec {
        id,
        demand: Resources::mem(1),
        constraints,
        ..Default::default()
    };
    let retry = spec(
        1,
        vec![Constraint::forbid_worker(1), Constraint::prefer_worker(1)],
    );
    p.handle(Input::Submit(retry), 0.0);
    let pinned = spec(2, vec![Constraint::require_class("h200")]);
    p.handle(Input::Submit(pinned), 0.0);
    assert_eq!(starts(&mut p, 0.0), vec![(1, 2), (2, 1)]);
    // A job excluded everywhere waits, and says why.
    let nowhere = spec(3, vec![Constraint::require_class("v100")]);
    p.handle(Input::Submit(nowhere), 1.0);
    assert!(starts(&mut p, 1.0).is_empty());
    assert!(p.explain(3).unwrap().contains("2 worker(s) excluded"));
    join(&mut p, 3, "v100", 1, 2.0);
    assert_eq!(starts(&mut p, 2.0), vec![(3, 3)]);
}

/// The exact longest path below each job, by brute force.
fn exact_ranks(
    work: &BTreeMap<JobId, f64>,
    deps: &BTreeMap<JobId, Vec<JobId>>,
) -> BTreeMap<JobId, f64> {
    /// Memoised longest path below `j`.
    fn go(
        j: JobId,
        work: &BTreeMap<JobId, f64>,
        kids: &BTreeMap<JobId, Vec<JobId>>,
        memo: &mut BTreeMap<JobId, f64>,
    ) -> f64 {
        if let Some(&r) = memo.get(&j) {
            return r;
        }
        let below = kids.get(&j).map_or(0.0, |k| {
            k.iter()
                .map(|&c| go(c, work, kids, memo))
                .fold(0.0, f64::max)
        });
        let r = work[&j] + below;
        memo.insert(j, r);
        r
    }
    let mut kids: BTreeMap<JobId, Vec<JobId>> = BTreeMap::new();
    for (&j, ds) in deps {
        for &d in ds {
            kids.entry(d).or_default().push(j);
        }
    }
    let mut memo = BTreeMap::new();
    work.keys()
        .map(|&j| (j, go(j, work, &kids, &mut memo)))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// With `rank_epsilon = 0`, ranks stay exact longest paths through declarations (in random
    /// batches, so forward references occur) and arbitrary work updates up and down.
    #[test]
    fn ranks_stay_exact(
        n in 1usize..30,
        edges in prop::collection::vec((0usize..30, 0usize..30), 0..80),
        works in prop::collection::vec(0.0f64..10.0, 30),
        updates in prop::collection::vec((0usize..30, 0.0f64..20.0), 0..20),
        batch in 1usize..6,
    ) {
        let mut deps: BTreeMap<JobId, Vec<JobId>> = (0..n as JobId).map(|i| (i, Vec::new())).collect();
        for (a, b) in edges {
            let (a, b) = (a.min(b), a.max(b));
            if a != b && b < n {
                deps.get_mut(&(b as JobId)).unwrap().push(a as JobId);
            }
        }
        let mut work: BTreeMap<JobId, f64> = (0..n as JobId).map(|i| (i, works[i as usize])).collect();
        // No workers: nothing runs, so every job stays in the graph.
        let mut d = DagScheduler::new(
            DagConfig { rank_epsilon: 0.0, ..DagConfig::default() },
            Scheduler::new(Config::default()),
        );
        let ids: Vec<JobId> = (0..n as JobId).rev().collect();
        for chunk in ids.chunks(batch) {
            let jobs: Vec<DagJob> = chunk.iter().map(|&i| worked(i, &deps[&i], work[&i])).collect();
            d.declare(jobs, 0.0).unwrap();
        }
        for (j, w) in updates {
            if j < n {
                d.update_work(j as JobId, w);
                work.insert(j as JobId, w);
            }
        }
        let exact = exact_ranks(&work, &deps);
        for (&j, &r) in &exact {
            let got = d.rank(j).unwrap();
            prop_assert!((got - r).abs() < 1e-9, "job {}: rank {} vs exact {}", j, got, r);
        }
    }
}

/// Descendant sets of every node, by brute force.
fn closure(t: &DagTemplate) -> Vec<std::collections::BTreeSet<usize>> {
    (0..t.len())
        .map(|v| {
            let mut seen = std::collections::BTreeSet::new();
            let mut stack: Vec<usize> = t.successors(v).iter().map(|&c| c as usize).collect();
            while let Some(c) = stack.pop() {
                if seen.insert(c) {
                    stack.extend(t.successors(c).iter().map(|&x| x as usize));
                }
            }
            seen
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The reduction keeps reachability and critical paths, and no kept edge is implied.
    #[test]
    fn transitive_reduction_is_minimal_and_equivalent(
        n in 1usize..40,
        edges in prop::collection::vec((0u32..40, 0u32..40), 0..200),
        works in prop::collection::vec(0.0f64..10.0, 40),
    ) {
        let edges: Vec<(u32, u32)> =
            edges.into_iter().filter(|&(a, b)| a < b && (b as usize) < n).collect();
        let t = template_of(n, edges).unwrap();
        let r = t.transitive_reduction();
        prop_assert_eq!(closure(&t), closure(&r));
        prop_assert!((t.critical_path(|i| works[i]) - r.critical_path(|i| works[i])).abs() < 1e-9);
        let reach = closure(&r);
        for a in 0..n {
            for &c in r.successors(a) {
                for &other in r.successors(a) {
                    prop_assert!(
                        other == c || !reach[other as usize].contains(&(c as usize)),
                        "edge {} -> {} is implied via {}", a, c, other
                    );
                }
            }
        }
    }
}
