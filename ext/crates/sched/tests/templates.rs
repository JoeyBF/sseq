//! Passthrough jobs, work updates, templates, and the avoid/class constraints.

use std::collections::BTreeMap;

use proptest::prelude::*;
use sched::{
    Config, Dag, DagConfig, DagError, DagJob, DagScheduler, DagTemplate, JobId, JobSpec, Policy,
    Resources, Scheduler, WorkerState,
};

/// A DAG layer over one 64-slot worker.
fn dag(config: DagConfig) -> DagScheduler<Scheduler> {
    let mut d = DagScheduler::new(config, Scheduler::new(Config::default()));
    d.worker_update(WorkerState::new(0, "x", 64, Resources::mem(1000)), 0.0);
    d
}

/// A unit job in group 0.
fn job(id: JobId, deps: &[JobId]) -> DagJob {
    DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps.to_vec())
}

/// The ids placed by a dispatch, sorted.
fn placed(d: &mut DagScheduler<Scheduler>, now: f64) -> Vec<JobId> {
    let mut v: Vec<JobId> = d.dispatch(now).into_iter().map(|p| p.0).collect();
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
        vec![
            job(1, &[]),
            DagJob::passthrough(2, 0, vec![1], 5.0),
            job(3, &[2]),
        ],
        0.0,
    )
    .unwrap();
    assert_eq!(placed(&mut d, 0.0), vec![1]);
    assert_eq!(
        d.rank(1),
        Some(7.0),
        "the passthrough's work counts in ranks"
    );
    d.completed(1, 1.0);
    assert_eq!(d.take_passed(), vec![2]);
    assert_eq!(placed(&mut d, 1.0), vec![3]);
    assert_eq!(d.stats().placements_total, 2);
}

/// A 200,000-long chain of passthroughs completes without recursing.
#[test]
fn long_passthrough_chains_do_not_recurse() {
    let mut d = dag(DagConfig::default());
    const N: u64 = 200_000;
    let mut jobs = vec![job(0, &[])];
    jobs.extend((1..N).map(|i| DagJob::passthrough(i, 0, vec![i - 1], 0.0)));
    jobs.push(job(N, &[N - 1]));
    d.declare(jobs, 0.0).unwrap();
    assert_eq!(placed(&mut d, 0.0), vec![0]);
    d.completed(0, 1.0);
    assert_eq!(placed(&mut d, 1.0), vec![N]);
    assert_eq!(d.dag_stats().pending, 0);
}

/// A passthrough with no dependencies completes during `declare`.
#[test]
fn a_ready_passthrough_completes_at_declaration() {
    let mut d = dag(DagConfig::default());
    d.declare(
        vec![DagJob::passthrough(1, 0, vec![], 0.0), job(2, &[1])],
        0.0,
    )
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
    let w = |id, deps: &[JobId], work| job(id, deps).with_work(work);
    d.declare(
        vec![
            w(1, &[], 1.0),
            w(2, &[1], 10.0),
            w(3, &[1], 2.0),
            w(4, &[2, 3], 1.0),
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
        DagTemplate::new(3, [(0, 1), (1, 2), (2, 0)]).unwrap_err(),
        DagError::Cycle { job: 0 }
    );
    let t = DagTemplate::new(4, [(0, 1), (0, 2), (1, 3), (2, 3), (0, 1)]).unwrap();
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
    let t = DagTemplate::new(5, [(0, 1), (0, 2), (1, 3), (2, 3)]).unwrap();
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

/// Two instances of a template chain through their entry dependencies.
#[test]
fn templates_instantiate_per_group() {
    // A diamond, instantiated twice; group 2's sources wait for group 1's sink.
    let t = DagTemplate::new(4, [(0, 1), (0, 2), (1, 3), (2, 3)]).unwrap();
    let mut d = dag(DagConfig::default());
    for g in 1..=2u64 {
        let entry: Vec<JobId> = if g == 2 { vec![13] } else { vec![] };
        d.declare_template(
            &t,
            |i| g * 10 + i as JobId,
            |i| {
                if i == 2 {
                    DagJob::passthrough(0, g, vec![], 0.0)
                } else {
                    DagJob::new(JobSpec::new(0, Resources::mem(1), g), vec![])
                }
            },
            &entry,
            0.0,
        )
        .unwrap();
    }
    let mut order = Vec::new();
    for step in 0..10 {
        let now = step as f64;
        let out = placed(&mut d, now);
        for &j in &out {
            d.completed(j, now + 0.5);
        }
        order.push(out);
    }
    let flat: Vec<JobId> = order.into_iter().flatten().collect();
    assert_eq!(flat, vec![10, 11, 13, 20, 21, 23]);
}

/// Avoid lists and classes exclude workers; a job excluded everywhere waits and says so.
#[test]
fn avoid_and_class_are_hard_constraints() {
    let mut p = Scheduler::new(Config::default());
    p.worker_update(WorkerState::new(1, "h200", 4, Resources::mem(100)), 0.0);
    p.worker_update(WorkerState::new(2, "l40s", 4, Resources::mem(100)), 0.0);
    let mut retry = JobSpec::new(1, Resources::mem(1), 0);
    retry.avoid = vec![1];
    retry.prefer = vec![1];
    p.submit(retry, 0.0);
    let mut pinned = JobSpec::new(2, Resources::mem(1), 0);
    pinned.class = Some("h200".into());
    p.submit(pinned, 0.0);
    assert_eq!(p.dispatch(0.0), vec![(1, 2), (2, 1)]);
    // A job excluded everywhere waits, and says why.
    let mut nowhere = JobSpec::new(3, Resources::mem(1), 0);
    nowhere.class = Some("v100".into());
    p.submit(nowhere, 1.0);
    assert!(p.dispatch(1.0).is_empty());
    assert!(p.explain(3).unwrap().contains("2 worker(s) excluded"));
    p.worker_update(WorkerState::new(3, "v100", 1, Resources::mem(10)), 2.0);
    assert_eq!(p.dispatch(2.0), vec![(3, 3)]);
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
            let jobs = chunk.iter().map(|&i| job(i, &deps[&i]).with_work(work[&i])).collect();
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
        let t = DagTemplate::new(n, edges).unwrap();
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
