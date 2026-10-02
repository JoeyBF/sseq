//! The DAG layer: readiness, forward references, cycles, cancellation, ranks, snapshots.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use sched::{
    BackfillConfig, Dag, DagConfig, DagError, DagJob, DagScheduler, JobId, JobSpec,
    PriorityBackfill, Resources, WorkerState,
};

/// A DAG layer over backfill with one roomy worker.
fn dag(config: DagConfig) -> DagScheduler<PriorityBackfill> {
    let mut d = DagScheduler::new(config, PriorityBackfill::new(BackfillConfig::default()));
    d.worker_update(WorkerState::new(0, "x", 64, Resources::mem(1000)), 0.0);
    d
}

/// A unit job in group 0 with the given dependencies.
fn job(id: JobId, deps: &[JobId]) -> DagJob {
    DagJob {
        spec: JobSpec::new(id, Resources::mem(1), 0),
        deps: deps.to_vec(),
        work_estimate: None,
    }
}

/// The ids placed by a dispatch, sorted.
fn placed(d: &mut DagScheduler<PriorityBackfill>, now: f64) -> Vec<JobId> {
    let mut v: Vec<JobId> = d.dispatch(now).into_iter().map(|p| p.0).collect();
    v.sort_unstable();
    v
}

/// Each job of a chain is submitted only after its predecessor completes.
#[test]
fn chain_becomes_ready_in_order() {
    let mut d = dag(DagConfig::default());
    d.declare(vec![job(1, &[]), job(2, &[1]), job(3, &[2])], 0.0)
        .unwrap();
    assert_eq!(placed(&mut d, 0.0), vec![1]);
    assert!(d.explain(3).unwrap().contains("waits for 1 dependency [2]"));
    d.completed(1, 1.0);
    assert_eq!(placed(&mut d, 1.0), vec![2]);
    d.completed(2, 2.0);
    assert_eq!(placed(&mut d, 2.0), vec![3]);
    d.completed(3, 3.0);
    let s = d.dag_stats();
    assert_eq!(
        (s.pending, s.submitted, s.edges, s.undeclared),
        (0, 0, 0, 0)
    );
}

/// A dependency on an undeclared job blocks until it is declared and completes.
#[test]
fn forward_references_wait_for_declaration() {
    let mut d = dag(DagConfig::default());
    d.declare(vec![job(2, &[1])], 0.0).unwrap();
    assert!(placed(&mut d, 0.0).is_empty());
    assert_eq!(d.dag_stats().undeclared, 1);
    assert!(d.explain(1).unwrap().contains("not declared yet"));
    d.declare(vec![job(1, &[])], 1.0).unwrap();
    assert_eq!(placed(&mut d, 1.0), vec![1]);
    d.completed(1, 2.0);
    assert_eq!(placed(&mut d, 2.0), vec![2]);
}

/// Completed (and forgotten) jobs satisfy later dependencies.
#[test]
fn dependency_on_a_completed_job_is_satisfied() {
    let mut d = dag(DagConfig::default());
    d.declare(vec![job(1, &[])], 0.0).unwrap();
    placed(&mut d, 0.0);
    d.completed(1, 1.0);
    d.declare(vec![job(2, &[1]), job(3, &[1, 1])], 2.0).unwrap();
    assert_eq!(placed(&mut d, 2.0), vec![2, 3]);
    // After forgetting, everything below the floor still counts as completed.
    d.forget_completed_below(2);
    assert_eq!(d.dag_stats().completed_remembered, 0);
    d.declare(vec![job(4, &[1])], 3.0).unwrap();
    assert_eq!(placed(&mut d, 3.0), vec![4]);
    assert_eq!(
        d.declare(vec![job(1, &[])], 3.0),
        Err(DagError::Duplicate(1))
    );
}

/// Self-loops, cycles through existing jobs and in-batch cycles are rejected atomically.
#[test]
fn cycles_are_rejected_without_a_trace() {
    let mut d = dag(DagConfig::default());
    assert_eq!(
        d.declare(vec![job(1, &[1])], 0.0),
        Err(DagError::Cycle { job: 1 })
    );
    assert_eq!(d.dag_stats(), sched::DagStats::default());

    d.declare(vec![job(1, &[2]), job(3, &[])], 0.0).unwrap();
    let before = d.dag_stats();
    // 2 -> 1 exists (1 depends on 2); declaring 2 depending on 1 closes the loop.
    assert!(matches!(
        d.declare(vec![job(2, &[1])], 1.0),
        Err(DagError::Cycle { .. })
    ));
    assert_eq!(d.dag_stats(), before);
    // A cycle entirely inside one batch, through a forward reference.
    assert!(matches!(
        d.declare(vec![job(10, &[12]), job(11, &[10]), job(12, &[11])], 1.0),
        Err(DagError::Cycle { .. })
    ));
    assert_eq!(d.dag_stats(), before);
    // Duplicates within a batch.
    assert_eq!(
        d.declare(vec![job(20, &[]), job(20, &[])], 1.0),
        Err(DagError::Duplicate(20))
    );
    // The graph still works: declare 2 properly.
    d.declare(vec![job(2, &[])], 2.0).unwrap();
    assert_eq!(placed(&mut d, 2.0), vec![2, 3]);
    d.completed(2, 3.0);
    assert_eq!(placed(&mut d, 3.0), vec![1]);
}

/// Cancelling removes all dependents and the forward references they alone held.
#[test]
fn cancel_cascades_to_dependents() {
    let mut d = dag(DagConfig::default());
    d.declare(
        vec![
            job(1, &[]),
            job(2, &[1]),
            job(3, &[2]),
            job(4, &[]),
            job(5, &[9]),
        ],
        0.0,
    )
    .unwrap();
    assert_eq!(placed(&mut d, 0.0), vec![1, 4]);
    let mut c = d.cancel(1);
    c.sort_unstable();
    assert_eq!(c, vec![1, 2, 3]);
    assert_eq!(d.stats().running, 1, "the policy released job 1");
    // Cancelling 5 drops the forward reference 9 it alone kept alive.
    assert_eq!(d.cancel(5), vec![5]);
    let s = d.dag_stats();
    assert_eq!((s.pending, s.undeclared, s.submitted), (0, 0, 1));
}

/// Upward ranks follow the longest descendant chain, plus placeholder tails.
#[test]
fn ranks_follow_the_critical_path() {
    let mut d = dag(DagConfig {
        rank_priority: true,
        rank_scale: 1.0,
        ..DagConfig::default()
    });
    let w = |id, deps: &[JobId], work| DagJob {
        work_estimate: Some(work),
        ..job(id, deps)
    };
    // 1 -> 2 -> 3 and 1 -> 4: rank(1) = 1 + max(2 + 3, 10).
    d.declare(vec![w(1, &[], 1.0), w(2, &[1], 2.0), w(3, &[2], 3.0)], 0.0)
        .unwrap();
    assert_eq!(d.rank(1), Some(6.0));
    d.declare(vec![w(4, &[1], 10.0)], 0.0).unwrap();
    assert_eq!(d.rank(1), Some(11.0));
    assert_eq!(d.rank(2), Some(5.0));
    // A placeholder group waiting on group 0 adds its cost below every job of group 0.
    d.declare_group_placeholder(7, vec![0], 100.0);
    assert_eq!(d.rank(3), Some(103.0));
    d.remove_group_placeholder(7);
    assert_eq!(d.rank(3), Some(3.0));
}

/// With rank priority, the job heading the longer chain runs first.
#[test]
fn rank_priority_orders_ready_jobs() {
    // One slot: the ready job with the longer chain below it goes first.
    let mut d = DagScheduler::new(
        DagConfig {
            rank_priority: true,
            ..DagConfig::default()
        },
        PriorityBackfill::new(BackfillConfig::default()),
    );
    d.worker_update(WorkerState::new(0, "x", 1, Resources::mem(1000)), 0.0);
    let w = |id, deps: &[JobId], work| DagJob {
        work_estimate: Some(work),
        ..job(id, deps)
    };
    d.declare(vec![w(1, &[], 1.0), w(2, &[], 1.0), w(3, &[2], 50.0)], 0.0)
        .unwrap();
    assert_eq!(placed(&mut d, 0.0), vec![2]);
}

/// Without auto-submit, ready jobs wait for `release`.
#[test]
fn held_jobs_wait_for_release() {
    let mut d = dag(DagConfig {
        auto_submit: false,
        ..DagConfig::default()
    });
    d.declare(vec![job(1, &[]), job(2, &[1])], 0.0).unwrap();
    assert_eq!(d.take_ready(), vec![1]);
    assert!(placed(&mut d, 0.0).is_empty());
    assert!(d.explain(1).unwrap().contains("held"));
    assert!(d.release(1, 5.0));
    assert!(!d.release(1, 5.0));
    assert_eq!(placed(&mut d, 5.0), vec![1]);
    d.completed(1, 6.0);
    assert_eq!(d.take_ready(), vec![2]);
}

/// A job lost with its worker can be resubmitted from the DAG.
#[test]
fn resubmit_after_worker_loss() {
    let mut d = dag(DagConfig::default());
    d.declare(vec![job(1, &[])], 0.0).unwrap();
    assert_eq!(placed(&mut d, 0.0), vec![1]);
    d.worker_gone(0, 1.0);
    assert!(d.resubmit(1, 1.0));
    d.worker_update(WorkerState::new(5, "x", 1, Resources::mem(10)), 2.0);
    assert_eq!(d.dispatch(2.0), vec![(1, 5)]);
}

/// A snapshot survives JSON and resumes with a fresh policy.
#[test]
fn snapshot_round_trip() {
    let mut d = dag(DagConfig::default());
    d.declare(vec![job(1, &[]), job(2, &[1]), job(3, &[2, 7])], 0.0)
        .unwrap();
    d.declare_group_placeholder(9, vec![0], 4.0);
    placed(&mut d, 0.0);
    let json = serde_json::to_string(&d.snapshot()).unwrap();
    let snap = serde_json::from_str(&json).unwrap();
    let mut r = DagScheduler::restore(snap, PriorityBackfill::new(BackfillConfig::default()), 10.0);
    r.worker_update(WorkerState::new(0, "x", 64, Resources::mem(1000)), 10.0);
    assert_eq!(r.dag_stats(), d.dag_stats());
    // Job 1 was submitted before the snapshot: it is submitted again to the new policy.
    assert_eq!(placed(&mut r, 10.0), vec![1]);
    r.completed(1, 11.0);
    assert_eq!(placed(&mut r, 11.0), vec![2]);
    r.completed(2, 12.0);
    assert!(
        placed(&mut r, 12.0).is_empty(),
        "3 still waits for the forward reference 7"
    );
    r.declare(vec![job(7, &[])], 13.0).unwrap();
    assert_eq!(placed(&mut r, 13.0), vec![7]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// A random DAG (edges from lower to higher ids), declared in random batches in random order
    /// (so forward references abound), with random worker churn: no job is dispatched before its
    /// dependencies completed, every job eventually runs, and an injected back edge is rejected.
    #[test]
    fn never_dispatched_before_dependencies(
        n in 1usize..40,
        edges in prop::collection::vec((0usize..40, 0usize..40), 0..120),
        order in Just(()).prop_perturb(|_, mut rng| {
            let mut v: Vec<usize> = (0..40).collect();
            for i in (1..v.len()).rev() { v.swap(i, rng.random_range(0..=i)); }
            v
        }),
        batch in 1usize..8,
        slots in 1usize..4,
    ) {
        let mut deps: BTreeMap<JobId, BTreeSet<JobId>> = (0..n as JobId).map(|i| (i, BTreeSet::new())).collect();
        for (a, b) in edges {
            let (a, b) = (a.min(b) as JobId, a.max(b) as JobId);
            if a != b && (b as usize) < n {
                deps.get_mut(&b).unwrap().insert(a);
            }
        }
        let mut d = DagScheduler::new(DagConfig::default(), PriorityBackfill::new(BackfillConfig::default()));
        d.worker_update(WorkerState::new(0, "x", slots, Resources::mem(10)), 0.0);
        let ids: Vec<JobId> = order.into_iter().filter(|&i| i < n).map(|i| i as JobId).collect();
        let mut done: BTreeSet<JobId> = BTreeSet::new();
        let mut running: Vec<JobId> = Vec::new();
        let mut t = 0.0;
        let step = |d: &mut DagScheduler<PriorityBackfill>,
                    done: &mut BTreeSet<JobId>,
                    running: &mut Vec<JobId>,
                    t: &mut f64|
         -> Result<(), TestCaseError> {
            *t += 1.0;
            for (j, _) in d.dispatch(*t) {
                prop_assert!(deps[&j].is_subset(done), "job {} dispatched before its deps", j);
                running.push(j);
            }
            if let Some(j) = running.first().copied() {
                running.remove(0);
                d.completed(j, *t);
                done.insert(j);
            }
            Ok(())
        };
        for chunk in ids.chunks(batch) {
            let jobs = chunk.iter().map(|&i| job(i, &deps[&i].iter().copied().collect::<Vec<_>>())).collect();
            d.declare(jobs, t).unwrap();
            step(&mut d, &mut done, &mut running, &mut t)?;
        }
        // A batch hanging off a live job and closing a loop among its new jobs is rejected, and
        // leaves the graph able to finish.
        if let Some(&b) = deps.keys().rev().find(|b| !done.contains(b)) {
            d.declare(vec![job(1000, &[b])], t).unwrap();
            let r = d.declare(vec![job(1001, &[1000, 1002]), job(1002, &[1001])], t);
            prop_assert!(matches!(r, Err(DagError::Cycle { .. })), "{:?}", r);
            prop_assert_eq!(d.cancel(1000), vec![1000]);
        }
        for _ in 0..(4 * n + 10) {
            step(&mut d, &mut done, &mut running, &mut t)?;
        }
        prop_assert_eq!(done.len(), n, "not all jobs ran: {:?}", d.dag_stats());
    }
}
