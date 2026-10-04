//! The DAG layer over plain jobs: readiness, references, cycles, cancellation, ranks, snapshots.

use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

use proptest::prelude::*;
use whelm::{
    Attempt, Config, DagConfig, DagError, DagJob, DagScheduler, Input, JobId, JobSpec, OrderTerm,
    Output, Policy, Resources, RetryConfig, Scheduler, Status, Time, WorkerState,
};

/// A DAG layer over backfill with one roomy worker.
fn dag(config: DagConfig) -> DagScheduler<Scheduler> {
    let mut d = DagScheduler::new(config, Scheduler::new(Config::default()));
    join(&mut d, worker(0, 64, 1000), Time::ORIGIN);
    d
}

/// Worker `id`, of class "x", with `slots` slots and `bytes` of memory.
fn worker(id: u64, slots: usize, bytes: u64) -> WorkerState {
    WorkerState {
        id,
        class: "x".into(),
        capacity: Resources::mem(bytes).with_slots(slots as u64),
        ..Default::default()
    }
}

/// A worker joins.
fn join(d: &mut DagScheduler<Scheduler>, w: WorkerState, now: Time) {
    d.handle(Input::Worker(w), now);
}

/// Report the first attempt of `job` done.
fn complete(d: &mut DagScheduler<Scheduler>, job: JobId, now: Time) {
    d.handle(Input::Done { job, attempt: 1 }, now);
}

/// A unit job in group 0 with the given dependencies.
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

/// The ids started by a poll, sorted.
fn placed(d: &mut DagScheduler<Scheduler>, now: Time) -> Vec<JobId> {
    let mut v: Vec<JobId> = d
        .poll(now)
        .into_iter()
        .filter_map(|o| match o {
            Output::Start { job, .. } => Some(job),
            _ => None,
        })
        .collect();
    v.sort_unstable();
    v
}

/// Each job of a chain is submitted only after its predecessor completes.
#[test]
fn chain_becomes_ready_in_order() {
    let mut d = dag(DagConfig::default());
    d.declare(vec![job(1, &[]), job(2, &[1]), job(3, &[2])], Time::ORIGIN)
        .unwrap();
    assert_eq!(placed(&mut d, Time::ORIGIN), vec![1]);
    let e = d.explain(3).unwrap();
    assert_eq!(
        e.status,
        Status::Pending {
            unit: 3,
            closed: false,
            unmet: vec![2]
        }
    );
    assert_eq!(e.to_string(), "job 3 waits for 1 dependency [2]");
    complete(&mut d, 1, Time(Duration::from_secs(1)));
    assert_eq!(placed(&mut d, Time(Duration::from_secs(1))), vec![2]);
    complete(&mut d, 2, Time(Duration::from_secs(2)));
    assert_eq!(placed(&mut d, Time(Duration::from_secs(2))), vec![3]);
    complete(&mut d, 3, Time(Duration::from_secs(3)));
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
    d.declare(vec![job(2, &[1])], Time::ORIGIN).unwrap();
    assert!(placed(&mut d, Time::ORIGIN).is_empty());
    assert_eq!(d.dag_stats().undeclared, 1);
    assert_eq!(
        d.explain(1).unwrap().status,
        Status::Undeclared { dependents: 1 }
    );
    d.declare(vec![job(1, &[])], Time(Duration::from_secs(1)))
        .unwrap();
    assert_eq!(placed(&mut d, Time(Duration::from_secs(1))), vec![1]);
    complete(&mut d, 1, Time(Duration::from_secs(2)));
    assert_eq!(placed(&mut d, Time(Duration::from_secs(2))), vec![2]);
}

/// Completed (and forgotten) jobs satisfy later dependencies.
#[test]
fn dependency_on_a_completed_job_is_satisfied() {
    let mut d = dag(DagConfig::default());
    d.declare(vec![job(1, &[])], Time::ORIGIN).unwrap();
    placed(&mut d, Time::ORIGIN);
    complete(&mut d, 1, Time(Duration::from_secs(1)));
    d.declare(
        vec![job(2, &[1]), job(3, &[1, 1])],
        Time(Duration::from_secs(2)),
    )
    .unwrap();
    assert_eq!(placed(&mut d, Time(Duration::from_secs(2))), vec![2, 3]);
    // After forgetting, everything below the floor still counts as completed.
    d.forget_completed_below(2);
    assert_eq!(d.dag_stats().completed_remembered, 0);
    d.declare(vec![job(4, &[1])], Time(Duration::from_secs(3)))
        .unwrap();
    assert_eq!(placed(&mut d, Time(Duration::from_secs(3))), vec![4]);
    assert_eq!(
        d.declare(vec![job(1, &[])], Time(Duration::from_secs(3))),
        Err(DagError::Duplicate(1))
    );
}

/// Self-loops, cycles through existing jobs and in-batch cycles are rejected atomically.
#[test]
fn cycles_are_rejected_without_a_trace() {
    let mut d = dag(DagConfig::default());
    assert_eq!(
        d.declare(vec![job(1, &[1])], Time::ORIGIN),
        Err(DagError::Cycle { job: 1 })
    );
    assert_eq!(d.dag_stats(), whelm::DagStats::default());

    d.declare(vec![job(1, &[2]), job(3, &[])], Time::ORIGIN)
        .unwrap();
    let before = d.dag_stats();
    // 2 -> 1 exists (1 depends on 2); declaring 2 depending on 1 closes the loop.
    assert!(matches!(
        d.declare(vec![job(2, &[1])], Time(Duration::from_secs(1))),
        Err(DagError::Cycle { .. })
    ));
    assert_eq!(d.dag_stats(), before);
    // A cycle entirely inside one batch, through a forward reference.
    assert!(matches!(
        d.declare(
            vec![job(10, &[12]), job(11, &[10]), job(12, &[11])],
            Time(Duration::from_secs(1))
        ),
        Err(DagError::Cycle { .. })
    ));
    assert_eq!(d.dag_stats(), before);
    // Duplicates within a batch.
    assert_eq!(
        d.declare(
            vec![job(20, &[]), job(20, &[])],
            Time(Duration::from_secs(1))
        ),
        Err(DagError::Duplicate(20))
    );
    // The graph still works: declare 2 properly.
    d.declare(vec![job(2, &[])], Time(Duration::from_secs(2)))
        .unwrap();
    assert_eq!(placed(&mut d, Time(Duration::from_secs(2))), vec![2, 3]);
    complete(&mut d, 2, Time(Duration::from_secs(3)));
    assert_eq!(placed(&mut d, Time(Duration::from_secs(3))), vec![1]);
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
        Time::ORIGIN,
    )
    .unwrap();
    assert_eq!(placed(&mut d, Time::ORIGIN), vec![1, 4]);
    let mut c = d.cancel(1);
    c.sort_unstable();
    assert_eq!(c, vec![1, 2, 3]);
    assert_eq!(d.stats().running, 1, "the policy released job 1");
    let stop = Output::Stop {
        job: 1,
        attempt: 1,
        worker: 0,
    };
    assert_eq!(d.poll(Time::ORIGIN), vec![stop], "and stopped its attempt");
    // Cancelling 5 drops the forward reference 9 it alone kept alive.
    assert_eq!(d.cancel(5), vec![5]);
    let s = d.dag_stats();
    assert_eq!((s.pending, s.undeclared, s.submitted), (0, 0, 1));
}

/// Upward ranks follow the longest descendant chain.
#[test]
fn ranks_follow_the_critical_path() {
    let mut d = dag(DagConfig::default());
    let w = |id, deps: &[JobId], work| DagJob {
        work_estimate: Some(work),
        ..job(id, deps)
    };
    // 1 -> 2 -> 3 and 1 -> 4: rank(1) = 1 + max(2 + 3, 10).
    d.declare(
        vec![
            w(1, &[], Duration::from_secs(1)),
            w(2, &[1], Duration::from_secs(2)),
            w(3, &[2], Duration::from_secs(3)),
        ],
        Time::ORIGIN,
    )
    .unwrap();
    assert_eq!(d.rank(1), Some(Duration::from_secs(6)));
    d.declare(vec![w(4, &[1], Duration::from_secs(10))], Time::ORIGIN)
        .unwrap();
    assert_eq!(d.rank(1), Some(Duration::from_secs(11)));
    assert_eq!(d.rank(2), Some(Duration::from_secs(5)));
    assert_eq!(d.rank(3), Some(Duration::from_secs(3)));
    assert_eq!(d.rank(99), None);
}

/// With [`OrderTerm::Rank`] in the order, the job heading the longer chain runs first; without
/// it, ranks are carried but order nothing.
#[test]
fn the_rank_term_orders_ready_jobs() {
    for (order, first) in [
        (
            vec![OrderTerm::Priority, OrderTerm::Rank, OrderTerm::Group],
            2,
        ),
        (Config::default().order, 1),
    ] {
        // One slot: only the first job in order is placed.
        let policy = Scheduler::new(Config {
            order,
            ..Config::default()
        });
        let mut d = DagScheduler::new(DagConfig::default(), policy);
        join(&mut d, worker(0, 1, 1000), Time::ORIGIN);
        let w = |id, deps: &[JobId], work| DagJob {
            work_estimate: Some(work),
            ..job(id, deps)
        };
        d.declare(
            vec![
                w(1, &[], Duration::from_secs(1)),
                w(2, &[], Duration::from_secs(1)),
                w(3, &[2], Duration::from_secs(50)),
            ],
            Time::ORIGIN,
        )
        .unwrap();
        assert_eq!(placed(&mut d, Time::ORIGIN), vec![first]);
    }
}

/// Without auto-submit, ready jobs are announced and wait for `release`.
#[test]
fn held_jobs_wait_for_release() {
    let mut d = dag(DagConfig {
        auto_submit: false,
        ..DagConfig::default()
    });
    d.declare(vec![job(1, &[]), job(2, &[1])], Time::ORIGIN)
        .unwrap();
    assert_eq!(d.poll(Time::ORIGIN), vec![Output::Ready { job: 1 }]);
    assert!(d.poll(Time::ORIGIN).is_empty(), "announced once");
    assert_eq!(d.explain(1).unwrap().status, Status::Held);
    assert!(d.release(1, Time(Duration::from_secs(5))));
    assert!(!d.release(1, Time(Duration::from_secs(5))));
    assert_eq!(placed(&mut d, Time(Duration::from_secs(5))), vec![1]);
    complete(&mut d, 1, Time(Duration::from_secs(6)));
    assert_eq!(
        d.poll(Time(Duration::from_secs(6))),
        vec![Output::Ready { job: 2 }]
    );
}

/// A job lost with its worker is retried by the policy (attempt 2, on the next worker), and a
/// late report of the lost attempt completes nothing.
#[test]
fn worker_loss_retries_automatically() {
    let mut d = dag(DagConfig::default());
    d.declare(vec![job(1, &[]), job(2, &[1])], Time::ORIGIN)
        .unwrap();
    assert_eq!(placed(&mut d, Time::ORIGIN), vec![1]);
    d.handle(Input::WorkerGone(0), Time(Duration::from_secs(1)));
    assert!(d.poll(Time(Duration::from_secs(1))).is_empty());
    complete(&mut d, 1, Time(Duration::from_millis(1500)));
    assert!(
        d.poll(Time(Duration::from_millis(1500))).is_empty(),
        "the lost attempt's report is stale"
    );
    join(&mut d, worker(5, 1, 10), Time(Duration::from_secs(2)));
    let retry = Output::Start {
        job: 1,
        attempt: 2,
        worker: 5,
    };
    assert_eq!(d.poll(Time(Duration::from_secs(2))), vec![retry]);
    d.handle(
        Input::Done { job: 1, attempt: 2 },
        Time(Duration::from_secs(3)),
    );
    assert_eq!(placed(&mut d, Time(Duration::from_secs(3))), vec![2]);
}

/// A snapshot survives JSON and resumes with a fresh policy.
#[cfg(feature = "serde")]
#[test]
fn snapshot_round_trip() {
    let mut d = dag(DagConfig::default());
    d.declare(
        vec![job(1, &[]), job(2, &[1]), job(3, &[2, 7])],
        Time::ORIGIN,
    )
    .unwrap();
    placed(&mut d, Time::ORIGIN);
    let json = serde_json::to_string(&d.snapshot()).unwrap();
    let snap = serde_json::from_str(&json).unwrap();
    let mut r = DagScheduler::restore(
        snap,
        Scheduler::new(Config::default()),
        None,
        Time(Duration::from_secs(10)),
    );
    join(&mut r, worker(0, 64, 1000), Time(Duration::from_secs(10)));
    assert_eq!(r.dag_stats(), d.dag_stats());
    // Job 1 was submitted before the snapshot: it is submitted again to the new policy.
    assert_eq!(placed(&mut r, Time(Duration::from_secs(10))), vec![1]);
    complete(&mut r, 1, Time(Duration::from_secs(11)));
    assert_eq!(placed(&mut r, Time(Duration::from_secs(11))), vec![2]);
    complete(&mut r, 2, Time(Duration::from_secs(12)));
    assert!(
        placed(&mut r, Time(Duration::from_secs(12))).is_empty(),
        "3 still waits for the forward reference 7"
    );
    r.declare(vec![job(7, &[])], Time(Duration::from_secs(13)))
        .unwrap();
    assert_eq!(placed(&mut r, Time(Duration::from_secs(13))), vec![7]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// A random DAG (edges from lower to higher ids), declared in random batches in random order
    /// (so forward references abound), with random worker churn: no job is dispatched before its
    /// dependencies completed (late reports of lost attempts complete nothing), every job
    /// eventually runs, and an injected back edge is rejected.
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
        churn in prop::collection::btree_set(0usize..200, 0..6),
    ) {
        let mut deps: BTreeMap<JobId, BTreeSet<JobId>> = (0..n as JobId).map(|i| (i, BTreeSet::new())).collect();
        for (a, b) in edges {
            let (a, b) = (a.min(b) as JobId, a.max(b) as JobId);
            if a != b && (b as usize) < n {
                deps.get_mut(&b).unwrap().insert(a);
            }
        }
        // Enough attempts that churn never makes the policy give up.
        let config = Config { retry: RetryConfig { max_attempts: 100 }, ..Config::default() };
        let mut d = DagScheduler::new(DagConfig::default(), Scheduler::new(config));
        join(&mut d, worker(0, slots, 10), Time::ORIGIN);
        let ids: Vec<JobId> = order.into_iter().filter(|&i| i < n).map(|i| i as JobId).collect();
        let mut done: BTreeSet<JobId> = BTreeSet::new();
        let mut running: Vec<(JobId, Attempt)> = Vec::new();
        let mut t = Time::ORIGIN;
        let mut steps = 0;
        let mut step = |d: &mut DagScheduler<Scheduler>,
                        done: &mut BTreeSet<JobId>,
                        running: &mut Vec<(JobId, Attempt)>,
                        t: &mut Time|
         -> Result<(), TestCaseError> {
            *t += Duration::from_secs(1);
            if churn.contains(&steps) {
                // The worker leaves with its attempts, whose late reports must complete nothing,
                // and comes back.
                d.handle(Input::WorkerGone(0), *t);
                for (job, attempt) in running.drain(..) {
                    d.handle(Input::Done { job, attempt }, *t);
                }
                join(d, worker(0, slots, 10), *t);
            }
            steps += 1;
            for o in d.poll(*t) {
                if let Output::Start { job, attempt, .. } = o {
                    prop_assert!(deps[&job].is_subset(done), "job {} dispatched before its deps", job);
                    prop_assert!(!done.contains(&job), "job {} dispatched after completing", job);
                    running.push((job, attempt));
                }
            }
            if !running.is_empty() {
                let (job, attempt) = running.remove(0);
                d.handle(Input::Done { job, attempt }, *t);
                done.insert(job);
            }
            Ok(())
        };
        for chunk in ids.chunks(batch) {
            let jobs: Vec<DagJob> = chunk.iter().map(|&i| job(i, &deps[&i].iter().copied().collect::<Vec<_>>())).collect();
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
