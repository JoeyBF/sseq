//! Hard and soft avoid lists.

use sched::{Config, JobSpec, Policy, Resources, Scheduler, WorkerState};

/// A policy with the given workers (id, slots), all of class "x".
fn policy(workers: &[(u64, usize)]) -> Scheduler {
    let mut p = Scheduler::new(Config::default());
    for &(id, slots) in workers {
        p.worker_update(WorkerState::new(id, "x", slots, Resources::mem(100)), 0.0);
    }
    p
}

/// A job avoiding `avoid`.
fn job(id: u64, avoid: &[u64], soft: bool) -> JobSpec {
    JobSpec {
        avoid: avoid.to_vec(),
        avoid_soft: soft,
        ..JobSpec::new(id, Resources::mem(1), 0)
    }
}

/// The incident: every live worker is on the avoid list. Hard avoid waits; soft avoid runs.
#[test]
fn soft_avoid_lapses_when_only_avoided_workers_are_live() {
    let mut p = policy(&[(1, 4)]);
    p.submit(job(10, &[1], false), 0.0);
    p.submit(job(11, &[1], true), 0.0);
    assert_eq!(p.dispatch(0.0), vec![(11, 1)]);
    // A worker the job does not avoid joins: the hard one runs there.
    p.worker_update(WorkerState::new(2, "x", 4, Resources::mem(100)), 1.0);
    assert_eq!(p.dispatch(1.0), vec![(10, 2)]);
}

/// While a worker off the list is live, soft avoid holds even if that worker is busy: a retry
/// waits for a healthy worker instead of returning to the one it failed on.
#[test]
fn soft_avoid_holds_while_another_worker_is_live() {
    let mut p = policy(&[(1, 4), (2, 1)]);
    p.submit(JobSpec::new(0, Resources::mem(1), 0), 0.0);
    // Worker 2 has the most free... both admit; fill worker 2 explicitly via avoid.
    p.submit(job(1, &[1], false), 0.0);
    let placed = p.dispatch(0.0);
    assert!(placed.contains(&(1, 2)), "{placed:?}");
    // Worker 2 is now full; a soft-avoid job for worker 1 waits.
    p.submit(job(2, &[1], true), 1.0);
    let placed = p.dispatch(1.0);
    assert!(!placed.iter().any(|&(j, _)| j == 2), "{placed:?}");
    // Worker 2 frees: it goes there.
    p.completed(1, 2.0);
    assert_eq!(p.dispatch(2.0), vec![(2, 2)]);
}

/// A worker with no slots is not live, and neither is a different class.
#[test]
fn soft_avoid_ignores_dead_and_foreign_workers() {
    let mut p = policy(&[(1, 2), (2, 0)]);
    p.worker_update(WorkerState::new(3, "y", 2, Resources::mem(100)), 0.0);
    p.submit(
        JobSpec {
            class: Some("x".into()),
            ..job(5, &[1], true)
        },
        0.0,
    );
    assert_eq!(p.dispatch(0.0), vec![(5, 1)]);
}
