//! Forbid and Avoid constraints, and the soft avoidance of retries.

use whelm::{
    Config, Constraint, FailKind, Input, JobId, JobSpec, Output, Policy, Resources, Scheduler,
    WorkerId, WorkerState,
};

/// The `(job, worker)` of each start in `out`.
fn starts(out: Vec<Output>) -> Vec<(JobId, WorkerId)> {
    out.into_iter()
        .filter_map(|o| match o {
            Output::Start { job, worker, .. } => Some((job, worker)),
            _ => None,
        })
        .collect()
}

/// A worker of class "x" with the given slots.
fn worker(id: WorkerId, slots: usize) -> WorkerState {
    WorkerState {
        id,
        class: "x".into(),
        slots,
        budget: Resources::mem(100),
        ..Default::default()
    }
}

/// A policy with the given workers (id, slots), all of class "x".
fn policy(workers: &[(WorkerId, usize)]) -> Scheduler {
    let mut p = Scheduler::new(Config::default());
    for &(id, slots) in workers {
        p.handle(Input::Worker(worker(id, slots)), 0.0);
    }
    p
}

/// A job that avoids (`soft`) or forbids every worker of `avoid`.
fn job(id: JobId, avoid: &[WorkerId], soft: bool) -> JobSpec {
    let constraint = if soft {
        Constraint::avoid_worker
    } else {
        Constraint::forbid_worker
    };
    JobSpec {
        id,
        demand: Resources::mem(1),
        constraints: avoid.iter().map(|&w| constraint(w)).collect(),
        ..Default::default()
    }
}

/// Every live worker is excluded. Forbid waits; Avoid runs.
#[test]
fn soft_avoid_lapses_when_only_avoided_workers_are_live() {
    let mut p = policy(&[(1, 4)]);
    p.handle(Input::Submit(job(10, &[1], false)), 0.0);
    p.handle(Input::Submit(job(11, &[1], true)), 0.0);
    assert_eq!(starts(p.poll(0.0)), vec![(11, 1)]);
    // A worker the job does not avoid joins: the hard one runs there.
    p.handle(Input::Worker(worker(2, 4)), 1.0);
    assert_eq!(starts(p.poll(1.0)), vec![(10, 2)]);
}

/// While a worker the job does not avoid is live, Avoid holds even if that worker is busy: a
/// retry waits for a healthy worker instead of returning to the one it failed on.
#[test]
fn soft_avoid_holds_while_another_worker_is_live() {
    let mut p = policy(&[(1, 4), (2, 1)]);
    p.handle(Input::Submit(job(0, &[], false)), 0.0);
    // Fill worker 2 explicitly by forbidding worker 1.
    p.handle(Input::Submit(job(1, &[1], false)), 0.0);
    let placed = starts(p.poll(0.0));
    assert!(placed.contains(&(1, 2)), "{placed:?}");
    // Worker 2 is now full; a job avoiding worker 1 waits.
    p.handle(Input::Submit(job(2, &[1], true)), 1.0);
    let placed = starts(p.poll(1.0));
    assert!(!placed.iter().any(|&(j, _)| j == 2), "{placed:?}");
    // Worker 2 frees: it goes there.
    p.handle(Input::Done { job: 1, attempt: 1 }, 2.0);
    assert_eq!(starts(p.poll(2.0)), vec![(2, 2)]);
}

/// A worker with no slots is not live, and one the hard constraints exclude does not count.
#[test]
fn soft_avoid_ignores_dead_and_foreign_workers() {
    let mut p = policy(&[(1, 2), (2, 0)]);
    p.handle(
        Input::Worker(WorkerState {
            class: "y".into(),
            ..worker(3, 2)
        }),
        0.0,
    );
    let mut pinned = job(5, &[1], true);
    pinned.constraints.push(Constraint::require_class("x"));
    p.handle(Input::Submit(pinned), 0.0);
    assert_eq!(starts(p.poll(0.0)), vec![(5, 1)]);
}

/// A failed attempt's retry softly avoids the worker it failed on: it waits for the busy healthy
/// worker, and returns to the failing one only once that is the only live worker left.
#[test]
fn retry_softly_avoids_the_worker_it_failed_on() {
    let mut p = policy(&[(1, 1), (2, 1)]);
    p.handle(Input::Submit(job(0, &[], false)), 0.0);
    p.handle(Input::Submit(job(1, &[], false)), 0.0);
    assert_eq!(starts(p.poll(0.0)), vec![(0, 1), (1, 2)]);
    p.handle(
        Input::Failed {
            job: 0,
            attempt: 1,
            kind: FailKind::Other,
            why: "boom".into(),
        },
        1.0,
    );
    // Worker 1 is free but avoided while worker 2 lives.
    assert_eq!(starts(p.poll(1.0)), vec![]);
    assert!(
        p.explain(0)
            .unwrap()
            .contains("failed 1 time(s), last on worker 1")
    );
    // Worker 2 leaves: its job fails too, and both retries fall back to worker 1, the only one.
    p.handle(Input::WorkerGone(2), 2.0);
    let out = p.poll(2.0);
    assert_eq!(
        out,
        vec![Output::Start {
            job: 0,
            attempt: 2,
            worker: 1
        }]
    );
    p.handle(Input::Done { job: 0, attempt: 2 }, 3.0);
    assert_eq!(
        p.poll(3.0),
        vec![Output::Start {
            job: 1,
            attempt: 2,
            worker: 1
        }]
    );
}
