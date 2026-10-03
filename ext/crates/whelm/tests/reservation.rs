//! Reservation edge cases: holder cancelled, worker loss and join, heartbeats.

use std::time::Duration;

use whelm::{
    Config, Constraint, Input, JobId, JobSpec, Output, Policy, Reservations, Resources, Scheduler,
    Time, WorkerId, WorkerState,
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

/// The first attempt of `job` finished.
fn done(job: JobId) -> Input {
    Input::Done { job, attempt: 1 }
}

/// A worker of class "x" with the given reported usage.
fn worker(id: u64, slots: usize, budget: u64, used: u64) -> WorkerState {
    WorkerState {
        id,
        class: "x".into(),
        slots,
        budget: Resources::mem(budget),
        reported_used: Resources::mem(used),
        ..Default::default()
    }
}

/// A job with the given demand and group.
fn job(id: u64, demand: u64, group: u64) -> JobSpec {
    JobSpec {
        id,
        demand: Resources::mem(demand),
        group,
        ..Default::default()
    }
}

/// Two workers (budget 100, 4 slots) each running one 60-unit job; a 50-unit job (group 0, the
/// most urgent) fits nowhere. After `reserve_after` it reserves a worker.
fn starving() -> Scheduler {
    let mut p = Scheduler::new(Config::default());
    p.handle(Input::Worker(worker(1, 4, 100, 0)), Time::ZERO);
    p.handle(Input::Worker(worker(2, 4, 100, 0)), Time::ZERO);
    p.handle(Input::Submit(job(10, 60, 1)), Time::ZERO);
    p.handle(Input::Submit(job(11, 60, 1)), Time::ZERO);
    assert_eq!(starts(p.poll(Time::ZERO)).len(), 2);
    p.handle(Input::Submit(job(1, 50, 0)), Time::from_secs(1));
    assert!(starts(p.poll(Time::from_secs(1))).is_empty());
    assert!(p.stats().reservations.is_empty(), "too early to reserve");
    assert!(starts(p.poll(Time::from_secs(61))).is_empty());
    let r = p.stats().reservations;
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].job, 1);
    p
}

/// The worker of the only reservation.
fn reserved_worker(p: &Scheduler) -> u64 {
    p.stats().reservations[0].worker
}

/// Others avoid the reserved worker; the holder gets it once it drains.
#[test]
fn reserved_worker_admits_only_its_holder() {
    let mut p = starving();
    let w = reserved_worker(&p);
    // A small job fits on both workers but must avoid the reserved one.
    p.handle(Input::Submit(job(20, 5, 2)), Time::from_secs(62));
    let out = starts(p.poll(Time::from_secs(62)));
    assert_eq!(out.len(), 1);
    assert_ne!(out[0].1, w);
    assert!(p.explain(1).unwrap().contains("holds the reservation"));
    // The reserved worker drains; the holder goes there by the escape hatch.
    let running_there = if w == 1 { 10 } else { 11 };
    p.handle(done(running_there), Time::from_secs(100));
    assert_eq!(starts(p.poll(Time::from_secs(100))), vec![(1, w)]);
    assert!(p.stats().reservations.is_empty());
    assert_eq!(p.stats().last_dispatch_holders, vec![1]);
}

/// Cancelling the holder frees its worker for everyone.
#[test]
fn holder_cancelled_releases_the_worker() {
    let mut p = starving();
    let w = reserved_worker(&p);
    p.handle(Input::Submit(job(20, 5, 2)), Time::from_secs(62));
    p.handle(Input::Submit(job(21, 5, 2)), Time::from_secs(62));
    p.handle(Input::Cancel(1), Time::from_secs(62));
    assert!(p.stats().reservations.is_empty());
    // Both workers take small jobs again (least loaded first: one on each).
    let out = starts(p.poll(Time::from_secs(62)));
    assert_eq!(out.len(), 2);
    assert!(out.iter().any(|&(_, x)| x == w));
}

/// Losing the reserved worker fails the job running there; its retry keeps its original place,
/// which is ahead of the holder, so it reserves the remaining worker and gets it as attempt 2.
#[test]
fn reserved_worker_leaves_and_its_lost_job_reserves_again() {
    let mut p = starving();
    let w = reserved_worker(&p);
    let (lost, other) = if w == 1 { (10, 2) } else { (11, 1) };
    let running_on_other = if other == 1 { 10 } else { 11 };
    p.handle(Input::WorkerGone(w), Time::from_secs(70));
    assert!(p.stats().reservations.is_empty());
    assert_eq!(
        p.stats().waiting,
        2,
        "the lost job is requeued, not forgotten"
    );
    // Both are admitted nowhere and old enough; the retry is the more urgent (older group).
    assert!(starts(p.poll(Time::from_secs(70))).is_empty());
    let r = p.stats().reservations;
    assert_eq!(r.len(), 1);
    assert_eq!((r[0].job, r[0].worker), (lost, other));
    assert!(
        p.explain(lost)
            .unwrap()
            .contains(&format!("failed 1 time(s), last on worker {w} (LinkDied"))
    );
    p.handle(done(running_on_other), Time::from_secs(100));
    assert_eq!(
        p.poll(Time::from_secs(100)),
        vec![Output::Start {
            job: lost,
            attempt: 2,
            worker: other
        }]
    );
    // The original holder reserves again once it qualifies.
    assert!(starts(p.poll(Time::from_secs(101))).is_empty());
    let r = p.stats().reservations;
    assert_eq!((r.len(), r[0].job), (1, 1));
}

/// A new empty worker takes the holder at once and ends the reservation.
#[test]
fn worker_joins_mid_reservation() {
    let mut p = starving();
    // A fresh, empty worker admits the holder at once (escape hatch); the reservation is released.
    p.handle(Input::Worker(worker(3, 4, 100, 0)), Time::from_secs(80));
    assert_eq!(starts(p.poll(Time::from_secs(80))), vec![(1, 3)]);
    assert!(p.stats().reservations.is_empty());
    assert!(p.stats().workers.iter().all(|w| w.reserved_for.is_none()));
}

/// A heartbeat making room admits the holder before its worker drains.
#[test]
fn heartbeat_lowering_usage_admits_the_holder_early() {
    let mut p = starving();
    let w = reserved_worker(&p);
    // Reported usage dominates the placed sum: 95 used, nothing fits.
    p.handle(Input::Worker(worker(w, 4, 100, 95)), Time::from_secs(90));
    assert!(starts(p.poll(Time::from_secs(90))).is_empty());
    // The heartbeat drops below what is placed (60): 60 + 50 > 100 still refuses ...
    p.handle(Input::Worker(worker(w, 4, 100, 10)), Time::from_secs(91));
    assert!(starts(p.poll(Time::from_secs(91))).is_empty());
    // ... until the budget grows (e.g. a recalibrated limit): 60 + 50 <= 120.
    p.handle(Input::Worker(worker(w, 4, 120, 10)), Time::from_secs(92));
    assert_eq!(starts(p.poll(Time::from_secs(92))), vec![(1, w)]);
    assert!(p.stats().reservations.is_empty());
}

/// Reported usage above the placed sum blocks admission until it drops.
#[test]
fn heartbeat_raising_usage_blocks_placements() {
    let mut p = Scheduler::new(Config::default());
    p.handle(Input::Worker(worker(1, 4, 100, 0)), Time::ZERO);
    p.handle(Input::Submit(job(1, 10, 0)), Time::ZERO);
    assert_eq!(starts(p.poll(Time::ZERO)).len(), 1);
    p.handle(Input::Worker(worker(1, 4, 100, 95)), Time::from_secs(1));
    p.handle(Input::Submit(job(2, 10, 0)), Time::from_secs(1));
    assert!(starts(p.poll(Time::from_secs(1))).is_empty());
    assert!(p.explain(2).unwrap().contains("memory short on 1"));
    p.handle(Input::Worker(worker(1, 4, 100, 20)), Time::from_secs(2));
    assert_eq!(starts(p.poll(Time::from_secs(2))), vec![(2, 1)]);
}

/// A holder placed on another worker releases its reservation.
#[test]
fn holder_placed_elsewhere_releases_its_reservation() {
    let mut p = starving();
    let w = reserved_worker(&p);
    let other = if w == 1 { 2 } else { 1 };
    let running_there = if other == 1 { 10 } else { 11 };
    p.handle(done(running_there), Time::from_secs(100));
    assert_eq!(starts(p.poll(Time::from_secs(100))), vec![(1, other)]);
    assert!(p.stats().reservations.is_empty());
    assert!(p.stats().last_dispatch_holders.is_empty());
}

/// Even an explicitly urgent, preferring job avoids a reserved worker.
#[test]
fn more_urgent_job_cannot_take_a_reserved_worker() {
    let mut p = starving();
    let w = reserved_worker(&p);
    // An explicitly urgent small job still may not use the reserved worker.
    let urgent = JobSpec {
        priority: Some(-10),
        constraints: vec![Constraint::prefer_worker(w)],
        ..job(30, 5, 9)
    };
    p.handle(Input::Submit(urgent), Time::from_secs(63));
    let out = starts(p.poll(Time::from_secs(63)));
    assert_eq!(out.len(), 1);
    assert_ne!(out[0].1, w);
}

/// Per-class limits allow one reservation per worker class.
#[test]
fn per_class_reservations() {
    let mut p = Scheduler::new(Config {
        reservations: Some(Reservations {
            per_class: true,
            ..Reservations::default()
        }),
        ..Config::best_fit()
    });
    for (id, class) in [(1, "a"), (2, "a"), (3, "b"), (4, "b")] {
        p.handle(
            Input::Worker(WorkerState {
                id,
                class: class.into(),
                slots: 4,
                budget: Resources::mem(100),
                ..Default::default()
            }),
            Time::ZERO,
        );
    }
    for i in 0..4 {
        p.handle(Input::Submit(job(10 + i, 60, 1)), Time::ZERO);
    }
    assert_eq!(starts(p.poll(Time::ZERO)).len(), 4);
    p.handle(Input::Submit(job(1, 50, 0)), Time::from_secs(1));
    p.handle(Input::Submit(job(2, 50, 0)), Time::from_secs(1));
    p.handle(Input::Submit(job(3, 50, 0)), Time::from_secs(1));
    assert!(starts(p.poll(Time::from_secs(100))).is_empty());
    let r = p.stats().reservations;
    assert_eq!(r.len(), 2, "one per class: {r:?}");
    assert_eq!(r.iter().map(|r| r.job).collect::<Vec<_>>(), vec![1, 2]);
}

/// [`starving`], with shadow-time backfill and known work: the two 60-unit jobs end at 100 s.
fn starving_shadow() -> Scheduler {
    let mut p = Scheduler::new(Config {
        reservations: Some(Reservations {
            shadow_backfill: true,
            ..Reservations::default()
        }),
        ..Config::default()
    });
    p.handle(Input::Worker(worker(1, 4, 100, 0)), Time::ZERO);
    p.handle(Input::Worker(worker(2, 4, 100, 0)), Time::ZERO);
    for id in [10, 11] {
        p.handle(
            Input::Submit(JobSpec {
                work: Some(Duration::from_secs(100)),
                ..job(id, 60, 1)
            }),
            Time::ZERO,
        );
    }
    assert_eq!(starts(p.poll(Time::ZERO)).len(), 2);
    p.handle(Input::Submit(job(1, 50, 0)), Time::from_secs(1));
    starts(p.poll(Time::from_secs(1)));
    assert!(starts(p.poll(Time::from_secs(61))).is_empty());
    assert_eq!(p.stats().reservations.len(), 1);
    // Fill the other worker, so only shadow backfill can place anything more.
    let w = p.stats().reservations[0].worker;
    p.handle(
        Input::Submit(JobSpec {
            work: Some(Duration::from_secs(1_000_000)),
            ..job(30, 40, 3)
        }),
        Time::from_secs(61),
    );
    assert_eq!(
        starts(p.poll(Time::from_secs(61))),
        vec![(30, if w == 1 { 2 } else { 1 })]
    );
    p
}

/// A short job backfills the reserved worker (it ends before the holder can start there); a long
/// one does not.
#[test]
fn shadow_backfill_admits_jobs_that_end_in_time() {
    let mut p = starving_shadow();
    let w = p.stats().reservations[0].worker;
    // 20 s of work ends at 81 s, before the shadow time 100 s; 50 s would end at 111 s.
    p.handle(
        Input::Submit(JobSpec {
            work: Some(Duration::from_secs(50)),
            ..job(21, 5, 2)
        }),
        Time::from_secs(62),
    );
    p.handle(
        Input::Submit(JobSpec {
            work: Some(Duration::from_secs(20)),
            ..job(20, 5, 2)
        }),
        Time::from_secs(62),
    );
    assert_eq!(starts(p.poll(Time::from_secs(62))), vec![(20, w)]);
    assert!(
        p.explain(21)
            .unwrap()
            .contains(&format!("worker {w} for job 1"))
    );
    // Unknown work never backfills.
    p.handle(Input::Submit(job(22, 5, 2)), Time::from_secs(63));
    assert!(starts(p.poll(Time::from_secs(63))).is_empty());
}

/// Once the shadow time has passed, the reserved worker drains strictly (overrunning jobs can no
/// longer delay the holder through new backfill).
#[test]
fn shadow_backfill_stops_at_the_shadow_time() {
    let mut p = starving_shadow();
    let w = p.stats().reservations[0].worker;
    // The shadow time is 100 s (the running jobs' expected end): a 1 s job at 95 s fits.
    p.handle(
        Input::Submit(JobSpec {
            work: Some(Duration::from_secs(1)),
            ..job(20, 1, 2)
        }),
        Time::from_secs(95),
    );
    assert_eq!(starts(p.poll(Time::from_secs(95))), vec![(20, w)]);
    p.handle(done(20), Time::from_secs(96));
    // The running jobs overrun; from 100 s on, nothing but the holder goes there.
    p.handle(
        Input::Submit(JobSpec {
            work: Some(Duration::from_millis(500)),
            ..job(21, 1, 2)
        }),
        Time::from_secs(101),
    );
    assert!(starts(p.poll(Time::from_secs(101))).is_empty());
    assert!(
        p.explain(21)
            .unwrap()
            .contains(&format!("worker {w} for job 1"))
    );
}
