//! Reservation edge cases: holder cancelled, worker loss and join, heartbeats.

use sched::{
    BackfillConfig, BestFit, BestFitConfig, JobSpec, Policy, PriorityBackfill, Resources,
    WorkerState,
};

/// A worker of class "x" with the given reported usage.
fn worker(id: u64, slots: usize, budget: u64, used: u64) -> WorkerState {
    WorkerState {
        reported_used: Resources::mem(used),
        ..WorkerState::new(id, "x", slots, Resources::mem(budget))
    }
}

/// A job with the given demand and group.
fn job(id: u64, demand: u64, group: u64) -> JobSpec {
    JobSpec::new(id, Resources::mem(demand), group)
}

/// Two workers (budget 100, 4 slots) each running one 60-unit job; a 50-unit job (group 0, the
/// most urgent) fits nowhere. After `reserve_after` it reserves a worker.
fn starving() -> PriorityBackfill {
    let mut p = PriorityBackfill::new(BackfillConfig::default());
    p.worker_update(worker(1, 4, 100, 0), 0.0);
    p.worker_update(worker(2, 4, 100, 0), 0.0);
    p.submit(job(10, 60, 1), 0.0);
    p.submit(job(11, 60, 1), 0.0);
    assert_eq!(p.dispatch(0.0).len(), 2);
    p.submit(job(1, 50, 0), 1.0);
    assert!(p.dispatch(1.0).is_empty());
    assert!(p.stats().reservations.is_empty(), "too early to reserve");
    assert!(p.dispatch(61.0).is_empty());
    let r = p.stats().reservations;
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].job, 1);
    p
}

/// The worker of the only reservation.
fn reserved_worker(p: &PriorityBackfill) -> u64 {
    p.stats().reservations[0].worker
}

/// Others avoid the reserved worker; the holder gets it once it drains.
#[test]
fn reserved_worker_admits_only_its_holder() {
    let mut p = starving();
    let w = reserved_worker(&p);
    // A small job fits on both workers but must avoid the reserved one.
    p.submit(job(20, 5, 2), 62.0);
    let out = p.dispatch(62.0);
    assert_eq!(out.len(), 1);
    assert_ne!(out[0].1, w);
    assert!(p.explain(1).unwrap().contains("holds the reservation"));
    // The reserved worker drains; the holder goes there by the escape hatch.
    let running_there = if w == 1 { 10 } else { 11 };
    p.completed(running_there, 100.0);
    assert_eq!(p.dispatch(100.0), vec![(1, w)]);
    assert!(p.stats().reservations.is_empty());
    assert_eq!(p.stats().last_dispatch_holders, vec![1]);
}

/// Cancelling the holder frees its worker for everyone.
#[test]
fn holder_cancelled_releases_the_worker() {
    let mut p = starving();
    let w = reserved_worker(&p);
    p.submit(job(20, 5, 2), 62.0);
    p.submit(job(21, 5, 2), 62.0);
    p.cancel(1);
    assert!(p.stats().reservations.is_empty());
    // Both workers take small jobs again (least loaded first: one on each).
    let out = p.dispatch(62.0);
    assert_eq!(out.len(), 2);
    assert!(out.iter().any(|&(_, x)| x == w));
}

/// Losing the reserved worker makes the holder reserve another.
#[test]
fn reserved_worker_leaves_and_the_holder_reserves_again() {
    let mut p = starving();
    let w = reserved_worker(&p);
    p.worker_gone(w, 70.0);
    assert!(p.stats().reservations.is_empty());
    // Still admitted nowhere and still old enough: it reserves the remaining worker.
    assert!(p.dispatch(70.0).is_empty());
    let r = p.stats().reservations;
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].job, 1);
    assert_ne!(r[0].worker, w);
}

/// A new empty worker takes the holder at once and ends the reservation.
#[test]
fn worker_joins_mid_reservation() {
    let mut p = starving();
    // A fresh, empty worker admits the holder at once (escape hatch); the reservation is released.
    p.worker_update(worker(3, 4, 100, 0), 80.0);
    assert_eq!(p.dispatch(80.0), vec![(1, 3)]);
    assert!(p.stats().reservations.is_empty());
    assert!(p.stats().workers.iter().all(|w| w.reserved_for.is_none()));
}

/// A heartbeat making room admits the holder before its worker drains.
#[test]
fn heartbeat_lowering_usage_admits_the_holder_early() {
    let mut p = starving();
    let w = reserved_worker(&p);
    // Reported usage dominates the placed sum: 95 used, nothing fits.
    p.worker_update(worker(w, 4, 100, 95), 90.0);
    assert!(p.dispatch(90.0).is_empty());
    // The heartbeat drops below what is placed (60): 60 + 50 > 100 still refuses ...
    p.worker_update(worker(w, 4, 100, 10), 91.0);
    assert!(p.dispatch(91.0).is_empty());
    // ... until the budget grows (e.g. a recalibrated limit): 60 + 50 <= 120.
    p.worker_update(worker(w, 4, 120, 10), 92.0);
    assert_eq!(p.dispatch(92.0), vec![(1, w)]);
    assert!(p.stats().reservations.is_empty());
}

/// Reported usage above the placed sum blocks admission until it drops.
#[test]
fn heartbeat_raising_usage_blocks_placements() {
    let mut p = PriorityBackfill::new(BackfillConfig::default());
    p.worker_update(worker(1, 4, 100, 0), 0.0);
    p.submit(job(1, 10, 0), 0.0);
    assert_eq!(p.dispatch(0.0).len(), 1);
    p.worker_update(worker(1, 4, 100, 95), 1.0);
    p.submit(job(2, 10, 0), 1.0);
    assert!(p.dispatch(1.0).is_empty());
    assert!(p.explain(2).unwrap().contains("memory short on 1"));
    p.worker_update(worker(1, 4, 100, 20), 2.0);
    assert_eq!(p.dispatch(2.0), vec![(2, 1)]);
}

/// A holder placed on another worker releases its reservation.
#[test]
fn holder_placed_elsewhere_releases_its_reservation() {
    let mut p = starving();
    let w = reserved_worker(&p);
    let other = if w == 1 { 2 } else { 1 };
    let running_there = if other == 1 { 10 } else { 11 };
    p.completed(running_there, 100.0);
    assert_eq!(p.dispatch(100.0), vec![(1, other)]);
    assert!(p.stats().reservations.is_empty());
    assert!(p.stats().last_dispatch_holders.is_empty());
}

/// Even an explicitly urgent, preferring job avoids a reserved worker.
#[test]
fn more_urgent_job_cannot_take_a_reserved_worker() {
    let mut p = starving();
    let w = reserved_worker(&p);
    // An explicitly urgent small job still may not use the reserved worker.
    let mut urgent = job(30, 5, 9);
    urgent.priority = Some(-10);
    urgent.prefer = vec![w];
    p.submit(urgent, 63.0);
    let out = p.dispatch(63.0);
    assert_eq!(out.len(), 1);
    assert_ne!(out[0].1, w);
}

/// Per-class limits allow one reservation per worker class.
#[test]
fn per_class_reservations() {
    let mut p = BestFit::new(BestFitConfig {
        backfill: BackfillConfig {
            per_class_reservations: true,
            ..BackfillConfig::default()
        },
        prefer_penalty: 0,
    });
    for (id, class) in [(1, "a"), (2, "a"), (3, "b"), (4, "b")] {
        p.worker_update(WorkerState::new(id, class, 4, Resources::mem(100)), 0.0);
    }
    for i in 0..4 {
        p.submit(job(10 + i, 60, 1), 0.0);
    }
    assert_eq!(p.dispatch(0.0).len(), 4);
    p.submit(job(1, 50, 0), 1.0);
    p.submit(job(2, 50, 0), 1.0);
    p.submit(job(3, 50, 0), 1.0);
    assert!(p.dispatch(100.0).is_empty());
    let r = p.stats().reservations;
    assert_eq!(r.len(), 2, "one per class: {r:?}");
    assert_eq!(r.iter().map(|r| r.job).collect::<Vec<_>>(), vec![1, 2]);
}

/// [`starving`], with shadow-time backfill and known work: the two 60-unit jobs end at 100 s.
fn starving_shadow() -> PriorityBackfill {
    let mut p = PriorityBackfill::new(BackfillConfig {
        shadow_backfill: true,
        ..BackfillConfig::default()
    });
    p.worker_update(worker(1, 4, 100, 0), 0.0);
    p.worker_update(worker(2, 4, 100, 0), 0.0);
    for id in [10, 11] {
        p.submit(
            JobSpec {
                work: Some(100.0),
                ..job(id, 60, 1)
            },
            0.0,
        );
    }
    assert_eq!(p.dispatch(0.0).len(), 2);
    p.submit(job(1, 50, 0), 1.0);
    p.dispatch(1.0);
    assert!(p.dispatch(61.0).is_empty());
    assert_eq!(p.stats().reservations.len(), 1);
    // Fill the other worker, so only shadow backfill can place anything more.
    let w = p.stats().reservations[0].worker;
    p.submit(
        JobSpec {
            work: Some(1e6),
            ..job(30, 40, 3)
        },
        61.0,
    );
    assert_eq!(p.dispatch(61.0), vec![(30, if w == 1 { 2 } else { 1 })]);
    p
}

/// A short job backfills the reserved worker (it ends before the holder can start there); a long
/// one does not.
#[test]
fn shadow_backfill_admits_jobs_that_end_in_time() {
    let mut p = starving_shadow();
    let w = p.stats().reservations[0].worker;
    // 20 s of work ends at 81 s, before the shadow time 100 s; 50 s would end at 111 s.
    p.submit(
        JobSpec {
            work: Some(50.0),
            ..job(21, 5, 2)
        },
        62.0,
    );
    p.submit(
        JobSpec {
            work: Some(20.0),
            ..job(20, 5, 2)
        },
        62.0,
    );
    assert_eq!(p.dispatch(62.0), vec![(20, w)]);
    assert!(
        p.explain(21)
            .unwrap()
            .contains(&format!("worker {w} for job 1"))
    );
    // Unknown work never backfills.
    p.submit(job(22, 5, 2), 63.0);
    assert!(p.dispatch(63.0).is_empty());
}

/// Once the shadow time has passed, the reserved worker drains strictly (overrunning jobs can no
/// longer delay the holder through new backfill).
#[test]
fn shadow_backfill_stops_at_the_shadow_time() {
    let mut p = starving_shadow();
    let w = p.stats().reservations[0].worker;
    // The shadow time is 100 s (the running jobs' expected end): a 1 s job at 95 s fits.
    p.submit(
        JobSpec {
            work: Some(1.0),
            ..job(20, 1, 2)
        },
        95.0,
    );
    assert_eq!(p.dispatch(95.0), vec![(20, w)]);
    p.completed(20, 96.0);
    // The running jobs overrun; from 100 s on, nothing but the holder goes there.
    p.submit(
        JobSpec {
            work: Some(0.5),
            ..job(21, 1, 2)
        },
        101.0,
    );
    assert!(p.dispatch(101.0).is_empty());
    assert!(
        p.explain(21)
            .unwrap()
            .contains(&format!("worker {w} for job 1"))
    );
}
