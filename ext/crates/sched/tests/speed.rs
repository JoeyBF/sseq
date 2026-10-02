//! Speed-aware placement: fastest-first, earliest finish with deferral, learning and spoliation.

use sched::{
    Config, Defer, JobId, JobSpec, Policy, Resources, Scheduler, SpeedConfig, SpeedPolicy,
    WorkerState,
};

/// A worker of the given speed with ample memory.
fn worker(id: u64, slots: usize, speed: f64) -> WorkerState {
    WorkerState {
        speed,
        ..WorkerState::new(
            id,
            if speed > 1.0 { "fast" } else { "slow" },
            slots,
            Resources::mem(1000),
        )
    }
}

/// A unit-demand job with optional work.
fn job(id: JobId, work: Option<f64>) -> JobSpec {
    JobSpec {
        work,
        ..JobSpec::new(id, Resources::mem(1), 0)
    }
}

/// A backfill policy with the given speed settings.
fn backfill(speed: SpeedConfig) -> Scheduler {
    Scheduler::new(Config {
        speed,
        ..Config::default()
    })
}

/// Fastest first beats load balancing, for every preset, including best fit's tight packing.
#[test]
fn fastest_first_picks_the_fast_worker() {
    let speed = SpeedConfig {
        policy: SpeedPolicy::FastestFirst,
        learn: None,
        spoliation: None,
    };
    for base in [Config::fifo(), Config::default(), Config::best_fit()] {
        let mut p = Scheduler::new(Config { speed, ..base });
        p.worker_update(worker(1, 4, 1.0), 0.0);
        // The fast worker has more room left, which best fit alone would avoid.
        p.worker_update(
            WorkerState {
                budget: Resources::mem(5000),
                ..worker(2, 4, 2.4)
            },
            0.0,
        );
        for i in 0..6 {
            p.submit(job(i, None), 0.0);
        }
        let out = p.dispatch(0.0);
        let on_fast = out.iter().filter(|x| x.1 == 2).count();
        assert_eq!(on_fast, 4, "{out:?}");
        assert_eq!(out.len(), 6, "the overflow still runs on the slow worker");
    }
}

/// Oblivious placement keeps the historical least-loaded choice.
#[test]
fn oblivious_ignores_speed() {
    let mut p = backfill(SpeedConfig::default());
    p.worker_update(worker(1, 4, 1.0), 0.0);
    p.worker_update(worker(2, 4, 2.4), 0.0);
    p.submit(job(0, None), 0.0);
    assert_eq!(p.dispatch(0.0), vec![(0, 1)]);
}

/// Earliest finish waits for a fast slot that frees soon, but not for one that frees late.
#[test]
fn earliest_finish_defers_only_when_it_pays() {
    let defer = Defer {
        max_wait: 100.0,
        min_gain: 0.0,
    };
    let speed = SpeedConfig {
        policy: SpeedPolicy::EarliestFinish(Some(defer)),
        learn: None,
        spoliation: None,
    };
    for (running_work, expect_defer) in [(10.0, true), (1000.0, false)] {
        let mut p = backfill(speed);
        p.worker_update(worker(1, 1, 1.0), 0.0);
        p.worker_update(worker(2, 1, 4.0), 0.0);
        // Occupy the fast worker: it frees at running_work / 4.
        p.submit(job(0, Some(running_work)), 0.0);
        assert_eq!(p.dispatch(0.0), vec![(0, 2)]);
        // 40 units of work: 40 s on the slow worker, or 10 s after the fast one frees.
        p.submit(job(1, Some(40.0)), 0.0);
        let out = p.dispatch(0.0);
        if expect_defer {
            assert!(out.is_empty(), "should wait for the fast worker: {out:?}");
            assert_eq!(p.stats().deferred, vec![(1, 2, 2.5)]);
            assert_eq!(p.next_wakeup(), Some(100.0));
            assert!(
                p.explain(1)
                    .unwrap()
                    .contains("waiting for faster worker 2")
            );
            p.completed(0, 2.5);
            assert_eq!(p.dispatch(2.5), vec![(1, 2)]);
        } else {
            assert_eq!(out, vec![(1, 1)]);
            assert!(p.stats().deferred.is_empty());
        }
    }
}

/// A deferred job stops waiting after `max_wait`, even if the fast worker is still busy.
#[test]
fn deferral_expires() {
    let defer = Defer {
        max_wait: 30.0,
        min_gain: 0.0,
    };
    let speed = SpeedConfig {
        policy: SpeedPolicy::EarliestFinish(Some(defer)),
        learn: None,
        spoliation: None,
    };
    let mut p = backfill(speed);
    p.worker_update(worker(1, 1, 1.0), 0.0);
    p.worker_update(worker(2, 1, 10.0), 0.0);
    p.submit(job(0, Some(200.0)), 0.0); // frees at 20 on the fast worker
    p.dispatch(0.0);
    p.submit(job(1, Some(100.0)), 0.0); // 100 s slow vs 30 s after waiting
    assert!(p.dispatch(0.0).is_empty());
    assert_eq!(p.next_wakeup(), Some(30.0));
    // The fast job overruns: still busy at the deadline, so the waiter gives up and runs slowly.
    assert!(p.dispatch(29.0).is_empty());
    assert_eq!(p.dispatch(30.0), vec![(1, 1)]);
    assert_eq!(p.next_wakeup(), None);
}

/// Several deferred jobs book successive slots of the fast worker in urgency order.
#[test]
fn deferrals_book_slots_in_order() {
    let defer = Defer {
        max_wait: 1e9,
        min_gain: 0.0,
    };
    let speed = SpeedConfig {
        policy: SpeedPolicy::EarliestFinish(Some(defer)),
        learn: None,
        spoliation: None,
    };
    let mut p = backfill(speed);
    p.worker_update(worker(1, 1, 1.0), 0.0);
    p.worker_update(worker(2, 1, 10.0), 0.0);
    p.submit(job(0, Some(10.0)), 0.0); // fast worker frees at 1
    p.dispatch(0.0);
    for i in 1..=3 {
        p.submit(job(i, Some(50.0)), 0.0); // 50 s slow, 5 s fast
    }
    // Job 1 waits (start 1, done 6); job 2 waits (start 6, done 11); job 3 would finish at 16 on
    // the fast worker but at 50 on the slow one, so it waits too.
    assert!(p.dispatch(0.0).is_empty());
    let d: Vec<_> = p.stats().deferred.iter().map(|d| (d.0, d.2)).collect();
    assert_eq!(d, vec![(1, 1.0), (2, 6.0), (3, 11.0)]);
}

/// Speeds learned from completion times override the reported ones once warmed up.
#[test]
fn learned_speeds_replace_reported_ones() {
    let speed = SpeedConfig {
        policy: SpeedPolicy::FastestFirst,
        learn: Some(sched::Learn::default()),
        spoliation: None,
    };
    let mut p = backfill(speed);
    // Both report 1.0; worker 2 really runs three times faster.
    p.worker_update(WorkerState::new(1, "a", 1, Resources::mem(1000)), 0.0);
    p.worker_update(WorkerState::new(2, "b", 1, Resources::mem(1000)), 0.0);
    let truth = |w: u64| if w == 2 { 3.0 } else { 1.0 };
    let mut now = 0.0;
    let mut id = 0;
    let mut running: Vec<(u64, u64, f64)> = Vec::new();
    for _ in 0..200 {
        // Keep both workers busy: one queued job per free worker.
        for _ in running.len()..2 {
            p.submit(job(id, Some(6.0)), now);
            id += 1;
        }
        for (j, w) in p.dispatch(now) {
            running.push((j, w, now + 6.0 / truth(w)));
        }
        running.sort_by(|a, b| a.2.total_cmp(&b.2));
        let (j, _, end) = running.remove(0);
        now = end;
        p.completed(j, now);
    }
    let loads = p.stats().workers;
    let learned: Vec<f64> = loads.iter().map(|l| l.speed).collect();
    assert!(
        (learned[0] - 1.0).abs() < 1e-9 && (learned[1] - 3.0).abs() < 1e-9,
        "{learned:?}"
    );
    // With both free, the truly fast worker is preferred now.
    for (j, _, _) in running.drain(..) {
        p.completed(j, now);
    }
    p.submit(job(10_000, Some(6.0)), now);
    assert_eq!(p.dispatch(now), vec![(10_000, 2)]);
}

/// A long job stuck on a slow worker moves to a fast worker that frees up; completing it then
/// releases the fast worker.
#[test]
fn spoliation_moves_a_stuck_job() {
    let speed = SpeedConfig {
        policy: SpeedPolicy::FastestFirst,
        learn: None,
        spoliation: Some(sched::Spoliation::default()),
    };
    let mut p = backfill(speed);
    p.worker_update(worker(1, 1, 1.0), 0.0);
    p.worker_update(worker(2, 1, 4.0), 0.0);
    // A short job takes the fast worker; the long one goes slow (100 s there, 25 s fast).
    p.submit(job(0, Some(4.0)), 0.0);
    p.submit(job(1, Some(100.0)), 0.0);
    let d = p.dispatch_full(0.0);
    assert_eq!(d.start, vec![(0, 2), (1, 1)]);
    assert!(d.preempt.is_empty());
    // The fast worker frees at 1 s: the long job ends at 100 s where it is, 26 s if restarted.
    p.completed(0, 1.0);
    let d = p.dispatch_full(1.0);
    assert!(d.start.is_empty());
    assert_eq!(
        d.preempt,
        vec![sched::Preemption {
            job: 1,
            from: 1,
            to: 2
        }]
    );
    // At most once: freeing the slow worker does not bounce it back.
    assert!(p.dispatch_full(1.0).preempt.is_empty());
    let loads = p.stats().workers;
    assert_eq!((loads[0].running, loads[1].running), (0, 1));
    p.completed(1, 26.0);
    assert!(p.stats().workers.iter().all(|w| w.running == 0));
    // Without spoliation, dispatch_full is plain dispatch.
    let mut q = backfill(SpeedConfig {
        spoliation: None,
        ..speed
    });
    q.worker_update(worker(1, 1, 1.0), 0.0);
    q.worker_update(worker(2, 1, 4.0), 0.0);
    q.submit(job(0, Some(4.0)), 0.0);
    q.submit(job(1, Some(100.0)), 0.0);
    q.dispatch_full(0.0);
    q.completed(0, 1.0);
    assert!(q.dispatch_full(1.0).preempt.is_empty());
}

/// A clock-capped worker of the same class is learned slower than its peers, so fastest-first
/// fills it last; peers within the resolution still share work by load.
#[test]
fn capped_worker_learned_per_worker() {
    let speed = SpeedConfig {
        policy: SpeedPolicy::FastestFirst,
        learn: Some(sched::Learn::default()),
        spoliation: None,
    };
    let mut p = backfill(speed);
    for w in 1..=3 {
        p.worker_update(WorkerState::new(w, "h200", 1, Resources::mem(1000)), 0.0);
    }
    let truth = |w: u64| if w == 3 { 0.765 } else { 1.0 };
    let mut now = 0.0;
    let mut id = 0;
    let mut running: Vec<(u64, u64, f64)> = Vec::new();
    for _ in 0..600 {
        for _ in running.len()..3 {
            p.submit(job(id, Some(10.0)), now);
            id += 1;
        }
        for (j, w) in p.dispatch(now) {
            running.push((j, w, now + 10.0 / truth(w)));
        }
        running.sort_by(|a, b| a.2.total_cmp(&b.2));
        let (j, _, end) = running.remove(0);
        now = end;
        p.completed(j, now);
    }
    let speeds: Vec<f64> = p.stats().workers.iter().map(|l| l.speed).collect();
    assert!(speeds[2] < 0.9 * speeds[0], "{speeds:?}");
    assert!((speeds[0] / speeds[1] - 1.0).abs() < 0.1, "{speeds:?}");
    // With all three free, the two healthy workers are taken first, in load order.
    for (j, _, _) in running.drain(..) {
        p.completed(j, now);
    }
    for k in 0..2 {
        p.submit(job(10_000 + k, Some(10.0)), now);
    }
    let mut placed: Vec<u64> = p.dispatch(now).into_iter().map(|x| x.1).collect();
    placed.sort();
    assert_eq!(placed, vec![1, 2]);
}
