//! Speed-aware placement: the speed score, deferral to a faster worker, learning and speculation.

use sched::{
    Attempt, Config, Defer, Input, JobId, JobSpec, Output, Policy, Resources, Scheduler, ScoreTerm,
    Speculate, SpeedConfig, WorkerId, WorkerState,
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

/// A speed term first beats load balancing, for every preset, including best fit's tight
/// packing.
#[test]
fn speed_first_picks_the_fast_worker() {
    for base in [Config::fifo(), Config::default(), Config::best_fit()] {
        let score = [vec![ScoreTerm::Speed], base.score.clone()].concat();
        let mut p = Scheduler::new(Config { score, ..base });
        p.handle(Input::Worker(worker(1, 4, 1.0)), 0.0);
        // The fast worker has more room left, which best fit alone would avoid.
        p.handle(
            Input::Worker(WorkerState {
                speed: 2.4,
                ..WorkerState::new(2, "fast", 4, Resources::mem(5000))
            }),
            0.0,
        );
        for i in 0..6 {
            p.handle(Input::Submit(job(i, None)), 0.0);
        }
        let out = starts(p.poll(0.0));
        let on_fast = out.iter().filter(|x| x.1 == 2).count();
        assert_eq!(on_fast, 4, "{out:?}");
        assert_eq!(out.len(), 6, "the overflow still runs on the slow worker");
    }
}

/// Without a speed term, placement ignores speed: the least loaded, then the smallest id.
#[test]
fn without_speed_term_speed_is_ignored() {
    let mut p = Scheduler::new(Config {
        score: vec![ScoreTerm::Preferred, ScoreTerm::Load],
        ..Config::default()
    });
    p.handle(Input::Worker(worker(1, 4, 1.0)), 0.0);
    p.handle(Input::Worker(worker(2, 4, 2.4)), 0.0);
    p.handle(Input::Submit(job(0, None)), 0.0);
    assert_eq!(starts(p.poll(0.0)), vec![(0, 1)]);
}

/// Deferral waits for a fast slot that frees soon, but not for one that frees late.
#[test]
fn earliest_finish_defers_only_when_it_pays() {
    let defer = Defer {
        max_wait: 100.0,
        min_gain: 0.0,
    };
    let speed = SpeedConfig {
        defer: Some(defer),
        ..SpeedConfig::default()
    };
    for (running_work, expect_defer) in [(10.0, true), (1000.0, false)] {
        let mut p = backfill(speed);
        p.handle(Input::Worker(worker(1, 1, 1.0)), 0.0);
        p.handle(Input::Worker(worker(2, 1, 4.0)), 0.0);
        // Occupy the fast worker: it frees at running_work / 4.
        p.handle(Input::Submit(job(0, Some(running_work))), 0.0);
        assert_eq!(starts(p.poll(0.0)), vec![(0, 2)]);
        // 40 units of work: 40 s on the slow worker, or 10 s after the fast one frees.
        p.handle(Input::Submit(job(1, Some(40.0))), 0.0);
        let out = starts(p.poll(0.0));
        if expect_defer {
            assert!(out.is_empty(), "should wait for the fast worker: {out:?}");
            assert_eq!(p.stats().deferred, vec![(1, 2, 2.5)]);
            // The wait lapses at `max_wait`, unless the job may reserve before then.
            let reserve_after = Config::default().reservations.unwrap().reserve_after;
            assert_eq!(p.next_wakeup(), Some(reserve_after.min(100.0)));
            assert!(
                p.explain(1)
                    .unwrap()
                    .contains("waiting for faster worker 2")
            );
            p.handle(done(0), 2.5);
            assert_eq!(starts(p.poll(2.5)), vec![(1, 2)]);
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
        defer: Some(defer),
        ..SpeedConfig::default()
    };
    let mut p = backfill(speed);
    p.handle(Input::Worker(worker(1, 1, 1.0)), 0.0);
    p.handle(Input::Worker(worker(2, 1, 10.0)), 0.0);
    p.handle(Input::Submit(job(0, Some(200.0))), 0.0); // frees at 20 on the fast worker
    starts(p.poll(0.0));
    p.handle(Input::Submit(job(1, Some(100.0))), 0.0); // 100 s slow vs 30 s after waiting
    assert!(starts(p.poll(0.0)).is_empty());
    assert_eq!(p.next_wakeup(), Some(30.0));
    // The fast job overruns: still busy at the deadline, so the waiter gives up and runs slowly.
    assert!(starts(p.poll(29.0)).is_empty());
    assert_eq!(starts(p.poll(30.0)), vec![(1, 1)]);
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
        defer: Some(defer),
        ..SpeedConfig::default()
    };
    let mut p = backfill(speed);
    p.handle(Input::Worker(worker(1, 1, 1.0)), 0.0);
    p.handle(Input::Worker(worker(2, 1, 10.0)), 0.0);
    p.handle(Input::Submit(job(0, Some(10.0))), 0.0); // fast worker frees at 1
    starts(p.poll(0.0));
    for i in 1..=3 {
        p.handle(Input::Submit(job(i, Some(50.0))), 0.0); // 50 s slow, 5 s fast
    }
    // Job 1 waits (start 1, done 6); job 2 waits (start 6, done 11); job 3 would finish at 16 on
    // the fast worker but at 50 on the slow one, so it waits too.
    assert!(starts(p.poll(0.0)).is_empty());
    let d: Vec<_> = p.stats().deferred.iter().map(|d| (d.0, d.2)).collect();
    assert_eq!(d, vec![(1, 1.0), (2, 6.0), (3, 11.0)]);
}

/// Speeds learned from completion times override the reported ones once warmed up.
#[test]
fn learned_speeds_replace_reported_ones() {
    let speed = SpeedConfig {
        learn: Some(sched::Learn::default()),
        ..SpeedConfig::default()
    };
    let mut p = backfill(speed);
    // Both report 1.0; worker 2 really runs three times faster.
    p.handle(
        Input::Worker(WorkerState::new(1, "a", 1, Resources::mem(1000))),
        0.0,
    );
    p.handle(
        Input::Worker(WorkerState::new(2, "b", 1, Resources::mem(1000))),
        0.0,
    );
    let truth = |w: u64| if w == 2 { 3.0 } else { 1.0 };
    let mut now = 0.0;
    let mut id = 0;
    let mut running: Vec<(u64, u64, f64)> = Vec::new();
    for _ in 0..200 {
        // Keep both workers busy: one queued job per free worker.
        for _ in running.len()..2 {
            p.handle(Input::Submit(job(id, Some(6.0))), now);
            id += 1;
        }
        for (j, w) in starts(p.poll(now)) {
            running.push((j, w, now + 6.0 / truth(w)));
        }
        running.sort_by(|a, b| a.2.total_cmp(&b.2));
        let (j, _, end) = running.remove(0);
        now = end;
        p.handle(done(j), now);
    }
    let loads = p.stats().workers;
    let learned: Vec<f64> = loads.iter().map(|l| l.speed).collect();
    assert!(
        (learned[0] - 1.0).abs() < 1e-9 && (learned[1] - 3.0).abs() < 1e-9,
        "{learned:?}"
    );
    // With both free, the truly fast worker is preferred now.
    for (j, _, _) in running.drain(..) {
        p.handle(done(j), now);
    }
    p.handle(Input::Submit(job(10_000, Some(6.0))), now);
    assert_eq!(starts(p.poll(now)), vec![(10_000, 2)]);
}

/// Two one-slot workers, slow (1) and fast (2, four times faster), with `speculate`: a short job
/// takes the fast worker and a long one the slow worker (100 s there, 25 s on the fast one); the
/// short one ends at 1 s.
fn stuck(speculate: Option<Speculate>) -> Scheduler {
    let mut p = backfill(SpeedConfig {
        speculate,
        ..SpeedConfig::default()
    });
    p.handle(Input::Worker(worker(1, 1, 1.0)), 0.0);
    p.handle(Input::Worker(worker(2, 1, 4.0)), 0.0);
    p.handle(Input::Submit(job(0, Some(4.0))), 0.0);
    p.handle(Input::Submit(job(1, Some(100.0))), 0.0);
    assert_eq!(starts(p.poll(0.0)), vec![(0, 2), (1, 1)]);
    p.handle(done(0), 1.0);
    p
}

/// The start of attempt `attempt` of job 1 on `worker`.
fn start1(attempt: Attempt, worker: WorkerId) -> Output {
    Output::Start {
        job: 1,
        attempt,
        worker,
    }
}

/// The idle fast worker starts a second attempt of the long job (ending at 26 s instead of
/// 100 s); the first attempt to finish wins and the other is stopped, and a late report of the
/// loser is ignored.
#[test]
fn speculation_starts_a_second_attempt_and_the_first_done_wins() {
    for winner in [2, 1] {
        let mut p = stuck(Some(Speculate::default()));
        assert_eq!(p.poll(1.0), vec![start1(2, 2)]);
        // At most `max_per_job` speculative attempts.
        assert_eq!(p.poll(1.0), vec![]);
        let s = p.stats();
        assert_eq!((s.running, s.placements_total), (1, 3));
        assert_eq!((s.workers[0].running, s.workers[1].running), (1, 1));
        assert!(
            p.explain(1)
                .unwrap()
                .contains("attempt 1 on worker 1, attempt 2 on worker 2")
        );
        // Attempt n runs on worker n.
        let (loser, at) = if winner == 2 { (1, 26.0) } else { (2, 100.0) };
        p.handle(
            Input::Done {
                job: 1,
                attempt: winner,
            },
            at,
        );
        assert_eq!(
            p.poll(at),
            vec![Output::Stop {
                job: 1,
                attempt: loser,
                worker: WorkerId::from(loser),
            }]
        );
        p.handle(
            Input::Done {
                job: 1,
                attempt: loser,
            },
            at + 1.0,
        );
        assert_eq!(p.poll(at + 1.0), vec![]);
        let s = p.stats();
        assert_eq!(s.running + s.waiting, 0);
        assert!(s.workers.iter().all(|w| w.running == 0));
    }
}

/// A failed speculative attempt leaves the original running: no retry, no give-up.
#[test]
fn failed_speculative_attempt_leaves_the_original_running() {
    let mut p = stuck(Some(Speculate::default()));
    assert_eq!(p.poll(1.0), vec![start1(2, 2)]);
    p.handle(Input::WorkerGone(2), 5.0);
    assert_eq!(p.poll(5.0), vec![]);
    let s = p.stats();
    assert_eq!((s.running, s.waiting), (1, 0));
    assert!(p.explain(1).unwrap().contains("attempt 1 on worker 1"));
    p.handle(Input::Done { job: 1, attempt: 1 }, 100.0);
    assert_eq!(p.poll(100.0), vec![]);
    assert_eq!(p.stats().running, 0);
}

/// Speculation only uses a slot no waiting job takes, and is off unless configured.
#[test]
fn speculation_yields_to_waiting_jobs_and_is_opt_in() {
    let mut p = stuck(Some(Speculate::default()));
    p.handle(Input::Submit(job(2, Some(4.0))), 1.0);
    assert_eq!(starts(p.poll(1.0)), vec![(2, 2)]);
    let mut q = stuck(None);
    assert_eq!(q.poll(1.0), vec![]);
}

/// A clock-capped worker of the same class is learned slower than its peers, so the speed term
/// fills it last; peers within the resolution still share work by load.
#[test]
fn capped_worker_learned_per_worker() {
    let speed = SpeedConfig {
        learn: Some(sched::Learn::default()),
        ..SpeedConfig::default()
    };
    let mut p = backfill(speed);
    for w in 1..=3 {
        p.handle(
            Input::Worker(WorkerState::new(w, "h200", 1, Resources::mem(1000))),
            0.0,
        );
    }
    let truth = |w: u64| if w == 3 { 0.765 } else { 1.0 };
    let mut now = 0.0;
    let mut id = 0;
    let mut running: Vec<(u64, u64, f64)> = Vec::new();
    for _ in 0..600 {
        for _ in running.len()..3 {
            p.handle(Input::Submit(job(id, Some(10.0))), now);
            id += 1;
        }
        for (j, w) in starts(p.poll(now)) {
            running.push((j, w, now + 10.0 / truth(w)));
        }
        running.sort_by(|a, b| a.2.total_cmp(&b.2));
        let (j, _, end) = running.remove(0);
        now = end;
        p.handle(done(j), now);
    }
    let speeds: Vec<f64> = p.stats().workers.iter().map(|l| l.speed).collect();
    assert!(speeds[2] < 0.9 * speeds[0], "{speeds:?}");
    assert!((speeds[0] / speeds[1] - 1.0).abs() < 0.1, "{speeds:?}");
    // With all three free, the two healthy workers are taken first, in load order.
    for (j, _, _) in running.drain(..) {
        p.handle(done(j), now);
    }
    for k in 0..2 {
        p.handle(Input::Submit(job(10_000 + k, Some(10.0))), now);
    }
    let mut placed: Vec<u64> = starts(p.poll(now)).into_iter().map(|x| x.1).collect();
    placed.sort();
    assert_eq!(placed, vec![1, 2]);
}
