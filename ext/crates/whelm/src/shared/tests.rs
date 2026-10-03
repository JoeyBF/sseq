//! Behaviour of [`SharedPolicy`] and its leases across threads.

use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

use crate::{
    Config, FailKind, Input, JobId, JobSpec, Policy, Resources, RetryConfig, Scheduler,
    SharedPolicy, Speculate, Time, WorkerId, WorkerState,
};

/// A one-slot worker of class "x".
fn worker(id: WorkerId) -> WorkerState {
    WorkerState {
        id,
        class: "x".into(),
        budget: Resources::mem(100),
        ..Default::default()
    }
}

/// A shared FIFO scheduler with `max_attempts` attempts per job, on a manual clock of 0.
fn shared(max_attempts: u32) -> SharedPolicy<Scheduler> {
    SharedPolicy::new(
        Scheduler::new(Config {
            retry: RetryConfig { max_attempts },
            ..Config::fifo()
        }),
        || Time::ZERO,
    )
}

/// Stopping the tickers ends those running; one spawned afterwards keeps running until it is
/// stopped in turn.
#[test]
fn ticker_restarts_after_stop() {
    let s = Arc::new(shared(1));
    let first = s.spawn_ticker(Duration::from_millis(1));
    s.stop_ticker();
    first.join().unwrap();
    let second = s.spawn_ticker(Duration::from_millis(1));
    std::thread::sleep(Duration::from_millis(20));
    assert!(!second.is_finished());
    s.stop_ticker();
    second.join().unwrap();
}

/// A job.
fn job(id: JobId) -> JobSpec {
    JobSpec {
        id,
        demand: Resources::mem(1),
        ..Default::default()
    }
}

/// A failed lease is retried on another worker, then completes.
#[test]
fn fail_returns_the_retry() {
    let s = shared(4);
    s.worker_update(worker(1));
    s.worker_update(worker(2));
    let l = s.lease(job(7));
    assert_eq!((l.worker(), l.attempt()), (1, 1));
    let l = l.fail(FailKind::Other, "boom").unwrap();
    assert_eq!((l.worker(), l.attempt()), (2, 2));
    assert!(!l.stopped());
    l.complete();
    let st = s.stats();
    assert_eq!((st.waiting, st.running), (0, 0));
}

/// The last allowed failure returns the give-up.
#[test]
fn fail_gives_up() {
    let s = shared(2);
    s.worker_update(worker(1));
    let l = s.lease(job(7)).fail(FailKind::DeviceOom, "oom").unwrap();
    let g = l.fail(FailKind::DeviceOom, "oom").err().unwrap();
    assert_eq!((g.job, g.tried.len(), g.retryable), (7, 2, true));
    assert_eq!(s.stats().running, 0);
    // The id can be leased afresh.
    s.lease(job(7)).complete();
}

/// A thread blocked in `lease` is woken by a worker joining; one whose worker leaves gets the
/// policy's retry when it reports the lost link, even though the policy failed that attempt
/// first.
#[test]
fn worker_gone_then_fail_link_died() {
    let s = Arc::new(shared(4));
    let (tx, rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let t = {
        let s = s.clone();
        std::thread::spawn(move || {
            let l = s.lease(job(7));
            tx.send((l.worker(), l.attempt())).unwrap();
            go_rx.recv().unwrap();
            let l = l.fail(FailKind::LinkDied, "connection reset").unwrap();
            tx.send((l.worker(), l.attempt())).unwrap();
            l.complete();
        })
    };
    while s.waiting() == 0 {
        std::thread::yield_now();
    }
    s.worker_update(worker(1));
    assert_eq!(rx.recv().unwrap(), (1, 1));
    s.worker_update(worker(2));
    assert_eq!(s.worker_gone(1), vec![7]);
    // The policy already retried the job on worker 2.
    assert_eq!(s.stats().running, 1);
    go_tx.send(()).unwrap();
    assert_eq!(rx.recv().unwrap(), (2, 2));
    t.join().unwrap();
    let st = s.stats();
    assert_eq!((st.waiting, st.running), (0, 0));
    assert_eq!(st.placements_total, 2);
}

/// Completing an attempt whose worker left cancels the policy's retry: the result is in hand.
#[test]
fn complete_after_worker_gone_cancels_the_retry() {
    let s = shared(4);
    s.worker_update(worker(1));
    s.worker_update(worker(2));
    let l = s.lease(job(7));
    s.worker_gone(l.worker());
    assert_eq!(s.stats().running, 1);
    l.complete();
    let st = s.stats();
    assert_eq!((st.waiting, st.running), (0, 0));
    assert!(st.workers.iter().all(|w| w.running == 0));
}

/// A speculative second attempt is rejected: the lease keeps its attempt and completes it.
#[test]
fn speculative_start_is_rejected() {
    let mut cfg = Config::fifo();
    cfg.speed.speculate = Some(Speculate::default());
    let s = SharedPolicy::new(Scheduler::new(cfg), || Time::ZERO);
    s.worker_update(worker(1));
    let mut j = job(7);
    j.work = Some(Duration::from_secs(100));
    let l = s.lease(j);
    s.worker_update(WorkerState {
        speed: 4.0,
        ..worker(2)
    });
    let st = s.stats();
    assert_eq!(st.placements_total, 2);
    assert!(st.workers.iter().all(|w| w.running == (w.id == 1) as usize));
    assert_eq!(l.attempt(), 1);
    l.complete();
    assert_eq!(s.stats().running, 0);
}

/// Dropping a lease cancels its job; a timed lease returns the job when it expires.
#[test]
fn drop_and_timeout_cancel() {
    let s = shared(4);
    s.worker_update(worker(1));
    drop(s.lease(job(1)));
    assert_eq!(s.stats().running, 0);
    let held = s.lease(job(2));
    let back = s.lease_timeout(job(3), Duration::from_millis(10)).err();
    assert_eq!(back.map(|j| j.id), Some(3));
    assert_eq!(s.stats().waiting, 0);
    held.complete();
}

/// A lease whose job is cancelled through `with` sees the stop.
#[test]
fn stop_is_visible() {
    let s = shared(4);
    s.worker_update(worker(1));
    let l = s.lease(job(1));
    s.with(|p, now| p.handle(Input::Cancel(1), now));
    assert!(l.stopped());
    l.complete();
    assert_eq!(s.stats().running, 0);
}
