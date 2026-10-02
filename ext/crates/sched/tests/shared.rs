//! The blocking front end: placement, timeouts, failures and retries, leases, the ticker, and a
//! many-thread stress test with worker churn.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use proptest::prelude::*;
use sched::{
    Config, Defer, FailKind, FailOutcome, JobSpec, Resources, RetryConfig, Scheduler, SharedPolicy,
    SpeedConfig, SpeedPolicy, WorkerState,
};

/// A worker of class "x".
fn worker(id: u64, slots: usize) -> WorkerState {
    WorkerState::new(id, "x", slots, Resources::mem(1 << 40))
}

/// A unit job.
fn job(id: u64) -> JobSpec {
    JobSpec::new(id, Resources::mem(1), 0)
}

/// A front end over the default backfill policy.
fn shared() -> SharedPolicy<Scheduler> {
    SharedPolicy::with_system_clock(Scheduler::new(Config::default()))
}

/// `place` blocks until a worker joins, then returns it.
#[test]
fn place_blocks_until_a_worker_joins() {
    let s = Arc::new(shared());
    let t = {
        let s = s.clone();
        std::thread::spawn(move || s.place(job(1)))
    };
    while s.waiting() == 0 {
        std::thread::yield_now();
    }
    s.worker_update(worker(7, 1));
    let p = t.join().unwrap();
    assert_eq!((p.worker, p.attempt), (7, 1));
    s.completed(1);
    assert_eq!(s.stats().running, 0);
}

/// A timed-out placement withdraws the job.
#[test]
fn timeout_withdraws() {
    let s = shared();
    let back = s
        .place_timeout(job(1), Duration::from_millis(20))
        .unwrap_err();
    assert_eq!(back.id, 1);
    let st = s.stats();
    assert_eq!((st.waiting, st.running), (0, 0));
    // The id can be placed again.
    s.worker_update(worker(1, 1));
    assert_eq!(s.place(job(1)).worker, 1);
}

/// Retries avoid the workers tried; after `max_attempts` the job gives up, retryable iff every
/// attempt was a device OOM; resources are freed each time.
#[test]
fn retries_avoid_tried_workers_then_give_up() {
    let s = shared();
    for w in 1..=5 {
        s.worker_update(worker(w, 1));
    }
    let mut tried = Vec::new();
    for attempt in 1..=4 {
        let p = s.place(job(9));
        assert_eq!(p.attempt, attempt);
        assert!(
            !tried.contains(&p.worker),
            "retried on {} ({tried:?})",
            p.worker
        );
        tried.push(p.worker);
        match s.failed(9, FailKind::DeviceOom, "oom") {
            FailOutcome::Retry { attempts, avoid } => {
                assert_eq!(attempts, attempt);
                assert_eq!(avoid, tried);
            }
            FailOutcome::GiveUp {
                tried: t,
                retryable,
            } => {
                assert_eq!(attempt, 4);
                assert_eq!(t.len(), 4);
                assert!(retryable);
            }
        }
        assert_eq!(s.stats().running, 0);
    }
    // The history is forgotten: a new placement is attempt 1 again.
    assert_eq!(s.place(job(9)).attempt, 1);
    s.completed(9);
    // A mixed history is not retryable.
    for kind in [FailKind::DeviceOom, FailKind::LinkDied, FailKind::DeviceOom] {
        s.place(job(10));
        assert!(matches!(s.failed(10, kind, "x"), FailOutcome::Retry { .. }));
    }
    s.place(job(10));
    assert!(matches!(
        s.failed(10, FailKind::DeviceOom, "x"),
        FailOutcome::GiveUp {
            retryable: false,
            ..
        }
    ));
}

/// With one live worker, a retry goes back to it (soft avoid) rather than waiting forever.
#[test]
fn retry_on_the_only_worker() {
    let s = shared();
    s.worker_update(worker(1, 1));
    assert_eq!(s.place(job(1)).worker, 1);
    assert!(matches!(
        s.failed(1, FailKind::DeviceOom, "oom"),
        FailOutcome::Retry { .. }
    ));
    let p = s.place_timeout(job(1), Duration::from_secs(5)).unwrap();
    assert_eq!((p.worker, p.attempt), (1, 2));
}

/// A dropped lease releases its job.
#[test]
fn dropped_lease_releases() {
    let s = shared();
    s.worker_update(worker(1, 1));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let lease = s.lease(job(1));
        assert_eq!(lease.worker(), 1);
        panic!("the task thread unwinds");
    }));
    assert!(r.is_err());
    assert_eq!(s.stats().running, 0);
    // The slot is free again.
    let lease = s.lease(job(2));
    assert_eq!(lease.placement().attempt, 1);
    lease.complete();
    assert_eq!(s.stats().running, 0);
}

/// Without events, the ticker lets a deferred job go once its deferral expires.
#[test]
fn ticker_releases_timed_waits() {
    let policy = Scheduler::new(Config {
        speed: SpeedConfig {
            policy: SpeedPolicy::EarliestFinish(Some(Defer {
                max_wait: 0.2,
                min_gain: 0.0,
            })),
            ..SpeedConfig::default()
        },
        ..Config::default()
    });
    let work = |id, work| JobSpec {
        work: Some(work),
        ..job(id)
    };
    let s = Arc::new(SharedPolicy::with_system_clock(policy));
    s.worker_update(WorkerState {
        speed: 2.0,
        ..worker(1, 1)
    });
    s.worker_update(worker(2, 1));
    // The fast worker is busy for 5 s; job 2 would still finish there first (15 s against 20 s on
    // the slow worker), so it waits for it.
    assert_eq!(s.place(work(1, 10.0)).worker, 1);
    assert!(
        s.place_timeout(work(2, 20.0), Duration::from_millis(50))
            .is_err(),
        "deferred"
    );
    let ticker = s.spawn_ticker(Duration::from_millis(20));
    let start = std::time::Instant::now();
    let p = s.place(work(2, 20.0));
    assert_eq!(p.worker, 2);
    assert!(start.elapsed() < Duration::from_secs(5));
    s.stop_ticker();
    ticker.join().unwrap();
}

#[derive(Clone, Debug)]
enum Op {
    Join(u64, usize),
    Gone(u64),
    Place(u64),
    Complete(usize),
    Fail(usize, bool),
}

/// A random event.
fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        2 => (0u64..5, 0usize..3).prop_map(|(w, s)| Op::Join(w, s)),
        1 => (0u64..5).prop_map(Op::Gone),
        4 => (0u64..6).prop_map(Op::Place),
        2 => any::<prop::sample::Index>().prop_map(|i| Op::Complete(i.index(1 << 16))),
        3 => (any::<prop::sample::Index>(), any::<bool>())
            .prop_map(|(i, oom)| Op::Fail(i.index(1 << 16), oom)),
    ]
}

proptest! {
    /// Attempt counting, the soft avoid rule, and no leaks, over random sequences (single
    /// threaded: placements use a zero timeout, so a job not placed at once is withdrawn).
    #[test]
    fn attempts_avoid_and_no_leaks(ops in prop::collection::vec(op(), 1..120)) {
        let s = SharedPolicy::with_retry(
            Scheduler::new(Config::default()),
            || 0.0,
            RetryConfig { max_attempts: 3 },
        );
        let mut live: BTreeMap<u64, usize> = BTreeMap::new();
        let mut running: BTreeMap<u64, u64> = BTreeMap::new();
        // Jobs whose worker left: the policy forgot them; their outcome is still to report.
        let mut lost: BTreeSet<u64> = BTreeSet::new();
        let mut tried: HashMap<u64, Vec<u64>> = HashMap::new();
        for op in &ops {
            match *op {
                Op::Join(w, slots) => {
                    s.worker_update(worker(w, slots));
                    live.insert(w, slots);
                }
                Op::Gone(w) => {
                    let jobs = s.worker_gone(w);
                    live.remove(&w);
                    let expected: Vec<u64> =
                        running.iter().filter(|r| *r.1 == w).map(|r| *r.0).collect();
                    prop_assert_eq!(&jobs, &expected);
                    lost.extend(jobs);
                }
                Op::Place(j) => {
                    if running.contains_key(&j) {
                        continue;
                    }
                    if let Ok(p) = s.place_timeout(job(j), Duration::ZERO) {
                        let t = tried.get(&j).cloned().unwrap_or_default();
                        prop_assert_eq!(p.attempt as usize, t.len() + 1);
                        prop_assert!(live.get(&p.worker).is_some_and(|&n| n > 0));
                        if t.contains(&p.worker) {
                            // Soft: only while every live worker was tried.
                            prop_assert!(
                                live.iter().all(|(w, &n)| n == 0 || t.contains(w)),
                                "retried on {} with an untried live worker", p.worker
                            );
                        }
                        running.insert(j, p.worker);
                    }
                }
                Op::Complete(i) => {
                    if let Some((&j, _)) = running.iter().nth(i % running.len().max(1)) {
                        running.remove(&j);
                        lost.remove(&j);
                        tried.remove(&j);
                        s.completed(j);
                    }
                }
                Op::Fail(i, oom) => {
                    if let Some((&j, &w)) = running.iter().nth(i % running.len().max(1)) {
                        running.remove(&j);
                        lost.remove(&j);
                        let kind = if oom { FailKind::DeviceOom } else { FailKind::Other };
                        let t = tried.entry(j).or_default();
                        t.push(w);
                        match s.failed(j, kind, "x") {
                            FailOutcome::Retry { attempts, avoid } => {
                                prop_assert!(attempts < 3);
                                prop_assert_eq!(attempts as usize, t.len());
                                let mut want = t.clone();
                                want.dedup();
                                prop_assert_eq!(avoid, want);
                            }
                            FailOutcome::GiveUp { tried: got, .. } => {
                                prop_assert_eq!(got.len(), 3);
                                tried.remove(&j);
                            }
                        }
                    }
                }
            }
            // The policy's view of every live worker matches ours.
            let st = s.stats();
            prop_assert_eq!(st.waiting, 0);
            for l in &st.workers {
                let n = running
                    .iter()
                    .filter(|&(j, &w)| w == l.id && !lost.contains(j))
                    .count();
                prop_assert_eq!(l.running, n, "worker {}", l.id);
            }
        }
        for (j, _) in std::mem::take(&mut running) {
            s.abandon(j);
        }
        let st = s.stats();
        prop_assert_eq!((st.waiting, st.running), (0, 0));
        prop_assert!(st.workers.iter().all(|l| l.running == 0 && l.placed == Resources::ZERO));
    }
}

/// 1,000 task threads with random run times, failures, and workers leaving and joining: every
/// job finishes (no lost wakeup), no job is placed twice at once, and no worker runs more jobs
/// than its slots.
#[test]
fn stress_many_threads_with_churn() {
    const THREADS: u64 = 1000;
    const JOBS_PER_THREAD: u64 = 5;
    const SLOTS: usize = 16;
    let s = Arc::new(SharedPolicy::with_retry(
        Scheduler::new(Config::default()),
        {
            let start = std::time::Instant::now();
            move || start.elapsed().as_secs_f64()
        },
        RetryConfig { max_attempts: 1000 },
    ));
    let ticker = s.spawn_ticker(Duration::from_millis(10));
    let next_worker = Arc::new(AtomicU64::new(0));
    let alive: Arc<Mutex<BTreeSet<u64>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let gone: Arc<Mutex<BTreeSet<u64>>> = Arc::new(Mutex::new(BTreeSet::new()));
    for _ in 0..6 {
        let w = next_worker.fetch_add(1, Ordering::SeqCst);
        alive.lock().unwrap().insert(w);
        s.worker_update(worker(w, SLOTS));
    }
    let load: Arc<Mutex<HashMap<u64, usize>>> = Arc::new(Mutex::new(HashMap::new()));
    let running: Arc<Mutex<BTreeSet<u64>>> = Arc::new(Mutex::new(BTreeSet::new()));
    let finished = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    // Churn: every few milliseconds a worker leaves and a new one joins.
    let churn = {
        let (s, next_worker, alive, gone, stop) = (
            s.clone(),
            next_worker.clone(),
            alive.clone(),
            gone.clone(),
            stop.clone(),
        );
        std::thread::spawn(move || {
            let mut rng = 12345u64;
            while !stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(3));
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let victim = {
                    let a = alive.lock().unwrap();
                    *a.iter().nth((rng >> 33) as usize % a.len()).unwrap()
                };
                alive.lock().unwrap().remove(&victim);
                gone.lock().unwrap().insert(victim);
                s.worker_gone(victim);
                let w = next_worker.fetch_add(1, Ordering::SeqCst);
                alive.lock().unwrap().insert(w);
                s.worker_update(worker(w, SLOTS));
            }
        })
    };
    let threads: Vec<_> = (0..THREADS)
        .map(|t| {
            let (s, load, running, gone, finished) = (
                s.clone(),
                load.clone(),
                running.clone(),
                gone.clone(),
                finished.clone(),
            );
            std::thread::spawn(move || {
                let mut rng = t.wrapping_mul(0x9E3779B97F4A7C15) | 1;
                for k in 0..JOBS_PER_THREAD {
                    let id = t * 1000 + k;
                    loop {
                        let p = s.place(job(id));
                        assert!(running.lock().unwrap().insert(id), "job {id} placed twice");
                        {
                            let mut l = load.lock().unwrap();
                            let n = l.entry(p.worker).or_default();
                            *n += 1;
                            assert!(*n <= SLOTS, "worker {} runs {} jobs", p.worker, n);
                        }
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        std::thread::sleep(Duration::from_micros(rng % 2000));
                        *load.lock().unwrap().get_mut(&p.worker).unwrap() -= 1;
                        running.lock().unwrap().remove(&id);
                        let lost = gone.lock().unwrap().contains(&p.worker);
                        if lost || rng % 10 == 0 {
                            let kind = if lost {
                                FailKind::LinkDied
                            } else {
                                FailKind::DeviceOom
                            };
                            match s.failed(id, kind, "stress") {
                                FailOutcome::Retry { .. } => continue,
                                FailOutcome::GiveUp { .. } => panic!("job {id} gave up"),
                            }
                        }
                        s.completed(id);
                        break;
                    }
                }
                finished.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    while finished.load(Ordering::SeqCst) < THREADS as usize {
        assert!(
            std::time::Instant::now() < deadline,
            "lost wakeup: {} of {THREADS} threads finished, {} waiting in place, stats {:?}",
            finished.load(Ordering::SeqCst),
            s.waiting(),
            s.stats().waiting
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    for t in threads {
        t.join().unwrap();
    }
    stop.store(true, Ordering::SeqCst);
    churn.join().unwrap();
    s.stop_ticker();
    ticker.join().unwrap();
    let st = s.stats();
    assert_eq!((st.waiting, st.running), (0, 0));
    assert!(st.workers.iter().all(|l| l.running == 0));
}

/// The frontier's size: 21 full workers of 16 slots and 1,000 waiting jobs, one completion and
/// one submission per event. `dispatch` stays under a millisecond at the 99th percentile
/// (release builds only; a regression guard, not a benchmark).
#[test]
fn dispatch_p99_at_frontier_size() {
    use sched::Policy;
    if cfg!(debug_assertions) {
        return;
    }
    let mut p = Scheduler::new(Config::default());
    for w in 0..21 {
        p.worker_update(
            WorkerState::new(
                w,
                if w < 7 { "h200" } else { "l40s" },
                16,
                Resources::mem_gb(150.0),
            ),
            0.0,
        );
    }
    let mut next = 0u64;
    let mut running = std::collections::VecDeque::new();
    let mut submit = |p: &mut Scheduler, t: f64| {
        let mut j = JobSpec::new(next, Resources::mem_gb(1.0 + (next % 13) as f64), next / 50);
        j.work = Some(60.0);
        p.submit(j, t);
        next += 1;
    };
    for _ in 0..1000 + 21 * 16 {
        submit(&mut p, 0.0);
    }
    running.extend(p.dispatch(0.0).into_iter().map(|x| x.0));
    let mut times = Vec::new();
    for e in 1..=3000 {
        let t = e as f64;
        if let Some(j) = running.pop_front() {
            p.completed(j, t);
        }
        submit(&mut p, t);
        let start = std::time::Instant::now();
        let out = p.dispatch(t);
        times.push(start.elapsed().as_secs_f64());
        running.extend(out.into_iter().map(|x| x.0));
    }
    assert!(p.stats().waiting >= 900);
    times.sort_by(f64::total_cmp);
    let p99 = times[times.len() * 99 / 100];
    assert!(p99 < 1e-3, "dispatch p99 {:.3} ms", p99 * 1e3);
}
