//! The blocking front end: leases, retries, the ticker, a model check and a many-thread stress.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread::Scope,
    time::Duration,
};

use proptest::prelude::*;
use whelm::{
    Attempt, Config, Defer, Explanation, FailKind, GaveUp, Input, JobId, JobSpec, Lease, Output,
    Policy, PolicyStats, Resources, RetryConfig, Scheduler, SharedPolicy, SpeedConfig, Time,
    WorkerId, WorkerState,
};

/// A worker of class "x".
fn worker(id: u64, slots: usize) -> WorkerState {
    WorkerState {
        id,
        class: "x".into(),
        slots,
        budget: Resources::mem(1 << 40),
        ..Default::default()
    }
}

/// A unit job.
fn job(id: u64) -> JobSpec {
    JobSpec {
        id,
        demand: Resources::mem(1),
        ..Default::default()
    }
}

/// A front end over the default backfill policy.
fn shared() -> SharedPolicy<Scheduler> {
    SharedPolicy::with_system_clock(Scheduler::new(Config::default()))
}

/// `lease` blocks until a worker joins, then returns it.
#[test]
fn lease_blocks_until_a_worker_joins() {
    let s = Arc::new(shared());
    let t = {
        let s = s.clone();
        std::thread::spawn(move || {
            let lease = s.lease(job(1));
            let got = (lease.worker(), lease.attempt());
            lease.complete();
            got
        })
    };
    while s.waiting() == 0 {
        std::thread::yield_now();
    }
    s.worker_update(worker(7, 1));
    assert_eq!(t.join().unwrap(), (7, 1));
    assert_eq!(s.stats().running, 0);
}

/// A timed-out lease withdraws the job.
#[test]
fn timeout_withdraws() {
    let s = shared();
    let back = s
        .lease_timeout(job(1), Duration::from_millis(20))
        .err()
        .unwrap();
    assert_eq!(back.id, 1);
    let st = s.stats();
    assert_eq!((st.waiting, st.running), (0, 0));
    // The id can be leased again.
    s.worker_update(worker(1, 1));
    let lease = s.lease(job(1));
    assert_eq!(lease.worker(), 1);
    lease.complete();
}

/// Retries avoid the workers tried; after `max_attempts` failures the job gives up, retryable iff
/// every attempt was a device OOM; resources are freed each time.
#[test]
fn retries_avoid_tried_workers_then_give_up() {
    let s = shared();
    for w in 1..=5 {
        s.worker_update(worker(w, 1));
    }
    let max = RetryConfig::default().max_attempts;
    let mut tried = Vec::new();
    let mut lease = s.lease(job(9));
    for attempt in 1..=max {
        assert_eq!(lease.attempt(), attempt);
        assert!(
            !tried.contains(&lease.worker()),
            "retried on {} ({tried:?})",
            lease.worker()
        );
        tried.push(lease.worker());
        match lease.fail(FailKind::DeviceOom, "oom") {
            Ok(next) => {
                assert!(attempt < max);
                assert_eq!(s.stats().running, 1);
                lease = next;
            }
            Err(g) => {
                assert_eq!(attempt, max);
                let workers: Vec<WorkerId> = g.tried.iter().map(|t| t.worker).collect();
                assert_eq!((g.job, workers, g.retryable), (9, tried.clone(), true));
                assert_eq!(s.stats().running, 0);
                break;
            }
        }
    }
    // The history is forgotten: a new lease is attempt 1 again.
    let lease = s.lease(job(9));
    assert_eq!(lease.attempt(), 1);
    lease.complete();
    // A mixed history is not retryable.
    let mut lease = s.lease(job(10));
    for kind in [FailKind::DeviceOom, FailKind::LinkDied, FailKind::DeviceOom] {
        lease = lease.fail(kind, "x").unwrap();
    }
    let g = lease.fail(FailKind::DeviceOom, "x").err().unwrap();
    assert_eq!((g.tried.len(), g.retryable), (4, false));
}

/// With one live worker, a retry goes back to it (soft avoid) rather than waiting forever.
#[test]
fn retry_on_the_only_worker() {
    let s = shared();
    s.worker_update(worker(1, 1));
    let lease = s.lease(job(1));
    assert_eq!(lease.worker(), 1);
    let lease = lease.fail(FailKind::DeviceOom, "oom").unwrap();
    assert_eq!((lease.worker(), lease.attempt()), (1, 2));
    lease.complete();
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
    assert_eq!(lease.attempt(), 1);
    lease.complete();
    assert_eq!(s.stats().running, 0);
}

/// Without events, the ticker lets a deferred job go once its deferral expires.
#[test]
fn ticker_releases_timed_waits() {
    let policy = Scheduler::new(Config {
        speed: SpeedConfig {
            defer: Some(Defer {
                max_wait: Duration::from_millis(200),
                min_gain: 0.0,
            }),
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
    let first = s.lease(work(1, Duration::from_secs(10)));
    assert_eq!(first.worker(), 1);
    assert!(
        s.lease_timeout(work(2, Duration::from_secs(20)), Duration::from_millis(50))
            .is_err(),
        "deferred"
    );
    let ticker = s.spawn_ticker(Duration::from_millis(20));
    let start = std::time::Instant::now();
    let second = s.lease(work(2, Duration::from_secs(20)));
    assert_eq!(second.worker(), 2);
    assert!(start.elapsed() < Duration::from_secs(5));
    s.stop_ticker();
    ticker.join().unwrap();
    first.complete();
    second.complete();
}

/// A thread whose worker left reports the lost link after the policy's retry was itself lost
/// with its worker before the thread picked it up: it gets the job's next start, not the lost one.
#[test]
fn lost_retry_is_not_handed_out() {
    let s = SharedPolicy::new(Scheduler::new(Config::default()), || Time::ORIGIN);
    s.worker_update(worker(1, 1));
    s.worker_update(worker(2, 1));
    let lease = s.lease(job(7));
    assert_eq!(lease.worker(), 1);
    assert_eq!(s.worker_gone(1), vec![7]);
    // The retry started on worker 2, which leaves before the thread asks for it.
    assert_eq!(s.worker_gone(2), Vec::<JobId>::new());
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|sc| {
        sc.spawn(move || {
            let next = lease.fail(FailKind::LinkDied, "reset").unwrap();
            tx.send((next.worker(), next.attempt())).unwrap();
            next.complete();
        });
        // No worker is left: the thread waits for one.
        let early = rx.recv_timeout(Duration::from_millis(200)).ok();
        assert_eq!(early, None, "handed a start on a departed worker");
        s.worker_update(worker(3, 1));
        assert_eq!(rx.recv().unwrap(), (3, 3));
    });
}

/// The same when the lost retry was the last allowed attempt: the thread gets the give-up.
#[test]
fn lost_last_retry_gives_up() {
    let config = Config {
        retry: RetryConfig { max_attempts: 2 },
        ..Config::default()
    };
    let s = SharedPolicy::new(Scheduler::new(config), || Time::ORIGIN);
    s.worker_update(worker(1, 1));
    s.worker_update(worker(2, 1));
    let lease = s.lease(job(7));
    s.worker_gone(1);
    s.worker_gone(2);
    let g = lease.fail(FailKind::LinkDied, "reset").err().unwrap();
    assert_eq!((g.job, g.tried.len()), (7, 2));
}

/// The policy under the model check: the default backfill policy, recording what it outputs and
/// counting the failures reported to it.
struct Probe {
    inner: Scheduler,
    outputs: Vec<Output>,
    failures: usize,
}

impl Policy for Probe {
    /// Counted if a failure, then forwarded.
    fn handle(&mut self, input: Input, now: Time) {
        self.failures += matches!(input, Input::Failed { .. }) as usize;
        self.inner.handle(input, now);
    }

    /// Forwarded, and recorded.
    fn poll(&mut self, now: Time) -> Vec<Output> {
        let out = self.inner.poll(now);
        self.outputs.extend(out.iter().cloned());
        out
    }

    /// Forwarded.
    fn next_wakeup(&self) -> Option<Time> {
        self.inner.next_wakeup()
    }

    /// Forwarded.
    fn explain(&self, job: JobId) -> Option<Explanation> {
        self.inner.explain(job)
    }

    /// Forwarded.
    fn stats(&self) -> PolicyStats {
        self.inner.stats()
    }
}

/// Failures before the model's policy gives a job up.
const MAX_ATTEMPTS: u32 = 3;

/// What a job's thread is doing.
enum Thread<'a> {
    /// Holding a lease.
    Held(Lease<'a, Probe>),
    /// Blocked in [`Lease::fail`]; its result comes on the channel.
    Failing(mpsc::Receiver<Result<Lease<'a, Probe>, GaveUp>>),
}

/// A leased job, as the model sees it.
struct Job<'a> {
    thread: Thread<'a>,
    /// The worker of the held attempt left.
    lost: bool,
    /// The policy's live attempt.
    live: Option<(Attempt, WorkerId)>,
    /// The policy gave the job up; the thread has not heard yet.
    gave_up: Option<GaveUp>,
    /// Failed attempts.
    tried: Vec<(WorkerId, FailKind)>,
}

/// The model: live workers and leased jobs, following a [`SharedPolicy`] driven from one thread
/// (plus one thread per [`Lease::fail`], which blocks).
struct Model<'scope, 'env> {
    s: &'env SharedPolicy<Probe>,
    scope: &'scope Scope<'scope, 'env>,
    /// Live workers and their slots.
    workers: BTreeMap<WorkerId, usize>,
    jobs: BTreeMap<JobId, Job<'env>>,
}

/// A random event.
#[derive(Clone, Debug)]
enum Op {
    Join(u64, usize),
    Gone(u64),
    Lease(u64),
    Complete(usize),
    Fail(usize, bool),
}

/// A random [`Op`].
fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        2 => (0u64..5, 0usize..3).prop_map(|(w, s)| Op::Join(w, s)),
        1 => (0u64..5).prop_map(Op::Gone),
        4 => (0u64..6).prop_map(Op::Lease),
        2 => any::<prop::sample::Index>().prop_map(|i| Op::Complete(i.index(1 << 16))),
        3 => (any::<prop::sample::Index>(), any::<bool>())
            .prop_map(|(i, oom)| Op::Fail(i.index(1 << 16), oom)),
    ]
}

impl<'scope, 'env> Model<'scope, 'env> {
    /// The id of the `i`-th job (cyclically) whose thread holds a lease.
    fn held(&self, i: usize) -> Option<JobId> {
        let held: Vec<JobId> = self
            .jobs
            .iter()
            .filter(|j| matches!(j.1.thread, Thread::Held(_)))
            .map(|j| *j.0)
            .collect();
        (!held.is_empty()).then(|| held[i % held.len()])
    }

    /// Take the lease job `id`'s failing thread returns.
    fn pick_up(&mut self, id: JobId) -> Result<Lease<'env, Probe>, GaveUp> {
        match &self.jobs[&id].thread {
            Thread::Failing(rx) => rx.recv().unwrap(),
            Thread::Held(_) => panic!("job {id} is not failing"),
        }
    }

    /// Follow the policy's outputs until there are none.
    fn settle(&mut self) -> Result<(), TestCaseError> {
        loop {
            let out = self.s.with(|p, _| std::mem::take(&mut p.outputs));
            if out.is_empty() {
                return Ok(());
            }
            for o in out {
                self.output(o)?;
            }
        }
    }

    /// Follow one output.
    fn output(&mut self, o: Output) -> Result<(), TestCaseError> {
        match o {
            Output::Start {
                job: id,
                attempt,
                worker: w,
            } => {
                let workers = &self.workers;
                let Some(j) = self.jobs.get_mut(&id) else {
                    return Err(TestCaseError::fail(format!("start of unknown job {id}")));
                };
                if j.live == Some((attempt, w)) {
                    // The start that made its lease.
                    return Ok(());
                }
                prop_assert_eq!(attempt as usize, j.tried.len() + 1);
                prop_assert!(workers.get(&w).is_some_and(|&n| n > 0));
                if j.tried.iter().any(|t| t.0 == w) {
                    // Soft: only while every live worker was tried.
                    prop_assert!(
                        workers
                            .iter()
                            .all(|(v, &n)| n == 0 || j.tried.iter().any(|t| t.0 == *v)),
                        "retried on {} with an untried live worker",
                        w
                    );
                }
                j.live = Some((attempt, w));
                if matches!(j.thread, Thread::Failing(_)) {
                    let lease = self.pick_up(id).map_err(|g| {
                        TestCaseError::fail(format!("started, yet the thread got {g:?}"))
                    })?;
                    prop_assert_eq!((lease.attempt(), lease.worker()), (attempt, w));
                    let j = self.jobs.get_mut(&id).unwrap();
                    j.thread = Thread::Held(lease);
                    j.lost = false;
                }
            }
            Output::GaveUp(g) => {
                let Some(j) = self.jobs.get_mut(&g.job) else {
                    return Err(TestCaseError::fail(format!("unknown job gave up: {g:?}")));
                };
                prop_assert_eq!(g.tried.len(), MAX_ATTEMPTS as usize);
                let tried: Vec<_> = g.tried.iter().map(|t| (t.worker, t.kind)).collect();
                prop_assert_eq!(&tried, &j.tried);
                let oom = tried.iter().all(|t| t.1 == FailKind::DeviceOom);
                prop_assert_eq!(g.retryable, oom);
                j.live = None;
                if matches!(j.thread, Thread::Failing(_)) {
                    let got = self.pick_up(g.job).err();
                    prop_assert_eq!(got.as_ref(), Some(&g));
                    self.jobs.remove(&g.job);
                } else {
                    j.gave_up = Some(g);
                }
            }
            // Stopping the retry of a lost attempt whose thread completed.
            Output::Stop { job, .. } => prop_assert!(!self.jobs.contains_key(&job)),
            o => prop_assert!(false, "unexpected output {:?}", o),
        }
        Ok(())
    }

    /// Apply one event, then follow the outputs.
    fn apply(&mut self, op: &Op) -> Result<(), TestCaseError> {
        match *op {
            Op::Join(w, slots) => {
                self.workers.insert(w, slots);
                self.s.worker_update(worker(w, slots));
            }
            Op::Gone(w) => {
                self.workers.remove(&w);
                let mut expected = Vec::new();
                for (&id, j) in &mut self.jobs {
                    if matches!(&j.thread, Thread::Held(l) if l.worker() == w) {
                        expected.push(id);
                        j.lost = true;
                    }
                    if j.live.is_some_and(|l| l.1 == w) {
                        j.live = None;
                        j.tried.push((w, FailKind::LinkDied));
                    }
                }
                prop_assert_eq!(self.s.worker_gone(w), expected);
            }
            Op::Lease(id) => {
                if !self.jobs.contains_key(&id)
                    && let Ok(lease) = self.s.lease_timeout(job(id), Duration::ZERO)
                {
                    prop_assert_eq!(lease.attempt(), 1);
                    let job = Job {
                        live: Some((1, lease.worker())),
                        thread: Thread::Held(lease),
                        lost: false,
                        gave_up: None,
                        tried: Vec::new(),
                    };
                    self.jobs.insert(id, job);
                }
            }
            Op::Complete(i) => {
                if let Some(id) = self.held(i)
                    && let Thread::Held(lease) = self.jobs.remove(&id).unwrap().thread
                {
                    lease.complete();
                }
            }
            Op::Fail(i, oom) => {
                if let Some(id) = self.held(i) {
                    self.fail(
                        id,
                        if oom {
                            FailKind::DeviceOom
                        } else {
                            FailKind::Other
                        },
                    )?;
                }
            }
        }
        self.settle()?;
        // The policy's view matches the model's.
        let st = self.s.stats();
        let waiting = self
            .jobs
            .values()
            .filter(|j| j.live.is_none() && j.gave_up.is_none());
        prop_assert_eq!(st.waiting, waiting.count());
        for l in &st.workers {
            let n = self
                .jobs
                .values()
                .filter(|j| j.live.is_some_and(|a| a.1 == l.id));
            prop_assert_eq!(l.running, n.count(), "worker {}", l.id);
        }
        let failing = self
            .jobs
            .values()
            .filter(|j| matches!(j.thread, Thread::Failing(_)));
        prop_assert_eq!(self.s.waiting(), failing.count(), "threads blocked");
        Ok(())
    }

    /// Fail job `id`'s held attempt from a thread of its own, and wait until the policy has heard.
    fn fail(&mut self, id: JobId, kind: FailKind) -> Result<(), TestCaseError> {
        let j = self.jobs.get_mut(&id).unwrap();
        let (tx, rx) = mpsc::channel();
        let Thread::Held(lease) = std::mem::replace(&mut j.thread, Thread::Failing(rx)) else {
            unreachable!("only held jobs fail");
        };
        if !j.lost {
            j.tried.push((lease.worker(), kind));
            j.live = None;
        }
        let before = self.s.with(|p, _| p.failures);
        self.scope.spawn(move || {
            let _ = tx.send(lease.fail(kind, "x"));
        });
        while self.s.with(|p, _| p.failures) == before {
            std::thread::yield_now();
        }
        let j = self.jobs.get_mut(&id).unwrap();
        if std::mem::take(&mut j.lost) {
            // The policy failed the attempt when its worker left: the thread gets what followed.
            if let Some(g) = j.gave_up.take() {
                prop_assert_eq!(self.pick_up(id).err(), Some(g));
                self.jobs.remove(&id);
            } else if let Some((attempt, w)) = j.live {
                let lease = self.pick_up(id).map_err(|g| {
                    TestCaseError::fail(format!("retry started, yet the thread got {g:?}"))
                })?;
                prop_assert_eq!((lease.attempt(), lease.worker()), (attempt, w));
                self.jobs.get_mut(&id).unwrap().thread = Thread::Held(lease);
            }
        }
        Ok(())
    }
}

proptest! {
    /// Attempt counting, the soft avoid rule, the outcome of a lost attempt's failure, and no
    /// leaks, over random sequences. Leases use a zero timeout (a job not started at once is
    /// withdrawn) and the clock stands still, so every start follows from an event.
    #[test]
    fn attempts_avoid_and_no_leaks(ops in prop::collection::vec(op(), 1..120)) {
        let config = Config { retry: RetryConfig { max_attempts: MAX_ATTEMPTS }, ..Config::default() };
        let probe = Probe { inner: Scheduler::new(config), outputs: Vec::new(), failures: 0 };
        let s = SharedPolicy::new(probe, || Time::ORIGIN);
        std::thread::scope(|scope| {
            let mut m = Model { s: &s, scope, workers: BTreeMap::new(), jobs: BTreeMap::new() };
            let r = ops.iter().try_for_each(|op| m.apply(op));
            // Dropping the leases cancels their jobs; a fresh worker unblocks failing threads,
            // whose leases are then dropped too.
            drop(m);
            s.worker_update(worker(99, 1 << 20));
            r
        })?;
        let st = s.stats();
        prop_assert_eq!((st.waiting, st.running), (0, 0));
        prop_assert!(st.workers.iter().all(|l| l.running == 0 && l.placed == Resources::ZERO));
    }
}

/// `THREADS` task threads with random run times, failures, and workers leaving (while threads hold
/// leases there, which then report the lost link) and joining: every job finishes (no lost
/// wakeup), no job runs twice at once, retries count up, and no worker runs more jobs than its
/// slots.
#[test]
fn stress_many_threads_with_churn() {
    const THREADS: u64 = 1000;
    const JOBS_PER_THREAD: u64 = 5;
    const SLOTS: usize = 16;
    let config = Config {
        retry: RetryConfig { max_attempts: 1000 },
        ..Config::default()
    };
    let s = Arc::new(SharedPolicy::with_system_clock(Scheduler::new(config)));
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
    let link_died = Arc::new(AtomicUsize::new(0));
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
            let mut hit = 0;
            while !stop.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(3));
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let victim = {
                    let a = alive.lock().unwrap();
                    *a.iter().nth((rng >> 33) as usize % a.len()).unwrap()
                };
                alive.lock().unwrap().remove(&victim);
                // Marked gone before the policy hears, so that a thread whose lease is hit
                // reports the lost link.
                gone.lock().unwrap().insert(victim);
                hit += s.worker_gone(victim).len();
                let w = next_worker.fetch_add(1, Ordering::SeqCst);
                alive.lock().unwrap().insert(w);
                s.worker_update(worker(w, SLOTS));
            }
            hit
        })
    };
    let threads: Vec<_> = (0..THREADS)
        .map(|t| {
            let (s, load, running, gone, finished, link_died) = (
                s.clone(),
                load.clone(),
                running.clone(),
                gone.clone(),
                finished.clone(),
                link_died.clone(),
            );
            std::thread::spawn(move || {
                let mut rng = t.wrapping_mul(0x9E3779B97F4A7C15) | 1;
                for k in 0..JOBS_PER_THREAD {
                    let id = t * 1000 + k;
                    let mut lease = s.lease(job(id));
                    let mut attempt = 0;
                    loop {
                        assert!(lease.attempt() > attempt, "job {id} attempt went back");
                        attempt = lease.attempt();
                        let w = lease.worker();
                        assert!(running.lock().unwrap().insert(id), "job {id} placed twice");
                        {
                            let mut l = load.lock().unwrap();
                            let n = l.entry(w).or_default();
                            *n += 1;
                            assert!(*n <= SLOTS, "worker {w} runs {n} jobs");
                        }
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        std::thread::sleep(Duration::from_micros(rng % 2000));
                        *load.lock().unwrap().get_mut(&w).unwrap() -= 1;
                        running.lock().unwrap().remove(&id);
                        let lost = gone.lock().unwrap().contains(&w);
                        if lost || rng % 10 == 0 {
                            let kind = if lost {
                                link_died.fetch_add(1, Ordering::SeqCst);
                                FailKind::LinkDied
                            } else {
                                FailKind::DeviceOom
                            };
                            match lease.fail(kind, "stress") {
                                Ok(next) => {
                                    lease = next;
                                    continue;
                                }
                                Err(g) => panic!("job {id} gave up: {g:?}"),
                            }
                        }
                        lease.complete();
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
            "lost wakeup: {} of {THREADS} threads finished, {} waiting in lease, stats {:?}",
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
    let hit = churn.join().unwrap();
    s.stop_ticker();
    ticker.join().unwrap();
    // A thread that checked `gone` just before its worker left completes instead of failing
    // (allowed: completing after the worker left cancels the retry), so `hit` only bounds the
    // lost links from above.
    assert!(hit > 0, "no worker left while a lease was held there");
    assert!(
        link_died.load(Ordering::SeqCst) > 0,
        "no thread reported a lost link"
    );
    let st = s.stats();
    assert_eq!((st.waiting, st.running), (0, 0));
    assert!(st.workers.iter().all(|l| l.running == 0));
}

/// A production-sized frontier: every worker full and a long queue waiting, one completion and
/// one submission per event. `poll`'s 99th percentile stays under the asserted bound (release
/// builds only; a regression guard, not a benchmark).
#[test]
fn poll_p99_at_frontier_size() {
    if cfg!(debug_assertions) {
        return;
    }
    let mut p = Scheduler::new(Config::default());
    for w in 0..21 {
        let class = if w < 7 { "h200" } else { "l40s" };
        let w = WorkerState {
            id: w,
            class: class.into(),
            slots: 16,
            budget: Resources::mem_gb(150.0),
            ..Default::default()
        };
        p.handle(Input::Worker(w), Time::ORIGIN);
    }
    let mut next = 0u64;
    let mut running = std::collections::VecDeque::new();
    let mut submit = |p: &mut Scheduler, t: Time| {
        let j = JobSpec {
            id: next,
            demand: Resources::mem_gb(1.0 + (next % 13) as f64),
            group: next / 50,
            work: Some(Duration::from_secs(60)),
            ..Default::default()
        };
        p.handle(Input::Submit(j), t);
        next += 1;
    };
    /// The attempts a poll started.
    fn started(out: Vec<Output>) -> impl Iterator<Item = (JobId, Attempt)> {
        out.into_iter().filter_map(|o| match o {
            Output::Start { job, attempt, .. } => Some((job, attempt)),
            _ => None,
        })
    }
    for _ in 0..1000 + 21 * 16 {
        submit(&mut p, Time::ORIGIN);
    }
    running.extend(started(p.poll(Time::ORIGIN)));
    let mut times = Vec::new();
    for e in 1..=3000 {
        let t = Time(Duration::from_secs(e));
        if let Some((job, attempt)) = running.pop_front() {
            p.handle(Input::Done { job, attempt }, t);
        }
        submit(&mut p, t);
        let start = std::time::Instant::now();
        let out = p.poll(t);
        times.push(start.elapsed().as_secs_f64());
        running.extend(started(out));
    }
    assert!(p.stats().waiting >= 900);
    times.sort_by(f64::total_cmp);
    let p99 = times[times.len() * 99 / 100];
    assert!(p99 < 1e-3, "poll p99 {:.3} ms", p99 * 1e3);
}
