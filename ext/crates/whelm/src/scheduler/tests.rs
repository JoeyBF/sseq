//! Behaviour of the whole [`Scheduler`], driven through [`Policy`].

use std::time::Duration;

use crate::{
    Attempt, Config, Constraint, DEVICE_MEMORY, FailKind, Input, JobId, JobSpec, MEMORY, OrderTerm,
    Output, Policy, Rejection, Reservations, Resource, Resources, RetryConfig, SLOTS, Scheduler,
    ScoreTerm, Speculate, SpeedConfig, Time, Verdict, WorkerId, WorkerState,
};

const GB: u64 = 1_000_000_000;

/// A worker of class "x" with a memory capacity in GB.
fn worker(id: WorkerId, slots: u64, budget_gb: u64) -> WorkerState {
    WorkerState {
        id,
        class: "x".into(),
        capacity: Resources::new()
            .with(MEMORY, budget_gb * GB)
            .with(SLOTS, slots),
        ..Default::default()
    }
}

/// A job with a demand in GB.
fn job(gb: u64, group: u64) -> JobSpec {
    JobSpec {
        demand: Resources::new().with(MEMORY, gb * GB),
        group,
        ..Default::default()
    }
}

/// Handle `inputs` at `t`, then poll.
fn feed(p: &mut Scheduler, t: Time, inputs: impl IntoIterator<Item = Input>) -> Vec<Output> {
    for i in inputs {
        p.handle(i, t);
    }
    p.poll(t)
}

/// The `(job, worker)` of each start.
fn starts(out: &[Output]) -> Vec<(JobId, WorkerId)> {
    out.iter()
        .filter_map(|o| match *o {
            Output::Start { job, worker, .. } => Some((job, worker)),
            _ => None,
        })
        .collect()
}

/// A submission.
fn submit(job: JobId, spec: JobSpec) -> Input {
    Input::Submit { job, spec }
}

/// A done message.
fn done(job: JobId, attempt: Attempt) -> Input {
    Input::Done { job, attempt }
}

/// A failure message of kind `kind`.
fn fail(job: JobId, attempt: Attempt, kind: FailKind) -> Input {
    Input::Failed {
        job,
        attempt,
        kind,
        why: "test".into(),
    }
}

/// FIFO spreads jobs over the least loaded workers in arrival order.
#[test]
fn fifo_fills_least_loaded_first() {
    let mut p = Scheduler::new(Config::fifo());
    let mut inputs = vec![
        Input::Worker(worker(1, 4, 100)),
        Input::Worker(worker(2, 4, 100)),
    ];
    inputs.extend((0..4).map(|i| submit(i, job(10, 0))));
    let out = feed(&mut p, Time::ORIGIN, inputs);
    assert_eq!(starts(&out), vec![(0, 1), (1, 2), (2, 1), (3, 2)]);
    assert!(
        out.iter()
            .all(|o| matches!(o, Output::Start { attempt: 1, .. }))
    );
}

/// A preferred worker is chosen over a less loaded one.
#[test]
fn preference_wins_over_load() {
    let mut p = Scheduler::new(Config::default());
    let inputs = [
        Input::Worker(worker(1, 4, 100)),
        Input::Worker(worker(2, 4, 100)),
        submit(0, job(10, 0)),
    ];
    feed(&mut p, Time::ORIGIN, inputs);
    let j = JobSpec {
        constraints: vec![Constraint::prefer_worker(1)],
        ..job(10, 0)
    };
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, [submit(1, j)])),
        vec![(1, 1)]
    );
}

/// Explicit priority first, then the oldest group, then FIFO.
#[test]
fn priority_order_is_group_arrival_then_fifo() {
    let mut p = Scheduler::new(Config::default());
    p.handle(submit(10, job(1, 7)), Time::ORIGIN); // group 7 arrives first
    p.handle(submit(11, job(1, 3)), Time(Duration::from_secs(1)));
    p.handle(submit(12, job(1, 7)), Time(Duration::from_secs(2)));
    let mut urgent = job(1, 3);
    urgent.priority = Some(-1);
    p.handle(submit(13, urgent), Time(Duration::from_secs(3)));
    p.handle(
        Input::Worker(worker(1, 1, 100)),
        Time(Duration::from_secs(4)),
    );
    let mut order = Vec::new();
    let mut finished = Vec::new();
    for t in 0..4 {
        let out = starts(&feed(
            &mut p,
            Time(Duration::from_secs(5 + t)),
            finished.drain(..),
        ));
        assert_eq!(out.len(), 1);
        order.push(out[0].0);
        finished.push(done(out[0].0, 1));
    }
    assert_eq!(order, vec![13, 10, 12, 11]);
}

/// Best fit picks the worker left with the smallest share of its capacity free.
#[test]
fn best_fit_packs_tightly() {
    let mut p = Scheduler::new(Config::best_fit());
    let inputs = [
        Input::Worker(worker(1, 4, 100)),
        Input::Worker(worker(2, 4, 50)),
        submit(0, job(1, 0)),
        submit(1, job(1, 0)),
    ];
    // Both empty workers admit; the smaller one is the tighter fit.
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, inputs)),
        vec![(0, 2), (1, 2)]
    );
    let inputs = [submit(2, job(50, 0)), submit(3, job(1, 0))];
    // Then worker 1 has 49 GB free (49%), worker 2 has 47 GB (94%): worker 1 is fuller.
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, inputs)),
        vec![(2, 1), (3, 1)]
    );
}

/// The tightest fit compares the bottleneck dimension.
#[test]
fn best_fit_ranks_by_the_bottleneck() {
    let mut p = Scheduler::new(Config::best_fit());
    // Worker 1 has most of its memory free but 20% of its device pool; worker 2 has 40% of
    // its memory free and all of its device pool.
    let w1 = WorkerState {
        id: 1,
        class: "x".into(),
        capacity: Resources::new()
            .with(MEMORY, 100 * GB)
            .with(DEVICE_MEMORY, 10 * GB)
            .with(SLOTS, 4),
        reported_used: Resources::new().with(DEVICE_MEMORY, 8 * GB),
        ..Default::default()
    };
    let w2 = WorkerState {
        id: 2,
        class: "x".into(),
        capacity: Resources::new()
            .with(MEMORY, 100 * GB)
            .with(DEVICE_MEMORY, 100 * GB)
            .with(SLOTS, 4),
        reported_used: Resources::new().with(MEMORY, 59 * GB),
        ..Default::default()
    };
    let inputs = [Input::Worker(w1), Input::Worker(w2), submit(0, job(1, 0))];
    assert_eq!(starts(&feed(&mut p, Time::ORIGIN, inputs)), vec![(0, 1)]);
}

/// A failed job is retried before less urgent jobs submitted after it, and keeps its age.
#[test]
fn retry_keeps_place_and_age() {
    let mut p = Scheduler::new(Config::fifo());
    let inputs = [
        Input::Worker(worker(1, 1, 100)),
        submit(0, job(1, 0)),
        submit(1, job(1, 0)),
    ];
    assert_eq!(starts(&feed(&mut p, Time::ORIGIN, inputs)), vec![(0, 1)]);
    assert_eq!(p.stats().longest_wait, Some((1, Duration::ZERO)));
    // Job 0 avoids worker 1 softly; it is the only live worker, so job 0 goes back there
    // ahead of job 1.
    let out = feed(
        &mut p,
        Time(Duration::from_secs(5)),
        [fail(0, 1, FailKind::Other)],
    );
    assert_eq!(
        out,
        vec![Output::Start {
            job: 0,
            attempt: 2,
            worker: 1
        }]
    );
    // Between the failure and the restart, job 0 counted as waiting since its submission.
    let mut q = Scheduler::new(Config::fifo());
    let inputs = [
        Input::Worker(worker(1, 1, 100)),
        Input::Worker(worker(2, 1, 100)),
        submit(0, job(1, 0)),
        submit(1, job(1, 0)),
        submit(2, job(1, 0)),
    ];
    assert_eq!(
        starts(&feed(&mut q, Time::ORIGIN, inputs)),
        vec![(0, 1), (1, 2)]
    );
    // Worker 1 frees, but job 0 avoids it while worker 2 lives: job 2 backfills it.
    let out = feed(
        &mut q,
        Time(Duration::from_secs(5)),
        [fail(0, 1, FailKind::Other)],
    );
    assert_eq!(starts(&out), vec![(2, 1)]);
    assert_eq!(q.stats().longest_wait, Some((0, Duration::from_secs(5))));
    let e = q.explain(0).unwrap();
    let w = e.waiting().unwrap();
    assert_eq!(
        w.tried.iter().map(|t| t.worker).collect::<Vec<_>>(),
        [1],
        "{e}"
    );
    // The avoidance of worker 1 holds while worker 2 lives, and excludes it like a constraint.
    assert_eq!(w.workers, [(1, Verdict::Ineligible), (2, slots_full())]);
    let out = feed(&mut q, Time(Duration::from_secs(6)), [done(1, 1)]);
    assert_eq!(
        out,
        vec![Output::Start {
            job: 0,
            attempt: 2,
            worker: 2
        }]
    );
}

/// Each failure adds its worker to the avoid list; once every live worker is avoided the soft
/// list lapses; after `max_attempts` failures the job is given up.
#[test]
fn avoid_grows_then_gives_up() {
    let mut p = Scheduler::new(Config {
        retry: RetryConfig { max_attempts: 4 },
        ..Config::fifo()
    });
    let mut inputs: Vec<Input> = (1..=3).map(|w| Input::Worker(worker(w, 1, 100))).collect();
    inputs.push(submit(0, job(1, 0)));
    let mut out = feed(&mut p, Time::ORIGIN, inputs);
    let mut seen = Vec::new();
    for attempt in 1..=4 {
        let [
            Output::Start {
                job: 0,
                attempt: a,
                worker,
            },
        ] = out[..]
        else {
            panic!("expected one start, got {out:?}");
        };
        assert_eq!(a, attempt);
        seen.push(worker);
        out = feed(
            &mut p,
            Time(Duration::from_secs(attempt.into())),
            [fail(0, a, FailKind::DeviceOom)],
        );
    }
    // Three distinct workers, then the lapsed soft list allows any (the first).
    assert_eq!(seen, vec![1, 2, 3, 1]);
    let [Output::GaveUp(g)] = &out[..] else {
        panic!("expected a give-up, got {out:?}");
    };
    assert_eq!(g.job, 0);
    assert_eq!(g.tried.len(), 4);
    assert!(g.retryable);
    assert_eq!(p.stats().waiting + p.stats().running, 0);
    assert_eq!(p.explain(0), None);
}

/// Retries avoid the workers tried softly, but the caller's Forbid stays hard.
#[test]
fn forbid_survives_retries() {
    let mut p = Scheduler::new(Config::fifo());
    let j = JobSpec {
        constraints: vec![Constraint::forbid_worker(1)],
        ..job(1, 0)
    };
    let inputs = [
        Input::Worker(worker(1, 1, 100)),
        Input::Worker(worker(2, 1, 100)),
        submit(0, j),
    ];
    assert_eq!(starts(&feed(&mut p, Time::ORIGIN, inputs)), vec![(0, 2)]);
    // Worker 2 is now avoided softly; the only other live worker is forbidden, so the
    // avoidance lapses and the retry goes back to 2.
    let out = feed(
        &mut p,
        Time(Duration::from_secs(1)),
        [fail(0, 1, FailKind::Other)],
    );
    assert_eq!(
        out,
        vec![Output::Start {
            job: 0,
            attempt: 2,
            worker: 2
        }]
    );
}

/// The next wakeup covers aging and reservation deadlines, not only deferrals.
#[test]
fn next_wakeup_reports_aging_and_reserving() {
    let cfg = Config {
        age_limit: Some(Duration::from_secs(100)),
        ..Config::default()
    };
    let reserve_after = cfg.reservations.as_ref().unwrap().reserve_after;
    let mut p = Scheduler::new(cfg);
    assert_eq!(p.next_wakeup(), None);
    feed(
        &mut p,
        Time(Duration::from_secs(10)),
        [submit(0, job(1, 0))],
    );
    let (age_limit, ten) = (Duration::from_secs(100), Time(Duration::from_secs(10)));
    assert_eq!(p.next_wakeup(), Some(ten + reserve_after.min(age_limit)));
    let later = ten + reserve_after.max(age_limit);
    p.poll(ten + reserve_after.min(age_limit));
    assert_eq!(
        p.next_wakeup(),
        (reserve_after != age_limit).then_some(later)
    );
    p.poll(later);
    assert_eq!(p.next_wakeup(), None);
}

/// A give-up is retryable only if every attempt ran out of device memory.
#[test]
fn give_up_is_retryable_only_for_device_oom() {
    let mut p = Scheduler::new(Config {
        retry: RetryConfig { max_attempts: 2 },
        ..Config::fifo()
    });
    let inputs = [Input::Worker(worker(1, 1, 100)), submit(0, job(1, 0))];
    feed(&mut p, Time::ORIGIN, inputs);
    feed(
        &mut p,
        Time(Duration::from_secs(1)),
        [fail(0, 1, FailKind::DeviceOom)],
    );
    let out = feed(
        &mut p,
        Time(Duration::from_secs(2)),
        [fail(0, 2, FailKind::Timeout)],
    );
    let [Output::GaveUp(g)] = &out[..] else {
        panic!("expected a give-up, got {out:?}");
    };
    assert!(!g.retryable);
    assert_eq!(
        g.tried.iter().map(|t| t.kind).collect::<Vec<_>>(),
        vec![FailKind::DeviceOom, FailKind::Timeout]
    );
}

/// A departing worker fails its attempts with `LinkDied`: the jobs are retried elsewhere
/// without the caller resubmitting, or given up when out of attempts.
#[test]
fn worker_gone_fails_and_requeues() {
    let mut p = Scheduler::new(Config {
        retry: RetryConfig { max_attempts: 2 },
        ..Config::fifo()
    });
    let inputs = [
        Input::Worker(worker(1, 2, 100)),
        submit(0, job(1, 0)),
        submit(1, job(1, 0)),
    ];
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, inputs)),
        vec![(0, 1), (1, 1)]
    );
    let out = feed(&mut p, Time(Duration::from_secs(1)), [Input::WorkerGone(1)]);
    assert!(out.is_empty());
    assert_eq!((p.stats().waiting, p.stats().running), (2, 0));
    let out = feed(
        &mut p,
        Time(Duration::from_secs(2)),
        [Input::Worker(worker(2, 2, 100))],
    );
    assert_eq!(starts(&out), vec![(0, 2), (1, 2)]);
    assert!(
        out.iter()
            .all(|o| matches!(o, Output::Start { attempt: 2, .. }))
    );
    // The second loss exhausts both jobs' attempts.
    let out = feed(&mut p, Time(Duration::from_secs(3)), [Input::WorkerGone(2)]);
    assert_eq!(out.len(), 2);
    for o in &out {
        let Output::GaveUp(g) = o else {
            panic!("expected give-ups, got {out:?}")
        };
        assert!(g.tried.iter().all(|t| t.kind == FailKind::LinkDied));
        assert_eq!(g.tried[0].why, "worker 1 left");
    }
}

/// Messages about attempts that are not live change nothing.
#[test]
fn stale_messages_are_ignored() {
    let mut p = Scheduler::new(Config::fifo());
    let inputs = [Input::Worker(worker(1, 1, 100)), submit(0, job(1, 0))];
    feed(&mut p, Time::ORIGIN, inputs);
    feed(
        &mut p,
        Time(Duration::from_secs(1)),
        [fail(0, 1, FailKind::Other)],
    );
    // Attempt 2 runs; reports about attempt 1, unknown attempts and unknown jobs are stale.
    let stale = [
        done(0, 1),
        fail(0, 1, FailKind::Other),
        done(0, 7),
        done(9, 1),
        fail(9, 1, FailKind::Other),
        Input::Cancel(9),
        Input::WorkerGone(9),
    ];
    assert!(feed(&mut p, Time(Duration::from_secs(2)), stale).is_empty());
    let st = p.stats();
    assert_eq!((st.waiting, st.running, st.workers[0].running), (0, 1, 1));
    assert!(feed(&mut p, Time(Duration::from_secs(3)), [done(0, 2)]).is_empty());
    let st = p.stats();
    assert_eq!((st.running, &st.workers[0].placed), (0, &Resources::new()));
    // A late duplicate of the winning report is stale too.
    assert!(feed(&mut p, Time(Duration::from_secs(4)), [done(0, 2)]).is_empty());
    assert_eq!(p.explain(0), None);
}

/// Cancelling a running job stops its attempt; a waiting one is dropped silently.
#[test]
fn cancel_stops_running_attempts() {
    let mut p = Scheduler::new(Config::fifo());
    let inputs = [
        Input::Worker(worker(1, 1, 100)),
        submit(0, job(1, 0)),
        submit(1, job(1, 0)),
    ];
    feed(&mut p, Time::ORIGIN, inputs);
    assert!(feed(&mut p, Time(Duration::from_secs(1)), [Input::Cancel(1)]).is_empty());
    let out = feed(&mut p, Time(Duration::from_secs(2)), [Input::Cancel(0)]);
    assert_eq!(
        out,
        vec![Output::Stop {
            job: 0,
            attempt: 1,
            worker: 1
        }]
    );
    let st = p.stats();
    assert_eq!((st.waiting, st.running, st.workers[0].running), (0, 0, 0));
}

/// A slow worker (speed 1) and, later, an idle fast one (speed 4), with speculation.
fn speculating() -> Scheduler {
    let mut cfg = Config::fifo();
    cfg.speed.speculate = Some(Speculate::default());
    let mut p = Scheduler::new(cfg);
    let mut j = job(1, 0);
    j.work = Some(Duration::from_secs(100));
    let inputs = [Input::Worker(worker(1, 1, 100)), submit(0, j)];
    assert_eq!(starts(&feed(&mut p, Time::ORIGIN, inputs)), vec![(0, 1)]);
    let fast = WorkerState {
        speed: 4.0,
        ..worker(2, 1, 100)
    };
    let out = feed(&mut p, Time(Duration::from_secs(1)), [Input::Worker(fast)]);
    assert_eq!(
        out,
        vec![Output::Start {
            job: 0,
            attempt: 2,
            worker: 2
        }]
    );
    let st = p.stats();
    assert_eq!(st.running, 1);
    assert_eq!(
        st.workers.iter().map(|w| w.running).collect::<Vec<_>>(),
        vec![1, 1]
    );
    // At most one speculative attempt per job.
    assert!(p.poll(Time(Duration::from_secs(2))).is_empty());
    p
}

/// An idle fast worker starts a second attempt of a job on a slow one; the first to finish
/// wins and the other is stopped.
#[test]
fn speculation_first_done_wins() {
    let mut p = speculating();
    let out = feed(&mut p, Time(Duration::from_secs(26)), [done(0, 2)]);
    assert_eq!(
        out,
        vec![Output::Stop {
            job: 0,
            attempt: 1,
            worker: 1
        }]
    );
    assert!(p.stats().workers.iter().all(|w| w.running == 0));
    assert!(feed(&mut p, Time(Duration::from_secs(27)), [done(0, 1)]).is_empty());

    let mut p = speculating();
    let out = feed(&mut p, Time(Duration::from_secs(10)), [done(0, 1)]);
    assert_eq!(
        out,
        vec![Output::Stop {
            job: 0,
            attempt: 2,
            worker: 2
        }]
    );
}

/// A failing speculative attempt leaves the original running; cancelling stops both.
#[test]
fn speculation_failure_and_cancel() {
    let mut p = speculating();
    assert!(
        feed(
            &mut p,
            Time(Duration::from_secs(5)),
            [fail(0, 2, FailKind::Other)]
        )
        .is_empty()
    );
    assert_eq!(p.stats().running, 1);
    let out = feed(&mut p, Time(Duration::from_secs(6)), [Input::Cancel(0)]);
    assert_eq!(
        out,
        vec![Output::Stop {
            job: 0,
            attempt: 1,
            worker: 1
        }]
    );

    let mut p = speculating();
    let out = feed(&mut p, Time(Duration::from_secs(5)), [Input::Cancel(0)]);
    assert_eq!(starts(&out), vec![]);
    assert_eq!(out.len(), 2);
}

/// A failed speculative attempt does not count against the retry limit: with two rounds
/// allowed, the original's failure after it is retried rather than given up.
#[test]
fn speculative_failure_is_not_a_round() {
    let mut p = speculating();
    p.config.retry.max_attempts = 2;
    assert!(
        feed(
            &mut p,
            Time(Duration::from_secs(5)),
            [fail(0, 2, FailKind::Other)]
        )
        .is_empty()
    );
    let out = feed(
        &mut p,
        Time(Duration::from_secs(6)),
        [fail(0, 1, FailKind::Other)],
    );
    assert!(
        matches!(
            out[..],
            [Output::Start {
                job: 0,
                attempt: 3,
                ..
            }]
        ),
        "{out:?}"
    );
    let out = feed(
        &mut p,
        Time(Duration::from_secs(7)),
        [fail(0, 3, FailKind::Other)],
    );
    assert!(
        matches!(&out[..], [Output::GaveUp(g)] if g.tried.len() == 3),
        "{out:?}"
    );
}

/// The order in which a one-slot worker runs `jobs`, submitted as jobs 0, 1, ... before it
/// joins.
fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
    let mut p = Scheduler::new(config);
    let n = jobs.len();
    let submits = (0..).zip(jobs).map(|(id, spec)| submit(id, spec));
    feed(&mut p, Time::ORIGIN, submits);
    p.handle(Input::Worker(worker(1, 1, 100)), Time::ORIGIN);
    let mut order = Vec::new();
    for t in 0..n {
        let t = Time(Duration::from_secs(t as u64));
        let out = starts(&p.poll(t));
        assert_eq!(out.len(), 1, "{out:?}");
        order.push(out[0].0);
        p.handle(done(out[0].0, 1), t + Duration::from_millis(500));
    }
    order
}

/// Smith's rule: largest weight over work first; jobs without work last, in arrival order.
#[test]
fn wspt_orders_by_weight_over_work() {
    let spec = |weight, work| JobSpec {
        weight,
        work,
        ..job(1, 0)
    };
    let jobs = vec![
        spec(1.0, None),
        spec(1.0, Some(Duration::from_secs(10))),
        spec(3.0, Some(Duration::from_secs(10))),
        spec(1.0, Some(Duration::from_secs(2))),
        spec(1.0, None),
    ];
    let order = run_order(Config::weighted_completion(), jobs);
    assert_eq!(order, vec![3, 2, 1, 0, 4]);
}

/// Jackson's rule: earliest due date first; jobs without one last.
#[test]
fn edd_orders_by_due_date() {
    let spec = |due| JobSpec { due, ..job(1, 0) };
    let jobs = vec![
        spec(None),
        spec(Some(Time(Duration::from_secs(50)))),
        spec(Some(Time(Duration::from_secs(3)))),
    ];
    assert_eq!(run_order(Config::lateness(), jobs), vec![2, 1, 0]);
}

/// The default order: explicit priority, then group arrival, ranks ignored; with
/// [`OrderTerm::Rank`] between them, largest rank first. A repeated term changes nothing.
#[test]
fn default_order_is_priority_group() {
    let spec = |group, priority, rank| JobSpec {
        priority,
        rank,
        ..job(1, group)
    };
    let jobs = || {
        vec![
            spec(5, None, None),
            spec(6, None, Some(Duration::from_secs(2))),
            spec(6, None, Some(Duration::from_secs(9))),
            spec(5, Some(-1), None),
        ]
    };
    assert_eq!(run_order(Config::default(), jobs()), vec![3, 0, 1, 2]);
    let mut repeated = Config::default();
    repeated
        .order
        .extend([OrderTerm::Priority, OrderTerm::Group]);
    assert_eq!(run_order(repeated, jobs()), vec![3, 0, 1, 2]);
    let ranked = Config {
        order: vec![OrderTerm::Priority, OrderTerm::Rank, OrderTerm::Group],
        ..Config::default()
    };
    assert_eq!(run_order(ranked, jobs()), vec![3, 2, 1, 0]);
    // Group before rank: the older group first.
    let group_first = Config {
        order: vec![OrderTerm::Group, OrderTerm::Rank],
        ..Config::default()
    };
    assert_eq!(run_order(group_first, jobs()), vec![0, 3, 2, 1]);
}

/// Requires of one kind are alternatives; of different kinds, all must hold. Forbid wins.
#[test]
fn requires_and_forbids() {
    let mut p = Scheduler::new(Config::fifo());
    let mut inputs: Vec<Input> = [(1, "a"), (2, "b"), (3, "c")]
        .into_iter()
        .map(|(id, class)| {
            Input::Worker(WorkerState {
                id,
                class: class.into(),
                capacity: Resources::new().with(MEMORY, 100 * GB).with(SLOTS, 4),
                ..Default::default()
            })
        })
        .collect();
    let constrained = |constraints| JobSpec {
        constraints,
        ..job(1, 0)
    };
    let either = constrained(vec![
        Constraint::require_class("c"),
        Constraint::require_class("b"),
    ]);
    let both = constrained(vec![
        Constraint::require_class("a"),
        Constraint::require_worker(2),
    ]);
    let forbidden = constrained(vec![
        Constraint::require_class("a"),
        Constraint::forbid_class("a"),
    ]);
    let worker_or = constrained(vec![
        Constraint::require_worker(3),
        Constraint::require_worker(1),
        Constraint::forbid_worker(1),
    ]);
    let jobs = [either, both, forbidden, worker_or];
    inputs.extend((0..).zip(jobs).map(|(id, spec)| submit(id, spec)));
    // Job 0 takes the less loaded of b and c; jobs 1 and 2 match nothing.
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, inputs)),
        vec![(0, 2), (3, 3)]
    );
    let e = p.explain(1).unwrap();
    let w = e.waiting().unwrap();
    assert!(w.workers.iter().all(|w| w.1 == Verdict::Ineligible), "{e}");
    assert_eq!(w.workers.len(), 3);
    assert!(
        e.to_string()
            .ends_with("; 3 worker(s) excluded by its constraints"),
        "{e}"
    );
}

/// A Prefer on a class ranks the whole class first; Loosest picks the emptiest worker.
#[test]
fn prefer_class_and_loosest() {
    let mut p = Scheduler::new(Config::default());
    let inputs = [
        Input::Worker(WorkerState {
            id: 1,
            class: "a".into(),
            capacity: Resources::new().with(MEMORY, 100 * GB).with(SLOTS, 4),
            ..Default::default()
        }),
        Input::Worker(WorkerState {
            id: 2,
            class: "b".into(),
            capacity: Resources::new().with(MEMORY, 100 * GB).with(SLOTS, 4),
            ..Default::default()
        }),
        submit(
            0,
            JobSpec {
                constraints: vec![Constraint::prefer_class("b")],
                ..job(1, 0)
            },
        ),
    ];
    assert_eq!(starts(&feed(&mut p, Time::ORIGIN, inputs)), vec![(0, 2)]);
    let mut p = Scheduler::new(Config {
        score: vec![ScoreTerm::Loosest],
        ..Config::default()
    });
    let inputs = [
        Input::Worker(worker(1, 4, 100)),
        Input::Worker(worker(2, 4, 50)),
        submit(0, job(1, 0)),
    ];
    assert_eq!(starts(&feed(&mut p, Time::ORIGIN, inputs)), vec![(0, 1)]);
}

/// The verdict of a worker with every slot taken.
fn slots_full() -> Verdict {
    Verdict::Full {
        dims: vec![SLOTS.name],
    }
}

/// Slots are a hard resource: a job takes one unless it says otherwise, and a full worker is
/// explained as such.
#[test]
fn slots_are_a_hard_resource() {
    let mut p = Scheduler::new(Config::fifo());
    let greedy = JobSpec {
        demand: Resources::new().with(SLOTS, 2),
        ..job(1, 0)
    };
    let inputs = [
        Input::Worker(worker(1, 3, 100)),
        submit(0, greedy),
        submit(1, job(1, 0)),
        submit(2, job(1, 0)),
    ];
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, inputs)),
        vec![(0, 1), (1, 1)]
    );
    let load = &p.stats().workers[0];
    assert_eq!(
        (load.running, load.placed.get(SLOTS), &load.headroom[2]),
        (2, 3, &(SLOTS.name, Some(0)))
    );
    let e = p.explain(2).unwrap();
    assert_eq!(e.waiting().unwrap().workers, [(1, slots_full())]);
}

/// A declaration of its own: a GPU count, hard and demanded by some jobs only, beside slots.
/// A job that needs GPUs waits for, reserves and defers to GPU workers only; the others run
/// anywhere; explanations name the GPUs.
#[test]
fn declared_gpu_count() {
    const GPUS: Resource = Resource::new("gpus").hard();
    let config = Config {
        resources: vec![SLOTS, GPUS],
        reservations: Some(Reservations {
            reserve_after: Duration::ZERO,
            ..Reservations::default()
        }),
        ..Config::fifo()
    };
    let w = |id, slots, g| WorkerState {
        id,
        class: if g > 0 { "gpu" } else { "cpu" }.into(),
        capacity: Resources::new().with(SLOTS, slots).with(GPUS, g),
        ..Default::default()
    };
    let needs = |g| JobSpec {
        demand: Resources::new().with(GPUS, g),
        ..Default::default()
    };
    let mut p = Scheduler::new(config);
    let inputs = [
        Input::Worker(w(1, 8, 0)),
        Input::Worker(w(2, 8, 2)),
        submit(0, needs(2)),
        submit(1, needs(1)),
        submit(2, needs(0)),
    ];
    // Job 0 takes both GPUs; job 1 reserves the GPU worker, not the idle CPU one with more
    // headroom, and job 2 runs on the CPU worker.
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, inputs)),
        vec![(0, 2), (2, 1)]
    );
    let r = &p.stats().reservations;
    assert_eq!((r.len(), r[0].job, r[0].worker), (1, 1, 2));
    let why = p.explain(1).unwrap();
    assert_eq!(
        why.waiting().unwrap().workers,
        [
            (
                1,
                Verdict::Full {
                    dims: vec![GPUS.name]
                }
            ),
            (
                2,
                Verdict::Full {
                    dims: vec![GPUS.name]
                }
            )
        ]
    );
    assert!(
        why.to_string()
            .contains("(demand [slots 1, gpus 1], group 0)")
            && why.to_string().contains("gpus full on 2 worker(s)"),
        "{why}"
    );
    let t = Time(Duration::from_secs(1));
    assert_eq!(
        starts(&feed(&mut p, t, [Input::Done { job: 0, attempt: 1 }])),
        vec![(1, 2)]
    );
}

/// A job may wait for a faster worker short of a hard resource it needs, but not for one that
/// could never fit it.
#[test]
fn deferral_waits_for_hard_capacity() {
    const GPUS: Resource = Resource::new("gpus").hard();
    let mut config = Config {
        speed: SpeedConfig {
            defer: Some(crate::Defer::default()),
            ..SpeedConfig::default()
        },
        ..Config::fifo()
    };
    config.resources.push(GPUS);
    let w = |id, g, speed| WorkerState {
        id,
        capacity: Resources::new().with(SLOTS, 4).with(GPUS, g),
        speed,
        ..Default::default()
    };
    let gpu_job = |work| JobSpec {
        demand: Resources::new().with(GPUS, 1),
        work: Some(Duration::from_secs(work)),
        ..Default::default()
    };
    for (fast_gpus, deferred) in [(1, true), (0, false)] {
        let mut p = Scheduler::new(config.clone());
        let inputs = [
            Input::Worker(w(1, fast_gpus, 4.0)),
            Input::Worker(w(2, 4, 1.0)),
            submit(0, gpu_job(10)),
        ];
        let first = feed(&mut p, Time::ORIGIN, inputs);
        let on = if fast_gpus > 0 { 1 } else { 2 };
        assert_eq!(starts(&first), vec![(0, on)]);
        let out = feed(&mut p, Time::ORIGIN, [submit(1, gpu_job(40))]);
        assert_eq!(out.is_empty(), deferred, "{out:?}");
        let held: Vec<_> = p.stats().deferred.iter().map(|d| (d.0, d.1)).collect();
        assert_eq!(held, if deferred { vec![(1, 1)] } else { vec![] });
    }
}

/// Without enough gain, nothing is speculated.
#[test]
fn speculation_needs_gain() {
    let mut cfg = Config::fifo();
    cfg.speed.speculate = Some(Speculate::default());
    let mut p = Scheduler::new(cfg);
    let mut j = job(1, 0);
    j.work = Some(Duration::from_secs(100));
    let inputs = [Input::Worker(worker(1, 1, 100)), submit(0, j)];
    feed(&mut p, Time::ORIGIN, inputs);
    // At t=90 the slow attempt is expected to end at 100; a fresh one on a worker twice as
    // fast would end at 140.
    let fast = WorkerState {
        speed: 2.0,
        ..worker(2, 1, 100)
    };
    assert!(feed(&mut p, Time(Duration::from_secs(90)), [Input::Worker(fast)]).is_empty());
}

/// A job demanding an undeclared resource is rejected at submission and forgotten; the others
/// are unaffected.
#[test]
fn undeclared_demand_is_rejected() {
    const GPUS: Resource = Resource::new("gpus").hard();
    let mut p = Scheduler::new(Config::fifo());
    let stray = JobSpec {
        demand: Resources::new().with(MEMORY, 1).with(GPUS, 1),
        ..job(1, 0)
    };
    let inputs = [
        Input::Worker(worker(1, 2, 100)),
        submit(0, stray),
        submit(1, job(1, 0)),
    ];
    let out = feed(&mut p, Time::ORIGIN, inputs);
    let reason = Rejection::Undeclared {
        resource: GPUS.name,
    };
    assert_eq!(out[0], Output::Rejected { job: 0, reason });
    assert_eq!(starts(&out), vec![(1, 1)]);
    assert_eq!((p.explain(0), p.stats().waiting), (None, 0));
}

/// A declaration naming a resource twice is refused.
#[test]
#[should_panic(expected = "Config::resources declares resource \"slots\" twice")]
fn duplicate_declaration_panics() {
    let mut config = Config::default();
    config.resources.push(Resource::new("slots"));
    Scheduler::new(config);
}

/// A worker reporting an undeclared resource is a caller bug.
#[test]
#[should_panic(expected = "worker 3 reports capacity of resource \"gpus\"")]
fn undeclared_worker_resource_panics() {
    let mut p = Scheduler::new(Config::default());
    let w = WorkerState {
        capacity: Resources::new()
            .with(SLOTS, 1)
            .with(Resource::new("gpus"), 1),
        ..worker(3, 1, 1)
    };
    p.handle(Input::Worker(w), Time::ORIGIN);
}
