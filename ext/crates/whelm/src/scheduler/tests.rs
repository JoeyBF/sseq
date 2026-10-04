//! Behaviour of the whole [`Scheduler`], driven through [`Policy`].

use std::time::Duration;

use crate::{
    Attempt, Config, Constraint, FailKind, Input, JobId, JobSpec, OrderTerm, Output, Policy,
    Reservations, Resource, ResourceId, Resources, RetryConfig, Scheduler, ScoreTerm, Speculate,
    SpeedConfig, Time, Verdict, WorkerId, WorkerState,
};

const GB: u64 = 1_000_000_000;

const SLOTS: ResourceId = ResourceId::SLOTS;

/// A worker of class "x" with a memory capacity in GB.
fn worker(id: WorkerId, slots: u64, budget_gb: u64) -> WorkerState {
    WorkerState {
        id,
        class: "x".into(),
        capacity: Resources::mem(budget_gb * GB).with_slots(slots),
        ..Default::default()
    }
}

/// A job with a demand in GB.
fn job(id: JobId, gb: u64, group: u64) -> JobSpec {
    JobSpec {
        id,
        demand: Resources::mem(gb * GB),
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
    inputs.extend((0..4).map(|i| Input::Submit(job(i, 10, 0))));
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
        Input::Submit(job(0, 10, 0)),
    ];
    feed(&mut p, Time::ORIGIN, inputs);
    let j = JobSpec {
        constraints: vec![Constraint::prefer_worker(1)],
        ..job(1, 10, 0)
    };
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, [Input::Submit(j)])),
        vec![(1, 1)]
    );
}

/// Explicit priority first, then the oldest group, then FIFO.
#[test]
fn priority_order_is_group_arrival_then_fifo() {
    let mut p = Scheduler::new(Config::default());
    p.handle(Input::Submit(job(10, 1, 7)), Time::ORIGIN); // group 7 arrives first
    p.handle(Input::Submit(job(11, 1, 3)), Time(Duration::from_secs(1)));
    p.handle(Input::Submit(job(12, 1, 7)), Time(Duration::from_secs(2)));
    let mut urgent = job(13, 1, 3);
    urgent.priority = Some(-1);
    p.handle(Input::Submit(urgent), Time(Duration::from_secs(3)));
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
        Input::Submit(job(0, 1, 0)),
        Input::Submit(job(1, 1, 0)),
    ];
    // Both empty workers admit; the smaller one is the tighter fit.
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, inputs)),
        vec![(0, 2), (1, 2)]
    );
    let inputs = [Input::Submit(job(2, 50, 0)), Input::Submit(job(3, 1, 0))];
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
        capacity: Resources::mem(100 * GB).with_dev(10 * GB).with_slots(4),
        reported_used: Resources::ZERO.with_dev(8 * GB),
        ..Default::default()
    };
    let w2 = WorkerState {
        id: 2,
        class: "x".into(),
        capacity: Resources::mem(100 * GB).with_dev(100 * GB).with_slots(4),
        reported_used: Resources::mem(59 * GB),
        ..Default::default()
    };
    let inputs = [
        Input::Worker(w1),
        Input::Worker(w2),
        Input::Submit(job(0, 1, 0)),
    ];
    assert_eq!(starts(&feed(&mut p, Time::ORIGIN, inputs)), vec![(0, 1)]);
}

/// A failed job is retried before less urgent jobs submitted after it, and keeps its age.
#[test]
fn retry_keeps_place_and_age() {
    let mut p = Scheduler::new(Config::fifo());
    let inputs = [
        Input::Worker(worker(1, 1, 100)),
        Input::Submit(job(0, 1, 0)),
        Input::Submit(job(1, 1, 0)),
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
        Input::Submit(job(0, 1, 0)),
        Input::Submit(job(1, 1, 0)),
        Input::Submit(job(2, 1, 0)),
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
    inputs.push(Input::Submit(job(0, 1, 0)));
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
        ..job(0, 1, 0)
    };
    let inputs = [
        Input::Worker(worker(1, 1, 100)),
        Input::Worker(worker(2, 1, 100)),
        Input::Submit(j),
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
        [Input::Submit(job(0, 1, 0))],
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
    let inputs = [
        Input::Worker(worker(1, 1, 100)),
        Input::Submit(job(0, 1, 0)),
    ];
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
        Input::Submit(job(0, 1, 0)),
        Input::Submit(job(1, 1, 0)),
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
    let inputs = [
        Input::Worker(worker(1, 1, 100)),
        Input::Submit(job(0, 1, 0)),
    ];
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
    assert_eq!((st.running, &st.workers[0].placed), (0, &Resources::ZERO));
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
        Input::Submit(job(0, 1, 0)),
        Input::Submit(job(1, 1, 0)),
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
    let mut j = job(0, 1, 0);
    j.work = Some(Duration::from_secs(100));
    let inputs = [Input::Worker(worker(1, 1, 100)), Input::Submit(j)];
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

/// The order in which a one-slot worker runs `jobs`, all submitted before it joins.
fn run_order(config: Config, jobs: Vec<JobSpec>) -> Vec<JobId> {
    let mut p = Scheduler::new(config);
    let n = jobs.len();
    feed(&mut p, Time::ORIGIN, jobs.into_iter().map(Input::Submit));
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
    let spec = |id, weight, work| JobSpec {
        weight,
        work,
        ..job(id, 1, 0)
    };
    let jobs = vec![
        spec(0, 1.0, None),
        spec(1, 1.0, Some(Duration::from_secs(10))),
        spec(2, 3.0, Some(Duration::from_secs(10))),
        spec(3, 1.0, Some(Duration::from_secs(2))),
        spec(4, 1.0, None),
    ];
    let order = run_order(Config::weighted_completion(), jobs);
    assert_eq!(order, vec![3, 2, 1, 0, 4]);
}

/// Jackson's rule: earliest due date first; jobs without one last.
#[test]
fn edd_orders_by_due_date() {
    let spec = |id, due| JobSpec {
        due,
        ..job(id, 1, 0)
    };
    let jobs = vec![
        spec(0, None),
        spec(1, Some(Time(Duration::from_secs(50)))),
        spec(2, Some(Time(Duration::from_secs(3)))),
    ];
    assert_eq!(run_order(Config::lateness(), jobs), vec![2, 1, 0]);
}

/// The default order: explicit priority, then group arrival, ranks ignored; with
/// [`OrderTerm::Rank`] between them, largest rank first. A repeated term changes nothing.
#[test]
fn default_order_is_priority_group() {
    let spec = |id, group, priority, rank| JobSpec {
        priority,
        rank,
        ..job(id, 1, group)
    };
    let jobs = || {
        vec![
            spec(0, 5, None, None),
            spec(1, 6, None, Some(Duration::from_secs(2))),
            spec(2, 6, None, Some(Duration::from_secs(9))),
            spec(3, 5, Some(-1), None),
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
                capacity: Resources::mem(100 * GB).with_slots(4),
                ..Default::default()
            })
        })
        .collect();
    let constrained = |id, constraints| JobSpec {
        constraints,
        ..job(id, 1, 0)
    };
    let either = constrained(
        0,
        vec![
            Constraint::require_class("c"),
            Constraint::require_class("b"),
        ],
    );
    let both = constrained(
        1,
        vec![
            Constraint::require_class("a"),
            Constraint::require_worker(2),
        ],
    );
    let forbidden = constrained(
        2,
        vec![
            Constraint::require_class("a"),
            Constraint::forbid_class("a"),
        ],
    );
    let worker_or = constrained(
        3,
        vec![
            Constraint::require_worker(3),
            Constraint::require_worker(1),
            Constraint::forbid_worker(1),
        ],
    );
    inputs.extend([either, both, forbidden, worker_or].map(Input::Submit));
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
            capacity: Resources::mem(100 * GB).with_slots(4),
            ..Default::default()
        }),
        Input::Worker(WorkerState {
            id: 2,
            class: "b".into(),
            capacity: Resources::mem(100 * GB).with_slots(4),
            ..Default::default()
        }),
        Input::Submit(JobSpec {
            constraints: vec![Constraint::prefer_class("b")],
            ..job(0, 1, 0)
        }),
    ];
    assert_eq!(starts(&feed(&mut p, Time::ORIGIN, inputs)), vec![(0, 2)]);
    let mut p = Scheduler::new(Config {
        score: vec![ScoreTerm::Loosest],
        ..Config::default()
    });
    let inputs = [
        Input::Worker(worker(1, 4, 100)),
        Input::Worker(worker(2, 4, 50)),
        Input::Submit(job(0, 1, 0)),
    ];
    assert_eq!(starts(&feed(&mut p, Time::ORIGIN, inputs)), vec![(0, 1)]);
}

/// The verdict of a worker with every slot taken.
fn slots_full() -> Verdict {
    Verdict::Full { dims: vec![SLOTS] }
}

/// Slots are a hard resource: a job takes one unless it says otherwise, and a full worker is
/// explained as such.
#[test]
fn slots_are_a_hard_resource() {
    let mut p = Scheduler::new(Config::fifo());
    let greedy = JobSpec {
        demand: Resources::ZERO.with_slots(2),
        ..job(0, 1, 0)
    };
    let inputs = [
        Input::Worker(worker(1, 3, 100)),
        Input::Submit(greedy),
        Input::Submit(job(1, 1, 0)),
        Input::Submit(job(2, 1, 0)),
    ];
    assert_eq!(
        starts(&feed(&mut p, Time::ORIGIN, inputs)),
        vec![(0, 1), (1, 1)]
    );
    let load = &p.stats().workers[0];
    assert_eq!(
        (load.running, load.placed[SLOTS], load.headroom[SLOTS.0]),
        (2, 3, Some(0))
    );
    let e = p.explain(2).unwrap();
    assert_eq!(e.waiting().unwrap().workers, [(1, slots_full())]);
}

/// A declaration of its own: a GPU count, hard and demanded by some jobs only, beside slots.
/// A job that needs GPUs waits for, reserves and defers to GPU workers only; the others run
/// anywhere; explanations name the GPUs.
#[test]
fn declared_gpu_count() {
    let gpus = ResourceId(1);
    let config = Config {
        resources: vec![
            Resource::slots(),
            Resource {
                name: "gpus".into(),
                hard: true,
                ..Default::default()
            },
        ],
        reservations: Some(Reservations {
            reserve_after: Duration::ZERO,
            ..Reservations::default()
        }),
        ..Config::fifo()
    };
    let w = |id, slots, g| WorkerState {
        id,
        class: if g > 0 { "gpu" } else { "cpu" }.into(),
        capacity: Resources::of([(ResourceId(0), slots), (gpus, g)]),
        ..Default::default()
    };
    let needs = |id, g| JobSpec {
        id,
        demand: Resources::of([(gpus, g)]),
        ..Default::default()
    };
    let mut p = Scheduler::new(config);
    let inputs = [
        Input::Worker(w(1, 8, 0)),
        Input::Worker(w(2, 8, 2)),
        Input::Submit(needs(0, 2)),
        Input::Submit(needs(1, 1)),
        Input::Submit(needs(2, 0)),
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
            (1, Verdict::Full { dims: vec![gpus] }),
            (2, Verdict::Full { dims: vec![gpus] })
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
    let gpus = ResourceId(3);
    let mut config = Config {
        speed: SpeedConfig {
            defer: Some(crate::Defer::default()),
            ..SpeedConfig::default()
        },
        ..Config::fifo()
    };
    config.resources.push(Resource {
        name: "gpus".into(),
        hard: true,
        ..Default::default()
    });
    let w = |id, g, speed| WorkerState {
        id,
        capacity: Resources::ZERO.with_slots(4).with(gpus, g),
        speed,
        ..Default::default()
    };
    let gpu_job = |id, work| JobSpec {
        id,
        demand: Resources::ZERO.with(gpus, 1),
        work: Some(Duration::from_secs(work)),
        ..Default::default()
    };
    for (fast_gpus, deferred) in [(1, true), (0, false)] {
        let mut p = Scheduler::new(config.clone());
        let inputs = [
            Input::Worker(w(1, fast_gpus, 4.0)),
            Input::Worker(w(2, 4, 1.0)),
            Input::Submit(gpu_job(0, 10)),
        ];
        let first = feed(&mut p, Time::ORIGIN, inputs);
        let on = if fast_gpus > 0 { 1 } else { 2 };
        assert_eq!(starts(&first), vec![(0, on)]);
        let out = feed(&mut p, Time::ORIGIN, [Input::Submit(gpu_job(1, 40))]);
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
    let mut j = job(0, 1, 0);
    j.work = Some(Duration::from_secs(100));
    let inputs = [Input::Worker(worker(1, 1, 100)), Input::Submit(j)];
    feed(&mut p, Time::ORIGIN, inputs);
    // At t=90 the slow attempt is expected to end at 100; a fresh one on a worker twice as
    // fast would end at 140.
    let fast = WorkerState {
        speed: 2.0,
        ..worker(2, 1, 100)
    };
    assert!(feed(&mut p, Time(Duration::from_secs(90)), [Input::Worker(fast)]).is_empty());
}
