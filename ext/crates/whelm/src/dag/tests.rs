//! Behaviour of the whole [`DagScheduler`], driven through [`Policy`].

use std::sync::Arc;

use crate::{
    Attempt, Config, DagConfig, DagError, DagJob, DagScheduler, DagStats, DagTemplate, FailKind,
    GaveUp, Input, Instant, JobId, JobSpec, Output, Policy, Resources, RetryConfig, Scheduler,
    Unit, WorkerState,
};

/// A DAG over a FIFO scheduler with one one-slot worker, and `max_attempts` attempts per job.
fn dag(config: DagConfig, max_attempts: u32) -> DagScheduler<Scheduler> {
    let mut d = DagScheduler::new(
        config,
        Scheduler::new(Config {
            retry: RetryConfig { max_attempts },
            ..Config::fifo()
        }),
    );
    d.handle(
        Input::Worker(WorkerState::new(1, "x", 1, Resources::mem(100))),
        0.0,
    );
    d
}

/// A job of group 0 with the given dependencies.
fn job(id: JobId, deps: &[JobId]) -> DagJob {
    DagJob::new(JobSpec::new(id, Resources::mem(1), 0), deps.to_vec())
}

/// The start of attempt `attempt` of `job` on worker 1.
fn start(job: JobId, attempt: Attempt) -> Output {
    Output::Start {
        job,
        attempt,
        worker: 1,
    }
}

/// Handle `inputs` at `t`, then poll.
fn feed(
    d: &mut DagScheduler<Scheduler>,
    t: Instant,
    inputs: impl IntoIterator<Item = Input>,
) -> Vec<Output> {
    for i in inputs {
        d.handle(i, t);
    }
    d.poll(t)
}

/// A done message.
fn done(job: JobId, attempt: Attempt) -> Input {
    Input::Done { job, attempt }
}

/// A failure message.
fn fail(job: JobId, attempt: Attempt) -> Input {
    Input::Failed {
        job,
        attempt,
        kind: FailKind::Other,
        why: "test".into(),
    }
}

/// A unit `id` of a template of `len` independent jobs at `base`, after `deps`.
fn unit(id: JobId, base: JobId, len: usize, deps: &[JobId]) -> Unit {
    Unit::new(
        id,
        base,
        Arc::new(DagTemplate::new(len, []).unwrap()),
        JobSpec::new(0, Resources::mem(1), 0),
        deps.to_vec(),
    )
}

/// The inner policy's starts come out of the DAG's poll, and a done attempt releases the
/// dependents.
#[test]
fn done_releases_dependents() {
    let mut d = dag(DagConfig::default(), 4);
    d.declare(vec![job(1, &[]), job(2, &[1])], 0.0).unwrap();
    assert_eq!(d.poll(0.0), vec![start(1, 1)]);
    assert_eq!(feed(&mut d, 1.0, [done(1, 1)]), vec![start(2, 1)]);
    assert!(feed(&mut d, 2.0, [done(2, 1)]).is_empty());
    let st = d.dag_stats();
    assert_eq!(
        (st.pending, st.submitted, st.completed_remembered),
        (0, 0, 2)
    );
    assert_eq!(d.explain(2), Some("job 2 completed".into()));
}

/// A retried job's stale report does not complete it here either.
#[test]
fn stale_done_does_not_complete() {
    let mut d = dag(DagConfig::default(), 4);
    d.declare(vec![job(1, &[]), job(2, &[1])], 0.0).unwrap();
    d.poll(0.0);
    assert_eq!(feed(&mut d, 1.0, [fail(1, 1)]), vec![start(1, 2)]);
    assert!(feed(&mut d, 2.0, [done(1, 1)]).is_empty());
    assert_eq!(d.dag_stats().pending, 1);
    assert_eq!(feed(&mut d, 3.0, [done(1, 2)]), vec![start(2, 1)]);
}

/// A worker leaving fails its attempts in the inner policy, which retries them: a late report
/// from the departed worker completes nothing.
#[test]
fn worker_gone_retries() {
    let mut d = dag(DagConfig::default(), 4);
    d.declare(vec![job(1, &[]), job(2, &[1])], 0.0).unwrap();
    d.poll(0.0);
    assert!(feed(&mut d, 1.0, [Input::WorkerGone(1)]).is_empty());
    assert!(feed(&mut d, 2.0, [done(1, 1)]).is_empty());
    let w = WorkerState::new(1, "x", 1, Resources::mem(100));
    assert_eq!(feed(&mut d, 3.0, [Input::Worker(w)]), vec![start(1, 2)]);
    assert_eq!(feed(&mut d, 4.0, [done(1, 2)]), vec![start(2, 1)]);
}

/// Local jobs are announced, never submitted, and completed with attempt 0.
#[test]
fn local_jobs_run_on_the_caller() {
    let mut d = dag(DagConfig::default(), 4);
    d.declare(vec![job(1, &[]).local(), job(2, &[1])], 0.0)
        .unwrap();
    assert_eq!(d.poll(0.0), vec![Output::RunLocal { job: 1 }]);
    assert_eq!(d.stats().waiting, 0);
    assert!(!d.release(1, 0.0));
    // Attempt numbers of workers' jobs do not complete a local job.
    assert!(feed(&mut d, 1.0, [done(1, 1)]).is_empty());
    assert_eq!(feed(&mut d, 1.0, [done(1, 0)]), vec![start(2, 1)]);
}

/// Without `auto_submit`, ready jobs are announced and held until released.
#[test]
fn held_jobs_are_announced() {
    let config = DagConfig {
        auto_submit: false,
        ..DagConfig::default()
    };
    let mut d = dag(config, 4);
    d.declare(vec![job(1, &[]), job(2, &[1]), job(3, &[])], 0.0)
        .unwrap();
    assert_eq!(
        d.poll(0.0),
        vec![Output::Ready { job: 1 }, Output::Ready { job: 3 }]
    );
    assert!(d.release(1, 1.0));
    assert!(!d.release(1, 1.0));
    assert_eq!(d.poll(1.0), vec![start(1, 1)]);
    assert_eq!(
        feed(&mut d, 2.0, [done(1, 1)]),
        vec![Output::Ready { job: 2 }]
    );
    // Cancelling withdraws an announcement not yet polled.
    d.declare(vec![job(4, &[])], 3.0).unwrap();
    assert_eq!(d.cancel(4), vec![4]);
    assert!(d.poll(3.0).is_empty());
}

/// `announcements` drains the layer's own outputs without placing anything; the next poll
/// places, and returns only what was announced since.
#[test]
fn announcements_come_before_placement() {
    let config = DagConfig {
        auto_submit: false,
        ..DagConfig::default()
    };
    let mut d = dag(config, 4);
    d.declare(vec![job(1, &[]), job(2, &[]), job(3, &[1])], 0.0)
        .unwrap();
    assert_eq!(
        d.announcements(),
        vec![Output::Ready { job: 1 }, Output::Ready { job: 2 }]
    );
    assert!(d.announcements().is_empty());
    // Job 2 is released first, so it takes the one slot.
    assert!(d.release(2, 0.0) && d.release(1, 0.0));
    assert_eq!(d.poll(0.0), vec![start(2, 1)]);
    d.handle(done(2, 1), 1.0);
    d.declare(vec![job(4, &[])], 1.0).unwrap();
    assert_eq!(d.poll(1.0), vec![Output::Ready { job: 4 }, start(1, 1)]);
}

/// Passthrough jobs and units are announced when recorded; a plain job's unit is not.
#[test]
fn passthroughs_are_announced() {
    let config = DagConfig {
        record_passthrough: true,
        ..DagConfig::default()
    };
    let mut d = dag(config, 4);
    d.declare(
        vec![job(1, &[]), DagJob::passthrough(2, 0, vec![1], 0.0)],
        0.0,
    )
    .unwrap();
    assert_eq!(d.poll(0.0), vec![start(1, 1)]);
    assert_eq!(
        feed(&mut d, 1.0, [done(1, 1)]),
        vec![Output::Passed { job: 2 }]
    );
    d.declare([unit(200, 100, 1, &[2])], 2.0).unwrap();
    assert_eq!(d.poll(2.0), vec![start(100, 1)]);
    assert_eq!(
        feed(&mut d, 3.0, [done(100, 1)]),
        vec![Output::Passed { job: 200 }]
    );
}

/// A give-up passes through and holds the job again; releasing it starts another round.
#[test]
fn give_up_holds_the_job() {
    let mut d = dag(DagConfig::default(), 1);
    d.declare(vec![job(1, &[]), job(2, &[1])], 0.0).unwrap();
    d.poll(0.0);
    let out = feed(&mut d, 1.0, [fail(1, 1)]);
    let [Output::GaveUp(GaveUp { job: 1, .. })] = out[..] else {
        panic!("expected a give-up, got {out:?}");
    };
    assert_eq!(d.dag_stats().held, 1);
    assert!(d.explain(1).unwrap().contains("held until release"));
    assert!(d.release(1, 2.0));
    assert_eq!(d.poll(2.0), vec![start(1, 1)]);
    assert_eq!(feed(&mut d, 3.0, [done(1, 1)]), vec![start(2, 1)]);
}

/// `Input::Cancel` cascades to dependents and stops running attempts.
#[test]
fn cancel_input_cascades() {
    let mut d = dag(DagConfig::default(), 4);
    d.declare(vec![job(1, &[]), job(2, &[1]), job(3, &[2])], 0.0)
        .unwrap();
    d.poll(0.0);
    let out = feed(&mut d, 1.0, [Input::Cancel(1)]);
    assert_eq!(
        out,
        vec![Output::Stop {
            job: 1,
            attempt: 1,
            worker: 1
        }]
    );
    assert_eq!(d.dag_stats(), DagStats::default());
    assert_eq!(d.stats().running, 0);
}

/// A two-job unit after job 1, and job 2 after it; job 1 done, leaf 100 running, and the unit
/// closed early.
fn closed_unit() -> DagScheduler<Scheduler> {
    let mut d = dag(DagConfig::default(), 4);
    d.declare(vec![job(1, &[]), job(2, &[200])], 0.0).unwrap();
    d.declare([unit(200, 100, 2, &[1])], 0.0).unwrap();
    assert_eq!(d.poll(0.0), vec![start(1, 1)]);
    assert_eq!(feed(&mut d, 1.0, [done(1, 1)]), vec![start(100, 1)]);
    assert_eq!(d.close(200, 2.0), Ok(vec![100]));
    // Leaf 101 was withdrawn; the unit completed, releasing job 2 behind the running leaf.
    let st = d.stats();
    assert_eq!((st.waiting, st.running), (1, 1));
    assert!(d.poll(2.0).is_empty());
    d
}

/// A running job of a unit closed early keeps its resources until its attempt ends; a
/// failure then is neither retried nor reported.
#[test]
fn closed_unit_job_fails() {
    let mut d = closed_unit();
    assert_eq!(feed(&mut d, 3.0, [fail(100, 1)]), vec![start(2, 1)]);
    assert_eq!(d.stats().running, 1);
    assert_eq!(d.explain(100), None);
}

/// The same when the job's worker leaves.
#[test]
fn closed_unit_job_worker_gone() {
    let mut d = closed_unit();
    assert!(feed(&mut d, 3.0, [Input::WorkerGone(1)]).is_empty());
    let st = d.stats();
    assert_eq!((st.waiting, st.running), (1, 0));
    let w = WorkerState::new(1, "x", 1, Resources::mem(100));
    assert_eq!(feed(&mut d, 4.0, [Input::Worker(w)]), vec![start(2, 1)]);
}

/// Its completion only frees its resources.
#[test]
fn closed_unit_job_done() {
    let mut d = closed_unit();
    assert_eq!(feed(&mut d, 3.0, [done(100, 1)]), vec![start(2, 1)]);
}

/// A snapshot restores held jobs as announcements and submitted ones as fresh submissions.
#[cfg(feature = "serde")]
#[test]
fn restore_announces_held_jobs() {
    let config = DagConfig {
        auto_submit: false,
        ..DagConfig::default()
    };
    let mut d = dag(config, 4);
    d.declare(
        vec![job(1, &[]), job(2, &[]), job(3, &[1]), job(4, &[]).local()],
        0.0,
    )
    .unwrap();
    d.poll(0.0);
    assert!(d.release(1, 0.0));
    assert_eq!(d.poll(0.0), vec![start(1, 1)]);
    let snap = d.snapshot();
    let mut r = DagScheduler::restore(snap, Scheduler::new(Config::fifo()), None, 5.0);
    let w = WorkerState::new(7, "x", 1, Resources::mem(100));
    assert_eq!(
        feed(&mut r, 5.0, [Input::Worker(w)]),
        vec![
            Output::RunLocal { job: 4 },
            Output::Ready { job: 2 },
            Output::Start {
                job: 1,
                attempt: 1,
                worker: 7
            }
        ]
    );
}

/// Ids of a unit may not collide with another's.
#[test]
fn overlapping_ids_are_refused() {
    let mut d = dag(DagConfig::default(), 4);
    d.declare([unit(99, 10, 3, &[])], 0.0).unwrap();
    assert_eq!(
        d.declare([unit(98, 12, 3, &[])], 0.0).unwrap_err(),
        DagError::Overlap(12)
    );
    assert_eq!(
        d.declare([job(11, &[])], 0.0).unwrap_err(),
        DagError::Overlap(11)
    );
    assert_eq!(
        d.declare([job(5, &[11])], 0.0).unwrap_err(),
        DagError::Overlap(11)
    );
    assert_eq!(
        d.declare([unit(11, 20, 3, &[])], 0.0).unwrap_err(),
        DagError::Overlap(11)
    );
    assert_eq!(
        d.declare([unit(30, 29, 3, &[])], 0.0).unwrap_err(),
        DagError::Overlap(30)
    );
    assert_eq!(
        d.declare([unit(97, 0, 30, &[])], 0.0).unwrap_err(),
        DagError::Overlap(10)
    );
    assert_eq!(
        d.declare([unit(99, 50, 3, &[])], 0.0).unwrap_err(),
        DagError::Duplicate(99)
    );
    d.declare([unit(98, 13, 3, &[99])], 0.0).unwrap();
}
