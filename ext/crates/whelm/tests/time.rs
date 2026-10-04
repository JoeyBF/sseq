//! Extreme and non-monotone times: the policies saturate rather than panic.

use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use whelm::{
    config::{Defer, Reservations, Speculate, SpeedConfig},
    dag::{DagConfig, DagJob, DagScheduler},
    job::JobId,
    log::Logged,
    prelude::*,
    shared::SharedPolicy,
};

/// The latest representable time.
const END: Time = Time(Duration::MAX);

/// Times at the edges, out of order: the origin, the end, back to the origin, just before the
/// end, an ordinary moment, the end again.
fn times() -> Vec<Time> {
    vec![
        Time::ORIGIN,
        END,
        Time::ORIGIN,
        END - Duration::from_nanos(1),
        Time(Duration::from_secs(5)),
        END,
    ]
}

/// A policy with every time-dependent mechanism on: aging, reservations, deferral, speculation.
fn policy() -> Scheduler {
    Scheduler::new(Config {
        age_limit: Some(Duration::from_secs(1)),
        reservations: Some(Reservations {
            reserve_after: Duration::ZERO,
            shadow_backfill: true,
            ..Reservations::default()
        }),
        speed: SpeedConfig {
            defer: Some(Defer {
                max_wait: Duration::MAX,
                min_gain: 0.0,
            }),
            speculate: Some(Speculate {
                min_gain: 0.0,
                restart_overhead: Duration::MAX,
                ..Speculate::default()
            }),
            ..SpeedConfig::default()
        },
        ..Config::default()
    })
}

/// A slow and a fast worker, each with one slot.
fn workers() -> [WorkerState; 2] {
    [(1, 1.0), (2, 4.0)].map(|(id, speed)| WorkerState {
        id,
        class: format!("w{id}"),
        capacity: Resources::new().with(MEMORY, 100).with(SLOTS, 1),
        speed,
        ..Default::default()
    })
}

/// Job `id`'s spec: work from tiny to the longest span by its parity, due at the end of time.
fn job(id: JobId) -> JobSpec {
    JobSpec {
        demand: Resources::new().with(MEMORY, 1),
        work: Some([Duration::from_nanos(1), Duration::MAX][id as usize % 2]),
        due: Some(END),
        rank: Some(Duration::MAX),
        ..Default::default()
    }
}

/// Drive `p` through [`times`]: submit, poll, and report each start done at the next time.
fn drive(p: &mut impl Policy) {
    for w in workers() {
        p.handle(Input::Worker(w), Time::ORIGIN);
    }
    let mut running = Vec::new();
    for (i, now) in times().into_iter().enumerate() {
        for (job, attempt) in running.drain(..) {
            p.handle(Input::Done { job, attempt }, now);
        }
        for id in 3 * i as JobId..3 * i as JobId + 3 {
            p.handle(
                Input::Submit {
                    job: id,
                    spec: job(id),
                },
                now,
            );
        }
        for o in p.poll(now) {
            if let Output::Start { job, attempt, .. } = o {
                running.push((job, attempt));
            }
        }
        p.next_wakeup();
        p.stats();
        p.explain(3 * i as JobId);
    }
}

/// The scheduler takes the edges of time without panicking.
#[test]
fn scheduler_saturates() {
    drive(&mut policy());
}

/// So does the event log, whose sampling measures the span since its last sample.
#[test]
fn logged_saturates() {
    drive(&mut Logged::new(policy(), Vec::new()).sample_every(Duration::from_secs(1)));
}

/// So does the DAG layer, with dependencies declared at the end of time.
#[test]
fn dag_saturates() {
    let mut d = DagScheduler::new(DagConfig::default(), policy());
    let chain = (1000..1004).map(|id| DagJob {
        id,
        deps: if id > 1000 { vec![id - 1] } else { vec![] },
        spec: job(id),
        ..Default::default()
    });
    d.declare(chain, END).unwrap();
    drive(&mut d);
}

/// A shared front end reads a clock that jumps to the end of time and back as staying at the end,
/// and takes the longest timeout.
#[test]
fn shared_clock_saturates() {
    let reads = AtomicUsize::new(0);
    let shared = SharedPolicy::new(policy(), move || {
        times()[reads.fetch_add(1, Ordering::Relaxed) % times().len()]
    });
    for w in workers() {
        shared.worker_update(w);
    }
    for _ in 0..times().len() {
        shared.tick();
    }
    assert_eq!(shared.stats().now, END);
    shared
        .lease_timeout(0, job(0), Duration::MAX)
        .unwrap()
        .complete();
}
