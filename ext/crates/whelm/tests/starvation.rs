//! No starvation: a stream of small jobs keeps every worker packed while a big job waits.

use std::{collections::BTreeMap, time::Duration};

use proptest::prelude::*;
use whelm::{
    Attempt, Config, Input, JobId, JobSpec, Output, Policy, Reservations, Resources, Scheduler,
    Time, WorkerState,
};

const TICK: Duration = Duration::from_secs(1);
const RESERVE_AFTER: Duration = Duration::from_secs(60);
const BIG: u64 = u64::MAX / 2;

struct Stream {
    workers: usize,
    slots: usize,
    budget: u64,
    /// (demand, duration) of the small jobs, cycled.
    small: Vec<(u64, u64)>,
    big_demand: u64,
    big_at: Time,
    horizon: Time,
}

/// Runs the stream; returns the big job's wait, or `None` if it was never placed.
///
/// The bound under test, for the most urgent waiting job: it reserves a worker at the first
/// poll after it has waited `reserve_after`, and from then on nothing new is admitted on that
/// worker, so it is placed at the latest when the jobs running there at that moment finish:
/// `wait <= reserve_after + D + tick`, where `D` is the longest small-job duration and `tick` the
/// poll granularity. (A job behind more urgent starving jobs waits for their reservations
/// first.) FIFO, as a control, starves the big job for the whole stream.
fn run(p: &mut dyn Policy, s: &Stream) -> Option<Duration> {
    for w in 0..s.workers {
        p.handle(
            Input::Worker(WorkerState {
                id: w as u64,
                class: "x".into(),
                slots: s.slots,
                budget: Resources::mem(s.budget),
                ..Default::default()
            }),
            Time::ZERO,
        );
    }
    // Running attempts: job -> (attempt, end).
    let mut ends: BTreeMap<JobId, (Attempt, Time)> = BTreeMap::new();
    let mut duration: BTreeMap<u64, u64> = BTreeMap::new();
    let mut next = 0u64;
    let mut big_submitted = false;
    let mut t = Time::ZERO;
    while t < s.horizon {
        // Keep a few small jobs waiting at all times, each in a new (younger) group.
        let waiting = p.stats().waiting;
        for _ in waiting..(s.workers * s.slots / 2).max(2) {
            let (demand, d) = s.small[next as usize % s.small.len()];
            let small = JobSpec {
                id: next,
                demand: Resources::mem(demand),
                group: 1_000 + next,
                work: Some(Duration::from_secs(d)),
                ..Default::default()
            };
            p.handle(Input::Submit(small), t);
            duration.insert(next, d);
            next += 1;
        }
        if !big_submitted && t >= s.big_at {
            let big = JobSpec {
                id: BIG,
                demand: Resources::mem(s.big_demand),
                priority: Some(-1),
                ..Default::default()
            };
            p.handle(Input::Submit(big), t);
            big_submitted = true;
        }
        let done: Vec<JobId> = ends.iter().filter(|e| e.1.1 <= t).map(|e| *e.0).collect();
        for job in done {
            let (attempt, _) = ends.remove(&job).unwrap();
            p.handle(Input::Done { job, attempt }, t);
        }
        for o in p.poll(t) {
            let Output::Start { job, attempt, .. } = o else {
                panic!("unexpected output {o:?}");
            };
            if job == BIG {
                return Some(t - s.big_at);
            }
            ends.insert(job, (attempt, t + Duration::from_secs(duration[&job])));
        }
        t += TICK;
    }
    None
}

/// The configurations that must not starve, with `RESERVE_AFTER`.
fn policies() -> Vec<(&'static str, Box<dyn Policy>)> {
    let reserve = |shadow_backfill| {
        Some(Reservations {
            reserve_after: RESERVE_AFTER,
            shadow_backfill,
            ..Reservations::default()
        })
    };
    let make = |base: Config, shadow| -> Box<dyn Policy> {
        Box::new(Scheduler::new(Config {
            reservations: reserve(shadow),
            ..base
        }))
    };
    vec![
        ("backfill", make(Config::default(), false)),
        ("backfill, shadow", make(Config::default(), true)),
        ("bestfit", make(Config::best_fit(), false)),
        ("bestfit, shadow", make(Config::best_fit(), true)),
    ]
}

/// FIFO starves the big job; the reserving configurations meet the bound.
#[test]
fn fifo_starves_and_backfill_does_not() {
    let s = Stream {
        workers: 3,
        slots: 8,
        budget: 100,
        small: vec![(12, 40), (15, 55), (9, 25), (14, 60)],
        big_demand: 70,
        big_at: Time::from_secs(30),
        horizon: Time::from_secs(5_000),
    };
    assert_eq!(
        run(&mut Scheduler::new(Config::fifo()), &s),
        None,
        "FIFO should starve it"
    );
    let d = Duration::from_secs(s.small.iter().map(|x| x.1).max().unwrap());
    for (name, mut p) in policies() {
        let wait = run(&mut *p, &s).unwrap_or_else(|| panic!("{name} starved the big job"));
        assert!(
            wait <= RESERVE_AFTER + d + 2 * TICK,
            "{name}: waited {wait:?}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// The bound holds over random streams.
    #[test]
    fn starving_job_waits_at_most_reserve_after_plus_longest_job(
        workers in 1usize..4,
        slots in 2usize..10,
        small in prop::collection::vec((5u64..40, 5u64..120), 1..6),
        big_demand in 41u64..250,
        big_at_ms in 0u64..200_000,
    ) {
        let (big_at, horizon) = (Time::from_millis(big_at_ms), Time::from_secs(2_000));
        let s = Stream { workers, slots, budget: 100, small, big_demand, big_at, horizon };
        let d = Duration::from_secs(s.small.iter().map(|x| x.1).max().unwrap());
        for (name, mut p) in policies() {
            let wait = run(&mut *p, &s);
            prop_assert!(wait.is_some(), "{} starved the big job", name);
            prop_assert!(wait.unwrap() <= RESERVE_AFTER + d + 2 * TICK, "{}: waited {:?}", name, wait);
        }
    }
}
