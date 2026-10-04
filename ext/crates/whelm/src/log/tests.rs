//! Logging a run and replaying it.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

#[cfg(feature = "log")]
use super::Event;
use super::{EventSink, Logged, TaskInfo, polls, replay};
use crate::{
    Attempt, Config, FailKind, Input, JobId, JobSpec, Output, Policy, Resources, Scheduler,
    Speculate, Time, Timing, WorkerId, WorkerState,
};

/// A configuration exercising learning, speculation, retries and reservations.
fn config() -> Config {
    let mut c = Config::default();
    c.speed.timing = Timing::learned();
    c.speed.speculate = Some(Speculate::default());
    c
}

/// Worker `w`: 2 slots, 10 GB, worker 3 three times as fast.
fn worker(w: WorkerId, used_gb: f64) -> WorkerState {
    WorkerState {
        id: w,
        class: "x".into(),
        capacity: Resources::mem_gb(10.0).with_slots(2),
        reported_used: Resources::mem_gb(used_gb),
        speed: if w == 3 { 3.0 } else { 1.0 },
        ..Default::default()
    }
}

/// Run a workload with failures, worker churn, speculation and a cancellation through
/// `Logged`, and return how many outputs of each kind it produced: (starts, stops, retries).
fn run(sink: impl EventSink + 'static) -> (usize, usize, usize) {
    let mut p = Logged::new(Scheduler::new(config()), sink).sample_every(Duration::ZERO);
    // (time, job, attempt) of each running attempt's end.
    let mut ends: Vec<(u32, JobId, Attempt)> = Vec::new();
    let (mut starts, mut stops, mut retries) = (0, 0, 0);
    for w in 1..=2 {
        p.handle(Input::Worker(worker(w, 0.0)), Time::ORIGIN);
    }
    for t in 0..400u32 {
        let now = Time(Duration::from_secs(t.into()));
        if t < 120 && t % 3 == 0 {
            let i = (t / 3) as JobId;
            let spec = JobSpec {
                id: i,
                demand: Resources::mem_gb(1.0 + (i * 5 % 6) as f64),
                group: i / 8,
                work: Some(Duration::from_secs(5 + i * 7 % 11)),
                ..Default::default()
            };
            p.annotate(i, TaskInfo::default());
            p.handle(Input::Submit(spec), now);
        }
        match t {
            20 => p.handle(Input::Worker(worker(3, 0.0)), now),
            40 => p.handle(Input::WorkerGone(2), now),
            60 => p.handle(Input::Worker(worker(2, 0.0)), now),
            30 => p.handle(Input::Cancel(9), now),
            _ if t % 10 == 5 => {
                for w in 1..=3 {
                    p.handle(Input::Worker(worker(w, (t % 7) as f64 * 0.3)), now);
                }
            }
            _ => {}
        }
        let (due, rest): (Vec<_>, Vec<_>) = ends.iter().partition(|e| e.0 <= t);
        ends = rest;
        for (_, job, attempt) in due {
            let input = if job % 7 == 3 && attempt == 1 {
                Input::Failed {
                    job,
                    attempt,
                    kind: FailKind::Other,
                    why: "flaky".into(),
                }
            } else {
                Input::Done { job, attempt }
            };
            p.handle(input, now);
        }
        for o in p.poll(now) {
            match o {
                Output::Start {
                    job,
                    attempt,
                    worker,
                } => {
                    starts += 1;
                    retries += (attempt > 1) as usize;
                    let speed = if worker == 3 { 3.0 } else { 1.0 };
                    let run = (5.0 + (job * 7 % 11) as f64) / speed;
                    ends.push((t + run.ceil() as u32, job, attempt));
                }
                Output::Stop { job, attempt, .. } => {
                    stops += 1;
                    ends.retain(|e| (e.1, e.2) != (job, attempt));
                }
                _ => {}
            }
        }
    }
    (starts, stops, retries)
}

/// Replaying a logged run's inputs and polls into a fresh policy reproduces every output.
#[test]
fn replay_reproduces_outputs() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let (starts, stops, retries) = run(log.clone());
    assert!(
        starts > 40 && stops > 0 && retries > 0,
        "{starts} {stops} {retries}"
    );
    let events = log.lock().unwrap().clone();
    let logged = polls(&events);
    assert_eq!(logged.len(), 400);
    let replayed = replay(&mut Scheduler::new(config()), events.clone());
    assert_eq!(replayed, logged);
    // A replay into a differently configured policy differs: the test can fail.
    let other = replay(&mut Scheduler::new(Config::fifo()), events);
    assert_ne!(other, logged);
}

/// The JSON lines a `JsonlSink` writes read back as the same events.
#[cfg(feature = "log")]
#[test]
fn json_round_trip_replays() {
    let log = Arc::new(Mutex::new(Vec::new()));
    run(log.clone());
    let events = log.lock().unwrap().clone();
    let text: Vec<String> = events
        .iter()
        .map(|e| serde_json::to_string(e).unwrap())
        .collect();
    let back: Vec<Event> = text
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(back, events);
    assert_eq!(replay(&mut Scheduler::new(config()), back), polls(&events));
    assert!(
        text.iter()
            .any(|l| l.starts_with(r#"{"type":"input","t":{"secs":0,"nanos":0},"input":{"worker""#))
    );
}
