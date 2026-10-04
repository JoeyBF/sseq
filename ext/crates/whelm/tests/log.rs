//! The event log round trip: a logged run replays exactly, from memory and from JSON lines.
#![cfg(feature = "log")]

use std::{
    io::BufRead,
    sync::{Arc, Mutex},
    time::Duration,
};

use whelm::{
    Attempt, Config, EventSink, FailKind, Input, JobId, JobSpec, MEMORY, Output, Policy, Resources,
    SLOTS, Scheduler, Time, WorkerState, gb,
    log::{Event, JsonlSink, Logged, TaskInfo, polls, replay},
};

const WORKERS: u64 = 3;
const WORKER_SLOTS: usize = 2;
const BUDGET_GB: f64 = 10.0;
const JOBS: u64 = 80;
const HEARTBEAT: Duration = Duration::from_secs(60);
/// When worker 0 leaves (and rejoins at the next heartbeat).
const LOSS: Time = Time(Duration::from_millis(50_050));

/// Job `i`: arrival, run time and demand (GB). Times are chosen so that no two events coincide.
fn job(i: u64) -> (Time, Duration, f64) {
    (
        Time(Duration::from_millis(370 + 1300 * i)),
        Duration::from_millis(5000 + (i * 7 % 11) * 1000 + 123 * i),
        1.0 + (i * 5 % 6) as f64,
    )
}

/// Whether job `i`'s first attempt fails.
fn flaky(i: u64) -> bool {
    i % 13 == 5
}

/// The policy under test.
fn policy() -> Scheduler {
    Scheduler::new(Config::best_fit())
}

/// Worker `w`'s heartbeat.
fn state(w: u64) -> WorkerState {
    WorkerState {
        id: w,
        class: "h200".into(),
        capacity: Resources::new()
            .with(MEMORY, gb(BUDGET_GB))
            .with(SLOTS, WORKER_SLOTS as u64),
        ..Default::default()
    }
}

/// Run the workload with a simple event-driven driver (jobs run at speed 1 whatever the
/// concurrency, every worker heartbeats each `HEARTBEAT` seconds, worker 0 leaves at `LOSS`,
/// flaky jobs fail their first attempt), logging to `sink`. Returns the starts, by attempt.
fn run(sink: impl EventSink + 'static) -> Vec<(JobId, Attempt)> {
    let mut p = Logged::new(policy(), sink);
    let mut started = Vec::new();
    // Ends of running attempts: (time, job, attempt).
    let mut ends: Vec<(Time, JobId, Attempt)> = Vec::new();
    // Poll and schedule the end of every attempt started.
    let poll = |p: &mut Logged<Scheduler>,
                t: Time,
                ends: &mut Vec<(Time, JobId, Attempt)>,
                started: &mut Vec<(JobId, Attempt)>| {
        for o in p.poll(t) {
            if let Output::Start {
                job: j, attempt, ..
            } = o
            {
                ends.push((t + job(j).1, j, attempt));
                started.push((j, attempt));
            }
        }
    };
    for w in 0..WORKERS {
        p.handle(Input::Worker(state(w)), Time::ORIGIN);
    }
    poll(&mut p, Time::ORIGIN, &mut ends, &mut started);
    let mut next_arrival = 0;
    let mut next_beat = Time::ORIGIN + HEARTBEAT;
    let mut lost = false;
    let mut done = 0;
    while done < JOBS {
        let arrival = (next_arrival < JOBS).then(|| job(next_arrival).0);
        let end = ends.iter().map(|e| e.0).min();
        let t = [arrival, end, Some(next_beat), (!lost).then_some(LOSS)]
            .into_iter()
            .flatten()
            .min()
            .unwrap();
        if arrival == Some(t) {
            let i = next_arrival;
            next_arrival += 1;
            let spec = JobSpec {
                id: i,
                demand: Resources::new().with(MEMORY, gb(job(i).2)),
                group: i / 10,
                work: Some(job(i).1),
                ..Default::default()
            };
            let info = TaskInfo {
                kind: "sig".into(),
                bidegree: (20 + (i / 10) as i64, 3),
                ..TaskInfo::default()
            };
            p.annotate(i, info);
            p.handle(Input::Submit(spec), t);
        } else if end == Some(t) {
            let k = ends.iter().position(|e| e.0 == t).unwrap();
            let (_, j, attempt) = ends.swap_remove(k);
            if flaky(j) && attempt == 1 {
                let why = "flaky".into();
                let kind = FailKind::Other;
                p.handle(
                    Input::Failed {
                        job: j,
                        attempt,
                        kind,
                        why,
                    },
                    t,
                );
            } else {
                // A report of an attempt lost with its worker is stale, and ignored.
                p.handle(Input::Done { job: j, attempt }, t);
                if p.explain(j).is_none() {
                    done += 1;
                    ends.retain(|e| e.1 != j);
                }
            }
        } else if t == next_beat {
            next_beat += HEARTBEAT;
            for w in 0..WORKERS {
                p.handle(Input::Worker(state(w)), t);
            }
        } else {
            lost = true;
            p.handle(Input::WorkerGone(0), t);
        }
        poll(&mut p, t, &mut ends, &mut started);
    }
    p.sink_mut().flush();
    started
}

/// The run exercised what it is meant to: every job started, retries after a failure and after
/// the worker loss.
fn check_run(started: &[(JobId, Attempt)]) {
    let first: Vec<JobId> = started.iter().filter(|s| s.1 == 1).map(|s| s.0).collect();
    assert_eq!(first.len(), JOBS as usize);
    let retried: Vec<JobId> = started.iter().filter(|s| s.1 > 1).map(|s| s.0).collect();
    assert!(retried.iter().any(|&j| flaky(j)), "{retried:?}");
    assert!(retried.iter().any(|&j| !flaky(j)), "{retried:?}");
}

/// Replaying a logged run's inputs and polls into a fresh policy reproduces every output.
#[test]
fn logged_run_replays_exactly() {
    let memory = Arc::new(Mutex::new(Vec::new()));
    check_run(&run(memory.clone()));
    let events = memory.lock().unwrap().clone();
    let submits = events.iter().filter(|e| {
        matches!(e, Event::Input { input: Input::Submit(_), info: Some(i), .. } if i.kind == "sig")
    });
    assert_eq!(submits.count(), JOBS as usize, "annotations are logged");
    assert!(events.iter().any(|e| matches!(e, Event::Sample { .. })));
    assert_eq!(replay(&mut policy(), events.clone()), polls(&events));
}

/// The JSON lines a [`JsonlSink`] writes read back as the logged events, and replay exactly.
#[test]
fn jsonl_round_trip_replays() {
    let dir = std::env::temp_dir().join(format!("whelm-log-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("run.jsonl");
    run(JsonlSink::create(&path).unwrap());
    let memory = Arc::new(Mutex::new(Vec::new()));
    run(memory.clone());
    let file = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let lines: Vec<String> = file.lines().map(Result::unwrap).collect();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(lines[0].starts_with(r#"{"type":"input","t":{"secs":0,"nanos":0},"input":{"worker""#));
    let back: Vec<Event> = lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(back, *memory.lock().unwrap());
    assert_eq!(replay(&mut policy(), back.clone()), polls(&back));
}
