//! A run logged through [`Logged`] reads back as a trace and replays to the same placements.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use whelm::{
    job::JobId,
    log::{Event, EventSink, JsonlSink, Logged, TaskInfo},
    policy::Attempt,
    prelude::*,
};
use whelm_sim::{
    model::fit,
    run::{Baseline, SimSetup, simulate},
    trace::Trace,
};

/// Workers in the logged run.
const WORKERS: u64 = 3;
/// Slots per worker.
const WORKER_SLOTS: usize = 2;
/// Memory budget per worker.
const BUDGET_GB: f64 = 10.0;
/// Jobs in the logged run.
const JOBS: u64 = 80;
/// Heartbeat period, seconds.
const HEARTBEAT: f64 = 60.0;

/// Job `i`: arrival, run time (on the reference class, alone or not) and demand (GB).
///
/// Times are chosen so that no two events coincide.
fn job(i: u64) -> (f64, f64, f64) {
    let x = i as f64;
    (
        0.37 + 1.3 * x,
        5.0 + (i * 7 % 11) as f64 + 0.123 * x,
        1.0 + (i * 5 % 6) as f64,
    )
}

/// The starts in a log, `(job, worker, time)`, with `name` mapping the logged ids.
fn placements(
    events: &[Event],
    name: impl Fn(JobId, u64) -> (JobId, String),
) -> Vec<(JobId, String, Time)> {
    let mut v = Vec::new();
    for e in events {
        if let Event::Poll { t, out } = e {
            for o in out {
                if let Output::Start { job, worker, .. } = o {
                    let (job, worker) = name(*job, *worker);
                    v.push((job, worker, *t));
                }
            }
        }
    }
    v.sort_by_key(|p| p.0);
    v
}

/// The policy under test.
fn policy() -> Scheduler {
    Scheduler::new(Config::best_fit())
}

/// Run the workload with a simple event-driven driver, logging to `sink`.
///
/// Jobs run at speed 1 whatever the concurrency, and every worker heartbeats each `HEARTBEAT`
/// seconds.
fn run(sink: impl EventSink + 'static) {
    let mut p = Logged::new(policy(), sink);
    let state = |w: u64| WorkerState {
        id: w,
        class: "h200".into(),
        capacity: Resources::new()
            .with(MEMORY, gb(BUDGET_GB))
            .with(SLOTS, WORKER_SLOTS as u64),
        ..Default::default()
    };
    // Completions: (time, job, attempt).
    let mut ends: Vec<(f64, JobId, Attempt)> = Vec::new();
    let poll = |p: &mut Logged<Scheduler>, t: f64, ends: &mut Vec<(f64, JobId, Attempt)>| {
        for o in p.poll(Time(Duration::from_secs_f64(t))) {
            if let Output::Start {
                job: j, attempt, ..
            } = o
            {
                ends.push((t + job(j).1, j, attempt));
            }
        }
    };
    for w in 0..WORKERS {
        p.handle(Input::Worker(state(w)), Time::ORIGIN);
        poll(&mut p, 0.0, &mut ends);
    }
    let mut next_arrival = 0;
    let mut next_beat = HEARTBEAT;
    let mut done = 0;
    while done < JOBS {
        ends.sort_by(|a, b| b.0.total_cmp(&a.0));
        let arrival = (next_arrival < JOBS).then(|| job(next_arrival).0);
        let end = ends.last().map(|e| e.0);
        let t = [arrival, end, Some(next_beat)]
            .into_iter()
            .flatten()
            .fold(f64::INFINITY, f64::min);
        if t == next_beat {
            next_beat += HEARTBEAT;
            for w in 0..WORKERS {
                p.handle(Input::Worker(state(w)), Time(Duration::from_secs_f64(t)));
                poll(&mut p, t, &mut ends);
            }
            continue;
        }
        if Some(t) == arrival {
            let id = next_arrival;
            next_arrival += 1;
            let spec = JobSpec {
                demand: Resources::new().with(MEMORY, gb(job(id).2)),
                group: id / 10,
                work: Some(Duration::from_secs_f64(job(id).1)),
                ..Default::default()
            };
            p.annotate(
                id,
                TaskInfo {
                    kind: "sig".into(),
                    bidegree: (20 + (id / 10) as i64, 3),
                    ..TaskInfo::default()
                },
            );
            p.handle(
                Input::Submit { job: id, spec },
                Time(Duration::from_secs_f64(t)),
            );
        } else {
            let (_, job, attempt) = ends.pop().unwrap();
            p.handle(
                Input::Done { job, attempt },
                Time(Duration::from_secs_f64(t)),
            );
            done += 1;
        }
        poll(&mut p, t, &mut ends);
    }
    p.sink_mut().flush();
}

/// The logged run reads back as a trace, and replaying it with the same policy reproduces it.
///
/// The run is written as gzip JSONL; the trace has the run's records, and the simulator's replay
/// places every job on the same worker at the same time.
#[test]
fn logged_run_replays_exactly() {
    let dir = std::env::temp_dir().join(format!("whelm-sim-log-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("run.jsonl.gz");
    run(JsonlSink::create(&path).unwrap());
    let memory = Arc::new(Mutex::new(Vec::new()));
    run(memory.clone());
    let original = placements(&memory.lock().unwrap(), |j, w| (j, w.to_string()));
    assert_eq!(original.len(), JOBS as usize);

    let trace = Trace::load(&path).unwrap();
    assert_eq!(trace.workers.len(), WORKERS as usize);
    assert_eq!(trace.tasks.len(), JOBS as usize);
    for w in &trace.workers {
        assert_eq!(
            (w.class.as_str(), w.slots, w.budget_gb),
            ("h200", WORKER_SLOTS, BUDGET_GB)
        );
    }
    for t in &trace.tasks {
        let (arrival, run, demand) = job(t.req);
        assert!((t.ready_s - arrival).abs() < 1e-9);
        assert!((t.done_s - t.placed_s - run).abs() < 1e-6, "task {}", t.req);
        assert!((t.est_gb - demand).abs() < 1e-9);
        assert_eq!(t.bidegree, (20 + (t.req / 10) as i64, 3));
    }

    let (model, _, work) = fit(&trace);
    let setup = SimSetup {
        trace: &trace,
        work: &work,
        model: &model,
        baseline: Baseline::PerWorker(vec![0.0; WORKERS as usize]),
        replay_rss: false,
        heartbeat_s: HEARTBEAT,
        closed_loop: None,
        big_gb: 7.5,
        explain: None,
        est_scale: 1.0,
        usage: None,
    };
    let replayed = Arc::new(Mutex::new(Vec::new()));
    let m = simulate(
        &setup,
        "replay",
        Box::new(Logged::new(policy(), replayed.clone())),
    );
    assert_eq!(m.completed, JOBS as usize);
    // The replay numbers jobs and workers by trace order; map back to the logged ids.
    let replay = placements(&replayed.lock().unwrap(), |j, w| {
        (
            trace.tasks[j as usize].req,
            trace.workers[w as usize].name.clone(),
        )
    });
    assert_eq!(replay.len(), original.len());
    for (a, b) in original.iter().zip(&replay) {
        assert_eq!((a.0, &a.1), (b.0, &b.1), "job {} placed elsewhere", a.0);
        assert!(
            (a.2 - b.2).max(b.2 - a.2) < Duration::from_micros(1),
            "job {} placed at {:?} vs {:?}",
            a.0,
            a.2,
            b.2
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
