//! The event log as a simulator input: a run logged through [`Logged`] is a trace `sched-sim`
//! reads, and replaying that trace with the same policy reproduces the run's placements.

use std::sync::{Arc, Mutex};

use sched::{
    Attempt, Config, EventSink, Input, JobId, JobSpec, Output, Policy, Resources, Scheduler,
    WorkerState,
    log::{Event, JsonlSink, Logged, TaskInfo},
};
use sched_sim::{
    model::fit,
    run::{Baseline, SimSetup, simulate},
    trace::Trace,
};

const WORKERS: u64 = 3;
const SLOTS: usize = 2;
const BUDGET_GB: f64 = 10.0;
const JOBS: u64 = 80;
const HEARTBEAT: f64 = 60.0;

/// Job `i`: arrival, run time (on the reference class, alone or not) and demand (GB). Times are
/// chosen so that no two events coincide.
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
) -> Vec<(JobId, String, f64)> {
    let mut v = Vec::new();
    for e in events {
        if let Event::Poll { t_s, out } = e {
            for o in out {
                if let Output::Start { job, worker, .. } = o {
                    let (job, worker) = name(*job, *worker);
                    v.push((job, worker, *t_s));
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

/// Run the workload with a simple event-driven driver (jobs run at speed 1 whatever the
/// concurrency, every worker heartbeats each `HEARTBEAT` seconds), logging to `sink`.
fn run(sink: impl EventSink + 'static) {
    let mut p = Logged::new(policy(), sink);
    let state = |w: u64| WorkerState::new(w, "h200", SLOTS, Resources::mem_gb(BUDGET_GB));
    // Completions: (time, job, attempt).
    let mut ends: Vec<(f64, JobId, Attempt)> = Vec::new();
    let poll = |p: &mut Logged<Scheduler>, t: f64, ends: &mut Vec<(f64, JobId, Attempt)>| {
        for o in p.poll(t) {
            if let Output::Start {
                job: j, attempt, ..
            } = o
            {
                ends.push((t + job(j).1, j, attempt));
            }
        }
    };
    for w in 0..WORKERS {
        p.handle(Input::Worker(state(w)), 0.0);
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
                p.handle(Input::Worker(state(w)), t);
                poll(&mut p, t, &mut ends);
            }
            continue;
        }
        if Some(t) == arrival {
            let id = next_arrival;
            next_arrival += 1;
            let mut spec = JobSpec::new(id, Resources::mem_gb(job(id).2), id / 10);
            spec.work = Some(job(id).1);
            p.annotate(
                id,
                TaskInfo {
                    kind: "sig".into(),
                    bidegree: (20 + (id / 10) as i64, 3),
                    ..TaskInfo::default()
                },
            );
            p.handle(Input::Submit(spec), t);
        } else {
            let (_, job, attempt) = ends.pop().unwrap();
            p.handle(Input::Done { job, attempt }, t);
            done += 1;
        }
        poll(&mut p, t, &mut ends);
    }
    p.sink_mut().flush();
}

/// The logged run, written as gzip JSONL, reads back as a trace with the run's records, and the
/// simulator's replay of that trace with the same policy places every job on the same worker at
/// the same time.
#[test]
fn logged_run_replays_exactly() {
    let dir = std::env::temp_dir().join(format!("sched-sim-log-{}", std::process::id()));
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
            ("h200", SLOTS, BUDGET_GB)
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
            (a.2 - b.2).abs() < 1e-6,
            "job {} placed at {} vs {}",
            a.0,
            a.2,
            b.2
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
