//! The event log round trip: a run logged through [`Logged`] is a trace `sched-sim` reads, and
//! replaying it with the same policy reproduces the run's placements.
#![cfg(feature = "sim")]

use std::sync::{Arc, Mutex};

use sched::{
    Config, EventSink, Policy, Resources, Scheduler, WorkerState,
    log::{Event, JsonlSink, Logged, TaskInfo},
    sim::{
        model::fit,
        run::{Baseline, SimSetup, simulate},
        trace::Trace,
    },
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

/// The placements in a log: `(job, worker, time)`.
fn placements(events: &[Event]) -> Vec<(u64, String, f64)> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Placed { t_s, job, worker } => Some((*job, worker.clone(), *t_s)),
            _ => None,
        })
        .collect()
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
    for w in 0..WORKERS {
        p.worker_update(state(w), 0.0);
        p.dispatch(0.0);
    }
    // Events: (time, kind, id), kind 0 = arrival, 1 = completion, 2 = heartbeat of worker id.
    let mut events: Vec<(f64, u8, u64)> = (0..JOBS).map(|i| (job(i).0, 0, i)).collect();
    let mut next_beat = HEARTBEAT;
    let mut done = 0;
    while done < JOBS {
        events.sort_by(|a, b| b.0.total_cmp(&a.0));
        let next = events.last().copied();
        let (t, kind, id) = match next {
            Some(e) if e.0 < next_beat => {
                events.pop();
                e
            }
            _ => {
                let t = next_beat;
                next_beat += HEARTBEAT;
                for w in 0..WORKERS {
                    p.worker_update(state(w), t);
                    for (j, _) in p.dispatch(t) {
                        events.push((t + job(j).1, 1, j));
                    }
                }
                continue;
            }
        };
        match kind {
            0 => {
                let mut spec = sched::JobSpec::new(id, Resources::mem_gb(job(id).2), id / 10);
                spec.work = Some(job(id).1);
                p.annotate(
                    id,
                    TaskInfo {
                        kind: "sig".into(),
                        bidegree: (20 + (id / 10) as i64, 3),
                        ..TaskInfo::default()
                    },
                );
                p.submit(spec, t);
            }
            _ => {
                p.completed(id, t);
                done += 1;
            }
        }
        for (j, _) in p.dispatch(t) {
            events.push((t + job(j).1, 1, j));
        }
    }
    p.sink_mut().flush();
}

/// The logged run, written as gzip JSONL, reads back as a trace with the run's records, and the
/// simulator's replay of that trace with the same policy places every job on the same worker at
/// the same time.
#[test]
fn logged_run_replays_exactly() {
    let dir = std::env::temp_dir().join(format!("sched-log-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("run.jsonl.gz");
    run(JsonlSink::create(&path).unwrap());
    let memory = Arc::new(Mutex::new(Vec::new()));
    run(memory.clone());
    let original = placements(&memory.lock().unwrap());
    assert_eq!(original.len(), JOBS as usize);

    let trace = Trace::load(&path).unwrap();
    assert_eq!(trace.workers.len(), WORKERS as usize);
    assert_eq!(trace.tasks.len(), JOBS as usize);
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
    let mut replay = placements(&replayed.lock().unwrap());
    // The replay numbers jobs by trace order; map back to the logged ids.
    for p in &mut replay {
        p.0 = trace.tasks[p.0 as usize].req;
    }
    let key = |v: &mut Vec<(u64, String, f64)>| v.sort_by_key(|a| a.0);
    let mut original = original;
    key(&mut original);
    key(&mut replay);
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
