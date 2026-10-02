//! Scale check of the DAG layer: declare a million pending jobs, then run them all.

use std::time::Instant;

use sched::{
    Config, DagConfig, DagJob, DagScheduler, Input, JobSpec, Output, Policy, Resources, Scheduler,
    WorkerState,
};

/// Resident set size of this process, in MB (Linux only; 0 elsewhere).
fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            let line = s.lines().find(|l| l.starts_with("VmRSS:"))?;
            line.split_whitespace().nth(1)?.parse::<f64>().ok()
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

/// Declare, then run to completion, a million jobs, reporting time and memory.
fn main() {
    const GROUPS: u64 = 100;
    const PER_GROUP: u64 = 10_000;
    let jobs_total = GROUPS * PER_GROUP;
    let base = rss_mb();
    let mut d = DagScheduler::new(
        DagConfig {
            rank_priority: true,
            ..DagConfig::default()
        },
        Scheduler::new(Config::default()),
    );
    let clock = Instant::now();
    let mut edges = 0;
    for g in 0..GROUPS {
        let first = g * PER_GROUP;
        let jobs: Vec<DagJob> = (first..first + PER_GROUP)
            .map(|id| {
                // Each group's zero job depends on the previous group's last job; every other job
                // on its zero job and two earlier jobs of the group.
                let deps = if id == first {
                    first.checked_sub(1).into_iter().collect()
                } else {
                    let mut v = vec![first, first + (id - first) / 2, id - 1];
                    v.dedup();
                    v
                };
                edges += deps.len();
                DagJob {
                    spec: JobSpec::new(id, Resources::mem(1), g),
                    deps,
                    work_estimate: None,
                    passthrough: false,
                    local: false,
                }
            })
            .collect();
        d.declare(jobs, 0.0).unwrap();
    }
    let declared = clock.elapsed().as_secs_f64();
    let grown = rss_mb() - base;
    println!(
        "declared {jobs_total} jobs / {edges} edges in {declared:.2}s ({:.2}us/job); pending {}, \
         rss +{grown:.0} MB ({:.0} B/job)",
        declared * 1e6 / jobs_total as f64,
        d.dag_stats().pending,
        grown * 1048576.0 / jobs_total as f64,
    );
    let clock = Instant::now();
    let worker = WorkerState::new(0, "x", 64, Resources::mem(1 << 40));
    d.handle(Input::Worker(worker), 1.0);
    let mut completed = 0;
    while completed < jobs_total {
        let out = d.poll(1.0);
        assert!(!out.is_empty(), "stalled after {completed} completions");
        for o in out {
            if let Output::Start { job, attempt, .. } = o {
                d.handle(Input::Done { job, attempt }, 1.0);
                completed += 1;
            }
        }
    }
    d.forget_completed_below(jobs_total);
    let completed = clock.elapsed().as_secs_f64();
    println!(
        "ran all in {completed:.2}s ({:.2}us/job, polls included); left: {:?}",
        completed * 1e6 / jobs_total as f64,
        d.dag_stats()
    );
}
