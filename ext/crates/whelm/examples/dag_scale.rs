//! Scale check of the DAG layer: 10^5 units of a 10^3-node template, 10^8 jobs in all.

use std::{sync::Arc, time::Instant};

use whelm::{
    DagConfig, DagScheduler, DagTemplate, Input, JobId, JobSpec, Output, Policy, PolicyStats,
    Resources, TemplateNode, Unit,
};

/// Units on a side of the coarse grid: unit `k` waits for its left and upper neighbours.
const WIDTH: u64 = 316;
/// Units in the coarse graph.
const UNITS: u64 = 100_000;
/// Nodes of the shared template.
const NODES: u32 = 1_000;
/// Units entered at once to measure materialised state.
const OPEN_UNITS: u64 = 10_000;

/// A field of `/proc/self/status`, in bytes (Linux only; 0 elsewhere).
fn status_bytes(field: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            let line = s.lines().find(|l| l.starts_with(field))?;
            line.split_whitespace().nth(1)?.parse::<f64>().ok()
        })
        .map_or(0.0, |kb| kb * 1024.0)
}

/// Resident set size, bytes.
fn rss() -> f64 {
    status_bytes("VmRSS:")
}

/// A policy that starts every submitted job at once on worker 0, so that only the DAG layer is
/// measured; with `run` false it drops submissions instead.
struct Immediate {
    run: bool,
    queue: Vec<JobId>,
    submitted: u64,
}

impl Policy for Immediate {
    /// Queue submissions.
    fn handle(&mut self, input: Input, _now: f64) {
        if let Input::Submit(spec) = input {
            self.submitted += 1;
            if self.run {
                self.queue.push(spec.id);
            }
        }
    }

    /// Start everything queued.
    fn poll(&mut self, _now: f64) -> Vec<Output> {
        self.queue
            .drain(..)
            .map(|job| Output::Start {
                job,
                attempt: 1,
                worker: 0,
            })
            .collect()
    }

    /// Nothing is timed.
    fn next_wakeup(&self) -> Option<f64> {
        None
    }

    /// Nothing to say.
    fn explain(&self, _job: JobId) -> Option<String> {
        None
    }

    /// Nothing counted.
    fn stats(&self) -> PolicyStats {
        PolicyStats::default()
    }
}

/// A layered template: node `i` waits for node `i - 1` and one pseudo-random earlier node.
fn template() -> Arc<DagTemplate> {
    let nodes = (0..NODES)
        .map(|i| TemplateNode::Job(1.0 + f64::from(i % 7)))
        .collect();
    let edges = (1..NODES).flat_map(|i| [(i - 1, i), ((i * 7919 + 13) % i, i)]);
    Arc::new(DagTemplate::with_nodes(nodes, edges).expect("forward edges"))
}

/// Unit `k`: id `k`, leaves from `(k + 1) << 20`.
fn unit(t: &Arc<DagTemplate>, k: u64, deps: Vec<JobId>) -> Unit {
    let spec = JobSpec::new(0, Resources::mem(1), k);
    Unit::new(k, (k + 1) << 20, t.clone(), spec, deps)
}

/// A DAG layer over an [`Immediate`] policy.
fn layer(run: bool) -> DagScheduler<Immediate> {
    let config = DagConfig {
        rank_priority: true,
        ..DagConfig::default()
    };
    let policy = Immediate {
        run,
        queue: Vec::new(),
        submitted: 0,
    };
    DagScheduler::new(config, policy)
}

/// Measure materialised state, then declare the grid and run it to completion.
fn main() {
    let t = template();
    let virtual_nodes = UNITS * u64::from(NODES);
    println!(
        "template: {} nodes, {} edges; {UNITS} units, {virtual_nodes} jobs",
        t.len(),
        t.edge_count()
    );

    let before = rss();
    let mut open = layer(false);
    let clock = Instant::now();
    open.declare((0..OPEN_UNITS).map(|k| unit(&t, k, Vec::new())), 0.0)
        .unwrap();
    let opened = clock.elapsed().as_secs_f64();
    let grown = rss() - before;
    let s = open.dag_stats();
    println!(
        "entered {OPEN_UNITS} units at once in {opened:.2}s: {} nodes materialised, rss +{:.1} MB \
         = {:.1} B/node ({:.1} B/node counted by dag_stats)",
        s.nodes,
        grown / 1e6,
        grown / s.nodes as f64,
        s.node_bytes as f64 / s.nodes as f64,
    );
    // Measured while `open` is alive, so that freed memory is not reused.
    let before = rss();
    let mut d = layer(true);
    let clock = Instant::now();
    let mut edges = 0;
    for k in 0..UNITS {
        let mut deps = Vec::new();
        if k % WIDTH != 0 {
            deps.push(k - 1);
        }
        if k >= WIDTH {
            deps.push(k - WIDTH);
        }
        edges += deps.len();
        d.declare([unit(&t, k, deps)], 0.0).unwrap();
    }
    let declared = clock.elapsed().as_secs_f64();
    let grown = rss() - before;
    println!(
        "declared {UNITS} units / {edges} edges in {declared:.3}s ({:.2} us/unit, {:.1} ns per \
         virtual job); rss +{:.1} MB = {:.0} B/unit",
        declared * 1e6 / UNITS as f64,
        declared * 1e9 / virtual_nodes as f64,
        grown / 1e6,
        grown / UNITS as f64,
    );

    drop(open);
    let clock = Instant::now();
    let (mut completed, mut rounds, mut peak_nodes, mut peak_open) = (0u64, 0u64, 0, 0);
    let mut next_sample = 0;
    while completed < virtual_nodes {
        let out = d.poll(1.0);
        assert!(!out.is_empty(), "stalled after {completed} completions");
        for o in out {
            if let Output::Start { job, attempt, .. } = o {
                d.handle(Input::Done { job, attempt }, 1.0);
                completed += 1;
            }
        }
        rounds += 1;
        if completed >= next_sample {
            let s = d.dag_stats();
            peak_nodes = peak_nodes.max(s.nodes);
            peak_open = peak_open.max(s.open);
            next_sample = completed + 10_000_000;
        }
    }
    let ran = clock.elapsed().as_secs_f64();
    let s = d.dag_stats();
    println!(
        "ran {completed} jobs in {rounds} polls, {ran:.2}s ({:.0} ns/job, polls included); peak \
         sampled frontier {peak_open} units / {peak_nodes} nodes; left {} units",
        ran * 1e9 / completed as f64,
        s.units
    );
    println!("peak rss (VmHWM) {:.1} MB", status_bytes("VmHWM:") / 1e6);
}
