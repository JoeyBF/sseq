//! The trace format: gzip-compressed JSONL with `worker`, `task` and `sample` records, or the
//! event log of [`crate::log`].

use std::{collections::HashMap, io::BufRead, path::Path};

use serde::Deserialize;

use crate::log::TaskInfo;

/// A worker of the trace.
#[derive(Clone, Debug)]
pub struct TraceWorker {
    /// Its name in the trace (`host:port`).
    pub name: String,
    /// GPU class, e.g. `"h200"`.
    pub class: String,
    /// Host memory budget, GB.
    pub budget_gb: f64,
    /// Execution slots.
    pub slots: usize,
    /// When it joined: its first sample or first placement, whichever is earlier.
    pub join_s: f64,
    /// Heartbeat samples, by time.
    pub samples: Vec<Sample>,
}

/// One per-minute heartbeat sample.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    /// Time, seconds.
    pub t_s: f64,
    /// Resident memory, GB.
    pub rss_gb: f64,
    /// Sum of estimates of the jobs running, GB.
    pub reserved_gb: f64,
    /// Jobs running.
    pub running: usize,
}

/// A task (job) of the trace.
#[derive(Clone, Debug)]
pub struct TraceTask {
    /// Request id (unique).
    pub req: u64,
    /// A bidegree's "zero" task (vs a signature task).
    pub zero: bool,
    /// The bidegree `(n, s)`.
    pub bidegree: (i64, i64),
    /// The bidegree as a group id.
    pub group: u64,
    /// Memory estimate, GB.
    pub est_gb: f64,
    /// Size covariates used by the service-model fit.
    pub target: f64,
    /// See `target`.
    pub next: f64,
    /// When it became ready (seconds).
    pub ready_s: f64,
    /// When production placed it.
    pub placed_s: f64,
    /// When production finished it.
    pub done_s: f64,
    /// Index into [`Trace::workers`] of the worker production used.
    pub worker: usize,
    /// Requests that must complete first.
    pub deps: Vec<u64>,
    /// Groups that must complete first (zero tasks only).
    pub after_groups: Vec<u64>,
    /// The signature's Milnor exponents (empty for zero tasks).
    pub sig: Vec<u32>,
}

/// A parsed trace.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    /// Workers, in order of first appearance.
    pub workers: Vec<TraceWorker>,
    /// Tasks, in file order.
    pub tasks: Vec<TraceTask>,
}

/// The group id of a bidegree.
pub fn group_id(n: i64, s: i64) -> u64 {
    ((n as u64) << 20) | (s as u64 & 0xfffff)
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Line {
    Worker {
        id: String,
        gpu: String,
        budget_gb: f64,
        slots: usize,
    },
    Task(TaskLine),
    Sample {
        t_s: f64,
        worker: String,
        rss_gb: f64,
        reserved_gb: f64,
        running: usize,
    },
    // The event log (`crate::log`): folded into task records.
    Submit {
        t_s: f64,
        job: u64,
        est_gb: f64,
        group: u64,
        #[serde(default)]
        info: Option<TaskInfo>,
    },
    Placed {
        t_s: f64,
        job: u64,
        worker: String,
    },
    Moved {
        t_s: f64,
        job: u64,
        to: String,
    },
    Done {
        t_s: f64,
        job: u64,
    },
    Failed {
        job: u64,
    },
    Cancel {
        job: u64,
    },
    Gone {},
    Reserved {},
}

/// A logged job being folded into a task record.
struct Pending {
    ready_s: f64,
    est_gb: f64,
    group: u64,
    info: Option<TaskInfo>,
    placed: Option<(f64, String)>,
}

#[derive(Deserialize)]
struct TaskLine {
    req: u64,
    kind: String,
    bidegree: (i64, i64),
    est_gb: f64,
    target: Option<f64>,
    next: Option<f64>,
    ready_s: f64,
    placed_s: Option<f64>,
    done_s: Option<f64>,
    worker: Option<String>,
    #[serde(default)]
    deps: Vec<u64>,
    #[serde(default)]
    after_groups: Vec<(i64, i64)>,
    #[serde(default)]
    sig: Vec<u32>,
}

impl Trace {
    /// Read a trace from a (possibly gzip-compressed) JSONL file. Tasks that never ran in
    /// production are dropped (they cannot be given a service time).
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader: Box<dyn BufRead> = if path.extension().is_some_and(|e| e == "gz") {
            Box::new(std::io::BufReader::new(flate2::read::MultiGzDecoder::new(
                file,
            )))
        } else {
            Box::new(std::io::BufReader::new(file))
        };
        let mut t = Trace::default();
        let mut pending: HashMap<u64, Pending> = HashMap::new();
        let mut names: HashMap<String, usize> = HashMap::new();
        let mut index = |t: &mut Trace, name: &str| -> usize {
            *names.entry(name.to_string()).or_insert_with(|| {
                t.workers.push(TraceWorker {
                    name: name.to_string(),
                    class: String::new(),
                    budget_gb: 0.0,
                    slots: 0,
                    join_s: f64::INFINITY,
                    samples: Vec::new(),
                });
                t.workers.len() - 1
            })
        };
        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let rec: Line =
                serde_json::from_str(&line).map_err(|e| format!("line {}: {e}", i + 1))?;
            match rec {
                Line::Worker {
                    id,
                    gpu,
                    budget_gb,
                    slots,
                } => {
                    let w = index(&mut t, &id);
                    let w = &mut t.workers[w];
                    w.class = gpu;
                    w.budget_gb = budget_gb;
                    w.slots = slots;
                }
                Line::Sample {
                    t_s,
                    worker,
                    rss_gb,
                    reserved_gb,
                    running,
                } => {
                    let w = index(&mut t, &worker);
                    t.workers[w].samples.push(Sample {
                        t_s,
                        rss_gb,
                        reserved_gb,
                        running,
                    });
                }
                Line::Submit {
                    t_s,
                    job,
                    est_gb,
                    group,
                    info,
                } => {
                    pending.insert(
                        job,
                        Pending {
                            ready_s: t_s,
                            est_gb,
                            group,
                            info,
                            placed: None,
                        },
                    );
                }
                Line::Placed { t_s, job, worker }
                | Line::Moved {
                    t_s,
                    job,
                    to: worker,
                } => {
                    if let Some(p) = pending.get_mut(&job) {
                        p.placed = Some((t_s, worker));
                    }
                }
                Line::Failed { job } => {
                    // The next placement is a retry; the task record keeps the last attempt.
                    if let Some(p) = pending.get_mut(&job) {
                        p.placed = None;
                    }
                }
                Line::Cancel { job } => {
                    pending.remove(&job);
                }
                Line::Done { t_s, job } => {
                    let Some(p) = pending.remove(&job) else {
                        continue;
                    };
                    let Some((placed_s, worker)) = p.placed else {
                        continue;
                    };
                    let worker = index(&mut t, &worker);
                    let info = p.info.unwrap_or_else(|| TaskInfo {
                        kind: "sig".into(),
                        ..TaskInfo::default()
                    });
                    t.tasks.push(TraceTask {
                        req: job,
                        zero: info.kind == "zero",
                        bidegree: info.bidegree,
                        // Without a bidegree, the logged group itself.
                        group: if info.bidegree == (0, 0) {
                            p.group
                        } else {
                            group_id(info.bidegree.0, info.bidegree.1)
                        },
                        est_gb: p.est_gb,
                        target: info.target.unwrap_or(1.0),
                        next: info.next.unwrap_or(0.0),
                        ready_s: p.ready_s,
                        placed_s,
                        done_s: t_s,
                        worker,
                        deps: info.deps,
                        sig: info.sig,
                        after_groups: info
                            .after_groups
                            .iter()
                            .map(|&(n, s)| group_id(n, s))
                            .collect(),
                    });
                }
                Line::Gone {} | Line::Reserved {} => {}
                Line::Task(x) => {
                    let (Some(placed_s), Some(done_s), Some(worker)) =
                        (x.placed_s, x.done_s, x.worker)
                    else {
                        continue;
                    };
                    let worker = index(&mut t, &worker);
                    t.tasks.push(TraceTask {
                        req: x.req,
                        zero: x.kind == "zero",
                        bidegree: x.bidegree,
                        group: group_id(x.bidegree.0, x.bidegree.1),
                        est_gb: x.est_gb,
                        target: x.target.unwrap_or(1.0),
                        next: x.next.unwrap_or(0.0),
                        ready_s: x.ready_s,
                        placed_s,
                        done_s,
                        worker,
                        deps: x.deps,
                        sig: x.sig,
                        after_groups: x
                            .after_groups
                            .iter()
                            .map(|&(n, s)| group_id(n, s))
                            .collect(),
                    });
                }
            }
        }
        for task in &t.tasks {
            let w = &mut t.workers[task.worker];
            w.join_s = w.join_s.min(task.placed_s);
        }
        for w in &mut t.workers {
            w.samples.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
            if let Some(s) = w.samples.first() {
                w.join_s = w.join_s.min(s.t_s);
            }
            if !w.join_s.is_finite() {
                w.join_s = 0.0;
            }
        }
        Ok(t)
    }

    /// The baseline production's worker reported at time `t` (GB): its memory gate's rolling RSS
    /// floor, the minimum resident set over the current and the previous `window_s` window
    /// (counted from the worker's join). It includes whatever jobs were resident, so it overstates
    /// the task-free footprint on a busy worker. `None` before the first sample.
    pub fn floor_baseline(&self, w: usize, t: f64, window_s: f64) -> Option<f64> {
        let tw = &self.workers[w];
        let idx = ((t - tw.join_s) / window_s).floor().max(0.0);
        let from = tw.join_s + (idx - 1.0).max(0.0) * window_s;
        let lo = tw.samples.partition_point(|s| s.t_s < from);
        let hi = tw.samples.partition_point(|s| s.t_s <= t);
        tw.samples[lo..hi]
            .iter()
            .map(|s| s.rss_gb)
            .min_by(f64::total_cmp)
    }

    /// Per-worker baseline memory (GB), an alternative `reported_baseline` for the replay: the median
    /// resident memory over the worker's samples with nothing running, or, for a worker never
    /// sampled idle, the median of that over its class (falling back to all workers).
    pub fn idle_baselines(&self) -> Vec<f64> {
        /// The upper median, or `None` for no samples.
        fn median(mut v: Vec<f64>) -> Option<f64> {
            if v.is_empty() {
                return None;
            }
            v.sort_by(f64::total_cmp);
            Some(v[v.len() / 2])
        }
        let idle = |w: &TraceWorker| -> Vec<f64> {
            w.samples
                .iter()
                .filter(|s| s.running == 0)
                .map(|s| s.rss_gb)
                .collect()
        };
        let all = median(self.workers.iter().flat_map(idle).collect()).unwrap_or(0.0);
        self.workers
            .iter()
            .map(|w| {
                median(idle(w)).unwrap_or_else(|| {
                    median(
                        self.workers
                            .iter()
                            .filter(|o| o.class == w.class)
                            .flat_map(idle)
                            .collect(),
                    )
                    .unwrap_or(all)
                })
            })
            .collect()
    }
}
