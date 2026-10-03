//! The trace format: gzip-compressed JSONL with `worker`, `task` and `sample` records, or the
//! event log of [`whelm::log`].

use std::{collections::HashMap, io::BufRead, path::Path};

use whelm::{Attempt, Input, MEM, Output, log::TaskInfo};
use serde::Deserialize;

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

/// One line of either format.
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
    // The event log ([`whelm::log::Event`]): folded into worker and task records.
    Input {
        t_s: f64,
        input: Input,
        #[serde(default)]
        info: Option<TaskInfo>,
    },
    Poll {
        t_s: f64,
        #[serde(default)]
        out: Vec<Output>,
    },
    Reserved {},
}

/// A logged job being folded into a task record.
struct Pending {
    ready_s: f64,
    est_gb: f64,
    group: u64,
    info: Option<TaskInfo>,
    /// Its attempts' starts: (attempt, time, worker).
    starts: Vec<(Attempt, f64, String)>,
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
                Line::Input { t_s, input, info } => match input {
                    Input::Worker(w) => {
                        let i = index(&mut t, &w.id.to_string());
                        let tw = &mut t.workers[i];
                        tw.class = w.class;
                        tw.budget_gb = w.budget[MEM] as f64 / 1e9;
                        tw.slots = w.slots;
                    }
                    Input::Submit(spec) => {
                        // A duplicate submission of a live job is ignored by the policy too.
                        pending.entry(spec.id).or_insert(Pending {
                            ready_s: t_s,
                            est_gb: spec.demand[MEM] as f64 / 1e9,
                            group: spec.group,
                            info,
                            starts: Vec::new(),
                        });
                    }
                    Input::Cancel(job) => {
                        pending.remove(&job);
                    }
                    Input::Done { job, attempt } => {
                        let Some(p) = pending.remove(&job) else {
                            continue;
                        };
                        // The task record keeps the attempt that finished.
                        let Some((_, placed_s, worker)) =
                            p.starts.into_iter().find(|s| s.0 == attempt)
                        else {
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
                    Input::Failed { .. } | Input::WorkerGone(_) => {}
                },
                Line::Poll { t_s, out } => {
                    for o in out {
                        match o {
                            Output::Start {
                                job,
                                attempt,
                                worker,
                            } => {
                                if let Some(p) = pending.get_mut(&job) {
                                    p.starts.push((attempt, t_s, worker.to_string()));
                                }
                            }
                            Output::GaveUp(g) => {
                                pending.remove(&g.job);
                            }
                            _ => {}
                        }
                    }
                }
                Line::Reserved {} => {}
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

    /// The baseline a worker reporting `baseline_excl` would send at time `t` (GB): the same rolling
    /// floor as [`floor_baseline`](Self::floor_baseline), of each sample's resident memory minus
    /// `scale` times the estimates of the jobs it was running (at least 0). `None` before the
    /// first sample.
    pub fn floor_baseline_excl(&self, w: usize, t: f64, window_s: f64, scale: f64) -> Option<f64> {
        let tw = &self.workers[w];
        let idx = ((t - tw.join_s) / window_s).floor().max(0.0);
        let from = tw.join_s + (idx - 1.0).max(0.0) * window_s;
        let lo = tw.samples.partition_point(|s| s.t_s < from);
        let hi = tw.samples.partition_point(|s| s.t_s <= t);
        tw.samples[lo..hi]
            .iter()
            .map(|s| (s.rss_gb - scale * s.reserved_gb).max(0.0))
            .min_by(f64::total_cmp)
    }

    /// Resident memory above idle per GB of estimate running, one value per sample with at least
    /// `min_reserved_gb` of estimates running: how much of an estimate a job actually occupies,
    /// summed over the jobs of a sample.
    pub fn usage_ratios(&self, idle: &[f64], min_reserved_gb: f64) -> Vec<f64> {
        let mut v: Vec<f64> = self
            .workers
            .iter()
            .zip(idle)
            .flat_map(|(w, &i)| {
                w.samples
                    .iter()
                    .filter(move |s| s.reserved_gb >= min_reserved_gb)
                    .map(move |s| (s.rss_gb - i).max(0.0) / s.reserved_gb)
            })
            .collect();
        v.sort_by(f64::total_cmp);
        v
    }

    /// Samples whose resident memory exceeded the worker's budget, and all samples.
    pub fn samples_over_budget(&self) -> (usize, usize) {
        let mut over = 0;
        let mut all = 0;
        for w in &self.workers {
            for s in &w.samples {
                all += 1;
                over += usize::from(s.rss_gb > w.budget_gb);
            }
        }
        (over, all)
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

#[cfg(test)]
mod tests {
    use whelm::{FailKind, GaveUp, JobSpec, Resources, WorkerState, log::Event};

    use super::*;

    /// Load `lines` as a trace file.
    fn load(name: &str, lines: &[String]) -> Trace {
        let path =
            std::env::temp_dir().join(format!("whelm-sim-{name}-{}.jsonl", std::process::id()));
        std::fs::write(&path, lines.join("\n")).unwrap();
        let t = Trace::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        t
    }

    /// The standalone format: workers, samples, and the tasks that ran.
    #[test]
    fn reads_standalone_records() {
        let lines = [
            r#"{"type":"worker","id":"a:1","gpu":"h200","budget_gb":100.0,"slots":4}"#,
            r#"{"type":"sample","t_s":5.0,"worker":"a:1","rss_gb":10.0,"reserved_gb":2.0,"running":1}"#,
            r#"{"type":"task","req":7,"kind":"zero","bidegree":[10,2],"est_gb":2.0,"ready_s":1.0,"placed_s":3.0,"done_s":9.0,"worker":"a:1"}"#,
            r#"{"type":"task","req":8,"kind":"sig","bidegree":[10,2],"est_gb":2.0,"ready_s":1.0}"#,
        ]
        .map(String::from);
        let t = load("standalone", &lines);
        assert_eq!(t.workers.len(), 1);
        let w = &t.workers[0];
        assert_eq!((w.class.as_str(), w.budget_gb, w.slots), ("h200", 100.0, 4));
        assert_eq!((w.join_s, w.samples.len()), (3.0, 1));
        assert_eq!(t.tasks.len(), 1);
        let task = &t.tasks[0];
        assert!(task.zero);
        assert_eq!((task.req, task.placed_s, task.done_s), (7, 3.0, 9.0));
        assert_eq!(task.group, group_id(10, 2));
    }

    /// The event log: a task record keeps the attempt that finished (a retry, or a speculative
    /// attempt that beat the original); cancelled and given-up jobs leave none.
    #[test]
    fn reads_event_log_attempts() {
        let input = |t_s: f64, input: Input| Event::Input {
            t_s,
            input,
            info: None,
        };
        let start = |job, attempt, worker| Output::Start {
            job,
            attempt,
            worker,
        };
        let poll = |t_s: f64, out: Vec<Output>| Event::Poll { t_s, out };
        let worker = |id| WorkerState::new(id, "l40s", 8, Resources::mem_gb(50.0));
        let submit = |id| Input::Submit(JobSpec::new(id, Resources::mem_gb(1.5), 9));
        let events = vec![
            input(0.0, Input::Worker(worker(1))),
            input(0.0, Input::Worker(worker(2))),
            Event::Input {
                t_s: 1.0,
                input: submit(1),
                info: Some(Box::new(TaskInfo {
                    kind: "zero".into(),
                    bidegree: (12, 3),
                    ..TaskInfo::default()
                })),
            },
            input(1.0, submit(2)),
            poll(1.0, vec![start(1, 1, 1), start(2, 1, 1)]),
            input(
                2.0,
                Input::Failed {
                    job: 2,
                    attempt: 1,
                    kind: FailKind::Other,
                    why: "x".into(),
                },
            ),
            poll(2.0, vec![start(2, 2, 2)]),
            poll(3.0, vec![start(1, 2, 2)]),
            input(4.0, Input::Done { job: 1, attempt: 2 }),
            poll(
                4.0,
                vec![Output::Stop {
                    job: 1,
                    attempt: 1,
                    worker: 1,
                }],
            ),
            input(6.0, Input::Done { job: 2, attempt: 2 }),
            input(6.0, submit(3)),
            input(6.0, Input::Cancel(3)),
            input(6.0, submit(4)),
            poll(6.0, vec![start(4, 1, 1)]),
            poll(
                7.0,
                vec![Output::GaveUp(GaveUp {
                    job: 4,
                    tried: Vec::new(),
                    retryable: false,
                })],
            ),
            Event::Sample {
                t_s: 7.0,
                worker: "1".into(),
                rss_gb: 3.0,
                baseline_gb: 1.0,
                reserved_gb: 0.0,
                running: 0,
                dev_per_task_gb: 0.5,
            },
        ];
        let lines: Vec<String> = events
            .iter()
            .map(|e| serde_json::to_string(e).unwrap())
            .collect();
        let t = load("log", &lines);
        let names: Vec<&str> = t.workers.iter().map(|w| w.name.as_str()).collect();
        assert_eq!(names, ["1", "2"]);
        assert_eq!((t.workers[0].budget_gb, t.workers[0].slots), (50.0, 8));
        assert_eq!(t.workers[0].samples.len(), 1);
        let got: Vec<(u64, f64, f64, f64, usize, bool, u64)> = t
            .tasks
            .iter()
            .map(|x| {
                (
                    x.req, x.ready_s, x.placed_s, x.done_s, x.worker, x.zero, x.group,
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                (1, 1.0, 3.0, 4.0, 1, true, group_id(12, 3)),
                (2, 1.0, 2.0, 6.0, 1, false, 9),
            ]
        );
        assert_eq!(t.tasks[0].est_gb, 1.5);
    }
}
