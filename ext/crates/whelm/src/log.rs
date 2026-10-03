//! A replayable log of a policy's inputs and outputs, readable by the trace simulator.
//!
//! [`Logged`] wraps a [`Policy`] and records an [`Event`] to an [`EventSink`] for every input it
//! handles and every poll it answers. A policy is deterministic, so those records are the whole
//! run: [`replay`] feeds them to a fresh policy built the same way and gets every output back,
//! which [`polls`] reads straight from the log. That reproduces a production run offline, and
//! `whelm-sim --trace` replays a written log against other configurations.
//!
//! The sinks provided keep events in a `Vec<Event>`, in an `Arc<Mutex<Vec<Event>>>` (readable from
//! outside while the run goes on), or, with the `log` feature, write them as JSON lines with
//! `JsonlSink`.
//!
//! Log a run in which a job fails once and is retried on the other worker, then replay it:
//!
//! ```
//! use std::sync::{Arc, Mutex};
//!
//! use whelm::{
//!     Config, FailKind, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState,
//!     log::{self, Event, Logged},
//! };
//!
//! let events = Arc::new(Mutex::new(Vec::<Event>::new()));
//! let mut p = Logged::new(Scheduler::new(Config::default()), events.clone());
//! for w in [1, 2] {
//!     p.handle(
//!         Input::Worker(WorkerState::new(w, "x", 1, Resources::mem_gb(8.0))),
//!         0.0,
//!     );
//! }
//! p.handle(
//!     Input::Submit(JobSpec::new(7, Resources::mem_gb(1.0), 0)),
//!     0.0,
//! );
//! assert_eq!(
//!     p.poll(0.0),
//!     [Output::Start {
//!         job: 7,
//!         attempt: 1,
//!         worker: 1
//!     }]
//! );
//! let why = "boom".to_string();
//! p.handle(
//!     Input::Failed {
//!         job: 7,
//!         attempt: 1,
//!         kind: FailKind::Other,
//!         why,
//!     },
//!     5.0,
//! );
//! assert_eq!(
//!     p.poll(5.0),
//!     [Output::Start {
//!         job: 7,
//!         attempt: 2,
//!         worker: 2
//!     }]
//! );
//! p.handle(Input::Done { job: 7, attempt: 2 }, 9.0);
//! assert_eq!(p.poll(9.0), []);
//!
//! let events = events.lock().unwrap().clone();
//! // Five inputs, three polls, and a sample of each worker's first heartbeat.
//! assert_eq!(events.len(), 10);
//! let logged = log::polls(&events);
//! assert_eq!(
//!     logged[1],
//!     (
//!         5.0,
//!         vec![Output::Start {
//!             job: 7,
//!             attempt: 2,
//!             worker: 2
//!         }]
//!     )
//! );
//! assert_eq!(
//!     log::replay(&mut Scheduler::new(Config::default()), events),
//!     logged
//! );
//! ```

use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{DEV, Input, Instant, JobId, MEM, Output, Policy, PolicyStats, WorkerId, WorkerState};

/// What a job is, for the simulator (optional; Nassau's vocabulary). Without it a logged job
/// replays as a signature task of its group.
///
/// The policy never reads it: [`Logged::annotate`] attaches it to the job's submission in the log.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TaskInfo {
    /// `"zero"` or `"sig"`.
    pub kind: String,
    /// The bidegree `(n, s)`.
    pub bidegree: (i64, i64),
    /// Size covariate of the service-model fit (target dimension).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub target: Option<f64>,
    /// Size covariate of the service-model fit (next dimension).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub next: Option<f64>,
    /// Jobs that had to complete first.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub deps: Vec<JobId>,
    /// Bidegrees `(n, s)` that had to complete first (zero tasks).
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub after_groups: Vec<(i64, i64)>,
    /// The signature's Milnor exponents.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub sig: Vec<u32>,
}

/// One logged event. Times are on the policy's clock, in seconds.
///
/// The [`Input`](Event::Input) and [`Poll`](Event::Poll) records are the whole run: feeding them
/// back with [`replay`] reproduces every output, reservations included. [`Sample`](Event::Sample)
/// summarises a worker's state for the trace reader, with its id written as a string (the trace
/// format names workers).
///
/// Events can be written by hand, e.g. to script a run for [`replay`]:
///
/// ```
/// use whelm::{Input, JobSpec, Resources, log::Event};
///
/// let submit = Event::Input {
///     t_s: 0.0,
///     input: Input::Submit(JobSpec::new(1, Resources::mem_gb(1.0), 0)),
///     info: None,
/// };
/// let poll = Event::Poll {
///     t_s: 0.0,
///     out: Vec::new(),
/// };
/// # let _ = (submit, poll);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "lowercase"))]
pub enum Event {
    /// An input, exactly as the policy handled it.
    Input {
        /// When.
        t_s: f64,
        /// The input.
        input: Input,
        /// What a submitted job is ([`Logged::annotate`]).
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        info: Option<Box<TaskInfo>>,
    },
    /// A poll, and what it returned.
    Poll {
        /// When.
        t_s: f64,
        /// Its outputs, in order.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Vec::is_empty")
        )]
        out: Vec<Output>,
    },
    /// A heartbeat (rate-limited): reported resident memory and the policy's own bookkeeping.
    Sample {
        /// When.
        t_s: f64,
        /// The worker.
        worker: String,
        /// Reported resident memory, GB.
        rss_gb: f64,
        /// Reported baseline, GB.
        baseline_gb: f64,
        /// Sum of the demands placed there, GB.
        reserved_gb: f64,
        /// Live attempts there.
        running: usize,
        /// The worker's learned device memory per job, GB (0: unknown).
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_zero"))]
        dev_per_task_gb: f64,
    },
}

/// Whether a logged quantity is zero (left out of the line).
#[cfg(feature = "serde")]
fn is_zero(x: &f64) -> bool {
    *x == 0.0
}

/// Where events go.
///
/// [`Logged`] calls [`record`](Self::record) for each event as it happens, and never calls
/// [`flush`](Self::flush) itself: the caller does, through [`Logged::sink_mut`].
///
/// A sink that only counts the starts it sees, readable from outside through a shared handle:
///
/// ```
/// use std::sync::{
///     Arc,
///     atomic::{AtomicUsize, Ordering},
/// };
///
/// use whelm::{
///     Config, EventSink, Input, JobSpec, Output, Policy, Resources, Scheduler, WorkerState,
///     log::{Event, Logged},
/// };
///
/// struct CountStarts(Arc<AtomicUsize>);
///
/// impl EventSink for CountStarts {
///     fn record(&mut self, event: &Event) {
///         if let Event::Poll { out, .. } = event {
///             let starts = out
///                 .iter()
///                 .filter(|o| matches!(o, Output::Start { .. }))
///                 .count();
///             self.0.fetch_add(starts, Ordering::Relaxed);
///         }
///     }
/// }
///
/// let starts = Arc::new(AtomicUsize::new(0));
/// let mut p = Logged::new(
///     Scheduler::new(Config::default()),
///     CountStarts(starts.clone()),
/// );
/// p.handle(
///     Input::Worker(WorkerState::new(1, "x", 2, Resources::mem_gb(8.0))),
///     0.0,
/// );
/// for id in 1..=3 {
///     p.handle(
///         Input::Submit(JobSpec::new(id, Resources::mem_gb(1.0), 0)),
///         0.0,
///     );
/// }
/// p.poll(0.0);
/// // Two slots: the third job waits.
/// assert_eq!(starts.load(Ordering::Relaxed), 2);
/// ```
pub trait EventSink: Send {
    /// Record one event.
    fn record(&mut self, event: &Event);
    /// Push buffered events to storage. The default does nothing, for sinks that do not buffer.
    fn flush(&mut self) {}
}

impl EventSink for Vec<Event> {
    /// Keep it in memory.
    fn record(&mut self, event: &Event) {
        self.push(event.clone());
    }
}

impl EventSink for std::sync::Arc<std::sync::Mutex<Vec<Event>>> {
    /// Keep it in memory, shared with whoever holds another handle.
    fn record(&mut self, event: &Event) {
        self.lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event.clone());
    }
}

/// Writes events as JSON lines, optionally gzip-compressed (the trace format: feed the file to
/// `whelm-sim --trace`).
///
/// Each line is one [`Event`], tagged by `"type"`, and reads back with `serde_json` as the same
/// event, so a written log replays like an in-memory one. Here the lines go to a buffer shared
/// with the caller:
///
/// ```
/// use std::{
///     io::Write,
///     sync::{Arc, Mutex},
/// };
///
/// use whelm::{
///     Config, Input, JobSpec, Policy, Resources, Scheduler, WorkerState,
///     log::{self, Event, JsonlSink, Logged},
/// };
///
/// /// A writer appending to a buffer the caller also holds.
/// #[derive(Clone, Default)]
/// struct Shared(Arc<Mutex<Vec<u8>>>);
///
/// impl Write for Shared {
///     fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
///         self.0.lock().unwrap().write(buf)
///     }
///
///     fn flush(&mut self) -> std::io::Result<()> {
///         Ok(())
///     }
/// }
///
/// let buf = Shared::default();
/// let mut p = Logged::new(
///     Scheduler::new(Config::default()),
///     JsonlSink::new(buf.clone()),
/// );
/// p.handle(
///     Input::Worker(WorkerState::new(1, "x", 1, Resources::mem_gb(8.0))),
///     0.0,
/// );
/// p.handle(
///     Input::Submit(JobSpec::new(1, Resources::mem_gb(1.0), 0)),
///     0.0,
/// );
/// p.poll(0.0);
///
/// let text = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
/// let lines: Vec<&str> = text.lines().collect();
/// assert!(lines[0].starts_with(r#"{"type":"input","t_s":0.0,"input":{"worker":{"id":1,"#));
/// assert!(lines[1].starts_with(r#"{"type":"sample","t_s":0.0,"worker":"1","#));
/// assert_eq!(
///     lines[3],
///     r#"{"type":"poll","t_s":0.0,"out":[{"start":{"job":1,"attempt":1,"worker":1}}]}"#
/// );
///
/// let events: Vec<Event> = lines
///     .iter()
///     .map(|l| serde_json::from_str(l).unwrap())
///     .collect();
/// let mut fresh = Scheduler::new(Config::default());
/// assert_eq!(log::replay(&mut fresh, events.clone()), log::polls(&events));
/// ```
#[cfg(feature = "log")]
pub struct JsonlSink {
    out: Box<dyn std::io::Write + Send>,
}

#[cfg(feature = "log")]
impl JsonlSink {
    /// A sink writing to `out`, one line per event. The sink owns `out`: to read what it wrote,
    /// pass a writer that shares its buffer, as in the [type's example](JsonlSink).
    pub fn new(out: impl std::io::Write + Send + 'static) -> Self {
        Self { out: Box::new(out) }
    }

    /// A sink writing to the file `path`, gzip-compressed if it ends in `.gz`. Lines are buffered:
    /// call [`EventSink::flush`] periodically so a killed process keeps most of its log (a gzip
    /// member is completed per flush, which `whelm-sim` reads).
    ///
    /// Log to a compressed file and flush after each round of the event loop:
    ///
    /// ```no_run
    /// use std::path::Path;
    ///
    /// use whelm::{
    ///     Config, EventSink, Policy, Scheduler,
    ///     log::{JsonlSink, Logged},
    /// };
    ///
    /// let sink = JsonlSink::create(Path::new("run.jsonl.gz")).unwrap();
    /// let mut p = Logged::new(Scheduler::new(Config::default()), sink);
    /// loop {
    ///     // Handle the events that arrived, then:
    ///     p.poll(0.0);
    ///     p.sink_mut().flush();
    /// #   break;
    /// }
    /// ```
    pub fn create(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(if path.extension().is_some_and(|e| e == "gz") {
            Self::new(GzMembers::new(file))
        } else {
            Self::new(std::io::BufWriter::new(file))
        })
    }
}

#[cfg(feature = "log")]
impl EventSink for JsonlSink {
    /// One line per event; I/O errors are dropped (logging must not stop placement).
    fn record(&mut self, event: &Event) {
        if let Ok(line) = serde_json::to_string(event) {
            let _ = writeln!(self.out, "{line}");
        }
    }

    /// Flush the writer.
    fn flush(&mut self) {
        let _ = self.out.flush();
    }
}

#[cfg(feature = "log")]
use std::io::Write as _;

/// A gzip writer that ends a member on every flush, so the file is readable up to the last
/// flush even if the process dies.
#[cfg(feature = "log")]
struct GzMembers {
    file: Option<std::fs::File>,
    gz: Option<flate2::write::GzEncoder<Vec<u8>>>,
}

#[cfg(feature = "log")]
impl GzMembers {
    /// A writer appending members to `file`.
    fn new(file: std::fs::File) -> Self {
        Self {
            file: Some(file),
            gz: None,
        }
    }
}

#[cfg(feature = "log")]
impl std::io::Write for GzMembers {
    /// Compress into the current member.
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.gz
            .get_or_insert_with(|| {
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default())
            })
            .write(buf)
    }

    /// Finish the member and append it to the file.
    fn flush(&mut self) -> std::io::Result<()> {
        if let (Some(gz), Some(file)) = (self.gz.take(), self.file.as_mut()) {
            file.write_all(&gz.finish()?)?;
            file.flush()?;
        }
        Ok(())
    }
}

#[cfg(feature = "log")]
impl Drop for GzMembers {
    /// Write out the last member.
    fn drop(&mut self) {
        let _ = std::io::Write::flush(self);
    }
}

/// A [`Policy`] that records every input it handles and every poll's outputs to an
/// [`EventSink`], so that [`replay`] can reproduce the run.
///
/// Heartbeats are also summarised as samples at most every `sample_every` seconds per worker.
/// Under a [`DagScheduler`](crate::DagScheduler), wrap the inner policy
/// (`DagScheduler<Logged<Scheduler>>`): the DAG's own operations are method calls, not inputs.
///
/// The [module example](self) logs a run and replays it.
pub struct Logged<P> {
    inner: P,
    sink: Box<dyn EventSink>,
    sample_every: f64,
    last_sample: HashMap<WorkerId, Instant>,
    info: HashMap<JobId, TaskInfo>,
}

/// Bytes to GB.
fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

impl<P: Policy> Logged<P> {
    /// Log `inner`'s events to `sink`, with a heartbeat sample at most every 60 s per worker
    /// (see [`sample_every`](Self::sample_every)).
    pub fn new(inner: P, sink: impl EventSink + 'static) -> Self {
        Self {
            inner,
            sink: Box::new(sink),
            sample_every: 60.0,
            last_sample: HashMap::new(),
            info: HashMap::new(),
        }
    }

    /// Sample heartbeats at most every `seconds` per worker (0: every heartbeat).
    ///
    /// A sample shows what the worker reported next to what the policy placed there. A worker's
    /// first heartbeat is always sampled; later ones only once `seconds` have passed:
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    ///
    /// use whelm::{
    ///     Config, Input, JobSpec, Policy, Resources, Scheduler, WorkerState,
    ///     log::{Event, Logged},
    /// };
    ///
    /// let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    /// let inner = Scheduler::new(Config::default());
    /// let mut p = Logged::new(inner, events.clone()).sample_every(30.0);
    /// let mut w = WorkerState::new(1, "x", 2, Resources::mem_gb(8.0));
    /// p.handle(Input::Worker(w.clone()), 0.0);
    /// p.handle(
    ///     Input::Submit(JobSpec::new(1, Resources::mem_gb(2.0), 0)),
    ///     0.0,
    /// );
    /// p.poll(0.0);
    /// w.reported_used = Resources::mem_gb(1.5);
    /// for t in [10.0, 20.0, 30.0] {
    ///     p.handle(Input::Worker(w.clone()), t);
    /// }
    ///
    /// let samples: Vec<Event> = events
    ///     .lock()
    ///     .unwrap()
    ///     .iter()
    ///     .filter(|e| matches!(e, Event::Sample { .. }))
    ///     .cloned()
    ///     .collect();
    /// assert_eq!(samples.len(), 2);
    /// assert_eq!(
    ///     samples[1],
    ///     Event::Sample {
    ///         t_s: 30.0,
    ///         worker: "1".into(),
    ///         rss_gb: 1.5,
    ///         baseline_gb: 0.0,
    ///         reserved_gb: 2.0,
    ///         running: 1,
    ///         dev_per_task_gb: 0.0,
    ///     },
    /// );
    /// ```
    pub fn sample_every(mut self, seconds: f64) -> Self {
        self.sample_every = seconds;
        self
    }

    /// Attach what a job is, to be logged with its submission (call before submitting it).
    ///
    /// The annotation is used once, by the job's next submission:
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    ///
    /// use whelm::{
    ///     Config, Input, JobSpec, Policy, Resources, Scheduler,
    ///     log::{Event, Logged, TaskInfo},
    /// };
    ///
    /// let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    /// let mut p = Logged::new(Scheduler::new(Config::default()), events.clone());
    /// let info = TaskInfo {
    ///     kind: "zero".into(),
    ///     bidegree: (20, 3),
    ///     ..TaskInfo::default()
    /// };
    /// p.annotate(1, info.clone());
    /// p.handle(
    ///     Input::Submit(JobSpec::new(1, Resources::mem_gb(1.0), 0)),
    ///     0.0,
    /// );
    /// p.handle(Input::Cancel(1), 1.0);
    /// p.handle(
    ///     Input::Submit(JobSpec::new(1, Resources::mem_gb(1.0), 0)),
    ///     2.0,
    /// );
    ///
    /// let events = events.lock().unwrap();
    /// let Event::Input { info: first, .. } = &events[0] else {
    ///     panic!()
    /// };
    /// let Event::Input { info: again, .. } = &events[2] else {
    ///     panic!()
    /// };
    /// assert_eq!(first.as_deref(), Some(&info));
    /// assert_eq!(*again, None);
    /// ```
    pub fn annotate(&mut self, job: JobId, info: TaskInfo) {
        self.info.insert(job, info);
    }

    /// The wrapped policy. Reads are not logged, and need not be: they do not change what
    /// the policy does.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// The wrapped policy, mutably. Changes made through it are not logged, so a log of a run
    /// that uses this may not replay.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// The sink, e.g. to [`flush`](EventSink::flush) it (see `JsonlSink::create`, feature
    /// `log`).
    pub fn sink_mut(&mut self) -> &mut dyn EventSink {
        &mut *self.sink
    }

    /// Log a sample of a worker's heartbeat, if one is due.
    fn log_sample(&mut self, w: &WorkerState, now: Instant) {
        let due = self
            .last_sample
            .get(&w.id)
            .is_none_or(|&t| now - t >= self.sample_every);
        if !due {
            return;
        }
        let load = self
            .inner
            .stats()
            .workers
            .into_iter()
            .find(|l| l.id == w.id);
        self.sink.record(&Event::Sample {
            t_s: now,
            worker: w.id.to_string(),
            rss_gb: gb(w.reported_used[MEM]),
            baseline_gb: gb(w.reported_baseline[MEM]),
            reserved_gb: load.as_ref().map_or(0.0, |l| gb(l.placed[MEM])),
            running: load.map_or(0, |l| l.running),
            dev_per_task_gb: gb(w.per_task[DEV]),
        });
        self.last_sample.insert(w.id, now);
    }
}

impl<P: Policy> Policy for Logged<P> {
    /// Logged (with the job's annotation, for a submission), then forwarded; a heartbeat is
    /// sampled after it is forwarded, so that the sample shows the policy's bookkeeping.
    fn handle(&mut self, input: Input, now: Instant) {
        let info = match &input {
            Input::Submit(spec) => self.info.remove(&spec.id).map(Box::new),
            _ => None,
        };
        self.sink.record(&Event::Input {
            t_s: now,
            input: input.clone(),
            info,
        });
        self.inner.handle(input.clone(), now);
        match input {
            Input::Worker(w) => self.log_sample(&w, now),
            Input::WorkerGone(w) => {
                self.last_sample.remove(&w);
            }
            _ => {}
        }
    }

    /// Forwarded, then logged.
    fn poll(&mut self, now: Instant) -> Vec<Output> {
        let out = self.inner.poll(now);
        self.sink.record(&Event::Poll {
            t_s: now,
            out: out.clone(),
        });
        out
    }

    /// Forwarded.
    fn next_wakeup(&self) -> Option<Instant> {
        self.inner.next_wakeup()
    }

    /// Forwarded.
    fn explain(&self, job: JobId) -> Option<String> {
        self.inner.explain(job)
    }

    /// Forwarded.
    fn stats(&self) -> PolicyStats {
        self.inner.stats()
    }
}

/// Feed a log's inputs and polls to `policy`, in order, and return what each poll returned, with
/// its time. Given a fresh policy built as the logged one was (same [`Config`](crate::Config),
/// same admission rule), the result equals [`polls`] of the same log.
///
/// The outputs are recomputed, not copied from the log: a scripted log with empty polls replays
/// into the policy's actual decisions.
///
/// ```
/// use whelm::{
///     Config, Input, JobSpec, Output, Resources, Scheduler, WorkerState,
///     log::{self, Event},
/// };
///
/// let w = WorkerState::new(1, "x", 1, Resources::mem_gb(8.0));
/// let events = vec![
///     Event::Input {
///         t_s: 0.0,
///         input: Input::Worker(w),
///         info: None,
///     },
///     Event::Input {
///         t_s: 0.0,
///         input: Input::Submit(JobSpec::new(1, Resources::mem_gb(1.0), 0)),
///         info: None,
///     },
///     Event::Poll {
///         t_s: 0.0,
///         out: Vec::new(),
///     },
/// ];
/// let replayed = log::replay(&mut Scheduler::new(Config::default()), events.clone());
/// assert_eq!(
///     replayed,
///     [(
///         0.0,
///         vec![Output::Start {
///             job: 1,
///             attempt: 1,
///             worker: 1
///         }]
///     )]
/// );
/// assert_eq!(log::polls(&events), [(0.0, vec![])]);
/// ```
pub fn replay<P: Policy + ?Sized>(
    policy: &mut P,
    events: impl IntoIterator<Item = Event>,
) -> Vec<(Instant, Vec<Output>)> {
    let mut out = Vec::new();
    for e in events {
        match e {
            Event::Input { t_s, input, .. } => policy.handle(input, t_s),
            Event::Poll { t_s, .. } => out.push((t_s, policy.poll(t_s))),
            Event::Sample { .. } => {}
        }
    }
    out
}

/// The polls recorded in a log, with their times and outputs: what [`replay`] should reproduce.
/// Inputs and samples are skipped (see the [module example](self)).
pub fn polls<'a>(events: impl IntoIterator<Item = &'a Event>) -> Vec<(Instant, Vec<Output>)> {
    events
        .into_iter()
        .filter_map(|e| match e {
            Event::Poll { t_s, out } => Some((*t_s, out.clone())),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{Attempt, Config, FailKind, JobSpec, Resources, Scheduler, Speculate, Timing};

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
            speed: if w == 3 { 3.0 } else { 1.0 },
            reported_used: Resources::mem_gb(used_gb),
            ..WorkerState::new(w, "x", 2, Resources::mem_gb(10.0))
        }
    }

    /// Run a workload with failures, worker churn, speculation and a cancellation through
    /// `Logged`, and return how many outputs of each kind it produced: (starts, stops, retries).
    fn run(sink: impl EventSink + 'static) -> (usize, usize, usize) {
        let mut p = Logged::new(Scheduler::new(config()), sink).sample_every(0.0);
        // (time, job, attempt) of each running attempt's end.
        let mut ends: Vec<(u32, JobId, Attempt)> = Vec::new();
        let (mut starts, mut stops, mut retries) = (0, 0, 0);
        for w in 1..=2 {
            p.handle(Input::Worker(worker(w, 0.0)), 0.0);
        }
        for t in 0..400u32 {
            let now = t as f64;
            if t < 120 && t % 3 == 0 {
                let i = (t / 3) as JobId;
                let mut spec = JobSpec::new(i, Resources::mem_gb(1.0 + (i * 5 % 6) as f64), i / 8);
                spec.work = Some(5.0 + (i * 7 % 11) as f64);
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
                .any(|l| l.starts_with(r#"{"type":"input","t_s":0.0,"input":{"worker""#))
        );
    }
}
