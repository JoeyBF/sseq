//! A replayable log of a policy's inputs and outputs, readable by the trace simulator.

use std::collections::{BTreeSet, HashMap};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{DEV, Input, Instant, JobId, MEM, Output, Policy, PolicyStats, WorkerId, WorkerState};

/// What a job is, for the simulator (optional; Nassau's vocabulary). Without it a logged job
/// replays as a signature task of its group.
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
/// back with [`replay`] reproduces every output. [`Sample`](Event::Sample) and
/// [`Reserved`](Event::Reserved) summarise the policy's state for the trace reader. Worker ids in
/// those two are written as strings (the trace format names workers).
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
    /// A job reserved a worker.
    Reserved {
        /// When.
        t_s: f64,
        /// The holder.
        job: JobId,
        /// The worker.
        worker: String,
    },
}

/// Whether a logged quantity is zero (left out of the line).
#[cfg(feature = "serde")]
fn is_zero(x: &f64) -> bool {
    *x == 0.0
}

/// Where events go.
pub trait EventSink: Send {
    /// Record one event.
    fn record(&mut self, event: &Event);
    /// Push buffered events to storage.
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
/// `sched-sim --trace`).
#[cfg(feature = "log")]
pub struct JsonlSink {
    out: Box<dyn std::io::Write + Send>,
}

#[cfg(feature = "log")]
impl JsonlSink {
    /// A sink writing to `out`.
    pub fn new(out: impl std::io::Write + Send + 'static) -> Self {
        Self { out: Box::new(out) }
    }

    /// A sink writing to the file `path`, gzip-compressed if it ends in `.gz`. Lines are buffered:
    /// call [`EventSink::flush`] periodically so a killed process keeps most of its log (a gzip
    /// member is completed per flush, which `sched-sim` reads).
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
/// Heartbeats are also summarised as samples at most every `sample_every` seconds per worker, and
/// reservations are logged when made (aging is not logged: it follows from submission times and
/// the age limit). Under a [`DagScheduler`](crate::DagScheduler), wrap the inner policy
/// (`DagScheduler<Logged<Scheduler>>`): the DAG's own operations are method calls, not inputs.
pub struct Logged<P> {
    inner: P,
    sink: Box<dyn EventSink>,
    sample_every: f64,
    last_sample: HashMap<WorkerId, Instant>,
    reserved: BTreeSet<(JobId, WorkerId)>,
    info: HashMap<JobId, TaskInfo>,
}

/// Bytes to GB.
fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

impl<P: Policy> Logged<P> {
    /// Log `inner`'s events to `sink`, with a heartbeat sample at most every 60 s per worker.
    pub fn new(inner: P, sink: impl EventSink + 'static) -> Self {
        Self {
            inner,
            sink: Box::new(sink),
            sample_every: 60.0,
            last_sample: HashMap::new(),
            reserved: BTreeSet::new(),
            info: HashMap::new(),
        }
    }

    /// Sample heartbeats at most every `seconds` per worker (0: every heartbeat).
    pub fn sample_every(mut self, seconds: f64) -> Self {
        self.sample_every = seconds;
        self
    }

    /// Attach what a job is, to be logged with its submission (call before submitting it).
    pub fn annotate(&mut self, job: JobId, info: TaskInfo) {
        self.info.insert(job, info);
    }

    /// The wrapped policy.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// The wrapped policy, mutably. Changes made through it are not logged, so a log of a run
    /// that uses this may not replay.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// The sink.
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

    /// Log the reservations made since the last poll.
    fn log_reservations(&mut self, now: Instant) {
        let current: BTreeSet<(JobId, WorkerId)> = self
            .inner
            .stats()
            .reservations
            .iter()
            .map(|r| (r.job, r.worker))
            .collect();
        for &(job, w) in current.difference(&self.reserved) {
            self.sink.record(&Event::Reserved {
                t_s: now,
                job,
                worker: w.to_string(),
            });
        }
        self.reserved = current;
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
                self.reserved.retain(|r| r.1 != w);
            }
            _ => {}
        }
    }

    /// Forwarded, then logged with any new reservation.
    fn poll(&mut self, now: Instant) -> Vec<Output> {
        let out = self.inner.poll(now);
        self.sink.record(&Event::Poll {
            t_s: now,
            out: out.clone(),
        });
        self.log_reservations(now);
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
pub fn replay<P: Policy + ?Sized>(
    policy: &mut P,
    events: impl IntoIterator<Item = Event>,
) -> Vec<(Instant, Vec<Output>)> {
    let mut out = Vec::new();
    for e in events {
        match e {
            Event::Input { t_s, input, .. } => policy.handle(input, t_s),
            Event::Poll { t_s, .. } => out.push((t_s, policy.poll(t_s))),
            Event::Sample { .. } | Event::Reserved { .. } => {}
        }
    }
    out
}

/// The polls recorded in a log, with their times and outputs.
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
    use crate::{
        Attempt, Config, FailKind, JobSpec, Learn, Resources, Scheduler, Speculate, SpeedPolicy,
    };

    /// A configuration exercising learning, speculation, retries and reservations.
    fn config() -> Config {
        let mut c = Config::default();
        c.speed.policy = SpeedPolicy::FastestFirst;
        c.speed.learn = Some(Learn::default());
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
