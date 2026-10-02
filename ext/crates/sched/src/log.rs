//! A structured event log of placement decisions, readable by the trace simulator.

use std::collections::{BTreeSet, HashMap};

use crate::{Dispatch, Instant, JobId, JobSpec, Policy, PolicyStats, WorkerId, WorkerState};

/// What a job is, for the simulator (optional; Nassau's vocabulary). Without it a logged job
/// replays as a signature task of its group.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// One logged event. Times are on the policy's clock, in seconds. Worker ids are written as
/// strings (the trace format names workers).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "lowercase"))]
pub enum Event {
    /// A worker's capacity, when first seen or when it changes.
    Worker {
        /// The worker.
        id: String,
        /// Its class.
        gpu: String,
        /// Its memory budget, GB.
        budget_gb: f64,
        /// Its slots.
        slots: usize,
        /// Its device memory capacity, GB (0: unknown).
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_zero"))]
        dev_cap_gb: f64,
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
        /// Jobs placed there.
        running: usize,
        /// The worker's learned device memory per job, GB (0: unknown).
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_zero"))]
        dev_per_task_gb: f64,
    },
    /// A worker left.
    Gone {
        /// When.
        t_s: f64,
        /// The worker.
        worker: String,
    },
    /// A job was submitted (became ready).
    Submit {
        /// When.
        t_s: f64,
        /// The job.
        job: JobId,
        /// Its demand, GB.
        est_gb: f64,
        /// Its device memory demand, GB.
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "is_zero"))]
        dev_gb: f64,
        /// Its group.
        group: u64,
        /// Its priority, if any.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        priority: Option<i64>,
        /// Its work estimate, if any.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        work: Option<f64>,
        /// What it is.
        #[cfg_attr(
            feature = "serde",
            serde(default, skip_serializing_if = "Option::is_none")
        )]
        info: Option<TaskInfo>,
    },
    /// A job was placed.
    Placed {
        /// When.
        t_s: f64,
        /// The job.
        job: JobId,
        /// Where.
        worker: String,
    },
    /// A running job was moved (spoliation).
    Moved {
        /// When.
        t_s: f64,
        /// The job.
        job: JobId,
        /// From.
        from: String,
        /// To.
        to: String,
    },
    /// A job completed.
    Done {
        /// When.
        t_s: f64,
        /// The job.
        job: JobId,
    },
    /// A running job failed.
    Failed {
        /// When.
        t_s: f64,
        /// The job.
        job: JobId,
        /// The caller's reason.
        why: String,
    },
    /// A job was withdrawn.
    Cancel {
        /// When.
        t_s: f64,
        /// The job.
        job: JobId,
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

/// A [`Policy`] that records every event it sees, and its own decisions, to an [`EventSink`].
///
/// Worker capacity is logged when first seen or changed; heartbeats as samples at most every
/// `sample_every` seconds per worker; reservations when made (aging is not logged: it follows
/// from submission times and the age limit).
pub struct Logged<P> {
    inner: P,
    sink: Box<dyn EventSink>,
    sample_every: f64,
    capacity: HashMap<WorkerId, (String, usize, crate::Resources)>,
    last_sample: HashMap<WorkerId, Instant>,
    reserved: BTreeSet<(JobId, WorkerId)>,
    info: HashMap<JobId, TaskInfo>,
    now: Instant,
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
            capacity: HashMap::new(),
            last_sample: HashMap::new(),
            reserved: BTreeSet::new(),
            info: HashMap::new(),
            now: 0.0,
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

    /// The wrapped policy, mutably (changes made through it are not logged).
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// The sink.
    pub fn sink_mut(&mut self) -> &mut dyn EventSink {
        &mut *self.sink
    }

    /// Log a placement decision and any new reservation.
    fn log_dispatch(&mut self, d: &Dispatch, now: Instant) {
        for &(job, w) in &d.start {
            self.sink.record(&Event::Placed {
                t_s: now,
                job,
                worker: w.to_string(),
            });
        }
        for p in &d.preempt {
            self.sink.record(&Event::Moved {
                t_s: now,
                job: p.job,
                from: p.from.to_string(),
                to: p.to.to_string(),
            });
        }
        let stats = self.inner.stats();
        let now_reserved: BTreeSet<(JobId, WorkerId)> = stats
            .reservations
            .iter()
            .map(|r| (r.job, r.worker))
            .collect();
        for &(job, w) in now_reserved.difference(&self.reserved) {
            self.sink.record(&Event::Reserved {
                t_s: now,
                job,
                worker: w.to_string(),
            });
        }
        self.reserved = now_reserved;
    }

    /// Log a worker's capacity if new or changed, and a sample if due.
    fn log_worker(&mut self, w: &WorkerState, now: Instant) {
        let cap = (w.class.clone(), w.slots, w.budget);
        if self.capacity.get(&w.id) != Some(&cap) {
            self.sink.record(&Event::Worker {
                id: w.id.to_string(),
                gpu: w.class.clone(),
                budget_gb: gb(w.budget.mem),
                slots: w.slots,
                dev_cap_gb: gb(w.budget.dev),
            });
            self.capacity.insert(w.id, cap);
        }
        let due = self
            .last_sample
            .get(&w.id)
            .is_none_or(|&t| now - t >= self.sample_every);
        if due {
            let load = self
                .inner
                .stats()
                .workers
                .into_iter()
                .find(|l| l.id == w.id);
            self.sink.record(&Event::Sample {
                t_s: now,
                worker: w.id.to_string(),
                rss_gb: gb(w.reported_used.mem),
                baseline_gb: gb(w.reported_baseline.mem),
                reserved_gb: load.as_ref().map_or(0.0, |l| gb(l.placed.mem)),
                running: load.map_or(0, |l| l.running),
                dev_per_task_gb: gb(w.dev_per_task),
            });
            self.last_sample.insert(w.id, now);
        }
    }
}

impl<P: Policy> Policy for Logged<P> {
    /// Logged, then forwarded.
    fn submit(&mut self, job: JobSpec, now: Instant) {
        self.now = now;
        self.sink.record(&Event::Submit {
            t_s: now,
            job: job.id,
            est_gb: gb(job.demand.mem),
            dev_gb: gb(job.demand.dev),
            group: job.group,
            priority: job.priority,
            work: job.work,
            info: self.info.remove(&job.id),
        });
        self.inner.submit(job, now);
    }

    /// Logged, then forwarded.
    fn cancel(&mut self, job: JobId) {
        self.sink.record(&Event::Cancel { t_s: self.now, job });
        self.inner.cancel(job);
    }

    /// Forwarded, then logged (the sample shows the policy's bookkeeping after the update).
    fn worker_update(&mut self, w: WorkerState, now: Instant) {
        self.now = now;
        self.inner.worker_update(w.clone(), now);
        self.log_worker(&w, now);
    }

    /// Logged, then forwarded.
    fn worker_gone(&mut self, w: WorkerId, now: Instant) {
        self.now = now;
        self.sink.record(&Event::Gone {
            t_s: now,
            worker: w.to_string(),
        });
        self.capacity.remove(&w);
        self.last_sample.remove(&w);
        self.reserved.retain(|r| r.1 != w);
        self.inner.worker_gone(w, now);
    }

    /// Logged, then forwarded.
    fn completed(&mut self, job: JobId, now: Instant) {
        self.now = now;
        self.sink.record(&Event::Done { t_s: now, job });
        self.inner.completed(job, now);
    }

    /// Logged, then forwarded.
    fn failed(&mut self, job: JobId, now: Instant, why: &str) {
        self.now = now;
        self.sink.record(&Event::Failed {
            t_s: now,
            job,
            why: why.to_string(),
        });
        self.inner.failed(job, now, why);
    }

    /// Forwarded, then the placements logged.
    fn dispatch(&mut self, now: Instant) -> Vec<(JobId, WorkerId)> {
        self.now = now;
        let start = self.inner.dispatch(now);
        let d = Dispatch {
            start,
            preempt: Vec::new(),
        };
        self.log_dispatch(&d, now);
        d.start
    }

    /// Forwarded, then the placements and preemptions logged.
    fn dispatch_full(&mut self, now: Instant) -> Dispatch {
        self.now = now;
        let d = self.inner.dispatch_full(now);
        self.log_dispatch(&d, now);
        d
    }

    /// Forwarded.
    fn explain(&self, job: JobId) -> Option<String> {
        self.inner.explain(job)
    }

    /// Forwarded.
    fn stats(&self) -> PolicyStats {
        self.inner.stats()
    }

    /// Forwarded.
    fn next_wakeup(&self) -> Option<Instant> {
        self.inner.next_wakeup()
    }
}
