//! The recording wrapper, [`Logged`].

use std::collections::HashMap;

#[cfg(doc)]
use super::replay;
use super::{Event, EventSink, TaskInfo};
use crate::{DEV, Input, Instant, JobId, MEM, Output, Policy, PolicyStats, WorkerId, WorkerState};

/// A [`Policy`] that records every input it handles and every poll's outputs to an
/// [`EventSink`], so that [`replay`] can reproduce the run.
///
/// Heartbeats are also summarised as samples at most every `sample_every` seconds per worker.
/// Under a [`DagScheduler`](crate::DagScheduler), wrap the inner policy
/// (`DagScheduler<Logged<Scheduler>>`): the DAG's own operations are method calls, not inputs.
///
/// The [module example](super) logs a run and replays it.
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

/// Seconds between heartbeat samples of one worker in a [`Logged`] log, unless set with
/// [`Logged::sample_every`]. Shorter gives the simulator a finer memory history and a larger log.
pub const DEFAULT_SAMPLE_EVERY: f64 = 60.0;

impl<P: Policy> Logged<P> {
    /// Log `inner`'s events to `sink`, with a heartbeat sample at most every
    /// [`DEFAULT_SAMPLE_EVERY`] seconds per worker (see [`sample_every`](Self::sample_every)).
    pub fn new(inner: P, sink: impl EventSink + 'static) -> Self {
        Self {
            inner,
            sink: Box::new(sink),
            sample_every: DEFAULT_SAMPLE_EVERY,
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
