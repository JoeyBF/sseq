//! The recording wrapper, [`Logged`].

use std::{collections::HashMap, time::Duration};

#[cfg(doc)]
use super::replay;
use super::{Event, EventSink, TaskInfo};
use crate::{
    DEVICE_MEMORY, Explanation, Input, JobId, MEMORY, Output, Policy, PolicyStats, Time, WorkerId,
    WorkerState,
};

/// A [`Policy`] that records every input it handles and every poll's outputs to an
/// [`EventSink`], so that [`replay`] can reproduce the run.
///
/// Heartbeats are also summarised as samples at most every `sample_every` per worker.
/// Under a [`DagScheduler`](crate::DagScheduler), wrap the inner policy
/// (`DagScheduler<Logged<Scheduler>>`): the DAG's own operations are method calls, not inputs.
///
/// The [module example](super) logs a run and replays it.
pub struct Logged<P> {
    inner: P,
    sink: Box<dyn EventSink>,
    sample_every: Duration,
    last_sample: HashMap<WorkerId, Time>,
    info: HashMap<JobId, TaskInfo>,
}

/// Bytes to GB.
fn in_gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// The time between heartbeat samples of one worker in a [`Logged`] log, unless set with
/// [`Logged::sample_every`]. Shorter gives the simulator a finer memory history and a larger log.
pub const DEFAULT_SAMPLE_EVERY: Duration = Duration::from_secs(60);

impl<P: Policy> Logged<P> {
    /// Log `inner`'s events to `sink`, with a heartbeat sample at most every
    /// [`DEFAULT_SAMPLE_EVERY`] per worker (see [`sample_every`](Self::sample_every)).
    pub fn new(inner: P, sink: impl EventSink + 'static) -> Self {
        Self {
            inner,
            sink: Box::new(sink),
            sample_every: DEFAULT_SAMPLE_EVERY,
            last_sample: HashMap::new(),
            info: HashMap::new(),
        }
    }

    /// Sample heartbeats at most every `every` per worker (zero: every heartbeat).
    ///
    /// A sample shows what the worker reported next to what the policy placed there. A worker's
    /// first heartbeat is always sampled; later ones only once `every` has passed:
    ///
    /// ```
    /// use std::{
    ///     sync::{Arc, Mutex},
    ///     time::Duration,
    /// };
    ///
    /// use whelm::{
    ///     Config, Input, JobSpec, MEMORY, Policy, Resources, SLOTS, Scheduler, Time, WorkerState, gb,
    ///     log::{Event, Logged},
    /// };
    ///
    /// let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    /// let inner = Scheduler::new(Config::default());
    /// let mut p = Logged::new(inner, events.clone()).sample_every(Duration::from_secs(30));
    /// let mut w = WorkerState {
    ///     id: 1,
    ///     capacity: Resources::new().with(MEMORY, gb(8.0)).with(SLOTS, 2),
    ///     ..Default::default()
    /// };
    /// p.handle(Input::Worker(w.clone()), Time::ORIGIN);
    /// let spec = JobSpec {
    ///     demand: Resources::new().with(MEMORY, gb(2.0)),
    ///     ..Default::default()
    /// };
    /// p.handle(Input::Submit { job: 1, spec }, Time::ORIGIN);
    /// p.poll(Time::ORIGIN);
    /// w.reported_used = Resources::new().with(MEMORY, gb(1.5));
    /// for t in [10, 20, 30] {
    ///     p.handle(Input::Worker(w.clone()), Time(Duration::from_secs(t)));
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
    ///         t: Time(Duration::from_secs(30)),
    ///         worker: "1".into(),
    ///         rss_gb: 1.5,
    ///         baseline_gb: 0.0,
    ///         reserved_gb: 2.0,
    ///         running: 1,
    ///         dev_per_task_gb: 0.0,
    ///     },
    /// );
    /// ```
    pub fn sample_every(mut self, every: Duration) -> Self {
        self.sample_every = every;
        self
    }

    /// Attach what a job is, to be logged with its submission (call before submitting it).
    ///
    /// The annotation is used once, by the job's next submission:
    ///
    /// ```
    /// use std::{
    ///     sync::{Arc, Mutex},
    ///     time::Duration,
    /// };
    ///
    /// use whelm::{
    ///     Config, Input, JobSpec, MEMORY, Policy, Resources, Scheduler, Time, gb,
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
    ///     Input::Submit {
    ///         job: 1,
    ///         spec: JobSpec {
    ///             demand: Resources::new().with(MEMORY, gb(1.0)),
    ///             ..Default::default()
    ///         },
    ///     },
    ///     Time::ORIGIN,
    /// );
    /// p.handle(Input::Cancel(1), Time(Duration::from_secs(1)));
    /// p.handle(
    ///     Input::Submit {
    ///         job: 1,
    ///         spec: JobSpec {
    ///             demand: Resources::new().with(MEMORY, gb(1.0)),
    ///             ..Default::default()
    ///         },
    ///     },
    ///     Time(Duration::from_secs(2)),
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
    fn log_sample(&mut self, w: &WorkerState, now: Time) {
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
            t: now,
            worker: w.id.to_string(),
            rss_gb: in_gb(w.reported_used.get(MEMORY)),
            baseline_gb: in_gb(w.reported_baseline.get(MEMORY)),
            reserved_gb: load.as_ref().map_or(0.0, |l| in_gb(l.placed.get(MEMORY))),
            running: load.map_or(0, |l| l.running),
            dev_per_task_gb: in_gb(w.per_task.get(DEVICE_MEMORY)),
        });
        self.last_sample.insert(w.id, now);
    }
}

impl<P: Policy> Policy for Logged<P> {
    /// Logged (with the job's annotation, for a submission), then forwarded; a heartbeat is
    /// sampled after it is forwarded, so that the sample shows the policy's bookkeeping.
    fn handle(&mut self, input: Input, now: Time) {
        let info = match &input {
            Input::Submit { job, .. } => self.info.remove(job).map(Box::new),
            _ => None,
        };
        self.sink.record(&Event::Input {
            t: now,
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
    fn poll(&mut self, now: Time) -> Vec<Output> {
        let out = self.inner.poll(now);
        self.sink.record(&Event::Poll {
            t: now,
            out: out.clone(),
        });
        out
    }

    /// Forwarded.
    fn next_wakeup(&self) -> Option<Time> {
        self.inner.next_wakeup()
    }

    /// Forwarded.
    fn explain(&self, job: JobId) -> Option<Explanation> {
        self.inner.explain(job)
    }

    /// Forwarded.
    fn stats(&self) -> PolicyStats {
        self.inner.stats()
    }
}
