//! Where a log's events go: [`EventSink`] and its implementations.

#[cfg(feature = "log")]
use std::io::Write as _;

use super::Event;
#[cfg(doc)]
use super::Logged;

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
///     Config, EventSink, Input, JobSpec, Output, Policy, Resources, Scheduler, Time, WorkerState,
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
/// let worker = WorkerState {
///     id: 1,
///     capacity: Resources::mem_gb(8.0).with_slots(2),
///     ..Default::default()
/// };
/// p.handle(Input::Worker(worker), Time::ORIGIN);
/// for id in 1..=3 {
///     p.handle(
///         Input::Submit(JobSpec {
///             id,
///             demand: Resources::mem_gb(1.0),
///             ..Default::default()
///         }),
///         Time::ORIGIN,
///     );
/// }
/// p.poll(Time::ORIGIN);
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
///     Config, Input, JobSpec, Policy, Resources, Scheduler, Time, WorkerState,
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
/// let worker = WorkerState {
///     id: 1,
///     capacity: Resources::mem_gb(8.0).with_slots(1),
///     ..Default::default()
/// };
/// p.handle(Input::Worker(worker), Time::ORIGIN);
/// p.handle(
///     Input::Submit(JobSpec {
///         id: 1,
///         demand: Resources::mem_gb(1.0),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// p.poll(Time::ORIGIN);
///
/// let text = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
/// let lines: Vec<&str> = text.lines().collect();
/// assert!(lines[0].starts_with(r#"{"type":"input","t":{"secs":0,"nanos":0},"input":"#));
/// assert!(lines[1].starts_with(r#"{"type":"sample","t":{"secs":0,"nanos":0},"worker":"1","#));
/// assert_eq!(
///     lines[3],
///     concat!(
///         r#"{"type":"poll","t":{"secs":0,"nanos":0},"#,
///         r#""out":[{"start":{"job":1,"attempt":1,"worker":1}}]}"#
///     )
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
    ///     Config, EventSink, Policy, Scheduler, Time,
    ///     log::{JsonlSink, Logged},
    /// };
    ///
    /// let sink = JsonlSink::create(Path::new("run.jsonl.gz")).unwrap();
    /// let mut p = Logged::new(Scheduler::new(Config::default()), sink);
    /// loop {
    ///     // Handle the events that arrived, then:
    ///     p.poll(Time::ORIGIN);
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
