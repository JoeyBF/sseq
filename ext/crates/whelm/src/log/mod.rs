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
//!     let worker = WorkerState {
//!         id: w,
//!         budget: Resources::mem_gb(8.0),
//!         ..Default::default()
//!     };
//!     p.handle(Input::Worker(worker), 0.0);
//! }
//! p.handle(
//!     Input::Submit(JobSpec {
//!         id: 7,
//!         demand: Resources::mem_gb(1.0),
//!         ..Default::default()
//!     }),
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

mod event;
mod logged;
mod playback;
mod sink;
#[cfg(test)]
mod tests;

pub use event::{Event, TaskInfo};
pub use logged::{DEFAULT_SAMPLE_EVERY, Logged};
pub use playback::{polls, replay};
pub use sink::EventSink;
#[cfg(feature = "log")]
pub use sink::JsonlSink;

#[cfg(doc)]
use crate::Policy;
