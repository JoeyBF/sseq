//! The names every event loop uses, for a glob import.
//!
//! `use whelm::prelude::*;` brings in what it takes to drive a [`Scheduler`]: the [`Policy`] trait
//! and its [`Input`] and [`Output`] messages, the [`Time`] each call carries, the [`JobSpec`] and
//! [`WorkerState`] that describe jobs and workers, and the [`Config`] a scheduler is built from.
//! Amounts of resources come with it: [`Resources`], the standard resources [`MEMORY`],
//! [`DEVICE_MEMORY`] and [`SLOTS`], and [`gb`] for writing memory in gigabytes.
//!
//! Nothing else is in the prelude. Everything outside the core loop, such as an explanation's
//! [`Verdict`](crate::explain::Verdict), a [`Defer`](crate::config::Defer) setting or the
//! [`DagScheduler`](crate::dag::DagScheduler), is imported from its module, so an import line
//! says which part of the crate a program uses.
//!
//! ```
//! use whelm::{explain::Status, prelude::*};
//!
//! let mut policy = Scheduler::new(Config::default());
//! let spec = JobSpec {
//!     demand: Resources::new().with(MEMORY, gb(2.0)),
//!     ..Default::default()
//! };
//! policy.handle(Input::Submit { job: 1, spec }, Time::ORIGIN);
//! assert!(policy.poll(Time::ORIGIN).is_empty()); // no worker yet
//! assert!(matches!(
//!     policy.explain(1).unwrap().status,
//!     Status::Waiting(_)
//! ));
//! ```

pub use crate::{
    config::Config,
    job::JobSpec,
    policy::{Input, Output, Policy},
    resources::{DEVICE_MEMORY, MEMORY, Resources, SLOTS, gb},
    scheduler::Scheduler,
    time::Time,
    worker::WorkerState,
};
