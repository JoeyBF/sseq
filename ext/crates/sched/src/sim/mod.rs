//! Offline trace replay, to compare policies (feature `sim`).

/// The processor-sharing service model and its fit to a trace.
pub mod model;
/// The event-driven replay and its metrics.
pub mod run;
/// The JSONL trace format.
pub mod trace;
