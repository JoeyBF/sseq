//! Offline trace replay and whole-run simulation, to compare policies (feature `sim`).

/// The Milnor-subalgebra combinatorics that shape Nassau's job DAG.
pub mod algebra;
/// The processor-sharing service model and its fit to a trace.
pub mod model;
/// The event-driven replay and its metrics.
pub mod run;
/// The JSONL trace format.
pub mod trace;
/// The whole-run DAG, its cost model and its simulation.
pub mod whole;
