//! Offline trace replay and whole-run simulation, to compare `sched` policies.
#![warn(missing_docs)]

/// The Milnor-subalgebra combinatorics that shape Nassau's job DAG.
pub mod algebra;
/// A synthetic device-memory scenario (small cards, a launch pool, device-aware admission).
pub mod device;
/// The pieces every simulator's event loop shares: the event queue and processor-sharing workers.
pub mod engine;
/// The processor-sharing service model and its fit to a trace.
pub mod model;
/// The event-driven replay and its metrics.
pub mod run;
/// Small flat instances, their simulation and perturbations (PISA).
pub mod small;
/// The JSONL trace format.
pub mod trace;
/// The whole-run DAG, its cost model and its simulation.
pub mod whole;
