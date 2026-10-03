//! Offline trace replay and whole-run simulation, to compare `whelm` policies.
#![warn(missing_docs)]

/// The Milnor-subalgebra combinatorics that shape Nassau's job DAG.
pub mod algebra;
/// A synthetic device-memory scenario (small cards, a launch pool, device-aware admission).
pub mod device;
/// The pieces every simulator's event loop shares: the event queue and processor-sharing workers.
pub mod engine;
/// An exact branch-and-bound oracle for tiny instances.
pub mod exact;
/// Offline HEFT planning on small instances.
pub mod heft;
/// The processor-sharing service model and its fit to a trace.
pub mod model;
/// Speed-aware placement and machine models, as the simulators' plans name them.
pub mod plan;
/// The event-driven replay and its metrics.
pub mod run;
/// Small flat instances, their simulation and perturbations (PISA).
pub mod small;
/// The JSONL trace format.
pub mod trace;
/// The whole-run DAG, its cost model and its simulation.
pub mod whole;
