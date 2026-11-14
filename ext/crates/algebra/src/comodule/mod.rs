//! Comodules over the duals of algebras, presented as modules whose operations lower degree.

mod action_block;
mod comodule_trait;
mod finite_comodule;

pub use action_block::{ActionBlock, ActionEntry, OpId, OpSet};
pub use comodule_trait::Comodule;
pub use finite_comodule::FiniteComodule;
