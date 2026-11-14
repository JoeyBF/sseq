//! Comodules over the duals of algebras, presented as modules whose operations lower degree.

mod action_block;
mod check;
mod comodule_trait;
mod finite_comodule;
mod polynomial_comodule;

pub use action_block::{ActionBlock, ActionEntry, OpId, OpSet};
pub use check::check_associativity;
pub use comodule_trait::Comodule;
pub use finite_comodule::FiniteComodule;
pub use polynomial_comodule::{Generator, MAX_GENERATORS, Monomial, PolynomialComodule};
