//! Comodules over the duals of algebras, presented as modules whose operations lower degree.

mod action_block;
mod check;
mod comodule_trait;
mod filtered;
mod finite_comodule;
mod matrix_comodule;
mod polynomial_comodule;
mod tensor_comodule;

pub use action_block::{ActionBlock, ActionEntry, OpId, OpSet};
pub use check::check_associativity;
pub use comodule_trait::Comodule;
pub use filtered::{FilteredComodule, QuotientComodule, SubComodule};
pub use finite_comodule::FiniteComodule;
pub use matrix_comodule::{ConjugateAction, MatrixQComodule, for_each_column, for_each_matrix};
pub use polynomial_comodule::{Generator, MAX_GENERATORS, Monomial, PolynomialComodule};
pub use tensor_comodule::TensorComodule;
