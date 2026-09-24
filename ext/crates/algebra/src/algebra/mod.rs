//! Traits describing algebras, and implementations thereof for different
//! representations of the Steenrod algebra.

pub mod adem_algebra;
pub use adem_algebra::AdemAlgebra;

mod algebra_trait;
pub use algebra_trait::{Algebra, GeneratedAlgebra, MuAlgebra, UnstableAlgebra};

mod bialgebra_trait;
pub use bialgebra_trait::Bialgebra;

pub mod combinatorics;

pub mod field;
pub use field::Field;

pub mod milnor_algebra;
pub use milnor_algebra::MilnorAlgebra;

// Framework-independent core of the batched Milnor multiply: the product descriptor, the
// output layout and the CPU reference. UNGATED on purpose -- it must not live behind any
// one backend's feature, or a second backend cannot use it without dragging the first along.
pub mod milnor_batch;
#[cfg(feature = "cuda")]
pub mod milnor_cuda;
#[cfg(feature = "gpu")]
pub mod milnor_gpu;
// Opt-in: an arithmetic alternative to the Milnor basis index map. Not wired in; see the module
// docs for what it costs and what it would take to adopt.
#[cfg(feature = "milnor-rank")]
pub mod milnor_rank;

mod steenrod_algebra;
pub use steenrod_algebra::{AlgebraType, SteenrodAlgebra};

pub mod pair_algebra;
