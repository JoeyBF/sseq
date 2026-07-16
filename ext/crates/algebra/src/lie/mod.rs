//! Finite-dimensional (restricted) Lie algebras and their Chevalley–Eilenberg cohomology.
//!
//! This module hosts the *large-prime* chromatic tool described in
//! `ext/docs/chromatic-computations.md` §3: the height-`n` Morava Lie algebra `L(n, n)` and a direct
//! Chevalley–Eilenberg cohomology computation of `H^*(L(n, n); F_p)` for `p > n + 1`. Unlike the
//! resolution engine (which computes the *infinite* comodule cohomology relevant at small primes),
//! this is a *finite* Koszul-complex computation — pure `F_p` linear algebra — appropriate to the
//! regime where Ravenel's May spectral sequence collapses immediately (Salch, arXiv:2312.17185) and
//! `H^*(S(n)) = H^*(L(n, n))`.

pub mod cohomology;
pub mod morava_lie;

pub use cohomology::{CohomologyStats, chevalley_eilenberg_cohomology};
pub use morava_lie::MoravaLie;
