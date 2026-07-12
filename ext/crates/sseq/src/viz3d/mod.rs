//! An interactive 3D visualizer for `Sseq<3>`, gated behind the `3d` feature.
//!
//! The existing charting (`crate::charting`) can only draw a bidegree-graded `Sseq<2>`. This
//! module renders a *three*-graded spectral sequence: classes are plotted at their tridegree
//! `(n, s, w)`, with differentials and products drawn as line segments and the pages `E_r`
//! browsable with the arrow keys.
//!
//! It is split into two layers:
//! - [`scene`] extracts a renderer-agnostic [`scene::SseqScene`] from any `Sseq<3, P>` (generic
//!   over the profile `P`, GPU-free, and unit-testable);
//! - [`render`] draws that scene with the `three-d` engine.
//!
//! ```no_run
//! # use sseq::{Sseq, SseqProfile};
//! # fn demo<P: SseqProfile<3>>(sseq: &Sseq<3, P>) {
//! // Open an interactive window (blocks until closed):
//! sseq::viz3d::show(sseq, &[]);
//! # }
//! ```

pub mod render;
pub mod scene;

pub use render::{render_to_png, render_to_png_from, run, show};
pub use scene::{Edge, EdgeKind, Node, Page, SseqScene, extract_scene};
