//! Interactive 3D visualizer demo for a three-graded spectral sequence.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p sseq --features 3d --example viz3d
//! ```
//!
//! This builds a small synthetic `Sseq<3>` (there is no `Sseq<3>` producer in the resolver yet),
//! with a placeholder profile, and opens the `three-d` window. Orbit with the mouse, zoom with the
//! scroll wheel, and switch pages `E_r` with the left/right arrow keys.
//!
//! Set `SSEQ_VIZ3D_SCREENSHOT=<path.png>` to instead render page E_2 offscreen to a PNG and exit
//! (no interactive window). On a headless machine, run it under `xvfb-run`.

use fp::{matrix::Matrix, prime::ValidPrime, vector::FpVector};
use once::MultiIndexed;
use sseq::{
    Product, Sseq, SseqProfile,
    coordinates::{MultiDegree, MultiDegreeElement},
};

type Deg = MultiDegree<3>;

/// A placeholder three-graded profile. It generalizes the Adams profile by keeping the third
/// coordinate (a "weight") fixed: `d_r` sends `(n, s, w)` to `(n - 1, s + r, w)`.
///
/// The visualizer is generic over `SseqProfile<3>`; this exists only to generate demo data and
/// imposes no particular mathematics.
struct DemoProfile;

impl SseqProfile<3> for DemoProfile {
    const MIN_R: i32 = 2;

    fn profile(r: i32, b: Deg) -> Deg {
        b + MultiDegree::new([-1, r, 0])
    }

    fn profile_inverse(r: i32, b: Deg) -> Deg {
        b + MultiDegree::new([1, -r, 0])
    }

    fn differential_length(offset: Deg) -> i32 {
        offset.coords()[1]
    }
}

fn main() {
    let p = ValidPrime::new(2);
    let mut sseq = Sseq::<3, DemoProfile>::new(p);

    // A small cloud of tridegrees. Two classes at (0,1,0) exercise the co-located-class spread.
    for (coords, dim) in [
        ([0, 0, 0], 1),
        ([1, 0, 0], 1),
        ([0, 0, 1], 1),
        ([1, 0, 1], 1),
        ([2, 0, 0], 1),
        ([1, 2, 0], 1), // target of the d2 below
        ([3, 0, 1], 1),
        ([2, 2, 1], 1), // intermediate-page (d2) target of the d3 source below
        ([2, 3, 1], 1), // target of the d3 below
        ([0, 1, 0], 2),
        ([1, 1, 0], 1),
    ] {
        sseq.set_dimension(MultiDegree::new(coords), dim);
    }

    // A d2 and a d3, in different weights, so the pages differ as you scrub.
    sseq.add_differential(
        2,
        &MultiDegreeElement::new(MultiDegree::new([2, 0, 0]), FpVector::from_slice(p, &[1])),
        FpVector::from_slice(p, &[1]).as_slice(),
    );
    sseq.add_differential(
        3,
        &MultiDegreeElement::new(MultiDegree::new([3, 0, 1]), FpVector::from_slice(p, &[1])),
        FpVector::from_slice(p, &[1]).as_slice(),
    );

    // A permanent class at the origin.
    sseq.add_permanent_class(&MultiDegreeElement::new(
        MultiDegree::new([0, 0, 0]),
        FpVector::from_slice(p, &[1]),
    ));

    sseq.update();

    // A product `v` that raises the weight by one: draws structure lines along the w axis.
    let matrices = MultiIndexed::<3, Matrix>::new();
    matrices.insert([0, 0, 0], Matrix::from_vec(p, &[vec![1]]));
    matrices.insert([1, 0, 0], Matrix::from_vec(p, &[vec![1]]));
    let product = Product {
        b: MultiDegree::new([0, 0, 1]),
        left: true,
        matrices,
    };
    let products = vec![("v".to_string(), product)];

    // Offscreen capture mode when a screenshot path is provided; otherwise open the window.
    if let Ok(path) = std::env::var("SSEQ_VIZ3D_SCREENSHOT") {
        sseq::viz3d::render_to_png_from(&sseq, &products, 0, 1280, 800, &path);
    } else {
        sseq::viz3d::show(&sseq, &products);
    }
}
