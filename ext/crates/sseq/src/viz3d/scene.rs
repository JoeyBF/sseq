//! Dimension-3 geometry extraction.
//!
//! This turns an `Sseq<3, P>` into a plain, renderer-agnostic [`SseqScene`]: a per-page list of
//! nodes (classes) and edges (differentials and products), each already resolved to world
//! coordinates. It contains no `three-d` types, so it is generic over the profile `P` and fully
//! unit-testable without a GPU.
//!
//! The traversal mirrors [`crate::Sseq::try_write_to_graph`] (the 2D SVG/TikZ charting entry
//! point), but records geometry into data instead of calling a bidegree-only `Backend`, and does
//! so for every page `E_r` rather than a single one.

use fp::matrix::Subquotient;

use crate::{Product, Sseq, SseqProfile};

/// How far apart co-located classes (several classes in the same tridegree) are spread, in world
/// units. The 3D analogue of the 2D frontend's `getPosition`.
const SPREAD: f32 = 0.2;

/// What a given edge represents, used to colour it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeKind {
    /// A product (structure line) with the given name.
    Structline(String),
    /// A `d_r` differential on the given page.
    Differential(i32),
}

/// A single class, placed at a world position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    pub pos: [f32; 3],
}

/// A line segment between two classes.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub src: [f32; 3],
    pub dst: [f32; 3],
    pub kind: EdgeKind,
}

/// The geometry of a single page `E_r`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Page {
    pub r: i32,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

/// A renderer-agnostic description of an `Sseq<3>` across all its pages.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SseqScene {
    /// Componentwise minimum of every occupied tridegree (for camera framing).
    pub min: [i32; 3],
    /// Componentwise maximum of every occupied tridegree (for camera framing).
    pub max: [i32; 3],
    /// One entry per page, in increasing `r`.
    pub pages: Vec<Page>,
}

/// World position of the `i`-th of `count` classes sharing tridegree `coords`.
///
/// Classes are spread along the first (`n`) axis, exactly as the 2D chart spreads them along `x`.
fn class_pos(coords: [i32; 3], i: usize, count: usize) -> [f32; 3] {
    let offset = (i as f32 - (count as f32 - 1.0) / 2.0) * SPREAD;
    [
        coords[0] as f32 + offset,
        coords[1] as f32,
        coords[2] as f32,
    ]
}

/// Build the geometry of a single page `E_r`.
fn extract_page<P: SseqProfile<3>>(
    sseq: &Sseq<3, P>,
    products: &[(String, Product<3>)],
    r: i32,
) -> Page {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    for b in sseq.iter_degrees() {
        let coords = b.coords();
        let bd = sseq.page_data(b).get_max(r);
        if bd.is_empty() {
            continue;
        }
        let dim = bd.dimension();

        // Nodes: one per surviving class on this page.
        for i in 0..dim {
            nodes.push(Node {
                pos: class_pos(coords, i, dim),
            });
        }

        // Products (structure lines) hitting this tridegree.
        for (name, prod) in products {
            let source_b = b - prod.b;
            if !sseq.defined(source_b) {
                continue;
            }
            let source_data = sseq.page_data(source_b).get_max(r);
            if source_data.is_empty() {
                continue;
            }
            let Some(matrix) = prod.matrices.get(source_b) else {
                continue;
            };
            let source_coords = source_b.coords();
            let source_dim = source_data.dimension();
            let reduced = Subquotient::reduce_matrix(matrix, source_data, bd);
            for (k, row) in reduced.into_iter().enumerate() {
                for (l, v) in row.into_iter().enumerate() {
                    if v != 0 {
                        edges.push(Edge {
                            src: class_pos(source_coords, k, source_dim),
                            dst: class_pos(coords, l, dim),
                            kind: EdgeKind::Structline(name.clone()),
                        });
                    }
                }
            }
        }

        // Differentials out of this tridegree on page `r`.
        let target_b = P::profile(r, b);
        if !sseq.defined(target_b) {
            continue;
        }
        let diffs = sseq.differentials(b);
        if diffs.len() <= r {
            continue;
        }
        let target_coords = target_b.coords();
        let target_data = sseq.page_data(target_b).get_max(r);
        let target_dim = target_data.dimension();
        for (mut s, mut t) in diffs[r].get_source_target_pairs() {
            let source = bd.reduce(s.as_slice_mut());
            let target = target_data.reduce(t.as_slice_mut());
            for (i, &sv) in source.iter().enumerate() {
                if sv == 0 {
                    continue;
                }
                for (j, &tv) in target.iter().enumerate() {
                    if tv == 0 {
                        continue;
                    }
                    edges.push(Edge {
                        src: class_pos(coords, i, dim),
                        dst: class_pos(target_coords, j, target_dim),
                        kind: EdgeKind::Differential(r),
                    });
                }
            }
        }
    }

    Page { r, nodes, edges }
}

/// The largest page worth drawing: one past the last differential, but never below `MIN_R`.
pub(crate) fn max_page<P: SseqProfile<3>>(sseq: &Sseq<3, P>) -> i32 {
    let mut max_r = P::MIN_R;
    for b in sseq.iter_degrees() {
        // `max_degree()` of an empty `BiVec` is `MIN_R - 1`, so this is always safe.
        max_r = max_r.max(sseq.differentials(b).max_degree() + 1);
    }
    max_r
}

/// Extract the full scene of an `Sseq<3, P>`, for pages `MIN_R..=max_r`.
///
/// `products` is the same `(name, Product<3>)` list the 2D charting takes; pass an empty slice to
/// draw differentials only.
pub fn extract_scene<P: SseqProfile<3>>(
    sseq: &Sseq<3, P>,
    products: &[(String, Product<3>)],
    max_r: i32,
) -> SseqScene {
    let min = sseq.min().coords();
    let max = sseq.max().coords();

    let pages = (P::MIN_R..=max_r)
        .map(|r| extract_page(sseq, products, r))
        .collect();

    SseqScene { min, max, pages }
}

#[cfg(test)]
mod tests {
    use fp::{prime::ValidPrime, vector::FpVector};

    use super::*;
    use crate::coordinates::{MultiDegree, MultiDegreeElement};

    /// The same placeholder profile as the demo: `d_r` maps `(n, s, w)` to `(n - 1, s + r, w)`.
    struct TestProfile;

    impl SseqProfile<3> for TestProfile {
        const MIN_R: i32 = 2;

        fn profile(r: i32, b: MultiDegree<3>) -> MultiDegree<3> {
            b + MultiDegree::new([-1, r, 0])
        }

        fn profile_inverse(r: i32, b: MultiDegree<3>) -> MultiDegree<3> {
            b + MultiDegree::new([1, -r, 0])
        }

        fn differential_length(offset: MultiDegree<3>) -> i32 {
            offset.coords()[1]
        }
    }

    #[test]
    fn extract_scene_tracks_pages_nodes_and_differentials() {
        let p = ValidPrime::new(2);
        let mut sseq = Sseq::<3, TestProfile>::new(p);

        sseq.set_dimension(MultiDegree::new([0, 0, 0]), 1);
        sseq.set_dimension(MultiDegree::new([2, 0, 0]), 1);
        sseq.set_dimension(MultiDegree::new([1, 2, 0]), 1); // target of the d2

        // d2: (2,0,0) -> profile(2, (2,0,0)) = (1,2,0).
        sseq.add_differential(
            2,
            &MultiDegreeElement::new(MultiDegree::new([2, 0, 0]), FpVector::from_slice(p, &[1])),
            FpVector::from_slice(p, &[1]).as_slice(),
        );
        sseq.update();

        let scene = extract_scene(&sseq, &[], 3);

        assert_eq!(scene.min, [0, 0, 0]);
        assert_eq!(scene.max, [2, 2, 0]);

        // Pages E_2 and E_3.
        assert_eq!(scene.pages.len(), 2);

        // E_2: all three classes present, one d2 differential drawn.
        let e2 = &scene.pages[0];
        assert_eq!(e2.r, 2);
        assert_eq!(e2.nodes.len(), 3);
        assert_eq!(e2.edges.len(), 1);
        assert_eq!(e2.edges[0].kind, EdgeKind::Differential(2));

        // E_3: source and target of the d2 have died; only the origin class survives.
        let e3 = &scene.pages[1];
        assert_eq!(e3.r, 3);
        assert_eq!(e3.nodes.len(), 1);
        assert_eq!(e3.edges.len(), 0);
    }
}
