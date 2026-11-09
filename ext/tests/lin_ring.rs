//! [`ExtAlgebra`]'s products from [`ExtAlgebra::compute_ring`] (W. Lin's method, and `h_i` read off
//! the differentials) against a chain map per basis element ([`ExtModule::extend_all_products`]):
//! every product of two basis elements in range agrees.

use std::sync::Arc;

use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::{ExtAlgebra, ExtModule, RingRange},
    utils::construct,
};
use sseq::coordinates::{Bidegree, BidegreeGenerator};

#[test]
fn lin_ring_matches_all_products() {
    let (max_n, max_s) = (36, 14);
    let res = Arc::new(construct("S_2@milnor", None).unwrap());
    res.compute_through_stem(Bidegree::n_s(max_n, max_s));

    let ring = ExtAlgebra::new(Arc::clone(&res));
    ring.compute_ring(RingRange {
        max_n,
        max_s,
        max_t: max_n + max_s,
    })
    .unwrap();
    assert_eq!(ring.with_ring(|r| r.check_h_associativity().1), Some(0));

    // The indecomposables through (n, s) = (36, 14), as listed by G. Wu's chart of Ext.
    let mut gens: Vec<(i32, i32)> = ring
        .indecomposables()
        .unwrap()
        .iter()
        .map(|g| (g.n(), g.s()))
        .collect();
    gens.sort_unstable();
    let known = [
        (0, 1),
        (1, 1),
        (3, 1),
        (7, 1),
        (8, 3),
        (9, 5),
        (11, 5),
        (14, 4),
        (15, 1),
        (16, 7),
        (17, 4),
        (17, 9),
        (18, 4),
        (19, 3),
        (19, 9),
        (20, 4),
        (22, 8),
        (23, 7),
        (24, 11),
        (25, 8),
        (25, 13),
        (26, 7),
        (27, 13),
        (29, 7),
        (30, 6),
        (30, 12),
        (31, 1),
        (31, 5),
        (32, 4),
        (32, 6),
        (32, 7),
        (33, 4),
        (33, 12),
        (34, 11),
        (35, 7),
        (36, 6),
    ];
    assert_eq!(gens, known);

    let e2 = ExtModule::intrinsic(Arc::clone(&res));
    e2.extend_all_products().unwrap();
    let bidegrees: Vec<Bidegree> = res
        .iter_stem()
        .filter(|&b| res.number_of_gens_in_bidegree(b) > 0)
        .collect();
    let mut compared = 0;
    for &b in &bidegrees {
        for &c in &bidegrees {
            let bc = b + c;
            if bc.n() > max_n || bc.s() > max_s {
                continue;
            }
            for x in 0..ring.dimension(b) {
                let g = BidegreeGenerator::new(b, x);
                let ours = ring.known_multiply_into(&ring.generator(g), c).unwrap();
                let expected = e2.multiply_into(&e2.generator(g), c).unwrap();
                for (y, row) in ours.iter().enumerate() {
                    let ours: Vec<u32> = row.iter().collect();
                    let theirs: Vec<u32> = expected.row(y).iter().collect();
                    assert_eq!(ours, theirs, "x_{b}_{x} · x_{c}_{y}");
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 7029, "products compared");
}
