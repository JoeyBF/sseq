use std::collections::BTreeMap;

use ext::{
    chain_complex::{AugmentedChainComplex, ChainComplex, FreeChainComplex},
    q_weight::{QWeights, module_q_weights},
    utils::construct_standard,
};
use rstest::rstest;
use serde_json::json;
use sseq::coordinates::Bidegree;

/// Summing the weighted dimensions over the weight gives the ordinary dimensions, and the weights
/// do not depend on the basis of the Steenrod algebra.
#[rstest]
#[trace]
#[case("S_3", 40)]
#[case("C3", 30)]
#[case("C9", 30)]
#[case("S_5", 60)]
fn stable(#[case] module_name: &str, #[case] max_n: i32) {
    let max = Bidegree::n_s(max_n, 10);
    let milnor = construct_standard::<false, _, _>((module_name, "milnor"), None).unwrap();
    let adem = construct_standard::<false, _, _>((module_name, "adem"), None).unwrap();
    milnor.compute_through_stem(max);
    adem.compute_through_stem(max);

    let q_milnor = QWeights::new(&milnor).unwrap();
    let q_adem = QWeights::new(&adem).unwrap();

    for b in milnor.iter_stem() {
        let dims = q_milnor.weighted_dimensions(b);
        assert_eq!(
            dims.values().sum::<usize>(),
            milnor.number_of_gens_in_bidegree(b),
            "at {b}"
        );
        assert_eq!(dims, q_adem.weighted_dimensions(b), "at {b}");
    }
}

#[rstest]
#[trace]
#[case("S_3[3]", 40)]
#[case("S_3[5]", 40)]
#[case("S_5[3]", 60)]
fn unstable(#[case] module_name: &str, #[case] max_n: i32) {
    let max = Bidegree::n_s(max_n, 10);
    let res = construct_standard::<true, _, _>((module_name, "milnor"), None).unwrap();
    res.compute_through_stem(max);
    let q = QWeights::new(&res).unwrap();

    for b in res.iter_stem() {
        assert_eq!(
            q.weighted_dimensions(b).values().sum::<usize>(),
            res.number_of_gens_in_bidegree(b),
            "at {b}"
        );
    }
}

/// $\mathrm{Ext}^1$ of the sphere is dual to the indecomposables of the Steenrod algebra: $a_0 =
/// [\tau_0]$ is dual to $\beta = Q_0$ and has weight 1, while $h_i = [\xi_1^{p^i}]$ is dual to
/// $P^{p^i}$ and has weight 0. The $a_0$-tower in stem 0 has weight $s$.
#[test]
fn sphere_known_values() {
    let res = construct_standard::<false, _, _>("S_3", None).unwrap();
    res.compute_through_stem(Bidegree::n_s(40, 10));
    let q = QWeights::new(&res).unwrap();

    // a_0^s
    for s in 0..=10 {
        assert_eq!(q.generator_weights(Bidegree::n_s(0, s)), &[s as u32]);
    }
    // h_0 and h_1
    assert_eq!(q.generator_weights(Bidegree::s_t(1, 4)), &[0]);
    assert_eq!(q.generator_weights(Bidegree::s_t(1, 12)), &[0]);
    // Nothing else in filtration 1
    let total: BTreeMap<u32, usize> = res
        .iter_stem()
        .filter(|b| b.s() == 1)
        .flat_map(|b| q.weighted_dimensions(b))
        .fold(BTreeMap::new(), |mut acc, (e, n)| {
            *acc.entry(e).or_default() += n;
            acc
        });
    assert_eq!(total, BTreeMap::from([(0, 3), (1, 1)]));
}

/// In the Moore spectrum the top cell is hit by the Bockstein, so it has weight 1.
#[test]
fn moore_module_weights() {
    let res = construct_standard::<false, _, _>(
        (
            json!({
                "p": 3,
                "type": "finite dimensional module",
                "gens": {"x0": 0, "x1": 1},
                "actions": ["b x0 = x1"]
            }),
            "milnor",
        ),
        None,
    )
    .unwrap();
    let module = res.target().module(0);
    assert_eq!(
        module_q_weights(&*module, 10).unwrap(),
        vec![vec![0], vec![1]]
    );

    res.compute_through_stem(Bidegree::n_s(20, 5));
    let q = QWeights::new(&res).unwrap();
    // h_0 survives; a_0 does not.
    assert_eq!(q.generator_weights(Bidegree::s_t(0, 0)), &[0]);
    assert_eq!(q.generator_weights(Bidegree::s_t(1, 4)), &[0]);
}

/// A module whose basis is not homogeneous when generators have weight 0.
#[test]
fn inhomogeneous_module() {
    let res = construct_standard::<false, _, _>(
        (
            json!({
                "p": 3,
                "type": "finite dimensional module",
                "gens": {"x0": 0, "u3": 3, "y4": 4},
                "actions": ["P1 x0 = y4", "b u3 = y4"]
            }),
            "milnor",
        ),
        None,
    )
    .unwrap();
    res.compute_through_stem(Bidegree::n_s(10, 3));
    let err = QWeights::new(&res).err().unwrap();
    assert!(
        format!("{err:#}").contains("not Q-weight homogeneous"),
        "{err:#}"
    );
}

#[test]
fn even_prime() {
    let res = construct_standard::<false, _, _>("S_2", None).unwrap();
    res.compute_through_stem(Bidegree::n_s(10, 3));
    let err = QWeights::new(&res).err().unwrap();
    assert!(format!("{err:#}").contains("odd primes"), "{err:#}");
}
