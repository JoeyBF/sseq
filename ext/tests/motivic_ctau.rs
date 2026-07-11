//! Phase 1 regression: resolving $A_C/\tau$ over $\mathbb{F}_2$ yields the
//! algebraic Novikov $E_2$ (motivic Adams $E_2$ of $C\tau$).
//!
//! These checks pin the object without a hand-transcribed chart:
//!
//! 1. **0-line = the $h_0$-tower.** $\mathrm{Ext}^{s,s}$ (stem 0) is rank 1 for
//!    every $s$.
//! 2. **1-line = exactly the $h_i$.** $\mathrm{Ext}^{1}$ is rank 1 precisely at
//!    stems $2^i - 1$ and zero elsewhere.
//! 3. **Dominance over classical $\mathrm{Ext}_A$.** Classical Adams $E_2$ is a
//!    subquotient of the algebraic Novikov $E_2$ (the algebraic Novikov filtration
//!    filters classical Ext), so stem-by-stem, $s$-by-$s$, the $A_C/\tau$ ranks are
//!    $\ge$ the classical ranks — and *strictly* greater somewhere (the extra
//!    classes that support algebraic Novikov differentials). We resolve the
//!    classical sphere in-process and compare directly.

use std::{collections::HashMap, sync::Arc};

use algebra::{CTauAlgebra, module::FDModule};
use bivec::BiVec;
use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
    resolution::Resolution,
    utils::construct_standard,
};
use sseq::coordinates::Bidegree;

/// The generator counts of the $A_C/\tau$ resolution of $\mathbb{F}_2$, keyed by
/// `(n, s)`, computed through the stem/filtration box `max`.
fn ctau_ranks(max: Bidegree) -> HashMap<(i32, i32), usize> {
    let algebra = Arc::new(CTauAlgebra::new());
    let trivial = Arc::new(FDModule::new(
        algebra,
        "F2".to_string(),
        BiVec::from_vec(0, vec![1]),
    ));
    let cc: Arc<FiniteChainComplex<FDModule<CTauAlgebra>>> =
        Arc::new(FiniteChainComplex::ccdz(trivial));
    let res = Resolution::new(cc);
    res.compute_through_stem(max);

    res.iter_stem()
        .map(|b| ((b.n(), b.s()), res.number_of_gens_in_bidegree(b)))
        .collect()
}

/// The classical Adams $E_2$ (Ext over the mod-2 Steenrod algebra) generator
/// counts, keyed by `(n, s)`, over the same box.
fn classical_ranks(max: Bidegree) -> HashMap<(i32, i32), usize> {
    let res = construct_standard::<false, _, _>("S_2", None).unwrap();
    res.compute_through_stem(max);
    res.iter_stem()
        .map(|b| ((b.n(), b.s()), res.number_of_gens_in_bidegree(b)))
        .collect()
}

#[test]
fn ctau_zero_and_one_lines() {
    let max = Bidegree::n_s(15, 8);
    let ranks = ctau_ranks(max);

    // 0-line is the h_0-tower: rank 1 for every s.
    for s in 0..=8 {
        assert_eq!(
            ranks.get(&(0, s)).copied().unwrap_or(0),
            1,
            "0-line (h_0-tower) should be rank 1 at s={s}"
        );
    }

    // 1-line is exactly the h_i, at stems 2^i - 1 = 0, 1, 3, 7, 15.
    let h_stems: [i32; 5] = [0, 1, 3, 7, 15];
    for n in 0..=15 {
        let expected = usize::from(h_stems.contains(&n));
        assert_eq!(
            ranks.get(&(n, 1)).copied().unwrap_or(0),
            expected,
            "1-line at stem {n} should be {expected} (h_i live only at 2^i - 1)"
        );
    }
}

#[test]
fn ctau_dominates_classical() {
    let max = Bidegree::n_s(14, 8);
    let ctau = ctau_ranks(max);
    let classical = classical_ranks(max);

    // Every classical class survives into the algebraic Novikov E_2: stem-by-stem,
    // s-by-s, the A_C/τ rank is at least the classical rank.
    let mut strictly_greater_somewhere = false;
    for (&(n, s), &cl) in &classical {
        if n > 14 || s > 8 {
            continue;
        }
        let ct = ctau.get(&(n, s)).copied().unwrap_or(0);
        assert!(
            ct >= cl,
            "A_C/τ rank {ct} < classical rank {cl} at (n={n}, s={s}) — classical class did not survive"
        );
        strictly_greater_somewhere |= ct > cl;
    }
    assert!(
        strictly_greater_somewhere,
        "expected the algebraic Novikov E_2 to be strictly larger than classical somewhere"
    );

    // A concrete witness: classical Ext_A vanishes at (n, s) = (4, 4), but the
    // algebraic Novikov E_2 has a class there.
    assert_eq!(classical.get(&(4, 4)).copied().unwrap_or(0), 0);
    assert!(
        ctau.get(&(4, 4)).copied().unwrap_or(0) >= 1,
        "expected an algebraic Novikov class at (4, 4)"
    );
}
