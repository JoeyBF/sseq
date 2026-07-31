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

/// Phase 3/4 regression: the full motivic chart (all three pages) matches the
/// committed golden fixture, end to end through the deformation pipeline.
#[test]
fn motivic_chart_matches_golden() {
    use ext::motivic::MotivicResolution;

    let golden = include_str!("../examples/benchmarks/motivic-S_2");
    let max = Bidegree::n_s(8, 6);
    let res = MotivicResolution::new(max);

    let mut out = String::from("n,s,alg_nov,classical,tau_torsion\n");
    for s in 0..=(max.s() - 1) {
        for n in 0..=max.n() {
            let t = n + s;
            let alg_nov = res.algebraic_novikov_rank(s, t);
            let module = res.tau_module(s, t);
            let torsion: String = module
                .torsion
                .iter()
                .map(|&k| {
                    if k == 1 {
                        "τ".to_string()
                    } else {
                        format!("τ^{k}")
                    }
                })
                .collect::<Vec<_>>()
                .join("+");
            if alg_nov > 0 || module.free > 0 || !torsion.is_empty() {
                out.push_str(&format!("{n},{s},{alg_nov},{},{torsion}\n", module.free));
            }
        }
    }

    assert_eq!(
        out, golden,
        "motivic chart drifted from the golden fixture (examples/benchmarks/motivic-S_2)"
    );
}

// --- Save store (PR #260 ZarrV3) round-trip and crash-recovery ------------------
//
// The motivic weights + lifted A_C differentials are cached under a `motivic`
// subgroup of the resolution's own save store (see `src/motivic/persist.rs`). The
// cache is a pure memoization of a deterministic function of `(module, box)`, so
// these tests pin two contracts through the *public* API: (1) a reload reproduces
// every observable exactly, and (2) a damaged cache recomputes rather than serving
// garbage. `LIFT_CACHE_LOADS` (public) distinguishes a genuine disk load from a
// byte-identical recompute, which the observables alone cannot.

use std::sync::atomic::Ordering;

use std::collections::{BTreeMap, BTreeSet};

use ext::motivic::{
    Gen, LIFT_CACHE_LOADS, LIFT_CELLS_REUSED, MotivicResolution, PRODUCT_CELLS_REUSED,
};

/// Serializes the two tests below: each asserts an exact delta of the global
/// [`LIFT_CACHE_LOADS`] counter, so their store builds must not overlap. They are
/// the only motivic tests that build with a save store.
static CACHE_LOAD_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Every public observable of the motivic $E_2$ over the reported box: the
/// τ-module — free rank and τ-torsion orders — at each `(s, t)`. These derive
/// entirely from the weights and the lifted differentials (the two cached
/// quantities), so equality here is equality of the whole cached computation.
/// `s` runs to `max.s()` exclusive because `tau_module`/`free_rank` read the lift
/// one homological degree higher.
fn observables(res: &MotivicResolution) -> Vec<(i32, i32, usize, Vec<u32>)> {
    let max = res.max();
    let mut out = Vec::new();
    for s in 0..max.s() {
        for n in 0..=max.n() {
            let t = n + s;
            let tm = res.tau_module(s, t);
            out.push((s, t, res.free_rank(s, t), tm.torsion));
        }
    }
    out
}

#[test]
fn motivic_save_load_round_trips() {
    let _guard = CACHE_LOAD_TEST_LOCK.lock().unwrap();
    // Resolve with a save store, then reload: the reload comes off disk (not a
    // recompute) and reproduces every observable, module structure and products alike.
    let dir = std::env::temp_dir().join(format!("motivic-save-rt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let max = Bidegree::n_s(8, 5);

    let before = LIFT_CACHE_LOADS.load(Ordering::Relaxed);
    let fresh =
        MotivicResolution::with_module(MotivicResolution::trivial_module(), max, Some(dir.clone()));
    assert_eq!(
        LIFT_CACHE_LOADS.load(Ordering::Relaxed),
        before,
        "a cold build has no cache to load"
    );

    let loaded =
        MotivicResolution::with_module(MotivicResolution::trivial_module(), max, Some(dir.clone()));
    assert_eq!(
        LIFT_CACHE_LOADS.load(Ordering::Relaxed),
        before + 1,
        "the second build must load the lift from the store, not recompute it"
    );

    assert_eq!(
        observables(&fresh),
        observables(&loaded),
        "τ-module (free rank + torsion) is identical after save/load"
    );
    // The ring path survives too, including a hidden-τ extension: h₀·h₂ = τ·h₁³.
    let h0 = Gen { s: 1, t: 1, idx: 0 };
    let h2 = Gen { s: 1, t: 4, idx: 0 };
    assert_eq!(
        fresh.motivic_product(h0, h2),
        loaded.motivic_product(h0, h2),
        "h₀·h₂ product survives save/load"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn motivic_grow_the_box_reuses_cache() {
    let _guard = CACHE_LOAD_TEST_LOCK.lock().unwrap();
    // Lazy box growth: a cell converged in a small box is byte-identical in a larger
    // one (it depends only on the resolution up to its own degree, not the box), so
    // growing reuses the small box's cached in-cone lifts and recomputes only the
    // frontier — yet lands on exactly the cold-build result.
    let dir = std::env::temp_dir().join(format!("motivic-grow-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // Big enough that the small box has high-filtration cells (small stem, large
    // internal degree t) near its top — the ones whose lift only *partially*
    // converges and must NOT be cached. A too-small box has none, which is why an
    // earlier version of this test missed a bug that cached those partials.
    let small = Bidegree::n_s(12, 8);
    let big = Bidegree::n_s(18, 11);

    // Populate the store at the small box.
    let _ = MotivicResolution::with_module(MotivicResolution::trivial_module(), small, Some(dir.clone()));

    // Grow to the big box on the same store: it must reuse the small box's cells.
    let reused_before = LIFT_CELLS_REUSED.load(Ordering::Relaxed);
    let grown =
        MotivicResolution::with_module(MotivicResolution::trivial_module(), big, Some(dir.clone()));
    let reused = LIFT_CELLS_REUSED.load(Ordering::Relaxed) - reused_before;
    assert!(
        reused > 0,
        "growing the box must reuse the smaller box's cached lifts, but reused {reused}"
    );

    // A cold build of the big box (fresh store) must be identical — τ-module *and*
    // products (products read specific lifted support entries, so they catch a
    // wrongly-cached partial that the rank-based τ-module can miss).
    let cold_dir = std::env::temp_dir().join(format!("motivic-grow-cold-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cold_dir);
    std::fs::create_dir_all(&cold_dir).unwrap();
    let cold = MotivicResolution::with_module(
        MotivicResolution::trivial_module(),
        big,
        Some(cold_dir.clone()),
    );
    assert_eq!(
        observables(&grown),
        observables(&cold),
        "the grown box's τ-module matches a from-scratch build"
    );
    // Report-box products only: a source generator b in the +1 stem margin has a
    // box-dependent (partial) product, intentionally not comparable across boxes.
    #[allow(clippy::type_complexity)]
    let report_products =
        |res: &MotivicResolution| -> BTreeSet<(i32, i32, i32, usize, i32, i32, usize, u32)> {
            let mut out = BTreeSet::new();
            for t in [1, 2, 4] {
                if t - 1 > big.n() {
                    continue;
                }
                for (b, terms) in res.motivic_products_by(Gen { s: 1, t, idx: 0 }) {
                    if b.t - b.s > big.n() {
                        continue;
                    }
                    for (g, p) in terms {
                        out.insert((t, b.s, b.t, b.idx, g.s, g.t, g.idx, p));
                    }
                }
            }
            out
        };
    assert_eq!(
        report_products(&grown),
        report_products(&cold),
        "the grown box's hᵢ products match a from-scratch build"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&cold_dir);
}

/// Products compared as sets — `motivic_products_by` returns `Vec` values whose
/// order follows nondeterministic `HashMap` iteration, so normalize before `assert_eq`.
fn norm_products(m: &HashMap<Gen, Vec<(Gen, u32)>>) -> BTreeMap<Gen, BTreeSet<(Gen, u32)>> {
    m.iter().map(|(b, v)| (*b, v.iter().copied().collect())).collect()
}

#[test]
fn motivic_product_cache_reuses_and_matches() {
    let _guard = CACHE_LOAD_TEST_LOCK.lock().unwrap();
    // The product lift φ_a is the biggest phase of a large chart. Cache it: the
    // first computation writes it, a second reads it back and skips the whole
    // τ-correction — landing on the same products as a from-scratch (store-less) run.
    let dir = std::env::temp_dir().join(format!("motivic-prod-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let max = Bidegree::n_s(8, 5);
    let res =
        MotivicResolution::with_module(MotivicResolution::trivial_module(), max, Some(dir.clone()));
    let h1 = Gen { s: 1, t: 2, idx: 0 };

    // First call computes φ_{h1} and writes it (no reuse yet).
    let mid_before = PRODUCT_CELLS_REUSED.load(Ordering::Relaxed);
    let first = res.motivic_products_by(h1);
    let mid = PRODUCT_CELLS_REUSED.load(Ordering::Relaxed);
    assert_eq!(mid, mid_before, "the first product computation has nothing to reuse");

    // Second call must reuse the cached φ cells rather than recompute.
    let second = res.motivic_products_by(h1);
    let after = PRODUCT_CELLS_REUSED.load(Ordering::Relaxed);
    assert!(
        after > mid,
        "the second product computation must reuse cached φ cells, reused {}",
        after - mid
    );
    assert_eq!(
        norm_products(&first),
        norm_products(&second),
        "product results agree across a fresh compute and a cache reuse"
    );

    // And both agree with a from-scratch, store-less computation (the ground truth).
    let plain = MotivicResolution::new(max);
    assert_eq!(
        norm_products(&plain.motivic_products_by(h1)),
        norm_products(&first),
        "cached products match a store-less computation"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn motivic_massey_cache_matches() {
    let _guard = CACHE_LOAD_TEST_LOCK.lock().unwrap();
    // The null-homotopy lift H is cached like the products (a different seed
    // boundary, same box-independence rule). A store-backed Massey bracket must
    // reuse H on a repeat and agree with a store-less computation.
    let dir = std::env::temp_dir().join(format!("motivic-massey-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let max = Bidegree::n_s(10, 7);
    let res =
        MotivicResolution::with_module(MotivicResolution::trivial_module(), max, Some(dir.clone()));
    let h0 = Gen { s: 1, t: 1, idx: 0 };
    let h1 = Gen { s: 1, t: 2, idx: 0 };

    let as_set = |v: Vec<(Gen, u32)>| -> BTreeSet<(Gen, u32)> { v.into_iter().collect() };

    // ⟨h₀, h₁, h₀⟩ = τ·h₁²: compute twice; the repeat must reuse cached H cells.
    let before = PRODUCT_CELLS_REUSED.load(Ordering::Relaxed);
    let first = res.motivic_massey(h0, h1, h0);
    let second = res.motivic_massey(h0, h1, h0);
    assert!(
        PRODUCT_CELLS_REUSED.load(Ordering::Relaxed) > before,
        "a repeated Massey bracket must reuse cached H (and φ) cells"
    );
    assert_eq!(as_set(first.clone()), as_set(second), "cached H gives the same bracket");

    // Ground truth: a store-less computation of the same bracket.
    let plain = MotivicResolution::new(max);
    assert_eq!(
        as_set(first),
        as_set(plain.motivic_massey(h0, h1, h0)),
        "cached Massey bracket matches a store-less computation"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn motivic_partial_cache_recomputes_not_garbage() {
    let _guard = CACHE_LOAD_TEST_LOCK.lock().unwrap();
    // A partial/damaged cache must read as a miss and recompute from scratch, never
    // load a half-written lift. Simulate a crash that left the motivic subgroup
    // incomplete by deleting it after a full build (the resolution's own cache is
    // left intact — exactly a crash between the resolution save and the lift save).
    let dir = std::env::temp_dir().join(format!("motivic-partial-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let max = Bidegree::n_s(8, 5);

    let fresh =
        MotivicResolution::with_module(MotivicResolution::trivial_module(), max, Some(dir.clone()));
    std::fs::remove_dir_all(dir.join("motivic")).unwrap();

    let before = LIFT_CACHE_LOADS.load(Ordering::Relaxed);
    let rebuilt =
        MotivicResolution::with_module(MotivicResolution::trivial_module(), max, Some(dir.clone()));
    assert_eq!(
        LIFT_CACHE_LOADS.load(Ordering::Relaxed),
        before,
        "a damaged cache must recompute, not report a load"
    );
    // A miss costs time, never correctness: the recompute matches the original exactly.
    assert_eq!(
        observables(&fresh),
        observables(&rebuilt),
        "recompute after a damaged cache is identical to the original"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
