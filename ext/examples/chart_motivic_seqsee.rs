//! Render the C-motivic Adams $E_2$ (= $\Ext_{A_C}(\mathbb{F}_2[\tau], \mathbb{F}_2[\tau])$) as a
//! [SeqSee](https://github.com/JoeyBF/SeqSee) JSON chart, following **Isaksen's notation** (Isaksen,
//! Wang, Xu, *Classical and C-motivic Adams charts*).
//!
//! Every dot is one $\mathbb{F}_2[\tau]$-module summand of the motivic $E_2$ at its $(n, s)$, colored
//! by its τ-torsion order (the structure theorem gives
//! $\mathbb{F}_2[\tau]^{\text{free}} \oplus \bigoplus \mathbb{F}_2[\tau]/\tau^{k}$):
//!
//! 1. **gray** — a free copy of $M_2 = \mathbb{F}_2[\tau]$;
//! 2. **red** — a copy of $M_2/\tau$;
//! 3. **blue** — a copy of $M_2/\tau^2$;
//! 4. **green** — a copy of $M_2/\tau^3$;
//! 5. **purple** — a copy of $M_2/\tau^k$ for some $k \ge 4$.
//!
//! The dots are the τ-free survivors and the τ-Bockstein torsion generators of the deformation SS;
//! their count and torsion orders at each bidegree are cross-checked against
//! [`MotivicResolution::tau_module`] (the pipeline's Smith-normal-form report).
//!
//! Lines are the filtration-one products, taken from [`MotivicResolution::motivic_product`], which
//! carries the hidden-τ-extension power on each term:
//!
//! 6. **$h_0$** (vertical), 7. **$h_1$** (slope 1), 8. **$h_2$** (slope 1/3) — ordinary products
//!    (τ-power 0), colored by their *target's* color, exactly as Isaksen;
//! 9. **red arrows** — $h_1$ multiplications into τ-torsion classes (the infinite $h_1$-towers
//!    annihilated by τ), drawn with an arrow tip;
//! 10. **magenta** — an extension hitting $\tau \cdot (\text{generator})$ (e.g. $h_0 \cdot h_0 h_2 =
//!     \tau h_1^3$ in the 3-stem);
//! 11. **orange** — an extension hitting $\tau^k \cdot (\text{generator})$ for some $k \ge 2$.
//!
//! Adams $d_2$ differentials (Isaksen rules 12–14) are intentionally omitted — those are supplied by
//! hand.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features concurrent --example chart_motivic_seqsee -- 30 17 > motivic.json
//! ```
//!
//! Prompts/args: `Max n` (default 30) and `Max s` (default 17); the reported box is stems
//! `0..=Max n`, filtrations `0..=Max s - 1`. `MOT_SAVE=<dir>` caches the resolution + lift.

use std::collections::HashMap;

use ext::motivic::{Gen, MotivicResolution};
use serde_json::json;
use sseq::{
    charting::{Backend, SeqSeeBackend},
    coordinates::{Bidegree, BidegreeGenerator},
};

/// The τ-torsion order of an $\mathbb{F}_2[\tau]$-module generator: `0` is a free copy of $M_2$,
/// `k ≥ 1` is a copy of $M_2/\tau^k$.
type Order = u32;

/// The node-color alias for a generator of torsion order `k` (Isaksen's palette).
fn color_alias(k: Order) -> &'static str {
    match k {
        0 => "m2",     // gray  — M₂
        1 => "m2tau",  // red   — M₂/τ
        2 => "m2tau2", // blue  — M₂/τ²
        3 => "m2tau3", // green — M₂/τ³
        _ => "m2tauk", // purple — M₂/τ^k, k ≥ 4
    }
}

/// A dot on the chart: an $\mathbb{F}_2[\tau]$-module generator located at `(n, s)`, stacked at
/// `pos` within that bidegree, of torsion `order`.
#[derive(Clone, Copy)]
struct Dot {
    n: i32,
    s: i32,
    pos: usize,
    order: Order,
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let max = Bidegree::n_s(
        query::with_default("Max n", "30", str::parse),
        query::with_default("Max s", "17", str::parse),
    );

    let res = match std::env::var("MOT_SAVE") {
        Ok(dir) if !dir.is_empty() => MotivicResolution::with_module(
            MotivicResolution::trivial_module(),
            max,
            Some(std::path::PathBuf::from(dir)),
        ),
        _ => MotivicResolution::new(max),
    };
    let sseq = res.deformation_sseq();

    // Reported box (mirrors chart_motivic): the top filtration needs the lift one degree higher, so
    // it is not reported.
    let max_n = res.max().n();
    let max_s = res.max().s() - 1;
    let in_box = |n: i32, s: i32| (0..=max_n).contains(&n) && (0..=max_s).contains(&s);

    // --- 1. Weight-grouping: recover, per (s, t), the generator index behind each E₁ slice
    // position. The deformation SS groups the `num_gens(s,t)` generators by weight (ascending), so
    // within a weight-`w` slice the k-th class is the k-th generator (by index) of that weight —
    // exactly how the sseq builds its slices. This lets us tie an E₁ slice position back to a `Gen`,
    // and hence to `motivic_product`. ---------------------------------------------------------------
    let mut slice_gens: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for s in 0..=max_s + 1 {
        for n in 0..=max_n {
            let t = n + s;
            for idx in 0..res.algebraic_novikov_rank(s, t) {
                let w = res.generator_weight(Gen { s, t, idx });
                slice_gens.entry((n, s, w)).or_default().push(idx);
            }
        }
    }

    // --- 2. τ-torsion order of every E₁ class that is an F₂[τ]-module *generator*. Two sources,
    // both robust (no reliance on a single pivot per differential):
    //   * a class that supports a d_r generates a copy of M₂/τ^r → order r (the shortest such r);
    //   * a class that survives to E_∞ is a free copy of M₂ → order 0, read off the E_∞ subquotient's
    //     representative cycles (`Subquotient::gens`), one per survivor.
    // A class that is merely hit by a differential is a τ-multiple, not a generator, and never
    // appears here. `order_of[(n,s,w)][i]` is the order of the ambient class `i` of that slice. -----
    let mut order_of: HashMap<(i32, i32, i32), HashMap<usize, Order>> = HashMap::new();

    // Torsion generators: sources of the τ-Bockstein differentials.
    for b in sseq.iter_degrees() {
        let [n, s, w] = b.coords();
        if !in_box(n, s) {
            continue;
        }
        let diffs = sseq.differentials(b);
        for r in diffs.min_degree()..diffs.len() {
            for (src, _tgt) in diffs[r].get_source_target_pairs() {
                if let Some(i) = src.first_nonzero().map(|(i, _)| i) {
                    order_of
                        .entry((n, s, w))
                        .or_default()
                        .entry(i)
                        .and_modify(|o| *o = (*o).min(r as u32))
                        .or_insert(r as u32);
                }
            }
        }
    }

    // Free generators: the E_∞ survivors.
    for b in sseq.iter_degrees() {
        let [n, s, w] = b.coords();
        if !in_box(n, s) {
            continue;
        }
        // A very large page clamps to E_∞ (the last computed page) for this degree.
        for g in sseq.page_data(b).get_max(i32::MAX).gens() {
            if let Some((i, _)) = g.first_nonzero() {
                order_of
                    .entry((n, s, w))
                    .or_default()
                    .entry(i)
                    .or_insert(0); // torsion (a death) always wins over survival
            }
        }
    }

    // --- 3. Assemble the dots: one per generator, keyed by its `Gen` so products can find their
    // endpoints. Positions stack per (n, s), ordered by (weight, ambient index). -------------------
    let mut dot_of: HashMap<(i32, i32, usize), Dot> = HashMap::new(); // (s, t, idx) -> Dot
    let mut next_pos: HashMap<(i32, i32), usize> = HashMap::new();
    let mut slices: Vec<(i32, i32, i32)> = order_of.keys().copied().collect();
    slices.sort_unstable();
    for key @ (n, s, _w) in slices {
        let gens = &slice_gens[&key];
        let mut entries: Vec<(usize, Order)> =
            order_of[&key].iter().map(|(&i, &o)| (i, o)).collect();
        entries.sort_unstable();
        for (i, order) in entries {
            let pos = next_pos.entry((n, s)).or_insert(0);
            dot_of.insert(
                (s, n + s, gens[i]),
                Dot {
                    n,
                    s,
                    pos: *pos,
                    order,
                },
            );
            *pos += 1;
        }
    }

    // --- 4. Emit the chart. --------------------------------------------------------------------
    let mut backend = SeqSeeBackend::new(std::io::stdout());
    backend.init(Bidegree::n_s(max_n, max_s))?;

    // Node / line colors (Isaksen's palette). Lines reuse the node-color aliases so that an
    // ordinary product line takes its target's color (rules 6–8).
    backend.define_attribute("m2", json!([{ "color": "#8c8c8c" }])); // gray
    backend.define_attribute("m2tau", json!([{ "color": "#e41a1c" }])); // red
    backend.define_attribute("m2tau2", json!([{ "color": "#377eb8" }])); // blue
    backend.define_attribute("m2tau3", json!([{ "color": "#4daf4a" }])); // green
    backend.define_attribute("m2tauk", json!([{ "color": "#984ea3" }])); // purple
    backend.define_attribute("hidden_tau", json!([{ "color": "#f200ff" }])); // magenta (τ¹)
    backend.define_attribute("hidden_tauk", json!([{ "color": "#ff7f00" }])); // orange (τ^{≥2})
    // Red arrow for the τ-annihilated infinite h₁-towers (rule 9).
    backend.define_attribute("h1tower", json!([{ "color": "#e41a1c", "arrowTip": "simple" }]));

    // Draw the dots.
    for dot in dot_of.values() {
        backend.styled_node(
            Bidegree::n_s(dot.n, dot.s),
            dot.pos,
            &[color_alias(dot.order).to_string()],
            None,
        )?;
    }

    // Cross-check the per-bidegree module structure against the pipeline's SNF report; warn (stderr)
    // without aborting.
    let mut free_at: HashMap<(i32, i32), usize> = HashMap::new();
    let mut tors_at: HashMap<(i32, i32), Vec<Order>> = HashMap::new();
    for dot in dot_of.values() {
        if dot.order == 0 {
            *free_at.entry((dot.n, dot.s)).or_insert(0) += 1;
        } else {
            tors_at.entry((dot.n, dot.s)).or_default().push(dot.order);
        }
    }
    for s in 0..=max_s {
        for n in 0..=max_n {
            let tm = res.tau_module(s, n + s);
            let free = free_at.get(&(n, s)).copied().unwrap_or(0);
            let mut got = tors_at.get(&(n, s)).cloned().unwrap_or_default();
            got.sort_unstable();
            let mut want = tm.torsion.clone();
            want.sort_unstable();
            if free != tm.free || got != want {
                eprintln!(
                    "warning: module mismatch at (n={n}, s={s}): chart free={free} tors={got:?} \
                     vs tau_module free={} tors={want:?}",
                    tm.free
                );
            }
        }
    }

    // --- 5. Filtration-one products h₀, h₁, h₂ (Isaksen draws only these). ----------------------
    let hopf: Vec<(usize, Gen)> = (0..3)
        .map(|i| (i, 1i32 << i)) // hᵢ at (n, s) = (2ⁱ − 1, 1), so t = 2ⁱ
        .filter(|&(_, t)| t - 1 <= max_n && res.algebraic_novikov_rank(1, t) > 0)
        .map(|(i, t)| (i, Gen { s: 1, t, idx: 0 }))
        .collect();

    for (&(s, t, idx), src) in &dot_of {
        if !in_box(src.n, src.s) {
            continue;
        }
        let g = Gen { s, t, idx };
        for &(i, hg) in &hopf {
            for (tgt, power) in res.motivic_product(hg, g) {
                let Some(dst) = dot_of.get(&(tgt.s, tgt.t, tgt.idx)) else {
                    // The product's τ = 0 shadow landed on a non-generator (an absorbed torsion
                    // top); it is not a generator-to-generator line.
                    continue;
                };
                if !in_box(dst.n, dst.s) {
                    continue;
                }
                let style = if power >= 2 {
                    "hidden_tauk" // orange (rule 11)
                } else if power == 1 {
                    "hidden_tau" // magenta (rule 10)
                } else if i == 1 && dst.order >= 1 {
                    "h1tower" // red arrow into τ-torsion (rule 9)
                } else {
                    color_alias(dst.order) // ordinary product, target's color (rules 6–8)
                };
                backend.structline(
                    BidegreeGenerator::new(Bidegree::n_s(src.n, src.s), src.pos),
                    BidegreeGenerator::new(Bidegree::n_s(dst.n, dst.s), dst.pos),
                    Some(style),
                )?;
            }
        }
    }

    Ok(())
}
