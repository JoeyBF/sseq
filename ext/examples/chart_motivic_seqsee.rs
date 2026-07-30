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

    // --- 3. Assemble the dots. The SS classification above gives *candidate* generators with a
    // `Gen` identity (needed to wire products), but the authoritative per-bidegree module structure
    // is `tau_module` (the SNF of the outgoing δ). So we draw exactly what `tau_module` reports and
    // attach a candidate `Gen` to each dot by matching torsion order, dropping any surplus candidate
    // and leaving a dot `Gen`-less if the classification came up short (rare, top-of-range only). ---

    // Candidate generators per (n, s), each `(order, (s, t, idx))`.
    let mut cand: HashMap<(i32, i32), Vec<(Order, (i32, i32, usize))>> = HashMap::new();
    let mut slices: Vec<(i32, i32, i32)> = order_of.keys().copied().collect();
    slices.sort_unstable();
    for key @ (n, s, _w) in slices {
        let gens = &slice_gens[&key];
        let mut entries: Vec<(usize, Order)> =
            order_of[&key].iter().map(|(&i, &o)| (i, o)).collect();
        entries.sort_unstable();
        for (i, order) in entries {
            cand.entry((n, s))
                .or_default()
                .push((order, (s, n + s, gens[i])));
        }
    }

    // Every dot to draw, and — when a candidate matched — the `Gen` behind it (for products).
    let mut dots: Vec<Dot> = Vec::new();
    let mut dot_of: HashMap<(i32, i32, usize), Dot> = HashMap::new(); // (s, t, idx) -> Dot
    for s in 0..=max_s {
        for n in 0..=max_n {
            let tm = res.tau_module(s, n + s);
            // The multiset of orders `tau_module` says must be present: `free` copies of order 0
            // plus one per torsion order.
            let mut wanted: Vec<Order> = vec![0; tm.free];
            wanted.extend(tm.torsion.iter().copied());
            wanted.sort_unstable();

            let mut avail = cand.remove(&(n, s)).unwrap_or_default();
            let mut used = vec![false; avail.len()];
            for (pos, &order) in wanted.iter().enumerate() {
                // Attach a not-yet-used candidate of the same order, if any.
                let matched = avail
                    .iter()
                    .enumerate()
                    .position(|(k, &(o, _))| !used[k] && o == order);
                let dot = Dot { n, s, pos, order };
                if let Some(k) = matched {
                    used[k] = true;
                    dot_of.insert(avail[k].1, dot);
                }
                dots.push(dot);
            }
            // Surplus candidates (the classification's over-counts) are simply not drawn.
            let _ = &mut avail;
        }
    }

    // A position key `(n, s, pos)` identifies a drawn dot; look up its order.
    let order_at: HashMap<(i32, i32, usize), Order> = dots
        .iter()
        .map(|d| ((d.n, d.s, d.pos), d.order))
        .collect();

    // --- 4. Filtration-one products h₀, h₁, h₂ (Isaksen draws only these), built in memory so the
    // h₁-towers can be post-processed before emission. Each edge records which hᵢ and the hidden-
    // extension τ-power on that term. ------------------------------------------------------------
    struct Edge {
        src: (i32, i32, usize),
        tgt: (i32, i32, usize),
        hi: usize,
        power: u32,
    }
    let hopf: Vec<(usize, Gen)> = (0..3)
        .map(|i| (i, 1i32 << i)) // hᵢ at (n, s) = (2ⁱ − 1, 1), so t = 2ⁱ
        .filter(|&(_, t)| t - 1 <= max_n && res.algebraic_novikov_rank(1, t) > 0)
        .map(|(i, t)| (i, Gen { s: 1, t, idx: 0 }))
        .collect();

    let mut edges: Vec<Edge> = Vec::new();
    for &(i, hg) in &hopf {
        // Lift φ_{hᵢ} once and read off hᵢ·b for every generator b, instead of re-lifting per pair.
        let products = res.motivic_products_by(hg);
        for (&(s, t, idx), src) in &dot_of {
            if !in_box(src.n, src.s) {
                continue;
            }
            let Some(terms) = products.get(&Gen { s, t, idx }) else {
                continue;
            };
            for &(tgt, power) in terms {
                let Some(dst) = dot_of.get(&(tgt.s, tgt.t, tgt.idx)) else {
                    continue; // τ = 0 shadow landed on a non-generator; not a generator line
                };
                if !in_box(dst.n, dst.s) {
                    continue;
                }
                // Drop spurious products: if the target is τ-torsion of order `ord`
                // (a copy of M₂/τ^ord), then a term with τ-power ≥ ord is
                // τ^power·(order-ord class) = 0 and no line should be drawn. Free
                // targets (order 0) are never annihilated. Filtering here — before the
                // edge exists — also keeps these zeros out of the h₁-tower analysis.
                if dst.order != 0 && power >= dst.order {
                    continue;
                }
                edges.push(Edge {
                    src: (src.n, src.s, src.pos),
                    tgt: (dst.n, dst.s, dst.pos),
                    hi: i,
                    power,
                });
            }
        }
    }

    // --- 5. Collapse the infinite h₁-towers into red arrows (Isaksen rule 9). --------------------
    //
    // Above the line s = n/2 + 3/2 every class is h₁-periodic (in an infinite, τ-annihilated
    // h₁-tower). Starting from those, we trim along h₁-multiplications through τ-torsion classes,
    // stopping at a class that either is a target of an h₀ or h₂ product or is not simple
    // (M₂/τ) torsion. Each such tower is removed and replaced by a single red arrow rising from its
    // anchor — the non-tower class just below the tower's foot.
    let above_line = |n: i32, s: i32| (s as f64) > (n as f64) / 2.0 + 1.5;

    // Classes that receive an h₀ or h₂ product (an "incoming h₀/h₂").
    let mut incoming_h0h2: std::collections::HashSet<(i32, i32, usize)> =
        std::collections::HashSet::new();
    for e in &edges {
        if e.hi == 0 || e.hi == 2 {
            incoming_h0h2.insert(e.tgt);
        }
    }
    // A class is a *tower interior* candidate iff it is simple τ-torsion and not h₀/h₂-anchored.
    let trimmable = |k: &(i32, i32, usize)| {
        order_at.get(k) == Some(&1) && !incoming_h0h2.contains(k)
    };

    // h₁ adjacency among trimmable classes (both directions), and the h₁-divisor lookup used to find
    // a tower's anchor.
    let mut h1_adj: HashMap<(i32, i32, usize), Vec<(i32, i32, usize)>> = HashMap::new();
    let mut h1_divisors: HashMap<(i32, i32, usize), Vec<(i32, i32, usize)>> = HashMap::new();
    for e in &edges {
        if e.hi == 1 && e.power == 0 {
            h1_divisors.entry(e.tgt).or_default().push(e.src);
            if trimmable(&e.src) && trimmable(&e.tgt) {
                h1_adj.entry(e.src).or_default().push(e.tgt);
                h1_adj.entry(e.tgt).or_default().push(e.src);
            }
        }
    }

    // Connected components of the trimmable h₁-graph; a component that reaches above the line is an
    // infinite tower to collapse.
    let mut removed: std::collections::HashSet<(i32, i32, usize)> = std::collections::HashSet::new();
    let mut arrow_anchors: Vec<(i32, i32, usize)> = Vec::new();
    let mut seen: std::collections::HashSet<(i32, i32, usize)> = std::collections::HashSet::new();
    let mut all_trimmable: Vec<(i32, i32, usize)> =
        order_at.keys().copied().filter(trimmable).collect();
    all_trimmable.sort_unstable();
    for start in all_trimmable {
        if seen.contains(&start) {
            continue;
        }
        // BFS the component.
        let mut comp: Vec<(i32, i32, usize)> = Vec::new();
        let mut stack = vec![start];
        seen.insert(start);
        while let Some(k) = stack.pop() {
            comp.push(k);
            for &nb in h1_adj.get(&k).into_iter().flatten() {
                if seen.insert(nb) {
                    stack.push(nb);
                }
            }
        }
        if !comp.iter().any(|&(n, s, _)| above_line(n, s)) {
            continue; // a finite h₁ segment below the line — keep it as ordinary red h₁ lines
        }
        // The foot of the tower: the lowest (min s, then min n) class.
        let foot = *comp.iter().min_by_key(|&&(n, s, _)| (s, n)).unwrap();
        // Its anchor: an h₁-divisor of the foot that is itself not part of the tower.
        let anchor = h1_divisors
            .get(&foot)
            .into_iter()
            .flatten()
            .copied()
            .find(|d| !trimmable(d));
        match anchor {
            Some(a) => {
                // Remove the whole tower; the arrow rises from the anchor below it.
                removed.extend(comp.iter().copied());
                arrow_anchors.push(a);
            }
            None => {
                // No class below: keep the foot as the arrow's base, remove the rest.
                for &k in &comp {
                    if k != foot {
                        removed.insert(k);
                    }
                }
                arrow_anchors.push(foot);
            }
        }
    }

    // --- 6. Emit. ------------------------------------------------------------------------------
    let mut backend = SeqSeeBackend::new(std::io::stdout());
    backend.init(Bidegree::n_s(max_n, max_s))?;

    // Node / line colors (Isaksen's palette). Lines reuse the node-color aliases so that an ordinary
    // product line takes its target's color (rules 6–8).
    backend.define_attribute("m2", json!([{ "color": "#8c8c8c" }])); // gray
    backend.define_attribute("m2tau", json!([{ "color": "#e41a1c" }])); // red
    backend.define_attribute("m2tau2", json!([{ "color": "#377eb8" }])); // blue
    backend.define_attribute("m2tau3", json!([{ "color": "#4daf4a" }])); // green
    backend.define_attribute("m2tauk", json!([{ "color": "#984ea3" }])); // purple
    backend.define_attribute("hidden_tau", json!([{ "color": "#f200ff" }])); // magenta (τ¹)
    backend.define_attribute("hidden_tauk", json!([{ "color": "#ff7f00" }])); // orange (τ^{≥2})
    // Red arrow for the τ-annihilated infinite h₁-towers (rule 9).
    backend.define_attribute("h1tower", json!([{ "color": "#e41a1c", "arrowTip": "simple" }]));

    // Dots (colors/counts authoritative from `tau_module`), minus the collapsed towers.
    for dot in &dots {
        if removed.contains(&(dot.n, dot.s, dot.pos)) {
            continue;
        }
        backend.styled_node(
            Bidegree::n_s(dot.n, dot.s),
            dot.pos,
            &[color_alias(dot.order).to_string()],
            None,
        )?;
    }

    // Product lines, minus any touching a collapsed-tower class.
    for e in &edges {
        if removed.contains(&e.src) || removed.contains(&e.tgt) {
            continue;
        }
        let tgt_order = *order_at.get(&e.tgt).unwrap_or(&0);
        let style = if e.power >= 2 {
            "hidden_tauk" // orange (rule 11)
        } else if e.power == 1 {
            "hidden_tau" // magenta (rule 10)
        } else {
            color_alias(tgt_order) // ordinary product, target's color (rules 6–8)
        };
        let _ = e.hi;
        backend.structline(
            BidegreeGenerator::new(Bidegree::n_s(e.src.0, e.src.1), e.src.2),
            BidegreeGenerator::new(Bidegree::n_s(e.tgt.0, e.tgt.1), e.tgt.2),
            Some(style),
        )?;
    }

    // One red h₁ arrow per collapsed tower, rising slope-1 from its anchor (rule 9).
    // Length 0.7 (not a full unit) so the arrowhead sits in the gap and isn't hidden
    // behind the dot at the next lattice point.
    for a in &arrow_anchors {
        backend.arrow(
            BidegreeGenerator::new(Bidegree::n_s(a.0, a.1), a.2),
            (0.7, 0.7),
            &["h1tower".to_string()],
        )?;
    }

    Ok(())
}
