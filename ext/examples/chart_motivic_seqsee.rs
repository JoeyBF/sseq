//! Render the C-motivic Adams $E_2$ (= $\Ext_{A_C}(\mathbb{F}_2[\tau], \mathbb{F}_2[\tau])$) as a
//! [SeqSee](https://github.com/JoeyBF/SeqSee) JSON chart on stdout, with **nodes colored by their
//! $\tau$-torsion type** and the **filtration-one products $h_i$** drawn as structlines.
//!
//! # What is drawn
//!
//! The motivic $E_2$ is computed by the deformation / τ-Bockstein spectral sequence of the merged
//! motivic pipeline: $E_1 = \Ext_{A_C/\tau}$ (the algebraic Novikov $E_2$, equivalently the Adams
//! $E_2$ of $C\tau$), and the τ-Bockstein differentials $d_r\colon (n,s,w) \to (n{+}1, s{-}1, w{+}r)$
//! assemble it into an $\mathbb{F}_2[\tau]$-module. We chart the $E_1$ classes, projected onto the
//! $(n, s)$ plane (weights stacked, deterministically ordered), and color each class by its fate in
//! that spectral sequence — which is exactly its role in the $\mathbb{F}_2[\tau]$-module structure
//! theorem $\mathbb{F}_2[\tau]^{\text{free}} \oplus \bigoplus \mathbb{F}_2[\tau]/\tau^{k}$:
//!
//! * **survives to $E_\infty$** ⇒ `tau_free` (black) — the bottom of a free $\mathbb{F}_2[\tau]$
//!   summand (the class is τ-permanent; inverting τ leaves the classical Adams $E_2$);
//! * **supports a $d_r$** ⇒ `tau_tors_r` (warm color, labelled `τ^r`) — the generator of an
//!   $\mathbb{F}_2[\tau]/\tau^{r}$ torsion summand;
//! * **is hit by a $d_r$** ⇒ `tau_killed` (grey) — a τ-multiple absorbed into a torsion summand,
//!   not an independent module generator.
//!
//! The per-bidegree $\mathbb{F}_2[\tau]$-module is cross-checked against
//! [`MotivicResolution::tau_module`], the pipeline's Smith-normal-form report.
//!
//! # Structlines
//!
//! The filtration-one Hopf generators $h_i$ live at $(n, s) = (2^i - 1, 1)$, weight $-(2^i - 1)$.
//! Multiplication by each $h_i$ is taken from [`MotivicResolution::deformation_products`] and applied
//! with [`Sseq::multiply`] on $E_1$ (the $C\tau$ ring); every nonzero source-generator → target-
//! generator entry becomes an `h_i` structline.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features concurrent --example chart_motivic_seqsee -- 30 17 > motivic.json
//! ```
//!
//! Prompts (or positional args) are `Max n` (default 30) and `Max s` (default 17); the reported box
//! is stems `0..=max_n`, filtrations `0..=max_s - 1`. `MOT_SAVE=<dir>` caches the resolution + lift.

use std::collections::BTreeMap;

use ext::motivic::MotivicResolution;
use fp::{prime::TWO, vector::FpVector};
use serde_json::json;
use sseq::{
    charting::{Backend, SeqSeeBackend},
    coordinates::{
        Bidegree, BidegreeGenerator, degree::MultiDegree, element::MultiDegreeElement,
    },
};

/// A SeqSee color for τ^k-torsion generators, ramped warm by order `k`.
fn torsion_color(k: u32) -> &'static str {
    match k {
        1 => "#d62728", // red
        2 => "#ff7f0e", // orange
        3 => "#bcbd22", // yellow-green
        4 => "#9467bd", // purple
        5 => "#e377c2", // pink
        _ => "#8c564b", // brown (order ≥ 6)
    }
}

/// The node style bucket assigned to a single $E_1$ class.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fate {
    /// Survives to $E_\infty$: a free $\mathbb{F}_2[\tau]$ generator.
    Free,
    /// Supports a $d_r$: generates an $\mathbb{F}_2[\tau]/\tau^{r}$ torsion summand.
    TorsionSource(u32),
    /// Hit by a $d_r$: a τ-multiple, not an independent generator.
    Killed,
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

    // Page-1 (E_1) dimension of the (n, s, w) slice. The SS is sparse — an unregistered slice was
    // never given classes, and `page_data` panics on it — so an undefined slice is dimension 0.
    let dim_at = |n: i32, s: i32, w: i32| -> usize {
        let deg = MultiDegree::from([n, s, w]);
        sseq.get_dimension(deg).unwrap_or(0)
    };

    // Weights present (E_1-nonzero) at each (n, s), sorted for a deterministic stacking order.
    let mut weights_at: BTreeMap<(i32, i32), Vec<i32>> = BTreeMap::new();
    for b in sseq.iter_degrees() {
        let [n, s, w] = b.coords();
        if in_box(n, s) && dim_at(n, s, w) > 0 {
            weights_at.entry((n, s)).or_default().push(w);
        }
    }
    for ws in weights_at.values_mut() {
        ws.sort_unstable();
        ws.dedup();
    }

    // Flat position of the weight-`w` block within the projected (n, s) node: the total dimension of
    // all lighter weights present there.
    let offset_at = |n: i32, s: i32, w: i32| -> usize {
        weights_at
            .get(&(n, s))
            .into_iter()
            .flatten()
            .take_while(|&&ww| ww < w)
            .map(|&ww| dim_at(n, s, ww))
            .sum()
    };

    // --- Classify every E_1 class by its τ-Bockstein fate. -------------------------------------
    //
    // `fate[(n, s, w)][i]` is the fate of raw class `i` of that slice. Start every present class as a
    // free-summand candidate; each τ-Bockstein differential then demotes its source pivot (torsion
    // generator) and target pivot (killed) accordingly. A class alive at the last computed page with
    // no differential touching it stays `Free`.
    let mut fate: BTreeMap<(i32, i32, i32), Vec<Fate>> = BTreeMap::new();
    for (&(n, s), ws) in &weights_at {
        for &w in ws {
            fate.insert((n, s, w), vec![Fate::Free; dim_at(n, s, w)]);
        }
    }

    for b in sseq.iter_degrees() {
        let [n, s, w] = b.coords();
        if !in_box(n, s) {
            continue;
        }
        let diffs = sseq.differentials(b);
        for r in diffs.min_degree()..diffs.len() {
            for (src, tgt) in diffs[r].get_source_target_pairs() {
                // Pivot (leading nonzero) raw index identifies the class canonically.
                if let Some(i) = src.first_nonzero().map(|(i, _)| i) {
                    if let Some(v) = fate.get_mut(&(n, s, w)) {
                        if let Some(slot) = v.get_mut(i) {
                            // A genuine torsion generator is the shortest differential off it.
                            if let Fate::TorsionSource(old) = *slot {
                                *slot = Fate::TorsionSource(old.min(r as u32));
                            } else if *slot == Fate::Free {
                                *slot = Fate::TorsionSource(r as u32);
                            }
                        }
                    }
                }
                let target = MultiDegree::from([n + 1, s - 1, w + r]);
                let [tn, ts, tw] = target.coords();
                if let Some(j) = tgt.first_nonzero().map(|(j, _)| j) {
                    if let Some(v) = fate.get_mut(&(tn, ts, tw)) {
                        if let Some(slot) = v.get_mut(j) {
                            if *slot == Fate::Free {
                                *slot = Fate::Killed;
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Emit the chart. -----------------------------------------------------------------------
    let mut backend = SeqSeeBackend::new(std::io::stdout());
    backend.init(Bidegree::n_s(max_n, max_s))?;

    // Register the node styles.
    backend.define_attribute("tau_free", json!([{ "color": "black" }]));
    backend.define_attribute("tau_killed", json!([{ "color": "#bbbbbb" }]));
    for k in 1..=8u32 {
        backend.define_attribute(&format!("tau_tors_{k}"), json!([{ "color": torsion_color(k) }]));
    }

    // Draw nodes (before structlines, whose endpoints must already exist), colored by fate.
    let mut torsion_summands: BTreeMap<(i32, i32), Vec<u32>> = BTreeMap::new();
    for (&(n, s), ws) in &weights_at {
        for &w in ws {
            let off = offset_at(n, s, w);
            let slice_fate = &fate[&(n, s, w)];
            for (i, &f) in slice_fate.iter().enumerate() {
                let (attr, label) = match f {
                    Fate::Free => ("tau_free".to_string(), None),
                    Fate::Killed => ("tau_killed".to_string(), None),
                    Fate::TorsionSource(k) => {
                        torsion_summands.entry((n, s)).or_default().push(k);
                        let lbl = if k == 1 {
                            "τ".to_string()
                        } else {
                            format!("τ^{k}")
                        };
                        (format!("tau_tors_{k}"), Some(lbl))
                    }
                };
                backend.styled_node(Bidegree::n_s(n, s), off + i, &[attr], label)?;
            }
        }
    }

    // Cross-check the torsion multiset against the pipeline's Smith-normal-form report and warn (on
    // stderr) about any discrepancy, without aborting the chart.
    for (&(n, s), orders) in &torsion_summands {
        let mut got = orders.clone();
        got.sort_unstable();
        let mut want = res.tau_module(s, n + s).torsion;
        want.sort_unstable();
        if got != want {
            eprintln!(
                "warning: τ-torsion mismatch at (n={n}, s={s}): chart {got:?} vs tau_module {want:?}"
            );
        }
    }

    // --- Filtration-one products h_i as structlines. -------------------------------------------
    //
    // h_i is at (2^i - 1, 1), index 0, for every i with 2^i - 1 within the reported stems.
    let mut hi: Vec<(i32, String)> = Vec::new();
    let mut i = 0;
    loop {
        let n = (1i32 << i) - 1;
        if n > max_n {
            break;
        }
        hi.push((n, format!("h{i}")));
        i += 1;
    }
    let prods = res.deformation_products(
        &hi.iter()
            .map(|&(n, _)| (Bidegree::n_s(n, 1), 0))
            .collect::<Vec<_>>(),
    );

    for b in sseq.iter_degrees() {
        let [n, s, w] = b.coords();
        if !in_box(n, s) {
            continue;
        }
        let dim = dim_at(n, s, w);
        if dim == 0 {
            continue;
        }
        let off_src = offset_at(n, s, w);
        for (k, prod) in prods.iter().enumerate() {
            let (hn, hname) = &hi[k];
            let (tn, ts) = (n + hn, s + 1);
            if !in_box(tn, ts) {
                continue;
            }
            for i in 0..dim {
                let mut v = FpVector::new(TWO, dim);
                v.set_entry(i, 1);
                let class = MultiDegreeElement::new(b, v);
                let Some(image) = sseq.multiply(&class, prod) else {
                    continue;
                };
                let [rn, rs, rw] = image.degree().coords();
                if (rn, rs) != (tn, ts) {
                    continue;
                }
                let off_tgt = offset_at(rn, rs, rw);
                let img = image.vec();
                for (j, entry) in img.iter().enumerate() {
                    if entry == 0 {
                        continue;
                    }
                    backend.structline(
                        BidegreeGenerator::new(Bidegree::n_s(n, s), off_src + i),
                        BidegreeGenerator::new(Bidegree::n_s(tn, ts), off_tgt + j),
                        Some(hname),
                    )?;
                }
            }
        }
    }

    Ok(())
}
