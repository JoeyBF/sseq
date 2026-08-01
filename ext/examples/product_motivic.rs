//! Products in the motivic Adams $E_2$ over $\mathbb{F}_2[\tau]$, including the
//! hidden τ-extensions (products that vanish mod τ but carry a τ-power) — the
//! deformation's payload.
//!
//! Prompts for a `Module` (default `S_2`, the sphere), an optional save directory,
//! and `Max n` / `Max s` (default 20 / 12), then prints:
//!   1. the motivic $E_2$ as an $\mathbb{F}_2[\tau]$-module at each bidegree, and
//!   2. the products $a \cdot b$ with $a$ ranging over low-stem classes, flagging
//!      the hidden τ-extensions $a \cdot b = \tau^k g$ ($k > 0$).
//!
//! The product is the ring structure of $\mathrm{Ext}_{A_C}$, so it is the honest
//! ring only for a ring spectrum — the sphere. For a non-sphere module the same
//! chain-level self-maps are computed and reported, but the genuine invariant there
//! is the $\mathrm{Ext}_{A_C}(\mathbb{S})$-module action, which needs a separate unit
//! resolution (the motivic analogue of `utils::get_unit`) — a TODO, not yet wired.

use algebra::module::Module;
use ext::motivic::{Gen, MotivicResolution, query_motivic_module};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let (module, save_dir) = query_motivic_module("Module", "S_2")?;
    let max = Bidegree::n_s(
        query::with_default("Max n", "20", str::parse),
        query::with_default("Max s", "12", str::parse),
    );
    let is_sphere = module.max_degree() == Some(0);
    let res = MotivicResolution::with_module(module, max, save_dir);
    let top_s = max.s() - 1;

    // 1. The motivic E₂ as an F₂[τ]-module at each bidegree (structure theorem):
    //    `free ⊕ ⊕ F₂[τ]/τ^k`, e.g. `2+τ` = F₂[τ]² ⊕ F₂[τ]/τ.
    println!("== motivic Adams E₂ as an F₂[τ]-module (n, s: module) ==");
    for s in 0..=top_s {
        for n in 0..=max.n() {
            let m = res.tau_module(s, n + s);
            if m.free > 0 || !m.torsion.is_empty() {
                println!("  ({n:>2}, {s:>2}): {m}");
            }
        }
    }

    if !is_sphere {
        println!(
            "\n(module is not the sphere — skipping ring products; the invariant here is the \
             Ext_{{A_C}}(𝕊)-module action, which needs a unit resolution: TODO)"
        );
        return Ok(());
    }

    // Name the filtration-1 Hopf generators hᵢ at (2ⁱ−1, 1); everything else by
    // its chart coordinate.
    let hopf: Vec<(i32, Gen)> = (0..)
        .map(|i| (1 << i) - 1)
        .take_while(|&n| n <= max.n())
        .map(|n| (n, Gen { s: 1, t: n + 1, idx: 0 }))
        .filter(|(_, g)| res.algebraic_novikov_rank(g.s, g.t) > 0)
        .collect();
    let name = |g: Gen| -> String {
        hopf.iter()
            .find(|(_, h)| *h == g)
            .map(|(i, _)| format!("h{}", (*i + 1).trailing_zeros()))
            .unwrap_or_else(|| format!("x({},{})", g.t - g.s, g.s))
    };

    // 2. Products a·b, flagging the hidden τ-extensions.
    println!("\n== products a·b (a of stem ≤ {}) — hidden τ-extensions flagged ==", max.n().min(8));
    let mut hidden = 0usize;
    for s_a in 1..=top_s {
        for t_a in s_a..=(s_a + 8) {
            for idx_a in 0..res.algebraic_novikov_rank(s_a, t_a) {
                let a = Gen { s: s_a, t: t_a, idx: idx_a };
                for (b, terms) in res.motivic_products_by(a) {
                    if b.t - b.s > max.n() {
                        continue;
                    }
                    for (g, power) in terms {
                        if power > 0 {
                            println!(
                                "  {}·{} = τ{}·{}   [hidden]",
                                name(a),
                                name(b),
                                if power == 1 { String::new() } else { format!("^{power}") },
                                name(g),
                            );
                            hidden += 1;
                        }
                    }
                }
            }
        }
    }
    println!("\n({hidden} hidden τ-extension term(s) in this range)");

    Ok(())
}
