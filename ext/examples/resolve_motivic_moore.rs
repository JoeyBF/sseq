//! Resolve a *non-trivial* module over $A_C/\tau$ and read its motivic Adams $E_2$:
//! the mod-2 Moore space $S/2$ (the cofiber of $2 = h_0$), whose two cells are
//! joined by $Q_0 = \mathrm{Sq}^1$.
//!
//! Demonstrates [`MotivicResolution::with_module`]: any [`FDModule`] over
//! [`CTauAlgebra`] flows through the whole deformation pipeline — weights, the
//! $A_C$ lift, the $\mathbb{F}_2[\tau]$-module structure, products, Masseys.
//! Because $2 = h_0$ is coned off, the $h_0$-tower on the bottom cell is truncated
//! (compare `resolve_motivic`, where $h_0^n \ne 0$ for all $n$).
//!
//! Prompts for `Max n` / `Max s` (default 12 / 8). Set `MOT_SAVE=<dir>` to cache.

use std::sync::Arc;

use algebra::{Algebra, CTauAlgebra, module::FDModule};
use bivec::BiVec;
use ext::motivic::MotivicResolution;
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let max = Bidegree::n_s(
        query::with_default("Max n", "12", str::parse),
        query::with_default("Max s", "8", str::parse),
    );

    // Build S/2: cells x₀ (degree 0) and x₁ (degree 1), with Q₀·x₀ = x₁.
    let algebra = Arc::new(CTauAlgebra::new());
    algebra.compute_basis(max.t() + 2);
    let mut module = FDModule::new(algebra, "S/2".to_string(), BiVec::from_vec(0, vec![1, 1]));
    module.set_action(1, 0, 0, 0, &[1]);

    let save_dir = std::env::var_os("MOT_SAVE").map(std::path::PathBuf::from);
    let res = MotivicResolution::with_module(Arc::new(module), max, save_dir);

    println!("n,s,alg_nov,classical,tau_torsion");
    for s in 0..=(max.s() - 1) {
        for n in 0..=max.n() {
            let alg_nov = res.algebraic_novikov_rank(s, n + s);
            let m = res.tau_module(s, n + s);
            let torsion: String = m
                .torsion
                .iter()
                .map(|&k| if k == 1 { "τ".to_string() } else { format!("τ^{k}") })
                .collect::<Vec<_>>()
                .join("+");
            if alg_nov > 0 || m.free > 0 || !torsion.is_empty() {
                println!("{n},{s},{alg_nov},{},{torsion}", m.free);
            }
        }
    }

    Ok(())
}
