//! Compare two ways to compute `Ext_A(M, F_2)` for a finite module `M`:
//!
//!  1. **Field-resolution trick** — resolve the base field `k = S_2` once (Nassau's algorithm),
//!     save it to disk, then reuse that saved resolution as the substrate and take the cohomology
//!     of `Hom_A(P_• ⊗ M, k)` (see [`ext::ext_algebra::field_resolution_ext`]).
//!  2. **Nassau from scratch** — resolve `M` directly with Nassau's algorithm; its generator counts
//!     are `Ext_A(M, k)`.
//!
//! The point of (1) is amortisation: the expensive `S_2` resolution is computed once and reused for
//! every module (and it is the *only* option for infinite modules). This script reports the timing
//! of each phase and checks the two answers agree.
//!
//! Run e.g. `cargo run --release --example field_vs_nassau` (p = 2, Milnor basis).

use std::{path::PathBuf, sync::Arc, time::Instant};

use algebra::{
    Algebra,
    module::{FDModule, Module},
};
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::field_resolution_ext,
    utils::{construct_nassau, parse_module_name},
};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let max_n: i32 = query::with_default("Max n", "80", str::parse);
    let max_s: i32 = query::with_default("Max s", "12", str::parse);
    let module_name: String = query::with_default("Finite module M", "C2", |s| {
        Ok::<_, std::convert::Infallible>(s.to_string())
    });
    let save_root: String = query::with_default(
        "Save directory for the S_2 substrate",
        "/tmp/claude-0/-home-user/42539ace-d89d-5b0c-9473-c34dc5b1d77a/scratchpad/S_2_n80",
        |s| Ok::<_, std::convert::Infallible>(s.to_string()),
    );
    let save_dir = PathBuf::from(&save_root);
    std::fs::create_dir_all(&save_dir)?;

    // The substrate must be resolved slightly beyond the display box: the incoming coboundary into
    // (n, s) is read off the source bidegree (n + 1, s - 1), so we need one extra stem in `n`; and
    // the outgoing coboundary lands in filtration s + 1, so we need one extra in `s`.
    let substrate_max = Bidegree::n_s(max_n + 1, max_s + 1);

    // ---- Phase 1: compute the S_2 substrate and save it -------------------------------------
    println!(
        "== Phase 1: resolve S_2 through n = {max_n}, s = {} and save ==",
        max_s + 1
    );
    let start = Instant::now();
    {
        let k_res = construct_nassau("S_2", Some(save_dir.clone()))?;
        k_res.compute_through_stem(substrate_max);
    }
    let compute_save = start.elapsed();
    let disk = du(&save_dir);
    println!("   computed + saved in {compute_save:.2?}  ({disk})\n");

    // ---- Phase 2: reload the saved substrate and run the field-resolution trick -------------
    println!("== Phase 2: reload the saved S_2 and compute Ext({module_name}, k) via the trick ==");
    let start = Instant::now();
    let substrate = Arc::new(construct_nassau("S_2", Some(save_dir.clone()))?);
    substrate.compute_through_stem(substrate_max); // reads from disk
    let reload = start.elapsed();
    println!("   reloaded the S_2 substrate from disk in {reload:.2?}");

    let m_json = parse_module_name(&module_name)?;
    let module = Arc::new(FDModule::from_json(substrate.algebra(), &m_json)?);
    let top = module.max_degree().unwrap();
    substrate.algebra().compute_basis(max_n + max_s + top + 2);
    module.compute_basis(max_n + max_s + 1);

    let ext = field_resolution_ext(Arc::clone(&substrate), Arc::clone(&module));

    let start = Instant::now();
    let mut field: Vec<(Bidegree, usize)> = Vec::new();
    for s in 0..=max_s {
        for n in 0..=max_n {
            let b = Bidegree::n_s(n, s);
            field.push((b, ext.cohomology_dimension(b).expect("in range")));
        }
    }
    let field_time = start.elapsed();
    println!("   field-resolution cohomology over the box in {field_time:.2?}\n");

    // ---- Phase 3: Nassau from scratch on M --------------------------------------------------
    println!("== Phase 3: resolve {module_name} from scratch with Nassau ==");
    let start = Instant::now();
    let direct = construct_nassau(module_name.as_str(), None)?;
    direct.compute_through_stem(Bidegree::n_s(max_n, max_s));
    let nassau_time = start.elapsed();
    println!("   resolved {module_name} from scratch in {nassau_time:.2?}\n");

    // ---- Compare ----------------------------------------------------------------------------
    let mut mismatches = 0;
    let mut nonzero = 0;
    for &(b, f) in &field {
        let d = direct.number_of_gens_in_bidegree(b);
        if f != d {
            if mismatches < 10 {
                println!("   MISMATCH at {b:?}: field = {f}, nassau = {d}");
            }
            mismatches += 1;
        }
        nonzero += usize::from(f != 0);
    }

    println!("\n== Summary (M = {module_name}, box n ≤ {max_n}, s ≤ {max_s}) ==");
    println!("   nonzero Ext bidegrees:      {nonzero}");
    println!(
        "   agreement:                  {}",
        if mismatches == 0 {
            "PERFECT".to_string()
        } else {
            format!("{mismatches} MISMATCHES")
        }
    );
    println!("   S_2 substrate compute+save: {compute_save:.2?}");
    println!("   S_2 substrate reload:       {reload:.2?}");
    println!("   field-trick (reusing S_2):  {field_time:.2?}");
    println!("   Nassau from scratch on M:   {nassau_time:.2?}");

    Ok(())
}

/// A rough human-readable size of a directory tree.
fn du(dir: &std::path::Path) -> String {
    fn walk(dir: &std::path::Path) -> u64 {
        let mut total = 0;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let Ok(md) = e.metadata() else { continue };
                total += if md.is_dir() {
                    walk(&e.path())
                } else {
                    md.len()
                };
            }
        }
        total
    }
    let bytes = walk(dir);
    if bytes >= 1 << 30 {
        format!("{:.1} GiB on disk", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1} MiB on disk", bytes as f64 / (1u64 << 20) as f64)
    }
}
