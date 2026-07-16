//! The full product-structure workflow behind the field trick, end to end and **persisted to disk**:
//!
//!  1. **Resolve + save the sphere** `S_2` once with Nassau's algorithm (the one expensive artifact).
//!  2. **Sphere products** — the ordinary `Ext(k, k)` ring, via the minimal path. Each multiplier's
//!     chain self-map is extended once across the chart and saved under the sphere's `products/` dir.
//!  3. **Module products** — `Ext(M, k)` as an `Ext(k, k)`-module, via the field trick reusing the
//!     *same* saved sphere as the substrate `P•`. Products read off the closed-form cup; the sphere
//!     self-maps and the `δ_Q` matrices are cached to disk too.
//!
//! Everything the products need is saved, so a re-run reloads instead of recomputing, and the
//! computation is safe to interrupt and resume. Run e.g.
//!
//! ```text
//! cargo run --release --features concurrent --example field_product_structure
//! ```
//!
//! (p = 2, Milnor basis; concurrency strongly recommended at large stems.)

use std::{path::PathBuf, sync::Arc, time::Instant};

use algebra::{Algebra, module::FDModule, module::Module};
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::{ExtAlgebra, field_resolution_products_with_save_dir},
    utils::{construct_nassau, parse_module_name},
};
use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

/// Multiply **every** class in the box by **every** generator of `Ext(k, k)` (the full module
/// action; for the sphere this is the full ring). Returns `(products computed, nonzero products, a
/// checksum)`. The checksum both reports work and defeats dead-code elimination.
fn full_products<CC, CCU>(alg: &ExtAlgebra<CC, CCU>, max: Bidegree) -> (u64, u64, u64)
where
    CC: FreeChainComplex,
    CCU: FreeChainComplex<Algebra = CC::Algebra> + ext::chain_complex::AugmentedChainComplex,
{
    let mut computed = 0u64;
    let mut nonzero = 0u64;
    let mut checksum = 0u64;

    // Collect the multipliers once: every generator of Ext(k, k) in the box.
    let mut multipliers: Vec<BidegreeElement> = Vec::new();
    for n in 0..=max.n() {
        for s in 0..=max.s() {
            let b = Bidegree::n_s(n, s);
            for i in 0..alg.unit_dimension(b) {
                multipliers.push(alg.unit_generator(BidegreeGenerator::new(b, i)));
            }
        }
    }

    for n in 0..=max.n() {
        for s in 0..=max.s() {
            let b = Bidegree::n_s(n, s);
            for i in 0..alg.dimension(b) {
                let x = alg.generator(BidegreeGenerator::new(b, i));
                for y in &multipliers {
                    // Only products that land inside the box: this keeps every lifted map a margin
                    // below the resolved edge (where a Nassau substrate has no quasi-inverse), and
                    // products landing outside the box are not part of the requested chart anyway.
                    let prod_deg = x.degree() + y.degree();
                    if prod_deg.n() > max.n() || prod_deg.s() > max.s() {
                        continue;
                    }
                    if let Some(p) = alg.try_multiply(&x, y) {
                        computed += 1;
                        if !p.vec().is_zero() {
                            nonzero += 1;
                            checksum += p.vec().iter().map(u64::from).sum::<u64>();
                        }
                    }
                }
            }
        }
    }
    (computed, nonzero, checksum)
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let max_n: i32 = query::with_default("Max n", "100", str::parse);
    let max_s: i32 = query::with_default("Max s", "30", str::parse);
    let module_name: String = query::with_default("Finite module M", "C2", |s| {
        Ok::<_, std::convert::Infallible>(s.to_string())
    });
    let save_root: String = query::with_default(
        "Save root (sphere substrate + module δ_Q live here)",
        "/tmp/claude-0/-home-user/42539ace-d89d-5b0c-9473-c34dc5b1d77a/scratchpad/field_products",
        |s| Ok::<_, std::convert::Infallible>(s.to_string()),
    );

    let box_max = Bidegree::n_s(max_n, max_s);
    // The substrate must be resolved a little beyond the display box: the field-trick coboundary into
    // (n, s) reads the source (n + 1, s − 1) and lands in filtration s + 1, and products shift up a
    // filtration, so keep a couple of stems of margin.
    let substrate_max = Bidegree::n_s(max_n + 2, max_s + 3);

    let sphere_dir = PathBuf::from(&save_root).join("S_2");
    let module_dir = PathBuf::from(&save_root).join(format!("{module_name}_delta"));
    std::fs::create_dir_all(&sphere_dir)?;
    std::fs::create_dir_all(&module_dir)?;

    // ---- Phase 1: resolve and save the sphere -----------------------------------------------
    println!("== Phase 1: resolve S_2 through {substrate_max:?} and save ==");
    let start = Instant::now();
    let sphere = Arc::new(construct_nassau("S_2", Some(sphere_dir.clone()))?);
    sphere.compute_through_stem(substrate_max);
    sphere.algebra().compute_basis(substrate_max.t() + 1);
    let phase1 = start.elapsed();
    println!("   done in {phase1:.2?}  ({})\n", du(&sphere_dir));

    // ---- Phase 2: the sphere's own product ring (minimal path) -------------------------------
    // Skippable: set `FIELD_SKIP_SPHERE_PRODUCTS=1` to regenerate only a module's products against a
    // sphere whose product ring (the shared `prod_*` self-maps) is already on disk — the "the sphere
    // is a fixed cost, each module is cheap" scenario.
    let skip_sphere = std::env::var_os("FIELD_SKIP_SPHERE_PRODUCTS").is_some();
    let phase2 = if skip_sphere {
        println!("== Phase 2: SKIPPED (reusing the sphere product ring already on disk) ==\n");
        None
    } else {
        println!("== Phase 2: Ext(k, k) products (full ring, maps saved under S_2/products) ==");
        let sphere_alg = ExtAlgebra::new(Arc::clone(&sphere), Arc::clone(&sphere));
        sphere_alg.compute_through_bidegree(box_max);
        let start = Instant::now();
        let (computed, nonzero, checksum) = full_products(&sphere_alg, box_max);
        let elapsed = start.elapsed();
        println!(
            "   {computed} products ({nonzero} nonzero, checksum {checksum}) in {elapsed:.2?}  \
             ({})\n",
            du(&sphere_dir)
        );
        Some((elapsed, nonzero))
    };

    // ---- Phase 3: Ext(M, k) products via the field trick, reusing the saved sphere ----------
    println!(
        "== Phase 3: Ext({module_name}, k) products via the field trick (reusing saved S_2) =="
    );
    let m_json = parse_module_name(&module_name)?;
    let module = Arc::new(FDModule::from_json(sphere.algebra(), &m_json)?);
    module.compute_basis(substrate_max.t() + 1);

    let field = field_resolution_products_with_save_dir(
        Arc::clone(&sphere),
        Arc::clone(&module),
        Some(module_dir.clone()),
    );
    field.compute_through_bidegree(box_max);

    let start = Instant::now();
    let (m_computed, m_nonzero, m_checksum) = full_products(&field, box_max);
    let phase3 = start.elapsed();
    println!(
        "   {m_computed} products ({m_nonzero} nonzero, checksum {m_checksum}) in {phase3:.2?}"
    );
    println!(
        "   δ_Q cache: {}   sphere self-maps: reused from S_2/products\n",
        du(&module_dir)
    );

    // ---- Summary ----------------------------------------------------------------------------
    println!("== Summary (M = {module_name}, box n ≤ {max_n}, s ≤ {max_s}) ==");
    println!("   S_2 resolve + save:      {phase1:.2?}");
    match phase2 {
        Some((elapsed, nonzero)) => {
            println!("   Ext(k, k) products:      {elapsed:.2?}  ({nonzero} nonzero)");
        }
        None => println!("   Ext(k, k) products:      (skipped; reused from disk)"),
    }
    println!("   Ext({module_name}, k) products:  {phase3:.2?}  ({m_nonzero} nonzero)");
    println!(
        "   total on disk:           {}",
        du(&PathBuf::from(&save_root))
    );

    Ok(())
}

/// A rough human-readable size and file count of a directory tree. `metadata()` on a `DirEntry`
/// follows symlinks and can transiently fail while files are being written, so we retry via
/// `symlink_metadata` and only skip an entry if both fail — otherwise a single hiccup would zero out
/// an entire subtree's reported size.
fn du(dir: &std::path::Path) -> String {
    fn walk(dir: &std::path::Path, bytes: &mut u64, files: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            let md = match std::fs::symlink_metadata(&path) {
                Ok(md) => md,
                Err(_) => continue,
            };
            if md.is_dir() {
                walk(&path, bytes, files);
            } else {
                *bytes += md.len();
                *files += 1;
            }
        }
    }
    let (mut bytes, mut files) = (0u64, 0u64);
    walk(dir, &mut bytes, &mut files);
    let size = if bytes >= 1 << 30 {
        format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1u64 << 20) as f64)
    };
    format!("{size} in {files} files")
}
