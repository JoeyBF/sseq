//! Truncate a save: drop every bidegree at or above a cut so a later run recomputes them.
//!
//! # Why this exists
//!
//! A wrong differential is not detected where it is written. It is detected later, as
//! `dx != 0`, in some bidegree that consumed it -- possibly days of compute later, possibly in a
//! run several snapshots downstream, since every restart seeds from the last save. Until now the
//! only recovery was to recompute from scratch, which at the stem-400 frontier is weeks. This
//! turns that total loss into a bounded rollback.
//!
//! # What has to be dropped
//!
//! `(n, s)` is computed from `(n-1, s)` and `(n, s-1)`, so a bad bidegree `(n0, s0)` contaminates
//! exactly the quadrant `{n >= n0, s >= s0}` -- nothing below or to the left can have consumed it.
//! Two cuts are offered:
//!
//! * **t-suffix** (`keep t <= T`) -- a superset of any quadrant with `t0 > T`. Leaves a contiguous
//!   prefix, which is the shape a fresh run would have had, so nothing downstream can be surprised
//!   by a hole. This is the default, and the one to use unless you have a reason not to.
//! * **quadrant** (`drop n >= n0, s >= s0`) -- minimal, keeps more work, but leaves a ragged
//!   frontier. Correct by the dependency argument above, but it produces a save shape no ordinary
//!   run would ever produce, so prefer the suffix cut when the extra recompute is affordable.
//!
//! # Two storage shapes, two deletion mechanisms
//!
//! `ZarrSaveStore::delete` handles the SHARDED kinds and asserts on `ResQi`/`NassauQi`, which are
//! not arrays but structured zarr groups under `qi/n{n}_s{s}/`. A tool that only called `delete`
//! would leave stale quasi-inverses behind for bidegrees whose differentials it had just removed
//! -- a half-truncated save that resumes into subtly wrong results, which is worse than no tool.
//! Both are handled here.
//!
//! # Safety
//!
//! Dry run by default: it prints what it would remove and exits. Deleting requires typing `yes`.
//! There is no undo -- the point of the tool is to destroy work.

use std::{collections::BTreeSet, path::PathBuf};

use ext::save::{SaveKind, ZarrSaveStore};
use sseq::coordinates::Bidegree;

/// Kinds stored as sharded arrays, which `delete` accepts.
const SHARDED: [SaveKind; 5] = [
    SaveKind::NassauDifferential,
    SaveKind::Kernel,
    SaveKind::Differential,
    SaveKind::AugmentationQi,
    SaveKind::ChainMap,
];

/// Kinds stored as structured groups under `qi/n{n}_s{s}/`, which `delete` refuses.
const QI_GROUPS: [(SaveKind, &str); 2] = [
    (SaveKind::ResQi, "res_qi"),
    (SaveKind::NassauQi, "nassau_qi"),
];

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let dir: PathBuf = query::raw("Save directory", str::parse);
    anyhow::ensure!(dir.is_dir(), "{dir:?} is not a directory");

    // Bounds to sweep. Read from the caller rather than discovered, because a sharded array's
    // extent is not the same thing as the set of bidegrees actually present, and guessing wrong
    // in the generous direction only costs a few no-op deletes.
    let max_n: i32 = query::with_default("Sweep n up to", "400", str::parse);
    let max_s: i32 = query::with_default("Sweep s up to", "202", str::parse);

    let mode = query::with_default(
        "Cut mode: 't' = drop t > T (safe, contiguous prefix), 'q' = drop quadrant n>=n0, s>=s0",
        "t",
        |s| match s {
            "t" | "q" => Ok(s.to_owned()),
            _ => Err("expected 't' or 'q'"),
        },
    );

    let doomed: Vec<Bidegree> = if mode == "t" {
        let keep_t: i32 = query::raw("Keep t <= ", str::parse);
        (0..=max_s)
            .flat_map(|s| (0..=max_n).map(move |n| Bidegree::n_s(n, s)))
            .filter(|b| b.t() > keep_t)
            .collect()
    } else {
        let n0: i32 = query::raw("Drop from n >= ", str::parse);
        let s0: i32 = query::raw("Drop from s >= ", str::parse);
        (s0..=max_s)
            .flat_map(|s| (n0..=max_n).map(move |n| Bidegree::n_s(n, s)))
            .collect()
    };

    let store = ZarrSaveStore::create(&dir)?;

    // Survey first. `exists` is a read, so this is safe to run against a live save, and the counts
    // are what makes the confirmation prompt meaningful rather than a formality.
    println!("\nsurveying {} candidate bidegrees ...", doomed.len());
    let mut present: Vec<(SaveKind, Bidegree)> = Vec::new();
    for &kind in &SHARDED {
        let mut n = 0usize;
        for &b in &doomed {
            if store.exists(kind, b) {
                present.push((kind, b));
                n += 1;
            }
        }
        if n > 0 {
            println!("  {:<22} {n} bidegrees", kind.name());
        }
    }

    let mut qi_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    for &b in &doomed {
        for (_, leaf) in QI_GROUPS {
            let p = dir.join("qi").join(format!("n{}_s{}", b.n(), b.s())).join(leaf);
            if p.is_dir() {
                qi_dirs.insert(p);
            }
        }
    }
    if !qi_dirs.is_empty() {
        println!("  {:<22} {} groups", "qi (structured)", qi_dirs.len());
    }

    if present.is_empty() && qi_dirs.is_empty() {
        println!("\nnothing to remove -- the save holds no data at or above this cut.");
        return Ok(());
    }

    let lo = present.iter().map(|(_, b)| b.t()).min();
    let hi = present.iter().map(|(_, b)| b.t()).max();
    println!(
        "\nwould remove {} sharded payloads and {} qi groups, t range {:?}..={:?}",
        present.len(),
        qi_dirs.len(),
        lo,
        hi
    );

    let go = query::with_default(
        "Type 'yes' to DELETE (anything else is a dry run)",
        "no",
        |s| Ok::<String, std::convert::Infallible>(s.to_owned()),
    );
    if go != "yes" {
        println!("dry run -- nothing was modified.");
        return Ok(());
    }

    let mut removed = 0usize;
    for (kind, b) in &present {
        store.delete(*kind, *b)?;
        removed += 1;
    }
    let mut rmdirs = 0usize;
    for p in &qi_dirs {
        std::fs::remove_dir_all(p)?;
        rmdirs += 1;
    }
    println!("removed {removed} sharded payloads and {rmdirs} qi groups.");

    // Verify rather than assume: re-survey and require the region to read as empty. A delete that
    // silently did nothing would otherwise look identical to success.
    let leftover = SHARDED
        .iter()
        .flat_map(|&k| doomed.iter().map(move |&b| (k, b)))
        .filter(|(k, b)| store.exists(*k, *b))
        .count();
    anyhow::ensure!(
        leftover == 0,
        "{leftover} payloads still readable after deletion -- the save is now half-truncated, do \
         NOT resume from it"
    );
    println!("verified: the cut region reads as empty.");

    Ok(())
}
