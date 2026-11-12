//! Lifts Lin-ring chain maps handed out by a ring computation's `EXT_LIN_RING_DISPATCH` (see
//! [`ext::ext_algebra::lin_ring::compute`]), saving their steps in the resolution's save store.
//!
//! The module, save directory and range are asked for as for any resolution (the command line
//! answers them first). `LIN_RING_JOBS` names a file of [`MapJob`] lines; the store must already
//! hold each map's earlier steps, and the steps lifted here are added to it. If `LIN_RING_COSTS` is
//! set, each map's name and the thread seconds its new steps took are written to that file, a tab
//! between them.

use std::{io::Write, sync::Arc};

use algebra::AlgebraType;
use anyhow::Context;
use ext::{
    ext_algebra::lin_ring::{MapJob, lift_jobs},
    utils::query_module,
};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let jobs_path = std::env::var_os("LIN_RING_JOBS").context("LIN_RING_JOBS is not set")?;
    let costs_path = std::env::var_os("LIN_RING_COSTS");

    let text = std::fs::read_to_string(&jobs_path).with_context(|| format!("{jobs_path:?}"))?;
    let jobs = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(MapJob::parse)
        .collect::<anyhow::Result<Vec<_>>>()?;

    let start = std::time::Instant::now();
    let res = Arc::new(query_module(Some(AlgebraType::Milnor), true)?);
    tracing::info!(
        maps = jobs.len(),
        open_s = start.elapsed().as_secs_f64(),
        "lin_ring_worker resolution open"
    );

    let start = std::time::Instant::now();
    let costs = lift_jobs(&res, &jobs)?;
    tracing::info!(
        maps = jobs.len(),
        lift_s = start.elapsed().as_secs_f64(),
        thread_s = costs.iter().map(|(_, c)| c).sum::<f64>(),
        "lin_ring_worker done"
    );

    if let Some(path) = costs_path {
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&path).with_context(|| format!("{path:?}"))?,
        );
        for (name, cost) in costs {
            writeln!(out, "{name}\t{cost}")?;
        }
        out.flush()?;
    }
    Ok(())
}
