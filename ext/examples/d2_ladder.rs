//! Times each pass of the d2 computation on the sphere resolved through a stem.

use std::{
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use ext::{
    chain_complex::ChainComplex,
    secondary::{SecondaryHomotopyPrep, SecondaryLift, SecondaryResolution},
};
use fp::vector::FpVector;
use sseq::coordinates::Bidegree;

/// Peak and current resident set size, in GiB.
fn rss() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<f64>().ok())
            .unwrap_or(0.0)
            / (1024.0 * 1024.0)
    };
    (field("VmHWM:"), field("VmRSS:"))
}

/// Usage: `d2_ladder <save_dir> <max_n> <max_s> <d2_out>`.
///
/// Prints one `LADDER` line to stderr with each pass's wall time, the summed time of the homotopy
/// steps' product, quasi-inverse and finishing parts, and the peak and current RSS after each pass.
/// Writes every nonzero d2, in the generator basis, to `d2_out`.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 5,
        "usage: d2_ladder <save_dir> <max_n> <max_s> <d2_out>"
    );
    let save_dir = std::path::PathBuf::from(&args[1]);
    let max = Bidegree::n_s(args[2].parse()?, args[3].parse()?);

    let start = Instant::now();
    let resolution = Arc::new(ext::utils::construct("S_2@milnor", Some(save_dir))?);
    resolution.compute_through_stem(max);
    let resolve_s = start.elapsed().as_secs_f64();
    let rss_resolve = rss();

    let lift = SecondaryResolution::new(Arc::clone(&resolution));

    let start = Instant::now();
    lift.initialize_homotopies();
    lift.compute_composites();
    let composite_s = start.elapsed().as_secs_f64();
    let rss_composite = rss();

    let start = Instant::now();
    lift.compute_intermediates();
    let intermediate_s = start.elapsed().as_secs_f64();
    let rss_intermediate = rss();

    // The body of `compute_homotopies`, with the product part (`prepare_homotopy_step`, which
    // applies the previous row's homotopy to d(g)) and the quasi-inverse solve timed apart.
    let prepare_ns = AtomicU64::new(0);
    let qi_ns = AtomicU64::new(0);
    let finish_ns = AtomicU64::new(0);
    let solves = AtomicU64::new(0);
    let elapsed = |since: Instant, acc: &AtomicU64| {
        acc.fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
    };

    let start = Instant::now();
    lift.seed_zero_homotopy();
    let shift = lift.shift();
    let min_t = lift.homotopies()[shift.s()].homotopies.min_degree();
    let s_range = lift.homotopies().range();
    let min = Bidegree::s_t(s_range.start + 1, min_t);
    let step = |b: Bidegree| {
        let t0 = Instant::now();
        let prep = lift.prepare_homotopy_step(b);
        elapsed(t0, &prepare_ns);
        match prep {
            SecondaryHomotopyPrep::Done(range) => range,
            SecondaryHomotopyPrep::NeedsLift(pending) => {
                let t0 = Instant::now();
                let mut results =
                    vec![FpVector::new(lift.prime(), pending.target_dim); pending.num_gens];
                assert!(resolution.apply_quasi_inverse(
                    &mut results,
                    pending.target_b,
                    &pending.intermediates,
                ));
                elapsed(t0, &qi_ns);
                solves.fetch_add(1, Ordering::Relaxed);
                let t0 = Instant::now();
                let range = lift.finish_homotopy_step(pending, &results).unwrap();
                elapsed(t0, &finish_ns);
                range
            }
        }
    };
    sseq::coordinates::iter_s_t(&step, min, lift.max().restrict(s_range.end));
    let homotopy_s = start.elapsed().as_secs_f64();
    let rss_homotopy = rss();

    // Every nonzero d2, read off the homotopy two rows up, as `e3_page` does.
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[4])?);
    let mut nonzero = 0;
    for b in resolution.iter_stem() {
        if b.t() == 0 || !resolution.has_computed_bidegree(b + Bidegree::n_s(-1, 2)) {
            continue;
        }
        let m = lift.homotopy(b.s() + 2).homotopies.hom_k(b.t());
        for (i, row) in m.iter().enumerate() {
            if row.iter().any(|&x| x != 0) {
                nonzero += 1;
                writeln!(out, "d2 x_({}, {}, {i}) = {row:?}", b.n(), b.s())?;
            }
        }
    }

    let secs = |acc: &AtomicU64| acc.load(Ordering::Relaxed) as f64 / 1e9;
    eprintln!(
        "LADDER n={} s={} threads={} resolve_s={resolve_s:.1} composite_s={composite_s:.1} \
         intermediate_s={intermediate_s:.1} homotopy_s={homotopy_s:.1} \
         step_prepare_sum_s={:.1} step_qi_sum_s={:.1} step_finish_sum_s={:.1} solves={} \
         nonzero_d2={nonzero} hwm_gb resolve={:.2} composite={:.2} intermediate={:.2} \
         homotopy={:.2} rss_gb resolve={:.2} composite={:.2} intermediate={:.2} homotopy={:.2}",
        max.n(),
        max.s(),
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        secs(&prepare_ns),
        secs(&qi_ns),
        secs(&finish_ns),
        solves.load(Ordering::Relaxed),
        rss_resolve.0,
        rss_composite.0,
        rss_intermediate.0,
        rss_homotopy.0,
        rss_resolve.1,
        rss_composite.1,
        rss_intermediate.1,
        rss_homotopy.1,
    );
    Ok(())
}
