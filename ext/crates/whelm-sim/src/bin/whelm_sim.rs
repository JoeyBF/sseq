//! Replay a scheduling trace against the policies and compare them.

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::Parser;
use whelm::{
    config::{OrderTerm, Reservations, ScoreTerm},
    dag::DagConfig,
    prelude::*,
};
use whelm_sim::{
    model::{ClassCurve, PsModel, fit},
    run::{Baseline, BoxPolicy, Metrics, SimSetup, Usage, production, simulate},
    trace::Trace,
};

/// Command-line arguments.
///
/// ```text
/// whelm-sim --trace sched_trace_40518773.jsonl.gz --json results.json
/// whelm-sim --trace ... --closed --rank      # closed-loop arrivals, DAG critical-path priority
/// ```
#[derive(Parser, Debug)]
#[command(about = "Replay a scheduling trace against placement policies")]
struct Args {
    /// The trace (JSONL, optionally gzipped).
    #[arg(long)]
    trace: PathBuf,
    /// Policies to run.
    ///
    /// fifo, backfill, bestfit, backfill-shadow, bestfit-shadow, backfill-noreserve or wspt. All
    /// rank workers by fit and load, not speed; wspt is backfill with Smith's rule (shortest work
    /// first) as the order.
    #[arg(long, value_delimiter = ',', default_value = "fifo,backfill,bestfit")]
    policies: Vec<String>,
    /// Closed-loop arrivals through the DAG layer, rather than at the trace's `ready_s`.
    ///
    /// A job arrives the measured gap after its dependencies complete in the simulation.
    #[arg(long)]
    closed: bool,
    /// With --closed: prioritise by the DAG's upward rank (critical path) rather than group order.
    #[arg(long)]
    rank: bool,
    /// Write all metrics and the model fit as JSON here.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Reservations::reserve_after, seconds.
    #[arg(long, default_value_t = 60.0)]
    reserve_after: f64,
    /// Reservations::max.
    #[arg(long, default_value_t = 1)]
    max_reservations: usize,
    /// Config::age_limit, seconds (aging; none by default).
    #[arg(long)]
    age_limit: Option<f64>,
    /// Count reservations per worker class.
    #[arg(long)]
    per_class: bool,
    /// Jobs above this estimate (GB) are "big" in the metrics.
    #[arg(long, default_value_t = 7.5)]
    big_gb: f64,
    /// Reported baseline: floor, excl, idle, or a constant in GB.
    ///
    /// "floor" is production's rolling RSS floor, replayed from the samples; "excl" the rolling
    /// floor of RSS minus the estimates running (`baseline_excl`); "idle" each worker's median idle
    /// RSS.
    #[arg(long, default_value = "floor")]
    baseline: String,
    /// Window of the rolling floor, seconds.
    #[arg(long, default_value_t = 300.0)]
    floor_window: f64,
    /// GB subtracted from the replayed rolling floor, calibrating for the trace's sampling.
    ///
    /// Larger values loosen admission; RESULTS.md ("Memory model and validation") has the
    /// calibration.
    #[arg(long, default_value_t = 3.0)]
    floor_offset_gb: f64,
    /// Do not replay the trace's RSS samples as reported usage (report the baseline only).
    #[arg(long)]
    no_rss: bool,
    /// Print the policy's explanation for this request periodically while it waits.
    #[arg(long)]
    explain: Option<u64>,
    /// Heartbeat period, seconds.
    #[arg(long, default_value_t = 60.0)]
    heartbeat: f64,
    /// Override the fitted model: every class uses this alpha (keeping fitted speeds)...
    #[arg(long)]
    alpha: Option<f64>,
    /// ... and this saturation point.
    #[arg(long)]
    k_sat: Option<usize>,
    /// Scale every estimate (demands, and what `excl` subtracts) by this.
    ///
    /// Below 1 mimics a recalibrated estimator: admission loosens and modelled overruns rise.
    /// RESULTS.md ("Admission without the double count") has the scales tried.
    #[arg(long, default_value_t = 1.0)]
    est_scale: f64,
    /// Spread of each job's modelled memory use, for counting modelled overruns.
    ///
    /// Jobs occupy their estimate times a per-job fraction, log-normal with this spread around the
    /// trace's median ratio of (RSS - idle) to estimates running...
    #[arg(long, default_value_t = 0.5)]
    usage_sd: f64,
    /// ... capped at this fraction of the (unscaled) estimate (default: no cap).
    #[arg(long, default_value_t = f64::INFINITY)]
    usage_cap: f64,
}

/// The named policy configured from the command line, or `None` for an unknown name.
fn make_policy(name: &str, a: &Args) -> Option<BoxPolicy> {
    let reservations = Reservations {
        reserve_after: Duration::from_secs_f64(a.reserve_after),
        max: a.max_reservations,
        per_class: a.per_class,
        shadow_backfill: false,
    };
    let order = match a.rank {
        true => vec![OrderTerm::Priority, OrderTerm::Rank, OrderTerm::Group],
        false => Config::default().order,
    };
    let backfill = Config {
        order,
        reservations: Some(reservations),
        age_limit: a.age_limit.map(Duration::from_secs_f64),
        score: vec![ScoreTerm::Preferred, ScoreTerm::Load],
        ..Config::default()
    };
    let shadow = Config {
        reservations: Some(Reservations {
            shadow_backfill: true,
            ..reservations
        }),
        ..backfill.clone()
    };
    let tightest = vec![ScoreTerm::Tightest, ScoreTerm::Preferred, ScoreTerm::Load];
    let config = match name {
        "fifo" => Config::fifo(),
        "backfill" => backfill,
        "backfill-shadow" => shadow,
        "bestfit-shadow" => Config {
            score: tightest,
            ..shadow
        },
        "backfill-noreserve" => Config {
            reservations: None,
            ..backfill
        },
        "bestfit" => Config {
            score: tightest,
            ..backfill
        },
        "wspt" => Config {
            order: Config::weighted_completion().order,
            ..backfill
        },
        _ => return None,
    };
    Some(Box::new(Scheduler::new(config)))
}

/// A duration in seconds, in the largest unit that keeps it readable.
fn h(s: f64) -> String {
    if s >= 3600.0 {
        format!("{:.1}h", s / 3600.0)
    } else if s >= 60.0 {
        format!("{:.1}m", s / 60.0)
    } else {
        format!("{s:.1}s")
    }
}

/// Load the trace, fit the service model, replay every policy and print the comparison.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    for p in &args.policies {
        if make_policy(p, &args).is_none() {
            return Err(format!("unknown policy {p}").into());
        }
    }
    let clock = Instant::now();
    let trace = Trace::load(&args.trace)?;
    eprintln!(
        "loaded {} tasks, {} workers in {:.1}s",
        trace.tasks.len(),
        trace.workers.len(),
        clock.elapsed().as_secs_f64()
    );
    let clock = Instant::now();
    let (mut model, report, work) = fit(&trace);
    eprintln!(
        "fitted the service model in {:.1}s",
        clock.elapsed().as_secs_f64()
    );
    if args.alpha.is_some() || args.k_sat.is_some() {
        let kmax = trace.workers.iter().map(|w| w.slots).max().unwrap_or(1);
        for c in model.classes.values_mut() {
            *c = ClassCurve {
                speed: c.speed,
                k_sat: args.k_sat.unwrap_or(c.k_sat).clamp(1, kmax.max(1)),
                alpha: args.alpha.unwrap_or(c.alpha),
            };
        }
    }
    println!("## Service model (f(k) = speed * min(k, k_sat)^alpha, per worker process)\n");
    println!(
        "fit on {} jobs, {} fixed effects: within R^2 {:.3}, residual sd of ln(service time) \
         {:.3} (alpha=0 model: SSE {:.0} vs best {:.0})",
        report.n,
        report.groups,
        report.r2_within,
        report.resid_sd,
        report.sse_alpha0,
        report.sse_best
    );
    println!(
        "ln W ~ group + {:.3} ln target + {:.3} ln(next+1) + {:.3} ln est_gb",
        report.beta[0], report.beta[1], report.beta[2]
    );
    println!(
        "\n| class | speed | k_sat | alpha | f(1) | f(8) | f(16) |\n|---|---|---|---|---|---|---|"
    );
    for (c, m) in &model.classes {
        println!(
            "| {c} | {:.3} | {} | {:.1} | {:.2} | {:.2} | {:.2} |",
            m.speed,
            m.k_sat,
            m.alpha,
            m.throughput(1),
            m.throughput(8),
            m.throughput(16)
        );
    }

    let baseline = match args.baseline.as_str() {
        "floor" => Baseline::RollingFloor {
            window_s: args.floor_window,
            offset_gb: args.floor_offset_gb,
        },
        "excl" => Baseline::RollingFloorExcl {
            window_s: args.floor_window,
            offset_gb: args.floor_offset_gb,
        },
        "idle" => Baseline::PerWorker(trace.idle_baselines()),
        gb => Baseline::PerWorker(vec![gb.parse::<f64>()?; trace.workers.len()]),
    };
    let idle = trace.idle_baselines();
    let ratios = trace.usage_ratios(&idle, 1.0);
    let q = |p: f64| {
        ratios
            .get((p * ratios.len() as f64) as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let (over, samples) = trace.samples_over_budget();
    println!(
        "\n## Memory\n\n(RSS - idle) / estimates running, over {} samples: p10 {:.3}, median \
         {:.3}, p90 {:.3}, p99 {:.3}; trace samples over budget: {over}/{samples}",
        ratios.len(),
        q(0.1),
        q(0.5),
        q(0.9),
        q(0.99)
    );
    let usage = Usage {
        idle,
        median: q(0.5),
        sd: args.usage_sd,
        cap: args.usage_cap,
    };
    let setup = SimSetup {
        trace: &trace,
        work: &work,
        model: &model as &PsModel,
        baseline,
        replay_rss: !args.no_rss,
        heartbeat_s: args.heartbeat,
        closed_loop: args.closed.then(DagConfig::default),
        big_gb: args.big_gb,
        explain: args.explain,
        est_scale: args.est_scale,
        usage: Some(usage),
    };
    let mut results: Vec<Metrics> = vec![production(&setup)];
    let clock = Instant::now();
    let runs: Vec<Metrics> = std::thread::scope(|s| {
        let handles: Vec<_> = args
            .policies
            .iter()
            .map(|p| {
                let policy = make_policy(p, &args).unwrap();
                let setup = &setup;
                s.spawn(move || simulate(setup, p, policy))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    eprintln!(
        "simulated {} policies in {:.1}s",
        runs.len(),
        clock.elapsed().as_secs_f64()
    );
    results.extend(runs);

    let mode = if args.closed {
        format!(
            "closed-loop{}",
            if args.rank { ", DAG-rank priority" } else { "" }
        )
    } else {
        "open-loop".to_string()
    };
    println!("\n## Policies ({mode}; big = est > {} GB)\n", args.big_gb);
    println!(
        "| policy | done | makespan | W/h | slot util | mem util | wait p50 | p90 | p99 | max | \
         big p50 | big p90 | big p99 | big max | group p50 | group p90 | group max | resv | idle \
         draining | dispatch mean/max | modelled over budget |"
    );
    println!("|{}", "---|".repeat(21));
    for m in &results {
        println!(
            "| {} | {}/{} | {} | {:.0} | {:.1}% | {:.1}% | {} | {} | {} | {} | {} | {} | {} | {} \
             | {} | {} | {} | {} | {:.1} slot-h ({:.2}%) | {} | {:.3}% (max +{:.1} GB) |",
            m.policy,
            m.completed,
            m.jobs,
            h(m.makespan_h * 3600.0),
            m.work_per_h,
            100.0 * m.slot_util,
            100.0 * m.mem_util,
            h(m.wait.p50),
            h(m.wait.p90),
            h(m.wait.p99),
            h(m.wait.max),
            h(m.wait_big.p50),
            h(m.wait_big.p90),
            h(m.wait_big.p99),
            h(m.wait_big.max),
            h(m.group_latency.p50),
            h(m.group_latency.p90),
            h(m.group_latency.max),
            m.reservations,
            m.reserved_idle_slot_h,
            100.0 * m.reserved_idle_frac,
            if m.dispatch_us.n > 0 {
                format!("{:.0}/{:.0}us", m.dispatch_us.mean, m.dispatch_us.max)
            } else {
                "-".into()
            },
            100.0 * m.overrun.frac,
            m.overrun.max_excess_gb,
        );
    }
    println!("\n| policy | class | jobs | W/h | slot util | mem util |\n|---|---|---|---|---|---|");
    for m in &results {
        for (c, x) in &m.per_class {
            println!(
                "| {} | {c} | {} | {:.0} | {:.1}% | {:.1}% |",
                m.policy,
                x.jobs,
                x.work_per_h,
                100.0 * x.slot_util,
                100.0 * x.mem_util
            );
        }
    }
    if let Some(path) = &args.json {
        let out =
            serde_json::json!({ "mode": mode, "fit": report, "model": model, "results": results });
        std::fs::write(path, serde_json::to_string_pretty(&out)?)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
