//! Replay a scheduling trace against the policies and compare them.

use std::{path::PathBuf, time::Instant};

use clap::Parser;
use sched::{
    BackfillConfig, BestFit, BestFitConfig, DagConfig, Greedy, GreedyConfig, LaneSet, Lanes,
    LanesConfig, PriorityBackfill, Resources,
    sim::{
        model::{ClassCurve, PsModel, fit},
        run::{Baseline, BoxPolicy, Metrics, SimSetup, production, simulate},
        trace::Trace,
    },
};

/// Command-line arguments.
///
/// ```text
/// sched-sim --trace sched_trace_40518773.jsonl.gz --json results.json
/// sched-sim --trace ... --closed --rank      # closed-loop arrivals, DAG critical-path priority
/// ```
#[derive(Parser, Debug)]
#[command(about = "Replay a scheduling trace against placement policies")]
struct Args {
    /// The trace (JSONL, optionally gzipped).
    #[arg(long)]
    trace: PathBuf,
    /// Policies to run (greedy, backfill, bestfit, lanes, backfill-noreserve).
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "greedy,backfill,bestfit,lanes"
    )]
    policies: Vec<String>,
    /// Closed-loop arrivals (a job arrives the measured gap after its dependencies complete in the
    /// simulation) through the DAG layer, instead of at the trace's `ready_s`.
    #[arg(long)]
    closed: bool,
    /// With --closed: prioritise by the DAG's upward rank (critical path) instead of group order.
    #[arg(long)]
    rank: bool,
    /// Write all metrics and the model fit as JSON here.
    #[arg(long)]
    json: Option<PathBuf>,
    /// BackfillConfig::reserve_after, seconds.
    #[arg(long, default_value_t = 60.0)]
    reserve_after: f64,
    /// BackfillConfig::max_reservations.
    #[arg(long, default_value_t = 1)]
    max_reservations: usize,
    /// BackfillConfig::age_limit, seconds (aging; none by default).
    #[arg(long)]
    age_limit: Option<f64>,
    /// Count reservations per worker class.
    #[arg(long)]
    per_class: bool,
    /// Lanes: the worker class used as big lanes.
    #[arg(long, default_value = "h200")]
    lane_class: String,
    /// Lanes: headroom (GB) a lane keeps free from small jobs.
    #[arg(long, default_value_t = 12.0)]
    lane_reserve_gb: f64,
    /// Jobs above this estimate (GB) are "big" (metrics, lanes).
    #[arg(long, default_value_t = 7.5)]
    big_gb: f64,
    /// Reported baseline: "floor" (production's rolling RSS floor, replayed from the samples),
    /// "idle" (each worker's median idle RSS), or a constant in GB.
    #[arg(long, default_value = "floor")]
    baseline: String,
    /// Window of the rolling floor, seconds.
    #[arg(long, default_value_t = 300.0)]
    floor_window: f64,
    /// GB subtracted from the replayed rolling floor (calibration; see RESULTS.md).
    #[arg(long, default_value_t = 3.0)]
    floor_offset_gb: f64,
    /// Do not replay the trace's RSS samples as reported usage (report the baseline only).
    #[arg(long)]
    no_rss: bool,
    /// Print the policy's explanation for this request every 10 simulated minutes while it waits.
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
}

/// The named policy configured from the command line, or `None` for an unknown name.
fn make_policy(name: &str, a: &Args) -> Option<BoxPolicy> {
    let backfill = BackfillConfig {
        reserve_after: a.reserve_after,
        max_reservations: a.max_reservations,
        per_class_reservations: a.per_class,
        age_limit: a.age_limit,
        ..BackfillConfig::default()
    };
    Some(match name {
        "greedy" => Box::new(Greedy::new(GreedyConfig::default())),
        "backfill" => Box::new(PriorityBackfill::new(backfill)),
        "backfill-noreserve" => Box::new(PriorityBackfill::new(BackfillConfig {
            max_reservations: 0,
            ..backfill
        })),
        "bestfit" => Box::new(BestFit::new(BestFitConfig {
            backfill,
            prefer_penalty: 0,
        })),
        "lanes" => Box::new(Lanes::new(LanesConfig {
            backfill,
            lanes: LaneSet::Classes(vec![a.lane_class.clone()]),
            big_threshold: Resources::mem_gb(a.big_gb),
            lane_reserve: Resources::mem_gb(a.lane_reserve_gb),
        })),
        _ => return None,
    })
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
        "idle" => Baseline::PerWorker(trace.idle_baselines()),
        gb => Baseline::PerWorker(vec![gb.parse::<f64>()?; trace.workers.len()]),
    };
    let setup = SimSetup {
        trace: &trace,
        work: &work,
        model: &model as &PsModel,
        baseline,
        replay_rss: !args.no_rss,
        heartbeat_s: args.heartbeat,
        closed_loop: args.closed.then(|| DagConfig {
            rank_priority: args.rank,
            rank_scale: 1.0,
            ..DagConfig::default()
        }),
        big_gb: args.big_gb,
        explain: args.explain,
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
         draining | dispatch mean/max |"
    );
    println!("|{}", "---|".repeat(20));
    for m in &results {
        println!(
            "| {} | {}/{} | {} | {:.0} | {:.1}% | {:.1}% | {} | {} | {} | {} | {} | {} | {} | {} \
             | {} | {} | {} | {} | {:.1} slot-h ({:.2}%) | {} |",
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
