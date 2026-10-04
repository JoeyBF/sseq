//! Simulate a whole Nassau run, built as one DAG a priori, under several dispatch plans.

use std::{path::PathBuf, time::Duration};

use clap::Parser;
use whelm::{config::Defer, dag::DagConfig, speed::Timing};
use whelm_sim::{
    model::fit,
    plan::{SpeedPlan, timing_named},
    trace::Trace,
    whole::{Census, Fleet, GroupKey, Pin, Placement, Plan, WholeConfig, World, simulate},
};

/// Whole-run simulation of a Nassau resolution.
///
/// ```text
/// whelm-whole --trace sched_trace.jsonl.gz --census a.csv --census b.csv --json out.json
/// ```
#[derive(Parser, Debug)]
#[command(about = "Simulate a whole Nassau run as one DAG under several dispatch plans")]
struct Args {
    /// The scheduling trace (for the service model and the cost model).
    #[arg(long)]
    trace: PathBuf,
    /// Census CSVs (later files override earlier ones).
    #[arg(long, required = true)]
    census: Vec<PathBuf>,
    /// Region: largest stem.
    #[arg(long, default_value_t = 400)]
    max_n: i32,
    /// Region: largest homological degree.
    #[arg(long, default_value_t = 202)]
    max_s: i32,
    /// Profile length cap (NASSAU_MAX_SUBALGEBRA + 1).
    ///
    /// Default: the cap that best reproduces the census.
    #[arg(long)]
    max_profile_len: Option<usize>,
    /// Fleet as class:workers:slots,... (default: the trace's workers).
    #[arg(long)]
    fleet: Option<String>,
    /// Today's open-bidegree cap (coordinator threads).
    #[arg(long, default_value_t = 24)]
    open: usize,
    /// Today's in-flight cap per bidegree (walk threads).
    #[arg(long, default_value_t = 32)]
    walk: usize,
    /// Aging for the rank plans, seconds.
    #[arg(long, default_value_t = 3600.0)]
    age_limit: f64,
    /// Plans: today, group, rank, rank-oracle, rank-noage, rank-oracle-noage, grouprank[-oracle].
    ///
    /// "grouprank[-oracle]" is oldest bidegree first, rank within. Each takes optional placement
    /// suffixes: "+fast" (fastest first), "+eft" (earliest finish, waiting up to --max-defer for a
    /// faster worker), "+eft0" (earliest finish, no waiting: the same as +fast), "+fastonly" (only
    /// the fast class), "+cpop" (critical tasks pinned to the fast class), "+learn" (speeds
    /// learned online from completions, every worker reporting 1: --timing q-learn for this plan),
    /// "+age" (aging at --age-limit for any plan), "+smajor"/"+tmajor"/"+stem" (bidegrees ordered
    /// by (s, t), (t, s) or (t - s, s) rather than by arrival).
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "today,group,rank,rank-oracle"
    )]
    plans: Vec<String>,
    /// Smallest task work (H200-seconds).
    #[arg(long, default_value_t = 0.05)]
    min_work: f64,
    /// Longest voluntary wait for a faster worker (+eft), seconds.
    ///
    /// Longer waits catch more fast slots but can leave urgent jobs queued behind scarce ones;
    /// RESULTS.md has the measurements.
    #[arg(long, default_value_t = Defer::default().max_wait.as_secs_f64())]
    max_defer: f64,
    /// +eft: wait only if the expected finish improves by this fraction of the job's work.
    ///
    /// Higher values wait less often: fewer cases where waiting backfires, and less of its gain.
    /// RESULTS.md (`whelm-pisa`) has the trade-off.
    #[arg(long, default_value_t = Defer::default().min_gain)]
    min_gain: f64,
    /// Seed of the true costs' noise (0: the reference draw); vary it to average over draws.
    #[arg(long, default_value_t = 0)]
    noise_seed: u64,
    /// `DagConfig::rank_epsilon` for the rank plans.
    #[arg(long, default_value_t = DagConfig::default().rank_epsilon)]
    rank_epsilon: f64,
    /// At most this many walks open at once (it changes the schedule).
    ///
    /// A frontier budget of the simulated coordinator, which holds the jobs of further walks back.
    #[arg(long)]
    max_open: Option<usize>,
    /// Machine model: p, q, q-learn or r.
    ///
    /// p (identical), q (related, reported speeds), q-learn (related, learned, every worker
    /// reporting 1) or r (unrelated: learned per job kind and worker class).
    #[arg(long, default_value = "q", value_parser = timing_named)]
    timing: Timing,
    /// Override a class's single-job speed, as class=speed,... (RESULTS.md has the measured ratio).
    #[arg(long, value_delimiter = ',')]
    class_speed: Vec<String>,
    /// Make throughput exactly linear up to the slot count (as dslab's exclusive cores).
    #[arg(long)]
    linear_ps: bool,
    /// Write the world as dslab-dag input (dag.yaml, dag_est.yaml, system.yaml) here, then exit.
    #[arg(long)]
    export_dslab: Option<PathBuf>,
    /// Write results as JSON here.
    #[arg(long)]
    json: Option<PathBuf>,
}

/// Format seconds as hours.
fn hours(s: f64) -> String {
    format!("{:.1} h", s / 3600.0)
}

/// The plan a command-line name stands for.
fn plan_named(name: &str, args: &Args) -> Plan {
    let rank = |oracle, age: bool, group_first| Plan::Rank {
        oracle,
        age_limit: age.then_some(Duration::from_secs_f64(args.age_limit)),
        group_first,
    };
    match name {
        "today" => Plan::Today {
            open: args.open,
            per_bidegree: args.walk,
        },
        "group" => Plan::Group,
        "rank" => rank(false, true, false),
        "rank-oracle" => rank(true, true, false),
        "rank-noage" => rank(false, false, false),
        "rank-oracle-noage" => rank(true, false, false),
        "grouprank" => rank(false, false, true),
        "grouprank-oracle" => rank(true, false, true),
        other => panic!("unknown plan {other}"),
    }
}

/// Build the world, print its fit and size, simulate every plan, and report.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let trace = Trace::load(&args.trace)?;
    let (mut model, _, work) = fit(&trace);
    if args.linear_ps {
        // Exclusive-core equivalent: per-job rate = speed at any concurrency (slots cap it).
        for c in model.classes.values_mut() {
            c.k_sat = usize::MAX;
            c.alpha = 1.0;
        }
    }
    for cs in &args.class_speed {
        let (class, speed) = cs.split_once('=').expect("--class-speed class=speed");
        model
            .classes
            .get_mut(class)
            .unwrap_or_else(|| panic!("no class {class} in the model"))
            .speed = speed.parse()?;
    }
    let census = Census::load(&args.census)?;
    eprintln!("[whole] census: {} rows", census.rows.len());

    println!(
        "## Profile rule against the census\n\n| cap (entries) | rows reproduced |\n|---|---|"
    );
    let mut best = (0, 0);
    for len in 1..=6 {
        let (ok, n) = census.profile_agreement(len);
        println!(
            "| {len} | {ok}/{n} ({:.2}%) |",
            100.0 * ok as f64 / n.max(1) as f64
        );
        if ok > best.1 {
            best = (len, ok);
        }
    }
    let max_profile_len = args.max_profile_len.unwrap_or(best.0);
    println!("\nusing cap {max_profile_len}");

    let fleet = match &args.fleet {
        Some(f) => Fleet {
            groups: f
                .split(',')
                .map(|g| {
                    let p: Vec<&str> = g.split(':').collect();
                    (
                        p[0].to_string(),
                        p[1].parse().unwrap(),
                        p[2].parse().unwrap(),
                    )
                })
                .collect(),
        },
        None => {
            let mut groups: Vec<(String, usize, usize)> = Vec::new();
            for w in &trace.workers {
                match groups.iter_mut().find(|g| g.0 == w.class && g.2 == w.slots) {
                    Some(g) => g.1 += 1,
                    None => groups.push((w.class.clone(), 1, w.slots)),
                }
            }
            Fleet { groups }
        }
    };
    let world = World::build(
        WholeConfig {
            max_n: args.max_n,
            max_s: args.max_s,
            max_profile_len,
            min_work: args.min_work,
            noise_seed: args.noise_seed,
        },
        &census,
        (&trace, &work),
    );
    let c = &world.cost;
    println!(
        "\n## Cost model (H200-seconds)\n\nacross bidegrees (census, {} rows, R^2 {:.3}): ln wall \
         = {:.3} + {:.3} ln tmd + {:.3} ln nd + {:.3} ln(sigs+1) + {:.3} live + file \
         effect\nlevel vs {} on {} trace bidegrees: {:.3} (x{:.1}), per-bidegree sd \
         {:.3}\nzero-step share {:.4}\nwithin a bidegree ({} signatures, R^2 {:.3}): weight = \
         tmd(t-deg)^{:.3} deg^{:.3}, per-signature sd {:.3}",
        c.n[0],
        c.r2_across,
        c.across[0],
        c.across[1],
        c.across[2],
        c.across[3],
        c.across[4],
        census.sources[c.reference],
        c.n[1],
        c.level,
        c.level.exp(),
        c.sd_bidegree,
        c.zero_share,
        c.n[2],
        c.r2_within,
        c.within[0],
        c.within[1],
        c.sd_signature
    );
    let summary = world.summary();
    println!(
        "\n## The whole-run DAG (n <= {}, s <= {})\n\n{} bidegrees ({} from the census, {} live); \
         {} signature tasks ({} template nodes); work: zero steps {}, signatures {} (H200-time)",
        args.max_n,
        args.max_s,
        summary.bidegrees,
        summary.from_census,
        summary.live,
        summary.signature_tasks,
        summary.template_nodes,
        hours(summary.zero_work),
        hours(summary.signature_work)
    );
    println!("\n| profile | bidegrees | signature tasks |\n|---|---|---|");
    for (p, b, t) in &summary.per_profile {
        println!("| {p} | {b} | {t} |");
    }
    let (cp, cap) = world.bounds(&fleet, &model);
    println!(
        "\nfleet {:?}\nlower bounds: critical path {} (unlimited workers at the fastest speed), \
         capacity {}",
        fleet.groups,
        hours(cp),
        hours(cap)
    );
    println!("(seconds: critical path {cp:.6}, capacity {cap:.6})");
    if let Some(dir) = &args.export_dslab {
        std::fs::create_dir_all(dir)?;
        let (dag, system) = world.export_dslab(&fleet, &model, true);
        std::fs::write(dir.join("dag.yaml"), dag)?;
        std::fs::write(dir.join("system.yaml"), system)?;
        let (dag, _) = world.export_dslab(&fleet, &model, false);
        std::fs::write(dir.join("dag_est.yaml"), dag)?;
        eprintln!("[whole] exported to {}", dir.display());
        return Ok(());
    }

    let plans: Vec<(Plan, Placement)> = args
        .plans
        .iter()
        .map(|p| {
            let mut parts = p.split('+');
            let plan = plan_named(parts.next().unwrap(), &args);
            let mut speed = SpeedPlan::default();
            speed.config.timing = args.timing;
            let mut pin = Pin::None;
            let mut group_key = GroupKey::Arrival;
            let mut age_limit = None;
            for part in parts {
                match part {
                    "fast" | "eft0" => speed.fast = true,
                    "eft" => {
                        speed.fast = true;
                        speed.config.defer = Some(Defer {
                            max_wait: Duration::from_secs_f64(args.max_defer),
                            min_gain: args.min_gain,
                        });
                    }
                    "learn" => speed.config.timing = Timing::learned(),
                    "age" => age_limit = Some(Duration::from_secs_f64(args.age_limit)),
                    "smajor" => group_key = GroupKey::SMajor,
                    "tmajor" => group_key = GroupKey::TMajor,
                    "stem" => group_key = GroupKey::StemMajor,
                    "fastonly" => pin = Pin::All,
                    "cpop" => pin = Pin::Critical,
                    other => panic!("unknown placement suffix +{other}"),
                }
            }
            (
                plan,
                Placement {
                    speed,
                    pin,
                    rank_epsilon: args.rank_epsilon,
                    max_open: args.max_open,
                    group_key,
                    age_limit,
                },
            )
        })
        .collect();
    let results: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = plans
            .iter()
            .map(|p| {
                let (world, fleet, model) = (&world, &fleet, &model);
                s.spawn(move || simulate(world, fleet, model, &p.0, &p.1))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!(
        "\n## Plans\n\n| plan | makespan | vs bound | tasks | slot util | bidegree latency p50 | \
         p90 | max | peak open | peak materialised nodes | dispatch mean/max | sim time \
         |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
    );
    let bound = cp.max(cap);
    for m in &results {
        println!(
            "| {} | {:.1} h | {:.2}x | {} | {:.1}% | {} | {} | {} | {} | {} | {:.0}/{:.0} us | \
             {:.0} s |",
            m.plan,
            m.makespan_h,
            m.makespan_h * 3600.0 / bound,
            m.tasks,
            100.0 * m.slot_util,
            hours(m.bidegree_latency.p50),
            hours(m.bidegree_latency.p90),
            hours(m.bidegree_latency.max),
            m.peak_open,
            m.peak_dag_nodes,
            m.dispatch_us.mean,
            m.dispatch_us.max,
            m.sim_s
        );
    }
    if let Some(path) = &args.json {
        let out = serde_json::json!({
            "config": world.config, "cost": world.cost, "summary": summary, "fleet": fleet,
            "bounds_h": { "critical_path": cp / 3600.0, "capacity": cap / 3600.0 }, "results": results,
        });
        std::fs::write(path, serde_json::to_string_pretty(&out)?)?;
    }
    Ok(())
}
