//! Compare two dispatch plans on many small instances, typically and adversarially (PISA).

use std::path::PathBuf;

use clap::Parser;
use sched::{
    Defer, SlowGate, SpeedConfig, SpeedPolicy,
    sim::{
        model::fit,
        small::{
            GridParams, Kind, Order, SmallInstance, SmallPlan, SmallResult, grid, perturb,
            simulate_small,
        },
        trace::Trace,
        whole::{Census, Fleet, WholeConfig, World},
    },
};

/// Compare plan A against plan B on small scheduling instances.
///
/// ```text
/// sched-pisa --a rank-oracle+fast --b group+fast typical --samples 2000
/// sched-pisa --a rank-oracle+fast --b group+fast anneal --restarts 8 --iters 3000
/// sched-pisa --a ... --b ... --family replica --trace T --census C... typical
/// ```
#[derive(Parser, Debug)]
#[command(about = "Typical-case and adversarial comparison of two dispatch plans")]
struct Args {
    /// Plan A: group | rank | rank-oracle | grouprank | grouprank-oracle, with optional suffixes
    /// +fast, +eft (wait up to --max-defer), +eft<percent> (wait only for that much gain), +gate,
    /// +age<seconds>.
    #[arg(long)]
    a: String,
    /// Plan B, as plan A.
    #[arg(long)]
    b: String,
    /// Instance family: "grid" (mini-Nassau grids with random parameters) or "replica" (the real
    /// world's first bidegrees, from --trace and --census, with perturbed costs).
    #[arg(long, default_value = "grid")]
    family: String,
    /// Replica: the scheduling trace.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Replica: census CSVs.
    #[arg(long)]
    census: Vec<PathBuf>,
    /// Replica region and fleet.
    #[arg(long, default_value_t = 40)]
    max_n: i32,
    /// See `max_n`.
    #[arg(long, default_value_t = 8)]
    max_s: i32,
    /// See `max_n`.
    #[arg(long, default_value = "l40s:2:4,h200:1:4")]
    fleet: String,
    /// Voluntary-wait bound for +eft and +gate, seconds.
    #[arg(long, default_value_t = 1e9)]
    max_defer: f64,
    /// Replica: random cost perturbations applied to each sampled instance.
    #[arg(long, default_value_t = 20)]
    perturbations: u64,
    /// First random seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Write details as JSON here.
    #[arg(long)]
    json: Option<PathBuf>,
    /// What to do.
    #[command(subcommand)]
    mode: Mode,
}

/// What `sched-pisa` does.
#[derive(clap::Subcommand, Debug)]
enum Mode {
    /// Sample instances (grid: random parameters; replica: random cost perturbations) and report
    /// the distribution of ln(makespan A / makespan B), with a regression on features.
    Typical {
        /// Instances to sample.
        #[arg(long, default_value_t = 1000)]
        samples: u64,
    },
    /// Simulated annealing towards instances where A is worst relative to B, then minimise.
    Anneal {
        /// Independent restarts.
        #[arg(long, default_value_t = 8)]
        restarts: u64,
        /// Iterations per restart.
        #[arg(long, default_value_t = 2000)]
        iters: u64,
    },
}

/// Parse a plan name with suffixes.
fn plan(name: &str, max_defer: f64) -> SmallPlan {
    let mut parts = name.split('+');
    let order = match parts.next().unwrap() {
        "group" => Order::Group,
        "rank" => Order::Rank { oracle: false },
        "rank-oracle" => Order::Rank { oracle: true },
        "grouprank" => Order::GroupRank { oracle: false },
        "grouprank-oracle" => Order::GroupRank { oracle: true },
        other => panic!("unknown order {other}"),
    };
    let mut p = SmallPlan {
        order,
        age_limit: None,
        speed: SpeedConfig::default(),
    };
    for s in parts {
        match s {
            "fast" => p.speed.policy = SpeedPolicy::FastestFirst,
            "eft" => {
                p.speed.policy = SpeedPolicy::EarliestFinish(Some(Defer {
                    max_wait: max_defer,
                    min_gain: 0.0,
                }))
            }
            "gate" => {
                if p.speed.policy == SpeedPolicy::Oblivious {
                    p.speed.policy = SpeedPolicy::FastestFirst;
                }
                p.speed.slow_gate = Some(SlowGate {
                    factor: 1.0,
                    max_wait: max_defer,
                });
            }
            s if s.starts_with("age") => p.age_limit = Some(s[3..].parse().expect("+age<seconds>")),
            s if s.starts_with("eft") => {
                let pct: f64 = s[3..].parse().expect("+eft<percent>");
                p.speed.policy = SpeedPolicy::EarliestFinish(Some(Defer {
                    max_wait: max_defer,
                    min_gain: pct / 100.0,
                }))
            }
            other => panic!("unknown suffix +{other}"),
        }
    }
    p
}

/// One comparison: ln(A / B) and both results.
fn compare(inst: &SmallInstance, a: &SmallPlan, b: &SmallPlan) -> (f64, SmallResult, SmallResult) {
    let (ra, rb) = (simulate_small(inst, a), simulate_small(inst, b));
    ((ra.makespan / rb.makespan.max(1e-12)).ln(), ra, rb)
}

/// Features for the regression: name and value.
fn features(inst: &SmallInstance, ra: &SmallResult, rb: &SmallResult) -> Vec<(&'static str, f64)> {
    let est_err: f64 = {
        let v: Vec<f64> = inst
            .tasks
            .iter()
            .filter(|t| t.kind == Kind::Sig && t.est > 0.0 && t.work > 0.0)
            .map(|t| (t.work / t.est).ln())
            .collect();
        let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
    };
    let fast = inst.classes.iter().map(|c| c.speed).fold(0.0, f64::max);
    let fast_share: f64 = {
        let tot: f64 = inst
            .classes
            .iter()
            .map(|c| c.speed * (c.workers * c.slots) as f64)
            .sum();
        inst.classes
            .iter()
            .filter(|c| c.speed >= fast)
            .map(|c| c.speed * (c.workers * c.slots) as f64)
            .sum::<f64>()
            / tot
    };
    vec![
        ("contention(B)", rb.contention),
        (
            "ln span/capacity",
            (rb.d_fast / rb.w_over_p.max(1e-12)).ln(),
        ),
        ("slow-on-crit(A)-(B)", ra.slow_on_crit - rb.slow_on_crit),
        ("estimate error sd", est_err),
        ("ln fast speed", fast.ln()),
        ("fast capacity share", fast_share),
        ("idle-with-ready(A)", ra.idle_ready),
    ]
}

/// Nearest-rank quantile of a sorted slice.
fn q(v: &[f64], p: f64) -> f64 {
    v[((v.len() - 1) as f64 * p).round() as usize]
}

/// Make groups' walks trivial and drop walk edges while A stays at least 95% as bad as `e`.
fn minimise(inst: &SmallInstance, a: &SmallPlan, b: &SmallPlan, e: f64) -> SmallInstance {
    let mut cur = inst.clone();
    let groups: Vec<u32> = {
        let mut g: Vec<u32> = cur
            .tasks
            .iter()
            .filter(|t| t.kind == Kind::Sig)
            .map(|t| t.group)
            .collect();
        g.dedup();
        g
    };
    let keep = |x: &SmallInstance| compare(x, a, b).0 >= e + (0.95f64).ln();
    for g in groups {
        let mut x = cur.clone();
        for t in x
            .tasks
            .iter_mut()
            .filter(|t| t.group == g && t.kind == Kind::Sig)
        {
            t.kind = Kind::Join;
            t.work = 0.0;
            t.est = 0.0;
        }
        if keep(&x) {
            cur = x;
        }
    }
    for i in 0..cur.tasks.len() {
        let mut d = 0;
        while d < cur.tasks[i].deps.len() {
            if cur.tasks[i].kind == Kind::Sig && cur.tasks[i].deps.len() > 1 {
                let mut x = cur.clone();
                x.tasks[i].deps.remove(d);
                if keep(&x) {
                    cur = x;
                    continue;
                }
            }
            d += 1;
        }
    }
    cur
}

/// Load the replica instance from real data.
fn replica(args: &Args) -> Result<SmallInstance, Box<dyn std::error::Error>> {
    let trace = Trace::load(
        args.trace
            .as_ref()
            .ok_or("--family replica needs --trace")?,
    )?;
    let (model, _, work) = fit(&trace);
    let census = Census::load(&args.census)?;
    let world = World::build(
        WholeConfig {
            max_n: args.max_n,
            max_s: args.max_s,
            max_profile_len: 5,
            min_work: 0.05,
            noise_seed: 0,
        },
        &census,
        (&trace, &work),
    );
    let fleet = Fleet {
        groups: args
            .fleet
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
    };
    Ok(world.to_small(&fleet, &model))
}

/// Run the comparison and report.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let (a, b) = (plan(&args.a, args.max_defer), plan(&args.b, args.max_defer));
    let base = (args.family == "replica")
        .then(|| replica(&args))
        .transpose()?;
    if let Some(r) = &base {
        eprintln!(
            "[pisa] replica: {} tasks ({} jobs)",
            r.tasks.len(),
            r.jobs()
        );
    }
    // The k-th instance of the family.
    let instance = |k: u64| -> SmallInstance {
        match &base {
            None => grid(&GridParams::random(args.seed.wrapping_add(k))),
            Some(r) => {
                let mut x = r.clone();
                for i in 0..args.perturbations {
                    x = perturb(
                        &x,
                        args.seed.wrapping_mul(1_000_003).wrapping_add(k * 64 + i),
                        false,
                    );
                }
                x
            }
        }
    };
    let mut out = serde_json::json!({ "a": args.a, "b": args.b, "family": args.family });
    match args.mode {
        Mode::Typical { samples } => {
            let (mut lns, mut x, mut names) = (Vec::new(), Vec::new(), Vec::new());
            for k in 0..samples {
                let inst = instance(k);
                let (l, ra, rb) = compare(&inst, &a, &b);
                if samples == 1 {
                    println!("A {ra:?}\nB {rb:?}");
                }
                let f = features(&inst, &ra, &rb);
                names = f.iter().map(|p| p.0).collect();
                let mut row = vec![1.0];
                row.extend(f.iter().map(|p| p.1));
                x.push(row);
                lns.push(l);
            }
            let (beta, r2, sd) = sched::sim::whole::ols_pub(&x, &lns);
            let mut sorted = lns.clone();
            sorted.sort_by(f64::total_cmp);
            let mean = lns.iter().sum::<f64>() / lns.len() as f64;
            let worse = lns.iter().filter(|&&l| l > 1e-9).count() as f64 / lns.len() as f64;
            let better = lns.iter().filter(|&&l| l < -1e-9).count() as f64 / lns.len() as f64;
            println!(
                "## {} vs {} on {} {} instances\n\nln(A/B): mean {:+.4} (x{:.4}), p5 {:+.4}, p50 \
                 {:+.4}, p95 {:+.4}, max {:+.4}; A worse in {:.1}%, better in {:.1}%, tied {:.1}%",
                args.a,
                args.b,
                samples,
                args.family,
                mean,
                mean.exp(),
                q(&sorted, 0.05),
                q(&sorted, 0.5),
                q(&sorted, 0.95),
                q(&sorted, 1.0),
                100.0 * worse,
                100.0 * better,
                100.0 * (1.0 - worse - better)
            );
            println!(
                "\nregression of ln(A/B) (R^2 {r2:.3}, residual sd {sd:.4}):\n\n| feature | \
                 coefficient | mean |\n|---|---|---|"
            );
            println!("| intercept | {:+.4} | |", beta[0]);
            for (i, n) in names.iter().enumerate() {
                let m = x.iter().map(|r| r[i + 1]).sum::<f64>() / x.len() as f64;
                println!("| {n} | {:+.4} | {m:.4} |", beta[i + 1]);
            }
            out["ln_ratio"] = serde_json::json!(lns);
            out["features"] = serde_json::json!(names);
            out["beta"] = serde_json::json!(beta);
            out["r2"] = serde_json::json!(r2);
        }
        Mode::Anneal { restarts, iters } => {
            let mut best_all: Option<(f64, SmallInstance)> = None;
            println!(
                "## Annealing ln({} / {}) on {}\n\n| restart | start | best | jobs after \
                 minimising | minimised |\n|---|---|---|---|---|",
                args.a, args.b, args.family
            );
            for r in 0..restarts {
                let mut cur = instance(r);
                let mut e_cur = compare(&cur, &a, &b).0;
                let start = e_cur;
                let (mut best, mut e_best) = (cur.clone(), e_cur);
                for it in 0..iters {
                    let t = 0.05 * (5e-4f64 / 0.05).powf(it as f64 / iters.max(1) as f64);
                    let key = args.seed.wrapping_mul(0x9e37).wrapping_add(r << 32 | it);
                    let next = perturb(&cur, key, base.is_none());
                    let e = compare(&next, &a, &b).0;
                    let accept = e >= e_cur
                        || sched::sim::whole::uniform_pub(key ^ 0xdead) < ((e - e_cur) / t).exp();
                    if accept {
                        cur = next;
                        e_cur = e;
                        if e > e_best {
                            best = cur.clone();
                            e_best = e;
                        }
                    }
                }
                let small = minimise(&best, &a, &b, e_best);
                let e_small = compare(&small, &a, &b).0;
                println!(
                    "| {r} | x{:.4} | x{:.4} | {} of {} | x{:.4} |",
                    start.exp(),
                    e_best.exp(),
                    small.jobs(),
                    best.jobs(),
                    e_small.exp()
                );
                if best_all.as_ref().is_none_or(|(e, _)| e_best > *e) {
                    best_all = Some((e_best, small));
                }
            }
            if let Some((e, inst)) = best_all {
                let (_, ra, rb) = compare(&inst, &a, &b);
                println!(
                    "\nworst witness: x{:.4}; A {ra:?}\nB {rb:?}\nfeatures {:?}",
                    e.exp(),
                    features(&inst, &ra, &rb)
                );
                out["witness"] = serde_json::to_value(&inst)?;
            }
        }
    }
    if let Some(p) = &args.json {
        std::fs::write(p, serde_json::to_string(&out)?)?;
    }
    Ok(())
}
