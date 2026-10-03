//! Compare two dispatch plans on many small instances, typically and adversarially (PISA).

use std::{path::PathBuf, time::Duration};

use clap::Parser;
use whelm::{Defer, Speculate, SpeedConfig, Timing};
use whelm_sim::{
    exact::{self, Limits, Solution},
    heft,
    model::fit,
    plan::{SpeedPlan, timing_named},
    small::{
        GridParams, Kind, Order, SmallInstance, SmallPlan, SmallResult, grid, perturb,
        simulate_small, tiny,
    },
    trace::Trace,
    whole::{Census, Fleet, WholeConfig, World},
};

/// Compare plan A against plan B on small scheduling instances.
///
/// ```text
/// whelm-pisa --a rank-oracle+fast --b group+fast typical --samples 2000
/// whelm-pisa --a rank-oracle+fast --b group+fast anneal --restarts 8 --iters 3000
/// whelm-pisa --a ... --b ... --family replica --trace T --census C... typical
/// whelm-pisa --a heft-oracle+fast --b group+fast --family tiny typical --samples 200
/// ```
#[derive(Parser, Debug)]
#[command(about = "Typical-case and adversarial comparison of two dispatch plans")]
struct Args {
    /// Plan A: an order with optional suffixes.
    ///
    /// Orders: group | rank | rank-oracle | grouprank | grouprank-oracle | heft | heft-oracle |
    /// wspt. Suffixes: +fast, +eft (wait up to --max-defer), `+eft<percent>` (wait only for that
    /// much gain), +spec (a second attempt of a running job on an idle faster worker),
    /// `+age<seconds>`. heft[-oracle]: an offline HEFT schedule (on estimated or true costs) whose
    /// start order is every job's priority. wspt: Smith's rule, shortest estimated work first.
    #[arg(long)]
    a: String,
    /// Plan B, as plan A.
    #[arg(long)]
    b: String,
    /// Instance family: grid, tiny or replica.
    ///
    /// "grid": mini-Nassau grids with random parameters; "tiny": grids small enough to solve
    /// exactly; "replica": the real world's first bidegrees, from --trace and --census, with
    /// perturbed costs.
    #[arg(long, default_value = "grid")]
    family: String,
    /// Machine model of both plans: p, q, q-learn or r.
    ///
    /// p (identical), q (related, reported speeds), q-learn (related, learned, every worker
    /// reporting 1) or r (unrelated: learned per job kind and worker class).
    #[arg(long, default_value = "q", value_parser = timing_named)]
    timing: Timing,
    /// Solve every instance exactly and report each plan's gap to the optimum.
    ///
    /// Always on for --family tiny; beyond tiny sizes the search mostly stops at its limits.
    #[arg(long)]
    exact: bool,
    /// Exact search: node limit per instance.
    #[arg(long, default_value_t = Limits::default().nodes)]
    nodes: u64,
    /// Exact search: time limit per instance, seconds.
    #[arg(long, default_value_t = Limits::default().seconds)]
    time_limit: f64,
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
    /// Voluntary-wait bound for +eft, seconds.
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

/// What `whelm-pisa` does.
#[derive(clap::Subcommand, Debug)]
enum Mode {
    /// Sample instances and report the distribution of ln(makespan A / makespan B).
    ///
    /// Grid samples random parameters, replica random cost perturbations; a regression on features
    /// follows.
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
fn plan(name: &str, max_defer: f64, timing: Timing) -> SmallPlan {
    let mut parts = name.split('+');
    let order = match parts.next().unwrap() {
        "group" => Order::Group,
        "rank" => Order::Rank { oracle: false },
        "rank-oracle" => Order::Rank { oracle: true },
        "grouprank" => Order::GroupRank { oracle: false },
        "grouprank-oracle" => Order::GroupRank { oracle: true },
        "heft" => Order::Heft { oracle: false },
        "heft-oracle" => Order::Heft { oracle: true },
        "wspt" => Order::Wspt,
        other => panic!("unknown order {other}"),
    };
    let mut p = SmallPlan {
        order,
        age_limit: None,
        speed: SpeedPlan {
            fast: false,
            config: SpeedConfig {
                timing,
                ..SpeedConfig::default()
            },
        },
    };
    for s in parts {
        match s {
            "fast" => p.speed.fast = true,
            "eft" => {
                p.speed.fast = true;
                p.speed.config.defer = Some(Defer {
                    max_wait: Duration::from_secs_f64(max_defer),
                    min_gain: 0.0,
                });
            }
            "spec" => {
                p.speed.fast = true;
                p.speed.config.speculate = Some(Speculate::default());
            }
            s if s.starts_with("age") => {
                p.age_limit = Some(Duration::from_secs_f64(
                    s[3..].parse().expect("+age<seconds>"),
                ))
            }
            s if s.starts_with("eft") => {
                let pct: f64 = s[3..].parse().expect("+eft<percent>");
                p.speed.fast = true;
                p.speed.config.defer = Some(Defer {
                    max_wait: Duration::from_secs_f64(max_defer),
                    min_gain: pct / 100.0,
                });
            }
            other => panic!("unknown suffix +{other}"),
        }
    }
    p
}

/// Each instance's gaps to the optimum, and the exact searches behind them.
#[derive(Default)]
struct Gaps {
    /// Per instance, makespan over the optimum, minus one, of `[A, B, HEFT]`.
    ///
    /// HEFT is its offline schedule on true costs.
    rows: Vec<[f64; 3]>,
    /// The exact searches.
    solutions: Vec<Solution>,
}

impl Gaps {
    /// Solve `inst` and record the gaps of A's and B's makespans and of HEFT's offline schedule.
    fn add(&mut self, inst: &SmallInstance, ra: &SmallResult, rb: &SmallResult, limits: Limits) {
        let sol = exact::solve(inst, limits);
        let opt = sol.makespan.max(1e-12);
        let h = heft::heft(inst, true).makespan;
        self.rows.push([
            ra.makespan / opt - 1.0,
            rb.makespan / opt - 1.0,
            h / opt - 1.0,
        ]);
        self.solutions.push(sol);
    }

    /// The report: a table of gaps and the search's effort.
    fn print(&self, a: &str, b: &str) {
        let n = self.rows.len();
        let proved = self.solutions.iter().filter(|s| s.optimal).count();
        println!(
            "\n## Gap to the optimum (exact branch and bound)\n\n{proved} of {n} instances proved \
             optimal{}\n\n| plan | mean gap | p50 | p90 | max | at the optimum \
             |\n|---|---|---|---|---|---|",
            if proved < n {
                "; the others are gaps to the best schedule found, which understate them"
            } else {
                ""
            }
        );
        for (c, name) in [a, b, "HEFT offline schedule (true costs)"]
            .iter()
            .enumerate()
        {
            let mut v: Vec<f64> = self.rows.iter().map(|r| r[c]).collect();
            v.sort_by(f64::total_cmp);
            let mean = v.iter().sum::<f64>() / n as f64;
            let hit = v.iter().filter(|&&g| g <= 1e-9).count();
            println!(
                "| {name} | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.1}% |",
                100.0 * mean,
                100.0 * q(&v, 0.5),
                100.0 * q(&v, 0.9),
                100.0 * q(&v, 1.0),
                100.0 * hit as f64 / n as f64
            );
        }
        let nodes: Vec<f64> = self.solutions.iter().map(|s| s.nodes as f64).collect();
        let secs: Vec<f64> = self.solutions.iter().map(|s| s.seconds).collect();
        println!(
            "\nsearch: nodes mean {:.0}, max {:.0}; seconds mean {:.4}, max {:.4}",
            nodes.iter().sum::<f64>() / n as f64,
            nodes.iter().copied().fold(0.0, f64::max),
            secs.iter().sum::<f64>() / n as f64,
            secs.iter().copied().fold(0.0, f64::max)
        );
    }
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

/// The fraction of a witness's makespan ratio A / B that minimising it must keep.
const MINIMISE_KEEP: f64 = 0.95;

/// Shrink a witness while A / B stays within [`MINIMISE_KEEP`] of `exp(e)`.
///
/// It makes groups' walks trivial and drops walk edges.
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
    let keep = |x: &SmallInstance| compare(x, a, b).0 >= e + MINIMISE_KEEP.ln();
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
    let (a, b) = (
        plan(&args.a, args.max_defer, args.timing),
        plan(&args.b, args.max_defer, args.timing),
    );
    let exact = args.exact || args.family == "tiny";
    let limits = Limits {
        nodes: args.nodes,
        seconds: args.time_limit,
    };
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
            None if args.family == "tiny" => tiny(args.seed.wrapping_add(k)),
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
            let mut gaps = Gaps::default();
            for k in 0..samples {
                let inst = instance(k);
                let (l, ra, rb) = compare(&inst, &a, &b);
                if exact {
                    gaps.add(&inst, &ra, &rb, limits);
                }
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
            let (beta, r2, sd) = whelm_sim::whole::ols_pub(&x, &lns);
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
            if exact {
                gaps.print(&args.a, &args.b);
                out["gaps"] = serde_json::json!(gaps.rows);
                out["exact"] = serde_json::json!(gaps.solutions);
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
                        || whelm_sim::whole::uniform_pub(key ^ 0xdead) < ((e - e_cur) / t).exp();
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
                if exact {
                    let mut gaps = Gaps::default();
                    gaps.add(&inst, &ra, &rb, limits);
                    gaps.print(&args.a, &args.b);
                }
                out["witness"] = serde_json::to_value(&inst)?;
            }
        }
    }
    if let Some(p) = &args.json {
        std::fs::write(p, serde_json::to_string(&out)?)?;
    }
    Ok(())
}
