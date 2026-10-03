//! Device-aware admission on a synthetic small-card scenario.

use clap::Parser;
use whelm_sim::device::{DeviceArm, DeviceScenario, simulate_device};

/// Compare host-only and device-aware admission when over-subscribing a card's launch pool slows
/// every job on it.
#[derive(Parser, Debug)]
#[command(about = "Device-aware admission on a synthetic small-card scenario")]
struct Args {
    /// Workers.
    #[arg(long, default_value_t = 14)]
    workers: usize,
    /// Jobs (all ready at the start).
    #[arg(long, default_value_t = 20_000)]
    jobs: usize,
    /// Launch-pool capacity per worker, GB.
    #[arg(long, default_value_t = 19.5)]
    cap_gb: f64,
    /// Median device demand per job, GB.
    #[arg(long, default_value_t = 2.4)]
    demand_gb: f64,
    /// Over-subscription penalties to try (default: none, and the one matching the live drop).
    #[arg(long, value_delimiter = ',', default_values_t = vec![0.0, (19.5f64 / 8.0).log2()])]
    gamma: Vec<f64>,
    /// Seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Write results as JSON here.
    #[arg(long)]
    json: Option<std::path::PathBuf>,
}

/// Run every arm under every penalty and print a table.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    let arms = [
        DeviceArm::HostOnly,
        DeviceArm::Count {
            quantile: 0.9,
            scale: 1.0,
        },
        DeviceArm::Count {
            quantile: 0.75,
            scale: 1.0,
        },
        DeviceArm::Count {
            quantile: 0.5,
            scale: 1.0,
        },
        DeviceArm::Count {
            quantile: 0.9,
            scale: 1.5,
        },
        DeviceArm::Sum { error_sd: 0.0 },
        DeviceArm::Sum { error_sd: 0.3 },
    ];
    let mut out = Vec::new();
    println!(
        "| gamma | admission | makespan | work/h | jobs running per worker | pool over-subscribed \
         |\n|---|---|---|---|---|---|"
    );
    for &gamma in &a.gamma {
        let sc = DeviceScenario {
            workers: a.workers,
            jobs: a.jobs,
            cap_gb: a.cap_gb,
            demand_gb: a.demand_gb,
            gamma,
            seed: a.seed,
            ..DeviceScenario::default()
        };
        let base = simulate_device(&sc, DeviceArm::HostOnly).makespan_h;
        for arm in arms {
            let m = simulate_device(&sc, arm);
            println!(
                "| {gamma:.2} | {} | {:.1} h ({:+.1}%) | {:.0} | {:.1} | {:.1}% |",
                m.arm,
                m.makespan_h,
                100.0 * (m.makespan_h / base - 1.0),
                m.work_per_h,
                m.mean_running,
                100.0 * m.oversubscribed
            );
            out.push((gamma, m));
        }
    }
    if let Some(p) = &a.json {
        std::fs::write(p, serde_json::to_string_pretty(&out)?)?;
    }
    Ok(())
}
