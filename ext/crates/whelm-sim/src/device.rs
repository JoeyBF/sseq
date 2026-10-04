//! A synthetic device-memory scenario: small cards whose over-subscribed launch pool slows jobs.

use std::time::Duration;

use serde::Serialize;
use whelm::{
    Config, DEVICE_MEMORY, Input, JobId, JobSpec, MEMORY, Output, Policy, Resources, SLOTS,
    Scheduler, Time, WorkerState, gb,
};

use crate::engine::{PsWorker, Queue, Run};

/// The scenario.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceScenario {
    /// Workers (all alike).
    pub workers: usize,
    /// Slots per worker.
    pub slots: usize,
    /// Device launch-pool capacity per worker, GB.
    pub cap_gb: f64,
    /// Median device demand of a job, GB.
    pub demand_gb: f64,
    /// Log-normal spread of the device demand between jobs.
    pub demand_sd: f64,
    /// Jobs, all ready at the start (a saturated frontier).
    pub jobs: usize,
    /// Median work of a job, seconds alone.
    pub work_s: f64,
    /// Log-normal spread of the work.
    pub work_sd: f64,
    /// Over-subscription penalty.
    ///
    /// With total device demand `S > C` running, each job runs at `(C / S)^(1 + gamma)` of its
    /// speed (0: the pool only serialises launches; > 0: contention wastes capacity too).
    pub gamma: f64,
    /// Seed of the job draws.
    pub seed: u64,
}

impl Default for DeviceScenario {
    /// An L40S fleet as in job 40506688.
    ///
    /// `gamma` reproduces the throughput drop measured there between running only the jobs that
    /// fit the pool and filling every slot (see "Device memory" in RESULTS.md).
    fn default() -> Self {
        Self {
            workers: 14,
            slots: 16,
            cap_gb: 19.5,
            demand_gb: 2.4,
            demand_sd: 0.3,
            jobs: 20_000,
            work_s: 600.0,
            work_sd: 0.6,
            gamma: (19.5f64 / 8.0).log2(),
            seed: 1,
        }
    }
}

/// How the coordinator admits on the device.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum DeviceArm {
    /// Host memory only: the device capacity is not reported.
    HostOnly,
    /// The per-worker count form: the worker reports its pool and a learned per-task demand.
    ///
    /// That demand is this quantile of the jobs' demands, times `scale` to price an over- or
    /// underestimate.
    Count {
        /// Quantile of the demand distribution reported as the device `per_task`.
        quantile: f64,
        /// Multiplier on it.
        scale: f64,
    },
    /// The per-job sum form: each job carries a device estimate.
    ///
    /// The estimate is its true demand times a log-normal error of this spread.
    Sum {
        /// Spread of the estimate's error.
        error_sd: f64,
    },
}

/// One arm's outcome.
#[derive(Clone, Debug, Serialize)]
pub struct DeviceMetrics {
    /// The arm.
    pub arm: String,
    /// Time to finish every job, hours.
    pub makespan_h: f64,
    /// Work per hour (seconds of work alone per hour of wall time).
    pub work_per_h: f64,
    /// Mean jobs running per worker.
    pub mean_running: f64,
    /// Fraction of worker-time with the pool over-subscribed.
    pub oversubscribed: f64,
}

/// A deterministic standard normal draw from a seed and an index.
fn normal(seed: u64, i: u64) -> f64 {
    let mix = |mut z: u64| {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let a = mix(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_mul(2).wrapping_add(1));
    let b = mix(a ^ 0xD1B5_4A32_D192_ED03);
    let u1 = ((a >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
    let u2 = (b >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A worker: its running jobs, and the time its pool spent over-subscribed.
#[derive(Default)]
struct Wk {
    ps: PsWorker,
    over: f64,
}

/// Run one arm of the scenario through the default [`Scheduler`] (production admission rule).
pub fn simulate_device(sc: &DeviceScenario, arm: DeviceArm) -> DeviceMetrics {
    let n = sc.jobs;
    let demand: Vec<f64> = (0..n as u64)
        .map(|i| sc.demand_gb * (sc.demand_sd * normal(sc.seed, 2 * i)).exp())
        .collect();
    let work: Vec<f64> = (0..n as u64)
        .map(|i| sc.work_s * (sc.work_sd * normal(sc.seed, 2 * i + 1)).exp())
        .collect();
    let (cap, per_task) = match arm {
        DeviceArm::HostOnly => (0.0, 0.0),
        DeviceArm::Count { quantile, scale } => {
            let mut d = demand.clone();
            d.sort_by(f64::total_cmp);
            let q = d[((quantile * n as f64) as usize).min(n - 1)];
            (sc.cap_gb, q * scale)
        }
        DeviceArm::Sum { .. } => (sc.cap_gb, 0.0),
    };
    let mut p = Scheduler::new(Config::default());
    for w in 0..sc.workers {
        let state = WorkerState {
            id: w as u64,
            class: "small".into(),
            capacity: Resources::new()
                .with(MEMORY, gb(1e6))
                .with(DEVICE_MEMORY, gb(cap))
                .with(SLOTS, sc.slots as u64),
            per_task: Resources::new().with(DEVICE_MEMORY, (per_task * 1e9).round() as u64),
            ..Default::default()
        };
        p.handle(Input::Worker(state), Time::ORIGIN);
    }
    for (j, &d) in demand.iter().enumerate() {
        let est = match arm {
            DeviceArm::Sum { error_sd } => {
                d * (error_sd * normal(sc.seed ^ 0xABCD, j as u64)).exp()
            }
            _ => 0.0,
        };
        let spec = JobSpec {
            id: j as JobId,
            demand: Resources::new()
                .with(MEMORY, 1)
                .with(DEVICE_MEMORY, gb(est)),
            ..Default::default()
        };
        p.handle(Input::Submit(spec), Time::ORIGIN);
    }
    let mut ws: Vec<Wk> = (0..sc.workers).map(|_| Wk::default()).collect();
    // Completion events `(worker, version)`, ties broken by worker.
    let mut queue: Queue<(usize, u64)> = Queue::new();
    let rate = |running: &[Run]| {
        let total: f64 = running.iter().map(|x| demand[x.job as usize]).sum();
        if total > sc.cap_gb {
            (sc.cap_gb / total).powf(1.0 + sc.gamma)
        } else {
            1.0
        }
    };
    let advance = |w: &mut Wk, now: f64| {
        if let Some((dt, r)) = w.ps.advance(now, rate)
            && r < 1.0
        {
            w.over += dt;
        }
    };
    let schedule = |w: &mut Wk, i: usize, now: f64, queue: &mut Queue<(usize, u64)>| {
        if let Some((at, v)) = w.ps.next_completion(now, rate) {
            queue.push_tied(at, i as u64, (i, v));
        }
    };
    let mut now = 0.0;
    let mut done = 0;
    let mut finished = vec![false; n];
    let place = |p: &mut Scheduler, ws: &mut Vec<Wk>, queue: &mut Queue<(usize, u64)>, now: f64| {
        let mut touched: Vec<usize> = Vec::new();
        for o in p.poll(Time(Duration::from_secs_f64(now))) {
            let (job, attempt, w, start) = match o {
                Output::Start {
                    job,
                    attempt,
                    worker,
                } => (job, attempt, worker as usize, true),
                Output::Stop {
                    job,
                    attempt,
                    worker,
                } => (job, attempt, worker as usize, false),
                Output::GaveUp(g) => unreachable!("job {} failed, but no attempt fails", g.job),
                Output::Rejected { job, reason } => unreachable!("job {job} rejected: {reason}"),
                _ => continue,
            };
            if !touched.contains(&w) {
                advance(&mut ws[w], now);
                touched.push(w);
            }
            if start {
                ws[w].ps.start(job, attempt, work[job as usize]);
            } else {
                ws[w].ps.stop(job, attempt);
            }
        }
        for w in touched {
            schedule(&mut ws[w], w, now, queue);
        }
    };
    place(&mut p, &mut ws, &mut queue, now);
    while let Some((t, (w, v))) = queue.pop() {
        if !ws[w].ps.is_current(v) {
            continue;
        }
        now = t;
        advance(&mut ws[w], now);
        for r in ws[w].ps.finish(|left, _| left <= 1e-9) {
            // A losing attempt finishing at the same time as the winner is not a second job.
            if std::mem::replace(&mut finished[r.job as usize], true) {
                continue;
            }
            p.handle(
                Input::Done {
                    job: r.job,
                    attempt: r.attempt,
                },
                Time(Duration::from_secs_f64(now)),
            );
            done += 1;
        }
        schedule(&mut ws[w], w, now, &mut queue);
        place(&mut p, &mut ws, &mut queue, now);
    }
    assert_eq!(done, n, "every job finishes");
    let span = now.max(1e-9);
    let total: f64 = work.iter().sum();
    DeviceMetrics {
        arm: match arm {
            DeviceArm::HostOnly => "host only".into(),
            DeviceArm::Count { quantile, scale } => {
                format!("count form, p{:.0} demand x{scale}", 100.0 * quantile)
            }
            DeviceArm::Sum { error_sd } => format!("sum form, estimate error sd {error_sd}"),
        },
        makespan_h: span / 3600.0,
        work_per_h: total / (span / 3600.0),
        mean_running: ws.iter().map(|w| w.ps.busy).sum::<f64>() / (span * sc.workers as f64),
        oversubscribed: ws.iter().map(|w| w.over).sum::<f64>() / (span * sc.workers as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Device-aware admission avoids the slowdown, at a price where over-subscription is harmless.
    ///
    /// With the observed penalty it beats host-only admission by a wide margin; without a penalty,
    /// exact per-job demands cost nothing while a pessimistic per-task demand (a high quantile)
    /// leaves capacity idle.
    #[test]
    fn device_aware_admission_avoids_the_slowdown() {
        let sc = DeviceScenario {
            workers: 3,
            jobs: 600,
            ..DeviceScenario::default()
        };
        let p90 = DeviceArm::Count {
            quantile: 0.9,
            scale: 1.0,
        };
        let exact = DeviceArm::Sum { error_sd: 0.0 };
        let host = simulate_device(&sc, DeviceArm::HostOnly);
        for arm in [p90, exact] {
            let m = simulate_device(&sc, arm);
            assert!(m.makespan_h < 0.7 * host.makespan_h, "{m:?} vs {host:?}");
            assert!(m.oversubscribed < 0.05, "{m:?}");
        }
        let flat = DeviceScenario { gamma: 0.0, ..sc };
        let host = simulate_device(&flat, DeviceArm::HostOnly);
        let m = simulate_device(&flat, exact);
        assert!(m.makespan_h < 1.05 * host.makespan_h, "{m:?} vs {host:?}");
        let m = simulate_device(&flat, p90);
        assert!(m.makespan_h > 1.2 * host.makespan_h, "{m:?} vs {host:?}");
    }
}
