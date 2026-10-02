//! A synthetic device-memory scenario: small cards whose launch pool makes over-subscribed jobs
//! wait, with and without device-aware admission.

use std::{cmp::Ordering, collections::BinaryHeap};

use serde::Serialize;

use crate::{Config, JobId, JobSpec, Policy, Resources, Scheduler, WorkerId, WorkerState};

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
    /// Over-subscription penalty: with total device demand `S > C` running, each job runs at
    /// `(C / S)^(1 + gamma)` of its speed (0: the pool only serialises launches; > 0: contention
    /// wastes capacity too).
    pub gamma: f64,
    /// Seed of the job draws.
    pub seed: u64,
}

impl Default for DeviceScenario {
    /// An L40S fleet as in job 40506688: 14 workers x 16 slots, a 19.5 GB launch pool, jobs
    /// needing about 2.4 GB each (8 fit), and the penalty that turns 16 jobs into the observed
    /// 0.41x of 8 jobs' throughput (`2^-gamma = 8 / 19.5`).
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
    /// The per-worker count form: the worker reports its pool and a learned per-task demand, this
    /// quantile of the jobs' demands (times `scale`, to price an over- or underestimate).
    Count {
        /// Quantile of the demand distribution reported as the device `per_task`.
        quantile: f64,
        /// Multiplier on it.
        scale: f64,
    },
    /// The per-job sum form: each job carries a device estimate, its true demand times a
    /// log-normal error of this spread.
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

/// A completion event (earliest first; the version invalidates stale ones).
struct Ev(f64, usize, u64);

impl PartialEq for Ev {
    /// Equal when [`Ord`] says so.
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

impl Eq for Ev {}

impl PartialOrd for Ev {
    /// The total order of [`Ord`].
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Ev {
    /// Reversed for a min-heap.
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// A worker: its running jobs `(job, remaining work, device demand)`.
#[derive(Default)]
struct Wk {
    running: Vec<(JobId, f64, f64)>,
    last: f64,
    version: u64,
    busy: f64,
    over: f64,
}

/// Run one arm of the scenario through the default [`Scheduler`] with the production admission
/// rule.
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
        p.worker_update(
            WorkerState {
                per_task: Resources::ZERO.with_dev((per_task * 1e9).round() as u64),
                ..WorkerState::new(
                    w as WorkerId,
                    "small",
                    sc.slots,
                    Resources::mem_gb(1e6).with_dev_gb(cap),
                )
            },
            0.0,
        );
    }
    for (j, &d) in demand.iter().enumerate() {
        let est = match arm {
            DeviceArm::Sum { error_sd } => {
                d * (error_sd * normal(sc.seed ^ 0xABCD, j as u64)).exp()
            }
            _ => 0.0,
        };
        p.submit(
            JobSpec::new(j as JobId, Resources::mem(1).with_dev_gb(est), 0),
            0.0,
        );
    }
    let mut ws: Vec<Wk> = (0..sc.workers).map(|_| Wk::default()).collect();
    let mut heap = BinaryHeap::new();
    let rate = |w: &Wk| {
        let total: f64 = w.running.iter().map(|x| x.2).sum();
        if total > sc.cap_gb {
            (sc.cap_gb / total).powf(1.0 + sc.gamma)
        } else {
            1.0
        }
    };
    let advance = |w: &mut Wk, now: f64| {
        let dt = now - w.last;
        if dt > 0.0 && !w.running.is_empty() {
            let r = rate(w);
            for x in &mut w.running {
                x.1 -= r * dt;
            }
            w.busy += w.running.len() as f64 * dt;
            if r < 1.0 {
                w.over += dt;
            }
        }
        w.last = now;
    };
    let schedule = |w: &mut Wk, i: usize, now: f64, heap: &mut BinaryHeap<Ev>| {
        w.version += 1;
        if let Some(min) = w.running.iter().map(|x| x.1).reduce(f64::min) {
            heap.push(Ev(now + min.max(0.0) / rate(w), i, w.version));
        }
    };
    let mut now = 0.0;
    let mut done = 0;
    let place = |p: &mut Scheduler, ws: &mut Vec<Wk>, heap: &mut BinaryHeap<Ev>, now: f64| {
        let out = p.dispatch(now);
        let mut touched: Vec<usize> = Vec::new();
        for (j, w) in out {
            let w = w as usize;
            if !touched.contains(&w) {
                advance(&mut ws[w], now);
                touched.push(w);
            }
            ws[w]
                .running
                .push((j, work[j as usize], demand[j as usize]));
        }
        for w in touched {
            schedule(&mut ws[w], w, now, heap);
        }
    };
    place(&mut p, &mut ws, &mut heap, now);
    while let Some(Ev(t, w, v)) = heap.pop() {
        if v != ws[w].version {
            continue;
        }
        now = t;
        advance(&mut ws[w], now);
        let finished: Vec<JobId> = ws[w]
            .running
            .iter()
            .filter(|x| x.1 <= 1e-9)
            .map(|x| x.0)
            .collect();
        ws[w].running.retain(|x| x.1 > 1e-9);
        for j in finished {
            p.completed(j, now);
            done += 1;
        }
        schedule(&mut ws[w], w, now, &mut heap);
        place(&mut p, &mut ws, &mut heap, now);
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
        mean_running: ws.iter().map(|w| w.busy).sum::<f64>() / (span * sc.workers as f64),
        oversubscribed: ws.iter().map(|w| w.over).sum::<f64>() / (span * sc.workers as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With the observed penalty, device-aware admission beats host-only admission by a wide
    /// margin; without a penalty, exact per-job demands cost nothing while a pessimistic
    /// per-task demand (the 90th percentile) leaves capacity idle.
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
