//! Online worker-speed estimation from completion times.

use std::collections::BTreeMap;

use crate::WorkerId;

/// Online speed learning: each completed job with work `w` (seconds at speed 1) that ran `d`
/// seconds is a sample `ln(w / d)` of its worker's speed, averaged in log space (durations are
/// log-normal). Unlike StarPU's history models there is no outlier filter: with per-job noise of
/// sd 0.6 a "50% off the mean" filter would discard most samples.
///
/// A class's estimate replaces the reported speed once the class has `min_samples` samples. With
/// `per_worker`, each worker's estimate is its own mean shrunk towards its class's estimate, so
/// that a clock-capped card shows up as a slower member of its class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Learn {
    /// Weight of a new sample once warmed up (an exponential moving average; plain averaging
    /// until then). It also caps how many samples a worker's own mean counts for, `1 / weight`.
    pub weight: f64,
    /// Samples a class needs before its learned speed replaces the reported one.
    pub min_samples: u32,
    /// Learn each worker's speed (shrunk towards its class), not only each class's.
    pub per_worker: bool,
    /// How many samples the class estimate counts for in a worker's estimate: a worker's own
    /// mean of `n` samples gets weight `min(n, 1 / weight) / (that + worker_prior)`.
    pub worker_prior: f64,
    /// Hysteresis: a worker's published speed changes only when its estimate moves by more than
    /// this fraction; speed-ordered placement also treats speeds within one such step as equal,
    /// so load still breaks ties within a class. 0 disables both.
    pub resolution: f64,
    /// How a worker's per-job speed depends on its concurrency, to normalise samples taken at
    /// different loads. `None`: per-job speed does not depend on load (our fitted model up to the
    /// slot count).
    pub sharing: Option<Sharing>,
}

impl Default for Learn {
    /// A 5% moving average after 20 samples, per worker with the class worth 5 samples, 10%
    /// hysteresis, load-independent per-job speed.
    fn default() -> Self {
        Self {
            weight: 0.05,
            min_samples: 20,
            per_worker: true,
            worker_prior: 5.0,
            resolution: 0.1,
            sharing: None,
        }
    }
}

/// Processor sharing with saturation: a worker running `k` jobs delivers a total throughput
/// `speed * min(k, k_sat)^alpha`, split equally, so each job runs at
/// `speed * min(k, k_sat)^alpha / k`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sharing {
    /// Concurrency beyond which throughput stops growing.
    pub k_sat: f64,
    /// Growth exponent below saturation (1: linear, each job at full speed).
    pub alpha: f64,
}

impl Sharing {
    /// `ln` of the factor that turns an observed per-job rate at mean concurrency `k` into the
    /// worker's `speed`.
    fn log_correction(&self, k: f64) -> f64 {
        let k = k.max(1.0);
        k.ln() - self.alpha * k.min(self.k_sat.max(1.0)).ln()
    }
}

/// A running mean of `ln speed`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Stat {
    mean: f64,
    n: u32,
}

impl Stat {
    /// Add a sample: plain averaging, then an exponential moving average of weight `weight`.
    fn add(&mut self, x: f64, weight: f64) {
        let a = weight.max(1.0 / f64::from(self.n + 1));
        self.mean += a * (x - self.mean);
        self.n = self.n.saturating_add(1);
    }
}

/// One worker's state.
#[derive(Clone, Debug, Default, PartialEq)]
struct WorkerStat {
    stat: Stat,
    /// The speed last handed out (hysteresis).
    published: Option<f64>,
}

/// Online estimates of worker speeds (see [`Learn`]), usable on its own to set
/// [`WorkerState::speed`](crate::WorkerState::speed), and what the policies use internally when
/// [`SpeedConfig::learn`](crate::SpeedConfig::learn) is set. Deterministic.
#[derive(Clone, Debug)]
pub struct SpeedEstimator {
    cfg: Learn,
    classes: BTreeMap<String, Stat>,
    workers: BTreeMap<WorkerId, WorkerStat>,
}

impl SpeedEstimator {
    /// An estimator with no samples.
    pub fn new(cfg: Learn) -> Self {
        Self {
            cfg,
            classes: BTreeMap::new(),
            workers: BTreeMap::new(),
        }
    }

    /// Its configuration.
    pub fn config(&self) -> &Learn {
        &self.cfg
    }

    /// Record a completion on `worker` (of `class`): `work` seconds at speed 1 took `duration`
    /// seconds while the worker ran `concurrency` jobs on average (this one included). Samples
    /// with a non-positive or non-finite work or duration are ignored. Returns whether it was used.
    pub fn observe(
        &mut self,
        worker: WorkerId,
        class: &str,
        work: f64,
        duration: f64,
        concurrency: f64,
    ) -> bool {
        if !(work > 0.0 && duration > 0.0 && work.is_finite() && duration.is_finite()) {
            return false;
        }
        let mut x = (work / duration).ln();
        if let Some(s) = self.cfg.sharing
            && concurrency.is_finite()
        {
            x += s.log_correction(concurrency);
        }
        let w = self.cfg.weight;
        match self.classes.get_mut(class) {
            Some(c) => c.add(x, w),
            None => {
                let mut c = Stat::default();
                c.add(x, w);
                self.classes.insert(class.to_string(), c);
            }
        }
        self.workers.entry(worker).or_default().stat.add(x, w);
        true
    }

    /// `ln` of a class's speed: learned once warmed up, else the prior.
    fn class_log(&self, class: &str, prior: f64) -> f64 {
        match self.classes.get(class) {
            Some(c) if c.n >= self.cfg.min_samples => c.mean,
            _ => prior.ln(),
        }
    }

    /// The raw estimate of `worker`'s speed, without hysteresis.
    pub fn estimate(&self, worker: WorkerId, class: &str, prior: f64) -> f64 {
        let prior = sane(prior);
        let class_log = self.class_log(class, prior);
        let log = match self.workers.get(&worker) {
            Some(ws) if self.cfg.per_worker && ws.stat.n > 0 => {
                let n = f64::from(ws.stat.n).min(1.0 / self.cfg.weight.max(1e-9));
                let k = self.cfg.worker_prior.max(0.0);
                (n * ws.stat.mean + k * class_log) / (n + k)
            }
            _ => class_log,
        };
        log.exp()
    }

    /// The speed to use for `worker` (of `class`, reporting `prior`): the estimate, moved only
    /// when it leaves the hysteresis band around the last value handed out.
    pub fn speed(&mut self, worker: WorkerId, class: &str, prior: f64) -> f64 {
        let est = self.estimate(worker, class, prior);
        let band = self.cfg.resolution.max(0.0).ln_1p();
        let ws = self.workers.entry(worker).or_default();
        match ws.published {
            Some(p) if (est.ln() - p.ln()).abs() <= band => p,
            _ => {
                ws.published = Some(est);
                est
            }
        }
    }

    /// Samples recorded for a class.
    pub fn class_samples(&self, class: &str) -> u32 {
        self.classes.get(class).map_or(0, |c| c.n)
    }

    /// Samples recorded for a worker.
    pub fn worker_samples(&self, worker: WorkerId) -> u32 {
        self.workers.get(&worker).map_or(0, |w| w.stat.n)
    }
}

/// A reported speed, guarded against nonsense.
pub(crate) fn sane(x: f64) -> f64 {
    if x > 0.0 && x.is_finite() { x } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Noise-free samples recover each class's speed exactly.
    #[test]
    fn exact_recovery() {
        let mut e = SpeedEstimator::new(Learn::default());
        for i in 0..40 {
            e.observe(1 + i % 2, "a", 10.0, 5.0, 1.0);
            e.observe(10, "b", 10.0, 10.0, 1.0);
        }
        assert!((e.speed(1, "a", 1.0) - 2.0).abs() < 1e-9);
        assert!((e.speed(10, "b", 7.0) - 1.0).abs() < 1e-9);
        // A worker never seen gets its class's estimate.
        assert!((e.speed(3, "a", 1.0) - 2.0).abs() < 1e-9);
        // An unknown class keeps its prior.
        assert!((e.speed(4, "c", 3.0) - 3.0).abs() < 1e-9);
    }

    /// A slow member of a class is estimated slower than its peers, but not below its own mean.
    #[test]
    fn capped_card_shows_up() {
        let mut e = SpeedEstimator::new(Learn::default());
        for _ in 0..200 {
            for w in 1..=5 {
                e.observe(w, "h200", 100.0, 100.0, 4.0);
            }
            e.observe(6, "h200", 76.5, 100.0, 4.0);
        }
        let normal = e.speed(1, "h200", 1.0);
        let capped = e.speed(6, "h200", 1.0);
        assert!(capped < 0.85 * normal, "{capped} vs {normal}");
        assert!(capped > 0.765 * 0.99);
    }

    /// One 3x-slow job does not move a warmed-up worker's published speed.
    #[test]
    fn hysteresis_absorbs_outliers() {
        let mut e = SpeedEstimator::new(Learn::default());
        for _ in 0..100 {
            e.observe(1, "a", 1.0, 1.0, 1.0);
        }
        let before = e.speed(1, "a", 1.0);
        e.observe(1, "a", 1.0, 3.0, 1.0);
        assert_eq!(e.speed(1, "a", 1.0), before);
        assert!(e.estimate(1, "a", 1.0) < before);
    }

    /// With linear sharing up to 16, concurrency does not matter; with no gain from sharing, a
    /// job at concurrency 4 took 4x longer and the correction recovers the speed.
    #[test]
    fn concurrency_correction() {
        let flat = Learn {
            sharing: Some(Sharing {
                k_sat: 1.0,
                alpha: 1.0,
            }),
            resolution: 0.0,
            ..Learn::default()
        };
        let mut e = SpeedEstimator::new(flat);
        for _ in 0..30 {
            e.observe(1, "a", 10.0, 40.0, 4.0);
        }
        assert!((e.speed(1, "a", 1.0) - 1.0).abs() < 1e-9);
        let linear = Learn {
            sharing: Some(Sharing {
                k_sat: 16.0,
                alpha: 1.0,
            }),
            resolution: 0.0,
            ..Learn::default()
        };
        let mut e = SpeedEstimator::new(linear);
        for _ in 0..30 {
            e.observe(1, "a", 10.0, 10.0, 4.0);
        }
        assert!((e.speed(1, "a", 1.0) - 1.0).abs() < 1e-9);
    }
}
