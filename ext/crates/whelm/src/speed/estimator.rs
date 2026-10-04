//! Online estimates of worker speeds, [`SpeedEstimator`].

use std::{collections::BTreeMap, time::Duration};

use super::Learn;
#[cfg(doc)]
use super::Timing;
use crate::worker::WorkerId;

/// A running mean of `ln speed`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Stat {
    pub(super) mean: f64,
    pub(super) n: u32,
}

impl Stat {
    /// Add a sample: plain averaging, then an exponential moving average of weight `weight`.
    pub(super) fn add(&mut self, x: f64, weight: f64) {
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
/// [`WorkerState::speed`](crate::worker::WorkerState::speed), and what the scheduler learns related
/// speeds with ([`Timing`]). Deterministic.
///
/// Used on its own, the estimator turns completions into the speed a worker reports in its next
/// heartbeat; the scheduler then trusts it under the default [`Timing`].
///
/// ```
/// use std::time::Duration;
///
/// use whelm::{
///     prelude::*,
///     speed::{Learn, SpeedEstimator},
/// };
///
/// let mut e = SpeedEstimator::new(Learn::default());
/// for _ in 0..Learn::default().min_samples {
///     e.observe(
///         7,
///         "h100",
///         Duration::from_secs(30),
///         Duration::from_secs(10),
///         1.0,
///     );
/// }
/// let state = WorkerState {
///     id: 7,
///     class: "h100".into(),
///     capacity: Resources::new().with(MEMORY, gb(80.0)).with(SLOTS, 4),
///     speed: e.speed(7, "h100", 1.0),
///     ..Default::default()
/// };
/// assert!((state.speed - 3.0).abs() < 1e-9);
/// let heartbeat = Input::Worker(state);
/// # let _ = heartbeat;
/// ```
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

    /// Record a completion on `worker` (of `class`): `work` at speed 1 took `duration` while the
    /// worker ran `concurrency` jobs on average (this one included). Samples with a zero work or
    /// duration are ignored. Returns whether it was used.
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::speed::{Learn, SpeedEstimator};
    ///
    /// let mut e = SpeedEstimator::new(Learn::default());
    /// let ten = Duration::from_secs(10);
    /// assert!(e.observe(1, "cpu", ten, Duration::from_secs(5), 1.0));
    /// assert!(!e.observe(1, "cpu", ten, Duration::ZERO, 1.0));
    /// assert_eq!((e.class_samples("cpu"), e.worker_samples(1)), (1, 1));
    /// ```
    pub fn observe(
        &mut self,
        worker: WorkerId,
        class: &str,
        work: Duration,
        duration: Duration,
        concurrency: f64,
    ) -> bool {
        let Some(x) = self.sample(work, duration, concurrency) else {
            return false;
        };
        self.add(worker, class, x, x);
        true
    }

    /// The `ln speed` sample of a completion (see [`observe`](Self::observe)), corrected for
    /// concurrency; `None` if the work or duration is unusable.
    pub(super) fn sample(
        &self,
        work: Duration,
        duration: Duration,
        concurrency: f64,
    ) -> Option<f64> {
        if work.is_zero() || duration.is_zero() {
            return None;
        }
        let mut x = (work.as_secs_f64() / duration.as_secs_f64()).ln();
        if let Some(s) = self.cfg.sharing
            && concurrency.is_finite()
        {
            x += s.log_correction(concurrency);
        }
        Some(x)
    }

    /// Record the sample `class_x` for `class` and `worker_x` for `worker`: the same sample,
    /// unless the worker's has a job kind's deviation taken out ([`Timing::Unrelated`]).
    pub(super) fn add(&mut self, worker: WorkerId, class: &str, class_x: f64, worker_x: f64) {
        let w = self.cfg.weight;
        match self.classes.get_mut(class) {
            Some(c) => c.add(class_x, w),
            None => {
                let mut c = Stat::default();
                c.add(class_x, w);
                self.classes.insert(class.to_string(), c);
            }
        }
        self.workers
            .entry(worker)
            .or_default()
            .stat
            .add(worker_x, w);
    }

    /// The mean `ln speed` of a class's samples, if it has any, warmed up or not.
    pub(super) fn class_mean(&self, class: &str) -> Option<f64> {
        self.classes.get(class).map(|c| c.mean)
    }

    /// `ln` of a class's speed: learned once warmed up, else the prior.
    fn class_log(&self, class: &str, prior: f64) -> f64 {
        match self.classes.get(class) {
            Some(c) if c.n >= self.cfg.min_samples => c.mean,
            _ => prior.ln(),
        }
    }

    /// The raw estimate of `worker`'s speed, without hysteresis. `prior` is the speed the worker
    /// reports, which stands for its class's until the class has [`Learn::min_samples`].
    ///
    /// A worker never seen gets its class's estimate, and a class never seen keeps the prior:
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::speed::{Learn, SpeedEstimator};
    ///
    /// let mut e = SpeedEstimator::new(Learn::default());
    /// for _ in 0..Learn::default().min_samples {
    ///     e.observe(
    ///         1,
    ///         "gpu",
    ///         Duration::from_secs(10),
    ///         Duration::from_secs(5),
    ///         1.0,
    ///     );
    /// }
    /// assert!((e.estimate(1, "gpu", 1.0) - 2.0).abs() < 1e-9);
    /// assert!((e.estimate(2, "gpu", 1.0) - 2.0).abs() < 1e-9);
    /// assert_eq!(e.estimate(3, "cpu", 0.5), 0.5);
    /// ```
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
    /// when it leaves the hysteresis band around the last value handed out
    /// ([`Learn::resolution`]).
    ///
    /// One job three times slower than usual moves the estimate but not the speed:
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::speed::{Learn, SpeedEstimator};
    ///
    /// let mut e = SpeedEstimator::new(Learn::default());
    /// let one = Duration::from_secs(1);
    /// for _ in 0..100 {
    ///     e.observe(1, "cpu", one, one, 1.0);
    /// }
    /// let before = e.speed(1, "cpu", 1.0);
    /// e.observe(1, "cpu", one, Duration::from_secs(3), 1.0);
    /// assert!(e.estimate(1, "cpu", 1.0) < before);
    /// assert_eq!(e.speed(1, "cpu", 1.0), before);
    /// ```
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
pub(super) fn sane(x: f64) -> f64 {
    if x > 0.0 && x.is_finite() { x } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::SpeedEstimator;
    use crate::speed::{Learn, Sharing};

    /// `n` seconds.
    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    /// Noise-free samples recover each class's speed exactly.
    #[test]
    fn exact_recovery() {
        let mut e = SpeedEstimator::new(Learn::default());
        for i in 0..40 {
            e.observe(1 + i % 2, "a", secs(10), secs(5), 1.0);
            e.observe(10, "b", secs(10), secs(10), 1.0);
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
                e.observe(w, "h200", secs(100), secs(100), 4.0);
            }
            e.observe(6, "h200", Duration::from_millis(76_500), secs(100), 4.0);
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
            e.observe(1, "a", secs(1), secs(1), 1.0);
        }
        let before = e.speed(1, "a", 1.0);
        e.observe(1, "a", secs(1), secs(3), 1.0);
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
            e.observe(1, "a", secs(10), secs(40), 4.0);
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
            e.observe(1, "a", secs(10), secs(10), 4.0);
        }
        assert!((e.speed(1, "a", 1.0) - 1.0).abs() < 1e-9);
    }
}
