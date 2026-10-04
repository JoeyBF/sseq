//! The scheduler's side of a [`Timing`]: interned classes and kinds, and the speeds it learns.

use std::{
    collections::{BTreeMap, HashMap},
    time::Duration,
};

use super::{
    SpeedEstimator, Timing,
    estimator::{Stat, sane},
};
use crate::worker::WorkerId;

/// An interned worker class.
pub(crate) type ClassId = u32;

/// An interned job kind.
pub(crate) type KindId = u32;

/// Interned names: a dense id per distinct string, in order of first sight.
#[derive(Clone, Debug, Default)]
struct Names {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

impl Names {
    /// The id of `name`, interning it if new.
    fn id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), id);
        id
    }

    /// The name behind an id.
    fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }
}

/// A kind's speed on one class relative to the class's average ([`Timing::Unrelated`]).
#[derive(Clone, Copy, Debug)]
struct KindStat {
    /// The kind's own `ln speed` samples on the class.
    stat: Stat,
    /// `ln` of the factor last handed out (hysteresis).
    published: Option<f64>,
    /// `published.exp()`, or 1 before anything is published.
    factor: f64,
}

/// The scheduler's side of a [`Timing`]: speeds per worker and per (job kind, worker class), and
/// what they learn from completions.
#[derive(Clone, Debug)]
pub(crate) struct Speeds {
    timing: Timing,
    /// The related speeds' learner, when they are learned.
    estimator: Option<SpeedEstimator>,
    classes: Names,
    kinds: Names,
    /// [`Timing::Unrelated`]: every kind's factor on every class it has run on.
    per_kind: BTreeMap<(ClassId, KindId), KindStat>,
}

impl Speeds {
    /// No samples yet.
    pub(crate) fn new(timing: Timing) -> Self {
        Self {
            estimator: timing.learn().copied().map(SpeedEstimator::new),
            timing,
            classes: Names::default(),
            kinds: Names::default(),
            per_kind: BTreeMap::new(),
        }
    }

    /// Speeds within one step of this fraction of each other rank equal (0: no rounding).
    pub(crate) fn resolution(&self) -> f64 {
        self.timing.learn().map_or(0.0, |l| l.resolution)
    }

    /// The id of a worker class.
    pub(crate) fn class(&mut self, name: &str) -> ClassId {
        self.classes.id(name)
    }

    /// The id of a job kind, if the timing distinguishes kinds.
    pub(crate) fn kind(&mut self, name: Option<&str>) -> Option<KindId> {
        match self.timing {
            Timing::Unrelated { .. } => name.map(|n| self.kinds.id(n)),
            _ => None,
        }
    }

    /// The speed of worker `id` of `class` that reports `reported`, for a job of no particular
    /// kind: 1 for identical machines, else learned once there are enough samples, else as
    /// reported.
    pub(crate) fn worker_speed(&mut self, id: WorkerId, class: ClassId, reported: f64) -> f64 {
        match (&self.timing, self.estimator.as_mut()) {
            (Timing::Identical, _) => 1.0,
            (_, Some(e)) => e.speed(id, self.classes.name(class), reported),
            (_, None) => sane(reported),
        }
    }

    /// The factor a job of `kind` runs at on `class` relative to the worker's speed.
    pub(crate) fn factor(&self, kind: Option<KindId>, class: ClassId) -> f64 {
        kind.and_then(|k| self.per_kind.get(&(class, k)))
            .map_or(1.0, |s| s.factor)
    }

    /// Every class `kind` has a factor on, and that factor, by class id.
    pub(crate) fn kind_factors(&self, kind: KindId) -> Vec<(&str, f64)> {
        (self.per_kind.iter())
            .filter(|((_, k), _)| *k == kind)
            .map(|(&(c, _), s)| (self.classes.name(c), s.factor))
            .collect()
    }

    /// `ln` of `kind`'s factor on `class` from its samples, without hysteresis: its mean shrunk
    /// towards the class's.
    fn raw_log(&self, class: ClassId, kind: KindId) -> f64 {
        let (Some(e), Some(s), Timing::Unrelated { learn, kind_prior }) = (
            self.estimator.as_ref(),
            self.per_kind.get(&(class, kind)),
            self.timing,
        ) else {
            return 0.0;
        };
        let Some(c) = e.class_mean(self.classes.name(class)) else {
            return 0.0;
        };
        let n = f64::from(s.stat.n).min(1.0 / learn.weight.max(1e-9));
        n * (s.stat.mean - c) / (n + kind_prior.max(0.0))
    }

    /// Learn from a completion on `worker` of `class` of a job of `kind` (see
    /// [`SpeedEstimator::observe`]). Returns whether the sample was used; if so, every worker of
    /// the class needs its speed refreshed ([`worker_speed`](Self::worker_speed)), and every
    /// kind's factor on the class has been.
    pub(crate) fn observe(
        &mut self,
        worker: WorkerId,
        class: ClassId,
        kind: Option<KindId>,
        work: Duration,
        duration: Duration,
        concurrency: f64,
    ) -> bool {
        let Some(x) = (self.estimator.as_ref()).and_then(|e| e.sample(work, duration, concurrency))
        else {
            return false;
        };
        let (Timing::Unrelated { learn, .. }, Some(kind)) = (self.timing, kind) else {
            let e = self.estimator.as_mut().unwrap();
            e.add(worker, self.classes.name(class), x, x);
            return true;
        };
        // The worker learns the class's average job: this kind's deviation is taken out.
        let deviation = self.raw_log(class, kind);
        let e = self.estimator.as_mut().unwrap();
        e.add(worker, self.classes.name(class), x, x - deviation);
        (self.per_kind.entry((class, kind)))
            .or_insert(KindStat {
                stat: Stat::default(),
                published: None,
                factor: 1.0,
            })
            .stat
            .add(x, learn.weight);
        // The class's average moved, so every kind's factor on it did.
        let band = learn.resolution.max(0.0).ln_1p();
        let kinds: Vec<KindId> = (self.per_kind.range((class, 0)..=(class, KindId::MAX)))
            .map(|(&(_, k), _)| k)
            .collect();
        for k in kinds {
            let raw = self.raw_log(class, k);
            let s = self.per_kind.get_mut(&(class, k)).unwrap();
            match s.published {
                Some(p) if (raw - p).abs() <= band => {}
                _ => {
                    s.published = Some(raw);
                    s.factor = raw.exp();
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::Speeds;
    use crate::speed::{Learn, Timing};

    /// Identical machines report speed 1 whatever the worker says; related ones without learning
    /// trust it; neither distinguishes kinds.
    #[test]
    fn identical_and_reported() {
        let mut p = Speeds::new(Timing::Identical);
        let x = p.class("x");
        assert_eq!(p.worker_speed(1, x, 4.0), 1.0);
        assert_eq!(p.kind(Some("a")), None);
        let second = Duration::from_secs(1);
        assert!(!p.observe(1, x, None, second, second, 1.0));
        let mut q = Speeds::new(Timing::default());
        let x = q.class("x");
        assert_eq!(q.worker_speed(1, x, 4.0), 4.0);
        assert_eq!(q.kind(Some("a")), None);
    }

    /// A kind's factor is its speed on the class over the class's average, shrunk by
    /// `kind_prior`; a kind never seen on the class runs at the worker's speed.
    #[test]
    fn kind_factors() {
        let timing = Timing::Unrelated {
            learn: Learn {
                resolution: 0.0,
                ..Learn::default()
            },
            kind_prior: 5.0,
        };
        let mut s = Speeds::new(timing);
        let x = s.class("x");
        let (a, b, new) = (s.kind(Some("a")), s.kind(Some("b")), s.kind(Some("new")));
        // Kind a takes a quarter of b's time.
        let (ten, quarter) = (Duration::from_secs(10), Duration::from_millis(2500));
        // Kind a alone: it is the class's average, so its factor is 1 and the worker learns 4.
        for _ in 0..40 {
            s.observe(1, x, a, ten, quarter, 1.0);
        }
        assert_eq!(s.factor(a, x), 1.0);
        assert!((s.worker_speed(1, x, 1.0) - 4.0).abs() < 1e-9);
        // Kind b at speed 1 drags the class average down; a's factor rises above 1, b's is
        // below, and each sits between 1 and its unshrunk ratio to the class mean.
        for _ in 0..40 {
            s.observe(1, x, b, ten, ten, 1.0);
            s.observe(1, x, a, ten, quarter, 1.0);
        }
        let (fa, fb) = (s.factor(a, x), s.factor(b, x));
        assert!(fa > 1.0 && fb < 1.0, "{fa} {fb}");
        let n = 1.0 / Learn::default().weight;
        let shrink = n / (n + 5.0);
        assert!((fa.ln() / fb.ln() + 1.0).abs() < 0.2, "{fa} {fb}");
        assert!(fa.ln() / 2f64.ln() < shrink + 0.1, "{fa}");
        assert_eq!(s.factor(new, x), 1.0);
        assert_eq!(s.factor(None, x), 1.0);
        assert_eq!(s.kind_factors(a.unwrap()), vec![("x", fa)]);
    }
}
