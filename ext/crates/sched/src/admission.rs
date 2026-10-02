//! The admission rule: whether a worker accepts a job right now.

use crate::{DIMS, Resources, WorkerState};

/// A worker as an [`Admission`] rule sees it: its state plus the library's own bookkeeping.
#[derive(Clone, Copy, Debug)]
pub struct WorkerView<'a> {
    /// The worker's last reported state.
    pub state: &'a WorkerState,
    /// Jobs placed on the worker and not yet completed.
    pub running: usize,
    /// Sum of the demands of those jobs.
    pub placed: Resources,
}

impl WorkerView<'_> {
    /// Free execution slots.
    pub fn free_slots(&self) -> usize {
        self.state.slots.saturating_sub(self.running)
    }

    /// Whether dimension `d` of the worker's capacity is known, and so enforced.
    pub fn enforced(&self, d: usize) -> bool {
        self.state.budget[d] > 0
    }

    /// What a job of demand `demand` counts for here: at least the worker's
    /// [`per_task`](WorkerState::per_task) in every dimension.
    pub fn charge(&self, demand: &Resources) -> Resources {
        demand.max(self.state.per_task)
    }

    /// Usage as the production rule counts it, per dimension: `max(reported_used,
    /// reported_baseline + max(placed, running * per_task))`.
    ///
    /// The reported figure lags (heartbeats); the placed sum is exact but only an estimate of what
    /// the jobs use. Taking the maximum is conservative in both directions.
    pub fn used(&self) -> Resources {
        let floor = self.state.per_task.saturating_mul(self.running as u64);
        self.state
            .reported_used
            .max(self.state.reported_baseline + self.placed.max(floor))
    }

    /// Headroom per dimension, `budget - used`; negative when over-committed, `None` where the
    /// capacity is unknown.
    pub fn headroom(&self) -> [Option<i64>; DIMS] {
        let used = self.used();
        std::array::from_fn(|d| {
            self.enforced(d).then(|| {
                (self.state.budget[d] as i128 - used[d] as i128)
                    .clamp(i64::MIN as i128, i64::MAX as i128) as i64
            })
        })
    }

    /// The enforced dimensions in which a job of demand `demand` does not fit beside the jobs
    /// already here: `used + charge > budget`.
    pub fn short(&self, demand: &Resources) -> impl Iterator<Item = usize> + '_ {
        let (used, charge) = (self.used(), self.charge(demand));
        (0..DIMS).filter(move |&d| {
            self.enforced(d) && used[d].saturating_add(charge[d]) > self.state.budget[d]
        })
    }

    /// The fraction of capacity left after placing `demand`, in the bottleneck dimension: `min_d
    /// (budget - used - charge) / budget` over the enforced dimensions (one minus the dominant
    /// share, as in dominant-resource fairness), infinite when none is enforced.
    ///
    /// This is the one scalar the policies compare workers by: smallest for the tightest fit,
    /// largest (with a zero demand) for the most headroom.
    pub fn free_share(&self, demand: &Resources) -> f64 {
        let (used, charge) = (self.used(), self.charge(demand));
        (0..DIMS)
            .filter(|&d| self.enforced(d))
            .map(|d| {
                let cap = self.state.budget[d] as f64;
                (cap - used[d] as f64 - charge[d] as f64) / cap
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Whether a worker admits a job.
///
/// # Contract
///
/// Implementations must be **monotone in load**: if a job is refused by a worker, it stays refused
/// after more jobs are placed on that worker (with no completion or heartbeat in between). The
/// scheduler relies on this to guarantee the priority invariant within one `dispatch`.
pub trait Admission {
    /// Whether `w` admits a job with demand `demand`.
    fn admits(&self, demand: &Resources, w: &WorkerView) -> bool;

    /// An upper bound on admitted demands, used only to skip hopeless checks quickly: if
    /// `admits(d, w)` then `d.fits_within(bound(w))`. `None` means `w` admits nothing. The default
    /// is no bound.
    fn bound(&self, _w: &WorkerView) -> Option<Resources> {
        Some(Resources::MAX)
    }
}

/// The production admission rule, one inequality per resource dimension `d`:
///
/// ```text
/// admit(job on w) iff running < slots
///                 and ( running == 0      // escape hatch: a job alone always goes
///                       or for every d with budget[d] > 0:
///                            max(reported_used[d],
///                                reported_baseline[d] + max(placed[d], running * per_task[d]))
///                              + max(demand[d], per_task[d]) <= budget[d] )
/// ```
///
/// A zero capacity component is unknown and not enforced. With a device `per_task` alone (jobs
/// without device demands) the device inequality is the per-worker count `(running + 1) *
/// per_task <= cap`; with per-job device demands it is their sum.
///
/// `reported_baseline` must exclude the running jobs (a worker's `baseline_excl`: its rolling RSS
/// floor minus their estimates). A floor that contains them counts them twice, once in it and
/// once in `placed`, and keeps a busy worker a few GB short of its budget.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProductionAdmission;

impl Admission for ProductionAdmission {
    /// The production rule itself.
    fn admits(&self, demand: &Resources, w: &WorkerView) -> bool {
        w.running < w.state.slots && (w.running == 0 || w.short(demand).next().is_none())
    }

    /// The headroom left under the production rule (unbounded when the worker is empty, and in
    /// the dimensions not enforced).
    fn bound(&self, w: &WorkerView) -> Option<Resources> {
        if w.running >= w.state.slots {
            return None;
        }
        if w.running == 0 {
            return Some(Resources::MAX);
        }
        let headroom = w.headroom();
        let mut bound = Resources::MAX;
        for (d, h) in headroom.into_iter().enumerate() {
            if let Some(h) = h {
                // Every job counts for at least `per_task`, so less room than that admits nothing.
                if h < 0 || (h as u64) < w.state.per_task[d] {
                    return None;
                }
                bound[d] = h as u64;
            }
        }
        Some(bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEV, MEM};

    /// A worker state with the given capacity and reported usage.
    fn worker(slots: usize, budget: u64, used: u64, baseline: u64) -> WorkerState {
        WorkerState {
            reported_used: Resources::mem(used),
            reported_baseline: Resources::mem(baseline),
            ..WorkerState::new(1, "x", slots, Resources::mem(budget))
        }
    }

    /// The rule, its escape hatch and its bound, at the boundaries.
    #[test]
    fn production_rule() {
        let a = ProductionAdmission;
        let s = worker(2, 100, 30, 10);
        let view = |running, placed| WorkerView {
            state: &s,
            running,
            placed: Resources::mem(placed),
        };
        // Escape hatch: alone, anything goes, even beyond the budget.
        assert!(a.admits(&Resources::mem(1000), &view(0, 0)));
        // max(30, 10 + 50) + 40 = 100 <= 100.
        assert!(a.admits(&Resources::mem(40), &view(1, 50)));
        assert!(!a.admits(&Resources::mem(41), &view(1, 50)));
        // Reported usage dominates: max(30, 10 + 5) + 70 = 100.
        assert!(a.admits(&Resources::mem(70), &view(1, 5)));
        assert!(!a.admits(&Resources::mem(71), &view(1, 5)));
        // Slots full.
        assert!(!a.admits(&Resources::mem(0), &view(2, 0)));
        assert_eq!(a.bound(&view(2, 0)), None);
        assert_eq!(
            a.bound(&view(1, 50)),
            Some(Resources::mem(40).with_dev(u64::MAX))
        );
        assert_eq!(view(1, 50).headroom(), [Some(40), None]);
    }

    /// `baseline_excl` (the rolling RSS floor minus the estimates running) as `reported_baseline`
    /// gives `running < slots && (running == 0 || max(rss, baseline_excl + Σ placed) + demand <=
    /// budget)`: the floor no longer counts the running jobs twice.
    #[test]
    fn baseline_excl_removes_the_double_count() {
        // RSS 60 with 40 of estimates running; the floor (50) contains those jobs.
        let (rss, floor, placed, budget) = (60, 50, 40, 100);
        let view = |s| WorkerView {
            state: s,
            running: 4,
            placed: Resources::mem(placed),
        };
        let with_floor = worker(16, budget, rss, floor);
        let with_excl = worker(16, budget, rss, floor - placed);
        // Floor: max(60, 50 + 40) + 15 = 105 > 100. Excl: max(60, 10 + 40) + 40 = 100.
        assert!(!ProductionAdmission.admits(&Resources::mem(15), &view(&with_floor)));
        assert!(ProductionAdmission.admits(&Resources::mem(40), &view(&with_excl)));
        assert!(!ProductionAdmission.admits(&Resources::mem(41), &view(&with_excl)));
        // RSS stays the safety term: estimates that undercount cannot admit past it.
        let undercount = worker(16, budget, 95, 0);
        assert!(!ProductionAdmission.admits(&Resources::mem(6), &view(&undercount)));
    }

    /// Every dimension follows the same inequality, with `per_task` as a per-job floor; a zero
    /// capacity component is not enforced, host memory included.
    #[test]
    fn one_rule_per_dimension() {
        let a = ProductionAdmission;
        let s = WorkerState {
            per_task: Resources::mem(20).with_dev(30),
            ..WorkerState::new(1, "x", 8, Resources::mem(100).with_dev(100))
        };
        let view = |running, placed| WorkerView {
            state: &s,
            running,
            placed,
        };
        // Device: max(10, 2 * 30) + max(5, 30) = 90 <= 100, and 60 + 41 > 100.
        let v = view(2, Resources::mem(40).with_dev(10));
        assert!(a.admits(&Resources::mem(10).with_dev(5), &v));
        assert!(!a.admits(&Resources::mem(10).with_dev(41), &v));
        assert_eq!(v.short(&Resources::mem(70)).collect::<Vec<_>>(), vec![MEM]);
        let v = view(3, Resources::mem(40));
        assert_eq!(v.short(&Resources::ZERO).collect::<Vec<_>>(), vec![DEV]);
        // Host: the floor counts 3 * 20 = 60 against the 40 placed.
        assert_eq!(v.headroom(), [Some(40), Some(10)]);
        assert_eq!(a.bound(&v), None);
        // A zero budget leaves its dimension unenforced.
        let free = WorkerState::new(2, "x", 8, Resources::ZERO);
        let v = WorkerView {
            state: &free,
            running: 7,
            placed: Resources::mem(1 << 40),
        };
        assert!(a.admits(&Resources::MAX, &v));
        assert_eq!(v.headroom(), [None, None]);
        assert_eq!(v.free_share(&Resources::MAX), f64::INFINITY);
    }

    /// The scalar comparison is the bottleneck dimension's free fraction.
    #[test]
    fn free_share_is_the_bottleneck() {
        let s = WorkerState::new(1, "x", 8, Resources::mem(100).with_dev(10));
        let v = WorkerView {
            state: &s,
            running: 1,
            placed: Resources::mem(50).with_dev(2),
        };
        // Host 1 - 60/100 = 0.4, device 1 - 4/10 = 0.6.
        let share = v.free_share(&Resources::mem(10).with_dev(2));
        assert!((share - 0.4).abs() < 1e-12);
    }

    /// Without slots, not even the escape hatch admits.
    #[test]
    fn zero_slots_admit_nothing() {
        let s = worker(0, 100, 0, 0);
        let v = WorkerView {
            state: &s,
            running: 0,
            placed: Resources::ZERO,
        };
        assert!(!ProductionAdmission.admits(&Resources::ZERO, &v));
    }
}
