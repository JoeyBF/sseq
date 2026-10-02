//! The admission rule: whether a worker accepts a job right now.

use crate::{Resources, WorkerState};

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

    /// Usage as the production rule counts it: `max(reported_used, reported_baseline + placed)`.
    ///
    /// The reported figure lags (heartbeats); the placed sum is exact but only an estimate of what
    /// the jobs use. Taking the maximum is conservative in both directions.
    pub fn effective_used(&self) -> Resources {
        self.state
            .reported_used
            .max(self.state.reported_baseline + self.placed)
    }

    /// Memory headroom, `budget - effective_used`, in bytes; negative when over-committed.
    pub fn headroom(&self) -> i64 {
        let h = self.state.budget.mem as i128 - self.effective_used().mem as i128;
        h.clamp(i64::MIN as i128, i64::MAX as i128) as i64
    }

    /// Device memory in use as admission counts it: `max(Σ placed dev, running * dev_per_task)`.
    pub fn dev_used(&self) -> u64 {
        self.placed
            .dev
            .max((self.running as u64).saturating_mul(self.state.dev_per_task))
    }

    /// Device headroom, `budget.dev - dev_used`, in bytes; `None` when the capacity is unknown.
    pub fn dev_headroom(&self) -> Option<i64> {
        let cap = self.state.budget.dev;
        (cap > 0).then(|| {
            (cap as i128 - self.dev_used() as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
        })
    }

    /// The device rule: a job needing `demand` bytes of device memory (at least the worker's
    /// `dev_per_task`) fits in the remaining device capacity, or the capacity is unknown.
    pub fn device_admits(&self, demand: u64) -> bool {
        let cap = self.state.budget.dev;
        cap == 0
            || self
                .dev_used()
                .saturating_add(demand.max(self.state.dev_per_task))
                <= cap
    }
}

/// Whether a worker admits a job.
///
/// # Contract
///
/// Implementations must be **monotone in load**: if a job is refused by a worker, it stays refused
/// after more jobs are placed on that worker (with no completion or heartbeat in between). The
/// policies rely on this to guarantee the priority invariant within one `dispatch`.
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

/// The production admission rule, host memory AND device memory:
///
/// ```text
/// admit(job on w) iff running(w) < slots(w)
///                 and ( running(w) == 0      // escape hatch: a job alone always goes
///                       or ( max(reported_used, reported_baseline + Σ placed) + demand <= budget
///                            and device fits ) )
/// device fits    iff budget.dev == 0         // unknown capacity: not enforced
///                    or max(Σ placed.dev, running * dev_per_task)
///                         + max(demand.dev, dev_per_task) <= budget.dev
/// ```
///
/// With `dev_per_task` alone (jobs without device demands) the device rule is the per-worker count
/// form `(running + 1) * dev_per_task <= cap`; with per-job device demands it is their sum.
///
/// `reported_baseline` must exclude the running jobs (a worker's `baseline_excl`: its rolling RSS
/// floor minus their estimates). A floor that contains them counts them twice, once in it and
/// once in `Σ placed`, and keeps a busy worker a few GB short of its budget.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProductionAdmission;

impl Admission for ProductionAdmission {
    /// The production rule itself.
    fn admits(&self, demand: &Resources, w: &WorkerView) -> bool {
        if w.running >= w.state.slots {
            return false;
        }
        w.running == 0
            || (w.effective_used().mem.saturating_add(demand.mem) <= w.state.budget.mem
                && w.device_admits(demand.dev))
    }

    /// The headroom left under the production rule (unbounded when the worker is empty).
    fn bound(&self, w: &WorkerView) -> Option<Resources> {
        if w.running >= w.state.slots {
            None
        } else if w.running == 0 {
            Some(Resources::MAX)
        } else {
            let used = w.effective_used().mem;
            let mem = (used <= w.state.budget.mem).then(|| w.state.budget.mem - used)?;
            let dev = match w.dev_headroom() {
                None => u64::MAX,
                Some(h) if h >= w.state.dev_per_task as i64 => h as u64,
                Some(_) => return None,
            };
            Some(Resources::mem(mem).with_dev(dev))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(view(1, 50).headroom(), 40);
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
