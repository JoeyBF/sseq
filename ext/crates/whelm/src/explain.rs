//! Why a job is where it is: the value [`Policy::explain`] returns.
//!
//! An [`Explanation`] names the job and its [`Status`]. A waiting job's status carries a
//! [`Waiting`] report: how long it has waited, the hold it owns, and one [`Verdict`] per worker
//! saying why that worker does not take it. Its [`Display`](fmt::Display) form is a one-line
//! summary for logs.

use std::{fmt, time::Duration};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{Admission, DagScheduler, Defer, Policy, Reservations, Scheduler, Timing, WorkerView};
use crate::{Attempt, DEV, DIMS, JobId, MEM, Resources, Time, Tried, WorkerId};

/// Bytes per gigabyte, for the [`Display`](fmt::Display) form.
const GB: f64 = 1e9;

/// Each dimension's name in the [`Display`](fmt::Display) form, indexed by dimension.
const DIM_NAMES: [&str; DIMS] = ["memory", "device memory", "slots"];

/// The most unmet dependencies [`Status::Pending`] lists in its
/// [`Display`](fmt::Display) form.
const SHOWN_DEPS: usize = 8;

/// Why a job is (not) running, from [`Policy::explain`].
///
/// [`Display`](fmt::Display) renders it as one line for logs; match on [`status`](Self::status)
/// to act on it.
///
/// # Examples
///
/// Job 2 waits behind job 1 on a one-slot worker.
///
/// ```
/// use whelm::{
///     Config, Explanation, Input, JobSpec, Policy, Scheduler, Status, Time, Verdict, WorkerState,
/// };
///
/// let mut p = Scheduler::new(Config::fifo());
/// p.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         slots: 1,
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// for id in 1..=2 {
///     p.handle(
///         Input::Submit(JobSpec {
///             id,
///             ..Default::default()
///         }),
///         Time::ORIGIN,
///     );
/// }
/// p.poll(Time::ORIGIN);
///
/// let e: Explanation = p.explain(2).unwrap();
/// let Status::Waiting(w) = &e.status else {
///     panic!("job 2 runs");
/// };
/// assert_eq!(w.workers, [(1, Verdict::SlotsFull)]);
/// assert_eq!(
///     e.to_string(),
///     "job 2 (demand 0.00 GB, group 0) waiting 0s, 0 more urgent job(s) waiting; slots full on \
///      1 worker(s)"
/// );
/// assert_eq!(
///     p.explain(1).unwrap().status,
///     Status::Running {
///         attempts: vec![(1, 1)]
///     }
/// );
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Explanation {
    /// The job (or unit) explained.
    pub job: JobId,
    /// Whether `job` names a [`DagScheduler`] unit of several jobs rather than one job.
    pub unit: bool,
    /// The job's label from its unit's [`NodeSource`](crate::NodeSource), if it has one.
    pub label: Option<String>,
    /// Where the job is.
    pub status: Status,
}

impl Explanation {
    /// An explanation of job `job`, with no label.
    pub fn new(job: JobId, status: Status) -> Self {
        Self {
            job,
            unit: false,
            label: None,
            status,
        }
    }

    /// The waiting report, if the job waits for a worker.
    pub fn waiting(&self) -> Option<&Waiting> {
        match &self.status {
            Status::Waiting(w) => Some(w),
            _ => None,
        }
    }
}

/// Where a job is. The variants after [`Waiting`](Self::Waiting) come from the
/// [`DagScheduler`] only.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Status {
    /// Running: its live attempts, oldest first, and their workers.
    Running {
        /// Each live attempt and the worker it runs on.
        attempts: Vec<(Attempt, WorkerId)>,
    },
    /// Submitted and waiting for a worker.
    Waiting(Box<Waiting>),
    /// Complete.
    Completed,
    /// Ready, and held until [`DagScheduler::release`] or run locally by the caller.
    Held,
    /// Named as a dependency but not declared yet.
    Undeclared {
        /// The units that name it.
        dependents: usize,
    },
    /// Unit `unit` waits for dependencies: the job's own (`unit` is the explained id), or the
    /// unit the job is a leaf of.
    Pending {
        /// The waiting unit.
        unit: JobId,
        /// Whether that unit is closed: declared to gain no more dependencies.
        closed: bool,
        /// Its dependencies not complete yet, by id.
        unmet: Vec<JobId>,
    },
    /// A unit whose dependencies are complete; its leaves are explained one by one.
    Open,
    /// A leaf of an open unit whose part of the unit is not materialised yet.
    Unentered {
        /// The unit.
        unit: JobId,
    },
    /// A leaf waiting for this many dependencies within its unit.
    PendingWithin {
        /// The dependencies not complete yet.
        unmet: usize,
    },
}

/// A waiting job, from the [`Scheduler`]: how long it has waited, and why no worker takes it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Waiting {
    /// Its demand.
    pub demand: Resources,
    /// Its group.
    pub group: u64,
    /// When it was submitted (or last went back to waiting).
    pub since: Time,
    /// How long it has waited.
    pub waited: Duration,
    /// Waiting jobs ahead of it in urgency order.
    pub ahead: usize,
    /// Whether it has waited past [`Config::age_limit`](crate::Config::age_limit), which puts it
    /// first in the scan.
    pub aged: bool,
    /// Its failed attempts, in order.
    pub tried: Vec<Tried>,
    /// The hold it owns, if any.
    pub hold: Option<Holding>,
    /// Its kind, if the job named one.
    pub kind: Option<String>,
    /// Under [`Timing::Unrelated`], the kind's learned speed factor per worker class: `(class,
    /// factor)`, by class name. Empty otherwise, and until the kind has samples.
    pub kind_factors: Vec<(String, f64)>,
    /// Every worker, by id, and whether it takes the job.
    pub workers: Vec<(WorkerId, Verdict)>,
}

impl Waiting {
    /// The workers that would take the job now.
    pub fn takers(&self) -> impl Iterator<Item = WorkerId> + '_ {
        (self.workers.iter())
            .filter(|(_, v)| *v == Verdict::Takes)
            .map(|(w, _)| *w)
    }
}

/// A worker a waiting job keeps on purpose.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Holding {
    /// The job reserves `worker` ([`Reservations`]), which drains for it.
    Reservation {
        /// The reserved worker.
        worker: WorkerId,
        /// When the reservation was made.
        since: Time,
        /// When the job is expected to fit there, once known: jobs expected to finish by then
        /// may backfill the worker.
        shadow: Option<Time>,
        /// Attempts still running on the worker.
        running: usize,
        /// The worker's slot count.
        slots: usize,
        /// The worker's usage as admission sees it ([`WorkerView::used`]).
        used: Resources,
    },
    /// The job waits for the faster, busy `worker` ([`Defer`]), declining every other worker.
    Deferral {
        /// The worker waited for.
        worker: WorkerId,
        /// When it is expected to free a slot.
        expected_free: Time,
        /// When the job stops waiting for it.
        until: Time,
    },
}

/// Whether a worker takes a waiting job, or the first reason it does not.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Verdict {
    /// It admits the job: the job starts there at the next poll unless a more urgent job takes
    /// the slot first.
    Takes,
    /// The job's constraints exclude it.
    Ineligible,
    /// It is reserved for job `by`, and the job cannot backfill it.
    Reserved {
        /// The job holding the reservation.
        by: JobId,
    },
    /// The job declines it, waiting for a faster worker ([`Holding::Deferral`]).
    Deferred,
    /// Admission refuses: every slot is taken.
    SlotsFull,
    /// Admission refuses with a slot free.
    Short {
        /// The dimensions the job does not fit in ([`WorkerView::short`]). None at all when the
        /// [`Admission`] rule refuses for reasons of its own.
        dims: [bool; DIMS],
        /// The worker's headroom per dimension ([`WorkerView::headroom`]).
        headroom: [Option<i64>; DIMS],
    },
}

/// A resource vector in words: host memory always, device memory when nonzero.
fn gb_list(r: &Resources) -> String {
    let mut s = format!("{:.2} GB", r[MEM] as f64 / GB);
    if r[DEV] > 0 {
        s += &format!(" + {:.2} GB device", r[DEV] as f64 / GB);
    }
    s
}

/// "1 dependency" or "`n` dependencies".
fn dependencies(n: usize) -> String {
    format!("{n} dependenc{}", if n == 1 { "y" } else { "ies" })
}

impl fmt::Display for Explanation {
    /// One line: the job, then what it waits for.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(l) = &self.label {
            write!(f, "[{l}] ")?;
        }
        let job = self.job;
        let noun = if self.unit { "unit" } else { "job" };
        match &self.status {
            Status::Running { attempts } => {
                write!(f, "{noun} {job} is running: ")?;
                for (i, (attempt, worker)) in attempts.iter().enumerate() {
                    let sep = if i == 0 { "" } else { ", " };
                    write!(f, "{sep}attempt {attempt} on worker {worker}")?;
                }
                Ok(())
            }
            Status::Waiting(w) => {
                write!(f, "{noun} {job} ")?;
                w.fmt(f)
            }
            Status::Completed => write!(f, "{noun} {job} completed"),
            Status::Held => write!(f, "{noun} {job} is ready and held until release"),
            Status::Undeclared { dependents } => write!(
                f,
                "{noun} {job} is not declared yet (named as a dependency of {dependents} unit(s))"
            ),
            Status::Pending {
                unit,
                closed,
                unmet,
            } => {
                if *unit == job {
                    write!(f, "{noun} {job} ")?;
                } else {
                    write!(f, "{noun} {job} waits for its unit: unit {unit} ")?;
                }
                if *closed {
                    write!(f, "is closed and ")?;
                }
                let shown = &unmet[..unmet.len().min(SHOWN_DEPS)];
                write!(f, "waits for {} {shown:?}", dependencies(unmet.len()))?;
                if unmet.len() > SHOWN_DEPS {
                    write!(f, " ...")?;
                }
                Ok(())
            }
            Status::Open => write!(f, "{noun} {job} is open"),
            Status::Unentered { unit } => write!(
                f,
                "{noun} {job} waits for its part of unit {unit} to be entered"
            ),
            Status::PendingWithin { unmet } => write!(
                f,
                "{noun} {job} waits for {} within its unit",
                dependencies(*unmet)
            ),
        }
    }
}

impl fmt::Display for Waiting {
    /// The waiting report, as it follows the job's id in an [`Explanation`]: the wait, the hold,
    /// then the workers summarised by verdict.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(demand {}, group {}) waiting {:.0}s",
            gb_list(&self.demand),
            self.group,
            self.waited.as_secs_f64()
        )?;
        if self.aged {
            write!(f, " (aged)")?;
        }
        write!(f, ", {} more urgent job(s) waiting", self.ahead)?;
        if let Some(last) = self.tried.last() {
            write!(
                f,
                "; failed {} time(s), last on worker {} ({:?}: {})",
                self.tried.len(),
                last.worker,
                last.kind,
                last.why
            )?;
        }
        match &self.hold {
            Some(Holding::Reservation {
                worker,
                running,
                slots,
                used,
                ..
            }) => write!(
                f,
                "; holds the reservation on worker {worker} (draining: {running}/{slots} running, \
                 used {})",
                gb_list(used)
            )?,
            Some(Holding::Deferral {
                worker,
                expected_free,
                ..
            }) => write!(
                f,
                "; waiting for faster worker {worker} (expected free at t={:.0})",
                expected_free.0.as_secs_f64()
            )?,
            None => {}
        }
        if let Some(name) = &self.kind
            && !self.kind_factors.is_empty()
        {
            write!(f, "; kind {name} runs ")?;
            for (i, (class, factor)) in self.kind_factors.iter().enumerate() {
                let sep = if i == 0 { "" } else { ", " };
                write!(f, "{sep}{factor:.2}x on class {class}")?;
            }
        }
        if self.workers.is_empty() {
            write!(f, "; no workers")?;
        }
        let count = |v: &Verdict| self.workers.iter().filter(|(_, w)| w == v).count();
        let full = count(&Verdict::SlotsFull);
        if full > 0 {
            write!(f, "; slots full on {full} worker(s)")?;
        }
        for (d, name) in DIM_NAMES.iter().enumerate() {
            // Workers short of `d`, and the first with the most headroom there.
            let mut short = 0;
            let mut best: Option<(i64, WorkerId)> = None;
            for (id, v) in &self.workers {
                if let Verdict::Short { dims, headroom } = v
                    && dims[d]
                {
                    short += 1;
                    let h = headroom[d].unwrap_or(i64::MAX);
                    if best.is_none_or(|(b, _)| h > b) {
                        best = Some((h, *id));
                    }
                }
            }
            if let Some((h, w)) = best {
                write!(
                    f,
                    "; {name} short on {short} worker(s) (best headroom {:.2} GB on worker {w})",
                    h as f64 / GB
                )?;
            }
        }
        let excluded = count(&Verdict::Ineligible);
        if excluded > 0 {
            write!(f, "; {excluded} worker(s) excluded by its constraints")?;
        }
        let mut reserved = (self.workers.iter()).filter_map(|(id, v)| match v {
            Verdict::Reserved { by } => Some((id, by)),
            _ => None,
        });
        if let Some((id, by)) = reserved.next() {
            write!(f, "; reserved: worker {id} for job {by}")?;
            for (id, by) in reserved {
                write!(f, ", worker {id} for job {by}")?;
            }
        }
        let takers: Vec<WorkerId> = self.takers().collect();
        if !takers.is_empty() {
            write!(
                f,
                "; admitted on worker(s) {takers:?} (placed at the next poll unless a more urgent \
                 job takes the slot)"
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FailKind;

    /// A waiting job with one worker of each verdict.
    fn waiting() -> Explanation {
        let mut short = [false; DIMS];
        short[MEM] = true;
        let w = Waiting {
            demand: Resources::mem_gb(2.0).with_dev_gb(1.0),
            group: 3,
            since: Time::ORIGIN,
            waited: Duration::from_secs(90),
            ahead: 4,
            aged: true,
            tried: vec![Tried {
                worker: 6,
                kind: FailKind::Timeout,
                why: "slow".into(),
            }],
            hold: Some(Holding::Reservation {
                worker: 2,
                since: Time::ORIGIN,
                shadow: None,
                running: 1,
                slots: 2,
                used: Resources::mem_gb(3.0),
            }),
            kind: Some("k".into()),
            kind_factors: vec![("a".into(), 1.5), ("b".into(), 0.5)],
            workers: vec![
                (1, Verdict::Takes),
                (2, Verdict::SlotsFull),
                (3, Verdict::Ineligible),
                (4, Verdict::Reserved { by: 9 }),
                (
                    5,
                    Verdict::Short {
                        dims: short,
                        headroom: [Some(1_000_000_000), None, Some(1)],
                    },
                ),
            ],
        };
        Explanation::new(7, Status::Waiting(Box::new(w)))
    }

    /// The one-line form names every part of the report.
    #[test]
    fn display() {
        assert_eq!(
            waiting().to_string(),
            "job 7 (demand 2.00 GB + 1.00 GB device, group 3) waiting 90s (aged), 4 more urgent \
             job(s) waiting; failed 1 time(s), last on worker 6 (Timeout: slow); holds the \
             reservation on worker 2 (draining: 1/2 running, used 3.00 GB); kind k runs 1.50x on \
             class a, 0.50x on class b; slots full on 1 worker(s); memory short on 1 worker(s) \
             (best headroom 1.00 GB on worker 5); 1 worker(s) excluded by its constraints; \
             reserved: worker 4 for job 9; admitted on worker(s) [1] (placed at the next poll \
             unless a more urgent job takes the slot)"
        );
        let pending = Explanation {
            unit: true,
            label: Some("l".into()),
            ..Explanation::new(
                20,
                Status::Pending {
                    unit: 20,
                    closed: true,
                    unmet: (1..=9).collect(),
                },
            )
        };
        assert_eq!(
            pending.to_string(),
            "[l] unit 20 is closed and waits for 9 dependencies [1, 2, 3, 4, 5, 6, 7, 8] ..."
        );
    }

    /// An explanation survives a JSON round trip.
    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        let e = waiting();
        let json = serde_json::to_string(&e).unwrap();
        assert_eq!(serde_json::from_str::<Explanation>(&json).unwrap(), e);
    }
}
