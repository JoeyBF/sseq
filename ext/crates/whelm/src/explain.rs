//! Why a job is where it is: the value [`Policy::explain`] returns.
//!
//! An [`Explanation`] names the job and its [`Status`]. A waiting job's status carries a
//! [`Waiting`] report: how long it has waited, the hold it owns, and one [`Verdict`] per worker
//! saying why that worker does not take it. Its [`Display`](fmt::Display) form is a one-line
//! summary for logs.

use std::{borrow::Cow, fmt, time::Duration};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{
    admission::{Admission, WorkerView},
    config::{Defer, Reservations},
    dag::DagScheduler,
    policy::Policy,
    scheduler::Scheduler,
    speed::Timing,
};
use crate::{
    job::JobId,
    policy::{Attempt, Tried},
    resources::{Resource, ResourceUnit, Resources},
    time::Time,
    worker::WorkerId,
};

/// Bytes per gigabyte, for the [`Display`](fmt::Display) form.
const GB: f64 = 1e9;

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
///     explain::{Explanation, Status, Verdict},
///     prelude::*,
/// };
///
/// let mut p = Scheduler::new(Config::fifo());
/// p.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         capacity: Resources::new().with(SLOTS, 1),
///         ..Default::default()
///     }),
///     Time::ORIGIN,
/// );
/// for job in 1..=2 {
///     p.handle(
///         Input::Submit {
///             job,
///             spec: JobSpec::default(),
///         },
///         Time::ORIGIN,
///     );
/// }
/// p.poll(Time::ORIGIN);
///
/// let e: Explanation = p.explain(2).unwrap();
/// let Status::Waiting(w) = &e.status else {
///     panic!("job 2 runs");
/// };
/// let full = Verdict::Full {
///     dims: vec![SLOTS.name],
/// };
/// assert_eq!(w.workers, [(1, full)]);
/// assert_eq!(
///     e.to_string(),
///     "job 2 (demand [slots 1], group 0) waiting 0s, 0 more urgent job(s) waiting; slots full \
///      on 1 worker(s)"
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
    /// The job's label from its unit's [`NodeSource`](crate::dag::NodeSource), if it has one.
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
    /// Its demand, as the scheduler holds it ([`Admission`] says how).
    pub demand: Resources,
    /// The resources the scheduler declares
    /// ([`Config::resources`](crate::config::Config::resources)): the order the
    /// [`Display`](fmt::Display) form lists amounts in, and their units.
    pub resources: Vec<Resource>,
    /// Its group.
    pub group: u64,
    /// When it was submitted (or last went back to waiting).
    pub since: Time,
    /// How long it has waited.
    pub waited: Duration,
    /// Waiting jobs ahead of it in urgency order.
    pub ahead: usize,
    /// Whether it has waited past [`Config::age_limit`](crate::config::Config::age_limit), which
    /// puts it first in the scan.
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
        /// The worker's capacity.
        capacity: Resources,
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
    /// Admission refuses for want of a [hard](Resource::hard) resource: under the default
    /// declaration, every slot is taken.
    Full {
        /// The names of the hard resources the job does not fit in ([`WorkerView::short`]), in
        /// declaration order.
        dims: Vec<Cow<'static, str>>,
    },
    /// Admission refuses with room in every hard resource.
    Short {
        /// The names of the soft resources the job does not fit in ([`WorkerView::short`]), in
        /// declaration order. None at all when the [`Admission`] rule refuses for reasons of its
        /// own.
        dims: Vec<Cow<'static, str>>,
        /// The worker's headroom in each declared resource, by name, in declaration order
        /// ([`WorkerView::headroom`]).
        headroom: Vec<(Cow<'static, str>, Option<i64>)>,
    },
}

/// An amount of `resource` in words.
fn amount(resource: &Resource, x: i128) -> String {
    match resource.unit {
        ResourceUnit::Count => x.to_string(),
        ResourceUnit::Bytes => format!("{:.2} GB", x as f64 / GB),
    }
}

/// Amounts in words: each nonzero one by name, in declaration order.
fn list(r: &Resources, resources: &[Resource]) -> String {
    let parts: Vec<String> = (resources.iter())
        .filter(|res| r.get(res) > 0)
        .map(|res| format!("{} {}", res.name, amount(res, r.get(res).into())))
        .collect();
    format!("[{}]", parts.join(", "))
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
            list(&self.demand, &self.resources),
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
                capacity,
                used,
                ..
            }) => write!(
                f,
                "; holds the reservation on worker {worker} (draining: {running} running, using \
                 {} of {})",
                list(used, &self.resources),
                list(capacity, &self.resources)
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
        for r in &self.resources {
            let full = (self.workers.iter())
                .filter(|(_, v)| matches!(v, Verdict::Full { dims } if dims.contains(&r.name)))
                .count();
            if full > 0 {
                write!(f, "; {} full on {full} worker(s)", r.name)?;
            }
        }
        for r in &self.resources {
            // Workers short of `r`, and the first with the most headroom there.
            let mut short = 0;
            let mut best: Option<(i64, WorkerId)> = None;
            for (id, v) in &self.workers {
                if let Verdict::Short { dims, headroom } = v
                    && dims.contains(&r.name)
                {
                    short += 1;
                    let h = (headroom.iter())
                        .find(|(name, _)| *name == r.name)
                        .and_then(|h| h.1)
                        .unwrap_or(i64::MAX);
                    if best.is_none_or(|(b, _)| h > b) {
                        best = Some((h, *id));
                    }
                }
            }
            if let Some((h, w)) = best {
                write!(
                    f,
                    "; {} short on {short} worker(s) (best headroom {} on worker {w})",
                    r.name,
                    amount(r, h.into())
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
    use crate::{
        config::Config,
        policy::FailKind,
        resources::{DEVICE_MEMORY, MEMORY, SLOTS, gb},
    };

    /// A waiting job with one worker of each verdict.
    fn waiting() -> Explanation {
        let w = Waiting {
            demand: Resources::new()
                .with(MEMORY, gb(2.0))
                .with(DEVICE_MEMORY, gb(1.0))
                .with(SLOTS, 1),
            resources: Config::default().resources,
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
                capacity: Resources::new().with(MEMORY, gb(8.0)).with(SLOTS, 2),
                used: Resources::new().with(MEMORY, gb(3.0)).with(SLOTS, 1),
            }),
            kind: Some("k".into()),
            kind_factors: vec![("a".into(), 1.5), ("b".into(), 0.5)],
            workers: vec![
                (1, Verdict::Takes),
                (
                    2,
                    Verdict::Full {
                        dims: vec![SLOTS.name],
                    },
                ),
                (3, Verdict::Ineligible),
                (4, Verdict::Reserved { by: 9 }),
                (
                    5,
                    Verdict::Short {
                        dims: vec![MEMORY.name],
                        headroom: vec![
                            (MEMORY.name, Some(1_000_000_000)),
                            (DEVICE_MEMORY.name, None),
                            (SLOTS.name, Some(1)),
                        ],
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
            "job 7 (demand [memory 2.00 GB, device memory 1.00 GB, slots 1], group 3) waiting 90s \
             (aged), 4 more urgent job(s) waiting; failed 1 time(s), last on worker 6 (Timeout: \
             slow); holds the reservation on worker 2 (draining: 1 running, using [memory 3.00 \
             GB, slots 1] of [memory 8.00 GB, slots 2]); kind k runs 1.50x on class a, 0.50x on \
             class b; slots full on 1 worker(s); memory short on 1 worker(s) (best headroom 1.00 \
             GB on worker 5); 1 worker(s) excluded by its constraints; reserved: worker 4 for job \
             9; admitted on worker(s) [1] (placed at the next poll unless a more urgent job takes \
             the slot)"
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
