//! Reading a log back: replaying its inputs, and the polls it recorded.

use super::Event;
use crate::{Instant, Output, Policy};

/// Feed a log's inputs and polls to `policy`, in order, and return what each poll returned, with
/// its time. Given a fresh policy built as the logged one was (same [`Config`](crate::Config),
/// same admission rule), the result equals [`polls`] of the same log.
///
/// The outputs are recomputed, not copied from the log: a scripted log with empty polls replays
/// into the policy's actual decisions.
///
/// ```
/// use whelm::{
///     Config, Input, JobSpec, Output, Resources, Scheduler, WorkerState,
///     log::{self, Event},
/// };
///
/// let w = WorkerState {
///     id: 1,
///     budget: Resources::mem_gb(8.0),
///     ..Default::default()
/// };
/// let events = vec![
///     Event::Input {
///         t_s: 0.0,
///         input: Input::Worker(w),
///         info: None,
///     },
///     Event::Input {
///         t_s: 0.0,
///         input: Input::Submit(JobSpec {
///             id: 1,
///             demand: Resources::mem_gb(1.0),
///             ..Default::default()
///         }),
///         info: None,
///     },
///     Event::Poll {
///         t_s: 0.0,
///         out: Vec::new(),
///     },
/// ];
/// let replayed = log::replay(&mut Scheduler::new(Config::default()), events.clone());
/// assert_eq!(
///     replayed,
///     [(
///         0.0,
///         vec![Output::Start {
///             job: 1,
///             attempt: 1,
///             worker: 1
///         }]
///     )]
/// );
/// assert_eq!(log::polls(&events), [(0.0, vec![])]);
/// ```
pub fn replay<P: Policy + ?Sized>(
    policy: &mut P,
    events: impl IntoIterator<Item = Event>,
) -> Vec<(Instant, Vec<Output>)> {
    let mut out = Vec::new();
    for e in events {
        match e {
            Event::Input { t_s, input, .. } => policy.handle(input, t_s),
            Event::Poll { t_s, .. } => out.push((t_s, policy.poll(t_s))),
            Event::Sample { .. } => {}
        }
    }
    out
}

/// The polls recorded in a log, with their times and outputs: what [`replay`] should reproduce.
/// Inputs and samples are skipped (see the [module example](super)).
pub fn polls<'a>(events: impl IntoIterator<Item = &'a Event>) -> Vec<(Instant, Vec<Output>)> {
    events
        .into_iter()
        .filter_map(|e| match e {
            Event::Poll { t_s, out } => Some((*t_s, out.clone())),
            _ => None,
        })
        .collect()
}
