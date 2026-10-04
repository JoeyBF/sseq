//! Reading a log back: replaying its inputs, and the polls it recorded.

use super::Event;
use crate::{
    policy::{Output, Policy},
    time::Time,
};

/// Feed a log's inputs and polls to `policy`, in order, and return what each poll returned, with
/// its time. Given a fresh policy built as the logged one was (same
/// [`Config`](crate::config::Config), same admission rule), the result equals [`polls`] of the same
/// log.
///
/// The outputs are recomputed, not copied from the log: a scripted log with empty polls replays
/// into the policy's actual decisions.
///
/// ```
/// use whelm::{
///     log::{self, Event},
///     prelude::*,
/// };
///
/// let w = WorkerState {
///     id: 1,
///     capacity: Resources::new().with(MEMORY, gb(8.0)).with(SLOTS, 1),
///     ..Default::default()
/// };
/// let events = vec![
///     Event::Input {
///         t: Time::ORIGIN,
///         input: Input::Worker(w),
///         info: None,
///     },
///     Event::Input {
///         t: Time::ORIGIN,
///         input: Input::Submit {
///             job: 1,
///             spec: JobSpec {
///                 demand: Resources::new().with(MEMORY, gb(1.0)),
///                 ..Default::default()
///             },
///         },
///         info: None,
///     },
///     Event::Poll {
///         t: Time::ORIGIN,
///         out: Vec::new(),
///     },
/// ];
/// let replayed = log::replay(&mut Scheduler::new(Config::default()), events.clone());
/// assert_eq!(
///     replayed,
///     [(
///         Time::ORIGIN,
///         vec![Output::Start {
///             job: 1,
///             attempt: 1,
///             worker: 1
///         }]
///     )]
/// );
/// assert_eq!(log::polls(&events), [(Time::ORIGIN, vec![])]);
/// ```
pub fn replay<P: Policy + ?Sized>(
    policy: &mut P,
    events: impl IntoIterator<Item = Event>,
) -> Vec<(Time, Vec<Output>)> {
    let mut out = Vec::new();
    for e in events {
        match e {
            Event::Input { t, input, .. } => policy.handle(input, t),
            Event::Poll { t, .. } => out.push((t, policy.poll(t))),
            Event::Sample { .. } => {}
        }
    }
    out
}

/// The polls recorded in a log, with their times and outputs: what [`replay`] should reproduce.
/// Inputs and samples are skipped (see the [module example](super)).
pub fn polls<'a>(events: impl IntoIterator<Item = &'a Event>) -> Vec<(Time, Vec<Output>)> {
    events
        .into_iter()
        .filter_map(|e| match e {
            Event::Poll { t, out } => Some((*t, out.clone())),
            _ => None,
        })
        .collect()
}
