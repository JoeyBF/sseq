//! Urgency keys, the waiting queue and the order `dispatch` scans it in.

use std::ops::Bound;

use super::{Job, Scheduler};
#[cfg(doc)]
use crate::Config;
use crate::{GroupOrder, JobId, JobSpec, OrderTerm, SLOTS, Time};

/// The most terms a [`Config::order`] has once repeats are dropped: one per [`OrderTerm`].
const ORDER_TERMS: usize = 5;

/// Urgency: smaller is more urgent. `terms[i]` is the key of `config.order[i]`; the rest are 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Key {
    terms: [i64; ORDER_TERMS],
    seq: u64,
}

/// Position of `dispatch`'s scan: first the aged jobs by age, then the rest by urgency.
#[derive(Clone, Copy, Debug)]
pub(super) enum Cursor {
    Aged(Option<u64>),
    Queue(Option<Key>),
}

/// A float as a totally ordered integer key (for order and score keys).
pub(super) fn ordered(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    b ^ (((b >> 63) as u64) >> 1) as i64
}

/// `terms` without repeats, keeping first occurrences.
pub(super) fn dedup<T: PartialEq + Copy>(terms: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(terms.len());
    for &t in terms {
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

impl Scheduler {
    /// The urgency key of a job submitted as number `seq`, recording its group's first arrival
    /// if groups are ordered by it.
    pub(super) fn key(&mut self, spec: &JobSpec, seq: u64) -> Key {
        let mut terms = [0; ORDER_TERMS];
        for (slot, term) in terms.iter_mut().zip(&self.config.order) {
            *slot = match term {
                OrderTerm::Priority => spec.priority.unwrap_or(self.config.default_priority),
                // Never `i64::MAX`, so a zero rank still beats none.
                OrderTerm::Rank => spec.rank.map_or(i64::MAX, |r| {
                    -(i64::try_from(r.as_nanos()).unwrap_or(i64::MAX))
                }),
                OrderTerm::Group => match self.config.group_order {
                    GroupOrder::Arrival => *self.groups.entry(spec.group).or_insert(seq) as i64,
                    // Order-preserving from u64 to i64.
                    GroupOrder::Id => spec.group as i64 ^ i64::MIN,
                },
                OrderTerm::Wspt => {
                    (spec.work).map_or(i64::MAX, |w| ordered(-(spec.weight / w.as_secs_f64())))
                }
                // Order-preserving from u64 to i64.
                OrderTerm::Edd => spec.due.map_or(i64::MAX, |d| {
                    u64::try_from(d.0.as_nanos()).unwrap_or(u64::MAX) as i64 ^ i64::MIN
                }),
            };
        }
        Key { terms, seq }
    }

    /// Queue a new job under its urgency key, demanding one slot.
    pub(super) fn submit(&mut self, mut spec: JobSpec, now: Time) {
        if self.waiting.contains_key(&spec.id) || self.running.contains_key(&spec.id) {
            return;
        }
        spec.demand[SLOTS] = 1;
        let seq = self.next_seq;
        self.next_seq += 1;
        let key = self.key(&spec, seq);
        let kind = self.speeds.kind(spec.kind.as_deref());
        self.enqueue(Job {
            kind,
            spec,
            key,
            since: now,
            attempts: 0,
            tried: Vec::new(),
            retry_avoid: Vec::new(),
            speculated: 0,
        });
    }

    /// Put a job in the waiting indexes under its key.
    pub(super) fn enqueue(&mut self, job: Job) {
        self.queue.insert(job.key, job.spec.id);
        self.by_age.insert(job.key.seq, job.spec.id);
        self.waiting.insert(job.spec.id, job);
    }

    /// Take a job out of the waiting indexes, releasing its hold.
    pub(super) fn remove_waiting(&mut self, job: JobId) -> Option<Job> {
        self.release_hold(job);
        let j = self.waiting.remove(&job)?;
        self.queue.remove(&j.key);
        self.by_age.remove(&j.key.seq);
        Some(j)
    }

    /// Whether `job` has waited past the age limit.
    pub(super) fn aged(&self, job: &Job) -> bool {
        self.config
            .age_limit
            .is_some_and(|a| self.now - job.since >= a)
    }

    /// The next job in scan order: aged jobs oldest first, then everything else by urgency.
    pub(super) fn next_job(&self, cursor: &mut Cursor) -> Option<JobId> {
        loop {
            match *cursor {
                Cursor::Aged(after) => {
                    if self.config.age_limit.is_some() {
                        let lower = after.map_or(Bound::Unbounded, Bound::Excluded);
                        // `by_age` is in submission order, so the aged jobs are a prefix of it.
                        if let Some((&seq, &job)) =
                            self.by_age.range((lower, Bound::Unbounded)).next()
                            && self.aged(&self.waiting[&job])
                        {
                            *cursor = Cursor::Aged(Some(seq));
                            return Some(job);
                        }
                    }
                    *cursor = Cursor::Queue(None);
                }
                Cursor::Queue(after) => {
                    let lower = after.map_or(Bound::Unbounded, Bound::Excluded);
                    let (&key, &job) = self.queue.range((lower, Bound::Unbounded)).next()?;
                    *cursor = Cursor::Queue(Some(key));
                    if !self.aged(&self.waiting[&job]) {
                        return Some(job);
                    }
                }
            }
        }
    }

    /// Scan order as a sortable value: aged jobs first by age, then the rest by urgency.
    pub(super) fn urgency(&self, job: &Job) -> (bool, Key) {
        if self.aged(job) {
            (
                false,
                Key {
                    terms: [0; ORDER_TERMS],
                    seq: job.key.seq,
                },
            )
        } else {
            (true, job.key)
        }
    }
}
