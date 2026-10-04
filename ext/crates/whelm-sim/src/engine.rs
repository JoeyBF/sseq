//! The event queue and processor-sharing workers the simulators' event loops are built from.

use std::{cmp::Ordering, collections::BinaryHeap};

use whelm::{job::JobId, policy::Attempt};

/// Timed events, earliest first.
///
/// Events at the same time come out by their tie key: push order for [`push`](Self::push), so
/// every run is deterministic.
pub struct Queue<E> {
    heap: BinaryHeap<Item<E>>,
    seq: u64,
}

/// A queued event with its time and tie key.
struct Item<E> {
    t: f64,
    tie: u64,
    ev: E,
}

impl<E> PartialEq for Item<E> {
    /// Equal when [`Ord`] says so.
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

impl<E> Eq for Item<E> {}

impl<E> PartialOrd for Item<E> {
    /// The total order of [`Ord`].
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl<E> Ord for Item<E> {
    /// Reversed: `BinaryHeap` is a max-heap and the earliest event comes first.
    fn cmp(&self, o: &Self) -> Ordering {
        o.t.total_cmp(&self.t).then(o.tie.cmp(&self.tie))
    }
}

impl<E> Default for Queue<E> {
    /// An empty queue.
    fn default() -> Self {
        Self {
            heap: BinaryHeap::new(),
            seq: 0,
        }
    }
}

impl<E> Queue<E> {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue `ev` at time `t`, after every event already queued at `t`.
    pub fn push(&mut self, t: f64, ev: E) {
        self.seq += 1;
        let tie = self.seq;
        self.push_tied(t, tie, ev);
    }

    /// Queue `ev` at time `t` with an explicit tie key (smaller first among events at `t`).
    pub fn push_tied(&mut self, t: f64, tie: u64, ev: E) {
        self.heap.push(Item { t, tie, ev });
    }

    /// The earliest event and its time.
    pub fn pop(&mut self) -> Option<(f64, E)> {
        self.heap.pop().map(|i| (i.t, i.ev))
    }

    /// Every event of the earliest instant, in order, and that instant.
    ///
    /// A coordinator that drains its event queue before placing applies them all before the next
    /// poll.
    pub fn pop_instant(&mut self) -> Option<(f64, Vec<E>)> {
        let (t, first) = self.pop()?;
        let mut events = vec![first];
        while self.heap.peek().is_some_and(|i| i.t == t) {
            events.push(self.heap.pop().unwrap().ev);
        }
        Some((t, events))
    }
}

/// One attempt running on a [`PsWorker`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Run {
    /// The job.
    pub job: JobId,
    /// Its attempt.
    pub attempt: Attempt,
    /// Work left, in the units the worker's rate is given in.
    pub left: f64,
}

/// A simulated worker sharing its throughput among its running attempts.
///
/// They all progress at one rate that the simulator derives from them (e.g. from how many there
/// are).
///
/// The worker has at most one pending completion event: each
/// [`next_completion`](Self::next_completion) supersedes the previous one, which the event loop
/// recognises as stale with [`is_current`](Self::is_current).
#[derive(Clone, Debug, Default)]
pub struct PsWorker {
    /// The running attempts, in start order.
    pub running: Vec<Run>,
    /// Attempt-seconds spent running.
    pub busy: f64,
    last: f64,
    version: u64,
}

impl PsWorker {
    /// Progress every running attempt to `now` at `rate(running)` per attempt.
    ///
    /// Returns the time elapsed and the rate, if anything ran.
    pub fn advance(&mut self, now: f64, rate: impl FnOnce(&[Run]) -> f64) -> Option<(f64, f64)> {
        let dt = now - self.last;
        self.last = now;
        let k = self.running.len();
        if dt <= 0.0 || k == 0 {
            return None;
        }
        let r = rate(&self.running);
        for x in &mut self.running {
            x.left -= r * dt;
        }
        self.busy += k as f64 * dt;
        Some((dt, r))
    }

    /// Start an attempt with `work` to do (call [`advance`](Self::advance) first).
    pub fn start(&mut self, job: JobId, attempt: Attempt, work: f64) {
        self.running.push(Run {
            job,
            attempt,
            left: work,
        });
    }

    /// Remove an attempt without finishing it (call [`advance`](Self::advance) first).
    ///
    /// Returns whether it was running here.
    pub fn stop(&mut self, job: JobId, attempt: Attempt) -> bool {
        let before = self.running.len();
        self.running
            .retain(|r| (r.job, r.attempt) != (job, attempt));
        self.running.len() < before
    }

    /// The time and version of the next completion at the rate `rate(running)`, if anything runs.
    ///
    /// Supersedes every earlier one.
    pub fn next_completion(
        &mut self,
        now: f64,
        rate: impl FnOnce(&[Run]) -> f64,
    ) -> Option<(f64, u64)> {
        self.version += 1;
        if self.running.is_empty() {
            return None;
        }
        let min = self
            .running
            .iter()
            .map(|x| x.left)
            .fold(f64::INFINITY, f64::min)
            .max(0.0);
        Some((now + min / rate(&self.running), self.version))
    }

    /// Whether a completion event of this version is the latest one.
    pub fn is_current(&self, version: u64) -> bool {
        version == self.version
    }

    /// Remove and return the attempts `done(left, least)` says are finished, in start order.
    ///
    /// `least` is the smallest work left of any running attempt.
    pub fn finish(&mut self, done: impl Fn(f64, f64) -> bool) -> Vec<Run> {
        let least = self
            .running
            .iter()
            .map(|x| x.left)
            .fold(f64::INFINITY, f64::min);
        let mut finished = Vec::new();
        self.running.retain(|&r| {
            let fin = done(r.left, least);
            if fin {
                finished.push(r);
            }
            !fin
        });
        finished
    }
}
