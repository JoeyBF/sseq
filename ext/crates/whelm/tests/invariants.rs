//! Property tests of the policy invariants over random event streams.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

use proptest::prelude::*;
use whelm::{
    Attempt, Config, Constraint, DIMS, Defer, Explanation, FailKind, GaveUp, GroupOrder, Input,
    JobId, JobSpec, Learn, OrderTerm, Output, Policy, Reservations, Resources, RetryConfig, SLOTS,
    Scheduler, ScoreTerm, Selector, Speculate, SpeedConfig, Status, Strength, Time, Timing, Tried,
    WorkerId, WorkerState,
};

/// `x` seconds rounded to the nanosecond, as the policy turns its run-time arithmetic back into a
/// span (negative gives zero).
fn secs(x: f64) -> Duration {
    Duration::from_nanos((x * 1e9).round().max(0.0) as u64)
}

/// The configuration under test, minus speed and retries.
#[derive(Clone, Debug)]
struct Rule {
    order: Vec<OrderTerm>,
    group_order: GroupOrder,
    default_priority: i64,
    age: Option<Duration>,
    /// Reservation limit and whether it is per class; `None` for no reservations.
    reservations: Option<(usize, bool)>,
    score: Vec<ScoreTerm>,
}

impl Rule {
    /// The policy under test.
    fn build(&self, speed: SpeedConfig, retry: RetryConfig) -> Box<dyn Policy> {
        Box::new(Scheduler::new(Config {
            order: self.order.clone(),
            group_order: self.group_order,
            default_priority: self.default_priority,
            age_limit: self.age,
            reservations: self.reservations.map(|(max, per_class)| Reservations {
                reserve_after: Duration::from_secs(30),
                max,
                per_class,
                shadow_backfill: false,
            }),
            score: self.score.clone(),
            speed,
            retry,
        }))
    }
}

#[derive(Clone, Debug)]
enum Op {
    Submit {
        demand: u64,
        /// What the caller writes in the slot component, which the policy overrides.
        slots: u64,
        group: u64,
        priority: Option<i64>,
        rank: Option<u8>,
        weight: f64,
        due: Option<u16>,
        constraints: Vec<Constraint>,
        work: Option<u32>,
        dev: Option<u64>,
        kind: Option<u8>,
    },
    /// A live attempt (running job, then attempt, by index) finishes.
    Complete(usize, usize),
    /// A live attempt fails.
    Fail(usize, usize, FailKind),
    /// A report about an attempt that is not live (finished, failed, stopped or never started).
    Stale {
        job: usize,
        attempt: usize,
        done: bool,
    },
    Cancel(usize),
    Worker {
        id: WorkerId,
        slots: usize,
        budget: u64,
        used: u64,
        baseline: u64,
        class: u8,
        dev_cap: u64,
        per_task: u64,
    },
    Gone(WorkerId),
    Tick(u32),
}

/// An index into a collection of up to `1 << 16` elements.
fn index() -> impl Strategy<Value = usize> {
    any::<prop::sample::Index>().prop_map(|i| i.index(1 << 16))
}

/// A random constraint over the workers and classes the streams use.
fn constraint() -> impl Strategy<Value = Constraint> {
    let on = prop_oneof![
        (0u64..4).prop_map(Selector::Worker),
        (0u8..2).prop_map(|c| Selector::Class(format!("c{c}"))),
    ];
    let strength = prop_oneof![
        Just(Strength::Require),
        Just(Strength::Forbid),
        Just(Strength::Avoid),
        Just(Strength::Prefer),
    ];
    (on, strength).prop_map(|(on, strength)| Constraint { on, strength })
}

/// A random event.
fn op() -> impl Strategy<Value = Op> {
    let kind = prop_oneof![
        Just(FailKind::DeviceOom),
        Just(FailKind::Other),
        Just(FailKind::Timeout)
    ];
    let submit = (
        (1u64..80, 0u64..3, 0u64..4),
        prop::option::weighted(0.2, -2i64..3),
        prop::option::weighted(0.5, 0u8..6),
        prop_oneof![Just(1.0), Just(0.5), Just(3.0)],
        prop::option::weighted(0.5, 0u16..300),
        prop::option::weighted(0.4, prop::collection::vec(constraint(), 1..3)),
        prop::option::weighted(0.7, 1u32..120),
        prop::option::weighted(0.5, 1u64..40),
        prop::option::weighted(0.6, 0u8..3),
    )
        .prop_map(
            |(
                (demand, slots, group),
                priority,
                rank,
                weight,
                due,
                constraints,
                work,
                dev,
                kind,
            )| {
                Op::Submit {
                    demand,
                    slots,
                    group,
                    priority,
                    rank,
                    weight,
                    due,
                    constraints: constraints.unwrap_or_default(),
                    work,
                    dev,
                    kind,
                }
            },
        );
    prop_oneof![
        6 => submit,
        4 => (index(), index()).prop_map(|(j, a)| Op::Complete(j, a)),
        2 => (index(), index(), kind).prop_map(|(j, a, k)| Op::Fail(j, a, k)),
        1 => (index(), index(), any::<bool>())
            .prop_map(|(job, attempt, done)| Op::Stale { job, attempt, done }),
        1 => index().prop_map(Op::Cancel),
        2 => (
            0u64..4,
            0usize..5,
            20u64..150,
            0u64..150,
            0u64..60,
            0u8..2,
            prop_oneof![Just(0u64), 20u64..100],
            prop_oneof![Just(0u64), 1u64..30],
        )
            .prop_map(|(id, slots, budget, used, baseline, class, dev_cap, per_task)| {
                Op::Worker { id, slots, budget, used, baseline, class, dev_cap, per_task }
            }),
        1 => (0u64..4).prop_map(Op::Gone),
        3 => (0u32..90).prop_map(Op::Tick),
    ]
}

/// Speed of a worker class in the tests: class 1 is the fast class.
fn class_speed(class: u8) -> f64 {
    if class == 1 { 2.5 } else { 1.0 }
}

/// A random speed configuration.
fn speed() -> impl Strategy<Value = SpeedConfig> {
    let defer = prop::option::of(
        (
            prop_oneof![Just(0.0), Just(0.2)],
            prop_oneof![
                Just(Duration::from_secs(40)),
                Just(Duration::from_secs(500))
            ],
        )
            .prop_map(|(min_gain, max_wait)| Defer { max_wait, min_gain }),
    );
    let speculate = prop::option::weighted(
        0.4,
        (prop_oneof![Just(0.0), Just(0.25)], 1u32..3).prop_map(|(min_gain, max_per_job)| {
            Speculate {
                min_gain,
                restart_overhead: Duration::ZERO,
                max_per_job,
            }
        }),
    );
    (timing(), defer, speculate).prop_map(|(timing, defer, speculate)| SpeedConfig {
        timing,
        defer,
        speculate,
    })
}

/// A random machine model; learning warms up within a stream and is not corrected for
/// concurrency (the model does not track it).
fn timing() -> impl Strategy<Value = Timing> {
    let learn = (
        prop_oneof![Just(0.05), Just(0.5)],
        0u32..3,
        any::<bool>(),
        prop_oneof![Just(0.0), Just(5.0)],
        prop_oneof![Just(0.0), Just(0.1)],
    )
        .prop_map(
            |(weight, min_samples, per_worker, worker_prior, resolution)| Learn {
                weight,
                min_samples,
                per_worker,
                worker_prior,
                resolution,
                sharing: None,
            },
        );
    prop_oneof![
        1 => Just(Timing::Identical),
        1 => Just(Timing::Related { learn: None }),
        1 => learn.clone().prop_map(|l| Timing::Related { learn: Some(l) }),
        2 => (learn, prop_oneof![Just(0.0), Just(5.0)])
            .prop_map(|(learn, kind_prior)| Timing::Unrelated { learn, kind_prior }),
    ]
}

/// A random list-scheduling rule: any order and score terms, in any order, sometimes repeated.
fn rule() -> impl Strategy<Value = Rule> {
    let order = prop::sample::subsequence(
        vec![
            OrderTerm::Priority,
            OrderTerm::Rank,
            OrderTerm::Group,
            OrderTerm::Wspt,
            OrderTerm::Edd,
        ],
        0..=5,
    )
    .prop_shuffle();
    let score = prop::sample::subsequence(
        vec![
            ScoreTerm::Speed,
            ScoreTerm::Tightest,
            ScoreTerm::Loosest,
            ScoreTerm::Preferred,
            ScoreTerm::Load,
        ],
        0..=5,
    )
    .prop_shuffle();
    // Repeat the first term at the end, sometimes.
    let repeat = |mut v: Vec<OrderTerm>, again: bool| {
        if again && let Some(&t) = v.first() {
            v.push(t);
        }
        v
    };
    let age = prop::option::of(prop_oneof![
        Just(Duration::ZERO),
        Just(Duration::from_secs(45)),
        Just(Duration::from_secs(200)),
    ]);
    let reservations = prop::option::weighted(0.7, (0usize..3, any::<bool>()));
    (
        (order, any::<bool>()).prop_map(move |(o, again)| repeat(o, again)),
        prop_oneof![Just(GroupOrder::Arrival), Just(GroupOrder::Id)],
        -1i64..2,
        age,
        reservations,
        score,
    )
        .prop_map(
            |(order, group_order, default_priority, age, reservations, score)| Rule {
                order,
                group_order,
                default_priority,
                age,
                reservations,
                score,
            },
        )
}

/// A random retry limit (0 counts as 1).
fn retry() -> impl Strategy<Value = RetryConfig> {
    (0u32..5).prop_map(|max_attempts| RetryConfig { max_attempts })
}

/// One order term's key as the model compares it: an integer, or a float that may be missing
/// (missing sorts last).
#[derive(Clone, Copy, Debug)]
enum TermKey {
    Int(i128),
    Real(Option<f64>),
}

impl TermKey {
    /// Smaller is more urgent.
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Real(a), Self::Real(b)) => match (a, b) {
                (Some(a), Some(b)) => a.total_cmp(b),
                _ => a.is_none().cmp(&b.is_none()),
            },
            _ => unreachable!("one term, one kind of key"),
        }
    }
}

/// A job's place in the scan: aged jobs first by submission, then the rest by the order terms,
/// then by submission.
#[derive(Clone, Debug)]
struct Urgency {
    aged: bool,
    terms: Vec<TermKey>,
    seq: u64,
}

impl Urgency {
    /// Smaller is more urgent.
    fn cmp(&self, other: &Self) -> Ordering {
        (!self.aged).cmp(&!other.aged).then_with(|| {
            let terms = if self.aged {
                Ordering::Equal
            } else {
                (self.terms.iter().zip(&other.terms))
                    .map(|(a, b)| a.cmp(b))
                    .find(|o| o.is_ne())
                    .unwrap_or(Ordering::Equal)
            };
            terms.then(self.seq.cmp(&other.seq))
        })
    }
}

/// A job as the model sees it, across its attempts.
#[derive(Clone, Debug)]
struct SJob {
    spec: JobSpec,
    since: Time,
    seq: u64,
    /// The last attempt number started.
    attempts: Attempt,
    tried: Vec<Tried>,
    speculated: u32,
}

/// A live attempt.
#[derive(Clone, Debug)]
struct Live {
    attempt: Attempt,
    worker: WorkerId,
    started: Time,
}

/// A running job and its live attempts, in start order.
#[derive(Clone, Debug)]
struct SRun {
    job: SJob,
    live: Vec<Live>,
}

/// A moving mean of `ln speed` samples: plain averaging, then weight `weight` per sample.
#[derive(Clone, Copy, Debug, Default)]
struct Mean {
    mean: f64,
    n: u32,
}

impl Mean {
    /// Add a sample.
    fn add(&mut self, x: f64, weight: f64) {
        let a = weight.max(1.0 / f64::from(self.n + 1));
        self.mean += a * (x - self.mean);
        self.n += 1;
    }

    /// The count a prior is weighed against: `n`, capped at `1 / weight`.
    fn count(&self, weight: f64) -> f64 {
        f64::from(self.n).min(1.0 / weight.max(1e-9))
    }
}

/// The machine model, written out again: per worker its published speed (1 for identical
/// machines, the reported one, or learned per worker with its class as prior, behind a hysteresis
/// band), and per (class, kind) a published factor, the kind's mean shrunk towards the class's.
/// A worker's samples have the kind's current deviation from the class taken out.
#[derive(Debug, Default)]
struct Speeds {
    timing: Timing,
    classes: BTreeMap<String, Mean>,
    /// Per worker: its own samples and the speed last published.
    workers: BTreeMap<WorkerId, (Mean, Option<f64>)>,
    /// Per (class, kind): its samples, and `ln` of the factor last published.
    kinds: BTreeMap<(String, String), (Mean, Option<f64>)>,
    /// The speed each live worker currently has.
    published: BTreeMap<WorkerId, f64>,
}

impl Speeds {
    /// The learning configuration, and the kind prior for unrelated machines.
    fn learn(&self) -> Option<(Learn, Option<f64>)> {
        match self.timing {
            Timing::Identical | Timing::Related { learn: None } => None,
            Timing::Related { learn: Some(l) } => Some((l, None)),
            Timing::Unrelated { learn, kind_prior } => Some((learn, Some(kind_prior))),
        }
    }

    /// The width of the hysteresis band and of a ranking step, in `ln speed`.
    fn band(&self) -> f64 {
        self.learn()
            .map_or(0.0, |(l, _)| l.resolution.max(0.0).ln_1p())
    }

    /// Publish worker `id`'s speed, as on a worker report or after its class learned something.
    fn refresh(&mut self, id: WorkerId, class: &str, reported: f64) {
        let speed = match self.learn() {
            None if self.timing == Timing::Identical => 1.0,
            None => reported,
            Some((l, _)) => {
                let class_log = match self.classes.get(class) {
                    Some(c) if c.n >= l.min_samples => c.mean,
                    _ => reported.ln(),
                };
                let log = match self.workers.get(&id) {
                    Some((own, _)) if l.per_worker && own.n > 0 => {
                        let (n, k) = (own.count(l.weight), l.worker_prior.max(0.0));
                        (n * own.mean + k * class_log) / (n + k)
                    }
                    _ => class_log,
                };
                let est = log.exp();
                let band = self.band();
                let (_, published) = self.workers.entry(id).or_default();
                match *published {
                    Some(p) if (est.ln() - p.ln()).abs() <= band => p,
                    _ => *published.insert(est),
                }
            }
        };
        self.published.insert(id, speed);
    }

    /// `ln` of a kind's factor on a class from its samples.
    fn kind_log(&self, class: &str, kind: &str) -> f64 {
        let (Some((l, Some(prior))), Some((m, _))) = (
            self.learn(),
            self.kinds.get(&(class.to_string(), kind.to_string())),
        ) else {
            return 0.0;
        };
        let n = m.count(l.weight);
        n * (m.mean - self.classes[class].mean) / (n + prior.max(0.0))
    }

    /// Learn from `work` taking `dt` on worker `id` of `class`; returns whether it was a sample,
    /// after which the caller refreshes the class's workers.
    fn observe(
        &mut self,
        id: WorkerId,
        class: &str,
        kind: Option<&str>,
        work: Duration,
        dt: Duration,
    ) -> bool {
        let Some((l, prior)) = self.learn() else {
            return false;
        };
        if work.is_zero() || dt.is_zero() {
            return false;
        }
        let x = (work.as_secs_f64() / dt.as_secs_f64()).ln();
        let kind = kind.filter(|_| prior.is_some());
        let deviation = kind.map_or(0.0, |k| self.kind_log(class, k));
        self.classes
            .entry(class.to_string())
            .or_default()
            .add(x, l.weight);
        self.workers
            .entry(id)
            .or_default()
            .0
            .add(x - deviation, l.weight);
        if let Some(k) = kind {
            (self
                .kinds
                .entry((class.to_string(), k.to_string()))
                .or_default()
                .0)
                .add(x, l.weight);
            let band = self.band();
            let on_class: Vec<String> = (self.kinds.keys())
                .filter(|(c, _)| c == class)
                .map(|(_, k)| k.clone())
                .collect();
            for k in on_class {
                let raw = self.kind_log(class, &k);
                let published = &mut self.kinds.get_mut(&(class.to_string(), k)).unwrap().1;
                if published.is_none_or(|p| (raw - p).abs() > band) {
                    *published = Some(raw);
                }
            }
        }
        true
    }

    /// A job's speed on worker `w` of `class`: the worker's speed times its kind's factor there.
    fn speed(&self, spec: &JobSpec, w: WorkerId, class: &str) -> f64 {
        let factor = match (self.learn(), &spec.kind) {
            (Some((_, Some(_))), Some(k)) => self
                .kinds
                .get(&(class.to_string(), k.clone()))
                .and_then(|(_, p)| *p)
                .map_or(1.0, f64::exp),
            _ => 1.0,
        };
        self.published[&w] * factor
    }

    /// Whether `a` is strictly faster than `b` as the speed term ranks: by whole steps of the
    /// resolution when there is one.
    fn faster(&self, a: f64, b: f64) -> bool {
        let band = self.band();
        if band > 0.0 {
            (a.ln() / band).round() > (b.ln() / band).round()
        } else {
            a > b
        }
    }
}

#[derive(Default)]
struct Shadow {
    speeds: Speeds,
    now: Time,
    workers: BTreeMap<WorkerId, WorkerState>,
    waiting: BTreeMap<JobId, SJob>,
    running: BTreeMap<JobId, SRun>,
    /// The last attempt number ever started per job, kept after the job ends.
    started: BTreeMap<JobId, Attempt>,
    groups: BTreeMap<u64, u64>,
    seq: u64,
    next_id: JobId,
    rule: Option<Rule>,
    max_attempts: u32,
    /// Non-start outputs the next poll must return, in order.
    expect: Vec<Output>,
}

impl Shadow {
    /// Live attempts and their summed demand on a worker.
    fn load(&self, w: WorkerId) -> (usize, Resources) {
        self.running
            .values()
            .flat_map(|r| {
                r.live
                    .iter()
                    .filter(move |l| l.worker == w)
                    .map(move |_| r.job.spec.demand)
            })
            .fold((0, Resources::ZERO), |(n, m), d| (n + 1, m + d))
    }

    /// The production rule, written out again: in every dimension that is enforced (slots
    /// always, memory where its capacity is nonzero), each job counting at least `per_task`;
    /// memory may be exceeded by a job alone on the worker, slots never.
    fn admits(&self, demand: Resources, w: WorkerId) -> bool {
        let s = &self.workers[&w];
        let (running, placed) = self.load(w);
        (0..DIMS).all(|d| {
            let hard = d == SLOTS;
            let cap = if hard { s.slots as u64 } else { s.budget[d] };
            if !hard && cap == 0 {
                return true;
            }
            let held = placed[d].max(running as u64 * s.per_task[d]);
            let used = s.reported_used[d].max(s.reported_baseline[d] + held);
            used + demand[d].max(s.per_task[d]) <= cap || (!hard && running == 0)
        })
    }

    /// The constraints, written out again: no Forbid selects the worker; for workers and for
    /// classes alike, if the job requires any, one of them selects it; Avoids and the workers of
    /// failed attempts are avoided while some live worker passing the hard constraints is free
    /// of them.
    fn eligible(&self, j: &SJob, w: WorkerId) -> bool {
        let selects = |on: &Selector, s: &WorkerState| match on {
            Selector::Worker(id) => *id == s.id,
            Selector::Class(c) => *c == s.class,
        };
        let with = |strength| {
            j.spec
                .constraints
                .iter()
                .filter(move |c| c.strength == strength)
        };
        let hard = |s: &WorkerState| {
            let requires = |class: bool| {
                let mut of_kind = with(Strength::Require)
                    .filter(|c| matches!(c.on, Selector::Class(_)) == class)
                    .peekable();
                of_kind.peek().is_none() || of_kind.any(|c| selects(&c.on, s))
            };
            !with(Strength::Forbid).any(|c| selects(&c.on, s)) && requires(false) && requires(true)
        };
        let avoided = |s: &WorkerState| {
            with(Strength::Avoid).any(|c| selects(&c.on, s))
                || j.tried.iter().any(|t| t.worker == s.id)
        };
        let s = &self.workers[&w];
        hard(s)
            && (!avoided(s)
                || !self
                    .workers
                    .values()
                    .any(|o| o.slots > 0 && hard(o) && !avoided(o)))
    }

    /// Scan order: aged jobs by age, then the configured order terms (each once, at its first
    /// mention), then submission.
    fn urgency(&self, j: &SJob) -> Urgency {
        let rule = self.rule.as_ref().unwrap();
        let mut seen = Vec::new();
        let mut terms = Vec::new();
        for &t in &rule.order {
            if seen.contains(&t) {
                continue;
            }
            seen.push(t);
            let spec = &j.spec;
            terms.push(match t {
                OrderTerm::Priority => {
                    TermKey::Int(spec.priority.unwrap_or(rule.default_priority) as i128)
                }
                OrderTerm::Rank => TermKey::Real(spec.rank.map(|r| -r.as_secs_f64())),
                OrderTerm::Group => TermKey::Int(match rule.group_order {
                    GroupOrder::Id => spec.group as i128,
                    GroupOrder::Arrival => self.groups[&spec.group] as i128,
                }),
                OrderTerm::Wspt => {
                    TermKey::Real(spec.work.map(|w| -(spec.weight / w.as_secs_f64())))
                }
                OrderTerm::Edd => TermKey::Real(spec.due.map(|d| d.0.as_secs_f64())),
            });
        }
        Urgency {
            aged: rule.age.is_some_and(|a| self.now - j.since >= a),
            terms,
            seq: j.seq,
        }
    }

    /// The expected run time of `spec` on worker `w`, rounded to the nanosecond.
    fn eta(&self, spec: &JobSpec, w: WorkerId) -> Option<Duration> {
        Some(secs(spec.work?.as_secs_f64() / self.speed(spec, w)))
    }

    /// `spec`'s speed on worker `w`.
    fn speed(&self, spec: &JobSpec, w: WorkerId) -> f64 {
        self.speeds.speed(spec, w, &self.workers[&w].class)
    }

    /// Live attempt `k` of `job` finished: learn from it, and refresh its class's workers.
    fn learn(&mut self, job: JobId, k: usize) {
        let r = &self.running[&job];
        let l = &r.live[k];
        let Some(work) = r.job.spec.work else {
            return;
        };
        let class = self.workers[&l.worker].class.clone();
        let kind = r.job.spec.kind.as_deref();
        if self
            .speeds
            .observe(l.worker, &class, kind, work, self.now - l.started)
        {
            for (&id, s) in &self.workers {
                if s.class == class {
                    self.speeds.refresh(id, &class, s.speed);
                }
            }
        }
    }

    /// When a running job is expected to end: its earliest live attempt's expected end, an
    /// overrunning attempt counting as half done.
    fn expected_end(&self, r: &SRun) -> Option<Time> {
        r.live
            .iter()
            .map(|l| {
                let end = l.started + self.eta(&r.job.spec, l.worker)?;
                Some(if end > self.now {
                    end
                } else {
                    self.now + (self.now - l.started)
                })
            })
            .reduce(|a, b| Some(a?.min(b?)))
            .flatten()
    }

    /// The `i`-th running job and its `k`-th live attempt, if any job runs.
    fn pick_live(&self, i: usize, k: usize) -> Option<(JobId, usize, Attempt)> {
        if self.running.is_empty() {
            return None;
        }
        let (&j, r) = self.running.iter().nth(i % self.running.len()).unwrap();
        let k = k % r.live.len();
        Some((j, k, r.live[k].attempt))
    }

    /// Submit to both the model and the policy. The model's copy demands one slot.
    fn submit(&mut self, spec: JobSpec, p: &mut dyn Policy) {
        let seq = self.seq;
        self.seq += 1;
        self.groups.entry(spec.group).or_insert(seq);
        let mut model = spec.clone();
        model.demand[SLOTS] = 1;
        self.waiting.insert(
            spec.id,
            SJob {
                spec: model,
                since: self.now,
                seq,
                attempts: 0,
                tried: Vec::new(),
                speculated: 0,
            },
        );
        p.handle(Input::Submit(spec), self.now);
    }

    /// End a running job: every live attempt but `except` is stopped.
    fn stop_all(&mut self, job: JobId, except: Option<Attempt>) {
        let r = self.running.remove(&job).unwrap();
        for l in r.live {
            if Some(l.attempt) != except {
                self.expect.push(Output::Stop {
                    job,
                    attempt: l.attempt,
                    worker: l.worker,
                });
            }
        }
    }

    /// Live attempt `k` of `job` failed: once no attempt is live, the job is retried in its
    /// original place, softly avoiding every worker it failed on, or given up after
    /// `max_attempts` rounds (speculative attempts are not rounds).
    fn fail(&mut self, job: JobId, k: usize, kind: FailKind, why: String) {
        let r = self.running.get_mut(&job).unwrap();
        let l = r.live.remove(k);
        r.job.tried.push(Tried {
            worker: l.worker,
            kind,
            why,
        });
        if !r.live.is_empty() {
            return;
        }
        let j = self.running.remove(&job).unwrap().job;
        if j.attempts - j.speculated < self.max_attempts.max(1) {
            self.waiting.insert(job, j);
        } else {
            let retryable = j.tried.iter().all(|t| t.kind == FailKind::DeviceOom);
            self.expect.push(Output::GaveUp(GaveUp {
                job,
                tried: j.tried,
                retryable,
            }));
        }
    }
}

/// Runs the stream, checking every invariant; returns the outputs and explanations made.
///
/// Every policy is driven by the same random stream while a model, written independently of the
/// engine, keeps its own bookkeeping of jobs and their attempts and re-checks each start as it is
/// made:
///
/// - **messages**: `Done` completes a job and stops its other attempts, `Failed` and `WorkerGone`
///   retry it in its original place (softly avoiding where it failed) or give it up, `Cancel`
///   stops every live attempt, and reports about attempts that are not live change nothing --
///   every non-start output is exactly what the model predicts;
/// - **attempts**: each start is the job's next attempt number; a job has two live attempts only
///   by speculation, and a job with a live attempt is never also waiting;
/// - **no over-commit**: the production admission rule held at the moment of each start, in every
///   dimension (no slot over-used; no known memory capacity exceeded, escape hatch aside);
/// - **constraints**: no attempt runs on a worker its Requires or Forbids exclude, nor on an
///   avoided one while some live worker they allow is not avoided;
/// - **escape hatch**: after a poll, no worker with a free slot is empty while a job that may run
///   there waits;
/// - **priority** (the configured order): when B is placed on w, every more urgent waiting job was
///   refused by w at that moment (by the admission rule, or because w was B's reservation, or
///   because it chose to wait for a faster worker);
/// - **speculation**: a second attempt goes to a strictly faster, unreserved worker that no
///   waiting job wanted, is expected to gain at least `min_gain` of its run time, and respects
///   `max_per_job`;
/// - **bookkeeping**: per-worker slots and placed demand equal the sum over live attempts, job
///   counts and reservations agree with the model after every event, and everything is released
///   at the end;
/// - **determinism**: replaying the stream gives identical outputs and explanations.
fn run(
    rule: &Rule,
    speed: SpeedConfig,
    retry: RetryConfig,
    ops: &[Op],
) -> Result<Vec<String>, TestCaseError> {
    let mut p = rule.build(speed, retry);
    let mut sh = Shadow {
        rule: Some(rule.clone()),
        max_attempts: retry.max_attempts,
        speeds: Speeds {
            timing: speed.timing,
            ..Speeds::default()
        },
        ..Shadow::default()
    };
    let mut log = Vec::new();
    for op in ops {
        match *op {
            Op::Submit {
                demand,
                slots,
                group,
                priority,
                rank,
                weight,
                due,
                ref constraints,
                work,
                dev,
                kind,
            } => {
                let id = sh.next_id;
                sh.next_id += 1;
                let spec = JobSpec {
                    id,
                    demand: Resources::mem(demand)
                        .with_dev(dev.unwrap_or(0))
                        .with_slots(slots),
                    group,
                    priority,
                    rank: rank.map(|r| Duration::from_secs(r.into())),
                    weight,
                    due: due.map(|d| Time(Duration::from_secs(d.into()))),
                    constraints: constraints.clone(),
                    work: work.map(|w| Duration::from_secs(w.into())),
                    kind: kind.map(|k| format!("k{k}")),
                };
                sh.submit(spec, &mut *p);
            }
            Op::Complete(i, k) => {
                if let Some((job, k, attempt)) = sh.pick_live(i, k) {
                    sh.learn(job, k);
                    sh.stop_all(job, Some(attempt));
                    p.handle(Input::Done { job, attempt }, sh.now);
                }
            }
            Op::Fail(i, k, kind) => {
                if let Some((job, k, attempt)) = sh.pick_live(i, k) {
                    let why = format!("failed at {:?}", sh.now.0);
                    sh.fail(job, k, kind, why.clone());
                    p.handle(
                        Input::Failed {
                            job,
                            attempt,
                            kind,
                            why,
                        },
                        sh.now,
                    );
                }
            }
            Op::Stale { job, attempt, done } => {
                if sh.next_id > 0 {
                    let job = job as JobId % sh.next_id;
                    let live: Vec<Attempt> = sh
                        .running
                        .get(&job)
                        .map(|r| r.live.iter().map(|l| l.attempt).collect())
                        .unwrap_or_default();
                    // Attempt 0 and the next attempt number were never started.
                    let last = sh.started.get(&job).copied().unwrap_or(0);
                    let stale: Vec<Attempt> =
                        (0..=last + 1).filter(|a| !live.contains(a)).collect();
                    let attempt = stale[attempt % stale.len()];
                    p.handle(
                        if done {
                            Input::Done { job, attempt }
                        } else {
                            Input::Failed {
                                job,
                                attempt,
                                kind: FailKind::Other,
                                why: "stale".into(),
                            }
                        },
                        sh.now,
                    );
                }
            }
            Op::Cancel(i) => {
                let live: Vec<JobId> = sh
                    .running
                    .keys()
                    .chain(sh.waiting.keys())
                    .copied()
                    .collect();
                if !live.is_empty() {
                    let j = live[i % live.len()];
                    if sh.running.contains_key(&j) {
                        sh.stop_all(j, None);
                    }
                    sh.waiting.remove(&j);
                    p.handle(Input::Cancel(j), sh.now);
                }
            }
            Op::Worker {
                id,
                slots,
                budget,
                used,
                baseline,
                class,
                dev_cap,
                per_task,
            } => {
                let s = WorkerState {
                    id,
                    class: format!("c{class}"),
                    slots,
                    budget: Resources::mem(budget).with_dev(dev_cap),
                    reported_used: Resources::mem(used),
                    reported_baseline: Resources::mem(baseline),
                    speed: class_speed(class),
                    per_task: Resources::ZERO.with_dev(per_task),
                };
                sh.speeds.refresh(id, &s.class, s.speed);
                sh.workers.insert(id, s.clone());
                p.handle(Input::Worker(s), sh.now);
            }
            Op::Gone(w) => {
                if sh.workers.remove(&w).is_some() {
                    sh.speeds.published.remove(&w);
                    // Each live attempt there fails; the caller resubmits nothing.
                    let lost: Vec<(JobId, usize)> = sh
                        .running
                        .iter()
                        .filter_map(|(&j, r)| Some((j, r.live.iter().position(|l| l.worker == w)?)))
                        .collect();
                    for (j, k) in lost {
                        sh.fail(j, k, FailKind::LinkDied, format!("worker {w} left"));
                    }
                }
                p.handle(Input::WorkerGone(w), sh.now);
            }
            Op::Tick(dt) => sh.now += Duration::from_secs(dt.into()),
        }

        let before = p.stats();
        let out = p.poll(sh.now);
        let after = p.stats();
        if std::env::var_os("SCHED_TRACE").is_some() {
            eprintln!("op {op:?} -> {out:?} deferred {:?}", after.deferred);
            for j in sh.waiting.keys() {
                eprintln!("  {}", p.explain(*j).unwrap());
            }
        }
        let mut starts = Vec::new();
        let mut other = Vec::new();
        for o in &out {
            match *o {
                Output::Start {
                    job,
                    attempt,
                    worker,
                } => starts.push((job, attempt, worker)),
                _ => other.push(o.clone()),
            }
        }
        prop_assert_eq!(&other, &std::mem::take(&mut sh.expect), "after {:?}", op);
        let holders: BTreeSet<JobId> = after.last_dispatch_holders.iter().copied().collect();
        let deferred: BTreeSet<JobId> = after.deferred.iter().map(|d| d.0).collect();
        let deferred_any: BTreeSet<JobId> = after.deferred_any.iter().copied().collect();
        prop_assert!(deferred.is_subset(&deferred_any));
        // Deferral is only for jobs with work, within their waiting window, onto a full, faster
        // worker; and its expiry is a wakeup.
        for &(j, w, at) in &after.deferred {
            let job = &sh.waiting[&j];
            let Some(d) = speed.defer else {
                prop_assert!(false, "deferral without a Defer config");
                unreachable!()
            };
            prop_assert!(job.spec.work.is_some() && sh.now - job.since < d.max_wait);
            prop_assert!(
                !rule.age.is_some_and(|a| sh.now - job.since >= a),
                "aged job deferred"
            );
            prop_assert!(at >= sh.now);
            prop_assert!(sh.workers.contains_key(&w));
            prop_assert!(
                p.next_wakeup()
                    .is_some_and(|t| t > sh.now && t <= job.since + d.max_wait)
            );
        }
        if let Some(t) = p.next_wakeup() {
            prop_assert!(t > sh.now, "wakeup {:?} not in the future", t);
        }
        for &(j, attempt, w) in &starts {
            prop_assert!(sh.workers.contains_key(&w), "started on unknown worker {w}");
            if let Some(job) = sh.waiting.get(&j).cloned() {
                prop_assert_eq!(attempt, job.attempts + 1, "job {} attempt", j);
                prop_assert!(
                    sh.eligible(&job, w),
                    "{rule:?}: job {j} placed on excluded worker {w}"
                );
                // No over-commit (slots included).
                prop_assert!(
                    sh.admits(job.spec.demand, w),
                    "{rule:?}: job {j} over-commits worker {w}"
                );
                // Priority: every more urgent waiting job is refused here (unless this is a holder
                // taking its own reserved worker, which nobody else could take).
                if !holders.contains(&j) {
                    let mine = sh.urgency(&job);
                    for a in sh.waiting.values() {
                        if a.spec.id != j && sh.urgency(a).cmp(&mine).is_lt() {
                            let refused = deferred_any.contains(&a.spec.id)
                                || !sh.eligible(a, w)
                                || !sh.admits(a.spec.demand, w);
                            prop_assert!(
                                refused,
                                "{rule:?}: job {j} placed on {w} while more urgent job {} is \
                                 admitted there",
                                a.spec.id
                            );
                        }
                    }
                }
                // Speed first: no other worker that would take the job is faster for it. Only
                // checked in polls that made no reservation (one made and dropped within a poll
                // leaves no trace in the stats); a reservation the poll started with counts as
                // keeping its worker throughout.
                if rule.score.first() == Some(&ScoreTerm::Speed)
                    && after.reservations_total == before.reservations_total
                {
                    let mine = sh.speed(&job.spec, w);
                    for &v in sh.workers.keys() {
                        let reserved =
                            (before.reservations.iter()).any(|r| r.worker == v && r.job != j);
                        if v != w
                            && !reserved
                            && sh.eligible(&job, v)
                            && sh.admits(job.spec.demand, v)
                        {
                            prop_assert!(
                                !sh.speeds.faster(sh.speed(&job.spec, v), mine),
                                "{rule:?}: job {j} placed on {w} while {v} is faster for it"
                            );
                        }
                    }
                }
                sh.waiting.remove(&j);
                sh.running.insert(
                    j,
                    SRun {
                        job: SJob {
                            attempts: attempt,
                            ..job
                        },
                        live: vec![Live {
                            attempt,
                            worker: w,
                            started: sh.now,
                        }],
                    },
                );
            } else if let Some(r) = sh.running.get(&j) {
                // A second live attempt: only by speculation, onto a strictly faster worker that
                // admits it, whose free slot no waiting job wanted, at most `max_per_job` times,
                // when it gains enough.
                let Some(cfg) = speed.speculate else {
                    prop_assert!(false, "job {j} has two live attempts without speculation");
                    unreachable!()
                };
                prop_assert_eq!(attempt, r.job.attempts + 1, "job {} attempt", j);
                prop_assert!(
                    r.job.speculated < cfg.max_per_job,
                    "job {j} over-speculated"
                );
                let spec = r.job.spec.clone();
                let sjob = r.job.clone();
                let here = sh.speed(&spec, w);
                prop_assert!(
                    r.live
                        .iter()
                        .all(|l| sh.speeds.faster(here, sh.speed(&spec, l.worker))),
                    "job {j} speculated onto {w}, not faster for it than {:?}",
                    r.live
                );
                prop_assert!(
                    sh.eligible(&sjob, w) && sh.admits(spec.demand, w),
                    "job {j} speculated onto {w}, which does not take it"
                );
                prop_assert!(
                    !after.reservations.iter().any(|r| r.worker == w),
                    "speculated onto a reserved worker"
                );
                let wanted = sh.waiting.values().find(|a| {
                    !deferred.contains(&a.spec.id)
                        && sh.eligible(a, w)
                        && sh.admits(a.spec.demand, w)
                });
                prop_assert!(
                    wanted.is_none(),
                    "speculated onto {w} while job {:?} waits for it",
                    wanted.map(|a| a.spec.id)
                );
                let end = sh.expected_end(r);
                let run = sh.eta(&spec, w);
                prop_assert!(
                    end.is_some() && run.is_some(),
                    "speculated without a run time"
                );
                let (end, run) = (end.unwrap(), run.unwrap());
                let end_here = sh.now + run + cfg.restart_overhead;
                prop_assert!(
                    end > end_here && end - end_here >= secs(run.as_secs_f64() * cfg.min_gain),
                    "job {j} speculated onto {w} for too little: {end:?} vs {end_here:?}"
                );
                let r = sh.running.get_mut(&j).unwrap();
                r.job.attempts = attempt;
                r.job.speculated += 1;
                r.live.push(Live {
                    attempt,
                    worker: w,
                    started: sh.now,
                });
            } else {
                prop_assert!(
                    false,
                    "started job {j}, which is neither waiting nor running"
                );
            }
            sh.started.insert(j, attempt);
        }
        let held: BTreeMap<JobId, WorkerId> = after
            .reservations
            .iter()
            .map(|r| (r.job, r.worker))
            .collect();
        // Escape hatch: an empty worker with a free slot leaves no job waiting that may run there,
        // except one waiting for a faster worker by choice.
        for (&w, s) in &sh.workers {
            if s.slots > 0 && sh.load(w).0 == 0 {
                let stuck = sh
                    .waiting
                    .values()
                    .find(|j| sh.eligible(j, w) && !deferred.contains(&j.spec.id));
                prop_assert!(
                    stuck.is_none(),
                    "{rule:?}: worker {w} empty while job {:?} waits",
                    stuck.map(|j| j.spec.id)
                );
            }
        }
        // Bookkeeping agrees with the model: job counts, live attempts per job, per-worker slots
        // and placed demand as the sum over live attempts, and per-worker speeds.
        prop_assert_eq!(after.waiting, sh.waiting.len());
        prop_assert_eq!(after.running, sh.running.len());
        prop_assert_eq!(after.workers.len(), sh.workers.len());
        for l in &after.workers {
            let (n, m) = sh.load(l.id);
            prop_assert_eq!((l.running, l.placed), (n, m), "worker {} load", l.id);
            prop_assert_eq!(l.speed, sh.speeds.published[&l.id], "worker {} speed", l.id);
        }
        for (&j, r) in &sh.running {
            prop_assert!(
                speed.speculate.is_some() || r.live.len() == 1,
                "job {j} has live attempts {:?}",
                r.live
            );
            let attempts = r.live.iter().map(|l| (l.attempt, l.worker)).collect();
            prop_assert_eq!(
                p.explain(j),
                Some(Explanation::new(j, Status::Running { attempts }))
            );
        }
        for &j in sh.waiting.keys() {
            let e = p.explain(j);
            prop_assert!(
                e.as_ref().is_some_and(|e| e.waiting().is_some()),
                "waiting job {j} explained as {e:?}"
            );
        }
        let (max_res, per_class) = rule.reservations.unwrap_or((0, false));
        let mut per: BTreeMap<String, usize> = BTreeMap::new();
        for (&job, &worker) in &held {
            prop_assert!(
                sh.waiting.contains_key(&job),
                "holder {} is not waiting",
                job
            );
            let l = after.workers.iter().find(|l| l.id == worker);
            prop_assert!(l.is_some_and(|l| l.reserved_for == Some(job)));
            *per.entry(if per_class {
                sh.workers[&worker].class.clone()
            } else {
                String::new()
            })
            .or_default() += 1;
        }
        prop_assert!(
            per.values().all(|&n| n <= max_res),
            "too many reservations: {per:?}"
        );
        prop_assert!(before.placements_total + starts.len() as u64 == after.placements_total);
        log.push(format!(
            "{out:?} {:?} {:?}",
            after.deferred,
            p.next_wakeup()
        ));
        for id in 0..sh.next_id {
            log.push(format!("{:?}", p.explain(id)));
        }
    }
    // Liveness of bookkeeping: completing and cancelling everything releases everything.
    for (j, r) in std::mem::take(&mut sh.running) {
        p.handle(
            Input::Done {
                job: j,
                attempt: r.live[0].attempt,
            },
            sh.now,
        );
    }
    for j in sh.waiting.keys() {
        p.handle(Input::Cancel(*j), sh.now);
    }
    let end = p.stats();
    prop_assert_eq!(end.waiting + end.running, 0);
    prop_assert!(end.reservations.is_empty());
    prop_assert!(
        end.workers
            .iter()
            .all(|l| l.running == 0 && l.placed == Resources::ZERO && l.reserved_for.is_none())
    );
    Ok(log)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(768))]

    /// Every invariant holds on random streams, and replays are identical.
    #[test]
    fn invariants_hold(
        rule in rule(),
        speed in speed(),
        retry in retry(),
        ops in prop::collection::vec(op(), 1..160),
    ) {
        let first = run(&rule, speed, retry, &ops)?;
        let second = run(&rule, speed, retry, &ops)?;
        prop_assert_eq!(first, second, "not deterministic");
    }
}
