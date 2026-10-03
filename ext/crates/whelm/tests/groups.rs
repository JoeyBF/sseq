//! Group order: restart-stable ordering by id, and aging behind a wide old group.

use whelm::{
    Config, DEFAULT_AGE_LIMIT, GroupOrder, Input, JobId, JobSpec, Output, Policy, Resources,
    Scheduler, WorkerId, WorkerState, nassau,
};

/// The `(job, worker)` of each start in `out`.
fn starts(out: Vec<Output>) -> Vec<(JobId, WorkerId)> {
    out.into_iter()
        .filter_map(|o| match o {
            Output::Start { job, worker, .. } => Some((job, worker)),
            _ => None,
        })
        .collect()
}

/// The order in which a one-slot worker runs jobs submitted in `order` (group = job id here).
fn run_order(order: &[u64], group_order: GroupOrder) -> Vec<u64> {
    let mut p = Scheduler::new(Config {
        group_order,
        ..Config::default()
    });
    for (i, &g) in order.iter().enumerate() {
        p.handle(
            Input::Submit(JobSpec::new(g, Resources::mem(1), g)),
            i as f64,
        );
    }
    p.handle(
        Input::Worker(WorkerState::new(0, "x", 1, Resources::mem(100))),
        10.0,
    );
    let mut ran = Vec::new();
    for t in 0..order.len() {
        let out = starts(p.poll(10.0 + t as f64));
        assert_eq!(out.len(), 1);
        ran.push(out[0].0);
        p.handle(
            Input::Done {
                job: out[0].0,
                attempt: 1,
            },
            10.5 + t as f64,
        );
    }
    ran
}

/// By id, the order does not depend on submission order (a restarted coordinator resubmitting
/// in another order runs the same groups first); by arrival it does.
#[test]
fn group_order_by_id_ignores_submission_order() {
    let ids = [
        nassau::group(2, 30),
        nassau::group(0, 31),
        nassau::group(1, 31),
        nassau::group(5, 12),
    ];
    let a = run_order(&ids, GroupOrder::Id);
    let mut rev = ids;
    rev.reverse();
    let b = run_order(&rev, GroupOrder::Id);
    assert_eq!(a, b);
    let mut sorted = ids.to_vec();
    sorted.sort();
    assert_eq!(a, sorted);
    assert_ne!(
        run_order(&ids, GroupOrder::Arrival),
        run_order(&rev, GroupOrder::Arrival)
    );
}

/// The incident of 2026-10-02: a young group behind a wide old one whose walk keeps releasing
/// jobs. Strict group order makes the young job wait for the whole old group; aging (on by
/// default) bounds its wait by the age limit plus one old job's run time. (Aged jobs run in
/// submission order, so aging bounds the wait behind work released after the job, not behind
/// work already queued before it.)
#[test]
fn young_group_behind_wide_old_group_waits_at_most_age_limit() {
    const SLOTS: usize = 4;
    const OLD_JOBS: u64 = 1000;
    const RUN: f64 = 100.0;
    const YOUNG: u64 = 1_000_000;
    let wait = |policy: &mut dyn Policy| -> f64 {
        policy.handle(
            Input::Worker(WorkerState::new(0, "x", SLOTS, Resources::mem(100))),
            0.0,
        );
        let mut released = 0;
        let mut running: Vec<(u64, f64)> = Vec::new();
        let mut t = 0.0;
        loop {
            // The old walk keeps two jobs per slot ready.
            while released < OLD_JOBS && policy.stats().waiting < 2 * SLOTS {
                policy.handle(
                    Input::Submit(JobSpec::new(
                        released,
                        Resources::mem(1),
                        nassau::group(1, 20),
                    )),
                    t,
                );
                released += 1;
            }
            if t == 10.0 {
                policy.handle(
                    Input::Submit(JobSpec::new(YOUNG, Resources::mem(1), nassau::group(3, 40))),
                    t,
                );
            }
            running.retain(|&(j, end)| {
                if end <= t {
                    policy.handle(Input::Done { job: j, attempt: 1 }, t);
                    false
                } else {
                    true
                }
            });
            for (j, _) in starts(policy.poll(t)) {
                if j == YOUNG {
                    return t - 10.0;
                }
                running.push((j, t + RUN));
            }
            t += 1.0;
            assert!(t < 1e6, "the young job never ran");
        }
    };
    let by_id = |base: Config, age_limit| Config {
        group_order: GroupOrder::Id,
        age_limit,
        ..base
    };
    let bound = DEFAULT_AGE_LIMIT + RUN + 1.0;
    let w = wait(&mut Scheduler::new(by_id(
        Config::default(),
        Some(DEFAULT_AGE_LIMIT),
    )));
    assert!(w <= bound, "backfill: {w}");
    let w = wait(&mut Scheduler::new(by_id(
        Config::best_fit(),
        Some(DEFAULT_AGE_LIMIT),
    )));
    assert!(w <= bound, "best fit: {w}");
    // The default configuration ages.
    assert_eq!(Config::default().age_limit, Some(DEFAULT_AGE_LIMIT));
    // Control: strict priority waits for the whole old group.
    let w = wait(&mut Scheduler::new(by_id(Config::default(), None)));
    assert!(
        w >= (OLD_JOBS as f64 / SLOTS as f64 - 1.0) * RUN,
        "strict: {w}"
    );
}
