//! The service-time model: processor sharing per worker.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use super::trace::Trace;

/// Total throughput of a worker as a function of its concurrency. Swap it to test other models.
pub trait ServiceModel: Sync {
    /// Work per second delivered by a worker of `class` running `k >= 1` jobs.
    fn throughput(&self, class: &str, k: usize) -> f64;
}

/// One class's throughput curve.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ClassCurve {
    /// Throughput of one job alone, relative to the reference class.
    pub speed: f64,
    /// Concurrency beyond which throughput stops growing.
    pub k_sat: usize,
    /// Growth exponent below saturation (0: no gain from concurrency; 1: linear).
    pub alpha: f64,
}

impl ClassCurve {
    /// `min(k, k_sat)^alpha` (without the speed).
    pub fn shape(&self, k: usize) -> f64 {
        (k.min(self.k_sat).max(1) as f64).powf(self.alpha)
    }

    /// `speed * shape(k)`.
    pub fn throughput(&self, k: usize) -> f64 {
        self.speed * self.shape(k)
    }
}

/// The saturating processor-sharing model, one curve per class.
///
/// Each job carries a work amount `W`. A worker running `k` jobs delivers a total throughput
/// `f_class(k)` (work per second), split equally, so each job progresses at `f_class(k) / k`.
/// The shape fitted here is
///
/// ```text
/// f_class(k) = speed_class * min(k, k_sat_class) ^ alpha_class
/// ```
///
/// normalised so that the reference class (the first class in name order, `h200` in our traces)
/// has `speed = 1`: one unit of work is one second of a reference worker running that job alone.
#[derive(Clone, Debug, Serialize)]
pub struct PsModel {
    /// Curves by class.
    pub classes: BTreeMap<String, ClassCurve>,
}

impl ServiceModel for PsModel {
    /// The class's fitted curve; an unknown class gets linear scaling at reference speed.
    fn throughput(&self, class: &str, k: usize) -> f64 {
        self.classes
            .get(class)
            .map_or(k as f64, |c| c.throughput(k))
    }
}

/// What the fit found, for the report.
#[derive(Clone, Debug, Serialize)]
pub struct FitReport {
    /// The reference class (speed 1).
    pub reference: String,
    /// Fitted curves.
    pub classes: BTreeMap<String, ClassCurve>,
    /// Jobs used.
    pub n: usize,
    /// Fixed effects (bidegree x kind).
    pub groups: usize,
    /// Coefficients of `ln target`, `ln(next+1)`, `ln est_gb`.
    pub beta: [f64; 3],
    /// R^2 of `ln s` explained, after removing the group means (within R^2).
    pub r2_within: f64,
    /// Residual standard deviation of `ln s` (so `exp` of it is the typical multiplicative error).
    pub resid_sd: f64,
    /// Residual sum of squares of the best `(k_sat, alpha)` for each class, and for the "no
    /// sharing effect" model (`alpha = 0`), to show how much concurrency explains.
    pub sse_best: f64,
    /// See `sse_best`.
    pub sse_alpha0: f64,
}

/// Per-job time spent at each concurrency level during its production run.
struct Occupancy {
    /// `hist[j][k]` = seconds job `j` ran while its worker ran `k` jobs.
    hist: Vec<Vec<f64>>,
}

impl Occupancy {
    /// Reconstruct each worker's exact concurrency from the trace's placements.
    fn new(trace: &Trace, kmax: usize) -> Self {
        let mut hist = vec![vec![0.0; kmax + 1]; trace.tasks.len()];
        let mut by_worker: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, t) in trace.tasks.iter().enumerate() {
            by_worker.entry(t.worker).or_default().push(i);
        }
        for tasks in by_worker.values() {
            // Concurrency step function: +1 at placement, -1 at completion (ends first at ties).
            let mut ev: Vec<(f64, i32)> = Vec::with_capacity(2 * tasks.len());
            for &i in tasks {
                ev.push((trace.tasks[i].placed_s, 1));
                ev.push((trace.tasks[i].done_s, -1));
            }
            ev.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            // cum[i][k] = time at concurrency k before event i.
            let mut times = Vec::with_capacity(ev.len());
            let mut cum: Vec<Vec<f64>> = Vec::with_capacity(ev.len());
            let mut acc = vec![0.0; kmax + 1];
            let (mut k, mut last) = (0i64, ev.first().map_or(0.0, |e| e.0));
            for &(t, d) in &ev {
                let kk = (k.max(0) as usize).min(kmax);
                acc[kk] += t - last;
                last = t;
                times.push(t);
                cum.push(acc.clone());
                k += d as i64;
            }
            let at = |t: f64| -> &Vec<f64> {
                // Any event at time t has the same cumulative vector.
                let i = times.partition_point(|&x| x < t);
                &cum[i.min(cum.len() - 1)]
            };
            for &i in tasks {
                let (a, b) = (at(trace.tasks[i].placed_s), at(trace.tasks[i].done_s));
                for k in 0..=kmax {
                    hist[i][k] = (b[k] - a[k]).max(0.0);
                }
                // A job is running during its own run, so k >= 1.
                let zero = hist[i][0];
                hist[i][1] += zero;
                hist[i][0] = 0.0;
            }
        }
        Self { hist }
    }
}

/// Solve the normal equations `A x = b` (small, symmetric) by Gaussian elimination with partial
/// pivoting. Singular directions get 0.
pub(crate) fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, p);
        b.swap(c, p);
        if a[c][c].abs() < 1e-12 {
            continue;
        }
        let pivot = a[c].clone();
        for r in 0..n {
            if r != c {
                let f = a[r][c] / pivot[c];
                if f != 0.0 {
                    for (x, p) in a[r][c..].iter_mut().zip(&pivot[c..]) {
                        *x -= f * p;
                    }
                    b[r] -= f * b[c];
                }
            }
        }
    }
    (0..n)
        .map(|i| {
            if a[i][i].abs() < 1e-12 {
                0.0
            } else {
                b[i] / a[i][i]
            }
        })
        .collect()
}

struct Design {
    rows: Vec<usize>,
    ln_s: Vec<f64>,
    /// Group-demeaned covariates: 3 size covariates + one dummy per non-reference class.
    x: Vec<Vec<f64>>,
    group: Vec<usize>,
    ngroups: usize,
    class: Vec<usize>,
}

/// Fit the model to a trace. Returns the model, the report, and each task's work `W`.
///
/// In production a job's service time is `s = ∫ k(t) / f(k(t)) dt`-weighted work, i.e.
/// `ln s = ln W + ln mean_run(k / g(k)) - ln speed` with `g(k) = min(k, k_sat)^alpha` and `k(t)`
/// the exact concurrency on its worker (reconstructed from the trace's placements). `ln W` is
/// modelled as a per-(bidegree, kind) fixed effect plus `ln target`, `ln(next + 1)`, `ln est_gb`;
/// `ln speed` as a class dummy. For each candidate `(k_sat, alpha)` per class the regression is
/// ordinary least squares after the within-group transform; the candidate with the smallest
/// residual sum of squares wins (coordinate descent over classes). Each job's `W` is then the
/// integral of its own observed progress rate, `W = ∫ f(k(t)) / k(t) dt` over its production run,
/// so replaying production's placements reproduces production's service times exactly.
pub fn fit(trace: &Trace) -> (PsModel, FitReport, Vec<f64>) {
    let mut class_names: Vec<String> = trace.workers.iter().map(|w| w.class.clone()).collect();
    class_names.sort();
    class_names.dedup();
    let kmax = trace
        .workers
        .iter()
        .map(|w| w.slots)
        .max()
        .unwrap_or(1)
        .max(1);
    let occ = Occupancy::new(trace, kmax);
    let class_of = |w: usize| class_names.binary_search(&trace.workers[w].class).unwrap();

    // Rows: jobs with a meaningful service time.
    let mut gid: HashMap<(u64, bool), usize> = HashMap::new();
    let mut d = Design {
        rows: Vec::new(),
        ln_s: Vec::new(),
        x: Vec::new(),
        group: Vec::new(),
        ngroups: 0,
        class: Vec::new(),
    };
    let nc = class_names.len();
    for (i, t) in trace.tasks.iter().enumerate() {
        let s = t.done_s - t.placed_s;
        if s <= 1.0 || t.target <= 0.0 || t.est_gb <= 0.0 {
            continue;
        }
        let n = gid.len();
        let g = *gid.entry((t.group, t.zero)).or_insert(n);
        let c = class_of(t.worker);
        let mut x = vec![t.target.ln(), (t.next + 1.0).ln(), t.est_gb.ln()];
        x.extend((1..nc).map(|k| f64::from(u8::from(k == c))));
        d.rows.push(i);
        d.ln_s.push(s.ln());
        d.x.push(x);
        d.group.push(g);
        d.class.push(c);
    }
    d.ngroups = gid.len();
    let p = 3 + nc - 1;
    // Within-transform the covariates once (they do not depend on the curve parameters).
    let demean = |v: &mut Vec<f64>, d: &Design, col: Option<usize>| {
        let mut sum = vec![0.0; d.ngroups];
        let mut cnt = vec![0.0; d.ngroups];
        for (r, &g) in d.group.iter().enumerate() {
            sum[g] += col.map_or(v[r], |c| d.x[r][c]);
            cnt[g] += 1.0;
        }
        for (r, &g) in d.group.iter().enumerate() {
            v[r] -= sum[g] / cnt[g];
        }
    };
    let mut xw: Vec<Vec<f64>> = vec![vec![0.0; p]; d.rows.len()];
    for c in 0..p {
        let mut col: Vec<f64> = d.x.iter().map(|x| x[c]).collect();
        demean(&mut col, &d, None);
        for (r, v) in col.into_iter().enumerate() {
            xw[r][c] = v;
        }
    }
    let mut xtx = vec![vec![0.0; p]; p];
    for x in &xw {
        for a in 0..p {
            for b in 0..p {
                xtx[a][b] += x[a] * x[b];
            }
        }
    }

    // y(curves) = ln s - ln mean_run(k / g(k)); regress within groups; return (sse, coef, sst).
    let regress = |curves: &[ClassCurve]| -> (f64, Vec<f64>, f64) {
        let k_over_g: Vec<Vec<f64>> = curves
            .iter()
            .map(|c| (0..=kmax).map(|k| k as f64 / c.shape(k)).collect())
            .collect();
        let mut y: Vec<f64> = d
            .rows
            .iter()
            .enumerate()
            .map(|(r, &i)| {
                let kg = &k_over_g[d.class[r]];
                let h = &occ.hist[i];
                let tot: f64 = h.iter().sum();
                let m: f64 = (1..=kmax).map(|k| h[k] * kg[k]).sum::<f64>() / tot;
                d.ln_s[r] - m.ln()
            })
            .collect();
        demean(&mut y, &d, None);
        let mut xty = vec![0.0; p];
        for (x, &yy) in xw.iter().zip(&y) {
            for a in 0..p {
                xty[a] += x[a] * yy;
            }
        }
        let beta = solve(xtx.clone(), xty);
        let mut sse = 0.0;
        let mut sst = 0.0;
        for (x, &yy) in xw.iter().zip(&y) {
            let fit: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum();
            sse += (yy - fit).powi(2);
            sst += yy * yy;
        }
        (sse, beta, sst)
    };

    let alphas: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
    let mut curves = vec![
        ClassCurve {
            speed: 1.0,
            k_sat: kmax,
            alpha: 0.5
        };
        nc
    ];
    let mut best = regress(&curves).0;
    for _sweep in 0..3 {
        let before = best;
        for c in 0..nc {
            for k_sat in 1..=kmax {
                for &alpha in &alphas {
                    let mut trial = curves.clone();
                    trial[c] = ClassCurve {
                        speed: 1.0,
                        k_sat,
                        alpha,
                    };
                    let sse = regress(&trial).0;
                    if sse < best - 1e-9 {
                        best = sse;
                        curves = trial;
                    }
                }
            }
        }
        if best >= before - 1e-9 {
            break;
        }
    }
    let (sse, beta, sst) = regress(&curves);
    let sse_alpha0 = regress(&vec![
        ClassCurve {
            speed: 1.0,
            k_sat: kmax,
            alpha: 0.0
        };
        nc
    ])
    .0;
    // ln s = ... - ln speed, so the class dummy's coefficient is -ln(speed / speed_ref).
    for c in 1..nc {
        curves[c].speed = (-beta[3 + c - 1]).exp();
    }
    let classes: BTreeMap<String, ClassCurve> = class_names
        .iter()
        .cloned()
        .zip(curves.iter().copied())
        .collect();
    let model = PsModel {
        classes: classes.clone(),
    };

    // Each job's work: the integral of its observed progress rate f(k)/k over its run.
    let work: Vec<f64> = trace
        .tasks
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let c = &curves[class_of(t.worker)];
            let h = &occ.hist[i];
            let w: f64 = (1..=kmax).map(|k| h[k] * c.throughput(k) / k as f64).sum();
            w.max(1e-3)
        })
        .collect();

    let n = d.rows.len();
    let report = FitReport {
        reference: class_names.first().cloned().unwrap_or_default(),
        classes,
        n,
        groups: d.ngroups,
        beta: [beta[0], beta[1], beta[2]],
        r2_within: 1.0 - sse / sst,
        resid_sd: (sse / n.max(1) as f64).sqrt(),
        sse_best: sse,
        sse_alpha0,
    };
    (model, report, work)
}
