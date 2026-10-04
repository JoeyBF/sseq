//! The whole-run DAG of a Nassau resolution, its cost model, and its simulation.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::Path,
    sync::Arc,
    time::Duration,
};

use serde::Serialize;
use whelm::{
    Config, Constraint, DagConfig, DagJob, DagScheduler, DagTemplate, GroupOrder, Input, JobId,
    JobSpec, MEMORY, NodeSource, OrderTerm, Output, Policy, Resources, SLOTS, Scheduler, Selector,
    Strength, Time, Unit, WorkerState,
};

use crate::{
    algebra,
    engine::{PsWorker, Queue},
    model::{ServiceModel, solve},
    plan::SpeedPlan,
    run::Quantiles,
    trace::Trace,
};

/// One census row (`ext::nassau` per-bidegree counters).
#[derive(Clone, Copy, Debug)]
pub struct CensusRow {
    /// Dimension of the zero step's masked target.
    pub target_masked_dim: f64,
    /// Dimension of the next module.
    pub next_dim: f64,
    /// Generators found.
    pub gens: u32,
    /// Signatures the subalgebra has at this degree (non-zero ones).
    pub signatures: u32,
    /// Dimension of the subalgebra (`2^sum(profile)`).
    pub subalgebra_dim: u64,
    /// Wall time of the bidegree, seconds.
    pub wall_s: f64,
    /// Index into [`Census::sources`] of the file the row came from.
    pub source: usize,
}

/// Census rows by `(s, t)`; later files override earlier ones.
#[derive(Clone, Debug, Default)]
pub struct Census {
    /// The rows.
    pub rows: HashMap<(i32, i32), CensusRow>,
    /// The files, in load order.
    pub sources: Vec<String>,
}

impl Census {
    /// Read census CSV files.
    pub fn load(paths: &[impl AsRef<Path>]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut c = Census::default();
        for path in paths {
            let source = c.sources.len();
            c.sources.push(path.as_ref().display().to_string());
            let text = std::fs::read_to_string(path)?;
            let mut lines = text.lines();
            let header: Vec<&str> = lines.next().unwrap_or_default().split(',').collect();
            let col = |name: &str| header.iter().position(|h| *h == name);
            let (Some(s), Some(t), Some(tm), Some(nd), Some(g), Some(sd)) = (
                col("s"),
                col("t"),
                col("target_masked_dim"),
                col("next_dim"),
                col("num_new_gens"),
                col("subalgebra_dim"),
            ) else {
                return Err(format!("{}: missing census columns", path.as_ref().display()).into());
            };
            let sig = col("signatures_total").or(col("signatures"));
            let wall = col("wall_us");
            for line in lines {
                let f: Vec<&str> = line.split(',').collect();
                if f.len() < header.len() {
                    continue;
                }
                let num = |i: usize| f[i].parse::<f64>().unwrap_or(0.0);
                c.rows.insert(
                    (num(s) as i32, num(t) as i32),
                    CensusRow {
                        target_masked_dim: num(tm),
                        next_dim: num(nd),
                        gens: num(g) as u32,
                        signatures: sig.map_or(0, |i| num(i) as u32),
                        subalgebra_dim: num(sd) as u64,
                        wall_s: wall.map_or(0.0, |i| num(i) / 1e6),
                        source,
                    },
                );
            }
        }
        Ok(c)
    }

    /// How many rows the profile rule (capped at `max_len` entries) reproduces, of those compared.
    ///
    /// A row is reproduced when its subalgebra dimension and signature count match. Returns
    /// `(matching, compared)`.
    pub fn profile_agreement(&self, max_len: usize) -> (usize, usize) {
        let mut ok = 0;
        let mut n = 0;
        for (&(s, t), r) in &self.rows {
            if r.subalgebra_dim == 0 {
                continue;
            }
            n += 1;
            let p = algebra::optimal_profile(s, t, max_len);
            let dim = 1u64 << p.iter().map(|&x| x as u32).sum::<u32>();
            if dim == r.subalgebra_dim
                && (r.signatures == 0
                    || algebra::active_signatures(&p, t).len() == r.signatures as usize)
            {
                ok += 1;
            }
        }
        (ok, n)
    }
}

/// SplitMix64: a deterministic hash for per-task noise.
pub(crate) fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A uniform in `(0, 1)` from a key.
pub(crate) fn uniform(key: u64) -> f64 {
    ((mix(key) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// A standard normal from a key (Box-Muller).
pub(crate) fn normal(key: u64) -> f64 {
    let (u, v) = (uniform(key), uniform(key ^ 0x5555_5555_5555_5555));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// Dimensions and liveness for every bidegree of the region.
///
/// The census gives them where it has a row; elsewhere they are extrapolated.
#[derive(Clone, Debug)]
pub struct Dims {
    max_n: i32,
    max_s: i32,
    tmd: Vec<f64>,
    nd: Vec<f64>,
    live: Vec<bool>,
    known: Vec<bool>,
}

impl Dims {
    /// Index of `(s, n)`, if in the region.
    fn idx(&self, s: i32, n: i32) -> Option<usize> {
        ((0..=self.max_s).contains(&s) && (0..=self.max_n).contains(&n))
            .then(|| (s * (self.max_n + 1) + n) as usize)
    }

    /// Fill the region `n <= max_n, s <= max_s`.
    ///
    /// Missing dimensions continue each row's exponential trend (a least-squares line through the
    /// log of its last known points); missing liveness (whether the bidegree has generators) is
    /// drawn with the row's recent rate.
    pub fn build(census: &Census, max_s: i32, max_n: i32) -> Self {
        let len = ((max_s + 1) * (max_n + 1)) as usize;
        let mut d = Dims {
            max_n,
            max_s,
            tmd: vec![0.0; len],
            nd: vec![0.0; len],
            live: vec![false; len],
            known: vec![false; len],
        };
        for s in 0..=max_s {
            let known: Vec<(i32, CensusRow)> = (0..=max_n)
                .filter_map(|n| census.rows.get(&(s, n + s)).map(|r| (n, *r)))
                .collect();
            // ln(value) ~ a + b n over the last points with positive values.
            let trend = |get: &dyn Fn(&CensusRow) -> f64| -> Option<(f64, f64, i32)> {
                let pts: Vec<(f64, f64)> = known
                    .iter()
                    .filter(|(_, r)| get(r) > 0.0)
                    .map(|(n, r)| (*n as f64, get(r).ln()))
                    .collect();
                let pts = &pts[pts.len().saturating_sub(30)..];
                if pts.len() < 3 {
                    return pts.last().map(|&(n, y)| (y, 0.0, n as i32));
                }
                let m = pts.len() as f64;
                let (sx, sy) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
                let (mx, my) = (sx / m, sy / m);
                let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
                let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
                let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
                Some((my - b * mx, b, pts.last().unwrap().0 as i32))
            };
            let tmd_trend = trend(&|r| r.target_masked_dim);
            let nd_trend = trend(&|r| r.next_dim);
            let recent = &known[known.len().saturating_sub(50)..];
            let p_live = if recent.is_empty() {
                0.5
            } else {
                recent.iter().filter(|(_, r)| r.gens > 0).count() as f64 / recent.len() as f64
            };
            let rows: HashMap<i32, CensusRow> = known.into_iter().collect();
            for n in 0..=max_n {
                let i = d.idx(s, n).unwrap();
                if let Some(r) = rows.get(&n) {
                    d.tmd[i] = r.target_masked_dim;
                    d.nd[i] = r.next_dim;
                    d.live[i] = r.gens > 0;
                    d.known[i] = true;
                } else {
                    let ext = |tr: Option<(f64, f64, i32)>| {
                        tr.map_or(0.0, |(a, b, last)| {
                            if n > last {
                                (a + b * n as f64).exp()
                            } else {
                                0.0
                            }
                        })
                    };
                    d.tmd[i] = ext(tmd_trend);
                    d.nd[i] = ext(nd_trend);
                    d.live[i] = uniform(((s as u64) << 32) | n as u64) < p_live;
                }
            }
        }
        d
    }

    /// Masked target dimension of the zero step at `(s, t)` (0 outside the region).
    pub fn tmd(&self, s: i32, t: i32) -> f64 {
        self.idx(s, t - s).map_or(0.0, |i| self.tmd[i])
    }

    /// Next-module dimension at `(s, t)`.
    pub fn nd(&self, s: i32, t: i32) -> f64 {
        self.idx(s, t - s).map_or(0.0, |i| self.nd[i])
    }

    /// Whether `(s, t)` has generators (so its signature walk runs).
    pub fn live(&self, s: i32, t: i32) -> bool {
        self.idx(s, t - s).is_some_and(|i| self.live[i])
    }

    /// Whether `(s, t)` came from the census rather than extrapolation.
    pub fn known(&self, s: i32, t: i32) -> bool {
        self.idx(s, t - s).is_some_and(|i| self.known[i])
    }
}

/// Task cost (H200-seconds of work), from a-priori dimensions.
///
/// Scale and shape come from different data, because the trace alone covers too narrow a region
/// to say how cost scales:
///
/// - **Across bidegrees**, the census: `ln wall = a . [1, ln tmd, ln nd, ln(signatures + 1),
///   live] + effect(file)`, over every usable census row, with a per-file effect absorbing
///   different code versions and hardware.
/// - **Level**: on the trace's own bidegrees, `level` = median of `ln(total work) - prediction`
///   for the trace run's census file; its spread is the per-bidegree noise.
/// - **Within a bidegree**: the zero step takes `zero_share` of the work, and signature `sigma`
///   a share proportional to `exp(w . [ln tmd(s, t - deg sigma), ln deg sigma])`, fitted on the
///   trace with bidegree fixed effects (residual sd = the per-signature noise).
#[derive(Clone, Debug, Serialize)]
pub struct CostModel {
    /// Across-bidegree coefficients of `[1, ln tmd, ln nd, ln(signatures + 1), live]`.
    pub across: Vec<f64>,
    /// Per-census-file effect on `ln wall` (the first file is 0).
    pub file_effect: Vec<f64>,
    /// The file whose effect the level is calibrated against (the trace run's census).
    pub reference: usize,
    /// `ln` of trace work over the reference prediction (median over the trace's bidegrees).
    pub level: f64,
    /// Robust sd of that log ratio: per-bidegree noise of the "true" costs.
    pub sd_bidegree: f64,
    /// Median share of a bidegree's work in its zero step.
    pub zero_share: f64,
    /// Within-bidegree coefficients of `[ln tmd(s, t - deg), ln deg]`.
    pub within: Vec<f64>,
    /// Residual sd of the within fit: per-signature noise of the "true" costs.
    pub sd_signature: f64,
    /// R^2 of the across fit.
    pub r2_across: f64,
    /// R^2 of the within fit (after removing bidegree means).
    pub r2_within: f64,
    /// Rows in the across fit, bidegrees in the level, signatures in the within fit.
    pub n: [usize; 3],
}

/// Median and robust (inter-quartile) sd.
fn median_sd(mut v: Vec<f64>) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    (q(0.5), (q(0.75) - q(0.25)) / 1.349)
}

/// Signature degree of a Milnor exponent tuple.
fn degree_of(sig: &[u32]) -> i32 {
    sig.iter()
        .enumerate()
        .map(|(i, &r)| ((1 << (i + 1)) - 1) * r as i32)
        .sum()
}

/// Least squares `y ~ X` (coefficients, R^2, residual sd), for the binaries.
pub fn ols_pub(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64, f64) {
    ols(x, y)
}

/// A deterministic uniform in `(0, 1)` from a key, for the binaries.
pub fn uniform_pub(key: u64) -> f64 {
    uniform(key)
}

/// Least squares `y ~ X`; returns coefficients, R^2 and residual sd.
pub(crate) fn ols(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64, f64) {
    let p = x.first().map_or(0, Vec::len);
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for (r, &yy) in x.iter().zip(y) {
        for a in 0..p {
            xty[a] += r[a] * yy;
            for b in 0..p {
                xtx[a][b] += r[a] * r[b];
            }
        }
    }
    let beta = solve(xtx, xty);
    let n = y.len().max(1) as f64;
    let mean = y.iter().sum::<f64>() / n;
    let (mut sse, mut sst) = (0.0, 0.0);
    for (r, &yy) in x.iter().zip(y) {
        let f: f64 = r.iter().zip(&beta).map(|(a, b)| a * b).sum();
        sse += (yy - f).powi(2);
        sst += (yy - mean).powi(2);
    }
    (beta, 1.0 - sse / sst.max(1e-12), (sse / n).sqrt())
}

impl CostModel {
    /// Fit from the census (scale) and the trace's tasks with work `work` (level and shape).
    pub fn fit(trace: &Trace, work: &[f64], census: &Census) -> Self {
        let files = census.sources.len().max(1);
        let features = |r: &CensusRow| {
            let mut x = vec![
                1.0,
                r.target_masked_dim.ln(),
                r.next_dim.ln(),
                ((r.signatures + 1) as f64).ln(),
                f64::from(u8::from(r.gens > 0)),
            ];
            x.extend((1..files).map(|f| f64::from(u8::from(r.source == f))));
            x
        };
        let usable =
            |r: &CensusRow| r.wall_s > 0.0 && r.target_masked_dim > 0.0 && r.next_dim > 0.0;
        let (mut x, mut y) = (Vec::new(), Vec::new());
        let mut keys: Vec<&(i32, i32)> = census.rows.keys().collect();
        keys.sort_unstable();
        for key in keys {
            let r = &census.rows[key];
            if usable(r) {
                x.push(features(r));
                y.push(r.wall_s.ln());
            }
        }
        let (beta, r2_across, _) = ols(&x, &y);
        let across = beta[..5].to_vec();
        let mut file_effect = vec![0.0];
        file_effect.extend_from_slice(&beta[5..]);

        // The trace's bidegrees: total work, zero work, and their census rows.
        let mut per: HashMap<(i32, i32), (f64, f64)> = HashMap::new();
        for (task, &w) in trace.tasks.iter().zip(work) {
            let (n, s) = task.bidegree;
            let e = per.entry((s as i32, (n + s) as i32)).or_default();
            e.0 += w;
            if task.zero {
                e.1 += w;
            }
        }
        let mut votes = vec![0usize; files];
        for b in per.keys() {
            if let Some(r) = census.rows.get(b) {
                votes[r.source] += 1;
            }
        }
        let reference = (0..files).max_by_key(|&f| (votes[f], f)).unwrap_or(0);
        let predict = |r: &CensusRow| -> f64 {
            let x = features(&CensusRow { gens: 1, ..*r });
            across.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>() + file_effect[reference]
        };
        let mut ratios = Vec::new();
        let mut shares = Vec::new();
        let mut bkeys: Vec<_> = per.keys().copied().collect();
        bkeys.sort_unstable();
        for b in &bkeys {
            let (total, zero) = per[b];
            if let Some(r) = census.rows.get(b).filter(|r| usable(r)) {
                ratios.push(total.ln() - predict(r));
            }
            if total > 0.0 {
                shares.push(zero / total);
            }
        }
        let (level, sd_bidegree) = median_sd(ratios.clone());
        let zero_share = median_sd(shares).0;

        // Within-bidegree shape: demean by bidegree, then least squares.
        let mut groups: HashMap<(i32, i32), Vec<[f64; 3]>> = HashMap::new();
        for (task, &w) in trace.tasks.iter().zip(work) {
            if task.zero || w <= 0.0 {
                continue;
            }
            let (n, s) = task.bidegree;
            let (s, t) = (s as i32, (n + s) as i32);
            let deg = degree_of(&task.sig);
            let Some(r) = census
                .rows
                .get(&(s, t - deg))
                .filter(|r| deg > 0 && r.target_masked_dim > 0.0)
            else {
                continue;
            };
            groups.entry((s, t)).or_default().push([
                r.target_masked_dim.ln(),
                (deg as f64).ln(),
                w.ln(),
            ]);
        }
        let (mut xw, mut yw) = (Vec::new(), Vec::new());
        let mut gkeys: Vec<_> = groups.keys().copied().collect();
        gkeys.sort_unstable();
        for g in gkeys {
            let rows = &groups[&g];
            let m = rows.len() as f64;
            let mean: [f64; 3] =
                std::array::from_fn(|c| rows.iter().map(|r| r[c]).sum::<f64>() / m);
            for r in rows {
                xw.push(vec![r[0] - mean[0], r[1] - mean[1]]);
                yw.push(r[2] - mean[2]);
            }
        }
        let (within, r2_within, sd_signature) = ols(&xw, &yw);
        CostModel {
            across,
            file_effect,
            reference,
            level,
            sd_bidegree,
            zero_share,
            within,
            sd_signature,
            r2_across,
            r2_within,
            n: [y.len(), ratios.len(), yw.len()],
        }
    }

    /// Median total work of `(s, t)`'s zero step and live signature walk.
    ///
    /// `signatures` is its number of active signatures; `None` when its dimensions are empty.
    pub fn total(&self, dims: &Dims, s: i32, t: i32, signatures: usize) -> Option<f64> {
        let (tmd, nd) = (dims.tmd(s, t), dims.nd(s, t));
        (tmd > 0.0 && nd > 0.0).then(|| {
            let x = [1.0, tmd.ln(), nd.ln(), ((signatures + 1) as f64).ln(), 1.0];
            let lin: f64 = self.across.iter().zip(&x).map(|(a, b)| a * b).sum();
            (lin + self.file_effect[self.reference] + self.level).exp()
        })
    }

    /// Relative weight of a signature of degree `deg` at `(s, t)` within its bidegree.
    ///
    /// 0 if its shifted problem is empty.
    pub fn shape(&self, dims: &Dims, s: i32, t: i32, deg: i32) -> f64 {
        let tmd = dims.tmd(s, t - deg);
        if deg <= 0 || tmd <= 0.0 {
            return 0.0;
        }
        (self.within[0] * tmd.ln() + self.within[1] * (deg as f64).ln()).exp()
    }
}

/// The simulated fleet, by worker class.
#[derive(Clone, Debug, Serialize)]
pub struct Fleet {
    /// `(class, workers, slots each)`.
    pub groups: Vec<(String, usize, usize)>,
}

/// A profile's signature DAG and its signatures' degrees.
#[derive(Debug)]
struct ProfileInfo {
    template: Arc<DagTemplate>,
    degree: Vec<i32>,
}

/// One bidegree of the region, with everything the simulation needs precomputed.
#[derive(Clone, Debug)]
struct Bideg {
    s: i32,
    t: i32,
    /// Index into `World::profiles`, or `None` for `F_2` (no signatures).
    profile: Option<usize>,
    live: bool,
    zero_est: f64,
    zero_true: f64,
    /// Critical path of the signature walk (excluding the zero step), estimated and true.
    cp_est: f64,
    cp_true: f64,
    /// Signature tasks that run, and their total true work.
    tasks: u32,
    work_true: f64,
    /// Work of the signature walk (estimated, true) and the sum of its signatures' shapes.
    pool_est: f64,
    pool_true: f64,
    shape_sum: f64,
    /// The walk's template, if it runs: the profile's signature DAG.
    ///
    /// [`Walks`] makes its signatures that do not run here passthroughs.
    walk: Option<Arc<DagTemplate>>,
}

/// Configuration of the whole-run world.
#[derive(Clone, Debug, Serialize)]
pub struct WholeConfig {
    /// Region: stems `0..=max_n`.
    pub max_n: i32,
    /// Region: homological degrees `0..=max_s`.
    pub max_s: i32,
    /// Profile length cap (`NASSAU_MAX_SUBALGEBRA` + 1).
    pub max_profile_len: usize,
    /// Smallest work of a dispatched task (H200-seconds).
    pub min_work: f64,
    /// Seed of the "true" costs' noise around the estimates (0: the reference draw).
    pub noise_seed: u64,
}

/// The whole run, built a priori: bidegrees, templates, costs.
#[derive(Clone)]
pub struct World {
    /// Its configuration.
    pub config: WholeConfig,
    /// The cost model.
    pub cost: CostModel,
    bideg: Vec<Bideg>,
    profiles: Vec<Arc<ProfileInfo>>,
    profile_names: Vec<Vec<u8>>,
    /// First signature id of each bidegree.
    offsets: Vec<u64>,
    dims: Dims,
}

/// The first signature job id.
///
/// Bidegree `k` has its zero step `4k`, its "registered" passthrough `4k + 1` and its "walk done"
/// passthrough `4k + 2`; signature `i` of bidegree `k` is `SIG_BASE + offsets[k] + i`.
const SIG_BASE: JobId = 1 << 62;

/// dslab-dag flops per unit of work, and resource speed per unit of single-job throughput.
const DSLAB_SCALE: f64 = 10.0;

/// Flops of a dslab-dag task that stands for no work: a join, or a signature that does not run.
const DSLAB_EPS: f64 = 1e-6;

/// Tasks dispatched between samples of [`WholeMetrics::peak_dag_nodes`].
const NODE_SAMPLE_TASKS: u64 = 65_536;

/// What a job id names.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Node {
    Zero(usize),
    Registered(usize),
    WalkDone(usize),
    Sig(usize, usize),
}

impl World {
    /// Build the world. Profiles' signature DAGs are built once each.
    pub fn build(config: WholeConfig, census: &Census, cost_from: (&Trace, &[f64])) -> Self {
        let dims = Dims::build(census, config.max_s, config.max_n);
        let cost = CostModel::fit(cost_from.0, cost_from.1, census);
        let mut profile_names: Vec<Vec<u8>> = Vec::new();
        let mut profile_of: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut bideg = Vec::new();
        for s in 0..=config.max_s {
            for n in 0..=config.max_n {
                let t = n + s;
                let p = algebra::optimal_profile(s, t, config.max_profile_len);
                // Rows 0 and 1 run `step0`/`step1`, which have no signature walk.
                let p = if s <= 1 { Vec::new() } else { p };
                let profile = (!p.is_empty()).then(|| {
                    let next = profile_names.len();
                    *profile_of.entry(p.clone()).or_insert_with(|| {
                        profile_names.push(p.clone());
                        next
                    })
                });
                bideg.push(Bideg {
                    s,
                    t,
                    profile,
                    live: dims.live(s, t),
                    zero_est: 0.0,
                    zero_true: 0.0,
                    cp_est: 0.0,
                    cp_true: 0.0,
                    tasks: 0,
                    work_true: 0.0,
                    pool_est: 0.0,
                    pool_true: 0.0,
                    shape_sum: 0.0,
                    walk: None,
                });
            }
        }
        let profiles: Vec<Arc<ProfileInfo>> = profile_names
            .iter()
            .map(|p| {
                let clock = std::time::Instant::now();
                let direct = algebra::signature_dag(p);
                let template = direct.transitive_reduction();
                let degree = (0..template.len())
                    .map(|i| algebra::signature(i, p).1)
                    .collect();
                eprintln!(
                    "[whole] profile {p:?}: {} signatures, {} direct edges, {} after reduction, \
                     built in {:.1}s",
                    template.len(),
                    direct.edge_count(),
                    template.edge_count(),
                    clock.elapsed().as_secs_f64()
                );
                Arc::new(ProfileInfo {
                    template: Arc::new(template),
                    degree,
                })
            })
            .collect();
        let mut w = World {
            config,
            cost,
            bideg,
            profiles,
            profile_names,
            offsets: Vec::new(),
            dims,
        };
        let mut offset = 0u64;
        for k in 0..w.bideg.len() {
            w.offsets.push(offset);
            let (s, t) = (w.bideg[k].s, w.bideg[k].t);
            let info = w.bideg[k].profile.map(|pi| Arc::clone(&w.profiles[pi]));
            let active: Vec<usize> = info.as_ref().map_or(Vec::new(), |info| {
                (1..info.template.len())
                    .filter(|&i| info.degree[i] <= t)
                    .collect()
            });
            let total = w.cost.total(&w.dims, s, t, active.len()).unwrap_or(0.0);
            let total_true = total
                * (w.cost.sd_bidegree * normal((k as u64 * 2 + 1) ^ mix(w.config.noise_seed)))
                    .exp();
            let zs = w.cost.zero_share;
            w.bideg[k].zero_est = (zs * total).max(w.config.min_work);
            w.bideg[k].zero_true = (zs * total_true).max(w.config.min_work);
            if w.bideg[k].live {
                w.bideg[k].pool_est = (1.0 - zs) * total;
                w.bideg[k].pool_true = (1.0 - zs) * total_true;
                w.bideg[k].shape_sum = active
                    .iter()
                    .map(|&i| {
                        w.cost
                            .shape(&w.dims, s, t, info.as_ref().unwrap().degree[i])
                    })
                    .sum();
            }
            if let Some(info) = info {
                offset += info.template.len() as u64;
                let cp = |truth| {
                    (info.template)
                        .critical_path(|i| Duration::from_secs_f64(w.sig_work(k, i, truth)))
                        .as_secs_f64()
                };
                let (cp_est, cp_true) = (cp(false), cp(true));
                w.bideg[k].cp_est = cp_est;
                w.bideg[k].cp_true = cp_true;
                let mut tasks = 0;
                let mut work = 0.0;
                for i in 0..info.template.len() {
                    let x = w.sig_work(k, i, true);
                    if x > 0.0 {
                        tasks += 1;
                        work += x;
                    }
                }
                w.bideg[k].tasks = tasks;
                w.bideg[k].work_true = work;
                if w.bideg[k].live {
                    w.bideg[k].walk = Some(Arc::clone(&info.template));
                }
            }
        }
        w
    }

    /// Work of signature `i` of bidegree `k`, 0 if it does not run.
    ///
    /// It is its share of the walk's estimated work, or of the "true" work times a deterministic
    /// log-normal per-signature error.
    fn sig_work(&self, k: usize, i: usize, truth: bool) -> f64 {
        let b = &self.bideg[k];
        let Some(pi) = b.profile else { return 0.0 };
        let deg = self.profiles[pi].degree[i];
        if i == 0 || !b.live || deg > b.t || b.shape_sum <= 0.0 {
            return 0.0;
        }
        let share = self.cost.shape(&self.dims, b.s, b.t, deg) / b.shape_sum;
        if share <= 0.0 {
            return 0.0;
        }
        if truth {
            let noise = (self.cost.sd_signature
                * normal(mix(k as u64) ^ i as u64 ^ mix(self.config.noise_seed.wrapping_add(1))))
            .exp();
            (b.pool_true * share * noise).max(self.config.min_work)
        } else {
            (b.pool_est * share).max(self.config.min_work)
        }
    }

    /// Bidegree index of `(s, t)`, if in the region.
    fn index(&self, s: i32, t: i32) -> Option<usize> {
        self.dims.idx(s, t - s)
    }

    /// What an id names.
    fn node(&self, id: JobId) -> Node {
        if id >= SIG_BASE {
            let off = id - SIG_BASE;
            let k = self.offsets.partition_point(|&o| o <= off) - 1;
            Node::Sig(k, (off - self.offsets[k]) as usize)
        } else {
            let k = (id / 4) as usize;
            match id % 4 {
                0 => Node::Zero(k),
                1 => Node::Registered(k),
                _ => Node::WalkDone(k),
            }
        }
    }

    /// The bidegrees a zero step reads (`depgraph`'s edges into `Compute(s, t)`), as indices.
    fn compute_deps(&self, k: usize) -> Vec<usize> {
        let b = &self.bideg[k];
        let (s, t) = (b.s, b.t);
        let same_row = if s <= 1 {
            t - 1
        } else {
            t - b
                .profile
                .map_or(1, |p| algebra::zero_sig_floor(&self.profile_names[p]))
        };
        let mut deps: Vec<usize> = self.index(s, same_row).into_iter().collect();
        if s == 1 {
            deps.extend(self.index(0, t));
        } else if s >= 2 {
            deps.extend(self.index(s - 1, t - 1));
        }
        deps
    }

    /// The world as dslab-dag input (DAG YAML, system YAML), for cross-validation.
    ///
    /// Model alignment: one resource per worker with `cores = slots` and speed `DSLAB_SCALE` times
    /// the class's single-job throughput; task flops `DSLAB_SCALE` times work (true or estimated),
    /// so durations are in seconds; data items of size 0 on an infinitely fast network. Our two
    /// passthroughs per bidegree ("walk done", "registered") become one join task of `DSLAB_EPS`
    /// flops, as do signatures with no work. Tasks are listed in topological order (bidegrees by
    /// `(t, s)`, then template order), which dslab's static schedulers need.
    pub fn export_dslab(
        &self,
        fleet: &Fleet,
        model: &dyn ServiceModel,
        truth: bool,
    ) -> (String, String) {
        use std::fmt::Write;
        let mut dag = String::from("tasks:\n");
        let mut task = |name: &str, flops: f64, inputs: &[String]| {
            let inputs: Vec<String> = inputs.iter().map(|i| format!("\"{i}\"")).collect();
            writeln!(
                dag,
                "  - name: {name}\n    flops: {:.9}\n    memory: 0\n    min_cores: 1\n    \
                 max_cores: 1\n    inputs: [{}]\n    outputs: [{{\"name\": \"{name}\", \"size\": \
                 0}}]",
                flops.max(DSLAB_EPS),
                inputs.join(", ")
            )
            .unwrap();
        };
        let mut order: Vec<usize> = (0..self.bideg.len()).collect();
        order.sort_by_key(|&k| (self.bideg[k].t, self.bideg[k].s));
        for k in order {
            let b = &self.bideg[k];
            let deps: Vec<String> = self
                .compute_deps(k)
                .iter()
                .map(|d| format!("r{d}"))
                .collect();
            let zero = if truth { b.zero_true } else { b.zero_est };
            task(&format!("z{k}"), DSLAB_SCALE * zero, &deps);
            let mut join = vec![format!("z{k}")];
            if let Some(pi) = b.profile.filter(|_| b.live) {
                let t = &self.profiles[pi].template;
                for &i in t.topological_order() {
                    let i = i as usize;
                    let mut inputs: Vec<String> = t
                        .predecessors(i)
                        .iter()
                        .map(|&p| format!("s{k}_{p}"))
                        .collect();
                    if inputs.is_empty() {
                        inputs.push(format!("z{k}"));
                    }
                    task(
                        &format!("s{k}_{i}"),
                        DSLAB_SCALE * self.sig_work(k, i, truth),
                        &inputs,
                    );
                }
                join.extend(t.sinks().map(|i| format!("s{k}_{i}")));
            }
            join.extend(self.index(b.s, b.t - 1).map(|p| format!("r{p}")));
            task(&format!("r{k}"), 0.0, &join);
        }
        let mut sys = String::from("resources:\n");
        let mut id = 0;
        for (class, count, slots) in &fleet.groups {
            for _ in 0..*count {
                writeln!(
                    sys,
                    "  - name: w{id}_{class}\n    speed: {:.9}\n    cores: {slots}\n    memory: 0",
                    DSLAB_SCALE * model.throughput(class, 1)
                )
                .unwrap();
                id += 1;
            }
        }
        sys += "network:\n  model: ConstantBandwidth\n  bandwidth: 1000000000\n  latency: 0\n";
        (dag, sys)
    }

    /// The world flattened into a small instance (the PISA replica family); use a small region.
    ///
    /// Every task keeps its true work and estimate, the two passthroughs of a bidegree merge into
    /// one join, and groups are numbered in `(s, t)` order.
    pub fn to_small(&self, fleet: &Fleet, model: &dyn ServiceModel) -> crate::small::SmallInstance {
        use crate::small::{Class, Kind, SmallInstance, SmallTask};
        // s-major, as `simulate`'s ids: the DAG layer releases simultaneous dependents in id
        // order, so this keeps both simulators' tie-breaks between bidegrees identical. (It is
        // topological: every bidegree edge goes to a higher s, or to the same s at a higher t.)
        let mut order: Vec<usize> = (0..self.bideg.len()).collect();
        order.sort_by_key(|&k| (self.bideg[k].s, self.bideg[k].t));
        let mut join = vec![u32::MAX; self.bideg.len()];
        let mut tasks: Vec<SmallTask> = Vec::new();
        for (g, &k) in order.iter().enumerate() {
            let b = &self.bideg[k];
            let (row, col) = (b.s as u32, (b.t - b.s) as u32);
            let task = |kind, work, est, deps| SmallTask {
                group: g as u32,
                row,
                col,
                kind,
                work,
                est,
                deps,
            };
            let zero = tasks.len() as u32;
            let deps = self.compute_deps(k).iter().map(|&d| join[d]).collect();
            tasks.push(task(Kind::Zero, b.zero_true, b.zero_est, deps));
            let mut sinks = vec![zero];
            if let Some(pi) = b.profile.filter(|_| b.live) {
                let t = &self.profiles[pi].template;
                let mut id = vec![u32::MAX; t.len()];
                for &i in t.topological_order() {
                    let i = i as usize;
                    let mut deps: Vec<u32> =
                        t.predecessors(i).iter().map(|&p| id[p as usize]).collect();
                    if deps.is_empty() {
                        deps.push(zero);
                    }
                    id[i] = tasks.len() as u32;
                    let (w, e) = (self.sig_work(k, i, true), self.sig_work(k, i, false));
                    // A signature with nothing to do is a passthrough in `simulate`: a join here.
                    let kind = if w > 0.0 { Kind::Sig } else { Kind::Join };
                    tasks.push(task(kind, w, e, deps));
                }
                sinks = t.sinks().map(|i| id[i]).collect();
            }
            let mut deps = sinks;
            deps.push(zero);
            deps.extend(self.index(b.s, b.t - 1).map(|p| join[p]));
            deps.sort_unstable();
            deps.dedup();
            join[k] = tasks.len() as u32;
            tasks.push(task(Kind::Join, 0.0, 0.0, deps));
        }
        let classes = fleet
            .groups
            .iter()
            .map(|(c, n, slots)| Class {
                name: c.clone(),
                speed: model.throughput(c, 1),
                workers: *n as u32,
                slots: *slots as u32,
            })
            .collect();
        SmallInstance { tasks, classes }
    }

    /// Summary of the world for the report.
    pub fn summary(&self) -> WorldSummary {
        let known = self
            .bideg
            .iter()
            .filter(|b| self.dims.known(b.s, b.t))
            .count();
        let mut per_profile: HashMap<String, (usize, u64)> = HashMap::new();
        for b in &self.bideg {
            let name = b
                .profile
                .map_or("F2".to_string(), |p| format!("{:?}", self.profile_names[p]));
            let e = per_profile.entry(name).or_default();
            e.0 += 1;
            e.1 += b.tasks as u64;
        }
        let mut per_profile: Vec<(String, usize, u64)> = per_profile
            .into_iter()
            .map(|(k, v)| (k, v.0, v.1))
            .collect();
        per_profile.sort();
        WorldSummary {
            bidegrees: self.bideg.len(),
            from_census: known,
            live: self.bideg.iter().filter(|b| b.live).count(),
            signature_tasks: self.bideg.iter().map(|b| b.tasks as u64).sum(),
            template_nodes: self.offsets.last().copied().unwrap_or(0)
                + self
                    .bideg
                    .last()
                    .and_then(|b| b.profile)
                    .map_or(0, |p| self.profiles[p].template.len() as u64),
            zero_work: self.bideg.iter().map(|b| b.zero_true).sum(),
            signature_work: self.bideg.iter().map(|b| b.work_true).sum(),
            per_profile,
        }
    }

    /// Lower bounds on any schedule's makespan on `fleet`: the critical path, and work/capacity.
    ///
    /// The critical path is at the fastest class's single-job speed with unlimited workers;
    /// work/capacity is total work over total throughput.
    pub fn bounds(&self, fleet: &Fleet, model: &dyn ServiceModel) -> (f64, f64) {
        let fastest = fleet
            .groups
            .iter()
            .map(|g| model.throughput(&g.0, 1))
            .fold(0.0, f64::max);
        let capacity: f64 = fleet
            .groups
            .iter()
            .map(|g| g.1 as f64 * model.throughput(&g.0, g.2))
            .sum();
        // Finish times in increasing t: every edge goes to a larger t.
        let mut order: Vec<usize> = (0..self.bideg.len()).collect();
        order.sort_by_key(|&k| (self.bideg[k].t, self.bideg[k].s));
        let mut registered = vec![0.0f64; self.bideg.len()];
        let mut best = 0.0f64;
        for k in order {
            let b = &self.bideg[k];
            let start = self
                .compute_deps(k)
                .iter()
                .map(|&d| registered[d])
                .fold(0.0, f64::max);
            let walk_done = start + (b.zero_true + b.cp_true) / fastest;
            let prev = self.index(b.s, b.t - 1).map_or(0.0, |p| registered[p]);
            registered[k] = walk_done.max(prev);
            best = best.max(registered[k]);
        }
        let work: f64 = self.bideg.iter().map(|b| b.zero_true + b.work_true).sum();
        (best, work / capacity)
    }
}

/// Size of the whole-run DAG.
#[derive(Clone, Debug, Serialize)]
pub struct WorldSummary {
    /// Bidegrees in the region.
    pub bidegrees: usize,
    /// Of which measured by the census (the rest extrapolated).
    pub from_census: usize,
    /// Bidegrees with generators (their signature walks run).
    pub live: usize,
    /// Signature tasks that run.
    pub signature_tasks: u64,
    /// Signature-DAG nodes over all bidegrees (including no-ops).
    pub template_nodes: u64,
    /// Total true work of zero steps and of signatures (H200-seconds).
    pub zero_work: f64,
    /// See `zero_work`.
    pub signature_work: f64,
    /// `(profile, bidegrees, signature tasks)`.
    pub per_profile: Vec<(String, usize, u64)>,
}

/// How the run is driven.
#[derive(Clone, Debug, Serialize)]
pub enum Plan {
    /// Today's coordinator: capped bidegrees and walk tasks in flight, oldest bidegree first.
    ///
    /// At most `open` bidegrees are in flight (one coordinator thread each), and at most
    /// `per_bidegree` signature tasks per bidegree (walk threads).
    Today {
        /// Open-bidegree cap.
        open: usize,
        /// In-flight cap per bidegree.
        per_bidegree: usize,
    },
    /// Event-driven, oldest bidegree first, no caps.
    Group,
    /// Event-driven, by upward rank over the whole DAG (estimated or true costs), with aging.
    Rank {
        /// Use the true costs for ranks.
        oracle: bool,
        /// `Config::age_limit`.
        age_limit: Option<Duration>,
        /// Oldest bidegree first, rank only within a bidegree.
        group_first: bool,
    },
}

/// Which jobs are restricted to the fastest worker class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub enum Pin {
    /// None: placement alone decides.
    #[default]
    None,
    /// Every job: the run uses only the fast class.
    All,
    /// CPOP: zero steps and each bidegree's critical signatures (by estimated cost).
    Critical,
}

/// How the simulated coordinator places jobs and maintains ranks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// Speed-aware worker choice and the machine model.
    pub speed: SpeedPlan,
    /// Pinning to the fast class.
    pub pin: Pin,
    /// `DagConfig::rank_epsilon` (approximate rank propagation).
    pub rank_epsilon: f64,
    /// At most this many walks open at once; it changes the schedule.
    ///
    /// A walk opens when its first signature job is ready, and while the budget is spent its jobs
    /// are held, in the order the walks became ready. The budget belongs to the simulated
    /// coordinator: the DAG layer materialises lazily and has none.
    pub max_open: Option<usize>,
    /// How bidegrees are ordered against each other.
    pub group_key: GroupKey,
    /// Aging for any plan (overrides the rank plans' own).
    pub age_limit: Option<Duration>,
}

/// The order between bidegrees ("oldest first" and its restart-stable stand-ins).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum GroupKey {
    /// First submission ([`GroupOrder::Arrival`]).
    ///
    /// It depends on release order, so it is not stable across a coordinator restart.
    #[default]
    Arrival,
    /// `(s, t)` ([`GroupOrder::Id`]).
    SMajor,
    /// `(t, s)`.
    TMajor,
    /// `(t - s, s)`: by stem.
    StemMajor,
}

impl GroupKey {
    /// Each bidegree's group id: its index for `Arrival`, its rank under the key otherwise.
    fn ids(self, world: &World) -> Vec<u64> {
        let n = world.bideg.len();
        let key = |k: usize| {
            let (s, t) = (world.bideg[k].s as i64, world.bideg[k].t as i64);
            match self {
                GroupKey::Arrival => (0, k as i64),
                GroupKey::SMajor => (s, t),
                GroupKey::TMajor => (t, s),
                GroupKey::StemMajor => (t - s, s),
            }
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&k| key(k));
        let mut ids = vec![0; n];
        for (rank, k) in order.into_iter().enumerate() {
            ids[k] = rank as u64;
        }
        ids
    }
}

impl Default for Placement {
    /// Speed-oblivious placement, every other option at its library default or off.
    fn default() -> Self {
        Self {
            speed: SpeedPlan::default(),
            pin: Pin::None,
            rank_epsilon: DagConfig::default().rank_epsilon,
            max_open: None,
            group_key: GroupKey::Arrival,
            age_limit: None,
        }
    }
}

impl Plan {
    /// Short name for tables.
    pub fn name(&self) -> String {
        match self {
            Plan::Today { open, per_bidegree } => {
                format!("today (open<={open}, walk<={per_bidegree})")
            }
            Plan::Group => "group order, uncapped".into(),
            Plan::Rank {
                oracle,
                age_limit,
                group_first,
            } => format!(
                "{}DAG rank{}{}",
                if *group_first {
                    "group order, then "
                } else {
                    ""
                },
                if *oracle {
                    " (oracle costs)"
                } else {
                    " (estimated costs)"
                },
                age_limit.map_or(String::new(), |a| format!(
                    ", aging {:.0}s",
                    a.as_secs_f64()
                ))
            ),
        }
    }
}

/// One plan's result.
#[derive(Clone, Debug, Serialize)]
pub struct WholeMetrics {
    /// Plan name.
    pub plan: String,
    /// Makespan, hours.
    pub makespan_h: f64,
    /// Tasks dispatched to workers.
    pub tasks: u64,
    /// Busy slot-time over available slot-time.
    pub slot_util: f64,
    /// Bidegree latency (zero step ready to walk done), seconds.
    pub bidegree_latency: Quantiles,
    /// Most bidegrees open at once.
    pub peak_open: usize,
    /// Most template nodes with materialised state at once ([`DagStats::nodes`](whelm::DagStats)).
    ///
    /// Sampled every `NODE_SAMPLE_TASKS` tasks dispatched.
    pub peak_dag_nodes: usize,
    /// Wall time of the placing `poll` calls, microseconds.
    pub dispatch_us: Quantiles,
    /// Wall time of the simulation, seconds.
    pub sim_s: f64,
}

/// An event of the simulation.
#[derive(Clone, Copy, Debug)]
enum Ev {
    /// A worker's next completion, of this [`PsWorker`] version.
    Done(usize, u64),
    /// The policy asked to be polled again ([`Policy::next_wakeup`]).
    Wake,
}

/// A simulated worker: processor sharing over its running tasks.
struct Wk {
    class: String,
    ps: PsWorker,
}

/// The simulated coordinator's DAG layer.
type Dag = DagScheduler<Scheduler>;

/// The walks' signatures, as the DAG layer's [`NodeSource`].
///
/// Walk `4k + 2`'s leaf `i` is signature `i` of bidegree `k`.
struct Walks {
    world: World,
    /// Work is the true cost rather than the estimate.
    oracle: bool,
    /// [`Pin::Critical`]: the fast class, and each live bidegree's critical signatures.
    critical: Option<(String, HashMap<usize, Vec<bool>>)>,
}

impl NodeSource for Walks {
    /// The signature's estimated or true work.
    fn work(&self, unit: JobId, leaf: u32) -> Duration {
        let (k, i) = ((unit / 4) as usize, leaf as usize);
        Duration::from_secs_f64(self.world.sig_work(k, i, self.oracle))
    }

    /// The signature does not run at this bidegree: it has no true work.
    fn passthrough(&self, unit: JobId, leaf: u32) -> bool {
        let (k, i) = ((unit / 4) as usize, leaf as usize);
        self.world.sig_work(k, i, true) <= 0.0
    }

    /// Pins a critical signature to the fast class.
    fn spec(&self, unit: JobId, leaf: u32, spec: &mut JobSpec) {
        if let Some((class, critical)) = &self.critical
            && critical
                .get(&((unit / 4) as usize))
                .is_some_and(|c| c[leaf as usize])
        {
            spec.constraints.push(Constraint {
                on: Selector::Class(class.clone()),
                strength: Strength::Require,
            });
        }
    }
}

/// The simulated coordinator's caps on releasing signature jobs.
///
/// Today's per-bidegree in-flight cap, and the open-walk budget ([`Placement::max_open`]).
struct SigGates {
    /// Today's in-flight cap per bidegree.
    per_bidegree: Option<usize>,
    inflight: Vec<usize>,
    queued: HashMap<usize, VecDeque<JobId>>,
    /// The open-walk budget.
    max_open: Option<usize>,
    walk_open: Vec<bool>,
    open_walks: usize,
    /// Walks waiting for the budget, and their jobs announced ready meanwhile.
    walk_queue: VecDeque<usize>,
    walk_held: HashMap<usize, Vec<JobId>>,
}

impl SigGates {
    /// Signature job `id` of bidegree `k` is ready: release it unless a cap holds it.
    fn ready(&mut self, dag: &mut Dag, k: usize, id: JobId, now: Time) {
        if let Some(max) = self.max_open
            && !self.walk_open[k]
        {
            if self.open_walks < max {
                self.walk_open[k] = true;
                self.open_walks += 1;
            } else {
                let held = self.walk_held.entry(k).or_default();
                if held.is_empty() {
                    self.walk_queue.push_back(k);
                }
                held.push(id);
                return;
            }
        }
        self.release(dag, k, id, now);
    }

    /// Release signature job `id` of open walk `k`, or queue it behind the per-bidegree cap.
    fn release(&mut self, dag: &mut Dag, k: usize, id: JobId, now: Time) {
        match self.per_bidegree {
            Some(per) if self.inflight[k] >= per => self.queued.entry(k).or_default().push_back(id),
            _ => {
                self.inflight[k] += 1;
                dag.release(id, now);
            }
        }
    }

    /// A signature job of bidegree `k` completed: release the next one it held back.
    fn done(&mut self, dag: &mut Dag, k: usize, now: Time) {
        if self.per_bidegree.is_none() {
            return;
        }
        self.inflight[k] -= 1;
        if let Some(next) = self.queued.get_mut(&k).and_then(VecDeque::pop_front) {
            self.inflight[k] += 1;
            dag.release(next, now);
        }
    }

    /// Bidegree `k`'s walk completed: open the walks waiting for its share of the budget.
    fn walk_done(&mut self, dag: &mut Dag, k: usize, now: Time) {
        let Some(max) = self.max_open else {
            return;
        };
        if self.walk_open[k] {
            self.walk_open[k] = false;
            self.open_walks -= 1;
        }
        while self.open_walks < max {
            let Some(q) = self.walk_queue.pop_front() else {
                break;
            };
            self.walk_open[q] = true;
            self.open_walks += 1;
            for id in self.walk_held.remove(&q).unwrap_or_default() {
                self.release(dag, q, id, now);
            }
        }
    }
}

/// Simulate the whole run under `plan` on `fleet`, placing as `place` says.
///
/// Each worker's [`WorkerState::speed`] is its class's single-job throughput, or 1 when speeds are
/// learned.
///
/// The whole DAG is declared up front: per bidegree, its zero step, its walk and its "registered"
/// passthrough (ids as on `SIG_BASE`). The walk is a unit of the profile's signature DAG, whose
/// leaves' costs, and which of them run, come from the world, or a passthrough when it has none;
/// the DAG layer materialises it when the zero step completes.
pub fn simulate(
    world: &World,
    fleet: &Fleet,
    model: &dyn ServiceModel,
    plan: &Plan,
    place: &Placement,
) -> WholeMetrics {
    let clock = std::time::Instant::now();
    let (rank, oracle, age_limit) = match plan {
        Plan::Rank {
            oracle, age_limit, ..
        } => (true, *oracle, *age_limit),
        _ => (false, false, None),
    };
    let age_limit = place.age_limit.or(age_limit);
    let group_first = matches!(
        plan,
        Plan::Rank {
            group_first: true,
            ..
        }
    );
    let gid = place.group_key.ids(world);
    let caps = match plan {
        Plan::Today { open, per_bidegree } => Some((*open, *per_bidegree)),
        _ => None,
    };
    let order = match (rank, group_first) {
        (false, _) => vec![OrderTerm::Group],
        (true, false) => vec![OrderTerm::Rank, OrderTerm::Group],
        (true, true) => vec![OrderTerm::Group, OrderTerm::Rank],
    };
    let policy = Scheduler::new(Config {
        order,
        group_order: if place.group_key == GroupKey::Arrival {
            GroupOrder::Arrival
        } else {
            GroupOrder::Id
        },
        age_limit,
        score: place.speed.score(),
        speed: place.speed.config,
        ..Config::default()
    });
    let fast_class = fleet
        .groups
        .iter()
        .max_by(|a, b| {
            model
                .throughput(&a.0, 1)
                .total_cmp(&model.throughput(&b.0, 1))
        })
        .map(|g| g.0.clone());
    let n = world.bideg.len();
    let critical = (place.pin == Pin::Critical).then(|| {
        let marks = (0..n)
            .filter_map(|k| {
                let t = world.bideg[k].walk.as_ref()?;
                let work = |i| Duration::from_secs_f64(world.sig_work(k, i, false));
                Some((k, t.critical_nodes(work, 1e-9)))
            })
            .collect();
        (fast_class.clone().unwrap_or_default(), marks)
    });
    let source = Walks {
        world: world.clone(),
        oracle,
        critical,
    };
    let mut dag: Dag = DagScheduler::new(
        DagConfig {
            default_work: Duration::ZERO,
            rank_epsilon: place.rank_epsilon,
            auto_submit: false,
            record_passthrough: true,
            track_ranks: rank,
        },
        policy,
    )
    .with_source(Arc::new(source));
    let mut workers: Vec<Wk> = Vec::new();
    for (class, count, slots) in &fleet.groups {
        for _ in 0..*count {
            let id = workers.len() as u64;
            let state = WorkerState {
                id,
                class: class.clone(),
                capacity: Resources::new()
                    .with(MEMORY, 1 << 60)
                    .with(SLOTS, *slots as u64),
                speed: if place.speed.learned() {
                    1.0
                } else {
                    model.throughput(class, 1)
                },
                ..Default::default()
            };
            dag.handle(Input::Worker(state), Time::ORIGIN);
            workers.push(Wk {
                class: class.clone(),
                ps: PsWorker::default(),
            });
        }
    }
    let slots_total: usize = fleet.groups.iter().map(|g| g.1 * g.2).sum();
    let spec = |id: JobId, k: usize, kind: &str, pinned: bool| {
        let constraints = match (&fast_class, pinned) {
            (Some(c), true) => vec![Constraint::require_class(c.clone())],
            _ => Vec::new(),
        };
        JobSpec {
            id,
            group: gid[k],
            kind: Some(kind.into()),
            constraints,
            ..Default::default()
        }
    };
    // A barrier of no work, completing by itself once `deps` are done.
    let pass = |id: JobId, k: usize, deps: Vec<JobId>| -> Unit {
        DagJob {
            spec: JobSpec {
                id,
                group: gid[k],
                ..Default::default()
            },
            deps,
            work_estimate: Some(Duration::ZERO),
            passthrough: true,
            ..Default::default()
        }
        .into()
    };

    let cost = |est: f64, truth: f64| if oracle { truth } else { est };
    let mut units: Vec<Unit> = Vec::with_capacity(3 * n);
    for k in 0..n {
        let b = &world.bideg[k];
        let zero = 4 * k as u64;
        let deps: Vec<JobId> = world
            .compute_deps(k)
            .iter()
            .map(|&d| 4 * d as u64 + 1)
            .collect();
        let job = DagJob {
            spec: spec(zero, k, "zero", place.pin != Pin::None),
            deps,
            work_estimate: Some(Duration::from_secs_f64(cost(b.zero_est, b.zero_true))),
            ..Default::default()
        };
        units.push(job.into());
        units.push(match &b.walk {
            Some(t) => Unit {
                id: zero + 2,
                base: SIG_BASE + world.offsets[k],
                template: Arc::clone(t),
                deps: vec![zero],
                spec: spec(0, k, "sig", place.pin == Pin::All),
                scale: Some(1.0),
                sourced: true,
                ..Default::default()
            },
            // A dead bidegree's walk is all no-ops.
            None => pass(zero + 2, k, vec![zero]),
        });
        let mut reg = vec![zero, zero + 2];
        reg.extend(world.index(b.s, b.t - 1).map(|p| 4 * p as u64 + 1));
        units.push(pass(zero + 1, k, reg));
    }
    dag.declare(units, Time::ORIGIN)
        .expect("the whole-run DAG is acyclic");

    let mut queue = Queue::new();
    let mut ready_at = vec![f64::NAN; n];
    let mut done_at = vec![f64::NAN; n];
    let mut open = 0usize;
    let mut peak_open = 0usize;
    let mut open_queue: VecDeque<usize> = VecDeque::new();
    let mut gates = SigGates {
        per_bidegree: caps.map(|c| c.1),
        inflight: vec![0; n],
        queued: HashMap::new(),
        max_open: place.max_open.map(|m| m.max(1)),
        walk_open: vec![false; n],
        open_walks: 0,
        walk_queue: VecDeque::new(),
        walk_held: HashMap::new(),
    };
    let mut tasks = 0u64;
    let mut dispatch_us = Vec::new();
    let mut peak_nodes = 0usize;
    let mut next_sample = 0u64;
    let mut wake_at = None;
    let mut now = 0.0;
    // Announcements the placing poll returned, for the next instant.
    let mut carry: Vec<Output> = Vec::new();
    // Jobs with a running attempt: the first attempt to finish completes the job.
    let mut running: HashSet<JobId> = HashSet::new();

    loop {
        // Newly ready jobs: release them (through the simulated coordinator's caps) before
        // anything is placed.
        let at = Time(Duration::from_secs_f64(now));
        let mut announced = std::mem::take(&mut carry);
        announced.extend(dag.announcements());
        for o in &announced {
            let &Output::Ready { job: id } = o else {
                continue;
            };
            match world.node(id) {
                Node::Zero(k) => {
                    ready_at[k] = now;
                    match caps {
                        Some((cap, _)) if open >= cap => open_queue.push_back(k),
                        _ => {
                            open += 1;
                            dag.release(id, at);
                        }
                    }
                }
                Node::Sig(k, _) => gates.ready(&mut dag, k, id, at),
                other => unreachable!("{other:?} is a passthrough"),
            }
        }
        for o in &announced {
            let &Output::Passed { job: id } = o else {
                continue;
            };
            if let Node::WalkDone(k) = world.node(id) {
                done_at[k] = now;
                open -= 1;
                if let Some((cap, _)) = caps {
                    while open < cap {
                        let Some(q) = open_queue.pop_front() else {
                            break;
                        };
                        open += 1;
                        dag.release(4 * q as u64, at);
                    }
                }
                gates.walk_done(&mut dag, k, at);
            }
        }
        peak_open = peak_open.max(open);
        let c = std::time::Instant::now();
        let out = dag.poll(at);
        dispatch_us.push(c.elapsed().as_secs_f64() * 1e6);
        let mut dirty = Vec::new();
        for o in out {
            match o {
                Output::Start {
                    job: id,
                    attempt,
                    worker,
                } => {
                    let w = worker as usize;
                    advance(&mut workers[w], model, now);
                    let work = match world.node(id) {
                        Node::Zero(k) => world.bideg[k].zero_true,
                        Node::Sig(k, i) => world.sig_work(k, i, true),
                        other => unreachable!("{other:?} was placed"),
                    };
                    workers[w].ps.start(id, attempt, work);
                    running.insert(id);
                    tasks += 1;
                    dirty.push(w);
                }
                Output::Stop {
                    job,
                    attempt,
                    worker,
                } => {
                    let w = worker as usize;
                    advance(&mut workers[w], model, now);
                    if workers[w].ps.stop(job, attempt) {
                        dirty.push(w);
                    }
                }
                Output::Ready { .. } | Output::Passed { .. } => carry.push(o),
                Output::GaveUp(g) => unreachable!("job {} failed, but no attempt fails", g.job),
                Output::Rejected { job, reason } => unreachable!("job {job} rejected: {reason}"),
                Output::RunLocal { .. } => {}
            }
        }
        dirty.sort_unstable();
        dirty.dedup();
        for w in dirty {
            schedule(&mut workers[w], w, model, now, &mut queue);
        }
        // Deferrals expire without an event: wake the policy then.
        if let Some(t) = dag.next_wakeup()
            && Some(t) != wake_at
        {
            wake_at = Some(t);
            queue.push(t.0.as_secs_f64(), Ev::Wake);
        }
        if tasks >= next_sample {
            next_sample = tasks + NODE_SAMPLE_TASKS;
            peak_nodes = peak_nodes.max(dag.dag_stats().nodes);
        }

        // Next instant: every event at it is applied before the next poll, as a coordinator that
        // drains its event queue would (placing between simultaneous completions would let
        // whichever is processed first grab the free slots, regardless of priority).
        let Some((t, events)) = queue.pop_instant() else {
            break;
        };
        now = t;
        let at = Time(Duration::from_secs_f64(now));
        for ev in events {
            let Ev::Done(w, v) = ev else {
                // Stale wakeups (for jobs placed before their deadline) are harmless no-ops.
                continue;
            };
            if !workers[w].ps.is_current(v) {
                continue;
            }
            advance(&mut workers[w], model, now);
            // The event was scheduled for the task(s) with the least work left: finish them even
            // if rounding left a sliver (a completion at `now + tiny` can round to `now`).
            let mut finished = workers[w]
                .ps
                .finish(|left, least| left <= least.max(0.0) + 1e-9 * (1.0 + least.abs()));
            finished.sort_unstable_by_key(|r| (r.job, r.attempt));
            for r in finished {
                let id = r.job;
                // Another attempt finished first; this one's stop is on its way.
                if !running.remove(&id) {
                    continue;
                }
                dag.handle(
                    Input::Done {
                        job: id,
                        attempt: r.attempt,
                    },
                    at,
                );
                if let Node::Sig(k, _) = world.node(id) {
                    gates.done(&mut dag, k, at);
                }
            }
            schedule(&mut workers[w], w, model, now, &mut queue);
        }
    }
    let unfinished = done_at.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        unfinished,
        0,
        "{} bidegrees never finished under {}",
        unfinished,
        plan.name()
    );
    // The last completion, not the last event: stale wakeups may fire later.
    let makespan = done_at.iter().copied().fold(0.0, f64::max);
    let busy: f64 = workers.iter().map(|w| w.ps.busy).sum();
    WholeMetrics {
        plan: plan.name()
            + &place.speed.name()
            + &place
                .max_open
                .map_or(String::new(), |m| format!(", <= {m} open walks"))
            + &if place.rank_epsilon != DagConfig::default().rank_epsilon {
                format!(", rank eps {}", place.rank_epsilon)
            } else {
                String::new()
            }
            + match place.pin {
                Pin::None => "",
                Pin::All => ", fast class only",
                Pin::Critical => ", critical pinned to fast (CPOP)",
            }
            + &match place.age_limit {
                Some(a) if !matches!(plan, Plan::Rank { .. }) => {
                    format!(", aging {:.0}s", a.as_secs_f64())
                }
                _ => String::new(),
            }
            + match place.group_key {
                GroupKey::Arrival => "",
                GroupKey::SMajor => ", groups by (s, t)",
                GroupKey::TMajor => ", groups by (t, s)",
                GroupKey::StemMajor => ", groups by (t - s, s)",
            },
        makespan_h: makespan / 3600.0,
        tasks,
        slot_util: busy / (slots_total as f64 * makespan).max(1e-9),
        bidegree_latency: Quantiles::of((0..n).map(|k| done_at[k] - ready_at[k]).collect()),
        peak_open,
        peak_dag_nodes: peak_nodes,
        dispatch_us: Quantiles::of(dispatch_us),
        sim_s: clock.elapsed().as_secs_f64(),
    }
}

/// A worker's per-task rate with `k` tasks running: its class's throughput, shared equally.
fn ps_rate(w: &str, model: &dyn ServiceModel, k: usize) -> f64 {
    model.throughput(w, k) / k as f64
}

/// Progress a worker's running tasks to `now` at their processor-sharing rate.
fn advance(w: &mut Wk, model: &dyn ServiceModel, now: f64) {
    w.ps.advance(now, |r| ps_rate(&w.class, model, r.len()));
}

/// Schedule a worker's next completion (invalidating any earlier one).
fn schedule(w: &mut Wk, idx: usize, model: &dyn ServiceModel, now: f64, queue: &mut Queue<Ev>) {
    if let Some((at, v)) =
        w.ps.next_completion(now, |r| ps_rate(&w.class, model, r.len()))
    {
        queue.push(at, Ev::Done(idx, v));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        model::{ClassCurve, PsModel},
        trace::{TraceTask, group_id},
    };

    /// A synthetic census over a small region.
    ///
    /// Dimensions grow geometrically per stem, some bidegrees have generators, and wall times
    /// follow the dimensions.
    fn census() -> Census {
        let mut c = Census {
            sources: vec!["synthetic".into()],
            ..Census::default()
        };
        for s in 0..=4 {
            for n in 0..=40 {
                let t = n + s;
                let p = algebra::optimal_profile(s, t, 5);
                let tmd = 10.0 * 1.2f64.powi(n);
                c.rows.insert(
                    (s, t),
                    CensusRow {
                        target_masked_dim: tmd,
                        next_dim: 2.0 * tmd,
                        gens: u32::from((s + n) % 3 == 0),
                        signatures: algebra::active_signatures(&p, t).len() as u32,
                        subalgebra_dim: 1 << p.iter().map(|&x| x as u32).sum::<u32>(),
                        wall_s: 1e-3 * tmd * (1.0 + (s as f64)),
                        source: 0,
                    },
                );
            }
        }
        c
    }

    /// A trace of one bidegree's zero step and signature tasks, work falling with degree.
    fn trace() -> (Trace, Vec<f64>) {
        let (n, s) = (30i64, 3i64);
        let p = algebra::optimal_profile(s as i32, (n + s) as i32, 5);
        let mut tasks = Vec::new();
        let mut work = Vec::new();
        let task = |req, zero, sig: Vec<u32>| TraceTask {
            req,
            zero,
            bidegree: (n, s),
            group: group_id(n, s),
            est_gb: 1.0,
            target: 1.0,
            next: 1.0,
            ready_s: 0.0,
            placed_s: 0.0,
            done_s: 1.0,
            worker: 0,
            deps: Vec::new(),
            after_groups: Vec::new(),
            sig,
        };
        tasks.push(task(0, true, Vec::new()));
        work.push(5.0);
        for i in algebra::active_signatures(&p, (n + s) as i32) {
            let (sig, deg) = algebra::signature(i, &p);
            tasks.push(task(1 + i as u64, false, sig));
            work.push(100.0 / deg as f64 * (1.0 + 0.1 * (i % 3) as f64));
        }
        (
            Trace {
                workers: Vec::new(),
                tasks,
            },
            work,
        )
    }

    /// Every plan finishes every bidegree, dispatches the same tasks, and respects the bounds.
    #[test]
    fn whole_run_plans_finish_within_bounds() {
        let census = census();
        let (trace, work) = trace();
        let world = World::build(
            WholeConfig {
                max_n: 40,
                max_s: 4,
                max_profile_len: 5,
                min_work: 0.01,
                noise_seed: 0,
            },
            &census,
            (&trace, &work),
        );
        assert!(
            world.cost.within[1] < 0.0,
            "work falls with degree: {:?}",
            world.cost.within
        );
        let model = PsModel {
            classes: BTreeMap::from([(
                "x".to_string(),
                ClassCurve {
                    speed: 1.0,
                    k_sat: 4,
                    alpha: 1.0,
                },
            )]),
        };
        let fleet = Fleet {
            groups: vec![("x".into(), 2, 4)],
        };
        let (cp, cap) = world.bounds(&fleet, &model);
        let plans = [
            Plan::Today {
                open: 2,
                per_bidegree: 3,
            },
            Plan::Group,
            Plan::Rank {
                oracle: false,
                age_limit: Some(Duration::from_secs(600)),
                group_first: false,
            },
            Plan::Rank {
                oracle: true,
                age_limit: None,
                group_first: true,
            },
        ];
        let results: Vec<WholeMetrics> = plans
            .iter()
            .map(|p| simulate(&world, &fleet, &model, p, &Placement::default()))
            .collect();
        let expected = world.summary().signature_tasks + world.bideg.len() as u64;
        for m in &results {
            assert_eq!(m.tasks, expected, "{}", m.plan);
            assert!(
                m.makespan_h * 3600.0 >= cp.max(cap) * (1.0 - 1e-9),
                "{}: beats a bound",
                m.plan
            );
        }
        assert_eq!(results[0].peak_open, 2);
        // The simulated open-walk budget, and pinning per signature, keep every task.
        for place in [
            Placement {
                max_open: Some(1),
                ..Placement::default()
            },
            Placement {
                pin: Pin::Critical,
                ..Placement::default()
            },
        ] {
            let m = simulate(&world, &fleet, &model, &Plan::Group, &place);
            assert_eq!(m.tasks, expected, "{}", m.plan);
            assert!(m.makespan_h * 3600.0 >= cp.max(cap) * (1.0 - 1e-9));
        }
        // Speed-aware placement on a mixed fleet; waiting for a fast slot must not drag the
        // makespan out to its wait limit, as counting stale wakeups after the last completion
        // would.
        let mixed = Fleet {
            groups: vec![("x".into(), 1, 4), ("y".into(), 1, 4)],
        };
        let model = PsModel {
            classes: BTreeMap::from([
                (
                    "x".to_string(),
                    ClassCurve {
                        speed: 1.0,
                        k_sat: 4,
                        alpha: 1.0,
                    },
                ),
                (
                    "y".to_string(),
                    ClassCurve {
                        speed: 3.0,
                        k_sat: 4,
                        alpha: 1.0,
                    },
                ),
            ]),
        };
        let place = |fast, defer| Placement {
            speed: SpeedPlan {
                fast,
                config: whelm::SpeedConfig {
                    defer,
                    ..whelm::SpeedConfig::default()
                },
            },
            ..Placement::default()
        };
        let fast = simulate(&world, &mixed, &model, &Plan::Group, &place(true, None));
        let defer = whelm::Defer {
            max_wait: Duration::from_secs(1_000_000),
            min_gain: 0.0,
        };
        let eft = simulate(
            &world,
            &mixed,
            &model,
            &Plan::Group,
            &place(true, Some(defer)),
        );
        assert_eq!(eft.tasks, expected, "{}", eft.plan);
        assert!(
            eft.makespan_h < 1.5 * fast.makespan_h,
            "{}: {} vs {}",
            eft.plan,
            eft.makespan_h,
            fast.makespan_h
        );
    }

    /// Missing census rows continue their row's exponential trend.
    #[test]
    fn dims_extrapolate_rows() {
        let mut c = census();
        c.rows.retain(|&(s, t), _| t - s <= 30);
        let d = Dims::build(&c, 4, 40);
        let want = 10.0 * 1.2f64.powi(40);
        assert!(
            (d.tmd(2, 42) / want - 1.0).abs() < 1e-6,
            "{} vs {want}",
            d.tmd(2, 42)
        );
        assert!(!d.known(2, 42) && d.known(2, 32));
    }
}
