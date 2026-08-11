//! Work/memory census for sizing the reach of a long run.
//!
//! Everything here answers a question of the form "how much of X is there", never "how long did X
//! take". That distinction is the whole design: a census run may be arbitrarily slower than
//! production without invalidating a single number, which is what lets us instrument heavily.
//!
//! Timing-dependent quantities (queue depth, per-device balance, speculation coverage) are
//! deliberately NOT collected here — under census slowdown they would measure the instrumentation.
//! They need their own clean run.
//!
//! Enable with `NASSAU_CENSUS=1`; `NASSAU_CENSUS_CSV=<path>` sets the per-bidegree log
//! (default `nassau_census.csv`). Everything is a relaxed atomic behind an `enabled()` check, so a
//! non-census build path costs one predictable branch.

use std::{
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

pub static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NASSAU_CENSUS").is_some());

#[inline]
pub fn enabled() -> bool {
    *ENABLED
}

/// Bytes copied, by site. Counted rather than timed: bytes divided by achievable bandwidth gives a
/// defensible floor for the copy cost without needing a clean-timing run.
pub static SELECT_ROWS_BYTES: AtomicU64 = AtomicU64::new(0);
pub static ADD_MASKED_BYTES: AtomicU64 = AtomicU64::new(0);
pub static AUGMENTED_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn add_bytes(counter: &AtomicU64, bytes: u64) {
    if enabled() {
        counter.fetch_add(bytes, Relaxed);
    }
}

/// One row per bidegree. This is the extrapolation substrate: with the dimensions and the byte
/// figures logged per `(s, t)`, both the work curve and the memory curve can be fitted offline and
/// pushed out to a target stem WITHOUT running that stem.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bidegree {
    pub s: i32,
    pub t: i32,
    /// Restricted source dimension (all signatures) — the rows we multiply today.
    pub target_dim: u64,
    /// Zero-signature rows only — the rows we would multiply under signature-shift reuse.
    pub target_masked_dim: u64,
    /// Restricted target dimension (all signatures) — the column count of `full_matrix`.
    pub next_dim: u64,
    /// Number of signatures of the chosen subalgebra at this bidegree.
    pub signatures: u64,
    /// Signature-space dimension of the subalgebra (`dim B`), for the 1/dim-B row-share check.
    pub subalgebra_dim: u64,
    pub num_new_gens: u64,
    /// Highest signature index at which any `dx` was still non-zero on entry. Everything after this
    /// index is a provable no-op under `EXT_NASSAU_NO_SAVE_QI`.
    pub last_live_sig: i64,
    /// Rows of `full_matrix` actually consumed (`Σ` over signatures of the correction support).
    /// The ratio `rows_consumed / target_dim` sizes the on-demand row cost of shift reuse.
    pub rows_consumed: u64,
    /// Dense bytes of the full restricted matrix at this bidegree (bit-packed, limb-padded).
    pub matrix_bytes: u64,
    pub wall_us: u64,
}

static RECORDS: LazyLock<Mutex<Vec<Bidegree>>> = LazyLock::new(|| Mutex::new(Vec::new()));

pub fn record(b: Bidegree) {
    if enabled() {
        RECORDS.lock().unwrap().push(b);
    }
}

/// Accumulator threaded through one bidegree's signature loop.
#[derive(Debug)]
pub struct BidegreeCensus {
    rec: Bidegree,
    start: Instant,
}

impl BidegreeCensus {
    pub fn new(s: i32, t: i32) -> Self {
        Self {
            rec: Bidegree {
                s,
                t,
                last_live_sig: -1,
                ..Default::default()
            },
            start: Instant::now(),
        }
    }

    pub fn dims(
        &mut self,
        target_dim: usize,
        target_masked_dim: usize,
        next_dim: usize,
        subalgebra_dim: usize,
    ) {
        self.rec.target_dim = target_dim as u64;
        self.rec.target_masked_dim = target_masked_dim as u64;
        self.rec.next_dim = next_dim as u64;
        self.rec.subalgebra_dim = subalgebra_dim as u64;
        // Bit-packed with rows padded to whole 64-bit limbs, matching `fp`'s layout.
        self.rec.matrix_bytes = (target_dim as u64) * (next_dim as u64).div_ceil(64) * 8;
    }

    pub fn set_new_gens(&mut self, n: usize) {
        self.rec.num_new_gens = n as u64;
    }

    /// Total signature count for the bidegree. Counted during the loop rather than by re-running
    /// `iter_signatures`, so the census does not change the enumeration work.
    pub fn set_signatures(&mut self, n: usize) {
        self.rec.signatures = n as u64;
    }

    /// Called once per signature index, before that signature's work, with whether any `dx` is
    /// still non-zero. Records the last index at which real work remained.
    pub fn sig_live(&mut self, sig_index: usize, live: bool) {
        if live {
            self.rec.last_live_sig = sig_index as i64;
        }
    }

    pub fn add_rows_consumed(&mut self, n: usize) {
        self.rec.rows_consumed += n as u64;
    }

    pub fn finish(mut self) {
        self.rec.wall_us = self.start.elapsed().as_micros() as u64;
        record(self.rec);
    }
}

fn vm_hwm_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

/// Write the per-bidegree CSV and print the aggregate block. Call once at the end of a run.
pub fn report() {
    if !enabled() {
        return;
    }
    let records = RECORDS.lock().unwrap();
    let path = std::env::var("NASSAU_CENSUS_CSV")
        .unwrap_or_else(|_| "nassau_census.csv".to_string());
    let mut csv = String::from(
        "s,t,target_dim,target_masked_dim,next_dim,signatures,subalgebra_dim,num_new_gens,\
         last_live_sig,rows_consumed,matrix_bytes,wall_us\n",
    );
    for r in records.iter() {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.s,
            r.t,
            r.target_dim,
            r.target_masked_dim,
            r.next_dim,
            r.signatures,
            r.subalgebra_dim,
            r.num_new_gens,
            r.last_live_sig,
            r.rows_consumed,
            r.matrix_bytes,
            r.wall_us,
        ));
    }
    if let Err(e) = std::fs::write(&path, csv) {
        eprintln!("[census] failed to write {path}: {e}");
    }

    // Work-weighted aggregates. Every fraction below is weighted by that bidegree's multiply work
    // (`target_dim * next_dim`), never by bidegree count — an event-weighted fraction would repeat
    // the "60% of bidegrees, 2% of time" mistake.
    let w = |r: &Bidegree| (r.target_dim as u128) * (r.next_dim as u128);
    let total_w: u128 = records.iter().map(w).sum();
    let zero_gen_w: u128 = records.iter().filter(|r| r.num_new_gens == 0).map(w).sum();
    let dead_sig_w: u128 = records
        .iter()
        .map(|r| {
            // Signature iterations after the last live `dx`, as a share of this bidegree's work.
            let dead = (r.signatures as i64 - 1 - r.last_live_sig).max(0) as u128;
            if r.signatures == 0 {
                0
            } else {
                w(r) * dead / r.signatures as u128
            }
        })
        .sum();
    let rows_all: u128 = records.iter().map(|r| r.target_dim as u128).sum();
    let rows_zero_sig: u128 = records.iter().map(|r| r.target_masked_dim as u128).sum();
    let rows_consumed: u128 = records.iter().map(|r| r.rows_consumed as u128).sum();
    let peak_matrix = records.iter().map(|r| r.matrix_bytes).max().unwrap_or(0);
    let pct = |a: u128, b: u128| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };

    eprintln!(
        "[census] bidegrees={} csv={path}\n\
         [census] work-weighted: zero_gen_bidegrees={:.1}% dead_signature_tail={:.1}%\n\
         [census] rows: computed={rows_all} zero_sig={rows_zero_sig} ({:.2}% of computed) \
         consumed_on_demand={rows_consumed} ({:.3}% of computed)\n\
         [census] shift-reuse rows needed = zero_sig + on_demand = {} ({:.2}% of computed)\n\
         [census] copy bytes: select_rows={:.1}GB add_masked={:.1}GB augmented_alloc={:.1}GB\n\
         [census] peak single dense matrix={:.2}GB  VmHWM={:.1}GB",
        records.len(),
        pct(zero_gen_w, total_w),
        pct(dead_sig_w, total_w),
        pct(rows_zero_sig, rows_all),
        pct(rows_consumed, rows_all),
        rows_zero_sig + rows_consumed,
        pct(rows_zero_sig + rows_consumed, rows_all),
        SELECT_ROWS_BYTES.load(Relaxed) as f64 / 1e9,
        ADD_MASKED_BYTES.load(Relaxed) as f64 / 1e9,
        AUGMENTED_ALLOC_BYTES.load(Relaxed) as f64 / 1e9,
        peak_matrix as f64 / 1e9,
        vm_hwm_kb() as f64 / 1e6,
    );
}
