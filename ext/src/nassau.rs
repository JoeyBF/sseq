//! This module implements [Nassau's algorithm](https://arxiv.org/abs/1910.04063).
//!
//! The main export is the [`Resolution`] object, which is a resolution of the sphere at the prime 2
//! using Nassau's algorithm. It aims to provide an API similar to
//! [`resolution::Resolution`](crate::resolution::Resolution). From an API point of view, the main
//! difference between the two is that our `Resolution` is a chain complex over [`MilnorAlgebra`]
//! over [`SteenrodAlgebra`](algebra::SteenrodAlgebra).
//!
//! To make use of this resolution in the example scripts, enable the `nassau` feature. This will
//! cause [`utils::query_module`](crate::utils::query_module) to return the `Resolution` from this
//! module instead of [`resolution`](crate::resolution). There is no formal polymorphism involved;
//! the feature changes the return type of the function. While this is an incorrect use of features,
//! we find that this the easiest way to make all scripts support both types of resolutions.

use std::{
    fmt::Display,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
};

use algebra::{
    Algebra, combinatorics,
    milnor_algebra::{MilnorAlgebra, PPart, PPartEntry},
    module::{
        FreeModule, GeneratorData, Module, ZeroModule,
        homomorphism::{FreeModuleHomomorphism, FullModuleHomomorphism, ModuleHomomorphism},
    },
};
use anyhow::anyhow;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use fp::{
    matrix::{AugmentedMatrix, Matrix, Subspace},
    prime::{Prime, TWO, ValidPrime},
    vector::{FpSlice, FpSliceMut, FpVector},
};
use itertools::Itertools;
// If `concurrent` is enabled, the `enumerate`/`for_each` used in `restricted_partial_matrix` come
// from `rayon::prelude::IndexedParallelIterator`, loaded via the `maybe_rayon` prelude. If it is
// disabled, `MaybeIndexedParallelIterator` implements `Iterator`, and those methods come from
// `std::iter::Iterator` instead, leaving this import unused — the same single code path either way.
// Mirrors `algebra::module::homomorphism`.
#[allow(unused_imports)]
use maybe_rayon::prelude::*;
use once::OnceBiVec;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

use crate::{
    chain_complex::{AugmentedChainComplex, ChainComplex, FiniteChainComplex, FreeChainComplex},
    save::{NassauCommand, NassauQiWriter, SaveDirectory, SaveKind},
    utils::parallel::ParallelGuard,
};

/// See [`resolution::SenderData`](../resolution/struct.SenderData.html). This differs by not having the `new` field.
struct SenderData {
    b: Bidegree,
    retry: bool,
    sender: mpsc::Sender<Self>,
}

impl SenderData {
    pub(crate) fn send(b: Bidegree, sender: mpsc::Sender<Self>) {
        sender
            .send(Self {
                b,
                retry: false,
                sender: sender.clone(),
            })
            .unwrap()
    }

    pub(crate) fn send_retry(b: Bidegree, sender: mpsc::Sender<Self>) {
        tracing::info!(%b, "retrying");
        sender
            .send(Self {
                b,
                retry: true,
                sender: sender.clone(),
            })
            .unwrap()
    }
}

const MAX_NEW_GENS: usize = 10;

/// A Milnor subalgebra to be used in [Nassau's algorithm](https://arxiv.org/abs/1910.04063). This
/// is equipped with an ordering of the signature as in Lemma 2.4 of the paper.
///
/// To simplify implementation, we pick the ordering so that the (reverse) lexicographic ordering
/// in Lemma 2.4 is just the (reverse) lexicographic ordering of the P parts. This corresponds to
/// the ordering of $\mathcal{P}$ where $P^s_t < P^{s'}_t$ if $s < s'$).
#[derive(Clone)]
struct MilnorSubalgebra {
    profile: Vec<u8>,
}

impl MilnorSubalgebra {
    /// This should be used when you want an entry of the profile to be infinity
    #[allow(dead_code)]
    const INFINITY: u8 = (std::mem::size_of::<PPartEntry>() * 4 - 1) as u8;

    fn new(profile: Vec<u8>) -> Self {
        Self { profile }
    }

    /// The algebra with trivial profile, corresponding to the trivial algebra.
    fn zero_algebra() -> Self {
        Self { profile: vec![] }
    }

    /// The test "does this element have this signature" compiled into a `(mask, value)` pair to
    /// match against the packed p-part.
    ///
    /// The per-entry test is `ppart[i] & ((1 << profile[i]) - 1) == signature[i]`. Because each
    /// entry occupies a fixed field of the packed word, the low `profile[i]` bits of entry `i` are
    /// a fixed bit range of that word, so the whole conjunction is a single `&` and `==`. Entries
    /// past the end of the p-part read as zero, which the packing already gives us for free.
    /// Returns `None` if no element can have this signature, which the caller turns into an empty
    /// result. A profile is not bounded by [`PPart::MAX_LEN`] -- `SubalgebraIterator` grows one
    /// without limit and `from_bytes` reads whatever length a file gives -- so both ways a
    /// signature can fail to be representable have to be handled here rather than assumed away.
    fn packed_signature(&self, signature: &[PPartEntry]) -> Option<(u64, u64)> {
        let mut mask = 0;
        let mut value = 0;
        for (i, (&profile, &entry)) in self.profile.iter().zip(signature).enumerate() {
            if i >= PPart::MAX_LEN {
                // No p-part of a representable degree has an entry this far out, so it reads as
                // zero: a non-zero constraint is unsatisfiable and a zero one is vacuous.
                if entry != 0 {
                    return None;
                }
                continue;
            }
            // A profile wider than the field constrains the whole field.
            let width = std::cmp::min(profile as u32, PPart::width(i));
            // The masked entry has only `width` bits, so a signature wanting more matches nothing.
            // Packing it anyway would spill into the neighbouring field.
            if (entry as u64) >> width != 0 {
                return None;
            }
            mask |= ((1u64 << width) - 1) << PPart::shift(i);
            value |= (entry as u64) << PPart::shift(i);
        }
        Some((mask, value))
    }

    fn zero_signature(&self) -> Vec<PPartEntry> {
        vec![0; self.profile.len()]
    }

    /// Give a list of basis elements in degree `degree` that has signature `signature`.
    ///
    /// Only basis elements coming from generators of degree strictly less than `max_gen_degree` are
    /// considered. Passing [`i32::MAX`] recovers the full mask. Restricting the generator degree is
    /// used by [`Resolution::compute_through_stem`] to read a module while ignoring the generators
    /// of the current internal degree, which may be added concurrently by another thread. Because
    /// generators are laid out in increasing degree, this is exactly a prefix of the full mask.
    ///
    /// This requires passing the algebra for borrow checker reasons.
    fn signature_mask<'a>(
        &'a self,
        algebra: &'a MilnorAlgebra,
        module: &'a FreeModule<MilnorAlgebra>,
        degree: i32,
        signature: &'a [PPartEntry],
        max_gen_degree: i32,
    ) -> impl Iterator<Item = usize> + 'a {
        // The mask depends only on the signature, so compute it once for the whole sweep (PR #280).
        // An unrepresentable signature yields no elements at all.
        //
        // `take_while` is retained from this branch and is NOT optional: `max_gen_degree` exists so
        // a reader can ignore the generators of the current internal degree, which another thread
        // may be adding concurrently. Generators are laid out in increasing degree, so this is
        // exactly a prefix.
        self.packed_signature(signature)
            .into_iter()
            .flat_map(move |(mask, value)| {
                module
                    .iter_gen_offsets([degree])
                    .take_while(move |gen_data| gen_data.gen_deg < max_gen_degree)
                    .flat_map(
                        move |GeneratorData {
                                  gen_deg,
                                  start: [offset],
                                  end: _,
                              }| {
                            algebra
                                .ppart_table(degree - gen_deg)
                                .iter()
                                .enumerate()
                                .filter_map(move |(n, op)| {
                                    if op.bits() & mask == value {
                                        Some(offset + n)
                                    } else {
                                        None
                                    }
                                })
                        },
                    )
            })
    }

    /// The number of basis elements in `degree` coming from generators of `module` of degree
    /// strictly less than `max_gen_degree`. This is the dimension of `module` in `degree` when we
    /// pretend the generators of degree `>= max_gen_degree` do not exist. Unlike
    /// [`Module::dimension`], it only reads generator counts that are frozen once the previous
    /// internal degrees have been committed, so it is safe to call while another thread is adding
    /// generators of the current internal degree.
    fn restricted_dimension(
        module: &FreeModule<MilnorAlgebra>,
        degree: i32,
        max_gen_degree: i32,
    ) -> usize {
        module
            .iter_gen_offsets([degree])
            .take_while(|gen_data| gen_data.gen_deg < max_gen_degree)
            .map(|gen_data| gen_data.end[0])
            .last()
            .unwrap_or(0)
    }

    /// Iterate through all signatures of this algebra that contain elements of degree at most
    /// `degree` (inclusive). This skips the initial zero signature.
    fn iter_signatures(&self, degree: i32) -> impl Iterator<Item = Vec<PPartEntry>> + '_ {
        SignatureIterator::new(self, degree)
    }

    /// Internal degree of a signature. `xi_i` has degree `2^i - 1`, so entry `idx` (which is
    /// `xi_{idx+1}`) carries weight `2^(idx+1) - 1` — the same weighting [`Self::top_degree`] uses.
    fn signature_degree(signature: &[PPartEntry]) -> i32 {
        signature
            .iter()
            .enumerate()
            .map(|(idx, &r)| ((1i32 << (idx + 1)) - 1) * r as i32)
            .sum()
    }

    fn top_degree(&self) -> i32 {
        self.profile
            .iter()
            .map(|&entry| (1 << entry) - 1)
            .enumerate()
            .map(|(idx, entry)| ((1 << (idx + 1)) - 1) * entry)
            .sum()
    }

    fn optimal_for(b: Bidegree) -> Self {
        let b_is_in_vanishing_region = |subalgebra: &Self| {
            let coeff = (1 << subalgebra.profile.len()) - 1;
            b.t() >= coeff * (b.s() + 1) + subalgebra.top_degree()
        };
        SubalgebraIterator::new()
            .take_while(b_is_in_vanishing_region)
            .last()
            .unwrap_or(Self::zero_algebra())
    }
}

impl Display for MilnorSubalgebra {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        if self.profile.is_empty() {
            write!(out, "F_2")
        } else if self.profile.len() as u8 == self.profile[0] {
            write!(out, "A({})", self.profile.len() - 1)
        } else {
            write!(out, "Algebra with profile {:?}", self.profile)
        }
    }
}

/// An iterator that iterates through a sequence of [`MilnorSubalgebra`] of increasing size. This
/// is used by [`MilnorSubalgebra::optimal_for`] to find the largest subalgebra in this sequence
/// that is applicable to a bidegree.
struct SubalgebraIterator {
    current: MilnorSubalgebra,
}

impl SubalgebraIterator {
    fn new() -> Self {
        Self {
            current: MilnorSubalgebra::new(vec![]),
        }
    }
}

impl Iterator for SubalgebraIterator {
    type Item = MilnorSubalgebra;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.profile.is_empty()
            || self.current.profile[0] == self.current.profile.len() as u8
        {
            // We are at F_2 or at A(n) where n = self.current.profile.len() - 1.
            self.current.profile.push(1);
            Some(self.current.clone())
        } else {
            // We find the first entry that can be incremented and increment it
            if let Some((_, entry)) = self
                .current
                .profile
                .iter_mut()
                .rev()
                .enumerate()
                .find(|(idx, entry)| **entry == *idx as u8)
            {
                *entry += 1;
            }
            Some(self.current.clone())
        }
    }
}

/// See [`MilnorSubalgebra::iter_signatures`].
struct SignatureIterator<'a> {
    subalgebra: &'a MilnorSubalgebra,
    current: Vec<PPartEntry>,
    signature_degree: i32,
    degree: i32,
}

impl<'a> SignatureIterator<'a> {
    fn new(subalgebra: &'a MilnorSubalgebra, degree: i32) -> Self {
        Self {
            current: vec![0; subalgebra.profile.len()],
            degree,
            subalgebra,
            signature_degree: 0,
        }
    }
}

impl Iterator for SignatureIterator<'_> {
    type Item = Vec<PPartEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        let xi_degrees = combinatorics::xi_degrees(TWO);
        let len = self.current.len();
        for (i, current) in self.current.iter_mut().enumerate() {
            *current += 1;
            self.signature_degree += xi_degrees[i];

            if self.signature_degree > self.degree || *current == 1 << self.subalgebra.profile[i] {
                self.signature_degree -= xi_degrees[i] * *current as i32;
                *current = 0;
                if i + 1 == len {
                    return None;
                }
            } else {
                return Some(self.current.clone());
            }
        }
        // This only happens when the profile is trivial
        assert!(self.current.is_empty());
        None
    }
}

/// Whether to persist quasi-inverses to disk during resolution. Disabled by
/// `EXT_NASSAU_NO_SAVE_QI`, in which case only the differentials are written (the quasi-inverses are
/// ~260-460x larger) and every downstream lift recomputes its quasi-inverse on demand.
static SAVE_QI: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("EXT_NASSAU_NO_SAVE_QI").is_none());

/// Force `apply_quasi_inverse` to recompute rather than read a saved quasi-inverse, even when one
/// exists on disk. Used to measure the recompute cost in isolation.
static RECOMPUTE_QI: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("EXT_NASSAU_RECOMPUTE_QI").is_some());

/// Build the matrix of `hom` on the basis elements `inputs`, with the target truncated to its first
/// `target_dim` basis elements. A free-function form of the restricted partial-matrix build (PR
/// #272): it does not read the (possibly concurrently growing) full target dimension, relying on
/// minimality so the truncated image loses nothing. This is the CPU reference and fallback.
fn restricted_partial_matrix(
    hom: &FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>,
    degree: i32,
    inputs: &[usize],
    target_dim: usize,
) -> Matrix {
    // Spanned because this is the fallback every build below `NASSAU_GPU_MIN_WORK` takes, and it
    // was invisible: the trace attributed 902.8 s to `extract_restricted` over 4915 GPU builds, but
    // a stem-150 run issues ~21 500 builds, so most of them landed here and were never counted.
    let _s = tracing::trace_span!("cpu_restricted", rows = inputs.len(), target_dim).entered();
    let mut matrix = Matrix::new(hom.prime(), inputs.len(), target_dim);
    if target_dim > 0 {
        matrix
            .maybe_par_iter_mut()
            .enumerate()
            .for_each(|(i, row)| hom.apply_to_basis_element_restricted(row, 1, degree, inputs[i]));
    }
    matrix
}

/// CPU restricted build with the column mask applied as it goes: `out[i][j]` is the restricted
/// apply of `inputs[i]` read at column `col_mask[j]`.
///
/// The scratch is one full-width ROW per worker (via `map_init`), never a full-width matrix, so the
/// wide object the mask exists to avoid is never allocated.
fn restricted_partial_matrix_masked(
    hom: &FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>,
    degree: i32,
    inputs: &[usize],
    target_dim: usize,
    col_mask: &[usize],
) -> Matrix {
    let _s = tracing::trace_span!(
        "cpu_restricted_masked",
        rows = inputs.len(),
        target_dim,
        cols = col_mask.len()
    )
    .entered();
    let p = hom.prime();
    let mut matrix = Matrix::new(p, inputs.len(), col_mask.len());
    if target_dim == 0 || col_mask.is_empty() {
        return matrix;
    }
    // Rows are independent and land in distinct matrix rows, so this parallelises exactly as the
    // unmasked `restricted_partial_matrix` above already does. It was left serial while it only
    // served builds below `NASSAU_GPU_MIN_WORK`; if the multiply moves to the CPU it becomes the
    // hot path, on a machine where the DAG leaves ~118 cores idle.
    matrix
        .maybe_par_iter_mut()
        .enumerate()
        .for_each(|(i, mut row)| {
            let mut scratch = FpVector::new(p, target_dim);
            hom.apply_to_basis_element_restricted(scratch.as_slice_mut(), 1, degree, inputs[i]);
            row.add_masked(scratch.as_slice(), 1, col_mask);
        });
    matrix
}

/// Restricted partial-matrix build with the destination narrowed to `col_mask`.
///
/// Every consumer of a restricted partial matrix immediately masks its columns; the mask keeps
/// ~2% of them at the frontier, so building the full width first is pure waste. Measured on the
/// running instrumented job at stem 285: 37703 x 1914035 is 9.0 GB, against 183 MB masked, and the
/// frontier's bidegrees build 143 GB to use 7.5 GB. That 143 GB matches the 142.9 GB a heap dump
/// independently attributes to in-flight matrices.
///
/// `NASSAU_MASKED_COLS=0` restores the full-width build for A/B and bisection.
fn restricted_partial_matrix_masked_maybe_gpu(
    diff: &FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>,
    t: i32,
    inputs: &[usize],
    target_dim: usize,
    col_mask: &[usize],
) -> Matrix {
    #[cfg(feature = "gpu")]
    {
        if std::env::var_os("NASSAU_GPU").is_some() && crate::nassau_gpu::applicable(diff) {
            let min_work: u64 = std::env::var("NASSAU_GPU_MIN_WORK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4_000_000);
            // Gate on the work the MULTIPLY does, which is still full width -- masking narrows the
            // destination, not the launch.
            let work = inputs.len() as u64 * target_dim as u64;
            if work >= min_work {
                return if std::env::var_os("NASSAU_GPU_VERIFY").is_some() {
                    crate::nassau_gpu::get_partial_matrix_restricted_masked_verified(
                        diff, t, inputs, target_dim, col_mask,
                    )
                } else {
                    crate::nassau_gpu::get_partial_matrix_restricted_masked(
                        diff, t, inputs, target_dim, col_mask,
                    )
                };
            }
        }
    }
    restricted_partial_matrix_masked(diff, t, inputs, target_dim, col_mask)
}

/// Process RSS in bytes, from `/proc/self/status`. Returns 0 where that is unavailable.
fn proc_rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or(0, |kb| kb * 1024)
}

/// EXACT jemalloc totals -- not sampled -- and the process RSS they sit inside.
///
/// This exists to size two things that a heap *profile* cannot separate:
///
/// * **Sampling bias.** `lg_prof_sample:23` samples one allocation per 8MB, so a long tail of
///   small allocations is under-represented in the profile. `allocated` counts every byte, so
///   `allocated - (summed profile stacks)` is the tail the profiler misses.
/// * **Memory outside the Rust heap.** The CUDA driver and cubecl allocate host memory with their
///   own mmap, never through the global allocator, so heap profiling cannot see them *by
///   construction* -- not because of a sampling threshold. `RSS - resident` sizes that.
///
/// Reading these without advancing the epoch returns stale cached values, which would look like a
/// workload that allocates nothing.
#[cfg(feature = "heapprof")]
fn heap_stats_report() {
    use tikv_jemalloc_ctl::{epoch, stats};
    let _ = epoch::advance();
    let gb = |b: usize| b as f64 / 1e9;
    let allocated = stats::allocated::read().unwrap_or(0);
    let active = stats::active::read().unwrap_or(0);
    let resident = stats::resident::read().unwrap_or(0);
    let mapped = stats::mapped::read().unwrap_or(0);
    let rss = proc_rss_bytes();
    eprintln!(
        "[HEAP] jemalloc allocated={:.1} active={:.1} resident={:.1} mapped={:.1}GB | \
         RSS={:.1}GB | outside-jemalloc={:.1}GB",
        gb(allocated),
        gb(active),
        gb(resident),
        gb(mapped),
        rss as f64 / 1e9,
        (rss as f64 - resident as f64) / 1e9,
    );
}

#[cfg(not(feature = "heapprof"))]
fn heap_stats_report() {}


/// Whether to build restricted partial matrices directly in masked column coordinates. Default ON;
/// `NASSAU_MASKED_COLS=0` disables.
fn masked_cols_enabled() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var("NASSAU_MASKED_COLS").as_deref() != Ok("0"));
    *ON
}

/// Restricted partial-matrix build, dispatching to the GPU Milnor-multiply path when it is compiled
/// in, opted into (`NASSAU_GPU`), applicable, and the launch is large enough to amortise the fixed
/// per-launch GPU cost.
///
/// Defaults to the CPU [`restricted_partial_matrix`], so behaviour is unchanged unless a caller sets
/// `NASSAU_GPU`. A resolution issues thousands of small signature-masked launches (avg ~10³
/// term-pairs), for which the GPU's per-launch overhead (kernel launch + readback sync, ~0.7 ms)
/// dwarfs the multiply; only launches whose `rows × cols` (`inputs.len() * target_dim`) exceeds
/// `NASSAU_GPU_MIN_WORK` (default 4M) are offloaded. `NASSAU_GPU_VERIFY` builds the CPU matrix too
/// and asserts they agree. Without the `gpu` feature this is exactly [`restricted_partial_matrix`].
fn restricted_partial_matrix_maybe_gpu(
    diff: &FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>,
    t: i32,
    inputs: &[usize],
    target_dim: usize,
) -> Matrix {
    #[cfg(feature = "gpu")]
    {
        if std::env::var_os("NASSAU_GPU").is_some() && crate::nassau_gpu::applicable(diff) {
            // Below this `rows x cols`, the CPU beats a GPU launch.
            //
            // REGIME-DEPENDENT, and the crossover sits between stem 130 and 150. Measured at
            // theta=0, interleaved, timing runs to /dev/null (the differentials dump is 400-600MB
            // at stem 150 and writing it to panfs swamps the measurement):
            //
            //     stem 110    4e6  13.7/13.3s      2.56e8   6.09/6.22s     CPU-only  ~6.3s
            //     stem 130    4e6  53.0/52.1s      2.56e8  27.1/26.7s      CPU-only ~27.8s
            //     stem 150    4e6 168.2/169.9s     2.56e8 523.8/512.4s     CPU-only 505/521s
            //     stem 170    4e6 496.5s           2.56e8 1452.6s
            //
            // It INVERTS: 2.2x better at stem 110-130, 3.1x WORSE at 150 and 2.9x worse at 170.
            // The deciding factor is not matrix size (which is what this knob measures) but
            // submission concurrency -- `depth mean` is 1.6 at stem 130 and 7.5 at stem 150, so at
            // high stem the pipeline is deep enough to amortise per-launch overhead that dominates
            // at low stem. A fixed size threshold cannot express that.
            //
            // Kept at 4e6 because the target regime is high stem: stems 110-130 run in seconds
            // either way, and stem 150+ is where a 3x matters. Set `NASSAU_GPU_MIN_WORK=256000000`
            // for small-stem work. Making this adaptive on queue depth is the real fix.
            let min_work: u64 = std::env::var("NASSAU_GPU_MIN_WORK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4_000_000);
            let work = inputs.len() as u64 * target_dim as u64;
            if work >= min_work {
                return if std::env::var_os("NASSAU_GPU_VERIFY").is_some() {
                    crate::nassau_gpu::get_partial_matrix_restricted_verified(
                        diff, t, inputs, target_dim,
                    )
                } else {
                    crate::nassau_gpu::get_partial_matrix_restricted(diff, t, inputs, target_dim)
                };
            }
        }
    }
    restricted_partial_matrix(diff, t, inputs, target_dim)
}

/// Whether to compute the full restricted differential matrix once per bidegree and reuse row
/// slices across the signature passes, instead of relaunching the multiply once per signature.
///
/// Each signature's partial matrix is a *row subset* of one full matrix — the signature masks
/// partition the (restricted) source basis, so the per-signature builds together compute every row
/// exactly once, the same total multiply work as one all-rows build. On the CPU that restructuring
/// is roughly neutral, but for the GPU it turns thousands of small (often sub-threshold,
/// CPU-fallback) launches into one big launch per bidegree that amortises all fixed per-launch
/// overhead. So it is gated on the same opt-in as the GPU path; without it the per-signature build
/// is unchanged.
fn reuse_full_matrix(_diff: &FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>) -> bool {
    #[cfg(feature = "gpu")]
    {
        std::env::var_os("NASSAU_GPU").is_some() && crate::nassau_gpu::applicable(_diff)
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Max `rows × cols` of the full restricted matrix for which [`reuse_full_matrix`] builds it all at
/// once. Above this the all-rows build (and its dense GPU readback, both held across the whole
/// signature loop) dominates host memory at high stems — the ~12 GB dense regions behind the stem-180
/// OOM. Past the cap we fall back to per-signature builds (each a bounded row subset, like the CPU),
/// trading a little launch amortization for a peak that scales with the largest single signature
/// rather than the whole bidegree. `NASSAU_GPU_REUSE_MAX_WORK` overrides (0 = never reuse). Default
/// ~1e10 (rows×cols) ≈ a ~1.2 GB restricted matrix.
#[cfg(feature = "gpu")]
fn gpu_reuse_max_work() -> u64 {
    static W: std::sync::LazyLock<u64> = std::sync::LazyLock::new(|| {
        std::env::var("NASSAU_GPU_REUSE_MAX_WORK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000_000_000)
    });
    *W
}

/// Whether the full restricted matrix (`rows × cols`) is small enough to build all at once (see
/// [`gpu_reuse_max_work`]). Always true without the `gpu` feature (the reuse path is off anyway).
fn reuse_within_cap(_rows: usize, _cols: usize) -> bool {
    #[cfg(feature = "gpu")]
    {
        let w = gpu_reuse_max_work();
        w > 0 && (_rows as u64).saturating_mul(_cols as u64) <= w
    }
    #[cfg(not(feature = "gpu"))]
    {
        true
    }
}

/// Probe (`NASSAU_PROBE_SHIFT=1`) for signature-shift reuse, the next candidate lever: the census
/// says only 33.35% of the rows we multiply are needed if a signature-`sigma` problem at internal
/// degree `t` is the ZERO-signature problem at `t - deg(sigma)`.
///
/// That is the design the reference implementation uses (`solve_signature_lifts` /
/// `signature_to_zero_translation`): it never rebuilds the matrix for a nonzero signature, it
/// looks up the zero-signature solver at the shifted degree and relabels the basis around it.
///
/// This records each bidegree's zero-signature `(target_masked_dim, next_masked_dim)` and then,
/// for every nonzero signature, checks those dims against the shifted bidegree's. Dimension
/// equality is only a NECESSARY condition — it does not prove the linear maps agree — but it is
/// cheap and it is decisive in the negative: a single mismatch kills the lever outright.
static SHIFT_MATCH: AtomicUsize = AtomicUsize::new(0);
static SHIFT_MISMATCH: AtomicUsize = AtomicUsize::new(0);
static SHIFT_ABSENT: AtomicUsize = AtomicUsize::new(0);
/// Launch-site counters (`NASSAU_SHIFT_STATS=1`). After shift reuse the run issues 4.3x MORE
/// launches at 13x smaller size (stem 130: 39000 calls, 331 Rs each, `waves/SM` 0.017), and
/// readback is 80% of worker time. Batching is the fix, but only after knowing WHICH site emits
/// them: the shifted builds are deterministic and hoistable, the on-demand lift rows are
/// data-dependent and are not.
static N_SHIFT_BUILD: AtomicUsize = AtomicUsize::new(0);
static N_SHIFT_HIT: AtomicUsize = AtomicUsize::new(0);
static N_ONDEMAND: AtomicUsize = AtomicUsize::new(0);
static N_ZS: AtomicUsize = AtomicUsize::new(0);
static ROWS_SHIFT_BUILD: AtomicUsize = AtomicUsize::new(0);
static ROWS_ONDEMAND: AtomicUsize = AtomicUsize::new(0);
/// Of the shift builds, the ones that may NOT publish (`signature_degree == 1`): they build at
/// their own narrower bound and are dropped unshared, so each is a private per-signature matrix
/// held live for the duration of the signature. The heap dump attributes ~164GB to two
/// `restricted_partial_matrix_maybe_gpu` stacks and this is the candidate; these counters decide
/// it. Bytes, not just rows, because the question is memory -- see [`crate::census::matrix_bytes`].
static N_SHIFT_PRIVATE: AtomicUsize = AtomicUsize::new(0);
static ROWS_SHIFT_PRIVATE: AtomicUsize = AtomicUsize::new(0);
static BYTES_SHIFT_PRIVATE: AtomicU64 = AtomicU64::new(0);
static BYTES_SHIFT_PUBLISHED: AtomicU64 = AtomicU64::new(0);
static BYTES_ONDEMAND: AtomicU64 = AtomicU64::new(0);
/// Live bytes in private shift builds right now, and the high-water mark of that -- the number
/// that has to come down. Incremented at build, decremented when the `Arc` is dropped.
static LIVE_PRIVATE: AtomicU64 = AtomicU64::new(0);
static LIVE_PRIVATE_MAX: AtomicU64 = AtomicU64::new(0);

/// Charges a private (unpublishable) shift build against [`LIVE_PRIVATE`] for exactly as long as
/// the matrix is held, so the high-water mark reflects real concurrent residency rather than a
/// running total. Bound as a sibling of `shifted`, so it drops when that matrix does.
struct PrivateLive(u64);

impl PrivateLive {
    fn new(bytes: u64) -> Self {
        let now = LIVE_PRIVATE.fetch_add(bytes, Ordering::Relaxed) + bytes;
        LIVE_PRIVATE_MAX.fetch_max(now, Ordering::Relaxed);
        Self(bytes)
    }
}

impl Drop for PrivateLive {
    fn drop(&mut self) {
        LIVE_PRIVATE.fetch_sub(self.0, Ordering::Relaxed);
    }
}

fn shift_stats_report() {
    if std::env::var("NASSAU_SHIFT_STATS").as_deref() != Ok("1") {
        return;
    }
    const GB: f64 = (1u64 << 30) as f64;
    let (build, private) = (
        N_SHIFT_BUILD.load(Ordering::Relaxed),
        N_SHIFT_PRIVATE.load(Ordering::Relaxed),
    );
    eprintln!(
        "[shift-stats] launches: zs_select={} shift_build={build} (rows {}) ondemand={} (rows {}) \
         | cache hits={}",
        N_ZS.load(Ordering::Relaxed),
        ROWS_SHIFT_BUILD.load(Ordering::Relaxed),
        N_ONDEMAND.load(Ordering::Relaxed),
        ROWS_ONDEMAND.load(Ordering::Relaxed),
        N_SHIFT_HIT.load(Ordering::Relaxed),
    );
    // The memory question: of the shift builds, how many could not be published (`deg(sigma) ==
    // 1`) and so were held privately per signature?
    eprintln!(
        "[shift-stats] private (deg-sig 1, unshared): {private}/{build} builds ({:.1}%) rows={} | \
         bytes built: private={:.3}GB published={:.3}GB ondemand={:.3}GB | live private: \
         now={:.3}GB peak={:.3}GB",
        if build > 0 {
            100.0 * private as f64 / build as f64
        } else {
            0.0
        },
        ROWS_SHIFT_PRIVATE.load(Ordering::Relaxed),
        BYTES_SHIFT_PRIVATE.load(Ordering::Relaxed) as f64 / GB,
        BYTES_SHIFT_PUBLISHED.load(Ordering::Relaxed) as f64 / GB,
        BYTES_ONDEMAND.load(Ordering::Relaxed) as f64 / GB,
        LIVE_PRIVATE.load(Ordering::Relaxed) as f64 / GB,
        LIVE_PRIVATE_MAX.load(Ordering::Relaxed) as f64 / GB,
    );
}

static SHIFT_MAT_MATCH: AtomicUsize = AtomicUsize::new(0);
static SHIFT_MAT_MISMATCH: AtomicUsize = AtomicUsize::new(0);

fn shift_probe_enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NASSAU_PROBE_SHIFT").is_some());
    *ON
}

/// `NASSAU_PROBE_SHIFT=2`: also rebuild the zero-signature matrix at the shifted degree and diff
/// it against this signature's. This is the SUFFICIENT check that dimension equality is not —
/// and it is expensive (an extra restricted multiply per signature), so run it at small stem.
fn shift_probe_matrices() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var("NASSAU_PROBE_SHIFT").as_deref() == Ok("2"));
    *ON
}

fn shift_probe_report() {
    if !shift_probe_enabled() {
        return;
    }
    let (m, x, a) = (
        SHIFT_MATCH.load(Ordering::Relaxed),
        SHIFT_MISMATCH.load(Ordering::Relaxed),
        SHIFT_ABSENT.load(Ordering::Relaxed),
    );
    let total = m + x;
    eprintln!(
        "[shift-probe] signature instances: dims_match={m} dims_MISMATCH={x} ({:.2}% match) \
         shifted_bidegree_absent={a}",
        if total == 0 {
            0.0
        } else {
            100.0 * m as f64 / total as f64
        }
    );
    if shift_probe_matrices() {
        let (mm, mx) = (
            SHIFT_MAT_MATCH.load(Ordering::Relaxed),
            SHIFT_MAT_MISMATCH.load(Ordering::Relaxed),
        );
        let mt = mm + mx;
        eprintln!(
            "[shift-probe] MATRICES: match={mm} MISMATCH={mx} ({:.2}% match)",
            if mt == 0 {
                0.0
            } else {
                100.0 * mm as f64 / mt as f64
            }
        );
    }
}

/// Cross-bidegree cache of ZERO-SIGNATURE restricted matrices, the substrate for shift reuse
/// (`NASSAU_SHIFT_REUSE=1`).
///
/// Proven by `NASSAU_PROBE_SHIFT=2` (0 mismatches in 16056 signature instances): the masked matrix
/// for signature `sigma` at internal degree `t` IS the zero-signature masked matrix at
/// `t - deg(sigma)`, row for row, with no permutation. So every signature's matrix can come from
/// one per-degree zero-signature build shared across every bidegree that shifts onto it, instead of
/// being rebuilt inside each bidegree. The census sizes what survives at 33.35% of rows.
///
/// # The key must include the SUBALGEBRA
///
/// `MilnorSubalgebra::optimal_for` picks a profile per bidegree, so two consumers landing on the
/// same `(s, shifted_t)` need not share a subalgebra — and a different profile means a different
/// zero signature, hence a different mask at the same degree. Keying on `(s, degree)` alone hands
/// an entry built under one profile to a consumer using another. That is not a subtle aliasing
/// bug: at `(15,3) sig=[2,0] shifted_t=16` it produced a 14-row matrix where the consumer's mask
/// had 9 rows. It looked like a monotonicity violation (a larger generator bound yielding FEWER
/// rows, which is impossible) precisely because the two counts came from unrelated subalgebras.
///
/// # Why the producer must be a consumer
///
/// The entry for degree `d` must include rows from generators of degree EXACTLY `d`, because a
/// consumer at `t = d + deg(sigma)` masks with `target_bound = t > d` and so sees them. Bidegree
/// `(s, d)` cannot build that: it deliberately uses `target_bound = d`, excluding those very
/// generators, precisely so it can run concurrently with `(s - 1, d)` which is still adding them.
///
/// A consumer is safe where the producer is not. A consumer at `t >= d + 1` only runs once
/// `progress[s - 1] >= t - 1 >= d`, i.e. once `(s - 1, d)` has committed, so the generators are
/// frozen by the time it asks. Hence entries are filled lazily on first demand from a consumer and
/// never by the bidegree they are named after. Getting this backwards is what made the first
/// dimension probe read 21% instead of 100%.
#[cfg(feature = "gpu")]
mod shift {
    use std::sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use fp::matrix::Matrix;
    use rustc_hash::FxHashMap;

    /// `(s, degree, subalgebra profile) -> zero-signature restricted matrix at that degree.`
    type Key = (i32, i32, Vec<u8>);

    static CACHE: LazyLock<Mutex<FxHashMap<Key, Arc<Matrix>>>> =
        LazyLock::new(|| Mutex::new(FxHashMap::default()));
    static BYTES: AtomicUsize = AtomicUsize::new(0);

    /// Byte ceiling for this cache (`NASSAU_SHIFT_CACHE_GB`, default 0 = unlimited).
    ///
    /// DEFAULT IS UNLIMITED so no run gets slower by default; this is a TIME-FOR-MEMORY DIAL to
    /// reach for when a run is near the memory wall, not a free win. Measured at stem 170:
    ///
    /// | cap      | wall   | host RSS | cache |
    /// |----------|--------|----------|-------|
    /// | unlimited| 496.5s | 50.9 GB  | 26.8  |
    /// | 8 GB     | 636.3s | 38.1 GB  |  8.0  |
    /// | 4 GB     | 740.5s | 34.0 GB  |  4.0  |
    ///
    /// Non-cache memory is ~24 GB, so ~30 GB is the floor: at most ~1.7x memory for ~1.5x time.
    /// Since memory grows ~2.6x per 20 stems and time ~3.5x, 1.7x of memory is worth roughly 8
    /// stems of reach against a time cost worth about 6 -- marginally positive only when memory,
    /// not time, is what stops the run.
    ///
    /// Unbounded, this was the run's LARGEST memory consumer and the fastest-growing one:
    /// 7.15 GB of 16.2 GB RSS at stem 150 (44%), 26.8 GB of 50.3 GB at stem 170 (53%), growing
    /// 3.75x per 20 stems against RSS's 3.1x. Since memory -- not time -- is what bounds reachable
    /// stems, capping it buys REACH.
    ///
    /// Eviction is by LOWEST DEGREE first, which is exact rather than heuristic: an entry for
    /// degree `d` serves consumers at `t = d + deg(sigma)`, and the wavefront never returns to a
    /// degree it has passed, so the smallest degrees are always the deadest. A miss merely rebuilds
    /// the entry, so eviction can never change results.
    fn cache_bytes_cap() -> usize {
        static N: LazyLock<usize> = LazyLock::new(|| {
            let gb: f64 = std::env::var("NASSAU_SHIFT_CACHE_GB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            if gb <= 0.0 {
                usize::MAX
            } else {
                (gb * 1e9) as usize
            }
        });
        *N
    }

    /// Default ON; `NASSAU_SHIFT_REUSE=0` disables.
    pub(super) fn enabled() -> bool {
        static ON: LazyLock<bool> =
            LazyLock::new(|| std::env::var("NASSAU_SHIFT_REUSE").as_deref() != Ok("0"));
        *ON
    }

    pub(super) fn get(s: i32, degree: i32, profile: &[u8]) -> Option<Arc<Matrix>> {
        CACHE
            .lock()
            .unwrap()
            .get(&(s, degree, profile.to_vec()))
            .map(Arc::clone)
    }

    pub(super) fn put(s: i32, degree: i32, profile: &[u8], m: Arc<Matrix>) {
        let size = (m.rows()) * (m.columns()).div_ceil(8);
        let mut c = CACHE.lock().unwrap();
        if c.insert((s, degree, profile.to_vec()), m).is_none() {
            BYTES.fetch_add(size, Ordering::Relaxed);
        }
        let cap = cache_bytes_cap();
        if cap != usize::MAX && BYTES.load(Ordering::Relaxed) > cap {
            // Lowest degree first: the wavefront never comes back for those.
            let mut keys: Vec<Key> = c.keys().cloned().collect();
            keys.sort_by_key(|k| k.1);
            for k in keys {
                if BYTES.load(Ordering::Relaxed) <= cap {
                    break;
                }
                if let Some(old) = c.remove(&k) {
                    let freed = old.rows() * old.columns().div_ceil(8);
                    BYTES.fetch_sub(freed.min(BYTES.load(Ordering::Relaxed)), Ordering::Relaxed);
                }
            }
        }
    }

    /// Drop everything once the wavefront is done, mirroring [`super::blocks::clear`].
    pub(super) fn clear() {
        CACHE.lock().unwrap().clear();
        BYTES.store(0, Ordering::Relaxed);
    }

    /// Entries and approximate bytes held. This cache NEVER EVICTS during a run, so it is a prime
    /// suspect for the steady RSS climb (stem 170: 0 -> 49 GB monotone, peak single dense matrix
    /// only 4 GB of a 51.4 GB high-water mark -- i.e. retained state, not transient matrices).
    pub(super) fn stats() -> (usize, u64) {
        let c = CACHE.lock().unwrap();
        let bytes: u64 = c
            .values()
            .map(|m| (m.rows() as u64) * (m.columns() as u64).div_ceil(8))
            .sum();
        (c.len(), bytes)
    }
}

/// One signature's partial differential matrix, which is either built for that signature alone or
/// is a row-subset of a cached full matrix (see [`reuse_full_matrix`]).
///
/// The subset case used to be materialised by a `select_rows` helper: allocate `rows.len()`
/// rows and `assign` each one out of `full`. That copy was pure overhead — a row gather is just an
/// index indirection, and every consumer (`add_masked` into the augmented matrix, the lift, and
/// [`Resolution::write_qi`]) reads rows one at a time and never needs them contiguous. The census
/// measured it at **7.2 GB copied at stem 110 alone**.
///
/// It buys no wall time — measured below noise at stem 110 (19.44 s vs 19.42 s) and at stem 140
/// uncapped (72.2 s vs 74.3 s, ~3% noise floor). That is expected rather than disappointing: the
/// run sits at 21-30% CPU and is release-limited, so CPU work removed off the critical path does
/// not shorten it. Keep the change anyway — strictly less allocation and memory traffic for
/// byte-identical output — but do not expect copy elimination alone to move this workload.
///
/// `ADD_MASKED_BYTES` and `AUGMENTED_ALLOC_BYTES` are now wired, at the `zs_assemble` and
/// `sig_assemble` blocks below — one increment per assembled matrix, never per row, since those
/// loops run up to `target_dim` times. They previously printed `0.0GB` with no incrementing call
/// site, which read as "never copied" rather than "never instrumented". The `add_masked` into the
/// augmented matrix and the per-signature `AugmentedMatrix::new` are real copies — just
/// unavoidable ones, since row reduction needs a mutable working matrix — and the per-signature
/// one runs ~1000 times per frontier bidegree, so it is the term that matters.
///
/// `rows` must index within `full`; the signature masks are subsets of `0..full.rows()` (the
/// restricted source basis), so this holds.
enum PartialMatrix<'a> {
    Owned(Matrix),
    /// Already built in masked column coordinates: `row(i)` has one entry per mask index, so the
    /// consumer must NOT mask it again. See [`restricted_partial_matrix_masked_maybe_gpu`].
    PreMasked(Matrix),
    /// `row(i) == full.row(rows[i])`, without materialising the gather.
    Gather {
        full: &'a Matrix,
        rows: &'a [usize],
    },
}

impl PartialMatrix<'_> {
    fn columns(&self) -> usize {
        match self {
            Self::Owned(m) | Self::PreMasked(m) => m.columns(),
            Self::Gather { full, .. } => full.columns(),
        }
    }

    fn row(&self, i: usize) -> FpSlice<'_> {
        match self {
            Self::Owned(m) | Self::PreMasked(m) => m.row(i),
            Self::Gather { full, rows } => full.row(rows[i]),
        }
    }

    /// Whether `row(i)` is already in masked coordinates.
    fn is_pre_masked(&self) -> bool {
        matches!(self, Self::PreMasked(_))
    }

    /// `dst += self.row(i)`, masked by `mask` unless the matrix was already built that way.
    /// Centralised so a caller cannot forget which coordinates it is holding.
    fn add_row_masked(&self, dst: FpSliceMut<'_>, i: usize, mask: &[usize]) {
        let mut dst = dst;
        if self.is_pre_masked() {
            dst.add(self.row(i), 1);
        } else {
            dst.add_masked(self.row(i), 1, mask);
        }
    }
}

/// Speculative full-matrix precomputation: build a bidegree's full restricted differential matrix
/// *before* the bidegree runs, on background threads, and hand it over when the bidegree starts.
///
/// # Why this is sound
///
/// `step_resolution_with_subalgebra` deliberately treats `C_{s-1}` as having no generators of degree
/// `>= t` and `C_{s-2}` as having none of degree `>= t - 1`, so the full matrix it builds reads only
/// data frozen once `(s - 1, t - 1)` is committed — see the comment there. That is *strictly weaker*
/// than the condition for running `(s, t)`, which additionally needs `(s, t - 1)`. The gap between
/// the two is the speculation window: whenever row `s` lags row `s - 1`, every bidegree in between
/// already has a fully determined matrix and nothing but scheduling stops us from building it.
///
/// The window is widest exactly where it pays. The bottleneck in practice is the low-`s` rows, and
/// for those the rows below are long since finished — row 2's matrices are all computable the moment
/// row 1 completes. So the lookahead available to the critical path is bounded by `NASSAU_SPECULATE`
/// depth, not by the wavefront.
///
/// The row-block claim underneath (a matrix built over a row subset equals those rows of the all-rows
/// build) is checked empirically by `NASSAU_SPLIT_VERIFY`; this cache goes further and builds the
/// whole matrix early, which `NASSAU_SPECULATE_VERIFY` checks by rebuilding at consumption time and
/// asserting equality.
///
/// # Why it should win at capped theta
///
/// At high stems the resident master must be capped, so most `R`'s are enumerated on the GPU on the
/// critical path, and an enumeration launch costs the length of its longest odometer chain. Building
/// matrices ahead moves that latency off the critical path entirely: the speculative threads issue
/// their launches while the wavefront is busy elsewhere, and a bidegree that finds its matrix in the
/// cache skips the multiply *and* the enumeration behind it.
///
/// # Configuration
///
/// * `NASSAU_SPECULATE` — number of background builder threads (default 0, off).
/// * `NASSAU_SPECULATE_AHEAD` — how many degrees past a row's own frontier to speculate (default 64).
/// * `NASSAU_SPECULATE_MAX_GB` — cap on retained speculative matrices (default 64 GB).
/// * `NASSAU_SPECULATE_VERIFY` — rebuild at consumption and assert the cached matrix is identical.
mod speculate {
    use std::{
        cmp::Reverse,
        collections::{BinaryHeap, HashMap},
        sync::{
            Condvar, LazyLock, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use fp::matrix::Matrix;
    use sseq::coordinates::Bidegree;

    /// Number of background builder threads; 0 disables speculation entirely.
    pub fn threads() -> usize {
        static N: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("NASSAU_SPECULATE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        });
        *N
    }

    /// How far past a row's own committed frontier to speculate. Bounds the queue (and hence the
    /// retained memory) when a row lags very far behind the one below it.
    pub fn ahead() -> i32 {
        static A: LazyLock<i32> = LazyLock::new(|| {
            std::env::var("NASSAU_SPECULATE_AHEAD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64)
        });
        *A
    }

    /// Cap on the bytes held by cached matrices. Speculation is skipped while over it. Shared with
    /// [`super::blocks`], which is an alternative granularity for the same cache, never a second one.
    pub fn max_bytes() -> usize {
        static B: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("NASSAU_SPECULATE_MAX_GB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64)
                << 30
        });
        *B
    }

    /// How long a consumer will wait for an in-flight speculative build before giving up and
    /// building the matrix itself (`NASSAU_SPECULATE_WAIT_MS`, default 5000).
    ///
    /// The wait MUST be bounded. Builders run `restricted_partial_matrix`, which is rayon-parallel,
    /// from outside the pool: rayon injects the work and blocks the builder until a pool worker
    /// picks it up. A consumer is a pool worker, and a worker parked in a condvar cannot steal — so
    /// an unbounded wait closes a cycle (consumer waits on builder, builder waits on the workers one
    /// of which is the consumer) and the run hangs. It did, immediately, at stem 40. A timeout turns
    /// that cycle into at worst one duplicated build.
    fn wait_limit() -> std::time::Duration {
        static W: LazyLock<std::time::Duration> = LazyLock::new(|| {
            std::time::Duration::from_millis(
                std::env::var("NASSAU_SPECULATE_WAIT_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(5000),
            )
        });
        *W
    }

    /// Dedicated pool for speculative CPU builds.
    ///
    /// `restricted_partial_matrix` is itself rayon-parallel, so a speculative build issued from a
    /// builder thread injects a large parallel job into the GLOBAL pool -- the one serving the
    /// wavefront. The cores it uses are not free in that case: wavefront tasks queue behind
    /// speculative chunks, and work-stealing spreads the damage. Measured at stem 150 / theta=125,
    /// pin 20: 16 CPU builders took 649 s against a 310 s baseline, a 2.1x regression, while the
    /// cache itself worked fine (53% hit rate, zero duplication).
    ///
    /// Installing the build into a separate pool keeps every nested `par_iter` inside that pool, so
    /// speculation can only consume cores the wavefront is not asking for.
    ///
    /// SIZE IT GENEROUSLY. This defaulted to a quarter of the machine, which throttled speculation
    /// by construction: measured CPU was ~20 of 128 cores, and block coverage plateaued near 52%
    /// however the pieces were reshaped. Stem 150, θ=125, control repeated last:
    ///
    /// | builders / pool | wall        | row coverage |
    /// |-----------------|-------------|--------------|
    /// | 16 / 32         | 170 s 168 s | 52.2% 50.6%  |
    /// | 24 / 64         | 134 s       | 63.6%        |
    /// | 32 / 96         | 119 s       | 70.8%        |
    /// | 48 / 120        | 134 s       | 77.0%        |
    ///
    /// So the contention this pool exists to prevent is real, but only past ~3/4 of the machine: at
    /// 120 threads coverage still climbs while wall time regresses. Default is 3/4.
    /// `NASSAU_SPECULATE_POOL` overrides.
    pub fn pool() -> &'static maybe_rayon::MaybeThreadPool {
        static P: LazyLock<maybe_rayon::MaybeThreadPool> = LazyLock::new(|| {
            let n = std::env::var("NASSAU_SPECULATE_POOL")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&v| v > 0)
                .unwrap_or_else(|| (maybe_rayon::max_num_threads() * 3 / 4).max(1));
            maybe_rayon::MaybeThreadPool::new(n, "nassau-spec")
        });
        &P
    }

    /// Whether speculative builds take the CPU path (`NASSAU_SPECULATE_CPU`). Off by default.
    pub fn on_cpu() -> bool {
        static C: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("NASSAU_SPECULATE_CPU").is_some());
        *C
    }

    pub fn verify() -> bool {
        static V: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("NASSAU_SPECULATE_VERIFY").is_some());
        *V
    }

    /// What the cache knows about a bidegree's matrix.
    ///
    /// The state matters as much as the matrix. A first cut cached only finished matrices, so a
    /// builder and the bidegree itself could build the same matrix concurrently — and since a full
    /// build is most of a bidegree's cost, that duplication was the whole cost of speculation (at
    /// stem 150/θ=125: 442 built, 99 consumed, +18% wall). Every bidegree therefore passes through
    /// this map exactly once, and the entry is what stops the second builder from starting.
    enum Slot {
        /// A builder is building this right now. A consumer that arrives waits rather than
        /// duplicating: waiting costs at most what building costs, and usually much less, because
        /// the builder has a head start by construction.
        InFlight,
        Ready(Matrix),
        /// Consumed, or claimed by the consumer on a miss. Builders skip it.
        Taken,
    }

    /// Slots keyed by bidegree, with a condvar for consumers waiting on [`Slot::InFlight`]. A
    /// process resolves one `Resolution`, so the bidegree alone identifies the matrix; the map is
    /// only ever populated when `NASSAU_SPECULATE` is set.
    static CACHE: LazyLock<(Mutex<HashMap<(i32, i32), Slot>>, Condvar)> =
        LazyLock::new(|| (Mutex::new(HashMap::new()), Condvar::new()));
    static BYTES: AtomicUsize = AtomicUsize::new(0);
    static HITS: AtomicUsize = AtomicUsize::new(0);
    static MISSES: AtomicUsize = AtomicUsize::new(0);
    static BUILT: AtomicUsize = AtomicUsize::new(0);
    static WAITED: AtomicUsize = AtomicUsize::new(0);
    static TIMEOUTS: AtomicUsize = AtomicUsize::new(0);
    static DROPPED: AtomicUsize = AtomicUsize::new(0);

    fn size_of(m: &Matrix) -> usize {
        m.rows() * m.columns().div_ceil(64) * 8
    }

    /// Whether there is room to build another matrix. Checked before building, not after: a
    /// speculative build we cannot afford to keep is wasted work, not just wasted space.
    pub fn has_room() -> bool {
        BYTES.load(Ordering::Relaxed) < max_bytes()
    }

    /// Take ownership of building `b`, or report that someone else already owns it.
    ///
    /// A `false` means another builder is on it or the consumer has claimed it — either way there is
    /// nothing to gain by proceeding, only a duplicate build. A `true` obliges the caller to call
    /// [`publish`]: a consumer may now be waiting on this slot.
    pub fn claim(b: Bidegree) -> bool {
        let (m, _) = &*CACHE;
        let mut g = m.lock().unwrap();
        if g.contains_key(&(b.s(), b.t())) {
            return false;
        }
        g.insert((b.s(), b.t()), Slot::InFlight);
        true
    }

    /// Releases a claim that never produced a matrix, so a consumer waiting on the slot falls
    /// through and builds it itself instead of hanging forever. The only way that happens is a panic
    /// in the builder, which would otherwise turn a crash into a deadlock.
    pub struct ClaimGuard(Option<Bidegree>);

    impl ClaimGuard {
        pub fn new(b: Bidegree) -> Self {
            Self(Some(b))
        }

        /// The matrix was published; nothing to release.
        pub fn done(mut self) {
            self.0 = None;
        }
    }

    impl Drop for ClaimGuard {
        fn drop(&mut self) {
            if let Some(b) = self.0 {
                let (m, cv) = &*CACHE;
                m.lock().unwrap().remove(&(b.s(), b.t()));
                cv.notify_all();
            }
        }
    }

    /// Hand over a finished speculative matrix, waking anyone waiting on it.
    pub fn publish(b: Bidegree, matrix: Matrix) {
        let (m, cv) = &*CACHE;
        let mut g = m.lock().unwrap();
        // If the consumer timed out and built its own, the slot is already `Taken`. Drop this copy
        // rather than parking a matrix nobody will ever collect.
        if !matches!(g.get(&(b.s(), b.t())), Some(Slot::InFlight)) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        BYTES.fetch_add(size_of(&matrix), Ordering::Relaxed);
        BUILT.fetch_add(1, Ordering::Relaxed);
        g.insert((b.s(), b.t()), Slot::Ready(matrix));
        cv.notify_all();
    }

    /// The consumer side: the matrix for `b` if it was built ahead, else `None`.
    ///
    /// Never races a builder. If a build is in flight this blocks until it lands; if nothing is
    /// known, it claims the slot so no builder starts the work the caller is about to do itself.
    pub fn take_or_claim(b: Bidegree) -> Option<Matrix> {
        if threads() == 0 {
            return None;
        }
        let (m, cv) = &*CACHE;
        let mut g = m.lock().unwrap();
        let mut waited = false;
        loop {
            match g.get(&(b.s(), b.t())) {
                Some(Slot::Ready(_)) => {
                    let Some(Slot::Ready(matrix)) = g.insert((b.s(), b.t()), Slot::Taken) else {
                        unreachable!("just matched Ready under the same lock")
                    };
                    BYTES.fetch_sub(size_of(&matrix), Ordering::Relaxed);
                    HITS.fetch_add(1, Ordering::Relaxed);
                    return Some(matrix);
                }
                Some(Slot::InFlight) => {
                    if !waited {
                        waited = true;
                        WAITED.fetch_add(1, Ordering::Relaxed);
                    }
                    let (guard, res) = cv.wait_timeout(g, wait_limit()).unwrap();
                    g = guard;
                    if res.timed_out() {
                        // Stop waiting on the builder and take the slot: it will find `Taken` and
                        // drop its result. One duplicated build beats a stalled worker.
                        g.insert((b.s(), b.t()), Slot::Taken);
                        TIMEOUTS.fetch_add(1, Ordering::Relaxed);
                        MISSES.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }
                }
                Some(Slot::Taken) => {
                    MISSES.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                None => {
                    g.insert((b.s(), b.t()), Slot::Taken);
                    MISSES.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
        }
    }

    /// `(built, hits, misses, waits, timeouts, dropped, retained bytes)`, for reporting.
    /// `built - hits` plus `dropped` is the wasted work this design exists to keep near zero;
    /// `timeouts` says how often a consumer gave up on a builder.
    pub fn stats() -> (usize, usize, usize, usize, usize, usize, usize) {
        (
            BUILT.load(Ordering::Relaxed),
            HITS.load(Ordering::Relaxed),
            MISSES.load(Ordering::Relaxed),
            WAITED.load(Ordering::Relaxed),
            TIMEOUTS.load(Ordering::Relaxed),
            DROPPED.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        )
    }

    /// Drop everything. Called when the wavefront finishes so a second `compute_through_stem` in the
    /// same process cannot see stale entries.
    pub fn clear() {
        let (m, cv) = &*CACHE;
        m.lock().unwrap().clear();
        BYTES.store(0, Ordering::Relaxed);
        cv.notify_all();
    }

    struct Queue {
        heap: BinaryHeap<Reverse<(i32, i32)>>,
        closed: bool,
    }

    /// Pending speculative builds, ordered by `(t, s)` ascending.
    ///
    /// The order matters more than it looks. The queue can run far longer than the builders can
    /// drain it, and a matrix that lands after its bidegree has already started is pure waste — so
    /// builders must always take the bidegree the wavefront will reach soonest, which is the
    /// smallest `t`. A FIFO would instead spend the builders on the deepest speculation first.
    static QUEUE: LazyLock<(Mutex<Queue>, Condvar)> = LazyLock::new(|| {
        (
            Mutex::new(Queue {
                heap: BinaryHeap::new(),
                closed: false,
            }),
            Condvar::new(),
        )
    });

    pub fn open() {
        let (m, _) = &*QUEUE;
        let mut q = m.lock().unwrap();
        q.heap.clear();
        q.closed = false;
    }

    pub fn push(b: Bidegree) {
        let (m, cv) = &*QUEUE;
        m.lock().unwrap().heap.push(Reverse((b.t(), b.s())));
        cv.notify_one();
    }

    /// Block for the next bidegree to build, or `None` once the queue is closed and drained.
    pub fn pop() -> Option<Bidegree> {
        let (m, cv) = &*QUEUE;
        let mut q = m.lock().unwrap();
        loop {
            if let Some(Reverse((t, s))) = q.heap.pop() {
                return Some(Bidegree::s_t(s, t));
            }
            if q.closed {
                return None;
            }
            q = cv.wait(q).unwrap();
        }
    }

    /// Stop the builders once the wavefront is done. Anything still queued is dropped: its bidegree
    /// has either run already or is out of region.
    pub fn closed() -> bool {
        let (m, _) = &*QUEUE;
        m.lock().unwrap().closed
    }

    pub fn close() {
        let (m, cv) = &*QUEUE;
        let mut q = m.lock().unwrap();
        q.closed = true;
        q.heap.clear();
        cv.notify_all();
    }
}

/// Block-granular speculation: precompute *row blocks* of a bidegree's full restricted matrix.
///
/// # What this is actually competing against (measured Aug 2026)
///
/// At stem 110 / theta=0, with the GPU work already fixed (`NASSAU_GPU_CLEANUP_EVERY=0`, enum split
/// on), the run uses **2705% of a possible 12800% — 21% of 128 cores**. It is neither CPU-bound nor
/// GPU-bound: the GPU workers are 31% busy and the devices less. It is STARVED, because the
/// dependency structure only exposes `[wavefront] in-flight bidegrees: mean=6.6 max=19` at a time,
/// each using ~4 cores.
///
/// Turning this on spends some of the idle 79%. Interleaved, 2 rounds, output identical:
///
/// ```text
/// off              26.0, 25.7 s   cpu 2705%, 2723%
/// blocks, 32 bld   22.9, 22.6 s   cpu 3157%, 3200%
/// blocks, 96 bld   23.0, 22.1 s   cpu 3140%, 3291%
/// ```
///
/// 1.14x, and CPU only 21% -> 25%. Tripling the builders buys nothing (96 ~ 32), so what bounds it
/// is the amount of work ELIGIBLE to speculate, not builder capacity — consistent with the ~33%
/// row-coverage ceiling and the idle-builder counters below.
///
/// It is also worth LESS than it looks, and the reason generalises. Little's law: in-flight
/// `L = λ·W`. Speculation cut per-bidegree latency `W = L × wall` by 1.40x (366 -> 262) but the
/// wavefront narrowed 1.23x (14.15 -> 11.5) in near-exact compensation, and 1.40/1.23 = 1.14x, the
/// measured wall win. Arrivals are gated by the DAG, not by free capacity, so the system is
/// RELEASE-limited: making bidegrees faster does not create more of them, it just drains the queue.
/// Any change that only shifts work off the critical path pays this tax. Uniform work REMOVAL does
/// not — see the dead-signature-tail skip in [`Resolution::step_resolution_with_subalgebra`], which
/// took 1.33x with the wavefront holding at 13-14, and which SUBSUMES this module: stacked, the two
/// measure 19.9/19.2 s against 19.0-20.3 s for the skip alone.
///
/// The obvious next lever — publish generators early so `progress` advances before a bidegree
/// finishes — is a DEAD END, and specifically not for the reason an earlier version of this comment
/// gave. `add_generators` already runs before the signature loop, so the generator COUNT is already
/// published early. What a successor blocks on is the differential VALUES, and the signature loop
/// mutates `xs` in place until it converges. There is no artificial barrier here to remove.
///
/// [`speculate`] caches one whole matrix per bidegree, which bounds how early the work can start.
/// The matrix at `(s, t)` is only fully determined once `(s - 1, t - 1)` is committed, so a builder
/// gets at most the gap between "matrix determined" and "bidegree runs" — one degree of slack per
/// row, and the measured hit rate is ~30%.
///
/// A *block* is the set of rows coming from generators of `modules[s - 1]` in a single degree
/// `gen_deg` — the rows `Sq(R) · x` with `deg x = gen_deg` and `deg Sq(R) = t - gen_deg`, which is
/// exactly the `(bidegree, input_deg, output_deg)` decomposition. A block is determined *far*
/// earlier than the matrix it belongs to:
///
/// * **Its rows.** Basis elements of a [`FreeModule`] at degree `t` are laid out generator-major and
///   the table is append-only (`add_generators` appends to every already-computed degree, so
///   existing indices never move). The row range of block `gen_deg` is therefore
///   `sum over h < gen_deg of num_gens[h] * dim(t - h)`, frozen once the generator counts below
///   `gen_deg` are — i.e. once `(s - 1, gen_deg)` is committed. Not `t - 1`. This is the same
///   row-block claim `NASSAU_SPLIT_VERIFY` checks against real data.
/// * **Its columns.** `d(x)` for a generator `x` of degree `gen_deg` lands in the radical (that is
///   minimality, and it is already what licenses the existing truncation in
///   [`FreeModuleHomomorphism::apply_to_basis_element_restricted`]), so it is supported on
///   generators of `modules[s - 2]` of degree *strictly* below `gen_deg`. Acting by `Sq(R)` does not
///   change the generator, so the whole block is supported in the first
///   `restricted_dimension(modules[s - 2], t, gen_deg)` columns. The block is built that narrow and
///   zero-extended at assembly, so it needs only `(s - 2, gen_deg - 1)`, not `(s - 2, t - 2)`.
///
/// So block `gen_deg` of `(s, t)` is buildable as soon as `min(progress[s-1], progress[s-2])`
/// reaches `gen_deg`, for *every* `t > gen_deg` still in region. That is what deepens the queue: a
/// single commit opens blocks across a whole column of future bidegrees instead of one matrix.
///
/// # Consuming
///
/// The consumer takes whatever blocks are ready, copies them into the full matrix, and collects the
/// rows it did not get into ONE coalesced build. A miss therefore costs nothing beyond the rows it
/// covers — unlike [`speculate`], where a miss means rebuilding everything — so there is no waiting
/// on in-flight work here, and hence none of that module's deadlock hazard: a block that has not
/// landed is simply folded into the launch the consumer was making anyway.
///
/// # Configuration
///
/// * `NASSAU_SPECULATE_BLOCKS` — use block granularity instead of whole matrices.
/// * `NASSAU_SPECULATE_BLOCK_AHEAD` — degrees past a row's frontier to speculate (default 24).
/// * `NASSAU_SPECULATE_BLOCK_QUEUE` — cap on queued blocks (default 1M), a flood guard.
mod blocks {
    use std::{
        cmp::Reverse,
        collections::{BinaryHeap, HashMap},
        sync::{
            Condvar, LazyLock, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use fp::matrix::Matrix;
    use sseq::coordinates::Bidegree;

    pub fn enabled() -> bool {
        static E: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("NASSAU_SPECULATE_BLOCKS").is_some());
        *E
    }

    /// How far past a row's own committed frontier to speculate blocks. Smaller than
    /// [`super::speculate::ahead`] by default because each degree contributes many items, not one.
    pub fn ahead() -> i32 {
        static A: LazyLock<i32> = LazyLock::new(|| {
            std::env::var("NASSAU_SPECULATE_BLOCK_AHEAD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(24)
        });
        *A
    }

    fn queue_cap() -> usize {
        static Q: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("NASSAU_SPECULATE_BLOCK_QUEUE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1 << 20)
        });
        *Q
    }

    /// One generator degree's claim, holding whatever pieces have been finished for it so far.
    ///
    /// A degree is claimed as a unit (one queue item) but DELIVERED per generator, incrementally.
    /// `iter_gen_offsets` yields one entry per generator, so a degree with several generators is
    /// several independent row ranges; building them as one range made blocks coarse and
    /// all-or-nothing, and coverage fell (86.3% -> 68.1% at stem 30) because fewer finished before
    /// their bidegree ran. Publishing each generator as it lands keeps the granularity fine while
    /// still covering every generator, instead of only the first one in each degree.
    #[derive(Default)]
    struct DegreeSlot {
        pieces: Vec<(usize, Matrix)>,
    }

    /// A finished piece carries the row offset it was BUILT against, not just its contents.
    ///
    /// That offset is the fix for a real bug. The consumer used to re-derive each block's row range
    /// from the module's generator counts at consumption time and got a different answer than the
    /// builder had (`start=23` for a block whose rows actually sat at 22), silently misplacing every
    /// row. A free module's basis at a fixed degree is append-only and generator-major, so the
    /// offset a block was built against stays valid forever — recomputing it later is what is
    /// fragile. So the block owns its position, and the consumer never re-derives it.

    /// Every block known for one bidegree.
    ///
    /// `done` is set the moment the consumer collects, and is what stops a builder from starting (or
    /// publishing) work whose bidegree has already assembled its matrix. The entry outlives the
    /// collection precisely so that a straggling `publish` finds `done` and drops its result instead
    /// of resurrecting the bidegree.
    #[derive(Default)]
    struct BidegreeBlocks {
        done: bool,
        slots: HashMap<i32, DegreeSlot>,
        /// Signatures already consumed. A bidegree above `reuse_within_cap` never assembles a full
        /// matrix, so `done` never fires for it; delivery is per signature instead and this is what
        /// stops a straggling piece from being stored for a signature already row-reduced.
        sig_done: std::collections::HashSet<usize>,
        /// `sig_idx -> pieces`, each `(offset within the signature matrix, rows)`.
        sig: HashMap<usize, Vec<(usize, Matrix)>>,
    }

    static CACHE: LazyLock<Mutex<HashMap<(i32, i32), BidegreeBlocks>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    static BYTES: AtomicUsize = AtomicUsize::new(0);
    static BUILT: AtomicUsize = AtomicUsize::new(0);
    static HITS: AtomicUsize = AtomicUsize::new(0);
    static MISSES: AtomicUsize = AtomicUsize::new(0);
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    static ROWS_HIT: AtomicUsize = AtomicUsize::new(0);
    static ROWS_MISSED: AtomicUsize = AtomicUsize::new(0);
    static FLOODED: AtomicUsize = AtomicUsize::new(0);
    static RELEASED: AtomicUsize = AtomicUsize::new(0);
    /// Why builders declined work, so a coverage plateau can be attributed instead of guessed at.
    /// Stem 200 sits at 33% row coverage while total CPU is ~13 of 128 cores -- the builders are
    /// IDLE, not contended -- so the question is which bail-out they are taking.
    static BAIL_ROOM: AtomicUsize = AtomicUsize::new(0);
    static BAIL_DONE: AtomicUsize = AtomicUsize::new(0);
    static BAIL_CLAIMED: AtomicUsize = AtomicUsize::new(0);
    static BAIL_OTHER: AtomicUsize = AtomicUsize::new(0);
    /// Queue depth at each successful pop, and the time builders spend parked because the queue is
    /// EMPTY. Only 36% of popped items got built before their bidegree ran, while builders were not
    /// saturated (1310% CPU of 128 cores) -- so they were neither too slow nor too busy. Either the
    /// queue is mostly empty (the producer's eligibility gate admits far less than the nominal
    /// 24-degree window) or builders are blocked on something that does not show as CPU. Depth near
    /// zero with large idle time means the former.
    static QDEPTH_SUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static QDEPTH_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static QDEPTH_MAX: AtomicUsize = AtomicUsize::new(0);
    static IDLE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static BUILD_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static SKIPPED_SMALL: AtomicUsize = AtomicUsize::new(0);
    static TW_SUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static TW_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static TW_EMPTY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static ACTIVE_BUILDERS: AtomicUsize = AtomicUsize::new(0);
    /// Rows served vs rebuilt, bucketed by OPERATION degree (`t - gen_deg`) in steps of 25.
    ///
    /// A block's lookahead is ~`t - gen_deg`: it becomes eligible when `progress[s-1]` reaches
    /// `gen_deg`, and its bidegree runs when `progress[s-1]` reaches `t-1`. So blocks with `gen_deg`
    /// near `t` have almost no warning. The claim to test is that the ~33% coverage ceiling at stem
    /// 200 is this and not a tunable: if so, misses concentrate in the LOW operation-degree buckets
    /// and hits in the high ones, and the row mass shifts low between stem 150 and stem 200.
    static HIT_BY_OPDEG: [std::sync::atomic::AtomicU64; 16] =
        [const { std::sync::atomic::AtomicU64::new(0) }; 16];
    static MISS_BY_OPDEG: [std::sync::atomic::AtomicU64; 16] =
        [const { std::sync::atomic::AtomicU64::new(0) }; 16];

    pub fn opdeg_hist() -> bool {
        static H: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("NASSAU_OPDEG_HIST").is_some());
        *H
    }

    pub fn record_opdeg(op_deg: i32, rows: u64, hit: bool) {
        let b = ((op_deg.max(0) / 25) as usize).min(15);
        if hit {
            HIT_BY_OPDEG[b].fetch_add(rows, Ordering::Relaxed);
        } else {
            MISS_BY_OPDEG[b].fetch_add(rows, Ordering::Relaxed);
        }
    }

    pub fn dump_opdeg() {
        if !opdeg_hist() {
            return;
        }
        for b in 0..16 {
            let (h, m) = (
                HIT_BY_OPDEG[b].load(Ordering::Relaxed),
                MISS_BY_OPDEG[b].load(Ordering::Relaxed),
            );
            if h + m > 0 {
                eprintln!(
                    "[opdeg] op_deg {:>3}-{:<3} rows_hit={h} rows_missed={m} coverage={:.1}%",
                    b * 25,
                    b * 25 + 24,
                    100.0 * h as f64 / (h + m) as f64,
                );
            }
        }
    }
    /// Depth and queued rows maintained on push/pop, so the sampler never takes the queue lock.
    ///
    /// Reading them under the mutex does not work: with ~343k items and 32 builders contending, a
    /// 100 ms sampler simply never wins the lock, and iterating the heap under it would block every
    /// builder. The first version did both and emitted nothing at stem 200 while working fine on a
    /// small queue.
    static Q_DEPTH: AtomicUsize = AtomicUsize::new(0);
    static Q_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    pub fn skipped_small() -> usize {
        SKIPPED_SMALL.load(Ordering::Relaxed)
    }

    pub fn record_build_nanos(n: u64) {
        BUILD_NANOS.fetch_add(n, Ordering::Relaxed);
    }

    /// Time-weighted queue depth, sampled by a timer rather than at pops.
    ///
    /// Sampling at pops is biased by construction: a pop only happens when there is work, so the
    /// statistic can only ever report "deep". It measured mean=133 589 while `builder_idle` showed
    /// builders parked 91% of their thread-time — both true, describing different moments. The
    /// producer emits a whole strip per commit, floods the queue, and builders drain it over
    /// minutes; between bursts the queue is empty and nothing observes it.
    pub fn sample_depth() {
        let d = Q_DEPTH.load(Ordering::Relaxed);
        TW_SUM.fetch_add(d as u64, Ordering::Relaxed);
        TW_N.fetch_add(1, Ordering::Relaxed);
        if d == 0 {
            TW_EMPTY.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// One line of the queue-depth TIME SERIES, with a breakdown of what is queued.
    ///
    /// A mean hides the shape, and the shape is the question: enqueues are triggered by COMMITS,
    /// and commits get sparser as bidegrees get larger, so the queue is expected to be front-loaded
    /// and to starve later — exactly when the expensive bidegrees need it. `tmin`/`tmax` show
    /// whether what is queued is near or far work, `rows` whether it is worth anything, and
    /// `building` how many builders are actually occupied at that instant.
    pub fn trace_depth(elapsed_s: f64) {
        eprintln!(
            "[qtrace] t={elapsed_s:.0}s depth={} rows={} building={} built={}",
            Q_DEPTH.load(Ordering::Relaxed),
            Q_ROWS.load(Ordering::Relaxed),
            ACTIVE_BUILDERS.load(Ordering::Relaxed),
            BUILT.load(Ordering::Relaxed),
        );
    }

    pub fn builder_enter() {
        ACTIVE_BUILDERS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn builder_exit() {
        ACTIVE_BUILDERS.fetch_sub(1, Ordering::Relaxed);
    }

    /// `(time-weighted mean depth, fraction of samples with an EMPTY queue)`.
    pub fn timed_depth() -> (f64, f64) {
        let n = TW_N.load(Ordering::Relaxed).max(1);
        (
            TW_SUM.load(Ordering::Relaxed) as f64 / n as f64,
            TW_EMPTY.load(Ordering::Relaxed) as f64 / n as f64,
        )
    }

    /// `(mean depth, max depth, builder idle seconds, builder build seconds)`.
    pub fn queue_stats() -> (f64, usize, f64, f64) {
        let n = QDEPTH_N.load(Ordering::Relaxed).max(1);
        (
            QDEPTH_SUM.load(Ordering::Relaxed) as f64 / n as f64,
            QDEPTH_MAX.load(Ordering::Relaxed),
            IDLE_NANOS.load(Ordering::Relaxed) as f64 / 1e9,
            BUILD_NANOS.load(Ordering::Relaxed) as f64 / 1e9,
        )
    }

    pub fn bail(kind: u8) {
        match kind {
            0 => &BAIL_ROOM,
            1 => &BAIL_DONE,
            2 => &BAIL_CLAIMED,
            _ => &BAIL_OTHER,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    pub fn bails() -> (usize, usize, usize, usize) {
        (
            BAIL_ROOM.load(Ordering::Relaxed),
            BAIL_DONE.load(Ordering::Relaxed),
            BAIL_CLAIMED.load(Ordering::Relaxed),
            BAIL_OTHER.load(Ordering::Relaxed),
        )
    }

    fn size_of(m: &Matrix) -> usize {
        m.rows() * m.columns().div_ceil(64) * 8
    }

    pub fn has_room() -> bool {
        BYTES.load(Ordering::Relaxed) < super::speculate::max_bytes()
    }

    /// Take ownership of building one block, or report that it is already spoken for.
    pub fn claim(b: Bidegree, gen_deg: i32) -> bool {
        let mut g = CACHE.lock().unwrap();
        let e = g.entry((b.s(), b.t())).or_default();
        if e.done || e.slots.contains_key(&gen_deg) {
            return false;
        }
        e.slots.insert(gen_deg, DegreeSlot::default());
        true
    }

    /// Releases a claim that produced nothing, so a later builder may retry it. Only reachable if
    /// the builder panicked.
    pub struct ClaimGuard(Option<(Bidegree, i32)>);

    impl ClaimGuard {
        pub fn new(b: Bidegree, gen_deg: i32) -> Self {
            Self(Some((b, gen_deg)))
        }

        pub fn done(mut self) {
            self.0 = None;
        }
    }

    impl Drop for ClaimGuard {
        fn drop(&mut self) {
            if let Some((b, gen_deg)) = self.0
                && let Some(e) = CACHE.lock().unwrap().get_mut(&(b.s(), b.t()))
                && e.slots.get(&gen_deg).is_some_and(|d| d.pieces.is_empty())
            {
                e.slots.remove(&gen_deg);
            }
        }
    }

    /// Hand over one finished piece. Called once per generator, as each lands.
    pub fn publish(b: Bidegree, gen_deg: i32, start: usize, matrix: Matrix) {
        let mut g = CACHE.lock().unwrap();
        let Some(e) = g.get_mut(&(b.s(), b.t())) else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let done = e.done;
        let Some(slot) = e.slots.get_mut(&gen_deg).filter(|_| !done) else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        BYTES.fetch_add(size_of(&matrix), Ordering::Relaxed);
        BUILT.fetch_add(1, Ordering::Relaxed);
        slot.pieces.push((start, matrix));
    }

    /// Take ownership of building one signature's pieces for a generator degree.
    pub fn claim_sig(b: Bidegree, gen_deg: i32, sig_idx: usize) -> bool {
        let mut g = CACHE.lock().unwrap();
        let e = g.entry((b.s(), b.t())).or_default();
        if e.done || e.sig_done.contains(&sig_idx) {
            return false;
        }
        // One claim per (signature, generator degree): the degree's queue item already serialises
        // builders, this only guards against a re-queued degree redoing finished work.
        e.sig.entry(sig_idx).or_default();
        let _ = gen_deg;
        true
    }

    pub fn publish_sig(b: Bidegree, sig_idx: usize, start: usize, matrix: Matrix) {
        let mut g = CACHE.lock().unwrap();
        let Some(e) = g.get_mut(&(b.s(), b.t())) else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if e.done || e.sig_done.contains(&sig_idx) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        BYTES.fetch_add(size_of(&matrix), Ordering::Relaxed);
        BUILT.fetch_add(1, Ordering::Relaxed);
        e.sig.entry(sig_idx).or_default().push((start, matrix));
    }

    /// Collect one signature's pieces and close that signature. Sorted by offset.
    pub fn take_sig(b: Bidegree, sig_idx: usize) -> Vec<(usize, Matrix)> {
        let mut g = CACHE.lock().unwrap();
        let e = g.entry((b.s(), b.t())).or_default();
        e.sig_done.insert(sig_idx);
        let mut out = e.sig.remove(&sig_idx).unwrap_or_default();
        for (_, m) in &out {
            BYTES.fetch_sub(size_of(m), Ordering::Relaxed);
        }
        out.sort_by_key(|(start, _)| *start);
        out
    }

    /// Whether the bidegree has already assembled, so a builder should stop mid-degree.
    pub fn is_done(b: Bidegree) -> bool {
        CACHE
            .lock()
            .unwrap()
            .get(&(b.s(), b.t()))
            .is_some_and(|e| e.done)
    }

    /// Whether any signature speculation exists for `b` — lets the consumer skip the assembly path
    /// entirely (and its mask bookkeeping) when nothing was precomputed.
    pub fn has_sig(b: Bidegree) -> bool {
        CACHE
            .lock()
            .unwrap()
            .get(&(b.s(), b.t()))
            .is_some_and(|e| !e.sig.is_empty())
    }

    /// Collect every ready block for `b` and close the bidegree to further speculation.
    /// Collect every ready block for `b` as `(start, matrix)` sorted by row offset, and close the
    /// bidegree to further speculation.
    pub fn take_all(b: Bidegree) -> Vec<(usize, Matrix)> {
        let mut g = CACHE.lock().unwrap();
        let e = g.entry((b.s(), b.t())).or_default();
        e.done = true;
        let mut out = Vec::new();
        for (_, slot) in std::mem::take(&mut e.slots) {
            for (start, m) in slot.pieces {
                BYTES.fetch_sub(size_of(&m), Ordering::Relaxed);
                out.push((start, m));
            }
        }
        out.sort_by_key(|(start, _)| *start);
        out
    }

    /// Drop every block for a bidegree that will never assemble one, and close it to further
    /// speculation. Counted separately so the waste is visible rather than inferred.
    /// Drop the contiguous block pieces for a bidegree that is building per signature.
    ///
    /// Deliberately does NOT set `done`: above the cap the bidegree still takes delivery along the
    /// signature axis, and `done` would silence exactly that.
    pub fn release(b: Bidegree) {
        let mut g = CACHE.lock().unwrap();
        let Some(e) = g.get_mut(&(b.s(), b.t())) else {
            return;
        };
        for (_, slot) in std::mem::take(&mut e.slots) {
            RELEASED.fetch_add(slot.pieces.len(), Ordering::Relaxed);
            for (_, m) in &slot.pieces {
                BYTES.fetch_sub(size_of(m), Ordering::Relaxed);
            }
        }
    }

    /// Drop every signature piece for a bidegree that has finished its signature loop.
    pub fn release_sig(b: Bidegree) {
        let mut g = CACHE.lock().unwrap();
        if let Some(e) = g.get_mut(&(b.s(), b.t())) {
            for (_, pieces) in std::mem::take(&mut e.sig) {
                RELEASED.fetch_add(pieces.len(), Ordering::Relaxed);
                for (_, m) in &pieces {
                    BYTES.fetch_sub(size_of(m), Ordering::Relaxed);
                }
            }
        }
    }

    /// Book-keeping from one assembly: blocks reused, blocks absent, and the row counts behind them
    /// (the rows are what actually matter — a hit on a one-row block saves nothing).
    pub fn record(hits: usize, misses: usize, rows_hit: usize, rows_missed: usize) {
        HITS.fetch_add(hits, Ordering::Relaxed);
        MISSES.fetch_add(misses, Ordering::Relaxed);
        ROWS_HIT.fetch_add(rows_hit, Ordering::Relaxed);
        ROWS_MISSED.fetch_add(rows_missed, Ordering::Relaxed);
    }

    pub fn stats() -> (usize, usize, usize, usize, usize, usize, usize, usize) {
        (
            BUILT.load(Ordering::Relaxed),
            HITS.load(Ordering::Relaxed),
            MISSES.load(Ordering::Relaxed),
            ROWS_HIT.load(Ordering::Relaxed),
            ROWS_MISSED.load(Ordering::Relaxed),
            DROPPED.load(Ordering::Relaxed),
            RELEASED.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        )
    }

    pub fn clear() {
        CACHE.lock().unwrap().clear();
        BYTES.store(0, Ordering::Relaxed);
    }

    struct Queue {
        /// `(Reverse(t), rows, s, gen_deg)` in a MAX-heap: smallest `t` first, then most rows.
        heap: BinaryHeap<(Reverse<i32>, usize, i32, i32)>,
        closed: bool,
    }

    /// Pending blocks, ordered by internal degree ascending, then ROW COUNT descending.
    ///
    /// Blocks have DEADLINES: one is worthless the moment its bidegree runs, and equally useful any
    /// time before. Since the queue never fully drains (36% of items built at stem 200), near blocks
    /// expire while far ones stay valid, so imminence-first is earliest-deadline-first — the optimal
    /// policy for meeting deadlines. Ordering by size INSTEAD threw that away and let builders spend
    /// time on far-future blocks while imminent ones expired: row coverage fell from ~84% to 46.8%
    /// at stem 30.
    ///
    /// Size belongs one level down. Within a deadline the choice is free, and there the stem-200
    /// numbers apply: 36% of items built produced 33% of rows, i.e. selection was size-neutral, so
    /// taking the biggest first converts the same builder time into more coverage.
    static QUEUE: LazyLock<(Mutex<Queue>, Condvar)> = LazyLock::new(|| {
        (
            Mutex::new(Queue {
                heap: BinaryHeap::new(),
                closed: false,
            }),
            Condvar::new(),
        )
    });

    pub fn open() {
        let (m, _) = &*QUEUE;
        let mut q = m.lock().unwrap();
        q.heap.clear();
        q.closed = false;
    }

    /// Smallest block worth queueing, in rows (`NASSAU_BLOCK_MIN_ROWS`).
    ///
    /// A block that is NOT precomputed costs the consumer nothing extra: its rows simply join the
    /// single coalesced build it was already making. So a small block has almost no value
    /// precomputed while still costing a queue slot, a claim, a lock, an allocation and a publish.
    /// At stem 200 the queue carried 718 171 items; the tail is most of the bookkeeping and almost
    /// none of the benefit.
    pub fn min_rows() -> usize {
        static M: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("NASSAU_BLOCK_MIN_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1024)
        });
        *M
    }

    pub fn push(b: Bidegree, gen_deg: i32, rows: usize) {
        if rows < min_rows() {
            SKIPPED_SMALL.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let (m, cv) = &*QUEUE;
        let mut q = m.lock().unwrap();
        // The block window is two-dimensional, so a lagging row can enqueue a quadratic number of
        // items. Dropping the excess is safe: an unbuilt block is a miss, and a miss costs only the
        // rows it covers.
        if q.heap.len() >= queue_cap() {
            FLOODED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        q.heap.push((Reverse(b.t()), rows, b.s(), gen_deg));
        Q_DEPTH.fetch_add(1, Ordering::Relaxed);
        Q_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
        cv.notify_one();
    }

    pub fn pop() -> Option<(Bidegree, i32)> {
        let (m, cv) = &*QUEUE;
        let mut q = m.lock().unwrap();
        let mut idle = std::time::Duration::ZERO;
        loop {
            let depth = q.heap.len();
            if let Some((Reverse(t), popped_rows, s, gen_deg)) = q.heap.pop() {
                Q_DEPTH.fetch_sub(1, Ordering::Relaxed);
                Q_ROWS.fetch_sub(popped_rows as u64, Ordering::Relaxed);
                QDEPTH_SUM.fetch_add(depth as u64, Ordering::Relaxed);
                QDEPTH_N.fetch_add(1, Ordering::Relaxed);
                QDEPTH_MAX.fetch_max(depth, Ordering::Relaxed);
                IDLE_NANOS.fetch_add(idle.as_nanos() as u64, Ordering::Relaxed);
                return Some((Bidegree::s_t(s, t), gen_deg));
            }
            if q.closed {
                IDLE_NANOS.fetch_add(idle.as_nanos() as u64, Ordering::Relaxed);
                return None;
            }
            let t0 = std::time::Instant::now();
            q = cv.wait(q).unwrap();
            idle += t0.elapsed();
        }
    }

    pub fn closed() -> bool {
        let (m, _) = &*QUEUE;
        m.lock().unwrap().closed
    }

    pub fn close() {
        let (m, cv) = &*QUEUE;
        let mut q = m.lock().unwrap();
        q.closed = true;
        Q_DEPTH.store(0, Ordering::Relaxed);
        Q_ROWS.store(0, Ordering::Relaxed);
        q.heap.clear();
        cv.notify_all();
    }
}

/// Multiply work (`rows x cols`) split by whether the bidegree could use the full-matrix reuse
/// path at all. Speculation can only ever touch the REUSE side, so this bounds the whole direction:
/// if most of the work is on the no-reuse side, no amount of block tuning can reach it.
/// Bidegrees actually executing right now, and its running statistics.
///
/// Speculation shortens each bidegree; it does not widen the WAVEFRONT, because admission is
/// unchanged -- `(s,t)` is eligible only once `(s,t-1)` and `(s-1,t-1)` are committed, so the
/// frontier stays a slope-1 staircase over `s`. If neither CPU nor GPU is saturated (stem 200
/// measured ~13 of 128 cores and 11-47% GPU), the machine is waiting, and the question is whether
/// too few bidegrees are ELIGIBLE or whether eligible ones are not being run. This counts the
/// latter; a low mean with idle hardware means the dependency graph is the constraint and no amount
/// of speculation or granularity tuning can matter.
static INFLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static INFLIGHT_MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static INFLIGHT_SUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static INFLIGHT_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static REUSE_WORK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REUSE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static NOREUSE_WORK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The same work measured in the NARROW block representation: sum over generator degrees of
/// `rows(g) x cols(g)`, where `cols(g)` stops below generator degree `g` by minimality.
///
/// The full matrix pads every row to `next_dim`; the block set is triangular. The ratio says whether
/// serving a signature's rows directly from blocks -- never materialising the full matrix -- would
/// fit under a memory ceiling the full matrix does not. Near 0.5 that is worth building; near 1.0
/// the triangle is too shallow and relaxing `reuse_within_cap` is the only lever left.
static NARROW_WORK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NOREUSE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A resolution of `S_2` using Nassau's algorithm.
///
/// This aims to have an API similar to that of
/// [`resolution::Resolution`](crate::resolution::Resolution). From an API point of view, the main
/// difference between the two is that this is a chain complex over [`MilnorAlgebra`] over
/// [`SteenrodAlgebra`](algebra::SteenrodAlgebra).
pub struct Resolution<M: ZeroModule<Algebra = MilnorAlgebra>> {
    lock: Mutex<()>,
    name: String,
    max_degree: i32,
    modules: OnceBiVec<Arc<FreeModule<MilnorAlgebra>>>,
    zero_module: Arc<FreeModule<MilnorAlgebra>>,
    differentials: OnceBiVec<Arc<FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>>>,
    target: Arc<FiniteChainComplex<M>>,
    chain_maps: OnceBiVec<Arc<FreeModuleHomomorphism<M>>>,
    save_dir: SaveDirectory,
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> Resolution<M> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, name: String) {
        // Record the label in the save store (if any) so the on-disk `zarr.json` is
        // self-describing. Best-effort and purely informational — the module spec written by
        // `bind_module_spec` is what actually guards loading.
        if !name.is_empty()
            && let Some(store) = self.save_dir.store()
            && let Err(e) = store.set_complex_name(&name)
        {
            tracing::warn!("Failed to record complex name in save store: {e}");
        }
        self.name = name;
    }

    pub fn new(module: Arc<M>) -> Self {
        Self::new_with_save(module, None).unwrap()
    }

    pub fn new_with_save(
        module: Arc<M>,
        save_dir: impl TryInto<SaveDirectory, Error: Into<anyhow::Error>>,
    ) -> anyhow::Result<Self> {
        let save_dir = save_dir.try_into().map_err(Into::into)?;
        let max_degree = module
            .max_degree()
            .ok_or_else(|| anyhow!("Nassau's algorithm requires bounded module"))?;
        let target = Arc::new(FiniteChainComplex::ccdz(module));
        if let Some(store) = save_dir.store() {
            let algebra = target.algebra();
            store.bind_to_algebra(algebra.magic(), algebra.prime().as_u32(), algebra.prefix())?;
        }

        Ok(Self {
            lock: Mutex::new(()),
            zero_module: Arc::new(FreeModule::new(target.algebra(), "F_{-1}".to_string(), 0)),
            name: String::new(),
            modules: OnceBiVec::new(0),
            differentials: OnceBiVec::new(0),
            chain_maps: OnceBiVec::new(0),
            target,
            max_degree,
            save_dir,
        })
    }

    fn add_generators(&self, b: Bidegree, num_new_gens: usize) {
        let gen_names = (0..num_new_gens)
            .map(|idx| format!("x_{:#}", BidegreeGenerator::new(b, idx)))
            .collect();
        self.module(b.s())
            .add_generators(b.t(), num_new_gens, Some(gen_names));
    }

    /// This function prepares the Resolution object to perform computations up to the
    /// specified s degree. It does *not* perform any computations by itself. It simply lengthens
    /// the `OnceVec`s `modules`, `chain_maps`, etc. to the right length.
    fn extend_through_degree(&self, max_s: i32) {
        let min_degree = self.min_degree();

        self.modules.extend(max_s, |i| {
            Arc::new(FreeModule::new(
                Arc::clone(&self.algebra()),
                format!("F{i}"),
                min_degree,
            ))
        });

        self.differentials.extend(0, |_| {
            Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[0]),
                Arc::clone(&self.zero_module),
                0,
            ))
        });

        self.differentials.extend(max_s, |i| {
            Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[i]),
                Arc::clone(&self.modules[i - 1]),
                0,
            ))
        });

        self.chain_maps.extend(max_s, |i| {
            Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[i]),
                self.target.module(i),
                0,
            ))
        });
    }

    #[tracing::instrument(skip_all)]
    fn write_qi(
        w: &mut Option<NassauQiWriter>,
        scratch: &mut FpVector,
        signature: &[PPartEntry],
        next_mask: &[usize],
        full_matrix: &PartialMatrix<'_>,
        masked_matrix: &AugmentedMatrix<2>,
    ) -> anyhow::Result<()> {
        let w = match w {
            Some(w) => w,
            None => return Ok(()),
        };

        let pivots = &masked_matrix.pivots()[0..masked_matrix.end[0]];
        if !pivots.iter().any(|&x| x >= 0) {
            return Ok(());
        }

        // Emit a signature command if non-zero.
        if signature.iter().any(|&x| x > 0) {
            let sig_u16: Vec<u16> = signature.iter().map(|&x| x as u16).collect();
            w.write_signature(&sig_u16)?;
        }

        // Emit one pivot command per non-trivial pivot row.
        for (col, &row) in pivots.iter().enumerate() {
            if row < 0 {
                continue;
            }
            let preimage = masked_matrix.row_segment(row as usize, 1, 1);
            scratch.set_scratch_vector_size(preimage.len());
            scratch.as_slice_mut().assign(preimage);
            // The lift slice we want to write is `scratch.as_slice()` here;
            // we have to capture it before reusing `scratch` for the image.
            let lift_vec = scratch.clone();

            scratch.set_scratch_vector_size(full_matrix.columns());
            for (i, _) in preimage.iter_nonzero() {
                scratch.as_slice_mut().add(full_matrix.row(i), 1);
            }
            w.write_pivot(
                next_mask[col] as u64,
                lift_vec.as_slice(),
                scratch.as_slice(),
            )?;
        }

        Ok(())
    }

    /// Build the [`NassauCommand`] stream for one signature block of a quasi-inverse.
    ///
    /// Mirrors [`Self::write_qi`] but returns owned commands instead of writing them to a store;
    /// used by the on-demand recompute path ([`RecomputeReader`]). Returns an empty vec if the
    /// block has no pivots (nothing to emit). Never emits [`NassauCommand::Fix`] — that is only
    /// written when the bidegree was resolved through stem, which never happens on the recompute
    /// path (the resolution is fully computed by then).
    fn qi_commands(
        scratch: &mut FpVector,
        signature: &[PPartEntry],
        next_mask: &[usize],
        full_matrix: &Matrix,
        masked_matrix: &AugmentedMatrix<2>,
    ) -> anyhow::Result<Vec<NassauCommand>> {
        let mut cmds = Vec::new();

        let pivots = &masked_matrix.pivots()[0..masked_matrix.end[0]];
        if !pivots.iter().any(|&x| x >= 0) {
            return Ok(cmds);
        }

        // Emit a signature command if non-zero.
        if signature.iter().any(|&x| x > 0) {
            let sig_u16: Vec<u16> = signature.iter().map(|&x| x as u16).collect();
            cmds.push(NassauCommand::Signature(sig_u16));
        }

        // Emit one pivot command per non-trivial pivot row.
        for (col, &row) in pivots.iter().enumerate() {
            if row < 0 {
                continue;
            }
            let preimage = masked_matrix.row_segment(row as usize, 1, 1);
            scratch.set_scratch_vector_size(preimage.len());
            scratch.as_slice_mut().assign(preimage);
            // The lift slice we want to write is `scratch.as_slice()` here;
            // we have to capture it before reusing `scratch` for the image.
            let lift_vec = scratch.clone();

            scratch.set_scratch_vector_size(full_matrix.columns());
            for (i, _) in preimage.iter_nonzero() {
                scratch.as_slice_mut().add(full_matrix.row(i), 1);
            }

            let mut lift_bytes = Vec::new();
            lift_vec.to_bytes(&mut lift_bytes)?;
            let mut image_bytes = Vec::new();
            scratch.to_bytes(&mut image_bytes)?;
            cmds.push(NassauCommand::Pivot {
                col: next_mask[col] as u64,
                lift_bytes,
                image_bytes,
            });
        }

        Ok(cmds)
    }

    fn write_differential(
        &self,
        b: Bidegree,
        num_new_gens: usize,
        target_dim: usize,
    ) -> anyhow::Result<()> {
        if let Some(store) = self.save_dir.store() {
            let mut buf = Vec::new();
            buf.write_u64::<LittleEndian>(num_new_gens as u64)?;
            buf.write_u64::<LittleEndian>(target_dim as u64)?;

            for n in 0..num_new_gens {
                self.differential(b.s())
                    .output(b.t(), n)
                    .to_bytes(&mut buf)?;
            }
            store.write(SaveKind::NassauDifferential, b, &buf)?;
        }
        Ok(())
    }

    /// Dimensions of the full restricted matrix at `b`: the restricted source dimension (its row
    /// count) and the restricted target dimension (its column count), exactly as
    /// [`Self::step_resolution_with_subalgebra`] computes them.
    ///
    /// Note there is no [`MilnorSubalgebra`] argument. The restriction bounds are `b.t()` and
    /// `b.t() - 1` — properties of the bidegree, not of the signature — which is precisely why this
    /// matrix can be built before the bidegree's subalgebra is chosen, and hence before the bidegree
    /// runs at all. See [`speculate`].
    fn restricted_dims(&self, b: Bidegree) -> (usize, usize) {
        let target = &*self.modules[b.s() - 1];
        target.compute_basis(b.t());
        let target_dim = MilnorSubalgebra::restricted_dimension(target, b.t(), b.t());
        let next = &self.modules[b.s() - 2];
        next.compute_basis(b.t());
        let next_dim = MilnorSubalgebra::restricted_dimension(next, b.t(), b.t() - 1);
        (target_dim, next_dim)
    }

    /// The full restricted differential matrix at `b`, over every restricted source row.
    ///
    /// `on_cpu` forces the CPU path regardless of the GPU settings. Speculative builders use it
    /// (`NASSAU_SPECULATE_CPU`) because the two paths compete for different resources: sampling a
    /// stem-200 run shows GPU 0 bursting to 100% while ~120 of the 128 cores idle, so a speculative
    /// build sent to the GPU merely queues behind the critical path, while one sent to the CPU is
    /// nearly free AND removes a launch from the queue the critical path is waiting on.
    fn build_full_restricted(
        &self,
        b: Bidegree,
        target_dim: usize,
        next_dim: usize,
        on_cpu: bool,
    ) -> Matrix {
        let all_rows: Vec<usize> = (0..target_dim).collect();
        let diff = &self.differentials[b.s() - 1];
        if on_cpu {
            restricted_partial_matrix(diff, b.t(), &all_rows, next_dim)
        } else {
            restricted_partial_matrix_maybe_gpu(diff, b.t(), &all_rows, next_dim)
        }
    }

    /// The row blocks of `b`'s full restricted matrix, as `(gen_deg, start, end)` with `start..end`
    /// the rows coming from generators of `modules[b.s() - 1]` in degree `gen_deg`.
    ///
    /// The blocks tile `0..target_dim` exactly and in order, because a free module's basis is
    /// generator-major. Only reads generator counts strictly below `b.t()`, the same frozen prefix
    /// [`MilnorSubalgebra::restricted_dimension`] reads, so it is safe to call while generators of
    /// degree `b.t()` are being added concurrently.
    fn block_ranges(&self, b: Bidegree) -> Vec<(i32, usize, usize)> {
        let target = &*self.modules[b.s() - 1];
        // Per-bidegree, NOT hoisted over the whole range: a module whose basis is computed through
        // `max.t()` up front can no longer have generators added back into it.
        target.compute_basis(b.t());
        // `iter_gen_offsets` yields one entry per GENERATOR, not per generator degree — `iter_gens`
        // flat-maps over `0..num_gens[t]`, so a degree with several generators appears several
        // times and `start`/`end` bound one generator each. Keying blocks by degree while reading
        // these entries one-for-one meant only the FIRST generator of each degree was ever built;
        // every other generator silently fell through to the consumer's rebuild, which is why row
        // coverage sat at 60-84% instead of near 100%.
        //
        // So the entries are kept per generator and a degree simply yields several pieces, each
        // published as it lands. Merging them into one range per degree also covers every generator
        // but makes the unit coarse and all-or-nothing, and coverage fell (86.3% -> 68.1% at stem
        // 30) because fewer finished before their bidegree ran.
        target
            .iter_gen_offsets([b.t()])
            .take_while(|g| g.gen_deg < b.t())
            .filter(|g| g.end[0] > g.start[0])
            .map(|g| (g.gen_deg, g.start[0], g.end[0]))
            .collect()
    }

    /// EVERY signature piece for one generator degree, in a single pass.
    ///
    /// Returns `(sig_idx, offset within that signature's matrix, rows)`.
    ///
    /// Doing one signature at a time cost O(generators x ppart_table x signatures) per degree,
    /// because it rescanned every generator and its whole operation table for each signature in
    /// turn. At stem 200 that produced 2 988 096 tiny pieces and ran 5077 s against 3799 s for plain
    /// blocks. Every operation has exactly ONE signature, and all signatures of a subalgebra share
    /// the same packed MASK (it depends only on the profile), so a single pass bucketing ops by
    /// `op.bits() & mask` does the whole degree for what one signature used to cost.
    ///
    /// Frontier-free for the same reason `signature_mask` is: the op indices come from
    /// `ppart_table(t - gen_deg)` and the packed signature alone, never the module's degree-`t`
    /// basis. The offsets count matching rows from lower generator degrees, which is what makes each
    /// piece a contiguous run of its signature's matrix.
    fn signature_pieces(
        &self,
        b: Bidegree,
        subalgebra: &MilnorSubalgebra,
        signatures: &[Vec<PPartEntry>],
        gen_deg: i32,
    ) -> Vec<(usize, usize, Vec<usize>)> {
        let target = &*self.modules[b.s() - 1];
        target.compute_basis(b.t());
        let algebra = target.algebra();

        let mut mask = 0u64;
        let mut index: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        for (i, sig) in signatures.iter().enumerate() {
            if let Some((m, v)) = subalgebra.packed_signature(sig) {
                mask = m;
                index.insert(v, i);
            }
        }
        if index.is_empty() {
            return Vec::new();
        }

        let mut starts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut rows: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for gd in target
            .iter_gen_offsets([b.t()])
            .take_while(|g| g.gen_deg < b.t())
        {
            if gd.gen_deg > gen_deg {
                break;
            }
            let table = algebra.ppart_table(b.t() - gd.gen_deg);
            if gd.gen_deg < gen_deg {
                for op in table {
                    if let Some(&i) = index.get(&(op.bits() & mask)) {
                        *starts.entry(i).or_insert(0) += 1;
                    }
                }
            } else {
                for (n, op) in table.iter().enumerate() {
                    if let Some(&i) = index.get(&(op.bits() & mask)) {
                        rows.entry(i).or_default().push(gd.start[0] + n);
                    }
                }
            }
        }
        rows.into_iter()
            .map(|(i, r)| (i, starts.get(&i).copied().unwrap_or(0), r))
            .collect()
    }

    /// Column count for the block at `gen_deg`: the restricted dimension of `modules[b.s() - 2]`
    /// counting only generators of degree *strictly* below `gen_deg`.
    ///
    /// That bound is minimality. `d(x)` for a generator `x` of degree `gen_deg` lands in the radical,
    /// so it is a combination of `op · y` with `deg op > 0` and hence `deg y < gen_deg`; acting by
    /// `Sq(R)` keeps the generator, so the block cannot touch a column beyond this. Building the
    /// block this narrow is what lets it be built before `modules[b.s() - 2]` has grown to `b.t()`.
    fn block_cols(&self, b: Bidegree, gen_deg: i32) -> usize {
        let next = &self.modules[b.s() - 2];
        next.compute_basis(b.t());
        MilnorSubalgebra::restricted_dimension(next, b.t(), gen_deg.min(b.t() - 1))
    }

    /// Total size of `b`'s matrix in the narrow block representation, in `rows x cols` units.
    fn narrow_work(&self, b: Bidegree) -> u64 {
        self.block_ranges(b)
            .into_iter()
            .map(|(gen_deg, start, end)| {
                ((end - start) as u64).saturating_mul(self.block_cols(b, gen_deg) as u64)
            })
            .sum()
    }

    /// Smallest piece worth precomputing, in rows (`NASSAU_SIG_MIN_ROWS`).
    ///
    /// Signature pieces are naturally tiny — stem 200 averaged 32 rows each — and below some size a
    /// piece cannot repay its allocation, lock and bookkeeping. Signature mass is also very uneven
    /// (one signature is often ~96% of its bidegree), so a threshold discards a great many pieces
    /// while keeping nearly all the rows.
    fn sig_min_rows() -> usize {
        static M: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("NASSAU_SIG_MIN_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512)
        });
        *M
    }

    /// Build one row block of `b`'s matrix ahead of time and park it in the [`blocks`] cache.
    #[tracing::instrument(skip(self), fields(%b, gen_deg, rows, cols))]
    fn speculate_block(&self, b: Bidegree, gen_deg: i32) {
        if !blocks::has_room() {
            blocks::bail(0);
            return;
        }
        if self.has_computed_bidegree(b) {
            blocks::bail(1);
            return;
        }
        if let Some(store) = self.save_dir.store()
            && store.exists(SaveKind::NassauDifferential, b)
        {
            return;
        }
        if !reuse_full_matrix(&self.differentials[b.s() - 1]) {
            blocks::bail(3);
            return;
        }
        // Don't precompute for a bidegree that will never collect. The consumer only assembles when
        // the matrix is within `reuse_within_cap`; above it, it builds per signature and never calls
        // `take_all`, so every block built here would be wasted work AND would sit in the cache
        // until the run ended — which is where `unused=150.3GB` at stem 200 came from.
        //
        // The dims are read early, before the bidegree's frontier is frozen, so they are a LOWER
        // bound on the final ones. That makes this sound in one direction only: already over the cap
        // now means over it for good. The consumer releases whatever slips through.
        let ranges = self.block_ranges(b);
        if !ranges.iter().any(|&(g, _, _)| g == gen_deg) {
            // The generators of `gen_deg` were not there after all, or contribute no rows.
            return;
        }
        // The cap check reuses what `block_ranges` already computed rather than calling
        // `restricted_dims`, which would repeat the same `compute_basis` work per block: doing that
        // slowed the builders enough to cut row coverage from 93% to 55% at stem 30. The blocks tile
        // the restricted source basis, so the last range's end IS the row count, and the column
        // count is the widest block's.
        let target_dim = ranges.last().map_or(0, |&(_, _, e)| e);
        let next_dim = self.block_cols(b, b.t() - 1);
        let cols = self.block_cols(b, gen_deg);
        if cols == 0 {
            return;
        }
        // Claim last, once every bail-out is behind us.
        if !blocks::claim(b, gen_deg) {
            blocks::bail(2);
            return;
        }
        let guard = blocks::ClaimGuard::new(b, gen_deg);
        let diff = &self.differentials[b.s() - 1];
        let build = |rows: &[usize]| {
            if speculate::on_cpu() {
                restricted_partial_matrix(diff, b.t(), rows, cols)
            } else {
                restricted_partial_matrix_maybe_gpu(diff, b.t(), rows, cols)
            }
        };

        if reuse_within_cap(target_dim, next_dim) {
            // Within the cap the consumer assembles ONE full matrix, so deliver one contiguous
            // range covering EVERY generator of this degree. Generators of a degree are adjacent,
            // so the merged range is still contiguous.
            //
            // Publishing them as separate pieces instead covers the same rows but costs 17% of wall
            // time at stem 150 (203/201 s against 172/173 s) and covers slightly LESS: ~36% more
            // pieces means more work per queue item, so builders reach fewer degrees before the
            // wavefront arrives. Granularity here is not free.
            // One piece per degree covering only its FIRST generator, by default.
            //
            // That sounds like it leaves work on the table, and it does — but not measurably, and
            // covering more costs. Stem 150, θ=125, one binary, interleaved:
            //
            //     first generator only   173 s   coverage 53.4%
            //     merged over the degree 194 s   coverage 53.3%
            //     one piece per generator 202 s  coverage 51.4%
            //
            // Coverage is flat because a degree almost always carries a single generator here, so
            // the extra generators are nearly empty; what changes is piece size and builder load,
            // and both bigger pieces and more pieces lose. Speculation is already at its useful
            // limit — past it the builders just compete with the wavefront. `NASSAU_BLOCK_MERGE=1`
            // covers every generator of the degree, for regimes where generators are denser.
            let merge = std::env::var("NASSAU_BLOCK_MERGE").as_deref() == Ok("1");
            let mut it = ranges.iter().filter(|&&(g, _, _)| g == gen_deg);
            let first = *it.next().unwrap();
            let (start, end) = if merge {
                (
                    first.1,
                    it.fold(first.2, |acc, &(_, _, e)| std::cmp::max(acc, e)),
                )
            } else {
                (first.1, first.2)
            };
            let rows: Vec<usize> = (start..end).collect();
            let t0 = std::time::Instant::now();
            blocks::builder_enter();
            let m = speculate::pool().install(|| build(&rows));
            blocks::builder_exit();
            blocks::record_build_nanos(t0.elapsed().as_nanos() as u64);
            blocks::publish(b, gen_deg, start, m);
            guard.done();
            return;
        }

        // Above the cap the consumer never assembles a full matrix -- it builds one signature at a
        // time and discards it, which is exactly the memory bound the cap exists to hold. So deliver
        // along the SIGNATURE axis instead: one piece per (signature, generator degree), each a
        // contiguous run of the signature's matrix. Peak stays at one signature however large the
        // bidegree is, so this reaches the work plain blocks cannot (86.9% of it at stem 200), and
        // it costs no extra multiplies -- signatures partition the operations, so this is a strict
        // refinement of the same rows.
        let subalgebra = MilnorSubalgebra::optimal_for(b);
        let signatures: Vec<Vec<PPartEntry>> = subalgebra.iter_signatures(b.t()).collect();
        let pieces = self.signature_pieces(b, &subalgebra, &signatures, gen_deg);
        let min_rows = Self::sig_min_rows();
        speculate::pool().install(|| {
            for (sig_idx, start, rows) in pieces {
                if blocks::is_done(b) || !blocks::has_room() {
                    break;
                }
                if rows.len() < min_rows || !blocks::claim_sig(b, gen_deg, sig_idx) {
                    continue;
                }
                blocks::publish_sig(b, sig_idx, start, build(&rows));
            }
        });
        guard.done();
    }

    /// Assemble `b`'s full restricted matrix from whatever row blocks were precomputed, building
    /// everything else in a single coalesced launch.
    ///
    /// A missing block is cheap here — its rows simply join the build the consumer was making
    /// anyway — which is why nothing waits on an in-flight block. The one big launch per bidegree
    /// that [`reuse_full_matrix`] exists to create therefore survives: at worst it is the original
    /// launch, at best it shrinks to the rows speculation did not cover.
    fn assemble_full_restricted(&self, b: Bidegree, target_dim: usize, next_dim: usize) -> Matrix {
        let mut full = Matrix::new(self.prime(), target_dim, next_dim);
        let ready = blocks::take_all(b);
        // Walk the row axis once, alternating between "a cached block covers these rows" and "these
        // rows are mine to build". Blocks arrive sorted by offset and cannot overlap (one block per
        // generator degree, each claimed once), but a stale one is dropped rather than trusted: a
        // block that runs past `target_dim` or backwards over a row already covered would corrupt
        // the matrix silently, and re-deriving those rows costs only their own multiply.
        let mut missing: Vec<usize> = Vec::new();
        let (mut hits, mut misses, mut rows_hit) = (0usize, 0usize, 0usize);
        let mut cursor = 0usize;
        for (start, m) in &ready {
            let (start, end) = (*start, start + m.rows());
            if start < cursor || end > target_dim || m.columns() > next_dim {
                tracing::warn!(
                    %b,
                    "discarding speculative block: rows {start}..{end} x {} against {target_dim} x \
                     {next_dim} (cursor {cursor})",
                    m.columns(),
                );
                misses += 1;
                continue;
            }
            missing.extend(cursor..start);
            for (i, r) in (start..end).enumerate() {
                full.row_mut(r).slice_mut(0, m.columns()).assign(m.row(i));
            }
            hits += 1;
            rows_hit += m.rows();
            cursor = end;
        }
        missing.extend(cursor..target_dim);
        blocks::record(hits, misses, rows_hit, missing.len());
        if blocks::opdeg_hist() {
            let covered: std::collections::HashSet<usize> =
                ready.iter().map(|(start, _)| *start).collect();
            for (gen_deg, start, end) in self.block_ranges(b) {
                blocks::record_opdeg(
                    b.t() - gen_deg,
                    (end - start) as u64,
                    covered.contains(&start),
                );
            }
        }
        if !missing.is_empty() {
            let part = restricted_partial_matrix_maybe_gpu(
                &self.differentials[b.s() - 1],
                b.t(),
                &missing,
                next_dim,
            );
            for (i, &r) in missing.iter().enumerate() {
                full.row_mut(r).assign(part.row(i));
            }
        }
        full
    }

    /// Assemble one signature's matrix from precomputed pieces, building the rest in a single
    /// coalesced call.
    ///
    /// `target_mask` lists the signature's rows in layout order, so a piece covering generator
    /// degree `g` occupies a contiguous run of it — the same walk-and-fill as
    /// [`Self::assemble_full_restricted`], one axis over. Pieces are narrow (minimality) and the
    /// destination is allocated at `next_dim` up front, so every write is a row copy into an
    /// existing row: matrices are row-major, and widening one afterwards would restride every row.
    fn assemble_signature(
        &self,
        b: Bidegree,
        sig_idx: usize,
        target_mask: &[usize],
        next_dim: usize,
    ) -> Matrix {
        let ready = blocks::take_sig(b, sig_idx);
        // Nothing precomputed: build straight into the result. Going through the assembly path
        // anyway would allocate this matrix, build a SECOND one for the missing rows, and copy row
        // by row -- pure overhead, and it cost 7% at stem 200 on bidegrees where no piece landed.
        if ready.is_empty() {
            blocks::record(0, 0, 0, target_mask.len());
            return restricted_partial_matrix_maybe_gpu(
                &self.differentials[b.s() - 1],
                b.t(),
                target_mask,
                next_dim,
            );
        }
        let mut full = Matrix::new(self.prime(), target_mask.len(), next_dim);
        let mut missing: Vec<usize> = Vec::new();
        let (mut hits, mut misses, mut rows_hit) = (0usize, 0usize, 0usize);
        let mut cursor = 0usize;
        for (start, m) in &ready {
            let (start, end) = (*start, start + m.rows());
            if start < cursor || end > target_mask.len() || m.columns() > next_dim {
                tracing::warn!(
                    %b, sig_idx,
                    "discarding speculative signature piece: rows {start}..{end} x {} against {} x \
                     {next_dim} (cursor {cursor})",
                    m.columns(),
                    target_mask.len(),
                );
                misses += 1;
                continue;
            }
            missing.extend(cursor..start);
            for (i, pos) in (start..end).enumerate() {
                full.row_mut(pos).slice_mut(0, m.columns()).assign(m.row(i));
            }
            hits += 1;
            rows_hit += m.rows();
            cursor = end;
        }
        missing.extend(cursor..target_mask.len());
        blocks::record(hits, misses, rows_hit, missing.len());
        let verify = speculate::verify() && !ready.is_empty();
        if !missing.is_empty() {
            let rows: Vec<usize> = missing.iter().map(|&pos| target_mask[pos]).collect();
            let part = restricted_partial_matrix_maybe_gpu(
                &self.differentials[b.s() - 1],
                b.t(),
                &rows,
                next_dim,
            );
            for (i, &pos) in missing.iter().enumerate() {
                full.row_mut(pos).assign(part.row(i));
            }
        }
        if verify {
            let fresh = restricted_partial_matrix(
                &self.differentials[b.s() - 1],
                b.t(),
                target_mask,
                next_dim,
            );
            for r in 0..target_mask.len() {
                assert_eq!(
                    full.row(r).iter_nonzero().collect::<Vec<_>>(),
                    fresh.row(r).iter_nonzero().collect::<Vec<_>>(),
                    "assembled signature matrix mismatch at {b}, signature {sig_idx}, row {r}"
                );
            }
        }
        full
    }

    /// Build `b`'s full matrix ahead of time and park it in the [`speculate`] cache.
    ///
    /// Only called for `b.s() >= 2` and only once `(b.s() - 1, b.t() - 1)` is committed, which is
    /// what makes every input frozen. Bails out whenever the result would not be used: when the
    /// bidegree is already computed or on disk, when the reuse path is off or the matrix exceeds its
    /// cap (in which case the consumer builds per-signature instead), or when the cache is full.
    #[tracing::instrument(skip(self), fields(%b))]
    fn speculate_build(&self, b: Bidegree) {
        if self.has_computed_bidegree(b) || !speculate::has_room() {
            return;
        }
        if let Some(store) = self.save_dir.store()
            && store.exists(SaveKind::NassauDifferential, b)
        {
            return;
        }
        if !reuse_full_matrix(&self.differentials[b.s() - 1]) {
            return;
        }
        let (target_dim, next_dim) = self.restricted_dims(b);
        if !reuse_within_cap(target_dim, next_dim) {
            return;
        }
        // Claim last, once every bail-out is behind us: a claim obliges us to publish, because a
        // consumer reaching this bidegree will wait on the slot rather than build it itself.
        if !speculate::claim(b) {
            return;
        }
        let guard = speculate::ClaimGuard::new(b);
        // Inside the speculative pool, so the build's own `par_iter`s cannot take workers from the
        // wavefront's pool -- see [`speculate::pool`].
        let on_cpu = speculate::on_cpu();
        let m = speculate::pool()
            .install(|| self.build_full_restricted(b, target_dim, next_dim, on_cpu));
        speculate::publish(b, m);
        guard.done();
    }

    #[tracing::instrument(skip(self), fields(%b, %subalgebra, num_new_gens, density))]
    fn step_resolution_with_subalgebra(
        &self,
        b: Bidegree,
        subalgebra: MilnorSubalgebra,
    ) -> anyhow::Result<()> {
        let end = || {
            tracing::Span::current().record("num_new_gens", self.number_of_gens_in_bidegree(b));
            tracing::Span::current().record(
                "density",
                self.differentials[b.s()].differential_density(b.t()) * 100.0,
            );
        };

        let p = self.prime();
        let mut scratch = FpVector::new(p, 0);

        // Census records counts, never durations, so a census run may be arbitrarily slower than
        // production without invalidating a number. See [`crate::census`].
        let mut census =
            crate::census::enabled().then(|| crate::census::BidegreeCensus::new(b.s(), b.t()));

        let target = &*self.modules[b.s() - 1];
        let algebra = target.algebra();

        // We compute this bidegree treating the target `C_{b.s() - 1}` as if it had no generators of
        // degree `>= b.t()`, and `C_{b.s() - 2}` as if it had no generators of degree `>= b.t() - 1`.
        // By minimality this loses no information (the differentials we care about land in the
        // radical, hence in strictly lower-degree generators), and it makes the computation depend
        // only on data that is frozen once `(b.s() - 1, b.t() - 1)` and `(b.s(), b.t() - 1)` have
        // been committed. This is what lets [`Self::compute_through_stem`] compute `(b.s(), b.t())`
        // concurrently with `(b.s() - 1, b.t())`, which is adding those degree-`b.t()` generators.
        let target_bound = b.t();
        let next_bound = b.t() - 1;

        let zero_sig = subalgebra.zero_signature();
        let target_dim = MilnorSubalgebra::restricted_dimension(target, b.t(), target_bound);
        let target_mask: Vec<usize> = subalgebra
            .signature_mask(&algebra, target, b.t(), &zero_sig, target_bound)
            .collect();
        let target_masked_dim = target_mask.len();

        let next = &self.modules[b.s() - 2];
        next.compute_basis(b.t());
        let next_dim = MilnorSubalgebra::restricted_dimension(next, b.t(), next_bound);

        if let Some(c) = census.as_mut() {
            // `target_masked_dim` is the zero-signature row count: the rows that signature-shift
            // reuse would still have to multiply. `target_dim` is what we multiply today.
            let dim_b: usize = subalgebra
                .profile
                .iter()
                .map(|&e| 1usize << e)
                .product::<usize>()
                .max(1);
            c.dims(target_dim, target_masked_dim, next_dim, dim_b);
        }

        // Skip writing the quasi-inverse when `EXT_NASSAU_NO_SAVE_QI` is set; `apply_quasi_inverse`
        // recomputes it on demand from the differential.
        let mut f = if *SAVE_QI && let Some(store) = self.save_dir.store() {
            let qi_b = b - Bidegree::s_t(1, 0);
            Some(store.nassau_qi_writer(
                qi_b,
                next_dim as u64,
                target_masked_dim as u64,
                &subalgebra.profile,
            )?)
        } else {
            None
        };

        let guard = tracing::info_span!("step", signature = ?zero_sig).entered();
        let next_mask: Vec<usize> = tracing::trace_span!("zs_masks").in_scope(|| {
            subalgebra
                .signature_mask(&algebra, next, b.t(), &zero_sig, next_bound)
                .collect()
        });
        let next_masked_dim = next_mask.len();

        // When GPU reuse is active, build ONE full restricted matrix over every (restricted) source
        // row at degree `b.t()` in a single launch, then slice each signature's rows out of it.
        // `target_dim` is the restricted source dimension, so `0..target_dim` is exactly the row set
        // the per-signature masks partition; `next_dim` is the restricted column count.
        // With shift reuse on, every nonzero signature's matrix comes from the shift cache, so
        // this bidegree's full matrix is needed only for the ZERO-signature step and for the rows
        // the lift actually consumes. Stop building all `target_dim` rows and serve those two on
        // demand -- the census's 11.88% + 21.47% = 33.35%. Gated on `f.is_none()` because
        // `write_qi` reads the full matrix directly.
        #[cfg(feature = "gpu")]
        let shift_skip_full = shift::enabled() && f.is_none();
        #[cfg(not(feature = "gpu"))]
        let shift_skip_full = false;

        let full_reuse: Option<Matrix> = if !shift_skip_full
            && reuse_full_matrix(&self.differentials[b.s() - 1])
            && reuse_within_cap(target_dim, next_dim)
        {
            NARROW_WORK.fetch_add(self.narrow_work(b), std::sync::atomic::Ordering::Relaxed);
            REUSE_WORK.fetch_add(
                (target_dim as u64).saturating_mul(next_dim as u64),
                std::sync::atomic::Ordering::Relaxed,
            );
            REUSE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let all_rows: Vec<usize> = (0..target_dim).collect();
            // A background builder may already have this matrix. Its inputs are frozen once
            // `(b.s() - 1, b.t() - 1)` is committed, strictly earlier than `(b.s(), b.t())` can run,
            // so a cached matrix is the same matrix — see [`speculate`]. A shape mismatch would mean
            // that reasoning is wrong somewhere, so say so loudly and rebuild rather than trusting
            // it; `NASSAU_SPECULATE_VERIFY` checks the contents too.
            let full = if blocks::enabled() {
                let m = self.assemble_full_restricted(b, target_dim, next_dim);
                if speculate::verify() {
                    let fresh = self.build_full_restricted(b, target_dim, next_dim, false);
                    for r in 0..target_dim {
                        assert_eq!(
                            m.row(r).iter_nonzero().collect::<Vec<_>>(),
                            fresh.row(r).iter_nonzero().collect::<Vec<_>>(),
                            "assembled block matrix mismatch at {b}, row {r}"
                        );
                    }
                }
                m
            } else {
                match speculate::take_or_claim(b) {
                    Some(m) if m.rows() == target_dim && m.columns() == next_dim => {
                        if speculate::verify() {
                            let fresh = self.build_full_restricted(b, target_dim, next_dim, false);
                            for r in 0..target_dim {
                                assert_eq!(
                                    m.row(r).iter_nonzero().collect::<Vec<_>>(),
                                    fresh.row(r).iter_nonzero().collect::<Vec<_>>(),
                                    "speculative matrix mismatch at {b}, row {r}"
                                );
                            }
                        }
                        m
                    }
                    stale => {
                        if let Some(m) = stale {
                            tracing::warn!(
                                %b,
                                "discarding speculative matrix: got {}x{}, want {target_dim}x{next_dim}",
                                m.rows(),
                                m.columns(),
                            );
                        }
                        self.build_full_restricted(b, target_dim, next_dim, false)
                    }
                }
            };
            // `NASSAU_SPLIT_VERIFY`: check the ROW-BLOCK DECOMPOSITION empirically.
            //
            // The speculative plan for capped-theta high stems rests on one claim: the rows of this
            // matrix coming from generators that already exist can be computed AHEAD of the bidegree
            // and concatenated with the rest, because rows are generator-major and the module's
            // basis tables are append-only (`OnceVec`), so an early row block stays valid verbatim.
            // That is an argument from data-structure semantics; this checks it against real data
            // before anything is built on it. A row-range build must equal the corresponding rows of
            // the all-rows build, bit for bit.
            if std::env::var_os("NASSAU_SPLIT_VERIFY").is_some() && target_dim > 1 {
                let mid = target_dim / 2;
                let lo = restricted_partial_matrix_maybe_gpu(
                    &self.differentials[b.s() - 1],
                    b.t(),
                    &all_rows[..mid],
                    next_dim,
                );
                let hi = restricted_partial_matrix_maybe_gpu(
                    &self.differentials[b.s() - 1],
                    b.t(),
                    &all_rows[mid..],
                    next_dim,
                );
                for r in 0..target_dim {
                    let split = if r < mid { lo.row(r) } else { hi.row(r - mid) };
                    assert_eq!(
                        full.row(r).iter_nonzero().collect::<Vec<_>>(),
                        split.iter_nonzero().collect::<Vec<_>>(),
                        "row-block decomposition mismatch at {b}, row {r} (mid {mid}, target_dim \
                         {target_dim}, next_dim {next_dim})"
                    );
                }
            }
            Some(full)
        } else {
            // How much of the run is out of reach of blocks entirely?
            //
            // `row_rate` only measures bidegrees that take the reuse path; a bidegree over
            // `reuse_within_cap` builds per signature and contributes nothing to it, so a high
            // `row_rate` is consistent with speculation being irrelevant to most of the work. This
            // splits the total multiply work (`rows x cols`) both ways to say which.
            if reuse_full_matrix(&self.differentials[b.s() - 1]) {
                NARROW_WORK.fetch_add(self.narrow_work(b), std::sync::atomic::Ordering::Relaxed);
                NOREUSE_WORK.fetch_add(
                    (target_dim as u64).saturating_mul(next_dim as u64),
                    std::sync::atomic::Ordering::Relaxed,
                );
                NOREUSE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // This bidegree is building per signature, so nothing will ever collect its blocks.
            // Release them here rather than leaving them pinned until the end of the run: they hold
            // memory against `has_room()`, which would starve speculation for the bidegrees that
            // DO collect.
            if blocks::enabled() {
                blocks::release(b);
            }
            None
        };

        let full_matrix =
            tracing::trace_span!("zs_select", rows = target_mask.len()).in_scope(|| {
                match &full_reuse {
                    Some(full) => {
                        debug_assert!(target_mask.iter().all(|&r| r < full.rows()));
                        PartialMatrix::Gather {
                            full,
                            rows: &target_mask,
                        }
                    }
                    None => {
                        N_ZS.fetch_add(1, Ordering::Relaxed);
                        // Build straight into masked columns when nothing downstream needs the
                        // full width. `write_qi` does (it reads `full_matrix.row(i)` against the
                        // full output basis), and so does the `NASSAU_PROBE_SHIFT=2` matrix probe,
                        // so both gate this off -- the same condition `shift_skip_full` uses.
                        if masked_cols_enabled() && f.is_none() && !shift_probe_matrices() {
                            crate::census::add_bytes(
                                &crate::census::MASKED_SAVED_BYTES,
                                crate::census::matrix_bytes(target_mask.len(), next_dim)
                                    .saturating_sub(crate::census::matrix_bytes(
                                        target_mask.len(),
                                        next_mask.len(),
                                    )),
                            );
                            PartialMatrix::PreMasked(restricted_partial_matrix_masked_maybe_gpu(
                                &self.differentials[b.s() - 1],
                                b.t(),
                                &target_mask,
                                next_dim,
                                &next_mask,
                            ))
                        } else {
                            PartialMatrix::Owned(restricted_partial_matrix_maybe_gpu(
                                &self.differentials[b.s() - 1],
                                b.t(),
                                &target_mask,
                                next_dim,
                            ))
                        }
                    }
                }
            });
        let mut masked_matrix = tracing::trace_span!(
            "zs_assemble",
            rows = target_masked_dim,
            cols = next_masked_dim
        )
        .in_scope(|| {
            let mut m =
                AugmentedMatrix::new(p, target_masked_dim, [next_masked_dim, target_masked_dim]);
            // Row gather and column mask fused into one pass. `Matrix::add_masked` would need a
            // materialised `full_matrix`; going row by row lets the gather stay an indirection.
            for (i, l) in m.segment(0, 0).iter_mut().enumerate() {
                full_matrix.add_row_masked(l, i, &next_mask);
            }
            // Counted per ASSEMBLY, not per row: the loop above runs up to `target_dim` times
            // (700k+ at the frontier), so an increment inside it would be a contended atomic per
            // row across 96 workers. One increment per assembled matrix carries the same total.
            crate::census::add_bytes(
                &crate::census::ADD_MASKED_BYTES,
                crate::census::matrix_bytes(target_masked_dim, next_masked_dim),
            );
            crate::census::add_bytes(
                &crate::census::AUGMENTED_ALLOC_BYTES,
                crate::census::matrix_bytes(target_masked_dim, next_masked_dim + target_masked_dim),
            );
            m.segment(1, 1).add_identity();
            m
        });

        tracing::trace_span!(
            "zs_row_reduce",
            rows = target_masked_dim,
            cols = next_masked_dim
        )
        .in_scope(|| masked_matrix.row_reduce());
        let kernel = tracing::trace_span!("zs_kernel").in_scope(|| masked_matrix.compute_kernel());

        Self::write_qi(
            &mut f,
            &mut scratch,
            &zero_sig,
            &next_mask,
            &full_matrix,
            &masked_matrix,
        )?;

        // The quasi-inverse is always computed on the restricted (degree `< b.t()`) target basis, so
        // from the point of view of a later `apply_quasi_inverse` it was computed with "incomplete
        // information": the differentials on the degree-`b.t()` generators of the target were not
        // available. We flag this unconditionally so the lift is corrected using those differentials
        // once they are known. The [`NassauCommand::Fix`] is a no-op when there is nothing to
        // correct (by minimality the excluded generators receive no differential).
        if let Some(f) = &mut f {
            f.write_fix()?;
        }

        // Compute image: d_s applied to the zero-signature source basis, column-masked to the
        // zero-signature target basis. This is the same restricted multiply as `full_matrix` above
        // (on d_s = differentials[b.s()] rather than d_{s-1}), and its target mask/dimension are
        // exactly the `target_mask`/`target_dim` already computed for this bidegree — d_s and
        // d_{s-1} share the target module `modules[b.s() - 1]`. So route it through the same
        // (GPU-offloaded, work-gated) restricted-matrix path and apply the column mask on CPU,
        // instead of `signature_matrix`'s serial per-row CPU multiply.
        let source_mask: Vec<usize> = tracing::trace_span!("zs_source_mask").in_scope(|| {
            subalgebra
                .signature_mask(&algebra, &self.modules[b.s()], b.t(), &zero_sig, i32::MAX)
                .collect()
        });
        // Built DIRECTLY in masked columns. This used to build `source_mask.len() x target_dim`
        // and immediately gather it down to `target_masked_dim` -- and, because the full matrix
        // was bound with `let`, it then stayed alive for the rest of the bidegree.
        //
        // Heap profiling (jemalloc, sampled) put this one site at **67.19GB across 18 live
        // objects, 3.73GB each, 58% of all live Rust bytes** -- one per in-flight bidegree. It is
        // the largest consumer in a frontier run by a wide margin, and it is the same
        // build-full-then-mask shape already removed from the signature path; it survived only
        // because it sits in the zero-signature preamble rather than the signature loop.
        //
        // Guessing had blamed the on-demand path, on the strength of a cumulative byte counter
        // (2.78TB allocated). Chunking that path measured ZERO change in peak RSS. Cumulative
        // counters measure churn; only a live-heap profile answers residency.
        let mut n = tracing::trace_span!("img_assemble", rows = source_mask.len()).in_scope(|| {
            restricted_partial_matrix_masked_maybe_gpu(
                &self.differentials[b.s()],
                b.t(),
                &source_mask,
                target_dim,
                &target_mask,
            )
        });
        debug_assert_eq!(n.columns(), target_masked_dim);
        tracing::trace_span!("img_row_reduce", rows = source_mask.len())
            .in_scope(|| n.row_reduce());
        let next_row = n.rows();

        let num_new_gens = tracing::trace_span!("extend_image")
            .in_scope(|| n.extend_image(0, n.columns(), &kernel, 0).len());

        if b.t() < b.s() {
            assert_eq!(num_new_gens, 0, "Adding generators at {b}");
        }

        if let Some(c) = census.as_mut() {
            c.set_new_gens(num_new_gens);
        }

        self.add_generators(b, num_new_gens);

        let mut xs = vec![FpVector::new(p, target_dim); num_new_gens];
        let mut dxs = vec![FpVector::new(p, next_dim); num_new_gens];

        {
            let _s = tracing::trace_span!("zs_dx_init", gens = xs.len()).entered();
            // `dx` is FULL width and must stay that way: the signature loop below reads it at
            // `next_mask` positions that DIFFER per signature, so the columns outside this
            // signature's mask are live information, not slack. When `full_matrix` was built in
            // masked coordinates it cannot serve this, so fetch exactly the rows the new
            // generators actually touch at full width -- the same on-demand shape the shift path
            // uses, and only `|support|` rows rather than all of `target_mask`.
            let dx_rows: Option<(Vec<usize>, Matrix)> = if full_matrix.is_pre_masked() {
                let mut needed: Vec<usize> = n
                    .iter()
                    .skip(next_row)
                    .take(xs.len())
                    .flat_map(|x_masked| x_masked.iter_nonzero().map(|(i, _)| i))
                    .collect();
                needed.sort_unstable();
                needed.dedup();
                if needed.is_empty() {
                    None
                } else {
                    let basis: Vec<usize> = needed.iter().map(|&i| target_mask[i]).collect();
                    let got =
                        tracing::trace_span!("zs_dx_ondemand", rows = basis.len()).in_scope(|| {
                            restricted_partial_matrix_maybe_gpu(
                                &self.differentials[b.s() - 1],
                                b.t(),
                                &basis,
                                next_dim,
                            )
                        });
                    Some((needed, got))
                }
            } else {
                None
            };
            for ((x, x_masked), dx) in xs
                .iter_mut()
                .zip_eq(n.iter().skip(next_row))
                .zip_eq(&mut dxs)
            {
                x.as_slice_mut().add_unmasked(x_masked, 1, &target_mask);
                let mut consumed = 0usize;
                for (i, _) in x_masked.iter_nonzero() {
                    match &dx_rows {
                        Some((needed, got)) => {
                            let k = needed
                                .binary_search(&i)
                                .expect("dx support row was collected above");
                            dx.as_slice_mut().add(got.row(k), 1);
                        }
                        None => dx.as_slice_mut().add(full_matrix.row(i), 1),
                    }
                    consumed += 1;
                }
                if let Some(c) = census.as_mut() {
                    c.add_rows_consumed(consumed);
                }
            }
        }

        // Now add correction terms
        let mut target_mask: Vec<usize> = Vec::new();
        let mut next_mask: Vec<usize> = Vec::new();

        drop(guard);

        // Probe (`NASSAU_PROBE_SIG_INDEP=1`): are the signature steps independent?
        //
        // Each step reads `dx.entry(v)` for `v` in its own `next_mask`, then writes `dx` with rows
        // of the *unmasked* `full_matrix`, whose support can extend outside that mask. If those
        // writes never land on a column another step later reads, the steps are solving against an
        // unchanging `dx` and the loop is parallelisable (solve independently, combine). If they do,
        // the loop is a forward substitution and must stay ordered. Comparing each read against a
        // pre-loop snapshot answers exactly that, without altering what the loop computes.
        //
        // MEASURED (S_2 stem 60, max_s 30, 2233 bidegrees): 27 028 of 140 614 reads — 19.2% — see a
        // value an earlier signature wrote. So the steps are NOT independent and the loop must stay
        // ordered: it is a forward substitution, and "solve every signature separately, then
        // combine" would be wrong, not merely racy.
        //
        // That is a negative result about the LIFT only, and the lift is the cheap end. Everything
        // above it — `sig_masks`, `sig_select` (where ~91% of `gpu_submit` lives), `sig_assemble`,
        // `sig_row_reduce`, `sig_quasi_inverse` — reads only `full_reuse`/the differentials and
        // never touches `dxs` or `xs`, so it CAN legally run several signatures at a time.
        //
        // DO NOT BOTHER: that was built (a windowed prepare stage feeding an ordered lift) and it is
        // worthless, because the work is not spread across the signatures. Per-bidegree `step` span
        // times over a stem-150 run, 10 738 bidegrees with >= 2 signatures:
        //
        //     sum of per-bidegree TOTAL signature time  4064.7 s
        //     sum of per-bidegree MAX   signature time  3895.7 s   -> ceiling = 1.04x
        //
        // One signature is ~96% of its bidegree, so the ideal speedup from parallelising the loop is
        // 4%. Measured end to end it was worse than that: a window of 4 ran 537 s against a 528 s
        // baseline while raising mean CPU 918% -> 1306%, i.e. it burned 42% more CPU to lose 1.7%.
        // The parallelism this resolution is missing is NOT inside a bidegree.
        let dx_snapshot: Option<Vec<FpVector>> =
            std::env::var_os("NASSAU_PROBE_SIG_INDEP").map(|_| dxs.clone());
        let mut probe_reads = 0usize;
        let mut probe_perturbed = 0usize;

        for (sig_idx, signature) in subalgebra.iter_signatures(b.t()).enumerate() {
            let _guard = tracing::info_span!("step", ?signature).entered();
            // Corrections only ever raise signature, so once every `dx` is zero the whole remaining
            // tail is a provable no-op: the lift below is guarded by `dx.entry(v) != 0`, so it adds
            // nothing to `xs`, and the only other side effect is `write_qi`.
            // So skip it. With no quasi-inverse writer there is no side effect left at all, and
            // everything above the lift (`sig_masks`, `sig_select`, `sig_assemble`,
            // `sig_row_reduce`, `sig_quasi_inverse`) is pure waste.
            //
            // MEASURED (census, S_2 stem 110/max_s 55, 7533 bidegrees): the dead tail is 27.4% of
            // work-weighted signature time — and it coincides exactly with `zero_gen_bidegrees`,
            // which is the whole story. A bidegree with `num_new_gens == 0` has an EMPTY `dxs`, so
            // it is vacuously converged before the first iteration and its entire signature loop
            // was always dead. Bidegrees that do add generators have a negligible tail; this is not
            // an early-convergence optimisation, it is 27.4% of the run computing quasi-inverses
            // nobody asked for.
            //
            // Gated on `f.is_none()` because writing the quasi-inverse is a real side effect: with
            // `EXT_NASSAU_NO_SAVE_QI` unset the loop must still run to completion.
            if f.is_none() && dxs.iter().all(|dx| dx.is_zero()) {
                break;
            }
            // Recorded AFTER the skip, so `signatures` counts iterations actually executed and
            // `dead_signature_tail` keeps meaning "waste still present in the run" — it reads ~0
            // once the skip is on. Counting the one aborted iteration as dead would peg a skipped
            // bidegree at 100% dead forever and hide whether the skip fired at all.
            if let Some(c) = census.as_mut() {
                c.set_signatures(sig_idx + 1);
                c.sig_live(sig_idx, dxs.iter().any(|dx| !dx.is_zero()));
            }
            // Spans below split what used to be one opaque `step`: the run's own accounting put
            // ~26% of worker time inside `step` but outside any named region, which is exactly the
            // shape that produced several wrong diagnoses earlier. One span per signature is cheap
            // (the bodies are substantial); do NOT push spans inside these loops.
            let _sm = tracing::trace_span!("sig_masks").entered();
            target_mask.clear();
            next_mask.clear();
            target_mask.extend(subalgebra.signature_mask(
                &algebra,
                target,
                b.t(),
                &signature,
                target_bound,
            ));
            next_mask.extend(subalgebra.signature_mask(
                &algebra,
                next,
                b.t(),
                &signature,
                next_bound,
            ));
            drop(_sm);

            // Does this signature's problem look like the zero-signature problem at the shifted
            // degree? Necessary condition only, but a mismatch would refute shift reuse outright.
            if shift_probe_enabled() {
                let shifted_t = b.t() - MilnorSubalgebra::signature_degree(&signature);
                if shifted_t < 0 {
                    SHIFT_ABSENT.fetch_add(1, Ordering::Relaxed);
                } else {
                    // Compute the ZERO-signature masks at the shifted degree, but with THIS
                    // bidegree's generator bounds. Reading them off the shifted bidegree instead
                    // compares different generator sets (its bound is `shifted_t`, not `b.t()`),
                    // which is what made the first cut of this probe read 21% for no real reason.
                    let zs_target = subalgebra
                        .signature_mask(&algebra, target, shifted_t, &zero_sig, target_bound)
                        .count();
                    let zs_next = subalgebra
                        .signature_mask(&algebra, next, shifted_t, &zero_sig, next_bound)
                        .count();
                    if zs_target == target_mask.len() && zs_next == next_mask.len() {
                        SHIFT_MATCH.fetch_add(1, Ordering::Relaxed);
                    } else {
                        SHIFT_MISMATCH.fetch_add(1, Ordering::Relaxed);
                        tracing::debug!(
                            "[shift-probe] {b} sig={signature:?} shifted_t={shifted_t} target {} \
                             vs zs {zs_target}, next {} vs zs {zs_next}",
                            target_mask.len(),
                            next_mask.len()
                        );
                    }
                }
            }

            // Shift reuse: take this signature's matrix from the shared zero-signature build at
            // `t - deg(sigma)` rather than from this bidegree. `sig_shift` carries its own column
            // mask, since the shifted matrix lives over the shifted degree's column space.
            // The `bool` is whether the cached matrix is already in masked column coordinates, in
            // which case the mask is only consulted for its LENGTH (the consumer reads a prefix).
            let mut shifted: Option<(Arc<Matrix>, Vec<usize>, bool)> = None;
            // Sibling of `shifted` so its scope is exactly the private matrix's lifetime.
            #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
            let mut _private_live: Option<PrivateLive> = None;
            #[cfg(feature = "gpu")]
            if shift_skip_full {
                let shifted_t = b.t() - MilnorSubalgebra::signature_degree(&signature);
                if shifted_t >= 0 {
                    let _sh = tracing::trace_span!("sig_shift", shifted_t).entered();
                    // Bound `shifted_t + 1`, NOT `i32::MAX`: at degree `shifted_t` only generators
                    // of degree `<= shifted_t` can contribute, so this selects exactly the same
                    // rows as the consumer's own `b.t()` bound while never reading the generator
                    // counts of higher degrees — which another thread is concurrently growing.
                    let zs_bound = shifted_t + 1;
                    // The COLUMN mask must use this bidegree's own `next_bound`, not `zs_bound`.
                    // They differ exactly when `deg(sigma) == 1`, where `shifted_t == b.t() - 1`
                    // and `next_bound` excludes the degree-`shifted_t` generators that `zs_bound`
                    // keeps. `next_bound` is what the matrix-level probe verified, and the cached
                    // matrix carries the maximal column set, so every consumer's mask indexes a
                    // prefix of it and stays in range.
                    let zs_next: Vec<usize> = subalgebra
                        .signature_mask(&algebra, next, shifted_t, &zero_sig, next_bound)
                        .collect();
                    // Who may BUILD the shared entry is a correctness question, not a policy
                    // one. `zs_bound` columns reach `modules[b.s() - 2]` generators of degree up
                    // to `shifted_t`. A consumer is only permitted to see degree `< next_bound`
                    // (= `b.t() - 1`) there -- that bound is exactly what lets this bidegree run
                    // concurrently with the one still ADDING those generators. For
                    // `deg(sigma) == 1`, `shifted_t == b.t() - 1`, so building at `zs_bound` would
                    // read generators being written right now: a race, and the reason this cache
                    // still diverged at stem 90 after the profile was added to the key.
                    //
                    // So only `deg(sigma) >= 2` may build (there `next_bound >= shifted_t + 1`,
                    // making `zs_bound` within its permitted view). A `deg(sigma) == 1` consumer
                    // may still USE an entry someone else built -- it only ever reads the prefix
                    // its own narrower mask names -- but when there is none it builds privately at
                    // its own bound and does not publish.
                    let may_publish = MilnorSubalgebra::signature_degree(&signature) >= 2;
                    let m = match shift::get(b.s() as i32, shifted_t, &subalgebra.profile) {
                        Some(m) => {
                            N_SHIFT_HIT.fetch_add(1, Ordering::Relaxed);
                            m
                        }
                        None => {
                            let bound = if may_publish { zs_bound } else { next_bound };
                            let zs_rows: Vec<usize> = subalgebra
                                .signature_mask(
                                    &algebra,
                                    target,
                                    shifted_t,
                                    &zero_sig,
                                    target_bound,
                                )
                                .collect();
                            let cols =
                                MilnorSubalgebra::restricted_dimension(next, shifted_t, bound);
                            // Store in MASKED column coordinates. The mask is taken at the BUILD
                            // bound, which makes it the maximal zero-signature column set for this
                            // cache key: `signature_mask` with a larger bound extends the set at
                            // the end, so every consumer's own mask is a PREFIX of this one and
                            // reads columns `0..its_len`. That is what keeps one entry shared
                            // across the consumers that differ only in `next_bound`.
                            //
                            // Unlike the transient zero-signature site, this cache is resident by
                            // construction, so the ~50x narrowing converts to peak RSS rather than
                            // to churn.
                            let build_mask: Option<Vec<usize>> = masked_cols_enabled().then(|| {
                                subalgebra
                                    .signature_mask(&algebra, next, shifted_t, &zero_sig, bound)
                                    .collect()
                            });
                            let stored_cols = build_mask.as_ref().map_or(cols, Vec::len);
                            N_SHIFT_BUILD.fetch_add(1, Ordering::Relaxed);
                            ROWS_SHIFT_BUILD.fetch_add(zs_rows.len(), Ordering::Relaxed);
                            let built_bytes =
                                crate::census::matrix_bytes(zs_rows.len(), stored_cols);
                            if build_mask.is_some() {
                                crate::census::add_bytes(
                                    &crate::census::MASKED_SAVED_BYTES,
                                    crate::census::matrix_bytes(zs_rows.len(), cols)
                                        .saturating_sub(built_bytes),
                                );
                            }
                            if may_publish {
                                BYTES_SHIFT_PUBLISHED.fetch_add(built_bytes, Ordering::Relaxed);
                            } else {
                                N_SHIFT_PRIVATE.fetch_add(1, Ordering::Relaxed);
                                ROWS_SHIFT_PRIVATE.fetch_add(zs_rows.len(), Ordering::Relaxed);
                                BYTES_SHIFT_PRIVATE.fetch_add(built_bytes, Ordering::Relaxed);
                                _private_live = Some(PrivateLive::new(built_bytes));
                            }
                            let m = Arc::new(match &build_mask {
                                Some(mask) => restricted_partial_matrix_masked_maybe_gpu(
                                    &self.differentials[b.s() - 1],
                                    shifted_t,
                                    &zs_rows,
                                    cols,
                                    mask,
                                ),
                                None => restricted_partial_matrix_maybe_gpu(
                                    &self.differentials[b.s() - 1],
                                    shifted_t,
                                    &zs_rows,
                                    cols,
                                ),
                            });
                            if may_publish {
                                shift::put(
                                    b.s() as i32,
                                    shifted_t,
                                    &subalgebra.profile,
                                    Arc::clone(&m),
                                );
                            }
                            m
                        }
                    };
                    // `NASSAU_SHIFT_VERIFY=1`: the probe validated a FRESHLY built shifted matrix;
                    // this checks the CACHED one actually delivered to the solver, which is the
                    // only difference between the validated claim and this code path.
                    if std::env::var("NASSAU_SHIFT_VERIFY").as_deref() == Ok("1") {
                        // Content, not shape: does the shifted matrix agree with this signature's
                        // own, masked? This is the probe's check applied to what actually reaches
                        // the solver. `full_reuse` is still populated, so `full_matrix` here is
                        // the independently-built truth.
                        let truth = match &full_reuse {
                            Some(full) => PartialMatrix::Gather {
                                full,
                                rows: &target_mask,
                            },
                            None => PartialMatrix::Owned(restricted_partial_matrix_maybe_gpu(
                                &self.differentials[b.s() - 1],
                                b.t(),
                                &target_mask,
                                next_dim,
                            )),
                        };
                        let mut a = FpVector::new(p, next_mask.len());
                        let mut c = FpVector::new(p, zs_next.len());
                        for i in 0..target_mask.len().min(m.rows()) {
                            a.set_to_zero();
                            c.set_to_zero();
                            a.as_slice_mut().add_masked(truth.row(i), 1, &next_mask);
                            if masked_cols_enabled() {
                                c.as_slice_mut().add(m.row(i).restrict(0, zs_next.len()), 1);
                            } else {
                                c.as_slice_mut().add_masked(m.row(i), 1, &zs_next);
                            }
                            assert_eq!(
                                a, c,
                                "shift CONTENT row {i} at {b} sig={signature:?} \
                                 shifted_t={shifted_t}"
                            );
                        }
                    }
                    // A consumer whose `next_bound` is narrower than the builder's reads a prefix,
                    // never past the end. Assert it rather than trusting the prefix argument.
                    debug_assert!(
                        !masked_cols_enabled() || zs_next.len() <= m.columns(),
                        "shift mask {} exceeds cached masked width {} at {b} sig={signature:?} \
                         shifted_t={shifted_t}",
                        zs_next.len(),
                        m.columns()
                    );
                    let pre_masked = masked_cols_enabled();
                    shifted = Some((m, zs_next, pre_masked));
                }
            }

            let full_matrix = tracing::trace_span!("sig_select", rows = target_mask.len())
                .in_scope(|| {
                    // Under shift reuse the solver comes from the cache and the lift builds its
                    // rows on demand, so nothing wants this matrix -- do not spend a multiply on
                    // it. An empty placeholder keeps the binding's type without allocating.
                    if shift_skip_full && shifted.is_some() {
                        return PartialMatrix::Owned(Matrix::new(p, 0, 0));
                    }
                    match &full_reuse {
                        Some(full) => {
                            debug_assert!(target_mask.iter().all(|&r| r < full.rows()));
                            PartialMatrix::Gather {
                                full,
                                rows: &target_mask,
                            }
                        }
                        // Above `reuse_within_cap` there is no full matrix to slice, so this is where
                        // signature-axis speculation is collected.
                        None if blocks::enabled() => PartialMatrix::Owned(self.assemble_signature(
                            b,
                            sig_idx,
                            &target_mask,
                            next_dim,
                        )),
                        None => PartialMatrix::Owned(restricted_partial_matrix_maybe_gpu(
                            &self.differentials[b.s() - 1],
                            b.t(),
                            &target_mask,
                            next_dim,
                        )),
                    }
                });

            // The sufficient check: is this signature's masked matrix literally the zero-signature
            // masked matrix at the shifted degree? The two live over DIFFERENT column spaces
            // (degree `b.t()` vs `shifted_t`), so they are only comparable after masking — which
            // is exactly the claim, since both mask down to `next_mask.len()` columns.
            if shift_probe_matrices() {
                let shifted_t = b.t() - MilnorSubalgebra::signature_degree(&signature);
                if shifted_t >= 0 {
                    let zs_target: Vec<usize> = subalgebra
                        .signature_mask(&algebra, target, shifted_t, &zero_sig, target_bound)
                        .collect();
                    let zs_next: Vec<usize> = subalgebra
                        .signature_mask(&algebra, next, shifted_t, &zero_sig, next_bound)
                        .collect();
                    let zs_next_dim =
                        MilnorSubalgebra::restricted_dimension(next, shifted_t, next_bound);
                    let zs_full = restricted_partial_matrix_maybe_gpu(
                        &self.differentials[b.s() - 1],
                        shifted_t,
                        &zs_target,
                        zs_next_dim,
                    );
                    let mut ours = FpVector::new(p, next_mask.len());
                    let mut theirs = FpVector::new(p, zs_next.len());
                    let mut same = zs_target.len() == target_mask.len();
                    if same {
                        for i in 0..target_mask.len() {
                            ours.set_to_zero();
                            theirs.set_to_zero();
                            ours.as_slice_mut()
                                .add_masked(full_matrix.row(i), 1, &next_mask);
                            theirs
                                .as_slice_mut()
                                .add_masked(zs_full.row(i), 1, &zs_next);
                            if ours != theirs {
                                same = false;
                                break;
                            }
                        }
                    }
                    if same {
                        SHIFT_MAT_MATCH.fetch_add(1, Ordering::Relaxed);
                    } else {
                        SHIFT_MAT_MISMATCH.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }

            let mut masked_matrix = tracing::trace_span!(
                "sig_assemble",
                rows = target_mask.len(),
                cols = next_mask.len()
            )
            .in_scope(|| {
                let mut m = AugmentedMatrix::new(
                    p,
                    target_mask.len(),
                    [next_mask.len(), target_mask.len()],
                );
                // Row gather and column mask fused into one pass. `Matrix::add_masked` would need a
                // materialised `full_matrix`; going row by row lets the gather stay an indirection.
                match &shifted {
                    // Same matrix, reached from the shared build at the shifted degree — hence its
                    // own column mask over the shifted degree's column space.
                    Some((zs, zs_next, pre_masked)) => {
                        for (i, mut l) in m.segment(0, 0).iter_mut().enumerate() {
                            if *pre_masked {
                                // Already in masked coordinates; this consumer's mask names a
                                // prefix of the cached one, so take that many columns.
                                l.add(zs.row(i).restrict(0, zs_next.len()), 1);
                            } else {
                                l.add_masked(zs.row(i), 1, zs_next);
                            }
                        }
                    }
                    None => {
                        for (i, mut l) in m.segment(0, 0).iter_mut().enumerate() {
                            l.add_masked(full_matrix.row(i), 1, &next_mask);
                        }
                    }
                }
                // Per assembly, not per row -- see the note at the zs_assemble site. This one runs
                // once per SIGNATURE, and a bidegree at the frontier has ~1000 of them, so the
                // total here is what sizes the per-signature copy cost the census was meant to
                // report and has been printing as a misleading 0.0GB.
                crate::census::add_bytes(
                    &crate::census::ADD_MASKED_BYTES,
                    crate::census::matrix_bytes(target_mask.len(), next_mask.len()),
                );
                crate::census::add_bytes(
                    &crate::census::AUGMENTED_ALLOC_BYTES,
                    crate::census::matrix_bytes(
                        target_mask.len(),
                        next_mask.len() + target_mask.len(),
                    ),
                );
                m.segment(1, 1).add_identity();
                m
            });

            // The CPU row reduction, once per signature. `gpu_row_reduce` only takes over at
            // >= 8192^2, so every one of these is host work.
            tracing::trace_span!(
                "sig_row_reduce",
                rows = target_mask.len(),
                cols = next_mask.len()
            )
            .in_scope(|| masked_matrix.row_reduce());

            let qi = tracing::trace_span!("sig_quasi_inverse")
                .in_scope(|| masked_matrix.compute_quasi_inverse());
            let pivots = qi.pivots().unwrap();
            let preimage = qi.preimage();

            if let Some(snap) = &dx_snapshot {
                for (dx, dx0) in dxs.iter().zip(snap) {
                    for &v in &next_mask {
                        probe_reads += 1;
                        if dx.entry(v) != dx0.entry(v) {
                            probe_perturbed += 1;
                        }
                    }
                }
            }

            let _lift = tracing::trace_span!("sig_lift", gens = xs.len()).entered();
            if shift_skip_full && shifted.is_some() {
                // Two passes, because there is no full matrix to read rows from any more.
                //
                // Sound because generators are INDEPENDENT within a signature: each owns its `x`
                // and `dx`, and the only shared state (`pivots`, `preimage`) is read-only. Per
                // generator the original is read-then-write on its own `dx`, so hoisting every
                // read ahead of every write changes nothing. (The signature LOOP is a forward
                // substitution and must stay ordered -- that is a different axis, see above.)
                let mut supports: Vec<Vec<usize>> = Vec::with_capacity(xs.len());
                for (x, dx) in xs.iter_mut().zip(dxs.iter()) {
                    scratch.set_scratch_vector_size(target_mask.len());
                    let mut row = 0;
                    for (i, &v) in next_mask.iter().enumerate() {
                        if pivots[i] < 0 {
                            continue;
                        }
                        if dx.entry(v) != 0 {
                            scratch.as_slice_mut().add(preimage.row(row), 1);
                        }
                        row += 1;
                    }
                    let mut sup = Vec::new();
                    for (i, _) in scratch.iter_nonzero() {
                        x.add_basis_element(target_mask[i], 1);
                        sup.push(i);
                    }
                    supports.push(sup);
                }
                let mut needed: Vec<usize> = supports.iter().flatten().copied().collect();
                needed.sort_unstable();
                needed.dedup();
                if let Some(c) = census.as_mut() {
                    c.add_rows_consumed(supports.iter().map(Vec::len).sum());
                }
                if !needed.is_empty() {
                    // One multiply for the whole signature's demand, not one per generator.
                    let basis: Vec<usize> = needed.iter().map(|&i| target_mask[i]).collect();
                    N_ONDEMAND.fetch_add(1, Ordering::Relaxed);
                    ROWS_ONDEMAND.fetch_add(basis.len(), Ordering::Relaxed);
                    BYTES_ONDEMAND.fetch_add(
                        crate::census::matrix_bytes(basis.len(), next_dim),
                        Ordering::Relaxed,
                    );
                    let got =
                        tracing::trace_span!("sig_ondemand", rows = basis.len()).in_scope(|| {
                            restricted_partial_matrix_maybe_gpu(
                                &self.differentials[b.s() - 1],
                                b.t(),
                                &basis,
                                next_dim,
                            )
                        });
                    for (sup, dx) in supports.iter().zip(dxs.iter_mut()) {
                        for &i in sup {
                            let k = needed.binary_search(&i).unwrap();
                            dx.as_slice_mut().add(got.row(k), 1);
                        }
                    }
                }
            } else {
                for (x, dx) in xs.iter_mut().zip(&mut dxs) {
                    scratch.set_scratch_vector_size(target_mask.len());
                    let mut row = 0;
                    for (i, &v) in next_mask.iter().enumerate() {
                        if pivots[i] < 0 {
                            continue;
                        }
                        if dx.entry(v) != 0 {
                            scratch.as_slice_mut().add(preimage.row(row), 1);
                        }
                        row += 1;
                    }
                    let mut consumed = 0usize;
                    for (i, _) in scratch.iter_nonzero() {
                        x.add_basis_element(target_mask[i], 1);
                        dx.as_slice_mut().add(full_matrix.row(i), 1);
                        consumed += 1;
                    }
                    if let Some(c) = census.as_mut() {
                        c.add_rows_consumed(consumed);
                    }
                }
            }
            drop(_lift);
            tracing::trace_span!("sig_write_qi").in_scope(|| {
                Self::write_qi(
                    &mut f,
                    &mut scratch,
                    &signature,
                    &next_mask,
                    &full_matrix,
                    &masked_matrix,
                )
            })?;
        }
        if dx_snapshot.is_some() {
            eprintln!(
                "[sig-probe] b={b} signatures_read_positions={probe_reads} \
                 perturbed_by_earlier_signature={probe_perturbed}"
            );
        }

        for dx in &dxs {
            assert!(dx.is_zero(), "dx non-zero at {b}");
        }
        self.differential(b.s()).add_generators_from_rows(b.t(), xs);

        end();

        if let Some(c) = census {
            c.finish();
        }

        if let Some(w) = f {
            w.finish()?;
        }

        // Anything the signature loop did not collect (an unrepresentable signature, a piece that
        // landed late) would otherwise stay pinned for the rest of the run.
        if blocks::enabled() {
            blocks::release_sig(b);
        }

        self.write_differential(b, num_new_gens, target_dim)?;
        Ok(())
    }

    /// Step resolution for s = 0
    #[tracing::instrument(skip(self))]
    fn step0(&self, t: i32) {
        self.zero_module.extend_by_zero(t);

        let source_module = &self.modules[0];
        let target_module = self.target.module(0);

        let chain_map = &self.chain_maps[0];
        let d = &self.differentials[0];

        let source_dim = source_module.dimension(t);
        let target_dim = target_module.dimension(t);

        source_module.compute_basis(t);
        target_module.compute_basis(t);

        if target_dim == 0 {
            source_module.extend_by_zero(t);
            chain_map.extend_by_zero(t);
        } else {
            let mut matrix = AugmentedMatrix::<2>::new_with_capacity(
                self.prime(),
                source_dim,
                &[target_dim, source_dim],
                source_dim + target_dim,
                0,
            );
            chain_map.get_matrix(matrix.segment(0, 0), t);
            matrix.segment(1, 1).add_identity();

            matrix.row_reduce();

            let num_new_gens = matrix.extend_to_surjection(0, target_dim, 0).len();

            self.add_generators(Bidegree::s_t(0, t), num_new_gens);

            chain_map.add_generators_from_matrix_rows(
                t,
                matrix
                    .segment(0, 0)
                    .row_slice(source_dim, source_dim + num_new_gens),
            );
        }
        chain_map.compute_auxiliary_data_through_degree(t);

        d.set_kernel(t, None);
        d.set_image(t, None);
        d.set_quasi_inverse(t, None);
        d.extend_by_zero(t);
    }

    /// Step resolution for s = 1
    #[tracing::instrument(skip(self))]
    fn step1(&self, t: i32) -> anyhow::Result<()> {
        let p = self.prime();

        let source_module = &self.modules[1];
        let target_module = &self.modules[0];
        let cc_module = self.target.module(0);

        let source_dim = source_module.dimension(t);
        let target_dim = target_module.dimension(t);

        // The desired image is the kernel of the augmentation `target_module -> cc_module` in this
        // degree. Whenever the target complex is empty in degree `t` that map has a zero-dimensional
        // codomain, so its kernel is the whole space and no computation can discover otherwise.
        // Taking it directly skips building and row-reducing a `target_dim x target_dim` augmented
        // identity, where `target_dim` is `dim(A_t)` -- large enough at high `t` to reach the GPU
        // RREF path.
        //
        // The guard is on the codomain being empty, not on which module is being resolved, so it is
        // not a sphere special case: every finite target module is concentrated in finitely many
        // degrees, so past its top cell this holds for all `t`, which is almost the whole
        // resolution. The sphere is only the extreme of it, firing from `t = 1`.
        //
        // When the codomain is NON-empty the reduction is still needed, but note it is wasteful
        // there too: the kernel has codimension at most `cc_module.dimension(t)`, typically a
        // handful, yet we materialise a full `(target_dim - c) x target_dim` basis for it.
        let desired_image = if cc_module.dimension(t) == 0 {
            Subspace::entire_space(p, target_dim)
        } else {
            let mut matrix =
                AugmentedMatrix::<2>::new(p, target_dim, [cc_module.dimension(t), target_dim]);
            self.chain_maps[0].get_matrix(matrix.segment(0, 0), t);
            matrix.segment(1, 1).add_identity();
            matrix.row_reduce();
            matrix.compute_kernel()
        };

        let mut matrix = AugmentedMatrix::<2>::new_with_capacity(
            p,
            source_dim,
            &[target_dim, source_dim],
            source_dim + MAX_NEW_GENS,
            0,
        );
        self.differentials[1].get_matrix(matrix.segment(0, 0), t);
        matrix.segment(1, 1).add_identity();
        matrix.row_reduce();

        let num_new_gens = matrix.extend_image(0, target_dim, &desired_image, 0).len();

        self.add_generators(Bidegree::s_t(1, t), num_new_gens);

        self.differentials[1].add_generators_from_matrix_rows(
            t,
            matrix
                .segment(0, 0)
                .row_slice(source_dim, source_dim + num_new_gens),
        );

        self.write_differential(Bidegree::s_t(1, t), num_new_gens, target_dim)?;
        Ok(())
    }

    fn step_resolution_with_result(&self, b: Bidegree) -> anyhow::Result<()> {
        let p = self.prime();
        let set_data = || {
            let d = &self.differentials[b.s()];
            let c = &self.chain_maps[b.s()];

            d.set_kernel(b.t(), None);
            d.set_image(b.t(), None);
            d.set_quasi_inverse(b.t(), None);

            c.set_kernel(b.t(), None);
            c.set_image(b.t(), None);
            c.set_quasi_inverse(b.t(), None);
        };
        self.modules[b.s()].compute_basis(b.t());
        if b.s() > 0 {
            self.modules[b.s() - 1].compute_basis(b.t());
        }

        if b.s() == 0 {
            self.step0(b.t());
            return Ok(());
        }

        if let Some(store) = self.save_dir.store()
            && let Some(data) = store.read(SaveKind::NassauDifferential, b)?
        {
            tracing::info!(%b, "Loading differential");

            let mut f = std::io::Cursor::new(data);
            let num_new_gens = f.read_u64::<LittleEndian>()? as usize;
            // This need not be equal to `target_res_dimension`. If we saved a big resolution
            // and now only want to load up to a small stem, then `target_res_dimension` will
            // be smaller. If we have previously saved a small resolution up to a stem and now
            // want to resolve further, it will be bigger.
            let saved_target_res_dimension = f.read_u64::<LittleEndian>()? as usize;

            self.add_generators(b, num_new_gens);

            let mut d_targets = Vec::with_capacity(num_new_gens);

            for _ in 0..num_new_gens {
                d_targets.push(FpVector::from_bytes(p, saved_target_res_dimension, &mut f)?);
            }

            self.differentials[b.s()].add_generators_from_rows(b.t(), d_targets);

            set_data();

            return Ok(());
        }

        if b.s() == 1 {
            self.step1(b.t())?;
            set_data();
            return Ok(());
        }

        self.step_resolution_with_subalgebra(
            b,
            MilnorSubalgebra::optimal_for(b - Bidegree::s_t(0, self.max_degree)),
        )?;
        self.chain_maps[b.s()].extend_by_zero(b.t());

        set_data();
        Ok(())
    }

    fn step_resolution(&self, b: Bidegree) {
        // One guard for the whole bidegree, rather than one per inner parallel section.
        //
        // This is correct by construction rather than by audit. A `step_resolution` job can only be
        // stolen onto a thread that is in rayon's steal loop, i.e. blocked at a join — and running
        // it there is exactly the priority inversion. So a bounce never discards useful work: it
        // declines precisely the runs that would invert. Holding the guard for the whole bidegree
        // therefore costs nothing and removes the need to know which callee happens to enter rayon
        // today, which the narrow per-section guards did depend on.
        //
        // Nesting is free: [`ParallelGuard`] counts depth, so the inner guards become depth 1+ and
        // keep their spans.
        //
        // Measurement note, for whoever revisits this: a single stem-200 A/B showed the worst step
        // improving (224 s -> 87 s) but the count of steps >=20 s rising (41 -> 67) and retries
        // rising (1363 -> 1613). That comparison is NOT conclusive — two runs differing in nothing
        // relevant to guarding moved 31 -> 41 and 1271 s -> 1969 s, so the noise floor is the same
        // size as the effect. Do not "fix" this on one run's numbers.
        let _guard = ParallelGuard::new();
        self.step_resolution_with_result(b)
            .unwrap_or_else(|e| panic!("Error computing bidegree {b}: {e}"));
    }

    /// This function resolves up till a fixed stem instead of a fixed t.
    ///
    /// The dependency graph we use is the relaxed one: computing `(s, t)` only requires `(s, t - 1)`
    /// and `(s - 1, t - 1)` (for `s >= 2`), rather than `(s - 1, t)` and `(s, t - 1)`. The read-only
    /// data `(s, t)` needs — the generators of the relevant modules of degree `< t` — is frozen once
    /// those two bidegrees have been committed (`step_resolution_with_subalgebra` ignores the
    /// degree-`t` generators of the target). This lets `(s, t)` run concurrently with
    /// `(s - 1, t)`, the bidegree that produces those degree-`t` generators, and keeps many
    /// `t`-diagonals (`n = t - s` fixed) in flight at once, which is where the parallelism comes
    /// from.
    ///
    /// The rows `s = 0` and `s = 1` are kept on the strict schedule: they are cheap, and `step0` and
    /// `step1` read their targets through full matrices, so they wait for `(s - 1, t)`.
    #[tracing::instrument(skip(self), fields(self = self.name, %max))]
    pub fn compute_through_stem(&self, max: Bidegree) {
        let _lock = self.lock.lock();

        self.extend_through_degree(max.s());
        self.algebra().compute_basis(max.t());

        let min_degree = self.min_degree();
        let max_s = max.s();
        let max_n = max.n();

        let in_region = |s: i32, t: i32| -> bool {
            (0..=max_s).contains(&s) && t >= min_degree && t - s <= max_n
        };

        // `(s, t)` may be computed once its same-row predecessor `(s, t - 1)` and its diagonal
        // predecessor are committed. For `s >= 2` the diagonal predecessor is `(s - 1, t - 1)` (the
        // relaxed graph); for `s == 1` it is `(0, t)`; `s == 0` has none. `progress[s]` is the
        // largest committed `t` in row `s`, so it doubles as a "predecessor committed" test.
        let ready = |s: i32, t: i32, progress: &[i32]| -> bool {
            in_region(s, t)
                && progress[s as usize] >= t - 1
                && match s {
                    0 => true,
                    // Row 1's diagonal predecessor is (0, t). At the stem edge (t = max_n + 1) that
                    // bidegree lies outside the computed region, so we treat it as satisfied.
                    1 => t > max_n || progress[0] >= t,
                    _ => progress[(s - 1) as usize] >= t - 1,
                }
        };

        let tracing_span = tracing::Span::current();
        let spec_threads = speculate::threads();
        speculate::open();
        blocks::open();

        // Speculative builders run *outside* the rayon pool on purpose. They are a background task
        // whose whole point is to use time the wavefront is not using; putting them in the pool
        // would let them take worker slots from the critical path, which is the opposite of the
        // intent. `std::thread::scope` keeps them borrowing `self` without any `Arc` plumbing.
        std::thread::scope(|spec_scope| {
            if spec_threads > 0 && blocks::enabled() {
                // Samples on a timer, so the depth statistic is time-weighted rather than
                // conditioned on there being work to pop.
                spec_scope.spawn(|| {
                    let t0 = std::time::Instant::now();
                    let mut ticks = 0u64;
                    while !blocks::closed() {
                        blocks::sample_depth();
                        ticks += 1;
                        // 60 Hz. The sampler reads two atomics, so this is free -- the earlier
                        // 10 Hz limit existed only because it took the queue lock. Resolution
                        // matters: the burst that showed speculation dying lasted under 2 s.
                        if ticks % 30 == 0 {
                            blocks::trace_depth(t0.elapsed().as_secs_f64());
                        }
                        std::thread::sleep(std::time::Duration::from_micros(16667));
                    }
                });
            }
            for _ in 0..spec_threads {
                let tracing_span = tracing_span.clone();
                spec_scope.spawn(move || {
                    let _tracing_guard = tracing_span.enter();
                    if blocks::enabled() {
                        while let Some((b, gen_deg)) = blocks::pop() {
                            self.speculate_block(b, gen_deg);
                        }
                    } else {
                        while let Some(b) = speculate::pop() {
                            self.speculate_build(b);
                        }
                    }
                });
            }

            maybe_rayon::in_place_scope(|scope| {
                let _tracing_guard = tracing_span.enter();

                let mut progress: Vec<i32> = vec![min_degree - 1; max_s as usize + 1];

                let (sender, receiver) = mpsc::channel();

                let spawn_bidegree = |b: Bidegree, sender: mpsc::Sender<SenderData>| {
                    if self.has_computed_bidegree(b) {
                        SenderData::send(b, sender);
                    } else {
                        let tracing_span = tracing_span.clone();
                        scope.spawn(move |_| {
                            let _tracing_guard = tracing_span.enter();
                            if crate::utils::parallel::is_in_parallel() {
                                SenderData::send_retry(b, sender);
                                return;
                            }
                            let n = INFLIGHT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            INFLIGHT_MAX.fetch_max(n, std::sync::atomic::Ordering::Relaxed);
                            self.step_resolution(b);
                            INFLIGHT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                            SenderData::send(b, sender);
                        });
                    }
                };

                // Seed the base of every row. A bidegree `(s, min_degree)` has no in-region
                // predecessors, so it is not spawned by the wavefront — except `(1, min_degree)`, whose
                // diagonal predecessor `(0, min_degree)` is in region, so we let it be spawned instead.
                for s in 0..=max_s {
                    if s != 1 {
                        spawn_bidegree(Bidegree::s_t(s, min_degree), sender.clone());
                    }
                }
                drop(sender);

                // Bidegrees whose spawned job was stolen onto a worker already inside a critical section
                // (`is_in_parallel` set on that worker) and so bounced back a retry rather than causing
                // a priority inversion. Because the check is per-thread, a job is only ever bounced when
                // its worker is a blocked guard holder; a job picked up by a free worker just runs. Such
                // bounces are therefore rare, but when the pool is momentarily saturated we still must
                // avoid re-spawning immediately in a tight loop, so we park bounced bidegrees here.
                //
                // The scheduler thread never holds a guard, so it cannot itself observe when a worker
                // frees; instead, while anything is parked we wait on the channel with a short timeout
                // and retry the parked work whenever a completion arrives (a worker likely just freed)
                // or the timeout elapses (periodic re-check). Incoming messages are still handled the
                // instant they arrive; the timeout only governs how promptly we retry while otherwise
                // idle. This cannot deadlock: parked entries keep their senders, so the channel stays
                // open, and the timeout guarantees parked work is retried until a free worker takes it.
                let mut deferred: Vec<(Bidegree, mpsc::Sender<SenderData>)> = Vec::new();
                // Diagnostic (`NASSAU_MEM_REPORT`): count committed bidegrees so we can periodically
                // report the retained-data heap split (differentials' `outputs` vs modules' tables).
                //
                // The value is the report INTERVAL in commits (bare `=1`, or any non-numeric value,
                // keeps the historical 400). Commit rate falls steeply with stem -- at stem 250 a
                // long run commits ~180 bidegrees per 6h, so a fixed 400 puts ~13 hours between
                // heap attributions, which is far too coarse to watch memory grow. The report walks
                // every differential and module to sum heap bytes, so it is O(bidegrees) per
                // emission and wants to stay well above once-per-commit.
                let mem_report_every = std::env::var("NASSAU_MEM_REPORT")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&n| n > 1)
                    .unwrap_or(400);
                let mem_report = std::env::var_os("NASSAU_MEM_REPORT").is_some();
                let mut commit_count = 0usize;
                // How long to wait for a message before retrying parked bidegrees. Small enough that a
                // freed worker is used promptly, large enough that the poll is negligible; it only ticks
                // while something is parked.
                const RETRY_POLL_INTERVAL: std::time::Duration =
                    std::time::Duration::from_micros(100);

                // Largest `t` already queued for speculation in each row, so a commit only enqueues the
                // window it newly opened.
                let mut spec_issued: Vec<i32> = vec![min_degree - 1; max_s as usize + 1];
                // Block mode issues over a RECTANGLE per row -- internal degrees `t` crossed with
                // generator degrees `gen_deg` -- so it needs the high-water mark of both axes to
                // enqueue only what a commit newly opened. Both grow monotonically, so the new work
                // is always two strips: the `t`s just opened (at every available `gen_deg`), and the
                // `gen_deg`s just opened (at every `t` already in the window).
                let mut spec_issued_g: Vec<i32> = vec![min_degree - 1; max_s as usize + 1];
                // Enqueue every bidegree whose matrix is now determined but which cannot run yet.
                //
                // `(r, t)`'s matrix needs only `(r - 1, t - 1)` — i.e. `t <= progress[r - 1] + 1` —
                // while running `(r, t)` additionally needs `(r, t - 1)`. Everything strictly between
                // those two bounds is a bidegree we can build for but not yet run: exactly the
                // speculation window. We start at `progress[r] + 2` because `progress[r] + 1` is the
                // bidegree the scheduler is spawning right now, whose matrix a builder would only race.
                //
                // In BLOCK mode the first bound disappears: block `gen_deg` of `(r, t)` needs only
                // `gen_deg <= min(progress[r - 1], progress[r - 2])`, for every `t > gen_deg` still
                // in region. So the `t` window is capped only by how far ahead we are willing to
                // speculate, and one commit opens blocks across a whole column of future bidegrees
                // rather than a single matrix — which is the point, since the queue depth is what
                // the GPU is starved of.
                let enqueue_spec =
                    |progress: &[i32], spec_issued: &mut Vec<i32>, spec_issued_g: &mut Vec<i32>| {
                        if spec_threads == 0 {
                            return;
                        }
                        if blocks::enabled() {
                            let algebra = self.algebra();
                            for r in 2..=max_s {
                                let ri = r as usize;
                                // Generator degrees whose rows AND columns are both frozen.
                                let gf = std::cmp::min(progress[ri - 1], progress[ri - 2]);
                                let hi_t = progress[ri] + 1 + blocks::ahead();
                                let lo_t = progress[ri] + 2;
                                let emit = |t: i32, lo_g: i32, hi_g: i32| {
                                    if !in_region(r, t) {
                                        return;
                                    }
                                    // `gen_deg < t`: a generator of degree `t` contributes no rows to
                                    // the restricted matrix at `t`.
                                    // Row count is `num_gens[g] x dim(A_{t-g})`: both O(1) lookups, so
                                    // sizing every candidate costs nothing and lets the queue order by
                                    // it.
                                    let source = &self.modules[r - 1];
                                    for g in lo_g..=std::cmp::min(hi_g, t - 1) {
                                        let rows = source.number_of_gens_in_degree(g)
                                            * algebra.ppart_table(t - g).len();
                                        blocks::push(Bidegree::s_t(r, t), g, rows);
                                    }
                                };
                                // Strip A: internal degrees newly in the window, at every `gen_deg`.
                                for t in std::cmp::max(lo_t, spec_issued[ri] + 1)..=hi_t {
                                    emit(t, min_degree, gf);
                                }
                                // Strip B: generator degrees newly frozen, at internal degrees already
                                // issued.
                                for t in lo_t..=std::cmp::min(spec_issued[ri], hi_t) {
                                    emit(t, spec_issued_g[ri] + 1, gf);
                                }
                                spec_issued[ri] = std::cmp::max(spec_issued[ri], hi_t);
                                spec_issued_g[ri] = std::cmp::max(spec_issued_g[ri], gf);
                            }
                            return;
                        }
                        for r in 2..=max_s {
                            let ri = r as usize;
                            let hi = std::cmp::min(
                                progress[ri - 1] + 1,
                                progress[ri] + 1 + speculate::ahead(),
                            );
                            let lo = std::cmp::max(spec_issued[ri] + 1, progress[ri] + 2);
                            for t in lo..=hi {
                                if in_region(r, t) {
                                    speculate::push(Bidegree::s_t(r, t));
                                }
                            }
                            spec_issued[ri] = std::cmp::max(spec_issued[ri], hi);
                        }
                    };

                loop {
                    let event = if deferred.is_empty() {
                        // Nothing parked: block until a message arrives or all senders drop.
                        match receiver.recv() {
                            Ok(data) => Some(data),
                            Err(_) => break,
                        }
                    } else {
                        // Something parked: wake periodically to retry it. Parked entries hold senders,
                        // so the channel cannot be disconnected here.
                        match receiver.recv_timeout(RETRY_POLL_INTERVAL) {
                            Ok(data) => Some(data),
                            Err(mpsc::RecvTimeoutError::Timeout) => None,
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    };

                    if let Some(SenderData { b, retry, sender }) = event {
                        if retry {
                            // Park until a worker frees; retried below on a completion or timeout.
                            deferred.push((b, sender));
                            continue;
                        }
                        assert!(progress[b.s() as usize] == b.t() - 1);
                        INFLIGHT_SUM.fetch_add(
                            INFLIGHT.load(std::sync::atomic::Ordering::Relaxed) as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        INFLIGHT_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        progress[b.s() as usize] = b.t();
                        enqueue_spec(&progress, &mut spec_issued, &mut spec_issued_g);

                        if mem_report {
                            commit_count += 1;
                            if commit_count % mem_report_every == 0 {
                                let diff_b: usize = self
                                    .differentials
                                    .iter()
                                    .map(|(_, d)| d.output_heap_bytes())
                                    .sum();
                                let mod_b: usize =
                                    self.modules.iter().map(|(_, m)| m.table_heap_bytes()).sum();
                                #[cfg(feature = "gpu")]
                                let (res_master, res_basis) =
                                    algebra::milnor_gpu::resident_host_bytes();
                                #[cfg(not(feature = "gpu"))]
                                let (res_master, res_basis) = (0usize, 0usize);
                                #[cfg(feature = "gpu")]
                                let (dev_master, dev_basis) =
                                    algebra::milnor_gpu::resident_dev_bytes();
                                #[cfg(feature = "gpu")]
                                let (dev_pool_use, dev_pool_res) =
                                    algebra::milnor_gpu::cubecl_device_usage();
                                #[cfg(not(feature = "gpu"))]
                                let ((dev_master, dev_basis), (dev_pool_use, dev_pool_res)) =
                                    ((0usize, 0usize), (0u64, 0u64));
                                let gb = |x: usize| x as f64 / (1u64 << 30) as f64;
                                let gbu = |x: u64| x as f64 / (1u64 << 30) as f64;
                                // Cumulative copy volume, carried on the [MEM] line rather than
                                // left to the end-of-run [census] block. Anything printed only at
                                // clean exit is lost to an OOM SIGKILL -- the failure mode this
                                // run is instrumented for -- which is the same reason the census
                                // CSV now flushes periodically.
                                let copy_masked = crate::census::ADD_MASKED_BYTES
                                    .load(std::sync::atomic::Ordering::Relaxed);
                                let copy_aug = crate::census::AUGMENTED_ALLOC_BYTES
                                    .load(std::sync::atomic::Ordering::Relaxed);
                                // PROCESS RSS, not just our own accounting. Job 40131303 was
                                // killed OUT_OF_MEMORY at 509GB of a 500G limit while the HOST[]
                                // fields below summed to 5.8GB -- about 1% of the process. The rest
                                // is GPU-runtime host memory (per-stream pinned pools, the resident
                                // host-master duplicate) that nothing here measures, so the run that
                                // died of it produced a postmortem instead of a trajectory.
                                // `proc_rss_bytes` already existed and was dead code; the build had
                                // been warning `never used` the whole time.
                                let rss = proc_rss_bytes();
                                eprintln!(
                                    "[MEM] commits={commit_count} last_b=({},{}) RSS={:.1}GB \
                                     HOST[diff={:.1} \
                                     mod={:.1} res_master={:.1} res_basis={:.1}]GB \
                                     DEV[master={:.1} basis={:.1} cubecl_use={:.1} \
                                     cubecl_reserved={:.1}]GB COPIED[add_masked={:.1} \
                                     augmented={:.1}]GB MASKED_SAVED={:.1}GB",
                                    b.n(),
                                    b.s(),
                                    gbu(rss),
                                    gb(diff_b),
                                    gb(mod_b),
                                    gb(res_master),
                                    gb(res_basis),
                                    gb(dev_master),
                                    gb(dev_basis),
                                    gbu(dev_pool_use),
                                    gbu(dev_pool_res),
                                    gbu(copy_masked),
                                    gbu(copy_aug),
                                    gbu(crate::census::MASKED_SAVED_BYTES
                                        .load(std::sync::atomic::Ordering::Relaxed)),
                                );
                                // Emit here too, not only at stage end: an OOM kill means no
                                // clean exit, and this is the run that most needs the numbers.
                                shift_stats_report();
                                heap_stats_report();
                            }
                        }

                        // Completing `b` can only make ready its same-row successor `(s, t + 1)` and one
                        // diagonal successor. `ready` requires *both* predecessors, so of the two
                        // completions that could spawn a given bidegree, only the later one does.
                        let same_row = b + Bidegree::s_t(0, 1);
                        let diagonal = if b.s() == 0 {
                            Bidegree::s_t(1, b.t())
                        } else {
                            b + Bidegree::s_t(1, 1)
                        };

                        for cand in [same_row, diagonal] {
                            if ready(cand.s(), cand.t(), &progress) {
                                spawn_bidegree(cand, sender.clone());
                            }
                        }
                    }

                    // Retry parked bidegrees — reached after a completion (a worker likely just freed)
                    // or a timeout (periodic re-check), but not after a retry (which `continue`s above,
                    // so a bounced job waits out the timeout before being retried). Each re-spawned job
                    // re-checks its own worker's flag: those on a free worker run, those stolen onto a
                    // blocked guard holder bounce and are re-parked. This stays cheap because per-thread
                    // bounces are rare, so `deferred` is normally empty.
                    if !deferred.is_empty() {
                        for (b, sender) in std::mem::take(&mut deferred) {
                            spawn_bidegree(b, sender);
                        }
                    }
                }
            });

            // The wavefront is done, so no further matrix can be wanted: release the builders.
            speculate::close();
            blocks::close();
        });

        if spec_threads > 0 && blocks::enabled() {
            let (built, hits, misses, rows_hit, rows_missed, dropped, released, bytes) =
                blocks::stats();
            // `row_rate` is the figure of merit, not `hit_rate`: a hit on a one-row block removes
            // almost nothing from the consumer's launch.
            eprintln!(
                "[blocks] threads={spec_threads} built={built} hits={hits} misses={misses} \
                 hit_rate={:.1}% rows_hit={rows_hit} rows_missed={rows_missed} row_rate={:.1}% \
                 dropped={dropped} released={released} unused={:.1}GB \
                 bails(room/done/claimed/other)={}/{}/{}/{}",
                100.0 * hits as f64 / (hits + misses).max(1) as f64,
                100.0 * rows_hit as f64 / (rows_hit + rows_missed).max(1) as f64,
                bytes as f64 / (1u64 << 30) as f64,
                blocks::bails().0,
                blocks::bails().1,
                blocks::bails().2,
                blocks::bails().3,
            );
        } else if spec_threads > 0 {
            let (built, hits, misses, waited, timeouts, dropped, bytes) = speculate::stats();
            eprintln!(
                "[speculate] threads={spec_threads} built={built} hits={hits} misses={misses} \
                 waited={waited} timeouts={timeouts} dropped={dropped} hit_rate={:.1}% wasted={} \
                 unused={:.1}GB",
                100.0 * hits as f64 / (hits + misses).max(1) as f64,
                built.saturating_sub(hits) + dropped,
                bytes as f64 / (1u64 << 30) as f64,
            );
        }
        {
            let (qd, qmax, idle_s, build_s) = blocks::queue_stats();
            let (twd, twe_frac) = blocks::timed_depth();
            let twe = twe_frac * 100.0;
            if blocks::enabled() {
                eprintln!(
                    "[specqueue] depth at-pop mean={qd:.1} max={qmax} | TIME-WEIGHTED \
                     mean={twd:.1} empty={twe:.1}% | builder_idle={idle_s:.0}s \
                     builder_build={build_s:.0}s threads={spec_threads}"
                );
            }
            shift_probe_report();
            shift_stats_report();
            let n = INFLIGHT_N.load(std::sync::atomic::Ordering::Relaxed);
            if n > 0 {
                eprintln!(
                    "[wavefront] in-flight bidegrees: mean={:.1} max={} (samples={n}, cores={})",
                    INFLIGHT_SUM.load(std::sync::atomic::Ordering::Relaxed) as f64 / n as f64,
                    INFLIGHT_MAX.load(std::sync::atomic::Ordering::Relaxed),
                    maybe_rayon::max_num_threads(),
                );
            }
        }
        {
            let (rw, rc) = (
                REUSE_WORK.load(std::sync::atomic::Ordering::Relaxed),
                REUSE_COUNT.load(std::sync::atomic::Ordering::Relaxed),
            );
            let (nw, nc) = (
                NOREUSE_WORK.load(std::sync::atomic::Ordering::Relaxed),
                NOREUSE_COUNT.load(std::sync::atomic::Ordering::Relaxed),
            );
            if rw + nw > 0 {
                eprintln!(
                    "[reuse-split] reusable={rc} bidegrees {:.1}Gwork ({:.1}%) | over-cap={nc} \
                     bidegrees {:.1}Gwork ({:.1}%) | narrow={:.1}Gwork (ratio {:.3})",
                    rw as f64 / 1e9,
                    100.0 * rw as f64 / (rw + nw) as f64,
                    nw as f64 / 1e9,
                    100.0 * nw as f64 / (rw + nw) as f64,
                    NARROW_WORK.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
                    NARROW_WORK.load(std::sync::atomic::Ordering::Relaxed) as f64
                        / (rw + nw).max(1) as f64,
                );
            }
        }
        blocks::dump_opdeg();
        crate::census::report();
        speculate::clear();
        blocks::clear();
        #[cfg(feature = "gpu")]
        {
            let (n, bytes) = shift::stats();
            eprintln!(
                "[shift-cache] entries={n} approx={:.2}GB (never evicted during the run)",
                bytes as f64 / 1e9
            );
            shift::clear();
        }

        // Eviction probe (`NASSAU_R_STATS`): dump the R-access distribution once the wavefront is done.
        #[cfg(feature = "gpu")]
        {
            algebra::milnor_gpu::dump_census();
            algebra::milnor_gpu::dump_r_stats();
            // Which theta would have fit: see `resident_degree_cap`.
            algebra::milnor_gpu::dump_master_by_degree();
        }
    }
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> ChainComplex for Resolution<M> {
    type Algebra = MilnorAlgebra;
    type Homomorphism = FreeModuleHomomorphism<FreeModule<Self::Algebra>>;
    type Module = FreeModule<Self::Algebra>;

    fn prime(&self) -> ValidPrime {
        TWO
    }

    fn algebra(&self) -> Arc<Self::Algebra> {
        self.zero_module.algebra()
    }

    fn module(&self, s: i32) -> Arc<Self::Module> {
        Arc::clone(&self.modules[s])
    }

    fn zero_module(&self) -> Arc<Self::Module> {
        Arc::clone(&self.zero_module)
    }

    fn min_degree(&self) -> i32 {
        0
    }

    fn has_computed_bidegree(&self, b: Bidegree) -> bool {
        self.differentials.len() > b.s() && self.differential(b.s()).next_degree() > b.t()
    }

    fn differential(&self, s: i32) -> Arc<Self::Homomorphism> {
        Arc::clone(&self.differentials[s])
    }

    #[tracing::instrument(skip(self), fields(self = self.name, %max))]
    fn compute_through_bidegree(&self, max: Bidegree) {
        let _lock = self.lock.lock();

        self.extend_through_degree(max.s());
        self.algebra().compute_basis(max.t());

        for t in 0..=max.t() {
            for s in 0..=max.s() {
                let b = Bidegree::s_t(s, t);
                if self.has_computed_bidegree(b) {
                    continue;
                }
                self.step_resolution(b);
            }
        }
    }

    fn next_homological_degree(&self) -> i32 {
        self.modules.len()
    }

    fn save_dir(&self) -> &SaveDirectory {
        &self.save_dir
    }

    fn apply_quasi_inverse<T, S>(&self, results: &mut [T], b: Bidegree, inputs: &[S]) -> bool
    where
        for<'a> &'a mut T: Into<FpSliceMut<'a>>,
        for<'a> &'a S: Into<FpSlice<'a>>,
    {
        self.apply_quasi_inverse_fallible(results, b, inputs)
            .unwrap_or_else(|e| panic!("apply_quasi_inverse failed at {b}: {e:#}"))
    }
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> Resolution<M> {
    /// The fallible core of [`ChainComplex::apply_quasi_inverse`].
    ///
    /// The quasi-inverse commands are read from the saved zarr stream unless recomputation is
    /// forced (`EXT_NASSAU_RECOMPUTE_QI`). When no saved stream is available — no store, the qis
    /// were never persisted (`EXT_NASSAU_NO_SAVE_QI`), or the store simply has no qi for this
    /// bidegree — the commands are regenerated on the fly from `differentials[b.s]` by
    /// [`RecomputeReader`]. The missing-qi case legitimately happens at the top of the computed
    /// region: nassau writes qi(s, t) while computing (s + 1, t), so qi(max_s, t) is never saved
    /// even though lifting into it is well-defined.
    ///
    /// Both sources yield the same [`NassauCommand`] stream, so the application loop below is
    /// identical either way. Returns `Ok(true)` once applied (the recompute fallback means we
    /// never fail to produce a quasi-inverse), and propagates `anyhow::Error` from the reader or
    /// `FpVector::update_from_bytes`.
    fn apply_quasi_inverse_fallible<T, S>(
        &self,
        results: &mut [T],
        b: Bidegree,
        inputs: &[S],
    ) -> anyhow::Result<bool>
    where
        for<'a> &'a mut T: Into<FpSliceMut<'a>>,
        for<'a> &'a S: Into<FpSlice<'a>>,
    {
        let p = self.prime();

        // Prefer the saved stream unless recomputation is forced.
        let saved = if *RECOMPUTE_QI {
            None
        } else if let Some(store) = self.save_dir.store() {
            store.nassau_qi_reader(b)?
        } else {
            None
        };

        type Commands<'a> = Box<dyn Iterator<Item = anyhow::Result<NassauCommand>> + 'a>;
        let (target_dim, zero_mask_dim, subalgebra, commands): (
            usize,
            usize,
            MilnorSubalgebra,
            Commands,
        ) = match saved {
            Some(reader) => {
                let target_dim = reader.target_dim() as usize;
                let zero_mask_dim = reader.zero_mask_dim() as usize;
                let subalgebra = MilnorSubalgebra::new(reader.subalgebra_profile().to_vec());
                (target_dim, zero_mask_dim, subalgebra, Box::new(reader))
            }
            None => {
                // Regenerate the quasi-inverse from the (fully computed) differential.
                let recompute = RecomputeReader::new(self, b);
                let target_dim = recompute.target_dim;
                let zero_mask_dim = recompute.zero_mask_dim;
                let subalgebra = recompute.subalgebra.clone();
                (target_dim, zero_mask_dim, subalgebra, Box::new(recompute))
            }
        };

        let source = &self.modules[b.s()];
        let target = &self.modules[b.s() - 1];
        let algebra = target.algebra();

        let mut inputs: Vec<FpVector> = inputs.iter().map(|x| x.into().to_owned()).collect();
        let mut mask: Vec<usize> = Vec::with_capacity(zero_mask_dim + 8);
        mask.extend(subalgebra.signature_mask(
            &algebra,
            source,
            b.t(),
            &subalgebra.zero_signature(),
            i32::MAX,
        ));

        let mut scratch0 = FpVector::new(p, zero_mask_dim);
        let mut scratch1 = FpVector::new(p, target_dim);

        // If the quasi-inverse was computed using incomplete information, we need to figure
        // out what the differentials in this bidegree hit and use them to lift. these
        // variables are trivial if there is no such problem.
        //
        // target_zero_mask is the signature mask of the target under the zero signature.
        //
        // dx_matrix is an AugmentedMatrix::<3>.
        //
        // Each row of this matrix is of the form [r; dx; x], where x is an element of the
        // source of signature zero, expressed in the masked basis, and dx is the value of
        // the differential on x. Then r is the entries of dx that have zero signature,
        // which we include so that the rref of the matix is nice. In practice, we keep r
        // empty until the very end, and then populate it manually.
        //
        // At the beginning the x's will be the new generators in this bidegree. As we read
        // in the quasi-inverses for the zero signature, we keep on reducing this so that dx
        // is zero in the pivot columns of the quasi-inverse. We can then use (the rref of)
        // this matrix to lift remaining elements with zero signature.
        let (mut target_zero_mask, mut dx_matrix) = if zero_mask_dim != mask.len() {
            let num_new_gens = source.number_of_gens_in_degree(b.t());
            assert_eq!(mask.len(), zero_mask_dim + num_new_gens);

            let target_zero_mask: Vec<usize> = subalgebra
                .signature_mask(
                    &algebra,
                    target,
                    b.t(),
                    &subalgebra.zero_signature(),
                    i32::MAX,
                )
                .collect();
            let mut matrix = AugmentedMatrix::<3>::new(
                p,
                num_new_gens,
                [target_zero_mask.len(), target.dimension(b.t()), mask.len()],
            );

            for i in 0..num_new_gens {
                let dx = self.differentials[b.s()].output(b.t(), i);
                matrix
                    .row_segment_mut(i, 1, 1)
                    .slice_mut(0, dx.len())
                    .add(dx.as_slice(), 1);
                matrix
                    .row_segment_mut(i, 2, 2)
                    .add_basis_element(zero_mask_dim + i, 1);
            }

            (target_zero_mask, matrix)
        } else {
            (Vec::new(), AugmentedMatrix::<3>::new(p, 0, [0, 0, 0]))
        };

        for cmd in commands {
            match cmd? {
                NassauCommand::Signature(sig_u16) => {
                    let signature: Vec<PPartEntry> =
                        sig_u16.iter().map(|&x| x as PPartEntry).collect();
                    mask.clear();
                    // At apply time the resolution is fully computed, so we read the full mask
                    // (no concurrently-growing generators to exclude).
                    mask.extend(subalgebra.signature_mask(
                        &algebra,
                        source,
                        b.t(),
                        &signature,
                        i32::MAX,
                    ));
                    scratch0.set_scratch_vector_size(mask.len());
                }
                NassauCommand::Fix => {
                    // We need to fix the differential problem
                    //
                    // First manually add_masked the second segment to the first, which we
                    // use for row reduction. We do this manually for borrow checker reasons.
                    for (j, &k) in target_zero_mask.iter().enumerate() {
                        for i in 0..dx_matrix.rows() {
                            if dx_matrix.row_segment(i, 1, 1).entry(k) != 0 {
                                dx_matrix.row_segment_mut(i, 0, 0).add_basis_element(j, 1);
                            }
                        }
                    }
                    dx_matrix.row_reduce();

                    // Now reduce by these elements
                    for i in 0..dx_matrix.rows() {
                        let masked_col = dx_matrix.row(i).first_nonzero().unwrap().0;
                        assert_eq!(dx_matrix.pivots()[masked_col], i as isize);
                        let col = target_zero_mask[masked_col];

                        for (input, output) in inputs.iter_mut().zip(results.iter_mut()) {
                            let entry = input.entry(col);
                            if entry != 0 {
                                output.into().add_unmasked(
                                    dx_matrix.row_segment(i, 2, 2),
                                    1,
                                    &mask,
                                );
                                input.as_slice_mut().add(dx_matrix.row_segment(i, 1, 1), 1);
                            }
                        }
                    }

                    // Drop these objects to save a bit of memory
                    target_zero_mask = Vec::new();
                    dx_matrix = AugmentedMatrix::<3>::new(p, 0, [0, 0, 0]);
                }
                NassauCommand::Pivot {
                    col,
                    lift_bytes,
                    image_bytes,
                } => {
                    let col = col as usize;
                    scratch0.update_from_bytes(&mut &lift_bytes[..])?;
                    scratch1.update_from_bytes(&mut &image_bytes[..])?;
                    for (input, output) in inputs.iter_mut().zip(results.iter_mut()) {
                        let entry = input.entry(col);
                        if entry != 0 {
                            output.into().add_unmasked(scratch0.as_slice(), 1, &mask);
                            // If we resume a resolve_through_stem, input may be longer
                            // than scratch1.
                            input
                                .slice_mut(0, scratch1.len())
                                .add(scratch1.as_slice(), 1);
                        }
                    }

                    // Row reduce the differentials
                    if !target_zero_mask.is_empty() {
                        for i in 0..dx_matrix.rows() {
                            if dx_matrix.row_segment(i, 1, 1).entry(col) != 0 {
                                dx_matrix
                                    .row_segment_mut(i, 2, 2)
                                    .slice_mut(0, zero_mask_dim)
                                    .add(scratch0.as_slice(), 1);
                                dx_matrix
                                    .row_segment_mut(i, 1, 1)
                                    .slice_mut(0, target_dim)
                                    .add(scratch1.as_slice(), 1);
                            }
                        }
                    }
                }
            }
        }

        for dx in inputs {
            assert!(
                dx.is_zero(),
                "remainder non-zero at {b}\nAlgebra: {subalgebra}\ndx: {}",
                target.element_to_string(b.t(), dx.as_slice())
            );
        }
        Ok(true)
    }
}

/// A streaming iterator that regenerates the quasi-inverse of `d_{b.s}` at bidegree `b` on the
/// fly, yielding the exact same [`NassauCommand`] stream that [`Resolution::write_qi`] wrote to the
/// save store.
///
/// This lets [`Resolution::apply_quasi_inverse_fallible`] apply a recomputed quasi-inverse through
/// the same code path as a saved one — the apply loop is unchanged; only the source of commands
/// differs. The quasi-inverse is re-derived from `differentials[b.s]` alone (plus the module bases
/// and the deterministically-chosen subalgebra), never the rest of the resolution.
///
/// It advances one signature at a time, holding only that signature's matrices (and the commands
/// of the current signature), so its peak memory matches what resolving this bidegree originally
/// required — never the whole quasi-inverse, which can reach hundreds of GB at record stems.
///
/// The recomputation reproduces the *restricted* quasi-inverse exactly as the writer computed it
/// (target basis truncated to generator degree `< b.t()`, next basis to `< b.t() - 1`), including
/// the unconditional [`NassauCommand::Fix`] emitted after the zero-signature block, so the command
/// stream is byte-for-byte identical to the saved one and the apply loop is unchanged.
struct RecomputeReader<'a, M: ZeroModule<Algebra = MilnorAlgebra>> {
    res: &'a Resolution<M>,
    b: Bidegree,
    subalgebra: MilnorSubalgebra,
    algebra: Arc<MilnorAlgebra>,
    /// Target dimension of the quasi-inverse: the *restricted* dimension of `F_{s-1}` at `t`
    /// (generators of degree `< t - 1` only). Matches the writer's first `nassau_qi_writer` arg.
    /// Header field.
    target_dim: usize,
    /// Restricted column dimension of the differential matrix (`= target_dim`); threaded into
    /// [`Resolution::restricted_partial_matrix`] so images match the writer's serialisation.
    next_dim: usize,
    /// Zero-signature masked source dimension. Header field.
    zero_mask_dim: usize,
    signatures: std::vec::IntoIter<Vec<PPartEntry>>,
    /// Whether the next signature pulled from `signatures` is the zero signature (the first one).
    first_signature: bool,
    /// Whether the unconditional `Fix` still needs to be emitted after the zero-signature block.
    fix_pending: bool,
    scratch: FpVector,
    /// Commands generated for the current signature, awaiting consumption.
    pending: std::vec::IntoIter<NassauCommand>,
}

impl<'a, M: ZeroModule<Algebra = MilnorAlgebra>> RecomputeReader<'a, M> {
    fn new(res: &'a Resolution<M>, b: Bidegree) -> Self {
        let s = b.s();
        let t = b.t();
        // The subalgebra is chosen deterministically, as in `step_resolution_with_result` for the
        // step that computed `(s + 1, t)` (the step that wrote this qi).
        let subalgebra = MilnorSubalgebra::optimal_for(
            Bidegree::s_t(s + 1, t) - Bidegree::s_t(0, res.max_degree),
        );
        // `src` is the source of `d_s` (= F_s), `tgt` its target (= F_{s-1}). From the point of
        // view of the writing step (which computed `(s + 1, t)`), `src` is its *target* module
        // (degree bound `t`) and `tgt` is its *next* module (degree bound `t - 1`).
        let src = &res.modules[s];
        let tgt = &res.modules[s - 1];
        src.compute_basis(t);
        tgt.compute_basis(t);
        let algebra = tgt.algebra();

        let target_bound = t;
        let next_bound = t - 1;

        // The writer's header stores the restricted next-module dimension as the qi target dim.
        let next_dim = MilnorSubalgebra::restricted_dimension(tgt, t, next_bound);
        let target_dim = next_dim;
        let zero_mask_dim = subalgebra
            .signature_mask(&algebra, src, t, &subalgebra.zero_signature(), target_bound)
            .count();

        // Zero signature first, then the rest — matching the write order.
        let signatures: Vec<Vec<PPartEntry>> = std::iter::once(subalgebra.zero_signature())
            .chain(subalgebra.iter_signatures(t))
            .collect();

        Self {
            res,
            b,
            subalgebra,
            algebra,
            target_dim,
            next_dim,
            zero_mask_dim,
            signatures: signatures.into_iter(),
            first_signature: true,
            fix_pending: false,
            scratch: FpVector::new(res.prime(), 0),
            pending: Vec::new().into_iter(),
        }
    }

    /// Row-reduce `d_s` restricted to `signature` and build the resulting [`NassauCommand`]s,
    /// exactly as [`Resolution::write_qi`] did during resolution. Returns an empty vec if the
    /// block has no pivots.
    fn commands_for_signature(
        &mut self,
        signature: &[PPartEntry],
    ) -> anyhow::Result<Vec<NassauCommand>> {
        let (s, t) = (self.b.s(), self.b.t());
        let p = self.res.prime();

        // `src` bound is `t` (target of the writing step), `tgt` bound is `t - 1` (its next).
        let src_mask: Vec<usize> = self
            .subalgebra
            .signature_mask(&self.algebra, &self.res.modules[s], t, signature, t)
            .collect();
        let tgt_mask: Vec<usize> = self
            .subalgebra
            .signature_mask(&self.algebra, &self.res.modules[s - 1], t, signature, t - 1)
            .collect();

        let full_matrix = {
            // Kept while the per-section guards elsewhere were removed: this path is NOT under the
            // whole-bidegree guard in `step_resolution`. `commands_for_signature` runs from
            // `RecomputeReader::next`, driven by `apply_quasi_inverse_fallible` — an accessor that
            // consumers call outside any bidegree, so nothing upstream has raised the depth.
            let _guard = ParallelGuard::new();
            restricted_partial_matrix(&self.res.differentials[s], t, &src_mask, self.next_dim)
        };
        let mut masked_matrix =
            AugmentedMatrix::new(p, src_mask.len(), [tgt_mask.len(), src_mask.len()]);
        masked_matrix
            .segment(0, 0)
            .add_masked(&full_matrix, &tgt_mask);
        masked_matrix.segment(1, 1).add_identity();
        masked_matrix.row_reduce();

        Resolution::<M>::qi_commands(
            &mut self.scratch,
            signature,
            &tgt_mask,
            &full_matrix,
            &masked_matrix,
        )
    }
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> Iterator for RecomputeReader<'_, M> {
    type Item = anyhow::Result<NassauCommand>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(cmd) = self.pending.next() {
                return Some(Ok(cmd));
            }
            // The writer emits an unconditional `Fix` right after the zero-signature block; mirror
            // it here once the zero signature's commands are drained.
            if self.fix_pending {
                self.fix_pending = false;
                return Some(Ok(NassauCommand::Fix));
            }
            let signature = self.signatures.next()?;
            let is_zero = self.first_signature;
            self.first_signature = false;
            match self.commands_for_signature(&signature) {
                Ok(cmds) => {
                    self.pending = cmds.into_iter();
                    if is_zero {
                        self.fix_pending = true;
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> AugmentedChainComplex for Resolution<M> {
    type ChainMap = FreeModuleHomomorphism<M>;
    type TargetComplex = FiniteChainComplex<M, FullModuleHomomorphism<M, M>>;

    fn target(&self) -> Arc<Self::TargetComplex> {
        Arc::clone(&self.target)
    }

    fn chain_map(&self, s: i32) -> Arc<Self::ChainMap> {
        Arc::clone(&self.chain_maps[s])
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;

    #[test]
    fn test_restart_stem() {
        let res = crate::utils::construct_nassau("S_2", None).unwrap();
        res.compute_through_stem(Bidegree::n_s(14, 8));
        res.compute_through_bidegree(Bidegree::s_t(5, 19));

        expect![[r#"
            ·                             
            ·                     ·       
            ·                   · ·     · 
            ·                 ·   ·     · 
            ·             ·   ·         · · 
            ·     ·       · · ·         · ·   
            ·   · ·     · · ·           · · ·   
            · ·   ·       ·               ·       
            ·                                       
        "#]]
        .assert_eq(&res.graded_dimension_string());
    }

    /// Cross-check the secondary (d2) computation on a *save-backed* Nassau resolution computed with
    /// the relaxed [`Resolution::compute_through_stem`] against the standard resolution. This
    /// exercises the quasi-inverse save files, which under the relaxed schedule are always written
    /// using the "incomplete information" (`Magic::Fix`) path, since a bidegree is computed while
    /// ignoring the same-degree generators of its target.
    #[test]
    fn test_stem_concurrent_secondary() {
        use std::sync::Arc;

        use algebra::pair_algebra::PairAlgebra;

        use crate::{
            chain_complex::FreeChainComplex, resolution::secondary::SecondaryResolution,
            secondary::SecondaryLift, utils::construct_standard,
        };

        fn d2_chart<CC>(lift: &SecondaryResolution<CC>) -> String
        where
            CC: FreeChainComplex,
            CC::Algebra: PairAlgebra,
        {
            let underlying = lift.underlying();
            let mut out = String::new();
            // Mirror the guarded iteration in `SecondaryResolution::e3_page`.
            for b in underlying.iter_stem() {
                if b.t() > 0 && underlying.has_computed_bidegree(b + Bidegree::n_s(-1, 2)) {
                    let matrix = lift.homotopy(b.s() + 2).homotopies.hom_k(b.t());
                    if matrix.iter().any(|row| !row.is_empty()) {
                        out.push_str(&format!("d2 {b}: {matrix:?}\n"));
                    }
                }
            }
            out
        }

        let max = Bidegree::n_s(20, 7);

        let dir = tempfile::TempDir::new().unwrap();
        let nassau = crate::utils::construct_nassau("S_2", Some(dir.path().to_owned())).unwrap();
        nassau.compute_through_stem(max);
        let nassau_lift = SecondaryResolution::new(Arc::new(nassau));
        nassau_lift.extend_all();

        let standard = construct_standard::<false, _, _>("S_2", None).unwrap();
        standard.compute_through_stem(max);
        let standard_lift = SecondaryResolution::new(Arc::new(standard));
        standard_lift.extend_all();

        assert_eq!(
            d2_chart(&nassau_lift),
            d2_chart(&standard_lift),
            "secondary d2 chart differs between Nassau (save-backed, relaxed schedule) and \
             standard"
        );
    }

    #[test]
    fn test_signature_iterator() {
        let subalgebra = MilnorSubalgebra::new(vec![2, 1]);
        assert_eq!(
            subalgebra.iter_signatures(6).collect::<Vec<_>>(),
            vec![
                vec![1, 0],
                vec![2, 0],
                vec![3, 0],
                vec![0, 1],
                vec![1, 1],
                vec![2, 1],
                vec![3, 1],
            ]
        );

        assert_eq!(
            subalgebra.iter_signatures(5).collect::<Vec<_>>(),
            vec![
                vec![1, 0],
                vec![2, 0],
                vec![3, 0],
                vec![0, 1],
                vec![1, 1],
                vec![2, 1],
            ]
        );
        assert_eq!(
            subalgebra.iter_signatures(4).collect::<Vec<_>>(),
            vec![vec![1, 0], vec![2, 0], vec![3, 0], vec![0, 1], vec![1, 1],]
        );
        assert_eq!(
            subalgebra.iter_signatures(3).collect::<Vec<_>>(),
            vec![vec![1, 0], vec![2, 0], vec![3, 0], vec![0, 1],]
        );
        assert_eq!(
            subalgebra.iter_signatures(2).collect::<Vec<_>>(),
            vec![vec![1, 0], vec![2, 0],]
        );
        assert_eq!(
            subalgebra.iter_signatures(1).collect::<Vec<_>>(),
            vec![vec![1, 0],]
        );
        assert_eq!(
            subalgebra.iter_signatures(0).collect::<Vec<_>>(),
            Vec::<Vec<PPartEntry>>::new()
        );
    }

    #[test]
    fn test_signature_iterator_large() {
        let subalgebra = MilnorSubalgebra::new(vec![
            0,
            MilnorSubalgebra::INFINITY,
            MilnorSubalgebra::INFINITY,
            MilnorSubalgebra::INFINITY,
        ]);
        assert_eq!(
            subalgebra.iter_signatures(7).collect::<Vec<_>>(),
            vec![vec![0, 1, 0, 0], vec![0, 2, 0, 0], vec![0, 0, 1, 0],]
        );
    }

    /// The packed signature test must agree with the per-entry comparison it replaced, including
    /// on signatures that no element can have. Packing those naively would spill bits into the
    /// neighbouring field and select unrelated elements.
    #[test]
    fn packed_signature_matches_per_entry_test() {
        // The comparison the packed mask replaced, kept here as the reference.
        fn has_signature(profile: &[u8], ppart: PPart, signature: &[PPartEntry]) -> bool {
            for (i, (&profile, &signature)) in profile.iter().zip(signature).enumerate() {
                if ppart.get(i) & ((1u64 << profile) - 1) as PPartEntry != signature {
                    return false;
                }
            }
            true
        }

        let algebra = MilnorAlgebra::new(TWO, false);
        algebra.compute_basis(60);

        for profile in [
            vec![1u8, 1, 1],
            vec![4, 3, 2, 1],
            vec![2, 0, 3],
            // Wider than the fields they constrain.
            vec![9, 9, 9, 9],
            // Longer than a p-part can be, so the tail entries can never be non-zero.
            vec![1; PPart::MAX_LEN + 3],
        ] {
            let subalgebra = MilnorSubalgebra::new(profile.clone());
            for signature in [
                vec![0; profile.len()],
                (0..profile.len()).map(|i| (i % 3) as PPartEntry).collect(),
                // An entry too wide for its field, which must match nothing.
                (0..profile.len())
                    .map(|i| if i == profile.len() - 1 { 255 } else { 0 })
                    .collect(),
            ] {
                let packed = subalgebra.packed_signature(&signature);
                for t in 0..=60 {
                    for &op in algebra.ppart_table(t) {
                        let expected = has_signature(&profile, op, &signature);
                        let actual = packed.is_some_and(|(mask, value)| op.bits() & mask == value);
                        assert_eq!(
                            actual, expected,
                            "profile {profile:?}, signature {signature:?}, element {op:?}"
                        );
                    }
                }
            }
        }
    }
}
