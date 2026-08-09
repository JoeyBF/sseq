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
    sync::{Arc, LazyLock, Mutex, mpsc},
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

/// Extract `rows` of `full` into a fresh matrix (`out.row(i) = full.row(rows[i])`), preserving the
/// column layout. Slices a precomputed full (restricted-column) differential matrix into one
/// signature's partial matrix (see [`reuse_full_matrix`]). `rows` must index within `full` — the
/// signature masks are subsets of `0..full.rows()` (the restricted source basis), so this holds.
fn select_rows(full: &Matrix, rows: &[usize]) -> Matrix {
    let mut out = Matrix::new(full.prime(), rows.len(), full.columns());
    for (dst, &src) in rows.iter().enumerate() {
        out.row_mut(dst).assign(full.row(src));
    }
    out
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

    /// Cap on the bytes held by cached matrices. Speculation is skipped while over it.
    fn max_bytes() -> usize {
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
    pub fn close() {
        let (m, cv) = &*QUEUE;
        let mut q = m.lock().unwrap();
        q.closed = true;
        q.heap.clear();
        cv.notify_all();
    }
}

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
        full_matrix: &Matrix,
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
        let m = self.build_full_restricted(b, target_dim, next_dim, speculate::on_cpu());
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
        let full_reuse: Option<Matrix> = if reuse_full_matrix(&self.differentials[b.s() - 1])
            && reuse_within_cap(target_dim, next_dim)
        {
            let all_rows: Vec<usize> = (0..target_dim).collect();
            // A background builder may already have this matrix. Its inputs are frozen once
            // `(b.s() - 1, b.t() - 1)` is committed, strictly earlier than `(b.s(), b.t())` can run,
            // so a cached matrix is the same matrix — see [`speculate`]. A shape mismatch would mean
            // that reasoning is wrong somewhere, so say so loudly and rebuild rather than trusting
            // it; `NASSAU_SPECULATE_VERIFY` checks the contents too.
            let full = match speculate::take_or_claim(b) {
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
            None
        };

        let full_matrix =
            tracing::trace_span!("zs_select", rows = target_mask.len()).in_scope(|| {
                match &full_reuse {
                    Some(full) => {
                        debug_assert!(target_mask.iter().all(|&r| r < full.rows()));
                        select_rows(full, &target_mask)
                    }
                    None => restricted_partial_matrix_maybe_gpu(
                        &self.differentials[b.s() - 1],
                        b.t(),
                        &target_mask,
                        next_dim,
                    ),
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
            m.segment(0, 0).add_masked(&full_matrix, &next_mask);
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
        let img_full = restricted_partial_matrix_maybe_gpu(
            &self.differentials[b.s()],
            b.t(),
            &source_mask,
            target_dim,
        );
        let mut n = tracing::trace_span!("img_assemble", rows = source_mask.len()).in_scope(|| {
            let mut n = Matrix::new(p, source_mask.len(), target_masked_dim);
            for (mut row, full_row) in std::iter::zip(n.iter_mut(), img_full.iter()) {
                row.add_masked(full_row, 1, &target_mask);
            }
            n
        });
        tracing::trace_span!("img_row_reduce", rows = source_mask.len())
            .in_scope(|| n.row_reduce());
        let next_row = n.rows();

        let num_new_gens = tracing::trace_span!("extend_image")
            .in_scope(|| n.extend_image(0, n.columns(), &kernel, 0).len());

        if b.t() < b.s() {
            assert_eq!(num_new_gens, 0, "Adding generators at {b}");
        }

        self.add_generators(b, num_new_gens);

        let mut xs = vec![FpVector::new(p, target_dim); num_new_gens];
        let mut dxs = vec![FpVector::new(p, next_dim); num_new_gens];

        {
            let _s = tracing::trace_span!("zs_dx_init", gens = xs.len()).entered();
            for ((x, x_masked), dx) in xs
                .iter_mut()
                .zip_eq(n.iter().skip(next_row))
                .zip_eq(&mut dxs)
            {
                x.as_slice_mut().add_unmasked(x_masked, 1, &target_mask);
                for (i, _) in x_masked.iter_nonzero() {
                    dx.as_slice_mut().add(full_matrix.row(i), 1);
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

        for signature in subalgebra.iter_signatures(b.t()) {
            let _guard = tracing::info_span!("step", ?signature).entered();
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

            let full_matrix = tracing::trace_span!("sig_select", rows = target_mask.len())
                .in_scope(|| match &full_reuse {
                    Some(full) => {
                        debug_assert!(target_mask.iter().all(|&r| r < full.rows()));
                        select_rows(full, &target_mask)
                    }
                    None => restricted_partial_matrix_maybe_gpu(
                        &self.differentials[b.s() - 1],
                        b.t(),
                        &target_mask,
                        next_dim,
                    ),
                });

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
                m.segment(0, 0).add_masked(&full_matrix, &next_mask);
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
                for (i, _) in scratch.iter_nonzero() {
                    x.add_basis_element(target_mask[i], 1);
                    dx.as_slice_mut().add(full_matrix.row(i), 1);
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

        if let Some(w) = f {
            w.finish()?;
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

        // Speculative builders run *outside* the rayon pool on purpose. They are a background task
        // whose whole point is to use time the wavefront is not using; putting them in the pool
        // would let them take worker slots from the critical path, which is the opposite of the
        // intent. `std::thread::scope` keeps them borrowing `self` without any `Arc` plumbing.
        std::thread::scope(|spec_scope| {
            for _ in 0..spec_threads {
                let tracing_span = tracing_span.clone();
                spec_scope.spawn(move || {
                    let _tracing_guard = tracing_span.enter();
                    while let Some(b) = speculate::pop() {
                        self.speculate_build(b);
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
                            self.step_resolution(b);
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
                // Enqueue every bidegree whose matrix is now determined but which cannot run yet.
                //
                // `(r, t)`'s matrix needs only `(r - 1, t - 1)` — i.e. `t <= progress[r - 1] + 1` —
                // while running `(r, t)` additionally needs `(r, t - 1)`. Everything strictly between
                // those two bounds is a bidegree we can build for but not yet run: exactly the
                // speculation window. We start at `progress[r] + 2` because `progress[r] + 1` is the
                // bidegree the scheduler is spawning right now, whose matrix a builder would only race.
                let enqueue_spec = |progress: &[i32], spec_issued: &mut Vec<i32>| {
                    if spec_threads == 0 {
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
                        progress[b.s() as usize] = b.t();
                        enqueue_spec(&progress, &mut spec_issued);

                        if mem_report {
                            commit_count += 1;
                            if commit_count % 400 == 0 {
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
                                eprintln!(
                                    "[MEM] commits={commit_count} last_b=({},{}) HOST[diff={:.1} \
                                     mod={:.1} res_master={:.1} res_basis={:.1}]GB \
                                     DEV[master={:.1} basis={:.1} cubecl_use={:.1} \
                                     cubecl_reserved={:.1}]GB",
                                    b.n(),
                                    b.s(),
                                    gb(diff_b),
                                    gb(mod_b),
                                    gb(res_master),
                                    gb(res_basis),
                                    gb(dev_master),
                                    gb(dev_basis),
                                    gbu(dev_pool_use),
                                    gbu(dev_pool_res),
                                );
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
        });

        if spec_threads > 0 {
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
        speculate::clear();

        // Eviction probe (`NASSAU_R_STATS`): dump the R-access distribution once the wavefront is done.
        #[cfg(feature = "gpu")]
        {
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
