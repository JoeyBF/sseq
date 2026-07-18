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
    milnor_algebra::{MilnorAlgebra, PPartEntry},
    module::{
        FreeModule, GeneratorData, Module, ZeroModule,
        homomorphism::{FreeModuleHomomorphism, FullModuleHomomorphism, ModuleHomomorphism},
    },
};
use anyhow::anyhow;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use fp::{
    matrix::{AugmentedMatrix, Matrix},
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
};

/// See [`resolution::SenderData`](../resolution/struct.SenderData.html). This differs by not having the `new` field.
struct SenderData {
    b: Bidegree,
    sender: mpsc::Sender<Self>,
}

impl SenderData {
    pub(crate) fn send(b: Bidegree, sender: mpsc::Sender<Self>) {
        sender
            .send(Self {
                b,
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

    /// Computes the signature of an element
    fn has_signature(&self, ppart: &[PPartEntry], signature: &[PPartEntry]) -> bool {
        for (i, (&profile, &signature)) in self.profile.iter().zip(signature).enumerate() {
            let ppart = ppart.get(i).copied().unwrap_or(0);
            if ppart & ((1 << profile) - 1) != signature {
                return false;
            }
        }
        true
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
                            if self.has_signature(op, signature) {
                                Some(offset + n)
                            } else {
                                None
                            }
                        })
                },
            )
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

    /// Get the matrix of a free module homomorphism when restricted to the subquotient given by
    /// the signature.
    ///
    /// Only generators of the target of degree strictly less than `target_max_gen_degree` are used
    /// (see [`Self::signature_mask`]).
    fn signature_matrix(
        &self,
        hom: &FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>,
        degree: i32,
        signature: &[PPartEntry],
        target_max_gen_degree: i32,
    ) -> Matrix {
        let p = hom.prime();
        let source = hom.source();
        let target = hom.target();
        let algebra = target.algebra();
        let target_degree = degree - hom.degree_shift();

        let target_mask: Vec<usize> = self
            .signature_mask(
                &algebra,
                &target,
                target_degree,
                signature,
                target_max_gen_degree,
            )
            .collect();

        let source_mask: Vec<usize> = self
            .signature_mask(&algebra, &source, degree, signature, i32::MAX)
            .collect();

        let mut scratch = FpVector::new(
            p,
            Self::restricted_dimension(&target, target_degree, target_max_gen_degree),
        );
        let mut result = Matrix::new(p, source_mask.len(), target_mask.len());

        for (mut row, &masked_index) in std::iter::zip(result.iter_mut(), &source_mask) {
            scratch.set_to_zero();
            hom.apply_to_basis_element_restricted(scratch.as_slice_mut(), 1, degree, masked_index);

            row.add_masked(scratch.as_slice(), 1, &target_mask);
        }
        result
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
        let next_mask: Vec<usize> = subalgebra
            .signature_mask(&algebra, next, b.t(), &zero_sig, next_bound)
            .collect();
        let next_masked_dim = next_mask.len();

        // When GPU reuse is active, build ONE full restricted matrix over every (restricted) source
        // row at degree `b.t()` in a single launch, then slice each signature's rows out of it.
        // `target_dim` is the restricted source dimension, so `0..target_dim` is exactly the row set
        // the per-signature masks partition; `next_dim` is the restricted column count.
        let full_reuse: Option<Matrix> = if reuse_full_matrix(&self.differentials[b.s() - 1]) {
            let all_rows: Vec<usize> = (0..target_dim).collect();
            Some(restricted_partial_matrix_maybe_gpu(
                &self.differentials[b.s() - 1],
                b.t(),
                &all_rows,
                next_dim,
            ))
        } else {
            None
        };

        let full_matrix = match &full_reuse {
            Some(full) => {
                debug_assert!(target_mask.iter().all(|&r| r < full.rows()));
                select_rows(full, &target_mask)
            }
            None => {
                restricted_partial_matrix_maybe_gpu(
                    &self.differentials[b.s() - 1],
                    b.t(),
                    &target_mask,
                    next_dim,
                )
            }
        };
        let mut masked_matrix =
            AugmentedMatrix::new(p, target_masked_dim, [next_masked_dim, target_masked_dim]);

        masked_matrix
            .segment(0, 0)
            .add_masked(&full_matrix, &next_mask);
        masked_matrix.segment(1, 1).add_identity();
        masked_matrix.row_reduce();
        let kernel = masked_matrix.compute_kernel();

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

        // Compute image
        let mut n =
            subalgebra.signature_matrix(&self.differentials[b.s()], b.t(), &zero_sig, target_bound);
        n.row_reduce();
        let next_row = n.rows();

        let num_new_gens = n.extend_image(0, n.columns(), &kernel, 0).len();

        if b.t() < b.s() {
            assert_eq!(num_new_gens, 0, "Adding generators at {b}");
        }

        self.add_generators(b, num_new_gens);

        let mut xs = vec![FpVector::new(p, target_dim); num_new_gens];
        let mut dxs = vec![FpVector::new(p, next_dim); num_new_gens];

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

        // Now add correction terms
        let mut target_mask: Vec<usize> = Vec::new();
        let mut next_mask: Vec<usize> = Vec::new();

        drop(guard);

        for signature in subalgebra.iter_signatures(b.t()) {
            let _guard = tracing::info_span!("step", ?signature).entered();
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

            let full_matrix = match &full_reuse {
                Some(full) => {
                    debug_assert!(target_mask.iter().all(|&r| r < full.rows()));
                    select_rows(full, &target_mask)
                }
                None => {
                    restricted_partial_matrix_maybe_gpu(
                        &self.differentials[b.s() - 1],
                        b.t(),
                        &target_mask,
                        next_dim,
                    )
                }
            };

            let mut masked_matrix =
                AugmentedMatrix::new(p, target_mask.len(), [next_mask.len(), target_mask.len()]);
            masked_matrix
                .segment(0, 0)
                .add_masked(&full_matrix, &next_mask);
            masked_matrix.segment(1, 1).add_identity();
            masked_matrix.row_reduce();

            let qi = masked_matrix.compute_quasi_inverse();
            let pivots = qi.pivots().unwrap();
            let preimage = qi.preimage();

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
            Self::write_qi(
                &mut f,
                &mut scratch,
                &signature,
                &next_mask,
                &full_matrix,
                &masked_matrix,
            )?;
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
            {
                chain_map.get_matrix(matrix.segment(0, 0), t);
            }
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

        let mut matrix =
            AugmentedMatrix::<2>::new(p, target_dim, [cc_module.dimension(t), target_dim]);
        {
            self.chain_maps[0].get_matrix(matrix.segment(0, 0), t);
        }
        matrix.segment(1, 1).add_identity();
        matrix.row_reduce();
        let desired_image = matrix.compute_kernel();

        let mut matrix = AugmentedMatrix::<2>::new_with_capacity(
            p,
            source_dim,
            &[target_dim, source_dim],
            source_dim + MAX_NEW_GENS,
            0,
        );
        {
            self.differentials[1].get_matrix(matrix.segment(0, 0), t);
        }
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
        maybe_rayon::in_place_scope(|scope| {
            let _tracing_guard = tracing_span.enter();

            let mut progress: Vec<i32> = vec![min_degree - 1; max_s as usize + 1];

            let (sender, receiver) = mpsc::channel();

            let f = |b: Bidegree, sender: mpsc::Sender<SenderData>| {
                if self.has_computed_bidegree(b) {
                    SenderData::send(b, sender);
                } else {
                    let tracing_span = tracing_span.clone();
                    scope.spawn(move |_| {
                        let _tracing_guard = tracing_span.enter();
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
                    f(Bidegree::s_t(s, min_degree), sender.clone());
                }
            }
            drop(sender);

            while let Ok(SenderData { b, sender }) = receiver.recv() {
                assert!(progress[b.s() as usize] == b.t() - 1);
                progress[b.s() as usize] = b.t();

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
                        f(cand, sender.clone());
                    }
                }
            }
        });
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
                    mask.extend(
                        subalgebra.signature_mask(&algebra, source, b.t(), &signature, i32::MAX),
                    );
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
}
