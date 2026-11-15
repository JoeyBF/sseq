//! This module defines [`MuResolutionHomomorphism`], which is a chain map from a
//! [`FreeChainComplex`].
use std::{ops::Range, sync::Arc};

use algebra::{
    Algebra, MuAlgebra,
    module::{
        Module,
        homomorphism::{ModuleHomomorphism, MuFreeModuleHomomorphism},
    },
};
use fp::{
    matrix::Matrix,
    vector::{FpSliceMut, FpVector},
};
use once::OnceBiVec;
use sseq::coordinates::{Bidegree, BidegreeGenerator, BidegreeRange};

use crate::{
    chain_complex::{AugmentedChainComplex, BoundedChainComplex, ChainComplex, FreeChainComplex},
    lift::{LiftPrep, LiftRequest, Liftable},
    save::{SaveDirectory, SaveKind},
};

pub type ResolutionHomomorphism<CC1, CC2> = MuResolutionHomomorphism<false, CC1, CC2>;
pub type UnstableResolutionHomomorphism<CC1, CC2> = MuResolutionHomomorphism<true, CC1, CC2>;

/// A chain complex homomorphims from a [`FreeChainComplex`]. This contains logic to lift chain
/// maps using the freeness.
pub struct MuResolutionHomomorphism<const U: bool, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: ChainComplex<Algebra = CC1::Algebra>,
{
    name: String,
    pub source: Arc<CC1>,
    pub target: Arc<CC2>,
    maps: OnceBiVec<Arc<MuFreeModuleHomomorphism<U, CC2::Module>>>,
    pub shift: Bidegree,
    /// The `products/{name}` subgroup of the source's save directory.
    ///
    /// It holds this chain map and its secondary lifts, and is `None` for an unnamed homomorphism.
    save_dir: SaveDirectory,
}

impl<const U: bool, CC1, CC2> MuResolutionHomomorphism<U, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: ChainComplex<Algebra = CC1::Algebra>,
{
    pub fn new(name: String, source: Arc<CC1>, target: Arc<CC2>, shift: Bidegree) -> Self {
        let save_dir = if !name.is_empty()
            && let Some(parent) = source.save_dir().store()
        {
            SaveDirectory::Store(Arc::new(
                parent
                    .subgroup(&format!("products/{name}"))
                    .expect("Failed to create products subgroup"),
            ))
        } else {
            SaveDirectory::None
        };
        Self {
            name,
            source,
            target,
            maps: OnceBiVec::new(shift.s()),
            shift,
            save_dir,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn algebra(&self) -> Arc<CC1::Algebra> {
        self.source.algebra()
    }

    pub fn next_homological_degree(&self) -> i32 {
        self.maps.len()
    }

    fn get_map_ensure_length(&self, input_s: i32) -> &MuFreeModuleHomomorphism<U, CC2::Module> {
        self.maps.extend(input_s, |input_s| {
            let output_s = input_s - self.shift.s();
            let map = MuFreeModuleHomomorphism::new(
                self.source.module(input_s),
                self.target.module(output_s),
                self.shift.t(),
            );
            if let Some(cache) = lazy_maps::cache()
                && let Some(store) = self.save_dir.store_arc()
            {
                let (source, target) = (self.source.module(input_s), self.target.module(output_s));
                let (p, shift_t, name) = (self.source.prime(), self.shift.t(), self.name.clone());
                map.set_loader(
                    Arc::new(move |t| {
                        let input = Bidegree::s_t(input_s, t);
                        let data = store
                            .read(SaveKind::ChainMap, input)
                            .unwrap_or_else(|e| {
                                panic!("reading chain map {name:?} at {input}: {e:#}")
                            })
                            .unwrap_or_else(|| {
                                panic!("chain map {name:?} at {input} is not saved")
                            });
                        let dim = target.dimension(t - shift_t);
                        let mut cursor = &data[..];
                        (0..source.number_of_gens_in_degree(t))
                            .map(|_| {
                                FpVector::from_bytes(p, dim, &mut cursor).unwrap_or_else(|e| {
                                    panic!("decoding chain map {name:?} at {input}: {e}")
                                })
                            })
                            .collect()
                    }),
                    Some(Arc::clone(cache)),
                );
            }
            Arc::new(map)
        });
        &self.maps[input_s]
    }

    /// Bytes this map holds for good: its steps that are not read through
    /// [`lazy_maps`]'s cache.
    pub fn resident_bytes(&self) -> u64 {
        self.maps.values().map(|m| m.resident_bytes()).sum()
    }

    /// Returns the chain map on the `s`th source module.
    pub fn get_map(&self, input_s: i32) -> Arc<MuFreeModuleHomomorphism<U, CC2::Module>> {
        Arc::clone(&self.maps[input_s])
    }

    pub fn save_dir(&self) -> &SaveDirectory {
        &self.save_dir
    }
}

impl<const U: bool, CC1, CC2> MuResolutionHomomorphism<U, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: ChainComplex<Algebra = CC1::Algebra>,
{
    /// Extend the resolution homomorphism such that it is defined on degrees
    /// (`max_s`, `max_t`).
    ///
    /// This assumes in yet-uncomputed bidegrees, the homology of the source consists only of
    /// decomposables (e.g. it is trivial). More precisely, we assume
    /// [`MuResolutionHomomorphism::extend_step_raw`] can be called with `extra_images = None`.
    #[tracing::instrument(fields(self = self.name, %max))]
    pub fn extend(&self, max: Bidegree) {
        self.extend_profile(BidegreeRange::new(&(), max.s() + 1, &|_, _| max.t() + 1))
    }

    /// Extend the resolution homomorphism such that it is defined on degrees
    /// (`max_n`, `max_s`).
    ///
    /// This assumes in yet-uncomputed bidegrees, the homology of the source consists only of
    /// decomposables (e.g. it is trivial). More precisely, we assume
    /// [`MuResolutionHomomorphism::extend_step_raw`] can be called with `extra_images = None`.
    #[tracing::instrument(fields(self = self.name, %max))]
    pub fn extend_through_stem(&self, max: Bidegree) {
        self.extend_profile(BidegreeRange::new(&(), max.s() + 1, &|_, s| {
            max.n() + s + 1
        }))
    }

    /// Extend the resolution homomorphism as far as possible, as constrained by how much the
    /// source and target have been resolved.
    ///
    /// This assumes in yet-uncomputed bidegrees, the homology of the source consists only of
    /// decomposables (e.g. it is trivial). More precisely, we assume
    /// [`MuResolutionHomomorphism::extend_step_raw`] can be called with `extra_images = None`.
    #[tracing::instrument(fields(self = self.name))]
    pub fn extend_all(&self) {
        self.extend_profile(BidegreeRange::new(
            self,
            std::cmp::min(
                self.target.next_homological_degree() + self.shift.s(),
                self.source.next_homological_degree(),
            ),
            &|selff, s| {
                std::cmp::min(
                    selff
                        .target
                        .module(s - selff.shift.s())
                        .max_computed_degree()
                        + selff.shift.t(),
                    selff.source.module(s).max_computed_degree(),
                ) + 1
            },
        ));
    }

    /// Build the chain map `∑_i coeff_i · summand_i` from chain maps that share a source, target,
    /// and shift.
    ///
    /// Every summand must already be defined at least as far as `max`. The combined map's image on
    /// each generator is the same linear combination of the summands' images, so unlike
    /// [`from_class`](Self::from_class) no quasi-inverse lift is performed.
    ///
    /// Call with an **empty `name`**: this populates the maps directly (not through
    /// [`extend_step_raw`](Self::extend_step_raw)), so a non-empty name on a save-enabled source
    /// would create a `products/{name}` save directory that is never written to.
    pub fn linear_combination(name: String, summands: &[(u32, Arc<Self>)], max: Bidegree) -> Self {
        assert!(
            !summands.is_empty(),
            "linear_combination requires at least one summand"
        );
        assert!(
            name.is_empty(),
            "linear_combination populates maps directly and never writes to a save directory; \
             pass an empty name to avoid creating a `products/{{name}}` directory that stays empty"
        );
        let shift = summands[0].1.shift;
        let source = Arc::clone(&summands[0].1.source);
        let target = Arc::clone(&summands[0].1.target);
        for (_, m) in summands {
            assert_eq!(
                m.shift, shift,
                "linear_combination summands must share a shift"
            );
            assert!(
                Arc::ptr_eq(&m.source, &source),
                "linear_combination summands must share a source"
            );
            assert!(
                Arc::ptr_eq(&m.target, &target),
                "linear_combination summands must share a target"
            );
        }

        let p = source.prime();
        let result = Self::new(name, source, target, shift);

        // The combined map is defined wherever every summand is. Iterate `(s, t)` in the same order
        // the summands were built and sum their generator images.
        let max_s = summands
            .iter()
            .map(|(_, m)| m.next_homological_degree())
            .min()
            .unwrap()
            .min(max.s() + 1);

        for input_s in shift.s()..max_s {
            let out_map = result.get_map_ensure_length(input_s);
            let src_mod = out_map.source();
            let tgt_mod = out_map.target();
            let summand_maps: Vec<_> = summands
                .iter()
                .map(|(c, m)| (*c, m.get_map(input_s)))
                .collect();
            let next_t = summand_maps
                .iter()
                .map(|(_, m)| m.next_degree())
                .min()
                .unwrap();

            for t in out_map.next_degree()..next_t {
                let num_gens = src_mod.number_of_gens_in_degree(t);
                let dim = tgt_mod.dimension(t - shift.t());
                let mut rows = vec![FpVector::new(p, dim); num_gens];
                for (c, m) in &summand_maps {
                    for (k, row) in rows.iter_mut().enumerate() {
                        row.add(&m.output(t, k), *c);
                    }
                }
                out_map.add_generators_from_rows(t, rows);
            }
        }
        result
    }

    /// Extends the resolution homomorphism up to a given range. This range is first specified by
    /// the maximum `s`, then the maximum `t` for each `s`. This should rarely be used directly;
    /// instead one should use [`MuResolutionHomomorphism::extend`],
    /// [`MuResolutionHomomorphism::extend_through_stem`] and [`ResolutionHomomorphism::extend_all`]
    /// as appropriate.
    ///
    /// Note that unlike the more specific versions of this function, the bounds `max_s` and
    /// `max_t` are exclusive.
    ///
    /// This assumes in yet-uncomputed bidegrees, the homology of the source consists only of
    /// decomposables (e.g. it is trivial). More precisely, we assume
    /// [`MuResolutionHomomorphism::extend_step_raw`] can be called with `extra_images = None`.
    pub fn extend_profile<AUX: Sync>(&self, max: BidegreeRange<AUX>) {
        self.get_map_ensure_length(max.s() - 1);

        sseq::coordinates::iter_s_t(
            &|b| self.extend_step_raw(b, None),
            Bidegree::s_t(
                self.shift.s(),
                self.get_map_ensure_length(self.shift.s()).min_degree(),
            ),
            max,
        );

        for s in self.shift.s()..max.s() {
            assert_eq!(
                Vec::<i32>::new(),
                self.maps[s].ooo_outputs(),
                "Map {s} has out of order elements"
            );
        }
    }

    /// Extend the [`MuResolutionHomomorphism`] to be defined on `(input_s, input_t)`. The resulting
    /// homomorphism `f` is a chain map such that if `g` is the `k`th generator in the source such
    /// that `d(g) = 0`, then `f(g)` is the `k`th row of `extra_images`.
    ///
    /// The user should call this function explicitly to manually define the chain map where the
    /// chain complex is not exact, and then call [`MuResolutionHomomorphism::extend_all`] to extend
    /// the rest by exactness.
    #[tracing::instrument(skip(self, extra_images), fields(self = self.name, %input))]
    pub fn extend_step_raw(
        &self,
        input: Bidegree,
        extra_images: Option<Vec<FpVector>>,
    ) -> Range<i32> {
        match self.prepare_step(input, extra_images) {
            StepPrep::Done(range) => range,
            StepPrep::NeedsLift(mut pending) => {
                let p = self.source.prime();
                let mut results =
                    vec![FpVector::new(p, pending.fx_dimension); pending.fdx_vectors.len()];
                if !pending.fdx_vectors.is_empty() {
                    assert!(self.target.apply_quasi_inverse(
                        &mut results,
                        pending.output,
                        &pending.fdx_vectors
                    ));
                }
                self.finish_step(&mut pending, &results)
            }
        }
    }

    /// The part of a lift step *before* the quasi-inverse solve: it resolves the cases that need no
    /// quasi-inverse (already computed, loaded from the save store, zero-dimensional, or an
    /// augmentation step) itself, returning [`StepPrep::Done`]; otherwise it returns
    /// [`StepPrep::NeedsLift`] carrying the fdx vectors to lift at `output = input - shift` and the
    /// partially-filled outputs, to be completed by [`Self::finish_step`].
    ///
    /// Splitting the step this way lets [`MultiLift`](crate::lift::MultiLift) gather the fdx
    /// vectors of many maps at a common output bidegree and issue a single batched
    /// `apply_quasi_inverse`.
    pub(crate) fn prepare_step(
        &self,
        input: Bidegree,
        extra_images: Option<Vec<FpVector>>,
    ) -> StepPrep {
        match self.prepare_step_inner(input, extra_images, false) {
            Ok(step) => step,
            Err(_) => unreachable!("a step is deferred only when asked to"),
        }
    }

    /// [`Self::prepare_step`], which with `defer` (see [`lift_batch`]) returns the step's
    /// products instead of multiplying them, as `Err`, to be finished by
    /// [`DeferredStep::finish`] once multiplied.
    fn prepare_step_inner(
        &self,
        input: Bidegree,
        extra_images: Option<Vec<FpVector>>,
        defer: bool,
    ) -> Result<StepPrep, Box<DeferredStep>> {
        let output = input - self.shift;
        assert!(self.target.has_computed_bidegree(output));
        assert!(self.source.has_computed_bidegree(input));
        assert!(input.s() >= self.shift.s());

        let f_cur = self.get_map_ensure_length(input.s());
        if input.t() < f_cur.next_degree() {
            assert!(extra_images.is_none());
            // We need to signal to compute the dependents of this
            return Ok(StepPrep::Done(input.t()..input.t() + 1));
        }

        let p = self.source.prime();

        let num_gens = f_cur.source().number_of_gens_in_degree(input.t());
        let fx_dimension = f_cur.target().dimension(output.t());

        if num_gens == 0 || fx_dimension == 0 {
            return Ok(StepPrep::Done(f_cur.add_generators_from_rows_ooo(
                input.t(),
                vec![FpVector::new(p, fx_dimension); num_gens],
            )));
        }

        if f_cur.has_loader()
            && let Some(store) = self.save_dir.store()
            && store
                .read_prefix(SaveKind::ChainMap, input, 0)
                .unwrap_or_else(|e| panic!("reading chain map {:?} at {input}: {e:#}", self.name))
                .is_some()
        {
            return Ok(StepPrep::Done(f_cur.add_generators_lazily_ooo(input.t())));
        }

        if let Some(store) = self.save_dir.store()
            && let Some(data) = store
                .read(SaveKind::ChainMap, input)
                .unwrap_or_else(|e| panic!("reading chain map {:?} at {input}: {e:#}", self.name))
        {
            let mut cursor = &data[..];
            let mut outputs = Vec::with_capacity(num_gens);
            for _ in 0..num_gens {
                outputs.push(
                    FpVector::from_bytes(p, fx_dimension, &mut cursor).unwrap_or_else(|e| {
                        panic!("decoding chain map {:?} at {input}: {e}", self.name)
                    }),
                );
            }
            return Ok(StepPrep::Done(
                f_cur.add_generators_from_rows_ooo(input.t(), outputs),
            ));
        }

        if output.s() == 0 {
            let outputs =
                extra_images.unwrap_or_else(|| vec![FpVector::new(p, fx_dimension); num_gens]);
            self.save_chain_map(input, &outputs);
            return Ok(StepPrep::Done(
                f_cur.add_generators_from_rows_ooo(input.t(), outputs),
            ));
        }

        let mut outputs = vec![FpVector::new(p, fx_dimension); num_gens];
        let d_source = self.source.differential(input.s());
        let d_target = self.target.differential(output.s());
        let f_prev = self.get_map(input.s() - 1);
        assert!(Arc::ptr_eq(&d_source.source(), &f_cur.source()));
        assert!(Arc::ptr_eq(&d_source.target(), &f_prev.source()));
        assert!(Arc::ptr_eq(&d_target.source(), &f_cur.target()));
        assert!(Arc::ptr_eq(&d_target.target(), &f_prev.target()));
        let fdx_dimension = f_prev.target().dimension(output.t());

        // First take care of generators that hit the target chain complex.
        let mut extra_image_row = 0;
        for (k, output_row) in outputs.iter_mut().enumerate() {
            if d_source.output(input.t(), k).is_zero() {
                let extra_image_matrix = extra_images.as_ref().expect("Missing extra image rows");
                output_row.assign(&extra_image_matrix[extra_image_row]);
                extra_image_row += 1;
            }
        }

        // Now compute the fdx vectors to lift, and remember which output row each fills.
        d_target.compute_auxiliary_data_through_degree(output.t());

        let qi_rows: Vec<usize> = (0..num_gens)
            .filter(|&k| !d_source.output(input.t(), k).is_zero())
            .collect();

        if defer && lift_batch::enabled() && p == 2 {
            let dx_terms = qi_rows
                .iter()
                .map(|&k| d_source.output(input.t(), k).iter_nonzero().count())
                .sum();
            if let Some(algebra) = lift_batch::applicable(&*f_prev.source(), &*f_prev.target()) {
                return Err(Box::new(DeferredStep {
                    input,
                    output,
                    outputs,
                    qi_rows,
                    fx_dimension,
                    fdx_dimension,
                    dx_terms,
                    algebra,
                }));
            }
        }

        let terms = terms_log::Cell::default();
        let compute_fdx_vector = |k: usize| {
            let dx_vector = d_source.output(input.t(), k);
            if dx_vector.is_zero() {
                None
            } else {
                let start = terms_log::enabled().then(std::time::Instant::now);
                let mut fdx_vector = FpVector::new(p, fdx_dimension);
                f_prev.apply(
                    fdx_vector.as_slice_mut(),
                    1,
                    input.t(),
                    dx_vector.as_slice(),
                );
                if let Some(start) = start {
                    let elapsed = start.elapsed();
                    let source = f_prev.source();
                    let (mut dx_terms, mut products, mut max_fw, mut op_deg_sum) = (0, 0, 0, 0);
                    for (idx, _) in dx_vector.iter_nonzero() {
                        let og = source.index_to_op_gen(input.t(), idx);
                        dx_terms += 1;
                        // As in `apply`: generators below the map's range map to zero.
                        if og.generator_degree < f_prev.min_degree() {
                            continue;
                        }
                        let fw = f_prev
                            .output(og.generator_degree, og.generator_index)
                            .iter_nonzero()
                            .count() as u64;
                        products += fw;
                        max_fw = max_fw.max(fw);
                        op_deg_sum += fw * og.operation_degree as u64;
                    }
                    let fdx_nnz = fdx_vector.iter_nonzero().count() as u64;
                    terms.add(dx_terms, products, max_fw, op_deg_sum, fdx_nnz, elapsed);
                }
                Some(fdx_vector)
            }
        };

        // Same order as `qi_rows` (ascending generator index over non-zero differentials).
        let fdx_vectors: Vec<FpVector> = self
            .shared_fdx_vectors(input, &qi_rows, fdx_dimension)
            .unwrap_or_else(|| {
                crate::utils::parallel::guarded_map(qi_rows.len(), |i| {
                    compute_fdx_vector(qi_rows[i])
                })
                .into_iter()
                .flatten()
                .collect()
            });

        if terms_log::enabled() {
            // The (a, w) pairs of all the d x at this step: positions in the source's basis. With
            // a·f(w) shared across generators, f(d x) would cost one product per pair and term.
            let source = f_prev.source();
            let target = f_prev.target();
            let algebra = source.algebra();
            let mut union = vec![false; source.dimension(input.t())];
            for &k in &qi_rows {
                for (idx, _) in d_source.output(input.t(), k).iter_nonzero() {
                    union[idx] = true;
                }
            }
            let (mut distinct, mut shared) = (0, 0);
            for idx in (0..union.len()).filter(|&i| union[i]) {
                distinct += 1;
                let og = source.index_to_op_gen(input.t(), idx);
                if og.generator_degree < f_prev.min_degree() {
                    continue;
                }
                let fw = f_prev.output(og.generator_degree, og.generator_index);
                let fw_degree = og.generator_degree - f_prev.degree_shift();
                for (j, _) in fw.iter_nonzero() {
                    shared += 1;
                    if terms_log::sample_pair() {
                        terms_log::write_pair(
                            og.operation_degree,
                            &algebra
                                .basis_element_to_string(og.operation_degree, og.operation_index),
                            fw_degree,
                            &target.basis_element_to_string(fw_degree, j),
                        );
                    }
                }
            }
            terms.add_shared(distinct, shared);
        }
        terms.write(input, output, fdx_vectors.len());

        Ok(StepPrep::NeedsLift(PendingStep {
            input,
            output,
            outputs,
            fdx_vectors,
            qi_rows,
            fx_dimension,
        }))
    }

    /// `f(d x)` for the generators `qi_rows` at `input`, on this CPU: as [`Self::prepare_step`]
    /// computes it, without the measurement hooks.
    fn cpu_fdx_vectors(
        &self,
        input: Bidegree,
        qi_rows: &[usize],
        fdx_dimension: usize,
    ) -> Vec<FpVector> {
        let p = self.source.prime();
        let d_source = self.source.differential(input.s());
        let f_prev = self.get_map(input.s() - 1);
        self.shared_fdx_vectors(input, qi_rows, fdx_dimension)
            .unwrap_or_else(|| {
                crate::utils::parallel::guarded_map(qi_rows.len(), |i| {
                    let mut v = FpVector::new(p, fdx_dimension);
                    f_prev.apply(
                        v.as_slice_mut(),
                        1,
                        input.t(),
                        d_source.output(input.t(), qi_rows[i]).as_slice(),
                    );
                    v
                })
            })
    }

    /// `f(d x)` for the generators `qi_rows` at `input`, computing each product `a·f(w)` once even
    /// when several of the `d x` contain the term `a·w`.
    ///
    /// The terms are grouped by which of the `d x` contain them. For each group `S`, one
    /// application of `f` gives `f(Σ_{a·w ∈ S} a·w)`, and `f(d x_k)` is the sum of the groups
    /// containing `x_k`. `None` when this does not apply (an odd prime, fewer than two generators,
    /// or too many groups for the sums to be cheap), or with `EXT_LIFT_SHARE=0`.
    fn shared_fdx_vectors(
        &self,
        input: Bidegree,
        qi_rows: &[usize],
        fdx_dimension: usize,
    ) -> Option<Vec<FpVector>> {
        /// At most this many groups: each costs one dense sum per generator containing it.
        const MAX_GROUPS: usize = 256;
        static SHARE: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var("EXT_LIFT_SHARE").map_or(true, |v| v != "0"));

        let p = self.source.prime();
        if p != 2 || qi_rows.len() < 2 || qi_rows.len() > 64 || !*SHARE {
            return None;
        }
        let d_source = self.source.differential(input.s());
        let f_prev = self.get_map(input.s() - 1);

        // (term, the d x containing it), then one group per set of d x.
        let mut terms: Vec<(usize, u64)> = vec![];
        for (j, &k) in qi_rows.iter().enumerate() {
            terms.extend(
                d_source
                    .output(input.t(), k)
                    .iter_nonzero()
                    .map(|(idx, _)| (idx, 1u64 << j)),
            );
        }
        terms.sort_unstable_by_key(|&(idx, _)| idx);
        let mut groups: std::collections::HashMap<u64, Vec<usize>> = Default::default();
        let mut i = 0;
        while i < terms.len() {
            let idx = terms[i].0;
            let mut mask = 0;
            while i < terms.len() && terms[i].0 == idx {
                mask |= terms[i].1;
                i += 1;
            }
            groups.entry(mask).or_default().push(idx);
        }
        if groups.len() > MAX_GROUPS {
            return None;
        }
        let groups: Vec<(u64, Vec<usize>)> = groups.into_iter().collect();

        let source_dimension = f_prev.source().dimension(input.t());
        let sums = crate::utils::parallel::guarded_map(groups.len(), |g| {
            let mut indicator = FpVector::new(p, source_dimension);
            for &idx in &groups[g].1 {
                indicator.set_entry(idx, 1);
            }
            let mut sum = FpVector::new(p, fdx_dimension);
            f_prev.apply(sum.as_slice_mut(), 1, input.t(), indicator.as_slice());
            sum
        });
        Some(
            (0..qi_rows.len())
                .map(|j| {
                    let mut v = FpVector::new(p, fdx_dimension);
                    for ((mask, _), sum) in groups.iter().zip(&sums) {
                        if mask >> j & 1 == 1 {
                            v.add(sum, 1);
                        }
                    }
                    v
                })
                .collect(),
        )
    }

    /// Complete a lift step: scatter the lifted `results` into the pending outputs, persist the
    /// chain map, and register the generators. `results[i]` is the lift of `pending.fdx_vectors[i]`
    /// and fills output row `pending.qi_rows[i]`.
    pub(crate) fn finish_step(
        &self,
        pending: &mut PendingStep,
        results: &[FpVector],
    ) -> Range<i32> {
        assert_eq!(results.len(), pending.qi_rows.len());
        for (&k, result) in pending.qi_rows.iter().zip(results) {
            pending.outputs[k].assign(result);
        }
        self.save_chain_map(pending.input, &pending.outputs);
        let map = self.get_map(pending.input.s());
        if map.has_loader() {
            // Saved just now; read back through the cache when next needed.
            pending.outputs.clear();
            map.add_generators_lazily_ooo(pending.input.t())
        } else {
            map.add_generators_from_rows_ooo(
                pending.input.t(),
                std::mem::take(&mut pending.outputs),
            )
        }
    }

    /// Write the chain map's value at `input` to the save store, if there is one to write to.
    fn save_chain_map(&self, input: Bidegree, outputs: &[FpVector]) {
        if let Some(store) = self.save_dir.store() {
            let mut buf = Vec::new();
            for row in outputs {
                row.to_bytes(&mut buf).unwrap();
            }
            store
                .write(SaveKind::ChainMap, input, &buf)
                .unwrap_or_else(|e| panic!("writing chain map {:?} at {input}: {e:#}", self.name));
        }
    }
}

/// Outcome of [`MuResolutionHomomorphism::prepare_step`]; see [`LiftPrep`].
pub(crate) type StepPrep = LiftPrep<PendingStep>;

/// A lift step whose products `a·f(w)` were left to the caller; see
/// [`crate::lift::Liftable::prepare_batched`].
pub(crate) struct DeferredStep {
    input: Bidegree,
    output: Bidegree,
    outputs: Vec<FpVector>,
    qi_rows: Vec<usize>,
    fx_dimension: usize,
    fdx_dimension: usize,
    /// The number of terms of the `d x` of `qi_rows`. The terms themselves are listed only when
    /// the step is built: a bidegree defers all of its maps' steps at once, and their lists held
    /// from then until multiplication took most of the memory of the n400 ring.
    dx_terms: usize,
    algebra: Arc<algebra::MilnorAlgebra>,
}

/// A [`DeferredStep`] together with the map it belongs to, for [`crate::lift::MultiLift`].
struct DeferredLift<'a, const U: bool, CC1: FreeChainComplex<U>, CC2>
where
    CC1::Algebra: MuAlgebra<U>,
    CC2: ChainComplex<Algebra = CC1::Algebra>,
{
    hom: &'a MuResolutionHomomorphism<U, CC1, CC2>,
    step: Box<DeferredStep>,
}

impl<'a, const U: bool, CC1, CC2> crate::lift::DeferredPrepare<'a> for DeferredLift<'a, U, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: ChainComplex<Algebra = CC1::Algebra> + 'static,
{
    fn out_cols(&self) -> usize {
        self.step.fdx_dimension
    }

    fn build(&self) -> Vec<lift_batch::Product> {
        let step = &*self.step;
        let f_prev = self.hom.get_map(step.input.s() - 1);
        let d_source = self.hom.source.differential(step.input.s());
        let dx: Vec<Vec<usize>> = step
            .qi_rows
            .iter()
            .map(|&k| {
                d_source
                    .output(step.input.t(), k)
                    .iter_nonzero()
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        let products = lift_batch::build_shared(
            &*f_prev.source(),
            &*f_prev.target(),
            f_prev.min_degree(),
            f_prev.degree_shift(),
            step.input.t(),
            &dx,
            &|deg, idx| f_prev.output(deg, idx).clone(),
        )
        .map_or_else(Vec::new, |batch| batch.products);
        let terms: usize = products.iter().map(|p| p.term_indices.len()).sum();
        if terms >= lift_batch::step_log_min_terms() {
            tracing::info!(
                target: "ext::lift::steps",
                kind = "step",
                map = self.hom.name(),
                n = step.output.n(),
                s = step.output.s(),
                terms,
            );
        }
        products
    }

    fn size(&self) -> usize {
        self.step.dx_terms
    }

    fn num_rows(&self) -> usize {
        self.step.qi_rows.len()
    }

    fn algebra(&self) -> Arc<algebra::MilnorAlgebra> {
        Arc::clone(&self.step.algebra)
    }

    fn finish(self: Box<Self>, rows: Option<&[FpVector]>) -> Option<crate::lift::LiftRequest<'a>> {
        let Self { hom, step } = *self;
        let DeferredStep {
            input,
            output,
            outputs,
            qi_rows,
            fx_dimension,
            fdx_dimension,
            dx_terms: _,
            algebra: _,
        } = *step;
        let fdx_vectors = match rows {
            Some(rows) => rows.to_vec(),
            None => hom.cpu_fdx_vectors(input, &qi_rows, fdx_dimension),
        };
        if lift_batch::check() {
            assert_eq!(
                fdx_vectors,
                hom.cpu_fdx_vectors(input, &qi_rows, fdx_dimension),
                "a batched lift step differs from the CPU at {input}"
            );
        }
        let mut pending = PendingStep {
            input,
            output,
            outputs,
            fdx_vectors,
            qi_rows,
            fx_dimension,
        };
        let inputs = std::mem::take(&mut pending.fdx_vectors);
        Some(crate::lift::LiftRequest {
            inputs,
            finish: Box::new(move |results| {
                hom.finish_step(&mut pending, results);
            }),
        })
    }
}

/// A lift step awaiting its quasi-inverse solve; see [`MuResolutionHomomorphism::prepare_step`].
pub(crate) struct PendingStep {
    input: Bidegree,
    output: Bidegree,
    /// Outputs with the extra-image rows filled and the quasi-inverse rows left zero.
    outputs: Vec<FpVector>,
    /// The vectors to lift (`d(f(x))` preimages), one per non-zero-differential generator.
    fdx_vectors: Vec<FpVector>,
    /// `qi_rows[i]` is the output row that `fdx_vectors[i]`'s lift fills.
    qi_rows: Vec<usize>,
    /// Dimension of each lifted result (the target module dimension at `output`).
    fx_dimension: usize,
}

impl<const U: bool, CC1, CC2> Liftable for MuResolutionHomomorphism<U, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: ChainComplex<Algebra = CC1::Algebra> + 'static,
{
    fn prepare(&self, b: Bidegree) -> Option<LiftRequest<'_>> {
        let input = b + self.shift;
        if input.s() < self.shift.s()
            || !self.source.has_computed_bidegree(input)
            || !self.target.has_computed_bidegree(b)
        {
            return None;
        }
        match self.prepare_step(input, None) {
            StepPrep::Done(_) => None,
            StepPrep::NeedsLift(mut pending) => {
                let inputs = std::mem::take(&mut pending.fdx_vectors);
                Some(LiftRequest {
                    inputs,
                    finish: Box::new(move |results| {
                        self.finish_step(&mut pending, results);
                    }),
                })
            }
        }
    }

    fn lifts_through(&self, target: &dyn std::any::Any) -> bool {
        target
            .downcast_ref::<CC2>()
            .is_some_and(|target| std::ptr::eq(target, &*self.target))
    }

    fn prepare_batched(&self, b: Bidegree) -> crate::lift::Prepared<'_> {
        use crate::lift::Prepared;
        let input = b + self.shift;
        if input.s() < self.shift.s()
            || !self.source.has_computed_bidegree(input)
            || !self.target.has_computed_bidegree(b)
        {
            return Prepared::Ready(None);
        }
        match self.prepare_step_inner(input, None, true) {
            Ok(StepPrep::Done(_)) => Prepared::Ready(None),
            Ok(StepPrep::NeedsLift(mut pending)) => {
                let inputs = std::mem::take(&mut pending.fdx_vectors);
                Prepared::Ready(Some(crate::lift::LiftRequest {
                    inputs,
                    finish: Box::new(move |results| {
                        self.finish_step(&mut pending, results);
                    }),
                }))
            }
            Err(step) => Prepared::Deferred(Box::new(DeferredLift { hom: self, step })),
        }
    }
}

impl<const U: bool, CC1, CC2> MuResolutionHomomorphism<U, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: AugmentedChainComplex<Algebra = CC1::Algebra>,
{
    pub fn from_class(
        name: String,
        source: Arc<CC1>,
        target: Arc<CC2>,
        shift: Bidegree,
        class: &[u32],
    ) -> Self {
        let result = Self::new(name, source, target, shift);

        let num_gens = result
            .source
            .module(shift.s())
            .number_of_gens_in_degree(shift.t());
        assert_eq!(num_gens, class.len());

        let mut matrix = Matrix::new(result.source.prime(), num_gens, 1);
        for (k, &v) in class.iter().enumerate() {
            matrix.row_mut(k).set_entry(0, v);
        }

        result.extend_step(shift, Some(&matrix));
        result
    }

    /// Extend the [`MuResolutionHomomorphism`] to be defined on `(input_s, input_t)`. The resulting
    /// homomorphism `f` is a chain map such that if `g` is the `k`th generator in the source such
    /// that `d(g) = 0`, then the image of `f(g)` in the augmentation of the target is the `k`th
    /// row of `extra_images`.
    ///
    /// The user should call this function explicitly to manually define the chain map where the
    /// chain complex is not exact, and then call [`MuResolutionHomomorphism::extend_all`] to extend
    /// the rest by exactness.
    pub fn extend_step(&self, input: Bidegree, extra_images: Option<&Matrix>) -> Range<i32> {
        self.extend_step_raw(
            input,
            extra_images.map(|m| {
                let p = self.target.prime();
                let output = input - self.shift;

                let mut outputs =
                    vec![
                        FpVector::new(p, self.target.module(output.s()).dimension(output.t()));
                        m.rows()
                    ];
                let chain_map = self.target.chain_map(output.s());
                chain_map.compute_auxiliary_data_through_degree(output.t());
                for (output_vec, input) in std::iter::zip(&mut outputs, m.iter()) {
                    assert!(chain_map.apply_quasi_inverse(
                        output_vec.as_slice_mut(),
                        output.t(),
                        input,
                    ));
                }
                outputs
            }),
        )
    }
}

impl<const U: bool, CC1, CC2> MuResolutionHomomorphism<U, CC1, CC2>
where
    CC1: AugmentedChainComplex + FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: AugmentedChainComplex<Algebra = CC1::Algebra>,
    CC1::TargetComplex: BoundedChainComplex,
    CC2::TargetComplex: BoundedChainComplex,
{
    /// Construct a chain map that lifts a given module homomorphism.
    pub fn from_module_homomorphism(
        name: String,
        source: Arc<CC1>,
        target: Arc<CC2>,
        f: &impl ModuleHomomorphism<
            Source = <<CC1 as AugmentedChainComplex>::TargetComplex as ChainComplex>::Module,
            Target = <<CC2 as AugmentedChainComplex>::TargetComplex as ChainComplex>::Module,
        >,
    ) -> Self {
        assert_eq!(source.target().max_s(), 1);
        assert_eq!(target.target().max_s(), 1);

        let source_module = source.target().module(0);
        let target_module = target.target().module(0);
        assert!(Arc::ptr_eq(&source_module, &f.source()));
        assert!(Arc::ptr_eq(&target_module, &f.target()));

        let p = source.prime();
        let shift = Bidegree::s_t(0, f.degree_shift());

        let max_degree = source_module.max_generator_degree().expect(
            "MuResolutionHomomorphism::from_module_homomorphism requires finite \
             max_generator_degree",
        );

        let hom = Self::new(name, source, target, shift);

        source_module.compute_basis(max_degree);
        target_module.compute_basis(shift.t() + max_degree);

        let max = Bidegree::s_t(0, max_degree);
        hom.source.compute_through_bidegree(max);
        hom.target.compute_through_bidegree(max + shift);

        for t in source_module.min_degree()..=max_degree {
            let mut m = Matrix::new(
                p,
                source_module.dimension(t),
                target_module.dimension(t + shift.t()),
            );

            f.get_matrix(m.as_slice_mut(), t);
            hom.extend_step(Bidegree::s_t(0, t), Some(&m));
        }
        hom
    }
}

impl<const U: bool, CC1, CC2> MuResolutionHomomorphism<U, CC1, CC2>
where
    CC1: FreeChainComplex<U>,
    CC1::Algebra: MuAlgebra<U>,
    CC2: FreeChainComplex<U, Algebra = CC1::Algebra>,
{
    /// Given a chain map $f: C \to C'$ between free chain complexes, apply
    /// $$ \Hom(f, k): \Hom(C', k) \to \Hom(C, k) $$
    /// to the specified generator of $\Hom(C', k)$.
    pub fn act(&self, mut result: FpSliceMut, coef: u32, g: BidegreeGenerator) {
        let source = g.degree() + self.shift;

        assert_eq!(
            result.as_slice().len(),
            self.source
                .module(source.s())
                .number_of_gens_in_degree(source.t())
        );

        let target_module = self.target.module(g.s());

        let map = self.get_map(source.s());
        let j = target_module.operation_generator_to_index(0, 0, g.t(), g.idx());
        for i in 0..result.as_slice().len() {
            result.add_basis_element(i, coef * map.output(source.t(), i).entry(j));
        }
    }
}

// The secondary lift of a `ResolutionHomomorphism` lives here, beside the primary object it lifts,
// rather than in the monolithic `secondary` module. This keeps `secondary.rs` to the shared lift
// machinery and pairs each variant with its primary for locality. The module is `pub(crate)`;
// `SecondaryResolutionHomomorphism` is re-exported from `crate::secondary` so the public API path is
// unchanged.
pub(crate) mod secondary {
    use std::sync::Arc;

    use algebra::{
        module::{Module, homomorphism::ModuleHomomorphism},
        pair_algebra::PairAlgebra,
    };
    use dashmap::DashMap;
    use fp::{
        matrix::{Matrix, Subquotient},
        vector::{FpSlice, FpSliceMut, FpVector},
    };
    use itertools::Itertools;
    use once::OnceBiVec;
    use sseq::coordinates::{Bidegree, BidegreeGenerator, BidegreeRange};

    use super::ResolutionHomomorphism;
    use crate::{
        chain_complex::FreeChainComplex,
        save::SaveDirectory,
        secondary::{
            CompositeData, LAMBDA_BIDEGREE, SecondaryHomotopy, SecondaryLift, SecondaryResolution,
        },
    };

    // Rustdoc ICE's when trying to document this struct. See
    // https://github.com/rust-lang/rust/issues/91380
    #[doc(hidden)]
    pub struct SecondaryResolutionHomomorphism<
        CC1: FreeChainComplex,
        CC2: FreeChainComplex<Algebra = CC1::Algebra>,
    >
    where
        CC1::Algebra: PairAlgebra,
    {
        source: Arc<SecondaryResolution<CC1>>,
        target: Arc<SecondaryResolution<CC2>>,
        underlying: Arc<ResolutionHomomorphism<CC1, CC2>>,
        /// input s -> homotopy
        homotopies: OnceBiVec<SecondaryHomotopy<CC1::Algebra>>,
        intermediates: DashMap<BidegreeGenerator, FpVector>,
    }

    impl<CC1: FreeChainComplex, CC2: FreeChainComplex<Algebra = CC1::Algebra>> SecondaryLift
        for SecondaryResolutionHomomorphism<CC1, CC2>
    where
        CC1::Algebra: PairAlgebra,
    {
        type Algebra = CC1::Algebra;
        type Source = CC1;
        type Target = CC2;
        type Underlying = ResolutionHomomorphism<CC1, CC2>;

        fn underlying(&self) -> Arc<Self::Underlying> {
            Arc::clone(&self.underlying)
        }

        fn algebra(&self) -> Arc<Self::Algebra> {
            self.source.algebra()
        }

        fn source(&self) -> Arc<Self::Source> {
            Arc::clone(&self.source.underlying())
        }

        fn target(&self) -> Arc<Self::Target> {
            Arc::clone(&self.target.underlying())
        }

        fn shift(&self) -> Bidegree {
            self.underlying.shift + Bidegree::s_t(1, 0)
        }

        fn max(&self) -> BidegreeRange<'_, Self> {
            BidegreeRange::new(
                self,
                self.underlying.next_homological_degree(),
                &|selff, s| {
                    std::cmp::min(
                        selff.underlying.get_map(s).next_degree(),
                        std::cmp::min(
                            selff.source.homotopies[s].homotopies.next_degree(),
                            if s == selff.shift().s() {
                                i32::MAX
                            } else {
                                selff.target.homotopies[s + 1 - selff.shift().s()]
                                    .composites
                                    .max_degree()
                                    + selff.shift().t()
                                    + 1
                            },
                        ),
                    )
                },
            )
        }

        fn homotopies(&self) -> &OnceBiVec<SecondaryHomotopy<Self::Algebra>> {
            &self.homotopies
        }

        fn intermediates(&self) -> &DashMap<BidegreeGenerator, FpVector> {
            &self.intermediates
        }

        fn save_dir(&self) -> &SaveDirectory {
            self.underlying.save_dir()
        }

        fn composite(&self, s: i32) -> CompositeData<Self::Algebra> {
            let p = self.prime();
            // This is -1 mod p^2
            let neg_1 = p * p - 1;

            let d_source = self.source.underlying().differential(s);
            let d_target = self
                .target
                .underlying()
                .differential(s + 1 - self.shift().s());

            let c1 = self.underlying.get_map(s);
            let c0 = self.underlying.get_map(s - 1);

            vec![(neg_1, d_source, c0), (1, c1, d_target)]
        }

        fn compute_intermediate(&self, g: BidegreeGenerator) -> FpVector {
            let p = self.prime();
            let neg_1 = p - 1;
            let shifted_b = g.degree() - self.shift();
            let target = self.target().module(shifted_b.s() - 1);

            let mut result = FpVector::new(p, target.dimension(shifted_b.t() - 1));
            let d = self.source().differential(g.s());

            self.homotopies[g.s() - 1].act(
                result.as_slice_mut(),
                neg_1,
                g.t(),
                d.output(g.t(), g.idx()).as_slice(),
                false,
            );
            self.target.homotopy(shifted_b.s() + 1).act(
                result.as_slice_mut(),
                neg_1,
                shifted_b.t(),
                self.underlying
                    .get_map(g.s())
                    .output(g.t(), g.idx())
                    .as_slice(),
                true,
            );
            self.underlying.get_map(g.s() - 2).apply(
                result.as_slice_mut(),
                1,
                g.t() - 1,
                self.source
                    .homotopy(g.s())
                    .homotopies
                    .output(g.t(), g.idx())
                    .as_slice(),
            );

            result
        }
    }

    impl<CC1: FreeChainComplex, CC2: FreeChainComplex<Algebra = CC1::Algebra>>
        SecondaryResolutionHomomorphism<CC1, CC2>
    where
        CC1::Algebra: PairAlgebra,
    {
        pub fn new(
            source: Arc<SecondaryResolution<CC1>>,
            target: Arc<SecondaryResolution<CC2>>,
            underlying: Arc<ResolutionHomomorphism<CC1, CC2>>,
        ) -> Self {
            assert!(Arc::ptr_eq(&underlying.source, &source.underlying()));
            assert!(Arc::ptr_eq(&underlying.target, &target.underlying()));

            Self {
                source,
                target,
                homotopies: OnceBiVec::new(underlying.shift.s() + 1),
                underlying,
                intermediates: DashMap::new(),
            }
        }

        pub fn name(&self) -> String {
            let name = self.underlying.name();
            if name.starts_with('[') || name.starts_with('λ') {
                name.to_owned()
            } else {
                format!("[{name}]")
            }
        }

        pub fn homotopy(&self, s: i32) -> &SecondaryHomotopy<CC1::Algebra> {
            &self.homotopies[s]
        }

        pub fn secondary_source(&self) -> Arc<SecondaryResolution<CC1>> {
            Arc::clone(&self.source)
        }

        pub fn secondary_target(&self) -> Arc<SecondaryResolution<CC2>> {
            Arc::clone(&self.target)
        }

        /// A version of [`hom_k`] but with a non-trivial λ part.
        pub fn hom_k_with<'a>(
            &self,
            lambda_part: Option<&ResolutionHomomorphism<CC1, CC2>>,
            e3_page: Option<&dyn Fn(Bidegree) -> Option<Subquotient>>,
            b: Bidegree,
            inputs: impl Iterator<Item = FpSlice<'a>>,
            outputs: impl Iterator<Item = FpSliceMut<'a>>,
        ) {
            let source = b + self.shift() - Bidegree::s_t(1, 0);
            let lambda_source = source + LAMBDA_BIDEGREE;

            let p = self.prime();
            let h_0 = self.algebra().p_tilde();

            let source_num_gens = self.source().number_of_gens_in_bidegree(source);
            let lambda_num_gens = self.source().number_of_gens_in_bidegree(lambda_source);

            let m0 = self.underlying.get_map(source.s()).hom_k(b.t());
            let mut m1 =
                Matrix::from_vec(p, &self.homotopy(lambda_source.s()).homotopies.hom_k(b.t()));
            if let Some(lambda_part) = lambda_part {
                m1 += &Matrix::from_vec(p, &lambda_part.get_map(lambda_source.s()).hom_k(b.t()));
            }

            // The multiplication by p map
            let mp = Matrix::from_vec(
                p,
                &self
                    .source()
                    .filtration_one_product(1, h_0, source)
                    .unwrap(),
            );

            let sign = if (self.underlying.shift.s() * b.t()) % 2 == 1 {
                p * p - 1
            } else {
                1
            };
            let filtration_one_sign = if (b.t() % 2) == 1 { p - 1 } else { 1 };

            // The quotient of the E3 page at the λ-part's source (the image of d₂)
            // reduces the λ part of each output.
            let lambda_page = e3_page.and_then(|f| f(lambda_source));

            let mut scratch0: Vec<u32> = Vec::new();
            for (input, mut out) in inputs.zip_eq(outputs) {
                scratch0.clear();
                scratch0.resize(source_num_gens, 0);
                for (i, v) in input.iter_nonzero() {
                    scratch0
                        .iter_mut()
                        .zip_eq(&m0[i])
                        .for_each(|(a, b)| *a += v * b * sign);
                    out.slice_mut(source_num_gens, source_num_gens + lambda_num_gens)
                        .add(m1.row(i), (v * sign) % p);
                }
                for (i, v) in scratch0.iter().enumerate() {
                    out.add_basis_element(i, *v % p);

                    let extra = *v / p;
                    out.slice_mut(source_num_gens, source_num_gens + lambda_num_gens)
                        .add(mp.row(i), (extra * filtration_one_sign) % p);
                }
                if let Some(lambda_page) = &lambda_page {
                    lambda_page.reduce_by_quotient(
                        out.slice_mut(source_num_gens, source_num_gens + lambda_num_gens),
                    );
                }
            }
        }

        /// Compute the induced map on Mod_{C\lambda^2} homotopy groups. This only computes it on
        /// standard lifts on elements in Ext. `outputs` is an iterator of `FpSliceMut`s whose lengths
        /// are equal to the total dimension of `(s + shift_s, t + shift_t)` and `(s + shift_s + 1, t +
        /// shift_t + 1)`. The first chunk records the Ext part of the result, and the second chunk
        /// records the λ part of the result.
        ///
        /// This reduces the λ part of the result by the image of d₂.
        ///
        /// # Arguments
        /// - `e3_page`: the $E_3$ subquotient as a function of bidegree. If present, the
        ///   quotient (image of $d_2$) at the λ-part's source reduces the λ part of the
        ///   result. Passed as a function so the caller need not reconstruct that bidegree.
        pub fn hom_k<'a>(
            &self,
            e3_page: Option<&dyn Fn(Bidegree) -> Option<Subquotient>>,
            b: Bidegree,
            inputs: impl Iterator<Item = FpSlice<'a>>,
            outputs: impl Iterator<Item = FpSliceMut<'a>>,
        ) {
            self.hom_k_with(None, e3_page, b, inputs, outputs);
        }

        /// Given an element b whose product with this is null, find the element whose $d_2$ hits the
        /// λ part of the composition.
        ///
        /// # Arguments:
        /// - `sseq`: spectral sequence object of the source
        pub fn product_nullhomotopy(
            &self,
            lambda_part: Option<&ResolutionHomomorphism<CC1, CC2>>,
            sseq: &sseq::Sseq<2, sseq::Adams>,
            b: Bidegree,
            class: FpSlice,
        ) -> FpVector {
            let p = self.prime();
            let shift = self.underlying.shift;

            let result_num_gens = self
                .source()
                .number_of_gens_in_bidegree(shift + b - Bidegree::s_t(1, 0));

            let lambda_num_gens = self
                .source()
                .number_of_gens_in_bidegree(b + shift + LAMBDA_BIDEGREE);

            let lower_num_gens = self.source().number_of_gens_in_bidegree(b + shift);

            let target_num_gens = self.target().number_of_gens_in_bidegree(b);
            let target_lambda_num_gens = self
                .target()
                .number_of_gens_in_bidegree(b + LAMBDA_BIDEGREE);

            let mut output_class = FpVector::new(p, result_num_gens);
            if result_num_gens == 0 || lambda_num_gens == 0 {
                return output_class;
            }

            let mut prod_value = FpVector::new(p, lower_num_gens + lambda_num_gens);
            self.hom_k_with(
                lambda_part,
                None,
                b,
                [class.restrict(0, target_num_gens)].into_iter(),
                [prod_value.as_slice_mut()].into_iter(),
            );
            assert!(prod_value.slice(0, lower_num_gens).is_zero());

            let matrix = Matrix::from_vec(
                p,
                &self
                    .underlying
                    .get_map((b + shift + LAMBDA_BIDEGREE).s())
                    .hom_k((b + LAMBDA_BIDEGREE).t()),
            );
            matrix.apply(
                prod_value.slice_mut(lower_num_gens, lower_num_gens + lambda_num_gens),
                1,
                class.restrict(target_num_gens, target_num_gens + target_lambda_num_gens),
            );

            let diff_source = b + shift - Bidegree::n_s(-1, 1);
            sseq.differentials(diff_source)[2].quasi_inverse(
                output_class.as_slice_mut(),
                prod_value.slice(lower_num_gens, lower_num_gens + lambda_num_gens),
            );

            output_class
        }
    }
}

/// Measurement only: when `EXT_LIFT_TERMS_LOG` names a file, every lift step's `prepare` appends
/// `in_s,in_t,out_s,out_t,gens,dx_terms,products,max_fw,op_deg_sum,fdx_nnz,fdx_ns,end_s`, where
/// `products` = Σ over the terms `a·w` of every `d x` of `nnz(f(w))` (the Milnor products
/// `f(d x)` costs), `max_fw` the largest such `nnz(f(w))`, `op_deg_sum` = Σ `nnz(f(w))·|a|`,
/// `fdx_nnz` the non-zeros of the results, `fdx_ns` the thread time computing them, and `end_s`
/// the wall time since the first step, then `distinct_terms` (the (a, w) pairs over all the step's
/// `d x`) and `shared_products` (Σ `nnz(f(w))` over those pairs).
///
/// `EXT_LIFT_PAIRS_LOG=<file>` additionally samples one product `a·b` in every
/// `EXT_LIFT_PAIRS_EVERY` (default 1000000), `b` a term of `f(w)`, as `|a|,a,|f(w)|,b` with `b`
/// written as its module element (operation and generator).
mod terms_log {
    use std::{
        io::Write,
        sync::{
            LazyLock, Mutex,
            atomic::{AtomicU64, Ordering::Relaxed},
        },
        time::{Duration, Instant},
    };

    use sseq::coordinates::Bidegree;

    struct Log {
        origin: Instant,
        out: Mutex<std::io::BufWriter<std::fs::File>>,
    }

    static LOG: LazyLock<Option<Log>> = LazyLock::new(|| {
        let path = std::env::var_os("EXT_LIFT_TERMS_LOG")?;
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(
            out,
            "in_s,in_t,out_s,out_t,gens,dx_terms,products,max_fw,op_deg_sum,fdx_nnz,fdx_ns,end_s,\
             distinct_terms,shared_products"
        )
        .unwrap();
        Some(Log {
            origin: Instant::now(),
            out: Mutex::new(out),
        })
    });

    pub(super) fn enabled() -> bool {
        LOG.is_some()
    }

    static PAIRS: LazyLock<Option<(u64, Mutex<std::io::BufWriter<std::fs::File>>)>> =
        LazyLock::new(|| {
            let path = std::env::var_os("EXT_LIFT_PAIRS_LOG")?;
            let every = std::env::var("EXT_LIFT_PAIRS_EVERY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000_000);
            let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
            writeln!(out, "deg_a,a,deg_b,b").unwrap();
            Some((every, Mutex::new(out)))
        });
    static PAIR_COUNT: AtomicU64 = AtomicU64::new(0);

    pub(super) fn sample_pair() -> bool {
        PAIRS
            .as_ref()
            .is_some_and(|(every, _)| PAIR_COUNT.fetch_add(1, Relaxed) % every == 0)
    }

    pub(super) fn write_pair(deg_a: i32, a: &str, deg_b: i32, b: &str) {
        if let Some((_, out)) = &*PAIRS {
            let mut out = out.lock().unwrap();
            writeln!(out, "{deg_a},\"{a}\",{deg_b},\"{b}\"").unwrap();
            out.flush().unwrap();
        }
    }

    #[derive(Default)]
    pub(super) struct Cell {
        dx_terms: AtomicU64,
        products: AtomicU64,
        max_fw: AtomicU64,
        op_deg_sum: AtomicU64,
        fdx_nnz: AtomicU64,
        fdx_ns: AtomicU64,
        distinct_terms: AtomicU64,
        shared_products: AtomicU64,
    }

    impl Cell {
        pub(super) fn add(
            &self,
            dx_terms: u64,
            products: u64,
            max_fw: u64,
            op_deg_sum: u64,
            fdx_nnz: u64,
            elapsed: Duration,
        ) {
            self.dx_terms.fetch_add(dx_terms, Relaxed);
            self.products.fetch_add(products, Relaxed);
            self.max_fw.fetch_max(max_fw, Relaxed);
            self.op_deg_sum.fetch_add(op_deg_sum, Relaxed);
            self.fdx_nnz.fetch_add(fdx_nnz, Relaxed);
            self.fdx_ns.fetch_add(elapsed.as_nanos() as u64, Relaxed);
        }

        pub(super) fn add_shared(&self, distinct_terms: u64, shared_products: u64) {
            self.distinct_terms.fetch_add(distinct_terms, Relaxed);
            self.shared_products.fetch_add(shared_products, Relaxed);
        }

        pub(super) fn write(&self, input: Bidegree, output: Bidegree, gens: usize) {
            let Some(log) = &*LOG else { return };
            let g = |a: &AtomicU64| a.load(Relaxed);
            let mut out = log.out.lock().unwrap();
            writeln!(
                out,
                "{},{},{},{},{gens},{},{},{},{},{},{},{:.3},{},{}",
                input.s(),
                input.t(),
                output.s(),
                output.t(),
                g(&self.dx_terms),
                g(&self.products),
                g(&self.max_fw),
                g(&self.op_deg_sum),
                g(&self.fdx_nnz),
                g(&self.fdx_ns),
                log.origin.elapsed().as_secs_f64(),
                g(&self.distinct_terms),
                g(&self.shared_products),
            )
            .unwrap();
            // Flush often enough that a killed run keeps its data.
            if log.origin.elapsed().as_secs() % 10 == 0 {
                out.flush().unwrap();
            }
        }
    }
}

/// Lift steps' products `a·f(w)` as Milnor multiply batches ([`algebra::milnor_algebra::batch`]),
/// which run on a GPU when there is one.
///
/// With `EXT_LIFT_BATCH` set to `1`, or unset with a GPU present, a lift step through
/// [`crate::lift::MultiLift`] leaves its products to the sweep
/// ([`crate::lift::Liftable::prepare_batched`]), which multiplies those of every map at a bidegree
/// together, `EXT_LIFT_BATCH_TERMS` terms at a time. `EXT_LIFT_BATCH=0` computes every step's
/// products on its own, on the CPU. `EXT_LIFT_BATCH_CHECK=1` compares every batched step with that.
pub mod lift_batch {
    use std::{
        collections::HashMap,
        sync::{
            Arc, LazyLock,
            atomic::{AtomicU64, Ordering},
        },
        time::Instant,
    };

    pub use algebra::milnor_algebra::batch::BatchProduct as Product;
    use algebra::{
        MilnorAlgebra,
        milnor_algebra::batch,
        module::{FreeModule, Module},
    };
    use fp::{prime::TWO, vector::FpVector};

    static ENABLED: LazyLock<bool> = LazyLock::new(|| match std::env::var("EXT_LIFT_BATCH") {
        Ok(v) => v != "0",
        Err(_) => batch::gpu_available(),
    });
    static CHECK: LazyLock<bool> =
        LazyLock::new(|| std::env::var("EXT_LIFT_BATCH_CHECK").is_ok_and(|v| v != "0"));

    /// Batches multiplied, their terms, and nanoseconds building them and multiplying them.
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static TERMS: AtomicU64 = AtomicU64::new(0);
    static BUILD_NS: AtomicU64 = AtomicU64::new(0);
    static MULTIPLY_NS: AtomicU64 = AtomicU64::new(0);

    /// Whether lift steps leave their products to batches.
    pub fn enabled() -> bool {
        *ENABLED
    }

    pub(super) fn check() -> bool {
        *CHECK
    }

    /// `EXT_STEP_LOG_MIN_TERMS` (default 10^6): the fewest product terms a batched step has for the
    /// step log to record it.
    pub(crate) fn step_log_min_terms() -> usize {
        static MIN: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("EXT_STEP_LOG_MIN_TERMS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000_000)
        });
        *MIN
    }

    /// `EXT_LIFT_BATCH_PRODUCTS` (default 2 * 10^7): the most products one multiply takes. Each is
    /// a few dozen bytes, so this bounds the memory of a bidegree's chunk.
    pub fn batch_products() -> usize {
        static CAP: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("EXT_LIFT_BATCH_PRODUCTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20_000_000)
        });
        *CAP
    }

    /// `EXT_LIFT_BATCH_TERMS` (default 10^8): the most terms one multiply takes.
    pub fn batch_terms() -> usize {
        static CAP: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("EXT_LIFT_BATCH_TERMS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100_000_000)
        });
        *CAP
    }

    /// One line with the counters, for logs.
    /// The product terms multiplied so far and the thread time spent multiplying them, in
    /// nanoseconds.
    pub fn multiply_totals() -> (u64, u64) {
        (
            TERMS.load(Ordering::Relaxed),
            MULTIPLY_NS.load(Ordering::Relaxed),
        )
    }

    pub fn summary() -> String {
        format!(
            "lift_batch calls={} terms={} build_s={:.1} multiply_s={:.1} gpu_launches={}",
            CALLS.load(Ordering::Relaxed),
            TERMS.load(Ordering::Relaxed),
            BUILD_NS.load(Ordering::Relaxed) as f64 / 1e9,
            MULTIPLY_NS.load(Ordering::Relaxed) as f64 / 1e9,
            batch::gpu_launches(),
        )
    }

    /// A step's products: row `k` is `f(d x_k)`, holding `a·f(w)` for every term `a·w` of
    /// `d x_k`.
    ///
    /// A row per distinct term instead, summed into the `f(d x)` afterwards, multiplies each
    /// product once but reads back a full-width row per term, almost all zeros: on a GPU that
    /// readback, not the multiply, was the cost.
    pub struct SharedBatch {
        pub algebra: Arc<MilnorAlgebra>,
        pub products: Vec<Product>,
        pub num_rows: usize,
    }

    /// The algebra of a step's products, when both its modules are free over the Milnor algebra
    /// and that algebra takes batches.
    pub(super) fn applicable(
        source: &dyn std::any::Any,
        target: &dyn std::any::Any,
    ) -> Option<Arc<MilnorAlgebra>> {
        source.downcast_ref::<FreeModule<MilnorAlgebra>>()?;
        let algebra = target
            .downcast_ref::<FreeModule<MilnorAlgebra>>()?
            .algebra();
        batch::batch_applicable(&algebra).then_some(algebra)
    }

    /// The batch of a step whose `d x_k` have the source basis indices `dx[k]` (ascending), each
    /// `f(w)`'s blocks built once and shared; `None` unless both modules are free over the Milnor
    /// algebra and it takes batches, or when nothing is multiplied.
    pub(super) fn build_shared(
        source: &dyn std::any::Any,
        target: &dyn std::any::Any,
        min_degree: i32,
        shift: i32,
        t: i32,
        dx: &[Vec<usize>],
        f_prev: &dyn Fn(i32, usize) -> FpVector,
    ) -> Option<SharedBatch> {
        let start = Instant::now();
        let source = source.downcast_ref::<FreeModule<MilnorAlgebra>>()?;
        let target = target.downcast_ref::<FreeModule<MilnorAlgebra>>()?;
        let algebra = target.algebra();
        if !batch::batch_applicable(&algebra) {
            return None;
        }
        let out_t = t - shift;
        type Blocks = Arc<Vec<(i32, batch::Terms, usize)>>;
        let mut blocks: HashMap<(i32, usize), Blocks> = HashMap::new();
        let mut products = vec![];
        // A row's terms are ascending, so a cursor decodes them without a search per term.
        let mut cursor = source.opgen_cursor(t);
        for (row, terms) in dx.iter().enumerate() {
            for &idx in terms {
                let og = cursor.get(idx);
                if og.generator_degree < min_degree || og.operation_degree == 0 {
                    continue;
                }
                let w = blocks
                    .entry((og.generator_degree, og.generator_index))
                    .or_insert_with(|| {
                        // f(w) is kept once and its blocks read as bit ranges of it: listing its set
                        // bits (about a fifth of them) as indices took ~14 times its memory.
                        let fw = Arc::new(f_prev(og.generator_degree, og.generator_index));
                        let fw_degree = og.generator_degree - shift;
                        Arc::new(
                            target
                                .iter_slices(fw_degree, fw.as_slice())
                                .filter_map(|(h_deg, h_idx, op_deg, slice)| {
                                    let start = target.generator_offset(fw_degree, h_deg, h_idx);
                                    let ts = batch::Terms::bits(
                                        Arc::clone(&fw),
                                        start,
                                        start + slice.len(),
                                    );
                                    (!ts.is_empty()).then(|| {
                                        (op_deg, ts, target.generator_offset(out_t, h_deg, h_idx))
                                    })
                                })
                                .collect(),
                        )
                    })
                    .clone();
                for (s_degree, ts, out_offset) in w.iter() {
                    products.push(Product {
                        r_degree: og.operation_degree,
                        r_idx: batch::narrow(og.operation_index),
                        s_degree: *s_degree,
                        term_indices: ts.clone(),
                        row: batch::narrow(row),
                        out_offset: batch::narrow(*out_offset),
                    });
                }
            }
        }
        if products.is_empty() {
            return None;
        }
        BUILD_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(SharedBatch {
            algebra,
            products,
            num_rows: dx.len(),
        })
    }

    /// Multiply `products` over `num_rows` rows of `out_cols` columns, on a GPU if there is one.
    pub fn multiply(
        algebra: &MilnorAlgebra,
        out_cols: usize,
        num_rows: usize,
        products: &[Product],
    ) -> Vec<FpVector> {
        let start = Instant::now();
        let degree = products
            .iter()
            .map(|p| p.r_degree + p.s_degree)
            .max()
            .unwrap_or(0);
        algebra.compute_seqno_tables(degree);
        let out = batch::multiply_batch(algebra, out_cols, None, num_rows, products);
        let rows = out
            .iter_rows()
            .map(|limbs| vector(out_cols, limbs))
            .collect();
        MULTIPLY_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        TERMS.fetch_add(
            products.iter().map(|p| p.term_indices.len() as u64).sum(),
            Ordering::Relaxed,
        );
        rows
    }

    /// An `FpVector` of length `len` from bit-packed `u32` limbs (entry `i` at limb `i / 32`, bit
    /// `i % 32`), read as the little-endian `u64` limbs `FpVector::from_bytes` takes.
    fn vector(len: usize, limbs: &[u32]) -> FpVector {
        let mut bytes = Vec::with_capacity(len.div_ceil(64) * 8);
        for &l in limbs {
            bytes.extend_from_slice(&l.to_le_bytes());
        }
        bytes.resize(len.div_ceil(64) * 8, 0);
        FpVector::from_bytes(TWO, len, &mut &bytes[..]).expect("limbs cover the row")
    }
}

/// Chain maps whose steps are read back from the save store on demand, under a byte budget
/// (`EXT_LIFT_MAP_BUDGET_GIB`), rather than all kept in memory. Only maps with a save store, whose
/// steps are saved as they are computed, read lazily.
pub mod lazy_maps {
    use std::sync::{Arc, LazyLock};

    use algebra::module::homomorphism::OutputCache;

    static CACHE: LazyLock<Option<Arc<OutputCache>>> = LazyLock::new(|| {
        let gib: f64 = std::env::var("EXT_LIFT_MAP_BUDGET_GIB")
            .ok()?
            .parse()
            .unwrap_or_else(|e| panic!("EXT_LIFT_MAP_BUDGET_GIB is not a number: {e}"));
        assert!(
            gib.is_finite() && gib > 0.0,
            "EXT_LIFT_MAP_BUDGET_GIB must be a positive number of GiB"
        );
        Some(OutputCache::new((gib * (1u64 << 30) as f64) as u64))
    });

    /// The cache every lazily read chain map shares, if `EXT_LIFT_MAP_BUDGET_GIB` is set.
    pub fn cache() -> Option<&'static Arc<OutputCache>> {
        CACHE.as_ref()
    }
}
