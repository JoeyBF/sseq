//! The $Q$-weight grading of Ext at odd primes.
//!
//! At an odd prime $p$, the Milnor basis element $Q(E) P(R)$ has *$Q$-weight* $|E|$, the number of
//! Milnor primitives $Q_i$ it contains. Products preserve the $Q$-weight. Dually, $\mathcal{A}_* =
//! E(\tau_0, \tau_1, \ldots) \otimes P(\xi_1, \xi_2, \ldots)$ is graded by the number of $\tau$'s,
//! and the coproduct $\Delta \tau_k = \tau_k \otimes 1 + \sum_i \xi_{k - i}^{p^i} \otimes \tau_i$
//! preserves it. In the admissible basis, the $Q$-weight of $\beta^{\epsilon_0} P^{s_1}
//! \beta^{\epsilon_1} \cdots$ is the number of Bocksteins, since $\beta = Q_0$ and $P^i$ has
//! weight 0. So the Steenrod algebra is bigraded, and so is $\mathrm{Ext}_\mathcal{A}^{s, t}(M,
//! \mathbb{F}_p)$ whenever $M$ is.
//!
//! The minimal resolutions in this crate are automatically homogeneous with respect to this
//! weight, and so are the unstable ones: in the Milnor basis, excess is $|E| + 2 \sum r_i$, so the
//! relations of a free unstable module are spanned by homogeneous elements. Every matrix that is
//! row reduced is block diagonal by weight up to a permutation of its columns, and row reduction
//! never mixes rows of different weights. [`QWeights`] reads the weight of each generator off its
//! differential and its augmentation, and returns an error instead of a wrong answer if anything
//! fails to be homogeneous.
//!
//! The weights of the module being resolved are determined by propagating along the action: an
//! element that is not hit by the action of a positive degree basis element on a lower degree
//! basis element gets weight 0, and $a \cdot x$ gets weight $w(a) + w(x)$. This requires the basis
//! of the module to be homogeneous. For example, for the Moore spectrum $C(p)$ the top class
//! $\beta x_0$ gets weight 1.
//!
//! At $p = 2$ there is no such grading: the dual Steenrod algebra is polynomial on the $\xi_i$ and
//! has no exterior part, and counting $\mathrm{Sq}^1$'s is not compatible with the relations (e.g.
//! $\mathrm{Sq}^1 \mathrm{Sq}^2 = \mathrm{Sq}^3$). [`QWeights::new`] returns an error there.

use std::collections::BTreeMap;

use algebra::{
    AdemAlgebra, Algebra, MilnorAlgebra, MuAlgebra, SteenrodAlgebra,
    module::{Module, homomorphism::ModuleHomomorphism},
};
use anyhow::{Context, anyhow, ensure};
use fp::vector::FpVector;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

use crate::chain_complex::{AugmentedChainComplex, ChainComplex, FreeChainComplex};

/// An algebra whose basis elements have a $Q$-weight.
pub trait QWeightAlgebra: Algebra {
    /// The $Q$-weight of the basis element `idx` in degree `degree`.
    fn q_weight(&self, degree: i32, idx: usize) -> u32;
}

impl QWeightAlgebra for MilnorAlgebra {
    fn q_weight(&self, degree: i32, idx: usize) -> u32 {
        self.basis_element_from_index(degree, idx)
            .q_part
            .count_ones()
    }
}

impl QWeightAlgebra for AdemAlgebra {
    fn q_weight(&self, degree: i32, idx: usize) -> u32 {
        self.basis_element_from_index(degree, idx)
            .bocksteins
            .count_ones()
    }
}

impl QWeightAlgebra for SteenrodAlgebra {
    fn q_weight(&self, degree: i32, idx: usize) -> u32 {
        match self {
            Self::AdemAlgebra(a) => a.q_weight(degree, idx),
            Self::MilnorAlgebra(a) => a.q_weight(degree, idx),
        }
    }
}

/// Records `w` as the weight of `slot`, failing if it already has a different weight.
fn assign(slot: &mut Option<u32>, w: u32, what: impl FnOnce() -> String) -> anyhow::Result<()> {
    match *slot {
        None => *slot = Some(w),
        Some(old) if old == w => (),
        Some(old) => return Err(anyhow!("{} has weights {old} and {w}", what())),
    }
    Ok(())
}

/// Computes the $Q$-weights of the basis of `module` in degrees up to `max_degree`, indexed by
/// `[degree - module.min_degree()][index]`.
///
/// Basis elements not hit by the action of a positive degree basis element of the algebra on a
/// lower degree basis element get weight 0. Returns an error if the basis of the module is not
/// homogeneous.
pub fn module_q_weights<M>(module: &M, max_degree: i32) -> anyhow::Result<Vec<Vec<u32>>>
where
    M: Module + ?Sized,
    M::Algebra: QWeightAlgebra,
{
    propagate(module, max_degree, |_, _| Ok(None))
}

/// As [`module_q_weights`], but a basis element not hit by the action gets the weight `seed`
/// returns for it, or 0 if that is `None`. If `seed` returns a weight for an element that is hit
/// by the action, the two must agree.
fn propagate<M>(
    module: &M,
    max_degree: i32,
    mut seed: impl FnMut(i32, usize) -> anyhow::Result<Option<u32>>,
) -> anyhow::Result<Vec<Vec<u32>>>
where
    M: Module + ?Sized,
    M::Algebra: QWeightAlgebra,
{
    let algebra = module.algebra();
    let min_degree = module.min_degree();
    let max_degree = match module.max_degree() {
        Some(d) => std::cmp::min(d, max_degree),
        None => max_degree,
    };
    if max_degree < min_degree {
        return Ok(Vec::new());
    }
    algebra.compute_basis(max_degree - min_degree);
    module.compute_basis(max_degree);

    let mut weights: Vec<Vec<Option<u32>>> = (min_degree..=max_degree)
        .map(|d| vec![None; module.dimension(d)])
        .collect();
    let mut result = Vec::with_capacity(weights.len());

    for degree in min_degree..=max_degree {
        let mut current = Vec::with_capacity(module.dimension(degree));
        for (idx, &hit) in weights[(degree - min_degree) as usize].iter().enumerate() {
            let w = match (hit, seed(degree, idx)?) {
                (Some(a), Some(b)) if a != b => {
                    return Err(anyhow!(
                        "Basis element {idx} in degree {degree} of the module has weights {a} and \
                         {b}"
                    ));
                }
                (a, b) => a.or(b).unwrap_or(0),
            };
            current.push(w);
        }

        for (idx, &w) in current.iter().enumerate() {
            for op_deg in 1..=max_degree - degree {
                let target_degree = degree + op_deg;
                let target_dim = module.dimension(target_degree);
                if target_dim == 0 {
                    continue;
                }
                let mut v = FpVector::new(module.prime(), target_dim);
                for op_idx in 0..algebra.dimension(op_deg) {
                    v.set_to_zero();
                    module.act_on_basis(v.as_slice_mut(), 1, op_deg, op_idx, degree, idx);
                    let tw = w + algebra.q_weight(op_deg, op_idx);
                    for (j, _) in v.iter_nonzero() {
                        assign(
                            &mut weights[(target_degree - min_degree) as usize][j],
                            tw,
                            || format!("Basis element {j} in degree {target_degree} of the module"),
                        )
                        .context("The basis of the module is not Q-weight homogeneous")?;
                    }
                }
            }
        }
        result.push(current);
    }
    Ok(result)
}

/// The $Q$-weights of the generators of a minimal resolution.
pub struct QWeights {
    min_degree: i32,
    /// Indexed by `[s][t - min_degree][idx]`.
    weights: Vec<Vec<Vec<u32>>>,
}

impl QWeights {
    /// Computes the $Q$-weights of all generators of `res` in the bidegrees computed so far.
    ///
    /// Returns an error at $p = 2$, if the module being resolved does not have a homogeneous basis,
    /// or if the image of a generator under the differential and augmentation is not homogeneous.
    pub fn new<const U: bool, CC>(res: &CC) -> anyhow::Result<Self>
    where
        CC: FreeChainComplex<U> + AugmentedChainComplex,
        CC::Algebra: MuAlgebra<U> + QWeightAlgebra,
    {
        let p = res.prime();
        ensure!(p != 2, "Q-weights are only defined at odd primes");

        let algebra = res.algebra();
        let target = res.target();
        let min_degree = res.min_degree();
        let mut weights: Vec<Vec<Vec<u32>>> = Vec::new();
        let mut target_weights_prev: Vec<Vec<u32>> = Vec::new();
        // The weights of the complex being resolved are computed through the same degree for
        // every s, so that the differential of C can be looked up.
        let target_max_t = (0..res.next_homological_degree())
            .map(|s| res.module(s).max_computed_degree())
            .max()
            .unwrap_or(min_degree);

        for s in 0..res.next_homological_degree() {
            let module = res.module(s);
            let max_t = module.max_computed_degree();
            let target_module = target.module(s);
            let target_min = target_module.min_degree();
            // A basis element of C_s that is not hit by the action takes the weight of its
            // image under the differential of C, if that is nonzero.
            let target_weights = if s == 0 {
                module_q_weights(&*target_module, target_max_t)?
            } else {
                let d = target.differential(s);
                let prev = target.module(s - 1);
                let prev_min = prev.min_degree();
                let prev_weights = &target_weights_prev;
                propagate(&*target_module, target_max_t, |degree, idx| {
                    let mut v = FpVector::new(p, prev.dimension(degree));
                    d.apply_to_basis_element(v.as_slice_mut(), 1, degree, idx);
                    let mut w = None;
                    for (j, _) in v.iter_nonzero() {
                        assign(
                            &mut w,
                            prev_weights[(degree - prev_min) as usize][j],
                            || format!("The image of basis element {idx} in degree {degree}"),
                        )?;
                    }
                    Ok(w)
                })
                .with_context(|| format!("The complex being resolved at s = {s}"))?
            };
            let chain_map = res.chain_map(s);

            let mut row = Vec::new();
            for t in min_degree..=max_t {
                let b = Bidegree::s_t(s, t);
                let num_gens = module.number_of_gens_in_degree(t);
                let mut gen_weights = Vec::with_capacity(num_gens);

                let target_dim = target_module.dimension(t);
                let mut fx = FpVector::new(p, target_dim);

                for i in 0..num_gens {
                    let g = BidegreeGenerator::new(b, i);
                    let mut w = None;
                    let what = || format!("The image of generator {g}");

                    if s > 0 {
                        let prev = res.module(s - 1);
                        for (idx, _) in res.differential(s).output(t, i).iter_nonzero() {
                            let op_gen = prev.index_to_op_gen(t, idx);
                            let tw = weights[(s - 1) as usize]
                                [(op_gen.generator_degree - min_degree) as usize]
                                [op_gen.generator_index]
                                + algebra.q_weight(op_gen.operation_degree, op_gen.operation_index);
                            assign(&mut w, tw, what)?;
                        }
                    }

                    if target_dim > 0 {
                        fx.set_to_zero();
                        let idx = module.operation_generator_to_index(0, 0, t, i);
                        chain_map.apply_to_basis_element(fx.as_slice_mut(), 1, t, idx);
                        for (j, _) in fx.iter_nonzero() {
                            assign(&mut w, target_weights[(t - target_min) as usize][j], what)?;
                        }
                    }

                    gen_weights.push(w.with_context(|| {
                        format!("Generator {g} maps to zero; cannot determine its Q-weight")
                    })?);
                }
                row.push(gen_weights);
            }
            weights.push(row);
            target_weights_prev = target_weights;
        }

        Ok(Self {
            min_degree,
            weights,
        })
    }

    /// The $Q$-weights of the generators in bidegree `b`, in the order of their indices.
    pub fn generator_weights(&self, b: Bidegree) -> &[u32] {
        self.weights
            .get(b.s() as usize)
            .and_then(|row| row.get((b.t() - self.min_degree) as usize))
            .map_or(&[], |v| v.as_slice())
    }

    /// The number of generators in bidegree `b` of each $Q$-weight. Weights with no generators
    /// are omitted.
    pub fn weighted_dimensions(&self, b: Bidegree) -> BTreeMap<u32, usize> {
        let mut result = BTreeMap::new();
        for &w in self.generator_weights(b) {
            *result.entry(w).or_default() += 1;
        }
        result
    }
}
