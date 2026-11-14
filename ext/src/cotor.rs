//! Cotor over the dual of the Steenrod algebra, through the free resolution of the ground field.

use std::sync::{Arc, OnceLock};

use algebra::{
    Algebra,
    comodule::{ActionBlock, Comodule, ConjugateAction, FilteredComodule, OpId, OpSet},
    milnor_algebra::PPart,
};
use fp::{matrix::Matrix, vector::FpVector};
use sseq::coordinates::Bidegree;

use crate::chain_complex::FreeChainComplex;

/// The complex $\mathrm{Hom}_A(C, M)$ for a free resolution $C$ of the ground field and a
/// [`Comodule`] $M$, whose cohomology is $\mathrm{Cotor}(M, \mathbb{F}_p) = \mathrm{Ext}_A(
/// \mathbb{F}_p, M)$.
///
/// A cochain of filtration `s` and weight `u` assigns to each generator $x$ of $C_s$ an element of
/// $M_{u - |x|}$, and the coboundary is $(\delta f)(z) = f(dz)$. Every `(s, u)` is independent of
/// every other.
pub struct CotorComplex<CC, M: Comodule> {
    backbone: Arc<CC>,
    comodule: Arc<M>,
    /// Every operation in a differential of the backbone, in the range it is computed.
    ops: OpSet,
    /// The action of [`Self::ops`] on each degree of the comodule, computed when first needed and
    /// kept.
    actions: Vec<OnceLock<ActionBlock>>,
    /// If set, the backbone conjugated: the coboundary reads its coefficients from here and acts
    /// by [`ConjugateAction`], and [`Self::actions`] is never filled.
    conjugate: Option<Conjugate>,
}

/// The differentials of a backbone with each coefficient $a$ rewritten as $\chi(a)$, and the
/// comodule's action by those, for [`CotorComplex::new_conjugate`].
struct Conjugate {
    /// `slices[s][t][l]` lists, for generator `l` of degree `t` in filtration `s + 1`, each
    /// generator $x$ of filtration `s` with $\chi(a_{zx})$ nonzero, as its degree, its index and
    /// the $M$ of the $P(M)$ in $\chi(a_{zx})$.
    slices: Vec<Vec<Vec<Vec<(i32, usize, Vec<PPart>)>>>>,
    /// The action by $\chi(P(M))$ on a basis element, through [`ConjugateAction`].
    #[allow(clippy::type_complexity)]
    act: Box<dyn Fn(PPart, i32, usize, &mut dyn FnMut(usize, u32)) + Send + Sync>,
}

impl<CC, M> CotorComplex<CC, M>
where
    CC: FreeChainComplex,
    M: Comodule<Algebra = CC::Algebra>,
{
    /// The complex for weights up to `max_u` and filtrations up to `max_s`, which reads the part
    /// of the backbone already computed through filtration `max_s + 1` and internal degree
    /// `max_u - M.min_degree()`; a coboundary that needs more of it panics.
    pub fn new(backbone: Arc<CC>, comodule: Arc<M>, max_s: i32, max_u: i32) -> Self {
        let max_t = max_u - comodule.min_degree();
        comodule.compute_basis(max_u - backbone.min_degree());
        let mut ops = Vec::new();
        for s in 1..=max_s + 1 {
            let d = backbone.differential(s);
            let target = backbone.module(s - 1);
            for t in 0..=max_t {
                if !backbone.has_computed_bidegree(Bidegree::s_t(s, t)) {
                    continue;
                }
                for l in 0..backbone.number_of_gens_in_bidegree(Bidegree::s_t(s, t)) {
                    for (idx, _) in d.output(t, l).iter_nonzero() {
                        let og = target.index_to_op_gen(t, idx);
                        ops.push(OpId {
                            degree: og.operation_degree,
                            index: og.operation_index as u32,
                        });
                    }
                }
            }
        }
        let max_d = max_u - backbone.min_degree();
        Self {
            backbone,
            comodule,
            ops: ops.into_iter().collect(),
            actions: (0..=max_d.max(0)).map(|_| OnceLock::new()).collect(),
            conjugate: None,
        }
    }

    /// The action of every backbone operation on degree `d` of the comodule.
    fn action(&self, d: i32) -> &ActionBlock {
        self.actions[d as usize].get_or_init(|| self.comodule.act_batch(&self.ops, d))
    }

    /// Computes the action on every degree up to `max_d` at once, through
    /// [`Comodule::act_batches`], and returns the number of stored entries.
    pub fn precompute_actions(&self, max_d: i32) -> usize {
        let max_d = max_d.min(self.actions.len() as i32 - 1);
        let mut nnz = 0;
        for (d, block) in self
            .comodule
            .act_batches(&self.ops, max_d)
            .into_iter()
            .enumerate()
        {
            nnz += block.nnz();
            let _ = self.actions[d].set(block);
        }
        nnz
    }

    /// The generators of $C_s$ that carry cochains of weight `u`, as `(degree, count, offset)`,
    /// and the total dimension.
    fn blocks(&self, s: i32, u: i32) -> (Vec<(i32, usize, usize)>, usize) {
        let mut out = Vec::new();
        let mut offset = 0;
        for t in self.backbone.min_degree()..=u - self.comodule.min_degree() {
            let gens = self
                .backbone
                .number_of_gens_in_bidegree(Bidegree::s_t(s, t));
            let dim = self.comodule.dimension(u - t);
            if gens > 0 && dim > 0 {
                out.push((t, gens, offset));
                offset += gens * dim;
            }
        }
        (out, offset)
    }

    /// The dimension of the cochains of filtration `s` and weight `u`.
    pub fn cochain_dimension(&self, s: i32, u: i32) -> usize {
        self.blocks(s, u).1
    }

    /// The coboundary from filtration `s` to `s + 1` at weight `u`, one row per source cochain.
    pub fn coboundary(&self, s: i32, u: i32) -> Matrix {
        if let Some(conjugate) = &self.conjugate {
            return self.coboundary_conjugate(conjugate, s, u);
        }
        let p = self.comodule.prime();
        let (src, src_dim) = self.blocks(s, u);
        let (tgt, tgt_dim) = self.blocks(s + 1, u);
        let mut matrix = Matrix::new(p, src_dim, tgt_dim);
        let src_offset = |t: i32| src.iter().find(|b| b.0 == t).map(|b| b.2);
        let d = self.backbone.differential(s + 1);
        let source_module = self.backbone.module(s);
        for &(t_z, gens_z, offset_z) in &tgt {
            let dim_z = self.comodule.dimension(u - t_z);
            for l in 0..gens_z {
                let column_base = offset_z + l * dim_z;
                for (idx, c) in d.output(t_z, l).iter_nonzero() {
                    let og = source_module.index_to_op_gen(t_z, idx);
                    let t_x = og.generator_degree;
                    let Some(offset_x) = src_offset(t_x) else {
                        continue;
                    };
                    let op = OpId {
                        degree: og.operation_degree,
                        index: og.operation_index as u32,
                    };
                    let local = self.ops.position(op).unwrap() as u32;
                    let d_x = u - t_x;
                    let block = self.action(d_x);
                    let row_base = offset_x + og.generator_index * self.comodule.dimension(d_x);
                    for m in 0..self.comodule.dimension(d_x) {
                        let mut row = matrix.row_mut(row_base + m);
                        for (target, v) in block.act(m, local).iter() {
                            row.add_basis_element(column_base + target, c * v % p);
                        }
                    }
                }
            }
        }
        matrix
    }

    /// [`Self::coboundary`] from the conjugated backbone: $(\delta f)(z) = \sum_x f(x) \cdot
    /// \chi(a_{zx})$, with no antipode applied to the comodule.
    fn coboundary_conjugate(&self, conjugate: &Conjugate, s: i32, u: i32) -> Matrix {
        let p = self.comodule.prime();
        let (src, src_dim) = self.blocks(s, u);
        let (tgt, tgt_dim) = self.blocks(s + 1, u);
        let mut matrix = Matrix::new(p, src_dim, tgt_dim);
        let src_offset = |t: i32| src.iter().find(|b| b.0 == t).map(|b| b.2);
        for &(t_z, gens_z, offset_z) in &tgt {
            let dim_z = self.comodule.dimension(u - t_z);
            for l in 0..gens_z {
                let column_base = offset_z + l * dim_z;
                for (t_x, x, terms) in &conjugate.slices[s as usize][t_z as usize][l] {
                    let Some(offset_x) = src_offset(*t_x) else {
                        continue;
                    };
                    let d_x = u - t_x;
                    let row_base = offset_x + x * self.comodule.dimension(d_x);
                    for m in 0..self.comodule.dimension(d_x) {
                        let mut row = matrix.row_mut(row_base + m);
                        for &r in terms {
                            (conjugate.act)(r, d_x, m, &mut |target, v| {
                                row.add_basis_element(column_base + target, v);
                            });
                        }
                    }
                }
            }
        }
        matrix
    }

    /// The rank of [`Self::coboundary`].
    pub fn coboundary_rank(&self, s: i32, u: i32) -> usize {
        let mut m = self.coboundary(s, u);
        if m.rows() == 0 || m.columns() == 0 {
            return 0;
        }
        m.row_reduce()
    }

    /// The dimension of $\mathrm{Cotor}^{s, u}$.
    pub fn cotor(&self, s: i32, u: i32) -> usize {
        let incoming = if s > 0 {
            self.coboundary_rank(s - 1, u)
        } else {
            0
        };
        self.cochain_dimension(s, u) - self.coboundary_rank(s, u) - incoming
    }
}

impl<CC, M> CotorComplex<CC, M>
where
    CC: FreeChainComplex<Algebra = algebra::MilnorAlgebra> + Sync,
    M: ConjugateAction + Sync,
{
    /// [`Self::new`] reading the backbone with each differential coefficient $a_{zx}$ rewritten
    /// once as $\chi(a_{zx})$ in the Milnor basis, so that the comodule acts by its
    /// antipode-free side, [`ConjugateAction`], and no table of its action is kept.
    ///
    /// The coboundary is the same matrix: $a \cdot f(x) = f(x) \cdot \chi(a)$.
    pub fn new_conjugate(backbone: Arc<CC>, comodule: Arc<M>, max_s: i32, max_u: i32) -> Self {
        use algebra::MilnorAntipode;
        use maybe_rayon::prelude::*;
        let mut out = Self::new(Arc::clone(&backbone), Arc::clone(&comodule), max_s, max_u);
        let max_t = max_u - comodule.min_degree();
        let algebra = backbone.algebra();
        let chi = MilnorAntipode::new(Arc::clone(&algebra));
        let slices: Vec<Vec<Vec<Vec<(i32, usize, Vec<PPart>)>>>> = (0..=max_s)
            .into_maybe_par_iter()
            .map(|s| {
                let d = backbone.differential(s + 1);
                let target = backbone.module(s);
                (0..=max_t)
                    .map(|t| {
                        if !backbone.has_computed_bidegree(Bidegree::s_t(s + 1, t)) {
                            return Vec::new();
                        }
                        (0..backbone.number_of_gens_in_bidegree(Bidegree::s_t(s + 1, t)))
                            .map(|l| {
                                // chi of each coefficient, summed mod 2 over its terms.
                                let mut odd: rustc_hash::FxHashMap<(i32, usize, PPart), bool> =
                                    Default::default();
                                for (idx, _) in d.output(t, l).iter_nonzero() {
                                    let og = target.index_to_op_gen(t, idx);
                                    let r = algebra
                                        .basis_element_from_index(
                                            og.operation_degree,
                                            og.operation_index,
                                        )
                                        .p_part;
                                    for &(m, _) in chi.of(r).iter() {
                                        let e = odd
                                            .entry((og.generator_degree, og.generator_index, m))
                                            .or_insert(false);
                                        *e = !*e;
                                    }
                                }
                                let mut by_x: std::collections::BTreeMap<(i32, usize), Vec<PPart>> =
                                    Default::default();
                                for ((tx, x, m), keep) in odd {
                                    if keep {
                                        by_x.entry((tx, x)).or_default().push(m);
                                    }
                                }
                                by_x.into_iter().map(|((tx, x), ms)| (tx, x, ms)).collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let acting = Arc::clone(&comodule);
        out.conjugate = Some(Conjugate {
            slices,
            act: Box::new(move |r, d, idx, emit| acting.act_by_conjugate(r, d, idx, emit)),
        });
        out
    }

    /// The number of terms of the conjugated backbone.
    pub fn conjugate_terms(&self) -> usize {
        self.conjugate.as_ref().map_or(0, |c| {
            c.slices
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .map(|x| x.2.len())
                .sum()
        })
    }
}

impl<CC, M> CotorComplex<CC, M>
where
    CC: FreeChainComplex,
    M: FilteredComodule<Algebra = CC::Algebra>,
{
    /// The filtration of each cochain of filtration `s` and weight `u`, in cochain order.
    fn cochain_filtrations(&self, s: i32, u: i32) -> Vec<i32> {
        let (blocks, dim) = self.blocks(s, u);
        let mut out = Vec::with_capacity(dim);
        for &(t, gens, _) in &blocks {
            for _ in 0..gens {
                out.extend(
                    (0..self.comodule.dimension(u - t)).map(|i| self.comodule.filtration(u - t, i)),
                );
            }
        }
        out
    }

    /// The rank of the coboundary from filtration `s` at weight `u` restricted to the cochains
    /// valued in the filtration piece $F_f$, for each `f` in `0..=max_f`.
    ///
    /// The action does not raise filtration, so this is the coboundary of the complex of the
    /// subcomodule $F_f$; one reduction, with the source cochains ordered by filtration, gives
    /// every `f` at once.
    pub fn coboundary_ranks_by_filtration(&self, s: i32, u: i32, max_f: i32) -> Vec<usize> {
        let p = self.comodule.prime();
        let filtrations = self.cochain_filtrations(s, u);
        let mut order: Vec<usize> = (0..filtrations.len()).collect();
        order.sort_by_key(|&i| filtrations[i]);
        let mut position = vec![0; order.len()];
        for (new, &old) in order.iter().enumerate() {
            position[old] = new;
        }
        let forward = self.coboundary(s, u);
        let mut transposed = Matrix::new(p, forward.columns(), forward.rows());
        for (i, &pos) in position.iter().enumerate() {
            for (j, v) in forward.row(i).iter_nonzero() {
                transposed.row_mut(j).set_entry(pos, v);
            }
        }
        let rank_upto = if transposed.rows() > 0 && transposed.columns() > 0 {
            transposed.row_reduce();
            let pivots = transposed.pivots();
            let mut prefix = vec![0; order.len() + 1];
            for c in 0..order.len() {
                prefix[c + 1] = prefix[c] + usize::from(pivots[c] >= 0);
            }
            prefix
        } else {
            vec![0; order.len() + 1]
        };
        (0..=max_f)
            .map(|f| rank_upto[filtrations.iter().filter(|&&x| x <= f).count()])
            .collect()
    }

    /// The dimension of the cochains of filtration `s` and weight `u` valued in $F_f$, for each
    /// `f` in `0..=max_f`.
    pub fn cochain_dimensions_by_filtration(&self, s: i32, u: i32, max_f: i32) -> Vec<usize> {
        let filtrations = self.cochain_filtrations(s, u);
        (0..=max_f)
            .map(|f| filtrations.iter().filter(|&&x| x <= f).count())
            .collect()
    }

    /// The dimension of $\mathrm{Cotor}^{s, u}(F_f)$ for each `f` in `0..=max_f`.
    pub fn cotor_by_filtration(&self, s: i32, u: i32, max_f: i32) -> Vec<usize> {
        let out_ranks = self.coboundary_ranks_by_filtration(s, u, max_f);
        let in_ranks = if s > 0 {
            self.coboundary_ranks_by_filtration(s - 1, u, max_f)
        } else {
            vec![0; max_f as usize + 1]
        };
        let dims = self.cochain_dimensions_by_filtration(s, u, max_f);
        (0..=max_f as usize)
            .map(|f| dims[f] - out_ranks[f] - in_ranks[f])
            .collect()
    }
}

/// The comodule dual to $B \backslash A = \mathbb{F}_p \otimes_B A$ in degrees up to `max_degree`,
/// for the subalgebra $B$ generated by `generators`: the functionals on $A$ vanishing on
/// $B^+ A$, with $\langle b, a \cdot f \rangle = f(ba)$.
///
/// Its Cotor over $A$ is $\mathrm{Ext}_B(\mathbb{F}_p, \mathbb{F}_p)$ by change of rings, in
/// every weight up to `max_degree`.
pub fn coinduced<A: Algebra>(
    algebra: Arc<A>,
    generators: &[(i32, usize)],
    max_degree: i32,
) -> algebra::comodule::FiniteComodule<A> {
    use fp::matrix::Subspace;
    let p = algebra.prime();
    algebra.compute_basis(max_degree);
    // The right ideal B^+ A in each degree, and the coordinates of the quotient.
    let ideals: Vec<Subspace> = (0..=max_degree)
        .map(|d| {
            let mut sub = Subspace::new(p, algebra.dimension(d));
            for &(n, g) in generators {
                for b in 0..algebra
                    .dimension(d - n)
                    .min(if d >= n { usize::MAX } else { 0 })
                {
                    let mut v = FpVector::new(p, algebra.dimension(d));
                    algebra.multiply_basis_elements(v.as_slice_mut(), 1, n, g, d - n, b);
                    sub.add_vector(v.as_slice());
                }
            }
            sub
        })
        .collect();
    let quotient_basis: Vec<Vec<usize>> = ideals
        .iter()
        .map(|sub| {
            (0..sub.ambient_dimension())
                .filter(|&c| sub.pivots()[c] < 0)
                .collect()
        })
        .collect();
    let dims = bivec::BiVec::from_vec(0, quotient_basis.iter().map(Vec::len).collect());
    let mut m =
        algebra::comodule::FiniteComodule::new(Arc::clone(&algebra), "coinduced".into(), dims);
    for d in 1..=max_degree {
        for n in 1..=d {
            for a in 0..algebra.dimension(n) {
                // a . f_c = sum over basis b of the quotient of [coefficient of c in b a] f_b.
                let qd = &quotient_basis[d as usize];
                let qe = &quotient_basis[(d - n) as usize];
                let mut images = vec![vec![0u32; qe.len()]; qd.len()];
                for (j, &b) in qe.iter().enumerate() {
                    let mut ba = FpVector::new(p, algebra.dimension(d));
                    algebra.multiply_basis_elements(ba.as_slice_mut(), 1, d - n, b, n, a);
                    ideals[d as usize].reduce(ba.as_slice_mut());
                    for (i, &c) in qd.iter().enumerate() {
                        images[i][j] = ba.entry(c);
                    }
                }
                for (i, image) in images.iter().enumerate() {
                    m.set_action(n, a, d, i, image);
                }
            }
        }
    }
    m
}

/// The dual $N^\vee$ of a finite module over the Milnor algebra at $p = 2$, with
/// $(a \cdot f)(n) = f(\chi(a) n)$, in the degrees of $N$ with the basis dual to its.
///
/// Its Cotor is $\mathrm{Ext}_A(N, \mathbb{F}_2)$: the vector-space dual of a module is a left
/// comodule, and the antipode makes it the right one a [`Comodule`] presents.
pub fn dual_comodule(
    module: &algebra::module::FDModule<algebra::MilnorAlgebra>,
) -> algebra::comodule::FiniteComodule<algebra::MilnorAlgebra> {
    use algebra::{
        MilnorAlgebra,
        milnor_algebra::{MilnorBasisElement, PPart},
        module::Module,
    };

    use crate::algebraic_novikov::{chi_table, pack, unpack, xi_degree};
    let algebra: Arc<MilnorAlgebra> = module.algebra();
    assert_eq!(
        algebra.prime(),
        2,
        "the antipode is only implemented at p = 2"
    );
    let (min, max) = (module.min_degree(), module.max_degree().unwrap());
    let span = max - min;
    algebra.compute_basis(span);
    // chi[S] lists the R with chi(P(S)) = sum of P(R).
    let chi = chi_table(span);
    let index_of = |r: u128| -> (i32, usize) {
        let d = xi_degree(r);
        let elt = MilnorBasisElement {
            q_part: 0,
            p_part: PPart::from_slice(&unpack(r)),
            degree: d,
        };
        (d, algebra.basis_element_to_index(&elt))
    };
    let dims = bivec::BiVec::from_vec(min, (min..=max).map(|d| module.dimension(d)).collect());
    let mut m =
        algebra::comodule::FiniteComodule::new(Arc::clone(&algebra), format!("D{module}"), dims);
    let p = algebra.prime();
    for d in min..=max {
        for n in 1..=d - min {
            for a in 0..algebra.dimension(n) {
                let s: Vec<u32> = algebra
                    .basis_element_from_index(n, a)
                    .p_part
                    .iter()
                    .collect();
                let chi_a: Vec<usize> = chi
                    .get(&pack(&s))
                    .into_iter()
                    .flatten()
                    .map(|&r| index_of(r).1)
                    .collect();
                // a . f_c = sum over b of [coefficient of n_c in chi(a) n_b] f_b.
                let products: Vec<FpVector> = (0..module.dimension(d - n))
                    .map(|b| {
                        let mut v = FpVector::new(p, module.dimension(d));
                        for &r in &chi_a {
                            module.act_on_basis(v.as_slice_mut(), 1, n, r, d - n, b);
                        }
                        v
                    })
                    .collect();
                let images: Vec<Vec<u32>> = (0..module.dimension(d))
                    .map(|c| products.iter().map(|v| v.entry(c)).collect())
                    .collect();
                for (c, image) in images.iter().enumerate() {
                    m.set_action(n, a, d, c, image);
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use algebra::comodule::{Comodule, FiniteComodule};

    use super::*;
    use crate::{chain_complex::ChainComplex, utils::construct};

    /// The Cotor of the coregular comodule is the ground field in degree zero.
    #[test]
    fn coregular_is_acyclic() {
        let res = Arc::new(construct("S_2", None).unwrap());
        let (max_s, max_u) = (4, 14);
        res.compute_through_stem(Bidegree::n_s(max_u + 1, max_s + 2));
        let m = Arc::new(FiniteComodule::coregular(res.algebra(), max_u));
        let cx = CotorComplex::new(Arc::clone(&res), m, max_s, max_u);
        for s in 0..=max_s {
            for u in 0..=max_u {
                assert_eq!(
                    cx.cotor(s, u),
                    usize::from(s == 0 && u == 0),
                    "(s, u) = ({s}, {u})"
                );
            }
        }
    }

    /// The Cotor of the unit is the Ext of the sphere.
    #[test]
    fn unit_is_the_sphere() {
        let res = Arc::new(construct("S_2", None).unwrap());
        let (max_s, max_u) = (5, 20);
        res.compute_through_stem(Bidegree::n_s(max_u + 1, max_s + 2));
        let m = Arc::new(FiniteComodule::coregular(res.algebra(), 0));
        let cx = CotorComplex::new(Arc::clone(&res), m, max_s, max_u);
        for s in 0..=max_s {
            for u in 0..=max_u {
                let gens = res.number_of_gens_in_bidegree(Bidegree::s_t(s, u));
                assert_eq!(cx.cotor(s, u), gens, "(s, u) = ({s}, {u})");
            }
        }
    }

    /// Change of rings: the Cotor of the comodule coinduced from $A(1)$ is the Ext of $A(1)$.
    #[test]
    fn change_of_rings_to_a1() {
        let res = Arc::new(construct("S_2", None).unwrap());
        let (max_s, max_u) = (5, 16);
        res.compute_through_stem(Bidegree::n_s(max_u + 1, max_s + 2));
        let alg = res.algebra();
        let gens: Vec<(i32, usize)> = ["Sq1", "Sq2"]
            .iter()
            .map(|g| alg.basis_element_from_string(g).unwrap())
            .collect();
        let m = Arc::new(coinduced(Arc::clone(&alg), &gens, max_u));
        let cx = CotorComplex::new(Arc::clone(&res), m, max_s, max_u);

        let a1 = serde_json::json!({
            "p": 2,
            "algebra": { "profile": { "truncated": true, "p_part": [2, 1] } },
            "type": "finite dimensional module",
            "gens": { "x0": 0 },
            "actions": []
        });
        let a1_res = construct((a1, algebra::AlgebraType::Milnor), None).unwrap();
        a1_res.compute_through_stem(Bidegree::n_s(max_u, max_s + 1));
        for s in 0..=max_s {
            for u in 0..=max_u {
                let gens = a1_res.number_of_gens_in_bidegree(Bidegree::s_t(s, u));
                assert_eq!(cx.cotor(s, u), gens, "(s, u) = ({s}, {u})");
            }
        }
    }

    /// The Cotor of $\mathbb{F}_2[\xi_1^4, \xi_2^2, \xi_3, \dots]$, presented by generators, is the
    /// Ext of $A(1)$, and that of $\mathbb{F}_2[\xi_1^2, \xi_2, \dots]$ is $\mathbb{F}_2[h_0]$.
    #[test]
    fn change_of_rings_by_generators() {
        use algebra::comodule::PolynomialComodule;

        use crate::algebraic_novikov::unit_module;
        let res = Arc::new(crate::nassau::Resolution::new(Arc::new(unit_module())));
        let (max_s, max_u) = (5, 20);
        res.compute_through_stem(Bidegree::n_s(max_u + 1, max_s + 2));
        let alg = res.algebra();

        let a1 = serde_json::json!({
            "p": 2,
            "algebra": { "profile": { "truncated": true, "p_part": [2, 1] } },
            "type": "finite dimensional module",
            "gens": { "x0": 0 },
            "actions": []
        });
        let a1_res = construct((a1, algebra::AlgebraType::Milnor), None).unwrap();
        a1_res.compute_through_stem(Bidegree::n_s(max_u, max_s + 1));
        let e = [2, 1];
        let m = Arc::new(PolynomialComodule::dual_steenrod_subalgebra(
            Arc::clone(&alg),
            move |k| e.get(k - 1).copied().unwrap_or(0),
            max_u,
        ));
        let cx = CotorComplex::new(Arc::clone(&res), m, max_s, max_u);
        for s in 0..=max_s {
            for u in 0..=max_u {
                let gens = a1_res.number_of_gens_in_bidegree(Bidegree::s_t(s, u));
                assert_eq!(cx.cotor(s, u), gens, "A(1) at (s, u) = ({s}, {u})");
            }
        }

        let m = Arc::new(PolynomialComodule::dual_steenrod_subalgebra(
            Arc::clone(&alg),
            |k| (k == 1) as u32,
            max_u,
        ));
        let cx = CotorComplex::new(Arc::clone(&res), m, max_s, max_u);
        for s in 0..=max_s {
            for u in 0..=max_u {
                assert_eq!(
                    cx.cotor(s, u),
                    usize::from(u == s),
                    "E(0) at (s, u) = ({s}, {u})"
                );
            }
        }
    }

    /// $\mathrm{Ext}_P(Q)$ through [`CotorComplex`] agrees with [`ctau_e2_untwisted_on`].
    ///
    /// [`ctau_e2_untwisted_on`]: crate::algebraic_novikov::ctau_e2_untwisted_on
    #[test]
    fn ext_of_q_matches_untwisted() {
        use crate::algebraic_novikov::{
            ctau_e2_untwisted_on, q_comodule, unit_module, untwisted_backbone_range,
        };
        let (max_n, max_s) = (24, 8);
        let res = Arc::new(crate::nassau::Resolution::new(Arc::new(unit_module())));
        let max_u = (max_n + max_s) / 2;
        res.compute_through_stem(untwisted_backbone_range(max_n, max_s));
        res.compute_through_stem(Bidegree::n_s(max_u + 1, max_s + 2));
        let expected = ctau_e2_untwisted_on(&*res, max_n, max_s).0;

        let mut ours = std::collections::BTreeMap::new();
        for k in 0..=max_s {
            let max_u_k = (max_n + max_s - k) / 2;
            let m = Arc::new(q_comodule(res.algebra(), k as u32, max_u_k));
            let cx = CotorComplex::new(Arc::clone(&res), m, max_s - k, max_u_k);
            for u in 0..=max_u_k {
                for a in 0..=max_s - k {
                    let n = 2 * u - a;
                    if !(0..=max_n).contains(&n) {
                        continue;
                    }
                    let d = cx.cotor(a, u);
                    if d > 0 {
                        ours.insert((n, a + k, u), d);
                    }
                }
            }
        }
        assert_eq!(ours, expected);
    }

    /// $\mathrm{Ext}_P(Q_k)$ for every $k$ at once, as the filtration pieces of $Q/(v_0 - 1)$,
    /// agrees with [`ctau_e2_untwisted_on`].
    ///
    /// [`ctau_e2_untwisted_on`]: crate::algebraic_novikov::ctau_e2_untwisted_on
    #[test]
    fn ext_of_q_by_length_matches_untwisted() {
        use crate::algebraic_novikov::{
            ctau_e2_untwisted_on, q_mod_v0_comodule, unit_module, untwisted_backbone_range,
        };
        let (max_n, max_s) = (24, 8);
        let res = Arc::new(crate::nassau::Resolution::new(Arc::new(unit_module())));
        let max_u = (max_n + max_s) / 2;
        res.compute_through_stem(untwisted_backbone_range(max_n, max_s));
        res.compute_through_stem(Bidegree::n_s(max_u + 1, max_s + 2));
        let expected = ctau_e2_untwisted_on(&*res, max_n, max_s).0;

        let m = Arc::new(q_mod_v0_comodule(res.algebra(), max_u));
        let cx = CotorComplex::new(Arc::clone(&res), m, max_s, max_u);
        let mut ours = std::collections::BTreeMap::new();
        for u in 0..=max_u {
            for a in 0..=max_s {
                let n = 2 * u - a;
                if !(0..=max_n).contains(&n) {
                    continue;
                }
                for (k, d) in cx
                    .cotor_by_filtration(a, u, max_s - a)
                    .into_iter()
                    .enumerate()
                {
                    let k = k as i32;
                    if d > 0 && 2 * u - a <= max_n && u <= (max_n + max_s - k) / 2 {
                        ours.insert((n, a + k, u), d);
                    }
                }
            }
        }
        assert_eq!(ours, expected);
    }

    /// $M \otimes \Gamma$ is cofree, so its Cotor is $M$ in filtration zero.
    #[test]
    fn tensor_with_coregular_is_cofree() {
        use algebra::comodule::TensorComodule;

        use crate::algebraic_novikov::{q_comodule, unit_module};
        let res = Arc::new(crate::nassau::Resolution::new(Arc::new(unit_module())));
        let (max_s, max_u) = (4, 14);
        res.compute_through_stem(Bidegree::n_s(max_u + 1, max_s + 2));
        let m = Arc::new(q_comodule(res.algebra(), 2, max_u));
        let gamma = Arc::new(FiniteComodule::coregular(res.algebra(), max_u));
        let t = Arc::new(TensorComodule::new(Arc::clone(&m), gamma));
        let cx = CotorComplex::new(Arc::clone(&res), t, max_s, max_u);
        m.compute_basis(max_u);
        for s in 0..=max_s {
            for u in 0..=max_u {
                let expected = if s == 0 { m.dimension(u) } else { 0 };
                assert_eq!(cx.cotor(s, u), expected, "(s, u) = ({s}, {u})");
            }
        }
    }

    /// Resolutions over quotients of the Milnor algebra: the quotient dual to
    /// $\mathbb{F}_2[\xi_k^2]$ is the Steenrod algebra doubled, and the one dual to
    /// $\mathbb{F}_2[\xi_1]$ has Ext $\mathbb{F}_2[h_{1j}]$ with $h_{1j}$ in $(1, 2^j)$.
    #[test]
    fn ext_of_quotient_algebras() {
        use algebra::{MilnorAlgebra, MilnorQuotientAlgebra, module::FDModule};
        use bivec::BiVec;

        use crate::{chain_complex::FiniteChainComplex, resolution::Resolution};

        /// The resolution of the ground field over `algebra` through `(stem, s)`.
        fn resolve<A: algebra::Algebra>(
            algebra: A,
            max: Bidegree,
        ) -> Resolution<FiniteChainComplex<FDModule<A>>> {
            let unit = FDModule::new(Arc::new(algebra), "F2".into(), BiVec::from_vec(0, vec![1]));
            let res = Resolution::new(Arc::new(FiniteChainComplex::ccdz(Arc::new(unit))));
            res.compute_through_stem(max);
            res
        }

        let p = fp::prime::ValidPrime::new(2);
        let (max_n, max_s) = (20, 6);
        let sphere = resolve(MilnorAlgebra::new(p, false), Bidegree::n_s(max_n, max_s));
        let doubled = resolve(
            MilnorQuotientAlgebra::new(MilnorAlgebra::new(p, false), vec![Some(1); 6]),
            Bidegree::n_s(2 * max_n + max_s, max_s),
        );
        for s in 0..=max_s {
            for t in 0..=2 * (max_n + s) {
                let expected = if t % 2 == 0 && t / 2 - s <= max_n {
                    sphere.number_of_gens_in_bidegree(Bidegree::s_t(s, t / 2))
                } else {
                    0
                };
                if t - s <= 2 * max_n + max_s {
                    assert_eq!(
                        doubled.number_of_gens_in_bidegree(Bidegree::s_t(s, t)),
                        expected,
                        "doubled at (s, t) = ({s}, {t})"
                    );
                }
            }
        }

        let (max_t, max_s) = (24, 5);
        let xi1 = resolve(
            MilnorQuotientAlgebra::new(MilnorAlgebra::new(p, false), vec![Some(0)]),
            Bidegree::n_s(max_t, max_s),
        );
        // Monomials in h_{1j}, of degree (1, 2^j), counted by (s, t).
        let mut count = std::collections::BTreeMap::new();
        count.insert((0, 0), 1usize);
        for j in 0..5 {
            let w = 1 << j;
            let old = count.clone();
            for ((s, t), c) in old {
                for e in 1.. {
                    if s + e > max_s || t + e * w > max_t {
                        break;
                    }
                    *count.entry((s + e, t + e * w)).or_default() += c;
                }
            }
        }
        for s in 0..=max_s {
            for t in s..=max_t {
                assert_eq!(
                    xi1.number_of_gens_in_bidegree(Bidegree::s_t(s, t)),
                    count.get(&(s, t)).copied().unwrap_or(0),
                    "F2[xi1] at (s, t) = ({s}, {t})"
                );
            }
        }
    }

    /// The Cotor of the dual of a finite module is the module's Ext.
    #[test]
    fn dual_of_module_gives_its_ext() {
        use algebra::{MilnorAlgebra, module::FDModule};

        use crate::{algebraic_novikov::unit_module, utils::load_module_json};
        let res = Arc::new(crate::nassau::Resolution::new(Arc::new(unit_module())));
        let (max_s, max_u) = (5, 20);
        res.compute_through_stem(Bidegree::n_s(max_u + 1, max_s + 2));
        for name in ["Ceta", "Cnu", "Joker", "C2", "RP4"] {
            let json = load_module_json(name).unwrap();
            let module: FDModule<MilnorAlgebra> =
                FDModule::from_json(res.algebra(), &json).unwrap();
            let theirs = construct(name, None).unwrap();
            theirs.compute_through_stem(Bidegree::n_s(max_u, max_s));
            let m = Arc::new(dual_comodule(&module));
            let cx = CotorComplex::new(Arc::clone(&res), m, max_s, max_u);
            for s in 0..=max_s {
                for u in s..=max_u {
                    let t = u;
                    if t - s > max_u - max_s {
                        continue;
                    }
                    assert_eq!(
                        cx.cotor(s, u),
                        theirs.number_of_gens_in_bidegree(Bidegree::s_t(s, t)),
                        "{name} at (s, u) = ({s}, {u})"
                    );
                }
            }
        }
    }

    /// The action of [`algebra::comodule::MatrixQComodule`], from Milnor matrices and the
    /// antipode, agrees entry by entry with the walk of the same comodule presented by
    /// generators.
    #[test]
    fn matrix_action_matches_generators() {
        use algebra::{
            MilnorAlgebra,
            comodule::{MatrixQComodule, OpId},
        };

        use crate::algebraic_novikov::q_mod_v0_comodule;
        let max = 40;
        let alg = Arc::new(MilnorAlgebra::new(fp::prime::TWO, false));
        alg.compute_basis(max);
        let ops: OpSet = (0..=max)
            .flat_map(|n| {
                (0..alg.dimension(n)).map(move |i| OpId {
                    degree: n,
                    index: i as u32,
                })
            })
            .collect();
        let walk = q_mod_v0_comodule(Arc::clone(&alg), max).act_batches(&ops, max);
        let matrix = MatrixQComodule::new(Arc::clone(&alg), max);
        matrix.compute_basis(max);
        let mut nonzero = 0;
        for d in 0..=max {
            let block = matrix.act_batch(&ops, d);
            for idx in 0..matrix.dimension(d) {
                let a: Vec<_> = walk[d as usize]
                    .row(idx)
                    .map(|(l, e)| (l, e.iter().collect::<Vec<_>>()))
                    .collect();
                let b: Vec<_> = block
                    .row(idx)
                    .map(|(l, e)| (l, e.iter().collect::<Vec<_>>()))
                    .collect();
                nonzero += b.len();
                assert_eq!(a, b, "degree {d}, index {idx}");
            }
        }
        assert!(nonzero > 1000);
    }

    /// The conjugated backbone with the antipode-free action gives the same coboundaries as the
    /// backbone with the action through the antipode, and so the same Ext of $Q$.
    #[test]
    fn conjugate_backbone_matches() {
        use algebra::comodule::MatrixQComodule;

        use crate::algebraic_novikov::{unit_module, untwisted_backbone_range};
        let (max_n, max_s) = (30, 10);
        let res = Arc::new(crate::nassau::Resolution::new(Arc::new(unit_module())));
        let max_u = (max_n + max_s) / 2;
        res.compute_through_stem(untwisted_backbone_range(max_n, max_s));
        let m = Arc::new(MatrixQComodule::new(res.algebra(), max_u));
        let plain = CotorComplex::new(Arc::clone(&res), Arc::clone(&m), max_s, max_u);
        let conj = CotorComplex::new_conjugate(Arc::clone(&res), Arc::clone(&m), max_s, max_u);
        let mut checked = 0;
        for u in 0..=max_u {
            for a in 0..max_s {
                if 2 * u - a > max_n + 1 {
                    continue;
                }
                let (x, y) = (plain.coboundary(a, u), conj.coboundary(a, u));
                assert_eq!((x.rows(), x.columns()), (y.rows(), y.columns()));
                for i in 0..x.rows() {
                    assert_eq!(
                        x.row(i).iter_nonzero().collect::<Vec<_>>(),
                        y.row(i).iter_nonzero().collect::<Vec<_>>(),
                        "(a, u) = ({a}, {u}), row {i}"
                    );
                }
                checked += x.rows();
            }
        }
        assert!(checked > 500);
    }
}
