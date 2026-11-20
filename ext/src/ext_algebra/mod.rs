//! Ext as a bigraded ring and its modules.

pub mod massey;
pub mod secondary;
pub mod tensor_resolution;

use std::sync::Arc;

use dashmap::DashMap;
use fp::{
    matrix::{AugmentedMatrix, Matrix, Subquotient, Subspace},
    prime::ValidPrime,
    vector::{FpSlice, FpVector},
};
use sseq::coordinates::{Bidegree, BidegreeElement, BidegreeGenerator};

pub use self::{
    secondary::{SecondaryExtAlgebra, SecondaryProduct},
    tensor_resolution::{
        Antipode, TensorResolutionDifferential, tensor_resolution_ext,
        tensor_resolution_ext_with_save_dir, tensor_resolution_products,
    },
};
use crate::{
    chain_complex::{AugmentedChainComplex, FreeChainComplex},
    resolution_homomorphism::ResolutionHomomorphism,
    utils::{QueryModuleResolution, get_unit},
};

/// The coboundary of the Ext cochain complex $\Hom(P_\bullet, k) = k^{\text{gens}}$, whose
/// cohomology is $\Ext$.
///
/// It shifts bidegree by a fixed [`shift`](ExtDifferential::shift) and, at each bidegree, gives its
/// matrix in the generator bases. For a **minimal** resolution this coboundary is identically zero.
/// For a **non-minimal** resolution it is the canonical dualised differential $\Hom(d, k)$.
///
/// This is also how a deformation encodes the connecting differential in this complex — the Adams
/// $d_2$ ([`secondary`]) or the motivic $\delta$ — as instances defined in their own modules.
pub trait ExtDifferential: Send + Sync {
    /// The fixed bidegree shift the differential applies: $\delta\colon \Ext_b \to
    /// \Ext_{b + \mathrm{shift}}$.
    fn shift(&self) -> Bidegree;

    /// The matrix of $\delta$ out of bidegree `b`: rows index the generators at `b`, columns the
    /// generators at `b + shift`. `None` if the differential out of `b` is out of the computed
    /// range.
    fn matrix(&self, b: Bidegree) -> Option<Matrix>;

    /// The number of cochain generators at `b`, when the differential's cochain complex is *not*
    /// the one read off the backing resolution's generators.
    ///
    /// The default `None` means "use the backing resolution's generator count", which is right for
    /// a minimal resolution or a $d_2$ page. The untwisted tensor resolution coboundary
    /// ([`TensorResolutionDifferential`]) overrides this: it resolves a *different* module `M` by
    /// tensoring the backing $k$-resolution with `M`, so its cochain generators at $(n, s)$ number
    /// $\sum_i \dim M_{t - d_i}$, not the $k$-resolution's `number_of_gens_in_bidegree`.
    fn dimension(&self, b: Bidegree) -> Option<usize> {
        let _ = b;
        None
    }
}

/// A cochain-level **cup product** $x \cup -$ by a class $x \in \Ext(k, k)$, on the cochain complex
/// $\Hom_A(Q_\bullet, k)$.
///
/// This is what lets [`ExtModule::massey`] compute Massey products on a **non-minimal** resolution:
/// the chain-map/null-homotopy bracket degenerates there, but the cochain-DGA formula $\langle
/// a,b,c\rangle = [a \cup v]$, $\delta_Q v = b\cup c$, stays correct precisely because $\delta_Q
/// \neq 0$. It is attached by the untwisted tensor resolution ([`tensor_resolution_products`]); a
/// minimal resolution attaches none and [`ExtModule::massey`] uses the chain-map construction.
pub trait CochainCup<CC: FreeChainComplex>: Send + Sync {
    /// The matrix of $x \cup -\colon \Hom_A(Q_s, k)_t \to \Hom_A(Q_{s + \mathrm{shift}.s}, k)_{t +
    /// \mathrm{shift}.t}$: rows index the source cochain basis at `(src_s, src_t)`, columns the
    /// target. `f_x` is the chain self-map of the unit realising `x` (its shift is `x`'s bidegree),
    /// extended through the target filtration.
    fn cup_matrix(
        &self,
        f_x: &ResolutionHomomorphism<CC, CC>,
        src_s: i32,
        src_t: i32,
        shift: Bidegree,
    ) -> Matrix;
}

/// A cochain of the Ext cochain complex $\Hom_A(P_\bullet, k)$ at a bidegree: a vector over the
/// **cochain generators** (dual to the resolution's free generators).
///
/// A [`BidegreeElement`] of an [`ExtModule`] is a cohomology **class**. The two coincide only when
/// the coboundary vanishes (a minimal resolution); in general convert with [`ExtModule::lift`]
/// (class → cocycle representative) and [`ExtModule::project`] (cochain → class).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cochain {
    degree: Bidegree,
    vec: FpVector,
}

impl Cochain {
    pub fn new(degree: Bidegree, vec: FpVector) -> Self {
        Self { degree, vec }
    }

    pub fn degree(&self) -> Bidegree {
        self.degree
    }

    pub fn vec(&self) -> FpSlice<'_> {
        self.vec.as_slice()
    }

    pub fn into_vec(self) -> FpVector {
        self.vec
    }
}

/// The ring $\Ext(k, k)$, backed by a resolution of the base field `k`.
///
/// A product is realised by a [`ResolutionHomomorphism`] from a fixed multiplier class, which
/// computes the products of that class with all of $\Ext(k, k)$ at once. One such map is cached per
/// generator; a product by a general class is assembled from them on request (see
/// [`class_product_map`](Self::class_product_map)). Products are computed up to a sign.
pub struct ExtAlgebra<CC: FreeChainComplex> {
    /// Resolution of the base field `k`. Ring products live here.
    resolution: Arc<CC>,
    /// One multiplication map per generator of $\Ext(k, k)$, `res(k) → res(k)`, built on demand.
    products: DashMap<BidegreeGenerator, Arc<ResolutionHomomorphism<CC, CC>>>,
}

impl<CC: FreeChainComplex> ExtAlgebra<CC> {
    /// Build the ring $\Ext(k, k)$ from a resolution of `k`.
    pub fn new(resolution: Arc<CC>) -> Self {
        Self {
            resolution,
            products: DashMap::new(),
        }
    }

    /// The resolution of `k` backing this ring.
    pub fn resolution(&self) -> &Arc<CC> {
        &self.resolution
    }

    /// The prime of the underlying resolution.
    pub fn prime(&self) -> ValidPrime {
        self.resolution.prime()
    }

    /// The dimension of $\Ext^{s,t}(k, k)$ at the given bidegree.
    pub fn dimension(&self, b: Bidegree) -> usize {
        self.resolution.number_of_gens_in_bidegree(b)
    }

    /// The basis generators of $\Ext(k, k)$ at the given bidegree.
    pub fn basis(&self, b: Bidegree) -> Vec<BidegreeGenerator> {
        (0..self.dimension(b))
            .map(|i| BidegreeGenerator::new(b, i))
            .collect()
    }

    /// A class in $\Ext(k, k)$ from its coordinates in the generator basis at bidegree `b`.
    pub fn element(&self, b: Bidegree, coords: &[u32]) -> BidegreeElement {
        assert_eq!(self.dimension(b), coords.len());
        BidegreeElement::new(b, FpVector::from_slice(self.prime(), coords))
    }

    /// A single generator of $\Ext(k, k)$ as a class.
    pub fn generator(&self, g: BidegreeGenerator) -> BidegreeElement {
        let ambient = self.dimension(g.degree());
        assert!(ambient > g.idx());
        g.into_element(self.prime(), ambient)
    }
}

impl<CC> ExtAlgebra<CC>
where
    CC: FreeChainComplex + AugmentedChainComplex,
{
    /// The multiplication map for a single generator `g` of $\Ext(k, k)$ (`res(k) → res(k)`), built
    /// and cached on first use. The returned map is *not* guaranteed to be extended.
    pub fn generator_product_map(
        &self,
        g: BidegreeGenerator,
    ) -> Arc<ResolutionHomomorphism<CC, CC>> {
        cached_generator_product_map(&self.products, &self.resolution, &self.resolution, g)
    }

    /// The multiply-by-`x` chain self-map of `res(k)` (`res(k) → res(k)`), extended through `max`.
    ///
    /// A generator with coefficient one returns the cached
    /// [`generator_product_map`](Self::generator_product_map) itself; any other nonzero class adds
    /// the cached generator maps via [`ResolutionHomomorphism::linear_combination`] (no
    /// quasi-inverse lift).
    pub fn class_product_map(
        &self,
        x: &BidegreeElement,
        max: Bidegree,
    ) -> Arc<ResolutionHomomorphism<CC, CC>> {
        let summands: Vec<(u32, Arc<ResolutionHomomorphism<CC, CC>>)> = x
            .vec()
            .iter_nonzero()
            .map(|(idx, c)| {
                let map = self.generator_product_map(BidegreeGenerator::new(x.degree(), idx));
                map.extend_through_stem(max);
                (c, map)
            })
            .collect();
        match summands.as_slice() {
            [(1, map)] => Arc::clone(map),
            [] => {
                // With no initial images, extending lifts zero everywhere.
                let hom = Arc::new(ResolutionHomomorphism::new(
                    String::new(),
                    Arc::clone(&self.resolution),
                    Arc::clone(&self.resolution),
                    x.degree(),
                ));
                hom.extend_through_stem(max);
                hom
            }
            _ => Arc::new(ResolutionHomomorphism::linear_combination(
                String::new(),
                &summands,
                max,
            )),
        }
    }

    /// Left-multiplication by `x ∈ Ext(k, k)`, applied to every basis generator of $\Ext(k, k)$ at
    /// bidegree `b`. See [`ExtModule::multiply_into`] for the return convention.
    pub fn multiply_into(&self, x: &BidegreeElement, b: Bidegree) -> Option<Matrix> {
        products_into(
            &self.resolution,
            &self.resolution,
            &self.products,
            self.prime(),
            x,
            b,
        )
    }

    /// The ring product `x · y` (both in $\Ext(k, k)$) if it lies in the computed range, else
    /// `None`. The result lies in bidegree `x.degree() + y.degree()`.
    pub fn try_multiply(
        &self,
        x: &BidegreeElement,
        y: &BidegreeElement,
    ) -> Option<BidegreeElement> {
        let matrix = self.multiply_into(x, y.degree())?;
        Some(combine_product(
            &matrix,
            y,
            x.degree() + y.degree(),
            self.prime(),
        ))
    }

    /// The ring product `x · y`, both in $\Ext(k, k)$. Panics if out of the computed range; use
    /// [`try_multiply`](Self::try_multiply) to handle that case.
    pub fn multiply(&self, x: &BidegreeElement, y: &BidegreeElement) -> BidegreeElement {
        self.try_multiply(x, y).expect(
            "multiply: product is out of the computed range; compute further or use try_multiply",
        )
    }
}

/// The module $\Ext(M, k)$ over the ring [`ExtAlgebra`] $\Ext(k, k)$, backed by a resolution of
/// `M`.
///
/// Products follow the [`ExtAlgebra`] conventions, with the multiplier in $\Ext(M, k)$ and the
/// chain maps running `res(M) → res(k)`. When `M == k` the module and ring share one resolution
/// (see [`is_unit`](Self::is_unit)).
pub struct ExtModule<CC: FreeChainComplex> {
    /// Resolution of `M`; the module's classes and the module-action products land in its Ext.
    resolution: Arc<CC>,
    /// Shared handle to the ring $\Ext(k, k)$. `Arc`-shared so all modules over the same `k` reuse
    /// one ring cache.
    algebra: Arc<ExtAlgebra<CC>>,
    /// One multiplication map per generator of $\Ext(M, k)$, `res(M) → res(k)`, built on demand.
    /// Read through [`product_cache`](Self::product_cache).
    products: DashMap<BidegreeGenerator, Arc<ResolutionHomomorphism<CC, CC>>>,
    /// The DGA differential, if any. `None` is the field/minimal case (zero
    /// coboundary), where the cohomology is just the generators.
    differential: Option<Arc<dyn ExtDifferential>>,
    /// The cochain cup product $x \cup -$, if any. Attached for the untwisted tensor resolution
    /// so [`massey`](Self::massey) and [`multiply_into`](Self::multiply_into) use the cochain DGA.
    cup: Option<Arc<dyn CochainCup<CC>>>,
    /// Cup matrices `{y_j ∪ -}` keyed by `(src, mult_deg)`. They depend only on those two
    /// bidegrees, not on the multiplier's coordinates.
    cup_cache: DashMap<(Bidegree, Bidegree), Arc<Vec<Matrix>>>,
    /// Quasi-inverse of the coboundary out of each bidegree, for the Massey `δ`-preimage.
    delta_qi_cache: DashMap<Bidegree, Arc<fp::matrix::QuasiInverse>>,
    /// The cohomology of [`differential`](Self::differential) at each bidegree; see
    /// [`cohomology`](Self::cohomology).
    cohomology_cache: DashMap<Bidegree, Arc<Subquotient>>,
}

impl ExtAlgebra<QueryModuleResolution> {
    /// Ensure the resolution of `k` is computed through the given stem.
    pub fn compute_through_stem(&self, max: Bidegree) {
        self.resolution.compute_through_stem(max);
    }
}

impl ExtModule<QueryModuleResolution> {
    /// Build an [`ExtModule`] from a resolution of `M`, deriving the unit `k` via [`get_unit`].
    ///
    /// This may prompt for the unit's save directory when `M != k` (see [`get_unit`]); for a fully
    /// non-interactive setup, use [`ExtModule::new`] with an explicit ring instead.
    pub fn from_resolution(resolution: Arc<QueryModuleResolution>) -> anyhow::Result<Self> {
        let (_, unit) = get_unit(Arc::clone(&resolution))?;
        Ok(Self::new(resolution, Arc::new(ExtAlgebra::new(unit))))
    }

    /// Ensure both the module's resolution and the ring's resolution are computed through the given
    /// stem.
    pub fn compute_through_stem(&self, max: Bidegree) {
        self.algebra.compute_through_stem(max);
        if !self.is_unit() {
            self.resolution.compute_through_stem(max);
        }
    }
}

impl<CC: FreeChainComplex> ExtModule<CC> {
    /// Build $\Ext(M, k)$ from a resolution of `M` and the ring $\Ext(k, k)$.
    pub fn new(resolution: Arc<CC>, algebra: Arc<ExtAlgebra<CC>>) -> Self {
        assert_eq!(resolution.prime(), algebra.prime());
        Self {
            resolution,
            algebra,
            products: DashMap::new(),
            differential: None,
            cup: None,
            cup_cache: DashMap::new(),
            delta_qi_cache: DashMap::new(),
            cohomology_cache: DashMap::new(),
        }
    }

    /// Build the module `M == k`, i.e. $\Ext(k, k)$ as a module over itself, sharing one resolution
    /// (and hence one ring cache) between the module and its ring.
    pub fn over_unit(algebra: Arc<ExtAlgebra<CC>>) -> Self {
        let resolution = Arc::clone(algebra.resolution());
        Self::new(resolution, algebra)
    }

    /// Build a module for resolution-*intrinsic* operations that do not involve the unit (notably
    /// the secondary `d2` differential), using the resolution itself as its own `k`.
    ///
    /// This avoids the unit-resolution setup (and any associated prompt) that
    /// [`from_resolution`](Self::from_resolution) performs. The product/action methods and the
    /// ring-side queries are only meaningful here when `M == k`; for products with `M != k`, build
    /// with [`from_resolution`](Self::from_resolution) or [`new`](Self::new) instead.
    pub fn intrinsic(resolution: Arc<CC>) -> Self {
        let algebra = Arc::new(ExtAlgebra::new(Arc::clone(&resolution)));
        Self::new(resolution, algebra)
    }

    /// Attach a differential, so that the cohomology queries compute kernel mod image under it.
    ///
    /// Without one the coboundary is zero and the cohomology is the generators.
    #[must_use]
    pub fn with_differential(mut self, differential: Arc<dyn ExtDifferential>) -> Self {
        self.differential = Some(differential);
        // These caches are derived from the differential.
        self.cohomology_cache.clear();
        self.delta_qi_cache.clear();
        self
    }

    /// The differential this DGA carries, if any.
    pub fn differential(&self) -> Option<&Arc<dyn ExtDifferential>> {
        self.differential.as_ref()
    }

    /// Attach a cochain cup product, so [`massey`](Self::massey) and
    /// [`multiply_into`](Self::multiply_into) use the cochain DGA (correct on a non-minimal
    /// resolution). See [`CochainCup`].
    #[must_use]
    pub fn with_cup(mut self, cup: Arc<dyn CochainCup<CC>>) -> Self {
        self.cup = Some(cup);
        self.cup_cache.clear();
        self
    }

    /// The cochain cup product this carries, if any.
    pub fn cup(&self) -> Option<&Arc<dyn CochainCup<CC>>> {
        self.cup.as_ref()
    }

    /// The dimension of the cohomology at `b`:
    /// $\mathrm{gens}(b) - \mathrm{rank}\,\delta_{\text{out}}(b) - \mathrm{rank}\,\delta_{\text{in}}(b)$.
    ///
    /// `None` if either differential at `b` is out of the computed range. An unknown incoming rank
    /// is not read as zero: that would overstate the cohomology at the edge of the computed region.
    pub fn cohomology_dimension(&self, b: Bidegree) -> Option<usize> {
        let gens = self.cochain_dimension(b);
        let Some(d) = &self.differential else {
            return Some(gens);
        };
        // Nothing survives in an empty bidegree, so neither differential has to be known.
        if gens == 0 {
            return Some(0);
        }
        let shift = d.shift();
        let source = Bidegree::n_s(b.n() - shift.n(), b.s() - shift.s());
        // An undersized matrix would understate a rank and overstate the cohomology.
        let mut out = d.matrix(b)?;
        assert_eq!(
            out.rows(),
            gens,
            "ExtDifferential::matrix({b:?}) must have gens = {gens} rows, got {}",
            out.rows()
        );
        let rank_out = out.row_reduce();
        let mut incoming = d.matrix(source)?;
        assert_eq!(
            incoming.columns(),
            gens,
            "ExtDifferential::matrix({source:?}) into {b:?} must have gens = {gens} columns, got \
             {}",
            incoming.columns()
        );
        let rank_in = incoming.row_reduce();
        // ker ⊇ im requires d∘d = 0. `ExtDifferential` is a public trait, so a malformed
        // implementor can violate it; panic rather than wrap around.
        Some(gens.checked_sub(rank_out + rank_in).unwrap_or_else(|| {
            panic!(
                "ExtDifferential violates d∘d=0 at {b:?}: rank_out={rank_out}, rank_in={rank_in}, \
                 gens={gens}"
            )
        }))
    }

    /// The cohomology at `b` as a [`Subquotient`] of the generators, giving representatives of the
    /// surviving classes.
    ///
    /// The numerator is $\ker(\delta \text{ out of } b)$ and the denominator is
    /// $\operatorname{im}(\delta \text{ into } b)$.
    ///
    /// `None` if either differential at `b` is out of the computed range, as with
    /// [`cohomology_dimension`](Self::cohomology_dimension).
    pub fn cohomology_subquotient(&self, b: Bidegree) -> Option<Subquotient> {
        self.cohomology(b).map(|h| (*h).clone())
    }

    /// The cohomology at `b`, memoised: its `dimension()` is the Ext dimension and its `gens()` are
    /// cocycle representatives of the basis classes, behind [`dimension`](Self::dimension),
    /// [`lift`](Self::lift) and [`project`](Self::project).
    ///
    /// `None` when either differential at `b` is out of the computed range.
    pub fn cohomology(&self, b: Bidegree) -> Option<Arc<Subquotient>> {
        if let Some(h) = self.cohomology_cache.get(&b) {
            return Some(Arc::clone(&h));
        }
        let h = Arc::new(self.compute_cohomology(b)?);
        Some(Arc::clone(
            self.cohomology_cache.entry(b).or_insert(h).value(),
        ))
    }

    fn compute_cohomology(&self, b: Bidegree) -> Option<Subquotient> {
        let p = self.prime();
        let dim = self.cochain_dimension(b);
        if dim == 0 {
            return Some(Subquotient::new_full(p, 0));
        }
        let Some(d) = &self.differential else {
            return Some(Subquotient::new_full(p, dim));
        };

        // Numerator: ker(δ out of b), via the standard augmented-identity kernel.
        let out = d.matrix(b)?;
        assert_eq!(
            out.rows(),
            dim,
            "ExtDifferential::matrix({b:?}) must have gens(b) = {dim} rows, got {}",
            out.rows()
        );
        let target_dim = out.columns();
        let mut aug = AugmentedMatrix::<2>::new(p, dim, [target_dim, dim]);
        aug.segment(1, 1).add_identity();
        for i in 0..dim {
            aug.row_mut(i).slice_mut(0, target_dim).add(out.row(i), 1);
        }
        aug.row_reduce();
        let numerator = aug.compute_kernel();

        // Denominator: im(δ into b) = row space of δ out of the source bidegree.
        let shift = d.shift();
        let source = Bidegree::n_s(b.n() - shift.n(), b.s() - shift.s());
        let m = d.matrix(source)?;
        assert_eq!(
            m.columns(),
            dim,
            "ExtDifferential::matrix({source:?}) into {b:?} must have gens(b) = {dim} columns, \
             got {}",
            m.columns()
        );
        let denominator = Subspace::from_matrix(m);

        Some(Subquotient::from_parts(numerator, denominator))
    }

    /// The resolution of `M` backing this module.
    pub fn resolution(&self) -> &Arc<CC> {
        &self.resolution
    }

    /// The ring $\Ext(k, k)$ this is a module over.
    pub fn algebra(&self) -> &Arc<ExtAlgebra<CC>> {
        &self.algebra
    }

    /// Whether `M == k`, i.e. the module shares its resolution with its ring.
    pub fn is_unit(&self) -> bool {
        Arc::ptr_eq(&self.resolution, self.algebra.resolution())
    }

    /// The prime of the underlying resolution.
    pub fn prime(&self) -> ValidPrime {
        self.resolution.prime()
    }

    /// The dimension of $\Ext^{s,t}(M, k)$ at the given bidegree: the dimension of the cohomology
    /// of the attached [`differential`](Self::differential). With no differential this is the
    /// generator count. Returns `0` when out of the computed range.
    pub fn dimension(&self, b: Bidegree) -> usize {
        self.cohomology_dimension(b).unwrap_or(0)
    }

    /// The number of **cochain generators** at `b`, the ambient dimension of the cochain group the
    /// cohomology is a subquotient of. It equals [`dimension`](Self::dimension) exactly when the
    /// coboundary is zero. An uncomputed bidegree has no cochains.
    pub fn cochain_dimension(&self, b: Bidegree) -> usize {
        self.differential
            .as_ref()
            .and_then(|d| d.dimension(b))
            .unwrap_or_else(|| {
                if b.s() >= 0 && self.resolution.has_computed_bidegree(b) {
                    self.resolution.number_of_gens_in_bidegree(b)
                } else {
                    0
                }
            })
    }

    /// The basis classes of $\Ext(M, k)$ at the given bidegree.
    pub fn basis(&self, b: Bidegree) -> Vec<BidegreeGenerator> {
        (0..self.dimension(b))
            .map(|i| BidegreeGenerator::new(b, i))
            .collect()
    }

    /// A class in $\Ext(M, k)$ from its coordinates in the cohomology basis at bidegree `b`.
    pub fn element(&self, b: Bidegree, coords: &[u32]) -> BidegreeElement {
        assert_eq!(self.dimension(b), coords.len());
        BidegreeElement::new(b, FpVector::from_slice(self.prime(), coords))
    }

    /// A single basis class of $\Ext(M, k)$.
    pub fn generator(&self, g: BidegreeGenerator) -> BidegreeElement {
        let ambient = self.dimension(g.degree());
        assert!(ambient > g.idx());
        g.into_element(self.prime(), ambient)
    }

    /// Lift a cohomology **class** to a **cocycle representative** whose class is `x`. Inverse to
    /// [`project`](Self::project) up to a coboundary; the identity when there is no differential.
    pub fn lift(&self, x: &BidegreeElement) -> Cochain {
        let b = x.degree();
        let h = self
            .cohomology(b)
            .expect("lift: bidegree out of computed range");
        let mut vec = FpVector::new(self.prime(), h.ambient_dimension());
        for (i, c) in x.vec().iter_nonzero() {
            vec.as_slice_mut()
                .add(h.gens().nth(i).expect("class coordinate out of range"), c);
        }
        Cochain::new(b, vec)
    }

    /// Project a cochain to its cohomology **class** by reducing modulo coboundaries. Pass a
    /// cocycle for a meaningful class. The identity when there is no differential.
    pub fn project(&self, c: &Cochain) -> BidegreeElement {
        let b = c.degree();
        let h = self
            .cohomology(b)
            .expect("project: bidegree out of computed range");
        let mut v = FpVector::new(self.prime(), h.ambient_dimension());
        v.as_slice_mut().add(c.vec(), 1);
        let coords = h.reduce(v.as_slice_mut());
        BidegreeElement::new(b, FpVector::from_slice(self.prime(), &coords))
    }

    /// The quasi-inverse of the coboundary out of `v_deg` (given its `r × c` matrix `d`),
    /// memoised. It is intrinsic to the coboundary, so every Massey bracket landing at `v_deg`
    /// shares it.
    pub(crate) fn delta_quasi_inverse(
        &self,
        v_deg: Bidegree,
        d: &Matrix,
        r: usize,
        c: usize,
    ) -> Arc<fp::matrix::QuasiInverse> {
        if let Some(q) = self.delta_qi_cache.get(&v_deg) {
            return Arc::clone(&q);
        }
        let mut aug = AugmentedMatrix::<2>::new(self.prime(), r, [c, r]);
        for i in 0..r {
            aug.row_mut(i).slice_mut(0, c).add(d.row(i), 1);
        }
        aug.segment(1, 1).add_identity();
        aug.row_reduce();
        let qi = Arc::new(aug.compute_quasi_inverse());
        Arc::clone(self.delta_qi_cache.entry(v_deg).or_insert(qi).value())
    }

    /// A [`Cochain`] from its coordinates in the cochain-generator basis at bidegree `b`.
    pub fn cochain_element(&self, b: Bidegree, coords: &[u32]) -> Cochain {
        assert_eq!(self.cochain_dimension(b), coords.len());
        Cochain::new(b, FpVector::from_slice(self.prime(), coords))
    }

    /// A single cochain generator, as a [`Cochain`].
    pub fn cochain_generator(&self, g: BidegreeGenerator) -> Cochain {
        let ambient = self.cochain_dimension(g.degree());
        assert!(ambient > g.idx());
        Cochain::new(g.degree(), g.into_element(self.prime(), ambient).into_vec())
    }
}

impl<CC> ExtModule<CC>
where
    CC: FreeChainComplex + AugmentedChainComplex,
{
    /// The per-generator product maps for this module.
    ///
    /// When `M == k` these are the ring's maps, so the module and its ring build each one once.
    fn product_cache(&self) -> &DashMap<BidegreeGenerator, Arc<ResolutionHomomorphism<CC, CC>>> {
        if self.is_unit() {
            &self.algebra.products
        } else {
            &self.products
        }
    }

    /// The multiplication map for a single generator `g` of $\Ext(M, k)$ (`res(M) → res(k)`), built
    /// and cached on first use. The returned map is *not* guaranteed to be extended;
    /// [`multiply_into`](Self::multiply_into) extends it as needed.
    pub fn generator_product_map(
        &self,
        g: BidegreeGenerator,
    ) -> Arc<ResolutionHomomorphism<CC, CC>> {
        cached_generator_product_map(
            self.product_cache(),
            &self.resolution,
            self.algebra.resolution(),
            g,
        )
    }

    /// Left-multiplication by the class `x` (in $\Ext(M, k)$), applied to every basis generator of
    /// $\Ext(k, k)$ at bidegree `b`.
    ///
    /// Returns `None` when the product is out of the computed range — that is, when `b` or
    /// `b + x.degree()` has not been resolved — so callers never mistake an uncomputed product for a
    /// zero one. Otherwise returns a matrix with one row per generator of $\Ext(k, k)$ at `b`; row
    /// `j` is the product `x · g_j` expressed in the generator basis of $\Ext(M, k)$ at bidegree
    /// `b + x.degree()`. A computed-but-empty bidegree yields a valid zero-dimension matrix, not
    /// `None`.
    pub fn multiply_into(&self, x: &BidegreeElement, b: Bidegree) -> Option<Matrix> {
        if self.differential.is_none() {
            return products_into(
                &self.resolution,
                self.algebra.resolution(),
                self.product_cache(),
                self.prime(),
                x,
                b,
            );
        }

        // A non-minimal complex: realise `x` as a cocycle, multiply at the cochain level and
        // project each product back to its class.
        let shift = x.degree();
        let target = b + shift;
        let unit = self.algebra.resolution();
        if !unit.has_computed_bidegree(b) || !self.resolution.has_computed_bidegree(target) {
            return None;
        }
        self.cohomology(shift)?;
        self.cohomology(target)?;
        let x_cochain = self.lift(x);
        let unit_dim = unit.number_of_gens_in_bidegree(b);
        let mut matrix = Matrix::new(self.prime(), unit_dim, self.dimension(target));

        if let Some(cup) = &self.cup {
            // The untwisted tensor resolution: `x · y_j` is the class of `y_j ∪ x`, read off the
            // closed-form cup with no chain map out of the non-minimal complex.
            let cochain_cols = self.cochain_dimension(target);
            let cup_mats = self.cup_matrices(cup, shift, b, target, unit_dim);
            for (j, cup_mat) in cup_mats.iter().enumerate() {
                let mut out = FpVector::new(self.prime(), cochain_cols);
                cup_mat.apply(out.as_slice_mut(), 1, x_cochain.vec());
                let class = self.project(&Cochain::new(target, out));
                matrix.row_mut(j).add(class.vec(), 1);
            }
            return Some(matrix);
        }

        let raw = products_into(
            &self.resolution,
            unit,
            self.product_cache(),
            self.prime(),
            &BidegreeElement::new(shift, x_cochain.into_vec()),
            b,
        )?;
        for j in 0..unit_dim {
            let class = self.project(&Cochain::new(target, raw.row(j).to_owned()));
            matrix.row_mut(j).add(class.vec(), 1);
        }
        Some(matrix)
    }

    /// The cup matrices `{y_j ∪ -}` for each generator `y_j` of $\Ext(k, k)$ at `mult_deg`, mapping
    /// cochains at `src` to cochains at `target = src + mult_deg`. The self-maps realising `y_j` are
    /// the ring's cached product maps, so every module over the same ring shares them.
    fn cup_matrices(
        &self,
        cup: &Arc<dyn CochainCup<CC>>,
        src: Bidegree,
        mult_deg: Bidegree,
        target: Bidegree,
        unit_dim: usize,
    ) -> Arc<Vec<Matrix>> {
        if let Some(m) = self.cup_cache.get(&(src, mult_deg)) {
            return Arc::clone(&m);
        }
        let mats: Vec<Matrix> = (0..unit_dim)
            .map(|j| {
                let f_y = self
                    .algebra
                    .generator_product_map(BidegreeGenerator::new(mult_deg, j));
                f_y.extend_through_stem(target);
                cup.cup_matrix(&f_y, src.s(), src.t(), mult_deg)
            })
            .collect();
        Arc::clone(
            self.cup_cache
                .entry((src, mult_deg))
                .or_insert_with(|| Arc::new(mats))
                .value(),
        )
    }

    /// The cup matrix `x ∪ -` for a class `x` of $\Ext(k, k)$, mapping cochains at `src` to cochains
    /// at `src + x.degree()`, combined from the cached per-generator cup matrices.
    pub(crate) fn cup_class_matrix(
        &self,
        cup: &Arc<dyn CochainCup<CC>>,
        x: &BidegreeElement,
        src: Bidegree,
    ) -> Matrix {
        let x_deg = x.degree();
        let target = src + x_deg;
        let unit_dim = self.algebra.resolution().number_of_gens_in_bidegree(x_deg);
        let mats = self.cup_matrices(cup, src, x_deg, target, unit_dim);
        let rows = mats.first().map_or(0, Matrix::rows);
        let cols = mats.first().map_or(0, Matrix::columns);
        let mut out = Matrix::new(self.prime(), rows, cols);
        for (j, coef) in x.vec().iter_nonzero() {
            for i in 0..rows {
                out.row_mut(i).add(mats[j].row(i), coef);
            }
        }
        out
    }

    /// The product `x · y` if it lies in the computed range, else `None`. See
    /// [`multiply_into`](Self::multiply_into) for the operand conventions (`x ∈ Ext(M, k)`, `y ∈
    /// Ext(k, k)`). The result lies in bidegree `x.degree() + y.degree()`.
    pub fn try_multiply(
        &self,
        x: &BidegreeElement,
        y: &BidegreeElement,
    ) -> Option<BidegreeElement> {
        let matrix = self.multiply_into(x, y.degree())?;
        Some(combine_product(
            &matrix,
            y,
            x.degree() + y.degree(),
            self.prime(),
        ))
    }

    /// The product `x · y`, where `x ∈ Ext(M, k)` and `y ∈ Ext(k, k)`. When `M == k` both operands
    /// live in the same algebra $\Ext(k, k)$. The result lies in bidegree `x.degree() + y.degree()`.
    ///
    /// Panics if the product is out of the computed range; use
    /// [`try_multiply`](Self::try_multiply) to handle that case.
    pub fn multiply(&self, x: &BidegreeElement, y: &BidegreeElement) -> BidegreeElement {
        self.try_multiply(x, y).expect(
            "multiply: product is out of the computed range; compute further or use try_multiply",
        )
    }
}

/// Build/cache the per-generator product map `res(source) → res(target)` for generator `g`.
fn cached_generator_product_map<CC>(
    products: &DashMap<BidegreeGenerator, Arc<ResolutionHomomorphism<CC, CC>>>,
    source: &Arc<CC>,
    target: &Arc<CC>,
    g: BidegreeGenerator,
) -> Arc<ResolutionHomomorphism<CC, CC>>
where
    CC: FreeChainComplex + AugmentedChainComplex,
{
    if let Some(map) = products.get(&g) {
        return Arc::clone(&map);
    }

    let dim = source.number_of_gens_in_bidegree(g.degree());
    let mut class = vec![0u32; dim];
    class[g.idx()] = 1;

    let name = format!("prod_{}_{}_{}", g.n(), g.s(), g.idx());
    let hom = Arc::new(ResolutionHomomorphism::from_class(
        name,
        Arc::clone(source),
        Arc::clone(target),
        g.degree(),
        &class,
    ));

    Arc::clone(products.entry(g).or_insert(hom).value())
}

/// The shared body of `multiply_into`: left-multiplication by `x` (a class in `Ext(source, k)`)
/// applied to every generator of `Ext(target, k)` at bidegree `b`. Products land in `Ext(source,
/// k)` at `b + x.degree()`. Returns `None` when out of the computed range.
fn products_into<CC>(
    source: &Arc<CC>,
    target: &Arc<CC>,
    products: &DashMap<BidegreeGenerator, Arc<ResolutionHomomorphism<CC, CC>>>,
    prime: ValidPrime,
    x: &BidegreeElement,
    b: Bidegree,
) -> Option<Matrix>
where
    CC: FreeChainComplex + AugmentedChainComplex,
{
    let shift = x.degree();
    let result_deg = b + shift;

    if !target.has_computed_bidegree(b) || !source.has_computed_bidegree(result_deg) {
        return None;
    }

    let mult_dim = target.number_of_gens_in_bidegree(b);
    let res_dim = source.number_of_gens_in_bidegree(result_deg);
    let mut matrix = Matrix::new(prime, mult_dim, res_dim);

    for (i, c) in x.vec().iter_nonzero() {
        let map = cached_generator_product_map(
            products,
            source,
            target,
            BidegreeGenerator::new(shift, i),
        );
        map.extend_all();

        // `hom_k(b.t())[j][k]`: `j` indexes the multiplicand generator of `Ext(k, k)` at `b`, `k`
        // indexes the result generator of `Ext(source, k)` at `result_deg`.
        let hom_k = map.get_map(result_deg.s()).hom_k(b.t());
        for (j, row) in hom_k.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                matrix.row_mut(j).add_basis_element(k, c * v);
            }
        }
    }
    Some(matrix)
}

/// Combine the per-generator product `matrix` (rows indexed by generators of `y`'s bidegree) with
/// the coordinates of `y` into the class `x · y` at bidegree `target`.
fn combine_product(
    matrix: &Matrix,
    y: &BidegreeElement,
    target: Bidegree,
    prime: ValidPrime,
) -> BidegreeElement {
    let mut out = FpVector::new(prime, matrix.columns());
    for (j, c) in y.vec().iter_nonzero() {
        out.as_slice_mut().add(matrix.row(j), c);
    }
    BidegreeElement::new(target, out)
}

#[cfg(test)]
mod tests {
    use fp::prime::TWO;

    use super::*;
    use crate::{chain_complex::ChainComplex, utils::construct_standard};

    /// A module over itself reads the ring's product cache, so the two share each generator map.
    #[test]
    fn test_unit_module_shares_ring_cache() {
        let res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        res.compute_through_stem(Bidegree::n_s(4, 4));
        let module = ExtModule::intrinsic(res);

        let h0 = BidegreeGenerator::new(Bidegree::n_s(0, 1), 0);
        assert!(Arc::ptr_eq(
            &module.generator_product_map(h0),
            &module.algebra().generator_product_map(h0),
        ));
    }

    #[test]
    fn test_zero_differential_cohomology_is_generators() {
        // The field/minimal case: with no differential the DGA cohomology is just
        // the generators — "taking the Ext" is a no-op.
        let res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        res.compute_through_stem(Bidegree::n_s(8, 8));
        let alg = ExtModule::intrinsic(res);
        for s in 0..=8 {
            for n in 0..=8 {
                let b = Bidegree::n_s(n, s);
                assert_eq!(alg.cohomology_dimension(b), Some(alg.dimension(b)));
            }
        }
    }

    #[test]
    fn test_differential_cohomology_kills_kernel_and_image() {
        // A synthetic rank-1 differential (0,2) -> (0,1) must kill both ends in
        // cohomology: h_0^2 by the outgoing rank, h_0 by the incoming rank. An
        // untouched bidegree (h_1) is unchanged.
        // Shaped per the `matrix` contract, with the single nonzero entry d(h_0^2) = h_0.
        struct MockDiff {
            dims: Arc<dyn Fn(Bidegree) -> usize + Send + Sync>,
        }
        impl ExtDifferential for MockDiff {
            fn shift(&self) -> Bidegree {
                Bidegree::n_s(0, -1) // lowers filtration: (0,2) -> (0,1)
            }

            fn matrix(&self, b: Bidegree) -> Option<Matrix> {
                let rows = (self.dims)(b);
                let cols = (self.dims)(b + self.shift());
                let mut m = Matrix::new(TWO, rows, cols);
                if b == Bidegree::n_s(0, 2) && rows == 1 && cols == 1 {
                    m.row_mut(0).set_entry(0, 1);
                }
                Some(m)
            }
        }

        let res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        res.compute_through_stem(Bidegree::n_s(8, 8));
        let dims: Arc<dyn Fn(Bidegree) -> usize + Send + Sync> = {
            let res = Arc::clone(&res);
            Arc::new(move |b: Bidegree| {
                if b.n() >= 0 && b.s() >= 0 && res.has_computed_bidegree(b) {
                    res.number_of_gens_in_bidegree(b)
                } else {
                    0
                }
            })
        };
        let alg =
            ExtModule::intrinsic(Arc::clone(&res)).with_differential(Arc::new(MockDiff { dims }));

        // Sanity: all three source bidegrees are 1-dimensional on the E-page.
        assert_eq!(alg.cochain_dimension(Bidegree::n_s(0, 1)), 1); // h_0
        assert_eq!(alg.cochain_dimension(Bidegree::n_s(0, 2)), 1); // h_0^2
        assert_eq!(alg.cochain_dimension(Bidegree::n_s(1, 1)), 1); // h_1

        // `dimension` is the cohomology: the two killed classes drop to 0, h_1 survives.
        assert_eq!(alg.dimension(Bidegree::n_s(0, 2)), 0);
        assert_eq!(alg.dimension(Bidegree::n_s(0, 1)), 0);
        assert_eq!(alg.dimension(Bidegree::n_s(1, 1)), 1);

        assert_eq!(alg.cohomology_dimension(Bidegree::n_s(0, 2)), Some(0)); // outgoing rank 1
        assert_eq!(alg.cohomology_dimension(Bidegree::n_s(0, 1)), Some(0)); // incoming rank 1
        assert_eq!(alg.cohomology_dimension(Bidegree::n_s(1, 1)), Some(1)); // untouched

        assert_eq!(
            alg.cohomology_subquotient(Bidegree::n_s(0, 2))
                .unwrap()
                .dimension(),
            0
        );
        assert_eq!(
            alg.cohomology_subquotient(Bidegree::n_s(0, 1))
                .unwrap()
                .dimension(),
            0
        );
        assert_eq!(
            alg.cohomology_subquotient(Bidegree::n_s(1, 1))
                .unwrap()
                .dimension(),
            1
        );
    }

    #[test]
    fn test_sphere_products() {
        let res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        res.compute_through_stem(Bidegree::n_s(8, 8));
        let module = ExtModule::intrinsic(res);

        // h_i live in Ext^{1, *}: h_0 = (n=0, s=1), h_1 = (n=1, s=1), h_2 = (n=3, s=1).
        let h0 = module.generator(BidegreeGenerator::new(Bidegree::n_s(0, 1), 0));
        let h1 = module.generator(BidegreeGenerator::new(Bidegree::n_s(1, 1), 0));

        // h_0^2 is the nonzero generator of Ext^{2,2} = (n=0, s=2).
        let h0_sq = module.multiply(&h0, &h0);
        assert_eq!(h0_sq.degree(), Bidegree::n_s(0, 2));
        assert_eq!(module.dimension(Bidegree::n_s(0, 2)), 1);
        assert!(!h0_sq.vec().is_zero(), "h_0^2 should be nonzero");

        // The Adams relations h_0 h_1 = 0 = h_1 h_0.
        assert!(
            module.multiply(&h0, &h1).vec().is_zero(),
            "h_0 h_1 should vanish"
        );
        assert!(
            module.multiply(&h1, &h0).vec().is_zero(),
            "h_1 h_0 should vanish"
        );

        // Cross-check `multiply` against a direct `hom_k` read for h_0 · h_1.
        let rows = module
            .multiply_into(&h0, h1.degree())
            .expect("h_0 · h_1 is in range");
        let direct: u32 = rows.row(0).iter().sum();
        assert_eq!(direct, 0);
    }

    /// Products of `Ext(M, k)` classes with `M != k`, where the chain maps run `res(M) → res(k)`.
    /// The unit `1 ∈ Ext^{0,0}(k, k)` acts trivially, so `x · 1 = x` for any `x ∈ Ext(M, k)`.
    #[test]
    fn test_non_unit_products() {
        let max = Bidegree::n_s(8, 8);
        let unit = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let m = Arc::new(construct_standard::<false, _, _>("C2", None).unwrap());
        unit.compute_through_stem(max);
        m.compute_through_stem(max);
        let module = ExtModule::new(m, Arc::new(ExtAlgebra::new(unit)));
        assert!(!module.is_unit(), "C2 is not the sphere, so M != k");

        // The unit class 1 ∈ Ext^{0,0}(k, k).
        let unit_deg = Bidegree::n_s(0, 0);
        assert_eq!(module.algebra().dimension(unit_deg), 1);
        let one = module
            .algebra()
            .generator(BidegreeGenerator::new(unit_deg, 0));

        // The bottom class of Ext(C2, k) at (0, 0); x · 1 = x.
        assert_eq!(module.dimension(Bidegree::n_s(0, 0)), 1);
        let x = module.generator(BidegreeGenerator::new(Bidegree::n_s(0, 0), 0));
        let prod = module.multiply(&x, &one);
        assert_eq!(prod.degree(), x.degree());
        assert_eq!(
            prod.vec().iter().collect::<Vec<_>>(),
            x.vec().iter().collect::<Vec<_>>(),
            "x · 1 = x"
        );
    }

    /// Exercise an odd prime (`p = 3`), the regime where the product API's up-to-Koszul-sign caveat
    /// bites. We assert only the sign-robust fact that `a_0^2 != 0` (the `a_0`-Bockstein tower).
    #[test]
    fn test_odd_prime_products() {
        let res = Arc::new(construct_standard::<false, _, _>("S_3", None).unwrap());
        res.compute_through_stem(Bidegree::n_s(4, 4));
        let module = ExtModule::intrinsic(res);

        // a_0 ∈ Ext^{1,1}(F_3, F_3) at (n = 0, s = 1); a_0^2 ∈ Ext^{2,2} at (0, 2) is nonzero.
        let a0_deg = Bidegree::n_s(0, 1);
        assert_eq!(module.dimension(a0_deg), 1);
        let a0 = module.generator(BidegreeGenerator::new(a0_deg, 0));
        let a0_sq = module.multiply(&a0, &a0);
        assert_eq!(a0_sq.degree(), Bidegree::n_s(0, 2));
        assert!(
            !a0_sq.vec().is_zero(),
            "a_0^2 should be nonzero at p = 3 (up to sign)"
        );
    }

    /// Check that `class_product_map(x)` induces the same products as [`ExtModule::multiply_into`],
    /// which sums the per-generator maps at the `hom_k` level rather than the chain level.
    fn check_class_product_map<CC: FreeChainComplex + AugmentedChainComplex>(
        module: &ExtModule<CC>,
        x: &BidegreeElement,
        max: Bidegree,
    ) {
        let algebra = module.algebra();
        let map = algebra.class_product_map(x, max);
        map.extend_all();

        let mut compared = 0;
        for b in algebra.resolution().iter_nonzero_stem() {
            // `multiply_into` returns `None` once `b + x.degree()` is out of the computed range.
            let Some(reference) = module.multiply_into(x, b) else {
                continue;
            };
            let target = b + x.degree();
            let hom_k = map.get_map(target.s()).hom_k(b.t());
            assert_eq!(reference.rows(), hom_k.len());
            for (j, row) in hom_k.iter().enumerate() {
                let via_ref: Vec<u32> = reference.row(j).iter().collect();
                assert_eq!(&via_ref, row, "product mismatch at multiplicand {b}");
                compared += 1;
            }
        }
        assert!(compared > 0, "expected at least one product comparison");
    }

    /// A multi-generator class takes the chain-level
    /// [`ResolutionHomomorphism::linear_combination`] path; agreeing with
    /// [`ExtModule::multiply_into`] pins `linear_combination`.
    #[test]
    fn test_class_product_map_multi_generator() {
        let res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let max = Bidegree::n_s(20, 9);
        res.compute_through_stem(max);
        let module = ExtModule::intrinsic(res);

        // (n = 15, s = 5) is the first bidegree of Ext(F_2, F_2) with two generators.
        let x_deg = Bidegree::n_s(15, 5);
        assert_eq!(module.algebra().dimension(x_deg), 2);
        let x = module.algebra().element(x_deg, &[1, 1]);
        check_class_product_map(&module, &x, max);
    }

    /// A single generator with a coefficient other than one is a one-summand
    /// [`ResolutionHomomorphism::linear_combination`].
    #[test]
    fn test_class_product_map_scaled_generator() {
        let res = Arc::new(construct_standard::<false, _, _>("S_3", None).unwrap());
        let max = Bidegree::n_s(20, 6);
        res.compute_through_stem(max);
        let module = ExtModule::intrinsic(res);

        let a0 = Bidegree::n_s(0, 1);
        assert_eq!(module.algebra().dimension(a0), 1);
        let x = module.algebra().element(a0, &[2]);
        check_class_product_map(&module, &x, max);
    }

    /// The zero class has no summands and yields the zero map.
    #[test]
    fn test_class_product_map_zero() {
        let res = Arc::new(construct_standard::<false, _, _>("S_2", None).unwrap());
        let max = Bidegree::n_s(20, 9);
        res.compute_through_stem(max);
        let module = ExtModule::intrinsic(res);

        let x = module.algebra().element(Bidegree::n_s(15, 5), &[0, 0]);
        check_class_product_map(&module, &x, max);
    }
}
