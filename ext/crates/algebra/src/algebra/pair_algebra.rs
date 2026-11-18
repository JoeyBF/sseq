//! This implements the notion of a split pair algebra in the sense of
//! <https://arxiv.org/abs/2105.07628v1>, whose notation we will use throughout.
//!
//! The Steenrod algebra admits a lift to a split pair algebra, which can be used to compute d2
//! differentials algorithmically.
//!
//! To keep the pair algebra business contained, we put the implementation of the Milnor algebra as
//! a pair algebra in this file instead of `milnor_algebra.rs`.

use fp::{
    prime::TWO,
    vector::{FpSlice, FpSliceMut, FpVector},
};
use rustc_hash::FxHasher;

use crate::{Algebra, combinatorics};

type HashMap<K, V> = hashbrown::HashMap<K, V, std::hash::BuildHasherDefault<FxHasher>>;

use std::{
    io,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

/// A lift of an algebra to a split pair algebra. See module introduction for more.
// We can't use `enum_dispatch` here because of the associated type.
pub trait PairAlgebra: Algebra {
    /// An element in the cohomological degree zero part of the pair algebra. This tends to not be
    /// a ring over Fp, so we let the algebra specify how it wants to represent the elements.
    type Element: Send + Sync;

    fn element_is_zero(elt: &Self::Element) -> bool;

    /// Assert that `elt` is in the image of the differential. Drop the data recording the
    /// complement of the image of the differential.
    fn finalize_element(_elt: &mut Self::Element) {}

    /// Create a new zero element in the given degree.
    fn new_pair_element(&self, degree: i32) -> Self::Element;

    /// Given $r, s \in \pi_0(A)$, compute $\sigma(r) \sigma(s)$ and add the result to
    /// `result`.
    fn sigma_multiply_basis(
        &self,
        result: &mut Self::Element,
        coeff: u32,
        r_degree: i32,
        r_idx: usize,
        s_degree: i32,
        s_idx: usize,
    );

    /// Same as [`PairAlgebra::sigma_multiply_basis`] but with non-basis elements.
    fn sigma_multiply(
        &self,
        result: &mut Self::Element,
        coeff: u32,
        r_degree: i32,
        r: FpSlice,
        s_degree: i32,
        s: FpSlice,
    ) {
        if coeff == 0 {
            return;
        }
        for (r_idx, c) in r.iter_nonzero() {
            for (s_idx, d) in s.iter_nonzero() {
                self.sigma_multiply_basis(result, c * d * coeff, r_degree, r_idx, s_degree, s_idx);
            }
        }
    }

    /// Compute $A(r, s)$ and write the result to `result`.
    fn a_multiply(
        &self,
        result: FpSliceMut,
        coeff: u32,
        r_degree: i32,
        r: FpSlice,
        s_degree: i32,
        s: &Self::Element,
    );

    /// The element p is classified by a filtration on element in Ext of the underlying algebra,
    /// which is represented by an indecomposable in degree 1. This returns the index of said
    /// indecomposable.
    fn p_tilde(&self) -> usize;

    fn element_to_bytes(&self, elt: &Self::Element, buffer: &mut impl io::Write) -> io::Result<()>;
    fn element_from_bytes(
        &self,
        degree: i32,
        buffer: &mut impl io::Read,
    ) -> io::Result<Self::Element>;
}

use std::cell::RefCell;

use crate::{
    MilnorAlgebra,
    milnor_algebra::{
        MilnorAlgebraInner, MilnorBasisElement as MilnorElt, NoExterior, PPart, PPartAllocation,
        for_each_term_mod4,
    },
};

macro_rules! sub {
    ($elt:ident, $k:expr, $n:expr) => {
        if $k > 0 {
            let entry = $elt.p_part.get($k - 1);
            if entry < (1 << $n) {
                continue;
            }
            $elt.p_part.set($k - 1, entry - (1 << $n));
            $elt.degree -= combinatorics::xi_degrees(TWO)[$k - 1] * (1 << $n);
        }
    };
}
macro_rules! unsub {
    ($elt:ident, $k:expr, $n:expr) => {
        if $k > 0 {
            $elt.p_part.set($k - 1, $elt.p_part.get($k - 1) + (1 << $n));
            $elt.degree += combinatorics::xi_degrees(TWO)[$k - 1] * (1 << $n);
        }
    };
}

pub struct MilnorPairElement {
    pub(crate) ones: FpVector,
    pub(crate) twos: FpVector,
    #[cfg(debug_assertions)]
    degree: i32,
    pub(crate) ys: Vec<Vec<FpVector>>,
}

/// Forward a [`PairAlgebra`] method to the classical shape, which is the only one that has one.
macro_rules! dispatch_pair_milnor {
    () => {};
    ($vis:vis fn $method:ident(&self$(, $arg:ident: $ty:ty )*$(,)?) $(-> $ret:ty)?; $($tail:tt)*) => {
        $vis fn $method(&self, $($arg: $ty),* ) $(-> $ret)* {
            match self {
                MilnorAlgebra::Polynomial(a) => a.$method($($arg),*),
                MilnorAlgebra::Exterior(_) => unimplemented!(
                    "the secondary Steenrod algebra is only defined for the classical algebra at p = 2"
                ),
            }
        }
        dispatch_pair_milnor!{$($tail)*}
    };
}

/// Written for the classical algebra at `p = 2`, hence the shape bound and the assert on the
/// prime; see [`Bialgebra`](crate::algebra::Bialgebra) for the same pair of guards.
impl PairAlgebra for MilnorAlgebraInner<NoExterior> {
    type Element = MilnorPairElement;

    fn new_pair_element(&self, degree: i32) -> Self::Element {
        let p = self.prime();
        assert_eq!(p, TWO);

        let max_k = if degree == 0 {
            0
        } else {
            fp::prime::log2(degree as usize) + 1
        };
        let mut ys = Vec::with_capacity(max_k);
        for k in 0..max_k {
            let rem_degree = degree as usize + 1 - (1 << k);
            let max_l = fp::prime::log2(rem_degree) + 1;
            let mut row = Vec::with_capacity(max_l);
            for l in 0..max_l {
                row.push(FpVector::new(
                    p,
                    self.dimension((rem_degree - (1 << l)) as i32),
                ));
            }
            ys.push(row);
        }

        MilnorPairElement {
            ones: FpVector::new(p, self.dimension(degree)),
            twos: FpVector::new(p, self.dimension(degree)),
            ys,
            #[cfg(debug_assertions)]
            degree,
        }
    }

    fn element_is_zero(elt: &Self::Element) -> bool {
        elt.ones.is_zero()
            && elt.twos.is_zero()
            && elt.ys.iter().all(|v| v.iter().all(|x| x.is_zero()))
    }

    fn p_tilde(&self) -> usize {
        0
    }

    /// Assert that `elt` is in the image of the differential. Drop the data recording the
    /// complement of the image of the differential.
    fn finalize_element(elt: &mut Self::Element) {
        assert!(elt.ones.is_zero());
        elt.ones = FpVector::new(elt.twos.prime(), 0);
    }

    fn sigma_multiply_basis(
        &self,
        result: &mut Self::Element,
        coeff: u32,
        r_degree: i32,
        r_idx: usize,
        s_degree: i32,
        s_idx: usize,
    ) {
        #[cfg(debug_assertions)]
        assert_eq!(r_degree + s_degree, result.degree);

        // The cached terms are those of the product with coefficient 1, which is the only
        // coefficient the composites pass.
        if coeff != 1 || !*SIGMA_CACHE_ENABLED {
            self.sigma_multiply_basis_direct(result, coeff, r_degree, r_idx, s_degree, s_idx);
            return;
        }
        let key = (r_degree, r_idx as u32, s_degree, s_idx as u32);
        let terms = self.sigma_cache.get(key).unwrap_or_else(|| {
            let degree = r_degree + s_degree;
            let mut scratch = self
                .sigma_cache
                .take_scratch(degree)
                .unwrap_or_else(|| self.new_pair_element(degree));
            self.sigma_multiply_basis_direct(&mut scratch, 1, r_degree, r_idx, s_degree, s_idx);
            let terms = SigmaCache::terms(&scratch);
            SigmaCache::clear_terms(&mut scratch, &terms);
            self.sigma_cache.put_scratch(degree, scratch);
            self.sigma_cache.insert(key, Arc::clone(&terms));
            terms
        });
        SigmaCache::add_terms(result, &terms);
    }

    fn a_multiply(
        &self,
        result: FpSliceMut,
        coeff: u32,
        r_degree: i32,
        r: FpSlice<'_>,
        s_degree: i32,
        s: &Self::Element,
    ) {
        self.a_multiply_impl(result, coeff, r_degree, r, s_degree, s);
    }

    fn element_to_bytes(&self, elt: &Self::Element, buffer: &mut impl io::Write) -> io::Result<()> {
        elt.twos.to_bytes(buffer)?;
        for row in &elt.ys {
            for v in row {
                v.to_bytes(buffer)?;
            }
        }
        Ok(())
    }

    fn element_from_bytes(
        &self,
        degree: i32,
        buffer: &mut impl io::Read,
    ) -> io::Result<Self::Element> {
        let p = self.prime();
        assert_eq!(p, TWO);

        let max_k = if degree == 0 {
            0
        } else {
            fp::prime::log2(degree as usize) + 1
        };

        let twos = FpVector::from_bytes(p, self.dimension(degree), buffer)?;

        let mut ys = Vec::with_capacity(max_k);
        for k in 0..max_k {
            let rem_degree = degree as usize + 1 - (1 << k);
            let max_l = fp::prime::log2(rem_degree) + 1;
            let mut row = Vec::with_capacity(max_l);
            for l in 0..max_l {
                row.push(FpVector::from_bytes(
                    p,
                    self.dimension((rem_degree - (1 << l)) as i32),
                    buffer,
                )?);
            }
            ys.push(row);
        }

        Ok(MilnorPairElement {
            ones: FpVector::new(p, 0),
            twos,
            ys,
            #[cfg(debug_assertions)]
            degree,
        })
    }
}

/// Whether [`SigmaCache`] is used. `EXT_SIGMA_CACHE=0` turns it off, to compare against.
static SIGMA_CACHE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("EXT_SIGMA_CACHE").as_deref() != Ok("0"));

/// The most products [`SigmaCache`] keeps, from `EXT_SIGMA_CACHE_MAX`; unbounded by default.
static SIGMA_CACHE_MAX: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("EXT_SIGMA_CACHE_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX)
});

/// The number of shards of [`SigmaCache`]. More shards make two threads less likely to want the
/// same lock at once.
const SIGMA_SHARDS: usize = 256;

/// One shard of [`SigmaCache`], on cache lines of its own so that taking one shard's lock does not
/// stall threads using its neighbours.
#[repr(align(128))]
#[derive(Default)]
struct SigmaShard(Mutex<HashMap<(i32, u32, i32, u32), Arc<[u64]>>>);

/// The terms of products of pairs of basis elements in the pair algebra, keyed by the two basis
/// elements.
///
/// Building a secondary composite multiplies the same pairs of basis elements over and over, from
/// different generators, and a product is a few dozen Milnor products. A product's terms are
/// encoded as `u64`s: a coefficient mod 4 of the product part is `KIND_MOD4 | c << 32 | idx`, and
/// a nonzero entry of `ys[i][j]` is `KIND_Y | i << 40 | j << 32 | idx`.
pub(crate) struct SigmaCache {
    shards: Box<[SigmaShard]>,
    len: AtomicUsize,
    /// Zero pair elements by degree, to compute missing products into without allocating.
    scratch: Box<[ScratchShard]>,
}

/// One shard of [`SigmaCache::scratch`], padded like [`SigmaShard`].
#[repr(align(128))]
#[derive(Default)]
struct ScratchShard(Mutex<HashMap<i32, Vec<MilnorPairElement>>>);

const KIND_MOD4: u64 = 0;
const KIND_Y: u64 = 1 << 63;

impl Default for SigmaCache {
    fn default() -> Self {
        Self {
            shards: (0..SIGMA_SHARDS).map(|_| SigmaShard::default()).collect(),
            len: AtomicUsize::new(0),
            scratch: (0..SIGMA_SHARDS).map(|_| ScratchShard::default()).collect(),
        }
    }
}

impl SigmaCache {
    /// The shard holding `key`.
    fn shard(
        &self,
        key: &(i32, u32, i32, u32),
    ) -> &Mutex<HashMap<(i32, u32, i32, u32), Arc<[u64]>>> {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        &self.shards[(hasher.finish() >> 32) as usize % SIGMA_SHARDS].0
    }

    /// The cached terms of the product keyed by `key`, if any.
    fn get(&self, key: (i32, u32, i32, u32)) -> Option<Arc<[u64]>> {
        self.shard(&key).lock().unwrap().get(&key).cloned()
    }

    /// Cache `terms` under `key`, unless the cache already holds [`SIGMA_CACHE_MAX`] products.
    fn insert(&self, key: (i32, u32, i32, u32), terms: Arc<[u64]>) {
        if self.len.load(Ordering::Relaxed) >= *SIGMA_CACHE_MAX {
            return;
        }
        if self
            .shard(&key)
            .lock()
            .unwrap()
            .insert(key, terms)
            .is_none()
        {
            self.len.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A zero pair element of degree `degree` returned by [`Self::put_scratch`], if any.
    fn take_scratch(&self, degree: i32) -> Option<MilnorPairElement> {
        let shard = &self.scratch[degree as usize % SIGMA_SHARDS].0;
        shard.lock().unwrap().get_mut(&degree)?.pop()
    }

    /// Keep `elt`, which must be zero, for [`Self::take_scratch`].
    fn put_scratch(&self, degree: i32, elt: MilnorPairElement) {
        let shard = &self.scratch[degree as usize % SIGMA_SHARDS].0;
        shard.lock().unwrap().entry(degree).or_default().push(elt);
    }

    /// Zero the entries of `elt` named by `terms`, its terms from [`Self::terms`].
    fn clear_terms(elt: &mut MilnorPairElement, terms: &[u64]) {
        for &t in terms {
            let idx = (t & 0xffff_ffff) as usize;
            if t & KIND_Y != 0 {
                let i = ((t >> 40) & 0xff) as usize;
                let j = ((t >> 32) & 0xff) as usize;
                elt.ys[i][j].set_entry(idx, 0);
            } else {
                elt.ones.set_entry(idx, 0);
                elt.twos.set_entry(idx, 0);
            }
        }
    }

    /// The nonzero terms of `elt`, which must start from zero in a pair element.
    fn terms(elt: &MilnorPairElement) -> Arc<[u64]> {
        let mut terms = Vec::new();
        let mut indices: Vec<usize> = elt
            .ones
            .iter_nonzero()
            .chain(elt.twos.iter_nonzero())
            .map(|(idx, _)| idx)
            .collect();
        indices.sort_unstable();
        indices.dedup();
        for idx in indices {
            let c = elt.ones.entry(idx) + 2 * elt.twos.entry(idx);
            terms.push(KIND_MOD4 | u64::from(c) << 32 | idx as u64);
        }
        for (i, row) in elt.ys.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                for (idx, _) in v.iter_nonzero() {
                    terms.push(KIND_Y | (i as u64) << 40 | (j as u64) << 32 | idx as u64);
                }
            }
        }
        terms.into()
    }

    /// Add terms recorded by [`Self::terms`] to `result`.
    fn add_terms(result: &mut MilnorPairElement, terms: &[u64]) {
        for &t in terms {
            let idx = (t & 0xffff_ffff) as usize;
            if t & KIND_Y != 0 {
                let i = ((t >> 40) & 0xff) as usize;
                let j = ((t >> 32) & 0xff) as usize;
                result.ys[i][j].add_basis_element(idx, 1);
            } else {
                // Add c mod 4 to the digit pair (ones, twos).
                let c = ((t >> 32) & 0x3) as u32;
                let sum = result.ones.entry(idx) + c;
                result.ones.set_entry(idx, sum & 1);
                result.twos.add_basis_element(idx, (sum >> 1) & 1);
            }
        }
    }
}

impl MilnorAlgebraInner<NoExterior> {
    /// [`PairAlgebra::sigma_multiply_basis`] computed from scratch.
    pub(crate) fn sigma_multiply_basis_direct(
        &self,
        result: &mut MilnorPairElement,
        coeff: u32,
        r_degree: i32,
        r_idx: usize,
        s_degree: i32,
        s_idx: usize,
    ) {
        let r = self.basis_element_from_index(r_degree, r_idx);
        let s = self.basis_element_from_index(s_degree, s_idx);
        self.sigma_y_terms(result, coeff, r, s);
        self.sigma_product_terms(result, coeff, &r, &s);
    }

    /// The Y terms of `sigma(r, s)`: for each `k` with `s_{k+1} > 0` and `0 <= m < n`, the
    /// product `P(r - 2^k (e_m + e_n)) P(s - e_{k+1})` mod 2, added into `ys[m + k][n + k]`, where
    /// `e_0` subtracts nothing.
    pub(crate) fn sigma_y_terms(
        &self,
        result: &mut MilnorPairElement,
        coeff: u32,
        mut r: MilnorElt,
        mut s: MilnorElt,
    ) {
        PPartAllocation::with_local(|mut allocation| {
            for k in 0..s.p_part.len() {
                sub!(s, k + 1, 0);
                for n in 1..r.p_part.len() + 1 {
                    sub!(r, n, k);
                    for m in 0..n {
                        sub!(r, m, k);
                        allocation = self.multiply_with_allocation(
                            result.ys[m + k][n + k].as_slice_mut(),
                            coeff,
                            r,
                            s,
                            i32::MAX,
                            allocation,
                        );
                        unsub!(r, m, k);
                    }
                    unsub!(r, n, k);
                }
                unsub!(s, k + 1, 0);
            }
            allocation
        });
    }

    /// The product terms of `sigma(r, s)`: `P(r) P(s)` over Z/4, into `(ones, twos)`.
    pub(crate) fn sigma_product_terms(
        &self,
        result: &mut MilnorPairElement,
        coeff: u32,
        r: &MilnorElt,
        s: &MilnorElt,
    ) {
        let degree = r.degree + s.degree;
        // coeff should always be 1, so no need to optimize the even case.
        for_each_term_mod4(degree, r.p_part, s.p_part, |p_part, c| {
            let idx = self.basis_element_to_index(&MilnorElt {
                q_part: 0,
                p_part,
                degree,
            });
            let c = c * coeff;
            if c == 2 {
                result.twos.add_basis_element(idx, 1);
            } else {
                // c = 1 or 3
                let existing = result.ones.entry(idx);
                let c = c + existing;
                result.ones.set_entry(idx, c & 1);
                result.twos.add_basis_element(idx, (c >> 1) & 1);
            }
        });
    }

    /// [`PairAlgebra::a_multiply`].
    fn a_multiply_impl(
        &self,
        mut result: FpSliceMut,
        coeff: u32,
        r_degree: i32,
        r: FpSlice<'_>,
        s_degree: i32,
        s: &MilnorPairElement,
    ) {
        assert!(s.ones.is_zero());

        if r_degree == 0 {
            return;
        }

        // The twos terms
        for (r_idx, c) in r.iter_nonzero() {
            let mut r = self.basis_element_from_index(r_degree, r_idx);
            sub!(r, 1, 0);
            self.multiply_basis_by_element(
                result.copy(),
                coeff * c,
                r,
                s_degree,
                s.twos.as_slice(),
            );
            // No matching `unsub!`: unlike the loops above, `r` is a fresh copy of the basis
            // element on each iteration, so there is nothing to restore.
        }

        // The Y terms
        for (k, row) in s.ys.iter().enumerate() {
            for (l, vec) in row.iter().enumerate() {
                if vec.is_zero() {
                    continue;
                }
                let degree_shift = (1 << k) + (1 << l) - 1;
                let a_degree = r_degree + degree_shift - 1;
                // The product is bilinear and every A(r, Y_{k, l}) has degree `a_degree`, so sum
                // them over the terms r first and multiply once.
                let mut a_y = FpVector::new(TWO, self.dimension(a_degree));
                for (r_idx, c) in r.iter_nonzero() {
                    let r = self.basis_element_from_index(r_degree, r_idx);
                    a_y_cached(self, r, k, l, |v| a_y.add(v, c));
                }
                if a_y.is_zero() {
                    continue;
                }
                self.multiply_element_by_element(
                    result.copy(),
                    coeff,
                    a_degree,
                    a_y.as_slice(),
                    s_degree - degree_shift,
                    vec.as_slice(),
                );
            }
        }
    }
}

// Use thread-local storage to memoize a_y computation. Since the possible values of k, l grow as
// log n, in practice it is going to be at most, say, 64, and the memory usage here should be
// dwarfed by that of storing a single quasi-inverse

use std::hash::{BuildHasher, Hash, Hasher};

thread_local! {
    static AY_CACHE: RefCell<HashMap<(MilnorElt, (usize, usize)), FpVector>> = RefCell::new(HashMap::default());
}

/// Compute $A(Sq(R), Y_{k, l})$ where $a = Sq(R)$. This queries the cache and computes it using
/// [`a_y_inner`] if not available.
fn a_y_cached(
    algebra: &MilnorAlgebraInner<NoExterior>,
    a: MilnorElt,
    k: usize,
    l: usize,
    f: impl FnOnce(&FpVector),
) {
    AY_CACHE.with(|cache| {
        let cache = &mut *cache.try_borrow_mut().unwrap();
        let mut hasher = cache.hasher().build_hasher();
        a.hash(&mut hasher);
        (k, l).hash(&mut hasher);

        let raw_entry = cache.raw_entry();
        let result = raw_entry
            .from_hash(hasher.finish(), |v| v.0 == a && v.1 == (k, l))
            .map(|(_, y)| y);

        match result {
            Some(v) => f(v),
            None => {
                let v = a_y_inner(algebra, a, k, l);
                f(&v);
                cache.insert((a, (k, l)), v);
            }
        }
    })
}

/// Actually computes $A(a, Y_{k, l})$ and returns the result.
fn a_y_inner(
    algebra: &MilnorAlgebraInner<NoExterior>,
    a: MilnorElt,
    k: usize,
    l: usize,
) -> FpVector {
    let mut a = a;
    let mut result = FpVector::new(TWO, algebra.dimension(a.degree + (1 << k) + (1 << l) - 2));
    let mut t = MilnorElt {
        q_part: 0,
        p_part: PPart::zero(),
        degree: 0,
    };

    for i in 0..=a.p_part.len() {
        if i + k < l {
            continue;
        }

        sub!(a, i, k);
        for j in 0..=std::cmp::min(i + k - l, a.p_part.len()) {
            sub!(a, j, l);

            t.p_part = PPart::zero();
            t.p_part.set(k + i - 1, 1);
            t.p_part.set(l + j - 1, t.p_part.get(l + j - 1) + 1);

            t.degree = (1 << (k + i)) + (1 << (l + j)) - 2;

            // We can just read off the value of the product instead of passing through the
            // algorithm, but this is cached so problem for another day...
            algebra.multiply(result.as_slice_mut(), 1, t, a);

            unsub!(a, j, l);
        }
        unsub!(a, i, k);
    }
    result
}

/// Forwards to the classical shape, the only one with a secondary algebra.
impl PairAlgebra for MilnorAlgebra {
    type Element = MilnorPairElement;

    dispatch_pair_milnor! {
        fn new_pair_element(&self, degree: i32) -> Self::Element;
        fn sigma_multiply_basis(&self, result: &mut Self::Element, coeff: u32, r_degree: i32, r_idx: usize, s_degree: i32, s_idx: usize);
        fn sigma_multiply(&self, result: &mut Self::Element, coeff: u32, r_degree: i32, r: FpSlice, s_degree: i32, s: FpSlice);
        fn a_multiply(&self, result: FpSliceMut, coeff: u32, r_degree: i32, r: FpSlice, s_degree: i32, s: &Self::Element);
        fn element_to_bytes(&self, elt: &Self::Element, buffer: &mut impl std::io::Write) -> std::io::Result<()>;
        fn element_from_bytes(&self, degree: i32, buffer: &mut impl std::io::Read) -> std::io::Result<Self::Element>;
    }

    /// Answered directly rather than dispatched, so that it stays total at odd primes.
    fn p_tilde(&self) -> usize {
        0
    }

    /// See [`PairAlgebra::element_is_zero`].
    fn element_is_zero(elt: &Self::Element) -> bool {
        MilnorAlgebraInner::<NoExterior>::element_is_zero(elt)
    }

    /// See [`PairAlgebra::finalize_element`].
    fn finalize_element(elt: &mut Self::Element) {
        MilnorAlgebraInner::<NoExterior>::finalize_element(elt);
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};

    use super::*;
    use crate::milnor_algebra::PPartEntry;

    fn from_p_part(p_part: &[PPartEntry]) -> MilnorElt {
        let degree = p_part
            .iter()
            .enumerate()
            .map(|(i, &n)| combinatorics::xi_degrees(TWO)[i] * (n as i32))
            .sum();

        MilnorElt {
            q_part: 0,
            p_part: PPart::from_slice(p_part),
            degree,
        }
    }

    /// One CPU thread's time for the Y terms and for the product terms of `sigma(r, s)`, on
    /// `SIGMA_SPLIT_PAIRS` (default 20000) random pairs of each total degree `SIGMA_SPLIT_DEGREES`
    /// (default 100,200,300), the degree of `r` uniform. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn sigma_split_bench() {
        let degrees: Vec<i32> = std::env::var("SIGMA_SPLIT_DEGREES")
            .map(|v| v.split(',').map(|d| d.parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![100, 200, 300]);
        let pairs: usize = std::env::var("SIGMA_SPLIT_PAIRS").map_or(20000, |v| v.parse().unwrap());
        let algebra = MilnorAlgebraInner::<NoExterior>::new(TWO, false);
        algebra.compute_basis(*degrees.iter().max().unwrap() + 1);
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut rand = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        for total in degrees {
            let mut sample = Vec::with_capacity(pairs);
            while sample.len() < pairs {
                let a = 1 + rand(total as usize - 1) as i32;
                let (da, db) = (algebra.dimension(a), algebra.dimension(total - a));
                if da == 0 || db == 0 {
                    continue;
                }
                sample.push((
                    algebra.basis_element_from_index(a, rand(da)),
                    algebra.basis_element_from_index(total - a, rand(db)),
                ));
            }
            let mut result = algebra.new_pair_element(total);
            let start = std::time::Instant::now();
            for (r, s) in &sample {
                algebra.sigma_y_terms(&mut result, 1, *r, *s);
            }
            let y_s = start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            for (r, s) in &sample {
                algebra.sigma_product_terms(&mut result, 1, r, s);
            }
            let p_s = start.elapsed().as_secs_f64();
            println!(
                "SIGMASPLIT degree={total} pairs={pairs} y_us={:.2} product_us={:.2} y_share={:.2}",
                y_s / pairs as f64 * 1e6,
                p_s / pairs as f64 * 1e6,
                y_s / (y_s + p_s)
            );
        }
    }

    #[test]
    fn test_a_y() {
        let algebra = MilnorAlgebraInner::<NoExterior>::new(TWO, false);

        let mut result = FpVector::new(TWO, 0);

        let mut check = |p_part: &[PPartEntry], k, l, ans: Expect| {
            let a = from_p_part(p_part);

            let target_deg = a.degree + (1 << k) + (1 << l) - 2;
            algebra.compute_basis(target_deg + 1);
            result.set_scratch_vector_size(algebra.dimension(target_deg));
            a_y_cached(&algebra, a, k, l, |v| result.add(v, 1));
            ans.assert_eq(&algebra.element_to_string(target_deg, result.as_slice()));
        };

        check(&[1], 0, 1, expect![["P(2)"]]);
        check(&[1], 1, 2, expect![["0"]]);
        check(&[0, 1], 0, 1, expect![["P(1, 1)"]]);
        check(&[0, 2], 1, 3, expect![["P(0, 0, 2)"]]);
        check(&[1, 2], 0, 1, expect![["P(2, 2)"]]);
    }
}
