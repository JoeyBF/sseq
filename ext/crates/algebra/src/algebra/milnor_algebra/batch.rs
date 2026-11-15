//! Many Milnor products at once, at the prime 2.
//!
//! Building a matrix of a map out of a free module multiplies the same few operations into many
//! elements, and the results land in known places of known rows. A [`BatchProduct`] records one
//! such product `Sq(R) · s`, where `s` is a sum of basis elements, together with where its result
//! goes; [`multiply_batch`] evaluates a list of them into row-major `u32` limbs.
//!
//! With the `gpu` feature, large batches run on an NVIDIA GPU (see [`super::cuda`]), and
//! [`cpu_multiply_batch`] is the reference the device is checked against bit for bit.
//!
//! Only the classical algebra at `p = 2` with trivial profile is supported: the terms are ranked
//! with [`SeqnoTables`](super::SeqnoTables), which exist exactly there.

use std::sync::Arc;

use fp::vector::FpVector;

use super::{MilnorAlgebra, PPart, antidiagonal};
use crate::algebra::Algebra;

/// Marks a column of a `col_map` that the output does not keep.
pub const COL_MAP_DROP: u32 = u32::MAX;

/// One product `Sq(R) · s` and where its result lands.
///
/// `R` is basis element `r_idx` of degree `r_degree`, and `s` is the sum of the basis elements
/// `term_indices` of degree `s_degree`. The product, of degree `r_degree + s_degree`, is added
/// into row `row` with its basis element `i` at column `out_offset + i`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BatchProduct {
    pub r_degree: i32,
    pub s_degree: i32,
    /// `u32`s, as is the [`Terms`] handle a pointer: one lift step can make hundreds of millions
    /// of products, so their size is much of a lift's memory (32 bytes each, from 72).
    pub r_idx: u32,
    pub row: u32,
    pub out_offset: u32,
    /// Shared, because one list usually serves many products: it depends only on the element
    /// being multiplied, not on the operation or the row.
    pub term_indices: Terms,
}

// A lift's memory is mostly its products (see `BatchProduct`); keep them this small.
const _: () = assert!(size_of::<BatchProduct>() <= 32);

/// A product's index, row or column as the `u32` [`BatchProduct`] keeps.
///
/// # Panics
///
/// If it does not fit, which no lift comes near (a chunk's rows, a degree's Milnor basis, a
/// bidegree's width are all far below `u32::MAX`).
pub fn narrow(i: usize) -> u32 {
    u32::try_from(i).expect("a product's index exceeds u32")
}

/// The basis elements of a product's element `s`: listed, or the set bits of a range of a vector.
///
/// Either way it is shared, because one element usually serves many products. A chain map's value
/// on a generator is dense (about a fifth of its bits set at the n400 ring's stems), so listing its
/// set bits as indices takes ~14 times the memory of the bits themselves; [`Terms::bits`] keeps the
/// bits and reads them where the products are evaluated.
#[derive(Clone, Debug)]
pub struct Terms(Arc<TermsKind>);

#[derive(Debug)]
enum TermsKind {
    Indices(Arc<[usize]>),
    /// The set bits of `vec[start..end]`, as indices from `start`; `len` of them.
    Bits {
        vec: Arc<FpVector>,
        start: usize,
        end: usize,
        len: usize,
    },
}

impl Terms {
    /// The set bits of `vec[start..end]`.
    pub fn bits(vec: Arc<FpVector>, start: usize, end: usize) -> Self {
        let len = vec.slice(start, end).iter_nonzero().count();
        Self(Arc::new(TermsKind::Bits {
            vec,
            start,
            end,
            len,
        }))
    }

    pub fn len(&self) -> usize {
        match &*self.0 {
            TermsKind::Indices(v) => v.len(),
            TermsKind::Bits { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Calls `f` on each index, in ascending order.
    pub fn for_each(&self, mut f: impl FnMut(usize)) {
        match &*self.0 {
            TermsKind::Indices(v) => v.iter().for_each(|&i| f(i)),
            TermsKind::Bits {
                vec, start, end, ..
            } => vec.slice(*start, *end).iter_nonzero().for_each(|(i, _)| f(i)),
        }
    }
}

/// Terms are equal when they list the same indices, however they are stored.
impl PartialEq for Terms {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let (mut a, mut b) = (Vec::with_capacity(self.len()), Vec::with_capacity(other.len()));
        self.for_each(|i| a.push(i));
        other.for_each(|i| b.push(i));
        a == b
    }
}

impl Eq for Terms {}

impl std::hash::Hash for Terms {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        self.for_each(|i| i.hash(state));
    }
}

impl From<Arc<[usize]>> for Terms {
    fn from(v: Arc<[usize]>) -> Self {
        Self(Arc::new(TermsKind::Indices(v)))
    }
}

/// One block of output limbs, row-major.
///
/// A trait so that a backend can hand over its own buffer (for instance page-locked memory the
/// device wrote into) without copying it.
pub trait LimbBlock: Send + Sync {
    fn limbs(&self) -> &[u32];
}

impl LimbBlock for Vec<u32> {
    fn limbs(&self) -> &[u32] {
        self
    }
}

/// The result of [`multiply_batch`]: rows of `num_limbs` little-endian `u32` limbs, bit `i` of a
/// row being column `i`, stored as consecutive blocks of whole rows.
pub struct BatchOutput {
    blocks: Vec<Box<dyn LimbBlock>>,
    num_limbs: usize,
}

impl BatchOutput {
    /// Wrap blocks of whole rows, in row order.
    pub fn from_blocks(blocks: Vec<Box<dyn LimbBlock>>, num_limbs: usize) -> Self {
        debug_assert!(
            blocks
                .iter()
                .all(|b| b.limbs().len().is_multiple_of(num_limbs))
        );
        Self { blocks, num_limbs }
    }

    /// Wrap row-major limbs.
    pub fn from_limbs(limbs: Vec<u32>, num_limbs: usize) -> Self {
        Self::from_blocks(vec![Box::new(limbs)], num_limbs)
    }

    pub fn num_limbs(&self) -> usize {
        self.num_limbs
    }

    pub fn rows(&self) -> usize {
        self.blocks.iter().map(|b| b.limbs().len()).sum::<usize>() / self.num_limbs
    }

    /// The rows, in order.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[u32]> {
        let n = self.num_limbs;
        self.blocks
            .iter()
            .flat_map(move |b| b.limbs().chunks_exact(n))
    }
}

impl PartialEq for BatchOutput {
    fn eq(&self, other: &Self) -> bool {
        self.num_limbs == other.num_limbs && self.iter_rows().eq(other.iter_rows())
    }
}

impl Eq for BatchOutput {}

impl std::fmt::Debug for BatchOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchOutput")
            .field("rows", &self.rows())
            .field("num_limbs", &self.num_limbs)
            .finish()
    }
}

/// The limbs per row of an output `num_cols` wide. At least one, so that every row has a place.
pub(super) fn num_limbs(num_cols: usize) -> usize {
    num_cols.div_ceil(32).max(1)
}

/// Where the term of basis index `idx` of a product with column offset `out_offset` lands, if it
/// is kept.
///
/// Without a `col_map` this is `out_offset + idx`. With one, the output is in masked coordinates:
/// `col_map[out_offset + idx]` is the masked column of each full column, or [`COL_MAP_DROP`]. A
/// column at or past `num_limbs * 32` is dropped either way, which is how an output narrower than
/// its target degree keeps only a prefix. The device kernel applies the same rule.
#[inline]
pub(super) fn output_bit(
    out_offset: usize,
    idx: usize,
    col_map: Option<&[u32]>,
    num_limbs: usize,
) -> Option<usize> {
    let mut bit = out_offset + idx;
    if let Some(map) = col_map {
        bit = match map.get(bit) {
            Some(&m) if m != COL_MAP_DROP => m as usize,
            _ => return None,
        };
    }
    (bit / 32 < num_limbs).then_some(bit)
}

/// Whether [`multiply_batch`] supports `algebra`.
pub fn batch_applicable(algebra: &MilnorAlgebra) -> bool {
    algebra.seqno_applicable()
}

/// Whether [`multiply_batch`] can run batches on a GPU: built with the `gpu` feature, and a
/// device was found. Opens the device on first use.
pub fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    return super::cuda::available();
    #[cfg(not(feature = "gpu"))]
    false
}

/// Kernel launches so far, process-wide (always 0 without the `gpu` feature). For tests, which
/// check that a batch they compare against [`cpu_multiply_batch`] actually ran on the device.
pub fn gpu_launches() -> u64 {
    #[cfg(feature = "gpu")]
    return super::cuda::LAUNCHES.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "gpu"))]
    0
}

/// Batches that failed on the GPU and were recomputed on the CPU so far, process-wide. The result
/// is right either way, so this is how a test notices a device that is not doing the work.
pub fn gpu_failures() -> u64 {
    #[cfg(feature = "gpu")]
    return super::cuda::FAILURES.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "gpu"))]
    0
}

/// Evaluate `products` into `num_rows` rows of `num_cols` columns, on the CPU.
///
/// See [`multiply_batch`] for the requirements.
pub fn cpu_multiply_batch(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    col_map: Option<&[u32]>,
    num_rows: usize,
    products: &[BatchProduct],
) -> BatchOutput {
    let num_limbs = num_limbs(num_cols);
    let mut limbs = vec![0u32; num_rows * num_limbs];
    if products.is_empty() {
        return BatchOutput::from_limbs(limbs, num_limbs);
    }
    let tables = algebra.seqno_tables();
    for prod in products {
        let degree = prod.r_degree + prod.s_degree;
        if algebra.dimension(degree) == 0 {
            continue;
        }
        let r = algebra
            .basis_element_from_index(prod.r_degree, prod.r_idx as usize)
            .p_part;
        let row = &mut limbs[prod.row as usize * num_limbs..][..num_limbs];
        prod.term_indices.for_each(|ti| {
            let s = algebra.basis_element_from_index(prod.s_degree, ti).p_part;
            antidiagonal::for_each_term(degree, r, s, |t: PPart| {
                let idx = tables.rank(t, degree);
                if let Some(bit) = output_bit(prod.out_offset as usize, idx, col_map, num_limbs) {
                    row[bit / 32] ^= 1 << (bit % 32);
                }
            });
        });
    }
    BatchOutput::from_limbs(limbs, num_limbs)
}

/// Evaluate `products` into `num_rows` rows of `num_cols` columns.
///
/// Each product is added (mod 2) into its row; see [`BatchProduct`]. With `col_map`, a term lands
/// in column `col_map[out_offset + idx]`, or nowhere if that is [`COL_MAP_DROP`] or past the end of
/// the map; `num_cols` is then the masked width. Terms landing at or past `num_cols` rounded up to
/// a multiple of 32 are dropped.
///
/// The algebra must satisfy [`batch_applicable`], every product's `row` must be below `num_rows`,
/// and the basis and the seqno tables of the algebra must be computed through every product's
/// degree.
///
/// With the `gpu` feature a large batch runs on the GPU when one is available, and the result is
/// the same bit for bit. A failure on the device is reported and the batch is computed on the CPU
/// instead.
pub fn multiply_batch(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    col_map: Option<&[u32]>,
    num_rows: usize,
    products: &[BatchProduct],
) -> BatchOutput {
    #[cfg(feature = "gpu")]
    if let Some(result) =
        super::cuda::multiply_batch(algebra, num_cols, col_map, num_rows, products)
    {
        match result {
            Ok(out) => return out,
            Err(e) => super::cuda::report_failure(&e),
        }
    }
    cpu_multiply_batch(algebra, num_cols, col_map, num_rows, products)
}

#[cfg(test)]
pub(super) mod tests {
    use fp::{prime::TWO, vector::FpVector};

    use super::*;

    /// A small deterministic generator (xorshift64*), so the tests need no `rand`.
    pub(crate) struct TestRng(u64);

    impl TestRng {
        pub(crate) fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
        }

        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// Uniform in `lo..=hi`.
        pub(crate) fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next() % (hi - lo + 1) as u64) as usize
        }

        /// True with probability `p`.
        pub(crate) fn chance(&mut self, p: f64) -> bool {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
        }
    }

    /// A random batch: products of random operations into random sums, `num_rows` rows of
    /// several blocks each, through degree `max_degree`. Returns the products and the width.
    pub(crate) fn random_batch(
        algebra: &MilnorAlgebra,
        rng: &mut TestRng,
        max_degree: i32,
        num_rows: usize,
        blocks_per_row: usize,
    ) -> (Vec<BatchProduct>, usize) {
        let mut products = Vec::new();
        let mut width = 0;
        for row in 0..num_rows {
            let out_degree = rng.range(max_degree as usize / 2, max_degree as usize) as i32;
            let block = algebra.dimension(out_degree);
            for b in 0..blocks_per_row {
                let r_degree = rng.range(1, out_degree as usize) as i32;
                let s_degree = out_degree - r_degree;
                let r_dim = algebra.dimension(r_degree);
                let s_dim = algebra.dimension(s_degree);
                if r_dim == 0 || s_dim == 0 {
                    continue;
                }
                let terms: Vec<usize> = (0..s_dim).filter(|_| rng.chance(0.5)).collect();
                if terms.is_empty() {
                    continue;
                }
                products.push(BatchProduct {
                    r_degree,
                    r_idx: rng.range(0, r_dim - 1) as u32,
                    s_degree,
                    term_indices: Arc::<[usize]>::from(terms).into(),
                    row: row as u32,
                    out_offset: (b * block) as u32,
                });
            }
            width = width.max(blocks_per_row * block);
        }
        (products, width)
    }

    pub(crate) fn algebra(max_degree: i32) -> MilnorAlgebra {
        let algebra = MilnorAlgebra::new(TWO, false);
        algebra.compute_basis(max_degree);
        algebra.compute_seqno_tables(max_degree);
        algebra
    }

    /// A product whose terms are the set bits of a range of a vector gives what the same terms
    /// listed give.
    #[test]
    fn bit_terms_match_listed_terms() {
        let max_degree = 60;
        let algebra = algebra(max_degree);
        let mut rng = TestRng::new(7);
        let (products, width) = random_batch(&algebra, &mut rng, max_degree, 16, 3);
        let as_bits: Vec<BatchProduct> = products
            .iter()
            .map(|p| {
                // The terms as the set bits of a vector with a few unrelated bits before them.
                let pad = 5;
                let dim = algebra.dimension(p.s_degree);
                let mut v = FpVector::new(fp::prime::TWO, pad + dim + 3);
                v.set_entry(1, 1);
                v.set_entry(pad + dim + 1, 1);
                p.term_indices.for_each(|i| v.set_entry(pad + i, 1));
                BatchProduct {
                    term_indices: Terms::bits(Arc::new(v), pad, pad + dim),
                    ..p.clone()
                }
            })
            .collect();
        assert_eq!(as_bits, products);
        let want = cpu_multiply_batch(&algebra, width, None, 16, &products);
        let got = cpu_multiply_batch(&algebra, width, None, 16, &as_bits);
        assert_eq!(got.iter_rows().collect::<Vec<_>>(), want.iter_rows().collect::<Vec<_>>());
    }

    /// The batch against one multiplication per term.
    #[test]
    fn cpu_batch_matches_multiply() {
        let max_degree = 80;
        let algebra = algebra(max_degree);
        let mut rng = TestRng::new(0);
        let (products, width) = random_batch(&algebra, &mut rng, max_degree, 24, 3);
        let out = cpu_multiply_batch(&algebra, width, None, 24, &products);

        let mut want = vec![FpVector::new(TWO, width); 24];
        for p in &products {
            let degree = p.r_degree + p.s_degree;
            let mut s = FpVector::new(TWO, algebra.dimension(p.s_degree));
            p.term_indices.for_each(|i| s.set_entry(i, 1));
            let mut prod = FpVector::new(TWO, algebra.dimension(degree));
            for (i, _) in s.iter_nonzero() {
                algebra.multiply_basis_elements(
                    prod.as_slice_mut(),
                    1,
                    p.r_degree,
                    p.r_idx as usize,
                    p.s_degree,
                    i,
                );
            }
            for (i, c) in prod.iter_nonzero() {
                want[p.row as usize].add_basis_element(p.out_offset as usize + i, c);
            }
        }
        for (row, (got, want)) in out.iter_rows().zip(&want).enumerate() {
            let got: Vec<usize> = (0..width)
                .filter(|&c| got[c / 32] >> (c % 32) & 1 == 1)
                .collect();
            let want: Vec<usize> = want.iter_nonzero().map(|(i, _)| i).collect();
            assert_eq!(got, want, "row {row}");
        }
    }

    /// A `col_map` gathers the columns of the full output.
    #[test]
    fn cpu_batch_col_map_gathers() {
        let max_degree = 60;
        let algebra = algebra(max_degree);
        let mut rng = TestRng::new(1);
        let (products, width) = random_batch(&algebra, &mut rng, max_degree, 16, 2);
        let full = cpu_multiply_batch(&algebra, width, None, 16, &products);

        let kept: Vec<usize> = (0..width).filter(|_| rng.chance(0.3)).collect();
        let mut map = vec![COL_MAP_DROP; width];
        for (j, &c) in kept.iter().enumerate() {
            map[c] = j as u32;
        }
        let masked = cpu_multiply_batch(&algebra, kept.len(), Some(&map), 16, &products);
        for (f, m) in full.iter_rows().zip(masked.iter_rows()) {
            for (j, &c) in kept.iter().enumerate() {
                assert_eq!(f[c / 32] >> (c % 32) & 1, m[j / 32] >> (j % 32) & 1);
            }
        }
    }
}
