//! The batched Milnor multiply, independent of any GPU framework.
//!
//! What a backend needs to *agree on* — the product descriptor, the output layout, and the CPU
//! reference — lives here rather than in any one backend's module.
//!
//! This exists for a specific reason. [`GpuProduct`], [`BatchOutput`] and [`cpu_multiply_batch`]
//! used to sit inside `milnor_gpu`, which is `#[cfg(feature = "gpu")]` and therefore behind cubecl.
//! Any second backend then had to either depend on the `gpu` feature — dragging cubecl along — or
//! duplicate the types, which would leave the CPU reference unable to check both. Neither is
//! acceptable while cubecl is being replaced, and neither survives its deletion.
//!
//! The CPU path here is the ULTIMATE REFERENCE. It owes nothing to cubecl, cudarc, a driver or a
//! card, so a digest computed from it pins the right answer permanently and outlives whatever
//! produced it. Measured: 14.90s for 10,000 products, which is ~25x cheaper than the "hours of
//! scalar work" the surrounding comments long assumed.

use std::sync::Arc;

use crate::algebra::{Algebra, MilnorAlgebra};

/// Sentinel in a `col_map`: this full-width column has no masked position.
pub const COL_MAP_DROP: u32 = u32::MAX;

/// One `Sq(R) · s` product, and where its result lands.
///
/// Separates WHAT is computed — `(r_degree, r_idx, s_degree, term_indices)` — from WHERE it goes —
/// `(row, out_offset)`. Two rows carrying the same operation over different same-degree generators
/// produce byte-identical work with different destinations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuProduct {
    pub r_degree: i32,
    pub r_idx: usize,
    pub s_degree: i32,
    /// `Arc<[usize]>`, not `Vec<usize>`, purely so cloning a `GpuProduct` is a refcount bump.
    ///
    /// The terms are written once at construction and only ever read afterwards, but products get
    /// cloned twice on the way to a device — once to compact rows into a dense range per hot/cold
    /// group, once to fan out into per-device buckets — and with a `Vec` each of those duplicated
    /// every term list. A call-graph profile of an uncapped stem-150 run put 5.45% of all user
    /// cycles in `_int_free` under the drop of these vectors alone (16.1% total in the allocator).
    /// Sharing makes the clones free and the drops O(1).
    pub term_indices: Arc<[usize]>,
    pub row: usize,
    pub out_offset: usize,
}

/// Storage for one block of output limbs.
///
/// A trait rather than a concrete type so a backend can hand over ITS OWN buffer without a copy —
/// cubecl's `Bytes` may already be pinned, and a cudarc backend has its own pinned host memory —
/// while this module stays ignorant of both. That ignorance is the point: it is what lets the CPU
/// reference and a new backend share an output type without either depending on the other.
pub trait LimbBlock: Send + Sync {
    /// The block's limbs, row-major.
    fn limbs(&self) -> &[u32];
}

impl LimbBlock for Vec<u32> {
    fn limbs(&self) -> &[u32] {
        self
    }
}

/// The result of a batched multiply: row-major `u32` limbs, in one or more blocks.
///
/// Blocks exist because a bounded launch lands one per row-block, in row order, and re-joining them
/// would mean copying gigabytes for nothing.
pub struct BatchOutput {
    blocks: Vec<Box<dyn LimbBlock>>,
    num_limbs: usize,
}

impl BatchOutput {
    /// Wrap backend-owned landing buffers (zero copy).
    pub fn from_blocks(blocks: Vec<Box<dyn LimbBlock>>, num_limbs: usize) -> Self {
        Self { blocks, num_limbs }
    }

    /// Wrap owned row-major limbs (eviction merge, CPU reference).
    pub fn from_limbs(limbs: Vec<u32>, num_limbs: usize) -> Self {
        Self {
            blocks: vec![Box::new(limbs)],
            num_limbs,
        }
    }

    /// Build from per-row limb vectors (test/reference helper).
    pub fn from_rows(rows: &[Vec<u32>], num_limbs: usize) -> Self {
        Self::from_limbs(rows.concat(), num_limbs)
    }

    /// Limbs per row.
    pub fn num_limbs(&self) -> usize {
        self.num_limbs
    }

    /// Number of rows across all blocks.
    pub fn rows(&self) -> usize {
        if self.num_limbs == 0 {
            return 0;
        }
        self.blocks.iter().map(|b| b.limbs().len()).sum::<usize>() / self.num_limbs
    }

    /// Row limb-slices in row order, as views into the landing buffers.
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

/// The reference implementation: compute a batch entirely on the CPU.
///
/// Scalar, obvious, and framework-free on purpose — this is what every backend is checked against.
pub fn cpu_multiply_batch(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    num_rows: usize,
    products: &[GpuProduct],
) -> BatchOutput {
    use fp::vector::FpVector;
    let p = algebra.prime();
    let num_limbs = num_cols.div_ceil(32).max(1);
    let mut rows = vec![vec![0u32; num_limbs]; num_rows];
    for prod in products {
        let out_degree = prod.r_degree + prod.s_degree;
        let block_dim = algebra.dimension(out_degree);
        if block_dim == 0 {
            continue;
        }
        let s_dim = algebra.dimension(prod.s_degree);
        let mut s = FpVector::new(p, s_dim);
        for &ti in prod.term_indices.iter() {
            s.set_entry(ti, 1);
        }
        let mut tmp = FpVector::new(p, block_dim);
        algebra.multiply_basis_element_by_element_2(
            tmp.as_slice_mut(),
            1,
            prod.r_degree,
            prod.r_idx,
            prod.s_degree,
            s.as_slice(),
        );
        for (i, _) in tmp.iter_nonzero() {
            let col = prod.out_offset + i;
            rows[prod.row][col / 32] ^= 1u32 << (col % 32);
        }
    }
    BatchOutput::from_limbs(rows.concat(), num_limbs)
}

/// [`cpu_multiply_batch`], honouring a `col_map` so it can stand in for a masked launch.
///
/// `cpu_multiply_batch` writes at FULL column indices (`out_offset + i`) while a masked launch
/// writes at restricted ones. The full width is recoverable — it is `col_map.len()` — so compute
/// wide and gather, which is the same "full-width launch plus host-side gather" the unmasked path
/// already performs.
///
/// The gather walks SET BITS, not columns: at frontier widths (`full` in the hundreds of thousands,
/// rows in the millions) a per-column scan would be quadratic enough to matter.
pub fn cpu_multiply_batch_masked(
    algebra: &MilnorAlgebra,
    out_cols: usize,
    col_map: Option<Arc<[u32]>>,
    num_rows: usize,
    products: &[GpuProduct],
) -> BatchOutput {
    let Some(map) = col_map else {
        return cpu_multiply_batch(algebra, out_cols, num_rows, products);
    };
    let full = map.len();
    let wide = cpu_multiply_batch(algebra, full, num_rows, products);
    let out_limbs = out_cols.div_ceil(32).max(1);
    let mut rows = vec![vec![0u32; out_limbs]; num_rows];
    for (r, row) in wide.iter_rows().enumerate() {
        for (li, &limb) in row.iter().enumerate() {
            let mut bits = limb;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let col = li * 32 + b;
                if col < full {
                    let to = map[col];
                    if to != COL_MAP_DROP {
                        let to = to as usize;
                        rows[r][to / 32] ^= 1u32 << (to % 32);
                    }
                }
            }
        }
    }
    BatchOutput::from_limbs(rows.concat(), out_limbs)
}
