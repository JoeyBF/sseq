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

use std::sync::{
    Arc, LazyLock,
    atomic::{AtomicU64, Ordering},
};

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

/// Capture/replay (`NASSAU_CAPTURE_PRODUCTS`): dump one REAL batch to disk so a bench can replay it.
///
/// A synthetic bench cannot stand in for the frontier here. The in-tree one samples `R`s on a stride
/// and gives every product the same term count, and measured against real frontier work it ranks tile
/// 4x2 at 0.706x where the truth is 1.16x -- the WRONG SIGN, with a 0.4% noise floor, so its
/// confidence is the dangerous part. Scaling its dimensions does not obviously fix that, because the
/// thing it flattens is the DISTRIBUTION: real work has a steeply skewed spread over both `num_mats`
/// per `R` (top 1% of `R`s carry 31% of references) and terms per product, and the tile's ragged-tail
/// cost is a function of exactly that spread.
///
/// So capture the real thing instead of approximating it. The batch is self-contained -- products name
/// algebra basis elements by `(degree, index)`, so a replay only needs the basis computed to the same
/// degree -- and replaying it costs nothing but the multiply itself: no resolution, no save, no
/// signature walk.
///
/// `NASSAU_CAPTURE_NTH` (default 200) picks WHICH call to keep. The first calls of a run are small
/// warm-up batches from low bidegrees; the interesting one is a steady-state frontier batch.
static CAPTURE_PATH: LazyLock<Option<String>> =
    LazyLock::new(|| std::env::var("NASSAU_CAPTURE_PRODUCTS").ok());
static CAPTURE_NTH: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("NASSAU_CAPTURE_NTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200)
});
/// How many CONSECUTIVE batches to keep, starting at [`CAPTURE_NTH`]. With more than one, the path is
/// treated as a DIRECTORY and files are written as `batch_00000.bin`, ...
///
/// One batch is the wrong unit. Replaying a single batch in a loop gets both halves wrong: the batch
/// is a sample from a wide distribution (the one first captured carried 48.9 terms per product against
/// a 165 population mean, and the tile comparison is a direct function of that number), and looping it
/// holds the resident master and shift cache at an artificially warm steady state that a real run
/// never sees. A consecutive RUN of batches carries the true mix of shapes and the natural growth.
static CAPTURE_COUNT: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("NASSAU_CAPTURE_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
});
/// Stop capturing once this many bytes have been written, so a long capture cannot fill the disk.
static CAPTURE_MAX_GB: LazyLock<f64> = LazyLock::new(|| {
    std::env::var("NASSAU_CAPTURE_MAX_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40.0)
});
static CAPTURE_SEEN: AtomicU64 = AtomicU64::new(0);
static CAPTURE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Report capture-path entry counts on stderr. Opt-in, because the case it exists for is a
/// capture that silently produces nothing, where every other signal is absent by construction.
static CAPTURE_PROBE: LazyLock<bool> =
    LazyLock::new(|| std::env::var("NASSAU_CAPTURE_PROBE").is_ok_and(|v| v != "0"));
static CAPTURE_WRITTEN: AtomicU64 = AtomicU64::new(0);
static CAPTURE_BYTES: AtomicU64 = AtomicU64::new(0);

/// `NASPROD1` + u64 fields, little-endian throughout. Deliberately a flat dump rather than a serde
/// format: the term lists dominate the file (a frontier batch is ~540k products x ~165 terms), so the
/// layout that matters is that they are one contiguous run of u32.
const CAPTURE_MAGIC: &[u8; 8] = b"NASPROD1";

pub fn capture_batch(
    out_cols: usize,
    col_map: Option<&[u32]>,
    num_rows: usize,
    products: &[GpuProduct],
) {
    // Count before the path guard, so "never called" and "called but unconfigured" stay
    // distinguishable. An earlier probe sat after this early return and so printed nothing in
    // either case, which is what let five wrong explanations survive as long as they did.
    //
    // Note for anyone capturing at the frontier: this function is entered roughly ONCE EVERY FEW
    // MINUTES there, because a single entry fans out into thousands of kernel launches. The
    // default `NASSAU_CAPTURE_NTH` is calibrated on small stems, where entries are a steady
    // stream, and is simply unreachable in a frontier run of any practical length. Use 1..3.
    let calls = CAPTURE_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if *CAPTURE_PROBE && (calls == 1 || calls % 100 == 0) {
        let set = CAPTURE_PATH.is_some();
        eprintln!("[capture] entered={calls} path_set={set} rows={num_rows} cols={out_cols}");
    }
    let Some(path) = CAPTURE_PATH.as_deref() else {
        return;
    };
    // `fetch_add` returns a unique ticket per caller, so exactly one call sees the target. Comparing a
    // separate `load` against `== n` can fire never under ~100 concurrent callers.
    let n = CAPTURE_SEEN.fetch_add(1, Ordering::Relaxed) + 1;
    if n % 100 == 0 || n == *CAPTURE_NTH {
        let masked = col_map.is_some();
        let np = products.len();
        let nth = *CAPTURE_NTH;
        eprintln!("[capture] seen={n}/{nth} rows={num_rows} cols={out_cols} mask={masked} p={np}");
    }
    if n < *CAPTURE_NTH || n >= *CAPTURE_NTH + *CAPTURE_COUNT {
        return;
    }
    if (CAPTURE_BYTES.load(Ordering::Relaxed) as f64) > *CAPTURE_MAX_GB * 1e9 {
        return;
    }
    let seq = CAPTURE_WRITTEN.fetch_add(1, Ordering::Relaxed);
    // One batch keeps the old single-file behaviour; a range writes into a directory.
    let target = if *CAPTURE_COUNT > 1 {
        let _ = std::fs::create_dir_all(path);
        format!("{path}/batch_{seq:05}.bin")
    } else {
        path.to_owned()
    };
    let path = target.as_str();
    use std::io::Write as _;
    let write = || -> std::io::Result<()> {
        let f = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
        let u = |v: usize| (v as u64).to_le_bytes();
        w.write_all(CAPTURE_MAGIC)?;
        w.write_all(&u(num_rows))?;
        w.write_all(&u(out_cols))?;
        match col_map {
            Some(m) => {
                w.write_all(&u(1))?;
                w.write_all(&u(m.len()))?;
                for &c in m {
                    w.write_all(&c.to_le_bytes())?;
                }
            }
            None => {
                w.write_all(&u(0))?;
                w.write_all(&u(0))?;
            }
        }
        w.write_all(&u(products.len()))?;
        for p in products {
            w.write_all(&(p.r_degree as i64).to_le_bytes())?;
            w.write_all(&(p.s_degree as i64).to_le_bytes())?;
            w.write_all(&u(p.r_idx))?;
            w.write_all(&u(p.row))?;
            w.write_all(&u(p.out_offset))?;
            w.write_all(&u(p.term_indices.len()))?;
            for &t in p.term_indices.iter() {
                w.write_all(&(t as u32).to_le_bytes())?;
            }
        }
        w.flush()
    };
    let terms: usize = products.iter().map(|p| p.term_indices.len()).sum();
    let maxr = products.iter().map(|p| p.r_degree).max().unwrap_or(0);
    let maxs = products.iter().map(|p| p.s_degree).max().unwrap_or(0);
    match write() {
        Ok(()) => eprintln!(
            "[capture] call #{n} -> {path}: products={} terms={terms} (mean {:.1}) \
             rows={num_rows} out_cols={out_cols} col_map={} max_r_degree={maxr} \
             max_s_degree={maxs}",
            products.len(),
            terms as f64 / products.len().max(1) as f64,
            col_map.map_or(0, <[u32]>::len),
        ),
        Err(e) => eprintln!("[capture] FAILED to write {path}: {e}"),
    }
    if let Ok(md) = std::fs::metadata(path) {
        CAPTURE_BYTES.fetch_add(md.len(), Ordering::Relaxed);
    }
}

/// Read back what [`capture_batch`] wrote. Returns `(num_rows, out_cols, col_map, products)`.
pub fn load_captured_batch(
    path: &str,
) -> std::io::Result<(usize, usize, Option<Arc<[u32]>>, Vec<GpuProduct>)> {
    use std::io::Read as _;
    let f = std::fs::File::open(path)?;
    let mut r = std::io::BufReader::with_capacity(1 << 22, f);
    let mut m = [0u8; 8];
    r.read_exact(&mut m)?;
    if &m != CAPTURE_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a captured product batch",
        ));
    }
    let mut u64b = [0u8; 8];
    let mut rd = |r: &mut std::io::BufReader<std::fs::File>| -> std::io::Result<u64> {
        r.read_exact(&mut u64b)?;
        Ok(u64::from_le_bytes(u64b))
    };
    let num_rows = rd(&mut r)? as usize;
    let out_cols = rd(&mut r)? as usize;
    let has_map = rd(&mut r)? == 1;
    let map_len = rd(&mut r)? as usize;
    let col_map = if has_map {
        let mut v = vec![0u32; map_len];
        let mut buf = [0u8; 4];
        for slot in v.iter_mut() {
            r.read_exact(&mut buf)?;
            *slot = u32::from_le_bytes(buf);
        }
        Some(Arc::from(v))
    } else {
        None
    };
    let np = rd(&mut r)? as usize;
    let mut products = Vec::with_capacity(np);
    let mut buf4 = [0u8; 4];
    for _ in 0..np {
        let r_degree = rd(&mut r)? as i64 as i32;
        let s_degree = rd(&mut r)? as i64 as i32;
        let r_idx = rd(&mut r)? as usize;
        let row = rd(&mut r)? as usize;
        let out_offset = rd(&mut r)? as usize;
        let nt = rd(&mut r)? as usize;
        let mut terms = Vec::with_capacity(nt);
        for _ in 0..nt {
            r.read_exact(&mut buf4)?;
            terms.push(u32::from_le_bytes(buf4) as usize);
        }
        products.push(GpuProduct {
            r_degree,
            s_degree,
            r_idx,
            term_indices: Arc::from(terms),
            row,
            out_offset,
        });
    }
    Ok((num_rows, out_cols, col_map, products))
}
