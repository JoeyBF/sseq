//! Verifying that a matrix IS the reduced row echelon form of another, without reducing again.
//!
//! The reduced row echelon form of a matrix (fixed column order) is unique, and so is that of
//! `[A | 0 | I]`. So a claimed reduction is right iff it IS that RREF, and most of that can be
//! checked cheaply, whichever backend produced it (M4RI, a GPU kernel, a remote process), with
//! randomized Freivalds-style products over F2: `K` random vectors ride in the bits of one `u64`
//! per column ("lanes"), so one pass over a matrix multiplies it by all of them.
//!
//! What is checked (`p` = pivot columns, `R` the claimed rows):
//!
//! * Plain ([`verify_rref_rows`]: input `M`, `R` = the nonzero rows of RREF(M)):
//!   - form, exactly: row `k` has its first set bit at `p[k]`, nothing past the last column;
//!   - pivot columns are unit columns: `R x̂ = x` for random `x` placed at `p` (Freivalds);
//!   - rowspace(M) ⊆ rowspace(R): a random `z` in ker(R) (free coordinates random, pivot ones
//!     `R z'`) must satisfy `M z = 0`.
//!   - NOT checkable without the transform: rowspace(R) ⊆ rowspace(M), i.e. an `R` with an
//!     EXTRA independent row in RREF form passes (rank inflation). [`Matrix::verify_rref_of`]
//!     closes it when given the transform `T` (`R = T M`).
//! * Identity ([`verify_identity`]: input `A` (n x c); claimed: all n pivots of `[A | 0 | I]` (`p`
//!   in `A`, `J` in the identity block), `rank_a`, and the identity parts `Q` of the pivot rows
//!   and/or `Z` of the zero rows). Every row `[T A | 0 | T]` is in the row space by construction,
//!   so the claim is the RREF iff it has RREF form:
//!   - `Z` (if given): exact form in identity coordinates (first bit at `J[k]`), `Z[:, J] = I`, and
//!     `Z A = 0`. With `rank_a` independent rows from `Q` or the count `n - rank_a`, this pins Z
//!     to the RREF of the left kernel: complete.
//!   - `Q` (if given): `Q[:, J] = 0`, `(Q A)[:, p] = I`, and (when `Z` is not given)
//!     rank(A) <= rank_a through a random `z` in ker(QA) that must satisfy `A z = 0`.
//!   - NOT checked: that `p` is the FIRST pivot set (the leading zeros of `Q A` before each
//!     pivot), since `Q A` is not given. A claim passing everything else is a reduced basis of
//!     rowspace(A) with identity at `p`, so solves through it are still correct; only the
//!     canonical choice of `p` is unverified.
//!
//! False accept: each randomized check passes a wrong claim with probability at most 2^-K (the
//! failing event has probability >= 1/2 per independent random vector).

use std::io;

use super::Matrix;

/// Why a claimed reduction is not the RREF of its input: which check failed, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RrefVerifyError {
    /// The check that failed: `shape`, `form`, `unit_columns`, `row_space`, `transform`,
    /// `kernel_form`, `kernel_unit_columns`, `kernel_annihilates`, `pivot_rows_at_kernel_pivots`,
    /// `pivot_identity`, `rank`, or `io`.
    pub check: &'static str,
    pub detail: String,
}

impl std::fmt::Display for RrefVerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.check, self.detail)
    }
}

impl std::error::Error for RrefVerifyError {}

fn fail(check: &'static str, detail: impl Into<String>) -> RrefVerifyError {
    RrefVerifyError {
        check,
        detail: detail.into(),
    }
}

fn io_fail(e: io::Error) -> RrefVerifyError {
    fail("io", e.to_string())
}

/// Rows of a matrix over F2, row `i` as limbs (bit `j` of limb `l` = column `64 l + j`; bits past
/// `cols` are ignored).
pub trait Rows: Sync {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    /// Call `f(first_row, block)` on consecutive blocks of rows, each row `cols.div_ceil(64)`
    /// limbs with row stride `stride` in `block`.
    fn for_each_block(
        &self,
        f: &mut dyn FnMut(usize, &[u64], usize) -> io::Result<()>,
    ) -> io::Result<()>;
}

/// Rows held in memory: `rows` rows of stride `stride` limbs.
pub struct MemRows<'a> {
    pub data: &'a [u64],
    pub stride: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Rows for MemRows<'_> {
    fn nrows(&self) -> usize {
        self.rows
    }

    fn ncols(&self) -> usize {
        self.cols
    }

    fn for_each_block(
        &self,
        f: &mut dyn FnMut(usize, &[u64], usize) -> io::Result<()>,
    ) -> io::Result<()> {
        if self.rows == 0 {
            return Ok(());
        }
        let used = (self.rows - 1) * self.stride + self.cols.div_ceil(64);
        f(0, &self.data[..used], self.stride)
    }
}

fn mask(k: u32) -> u64 {
    if k >= 64 { u64::MAX } else { (1u64 << k) - 1 }
}

/// Random words (splitmix64).
pub struct Rng(u64);

impl Rng {
    /// From a fixed seed: reproducible (tests).
    pub fn seeded(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// A seed nobody else knows (whoever produced the claim must not be able to fit it to the
    /// vectors).
    pub fn fresh_seed() -> u64 {
        Self::from_entropy().next()
    }

    fn from_entropy() -> Self {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        Self(h.finish() | 1)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Columns per table tile: 64 limbs, so the tables (8 per limb, 256 entries) are 1 MB.
const TILE_LIMBS: usize = 64;

/// `out[i] ^= (row i) . x` for every row of `block` (`n` rows of stride `stride`), all lanes at
/// once, for the limbs `l0..l1` whose tables are `tab` (`(l1 - l0) * 8 * 256` entries).
fn apply_tile(
    block: &[u64],
    stride: usize,
    n: usize,
    l0: usize,
    l1: usize,
    tab: &[u64],
    out: &mut [u64],
) {
    for i in 0..n {
        let row = &block[i * stride + l0..i * stride + l1];
        let mut acc = 0u64;
        for (j, &w) in row.iter().enumerate() {
            if w == 0 {
                continue;
            }
            let t = &tab[j * 8 * 256..(j + 1) * 8 * 256];
            let b = w.to_le_bytes();
            acc ^= t[b[0] as usize]
                ^ t[256 + b[1] as usize]
                ^ t[512 + b[2] as usize]
                ^ t[768 + b[3] as usize]
                ^ t[1024 + b[4] as usize]
                ^ t[1280 + b[5] as usize]
                ^ t[1536 + b[6] as usize]
                ^ t[1792 + b[7] as usize];
        }
        out[i] ^= acc;
    }
}

/// Byte tables of `x` (one lane word per column) for limbs `l0..l1`: entry `(j, b, v)` is the XOR
/// of `x[64 (l0 + j) + 8 b + t]` over the set bits `t` of `v`. Columns past `cols` count as zero.
fn tables(x: &[u64], cols: usize, l0: usize, l1: usize, tab: &mut Vec<u64>) {
    tab.clear();
    tab.resize((l1 - l0) * 8 * 256, 0);
    for j in 0..l1 - l0 {
        for b in 0..8 {
            let base = 64 * (l0 + j) + 8 * b;
            let t = &mut tab[(j * 8 + b) * 256..(j * 8 + b + 1) * 256];
            for v in 1..256usize {
                let low = v.trailing_zeros() as usize;
                let c = base + low;
                let xc = if c < cols { x[c] } else { 0 };
                t[v] = t[v & (v - 1)] ^ xc;
            }
        }
    }
}

/// `M x`: for every row `i` of `m`, the XOR of `x[c]` over the set bits `c` of row `i` -- i.e. the
/// product with up to 64 vectors at once. `x` has one word per column of `m`. Multithreaded over
/// rows; tile-by-tile over columns so the tables stay in cache.
pub fn mul(m: &dyn Rows, x: &[u64]) -> io::Result<Vec<u64>> {
    let (rows, cols) = (m.nrows(), m.ncols());
    let limbs = cols.div_ceil(64);
    let mut out = vec![0u64; rows];
    let nt = threads();
    let mut tab = Vec::new();
    m.for_each_block(&mut |first, block, stride| {
        let n = (block.len() + stride - limbs)
            .checked_div(stride)
            .unwrap_or(0);
        let n = n.min(rows - first);
        let dst = &mut out[first..first + n];
        let mut l0 = 0;
        while l0 < limbs {
            let l1 = (l0 + TILE_LIMBS).min(limbs);
            tables(x, cols, l0, l1, &mut tab);
            let tab = &tab[..];
            let per = n.div_ceil(nt).max(1024);
            std::thread::scope(|sc| {
                for (ci, chunk) in dst.chunks_mut(per).enumerate() {
                    let r0 = ci * per;
                    let len = chunk.len();
                    let blk = &block[r0 * stride..];
                    sc.spawn(move || apply_tile(blk, stride, len, l0, l1, tab, chunk));
                }
            });
            l0 = l1;
        }
        Ok(())
    })?;
    Ok(out)
}

/// Exact RREF form of `r` (rows of stride `stride`, `cols` columns) against `pivots`: row `k` has
/// its first set bit at `pivots[k]` and no bits past `cols`. One scan.
pub fn check_form(r: &[u64], stride: usize, cols: usize, pivots: &[usize]) -> Result<(), String> {
    let limbs = cols.div_ceil(64);
    let tail = if cols.is_multiple_of(64) {
        u64::MAX
    } else {
        (1u64 << (cols % 64)) - 1
    };
    for (k, &p) in pivots.iter().enumerate() {
        let row = &r[k * stride..k * stride + limbs];
        let (pl, pb) = (p / 64, p % 64);
        if row[..pl].iter().any(|&w| w != 0) || row[pl] & ((1u64 << pb) - 1) != 0 {
            return Err(format!("row {k} is nonzero before its pivot column {p}"));
        }
        if row[pl] >> pb & 1 == 0 {
            return Err(format!("row {k} has no 1 at its pivot column {p}"));
        }
        if let Some(&last) = row.last()
            && last & !tail != 0
        {
            return Err(format!("row {k} has bits past column {cols}"));
        }
    }
    Ok(())
}

/// Lane words at the given columns: `x̂[cols[k]] = x[k]`, zero elsewhere (length `n`).
fn scatter(n: usize, at: &[usize], x: &[u64]) -> Vec<u64> {
    let mut v = vec![0u64; n];
    for (k, &c) in at.iter().enumerate() {
        v[c] = x[k];
    }
    v
}

fn random(n: usize, rng: &mut Rng, m: u64) -> Vec<u64> {
    (0..n).map(|_| rng.next() & m).collect()
}

fn first_diff(a: &[u64], b: &[u64], m: u64) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| (x ^ y) & m != 0)
}

/// A plain reduction: `r` (`pivots.len()` rows, tight stride `cols.div_ceil(64)`) claimed to be the
/// nonzero rows of RREF(`m`), with `pivots` its pivot columns. Checks the exact RREF form, that
/// the pivot columns are unit columns (`k` vectors), and rowspace(m) ⊆ rowspace(r) (`k` random
/// kernel vectors of `r`). Not checked (needs a transform, see [`Matrix::verify_rref_of`]):
/// rowspace(r) ⊆ rowspace(m).
pub fn verify_rref_rows(
    m: &dyn Rows,
    r: &[u64],
    pivots: &[usize],
    seed: u64,
    k: u32,
) -> Result<(), RrefVerifyError> {
    let cols = m.ncols();
    let stride = cols.div_ceil(64);
    let rank = pivots.len();
    if r.len() < rank * stride
        || pivots.windows(2).any(|w| w[0] >= w[1])
        || pivots.last().is_some_and(|&p| p >= cols)
    {
        return Err(fail(
            "shape",
            format!("{rank} pivots, {} limbs for {cols} columns", r.len()),
        ));
    }
    check_form(r, stride, cols, pivots).map_err(|e| fail("form", e))?;
    let rr = MemRows {
        data: r,
        stride,
        rows: rank,
        cols,
    };
    let mk = mask(k);
    let hi = mk << 32;
    let mut rng = Rng::seeded(seed);
    // Lanes 0..k: pivot columns are unit columns. Lanes 32..32+k: a kernel vector's free part.
    let x = random(rank, &mut rng, mk);
    let mut is_pivot = vec![false; cols];
    for &p in pivots {
        is_pivot[p] = true;
    }
    let mut v = scatter(cols, pivots, &x);
    let mut zfree = vec![0u64; cols];
    for c in 0..cols {
        let w = rng.next() & hi;
        if !is_pivot[c] {
            zfree[c] = w;
            v[c] ^= w;
        }
    }
    let y = mul(&rr, &v).map_err(io_fail)?;
    if let Some(row) = first_diff(&y, &x, mk) {
        return Err(fail(
            "unit_columns",
            format!("a pivot column is not a unit column (row {row}): not fully reduced"),
        ));
    }
    // z = z' + scatter(p, R z') is in ker(R); M z = 0 for all such z iff rowspace(M) ⊆ rowspace(R).
    let mut z = zfree;
    for (row, &p) in pivots.iter().enumerate() {
        z[p] = y[row] & hi;
    }
    let mz = mul(m, &z).map_err(io_fail)?;
    if let Some(row) = mz.iter().position(|&w| w & hi != 0) {
        return Err(fail(
            "row_space",
            format!(
                "input row {row} is not in the row space of the claimed RREF (dropped or wrong \
                 rows)"
            ),
        ));
    }
    Ok(())
}

/// An identity (`[A | 0 | I]`) reduction, given by its parts, against `a`
/// (`n x c`): all `n` pivot columns of the augmented matrix (identity block from
/// `c.div_ceil(64) * 64`), `rank_a` of them in `A`; `q` the identity parts of the pivot rows
/// (`rank_a x n`, `(data, stride)`) and `z` of the zero rows (`(n - rank_a) x n`), each if present.
/// See the module documentation for what is and is not checked.
pub fn verify_identity(
    a: &dyn Rows,
    pivots: &[usize],
    rank_a: usize,
    q: Option<(&[u64], usize)>,
    z: Option<(&[u64], usize)>,
    seed: u64,
    k: u32,
) -> Result<(), RrefVerifyError> {
    let (n, c) = (a.nrows(), a.ncols());
    let first_id = c.div_ceil(64) * 64;
    if pivots.len() != n || rank_a > n || pivots.windows(2).any(|w| w[0] >= w[1]) {
        return Err(fail(
            "shape",
            format!("{} pivots, rank {rank_a} for {n} rows", pivots.len()),
        ));
    }
    let p = &pivots[..rank_a];
    if p.last().is_some_and(|&x| x >= c) || pivots.get(rank_a).is_some_and(|&x| x < c) {
        return Err(fail("shape", "the pivots of A disagree with its rank"));
    }
    let mut jcols = Vec::with_capacity(n - rank_a);
    for &x in &pivots[rank_a..] {
        if x < first_id || x >= first_id + n {
            return Err(fail(
                "shape",
                format!("pivot column {x} neither in A nor in the identity block"),
            ));
        }
        jcols.push(x - first_id);
    }
    let mk = mask(k);
    let mut rng = Rng::seeded(seed);
    let mut is_pivot = vec![false; c];
    for &x in p {
        is_pivot[x] = true;
    }
    // Three lane groups of k lanes share every pass:
    //   G1 (lanes 0..)  : x̂_p             -> (Q A)[:, p] = I
    //   G2 (lanes 21..) : x over A's columns (Z A = 0), or without Z the free part z' of a
    //                     kernel vector of Q A (rank(A) <= rank_a)
    //   G3 (lanes 42..) : x̂_J             -> Z[:, J] = I, Q[:, J] = 0
    let (g1, g2, g3) = (mk, mk << 21, mk << 42);
    let xp = random(rank_a, &mut rng, mk);
    let mut v = scatter(c, p, &xp);
    let mut zprime = vec![0u64; c];
    for col in 0..c {
        let r = rng.next() & g2;
        if z.is_some() || !is_pivot[col] {
            zprime[col] = r;
            v[col] ^= r;
        }
    }
    // One pass over A.
    let av = mul(a, &v).map_err(io_fail)?;
    let xj: Vec<u64> = random(n - rank_a, &mut rng, mk)
        .into_iter()
        .map(|w| w << 42)
        .collect();
    let vj = scatter(n, &jcols, &xj);
    if let Some((zd, zst)) = z {
        check_form(zd, zst, n, &jcols).map_err(|e| fail("kernel_form", e))?;
        let zr = MemRows {
            data: zd,
            stride: zst,
            rows: n - rank_a,
            cols: n,
        };
        // One pass over Z: G2 = Z (A x) must vanish, G3 = Z x̂_J must be x.
        let input: Vec<u64> = av.iter().zip(&vj).map(|(&a, &j)| (a & g2) | j).collect();
        let y = mul(&zr, &input).map_err(io_fail)?;
        if let Some(row) = first_diff(&y, &xj, g3) {
            return Err(fail(
                "kernel_unit_columns",
                format!("kernel row {row}: a pivot column is not a unit column"),
            ));
        }
        if let Some(row) = y.iter().position(|&w| w & g2 != 0) {
            return Err(fail(
                "kernel_annihilates",
                format!("kernel row {row} does not annihilate A"),
            ));
        }
    }
    if let Some((qd, qst)) = q {
        let qr = MemRows {
            data: qd,
            stride: qst,
            rows: rank_a,
            cols: n,
        };
        // One pass over Q: G1 = (Q A) x̂_p must be x, G2 = Q A z', G3 = Q x̂_J must vanish.
        let input: Vec<u64> = av
            .iter()
            .zip(&vj)
            .map(|(&a, &j)| (a & (g1 | g2)) | j)
            .collect();
        let y = mul(&qr, &input).map_err(io_fail)?;
        if let Some(row) = y.iter().position(|&w| w & g3 != 0) {
            return Err(fail(
                "pivot_rows_at_kernel_pivots",
                format!("pivot row {row} is not zero at the kernel's pivot columns"),
            ));
        }
        if let Some(row) = first_diff(&y, &xp, g1) {
            return Err(fail(
                "pivot_identity",
                format!("(Q A) is not the identity at the pivots (row {row})"),
            ));
        }
        if z.is_none() {
            // z = z' + scatter(p, Q A z') lies in ker(QA); A z = 0 iff rank(A) <= rank_a.
            let mut zv = zprime;
            for (row, &x) in p.iter().enumerate() {
                zv[x] = y[row] & g2;
            }
            let az = mul(a, &zv).map_err(io_fail)?;
            if let Some(row) = az.iter().position(|&w| w & g2 != 0) {
                return Err(fail(
                    "rank",
                    format!(
                        "rank of A exceeds {rank_a}: row {row} of A is outside the pivot rows' \
                         span"
                    ),
                ));
            }
        }
    }
    Ok(())
}

impl Matrix {
    /// Whether `self` IS the reduced row echelon form of `original` (p = 2), as `row_reduce`
    /// leaves it: same shape, the nonzero rows first, each leading at a column after the
    /// previous one's, the zero rows last. Without reducing again: the exact form is scanned, and
    /// the rest is tested with `k` random vectors (at most 32), from `seed`, for each of
    /// - the pivot columns are unit columns (`self x̂ = x`),
    /// - rowspace(original) ⊆ rowspace(self) (random kernel vectors `z` of `self`: `original z = 0`),
    /// - with `transform` (`T`, rows x rows): `self = T original` (`self x = T (original x)`),
    ///   which gives rowspace(self) ⊆ rowspace(original) and so makes the check complete.
    ///
    /// A wrong `self` passes with probability at most 2^-k. Without `transform`, an RREF of a
    /// strictly LARGER row space (an extra independent row) is not detected: see
    /// [`verify_rref_rows`].
    pub fn verify_rref_of(
        &self,
        original: &Self,
        transform: Option<&Self>,
        seed: u64,
        k: u32,
    ) -> Result<(), RrefVerifyError> {
        let (rows, cols) = (original.rows(), original.columns());
        if self.rows() != rows || self.columns() != cols {
            return Err(fail(
                "shape",
                format!(
                    "{}x{} for a {rows}x{cols} input",
                    self.rows(),
                    self.columns()
                ),
            ));
        }
        if self.prime() != crate::prime::TWO || original.prime() != crate::prime::TWO {
            return Err(fail("shape", "only p = 2"));
        }
        let limbs = cols.div_ceil(64);
        let ss = self.stride();
        // Pivots: each nonzero row's first set bit; zero rows only at the end.
        let mut pivots = Vec::new();
        let mut zero_from = None;
        for i in 0..rows {
            let row = &self.data()[i * ss..i * ss + limbs];
            match row.iter().position(|&w| w != 0) {
                Some(l) => {
                    if let Some(z) = zero_from {
                        return Err(fail("form", format!("nonzero row {i} after zero row {z}")));
                    }
                    pivots.push(64 * l + row[l].trailing_zeros() as usize);
                }
                None => {
                    zero_from.get_or_insert(i);
                }
            }
        }
        if pivots.windows(2).any(|w| w[0] >= w[1]) {
            return Err(fail("form", "pivot columns not strictly increasing"));
        }
        // A pivots array, where kept, must agree.
        if self.pivots().len() == cols {
            for (c, &pv) in self.pivots().iter().enumerate() {
                let want = pivots
                    .iter()
                    .position(|&p| p == c)
                    .map_or(-1, |k| k as isize);
                if pv != want && !(pv < 0 && want < 0) {
                    return Err(fail(
                        "form",
                        format!("pivots array says {pv} for column {c}, the rows {want}"),
                    ));
                }
            }
        }
        let tight: Vec<u64> = (0..pivots.len())
            .flat_map(|i| self.data()[i * ss..i * ss + limbs].iter().copied())
            .collect();
        let m = MemRows {
            data: original.data(),
            stride: original.stride(),
            rows,
            cols,
        };
        verify_rref_rows(&m, &tight, &pivots, seed, k.min(32))?;
        if let Some(t) = transform {
            if t.rows() != rows || t.columns() != rows {
                return Err(fail(
                    "shape",
                    format!("transform {}x{} for {rows} rows", t.rows(), t.columns()),
                ));
            }
            let mut rng = Rng::seeded(seed ^ 0x5bd1_e995);
            let mk = mask(k.min(64));
            let x = random(cols, &mut rng, mk);
            let me = MemRows {
                data: self.data(),
                stride: ss,
                rows,
                cols,
            };
            let lhs = mul(&me, &x).map_err(io_fail)?;
            let ox = mul(&m, &x).map_err(io_fail)?;
            let tr = MemRows {
                data: t.data(),
                stride: t.stride(),
                rows,
                cols: rows,
            };
            let rhs = mul(&tr, &ox).map_err(io_fail)?;
            if let Some(row) = first_diff(&lhs, &rhs, mk) {
                return Err(fail(
                    "transform",
                    format!("row {row} is not the transform's combination of the input"),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use proptest::{prelude::*, test_runner::RngSeed};

    use super::*;
    use crate::{
        matrix::{AugmentedMatrix, Matrix, arbitrary::MatrixArbParams},
        prime::TWO,
    };

    const K: u32 = 20;

    fn arb_matrix(max_dim: usize) -> impl Strategy<Value = Matrix> {
        (1usize..=max_dim, 1usize..=max_dim).prop_flat_map(|(rows, cols)| {
            Matrix::arbitrary_with(MatrixArbParams {
                p: Some(TWO),
                rows: Just(rows).boxed(),
                columns: Just(cols).boxed(),
            })
        })
    }

    fn arb_low_rank() -> impl Strategy<Value = Matrix> {
        (20usize..200, 20usize..200, 1usize..20).prop_flat_map(|(rows, cols, rank_bound)| {
            let a = Matrix::arbitrary_with(MatrixArbParams {
                p: Some(TWO),
                rows: Just(rows).boxed(),
                columns: Just(rank_bound).boxed(),
            });
            let b = Matrix::arbitrary_with(MatrixArbParams {
                p: Some(TWO),
                rows: Just(rank_bound).boxed(),
                columns: Just(cols).boxed(),
            });
            (a, b).prop_map(|(a, b)| &a * &b)
        })
    }

    /// `[m | 0 | I]`, reduced.
    fn reduced_augmented(m: &Matrix) -> AugmentedMatrix<2> {
        let (rows, cols) = (m.rows(), m.columns());
        let mut aug = AugmentedMatrix::<2>::new(TWO, rows, [cols, rows]);
        for i in 0..rows {
            aug.row_segment_mut(i, 0, 0).assign(m.row(i));
        }
        aug.segment(1, 1).add_identity();
        aug.row_reduce();
        aug
    }

    /// The `A` part (as a `rows x cols` matrix) and the identity part (`rows x rows`) of a reduced
    /// augmented matrix, as packed matrices.
    fn split(aug: &AugmentedMatrix<2>, rows: usize, cols: usize) -> (Matrix, Matrix) {
        let (sc, sn) = (cols.div_ceil(64), rows.div_ceil(64));
        let first = sc;
        let st = aug.stride();
        let mut r = Vec::with_capacity(rows * sc);
        let mut t = Vec::with_capacity(rows * sn);
        for i in 0..rows {
            r.extend_from_slice(&aug.data()[i * st..i * st + sc]);
            t.extend_from_slice(&aug.data()[i * st + first..i * st + first + sn]);
        }
        let mut r = Matrix::from_data(TWO, rows, cols, r);
        let mut t = Matrix::from_data(TWO, rows, rows, t);
        // Bits past `cols` (or `rows`) are zero in a well-formed augmented matrix; keep them so.
        let _ = (&mut r, &mut t);
        (r, t)
    }

    /// An identity reduction's parts, as [`verify_identity`] takes them: all pivots, rank of A, Q, Z.
    fn identity_parts(m: &Matrix) -> (Vec<usize>, usize, Vec<u64>, Vec<u64>, usize) {
        let (rows, cols) = (m.rows(), m.columns());
        let aug = reduced_augmented(m);
        let pivots: Vec<usize> = (0..aug.columns())
            .filter(|&c| aug.pivots()[c] >= 0)
            .collect();
        let rank_a = pivots.iter().take_while(|&&c| c < cols).count();
        let (_, t) = split(&aug, rows, cols);
        let sn = rows.div_ceil(64);
        let ts = t.stride();
        let mut q = Vec::new();
        let mut z = Vec::new();
        for i in 0..rows {
            let row = &t.data()[i * ts..i * ts + sn];
            if i < rank_a {
                q.extend_from_slice(row)
            } else {
                z.extend_from_slice(row)
            }
        }
        (pivots, rank_a, q, z, sn)
    }

    fn mem(m: &Matrix) -> MemRows<'_> {
        MemRows {
            data: m.data(),
            stride: m.stride(),
            rows: m.rows(),
            cols: m.columns(),
        }
    }

    /// Every reduction backend's output passes, with and without its transform, and the identity
    /// reduction passes for every claim shape.
    fn backends_pass(m: &Matrix, seed: u64) -> Result<(), TestCaseError> {
        let (rows, cols) = (m.rows(), m.columns());
        let mut a = m.clone();
        a.row_reduce();
        prop_assert_eq!(a.verify_rref_of(m, None, seed, K), Ok(()), "row_reduce");
        let aug = reduced_augmented(m);
        let (r, t) = split(&aug, rows, cols);
        prop_assert_eq!(
            r.verify_rref_of(m, Some(&t), seed, K),
            Ok(()),
            "augmented, with transform"
        );
        let (p, ra, q, z, sn) = identity_parts(m);
        let a_rows = mem(m);
        for (wq, wz) in [(true, true), (true, false), (false, true)] {
            prop_assert_eq!(
                verify_identity(
                    &a_rows,
                    &p,
                    ra,
                    wq.then_some((&q[..], sn)),
                    wz.then_some((&z[..], sn)),
                    seed,
                    K
                ),
                Ok(()),
                "identity q={} z={}",
                wq,
                wz
            );
        }
        Ok(())
    }

    proptest! {
        /// ORACLE-FREE: whatever reduces it, the result is verified as the RREF of the input.
        #[test]
        fn every_backend_passes_the_verifier(m in arb_matrix(150), seed in any::<u64>()) {
            backends_pass(&m, seed)?;
        }

        #[test]
        fn every_backend_passes_the_verifier_low_rank(m in arb_low_rank(), seed in any::<u64>()) {
            backends_pass(&m, seed)?;
        }
    }

    /// One corruption of a valid RREF `r` (with `rank` nonzero rows), chosen by `which`, or `None`
    /// if this matrix admits none of that kind.
    fn corrupt(r: &Matrix, rank: usize, which: usize, pick: u64) -> Option<Matrix> {
        let (rows, cols) = (r.rows(), r.columns());
        let mut bad = r.clone();
        let st = bad.stride();
        let pivot = |m: &Matrix, k: usize| -> usize {
            let row = &m.data()[k * st..];
            let l = row.iter().position(|&w| w != 0).unwrap();
            64 * l + row[l].trailing_zeros() as usize
        };
        let row_xor = |m: &mut Matrix, dst: usize, src: usize| {
            for w in 0..cols.div_ceil(64) {
                let v = m.data()[src * st + w];
                m.data_mut()[dst * st + w] ^= v;
            }
        };
        match which {
            // A flipped bit in a nonzero row. (In a zero row it can make an RREF of a LARGER row
            // space -- rank inflation -- which only the transform reveals: kind 5.)
            0 => {
                if rank < 1 {
                    return None;
                }
                let i = (pick as usize) % rank;
                let c = ((pick >> 20) as usize) % cols;
                bad.data_mut()[i * st + c / 64] ^= 1u64 << (c % 64);
            }
            // Two nonzero rows swapped.
            1 => {
                if rank < 2 {
                    return None;
                }
                let i = (pick as usize) % (rank - 1);
                for w in 0..cols.div_ceil(64) {
                    bad.data_mut().swap(i * st + w, (i + 1) * st + w);
                }
            }
            // A nonzero row dropped (the rows below move up, a zero row at the end).
            2 => {
                if rank < 1 {
                    return None;
                }
                let i = (pick as usize) % rank;
                for k in i..rows - 1 {
                    for w in 0..cols.div_ceil(64) {
                        let v = bad.data()[(k + 1) * st + w];
                        bad.data_mut()[k * st + w] = v;
                    }
                }
                for w in 0..cols.div_ceil(64) {
                    bad.data_mut()[(rows - 1) * st + w] = 0;
                }
            }
            // A skipped back-substitution: a later pivot row added back into an earlier one.
            3 => {
                if rank < 2 {
                    return None;
                }
                let j = (pick as usize) % (rank - 1);
                let k = j + 1 + ((pick >> 20) as usize) % (rank - 1 - j);
                row_xor(&mut bad, j, k);
            }
            // A flipped bit in a zero row (checked with the transform only).
            5 => {
                if rank >= rows {
                    return None;
                }
                let i = rank + (pick as usize) % (rows - rank);
                let c = ((pick >> 20) as usize) % cols;
                bad.data_mut()[i * st + c / 64] ^= 1u64 << (c % 64);
            }
            // A shifted pivot: row k's leading 1 moved one column right (into a free column).
            _ => {
                if rank < 1 {
                    return None;
                }
                let k = (pick as usize) % rank;
                let p = pivot(r, k);
                let next = if k + 1 < rank { pivot(r, k + 1) } else { cols };
                if p + 1 >= next {
                    return None;
                }
                bad.data_mut()[k * st + p / 64] ^= 1u64 << (p % 64);
                bad.data_mut()[k * st + (p + 1) / 64] ^= 1u64 << ((p + 1) % 64);
            }
        }
        bad.initialize_pivots();
        Some(bad)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 512,
            rng_seed: RngSeed::Fixed(0x5eed_0ff2_0ddb),
            ..ProptestConfig::default()
        })]

        /// NEGATIVE: a valid RREF corrupted in any single way -- a flipped bit, a swapped row, a
        /// dropped row, a skipped back-substitution, a shifted pivot -- is rejected, with and
        /// without the transform. Fixed seeds (cases and verifier): deterministic. With K = 20
        /// each randomized check misses a given corruption with probability <= 2^-20 ~ 1e-6.
        #[test]
        fn single_corruptions_are_rejected(m in arb_matrix(120), which in 0usize..6, pick in any::<u64>()) {
            let mut r = m.clone();
            let rank = r.row_reduce();
            r.initialize_pivots();
            if let Some(bad) = corrupt(&r, rank, which, pick) && bad != r {
                if which != 5 {
                    prop_assert!(bad.verify_rref_of(&m, None, pick ^ 0xabcd, K).is_err(), "kind {} accepted", which);
                }
                let aug = reduced_augmented(&m);
                let (_, t) = split(&aug, m.rows(), m.columns());
                prop_assert!(bad.verify_rref_of(&m, Some(&t), pick ^ 0xabcd, K).is_err(), "kind {} accepted with T", which);
            }
        }

        /// NEGATIVE, identity claims: a flipped bit in Q or Z, or a dropped pivot row, is
        /// rejected for every claim shape that carries it.
        #[test]
        fn identity_corruptions_are_rejected(m in arb_matrix(100), pick in any::<u64>()) {
            let (rows, cols) = (m.rows(), m.columns());
            let (p, ra, q, z, sn) = identity_parts(&m);
            let a = mem(&m);
            let seed = pick ^ 0x1234;
            if ra > 0 {
                let mut bq = q.clone();
                let (i, c) = ((pick as usize) % ra, ((pick >> 20) as usize) % rows);
                bq[i * sn + c / 64] ^= 1u64 << (c % 64);
                prop_assert!(verify_identity(&a, &p, ra, Some((&bq, sn)), None, seed, K).is_err());
                prop_assert!(verify_identity(&a, &p, ra, Some((&bq, sn)), Some((&z, sn)), seed, K).is_err());
            }
            if ra < rows {
                let mut bz = z.clone();
                let (i, c) = ((pick as usize) % (rows - ra), ((pick >> 20) as usize) % rows);
                bz[i * sn + c / 64] ^= 1u64 << (c % 64);
                prop_assert!(verify_identity(&a, &p, ra, None, Some((&bz, sn)), seed, K).is_err());
                prop_assert!(verify_identity(&a, &p, ra, Some((&q, sn)), Some((&bz, sn)), seed, K).is_err());
            }
            if ra >= 1 && ra < rows {
                // The last pivot row dropped, its pivot replaced by a spare identity column.
                let first_id = cols.div_ceil(64) * 64;
                let spare = (first_id..first_id + rows).find(|c| !p[ra..].contains(c)).unwrap();
                let mut dp: Vec<usize> = p[..ra - 1].to_vec();
                let mut tail: Vec<usize> = p[ra..].to_vec();
                tail.push(spare);
                tail.sort();
                dp.extend(tail);
                prop_assert!(verify_identity(&a, &dp, ra - 1, Some((&q[..(ra - 1) * sn], sn)), None, seed, K).is_err());
            }
        }
    }

    /// Sizes the O(n^3) reference cannot afford in a test suite: reduced by the blocked CPU
    /// backend and checked by the verifier alone (`FP_VERIFY_BIG_ROWS` x `FP_VERIFY_BIG_COLS`,
    /// default 20000 x 30000, rank 8000), plus an identity reduction at a quarter of that.
    #[test]
    fn large_reductions_checked_only_by_the_verifier() {
        let env = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let (rows, cols) = (
            env("FP_VERIFY_BIG_ROWS", 20_000),
            env("FP_VERIFY_BIG_COLS", 30_000),
        );
        let rank = rows * 2 / 5;
        let mut rng = super::Rng::seeded(42);
        let rand_m = |r: usize, c: usize, rng: &mut super::Rng| {
            let s = c.div_ceil(64);
            let mut d: Vec<u64> = (0..r * s).map(|_| rng.next()).collect();
            if !c.is_multiple_of(64) {
                for i in 0..r {
                    d[i * s + s - 1] &= (1u64 << (c % 64)) - 1;
                }
            }
            Matrix::from_data(TWO, r, c, d)
        };
        let m = &rand_m(rows, rank, &mut rng) * &rand_m(rank, cols, &mut rng);
        let t = std::time::Instant::now();
        let mut r = m.clone();
        let got = r.row_reduce();
        let tr = t.elapsed();
        let t = std::time::Instant::now();
        assert_eq!(r.verify_rref_of(&m, None, 99, K), Ok(()));
        eprintln!(
            "{rows}x{cols} rank {got}: reduced in {:.1}s, verified in {:.2}s",
            tr.as_secs_f64(),
            t.elapsed().as_secs_f64()
        );
        let (ir, ic) = (rows / 4, cols / 4);
        let a = &rand_m(ir, ir / 2, &mut rng) * &rand_m(ir / 2, ic, &mut rng);
        let (p, ra, q, z, sn) = identity_parts(&a);
        let t = std::time::Instant::now();
        for (wq, wz) in [(true, true), (true, false), (false, true)] {
            assert_eq!(
                verify_identity(
                    &mem(&a),
                    &p,
                    ra,
                    wq.then_some((&q[..], sn)),
                    wz.then_some((&z[..], sn)),
                    99,
                    K
                ),
                Ok(()),
                "identity {ir}x{ic} q={wq} z={wz}"
            );
        }
        eprintln!(
            "identity {ir}x{ic} rank {ra}: three claim shapes verified in {:.2}s",
            t.elapsed().as_secs_f64()
        );
    }
}
