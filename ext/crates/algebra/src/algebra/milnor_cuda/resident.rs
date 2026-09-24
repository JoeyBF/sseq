//! Device-resident state that outlives a launch: the admissible-matrix master, the Milnor basis,
//! and the seqno tables.
//!
//! # Why this exists
//!
//! A batch's per-product arrays are small -- tens of thousands of `u32` -- but the data those
//! arrays POINT AT is not. The admissible matrices of one `R` are reused by every product carrying
//! that `R`, and the same `R`s recur launch after launch; the basis grows only when a higher degree
//! is first seen. Re-marshalling and re-uploading them per call, which is what the first version of
//! this port did, makes host transfer scale with the NUMBER OF LAUNCHES instead of with the amount
//! of distinct data. Keeping them resident makes each launch upload only what is genuinely new.
//!
//! # What VMM deletes
//!
//! Under cubecl this same store needed a segment table, a per-segment upload path, a copy kernel to
//! move old contents into a larger allocation, a pinned host staging chunk, and a hard ceiling of
//! `MASTER_MAX_SEG` segments. All of it existed to work around one missing capability: growing an
//! allocation without moving it. CUDA's virtual memory API has that capability, so what is left
//! here is a pair of offsets and a `HashMap`.
//!
//! That ceiling was not cosmetic. It is what made every smaller card unusable: transient scratch
//! needed 18 to 23 segments against a cap of 16, so the work could not be expressed at all, and the
//! resulting allocation failure surfaced as an all-zero result at exit 0.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::driver::sys;

use rustc_hash::FxHashMap;

use super::{
    CudaError, GrowBuf, MilnorCuda, Result,
    enumerate::{enumerate_into, r_dims},
};
use crate::algebra::{Algebra, MilnorAlgebra, combinatorics::xi_degrees, milnor_algebra::PPart};

/// Address space reserved per buffer, in bytes, overridable by env var.
///
/// Reserve GENEROUSLY. Address space costs nothing -- no physical page is touched until `grow_to`
/// commits one -- and the reservation is the only hard ceiling on later growth, so an over-tight
/// value fails a run that would otherwise have fit. This is exactly the knob that does not exist
/// under cubecl, where the equivalent ceiling is `MASTER_MAX_SEG * seg_elems` and has to be tuned
/// per card against total device memory.
fn reserve_bytes(name: &str, default_gib: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_gib)
        << 30
}

/// How a caller names an `R`: `(r_degree, r_idx)`, exactly as it appears in a `GpuProduct`.
pub type RKey = (i32, usize);

/// Where one `R`'s admissible matrices live in the master, and how many there are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RInfo {
    /// ELEMENT offsets (not byte offsets) into the `cs`/`mk` masters.
    pub cs_offset: u64,
    pub mk_offset: u64,
    pub cs_len: u32,
    pub mk_len: u32,
    pub num_mats: u32,
}

/// `col_sums`/`masks` arrive as `u32` but are stored as `u16` on the device, halving the master.
///
/// CHECKED, never truncated. The one production bug this port must not reintroduce was an
/// unchecked narrowing of exactly this kind: the masks master crossing `2^32` was indexed through a
/// `u16` and silently truncated, which surfaced stems later as a non-zero differential at (180,92).
fn narrow(values: &[u32], what: &str) -> Result<Vec<u16>> {
    values
        .iter()
        .map(|&v| {
            u16::try_from(v).map_err(|_| {
                CudaError::Compile(format!(
                    "{what} entry {v} does not fit u16; master layout wrong"
                ))
            })
        })
        .collect()
}

/// Where each `R` sits in the master, and how far the master has been filled.
///
/// DEVICE-FREE on purpose. This is the bookkeeping that decides every offset the kernel indexes
/// with, and it is also the part most likely to be wrong -- an off-by-one here reads a different
/// `R`'s matrices and yields a plausible wrong answer rather than a crash. Keeping it separate lets
/// the device-free walk in `multiply.rs` exercise THE SAME CODE the real path uses, instead of a
/// reimplementation that could agree with itself while both are wrong.
#[derive(Default)]
pub struct MasterLayout {
    cs_elems: usize,
    mk_elems: usize,
    /// Keyed by `(r_degree, r_idx)`, NOT by the p-part.
    ///
    /// The two identify the same thing -- for a fixed degree the basis index determines the
    /// p-part, and the p-part determines the degree -- so this is a bijection and either key is
    /// correct. The pair is what a CALLER already has. Keying by p-part meant every launch called
    /// `basis_element_from_index` for each of its distinct `R`s just to ask "is this resident?",
    /// which on a warm launch is 54,107 constructions to answer yes 54,107 times.
    index: FxHashMap<RKey, RInfo>,
}

impl MasterLayout {
    pub fn get(&self, key: RKey) -> Option<RInfo> {
        self.index.get(&key).copied()
    }

    /// Reserve space for a newly enumerated `R` and record where it went.
    pub fn place(
        &mut self,
        key: RKey,
        cs_len: usize,
        mk_len: usize,
        num_mats: usize,
        cs_total: usize,
        mk_total: usize,
    ) -> RInfo {
        let info = RInfo {
            cs_offset: self.cs_elems as u64,
            mk_offset: self.mk_elems as u64,
            cs_len: cs_len as u32,
            mk_len: mk_len as u32,
            num_mats: num_mats as u32,
        };
        self.cs_elems += cs_total;
        self.mk_elems += mk_total;
        self.index.insert(key, info);
        info
    }

    pub fn cs_elems(&self) -> usize {
        self.cs_elems
    }

    pub fn mk_elems(&self) -> usize {
        self.mk_elems
    }
}

/// How far the resident basis has been appended, and where each degree starts.
///
/// Device-free for the same reason as [`MasterLayout`]: `gei` is what every uploaded `term_gei`
/// refers to, and it is fixed the first time a degree is appended.
#[derive(Debug)]
pub struct BasisLayout {
    global_base: Vec<u32>,
    elems: usize,
    degree: i32,
}

impl Default for BasisLayout {
    fn default() -> Self {
        Self {
            // `global_base[0] = 0`: no elements below degree 0.
            global_base: vec![0],
            elems: 0,
            // Nothing built yet; 0 would spuriously claim degree 0 is present.
            degree: -1,
        }
    }
}

impl BasisLayout {
    pub fn degree(&self) -> i32 {
        self.degree
    }

    pub fn elems(&self) -> usize {
        self.elems
    }

    /// `gei` for a term of degree `s_degree` at basis index `term_index`.
    pub fn gei(&self, s_degree: i32, term_index: usize) -> u32 {
        self.global_base[s_degree as usize] + term_index as u32
    }

    /// Record that `counts[i]` elements were appended for degree `degree + 1 + i`.
    pub fn extend(&mut self, counts: &[usize]) {
        for &n in counts {
            self.elems += n;
            self.global_base.push(self.elems as u32);
        }
        self.degree += counts.len() as i32;
    }
}

/// Enumerate one `R`'s admissible matrices in device layout:
/// `(cs_len, mk_len, num_mats, col_sums, masks)`.
pub fn r_tables(
    algebra: &MilnorAlgebra,
    r_p_part: &PPart,
) -> Result<(usize, usize, usize, Vec<u16>, Vec<u16>)> {
    // `Sq(empty) = 1` has no admissible matrices and is the caller's job, exactly as in
    // `AdmissibleMatrix::new`.
    if r_p_part.is_empty() {
        return Err(CudaError::Compile(
            "empty R has no admissible matrices; Sq(1) is not a batch product".to_owned(),
        ));
    }
    let (cs_len, mk_len, cs_v, mk_v) = algebra.admissible_matrices(r_p_part.clone());
    // Every matrix of a fixed `R` shares `cs_len`/`mk_len`, so the flattening is rectangular and
    // the count divides exactly. `cs_len` is 0 when the matrix has a single column, hence the
    // fallback to `mk`.
    let num_mats = if cs_len > 0 {
        cs_v.len() / cs_len
    } else {
        mk_v.len() / mk_len.max(1)
    };
    Ok((
        cs_len,
        mk_len,
        num_mats,
        narrow(&cs_v, "col_sums")?,
        narrow(&mk_v, "masks")?,
    ))
}

/// Packed p-parts, true lengths, and the per-degree element counts, for degrees `from..=to`.
///
/// ONE `u64` PER ELEMENT, not a width-padded run of `u16`.
///
/// `PPart` already IS a single `u64` -- ten fields of widths 11, 10, 9, 8, 7, 6, 5, 4, 3, 1, which
/// is exactly 64 bits -- so storing it unpacked was storing 20 bytes for something that is 8, and
/// making the kernel do a global load per COLUMN for a value one shift away in a register. The
/// device-side accumulator was already using this exact layout (`PP_SHIFT`/`PP_MASK` come from
/// `PPart::shift`/`width`), so the term and the accumulator now speak the same representation and
/// no conversion happens anywhere.
///
/// `ln` stays: the packed word gives the entries, but the column loop still needs the trimmed
/// LENGTH to bound itself and to compute `min(term_len, cs_len)`.
pub fn basis_tables(
    algebra: &MilnorAlgebra,
    from: i32,
    to: i32,
) -> Result<(Vec<u64>, Vec<u32>, Vec<usize>)> {
    let mut pp: Vec<u64> = Vec::new();
    let mut ln: Vec<u32> = Vec::new();
    let mut counts: Vec<usize> = Vec::new();
    for d in from..=to {
        let dim = algebra.dimension(d);
        counts.push(dim);
        for i in 0..dim {
            let elt = algebra.basis_element_from_index(d, i);
            ln.push(elt.p_part.len() as u32);
            pp.push(elt.p_part.bits());
        }
    }
    Ok((pp, ln, counts))
}

/// The xi-degrees, padded so the seqno loop's fixed `PPART_MAX_LEN` trip count stays in bounds.
///
/// Entries past a p-part's true length are zero and contribute `0 * xi`, so the loop running long
/// is harmless -- but only if the table is long enough to read.
pub fn xi_table(algebra: &MilnorAlgebra) -> Vec<u32> {
    let mut xi: Vec<u32> = xi_degrees(algebra.prime())
        .iter()
        .map(|&d| d as u32)
        .collect();
    xi.resize(xi.len().max(super::params::PPART_MAX_LEN), 0);
    xi
}

/// Everything the multiply kernel reads that is not specific to one launch.
///
/// Append-only in every field. That is what makes the device pointers safe to hand to a kernel
/// while more is being appended: existing bytes never move and are never freed, which is the
/// property the whole segmented master was a workaround for.
pub struct Resident {
    rt: Arc<MilnorCuda>,

    /// Admissible-matrix master, concatenated over distinct `R`s, as `u16`.
    cs: GrowBuf,
    mk: GrowBuf,
    master: MasterLayout,

    /// The Milnor basis as one packed `u64` per element, and the true p-part lengths.
    pp: GrowBuf,
    ln: GrowBuf,
    basis: BasisLayout,

    /// Seqno tables. `g` is REPLACED rather than appended when the degree grows -- the algebra
    /// rebuilds the whole dense table -- but it is small next to the master.
    g: GrowBuf,
    xi: GrowBuf,
    seqno_degree: i32,

    /// Shared stride of `g` and the padded basis.
    width: usize,
}

impl Resident {
    fn new(rt: Arc<MilnorCuda>) -> Result<Self> {
        let dev = rt.device();
        rt.context().bind_to_thread().map_err(|_| {
            CudaError::Driver("bind_to_thread", sys::CUresult::CUDA_ERROR_INVALID_CONTEXT)
        })?;
        Ok(Self {
            cs: GrowBuf::reserve(dev, reserve_bytes("NASSAU_CUDA_RESERVE_CS_GIB", 48))?,
            mk: GrowBuf::reserve(dev, reserve_bytes("NASSAU_CUDA_RESERVE_MK_GIB", 48))?,
            master: MasterLayout::default(),
            pp: GrowBuf::reserve(dev, reserve_bytes("NASSAU_CUDA_RESERVE_PP_GIB", 8))?,
            ln: GrowBuf::reserve(dev, reserve_bytes("NASSAU_CUDA_RESERVE_LN_GIB", 2))?,
            basis: BasisLayout::default(),
            g: GrowBuf::reserve(dev, reserve_bytes("NASSAU_CUDA_RESERVE_G_GIB", 2))?,
            xi: GrowBuf::reserve(dev, 1)?,
            seqno_degree: -1,
            width: 0,
            rt,
        })
    }

    /// Bind this device's context. Every raw `sys::` path below needs it, and forgetting it is the
    /// trap that once had `probe_max_alloc` reporting 0 GiB on a perfectly healthy card.
    fn bind(&self) -> Result<()> {
        self.rt.context().bind_to_thread().map_err(|_| {
            CudaError::Driver("bind_to_thread", sys::CUresult::CUDA_ERROR_INVALID_CONTEXT)
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn basis(&self) -> &BasisLayout {
        &self.basis
    }

    /// Where an `R` lives, if it is resident. `None` means it was never enumerated.
    pub fn master_get(&self, key: RKey) -> Option<RInfo> {
        self.master.get(key)
    }

    pub fn cs_ptr(&self) -> sys::CUdeviceptr {
        self.cs.ptr()
    }

    pub fn mk_ptr(&self) -> sys::CUdeviceptr {
        self.mk.ptr()
    }

    pub fn pp_ptr(&self) -> sys::CUdeviceptr {
        self.pp.ptr()
    }

    pub fn ln_ptr(&self) -> sys::CUdeviceptr {
        self.ln.ptr()
    }

    pub fn g_ptr(&self) -> sys::CUdeviceptr {
        self.g.ptr()
    }

    pub fn xi_ptr(&self) -> sys::CUdeviceptr {
        self.xi.ptr()
    }

    /// Device memory actually COMMITTED across every resident buffer.
    ///
    /// Committed, not reserved: reserved address space is free, and reporting it would make an idle
    /// process look like it is holding 100+ GB.
    pub fn committed_bytes(&self) -> usize {
        self.cs.committed()
            + self.mk.committed()
            + self.pp.committed()
            + self.ln.committed()
            + self.g.committed()
            + self.xi.committed()
    }

    /// Upload the seqno tables if the degree has grown past what is resident.
    pub fn ensure_seqno(&mut self, algebra: &MilnorAlgebra, max_degree: i32) -> Result<()> {
        if self.width != 0 && self.seqno_degree >= max_degree {
            return Ok(());
        }
        self.bind()?;
        let (width, g) = algebra.seqno_table_u32();
        if self.width != 0 && self.width != width {
            // The stride is shared by `g` and the padded basis, so a change would silently
            // reinterpret every p-part already resident.
            return Err(CudaError::Compile(format!(
                "seqno table width changed from {} to {width}; the resident basis stride is fixed",
                self.width
            )));
        }
        self.width = width;
        self.g.write_at(0, &g)?;
        self.xi.write_at(0, &xi_table(algebra))?;
        self.seqno_degree = max_degree;
        Ok(())
    }

    /// Append every basis element of the degrees not yet resident, up to `max_degree`.
    ///
    /// Append-only, so an element's `gei` is fixed the first time its degree is seen and every
    /// `term_gei` ever uploaded stays valid.
    pub fn ensure_basis(&mut self, algebra: &MilnorAlgebra, max_degree: i32) -> Result<()> {
        if self.basis.degree() >= max_degree {
            return Ok(());
        }
        assert!(self.width != 0, "ensure_seqno must run before ensure_basis");
        self.bind()?;
        let (pp, ln, counts) = basis_tables(algebra, self.basis.degree() + 1, max_degree)?;
        // Write BEFORE advancing the layout, so a failed upload cannot leave the layout claiming
        // elements the device does not have.
        self.pp
            .write_at(self.basis.elems() * size_of::<u64>(), &pp)?;
        self.ln
            .write_at(self.basis.elems() * size_of::<u32>(), &ln)?;
        self.basis.extend(&counts);
        Ok(())
    }

    /// Make every `R` in `p_parts` resident, enumerating the new ones ON THE DEVICE, and return
    /// their placements in the same order.
    ///
    /// This is the path that matters. [`Self::ensure_r`] enumerates on the CPU, one `R` at a time,
    /// which is fine for a test and wrong for a workload twice over: the host enumeration is the
    /// cost the enum kernel exists to remove, and doing it per `R` means one launch per `R` when
    /// what the device wants is one launch per BATCH. A production batch's distinct `R` count is in
    /// the thousands, and the enum kernel's duration is set by its longest single `R` rather than
    /// by how many it carries -- so merging them turns a sum into a max.
    ///
    /// The enumerated matrices never touch the host. The count pass sizes each block, the layout
    /// places it, the buffers grow, and the emit pass writes straight into them.
    pub fn ensure_rs(
        &mut self,
        rt: &Arc<MilnorCuda>,
        algebra: &MilnorAlgebra,
        keys: &[RKey],
    ) -> Result<Vec<RInfo>> {
        // Not already resident, in first-seen order. `keys` is already distinct (`plan_rs` sees to
        // that), so the only question per key is whether the master has it -- and ONLY a miss pays
        // for `basis_element_from_index`. On a warm launch nothing here materialises a p-part.
        let mut fresh_keys: Vec<RKey> = Vec::new();
        let mut fresh: Vec<PPart> = Vec::new();
        for &key in keys {
            if self.master.get(key).is_none() {
                let p = algebra.basis_element_from_index(key.0, key.1).p_part;
                if p.is_empty() {
                    return Err(CudaError::Compile(
                        "empty R has no admissible matrices; Sq(1) is not a batch product"
                            .to_owned(),
                    ));
                }
                fresh_keys.push(key);
                fresh.push(p);
            }
        }

        if !fresh.is_empty() {
            self.bind()?;
            // `cs_len`/`mk_len` come from the p-part's SHAPE, so they are known before anything is
            // enumerated; only the matrix COUNT needs the device.
            let dims: Vec<(usize, usize)> = fresh.iter().map(r_dims).collect();
            let lens: Vec<(usize, usize)> = dims
                .iter()
                .map(|&(rows, cols)| (cols - 1, rows + cols - 1))
                .collect();

            // PASS 1: counts. Nothing is written, so the pointers only have to bind.
            let zero = vec![0u64; fresh.len()];
            let counts = enumerate_into(
                rt,
                &fresh,
                false,
                self.cs.ptr(),
                self.mk.ptr(),
                &zero,
                &zero,
            )?;

            // Place each block, then commit the backing store BEFORE the emit pass writes into it.
            let mut cs_offsets = Vec::with_capacity(fresh.len());
            let mut mk_offsets = Vec::with_capacity(fresh.len());
            let mut infos = Vec::with_capacity(fresh.len());
            for (i, &key) in fresh_keys.iter().enumerate() {
                let (cs_len, mk_len) = lens[i];
                let n = counts[i] as usize;
                let info = self
                    .master
                    .place(key, cs_len, mk_len, n, n * cs_len, n * mk_len);
                cs_offsets.push(info.cs_offset);
                mk_offsets.push(info.mk_offset);
                infos.push(info);
            }
            self.cs.grow_to(self.master.cs_elems() * size_of::<u16>())?;
            self.mk.grow_to(self.master.mk_elems() * size_of::<u16>())?;

            // PASS 2: fill, directly into the resident buffers.
            let recount = enumerate_into(
                rt,
                &fresh,
                true,
                self.cs.ptr(),
                self.mk.ptr(),
                &cs_offsets,
                &mk_offsets,
            )?;
            // The passes run the same odometer, so a disagreement means the emit pass wrote outside
            // the space the count pass sized -- the master is already corrupt, and the multiply that
            // reads it would return a plausible wrong answer rather than fail.
            if recount != counts {
                return Err(CudaError::Compile(
                    "the enumeration's count and emit passes disagree; the master is not \
                     trustworthy"
                        .to_owned(),
                ));
            }
        }

        keys.iter()
            .map(|&k| {
                self.master
                    .get(k)
                    .ok_or_else(|| CudaError::Compile(format!("R {k:?} was not made resident")))
            })
            .collect()
    }

    /// Enumerate and append one `R`'s admissible matrices ON THE CPU, or return what is
    /// already resident.
    ///
    /// Kept as the fallback and as the thing [`Self::ensure_rs`] is checked against: the two
    /// must produce identical masters, which is what `cuda_enumerate_matches_cpu_reference`
    /// asserts element by element. Prefer `ensure_rs` for real work -- this enumerates on the
    /// host, which is the cost the enum kernel exists to remove.
    pub fn ensure_r(&mut self, algebra: &MilnorAlgebra, key: RKey) -> Result<RInfo> {
        if let Some(info) = self.master.get(key) {
            return Ok(info);
        }
        self.bind()?;
        let r_p_part = algebra.basis_element_from_index(key.0, key.1).p_part;
        let (cs_len, mk_len, num_mats, cs_u, mk_u) = r_tables(algebra, &r_p_part)?;
        let cs_at = self.master.cs_elems() * size_of::<u16>();
        let mk_at = self.master.mk_elems() * size_of::<u16>();
        self.cs.write_at(cs_at, &cs_u)?;
        self.mk.write_at(mk_at, &mk_u)?;
        Ok(self
            .master
            .place(key, cs_len, mk_len, num_mats, cs_u.len(), mk_u.len()))
    }
}

type ResidentMap = Mutex<HashMap<i32, &'static Mutex<Resident>>>;

static RESIDENTS: OnceLock<ResidentMap> = OnceLock::new();

/// The process-wide resident state for `rt`'s device, created on first use.
///
/// One mutex per device, held across a whole launch. That is coarse, and deliberately so for now:
/// correctness first. The cubecl path's finer scheme -- a read lock on the handles, a separate
/// upload mutex, segment handles cloned per launch so growth never invalidates a live kernel's
/// view -- is a later commit, made against a digest that is already pinned.
pub fn resident(rt: &Arc<MilnorCuda>) -> Result<&'static Mutex<Resident>> {
    let map = RESIDENTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    if let Some(r) = guard.get(&rt.device()) {
        return Ok(r);
    }
    // Leaked on purpose: the resident store lives for the process, and its device allocations must
    // outlive every launch holding a pointer into them.
    let slot: &'static Mutex<Resident> =
        Box::leak(Box::new(Mutex::new(Resident::new(rt.clone())?)));
    guard.insert(rt.device(), slot);
    Ok(slot)
}
