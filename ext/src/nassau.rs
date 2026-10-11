//! This module implements [Nassau's algorithm](https://arxiv.org/abs/1910.04063).
//!
//! The main export is the [`Resolution`] object, which is a resolution of the sphere at the prime 2
//! using Nassau's algorithm. It aims to provide an API similar to
//! [`resolution::Resolution`](crate::resolution::Resolution). From an API point of view, the main
//! difference between the two is that our `Resolution` is a chain complex over [`MilnorAlgebra`]
//! over [`SteenrodAlgebra`](algebra::SteenrodAlgebra).
//!
//! To make use of this resolution in the example scripts, enable the `nassau` feature. This will
//! cause [`utils::query_module`](crate::utils::query_module) to return the `Resolution` from this
//! module instead of [`resolution`](crate::resolution). There is no formal polymorphism involved;
//! the feature changes the return type of the function. While this is an incorrect use of features,
//! we find that this the easiest way to make all scripts support both types of resolutions.

use std::{
    collections::HashMap,
    fmt::Display,
    io,
    sync::{Arc, Mutex, mpsc},
};

use algebra::{
    Algebra, combinatorics,
    milnor_algebra::{MilnorAlgebra, PPart, PPartEntry},
    module::{
        FreeModule, GeneratorData, Module, ZeroModule,
        homomorphism::{FreeModuleHomomorphism, FullModuleHomomorphism, ModuleHomomorphism},
    },
};
use anyhow::anyhow;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use fp::{
    matrix::{AugmentedMatrix, Matrix},
    prime::{TWO, ValidPrime},
    vector::{FpSlice, FpSliceMut, FpVector},
};
use itertools::{Either, Itertools};
use once::OnceBiVec;
use sseq::coordinates::{Bidegree, BidegreeGenerator};

use crate::{
    chain_complex::{AugmentedChainComplex, ChainComplex, FiniteChainComplex},
    save::{SaveDirectory, SaveKind},
    utils::{LogWriter, parallel::ParallelGuard},
};

/// What a computed bidegree still has to register.
///
/// `modules[s]` and `differentials[s]` are append-only in increasing degree, so registration has to
/// happen in `t` order within a row even when the computations that produced it did not. Carrying
/// it as a value lets the scheduler apply it in graph order; nothing waits on a lock to do so.
pub(crate) struct PendingRegistration {
    b: Bidegree,
    num_new_gens: usize,
    /// One row per new generator: its differential, in the target's full basis.
    rows: Vec<FpVector>,
    /// Column count the rows were built against, needed by the save format.
    target_dim: usize,
    /// False when the differential was just READ from a save file and must not be rewritten.
    write_save: bool,
    /// Whether to extend the chain map by zero. The main path does; the save-load path never did,
    /// and this keeps that difference rather than quietly changing it.
    extend_chain_map: bool,
}

/// See [`resolution::SenderData`](../resolution/struct.SenderData.html).
///
/// This differs by carrying a [`PendingRegistration`] instead of the `new` field.
struct SenderData {
    b: Bidegree,
    retry: bool,
    /// What the worker computed and the scheduler still has to register, if anything.
    pending: Option<PendingRegistration>,
    sender: mpsc::Sender<Self>,
}

impl SenderData {
    /// Report `b` as computed, with whatever the scheduler still has to register.
    pub(crate) fn send(
        b: Bidegree,
        pending: Option<PendingRegistration>,
        sender: mpsc::Sender<Self>,
    ) {
        sender
            .send(Self {
                b,
                retry: false,
                pending,
                sender: sender.clone(),
            })
            .unwrap()
    }

    /// Hand `b` back to the scheduler to respawn, because its worker was already inside a parallel
    /// section.
    pub(crate) fn send_retry(b: Bidegree, sender: mpsc::Sender<Self>) {
        tracing::info!(%b, "retrying");
        sender
            .send(Self {
                b,
                retry: true,
                pending: None,
                sender: sender.clone(),
            })
            .unwrap()
    }
}

const MAX_NEW_GENS: usize = 10;

/// A Milnor subalgebra to be used in [Nassau's algorithm](https://arxiv.org/abs/1910.04063). This
/// is equipped with an ordering of the signature as in Lemma 2.4 of the paper.
///
/// To simplify implementation, we pick the ordering so that the (reverse) lexicographic ordering
/// in Lemma 2.4 is just the (reverse) lexicographic ordering of the P parts. This corresponds to
/// the ordering of $\mathcal{P}$ where $P^s_t < P^{s'}_t$ if $s < s'$).
#[derive(Clone)]
struct MilnorSubalgebra {
    profile: Vec<u8>,
}

impl MilnorSubalgebra {
    /// This should be used when you want an entry of the profile to be infinity
    #[allow(dead_code)]
    const INFINITY: u8 = (std::mem::size_of::<PPartEntry>() * 4 - 1) as u8;

    fn new(profile: Vec<u8>) -> Self {
        Self { profile }
    }

    /// The algebra with trivial profile, corresponding to the trivial algebra.
    fn zero_algebra() -> Self {
        Self { profile: vec![] }
    }

    /// The test "does this element have this signature" compiled into a `(mask, value)` pair to
    /// match against the packed p-part.
    ///
    /// The per-entry test is `ppart[i] & ((1 << profile[i]) - 1) == signature[i]`. Because each
    /// entry occupies a fixed field of the packed word, the low `profile[i]` bits of entry `i` are
    /// a fixed bit range of that word, so the whole conjunction is a single `&` and `==`. Entries
    /// past the end of the p-part read as zero, which the packing already gives us for free.
    ///
    /// Returns `None` when an entry is too large to be one, since then no element matches.
    fn packed_signature(&self, signature: &[PPartEntry]) -> Option<(u64, u64)> {
        let mut mask = 0;
        let mut value = 0;
        for (i, (&profile, &entry)) in self.profile.iter().zip(signature).enumerate() {
            // An entry wider than its field would shift into the next one, so it cannot simply be
            // packed and compared.
            if entry > PPart::max_entry(i) {
                return None;
            }
            // A profile wider than the field constrains the whole field.
            let width = std::cmp::min(profile as u32, PPart::width(i));
            mask |= ((1u64 << width) - 1) << PPart::shift(i);
            value |= (entry as u64) << PPart::shift(i);
        }
        Some((mask, value))
    }

    fn zero_signature(&self) -> Vec<PPartEntry> {
        vec![0; self.profile.len()]
    }

    /// The smallest POSITIVE degree of an operation carrying the zero signature.
    ///
    /// An operation has the zero signature when every p-part entry is zero modulo its field
    /// width, so the smallest non-trivial one is `min_i 2^{w_i} (2^i - 1)`; for `A(k)` that is
    /// `2^{k + 1}`, i.e. 2, 4, 8, 16, 32 for `A(0)` through `A(4)`.
    ///
    /// This bounds how far ahead of its own row a bidegree may be computed. The image is built at
    /// the zero signature, so a source generator whose operation degree is below this floor
    /// contributes no basis element to it, and excluding such generators cannot change the answer.
    ///
    /// It is a property of the SUBALGEBRA, hence of the bidegree. A single global bound is wrong:
    /// one large enough to help an `A(3)` bidegree exceeds the floor of a nearby `A(0)` or `A(1)`
    /// one, and excluding generators that do contribute inflates the generator count without bound.
    fn zero_signature_floor(&self) -> i32 {
        self.profile
            .iter()
            .enumerate()
            .map(|(i, &entry)| {
                let width = std::cmp::min(entry as u32, PPart::width(i));
                ((1i64 << width) * ((1i64 << (i + 1)) - 1)) as i32
            })
            .min()
            .unwrap_or(1)
    }

    /// The internal degree of a signature: entry `i` stands for `xi_{i + 1}`, of degree
    /// `2^{i + 1} - 1`.
    fn signature_degree(signature: &[PPartEntry]) -> i32 {
        signature
            .iter()
            .enumerate()
            .map(|(i, &r)| ((1i32 << (i + 1)) - 1) * r as i32)
            .sum()
    }

    /// Give a list of basis elements in degree `degree` that has signature `signature`.
    ///
    /// Only basis elements coming from generators of degree strictly less than `max_gen_degree`
    /// are considered; `None` imposes no restriction. Because generators are laid out in
    /// increasing degree, a restricted result is a prefix of the unrestricted one; see
    /// [`Resolution::step_resolution_with_subalgebra`] for why we restrict.
    ///
    /// This requires passing the algebra for borrow checker reasons.
    fn signature_mask<'a>(
        &'a self,
        algebra: &'a MilnorAlgebra,
        module: &'a FreeModule<MilnorAlgebra>,
        degree: i32,
        signature: &'a [PPartEntry],
        max_gen_degree: Option<i32>,
    ) -> impl Iterator<Item = usize> + 'a {
        // Every element is tested against the same signature, so compile it once.
        let Some((mask, value)) = self.packed_signature(signature) else {
            return Either::Right(std::iter::empty());
        };

        let matching = module
            .iter_gen_offsets([degree])
            .take_while(move |gen_data| max_gen_degree.is_none_or(|bound| gen_data.gen_deg < bound))
            .flat_map(move |gen_data| {
                let GeneratorData {
                    gen_deg,
                    start: [offset],
                    ..
                } = gen_data;
                algebra
                    .ppart_table(degree - gen_deg)
                    .iter()
                    .enumerate()
                    .filter_map(move |(n, op)| (op.bits() & mask == value).then_some(offset + n))
            });

        Either::Left(matching)
    }

    /// Get the matrix of a free module homomorphism when restricted to the subquotient given by
    /// the signature.
    ///
    /// Only generators of the target of degree strictly less than `target_max_gen_degree` are used
    /// (see [`Self::signature_mask`]).
    fn signature_matrix(
        &self,
        hom: &FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>,
        degree: i32,
        signature: &[PPartEntry],
        target_max_gen_degree: i32,
        // Generators of the SOURCE at or above this degree are excluded; `None` reads them all. The
        // image is built at the zero signature, so passing the zero-signature floor here lets a
        // bidegree be computed before its own row predecessor has registered.
        source_max_gen_degree: Option<i32>,
    ) -> Matrix {
        let p = hom.prime();
        let source = hom.source();
        let target = hom.target();
        let algebra = target.algebra();
        let target_degree = degree - hom.degree_shift();

        let target_mask: Vec<usize> = self
            .signature_mask(
                &algebra,
                &target,
                target_degree,
                signature,
                Some(target_max_gen_degree),
            )
            .collect();

        let source_mask: Vec<usize> = self
            .signature_mask(&algebra, &source, degree, signature, source_max_gen_degree)
            .collect();

        let mut scratch = FpVector::new(
            p,
            target.dimension_from_gens_below(target_degree, target_max_gen_degree),
        );
        let mut result = Matrix::new(p, source_mask.len(), target_mask.len());

        for (mut row, &masked_index) in std::iter::zip(result.iter_mut(), &source_mask) {
            scratch.set_to_zero();
            hom.apply_to_basis_element_restricted(scratch.as_slice_mut(), 1, degree, masked_index);

            row.add_masked(scratch.as_slice(), 1, &target_mask);
        }
        result
    }

    /// Iterate through all signatures of this algebra that contain elements of degree at most
    /// `degree` (inclusive). This skips the initial zero signature.
    fn iter_signatures(&self, degree: i32) -> impl Iterator<Item = Vec<PPartEntry>> + '_ {
        SignatureIterator::new(self, degree)
    }

    fn top_degree(&self) -> i32 {
        self.profile
            .iter()
            .map(|&entry| (1 << entry) - 1)
            .enumerate()
            .map(|(idx, entry)| ((1 << (idx + 1)) - 1) * entry)
            .sum()
    }

    fn optimal_for(b: Bidegree) -> Self {
        let b_is_in_vanishing_region = |subalgebra: &Self| {
            let coeff = (1 << subalgebra.profile.len()) - 1;
            b.t() >= coeff * (b.s() + 1) + subalgebra.top_degree()
        };
        SubalgebraIterator::new()
            .take_while(b_is_in_vanishing_region)
            .last()
            .unwrap_or(Self::zero_algebra())
    }

    fn to_bytes(&self, buffer: &mut impl io::Write) -> io::Result<()> {
        buffer.write_u64::<LittleEndian>(self.profile.len() as u64)?;
        buffer.write_all(&self.profile)?;

        let len = self.profile.len();
        let zeros = [0; 8];
        let padding = len - ((len / 8) * 8);
        buffer.write_all(&zeros[0..padding])
    }

    fn from_bytes(data: &mut impl io::Read) -> io::Result<Self> {
        // The packed p-part has no entry past `PPart::MAX_LEN`, so a longer profile cannot be
        // matched against one. This is the only place a profile is built from outside data, and
        // the bound has to hold before narrowing, which truncates where `usize` is 32 bits.
        let len = data.read_u64::<LittleEndian>()?;
        if len > PPart::MAX_LEN as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("profile length {len} exceeds {}", PPart::MAX_LEN),
            ));
        }
        let len = len as usize;
        let mut profile = vec![0; len];

        data.read_exact(&mut profile)?;

        let padding = len - ((len / 8) * 8);
        if padding > 0 {
            let mut buf: [u8; 8] = [0; 8];
            data.read_exact(&mut buf[0..padding])?;
            assert_eq!(buf, [0; 8]);
        }
        Ok(Self { profile })
    }

    fn signature_to_bytes(signature: &[PPartEntry], buffer: &mut impl io::Write) -> io::Result<()> {
        if cfg!(target_endian = "little") && std::mem::size_of::<PPartEntry>() == 2 {
            unsafe {
                let buf: &[u8] = std::slice::from_raw_parts(
                    signature.as_ptr() as *const u8,
                    signature.len() * 2,
                );
                buffer.write_all(buf).unwrap();
            }
        } else {
            for &entry in signature {
                buffer.write_u16::<LittleEndian>(entry as u16)?;
            }
        }

        let len = signature.len();
        let zeros = [0; 8];
        let padding = len - ((len / 4) * 4);

        if padding > 0 {
            buffer.write_all(&zeros[0..padding * 2])?;
        }
        Ok(())
    }

    fn signature_from_bytes(&self, data: &mut impl io::Read) -> io::Result<Vec<PPartEntry>> {
        let len = self.profile.len();
        let mut signature: Vec<PPartEntry> = vec![0; len];

        if cfg!(target_endian = "little") && std::mem::size_of::<PPartEntry>() == 2 {
            unsafe {
                let buf: &mut [u8] =
                    std::slice::from_raw_parts_mut(signature.as_mut_ptr() as *mut u8, len * 2);
                data.read_exact(buf).unwrap();
            }
        } else {
            for entry in &mut signature {
                *entry = data.read_u16::<LittleEndian>()? as PPartEntry;
            }
        }

        let padding = len - ((len / 4) * 4);
        if padding > 0 {
            let mut buffer: [u8; 8] = [0; 8];
            data.read_exact(&mut buffer[0..padding * 2])?;
            assert_eq!(buffer, [0; 8]);
        }
        Ok(signature)
    }
}

impl Display for MilnorSubalgebra {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        if self.profile.is_empty() {
            write!(out, "F_2")
        } else if self.profile.len() as u8 == self.profile[0] {
            write!(out, "A({})", self.profile.len() - 1)
        } else {
            write!(out, "B({})", self.profile.iter().join(","))
        }
    }
}

/// An iterator that iterates through a sequence of [`MilnorSubalgebra`] of increasing size. This
/// is used by [`MilnorSubalgebra::optimal_for`] to find the largest subalgebra in this sequence
/// that is applicable to a bidegree.
struct SubalgebraIterator {
    current: MilnorSubalgebra,
}

impl SubalgebraIterator {
    fn new() -> Self {
        Self {
            current: MilnorSubalgebra::new(vec![]),
        }
    }
}

impl Iterator for SubalgebraIterator {
    type Item = MilnorSubalgebra;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.profile.is_empty()
            || self.current.profile[0] == self.current.profile.len() as u8
        {
            // We are at F_2 or at A(n) where n = self.current.profile.len() - 1.
            self.current.profile.push(1);
            Some(self.current.clone())
        } else {
            // We find the first entry that can be incremented and increment it
            if let Some((_, entry)) = self
                .current
                .profile
                .iter_mut()
                .rev()
                .enumerate()
                .find(|(idx, entry)| **entry == *idx as u8)
            {
                *entry += 1;
            }
            Some(self.current.clone())
        }
    }
}

/// See [`MilnorSubalgebra::iter_signatures`].
struct SignatureIterator<'a> {
    subalgebra: &'a MilnorSubalgebra,
    current: Vec<PPartEntry>,
    signature_degree: i32,
    degree: i32,
}

impl<'a> SignatureIterator<'a> {
    fn new(subalgebra: &'a MilnorSubalgebra, degree: i32) -> Self {
        Self {
            current: vec![0; subalgebra.profile.len()],
            degree,
            subalgebra,
            signature_degree: 0,
        }
    }
}

impl Iterator for SignatureIterator<'_> {
    type Item = Vec<PPartEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        let xi_degrees = combinatorics::xi_degrees(TWO);
        let len = self.current.len();
        for (i, current) in self.current.iter_mut().enumerate() {
            *current += 1;
            self.signature_degree += xi_degrees[i];

            if self.signature_degree > self.degree || *current == 1 << self.subalgebra.profile[i] {
                self.signature_degree -= xi_degrees[i] * *current as i32;
                *current = 0;
                if i + 1 == len {
                    return None;
                }
            } else {
                return Some(self.current.clone());
            }
        }
        // This only happens when the profile is trivial
        assert!(self.current.is_empty());
        None
    }
}

/// Some magic constants used in the save file
enum Magic {
    End = -1,
    Signature = -2,
    Fix = -3,
}

/// Signature-shift reuse: serving every nonzero signature's matrix from one zero-signature block.
///
/// For a signature `sigma` of degree `delta`, `Sq(R) g -> Sq(R - sigma) g` is a bijection from the
/// basis elements of signature `sigma` in degree `t` onto those of signature zero in degree
/// `t - delta`, and in the signature-graded pieces it carries the differential to the
/// differential. So the masked matrix the signature step at `(s, t)` reduces is, row for row and
/// column for column, a prefix of the zero-signature matrix of `d_{s - 1}` in degree `t - delta`.
/// That matrix depends only on `(s, t - delta, profile)`, so one build serves every signature of
/// that degree, in every bidegree of row `s` that shifts onto it.
///
/// The block for key `(s, d, P)` has
///
/// * rows: the zero-signature basis of `C_{s - 1}` in degree `d`, from every generator of degree
///   `<= d` (no generator of higher degree has an element there);
/// * columns: the zero-signature basis of `C_{s - 2}` in degree `d`, from every generator of
///   degree `<= d`, stored masked.
///
/// A consumer at `(s, t)` reads all the rows and a prefix of the columns: its own column bound,
/// `t - 1`, keeps exactly the generators of degree `<= d` when `delta >= 2`, and one degree fewer
/// when `delta == 1`. Building the full block reads `C_{s - 2}`'s generators of degree `d`, which
/// for `delta == 1` (`d = t - 1`) another worker may still be adding, so such a consumer builds a
/// private, narrower block on a miss and does not share it.
///
/// A miss always rebuilds, so the cache can only change how often a block is built, never what
/// any step computes.
mod shift {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use fp::matrix::Matrix;
    use rustc_hash::FxHashMap;

    /// `(s, d, profile)`: the block of `d_{s - 1}` in degree `d` under that subalgebra.
    ///
    /// The profile is part of the key because neighbouring bidegrees may pick different
    /// subalgebras, and a different zero signature is a different mask in the same degree.
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    pub(super) struct Key {
        pub s: i32,
        pub degree: i32,
        pub profile: Vec<u8>,
    }

    /// See the [module documentation](self).
    pub(super) type Block = Arc<Matrix>;

    /// Counters describing how blocks were obtained. All are cumulative over the resolution.
    #[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ShiftStats {
        /// Signature steps that asked for a block.
        pub lookups: usize,
        /// Lookups served from the cache.
        pub hits: usize,
        /// Misses that built a full block and cached it.
        pub shared_builds: usize,
        /// Misses that built a private block (`delta == 1`) and did not cache it.
        pub private_builds: usize,
        /// Shared builds that found the key already present when inserting: two consumers raced.
        pub duplicate_builds: usize,
    }

    /// Lock-free tallies behind [`ShiftStats`].
    #[derive(Default)]
    struct Counters {
        lookups: AtomicUsize,
        hits: AtomicUsize,
        shared_builds: AtomicUsize,
        private_builds: AtomicUsize,
        duplicate_builds: AtomicUsize,
    }

    /// The per-resolution block cache.
    pub(super) struct ShiftCache {
        enabled: bool,
        /// Check every block a step uses against a direct build of that step's matrix.
        verify: bool,
        blocks: Mutex<FxHashMap<Key, Block>>,
        counters: Counters,
    }

    impl ShiftCache {
        /// An empty cache, enabled.
        pub fn new() -> Self {
            Self {
                enabled: true,
                verify: false,
                blocks: Mutex::new(FxHashMap::default()),
                counters: Counters::default(),
            }
        }

        /// Whether signature steps take their matrix from a block.
        pub fn enabled(&self) -> bool {
            self.enabled
        }

        /// See [`Self::enabled`].
        pub fn set_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }

        /// Whether every use is cross-checked against a direct build.
        pub fn verify(&self) -> bool {
            self.verify
        }

        /// See [`Self::verify`].
        pub fn set_verify(&mut self, verify: bool) {
            self.verify = verify;
        }

        /// The cached block for `key`, counting the lookup.
        pub fn lookup(&self, key: &Key) -> Option<Block> {
            self.counters.lookups.fetch_add(1, Ordering::Relaxed);
            let found = self.blocks.lock().unwrap().get(key).cloned();
            if found.is_some() {
                self.counters.hits.fetch_add(1, Ordering::Relaxed);
            }
            found
        }

        /// Record a private build, which is never cached.
        pub fn note_private_build(&self) {
            self.counters.private_builds.fetch_add(1, Ordering::Relaxed);
        }

        /// Cache a block a consumer built on a miss.
        pub fn insert_built(&self, key: Key, block: Block) {
            self.counters.shared_builds.fetch_add(1, Ordering::Relaxed);
            if self.blocks.lock().unwrap().insert(key, block).is_some() {
                self.counters
                    .duplicate_builds
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        /// Drop every block.
        pub fn clear(&self) {
            self.blocks.lock().unwrap().clear();
        }

        /// A snapshot of the counters.
        pub fn stats(&self) -> ShiftStats {
            let c = &self.counters;
            ShiftStats {
                lookups: c.lookups.load(Ordering::Relaxed),
                hits: c.hits.load(Ordering::Relaxed),
                shared_builds: c.shared_builds.load(Ordering::Relaxed),
                private_builds: c.private_builds.load(Ordering::Relaxed),
                duplicate_builds: c.duplicate_builds.load(Ordering::Relaxed),
            }
        }
    }
}

pub use shift::ShiftStats;

/// Full-width rows of `d_{s - 1}` on a signature's target basis, indexed by position in the
/// signature's target mask: all of them, or only those a step reads.
enum SigRows {
    All(Matrix),
    Some {
        /// Position in `rows` of each masked index, or `usize::MAX` where it was not built.
        pos: Vec<usize>,
        rows: Matrix,
    },
}

impl SigRows {
    /// The row for masked index `i`, which must have been built.
    fn row(&self, i: usize) -> FpSlice<'_> {
        match self {
            Self::All(m) => m.row(i),
            Self::Some { pos, rows } => {
                debug_assert_ne!(pos[i], usize::MAX, "row {i} was not built");
                rows.row(pos[i])
            }
        }
    }

    /// The width of every row.
    fn columns(&self) -> usize {
        match self {
            Self::All(m) => m.columns(),
            Self::Some { rows, .. } => rows.columns(),
        }
    }
}

/// A resolution of `S_2` using Nassau's algorithm.
///
/// This aims to have an API similar to that of
/// [`resolution::Resolution`](crate::resolution::Resolution). From an API point of view, the main
/// difference between the two is that this is a chain complex over [`MilnorAlgebra`] over
/// [`SteenrodAlgebra`](algebra::SteenrodAlgebra).
pub struct Resolution<M: ZeroModule<Algebra = MilnorAlgebra>> {
    lock: Mutex<()>,
    name: String,
    max_degree: i32,
    modules: OnceBiVec<Arc<FreeModule<MilnorAlgebra>>>,
    zero_module: Arc<FreeModule<MilnorAlgebra>>,
    differentials: OnceBiVec<Arc<FreeModuleHomomorphism<FreeModule<MilnorAlgebra>>>>,
    target: Arc<FiniteChainComplex<M>>,
    chain_maps: OnceBiVec<Arc<FreeModuleHomomorphism<M>>>,
    save_dir: SaveDirectory,
    shift: shift::ShiftCache,
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> Resolution<M> {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn new(module: Arc<M>) -> Self {
        Self::new_with_save(module, None).unwrap()
    }

    pub fn new_with_save(
        module: Arc<M>,
        save_dir: impl Into<SaveDirectory>,
    ) -> anyhow::Result<Self> {
        let save_dir = save_dir.into();
        let max_degree = module
            .max_degree()
            .ok_or_else(|| anyhow!("Nassau's algorithm requires bounded module"))?;
        let target = Arc::new(FiniteChainComplex::ccdz(module));

        if let Some(p) = save_dir.write() {
            for subdir in SaveKind::nassau_data() {
                subdir.create_dir(p)?;
            }
        }

        Ok(Self {
            lock: Mutex::new(()),
            zero_module: Arc::new(FreeModule::new(target.algebra(), "F_{-1}".to_string(), 0)),
            name: String::new(),
            modules: OnceBiVec::new(0),
            differentials: OnceBiVec::new(0),
            chain_maps: OnceBiVec::new(0),
            target,
            max_degree,
            save_dir,
            shift: shift::ShiftCache::new(),
        })
    }

    /// Whether nonzero signatures take their matrix from a shared zero-signature block (see
    /// [`ShiftStats`]). On by default; turning it off changes no result, only the work done.
    pub fn set_shift_reuse(&mut self, enabled: bool) {
        self.shift.set_enabled(enabled);
    }

    /// Cross-check every shared block a signature step uses against a direct build of that step's
    /// matrix, panicking on any difference. Slow; for tests.
    pub fn set_shift_verify(&mut self, verify: bool) {
        self.shift.set_verify(verify);
    }

    /// How the signature steps have obtained their blocks so far.
    pub fn shift_stats(&self) -> ShiftStats {
        self.shift.stats()
    }

    fn add_generators(&self, b: Bidegree, num_new_gens: usize) {
        let gen_names = (0..num_new_gens)
            .map(|idx| format!("x_{:#}", BidegreeGenerator::new(b, idx)))
            .collect();
        self.module(b.s())
            .add_generators(b.t(), num_new_gens, Some(gen_names));
    }

    /// This function prepares the Resolution object to perform computations up to the
    /// specified s degree. It does *not* perform any computations by itself. It simply lengthens
    /// the `OnceVec`s `modules`, `chain_maps`, etc. to the right length.
    fn extend_through_degree(&self, max_s: i32) {
        let min_degree = self.min_degree();

        self.modules.extend(max_s, |i| {
            Arc::new(FreeModule::new(
                Arc::clone(&self.algebra()),
                format!("F{i}"),
                min_degree,
            ))
        });

        self.differentials.extend(0, |_| {
            Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[0]),
                Arc::clone(&self.zero_module),
                0,
            ))
        });

        self.differentials.extend(max_s, |i| {
            Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[i]),
                Arc::clone(&self.modules[i - 1]),
                0,
            ))
        });

        self.chain_maps.extend(max_s, |i| {
            Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[i]),
                self.target.module(i),
                0,
            ))
        });
    }

    #[tracing::instrument(skip_all, fields(throughput))]
    fn write_qi(
        f: &mut Option<impl io::Write>,
        scratch: &mut FpVector,
        signature: &[PPartEntry],
        next_mask: &[usize],
        full_rows: &SigRows,
        masked_matrix: &AugmentedMatrix<2>,
    ) -> io::Result<()> {
        let f = match f {
            Some(f) => f,
            None => return Ok(()),
        };

        let mut own_f = LogWriter::new(f);
        let f = &mut own_f;

        let pivots = &masked_matrix.pivots()[0..masked_matrix.end[0]];
        if !pivots.iter().any(|&x| x >= 0) {
            return Ok(());
        }

        // Write signature if non-zero.
        if signature.iter().any(|&x| x > 0) {
            f.write_u64::<LittleEndian>(Magic::Signature as u64)?;
            MilnorSubalgebra::signature_to_bytes(signature, f)?;
        }

        // Write quasi-inverses
        for (col, &row) in pivots.iter().enumerate() {
            if row < 0 {
                continue;
            }
            f.write_u64::<LittleEndian>(next_mask[col] as u64)?;
            let preimage = masked_matrix.row_segment(row as usize, 1, 1);
            scratch.set_scratch_vector_size(preimage.len());
            scratch.as_slice_mut().assign(preimage);
            scratch.to_bytes(f)?;

            scratch.set_scratch_vector_size(full_rows.columns());
            for (i, _) in preimage.iter_nonzero() {
                scratch.as_slice_mut().add(full_rows.row(i), 1);
            }
            scratch.to_bytes(f)?;
        }

        tracing::Span::current().record(
            "throughput",
            tracing::field::display(own_f.into_throughput()),
        );
        Ok(())
    }

    fn write_differential(
        &self,
        b: Bidegree,
        num_new_gens: usize,
        target_dim: usize,
    ) -> anyhow::Result<()> {
        if let Some(dir) = self.save_dir.write() {
            let mut f = self
                .save_file(SaveKind::NassauDifferential, b)
                .create_file(dir.clone(), false);
            f.write_u64::<LittleEndian>(num_new_gens as u64)?;
            f.write_u64::<LittleEndian>(target_dim as u64)?;

            for n in 0..num_new_gens {
                self.differential(b.s()).output(b.t(), n).to_bytes(&mut f)?;
            }
        }
        Ok(())
    }

    /// Build the zero-signature block of `d_{s - 1}` in degree `d` (see [`shift`]), with columns
    /// from the generators of `C_{s - 2}` of degree `< col_bound`.
    ///
    /// Reads the generators of `C_{s - 1}` of degree `<= d` and those of `C_{s - 2}` of degree
    /// `< col_bound`, all of which must be registered.
    fn build_shift_block(
        &self,
        s: i32,
        d: i32,
        subalgebra: &MilnorSubalgebra,
        col_bound: i32,
    ) -> Matrix {
        let p = self.prime();
        let target = &*self.modules[s - 1];
        let next = &*self.modules[s - 2];
        let algebra = target.algebra();
        let zero_sig = subalgebra.zero_signature();

        let rows: Vec<usize> = subalgebra
            .signature_mask(&algebra, target, d, &zero_sig, Some(d + 1))
            .collect();
        let cols: Vec<usize> = subalgebra
            .signature_mask(&algebra, next, d, &zero_sig, Some(col_bound))
            .collect();
        // By minimality the image of a degree-`d` element involves only generators of degree `< d`,
        // so this width holds every row.
        let next_dim = next.dimension_from_gens_below(d, col_bound);

        let full = {
            let _guard = ParallelGuard::new();
            self.differentials[s - 1].get_partial_matrix_restricted(d, &rows, next_dim)
        };
        let mut block = Matrix::new(p, rows.len(), cols.len());
        block.as_slice_mut().add_masked(&full, &cols);
        block
    }

    /// The block the signature step for `signature` at `b` reads, from the cache or built on a
    /// miss. See [`shift`] for which builds may be shared.
    fn shift_block(
        &self,
        b: Bidegree,
        subalgebra: &MilnorSubalgebra,
        signature: &[PPartEntry],
    ) -> shift::Block {
        let delta = MilnorSubalgebra::signature_degree(signature);
        let d = b.t() - delta;
        let key = shift::Key {
            s: b.s(),
            degree: d,
            profile: subalgebra.profile.clone(),
        };
        if let Some(block) = self.shift.lookup(&key) {
            tracing::debug!(target: "nassau::shift", %b, d, hit = true, "shift block");
            return block;
        }
        tracing::debug!(target: "nassau::shift", %b, d, hit = false, "shift block");
        if delta >= 2 {
            let block = Arc::new(self.build_shift_block(b.s(), d, subalgebra, d + 1));
            self.shift.insert_built(key, Arc::clone(&block));
            block
        } else {
            self.shift.note_private_build();
            Arc::new(self.build_shift_block(b.s(), d, subalgebra, b.t() - 1))
        }
    }

    #[tracing::instrument(skip(self), fields(%b, %subalgebra, num_new_gens, density))]
    fn step_resolution_with_subalgebra(
        &self,
        b: Bidegree,
        subalgebra: MilnorSubalgebra,
    ) -> anyhow::Result<PendingRegistration> {
        // Takes the count rather than reading it back from the module: registration now happens in
        // a later phase, so `number_of_gens_in_bidegree` would still be zero here. `density` is
        // dropped for the same reason -- it reads the registered differential -- and is reported by
        // the registration phase instead.
        let end = |num_new_gens: usize| {
            tracing::Span::current().record("num_new_gens", num_new_gens);
        };

        let p = self.prime();
        let mut scratch = FpVector::new(p, 0);

        let target = &*self.modules[b.s() - 1];
        let algebra = target.algebra();

        // We compute this bidegree treating the target `C_{b.s() - 1}` as if it had no generators
        // of degree `>= b.t()`, and `C_{b.s() - 2}` as if it had none of degree `>= b.t() - 1`.
        // By minimality this loses no information (the differentials we care about land in the
        // radical, hence in strictly lower-degree generators), and it makes the computation depend
        // only on data that is frozen once `(b.s() - 1, b.t() - 1)` and `(b.s(), b.t() - 1)` have
        // been committed. This is what lets [`Self::compute_through_stem`] compute `(b.s(), b.t())`
        // concurrently with `(b.s() - 1, b.t())`, which is adding those degree-`b.t()` generators.
        let target_bound = b.t();
        let next_bound = b.t() - 1;

        let zero_sig = subalgebra.zero_signature();
        let target_dim = target.dimension_from_gens_below(b.t(), target_bound);
        let target_mask: Vec<usize> = subalgebra
            .signature_mask(&algebra, target, b.t(), &zero_sig, Some(target_bound))
            .collect();
        let target_masked_dim = target_mask.len();

        let next = &self.modules[b.s() - 2];
        next.compute_basis(b.t());
        let next_dim = next.dimension_from_gens_below(b.t(), next_bound);

        let mut f = if let Some(dir) = self.save_dir().write() {
            let mut f = self
                .save_file(SaveKind::NassauQi, b - Bidegree::s_t(1, 0))
                .create_file(dir.to_owned(), true);
            f.write_u64::<LittleEndian>(next_dim as u64)?;
            f.write_u64::<LittleEndian>(target_masked_dim as u64)?;
            subalgebra.to_bytes(&mut f)?;
            Some(f)
        } else {
            None
        };

        let guard = tracing::info_span!("step", signature = ?zero_sig).entered();
        let next_mask: Vec<usize> = subalgebra
            .signature_mask(&algebra, next, b.t(), &zero_sig, Some(next_bound))
            .collect();
        let next_masked_dim = next_mask.len();

        let full_matrix = {
            let _guard = ParallelGuard::new();
            self.differentials[b.s() - 1].get_partial_matrix_restricted(
                b.t(),
                &target_mask,
                next_dim,
            )
        };
        let mut masked_matrix =
            AugmentedMatrix::new(p, target_masked_dim, [next_masked_dim, target_masked_dim]);

        masked_matrix
            .segment(0, 0)
            .add_masked(&full_matrix, &next_mask);
        masked_matrix.segment(1, 1).add_identity();
        masked_matrix.row_reduce();
        let kernel = masked_matrix.compute_kernel();
        let full_matrix = SigRows::All(full_matrix);

        Self::write_qi(
            &mut f,
            &mut scratch,
            &zero_sig,
            &next_mask,
            &full_matrix,
            &masked_matrix,
        )?;

        // The quasi-inverse is always computed on the restricted (degree `< b.t()`) target basis,
        // so from the point of view of a later `apply_quasi_inverse` it was computed with
        // "incomplete information": the differentials on the degree-`b.t()` generators of the
        // target were not available. We flag this unconditionally so the lift is corrected using
        // those differentials once they are known.
        if let Some(f) = &mut f {
            f.write_u64::<LittleEndian>(Magic::Fix as u64)?;
        }

        // Compute image
        let mut n = subalgebra.signature_matrix(
            &self.differentials[b.s()],
            b.t(),
            &zero_sig,
            target_bound,
            Some(b.t() - (subalgebra.zero_signature_floor() - 1)),
        );
        n.row_reduce();
        let next_row = n.rows();

        let num_new_gens = n.extend_image(0, n.columns(), &kernel, 0).len();

        if b.t() < b.s() {
            assert_eq!(num_new_gens, 0, "Adding generators at {b}");
        }

        // NOT registered here: `add_generators` appends to `modules[b.s()]` in increasing degree,
        // and this bidegree may have been computed before its row predecessor. The count travels
        // back to the scheduler in `PendingRegistration` instead. Nothing between here and the end
        // of the signature loop reads `modules[b.s()]` -- the loop works on rows `s-1` and `s-2`.
        let mut xs = vec![FpVector::new(p, target_dim); num_new_gens];
        let mut dxs = vec![FpVector::new(p, next_dim); num_new_gens];

        for ((x, x_masked), dx) in xs
            .iter_mut()
            .zip_eq(n.iter().skip(next_row))
            .zip_eq(&mut dxs)
        {
            x.as_slice_mut().add_unmasked(x_masked, 1, &target_mask);
            for (i, _) in x_masked.iter_nonzero() {
                dx.as_slice_mut().add(full_matrix.row(i), 1);
            }
        }

        // Now add correction terms
        let mut target_mask: Vec<usize> = Vec::new();
        let mut next_mask: Vec<usize> = Vec::new();

        drop(guard);

        for signature in subalgebra.iter_signatures(b.t()) {
            let _guard = tracing::info_span!("step", ?signature).entered();
            target_mask.clear();
            next_mask.clear();
            target_mask.extend(subalgebra.signature_mask(
                &algebra,
                target,
                b.t(),
                &signature,
                Some(target_bound),
            ));
            next_mask.extend(subalgebra.signature_mask(
                &algebra,
                next,
                b.t(),
                &signature,
                Some(next_bound),
            ));

            let mut masked_matrix =
                AugmentedMatrix::new(p, target_mask.len(), [next_mask.len(), target_mask.len()]);

            // With shift reuse the masked matrix comes from a block and only the full-width rows
            // the lift reads are built, after the reduction says which those are.
            let full_matrix = if self.shift.enabled() && !target_mask.is_empty() {
                let block = self.shift_block(b, &subalgebra, &signature);
                assert_eq!(
                    block.rows(),
                    target_mask.len(),
                    "block rows at {b} {signature:?}"
                );
                assert!(
                    block.columns() >= next_mask.len(),
                    "block narrower than its consumer at {b} {signature:?}"
                );
                if self.shift.verify() {
                    let direct = self.differentials[b.s() - 1].get_partial_matrix_restricted(
                        b.t(),
                        &target_mask,
                        next_dim,
                    );
                    for (i, row) in direct.iter().enumerate() {
                        let mut expected = FpVector::new(p, next_mask.len());
                        expected.as_slice_mut().add_masked(row, 1, &next_mask);
                        let mut actual = FpVector::new(p, next_mask.len());
                        actual
                            .as_slice_mut()
                            .add(block.row(i).restrict(0, next_mask.len()), 1);
                        assert_eq!(
                            expected, actual,
                            "shift block row {i} differs from the direct build at {b} \
                             {signature:?}"
                        );
                    }
                }
                let mut segment = masked_matrix.segment(0, 0);
                for (i, mut row) in segment.iter_mut().enumerate() {
                    row.add(block.row(i).restrict(0, next_mask.len()), 1);
                }
                None
            } else {
                let full_matrix = {
                    let _guard = ParallelGuard::new();
                    self.differentials[b.s() - 1].get_partial_matrix_restricted(
                        b.t(),
                        &target_mask,
                        next_dim,
                    )
                };
                masked_matrix
                    .segment(0, 0)
                    .add_masked(&full_matrix, &next_mask);
                Some(full_matrix)
            };
            masked_matrix.segment(1, 1).add_identity();
            masked_matrix.row_reduce();

            let qi = masked_matrix.compute_quasi_inverse();
            let pivots = qi.pivots().unwrap();
            let preimage = qi.preimage();

            // Each generator's correction, in masked target coordinates. Generators are
            // independent -- each reads only its own `dx` -- so every correction can be found
            // before any is applied.
            let corrections: Vec<FpVector> = dxs
                .iter()
                .map(|dx| {
                    let mut correction = FpVector::new(p, target_mask.len());
                    let mut row = 0;
                    for (i, &v) in next_mask.iter().enumerate() {
                        if pivots[i] < 0 {
                            continue;
                        }
                        if dx.entry(v) != 0 {
                            correction.as_slice_mut().add(preimage.row(row), 1);
                        }
                        row += 1;
                    }
                    correction
                })
                .collect();

            let full_matrix = match full_matrix {
                Some(m) => SigRows::All(m),
                None => {
                    // The rows the corrections touch, and those the quasi-inverse save file maps
                    // through.
                    let mut wanted = vec![false; target_mask.len()];
                    for correction in &corrections {
                        for (i, _) in correction.iter_nonzero() {
                            wanted[i] = true;
                        }
                    }
                    if f.is_some() {
                        let end = masked_matrix.end[0];
                        for &row in masked_matrix.pivots()[0..end].iter().filter(|&&r| r >= 0) {
                            for (i, _) in
                                masked_matrix.row_segment(row as usize, 1, 1).iter_nonzero()
                            {
                                wanted[i] = true;
                            }
                        }
                    }
                    let mut pos = vec![usize::MAX; target_mask.len()];
                    let mut inputs = Vec::new();
                    for (i, _) in wanted.iter().enumerate().filter(|(_, w)| **w) {
                        pos[i] = inputs.len();
                        inputs.push(target_mask[i]);
                    }
                    let rows = {
                        let _guard = ParallelGuard::new();
                        self.differentials[b.s() - 1].get_partial_matrix_restricted(
                            b.t(),
                            &inputs,
                            next_dim,
                        )
                    };
                    SigRows::Some { pos, rows }
                }
            };

            for ((x, dx), correction) in xs.iter_mut().zip(&mut dxs).zip(&corrections) {
                for (i, _) in correction.iter_nonzero() {
                    x.add_basis_element(target_mask[i], 1);
                    dx.as_slice_mut().add(full_matrix.row(i), 1);
                }
            }
            Self::write_qi(
                &mut f,
                &mut scratch,
                &signature,
                &next_mask,
                &full_matrix,
                &masked_matrix,
            )?;
        }
        for dx in &dxs {
            assert!(dx.is_zero(), "dx non-zero at {b}");
        }
        end(num_new_gens);

        if let Some(f) = &mut f {
            f.write_u64::<LittleEndian>(Magic::End as u64)?;
        }

        Ok(PendingRegistration {
            b,
            num_new_gens,
            rows: xs,
            target_dim,
            write_save: true,
            extend_chain_map: true,
        })
    }

    /// Step resolution for s = 0
    #[tracing::instrument(skip(self))]
    fn step0(&self, t: i32) {
        self.zero_module.extend_by_zero(t);

        let source_module = &self.modules[0];
        let target_module = self.target.module(0);

        let chain_map = &self.chain_maps[0];
        let d = &self.differentials[0];

        let source_dim = source_module.dimension(t);
        let target_dim = target_module.dimension(t);

        source_module.compute_basis(t);
        target_module.compute_basis(t);

        if target_dim == 0 {
            source_module.extend_by_zero(t);
            chain_map.extend_by_zero(t);
        } else {
            let mut matrix = AugmentedMatrix::<2>::new_with_capacity(
                self.prime(),
                source_dim,
                &[target_dim, source_dim],
                source_dim + target_dim,
                0,
            );
            {
                let _guard = ParallelGuard::new();
                chain_map.get_matrix(matrix.segment(0, 0), t);
            }
            matrix.segment(1, 1).add_identity();

            matrix.row_reduce();

            let num_new_gens = matrix.extend_to_surjection(0, target_dim, 0).len();

            self.add_generators(Bidegree::s_t(0, t), num_new_gens);

            chain_map.add_generators_from_matrix_rows(
                t,
                matrix
                    .segment(0, 0)
                    .row_slice(source_dim, source_dim + num_new_gens),
            );
        }
        chain_map.compute_auxiliary_data_through_degree(t);

        d.set_kernel(t, None);
        d.set_image(t, None);
        d.set_quasi_inverse(t, None);
        d.extend_by_zero(t);
    }

    /// Step resolution for s = 1
    #[tracing::instrument(skip(self))]
    fn step1(&self, t: i32) -> anyhow::Result<()> {
        let p = self.prime();

        let source_module = &self.modules[1];
        let target_module = &self.modules[0];
        let cc_module = self.target.module(0);

        let source_dim = source_module.dimension(t);
        let target_dim = target_module.dimension(t);

        let mut matrix =
            AugmentedMatrix::<2>::new(p, target_dim, [cc_module.dimension(t), target_dim]);
        {
            let _guard = ParallelGuard::new();
            self.chain_maps[0].get_matrix(matrix.segment(0, 0), t);
        }
        matrix.segment(1, 1).add_identity();
        matrix.row_reduce();
        let desired_image = matrix.compute_kernel();

        let mut matrix = AugmentedMatrix::<2>::new_with_capacity(
            p,
            source_dim,
            &[target_dim, source_dim],
            source_dim + MAX_NEW_GENS,
            0,
        );
        {
            let _guard = ParallelGuard::new();
            self.differentials[1].get_matrix(matrix.segment(0, 0), t);
        }
        matrix.segment(1, 1).add_identity();
        matrix.row_reduce();

        let num_new_gens = matrix.extend_image(0, target_dim, &desired_image, 0).len();

        self.add_generators(Bidegree::s_t(1, t), num_new_gens);

        self.differentials[1].add_generators_from_matrix_rows(
            t,
            matrix
                .segment(0, 0)
                .row_slice(source_dim, source_dim + num_new_gens),
        );

        self.write_differential(Bidegree::s_t(1, t), num_new_gens, target_dim)?;
        Ok(())
    }

    /// Compute `b`, returning what still has to be registered.
    ///
    /// `None` means the bidegree registered itself: rows 0 and 1 keep the strict schedule, so their
    /// row order is already guaranteed and there is nothing for the scheduler to sequence.
    fn step_resolution_with_result(
        &self,
        b: Bidegree,
    ) -> anyhow::Result<Option<PendingRegistration>> {
        let p = self.prime();
        let set_data = || {
            let d = &self.differentials[b.s()];
            let c = &self.chain_maps[b.s()];

            d.set_kernel(b.t(), None);
            d.set_image(b.t(), None);
            d.set_quasi_inverse(b.t(), None);

            c.set_kernel(b.t(), None);
            c.set_image(b.t(), None);
            c.set_quasi_inverse(b.t(), None);
        };
        self.modules[b.s()].compute_basis(b.t());
        if b.s() > 0 {
            self.modules[b.s() - 1].compute_basis(b.t());
        }

        if b.s() == 0 {
            self.step0(b.t());
            return Ok(None);
        }

        if let Some(dir) = self.save_dir.read()
            && let Some(mut f) = self
                .save_file(SaveKind::NassauDifferential, b)
                .open_file(dir.clone())
        {
            tracing::info!(%b, "Loading differential");

            let num_new_gens = f.read_u64::<LittleEndian>()? as usize;
            // This need not be equal to `target_res_dimension`. If we saved a big resolution
            // and now only want to load up to a small stem, then `target_res_dimension` will
            // be smaller. If we have previously saved a small resolution up to a stem and now
            // want to resolve further, it will be bigger.
            let saved_target_res_dimension = f.read_u64::<LittleEndian>()? as usize;

            let mut d_targets = Vec::with_capacity(num_new_gens);

            for _ in 0..num_new_gens {
                d_targets.push(FpVector::from_bytes(p, saved_target_res_dimension, &mut f)?);
            }

            return Ok(Some(PendingRegistration {
                b,
                num_new_gens,
                rows: d_targets,
                target_dim: saved_target_res_dimension,
                // Read from a save file; rewriting it would be pointless work.
                write_save: false,
                extend_chain_map: false,
            }));
        }

        if b.s() == 1 {
            self.step1(b.t())?;
            set_data();
            return Ok(None);
        }

        let pending = self.step_resolution_with_subalgebra(
            b,
            MilnorSubalgebra::optimal_for(b - Bidegree::s_t(0, self.max_degree)),
        )?;
        Ok(Some(pending))
    }

    /// Apply a computed bidegree's registration.
    ///
    /// Everything here appends per degree -- `modules[s]`, the differential's outputs, the chain
    /// map, and the per-degree subspace caches -- so it must run in `t` order within a row. The
    /// scheduler guarantees that by ordering the `Register` nodes; nothing here waits on a lock.
    fn register(&self, pending: PendingRegistration) -> anyhow::Result<()> {
        let PendingRegistration {
            b,
            num_new_gens,
            rows,
            target_dim,
            write_save,
            extend_chain_map,
        } = pending;

        self.add_generators(b, num_new_gens);
        self.differentials[b.s()].add_generators_from_rows(b.t(), rows);

        if write_save {
            self.write_differential(b, num_new_gens, target_dim)?;
        }
        if extend_chain_map {
            self.chain_maps[b.s()].extend_by_zero(b.t());
        }

        // `density` used to be recorded as a span field on the compute span; it reads the
        // registered differential, so it belongs here now and is emitted as an event instead.
        tracing::info!(
            %b,
            num_new_gens,
            density = self.differentials[b.s()].differential_density(b.t()) * 100.0,
            "registered"
        );

        let d = &self.differentials[b.s()];
        let c = &self.chain_maps[b.s()];
        d.set_kernel(b.t(), None);
        d.set_image(b.t(), None);
        d.set_quasi_inverse(b.t(), None);
        c.set_kernel(b.t(), None);
        c.set_image(b.t(), None);
        c.set_quasi_inverse(b.t(), None);
        Ok(())
    }

    /// [`Self::step_resolution_with_result`], panicking rather than returning the error.
    fn step_resolution(&self, b: Bidegree) -> Option<PendingRegistration> {
        self.step_resolution_with_result(b)
            .unwrap_or_else(|e| panic!("Error computing bidegree {b}: {e}"))
    }

    /// This function resolves up till a fixed stem instead of a fixed t.
    ///
    /// The dependency graph is built explicitly, with each bidegree split into a `Compute` node and
    /// a `Register` node; see `depgraph` for why. `Compute` runs on a worker and returns what has
    /// to be registered; `Register` is applied by this function, in graph order, so appends to
    /// `modules[s]` and `differentials[s]` stay in increasing degree without any worker ever
    /// blocking on its row predecessor.
    #[tracing::instrument(skip(self), fields(self = self.name, %max))]
    pub fn compute_through_stem(&self, max: Bidegree) {
        use depgraph::{Node, Phase};

        let _lock = self.lock.lock();

        self.extend_through_degree(max.s());
        self.algebra().compute_basis(max.t());

        // How far back in its own row `b` actually reads.
        //
        // The image is built at the ZERO signature, and no operation below the subalgebra's
        // zero-signature floor carries it, so generators within that many degrees contribute
        // nothing to the image and need not be registered yet. `signature_matrix` is passed the
        // matching bound, so the two agree by construction.
        //
        // Rows 0 and 1 keep the strict schedule: `step0` and `step1` read their targets through
        // full matrices rather than the signature-masked image, so this reasoning does not apply to
        // them.
        let same_row_dep = |b: Bidegree| {
            if b.s() <= 1 {
                return b - Bidegree::s_t(0, 1);
            }
            let subalgebra = MilnorSubalgebra::optimal_for(b - Bidegree::s_t(0, self.max_degree));
            b - Bidegree::s_t(0, subalgebra.zero_signature_floor())
        };

        let mut graph = depgraph::Graph::new(self.min_degree(), max, same_row_dep);

        let tracing_span = tracing::Span::current();
        maybe_rayon::in_place_scope(|scope| {
            let _tracing_guard = tracing_span.enter();

            let (sender, receiver) = mpsc::channel();

            let spawn_compute = |b: Bidegree, sender: mpsc::Sender<SenderData>| {
                if self.has_computed_bidegree(b) {
                    // Already present, so there is nothing to compute and nothing to register. It
                    // still travels the normal completion path so its successors are released in
                    // the one place that does that.
                    SenderData::send(b, None, sender);
                } else {
                    let tracing_span = tracing_span.clone();
                    scope.spawn(move |_| {
                        let _tracing_guard = tracing_span.enter();
                        if crate::utils::parallel::is_in_parallel() {
                            SenderData::send_retry(b, sender);
                            return;
                        }
                        let pending = self.step_resolution(b);
                        SenderData::send(b, pending, sender);
                    });
                }
            };

            // A computed bidegree's registration, waiting for its `Register` node to come up.
            let mut payloads: HashMap<Bidegree, Option<PendingRegistration>> = HashMap::new();
            let mut in_flight = 0usize;

            loop {
                // Dispatch everything the graph has freed. Running a `Register` can free more, so
                // this drains rather than taking one pass.
                while let Some(node) = graph.pop_ready() {
                    let b = node.b;
                    match node.phase {
                        Phase::Compute => {
                            in_flight += 1;
                            spawn_compute(b, sender.clone());
                        }
                        Phase::Register => {
                            if let Some(pending) = payloads.remove(&b).flatten() {
                                self.register(pending).unwrap_or_else(|e| {
                                    panic!("Error registering bidegree {b}: {e}")
                                });
                            }
                            graph.complete(node);
                        }
                    }
                }

                if in_flight == 0 {
                    break;
                }

                let Ok(SenderData {
                    b,
                    retry,
                    pending,
                    sender,
                }) = receiver.recv()
                else {
                    break;
                };
                if retry {
                    // Bounced off a worker already inside a parallel section. Still in flight; hand
                    // it back out without touching the graph.
                    spawn_compute(b, sender);
                    continue;
                }
                in_flight -= 1;
                payloads.insert(b, pending);
                graph.complete(Node::compute(b));
            }

            // A stalled graph is an edge bug, not a slow run.
            let stuck = graph.undispatched();
            assert!(
                stuck.is_empty(),
                "dependency graph stalled: {} nodes never dispatched, first {:?}",
                stuck.len(),
                &stuck[..stuck.len().min(5)]
            );
        });
        self.shift.clear();
    }
}

mod depgraph {
    use sseq::coordinates::Bidegree;

    /// Which half of a bidegree's work a [`Node`] stands for.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Phase {
        Compute,
        Register,
    }

    /// One half of a bidegree's work.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct Node {
        pub phase: Phase,
        pub b: Bidegree,
    }

    impl Node {
        /// The node that computes `b` without registering it.
        pub fn compute(b: Bidegree) -> Self {
            Self {
                phase: Phase::Compute,
                b,
            }
        }

        /// The node that appends `b`'s computed generators and differential.
        pub fn register(b: Bidegree) -> Self {
            Self {
                phase: Phase::Register,
                b,
            }
        }
    }

    /// One degree along a row.
    const STEP: Bidegree = Bidegree::s_t(0, 1);

    /// The bidegree in the row below whose registration `Compute(b)` reads.
    ///
    /// Row 1 reads `(0, t)` because `step1` reads its target through a full matrix; later rows need
    /// only the relaxed diagonal, see `step_resolution_with_subalgebra`. Row 0 lands outside the
    /// region, so it has no such edge.
    fn below(b: Bidegree) -> Bidegree {
        if b.s() == 1 {
            b - Bidegree::s_t(1, 0)
        } else {
            b - Bidegree::s_t(1, 1)
        }
    }

    /// The inverse of [`below`].
    fn above(b: Bidegree) -> Bidegree {
        if b.s() == 0 {
            b + Bidegree::s_t(1, 0)
        } else {
            b + Bidegree::s_t(1, 1)
        }
    }

    /// The dependency graph for [`super::Resolution::compute_through_stem`].
    ///
    /// Each bidegree is TWO nodes, because its two halves have different dependencies:
    ///
    /// * `Compute(b)` does the expensive work. It reads the rows below, so it needs those
    ///   registered -- but of its OWN row it needs only what the image computation reads.
    /// * `Register(b)` appends to `modules[s]` and `differentials[s]`, which are append-only in
    ///   increasing degree, so it needs `Register(b - STEP)`.
    ///
    /// Splitting them is what lets a row compute out of order while still registering in order. The
    /// scheduler dispatches `Compute` to workers and runs `Register` itself, so a node is only ever
    /// handed out when it can run immediately -- nothing blocks a worker waiting for its
    /// predecessor.
    ///
    /// A node becomes ready when its count of incomplete predecessors reaches zero, so it leaves
    /// the blocked set exactly once however many predecessors release it.
    ///
    /// # Representation
    ///
    /// Every node has at most two predecessors, all in the region:
    ///
    /// ```text
    /// Compute(b)   <- Register(same_row_dep(b)), Register(below(b))
    /// Register(b)  <- Compute(b),                Register(b - STEP)
    /// ```
    ///
    /// Every row of the region is one contiguous run of `t`, so a node's slot is arithmetic rather
    /// than a hash key, and the edges are recomputed from these rules rather than stored. The only
    /// per-node state is a predecessor count and a dispatched flag. `same_row_dep` is kept because
    /// it is the one input the rules cannot recompute cheaply.
    pub struct Graph {
        min_degree: i32,
        max: Bidegree,
        /// Prefix sums of the row lengths, so `idx` is a single add.
        row_offset: Vec<usize>,
        /// The `t` of `same_row_dep`, indexed by `idx`.
        same_row: Vec<i32>,
        /// The widest `t - same_row[..]`, which bounds the scan in [`Self::successors`].
        max_gap: i32,
        /// Predecessors not yet complete, indexed by slot.
        blocked: Vec<u32>,
        ready: Vec<u32>,
        dispatched: Vec<bool>,
        /// Reused by [`Self::complete`] so releasing successors never allocates.
        succ_buf: Vec<Node>,
    }

    impl Graph {
        /// Build the graph for bidegrees with `t >= min_degree` through stem `max.n()` and
        /// homological degree `max.s()`, and prime the ready queue.
        ///
        /// `same_row_dep(b)` is the earliest bidegree in `b`'s row that `Compute(b)` reads. One
        /// outside the region imposes no edge, which is what makes the base of each row a source.
        pub fn new(
            min_degree: i32,
            max: Bidegree,
            same_row_dep: impl Fn(Bidegree) -> Bidegree,
        ) -> Self {
            let mut row_offset = Vec::with_capacity(max.s() as usize + 2);
            let mut total = 0usize;
            for s in 0..=max.s() {
                row_offset.push(total);
                total += (max.n() + s - min_degree + 1).max(0) as usize;
            }
            row_offset.push(total);

            let mut same_row = Vec::with_capacity(total);
            let mut max_gap = 1;
            for s in 0..=max.s() {
                for t in min_degree..=(max.n() + s) {
                    let b = Bidegree::s_t(s, t);
                    let dep = same_row_dep(b);
                    assert!(
                        dep.s() == s && dep.t() < t,
                        "same-row dependency {dep} of {b} is not earlier in its row"
                    );
                    max_gap = max_gap.max(t - dep.t());
                    same_row.push(dep.t());
                }
            }

            let mut g = Self {
                min_degree,
                max,
                row_offset,
                same_row,
                max_gap,
                blocked: Vec::new(),
                ready: Vec::new(),
                dispatched: vec![false; 2 * total],
                succ_buf: Vec::new(),
            };
            g.blocked = (0..2 * total)
                .map(|slot| g.predecessors(g.node_at(slot)).count() as u32)
                .collect();
            // Descending, so `pop` hands out ascending slots and two runs are diffable.
            g.ready = (0..2 * total)
                .rev()
                .filter(|&slot| g.blocked[slot] == 0)
                .map(|slot| slot as u32)
                .collect();
            g
        }

        /// Whether `b` is one of the bidegrees this graph schedules.
        fn in_region(&self, b: Bidegree) -> bool {
            (0..=self.max.s()).contains(&b.s()) && b.t() >= self.min_degree && b.n() <= self.max.n()
        }

        /// The position of `b` in the row-major order of the region.
        fn idx(&self, b: Bidegree) -> usize {
            self.row_offset[b.s() as usize] + (b.t() - self.min_degree) as usize
        }

        /// The position of `n` in the per-node arrays, which interleave the two phases.
        fn slot(&self, n: Node) -> usize {
            2 * self.idx(n.b) + usize::from(n.phase == Phase::Register)
        }

        /// The inverse of [`Self::slot`].
        fn node_at(&self, slot: usize) -> Node {
            let idx = slot / 2;
            // The row is the last one starting at or before `idx`.
            let s = self.row_offset.partition_point(|&o| o <= idx) - 1;
            let b = Bidegree::s_t(
                s as i32,
                self.min_degree + (idx - self.row_offset[s]) as i32,
            );
            if slot.is_multiple_of(2) {
                Node::compute(b)
            } else {
                Node::register(b)
            }
        }

        /// The earliest bidegree in `b`'s row that `Compute(b)` reads.
        fn same_row_dep(&self, b: Bidegree) -> Bidegree {
            Bidegree::s_t(b.s(), self.same_row[self.idx(b)])
        }

        /// The nodes that must complete before `n` can run.
        fn predecessors(&self, n: Node) -> impl Iterator<Item = Node> {
            let b = n.b;
            let candidates = match n.phase {
                Phase::Compute => [
                    Node::register(self.same_row_dep(b)),
                    Node::register(below(b)),
                ],
                Phase::Register => [Node::compute(b), Node::register(b - STEP)],
            };
            candidates.into_iter().filter(|p| self.in_region(p.b))
        }

        /// The nodes `n` blocks: the inverse of [`Self::predecessors`].
        fn successors(&self, n: Node, out: &mut Vec<Node>) {
            out.clear();
            let b = n.b;
            match n.phase {
                Phase::Compute => out.push(Node::register(b)),
                Phase::Register => {
                    out.push(Node::register(b + STEP));
                    out.push(Node::compute(above(b)));
                    // The same-row consumers are within `max_gap`, so this is a short scan rather
                    // than a stored edge list.
                    out.extend(
                        (1..=self.max_gap)
                            .map(|k| b + Bidegree::s_t(0, k))
                            .filter(|&c| self.in_region(c) && self.same_row_dep(c) == b)
                            .map(Node::compute),
                    );
                    out.retain(|c| self.in_region(c.b));
                }
            }
        }

        /// The next node whose dependencies are all complete, or `None` if there is none right
        /// now. A node is handed out at most once, however many predecessors freed it.
        pub fn pop_ready(&mut self) -> Option<Node> {
            while let Some(slot) = self.ready.pop() {
                let slot = slot as usize;
                if !self.dispatched[slot] {
                    self.dispatched[slot] = true;
                    return Some(self.node_at(slot));
                }
            }
            None
        }

        /// Mark `n` complete, moving anything it was blocking into the ready queue.
        ///
        /// This is the ONLY way a node becomes ready, so "reports completion" and "releases
        /// successors" cannot diverge.
        pub fn complete(&mut self, n: Node) {
            let mut buf = std::mem::take(&mut self.succ_buf);
            self.successors(n, &mut buf);
            for &d in &buf {
                let slot = self.slot(d);
                debug_assert!(self.blocked[slot] > 0, "releasing {d:?} twice");
                self.blocked[slot] -= 1;
                if self.blocked[slot] == 0 {
                    self.ready.push(slot as u32);
                }
            }
            self.succ_buf = buf;
        }

        /// Nodes never dispatched. Non-empty at the end means the edges are wrong; report it rather
        /// than exiting quietly with a partial resolution.
        pub fn undispatched(&self) -> Vec<Node> {
            (0..self.dispatched.len())
                .filter(|&slot| !self.dispatched[slot])
                .map(|slot| self.node_at(slot))
                .collect()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A graph whose same-row reads reach back an irregular distance, like the per-subalgebra
        /// floors do.
        fn graph() -> Graph {
            Graph::new(0, Bidegree::n_s(12, 6), |b| {
                let gap = if b.s() <= 1 { 1 } else { 1 + (b.t() * 7) % 5 };
                b - Bidegree::s_t(0, gap)
            })
        }

        /// Every successor edge is a predecessor edge, and the reverse.
        #[test]
        fn successors_invert_predecessors() {
            let g = graph();
            let nodes: Vec<Node> = (0..g.blocked.len()).map(|slot| g.node_at(slot)).collect();
            let mut succ = Vec::new();
            for &n in &nodes {
                assert_eq!(g.node_at(g.slot(n)), n);
                g.successors(n, &mut succ);
                for &m in &succ {
                    assert!(g.predecessors(m).any(|p| p == n), "{n:?} -> {m:?}");
                }
                for p in g.predecessors(n) {
                    g.successors(p, &mut succ);
                    assert!(succ.contains(&n), "{p:?} -> {n:?} missing");
                }
            }
        }

        /// Draining the graph dispatches every node and registers each row in increasing `t`.
        #[test]
        fn drains_registering_rows_in_order() {
            let mut g = graph();
            let mut registered = vec![g.min_degree - 1; g.max.s() as usize + 1];
            // Completing the newest node first stresses the ordering more than a FIFO would.
            let mut running = Vec::new();
            loop {
                while let Some(n) = g.pop_ready() {
                    running.push(n);
                }
                let Some(n) = running.pop() else { break };
                if n.phase == Phase::Register {
                    let last = &mut registered[n.b.s() as usize];
                    assert_eq!(*last + 1, n.b.t(), "{n:?} registered out of order");
                    *last = n.b.t();
                }
                g.complete(n);
            }
            assert!(g.undispatched().is_empty());
        }
    }
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> ChainComplex for Resolution<M> {
    type Algebra = MilnorAlgebra;
    type Homomorphism = FreeModuleHomomorphism<FreeModule<Self::Algebra>>;
    type Module = FreeModule<Self::Algebra>;

    fn prime(&self) -> ValidPrime {
        TWO
    }

    fn algebra(&self) -> Arc<Self::Algebra> {
        self.zero_module.algebra()
    }

    fn module(&self, s: i32) -> Arc<Self::Module> {
        Arc::clone(&self.modules[s])
    }

    fn zero_module(&self) -> Arc<Self::Module> {
        Arc::clone(&self.zero_module)
    }

    fn min_degree(&self) -> i32 {
        0
    }

    fn has_computed_bidegree(&self, b: Bidegree) -> bool {
        self.differentials.len() > b.s() && self.differential(b.s()).next_degree() > b.t()
    }

    fn differential(&self, s: i32) -> Arc<Self::Homomorphism> {
        Arc::clone(&self.differentials[s])
    }

    #[tracing::instrument(skip(self), fields(self = self.name, %max))]
    fn compute_through_bidegree(&self, max: Bidegree) {
        let _lock = self.lock.lock();

        self.extend_through_degree(max.s());
        self.algebra().compute_basis(max.t());

        for t in 0..=max.t() {
            for s in 0..=max.s() {
                let b = Bidegree::s_t(s, t);
                if self.has_computed_bidegree(b) {
                    continue;
                }
                // This walks `t` then `s`, so registering each bidegree as it is computed is
                // already in order. Dropping the returned registration would leave the generators
                // unadded and every later bidegree reading a differential that is not there.
                if let Some(pending) = self.step_resolution(b) {
                    self.register(pending)
                        .unwrap_or_else(|e| panic!("Error registering bidegree {b}: {e}"));
                }
            }
        }
        self.shift.clear();
    }

    fn next_homological_degree(&self) -> i32 {
        self.modules.len()
    }

    fn save_dir(&self) -> &SaveDirectory {
        &self.save_dir
    }

    fn apply_quasi_inverse<T, S>(&self, results: &mut [T], b: Bidegree, inputs: &[S]) -> bool
    where
        for<'a> &'a mut T: Into<FpSliceMut<'a>>,
        for<'a> &'a S: Into<FpSlice<'a>>,
    {
        let mut f = if let Some(dir) = self.save_dir.read() {
            if let Some(f) = self.save_file(SaveKind::NassauQi, b).open_file(dir.clone()) {
                f
            } else {
                return false;
            }
        } else {
            return false;
        };

        let p = self.prime();

        let target_dim = f.read_u64::<LittleEndian>().unwrap() as usize;
        let zero_mask_dim = f.read_u64::<LittleEndian>().unwrap() as usize;
        let subalgebra = MilnorSubalgebra::from_bytes(&mut f).unwrap();
        let source = &self.modules[b.s()];
        let target = &self.modules[b.s() - 1];
        let algebra = target.algebra();

        let mut inputs: Vec<FpVector> = inputs.iter().map(|x| x.into().to_owned()).collect();
        let mut mask: Vec<usize> = Vec::with_capacity(zero_mask_dim + 8);
        mask.extend(subalgebra.signature_mask(
            &algebra,
            source,
            b.t(),
            &subalgebra.zero_signature(),
            None,
        ));

        let mut scratch0 = FpVector::new(p, zero_mask_dim);
        let mut scratch1 = FpVector::new(p, target_dim);

        // If the quasi-inverse was computed using incomplete information, we need to figure out
        // what the differentials in this bidegree hit and use them to lift. these variables are
        // trivial if there is no such problem.
        //
        // target_zero_mask is the signature mask of the target under the zero signature.
        //
        // dx_matrix is an AugmentedMatrix::<3>.
        //
        // Each row of this matrix is of the form [r; dx; x], where x is an element of the source
        // of signature zero, expressed in the masked basis, and dx is the value of the
        // differential on x. Then r is the entries of dx that have zero signature, which we
        // include so that the rref of the matix is nice. In practice, we keep r empty until the
        // very end, and then populate it manually.
        //
        // At the beginning the x's will be the new generators in this bidegree. As we read in the
        // quasi-inverses for the zero signature, we keep on reducing this so that dx is zero in
        // the pivot columns of the quasi-inverse. We can then use (the rref of) this matrix to
        // lift remaining elements with zero signature.
        let (mut target_zero_mask, mut dx_matrix) = if zero_mask_dim != mask.len() {
            let num_new_gens = source.number_of_gens_in_degree(b.t());
            assert_eq!(mask.len(), zero_mask_dim + num_new_gens);

            let target_zero_mask: Vec<usize> = subalgebra
                .signature_mask(&algebra, target, b.t(), &subalgebra.zero_signature(), None)
                .collect();
            let mut matrix = AugmentedMatrix::<3>::new(
                p,
                num_new_gens,
                [target_zero_mask.len(), target.dimension(b.t()), mask.len()],
            );

            for i in 0..num_new_gens {
                let dx = self.differentials[b.s()].output(b.t(), i);
                matrix
                    .row_segment_mut(i, 1, 1)
                    .slice_mut(0, dx.len())
                    .add(dx.as_slice(), 1);
                matrix
                    .row_segment_mut(i, 2, 2)
                    .add_basis_element(zero_mask_dim + i, 1);
            }

            (target_zero_mask, matrix)
        } else {
            (Vec::new(), AugmentedMatrix::<3>::new(p, 0, [0, 0, 0]))
        };

        loop {
            let col = f.read_u64::<LittleEndian>().unwrap() as usize;
            if col == Magic::End as usize {
                break;
            } else if col == Magic::Signature as usize {
                let signature = subalgebra.signature_from_bytes(&mut f).unwrap();

                mask.clear();
                mask.extend(subalgebra.signature_mask(&algebra, source, b.t(), &signature, None));
                scratch0.set_scratch_vector_size(mask.len());
            } else if col == Magic::Fix as usize {
                // We need to fix the differential problem
                //
                // First manually add_masked the second segment to the first, which we use for
                // row reduction. We do this manually for borrow checker reasons.
                for (j, &k) in target_zero_mask.iter().enumerate() {
                    for i in 0..dx_matrix.rows() {
                        if dx_matrix.row_segment(i, 1, 1).entry(k) != 0 {
                            dx_matrix.row_segment_mut(i, 0, 0).add_basis_element(j, 1);
                        }
                    }
                }
                dx_matrix.row_reduce();

                // Now reduce by these elements
                for i in 0..dx_matrix.rows() {
                    let masked_col = dx_matrix.row(i).first_nonzero().unwrap().0;
                    assert_eq!(dx_matrix.pivots()[masked_col], i as isize);
                    let col = target_zero_mask[masked_col];

                    for (input, output) in inputs.iter_mut().zip(results.iter_mut()) {
                        let entry = input.entry(col);
                        if entry != 0 {
                            output
                                .into()
                                .add_unmasked(dx_matrix.row_segment(i, 2, 2), 1, &mask);
                            input.as_slice_mut().add(dx_matrix.row_segment(i, 1, 1), 1);
                        }
                    }
                }

                // Drop these objects to save a bit of memory
                target_zero_mask = Vec::new();
                dx_matrix = AugmentedMatrix::<3>::new(p, 0, [0, 0, 0]);
            } else {
                scratch0.update_from_bytes(&mut f).unwrap();
                scratch1.update_from_bytes(&mut f).unwrap();
                for (input, output) in inputs.iter_mut().zip(results.iter_mut()) {
                    let entry = input.entry(col);
                    if entry != 0 {
                        output.into().add_unmasked(scratch0.as_slice(), 1, &mask);
                        // If we resume a resolve_through_stem, input may be longer than scratch1.
                        input
                            .slice_mut(0, scratch1.len())
                            .add(scratch1.as_slice(), 1);
                    }
                }

                // Row reduce the differentials
                if !target_zero_mask.is_empty() {
                    for i in 0..dx_matrix.rows() {
                        if dx_matrix.row_segment(i, 1, 1).entry(col) != 0 {
                            dx_matrix
                                .row_segment_mut(i, 2, 2)
                                .slice_mut(0, zero_mask_dim)
                                .add(scratch0.as_slice(), 1);
                            dx_matrix
                                .row_segment_mut(i, 1, 1)
                                .slice_mut(0, target_dim)
                                .add(scratch1.as_slice(), 1);
                        }
                    }
                }
            }
        }
        // Make sure we have finished reading everything
        drop(f);

        for dx in inputs {
            assert!(
                dx.is_zero(),
                "remainder non-zero at {b}\nAlgebra: {subalgebra}\ndx: {}",
                target.element_to_string(b.t(), dx.as_slice())
            );
        }
        true
    }
}

impl<M: ZeroModule<Algebra = MilnorAlgebra>> AugmentedChainComplex for Resolution<M> {
    type ChainMap = FreeModuleHomomorphism<M>;
    type TargetComplex = FiniteChainComplex<M, FullModuleHomomorphism<M, M>>;

    fn target(&self) -> Arc<Self::TargetComplex> {
        Arc::clone(&self.target)
    }

    fn chain_map(&self, s: i32) -> Arc<Self::ChainMap> {
        Arc::clone(&self.chain_maps[s])
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;
    // Pulled in here rather than at file scope: the resolution itself no longer calls a
    // `FreeChainComplex` method, so a top-level import would be unused in a non-test build.
    use crate::chain_complex::FreeChainComplex;

    #[test]
    fn test_restart_stem() {
        let res = crate::utils::construct_nassau("S_2", None).unwrap();
        res.compute_through_stem(Bidegree::n_s(14, 8));
        res.compute_through_bidegree(Bidegree::s_t(5, 19));

        expect![[r#"
            ·                             
            ·                     ·       
            ·                   · ·     · 
            ·                 ·   ·     · 
            ·             ·   ·         · · 
            ·     ·       · · ·         · ·   
            ·   · ·     · · ·           · · ·   
            · ·   ·       ·               ·       
            ·                                       
        "#]]
        .assert_eq(&res.graded_dimension_string());
    }

    /// An entry too wide for its field would shift into the next one, so it must not be packed
    /// and compared. Nothing has such a signature, so the mask is unsatisfiable.
    #[test]
    fn test_packed_signature_rejects_oversized_entry() {
        let subalgebra = MilnorSubalgebra::new(vec![1, 1]);

        assert!(subalgebra.packed_signature(&[0, 0]).is_some());
        assert!(
            subalgebra
                .packed_signature(&[PPart::max_entry(0), 0])
                .is_some()
        );
        assert!(
            subalgebra
                .packed_signature(&[PPart::max_entry(0) + 1, 0])
                .is_none()
        );
    }

    /// Cross-check the secondary (d2) computation on a *save-backed* Nassau resolution computed
    /// with the relaxed [`Resolution::compute_through_stem`] against the standard resolution. This
    /// exercises the quasi-inverse save files, which under the relaxed schedule are always written
    /// using the "incomplete information" (`Magic::Fix`) path, since a bidegree is computed while
    /// ignoring the same-degree generators of its target.
    #[test]
    fn test_stem_concurrent_secondary() {
        use std::sync::Arc;

        use algebra::pair_algebra::PairAlgebra;

        use crate::{
            chain_complex::FreeChainComplex, resolution::secondary::SecondaryResolution,
            secondary::SecondaryLift, utils::construct_standard,
        };

        /// Render every non-trivial d2 in `lift` as one line, for comparison across resolutions.
        fn d2_chart<CC>(lift: &SecondaryResolution<CC>) -> String
        where
            CC: FreeChainComplex,
            CC::Algebra: PairAlgebra,
        {
            let underlying = lift.underlying();
            let mut out = String::new();
            // Mirror the guarded iteration in `SecondaryResolution::e3_page`.
            for b in underlying.iter_stem() {
                if b.t() > 0 && underlying.has_computed_bidegree(b + Bidegree::n_s(-1, 2)) {
                    let matrix = lift.homotopy(b.s() + 2).homotopies.hom_k(b.t());
                    if matrix.iter().any(|row| !row.is_empty()) {
                        out.push_str(&format!("d2 {b}: {matrix:?}\n"));
                    }
                }
            }
            out
        }

        // Far enough to carry a nonzero d2 (the first is `d2(h4) = h0 h3^2`, out of stem 15) and
        // no further: this runs on every `cargo test`, and the assertion below fails loudly if the
        // range is ever trimmed past the last differential it is meant to compare.
        let max = Bidegree::n_s(16, 5);

        let dir = tempfile::TempDir::new().unwrap();
        let nassau = crate::utils::construct_nassau("S_2", Some(dir.path().to_owned())).unwrap();
        nassau.compute_through_stem(max);
        let nassau_lift = SecondaryResolution::new(Arc::new(nassau));
        nassau_lift.extend_all();

        let standard = construct_standard::<false, _, _>("S_2", None).unwrap();
        standard.compute_through_stem(max);
        let standard_lift = SecondaryResolution::new(Arc::new(standard));
        standard_lift.extend_all();

        let nassau_chart = d2_chart(&nassau_lift);
        // Both charts are built by the same guarded iteration, so an empty pair would compare
        // equal without having compared any d2 at all.
        assert!(
            !nassau_chart.is_empty(),
            "no d2 differentials were compared"
        );

        assert_eq!(
            nassau_chart,
            d2_chart(&standard_lift),
            "secondary d2 chart differs between Nassau (save-backed, relaxed schedule) and \
             standard"
        );
    }

    /// Resolve `S_2` through `max`, with shift reuse on or off, optionally into a save directory.
    fn resolve_s2(
        max: Bidegree,
        shift_reuse: bool,
        verify: bool,
        save: Option<&std::path::Path>,
    ) -> Resolution<algebra::module::FDModule<MilnorAlgebra>> {
        let mut res = crate::utils::construct_nassau("S_2", save.map(|p| p.to_owned())).unwrap();
        res.set_shift_reuse(shift_reuse);
        res.set_shift_verify(verify);
        res.compute_through_stem(max);
        res
    }

    /// Every generator and every differential agree, bit for bit.
    fn assert_same_resolution<M: ZeroModule<Algebra = MilnorAlgebra>>(
        a: &Resolution<M>,
        b: &Resolution<M>,
        max: Bidegree,
    ) {
        assert_eq!(a.graded_dimension_string(), b.graded_dimension_string());
        let mut compared = 0;
        for s in 0..=max.s() {
            for t in 0..=max.n() + s {
                let bd = Bidegree::s_t(s, t);
                assert_eq!(
                    a.has_computed_bidegree(bd),
                    b.has_computed_bidegree(bd),
                    "{bd}"
                );
                if !a.has_computed_bidegree(bd) {
                    continue;
                }
                let n = a.number_of_gens_in_bidegree(bd);
                assert_eq!(n, b.number_of_gens_in_bidegree(bd), "generators at {bd}");
                for i in 0..n {
                    assert_eq!(
                        a.differential(s).output(t, i),
                        b.differential(s).output(t, i),
                        "differential on generator {i} at {bd}"
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared > 0, "compared no differentials");
    }

    /// Every file under `root`, by relative path, with its bytes.
    fn dir_contents(root: &std::path::Path) -> std::collections::BTreeMap<String, Vec<u8>> {
        fn walk(
            root: &std::path::Path,
            dir: &std::path::Path,
            out: &mut std::collections::BTreeMap<String, Vec<u8>>,
        ) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(root, &path, out);
                } else {
                    let rel = path
                        .strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    out.insert(rel, std::fs::read(&path).unwrap());
                }
            }
        }
        let mut out = std::collections::BTreeMap::new();
        walk(root, root, &mut out);
        out
    }

    /// Resolve with and without shift reuse, saving both, and require identical differentials and
    /// byte-identical save files (differentials and quasi-inverses).
    fn check_shift_reuse_identical(max: Bidegree) {
        let dir_on = tempfile::TempDir::new().unwrap();
        let dir_off = tempfile::TempDir::new().unwrap();
        let on = resolve_s2(max, true, false, Some(dir_on.path()));
        let off = resolve_s2(max, false, false, Some(dir_off.path()));
        assert_same_resolution(&on, &off, max);

        let stats = on.shift_stats();
        assert!(stats.hits > 0, "shift reuse never hit: {stats:?}");
        assert_eq!(off.shift_stats(), ShiftStats::default());

        let files_on = dir_contents(dir_on.path());
        let files_off = dir_contents(dir_off.path());
        assert!(
            files_on.keys().any(|k| k.contains("nassau_qi")),
            "no qi files saved"
        );
        assert_eq!(
            files_on.keys().collect::<Vec<_>>(),
            files_off.keys().collect::<Vec<_>>()
        );
        for (name, bytes) in &files_on {
            assert!(bytes == &files_off[name], "save file {name} differs");
        }
    }

    /// Every block a signature step reads is a prefix of a direct build of that step's matrix.
    #[test]
    fn test_shift_block_matches_direct_build() {
        let max = Bidegree::n_s(30, 12);
        let res = resolve_s2(max, true, true, None);
        let stats = res.shift_stats();
        assert!(stats.hits > 0 && stats.shared_builds > 0, "{stats:?}");
    }

    #[test]
    fn test_shift_reuse_identical() {
        check_shift_reuse_identical(Bidegree::n_s(30, 12));
    }

    /// [`test_shift_reuse_identical`] across the `A(1)`/`A(2)` boundaries. Run with `--release
    /// --ignored`.
    #[test]
    #[ignore]
    fn test_shift_reuse_identical_large() {
        check_shift_reuse_identical(Bidegree::n_s(80, 40));
    }

    #[test]
    fn test_signature_iterator() {
        let subalgebra = MilnorSubalgebra::new(vec![2, 1]);
        assert_eq!(
            subalgebra.iter_signatures(6).collect::<Vec<_>>(),
            vec![
                vec![1, 0],
                vec![2, 0],
                vec![3, 0],
                vec![0, 1],
                vec![1, 1],
                vec![2, 1],
                vec![3, 1],
            ]
        );

        assert_eq!(
            subalgebra.iter_signatures(5).collect::<Vec<_>>(),
            vec![
                vec![1, 0],
                vec![2, 0],
                vec![3, 0],
                vec![0, 1],
                vec![1, 1],
                vec![2, 1],
            ]
        );
        assert_eq!(
            subalgebra.iter_signatures(4).collect::<Vec<_>>(),
            vec![vec![1, 0], vec![2, 0], vec![3, 0], vec![0, 1], vec![1, 1],]
        );
        assert_eq!(
            subalgebra.iter_signatures(3).collect::<Vec<_>>(),
            vec![vec![1, 0], vec![2, 0], vec![3, 0], vec![0, 1],]
        );
        assert_eq!(
            subalgebra.iter_signatures(2).collect::<Vec<_>>(),
            vec![vec![1, 0], vec![2, 0],]
        );
        assert_eq!(
            subalgebra.iter_signatures(1).collect::<Vec<_>>(),
            vec![vec![1, 0],]
        );
        assert_eq!(
            subalgebra.iter_signatures(0).collect::<Vec<_>>(),
            Vec::<Vec<PPartEntry>>::new()
        );
    }

    #[test]
    fn test_signature_iterator_large() {
        let subalgebra = MilnorSubalgebra::new(vec![
            0,
            MilnorSubalgebra::INFINITY,
            MilnorSubalgebra::INFINITY,
            MilnorSubalgebra::INFINITY,
        ]);
        assert_eq!(
            subalgebra.iter_signatures(7).collect::<Vec<_>>(),
            vec![vec![0, 1, 0, 0], vec![0, 2, 0, 0], vec![0, 0, 1, 0],]
        );
    }

    #[test]
    fn test_subalgebra_fmt() {
        let f2 = MilnorSubalgebra::zero_algebra();
        let a2 = MilnorSubalgebra::new(vec![3, 2, 1]);
        let b3321 = MilnorSubalgebra::new(vec![3, 3, 2, 1]);

        assert_eq!(f2.to_string(), "F_2");
        assert_eq!(a2.to_string(), "A(2)");
        assert_eq!(b3321.to_string(), "B(3,3,2,1)");
    }
}
