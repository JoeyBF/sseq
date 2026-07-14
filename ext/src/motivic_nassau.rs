//! The **motivic Nassau** spike: a signature-filtration resolution engine for the
//! trivial module over `A_C/τ` (the [`CTauAlgebra`]), prototyping Christian
//! Nassau's Algorithm 2 (arXiv:1910.04063) as an alternative to the generic
//! minimal-resolution engine.
//!
//! This is a **spike**: additive, feature-gated (`sig-nassau`), and validated
//! against the generic engine rather than replacing it. The generic engine is
//! never touched; wherever Nassau's shortcut is not provably applicable at a
//! bidegree, this engine **falls back** to a plain (all-of-`A_C/τ`) step there, so
//! the composite is always correct.
//!
//! # Unified engine over a [`PAlgebra`]
//!
//! The resolution engine [`SignatureResolution`] is generic over two traits that
//! together capture Margolis's notion of a P-algebra presented for Nassau's
//! algorithm:
//!
//! - [`Profile`] — the coset combinatorics of a single finite sub-Hopf-algebra
//!   `B` (its signature space `S_B ≅ B\\A`): `zero_signature`, `iter_signatures`,
//!   `dimension`, `name`. No basis, no module — pure combinatorics.
//! - [`PAlgebra`] — the algebra `A`: its profile family (`optimal_profile`, from
//!   the vanishing lines) and the single basis-dependent primitive, the coset
//!   projection [`PAlgebra::basis_element_signature`] (`sig_B`). Everything else
//!   ([`signature_mask`], [`signature_matrix`]) is *derived* from it generically.
//!
//! Instantiated over the motivic $A_C/\tau$ (via the opposite algebra
//! [`CTauOpAlgebra`]) **and** the mod-$p$ Steenrod algebra `MilnorAlgebra` for
//! every prime — classical ($p = 2$) and odd-primary ($p = 3, 5, \dots$) alike,
//! via [`SteenrodProfile`]. One engine, three algebra families. The engine is
//! $\mathbb{F}_p$-generic (matrices/vectors over `algebra.prime()`, corrections
//! through `QuasiInverse::apply`); the handedness (which the note in
//! `notes/opposite-algebra-nassau.tex` explains) and the prime live entirely in
//! the trait implementations. The Hopf/freeness axioms of a P-algebra are a
//! precondition the implementor is trusted to meet (guarded in practice by the
//! `d²=0` fallback).
//!
//! # The odd-primary shape
//!
//! `A_C/τ ≅ F₂[ξ₁, ξ₂, …] ⊗ E(τ₀, τ₁, …)` is the *odd-primary-shaped* Steenrod
//! algebra of Nassau §4/§7: a basis element is `Q(E)P(R)` with an exterior part
//! `E` (the `τ_i`/`Q_i`, a squarefree bitmask) and a polynomial part `R` (the
//! `ξ_j` exponents). Every signature, product, and vanishing cone here carries
//! **both** parts — the polynomial-only body of §2/§3 is not enough (plan
//! subtlety #1).
//!
//! # Signatures (§2.1)
//!
//! Fix a finite admissible sub-Hopf-algebra `B ⊂ A_C/τ` by a
//! [`MotivicSubalgebra`] profile: a polynomial profile `p` (so `B` contains
//! `P(R)` with `r_{i+1} < 2^{p_i}`) and an exterior length `q_len` (so `B`
//! contains `Q_0, …, Q_{q_len-1}`). The **signature** of a basis element
//! `Q(E)P(R)` is its `B`-component:
//! - polynomial: `sig_p[i] = r_{i+1} mod 2^{p_i}`;
//! - exterior:   `sig_q   = E ∩ {Q_0, …, Q_{q_len-1}}` (the low exterior bits).
//!
//! The zero signature `E₀` collects the elements whose `B`-component is trivial
//! (`r_{i+1} ≡ 0 mod 2^{p_i}` and `E ⊆ {Q_i : i ≥ q_len}`); this is where the
//! homology work happens, and `dim E₀C ≈ dim C / dim B` (Nassau's shrink).

use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

use algebra::{
    Algebra, combinatorics,
    milnor_algebra::MilnorAlgebra,
    module::{
        FDModule, FreeModule, GeneratorData, Module,
        homomorphism::{FreeModuleHomomorphism, ModuleHomomorphism},
    },
    motivic::CTauAlgebra,
};
use bivec::BiVec;
use fp::{
    matrix::{AugmentedMatrix, Matrix},
    prime::{Prime, TWO, ValidPrime},
    vector::{FpSliceMut, FpVector},
};

use crate::chain_complex::{ChainComplex, FiniteChainComplex};

/// Cap on new generators introduced at a single `(1, t)` bidegree (mirrors the
/// classical engine's headroom for the augmented matrix).
const MAX_NEW_GENS: usize = 10;

/// The **opposite** algebra of `A_C/τ`: the same underlying `F₂`-vector space and
/// basis, with the product reversed (`a ·ᵒᵖ b = b · a`).
///
/// # Why the spike resolves over the opposite algebra
///
/// A free left module over `A_C/τ` has differential `d(m·g) = m·d(g) = Σ (m·n_j)
/// g_j`, so the source operation `m` is the **left** factor of every output
/// product. Nassau's signature-by-signature sweep needs those products to raise
/// the signature of `m` (block-triangularity by source signature). Empirically —
/// see `sig_diag` and [`tests::product_raises_signature`] — over `A_C/τ` it is the
/// **right** factor whose signature is a floor (left multiplication preserves
/// `sig ≥ R`); the left factor's signature is *not* stable (the τ/ξ coupling of
/// the odd-primary product, plan subtlety #1). Reversing the product makes the
/// source operation the right factor, restoring the floor.
///
/// This is free: `A_C/τ` is a Hopf algebra, so its antipode is an algebra
/// isomorphism `(A_C/τ)^{op} ≅ A_C/τ`, and `Ext_{A^{op}}(k, k) ≅ Ext_A(k, k)`.
/// The generator ranks are therefore **identical** to the generic (left) engine's
/// golden fixture — as [`tests::opposite_algebra_matches_golden_total`] checks.
#[derive(Default)]
pub struct CTauOpAlgebra {
    inner: CTauAlgebra,
}

impl CTauOpAlgebra {
    pub fn new() -> Self {
        Self {
            inner: CTauAlgebra::new(),
        }
    }

    /// The underlying (left) `A_C/τ` view, for basis-monomial access and weights.
    pub fn inner(&self) -> &CTauAlgebra {
        &self.inner
    }
}

impl std::fmt::Display for CTauOpAlgebra {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "CTauOpAlgebra(A_C/τ ᵒᵖ, p=2)")
    }
}

impl Algebra for CTauOpAlgebra {
    fn prefix(&self) -> &str {
        "motivic_ctau_op"
    }

    fn magic(&self) -> u32 {
        // Distinct from CTauAlgebra so save files never cross-load.
        0x004D_0004
    }

    fn prime(&self) -> ValidPrime {
        TWO
    }

    fn compute_basis(&self, degree: i32) {
        self.inner.compute_basis(degree);
    }

    fn dimension(&self, degree: i32) -> usize {
        self.inner.dimension(degree)
    }

    fn multiply_basis_elements(
        &self,
        result: FpSliceMut,
        coeff: u32,
        r_degree: i32,
        r_idx: usize,
        s_degree: i32,
        s_idx: usize,
    ) {
        // The opposite product: a ·ᵒᵖ b = b · a.
        self.inner
            .multiply_basis_elements(result, coeff, s_degree, s_idx, r_degree, r_idx);
    }

    fn basis_element_to_string(&self, degree: i32, idx: usize) -> String {
        self.inner.basis_element_to_string(degree, idx)
    }

    fn basis_element_from_string(&self, elt: &str) -> Option<(i32, usize)> {
        self.inner.basis_element_from_string(elt)
    }
}

/// A finite admissible sub-Hopf-algebra `B ⊂ A_C/τ`, with the ordering of its
/// signatures used by Algorithm 2 (Lemma 2.4).
///
/// `B` is specified by a **polynomial profile** and an **exterior length**:
/// - `p_profile[i]` is the profile exponent for `ξ_{i+1}`, so `B` contains
///   `P(R)` exactly when `r_{i+1} < 2^{p_profile[i]}` for every `i` (and
///   `r_j = 0` for `j > p_profile.len()`);
/// - `q_len` is the number of exterior generators in `B`: `Q_0, …, Q_{q_len-1}`.
///
/// So `A(n)` is `p_profile = [n, n-1, …, 1]`, `q_len = n+1` (e.g. `A(0) = E(Q_0)`
/// is `p_profile = []`, `q_len = 1`), and `F₂` is `p_profile = []`, `q_len = 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotivicSubalgebra {
    /// `p_profile[i]` = profile exponent for `ξ_{i+1}` (B allows `r_{i+1} < 2^{p_profile[i]}`;
    /// `p_profile[i] = 0` means `ξ_{i+1} ∉ B`). Entries beyond the vector are `0`.
    p_profile: Vec<u8>,
    /// `Q_0, …, Q_{q_len-1}` are the exterior generators in B.
    q_len: usize,
}

/// A signature: the `B`-component of a basis element `Q(E)P(R)` — see the module
/// docs. `q` is the exterior component (`E ∩ {Q_0, …, Q_{q_len-1}}`), `p[i]` the
/// polynomial component `r_{i+1} mod 2^{p_profile[i]}`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Signature {
    /// Exterior component: a submask of `(1 << q_len) - 1`.
    pub q: u32,
    /// Polynomial component: `p[i] = r_{i+1} mod 2^{p_profile[i]}`.
    pub p: Vec<u32>,
}

impl MotivicSubalgebra {
    /// A subalgebra with the given polynomial profile and exterior length.
    pub fn new(p_profile: Vec<u8>, q_len: usize) -> Self {
        Self { p_profile, q_len }
    }

    /// `F₂` — the trivial subalgebra (no signatures; forces a plain step).
    pub fn trivial() -> Self {
        Self {
            p_profile: vec![],
            q_len: 0,
        }
    }

    /// `A(n)`: `Q_0, …, Q_n` and the polynomial profile `[n, n-1, …, 1]`.
    ///
    /// In the odd-primary/EA shape `A(n) = ⟨Q_0, …, Q_n, P^{2^0}, …, P^{2^{n-1}}⟩`;
    /// dually `ξ_{i+1}` is truncated at exponent `2^{n-i}`, i.e. profile
    /// `p_profile[i] = n - i` for `i = 0, …, n-1`.
    pub fn a_n(n: usize) -> Self {
        Self {
            p_profile: (0..n).map(|i| (n - i) as u8).collect(),
            q_len: n + 1,
        }
    }

    pub fn p_profile(&self) -> &[u8] {
        &self.p_profile
    }

    pub fn q_len(&self) -> usize {
        self.q_len
    }

    /// Is this the trivial subalgebra `F₂` (no nonzero signatures)?
    pub fn is_trivial(&self) -> bool {
        self.q_len == 0 && self.p_profile.iter().all(|&e| e == 0)
    }

    /// The `F₂`-dimension of `B`: `2^{q_len} · Π_i 2^{p_profile[i]}`.
    pub fn dimension(&self) -> u64 {
        let poly: u32 = self.p_profile.iter().map(|&e| e as u32).sum();
        1u64 << (self.q_len as u32 + poly)
    }

    /// The zero signature.
    pub fn zero_signature(&self) -> Signature {
        Signature {
            q: 0,
            p: vec![0; self.p_profile.len()],
        }
    }

    /// The mask selecting the exterior generators of `B`.
    fn q_mask(&self) -> u32 {
        if self.q_len >= 32 {
            !0
        } else {
            (1u32 << self.q_len) - 1
        }
    }

    /// The signature of a basis element `Q(E)P(R)` given as its monomial
    /// `(e_mask, r)` (the [`MotivicMilnorAlgebra`](algebra::motivic::MotivicMilnorAlgebra)
    /// encoding: bit `i` of `e_mask` is `τ_i`; `r[j]` is the exponent of `ξ_j`,
    /// with `r[0]` unused).
    pub fn signature_of(&self, e_mask: u32, r: &[u32]) -> Signature {
        let q = e_mask & self.q_mask();
        let p = self
            .p_profile
            .iter()
            .enumerate()
            .map(|(i, &e)| r.get(i + 1).copied().unwrap_or(0) & ((1u32 << e) - 1))
            .collect();
        Signature { q, p }
    }

    /// Whether `Q(E)P(R)` has the given signature.
    pub fn has_signature(&self, e_mask: u32, r: &[u32], sig: &Signature) -> bool {
        if e_mask & self.q_mask() != sig.q {
            return false;
        }
        for (i, &e) in self.p_profile.iter().enumerate() {
            let ri = r.get(i + 1).copied().unwrap_or(0);
            if ri & ((1u32 << e) - 1) != sig.p.get(i).copied().unwrap_or(0) {
                return false;
            }
        }
        true
    }

    /// The topological degree of the `B`-basis element a signature represents
    /// (`Σ sig_q bits · |Q_i| + Σ sig_p[i] · |ξ_{i+1}|`). `|Q_i| = 2^{i+1}-1`,
    /// `|ξ_j| = 2^{j+1}-2`.
    pub fn signature_degree(&self, sig: &Signature) -> i32 {
        let mut d = 0i32;
        for i in 0..self.q_len {
            if (sig.q >> i) & 1 != 0 {
                d += (1 << (i + 1)) - 1;
            }
        }
        for (i, &v) in sig.p.iter().enumerate() {
            // ξ_{i+1}: |ξ_{i+1}| = 2^{i+2} - 2.
            d += v as i32 * ((1 << (i + 2)) - 2);
        }
        d
    }

    /// The top topological degree of `B` (the degree of its longest monomial).
    pub fn top_degree(&self) -> i32 {
        let mut d = 0i32;
        for i in 0..self.q_len {
            d += (1 << (i + 1)) - 1;
        }
        for (i, &e) in self.p_profile.iter().enumerate() {
            d += ((1u32 << e) - 1) as i32 * ((1 << (i + 2)) - 2);
        }
        d
    }

    /// Enumerate every **nonzero** signature whose representing `B`-element has
    /// topological degree at most `degree`, in Algorithm 2's admissible order
    /// (Lemma 2.4): **signature degree ascending**, the linear extension of
    /// "left-multiplication raises signature" that `sig_diag` verifies is
    /// degree-monotone over `A_C/τ` (with the coordinate vector as a deterministic
    /// tie-break). The initial zero signature is skipped.
    ///
    /// [`tests::product_raises_signature`] checks the underlying invariant against
    /// the real `A_C/τ` product.
    pub fn iter_signatures(&self, degree: i32) -> Vec<Signature> {
        let coords = self.coordinates();
        let mut out = Vec::new();
        let mut cur = vec![0u32; coords.len()];
        loop {
            // Increment the mixed-radix counter `cur` (little-endian over coords).
            let mut i = 0;
            loop {
                if i == coords.len() {
                    // Overflow: done.
                    let mut sorted = out;
                    sorted.sort_by(|a, b| self.order_degree_key(a).cmp(&self.order_degree_key(b)));
                    return sorted;
                }
                cur[i] += 1;
                if cur[i] < coords[i].radix {
                    break;
                }
                cur[i] = 0;
                i += 1;
            }
            let sig = self.counter_to_signature(&coords, &cur);
            if self.signature_degree(&sig) <= degree {
                out.push(sig);
            }
        }
    }

    /// Sort key for the correction order: **signature degree ascending** — the
    /// linear extension of "left-multiplication raises signature" validated by
    /// `sig_diag` (`deg_monotone`) — with the coordinate vector as a deterministic
    /// tie-break. See [`Self::iter_signatures`].
    fn order_degree_key(&self, sig: &Signature) -> (i32, Vec<u32>) {
        (self.signature_degree(sig), self.order_key(sig))
    }

    /// A signature coordinate: one generator's contribution.
    fn coordinates(&self) -> Vec<Coord> {
        let mut coords = Vec::new();
        // List by generator degree ascending: Q_0 (deg 1), ξ_1 (deg 2), Q_1 (deg 3), ξ_2 (deg 6), …
        let n_poly = self.p_profile.len();
        let n = self.q_len.max(n_poly + 1);
        for k in 0..n {
            // Q_k (exterior index k), degree 2^{k+1} - 1.
            if k < self.q_len {
                coords.push(Coord {
                    kind: CoordKind::Ext(k),
                    radix: 2,
                });
            }
            // ξ_{k+1} (polynomial index k), degree 2^{k+2} - 2.
            if k < n_poly && self.p_profile[k] > 0 {
                coords.push(Coord {
                    kind: CoordKind::Poly(k),
                    radix: 1u32 << self.p_profile[k],
                });
            }
        }
        coords
    }

    fn counter_to_signature(&self, coords: &[Coord], cur: &[u32]) -> Signature {
        let mut sig = self.zero_signature();
        for (c, &v) in coords.iter().zip(cur) {
            match c.kind {
                CoordKind::Ext(k) => {
                    if v != 0 {
                        sig.q |= 1 << k;
                    }
                }
                CoordKind::Poly(k) => sig.p[k] = v,
            }
        }
        sig
    }

    /// Total order key for the admissible order: coordinate vector by generator
    /// degree ascending, most-significant generator first for lexicographic
    /// comparison. (A linear extension of the signature partial order.)
    fn order_key(&self, sig: &Signature) -> Vec<u32> {
        let coords = self.coordinates();
        // Reverse so the highest-degree generator is compared first (lexicographic
        // on the reversed coordinate list), matching Nassau's reverse-lex ordering.
        coords
            .iter()
            .rev()
            .map(|c| match c.kind {
                CoordKind::Ext(k) => (sig.q >> k) & 1,
                CoordKind::Poly(k) => sig.p.get(k).copied().unwrap_or(0),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// M2: signature masking of the free module + the differential (Lemma 2.5).
//
// The "B-trivial signature-graded action" is realized, exactly as in the
// classical engine, by *masking* the ordinary A_C/τ product to the basis
// elements of a fixed signature — rather than a separate B-trivial matrix
// enumeration. Restricting d : E_R C → E_R C_{s-1} to the signature-`R` masks of
// source and target is the profile-restricted right action the resolver needs.
// ---------------------------------------------------------------------------

impl MotivicSubalgebra {
    /// The largest exterior index in `B` (`Q_{q_len-1}`), or `None` if `B` has no
    /// exterior part.
    fn max_bockstein(&self) -> Option<usize> {
        self.q_len.checked_sub(1)
    }

    /// Whether Algorithm 2 is provably valid at `(s, t)` for this `B`, by the
    /// below-line bound (Nassau Thm 3.1): with `B ⊆ A(n)` where `n = q_len-1` is
    /// the largest Bockstein, `A(n)` has slope `ρ_n = |Q_n| = 2^{n+1}-1`, and the
    /// step is valid once `t > ρ_n·(s+1) + τ_B` with `τ_B = top_degree(B)`. The
    /// slope is set by the largest Bockstein (Lemma 2.6), so a `B ⊊ A(n)` shares
    /// `A(n)`'s slope but has a smaller intercept `τ_B`, hence applies slightly
    /// sooner. This is **sound** (never applies `B` below its vanishing line), so
    /// the resulting ranks are always correct — not merely fallback-corrected.
    pub fn applicable(&self, s: i32, t: i32) -> bool {
        match self.max_bockstein() {
            None => false, // trivial B: no shortcut, handled as a plain step.
            Some(n) => {
                let rho = (1i64 << (n + 1)) - 1;
                (t as i64) > rho * (s as i64 + 1) + self.top_degree() as i64
            }
        }
    }

    /// Choose the applicable subalgebra of **largest `F₂`-dimension** at `(s, t)`
    /// — the one with the smallest zero-signature piece `E₀C ≈ C/\dim B`, Nassau's
    /// practical heuristic. The candidate ladder is the `A(n)` family together with
    /// the pure-exterior `E(Q_0, …, Q_{k-1})` subalgebras (which share `A(k-1)`'s
    /// vanishing slope but a smaller intercept, so they can apply in a thin band
    /// just below the next `A(n)`). Returns [`MotivicSubalgebra::trivial`] when
    /// nothing qualifies — the plain step.
    ///
    /// Admissibility couples the two parts: `Q_0·ξ_1 ∋ Q_1`, so any `B` containing
    /// `ξ_1` and `Q_0` must contain `Q_1`. Consequently no sound `B` improves on
    /// `A(0)`'s slope-1 line, and the large `A(0)`-region is essentially
    /// irreducible; the finer ladder only helps in the narrow inter-`A(n)` bands.
    pub fn optimal_for(s: i32, t: i32) -> Self {
        let mut best = Self::trivial();
        let mut best_dim = 1u64;
        // A(n) ladder and pure-exterior E(Q_0..Q_{k-1}) ladder. 8 levels covers any
        // feasible box (A(8) already has top_degree in the thousands).
        let candidates = (0..=8)
            .map(Self::a_n)
            .chain((1..=9).map(|k| Self::new(vec![], k)));
        for cand in candidates {
            if cand.applicable(s, t) && cand.dimension() > best_dim {
                best_dim = cand.dimension();
                best = cand;
            }
        }
        best
    }
}

#[derive(Clone, Copy)]
enum CoordKind {
    Ext(usize),
    Poly(usize),
}

#[derive(Clone, Copy)]
struct Coord {
    kind: CoordKind,
    radix: u32,
}

impl std::fmt::Display for MotivicSubalgebra {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_trivial() {
            write!(f, "F₂")
        } else if self.q_len > 0 && self.p_profile == Self::a_n(self.q_len - 1).p_profile {
            write!(f, "A({})", self.q_len - 1)
        } else if self.p_profile.iter().all(|&e| e == 0) {
            // Pure exterior E(Q_0, …, Q_{q_len-1}).
            write!(f, "E(Q_0..Q_{})", self.q_len - 1)
        } else {
            write!(f, "B(q_len={}, p={:?})", self.q_len, self.p_profile)
        }
    }
}

/// Iterate over the `(e_mask, r)` monomials of every `A_C/τ` basis element of a
/// given topological degree. Reads the shared
/// [`MotivicMilnorAlgebra`](algebra::motivic::MotivicMilnorAlgebra) basis under
/// `CTauAlgebra`. (Used by the signature tests.)
#[cfg(test)]
pub(crate) fn basis_monomials(alg: &CTauOpAlgebra, degree: i32) -> Vec<(u32, Vec<u32>)> {
    alg.compute_basis(degree);
    let engine = alg.inner().engine();
    (0..alg.dimension(degree))
        .map(|idx| engine.basis_element(degree, idx).clone())
        .collect()
}

// ---------------------------------------------------------------------------
// The P-algebra abstraction, split into two traits.
//
//   * `Profile`   — the coset combinatorics of a single finite sub-Hopf-algebra
//                   B: its signature space S_B (= B\\A), independent of any basis
//                   or module. This is where the signature methods live.
//   * `PAlgebra`  — the algebra A itself: its profile family (with the
//                   vanishing-line B-selection) and the one basis-dependent
//                   primitive, the coset projection `sig_B` on basis elements.
//
// Everything else — `signature_mask`, `signature_matrix` — is *derived*
// generically from `PAlgebra::basis_element_signature`, so the two engines share
// it verbatim.
// ---------------------------------------------------------------------------

/// The signature combinatorics of a single finite sub-Hopf-algebra `B` — its
/// coset space `S_B ≅ B\\A` as a graded, ordered set — with **no reference to a
/// basis, a module, or the ambient algebra**. (These are exactly the
/// [`PAlgebra`] methods that were really about signatures, factored out.)
///
/// Mathematically `Profile` packages the data Margolis's freeness `A ≅ B ⊗
/// (B\\A)` produces: a distinguished zero coset (`E₀`) and the finite set of
/// cosets carrying an element up to a given degree, in the order the correction
/// sweep must process them (a linear extension of "right-multiplication raises
/// signature").
pub trait Profile {
    /// A `B`-signature: a coset label in `S_B`.
    type Signature: Clone + Eq;

    /// Whether this is the trivial profile `F₂` (only the zero signature, so the
    /// step degenerates to the ordinary generic step).
    fn is_trivial(&self) -> bool;
    /// A display name (for stats), e.g. `A(1)` or `E(Q_0..Q_1)`.
    fn name(&self) -> String;
    /// The `F₂`-dimension of `B` — the ideal shrink factor `dim C / dim E₀C`.
    fn dimension(&self) -> u64;
    /// The zero signature (the `E₀` coset, where the homology is computed).
    fn zero_signature(&self) -> Self::Signature;
    /// The **nonzero** signatures with a representative of degree `≤ degree`, in
    /// the correction order. The zero signature is excluded.
    fn iter_signatures(&self, degree: i32) -> Vec<Self::Signature>;
}

/// A **P-algebra** (Margolis) presented for Nassau's algorithm: a graded connected
/// $\mathbb{F}_2$-Hopf algebra, free over each finite sub-Hopf-algebra, together
/// with a family of [`Profile`]s and the coset projection $\mathrm{sig}_B$ on
/// basis elements.
///
/// The trait is the *computational shadow* of a P-algebra: it requires only
/// [`Algebra`] (a basis and product). The Hopf structure and freeness over `B` —
/// which make the signatures well-defined — are a **precondition** the implementor
/// is trusted to satisfy (in practice guarded by the engine's `d² = 0` fallback
/// and rank validation). The implementor must also present the algebra in the
/// **handedness** the sweep needs — the signature filtration by *right* ideals —
/// so that the free-left-module differential is filtered (see
/// `notes/opposite-algebra-nassau.tex`). For $A_C/\tau$ that means the *opposite*
/// algebra [`CTauOpAlgebra`]; the classical Milnor basis is already right-stable.
///
/// The single basis-dependent primitive is [`Self::basis_element_signature`], the
/// coset projection $\mathrm{sig}_B\colon A \to S_B$; [`signature_mask`] and
/// [`signature_matrix`] are derived from it generically.
pub trait PAlgebra: Algebra + Sized {
    /// The profile type — a finite sub-Hopf-algebra `B` with its coset
    /// combinatorics.
    type Profile: Profile;

    /// The profile to use at bidegree `(s, t)`: the applicable one of largest
    /// dimension (smallest `E₀`), or the **trivial** profile `A(-1) = k` (a plain
    /// step) when none is applicable. Must return a profile the sweep converges on
    /// — a provably-applicable `B` (Thm 3.1) or the trivial one; the trivial
    /// profile always works, so there is no runtime fallback. Takes `&self` because
    /// the choice (and, for variable-prime algebras, the profile) depends on the
    /// algebra.
    fn optimal_profile(&self, s: i32, t: i32) -> Self::Profile;

    /// The signature $\mathrm{sig}_B$ of the `idx`-th algebra basis element in
    /// `degree` — the coset of `B` it lies in. This is the only place the
    /// signature structure touches the algebra's basis.
    fn basis_element_signature(
        &self,
        profile: &Self::Profile,
        degree: i32,
        idx: usize,
    ) -> <Self::Profile as Profile>::Signature;
}

/// The signature of a [`PAlgebra`]'s profile — shorthand for the nested associated
/// type.
type Sig<A> = <<A as PAlgebra>::Profile as Profile>::Signature;

/// The free-module basis indices in `degree` whose operation part has the given
/// `signature` — the mask defining `E_signature(module)_degree`. Derived from
/// [`PAlgebra::basis_element_signature`]: a free-module basis element `op·g` at
/// `offset + idx` has the signature of its operation `op` (the `idx`-th algebra
/// basis element in `degree − |g|`). Generic over any [`PAlgebra`].
fn signature_mask<A: PAlgebra>(
    alg: &A,
    profile: &A::Profile,
    module: &FreeModule<A>,
    degree: i32,
    signature: &Sig<A>,
) -> Vec<usize> {
    let mut out = Vec::new();
    for GeneratorData {
        gen_deg,
        start: [offset],
        ..
    } in module.iter_gen_offsets([degree])
    {
        let op_deg = degree - gen_deg;
        alg.compute_basis(op_deg);
        for idx in 0..alg.dimension(op_deg) {
            if &alg.basis_element_signature(profile, op_deg, idx) == signature {
                out.push(offset + idx);
            }
        }
    }
    out
}

/// The matrix of a free-module homomorphism restricted to a signature-graded piece
/// (rows = source basis of `signature`, columns = target basis of `signature`):
/// `d : E_signature C → E_signature C_{s-1}` in the masked bases. Generic over any
/// [`PAlgebra`].
fn signature_matrix<A: PAlgebra>(
    alg: &A,
    profile: &A::Profile,
    hom: &FreeModuleHomomorphism<FreeModule<A>>,
    degree: i32,
    signature: &Sig<A>,
) -> Matrix {
    let source = hom.source();
    let target = hom.target();
    let target_degree = degree - hom.degree_shift();

    let p = alg.prime();
    let target_mask = signature_mask(alg, profile, &target, target_degree, signature);
    let source_mask = signature_mask(alg, profile, &source, degree, signature);

    let mut scratch = FpVector::new(p, target.dimension(target_degree));
    let mut result = Matrix::new(p, source_mask.len(), target_mask.len());
    for (mut row, &masked_index) in std::iter::zip(result.iter_mut(), &source_mask) {
        scratch.set_to_zero();
        hom.apply_to_basis_element(scratch.as_slice_mut(), 1, degree, masked_index);
        row.add_masked(scratch.as_slice(), 1, &target_mask);
    }
    result
}

/// [`Profile`] combinatorics for the motivic $A_C/\tau$: `S_B` of a
/// [`MotivicSubalgebra`].
impl Profile for MotivicSubalgebra {
    type Signature = Signature;

    fn is_trivial(&self) -> bool {
        MotivicSubalgebra::is_trivial(self)
    }

    fn name(&self) -> String {
        self.to_string()
    }

    fn dimension(&self) -> u64 {
        MotivicSubalgebra::dimension(self)
    }

    fn zero_signature(&self) -> Signature {
        MotivicSubalgebra::zero_signature(self)
    }

    fn iter_signatures(&self, degree: i32) -> Vec<Signature> {
        MotivicSubalgebra::iter_signatures(self, degree)
    }
}

/// [`PAlgebra`] for the motivic $A_C/\tau$, on the opposite algebra so the
/// filtration is right-stable (see [`CTauOpAlgebra`]).
impl PAlgebra for CTauOpAlgebra {
    type Profile = MotivicSubalgebra;

    fn optimal_profile(&self, s: i32, t: i32) -> MotivicSubalgebra {
        MotivicSubalgebra::optimal_for(s, t)
    }

    fn basis_element_signature(
        &self,
        profile: &MotivicSubalgebra,
        degree: i32,
        idx: usize,
    ) -> Signature {
        let (e, r) = self.inner().engine().basis_element(degree, idx);
        profile.signature_of(*e, r)
    }
}

/// A finite admissible sub-Hopf-algebra `B` of the mod-`p` Steenrod algebra for
/// **any** prime `p`, in the standard Milnor basis `Q(E)P(R)` — the profile the
/// generic engine uses over `MilnorAlgebra`. Uniform across primes: at `p = 2`
/// the exterior part is vacuous (`q_part = 0`, basis elements have no `Q`s) and
/// this reduces to the classical polynomial profile; at odd `p` it carries both
/// the polynomial truncation and the exterior `Q_i ∈ B`.
///
/// - `p_profile[i]` truncates `ξ_{i+1}` at exponent `p^{p_profile[i]}`;
/// - `q_part` bit `i` set ⟺ `Q_i ∈ B`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SteenrodProfile {
    prime: ValidPrime,
    p_profile: Vec<u8>,
    q_part: u32,
}

/// A signature over the mod-`p` Steenrod algebra: the exterior (`Q`) and
/// polynomial (`ξ`) `B`-components of a `Q(E)P(R)` basis element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SteenrodSignature {
    q: u32,
    p: Vec<u32>,
}

impl SteenrodProfile {
    fn trivial(prime: ValidPrime) -> Self {
        Self {
            prime,
            p_profile: vec![],
            q_part: 0,
        }
    }

    /// `A(n)` at prime `p`. At `p = 2`: the polynomial profile `[n+1, n, …, 1]`
    /// (no exterior). At odd `p`: `Q_0, …, Q_n` and `[n, n-1, …, 1]`.
    fn a_n(prime: ValidPrime, n: usize) -> Self {
        if prime.as_u32() == 2 {
            Self {
                prime,
                p_profile: (1..=n + 1).rev().map(|x| x as u8).collect(),
                q_part: 0,
            }
        } else {
            Self {
                prime,
                p_profile: (1..=n).rev().map(|x| x as u8).collect(),
                q_part: (1u32 << (n + 1)) - 1,
            }
        }
    }

    fn q_mask(&self) -> u32 {
        self.q_part
    }

    fn signature_of(&self, elt: &algebra::milnor_algebra::MilnorBasisElement) -> SteenrodSignature {
        let p = self.prime.as_u32();
        let q = elt.q_part & self.q_mask();
        let sp = self
            .p_profile
            .iter()
            .enumerate()
            .map(|(i, &e)| elt.p_part.get(i).copied().unwrap_or(0) % p.pow(e as u32))
            .collect();
        SteenrodSignature { q, p: sp }
    }

    /// The topological degree of the `B`-element a signature represents.
    fn signature_degree(&self, sig: &SteenrodSignature) -> i32 {
        let td = combinatorics::tau_degrees(self.prime);
        let xd = combinatorics::xi_degrees(self.prime);
        let mut d = 0i32;
        for i in 0..32 {
            if (sig.q >> i) & 1 != 0 {
                d += td[i as usize];
            }
        }
        for (i, &v) in sig.p.iter().enumerate() {
            d += v as i32 * xd[i];
        }
        d
    }

    fn top_degree(&self) -> i32 {
        let p = self.prime.as_u32();
        let td = combinatorics::tau_degrees(self.prime);
        let xd = combinatorics::xi_degrees(self.prime);
        let mut d = 0i32;
        for i in 0..32 {
            if (self.q_part >> i) & 1 != 0 {
                d += td[i as usize];
            }
        }
        for (i, &e) in self.p_profile.iter().enumerate() {
            d += (p.pow(e as u32) - 1) as i32 * xd[i];
        }
        d
    }

    /// The vanishing-line slope `ρ` (Thm 3.1): `2^{profile.len()}−1` at `p = 2`
    /// (poly-only `A(n)`), `|Q_n| = 2p^n−1` at odd `p` (largest Bockstein).
    fn rho(&self) -> i64 {
        if self.prime.as_u32() == 2 {
            (1i64 << self.p_profile.len()) - 1
        } else if self.q_part != 0 {
            let n = 31 - self.q_part.leading_zeros();
            combinatorics::tau_degrees(self.prime)[n as usize] as i64
        } else {
            0
        }
    }

    fn applicable(&self, s: i32, t: i32) -> bool {
        if self.is_trivial() {
            return false;
        }
        (t as i64) > self.rho() * (s as i64 + 1) + self.top_degree() as i64
    }

    fn optimal_for(prime: ValidPrime, s: i32, t: i32) -> Self {
        let mut best = Self::trivial(prime);
        let mut best_dim = 1u64;
        // A(n) ladder for every prime; at odd p also the pure-exterior
        // E(Q_0..Q_{k-1}) intermediates (which share A(k-1)'s slope, smaller τ_B).
        let a_ns = (0..=8).map(|n| Self::a_n(prime, n));
        let exteriors: Vec<Self> = if prime.as_u32() == 2 {
            vec![]
        } else {
            (1..=9)
                .map(|k| Self {
                    prime,
                    p_profile: vec![],
                    q_part: (1u32 << k) - 1,
                })
                .collect()
        };
        for cand in a_ns.chain(exteriors) {
            if cand.applicable(s, t) && cand.dimension() > best_dim {
                best_dim = cand.dimension();
                best = cand;
            }
        }
        best
    }
}

impl std::fmt::Display for SteenrodProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_trivial() {
            return write!(f, "F_{}", self.prime.as_u32());
        }
        // A(n) detection.
        for n in 0..=12 {
            if *self == Self::a_n(self.prime, n) {
                return write!(f, "A({n})");
            }
        }
        if self.p_profile.iter().all(|&e| e == 0) && self.q_part != 0 {
            return write!(f, "E(Q_0..Q_{})", 31 - self.q_part.leading_zeros());
        }
        write!(f, "B(q={:#x}, p={:?})", self.q_part, self.p_profile)
    }
}

/// [`Profile`] combinatorics for the mod-`p` Steenrod algebra.
impl Profile for SteenrodProfile {
    type Signature = SteenrodSignature;

    fn is_trivial(&self) -> bool {
        self.q_part == 0 && self.p_profile.iter().all(|&e| e == 0)
    }

    fn name(&self) -> String {
        self.to_string()
    }

    fn dimension(&self) -> u64 {
        let p = self.prime.as_u32() as u64;
        let poly: u64 = self.p_profile.iter().map(|&e| p.pow(e as u32)).product();
        poly << self.q_part.count_ones()
    }

    fn zero_signature(&self) -> SteenrodSignature {
        SteenrodSignature {
            q: 0,
            p: vec![0; self.p_profile.len()],
        }
    }

    fn iter_signatures(&self, degree: i32) -> Vec<SteenrodSignature> {
        // Mixed-radix enumeration over the signature coordinates (Q_i radix 2,
        // ξ_{i+1} radix p^{p_profile[i]}), keeping those of degree ≤ `degree`,
        // ordered by signature degree ascending (a valid linear extension).
        let p = self.prime.as_u32();
        let mut radices: Vec<(bool, usize, u32)> = Vec::new(); // (is_ext, index, radix)
        let n = self.p_profile.len().max(32);
        for k in 0..n {
            if (self.q_part >> k) & 1 != 0 {
                radices.push((true, k, 2));
            }
            if let Some(&e) = self.p_profile.get(k)
                && e > 0
            {
                radices.push((false, k, p.pow(e as u32)));
            }
        }
        let mut out = Vec::new();
        let mut cur = vec![0u32; radices.len()];
        loop {
            let mut i = 0;
            loop {
                if i == radices.len() {
                    out.sort_by_key(|s| (self.signature_degree(s), s.q, s.p.clone()));
                    return out;
                }
                cur[i] += 1;
                if cur[i] < radices[i].2 {
                    break;
                }
                cur[i] = 0;
                i += 1;
            }
            let mut sig = self.zero_signature();
            for (c, &v) in radices.iter().zip(&cur) {
                match c {
                    (true, k, _) if v != 0 => sig.q |= 1 << k,
                    (false, k, _) => sig.p[*k] = v,
                    _ => {}
                }
            }
            if self.signature_degree(&sig) <= degree {
                out.push(sig);
            }
        }
    }
}

/// [`PAlgebra`] for the mod-`p` Steenrod algebra in the standard Milnor basis,
/// for **every** prime. The standard basis is right-stable (no opposite needed),
/// so the same [`SignatureResolution`] engine drives the classical (`p = 2`) and
/// odd-primary cases; only the coset projection and the profile family, here,
/// are prime-aware.
impl PAlgebra for MilnorAlgebra {
    type Profile = SteenrodProfile;

    fn optimal_profile(&self, s: i32, t: i32) -> SteenrodProfile {
        if self.profile().is_trivial() {
            // Full Steenrod algebra: the A(n)/vanishing-line ladder (Thm 3.1).
            SteenrodProfile::optimal_for(self.prime(), s, t)
        } else {
            // A finite ambient (e.g. A(2), for tmf): the full-algebra vanishing
            // lines do not apply, and — with no runtime fallback — applying a
            // sub-`B` below its (different, and here unimplemented) vanishing
            // region would give *silently* wrong ranks. So we take plain steps
            // (trivial `B`), which are always correct. A finite algebra is small,
            // so the plain resolution is cheap; a sound sub-`B` shortcut for
            // finite ambients (a change-of-rings vanishing line) is future work.
            SteenrodProfile::trivial(self.prime())
        }
    }

    fn basis_element_signature(
        &self,
        profile: &SteenrodProfile,
        degree: i32,
        idx: usize,
    ) -> SteenrodSignature {
        profile.signature_of(self.basis_element_from_index(degree, idx))
    }
}

// ---------------------------------------------------------------------------
// The generic signature-filtration resolution engine (Algorithm 2), over any
// `PAlgebra`. Instantiated as `SignatureResolution<CTauOpAlgebra>` (motivic) and
// `SignatureResolution<MilnorAlgebra>` (classical).
// ---------------------------------------------------------------------------

/// One bidegree's shrink record: `dim E₀C_s` vs `dim C_s` (the space the homology
/// work runs in vs the full free module), and the `B` used. The ratio `dim C /
/// dim E₀C` is Nassau's speedup lever.
#[derive(Clone, Debug)]
pub struct ShrinkRecord {
    pub s: i32,
    pub t: i32,
    pub b: String,
    pub dim_c: usize,
    pub dim_e0: usize,
}

/// A standalone minimal resolution of the trivial module `k` over a [`PAlgebra`],
/// computed by Nassau's **signature-filtration** Algorithm 2. Additive and
/// decoupled from the generic engine: it carries its own free modules and
/// differentials, and its generator ranks are validated against the generic
/// engine.
///
/// At each bidegree `(s, t)` with `s ≥ 2` it chooses the largest applicable
/// profile ([`PAlgebra::optimal_profile`]); where none applies it uses the trivial
/// profile, which makes the algorithm's zero-signature block the whole module —
/// i.e. the ordinary (generic) step. So the composite is **always correct**, and
/// the signature shortcut is used only where provably applicable.
///
/// Instantiated as `SignatureResolution<CTauOpAlgebra>` (motivic $A_C/\tau$) or
/// `SignatureResolution<MilnorAlgebra>` (classical mod-$2$ Steenrod algebra).
pub struct SignatureResolution<A: PAlgebra> {
    algebra: Arc<A>,
    target: Arc<FiniteChainComplex<FDModule<A>>>,
    zero_module: Arc<FreeModule<A>>,
    modules: Vec<Arc<FreeModule<A>>>,
    differentials: Vec<Arc<FreeModuleHomomorphism<FreeModule<A>>>>,
    chain_maps: Vec<Arc<FreeModuleHomomorphism<FDModule<A>>>>,
    max_s: i32,
    shrink: RefCell<Vec<ShrinkRecord>>,
    plain_steps: Cell<usize>,
    sig_steps: Cell<usize>,
}

impl<A: PAlgebra> SignatureResolution<A> {
    /// A resolution of the trivial module `k = F₂` over the given algebra.
    pub fn new(algebra: Arc<A>) -> Self {
        let module = Arc::new(FDModule::new(
            Arc::clone(&algebra),
            "k".to_string(),
            BiVec::from_vec(0, vec![1]),
        ));
        let target = Arc::new(FiniteChainComplex::<FDModule<A>>::ccdz(module));
        let zero_module = Arc::new(FreeModule::new(
            Arc::clone(&algebra),
            "F_{-1}".to_string(),
            0,
        ));
        Self {
            algebra,
            target,
            zero_module,
            modules: Vec::new(),
            differentials: Vec::new(),
            chain_maps: Vec::new(),
            max_s: 0,
            shrink: RefCell::new(Vec::new()),
            plain_steps: Cell::new(0),
            sig_steps: Cell::new(0),
        }
    }

    /// Prepare `modules`, `differentials`, and `chain_maps` up to filtration `max_s`.
    fn extend_through_degree(&mut self, max_s: i32) {
        self.max_s = max_s;
        for i in 0..=max_s {
            self.modules.push(Arc::new(FreeModule::new(
                Arc::clone(&self.algebra),
                format!("F{i}"),
                0,
            )));
        }
        // d_0 : F_0 → 0.
        self.differentials
            .push(Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[0]),
                Arc::clone(&self.zero_module),
                0,
            )));
        for i in 1..=max_s as usize {
            self.differentials
                .push(Arc::new(FreeModuleHomomorphism::new(
                    Arc::clone(&self.modules[i]),
                    Arc::clone(&self.modules[i - 1]),
                    0,
                )));
        }
        for i in 0..=max_s as usize {
            self.chain_maps.push(Arc::new(FreeModuleHomomorphism::new(
                Arc::clone(&self.modules[i]),
                self.target.module(i as i32),
                0,
            )));
        }
    }

    /// The number of generators (`Ext` rank) in bidegree `(s, t)`.
    pub fn number_of_gens_in_bidegree(&self, s: i32, t: i32) -> usize {
        self.modules[s as usize].number_of_gens_in_degree(t)
    }

    /// The recorded per-bidegree shrink factors (`dim C / dim E₀C`).
    pub fn shrink_records(&self) -> Vec<ShrinkRecord> {
        self.shrink.borrow().clone()
    }

    /// How many `(s, t)` steps used a nontrivial signature shortcut, and how many
    /// were plain steps (`optimal_profile` returned the trivial `B` — below every
    /// vanishing line). There is no runtime fallback.
    pub fn stats(&self) -> (usize, usize) {
        (self.sig_steps.get(), self.plain_steps.get())
    }

    fn add_generators(&self, s: i32, t: i32, num: usize) {
        self.modules[s as usize].add_generators(t, num, None);
    }

    /// Resolve `k` through stem `n = max_t - max_s`, i.e. all `(s, t)` with
    /// `s ≤ max_s` and `t ≤ max_t`.
    pub fn compute_through_stem(&mut self, max_s: i32, max_t: i32) {
        self.extend_through_degree(max_s);
        self.algebra.compute_basis(max_t);
        for t in 0..=max_t {
            for s in 0..=max_s {
                // (s, t) depends on (s-1, t) [done: smaller s this t] and (s, t-1)
                // [done: previous t]. Serial nested loop respects both.
                self.step(s, t);
            }
        }
    }

    fn step(&self, s: i32, t: i32) {
        self.modules[s as usize].compute_basis(t);
        if s > 0 {
            self.modules[s as usize - 1].compute_basis(t);
        }
        if s == 0 {
            self.step0(t);
        } else if s == 1 {
            self.step1(t);
        } else {
            // `optimal_profile` returns a provably-applicable B (Thm 3.1) or the
            // trivial profile `A(-1) = k` — the plain step, which always works. So
            // the step never needs a runtime fallback.
            let b = self.algebra.optimal_profile(s, t);
            if b.is_trivial() {
                self.plain_steps.set(self.plain_steps.get() + 1);
            } else {
                self.sig_steps.set(self.sig_steps.get() + 1);
            }
            self.step_general(s, t, &b);
        }
    }

    /// s = 0: cover the trivial module in degree 0 (one generator there).
    fn step0(&self, t: i32) {
        self.zero_module.extend_by_zero(t);
        let source = &self.modules[0];
        let cc = self.target.module(0);
        let chain_map = &self.chain_maps[0];
        let d = &self.differentials[0];

        source.compute_basis(t);
        cc.compute_basis(t);
        let source_dim = source.dimension(t);
        let target_dim = cc.dimension(t);

        if target_dim == 0 {
            source.extend_by_zero(t);
            chain_map.extend_by_zero(t);
        } else {
            let mut matrix = AugmentedMatrix::<2>::new_with_capacity(
                self.algebra.prime(),
                source_dim,
                &[target_dim, source_dim],
                source_dim + target_dim,
                0,
            );
            chain_map.get_matrix(matrix.segment(0, 0), t);
            matrix.segment(1, 1).add_identity();
            matrix.row_reduce();
            let num_new_gens = matrix.extend_to_surjection(0, target_dim, 0).len();
            self.add_generators(0, t, num_new_gens);
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

    /// s = 1: generators map onto `ker(F_0 → k)`.
    fn step1(&self, t: i32) {
        let source = &self.modules[1];
        let target = &self.modules[0];
        let cc = self.target.module(0);

        let source_dim = source.dimension(t);
        let target_dim = target.dimension(t);

        let p = self.algebra.prime();
        let mut matrix = AugmentedMatrix::<2>::new(p, target_dim, [cc.dimension(t), target_dim]);
        self.chain_maps[0].get_matrix(matrix.segment(0, 0), t);
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
        self.differentials[1].get_matrix(matrix.segment(0, 0), t);
        matrix.segment(1, 1).add_identity();
        matrix.row_reduce();
        let num_new_gens = matrix.extend_image(0, target_dim, &desired_image, 0).len();
        self.add_generators(1, t, num_new_gens);
        self.differentials[1].add_generators_from_matrix_rows(
            t,
            matrix
                .segment(0, 0)
                .row_slice(source_dim, source_dim + num_new_gens),
        );
        self.chain_maps[1].extend_by_zero(t);
    }

    /// The Algorithm 2 inductive step at `(s, t)` with subalgebra `b`. Returns
    /// `false` if the correction sweep left `d² ≠ 0` (signature order insufficient
    /// at this bidegree), signalling the caller to fall back to a plain step.
    fn step_general(&self, s: i32, t: i32, b: &A::Profile) {
        let alg = &*self.algebra;
        let p = alg.prime();
        let target = &self.modules[s as usize - 1];
        let next = &self.modules[s as usize - 2];
        next.compute_basis(t);

        let zero_sig = b.zero_signature();
        let target_dim = target.dimension(t);
        let target_mask = signature_mask(alg, b, target, t, &zero_sig);
        let next_mask = signature_mask(alg, b, next, t, &zero_sig);

        // Shrink record: E₀ (zero-sig) masked dim vs full dim of C_{s-1}.
        if !b.is_trivial() {
            self.shrink.borrow_mut().push(ShrinkRecord {
                s,
                t,
                b: b.name(),
                dim_c: target_dim,
                dim_e0: target_mask.len(),
            });
        }

        // Kernel of d_{s-1} in the zero-signature block.
        let full_matrix = self.differentials[s as usize - 1].get_partial_matrix(t, &target_mask);
        let mut masked =
            AugmentedMatrix::new(p, target_mask.len(), [next_mask.len(), target_mask.len()]);
        masked.segment(0, 0).add_masked(&full_matrix, &next_mask);
        masked.segment(1, 1).add_identity();
        masked.row_reduce();
        let kernel = masked.compute_kernel();

        // Image of d_s in the zero-signature block; new generators cover the rest.
        let mut n = signature_matrix(alg, b, &self.differentials[s as usize], t, &zero_sig);
        n.row_reduce();
        let next_row = n.rows();
        let num_new_gens = n.extend_image(0, n.columns(), &kernel, 0).len();

        if t < s {
            assert_eq!(num_new_gens, 0, "adding generators at ({s}, {t})");
        }
        self.add_generators(s, t, num_new_gens);

        let mut xs = vec![FpVector::new(p, target_dim); num_new_gens];
        let mut dxs = vec![FpVector::new(p, next.dimension(t)); num_new_gens];
        for ((x, x_masked), dx) in xs.iter_mut().zip(n.iter().skip(next_row)).zip(&mut dxs) {
            // x := x_masked scattered into the full C_{s-1} basis (coeff 1: this
            // *defines* x). dx := d_{s-1}(x) = Σ_i x_masked[i]·d(target_mask[i]).
            x.as_slice_mut().add_unmasked(x_masked, 1, &target_mask);
            for (i, c) in x_masked.iter_nonzero() {
                dx.as_slice_mut().add(full_matrix.row(i), c);
            }
        }

        // Signature-ordered corrections.
        let mut tmask: Vec<usize> = Vec::new();
        let mut nmask: Vec<usize> = Vec::new();
        for signature in b.iter_signatures(t) {
            tmask.clear();
            nmask.clear();
            tmask.extend(signature_mask(alg, b, target, t, &signature));
            nmask.extend(signature_mask(alg, b, next, t, &signature));

            let full_matrix = self.differentials[s as usize - 1].get_partial_matrix(t, &tmask);
            let mut masked = AugmentedMatrix::new(p, tmask.len(), [nmask.len(), tmask.len()]);
            masked.segment(0, 0).add_masked(&full_matrix, &nmask);
            masked.segment(1, 1).add_identity();
            masked.row_reduce();

            let qi = masked.compute_quasi_inverse();

            let mut scratch = FpVector::new(p, tmask.len());
            let mut dx_masked = FpVector::new(p, nmask.len());
            for (x, dx) in xs.iter_mut().zip(&mut dxs) {
                // The E_signature (masked) component of dx over C_{s-2}.
                dx_masked.set_to_zero();
                for (i, &v) in nmask.iter().enumerate() {
                    dx_masked.set_entry(i, dx.entry(v));
                }
                // scratch := −f where d(f) = dx on this block (quasi-inverse, scaled
                // by −1 ≡ p−1 so the `+=` updates below realise x −= f, dx −= d(f)).
                // `apply` does the Fₚ arithmetic (scales each preimage row by its
                // coefficient and skips non-pivot columns).
                scratch.set_to_zero();
                qi.apply(scratch.as_slice_mut(), p - 1, dx_masked.as_slice());
                for (i, c) in scratch.iter_nonzero() {
                    x.add_basis_element(tmask[i], c);
                    dx.as_slice_mut().add(full_matrix.row(i), c);
                }
            }
        }

        // d² = 0. This must hold: the correction sweep drives every dx to zero
        // whenever B is applicable (which `optimal_profile` guarantees — it returns
        // only a provably-applicable B or the trivial one, and the trivial B is the
        // plain step that always works). So there is no runtime fallback; a failure
        // here is a genuine bug (a bad vanishing line or signature order), surfaced
        // rather than silently papered over.
        assert!(
            dxs.iter().all(|dx| dx.is_zero()),
            "d² ≠ 0 at (s={s}, t={t}) with B={} — B applied outside its vanishing region, or the \
             signature order is not a valid linear extension",
            b.name(),
        );

        self.differentials[s as usize].add_generators_from_rows(t, xs);
        self.chain_maps[s as usize].extend_by_zero(t);
    }
}

impl SignatureResolution<CTauOpAlgebra> {
    /// A resolution of `k` over the motivic $A_C/\tau$ (via its opposite algebra).
    pub fn motivic() -> Self {
        Self::new(Arc::new(CTauOpAlgebra::new()))
    }
}

impl Default for SignatureResolution<CTauOpAlgebra> {
    fn default() -> Self {
        Self::motivic()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a0_is_e_q0() {
        let b = MotivicSubalgebra::a_n(0);
        assert_eq!(b.q_len(), 1);
        assert!(b.p_profile().is_empty());
        // dim E(Q_0) = 2 (1, Q_0).
        assert_eq!(b.dimension(), 2);
        assert_eq!(b.to_string(), "A(0)");
    }

    #[test]
    fn a1_dimension_is_8() {
        let b = MotivicSubalgebra::a_n(1);
        // A(1) = E(Q_0, Q_1) ⊗ F₂[ξ_1]/(ξ_1^2): dim 4 · 2 = 8.
        assert_eq!(b.dimension(), 8);
        assert_eq!(b.to_string(), "A(1)");
    }

    #[test]
    fn a2_dimension_is_64() {
        let b = MotivicSubalgebra::a_n(2);
        // A(2): exterior E(Q_0,Q_1,Q_2) dim 8; polynomial ξ_1<2^2, ξ_2<2^1 → 4·2 = 8. Total 64.
        assert_eq!(b.dimension(), 64);
    }

    #[test]
    fn e_q0_q1_signatures() {
        // B = E(Q_0, Q_1): q_len = 2, no polynomial part.
        let b = MotivicSubalgebra::new(vec![], 2);
        assert_eq!(b.dimension(), 4);
        // Nonzero signatures: Q_0 (deg 1), Q_1 (deg 3), Q_0Q_1 (deg 4).
        let sigs = b.iter_signatures(10);
        assert_eq!(sigs.len(), 3);
        // Degrees present.
        let mut degs: Vec<i32> = sigs.iter().map(|s| b.signature_degree(s)).collect();
        degs.sort();
        assert_eq!(degs, vec![1, 3, 4]);
    }

    #[test]
    fn signatures_partition_the_basis() {
        // Every basis element of A_C/τ in a range has exactly one signature, and
        // the signatures with elements up to that degree (plus zero) cover all of
        // S_B; the count of elements per signature is uniform (= a coset of B).
        let alg = CTauOpAlgebra::new();
        let b = MotivicSubalgebra::a_n(1);
        alg.compute_basis(20);
        for t in 0..=20 {
            for (e, r) in basis_monomials(&alg, t) {
                // The computed signature is self-consistent with `has_signature`.
                let sig = b.signature_of(e, &r);
                assert!(b.has_signature(e, &r, &sig));
                // Signature components are in range.
                assert_eq!(sig.q, sig.q & ((1 << b.q_len()) - 1));
            }
        }
    }

    #[test]
    fn product_raises_signature() {
        // The linchpin of the correction loop: over A_C/τ the *right* factor's
        // signature is a floor (left multiplication preserves sig ≥ R) — the
        // opposite handedness to the classical polynomial case, which is why the
        // engine resolves over the opposite algebra. So in a·b, every output term
        // has signature ≥ sig(b) in the degree order. Checked against the real
        // A_C/τ product for the B the engine uses (verified acyclic by `sig_diag`).
        use fp::vector::FpVector;
        let alg = CTauAlgebra::new();
        alg.compute_basis(24);
        for b in [
            MotivicSubalgebra::a_n(0),
            MotivicSubalgebra::a_n(1),
            MotivicSubalgebra::new(vec![], 2),
            MotivicSubalgebra::new(vec![], 3),
        ] {
            for ta in 0..=10 {
                for ia in 0..alg.dimension(ta) {
                    for tb in 0..=(10 - ta) {
                        for ib in 0..alg.dimension(tb) {
                            // Right factor b determines the floor.
                            let (eb, rb) = alg.engine().basis_element(tb, ib).clone();
                            let key_b = b.order_degree_key(&b.signature_of(eb, &rb));
                            let mut out = FpVector::new(TWO, alg.dimension(ta + tb));
                            alg.multiply_basis_elements(out.as_slice_mut(), 1, ta, ia, tb, ib);
                            for (idx, _) in out.iter_nonzero() {
                                let (e, r) = alg.engine().basis_element(ta + tb, idx);
                                let sig = b.signature_of(*e, r);
                                assert!(
                                    b.order_degree_key(&sig) >= key_b,
                                    "product lowered signature: term of sig {:?} < sig(right={}) \
                                     for B={b}",
                                    sig,
                                    alg.basis_element_to_string(tb, ib),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn signatures_sum_to_full_dimension() {
        // Σ_R dim E_R A = dim A: the signatures partition the basis. Also every
        // nonzero-signature bucket plus E_0 accounts for every basis element.
        let alg = CTauOpAlgebra::new();
        let b = MotivicSubalgebra::a_n(1);
        alg.compute_basis(24);
        for t in 0..=24 {
            let mut sigs = b.iter_signatures(t);
            sigs.push(b.zero_signature());
            let mut counted = 0usize;
            for (e, r) in basis_monomials(&alg, t) {
                let sig = b.signature_of(e, &r);
                // The signature of every basis element is one we enumerated (its
                // representative degree ≤ t since sig components are bounded by the
                // element itself).
                assert!(
                    sigs.contains(&sig),
                    "unenumerated signature {sig:?} at t={t}"
                );
                counted += 1;
            }
            assert_eq!(counted, alg.dimension(t));
        }
    }

    #[test]
    fn optimal_for_picks_larger_b_higher_up() {
        // Far below the vanishing line only F₂ (a plain step); high t admits A(n).
        assert!(MotivicSubalgebra::optimal_for(5, 3).is_trivial());
        let big = MotivicSubalgebra::optimal_for(2, 200);
        assert!(
            !big.is_trivial(),
            "expected a nontrivial B high up, got {big}"
        );
    }

    #[test]
    fn signature_resolution_matches_generic() {
        // The payoff correctness gate (M5 step 1): the signature engine's per-(s,t)
        // generator ranks equal the generic engine's, bidegree for bidegree, on a
        // small box. Also checks d²=0 held everywhere (no fallback was forced by a
        // failed correction sweep — only by B being trivial).
        use std::sync::Arc;

        use algebra::module::FDModule;
        use bivec::BiVec;
        use sseq::coordinates::Bidegree;

        use crate::{
            chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
            resolution::Resolution,
        };

        let max_s = 12;
        let max_t = 32;

        // Generic (left) engine — the reference.
        let galg = Arc::new(CTauAlgebra::new());
        let gmod = Arc::new(FDModule::new(
            Arc::clone(&galg),
            "k".to_string(),
            BiVec::from_vec(0, vec![1]),
        ));
        let gcc = Arc::new(FiniteChainComplex::<FDModule<CTauAlgebra>>::ccdz(gmod));
        let gres = Resolution::new(gcc);
        gres.compute_through_stem(Bidegree::s_t(max_s, max_t));

        // Signature engine.
        let mut sres = SignatureResolution::motivic();
        sres.compute_through_stem(max_s, max_t);

        // Compare over the bidegrees the generic engine actually computed (a stem
        // region); the signature engine's full rectangle covers all of them.
        let mut checked = 0usize;
        for b in gres.iter_stem() {
            if b.t() > max_t {
                continue;
            }
            let want = gres.number_of_gens_in_bidegree(b);
            let got = sres.number_of_gens_in_bidegree(b.s(), b.t());
            assert_eq!(
                got,
                want,
                "rank mismatch at (s={}, t={}): signature={got} generic={want}",
                b.s(),
                b.t()
            );
            checked += 1;
        }
        assert!(checked > 100, "too few bidegrees checked: {checked}");
    }

    #[test]
    fn classical_signature_resolution_matches_generic() {
        // The unification payoff: the SAME generic engine, instantiated over the
        // classical MilnorAlgebra via `impl PAlgebra for MilnorAlgebra`, reproduces
        // the classical Adams E₂ — validated rank-for-rank against the generic
        // minimal-resolution engine over the same algebra. No opposite algebra:
        // the standard Milnor basis is right-stable.
        use std::sync::Arc;

        use algebra::{milnor_algebra::MilnorAlgebra, module::FDModule};
        use bivec::BiVec;
        use fp::prime::TWO;
        use sseq::coordinates::Bidegree;

        use crate::{
            chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
            resolution::Resolution,
        };

        let max_s = 10;
        let max_t = 20;

        // Generic reference over MilnorAlgebra.
        let galg = Arc::new(MilnorAlgebra::new(TWO, false));
        let gmod = Arc::new(FDModule::new(
            Arc::clone(&galg),
            "k".to_string(),
            BiVec::from_vec(0, vec![1]),
        ));
        let gcc = Arc::new(FiniteChainComplex::<FDModule<MilnorAlgebra>>::ccdz(gmod));
        let gres = Resolution::new(gcc);
        gres.compute_through_stem(Bidegree::s_t(max_s, max_t));

        // The generic signature engine over the classical algebra.
        let mut sres = SignatureResolution::new(Arc::new(MilnorAlgebra::new(TWO, false)));
        sres.compute_through_stem(max_s, max_t);

        let mut checked = 0usize;
        for b in gres.iter_stem() {
            if b.t() > max_t {
                continue;
            }
            let want = gres.number_of_gens_in_bidegree(b);
            let got = sres.number_of_gens_in_bidegree(b.s(), b.t());
            assert_eq!(
                got,
                want,
                "classical rank mismatch at (s={}, t={}): signature={got} generic={want}",
                b.s(),
                b.t()
            );
            checked += 1;
        }
        assert!(checked > 50, "too few bidegrees checked: {checked}");
        // Sanity spot-check: h_0, h_1, h_2, h_3 are the degree-(1, 2^i) classes.
        assert_eq!(sres.number_of_gens_in_bidegree(1, 1), 1); // h_0
        assert_eq!(sres.number_of_gens_in_bidegree(1, 2), 1); // h_1
        assert_eq!(sres.number_of_gens_in_bidegree(1, 4), 1); // h_2
    }

    #[test]
    fn tmf_resolution_over_a2_matches_generic() {
        // tmf: Ext_{A(2)}(F₂, F₂) is the tmf Adams E₂. Resolve k over A(2) (a
        // finite MilnorAlgebra ambient) with the signature engine and check it
        // against the generic engine rank-for-rank, plus a few certain structural
        // facts of the tmf chart — including the *absence* of h₃ (Sq⁸ ∉ A(2)),
        // which distinguishes A(2) from the full Steenrod algebra.
        use std::sync::Arc;

        use algebra::{
            milnor_algebra::{MilnorAlgebra, MilnorProfile},
            module::FDModule,
        };
        use bivec::BiVec;
        use fp::prime::TWO;
        use sseq::coordinates::Bidegree;

        use crate::{
            chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
            resolution::Resolution,
        };

        // A(2): ξ_1 < 2³, ξ_2 < 2², ξ_3 < 2¹ (dim 64).
        let a2 = || {
            MilnorAlgebra::new_with_profile(
                TWO,
                MilnorProfile {
                    q_part: !0,
                    p_part: vec![3, 2, 1],
                    truncated: true,
                },
                false,
            )
        };

        let max_s = 14;
        let max_t = 44;

        let galg = Arc::new(a2());
        let gmod = Arc::new(FDModule::new(
            Arc::clone(&galg),
            "k".to_string(),
            BiVec::from_vec(0, vec![1]),
        ));
        let gcc = Arc::new(FiniteChainComplex::<FDModule<MilnorAlgebra>>::ccdz(gmod));
        let gres = Resolution::new(gcc);
        gres.compute_through_stem(Bidegree::s_t(max_s, max_t));

        let mut sres = SignatureResolution::new(Arc::new(a2()));
        sres.compute_through_stem(max_s, max_t);

        let mut checked = 0usize;
        for b in gres.iter_stem() {
            if b.t() > max_t {
                continue;
            }
            let want = gres.number_of_gens_in_bidegree(b);
            let got = sres.number_of_gens_in_bidegree(b.s(), b.t());
            assert_eq!(
                got,
                want,
                "tmf rank mismatch at (s={}, n={}): signature={got} generic={want}",
                b.s(),
                b.n()
            );
            checked += 1;
        }
        assert!(checked > 80, "too few bidegrees checked: {checked}");

        // Certain tmf E₂ facts (s = filtration, stem n = t − s):
        let gens = |s: i32, n: i32| sres.number_of_gens_in_bidegree(s, n + s);
        assert_eq!(gens(0, 0), 1, "unit"); //  1
        assert_eq!(gens(1, 0), 1, "h_0"); //  h_0 ∈ Ext^{1,1}
        assert_eq!(gens(1, 1), 1, "h_1"); //  h_1 ∈ Ext^{1,2}
        assert_eq!(gens(1, 3), 1, "h_2"); //  h_2 ∈ Ext^{1,4}
        assert_eq!(gens(1, 7), 0, "no h_3"); //  Sq⁸ ∉ A(2) ⇒ no h_3
        assert_eq!(gens(1, 2), 0, "no class in (1, stem 2)");
        // The h_0-tower in stem 0: Ext^{s,s} = 1 for every s.
        for s in 0..=max_s {
            assert_eq!(gens(s, 0), 1, "h_0-tower rank at s={s}");
        }
    }

    #[test]
    fn odd_primary_signature_resolution_matches_generic() {
        // The odd-primary payoff: the SAME engine, over MilnorAlgebra at p=3 and
        // p=5, reproduces the odd-primary Adams E₂ rank-for-rank vs the generic
        // engine — and the signature shortcut actually fires (sig_steps > 0),
        // confirming the standard odd-p Milnor basis is right-stable (no opposite
        // needed) and the Fₚ correction arithmetic is correct.
        use std::sync::Arc;

        use algebra::{milnor_algebra::MilnorAlgebra, module::FDModule};
        use bivec::BiVec;
        use fp::prime::ValidPrime;
        use sseq::coordinates::Bidegree;

        use crate::{
            chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
            resolution::Resolution,
        };

        for p in [3u32, 5] {
            let prime = ValidPrime::new(p);
            let max_s = 6;
            let max_t = if p == 3 { 60 } else { 80 };

            let galg = Arc::new(MilnorAlgebra::new(prime, false));
            let gmod = Arc::new(FDModule::new(
                Arc::clone(&galg),
                "k".to_string(),
                BiVec::from_vec(0, vec![1]),
            ));
            let gcc = Arc::new(FiniteChainComplex::<FDModule<MilnorAlgebra>>::ccdz(gmod));
            let gres = Resolution::new(gcc);
            gres.compute_through_stem(Bidegree::s_t(max_s, max_t));

            let mut sres = SignatureResolution::new(Arc::new(MilnorAlgebra::new(prime, false)));
            sres.compute_through_stem(max_s, max_t);

            let mut checked = 0usize;
            for b in gres.iter_stem() {
                if b.t() > max_t {
                    continue;
                }
                let want = gres.number_of_gens_in_bidegree(b);
                let got = sres.number_of_gens_in_bidegree(b.s(), b.t());
                assert_eq!(
                    got,
                    want,
                    "p={p} rank mismatch at (s={}, t={}): signature={got} generic={want}",
                    b.s(),
                    b.t()
                );
                checked += 1;
            }
            assert!(checked > 40, "p={p}: too few bidegrees checked: {checked}");
            let (sig_steps, _fallbacks) = sres.stats();
            assert!(
                sig_steps > 0,
                "p={p}: no signature shortcut fired — handedness or Fₚ arithmetic wrong"
            );
        }
    }

    #[test]
    fn opposite_algebra_matches_golden_total() {
        // The antipode gives (A_C/τ)^op ≅ A_C/τ, so resolving k over the opposite
        // algebra with the generic engine yields the *same* generator ranks as the
        // left engine — the golden fixture. Spot-check the total on a small box.
        use std::sync::Arc;

        use algebra::module::FDModule;
        use bivec::BiVec;
        use sseq::coordinates::Bidegree;

        use crate::{
            chain_complex::{ChainComplex, FiniteChainComplex, FreeChainComplex},
            resolution::Resolution,
        };

        let algebra = Arc::new(CTauOpAlgebra::new());
        let module = Arc::new(FDModule::new(
            Arc::clone(&algebra),
            "k".to_string(),
            BiVec::from_vec(0, vec![1]),
        ));
        let cc = Arc::new(FiniteChainComplex::<FDModule<CTauOpAlgebra>>::ccdz(module));
        let res = Resolution::new(cc);
        res.compute_through_stem(Bidegree::s_t(12, 32));

        let total: usize = res
            .iter_stem()
            .map(|b| res.number_of_gens_in_bidegree(b))
            .sum();
        // Same total as the generic (left) engine on the n≤20, s≤12 box (M0 gave
        // 130 there); this pins the antipode-invariance of the ranks.
        assert_eq!(total, 130, "opposite-algebra ranks differ from the golden");
    }

    #[test]
    fn iter_signatures_is_ordered_and_deduped() {
        let b = MotivicSubalgebra::a_n(2);
        let sigs = b.iter_signatures(30);
        // Strictly increasing in the admissible order, no dupes.
        for w in sigs.windows(2) {
            assert!(
                b.order_degree_key(&w[0]) < b.order_degree_key(&w[1]),
                "not strictly increasing: {:?} !< {:?}",
                w[0],
                w[1]
            );
        }
        // None is the zero signature.
        assert!(!sigs.contains(&b.zero_signature()));
    }
}
