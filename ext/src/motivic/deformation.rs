//! The deformation (algebraic Novikov / τ-Bockstein) spectral sequence: assemble
//! the trigraded [`Sseq`] whose $E_1 = \mathrm{Ext}_{A_C/\tau}$ and whose $d_r$ are
//! the τ-Bockstein differentials read off the lifted δ. Inverting τ gives the
//! classical Adams $E_2$ ([`MotivicResolution::free_rank`]); the finite-page deaths
//! are the motivic τ-torsion.

use std::collections::{BTreeMap, HashMap};

use fp::{matrix::Matrix, prime::TWO, vector::FpVector};
use once::MultiIndexed;
use sseq::{
    Product, Sseq, SseqProfile,
    coordinates::{Bidegree, BidegreeGenerator, degree::MultiDegree, element::MultiDegreeElement},
};

use super::{Gen, MotivicResolution};

/// The direction of the **deformation spectral sequence** (the algebraic Novikov
/// / $\tau$-Bockstein SS), trigraded by (stem $n$, Adams filtration $s$, weight
/// $w$). Each $d_r$ carries δ's fixed $(n, s) \mapsto (n+1, s-1)$ shift — δ lowers
/// Novikov filtration at fixed internal degree, so the stem rises — and jumps the
/// weight by $r$ (the $\tau$-power). $E_1 = \Ext_{A_C/\tau}$; inverting $\tau$
/// ($w \to \infty$) gives $E_\infty = $ classical $\Ext_A$, and finite-page deaths
/// are the motivic $\tau$-torsion.
pub struct Deformation;

impl SseqProfile<3> for Deformation {
    const MIN_R: i32 = 1;

    fn profile(r: i32, b: MultiDegree<3>) -> MultiDegree<3> {
        b + MultiDegree::from([1, -1, r])
    }

    fn profile_inverse(r: i32, b: MultiDegree<3>) -> MultiDegree<3> {
        b + MultiDegree::from([-1, 1, -r])
    }

    fn differential_length(offset: MultiDegree<3>) -> i32 {
        offset.coords()[2] // the weight component
    }
}

impl MotivicResolution {
    /// Group generators by their multidegree `(n, s, w)`. Returns the per-multidegree
    /// generator-index lists (a generator's position in its list is its Sseq
    /// coordinate) and the reverse map `Gen ↦ (multidegree, position)`.
    fn sseq_grouping(
        &self,
    ) -> (
        HashMap<[i32; 3], Vec<usize>>,
        HashMap<Gen, (MultiDegree<3>, usize)>,
    ) {
        let mut groups: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
        let mut pos: HashMap<Gen, (MultiDegree<3>, usize)> = HashMap::new();
        for s in 0..=self.max_s() {
            let t_max = self.compute.n() + s;
            for t in 0..=t_max {
                for idx in 0..self.num_gens(s, t) {
                    let g = Gen { s, t, idx };
                    if let Some(&w) = self.weights.get(&g) {
                        let key = [t - s, s, w];
                        let list = groups.entry(key).or_default();
                        pos.insert(g, (MultiDegree::from(key), list.len()));
                        list.push(idx);
                    }
                }
            }
        }
        (groups, pos)
    }

    /// A [`Product`] on the deformation SS for each requested Cτ generator `a`
    /// (given as `(bidegree, index)`): multiplication by `a`, taken from
    /// [`ExtAlgebra::multiply_into`] on the mod-τ resolution and split into weight
    /// blocks for the trigrading. Feeding one to [`Sseq::multiply`] applies it on
    /// any page — the Cτ ring on $E_1$, the motivic Adams $E_2$ ring on $E_\infty$.
    pub fn deformation_products(&self, gens: &[(Bidegree, usize)]) -> Vec<Product<3>> {
        let ext = self.ext();
        let (groups, _) = self.sseq_grouping();
        let empty = Vec::new();
        gens.iter()
            .map(|&(a_deg, a_idx)| {
                let a_gen = Gen {
                    s: a_deg.s(),
                    t: a_deg.t(),
                    idx: a_idx,
                };
                let a_w = self.weights[&a_gen];
                let a_elem = ext.generator(BidegreeGenerator::new(a_deg, a_idx));
                let matrices = MultiIndexed::new();
                for (&[n, s, w], src_group) in &groups {
                    let Some(full) = ext.multiply_into(&a_elem, Bidegree::n_s(n, s)) else {
                        continue;
                    };
                    let tgt = groups
                        .get(&[n + a_deg.n(), s + a_deg.s(), w + a_w])
                        .unwrap_or(&empty);
                    // Row `i` = a · (source generator i), restricted to the
                    // weight-`(w + a_w)` target generators, in group coordinates.
                    let rows: Vec<Vec<u32>> = src_group
                        .iter()
                        .map(|&raw_i| {
                            tgt.iter()
                                .map(|&raw_j| full.row(raw_i).entry(raw_j))
                                .collect()
                        })
                        .collect();
                    matrices.insert(MultiDegree::from([n, s, w]), Matrix::from_vec(TWO, &rows));
                }
                Product {
                    b: MultiDegree::from([a_deg.n(), a_deg.s(), a_w]),
                    left: true,
                    matrices,
                }
            })
            .collect()
    }

    /// The deformation spectral sequence as an [`Sseq`], trigraded by $(n, s, w)$
    /// ([`Deformation`]). $E_1 = \Ext_{A_C/\tau}$ — the mod-τ generators, grouped by
    /// weight — and **every** $d_r$ is read straight off the graded Smith normal form
    /// of the outgoing δ ([`MotivicResolution::motivic_differentials`]): a pivot
    /// $\tau^r$ at $(s, t)$ is a length-$r$ differential whose source and target are
    /// the weight-pure ($\tau^0$) parts of the homogeneous SNF combinations. This is
    /// the **same** set of invariant factors [`tau_module`](Self::tau_module) reads as
    /// the τ-torsion, so the module structure and the SS share one source of truth —
    /// there is no separate τ-Bockstein zig-zag. [`Sseq::update`] then quotients each
    /// page into its `Subquotient`s, and the products (via [`Sseq::multiply`]) build
    /// on the result.
    ///
    /// [`page_data`]: Sseq::page_data
    pub fn deformation_sseq(&self) -> &Sseq<3, Deformation> {
        self.deformation
            .get_or_init(|| self.build_deformation_sseq())
    }

    #[tracing::instrument(
        skip(self),
        fields(max = %self.max, top_page = tracing::field::Empty, num_higher_diffs = tracing::field::Empty)
    )]
    fn build_deformation_sseq(&self) -> Sseq<3, Deformation> {
        let mut sseq = Sseq::<3, Deformation>::new(TWO);

        // Group generators by (n, s, w) — a generator's position within its group
        // is its coordinate in that multidegree.
        let (groups, pos) = self.sseq_grouping();
        for (&key, list) in &groups {
            sseq.set_dimension(MultiDegree::from(key), list.len());
        }

        // Permanent cycles: the s = 0 unit, and every generator with δ = ∅. A
        // δ-empty generator is never a differential source (δ SNF has no pivot on it),
        // so it survives to E_∞.
        for (&g, &(deg, p)) in &pos {
            if g.s == 0 || self.delta(g).is_empty() {
                let mut source = FpVector::new(TWO, sseq.dimension(deg));
                source.set_entry(p, 1);
                sseq.add_permanent_class(&MultiDegreeElement::new(deg, source));
            }
        }

        // Every d_r, read straight off the graded δ SNF ([`Self::motivic_differentials`])
        // — the SAME invariant factors `tau_module` reads as torsion, now carrying
        // their source/target vectors. No τ-Bockstein zig-zag: a pivot τ^r at (s, t)
        // is a length-r differential from a weight-pure source at (n, s, w) to a
        // weight-pure target at (n+1, s-1, w+r) (the τ⁰ parts of the homogeneous SNF
        // combinations). Only report-box sources (stem n ≤ max.n()) are added — their
        // δ-targets (stem n+1 ≤ compute.n()) are resolved, and every differential
        // touching a report degree has a report-box source, so the report E_∞ is
        // exact; margin sources would reach the unresolved stem max.n+2.
        //
        // Differentials are grouped by length and applied in ascending r with an
        // `update` between pages, so each d_r sees the correct E_r subquotient. A
        // source already killed on an earlier page reduces to zero and is dropped by
        // `add_differential` (which returns `false`); by δ² = 0 a genuine d_r source
        // (a non-cycle of δ_s) is never such a boundary, so nothing real is lost.
        let group_vec = |key: [i32; 3], gens: &[usize]| -> FpVector {
            let list = groups.get(&key).map(Vec::as_slice).unwrap_or(&[]);
            let mut v = FpVector::new(TWO, list.len());
            for &gi in gens {
                let p = list
                    .iter()
                    .position(|&x| x == gi)
                    .expect("differential generator lies in its own weight group");
                v.set_entry(p, 1);
            }
            v
        };
        let mut by_order: BTreeMap<i32, Vec<(MultiDegree<3>, FpVector, FpVector)>> = BTreeMap::new();
        for s in 1..=self.max_s() {
            for t in 0..=(self.max.n() + s) {
                let n = t - s;
                if n < 0 || n > self.max.n() {
                    continue;
                }
                for (r, w, source_gens, target_gens) in self.motivic_differentials(s, t) {
                    let src_deg = MultiDegree::from([n, s, w]);
                    let source = group_vec([n, s, w], &source_gens);
                    let target = group_vec([n + 1, s - 1, w + r], &target_gens);
                    by_order.entry(r).or_default().push((src_deg, source, target));
                }
            }
        }

        let mut top_page = 1;
        let mut num_higher_diffs = 0usize;
        for (r, list) in by_order {
            for (src_deg, source, target) in list {
                let added = sseq.add_differential(
                    r,
                    &MultiDegreeElement::new(src_deg, source),
                    target.as_slice(),
                );
                if added && r >= 2 {
                    num_higher_diffs += 1;
                }
            }
            sseq.update();
            top_page = top_page.max(r + 1);
        }
        let span = tracing::Span::current();
        span.record("top_page", top_page);
        span.record("num_higher_diffs", num_higher_diffs);

        sseq
    }
}
