# The secondary cross term of P ⊗ M from sphere data

Setting: `P` the sphere's minimal resolution with its secondary homotopy `H` (and composites),
`M` a finite module, `Q = P ⊗ M` the untwisted tensor resolution (sseq's `TensorResolution`,
whose `δ_Q` coefficient is `[c](χ(a)·b)`). The Adams `d₂` of anything with cohomology `M` is
`Z ↦ Z∘H_Q` on cocycles `Z` of `Hom(Q)`, and only `H_Q`'s unit coefficients matter: one block
`U_{c→b}` per pair of cells, an operator `P_{s+2} → P_s`.

## The law

    U_{c→b}(g, x) = Σ_e [e](H g)_x · ⟨c, χ(e)·b⟩            (the antipode shear of H)
                  + Σ_y W_{c→b}([a](dg)_y, [a'](dy)_x)       (a path weight on d∘d paths)

where `W` is **any** solution of the twisted cobar equation (`twisted_w.rs.txt`)

    W(a1·a2, a3) + W(a1, a2·a3) + Σ_{b'} D_{c→b'}(a3) W_{b'→b}(a1, a2)
                                + Σ_{b'} W_{c→b'}(a2, a3) D_{b'→b}(a1)  =  O_{c→b}(a1, a2, a3)

with `D_{c→b'}(a) = ⟨c, χ(a)·b'⟩` and the obstruction
`O_{c→b}(a1, a2, a3) = ⟨c, χ(A(a1, σ(a2)σ(a3)))·b⟩`, `A` the Baues/Nassau A-map of sseq's
pair algebra (twos through κ, `Y_{k,l}` through `A_Y`). Solutions differ by things invisible on
cocycles modulo coboundaries (coboundaries vanish on the relations `Σ_y a_y ⊗ a'_y`, and the
`h0 h_j` class vanishes on the kernel classes).

## Two cells: where CARRY_j and Q_j come from

For `C(h_j)` the shear is `H:Sq^{2^j}` and `O` is computed by `amap_obstruction.rs.txt`:

| j | twos part | Y part | O |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | [Sq1\|Sq1\|Sq1] | [Sq1\|Sq1\|Sq1] | 0 (they cancel: `A(Sq1, Y_{01}) = Sq2 = κ(Sq1)·Sq2`) |
| 2 | [Sq1\|Sq2\|Sq2] | 0 | [ξ1\|ξ1²\|ξ1²] |
| 3 | [Sq1\|Sq4\|Sq4] | 0 | [ξ1\|ξ1⁴\|ξ1⁴] |

i.e. the cobar representative of `d₂(h_j) = h0 h_{j-1}²` (the Z/4 Bockstein of `[ξ1^{2^j}]`,
`binom(2^j, 2^{j-1}) ≡ 2 mod 4`, with the Y-correction killing `h0³` at `j = 1`). The fitted
`CARRY_j + Q_j` satisfies `δW = O` exactly (`cobar.py`); every solution solved from scratch
(`solveW.py`) gives the same agreement with the stem-58 ground truth (0 failures mod B2 + h_j E2).

## Computing the ground truth cheaply (`tensor_shear_lifting.patch`, against `cofiber-export`)

As a complex of vector spaces `P ⊗ M = ⊕_m P ⊗ m`, and `∂_Q = Ψ (∂_P ⊗ 1) Φ`, so
`TensorResolution::apply_quasi_inverse` lifts through `∂_Q` with the sphere's own (saved Nassau)
quasi-inverses: Φ splits the input per basis element of `M`, each piece is lifted in `P`, Ψ
reassembles. No row reduction of `∂_Q`. Under the P-stem truncation at `n`, generators whose P-stem
exceeds `n - 2` are untrusted (`ChainComplex::generator_trusted`) and get a zero homotopy.
`TENSOR_DENSE_QI=1` restores the dense path. Stem 40, seven modules: 6.5 s (dense: 28 s), equal on
cocycles mod coboundaries; stem 70, four multi-cell modules: 404 s, 2.1 GB (dense: over an hour for
the first module, unfinished).

## Multi-cell results (`cell_truth`, `cocfit.py`, cocycles mod coboundaries, χ-twisted δ_Q)

Ground truth to stem 40 (s ≤ 20), tested to stem 36:

| module | shear only (W = 0) | shear + W from the twisted equation |
|---|---|---|
| C2 ∧ Cη (split) | consistent | consistent (W = 0) |
| C2_eta (Sq1, then Sq2) | 5 failures | consistent |
| Joker | 11 failures | consistent |
| Cν, Cσ | fail | consistent |

The antipode matters: the plain shear `⟨c, e·b⟩` fails on C2_eta and the Joker. Without the
twisting terms the equations themselves are unsolvable for C2 ∧ Cν (6) and the Joker (3); with
them C2 ∧ Cν is solvable and the Joker has one contradictory equation, from triples
`(Sq1 | · | Sq1)` around `χ(P(3)) = P(3) + Q1`, which carries `x1 → x4` twice: Q reduces that
coefficient mod 2 before lifting, the P-level A-map does not. That is where M's own Z/4 lift
(its secondary structure θ) has to enter `O`.

At stem 66 (s ≤ 26, from the stem-70 truth) the derived `W` stays consistent for C2_eta (106
nonzero cross d₂ mod B), the Joker (109) and C2 ∧ Cη (204). For C2 ∧ Cν the equation leaves a
visible freedom: `W_{4→1}` may carry the cocycle `[ξ1²|ξ1²]` (h1², no attaching map between those
cells), and `twisted_w`'s free-variable choice takes it, which fails (82 places). Taking Cν's `W`
on both ν-blocks, `W_{4→1} = 0` and fitting only `W_{5→0}` is consistent (115). The equation fixes
`W` up to `Ext^{2,δ+1}` classes per block, which is the secondary-structure freedom θ; the rule
picking the representative for sseq's split structure is open.

## The secondary structure θ: normalization (`twisted_w` with `NORM=nopath`, `thetafit.sh`)

W has internal degree 1 as an End(M)-valued 2-cochain, so the homogeneous solutions of the twisted
equation modulo twisted coboundaries are `Ext^{2,1}_A(M, M)`: the torsor of secondary structures.
Fitting only the homogeneous directions (`COMBOS`, `thetafit.sh`) on C2 ∧ Cν: of 12 directions
exactly one is visible in the stem-66 data, it is `h1² ⊗ (x1 → x4)` (cells with no primary action
between them), and sseq's structure uses it. The rule that reproduces sseq's split structure:

    W_{c→b} = 0 whenever c ∉ A·b

With it the derived W, with nothing fitted, is consistent with the ground truth for C2 ∧ Cν (115
nonzero cross d₂ mod B, stem 66), C2 ∧ Cσ (32, stem 56; 116 failures without the rule), Cη ∧ Cν
(172, stem 56), C2_eta (106), the Joker (109) and C2 ∧ Cη (204). A non-split structure (e.g. the
cofiber of a class α in Adams filtration 2, whose module is split) adds its cocycle (for C(α), the
cobar representative of α on the top → bottom block). The Joker's single contradictory equation
remains: the derived W still matches its truth, so the missing obstruction term is invisible there.
