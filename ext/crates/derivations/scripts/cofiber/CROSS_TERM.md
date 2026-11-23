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
