# Exact d₂ of the two-cell complexes C(h_j) from the sphere

p = 2, j ∈ {0, 1, 2, 3}, k = 2^j, M = H^*C(h_j) with basis x₀ (degree 0), x₁ (degree k),
Sq^k x₀ = x₁.

## Inputs

- P: the sphere's minimal resolution, with sseq's secondary resolution: composites
  Γ(g) = Σ σ(a)σ(b) over the d∘d paths of g, and homotopies H solving

      d H(g) = I(g) + H(dg),   I(g) = Σ_{a e ∈ dg} A(a, Γ(e)),   H = 0 on P₂.        (R)

- A chain map φ: P → P of degree (−1, k), and its secondary chain map Φ as computed by
  `SecondaryResolutionHomomorphism`: composites Ψ(g) = −σ(d)σ(φ) + σ(φ)σ(d) over the two kinds of
  paths, and Φ solving

      d Φ(g) = Σ_{a e ∈ dg} A(a, Ψ(e)) + Φ(dg) + Σ_{c e' ∈ φg} A(c, Γ(e')) + H(φ g) + φ(H g),
      Φ = 0 on P₂.                                                                      (R_φ)

## 1. P ⊗ M is the cone of δ'

M ⊗ (−) with the diagonal action, written on the free basis g ⊗ x_i through the untwisting
(a e) ⊗ m = Σ a'(e ⊗ χ(a'') m), has the differential

    D(g ⊗ x₁) = dg ⊗ x₁,    D(g ⊗ x₀) = dg ⊗ x₀ + δ'(g) ⊗ x₁,    δ'(g) = Σ_{a e ∈ dg} θ(a) e,

where θ(a) = Σ a' ⟨χ(a''), ξ₁^k⟩. The x₁-coefficient of b·x₀ (|b| = k) vanishes on decomposables,
because M is zero in degrees 1, …, k−1, so it is pairing with the primitive ξ₁^k; χ(ξ₁) = ξ₁, and
the Milnor coproduct gives θ(Sq(R)) = Sq(R − k e₁) (0 if r₁ < k). D² = 0 makes δ' a chain map, and
its unit coefficients on P₁ → P₀ are those of h_j, so δ' lifts h_j.

## 2. The cone of a secondary chain map is a secondary resolution

Let R = cone(φ): R_s = P_s ⊕ P_s, d_R = [[d, 0], [φ, −d]] (first summand x₀, second x₁). Then

    Γ_R = [[Γ, 0], [Ψ, Γ]],   H_R = [[H, 0], [Φ, H]]

satisfy R's recursion d_R H_R = I_R + H_R d_R with zero base. Block by block:

- The diagonal blocks are (R) (for −d: σ(−a)σ(−b) = σ(a)σ(b), and A(−a, ·) ≡ A(a, ·) mod 2).
- The x₀ → x₁ block of d_R H_R(g) is φ(H g) − dΦ(g). That of I_R(g) is Σ A(a, Ψ(e)) over the dg
  terms plus Σ A(c, Γ(e')) over the φg terms, since d_R(g) = dg + φg and Γ_R(e) has these blocks.
  That of H_R(d_R g) is Φ(dg) + H(φ g). Equating gives (R_φ). Ψ enters up to an overall sign, which
  does not matter: it lies in the image of ∂, where the twos are read mod 4 and the Y terms mod 2.

Every step of (R_φ) past the first is solvable because P is exact in positive degrees. The first
step (s = 3) needs the base to be valid; `cone_d2` checks d Φ(g) = I(g) on every generator there
(`CD_VERIFY`).

## 3. The d₂

With φ = δ', R = P ⊗ M, so the cocycles are those of `cofiber_slots`: i_* z = z ⊗ x₀, and the
lift ỹ = y ⊗ x₁ of y with h_j y = 0. d₂ is the class of the cocycle composed with H_R:

    d₂(i_* z) = i_*(d₂ z),    d₂(ỹ) = (d₂ y)~ + i_*(Φ^* y),   Φ^* y = (g ↦ ⟨y, Φ(g)⟩₁).

(d₂ y)~ is read through the kernel basis; Φ^* y through i_*. No term is fitted. The only inputs
are the sphere's secondary resolution and sseq's exact solve of (R_{δ'}).

## 4. Which spectrum

A secondary chain map lifting h_j is h_j + λw with w ∈ Ext^{2, k+1}: 0 for j = 1, and h₀h_j for
j = 0, 2, 3, which is permanent and detects 2α. So the result is the d₂ of C(uα) for a 2-adic unit
u (u = 1 or 3), and C(uα) ≃ C(α) by u on the top cell. That equivalence induces the identity on
H^*C, hence on E₂, so the d₂ is that of C(α).

## 5. Reading d₂ off P ⊗ M needs the Z/4 carry

P ⊗ M is not minimal. The coboundary of the lift cocycle y ⊗ x₁ is (h_j y) ⊗ x₀ = 0 mod 2. Over Z/4,
with unit coefficients lifted as 1, it is 2w, where w_g = ½ Σ_i y_i [h_j e_i]_g. Since 2 = λh₀
in the pair algebra, d₂(ỹ) gains i_*(h₀ w). sseq's `hom_k_with` applies the same rule to
secondary products. Without it, the formula gives wrong values exactly where the carry fires
(Cη (175, 23, 6) and the stems 191–200 classes).

## 6. The formula's input is exact on the bar resolution

On the bar resolution, the unit-level part of d W' = W' d + J at g = [a₁|a₂|a₃] is the cobar
equation W(a₁a₂, a₃) + W(a₁, a₂a₃) = J₁(a₁, a₂, a₃). Only a₁[a₂|a₃] has a nonunit coefficient,
and A(1, ·) = 0, θ(1) = 0, so J₁ = A(a₁, Λ) + A(θa₁, Γ) + θA(a₁, Γ), with
Γ = σ(a₂)σ(a₃) + σ(a₂a₃) and Λ = σ(a₂)σ(θa₃) + σ(θa₂)σ(a₃) + σ(θ(a₂a₃)).

`examples/jbar.rs` computes J₁ in the Milnor pair algebra, Y_{k,l} terms and carries included, on
every triple of total degree 2^j + 1, j = 0, …, 3. J₁ = O_j everywhere:
- zero for j = 0, 1;
- [Sq1|Sq2|Sq2] for j = 2;
- [Sq1|Sq4|Sq4] for j = 3.

So W_j (δW_j = O_j) solves the bar-level equation exactly.

Remaining step: the transport from the bar resolution to the minimal P. The path weights are
W_j ∘ f₂, where f₂ is the comparison map to the bar resolution. The comparison composites that
#320's transport adds for a non-minimal target are taken here to contribute nothing at the unit
level on P. This is the same lemma as the conjectured invariance of the zero lift on minimal
resolutions (§ secondary_actions work), and it is not yet proved. The data through stem 200 are
consistent with it.

## Why the earlier formula failed

The formula itself, `H:P(k)` plus the cobar path weights W_j, was right (§ 6). The Ext-level read-off
in `to_d2.py`, and in the tensor ground-truth extraction, omitted the Z/4 carry of § 5. The errors
sat where the carry fires: Cη (175, 23, 6) and stems 191–200, and Cν (193, 19, 8). Because the
ground truth was extracted the same way, it agreed with the formula and hid the bug.

The cone of a *lifted* chain map φ of h_j is also correct, but in a different basis. φ ≃ δ' by a
homotopy K, and the comparison of the two cones moves each lift by i_*(y∘K). This is why the first
cone attempt disagreed with the slots basis while having the same d₂ ranks.
