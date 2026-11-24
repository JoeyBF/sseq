# Exact Adams d2 of cofibers from the sphere

Research artifacts, not built by the workspace. `DERIVATION.md` has the argument. The examples ran
in scratch trees with the patches applied.

- `examples/module_d2.rs`: the cofiber's secondary homotopy blocks from the sphere's secondary
  resolution (`specs/*.spec`: H coefficients and d∘d path weights per pair of cells). Free: no
  lifting.
- `scripts/to_d2.py`: the blocks to Ext-level d2 in the `cofiber_slots` basis, with the Z/4 carry
  (`DERIVATION.md` § 5).
- `examples/cone_d2.rs`: the exact reference, the secondary chain map of δ' (whose cone is P ⊗ M).
  It is resolution-sized.
- `examples/jbar.rs`: the bar-level obstruction cochain, checked to equal O_j (§ 6).
- `examples/bar_o.rs`, `twisted_w.rs`: the exact obstruction and exact solve for any M (multi-cell
  specs).
- `examples/carry_check.rs`: d2 ranks of P ⊗ M, with and without the carry, against M's minimal
  resolution.
- `patches/`: `ResolutionHomomorphism::set_map_rows`, and the carry in
  `TensorResolutionSecondary::d2`.

Status through stem 200 (C2, Cη, Cν, Cσ), formula plus carry:
- matches the easy λ-products on about 77k classes, and the exact δ' cone through stem 98;
- d2∘d2 = 0;
- 0 engine contradictions.

Open: the transport from the bar resolution to the minimal one.
