# Growing the diagram

The engine's diagram of spectra grows at runtime: a class proven to be a homotopy element
`S^{t-1} -> X` (later: an element of `DX ∧ Y`, a map `X -> Y`) adds its cofiber as a node, with the
cofiber sequence's maps. Everything a new node needs comes from the sphere's saved resolution and
products; nothing is resolved or lifted.

- `attach.py`: the cofiber's cohomology when the class has filtration 1 (an A-module extension by
  one cell). `--cocycle` takes the class in the `P_1 ⊗ M` coordinates of `module_slots`.
- `module_slots.rs` (an `ext` example; build it in a tree with the `nassau` feature): E2 of any
  finite module as the cohomology of `Hom(P) ⊗ M^*`, with a basis that extends `cofiber_slots`'
  two-cell convention. Valid for stems below the resolution's max stem and filtrations below its
  max filtration.
- `slots.py`: coordinates of cocycles in that basis (maps induced by cells).
- `cmp_slots.py`, `cmp_hact.py`: checks against `cofiber_slots` and the h0-h7 product table.

`module_slots` is the closed form of `tensor_resolution_ext` (the untwisted tensor resolution
`P ⊗ M`, never materialised), with the basis pinned to `cofiber_slots`. The h_i action uses the
chain map `g ↦ θ_i(dg)`, `θ_i(Sq(R)) = Sq(R - 2^i e_1)`, so it needs no lifting either; other sphere
classes act through their saved maps of `P` (`tensor_resolution_products`). d2 comes from
`module_d2`'s formula, not `TensorResolutionSecondary`, which materialises `P ⊗ M`.
