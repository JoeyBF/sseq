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
- `cmp_slots.py`: check against `cofiber_slots` on two-cell complexes.
