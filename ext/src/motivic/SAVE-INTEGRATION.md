# Motivic ↔ #260 (ZarrV3 save) integration plan

Handoff doc. Branch `motivic-seqsee-zarrs` (`origin`, JoeyBF/sseq) = the motivic
pipeline merged on top of PR #260 (`zarrs-save-rework`, an OPEN upstream PR that
rewrites the save system to a ZarrV3 store). Builds; 23 motivic lib tests pass incl
`save_load_round_trips`. Fallback branch `motivic-seqsee` (pre-merge) untouched.

## Why (measured, not guessed)

Full cold n=100 profile (`ext/perf/FINDINGS-n100.md`), 80-min run — three phases are
99.96%:

- **products** (3× h_i lift) — 2536 s / 42 min
- **differential lift** — 1627 s / 27 min
- resolution — 670 s / 11 min (ALREADY cached by #260)
- everything else (weights, deformation SS/SNF, δ, tau_module sweep) — **~2 s total**

So **cache exactly two things: the differential lift and the product lifts** (86% of
the run). Resolution is #260's job; weights/SS/δ/tau_module recompute in ~2 s → never
cache. The save layer is a **pure memoization cache**: everything is a deterministic
function of (module, box); delete the store → recompute → byte-identical. A miss /
mismatch / corruption is always just "recompute", never a failure. The only guard
worth keeping is #260's algebra/module binding (is this cache for MY problem?).

Orthogonal lever (not persistence): **truncation** — restrict each lift to the cells &
weight-orders in-box products actually read (cone + weight-order). Cuts the same 69 min
on the FIRST run, before any cache is warm. Do after, or interleave.

## The one math invariant that makes it sound

A CONVERGED (in-cone) lift cell depends only on the resolution up to its own internal
degree — NOT on the box. So a cell computed at n=50 is byte-identical at n=100 ⇒
per-cell caching + lazy growth is valid. Only out-of-cone PARTIAL cells are
box-dependent ⇒ never persist partials; recompute the frontier. Weights are
per-generator, equally box-independent.

## What is primary (cache) vs derived (recompute)

- PRIMARY (cache, by cost): `self.lifted` (differential lift supports) + `weights`;
  the product lifts `φ_a` and Massey null-homotopies `H_{a,b}` (supports).
- DERIVED (recompute on load, ~ms): δ (`MotivicCoboundary`, = f(lifted,weights)), the
  deformation SS, `tau_module`, and every τ-power (= w_target − w_source from weights;
  never stored — persist F₂ supports only).

## #260 save API map (verbatim surface, `ext/src/save.rs` post-merge)

`SaveKind` (`#[non_exhaustive]`): Kernel, Differential, ResQi, AugmentationQi,
SecondaryComposite, SecondaryIntermediate, SecondaryHomotopy, **ChainMap**,
**ChainHomotopy**, NassauDifferential, NassauQi.

`ZarrSaveStore`:
- `create(path: impl AsRef<Path>) -> Result<Self>`
- `write<const N>(&self, kind: SaveKind, location: impl SaveCoords<N>, data: &[u8]) -> Result<()>` (raw bytes; asserts kind ∉ {ResQi,NassauQi})
- `read<const N>(&self, kind, location) -> Result<Option<Vec<u8>>>` (Some iff non-empty)
- `exists<const N>(&self, kind, location) -> bool`; `delete<const N>(...)`
- `subgroup(&self, name: &str) -> Result<Self>` (own zarr group ⇒ its OWN shard arrays per kind — this is why reusing existing SaveKinds inside a subgroup is collision-free)
- `path(&self) -> &Path`; `bind_to_algebra(magic,prime,prefix)`; `set_complex_name`; `bind_module_spec(&serde_json::Value)`; `read_root_attributes(path)` (assoc)

`SaveCoords<N>`: `SaveCoords<2> for Bidegree` → `[n,s]`; `SaveCoords<3> for
BidegreeGenerator` → `[n,s,idx]`. Shard shape `[8,8(,8)]`, n∈[-1024,3072).

`SaveDirectory` = `enum { None, Store(ZarrSaveStore) }`; `is_none/is_some/store()`;
only conversion is `TryFrom<Option<PathBuf>>` (`None→None`, `Some(p)→create(p)`).
`Resolution::new_with_save(complex, impl TryInto<SaveDirectory,...>)` — pulls the store
via `self.save_dir.store()`; there IS a `Resolution::save_dir()` accessor
(`resolution_homomorphism.rs` uses `source.save_dir().store()`).

Patterns to MIRROR:
- Differential (`resolution.rs:695`/`:416`): `bitcode::serialize(payload)` →
  `store.write(SaveKind::Differential, b /*Bidegree*/, &bytes)`; read → `store.read(...)`
  → `bitcode::deserialize`.
- ChainMap (`resolution_homomorphism.rs`): ctor makes `parent.subgroup("products/{name}")`;
  write `SaveKind::ChainMap` keyed by `Bidegree input`, payload = concatenated
  `FpVector::to_bytes`. ChainHomotopy: `subgroup("homotopies/{l}__{r}")`,
  `SaveKind::ChainHomotopy`.
- Atomicity: shard writes are a single element under a per-shard mutex ⇒ a crash leaves
  a cell absent, never partial; `exists`/non-empty-read is the presence test.

Current motivic persistence to REPLACE: `persist.rs` writes a sidecar
`motivic-lift.bin` via `byteorder` into the `save_dir` path (magic `0x004D_0004` +
`(max,compute)` box + weights + lifted). `with_module(save_dir: Option<PathBuf>)`
clones it into `Resolution::new_with_save(cc, save_dir.clone())` and passes `&save_dir`
to `load_lift`/`save_lift`. Keep the `Option<PathBuf>` param (it TryInto's to
SaveDirectory); the sidecar file+format go away.

## Design decision: pure subgroups, ZERO edits to #260

Because a subgroup owns its shard arrays, motivic reuses #260's existing kinds inside
its own namespace — no `SaveKind` enum edit (matters: #260 is an open, moving PR).

| motivic data | store location (subgroup of the resolution's store) | reuse kind | key |
|---|---|---|---|
| lift + weights | `subgroup("motivic")` | `Differential` | `Bidegree` (per (s,t)) |
| product lift φ_a | `subgroup("motivic/products/{a}")` | `ChainMap` | `Bidegree` |
| null-homotopy H_{a,b} | `subgroup("motivic/homotopies/{a}__{b}")` | `ChainHomotopy` | `Bidegree` |

Payload: `bitcode`-serialize a per-(s,t) record `{num_gens, [(weight_i32, support)]}`
(support = the `BTreeSet<usize>` we already hold). `{a}` name = a stable `Gen` label,
e.g. `s_t_idx` or the `Display` `(n, s)#idx` sanitized.

Store access: motivic grabs the SAME store the resolution opened —
`self.resolution.save_dir().store()?.subgroup("motivic")` — NOT a second store, so it
inherits #260's `bind_to_algebra` + `bind_module_spec` guards for free.

Incremental keying: per-(s,t) elements are box-independent ⇒ DROP the old
`(max,compute)` rejection. `load` reads whatever cells exist; `lift`/`compute_weights`
skip present cells (`store.exists(...)`) and compute only the frontier; write new
converged cells back.

## Phases (execute in order)

- **Phase 1 — lift + weights through the store.** Rewrite `persist.rs` to
  read/write `weights`+`lifted` per-(s,t) under `subgroup("motivic")` via the
  Differential kind + bitcode. Delete the byteorder/`motivic-lift.bin` format. First
  check: confirm `Resolution::save_dir()` is reachable from `motivic` (may need
  `pub(crate)`).
- **Phase 2 — incremental. DONE.** `load_lift` is now partial (populates whatever
  cells exist, no all-or-nothing gate); `compute_weights`/`lift` skip cells already
  loaded and compute only the frontier; `save_lift` persists the lifted support only
  for **box-independent** cells (`s < 2` seed, or in-cone converged — never an
  out-of-cone seed placeholder, via `lift_is_box_independent`). `with_module` always
  load→compute→lift→save. Counters: `LIFT_CACHE_LOADS` (coarse: a load found ≥1
  cell) + new `LIFT_CELLS_REUSED` (cells skipped by reuse). Free-module bases forced
  after `compute_through_stem` (Phase-1 fix) is what makes the grow-the-box lift
  correct on a reloaded resolution. Tests: `motivic_grow_the_box_reuses_cache`
  (small→big on one store reuses cells and equals a cold big build) in
  `tests/motivic_ctau.rs`.
- **Phase 3 — product lifts + null-homotopies.** Persist `φ_a`/`H_{a,b}` under the
  product/homotopy subgroups; `lift_product`/`lift_nullhomotopy` check the store and
  lift only the frontier. This is the biggest payoff (42 min of products).
- **Phase 4 — free win (already there):** the CTauResolution's diffs+QIs are persisted
  by #260 (`res_qi` stream tier); `differential(s).quasi_inverse(t)` comes off disk on
  a warm store with no motivic work.

## Testing

- Round-trip (extend `save_load_round_trips`): store-build → reload → `weights`,
  `lifted`, `tau_module`, products identical.
- Incremental: build n=A with store, then n=B>A on the same store; assert (i) == cold
  n=B, (ii) only frontier recomputed (reuse counter).
- Golden: store-backed chart == `motivic_chart_matches_golden`.
- Crash/partial: delete a cell (or kill mid-write) → reload recomputes it, not garbage.

## Risks

- #260 unmerged/moving → subgroups minimize coupling; pin the `zarrs-save-rework` sha
  and re-merge on updates. Current merge = commit 83253b6dac.
- Never persist out-of-cone partials → explicit `converged` check at write time + a test.
- Motivic must open the SAME store as the resolution (not a 2nd), so binding guards and
  `res_qi` reuse line up.
