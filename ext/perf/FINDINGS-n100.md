# Motivic pipeline — full n=100 profile (2026-07-31)

Cold run (no cache), `chart_motivic_seqsee` at n=100, s=52, `--features "logging
concurrent"`, `MOT_PROFILE=1`. Logs: `chart-n100-trace.log`, output `motivic_n100.json`
(1.70 MB). Total wall ≈ **80.6 min**.

## Per-phase breakdown

| phase | n=100 wall | n=50 | share | notes |
|---|---|---|---|---|
| **products (3× h_i lift)** | **2536 s (42.3 min)** | ~5 s | **52%** | h₀ 1036s / h₁ 858s / h₂ 642s |
| **differential lift** | **1627 s (27.1 min)** | 1.4 s | **34%** | 658,211 corrections, 20,580 gens |
| resolution | 670 s (11.2 min) | 2.0 s | 14% | already cached by #260 |
| weights | 0.34 s | — | — | noise |
| SS build (snf+apply+setup) | 1.1 s | ~10 ms | — | noise |
| build_ext (δ) | 0.25 s | 2 ms | — | noise |
| chart_dots (tau_module sweep) | 0.77 s | 0.4 ms | — | noise |

Per product lift: h₀ 1036s / 845,713 corr / 13,306 products; h₁ 858s / 828,069 corr /
18,796; h₂ 642s / 821,942 corr / 11,260.

## Conclusions

1. **Three phases are 99.96% of the run**: products (42m) > lift (27m) > resolution
   (11m). Everything else — weights, the whole deformation SS / SNF, δ, the
   `tau_module` sweep — is **~2 seconds combined**. At scale they are pure noise, so
   none of them is worth caching or optimizing.
2. **The τ-adic correction loop is 86% of the run** (lift 1627s + products 2536s =
   4163s = 69 min). Lift and products are the *same disease*: A_C product-op volume in
   the order-by-order correction (98.3% cache hit — it's volume, not misses).
3. **The n≤70 ranking inverted as predicted.** At n≤70 products dwarfed the lift
   (~100×); at n=100 they're the same order (42 vs 27 min). The lift grew ~1160×
   (1.4s→1627s) over a 2× box.

## Implications

- **Cache targets (by measured cost): the product lifts and the differential lift** —
  together 69 min / 86% of the run. The resolution (11 min) is already persisted by
  #260. Weights / SS / δ / tau_module recompute in ~2 s → never cache.
- **The truncation optimization** attacks the *same* 69 min (cut the correction
  product-op volume). Caching = "do it once"; truncation = "do less each time." Both
  target the lift+products; neither touches the (already-trivial) SS/module layer.
- Prior worry that the duplicate per-(s,t) SNF (SS build vs `tau_module` sweep) would
  grow: **false** — snf 0.38s, tau_module sweep 0.77s at n=100. Leave it.
