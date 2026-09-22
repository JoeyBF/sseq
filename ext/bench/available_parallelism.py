"""How much concurrency does the signature DAG actually expose at the frontier?

depth  = sum_i p_i(p_i+1)/2                (the schedule's serial length)
width  = signatures / depth                (mean concurrency available within one bidegree)

Weighted by where the frontier census says the wall time really is.
"""
import csv

PROF = {2: [1], 4: [1, 1], 8: [2, 1], 16: [2, 1, 1], 32: [2, 2, 1], 64: [3, 2, 1],
        128: [3, 2, 1, 1], 256: [3, 2, 2, 1], 512: [3, 3, 2, 1], 1024: [4, 3, 2, 1],
        2048: [4, 3, 2, 1, 1], 4096: [4, 3, 2, 2, 1], 8192: [4, 3, 3, 2, 1],
        32768: [5, 4, 3, 2, 1]}

rows = []
with open('/rs/rs_grp_csht/resolutions/sphere_gpu_n400/nassau_census_40178307.csv') as f:
    for r in csv.DictReader(f):
        try:
            rows.append({k: int(v) for k, v in r.items()})
        except (ValueError, TypeError):
            continue

tot = sum(r['wall_us'] for r in rows)
print(f"{'subalg':>8} {'profile':>16} {'sigs':>7} {'depth':>6} {'width':>8} {'% of wall':>10}")
acc = 0.0
for d in sorted({r['subalgebra_dim'] for r in rows}):
    p = PROF.get(d)
    if not p:
        continue
    depth = sum(v * (v + 1) // 2 for v in p)
    sigs = d - 1
    width = sigs / depth if depth else 1.0
    w = sum(r['wall_us'] for r in rows if r['subalgebra_dim'] == d)
    if w == 0:
        continue
    acc += (w / tot) * width
    print(f"{d:8,} {str(p):>16} {sigs:7,} {depth:6} {width:8.1f} {100*w/tot:9.1f}%")

print(f"\nwall-weighted concurrency available WITHIN a bidegree: {acc:.1f}x")
print("cross-bidegree parallelism is capped at 8.1x (bidegree DAG critical path, measured)")
print(f"combined structural ceiling: ~{acc*8.1:,.0f}x")
print("\nA(4) for comparison: 32767 sigs / depth 35 = 936x within a bidegree.")
