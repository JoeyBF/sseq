"""Are the EXPENSIVE frontier bidegrees also the WIDE ones?

If the 91% of wall that sits at low s also carries the large subalgebras, then the signature DAG's
concurrency is concentrated exactly where the time is, and distribution targets the right work.
If the expensive ones were narrow, the ~102x would be averaging over the wrong bidegrees.
"""
import csv, collections

PROF = {8: '[2,1]', 64: '[3,2,1]', 128: '[3,2,1,1]', 256: '[3,2,2,1]', 512: '[3,3,2,1]',
        1024: '[4,3,2,1]', 2048: '[4,3,2,1,1]', 4096: '[4,3,2,2,1]', 8192: '[4,3,3,2,1]',
        32768: '[5,4,3,2,1]'}
DEPTH = {8: 4, 64: 10, 128: 11, 256: 13, 512: 16, 1024: 20, 2048: 21, 4096: 23, 8192: 26,
         32768: 35}

rows = []
with open('/rs/rs_grp_csht/resolutions/sphere_gpu_n400/nassau_census_40178307.csv') as f:
    for r in csv.DictReader(f):
        try:
            rows.append({k: int(v) for k, v in r.items()})
        except (ValueError, TypeError):
            continue

hi = [r for r in rows if r['t'] >= 300]
tot = sum(r['wall_us'] for r in hi)

print(f"{'s band':>9} {'wall %':>8} {'dominant subalgebra':>21} {'sigs':>7} {'width':>8}")
wsum = 0.0
for lo, hiB in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 250)]:
    sel = [r for r in hi if lo <= r['s'] < hiB]
    if not sel:
        continue
    w = sum(r['wall_us'] for r in sel)
    bysub = collections.Counter()
    for r in sel:
        bysub[r['subalgebra_dim']] += r['wall_us']
    d, _ = bysub.most_common(1)[0]
    width = (d - 1) / DEPTH.get(d, 1)
    wsum += (w / tot) * width
    print(f"{f'{lo}-{hiB}':>9} {100*w/tot:7.1f}% {PROF.get(d, str(d)):>21} {d-1:7,} {width:7.1f}x")

print(f"\nwall-weighted width over t>=300 only: {wsum:.1f}x")
