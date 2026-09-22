"""What IS the per-stem cost growth rate?

Recorded as 14.2%/stem (x2 per 5.2 stems). Nassau's historical figure is x2 per 10 stems
(7.2%/stem). The difference decides the whole project: from stem 312 to 400 (88 stems),
  14.2%/stem -> 1.2e5x needed
   7.2%/stem -> 448x needed
and the second is inside the signature DAG's structural ceiling while the first is not.

Fit log(wall) vs stem SEPARATELY FOR EACH s. Mixing s values conflates two different things --
the cost of advancing a row, and the cost of adding rows -- and would inflate the apparent rate.
"""
import csv, collections, math

rows = []
with open('/rs/rs_grp_csht/resolutions/sphere_gpu_n400/nassau_census_40178307.csv') as f:
    for r in csv.DictReader(f):
        try:
            rows.append({k: int(v) for k, v in r.items()})
        except (ValueError, TypeError):
            continue

by_s = collections.defaultdict(list)
for r in rows:
    if r['wall_us'] > 0 and r['num_new_gens'] >= 0:
        by_s[r['s']].append((r['t'] - r['s'], math.log(r['wall_us'])))


def fit(pts):
    n = len(pts)
    if n < 4:
        return None, None, n
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, y in pts)
    if den == 0:
        return None, None, n
    slope = num / den
    # r^2
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return slope, r2, n


print(f"{'s':>4} {'n':>4} {'%/stem':>8} {'x2 every':>10} {'r^2':>6}")
good = []
for s in sorted(by_s):
    slope, r2, n = fit(by_s[s])
    if slope is None or slope <= 0:
        continue
    pct = (math.exp(slope) - 1) * 100
    dbl = math.log(2) / slope
    if n >= 8 and r2 is not None and r2 > 0.5:
        good.append((slope, n))
    flag = "" if n >= 8 else "  (few pts)"
    print(f"{s:4} {n:4} {pct:7.1f}% {dbl:9.1f} {r2:6.2f}{flag}")

if good:
    # weight by number of points
    tw = sum(n for _, n in good)
    sl = sum(sl * n for sl, n in good) / tw
    pct = (math.exp(sl) - 1) * 100
    dbl = math.log(2) / sl
    print(f"\nweighted fit over {len(good)} rows with r^2>0.5: {pct:.1f}%/stem, x2 every {dbl:.1f} stems")
    for target, frm in ((400, 312),):
        need = math.exp(sl * (target - frm))
        print(f"  stem {frm} -> {target} ({target-frm} stems) needs {need:,.0f}x")
        print(f"  at 7.2%/stem (x2 per 10) it would need "
              f"{math.exp(math.log(2)/10*(target-frm)):,.0f}x")
