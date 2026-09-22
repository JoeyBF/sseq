"""Does the available parallelism GROW as we advance, offsetting the cost growth?

Subalgebra boundaries are affine in s, so for fixed s the subalgebra increases with t. Each step up
multiplies the signature DAG's width enormously:

    A(3) [4,3,2,1]      1,023 sigs / depth 20 =     51x
    A(4) [5,4,3,2,1]   32,767 sigs / depth 35 =    936x     (18.3x more)
    A(5) [6,5,4,3,2,1] 2,097,151 / depth 56   = 37,449x     (40.0x more)

Cost grows 6.8%/stem. If width jumps 18x at a crossing, that pays for ln(18)/ln(1.068) = 44 stems
of cost growth in one step. This measures where the crossings actually are.
"""
import csv, collections, math

WIDTH = {8: 7/4, 64: 63/10, 128: 127/11, 256: 255/13, 512: 511/16, 1024: 1023/20,
         2048: 2047/21, 4096: 4095/23, 8192: 8191/26, 32768: 32767/35}
NAME = {64: 'A(3)-ish [3,2,1]', 1024: 'A(3) [4,3,2,1]', 8192: '[4,3,3,2,1]',
        32768: 'A(4) [5,4,3,2,1]', 2048: '[4,3,2,1,1]', 4096: '[4,3,2,2,1]',
        512: '[3,3,2,1]', 256: '[3,2,2,1]', 128: '[3,2,1,1]', 8: '[2,1]'}

rows = []
with open('/rs/rs_grp_csht/resolutions/sphere_gpu_n400/nassau_census_40178307.csv') as f:
    for r in csv.DictReader(f):
        try:
            rows.append({k: int(v) for k, v in r.items()})
        except (ValueError, TypeError):
            continue

# for each s, the stem at which each subalgebra_dim first appears
first = collections.defaultdict(dict)
for r in rows:
    n = r['t'] - r['s']
    d = r['subalgebra_dim']
    s = r['s']
    if d not in first[s] or n < first[s][d]:
        first[s][d] = n

print("For each s: the stem at which each subalgebra FIRST appears (census window only)")
print(f"{'s':>4}  crossings (stem -> subalgebra, width)")
slopes = []
for s in sorted(first)[:24]:
    ds = sorted(first[s].items(), key=lambda kv: kv[1])
    parts = [f"{n}->{d}({WIDTH.get(d,1):.0f}x)" for d, n in ds]
    print(f"{s:4}  {'  '.join(parts)}")
    # if two subalgebras seen, record the crossing stem for the larger
    if len(ds) >= 2:
        slopes.append((s, ds[-1][1], ds[-1][0]))

print("\ncrossings observed (s, stem, new subalgebra):")
for s, n, d in slopes[:20]:
    print(f"  s={s:3}  stem {n:4}  -> {NAME.get(d, str(d))}  width {WIDTH.get(d,1):.0f}x")

if len(slopes) >= 2:
    # fit stem = a*s + b for the crossing line
    xs = [s for s, _, _ in slopes]
    ys = [n for _, n, _ in slopes]
    k = len(xs)
    mx, my = sum(xs)/k, sum(ys)/k
    den = sum((x-mx)**2 for x in xs)
    if den:
        a = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/den
        b = my - a*mx
        print(f"\ncrossing line fit: stem ~= {a:.1f}*s + {b:.0f}")

print("\nWhat a width jump is worth, at 6.8%/stem cost growth:")
for lo, hi, nm in ((1023/20, 32767/35, 'A(3)->A(4)'), (32767/35, 2097151/56, 'A(4)->A(5)')):
    jump = hi/lo
    print(f"  {nm}: width x{jump:.1f}  == {math.log(jump)/math.log(1.068):.0f} stems of cost growth")
