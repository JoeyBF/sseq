"""Leibniz audit of a module d2 file against the slots action: for each indecomposable product
a.m = p with d2(a) = 0 on the sphere, checks d2(p) = a.d2(m), where a.m' for the classes m' of d2(m)
must themselves be known indecomposable products (else the entry is skipped).

usage: leibniz.py <slots dir> <name> <module d2 csv> <sphere d2 csv> <max stem>
"""
import csv
import sys
from collections import Counter, defaultdict

slots, name, mod_d2, sph_d2, max_n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])


def mask(s):
    s = s.strip().strip('"').strip('[]')
    return sum(int(t) << i for i, t in enumerate(s.split(','))) if s.strip() else 0


def load_d2(path):
    out = {}
    for r in csv.reader(open(path)):
        if r[0] == 'stem':
            continue
        out[(int(r[0]), int(r[1]), int(r[2]))] = mask(r[3])
    return out


d2m = load_d2(mod_d2)
d2s = load_d2(sph_d2)
pstar = defaultdict(dict)
for r in csv.reader(open(f"{slots}/{name}_pstar.csv")):
    if r[0] != 'stem':
        pstar[(int(r[0]), int(r[1]))][int(r[2])] = 1
act = {}
for r in csv.reader(open(f"{slots}/{name}_indaction.csv")):
    if r[0].startswith('stem'):
        continue
    a = (int(r[0]), int(r[1]), int(r[2]))
    m = (int(r[3]), int(r[4]), int(r[5]))
    act[(a, m)] = mask(r[6])
unknown = set()
for r in csv.reader(open(f"{slots}/{name}_unknown.csv")):
    if r[0].startswith('stem'):
        continue
    unknown.add(((int(r[0]), int(r[1]), int(r[2])), (int(r[3]), int(r[4]), int(r[5]))))

stats = Counter()
bad = []
for (a, m), p in act.items():
    n, f = a[0] + m[0], a[1] + m[1]
    if n > max_n or d2s.get(a, None) != 0:
        stats['sphere d2 nonzero or unknown'] += 1
        continue
    if m not in d2m:
        stats['m out of range'] += 1
        continue
    # d2(p)
    lhs = 0
    ok = True
    i = 0
    q = p
    while q:
        if q & 1:
            v = d2m.get((n, f, i))
            if v is None:
                ok = False
                break
            lhs ^= v
        q >>= 1
        i += 1
    if not ok:
        stats['product out of range'] += 1
        continue
    rhs = 0
    t = d2m[m]
    j = 0
    while t:
        if t & 1:
            key = (a, (m[0] - 1, m[1] + 2, j))
            if key in act:
                rhs ^= act[key]
            elif key in unknown:
                ok = False
                break
            # absent and not unknown: the product is zero
        t >>= 1
        j += 1
    if not ok:
        stats['a.d2(m) unknown'] += 1
        continue
    stats['checked'] += 1
    if lhs != rhs:
        kind = 'lift' if m[2] in pstar.get((m[0], m[1]), {}) else 'coker'
        bad.append((n, a, m, kind, lhs, rhs))
        stats[f'violated ({kind} m)'] += 1
for k, v in sorted(stats.items()):
    print(f"{k}: {v}")
bad.sort()
for n, a, m, kind, lhs, rhs in bad[:25]:
    print(f"stem {n}: a={a} m={m} ({kind}) d2(p)={lhs:b} a.d2(m)={rhs:b}")
if bad:
    print("first violating product stem:", bad[0][0], " stems:", sorted(Counter(b[0] for b in bad).items())[:20])
