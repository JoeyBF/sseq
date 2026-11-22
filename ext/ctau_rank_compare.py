"""Compares two exports in the per-tridegree format of `ctau_legacy.py` up to a change of basis.

    python3 -I ctau_rank_compare.py <legacy export> <legacy export> [complex ...]

The reduction by columns and the transposed one (`ctau_dual.rs`) find the same towers and
different bases of each tridegree, so what must agree is every rank, and the rank of every
`h_j`, cell map and composite `h_i h_j` out of each tridegree. Complexes default to `Ctau`.
"""

import csv
import sys
from collections import defaultdict

one, two = sys.argv[1], sys.argv[2]
names = sys.argv[3:] or ["Ctau"]
hdeg = {0: (0, 1, 0), 1: (1, 1, 1), 2: (3, 1, 2), 3: (7, 1, 4), 4: (15, 1, 8)}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def tri(r):
    return (int(r["stem"]), int(r["filtration"]), int(r["weight"]))


def vec(cell):
    v = 0
    for i, c in enumerate(cell.strip("[]").split(",")):
        if c.strip() == "1":
            v |= 1 << i
    return v


def rank_of(vectors):
    basis = {}
    for v in vectors:
        while v:
            top = v.bit_length() - 1
            if top not in basis:
                basis[top] = v
                break
            v ^= basis[top]
    return len(basis)


def load(d, name):
    ranks = {tri(r): int(r["rank"]) for r in rows(f"{d}/{name}_rank.csv")}
    # maps[(j, source tridegree)][index] = image as a bit vector
    maps = defaultdict(dict)
    for r in rows(f"{d}/{name}_h.csv"):
        maps[(int(r["j"]), tri(r))][int(r["index"])] = vec(r["product"])
    return ranks, maps


def apply(maps, j, x, v):
    """The image of the vector `v` at `x` under `h_j`."""
    out = 0
    images = maps.get((j, x), {})
    i = 0
    while v:
        if v & 1:
            out ^= images.get(i, 0)
        v >>= 1
        i += 1
    return out


failures = 0
for name in names:
    r1, m1 = load(one, name)
    r2, m2 = load(two, name)
    bad = [x for x in set(r1) | set(r2) if r1.get(x, 0) != r2.get(x, 0)]
    failures += len(bad)
    print(f"{name}: ranks: {len(r1)} tridegrees, {sum(r1.values())} classes, {len(bad)} differ {sorted(bad)[:5]}")
    keys = set(m1) | set(m2)
    bad = [k for k in keys if rank_of(m1.get(k, {}).values()) != rank_of(m2.get(k, {}).values())]
    failures += len(bad)
    total = sum(rank_of(m1.get(k, {}).values()) for k in keys)
    print(f"{name}: h_j: {len(keys)} maps of total rank {total}, {len(bad)} ranks differ {sorted(bad)[:5]}")
    # composites h_i h_j out of each tridegree
    count = bad = total = 0
    for j, x in keys:
        y = tuple(a + b for a, b in zip(x, hdeg[j]))
        for i in range(5):
            if (i, y) not in m1 and (i, y) not in m2:
                continue
            rk = []
            for maps, ranks in ((m1, r1), (m2, r2)):
                images = [apply(maps, i, y, v) for v in maps.get((j, x), {}).values()]
                rk.append(rank_of(images))
            count += 1
            total += rk[0]
            bad += rk[0] != rk[1]
    failures += bad
    print(f"{name}: h_i h_j: {count} composites of total rank {total}, {bad} ranks differ")
    for kind in ("istar", "pstar"):
        try:
            c1 = defaultdict(list)
            c2 = defaultdict(list)
            for d, c in ((one, c1), (two, c2)):
                for r in rows(f"{d}/{name}_{kind}.csv"):
                    c[tri(r)].append(vec(r["image"]))
        except FileNotFoundError:
            continue
        keys2 = set(c1) | set(c2)
        bad = [k for k in keys2 if rank_of(c1.get(k, [])) != rank_of(c2.get(k, []))]
        failures += len(bad)
        print(f"{name}: {kind}: {len(keys2)} maps, {len(bad)} ranks differ {sorted(bad)[:5]}")
print("OK" if failures == 0 else f"{failures} FAILURES")
sys.exit(1 if failures else 0)
