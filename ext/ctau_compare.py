"""Compares an export of the Cτ E2 page with a native minimal resolution over A_C/τ.

    python3 -I ctau_compare.py <export directory> <reference directory>

The reference holds `ref_rank.csv` (`stem,filtration,weight,rank`: the generators of the
resolution by weight) and `ref_h.csv` (`j,stem,filtration,weight,rank`: the rank of multiplication
by `h_j` out of a tridegree, read off the coefficient of `Sq^{2^j}` in the differential). The
export is compared on the box both cover: every rank, and the rank of every `h_j` whose target
lies in the box. The bases differ, so ranks are what can be compared.
"""

import csv
import sys
from collections import defaultdict

out, ref = sys.argv[1], sys.argv[2]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def tri(r):
    return (int(r["stem"]), int(r["filtration"]), int(r["weight"]))


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


def vec(cell):
    v = 0
    for i, c in enumerate(cell.strip("[]").split(",")):
        if c.strip() == "1":
            v |= 1 << i
    return v


ours = {tri(r): int(r["rank"]) for r in rows(f"{out}/Ctau_rank.csv")}
theirs = {tri(r): int(r["rank"]) for r in rows(f"{ref}/ref_rank.csv")}
max_n = min(max(n for n, _, _ in ours), max(n for n, _, _ in theirs))
max_s = min(max(s for _, s, _ in ours), max(s for _, s, _ in theirs))
print(f"box: stem <= {max_n}, filtration <= {max_s}")

failures = 0
keys = {x for x in set(ours) | set(theirs) if x[0] <= max_n and x[1] <= max_s}
bad = [x for x in sorted(keys) if ours.get(x, 0) != theirs.get(x, 0)]
for x in bad[:10]:
    print(f"  RANK at {x}: export {ours.get(x, 0)}, reference {theirs.get(x, 0)}")
failures += len(bad)
print(f"ranks: {len(keys)} tridegrees, {sum(theirs.get(x, 0) for x in keys)} classes,"
      f" {len(bad)} differ")

hdeg = {0: (0, 1, 0), 1: (1, 1, 1), 2: (3, 1, 2), 3: (7, 1, 4)}
ours_h = defaultdict(list)
for r in rows(f"{out}/Ctau_h.csv"):
    ours_h[(int(r["j"]), tri(r))].append(vec(r["product"]))
ours_h = {k: rank_of(v) for k, v in ours_h.items()}
theirs_h = {(int(r["j"]), tri(r)): int(r["rank"]) for r in rows(f"{ref}/ref_h.csv")}
for j in range(4):
    keys = set()
    for (jj, x) in set(ours_h) | set(theirs_h):
        if jj == j and x[0] + hdeg[j][0] <= max_n and x[1] + 1 <= max_s:
            keys.add(x)
    bad = [x for x in sorted(keys) if ours_h.get((j, x), 0) != theirs_h.get((j, x), 0)]
    for x in bad[:10]:
        print(f"  h{j} out of {x}: export {ours_h.get((j, x), 0)}, reference {theirs_h.get((j, x), 0)}")
    failures += len(bad)
    print(f"h{j}: {len(keys)} tridegrees with a nonzero product,"
          f" total rank {sum(theirs_h.get((j, x), 0) for x in keys)}, {len(bad)} differ")

print("FAILURES:", failures)
sys.exit(1 if failures else 0)
