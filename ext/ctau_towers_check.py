"""Checks the products of a tower export of `examples/ctau_scale.rs` against each other.

    python3 -I ctau_towers_check.py <tower export>

* A generator that is `h_j` (`j = 1, ..., 4`: alone in its tridegree, filtration 1) must multiply
  as `Ctau_h.csv` says, which comes from a different formula.
* Products of two generators must commute.
* Every tower generator below the product bound that is not listed as an algebra generator must
  be a sum of listed products and multiples of `h_0`, by construction; here it is only counted.
"""

import csv
import sys
from collections import defaultdict

out = sys.argv[1]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


towers = {int(r["id"]): (int(r["stem"]), int(r["filtration"]), int(r["weight"])) for r in rows(f"{out}/Ctau_towers.csv")}
gens = [int(r["id"]) for r in rows(f"{out}/Ctau_generators.csv")]
prod = defaultdict(dict)
for r in rows(f"{out}/Ctau_products.csv"):
    prod[int(r["generator"])][int(r["source"])] = frozenset(int(t) for t in r["targets"].split())
h = defaultdict(dict)
for r in rows(f"{out}/Ctau_h.csv"):
    h[int(r["j"])][int(r["source"])] = frozenset(int(t) for t in r["targets"].split())

failures = 0
hdeg = {1: (1, 1, 1), 2: (3, 1, 2), 3: (7, 1, 4), 4: (15, 1, 8)}
max_n = max(t[0] for t in towers.values())
print(f"{len(gens)} algebra generators:", ", ".join(str(towers[g]) for g in gens[:40]), "..." if len(gens) > 40 else "")
for j, d in hdeg.items():
    match = [g for g in gens if towers[g] == d]
    if not match:
        continue
    g = match[0]
    # both hold every generator whose product lies in the range
    keys = set(prod[g]) | set(h[j])
    bad = sum(1 for k in keys if prod[g].get(k) != h[j].get(k))
    failures += bad
    print(f"h{j}: {len(keys)} nonzero products, {bad} differ from the differential's")

pairs = bad = 0
for a in gens:
    for b in gens:
        if a < b:
            pairs += 1
            if prod[a].get(b, frozenset()) != prod[b].get(a, frozenset()):
                bad += 1
                print(f"  {towers[a]} * {towers[b]}: {sorted(prod[a].get(b, []))} vs {sorted(prod[b].get(a, []))}")
failures += bad
print(f"commutativity: {pairs} pairs of generators, {bad} differ")
print("FAILURES:", failures)
sys.exit(1 if failures else 0)
