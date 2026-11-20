"""Checks an export of `examples/ctau_e2.rs` against Isaksen's tables and against classical Ext.

    python3 -I ctau_check.py <export directory> <directory of the transcribed tables>

Three checks, each on the part of the export's range that decides it:

* every generator of the Cτ E2 page that "Stable stems" lists has a nonzero group in its tridegree;
* every h0, h1, h2 extension it lists (invisible in the E2 page of the sphere) is a nonzero product
  into the listed tridegree;
* for each classical internal degree t, the Cτ E2 page summed over the weights is the E1 page of a
  spectral sequence (τ-Bockstein with τ inverted, the Cartan-Eilenberg spectral sequence of
  E -> A -> P) converging to classical Ext, with differentials raising the filtration by one. So its
  Euler characteristic in s equals that of Ext_A^{s,t}, and each of its ranks is at least that of
  Ext_A^{s,t}.
"""

import csv
import sys
from collections import defaultdict

out, tables = sys.argv[1], sys.argv[2]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


rank = {}
for r in rows(f"{out}/Ctau_rank.csv"):
    rank[(int(r["stem"]), int(r["filtration"]), int(r["weight"]))] = int(r["rank"])
max_n = max(n for n, _, _ in rank)
max_s = max(s for _, s, _ in rank)

# h[j][(n, s, w)] = number of basis classes with a nonzero product by h_j
h = defaultdict(lambda: defaultdict(int))
for r in rows(f"{out}/Ctau_h.csv"):
    h[int(r["j"])][(int(r["stem"]), int(r["filtration"]), int(r["weight"]))] += 1

failures = 0


def in_range(n, s):
    return 0 <= n <= max_n and 0 <= s <= max_s


print("== generators of the Cτ E2 page (ss_Ctau_E2.csv)")
seen = set()
checked = 0
for r in rows(f"{tables}/ss_Ctau_E2.csv"):
    x = (int(r["source_stem"]), int(r["source_filtration"]), int(r["source_weight"]))
    if x in seen or not in_range(x[0], x[1]):
        continue
    seen.add(x)
    checked += 1
    if rank.get(x, 0) == 0:
        failures += 1
        print(f"  MISSING {r['source_name']} at {x}")
print(f"  {checked} tridegrees in range, all others nonzero")

print("== h0, h1, h2 extensions (ss_Ctau_h{0,1,2}.csv)")
step = {0: (0, 1, 0), 1: (1, 1, 1), 2: (3, 1, 2)}
checked = 0
for j in range(3):
    for r in rows(f"{tables}/ss_Ctau_h{j}.csv"):
        if not r["target_stem"] or not r["source_stem"]:
            continue
        x = (int(r["source_stem"]), int(r["source_filtration"]), int(r["source_weight"]))
        y = (int(r["target_stem"]), int(r["target_filtration"]), int(r["target_weight"]))
        if not (in_range(*x[:2]) and in_range(*y[:2])):
            continue
        checked += 1
        expect = tuple(a + b for a, b in zip(x, step[j]))
        if y != expect:
            failures += 1
            print(f"  DEGREE h{j} * {r['source_name']} {x} -> {r['target_name']} {y}")
        elif h[j].get(x, 0) == 0:
            failures += 1
            print(f"  ZERO h{j} * {r['source_name']} {x} -> {r['target_name']} {y}")
print(f"  {checked} extensions in range, all others nonzero")

print("== Euler characteristic against classical Ext")
classical = defaultdict(int)
have_t = set()
for r in rows(f"{out}/classical_rank.csv"):
    classical[(int(r["filtration"]), int(r["degree"]))] = int(r["rank"])
total = defaultdict(int)
for (n, s, w), d in rank.items():
    total[(s, n + s)] += d
checked = 0
# degree t is complete in the export when every (stem t - s, s) is in range
for t in range(0, min(max_n, max_s) + 1):
    chi = sum((-1) ** s * total[(s, t)] for s in range(t + 1))
    chi_cl = sum((-1) ** s * classical[(s, t)] for s in range(t + 1))
    checked += 1
    if chi != chi_cl:
        failures += 1
        print(f"  EULER t = {t}: Cτ {chi}, classical {chi_cl}")
    for s in range(t + 1):
        if total[(s, t)] < classical[(s, t)]:
            failures += 1
            print(f"  RANK (s, t) = ({s}, {t}): Cτ {total[(s, t)]} < classical {classical[(s, t)]}")
print(f"  {checked} internal degrees complete in range, all others agree")

print("FAILURES:", failures)
sys.exit(1 if failures else 0)
