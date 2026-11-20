"""Sizes the coboundaries of the Cτ computation of `src/ctau.rs` from the ranks of a backbone.

    python3 -I ctau_plan.py <rank.csv> <max motivic stem> [<max polynomial degree k>]

`rank.csv` lists `stem,Adams_filtration,rank` of the classical resolution. Nothing is built: the
cochains of filtration a and weight u have dimension sum_t gens(a, t) * #{monomials in q_1, q_2, ...
of degree u - t and length <= k}, and the coboundary out of them is one matrix. The report is the
largest matrix, the number of matrices by size, and the total of rows * cols * min(rows, cols),
which is proportional to the cost of reducing them densely.

A bound on k bounds the Chow degree stem + filtration - 2 weight, not the stem: the classes of
Chow degree at most k through the stem are exactly what the bounded computation gives.
"""

import csv
import sys
from collections import defaultdict

# One core of `fp`'s row reduction on a dense full-rank matrix, per rows * cols * min(rows, cols);
# see MOTIVIC_CTAU.md for the measurement.
SECONDS_PER_UNIT = 1.5e-12

path, max_n = sys.argv[1], int(sys.argv[2])
max_k = int(sys.argv[3]) if len(sys.argv) > 3 else None

gens = defaultdict(int)
top_stem = 0
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        n, s = int(r["stem"]), int(r["Adams_filtration"])
        gens[(s, n + s)] = int(r["rank"])
        top_stem = max(top_stem, n)
max_a = max(a for a, _ in gens)

# the largest weight read at any filtration, and the stems of the generators that takes
max_u = (max_n + max_a + 2) // 2
need = (max_n + 1) // 2
if need > top_stem:
    sys.exit(f"stem {max_n} needs the backbone through stem {need}, it has {top_stem}")

# count[e][j]: monomials in q_1, q_2, ... of degree e and length exactly j
max_len = max_u if max_k is None else max_k
count = [[0] * (max_len + 1) for _ in range(max_u + 1)]
count[0][0] = 1
i = 1
while (1 << i) - 1 <= max_u:
    part = (1 << i) - 1
    for e in range(part, max_u + 1):
        row, below = count[e], count[e - part]
        for j in range(1, max_len + 1):
            row[j] += below[j - 1]
    i += 1
upto = [sum(row) for row in count]

by_a = defaultdict(list)
for (a, t), g in gens.items():
    by_a[a].append((t, g))


def dim(a, u):
    return sum(g * upto[u - t] for t, g in by_a[a] if t <= u)


largest = (0, 0, 0, 0)
cost = 0
sizes = defaultdict(int)
cochains = 0
for a in range(0, max_a + 1):
    for u in range(0, (max_n + a + 1) // 2 + 1):
        rows, cols = dim(a, u), dim(a + 1, u)
        cochains += rows
        if rows == 0 or cols == 0:
            continue
        cost += rows * cols * min(rows, cols)
        sizes[max(rows, cols).bit_length()] += 1
        if rows * cols > largest[0] * largest[1]:
            largest = (rows, cols, a, u)

print(f"stem <= {max_n}, k <= {'any' if max_k is None else max_k}")
print(f"  cochains in total: {cochains}")
print(f"  largest coboundary: {largest[0]} x {largest[1]} at a = {largest[2]}, u = {largest[3]}"
      f" ({largest[0] * largest[1] / 8 / 2**30:.1f} GiB dense)")
print(f"  sum of rows * cols * min: {cost:.3e}"
      f" (about {cost * SECONDS_PER_UNIT / 3600:.3g} CPU-hours of dense reduction)")
for b in sorted(sizes):
    print(f"  side < 2^{b}: {sizes[b]}")
