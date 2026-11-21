"""Sizes the cochains of `Cτ/I_n` (the quotients `Q/(q_0, ..., q_{n-1})`) from the ranks of a backbone.

    python3 -I ctau_plan_quotients.py <rank.csv> <max motivic stem>...

For each `n`, the cochains of filtration a and weight u have dimension sum_t gens(a, t) *
#{monomials in q_n, q_{n+1}, ... of degree u - t}; `n = 0` is `Cτ` itself. The report is the
total, the largest space, and the sum of rows * cols over the coboundaries, to which the entries
of the coboundaries are proportional at a given density.
"""

import csv
import sys
from collections import defaultdict

path = sys.argv[1]
gens = defaultdict(int)
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        n, s = int(r["stem"]), int(r["Adams_filtration"])
        gens[(s, n + s)] = int(r["rank"])
max_a = max(a for a, _ in gens)
by_a = defaultdict(list)
for (a, t), g in gens.items():
    by_a[a].append((t, g))

for max_n in map(int, sys.argv[2:]):
    max_u = (max_n + max_a + 2) // 2
    print(f"stem <= {max_n}")
    for first in range(0, 6):
        # monomials of each degree in q_first, q_{first+1}, ...; q_0 has degree 0 and is the
        # polynomial degree itself, so it does not multiply the count
        count = [0] * (max_u + 1)
        count[0] = 1
        i = max(first, 1)
        while (1 << i) - 1 <= max_u:
            part = (1 << i) - 1
            for e in range(part, max_u + 1):
                count[e] += count[e - part]
            i += 1
        total = area = largest = 0
        prev = {}
        for a in range(0, max_a + 2):
            cur = {}
            for u in range(0, (max_n + a + 1) // 2 + 1):
                d = sum(g * count[u - t] for t, g in by_a[a] if t <= u)
                cur[u] = d
                total += d
                largest = max(largest, d)
                area += prev.get(u, 0) * d
            prev = cur
        print(f"  n = {first}: {total:.3e} cochains, largest space {largest:.3e},"
              f" sum of rows * cols {area:.3e}")
