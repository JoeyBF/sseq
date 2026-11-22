"""Writes the j module directory the derivations engine reads, from the restrict/ j data.

- j_rank.csv: Ext ranks, counted from the resolution's generators (iota_j_basis.csv);
- j_indaction.csv: h0..h3 acting on Ext(j) (iota_j_hB.csv); h_i is the only class at (2^i - 1, 1);
- j_unknown_all: every other indecomposable product is unknown, not zero;
- j_istar.csv, j_pstar.csv: empty (j is not a two-cell complex; the unit is a `map` line).
"""
import csv
import os
import sys
from collections import Counter

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

rank = Counter()
with open(f"{src}/iota_j_basis.csv") as f:
    for row in csv.DictReader(f):
        rank[(int(row["stem"]), int(row["Adams_filtration"]))] += 1
with open(f"{out}/j_rank.csv", "w") as f:
    f.write("stem,Adams_filtration,rank\n")
    for (n, s), r in sorted(rank.items()):
        f.write(f"{n},{s},{r}\n")

n_act = 0
with open(f"{src}/iota_j_hB.csv") as f, open(f"{out}/j_indaction.csv", "w") as g:
    g.write("stem_1,Adams_filtration_1,index_1,stem_2,Adams_filtration_2,index_2,product\n")
    rd = csv.reader(f)
    next(rd)
    for i, n, s, k, prod in rd:
        g.write(f"{(1 << int(i)) - 1},1,0,{n},{s},{k},{prod.replace(' ', '')}\n")
        n_act += 1

open(f"{out}/j_unknown_all", "w").close()
for name in ("istar", "pstar"):
    with open(f"{out}/j_{name}.csv", "w") as f:
        f.write("stem,Adams_filtration,index,image\n")
print(f"{len(rank)} bidegrees, {sum(rank.values())} classes, {n_act} h_i products")
