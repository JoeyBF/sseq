"""Expands a tower export of `examples/ctau_scale.rs` into the per-tridegree format of `ctau_e2.rs`.

    python3 -I ctau_legacy.py <tower export> <output directory> <max stem> <max filtration>

The basis of a tridegree is the classes `h_0^i g` it holds, ordered by the id of the tower, which
is an opaque string (a number from the in-memory driver, `filtration_weight_index` from the
spilling one). Every
map of the tower export commutes with `h_0`, so its value on `h_0^i g` is the sum of the targets
of `g` whose towers still hold a class in the tridegree of the image. The output can be checked
with `ctau_check.py` and `ctau_compare.py`; it is for small ranges.
"""

import csv
import os
import sys
from collections import defaultdict

src, out, max_n, max_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
names = [n for n in ["Ctau", "C2", "Ceta", "Cnu", "Csigma"] if os.path.exists(f"{src}/{n}_towers.csv")]
hdeg = {0: (0, 1, 0), 1: (1, 1, 1), 2: (3, 1, 2), 3: (7, 1, 4), 4: (15, 1, 8)}
top = {"C2": (1, 0, 0), "Ceta": (2, 0, 1), "Cnu": (4, 0, 2), "Csigma": (8, 0, 4)}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load(name):
    """towers[id] = (stem, first filtration, weight, length), and the basis of each tridegree."""
    towers = {}
    basis = defaultdict(list)
    for r in rows(f"{src}/{name}_towers.csv"):
        n, s, w = int(r["stem"]), int(r["filtration"]), int(r["weight"])
        length = max_s + 1 - s if r["length"] == "inf" else int(r["length"].rstrip("+"))
        towers[r["id"]] = (n, s, w, length)
        if n > max_n:
            continue
        for i in range(length):
            if s + i <= max_s:
                basis[(n, s + i, w)].append(r["id"])
    for b in basis.values():
        b.sort()
    return towers, basis


def vector(ids, basis):
    return '"[' + ", ".join("1" if b in ids else "0" for b in basis) + ']"'


def expand(w, prefix, shift, sources, targets, table):
    """Writes the map with value `table[g]` on generators, shifting tridegrees by `shift`."""
    s_towers, s_basis = sources
    _, t_basis = targets
    for g, image in table.items():
        n, s, wt, length = s_towers[g]
        for i in range(length):
            x = (n, s + i, wt)
            y = (x[0] + shift[0], x[1] + shift[1], x[2] + shift[2])
            if x not in s_basis or y not in t_basis or y[0] > max_n or y[1] > max_s:
                continue
            ids = set(image) & set(t_basis[y])
            if ids:
                w.write(f"{prefix}{x[0]},{x[1]},{x[2]},{s_basis[x].index(g)},{vector(ids, t_basis[y])}\n")


data = {name: load(name) for name in names}
for name in names:
    towers, basis = data[name]
    with open(f"{out}/{name}_rank.csv", "w") as w:
        w.write("stem,filtration,weight,rank\n")
        for x in sorted(basis):
            w.write(f"{x[0]},{x[1]},{x[2]},{len(basis[x])}\n")
    with open(f"{out}/{name}_h.csv", "w") as w:
        w.write("j,stem,filtration,weight,index,product\n")
        # h_0 moves one step up each tower
        if name != "C2":
            expand(w, "0,", hdeg[0], data[name], data[name], {g: [g] for g in towers})
        by_j = defaultdict(dict)
        for r in rows(f"{src}/{name}_h.csv"):
            by_j[int(r["j"])][r["source"]] = r["targets"].split()
        for j in sorted(by_j):
            expand(w, f"{j},", hdeg[j], data[name], data[name], by_j[j])
    if name == "Ctau":
        continue
    for kind, a, b, sign in (("istar", "Ctau", name, 0), ("pstar", name, "Ctau", -1)):
        table = {r["source"]: r["targets"].split() for r in rows(f"{src}/{name}_{kind}.csv")}
        shift = tuple(sign * c for c in top[name])
        with open(f"{out}/{name}_{kind}.csv", "w") as w:
            w.write("stem,filtration,weight,index,image\n")
            expand(w, "", shift, data[a], data[b], table)
