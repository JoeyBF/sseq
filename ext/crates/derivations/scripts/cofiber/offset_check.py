"""Is the h_j-side L (secondary lift of h_j's chain map, cofiber_cone) equal to the z-side L
(cofiber_easy_cycles, exact) modulo d2-boundaries + h_j * E2, on every d2-cycle?

usage: offset_check.py <cone csv> <easy out> <j> <sphere d2 csv> <max stem>
"""
import csv
import re
import sys
from collections import defaultdict

cone_path, easy_path, j, sd2_path, max_n = sys.argv[1:6]
j, max_n = int(j), int(max_n)
shift = 1 << j


def vec(s):
    return [int(x) for x in s.strip('[]" ').split(",") if x.strip()]


def mask(v):
    return sum(1 << i for i, x in enumerate(v) if x)


Lh, Eh = {}, {}
for row in csv.DictReader(open(cone_path)):
    if int(row["j"]) == j:
        k = (int(row["stem"]), int(row["Adams_filtration"]), int(row["index"]))
        Lh[k], Eh[k] = mask(vec(row["L"])), mask(vec(row["E"]))
d2s = defaultdict(dict)
for row in csv.DictReader(open(sd2_path)):
    d2s[(int(row["stem"]), int(row["Adams_filtration"]))][int(row["index"])] = mask(vec(row["target"]))


def lin(table, n, s, v):
    out, l = 0, 0
    while v:
        if v & 1:
            out ^= table.get((n, s, l), 0)
        v >>= 1
        l += 1
    return out


def span_reduce(gens, v):
    piv = {}
    for g in gens:
        while g:
            p = g.bit_length() - 1
            if p in piv:
                g ^= piv[p]
            else:
                piv[p] = g
                break
    while v:
        p = v.bit_length() - 1
        if p not in piv:
            return v
        v ^= piv[p]
    return 0


pat = re.compile(r"^\[(.*)\] \[h(\d+)\] = \[(.*)\] \+ λ \[(.*)\]$")
same = in_span = off = 0
for line in open(easy_path):
    m = pat.match(line.strip())
    if not m or int(m.group(2)) != j:
        continue
    y, n, s = 0, None, None
    for term in m.group(1).split(" + "):
        a, b, c = (int(x) for x in term.strip()[3:-1].split(","))
        n, s, y = a, b, y ^ (1 << c)
    if n > max_n:
        continue
    lz = mask(vec(m.group(4)))
    lh = lin(Lh, n, s, y)
    if lz == lh:
        same += 1
        continue
    tn, ts = n + shift - 1, s + 2  # where L lives
    gens = list(d2s.get((tn + 1, ts - 2), {}).values())  # d2 image into (tn, ts)
    nb = 0
    while (n, s + 1, nb) in Eh:  # h_j * E2(n, s + 1) lands in (tn, ts)
        gens.append(Eh[(n, s + 1, nb)])
        nb += 1
    if span_reduce(gens, lz ^ lh) == 0:
        in_span += 1
    else:
        off += 1
        print(f"y at ({n},{s}) {y:b}: h_j-side {lh:b}, z-side {lz:b}")
print(f"equal {same}, differ by B2 + h_j E2 {in_span}, differ otherwise {off}")
