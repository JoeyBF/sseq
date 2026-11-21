"""Checks the export of `Cτ/I_{n+1}` against that of `Cτ/I_n` through the long exact sequence of q_n.

    python3 -I ctau_quotient_check.py <export of Cτ/I_n> <export of Cτ/I_{n+1}> <n>

`0 -> Q/I_n -> Q/I_n -> Q/I_{n+1} -> 0`, with the first map multiplication by `q_n`, gives a long
exact sequence in which `q_n` raises the motivic tridegree by `(2d, 1, d)`, `d = 2^n - 1`, and
the connecting map lowers the stem by `2d + 1` and the weight by `d`. So the rank of `Cτ/I_{n+1}`
at a tridegree is the corank of `q_n` into it plus the nullity of `q_n` out of the tridegree the
connecting map lands in. Both exports must be of the quotient kind (every tower of length one),
with `Ctau_q.csv` in the first.
"""

import csv
import sys
from collections import defaultdict

lower, upper, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = (1 << n) - 1


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def towers(path):
    where, basis = {}, defaultdict(list)
    for r in rows(path):
        x = (int(r["stem"]), int(r["filtration"]), int(r["weight"]))
        where[r["id"]] = (x, len(basis[x]))
        basis[x].append(r["id"])
    return where, basis


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


where, basis = towers(f"{lower}/Ctau_towers.csv")
_, quotient = towers(f"{upper}/Ctau_towers.csv")
max_n = max(x[0] for x in basis)
# images[x]: the images of the basis classes at x under q_n, as bit masks at x + (2d, 1, d)
images = defaultdict(list)
for r in rows(f"{lower}/Ctau_q.csv"):
    x, _ = where[r["source"]]
    v = 0
    for t in r["targets"].split():
        v |= 1 << where[t][1]
    images[x].append(v)

checked = bad = 0
for x in sorted(set(basis) | set(quotient)):
    stem, s, w = x
    if stem > max_n:
        continue
    into = (stem - 2 * d, s - 1, w - d)
    corank = len(basis.get(x, [])) - rank_of(images.get(into, []))
    y = (stem - 2 * d - 1, s, w - d)
    nullity = len(basis.get(y, [])) - rank_of(images.get(y, []))
    checked += 1
    if len(quotient.get(x, [])) != corank + nullity:
        bad += 1
        if bad <= 5:
            print(f"  at {x}: rank {len(quotient.get(x, []))}, corank {corank}, nullity {nullity}")
print(f"q_{n}: {checked} tridegrees, {sum(len(b) for b in quotient.values())} classes of the"
      f" quotient, {bad} inexact")
sys.exit(1 if bad else 0)
