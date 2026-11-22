"""Compare the true cofiber cross term K (cofiber_truth) with what is cheap or already known.

  A. top part == d2_S(e) for every sphere basis cochain e (sanity of the truth run)
  B. on d2-cycles y: K(y) == z-side exact L(y) (cofiber_easy_cycles)?
  C. on kernel cochains z (h_j z = 0) that are not d2-cycles: K(z) vs the h_j-side L(z)
     (cofiber_cone), classified mod B2 and mod B2 + h_j E2; residuals listed for fitting.

usage: fit.py <truth csv> <easy out> <cone csv> <sphere d2 csv> <j> <max stem>
"""
import csv
import re
import sys
from collections import defaultdict

truth_p, easy_p, cone_p, sd2_p, j, max_n = sys.argv[1:7]
j, max_n = int(j), int(max_n)
shift = 1 << j


def vec(s):
    return [int(x) for x in s.strip('[]" ').split(",") if x.strip()]


def mask(v):
    return sum(1 << i for i, x in enumerate(v) if x)


def bits(v):
    l = 0
    while v:
        if v & 1:
            yield l
        v >>= 1
        l += 1


K, TOP = {}, {}
for r in csv.DictReader(open(truth_p)):
    if int(r["j"]) == j:
        k = (int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))
        K[k], TOP[k] = mask(vec(r["bottom"])), mask(vec(r["top"]))
Lh, Eh = {}, {}
for r in csv.DictReader(open(cone_p)):
    if int(r["j"]) == j:
        k = (int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))
        Lh[k], Eh[k] = mask(vec(r["L"])), mask(vec(r["E"]))
D2 = {}
for r in csv.DictReader(open(sd2_p)):
    D2[(int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = mask(vec(r["target"]))


def lin(table, n, s, v):
    out = 0
    for l in bits(v):
        out ^= table.get((n, s, l), 0)
    return out


def reducer(gens):
    piv = {}
    for g in gens:
        while g:
            p = g.bit_length() - 1
            if p in piv:
                g ^= piv[p]
            else:
                piv[p] = g
                break

    def red(v):
        while v:
            p = v.bit_length() - 1
            if p not in piv:
                return v
            v ^= piv[p]
        return 0

    return red


def dims(n, s):
    d = 0
    while (n, s, d) in K or (n, s, d) in Lh:
        d += 1
    return d


# A
bad_top = sum(1 for (n, s, l), t in K.items() if n <= max_n and TOP[(n, s, l)] != D2.get((n, s, l), 0))
print(f"A. top part != d2_S on {bad_top} of {sum(1 for k in K if k[0] <= max_n)} basis cochains")

# B
pat = re.compile(r"^\[(.*)\] \[h(\d+)\] = \[(.*)\] \+ λ \[(.*)\]$")
eq = ne = 0
bcls = defaultdict(int)
for line in open(easy_p):
    m = pat.match(line.strip())
    if not m or int(m.group(2)) != j:
        continue
    y, n, s = 0, None, None
    for term in m.group(1).split(" + "):
        a, b, c = (int(x) for x in term.strip()[3:-1].split(","))
        n, s, y = a, b, y ^ (1 << c)
    if n > max_n or any((n, s, l) not in K for l in bits(y)):
        continue
    if mask(vec(m.group(3))):
        continue  # h_j y != 0: no cofiber class over y
    r = lin(K, n, s, y) ^ mask(vec(m.group(4)))
    if r == 0:
        eq += 1
        continue
    tn, ts = n + shift - 1, s + 2
    b2 = [D2.get((tn + 1, ts - 2, k), 0) for k in range(dims(tn + 1, ts - 2))]
    hj = [Eh.get((n, s + 1, k), 0) for k in range(dims(n, s + 1))]
    if reducer(b2)(r) == 0:
        bcls["B2"] += 1
    elif reducer(b2 + hj)(r) == 0:
        bcls["B2 + h_j E2"] += 1
    else:
        bcls["other"] += 1
        if bcls["other"] <= 5:
            print(f"   y at ({n},{s}) = {y:b}: residual {reducer(b2 + hj)(r):b}")
    ne += 1
print(f"B. on d2-cycles: K == z-side L on {eq}, differs on {ne}: {dict(bcls)}")

# C
cnt = defaultdict(int)
examples = []
for (n, s) in sorted({(k[0], k[1]) for k in K}):
    if n > max_n:
        continue
    d = dims(n, s)
    if any((n, s, l) not in Lh for l in range(d)):
        continue
    # kernel of h_j at (n, s), as masks
    rows = [(Eh[(n, s, l)], 1 << l) for l in range(d)]
    piv, ker = {}, []
    for v, c in rows:
        while v:
            p = v.bit_length() - 1
            if p in piv:
                v ^= piv[p][0]
                c ^= piv[p][1]
            else:
                piv[p] = (v, c)
                break
        if v == 0:
            ker.append(c)
    tn, ts = n + shift - 1, s + 2
    b2 = [D2.get((tn + 1, ts - 2, k), 0) for k in range(dims(tn + 1, ts - 2))]
    hj = [Eh.get((n, s + 1, k), 0) for k in range(dims(n, s + 1))]
    red_b2, red_all = reducer(b2), reducer(b2 + hj)
    for z in ker:
        if lin(D2, n, s, z) == 0:
            continue  # a d2-cycle: case B
        r = lin(K, n, s, z) ^ lin(Lh, n, s, z)
        if r == 0:
            cnt["equal"] += 1
        elif red_b2(r) == 0:
            cnt["differ by B2"] += 1
        elif red_all(r) == 0:
            cnt["differ by B2 + h_j E2"] += 1
        else:
            cnt["differ otherwise"] += 1
            examples.append((n, s, z, red_all(r), lin(D2, n, s, z)))
print("C. non-cycle kernel cochains, true K vs h_j side:", dict(cnt))
for n, s, z, r, w in examples[:15]:
    print(f"   z at ({n},{s}) = {z:b}: residual {r:b} at ({n + shift - 1},{s + 2}); d2 z = {w:b} at ({n - 1},{s + 2})")
