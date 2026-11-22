"""For non-cycle kernel cochains z: the true K(z) against cheap features.

Writes one row per (j, z) with: the truth K(z), the h_j-side value, d2 z = w, and whether the
residual (truth - h_j side) lies in each candidate span, modulo B2 + h_j E2:
  h_{j-1} E2, h0 E2, h1 E2, h2 E2, h3 E2 (products from the cone file's E column).

usage: residuals.py <truth csv> <cone csv> <sphere d2 csv> <max stem>
"""
import csv
import sys

truth_p, cone_p, sd2_p, max_n = sys.argv[1:5]
max_n = int(max_n)


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


K = {}
for r in csv.DictReader(open(truth_p)):
    K[(int(r["j"]), int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = mask(vec(r["bottom"]))
Lh, Eh = {}, {}
for r in csv.DictReader(open(cone_p)):
    k = (int(r["j"]), int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))
    Lh[k], Eh[k] = mask(vec(r["L"])), mask(vec(r["E"]))
D2 = {}
for r in csv.DictReader(open(sd2_p)):
    D2[(int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = mask(vec(r["target"]))


def dims(n, s):
    d = 0
    while (0, n, s, d) in Lh:
        d += 1
    return d


def lin(table, key, v):
    out = 0
    for l in bits(v):
        out ^= table.get(key + (l,), 0)
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


def products(i, n, s):
    """h_i times the basis at (n, s), landing at (n + 2^i - 1, s + 1)."""
    return [Eh.get((i, n, s, k), 0) for k in range(dims(n, s))]


for j in range(4):
    shift = 1 << j
    for (n, s) in sorted({(k[1], k[2]) for k in K if k[0] == j}):
        if n > max_n:
            continue
        d = dims(n, s)
        if d == 0 or any((j, n, s, l) not in Lh for l in range(d)):
            continue
        piv, ker = {}, []
        for l in range(d):
            v, c = Eh[(j, n, s, l)], 1 << l
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
        base = [D2.get((tn + 1, ts - 2, k), 0) for k in range(dims(tn + 1, ts - 2))]
        base += products(j, n, s + 1)
        red = reducer(base)
        for z in ker:
            w = lin(D2, (), z) if False else 0
            for l in bits(z):
                w ^= D2.get((n, s, l), 0)
            if w == 0:
                continue
            r = red(lin(K, (j, n, s), z) ^ lin(Lh, (j, n, s), z))
            if r == 0:
                continue
            spans = {}
            for i in range(4):
                src = (tn - (1 << i) + 1, ts - 1)
                spans[f"h{i}"] = reducer(base + products(i, *src))(r) == 0
            print(f"j={j} z=({n},{s}):{z:b} w={w:b} residual {r:b} at ({tn},{ts}); in span: "
                  + " ".join(k for k, v in spans.items() if v))
