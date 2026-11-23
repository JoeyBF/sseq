"""Fit the cofiber cross term K as a fixed F2-combination of local cochain operators.

For each j, solves for coefficients c_f (one per feature, the same at every bidegree) with
    sum_f c_f Phi_f(z) == T(z)   modulo B2 + h_j E2 at the target bidegree,
over sample cochains z in ker(h_j) (non-cycles; optionally cycles too). Targets:
    A: T = K_true            B: T = K_true - K_h (the h_j-side value)
Trains on stems <= split and reports the held-out failures above it.

usage: fit_features.py <truth csv> <cone csv> <features csv> <sphere d2 csv> <max stem> <split>
"""
import csv
import re
import sys
from collections import defaultdict

truth_p, cone_p, feat_p, sd2_p, max_n, split = sys.argv[1:7]
max_n, split = int(max_n), int(split)


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
F = defaultdict(dict)  # (j, n, s) -> feature -> [row mask per x]
names = defaultdict(set)
pat = re.compile(r'^(\d+),(\d+),(\d+),(\d+),(.*),"(\[.*\])"$')
for line in open(feat_p):
    m = pat.match(line.strip())
    if not m:
        continue
    j, n, s, x = (int(m.group(i)) for i in range(1, 5))
    F[(j, n, s)].setdefault(m.group(5), {})[x] = mask(vec(m.group(6)))
    names[j].add(m.group(5))


def dims(n, s):
    d = 0
    while (0, n, s, d) in Lh:
        d += 1
    return d


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
        out = 0
        while v:
            p = v.bit_length() - 1
            if p in piv:
                v ^= piv[p]
            else:
                out |= 1 << p
                v ^= 1 << p
        return out

    return red


def lin(table, key, v):
    out = 0
    for l in bits(v):
        out ^= table.get(key + (l,), 0)
    return out


def samples(j, with_cycles):
    shift = 1 << j
    out = []
    for (jj, n, s) in sorted({(k[0], k[1], k[2]) for k in K if k[0] == j}):
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
        base += [Eh.get((j, n, s + 1, k), 0) for k in range(dims(n, s + 1))]
        red = reducer(base)
        for z in ker:
            w = 0
            for l in bits(z):
                w ^= D2.get((n, s, l), 0)
            if w == 0 and not with_cycles:
                continue
            out.append((n, s, z, red))
    return out


def solve(rows, nvar):
    """rows: list of (coef mask over vars, rhs bit). Returns a solution mask or None."""
    piv = {}
    for c, r in rows:
        while c:
            p = c.bit_length() - 1
            if p in piv:
                c ^= piv[p][0]
                r ^= piv[p][1]
            else:
                piv[p] = (c, r)
                break
        if c == 0 and r:
            return None
    sol = 0
    for p in sorted(piv):
        c, r = piv[p]
        val = r
        rest = c ^ (1 << p)
        for q in bits(rest):
            val ^= sol >> q & 1
        if val:
            sol |= 1 << p
    return sol


import os
QS = {"P(1)", "P(0, 1)", "P(0, 0, 1)", "P(0, 0, 0, 1)"}
FIXED = {0: ["H:P(1)"], 1: ["H:P(2)"], 2: ["H:P(4)", "CARRY"], 3: ["H:P(8)", "CARRY"]}
for j in range(4):
    feats = sorted(f for f in names[j] if f.startswith("d:") and f.split("|")[1] in QS)
    for variant, with_cycles in (("A (all kernel)", True),):
        smp = samples(j, with_cycles)
        train = [t for t in smp if t[0] <= split]
        test = [t for t in smp if t[0] > split]

        def equations(ts):
            eqs = []
            for n, s, z, red in ts:
                fz = []
                for f in feats:
                    rows = F[(j, n, s)].get(f, {})
                    v = 0
                    for l in bits(z):
                        v ^= rows.get(l, 0)
                    fz.append(red(v))
                tgt = lin(K, (j, n, s), z)
                for f in FIXED[j]:
                    rows = F[(j, n, s)].get(f, {})
                    for l in bits(z):
                        tgt ^= rows.get(l, 0)
                tgt = red(tgt)
                coords = tgt
                for v in fz:
                    coords |= v
                for c in bits(coords):
                    cm = sum(1 << i for i, v in enumerate(fz) if v >> c & 1)
                    eqs.append((cm, tgt >> c & 1, (n, s, z)))
            return eqs

        tr = equations(train)
        sol = solve([(c, r) for c, r, _ in tr], len(feats))
        # dimension of the solution space: rank of the homogeneous system
        piv = {}
        for c, r, _ in tr:
            while c:
                p = c.bit_length() - 1
                if p in piv:
                    c ^= piv[p]
                else:
                    piv[p] = c
                    break
        print(f"   j={j}: {len(feats)} Q_k features, solution space dimension {len(feats) - len(piv)}")
        if sol is None:
            print(f"j={j} {variant}: {len(train)} train samples, NO solution with {len(feats)} features")
            continue
        bad = {lab for c, r, lab in equations(test) if (bin(c & sol).count("1") & 1) != r}
        used = [feats[i] for i in bits(sol)]
        print(f"j={j} {variant}: {len(train)} train / {len(test)} test samples; held-out failures "
              f"{len(bad)}; solution uses {len(used)} of {len(feats)}: {used}")
