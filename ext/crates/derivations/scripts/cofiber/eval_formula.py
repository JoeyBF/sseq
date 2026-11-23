"""Evaluate a fixed cross-term formula (a sum of named operators) against the ground truth, on
kernel cochains, split into d2-cycles and non-cycles, modulo B2 + h_j E2, and exactly.

usage: eval_formula.py <truth csv> <cone csv> <sphere d2 csv> <max stem> <feature csv>... -- <j>:<op>+<op>...
"""
import csv
import re
import sys
from collections import defaultdict

args = sys.argv[1:]
sep = args.index("--")
truth_p, cone_p, sd2_p, max_n = args[:4]
max_n = int(max_n)
feat_ps = args[4:sep]
specs = {int(x.split(":")[0]): x.split(":", 1)[1].split("+") for x in args[sep + 1:]}


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
Eh, dims = {}, defaultdict(int)
for r in csv.DictReader(open(cone_p)):
    k = (int(r["j"]), int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))
    Eh[k] = mask(vec(r["E"]))
    dims[k[:3]] = max(dims[k[:3]], k[3] + 1)
D2 = {}
for r in csv.DictReader(open(sd2_p)):
    D2[(int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = mask(vec(r["target"]))
PRED = defaultdict(int)
pat = re.compile(r'^(\d+),(\d+),(\d+),(\d+),(.*),"(\[.*\])"$')
for fp_ in feat_ps:
    for line in open(fp_):
        m = pat.match(line.strip())
        if not m:
            continue
        j = int(m.group(1))
        if m.group(5) in specs.get(j, []):
            PRED[(j, int(m.group(2)), int(m.group(3)), int(m.group(4)))] ^= mask(vec(m.group(6)))


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


for j in sorted(specs):
    shift = 1 << j
    stats = defaultdict(int)
    fails = []
    for (n, s) in sorted({(k[1], k[2]) for k in K if k[0] == j}):
        if n > max_n:
            continue
        d = dims.get((j, n, s), 0)
        if d == 0:
            continue
        piv, ker = {}, []
        for l in range(d):
            v, c = Eh.get((j, n, s, l), 0), 1 << l
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
        base = [D2.get((tn + 1, ts - 2, k), 0) for k in range(dims.get((0, tn + 1, ts - 2), 0))]
        base += [Eh.get((j, n, s + 1, k), 0) for k in range(dims.get((j, n, s + 1), 0))]
        red = reducer(base)
        for z in ker:
            w = 0
            for l in bits(z):
                w ^= D2.get((n, s, l), 0)
            kind = "cycle" if w == 0 else "non-cycle"
            truth = pred = 0
            for l in bits(z):
                truth ^= K.get((j, n, s, l), 0)
                pred ^= PRED.get((j, n, s, l), 0)
            stats[kind] += 1
            if truth == pred:
                stats[kind + " exact"] += 1
            if red(truth ^ pred) == 0:
                stats[kind + " mod B2+hjE2"] += 1
            else:
                fails.append((n, s, kind))
    print(f"j={j} {'+'.join(specs[j])}: {dict(stats)}; failures {fails[:10]}")
