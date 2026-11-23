"""The fitted cross-term formula against the easy-way L on kernel d2-cycles, over the whole range:
both should agree modulo B2 + h_j E2 wherever the formula is right.

usage: formula_vs_easy.py <features csv> <easy out> <cone csv (E column: h_j products)> <sphere d2>
                          <j> <max stem>
"""
import csv
import re
import sys
from collections import defaultdict

feat_p, easy_p, cone_p, sd2_p, j, max_n = sys.argv[1:7]
j, max_n = int(j), int(max_n)
shift = 1 << j
FORMULA = {0: ["H:P(1)"], 1: ["H:P(2)"], 2: ["H:P(4)", "d:P(2)|P(0, 1)", "d:P(2)|P(3)", "d:P(3)|P(2)"],
           3: ["H:P(8)", "d:P(4)|P(5)", "d:P(5)|P(4)", "d:P(0, 2)|P(0, 1)", "d:P(6)|P(0, 1)",
               "d:P(2, 2)|P(1)", "d:P(2)|P(0, 0, 1)"]}


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


K = defaultdict(int)
pat = re.compile(r'^(\d+),(\d+),(\d+),(\d+),(.*),"(\[.*\])"$')
for line in open(feat_p):
    m = pat.match(line.strip())
    if m and int(m.group(1)) == j and m.group(5) in FORMULA[j]:
        K[(int(m.group(2)), int(m.group(3)), int(m.group(4)))] ^= mask(vec(m.group(6)))
D2 = {}
for r in csv.DictReader(open(sd2_p)):
    D2[(int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = mask(vec(r["target"]))
Eh = defaultdict(dict)
for line in open(feat_p):
    m = pat.match(line.strip())
    if m and int(m.group(1)) == j and m.group(5) == "E:h":
        Eh[(int(m.group(2)), int(m.group(3)))][int(m.group(4))] = mask(vec(m.group(6)))


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


epat = re.compile(r"^\[(.*)\] \[h(\d+)\] = \[(.*)\] \+ λ \[(.*)\]$")
ok, bad, unknown = 0, [], 0
for line in open(easy_p):
    m = epat.match(line.strip())
    if not m or int(m.group(2)) != j or mask(vec(m.group(3))):
        continue
    y, n, s = 0, None, None
    for term in m.group(1).split(" + "):
        a, b, c = (int(x) for x in term.strip()[3:-1].split(","))
        n, s, y = a, b, y ^ (1 << c)
    if n > max_n:
        continue
    tn, ts = n + shift - 1, s + 2
    base = [D2.get((tn + 1, ts - 2, k), 0) for k in range(64) if (tn + 1, ts - 2, k) in D2]
    base += list(Eh.get((n, s + 1), {}).values())
    kz = 0
    for l in bits(y):
        kz ^= K.get((n, s, l), 0)
    if reducer(base)(kz ^ mask(vec(m.group(4)))) == 0:
        ok += 1
    else:
        bad.append((n, s))
print(f"j={j}: formula == easy L on {ok} kernel cycles, differs on {len(bad)}; first failures "
      f"{sorted(bad)[:15]}")
