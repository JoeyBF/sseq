"""Fits the exact bottom-cell block of C(h_j) at the Ext level by local features.

For every lift y~ (cofiber_slots basis, y = p_* of it), the exact i_* part of d2 is i_*(Phi^* y),
read from cone_d2's delta' output. A formula sum_F c_F F gives i_*(sum c_F F^* y). Solves for c
over F2 and prints the solution(s) and the classes no solution fits.

usage: fit.py <slots dir> <name> <k> <features csv> <exact cone csv> <max stem> <max f>
"""
import csv
import sys
from collections import defaultdict

slots, name, k, feat_p, exact_p, max_n, max_f = sys.argv[1:8]
k, max_n, max_f = int(k), int(max_n), int(max_f)


def mask(s):
    s = s.strip().strip('"').strip('[]')
    return sum(int(t) << i for i, t in enumerate(s.split(','))) if s.strip() else 0


istar = defaultdict(dict)
for r in csv.reader(open(f"{slots}/{name}_istar.csv")):
    if r[0] != 'stem':
        istar[(int(r[0]), int(r[1]))][int(r[2])] = mask(r[3])
pstar = defaultdict(dict)
for r in csv.reader(open(f"{slots}/{name}_pstar.csv")):
    if r[0] != 'stem':
        pstar[(int(r[0]), int(r[1]))][int(r[2])] = mask(r[3])

# feature -> (n, s) of y -> y index -> row mask over g
feat = defaultdict(lambda: defaultdict(dict))
for line in open(feat_p):
    if line.startswith('feature,'):
        continue
    head, vec = line.rstrip('\n').split('"', 1)
    fname, n, s_, y, _ = head.rsplit(',', 4)
    feat[fname][(int(n), int(s_))][int(y)] = mask(vec)
exact = defaultdict(dict)
for r in csv.reader(open(exact_p)):
    if r[0] == 'module':
        continue
    exact[(int(r[1]), int(r[2]))][int(r[3])] = mask(r[8])


def pull(tab, v):
    out, i = 0, 0
    while v:
        if v & 1:
            out ^= tab.get(i, 0)
        v >>= 1
        i += 1
    return out


names = sorted(feat)
rows = []  # (coefficient mask over features, rhs bit, label)
for (N, f), lifts in sorted(pstar.items()):
    if N > max_n or f > max_f:
        continue
    tgt = (N - 1, f + 2)
    for i, y in lifts.items():
        yb = (N - k, f)
        cols = [istar_of for istar_of in
                (pull(istar.get(tgt, {}), pull(feat[F].get(yb, {}), y)) for F in names)]
        rhs = pull(istar.get(tgt, {}), pull(exact.get(yb, {}), y))
        width = max([c.bit_length() for c in cols] + [rhs.bit_length()])
        for b in range(width):
            m = sum(1 << fi for fi, c in enumerate(cols) if c >> b & 1)
            r = rhs >> b & 1
            if m or r:
                rows.append((m, r, (N, f, i, b)))

# Gaussian elimination on [m | r]
piv = {}
bad = []
for m, r, lab in rows:
    v = m | (r << len(names))
    while v & ((1 << len(names)) - 1):
        p = (v & -v).bit_length() - 1
        if p in piv:
            v ^= piv[p]
        else:
            piv[p] = v
            break
    else:
        if v:
            bad.append(lab)
print(f"{name}: {len(rows)} equations over features {names}")
print(f"inconsistent equations: {len(bad)} {sorted(bad)[:10]}")
# back-substitute a particular solution (free variables 0)
sol = 0
for p in sorted(piv, reverse=True):
    v = piv[p]
    rhs = v >> len(names) & 1
    rest = v & ((1 << len(names)) - 1) & ~(1 << p)
    if (bin(rest & sol).count('1') + rhs) % 2:
        sol |= 1 << p
print("solution:", {F: sol >> i & 1 for i, F in enumerate(names)},
      "free:", [F for i, F in enumerate(names) if i not in piv])
