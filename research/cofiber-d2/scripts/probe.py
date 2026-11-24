"""Each feature's i_* contribution to d2 of one lift of C(h_j).

usage: probe.py <slots dir> <name> <k> <features csv> <N> <f> <index>
"""
import csv
import sys
from collections import defaultdict

slots, name, k, feat_p, N, f, idx = sys.argv[1:8]
k, N, f, idx = int(k), int(N), int(f), int(idx)


def mask(s):
    s = s.strip().strip('"').strip('[]')
    return sum(int(t) << i for i, t in enumerate(s.split(','))) if s.strip() else 0


def load(path):
    d = defaultdict(dict)
    for r in csv.reader(open(path)):
        if r[0] != 'stem':
            d[(int(r[0]), int(r[1]))][int(r[2])] = mask(r[3])
    return d


istar, pstar = load(f"{slots}/{name}_istar.csv"), load(f"{slots}/{name}_pstar.csv")
y = pstar[(N, f)][idx]
yb, tgt = (N - k, f), (N - 1, f + 2)


def pull(tab, v):
    out, i = 0, 0
    while v:
        if v & 1:
            out ^= tab.get(i, 0)
        v >>= 1
        i += 1
    return out


print(f"lift ({N},{f},{idx}): y = {y:b} at {yb}")
for line in open(feat_p):
    if line.startswith('feature,'):
        continue
    head, vec = line.rstrip('\n').split('"', 1)
    fname, n, s_, yi, _ = head.rsplit(',', 4)
    if (int(n), int(s_)) != yb or not (y >> int(yi) & 1):
        continue
    print(f"  {fname} on y_{yi}: g-row {mask(vec):b}")
feats = defaultdict(int)
for line in open(feat_p):
    if line.startswith('feature,'):
        continue
    head, vec = line.rstrip('\n').split('"', 1)
    fname, n, s_, yi, _ = head.rsplit(',', 4)
    if (int(n), int(s_)) == yb and y >> int(yi) & 1:
        feats[fname] ^= mask(vec)
for F, v in sorted(feats.items()):
    print(f"  {F}: cochain {v:b} -> i_* {pull(istar.get(tgt, {}), v):b}")
