"""d2 ranks per source bidegree of a d2 file, through (max stem, max f), printed as a dict.

usage: ranks.py <d2 csv> <max stem> <max f>
"""
import csv
import sys
from collections import defaultdict

path, max_n, max_f = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = defaultdict(list)
for r in csv.reader(open(path)):
    if r[0] == 'stem':
        continue
    n, f = int(r[0]), int(r[1])
    if n <= max_n and f <= max_f:
        v = [int(x) for x in r[3].strip('"[] ').split(',') if x.strip()]
        rows[(n, f)].append(sum(b << i for i, b in enumerate(v)))


def rank(vs):
    piv = []
    for v in vs:
        for p in piv:
            v = min(v, v ^ p)
        if v:
            piv.append(v)
    return len(piv)


print({k: rank(v) for k, v in sorted(rows.items()) if rank(v)})
