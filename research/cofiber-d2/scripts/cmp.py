"""Compares two d2 files (`stem,Adams_filtration,index,target`) on their common classes, through a
stem bound.

usage: cmp.py <label> <a.csv> <b.csv> [max stem]
"""
import csv
import sys

label, pa, pb = sys.argv[1:4]
max_n = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9


def load(p):
    return {tuple(map(int, r[:3])): r[3].replace(' ', '') for r in csv.reader(open(p))
            if r[0] != 'stem' and int(r[0]) <= max_n}


a, b = load(pa), load(pb)
c = set(a) & set(b)
d = sorted(k for k in c if a[k] != b[k])
print(f"{label}: {len(a)} vs {len(b)}, common {len(c)}, differ {len(d)} {d[:8]}")
