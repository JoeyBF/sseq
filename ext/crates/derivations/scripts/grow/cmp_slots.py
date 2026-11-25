"""Check module_slots against cofiber_slots on a two-cell complex: i_* and p_* in both bases.

usage: cmp_slots.py <module_slots dir> <name> <cofiber_slots dir> <bottom cell> <top cell> <k> <max n>
"""
import csv
import sys

from slots import Slots


def vec(s):
    s = s.strip().strip('"').strip('[]')
    return sum(int(t) << i for i, t in enumerate(s.split(','))) if s.strip() else 0


d, name, cd, bot, top, k, max_n = sys.argv[1:8]
k, max_n = int(k), int(max_n)
sl = Slots(d, name)
bad = good = 0
for r in csv.DictReader(open(f"{cd}/{name}_istar.csv")):
    n, f, i = int(r['stem']), int(r['Adams_filtration']), int(r['index'])
    if n >= max_n or f > 30:
        continue
    mine = sl.coords((n, f), {(bot, i)})
    if mine == vec(r['image']):
        good += 1
    else:
        bad += 1
        if bad < 6:
            print("istar differs", n, f, i, bin(mine), r['image'])
for r in csv.DictReader(open(f"{cd}/{name}_pstar.csv")):
    n, f, i = int(r['stem']), int(r['Adams_filtration']), int(r['index'])
    if n >= max_n or f > 30:
        continue
    v = sl.basis[(n, f)][i]
    mine = sum(1 << g for c, g in v if c == top)
    if mine == vec(r['image']):
        good += 1
    else:
        bad += 1
        if bad < 6:
            print("pstar differs", n, f, i, bin(mine), r['image'])
print(f"{name}: {good} agree, {bad} differ")
