"""Coverage check: the (0,0) block of module_d2's output is the sphere's d2 on generators; compare it
with a sphere d2 file (`stem,Adams_filtration,index,target`) through a stem bound.

usage: cover.py <blocks csv> <sphere d2 csv> <max stem> <max f>
"""
import csv
import sys

blocks_path, d2_path, max_n, max_f = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])


def mask(s):
    s = s.strip().strip('"').strip('[]')
    return sum(int(t) << i for i, t in enumerate(s.split(','))) if s.strip() else 0


blk = {}
for r in csv.reader(open(blocks_path)):
    if r[0] == 'module' or r[4] != '0' or r[6] != '0':
        continue
    blk[(int(r[1]), int(r[2]), int(r[3]))] = mask(r[8])
ok = bad = 0
for r in csv.reader(open(d2_path)):
    if r[0] == 'stem':
        continue
    n, f, i = int(r[0]), int(r[1]), int(r[2])
    if n > max_n or f > max_f:
        continue
    want = mask(r[3])
    got = blk.get((n, f, i), 0)
    if want == got:
        ok += 1
    else:
        bad += 1
        if bad <= 10:
            print(f"mismatch ({n},{f}) #{i}: file {want:b} blocks {got:b}")
print(f"{ok} agree, {bad} differ")
