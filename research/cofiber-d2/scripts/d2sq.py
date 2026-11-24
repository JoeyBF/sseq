"""Checks d2 o d2 = 0 on a module d2 file wherever both steps are in the file.

usage: d2sq.py <d2 csv>...
"""
import csv
import sys

for path in sys.argv[1:]:
    d = {}
    for r in csv.reader(open(path)):
        if r[0] != 'stem':
            d[(int(r[0]), int(r[1]), int(r[2]))] = sum(
                int(x) << i for i, x in enumerate(r[3].strip('"[] ').split(',')) if x.strip())
    chk = bad = 0
    ex = []
    for (n, f, i), v in d.items():
        w, ok, j = 0, True, 0
        while v:
            if v & 1:
                t = d.get((n - 1, f + 2, j))
                if t is None:
                    ok = False
                    break
                w ^= t
            v >>= 1
            j += 1
        if ok:
            chk += 1
            if w:
                bad += 1
                ex.append((n, f, i))
    print(f"{path}: d2 d2 checked {chk}, nonzero {bad} {sorted(ex)[:6]}")
