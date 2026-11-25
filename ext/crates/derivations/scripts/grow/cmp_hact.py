"""Compare module_slots' h_i action with a reference indaction table (same basis).

usage: cmp_hact.py <module_slots hact csv> <reference indaction csv> <max stem> <max filtration>
"""
import sys

mine_p, ref_p, max_n, max_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])


def rows(p):
    out = {}
    for line in open(p):
        r = line.rstrip('\n').split(',', 6)
        if r[0].startswith('stem'):
            continue
        key = tuple(int(x) for x in r[:6])
        out[key] = [int(t) for t in r[6].strip('"[] ').replace(' ', '').split(',') if t != '']
    return out


mine, ref = rows(mine_p), rows(ref_p)
agree = differ = 0
for key, v in mine.items():
    an, as_, ai, n, f, i = key
    if as_ != 1 or n + an >= max_n or f + 1 > max_s or key not in ref:
        continue
    if ref[key] == v:
        agree += 1
    else:
        differ += 1
        if differ <= 5:
            print("differ", key, v, ref[key])
missing = sum(1 for k in ref if k[1] == 1 and k[0] + k[3] < max_n and k[4] + 1 <= max_s
              and any(v for v in ref[k]) and k not in mine)
print(f"{agree} agree, {differ} differ; reference nonzero rows absent from mine: {missing}")
