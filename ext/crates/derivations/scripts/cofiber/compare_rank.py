"""Basis-free check of the engine's d2 on C2 against a direct secondary computation: the rank
of d2 out of each bidegree, wherever the engine determined every class there."""
import csv
import re
import sys
from collections import defaultdict


def rank(vecs):
    rows = [int("".join(map(str, v))[::-1] or "0", 2) for v in vecs]
    r = 0
    piv = {}
    for x in rows:
        while x:
            p = x.bit_length() - 1
            if p in piv:
                x ^= piv[p]
            else:
                piv[p] = x
                r += 1
                break
    return r


direct = defaultdict(list)
for line in open(sys.argv[1]):
    m = re.match(r"d_2 x_\((\d+), (\d+), (\d+)\) = \[(.*)\]", line.strip())
    if m:
        v = [int(x) for x in m.group(4).split(",") if x.strip()]
        direct[(int(m.group(1)), int(m.group(2)))].append(v)

engine = defaultdict(list)
open_at = set()
for row in csv.DictReader(open(sys.argv[2])):
    b = (int(row["stem"]), int(row["Adams_filtration"]))
    if row["status"] != "determined":
        open_at.add(b)
        continue
    v = [int(x) for x in row["target"].strip("[]\"").split(",") if x.strip()]
    engine[b].append(v)

agree = disagree = skipped = 0
for b in sorted(set(direct) | set(engine)):
    if b in open_at:
        skipped += 1
        continue
    rd, re_ = rank(direct.get(b, [])), rank(engine.get(b, []))
    if len(direct.get(b, [])) != len(engine.get(b, [])):
        print(f"{b}: class counts differ, direct {len(direct.get(b, []))}, engine {len(engine.get(b, []))}")
    if rd == re_:
        agree += 1
    else:
        disagree += 1
        print(f"{b}: rank direct {rd}, engine {re_}")
print(f"bidegrees: {agree} agree, {disagree} disagree, {skipped} with open classes")
