"""Exact d2 of a cofiber C(h_j) where the easy way reaches, from `cofiber_easy_cycles` output
(cochain-exact L over a basis of all d2-cycles):

    d2(e_c . bottom) = i_*(d2_S e_c)              (cokernel classes)
    d2(z . top)      = i_*(L(z))                  (kernel classes with z a d2-cycle)

Kernel classes over non-cycles are left out (the engine solves them).

usage: assemble_easy.py <easy out> <j> <module dir> <name> <sphere d2 csv> <max stem> <out csv>
"""
import csv
import re
import sys
from collections import defaultdict

easy_path, j, mdir, name, sd2_path, max_n, out_path = sys.argv[1:8]
# optional: <cone csv> <modb out>: non-cycle lifts from h_j's side, valid mod i_*(B2)
cone_path, modb_path = (sys.argv[8], sys.argv[9]) if len(sys.argv) > 9 else (None, None)
j, max_n = int(j), int(max_n)
shift = 1 << j


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


# cycle basis per sphere bidegree, with E and L for h_j
cyc = defaultdict(list)  # (n, s) -> [(y mask, E mask, L mask)]
pat = re.compile(r"^\[(.*)\] \[h(\d+)\] = \[(.*)\] \+ λ \[(.*)\]$")
for line in open(easy_path):
    m = pat.match(line.strip())
    if not m or int(m.group(2)) != j:
        continue
    y = 0
    n = s = None
    for term in m.group(1).split(" + "):
        a, b, c = (int(x) for x in term.strip()[3:-1].split(","))
        n, s = a, b
        y ^= 1 << c
    cyc[(n, s)].append((y, mask(vec(m.group(3))), mask(vec(m.group(4)))))

d2s = {}
for row in csv.DictReader(open(sd2_path)):
    d2s[(int(row["stem"]), int(row["Adams_filtration"]), int(row["index"]))] = mask(vec(row["target"]))
rank = {}
for row in csv.DictReader(open(f"{mdir}/{name}_rank.csv")):
    rank[(int(row["stem"]), int(row["Adams_filtration"]))] = int(row["rank"])
istar, pstar = {}, {}
for row in csv.DictReader(open(f"{mdir}/{name}_istar.csv")):
    istar[(int(row["stem"]), int(row["Adams_filtration"]), int(row["index"]))] = mask(vec(row["image"]))
for row in csv.DictReader(open(f"{mdir}/{name}_pstar.csv")):
    pstar[(int(row["stem"]), int(row["Adams_filtration"]), int(row["index"]))] = mask(vec(row["image"]))


def i_star(n, s, v):
    out = 0
    for l in bits(v):
        out ^= istar.get((n, s, l), 0)
    return out


def express(n, s, z):
    """L and E of the cochain z at sphere (n, s) as a combination of the cycle basis, or None."""
    piv = {}
    for y, e, l in cyc.get((n, s), []):
        acc = (y, e, l)
        while acc[0]:
            p = acc[0].bit_length() - 1
            if p in piv:
                q = piv[p]
                acc = (acc[0] ^ q[0], acc[1] ^ q[1], acc[2] ^ q[2])
            else:
                piv[p] = acc
                break
    e = l = 0
    while z:
        p = z.bit_length() - 1
        if p not in piv:
            return None
        q = piv[p]
        z ^= q[0]
        e ^= q[1]
        l ^= q[2]
    return e, l


Lh = {}
if cone_path:
    for row in csv.DictReader(open(cone_path)):
        if int(row["j"]) == j:
            Lh[(int(row["stem"]), int(row["Adams_filtration"]), int(row["index"]))] = mask(vec(row["L"]))


def solve_top(n, s, z):
    rows = [(k, pstar[(n, s, k)]) for k in range(rank.get((n, s), 0)) if pstar.get((n, s, k))]
    piv = {}
    for k, v in rows:
        c = 1 << k
        while v:
            p = v.bit_length() - 1
            if p in piv:
                v ^= piv[p][0]
                c ^= piv[p][1]
            else:
                piv[p] = (v, c)
                break
    coef = 0
    while z:
        p = z.bit_length() - 1
        if p not in piv:
            return None
        z ^= piv[p][0]
        coef ^= piv[p][1]
    return coef


modb = []
coker_gen = {}
for (n, s, i), img in istar.items():
    if img and img & (img - 1) == 0:
        coker_gen.setdefault((n, s, img.bit_length() - 1), i)

lines, kinds = [], defaultdict(int)
for (n, s), r in sorted(rank.items()):
    if n > max_n or (n - 1, s + 2) not in rank:
        continue
    dim = rank[(n - 1, s + 2)]
    for k in range(r):
        if pstar.get((n, s, k)):
            z = pstar[(n, s, k)]
            el = express(n - shift, s, z)
            if el is None:
                bitsz = [l for l in bits(z)]
                d2z = 0
                for l in bitsz:
                    d2z ^= d2s.get((n - shift, s, l), 0)
                top = solve_top(n - 1, s + 2, d2z)
                if not cone_path or top is None or any((n - shift, s, l) not in Lh for l in bitsz):
                    kinds["top cell over a non-cycle (left open)"] += 1
                    continue
                lh = 0
                for l in bitsz:
                    lh ^= Lh[(n - shift, s, l)]
                tgt = top ^ i_star(n - 1, s + 2, lh)
                src = ",".join("1" if q == k else "0" for q in range(r))
                modb.append(f'{n},{s},"[{src}]","[{",".join(str(tgt >> l & 1) for l in range(dim))}]",nc')
                kinds["top cell over a non-cycle (mod i_*(B2), h_j side)"] += 1
                continue
            e, l = el
            assert e == 0, f"h_j z != 0 for kernel class {(n, s, k)}"
            tgt = i_star(n - 1, s + 2, l)
            kinds["top cell over a cycle"] += 1
        elif (n, s, k) in coker_gen:
            tgt = i_star(n - 1, s + 2, d2s.get((n, s, coker_gen[(n, s, k)]), 0))
            kinds["bottom cell"] += 1
        else:
            kinds["unidentified"] += 1
            continue
        lines.append(f'{n},{s},{k},"[{",".join(str(tgt >> l & 1) for l in range(dim))}]"')
with open(out_path, "w") as f:
    f.write("stem,Adams_filtration,index,target\n")
    f.write("\n".join(lines) + "\n")
if modb_path:
    with open(modb_path, "w") as f:
        f.write("stem,Adams_filtration,source,target,label\n")
        f.write("\n".join(modb) + "\n")
for k, v in kinds.items():
    print(f"{v:6d} {k}")
