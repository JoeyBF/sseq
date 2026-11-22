"""The exact cofiber d2 from the ground truth (cofiber_truth), in the engine's d2-file format:

    d2(e_c . bottom) = i_*(d2_S e_c)
    d2(z . top)      = [d2_S z] . top + i_*(K(z))      (K from the truth run)

usage: assemble_truth.py <truth csv> <j> <module dir> <name> <sphere d2 csv> <max stem> <out>
"""
import csv
import sys

truth_p, j, mdir, name, sd2_p, max_n, out_p = sys.argv[1:8]
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


K = {}
for r in csv.DictReader(open(truth_p)):
    if int(r["j"]) == j:
        K[(int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = mask(vec(r["bottom"]))
D2 = {}
for r in csv.DictReader(open(sd2_p)):
    D2[(int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = mask(vec(r["target"]))
rank, istar, pstar = {}, {}, {}
for r in csv.DictReader(open(f"{mdir}/{name}_rank.csv")):
    rank[(int(r["stem"]), int(r["Adams_filtration"]))] = int(r["rank"])
for r in csv.DictReader(open(f"{mdir}/{name}_istar.csv")):
    istar[(int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = mask(vec(r["image"]))
for r in csv.DictReader(open(f"{mdir}/{name}_pstar.csv")):
    pstar[(int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = mask(vec(r["image"]))


def i_star(n, s, v):
    out = 0
    for l in bits(v):
        out ^= istar.get((n, s, l), 0)
    return out


def solve_top(n, s, z):
    piv = {}
    for k in range(rank.get((n, s), 0)):
        v, c = pstar.get((n, s, k), 0), 1 << k
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


coker_gen = {}
for (n, s, i), img in istar.items():
    if img and img & (img - 1) == 0:
        coker_gen.setdefault((n, s, img.bit_length() - 1), i)

lines, missing = [], 0
for (n, s), r in sorted(rank.items()):
    if n > max_n or (n - 1, s + 2) not in rank:
        continue
    dim = rank[(n - 1, s + 2)]
    for k in range(r):
        if pstar.get((n, s, k)):
            z = pstar[(n, s, k)]
            if any((n - shift, s, l) not in K for l in bits(z)):
                missing += 1
                continue
            d2z = 0
            kz = 0
            for l in bits(z):
                d2z ^= D2.get((n - shift, s, l), 0)
                kz ^= K[(n - shift, s, l)]
            top = solve_top(n - 1, s + 2, d2z)
            if top is None:
                missing += 1
                continue
            tgt = top ^ i_star(n - 1, s + 2, kz)
        elif (n, s, k) in coker_gen:
            tgt = i_star(n - 1, s + 2, D2.get((n, s, coker_gen[(n, s, k)]), 0))
        else:
            missing += 1
            continue
        lines.append(f'{n},{s},{k},"[{",".join(str(tgt >> l & 1) for l in range(dim))}]"')
with open(out_p, "w") as f:
    f.write("stem,Adams_filtration,index,target\n" + "\n".join(lines) + "\n")
print(f"{name}: {len(lines)} exact d2 values, {missing} not covered")
