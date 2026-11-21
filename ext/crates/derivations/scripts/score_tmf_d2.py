"""Score a --compare-solution file for tmf's d2 against Bruner–Rognes' tmf d2 table.

Usage: score_tmf_d2.py <ring dir> <solution csv> <BR table csv> <max stem> <max filtration>
Names in the table are monomials (with ^ powers) in the 13 generators, possibly sums with " + ".
"""
import csv
import re
import sys

ring, sol_path, table, max_n, max_f = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])


def vec(s):
    s = s.strip().strip('"').strip("[]")
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def mask(v):
    return sum(b << i for i, b in enumerate(v))


rank = {}
for r in csv.DictReader(open(f"{ring}/rank.csv")):
    rank[(int(r["stem"]), int(r["Adams_filtration"]))] = int(r[list(r.keys())[2]])
gens = {}
for r in csv.DictReader(open(f"{ring}/indecomposables.csv")):
    gens[r["name"]] = (int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))
prod = {}
for line in open(f"{ring}/generator_products.csv").readlines()[1:]:
    p = line.rstrip("\n").split(",", 6)
    key = tuple(int(x) for x in p[:6])
    prod[key] = mask(vec(p[6]))


def times_gen(g, n, f, m):
    """generator g times the vector m in bidegree (n, f)"""
    gn, gf, gi = gens[g]
    out = 0
    for k in range(rank.get((n, f), 0)):
        if m >> k & 1:
            out ^= prod.get((gn, gf, gi, n, f, k), 0)
    return gn + n, gf + f, out


def monomial(name):
    n, f, m = 0, 0, 1  # the unit
    for tok in name.split():
        g, _, e = tok.partition("^")
        for _ in range(int(e) if e else 1):
            if g not in gens:
                raise KeyError(g)
            n, f, m = times_gen(g, n, f, m)
    return n, f, m


def evaluate(expr):
    total, deg = 0, None
    for term in expr.split(" + "):
        n, f, m = monomial(term.strip())
        deg = deg or (n, f)
        if (n, f) != deg:
            raise ValueError("inhomogeneous")
        total ^= m
    return deg, total


sol = {}
for r in csv.DictReader(open(sol_path)):
    sol[(int(r["stem"]), int(r["Adams_filtration"]), int(r["index"]))] = (r["status"], mask(vec(r["target"])))

counts = {"correct": 0, "WRONG": 0, "open": 0, "skipped": 0}
for r in csv.DictReader(open(table)):
    if r["r"] != "2" or r["generated"]:
        continue
    sn, sf = int(r["source_stem"]), int(r["source_filtration"])
    if sn > max_n or sf + 2 > max_f:
        continue
    try:
        (n, f), src = evaluate(r["source_name"])
        (tn, tf), tgt = evaluate(r["target_name"]) if r["target_name"].strip() not in ("", "0") else ((sn - 1, sf + 2), 0)
    except (KeyError, ValueError) as e:
        counts["skipped"] += 1
        print("skip", r["source_name"], "->", r["target_name"], e)
        continue
    if src == 0:
        counts["skipped"] += 1
        print("skip (source is 0 in our basis)", r["source_name"])
        continue
    val, det = 0, True
    for k in range(rank.get((n, f), 0)):
        if src >> k & 1:
            st, t = sol.get((n, f, k), ("open", 0))
            det &= st == "determined"
            val ^= t
    if not det:
        counts["open"] += 1
    elif val == tgt:
        counts["correct"] += 1
    else:
        counts["WRONG"] += 1
        print("WRONG", r["source_name"], "->", r["target_name"], "ours", bin(val), "theirs", bin(tgt))
print(counts)
