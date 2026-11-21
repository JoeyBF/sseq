"""Turn the d_r rows of a tmf differentials table into a statement file over a tmf ring export.

Usage: tmf_statements.py <ring dir> <table csv> <r> <max stem> <max filtration> > out.csv

The output (`stem,Adams_filtration,source,target,label`) is read by `derivations --check
r:<node>:<file>`, which scores each row against the joint solution on E_r. Names in the table are
monomials (with ^ powers) in the ring's indecomposables, possibly sums with " + ". Rows that do not
parse, or whose source is 0 in this basis, are reported on stderr and left out.
"""
import csv
import sys

ring, table, r_want = sys.argv[1], sys.argv[2], int(sys.argv[3])
max_n, max_f = int(sys.argv[4]), int(sys.argv[5])


def vec(s):
    s = s.strip().strip('"').strip("[]")
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def mask(v):
    return sum(b << i for i, b in enumerate(v))


rank = {}
for row in csv.DictReader(open(f"{ring}/rank.csv")):
    rank[(int(row["stem"]), int(row["Adams_filtration"]))] = int(row[list(row.keys())[2]])
gens = {}
for row in csv.DictReader(open(f"{ring}/indecomposables.csv")):
    gens[row["name"]] = (int(row["stem"]), int(row["Adams_filtration"]), int(row["index"]))
prod = {}
for line in open(f"{ring}/generator_products.csv").readlines()[1:]:
    p = line.rstrip("\n").split(",", 6)
    prod[tuple(int(x) for x in p[:6])] = mask(vec(p[6]))


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


def show(m, d):
    return '"[' + ",".join(str(m >> k & 1) for k in range(d)) + ']"'


print("stem,Adams_filtration,source,target,label")
for row in csv.DictReader(open(table)):
    if int(row["r"]) != r_want:
        continue
    sn, sf = int(row["source_stem"]), int(row["source_filtration"])
    if sn > max_n or sf + r_want > max_f:
        continue
    tname = row["target_name"].strip()
    try:
        (n, f), src = evaluate(row["source_name"])
        tgt = evaluate(tname)[1] if tname not in ("", "0") else 0
    except (KeyError, ValueError) as e:
        print("skip", row["source_name"], "->", tname, e, file=sys.stderr)
        continue
    if src == 0:
        print("skip (source is 0 in this basis)", row["source_name"], file=sys.stderr)
        continue
    label = f"{row['source_name']} -> {tname}".replace(",", ";")
    print(f"{n},{f},{show(src, rank[(n, f)])},{show(tgt, rank.get((n - 1, f + r_want), 0))},{label}")
