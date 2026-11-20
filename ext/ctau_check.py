"""Checks an export of `examples/ctau_e2.rs` against Isaksen's tables and against classical Ext.

    python3 -I ctau_check.py <export directory> <directory of the transcribed tables>

The checks, each on the part of the export's range that decides it:

* every generator of the Cτ E2 page that "Stable stems" lists has a nonzero group in its tridegree;
* every h0, h1, h2 extension it lists (invisible in the E2 page of the sphere) is a nonzero product
  into the listed tridegree;
* for each classical internal degree t, the Cτ E2 page summed over the weights is the E1 page of a
  spectral sequence (τ-Bockstein with τ inverted, the Cartan-Eilenberg spectral sequence of
  E -> A -> P) converging to classical Ext, with differentials raising the filtration by one. So its
  Euler characteristic in s equals that of Ext_A^{s,t}, and each of its ranks is at least that of
  Ext_A^{s,t};
* the products by h_1, h_2, h_3 read off the differential, and those by the classes with a saved
  chain map, agree with the products by the same classes through their bar cocycles;
* the bar cocycle products are associative;
* each two-cell complex sits in the long exact sequence of its attaching map: its ranks, and the
  ranks of the maps of its cells, are those of the cokernel and the kernel of h_j.
"""

import csv
import sys
from collections import defaultdict

out, tables = sys.argv[1], sys.argv[2]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


rank = {}
for r in rows(f"{out}/Ctau_rank.csv"):
    rank[(int(r["stem"]), int(r["filtration"]), int(r["weight"]))] = int(r["rank"])
max_n = max(n for n, _, _ in rank)
max_s = max(s for _, s, _ in rank)

# h[j][(n, s, w)] = number of basis classes with a nonzero product by h_j
h = defaultdict(lambda: defaultdict(int))
for r in rows(f"{out}/Ctau_h.csv"):
    h[int(r["j"])][(int(r["stem"]), int(r["filtration"]), int(r["weight"]))] += 1

failures = 0


def in_range(n, s):
    return 0 <= n <= max_n and 0 <= s <= max_s


# named[name]: the tridegrees the extension tables give a source class
named = defaultdict(set)
for j in range(3):
    for r in rows(f"{tables}/ss_Ctau_h{j}.csv"):
        if r["source_stem"]:
            named[r["source_name"]].add(
                (int(r["source_stem"]), int(r["source_filtration"]), int(r["source_weight"]))
            )

print("== generators of the Cτ E2 page (ss_Ctau_E2.csv)")
seen = set()
checked = 0
for r in rows(f"{tables}/ss_Ctau_E2.csv"):
    x = (int(r["source_stem"]), int(r["source_filtration"]), int(r["source_weight"]))
    if x in seen or not in_range(x[0], x[1]):
        continue
    seen.add(x)
    checked += 1
    if rank.get(x, 0) == 0:
        # the extension tables give the tridegree of the same class independently
        elsewhere = {y for y in named[r["source_name"]] if y != x and rank.get(y, 0) > 0}
        if elsewhere:
            print(f"  TABLES DISAGREE on {r['source_name']}: {x} here, {sorted(elsewhere)} in the"
                  " extension tables, where the group is nonzero")
        else:
            failures += 1
            print(f"  MISSING {r['source_name']} at {x}")
print(f"  {checked} tridegrees in range, all others nonzero")

print("== h0, h1, h2 extensions (ss_Ctau_h{0,1,2}.csv)")
step = {0: (0, 1, 0), 1: (1, 1, 1), 2: (3, 1, 2)}
checked = 0
for j in range(3):
    for r in rows(f"{tables}/ss_Ctau_h{j}.csv"):
        if not r["target_stem"] or not r["source_stem"]:
            continue
        x = (int(r["source_stem"]), int(r["source_filtration"]), int(r["source_weight"]))
        y = (int(r["target_stem"]), int(r["target_filtration"]), int(r["target_weight"]))
        if not (in_range(*x[:2]) and in_range(*y[:2])):
            continue
        checked += 1
        expect = tuple(a + b for a, b in zip(x, step[j]))
        if y != expect:
            failures += 1
            print(f"  DEGREE h{j} * {r['source_name']} {x} -> {r['target_name']} {y}")
        elif h[j].get(x, 0) == 0:
            failures += 1
            print(f"  ZERO h{j} * {r['source_name']} {x} -> {r['target_name']} {y}")
print(f"  {checked} extensions in range, all others nonzero")

print("== Euler characteristic against classical Ext")
classical = defaultdict(int)
have_t = set()
for r in rows(f"{out}/classical_rank.csv"):
    classical[(int(r["filtration"]), int(r["degree"]))] = int(r["rank"])
total = defaultdict(int)
for (n, s, w), d in rank.items():
    total[(s, n + s)] += d
checked = 0
# degree t is complete in the export when every (stem t - s, s) is in range
for t in range(0, min(max_n, max_s) + 1):
    chi = sum((-1) ** s * total[(s, t)] for s in range(t + 1))
    chi_cl = sum((-1) ** s * classical[(s, t)] for s in range(t + 1))
    checked += 1
    if chi != chi_cl:
        failures += 1
        print(f"  EULER t = {t}: Cτ {chi}, classical {chi_cl}")
    for s in range(t + 1):
        if total[(s, t)] < classical[(s, t)]:
            failures += 1
            print(f"  RANK (s, t) = ({s}, {t}): Cτ {total[(s, t)]} < classical {classical[(s, t)]}")
print(f"  {checked} internal degrees complete in range, all others agree")



def vec(cell):
    """A coordinate vector as a bitmask."""
    v = 0
    for i, c in enumerate(cell.strip("[]").split(",")):
        if c.strip() == "1":
            v |= 1 << i
    return v


def rank_of(vectors):
    """The rank of a list of bitmask vectors over F_2."""
    basis = {}
    for v in vectors:
        while v:
            top = v.bit_length() - 1
            if top not in basis:
                basis[top] = v
                break
            v ^= basis[top]
    return len(basis)


def tri(r, suffix=""):
    return (int(r["stem" + suffix]), int(r["filtration" + suffix]), int(r["weight" + suffix]))


def add(x, y):
    return tuple(a + b for a, b in zip(x, y))


def sub(x, y):
    return tuple(a - b for a, b in zip(x, y))


# hmat[j][(x, i)]: h_j times basis class i at x
hmat = defaultdict(dict)
for r in rows(f"{out}/Ctau_h.csv"):
    hmat[int(r["j"])][(tri(r), int(r["index"]))] = vec(r["product"])
hdeg = {0: (0, 1, 0), 1: (1, 1, 1), 2: (3, 1, 2), 3: (7, 1, 4)}

# prod[(b, i)][(x, j)]: basis class i at b times basis class j at x
prod = defaultdict(dict)
try:
    for r in rows(f"{out}/Ctau_products.csv"):
        prod[(tri(r, "_1"), int(r["index_1"]))][(tri(r, "_2"), int(r["index_2"]))] = vec(r["product"])
except FileNotFoundError:
    pass

if prod:
    print("== h_j by the differential against h_j by its bar cocycle")
    for j in (1, 2, 3):
        b = (hdeg[j], 0)
        if b not in prod:
            continue
        # both run over every class whose product lies in the range
        bad = sum(1 for key in set(prod[b]) | set(hmat[j]) if prod[b].get(key, 0) != hmat[j].get(key, 0))
        failures += bad
        print(f"  h{j}: {len(set(prod[b]) | set(hmat[j]))} nonzero products, {bad} differ")

    print("== saved chain maps against bar cocycles")
    lin = defaultdict(dict)
    for r in rows(f"{out}/Ctau_indaction.csv"):
        n, s, i = int(r["classical_stem"]), int(r["classical_filtration"]), int(r["classical_index"])
        lin[((2 * n + s, s, n + s), i)][(tri(r), int(r["index"]))] = vec(r["product"])
    unknown = defaultdict(set)
    for r in rows(f"{out}/Ctau_unknown.csv"):
        n, s, i = int(r["classical_stem"]), int(r["classical_filtration"]), int(r["classical_index"])
        unknown[((2 * n + s, s, n + s), i)].add(tri(r))
    for b in sorted(lin):
        if b not in prod:
            continue
        keys = [k for k in set(prod[b]) | set(lin[b]) if k[0] not in unknown[b]]
        bad = sum(1 for k in keys if prod[b].get(k, 0) != lin[b].get(k, 0))
        failures += bad
        print(f"  class {b}: {len(keys)} nonzero products with a saved step, {bad} differ")

    print("== associativity of the bar cocycle products")
    dim = rank
    checked = bad = 0
    multipliers = sorted(prod)
    for b in multipliers:
        for c in multipliers:
            bc = add(b[0], c[0])
            # (b c) is a combination of multipliers only if its tridegree is one
            if not any(m[0] == bc for m in multipliers) and dim.get(bc, 0) > 0:
                continue
            bc_vec = prod[b].get(c, 0)
            for (x, j), cx in prod[c].items():
                y = add(c[0], x)
                z = add(b[0], y)
                if dim.get(z, 0) == 0:
                    continue
                # b (c x)
                lhs = 0
                for i in range(dim.get(y, 0)):
                    if cx >> i & 1:
                        lhs ^= prod[b].get((y, i), 0)
                # (b c) x
                rhs = 0
                for i in range(dim.get(bc, 0)):
                    if bc_vec >> i & 1:
                        rhs ^= prod[(bc, i)].get((x, j), 0)
                checked += 1
                bad += lhs != rhs
    failures += bad
    print(f"  {checked} triples with (c x) nonzero, {bad} differ")

print("== long exact sequences of the two-cell complexes")
for j, name in enumerate(["C2", "Ceta", "Cnu", "Csigma"]):
    try:
        crank = {tri(r): int(r["rank"]) for r in rows(f"{out}/{name}_rank.csv")}
    except FileNotFoundError:
        continue
    istar, pstar = defaultdict(list), defaultdict(list)
    for r in rows(f"{out}/{name}_istar.csv"):
        istar[tri(r)].append(vec(r["image"]))
    for r in rows(f"{out}/{name}_pstar.csv"):
        pstar[tri(r)].append(vec(r["image"]))
    hd = hdeg[j]
    top = add(hd, (1, -1, 0))
    hrows = defaultdict(list)
    for (x, i), v in hmat[j].items():
        hrows[x].append(v)
    checked = bad = 0
    for x in set(rank) | set(crank):
        n, s, w = x
        # the map out of the top cell's class lands one filtration up
        if s >= max_s or n > max_n:
            continue
        coker = rank.get(x, 0) - rank_of(hrows[sub(x, hd)])
        below = sub(x, top)
        ker = rank.get(below, 0) - rank_of(hrows[below])
        checked += 1
        ok = (
            crank.get(x, 0) == coker + ker
            and rank_of(istar[x]) == coker
            and rank_of(pstar[x]) == ker
        )
        if not ok:
            bad += 1
            if bad <= 5:
                print(f"  {name} at {x}: rank {crank.get(x, 0)}, coker {coker} (i_* {rank_of(istar[x])}), ker {ker} (p_* {rank_of(pstar[x])})")
    failures += bad
    print(f"  {name}: {checked} tridegrees, {bad} inexact")

print("FAILURES:", failures)
sys.exit(1 if failures else 0)
