"""Shortest certificate within a pool written by `derivations --dump-pool`, by MaxSAT.

A certificate is a set of Leibniz equations whose sum (over F2) is the target relation. Selector
y_e marks equation e as used; for every variable, the parity of the selected equations containing
it must equal its coefficient in the relation (hard XOR constraints, Tseitin-encoded); each y_e is
a soft clause of weight 1, so the optimum is a certificate with the fewest equations. With
`--weights`, each equation instead costs its complexity (see `weight`), so the optimum is the
certificate that is easiest to check by hand rather than the shortest.

Usage: python3 min_certificate.py <pool file> [--weights]
"""

import re
import sys


def weight(origin, vs):
    """How hard an equation is to check: extra terms, a multiplier of high Adams filtration (not
    high stem: h_i is cheap for every i), factors that are not the first class of their bidegree,
    and module or naturality equations rather than plain sphere Leibniz."""
    w = 1 + 2 * max(len(vs) - 2, 0)
    factors = [tuple(map(int, f)) for f in re.findall(r"x_\((\d+),(\d+),(\d+)\)", origin)]
    if "*" in origin and len(factors) >= 2:
        w += min(s for _, s, _ in factors[:2]) - 1
        w += sum(1 for _, _, i in factors[:2] if i > 0)
    if "module" in origin:
        w += 2
    elif "naturality" in origin:
        w += 1
    return w
import time
from collections import defaultdict

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def load(path):
    rel, names, eqs = [], {}, []
    for line in open(path):
        tag, rest = line.split(" ", 1)
        if tag == "relation":
            rel = [int(x) for x in rest.split()]
        elif tag == "name":
            v, n = rest.split(" ", 1)
            names[int(v)] = n.strip()
        elif tag == "eq":
            head, vs = rest.split(" : ")
            eid, origin = head.split(" ", 1)
            eqs.append((int(eid), origin, [int(x) for x in vs.split()]))
    return rel, names, eqs


def main():
    import functools, builtins
    globals()["print"] = functools.partial(builtins.print, flush=True)
    rel, names, eqs = load(sys.argv[1])
    t = time.time()
    occ = defaultdict(list)
    for k, (_, _, vs) in enumerate(eqs):
        for v in vs:
            occ[v].append(k + 1)  # selector literal of equation k
    target = set(rel)
    assert target <= set(occ), "relation mentions a variable outside the pool"
    wcnf = WCNF()
    top = len(eqs)

    def xor_to(lits, parity):
        # chain z_i = z_{i-1} xor l_i, then fix the last one
        nonlocal top
        acc = lits[0]
        for l in lits[1:]:
            top += 1
            z = top
            wcnf.extend([[-z, acc, l], [-z, -acc, -l], [z, -acc, l], [z, acc, -l]])
            acc = z
        wcnf.append([acc] if parity else [-acc])

    for v, lits in occ.items():
        xor_to(lits, v in target)
    weighted = "--weights" in sys.argv
    for k, (_, origin, vs) in enumerate(eqs):
        wcnf.append([-(k + 1)], weight=weight(origin, vs) if weighted else 1)
    print(f"pool: {len(eqs)} equations, {len(occ)} variables, {len(wcnf.hard)} hard clauses")
    with RC2(wcnf, solver="cd19") as rc2:
        model = rc2.compute()
    assert model is not None, "relation not implied by the pool"
    chosen = [eqs[k] for k in range(len(eqs)) if model[k] > 0]
    # check: the chosen equations sum to the relation
    par = defaultdict(int)
    for _, _, vs in chosen:
        for v in vs:
            par[v] ^= 1
    assert {v for v, p in par.items() if p} == target
    cost = sum(weight(o, vs) for _, o, vs in chosen)
    print(
        f"minimum certificate: {len(chosen)} equations, total weight {cost} "
        f"({time.time() - t:.1f}s)"
    )
    print("relation: " + " + ".join(names.get(v, str(v)) for v in rel) + " = 0")
    for eid, origin, vs in chosen:
        print(f"  {origin}: " + " + ".join(names.get(v, str(v)) for v in vs) + " = 0")


if __name__ == "__main__":
    main()
