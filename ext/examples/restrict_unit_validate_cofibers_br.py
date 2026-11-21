#!/usr/bin/env python3
"""Checks `restrict_unit cofibers` output against Bruner–Rognes, "The Adams spectral sequence for
topological modular forms", Tables 1.5–1.7 and Lemma 1.36.

Tables 1.5–1.7 list the Ext_{A(2)}(F_2, F_2)-module generators of Ext_{A(2)}(M_k, F_2) for
t - s <= 200 (k = 1, 2, 4). From the indecomposables' action we compute, per bidegree, the
dimension of Ext(C) modulo the span of (indecomposable) * (class), which is the number of module
generators there, and compare with the tables over the whole exported range.

usage: restrict_unit_validate_cofibers_br.py <ring dir> <module dir>
"""

import sys
from collections import defaultdict

# (stem, s) of the module generators, Bruner-Rognes Tables 1.5, 1.6, 1.7
BR = {
    "C2": [(0, 0), (2, 1), (7, 2), (9, 3), (18, 6), (26, 5), (31, 6), (32, 8), (33, 7), (36, 7),
           (42, 10)],
    "Ceta": [(0, 0), (2, 1), (5, 1), (11, 4), (14, 3), (17, 3), (36, 8)],
    "Cnu": [(0, 0), (4, 3), (5, 1), (7, 2), (10, 2), (12, 3), (16, 5), (24, 4), (28, 7), (29, 5),
            (31, 6), (34, 6), (36, 7), (40, 9)],
}


def parse_vec(s):
    s = s.strip().strip('"').strip("[]")
    return sum(1 << i for i, e in enumerate(s.split(",")) if e.strip() == "1")


def rows(path):
    with open(path) as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def rank(vecs):
    basis = []
    for v in vecs:
        for p, b in basis:
            if v >> p & 1:
                v ^= b
        if v:
            p = (v & -v).bit_length() - 1
            basis = [(q, b ^ v if b >> p & 1 else b) for q, b in basis]
            basis.append((p, v))
    return len(basis)


def main():
    ring_dir, mod_dir = sys.argv[1], sys.argv[2]
    names = {}
    for l in rows(f"{ring_dir}/indecomposables.csv"):
        n, s, i, name = l.split(",")
        names[(int(n), int(s), int(i))] = name
    bad = 0
    for name, br in BR.items():
        C = {}
        for l in rows(f"{mod_dir}/{name}_rank.csv"):
            n, s, r = map(int, l.split(","))
            C[(n, s)] = r
        dec = defaultdict(list)
        act = {}
        for l in rows(f"{mod_dir}/{name}_indaction.csv"):
            v = l.split(",", 6)
            a = tuple(map(int, v[:3]))
            n, s, x = int(v[3]), int(v[4]), int(v[5])
            p = parse_vec(v[6])
            dec[(n + a[0], s + a[1])].append(p)
            act[(names[a], n, s, x)] = p
        gens = {}
        for b, d in C.items():
            g = d - rank(dec[b])
            if g:
                gens[b] = g
        max_n = max(n for n, _ in C)
        ours = sorted(b for b, g in gens.items() for _ in range(g))
        theirs = sorted(br)
        ok = ours == theirs
        bad += not ok
        print(f"{name}: module generators through stem {max_n}: {ours}")
        print(f"{name}: {'MATCHES' if ok else 'DIFFERS FROM'} Bruner-Rognes {theirs}")
        if name == "C2":
            # Lemma 1.36: h0 i(1) = 0 but h0 * (the class at (2,1)) != 0
            print("C2: h0 i(1) =", act.get(("h0", 0, 0, 0), 0), "; h0 * h1bar =", act.get(("h0", 2, 1, 0), 0))
            bad += act.get(("h0", 0, 0, 0), 0) != 0 or act.get(("h0", 2, 1, 0), 0) == 0
        if name == "Ceta":
            print("Ceta: h1 i(1) =", act.get(("h1", 0, 0, 0), 0), "; h1 * h0bar =", act.get(("h1", 2, 1, 0), 0))
            bad += act.get(("h1", 0, 0, 0), 0) != 0 or act.get(("h1", 2, 1, 0), 0) == 0
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
