#!/usr/bin/env python3
"""Checks that a larger ring export agrees with a smaller one on the smaller one's range.

Ranks, indecomposables and generator products (with both factors and the product in the smaller
range) must be identical, as they are when both come from the same save.

usage: restrict_unit_compare_rings.py <small ring dir> <large ring dir>
"""

import sys


def rows(path):
    with open(path) as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def load(d):
    rank = {}
    for l in rows(f"{d}/rank.csv"):
        n, s, r = map(int, l.split(","))
        rank[(n, s)] = r
    inds = {}
    for l in rows(f"{d}/indecomposables.csv"):
        n, s, i, name = l.split(",")
        inds[(int(n), int(s), int(i))] = name
    prods = {}
    for l in rows(f"{d}/generator_products.csv"):
        v = l.split(",", 6)
        prods[tuple(map(int, v[:6]))] = v[6]
    return rank, inds, prods


def main():
    (r1, i1, p1), (r2, i2, p2) = load(sys.argv[1]), load(sys.argv[2])
    max_n = max(n for n, _ in r1)
    max_s = max(s for _, s in r1)
    inr = lambda n, s: n <= max_n and s <= max_s
    bad = 0
    r2s = {b: r for b, r in r2.items() if inr(*b)}
    if r1 != r2s:
        bad += 1
        print("ranks differ:", sorted(set(r1.items()) ^ set(r2s.items()))[:10])
    i2s = {g: n for g, n in i2.items() if inr(g[0], g[1])}
    if i1 != i2s:
        bad += 1
        print("indecomposables differ:", i1, i2s)
    p2s = {k: v for k, v in p2.items() if inr(k[0] + k[3], k[1] + k[4])}
    if p1 != p2s:
        bad += 1
        diff = sorted(set(p1.items()) ^ set(p2s.items()))
        print(f"{len(diff)} product rows differ, e.g.", diff[:5])
    print(
        f"stems <= {max_n}, s <= {max_s}: {len(r1)} ranks, {len(i1)} indecomposables, "
        f"{len(p1)} products; {'IDENTICAL' if not bad else 'DIFFERENT'}"
    )
    print(f"large export: {len(r2)} ranks, {len(i2)} indecomposables {sorted(i2.values())}, "
          f"{len(p2)} products, max rank {max(r2.values())}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
