#!/usr/bin/env python3
"""Checks `restrict_unit cofibers` output against the ring export it was made over.

For C = S ∪_{h_j} e^{k+1} the long exact sequence

    ... -> Ext^{s-1}_{n-k+1}(S) --h_j--> Ext^{s}_{n}(S) --i_*--> Ext^{s}_{n}(C) --p_*--> Ext^{s}_{n-k}(S)
        --h_j--> Ext^{s+1}_{n-1}(S) -> ...

is checked bidegree by bidegree using only the ring export's h_j products:
  dim Ext(C) = dim coker h_j + dim ker h_j,  ker i_* = im h_j,  im i_* = ker p_*,  im p_* = ker h_j.
The kernel of h_j at filtration max_s needs filtration max_s + 1, beyond the ring export, so the
conditions involving ker h_j are checked for s < max_s only.

usage: restrict_unit_validate_cofibers.py <ring dir> <module dir> [name:k ...]
"""

import sys
from collections import defaultdict


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


def rank_file(path):
    out = {}
    for l in rows(path):
        n, s, r = map(int, l.split(","))
        out[(n, s)] = r
    return out


def rank(vecs):
    basis = []  # (pivot, vec)
    for v in vecs:
        for p, b in basis:
            if v >> p & 1:
                v ^= b
        if v:
            p = (v & -v).bit_length() - 1
            basis = [(q, b ^ v if b >> p & 1 else b) for q, b in basis]
            basis.append((p, v))
    return len(basis)


def span_contains(vecs, v):
    return rank(list(vecs) + [v]) == rank(vecs)


def kernel(images, dim):
    """Basis of the kernel of e_i -> images[i] (i < dim)."""
    done = []
    ker = []
    for i in range(dim):
        f, src = images.get(i, 0), 1 << i
        for p, g, s in done:
            if f >> p & 1:
                f ^= g
                src ^= s
        if f == 0:
            ker.append(src)
        else:
            done.append(((f & -f).bit_length() - 1, f, src))
    return ker


def main():
    ring_dir, mod_dir = sys.argv[1], sys.argv[2]
    specs = sys.argv[3:] or ["C2:1", "Ceta:2", "Cnu:4"]
    S = rank_file(f"{ring_dir}/rank.csv")
    max_n = max(n for n, _ in S)
    max_s = max(s for _, s in S)
    inds = {}
    for l in rows(f"{ring_dir}/indecomposables.csv"):
        n, s, i, name = l.split(",")
        inds[name] = (int(n), int(s), int(i))
    # h_j products: (j, n, s) -> {index: product}
    hprod = defaultdict(dict)
    hgen = {inds["h0"]: 0, inds["h1"]: 1, inds["h2"]: 2}
    for l in rows(f"{ring_dir}/generator_products.csv"):
        v = l.split(",", 6)
        g = (int(v[0]), int(v[1]), int(v[2]))
        if g in hgen:
            hprod[(hgen[g], int(v[3]), int(v[4]))][int(v[5])] = parse_vec(v[6])
    dS = lambda n, s: S.get((n, s), 0)

    failures = 0
    for spec in specs:
        name, k = spec.split(":")
        k = int(k)
        j = k.bit_length() - 1
        C = rank_file(f"{mod_dir}/{name}_rank.csv")
        # the module's exported range (it may stop below the ring export's)
        max_n = min(max(n for n, _ in S), max(n for n, _ in C))
        max_s = min(max(s for _, s in S), max(s for _, s in C))
        istar = defaultdict(dict)
        for l in rows(f"{mod_dir}/{name}_istar.csv"):
            n, s, i, img = l.split(",", 3)
            istar[(int(n), int(s))][int(i)] = parse_vec(img)
        pstar = defaultdict(dict)
        for l in rows(f"{mod_dir}/{name}_pstar.csv"):
            n, s, i, img = l.split(",", 3)
            pstar[(int(n), int(s))][int(i)] = parse_vec(img)
        checked = bad = 0

        def fail(msg):
            nonlocal bad
            bad += 1
            if bad <= 10:
                print(f"  {name}: {msg}")

        for n in range(max_n + 1):
            for s in range(max_s + 1):
                dc = C.get((n, s), 0)
                # im h_j into (n, s)
                src = (n - k + 1, s - 1)
                im_h = [hprod[(j,) + src].get(i, 0) for i in range(dS(*src))]
                coker = dS(n, s) - rank(im_h)
                # ker h_j out of (n - k, s)
                top = (n - k, s)
                ker_h = kernel(hprod[(j,) + top], dS(*top))
                if s < max_s:
                    checked += 1
                    if dc != coker + len(ker_h):
                        fail(f"({n},{s}) dim {dc} != coker {coker} + ker {len(ker_h)}")
                elif dc < coker:
                    fail(f"({n},{s}) dim {dc} < coker {coker}")
                # i_*: kernel = im h_j
                ii = istar[(n, s)]
                ker_i = kernel(ii, dS(n, s))
                if rank(ker_i) != len(ker_i) or rank(ker_i + im_h) != rank(im_h) or len(
                    ker_i
                ) != rank(im_h):
                    fail(f"({n},{s}) ker i_* != im h_j")
                # p_* i_* = 0, im i_* = ker p_*
                pp = pstar[(n, s)]
                for g, v in ii.items():
                    w = 0
                    for x in range(dc):
                        if v >> x & 1:
                            w ^= pp.get(x, 0)
                    if w:
                        fail(f"({n},{s}) p_* i_* (gen {g}) != 0")
                ker_p = kernel(pp, dc)
                im_i = list(ii.values())
                if rank(im_i) != len(ker_p) or rank(im_i + ker_p) != len(ker_p):
                    fail(f"({n},{s}) im i_* != ker p_*")
                # im p_* = ker h_j
                im_p = list(pp.values())
                if s < max_s:
                    if rank(im_p) != len(ker_h) or rank(im_p + ker_h) != len(ker_h):
                        fail(f"({n},{s}) im p_* != ker h_j")
        print(f"{name}: LES checked at {checked} bidegrees (s < {max_s}) + all i_*/p_* checks; {bad} failures")
        failures += bad
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
