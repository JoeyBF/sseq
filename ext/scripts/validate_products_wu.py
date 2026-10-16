#!/usr/bin/env python3
"""Validate computed products (``save_products`` output) against L. Wu's Ext chart (t <= 350).

    validate_products_wu.py OURS.jsonl[.gz] [--wu wu_exta_t350_chart-data.json.gz] [--max-report N]

Wu's ``chart-data.json`` (schema ``ext_product.ext_chart.v1``):

* ``nodes`` is an additive basis of Ext: one entry per basis element, with ``id``, ``s``, ``t``.
  The basis of ``Ext^{s,t}`` is the nodes at ``(s, t)`` in ``id`` order. (``coordinates`` expresses
  a node in their resolution's generator basis and ``factors`` names it as a product of
  indecomposables; neither is needed here.)
* ``actions`` holds multiplication by ``h0 .. h8``: per action ``name`` (``hK``, bidegree
  ``(s, t) = (1, 2^K)``), ``products`` lists ``{"source": node, "targets": [node, ...]}`` meaning
  ``hK * source = sum(targets)``. Sources that are absent multiply to zero (in range) -- their
  ``zero_products`` count says so -- or leave the chart (``out_of_range_products``, target t > 350).

Our basis (dual to our minimal resolution's generators) differs from theirs by an unknown
invertible matrix in every bidegree, so matrices cannot be compared entry by entry. Everything
compared is BASIS-INVARIANT:

1. rank of each ``hK : Ext^{s,t} -> Ext^{s+1,t+2^K}``;
2. rank of every two-step composite ``hJ hK`` (a product of two of our matrices vs two of theirs);
3. per target bidegree, the dimension of the sum of the images of all compared ``hK`` (the part of
   ``Ext`` that is ``h``-decomposable);
4. per source bidegree, the dimension of the joint kernel of all compared ``hK``.

plus exact checks that need no basis change, run on ours alone: commutativity ``hJ hK == hK hJ``
as matrices, and the relations ``hK h(K+1) = 0``, ``hK^2 h(K+2) = h(K+1)^3``.

Exit status 0 iff no invariant disagrees and every self-check holds.
"""

import argparse
import collections
import gzip
import json
import sys

WU_DEFAULT = "/rs/rs_grp_csht/resolutions/validation/wu_exta_t350_chart-data.json.gz"


def open_any(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def rank(rows):
    """Rank over F2 of a list of int bitmasks."""
    pivots = {}
    r = 0
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in pivots:
                v ^= pivots[h]
            else:
                pivots[h] = v
                r += 1
                break
    return r


def compose(a, b, b_cols):
    """Row-vector convention: (x a) b. ``a`` rows are bitmasks over b's rows; ``b`` rows over
    ``b_cols`` columns. Returns the rows of a @ b."""
    out = []
    for row in a:
        acc = 0
        i = 0
        while row:
            if row & 1:
                acc ^= b[i]
            row >>= 1
            i += 1
        out.append(acc)
    return out


def bits(idxs):
    v = 0
    for j in idxs:
        v |= 1 << j
    return v


def load_ours(path):
    dims = {}
    mats = {}  # (K, s, t) -> list of row bitmasks (rows: gens at (s,t); cols: gens at target)
    summary = None
    with open_any(path) as f:
        for line in f:
            rec = json.loads(line)
            if "dims" in rec:
                for n, s, d in rec["dims"]:
                    dims[(s, n + s)] = d
            elif "summary" in rec:
                summary = rec["summary"]
            else:
                name = rec["class"]
                if not (name.startswith("h") and name[1:].isdigit()):
                    continue
                k = int(name[1:])
                s, t = rec["s"], rec["n"] + rec["s"]
                mats[(k, s, t)] = [bits(r) for r in rec["prod"]]
    if summary is None:
        sys.exit(f"{path}: no summary line (truncated run?)")
    return dims, mats, summary


def load_wu(path):
    d = json.load(open_any(path))
    assert d["schema"] == "ext_product.ext_chart.v1", d["schema"]
    t_max = d["t_max"]
    index = {}  # node id -> (s, t, position within bidegree)
    dims = collections.Counter()
    for node in sorted(d["nodes"], key=lambda x: x["id"]):
        key = (node["s"], node["t"])
        index[node["id"]] = (node["s"], node["t"], dims[key])
        dims[key] += 1
    mats = {}  # (K, s, t) -> rows over target positions; only sources with non-zero products
    for act in d["actions"]:
        k = int(act["name"][1:])
        assert (act["s"], act["t"]) == (1, 1 << k), act["name"]
        for p in act["products"]:
            s, t, i = index[p["source"]]
            tgt = [index[x] for x in p["targets"]]
            assert all((ts, tt) == (s + 1, t + (1 << k)) for ts, tt, _ in tgt), p
            m = mats.setdefault((k, s, t), {})
            m[i] = m.get(i, 0) ^ bits(j for _, _, j in tgt)
    return t_max, dims, mats


def wu_matrix(wu_mats, wu_dims, k, s, t):
    rows = wu_mats.get((k, s, t), {})
    return [rows.get(i, 0) for i in range(wu_dims.get((s, t), 0))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ours")
    ap.add_argument("--wu", default=WU_DEFAULT)
    ap.add_argument("--max-report", type=int, default=20)
    args = ap.parse_args()

    dims, mats, summary = load_ours(args.ours)
    max_n, max_s = summary["max_n"], summary["max_s"]
    t_max, wu_dims, wu_mats = load_wu(args.wu)
    ks = sorted({k for k, _, _ in mats})
    print(f"ours: {args.ours}: classes h{ks}, range n<={max_n} s<={max_s}, {len(mats)} blocks")
    print(f"wu:   t_max={t_max}, {sum(wu_dims.values())} basis elements")

    def in_range(s, t):
        return 0 <= s <= max_s and 0 <= t - s <= max_n

    def ours_matrix(k, s, t):
        """None if not computed; else rows (zero rows/blocks filled in)."""
        tgt = (s + 1, t + (1 << k))
        if not (in_range(s, t) and in_range(*tgt)):
            return None
        d = dims.get((s, t), 0)
        m = mats.get((k, s, t))
        if m is None:
            return [0] * d
        assert len(m) == d, (k, s, t, len(m), d)
        return m

    bad = []
    counts = collections.Counter()

    def check(kind, key, a, b):
        counts[kind] += 1
        if a != b:
            counts[kind + "_bad"] += 1
            bad.append(f"{kind} {key}: ours {a} wu {b}")

    # 0. dimensions (already verified globally, but a mismatch would void everything below)
    for (s, t), d in dims.items():
        if t <= t_max:
            check("dim", (s, t), d, wu_dims.get((s, t), 0))
    for (s, t), d in wu_dims.items():
        if in_range(s, t) and (s, t) not in dims:
            check("dim", (s, t), 0, d)

    srcs = sorted({(s, t) for (s, t) in dims if t <= t_max})
    # 1. ranks of each hK
    for k in ks:
        for s, t in srcs:
            if t + (1 << k) > t_max:
                continue
            ours = ours_matrix(k, s, t)
            if ours is None:
                continue
            check("rank", (f"h{k}", s, t), rank(ours), rank(wu_matrix(wu_mats, wu_dims, k, s, t)))
    # 2. two-step composites hJ (hK x)
    for k in ks:
        for j in ks:
            for s, t in srcs:
                t2 = t + (1 << k) + (1 << j)
                if t2 > t_max:
                    continue
                a = ours_matrix(k, s, t)
                b = ours_matrix(j, s + 1, t + (1 << k))
                if a is None or b is None:
                    continue
                wa = wu_matrix(wu_mats, wu_dims, k, s, t)
                wb = wu_matrix(wu_mats, wu_dims, j, s + 1, t + (1 << k))
                check(
                    "composite",
                    (f"h{j}h{k}", s, t),
                    rank(compose(a, b, None)),
                    rank(compose(wa, wb, None)),
                )
    # 3. h-decomposables per target, 4. joint kernel per source
    for s, t in srcs:
        rows_o, rows_w, ok = [], [], True
        for k in ks:
            src = (s - 1, t - (1 << k))
            if src[0] < 0 or src[1] < 0 or src not in dims:
                continue
            m = ours_matrix(k, *src)
            if m is None:
                ok = False
                break
            rows_o += m
            rows_w += wu_matrix(wu_mats, wu_dims, k, *src)
        if ok and rows_o:
            check("decomposable", (s, t), rank(rows_o), rank(rows_w))
        cat_o, cat_w, ok = [0] * dims[(s, t)], [0] * dims[(s, t)], True
        shift = 0
        for k in ks:
            if t + (1 << k) > t_max:
                ok = False
                break
            m = ours_matrix(k, s, t)
            if m is None:
                ok = False
                break
            w = wu_matrix(wu_mats, wu_dims, k, s, t)
            width = dims.get((s + 1, t + (1 << k)), 0)
            cat_o = [x | (y << shift) for x, y in zip(cat_o, m)]
            cat_w = [x | (y << shift) for x, y in zip(cat_w, w)]
            shift += width
        if ok:
            check("joint_kernel", (s, t), dims[(s, t)] - rank(cat_o), dims[(s, t)] - rank(cat_w))

    # Self-checks on ours (exact, same basis on both sides).
    self_bad = []
    for s, t in sorted(dims):
        for k in ks:
            for j in ks:
                if j <= k:
                    continue
                a1 = ours_matrix(k, s, t)
                b1 = ours_matrix(j, s + 1, t + (1 << k))
                a2 = ours_matrix(j, s, t)
                b2 = ours_matrix(k, s + 1, t + (1 << j))
                if None in (a1, b1, a2, b2):
                    continue
                counts["commute"] += 1
                if compose(a1, b1, None) != compose(a2, b2, None):
                    self_bad.append(f"h{k}h{j} != h{j}h{k} at {(s, t)}")
                if j == k + 1:
                    counts["hkhk1"] += 1
                    if any(compose(a1, b1, None)):
                        self_bad.append(f"h{k}h{j} != 0 at {(s, t)}")
            if k + 2 in ks and k + 1 in ks:
                # hK^2 h(K+2) == h(K+1)^3, both as maps out of (s, t).
                path1 = [(k, 0), (k, 1 << k), (k + 2, 2 << k)]
                path2 = [(k + 1, 0), (k + 1, 2 << k), (k + 1, 4 << k)]
                prods = []
                for path in (path1, path2):
                    cur, ss, tt = None, s, t
                    for kk, _ in path:
                        m = ours_matrix(kk, ss, tt)
                        if m is None:
                            cur = None
                            break
                        cur = m if cur is None else compose(cur, m, None)
                        ss, tt = ss + 1, tt + (1 << kk)
                    prods.append(cur)
                if None not in prods:
                    counts["relation"] += 1
                    if prods[0] != prods[1]:
                        self_bad.append(f"h{k}^2 h{k + 2} != h{k + 1}^3 at {(s, t)}")

    print()
    for kind in ["dim", "rank", "composite", "decomposable", "joint_kernel"]:
        print(f"  {kind:13s} compared {counts[kind]:7d}   disagree {counts[kind + '_bad']}")
    print(f"  self-checks: commutativity {counts['commute']}, hK h(K+1)=0 {counts['hkhk1']}, "
          f"hK^2h(K+2)=h(K+1)^3 {counts['relation']}; failures {len(self_bad)}")
    for line in (bad + self_bad)[: args.max_report]:
        print("   ", line)
    compared = sum(counts[k] for k in ["rank", "composite", "decomposable", "joint_kernel"])
    if compared == 0:
        print("NOTHING COMPARED -- empty overlap; not a pass")
        return 2
    ok = not bad and not self_bad
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
