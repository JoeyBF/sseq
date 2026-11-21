"""Validate iota_tmf against Bruner-Rognes Table 1.1 (t-s <= 48).

usage: validate_br.py <prefix of restrict_unit output with products> <names.csv>
Reads <prefix>.csv (iota), <prefix>_hB.csv, <prefix>_hA.csv, <prefix>_products.csv,
<prefix>_basis.csv.
"""
import csv, sys
from collections import defaultdict

pre, names_path = sys.argv[1], sys.argv[2]
vec = lambda s: [int(x) for x in s.strip('[]').split(',') if x.strip() != '']

iota = {}  # (n,s) -> rows (A gens) -> vector over B gens
for r in csv.DictReader(open(pre + '.csv')):
    iota.setdefault((int(r['stem']), int(r['Adams_filtration'])), []).append(vec(r['image']))


def hmat(path):
    h = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        h[(int(r['i']), int(r['stem']), int(r['Adams_filtration']))][int(r['index'])] = vec(r['product'])
    return h


hA, hB = hmat(pre + '_hA.csv'), hmat(pre + '_hB.csv')

prod = defaultdict(dict)  # y -> (x_stem,x_s) -> {k: vec}
for r in csv.DictReader(open(pre + '_products.csv')):
    y = (int(r['y_stem']), int(r['y_s']), int(r['y_index']))
    prod[y].setdefault((int(r['x_stem']), int(r['x_s'])), {})[int(r['x_index'])] = vec(r['product'])

dimB = defaultdict(int)
for r in csv.DictReader(open(pre + '_basis.csv')):
    dimB[(int(r['stem']), int(r['Adams_filtration']))] += 1


class C:
    def __init__(s, n, f, v):
        s.n, s.f, s.v = n, f, v

    def __repr__(s):
        return f"({s.n},{s.f}){s.v}"


def apply_rows(rows, v, outdim):
    out = [0] * outdim
    for k, c in enumerate(v):
        if c:
            for j, x in enumerate(rows[k]):
                out[j] ^= x
    return out


def mul_h(i, x):
    d = 1 << i
    tgt = (x.n + d - 1, x.f + 1)
    rows = hB[(i, x.n, x.f)]
    return C(*tgt, apply_rows([rows[k] for k in range(len(x.v))], x.v, dimB[tgt]))


gen = {}


def mul(y, x):
    yk = gen[y]
    tgt = (x.n + yk[0], x.f + yk[1])
    if dimB[tgt] == 0:
        return C(*tgt, [])
    tab = prod[yk].get((x.n, x.f), {})
    return C(*tgt, apply_rows([tab[k] for k in range(len(x.v))], x.v, dimB[tgt]))


def unique(n, f):
    assert dimB[(n, f)] == 1, f"Ext_A(2) at ({n},{f}) has dim {dimB[(n, f)]}"
    return C(n, f, [1])


G = {}
for nm, (n, f) in dict(h0=(0, 1), h1=(1, 1), h2=(3, 1), c0=(8, 3), w1=(8, 4), alpha=(12, 3),
                       d0=(14, 4), beta=(15, 3), e0=(17, 4), g=(20, 4), gamma=(25, 5)).items():
    G[nm] = unique(n, f)
    gen[nm] = (n, f, 0)


def w1p(x, k):
    for _ in range(k):
        x = mul('w1', x)
    return x


h1, h2 = G['h1'], G['h2']
E = {
    'h0': G['h0'], 'h1': h1, 'h2': h2, 'h3': 0, 'c0': G['c0'], 'P h1': w1p(h1, 1), 'P h2': w1p(h2, 1),
    'd0': G['d0'], 'h4': 0, 'P c0': w1p(G['c0'], 1), 'e0': G['e0'], 'P^2 h1': w1p(h1, 2),
    'f0': mul_h(2, G['beta']), 'c1': 0, 'P^2 h2': w1p(h2, 2), 'tau g': G['g'], 'P d0': w1p(G['d0'], 1),
    'i': w1p(G['beta'], 1), 'P^2 c0': w1p(G['c0'], 2), 'P e0': w1p(G['e0'], 1), 'P^3 h1': w1p(h1, 3),
    'j': mul('alpha', G['d0']), 'P^3 h2': w1p(h2, 3), 'k': mul('alpha', G['e0']),
    'r': mul('beta', G['beta']), 'P^2 d0': w1p(G['d0'], 2), 'h5': 0, 'n': 0, 'd1': 0, 'q': 0,
    'l': mul('alpha', G['g']), 'P^3 c0': w1p(G['c0'], 3), 'p': 0, 'P^2 e0': w1p(G['e0'], 2),
    'P^4 h1': w1p(h1, 4), 'P j': w1p(mul('alpha', G['d0']), 1), 'm': mul('beta', G['g']),
    'P^4 h2': w1p(h2, 4), 't': 0, 'x': 0, 'e1': 0, 'y': 0, 'P^3 d0': w1p(G['d0'], 3),
    'u': mul('gamma', G['d0']), 'P^2 i': w1p(G['beta'], 3), 'f1': 0, 'P^4 c0': w1p(G['c0'], 4),
    'c2': 0, 'z': mul('alpha', mul('alpha', G['e0'])), 'P^3 e0': w1p(G['e0'], 3), 'P^5 h1': w1p(h1, 5),
    'v': mul('gamma', G['e0']), 'P^2 j': w1p(mul('alpha', G['d0']), 2), 'P^5 h2': w1p(h2, 5),
    'g2': 0, 'tau w': mul('gamma', G['g']), 'B1': 0, 'N': 0, 'P^4 d0': w1p(G['d0'], 4),
    'Q': 0, 'P u': w1p(mul('gamma', G['d0']), 1), 'B2': 0, 'P^5 c0': w1p(G['c0'], 5),
}

names = {}
for r in csv.DictReader(open(names_path)):
    alts = [vec(a.strip()) for a in r['vector'].split('|')]
    names.setdefault(r['name'], (int(r['stem']), int(r['filtration']), alts))


def iota_of(n, f, v):
    rows = iota[(n, f)]
    return apply_rows(rows, v, len(rows[0]) if rows else 0)


ker47 = [v for v in ([1, 0], [0, 1], [1, 1]) if not any(iota_of(47, 13, v))]
print("kernel of iota at (47,13):", ker47, " names: Q' =", names["Q'"][2][0], " P u =", names['P u'][2][0])

ok = bad = 0
for br, exp in E.items():
    if br == 'Q':
        if len(ker47) != 1:
            print("MISMATCH Q: kernel", ker47)
            bad += 1
        else:
            q = ker47[0]
            ha0, ha1 = hA[(0, 47, 13)], hA[(1, 47, 13)]
            print(f"  Q = {q}: h0 Q = {apply_rows(ha0, q, len(ha0[0]))}, h1 Q = {apply_rows(ha1, q, len(ha1[0]))}")
            for nm, v in [("P u", names['P u'][2][0]), ("Q'", names["Q'"][2][0])]:
                print(f"  {nm} = {v}: h0 = {apply_rows(ha0, v, len(ha0[0]))}, h1 = {apply_rows(ha1, v, len(ha1[0]))}")
            ok += 1
        continue
    n, f, alts = names[br]
    if exp != 0:
        assert (exp.n, exp.f) == (n, f), (br, exp, n, f)
    good = True
    for v in alts:  # every representative named for an ambiguous class
        got = iota_of(n, f, v)
        want = [0] * len(got) if exp == 0 else exp.v
        if got != want:
            good = False
            print(f"MISMATCH {br} ({n},{f}) vec {v}: iota = {got}, BR = {want}")
    ok += good
    bad += not good
    if exp != 0 and not any(exp.v):
        print(f"WARNING: expected image of {br} is zero in our basis")
print(f"Table 1.1: {ok} rows agree, {bad} mismatches (of {len(E)})")
print("A(2) dims at generator bidegrees:", {k: dimB[(c.n, c.f)] for k, c in G.items()})
