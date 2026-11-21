"""Validate the Ext_{A(2)} ring export.

usage: validate_ring.py <ring dir> <relations.txt> [<iota prefix> <sphere ring dir (stem-filtered ok)>]
 1. h_i products agree with <iota prefix>_hB.csv (read off the A(2)-resolution's differential);
 2. the 54 Shimada-Iwai relations and the extra Groebner elements listed in relations.txt hold;
 3. multiplication by w1 is injective wherever the target is in range;
 4. iota_tmf is a ring map: iota(y x) = iota(y) iota(x) for every sphere indecomposable y and every
    basis element x with y x in range (sphere products from the sphere ring export, same A-basis).
"""
import csv, re, sys
from collections import defaultdict
from itertools import product as iproduct

ring, rel_path = sys.argv[1], sys.argv[2]
iota_pre = sys.argv[3] if len(sys.argv) > 3 else None
sphere = sys.argv[4] if len(sys.argv) > 4 else None
vec = lambda s: [int(x) for x in s.strip().strip('"').strip('[]').split(',') if x.strip() != '']

rank = {}
for r in csv.DictReader(open(f'{ring}/rank.csv')):
    rank[(int(r['stem']), int(r['Adams_filtration']))] = int(r['rank'])
maxn = max(n for n, s in rank)
maxs = max(s for n, s in rank)
dim = lambda b: rank.get(b, 0)
inrange = lambda b: b[0] <= maxn and b[1] <= maxs

ind = []  # (n, s, i, name)
for r in csv.DictReader(open(f'{ring}/indecomposables.csv')):
    ind.append((int(r['stem']), int(r['Adams_filtration']), int(r['index']), r['name']))
byname = {nm: (n, s, i) for n, s, i, nm in ind}
gp = defaultdict(dict)  # y -> (xn,xs,xk) -> vec
for r in csv.reader(open(f'{ring}/generator_products.csv')):
    if r[0] == 'stem_1':
        continue
    y = tuple(map(int, r[:3]))
    gp[y][tuple(map(int, r[3:6]))] = vec(r[6])


def mul_gen(y, b, v):
    """y * (vector v at bidegree b); None if out of range."""
    tb = (b[0] + y[0], b[1] + y[1])
    if not inrange(tb):
        return None, tb
    out = [0] * dim(tb)
    for k, c in enumerate(v):
        if c:
            p = gp[y].get((b[0], b[1], k))
            if p:
                for j, x in enumerate(p):
                    out[j] ^= x
    return out, tb


def monomial(word):
    """Evaluate a monomial given as list of generator names; returns (vec, bidegree) or None."""
    v, b = [1], (0, 0)
    for nm in word:
        v, b = mul_gen(byname[nm], b, v)
        if v is None:
            return None
    return v, b


def parse_mono(m):
    word = []
    for tok in m.split():
        nm, _, e = tok.partition('^')
        word += [nm] * (int(e) if e else 1)
    return word


def eval_sum(expr):
    tot, bid = None, None
    for m in expr.split('+'):
        m = m.strip()
        if m == '0':
            continue
        r = monomial(parse_mono(m))
        if r is None:
            return 'out'
        v, b = r
        if tot is None:
            tot, bid = v, b
        else:
            assert b == bid, (expr, b, bid)
            tot = [a ^ c for a, c in zip(tot, v)]
    return tot


# 2. relations
ok = bad = skip = 0
lines = open(rel_path).read().splitlines()
for ln in lines:
    m = re.match(r'^\s*(\d+)\s+(\d+)\s*:\s*(.*?)\s*=\s*(.*?)\s{2,}', ln + '  ')
    if not m:
        continue
    lhs, rhs = m.group(3), m.group(4).split('(')[0].strip()
    a, c = eval_sum(lhs), eval_sum(rhs if rhs else '0')
    if a == 'out' or c == 'out':
        skip += 1
        continue
    if c is None:
        c = [0] * len(a)
    if a is None:
        a = [0] * len(c)
    if a == c:
        ok += 1
    else:
        bad += 1
        print('RELATION FAILS:', ln.strip(), a, c)
# extra Groebner elements (each = 0)
txt = ' '.join(l.lstrip('#').strip() for l in lines)
gb = txt.split('Extra (non-presentation) Groebner elements include')[1].split('- Massey')[0]
gok = gbad = 0
for e in gb.replace('\n', ' ').split(','):
    e = e.strip().rstrip('.')
    if not e:
        continue
    r = eval_sum(e)
    if r == 'out':
        continue
    if r is None or not any(r):
        gok += 1
    else:
        gbad += 1
        print('GROEBNER ELEMENT NONZERO:', e, r)
print(f'relations: {ok} hold, {bad} fail, {skip} out of range; Groebner elements: {gok} zero, {gbad} nonzero')

# delta characterization (BR: h0 delta != 0, h1 delta != 0; alpha g = the other)
d = byname['delta']
h0d, _ = mul_gen(byname['h0'], d[:2], [1 if k == d[2] else 0 for k in range(dim(d[:2]))])
h1d, _ = mul_gen(byname['h1'], d[:2], [1 if k == d[2] else 0 for k in range(dim(d[:2]))])
ag = monomial(['alpha', 'g'])
print(f'delta = x_{d}: h0 delta = {h0d}, h1 delta = {h1d}; alpha g = {ag[0]}')

# 3. w1 injective
w1 = byname['w1']
inj_bad = inj_ok = 0
for b, r in rank.items():
    if not inrange((b[0] + 8, b[1] + 4)):
        continue
    rows = [mul_gen(w1, b, [int(i == k) for i in range(r)])[0] for k in range(r)]
    # rank over F2
    piv, rk = {}, 0
    for row in rows:
        x = int(''.join(map(str, row)) or '0', 2)
        for p in sorted(piv, reverse=True):
            if x >> p & 1:
                x ^= piv[p]
        if x:
            piv[x.bit_length() - 1] = x
            rk += 1
    if rk == r:
        inj_ok += 1
    else:
        inj_bad += 1
        print('w1 NOT injective at', b)
print(f'w1 injective on {inj_ok} bidegrees, fails on {inj_bad}')

if iota_pre:
    # 1. h_i vs differential read-off
    hb_ok = hb_bad = 0
    for r in csv.reader(open(iota_pre + '_hB.csv')):
        if r[0] == 'i' or int(r[0]) > 2:
            continue
        i, n, s, k = map(int, r[:4])
        want = vec(r[4])
        tb = (n + (1 << i) - 1, s + 1)
        if not inrange(tb) or not inrange((n, s)):
            continue
        got = gp[byname[f'h{i}']].get((n, s, k), [0] * dim(tb))
        if got == want:
            hb_ok += 1
        else:
            hb_bad += 1
            print('h_i MISMATCH', i, n, s, k, got, want)
    print(f'h_i products vs differential: {hb_ok} agree, {hb_bad} differ')

if iota_pre and sphere:
    # 4. iota ring map. Express every A(2) basis element as a polynomial in indecomposables.
    iota = defaultdict(list)
    for r in csv.reader(open(iota_pre + '.csv')):
        if r[0] == 'stem':
            continue
        iota[(int(r[0]), int(r[1]))].append(vec(r[3]))
    # monomials by bidegree, built by multiplying generators onto monomials (sorted words)
    gens = [nm for _, _, _, nm in ind]
    monos = {(0, 0): [((), [1])]}
    frontier = [((), [1], (0, 0))]
    while frontier:
        nxt = []
        for word, v, b in frontier:
            last = gens.index(word[-1]) if word else 0
            for gi in range(last, len(gens)):
                nv, tb = mul_gen(byname[gens[gi]], b, v)
                if nv is None or not any(nv):
                    continue
                w = word + (gens[gi],)
                monos.setdefault(tb, []).append((w, nv))
                nxt.append((w, nv, tb))
        frontier = nxt

    def express(b, v):
        """v at b as a sum of monomial words (Gaussian elimination)."""
        basis = []  # (pivot vec as int, set of words)
        for w, mv in monos.get(b, []):
            x, ws = int(''.join(map(str, mv)), 2), {w}
            for px, pws in basis:
                if x ^ px < x:
                    x ^= px
                    ws ^= pws
            if x:
                basis.append((x, ws))
                basis.sort(key=lambda t: -t[0])
        x, ws = int(''.join(map(str, v)), 2), set()
        for px, pws in basis:
            if x ^ px < x:
                x ^= px
                ws ^= pws
        assert x == 0, ('not spanned by monomials', b)
        return ws

    def times_word(word, b, v):
        for nm in word:
            v, b = mul_gen(byname[nm], b, v)
            if v is None:
                return None, b
        return v, b

    srank = {}
    for r in csv.DictReader(open(f'{sphere}/rank.csv')):
        srank[(int(r['stem']), int(r['Adams_filtration']))] = int(r['rank'])
    # the iota rows must be over the same A-basis as the sphere export
    assert all(len(iota[b]) == srank.get(b, 0) for b in iota if srank.get(b, 0) or iota[b]), 'rank mismatch'
    rm_ok = rm_bad = rm_skip = 0
    for r in csv.reader(open(f'{sphere}/generator_products.csv')):
        if r[0] == 'stem_1':
            continue
        n1, s1, i1, n2, s2, i2 = map(int, r[:6])
        pv = vec(r[6])
        tb = (n1 + n2, s1 + s2)
        if not inrange(tb) or tb not in iota:
            rm_skip += 1
            continue
        # lhs = iota(y x)
        rows = iota[tb]
        lhs = [0] * dim(tb)
        for k, c in enumerate(pv):
            if c:
                lhs = [a ^ b for a, b in zip(lhs, rows[k])]
        # rhs = iota(y) iota(x)
        iy = iota[(n1, s1)][i1]
        ix = iota[(n2, s2)][i2]
        rhs = [0] * dim(tb)
        if any(iy) and any(ix):
            for w in express((n1, s1), iy):
                v, b2 = times_word(w, (n2, s2), ix)
                if v is None:
                    rhs = None
                    break
                rhs = [a ^ b for a, b in zip(rhs, v)]
        if rhs is None:
            rm_skip += 1
        elif lhs == rhs:
            rm_ok += 1
        else:
            rm_bad += 1
            print('RING MAP FAILS', r[:6], lhs, rhs)
    print(f'iota ring map on sphere products y*x: {rm_ok} agree, {rm_bad} fail, {rm_skip} out of range')
    # zero products: iota(y) iota(x) = 0 whenever y x = 0 (rows absent from the sphere export)
    present = set()
    for r in csv.reader(open(f'{sphere}/generator_products.csv')):
        if r[0] != 'stem_1':
            present.add(tuple(map(int, r[:6])))
    z_ok = z_bad = 0
    for r in csv.DictReader(open(f'{sphere}/indecomposables.csv')):
        y = (int(r['stem']), int(r['Adams_filtration']), int(r['index']))
        if y[:2] not in iota or not any(iota[y[:2]][y[2]]):
            continue
        iy = iota[y[:2]][y[2]]
        ws = express(y[:2], iy)
        for b2, rows in iota.items():
            tb = (y[0] + b2[0], y[1] + b2[1])
            if not inrange(tb) or tb not in iota:
                continue
            for k, ix in enumerate(rows):
                if (y + (b2[0], b2[1], k)) in present or not any(ix):
                    continue
                rhs = [0] * dim(tb)
                for w in ws:
                    v, _ = times_word(w, b2, ix)
                    rhs = [a ^ b for a, b in zip(rhs, v)]
                if any(rhs):
                    z_bad += 1
                    print('RING MAP FAILS (zero product)', y, b2, k, rhs)
                else:
                    z_ok += 1
    print(f'iota ring map on zero sphere products: {z_ok} agree, {z_bad} fail')
