"""Fit the cross blocks of Q = P (x) M's secondary homotopy to local operators, as it acts on
cocycles of the cofiber cochain complex Hom(Q), modulo coboundaries.

Inputs: cell_truth output with CELL_SPHERE (blocks, #act, #dim, #d1) and cofiber_features output
with FEAT_DEGS (H:e and d:a|a' features keyed by cross degree).

For a module M, the unknowns are coef[(c, b, f)] for every pair of cells with |c| > |b| and every
feature f of cross degree |c| - |b|.  For every Q-bidegree (s, T) in range, every cocycle Z of
Hom(Q_s)_T and every coordinate of Hom(Q_{s+2})_{T+1} / im(delta):
    [U_true Z] = [diag Z] + sum coef[(c, b, f)] [F_{c,b,f} Z].

usage: cocfit.py <cell csv> <feature csv> <module> <max stem> <max s> [fixed spec ...]
  a fixed spec "c_deg>b_deg:op+op" fixes that block's operator set to the listed features (no fit);
  "c_deg>b_deg:fit" (default) fits it.
"""
import re
import sys
from collections import defaultdict

cell_p, feat_p, module, max_n, max_s = sys.argv[1:6]
max_n, max_s = int(max_n), int(max_s)
fixed = {}
import os
DIAG = os.environ.get('DIAG')
for spec in sys.argv[6:]:
    k, v = spec.split(":", 1)
    cd, bd = map(int, k.split(">"))
    fixed[(cd, bd)] = None if v == "fit" else ([] if v == "0" else v.split("+"))


def vec(s):
    return [int(x) for x in s.strip('[]" ').split(",") if x.strip()]


def mask(v):
    return sum(1 << i for i, x in enumerate(v) if x)


norm = lambda a: a.replace(";", ", ")
D = {}
D1 = defaultdict(dict)  # (n, s, l) -> {a: mask over y}
ACT = []  # (a, (bd, bi), (cd, ci))
U = {}  # (n, s, l, c, b) -> mask over g
cells = set()
for line in open(cell_p):
    f = line.rstrip("\n").split(",", 4)
    if f[0] == "#dim":
        D[(int(f[1]), int(f[2]))] = int(f[3].split(",")[0])
    elif f[0] == "#d1":
        rest = line.rstrip("\n").split(",", 5)
        D1[(int(rest[1]), int(rest[2]), int(rest[3]))][norm(rest[4])] = mask(vec(rest[5]))
    elif f[0] == "#act":
        r = line.strip().split(",")
        if r[1] == module:
            b, c = (int(r[3]), int(r[4])), (int(r[5]), int(r[6]))
            ACT.append((norm(r[2]), b, c))
            cells |= {b, c}
    elif f[0] == module:
        r = line.strip().split(",", 8)
        n, s, l = int(r[1]), int(r[2]), int(r[3])
        c, b = (int(r[4]), int(r[5])), (int(r[6]), int(r[7]))
        cells |= {b, c}
        U[(n, s, l, c, b)] = mask(vec(r[8]))
cells = sorted(cells)
UALT = {}
if os.environ.get("ALT"):
    for line in open(os.environ["ALT"]):
        if line.startswith(module + ","):
            r = line.strip().split(",", 8)
            UALT[(int(r[1]), int(r[2]), int(r[3]), (int(r[4]), int(r[5])), (int(r[6]), int(r[7])))] = mask(vec(r[8]))

F = defaultdict(dict)  # (delta, n, s, l) -> {name: mask}
pat = re.compile(r'^(\d+),(\d+),(\d+),(\d+),(.*),"(\[.*\])"$')
for line in open(feat_p):
    m = pat.match(line.strip())
    if m and m.group(5)[:2] in ("H:", "d:", "C:"):
        F[(int(m.group(1)),) + tuple(int(m.group(i)) for i in range(2, 5))][m.group(5)] = mask(vec(m.group(6)))

blocks = [(c, b) for c in cells for b in cells if c[0] > b[0]]
SHEAR = os.environ.get('SHEAR')  # 'chi' (antipode shear) or 'plain'
CHI = defaultdict(list)
for line in open(os.environ.get('CHI', 'chi10.txt')):
    f = line.strip().split(",")
    CHI[norm(f[2])] = [norm(x) for x in f[3].split()]
shear = defaultdict(list)  # (c, b) -> H features fixed by the shear
if SHEAR:
    acts = {(a, b, c) for (a, b, c) in ACT}
    for (c, b) in blocks:
        hn = sorted({f[2:] for k, fs in F.items() if k[0] == c[0] - b[0] for f in fs if f.startswith('H:')})
        for e in hn:
            terms = CHI[e] if SHEAR == 'chi' else [e]
            if sum((o, b, c) in acts for o in terms) % 2:
                shear[(c, b)].append('H:' + e)
# the action used by delta_Q: <c, a b> (DCHI unset) or <c, chi(a) b> (DCHI=chi, as in sseq)
_acts = {(a, b, c) for (a, b, c) in ACT}
_alla = sorted(set(CHI))
if os.environ.get('DCHI') == 'chi':
    DACT = [(a, b, c) for a in _alla for b in cells for c in cells
            if sum((o, b, c) in _acts for o in CHI[a]) % 2]
else:
    DACT = ACT
unknowns = []
for (c, b) in blocks:
    delta = c[0] - b[0]
    spec = fixed.get((c[0], b[0]))
    if spec is not None:
        continue
    names = sorted({f for k, fs in F.items() if k[0] == delta for f in fs
                    if not (SHEAR and f.startswith('H:'))})
    unknowns += [(c, b, f) for f in names]
# COMBOS=<file>: the unknowns are whole homogeneous solutions `name#c>b:op+op#c>b:op...`
COMBO = {}
if os.environ.get("COMBOS"):
    for line in open(os.environ["COMBOS"]):
        parts = line.strip().split("#")
        COMBO[parts[0]] = {tuple(map(int, p.split(":", 1)[0].split(">"))): p.split(":", 1)[1].split("+")
                           for p in parts[1:]}
    unknowns = [(None, None, name) for name in COMBO]
ucol = {u: i for i, u in enumerate(unknowns)}


def combo_mask(name, c, b, n, l, s):
    m = 0
    for f in COMBO[name].get((c[0], b[0]), []):
        m ^= F.get((c[0] - b[0], n, s, l), {}).get(f, 0)
    return m


def basis(s, T):
    """basis of Hom(Q_s)_T: list of (cell, n, l)."""
    out = []
    for c in cells:
        n = T - c[0] - s
        for l in range(D.get((n, s), 0)):
            out.append((c, n, l))
    return out


def in_range(s, T):
    return all(T - c[0] - s <= max_n and (T - c[0] - s, s) in D for c in cells
               if T - c[0] - s >= 0) and s <= max_s


def delta_img(s, T, src):
    """delta_Q of basis element src = (c, n, l) in Hom(Q_{s+1})_T, as {(b, n', l'): 1}."""
    c, n, l = src
    out = defaultdict(int)
    for (a, b, cc) in DACT:
        if cc != c:
            continue
        m = D1.get((n, s, l), {}).get(a, 0)
        ny = n + (c[0] - b[0]) - 1
        y = 0
        while m:
            if m & 1:
                out[(b, ny, y)] ^= 1
            m >>= 1
            y += 1
    return {k for k, v in out.items() if v}


def d2_img(s, T, src, blockmask):
    """sum over blocks (c, b) of blockmask(c, b, n, l) placed at (b, n', g)."""
    c, n, l = src
    out = set()
    for b in cells:
        if b[0] > c[0]:
            continue
        m = blockmask(c, b, n, l)
        ng = n + (c[0] - b[0]) - 1
        g = 0
        while m:
            if m & 1:
                out ^= {(b, ng, g)}
            m >>= 1
            g += 1
    return out


class Echelon:
    def __init__(self):
        self.piv = {}

    def reduce(self, v):
        # full normal form: clear every pivot bit (pivot rows have no higher pivot bits)
        for p in sorted(self.piv, reverse=True):
            if v >> p & 1:
                v ^= self.piv[p]
        return v

    def add(self, v):
        v = self.reduce(v)
        if v:
            p = v.bit_length() - 1
            for q in self.piv:  # keep rows reduced w.r.t. each other
                if self.piv[q] >> p & 1:
                    self.piv[q] ^= v
            self.piv[p] = v
        return v


def fixed_mask(c, b, n, l, s):
    if os.environ.get("ALT"):
        return UALT.get((n, s, l, c, b), 0)
    spec = fixed.get((c[0], b[0]))
    if c[0] == b[0]:
        return U.get((n, s, l, c, b), 0)
    m = 0
    for f in shear.get((c, b), []):
        m ^= F.get((c[0] - b[0], n, s, l), {}).get(f, 0)
    if spec is None:
        return m
    for f in spec:
        m ^= F.get((c[0] - b[0], n, s, l), {}).get(f, 0)
    return m


rows = []  # (unknown mask, rhs)
stats = defaultdict(int)
for s in range(0, max_s - 1):
    for T in range(s, max_n + s + 8):
        if not (in_range(s, T) and in_range(s + 1, T) and in_range(s + 1, T + 1) and in_range(s + 2, T + 1)):
            continue
        src = basis(s, T)
        if not src:
            continue
        tgt = basis(s + 2, T + 1)
        tidx = {t: i for i, t in enumerate(tgt)}
        mid = basis(s + 1, T)
        midx = {t: i for i, t in enumerate(mid)}
        # cocycles: kernel of delta on Hom(Q_s)_T
        ech, comb = {}, []
        cocycles = []
        for i, e in enumerate(src):
            v = 0
            for t in delta_img(s, T, e):
                v ^= 1 << midx[t]
            cmb = 1 << i
            while v:
                p = v.bit_length() - 1
                if p in ech:
                    v ^= ech[p][0]
                    cmb ^= ech[p][1]
                else:
                    ech[p] = (v, cmb)
                    break
            if not v:
                cocycles.append(cmb)
        # coboundaries in Hom(Q_{s+2})_{T+1}
        B = Echelon()
        for e in basis(s + 1, T + 1):
            v = 0
            for t in delta_img(s + 1, T + 1, e):
                v ^= 1 << tidx[t]
            B.add(v)

        def apply(Z, blockmask):
            v = 0
            i = 0
            while Z:
                if Z & 1:
                    for t in d2_img(s, T, src[i], blockmask):
                        v ^= 1 << tidx[t]
                Z >>= 1
                i += 1
            return B.reduce(v)

        if os.environ.get('CHECKCOC') and in_range(s + 3, T + 1):
            # the true d2 must send cocycles to cocycles: delta(Z o U) = 0 in Hom(Q_{s+3})_{T+1}
            nxt = {t: i for i, t in enumerate(basis(s + 3, T + 1))}
            for Z in cocycles:
                img = set()
                i, ZZ = 0, Z
                while ZZ:
                    if ZZ & 1:
                        img ^= d2_img(s, T, src[i], lambda c, b, n, l: U.get((n, s, l, c, b), 0))
                    ZZ >>= 1
                    i += 1
                out = set()
                for t in img:
                    out ^= delta_img(s + 2, T + 1, t)
                stats["U(Z) cocycle"] += not out
                stats["U(Z) NOT cocycle"] += bool(out)
        for Z in cocycles:
            truth = apply(Z, lambda c, b, n, l: U.get((n, s, l, c, b), 0))
            known = apply(Z, lambda c, b, n, l: fixed_mask(c, b, n, l, s))
            rhs = truth ^ known
            if rhs and not unknowns and DIAG:
                sup = [src[i] for i in range(len(src)) if Z >> i & 1]
                res = [tgt[i] for i in range(len(tgt)) if rhs >> i & 1]
                print(f"  fail s={s} T={T}: Z={sup} residual={res}")
            cols = {}
            for u in unknowns:
                c0, b0, f = u
                if c0 is None:
                    fv = apply(Z, lambda c, b, n, l: combo_mask(f, c, b, n, l, s) if c[0] > b[0] else 0)
                else:
                    fv = apply(Z, lambda c, b, n, l: F.get((c0[0] - b0[0], n, s, l), {}).get(f, 0)
                               if (c, b) == (c0, b0) else 0)
                if fv:
                    cols[u] = fv
            width = max([rhs.bit_length()] + [v.bit_length() for v in cols.values()])
            for bit in range(width):
                m = 0
                for u, v in cols.items():
                    if v >> bit & 1:
                        m |= 1 << ucol[u]
                r = rhs >> bit & 1
                if m or r:
                    rows.append((m, r, (s, T)))
            stats["cocycles"] += 1
            stats["nonzero d2 mod B"] += truth != 0

# solve
nc = len(unknowns)
piv = {}
bad = []
for m, r, where in rows:
    v = m | (r << nc)
    while v & ((1 << nc) - 1):
        p = (v & ((1 << nc) - 1)).bit_length() - 1
        if p in piv:
            v ^= piv[p]
        else:
            piv[p] = v
            break
    else:
        if v >> nc:
            bad.append(where)
sol = 0
for p in sorted(piv):
    v = piv[p]
    rest = v & ((1 << p) - 1)
    val = ((v >> nc) & 1) ^ (bin(rest & sol).count("1") & 1)
    if val:
        sol |= 1 << p
print(f"{module}: {dict(stats)}, {len(rows)} equations, {nc} unknowns, rank {len(piv)}, "
      f"{'CONSISTENT' if not bad else 'INCONSISTENT at ' + str(sorted(set(bad))[:12]) + f' ({len(bad)})'}")
free = [unknowns[p] for p in range(nc) if p not in piv]
by_block = defaultdict(list)
for i, u in enumerate(unknowns):
    if sol >> i & 1:
        by_block[(u[0], u[1])].append(u[2])
for (c, b) in blocks:
    if fixed.get((c[0], b[0])) is not None:
        continue
    print(f"  {c}->{b}: {' + '.join(by_block.get((c, b), [])) or '0'}")
undet = defaultdict(int)
for u in free:
    undet[(u[0], u[1])] += 1
print("  undetermined per block:", dict(undet))
if COMBO:
    print("  combos used:", [unknowns[i][2] for i in range(nc) if sol >> i & 1])
    print("  combos undetermined:", [unknowns[p][2] for p in range(nc) if p not in piv])
