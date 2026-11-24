"""D-model prototype for the sphere: orders 3 and 4 of 'D is a derivation of mu' against h_0..h_3.

Adapted basis per bidegree: [R | B | N] = easy-product cycle representatives (a complement of the
d2-boundaries in Z2), a basis of the d2-boundaries, a complement of Z2. Gauge G3: D3 maps Z2 into
span(R) and kills B; D3 maps N into Z2. D h_i = 0. mu1(h_i, r) = L for r in R (the easy table);
mu1(h_i, -) on B and N, D4 on Z2 and mu2(h_i, -) on R are unknown.

Equations:
  O3 on every basis element: D3(h_i x) + h_i D3 x + D2 mu1(h_i, x) + mu1(h_i, D2 x) = 0
  O4 on every r in R:        D4(h_i r) + h_i D4 r + D3 L(h_i, r) + mu1(h_i, D3 r) + D2 mu2(h_i, r) = 0
Queries: the R-coordinates of D3 r for r in R, which is d3 on E3 in the basis R.

usage: dproto.py <max stem> <use O4: 0|1> [hold_d3 csv]
"""
import csv
import sys
from collections import defaultdict

T = '/wsu/home/hd/hd72/hd7264/.claude/jobs/888a5512/tmp/'
R_ = '/rs/rs_grp_csht/resolutions/sphere_gpu_n400/ring_n400_nsweep/'
NMAX = int(sys.argv[1])
USE_O4 = sys.argv[2] == '1'
HOLD = sys.argv[3] if len(sys.argv) > 3 else None
HS = [0, 1, 3, 7]  # stems of h_0..h_3
PAD = 9


def mask(s):
    s = s.strip().strip('"').strip('[]')
    return sum(1 << i for i, t in enumerate(s.split(',')) if t.strip() == '1')


def bits(v):
    i = 0
    while v:
        if v & 1:
            yield i
        v >>= 1
        i += 1


dim = {}
for r in csv.reader(open(R_ + 'rank.csv')):
    if r[0] != 'stem' and int(r[0]) <= NMAX + PAD:
        dim[(int(r[0]), int(r[1]))] = int(r[2])
Dm = lambda n, f: dim.get((n, f), 0)
d2 = {}
for r in csv.reader(open(T + 'full/d2_secondary_n200.csv')):
    if r[0] != 'stem' and int(r[0]) <= NMAX + PAD:
        d2[(int(r[0]), int(r[1]), int(r[2]))] = mask(r[3])
hp = defaultdict(dict)
for r in csv.reader(open(T + 'cd/hprod.csv')):
    if r[0].startswith('stem') or int(r[0]) not in HS or int(r[3]) > NMAX + PAD:
        continue
    hp[(HS.index(int(r[0])), int(r[3]), int(r[4]))][int(r[5])] = mask(r[6])
easy = defaultdict(dict)  # (n, f) -> y -> {i: L}
for line in open(T + 'secprod/easy3_n200.out'):
    if not line.startswith('[x_'):
        continue
    lhs, rhs = line.strip().split(' = ')
    ys, hs = lhs.rsplit('] [h', 1)
    i = int(hs.rstrip(']'))
    if i > 3:
        continue
    _, l = rhs.split(' + λ ')
    y = 0
    for term in ys.lstrip('[').split(' + '):
        w = [int(x) for x in term.strip()[3:-1].split(',')]
        n, f, y = w[0], w[1], y | 1 << w[2]
    if n <= NMAX + PAD:
        easy[(n, f)].setdefault(y, {})[i] = mask(l)


def lin(op, n, f, v):
    out = 0
    for k in bits(v):
        out ^= op(n, f, k)
    return out


d2b = lambda n, f, k: d2.get((n, f, k), 0)
# products c (at (2^i - 1, 2)) x basis, from generator_products (c may be h0 h_i: a product of h's)
def cmul(i, t, n, f, v):
    """the t-th basis element of (2^i - 1, 2) times v: (2^i - 1, 2) is spanned by h0 h_i here"""
    if HS[i] == 0:
        return hmul(0, n + 0, f + 1, hmul(0, n, f, v))  # h0^2
    return hmul(0, n + HS[i], f + 1, hmul(i, n, f, v))  # h0 h_i
hmul = lambda i, n, f, v: lin(lambda a, b, k: hp.get((i, a, b), {}).get(k, 0), n, f, v)
d2v = lambda n, f, v: lin(d2b, n, f, v)


class Basis:
    """adapted basis of one bidegree and the change of coordinates from E2"""

    def __init__(self, n, f):
        d = Dm(n, f)
        self.n, self.f, self.d = n, f, d
        reps = list(easy.get((n, f), {}).keys())
        bnd = []
        if f >= 2:
            for k in range(Dm(n + 1, f - 2)):
                bnd.append(d2v(n + 1, f - 2, 1 << k))
        # independent subsets, in order R, B, N
        self.vecs, self.kind = [], []
        piv = []

        def push(v, kind):
            w = v
            for p, pv in piv:
                if w >> p & 1:
                    w ^= pv
            if w == 0:
                return False
            p = (w & -w).bit_length() - 1
            piv.append((p, w))
            self.vecs.append(v)
            self.kind.append(kind)
            return True

        for b in bnd:
            push(b, 'B')
        self.dropped = 0
        for r in reps:
            if not push(r, 'R'):
                self.dropped += 1  # an easy representative inside the boundaries: L unused
        # the kernel of d2, completed beyond the easy representatives (kind Q: mu1 unknown)
        imgs = [d2v(n, f, 1 << k) for k in range(d)]
        kp = []
        for k in range(d):
            v, c = imgs[k], 1 << k
            for p_, pv, pc in kp:
                if v >> p_ & 1:
                    v ^= pv
                    c ^= pc
            if v:
                kp.append(((v & -v).bit_length() - 1, v, c))
            else:
                push(c, 'Q')
        self.ok = True
        for k in range(d):
            push(1 << k, 'N')
        assert len(self.vecs) == d
        order = {'R': 0, 'Q': 1, 'B': 2, 'N': 3}
        pairs = sorted(zip(self.vecs, self.kind), key=lambda x: order[x[1]])
        self.vecs = [v for v, _ in pairs]
        self.kind = [k for _, k in pairs]
        self.nR = self.kind.count('R') + self.kind.count('Q')  # cycle representatives
        self.nB = self.kind.count('B')
        # inverse: E2 coords -> adapted coords, by elimination with tracking
        self.inv = []
        rows = [(v, 1 << j) for j, v in enumerate(self.vecs)]
        ech = []
        for v, c in rows:
            for p, pv, pc in ech:
                if v >> p & 1:
                    v ^= pv
                    c ^= pc
            p = (v & -v).bit_length() - 1
            ech.append((p, v, c))
        self.ech = ech

    def coords(self, v):
        c = 0
        for p, pv, pc in self.ech:
            if v >> p & 1:
                v ^= pv
                c ^= pc
        assert v == 0
        return c


def span(vs):
    piv = []
    for v in vs:
        for p, pv in piv:
            if v >> p & 1:
                v ^= pv
        if v:
            piv.append(((v & -v).bit_length() - 1, v))
    return piv


B = {}


def basis(n, f):
    if (n, f) not in B:
        B[(n, f)] = Basis(n, f) if Dm(n, f) else None
    return B[(n, f)]


bad_z = [b for b in dim if b[0] <= NMAX + 4 and Dm(*b) and basis(*b).dropped]
print(f"bidegrees with an easy rep inside the boundaries: {len(bad_z)} {sorted(bad_z)[:8]}",
      file=sys.stderr)

# --- linear expressions: vector over adapted coords of a bidegree, each coord a set of vars + const
nv = 0
VAR = {}
NAME = []


def var(key):
    global nv
    if key not in VAR:
        VAR[key] = nv
        NAME.append(key)
        nv += 1
    return VAR[key]


class Expr:
    """sum over adapted coordinates t of (xor of vars) + const, in bidegree (n, f)"""

    def __init__(self, n, f):
        self.n, self.f = n, f
        self.c = defaultdict(int)  # coord -> var bitmask (bit nv_const reserved later)
        self.k = 0  # constant, as adapted-coordinate mask

    def addvar(self, t, v):
        self.c[t] ^= 1 << v

    def addconst(self, cmask):
        self.k ^= cmask

    def addexpr(self, e):
        for t, m in e.c.items():
            self.c[t] ^= m
        self.k ^= e.k


def const_vec(n, f, v):
    """known E2 vector -> Expr"""
    e = Expr(n, f)
    b = basis(n, f)
    if b:
        e.k = b.coords(v)
    return e


def apply_known(op_vec, e, n2, f2):
    """apply a known linear map (E2 vector at e's bidegree -> E2 vector at (n2, f2)) to an Expr"""
    out = Expr(n2, f2)
    b, b2 = basis(e.n, e.f), basis(n2, f2)
    if not b or not b2:
        return out
    img = [b2.coords(op_vec(b.vecs[j])) for j in range(b.d)]
    for t, m in e.c.items():
        for t2 in bits(img[t]):
            out.c[t2] ^= m
    for t in bits(e.k):
        out.k ^= img[t]
    return out


def D3_basis(n, f, j):
    """D3 of adapted basis element j at (n, f), an Expr at (n - 1, f + 3): no gauge fixing beyond
    D h_i = 0; D3 lands in Z2 of the target (D2 D3 = D3 D2 on cycles)"""
    e = Expr(n - 1, f + 3)
    b, bt = basis(n, f), basis(n - 1, f + 3)
    if not b or not bt:
        return e
    if (n, f) in [(s, 1) for s in HS]:
        return e  # gauge: D h_i = 0
    for k in range(bt.nR + bt.nB):
        e.addvar(k, var(('a', n, f, j, k)))
    return e


def D3(ex):
    out = Expr(ex.n - 1, ex.f + 3)
    if not basis(ex.n, ex.f):
        return out
    for t in bits(ex.k):
        out.addexpr(D3_basis(ex.n, ex.f, t))
    if ex.c:
        raise ValueError("D3 of a non-constant expression is bilinear")
    return out


def D4_basis(n, f, j):
    e = Expr(n - 1, f + 4)
    b, bt = basis(n, f), basis(n - 1, f + 4)
    if not b or not bt or b.kind[j] == 'N':
        return e
    if (n, f) in [(s, 1) for s in HS]:
        return e
    for t in range(bt.d):
        e.addvar(t, var(('d4', n, f, j, t)))
    return e


def g1_vec(n, f, v):
    """g1 (the gauge, filtration +1) of an E2 vector at (n, f), an Expr at (n, f + 1) over
    adapted coordinates; g1 of each adapted basis element is unknown"""
    e = Expr(n, f + 1)
    b, bt = basis(n, f), basis(n, f + 1)
    if not b or not bt:
        return e
    for j in bits(b.coords(v)):
        for t in range(bt.d):
            e.addvar(t, var(('g', n, f, j, t)))
    return e


def mu1_basis(i, n, f, j):
    """mu1(h_i, basis element j) at (n + 2^i - 1, f + 2). On an easy representative y:
    L_i(y) + g1(h_i y) + h_i g1(y) + g1(h_i) y (the orbit of the standard-lift value under the
    gauge g1); elsewhere unknown outright."""
    tn, tf = n + HS[i], f + 2
    b, bt = basis(n, f), basis(tn, tf)
    e = Expr(tn, tf)
    if not b or not bt:
        return e
    if b.kind[j] == 'R':
        y = b.vecs[j]
        L = easy[(n, f)][y].get(i)
        if L is not None:
            e.k = bt.coords(L)
            hy = hmul(i, n, f, y)
            if hy:
                e.addexpr(g1_vec(tn, tf - 1, hy))
            e.addexpr(hmul_expr(i, g1_vec(n, f, y)))
            # g1(h_i) y: g1(h_i) over the E2 basis at (2^i - 1, 2)
            for t in range(Dm(HS[i], 2)):
                cy = cmul(i, t, n, f, y)
                if cy:
                    c = var(('g', HS[i], 1, 0, t))
                    for t2 in bits(bt.coords(cy)):
                        e.c[t2] ^= 1 << c
            return e
    for t in range(bt.d):
        e.addvar(t, var(('m1', i, n, f, j, t)))
    return e


def mu2_basis(i, n, f, j):
    tn, tf = n + HS[i], f + 3
    bt = basis(tn, tf)
    e = Expr(tn, tf)
    if not bt:
        return e
    for t in range(bt.d):
        e.addvar(t, var(('m2', i, n, f, j, t)))
    return e


EQ = []  # (var bitmask, const bit)


def emit(e, label):
    for t in set(e.c) | set(bits(e.k)):
        m = e.c.get(t, 0)
        c = e.k >> t & 1
        if m or c:
            EQ.append((m, c, label))


def hmul_expr(i, e):
    return apply_known(lambda v: hmul(i, e.n, e.f, v), e, e.n + HS[i], e.f + 1)


def d2_expr(e):
    return apply_known(lambda v: d2v(e.n, e.f, v), e, e.n - 1, e.f + 2)


def expr_of_vec(n, f, v):
    return const_vec(n, f, v)


for (n, f) in sorted(dim):
    if n > NMAX or not Dm(n, f):
        continue
    b = basis(n, f)
    for j in range(b.d):
        x = b.vecs[j]
        for i in range(4):
            tn, tf = n + HS[i] - 1, f + 4
            # O3
            e = Expr(tn, tf)
            hx = hmul(i, n, f, x)
            if basis(n + HS[i], f + 1):
                e.addexpr(D3(const_vec(n + HS[i], f + 1, hx)))
            e.addexpr(hmul_expr(i, D3_basis(n, f, j)))
            e.addexpr(d2_expr(mu1_basis(i, n, f, j)))
            d2x = d2v(n, f, x)
            if d2x and basis(n - 1, f + 2):
                bt = basis(n - 1, f + 2)
                for t in bits(bt.coords(d2x)):
                    e.addexpr(mu1_basis(i, n - 1, f + 2, t))
            emit(e, ('O3', i, n, f, j))
            # O4 on representatives
            if USE_O4 and b.kind[j] == 'R' and easy[(n, f)][x].get(i) is not None:
                e = Expr(tn, tf + 1)
                if basis(n + HS[i], f + 1):
                    hb = basis(n + HS[i], f + 1)
                    for t in bits(hb.coords(hx)):
                        e.addexpr(D4_basis(n + HS[i], f + 1, t))
                e.addexpr(hmul_expr(i, D4_basis(n, f, j)))
                L = easy[(n, f)][x][i]
                if L and basis(n + HS[i], f + 2):
                    e.addexpr(D3(const_vec(n + HS[i], f + 2, L)))
                # mu1(h_i, D3 x): D3 x = sum_k a[k] R_k at (n - 1, f + 3)
                bt = basis(n - 1, f + 3)
                if bt and (n, f) not in [(s, 1) for s in HS]:
                    for k in range(bt.nR):
                        a = var(('a', n, f, j, k))
                        m = mu1_basis(i, n - 1, f + 3, k)
                        if m.c:
                            skip = True  # mu1 unknown on this representative: bilinear
                            break
                        for t in bits(m.k):
                            e.c[t] ^= 1 << a
                    else:
                        skip = False
                    if skip:
                        continue
                e.addexpr(d2_expr(mu2_basis(i, n, f, j)))
                emit(e, ('O4', i, n, f, j))

for (n, f) in sorted(dim):
    if n > NMAX or not Dm(n, f):
        continue
    b = basis(n, f)
    for j in range(b.d):
        if b.kind[j] != 'N':
            continue
        e = Expr(n - 2, f + 5)
        dz = d2v(n, f, b.vecs[j])
        if basis(n - 1, f + 2):
            e.addexpr(D3(const_vec(n - 1, f + 2, dz)))
        e.addexpr(d2_expr(D3_basis(n, f, j)))
        emit(e, ('DD', n, f, j))
print(f"{nv} variables, {len(EQ)} equations", file=sys.stderr)

# --- elimination over GF(2): rows as (mask, const)
piv = {}
contra = []
for m, c, lab in EQ:
    while m:
        p = (m & -m).bit_length() - 1
        if p in piv:
            pm, pc, _ = piv[p]
            m ^= pm
            c ^= pc
        else:
            piv[p] = (m, c, lab)
            break
    else:
        if c:
            contra.append(lab)
print(f"rank {len(piv)}, contradictions {len(contra)} {contra[:5]}", file=sys.stderr)
# back-substitute to reduced form for the query variables
for p in sorted(piv, reverse=True):
    m, c, lab = piv[p]
    # rows of higher pivots are already reduced: they hold their pivot and free variables only
    for q in [q for q in bits(m >> (p + 1)) if q + p + 1 in piv]:
        qm, qc, _ = piv[q + p + 1]
        m ^= qm
        c ^= qc
    piv[p] = (m, c, lab)


def value(v):
    """(determined?, value) of variable v"""
    if v not in piv:
        return (False, None)
    m, c, _ = piv[v]
    if m & ~(1 << v):
        return (False, None)
    return (True, c)


q = [(k, v) for k, v in VAR.items() if k[0] == 'a']
det = sum(1 for _, v in q if value(v)[0])
nonzero = sum(1 for _, v in q if value(v) == (True, 1))
print(f"d3 coordinates (R -> R): {len(q)}, determined {det}, determined nonzero {nonzero}", file=sys.stderr)

if HOLD:
    ok = wrong = opn = 0
    for r in csv.reader(open(HOLD)):
        if r[0] == 'stem' or int(r[0]) > NMAX:
            continue
        n, f = int(r[0]), int(r[1])
        b, bt = basis(n, f), basis(n - 1, f + 3)
        if not b:
            continue
        xc = b.coords(mask(r[2]))
        want = bt.coords(mask(r[3])) & ((1 << bt.nR) - 1) if bt else 0
        got, known = 0, True
        for j in bits(xc):
            if b.kind[j] == 'N':
                known = False  # not a cycle
                break
            if b.kind[j] == 'B' or not bt:
                continue
            for k in range(bt.nR):
                key = ('a', n, f, j, k)
                if key not in VAR:
                    known = False
                    break
                dv = value(VAR[key])
                if not dv[0]:
                    known = False
                    break
                got ^= dv[1] << k
        if not known:
            opn += 1
        elif got == want:
            ok += 1
            print(f"FORCED d3 ({n},{f}) {r[4]}", file=sys.stderr)
        else:
            wrong += 1
            print(f"WRONG d3 at ({n},{f}) {r[4]}: forced {got:b}, known {want:b}", file=sys.stderr)
    print(f"known d3: {ok} correct and forced, {wrong} wrong, {opn} open", file=sys.stderr)
