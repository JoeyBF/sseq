"""Module d2 files (`stem,Adams_filtration,index,target` in the cofiber_slots basis) from module_d2's
cochain blocks, for a two-cell complex C_j = S u_{h_j} e^k on the minimal resolution P.

On P, Hom(P (x) H*C_j) = Hom(P).bottom + Hom(P).top with coboundary y.top -> (h_j y).bottom, and the
slots basis is: coker classes = i_*(sphere classes), lifts = the cocycles y_k.top with y_k = p_*(lift).
A block (c, b) at sphere (n, s, x) is the b-cell part of d2 of the cochain x.c over sphere generators
at (n + c - b - 1, s + 2). The d2 cochain of (u, v) is (u B00 + v B10, v B11); its class is
i_*(bottom) + sum of lifts whose y_k sum to the top part (which must lie in their span).

With a product file (cone blocks), the lift's cochain y.top also carries the Z/4 carry of y o phi:
c_g = (sum_i y_i [h_k e_i]_g) / 2, adding h_0 c to the bottom part.

usage: to_d2.py <slots dir> <name> <k> <blocks csv> <out csv> <max stem> <max f> [h products csv]
"""
import csv
import sys
from collections import defaultdict

slots, name, k, blocks_path, out_path = sys.argv[1:6]
k = int(k)
max_n, max_f = int(sys.argv[6]), int(sys.argv[7])
# (stem_1 of h) -> sphere (n, f) of the other factor -> index -> product mask at (n + stem_1, f + 1)
hprod = defaultdict(lambda: defaultdict(dict))
if len(sys.argv) > 8:
    with open(sys.argv[8]) as fh:
        for r in csv.reader(fh):
            if r[0].startswith('stem'):
                continue
            hprod[int(r[0])][(int(r[3]), int(r[4]))][int(r[5])] = int(
                sum(int(x) << i for i, x in enumerate(r[6].strip('"[]').split(',')) if x.strip()))


def vec(s):
    s = s.strip().strip('"').strip('[]')
    return [int(t) for t in s.split(',')] if s.strip() else []


def mask(v):
    m = 0
    for i, e in enumerate(v):
        if e & 1:
            m |= 1 << i
    return m


rank = {}
with open(f"{slots}/{name}_rank.csv") as fh:
    for r in csv.DictReader(fh):
        rank[(int(r['stem']), int(r['Adams_filtration']))] = int(r['rank'])
istar = defaultdict(dict)  # sphere (n,f) -> gen -> module mask at (n,f)
with open(f"{slots}/{name}_istar.csv") as fh:
    for r in csv.reader(fh):
        if r[0] == 'stem':
            continue
        istar[(int(r[0]), int(r[1]))][int(r[2])] = mask(vec(r[3]))
pstar = defaultdict(dict)  # module (N,f) -> module index -> sphere mask at (N-k,f)
with open(f"{slots}/{name}_pstar.csv") as fh:
    for r in csv.reader(fh):
        if r[0] == 'stem':
            continue
        pstar[(int(r[0]), int(r[1]))][int(r[2])] = mask(vec(r[3]))

# blocks[(c,b)][(n,s)][x] = mask over sphere gens at (n+c-b-1, s+2)
blocks = defaultdict(lambda: defaultdict(dict))
with open(blocks_path) as fh:
    for r in csv.reader(fh):
        if r[0] == 'module':
            continue
        n, s, x, cd, bd = int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[6])
        blocks[(cd, bd)][(n, s)][x] = mask(vec(r[8]))


def apply(cb, n, s, u):
    out = 0
    tab = blocks[cb].get((n, s), {})
    i = 0
    while u:
        if u & 1:
            out ^= tab.get(i, 0)
        u >>= 1
        i += 1
    return out


def solve(rows, target):
    """Coefficients c with xor of rows[i] for c_i = 1 equal to target, or None."""
    piv = []  # (pivot bit, row mask, combination mask)
    for i, r in enumerate(rows):
        c = 1 << i
        for p, pr, pc in piv:
            if r >> p & 1:
                r ^= pr
                c ^= pc
        if r:
            piv.append(((r & -r).bit_length() - 1, r, c))
    c = 0
    for p, pr, pc in piv:
        if target >> p & 1:
            target ^= pr
            c ^= pc
    return c if target == 0 else None


def istar_of(nf, u):
    out = 0
    tab = istar.get(nf, {})
    i = 0
    while u:
        if u & 1:
            out ^= tab.get(i, 0)
        u >>= 1
        i += 1
    return out


bad = 0
carries = 0
written = 0
nonzero = 0
with open(out_path, 'w') as out:
    out.write("stem,Adams_filtration,index,target\n")
    for (N, f), dim in sorted(rank.items()):
        if N > max_n or f > max_f or dim == 0:
            continue
        tgt = (N - 1, f + 2)
        tdim = rank.get(tgt, 0)
        lifts = pstar.get((N, f), {})
        # sphere gens at (N,f) whose istar images span the coker part
        gens = sorted(istar.get((N, f), {}).items())
        tlifts = sorted(pstar.get(tgt, {}).items())
        for i in range(dim):
            if i in lifts:
                u, v = 0, lifts[i]
            else:
                c = solve([m for _, m in gens], 1 << i)
                if c is None:
                    bad += 1
                    print(f"no i_* preimage for ({N},{f}) #{i}", file=sys.stderr)
                    continue
                u = 0
                for j, (g, _) in enumerate(gens):
                    if c >> j & 1:
                        u |= 1 << g
                v = 0
            bottom = apply((0, 0), N, f, u) ^ apply((k, 0), N - k, f, v)
            if v and len(sys.argv) > 8:
                counts = defaultdict(int)
                tab = hprod[k - 1].get((N - k, f), {})
                for yi in range(v.bit_length()):
                    if v >> yi & 1:
                        m = tab.get(yi, 0)
                        for g in range(m.bit_length()):
                            if m >> g & 1:
                                counts[g] += 1
                assert all(c % 2 == 0 for c in counts.values()), "y is not a cocycle"
                w = 0
                for g, c in counts.items():
                    if c % 4 == 2:
                        w |= 1 << g
                h0 = hprod[0].get((N - 1, f + 1), {})
                carries += w != 0
                for g in range(w.bit_length()):
                    if w >> g & 1:
                        bottom ^= h0.get(g, 0)
            top = apply((k, k), N - k, f, v)
            t = istar_of(tgt, bottom)
            if top:
                c = solve([m for _, m in tlifts], top)
                if c is None:
                    bad += 1
                    print(f"top part outside the lifts at ({N},{f}) #{i}", file=sys.stderr)
                    continue
                for j, (idx, _) in enumerate(tlifts):
                    if c >> j & 1:
                        t ^= 1 << idx
            if t >> tdim:
                bad += 1
                print(f"target out of range at ({N},{f}) #{i}", file=sys.stderr)
                continue
            bits = [t >> l & 1 for l in range(tdim)]
            out.write(f'{N},{f},{i},"[{", ".join(map(str, bits))}]"\n')
            written += 1
            nonzero += t != 0
print(f"{name}: {written} classes written, {nonzero} nonzero d2, {bad} failures, {carries} carries",
      file=sys.stderr)
