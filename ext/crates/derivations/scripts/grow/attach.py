"""Cone off a filtration-1 map S^{t-1} -> X: attach one cell to the finite A-module H*X (p = 2).

Ext^{1,t}_A(H*X, F2) = extensions 0 -> F2 e_t -> M' -> H*X -> 0. Unknowns c(k,x) = e-coefficient of
Sq^k x (deg x + k = t); constraints = Adem relations evaluated on M' (Sq^a e = 0 for a > 0); modulo
base changes x -> x + eps e (deg x = t). Prints dim Ext^{1,t}; writes one module json per class.

usage: attach.py <module.json> <t> [outdir] [--all]
"""
import itertools
import json
import os
import re
import sys



def load(path):
    d = json.load(open(path))
    act = {}
    for a in d['actions']:
        m = re.match(r'\s*Sq(\d+)\s+(\S+)\s*=\s*(.*)$', a)
        k, x, rhs = int(m.group(1)), m.group(2), m.group(3)
        act[(k, x)] = {s.strip() for s in rhs.split('+') if s.strip() not in ('', '0')}
    return d['gens'], act


def sq(act, k, v):
    if k == 0:
        return set(v)
    out = set()
    for x in v:
        out ^= act.get((k, x), set())
    return out


def adem(a, b):
    """Sq^a Sq^b (0 < a < 2b) = sum_c binom(b-c-1, a-2c) Sq^{a+b-c} Sq^c."""
    return [(a + b - c, c) for c in range(a // 2 + 1) if (b - c - 1) & (a - 2 * c) == a - 2 * c]


def ext1(gens, act, t):
    names = sorted(gens, key=lambda x: (gens[x], x))
    var = [(t - gens[x], x) for x in names if t - gens[x] >= 1]
    vi = {v: i for i, v in enumerate(var)}

    def ecoef(k, v):
        r = 0
        if k:
            for x in v:
                if (k, x) in vi:
                    r ^= 1 << vi[(k, x)]
        return r

    eqs = []
    for x in names:
        d = t - gens[x]
        for a in range(1, d):
            b = d - a
            if a < 2 * b:
                r = ecoef(a, sq(act, b, {x}))
                for p, q in adem(a, b):
                    r ^= ecoef(p, sq(act, q, {x}))
                if r:
                    eqs.append(r)
    cob = []
    for y in names:
        if gens[y] == t:
            cob.append(sum(1 << vi[(t - gens[x], x)] for x in names
                           if t - gens[x] >= 1 and y in act.get((t - gens[x], x), set())))
    return var, eqs, cob


def rref(rows):
    piv = {}
    for r in rows:
        for p in sorted(piv, reverse=True):
            if r >> p & 1:
                r ^= piv[p]
        if r:
            p = r.bit_length() - 1
            for q in piv:
                if piv[q] >> p & 1:
                    piv[q] ^= r
            piv[p] = r
    return piv


def kernel(eqs, n):
    piv = rref(eqs)
    out = []
    for f in range(n):
        if f not in piv:
            v = 1 << f
            for p, pr in piv.items():
                if pr >> f & 1:
                    v |= 1 << p
            out.append(v)
    return out


def quotient(ker, cob):
    piv = rref(cob)
    reps = []
    for v in ker:
        w = v
        for p in sorted(piv, reverse=True):
            if w >> p & 1:
                w ^= piv[p]
        if w:
            piv = rref(list(piv.values()) + [w])
            reps.append(v)
    return reps


def module_json(gens, act, var, cls, t, e):
    g = dict(gens)
    g[e] = t
    acts = {k: set(v) for k, v in act.items()}
    for i, kx in enumerate(var):
        if cls >> i & 1:
            acts.setdefault(kx, set()).symmetric_difference_update({e})
    key = lambda kv: (g[kv[0][1]], kv[0][0], kv[0][1])
    lines = [f"Sq{k} {x} = {' + '.join(sorted(v, key=lambda y: (g[y], y)))}"
             for (k, x), v in sorted(acts.items(), key=key) if v]
    return {"p": 2, "type": "finite dimensional module", "gens": g, "actions": lines}


def chi_sq(act, n, m):
    """chi(Sq^n) m, from sum_{i=0}^n Sq^i chi(Sq^{n-i}) = 0 (n > 0)."""
    memo = [{m}]
    for k in range(1, n + 1):
        v = set()
        for i in range(1, k + 1):
            v ^= sq(act, i, memo[k - i])
        memo.append(v)
    return memo[n]


def cocycle(gens, act, var, cls, t):
    """The class on P_1 (x) M of the minimal sphere resolution P: phi(g_k (x) m) for
    |g_k| + |m| = t, phi(g_k (x) m) = sum_{i < 2^k} c(2^k - i, chi(Sq^i) m)."""
    c = {kx for i, kx in enumerate(var) if cls >> i & 1}
    phi = set()
    for m in gens:
        n = t - gens[m]
        if n >= 1 and n & (n - 1) == 0:
            bit = 0
            for i in range(n):
                for y in chi_sq(act, i, m):
                    bit ^= (n - i, y) in c
            if bit:
                phi.add((n.bit_length() - 1, m))
    return phi


def main():
    path, t = sys.argv[1], int(sys.argv[2])
    out = next((a for a in sys.argv[3:] if not a.startswith('--')), None)
    gens, act = load(path)
    var, eqs, cob = ext1(gens, act, t)
    ker = kernel(eqs, len(var))
    reps = quotient(ker, cob)
    print(f"Ext^1,{t}: dim {len(reps)}")
    classes = reps
    if '--all' in sys.argv:
        classes = [sum_xor(r for b, r in zip(bits, reps) if b)
                   for bits in itertools.product([0, 1], repeat=len(reps)) if any(bits)]
    if '--cocycle' in sys.argv:
        # the extension whose class is the given h_k (x) m terms ("k:m,k:m")
        want = {(int(s.split(':')[0]), s.split(':')[1])
                for s in sys.argv[sys.argv.index('--cocycle') + 1].split(',')}
        keys = sorted({kx for v in ker for kx in cocycle(gens, act, var, v, t)} | want)
        col = lambda phi: sum(1 << keys.index(kx) for kx in phi)
        piv = []
        for v in ker:
            r, c = col(cocycle(gens, act, var, v, t)), v
            for p, pr, pc in piv:
                if r >> p & 1:
                    r, c = r ^ pr, c ^ pc
            if r:
                piv.append((r.bit_length() - 1, r, c))
        r, c = col(want), 0
        for p, pr, pc in piv:
            if r >> p & 1:
                r, c = r ^ pr, c ^ pc
        if r:
            sys.exit(f"no extension has class {sorted(want)}")
        classes = [c]
    for ci, cls in enumerate(classes):
        phi = sorted(cocycle(gens, act, var, cls, t))
        print(f"  class {ci}: e{t} = " + " + ".join(f"Sq{k} {x}" for i, (k, x) in enumerate(var)
                                                    if cls >> i & 1)
              + "   [" + " + ".join(f"h{k}.{m}" for k, m in phi) + "]")
        if out:
            os.makedirs(out, exist_ok=True)
            with open(f"{out}/c{ci}.json", "w") as fh:
                json.dump(module_json(gens, act, var, cls, t, f"e{t}"), fh, indent=1)


def sum_xor(it):
    v = 0
    for r in it:
        v ^= r
    return v


if __name__ == '__main__':
    main()
