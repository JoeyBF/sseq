"""The cofiber path weight W_j as a cobar 2-cochain over the dual Steenrod algebra A_* (p = 2).

Monomials xi^R are dual to Milnor P(R).  A path weight W(a, b) on Milnor basis pairs is the
cochain sum W(P(R), P(S)) [xi^R | xi^S].  Only its class mod delta C^1 matters on relations.
Prints delta W, and whether W (or W + [xi1 | xi1^{2^j}]) is a coboundary.
"""
import itertools
import sys
from collections import defaultdict


def deg(R):
    return sum(r * ((1 << (i + 1)) - 1) for i, r in enumerate(R))


def trim(R):
    R = list(R)
    while R and R[-1] == 0:
        R.pop()
    return tuple(R)


def monos(d):
    """All R with deg(R) = d."""
    out = []

    def rec(i, rem, cur):
        w = (1 << (i + 1)) - 1
        if rem == 0:
            out.append(trim(cur))
            return
        if w > rem:
            return
        for r in range(rem // w, -1, -1):
            rec(i + 1, rem - r * w, cur + [r])

    rec(0, d, [])
    return out


def mul(x, y):  # polynomials: dict mono->1 over F2 ; mono = tuple R
    out = defaultdict(int)
    for a in x:
        for b in y:
            n = max(len(a), len(b))
            m = trim(tuple((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)))
            out[m] ^= 1
    return {k for k, v in out.items() if v}


def tmul(x, y):  # tensors: set of (mono, mono)
    out = defaultdict(int)
    for (a1, a2) in x:
        for (b1, b2) in y:
            out[(next(iter(mul({a1}, {b1}))), next(iter(mul({a2}, {b2}))))] ^= 1
    return {k for k, v in out.items() if v}


def xi(n, power=1):
    if n == 0:
        return ()
    R = [0] * n
    R[n - 1] = power
    return tuple(R)


_cop = {}


def coproduct(R):
    """Delta(xi^R) as a set of (mono, mono)."""
    if R in _cop:
        return _cop[R]
    res = {((), ())}
    for n, r in enumerate(R, start=1):
        # Delta xi_n = sum_i xi_{n-i}^{2^i} (x) xi_i
        dxi = {(xi(n - i, 1 << i) if n - i > 0 else (), xi(i)) for i in range(n + 1)}
        for _ in range(r):
            res = tmul(res, dxi)
    _cop[R] = res
    return res


def red(R):
    return {(a, b) for (a, b) in coproduct(R) if a != () and b != ()}


def delta2(W):
    """cobar differential of a 2-cochain (set of (a, b))."""
    out = defaultdict(int)
    for (a, b) in W:
        for (a1, a2) in red(a):
            out[(a1, a2, b)] ^= 1
        for (b1, b2) in red(b):
            out[(a, b1, b2)] ^= 1
    return {k for k, v in out.items() if v}


def is_coboundary(W, d):
    """Is W in the span of red(R), deg R = d?  Gaussian elimination over F2."""
    basis = {}
    cols = {}

    def key(t):
        if t not in cols:
            cols[t] = len(cols)
        return cols[t]

    def vec(S):
        v = 0
        for t in S:
            v ^= 1 << key(t)
        return v

    rows = []
    for R in monos(d):
        v = vec(red(R))
        for p, (r, _) in sorted(basis.items(), reverse=True):
            pass
        rows.append((v, R))
    piv = {}
    for v, R in rows:
        comb = {R}
        while v:
            p = v.bit_length() - 1
            if p in piv:
                pv, pc = piv[p]
                v ^= pv
                comb ^= pc
            else:
                piv[p] = (v, comb)
                break
    w = vec(W)
    comb = set()
    while w:
        p = w.bit_length() - 1
        if p not in piv:
            return None
        w ^= piv[p][0]
        comb ^= piv[p][1]
    return comb


def show(S):
    def m(R):
        return "*".join(f"x{i+1}^{r}" if r > 1 else f"x{i+1}" for i, r in enumerate(R) if r) or "1"
    return " + ".join("[" + "|".join(m(x) for x in t) + "]" for t in sorted(S))


def W_of(j):
    top = 1 << j
    W = set()
    if j >= 2:
        h = 1 << (j - 1)
        W ^= {(xi(1, h), xi(1, h + 1)), (xi(1, h + 1), xi(1, h))}
    k = 0
    while (1 << (k + 1)) - 1 <= top + 1:
        dR = (1 << (j - 1)) + 1 - (1 << k) if j >= 1 else None
        if dR is not None and dR >= 1:
            for R in monos(dR):
                a = tuple(2 * r for r in R)
                if a == xi(1, top):
                    continue
                W ^= {(a, xi(k + 1))}
        k += 1
    return W


if __name__ == "__main__":
    for j in range(0, int(sys.argv[1]) + 1 if len(sys.argv) > 1 else 5):
        top = 1 << j
        W = W_of(j)
        print(f"j={j}: W = {show(W)}")
        dW = delta2(W)
        print(f"  delta W = {show(dW) or 0}")
        hh = {(xi(1), xi(1, top))}
        for name, X in [("W", W), ("W + h0hj", W ^ hh)]:
            c = is_coboundary(X, top + 1)
            print(f"  {name} coboundary: {'no' if c is None else 'delta of ' + show({(r,) for r in c}) if c else 'zero'}")
