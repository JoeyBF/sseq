"""Solve delta W = O_j over the cobar complex from scratch (no fitting) and print the solution
as eval_formula operators.  O_j = [xi1 | xi1^h | xi1^h], h = 2^(j-1), for j >= 2; 0 otherwise.
usage: solveW.py <j> [seed]"""
import random
import sys
from cobar import monos, delta2, xi, show

j = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
top = 1 << j
O = {(xi(1), xi(1, top >> 1), xi(1, top >> 1))} if j >= 2 else set()
basis = [(a, b) for da in range(1, top + 1) for a in monos(da) for b in monos(top + 1 - da)]
random.Random(seed).shuffle(basis)
cols = {}
def vec(S):
    v = 0
    for t in S:
        v ^= 1 << cols.setdefault(t, len(cols))
    return v
piv = {}
for i, t in enumerate(basis):
    v, comb = vec(delta2({t})), 1 << i
    while v:
        p = v.bit_length() - 1
        if p in piv:
            v ^= piv[p][0]; comb ^= piv[p][1]
        else:
            piv[p] = (v, comb); break
w, comb = vec(O), 0
while w:
    p = w.bit_length() - 1
    assert p in piv, "no solution"
    w ^= piv[p][0]; comb ^= piv[p][1]
W = {basis[i] for i in range(len(basis)) if comb >> i & 1}
assert delta2(W) == O
P = lambda R: "P(" + ", ".join(map(str, R)) + ")"
print(f"{j}:H:P({top})" + "".join(f"+d:{P(a)}|{P(b)}" for a, b in sorted(W)))
