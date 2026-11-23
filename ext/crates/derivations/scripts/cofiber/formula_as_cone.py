"""Writes the fitted cross-term formula in cofiber_cone's CSV format (j,stem,Adams_filtration,
index,E,L), so assemble_easy.py can use it for the non-cycle lifts (mod i_*(B2)).

usage: formula_as_cone.py <features csv> <out csv>
"""
import re
import sys
from collections import defaultdict

FORMULA = {0: ["H:P(1)"], 1: ["H:P(2)"], 2: ["H:P(4)", "d:P(2)|P(0, 1)", "d:P(2)|P(3)", "d:P(3)|P(2)"]}
pat = re.compile(r'^(\d+),(\d+),(\d+),(\d+),(.*),"(\[.*\])"$')
K = defaultdict(lambda: None)
width = {}
seen = set()
for line in open(sys.argv[1]):
    m = pat.match(line.strip())
    if not m:
        continue
    j, n, s, x = (int(m.group(i)) for i in range(1, 5))
    seen.add((j, n, s, x))
    if m.group(5) not in FORMULA.get(j, []):
        continue
    v = [int(c) for c in m.group(6).strip("[]").split(",") if c.strip()]
    width[(j, n, s)] = len(v)
    old = K[(j, n, s, x)]
    K[(j, n, s, x)] = v if old is None else [a ^ b for a, b in zip(old, v)]
with open(sys.argv[2], "w") as f:
    f.write("j,stem,Adams_filtration,index,E,L\n")
    for (j, n, s, x), v in K.items():
        if v is not None:
            f.write(f'{j},{n},{s},{x},"[]","{v}"\n')
