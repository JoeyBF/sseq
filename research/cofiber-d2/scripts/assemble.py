"""Builds module_d2-format blocks from a feature dump: the (k, 0) block is the sum of the listed
features, the diagonal blocks are copied from a module_d2 output.

usage: assemble.py <name> <k> <features csv> <module_d2 csv (diagonal blocks)> <out csv> <feature>...
"""
import sys

name, k, feat_p, md_p, out_p = sys.argv[1:6]
want = set(sys.argv[6:])
rows = {}
for line in open(feat_p):
    if line.startswith('feature,'):
        continue
    head, vec = line.rstrip('\n').split('"', 1)
    fname, n, s, y, _ = head.rsplit(',', 4)
    if fname not in want:
        continue
    v = [int(x) for x in vec.strip('"[] ').split(',') if x.strip()]
    key = (int(n), int(s), int(y))
    rows[key] = [a ^ b for a, b in zip(rows[key], v)] if key in rows else v
with open(out_p, 'w') as out:
    for i, line in enumerate(open(md_p)):
        f = line.split(',')
        if i == 0 or f[4] == f[6]:
            out.write(line)
    for (n, s, y), v in sorted(rows.items()):
        if any(v):
            out.write(f'{name},{n},{s},{y},{k},0,0,0,"{v}"\n')
missing = want - {l.split('"')[0].rsplit(',', 4)[0] for l in open(feat_p) if not l.startswith('feature,')}
print(f"{name}: {len(rows)} rows; features never nonzero: {sorted(missing)}", file=sys.stderr)
