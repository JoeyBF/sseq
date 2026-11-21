"""Compare restrict_unit's hA table (from the loaded A-resolution) with the sphere ring export's h_i
products: equal tables mean the two share the A-basis. usage: compare_hA.py <hA.csv> <sphere h rows>"""
import csv, sys

vec = lambda s: [int(x) for x in s.strip().strip('"').strip('[]').split(',') if x.strip() != '']
ours = {}
for r in csv.reader(open(sys.argv[1])):
    if r[0] == 'i':
        continue
    i, n, s, k = map(int, r[:4])
    ours[(i, n, s, k)] = vec(r[4])
sph = {}
for r in csv.reader(open(sys.argv[2])):
    i = {0: 0, 1: 1, 3: 2, 7: 3}[int(r[0])]
    sph[(i, int(r[3]), int(r[4]), int(r[5]))] = vec(r[6])
ok = bad = 0
for key, v in ours.items():
    w = sph.get(key, [0] * len(v))
    if v == w:
        ok += 1
    else:
        bad += 1
        if bad < 10:
            print('DIFF', key, v, w)
missing = [k for k in sph if k not in ours]
print(f'{ok} agree, {bad} differ; {len(missing)} sphere rows outside our table')
