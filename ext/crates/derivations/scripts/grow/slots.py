"""Reading `module_slots` output: Ext(X) bases as cochains, coordinates of any cocycle, and the
maps induced by inclusions/projections of cells.

A cochain is a set of (cell, generator) pairs; generator = index among the sphere's generators in
bidegree (n - |cell|, s), |cell| relative to the bottom cell.
"""
import csv
from collections import defaultdict


def parse(s):
    return frozenset((c, int(g)) for c, g in (t.split(':') for t in s.split(';') if t))


class Slots:
    def __init__(self, d, name):
        self.cells = {r['cell']: (int(r['index']), int(r['degree']))
                      for r in csv.DictReader(open(f"{d}/{name}_cells.csv"))}
        self.order = lambda cg: (self.cells[cg[0]][0], cg[1])
        self.basis = defaultdict(list)
        self.bound = defaultdict(list)
        for kind, tab in (('basis', self.basis), ('bound', self.bound)):
            for r in csv.DictReader(open(f"{d}/{name}_{kind}.csv")):
                tab[(int(r['stem']), int(r['Adams_filtration']))].append(parse(r['coords']))

    def coords(self, b, v):
        """Coordinates in the basis at b of the cocycle v (a set of pairs). Raises if v is not in
        span(basis) + span(bound)."""
        v = set(v)
        for r in self.bound[b]:
            p = min(r, key=self.order)
            if p in v:
                v ^= r
        out = 0
        for i, r in enumerate(self.basis[b]):
            p = min(r, key=self.order)
            if p in v:
                v ^= r
                out |= 1 << i
        if v:
            raise ValueError(f"not a cocycle mod boundaries at {b}: {sorted(v)}")
        return out
