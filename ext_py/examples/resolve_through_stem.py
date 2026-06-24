#!/usr/bin/env python3
"""Resolve a module up to an (n, s) and print an ASCII depiction of Ext.

Python port of ext/examples/resolve_through_stem.rs.
"""

import _query as query
from ext_py import sseq_py


def main():
    res = query.query_module_only("Module")

    max_bidegree = sseq_py.Bidegree.n_s(
        query.with_default("Max n", "30", int),
        query.with_default("Max s", "15", int),
    )

    res.compute_through_stem(max_bidegree)

    print(res.graded_dimension_string())


if __name__ == "__main__":
    main()
