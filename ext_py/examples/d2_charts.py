#!/usr/bin/env python3
"""Write E2/E3 page charts (with and without d2) of a resolution to TikZ files.

Python port of ext/examples/d2_charts.rs.
"""

import _query as query
import ext_py
from ext_py import algebra_py, sseq_py


def main():
    resolution = query.query_module(algebra_py.AlgebraType.Milnor)

    lift = ext_py.SecondaryResolution(resolution)
    lift.extend_all()

    sseq = lift.e3_page()
    products = [
        (name, resolution.filtration_one_products(op_deg, op_idx))
        for (name, op_deg, op_idx) in resolution.algebra().default_filtration_one_products()
    ]

    def write(path, page, diff, prod):
        # NOTE: depends on TikzBackend.EXT and Resolution.name() (API_PROPOSAL §6.3, §7.4).
        ext = sseq_py.TikzBackend.EXT
        backend = sseq_py.TikzBackend(
            open(f"{path}_{resolution.name()}.{ext}", "w")
        )
        sseq.write_to_graph(backend, page, diff, products[:prod], lambda _: None)

    write("e2", 2, False, 3)
    write("e2_d2", 2, True, 3)
    write("e3", 3, False, 3)

    write("e2_clean", 2, False, 2)
    write("e2_d2_clean", 2, True, 2)
    write("e3_clean", 3, False, 2)


if __name__ == "__main__":
    main()
