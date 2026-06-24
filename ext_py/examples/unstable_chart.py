#!/usr/bin/env python3
"""Compute and chart the suspension maps between unstable Ext groups.

Given an unstable Steenrod module M, compute the unstable Ext groups of the
suspensions of M for all shifts up to the stable range, writing a TikZ figure
for each shift to stdout.

Python port of ext/examples/unstable_chart.rs.
"""

import os
import sys

import _query as query
import ext_py
from ext_py import algebra_py, sseq_py


def query_unstable_module_only():
    """Inline mirror of ext::utils::query_unstable_module_only.

    Queries a single "Module" spec, parses the optional ``@adem``/``@milnor``
    algebra suffix (default Milnor) and the module name, builds the algebra with
    ``unstable=True`` and returns the corresponding Steenrod module.
    """

    def parse_spec(spec):
        # Mirror Config::try_from(&str): split on '@' for the algebra type.
        module_name, _, algebra_name = spec.partition("@")
        if algebra_name == "":
            algebra_type = algebra_py.AlgebraType.Milnor
        elif algebra_name == "adem":
            algebra_type = algebra_py.AlgebraType.Adem
        elif algebra_name == "milnor":
            algebra_type = algebra_py.AlgebraType.Milnor
        else:
            raise ValueError(f"Invalid algebra type: {algebra_name}")
        # NOTE: depends on ext_py.parse_module_name (API_PROPOSAL §7.4).
        module = ext_py.parse_module_name(module_name)
        return (module, algebra_type)

    module_json, algebra_type = query.raw("Module", parse_spec)
    algebra = algebra_py.SteenrodAlgebra.from_json(module_json, algebra_type, True)
    return algebra_py.steenrod_module_from_json(algebra, module_json)


def main():
    module = query_unstable_module_only()

    # Mirror the `save_dir` closure: an optional base directory under which each
    # shift gets its own `suspension{shift}` subdirectory.
    base = query.optional("Module save directory", str)

    def save_dir(shift):
        if base is None:
            return None
        return os.path.join(base, f"suspension{shift}")

    max = sseq_py.Bidegree.n_s(
        query.raw("Max n", int),
        query.raw("Max s", int),
    )

    disp_template = query.raw(
        "LaTeX name template (replace % with min degree)",
        str,
    )

    products = module.algebra().default_filtration_one_products()

    for shift_t in range(0, max.n() - module.min_degree() + 3):
        shift = sseq_py.Bidegree.s_t(0, shift_t)
        # NOTE: depends on ext_py.SuspensionModule, ext_py.ChainComplex.ccdz and
        # ext_py.UnstableResolution.new_with_save (API_PROPOSAL §7.1, §7.3, §7.5).
        res = ext_py.UnstableResolution.new_with_save(
            ext_py.ChainComplex.ccdz(
                algebra_py.SuspensionModule(module, shift.t())
            ),
            save_dir(shift.t()),
        )

        res.compute_through_stem(max + shift)

        print("\\begin{figure}[p]\\centering")

        sseq = res.to_sseq()
        shift_products = [
            (name, res.filtration_one_products(op_deg, op_idx))
            for (name, op_deg, op_idx) in products
        ]

        def header(g, shift_t=shift_t):
            return g.text(
                sseq_py.Bidegree.x_y(1, max.s() - 1),
                disp_template.replace("%", f"{shift_t}"),
                sseq_py.Orientation.Right,
            )

        sseq.write_to_graph(
            sseq_py.TikzBackend(sys.stdout),
            2,
            False,
            shift_products,
            header,
        )

        print("\\end{figure}")


if __name__ == "__main__":
    main()
