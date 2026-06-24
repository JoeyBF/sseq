#!/usr/bin/env python3
"""Compute the suspension map between different unstable Ext groups.

Given an unstable Steenrod module M, compute the unstable Ext groups of the
suspensions of M for all shifts up to the stable range. Each result is printed
in the form ``n s k: num_gens - matrix``.

Python port of ext/examples/unstable_suspension.rs.
"""

import os

import _query as query
import ext_py
from ext_py import algebra_py, fp_py, sseq_py


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
        # NOTE: depends on ext_py.parse_module_name (API_PROPOSAL §7.7).
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
    min_degree = sseq_py.Bidegree.s_t(0, module.min_degree())

    # NOTE: depends on ext_py.SuspensionModule, ext_py.ChainComplex.ccdz and
    # ext_py.UnstableResolution.new_with_save (API_PROPOSAL §5.3, §7.1, §7.2).
    res_b = ext_py.UnstableResolution.new_with_save(
        ext_py.ChainComplex.ccdz(algebra_py.SuspensionModule(module, 0)),
        save_dir(0),
    )
    res_b.compute_through_stem(max)

    for n in range(min_degree.n(), max.n() + 1):
        for s in range(0, max.s() + 1):
            b = sseq_py.Bidegree.n_s(n, s)
            source_num_gens = res_b.number_of_gens_in_bidegree(b)
            print(f"{n} {s} 0: {source_num_gens}")

    for shift_t in range(1, (max - min_degree).n() + 3):
        shift = sseq_py.Bidegree.s_t(0, shift_t)
        res_a = res_b
        res_b = ext_py.UnstableResolution.new_with_save(
            ext_py.ChainComplex.ccdz(algebra_py.SuspensionModule(module, shift_t)),
            save_dir(shift_t),
        )

        res_b.compute_through_stem(max + shift)

        suspension_shift = sseq_py.Bidegree.s_t(0, 1)
        # NOTE: depends on ext_py.UnstableResolutionHomomorphism
        # (API_PROPOSAL §7.3).
        hom = ext_py.UnstableResolutionHomomorphism(
            "suspension",
            res_b,
            res_a,
            suspension_shift,
        )

        # NOTE: depends on UnstableResolutionHomomorphism.extend_step_raw
        # (API_PROPOSAL §7.3 lists extend_step; sq0.py relies on the same hook).
        hom.extend_step_raw(
            min_degree + shift,
            [fp_py.FpVector.from_slice(module.prime(), [1])],
        )
        hom.extend_all()

        for n in range(2 * ((min_degree + shift).n() - 1), (max + shift).n() + 1):
            if n < (min_degree + shift).n():
                continue
            for s in range(0, max.s() + 1):
                source = sseq_py.Bidegree.n_s(n, s)
                target = source - suspension_shift
                source_num_gens = res_b.number_of_gens_in_bidegree(source)
                target_num_gens = res_a.number_of_gens_in_bidegree(target)
                if source_num_gens == 0 or target_num_gens == 0:
                    m = ""
                else:
                    mat = hom.get_map(target.s()).hom_k(target.t())
                    is_identity = source_num_gens == target_num_gens and all(
                        all(
                            (z == 1 if col == row else z == 0)
                            for (col, z) in enumerate(x)
                        )
                        for (row, x) in enumerate(mat)
                    )
                    if is_identity:
                        m = ""
                    else:
                        m = f" - {[list(row) for row in mat]}"
                print(f"{n - shift_t} {s} {shift_t}: {source_num_gens}{m}")


if __name__ == "__main__":
    main()
