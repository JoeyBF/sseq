#!/usr/bin/env python3
"""Tensor two modules together and print the result as module JSON.

Python port of ext/examples/tensor.rs.
"""

import json

import _query as query
import ext_py
from ext_py import algebra_py


def main():
    left = query.with_default("Left module", "S_2", ext_py.parse_module_name)
    p = left["p"]

    def parse_right(name):
        module = ext_py.parse_module_name(name)
        if module["p"] != p:
            raise ValueError("Two modules must be over the same prime")
        return module

    right = query.with_default("Right module", "S_2", parse_right)

    algebra = algebra_py.SteenrodAlgebra.adem(p)

    left_module = algebra_py.steenrod_module_from_json(algebra, left)
    right_module = algebra_py.steenrod_module_from_json(algebra, right)

    tensor_module = algebra_py.TensorModule(left_module, right_module)

    # Convert to a finite dimensional module for output.
    # NOTE: `from_tensor_module` is NOT yet bound (aspirational API); the class
    # was renamed FDModule -> FDModuleBuilder, but this conversion constructor is
    # still pending in the bindings. This line will not run until it is bound.
    tensor = algebra_py.FDModuleBuilder.from_tensor_module(tensor_module)
    tensor.name = ""

    output = {"p": p}
    output.update(tensor.to_json())

    # serde_json's Display is compact (no spaces) and preserves insertion order.
    print(json.dumps(output, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
