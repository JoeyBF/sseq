#!/usr/bin/env python3
"""
Compute dimensions of the Milnor algebra A_n for n = 0 to 125.
Python translation of algebra_dim.rs example.
"""

import ext_py
from ext_py import algebra_py


def main():
    # Create Milnor algebra over F_2
    algebra = algebra_py.MilnorAlgebra(p=2, unstable_enabled=False)

    # Compute basis up to degree 125
    algebra.compute_basis(125)

    # Print dimensions
    for n in range(126):
        print(f"dim A_{n} = {algebra.dimension(n)}")


if __name__ == "__main__":
    main()
