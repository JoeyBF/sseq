#!/usr/bin/env python3
"""
Compute dimensions of the Milnor algebra A_n for n = 0 to 125.
Python translation of algebra_dim.rs example.
"""

import ext_py


def main():
    ext_py.init_logging()

    # Create Milnor algebra over F_2
    algebra = ext_py.MilnorAlgebra(prime=2, truncated=False)

    # Compute basis up to degree 125
    algebra.compute_basis(125)

    # Print dimensions
    for n in range(126):
        print(f"dim A_{n} = {algebra.dimension(n)}")


if __name__ == "__main__":
    main()
