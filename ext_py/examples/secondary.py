#!/usr/bin/env python3
"""
Compute d_2 differentials in the Adams spectral sequence.
Python translation of secondary.rs example.
"""

import os
import ext_py

def main():
    ext_py.init_logging()
    
    # Query for module (must use Milnor basis)
    resolution = ext_py.query_module(algebra_type=ext_py.AlgebraType.MILNOR, save=True)
    
    # Create secondary resolution
    lift = ext_py.SecondaryResolution(resolution)
    
    # Check for distributed computation
    secondary_job = os.environ.get('SECONDARY_JOB')
    if secondary_job:
        s = int(secondary_job)
        lift.compute_partial(s)
        return
    
    # Extend all homotopies
    lift.extend_all()
    
    # d_2 differential has bidegree shift (-1, 2)
    d2_shift = ext_py.Bidegree.from_n_s(-1, 2)
    
    # Iterate through targets of d_2
    for bidegree in lift.underlying().iter_nonzero_stem():
        if bidegree.s < 3:
            continue
        
        if bidegree.t - 1 > resolution.module(bidegree.s - 2).max_computed_degree():
            continue
        
        homotopy = lift.homotopy(bidegree.s)
        m = homotopy.homotopies.hom_k(bidegree.t - 1)
        
        for i, entry in enumerate(m):
            source_gen = ext_py.BidegreeGenerator(bidegree - d2_shift, i)
            print(f"d_2 x_{source_gen} = {entry}")

if __name__ == "__main__":
    main()