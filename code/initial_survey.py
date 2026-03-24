"""Initial survey of Variant E stability."""

import numpy as np
from variant_e import orbit

def main():
    extent = 100
    # Resolution 200x200 uses exactly [-100, 100) range of integers
    vals = np.arange(-extent, extent)
    total = len(vals) * len(vals)
    converged = 0
    
    print(f"Evaluating Variant E stability over [{-extent}, {extent}]^2 (resolution {len(vals)}x{len(vals)}) ...")
    
    # Using integer lattice points Z[i]
    for b in vals:
        for a in vals:
            status, _, _ = orbit(complex(a, b), max_iter=500, div_threshold=5000)
            if status == 'converged':
                converged += 1
                
    fraction = converged / total
    print(f"Total points: {total}")
    print(f"Converged points: {converged}")
    print(f"Fraction of stable points: {fraction:.2%}")

if __name__ == "__main__":
    main()
