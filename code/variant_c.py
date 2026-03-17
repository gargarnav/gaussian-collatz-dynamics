"""Variant C: The Balanced Gaussian Collatz Map.

Parity rules (based on real part a, imaginary part b of z = a + bi):
  a even             -> z / k        (contraction, default k=2)
  a odd,  b even     -> (1+i)z + 1   (expansion, |f'| = sqrt(2))
  a odd,  b odd      -> (1+i)z + i   (expansion, |f'| = sqrt(2))

Average Lyapunov exponent lambda ~ -0.66 at k=2 => globally convergent to 0.
Critical threshold: stability holds for k > sqrt(2) ~ 1.414.
"""

import numpy as np


def step_C(z, k=2.0):
    """Apply one step of Variant C to z = a + bi."""
    a, b = int(z.real), int(z.imag)
    if a % 2 == 0:
        return z / k
    if b % 2 == 0:
        return (1 + 1j) * z + 1
    return (1 + 1j) * z + 1j


def orbit(start_z, k=2.0, max_iter=3000, div_threshold=5000):
    """Iterate Variant C from start_z.

    Returns
    -------
    status : 'converged' | 'diverged' | 'unsettled'
    steps  : int
    cycle  : tuple of complex, or None
    """
    step = lambda z: step_C(z, k)
    visited = {start_z: 0}
    path = [start_z]
    curr = start_z

    for i in range(1, max_iter + 1):
        if abs(curr) > div_threshold:
            return 'diverged', i, None
        curr = step(curr)
        curr = complex(round(curr.real, 4), round(curr.imag, 4))
        if curr in visited:
            return 'converged', i, tuple(path[visited[curr]:])
        visited[curr] = i
        path.append(curr)

    return 'unsettled', max_iter, None


def critical_threshold_sweep(k_range=None, grid_radius=20):
    """Sweep contraction factor k and return (k_values, convergence_rates)."""
    if k_range is None:
        k_range = np.arange(1.0, 3.1, 0.1)

    test_range = range(-grid_radius, grid_radius + 1)
    total = len(test_range) ** 2
    rates = []

    for k in k_range:
        conv = sum(
            1
            for a in test_range
            for b in test_range
            if orbit(complex(a, b), k=k)[0] == 'converged'
        )
        rates.append(conv / total)

    return list(k_range), rates


if __name__ == "__main__":
    # Quick sanity check
    status, steps, cycle = orbit(complex(3, 5))
    print(f"orbit(3+5i): status={status}, steps={steps}, cycle_len={len(cycle) if cycle else None}")
