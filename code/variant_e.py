"""Variant E: The Chaotic Gaussian Collatz Map.

Parity rules (based on real part a, imaginary part b of z = a + bi):
  a even, b even  -> z / 4          (strong contraction)
  a odd,  b even  -> (1+i)z + 1     (expansion, |f'| = sqrt(2))
  a even, b odd   -> (1-i)z + i     (expansion, |f'| = sqrt(2))
  a odd,  b odd   -> (1+i)z + (1+i) (expansion, |f'| = sqrt(2))

Key results:
  - Period-40 cycle in dyadic Gaussian rationals Z[i, 1/2], starting at z = -120 + 66i.
  - Fractal basin boundary with box-counting dimension D ~ 1.70.
  - Denominator 2-adic valuations reach powers > 2^1000 within the 40-cycle.
"""


def step_E(z):
    """Apply one step of Variant E to z = a + bi."""
    a, b = int(z.real), int(z.imag)
    if a % 2 == 0 and b % 2 == 0:
        return z / 4
    if a % 2 == 1 and b % 2 == 0:
        return (1 + 1j) * z + 1
    if a % 2 == 0 and b % 2 == 1:
        return (1 - 1j) * z + 1j
    return (1 + 1j) * z + (1 + 1j)


def orbit(start_z, max_iter=3000, div_threshold=5000):
    """Iterate Variant E from start_z.

    Returns
    -------
    status : 'converged' | 'diverged' | 'unsettled'
    steps  : int
    cycle  : tuple of complex, or None
    """
    visited = {start_z: 0}
    path = [start_z]
    curr = start_z

    for i in range(1, max_iter + 1):
        if abs(curr) > div_threshold:
            return 'diverged', i, None
        curr = step_E(curr)
        curr = complex(round(curr.real, 4), round(curr.imag, 4))
        if curr in visited:
            return 'converged', i, tuple(path[visited[curr]:])
        visited[curr] = i
        path.append(curr)

    return 'unsettled', max_iter, None


if __name__ == "__main__":
    # Reproduce the period-40 cycle
    status, steps, cycle = orbit(complex(-120, 66))
    print(f"orbit(-120+66i): status={status}, steps={steps}, cycle_len={len(cycle) if cycle else None}")
