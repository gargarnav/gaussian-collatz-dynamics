"""Exact rational arithmetic verification of the Variant E period-40 cycle.

Uses Python's fractions.Fraction to trace orbits without floating-point error,
confirming that the cycle lives in the dyadic Gaussian rationals Z[i, 1/2]
and that denominators are always powers of 2.

Outputs
-------
  data/40cycle_exact.txt  -- full list of exact (p/q + r/s * i) cycle elements
"""

import csv
import math
from fractions import Fraction
from pathlib import Path


# ---------------------------------------------------------------------------
# Exact complex arithmetic over Q
# ---------------------------------------------------------------------------

class ComplexFraction:
    """A Gaussian rational: real and imag parts are exact fractions."""

    def __init__(self, real, imag):
        self.real = Fraction(real)
        self.imag = Fraction(imag)

    def __add__(self, other):
        if isinstance(other, ComplexFraction):
            return ComplexFraction(self.real + other.real, self.imag + other.imag)
        if isinstance(other, (int, Fraction)):
            return ComplexFraction(self.real + other, self.imag)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ComplexFraction):
            return ComplexFraction(self.real - other.real, self.imag - other.imag)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, ComplexFraction):
            return ComplexFraction(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real,
            )
        if isinstance(other, (int, Fraction)):
            return ComplexFraction(self.real * other, self.imag * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, Fraction)):
            return ComplexFraction(self.real / other, self.imag / other)
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, ComplexFraction):
            return False
        return self.real == other.real and self.imag == other.imag

    def __hash__(self):
        return hash((self.real, self.imag))

    def __repr__(self):
        return f"({self.real}) + ({self.imag})i"

    def norm_sq(self):
        return self.real ** 2 + self.imag ** 2


# ---------------------------------------------------------------------------
# Step function (exact parity via rounding)
# ---------------------------------------------------------------------------

def _parity(frac):
    """Return True if the nearest integer to frac is even."""
    return round(float(frac)) % 2 == 0


def step_E_exact(z: ComplexFraction) -> ComplexFraction:
    """One step of Variant E using exact arithmetic."""
    a_even = _parity(z.real)
    b_even = _parity(z.imag)

    if a_even and b_even:
        return z / 4
    if (not a_even) and b_even:
        return ComplexFraction(1, 1) * z + 1
    if a_even and (not b_even):
        return ComplexFraction(1, -1) * z + ComplexFraction(0, 1)
    return ComplexFraction(1, 1) * z + ComplexFraction(1, 1)


# ---------------------------------------------------------------------------
# 2-adic valuation
# ---------------------------------------------------------------------------

def valuation_2(n: int) -> int:
    """Return the 2-adic valuation v_2(n): largest k such that 2^k | n."""
    if n == 0:
        return float('inf')
    v = 0
    while n % 2 == 0:
        v += 1
        n //= 2
    return v


def v2_fraction(x: Fraction) -> int:
    """2-adic valuation of a Fraction p/q: v_2(p) - v_2(q)."""
    return valuation_2(abs(x.numerator)) - valuation_2(x.denominator)


# ---------------------------------------------------------------------------
# Cycle verification
# ---------------------------------------------------------------------------

def find_exact_cycle(start: ComplexFraction, max_iter: int = 1000):
    """Trace the orbit from start and return the cycle once detected."""
    path = [start]
    visited = {start: 0}
    curr = start

    for i in range(1, max_iter + 1):
        curr = step_E_exact(curr)
        if curr in visited:
            return path[visited[curr]:]
        visited[curr] = i
        path.append(curr)

    raise RuntimeError("No cycle detected within max_iter steps.")


def verify_40_cycle(output_path: str = "../data/40cycle_exact.txt"):
    """Verify the period-40 cycle and write exact coordinates to file."""
    start = ComplexFraction(-120, 66)
    cycle = find_exact_cycle(start)

    print(f"Cycle length: {len(cycle)}")

    # Denominator analysis
    max_denom = max(max(z.real.denominator, z.imag.denominator) for z in cycle)
    all_pow2 = all(
        math.log2(z.real.denominator).is_integer()
        and math.log2(z.imag.denominator).is_integer()
        for z in cycle
    )
    print(f"Max denominator: {max_denom}")
    print(f"All denominators powers of 2: {all_pow2}")

    # 2-adic valuations
    valuations = [(v2_fraction(z.real), v2_fraction(z.imag)) for z in cycle]
    print(f"2-adic valuations (first 10): {valuations[:10]}")

    # Write to file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# Variant E period-40 cycle — exact rational coordinates\n")
        f.write("# Format: index | real numerator/denominator | imag numerator/denominator\n")
        for k, z in enumerate(cycle):
            f.write(f"{k}\t{z.real.numerator}/{z.real.denominator}\t"
                    f"{z.imag.numerator}/{z.imag.denominator}\n")

    print(f"Wrote exact cycle to {out}")
    return cycle


if __name__ == "__main__":
    verify_40_cycle()
