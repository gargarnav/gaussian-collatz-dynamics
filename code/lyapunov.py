"""Lyapunov exponent computation for Variant C.

The finite-time Lyapunov exponent is:
    lambda = (1/n) * sum_k log|f'(z_k)|

For Variant C the derivative magnitudes are:
  a even          -> |f'| = 1/k          (log = -log k)
  a odd (either b) -> |f'| = sqrt(2)     (log = 0.5 * log 2)

Theoretical value at k=2, assuming P(a even)=0.5:
    lambda_theory = 0.5*log(0.5) + 0.5*log(sqrt(2))
                  = 0.5*(-0.693) + 0.5*(0.346) ~ -0.174

Empirical mean over random orbits: lambda ~ -0.66
(The attractor spends more time in the contracting branch than the uniform
 probability argument suggests.)

Critical threshold (theoretical): k > sqrt(2) ~ 1.414.

Outputs
-------
  ../paper/figures/lyapunov_dist.png  -- histogram of per-orbit exponents
  ../data/lyapunov_results.csv        -- per-orbit (start_real, start_imag, lambda)
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LOG_SQRT2 = 0.5 * np.log(2)


def _orbit_lyapunov(start_z, k: float = 2.0, n_steps: int = 10000) -> float | None:
    """Compute the finite-time Lyapunov exponent for one orbit under Variant C."""
    curr = start_z
    log_derivs = []

    for _ in range(n_steps):
        if curr == 0:
            break
        a, b = int(curr.real), int(curr.imag)
        if a % 2 == 0:
            curr = curr / k
            log_derivs.append(-np.log(k))
        elif b % 2 == 0:
            curr = (1 + 1j) * curr + 1
            log_derivs.append(LOG_SQRT2)
        else:
            curr = (1 + 1j) * curr + 1j
            log_derivs.append(LOG_SQRT2)

    return float(np.mean(log_derivs)) if log_derivs else None


def compute_lyapunov_distribution(
    n_orbits: int = 1000,
    grid_radius: int = 100,
    k: float = 2.0,
    n_steps: int = 10000,
    seed: int = 42,
):
    """Sample random starting points and compute per-orbit Lyapunov exponents.

    Returns
    -------
    starts    : list of complex
    exponents : list of float
    """
    rng = np.random.default_rng(seed)
    starts = [
        complex(int(rng.integers(-grid_radius, grid_radius + 1)),
                int(rng.integers(-grid_radius, grid_radius + 1)))
        for _ in range(n_orbits)
    ]

    exponents = []
    valid_starts = []
    for z in starts:
        if z == 0:
            continue
        lam = _orbit_lyapunov(z, k=k, n_steps=n_steps)
        if lam is not None:
            exponents.append(lam)
            valid_starts.append(z)

    return valid_starts, exponents


def run(figures_dir: str = "../paper/figures", data_dir: str = "../data",
        n_orbits: int = 1000):
    """Compute Lyapunov distribution and save outputs."""
    print(f"Computing Lyapunov exponents for {n_orbits} random orbits under Variant C ...")
    starts, exponents = compute_lyapunov_distribution(n_orbits=n_orbits)

    avg_lambda = np.mean(exponents)
    print(f"Average Lyapunov exponent: {avg_lambda:.4f}")
    print(f"Fraction negative (stable): {np.mean(np.array(exponents) < 0):.1%}")

    # --- Histogram ---
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(exponents, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='r', linestyle='--', label='λ = 0')
    plt.axvline(avg_lambda, color='k', linestyle='-', label=f'Mean λ = {avg_lambda:.3f}')
    plt.xlabel('Lyapunov exponent λ')
    plt.ylabel('Count')
    plt.title('Lyapunov Exponent Distribution — Variant C (k=2)')
    plt.legend()
    plt.savefig(fig_dir / 'lyapunov_dist.png', dpi=150)
    plt.close()
    print(f"Saved {fig_dir / 'lyapunov_dist.png'}")

    # --- CSV ---
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / 'lyapunov_results.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['start_real', 'start_imag', 'lyapunov_exponent'])
        for z, lam in zip(starts, exponents):
            writer.writerow([int(z.real), int(z.imag), f'{lam:.6f}'])

    print(f"Saved {csv_path}")
    return avg_lambda


if __name__ == "__main__":
    run()
