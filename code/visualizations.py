"""Figure generation for all paper visualisations.

Running this script regenerates every figure in paper/figures/ and
updates all data files in data/.

Figures produced
----------------
  variant_e_julia.png   -- Variant E stability islands (colour = cycle ID)
  fractal_fit.png       -- Box-counting log-log regression
  denom_growth.png      -- Denominator 2-adic growth along random orbits
  lyapunov_dist.png     -- Lyapunov exponent histogram for Variant C

Usage
-----
  cd code/
  python visualizations.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exact_cycle import ComplexFraction, step_E_exact, find_exact_cycle
from fractal_dimension import run as run_fractal
from lyapunov import run as run_lyapunov
from variant_e import step_E, orbit as orbit_E

FIGURES_DIR = Path("../paper/figures_new")
DATA_DIR = Path("../data_new")


# ---------------------------------------------------------------------------
# Figure 1: Variant E stability islands
# ---------------------------------------------------------------------------

def plot_stability_islands(resolution: int = 2000, extent: int = 1000):
    """Plot converged regions of Variant E, coloured by cycle ID."""
    print("Plotting Variant E stability islands ...")
    # Computational grid: [-1000, 1000]^2 for comprehensive coverage
    # Note: Full [-1000,1000] grid may require significant computation time
    vals = np.arange(-extent, extent)
    grid = np.zeros((len(vals), len(vals)))

    cycle_map: dict = {}
    next_id = 1

    for i, b in enumerate(tqdm(list(reversed(vals)), desc="Computing variants")):
        for j, a in enumerate(vals):
            status, _, cycle = orbit_E(complex(a, b), max_iter=500)
            if status == 'converged' and cycle is not None:
                key = tuple(sorted((round(z.real), round(z.imag)) for z in cycle))
                if key not in cycle_map:
                    cycle_map[key] = next_id
                    next_id += 1
                grid[i, j] = cycle_map[key]

    masked = np.ma.masked_where(grid == 0, grid)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid == 0, cmap='gray', extent=[-extent, extent, -extent, extent], alpha=0.3)
    ax.imshow(masked, cmap='viridis', extent=[-extent, extent, -extent, extent])
    ax.set_title('Variant E — Stability Islands')
    ax.set_xlabel('Re(z)', fontsize=14)
    ax.set_ylabel('Im(z)', fontsize=14)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / 'variant_e_julia.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Period-40 cycle path
# ---------------------------------------------------------------------------

def plot_40_cycle():
    """Plot the trajectory of the Variant E period-40 cycle."""
    print("Plotting period-40 cycle path ...")
    start = complex(-120, 66)
    visited: dict = {start: 0}
    path = [start]
    curr = start

    for step in range(1, 1000):
        curr = step_E(curr)
        curr = complex(round(curr.real, 2), round(curr.imag, 2))
        if curr in visited:
            cycle = path[visited[curr]:]
            break
        visited[curr] = step
        path.append(curr)
    else:
        print("Warning: 40-cycle not reproduced.")
        return

    pts = list(cycle) + [cycle[0]]
    xs = [z.real for z in pts]
    ys = [z.imag for z in pts]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(xs, ys, 'b-', linewidth=0.8)
    ax.scatter(xs[:-1], ys[:-1], s=20, zorder=5)
    ax.set_title(f'Variant E — Period-{len(cycle)} Cycle')
    ax.set_xlabel('Re(z)', fontsize=14)
    ax.set_ylabel('Im(z)', fontsize=14)
    ax.grid(True, alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path_fig = FIGURES_DIR / 'variant_e_40cycle.png'
    fig.savefig(path_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path_fig}")


# ---------------------------------------------------------------------------
# Figure 3: Denominator growth
# ---------------------------------------------------------------------------

def plot_denominator_growth(n_orbits: int = 10, n_steps: int = 60, seed: int = 42):
    """Plot denominator growth along random Variant E orbits using exact arithmetic."""
    print("Plotting denominator growth ...")
    rng = np.random.default_rng(seed)
    starts = [
        ComplexFraction(int(rng.integers(-50, 51)), int(rng.integers(-50, 51)))
        for _ in range(n_orbits)
    ]

    fig, ax = plt.subplots()

    for s in starts:
        curr = s
        denoms = []
        for _ in range(n_steps):
            curr = step_E_exact(curr)
            d = max(curr.real.denominator, curr.imag.denominator)
            denoms.append(d)
            if abs(float(curr.real)) > 1e6:
                break
        ax.plot(denoms)

    ax.set_yscale('log')
    ax.set_title('Denominator Growth Along Variant E Orbits')
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Max denominator (log scale)', fontsize=14)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / 'denom_growth.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # These two also write data/ files
    run_lyapunov(figures_dir=str(FIGURES_DIR), data_dir=str(DATA_DIR))
    run_fractal(figures_dir=str(FIGURES_DIR), data_dir=str(DATA_DIR))

    plot_40_cycle()
    plot_denominator_growth()
    plot_stability_islands()

    print("\nAll figures regenerated.")


if __name__ == "__main__":
    main()
