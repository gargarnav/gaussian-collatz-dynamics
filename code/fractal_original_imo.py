"""Box-counting fractal dimension for Variant E stability boundaries.

Algorithm
---------
1. Rasterise a grid of Gaussian integers; mark each point as converged or diverged
   under Variant E iteration.
2. Extract boundary pixels (converged points with at least one diverged neighbour).
3. Fit log N(eps) ~ D * log(1/eps) by linear regression over a range of box sizes.

Result: D ~ 1.70 for the [-200, 200]^2 grid at resolution 400x400.

Outputs
-------
  ../paper/figures/fractal_fit.png   -- log-log regression plot
  ../data/stability_islands.csv      -- per-pixel (a, b, cycle_id) data
"""

import csv
from pathlib import Path

import numpy as np
from scipy import stats

from variant_e import step_E


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------

def _classify_grid(resolution: int, extent: int, max_iter: int = 100, div_threshold: int = 500):
    """Return a (resolution x resolution) integer array.

    0  = diverged
    1+ = converged (all treated as 1 here; cycle ID assignment is in stability_islands)
    """
    vals = np.linspace(-extent, extent, resolution)
    grid = np.zeros((resolution, resolution), dtype=np.int8)

    for i, y in enumerate(vals[::-1]):   # top row = +y
        for j, x in enumerate(vals):
            curr = complex(x, y)
            diverged = False
            for _ in range(max_iter):
                if abs(curr) > div_threshold:
                    diverged = True
                    break
                a, b = int(curr.real), int(curr.imag)
                curr = step_E(complex(a, b))   # integer-snap then step
            if not diverged:
                grid[i, j] = 1

    return grid, vals


def _boundary_points(grid: np.ndarray):
    """Return list of (row, col) indices on the converged/diverged boundary."""
    rows, cols = grid.shape
    pts = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r, c] == 1:
                if 0 in (grid[r + 1, c], grid[r - 1, c], grid[r, c + 1], grid[r, c - 1]):
                    pts.append((r, c))
    return pts


# ---------------------------------------------------------------------------
# Box counting
# ---------------------------------------------------------------------------

def box_counting_dimension(boundary_points, max_box_fraction: float = 0.25):
    """Estimate fractal dimension via box counting.

    Parameters
    ----------
    boundary_points : list of (row, col)
    max_box_fraction : largest box size as fraction of grid size

    Returns
    -------
    dimension : float
    r_squared : float
    log_sizes, log_counts : arrays for plotting
    """
    if not boundary_points:
        return 0.0, 0.0, np.array([]), np.array([])

    pts = np.array(boundary_points)
    grid_size = pts.max() + 1

    sizes, counts = [], []
    s = 1
    while s < grid_size * max_box_fraction:
        scaled = (pts // s)
        counts.append(len(set(map(tuple, scaled))))
        sizes.append(s)
        s *= 2

    log_sizes = np.log(1.0 / np.array(sizes))
    log_counts = np.log(np.array(counts))

    slope, intercept, r, *_ = stats.linregress(log_sizes, log_counts)
    return slope, r ** 2, log_sizes, log_counts


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(resolution: int = 400, extent: int = 200, figures_dir: str = "../paper/figures",
        data_dir: str = "../data"):
    """Compute fractal dimension and save outputs."""
    print(f"Classifying {resolution}x{resolution} grid over [{-extent}, {extent}]^2 ...")
    grid, vals = _classify_grid(resolution, extent)

    boundary = _boundary_points(grid)
    print(f"Boundary points: {len(boundary)}")

    dim, r2, log_s, log_n = box_counting_dimension(boundary)
    print(f"Fractal dimension D = {dim:.4f}  (R² = {r2:.4f})")

    # --- Save fractal fit figure ---
    import matplotlib.pyplot as plt

    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(log_s, log_n, 'o', label='Data')
    slope_line = dim * log_s + (log_n[0] - dim * log_s[0])
    plt.plot(log_s, slope_line, 'r-', label=f'Fit D={dim:.2f}, R²={r2:.3f}')
    plt.xlabel('log(1/ε)')
    plt.ylabel('log N(ε)')
    plt.title('Box-Counting Dimension — Variant E Boundary')
    plt.legend()
    plt.savefig(fig_dir / 'fractal_fit.png', dpi=150)
    plt.close()
    print(f"Saved {fig_dir / 'fractal_fit.png'}")

    # --- Save stability islands CSV ---
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / 'stability_islands.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['a', 'b', 'converged'])
        for i, y in enumerate(vals[::-1]):
            for j, x in enumerate(vals):
                writer.writerow([int(round(x)), int(round(y)), int(grid[i, j])])

    print(f"Saved {csv_path}")
    return dim, r2


if __name__ == "__main__":
    run()