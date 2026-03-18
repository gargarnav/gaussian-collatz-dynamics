# Gaussian Collatz Dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19078383.svg)](https://doi.org/10.5281/zenodo.19078383)

Computational investigation of Collatz-like dynamical systems on the Gaussian
integers **Z[i]**, focusing on two contrasting variants:

| Variant | Character | Key result |
|---------|-----------|------------|
| **C** | Balanced / globally stable | Average Lyapunov exponent λ ≈ −0.66; stable for k > √2 |
| **E** | Chaotic / fractal | Period-40 cycle in Z[i, 1/2]; basin boundary D ≈ 1.70 |

---

## Publication

This code accompanies the research paper:

**Garg, A. (2026).** *Collatz-Type Dynamics on Gaussian Integers: Fractal 
Stability Islands, Long Cycles, and Dyadic Rational Structure* (Version 1.0). 
Zenodo. https://doi.org/10.5281/zenodo.19078383

**Abstract:** We introduce and study a family of Collatz-type dynamical 
systems defined on the Gaussian integers. Our computational investigation 
reveals three principal findings: (i) a stable periodic orbit of length 40, 
(ii) orbits live in the ring of dyadic Gaussian rationals Z[1/2][i], and 
(iii) the stability boundary is fractal with dimension D ≈ 1.70.

**Paper:** [View on Zenodo](https://doi.org/10.5281/zenodo.19078383)

---

## Repository layout

```
gaussian-collatz-dynamics/
├── paper/
│   ├── collatz_arxiv.tex       LaTeX source
│   └── figures/                All figures (PNG)
├── code/
│   ├── variant_c.py            Variant C step function + orbit iterator
│   ├── variant_e.py            Variant E step function + orbit iterator
│   ├── exact_cycle.py          Exact rational arithmetic; verifies 40-cycle
│   ├── fractal_dimension.py    Box-counting dimension for Variant E boundary
│   ├── lyapunov.py             Lyapunov exponent distribution for Variant C
│   └── visualizations.py       Driver — regenerates all paper figures
├── data/
│   ├── 40cycle_exact.txt       Exact rational coordinates of the 40-cycle
│   ├── stability_islands.csv   Converged/diverged grid for Variant E
│   └── lyapunov_results.csv    Per-orbit Lyapunov exponents for Variant C
├── requirements.txt
└── LICENSE
```

---

## Quick start

```bash
pip install -r requirements.txt
cd code
python visualizations.py        # regenerates all figures and data files
```

Individual modules can also be run standalone:

```bash
python exact_cycle.py           # verify 40-cycle, write data/40cycle_exact.txt
python lyapunov.py              # compute Lyapunov distribution
python fractal_dimension.py     # compute box-counting dimension
```

---

## Map definitions

Both variants operate on **z = a + bi ∈ Z[i]** and branch on the parities of a and b.

### Variant C (balanced)

| Condition | Rule |
|-----------|------|
| a even | z / k  (default k = 2) |
| a odd, b even | (1+i)z + 1 |
| a odd, b odd | (1+i)z + i |

Contraction probability 1/2, expansion factor √2 per step.
Theoretical stability threshold: k > √2 ≈ 1.414.

### Variant E (chaotic)

| Condition | Rule |
|-----------|------|
| a even, b even | z / 4 |
| a odd, b even | (1+i)z + 1 |
| a even, b odd | (1−i)z + i |
| a odd, b odd | (1+i)z + (1+i) |

---

## Key results

### Variant C — global convergence

- Empirical convergence rate ≈ 100 % on Z[i] ∩ [−100, 100]².
- Mean Lyapunov exponent λ ≈ −0.66 (all 1 000 sampled orbits negative).
- Critical threshold k★ = √2 confirmed analytically and numerically.

### Variant E — period-40 cycle

- Cycle starts at z₀ = −120 + 66i.
- All 40 elements lie in the dyadic Gaussian rationals Z[i, 1/2]:
  denominators are powers of 2, peaking beyond 2^1000 mid-cycle.
- Verified with zero floating-point error via `fractions.Fraction`.

### Variant E — fractal basin boundary

- Box-counting dimension D ≈ 1.70 (R² > 0.99).
- Boundary computed on a 400 × 400 grid over [−200, 200]².

---

## Figures

| File | Description |
|------|-------------|
| `variant_e_julia.png` | Stability islands of Variant E, coloured by cycle ID |
| `fractal_fit.png` | Log-log regression for box-counting dimension |
| `denom_growth.png` | Denominator growth along random Variant E orbits |
| `lyapunov_dist.png` | Lyapunov exponent histogram for Variant C |

---

## License

MIT — see [LICENSE](LICENSE).
