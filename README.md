# FDM Simulation — Electrostatic Potential

Finite-difference simulation of the electrostatic potential distribution in the
C-AFM tip / TiO<sub>2</sub> / Pt bottom electrode geometry described in **Note S1** of the
Supporting Information.

## Physics

Solves the generalized Laplace equation with position-dependent permittivity:

$$\nabla \cdot (\varepsilon(\mathbf{r})\, \nabla \varphi) = 0$$

using a successive over-relaxation (SOR) finite-difference scheme on a 2D
Cartesian grid. Interface permittivities are evaluated with the harmonic mean.

## Geometry

| Region | Permittivity |
|--------|-------------|
| TiO₂ film (y ≤ 10 nm) | ε = 50 |
| Air (y > 10 nm) | ε = 1 |

- **AFM tip**: hemisphere, R = 20 nm, φ = +5.5 V  
- **Bottom electrode**: φ = 0 V, surface modeled as a Gaussian protrusion (h = 1.2 nm, σ = 0.8 nm)  
- **Domain**: 20 nm × 15 nm, grid spacing 0.02 nm  
- **Convergence**: max nodal update < 10⁻⁷

## Requirements

```
numpy
cupy      # optional — falls back to NumPy if unavailable
```

## Usage

```bash
python fdm_simulation.py
```

Parameters are defined in the `SimConfig` dataclass at the top of the script.
The converged potential and metadata are saved as a compressed `.npz` archive.

## Output fields (`.npz`)

| Key | Description |
|-----|-------------|
| `X`, `Y` | Coordinate grids [nm] |
| `V` | Electrostatic potential [V] |
| `epsilon` | Permittivity map |
| `mask_be` | Boolean mask — bottom electrode |
| `mask_tip` | Boolean mask — AFM tip |
| `res` | Grid spacing [nm] |
| `film_h` | TiO₂ thickness [nm] |
| `n_iter` | Iterations to convergence |
