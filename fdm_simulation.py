"""
FDM Simulation of Electrostatic Potential in C-AFM / TiO2 / Pt System

Solves the electrostatic equation:
    ∇·(ε(r) ∇φ) = 0

on a 2D domain representing the AFM tip–TiO2–Pt bottom electrode geometry.
GPU acceleration via CuPy is used when available; falls back to NumPy otherwise.

Usage
-----
    python fdm_simulation.py

Output
------
    Compressed NumPy archive (.npz) containing the converged potential field,
    permittivity map, electrode masks, and simulation metadata.

Reference
---------
    Note S1 of the Supporting Information:
    "Three-dimensional Visualization of Conductive Filaments in TiO2
     via Conductive Atomic Force Microscopy Tomography"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from tkinter import Tk, filedialog

try:
    import cupy as cp
    _device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"GPU detected: {_device}")
    USE_GPU = True
except Exception:
    import numpy as cp
    print("CuPy not available — running on CPU.")
    USE_GPU = False


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    W_sim: float = 20.0       # Domain width  [nm]
    H_sim: float = 15.0       # Domain height [nm]
    res: float = 0.02         # Grid spacing  [nm]

    eps_air: float = 1.0      # Relative permittivity — air
    eps_tio2: float = 50.0    # Relative permittivity — TiO2
    film_h: float = 10.0      # TiO2 film thickness [nm]

    V_tip: float = 5.5        # Tip bias [V]
    V_be: float = 0.0         # Bottom electrode bias [V]

    h_prot: float = 1.2       # BE protrusion height [nm]
    sigma_prot: float = 0.8   # BE protrusion Gaussian width [nm]

    R_tip: float = 20.0       # Tip hemisphere radius [nm]
    tip_gap: float = 0.0      # Tip–surface gap [nm]

    omega: float = 1.0        # SOR relaxation factor (1.0 = Jacobi)
    max_iter: int = 300_000   # Maximum iterations
    tol: float = 1e-7         # Convergence tolerance (max nodal update)
    check_every: int = 5_000  # Convergence check interval


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_numpy(arr: np.ndarray) -> np.ndarray:
    """Return a NumPy array regardless of whether the input is CuPy or NumPy."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def harmonic_mean(a, b):
    """Element-wise harmonic mean, used for interface permittivity."""
    return (2.0 * a * b) / (a + b)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def build_geometry(cfg: SimConfig):
    """Construct the simulation grid, permittivity map, and electrode masks."""
    x = cp.arange(-cfg.W_sim / 2, cfg.W_sim / 2 + cfg.res, cfg.res)
    y = cp.arange(0, cfg.H_sim + cfg.res, cfg.res)
    X, Y = cp.meshgrid(x, y)

    epsilon = cp.where(Y <= cfg.film_h, cfg.eps_tio2, cfg.eps_air)

    prot = cfg.h_prot * cp.exp(-X ** 2 / (2 * cfg.sigma_prot ** 2))
    mask_be = Y <= prot

    cy = cfg.film_h + cfg.tip_gap + cfg.R_tip
    mask_tip = (X ** 2 + (Y - cy) ** 2) <= cfg.R_tip ** 2

    return X, Y, epsilon, mask_be, mask_tip


def solve_potential(cfg: SimConfig):
    """
    Solve ∇·(ε ∇φ) = 0 using a finite-difference successive over-relaxation
    (SOR) scheme on the geometry defined by *cfg*.

    Parameters
    ----------
    cfg : SimConfig
        Simulation parameters.

    Returns
    -------
    X, Y : array
        Coordinate grids.
    V : array
        Converged electrostatic potential [V].
    epsilon : array
        Permittivity map.
    mask_be : array
        Boolean mask for the bottom electrode.
    mask_tip : array
        Boolean mask for the AFM tip.
    n_iter : int
        Number of iterations until convergence.
    """
    X, Y, epsilon, mask_be, mask_tip = build_geometry(cfg)

    active = cp.ones(X.shape, dtype=bool)
    active[0, :] = active[-1, :] = active[:, 0] = active[:, -1] = False
    active[mask_be | mask_tip] = False
    am_inner = active[1:-1, 1:-1]

    V = cp.zeros_like(X, dtype=cp.float64)
    V[mask_tip] = cfg.V_tip
    V[mask_be] = cfg.V_be

    eps = epsilon.astype(cp.float64)
    ec = eps[1:-1, 1:-1]
    e_up    = harmonic_mean(ec, eps[2:,   1:-1])
    e_down  = harmonic_mean(ec, eps[:-2,  1:-1])
    e_left  = harmonic_mean(ec, eps[1:-1, :-2])
    e_right = harmonic_mean(ec, eps[1:-1, 2:])
    e_sum   = e_up + e_down + e_left + e_right

    n_iter = cfg.max_iter
    for i in range(cfg.max_iter):
        V_jacobi = (
            e_up    * V[2:,   1:-1] +
            e_down  * V[:-2,  1:-1] +
            e_left  * V[1:-1, :-2]  +
            e_right * V[1:-1, 2:]
        ) / e_sum

        delta = cfg.omega * (V_jacobi - V[1:-1, 1:-1])
        V[1:-1, 1:-1][am_inner] += delta[am_inner]

        if i % cfg.check_every == 0:
            max_update = float(to_numpy(cp.max(cp.abs(delta[am_inner]))))
            print(f"  iter {i:7d} | max update = {max_update:.3e}")
            if max_update < cfg.tol:
                print(f"Converged at iteration {i}.")
                n_iter = i
                break

    return X, Y, V, epsilon, mask_be, mask_tip, n_iter


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = SimConfig()

    print("Starting FDM simulation...")
    X, Y, V, epsilon, mask_be, mask_tip, n_iter = solve_potential(cfg)

    root = Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(
        title="Save simulation result",
        defaultextension=".npz",
        initialfile=f"FDM_V{cfg.V_tip}V_h{cfg.h_prot}_s{cfg.sigma_prot}.npz",
        filetypes=[("NumPy archive", "*.npz")],
    )
    root.destroy()

    if not save_path:
        print("Save cancelled.")
        return

    np.savez_compressed(
        save_path,
        X=to_numpy(X),
        Y=to_numpy(Y),
        V=to_numpy(V),
        epsilon=to_numpy(epsilon),
        mask_be=to_numpy(mask_be),
        mask_tip=to_numpy(mask_tip),
        res=cfg.res,
        film_h=cfg.film_h,
        n_iter=n_iter,
    )
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
