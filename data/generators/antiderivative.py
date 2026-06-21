"""
Data generator for the antiderivative operator.

Operator: G(u)(x) = integral_0^x u(t) dt
Input function u(x): sampled from a Gaussian Random Field (GRF)
Output function s(x): computed via cumulative trapezoidal integration

GRF sampling via Cholesky of RBF kernel — gives smooth, diverse input functions.
"""

import numpy as np
from scipy.linalg import cholesky


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    """
    Radial Basis Function (squared exponential) kernel.
    K(x1, x2) = exp(-||x1 - x2||^2 / (2 * l^2))

    Args:
        x1: (n,) array
        x2: (m,) array
        length_scale: controls smoothness — larger = smoother functions

    Returns:
        (n, m) kernel matrix
    """
    diff = x1[:, None] - x2[None, :]   # (n, m)
    return np.exp(-0.5 * (diff / length_scale) ** 2)


def sample_grf(
    x: np.ndarray,
    n_samples: int,
    length_scale: float = 0.2,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Sample functions from a Gaussian Random Field using RBF kernel.

    Args:
        x: (m,) sensor locations
        n_samples: number of functions to sample
        length_scale: GRF smoothness parameter
        rng: numpy random generator (for reproducibility)

    Returns:
        u: (n_samples, m) — each row is one sampled function evaluated at x
    """
    if rng is None:
        rng = np.random.default_rng()

    K = rbf_kernel(x, x, length_scale)
    K += 1e-8 * np.eye(len(x))          # jitter for numerical stability
    L = cholesky(K, lower=True)          # Cholesky: K = L @ L.T

    z = rng.standard_normal((len(x), n_samples))   # (m, n_samples)
    u = (L @ z).T                                   # (n_samples, m)
    return u


def compute_antiderivative(
    u: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Compute s(x) = integral_0^x u(t) dt via cumulative trapezoidal rule.

    Args:
        u: (n_samples, m) input functions at sensor locations
        x: (m,) sensor locations (must be the same grid)

    Returns:
        s: (n_samples, m) antiderivative values at same locations
    """
    # np.cumulative trapz equivalent: manual implementation for clarity
    dx = np.diff(x)                                    # (m-1,)
    mid = 0.5 * (u[:, :-1] + u[:, 1:]) * dx[None, :]  # (n_samples, m-1)
    s = np.concatenate(
        [np.zeros((len(u), 1)), np.cumsum(mid, axis=1)], axis=1
    )                                                   # (n_samples, m)
    return s


def generate_dataset(
    n_samples: int,
    m: int,
    p_per_u: int,
    x_range: tuple = (0.0, 1.0),
    length_scale: float = 0.2,
    seed: int = 42,
):
    """
    Generate full dataset of (branch_input, trunk_input, target) triplets.

    The GRF is sampled on m fixed sensor points (branch net input).
    For each sampled function, p_per_u query points are drawn.
    Target is the antiderivative at those query points.

    Args:
        n_samples: number of input functions (ICs)
        m: number of fixed sensor points
        p_per_u: query points per function
        x_range: domain [a, b]
        length_scale: GRF smoothness
        seed: random seed

    Returns:
        branch_inputs: (n_samples * p_per_u, m)  — u evaluated at sensors
        trunk_inputs:  (n_samples * p_per_u, 1)  — query locations
        targets:       (n_samples * p_per_u, 1)  — s at query locations
    """
    rng = np.random.default_rng(seed)

    # Fixed sensor grid (same m points for all functions)
    x_sensors = np.linspace(x_range[0], x_range[1], m)

    # Sample input functions from GRF
    u_all = sample_grf(x_sensors, n_samples, length_scale, rng)       # (N, m)

    # Compute antiderivative on the sensor grid
    s_all = compute_antiderivative(u_all, x_sensors)                   # (N, m)

    # For each function, sample p_per_u query points from the sensor grid
    # (we interpolate from already-computed s — free after the solve)
    branch_list, trunk_list, target_list = [], [], []

    for i in range(n_samples):
        # Sample p query indices from the m sensor locations
        query_idx = rng.integers(0, m, size=p_per_u)
        x_query = x_sensors[query_idx]          # (p,)
        s_query = s_all[i, query_idx]           # (p,)

        branch_list.append(np.tile(u_all[i], (p_per_u, 1)))  # (p, m)
        trunk_list.append(x_query[:, None])                   # (p, 1)
        target_list.append(s_query[:, None])                  # (p, 1)

    branch_inputs = np.concatenate(branch_list, axis=0).astype(np.float32)
    trunk_inputs  = np.concatenate(trunk_list,  axis=0).astype(np.float32)
    targets       = np.concatenate(target_list, axis=0).astype(np.float32)

    return branch_inputs, trunk_inputs, targets, x_sensors


if __name__ == "__main__":
    # Quick sanity check
    B, T, S, x = generate_dataset(n_samples=10, m=100, p_per_u=50)
    print(f"Branch inputs: {B.shape}")   # (500, 100)
    print(f"Trunk inputs:  {T.shape}")   # (500, 1)
    print(f"Targets:       {S.shape}")   # (500, 1)
