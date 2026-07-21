"""Optimal-transport helpers (discrete Hungarian assignment + Gaussian OT).

`hungarian_match` gives the exact discrete OT pairing between two equal-sized
clouds under squared-Euclidean cost (via `scipy.optimize.linear_sum_assignment`,
not entropic). The Gaussian helpers — `gaussian_moments`, `psd_sqrt`,
`bures_map` — support the continuous OT map between Gaussian fits, used by the
McCann trajectory initialisation (`mccann_interpolate_trajectories`).

For *entropic* Sinkhorn divergence, prefer `ott-jax` if you need it
again — the in-repo Sinkhorn implementation was retired alongside the
Bridge ablation it served.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal-transport assignment between *x* and *y* (squared-Euclidean cost).

    Returns ``(row_idx, col_idx)`` such that ``x[row_idx[k]]`` is paired
    with ``y[col_idx[k]]`` under the discrete OT optimum. *x* and *y*
    may have different first-axis sizes; the returned pairing has length
    ``min(|x|, |y|)``.
    """
    cost = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return linear_sum_assignment(cost)


def gaussian_moments(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical mean and covariance of a point cloud.

    Args:
        points: ``(N, D)`` samples.

    Returns:
        ``(mu, cov)`` with ``mu`` of shape ``(D,)`` and ``cov`` of shape
        ``(D, D)``. With fewer than two points the covariance is undefined, so
        a zero matrix is returned (the caller regularises it before use).
    """
    P = np.asarray(points, dtype=np.float64)
    mu = P.mean(axis=0)
    if P.shape[0] < 2:
        return mu, np.zeros((P.shape[1], P.shape[1]))
    return mu, np.atleast_2d(np.cov(P, rowvar=False))


def psd_sqrt(A: np.ndarray) -> np.ndarray:
    """Unique symmetric positive-semidefinite square root of a symmetric matrix.

    Computed as ``V diag(sqrt(w)) Vᵀ`` from the eigendecomposition, with
    eigenvalues clipped at zero. The ``V … Vᵀ`` form cancels eigenvector sign
    ambiguity, so the result is a continuous function of *A* (unlike a raw
    eigenvector factor) and coordinate-free (unlike Cholesky).

    Args:
        A: ``(D, D)`` symmetric (PSD) matrix.

    Returns:
        The symmetric PSD square root ``S`` with ``S @ S == A`` for PSD *A*.
    """
    A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def bures_map(cov_source: np.ndarray, cov_target: np.ndarray) -> np.ndarray:
    """Linear optimal-transport map between two centred Gaussians.

    Returns the symmetric positive-definite matrix ``A`` that pushes
    ``N(0, cov_source)`` onto ``N(0, cov_target)`` under the quadratic OT cost,
    i.e. ``A = C0^{-1/2} (C0^{1/2} C1 C0^{1/2})^{1/2} C0^{-1/2}`` — the unique
    SPD solution of ``A C0 A = C1``. *cov_source* must be positive-definite
    (regularise it before calling).

    Args:
        cov_source: ``(D, D)`` source covariance ``C0`` (positive-definite).
        cov_target: ``(D, D)`` target covariance ``C1`` (PSD).

    Returns:
        The ``(D, D)`` symmetric PD OT map ``A``.
    """
    s0 = psd_sqrt(cov_source)
    s0_inv = np.linalg.inv(s0)
    return s0_inv @ psd_sqrt(s0 @ cov_target @ s0) @ s0_inv
