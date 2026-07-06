"""Optimal-transport helpers (discrete / Hungarian assignment).

Currently exposes only one primitive — `hungarian_match` — used wherever
two equal-sized point clouds need an OT-optimal pairing under
squared-Euclidean cost. Implemented via `scipy.optimize.linear_sum_assignment`,
i.e. exact (not entropic) OT.

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
