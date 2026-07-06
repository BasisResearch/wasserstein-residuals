"""Evaluation metrics: Wasserstein, Bures-Wasserstein, MMD, plus
iJKOnet UVPs and functional-accuracy metrics that operate on already-
sampled point clouds.

Uses the `ot` (Python Optimal Transport) library for fast, exact or
regularised Wasserstein distances.  POT uses compiled C/C++ solvers
so there is no JIT compilation overhead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import ot

from stitching._kde import SpatioTemporalData
from stitching.utils.solvers import jko_proximal_step
from stitching.velocity import WGFFunctional


def wasserstein(
    x: np.typing.ArrayLike,
    y: np.typing.ArrayLike,
    metric: Literal["emd", "w2", "w2_squared"] = "emd",
    epsilon: float | None = None,
) -> float:
    r"""Wasserstein distance between two point clouds.

    Uses POT's network-simplex solver (exact) by default, or entropic
    Sinkhorn when *epsilon* is given.

    Args:
        x: first point cloud, `(N, D)`.
        y: second point cloud, `(M, D)`.
        metric: which distance to compute:
            - `"emd"` — $W_1$ (Euclidean ground cost, default).
            - `"w2"` — $W_2$ (square root of OT cost with squared Euclidean ground cost).
            - `"w2_squared"` — $W_2^2$ (OT cost with squared Euclidean ground cost).
        epsilon: if given, use Sinkhorn with this regularisation
            strength instead of exact EMD.

    Returns:
        Scalar distance.
    """
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)

    if metric == "emd":
        M = ot.dist(x_np, y_np, metric="euclidean")
    else:
        M = ot.dist(x_np, y_np, metric="sqeuclidean")

    a = np.ones(x_np.shape[0], dtype=np.float64) / x_np.shape[0]
    b = np.ones(y_np.shape[0], dtype=np.float64) / y_np.shape[0]

    if epsilon is not None:
        cost = float(ot.sinkhorn2(a, b, M, reg=epsilon))
    else:
        cost = float(ot.emd2(a, b, M))

    if metric == "w2":
        return cost**0.5
    return cost


def mmd_rbf(
    x: np.typing.ArrayLike,
    y: np.typing.ArrayLike,
    bandwidths: tuple[float, ...] | None = None,
    seed: int = 0,
) -> float:
    r"""Squared maximum mean discrepancy with sum of Gaussian RBF kernels.

    $\mathrm{MMD}^2(P, Q) = \mathbb{E}_{x,x'\sim P}[k(x,x')]
        + \mathbb{E}_{y,y'\sim Q}[k(y,y')] - 2\,\mathbb{E}_{x\sim P,y\sim Q}[k(x,y)]$

    with $k(x,y) = \tfrac{1}{B}\sum_{b=1}^{B}\exp(-\|x-y\|^2 / (2\sigma_b^2))$.
    By default ``bandwidths`` is the median pairwise-distance heuristic
    scaled by $\{1/4, 1/2, 1, 2, 4\}$.

    ``seed`` controls the subsample drawn for the median heuristic when the
    pooled cloud exceeds 1024 points (it has no effect otherwise).
    """
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)

    Dxy = ot.dist(x_np, y_np, metric="sqeuclidean")
    Dxx = ot.dist(x_np, x_np, metric="sqeuclidean")
    Dyy = ot.dist(y_np, y_np, metric="sqeuclidean")

    if bandwidths is None:
        joint = np.vstack([x_np, y_np])
        if joint.shape[0] > 1024:
            sub = np.random.RandomState(seed).permutation(joint.shape[0])[:1024]
            joint = joint[sub]
        D = ot.dist(joint, joint, metric="euclidean")
        sigma = float(np.median(D[D > 0])) if (D > 0).any() else 1.0
        bandwidths = tuple(sigma * s for s in (0.25, 0.5, 1.0, 2.0, 4.0))

    mmd2 = 0.0
    for s in bandwidths:
        denom = 2.0 * s * s
        Kxx = np.exp(-Dxx / denom)
        Kyy = np.exp(-Dyy / denom)
        Kxy = np.exp(-Dxy / denom)
        mmd2 += float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    return mmd2 / len(bandwidths)


def mmd_dmsb(
    x: np.typing.ArrayLike,
    y: np.typing.ArrayLike,
    sample_size: int = 1000,
    kernel_num: int = 5,
    kernel_mul: float = 2.0,
    seed: int = 0,
) -> float:
    r"""DMSB-style multi-bandwidth Gaussian RBF MMD$^2$.

    Matches the `MMD_loss` of TianrongChen/DMSB (and iJKOnet's `MMD_DMSB`):
    pool both samples, set the base bandwidth to the **mean** of all squared
    pairwise distances (excluding self), use kernels at
    `bandwidth * kernel_mul^i` for ``i = -kernel_num//2 .. kernel_num//2``,
    and form ``MMD = mean(XX + YY - XY - YX)`` (note: no factor of 1/2 on
    the cross term — this is the DMSB convention). Subsamples each side
    to *sample_size* to match the baseline implementation.
    """
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)

    rng = np.random.RandomState(seed)
    n = min(sample_size, x_np.shape[0], y_np.shape[0])
    if x_np.shape[0] > n:
        x_np = x_np[rng.choice(x_np.shape[0], n, replace=False)]
    if y_np.shape[0] > n:
        y_np = y_np[rng.choice(y_np.shape[0], n, replace=False)]

    total = np.vstack([x_np, y_np])
    n_samples = total.shape[0]
    L2 = ot.dist(total, total, metric="sqeuclidean")
    bandwidth = float(L2.sum() / (n_samples * n_samples - n_samples))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidths = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    K = sum(np.exp(-L2 / b) for b in bandwidths)
    XX = K[:n, :n]
    YY = K[n:, n:]
    XY = K[:n, n:]
    YX = K[n:, :n]
    return float(np.mean(XX + YY - XY - YX))


def _matrix_sqrt_psd(A: np.ndarray) -> np.ndarray:
    """Symmetric matrix square root of a PSD matrix via eigendecomposition."""
    A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def _moments(P: np.ndarray, w: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted (or uniform) mean and covariance of a `(N, D)` point set."""
    if w is None:
        return P.mean(axis=0), np.atleast_2d(np.cov(P, rowvar=False))
    w_arr = np.asarray(w, dtype=np.float64)
    w_arr = w_arr / max(w_arr.sum(), 1e-12)
    mu = (w_arr[:, None] * P).sum(axis=0)
    diff = P - mu
    cov = np.atleast_2d((w_arr[:, None] * diff).T @ diff)
    return mu, cov


def bures_wasserstein_sq(
    P: np.ndarray,
    Q: np.ndarray,
    w_P: np.ndarray | None = None,
    w_Q: np.ndarray | None = None,
) -> float:
    r"""Squared Bures–Wasserstein distance between Gaussian fits to two point sets.

    Following iJKOnet (Persiianov et al., ICLR 2026), Eq. (21):
    $\mathrm{Bd}^2_{W_2}(P, Q) = \|\mu_P - \mu_Q\|^2 +
        \mathrm{tr}\!\left(\Sigma_P + \Sigma_Q -
                            2\,(\Sigma_P^{1/2} \Sigma_Q \Sigma_P^{1/2})^{1/2}\right)$.

    When ``w_P`` / ``w_Q`` are given the mean and covariance are computed
    with those importance weights; uniform otherwise.
    """
    mu_P, Sp = _moments(P, w_P)
    mu_Q, Sq = _moments(Q, w_Q)
    Sp_sqrt = _matrix_sqrt_psd(Sp)
    cross_sqrt = _matrix_sqrt_psd(Sp_sqrt @ Sq @ Sp_sqrt)
    cov_term = float(np.trace(Sp) + np.trace(Sq) - 2.0 * np.trace(cross_sqrt))
    mean_term = float(np.sum((mu_P - mu_Q) ** 2))
    return max(mean_term + cov_term, 0.0)


# ---------------------------------------------------------------------------
# iJKOnet-style UVP metrics
# ---------------------------------------------------------------------------


def _mean(by_time: dict[float, float]) -> float:
    return float(np.mean(list(by_time.values()))) if by_time else float("nan")


def bw2_uvp(
    samples_by_time_pred: dict[float, jax.Array],
    test_data: SpatioTemporalData,
    pred_weights: jax.Array | np.ndarray | None = None,
) -> tuple[float, dict[float, float]]:
    r"""Bures–Wasserstein UVP, iJKOnet Eq. (20).

    For each observation time $t_k$ with both ground-truth and predicted samples,
    computes
    $$\mathrm{Bd}^2_{W_2}\text{-UVP}(\rho_k, \hat\rho_k)
        = 100 \cdot \frac{\mathrm{Bd}^2_{W_2}(\rho_k, \hat\rho_k)}
                          {\tfrac{1}{2}\,\mathrm{Var}(\rho_k)},$$
    where $\mathrm{Var}(\rho_k) = \mathrm{tr}\,\Sigma_k$. The predicted-cloud
    Gaussian fit honours ``pred_weights`` (model mixture weights) when given.

    Returns ``(mean_uvp, {t: uvp_value})``.
    """
    test_t = np.asarray(test_data.t)
    test_x = np.asarray(test_data.x)
    w_pred = None if pred_weights is None else np.asarray(pred_weights)
    by_time: dict[float, float] = {}
    for t_key_pred, pred in samples_by_time_pred.items():
        gt = test_x[np.isclose(test_t, float(t_key_pred), atol=1e-6)]
        if gt.shape[0] < 2 or pred.shape[0] < 2:
            continue
        gt = gt.astype(np.float64)
        pr = np.asarray(pred, dtype=np.float64)
        var = float(np.trace(np.atleast_2d(np.cov(gt, rowvar=False))))
        if var <= 0:
            continue
        bw_sq = bures_wasserstein_sq(gt, pr, w_Q=w_pred)
        by_time[float(t_key_pred)] = 100.0 * bw_sq / (0.5 * var)
    return _mean(by_time), by_time


def l2_uvp(
    learned_grad_fn: Callable[[np.ndarray], np.typing.ArrayLike],
    true_grad_fn: Callable[[np.ndarray], np.typing.ArrayLike],
    train_data: SpatioTemporalData,
    tau: float,
) -> tuple[float, dict[float, float]]:
    r"""$L^2$-UVP on the data support, iJKOnet Eq. (19).

    $$L^2\text{-UVP}(\hat V, V^*) = 100 \cdot
        \frac{\tau^2 \,\mathbb{E}_{x\sim\rho_{k+1}}\,
              \|\nabla\hat V(x) - \nabla V^*(x)\|^2}{\mathrm{Var}(\rho_k)}.$$

    Both gradient callables take ``(N, D)`` and return ``(N, D)``.
    Computes per-time-step (k = 1..T) by taking $\rho_{k+1}$ samples from
    the training data and $\rho_k$ samples for the variance normaliser.

    Returns ``(mean_uvp, {t: uvp_value})``.
    """
    times = np.unique(np.asarray(train_data.t))
    x_np = np.asarray(train_data.x)
    t_np = np.asarray(train_data.t)

    by_time: dict[float, float] = {}
    for k in range(1, len(times)):
        t_prev = float(times[k - 1])
        t_next = float(times[k])
        x_prev = x_np[np.isclose(t_np, t_prev, atol=1e-6)]
        x_next = x_np[np.isclose(t_np, t_next, atol=1e-6)]
        if x_next.shape[0] < 2 or x_prev.shape[0] < 2:
            continue
        var = float(np.trace(np.atleast_2d(np.cov(x_prev, rowvar=False))))
        if var <= 0:
            continue
        g_pred = np.asarray(learned_grad_fn(x_next), dtype=np.float64)
        g_true = np.asarray(true_grad_fn(x_next), dtype=np.float64)
        sq = float(np.mean(np.sum((g_pred - g_true) ** 2, axis=-1)))
        by_time[t_next] = 100.0 * (tau**2) * sq / var
    return _mean(by_time), by_time


# ---------------------------------------------------------------------------
# Functional accuracy (synthetic mode)
# ---------------------------------------------------------------------------


def functional_accuracy(
    learned_V_fn: Callable[[np.ndarray], np.typing.ArrayLike],
    true_V_fn: Callable[[np.ndarray], np.typing.ArrayLike],
    train_data: SpatioTemporalData,
    n_grid: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Compare two potentials on a random grid in the data bounding box.

    Both callables take ``(N, D)`` and return ``(N,)``. ``seed`` fixes the
    random grid (and hence the comparison) for reproducibility. Returns
    ``potential_r2``, ``potential_rmse`` (relative), ``pattern_r2``
    (squared Pearson correlation, scale-invariant), ``pattern_corr``.
    """
    x_np = np.asarray(train_data.x)
    lo = x_np.min(axis=0) - 0.5
    hi = x_np.max(axis=0) + 0.5
    D = x_np.shape[1]
    rng = np.random.default_rng(seed)
    pts = rng.uniform(lo, hi, size=(n_grid, D)).astype(np.float32)

    V_true = np.asarray(true_V_fn(pts), dtype=np.float64)
    V_learned = np.asarray(learned_V_fn(pts), dtype=np.float64)

    V_true_c = V_true - V_true.mean()
    V_learned_c = V_learned - V_learned.mean()

    ss_res = np.sum((V_true_c - V_learned_c) ** 2)
    ss_tot = np.sum(V_true_c**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    val_range = V_true.max() - V_true.min()
    rmse = np.sqrt(np.mean((V_true_c - V_learned_c) ** 2)) / max(val_range, 1e-12)

    denom = float(np.sqrt(np.sum(V_true_c**2) * np.sum(V_learned_c**2)))
    corr = float(np.sum(V_true_c * V_learned_c) / max(denom, 1e-12))
    pattern_r2 = corr * corr

    return {
        "potential_r2": float(r2),
        "potential_rmse": float(rmse),
        "pattern_r2": pattern_r2,
        "pattern_corr": corr,
    }


# ---------------------------------------------------------------------------
# JKOnet*-style one-step-ahead push-forward
# ---------------------------------------------------------------------------


def one_step_ahead(
    functional: WGFFunctional,
    bandwidth: jax.Array,
    test_data: SpatioTemporalData,
    metric: Literal["emd", "w2", "w2_squared"] = "w2",
) -> tuple[float, dict[float, float]]:
    r"""One-step-ahead forward-Euler evaluation under a given functional.

    For each consecutive pair of observed timepoints $(t_k, t_{k+1})$:
    push test particles $x_k$ one explicit JKO Euler step forward under
    $\nabla\delta\mathcal{F}/\delta\rho$ and compare to $x_{k+1}$ via
    $W_p$. Matches jkonet-star's ``error_wasserstein_one_step_ahead``.

    Returns ``(mean_metric, {t_{k+1}: metric_value})``.
    """
    t_unique = np.unique(np.asarray(test_data.t))
    if len(t_unique) < 2:
        return 0.0, {}

    particles_by_t = {
        round(float(t), 8): test_data.x[jnp.isclose(test_data.t, t, atol=1e-6)]
        for t in t_unique
    }
    jit_step = jax.jit(lambda x, dt: jko_proximal_step(functional, x, dt, bandwidth))

    by_time: dict[float, float] = {}
    for i in range(len(t_unique) - 1):
        t_k = round(float(t_unique[i]), 8)
        t_next = round(float(t_unique[i + 1]), 8)
        x_pred = jit_step(particles_by_t[t_k], t_next - t_k)
        x_true = particles_by_t[t_next]
        n = min(x_pred.shape[0], x_true.shape[0])
        by_time[t_next] = wasserstein(x_pred[:n], x_true[:n], metric=metric)
    return _mean(by_time), by_time
