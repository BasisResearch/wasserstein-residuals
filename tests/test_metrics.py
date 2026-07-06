"""Unit tests for the pure, model-free evaluation metrics.

Covers the functions whose value depends only on their point-cloud inputs:
:func:`wasserstein` (exact on a pure translation: W = ‖shift‖), the two MMD
estimators (identical clouds → 0), and :func:`bures_wasserstein_sq` (closed
form on Gaussian-fit moments). The callable-dependent metrics
(``l2_uvp``/``functional_accuracy``/``one_step_ahead``) are left for an
integration test with a real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from stitching.utils.metrics import (
    _moments,
    bures_wasserstein_sq,
    mmd_dmsb,
    mmd_rbf,
    wasserstein,
)

# ---------------------------------------------------------------------------
# Wasserstein — exact on a pure translation (the OT map is the identity shift)
# ---------------------------------------------------------------------------


def _cloud(seed: int = 0, n: int = 64, d: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d))


def test_wasserstein_identical_cloud_is_zero() -> None:
    x = _cloud()
    for metric in ("emd", "w2", "w2_squared"):
        assert wasserstein(x, x, metric=metric) == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_unequal_sizes_pure_translation() -> None:
    # Unequal N vs M is the headline capability (uniform a/b weight vectors of
    # different lengths). Two point masses N=64 at 0 and M=32 at c: any coupling
    # must move all mass a distance ||c||, so W1 = W2 = ||c|| exactly even at N≠M.
    c = np.array([3.0, -4.0])  # ||c|| = 5
    P = np.zeros((64, 2))
    Q = np.broadcast_to(c, (32, 2))
    assert wasserstein(P, Q, metric="emd") == pytest.approx(5.0, rel=1e-6)
    assert wasserstein(P, Q, metric="w2") == pytest.approx(5.0, rel=1e-6)
    assert wasserstein(P, Q, metric="w2_squared") == pytest.approx(25.0, rel=1e-6)


def test_wasserstein_pure_translation_matches_shift_norm() -> None:
    # For y = x + c the optimal coupling is the identity map (the 1-Lipschitz
    # dual potential f(z) = <c/||c||, z> attains the W1 bound; Brenier's map
    # x -> x + c attains W2) — valid for *exact* EMD, not entropic OT. Every
    # point then pays exactly the shift: W1 = W2 = ||c||, and W2^2 = ||c||^2.
    x = _cloud()
    c = np.array([3.0, -4.0])  # ||c|| = 5
    y = x + c
    norm = float(np.linalg.norm(c))
    assert wasserstein(x, y, metric="emd") == pytest.approx(norm, rel=1e-6)
    assert wasserstein(x, y, metric="w2") == pytest.approx(norm, rel=1e-6)
    assert wasserstein(x, y, metric="w2_squared") == pytest.approx(norm**2, rel=1e-6)


# ---------------------------------------------------------------------------
# MMD — a distribution has zero discrepancy from itself
# ---------------------------------------------------------------------------


def test_mmd_rbf_identical_cloud_is_zero() -> None:
    x = _cloud()
    assert mmd_rbf(x, x) == pytest.approx(0.0, abs=1e-9)


def test_mmd_rbf_separated_clouds_is_positive() -> None:
    x = _cloud(seed=0)
    y = _cloud(seed=1) + 50.0  # far apart → cross-kernel ≈ 0
    # MMD^2 then collapses to the self-terms (~2/B per bandwidth) >> 0.1.
    assert mmd_rbf(x, y) > 0.1


def test_mmd_dmsb_identical_cloud_is_zero() -> None:
    x = _cloud()
    assert mmd_dmsb(x, x) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Bures–Wasserstein — closed form on the Gaussian fit
# ---------------------------------------------------------------------------


def test_bures_identical_is_zero() -> None:
    P = _cloud()
    assert bures_wasserstein_sq(P, P) == pytest.approx(0.0, abs=1e-9)


def test_bures_pure_mean_shift_is_squared_distance() -> None:
    # Same covariance, means offset by c → Bd^2 = ||c||^2 (the trace term
    # vanishes because both Gaussian fits share a covariance).
    P = _cloud()
    c = np.array([1.5, -2.0])
    Q = P + c
    assert bures_wasserstein_sq(P, Q) == pytest.approx(float(c @ c), rel=1e-6)


def test_bures_one_dim_variance_gap_matches_closed_form() -> None:
    # 1-D, zero mean: with ddof=1, np.cov of two points {-a, a} returns 2a^2,
    # so the Bures gap is (sqrt(2)|a| - sqrt(2)|b|)^2 = 2(|a| - |b|)^2. Here
    # a=1, b=2 — and this discriminates ddof=1 (gap 2) from ddof=0 (gap 1).
    P = np.array([[-1.0], [1.0]])
    Q = np.array([[-2.0], [2.0]])
    expected = 2.0 * (1.0 - 2.0) ** 2
    assert bures_wasserstein_sq(P, Q) == pytest.approx(expected, rel=1e-6)


def _cloud_with_cov(cov: np.ndarray) -> np.ndarray:
    # Zero-mean point set whose sample covariance (np.cov, ddof=1) equals `cov`
    # exactly: 2 antipodal points ±a_k·v_k per eigenpair (λ_k, v_k), with
    # a_k = sqrt(λ_k·(N-1)/2) and N = 2D so the ddof=1 normaliser is exact.
    w, V = np.linalg.eigh(cov)
    n = 2 * cov.shape[0]
    scale = np.sqrt(w * (n - 1) / 2.0)
    cols = V * scale  # (D, D), column k = a_k v_k
    return np.concatenate([cols.T, -cols.T], axis=0)  # (2D, D)


def test_bures_anisotropic_non_commuting_matches_scipy_oracle() -> None:
    # The previous Bures tests all use Σ_P = Σ_Q (commuting), so the symmetric
    # sandwich (Σ_P^½ Σ_Q Σ_P^½)^½ — the only non-trivial line, and the one
    # bw2_uvp exercises on real anisotropic clouds — is never run. Pin it
    # against an independent scipy.linalg.sqrtm oracle on NON-commuting covs.
    from scipy.linalg import sqrtm

    Sp = np.array([[4.0, 0.0], [0.0, 1.0]])
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Sq = R @ np.array([[1.0, 0.0], [0.0, 3.0]]) @ R.T
    assert not np.allclose(Sp @ Sq, Sq @ Sp)  # genuinely non-commuting

    P = _cloud_with_cov(Sp)
    Q = _cloud_with_cov(Sq)
    Sp_half = sqrtm(Sp).real
    cross = sqrtm(Sp_half @ Sq @ Sp_half).real
    expected = float(np.trace(Sp) + np.trace(Sq) - 2.0 * np.trace(cross))
    assert bures_wasserstein_sq(P, Q) == pytest.approx(expected, rel=1e-6)


def test_bures_is_symmetric_on_non_commuting_covariances() -> None:
    # Bd^2 is mathematically symmetric, but the implementation sandwiches with
    # Σ_P only — so symmetry is a real assertion, not a tautology.
    Sp = np.array([[4.0, 0.0], [0.0, 1.0]])
    Sq = np.array([[2.0, 1.2], [1.2, 3.0]])
    P, Q = _cloud_with_cov(Sp), _cloud_with_cov(Sq)
    assert bures_wasserstein_sq(P, Q) == pytest.approx(
        bures_wasserstein_sq(Q, P), rel=1e-6
    )


# ---------------------------------------------------------------------------
# _moments — importance-weighted mean/covariance (the bw2_uvp path)
# ---------------------------------------------------------------------------


def test_moments_weighted_matches_analytic() -> None:
    # bw2_uvp fits the predicted Gaussian with model mixture weights; pin the
    # weighted mean/cov against a hand computation. P = {0, 2}, w = {0.75, 0.25}
    # → mean = 0.5; cov = Σ w_i (x_i-μ)^2 = .75·.25 + .25·2.25 = 0.75.
    P = np.array([[0.0], [2.0]])
    w = np.array([0.75, 0.25])
    mu, cov = _moments(P, w)
    assert mu[0] == pytest.approx(0.5)
    assert cov[0, 0] == pytest.approx(0.75)
    # Weights are normalised internally, so a global rescale changes nothing.
    mu2, cov2 = _moments(P, 10.0 * w)
    assert mu2[0] == pytest.approx(0.5)
    assert cov2[0, 0] == pytest.approx(0.75)
