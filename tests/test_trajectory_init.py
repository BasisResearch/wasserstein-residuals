"""Unit tests for the trajectory initialisations and their Gaussian OT helpers.

Pins the numeric contracts of the McCann (Bures–Wasserstein) seeding and the
``stitching.utils.ot`` primitives it rests on: the seeded bundle matches the two
terminal Gaussians, each particle path is a straight non-crossing line, the
single-snapshot fallback is static, and the seed is deterministic. The default
``"ot"`` path is covered elsewhere (parity gate); here we pin the new ``"mccann"``
branch and the ``from_data`` dispatch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stitching._kde import SpatioTemporalData
from stitching.models import Stitching
from stitching.models.stitching import mccann_interpolate_trajectories
from stitching.utils.ot import bures_map, gaussian_moments, psd_sqrt

# Two non-commuting anisotropic covariances (different eigenvector orientations),
# the case that separates the interpolation schemes.
_M0 = np.array([-3.0, 1.0])
_C0 = np.array([[0.5, 0.4], [0.4, 0.6]])
_MT = np.array([2.0, -0.5])
_CT = np.array([[1.5, -0.9], [-0.9, 1.2]])


def _two_gaussian_data(
    m0: np.ndarray,
    c0: np.ndarray,
    mt: np.ndarray,
    ct: np.ndarray,
    *,
    n: int = 6000,
    seed: int = 0,
) -> SpatioTemporalData:
    """Snapshots at t=0 and t=1 sampled from ``N(m0,c0)`` and ``N(mt,ct)``."""
    rng = np.random.default_rng(seed)
    d = len(m0)
    x0 = m0 + rng.standard_normal((n, d)) @ np.linalg.cholesky(c0).T
    xt = mt + rng.standard_normal((n, d)) @ np.linalg.cholesky(ct).T
    x = np.concatenate([x0, xt]).astype(np.float32)
    t = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
    return SpatioTemporalData(x=jnp.asarray(x), t=jnp.asarray(t))


# ---------------------------------------------------------------------------
# Gaussian OT helpers
# ---------------------------------------------------------------------------


def test_psd_sqrt_is_symmetric_and_reconstructs() -> None:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    cov = A @ A.T + 0.1 * np.eye(3)  # symmetric PD
    s = psd_sqrt(cov)
    assert np.allclose(s, s.T, atol=1e-10)
    assert np.allclose(s @ s, cov, atol=1e-8)


def test_bures_map_pushes_source_cov_to_target() -> None:
    a = bures_map(_C0, _CT)
    assert np.allclose(a, a.T, atol=1e-8)  # symmetric
    assert np.allclose(a @ _C0 @ a, _CT, atol=1e-6)  # A C0 A = CT
    # A is positive-definite. This is the whole non-braiding guarantee: with A
    # SPD, M(a)=(1-a)I+aA has eigenvalues (1-a)+a·λ_k > 0 for all a in [0,1], so
    # M(a) is never singular and no two particle paths ever coincide.
    assert np.linalg.eigvalsh(a).min() > 0.0


def test_gaussian_moments_matches_numpy_and_guards_singletons() -> None:
    rng = np.random.default_rng(1)
    p = rng.standard_normal((100, 2))
    mu, cov = gaussian_moments(p)
    assert np.allclose(mu, p.mean(axis=0))
    assert np.allclose(cov, np.cov(p, rowvar=False))
    # A single point has no covariance: return zeros of the right shape, not a crash.
    mu1, cov1 = gaussian_moments(p[:1])
    assert cov1.shape == (2, 2)
    assert np.allclose(cov1, 0.0)


# ---------------------------------------------------------------------------
# McCann trajectory init
# ---------------------------------------------------------------------------


def test_mccann_matches_terminal_moments() -> None:
    data = _two_gaussian_data(_M0, _C0, _MT, _CT)
    traj = np.asarray(
        mccann_interpolate_trajectories(
            data, num_particles=4000, num_steps=40, key=jax.random.key(1)
        )[0]
    )
    assert np.allclose(np.cov(traj[0], rowvar=False), _C0, atol=0.06)
    assert np.allclose(np.cov(traj[-1], rowvar=False), _CT, atol=0.08)
    assert np.allclose(traj[0].mean(axis=0), _M0, atol=0.06)
    assert np.allclose(traj[-1].mean(axis=0), _MT, atol=0.06)


def test_mccann_handles_near_singular_terminal() -> None:
    # The motivating case (and the reason for float64 compute + cov_eps): a tight
    # near-degenerate t=0 blob makes C0 near-singular, which bures_map inverts.
    # The output must stay finite and still land on the spread terminal cov CT.
    n = 4000
    rng = np.random.default_rng(5)
    x0 = np.full((n, 2), _M0) + 1e-4 * rng.standard_normal((n, 2))  # ~rank-deficient
    xt = _MT + rng.standard_normal((n, 2)) @ np.linalg.cholesky(_CT).T
    x = np.concatenate([x0, xt]).astype(np.float32)
    t = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.float32)
    data = SpatioTemporalData(x=jnp.asarray(x), t=jnp.asarray(t))
    traj = np.asarray(
        mccann_interpolate_trajectories(
            data, num_particles=3000, num_steps=30, key=jax.random.key(6)
        )[0]
    )
    assert np.isfinite(traj).all()  # no NaN/inf from the near-singular inverse
    assert np.allclose(np.cov(traj[-1], rowvar=False), _CT, atol=0.08)


def test_mccann_is_dimension_general() -> None:
    # psd_sqrt / bures_map / the einsum are all shape-generic; pin D=3 so a
    # 2-D-only assumption can't slip in. Moderate, well-conditioned covariances
    # keep the sampling error small enough for a tight tolerance.
    m0, mt = np.array([0.0, 1.0, -2.0]), np.array([3.0, -1.0, 0.5])
    c0 = np.array([[1.0, 0.3, 0.1], [0.3, 1.2, -0.2], [0.1, -0.2, 0.8]])
    ct = np.array([[1.5, -0.4, 0.2], [-0.4, 1.1, 0.3], [0.2, 0.3, 1.3]])
    data = _two_gaussian_data(m0, c0, mt, ct, n=8000)
    traj = np.asarray(
        mccann_interpolate_trajectories(
            data, num_particles=5000, num_steps=20, key=jax.random.key(10)
        )[0]
    )
    assert traj.shape == (20, 5000, 3)
    assert np.allclose(np.cov(traj[0], rowvar=False), c0, atol=0.08)
    assert np.allclose(np.cov(traj[-1], rowvar=False), ct, atol=0.08)


def test_mccann_paths_are_straight_and_non_crossing() -> None:
    data = _two_gaussian_data(_M0, _C0, _MT, _CT)
    traj = np.asarray(
        mccann_interpolate_trajectories(
            data, num_particles=1500, num_steps=30, key=jax.random.key(2)
        )[0]
    )
    # Straight line: x_i(a) == (1-a) x_i(0) + a x_i(1).
    a = np.linspace(0.0, 1.0, traj.shape[0])
    line = (1 - a)[:, None, None] * traj[0][None] + a[:, None, None] * traj[-1][None]
    assert np.allclose(traj, line, atol=1e-4)
    # Laminar: no two particles coincide at the same time.
    for k in (0, traj.shape[0] // 2, traj.shape[0] - 1):
        dist = np.sqrt(((traj[k][:, None, :] - traj[k][None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        assert dist.min() > 0.0


def test_mccann_single_snapshot_is_static() -> None:
    n = 200
    rng = np.random.default_rng(3)
    x = (_M0 + rng.standard_normal((n, 2)) @ np.linalg.cholesky(_C0).T).astype(
        np.float32
    )
    data = SpatioTemporalData(x=jnp.asarray(x), t=jnp.zeros(n, dtype=jnp.float32))
    traj = np.asarray(
        mccann_interpolate_trajectories(
            data, num_particles=50, num_steps=8, key=jax.random.key(4)
        )[0]
    )
    assert traj.shape == (8, 50, 2)
    assert np.array_equal(traj[0], traj[-1])  # broadcast, no drift


def test_mccann_is_deterministic() -> None:
    data = _two_gaussian_data(_M0, _C0, _MT, _CT)
    a = mccann_interpolate_trajectories(data, 200, 20, jax.random.key(7))[0]
    b = mccann_interpolate_trajectories(data, 200, 20, jax.random.key(7))[0]
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_mccann_return_contract() -> None:
    data = _two_gaussian_data(_M0, _C0, _MT, _CT)
    traj, t_grid, key = mccann_interpolate_trajectories(
        data, num_particles=25, num_steps=12, key=jax.random.key(3)
    )
    assert traj.shape == (12, 25, 2)
    assert traj.dtype == jnp.float32
    assert t_grid.shape == (12,)
    assert np.isclose(float(t_grid[0]), 0.0)
    assert np.isclose(float(t_grid[-1]), 1.0)
    jax.random.normal(key, (2,))  # remaining key is a usable jax key


# ---------------------------------------------------------------------------
# from_data dispatch
# ---------------------------------------------------------------------------


def test_from_data_selects_mccann_init() -> None:
    data = _two_gaussian_data(_M0, _C0, _MT, _CT, n=500)
    model = Stitching.from_data(
        data,
        num_particles=20,
        num_steps=10,
        hidden=(16,),
        key=jax.random.key(0),
        trajectory_init="mccann",
    )
    assert model.trajectories.shape == (10, 20, 2)


def test_from_data_unknown_trajectory_init_raises() -> None:
    data = _two_gaussian_data(_M0, _C0, _MT, _CT, n=500)
    with pytest.raises(ValueError, match="Unknown trajectory_init"):
        Stitching.from_data(
            data,
            num_particles=20,
            num_steps=10,
            hidden=(16,),
            key=jax.random.key(0),
            trajectory_init="bogus",
        )
