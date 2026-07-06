"""Velocity parity: the shipped closed-form Wasserstein gradient must match an
independent autodiff oracle to a tight tolerance — "tested == shipped".

The oracle differentiates the first-variation potential
``Φ(y) = c_H·log p(y) + c_V·V(y) + c_W·Σ_j w_j·W(y − x_j)`` with ``jax.grad`` — a
different route from the shipped closed form (analytic KDE score + explicit
``Σ w_j jax.grad(W)``), so it genuinely checks the assembly, signs, and
coefficients. No framework dependency.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from stitching._kde import IsotropicGaussian, KernelDensity, kde_score
from stitching.velocity import WGFFunctional, wasserstein_grad

_RTOL, _ATOL = 1e-5, 1e-6


def _first_variation_grad(func: WGFFunctional, dist: KernelDensity):
    """Independent oracle: ``∇_W F(y) = ∇_y Φ(y)`` for the first-variation
    potential ``Φ(y) = c_H·log p(y) + c_V·V(y) + c_W·Σ_j w_j·W(y − x_j)``,
    obtained by autodiff of the (log-)density — not the shipped analytic score."""
    centers, weights = dist.centers, dist.weights

    def phi(y: jax.Array) -> jax.Array:
        out = func.c_H * dist.log_prob(y) + func.c_V * func.V(y)
        if func.W is not None:
            W = func.W  # bind so mypy narrows None away inside the lambda
            out = out + func.c_W * jnp.sum(
                weights * jax.vmap(lambda xj: W(y - xj))(centers)
            )
        return out

    return jax.grad(phi)


def _kde(centers, bw, log_weights=None):
    return KernelDensity(
        centers, kernel=IsotropicGaussian(bw=bw), log_weights=log_weights
    )


def _quadratic_V(scale=0.7):
    return lambda x: scale * jnp.sum(x**2)


def _radial_W():
    # radial: W(r) depends on ‖r‖² (matches build_functional's radial branch)
    return lambda r: jnp.exp(-jnp.sum(r**2))


def _vector_W():
    return lambda r: jnp.sum(jnp.sin(r))


@pytest.fixture
def cloud():
    key = jax.random.key(0)
    centers = jax.random.normal(key, (12, 2))
    bw = jnp.array([0.4, 0.6])
    log_weights = jax.nn.log_softmax(jax.random.normal(jax.random.key(1), (12,)))
    return centers, bw, log_weights


def test_kde_score_matches_autodiff(cloud):
    """Analytic `kde_score` == autodiff `KernelDensity.score` (the entropy term)."""
    centers, bw, log_weights = cloud
    stitch = _kde(centers, bw, log_weights)
    ys = jnp.concatenate([centers, centers + 0.3])  # on- and off-centre
    analytic = jax.vmap(lambda y: kde_score(stitch, y))(ys)
    autodiff = jax.vmap(stitch.score)(ys)
    assert jnp.allclose(analytic, autodiff, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "name,W,c_W",
    [
        ("potential+entropy", None, 0.0),
        ("radial-interaction", _radial_W(), 0.5),
        ("vector-interaction", _vector_W(), 0.3),
    ],
)
def test_velocity_matches_oracle(cloud, name, W, c_W):
    """Closed-form `wasserstein_grad` == autodiff of the first-variation potential."""
    centers, bw, log_weights = cloud
    func = WGFFunctional(V=_quadratic_V(), W=W, c_H=0.8, c_V=1.2, c_W=c_W)

    stitch = _kde(centers, bw, log_weights)
    vel_closed = wasserstein_grad(func)(stitch)
    vel_oracle = _first_variation_grad(func, stitch)

    ys = jnp.concatenate([centers, centers + 0.25])  # on- and off-centre queries
    got = jax.vmap(vel_closed)(ys)
    ref = jax.vmap(vel_oracle)(ys)
    assert jnp.allclose(got, ref, rtol=_RTOL, atol=_ATOL), (
        f"{name}: max|Δ|={float(jnp.max(jnp.abs(got - ref))):.2e}"
    )


def test_time_conditioned_potential(cloud):
    """A time-closed-over V(x) is handled identically (time-conditioned case)."""
    centers, bw, log_weights = cloud
    t = jnp.array([0.3])
    V = lambda x: jnp.sum((x - t) ** 2)  # V closes over t, as build_functional does
    func = WGFFunctional(V=V, W=None, c_H=0.5, c_V=1.0, c_W=0.0)

    stitch = _kde(centers, bw, log_weights)
    got = jax.vmap(wasserstein_grad(func)(stitch))(centers)
    ref = jax.vmap(_first_variation_grad(func, stitch))(centers)
    assert jnp.allclose(got, ref, rtol=_RTOL, atol=_ATOL)


def test_shipped_model_functional_parity():
    """End-to-end: a real Stitching model's `functional()` matches the oracle."""
    import numpy as np

    from stitching._kde import SpatioTemporalData
    from stitching.models import Stitching

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(60, 2)), dtype=jnp.float32)
    t = jnp.asarray(np.repeat([0.0, 1.0, 2.0], 20), dtype=jnp.float32)
    data = SpatioTemporalData(x=x, t=t)
    model = Stitching.from_data(
        data,
        num_particles=20,
        num_steps=5,
        hidden=(16,),
        key=jax.random.key(0),
        interaction_coeff=0.5,
        interaction_type="radial",
        interaction_hidden=(16,),
    )
    func = model.functional()
    centers = model.trajectories[0]
    bw = model.bandwidth
    stitch = KernelDensity(centers, kernel=IsotropicGaussian(bw=bw))

    got = jax.vmap(wasserstein_grad(func)(stitch))(centers)
    ref = jax.vmap(_first_variation_grad(func, stitch))(centers)
    assert jnp.allclose(got, ref, rtol=1e-4, atol=1e-5)
