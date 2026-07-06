"""Unit tests for the trajectory Lagrangian residuals.

Pins the pure discretisation/normalisation helpers (:func:`_discretise`,
:func:`_normalize_residual`), shows that :func:`wgf_residual` vanishes on a
trajectory that exactly solves the discrete WGF ODE, and characterises the
``StitchingLoss`` consistency term as the direct WGF residual.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from stitching._kde import InterpolationCurve, IsotropicGaussian, SpatioTemporalData
from stitching.models import Stitching, StitchingLoss
from stitching.residual import _discretise, _normalize_residual, wgf_residual
from stitching.velocity import WGFFunctional

# ---------------------------------------------------------------------------
# _discretise — schemes pick the right grid samples of the WGF velocity
# ---------------------------------------------------------------------------


def _linear_wg(t: jax.Array) -> jax.Array:
    # WGF velocity = t at every particle (N=2, D=1), i.e. *linear in t*. The
    # linearity is what makes midpoint and trapezoidal agree exactly below.
    return t * jnp.ones((2, 1))


def test_discretise_schemes_select_expected_samples() -> None:
    times = jnp.array([0.0, 1.0, 2.0])
    # trapez: per-interval average of the endpoints → [0.5, 1.5].
    trapez = _discretise(times, _linear_wg, "trapez")[:, 0, 0]
    assert jnp.allclose(trapez, jnp.array([0.5, 1.5]))
    # euler: left endpoint; implicit: right endpoint.
    euler = _discretise(times, _linear_wg, "euler")[:, 0, 0]
    assert jnp.allclose(euler, jnp.array([0.0, 1.0]))
    implicit = _discretise(times, _linear_wg, "implicit")[:, 0, 0]
    assert jnp.allclose(implicit, jnp.array([1.0, 2.0]))
    # midpoint == trapez for a linear field.
    midpoint = _discretise(times, _linear_wg, "midpoint")[:, 0, 0]
    assert jnp.allclose(midpoint, jnp.array([0.5, 1.5]))


def test_discretise_unknown_scheme_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scheme"):
        _discretise(jnp.array([0.0, 1.0]), _linear_wg, "rk4")


# ---------------------------------------------------------------------------
# _normalize_residual — raw vs relative weighted reductions
# ---------------------------------------------------------------------------


def test_normalize_residual_raw_is_weighted_square_norm() -> None:
    r = jnp.array([[[3.0], [4.0]]])  # (T_out=1, N=2, D=1) -> sq = [9, 16]
    weights = jnp.array([0.5, 0.5])
    out = _normalize_residual(r, r, r, weights, "raw")
    assert out[0] == pytest.approx(0.5 * 9 + 0.5 * 16)  # 12.5


def test_normalize_residual_relative_divides_by_dx_force_energy() -> None:
    r = jnp.array([[[3.0], [4.0]]])  # sq = [9, 16]
    dx = jnp.ones((1, 2, 1))
    force = jnp.ones((1, 2, 1))  # denom = ||dx||^2 + ||f||^2 = 1 + 1 = 2
    weights = jnp.array([0.5, 0.5])
    out = _normalize_residual(r, dx, force, weights, "relative")
    assert out[0] == pytest.approx(0.5 * (9 / 2) + 0.5 * (16 / 2))  # 6.25
    # stop_gradient is identity on the forward pass, so only the (untested
    # here) gradient differs; the reported value must match plain "relative".
    out_sg = _normalize_residual(r, dx, force, weights, "relative-sg")
    assert out_sg[0] == pytest.approx(float(out[0]))


def test_normalize_residual_unknown_norm_raises() -> None:
    r = jnp.zeros((1, 2, 1))
    with pytest.raises(ValueError, match="Unknown norm"):
        _normalize_residual(r, r, r, jnp.array([0.5, 0.5]), "l1")


# ---------------------------------------------------------------------------
# wgf_residual — vanishes on an exact discrete gradient-flow trajectory
# ---------------------------------------------------------------------------


def _quadratic_flow_functional() -> WGFFunctional:
    # Entropy off (c_H=0), pure quadratic potential V(x) = 1/2 ||x||^2 so the
    # Wasserstein gradient is the closed form ∇V(y) = y. The WGF ODE is then
    # ẋ = -x, whose forward-Euler step is x_{k+1} = (1 - h) x_k.
    return WGFFunctional(
        V=lambda x: 0.5 * jnp.sum(x**2), W=None, c_H=0.0, c_V=1.0, c_W=0.0
    )


def _euler_flow_curve(h: float = 0.1, T: int = 5) -> InterpolationCurve:
    x0 = jnp.array([[-1.0], [0.5], [2.0]])  # (N=3, D=1)
    steps = (1.0 - h) ** jnp.arange(T)  # exact forward-Euler decay factors
    centers = steps[:, None, None] * x0  # (T, N, D)
    t_grid = h * jnp.arange(T)
    return InterpolationCurve(centers, t_grid, kernel=IsotropicGaussian(bw=0.5))


def test_wgf_residual_vanishes_on_euler_gradient_flow() -> None:
    func = _quadratic_flow_functional()
    curve = _euler_flow_curve()
    # Forward-Euler scheme matches the trajectory's own discretisation, so the
    # displacement residual r = Δx + h·∇V is exactly zero.
    res = wgf_residual(lambda _t: func, scheme="euler", norm="raw")(curve)
    assert float(res) == pytest.approx(0.0, abs=1e-6)


def test_wgf_residual_positive_on_perturbed_trajectory() -> None:
    func = _quadratic_flow_functional()
    curve = _euler_flow_curve()
    bad = InterpolationCurve(
        curve.centers + 1.0, curve.t_grid, kernel=IsotropicGaussian(bw=0.5)
    )
    res = wgf_residual(lambda _t: func, scheme="euler", norm="raw")(bad)
    assert float(res) > 1e-3


# ---------------------------------------------------------------------------
# StitchingLoss dispatch — the consistency term is the direct WGF residual
# ---------------------------------------------------------------------------


def _tiny_stitching_model() -> tuple[Stitching, SpatioTemporalData]:
    rng = jax.random.key(0)
    x = jax.random.normal(jax.random.key(1), (40, 1))
    t = jnp.repeat(jnp.array([0.0, 1.0]), 20)
    data = SpatioTemporalData(x=x, t=t)
    model = Stitching.from_data(
        data, num_particles=8, num_steps=4, hidden=(16,), key=rng
    )
    return model, data


def test_consistency_equals_direct_wgf_residual() -> None:
    # With residual_alpha=0 the consistency term carries no extra additive or
    # scaling component, so it must *equal* (not merely track) a bare
    # wgf_residual on the same functional/curve; this pins that dispatch. The
    # white-box reach into the private `_functional_at` is intentional.
    model, data = _tiny_stitching_model()
    loss = StitchingLoss(
        w_nll=1.0,
        w_consistency=1.0,
        scheme="trapez",
        norm="raw",
        kinetic=1.0,
        residual_alpha=0.0,
    )
    _, aux = loss(model, data, jax.random.key(2))

    functional_at = loss._functional_at(model)
    direct = wgf_residual(functional_at, scheme="trapez", norm="raw", kinetic=1.0)(
        model.density_curve()
    )
    assert float(aux["residual"]) == pytest.approx(float(direct), rel=1e-6)
