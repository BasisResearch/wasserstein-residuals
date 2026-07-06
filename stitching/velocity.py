"""The single seam through which Stitching/Lightspeed obtain their
Wasserstein-gradient machinery — now a **closed form**, no framework.

Stitching's energy functional is

$$F[\\rho] = c_H\\,\\mathcal H[\\rho] + c_V\\,\\mathcal V[V_\\theta] + c_W\\,\\mathcal W[W_\\theta],$$

whose Wasserstein gradient at any query point $y$ is analytic:

$$\\nabla_W F(y) = c_H\\,\\nabla\\log\\rho(y) + c_V\\,\\nabla V_\\theta(y)
                 + c_W\\sum_j w_j\\,\\nabla W_\\theta(y - x_j).$$

The entropy term is the analytic KDE score (:func:`stitching._kde.kde_score`);
the potential and interaction terms are plain ``jax.grad``. There is no generic
Wasserstein autodiff — the velocity is the closed form throughout, with no
framework dependency. This module is the one and only place the velocity is
defined; :mod:`stitching.residual` and the models call :func:`wasserstein_grad`.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from stitching._kde import (
    Curve,
    KernelDensity,
    SpatioTemporalData,
    VectorField,
    integrate,
    kde_score,
)

__all__ = [
    "WGFFunctional",
    "wasserstein_grad",
    "nll",
]


class WGFFunctional(eqx.Module):
    r"""Lightweight spec of $F = c_H\mathcal H + c_V\mathcal V + c_W\mathcal W$.

    Carries just what the closed-form velocity needs: the potential
    ``V(x) -> scalar`` (already closing over time *t* when the model is
    time-conditioned), the composed interaction ``W(r) -> scalar`` (or
    ``None``), and the three scalar coefficients — the same math as a sum of
    potential / entropy / interaction functionals, consumed directly by the
    closed-form velocity, with no framework.
    """

    V: Callable[[jax.Array], jax.Array]
    W: Callable[[jax.Array], jax.Array] | None
    c_H: jax.Array = eqx.field(converter=jnp.asarray)
    c_V: jax.Array = eqx.field(converter=jnp.asarray)
    c_W: jax.Array = eqx.field(converter=jnp.asarray, default=0.0)


def stitching_velocity(
    func: WGFFunctional, dist: KernelDensity
) -> VectorField[jax.Array]:
    r"""Closed-form Wasserstein gradient $y \mapsto \nabla_W F(y)$ at *dist*.

    Evaluates
    $c_H\,\nabla\log\rho(y) + c_V\,\nabla V(y) + c_W\sum_j w_j\nabla W(y-x_j)$
    at an arbitrary query point *y* (not only the support points), so the
    $\alpha>0$ KDE-MC residual and the JKO eval step reuse it.
    """
    gV = jax.grad(func.V)
    gW = jax.grad(func.W) if func.W is not None else None
    centers = dist.centers
    weights = dist.weights

    def vel(y: jax.Array) -> jax.Array:
        v = func.c_H * kde_score(dist, y) + func.c_V * gV(y)
        if gW is not None:
            v = v + func.c_W * jnp.sum(
                weights[:, None] * jax.vmap(lambda xj: gW(y - xj))(centers),
                axis=0,
            )
        return v

    return vel


def wasserstein_grad(
    func: WGFFunctional,
) -> Callable[[KernelDensity], VectorField[jax.Array]]:
    """Return ``dist -> (y -> ∇_W F(dist)(y))`` for the L2 Wasserstein metric.

    Closed form (see :func:`stitching_velocity`) — the entropy term is the
    analytic KDE score, potential and interaction terms are plain ``jax.grad``.
    """
    return lambda dist: stitching_velocity(func, dist)


def nll(data: SpatioTemporalData) -> Callable[[Curve[jax.Array]], jax.Array]:
    r"""Negative log-likelihood: $-\mathbb{E}_{\text{data}}[\log p(x, t)]$.

    Args:
        data: observed spatiotemporal data.

    The returned action takes a `density_curve` mapping time to a
    `Distribution`.
    """

    def _action(density_curve: Curve[jax.Array]) -> jax.Array:
        def integrand(xt: jax.Array) -> jax.Array:
            x, t = xt[:-1], xt[-1]
            return -density_curve(t).log_prob(x)

        return integrate(data, integrand)

    return _action
