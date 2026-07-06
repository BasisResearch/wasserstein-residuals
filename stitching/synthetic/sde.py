r"""SDE components for Wasserstein gradient flows (closed form, framework-free).

A composite functional $\mathcal{F} = c_V\mathcal{V} + c_H\mathcal{H} +
c_W\mathcal{W}$ induces the overdamped Langevin dynamics

$$
dx = -\nabla\frac{\delta\mathcal{F}}{\delta\rho}(x)\,dt + \sigma\,dW,
$$

in which the entropy term contributes **diffusion** rather than drift because
$-\nabla\log\rho$ on an empirical distribution is undefined; the
Fokker–Planck equivalence realises it as isotropic noise $\sigma\,dW$.

Two primitives make this decomposition explicit, given a
:class:`~stitching.velocity.WGFFunctional`:

* :func:`drift` — the drift `VectorField` $x \mapsto -\nabla\delta F/\delta\rho(x)$,
  i.e. $-(c_V\nabla V + c_W\sum_j w_j\nabla W(x - x_j))$ (entropy contributes none).
* :func:`diffusion` — the scalar $\sigma = c_H\sqrt{2}$.

`drift` returns a `VectorField` (`x → (D,)`); callers vmap over particles
themselves. The closed forms use only ``jax.grad`` + the empirical mean — no
``Wasserstein.grad``.
"""

import jax
import jax.numpy as jnp

from stitching._kde import Empirical, VectorField
from stitching.velocity import WGFFunctional

__all__ = ["drift", "diffusion"]


def drift(func: WGFFunctional, dist: Empirical) -> VectorField[jax.Array]:
    r"""Drift field of the WGF of *func* under empirical *dist*:
    $x \mapsto -(c_V\nabla V(x) + c_W\sum_j w_j\nabla W(x - x_j))$.

    Entropy contributes no drift (it enters the SDE as diffusion).
    """
    gV = jax.grad(func.V)
    gW = jax.grad(func.W) if func.W is not None else None
    centers = dist.centers
    weights = dist.weights

    def _d(x: jax.Array) -> jax.Array:
        v = -func.c_V * gV(x)
        if gW is not None:
            v = v - func.c_W * jnp.sum(
                weights[:, None] * jax.vmap(lambda xj: gW(x - xj))(centers),
                axis=0,
            )
        return v

    return _d


def diffusion(func: WGFFunctional) -> jax.Array:
    r"""Isotropic diffusion coefficient $\sigma = c_H\sqrt{2}$ from *func*.

    Matches the framework's quadrature-sum convention exactly (entropy is the
    only term with non-zero diffusion), so regenerated datasets are unchanged.
    """
    return jnp.sqrt((func.c_H * jnp.sqrt(2.0)) ** 2)
