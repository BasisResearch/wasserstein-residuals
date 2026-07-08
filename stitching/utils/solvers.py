r"""Particle-flow solver for Wasserstein gradient flow.

- `jko_proximal_step` — single explicit-Euler step on the WGF ODE, used by the
  JKOnet$^\star$ (Lightspeed) forward-evaluation protocol.
"""

import jax

from stitching._kde import IsotropicGaussian, KernelDensity
from stitching.velocity import WGFFunctional, wasserstein_grad


def jko_proximal_step(
    functional: WGFFunctional,
    x_k: jax.Array,
    dt: float,
    bandwidth: jax.Array,
) -> jax.Array:
    r"""One explicit Euler step of the gradient-flow ODE driven by *functional*.

    $$\hat x_{k+1} = x_k - \Delta t\,\nabla\frac{\delta\mathcal{F}}{\delta\rho}\!(x_k)$$

    Matches the predictor used at evaluation time by Terpin et al.'s
    JKOnet$^\star$ reference implementation: a single forward Euler step per
    consecutive observation interval, with no internal substepping. The "chord"
    character of this predictor --- a tangent line drawn from $x_k$ across the
    snapshot gap --- is the bias of JKO-style methods on curved gradient flows.

    Args:
        functional: energy functional driving the WGF.
        x_k: source particles, ``(N, D)``.
        dt: step size.
        bandwidth: KDE bandwidth used to evaluate the Wasserstein gradient
            (``rho_k = KernelDensity(x_k, kernel=IsotropicGaussian(bw=bandwidth))``).
    """
    rho = KernelDensity(x_k, kernel=IsotropicGaussian(bw=bandwidth))
    grad_F = jax.vmap(wasserstein_grad(functional)(rho))
    return x_k - dt * grad_F(x_k)
