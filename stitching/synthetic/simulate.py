r"""Unified Wasserstein-gradient-flow SDE simulator.

Integrates

$$
dx = -\nabla\frac{\delta\mathcal{F}}{\delta\rho}(x)\,dt + \sigma\,dW
$$

with Euler–Maruyama via `diffrax`.  The drift and diffusion contributions of
$\mathcal{F}$ are extracted by the type-dispatched primitives
`sde_drift` / `sde_diffusion` from `stitching.synthetic.sde`, so any `Functional`
(atomic or composite) is accepted with no additional plumbing.
"""

import diffrax
import jax
import jax.numpy as jnp
import lineax

from stitching._kde import Empirical
from stitching.velocity import WGFFunctional

from . import sde


def simulate_wgf(
    functional: WGFFunctional,
    x0: jax.Array,
    key: jax.Array,
    n_timesteps: int,
    dt: float,
    n_substeps: int = 1,
) -> tuple[jax.Array, jax.Array]:
    r"""Integrate the WGF of *functional* from initial particles *x0*.

    Parameters
    ----------
    functional
        Energy functional driving the flow.  Drift and diffusion contributions
        are derived by dispatch on its type (`stitching.synthetic.sde.drift`,
        `stitching.synthetic.sde.diffusion`).
    x0
        `(N, D)` initial particle positions.
    key
        PRNG key for the Brownian path.
    n_timesteps
        Number of observation intervals.  The trajectory has
        ``n_timesteps + 1`` frames (including $t = 0$); total simulated time
        is ``n_timesteps * dt``.
    dt
        Observation-interval size.
    n_substeps
        Number of Euler–Maruyama substeps per observation interval; internal
        step is ``dt / n_substeps``.

    Returns
    -------
    trajectory, t_grid
        Arrays of shape ``(T+1, N, D)`` and ``(T+1,)``.
    """
    N, D = x0.shape
    t0, t1 = 0.0, float(n_timesteps * dt)
    t_grid = jnp.arange(n_timesteps + 1) * dt
    sub_dt = dt / n_substeps
    sigma = sde.diffusion(functional)

    def drift_vf(t, y, args):
        x = y.reshape(N, D)
        d = sde.drift(functional, Empirical(x))
        return jax.vmap(d)(x).reshape(-1)

    def diffusion_vf(t, y, args):
        return lineax.DiagonalLinearOperator(sigma * jnp.ones_like(y))

    brownian = diffrax.VirtualBrownianTree(
        t0=t0, t1=t1, tol=sub_dt * 0.5, shape=(N * D,), key=key
    )
    terms: diffrax.MultiTerm = diffrax.MultiTerm(
        diffrax.ODETerm(drift_vf),
        diffrax.ControlTerm(diffusion_vf, brownian),
    )

    sol = diffrax.diffeqsolve(
        terms,
        diffrax.Euler(),
        t0=t0,
        t1=t1,
        dt0=sub_dt,
        y0=x0.reshape(-1),
        saveat=diffrax.SaveAt(ts=t_grid),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=n_timesteps * n_substeps + 16,
    )
    trajectory = sol.ys.reshape(n_timesteps + 1, N, D)
    return trajectory, t_grid
