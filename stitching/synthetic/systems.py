"""The `SyntheticSystem` abstraction: a Wasserstein-gradient-flow specification.

A `SyntheticSystem` bundles a `Functional` (atomic or a `SumFunctional` of
potential / entropy / interaction terms) with the ambient dimension and the
initial-distribution sampler needed to simulate it.  Drift and diffusion are
derived by type-dispatch (`stitching.synthetic.sde`), so the simulator path is single
and unified.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from stitching.velocity import WGFFunctional

from .potentials import POTENTIALS_1D, POTENTIALS_2D
from .simulate import simulate_wgf

# ---------------------------------------------------------------------------
# Initial distributions
# ---------------------------------------------------------------------------


def uniform_box(
    low: float = -4.0, high: float = 4.0
) -> Callable[[jax.Array, int, int], jax.Array]:
    """Initial sampler: `Uniform([low, high])^D`."""

    def sample(key: jax.Array, n: int, dim: int) -> jax.Array:
        return jax.random.uniform(key, (n, dim), minval=low, maxval=high)

    return sample


def isotropic_normal(
    loc: float | jax.Array = 0.0, scale: float = 1.0
) -> Callable[[jax.Array, int, int], jax.Array]:
    """Initial sampler: `N(loc, scale² · I)`; *loc* is a scalar or per-dim vector."""

    def sample(key: jax.Array, n: int, dim: int) -> jax.Array:
        return loc + scale * jax.random.normal(key, (n, dim))

    return sample


# ---------------------------------------------------------------------------
# SyntheticSystem
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SyntheticSystem:
    r"""A synthetic Wasserstein-gradient-flow setup.

    The dynamics are
    $$
    dx = -\nabla\frac{\delta\mathcal{F}}{\delta\rho}(x)\,dt + \sigma\,dW,
    \quad \sigma = \sqrt{2\,c_{\text{entropy}}},
    $$
    where $\mathcal{F}$ is a sum of `PotentialFunctional`, `EntropyFunctional`
    and `InteractionFunctional` terms.  Entropy terms contribute diffusion
    noise; other terms contribute to the deterministic drift.

    Parameters
    ----------
    name
        Short human-readable identifier, used for error messages and plots.
    functional
        The full energy functional $\mathcal{F}$.  Any mix of potential,
        entropy and interaction terms is supported.
    dim
        Ambient dimension $D$.
    init
        Callable `(key, n, dim) → (n, D) jax.Array` that samples the initial
        particle cloud.  Defaults to `Uniform([-4, 4])^D`.
    """

    name: str
    functional: WGFFunctional
    dim: int = 2
    init: Callable[[jax.Array, int, int], jax.Array] = field(
        default_factory=lambda: uniform_box(-4.0, 4.0)
    )

    def simulate(
        self,
        key: jax.Array,
        n_particles: int = 2000,
        n_timesteps: int = 5,
        dt: float = 0.01,
        n_substeps: int = 1,
    ) -> tuple[jax.Array, jax.Array]:
        """Simulate the WGF trajectory.

        Returns `(trajectory[T+1, N, D], t_grid[T+1])`.
        """
        k_init, k_sde = jax.random.split(key)
        x0 = self.init(k_init, n_particles, self.dim)
        return simulate_wgf(self.functional, x0, k_sde, n_timesteps, dt, n_substeps)


# ---------------------------------------------------------------------------
# System lookup — builds on-the-fly from the potential registries
# ---------------------------------------------------------------------------

_ALL_POTENTIALS: dict[str, tuple[Callable, int]] = {
    **{name: (V, 2) for name, V in POTENTIALS_2D.items()},
    **{name: (V, 1) for name, V in POTENTIALS_1D.items()},
}


def list_systems() -> list[str]:
    """Return the sorted names of all available synthetic systems."""
    return sorted(_ALL_POTENTIALS)


def get_system(name: str) -> SyntheticSystem:
    """Build a `SyntheticSystem` for the named potential.

    Returns a bare potential-only system.  Entropy, initial distribution,
    and other simulation parameters are controlled by the experiment config.

    Legacy dataset strings prefixed with ``synthetic-`` are accepted for
    backward compatibility.
    """
    if name.startswith("synthetic-"):
        name = name[len("synthetic-") :]
    if name not in _ALL_POTENTIALS:
        raise KeyError(f"Unknown synthetic system {name!r}. Known: {list_systems()}")
    V, dim = _ALL_POTENTIALS[name]
    return make_system(name=name, V=V, dim=dim)


# ---------------------------------------------------------------------------
# Ad-hoc system builder for arbitrary user-specified functionals
# ---------------------------------------------------------------------------


def make_system(
    *,
    name: str = "custom",
    V: Callable | None = None,
    V_coeff: float = 1.0,
    entropy_coeff: float = 0.0,
    W: Callable | None = None,
    W_coeff: float = 0.0,
    dim: int = 2,
    init: Callable[[jax.Array, int, int], jax.Array] | None = None,
) -> SyntheticSystem:
    """Build a `SyntheticSystem` from coefficient-weighted components.

    Any of `V`, `entropy_coeff`, `W` may be omitted; the corresponding term is
    simply dropped (its coefficient set to zero).  The resulting system is ready
    to simulate and suitable as the ground truth for a fitting experiment. The
    entropy coefficient enters the SDE as diffusion ($\\sigma = c_H\\sqrt 2$),
    not drift.
    """
    if V is None and entropy_coeff <= 0.0 and W is None:
        raise ValueError(
            "make_system requires at least one of V, entropy_coeff, W to be set."
        )
    functional = WGFFunctional(
        V=V if V is not None else (lambda x: jnp.asarray(0.0)),
        W=W,
        c_H=jnp.asarray(entropy_coeff),
        c_V=jnp.asarray(V_coeff if V is not None else 0.0),
        c_W=jnp.asarray(W_coeff if W is not None else 0.0),
    )
    return SyntheticSystem(
        name=name,
        functional=functional,
        dim=dim,
        init=init if init is not None else uniform_box(-4.0, 4.0),
    )
