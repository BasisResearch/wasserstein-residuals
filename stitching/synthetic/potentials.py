"""Synthetic potential functions V(x) used by the WGF benchmarks.

Each V is scalar-valued and not vmapped — compose with `jax.grad` / `jax.vmap`
at the call site.  Ported from jkonet-star for benchmarking.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

ScalarFn = Callable[[jax.Array], jax.Array]


# ---------------------------------------------------------------------------
# 2-D potentials
# ---------------------------------------------------------------------------


def styblinski_tang(v: jax.Array) -> jax.Array:
    u = jnp.square(v)
    return 0.5 * jnp.sum(jnp.square(u) - 16 * u + 5 * v)


def holder_table(v: jax.Array) -> jax.Array:
    d = v.shape[0]
    v1 = jnp.mean(v[: d // 2])
    v2 = jnp.mean(v[d // 2 :])
    return 10 * jnp.abs(
        jnp.sin(v1)
        * jnp.cos(v2)
        * jnp.exp(jnp.abs(1 - jnp.sqrt(jnp.sum(jnp.square(v))) / jnp.pi))
    )


def zigzag_ridge(v: jax.Array) -> jax.Array:
    return jnp.sum(
        jnp.abs(v[:-1] - v[1:]) ** 2
        + jnp.cos(v[:-1]) * (v[:-1] + v[1:])
        + v[:-1] ** 2 * v[1:]
    )


def oakley_ohagan(v: jax.Array) -> jax.Array:
    return 5 * jnp.sum(jnp.sin(v) + jnp.cos(v) + jnp.square(v) + v)


def watershed(v: jax.Array) -> jax.Array:
    return jnp.sum(v[:-1] + v[:-1] ** 2 * (v[1:] + 4)) / 10


def ishigami(v: jax.Array) -> jax.Array:
    d = v.shape[0]
    v0 = jnp.mean(v[: d // 2])
    v1 = jnp.mean(v[d // 2 :])
    v2 = (v0 + v1) / 2
    return jnp.sin(v0) + 7 * jnp.sin(v1) ** 2 + 0.1 * v2**4 * jnp.sin(v0)


def friedman(v: jax.Array) -> jax.Array:
    v = 2 * (v - 7)
    d = v.shape[0]
    v1 = jnp.mean(v[: d // 2])
    v2 = jnp.mean(v[d // 2 :]) / 2
    v3 = v1 * jnp.sin(v2)
    v4 = v1 * jnp.cos(v2)
    v5 = v2 * jnp.sin(v1)
    return (
        10 * jnp.sin(jnp.pi * v1 * v2)
        + 20 * (v3 - 0.5) ** 2
        + 10 * (v4 - 1) ** 2
        + 0.1 * v5
    ) / 100


def sphere(v: jax.Array) -> jax.Array:
    return -10 * jnp.sum(jnp.square(v))


def bohachevsky(v: jax.Array) -> jax.Array:
    d = v.shape[0]
    v1 = jnp.mean(v[: d // 2])
    v2 = jnp.mean(v[d // 2 :])
    return 10 * (
        jnp.square(v1)
        + 2 * jnp.square(v2)
        - 0.3 * jnp.cos(3 * jnp.pi * v1)
        - 0.4 * jnp.cos(4 * jnp.pi * v2)
    )


def flowers(v: jax.Array) -> jax.Array:
    return jnp.sum(v + 2 * jnp.sin(jnp.abs(v) ** 1.2))


def wavy_plateau(v: jax.Array) -> jax.Array:
    return jnp.sum(jnp.cos(jnp.pi * v) + 0.5 * v**4 - 3 * v**2 + 1)


def double_exp(v: jax.Array) -> jax.Array:
    s = 20
    d = 3
    return 200 * (
        jnp.exp(-jnp.sum(jnp.square(v - d)) / s)
        + jnp.exp(-jnp.sum(jnp.square(v + d)) / s)
    )


def relu(v: jax.Array) -> jax.Array:
    r = -50 * jnp.clip(v, 0)
    if r.ndim > 0:
        return jnp.sum(r)
    return r


def rotational(v: jax.Array) -> jax.Array:
    d = v.shape[0]
    v1 = jnp.mean(v[: d // 2])
    v2 = jnp.mean(v[d // 2 :])
    theta = jnp.arctan2(v2 + 5, v1 + 5)
    return 10 * relu(theta + jnp.pi)


def flat(v: jax.Array) -> jax.Array:
    return jnp.float32(0.0)


def wavy_valley(v: jax.Array) -> jax.Array:
    r"""Tilted wavy valley used as a Fig. 1 testbed for stitching vs JKO.

    $V(x_1, x_2) = K (x_2 - A\sin(\omega x_1))^2 - \tau\, x_1$ with $A=1$,
    $\omega = \pi/2$, $K=0.6$, $\tau=0.5$. The minimum-energy curve is the
    sinusoid $x_2 = \sin(\pi x_1 / 2)$; the $-x_1$ tilt means $V$ decreases
    in $+x_1$, so particles released near the upper-left valley peak
    (around $(-3, +1)$) fall onto the valley and drift right along it.
    With $\tau=0.5$ and $\beta=0.00625$ at simulation time, particles
    reach $x_1 \approx +4$ over $t=30$, producing the Fig. 1 dataset.
    """
    A, OMEGA, K, TAU = 1.0, jnp.pi / 2, 0.6, 0.5
    x1, x2 = v[0], v[1]
    return K * (x2 - A * jnp.sin(OMEGA * x1)) ** 2 - TAU * x1


# ---------------------------------------------------------------------------
# 1-D potentials
# ---------------------------------------------------------------------------


def doublewell_1d(x: jax.Array) -> jax.Array:
    r"""Double-quadratic well $V(x) = \tfrac12(|x| - 7)^2$.

    Non-negative, zeros at $x = \pm 7$, $V(0) = 24.5$, quadratic growth.
    """
    return jnp.squeeze(0.5 * (jnp.abs(x) - 7.0) ** 2)


# ---------------------------------------------------------------------------
# n-D potentials (parameterised factories)
# ---------------------------------------------------------------------------


def doublewell_nd(alpha: float = 0.1, beta: float = 4.0) -> ScalarFn:
    r"""Radial double-well $V(x) = \alpha(\|x\|^2 - \beta)^2$."""

    def V(x: jax.Array) -> jax.Array:
        return alpha * (jnp.sum(jnp.square(x)) - beta) ** 2

    return V


# ---------------------------------------------------------------------------
# Interaction kernels W(r) where r = x - y
# ---------------------------------------------------------------------------


def gaussian_interaction(strength: float = -1.0, width: float = 1.0) -> ScalarFn:
    r"""Gaussian interaction $W(r) = \text{strength} \cdot \exp(-\|r\|^2 / \text{width}^2)$."""

    def W(r: jax.Array) -> jax.Array:
        return strength * jnp.exp(-jnp.sum(jnp.square(r)) / width**2)

    return W


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Named scalar potentials keyed by their canonical short name.
#: Each entry is a scalar function mapping `(D,) → scalar`.
POTENTIALS_2D: dict[str, ScalarFn] = {
    "styblinski_tang": styblinski_tang,
    "holder_table": holder_table,
    "zigzag_ridge": zigzag_ridge,
    "oakley_ohagan": oakley_ohagan,
    "watershed": watershed,
    "ishigami": ishigami,
    "friedman": friedman,
    "sphere": sphere,
    "bohachevsky": bohachevsky,
    "flowers": flowers,
    "wavy_plateau": wavy_plateau,
    "double_exp": double_exp,
    "relu": relu,
    "rotational": rotational,
    "flat": flat,
    "wavy_valley": wavy_valley,
}

POTENTIALS_1D: dict[str, ScalarFn] = {
    "doublewell": doublewell_1d,
}


# ---------------------------------------------------------------------------
# Spec-driven factories (used by JSON-preset loaders)
# ---------------------------------------------------------------------------

#: Parametric potential factories keyed by spec name.
#: Each factory takes ``**params`` and returns a `ScalarFn`.
POTENTIAL_FACTORIES: dict[str, Callable[..., ScalarFn]] = {
    "DoubleWellPotential": doublewell_nd,
}

#: Parametric interaction-kernel factories keyed by spec name.
INTERACTION_FACTORIES: dict[str, Callable[..., ScalarFn]] = {
    "GaussianInteraction": gaussian_interaction,
}


def make_from_spec(
    spec: dict, registry: dict[str, Callable[..., ScalarFn]]
) -> ScalarFn:
    """Instantiate a callable from ``{"type": ..., "params": ...}``."""
    name = spec["type"]
    factory = registry.get(name)
    if factory is None:
        raise ValueError(f"Unknown spec type {name!r}. Known: {list(registry)}")
    return factory(**spec.get("params", {}))
