"""Pairwise log-kernels for KDE.

A `Kernel` is a *raw* pairwise similarity function: it evaluates
$\\log k(x, y)$ for a single pair, with no normalisation. A
`DensityKernel` adds the log normalising constant $\\log Z = \\log\\int
k(x, y)\\,dx$ on top, so that $\\phi(x \\mid y) = k(x, y) / Z$ is a proper
density component. Mixture aggregation, weights, sampling — all of that
lives downstream in `KernelDensity` (in `stitching._kde.distributions`).

Stitching uses isotropic (per-dimension) Gaussian bandwidths only, so this
module ships the `IsotropicGaussian` kernel and the Silverman bandwidth rule.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Bandwidth utilities
# ---------------------------------------------------------------------------


def silverman_bandwidth(x: jax.Array) -> jax.Array:
    r"""Per-dimension Silverman rule-of-thumb bandwidth.

    $$
    h_d = 1.06\,\hat\sigma_d\,N^{-1/(D+4)}
    $$

    Works with both JAX arrays (inside JIT) and NumPy arrays (model init).

    Args:
        x: `(N, D)` data matrix.

    Returns:
        `(D,)` array of per-dimension bandwidths, clipped to `≥ 1e-6`.
    """
    N, D = x.shape
    sigma = jnp.std(x, axis=0)
    return jnp.maximum(1.06 * sigma * N ** (-1.0 / (D + 4.0)), 1e-6)


def chunked_vmap(
    f: Callable[[jax.Array], jax.Array],
    xs: jax.Array,
    chunk_size: int,
) -> jax.Array:
    """Apply *f* to each row of *xs* using vmap in chunks.

    Equivalent to ``jax.vmap(f)(xs)`` but materialises only
    ``chunk_size`` rows at a time, reducing peak memory from
    ``O(N * work_per_call)`` to ``O(chunk_size * work_per_call)``.

    *xs* is padded to a multiple of *chunk_size*; padding rows are
    discarded in the output.
    """
    N = xs.shape[0]
    pad_n = (-N) % chunk_size
    if pad_n:
        xs = jnp.concatenate([xs, jnp.zeros((pad_n, *xs.shape[1:]))])
    chunks = xs.reshape(-1, chunk_size, *xs.shape[1:])
    out = jax.lax.map(lambda chunk: jax.vmap(f)(chunk), chunks)
    return out.reshape(-1, *out.shape[2:])[:N]


# ---------------------------------------------------------------------------
# Kernel base classes
# ---------------------------------------------------------------------------


class Kernel(eqx.Module):
    r"""Pairwise log-kernel: raw $\log k(x, y)$ for a single pair.

    A `Kernel` is a *raw* similarity function — no probabilistic
    connotation, no normalisation. Subclasses implement
    `log_kernel(x, y) -> scalar`; the value form
    `__call__(x, y) = exp(log_kernel)` is derived.

    Per-particle and per-snapshot parameter axes are peeled lazily by
    `at(idx)`. Subclasses with such parameters override `at` to return
    a kernel with the leading axis sliced. `KernelDensity` vmaps
    `kernel.at(i).log_kernel(centers[i], y)` over `i`.
    """

    def log_kernel(self, x: jax.Array, y: jax.Array) -> jax.Array:
        raise NotImplementedError

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.exp(self.log_kernel(x, y))

    def at(self, idx) -> Self:
        """Peel one leading axis off any per-particle / per-snapshot
        parameter. Default identity — override in subclasses whose
        parameters carry such an axis."""
        return self

    def interp_at(self, alpha, idx_left, idx_right) -> Self:
        """Linearly interpolate between two snapshots along the leading
        axis. Default identity — override in subclasses with a leading
        snapshot axis."""
        return self


class DensityKernel(Kernel):
    r"""A kernel that can back a normalised density.

    Adds the log normalising constant $\log Z = \log\int k(x, y)\,dx$ on
    top of the raw `log_kernel`. The normalised component is
    $\phi(x \mid y) = k(x, y) / Z$, i.e.
    $\log\phi = \log k(x, y) - \log Z$; `KernelDensity` builds its
    mixture from these.

    The signature takes the ambient dimension `dim` because $Z$ depends
    on it (e.g. the Gaussian $(2\pi)^{D/2}$ factor) and a scalar
    bandwidth carries no dimension of its own.
    """

    def log_normalizer(self, dim: int) -> jax.Array:
        r"""$\log Z = \log\int k(x, y)\,dx$ for this kernel's parameters."""
        raise NotImplementedError

    def sample(self, center: jax.Array, key: jax.Array) -> jax.Array:
        r"""Draw one sample from the normalised component density
        $\phi(\cdot \mid \text{center}) = k(\text{center}, \cdot) / Z$.

        Per-particle / per-snapshot parameters are assumed already peeled
        via `at(idx)`, so `self` carries a single component's parameters
        and `center` is `(D,)`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no sampler; sampling a "
            "KernelDensity built on it is unavailable."
        )


# ---------------------------------------------------------------------------
# Concrete kernel
# ---------------------------------------------------------------------------


class IsotropicGaussian(DensityKernel):
    r"""Diagonal-bandwidth Gaussian kernel.

    Raw kernel $k(x, y) = \exp(-\tfrac12 \lVert (y-x)/h \rVert^2)$
    (unnormalised); the normaliser $\log Z = \sum_d \log h_d +
    \tfrac{D}{2}\log 2\pi$ lives in `log_normalizer`.

    `bw` shapes:

    - `()` / `(1,)` / `(D,)`: global — `log_kernel(x, y)` is well-defined
    - `(N, D)`: per-particle — `at(i)` peels axis 0
    - `(T, N, D)` (inside `InterpolationCurve`): per-snapshot
      per-particle — `at(t)` peels then `at(i)` peels again

    `(N,)` is **rejected**: it would pass through `at(i)` unchanged
    (ndim < 2) and silently broadcast to `(D,)` only when `N == D`.
    Use `(N, D)` for per-particle isotropic instead.
    """

    bw: jax.Array = eqx.field(converter=jnp.asarray)

    def _checked_bw(self, D: int) -> jax.Array:
        bw_shape = self.bw.shape
        # Static shape check — runs once at trace time, turns opaque
        # broadcast failures (the (N,) footgun) into a useful message.
        if bw_shape not in ((), (1,), (D,)):
            raise ValueError(
                f"IsotropicGaussian requires bw of shape (), (1,), or "
                f"(D,) where D={D}; got {bw_shape}. For per-particle "
                f"bandwidths pass shape (N, D); `KernelDensity` peels "
                f"axis 0 via `at(i)`."
            )
        return jnp.broadcast_to(self.bw, (D,))

    def log_kernel(self, x: jax.Array, y: jax.Array) -> jax.Array:
        bw = self._checked_bw(x.shape[-1])
        return -0.5 * jnp.sum(((y - x) / bw) ** 2)

    def log_normalizer(self, dim: int) -> jax.Array:
        bw = self._checked_bw(dim)
        return jnp.sum(jnp.log(bw)) + 0.5 * dim * jnp.log(2.0 * jnp.pi)

    def at(self, idx) -> IsotropicGaussian:
        if self.bw.ndim >= 2:
            return IsotropicGaussian(bw=self.bw[idx])
        return self

    def sample(self, center: jax.Array, key: jax.Array) -> jax.Array:
        bw = self._checked_bw(center.shape[-1])
        return center + jax.random.normal(key, center.shape) * bw
