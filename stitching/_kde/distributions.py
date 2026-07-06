"""Concrete distribution types: `Empirical`, `KernelDensity`, plus their
`integrate` registrations and the analytic `kde_score`.

Pairwise log-kernels live in `stitching._kde.kernel`; `KernelDensity` turns a
`DensityKernel` into a normalised mixture and owns all the mixture math.
Mixture sampling picks a component then defers the per-component draw to the
kernel's `sample`. Time-axis types (`SpatioTemporalData`,
`InterpolationCurve`) live in `stitching._kde.curves`.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ._core import integrate
from .kernel import DensityKernel, IsotropicGaussian, chunked_vmap


class Empirical(eqx.Module):
    """Weighted empirical distribution (mixture of Diracs).

    Carries `centers` and `log_weights` directly. Satisfies the
    `Distribution` protocol at the type level; `log_prob` raises
    (point masses have no smooth density). Use `KernelDensity` when
    callers need `log_prob`.
    """

    centers: jax.Array
    log_weights: jax.Array

    def __init__(
        self,
        centers: jax.Array,
        log_weights: jax.Array | None = None,
    ):
        self.centers = centers
        if log_weights is None:
            N = centers.shape[0]
            self.log_weights = jnp.full(
                (N,),
                -jnp.log(N),
                dtype=centers.dtype,
            )
        else:
            self.log_weights = log_weights

    @property
    def weights(self) -> jax.Array:
        """Mixture weights $w_k$ in the simplex (linear scale)."""
        return jnp.exp(self.log_weights)

    def log_prob(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError("Empirical distribution has no smooth density")

    def sample(self, *, key: jax.Array, n: int) -> jax.Array:
        idx = jax.random.choice(
            key,
            self.centers.shape[0],
            shape=(n,),
            p=self.weights,
        )
        return self.centers[idx]


@integrate.register(Empirical)
def _expectation_empirical(  # type: ignore[misc]
    dist: Empirical, f: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    r"""$\sum_i w_i f(x_i)$ — weighted sum over particles."""
    return jnp.sum(jax.vmap(lambda w, x: w * f(x))(dist.weights, dist.centers), axis=0)


class KernelDensity(eqx.Module):
    r"""Gaussian kernel density $\hat p(y) = \sum_k w_k \phi(x_k \mid y)$.

    Owns the mixture data — the centres `(N, D)` and mixture log-weights
    `(N,)` — and pairs it with a `DensityKernel`. It turns the kernel's
    *raw* $\log k$ and its normaliser $\log Z$ into normalised components
    $\phi = k/Z$ (`_log_components`) and aggregates the weighted mixture.
    All density operations — `log_prob`, `score`, `sample`,
    `loo_log_prob`, `nadaraya_watson` — derive from `_log_components`.
    Sampling picks a component then defers to the kernel's `sample`.
    """

    centers: jax.Array
    log_weights: jax.Array
    kernel: DensityKernel
    chunk_size: int | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        centers: jax.Array,
        *,
        kernel: DensityKernel,
        log_weights: jax.Array | None = None,
        chunk_size: int | None = None,
    ):
        self.centers = centers
        self.kernel = kernel
        if log_weights is None:
            N = centers.shape[0]
            self.log_weights = jnp.full(
                (N,),
                -jnp.log(N),
                dtype=centers.dtype,
            )
        else:
            self.log_weights = log_weights
        self.chunk_size = chunk_size

    @property
    def weights(self) -> jax.Array:
        """Mixture weights $w_k$ in the simplex (linear scale)."""
        return jnp.exp(self.log_weights)

    def _log_components(self, y: jax.Array) -> jax.Array:
        r"""Per-component log-densities $\log \phi_i(y) = \log\frac{k(x_i, y)}{Z_i}$.

        The kernel supplies the *raw* $\log k$ and its normaliser
        $\log Z_i$ separately; the density is the one place they combine:
        $\log\phi_i = \log k(x_i, y) - \log Z_i$. `at(i)` peels
        per-particle parameter axes; for global parameters it is the
        identity, and JAX traces a constant kernel under vmap.
        """
        N = self.centers.shape[0]
        dim = y.shape[-1]

        def log_phi(i: jax.Array) -> jax.Array:
            k = self.kernel.at(i)
            return k.log_kernel(self.centers[i], y) - k.log_normalizer(dim)

        return jax.vmap(log_phi)(jnp.arange(N))

    def _log_prob_single(self, y: jax.Array) -> jax.Array:
        return jax.nn.logsumexp(self.log_weights + self._log_components(y))

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Log-density at point(s) `x`.

        Args:
            x: query point(s), shape `(D,)` or `(M, D)`.

        Returns:
            Log-density, shape `()` or `(M,)`.
        """
        x_batched = jnp.atleast_2d(x)  # (D,) -> (1, D); (M, D) unchanged
        out = chunked_vmap(
            self._log_prob_single,
            x_batched,
            self.chunk_size or x_batched.shape[0],
        )
        return out.reshape(x.shape[:-1])

    def score(self, y: jax.Array) -> jax.Array:
        r"""Mixture score $\nabla_y \log p(y)$ via autodiff at a single point.

        The autodiff reference for the shipped analytic :func:`kde_score`:
        ``tests/test_velocity_parity.py`` pins ``kde_score`` against this
        (identical for `IsotropicGaussian`, but ``kde_score`` is faster and
        framework-free, so it is what the velocity path actually uses).
        """
        return jax.grad(self._log_prob_single)(y)

    def loo_log_prob(self) -> jax.Array:
        r"""Leave-one-out log-density at each support point: $\log \hat p_{-i}(x_i)$.

        The LOO mixture has mass $\sum_{j\ne i} w_j$, so we subtract
        $\log\sum_{j\ne i} w_j$ — recovers the proper conditional
        mixture after dropping component $i$.
        """
        N = self.centers.shape[0]
        log_w = self.log_weights

        def loo_at(i: jax.Array) -> jax.Array:
            log_phi = self._log_components(self.centers[i])
            mask = jnp.arange(N) == i
            log_terms = jnp.where(mask, -jnp.inf, log_w + log_phi)
            log_w_loo = jnp.where(mask, -jnp.inf, log_w)
            return jax.nn.logsumexp(log_terms) - jax.nn.logsumexp(log_w_loo)

        return jax.vmap(loo_at)(jnp.arange(N))

    def nadaraya_watson(self, xdot: jax.Array, y: jax.Array) -> jax.Array:
        r"""Kernel-weighted average of particle velocities at `y`.

        Returns $\sum_k r_k(y)\,\dot x_k$ where $r_k = w_k \phi_k(y) / p(y)$
        is the mixture responsibility — the velocity field induced by a
        KDE on $(x, \dot x)$ with this kernel.
        """
        r = jax.nn.softmax(self.log_weights + self._log_components(y))
        return jnp.sum(r[:, None] * xdot, axis=0)

    def sample(self, *, key: jax.Array, n: int) -> jax.Array:
        """Draw `n` samples from the (weighted) kernel mixture.

        Picks `n` components from `Categorical(weights)`, then draws one
        point from each picked component's density via the kernel's
        `sample` (with per-particle parameters peeled by `at(i)`).
        """
        k1, k2 = jax.random.split(key)
        idx = jax.random.choice(
            k1,
            self.centers.shape[0],
            shape=(n,),
            p=self.weights,
        )
        keys = jax.random.split(k2, n)
        return jax.vmap(lambda i, ky: self.kernel.at(i).sample(self.centers[i], ky))(
            idx, keys
        )


@integrate.register(KernelDensity)
def _expectation_kde(  # type: ignore[misc]
    dist: KernelDensity, f: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    r"""Approximate $\mathbb{E}_{\rho}[f]$ as $\sum_i w_i\,f(x_i)$.

    Evaluates *f* at the KDE support points and takes the *weighted*
    expectation under the mixture's component weights $w_i$ (uniform
    $1/N$ when ``log_weights`` is None). This is the empirical measure
    on the support points; the approximation is exact in the
    zero-bandwidth limit and matches the convention used by `Empirical`.
    """
    return integrate(Empirical(dist.centers, dist.log_weights), f)


def kde_score(dist: KernelDensity, y: jax.Array) -> jax.Array:
    r"""Analytic score $\nabla_y \log \hat p(y)$ of an isotropic-Gaussian KDE.

    For $\hat p(y) = \sum_k w_k \phi_k(y)$ with per-dimension bandwidths
    $h_d$, the mixture score is the responsibility-weighted average of the
    per-component scores $-(y - x_k)/h^2$:

    $$
    \nabla_y \log \hat p(y)
        = \sum_k r_k(y)\,\frac{-(y - x_k)}{h^2},
    \qquad r_k = \frac{w_k\,\phi_k(y)}{\hat p(y)} .
    $$

    Closed form (no autodiff), identical to ``KernelDensity.score`` for an
    `IsotropicGaussian` kernel. Isotropic-only — raises for other kernels.

    Args:
        dist: KDE with an `IsotropicGaussian` kernel (global bandwidth).
        y: query point, shape `(D,)`.

    Returns:
        Score vector, shape `(D,)`.
    """
    if not isinstance(dist.kernel, IsotropicGaussian):
        raise ValueError(
            "kde_score is isotropic-only; expected `dist.kernel` to be "
            f"`IsotropicGaussian`, got {type(dist.kernel).__name__}."
        )
    D = y.shape[-1]
    h2 = jnp.broadcast_to(dist.kernel.bw, (D,)) ** 2  # (D,) per-dimension
    r = jax.nn.softmax(dist.log_weights + dist._log_components(y))  # (N,)
    return jnp.sum(r[:, None] * (-(y - dist.centers) / h2), axis=0)
