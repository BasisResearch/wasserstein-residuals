"""Time-axis containers and parametric curves of distributions.

Holds the types that organise things along a time axis:

- `SpatioTemporalData`: a batch of (x, t) pairs.
- `InterpolationCurve`: time-varying KDE pairing a `(T, N, D)`
  trajectory of mixture centres with a `Kernel`. The trajectory analog
  of `KernelDensity`.

`KernelDensity` is imported lazily inside `InterpolationCurve.__call__`
and `__getitem__` to avoid the otherwise-circular dependency with
`stitching._kde.distributions`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from ._core import Time, integrate
from .kernel import DensityKernel

if TYPE_CHECKING:
    from .distributions import KernelDensity


# ---------------------------------------------------------------------------
# Spatiotemporal data container
# ---------------------------------------------------------------------------


class SpatioTemporalData(eqx.Module):
    """A batch of spatiotemporal data points (x, t).

    A pure data container, not a `Distribution`: time-axis expectations go
    through the :func:`stitching._kde.integrate` dispatch
    (:func:`_expectation_spatiotemporal`), not a ``log_prob``/``sample`` API.
    """

    x: jax.Array  # (N, D) spatial coordinates
    t: jax.Array  # (N,)   time values


@integrate.register(SpatioTemporalData)
def _expectation_spatiotemporal(  # type: ignore[misc]
    data: SpatioTemporalData,
    f: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Mean of $f(x_i, t_i)$ — uniform average over spatiotemporal samples."""
    return jnp.mean(
        jax.vmap(f)(jnp.concatenate([data.x, data.t.reshape(-1, 1)], axis=-1))
    )


# ---------------------------------------------------------------------------
# InterpolationCurve — particle trajectory + Kernel → density at any time
# ---------------------------------------------------------------------------


class InterpolationCurve(eqx.Module):
    """Time-varying KDE built from a particle trajectory and a kernel.

    Trajectory analog of `KernelDensity`: pairs a `(T, N, D)` trajectory
    of mixture centres with a `Kernel`. The mixture data (`centers`,
    `t_grid`, `log_weights`) lives here; the kernel owns its parameters.

    Two indexing modes; the syntax tells you which:

    - ``curve[k]`` — discrete: returns the `KernelDensity` at the
      `k`-th snapshot exactly. No interpolation; cheaper, and the
      natural form when iterating over the grid.
    - ``curve(t)`` — continuous: returns the `KernelDensity` at
      arbitrary time `t` via linear interpolation of particles.
      Required for off-grid times such as midpoint quadrature.

    Linear interpolation lets gradients flow through the time argument.
    """

    centers: jax.Array  # (T, N, D)
    t_grid: jax.Array  # (T,)
    log_weights: jax.Array  # (N,) shared across time
    kernel: DensityKernel

    def __init__(
        self,
        centers: jax.Array,
        t_grid: jax.Array,
        *,
        kernel: DensityKernel,
        log_weights: jax.Array | None = None,
    ):
        self.centers = centers
        self.t_grid = t_grid
        if log_weights is None:
            N = centers.shape[1]
            self.log_weights = jnp.full(
                (N,),
                -jnp.log(N),
                dtype=centers.dtype,
            )
        else:
            self.log_weights = log_weights
        self.kernel = kernel

    @property
    def weights(self) -> jax.Array:
        return jnp.exp(self.log_weights)

    def __getitem__(self, idx: int | jax.Array) -> KernelDensity:
        """`KernelDensity` at the `idx`-th snapshot — no interpolation.

        `kernel.at(idx)` peels per-snapshot parameter axes; for
        time-shared parameters it is the identity.
        """
        from .distributions import KernelDensity  # break cycle

        return KernelDensity(
            self.centers[idx],
            kernel=self.kernel.at(idx),
            log_weights=self.log_weights,
        )

    def __call__(self, t: Time) -> KernelDensity:
        """`KernelDensity` at arbitrary time `t` via linear interpolation.

        Out-of-range times are silently clipped to the grid endpoints,
        keeping gradients finite near the boundaries (useful for
        trapezoidal quadrature on `[t_grid[0], t_grid[-1]]`).
        """
        from .distributions import KernelDensity  # break cycle

        t_grid = self.t_grid
        t_val = jnp.asarray(t)
        t_val = jnp.clip(t_val, t_grid[0], t_grid[-1])
        idx_right = jnp.searchsorted(t_grid, t_val, side="right")
        idx_right = jnp.clip(idx_right, 1, len(t_grid) - 1)
        idx_left = idx_right - 1
        t_left = t_grid[idx_left]
        t_right = t_grid[idx_right]
        alpha = jnp.where(
            t_right == t_left,
            0.0,
            (t_val - t_left) / (t_right - t_left),
        )
        interp_centers = (1.0 - alpha) * self.centers[idx_left] + alpha * self.centers[
            idx_right
        ]
        return KernelDensity(
            interp_centers,
            kernel=self.kernel.interp_at(alpha, idx_left, idx_right),
            log_weights=self.log_weights,
        )
