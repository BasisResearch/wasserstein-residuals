"""Trajectory Lagrangian residuals on a particle curve.

The first-order overdamped WGF residuals (`wgf_residual`,
`wgf_kde_sampled_residual`) that tie a learned trajectory to the Wasserstein
gradient-flow velocity equation. Each builds the Wasserstein velocity through
the closed-form seam :func:`stitching.velocity.wasserstein_grad` (no autodiff
framework).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from stitching._kde import InterpolationCurve, IsotropicGaussian
from stitching.velocity import WGFFunctional, wasserstein_grad

_EPS = 1e-12


def _discretise(
    times: jax.Array,
    wg_at: Callable[[jax.Array], jax.Array],
    scheme: str,
) -> jax.Array:
    """Discretisation dispatch — returns WGF along the grid as `(T-1, N, D)`.

    `wg_at(t) -> (N, D)` evaluates the WGF velocity at time *t*.

    - `"trapez"`: trapezoidal rule; WGF evaluated at all *T* grid points,
      averaged per interval.
    - `"euler"`: forward-Euler; WGF at left endpoint.
    - `"implicit"`: backward-Euler; WGF at right endpoint.
    - `"midpoint"`: midpoint rule; WGF at interpolated midpoint.
    """
    if scheme == "trapez":
        wg_all = jax.vmap(wg_at)(times)  # (T, N, D)
        return 0.5 * (wg_all[:-1] + wg_all[1:])
    if scheme == "euler":
        return jax.vmap(wg_at)(times[:-1])
    if scheme == "implicit":
        return jax.vmap(wg_at)(times[1:])
    if scheme == "midpoint":
        return jax.vmap(wg_at)(0.5 * (times[:-1] + times[1:]))
    raise ValueError(f"Unknown scheme={scheme!r}")


def _normalize_residual(
    r: jax.Array,  # (T_out, N, D)
    dx: jax.Array,  # (T_out, N, D)
    force: jax.Array,  # (T_out, N, D)
    weights: jax.Array,  # (N,)
    norm: str,
) -> jax.Array:
    """Per-step weighted residual `(T_out,)`; caller does the time reduction.

    - `"raw"`: ‖r‖², no normalisation.
    - `"relative"`: ‖r‖² / (‖dx‖² + ‖f‖² + ε), gradient through the denominator.
    - `"relative-sg"`: same but `stop_gradient` on the denominator.
    """
    sq = jnp.sum(r**2, axis=-1)  # (T_out, N)
    if norm == "raw":
        return jnp.sum(weights * sq, axis=-1)
    if norm not in ("relative", "relative-sg"):
        raise ValueError(f"Unknown norm={norm!r}")
    denom = jnp.sum(dx**2, axis=-1) + jnp.sum(force**2, axis=-1) + _EPS
    if norm == "relative-sg":
        denom = jax.lax.stop_gradient(denom)
    return jnp.sum(weights * (sq / denom), axis=-1)


def wgf_residual(
    functional: Callable[[jax.Array], WGFFunctional],
    scheme: str = "trapez",
    norm: str = "raw",
    kinetic: float | jax.Array = 1.0,
) -> Callable[[InterpolationCurve], jax.Array]:
    r"""Wasserstein Residual Functional for the overdamped (WGF) regime.

    Penalises violation of the first-order Wasserstein gradient flow ODE
    $\dot x + \nabla_W F / \lambda = 0$ via the displacement-level
    residual $r_k = \Delta x_k + \tfrac{h_k}{\lambda}\,\nabla_W F_k$,
    producing *T − 1* residuals (one per interval).

    Parameters
    ----------
    functional
        Callable ``t -> F`` returning the energy functional at time *t*.
        For a static functional, pass ``lambda t: F``.
    scheme
        Discretisation scheme — see `_discretise`.
    norm
        Normalisation mode — see `_normalize_residual`.
    kinetic
        Kinetic coefficient $\lambda \ge 1$. Larger values suppress the
        WGF force, producing smoother trajectories.
    """

    def _action(curve: InterpolationCurve) -> jax.Array:
        traj = curve.centers  # (T, N, D)
        t_grid = curve.t_grid  # (T,)
        weights = curve.weights  # (N,) sums to 1; uniform 1/N when log_w is None
        dt = t_grid[1:] - t_grid[:-1]  # (T-1,)

        def wg_at(t: jax.Array) -> jax.Array:
            dist = curve(t)
            return jax.vmap(wasserstein_grad(functional(t))(dist))(dist.centers)

        wg = _discretise(t_grid, wg_at, scheme)  # (T-1, N, D)

        # Displacement-level residual:  r_j,k = Δx_j,k + h_j·wg_j,k / λ.
        # Discretises ∫ Σ_k w_k ‖ẋ + ∇δF/δρ‖² dt via forward-Euler ẋ ≈ Δx/h
        # and a left-Riemann sum; the 1/h_j Riemann weight is what makes this
        # converge to the continuous residual as the grid is refined.
        dx = traj[1:] - traj[:-1]  # (T-1, N, D)
        force = dt[:, None, None] * wg / kinetic  # (T-1, N, D)
        r = dx + force

        per_interval = _normalize_residual(r, dx, force, weights, norm)  # (T-1,)
        if norm == "raw":
            return jnp.sum(per_interval / dt)  # 1/h_j Riemann weight
        return jnp.mean(per_interval)

    return _action


def wgf_kde_sampled_residual(
    functional: Callable[[jax.Array], WGFFunctional],
    kinetic: float | jax.Array = 1.0,
) -> Callable[[InterpolationCurve, jax.Array], jax.Array]:
    r"""Velocity residual evaluated at one paired off-centre sample per particle.

    Stochastic sibling of `wgf_residual`: both discretise the paper's
    velocity-residual equation

    $$
    \mathcal R_{\mathrm{vel}} = \int_0^T \mathbb{E}_{\mathbf{x}\sim\rho_t^\theta}
        \left\| \mathbf{v}_t^\theta(\mathbf{x}) +
        \nabla_\mathbf{x}\tfrac{\delta F}{\delta\rho_t}(\mathbf{x}) \right\|^2 dt.
    $$

    The inner expectation is estimated by **stratified Monte-Carlo with one
    sample per particle**: at each snapshot $t_j$ and particle $k$, draw
    $\epsilon_k\sim\phi_\sigma$, form $y_{t_j,k} = x_{t_j,k}^\theta +
    \sigma\epsilon_k$, and average under the mixture weights. The velocity at
    each $y$ is the **Nadaraya–Watson velocity** from the *unperturbed* centres
    and finite-difference $\dot x_k = \Delta x_k / h_j$.

    Iso-only — Cholesky curves carry no scalar bandwidth to scale the
    perturbation noise; raises `ValueError` otherwise.

    Args:
        functional: callable ``t -> F`` returning the energy functional at
            time *t*. For a static functional, pass ``lambda t: F``.
        kinetic: kinetic coefficient $\lambda \ge 1$.
    """

    def _action(curve: InterpolationCurve, key: jax.Array) -> jax.Array:
        traj = curve.centers  # (T, N, D)
        t_grid = curve.t_grid  # (T,)
        weights = curve.weights  # (N,)
        dt = t_grid[1:] - t_grid[:-1]  # (T-1,)
        T, N, D = traj.shape

        if not isinstance(curve.kernel, IsotropicGaussian):
            raise ValueError(
                "wgf_kde_sampled_residual is isotropic-only; expected "
                "`curve.kernel` to be `IsotropicGaussian`, got "
                f"{type(curve.kernel).__name__}.",
            )
        bw = curve.kernel.bw

        # Particle velocities per interval, finite-difference of the
        # *unperturbed* trajectory. (T-1, N, D).
        xdot = (traj[1:] - traj[:-1]) / dt[:, None, None]

        keys = jax.random.split(key, T - 1)

        def _per_interval(j_key: jax.Array, j_idx: jax.Array) -> jax.Array:
            # Paired stratified MC: one ε per particle, no Cat(w) resampling —
            # the weighting is recovered by the `weights * sq` reduction below.
            eps = jax.random.normal(j_key, (N, D)) * bw  # (N, D)
            kde_j = curve[j_idx]  # snapshot KDE — no interp on grid points
            y = kde_j.centers + eps  # (N, D)

            # Nadaraya–Watson velocity at each y_k, built from *all* particles.
            v_NW = jax.vmap(lambda yi: kde_j.nadaraya_watson(xdot[j_idx], yi))(y)

            # Functional gradient at each sample, mixture weights flowing into
            # the score and interaction terms via the KDE.
            grad_fn = wasserstein_grad(functional(t_grid[j_idx]))(kde_j)
            g = jax.vmap(grad_fn)(y)  # (N, D)

            r = v_NW + g / kinetic  # (N, D)
            sq = jnp.sum(r**2, axis=-1)  # (N,)
            # Weighted MC × interval Riemann weight h_j (left-Riemann on time).
            return jnp.sum(weights * sq) * dt[j_idx]

        return jnp.sum(jax.vmap(_per_interval)(keys, jnp.arange(T - 1)))

    return _action
