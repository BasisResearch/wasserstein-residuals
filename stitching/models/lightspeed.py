r"""Lightspeed (JKOnet$^\\star$): potential-only model + per-pair JKO loss.

Implements Terpin et al. (NeurIPS 2024)'s JKOnet$^\\star$ paradigm:

- The model carries only a learned potential $V_\theta(x)$ and a
  *fixed* bandwidth (Silverman from the t=0 training data) used to
  evaluate the Wasserstein gradient at sampling time.
- Training minimises the **first-order JKO residual** on coupled
  snapshot pairs ``(x, y)`` with step $\tau$:

  $$
  \\big\\| \\tau \\nabla \\tfrac{\\delta\\mathcal F^\\theta}{\\delta\\rho}(y)
        + (y - x) \\big\\|^2.
  $$

  The pairs are typically Hungarian-matched between consecutive
  observed snapshots; the pairs are prepared via
  :func:`stitching.utils.ot.hungarian_match`.
- Forward evaluation chains :func:`stitching.utils.solvers.jko_proximal_step`
  (one explicit Euler step per snapshot interval) starting from the
  observed t=0 particles, mirroring the published
  ``jkonet-star/utils/sde_simulator.py:SDESimulator.forward_sampling``
  protocol.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from stitching._kde import IsotropicGaussian, KernelDensity, SpatioTemporalData
from stitching.utils.nn import MLP, field_mlp, inv_softplus
from stitching.utils.ot import hungarian_match
from stitching.utils.solvers import jko_proximal_step
from stitching.velocity import WGFFunctional, wasserstein_grad


class Lightspeed(eqx.Module):
    r"""JKOnet$^\\star$ potential-only model: just a learnable potential $V_\theta$.

    The bandwidth attribute is held only so that downstream evaluation
    code (``evaluate_one_step_ahead`` and ``make_functional`` in the
    experiment layer) can treat Lightspeed uniformly with other
    KDE-based models. It is *not* learned during training.
    """

    potential_net: MLP
    raw_entropy_coeff: jax.Array
    raw_potential_coeff: jax.Array
    bandwidth: jax.Array  # (D,) — fixed at construction

    def __init__(
        self,
        dim: int,
        hidden: tuple[int, ...],
        bandwidth: jax.Array,
        key: jax.Array,
        entropy_coeff: float = 0.0,
        potential_coeff: float = 1.0,
    ):
        self.potential_net = field_mlp(dim, hidden, key)
        self.raw_entropy_coeff = inv_softplus(max(entropy_coeff, 1e-6))
        self.raw_potential_coeff = inv_softplus(potential_coeff)
        self.bandwidth = bandwidth

    @classmethod
    def from_data(
        cls,
        train_data: SpatioTemporalData,
        *,
        hidden: tuple[int, ...],
        key: jax.Array,
        entropy_coeff: float = 0.0,
        potential_coeff: float = 1.0,
    ) -> Lightspeed:
        """Build from training data: bandwidth = Silverman on t=0 particles.

        Per the JKOnet$^\\star$ paradigm, the bandwidth is fixed at
        construction time (not learned).
        """
        from stitching._kde import silverman_bandwidth

        dim = train_data.x.shape[1]
        t_min = float(jnp.unique(train_data.t)[0])
        mask = jnp.isclose(train_data.t, t_min, atol=1e-6)
        pool = train_data.x[mask] if mask.any() else train_data.x
        bw = silverman_bandwidth(pool)
        return cls(
            dim=dim,
            hidden=hidden,
            bandwidth=bw,
            key=key,
            entropy_coeff=entropy_coeff,
            potential_coeff=potential_coeff,
        )

    @property
    def entropy_coeff(self) -> jax.Array:
        return jax.nn.softplus(self.raw_entropy_coeff)

    @property
    def potential_coeff(self) -> jax.Array:
        return jax.nn.softplus(self.raw_potential_coeff)

    def particle_weights(self) -> None:
        """Lightspeed carries no explicit particle cloud — returns ``None``.

        The model-API counterpart to :meth:`Stitching.particle_weights`; weighted
        metrics skip the per-particle weighting when this is ``None``.
        """
        return None

    def learned_potential_fn(self) -> Callable[[jax.Array], jax.Array]:
        """Return the per-point learned potential ``x -> V_θ(x)`` as a scalar fn.

        The model-API counterpart to :meth:`Stitching.learned_potential_fn`:
        squeezes the network output to a scalar (the *unscaled* value) so the
        evaluation and plotting layers need not reach into ``potential_net``
        directly.
        """
        return lambda x: jnp.squeeze(self.potential_net(x))

    def functional(self, **kwargs: float | jax.Array | None) -> WGFFunctional:
        """Energy functional ``F = c_V·V_θ + c_H·H`` driving the JKO chain."""
        from ._functional import build_functional

        return build_functional(self, **kwargs)

    def __repr__(self) -> str:
        D = self.bandwidth.shape[0]
        return (
            f"Lightspeed(dim={D})\n"
            f"  bandwidth: mean={float(jnp.mean(self.bandwidth)):.4f}  (fixed)\n"
            f"  entropy:   {float(self.entropy_coeff):.4f}\n"
            f"  potential: {float(self.potential_coeff):.4f}"
        )

    def sample(
        self,
        t_eval: jax.Array,
        train_data: SpatioTemporalData,
        *,
        num_samples: int | None = None,
        key: jax.Array | None = None,
    ) -> dict[float, jax.Array]:
        r"""Forward-Euler JKO chain seeded at the observed t=0 particles.

        Chains :func:`stitching.utils.solvers.jko_proximal_step` (one
        explicit Euler step per requested interval) under the model's
        own functional. Matches the published JKOnet$^\\star$
        forward-evaluation protocol.

        ``num_samples``/``key`` are accepted for API parity with the
        other models but unused: the JKO chain returns N=|t=0 particles|
        deterministic samples per timepoint.
        """
        del num_samples, key  # JKO chain is deterministic, fixed N
        functional = self.functional()
        t_sorted = jnp.sort(t_eval)
        train_t = jnp.unique(train_data.t)
        t0 = float(train_t[0])

        mask0 = jnp.isclose(train_data.t, t0, atol=1e-6)
        x = train_data.x[mask0]

        result: dict[float, jax.Array] = {round(t0, 8): x}
        prev = t0
        for tv in t_sorted:
            tv_f = float(tv)
            if tv_f <= prev + 1e-9:
                continue
            dt = tv_f - prev
            x = jko_proximal_step(functional, x, dt, self.bandwidth)
            result[round(tv_f, 8)] = x
            prev = tv_f
        return result


# ---------------------------------------------------------------------------
# JKO-residual snapshot pairs
# ---------------------------------------------------------------------------


def lightspeed_pairs(
    train_data: SpatioTemporalData,
    *,
    coupling: str = "ot",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Precompute snapshot pairs $(x_t, y_{t+\tau})$ and step sizes $\tau$.

    Reconstructs an ``(T, N, D)`` snapshot tensor from *train_data* by
    grouping on time and truncating to the smallest per-time particle
    count, then forms one source/target pair per consecutive interval.

    Used to feed :class:`LightspeedLoss` (the JKOnet$^\\star$ first-order
    JKO residual).

    Args:
        train_data: spatio-temporal observations.
        coupling:
            ``"natural"`` — use the per-time particle order as the pair
            index; only valid when *train_data* was simulated with coupled
            trajectories (same particles tracked across snapshots).
            ``"ot"`` (default) — Hungarian matching on squared Euclidean
            cost between consecutive snapshots. Matches the published
            JKOnet$^\\star$ recipe.

    Returns:
        ``(pairs_x, pairs_y, taus)`` where ``pairs_x`` and ``pairs_y``
        are ``(T-1, N, D)`` source/target particles per interval, and
        ``taus`` is ``(T-1,)`` step sizes :math:`t_{j+1} - t_j`.
    """
    obs_t = np.asarray(train_data.t)
    obs_x = np.asarray(train_data.x)
    unique_t = np.unique(obs_t)
    groups = [obs_x[obs_t == t] for t in unique_t]
    n = min(g.shape[0] for g in groups)
    snaps = np.stack([g[:n] for g in groups], axis=0)  # (T, n, D)

    pairs_x_list = []
    pairs_y_list = []
    for j in range(snaps.shape[0] - 1):
        x = snaps[j]
        y = snaps[j + 1]
        if coupling == "natural":
            pairs_x_list.append(x)
            pairs_y_list.append(y)
        elif coupling == "ot":
            row, col = hungarian_match(x, y)
            pairs_x_list.append(x[row])
            pairs_y_list.append(y[col])
        else:
            raise ValueError(f"Unknown lightspeed coupling: {coupling!r}")

    pairs_x = jnp.array(np.stack(pairs_x_list, axis=0), dtype=jnp.float32)
    pairs_y = jnp.array(np.stack(pairs_y_list, axis=0), dtype=jnp.float32)
    taus = jnp.array(np.diff(unique_t), dtype=jnp.float32)
    return pairs_x, pairs_y, taus


# ---------------------------------------------------------------------------
# Composite training objective
# ---------------------------------------------------------------------------


class LightspeedLoss(eqx.Module):
    r"""JKOnet$^\\star$ first-order-JKO residual on coupled snapshot pairs.

    For each interval $j$ and pair $(x, y)$ with step $\tau_j$, minimise

    $$
    \big\| \tau_j\,\nabla \tfrac{\delta\mathcal F^\theta}{\delta\rho}(y)
          + (y - x) \big\|^2.
    $$

    The functional gradient $\nabla \delta\mathcal F/\delta\rho(y)$
    collapses to $c_V \nabla V_\theta(y)$ when entropy and interaction
    are absent, which matches Terpin et al.'s "potential" variant; this
    is what the model instantiates by default.

    The pairs (and their step sizes) are precomputed once from the
    training data via :func:`lightspeed_pairs` — this is part of the
    JKOnet$^\\star$ paradigm: a fixed coupling chosen at the start.
    """

    pairs_x: jax.Array  # (T-1, N, D) source particles per interval
    pairs_y: jax.Array  # (T-1, N, D) target particles per interval
    taus: jax.Array  # (T-1,) step sizes t_{j+1} - t_j
    coupling: str = eqx.field(static=True, default="ot")
    freeze_entropy: float | None = eqx.field(static=True, default=None)
    freeze_potential: float | None = eqx.field(static=True, default=None)

    @classmethod
    def from_data(
        cls,
        train_data: SpatioTemporalData,
        *,
        coupling: str = "ot",
        freeze_entropy: float | None = None,
        freeze_potential: float | None = None,
    ) -> LightspeedLoss:
        px, py, taus = lightspeed_pairs(train_data, coupling=coupling)
        return cls(
            pairs_x=px,
            pairs_y=py,
            taus=taus,
            coupling=coupling,
            freeze_entropy=freeze_entropy,
            freeze_potential=freeze_potential,
        )

    def __call__(
        self,
        model: Lightspeed,
        data: SpatioTemporalData,
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del data, key  # unused — pairs encode the data, loss is deterministic
        functional = model.functional(
            freeze_entropy=self.freeze_entropy,
            freeze_potential=self.freeze_potential,
        )

        grad_fn = wasserstein_grad(functional)

        def grad_dF_at(y: jax.Array) -> jax.Array:
            rho = KernelDensity(y, kernel=IsotropicGaussian(bw=model.bandwidth))
            return jax.vmap(grad_fn(rho))(y)

        def per_interval(x_j, y_j, tau_j):
            g = grad_dF_at(y_j)
            residual = tau_j * g + (y_j - x_j)
            return jnp.mean(jnp.sum(residual**2, axis=-1))

        per = jax.vmap(per_interval)(self.pairs_x, self.pairs_y, self.taus)
        loss = jnp.sum(per)
        return loss, {"jko_residual": loss}

    def __repr__(self) -> str:
        T_minus_1 = self.pairs_x.shape[0]
        return f"LightspeedLoss(coupling={self.coupling!r}, n_intervals={T_minus_1})"
