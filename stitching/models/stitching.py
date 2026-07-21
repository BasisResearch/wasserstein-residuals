"""Stitching: directly-optimised trajectory + KDE marginals + composite loss.

The model parameterises a particle trajectory ``(T, N, D)`` together
with a learned potential `V_θ(x)` (optionally time-conditioned), an
optional pairwise interaction kernel, and a global ARD bandwidth. The
trajectory + bandwidth + mixture weights together define an
`InterpolationCurve` of `KernelDensity` marginals; the potential drives
the consistency residual that ties them together via the WGF velocity
equation.

`StitchingLoss` combines:

- NLL of the KDE marginals against snapshot data.
- A trajectory-level *consistency* residual penalising violation of
  the WGF velocity equation. The residual is a convex
  combination of two estimators (`residual_alpha`):

  - α = 0: deterministic centres-quadrature
    (`stitching.residual.wgf_residual`).
  - α = 1: stochastic KDE-MC with Nadaraya–Watson velocity
    (`stitching.residual.wgf_kde_sampled_residual`).

  Intermediate α blends the two estimators linearly.

The loss function is Config-free — the caller (typically
`stitching.build.make_loss_fn`) builds a ``functional_at: t -> F``
that closes over any freeze knobs and time-conditioning, and passes
explicit weights / scheme kwargs.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from stitching._kde import (
    InterpolationCurve,
    IsotropicGaussian,
    KernelDensity,
    SpatioTemporalData,
)
from stitching.residual import (
    wgf_kde_sampled_residual,
    wgf_residual,
)
from stitching.utils.nn import MLP, field_mlp, inv_softplus
from stitching.utils.ot import bures_map, gaussian_moments, hungarian_match, psd_sqrt
from stitching.velocity import WGFFunctional, nll

# ---------------------------------------------------------------------------
# Stitching — directly-optimised trajectory + KDE marginals
# ---------------------------------------------------------------------------


class Stitching(eqx.Module):
    """Directly-optimised trajectory + KDE marginals.

    Attributes:
        trajectories: particle positions at each snapshot, `(T, N, D)`.
        t_grid: snapshot times, `(T,)`.
        potential_net: learned potential V(x), or V(x, t) when
            *time_conditioned* is True.
        interaction_net: learned pairwise interaction kernel (or None).
        interaction_type: ``"radial"`` (MLP(||r||²)) or ``"full"`` (MLP(r)).
        raw_bandwidth: unconstrained ARD bandwidth (softplus -> positive).
        raw_entropy_coeff: unconstrained scalar coefficient (softplus -> positive).
        raw_potential_coeff: unconstrained scalar coefficient (softplus -> positive).
        raw_interaction_coeff: unconstrained scalar coefficient (softplus -> positive).
        raw_weights: pre-softmax mixture weights `(N,)`.

    Sampling:
        Call `sample` to retrieve (and optionally KDE-resample)
        trajectory particles at each requested time.
    """

    trajectories: jax.Array  # (T, N, D)
    t_grid: jax.Array  # (T,)
    potential_net: MLP  # x -> scalar, OR (x,t) -> scalar when time_conditioned
    interaction_net: MLP | None  # interaction kernel MLP (or None)
    interaction_type: str  # "radial" or "full"
    raw_bandwidth: jax.Array  # (D,)
    raw_entropy_coeff: jax.Array  # scalar
    raw_potential_coeff: jax.Array  # scalar
    raw_interaction_coeff: jax.Array  # scalar
    raw_weights: jax.Array  # (N,) softmax-normalised mixture weights
    time_conditioned: bool = eqx.field(static=True)

    def __init__(
        self,
        trajectories: jax.Array,
        t_grid: jax.Array,
        dim: int,
        hidden: tuple[int, ...],
        key: jax.Array,
        lengthscale: float | jax.Array = 0.5,
        entropy_coeff: float = 0.03,
        potential_coeff: float = 1.0,
        interaction_coeff: float = 0.0,
        interaction_type: str = "radial",
        interaction_hidden: tuple[int, ...] = (64, 64),
        time_conditioned: bool = False,
    ):
        self.trajectories = trajectories
        self.t_grid = t_grid
        self.time_conditioned = time_conditioned
        k1, k2 = jax.random.split(key)
        self.potential_net = field_mlp(
            dim,
            hidden,
            k1,
            time_conditioned=time_conditioned,
        )
        # Interaction kernel MLP (only built when interaction_coeff > 0).
        # The interaction kernel W(r) is a scalar field on the
        # difference vector r — radial mode reduces it to W(||r||²).
        if interaction_coeff > 0.0:
            in_dim_inter = 1 if interaction_type == "radial" else dim
            self.interaction_net = field_mlp(in_dim_inter, interaction_hidden, k2)
        else:
            self.interaction_net = None
        self.interaction_type = interaction_type

        # lengthscale can be a scalar (broadcast) or per-dimension array.
        ls = jnp.broadcast_to(jnp.asarray(lengthscale, dtype=jnp.float32), (dim,))
        self.raw_bandwidth = inv_softplus(ls)
        # Floor every coefficient before inv_softplus: inv_softplus(0) = -inf,
        # whose softplus gradient is a 0/0 → NaN that poisons the whole step
        # (e.g. a V-free run with potential_coeff=0). 1e-4 is negligible in
        # value but keeps the unconstrained parameter and its gradient finite.
        self.raw_entropy_coeff = inv_softplus(max(entropy_coeff, 1e-4))
        self.raw_potential_coeff = inv_softplus(max(potential_coeff, 1e-4))
        self.raw_interaction_coeff = inv_softplus(max(interaction_coeff, 1e-4))
        # Mixture weights: zero-init in raw space -> uniform softmax weights.
        N = trajectories.shape[1]
        self.raw_weights = jnp.zeros(N, dtype=jnp.float32)

    @classmethod
    def from_data(
        cls,
        train_data: SpatioTemporalData,
        *,
        num_particles: int,
        num_steps: int,
        hidden: tuple[int, ...],
        key: jax.Array,
        lengthscale: float | str = "silverman",
        entropy_coeff: float = 0.03,
        potential_coeff: float = 1.0,
        interaction_coeff: float = 0.0,
        interaction_type: str = "radial",
        interaction_hidden: tuple[int, ...] = (64, 64),
        time_conditioned: bool = False,
        trajectory_init: str = "ot",
    ) -> Stitching:
        """Build from training data: interpolated trajectory + Silverman bandwidth.

        ``trajectory_init`` selects how the particle bundle is seeded:
        ``"ot"`` (default) Hungarian-OT-matches the first and last observed
        snapshots and linearly interpolates the matched pairs (with small
        data-coherent jitter); ``"mccann"`` fits a Gaussian to each terminal
        snapshot and transports shared standard normals along the
        Bures–Wasserstein OT geodesic (non-braiding — see
        :func:`mccann_interpolate_trajectories`). ``lengthscale="silverman"``
        initialises the per-dimension bandwidth from the t=0 observed particles.
        """
        from stitching._kde import silverman_bandwidth

        if trajectory_init == "ot":
            trajectory, t_grid, key = ot_interpolate_trajectories(
                train_data,
                num_particles,
                num_steps,
                key,
            )
        elif trajectory_init == "mccann":
            trajectory, t_grid, key = mccann_interpolate_trajectories(
                train_data,
                num_particles,
                num_steps,
                key,
            )
        else:
            raise ValueError(
                f"Unknown trajectory_init {trajectory_init!r}; expected "
                '"ot" or "mccann".'
            )

        if isinstance(lengthscale, (int, float)):
            ls: float | jax.Array = float(lengthscale)
        elif isinstance(lengthscale, np.ndarray):
            ls = jnp.array(lengthscale)
        elif lengthscale == "silverman":
            obs_t = np.asarray(train_data.t).ravel()
            unique_np = np.unique(obs_t)
            pts0 = np.asarray(train_data.x)[obs_t == unique_np[0]]
            ls = silverman_bandwidth(jnp.array(pts0))
        else:
            raise ValueError(
                f"Unknown lengthscale {lengthscale!r}; expected a float, an "
                'array, or "silverman".'
            )

        return cls(
            trajectories=trajectory,
            t_grid=t_grid,
            dim=train_data.x.shape[1],
            hidden=hidden,
            key=key,
            lengthscale=ls,
            entropy_coeff=entropy_coeff,
            potential_coeff=potential_coeff,
            interaction_coeff=interaction_coeff,
            interaction_type=interaction_type,
            interaction_hidden=interaction_hidden,
            time_conditioned=time_conditioned,
        )

    @property
    def bandwidth(self) -> jax.Array:
        return jax.nn.softplus(self.raw_bandwidth)

    @property
    def entropy_coeff(self) -> jax.Array:
        return jax.nn.softplus(self.raw_entropy_coeff)

    @property
    def potential_coeff(self) -> jax.Array:
        return jax.nn.softplus(self.raw_potential_coeff)

    @property
    def interaction_coeff(self) -> jax.Array:
        return jax.nn.softplus(self.raw_interaction_coeff)

    @property
    def log_weights(self) -> jax.Array:
        """Mixture log-weights via log-softmax of the raw parameters."""
        return jax.nn.log_softmax(self.raw_weights)

    @property
    def weights(self) -> jax.Array:
        """Mixture weights via softmax of the raw parameters."""
        return jax.nn.softmax(self.raw_weights)

    def particle_weights(self) -> jax.Array:
        """Per-particle mixture weights (softmax of the raw parameters).

        The model-API accessor used by weighted metrics (e.g. the Bd²W₂-UVP in
        :mod:`stitching.evaluate`) so callers need not reach into ``weights`` /
        ``raw_weights`` directly. Lightspeed, which has no explicit particles,
        returns ``None`` from its counterpart.
        """
        return self.weights

    def learned_potential_fn(self) -> Callable[[jax.Array], jax.Array]:
        """Return the per-point learned potential ``x -> V_θ(x)`` as a scalar fn.

        Squeezes the network output to a scalar so it is directly vmappable and
        ``jax.grad``-able. The value is the *unscaled* network output: callers
        that need the functional's contribution multiply by
        :attr:`potential_coeff` themselves (matching the existing evaluation and
        plotting code). Provided so those layers stop reaching into
        ``potential_net`` directly. For a time-conditioned potential the network
        expects ``concat([x, t])``; this accessor passes ``x`` alone, mirroring
        the (non-time-conditioned) call sites that consume it.
        """
        return lambda x: jnp.squeeze(self.potential_net(x))

    def density_curve(self) -> InterpolationCurve:
        """Return the `InterpolationCurve` over trajectory snapshots.

        The curve is callable as ``curve(t) -> KernelDensity``, lifted
        through linear interpolation of the trajectory particles between
        snapshot times.
        """
        return InterpolationCurve(
            self.trajectories,
            self.t_grid,
            kernel=IsotropicGaussian(bw=self.bandwidth),
            log_weights=self.log_weights,
        )

    def functional(self, **kwargs: float | jax.Array | None) -> WGFFunctional:
        """Energy functional driving the WGF velocity residual.

        Adds the pairwise interaction term to the
        :class:`~stitching.velocity.WGFFunctional` when
        ``self.interaction_net`` is non-None (i.e. when the model was
        constructed with ``interaction_coeff > 0``).
        """
        from ._functional import build_functional

        return build_functional(self, **kwargs)

    def __repr__(self) -> str:
        T, N, D = self.trajectories.shape
        has_inter = self.interaction_net is not None
        lines = [
            f"Stitching(N={N}, T={T}, dim={D}, "
            f"time_conditioned={self.time_conditioned}, interaction={has_inter})",
            f"  bandwidth:   mean={float(jnp.mean(self.bandwidth)):.4f}",
            f"  entropy:     {float(self.entropy_coeff):.4f}",
            f"  potential:   {float(self.potential_coeff):.4f}",
        ]
        if has_inter:
            lines.append(f"  interaction: {float(self.interaction_coeff):.4f}")
        return "\n".join(lines)

    def sample(
        self,
        t_eval: jax.Array,
        num_samples: int | None = None,
        key: jax.Array | None = None,
    ) -> dict[float, jax.Array]:
        """Look up trajectory particles at each *t_eval* time.

        When *num_samples* is given, KDE samples are drawn around the
        trajectory particles using the model's bandwidth.
        """
        t_eval_sorted = jnp.sort(t_eval)
        result: dict[float, jax.Array] = {}
        _key = key if key is not None else jax.random.key(0)
        for t_val in t_eval_sorted:
            t_key = round(float(t_val), 8)
            idx = jnp.argmin(jnp.abs(self.t_grid - t_val))
            particles = self.trajectories[idx]
            if num_samples is not None:
                _key, sample_key = jax.random.split(_key)
                dist = KernelDensity(
                    particles,
                    kernel=IsotropicGaussian(bw=self.bandwidth),
                    log_weights=self.log_weights,
                )
                particles = dist.sample(key=sample_key, n=num_samples)
            result[t_key] = particles
        return result


# ---------------------------------------------------------------------------
# Trajectory initialisation
# ---------------------------------------------------------------------------


def ot_interpolate_trajectories(
    train_data: SpatioTemporalData,
    num_particles: int,
    num_steps: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build initial Stitching trajectories via OT matching + linear interpolation.

    Used by :meth:`Stitching.from_data`. Returns
    ``(trajectory, t_grid, remaining_key)`` where *trajectory* has shape
    ``(num_steps, num_particles, D)`` and *t_grid* has shape
    ``(num_steps,)``. A small data-coherent jitter (1 % of per-dimension
    range) is added to break degeneracy between adjacent particles.
    """
    dim = train_data.x.shape[1]
    t_unique = jnp.unique(train_data.t)
    t_grid = jnp.linspace(float(t_unique[0]), float(t_unique[-1]), num_steps)

    k1, k2 = jax.random.split(key)

    points = np.asarray(train_data.x)
    obs_times = np.asarray(train_data.t).ravel()
    unique_np = np.unique(obs_times)

    rng = np.random.default_rng(int(jax.random.randint(k1, (), 0, 2**30)))

    pts0 = points[obs_times == unique_np[0]]

    if unique_np.size >= 2:
        ptsT = points[obs_times == unique_np[-1]]
        row_ind, col_ind = hungarian_match(pts0, ptsT)

        n_pairs = len(row_ind)
        replace = n_pairs < num_particles
        sel = rng.choice(n_pairs, size=num_particles, replace=replace)
        init_pts = jnp.array(pts0[row_ind[sel]])
        final_pts = jnp.array(ptsT[col_ind[sel]])

        alpha = jnp.linspace(0.0, 1.0, num_steps)[:, None, None]
        trajectory = (1 - alpha) * init_pts[None] + alpha * final_pts[None]
    else:
        replace = pts0.shape[0] < num_particles
        sel = rng.choice(pts0.shape[0], size=num_particles, replace=replace)
        init_pts = jnp.array(pts0[sel])
        trajectory = jnp.broadcast_to(
            init_pts[None], (num_steps, num_particles, dim)
        ).copy()

    # Trajectory-coherent noise (1 % of data range per dimension)
    data_range = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-3)
    noise_scale = jnp.array(0.01 * data_range)
    offsets = jax.random.normal(k1, (num_particles, dim)) * noise_scale
    trajectory = trajectory + offsets[None, :, :]

    return trajectory, t_grid, k2


def mccann_interpolate_trajectories(
    train_data: SpatioTemporalData,
    num_particles: int,
    num_steps: int,
    key: jax.Array,
    *,
    cov_eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Build initial Stitching trajectories via the Gaussian OT geodesic.

    Fits a Gaussian to the first and last observed snapshots and transports one
    shared batch of standard-normal samples ``eps_i`` along the
    Bures–Wasserstein (McCann displacement) interpolation between them:

    .. math::

        x_i(a) = m(a) + [(1-a) I + a A]\,(x_i(0) - m_0),
        \quad x_i(0) = m_0 + \Sigma_0^{1/2} \varepsilon_i,

    where ``a`` runs linearly over ``num_steps`` grid points, ``m(a)`` is the
    interpolated mean, and ``A`` is the linear OT map from ``N(m0, C0)`` to
    ``N(mT, CT)`` (``bures_map``). Because ``eps_i`` is fixed across time and the
    map is a straight affine push-forward, each particle traces a smooth,
    non-crossing (laminar) path — this is the optimal-transport interpolation
    between the two Gaussians, so it does not braid (preferred over factor-linear
    or covariance-linear seedings, which can twist when the terminal covariances
    are oriented differently).

    Used by :meth:`Stitching.from_data` when ``trajectory_init="mccann"``.
    Returns ``(trajectory, t_grid, remaining_key)`` with the same shapes and
    key-reuse contract as :func:`ot_interpolate_trajectories`; ``trajectory`` is
    ``(num_steps, num_particles, D)`` and ``t_grid`` is ``(num_steps,)``. No
    jitter is added — independent ``eps_i`` already keep particles distinct, and
    jitter would perturb the exact terminal covariance. ``cov_eps`` regularises
    the covariances so ``C0`` is invertible (the tight ``t=0`` blob is
    near-singular).
    """
    dim = train_data.x.shape[1]
    t_unique = jnp.unique(train_data.t)
    t_grid = jnp.linspace(float(t_unique[0]), float(t_unique[-1]), num_steps)

    k1, k2 = jax.random.split(key)

    points = np.asarray(train_data.x)
    obs_times = np.asarray(train_data.t).ravel()
    unique_np = np.unique(obs_times)

    rng = np.random.default_rng(int(jax.random.randint(k1, (), 0, 2**30)))
    eps = rng.standard_normal((num_particles, dim))  # shared across time

    reg = cov_eps * np.eye(dim)
    m0, C0 = gaussian_moments(points[obs_times == unique_np[0]])
    C0 = C0 + reg
    x0 = m0 + eps @ psd_sqrt(C0).T  # (N, D) ~ N(m0, C0)

    alpha = np.linspace(0.0, 1.0, num_steps)
    if unique_np.size >= 2:
        mT, CT = gaussian_moments(points[obs_times == unique_np[-1]])
        CT = CT + reg
        ot_map = bures_map(C0, CT)  # SPD, ot_map @ C0 @ ot_map == CT
        maps = (1.0 - alpha)[:, None, None] * np.eye(dim)[None] + alpha[
            :, None, None
        ] * ot_map[None]  # (T, D, D)
        means = (1.0 - alpha)[:, None] * m0[None] + alpha[:, None] * mT[None]  # (T, D)
        trajectory = means[:, None, :] + np.einsum("tij,nj->tni", maps, x0 - m0)
    else:
        trajectory = np.broadcast_to(x0[None], (num_steps, num_particles, dim)).copy()

    return jnp.asarray(trajectory, dtype=jnp.float32), t_grid, k2


# ---------------------------------------------------------------------------
# Composite training objective
# ---------------------------------------------------------------------------


class StitchingLoss(eqx.Module):
    """Stitching training objective: NLL + WGF consistency residual.

    The consistency residual is a convex blend (``residual_alpha``) of:
    - α = 0: deterministic centres-quadrature (``wgf_residual``).
    - α = 1: stochastic KDE-MC with Nadaraya–Watson velocity
      (``wgf_kde_sampled_residual``).
    """

    w_nll: float = eqx.field(static=True, default=1.0)
    w_consistency: float = eqx.field(static=True, default=1.0)
    scheme: str = eqx.field(static=True, default="trapez")
    norm: str = eqx.field(static=True, default="raw")
    kinetic: float = eqx.field(static=True, default=1.0)
    residual_alpha: float = eqx.field(static=True, default=0.0)
    freeze_entropy: float | None = eqx.field(static=True, default=None)
    freeze_potential: float | None = eqx.field(static=True, default=None)
    freeze_interaction: float | None = eqx.field(static=True, default=None)

    def _functional_at(self, model: Stitching) -> Callable[[jax.Array], WGFFunctional]:
        kw = dict(
            freeze_entropy=self.freeze_entropy,
            freeze_potential=self.freeze_potential,
            freeze_interaction=self.freeze_interaction,
        )
        if model.time_conditioned:
            return lambda t: model.functional(t=t, **kw)
        F_static = model.functional(**kw)
        return lambda _t: F_static

    def __call__(
        self,
        model: Stitching,
        data: SpatioTemporalData,
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        if not 0.0 <= self.residual_alpha <= 1.0:
            raise ValueError(
                f"residual_alpha must be in [0, 1]; got {self.residual_alpha}"
            )

        functional_at = self._functional_at(model)
        is_time_cond = model.time_conditioned
        curve = model.density_curve()
        l_nll = nll(data)(curve)

        # The two `if` blocks below are not mutually exclusive — for α in
        # (0, 1) both estimators are computed and blended below. Extremes
        # (α=0 or α=1) short-circuit so the unused estimator is never invoked.
        if self.residual_alpha < 1.0:
            l_center = wgf_residual(
                functional_at,
                scheme=self.scheme,
                norm=self.norm,
                kinetic=self.kinetic,
            )(curve)
        if self.residual_alpha > 0.0:
            if is_time_cond:
                raise NotImplementedError(
                    "Time-conditioned potential not yet supported with"
                    " residual_alpha > 0 (KDE-MC residual)."
                )
            l_kde = wgf_kde_sampled_residual(
                functional_at,
                kinetic=self.kinetic,
            )(curve, key)

        if self.residual_alpha == 0.0:
            l_con = l_center
        elif self.residual_alpha == 1.0:
            l_con = l_kde
        else:
            l_con = (1.0 - self.residual_alpha) * l_center + self.residual_alpha * l_kde

        total = self.w_nll * l_nll + self.w_consistency * l_con
        aux: dict[str, jax.Array] = {
            "nll": l_nll,
            "residual": l_con,
            "bw": jnp.mean(model.bandwidth),
        }
        return total, aux

    def __repr__(self) -> str:
        bits = [
            f"scheme={self.scheme!r}",
            f"w_nll={self.w_nll}",
            f"w_consistency={self.w_consistency}",
        ]
        if self.residual_alpha > 0.0:
            bits.append(f"residual_alpha={self.residual_alpha}")
        return f"StitchingLoss({', '.join(bits)})"
