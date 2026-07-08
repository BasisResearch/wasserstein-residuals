"""Turn a simulated trajectory into train/test `SpatioTemporalData`, plus the
top-level `load_synthetic` entry point used by the experiment harness.

The observation-protocol utilities are decoupled from simulation so that the
same subsampling logic can be applied to any trajectory source.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from stitching._kde import SpatioTemporalData
from stitching.velocity import WGFFunctional

from .potentials import (
    INTERACTION_FACTORIES,
    POTENTIAL_FACTORIES,
    POTENTIALS_1D,
    POTENTIALS_2D,
    make_from_spec,
)
from .systems import (
    SyntheticSystem,
    get_system,
    isotropic_normal,
    make_system,
    uniform_box,
)

if TYPE_CHECKING:
    from stitching.config import Config


# ---------------------------------------------------------------------------
# Trajectory → SpatioTemporalData
# ---------------------------------------------------------------------------


def trajectory_to_data(
    trajectory: jax.Array,
    t_grid: jax.Array,
    train_fraction: float = 0.5,
    seed: int = 0,
    scale_time: bool = False,
) -> tuple[SpatioTemporalData, SpatioTemporalData]:
    """Convert a `(T, N, D)` trajectory into train/test `SpatioTemporalData`.

    The split is done along **trajectories** (particle index) rather than time
    or flat samples, matching jkonet-star's default.  When ``train_fraction
    >= 1`` the full data is returned for both splits.  When ``scale_time`` is
    true the time axis is rescaled to ``[0, 1]``.
    """
    T, N, _ = trajectory.shape
    if scale_time:
        t_min, t_max = float(t_grid[0]), float(t_grid[-1])
        t_norm = (t_grid - t_min) / (t_max - t_min) if t_max > t_min else t_grid
    else:
        t_norm = t_grid

    all_xs = [trajectory[i] for i in range(T)]
    all_ts = [jnp.full(N, t_norm[i]) for i in range(T)]
    full = SpatioTemporalData(x=jnp.concatenate(all_xs), t=jnp.concatenate(all_ts))
    if train_fraction >= 1.0:
        return full, full

    rng = jax.random.key(seed)
    n_train = int(N * train_fraction)
    perm = jax.random.permutation(rng, N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_xs = [trajectory[i, train_idx] for i in range(T)]
    train_ts = [jnp.full(n_train, t_norm[i]) for i in range(T)]
    test_xs = [trajectory[i, test_idx] for i in range(T)]
    test_ts = [jnp.full(N - n_train, t_norm[i]) for i in range(T)]

    train = SpatioTemporalData(x=jnp.concatenate(train_xs), t=jnp.concatenate(train_ts))
    test = SpatioTemporalData(x=jnp.concatenate(test_xs), t=jnp.concatenate(test_ts))
    return train, test


# ---------------------------------------------------------------------------
# Observation protocols
# ---------------------------------------------------------------------------


def apply_observation_protocol(
    data: SpatioTemporalData,
    protocol: str = "dense",
    n_per_snapshot: int | None = None,
    snapshot_ratio: float = 1.0,
    seed: int = 0,
) -> SpatioTemporalData:
    """Subsample *data* according to an observation protocol.

    Protocols
    ---------
    dense
        Keep all snapshots and all particles (no subsampling).
    terminal
        Keep only the first and last snapshot (two marginals).
    irregular
        Keep ``snapshot_ratio`` fraction of interior snapshots, chosen
        randomly, always including the first and last.
    trickle
        Keep all snapshots but subsample to ``n_per_snapshot`` cells per
        snapshot — designed for many timepoints with very few cells each.
    """
    x_np = np.asarray(data.x)
    t_np = np.asarray(data.t)
    unique_times = np.unique(t_np)
    T = len(unique_times)
    rng = np.random.default_rng(seed)

    if protocol == "dense":
        keep_times = unique_times
    elif protocol == "terminal":
        keep_times = np.array([unique_times[0], unique_times[-1]])
    elif protocol == "irregular":
        if T <= 2:
            keep_times = unique_times
        else:
            interior = unique_times[1:-1]
            n_keep = max(1, int(len(interior) * snapshot_ratio))
            n_keep = min(n_keep, len(interior))
            kept_interior = rng.choice(interior, size=n_keep, replace=False)
            keep_times = np.sort(
                np.concatenate([[unique_times[0]], kept_interior, [unique_times[-1]]])
            )
    elif protocol == "trickle":
        keep_times = unique_times
    else:
        raise ValueError(f"Unknown observation protocol: {protocol!r}")

    mask = np.isin(t_np, keep_times)
    x_sub = x_np[mask]
    t_sub = t_np[mask]

    if protocol == "trickle" and n_per_snapshot is not None:
        xs, ts = [], []
        for tv in np.unique(t_sub):
            idx = np.where(np.isclose(t_sub, tv, atol=1e-8))[0]
            n = min(n_per_snapshot, len(idx))
            chosen = rng.choice(idx, size=n, replace=False)
            xs.append(x_sub[chosen])
            ts.append(t_sub[chosen])
        x_sub = np.concatenate(xs)
        t_sub = np.concatenate(ts)

    return SpatioTemporalData(x=jnp.array(x_sub), t=jnp.array(t_sub))


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def simulate_uncoupled(
    sys: SyntheticSystem,
    *,
    key: jax.Array,
    n_particles: int,
    n_timesteps: int,
    dt: float,
    n_substeps: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Simulate snapshot k=0..T as *independent* fresh runs of length k·dt.

    Matches the iJKOnet "unpaired" population-dynamics setting: snapshots are
    not temporally correlated. Each timepoint k is simulated from its own
    initial sample for exactly k SDE steps; the final state is kept.

    Returns ``(trajectory[T+1, N, D], t_grid[T+1])``.
    """
    keys = jax.random.split(key, n_timesteps + 1)
    snaps: list[jax.Array] = []
    t_grid_full = None
    for k in range(n_timesteps + 1):
        if k == 0:
            x = sys.init(keys[k], n_particles, sys.dim)
            snaps.append(x)
            continue
        traj_k, t_grid_k = sys.simulate(
            key=keys[k],
            n_particles=n_particles,
            n_timesteps=k,
            dt=dt,
            n_substeps=n_substeps,
        )
        if t_grid_full is None:
            t_grid_full = t_grid_k  # length k+1; reuse for k = n_timesteps
        snaps.append(traj_k[-1])
    trajectory = jnp.stack(snaps, axis=0)
    if t_grid_full is None or len(t_grid_full) != n_timesteps + 1:
        t_grid_full = jnp.arange(n_timesteps + 1, dtype=jnp.float32) * dt
    return trajectory, t_grid_full


def simulate_at_snapshot_times(
    sys: SyntheticSystem,
    *,
    key: jax.Array,
    n_particles: int,
    snapshot_times: tuple[float, ...],
    dt: float,
    n_substeps: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Simulate on a fine grid and subsample to irregular *snapshot_times*.

    The SDE is integrated at a uniform fine step of ``dt`` up to
    ``max(snapshot_times)``, then the snapshot nearest each requested time is
    kept. This expresses irregular observation schedules (e.g. 0, 5, 10, 20, 30)
    that the uniform ``n_timesteps × dt`` grid cannot.

    Args:
        sys: The synthetic system to integrate.
        key: PRNG key for the simulation.
        n_particles: Number of SDE particles.
        snapshot_times: Requested (possibly irregular) observation times.
        dt: Fine integration step.
        n_substeps: SDE substeps per fine step.

    Returns:
        ``(trajectory[len(snapshot_times), N, D], t_grid[len(snapshot_times)])``.
    """
    times = np.asarray(snapshot_times, dtype=float)
    t_max = float(times.max())
    n_fine = max(1, int(np.ceil(t_max / dt)))
    fine_dt = t_max / n_fine
    fine_traj, fine_grid = sys.simulate(
        key=key,
        n_particles=n_particles,
        n_timesteps=n_fine,
        dt=fine_dt,
        n_substeps=n_substeps,
    )
    fine_grid_np = np.asarray(fine_grid)
    idx = jnp.array([int(np.argmin(np.abs(fine_grid_np - t))) for t in times])
    return fine_traj[idx], jnp.asarray(times, dtype=jnp.float32)


def load_synthetic(
    system: str | SyntheticSystem,
    *,
    n_particles: int = 2000,
    n_timesteps: int = 5,
    dt: float = 0.01,
    n_substeps: int = 1,
    seed: int = 0,
    train_fraction: float = 0.5,
    scale_time: bool = False,
    obs_protocol: str = "dense",
    obs_n_per_snapshot: int | None = None,
    obs_snapshot_ratio: float = 1.0,
    coupled: bool = True,
    snapshot_times: tuple[float, ...] | None = None,
) -> tuple[SpatioTemporalData, SpatioTemporalData]:
    """Simulate *system* and return (train_data, test_data).

    Parameters
    ----------
    system
        Either a name in the registry (e.g. ``"flowers"``, ``"doublewell"``)
        or a user-constructed `SyntheticSystem`.
    n_particles, n_timesteps, dt
        SDE simulation hyperparameters.
    seed
        PRNG seed shared between simulation and the train/test split.
    train_fraction, scale_time
        Forwarded to `trajectory_to_data`.
    obs_protocol, obs_n_per_snapshot, obs_snapshot_ratio
        Observation protocol applied to the *training* data only; the test
        split is always kept dense.
    coupled
        If True (default), a single trajectory is generated and snapshots
        come from the same particles (jkonet-star "paired" convention). If
        False, each snapshot is an independent fresh simulation — the
        iJKOnet "unpaired" / true population-dynamics setting.
    snapshot_times
        Irregular observation times. When set (coupled only), the system is
        simulated on a fine grid of step ``dt`` and subsampled to the nearest
        step per requested time, overriding the uniform ``n_timesteps`` grid.
    """
    sys = get_system(system) if isinstance(system, str) else system

    if snapshot_times is not None:
        if not coupled:
            raise ValueError("snapshot_times is only supported for coupled=True.")
        trajectory, t_grid = simulate_at_snapshot_times(
            sys,
            key=jax.random.key(seed),
            n_particles=n_particles,
            snapshot_times=snapshot_times,
            dt=dt,
            n_substeps=n_substeps,
        )
    elif coupled:
        trajectory, t_grid = sys.simulate(
            key=jax.random.key(seed),
            n_particles=n_particles,
            n_timesteps=n_timesteps,
            dt=dt,
            n_substeps=n_substeps,
        )
    else:
        trajectory, t_grid = simulate_uncoupled(
            sys,
            key=jax.random.key(seed),
            n_particles=n_particles,
            n_timesteps=n_timesteps,
            dt=dt,
            n_substeps=n_substeps,
        )
    train_data, test_data = trajectory_to_data(
        trajectory,
        t_grid,
        train_fraction=train_fraction,
        seed=seed,
        scale_time=scale_time,
    )
    if obs_protocol != "dense":
        train_data = apply_observation_protocol(
            train_data,
            protocol=obs_protocol,
            n_per_snapshot=obs_n_per_snapshot,
            snapshot_ratio=obs_snapshot_ratio,
            seed=seed,
        )
    return train_data, test_data


# ---------------------------------------------------------------------------
# Cfg-driven entry point
# ---------------------------------------------------------------------------


def load_synthetic_from_cfg(
    cfg: Config,
) -> tuple[SpatioTemporalData, SpatioTemporalData]:
    """Build a registered `SyntheticSystem` from *cfg* and simulate it.

    Reads ``cfg.potential`` (registered name), ``cfg.synth_beta`` (entropy),
    ``cfg.synth_init`` / ``cfg.synth_init_loc`` / ``cfg.synth_init_scale``
    (initial distribution), and forwards the simulation hyperparameters
    (``cfg.synth_n_particles``, ``cfg.synth_n_timesteps``, ...) to
    :func:`load_synthetic`.
    """
    assert cfg.potential is not None, "synthetic mode requires cfg.potential"
    sys = get_system(cfg.potential)
    if cfg.synth_beta > 0.0:
        # Add the entropy term: it enters the SDE as diffusion σ = c_H·√2.
        f = sys.functional
        sys = dataclasses.replace(
            sys,
            functional=WGFFunctional(
                V=f.V, W=f.W, c_H=jnp.asarray(cfg.synth_beta), c_V=f.c_V, c_W=f.c_W
            ),
        )
    if cfg.synth_init == "normal":
        # synth_init_loc may be a scalar (broadcast over dims) or a per-dim vector.
        loc = jnp.asarray(cfg.synth_init_loc, dtype=jnp.float32)
        init_fn = isotropic_normal(loc, cfg.synth_init_scale)
    else:
        init_fn = uniform_box(-cfg.synth_init_scale, cfg.synth_init_scale)
    sys = dataclasses.replace(sys, init=init_fn)
    return load_synthetic(
        sys,
        n_particles=cfg.synth_n_particles,
        n_timesteps=cfg.synth_n_timesteps,
        dt=cfg.synth_dt,
        n_substeps=cfg.synth_n_substeps,
        seed=cfg.seed,
        train_fraction=cfg.synth_train_fraction,
        scale_time=cfg.scale_time,
        obs_protocol=cfg.obs_protocol,
        obs_n_per_snapshot=cfg.obs_n_per_snapshot,
        obs_snapshot_ratio=cfg.obs_snapshot_ratio,
        coupled=cfg.synth_coupled,
        snapshot_times=cfg.synth_snapshot_times,
    )


def _resolve_mv_preset(cfg: Config) -> dict:
    """Resolve ``cfg.mv_preset`` to an in-memory preset dict."""
    preset = cfg.mv_preset
    if preset is None:
        raise ValueError("dataset='mckean-vlasov' requires mv_preset to be set")
    if isinstance(preset, str):
        p = Path(preset)
        if not p.is_absolute():
            p = p.resolve()
        with p.open("r", encoding="utf-8") as f:
            preset = json.load(f)
    return preset


def load_mckean_vlasov_data(
    cfg: Config,
) -> tuple[SpatioTemporalData, SpatioTemporalData, dict]:
    """Simulate a McKean-Vlasov system from a JSON preset.

    Parallel on-ramp to :func:`load_synthetic_from_cfg`: instead of
    looking up a registered closure, ``cfg.mv_preset`` specifies V/W via
    parametric factories::

        {
          "simulation": {"n_particles": 100, "sigma": 0.045, ...},
          "potentials": {
            "self_potential": {"type": "DoubleWellPotential", "params": {...}},
            "interaction_potential": {"type": "GaussianInteraction", "params": {...}}
          },
          "initialization": {"strategy": "gaussian", "std": 1.0, "seed": 42}
        }

    Returns ``(train_data, test_data, metadata)`` — *metadata* carries
    the true V/W callables, observation times, and raw trajectories for
    downstream comparison.
    """
    preset = _resolve_mv_preset(cfg)

    pot = preset.get("potentials", {})
    V = (
        make_from_spec(pot["self_potential"], POTENTIAL_FACTORIES)
        if "self_potential" in pot
        else None
    )
    W = (
        make_from_spec(pot["interaction_potential"], INTERACTION_FACTORIES)
        if "interaction_potential" in pot
        else None
    )

    sim = preset.get("simulation", {})
    n_particles = sim.get("n_particles", 100)
    sigma = sim.get("sigma", 0.0)
    dim = sim.get("dimension", cfg.dims)
    t0 = sim.get("t0", 0.0)
    t1 = sim.get("t1", 10.0)
    dt_save = sim.get("dt_save", 0.05)
    dt0 = sim.get("dt0", dt_save)

    # entropy_coeff = sigma^2 / 2 (since diffusion sigma = sqrt(2 * entropy_coeff))
    entropy_coeff = sigma**2 / 2.0
    n_timesteps = max(1, round((t1 - t0) / dt_save))
    n_substeps = max(1, round(dt_save / dt0))

    init = preset.get("initialization", {})
    init_std = init.get("std", 1.0)
    init_seed = init.get("seed", cfg.seed)

    sys = make_system(
        name="mckean-vlasov",
        V=V,
        entropy_coeff=entropy_coeff,
        W=W,
        W_coeff=1.0 if W is not None else 0.0,
        dim=dim,
        init=isotropic_normal(0.0, init_std),
    )
    trajectory, times = sys.simulate(
        key=jax.random.key(init_seed),
        n_particles=n_particles,
        n_timesteps=n_timesteps,
        dt=dt_save,
        n_substeps=n_substeps,
    )

    T_steps, N_part, D_dim = trajectory.shape
    x_all = np.asarray(trajectory).reshape(-1, D_dim).astype(np.float32)
    t_all = np.repeat(np.asarray(times).astype(np.float32), N_part)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(N_part)
    n_train = max(1, int(N_part * cfg.train_fraction))
    if n_train >= N_part:
        train_data = SpatioTemporalData(x=jnp.array(x_all), t=jnp.array(t_all))
        test_data = train_data
    else:
        train_idx = perm[:n_train]
        train_mask = np.isin(np.tile(np.arange(N_part), T_steps), train_idx)
        test_data = SpatioTemporalData(
            x=jnp.array(x_all[~train_mask]),
            t=jnp.array(t_all[~train_mask]),
        )
        train_data = SpatioTemporalData(
            x=jnp.array(x_all[train_mask]),
            t=jnp.array(t_all[train_mask]),
        )

    metadata = {
        "V_true": V,
        "W_true": W,
        "times": times,
        "trajectories": trajectory,
        "preset": preset,
    }
    return train_data, test_data, metadata


# ---------------------------------------------------------------------------
# True-potential lookup (for visualisation in train/evaluate)
# ---------------------------------------------------------------------------


def _as_numpy_batch_eval(scalar_field, dim: int):
    """Wrap a scalar field `x: (D,) → scalar` as `pts: (N, D) np.ndarray → (N,) np.ndarray`.

    Two-dimensional visualisation grids always pass `(N, 2)`, so a 1-D
    scalar field is applied to the first coordinate only.
    """

    def _eval(pts: np.ndarray) -> np.ndarray:
        x = jnp.asarray(pts, dtype=jnp.float32)
        if dim == 1 and x.ndim == 2:
            x = x[:, 0]
        return np.asarray(jax.vmap(scalar_field)(x))

    return _eval


def get_true_potential(name: str | None):
    """Return a numpy-compatible ``pts → V(pts)`` callable, or ``None``.

    Looks up the raw scalar potential by name from the potential registries.
    Legacy dataset strings prefixed with ``synthetic-`` are accepted.
    """
    if name is None:
        return None
    if name.startswith("synthetic-"):
        name = name[len("synthetic-") :]
    all_potentials = {**POTENTIALS_2D, **POTENTIALS_1D}
    V = all_potentials.get(name)
    if V is None:
        return None
    dim = 1 if name in POTENTIALS_1D else 2
    return _as_numpy_batch_eval(V, dim)


def get_true_potential_grad(name: str | None):
    """Return a numpy-compatible ``pts → ∇V(pts)`` callable, or ``None``."""
    if name is None:
        return None
    if name.startswith("synthetic-"):
        name = name[len("synthetic-") :]
    all_potentials = {**POTENTIALS_2D, **POTENTIALS_1D}
    V = all_potentials.get(name)
    if V is None:
        return None
    grad_V = jax.vmap(jax.grad(V))

    def _eval(pts: np.ndarray) -> np.ndarray:
        x = jnp.asarray(pts, dtype=jnp.float32)
        return np.asarray(grad_V(x))

    return _eval
