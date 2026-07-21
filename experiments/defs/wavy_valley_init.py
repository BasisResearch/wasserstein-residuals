"""Wavy-valley trajectory-init comparison — OT vs McCann seeding.

Two variants over the shared wavy-valley data schedule, differing only in
``trajectory_init``: ``ot`` (Hungarian-OT match + linear interp, the default)
and ``mccann`` (Bures–Wasserstein / OT-geodesic Gaussian init). Two plots:

    init  the initial particle bundle each seeding produces, side by side over
          the true potential — the crisscross check, needs NO training
          (``python -m experiments plot wavy_valley_init --plot init``).
    grid  the headline 2x3 comparison — columns ``GT / OT / McCann``, rows
          ``initialization / trained``. Row 1 overlays each seeding's initial
          bundle on the true potential; row 2 shows each method's learned
          ``V^theta`` with its trained bundle; the ``GT`` column is the true
          potential in both rows. Every panel shares one symmetric colour scale
          and a single colorbar (``python -m experiments all wavy_valley_init``).

Note: the wavy-valley t=0 snapshot is a tight, near-isotropic blob, so OT and
McCann seedings look nearly identical here — this experiment confirms *neither
braids*, rather than discriminating the two schemes (which only diverge for
anisotropic terminal covariances with different orientations). See
``docs/trajectory_init.md``.

``--smoke`` trims to 100 epochs (wall-time only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from experiments.registry import ExperimentSpec, Variant, register

if TYPE_CHECKING:
    from experiments._engine import Context

STITCH_CONFIG = "wavy_valley:stitching:wavy_valley_fig1_stitching"

_X_RANGE, _Y_RANGE = (-4.0, 6.0), (-2.0, 2.0)
_INIT_COLORS = {"ot": "#0b6d2c", "mccann": "#1f4e9c"}
_INIT_TITLES = {"ot": "(a) OT init", "mccann": "(b) McCann init"}
# The combined grid uses one consistent trajectory colour across methods (columns
# already label the method) and short column titles.
_TRAJ_COLOR = "#0b6d2c"
_METHOD_TITLES = {"ot": "OT", "mccann": "McCann"}


def _potential_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, Y, Z)`` mesh of the (mean-centred) wavy-valley potential."""
    import jax
    import jax.numpy as jnp

    from stitching.synthetic.potentials import wavy_valley

    xx = np.linspace(*_X_RANGE, 220)
    yy = np.linspace(*_Y_RANGE, 110)
    X, Y = np.meshgrid(xx, yy)
    pts = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32))
    Z = np.asarray(jax.vmap(wavy_valley)(pts)).reshape(X.shape)
    return X, Y, Z - Z.mean()


def _draw_snapshots(ax, train_data) -> None:
    """Scatter the observed snapshots, coloured per time."""
    import matplotlib.pyplot as plt

    obs_t = np.asarray(train_data.t)
    obs_x = np.asarray(train_data.x)
    snap_t = np.unique(obs_t)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(snap_t)))
    for j, t in enumerate(snap_t):
        sn = obs_x[np.isclose(obs_t, t, atol=1e-6)]
        ax.scatter(
            sn[:, 0],
            sn[:, 1],
            s=14,
            c=[colors[j]],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
            zorder=6,
        )


def _plot_init(ctx: Context) -> None:
    """Side-by-side initial particle bundles for each seeding (no training)."""
    import jax
    import matplotlib.pyplot as plt

    from stitching.build import build_model
    from stitching.data import load_data
    from stitching.utils.paper_plots import filled_contour, trajectory_overlay

    X, Y, Z = _potential_grid()
    v = float(np.abs(Z).max()) or 1.0
    seed = ctx.spec.seeds[0]

    n = len(ctx.spec.variants)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(3.3 * n, 2.4),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    axes = axes[0]  # single row of init panels
    for ax, variant in zip(axes, ctx.spec.variants, strict=True):
        cfg = ctx.build_cfg(variant, seed)
        train_data, _ = load_data(cfg)
        # Match fit()'s key derivation (it splits the seed key and builds from the
        # first half) so the plotted bundle is exactly what training starts from.
        k_model = jax.random.split(jax.random.key(cfg.seed))[0]
        model = build_model(cfg, train_data, k_model)

        filled_contour(ax, X, Y, Z, vmin=-v, vmax=v, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        ax.set_title(_INIT_TITLES.get(variant.name, variant.name))
        _draw_snapshots(ax, train_data)
        trajectory_overlay(
            ax,
            np.asarray(model.trajectories),
            color=_INIT_COLORS.get(variant.name, "#333333"),
            n_show=cfg.num_particles,
            alpha=0.7,
            zorder=5,
        )
    axes[0].set_ylabel(r"$x_2$")

    out_path = ctx.figure_path("wavy_valley_init")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_grid(ctx: Context) -> None:
    """Headline 2x3 comparison: columns ``GT / OT / McCann``, rows init / trained.

    Row 1 (initialization) overlays each seeding's *initial* particle bundle on
    the true wavy-valley potential; row 2 (trained) shows each method's learned
    ``V^theta`` with its trained bundle faint on top. The ``GT`` column is the
    true potential (with observed snapshots) in both rows — the reference the
    learned fields approximate. Every panel — true and learned fields alike —
    shares one symmetric colour scale and a single colorbar, and all trajectory
    bundles use one consistent colour so the columns are directly comparable.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from stitching.build import build_model
    from stitching.data import load_data
    from stitching.utils.paper_plots import filled_contour, trajectory_overlay

    variants = {v.name: v for v in ctx.spec.variants}
    methods = list(variants)  # ["ot", "mccann"]
    seed = ctx.spec.seeds[0]

    X, Y, Z_true = _potential_grid()
    pts = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32))

    # Init-row seeds: rebuild exactly what training starts from (as in _plot_init).
    # The data schedule is identical across variants, so keep one train_data for
    # the GT column's snapshots.
    seeds_bundle: dict[str, np.ndarray] = {}
    train_data = None
    for name in methods:
        cfg = ctx.build_cfg(variants[name], seed)
        train_data, _ = load_data(cfg)
        k_model = jax.random.split(jax.random.key(cfg.seed))[0]
        model = build_model(cfg, train_data, k_model)
        seeds_bundle[name] = np.asarray(model.trajectories)

    # Trained-row: reload checkpoints and their learned potential fields.
    loaded = ctx.load_variants(*methods)

    def learned_field(model) -> np.ndarray:
        c = float(model.potential_coeff)
        Z = c * np.asarray(
            jax.vmap(lambda x: jnp.squeeze(model.potential_net(x)))(pts)
        ).reshape(X.shape)
        return Z - Z.mean()

    fields = {n: learned_field(loaded[n].model) for n in methods}

    # One symmetric scale across the true potential and every learned field.
    v = max(
        [float(np.abs(Z_true).max())]
        + [float(np.abs(f).max()) for f in fields.values()]
    ) or 1.0

    ncols = 1 + len(methods)  # GT + one per method
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(3.4 * ncols, 3.6),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )

    def _field_panel(ax, Z, td, traj=None, traj_alpha=0.6) -> None:
        filled_contour(ax, X, Y, Z, vmin=-v, vmax=v, alpha=0.65)
        _draw_snapshots(ax, td)
        if traj is not None:
            trajectory_overlay(
                ax, traj, color=_TRAJ_COLOR, n_show=traj.shape[1],
                alpha=traj_alpha, zorder=5,
            )
        ax.set_aspect("equal")

    # Row 0 — initialization: true V everywhere, seeded bundle over the methods.
    _field_panel(axes[0, 0], Z_true, train_data)
    for j, name in enumerate(methods, start=1):
        _field_panel(axes[0, j], Z_true, train_data, seeds_bundle[name], 0.7)

    # Row 1 — trained: true V reference (GT), learned field + trained bundle.
    _field_panel(axes[1, 0], Z_true, train_data)
    for j, name in enumerate(methods, start=1):
        run = loaded[name]
        _field_panel(
            axes[1, j], fields[name], run.train_data,
            np.asarray(run.model.trajectories), 0.4,
        )

    col_titles = ["GT"] + [_METHOD_TITLES.get(n, n) for n in methods]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)
    axes[0, 0].set_ylabel("initialization\n" + r"$x_2$")
    axes[1, 0].set_ylabel("trained\n" + r"$x_2$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$x_1$")

    sm = ScalarMappable(norm=Normalize(vmin=-v, vmax=v), cmap="RdYlBu_r")
    fig.colorbar(sm, ax=axes, shrink=0.85, label=r"$V$ (mean-centred)")

    out_path = ctx.figure_path("wavy_valley_init_grid")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out_path}")


SPEC = ExperimentSpec(
    name="wavy_valley_init",
    description="Wavy-valley: OT vs McCann trajectory-init comparison.",
    variants=(
        Variant("ot", STITCH_CONFIG, {"trajectory_init": "ot"}),
        Variant("mccann", STITCH_CONFIG, {"trajectory_init": "mccann"}),
    ),
    smoke_epochs=100,
    plots={"init": _plot_init, "grid": _plot_grid},
)

register(SPEC)
