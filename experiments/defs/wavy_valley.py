"""Wavy-valley fig 1 — Stitching + Lightspeed, the 3-panel potential figure.

Two variants (``wavy_valley:stitching:wavy_valley_fig1_stitching`` and
``…:lightspeed:wavy_valley_fig1_lightspeed``) over one shared data schedule
(irregular snapshots 0, 5, 10, 20, 30; β = 0.00625; normal init at [-3, 1]) — the
two composites pin an *identical* data preset, so each variant loads the same
train/test split deterministically. The ``plot`` stage renders the side-by-side
V*/V_θ_stitching/V_θ_lightspeed contour with the data snapshots, the Stitching
particle trajectory, and the Lightspeed OT chords. No metrics stage.

``--smoke`` trims to 100 epochs (wall-time only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from experiments.registry import ExperimentSpec, Variant, register

if TYPE_CHECKING:
    from experiments._engine import Context

STITCH_CONFIG = "wavy_valley:stitching:wavy_valley_fig1_stitching"
LIGHT_CONFIG = "wavy_valley:lightspeed:wavy_valley_fig1_lightspeed"


def _plot_potentials(ctx: Context) -> None:
    """3-panel side-by-side comparison of true vs learned potentials."""
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from stitching.synthetic.potentials import wavy_valley as v_true
    from stitching.utils.ot import hungarian_match
    from stitching.utils.paper_plots import filled_contour, trajectory_overlay

    loaded = ctx.load_variants("stitching", "lightspeed")
    stitch = loaded["stitching"].model
    train_data = loaded["stitching"].train_data
    light = loaded["lightspeed"].model
    out_path = ctx.figure_path("wavy_valley_potentials")

    x_range, y_range = (-4.0, 6.0), (-2.0, 2.0)
    n = 220
    xx = np.linspace(*x_range, n)
    yy = np.linspace(*y_range, n // 2)
    X, Y = np.meshgrid(xx, yy)
    pts_jax = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32))

    def learned(model):
        c = float(model.potential_coeff)
        return c * np.asarray(
            jax.vmap(lambda x: jnp.squeeze(model.potential_net(x)))(pts_jax)
        ).reshape(X.shape)

    Z_true = np.asarray(jax.vmap(v_true)(pts_jax)).reshape(X.shape)
    Z_stitch = learned(stitch)
    Z_light = learned(light)
    panels = [
        (r"(a) True $V^\star$", Z_true),
        (r"(b) Stitching $V^\theta$", Z_stitch),
        (r"(c) Lightspeed $V^\theta$", Z_light),
    ]
    centered = [(name, Z - Z.mean()) for name, Z in panels]

    fig, axes = plt.subplots(
        1, 3, figsize=(9.72, 2.304), sharex=True, sharey=True, layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, hspace=0.0, wspace=0.02)
    for ax, (name, Z) in zip(axes, centered, strict=True):
        v = float(np.abs(Z).max()) or 1.0
        filled_contour(ax, X, Y, Z, vmin=-v, vmax=v, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
        if ax is axes[0]:
            ax.set_ylabel(r"$x_2$")
        ax.set_title(name)

    # Snapshot scatter (per-time colored)
    obs_t = np.asarray(train_data.t)
    obs_x = np.asarray(train_data.x)
    snap_t = np.unique(obs_t)
    snap_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(snap_t)))
    snap_show = [obs_x[np.isclose(obs_t, t, atol=1e-6)] for t in snap_t]
    for ax in axes:
        for j, sn in enumerate(snap_show):
            ax.scatter(
                sn[:, 0],
                sn[:, 1],
                s=14,
                c=[snap_colors[j]],
                alpha=0.85,
                edgecolors="white",
                linewidths=0.3,
                zorder=6,
            )

    # Stitching: trainable trajectory
    trajectory_overlay(
        axes[1], stitch.trajectories, color="#0b6d2c", n_show=40, alpha=0.55, zorder=5
    )

    # Lightspeed: OT chord between consecutive snapshots
    for j in range(len(snap_show) - 1):
        a, b = snap_show[j], snap_show[j + 1]
        m = min(a.shape[0], b.shape[0])
        a, b = a[:m], b[:m]
        row, col = hungarian_match(a, b)
        for r, c in zip(row, col, strict=True):
            axes[2].plot(
                [a[r, 0], b[c, 0]],
                [a[r, 1], b[c, 1]],
                color="#7a0a14",
                lw=0.6,
                alpha=0.5,
                zorder=5,
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="0.4",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6,
            label="Data snapshots",
        ),
        Line2D(
            [0],
            [0],
            color="#0b6d2c",
            lw=2.0,
            label="Stitching: KDE-particle trajectory",
        ),
        Line2D([0], [0], color="#7a0a14", lw=2.0, label="Lightspeed: JKO OT-chord"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        frameon=False,
        fontsize=9,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out_path}")


SPEC = ExperimentSpec(
    name="wavy_valley",
    description="Wavy-valley: Stitching vs Lightspeed potential panel (Fig. 1).",
    variants=(
        Variant("stitching", STITCH_CONFIG),
        Variant("lightspeed", LIGHT_CONFIG),
    ),
    smoke_epochs=100,
    plots={"potentials": _plot_potentials},
)

register(SPEC)
