"""Chiral McKean–Vlasov — train Stitching, render ``figures/chiral_snapshots.pdf``.

Composite preset ``chiral:stitching:chiral_fig5`` (data + run knobs in
``configs/data/chiral.json`` and ``configs/experiment/chiral_fig5.json``). The
chiral dataset (``data/chiral/chiral-simulation.npz``) comes from the
non-conservative kernel

    K(r) = ρ·(-∇W) + ω·R₉₀(-∇W),    W = generalised Morse,

with ω = 1.5, ρ = 0.2 → not a Wasserstein gradient flow. The radial Stitching
model can only match the conservative half of K (the Morse modulus W), the
paper's intended failure mode.

One variant, one figure: a 2-row × T-col panel (data scatter on top, Stitching
samples below). No metrics stage (wall-time only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiments.registry import ExperimentSpec, Variant, register

if TYPE_CHECKING:
    from experiments._engine import Context


def _plot_snapshots(ctx: Context) -> None:
    """2-row × T-col panel: data scatter (top) + Stitching samples (bottom)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from stitching.utils.paper_plots import (
        all_snap_times,
        blank_axes,
        frame_limits,
        pool_observations,
        scatter_row,
        snapshot_panels,
    )
    from stitching.utils.runners import sample_model_panels

    run = ctx.load("chiral")
    train_data, test_data = run.train_data, run.test_data
    snap_times = all_snap_times(train_data, test_data)
    out_path = ctx.figure_path("chiral_snapshots")

    obs_t, obs_x = pool_observations(train_data, test_data)
    data_panels = snapshot_panels(obs_t, obs_x, snap_times)
    n_med = int(np.median([p.shape[0] for _, p in data_panels]))
    sample_panels = sample_model_panels(run.model, snap_times, n_med)

    xlim, ylim = frame_limits([p for _, p in data_panels + sample_panels])

    n_snap = len(snap_times)
    fig, axes = plt.subplots(
        2,
        n_snap,
        layout="constrained",
        figsize=(9.2, 9.2 * 2.97 / (1.35 * n_snap)),
    )

    scatter_row(axes[0], data_panels, color="tab:gray", s=2.0, alpha=0.6)
    for j, (tv, _pts) in enumerate(data_panels):
        axes[0, j].set_title(f"$t={tv:g}$")
    scatter_row(axes[1], sample_panels, color="#b03030", s=2.0, alpha=0.55)

    axes[0, 0].set_ylabel("data")
    axes[1, 0].set_ylabel("samples")
    blank_axes(axes.ravel(), xlim, ylim, grid=True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out_path}")


SPEC = ExperimentSpec(
    name="chiral",
    description="Chiral McKean–Vlasov: Stitching snapshot panel (Fig. 5).",
    variants=(Variant("chiral", "chiral:stitching:chiral_fig5"),),
    plots={"snapshots": _plot_snapshots},
)

register(SPEC)
