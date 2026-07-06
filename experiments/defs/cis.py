"""CIS — train Stitching, render ``figures/cis_snapshots.pdf`` + metrics.

2D Cell-Interaction Simulation (``cis:stitching:cis_fig4``: learned V_θ + radial
W_θ). One variant. The ``plot`` stage renders a 2-row × (n_snap+1)-col panel
(data / sample scatters + a V_θ-vs-V* contour and a W_θ-vs-W* line); the ``eval``
stage writes the held-out distributional metrics (paper Table 3, ours).

Ground-truth references: V*(x) = 0.1(‖x‖²-4)², W*(r) = -2.2 exp(-r²) (the CIS
preset's MV potentials).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from experiments.registry import ExperimentSpec, Variant, register

if TYPE_CHECKING:
    from experiments._engine import Context

CONFIG = "cis:stitching:cis_fig4"
TIMES = [0.0, 10.0, 20.0, 50.0, 100.0, 130.0, 170.0, 200.0]


def _pattern_r2(a, b) -> float:
    """Squared Pearson correlation (scale + shift invariant)."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / max(denom, 1e-12)) ** 2


def _evaluate(ctx: Context) -> None:
    """Held-out distributional metrics (EMD/W1, W2, MMD) → ``metrics.csv``."""
    from stitching.evaluate import evaluate_test_metrics
    from stitching.utils.persistence import fit_id_for

    run = ctx.load("cis")
    mfull = evaluate_test_metrics(
        run.model, run.train_data, run.test_data, eval_reps=run.cfg.eval_reps
    )
    row = {
        "variant": "cis",
        "fit_id": fit_id_for(run.train_data, run.test_data),
        "emd_w1": float(mfull["emd"]["mean"]),
        "w2": float(mfull["w2"]["mean"]),
        "mmd": float(mfull["mmd"]["mean"]),
    }
    path = ctx.write_csv(
        "metrics.csv", [row], fieldnames=("variant", "fit_id", "emd_w1", "w2", "mmd")
    )
    print(
        f"  EMD/W1={row['emd_w1']:.3f} W2={row['w2']:.3f} MMD={row['mmd']:.4f} → {path}"
    )


def _plot_snapshots(ctx: Context) -> None:
    """2-row × (T+1)-col panel: data / samples scatters + V_θ/W_θ comparison."""
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from stitching.synthetic.potentials import doublewell_nd, gaussian_interaction
    from stitching.utils.paper_plots import (
        blank_axes,
        frame_limits,
        pool_observations,
        scatter_row,
        snapshot_panels,
    )
    from stitching.utils.runners import sample_model_panels

    v_true = doublewell_nd(alpha=0.1, beta=4.0)
    w_true = gaussian_interaction(strength=-2.2, width=1.0)

    run = ctx.load("cis")
    model, train_data, test_data = run.model, run.train_data, run.test_data
    out_path = ctx.figure_path("cis_snapshots")

    obs_t, obs_x = pool_observations(train_data, test_data)
    snap_t = [
        float(np.unique(obs_t)[np.argmin(np.abs(np.unique(obs_t) - t))]) for t in TIMES
    ]
    data_panels = snapshot_panels(obs_t, obs_x, snap_t)
    n_med = int(np.median([p.shape[0] for _, p in data_panels]))
    sample_panels = sample_model_panels(model, snap_t, n_med)

    xlim, ylim = frame_limits([p for _, p in data_panels + sample_panels])

    n_snap = len(snap_t)
    fig, axes = plt.subplots(
        2,
        n_snap + 1,
        layout="constrained",
        figsize=(9.2, 9.2 * 2.97 / (1.35 * n_snap)),
    )

    scatter_row(axes[0], data_panels, color="tab:gray", s=2.0, alpha=0.6)
    for j, (tv, _pts) in enumerate(data_panels):
        axes[0, j].set_title(f"$t={tv:g}$")
    scatter_row(axes[1], sample_panels, color="#b03030", s=2.0, alpha=0.55)

    # V_θ vs V* contour (top of last column)
    n_grid = 80
    gx = np.linspace(*xlim, n_grid)
    gy = np.linspace(*ylim, n_grid)
    XX, YY = np.meshgrid(gx, gy)
    flat = jnp.asarray(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=jnp.float32)
    inp = (
        jnp.concatenate(
            [
                flat,
                jnp.full(
                    (flat.shape[0], 1), float(model.t_grid[len(model.t_grid) // 2])
                ),
            ],
            axis=1,
        )
        if model.time_conditioned
        else flat
    )
    V_l = (
        np.asarray(jax.vmap(model.potential_net)(inp).squeeze(-1))
        .reshape(n_grid, n_grid)
        .copy()
    )
    V_l -= V_l.mean()
    V_t = np.asarray(jax.vmap(v_true)(flat)).reshape(n_grid, n_grid)
    axes[0, -1].contourf(XX, YY, V_l, levels=18, cmap="RdBu_r")
    axes[0, -1].contour(
        XX, YY, V_t, levels=6, colors="k", linewidths=0.6, linestyles="--", alpha=0.85
    )
    axes[0, -1].set_title(
        rf"$V^\theta$ ($R^2$ {_pattern_r2(V_l, V_t):.2f})", fontsize=8
    )

    # W_θ vs W* line (bottom of last column)
    if model.interaction_net is not None:
        sel = np.asarray(train_data.x)[
            np.isclose(
                np.asarray(train_data.t),
                np.unique(np.asarray(train_data.t))[len(np.unique(train_data.t)) // 2],
                atol=1e-6,
            )
        ]
        d = np.linalg.norm(sel[:, None, :] - sel[None, :, :], axis=-1)
        d = d[np.triu_indices_from(d, k=1)]
        r_lo, r_hi = np.percentile(d, [5, 95]) if d.size > 0 else (0.0, 1.0)
        r = jnp.linspace(0.01, 5.0, 400)
        xs = (
            (r**2).reshape(-1, 1)
            if model.interaction_type == "radial"
            else jnp.stack([r, jnp.zeros_like(r)], axis=1)
        )
        W_l = np.asarray(jax.vmap(model.interaction_net)(xs).squeeze(-1))
        r_np = np.asarray(r)
        W_t = np.asarray([float(w_true(jnp.array([rv, 0.0]))) for rv in r_np])
        in_band = (r_np >= r_lo) & (r_np <= r_hi)
        ax_w = axes[1, -1]
        ax_w.plot(
            r_np, W_l - W_l[in_band].mean(), c="tab:blue", lw=1.2, label=r"$W^\theta$"
        )
        ax_w.plot(
            r_np, W_t - W_t[in_band].mean(), c="k", lw=0.9, ls="--", label=r"$W^\star$"
        )
        ax_w.axvspan(r_lo, r_hi, color="tab:gray", alpha=0.15)
        ax_w.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax_w.set_xlim(0.0, 5.0)
        ax_w.set_xlabel("$r$", fontsize=8)
        ax_w.set_title(
            rf"$W^\theta$ ($R^2$ {_pattern_r2(W_l[in_band], W_t[in_band]):.2f})",
            fontsize=8,
        )
        ax_w.legend(loc="lower right", fontsize=6, frameon=False)
        ax_w.tick_params(labelsize=6)

    axes[0, 0].set_ylabel("data")
    axes[1, 0].set_ylabel("samples")
    # The snapshot columns share the framed/gridded styling; the trailing
    # comparison column (V_θ contour over W_θ line) is styled separately above.
    blank_axes(axes[:, :n_snap].ravel(), xlim, ylim, grid=True)
    axes[0, -1].set_xticks([])
    axes[0, -1].set_yticks([])

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out_path}")


SPEC = ExperimentSpec(
    name="cis",
    description="CIS: Stitching V_θ/W_θ recovery panel + metrics (Fig. 4 / Table 3).",
    variants=(Variant("cis", CONFIG),),
    evaluate=_evaluate,
    plots={"snapshots": _plot_snapshots},
)

register(SPEC)
