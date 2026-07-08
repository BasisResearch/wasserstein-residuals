"""RNA — full-data recovery panel + leave-two-out (gappy) interpolation table.

One experiment on the 5D RNA dataset, two hypotheses folded as variant groups:

  - ``full-{static,tcond}``  (``rna:stitching:rna_fig3``) — train on all observed
    timepoints; the ``plot`` stage renders the 2×9 V_θ(x) / V_θ(x,t) panel
    (``figures/rna_panels.pdf``) and the ``eval`` stage writes full-data EB metrics.
  - ``gappy-{static,tcond}`` (``rna-gappy:stitching:rna_gappy_table5``) —
    leave-two-out: train on t∈{0,2,4}, evaluate W2 at held-out t∈{1,3}; the
    ``eval`` stage writes ``rna_gappy_summary.csv`` (paper Table 5).

The static/tcond axis is ``time_conditioned_potential``. ``seeds`` defaults to a
single seed (fast reproduction); the paper's gappy table averages 5 seeds — set
``seeds=(0, 1, 2, 3, 4)`` for the full table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from experiments.registry import ExperimentSpec, Variant, register

if TYPE_CHECKING:
    from experiments._engine import Context

FULL = "rna:stitching:rna_fig3"
GAPPY = "rna-gappy:stitching:rna_gappy_table5"
GRID, N_BRIDGE = 120, 400


def _eval_grid(model, pc12, fixed_tail, t_value):
    """Evaluate V_θ on a PC1×PC2 grid (PC3..D fixed at fixed_tail, optional time)."""
    import jax
    import jax.numpy as jnp

    G = pc12.shape[0]
    flat = pc12.reshape(-1, 2)
    tail = np.broadcast_to(fixed_tail[None, :], (flat.shape[0], fixed_tail.shape[0]))
    full = np.concatenate([flat, tail], axis=1).astype(np.float32)
    if t_value is not None:
        full = np.concatenate(
            [full, np.full((full.shape[0], 1), t_value, dtype=np.float32)],
            axis=1,
        )
    return np.asarray(jax.vmap(model.potential_net)(jnp.asarray(full)).reshape(G, G))


def _bridge_samples(model, t_value, n, key):
    """Sample the KDE bridge marginal at the nearest trajectory node to t_value."""
    import jax.numpy as jnp

    from stitching._kde import IsotropicGaussian, KernelDensity

    idx = int(jnp.argmin(jnp.abs(model.t_grid - t_value)))
    return np.asarray(
        KernelDensity(
            model.trajectories[idx],
            kernel=IsotropicGaussian(bw=model.bandwidth),
            log_weights=getattr(model, "log_weights", None),
        ).sample(key=key, n=n)
    )


def _evaluate(ctx: Context) -> None:
    """Write the gappy leave-two-out summary CSV + the full-data metrics CSV.

    Each row's ``fit_id`` is the reloaded checkpoint's own ``manifest["fit_id"]``
    (not a recomputed HEAD id), so a row read off a stale checkpoint carries that
    checkpoint's id and any desync is visible in the CSV.
    """
    from stitching.evaluate import evaluate_test_metrics

    # Leave-two-out W2 table (paper Table 5) — byte-identical to run_rna_gappy.
    gappy_rows = []
    for vname, label in (("gappy-static", "static"), ("gappy-tcond", "tcond")):
        for seed in ctx.spec.seeds:
            run = ctx.load(vname, seed=seed)
            full = evaluate_test_metrics(
                run.model, run.train_data, run.test_data, eval_reps=run.cfg.eval_reps
            )
            gappy_rows.append(
                {
                    "variant": label,
                    "seed": seed,
                    "fit_id": run.manifest.get("fit_id", ""),
                    "wall_time_s": run.metrics.get("wall_time_s", 0.0),
                    "w2_mean": float(full["w2"]["mean"]),
                    **{
                        f"w2_t{t:.0f}": v
                        for t, v in sorted(full["w2"]["by_time"].items())
                    },
                }
            )
    path = ctx.write_csv(
        "rna_gappy_summary.csv",
        gappy_rows,
        fieldnames=(
            "variant",
            "seed",
            "fit_id",
            "w2_mean",
            "w2_t1",
            "w2_t3",
            "wall_time_s",
        ),
    )
    print(f"  gappy: {len(gappy_rows)} rows → {path}")

    # Full-data EB metrics (paper Table 2, ours) — additive over the old runner.
    full_rows = []
    for vname, label in (("full-static", "static"), ("full-tcond", "tcond")):
        run = ctx.load(vname)
        m = evaluate_test_metrics(
            run.model, run.train_data, run.test_data, eval_reps=run.cfg.eval_reps
        )
        full_rows.append(
            {
                "variant": label,
                "fit_id": run.manifest.get("fit_id", ""),
                "emd_w1": float(m["emd"]["mean"]),
                "w2": float(m["w2"]["mean"]),
                "mmd": float(m["mmd"]["mean"]),
            }
        )
    ctx.write_csv(
        "rna_metrics.csv",
        full_rows,
        fieldnames=("variant", "fit_id", "emd_w1", "w2", "mmd"),
    )


def _plot_panels(ctx: Context) -> None:
    """2-row × 9-col contour panel: static V_θ(x) and time-conditioned V_θ(x,t)."""
    import jax
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from stitching.utils.paper_plots import filled_contour

    loaded = ctx.load_variants("full-static", "full-tcond")
    static_model = loaded["full-static"].model
    tcond_model = loaded["full-tcond"].model
    test_data = loaded["full-static"].test_data
    out_path = ctx.figure_path("rna_panels")

    test_x = np.asarray(test_data.x)
    test_t = np.asarray(test_data.t)
    obs_times = sorted(np.unique(test_t).tolist())
    interp_times = [0.5 * (a + b) for a, b in zip(obs_times[:-1], obs_times[1:])]
    panel_times = sorted(obs_times + interp_times)

    pad = 0.5
    pc1_lo, pc1_hi = test_x[:, 0].min() - pad, test_x[:, 0].max() + pad
    pc2_lo, pc2_hi = test_x[:, 1].min() - pad, test_x[:, 1].max() + pad
    G1, G2 = np.meshgrid(
        np.linspace(pc1_lo, pc1_hi, GRID),
        np.linspace(pc2_lo, pc2_hi, GRID),
        indexing="xy",
    )
    pc12 = np.stack([G1, G2], axis=-1)
    fixed_tail = test_x.mean(axis=0).astype(np.float32)[2:]

    static_field = _eval_grid(static_model, pc12, fixed_tail, None)
    tcond_fields = [
        _eval_grid(tcond_model, pc12, fixed_tail, float(t)) for t in panel_times
    ]

    key = jax.random.key(0)
    bridge_static, bridge_tcond = [], []
    for t in panel_times:
        key, k1, k2 = jax.random.split(key, 3)
        bridge_static.append(_bridge_samples(static_model, float(t), N_BRIDGE, k1))
        bridge_tcond.append(_bridge_samples(tcond_model, float(t), N_BRIDGE, k2))

    fig, axes = plt.subplots(
        2,
        len(panel_times),
        sharex=True,
        sharey=True,
        layout="constrained",
        figsize=(0.958 * len(panel_times), 2.110),
    )

    rows = [
        (
            r"$V_\theta(x)$",
            [static_field] * len(panel_times),
            bridge_static,
            float(static_field.min()),
            float(static_field.max()),
        ),
        (
            r"$V_\theta(x,t)$",
            tcond_fields,
            bridge_tcond,
            float(np.stack(tcond_fields).min()),
            float(np.stack(tcond_fields).max()),
        ),
    ]
    DATA_COLOR, STITCH_COLOR = "#c8302c", "#0b6d2c"
    for r, (label, fields, bridges, vmin, vmax) in enumerate(rows):
        for c, (t, field, br) in enumerate(
            zip(panel_times, fields, bridges, strict=True)
        ):
            ax = axes[r, c]
            filled_contour(ax, G1, G2, field, vmin=vmin, vmax=vmax, alpha=0.55)
            if any(np.isclose(t, ot, atol=1e-6) for ot in obs_times):
                mask = np.isclose(test_t, t, atol=1e-6)
                ax.scatter(
                    test_x[mask, 0],
                    test_x[mask, 1],
                    s=4.0,
                    c=DATA_COLOR,
                    alpha=0.6,
                    edgecolors="white",
                    linewidths=0.3,
                    zorder=6,
                )
            ax.scatter(
                br[:, 0],
                br[:, 1],
                s=4.0,
                c=STITCH_COLOR,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.3,
                zorder=5,
            )
            if r == 0:
                title = f"$t={t:g}$"
                if not any(np.isclose(t, ot, atol=1e-6) for ot in obs_times):
                    title += "\n(interp.)"
                ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=11)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=DATA_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6,
            label="Held-out test cells",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=STITCH_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6,
            label="Stitching samples",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        fontsize=10,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out_path}")


SPEC = ExperimentSpec(
    name="rna",
    description="RNA: full-data V_θ panel (Fig. 3) + leave-two-out W2 table (Table 5).",
    variants=(
        Variant("full-static", FULL, {"time_conditioned_potential": False}),
        Variant("full-tcond", FULL, {"time_conditioned_potential": True}),
        Variant("gappy-static", GAPPY, {"time_conditioned_potential": False}),
        Variant("gappy-tcond", GAPPY, {"time_conditioned_potential": True}),
    ),
    evaluate=_evaluate,
    plots={"panels": _plot_panels},
)

register(SPEC)
