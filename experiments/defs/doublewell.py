"""Double-well — Stitching under three observation regimes, a 2×3 panel.

Illustrative (NOT a benchmark): the 1-D double-well V(x) = 0.5(|x|-7)² observed
three ways, each a data-preset variant composed with ``doublewell_panel``:

  - ``doublewell``           — 20 dense evenly-spaced snapshots × 500 cells
  - ``doublewell-terminal``  — only initial + terminal snapshot (2 × 500)
  - ``doublewell-trickle``   — 1000 fine snapshots × 10 cells (uncoupled)

The ``plot`` stage renders a 2×3 panel (data scatter on top, learned V_θ vs true
V* below). No metrics stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from experiments.registry import ExperimentSpec, Variant, register

if TYPE_CHECKING:
    from experiments._engine import Context

VARIANTS = ("doublewell", "doublewell-terminal", "doublewell-trickle")


def _plot_panel(ctx: Context) -> None:
    """2×3 panel: data scatter (top) + learned V vs true V (bottom)."""
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from stitching.synthetic.potentials import doublewell_1d as V_truth

    loaded = ctx.load_variants(*VARIANTS)
    results = {v: (loaded[v].model, loaded[v].train_data) for v in VARIANTS}
    out_path = ctx.figure_path("doublewell")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    titles = (
        "dense (20 snapshots × 500)",
        "terminal (t=0, t=4)",
        "trickle (1000 × 10)",
    )
    x_grid = np.linspace(-12, 12, 400).astype(np.float32)
    V_true_vals = np.asarray(jax.vmap(V_truth)(jnp.asarray(x_grid)))

    for j, (variant, title) in enumerate(zip(VARIANTS, titles)):
        model, train_data = results[variant]

        # --- Top: data scatter -------------------------------------------
        ax_d = axes[0, j]
        x_obs = np.asarray(train_data.x)[:, 0]
        t_obs = np.asarray(train_data.t)
        ax_d.scatter(t_obs, x_obs, s=4, alpha=0.4, c="tab:gray")
        ax_d.set_title(title)
        ax_d.set_xlabel("t")
        ax_d.set_ylabel("x")
        ax_d.set_ylim(-12, 12)

        # --- Bottom: learned V_θ vs true V*  -----------------------------
        ax_v = axes[1, j]
        c_pot = float(model.potential_coeff)
        V_learned = c_pot * np.asarray(
            jax.vmap(lambda x: jnp.squeeze(model.potential_net(x)))(
                jnp.asarray(x_grid)[:, None],
            )
        )
        # Centre both for shape comparison (V is identifiable up to const).
        ax_v.plot(
            x_grid, V_true_vals - V_true_vals.mean(), "k--", lw=1.5, label=r"$V^\star$"
        )
        ax_v.plot(
            x_grid,
            V_learned - V_learned.mean(),
            c="#0b6d2c",
            lw=2.0,
            label=r"$V^\theta$",
        )
        ax_v.set_xlabel("x")
        ax_v.set_ylabel("V (centred)")
        if j == 0:
            ax_v.legend(loc="upper center", ncols=2, frameon=False, fontsize=10)

    fig.suptitle("Double-well: Stitching under three observation regimes", fontsize=13)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


SPEC = ExperimentSpec(
    name="doublewell",
    description="Double-well: Stitching under three observation regimes (2×3 panel).",
    variants=tuple(Variant(v, f"{v}:stitching:doublewell_panel") for v in VARIANTS),
    plots={"panel": _plot_panel},
)

register(SPEC)
