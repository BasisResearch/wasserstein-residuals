"""Synthetic 2D panel — 15 potentials × {paired, unpaired}: true vs learnt V.

Trains Stitching on each of 15 synthetic 2D landscapes in both coupling regimes
(``synthetic:stitching:synthetic_panel``, sweeping ``potential`` and
``synth_coupled``). Each (potential, mode) is one variant → one checkpoint, so a
changed landscape retrains only its cell.

  - ``eval``  → ``synthetic_panel_summary.csv`` (per-cell EMD + ground-truth-aware
    functional metrics: potential/pattern R², L²-UVP, BW²-UVP). This is the CSV
    pinned bitwise by ``tests/test_parity.py``.
  - ``plot``  → several figures under ``figures/`` (select one with ``--plot <name>``):
      * ``panels`` → ``synthetic_panel_{paired,unpaired}.pdf`` (5×6 true/learned
        gallery).
      * ``compact`` → ``synthetic_compact.pdf`` (3×6 main-paper panel: true V*,
        learned paired, learned unpaired, over 6 selected potentials).
      * ``coupling`` → ``synthetic_coupling.pdf`` (OT-coupling cartoon on paired
        vs unpaired data; self-contained, no checkpoints).
      * ``paired_vs_unpaired`` → ``synthetics_paired_vs_unpaired.pdf`` (conceptual
        2×5 ancestry figure; self-contained Euler integration).
    All but ``panels`` are full-run paper figures and are skipped under ``--smoke``.

The contour grids are recomputed from the reloaded model (bit-exact), so the
panels no longer need the old per-cell ``.npz`` cache.

Sampling scheme (paper Sec. ``app:synthetic_full``): T=5 intervals at dt=0.01,
β=0, uniform init [-4,4]², 50/50 split. Stitching: 1000 particles, α=0, frozen
weights, 2000 epochs. ``--smoke`` → 50 epochs, 100 particles, 2 landscapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from experiments.registry import ExperimentSpec, Variant, register

if TYPE_CHECKING:
    from experiments._engine import Context

PANEL_CONFIG = "synthetic:stitching:synthetic_panel"
POTENTIALS = [
    "flowers",
    "styblinski_tang",
    "holder_table",
    "zigzag_ridge",
    "oakley_ohagan",
    "watershed",
    "ishigami",
    "friedman",
    "sphere",
    "bohachevsky",
    "wavy_plateau",
    "double_exp",
    "relu",
    "rotational",
    "flat",
]
SMOKE_POTENTIALS = ["flowers", "flat"]
_MODES = (("paired", True), ("unpaired", False))
CSV_KEYS = (
    "potential",
    "mode",
    "test_emd",
    "potential_r2",
    "pattern_r2",
    "l2_uvp",
    "bw2_uvp",
    "wall_time_s",
)

# Compact 3×6 paper panel: the 6 potentials from tab:synthetic2d_unpaired_ratio,
# in Table-1 order with their table numbers.
COMPACT_POTENTIALS = [
    ("flowers", 1),
    ("zigzag_ridge", 4),
    ("watershed", 6),
    ("ishigami", 7),
    ("friedman", 8),
    ("wavy_plateau", 11),
]
# OT-coupling cartoon (Sec. exp_synthetic): one fixed configuration.
COUPLING_POTENTIAL = "flowers"
COUPLING_N = 200
COUPLING_DT = 0.01
COUPLING_TIMESTEPS = 5
COUPLING_SEED = 0
# Paired-vs-unpaired ancestry figure: deterministic Euler over a small panel.
PVU_POTENTIALS = [
    "flowers",
    "styblinski_tang",
    "holder_table",
    "ishigami",
    "wavy_plateau",
]
PVU_N = 100
PVU_T = 5
PVU_DT = 0.01
PVU_INIT_SCALE = 4.0
PVU_SEED = 0


def _variant_name(pot: str, mode: str) -> str:
    return f"{pot}-{mode}"


def _grid(V_fn, X, Y):
    """Evaluate a scalar field on the X×Y mesh, returned as a 2D numpy array."""
    import jax
    import jax.numpy as jnp

    pts = jnp.stack([jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], axis=-1)
    return np.array(jax.vmap(V_fn)(pts)).reshape(X.shape)


def _evaluate(ctx: Context) -> None:
    """Per-cell EMD + functional metrics → ``synthetic_panel_summary.csv``."""
    from stitching.evaluate import run_synthetic_evaluation

    rows = []
    for variant, seed in ctx.cells():
        run = ctx.load(variant.name, seed=seed)
        test_emd, _, fm, _, _ = run_synthetic_evaluation(
            run.model, run.cfg, run.train_data, run.test_data
        )
        rows.append(
            {
                "potential": run.cfg.potential,
                "mode": "paired" if run.cfg.synth_coupled else "unpaired",
                "test_emd": float(test_emd),
                "wall_time_s": run.metrics.get("wall_time_s", 0.0),
                **{k: float(v) for k, v in fm.items() if not isinstance(v, dict)},
            }
        )
    path = ctx.write_csv("synthetic_panel_summary.csv", rows, fieldnames=CSV_KEYS)
    print(f"  {len(rows)} cells → {path}")


def _make_panel(panel, mode, X, Y, out, potentials, num_particles) -> None:
    """5×6 layout: each potential gets a (true | learned) cell pair."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        5, 6, figsize=(18, 15), layout="constrained", sharex=True, sharey=True
    )
    for i, pot in enumerate(potentials):
        row, c2 = divmod(i, 3)
        ax_t, ax_l = axes[row, c2 * 2], axes[row, c2 * 2 + 1]
        r = panel.get(pot)
        if r is None:
            ax_t.set_title(f"{pot}\nFAILED")
            continue
        true_c = r["true_vals"] - r["true_vals"].mean()
        learned_c = r["learned_vals"] - r["learned_vals"].mean()
        vmax = float(np.max(np.abs(true_c))) or 1.0
        kw = dict(levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_t.contourf(X, Y, true_c, **kw)
        ax_l.contourf(X, Y, learned_c, **kw)
        ax_t.set_title(f"{pot}\ntrue")
        # potential_r2 is undefined for flat/zero-variance V → show a dash.
        uvp = r["potential_r2"]
        uvp_s = f"{uvp:.2f}" if np.isfinite(uvp) and uvp > -1.0 else "—"
        ax_l.set_title(
            "learned\n"
            rf"$R^2_{{\mathrm{{pattern}}}}$ {r['pattern_r2']:.2f}  "
            rf"$R^2_{{\mathrm{{uvp}}}}$ {uvp_s}"
        )
        ax_t.set_aspect("equal")
        ax_l.set_aspect("equal")
    fig.suptitle(
        f"Stitching + {num_particles} particles, frozen weights — {mode} setup",
        fontsize=14,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _plot_panels(ctx: Context) -> None:
    """Render the paired/unpaired galleries from reloaded checkpoints + metrics CSV."""
    from stitching.synthetic.potentials import POTENTIALS_2D

    xs = np.linspace(-5.0, 5.0, 80)
    X, Y = np.meshgrid(xs, xs)
    metrics = {
        (r["potential"], r["mode"]): r
        for r in ctx.read_csv("synthetic_panel_summary.csv")
    }

    panels: dict[str, dict] = {"paired": {}, "unpaired": {}}
    num_particles = None
    for variant, seed in ctx.cells():
        run = ctx.load(variant.name, seed=seed)
        pot = run.cfg.potential
        mode = "paired" if run.cfg.synth_coupled else "unpaired"
        num_particles = run.cfg.num_particles
        c_pot = float(run.model.potential_coeff)
        m = metrics[(pot, mode)]
        panels[mode][pot] = {
            "true_vals": _grid(POTENTIALS_2D[pot], X, Y),
            "learned_vals": _grid(
                lambda x: c_pot * run.model.potential_net(x).reshape(()), X, Y
            ),
            "pattern_r2": float(m["pattern_r2"]),
            "potential_r2": float(m["potential_r2"]),
        }

    potentials = [
        p for p in POTENTIALS if p in panels["paired"] or p in panels["unpaired"]
    ]
    for mode in ("paired", "unpaired"):
        _make_panel(
            panels[mode],
            mode,
            X,
            Y,
            ctx.figure_path(f"synthetic_panel_{mode}"),
            potentials,
            num_particles,
        )


def _plot_compact(ctx: Context) -> None:
    """3×6 main-paper panel: true V*, learned (paired), learned (unpaired).

    Recomputes the contour grids from the reloaded checkpoints (bit-exact, same
    ``c·potential_net`` / ``POTENTIALS_2D`` evaluation the gallery uses) and reads
    the pattern-R² subtitles from ``synthetic_panel_summary.csv`` — so it replaces
    the old per-cell ``.npz`` cache. Needs the full 6-potential sweep.
    """
    if ctx.smoke:
        print("  compact: skipped (needs the full 6-potential sweep)")
        return
    import matplotlib.pyplot as plt

    from stitching.synthetic.potentials import POTENTIALS_2D

    xs = np.linspace(-5.0, 5.0, 80)
    X, Y = np.meshgrid(xs, xs)
    metrics = {
        (r["potential"], r["mode"]): r
        for r in ctx.read_csv("synthetic_panel_summary.csv")
    }

    fig, axes = plt.subplots(
        3, 6, figsize=(15, 7.5), layout="constrained", sharex=True, sharey=True
    )
    for col, (pot, num) in enumerate(COMPACT_POTENTIALS):
        paired = ctx.load(_variant_name(pot, "paired"))
        unpaired = ctx.load(_variant_name(pot, "unpaired"))
        cp_p = float(paired.model.potential_coeff)
        cp_u = float(unpaired.model.potential_coeff)
        true_p = _grid(POTENTIALS_2D[pot], X, Y)
        learned_p = _grid(
            lambda x: cp_p * paired.model.potential_net(x).reshape(()), X, Y
        )
        learned_u = _grid(
            lambda x: cp_u * unpaired.model.potential_net(x).reshape(()), X, Y
        )

        true_c = true_p - true_p.mean()
        paired_c = learned_p - learned_p.mean()
        unp_c = learned_u - learned_u.mean()
        vmax = float(np.max(np.abs(true_c)))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        kw = dict(levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, col].contourf(X, Y, true_c, **kw)
        axes[1, col].contourf(X, Y, paired_c, **kw)
        axes[2, col].contourf(X, Y, unp_c, **kw)

        pat_paired = float(metrics[(pot, "paired")]["pattern_r2"])
        pat_unpaired = float(metrics[(pot, "unpaired")]["pattern_r2"])
        axes[0, col].set_title(f"{num}. {pot}")
        axes[1, col].set_title(rf"$R^2_{{\mathrm{{pattern}}}}$ {pat_paired:.2f}")
        axes[2, col].set_title(rf"$R^2_{{\mathrm{{pattern}}}}$ {pat_unpaired:.2f}")
        for r in range(3):
            axes[r, col].set_aspect("equal")

    axes[0, 0].set_ylabel("ground truth $V^\\star$", fontsize=11)
    axes[1, 0].set_ylabel("learned (paired)", fontsize=11)
    axes[2, 0].set_ylabel("learned (unpaired)", fontsize=11)

    out_path = ctx.figure_path("synthetic_compact")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  saved {out_path}")


def _plot_coupling(ctx: Context) -> None:
    """OT-coupling cartoon: what JKOnet* couples on paired vs unpaired data.

    Self-contained (no checkpoints): draws N particles from Uniform([-4,4]²),
    integrates the deterministic gradient flow one Δt, and renders the OT coupling
    on the paired (shared identities) vs unpaired (independent draws) regime.
    """
    if ctx.smoke:
        print("  coupling: skipped in smoke")
        return
    import dataclasses

    import jax
    import matplotlib.pyplot as plt
    import ot

    from stitching.synthetic import get_system, uniform_box

    n = COUPLING_N
    sys_obj = get_system(COUPLING_POTENTIAL)
    # Force uniform [-4,4]^2 init so the visualisation matches sec 6.2.
    sys_obj = dataclasses.replace(sys_obj, init=uniform_box(-4.0, 4.0))

    # Paired: simulate one trajectory of N particles for T steps.
    traj, _t_grid = sys_obj.simulate(
        key=jax.random.key(COUPLING_SEED),
        n_particles=n,
        n_timesteps=COUPLING_TIMESTEPS,
        dt=COUPLING_DT,
        n_substeps=1,
    )
    x_paired = np.asarray(traj[0])  # t = 0
    y_paired = np.asarray(traj[1])  # t = dt

    # Unpaired: re-simulate from a fresh p0 for the t=dt snapshot.
    k0, k1 = jax.random.split(jax.random.key(COUPLING_SEED + 7))
    x_unp = np.asarray(sys_obj.init(k0, n, sys_obj.dim))
    traj_b, _ = sys_obj.simulate(
        key=k1, n_particles=n, n_timesteps=1, dt=COUPLING_DT, n_substeps=1
    )
    y_unp = np.asarray(traj_b[1])

    def hungarian(x, y):
        a = np.ones(x.shape[0]) / x.shape[0]
        M = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
        P = ot.emd(a, a, M)
        return P.argmax(axis=1)

    sigma_p = hungarian(x_paired, y_paired)
    sigma_u = hungarian(x_unp, y_unp)

    nat_p = float(np.linalg.norm(y_paired - x_paired, axis=-1).mean())
    ot_p = float(np.linalg.norm(y_paired[sigma_p] - x_paired, axis=-1).mean())
    match_p = float((sigma_p == np.arange(n)).mean())
    ot_u = float(np.linalg.norm(y_unp[sigma_u] - x_unp, axis=-1).mean())

    pts = np.concatenate([x_paired, y_paired, x_unp, y_unp])
    pad = 0.1 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    xlim = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ylim = (pts[:, 1].min() - pad, pts[:, 1].max() + pad)

    fig, axes = plt.subplots(
        1, 2, sharex=True, sharey=True, layout="constrained", figsize=(7.5, 3.7)
    )
    panels = [
        (
            "paired",
            x_paired,
            y_paired,
            sigma_p,
            f"\\textbf{{Paired}}: $\\sigma(i){{=}}i$ at "
            f"{match_p * 100:.0f}\\%; $|y{{-}}x|_\\mathrm{{nat}}{{=}}{nat_p:.3f}$, "
            f"$|y{{-}}x|_\\mathrm{{OT}}{{=}}{ot_p:.3f}$",
        ),
        (
            "unpaired",
            x_unp,
            y_unp,
            sigma_u,
            f"\\textbf{{Unpaired}}: $\\sigma(i){{=}}i$ undefined; "
            f"$|y{{-}}x|_\\mathrm{{OT}}{{=}}{ot_u:.3f}$",
        ),
    ]
    for ax, (label, xs, ys, sg, title) in zip(axes, panels):
        for i in range(xs.shape[0]):
            ax.plot(
                [xs[i, 0], ys[sg[i], 0]],
                [xs[i, 1], ys[sg[i], 1]],
                c="tab:blue",
                lw=0.5,
                alpha=0.55,
                zorder=1,
            )
        ax.scatter(
            xs[:, 0],
            xs[:, 1],
            s=8,
            c="tab:gray",
            alpha=0.85,
            zorder=3,
            label="$t_0{=}0$",
            rasterized=True,
        )
        ax.scatter(
            ys[:, 0],
            ys[:, 1],
            s=8,
            c="#b03030",
            alpha=0.85,
            zorder=3,
            label="$t_1{=}\\Delta t$",
            rasterized=True,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axisbelow(True)
        ax.grid(True, color="0.6", alpha=0.35, lw=0.4, zorder=0)
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, frameon=False)

    out_path = ctx.figure_path("synthetic_coupling")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {out_path}")
    print(
        f"  paired:   |y-x| natural = {nat_p:.4f}, OT = {ot_p:.4f}, "
        f"σ(i)==i overlap = {match_p * 100:.1f}%"
    )
    print(f"  unpaired: |y-x| OT      = {ot_u:.4f}")


def _plot_paired_vs_unpaired(ctx: Context) -> None:
    """2×N conceptual ancestry figure: paired (top) vs unpaired (bottom).

    Self-contained (no checkpoints): deterministic Euler integration of
    ``ẋ = -∇V(x)`` for a small panel of potentials.
    """
    if ctx.smoke:
        print("  paired_vs_unpaired: skipped in smoke")
        return
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from stitching.synthetic import potentials as P

    n, t_steps, dt, init_scale, seed = PVU_N, PVU_T, PVU_DT, PVU_INIT_SCALE, PVU_SEED
    potentials = [(name, getattr(P, name)) for name in PVU_POTENTIALS]

    def sample_init(key):
        return jax.random.uniform(key, (n, 2), minval=-init_scale, maxval=init_scale)

    def make_integrator(V_fn):
        step = jax.jit(jax.vmap(lambda x: x - dt * jax.grad(V_fn)(x)))

        def integrate(x0, n_steps):
            traj = [x0]
            x = x0
            for _ in range(n_steps):
                x = step(x)
                traj.append(x)
            return jnp.stack(traj)

        return integrate

    def draw_contour(ax, V_fn, Xg, Yg):
        pts = jnp.stack([jnp.asarray(Xg.ravel()), jnp.asarray(Yg.ravel())], axis=-1)
        V_grid = np.asarray(jax.vmap(V_fn)(pts)).reshape(Xg.shape)
        V_c = V_grid - V_grid.mean()
        vmax = float(np.max(np.abs(V_c))) or 1.0
        ax.contourf(
            Xg, Yg, V_c, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.55
        )

    def draw_paired(ax, V_fn, integrate, Xg, Yg, cmap_t):
        draw_contour(ax, V_fn, Xg, Yg)
        traj = np.asarray(integrate(sample_init(jax.random.key(seed)), t_steps))
        for j in range(n):
            ax.plot(traj[:, j, 0], traj[:, j, 1], color="black", lw=0.4, alpha=0.3)
        for k in range(t_steps + 1):
            ax.scatter(
                traj[k, :, 0],
                traj[k, :, 1],
                s=14,
                color=cmap_t(k / t_steps),
                edgecolors="white",
                linewidths=0.4,
            )

    def draw_unpaired(ax, V_fn, integrate, Xg, Yg, cmap_t):
        draw_contour(ax, V_fn, Xg, Yg)
        sub_keys = jax.random.split(jax.random.key(seed), t_steps + 1)
        snapshots: list = []
        origins: list = []
        for k in range(t_steps + 1):
            if k == 0:
                x0 = np.asarray(sample_init(sub_keys[0]))
                snapshots.append(x0)
                origins.append(x0)
                continue
            traj_k = np.asarray(integrate(sample_init(sub_keys[k]), k))
            snapshots.append(traj_k[-1])
            origins.append(traj_k[0])
        ax.scatter(
            snapshots[0][:, 0],
            snapshots[0][:, 1],
            s=14,
            color=cmap_t(0),
            edgecolors="white",
            linewidths=0.4,
        )
        for k in range(1, t_steps + 1):
            x0, xt, color = origins[k], snapshots[k], cmap_t(k / t_steps)
            for j in range(n):
                ax.plot(
                    [x0[j, 0], xt[j, 0]],
                    [x0[j, 1], xt[j, 1]],
                    color=color,
                    lw=0.6,
                    alpha=0.4,
                )
            ax.scatter(
                x0[:, 0],
                x0[:, 1],
                s=12,
                facecolors="none",
                edgecolors=color,
                linewidths=0.6,
                alpha=0.7,
            )
            ax.scatter(
                xt[:, 0],
                xt[:, 1],
                s=14,
                color=color,
                edgecolors="white",
                linewidths=0.4,
            )

    grid_n = 200
    xs = np.linspace(-5.0, 5.0, grid_n)
    Xg, Yg = np.meshgrid(xs, xs)
    n_cols = len(potentials)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(2.4 * n_cols, 4.8),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    cmap_t = plt.get_cmap("viridis")
    for j, (name, V_fn) in enumerate(potentials):
        integrate = make_integrator(V_fn)
        draw_paired(axes[0, j], V_fn, integrate, Xg, Yg, cmap_t)
        draw_unpaired(axes[1, j], V_fn, integrate, Xg, Yg, cmap_t)
        axes[0, j].set_title(name, fontsize=10)
        for r in range(2):
            axes[r, j].set_aspect("equal")
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("paired", fontsize=11)
    axes[1, 0].set_ylabel("unpaired", fontsize=11)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=cmap_t(k / t_steps),
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6,
            label=f"$t_{{{k}}}$",
        )
        for k in range(t_steps + 1)
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor="0.3",
            markeredgewidth=0.8,
            markersize=6,
            label=r"unobserved $x_0$",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="0.3",
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=6,
            label=r"observed $x_k$",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=False,
    )

    out_path = ctx.figure_path("synthetics_paired_vs_unpaired")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  saved {out_path}")


SPEC = ExperimentSpec(
    name="synthetic",
    description="Synthetic 2D: 15 potentials × paired/unpaired V-recovery gallery (Table 4).",
    variants=tuple(
        Variant(
            _variant_name(pot, mode),
            PANEL_CONFIG,
            {"potential": pot, "synth_coupled": coupled},
        )
        for pot in POTENTIALS
        for mode, coupled in _MODES
    ),
    smoke_epochs=50,
    smoke_overrides={"num_particles": 100},
    smoke_variants=tuple(
        _variant_name(pot, mode) for pot in SMOKE_POTENTIALS for mode, _ in _MODES
    ),
    evaluate=_evaluate,
    plots={
        "panels": _plot_panels,
        "compact": _plot_compact,
        "coupling": _plot_coupling,
        "paired_vs_unpaired": _plot_paired_vs_unpaired,
    },
)

register(SPEC)
