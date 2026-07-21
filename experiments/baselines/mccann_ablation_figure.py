#!/usr/bin/env python
"""Render the OT-vs-McCann potential-recovery figure for the ablation appendix.

Reuses the synthetic gallery machinery (``_grid``, ``COMPACT_POTENTIALS``,
``POTENTIALS_2D``) and the ``_plot_compact`` contour idiom, but stacks the two
*initializations* as rows: ground-truth $V^\\star$ / learned $V^\\theta$ (OT init)
/ learned $V^\\theta$ (McCann init), one column per sensitive potential. Loads the
trained checkpoints from both result trees (``results/synthetic`` and
``results/synthetic_mccann``) via :func:`stitching.utils.persistence.load_run`.

Usage (repo root, after both trees are trained)::

    uv run python experiments/baselines/mccann_ablation_figure.py [paired|unpaired]

Writes ``docs/figures/potential_recovery_<mode>.pdf`` (referenced by
``docs/mccann_ablation.tex``, Fig.~\\ref{fig:mccann-recovery}).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # allow running as a plain script from anywhere

from experiments.defs.synthetic import COMPACT_POTENTIALS, _grid  # noqa: E402
from stitching.synthetic.potentials import POTENTIALS_2D  # noqa: E402
from stitching.utils.persistence import load_run  # noqa: E402

RESULTS = REPO / "results"


def _learned_field(run, X, Y) -> np.ndarray:
    c = float(run.model.potential_coeff)
    return _grid(lambda x: c * run.model.potential_net(x).reshape(()), X, Y)


def _pattern_r2(tree: str, pot: str, mode: str) -> float:
    path = RESULTS / tree / "metrics" / "synthetic_panel_summary.csv"
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["potential"] == pot and r["mode"] == mode:
                return float(r["pattern_r2"])
    return float("nan")


def main(mode: str = "paired") -> None:
    import matplotlib.pyplot as plt

    xs = np.linspace(-5.0, 5.0, 80)
    X, Y = np.meshgrid(xs, xs)
    cols = COMPACT_POTENTIALS

    fig, axes = plt.subplots(
        3, len(cols), figsize=(2.5 * len(cols), 7.5),
        layout="constrained", sharex=True, sharey=True,
    )
    for col, (pot, num) in enumerate(cols):
        ot = load_run(RESULTS / "synthetic" / "runs" / f"{pot}-{mode}" / "s0")
        mc = load_run(RESULTS / "synthetic_mccann" / "runs" / f"{pot}-{mode}" / "s0")
        true_c = _grid(POTENTIALS_2D[pot], X, Y)
        true_c = true_c - true_c.mean()
        ot_c = _learned_field(ot, X, Y)
        ot_c = ot_c - ot_c.mean()
        mc_c = _learned_field(mc, X, Y)
        mc_c = mc_c - mc_c.mean()

        vmax = float(np.max(np.abs(true_c)))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        kw = dict(levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, col].contourf(X, Y, true_c, **kw)
        axes[1, col].contourf(X, Y, ot_c, **kw)
        axes[2, col].contourf(X, Y, mc_c, **kw)

        r2_ot = _pattern_r2("synthetic", pot, mode)
        r2_mc = _pattern_r2("synthetic_mccann", pot, mode)
        axes[0, col].set_title(f"{num}. {pot}")
        axes[1, col].set_title(rf"$R^2_{{\mathrm{{pat}}}}$ {r2_ot:.2f}")
        axes[2, col].set_title(rf"$R^2_{{\mathrm{{pat}}}}$ {r2_mc:.2f}")
        for r in range(3):
            axes[r, col].set_aspect("equal")

    axes[0, 0].set_ylabel(r"ground truth $V^\star$", fontsize=11)
    axes[1, 0].set_ylabel("learned (OT init)", fontsize=11)
    axes[2, 0].set_ylabel("learned (McCann init)", fontsize=11)
    fig.suptitle(f"Potential recovery: OT vs. McCann initialization ({mode})", fontsize=13)

    out = REPO / "docs" / "figures" / f"potential_recovery_{mode}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "paired")
