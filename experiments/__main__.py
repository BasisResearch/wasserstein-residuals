"""``python -m experiments`` — the one entry point for every experiment.

Usage::

    python -m experiments list                      # what's available
    python -m experiments all   <name> [opts]       # data→train→eval→plot
    python -m experiments train <name> [opts]       # (re)fit checkpoints
    python -m experiments eval  <name> [opts]       # metrics CSV, no retrain
    python -m experiments plot  <name> [opts]       # figures, no retrain/re-eval
    python -m experiments data  <name> [opts]       # inspect/prefetch datasets

``run`` is an alias for ``all``. Options: ``--smoke`` (fast, isolated
``<name>_smoke`` dir), ``--force`` (retrain fresh checkpoints), ``--variant V``
and ``--seed N`` (repeatable; restrict the sweep — ``data``/``train``/``all``
only, since ``eval``/``plot`` collect over the whole experiment), ``--plot P``
(repeatable; render only named plots), ``--out-dir DIR`` (override the results dir).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Default to CPU before any stage imports jax (matches the old per-runner
# ``os.environ.setdefault`` preamble); honoured only if the caller hasn't set it.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Importing the defs package registers every ExperimentSpec.
import experiments.defs  # noqa: F401
from experiments._engine import run_experiment
from experiments.registry import EXPERIMENT_REGISTRY, list_experiments

_STAGE_HELP = {
    "all": "data→train→eval→plot (train materialises each cell's data)",
    "run": "alias for `all`",
    "data": "load each cell's dataset and report shapes (inspect/prefetch)",
    "train": "fit every (variant, seed) into a checkpoint; idempotent",
    "eval": "reload checkpoints, compute metrics, write CSV(s); no retrain",
    "plot": "reload checkpoints / metrics CSV and render figures; no retrain",
}


def _add_common(
    sp: argparse.ArgumentParser,
    *,
    with_force: bool,
    with_plot: bool,
    with_filters: bool,
) -> None:
    """Attach the shared options to a stage subparser."""
    sp.add_argument("name", help="a registered experiment (see `list`)")
    sp.add_argument(
        "--smoke",
        action="store_true",
        help="fast wall-time-only run, isolated to a <name>_smoke dir",
    )
    if with_force:
        sp.add_argument(
            "--force",
            action="store_true",
            help="retrain even when a fresh checkpoint exists",
        )
    if with_filters:
        # Only the cell-iterating stages (data/train, and the train sub-stage of
        # all) honour these; eval/plot are whole-experiment collection stages, so
        # they don't accept a filter they would silently ignore.
        sp.add_argument(
            "--variant",
            action="append",
            dest="variants",
            metavar="V",
            help="restrict to this variant (repeatable)",
        )
        sp.add_argument(
            "--seed",
            action="append",
            type=int,
            dest="seeds",
            metavar="N",
            help="restrict to this seed (repeatable)",
        )
    if with_plot:
        sp.add_argument(
            "--plot",
            action="append",
            dest="only_plots",
            metavar="P",
            help="render only this named plot (repeatable)",
        )
    sp.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="write artifacts here instead of results/<name>",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m experiments`` argument parser."""
    p = argparse.ArgumentParser(prog="python -m experiments", description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("list", help="list registered experiments")
    for stage in ("all", "run", "data", "train", "eval", "plot"):
        sp = sub.add_parser(stage, help=_STAGE_HELP[stage])
        _add_common(
            sp,
            with_force=stage in ("all", "run", "train"),
            with_plot=stage in ("all", "run", "plot"),
            with_filters=stage in ("all", "run", "data", "train"),
        )
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse args and dispatch to :func:`run_experiment` (or ``list``)."""
    args = build_parser().parse_args(argv)

    if args.stage == "list":
        print("Available experiments:")
        for name in list_experiments():
            print(f"  {name:<16}{EXPERIMENT_REGISTRY[name].description}")
        return

    stage = "all" if args.stage == "run" else args.stage
    run_experiment(
        stage,
        args.name,
        smoke=args.smoke,
        force=getattr(args, "force", False),
        variant_filter=getattr(args, "variants", None),
        seed_filter=getattr(args, "seeds", None),
        only_plots=getattr(args, "only_plots", None),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
