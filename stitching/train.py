#!/usr/bin/env python
"""Train a JAX WGF model and report metrics.

Usage:
    python -m stitching.train -c rna:stitching
    python -m stitching.train --config rna:stitching --seed 42
    python -m stitching.train --potential flowers --model stitching
    python -m stitching.train -c synthetic:stitching --potential zigzag_ridge
    python -m stitching.train --help

All Config fields are available as CLI flags (hyphens preferred, underscores also accepted).
"""

from __future__ import annotations

import os
import sys

# Peek at --cuda before importing JAX so the platform is set correctly.
if "--cuda" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cuda"
else:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Suppress noisy XLA C++ warnings (e.g. "Empty bitcode string").
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402

from stitching.build import build_model, fit, make_loss_fn  # noqa: E402
from stitching.config import Config  # noqa: E402
from stitching.data import load_data  # noqa: E402
from stitching.evaluate import (  # noqa: E402
    run_blackbox_evaluation,
    run_synthetic_evaluation,
)
from stitching.synthetic import get_true_potential  # noqa: E402
from stitching.utils import set_random_seed  # noqa: E402
from stitching.utils.cli import parse_args  # noqa: E402
from stitching.utils.persistence import save_run  # noqa: E402
from stitching.utils.plotting import make_trajectory_callback, visualise  # noqa: E402


def print_config_summary(cfg: Config, train_data, model=None, loss=None) -> None:
    """Print a human-readable config/model summary to stdout.

    Shared header lines come from cfg; the model architecture + trainable
    coefficients come from ``repr(model)``; the loss recipe (scheme,
    weights, freeze knobs, cached pairs/bw, etc.) comes from ``repr(loss)``.
    """
    n_dims = train_data.x.shape[1]
    n_train = train_data.x.shape[0]
    print("=" * 60)
    if cfg.dataset is not None:
        print(f"Dataset:    {cfg.dataset}")
    if cfg.potential is not None:
        print(f"Potential:  {cfg.potential}")
    print(f"Dims:       {n_dims}")
    print(f"Train pts:  {n_train}")
    print(f"Epochs:     {cfg.epochs}")
    print(f"LR:         {cfg.lr}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Seed:       {cfg.seed}")
    if model is not None:
        print(repr(model))
    if loss is not None:
        print(repr(loss))

    if cfg.scale_time:
        print(f"Scale time: {cfg.scale_time}")
    if cfg.mode == "synthetic":
        print(f"Obs protocol:    {cfg.obs_protocol}")
        print(f"Synth particles: {cfg.synth_n_particles}")
        print(f"Synth timesteps: {cfg.synth_n_timesteps}")
    if cfg.mode == "blackbox" and cfg.train_timepoint_idx is not None:
        print(f"Train timepoints: {cfg.train_timepoint_idx}")
        print(f"Eval timepoints:  {cfg.eval_timepoint_idx}")
    if cfg.config_path:
        for i, p in enumerate(cfg.config_path.split(" + ")):
            label = "Config:    " if i == 0 else "            "
            print(f"{label} {p}")
    print("=" * 60)
    print()


def main() -> None:
    cfg = parse_args()[0]
    set_random_seed(cfg.seed)

    train_data, test_data = load_data(cfg)

    # Build a from-cfg model + loss purely for the pre-training summary. The
    # actual training goes through the shared `fit` recipe below, which rebuilds
    # them deterministically from the same cfg+seed (so the summary matches what
    # is trained); `fit` also trains the Lightspeed baseline full-batch, same as
    # the experiments — the CLI no longer mini-batches it as a special case.
    k_model, _ = jax.random.split(jax.random.key(cfg.seed))
    model = build_model(cfg, train_data, k_model)
    loss_fn = make_loss_fn(cfg, train_data)

    # ---- Print config + model + loss summary ----------------------------
    print_config_summary(cfg, train_data, model=model, loss=loss_fn.loss)

    # --- Trajectory evolution callback (1-D stitching/particle models) -------
    callback_every = 10
    traj_callback, traj_finalize = (None, None)
    if cfg.plot:
        traj_callback, traj_finalize = make_trajectory_callback(
            train_data,
            test_data,
            callback_every=callback_every,
        )

    t_train = time.time()
    history: list[dict] = []
    model = fit(
        cfg,
        train_data,
        epoch_callback=traj_callback,
        callback_every=callback_every,
        history=history,
    )
    wall_time_s = time.time() - t_train

    # --- Trained parameters summary ---------------------------------------
    print("\n--- Trained ---")
    print(repr(model))

    metrics: dict = {
        "wall_time_s": float(wall_time_s),
        "time_per_epoch_s": float(wall_time_s) / max(int(cfg.epochs), 1),
        "history": [{k: float(v) for k, v in h.items()} for h in history],
    }

    # --- Evaluation -------------------------------------------------------
    if cfg.mode == "synthetic":
        test_emd, emd_by_time, func_metrics, all_metrics, one_step_metrics = (
            run_synthetic_evaluation(model, cfg, train_data, test_data)
        )
        metrics.update(func_metrics)
    else:
        test_emd, emd_by_time, all_metrics, one_step_metrics = run_blackbox_evaluation(
            model, cfg, train_data, test_data
        )
    metrics.update(
        test_emd=float(test_emd),
        emd_by_time={float(t): float(v) for t, v in emd_by_time.items()},
        metrics={
            k: {
                "mean": float(v["mean"]),
                "by_time": {float(t): float(x) for t, x in v["by_time"].items()},
            }
            for k, v in all_metrics.items()
        },
        one_step_metrics={
            k: {
                "mean": float(v["mean"]),
                "by_time": {float(t): float(x) for t, x in v["by_time"].items()},
            }
            for k, v in one_step_metrics.items()
        },
    )

    # --- Persist run snapshot (model.eqx, cfg.json, metrics.json, data.npz) -
    run_dir = Path(cfg.output_dir) / f"{cfg.experiment_name}-{cfg.model}"
    save_run(run_dir, cfg, model, train_data, test_data, metrics=metrics)

    # Build true potential for synthetic datasets (used in functional.png)
    true_potential_fn = get_true_potential(cfg.potential)

    # Plot
    if cfg.plot:
        visualise(
            model,
            cfg,
            train_data,
            test_data,
            cfg.output_dir,
            true_potential_fn=true_potential_fn,
        )
        if traj_finalize is not None:
            traj_finalize(cfg.output_dir, f"{cfg.experiment_name}-{cfg.model}")
    print(f"\nResults saved to {run_dir}/")


if __name__ == "__main__":
    main()
