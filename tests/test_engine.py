"""Unit tests for the experiment stage engine's two load-bearing claims:

- **idempotency** — ``Context.is_fresh`` reuses a checkpoint only when all run
  files are present *and* the manifest fit-id matches the current ``(commit,
  data)``; a data change or a missing file makes it stale.
- **decoupling** — the ``plot`` stage reloads checkpoints and never retrains.

Both build a tiny spec/``Context`` directly (no global ``register``) so the
registry-contract test is unaffected.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments._engine import Context, _stage_plot
from experiments.registry import ExperimentSpec, Variant
from stitching._kde import SpatioTemporalData
from stitching.build import build_model
from stitching.config import Config
from stitching.utils.persistence import save_run


def _data() -> SpatioTemporalData:
    rng = np.random.default_rng(0)
    return SpatioTemporalData(
        x=jnp.asarray(rng.normal(size=(60, 2)), dtype=jnp.float32),
        t=jnp.asarray(np.repeat([0.0, 1.0, 2.0], 20), dtype=jnp.float32),
    )


def _cfg() -> Config:
    return Config(
        potential="doublewell",
        num_particles=20,
        num_steps=5,
        potential_hidden=(16,),
        seed=0,
    )


def _save_checkpoint(ctx: Context) -> tuple[Path, SpatioTemporalData]:
    cfg, data = _cfg(), _data()
    model = build_model(cfg, data, jax.random.key(cfg.seed))
    run_dir = ctx.run_dir("v", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dirty-tree provenance warning in tests
        save_run(run_dir, cfg, model, data, data, metrics={"wall_time_s": 0.0})
    return run_dir, data


def test_is_fresh_transitions(tmp_path: Path) -> None:
    """Fresh ⇔ all files present and fit-id matches; data drift / missing file ⇒ stale."""
    spec = ExperimentSpec(
        name="_t_fresh", description="x", variants=(Variant("v", "unused"),)
    )
    ctx = Context(spec, tmp_path, smoke=False)
    run_dir, data = _save_checkpoint(ctx)

    # Same (commit, data) as saved → reuse.
    assert ctx.is_fresh(run_dir, data, data) is True
    # Byte-different data → different fit-id → stale (would retrain).
    drifted = SpatioTemporalData(x=data.x + 1.0, t=data.t)
    assert ctx.is_fresh(run_dir, drifted, data) is False
    # A partially-deleted checkpoint is stale even if the manifest survives.
    (run_dir / "model.eqx").unlink()
    assert ctx.is_fresh(run_dir, data, data) is False


def test_plot_stage_reloads_without_retraining(tmp_path: Path, monkeypatch) -> None:
    """The ``plot`` stage reloads the checkpoint and never calls the trainer."""
    plotted: list = []

    def _plot(ctx: Context) -> None:
        plotted.append(ctx.load("v").model)

    spec = ExperimentSpec(
        name="_t_decouple",
        description="x",
        variants=(Variant("v", "unused"),),
        plots={"main": _plot},
    )
    ctx = Context(spec, tmp_path, smoke=False)
    _save_checkpoint(ctx)

    # Any fit during plot is a decoupling violation.
    import stitching.utils.runners as runners

    def _boom(*_a, **_k):
        raise AssertionError("plot stage retrained")

    monkeypatch.setattr(runners, "fit_seeded", _boom)

    _stage_plot(ctx, None)
    assert len(plotted) == 1  # reloaded model, no retrain
