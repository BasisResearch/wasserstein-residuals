"""Shared CLI + IO scaffolding for the experiments — used by the stage engine
and its ``defs/`` specs.

Centralises the cross-experiment plumbing so each experiment carries only its
own logic: the wall-time-only ``--smoke`` config trim (:func:`apply_smoke`),
seed-then-build-and-train (:func:`fit_seeded`, which seeds numpy *and* JAX so a
fit is reproducible), snapshot-panel sampling (:func:`sample_model_panels`),
multi-seed aggregation (:func:`aggregate_seeds`), and metrics-CSV IO
(:func:`write_csv`).

This lives in the installed ``stitching`` package (not under ``experiments/``)
so it stays import-light and reusable: numpy only at module scope, with the two
model-touching helpers below importing jax / ``stitching.build`` lazily inside
their bodies.
"""

from __future__ import annotations

import csv
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from stitching._kde import SpatioTemporalData
    from stitching.config import Config
    from stitching.models import Lightspeed, Stitching

# Default smoke-run epoch budget (wall-time-only, not a scientific knob). Every
# runner's ``--smoke`` shrinks to this unless it pins its own override (wavy
# keeps 100). One named constant replaces the per-runner ``SMOKE_EPOCHS = 50``.
DEFAULT_SMOKE_EPOCHS = 50


def repo_root() -> Path:
    """Return the repository root (where ``results/`` lives).

    Resolved from this installed module's location
    (``stitching/utils/runners.py`` → ``parents[2]``), which is the repo root
    for the editable/dev install the experiments always run under. Replaces the
    per-runner ``HERE = Path(__file__)...`` / ``REPO_ROOT = HERE.parents[1]``
    boilerplate. Test/CI runs that must not touch the repo set
    ``$STITCHING_RESULTS_DIR`` (honoured by :func:`stitching.utils.paths.results_dir`),
    so this value is only the default results-root anchor.

    Returns:
        The absolute repository root path.
    """
    return Path(__file__).resolve().parents[2]


def apply_smoke(cfg: Any, epochs: int, **overrides: Any) -> Any:
    """Return *cfg* with wall-time-only smoke overrides applied.

    ``--smoke`` is never a scientific configuration: it only shrinks wall time
    (epochs plus any extra cheapening knobs such as ``num_particles`` or
    ``eval_reps``). Routing every runner's smoke trim through here documents that
    these are non-scientific overrides.

    Args:
        cfg: A :class:`~stitching.config.Config`.
        epochs: Smoke epoch budget.
        **overrides: Extra wall-time-only fields (e.g. ``num_particles=100``).

    Returns:
        A new ``Config`` (via :func:`dataclasses.replace`) with the overrides.
    """
    return dataclasses.replace(cfg, epochs=epochs, **overrides)


def fit_seeded(
    cfg: Config, train_data: SpatioTemporalData, **fit_kwargs: Any
) -> Stitching | Lightspeed:
    """Seed the global RNGs from ``cfg.seed``, then build-and-train ``cfg.model``.

    The one reproducible build-and-train entry point for the experiments,
    replacing the per-runner ``set_random_seed(cfg.seed); fit(cfg, train_data)``
    pair (the old ``_fit`` / ``fit_one`` / ``train_one`` / ``_train`` wrappers).
    :func:`stitching.build.fit` already derives its JAX keys from ``cfg.seed``;
    the extra :func:`stitching.utils.set_random_seed` keeps any NumPy/Python-RNG
    consumer reproducible too (defensive — the current build/train path uses only
    local, seed-derived generators, so this never changes a trained-model byte).

    Args:
        cfg: Run configuration; ``seed`` drives both the global RNG seed and
            ``fit``'s JAX keys.
        train_data: Training snapshots, passed straight to :func:`fit`.
        **fit_kwargs: Forwarded to :func:`fit` (e.g. ``epoch_callback``).

    Returns:
        The trained model instance (:class:`~stitching.models.Stitching` or
        :class:`~stitching.models.Lightspeed`).
    """
    from stitching.build import fit
    from stitching.utils import set_random_seed

    set_random_seed(cfg.seed)
    return fit(cfg, train_data, **fit_kwargs)


def sample_model_panels(
    model: Any,
    snap_t: Sequence[float],
    n: int,
    *,
    seed: int = 0,
) -> list[tuple[float, np.ndarray]]:
    """Sample *model* at each snapshot time, returned as ``(time, points)`` panels.

    The model-side companion to
    :func:`stitching.utils.paper_plots.snapshot_panels`: draws ``n`` samples at
    the pooled ``snap_t`` and unpacks the returned ``{rounded_t: array}`` mapping
    back into per-snapshot numpy panels, so cis / chiral stop repeating
    the ``model.sample(...)`` + ``round(float(t), 8)`` key idiom. jax is imported
    lazily so importing this module stays jax-free.

    Precondition: *model* must use the Stitching-style ``sample`` signature; it
    is **not** compatible with :class:`~stitching.models.Lightspeed`, whose
    ``sample(t_eval, train_data, ...)`` takes ``train_data`` positionally (a
    Lightspeed model raises ``TypeError`` here). All current callers
    (cis/chiral panels) sample the Stitching model.

    Args:
        model: A trained model exposing
            ``sample(t_eval, *, num_samples, key) -> {rounded_t: (N, D) array}``
            (the Stitching-style signature the panel plots use). The returned
            mapping must be keyed by ``round(float(t), 8)`` — a mismatch raises
            ``KeyError`` rather than silently dropping a panel.
        snap_t: Snapshot times to sample at.
        n: Samples to draw per snapshot.
        seed: JAX sampling-key seed (default 0, matching the runners).

    Returns:
        ``[(t, points), ...]`` aligned with *snap_t*.
    """
    import jax
    import jax.numpy as jnp

    samples = model.sample(jnp.asarray(snap_t), num_samples=n, key=jax.random.key(seed))
    return [(float(t), np.asarray(samples[round(float(t), 8)])) for t in snap_t]


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    """Write *rows* to *path* as CSV with the given header; return *path*.

    Uses ``extrasaction="ignore"`` so a row may carry extra keys beyond the
    header (the runners build wide row dicts and select columns via *fieldnames*).

    Args:
        path: Destination ``.csv`` path.
        rows: Row mappings (column name → value).
        fieldnames: Ordered header columns.

    Returns:
        The written path.
    """
    out = Path(path)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def aggregate_seeds(
    per_seed: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Element-wise mean and standard error across seeds.

    Args:
        per_seed: One equal-length metric tuple per seed (at least one).

    Returns:
        ``(mean, se)`` tuples. The standard error is the sample std (``ddof=1``)
        over ``sqrt(n)``; ``se`` is all-zero for a single seed.

    Raises:
        ValueError: if *per_seed* is empty or its rows are not all the same
            length (a clear domain error instead of a cryptic NumPy failure).
    """
    arr = np.asarray(per_seed, dtype=float)  # (n_seeds, n_metrics)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(
            "aggregate_seeds expects a non-empty sequence of equal-length "
            f"per-seed metric tuples; got shape {arr.shape!r}."
        )
    mean = arr.mean(axis=0)
    n = arr.shape[0]
    se = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(arr.shape[1])
    return tuple(mean.tolist()), tuple(se.tolist())
