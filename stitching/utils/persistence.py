"""Persist / reload a single training run.

A run directory contains everything needed to inspect or re-evaluate a
trained model after the fact:

  <run_dir>/
    model.eqx     equinox-serialised parameter leaves
    cfg.json      Config used to build the model (drives reload)
    metrics.json  EMD, NLL, R²(V), RMSE, pattern R², wall, ...
    data.npz      train_x, train_t, test_x, test_t
    manifest.json git commit + library versions + data hash + fit-id (lineage)

Reload by reading `cfg.json`, rebuilding the same model from `cfg`, then
calling ``eqx.tree_deserialise_leaves`` on it.

This module owns the **run-IO** surface (``save_run`` / ``load_run`` /
``load_runs`` / ``Run``) plus the reload guards that compare a checkpoint
against the live git/data state. The **provenance / fit-id machinery** it
records and checks against lives in :mod:`stitching.utils._provenance`; its
public names (:func:`current_commit`, :func:`data_fingerprint`,
:func:`compute_fit_id`, :func:`fit_id_for`) are re-exported here so callers
keep importing them from ``persistence``.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import numpy as np

# `current_commit` and `data_fingerprint` are imported here both to re-export
# the provenance API *and* because the reload guards below
# (`_warn_stale_provenance`, `_check_data_drift`) resolve them from this module's
# namespace — they are load-bearing, not droppable "unused re-exports".
from stitching.utils._provenance import (
    capture_provenance,
    compute_fit_id,
    current_commit,
    data_fingerprint,
    fit_id_for,
)

# Re-exported provenance API (kept importable from ``persistence`` for callers
# and tests); the implementations live in ``_provenance``.
__all__ = [
    "Run",
    "compute_fit_id",
    "current_commit",
    "data_fingerprint",
    "fit_id_for",
    "load_run",
    "load_runs",
    "save_run",
]

# Artifacts every saved run must contain to be reloadable, in save order
# (model.eqx is written first by save_run, so an interrupted save can leave it
# alone — the guard must check the later-written ones too).
_REQUIRED_RUN_FILES = ("model.eqx", "cfg.json", "data.npz")


class Run(NamedTuple):
    """A reloaded run: the reconstructed model plus everything around it.

    Returned by :func:`load_run` / :func:`load_runs`. A ``NamedTuple`` so call
    sites can use attribute access (``run.model``, ``run.manifest``) — replacing
    the opaque ``loaded[role][1]`` / ``[3]`` indexing in the figure runners.

    ``manifest`` is the saved :file:`manifest.json` (provenance: git commit,
    library versions, data hash, fit-id), or ``{}`` for a checkpoint saved before
    provenance capture. It lets a caller cross-check a figure's checkpoint against
    a CSV's stamped fit-id before publishing.
    """

    cfg: Any
    model: Any
    train_data: Any
    test_data: Any
    metrics: dict[str, Any]
    manifest: dict[str, Any]


def _cfg_to_jsonable(cfg) -> dict[str, Any]:
    """Config dataclass -> JSON-friendly dict."""
    d = dataclasses.asdict(cfg)
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif v is None or isinstance(v, (str, int, float, bool, list, dict)):
            out[k] = v
        else:
            out[k] = repr(v)
    return out


def _metrics_to_jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Coerce numpy/jax scalars and dict-valued sub-metrics to JSON-friendly."""

    def _coerce(v):
        if isinstance(v, dict):
            return {str(k): _coerce(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_coerce(x) for x in v]
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:  # noqa: BLE001
                return float(v)
        return v

    return {k: _coerce(v) for k, v in metrics.items()}


def save_run(
    run_dir: str | Path,
    cfg,
    model,
    train_data,
    test_data,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Persist a complete run snapshot to *run_dir*.

    Returns the (created) directory.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    eqx.tree_serialise_leaves(str(out / "model.eqx"), model)

    with (out / "cfg.json").open("w", encoding="utf-8") as f:
        json.dump(_cfg_to_jsonable(cfg), f, indent=2)

    train_x = np.asarray(train_data.x)
    train_t = np.asarray(train_data.t)
    test_x = np.asarray(test_data.x)
    test_t = np.asarray(test_data.t)
    np.savez(
        out / "data.npz",
        train_x=train_x,
        train_t=train_t,
        test_x=test_x,
        test_t=test_t,
    )

    if metrics:
        with (out / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(_metrics_to_jsonable(metrics), f, indent=2)

    # data_fingerprint hashes the same four arrays in the same order as the npz
    # write above (np.asarray on identical data → identical bytes → same digest).
    data_hash = data_fingerprint(train_data, test_data)
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(capture_provenance(cfg, data_hash), f, indent=2)

    return out


def _read_manifest(run: Path, *, strict: bool = False) -> dict[str, Any]:
    """Load ``manifest.json`` for *run*.

    An *absent* manifest returns ``{}`` silently — that is the legitimate
    pre-provenance / installed-wheel case. A manifest that *exists but won't
    parse* is different: it is a damaged provenance record, so under *strict*
    (the figure-publishing path) it raises rather than degrading to ``{}`` and
    letting a drift check silently no-op. Non-strict callers get a warning and
    ``{}`` (provenance stays advisory for ad-hoc analysis).
    """
    manifest_path = run / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = (
            f"run at {run}: manifest.json exists but could not be read "
            f"({type(exc).__name__}); its provenance cannot be verified."
        )
        if strict:
            raise ValueError(msg) from exc
        warnings.warn(msg, stacklevel=3)
        return {}


def _warn_stale_provenance(run: Path, manifest: dict[str, Any]) -> None:
    """Warn if a run's checkpoint may not match the current code.

    Re-rendering a figure (``--plot-only``) from a checkpoint trained at a
    different commit — or from uncommitted code — can silently publish a stale
    artifact. Comparing the saved manifest against the live git state turns that
    into a visible warning. Never raises: a commit drift is the *expected* state
    when replotting after a (e.g. plotting-only) code change, so it stays
    advisory even under ``strict`` — only a *data* drift escalates.
    """
    saved = manifest.get("git") or {}
    saved_commit, saved_dirty = saved.get("commit"), saved.get("dirty")
    cur_commit = current_commit()
    if saved_commit and cur_commit and saved_commit != cur_commit:
        warnings.warn(
            f"run at {run} was saved at commit {saved_commit[:10]} but HEAD is "
            f"{cur_commit[:10]}; reloaded model/figures may not match the "
            "current code.",
            stacklevel=3,
        )
    elif saved_dirty:
        warnings.warn(
            f"run at {run} was saved from a dirty working tree; its provenance "
            "is uncommitted and may not be reproducible.",
            stacklevel=3,
        )


def _check_data_drift(
    run: Path,
    manifest: dict[str, Any],
    train_data,
    test_data,
    *,
    strict: bool,
) -> None:
    """Verify the reloaded data matches the hash recorded at save time.

    A mismatch means ``data.npz`` was corrupted or swapped after the run was
    saved — publishing a figure from it would be silently wrong. Unlike a commit
    drift this is never expected, so it warns by default and *raises* under
    *strict* (the figure-publishing path). A manifest without ``data_hash`` (a
    pre-provenance checkpoint) is skipped.
    """
    saved_hash = manifest.get("data_hash")
    if not saved_hash:
        return
    actual = data_fingerprint(train_data, test_data)
    if actual == saved_hash:
        return
    msg = (
        f"run at {run}: data.npz hash {actual[:10]} does not match the manifest's "
        f"{saved_hash[:10]} — the data was altered after the run was saved; any "
        "reloaded model or figure may be inconsistent with it."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=3)


def load_run(run_dir: str | Path, *, strict: bool = False) -> Run:
    """Reload a saved run.

    Returns a :class:`Run` ``(cfg, model, train_data, test_data, metrics,
    manifest)``. The model is rebuilt from *cfg* and weights are restored from
    ``model.eqx``. Emits a warning when the saved provenance does not match the
    current git state (so ``--plot-only`` cannot silently re-render a stale
    figure) and when ``data.npz`` no longer matches its saved hash.

    Args:
        run_dir: The run directory written by :func:`save_run`.
        strict: If True, a *data* drift (``data.npz`` altered after save) raises
            instead of warning — for the figure-publishing path, where rendering
            from inconsistent data is worse than for ad-hoc analysis. A *commit*
            drift always stays a warning (it is the expected state when
            replotting after a code change).

    Returns:
        The reloaded :class:`Run`.
    """
    import jax
    import jax.numpy as jnp

    from stitching._kde import SpatioTemporalData
    from stitching.build import build_model
    from stitching.config import Config
    from stitching.utils import set_random_seed

    run = Path(run_dir)
    manifest = _read_manifest(run, strict=strict)
    _warn_stale_provenance(run, manifest)
    with (run / "cfg.json").open("r", encoding="utf-8") as f:
        cfg = Config.from_dict(json.load(f))

    # Reconstruct as jax arrays so a reloaded run is indistinguishable from a fresh
    # ``load_data`` (which returns jax arrays). Reloaded numpy arrays break model
    # ``sample`` paths that index KDE centers under ``vmap`` (e.g. Lightspeed), and
    # the decoupled eval/plot stages reload before sampling. ``jnp.asarray`` of the
    # saved float32 npz preserves the exact bits, so data-hash / fit-id are unchanged.
    npz = np.load(run / "data.npz")
    train_data = SpatioTemporalData(
        x=jnp.asarray(npz["train_x"]), t=jnp.asarray(npz["train_t"])
    )
    test_data = SpatioTemporalData(
        x=jnp.asarray(npz["test_x"]), t=jnp.asarray(npz["test_t"])
    )
    _check_data_drift(run, manifest, train_data, test_data, strict=strict)

    set_random_seed(cfg.seed)
    k_model, _ = jax.random.split(jax.random.key(cfg.seed))
    model_skeleton = build_model(cfg, train_data, k_model)
    model = eqx.tree_deserialise_leaves(str(run / "model.eqx"), model_skeleton)

    metrics: dict[str, Any] = {}
    metrics_path = run / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    return Run(cfg, model, train_data, test_data, metrics, manifest)


def load_runs(
    run_dirs: Mapping[str, str | Path], *, strict: bool = True
) -> dict[str, Run]:
    """Reload several saved runs keyed by role, for ``--plot-only`` figures.

    Multi-model figures (e.g. wavy-valley's Stitching+Lightspeed panel) need a
    role → run-dir mapping rather than a single directory. Each value is the
    :class:`Run` ``(cfg, model, train_data, test_data, metrics, manifest)`` from
    :func:`load_run`.

    This is the figure-publishing entry point (its only callers are the runners'
    ``--plot-only`` branches), so it defaults to ``strict=True``: a checkpoint
    whose ``data.npz`` was altered after saving raises rather than silently
    publishing a figure from inconsistent data. (A *commit* drift still only
    warns — replotting after a code change is the intended workflow.)

    Args:
        run_dirs: Mapping of role name → run directory.
        strict: Forwarded to :func:`load_run`; ``True`` here (publish path).

    Returns:
        Mapping of role name → the :class:`Run` for that role.

    Raises:
        FileNotFoundError: naming the first role whose checkpoint is missing or
            incomplete, so a stale or partially-written ``results/`` tree fails
            loudly (with the role) instead of silently re-rendering against an
            incomplete set of runs or dying with an un-attributed error deep in
            :func:`load_run`.
        ValueError: under *strict*, if a reloaded run's data no longer matches
            its manifest hash (see :func:`load_run`).
    """
    out: dict[str, Run] = {}
    for role, run_dir in run_dirs.items():
        run = Path(run_dir)
        missing = [name for name in _REQUIRED_RUN_FILES if not (run / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"--plot-only: run for {role!r} at {run} is missing "
                f"{missing} (interrupted save?). "
                "Run the experiment without --plot-only first."
            )
        out[role] = load_run(run, strict=strict)
    return out
