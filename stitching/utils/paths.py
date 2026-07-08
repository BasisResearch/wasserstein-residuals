"""Resolve where experiment run-artifacts are written.

All runners write under a single top-level ``results/`` root (saved runs,
figures, caches, CSVs) rather than into the source tree. The root is:

  1. ``$STITCHING_RESULTS_DIR`` if set (used to isolate test / CI runs), else
  2. ``<default_root>/results`` — the repo root the caller passes in.

Keeping this in one place means the ``results/`` convention is defined once and
the runners stay free of path boilerplate.
"""

from __future__ import annotations

import os
from pathlib import Path


def _results_root(default_root: Path) -> Path:
    """Return the top-level results root (``$STITCHING_RESULTS_DIR`` or default).

    Args:
        default_root: Repo root to fall back to; the results root is
            ``default_root / "results"`` when the env var is unset.

    Returns:
        The resolved results-root directory (not created).
    """
    env = os.environ.get("STITCHING_RESULTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return default_root / "results"


def results_dir(name: str, default_root: Path) -> Path:
    """Return (and create) ``<results_root>/<name>`` for one experiment.

    Args:
        name: Per-experiment subdirectory (e.g. ``"wavy_valley"``).
        default_root: Repo root passed by the caller (``parents[1]`` of a
            ``experiments/run_*.py`` script).

    Returns:
        The created output directory.
    """
    out = _results_root(default_root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out
