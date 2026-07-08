"""Regression gate: the framework-free pipeline reproduces its own reference
metrics deterministically.

The closed-form gradient is cross-checked against an independent autodiff oracle
in ``test_velocity_parity.py``; here we pin *end-to-end* determinism — a
fixed-seed ``python -m experiments all synthetic --smoke`` must reproduce the
committed ``tests/reference/synthetic_panel_summary.csv`` (regenerated from this
code) to bit precision, so any accidental behaviour change is caught.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_REFERENCE = (
    Path(__file__).resolve().parent / "reference" / "synthetic_panel_summary.csv"
)

# Deterministic metric columns (exclude wall_time_s).
_METRIC_COLS = ("test_emd", "potential_r2", "pattern_r2", "l2_uvp", "bw2_uvp")
_TOL = 1e-9  # effectively bitwise; tolerates only last-ULP cross-platform noise.


def _load(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open() as f:
        return {(r["potential"], r["mode"]): r for r in csv.DictReader(f)}


@pytest.mark.slow
def test_synthetic_smoke_regression(tmp_path: Path) -> None:
    """`experiments all synthetic --smoke` reproduces the committed reference metrics."""
    # Run the engine from the repo root (so `experiments` imports) with a minimal
    # env; STITCHING_RESULTS_DIR redirects the results/ root into tmp, so all
    # outputs (checkpoints, csv, png) are isolated and the real repo is untouched.
    subprocess.run(
        [sys.executable, "-m", "experiments", "all", "synthetic", "--smoke"],
        check=True,
        cwd=_REPO,
        env={
            "JAX_PLATFORMS": "cpu",
            "PATH": os.environ["PATH"],
            "STITCHING_RESULTS_DIR": str(tmp_path),
        },
    )

    # --smoke writes to the isolated synthetic_smoke/ dir, never the production
    # synthetic/ dir; eval CSVs land under its metrics/ subdir.
    got = _load(
        tmp_path / "synthetic_smoke" / "metrics" / "synthetic_panel_summary.csv"
    )
    ref = _load(_REFERENCE)
    assert set(got) == set(ref), f"cells differ: {set(got) ^ set(ref)}"

    mismatches = []
    for key in ref:
        for col in _METRIC_COLS:
            a, b = float(ref[key][col]), float(got[key][col])
            if not math.isclose(a, b, rel_tol=_TOL, abs_tol=_TOL):
                mismatches.append(f"{key} {col}: ref={a!r} got={b!r}")
    assert not mismatches, "non-deterministic / broken pipeline:\n" + "\n".join(
        mismatches
    )
