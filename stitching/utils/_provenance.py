"""Run provenance and fit-id machinery (internal).

Split out of :mod:`stitching.utils.persistence` so the run-IO surface
(``save_run`` / ``load_run`` / ``load_runs`` / ``Run``) is kept apart from the
content-signature machinery that binds a summary CSV to the checkpoint it was
computed from. The public names here — :func:`current_commit`,
:func:`data_fingerprint`, :func:`compute_fit_id`, :func:`fit_id_for` — are
re-exported from ``persistence`` for backwards compatibility; external callers
should keep importing them from :mod:`stitching.utils.persistence`.

A saved run's ``manifest.json`` carries a **fit-id** — a short content signature
derived from the code commit and a hash of ``data.npz`` (:func:`compute_fit_id`).
Because the id is content-derived (no per-invocation token), a genuine
reproduction yields the *same* id — only a different commit or byte-different
data diverges. Two uses:

  * **Data-drift guard (live).** :func:`stitching.utils.persistence.load_run`
    recomputes the reloaded data's hash and compares it to the manifest; a
    mismatch (``data.npz`` altered after save) warns, and *raises* on the
    figure-publishing path (``load_runs``, ``strict=True``).
  * **CSV ↔ checkpoint cross-check (recorded, not yet auto-enforced).** A runner
    stamps the same fit-id (:func:`fit_id_for`) into its summary CSV rows, so the
    table and the checkpoint trained on the same data + commit can be matched —
    today by eye, or by a future reload step.

The fit-id keys on commit + data only (not the config or model weights), so two
different fits on the same data at one commit share an id — it discriminates
*which data + code*, not *which run*.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

# Libraries whose versions materially affect a run's numerics; recorded per run.
_PROVENANCE_PACKAGES = ("jax", "jaxlib", "equinox", "diffrax", "optax", "numpy")


def _git_commit() -> dict[str, Any] | None:
    """Return ``{"commit": <sha>, "dirty": <bool>}`` or ``None`` if unavailable.

    Resolves the commit of the repository containing this source file. Returns
    ``None`` when git is absent or the source is not in a git work tree (e.g. an
    installed wheel), so provenance capture never breaks a run.
    """
    repo = Path(__file__).resolve().parent

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return {"commit": commit, "dirty": dirty}


def current_commit() -> str | None:
    """The HEAD commit sha of the source repo, or ``None`` if unavailable.

    Thin accessor over :func:`_git_commit` for callers (e.g. a runner stamping a
    CSV) that want to compute a fit-id once and reuse the commit across rows.
    """
    info = _git_commit()
    return info["commit"] if info else None


def _data_hash(*arrays: np.ndarray) -> str:
    """SHA-256 over the given arrays' dtype, shape and bytes, in order.

    A *byte-level* digest: deterministic for byte-identical data (the same
    ``data.npz`` reused), so two runs — or a CSV and a checkpoint — built from
    the same saved arrays share a hash without a coordinating token. It is not a
    value-level hash: ``-0.0`` vs ``0.0``, or floats regenerated on a different
    platform/library version, can differ in bytes and so in hash. Order- and
    dtype-sensitive by design.
    """
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def data_fingerprint(train_data, test_data) -> str:
    """Content hash of a run's ``(train_x, train_t, test_x, test_t)`` arrays.

    The canonical-order :func:`_data_hash` of the four arrays ``save_run`` writes
    to ``data.npz`` — the data half of a fit-id.
    """
    return _data_hash(
        np.asarray(train_data.x),
        np.asarray(train_data.t),
        np.asarray(test_data.x),
        np.asarray(test_data.t),
    )


def compute_fit_id(data_hash: str, commit: str | None) -> str:
    """Short content signature binding a CSV's rows to a saved checkpoint.

    Derived from the data hash and the code commit only (not the sweep config),
    so every variant/seed row trained on one dataset at one commit shares the
    fit-id of the checkpoint persisted from that same data + code. A stale
    checkpoint (different commit) or swapped data (different hash) yields a
    different id; a genuine reproduction yields the same id.

    Args:
        data_hash: A :func:`data_fingerprint` of the run's data.
        commit: The code commit sha (``None`` when git is unavailable).

    Returns:
        A 12-hex-character signature.
    """
    h = hashlib.sha256()
    h.update((commit or "nocommit").encode())
    h.update(data_hash.encode())
    return h.hexdigest()[:12]


def fit_id_for(train_data, test_data, *, commit: str | None = None) -> str:
    """The fit-id a checkpoint saved from this data (at HEAD) will carry.

    Lets a runner stamp the same id into its CSV rows as ``save_run`` writes
    into ``manifest.json``, binding the table to the checkpoint without a shared
    token. Pass *commit* to reuse a cached sha across many rows (each call
    otherwise shells out to git).

    Args:
        train_data: Training data (uses ``.x`` and ``.t``).
        test_data: Test data (uses ``.x`` and ``.t``).
        commit: Pre-resolved HEAD sha; defaults to :func:`current_commit`.

    Returns:
        The 12-char fit-id.
    """
    if commit is None:
        commit = current_commit()
    return compute_fit_id(data_fingerprint(train_data, test_data), commit)


def capture_provenance(cfg, data_hash: str | None = None) -> dict[str, Any]:
    """Capture run lineage: commit, config identity, versions, data hash, fit-id.

    Args:
        cfg: The run config (reads ``experiment_name`` / ``config_path`` if set).
        data_hash: A :func:`data_fingerprint`; when given, the manifest also
            records it plus the derived ``fit_id``.

    Returns:
        A JSON-friendly provenance dict for ``manifest.json``.
    """
    versions: dict[str, str] = {}
    for pkg in _PROVENANCE_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    git = _git_commit()
    prov: dict[str, Any] = {
        "git": git,
        "config_name": getattr(cfg, "experiment_name", None),
        "config_path": getattr(cfg, "config_path", None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if data_hash is not None:
        prov["data_hash"] = data_hash
        prov["fit_id"] = compute_fit_id(data_hash, git["commit"] if git else None)
    return prov
