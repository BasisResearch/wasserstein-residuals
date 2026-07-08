"""Top-level data dispatcher for experiments.

`load_data(cfg)` is the only public entry point. It routes on ``cfg.mode`` /
``cfg.dataset`` to one of two kinds of source:

* **synthetic-from-potential** (``cfg.mode == "synthetic"``) — live SDE
  simulation of a registered potential, in :mod:`stitching.synthetic`. Selected
  by ``--potential``; not a named dataset.
* **a registered dataset** (``cfg.dataset``) — looked up in
  :data:`DATASET_REGISTRY`, the single table mapping a dataset name to its
  loader. Most are *blackbox-on-disk* (chiral, cis, rna — precomputed
  ``.npz``/``.npy`` committed under the top-level ``data/`` directory, not
  shipped in the wheel); ``mckean-vlasov`` is a *preset-driven live simulation*
  (``blackbox=False``).

Registering a dataset is a single :data:`DATASET_REGISTRY` entry — its loader,
a one-line description, and the blackbox flag; :func:`load_data` and the CLI's
``--list-datasets`` (via :func:`list_datasets`) both read from there. CIS and
chiral have small flat loaders; RNA carries two split modes (particle and
timepoint).

The datasets live at the top level rather than inside the package. The root is
``$STITCHING_DATA_DIR`` if set, else the first ``data/`` found walking up from
this file (the repo root in a checkout); ``cfg.data_root`` overrides per call.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from stitching._kde import SpatioTemporalData
from stitching.synthetic import (
    load_mckean_vlasov_data,
    load_synthetic_from_cfg,
)

if TYPE_CHECKING:
    from .config import Config


def _default_data_root() -> Path:
    """Resolve the top-level ``data/`` directory holding blackbox datasets.

    ``$STITCHING_DATA_DIR`` wins if set; otherwise the repo-root ``data/`` (this
    file is ``<root>/stitching/data.py``). Datasets are not shipped in the wheel,
    so an installed user sets the env var or ``cfg.data_root``.
    """
    env = os.environ.get("STITCHING_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "data"


_DATA_ROOT = _default_data_root()
_CHIRAL_DIR = _DATA_ROOT / "chiral"
_CIS_DIR = _DATA_ROOT / "cis"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything :func:`load_data` needs to know about one dataset.

    Co-locates a dataset's loader with its human description and its kind, so
    registering a dataset is a single :data:`DATASET_REGISTRY` entry and the
    dispatch + the CLI listing both read from here.

    Attributes:
        loader: ``(cfg) -> (train_data, test_data)``.
        description: One-line summary shown by ``--list-datasets``.
        blackbox: True for precomputed on-disk benchmarks; False for a
            preset-driven live simulation (``mckean-vlasov``).
    """

    loader: Callable[[Config], tuple[SpatioTemporalData, SpatioTemporalData]]
    description: str
    blackbox: bool = True


# ---------------------------------------------------------------------------
# Dataset loaders + split helpers
# ---------------------------------------------------------------------------


def _load_cis(cfg: Config) -> tuple[SpatioTemporalData, SpatioTemporalData]:
    """Load the CIS simulation benchmark, split along particle index."""
    data_dir = Path(cfg.data_root) if cfg.data_root is not None else _CIS_DIR
    npz = np.load(data_dir / "cis-simulation.npz")
    return _split_by_particle(
        traj=npz["trajectories"],
        times=npz["times"].astype(np.float32),
        train_fraction=cfg.train_fraction,
        scale_time=cfg.scale_time,
        seed=cfg.split_seed,
    )


def _load_chiral(cfg: Config) -> tuple[SpatioTemporalData, SpatioTemporalData]:
    """Load the chiral benchmark (pre-split by its generator; no re-splitting)."""
    # Pre-split by the generator (data/chiral/generate.py).
    data_dir = Path(cfg.data_root) if cfg.data_root is not None else _CHIRAL_DIR
    npz = np.load(data_dir / "chiral-simulation.npz")
    return (
        SpatioTemporalData(
            x=jnp.asarray(npz["train_x"]), t=jnp.asarray(npz["train_t"])
        ),
        SpatioTemporalData(x=jnp.asarray(npz["test_x"]), t=jnp.asarray(npz["test_t"])),
    )


def _load_mckean_vlasov(
    cfg: Config,
) -> tuple[SpatioTemporalData, SpatioTemporalData]:
    """Load McKean–Vlasov data via live SDE simulation from a preset."""
    train, test, _meta = load_mckean_vlasov_data(cfg)
    return train, test


def _split_by_particle(
    traj: np.ndarray,  # (T, N, D)
    times: np.ndarray,  # (T,)
    train_fraction: float,
    scale_time: bool,
    seed: int,
) -> tuple[SpatioTemporalData, SpatioTemporalData]:
    """Flatten a `(T, N, D)` trajectory and split along particle index."""
    if scale_time:
        t_min, t_max = times.min(), times.max()
        if t_max > t_min:
            times = (times - t_min) / (t_max - t_min)

    T, N, D = traj.shape
    x_all = traj.reshape(-1, D).astype(np.float32)
    t_all = np.repeat(times, N)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train = max(1, int(N * train_fraction))
    train_idx = perm[:n_train]
    train_mask = np.isin(np.tile(np.arange(N), T), train_idx)

    return (
        SpatioTemporalData(
            x=jnp.array(x_all[train_mask]), t=jnp.array(t_all[train_mask])
        ),
        SpatioTemporalData(
            x=jnp.array(x_all[~train_mask]), t=jnp.array(t_all[~train_mask])
        ),
    )


def _infer_rna_root() -> Path:
    """Walk up from CWD and file location to find the RNA data directory."""
    for anchor in [Path(__file__).resolve().parent, Path.cwd().resolve()]:
        for candidate in [anchor, *anchor.parents]:
            for sub in ("RNA_PCA_5", "RNA_PCA_100"):
                if (candidate / "data" / sub / "data.npy").exists():
                    return candidate
    raise FileNotFoundError(
        "Could not locate data/RNA_PCA_{5,100}/{data.npy,sample_labels.npy}."
    )


def _load_rna(cfg: Config) -> tuple[SpatioTemporalData, SpatioTemporalData]:
    """Load the RNA PCA benchmark.

    Two split modes:
    - *Timepoint split* (``cfg.train_timepoint_idx`` and
      ``cfg.eval_timepoint_idx`` both set): train/test are entire
      timepoints, enabling temporal-interpolation evaluation.
    - *Particle split* (default): every timepoint is shared, particles
      are split. ``cfg.split_path`` overrides; otherwise a deterministic
      split file ``split_{tr}_{te}_seed{N}.npz`` is reused if present.
    """
    root_path = (
        Path(cfg.data_root).resolve()
        if cfg.data_root is not None
        else _infer_rna_root()
    )
    folder = "RNA_PCA_5" if cfg.dims <= 5 else "RNA_PCA_100"
    data_dir = root_path / "data" / folder

    x = np.load(data_dir / "data.npy").astype(np.float32)[:, : cfg.dims]
    labels = np.load(data_dir / "sample_labels.npy").astype(np.float32)

    if cfg.scale_time:
        t_min, t_max = labels.min(), labels.max()
        if t_max > t_min:
            labels = (labels - t_min) / (t_max - t_min)

    # --- Timepoint-based split (interpolation protocol) ----------------
    if cfg.train_timepoint_idx is not None and cfg.eval_timepoint_idx is not None:
        sorted_times = np.sort(np.unique(labels))
        train_times = {float(sorted_times[i]) for i in cfg.train_timepoint_idx}
        eval_times = {float(sorted_times[i]) for i in cfg.eval_timepoint_idx}
        train_mask = np.array([float(l) in train_times for l in labels])
        eval_mask = np.array([float(l) in eval_times for l in labels])
        return (
            SpatioTemporalData(
                x=jnp.array(x[train_mask]), t=jnp.array(labels[train_mask])
            ),
            SpatioTemporalData(
                x=jnp.array(x[eval_mask]), t=jnp.array(labels[eval_mask])
            ),
        )

    # --- Particle-based split ------------------------------------------
    if cfg.split_path is not None:
        sp = Path(cfg.split_path)
        if not sp.is_absolute():
            sp = root_path / sp
        payload = np.load(sp)
        train_idx, test_idx = payload["train_idx"], payload["test_idx"]
    else:
        default_split = (
            data_dir
            / f"split_{int(cfg.train_fraction * 100)}_{int((1 - cfg.train_fraction) * 100)}_seed{cfg.split_seed}.npz"
        )
        if default_split.exists():
            payload = np.load(default_split)
            train_idx, test_idx = payload["train_idx"], payload["test_idx"]
        else:
            rng = np.random.default_rng(cfg.split_seed)
            perm = rng.permutation(len(x))
            n_train = int(cfg.train_fraction * len(x))
            train_idx, test_idx = perm[:n_train], perm[n_train:]

    return (
        SpatioTemporalData(x=jnp.array(x[train_idx]), t=jnp.array(labels[train_idx])),
        SpatioTemporalData(x=jnp.array(x[test_idx]), t=jnp.array(labels[test_idx])),
    )


# ---------------------------------------------------------------------------
# Dataset registry + dispatch
# ---------------------------------------------------------------------------

#: The single table mapping a dataset name to its loader. Add a dataset by
#: adding one entry here (and a data preset); :func:`load_data` and
#: :func:`list_datasets` (the CLI's ``--list-datasets``) both read from it. The
#: ``--potential`` synthetic path is *not* a dataset and is dispatched ahead of
#: this lookup in :func:`load_data`.
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "chiral": DatasetSpec(
        _load_chiral, "Chiral 2-D oscillator (generator-split train/test)."
    ),
    "cis": DatasetSpec(_load_cis, "CIS simulation trajectories (particle split)."),
    "rna": DatasetSpec(
        _load_rna, "RNA PCA developmental benchmark (particle or timepoint split)."
    ),
    "mckean-vlasov": DatasetSpec(
        _load_mckean_vlasov,
        "McKean–Vlasov live SDE from a preset.",
        blackbox=False,
    ),
}


def load_data(cfg: Config) -> tuple[SpatioTemporalData, SpatioTemporalData]:
    """Load ``(train_data, test_data)`` for the configured experiment.

    Synthetic-from-potential (``cfg.mode == "synthetic"``, selected by
    ``--potential``) routes to the live SDE simulation; every named dataset is
    resolved through :data:`DATASET_REGISTRY`.

    Raises:
        ValueError: If ``cfg.dataset`` is not a registered dataset name.
    """
    if cfg.mode == "synthetic":
        return load_synthetic_from_cfg(cfg)

    spec = DATASET_REGISTRY.get(cfg.dataset)
    if spec is None:
        raise ValueError(
            f"Unknown dataset: {cfg.dataset!r}. "
            f"Known datasets (use with --dataset): {list_datasets()}. "
            "For synthetic data use --potential <name>."
        )
    return spec.loader(cfg)


def list_datasets() -> list[str]:
    """Return the sorted names of all registered datasets (use with --dataset)."""
    return sorted(DATASET_REGISTRY)
