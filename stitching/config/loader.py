"""Preset resolution and the :func:`load_config` loader.

The presets live in a single top-level ``configs/`` tree at the repo root —
``configs/{data,method,experiment}/*.json`` — *outside* the installed package
(the standard ML-repo layout: config is an experiment artifact, not library
code). ``data`` and ``method`` presets are reusable building blocks; the
``experiment`` presets pin a specific published run's hyperparameters and form
the third composite tier.

The tree is located by, in order:
  1. ``$STITCHING_CONFIG_DIR`` if set (test/CI isolation, or a non-editable
     install pointing at an unpacked ``configs/``), else
  2. ``<repo_root>/configs`` — the repo root resolved from this module's
     location (``stitching/config/loader.py`` → ``parents[2]``), which is the
     editable/clone install the CLI and the experiment runners always use.

This trades the former "bare ``pip install`` with no checkout resolves presets"
property (presets are no longer wheel-bundled) for the conventional layout; the
real workflow is clone + editable install, where ``<repo_root>/configs`` is
always present.
"""

import json
import os
from pathlib import Path
from typing import Any

from .schema import Config


def _config_root() -> Path:
    """Return the top-level ``configs/`` preset root.

    Resolution order: ``$STITCHING_CONFIG_DIR`` if set, else
    ``<repo_root>/configs`` (repo root = ``parents[2]`` of this module).

    Returns:
        The preset-tree root directory (not required to exist; the
        ``list_*`` helpers and resolvers handle a missing tree).
    """
    env = os.environ.get("STITCHING_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "configs"


CONFIG_DIR = _config_root()
DATA_CONFIG_DIR = CONFIG_DIR / "data"
METHOD_CONFIG_DIR = CONFIG_DIR / "method"
EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiment"


def _resolve_preset(name: str) -> Path:
    """Resolve a bare preset name to its JSON file.

    Search order:
      1. `configs/data/<name>.json`
      2. `configs/method/<name>.json`

    Raises `FileNotFoundError` with a clear message if neither exists.
    """
    for directory in (DATA_CONFIG_DIR, METHOD_CONFIG_DIR):
        candidate = directory / f"{name}.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unknown config preset {name!r}. "
        f"Looked in {DATA_CONFIG_DIR} and {METHOD_CONFIG_DIR}. "
        f"Use --list-data / --list-methods to see available presets."
    )


def load_config(config: str | Path | None = None) -> Config:
    """Load a run configuration.

    *config* can be:
    - `None` — return defaults.
    - A path to a JSON file (absolute or relative).
    - A bare preset name (e.g. `"doublewell"` or `"stitching"`), resolved
      against the built-in `data/` then `method/` preset directories.
    - A composite preset `"<data>:<method>"` (e.g. `"doublewell:stitching"`),
      which merges the data preset and then overlays the method preset.
    - A three-way composite `"<data>:<method>:<experiment>"` (e.g.
      `"wavy_valley:stitching:wavy_valley_fig1"`), which additionally overlays an
      experiment preset from ``configs/experiment/`` (last wins). The
      experiment preset pins a specific published run's hyperparameters.
    """
    if config is None:
        return Config()

    def _read(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config {path} must be a JSON object.")
        return loaded

    merged: dict[str, Any] = {}
    resolved_paths: list[str] = []
    data_stem: str | None = None  # data-preset stem if any, used as experiment name

    experiment_stem: str | None = None  # experiment-preset stem, if any
    if isinstance(config, str) and ":" in config and not Path(config).exists():
        parts = config.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                "Composite config must have the form 'data:method' or "
                f"'data:method:experiment', got {config!r}."
            )
        data_name, method_name = parts[0], parts[1]
        data_path = DATA_CONFIG_DIR / f"{data_name}.json"
        method_path = METHOD_CONFIG_DIR / f"{method_name}.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Data preset not found: {data_path}")
        if not method_path.exists():
            raise FileNotFoundError(f"Method preset not found: {method_path}")
        merged.update(_read(data_path))
        merged.update(_read(method_path))
        resolved_paths = [str(data_path), str(method_path)]
        data_stem = data_name
        if len(parts) == 3:
            experiment_stem = parts[2]
            exp_path = EXPERIMENT_CONFIG_DIR / f"{experiment_stem}.json"
            if not exp_path.exists():
                raise FileNotFoundError(f"Experiment preset not found: {exp_path}")
            merged.update(_read(exp_path))
            resolved_paths.append(str(exp_path))
    else:
        path = Path(config)
        if not path.suffix:
            path = _resolve_preset(str(config))
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        merged.update(_read(path))
        resolved_paths = [str(path)]
        # If we resolved a data preset (not a method preset), use its stem.
        if path.parent.name == "data":
            data_stem = path.stem

    cfg = Config.from_dict(merged)
    cfg.config_path = " + ".join(resolved_paths)
    # Stamp the explicit experiment name. An experiment preset is the most
    # specific identity, so it wins. Otherwise use the data-preset stem when it's
    # more specific than the bare potential/dataset (e.g. ``doublewell-terminal``
    # beats ``doublewell``); else leave it None and let the property fall back.
    if experiment_stem:
        cfg._experiment_name = experiment_stem
    else:
        key = cfg.potential or cfg.dataset
        if data_stem and key and (data_stem == key or data_stem.startswith(f"{key}-")):
            cfg._experiment_name = data_stem
    return cfg


def list_data_presets() -> list[str]:
    """Return sorted names of built-in data presets."""
    if not DATA_CONFIG_DIR.exists():
        return []
    return sorted(path.stem for path in DATA_CONFIG_DIR.glob("*.json"))


def list_method_presets() -> list[str]:
    """Return sorted names of built-in method presets."""
    if not METHOD_CONFIG_DIR.exists():
        return []
    return sorted(path.stem for path in METHOD_CONFIG_DIR.glob("*.json"))


def list_experiment_presets() -> list[str]:
    """Return sorted names of built-in experiment presets (the 3rd composite tier)."""
    if not EXPERIMENT_CONFIG_DIR.exists():
        return []
    return sorted(path.stem for path in EXPERIMENT_CONFIG_DIR.glob("*.json"))


def list_presets() -> list[str]:
    """Return sorted names of all built-in presets (data + method)."""
    return sorted({*list_data_presets(), *list_method_presets()})
