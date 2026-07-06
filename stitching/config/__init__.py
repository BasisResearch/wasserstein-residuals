"""Experiment configuration for JAX Wasserstein gradient flow models.

Public surface (unchanged from the former single-module ``config.py``):

- :class:`Config` — the typed run schema (flat dataclass).
- :func:`load_config` — resolve a preset name / path into a ``Config``.
- :func:`list_data_presets` / :func:`list_method_presets` /
  :func:`list_experiment_presets` / :func:`list_presets`.
- The ``*_CONFIG_DIR`` path constants (all under the top-level ``configs/``).

Internals are split across :mod:`~stitching.config.schema` (the dataclass) and
:mod:`~stitching.config.loader` (loader/resolvers; the JSON presets live in the
top-level ``configs/`` tree at the repo root, resolved relative to the package
or via ``$STITCHING_CONFIG_DIR``).
"""

from .loader import (
    CONFIG_DIR,
    DATA_CONFIG_DIR,
    EXPERIMENT_CONFIG_DIR,
    METHOD_CONFIG_DIR,
    list_data_presets,
    list_experiment_presets,
    list_method_presets,
    list_presets,
    load_config,
)
from .schema import Config

__all__ = [
    "Config",
    "load_config",
    "list_data_presets",
    "list_method_presets",
    "list_experiment_presets",
    "list_presets",
    "CONFIG_DIR",
    "DATA_CONFIG_DIR",
    "METHOD_CONFIG_DIR",
    "EXPERIMENT_CONFIG_DIR",
]
