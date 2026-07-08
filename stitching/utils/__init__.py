"""Utility modules: optimisation, metrics, plotting, persistence, CLI.

The convenience re-exports (:func:`parse_args`, :func:`wasserstein`,
:func:`train`) are resolved **lazily** via :pep:`562` ``__getattr__``: importing
``stitching.utils`` (or any light submodule such as ``paper_plots`` / ``plotting``,
which goes through this package ``__init__``) no longer eagerly pulls in the
jax/optax-heavy ``optim`` and ``metrics`` modules. The name still resolves on
first access (``from stitching.utils import train`` works unchanged), but a
numpy-only caller doesn't pay the jax import. ``set_random_seed`` stays eager —
it is defined here and depends only on the stdlib + numpy.
"""

from __future__ import annotations

import importlib
import random
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # for type checkers / IDEs; not imported at runtime
    from .cli import parse_args
    from .metrics import wasserstein
    from .optim import train


def set_random_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility.

    Args:
        seed: The seed shared by the ``random`` and ``numpy`` global RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)


# Re-exported name -> the submodule that defines it. Loaded on first access so a
# light caller (e.g. paper_plots/plotting) never imports the jax-heavy ones.
_LAZY_EXPORTS = {"parse_args": "cli", "wasserstein": "metrics", "train": "optim"}


def __getattr__(name: str) -> Any:
    """Lazily resolve a re-exported name from its defining submodule (:pep:`562`).

    Args:
        name: The attribute being accessed on the ``stitching.utils`` package.

    Returns:
        The resolved object from the owning submodule.

    Raises:
        AttributeError: if *name* is not a known lazy re-export.
    """
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    """List package attributes including the lazy re-exports (for tab-completion)."""
    return sorted([*globals(), *_LAZY_EXPORTS])


__all__ = ["parse_args", "set_random_seed", "train", "wasserstein"]
