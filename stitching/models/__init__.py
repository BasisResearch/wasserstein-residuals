"""Trainable model containers + their composite training objectives.

Each submodule pairs a model class with its loss class:

- `Stitching` / `StitchingLoss` (`stitching.py`) — KDE marginals built
  from trainable trajectory particles (the paper's method).
- `Lightspeed` / `LightspeedLoss` (`lightspeed.py`) — JKOnet$^\\star$
  potential-only baseline.

Both are equinox modules with softplus-parameterised entropy and potential
coefficients, and expose a ``sample(t_eval, ...)`` method. Their Wasserstein
gradient is the closed form in :mod:`stitching.velocity` (no framework).
"""

from stitching.models.lightspeed import Lightspeed, LightspeedLoss
from stitching.models.stitching import Stitching, StitchingLoss

# Only the trainable model/loss containers are part of the package-root surface;
# the construction helpers (`lightspeed_pairs`, `ot_interpolate_trajectories`,
# `mccann_interpolate_trajectories`) stay importable from their defining submodules.
__all__ = [
    "Lightspeed",
    "LightspeedLoss",
    "Stitching",
    "StitchingLoss",
]
