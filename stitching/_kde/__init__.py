"""Self-contained, framework-free KDE primitives for Stitching.

This internal package holds the autodiff-free KDE primitives Stitching needs —
the distribution / kernel / curve data structures and the analytic KDE score.
There is no generic Wasserstein autodiff (no `Metric` / `Wasserstein.grad` /
`first_variation` / `divergence`); Stitching's Wasserstein gradient is closed
form (see :mod:`stitching.velocity`).
"""

from ._core import (
    Action,
    Curve,
    Distribution,
    Functional,
    Scalar,
    ScalarField,
    TangentField,
    Time,
    VectorField,
    integrate,
)
from .curves import InterpolationCurve, SpatioTemporalData
from .distributions import Empirical, KernelDensity, kde_score
from .kernel import (
    DensityKernel,
    IsotropicGaussian,
    Kernel,
    chunked_vmap,
    silverman_bandwidth,
)

__all__ = [
    # _core
    "Action",
    "Curve",
    "Distribution",
    "Functional",
    "Scalar",
    "ScalarField",
    "TangentField",
    "Time",
    "VectorField",
    "integrate",
    # curves
    "InterpolationCurve",
    "SpatioTemporalData",
    # distributions
    "Empirical",
    "KernelDensity",
    "kde_score",
    # kernel
    "DensityKernel",
    "IsotropicGaussian",
    "Kernel",
    "chunked_vmap",
    "silverman_bandwidth",
]
