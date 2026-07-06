"""Synthetic WGF benchmarks.

The package is organised around a single abstraction: a `SyntheticSystem`
bundles a `SumFunctional` (potential + optional entropy + optional
interaction) with a dimension and an initial-distribution sampler.  Any
system — whether registered under a short name or user-constructed via
`make_system` — is simulated through the same Euler–Maruyama integrator
(`simulate_wgf`) and turned into `SpatioTemporalData` via `load_synthetic`.

Registered systems:

* 15 two-dimensional potentials (``flowers``, ``styblinski_tang`` …)
* 1-D double-well Langevin (``doublewell``)
"""

from .dataset import (
    apply_observation_protocol,
    get_true_potential,
    get_true_potential_grad,
    load_mckean_vlasov_data,
    load_synthetic,
    load_synthetic_from_cfg,
    simulate_uncoupled,
    trajectory_to_data,
)
from .potentials import POTENTIALS_1D, POTENTIALS_2D
from .simulate import simulate_wgf
from .systems import (
    SyntheticSystem,
    get_system,
    isotropic_normal,
    list_systems,
    make_system,
    uniform_box,
)

__all__ = [
    "POTENTIALS_1D",
    "POTENTIALS_2D",
    "SyntheticSystem",
    "apply_observation_protocol",
    "get_system",
    "get_true_potential",
    "get_true_potential_grad",
    "isotropic_normal",
    "list_systems",
    "load_mckean_vlasov_data",
    "load_synthetic",
    "load_synthetic_from_cfg",
    "make_system",
    "simulate_uncoupled",
    "simulate_wgf",
    "trajectory_to_data",
    "uniform_box",
]
