"""Experiment definitions — one module per experiment, each self-registering.

Importing this package imports every ``defs/<name>.py`` module, whose top-level
:func:`experiments.registry.register` call adds its
:class:`~experiments.registry.ExperimentSpec` to ``EXPERIMENT_REGISTRY``. To add
an experiment: drop a new ``defs/<name>.py`` and list it here.
"""

from __future__ import annotations

from . import (  # noqa: F401
    chiral,
    cis,
    doublewell,
    rna,
    synthetic,
    wavy_valley,
    wavy_valley_init,
    mccann_ablation,  # last: derives *_mccann specs from the headline specs above
)
