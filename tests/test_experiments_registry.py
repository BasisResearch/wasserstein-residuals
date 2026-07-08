"""Contract tests for the experiment registry and the stage engine wiring.

Cheap, jax-free guards that every ``experiments/defs/<name>.py`` spec is
well-formed: its composite configs resolve, its sweep is non-empty, and its
eval/plot hooks are callable. Grows automatically as experiments are migrated —
a malformed or orphaned spec fails here without running a model.
"""

from __future__ import annotations

import pytest

# Importing the defs package registers every spec.
import experiments.defs  # noqa: F401
from experiments._engine import STAGES
from experiments.registry import EXPERIMENT_REGISTRY, ExperimentSpec, list_experiments
from stitching.config import load_config

_SPECS = sorted(EXPERIMENT_REGISTRY.values(), key=lambda s: s.name)
_IDS = [s.name for s in _SPECS]


def test_registry_is_populated() -> None:
    assert EXPERIMENT_REGISTRY, (
        "no experiments registered — defs/__init__ wiring broken?"
    )
    assert list_experiments() == sorted(EXPERIMENT_REGISTRY)


def test_engine_exposes_the_documented_stages() -> None:
    assert set(STAGES) == {"data", "train", "eval", "plot", "all"}


@pytest.mark.parametrize("spec", _SPECS, ids=_IDS)
def test_spec_is_wellformed(spec: ExperimentSpec) -> None:
    assert spec.variants, f"{spec.name} declares no variants"
    assert spec.seeds, f"{spec.name} declares no seeds"
    names = [v.name for v in spec.variants]
    assert len(names) == len(set(names)), f"{spec.name} has duplicate variant names"
    if spec.smoke_variants is not None:
        assert set(spec.smoke_variants) <= set(names), (
            f"{spec.name} smoke_variants names not all in variants"
        )
    assert spec.evaluate is None or callable(spec.evaluate)
    for plot_name, fn in spec.plots.items():
        assert callable(fn), f"{spec.name} plot {plot_name!r} is not callable"


@pytest.mark.parametrize("spec", _SPECS, ids=_IDS)
def test_spec_configs_resolve(spec: ExperimentSpec) -> None:
    # Every variant's composite preset string must resolve+parse (catches a typo
    # or an orphaned experiment preset after the configs/ relocation), without
    # loading any dataset or model.
    for variant in spec.variants:
        cfg = load_config(variant.config)
        assert cfg.config_path is not None
