"""Unit tests for the dataset registry and :func:`stitching.data.load_data`.

Phase-D made adding a dataset a single :data:`DATASET_REGISTRY` entry and routed
``load_data`` + the CLI's ``--list-datasets`` through it (replacing the inline
``cfg.dataset`` if-ladder and the drifted ``cli.BLACKBOX_DATASETS`` list). These
pin that contract: the registry is the single source of dataset names, the
dispatch goes through it, and an unknown name fails loudly.
"""

from __future__ import annotations

import pytest

import stitching.data as data
from stitching.config import Config
from stitching.data import DATASET_REGISTRY, DatasetSpec, list_datasets, load_data


def test_registry_covers_known_datasets_with_blackbox_flags() -> None:
    assert set(DATASET_REGISTRY) == {"chiral", "cis", "rna", "mckean-vlasov"}
    # The on-disk benchmarks are blackbox; mckean-vlasov is a live preset sim.
    assert all(DATASET_REGISTRY[n].blackbox for n in ("chiral", "cis", "rna"))
    assert DATASET_REGISTRY["mckean-vlasov"].blackbox is False
    # Every spec is fully populated (loader callable + a human description).
    for spec in DATASET_REGISTRY.values():
        assert callable(spec.loader)
        assert isinstance(spec.description, str) and spec.description


def test_list_datasets_is_sorted_registry_keys() -> None:
    assert list_datasets() == sorted(DATASET_REGISTRY)


def test_load_data_routes_named_dataset_through_registry(monkeypatch) -> None:
    # The dispatch must call the registered loader for cfg.dataset (and pass cfg).
    sentinel = object()
    seen: dict[str, object] = {}

    def fake_loader(cfg: Config) -> object:
        seen["cfg"] = cfg
        return sentinel

    monkeypatch.setitem(DATASET_REGISTRY, "fake", DatasetSpec(fake_loader, "fake ds"))
    cfg = Config(dataset="fake")  # potential is None -> mode == "blackbox"
    assert load_data(cfg) is sentinel
    assert seen["cfg"] is cfg


def test_load_data_routes_synthetic_to_simulation(monkeypatch) -> None:
    # cfg.mode == "synthetic" (a --potential run) must bypass the registry and
    # go to the live SDE simulation, regardless of any dataset entry.
    sentinel = object()
    monkeypatch.setattr(data, "load_synthetic_from_cfg", lambda cfg: sentinel)
    cfg = Config(potential="doublewell")
    assert load_data(cfg) is sentinel


def test_load_data_unknown_dataset_raises_naming_known() -> None:
    cfg = Config(dataset="not-a-dataset")
    with pytest.raises(ValueError, match="Unknown dataset: 'not-a-dataset'"):
        load_data(cfg)
