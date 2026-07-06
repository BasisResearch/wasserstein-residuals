"""Unit tests for the shared runner scaffolding in ``stitching.utils.runners``.

The stage engine and its ``defs/`` specs route their seeding, snapshot-panel
sampling, seed aggregation and metrics-CSV IO through this module, so its
contracts are pinned here: the single-seed standard-error branch, the
seed-then-fit ordering, the snapshot-panel realignment, and the ``Run`` field
order that reloads depend on by attribute access.
"""

from __future__ import annotations

import csv
import math
import types
from pathlib import Path

import numpy as np
import pytest

from stitching.utils.persistence import Run
from stitching.utils.runners import (
    DEFAULT_SMOKE_EPOCHS,
    aggregate_seeds,
    fit_seeded,
    repo_root,
    sample_model_panels,
    write_csv,
)

# ---------------------------------------------------------------------------
# aggregate_seeds — single-seed branch + input validation
# ---------------------------------------------------------------------------


def test_aggregate_seeds_single_seed_has_zero_se() -> None:
    # The n>1 guard: a bare std(ddof=1) over one sample would be NaN and silently
    # corrupt single-seed leaderboard rows — this pins the zero-SE branch.
    mean, se = aggregate_seeds([(1.0, 2.0, 3.0)])
    assert mean == (1.0, 2.0, 3.0)
    assert se == (0.0, 0.0, 0.0)


def test_aggregate_seeds_two_seeds_matches_oracle() -> None:
    mean, se = aggregate_seeds([(0.0, 10.0), (4.0, 10.0)])
    assert mean == (2.0, 10.0)
    # ddof=1 sample std over {0,4} is 2.828...; se = std / sqrt(n).
    assert math.isclose(se[0], math.sqrt(8.0) / math.sqrt(2))
    assert se[1] == 0.0


def test_aggregate_seeds_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        aggregate_seeds([])


def test_aggregate_seeds_ragged_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_seeds([(1.0, 2.0), (3.0,)])


# ---------------------------------------------------------------------------
# write_csv + Run field contract
# ---------------------------------------------------------------------------


def test_write_csv_writes_header_and_drops_extra_keys(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "o.csv",
        [{"a": 1, "b": 2, "junk": 9}],
        fieldnames=("a", "b"),
    )
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"a": "1", "b": "2"}]  # 'junk' dropped; header == fieldnames.


# ---------------------------------------------------------------------------
# fit_seeded — seed-then-fit contract (the de-duplicated _fit/fit_one/_train)
# ---------------------------------------------------------------------------


def test_fit_seeded_seeds_before_fit_and_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The de-dup invariant: set_random_seed(cfg.seed) runs BEFORE fit, fit gets
    # (cfg, train_data) + any kwargs, and fit's return value is passed through.
    calls: list[tuple] = []
    monkeypatch.setattr(
        "stitching.utils.set_random_seed", lambda s: calls.append(("seed", s))
    )
    monkeypatch.setattr(
        "stitching.build.fit",
        lambda cfg, td, **kw: calls.append(("fit", cfg.seed, td, kw)) or "MODEL",
    )
    cfg = types.SimpleNamespace(seed=7)
    out = fit_seeded(cfg, "TRAIN", epoch_callback="cb")
    assert out == "MODEL"
    assert calls == [("seed", 7), ("fit", 7, "TRAIN", {"epoch_callback": "cb"})]


# ---------------------------------------------------------------------------
# sample_model_panels — model.sample → per-snapshot numpy panels
# ---------------------------------------------------------------------------


def test_sample_model_panels_unpacks_rounded_keys() -> None:
    # A fake model whose sample() returns the {round(float(t),8): array} mapping
    # the real models do; the helper must realign panels with snap_t order.
    class _FakeModel:
        def sample(self, t_eval, *, num_samples, key):  # noqa: ANN001, ANN202
            return {
                round(float(t), 8): np.full((num_samples, 2), float(t)) for t in t_eval
            }

    panels = sample_model_panels(_FakeModel(), [0.0, 1.5], 4)
    assert [t for t, _ in panels] == [0.0, 1.5]
    assert all(isinstance(t, float) for t, _ in panels)
    assert panels[0][1].shape == (4, 2)
    assert np.allclose(panels[1][1], 1.5)  # right snapshot landed in the right slot


def test_sample_model_panels_follows_snap_t_order_not_model_sort() -> None:
    # A model that sorts its sample times internally (as Stitching does) must not
    # reorder the returned panels — they realign to the caller's snap_t order.
    # The happy-path test above uses pre-sorted input, so it cannot catch a
    # regression to "model sort order"; this one feeds unsorted snap_t.
    class _Sorting:
        def sample(self, t_eval, *, num_samples, key):  # noqa: ANN001, ANN202
            import jax.numpy as jnp

            return {
                round(float(t), 8): np.full((num_samples, 1), float(t))
                for t in jnp.sort(t_eval)
            }

    panels = sample_model_panels(_Sorting(), [2.0, 0.0, 1.0], 1)
    assert [t for t, _ in panels] == [2.0, 0.0, 1.0]
    assert [float(a[0, 0]) for _, a in panels] == [2.0, 0.0, 1.0]


def test_sample_model_panels_raises_on_rounded_key_mismatch() -> None:
    # The helper looks up round(float(t), 8); a model keying to fewer digits
    # must fail LOUDLY (KeyError), never silently drop or misalign a panel.
    class _Mismatch:
        def sample(self, t_eval, *, num_samples, key):  # noqa: ANN001, ANN202
            return {round(float(t), 2): np.zeros((num_samples, 2)) for t in t_eval}

    with pytest.raises(KeyError):
        sample_model_panels(_Mismatch(), [0.123456789], 1)


# ---------------------------------------------------------------------------
# repo_root / DEFAULT_SMOKE_EPOCHS — the shared anchors
# ---------------------------------------------------------------------------


def test_repo_root_points_at_the_repo() -> None:
    root = repo_root()
    # The editable/dev install the runners run under: pyproject + the package dir.
    assert (root / "pyproject.toml").is_file()
    assert (root / "stitching").is_dir()


def test_default_smoke_epochs_is_the_documented_default() -> None:
    assert DEFAULT_SMOKE_EPOCHS == 50


def test_run_field_order_is_stable() -> None:
    # Runners reach into a reloaded run by attribute (run.model, run.train_data)
    # AND legacy code unpacks it positionally; a silent reorder would break one.
    assert Run._fields == (
        "cfg",
        "model",
        "train_data",
        "test_data",
        "metrics",
        "manifest",
    )
