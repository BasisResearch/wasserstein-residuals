"""Unit tests for the config loader and composition.

The loader is the most-churned code in the repo (preset moves, composition
layers, field deletions), so its contracts are pinned here: preset
resolution, 2-/3-way composite merge order (last wins), experiment-name
stamping, strict unknown-key errors, and tuple coercion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stitching.config import (
    EXPERIMENT_CONFIG_DIR,
    Config,
    list_data_presets,
    list_experiment_presets,
    list_method_presets,
    load_config,
)

# ---------------------------------------------------------------------------
# Preset resolution & composition
# ---------------------------------------------------------------------------


def test_defaults_when_no_config() -> None:
    cfg = load_config(None)
    assert cfg.model == "stitching"
    assert cfg.config_path is None


def test_builtin_preset_lists() -> None:
    data, methods = list_data_presets(), list_method_presets()
    assert {"doublewell", "wavy_valley", "rna", "cis", "chiral"} <= set(data)
    assert set(methods) == {"stitching", "lightspeed"}
    assert not set(data) & set(methods)


def test_bare_data_preset() -> None:
    cfg = load_config("doublewell")
    assert cfg.potential == "doublewell"
    assert cfg.experiment_name == "doublewell"


def test_two_way_composite_merges_data_then_method() -> None:
    cfg = load_config("doublewell:stitching")
    assert cfg.potential == "doublewell"  # from the data preset
    assert cfg.model == "stitching"  # from the method preset
    assert cfg.config_path is not None and len(cfg.config_path.split(" + ")) == 2


def test_three_way_composite_experiment_wins() -> None:
    cfg = load_config("wavy_valley:stitching:wavy_valley_fig1_stitching")
    # method preset says epochs=1000; the experiment overlay must win.
    assert cfg.epochs == 10_000
    # data preset says synth_beta=0.05; the experiment overlay must win.
    assert cfg.synth_beta == 0.00625
    assert cfg.experiment_name == "wavy_valley_fig1_stitching"
    assert cfg.config_path is not None and len(cfg.config_path.split(" + ")) == 3


def test_shipped_composites_emit_no_misplaced_field_warning() -> None:
    # The model registry's ``only_fields`` now claims Stitching's structural
    # fields (num_particles/num_steps/lengthscale/time_conditioned_potential) in
    # addition to its loss knobs. The misplaced-field check fires when a field
    # owned by *another* model is set non-default for the active model, so a
    # lightspeed composite that set, say, num_particles would warn. Guard that
    # NO shipped composite trips it — the contract that lets the expanded
    # ownership be a real misuse-catcher rather than a false-positive generator.
    import warnings

    composites = [
        f"{data}:{method}"
        for data in list_data_presets()
        for method in list_method_presets()
    ]
    # The one 3-way lightspeed composite is the highest-risk case (a method that
    # does NOT own the structural fields, overlaid with an experiment preset).
    composites.append("wavy_valley:lightspeed:wavy_valley_fig1_lightspeed")

    for composite in composites:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_config(composite)
        misplaced = [w for w in caught if "misplaced override" in str(w.message)]
        assert not misplaced, (
            f"{composite} warned: {[str(w.message) for w in misplaced]}"
        )


def test_misplaced_structural_field_on_lightspeed_warns() -> None:
    # The complement of the guard above: setting a Stitching-only structural
    # field on a lightspeed run must now warn (it would be silently ignored).
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Config.from_dict(
            {"potential": "doublewell", "model": "lightspeed", "num_particles": 999}
        )
    assert any(
        "num_particles" in str(w.message) and "misplaced override" in str(w.message)
        for w in caught
    )


def test_trajectory_init_defaults_to_ot_and_round_trips() -> None:
    # The new Stitching-only structural field: default keeps the shipped "ot"
    # behaviour, and an explicit override survives from_dict.
    assert load_config(None).trajectory_init == "ot"
    cfg = Config.from_dict(
        {"potential": "doublewell", "model": "stitching", "trajectory_init": "mccann"}
    )
    assert cfg.trajectory_init == "mccann"


def test_trajectory_init_on_lightspeed_warns() -> None:
    # trajectory_init is Stitching-owned (in only_fields); setting it on a
    # lightspeed run must warn that it is a misplaced override, not silently
    # take effect (Lightspeed has no trajectory to seed).
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Config.from_dict(
            {
                "potential": "doublewell",
                "model": "lightspeed",
                "trajectory_init": "mccann",
            }
        )
    assert any(
        "trajectory_init" in str(w.message) and "misplaced override" in str(w.message)
        for w in caught
    )


def test_composite_with_too_many_parts_raises() -> None:
    with pytest.raises(ValueError, match="Composite config"):
        load_config("a:b:c:d")


def test_unknown_preset_raises_filenotfound() -> None:
    with pytest.raises(FileNotFoundError, match="no-such-preset"):
        load_config("no-such-preset:stitching")


def test_unknown_experiment_stem_raises_filenotfound() -> None:
    # The 3rd composite tier (the experiment overlay); a bad experiment stem
    # must fail loudly (and name the experiment tier), not silently merge nothing.
    with pytest.raises(FileNotFoundError, match="Experiment preset not found"):
        load_config("wavy_valley:stitching:no-such-experiment")


# The 3-way composites the paper runners pin (one per shipped experiment preset).
# Kept in sync with the CONFIG/PANEL_CONFIG constants in experiments/paper/*.py;
# loading each is the relocation guard for the 10 files moved into the package.
_SHIPPED_EXPERIMENT_COMPOSITES = [
    "chiral:stitching:chiral_fig5",
    "cis:stitching:cis_fig4",
    "doublewell:stitching:doublewell_panel",
    "rna:stitching:rna_fig3",
    "rna-gappy:stitching:rna_gappy_table5",
    "synthetic:stitching:synthetic_panel",
    "wavy_valley:stitching:wavy_valley_fig1_stitching",
    "wavy_valley:lightspeed:wavy_valley_fig1_lightspeed",
]


@pytest.mark.parametrize("composite", _SHIPPED_EXPERIMENT_COMPOSITES)
def test_every_shipped_experiment_preset_loads_as_composite(composite: str) -> None:
    # Each moved experiment JSON must still resolve+parse as the 3rd tier of its
    # runner composite (catches an orphaned/renamed/malformed preset after the
    # git mv, without running the paper runner). The experiment stem is the most
    # specific identity, so it must win as the experiment name.
    cfg = load_config(composite)
    assert cfg.config_path is not None
    assert len(cfg.config_path.split(" + ")) == 3
    assert cfg.experiment_name == composite.split(":")[2]


def test_shipped_composites_cover_every_experiment_preset() -> None:
    # The parametrized list above must exercise every shipped experiment preset,
    # so a newly added preset that no composite loads is caught here rather than
    # silently going untested.
    covered = {c.split(":")[2] for c in _SHIPPED_EXPERIMENT_COMPOSITES}
    assert covered == set(list_experiment_presets())


def test_config_from_json_file(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"potential": "doublewell", "epochs": 7}))
    cfg = load_config(path)
    assert (cfg.potential, cfg.epochs) == ("doublewell", 7)
    assert cfg.config_path == str(path)


def test_presets_live_in_top_level_configs_dir() -> None:
    # Presets live in the top-level ``configs/`` tree at the repo root (the
    # standard ML-repo layout: config is an experiment artifact, not library
    # code), resolved relative to the package or via $STITCHING_CONFIG_DIR — not
    # bundled in the wheel. The experiment tier is ``configs/experiment/``, and
    # list_experiment_presets() sees its stems so a 3-way composite resolves.
    assert EXPERIMENT_CONFIG_DIR.is_dir()
    assert EXPERIMENT_CONFIG_DIR.name == "experiment"
    assert EXPERIMENT_CONFIG_DIR.parent.name == "configs"
    stems = list_experiment_presets()
    assert stems == sorted(stems)
    assert {
        "wavy_valley_fig1_stitching",
        "wavy_valley_fig1_lightspeed",
        "doublewell_panel",
    } <= set(stems)


def test_config_dir_env_override(tmp_path: Path) -> None:
    # $STITCHING_CONFIG_DIR redirects the preset-tree root (test/CI isolation or
    # a non-editable install pointing at an unpacked configs/). _config_root()
    # reads the env each call, so a monkeypatched value wins over the default
    # repo-root-relative path.
    import os
    from unittest import mock

    from stitching.config.loader import _config_root

    default_root = _config_root()
    assert default_root.name == "configs"
    with mock.patch.dict(os.environ, {"STITCHING_CONFIG_DIR": str(tmp_path)}):
        assert _config_root() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# from_dict validation & coercion
# ---------------------------------------------------------------------------


def test_unknown_key_is_a_loud_error() -> None:
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_dict({"potential": "doublewell", "epochz": 1})


def test_removed_legacy_field_is_a_loud_error() -> None:
    # Config migrations were dropped (clone-only repo; results/ is regenerable).
    # A field the old _DROPPED_FIELDS silently swallowed -- chunk_size was present
    # in every legacy results/*/cfg.json -- is now an ordinary unknown key
    # and raises, like any other. Pins the deliberate "legacy cfg.json no longer
    # loads" decision so from_dict can't silently regain swallow-behaviour.
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_dict({"potential": "doublewell", "chunk_size": 0})


def test_underscore_keys_are_ignored() -> None:
    cfg = Config.from_dict({"potential": "doublewell", "_comment": "meta"})
    assert cfg.potential == "doublewell"


def test_dataset_and_potential_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="not both"):
        Config.from_dict({"dataset": "rna", "potential": "doublewell"})
    with pytest.raises(ValueError, match="must set either"):
        Config.from_dict({"epochs": 1})


def test_json_lists_coerce_to_tuples() -> None:
    cfg = Config.from_dict(
        {
            "potential": "doublewell",
            "potential_hidden": [32, 32],
            "synth_snapshot_times": [0.0, 5.0, 10.0],
            "synth_init_loc": [-3.0, 1.0],
        }
    )
    assert cfg.potential_hidden == (32, 32)
    assert cfg.synth_snapshot_times == (0.0, 5.0, 10.0)
    assert cfg.synth_init_loc == (-3.0, 1.0)
    # Scalar loc stays a scalar.
    assert (
        Config.from_dict({"potential": "x", "synth_init_loc": 2.0}).synth_init_loc
        == 2.0
    )


def test_misplaced_method_field_warns() -> None:
    with pytest.warns(UserWarning, match="stitching_coeff"):
        Config.from_dict(
            {"potential": "doublewell", "model": "lightspeed", "stitching_coeff": 0.5}
        )


def test_misplaced_method_field_warns_other_direction() -> None:
    # The registry-driven check must fire symmetrically: a lightspeed-only field
    # set under model=stitching is just as misplaced.
    with pytest.warns(UserWarning, match="lightspeed_coupling"):
        Config.from_dict(
            {
                "potential": "doublewell",
                "model": "stitching",
                "lightspeed_coupling": "natural",
            }
        )


def test_own_method_field_does_not_warn() -> None:
    # A field set for its OWN model is correct usage and must stay silent.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Config.from_dict(
            {"potential": "doublewell", "model": "stitching", "stitching_coeff": 2.0}
        )
