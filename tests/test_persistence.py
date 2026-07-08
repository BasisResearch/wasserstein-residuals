"""Unit tests for run persistence and the ``--plot-only`` reload seam.

Pins the reload seam's contracts: :func:`load_runs` reloads a saved
run by role and fails loudly (naming the role) on a missing or partially-written
checkpoint, and ``Stitching.from_data`` rejects an unknown ``lengthscale`` string
instead of silently falling through to Silverman.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stitching.utils.persistence as persistence
from stitching._kde import SpatioTemporalData
from stitching.build import build_model
from stitching.config import Config
from stitching.models import Stitching
from stitching.utils import _provenance
from stitching.utils.persistence import (
    compute_fit_id,
    data_fingerprint,
    fit_id_for,
    load_run,
    load_runs,
    save_run,
)


def _tiny_data() -> SpatioTemporalData:
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(60, 2)), dtype=jnp.float32)
    t = jnp.asarray(np.repeat([0.0, 1.0, 2.0], 20), dtype=jnp.float32)
    return SpatioTemporalData(x=x, t=t)


def _tiny_cfg() -> Config:
    return Config(
        potential="doublewell",
        num_particles=20,
        num_steps=5,
        potential_hidden=(16,),
        seed=0,
    )


def _tiny_lightspeed_cfg() -> Config:
    return Config(
        potential="doublewell",
        model="lightspeed",
        num_particles=20,
        num_steps=5,
        potential_hidden=(16,),
        seed=0,
    )


def test_reloaded_lightspeed_can_sample(tmp_path: Path) -> None:
    """A reloaded Lightspeed model must be able to ``sample`` (the symptom the
    jax-array reload fix targets).

    ``Lightspeed.sample`` builds a KDE whose centers come from ``train_data`` and
    indexes them under ``jax.vmap``; if the reloaded ``train_data`` held numpy
    arrays this raised ``TracerArrayConversionError``. The decoupled ``eval``/
    ``plot`` stages reload before sampling, so this is the exact path that broke.
    """
    cfg, data = _tiny_lightspeed_cfg(), _tiny_data()
    model = build_model(cfg, data, jax.random.key(cfg.seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # dirty-tree provenance warning in tests
        save_run(tmp_path / "ls", cfg, model, data, data, metrics={})
        run = load_run(tmp_path / "ls")

    t_eval = jnp.asarray(
        sorted({float(t) for t in np.unique(np.asarray(run.train_data.t))})
    )
    samples = run.model.sample(t_eval, run.train_data)  # must not raise
    assert set(samples) == {0.0, 1.0, 2.0}
    for arr in samples.values():
        assert np.isfinite(np.asarray(arr)).all()


def test_load_runs_roundtrip(tmp_path: Path) -> None:
    cfg, data = _tiny_cfg(), _tiny_data()
    model = build_model(cfg, data, jax.random.key(cfg.seed))
    save_run(tmp_path / "a", cfg, model, data, data, metrics={"emd": 0.5})

    loaded = load_runs({"a": tmp_path / "a"})
    assert set(loaded) == {"a"}
    cfg2, model2, train2, test2, metrics2, manifest2 = loaded["a"]
    # Deserialised leaves match the saved model exactly (numerical identity).
    assert jnp.allclose(model.trajectories, model2.trajectories)
    assert metrics2["emd"] == 0.5
    assert cfg2.num_particles == cfg.num_particles
    # A reloaded run must be indistinguishable from a fresh load_data: jax arrays,
    # not numpy. Numpy centers break model sample paths that index a KDE under vmap
    # (e.g. Lightspeed), which the decoupled eval/plot stages hit on reload.
    assert isinstance(train2.x, jnp.ndarray) and isinstance(train2.t, jnp.ndarray)
    assert isinstance(test2.x, jnp.ndarray) and isinstance(test2.t, jnp.ndarray)
    # The manifest is now returned and carries the data hash + fit-id.
    assert manifest2.get("data_hash")
    assert manifest2.get("fit_id")


def test_load_runs_missing_run_names_role(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="'missing'"):
        load_runs({"missing": tmp_path / "nope"})


def test_load_runs_partial_write_names_role(tmp_path: Path) -> None:
    # save_run writes model.eqx first; an interrupted save leaves only that.
    run = tmp_path / "partial"
    run.mkdir()
    (run / "model.eqx").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="'partial'"):
        load_runs({"partial": run})


# ---------------------------------------------------------------------------
# Provenance / fit-id binding
# ---------------------------------------------------------------------------


def test_fit_id_binds_csv_to_checkpoint(tmp_path: Path) -> None:
    # The CSV-side fit-id a runner stamps must equal the manifest fit-id the
    # checkpoint saved from the SAME data — that equality is the binding.
    cfg, data = _tiny_cfg(), _tiny_data()
    model = build_model(cfg, data, jax.random.key(cfg.seed))
    save_run(tmp_path / "a", cfg, model, data, data)
    manifest = load_run(tmp_path / "a").manifest
    assert manifest["fit_id"] == fit_id_for(data, data)


def test_fit_id_differs_for_different_data(tmp_path: Path) -> None:
    # The stale-checkpoint-vs-fresh-CSV scenario: a CSV computed against other
    # data must NOT match the checkpoint's fit-id, so the desync is detectable.
    cfg, data = _tiny_cfg(), _tiny_data()
    model = build_model(cfg, data, jax.random.key(cfg.seed))
    save_run(tmp_path / "a", cfg, model, data, data)
    manifest = load_run(tmp_path / "a").manifest

    other = SpatioTemporalData(x=data.x + 1.0, t=data.t)
    assert fit_id_for(other, other) != manifest["fit_id"]


def test_compute_fit_id_changes_with_commit() -> None:
    # A stale checkpoint (saved at a different commit) yields a different id even
    # on identical data — the other half of the binding.
    assert compute_fit_id("abc", "commit-A") != compute_fit_id("abc", "commit-B")
    assert compute_fit_id("abc", "commit-A") == compute_fit_id("abc", "commit-A")


def test_save_load_hash_roundtrip_is_stable(tmp_path: Path) -> None:
    # The linchpin: the save-side hash (from jax arrays via np.asarray) must equal
    # the load-side hash (from data.npz numpy arrays). If it ever silently broke
    # (e.g. a dtype narrowing), EVERY strict --plot-only reload would raise. Pin a
    # clean reload: no drift warning, and the reloaded data reproduces the hash.
    cfg, data = _tiny_cfg(), _tiny_data()
    model = build_model(cfg, data, jax.random.key(cfg.seed))
    save_run(tmp_path / "a", cfg, model, data, data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = load_runs({"a": tmp_path / "a"})["a"]  # strict path must not raise
    # A clean reload raises no data-drift warning (a dirty-tree/commit warning may
    # fire and is unrelated); and the reloaded data reproduces the saved hash.
    assert not [w for w in caught if "does not match the manifest" in str(w.message)]
    assert run.manifest["data_hash"] == data_fingerprint(run.train_data, run.test_data)


def test_fit_id_for_handles_git_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # On an installed wheel / .git-less CI runner, current_commit() is None; the
    # fit-id must still compute (falling back to the "nocommit" sentinel). Patch
    # current_commit in _provenance, where fit_id_for resolves it (persistence
    # only re-exports the name).
    monkeypatch.setattr(_provenance, "current_commit", lambda: None)
    data = _tiny_data()
    fid = fit_id_for(data, data)
    assert len(fid) == 12
    assert fid == compute_fit_id(data_fingerprint(data, data), None)


def test_corrupt_manifest_fails_closed_under_strict(tmp_path: Path) -> None:
    # An absent manifest is the legitimate pre-provenance case ({}). A manifest
    # that EXISTS but won't parse is a damaged record: the publish path (strict)
    # must raise rather than silently skip the drift check; analysis warns.
    cfg, data = _tiny_cfg(), _tiny_data()
    model = build_model(cfg, data, jax.random.key(cfg.seed))
    run = tmp_path / "a"
    save_run(run, cfg, model, data, data)
    (run / "manifest.json").write_text("{ not valid json", encoding="utf-8")

    with pytest.warns(UserWarning, match="could not be read"):
        loaded = load_run(run)  # non-strict: advisory, degrades to {}
    assert loaded.manifest == {}
    with pytest.raises(ValueError, match="could not be read"):
        load_runs({"a": run})  # strict publish path fails closed


def test_data_drift_warns_then_raises_under_strict(tmp_path: Path) -> None:
    # Corrupting data.npz after save must not silently reload: load_run warns,
    # and the figure-publishing load_runs (strict=True) raises.
    cfg, data = _tiny_cfg(), _tiny_data()
    model = build_model(cfg, data, jax.random.key(cfg.seed))
    run = tmp_path / "a"
    save_run(run, cfg, model, data, data)

    # Swap in different data, keeping the original manifest hash.
    np.savez(
        run / "data.npz",
        train_x=np.asarray(data.x) + 1.0,
        train_t=np.asarray(data.t),
        test_x=np.asarray(data.x) + 1.0,
        test_t=np.asarray(data.t),
    )
    with pytest.warns(UserWarning, match="does not match the manifest"):
        load_run(run)  # non-strict: advisory
    with pytest.raises(ValueError, match="does not match the manifest"):
        load_runs({"a": run})  # strict publish path


def test_commit_drift_warns_via_persistence_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _warn_stale_provenance lives in persistence and resolves current_commit
    # from persistence's namespace (the name re-imported from _provenance), so a
    # checkpoint saved at a different commit than HEAD must warn. Pins both the
    # commit-drift warning AND that the re-import stays wired — a future trim of
    # the persistence re-export block would NameError here. NB: the patch target
    # is `persistence`, NOT `_provenance` (contrast test_fit_id_for_handles_git_absent,
    # where fit_id_for resolves the name in _provenance instead).
    cfg, data = _tiny_cfg(), _tiny_data()
    model = build_model(cfg, data, jax.random.key(cfg.seed))
    run = tmp_path / "a"
    save_run(run, cfg, model, data, data)
    # Rewrite the saved commit hermetically (robust on a git-less runner where
    # save_run would record git=None and the drift branch could never fire).
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["git"] = {"commit": "a" * 40, "dirty": False}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(persistence, "current_commit", lambda: "b" * 40)

    with pytest.warns(UserWarning, match="but HEAD is"):
        load_run(run)


def test_from_data_unknown_lengthscale_raises() -> None:
    data = _tiny_data()
    with pytest.raises(ValueError, match="Unknown lengthscale"):
        Stitching.from_data(
            data,
            num_particles=10,
            num_steps=4,
            hidden=(8,),
            key=jax.random.key(0),
            lengthscale="pullback",  # documented-then-removed; must not silently pass
        )


def test_from_data_accepts_int_lengthscale() -> None:
    # Archived/hand-edited configs may carry an int bandwidth; coerce, don't crash.
    data = _tiny_data()
    model = Stitching.from_data(
        data,
        num_particles=10,
        num_steps=4,
        hidden=(8,),
        key=jax.random.key(0),
        lengthscale=2,
    )
    assert bool(jnp.all(jnp.isfinite(model.bandwidth)))
