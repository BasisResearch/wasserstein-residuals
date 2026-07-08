"""Unit tests for the cfg→artifact build layer (the model registry).

Pins the model registry's contracts: construction dispatches through
:data:`MODEL_REGISTRY`, an unknown ``cfg.model`` fails loudly and names the
known models, the lightspeed loss refuses to build without data, and the
registry's ``only_fields`` stay disjoint (the invariant the misplaced-field
check in :mod:`stitching.config` relies on).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stitching._kde import SpatioTemporalData
from stitching.build import (
    MODEL_REGISTRY,
    LossFn,
    ModelSpec,
    build_loss,
    build_model,
    fit,
    make_loss_fn,
    model_only_fields,
    spec_for_model,
)
from stitching.config import Config
from stitching.evaluate import run_blackbox_evaluation, run_synthetic_evaluation
from stitching.models import LightspeedLoss, Stitching, StitchingLoss


def _tiny_data() -> SpatioTemporalData:
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(40, 2)), dtype=jnp.float32)
    t = jnp.asarray(np.repeat([0.0, 1.0], 20), dtype=jnp.float32)
    return SpatioTemporalData(x=x, t=t)


def test_registry_covers_exactly_the_known_models() -> None:
    assert set(MODEL_REGISTRY) == {"stitching", "lightspeed"}
    # Every spec is fully populated (no half-registered model).
    for spec in MODEL_REGISTRY.values():
        assert spec.model_cls is not None
        assert callable(spec.make_model)
        assert callable(spec.make_loss)
        assert isinstance(spec.only_fields, tuple)


def test_build_dispatches_to_registered_class() -> None:
    data = _tiny_data()
    for name, spec in MODEL_REGISTRY.items():
        cfg = Config(potential="doublewell", model=name, num_particles=10, num_steps=4)
        model = build_model(cfg, data, jax.random.key(0))
        assert isinstance(model, spec.model_cls)


def test_build_model_unknown_model_raises_naming_known() -> None:
    cfg = Config(potential="doublewell", model="mystery")
    with pytest.raises(
        ValueError,
        match=r"Unknown cfg.model: 'mystery'\. Known: \['lightspeed', 'stitching'\]",
    ):
        build_model(cfg, _tiny_data(), jax.random.key(0))
    # build_loss takes the same dispatch path.
    with pytest.raises(ValueError, match="Unknown cfg.model"):
        build_loss(cfg)


def test_build_loss_lightspeed_without_data_raises() -> None:
    # The lightspeed loss needs snapshots to form OT pairs; omitting them must
    # raise an explicit error (not a bare assert that vanishes under `python -O`).
    cfg = Config(potential="doublewell", model="lightspeed")
    with pytest.raises(ValueError, match="train_data"):
        build_loss(cfg)  # train_data defaults to None


def test_make_loss_fn_returns_lossfn_wrapping_the_recipe() -> None:
    # The typed LossFn replaces the old monkey-patched closure: it must expose
    # the underlying loss at `.loss` and forward calls to it as (loss, aux).
    data = _tiny_data()
    expected_cls = {"stitching": StitchingLoss, "lightspeed": LightspeedLoss}
    for name, cls in expected_cls.items():
        cfg = Config(potential="doublewell", model=name)
        loss_fn = make_loss_fn(cfg, data)
        assert isinstance(loss_fn, LossFn)
        assert isinstance(loss_fn.loss, cls)
        out = loss_fn(
            build_model(cfg, data, jax.random.key(0)), data, jax.random.key(1)
        )
        loss, aux = out
        assert jnp.ndim(loss) == 0  # scalar
        assert isinstance(aux, dict)


def test_lightspeed_fit_through_lossfn_is_deterministic() -> None:
    # The parity gate trains only Stitching, so the LossFn change (closure ->
    # class) has no automated bitwise coverage on the Lightspeed path — where it
    # matters most, since LightspeedLoss carries array leaves captured as JIT
    # constants. Two fits with the same cfg/seed must be bitwise-identical,
    # exercising make_loss_fn -> LossFn -> train's filter_jit step end to end.
    data = _tiny_data()
    cfg = Config(potential="doublewell", model="lightspeed", epochs=3, lr=1e-3)
    m1 = fit(cfg, data)
    m2 = fit(cfg, data)
    leaves1 = [x for x in jax.tree_util.tree_leaves(m1) if eqx.is_array(x)]
    leaves2 = [x for x in jax.tree_util.tree_leaves(m2) if eqx.is_array(x)]
    assert leaves1 and len(leaves1) == len(leaves2)
    for a, b in zip(leaves1, leaves2, strict=True):
        assert jnp.array_equal(a, b)


def test_only_fields_are_disjoint_across_models() -> None:
    # The misplaced-field check assumes a field belongs to at most one model; if
    # two models ever shared one, the warning logic would mis-attribute it.
    owners = model_only_fields()
    for name, fields in owners.items():
        others: set[str] = set().union(
            *(set(owners[k]) for k in owners if k != name), set()
        )
        assert set(fields).isdisjoint(others), (
            f"{name} shares a field with another model"
        )


# ---------------------------------------------------------------------------
# Reverse lookup + capability flags (the evaluate.py isinstance-ladder seam)
# ---------------------------------------------------------------------------


def test_spec_for_model_reverse_lookup_matches_build() -> None:
    # spec_for_model(instance) is the complement of build_model(cfg): a model
    # built from a registry name must resolve back to that same spec.
    data = _tiny_data()
    for name, spec in MODEL_REGISTRY.items():
        cfg = Config(potential="doublewell", model=name, num_particles=10, num_steps=4)
        model = build_model(cfg, data, jax.random.key(0))
        assert spec_for_model(model) is spec


def test_spec_for_model_rejects_unregistered_type() -> None:
    with pytest.raises(TypeError, match="No registered model spec"):
        spec_for_model(object())


def test_capability_flags_are_complete_booleans() -> None:
    # Every spec must declare all four flags as bools — a half-specified model
    # would make the evaluate.py dispatch silently skip or mis-route a metric.
    flags = (
        "needs_train_data_for_sample",
        "supports_one_step_ahead",
        "supports_bw2_uvp",
        "has_particle_weights",
    )
    for spec in MODEL_REGISTRY.values():
        for flag in flags:
            assert isinstance(getattr(spec, flag), bool)


def test_capability_flags_match_model_api() -> None:
    # The flags must agree with the models' actual accessors, since evaluate.py
    # now trusts the flag instead of isinstance: has_particle_weights iff
    # particle_weights() returns an array; both models expose learned_potential_fn.
    data = _tiny_data()
    for name, spec in MODEL_REGISTRY.items():
        cfg = Config(potential="doublewell", model=name, num_particles=10, num_steps=4)
        model = build_model(cfg, data, jax.random.key(0))

        weights = model.particle_weights()
        if spec.has_particle_weights:
            assert weights is not None
            assert np.asarray(weights).shape == (model.trajectories.shape[1],)
        else:
            assert weights is None

        # learned_potential_fn is part of the uniform model API on both models.
        pot = model.learned_potential_fn()
        val = pot(jnp.zeros((data.x.shape[1],), dtype=jnp.float32))
        assert jnp.ndim(val) == 0  # scalar potential at a point


def test_registry_model_classes_are_mutually_non_subclassing() -> None:
    # spec_for_model resolves by first isinstance match over the registry. That
    # is only unambiguous if no registered model_cls subclasses another — else a
    # subclass instance would silently resolve to its parent's spec (wrong flags,
    # wrong eval branch) with the suite green. Pin the precondition.
    classes = [spec.model_cls for spec in MODEL_REGISTRY.values()]
    for a in classes:
        for b in classes:
            assert a is b or not issubclass(a, b), (
                f"{a.__name__} subclasses {b.__name__}; spec_for_model would "
                "mis-resolve it"
            )


def test_modelspec_rejects_bw2_without_particle_weights() -> None:
    # The coherence invariant: a spec that claims supports_bw2_uvp must also have
    # particle weights, else evaluate would feed np.asarray(None) into bw2_uvp.
    with pytest.raises(ValueError, match="supports_bw2_uvp=True requires"):
        ModelSpec(
            model_cls=Stitching,
            make_model=MODEL_REGISTRY["stitching"].make_model,
            make_loss=MODEL_REGISTRY["stitching"].make_loss,
            only_fields=(),
            needs_train_data_for_sample=False,
            supports_one_step_ahead=False,
            supports_bw2_uvp=True,  # inconsistent ...
            has_particle_weights=False,  # ... with this
        )


def test_lightspeed_eval_path_skips_weighted_and_one_step_metrics() -> None:
    # The parity gate trains only Stitching, so Lightspeed's eval branch (the new
    # capability-flag dispatch) is otherwise end-to-end-untested. Drive both
    # orchestrators on a tiny Lightspeed fit and pin the flag-gated behaviour:
    # the train-data sample route works, one-step-ahead is skipped
    # (supports_one_step_ahead=False), and Bd²W₂-UVP is absent
    # (supports_bw2_uvp=False) — i.e. no np.asarray(None) crash and no silently
    # mislabelled weighted metric.
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(60, 2)), dtype=jnp.float32)
    t = jnp.asarray(np.repeat([0.0, 1.0, 2.0], 20), dtype=jnp.float32)
    train = SpatioTemporalData(x=x, t=t)
    test = SpatioTemporalData(x=x + 0.1, t=t)
    cfg = Config(
        potential="flowers",
        model="lightspeed",
        potential_hidden=(16,),
        epochs=2,
        eval_reps=1,
        seed=0,
    )
    model = fit(cfg, train)

    emd, _by_time, _full, one_step = run_blackbox_evaluation(model, cfg, train, test)
    assert np.isfinite(emd)
    assert one_step == {}  # supports_one_step_ahead=False

    _, _, func, _, _ = run_synthetic_evaluation(model, cfg, train, test)
    assert "bw2_uvp" not in func  # supports_bw2_uvp=False -> block skipped
