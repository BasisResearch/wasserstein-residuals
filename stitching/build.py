"""Cfg-driven construction layer.

Translates a `Config` into the trainable artifacts the experiment harness
needs:

- :data:`MODEL_REGISTRY` — the single table every model is registered through
  (class, from-cfg constructors, owned config fields)
- :func:`build_model` — `cfg.model` → model instance from ``stitching.models``
- :func:`build_loss`  — `cfg.model` → matching ``<Model>Loss`` instance
- :func:`make_loss_fn` — wraps the loss as ``loss_fn(model, batch, key)``
- :func:`fit` — the shared build-and-train recipe used by the experiment
  runners

The library doesn't know about cfg; this module is the translation layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax

from stitching._kde import SpatioTemporalData
from stitching.models import (
    Lightspeed,
    LightspeedLoss,
    Stitching,
    StitchingLoss,
)

from .config import Config

# ---------------------------------------------------------------------------
# Model registry — the single place a model is wired to cfg
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Everything the cfg→artifact layer needs to know about one model.

    Co-locates the model class, its from-cfg constructors, the config fields
    that model (and only it) consumes, and the capability flags that drive the
    evaluation battery. Registering a new model is a single
    :data:`MODEL_REGISTRY` entry; the construction dispatch, the misplaced-field
    check, and :mod:`stitching.evaluate`'s metric dispatch all read from here.

    Attributes:
        model_cls: The trainable model class; the anchor for
            :func:`spec_for_model`'s instance→spec reverse lookup.
        make_model: ``(cfg, train_data, key) -> model``.
        make_loss: ``(cfg, train_data) -> loss``.
        only_fields: Config fields consumed by this model alone — read back by
            :meth:`Config._warn_misplaced_method_fields` to flag a field set
            for the wrong model.
        needs_train_data_for_sample: ``Model.sample`` requires *train_data*
            positionally (the JKO chain seeds from observed t=0 particles)
            rather than the ``(num_samples, key)`` KDE-resampling signature.
        supports_one_step_ahead: the model exposes a functional + bandwidth a
            single JKO proximal step can be measured against (one-step-ahead).
        supports_bw2_uvp: the ground-truth-aware Bd²W₂-UVP metric applies (the
            model samples a particle cloud with explicit mixture weights).
        has_particle_weights: the model carries per-particle mixture weights
            (:meth:`Model.particle_weights` returns an array, not ``None``).
    """

    model_cls: type
    make_model: Callable[[Config, SpatioTemporalData, jax.Array], Any]
    make_loss: Callable[[Config, SpatioTemporalData | None], Any]
    only_fields: tuple[str, ...]
    needs_train_data_for_sample: bool
    supports_one_step_ahead: bool
    supports_bw2_uvp: bool
    has_particle_weights: bool

    def __post_init__(self) -> None:
        # Coherence invariant enforced at the registry boundary: the Bd²W₂-UVP
        # metric weights the predicted cloud by the model's particle weights, so
        # a spec cannot advertise the metric without the weights. Without this
        # guard, `evaluate.run_synthetic_evaluation` would feed
        # `np.asarray(model.particle_weights())` == `np.asarray(None)` (a 0-d
        # object array) into `bw2_uvp` — silently wrong numbers, not a crash.
        if self.supports_bw2_uvp and not self.has_particle_weights:
            raise ValueError(
                "ModelSpec inconsistent: supports_bw2_uvp=True requires "
                "has_particle_weights=True (the metric weights by particle mass)."
            )


def _make_stitching(
    cfg: Config, train_data: SpatioTemporalData, key: jax.Array
) -> Stitching:
    return Stitching.from_data(
        train_data,
        num_particles=cfg.num_particles,
        num_steps=cfg.num_steps,
        hidden=cfg.potential_hidden,
        key=key,
        lengthscale=cfg.lengthscale,
        trajectory_init=cfg.trajectory_init,
        entropy_coeff=cfg.entropy_coeff,
        potential_coeff=cfg.potential_coeff,
        interaction_coeff=cfg.interaction_coeff,
        interaction_type=cfg.interaction_type,
        interaction_hidden=cfg.interaction_hidden,
        time_conditioned=cfg.time_conditioned_potential,
    )


def _make_stitching_loss(
    cfg: Config, train_data: SpatioTemporalData | None = None
) -> StitchingLoss:
    return StitchingLoss(
        w_nll=cfg.w_nll,
        w_consistency=cfg.stitching_coeff,
        scheme=cfg.stitching_scheme,
        norm=cfg.stitching_norm,
        kinetic=cfg.stitching_kinetic,
        residual_alpha=float(cfg.stitching_residual_alpha),
        freeze_entropy=cfg.freeze_entropy,
        freeze_potential=cfg.freeze_potential,
        freeze_interaction=cfg.freeze_interaction,
    )


def _make_lightspeed(
    cfg: Config, train_data: SpatioTemporalData, key: jax.Array
) -> Lightspeed:
    return Lightspeed.from_data(
        train_data,
        hidden=cfg.potential_hidden,
        key=key,
        entropy_coeff=cfg.entropy_coeff,
        potential_coeff=cfg.potential_coeff,
    )


def _make_lightspeed_loss(
    cfg: Config, train_data: SpatioTemporalData | None = None
) -> LightspeedLoss:
    if train_data is None:
        # An explicit raise (not assert) so it survives `python -O`.
        raise ValueError("lightspeed loss requires train_data to build OT pairs")
    return LightspeedLoss.from_data(
        train_data,
        coupling=cfg.lightspeed_coupling,
        freeze_entropy=cfg.freeze_entropy,
        freeze_potential=cfg.freeze_potential,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "stitching": ModelSpec(
        model_cls=Stitching,
        make_model=_make_stitching,
        make_loss=_make_stitching_loss,
        only_fields=(
            # Loss/objective knobs unique to Stitching.
            "stitching_coeff",
            "stitching_scheme",
            "stitching_norm",
            "stitching_kinetic",
            "stitching_residual_alpha",
            # Trajectory-structural fields only Stitching consumes (Lightspeed
            # is potential-only with a fixed Silverman bandwidth).
            "num_particles",
            "num_steps",
            "lengthscale",
            "trajectory_init",
            "time_conditioned_potential",
        ),
        needs_train_data_for_sample=False,
        supports_one_step_ahead=True,
        supports_bw2_uvp=True,
        has_particle_weights=True,
    ),
    "lightspeed": ModelSpec(
        model_cls=Lightspeed,
        make_model=_make_lightspeed,
        make_loss=_make_lightspeed_loss,
        only_fields=("lightspeed_coupling",),
        needs_train_data_for_sample=True,
        supports_one_step_ahead=False,
        supports_bw2_uvp=False,
        has_particle_weights=False,
    ),
}


def _spec(cfg: Config) -> ModelSpec:
    try:
        return MODEL_REGISTRY[cfg.model]
    except KeyError:
        raise ValueError(
            f"Unknown cfg.model: {cfg.model!r}. Known: {sorted(MODEL_REGISTRY)}"
        ) from None


def spec_for_model(model: Stitching | Lightspeed) -> ModelSpec:
    """Reverse lookup: a model *instance* → its :class:`ModelSpec`.

    The instance-direction complement of :func:`build_model` (cfg → model). Lets
    the evaluation layer branch on capability flags (e.g. ``supports_bw2_uvp``)
    rather than ``isinstance`` ladders on the concrete model classes, so adding
    a model wires up its evaluation behaviour through the registry alone.

    Args:
        model: A trained model instance.

    Returns:
        The :class:`ModelSpec` whose ``model_cls`` the instance matches.

    Raises:
        TypeError: If *model* is not an instance of any registered model class.
    """
    for spec in MODEL_REGISTRY.values():
        if isinstance(model, spec.model_cls):
            return spec
    raise TypeError(
        f"No registered model spec for {type(model).__name__!r}; "
        f"known model classes: {sorted(MODEL_REGISTRY)}."
    )


def model_only_fields() -> dict[str, tuple[str, ...]]:
    """Map model name → the config fields only that model consumes.

    The authority behind :meth:`Config._warn_misplaced_method_fields`. Lives
    with the registry (not in the schema) so registering a model wires up its
    misplaced-field check in the same edit.
    """
    return {name: spec.only_fields for name, spec in MODEL_REGISTRY.items()}


# ---------------------------------------------------------------------------
# cfg → model / loss
# ---------------------------------------------------------------------------


def build_model(
    cfg: Config,
    train_data: SpatioTemporalData,
    key: jax.Array,
) -> Stitching | Lightspeed:
    """Construct the model selected by ``cfg.model`` (registry dispatch)."""
    return _spec(cfg).make_model(cfg, train_data, key)


def build_loss(
    cfg: Config, train_data: SpatioTemporalData | None = None
) -> StitchingLoss | LightspeedLoss:
    """Construct the loss instance selected by ``cfg.model`` (registry dispatch)."""
    return _spec(cfg).make_loss(cfg, train_data)


class LossFn:
    """Callable ``(model, batch, key) -> (loss, aux)`` wrapping a loss recipe.

    A typed stand-in for what used to be a closure with a monkey-patched
    ``.loss`` attribute: the underlying :class:`StitchingLoss` /
    :class:`LightspeedLoss` stays accessible at :attr:`loss` for inspection or
    serialisation alongside the trained model. Captured (not passed) by the
    optimiser step, so its fields are JIT constants — same treatment as the
    former closure. For ``LightspeedLoss`` this bakes its precomputed OT-pair
    arrays into the trace as constants; that is intended (the coupling is fixed
    at construction) and matches the prior closure's behaviour.
    """

    def __init__(self, loss: StitchingLoss | LightspeedLoss) -> None:
        self.loss = loss

    def __call__(
        self, model: Stitching | Lightspeed, batch: SpatioTemporalData, key: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return self.loss(model, batch, key)


def make_loss_fn(cfg: Config, train_data: SpatioTemporalData | None = None) -> LossFn:
    """Return a :class:`LossFn` — ``loss_fn(model, batch, key) -> (loss, aux)``.

    The underlying loss recipe is always accessible at ``loss_fn.loss`` for
    inspection / serialisation alongside the trained model.
    """
    return LossFn(build_loss(cfg, train_data))


# ---------------------------------------------------------------------------
# cfg → trained model
# ---------------------------------------------------------------------------


def fit(
    cfg: Config,
    train_data: SpatioTemporalData,
    *,
    epoch_callback: Callable[[Stitching | Lightspeed, int], None] | None = None,
    callback_every: int = 10,
    history: list[dict[str, float]] | None = None,
) -> Stitching | Lightspeed:
    """Build ``cfg.model`` and train it on *train_data* (the shared recipe).

    The single build-and-train recipe used by both the experiments and the
    ``python -m stitching.train`` CLI: model/train keys split from ``cfg.seed``,
    full-batch training (one step per epoch), and particle mixture weights
    frozen iff ``cfg.freeze_particle_weights``. ``cfg.batch_size`` is not
    consulted — every model trains full-batch here. For Stitching this is a
    cost choice (the expensive consistency loss is independent of data size);
    for the Lightspeed baseline it is simply the recipe the experiments have
    always used, kept identical so the CLI matches the figure path. This recipe
    derives all its randomness from ``cfg.seed`` via JAX keys; it touches no
    global (NumPy/Python) RNG, so the result is a pure function of ``cfg`` and
    *train_data*. The experiments call it through
    :func:`stitching.utils.runners.fit_seeded`, which additionally seeds the
    global RNGs from ``cfg.seed`` as defensive cover for any future global-RNG
    consumer.

    Args:
        cfg: Run configuration; consumes ``seed``, ``epochs``, ``lr`` and
            ``freeze_particle_weights`` on top of the :func:`build_model` /
            :func:`make_loss_fn` fields.
        train_data: Training snapshots.
        epoch_callback: Optional ``(model, epoch) -> None`` invoked every
            *callback_every* epochs and at the final epoch (e.g. the CLI's
            trajectory-evolution plot hook).
        callback_every: How often (in epochs) to invoke *epoch_callback*.
        history: If given, per-epoch ``{"epoch", "loss", **aux}`` dicts are
            appended in place for the caller to inspect/persist.

    Returns:
        The trained model instance.
    """
    from stitching.utils.optim import train

    k_model, k_train = jax.random.split(jax.random.key(cfg.seed))
    frozen = ("raw_weights",) if cfg.freeze_particle_weights else ()
    return train(
        params=build_model(cfg, train_data, k_model),
        loss_fn=make_loss_fn(cfg, train_data),
        data=train_data,
        epochs=cfg.epochs,
        batch_size=train_data.x.shape[0],
        lr=cfg.lr,
        key=k_train,
        epoch_callback=epoch_callback,
        callback_every=callback_every,
        frozen_names=frozen,
        history=history,
    )
