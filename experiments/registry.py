"""The experiment registry: ``EXPERIMENT_REGISTRY`` and the ``ExperimentSpec`` type.

Mirrors ``stitching.build.MODEL_REGISTRY`` and ``stitching.data.DATASET_REGISTRY``:
a frozen spec per experiment, looked up by name. Adding an experiment is a single
:func:`register` call from a ``experiments/defs/<name>.py`` module — no engine
edit, no second list to keep in sync.

An :class:`ExperimentSpec` is *declarative data* plus a few experiment-specific
callables (the ``evaluate`` and ``plots`` hooks): the genuinely per-experiment
metric/figure code lives in the def, while the uniform plumbing (CLI, output
dirs, idempotent training, checkpoint reload) lives once in the engine.

Kept import-light (no jax): importing the registry to *list* experiments must not
drag in the model/data stack. The hook callables import their heavy deps lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from stitching.utils.runners import DEFAULT_SMOKE_EPOCHS

if TYPE_CHECKING:
    from experiments._engine import Context


@dataclass(frozen=True)
class Variant:
    """One trainable cell of an experiment (a model/regime/landscape choice).

    The engine trains each ``(variant, seed)`` into its own checkpoint under
    ``results/<exp>/runs/<variant.name>/s<seed>/``. ``config`` is a composite
    preset string resolved by :func:`stitching.config.load_config`; ``overrides``
    are applied on top via :func:`dataclasses.replace` (the sweep axes that vary
    per cell, e.g. ``{"time_conditioned_potential": True}``).

    Attributes:
        name: Directory-safe label, unique within the experiment.
        config: Composite preset, e.g. ``"chiral:stitching:chiral_fig5"``.
        overrides: ``Config`` fields to override for this cell.
    """

    name: str
    config: str
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    """A whole experiment: its cells, its smoke trim, and its eval/plot hooks.

    Attributes:
        name: Registry key and ``results/<name>`` subdir.
        description: One-line summary (shown by ``python -m experiments list``).
        variants: The cells to sweep (crossed with ``seeds``).
        seeds: Random seeds to sweep per variant (default a single seed 0).
        smoke_epochs: Epoch budget under ``--smoke`` (wall-time only).
        smoke_overrides: Extra wall-time-only ``apply_smoke`` kwargs under
            ``--smoke`` (e.g. ``{"num_particles": 100}``).
        smoke_variants: If set, the subset of variant names trained under
            ``--smoke`` (e.g. a 2-of-15 landscape subset); ``None`` keeps all.
        evaluate: Optional hook ``(ctx) -> None`` for the ``eval`` stage: reloads
            checkpoints, computes metrics, writes the experiment's CSV(s). ``None``
            means the experiment has no metrics stage (wall-time-only runs).
        plots: Named plot hooks ``{name: (ctx) -> None}`` for the ``plot`` stage;
            each reloads checkpoints (and/or reads ``metrics.csv``) and renders a
            figure. May be empty.
    """

    name: str
    description: str
    variants: Sequence[Variant]
    seeds: Sequence[int] = (0,)
    smoke_epochs: int = DEFAULT_SMOKE_EPOCHS
    smoke_overrides: Mapping[str, Any] = field(default_factory=dict)
    smoke_variants: Sequence[str] | None = None
    evaluate: Callable[[Context], None] | None = None
    plots: Mapping[str, Callable[[Context], None]] = field(default_factory=dict)

    def variant(self, name: str) -> Variant:
        """Return the variant with this name, or raise a clear ``KeyError``."""
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(
            f"experiment {self.name!r} has no variant {name!r}; "
            f"choose from {[v.name for v in self.variants]}."
        )

    def selected_variants(self, *, smoke: bool) -> list[Variant]:
        """Variants to run, honouring the ``--smoke`` subset if one is declared."""
        if smoke and self.smoke_variants is not None:
            keep = set(self.smoke_variants)
            return [v for v in self.variants if v.name in keep]
        return list(self.variants)


EXPERIMENT_REGISTRY: dict[str, ExperimentSpec] = {}


def register(spec: ExperimentSpec) -> ExperimentSpec:
    """Register *spec* under its name; raise on a duplicate. Returns *spec*.

    Called once per ``experiments/defs/<name>.py`` at import time.
    """
    if spec.name in EXPERIMENT_REGISTRY:
        raise ValueError(f"experiment {spec.name!r} is already registered.")
    EXPERIMENT_REGISTRY[spec.name] = spec
    return spec


def get_experiment(name: str) -> ExperimentSpec:
    """Return the registered spec for *name*, or raise a listing ``KeyError``."""
    try:
        return EXPERIMENT_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown experiment {name!r}; available: {list_experiments()}."
        ) from None


def list_experiments() -> list[str]:
    """Return the sorted names of all registered experiments."""
    return sorted(EXPERIMENT_REGISTRY)
